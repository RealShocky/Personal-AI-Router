// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"nvpair-shared/fabricwire"
)

type WorkerRecord struct {
	Heartbeat fabricwire.Heartbeat
	State     fabricwire.WorkerState
	LastSeen  time.Time
}

type FabricCapacity struct {
	Workers      uint32 `json:"workers"`
	MemoryFree   uint64 `json:"memoryFreeBytes"`
	GPUVramTotal uint64 `json:"gpuVramTotalBytes"`
	GPUVramFree  uint64 `json:"gpuVramFreeBytes"`
	GPUCount     uint32 `json:"gpuCount"`
}

type Manager struct {
	mu      sync.RWMutex
	timeout time.Duration
	workers map[string]WorkerRecord
}

func NewManager(timeout time.Duration) *Manager {
	return &Manager{timeout: timeout, workers: make(map[string]WorkerRecord)}
}

func (m *Manager) Capacity() FabricCapacity {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var capacity FabricCapacity
	for _, worker := range m.workers {
		if worker.State != fabricwire.WorkerReady {
			continue
		}
		capacity.Workers++
		capacity.MemoryFree += worker.Heartbeat.MemoryFree
		capacity.GPUVramTotal += worker.Heartbeat.GPUVramTotal
		capacity.GPUVramFree += worker.Heartbeat.GPUVramFree
		capacity.GPUCount += worker.Heartbeat.GPUCount
	}
	return capacity
}

func (m *Manager) AcceptHeartbeat(heartbeat fabricwire.Heartbeat, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	current, exists := m.workers[heartbeat.WorkerID]
	if !exists || current.State == fabricwire.WorkerQuarantined {
		m.workers[heartbeat.WorkerID] = WorkerRecord{
			Heartbeat: heartbeat,
			State:     heartbeat.State,
			LastSeen:  now,
		}
		return
	}
	current.Heartbeat = heartbeat
	current.State = heartbeat.State
	current.LastSeen = now
	m.workers[heartbeat.WorkerID] = current
}

func (m *Manager) Heartbeat(heartbeat fabricwire.Heartbeat, now time.Time) error {
	if heartbeat.WorkerID == "" || heartbeat.NodeID == "" || heartbeat.Epoch == 0 {
		return fmt.Errorf("heartbeat requires workerId, nodeId, and positive epoch")
	}
	m.mu.RLock()
	current, exists := m.workers[heartbeat.WorkerID]
	m.mu.RUnlock()
	if exists && current.State == fabricwire.WorkerQuarantined {
		return fmt.Errorf("worker is quarantined; fresh epoch rejoin required")
	}
	m.AcceptHeartbeat(heartbeat, now)
	return nil
}

func (m *Manager) Reconcile(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for id, worker := range m.workers {
		if now.Sub(worker.LastSeen) <= m.timeout {
			continue
		}
		if now.Sub(worker.LastSeen) > 2*m.timeout {
			worker.State = fabricwire.WorkerQuarantined
		} else if worker.State != fabricwire.WorkerSuspect {
			worker.State = fabricwire.WorkerSuspect
		}
		m.workers[id] = worker
	}
}

func (m *Manager) Rejoin(workerID string, epoch uint64, now time.Time) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	worker, ok := m.workers[workerID]
	if !ok || worker.State != fabricwire.WorkerQuarantined || epoch <= worker.Heartbeat.Epoch {
		return false
	}
	worker.Heartbeat.Epoch = epoch
	worker.State = fabricwire.WorkerReady
	worker.LastSeen = now
	m.workers[workerID] = worker
	return true
}

func (m *Manager) Worker(workerID string) (WorkerRecord, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	worker, ok := m.workers[workerID]
	return worker, ok
}

func (m *Manager) Workers() []WorkerRecord {
	m.mu.RLock()
	defer m.mu.RUnlock()
	workers := make([]WorkerRecord, 0, len(m.workers))
	for _, worker := range m.workers {
		workers = append(workers, worker)
	}
	return workers
}

func (m *Manager) Assignable(workerID string) bool {
	worker, ok := m.Worker(workerID)
	return ok && worker.State == fabricwire.WorkerReady
}

// ValidateTrainingPlacement checks the coordinator's live worker records
// before a torchrun world is launched. Training ranks are supplied by the
// caller, but they must still refer to ready workers with the requested
// backend and advertised endpoint. An empty model inventory means the worker
// has not implemented model inventory reporting; a non-empty inventory must
// contain the requested training artifact.
func (m *Manager) ValidateTrainingPlacement(request TrainingRequest) error {
	if err := request.Validate(); err != nil {
		return err
	}
	seen := make(map[string]bool, len(request.Nodes))
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, node := range request.Nodes {
		if seen[node.WorkerID] {
			return fmt.Errorf("training placement assigns worker %q more than once", node.WorkerID)
		}
		seen[node.WorkerID] = true
		worker, ok := m.workers[node.WorkerID]
		if !ok || worker.State != fabricwire.WorkerReady {
			return fmt.Errorf("training worker %q is not ready", node.WorkerID)
		}
		if !hasBackend(worker.Heartbeat.Backends, map[string]bool{node.Backend: true}) {
			return fmt.Errorf("training worker %q does not advertise backend %q", node.WorkerID, node.Backend)
		}
		if worker.Heartbeat.Endpoint != "" && node.Address != worker.Heartbeat.Endpoint {
			return fmt.Errorf("training worker %q endpoint changed", node.WorkerID)
		}
		if node.Backend == "cuda" && node.GPUCount > worker.Heartbeat.GPUCount {
			return fmt.Errorf("training worker %q reports %d GPUs, requested %d", node.WorkerID, worker.Heartbeat.GPUCount, node.GPUCount)
		}
		if len(worker.Heartbeat.ModelDigests) > 0 && !hasModelDigest(worker.Heartbeat.ModelDigests, request.ModelDigest) {
			return fmt.Errorf("training worker %q does not advertise model digest", node.WorkerID)
		}
	}
	return nil
}

func (m *Manager) PlanGroup(request fabricwire.GroupRequest) (fabricwire.GroupPlan, error) {
	if request.GroupID == "" || request.Runtime == "" || request.WorkerGoal == 0 {
		return fabricwire.GroupPlan{}, fmt.Errorf("group requires groupId, runtime, and positive workerGoal")
	}
	wantBackend := make(map[string]bool, len(request.Backends))
	for _, backend := range request.Backends {
		wantBackend[backend] = true
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	plan := fabricwire.GroupPlan{
		GroupID:   request.GroupID,
		Runtime:   request.Runtime,
		Epoch:     uint64(time.Now().UnixNano()),
		Endpoints: make(map[string]string),
		PeerIDs:   make(map[string]string),
	}
	type candidate struct {
		id     string
		worker WorkerRecord
	}
	eligible := make([]candidate, 0, len(m.workers))
	for id, worker := range m.workers {
		if worker.State != fabricwire.WorkerReady || !runtimeMatches(request.Runtime, worker.Heartbeat) {
			continue
		}
		if !request.AllowMixed && len(request.Backends) > 0 && len(plan.Workers) > 0 && (len(worker.Heartbeat.Backends) == 0 || worker.Heartbeat.Backends[0] != request.Backends[0]) {
			continue
		}
		if len(wantBackend) > 0 && !hasBackend(worker.Heartbeat.Backends, wantBackend) {
			continue
		}
		if request.ModelDigest != "" && !hasModelDigest(worker.Heartbeat.ModelDigests, request.ModelDigest) {
			continue
		}
		if request.RequireCheckpoint && !worker.Heartbeat.CheckpointSupport {
			continue
		}
		if request.MinMemoryFreeBytes > 0 && worker.Heartbeat.MemoryFree < request.MinMemoryFreeBytes {
			continue
		}
		if request.MinGPUVramTotalBytes > 0 && worker.Heartbeat.GPUVramTotal < request.MinGPUVramTotalBytes {
			continue
		}
		if request.MinGPUVramFreeBytes > 0 && worker.Heartbeat.GPUVramFree < request.MinGPUVramFreeBytes {
			continue
		}
		if request.MaxProbeLatencyMillis > 0 && (worker.Heartbeat.ProbeLatencyMillis == 0 || worker.Heartbeat.ProbeLatencyMillis > request.MaxProbeLatencyMillis) {
			continue
		}
		eligible = append(eligible, candidate{id: id, worker: worker})
	}
	if uint32(len(eligible)) < request.WorkerGoal {
		return fabricwire.GroupPlan{}, fmt.Errorf("only %d eligible workers available, need %d", len(eligible), request.WorkerGoal)
	}
	sort.Slice(eligible, func(i, j int) bool {
		left, right := eligible[i].worker.Heartbeat, eligible[j].worker.Heartbeat
		if request.MinAggregateGPUVramFreeBytes > 0 || request.MinAggregateGPUVramTotalBytes > 0 {
			if left.GPUVramFree != right.GPUVramFree {
				return left.GPUVramFree > right.GPUVramFree
			}
			if left.GPUVramTotal != right.GPUVramTotal {
				return left.GPUVramTotal > right.GPUVramTotal
			}
		}
		if request.MinAggregateMemoryFreeBytes > 0 && left.MemoryFree != right.MemoryFree {
			return left.MemoryFree > right.MemoryFree
		}
		return eligible[i].id < eligible[j].id
	})
	selected := eligible[:request.WorkerGoal]
	var aggregateMemory, aggregateGPUVramTotal, aggregateGPUVramFree uint64
	for _, item := range selected {
		id, worker := item.id, item.worker
		aggregateMemory += worker.Heartbeat.MemoryFree
		aggregateGPUVramTotal += worker.Heartbeat.GPUVramTotal
		aggregateGPUVramFree += worker.Heartbeat.GPUVramFree
		plan.Workers = append(plan.Workers, id)
		if worker.Heartbeat.Endpoint != "" {
			plan.Endpoints[id] = worker.Heartbeat.Endpoint
		}
		peerID := worker.Heartbeat.PeerID
		if peerID == "" {
			peerID = worker.Heartbeat.NodeID
		}
		if peerID != "" {
			plan.PeerIDs[id] = peerID
		}
		plan.MemoryFreeBytes = aggregateMemory
		plan.GPUVramTotalBytes = aggregateGPUVramTotal
		plan.GPUVramFreeBytes = aggregateGPUVramFree
		plan.GPUCount += worker.Heartbeat.GPUCount
	}
	if aggregateMemory < request.MinAggregateMemoryFreeBytes || aggregateGPUVramTotal < request.MinAggregateGPUVramTotalBytes || aggregateGPUVramFree < request.MinAggregateGPUVramFreeBytes {
		return fabricwire.GroupPlan{}, fmt.Errorf("aggregate capacity shortfall: aggregate memory free %d/%d bytes, aggregate GPU VRAM total %d/%d bytes, aggregate GPU VRAM free %d/%d bytes", aggregateMemory, request.MinAggregateMemoryFreeBytes, aggregateGPUVramTotal, request.MinAggregateGPUVramTotalBytes, aggregateGPUVramFree, request.MinAggregateGPUVramFreeBytes)
	}
	return plan, nil
}

func (m *Manager) BuildExecutionPlan(request fabricwire.GroupRequest, group fabricwire.GroupPlan, strategy fabricwire.ShardStrategy, transport fabricwire.Transport) (fabricwire.ExecutionPlan, error) {
	plan := fabricwire.ExecutionPlan{
		Version:       1,
		PlanID:        group.GroupID,
		Epoch:         group.Epoch,
		Runtime:       group.Runtime,
		ModelDigest:   request.ModelDigest,
		ShardStrategy: strategy,
		Transport:     transport,
		Checkpoint: fabricwire.CheckpointPolicy{
			Enabled:       request.RequireCheckpoint,
			IntervalSteps: 1,
		},
		Failover: fabricwire.FailoverPolicy{Enabled: true, MaxAttempts: 3},
		Workers:  make([]fabricwire.WorkerAssignment, 0, len(group.Workers)),
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	for index, workerID := range group.Workers {
		worker, ok := m.workers[workerID]
		if !ok || worker.State != fabricwire.WorkerReady {
			return fabricwire.ExecutionPlan{}, fmt.Errorf("worker %q is not ready for execution plan", workerID)
		}
		plan.Workers = append(plan.Workers, fabricwire.WorkerAssignment{
			WorkerID:           workerID,
			PeerID:             workerPeerID(worker.Heartbeat),
			Endpoint:           group.Endpoints[workerID],
			ShardIndex:         uint32(index),
			MemoryBudgetBytes:  worker.Heartbeat.MemoryFree,
			GPUVramBudgetBytes: worker.Heartbeat.GPUVramFree,
		})
	}
	if err := plan.Validate(); err != nil {
		return fabricwire.ExecutionPlan{}, err
	}
	return plan, nil
}

func workerPeerID(heartbeat fabricwire.Heartbeat) string {
	if heartbeat.PeerID != "" {
		return heartbeat.PeerID
	}
	return heartbeat.NodeID
}

// runtimeMatches keeps the engine runtime (for example, llama.cpp) separate
// from the worker execution runtime (for example, cuda). Older workers may
// report llama.cpp directly, while capability-aware workers report the
// backend they can execute. The backend filters below still decide whether a
// particular CPU, CUDA, or Metal worker is eligible.
func runtimeMatches(requestRuntime string, heartbeat fabricwire.Heartbeat) bool {
	if heartbeat.Runtime == requestRuntime {
		return true
	}
	if requestRuntime != "llama.cpp" {
		return false
	}
	for _, backend := range heartbeat.Backends {
		if backend == "cpu" || backend == "cuda" || backend == "metal" {
			return true
		}
	}
	return false
}

func hasModelDigest(digests []string, want string) bool {
	for _, digest := range digests {
		if digest == want {
			return true
		}
	}
	return false
}

func hasBackend(backends []string, wanted map[string]bool) bool {
	for _, backend := range backends {
		if wanted[backend] {
			return true
		}
	}
	return false
}
