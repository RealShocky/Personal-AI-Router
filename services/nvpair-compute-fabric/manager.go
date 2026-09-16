// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
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
		plan.Workers = append(plan.Workers, id)
		if worker.Heartbeat.Endpoint != "" {
			plan.Endpoints[id] = worker.Heartbeat.Endpoint
		}
		if worker.Heartbeat.NodeID != "" {
			plan.PeerIDs[id] = worker.Heartbeat.NodeID
		}
		if uint32(len(plan.Workers)) == request.WorkerGoal {
			return plan, nil
		}
	}
	return fabricwire.GroupPlan{}, fmt.Errorf("only %d eligible workers available, need %d", len(plan.Workers), request.WorkerGoal)
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
			PeerID:             worker.Heartbeat.NodeID,
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
