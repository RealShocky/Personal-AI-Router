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

type Manager struct {
	mu      sync.RWMutex
	timeout time.Duration
	workers map[string]WorkerRecord
}

func NewManager(timeout time.Duration) *Manager {
	return &Manager{timeout: timeout, workers: make(map[string]WorkerRecord)}
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
	}
	for id, worker := range m.workers {
		if worker.State != fabricwire.WorkerReady || worker.Heartbeat.Runtime != request.Runtime {
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
		if request.MaxProbeLatencyMillis > 0 && (worker.Heartbeat.ProbeLatencyMillis == 0 || worker.Heartbeat.ProbeLatencyMillis > request.MaxProbeLatencyMillis) {
			continue
		}
		plan.Workers = append(plan.Workers, id)
		if worker.Heartbeat.Endpoint != "" {
			plan.Endpoints[id] = worker.Heartbeat.Endpoint
		}
		if uint32(len(plan.Workers)) == request.WorkerGoal {
			return plan, nil
		}
	}
	return fabricwire.GroupPlan{}, fmt.Errorf("only %d eligible workers available, need %d", len(plan.Workers), request.WorkerGoal)
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
