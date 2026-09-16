// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestManagerQuarantinesSilentWorker(t *testing.T) {
	now := time.Unix(100, 0)
	m := NewManager(5 * time.Second)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1}, now)

	m.Reconcile(now.Add(6 * time.Second))
	worker, ok := m.Worker("w1")
	if !ok || worker.State != fabricwire.WorkerSuspect {
		t.Fatalf("worker after first timeout = %+v, want suspect", worker)
	}
	m.Reconcile(now.Add(12 * time.Second))
	worker, ok = m.Worker("w1")
	if !ok || worker.State != fabricwire.WorkerQuarantined {
		t.Fatalf("worker after second timeout = %+v, want quarantined", worker)
	}
	if m.Assignable("w1") {
		t.Fatal("quarantined worker is assignable")
	}
}

func TestManagerRequiresFreshEpochToRejoin(t *testing.T) {
	now := time.Unix(100, 0)
	m := NewManager(5 * time.Second)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 2}, now)
	m.Reconcile(now.Add(12 * time.Second))

	if m.Rejoin("w1", 2, now.Add(13*time.Second)) {
		t.Fatal("stale epoch rejoined")
	}
	if !m.Rejoin("w1", 3, now.Add(13*time.Second)) {
		t.Fatal("fresh epoch did not rejoin")
	}
	if !m.Assignable("w1") {
		t.Fatal("freshly rejoined worker is not assignable")
	}
}

func TestManagerDoesNotAssignSuspectWorker(t *testing.T) {
	now := time.Unix(100, 0)
	m := NewManager(5 * time.Second)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerServing, Epoch: 1}, now)
	m.Reconcile(now.Add(6 * time.Second))
	if m.Assignable("w1") {
		t.Fatal("suspect worker is assignable")
	}
}

func TestManagerRejectsHeartbeatFromQuarantinedWorkerUntilRejoin(t *testing.T) {
	now := time.Unix(100, 0)
	m := NewManager(5 * time.Second)
	heartbeat := fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1}
	if err := m.Heartbeat(heartbeat, now); err != nil {
		t.Fatalf("initial heartbeat: %v", err)
	}
	m.Reconcile(now.Add(12 * time.Second))
	if err := m.Heartbeat(heartbeat, now.Add(13*time.Second)); err == nil {
		t.Fatal("quarantined worker heartbeat should require a fresh rejoin epoch")
	}
	if !m.Rejoin("w1", 2, now.Add(13*time.Second)) {
		t.Fatal("fresh epoch rejoin failed")
	}
	heartbeat.Epoch = 2
	if err := m.Heartbeat(heartbeat, now.Add(14*time.Second)); err != nil {
		t.Fatalf("heartbeat after rejoin: %v", err)
	}
}

func TestManagerPlansOnlyCompatibleReadyWorkers(t *testing.T) {
	m := NewManager(time.Second)
	now := time.Unix(100, 0)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cuda-1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cpu-1", NodeID: "n2", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cpu"}}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "other", NodeID: "n3", State: fabricwire.WorkerServing, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}}, now)
	plan, err := m.PlanGroup(fabricwire.GroupRequest{GroupID: "g1", Runtime: "llama.cpp", Backends: []string{"cuda"}, WorkerGoal: 1})
	if err != nil || len(plan.Workers) != 1 || plan.Workers[0] != "cuda-1" {
		t.Fatalf("plan = %+v, err = %v", plan, err)
	}
	if _, err := m.PlanGroup(fabricwire.GroupRequest{GroupID: "g2", Runtime: "llama.cpp", Backends: []string{"metal"}, WorkerGoal: 1}); err == nil {
		t.Fatal("incompatible backend unexpectedly planned")
	}
}

func TestManagerMatchesLlamaEngineToWorkerBackendRuntime(t *testing.T) {
	m := NewManager(time.Second)
	now := time.Unix(100, 0)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cuda-1", NodeID: "n1", Endpoint: "https://cuda-1/v1/fabric/rpc", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "cuda", Backends: []string{"cuda"}}, now)

	plan, err := m.PlanGroup(fabricwire.GroupRequest{GroupID: "llama-cuda", Runtime: "llama.cpp", Backends: []string{"cuda"}, WorkerGoal: 1})
	if err != nil || len(plan.Workers) != 1 || plan.Workers[0] != "cuda-1" {
		t.Fatalf("plan = %+v, err = %v; llama.cpp should match a CUDA worker runtime", plan, err)
	}
}

func TestManagerAdmissionFiltersModelCheckpointAndMeasuredLatency(t *testing.T) {
	m := NewManager(time.Second)
	now := time.Unix(100, 0)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "good", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, ModelDigests: []string{"sha-good"}, CheckpointSupport: true, ProbeLatencyMillis: 4, MemoryFree: 20}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "wrong-model", NodeID: "n2", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, ModelDigests: []string{"sha-other"}, CheckpointSupport: true, ProbeLatencyMillis: 4, MemoryFree: 20}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "slow", NodeID: "n3", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, ModelDigests: []string{"sha-good"}, CheckpointSupport: true, ProbeLatencyMillis: 50}, now)
	plan, err := m.PlanGroup(fabricwire.GroupRequest{GroupID: "g1", Runtime: "llama.cpp", Backends: []string{"cuda"}, WorkerGoal: 1, ModelDigest: "sha-good", RequireCheckpoint: true, MaxProbeLatencyMillis: 10, MinMemoryFreeBytes: 10})
	if err != nil || len(plan.Workers) != 1 || plan.Workers[0] != "good" {
		t.Fatalf("plan = %+v, err = %v", plan, err)
	}
}

func TestManagerCapacityAggregatesReadyWorkerResources(t *testing.T) {
	m := NewManager(10 * time.Second)
	now := time.Now()
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1, MemoryFree: 10, GPUVramTotal: 20, GPUVramFree: 8, GPUCount: 1}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w2", NodeID: "n2", State: fabricwire.WorkerSuspect, Epoch: 1, MemoryFree: 30, GPUVramTotal: 40, GPUVramFree: 16, GPUCount: 2}, now)
	got := m.Capacity()
	if got.Workers != 1 || got.MemoryFree != 10 || got.GPUVramTotal != 20 || got.GPUVramFree != 8 || got.GPUCount != 1 {
		t.Fatalf("capacity = %+v, want only ready worker resources", got)
	}
}

func TestManagerAdmissionFiltersGPUVRAMCapacity(t *testing.T) {
	m := NewManager(time.Second)
	now := time.Unix(100, 0)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "small", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, GPUVramTotal: 8 << 30, GPUVramFree: 2 << 30}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "large", NodeID: "n2", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, GPUVramTotal: 16 << 30, GPUVramFree: 12 << 30}, now)
	plan, err := m.PlanGroup(fabricwire.GroupRequest{GroupID: "g-vram", Runtime: "llama.cpp", Backends: []string{"cuda"}, WorkerGoal: 1, MinGPUVramTotalBytes: 12 << 30, MinGPUVramFreeBytes: 8 << 30})
	if err != nil || len(plan.Workers) != 1 || plan.Workers[0] != "large" {
		t.Fatalf("VRAM-aware plan = %+v, err = %v", plan, err)
	}
}

func TestManagerBuildsVersionedExecutionPlanFromLiveWorkers(t *testing.T) {
	m := NewManager(time.Second)
	now := time.Unix(100, 0)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cuda-1", NodeID: "n1", PeerID: "peer-1", Endpoint: "https://cuda-1/v1/fabric/rpc", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, ModelDigests: []string{"sha256:model"}, CheckpointSupport: true, MemoryFree: 12 << 30, GPUVramFree: 8 << 30}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cuda-2", NodeID: "n2", PeerID: "peer-2", Endpoint: "https://cuda-2/v1/fabric/rpc", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cuda"}, ModelDigests: []string{"sha256:model"}, CheckpointSupport: true, MemoryFree: 16 << 30, GPUVramFree: 10 << 30}, now)
	request := fabricwire.GroupRequest{GroupID: "g-plan", Runtime: "llama.cpp", Backends: []string{"cuda"}, WorkerGoal: 2, ModelDigest: "sha256:model", RequireCheckpoint: true}
	group, err := m.PlanGroup(request)
	if err != nil {
		t.Fatalf("group plan: %v", err)
	}
	plan, err := m.BuildExecutionPlan(request, group, fabricwire.ShardTensor, fabricwire.TransportMTLSRPC)
	if err != nil {
		t.Fatalf("execution plan: %v", err)
	}
	if plan.Version != 1 || plan.ModelDigest != request.ModelDigest || len(plan.Workers) != 2 {
		t.Fatalf("execution plan = %+v", plan)
	}
	if plan.Workers[0].MemoryBudgetBytes == 0 || plan.Workers[0].GPUVramBudgetBytes == 0 || plan.Workers[0].Endpoint == "" {
		t.Fatalf("execution plan omitted live memory budgets: %+v", plan.Workers[0])
	}
	peerIDs := make(map[string]string, len(plan.Workers))
	for _, worker := range plan.Workers {
		peerIDs[worker.WorkerID] = worker.PeerID
	}
	if peerIDs["cuda-1"] != "peer-1" || peerIDs["cuda-2"] != "peer-2" {
		t.Fatalf("execution plan omitted peer identities: %+v", plan.Workers)
	}
}

func TestReadyTrainingReplacementNodesExcludeCurrentWorkers(t *testing.T) {
	now := time.Now()
	m := NewManager(time.Second)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "node-a", NodeID: "a", Endpoint: "https://node-a:14324", State: fabricwire.WorkerReady, Epoch: 1, Backends: []string{"cuda"}, GPUCount: 1}, now)
	m.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "node-b", NodeID: "b", Endpoint: "https://node-b:14324", State: fabricwire.WorkerReady, Epoch: 1, Backends: []string{"cuda"}, GPUCount: 2}, now)
	request := TrainingRequest{Nodes: []TrainingNode{{WorkerID: "node-a", Address: "https://node-a:14324", Backend: "cuda"}}}
	replacements := readyTrainingReplacementNodes(m, request)
	if len(replacements) != 1 || replacements[0].WorkerID != "node-b" || replacements[0].GPUCount != 2 {
		t.Fatalf("replacement nodes = %+v", replacements)
	}
}
