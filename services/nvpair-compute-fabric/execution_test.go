// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"reflect"
	"sync"
	"strings"
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestExecutionManagerBuildsDistributedLlamaCommand(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "")
	var got []string
	manager.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		got = append([]string{path}, args...)
		return exec.CommandContext(ctx, "go", "version")
	}
	plan := fabricwire.GroupPlan{GroupID: "g1", Runtime: "llama.cpp", Workers: []string{"w1", "w2"}}
	result, err := manager.Start(StartRequest{JobID: "j1", ServerPath: "llama-server", ModelPath: "model.gguf", HTTPPort: 11450}, plan)
	if err != nil {
		t.Fatal(err)
	}
	if result.RPCPeers != 0 || result.State != "running" {
		t.Fatalf("execution = %+v", result)
	}
	want := []string{"llama-server", "--model", "model.gguf", "--host", "127.0.0.1", "--port", "11450", "--n-gpu-layers", "all"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %#v, want %#v", got, want)
	}
}

func TestExecutionManagerCreatesOneRelayPerAdvertisedWorker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "missing-cluster")
	var got []string
	manager.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		got = append([]string{path}, args...)
		return exec.CommandContext(ctx, "go", "version")
	}
	plan := fabricwire.GroupPlan{GroupID: "g1", Runtime: "llama.cpp", Workers: []string{"w1", "w2"}, Endpoints: map[string]string{"w1": "https://worker-a/v1/fabric/rpc", "w2": "https://worker-b/v1/fabric/rpc"}}
	result, err := manager.Start(StartRequest{JobID: "j1", ServerPath: "llama-server", ModelPath: "model.gguf", HTTPPort: 11450}, plan)
	if err != nil {
		t.Fatal(err)
	}
	if result.RPCPeers != 2 {
		t.Fatalf("RPC peers = %d, want 2", result.RPCPeers)
	}
	if len(got) == 0 || got[len(got)-2] != "--rpc" {
		t.Fatalf("command missing --rpc: %#v", got)
	}
}

func TestExecutionManagerRejectsInvalidExecutionPlanBeforeLaunch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "")
	launched := false
	manager.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		launched = true
		return exec.CommandContext(ctx, "go", "version")
	}
	_, err := manager.StartWithExecutionPlan(
		StartRequest{JobID: "j-invalid", ModelDigest: "sha256:model", ServerPath: "llama-server", ModelPath: "model.gguf", HTTPPort: 11450},
		fabricwire.ExecutionPlan{Version: 1, PlanID: "p1", Runtime: "llama.cpp", ModelDigest: "sha256:model", ShardStrategy: fabricwire.ShardTensor, Transport: fabricwire.TransportLocal},
	)
	if err == nil || launched {
		t.Fatalf("invalid execution plan launched: err=%v launched=%v", err, launched)
	}
}

func TestExecutionManagerCarriesExecutionPlanGroupIntoRecoveryState(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "missing-cluster")
	manager.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "go", "version")
	}
	plan := fabricwire.ExecutionPlan{
		Version:       1,
		PlanID:        "plan-1",
		Epoch:         7,
		Runtime:       "llama.cpp",
		ModelDigest:   "sha256:model",
		ShardStrategy: fabricwire.ShardTensor,
		Transport:     fabricwire.TransportLocal,
		Workers: []fabricwire.WorkerAssignment{{
			WorkerID:          "w1",
			MemoryBudgetBytes: 1,
		}},
		Checkpoint: fabricwire.CheckpointPolicy{Enabled: true, IntervalSteps: 1},
		Failover:   fabricwire.FailoverPolicy{Enabled: true, MaxAttempts: 1},
	}
	if _, err := manager.StartWithExecutionPlan(StartRequest{
		JobID:       "job-1",
		ModelDigest: "sha256:model",
		ServerPath:  "llama-server",
		ModelPath:   "model.gguf",
		HTTPPort:    11450,
	}, plan); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	running := manager.executions["job-1"]
	manager.mu.Unlock()
	if running == nil || running.group.GroupID != "plan-1" || running.group.Runtime != "llama.cpp" || running.group.WorkerGoal != 1 {
		t.Fatalf("recovery group = %+v", running)
	}
}

func TestExecutionManagerRequiresCheckpointStorageForPeriodicCheckpoints(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "")
	_, err := manager.Start(StartRequest{
		JobID:                     "job-checkpoint",
		ServerPath:                "llama-server",
		ModelPath:                 "model.gguf",
		HTTPPort:                  11450,
		CheckpointIntervalSeconds: 5,
	}, fabricwire.GroupPlan{GroupID: "g1", Runtime: "llama.cpp", Workers: []string{"w1"}})
	if err == nil || !strings.Contains(err.Error(), "checkpoint interval") {
		t.Fatalf("err = %v, want checkpoint storage validation", err)
	}
}

func TestExecutionManagerRejectsCheckpointPathSeparators(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "")
	_, err := manager.Start(StartRequest{
		JobID:          "job-checkpoint-path",
		ServerPath:     "llama-server",
		ModelPath:      "model.gguf",
		HTTPPort:       11450,
		CheckpointFile: "nested/checkpoint.slot",
	}, fabricwire.GroupPlan{GroupID: "g1", Runtime: "llama.cpp", Workers: []string{"w1"}})
	if err == nil || !strings.Contains(err.Error(), "relative filename") {
		t.Fatalf("err = %v, want relative filename validation", err)
	}
}

func TestValidSlotFilename(t *testing.T) {
	for _, test := range []struct {
		name  string
		valid bool
	}{
		{name: "checkpoint.slot", valid: true},
		{name: "step-0001.slot", valid: true},
		{name: "nested/checkpoint.slot", valid: false},
		{name: `nested\\checkpoint.slot`, valid: false},
		{name: "..", valid: false},
		{name: "", valid: false},
	} {
		if got := validSlotFilename(test.name); got != test.valid {
			t.Errorf("validSlotFilename(%q) = %v, want %v", test.name, got, test.valid)
		}
	}
}

func TestSlotCheckpointRequestsUseJSONBody(t *testing.T) {
	var mu sync.Mutex
	requests := make([]string, 0, 2)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", request.Method)
		}
		if request.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content type = %q, want application/json", request.Header.Get("Content-Type"))
		}
		var payload map[string]string
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		if payload["filename"] != "checkpoint.slot" {
			t.Errorf("filename = %q, want checkpoint.slot", payload["filename"])
		}
		mu.Lock()
		requests = append(requests, request.URL.Query().Get("action"))
		mu.Unlock()
		writer.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	port := server.Listener.Addr().(*net.TCPAddr).Port

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewExecutionManager(ctx, "")
	manager.executions["job-slot"] = &runningExecution{execution: Execution{JobID: "job-slot", State: "running", HTTPPort: port}}
	if err := manager.SaveCheckpoint(fabricwire.Checkpoint{JobID: "job-slot", SlotID: 0, Filename: "checkpoint.slot"}); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}

	restoreDone := make(chan struct{})
	go func() {
		restoreSlotCheckpoint(ctx, port, 0, "checkpoint.slot")
		close(restoreDone)
	}()
	select {
	case <-restoreDone:
	case <-time.After(2 * time.Second):
		t.Fatal("restore checkpoint did not complete")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 || requests[0] != "save" || requests[1] != "restore" {
		t.Fatalf("actions = %v, want [save restore]", requests)
	}
}

func TestExecutionManagerAutomaticallyRecoversFailedJobIntoFreshGroupEpoch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewManager(time.Second)
	manager.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "w1", NodeID: "n1", State: fabricwire.WorkerReady, Epoch: 1, Runtime: "llama.cpp", Backends: []string{"cpu"}}, time.Now())
	executions := NewExecutionManager(ctx, "")
	starts := 0
	executions.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		starts++
		return exec.CommandContext(ctx, "go", "version")
	}
	plan := fabricwire.GroupPlan{GroupID: "g1", Runtime: "llama.cpp", Workers: []string{"w1"}, Epoch: 10}
	jobs := NewJobStore("")
	if err := jobs.Submit(JobRecord{JobID: "j1", GroupID: "g1", Epoch: 10}); err != nil {
		t.Fatal(err)
	}
	if _, err := executions.Start(StartRequest{JobID: "j1", ServerPath: "llama-server", ModelPath: "model.gguf", HTTPPort: 11450, Group: fabricwire.GroupRequest{GroupID: "g1", Runtime: "llama.cpp", Backends: []string{"cpu"}, WorkerGoal: 1}}, plan); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		status, ok := executions.Status("j1")
		if ok && status.State == "failed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	executions.RecoverFailed(manager, jobs)
	job, ok := jobs.Job("j1")
	if !ok || job.Epoch == 10 || job.GroupID == "g1" || starts < 2 {
		t.Fatalf("job=%+v starts=%d", job, starts)
	}
}
