// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os/exec"
	"reflect"
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
