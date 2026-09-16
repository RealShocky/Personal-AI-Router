// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os/exec"
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestStartInferenceJobPlansAndPersists(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mgr := NewManager(time.Second)
	mgr.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID:   "cuda-1",
		NodeID:     "n1",
		State:      fabricwire.WorkerReady,
		Epoch:      1,
		Runtime:    "cuda",
		Backends:   []string{"cuda"},
		MemoryFree: 1,
	}, time.Now())

	executions := NewExecutionManager(ctx, "")
	executions.command = func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "go", "version")
	}
	jobs := NewJobStore("")
	request := StartRequest{
		JobID:       "job-api",
		ModelDigest: "sha256:model",
		ServerPath:  "llama-server",
		ModelPath:   "model.gguf",
		HTTPPort:    19090,
		Group: fabricwire.GroupRequest{
			GroupID:    "job-api",
			Runtime:    "llama.cpp",
			Backends:   []string{"cuda"},
			WorkerGoal: 1,
		},
	}

	execution, err := startInferenceJob(mgr, jobs, executions, request)
	if err != nil {
		t.Fatalf("startInferenceJob() error = %v", err)
	}
	if execution.JobID != request.JobID || execution.State != "running" {
		t.Fatalf("execution = %+v", execution)
	}
	if _, ok := jobs.Job(request.JobID); !ok {
		t.Fatal("inference job was not persisted")
	}
}

func TestStartInferenceJobAdmitsWorkersWithoutLocalModelCopy(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mgr := NewManager(time.Second)
	mgr.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID: "cuda-remote",
		NodeID:   "remote-host",
		State:    fabricwire.WorkerReady,
		Epoch:    1,
		Runtime:  "llama.cpp",
		Backends: []string{"cuda"},
		MemoryFree: 1,
	}, time.Now())

	executions := NewExecutionManager(ctx, "")
	executions.command = func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		return exec.CommandContext(ctx, "go", "version")
	}
	jobs := NewJobStore("")
	request := StartRequest{
		JobID:       "remote-model-job",
		ModelDigest: "sha256:coordinator-model",
		ServerPath:  "llama-server",
		ModelPath:   "model.gguf",
		HTTPPort:    19091,
		Group: fabricwire.GroupRequest{
			GroupID:    "remote-model-job",
			Runtime:    "llama.cpp",
			Backends:   []string{"cuda"},
			WorkerGoal: 1,
		},
	}

	if _, err := startInferenceJob(mgr, jobs, executions, request); err != nil {
		t.Fatalf("startInferenceJob() should admit a remote model copy: %v", err)
	}
}
