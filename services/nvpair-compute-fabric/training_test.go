// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"os/exec"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestTorchRunTrainingCommandUsesRendezvousAndRank(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-a",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "10.0.0.1:29400",
		Parallelism:             TrainingFSDP,
		ProcessesPerNode:        2,
		CheckpointIntervalSteps: 100,
		Nodes: []TrainingNode{
			{WorkerID: "cuda-a", Address: "10.0.0.2", Backend: "cuda", GPUCount: 2},
			{WorkerID: "cuda-b", Address: "10.0.0.3", Backend: "cuda", GPUCount: 2},
		},
	}
	args, err := BuildTorchRunCommand(request, 1)
	if err != nil {
		t.Fatalf("build command: %v", err)
	}
	want := []string{"torchrun", "--nnodes", "2", "--nproc-per-node", "2", "--node-rank", "1", "--rdzv-backend", "c10d", "--rdzv-endpoint", "10.0.0.1:29400", "train.py", "--model", "model.safetensors", "--dataset", "data.jsonl", "--output-dir", "checkpoints", "--parallelism", "fsdp", "--checkpoint-interval-steps", "100", "--model-digest", "sha256:model"}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

func TestTrainingAdmissionRejectsMetalForTorchRun(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-metal",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "10.0.0.1:29400",
		Parallelism:             TrainingFSDP,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 100,
		Nodes:                   []TrainingNode{{WorkerID: "mac", Address: "10.0.0.4", Backend: "metal"}},
	}
	if _, err := BuildTorchRunCommand(request, 0); err == nil {
		t.Fatal("Metal training was accepted by the CUDA/CPU torchrun adapter")
	}
}

func TestTrainingRequestMapsToFabricCapabilities(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-cpu",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 100,
		Nodes:                   []TrainingNode{{WorkerID: "cpu", Address: "127.0.0.1", Backend: "cpu"}},
	}
	if err := request.Validate(); err != nil {
		t.Fatalf("CPU training request rejected: %v", err)
	}
	if request.Backend() != fabricwire.TransportLocal {
		t.Fatalf("training backend transport = %q, want local", request.Backend())
	}
}

func TestTrainingManagerSupervisesLocalTorchRunProcess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	manager := NewTrainingManager(ctx)
	var got []string
	manager.command = func(ctx context.Context, path string, args ...string) *exec.Cmd {
		got = append([]string{path}, args...)
		return exec.CommandContext(ctx, "go", "version")
	}
	request := TrainingRequest{
		JobID:                   "train-local",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "cpu", Address: "127.0.0.1", Backend: "cpu"}},
	}
	started, err := manager.Start(request, 0)
	if err != nil {
		t.Fatalf("start training: %v", err)
	}
	if started.JobID != request.JobID || started.PID == 0 || len(got) == 0 || got[0] != "torchrun" {
		t.Fatalf("execution = %+v command = %#v", started, got)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		status, ok := manager.Status(request.JobID)
		if ok && status.State == "failed" {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("training process did not reach terminal state: %+v", started)
}

func TestTrainingCoordinatorRollsBackPartialGroupLaunch(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-group",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "10.0.0.1:29400",
		Parallelism:             TrainingFSDP,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes: []TrainingNode{
			{WorkerID: "node-a", Address: "10.0.0.2", Backend: "cuda", GPUCount: 1},
			{WorkerID: "node-b", Address: "10.0.0.3", Backend: "cuda", GPUCount: 1},
		},
	}
	stopped := ""
	coordinator := NewTrainingCoordinator(
		func(_ context.Context, node TrainingNode, _ TrainingRequest, rank uint32) (TrainingExecution, error) {
			if node.WorkerID == "node-b" {
				return TrainingExecution{}, fmt.Errorf("worker unavailable")
			}
			return TrainingExecution{JobID: "train-group", NodeRank: rank, State: "running"}, nil
		},
		func(_ context.Context, jobID string, _ TrainingNode) error {
			stopped = jobID
			return nil
		},
	)
	if _, err := coordinator.StartGroup(context.Background(), request); err == nil {
		t.Fatal("partial group launch unexpectedly succeeded")
	}
	if stopped != request.JobID {
		t.Fatalf("rollback stopped job = %q, want %q", stopped, request.JobID)
	}
}

func TestTrainingCoordinatorTracksAndStopsSuccessfulGroup(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-tracked",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "10.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "node-a", Address: "10.0.0.2", Backend: "cpu"}},
	}
	coordinator := NewTrainingCoordinator(
		func(_ context.Context, _ TrainingNode, _ TrainingRequest, rank uint32) (TrainingExecution, error) {
			return TrainingExecution{JobID: request.JobID, NodeRank: rank, State: "running"}, nil
		},
		func(_ context.Context, jobID string, _ TrainingNode) error {
			if jobID != request.JobID {
				t.Fatalf("stopped unexpected job %q", jobID)
			}
			return nil
		},
	)
	if _, err := coordinator.StartGroup(context.Background(), request); err != nil {
		t.Fatalf("start group: %v", err)
	}
	status, ok := coordinator.StatusGroup(request.JobID)
	if !ok || status.State != "running" || len(status.Executions) != 1 {
		t.Fatalf("group status = %+v, found=%v", status, ok)
	}
	if err := coordinator.StopGroup(context.Background(), request.JobID); err != nil {
		t.Fatalf("stop group: %v", err)
	}
	status, _ = coordinator.StatusGroup(request.JobID)
	if status.State != "stopped" {
		t.Fatalf("group state after stop = %q", status.State)
	}
}

func TestTrainingCoordinatorRefreshesRemoteExecutionState(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-refresh",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "node-a", Address: "127.0.0.1", Backend: "cpu"}},
	}
	coordinator := NewTrainingCoordinator(
		func(_ context.Context, _ TrainingNode, _ TrainingRequest, rank uint32) (TrainingExecution, error) {
			return TrainingExecution{JobID: request.JobID, NodeRank: rank, State: "running"}, nil
		},
		func(_ context.Context, _ string, _ TrainingNode) error { return nil },
	)
	coordinator.SetStatusFunc(func(_ context.Context, jobID string, _ TrainingNode) (TrainingExecution, error) {
		return TrainingExecution{JobID: jobID, NodeRank: 0, PID: 42, State: "failed"}, nil
	})
	if _, err := coordinator.StartGroup(context.Background(), request); err != nil {
		t.Fatalf("start group: %v", err)
	}
	if err := coordinator.RefreshStatus(context.Background(), request.JobID); err != nil {
		t.Fatalf("refresh status: %v", err)
	}
	status, ok := coordinator.StatusGroup(request.JobID)
	if !ok || status.State != "recoverable" || status.Executions[0].PID != 42 || status.Executions[0].State != "failed" {
		t.Fatalf("refreshed status = %+v, found=%v", status, ok)
	}
}

func TestTrainingCoordinatorPersistsCheckpointAndGroupState(t *testing.T) {
	stateDir := t.TempDir()
	request := TrainingRequest{
		JobID:                   "train-persisted",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "cpu", Address: "127.0.0.1", Backend: "cpu"}},
	}
	start := func(_ context.Context, _ TrainingNode, _ TrainingRequest, rank uint32) (TrainingExecution, error) {
		return TrainingExecution{JobID: request.JobID, NodeRank: rank, State: "running"}, nil
	}
	stop := func(_ context.Context, _ string, _ TrainingNode) error { return nil }
	first := NewTrainingCoordinator(start, stop, stateDir)
	if _, err := first.StartGroup(context.Background(), request); err != nil {
		t.Fatalf("start group: %v", err)
	}
	status, ok := first.StatusGroup(request.JobID)
	if !ok {
		t.Fatal("started group not found")
	}
	checkpoint := fabricwire.Checkpoint{JobID: request.JobID, GroupID: request.JobID, Epoch: status.Epoch, Stage: 4, Filename: "step-0004.pt"}
	if err := first.SaveCheckpoint(request.JobID, checkpoint); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}
	second := NewTrainingCoordinator(start, stop, stateDir)
	reloaded, ok := second.StatusGroup(request.JobID)
	if !ok || reloaded.Checkpoint.Filename != checkpoint.Filename || reloaded.Checkpoint.Stage != checkpoint.Stage {
		t.Fatalf("reloaded status = %+v, found=%v", reloaded, ok)
	}
}

func TestTrainingCoordinatorRecoversFromCheckpointWithReplacementWorkers(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-recover",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "node-a", Address: "127.0.0.1", Backend: "cpu"}},
	}
	startedRequests := make([]TrainingRequest, 0, 2)
	coordinator := NewTrainingCoordinator(
		func(_ context.Context, _ TrainingNode, request TrainingRequest, _ uint32) (TrainingExecution, error) {
			startedRequests = append(startedRequests, request)
			return TrainingExecution{JobID: request.JobID, State: "running"}, nil
		},
		func(_ context.Context, _ string, _ TrainingNode) error { return nil },
	)
	if _, err := coordinator.StartGroup(context.Background(), request); err != nil {
		t.Fatalf("start group: %v", err)
	}
	status, _ := coordinator.StatusGroup(request.JobID)
	if err := coordinator.SaveCheckpoint(request.JobID, fabricwire.Checkpoint{JobID: request.JobID, GroupID: request.JobID, Epoch: status.Epoch, Stage: 8, Filename: "step-0008.pt"}); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}
	if err := coordinator.MarkFailed(request.JobID); err != nil {
		t.Fatalf("mark failed: %v", err)
	}
	replacement := []TrainingNode{{WorkerID: "node-b", Address: "127.0.0.2", Backend: "cpu"}}
	if _, err := coordinator.RecoverGroup(context.Background(), request.JobID, replacement, "127.0.0.1:29401"); err != nil {
		t.Fatalf("recover group: %v", err)
	}
	status, _ = coordinator.StatusGroup(request.JobID)
	if status.State != "running" || status.RecoveryAttempts != 1 || len(startedRequests) != 2 || startedRequests[1].ResumeCheckpoint != "step-0008.pt" {
		t.Fatalf("recovered status = %+v requests = %+v", status, startedRequests)
	}
}

func TestTrainingCoordinatorAutomaticallyRecoversAfterRankFailure(t *testing.T) {
	request := TrainingRequest{
		JobID:                   "train-auto-recover",
		ModelDigest:             "sha256:model",
		TrainerPath:             "train.py",
		ModelPath:               "model.safetensors",
		DatasetPath:             "data.jsonl",
		CheckpointDirectory:     "checkpoints",
		RendezvousEndpoint:      "127.0.0.1:29400",
		Parallelism:             TrainingDataParallel,
		ProcessesPerNode:        1,
		CheckpointIntervalSteps: 10,
		Nodes:                   []TrainingNode{{WorkerID: "node-a", Address: "127.0.0.1", Backend: "cpu"}},
	}
	startedRequests := make([]TrainingRequest, 0, 2)
	coordinator := NewTrainingCoordinator(
		func(_ context.Context, _ TrainingNode, next TrainingRequest, _ uint32) (TrainingExecution, error) {
			startedRequests = append(startedRequests, next)
			return TrainingExecution{JobID: next.JobID, State: "running"}, nil
		},
		func(_ context.Context, _ string, _ TrainingNode) error { return nil },
	)
	coordinator.SetStatusFunc(func(_ context.Context, _ string, _ TrainingNode) (TrainingExecution, error) {
		return TrainingExecution{}, fmt.Errorf("rank unavailable")
	})
	coordinator.SetRecoveryNodeProvider(func(_ TrainingRequest) []TrainingNode {
		return []TrainingNode{{WorkerID: "node-b", Address: "127.0.0.2", Backend: "cpu"}}
	})
	if _, err := coordinator.StartGroup(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	status, _ := coordinator.StatusGroup(request.JobID)
	if err := coordinator.SaveCheckpoint(request.JobID, fabricwire.Checkpoint{JobID: request.JobID, GroupID: request.JobID, Epoch: status.Epoch, Stage: 8, Filename: "step-0008.pt"}); err != nil {
		t.Fatal(err)
	}
	if err := coordinator.RefreshStatus(context.Background(), request.JobID); err != nil {
		t.Fatal(err)
	}
	status, _ = coordinator.StatusGroup(request.JobID)
	if status.State != "running" || status.RecoveryAttempts != 1 || len(startedRequests) != 2 || startedRequests[1].ResumeCheckpoint != "step-0008.pt" {
		t.Fatalf("auto-recovered status = %+v requests = %+v", status, startedRequests)
	}
}
