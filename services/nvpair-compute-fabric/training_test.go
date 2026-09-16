// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"testing"

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
		JobID:               "train-metal",
		ModelDigest:         "sha256:model",
		TrainerPath:         "train.py",
		ModelPath:           "model.safetensors",
		DatasetPath:         "data.jsonl",
		CheckpointDirectory: "checkpoints",
		RendezvousEndpoint:  "10.0.0.1:29400",
		Parallelism:         TrainingFSDP,
		ProcessesPerNode:    1,
		Nodes:               []TrainingNode{{WorkerID: "mac", Address: "10.0.0.4", Backend: "metal"}},
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
