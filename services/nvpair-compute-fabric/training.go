// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"strconv"

	"nvpair-shared/fabricwire"
)

type TrainingParallelism string

const (
	TrainingDataParallel TrainingParallelism = "data"
	TrainingFSDP         TrainingParallelism = "fsdp"
)

type TrainingNode struct {
	WorkerID string `json:"workerId"`
	Address  string `json:"address"`
	Backend  string `json:"backend"`
	GPUCount uint32 `json:"gpuCount,omitempty"`
}

type TrainingRequest struct {
	JobID                   string              `json:"jobId"`
	ModelDigest             string              `json:"modelDigest"`
	TrainerPath             string              `json:"trainerPath"`
	ModelPath               string              `json:"modelPath"`
	DatasetPath             string              `json:"datasetPath"`
	CheckpointDirectory     string              `json:"checkpointDirectory"`
	ResumeCheckpoint        string              `json:"resumeCheckpoint,omitempty"`
	RendezvousEndpoint      string              `json:"rendezvousEndpoint"`
	Parallelism             TrainingParallelism `json:"parallelism"`
	ProcessesPerNode        uint32              `json:"processesPerNode"`
	CheckpointIntervalSteps uint64              `json:"checkpointIntervalSteps"`
	Nodes                   []TrainingNode      `json:"nodes"`
}

func (r TrainingRequest) Validate() error {
	if r.JobID == "" || r.ModelDigest == "" || r.TrainerPath == "" || r.ModelPath == "" || r.DatasetPath == "" || r.CheckpointDirectory == "" || r.RendezvousEndpoint == "" {
		return fmt.Errorf("training request requires jobId, modelDigest, trainerPath, modelPath, datasetPath, checkpointDirectory, and rendezvousEndpoint")
	}
	if r.Parallelism != TrainingDataParallel && r.Parallelism != TrainingFSDP {
		return fmt.Errorf("unsupported torchrun parallelism %q", r.Parallelism)
	}
	if r.ProcessesPerNode == 0 || r.CheckpointIntervalSteps == 0 || len(r.Nodes) == 0 {
		return fmt.Errorf("training request requires nodes, positive processesPerNode, and checkpointIntervalSteps")
	}
	if len(r.Nodes) > 1 {
		backend := r.Nodes[0].Backend
		if backend != "cpu" && backend != "cuda" {
			return fmt.Errorf("torchrun adapter does not support backend %q", backend)
		}
		for _, node := range r.Nodes[1:] {
			if node.Backend != backend {
				return fmt.Errorf("torchrun adapter requires homogeneous backends, got %q and %q", backend, node.Backend)
			}
		}
	}
	for _, node := range r.Nodes {
		if node.WorkerID == "" || node.Address == "" {
			return fmt.Errorf("training node requires workerId and address")
		}
		if node.Backend == "cuda" && node.GPUCount == 0 {
			return fmt.Errorf("CUDA training node %q reports no GPU", node.WorkerID)
		}
	}
	return nil
}

func (r TrainingRequest) Backend() fabricwire.Transport {
	if len(r.Nodes) <= 1 {
		return fabricwire.TransportLocal
	}
	return fabricwire.TransportMTLSRPC
}

func BuildTorchRunCommand(request TrainingRequest, nodeRank uint32) ([]string, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if nodeRank >= uint32(len(request.Nodes)) {
		return nil, fmt.Errorf("node rank %d is outside the training world", nodeRank)
	}
	args := []string{
		"torchrun",
		"--nnodes", strconv.Itoa(len(request.Nodes)),
		"--nproc-per-node", strconv.FormatUint(uint64(request.ProcessesPerNode), 10),
		"--node-rank", strconv.FormatUint(uint64(nodeRank), 10),
		"--rdzv-backend", "c10d",
		"--rdzv-endpoint", request.RendezvousEndpoint,
		request.TrainerPath,
		"--model", request.ModelPath,
		"--dataset", request.DatasetPath,
		"--output-dir", request.CheckpointDirectory,
		"--parallelism", string(request.Parallelism),
		"--checkpoint-interval-steps", strconv.FormatUint(request.CheckpointIntervalSteps, 10),
		"--model-digest", request.ModelDigest,
	}
	if request.ResumeCheckpoint != "" {
		args = append(args, "--resume", request.ResumeCheckpoint)
	}
	return args, nil
}
