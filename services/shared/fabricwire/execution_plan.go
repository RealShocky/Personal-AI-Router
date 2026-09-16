// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fabricwire

import "fmt"

type ShardStrategy string

const (
	ShardReplicated ShardStrategy = "replicated"
	ShardTensor     ShardStrategy = "tensor"
	ShardPipeline   ShardStrategy = "pipeline"
)

type Transport string

const (
	TransportMTLSRPC Transport = "mtls-rpc"
	TransportLocal   Transport = "local"
)

type WorkerAssignment struct {
	WorkerID           string `json:"workerId"`
	PeerID             string `json:"peerId,omitempty"`
	Endpoint           string `json:"endpoint,omitempty"`
	ShardIndex         uint32 `json:"shardIndex"`
	MemoryBudgetBytes  uint64 `json:"memoryBudgetBytes"`
	GPUVramBudgetBytes uint64 `json:"gpuVramBudgetBytes,omitempty"`
}

type CheckpointPolicy struct {
	Enabled       bool   `json:"enabled"`
	IntervalSteps uint64 `json:"intervalSteps,omitempty"`
	Directory     string `json:"directory,omitempty"`
}

type FailoverPolicy struct {
	Enabled     bool   `json:"enabled"`
	MaxAttempts uint32 `json:"maxAttempts,omitempty"`
}

type ExecutionPlan struct {
	Version       uint32             `json:"version"`
	PlanID        string             `json:"planId"`
	Epoch         uint64             `json:"epoch"`
	Runtime       string             `json:"runtime"`
	ModelDigest   string             `json:"modelDigest"`
	ShardStrategy ShardStrategy      `json:"shardStrategy"`
	Transport     Transport          `json:"transport"`
	Workers       []WorkerAssignment `json:"workers"`
	Checkpoint    CheckpointPolicy   `json:"checkpoint"`
	Failover      FailoverPolicy     `json:"failover"`
}

func (p ExecutionPlan) Validate() error {
	if p.Version == 0 {
		return fmt.Errorf("execution plan requires a positive version")
	}
	if p.PlanID == "" || p.Runtime == "" || p.ModelDigest == "" {
		return fmt.Errorf("execution plan requires planId, runtime, and modelDigest")
	}
	if p.ShardStrategy != ShardReplicated && p.ShardStrategy != ShardTensor && p.ShardStrategy != ShardPipeline {
		return fmt.Errorf("unsupported shard strategy %q", p.ShardStrategy)
	}
	if p.Transport != TransportMTLSRPC && p.Transport != TransportLocal {
		return fmt.Errorf("unsupported transport %q", p.Transport)
	}
	if len(p.Workers) == 0 {
		return fmt.Errorf("execution plan requires at least one worker assignment")
	}
	seen := make(map[string]bool, len(p.Workers))
	for _, worker := range p.Workers {
		if worker.WorkerID == "" {
			return fmt.Errorf("execution plan contains an assignment without workerId")
		}
		if seen[worker.WorkerID] {
			return fmt.Errorf("execution plan assigns worker %q more than once", worker.WorkerID)
		}
		seen[worker.WorkerID] = true
		if worker.MemoryBudgetBytes == 0 {
			return fmt.Errorf("execution plan worker %q requires a memory budget", worker.WorkerID)
		}
	}
	if p.Checkpoint.Enabled && p.Checkpoint.IntervalSteps == 0 {
		return fmt.Errorf("enabled checkpoint policy requires intervalSteps")
	}
	if p.Failover.Enabled && p.Failover.MaxAttempts == 0 {
		return fmt.Errorf("enabled failover policy requires maxAttempts")
	}
	return nil
}
