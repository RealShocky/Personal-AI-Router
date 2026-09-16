// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fabricwire

import "fmt"

const LogicalDeviceProtocolVersion uint32 = 1

type Provider string

const (
	ProviderCPU   Provider = "cpu"
	ProviderCUDA  Provider = "cuda"
	ProviderMetal Provider = "metal"
)

type MemoryTierKind string

const (
	MemoryTierHost       MemoryTierKind = "host"
	MemoryTierGPU        MemoryTierKind = "gpu"
	MemoryTierRemoteHost MemoryTierKind = "remote-host"
	MemoryTierRemoteGPU  MemoryTierKind = "remote-gpu"
)

type MemoryTier struct {
	TierID               string         `json:"tierId"`
	Kind                 MemoryTierKind `json:"kind"`
	CapacityBytes        uint64         `json:"capacityBytes"`
	FreeBytes            uint64         `json:"freeBytes"`
	BandwidthBytesPerSec uint64         `json:"bandwidthBytesPerSec,omitempty"`
	LatencyMicros        uint64         `json:"latencyMicros,omitempty"`
	Local                bool           `json:"local"`
}

type ProviderCapability struct {
	Provider            Provider     `json:"provider"`
	DeviceID            string       `json:"deviceId"`
	SupportsExecution   bool         `json:"supportsExecution"`
	SupportsCollectives []string     `json:"supportsCollectives,omitempty"`
	MemoryTiers         []MemoryTier `json:"memoryTiers"`
	MaxPageBytes        uint64       `json:"maxPageBytes,omitempty"`
}

type LogicalDeviceDescribe struct {
	ProtocolVersion uint32               `json:"protocolVersion"`
	WorkerID        string               `json:"workerId"`
	Providers       []ProviderCapability `json:"providers"`
}

type LogicalDeviceRequest struct {
	ProtocolVersion uint32 `json:"protocolVersion"`
	RequestID       string `json:"requestId"`
	PlanID          string `json:"planId"`
	Epoch           uint64 `json:"epoch"`
	DeadlineUnixMS  int64  `json:"deadlineUnixMs,omitempty"`
}

type PageSpec struct {
	PageID string `json:"pageId"`
	Bytes  uint64 `json:"bytes"`
	DType  string `json:"dtype,omitempty"`
	Layout string `json:"layout,omitempty"`
}

type LogicalDevicePlanRequest struct {
	LogicalDeviceRequest
	Runtime       string        `json:"runtime"`
	ModelDigest   string        `json:"modelDigest"`
	ShardStrategy ShardStrategy `json:"shardStrategy"`
	Providers     []Provider    `json:"providers,omitempty"`
	WorkerGoal    uint32        `json:"workerGoal"`
	Pages         []PageSpec    `json:"pages"`
}

func (r LogicalDeviceRequest) ValidateForEpoch(current uint64) error {
	if r.ProtocolVersion != LogicalDeviceProtocolVersion {
		return fmt.Errorf("unsupported logical device protocol version %d", r.ProtocolVersion)
	}
	if r.RequestID == "" || r.PlanID == "" || r.Epoch == 0 {
		return fmt.Errorf("logical device request requires requestId, planId, and positive epoch")
	}
	if r.Epoch != current {
		return fmt.Errorf("logical device request epoch %d does not match current epoch %d", r.Epoch, current)
	}
	return nil
}

type PagePlacement struct {
	PageID   string `json:"pageId"`
	WorkerID string `json:"workerId"`
	TierID   string `json:"tierId"`
	Bytes    uint64 `json:"bytes"`
	DType    string `json:"dtype,omitempty"`
	Layout   string `json:"layout,omitempty"`
	Digest   string `json:"digest,omitempty"`
	Epoch    uint64 `json:"epoch"`
	Replica  bool   `json:"replica,omitempty"`
}

type LogicalDevicePlan struct {
	Version       uint32             `json:"version"`
	PlanID        string             `json:"planId"`
	Epoch         uint64             `json:"epoch"`
	Runtime       string             `json:"runtime"`
	ModelDigest   string             `json:"modelDigest"`
	ShardStrategy ShardStrategy      `json:"shardStrategy"`
	Workers       []WorkerAssignment `json:"workers"`
	Pages         []PagePlacement    `json:"pages"`
}

type LogicalDeviceState string

const (
	LogicalDevicePlanned   LogicalDeviceState = "planned"
	LogicalDeviceLeased    LogicalDeviceState = "leased"
	LogicalDeviceRunning   LogicalDeviceState = "running"
	LogicalDeviceDegraded  LogicalDeviceState = "degraded"
	LogicalDeviceRecovering LogicalDeviceState = "recovering"
	LogicalDeviceCompleted LogicalDeviceState = "completed"
	LogicalDeviceCancelled LogicalDeviceState = "cancelled"
)

type LogicalDeviceStatus struct {
	PlanID       string             `json:"planId"`
	Epoch        uint64             `json:"epoch"`
	State        LogicalDeviceState `json:"state"`
	Workers      []string           `json:"workers"`
	Pages        []PagePlacement    `json:"pages"`
	TransferIDs  []string           `json:"transferIds,omitempty"`
	UpdatedAtMS  int64              `json:"updatedAtMs"`
}

func (p LogicalDevicePlan) Validate() error {
	if p.Version != LogicalDeviceProtocolVersion {
		return fmt.Errorf("unsupported logical device plan version %d", p.Version)
	}
	if p.PlanID == "" || p.Runtime == "" || p.ModelDigest == "" || p.Epoch == 0 {
		return fmt.Errorf("logical device plan requires planId, runtime, modelDigest, and positive epoch")
	}
	if p.ShardStrategy != ShardReplicated && p.ShardStrategy != ShardTensor && p.ShardStrategy != ShardPipeline {
		return fmt.Errorf("unsupported shard strategy %q", p.ShardStrategy)
	}
	if len(p.Workers) == 0 {
		return fmt.Errorf("logical device plan requires at least one worker")
	}
	seenWorkers := make(map[string]bool, len(p.Workers))
	for _, worker := range p.Workers {
		if worker.WorkerID == "" || worker.MemoryBudgetBytes == 0 {
			return fmt.Errorf("logical device plan contains an invalid worker assignment")
		}
		if seenWorkers[worker.WorkerID] {
			return fmt.Errorf("logical device plan assigns worker %q more than once", worker.WorkerID)
		}
		seenWorkers[worker.WorkerID] = true
	}
	seenPages := make(map[string]bool, len(p.Pages))
	for _, page := range p.Pages {
		if page.PageID == "" || page.WorkerID == "" || page.TierID == "" || page.Bytes == 0 || page.Epoch != p.Epoch {
			return fmt.Errorf("logical device plan contains an invalid page placement")
		}
		if seenPages[page.PageID] && !page.Replica {
			return fmt.Errorf("logical device plan contains duplicate primary page %q", page.PageID)
		}
		seenPages[page.PageID] = true
	}
	return nil
}

type TransferState string

const (
	TransferQueued    TransferState = "queued"
	TransferAdmitted  TransferState = "admitted"
	TransferCopying   TransferState = "copying"
	TransferVerified  TransferState = "verified"
	TransferFailed    TransferState = "failed"
	TransferCancelled TransferState = "cancelled"
)

func CanTransitionTransfer(from, to TransferState) bool {
	switch from {
	case TransferQueued:
		return to == TransferAdmitted || to == TransferCancelled || to == TransferFailed
	case TransferAdmitted:
		return to == TransferCopying || to == TransferCancelled || to == TransferFailed
	case TransferCopying:
		return to == TransferVerified || to == TransferCancelled || to == TransferFailed
	default:
		return false
	}
}

type TransferRequest struct {
	LogicalDeviceRequest
	PageID         string `json:"pageId"`
	TargetWorkerID string `json:"targetWorkerId"`
	TargetTierID   string `json:"targetTierId"`
	ExpectedDigest string `json:"expectedDigest"`
}

type TransferStatus struct {
	TransferID     string        `json:"transferId"`
	PlanID         string        `json:"planId"`
	Epoch          uint64        `json:"epoch"`
	PageID         string        `json:"pageId"`
	SourceWorkerID string        `json:"sourceWorkerId"`
	TargetWorkerID string        `json:"targetWorkerId"`
	TargetTierID   string        `json:"targetTierId"`
	ExpectedDigest string        `json:"expectedDigest"`
	State          TransferState `json:"state"`
	Bytes          uint64        `json:"bytes"`
	Error          string        `json:"error,omitempty"`
}
