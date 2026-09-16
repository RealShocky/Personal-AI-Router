// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package fabricwire contains the versioned, runtime-neutral contract used by
// PAIR's distributed compute fabric. It deliberately describes leases and
// epochs rather than tensor operations; those belong to a runtime adapter.
package fabricwire

type WorkerState string

const (
	WorkerDiscovered    WorkerState = "discovered"
	WorkerAuthenticated WorkerState = "authenticated"
	WorkerProbing       WorkerState = "probing"
	WorkerReady         WorkerState = "ready"
	WorkerLeased        WorkerState = "leased"
	WorkerServing       WorkerState = "serving"
	WorkerSuspect       WorkerState = "suspect"
	WorkerQuarantined   WorkerState = "quarantined"
)

func CanTransition(from, to WorkerState) bool {
	switch from {
	case WorkerDiscovered:
		return to == WorkerAuthenticated
	case WorkerAuthenticated:
		return to == WorkerProbing
	case WorkerProbing:
		return to == WorkerReady
	case WorkerReady:
		return to == WorkerLeased
	case WorkerLeased:
		return to == WorkerServing
	case WorkerServing:
		return to == WorkerSuspect
	case WorkerSuspect:
		return to == WorkerQuarantined
	case WorkerQuarantined:
		return to == WorkerProbing
	default:
		return false
	}
}

type Heartbeat struct {
	WorkerID           string      `json:"workerId"`
	NodeID             string      `json:"nodeId"`
	Endpoint           string      `json:"endpoint,omitempty"`
	State              WorkerState `json:"state"`
	Epoch              uint64      `json:"epoch"`
	Runtime            string      `json:"runtime"`
	Backends           []string    `json:"backends"`
	MemoryFree         uint64      `json:"memoryFreeBytes"`
	GPUVramTotal       uint64      `json:"gpuVramTotalBytes,omitempty"`
	GPUVramFree        uint64      `json:"gpuVramFreeBytes,omitempty"`
	GPUCount           uint32      `json:"gpuCount,omitempty"`
	QueueDepth         uint32      `json:"queueDepth"`
	ModelDigests       []string    `json:"modelDigests,omitempty"`
	CheckpointSupport  bool        `json:"checkpointSupport,omitempty"`
	ProbeLatencyMillis uint64      `json:"probeLatencyMillis,omitempty"`
}

// GroupRequest describes an explicit runtime-level placement request. It does
// not pretend that the operating system has a pooled device; the adapter uses
// the resulting plan to place model work across these worker identities.
type GroupRequest struct {
	GroupID               string   `json:"groupId"`
	Runtime               string   `json:"runtime"`
	Backends              []string `json:"backends"`
	WorkerGoal            uint32   `json:"workerGoal"`
	AllowMixed            bool     `json:"allowMixed"`
	ModelDigest           string   `json:"modelDigest,omitempty"`
	RequireCheckpoint     bool     `json:"requireCheckpoint,omitempty"`
	MaxProbeLatencyMillis uint64   `json:"maxProbeLatencyMillis,omitempty"`
	MinMemoryFreeBytes    uint64   `json:"minMemoryFreeBytes,omitempty"`
}

type GroupPlan struct {
	GroupID   string            `json:"groupId"`
	Runtime   string            `json:"runtime"`
	Workers   []string          `json:"workers"`
	Endpoints map[string]string `json:"endpoints,omitempty"`
	Epoch     uint64            `json:"epoch"`
}

type Checkpoint struct {
	JobID    string `json:"jobId"`
	GroupID  string `json:"groupId"`
	Epoch    uint64 `json:"epoch"`
	Stage    uint32 `json:"stage"`
	SlotID   int    `json:"slotId,omitempty"`
	Filename string `json:"filename,omitempty"`
}

func (c Checkpoint) AcceptsEpoch(epoch uint64) bool {
	return c.Epoch == epoch
}
