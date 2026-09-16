// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fabricwire

import "testing"

func TestLogicalDevicePlanRejectsDuplicatePages(t *testing.T) {
	plan := LogicalDevicePlan{
		Version:       LogicalDeviceProtocolVersion,
		PlanID:        "plan-1",
		Epoch:         1,
		Runtime:       "pair",
		ModelDigest:   "sha256:model",
		ShardStrategy: ShardPipeline,
		Pages: []PagePlacement{
			{PageID: "page-1", WorkerID: "worker-a", TierID: "gpu-0", Bytes: 1024, Epoch: 1},
			{PageID: "page-1", WorkerID: "worker-b", TierID: "host-ram", Bytes: 1024, Epoch: 1},
		},
	}

	if err := plan.Validate(); err == nil {
		t.Fatal("Validate() accepted duplicate page ownership")
	}
}

func TestLogicalDevicePlanRejectsStalePageEpoch(t *testing.T) {
	plan := validLogicalDevicePlan()
	plan.Pages[0].Epoch = 0

	if err := plan.Validate(); err == nil {
		t.Fatal("Validate() accepted a page from an older epoch")
	}
}

func TestLogicalDeviceTransferAllowsOnlyForwardStates(t *testing.T) {
	if !CanTransitionTransfer(TransferQueued, TransferAdmitted) {
		t.Fatal("queued transfer should be admissible")
	}
	if CanTransitionTransfer(TransferVerified, TransferCopying) {
		t.Fatal("verified transfer must not return to copying")
	}
}

func TestLogicalDeviceRequestRejectsStaleEpoch(t *testing.T) {
	request := LogicalDeviceRequest{ProtocolVersion: LogicalDeviceProtocolVersion, RequestID: "request-1", PlanID: "plan-1", Epoch: 1}
	if err := request.ValidateForEpoch(2); err == nil {
		t.Fatal("request from an older epoch was accepted")
	}
}

func validLogicalDevicePlan() LogicalDevicePlan {
	return LogicalDevicePlan{
		Version:       LogicalDeviceProtocolVersion,
		PlanID:        "plan-1",
		Epoch:         1,
		Runtime:       "pair",
		ModelDigest:   "sha256:model",
		ShardStrategy: ShardPipeline,
		Workers: []WorkerAssignment{
			{WorkerID: "worker-a", ShardIndex: 0, MemoryBudgetBytes: 1024},
		},
		Pages: []PagePlacement{
			{PageID: "page-1", WorkerID: "worker-a", TierID: "gpu-0", Bytes: 1024, Epoch: 1},
		},
	}
}
