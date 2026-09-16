// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestLogicalDeviceManagerPlansCPUPageOnReadyWorker(t *testing.T) {
	now := time.Now()
	workers := NewManager(time.Minute)
	workers.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID:   "cpu-a",
		NodeID:     "host-a",
		Epoch:      1,
		State:      fabricwire.WorkerReady,
		Runtime:    "pair",
		Backends:   []string{"cpu"},
		MemoryFree: 8 << 30,
	}, now)
	logical := NewLogicalDeviceManager(workers)

	plan, err := logical.Plan(fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "request-1"},
		Runtime:              "pair",
		ModelDigest:          "sha256:model",
		ShardStrategy:        fabricwire.ShardPipeline,
		Providers:            []fabricwire.Provider{fabricwire.ProviderCPU},
		WorkerGoal:           1,
		Pages:                []fabricwire.PageSpec{{PageID: "page-1", Bytes: 1024}},
	})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(plan.Pages) != 1 || plan.Pages[0].WorkerID != "cpu-a" || plan.Pages[0].Epoch == 0 {
		t.Fatalf("Plan() = %#v, want one page on cpu-a with an epoch", plan)
	}
}

func TestLogicalDeviceManagerRejectsUnhealthyWorker(t *testing.T) {
	now := time.Now()
	workers := NewManager(time.Second)
	workers.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID:   "cpu-a",
		NodeID:     "host-a",
		Epoch:      1,
		State:      fabricwire.WorkerReady,
		Runtime:    "pair",
		Backends:   []string{"cpu"},
		MemoryFree: 8 << 30,
	}, now.Add(-3*time.Second))
	workers.Reconcile(now)
	logical := NewLogicalDeviceManager(workers)

	_, err := logical.Plan(fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "request-2"},
		Runtime:              "pair",
		ModelDigest:          "sha256:model",
		ShardStrategy:        fabricwire.ShardPipeline,
		Providers:            []fabricwire.Provider{fabricwire.ProviderCPU},
		WorkerGoal:           1,
		Pages:                []fabricwire.PageSpec{{PageID: "page-1", Bytes: 1024}},
	})
	if err == nil {
		t.Fatal("Plan() accepted a worker that missed its health deadline")
	}
}

func TestLogicalDeviceManagerReturnsIdempotentPlanAndStatus(t *testing.T) {
	now := time.Now()
	workers := NewManager(time.Minute)
	workers.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID: "cuda-a", NodeID: "host-a", Epoch: 1, State: fabricwire.WorkerReady,
		Runtime: "pair", Backends: []string{"cuda"}, MemoryFree: 8 << 30,
		GPUCount: 1, GPUVramTotal: 12 << 30, GPUVramFree: 10 << 30,
	}, now)
	logical := NewLogicalDeviceManager(workers)
	request := fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "status-request-1"},
		Runtime:              "pair", ModelDigest: "sha256:model", ShardStrategy: fabricwire.ShardPipeline,
		Providers: []fabricwire.Provider{fabricwire.ProviderCUDA}, WorkerGoal: 1,
		Pages: []fabricwire.PageSpec{{PageID: "page-1", Bytes: 1024}},
	}
	first, err := logical.Plan(request)
	if err != nil {
		t.Fatalf("first Plan() error = %v", err)
	}
	second, err := logical.Plan(request)
	if err != nil {
		t.Fatalf("idempotent Plan() error = %v", err)
	}
	if first.PlanID != second.PlanID || first.Epoch != second.Epoch {
		t.Fatalf("idempotent plans differ: first=%#v second=%#v", first, second)
	}
	status, ok := logical.Status(first.PlanID)
	if !ok || status.State != fabricwire.LogicalDevicePlanned || status.Epoch != first.Epoch {
		t.Fatalf("Status() = %#v, %v", status, ok)
	}
	describe := logical.Describe()
	if len(describe.Providers) != 1 || describe.Providers[0].Provider != fabricwire.ProviderCUDA {
		t.Fatalf("Describe() = %#v", describe)
	}
}

func TestLogicalDeviceManagerTransfersPageOnlyAfterDigestVerification(t *testing.T) {
	now := time.Now()
	workers := NewManager(time.Minute)
	for _, id := range []string{"cpu-a", "cpu-b"} {
		workers.AcceptHeartbeat(fabricwire.Heartbeat{
			WorkerID: id, NodeID: id + "-host", Epoch: 1, State: fabricwire.WorkerReady,
			Runtime: "pair", Backends: []string{"cpu"}, MemoryFree: 8 << 30,
		}, now)
	}
	logical := NewLogicalDeviceManager(workers)
	plan, err := logical.Plan(fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "transfer-plan"},
		Runtime:              "pair", ModelDigest: "sha256:model", ShardStrategy: fabricwire.ShardPipeline,
		Providers: []fabricwire.Provider{fabricwire.ProviderCPU}, WorkerGoal: 2,
		Pages: []fabricwire.PageSpec{{PageID: "page-1", Bytes: 1024}},
	})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	transfer, err := logical.Transfer(fabricwire.TransferRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "transfer-1", PlanID: plan.PlanID, Epoch: plan.Epoch},
		PageID:               "page-1", TargetWorkerID: "cpu-b", TargetTierID: "host", ExpectedDigest: "sha256:page",
	})
	if err != nil {
		t.Fatalf("Transfer() error = %v", err)
	}
	if transfer.State != fabricwire.TransferQueued {
		t.Fatalf("Transfer() state = %q, want queued", transfer.State)
	}
	if _, err := logical.CompleteTransfer(transfer.TransferID, "sha256:wrong"); err == nil {
		t.Fatal("CompleteTransfer() accepted a digest mismatch")
	}
	transfer, err = logical.Transfer(fabricwire.TransferRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "transfer-2", PlanID: plan.PlanID, Epoch: plan.Epoch},
		PageID:               "page-1", TargetWorkerID: "cpu-b", TargetTierID: "host", ExpectedDigest: "sha256:page",
	})
	if err != nil {
		t.Fatalf("retry Transfer() error = %v", err)
	}
	completed, err := logical.CompleteTransfer(transfer.TransferID, "sha256:page")
	if err != nil {
		t.Fatalf("CompleteTransfer() error = %v", err)
	}
	if completed.State != fabricwire.TransferVerified || completed.TargetWorkerID != "cpu-b" {
		t.Fatalf("completed transfer = %#v", completed)
	}
	status, ok := logical.Status(plan.PlanID)
	if !ok || len(status.Pages) != 1 || status.Pages[0].WorkerID != "cpu-b" {
		t.Fatalf("Status() after transfer = %#v, %v", status, ok)
	}
}
