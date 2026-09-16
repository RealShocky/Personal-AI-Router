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
