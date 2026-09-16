// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestLogicalDevicePlanRPCReturnsPlan(t *testing.T) {
	workers := NewManager(time.Minute)
	workers.AcceptHeartbeat(fabricwire.Heartbeat{
		WorkerID: "cpu-a", NodeID: "host-a", Epoch: 1, State: fabricwire.WorkerReady,
		Runtime: "pair", Backends: []string{"cpu"}, MemoryFree: 8 << 30,
	}, time.Now())
	logical := NewLogicalDeviceManager(workers)
	request := fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "rpc-request-1"},
		Runtime:              "pair",
		ModelDigest:          "sha256:model",
		ShardStrategy:        fabricwire.ShardPipeline,
		Providers:            []fabricwire.Provider{fabricwire.ProviderCPU},
		WorkerGoal:           1,
		Pages:                []fabricwire.PageSpec{{PageID: "page-1", Bytes: 1024}},
	}
	params, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	id := json.RawMessage(`1`)
	var output bytes.Buffer
	codec := NewCodec(&output)
	handleMessage(codec, workers, nil, nil, nil, logical, context.CancelFunc(func() {}), &Message{JSONRPC: "2.0", ID: &id, Method: "fabric:logical-device-plan", Params: params})

	var response Message
	if err := json.Unmarshal(output.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error != nil {
		t.Fatalf("RPC returned error: %#v", response.Error)
	}
	var plan fabricwire.LogicalDevicePlan
	if err := json.Unmarshal(response.Result, &plan); err != nil {
		t.Fatalf("decode plan: %v", err)
	}
	if plan.PlanID != "rpc-request-1" || len(plan.Pages) != 1 {
		t.Fatalf("RPC plan = %#v", plan)
	}
}
