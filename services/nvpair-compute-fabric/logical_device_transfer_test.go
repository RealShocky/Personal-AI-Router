// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nvpair-shared/fabricwire"
)

func TestLogicalDeviceManagerTransfersAndPublishesPageOwnership(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "remote page"
	digest := pageDigest([]byte(payload))
	if _, err := store.Put(context.Background(), "page-1", uint64(len(payload)), digest, strings.NewReader(payload)); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	peerStore, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("peer NewPageStore() error = %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		pageID := request.Header.Get("X-PAIR-Page-ID")
		size := request.ContentLength
		metadata, putErr := peerStore.Put(request.Context(), pageID, uint64(size), request.Header.Get("X-PAIR-Page-Digest"), request.Body)
		if putErr != nil {
			http.Error(writer, putErr.Error(), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(metadata)
	}))
	defer server.Close()

	workers := NewManager(time.Minute)
	workers.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cpu-a", NodeID: "host-a", Epoch: 1, State: fabricwire.WorkerReady, Runtime: "pair", Backends: []string{"cpu"}, MemoryFree: 8 << 30}, time.Now())
	workers.AcceptHeartbeat(fabricwire.Heartbeat{WorkerID: "cpu-b", NodeID: "host-b", Endpoint: server.URL, Epoch: 1, State: fabricwire.WorkerReady, Runtime: "pair", Backends: []string{"cpu"}, MemoryFree: 8 << 30}, time.Now())
	logical := NewLogicalDeviceManager(workers)
	plan, err := logical.Plan(fabricwire.LogicalDevicePlanRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "transfer-execution-plan"},
		Runtime:              "pair", ModelDigest: "sha256:model", ShardStrategy: fabricwire.ShardPipeline,
		Providers: []fabricwire.Provider{fabricwire.ProviderCPU}, WorkerGoal: 1,
		Pages: []fabricwire.PageSpec{{PageID: "page-1", Bytes: uint64(len(payload))}},
	})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	request := fabricwire.TransferRequest{
		LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "transfer-execution-1", PlanID: plan.PlanID, Epoch: plan.Epoch},
		PageID:               "page-1", TargetWorkerID: "cpu-b", TargetTierID: "host", ExpectedDigest: digest,
	}
	transfer, err := logical.TransferPage(context.Background(), http.DefaultClient, store, request)
	if err != nil {
		t.Fatalf("TransferPage() error = %v", err)
	}
	if transfer.State != fabricwire.TransferVerified {
		t.Fatalf("TransferPage() = %#v", transfer)
	}
	reader, metadata, err := peerStore.Open(context.Background(), "page-1")
	if err != nil {
		t.Fatalf("peer Open() error = %v", err)
	}
	defer reader.Close()
	got, err := io.ReadAll(reader)
	if err != nil || string(got) != payload || metadata.Digest != digest {
		t.Fatalf("peer page = %q/%#v, err=%v", got, metadata, err)
	}
}
