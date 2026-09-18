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

	"nvpair-shared/fabricwire"
)

func TestSendPageStreamsAndVerifiesPeerMetadata(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "distributed page"
	digest := pageDigest([]byte(payload))
	if _, err := store.Put(context.Background(), "page-1", uint64(len(payload)), digest, strings.NewReader(payload)); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.Header.Get("X-PAIR-Page-ID") != "page-1" || request.Header.Get("X-PAIR-Page-Digest") != digest {
			http.Error(writer, "invalid page headers", http.StatusBadRequest)
			return
		}
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil || string(body) != payload {
			http.Error(writer, "invalid page body", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(PageMetadata{PageID: "page-1", Bytes: uint64(len(payload)), Digest: digest})
	}))
	defer server.Close()

	metadata, err := sendPage(context.Background(), http.DefaultClient, store, server.URL, "page-1")
	if err != nil {
		t.Fatalf("sendPage() error = %v", err)
	}
	if metadata.PageID != "page-1" || metadata.Digest != digest || metadata.Bytes != uint64(len(payload)) {
		t.Fatalf("sendPage() metadata = %#v", metadata)
	}
}

func TestRemoteLogicalPageExecutionRoundTrip(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "remote execution"
	digest := pageDigest([]byte(payload))
	plan := fabricwire.LogicalDevicePlan{Version: fabricwire.LogicalDeviceProtocolVersion, PlanID: "plan", Epoch: 7, Runtime: "pair", ModelDigest: "sha256:model", ShardStrategy: fabricwire.ShardPipeline, Workers: []fabricwire.WorkerAssignment{{WorkerID: "cuda-b", MemoryBudgetBytes: 1}}, Pages: []fabricwire.PagePlacement{{PageID: "out", WorkerID: "cuda-b", TierID: "host", Bytes: uint64(len(payload)), Digest: digest, Epoch: 7}}}
	request := fabricwire.LogicalExecuteRequest{LogicalDeviceRequest: fabricwire.LogicalDeviceRequest{ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion, RequestID: "request", PlanID: "plan", Epoch: 7}, Provider: fabricwire.ProviderCUDA, Operation: LogicalOperationCopy, InputPageID: "in", OutputPageID: "out"}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, httpRequest *http.Request) {
		switch httpRequest.Method {
		case http.MethodPost:
			var envelope logicalRemoteExecuteRequest
			if err := json.NewDecoder(httpRequest.Body).Decode(&envelope); err != nil || envelope.Plan.PlanID != plan.PlanID || envelope.Request.RequestID != request.RequestID {
				http.Error(writer, "invalid execution envelope", http.StatusBadRequest)
				return
			}
			_ = json.NewEncoder(writer).Encode(fabricwire.LogicalExecuteResult{Provider: fabricwire.ProviderCUDA, Output: fabricwire.LogicalPageMetadata{PageID: "out", Bytes: uint64(len(payload)), Digest: digest}})
		case http.MethodGet:
			_, _ = writer.Write([]byte(payload))
		default:
			writer.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	result, err := executeRemoteLogicalPage(context.Background(), http.DefaultClient, server.URL, plan, request)
	if err != nil {
		t.Fatalf("executeRemoteLogicalPage() error = %v", err)
	}
	if err := receivePage(context.Background(), http.DefaultClient, store, server.URL, result.Output.PageID, result.Output); err != nil {
		t.Fatalf("receivePage() error = %v", err)
	}
	reader, metadata, err := store.Open(context.Background(), "out")
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil || string(data) != payload || metadata.Digest != digest {
		t.Fatalf("received page = %q %#v, want verified payload", string(data), metadata)
	}
}

func TestSendPageRejectsPeerDigestMismatch(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "page"
	digest := pageDigest([]byte(payload))
	if _, err := store.Put(context.Background(), "page-1", uint64(len(payload)), digest, strings.NewReader(payload)); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		_, _ = io.Copy(io.Discard, request.Body)
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(PageMetadata{PageID: "page-1", Bytes: uint64(len(payload)), Digest: "sha256:wrong"})
	}))
	defer server.Close()

	if _, err := sendPage(context.Background(), http.DefaultClient, store, server.URL, "page-1"); err == nil {
		t.Fatal("sendPage() accepted a peer digest mismatch")
	}
}
