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
