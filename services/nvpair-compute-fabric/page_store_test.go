// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
)

func TestPageStorePublishesVerifiedPageAtomically(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "PAIR page payload"
	digest := pageDigest([]byte(payload))
	metadata, err := store.Put(context.Background(), "weights/0", uint64(len(payload)), digest, strings.NewReader(payload))
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if metadata.PageID != "weights/0" || metadata.Bytes != uint64(len(payload)) || metadata.Digest != digest {
		t.Fatalf("Put() metadata = %#v", metadata)
	}
	reader, got, err := store.Open(context.Background(), "weights/0")
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if !bytes.Equal(data, []byte(payload)) || got != metadata {
		t.Fatalf("Open() data/metadata = %q/%#v", data, got)
	}
}

func TestPageStoreRejectsDigestMismatchWithoutPublishing(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	_, err = store.Put(context.Background(), "page-1", 4, "sha256:wrong", strings.NewReader("data"))
	if err == nil {
		t.Fatal("Put() accepted a digest mismatch")
	}
	if _, _, openErr := store.Open(context.Background(), "page-1"); openErr == nil {
		t.Fatal("mismatched page was published")
	}
}

func TestPageStoreRejectsUnsafeOrOversizedPage(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 4)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	for _, pageID := range []string{"../escape", "", "a\\b"} {
		if _, err := store.Put(context.Background(), pageID, 1, pageDigest([]byte("x")), strings.NewReader("x")); err == nil {
			t.Fatalf("Put() accepted unsafe page ID %q", pageID)
		}
	}
	if _, err := store.Put(context.Background(), "page-2", 5, pageDigest([]byte("12345")), strings.NewReader("12345")); err == nil {
		t.Fatal("Put() accepted page larger than store limit")
	}
}
