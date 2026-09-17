// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"io"
	"strings"
	"testing"
)

func TestCPUProviderExecutesPageCopyWithVerifiedOutput(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	const payload = "CPU logical execution"
	inputDigest := pageDigest([]byte(payload))
	if _, err := store.Put(context.Background(), "input", uint64(len(payload)), inputDigest, strings.NewReader(payload)); err != nil {
		t.Fatalf("Put(input) error = %v", err)
	}
	result, err := NewCPUProvider().Execute(context.Background(), store, LogicalExecuteRequest{
		Operation:    LogicalOperationCopy,
		InputPageID:  "input",
		OutputPageID: "output",
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if result.Provider != "cpu" || result.Output.PageID != "output" || result.Output.Digest != inputDigest {
		t.Fatalf("Execute() result = %#v", result)
	}
	reader, _, err := store.Open(context.Background(), "output")
	if err != nil {
		t.Fatalf("Open(output) error = %v", err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil || string(data) != payload {
		t.Fatalf("output = %q, err = %v", data, err)
	}
}

func TestCPUProviderRejectsUnknownOperation(t *testing.T) {
	store, err := NewPageStore(t.TempDir(), 1024)
	if err != nil {
		t.Fatalf("NewPageStore() error = %v", err)
	}
	if _, err := NewCPUProvider().Execute(context.Background(), store, LogicalExecuteRequest{Operation: "remote-pointer", InputPageID: "input", OutputPageID: "output"}); err == nil {
		t.Fatal("Execute() accepted an unsupported operation")
	}
}
