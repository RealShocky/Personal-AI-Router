// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestParseNvidiaMemoryAggregatesAdapters(t *testing.T) {
	got, ok := parseNvidiaMemory("1024,8192,4096\n2048,16384,9904\n")
	if !ok {
		t.Fatal("parseNvidiaMemory reported failure")
	}
	if got.Total != 24*1024*1024*1024 || got.Free != 14000*1024*1024 || got.Count != 2 {
		t.Fatalf("memory snapshot = %+v, want total=24576MiB free=14000MiB count=2", got)
	}
}

func TestParseNvidiaMemoryRejectsMalformedRows(t *testing.T) {
	if _, ok := parseNvidiaMemory("1024,not-a-number,4096\n"); ok {
		t.Fatal("malformed nvidia-smi row was accepted")
	}
}
