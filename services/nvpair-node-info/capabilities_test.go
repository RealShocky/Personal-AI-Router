// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestClassifyCapabilitiesCPUOnly(t *testing.T) {
	got := classifyCapabilities(nil)
	if got == nil {
		t.Fatal("capabilities = nil")
	}
	if len(got.Backends) != 1 || got.Backends[0] != "cpu" {
		t.Fatalf("backends = %v, want [cpu]", got.Backends)
	}
	if !got.DistributedWorker || !got.DistributedCoordinator {
		t.Fatalf("distributed capabilities = %+v, want worker and coordinator", got)
	}
}

func TestClassifyCapabilitiesNvidia(t *testing.T) {
	got := classifyCapabilities([]GPUInfo{{Name: "NVIDIA GeForce RTX 5060"}})
	if !testContainsString(got.Backends, "cuda") {
		t.Fatalf("backends = %v, want cuda", got.Backends)
	}
}

func TestClassifyCapabilitiesApple(t *testing.T) {
	got := classifyCapabilities([]GPUInfo{{Name: "Apple M3 Max"}})
	if !testContainsString(got.Backends, "metal") {
		t.Fatalf("backends = %v, want metal", got.Backends)
	}
}

func TestAnnotateGPUBackends(t *testing.T) {
	got := annotateGPUBackends([]GPUInfo{
		{Name: "NVIDIA GeForce RTX 5060"},
		{Name: "Apple M3 Max"},
		{Name: "Unknown Display Adapter"},
	})
	if got[0].Backend != "cuda" || got[1].Backend != "metal" || got[2].Backend != "" {
		t.Fatalf("annotated backends = %+v", got)
	}
}

func testContainsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
