// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "strings"

// HardwareCapabilities describes execution backends that a future managed
// runtime may use on this node. These are hardware/runtime candidates, not a
// claim that Ollama, LM Studio, or a distributed engine is installed.
type HardwareCapabilities struct {
	Backends               []string `json:"backends,omitempty"`
	DistributedWorker      bool     `json:"distributed_worker"`
	DistributedCoordinator bool     `json:"distributed_coordinator"`
}

func classifyCapabilities(gpus []GPUInfo) *HardwareCapabilities {
	backends := []string{"cpu"}
	for _, gpu := range gpus {
		backend := gpuBackend(gpu)
		if backend == "cuda" {
			backends = appendUnique(backends, "cuda")
		}
		if backend == "metal" {
			backends = appendUnique(backends, "metal")
		}
	}
	return &HardwareCapabilities{
		Backends:               backends,
		DistributedWorker:      true,
		DistributedCoordinator: true,
	}
}

func annotateGPUBackends(gpus []GPUInfo) []GPUInfo {
	annotated := make([]GPUInfo, len(gpus))
	copy(annotated, gpus)
	for index := range annotated {
		if annotated[index].Backend == "" {
			annotated[index].Backend = gpuBackend(annotated[index])
		}
	}
	return annotated
}

func gpuBackend(gpu GPUInfo) string {
	backend := strings.ToLower(gpu.Backend)
	if backend != "" {
		return backend
	}
	name := strings.ToLower(gpu.Name)
	if strings.Contains(name, "nvidia") {
		return "cuda"
	}
	if strings.Contains(name, "apple") {
		return "metal"
	}
	return ""
}

func appendUnique(values []string, value string) []string {
	if hasCapability(values, value) {
		return values
	}
	return append(values, value)
}

func hasCapability(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
