// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type gpuMemorySnapshot struct {
	Total uint64
	Free  uint64
	Count uint32
}

func parseNvidiaMemory(output string) (gpuMemorySnapshot, bool) {
	var snapshot gpuMemorySnapshot
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Split(line, ",")
		if len(fields) != 3 {
			if strings.TrimSpace(line) == "" {
				continue
			}
			return gpuMemorySnapshot{}, false
		}
		values := make([]uint64, 3)
		for index, field := range fields {
			value, err := strconv.ParseUint(strings.TrimSpace(field), 10, 64)
			if err != nil {
				return gpuMemorySnapshot{}, false
			}
			values[index] = value * 1024 * 1024
		}
		snapshot.Total += values[1]
		snapshot.Free += values[2]
		snapshot.Count++
	}
	return snapshot, snapshot.Count > 0
}

func readGPUMemory() gpuMemorySnapshot {
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	output, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return gpuMemorySnapshot{}
	}
	snapshot, ok := parseNvidiaMemory(string(output))
	if !ok {
		return gpuMemorySnapshot{}
	}
	return snapshot
}
