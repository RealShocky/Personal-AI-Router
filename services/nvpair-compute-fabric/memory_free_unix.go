// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"os"
	"strconv"
	"strings"
)

func systemMemoryFree() uint64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "MemAvailable:" {
			value, err := strconv.ParseUint(fields[1], 10, 64)
			if err == nil {
				return value * 1024
			}
		}
	}
	return 0
}
