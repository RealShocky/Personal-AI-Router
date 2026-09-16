// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestChildEnvironmentAddsExecutableLibraryDirectoryOnUnix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows uses its native DLL search path")
	}
	const executable = "/opt/llama-b8487/bin/rpc-server"
	values := childEnvironment(executable)
	variable := "LD_LIBRARY_PATH"
	if runtime.GOOS == "darwin" {
		variable = "DYLD_LIBRARY_PATH"
	}
	prefix := variable + "="
	for _, value := range values {
		if strings.HasPrefix(value, prefix) {
			if !strings.Contains(value, "/opt/llama-b8487/bin") {
				t.Fatalf("%s = %q, want executable directory", variable, value)
			}
			return
		}
	}
	t.Fatalf("%s missing from child environment (PATH=%q)", variable, os.Getenv(variable))
}
