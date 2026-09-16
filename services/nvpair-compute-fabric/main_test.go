// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net"
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

func TestRPCTargetAvailableReflectsReachability(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for RPC probe: %v", err)
	}
	address := listener.Addr().String()
	if !rpcTargetAvailable(address) {
		t.Fatal("reachable RPC target was reported unavailable")
	}
	if err := listener.Close(); err != nil {
		t.Fatalf("close RPC probe listener: %v", err)
	}
	if rpcTargetAvailable(address) {
		t.Fatal("closed RPC target was reported available")
	}
}

func TestFabricAcceptsSharedLogLevelValues(t *testing.T) {
	for _, level := range []string{"debug", "info", "warn", "error"} {
		if !validLogLevel(level) {
			t.Fatalf("log level %q was rejected", level)
		}
	}
	if validLogLevel("verbose") {
		t.Fatal("unsupported log level was accepted")
	}
}
