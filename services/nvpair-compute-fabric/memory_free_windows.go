// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

type fabricMemoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var globalMemoryStatusEx = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func systemMemoryFree() uint64 {
	status := fabricMemoryStatusEx{Length: uint32(unsafe.Sizeof(fabricMemoryStatusEx{}))}
	result, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&status)))
	if result == 0 {
		return 0
	}
	return status.AvailPhys
}
