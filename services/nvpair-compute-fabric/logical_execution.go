// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"nvpair-shared/fabricwire"
)

const LogicalOperationCopy = "copy"

type LogicalExecuteRequest struct {
	Operation    string
	InputPageID  string
	OutputPageID string
}

type LogicalExecuteResult struct {
	Provider string       `json:"provider"`
	Output   PageMetadata `json:"output"`
}

func logicalExecuteRequest(request fabricwire.LogicalExecuteRequest) LogicalExecuteRequest {
	return LogicalExecuteRequest{Operation: request.Operation, InputPageID: request.InputPageID, OutputPageID: request.OutputPageID}
}

type LogicalProvider interface {
	Execute(context.Context, *PageStore, LogicalExecuteRequest) (LogicalExecuteResult, error)
}

type CPUProvider struct{}

type CUDAProvider struct {
	helperPath string
	deviceID   string
}

type MetalProvider struct {
	helperPath string
	deviceID   string
}

func NewCUDAProvider(helperPath, deviceID string) CUDAProvider {
	return CUDAProvider{helperPath: helperPath, deviceID: deviceID}
}

func NewCPUProvider() CPUProvider {
	return CPUProvider{}
}

func NewMetalProvider(helperPath, deviceID string) MetalProvider {
	return MetalProvider{helperPath: helperPath, deviceID: deviceID}
}

func (CPUProvider) Execute(ctx context.Context, store *PageStore, request LogicalExecuteRequest) (LogicalExecuteResult, error) {
	if request.Operation != LogicalOperationCopy {
		return LogicalExecuteResult{}, fmt.Errorf("CPU provider does not support operation %q", request.Operation)
	}
	if store == nil || request.InputPageID == "" || request.OutputPageID == "" {
		return LogicalExecuteResult{}, fmt.Errorf("CPU execution requires a page store and input/output page IDs")
	}
	reader, input, err := store.Open(ctx, request.InputPageID)
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("read input page: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return LogicalExecuteResult{}, err
	}
	output, err := store.Put(ctx, request.OutputPageID, input.Bytes, input.Digest, bytes.NewReader(data))
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	return LogicalExecuteResult{Provider: "cpu", Output: output}, nil
}

func (p CUDAProvider) Execute(ctx context.Context, store *PageStore, request LogicalExecuteRequest) (LogicalExecuteResult, error) {
	return executeStagedPage(ctx, store, request, p.helperPath, p.deviceID, "CUDA")
}

func (p MetalProvider) Execute(ctx context.Context, store *PageStore, request LogicalExecuteRequest) (LogicalExecuteResult, error) {
	return executeStagedPage(ctx, store, request, p.helperPath, p.deviceID, "Metal")
}

func executeStagedPage(ctx context.Context, store *PageStore, request LogicalExecuteRequest, helperPath, deviceID, providerName string) (LogicalExecuteResult, error) {
	if request.Operation != LogicalOperationCopy {
		return LogicalExecuteResult{}, fmt.Errorf("%s provider does not support operation %q", providerName, request.Operation)
	}
	if helperPath == "" {
		return LogicalExecuteResult{}, fmt.Errorf("%s page helper is not configured", providerName)
	}
	if store == nil || request.InputPageID == "" || request.OutputPageID == "" {
		return LogicalExecuteResult{}, fmt.Errorf("%s execution requires a page store and input/output page IDs", providerName)
	}
	reader, input, err := store.Open(ctx, request.InputPageID)
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("read %s input page: %w", providerName, err)
	}
	work, err := os.MkdirTemp("", "pair-page-")
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("create %s staging directory: %w", providerName, err)
	}
	defer os.RemoveAll(work)
	inputPath := filepath.Join(work, "input.page")
	outputPath := filepath.Join(work, "output.page")
	if err := os.WriteFile(inputPath, data, 0o600); err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("stage %s input page: %w", providerName, err)
	}
	if deviceID == "" {
		deviceID = "0"
	}
	command := exec.CommandContext(ctx, helperPath, inputPath, outputPath, deviceID)
	if _, err := command.CombinedOutput(); err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("%s page helper failed: %w", providerName, err)
	}
	if err := ctx.Err(); err != nil {
		return LogicalExecuteResult{}, err
	}
	data, err = os.ReadFile(outputPath)
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("read %s output page: %w", providerName, err)
	}
	if uint64(len(data)) != input.Bytes {
		return LogicalExecuteResult{}, fmt.Errorf("%s output page has %d bytes, expected %d", providerName, len(data), input.Bytes)
	}
	output, err := store.Put(ctx, request.OutputPageID, input.Bytes, input.Digest, bytes.NewReader(data))
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	return LogicalExecuteResult{Provider: strings.ToLower(providerName), Output: output}, nil
}
