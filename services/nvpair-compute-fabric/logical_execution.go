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

func NewCUDAProvider(helperPath, deviceID string) CUDAProvider {
	return CUDAProvider{helperPath: helperPath, deviceID: deviceID}
}

func NewCPUProvider() CPUProvider {
	return CPUProvider{}
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
	if request.Operation != LogicalOperationCopy {
		return LogicalExecuteResult{}, fmt.Errorf("CUDA provider does not support operation %q", request.Operation)
	}
	if p.helperPath == "" {
		return LogicalExecuteResult{}, fmt.Errorf("CUDA page helper is not configured")
	}
	if store == nil || request.InputPageID == "" || request.OutputPageID == "" {
		return LogicalExecuteResult{}, fmt.Errorf("CUDA execution requires a page store and input/output page IDs")
	}
	reader, input, err := store.Open(ctx, request.InputPageID)
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("read CUDA input page: %w", err)
	}
	work, err := os.MkdirTemp("", "pair-cuda-page-")
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("create CUDA staging directory: %w", err)
	}
	defer os.RemoveAll(work)
	inputPath := filepath.Join(work, "input.page")
	outputPath := filepath.Join(work, "output.page")
	if err := os.WriteFile(inputPath, data, 0o600); err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("stage CUDA input page: %w", err)
	}
	deviceID := p.deviceID
	if deviceID == "" {
		deviceID = "0"
	}
	command := exec.CommandContext(ctx, p.helperPath, inputPath, outputPath, deviceID)
	if _, err := command.CombinedOutput(); err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("CUDA page helper failed: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return LogicalExecuteResult{}, err
	}
	data, err = os.ReadFile(outputPath)
	if err != nil {
		return LogicalExecuteResult{}, fmt.Errorf("read CUDA output page: %w", err)
	}
	if uint64(len(data)) != input.Bytes {
		return LogicalExecuteResult{}, fmt.Errorf("CUDA output page has %d bytes, expected %d", len(data), input.Bytes)
	}
	output, err := store.Put(ctx, request.OutputPageID, input.Bytes, input.Digest, bytes.NewReader(data))
	if err != nil {
		return LogicalExecuteResult{}, err
	}
	return LogicalExecuteResult{Provider: "cuda", Output: output}, nil
}
