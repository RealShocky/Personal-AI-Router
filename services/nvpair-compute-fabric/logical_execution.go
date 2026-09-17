// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"

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
