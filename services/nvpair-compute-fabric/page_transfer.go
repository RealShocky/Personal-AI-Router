// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"nvpair-shared/fabricwire"
)

type logicalRemoteExecuteRequest struct {
	Plan    fabricwire.LogicalDevicePlan     `json:"plan"`
	Request fabricwire.LogicalExecuteRequest `json:"request"`
}

func sendPage(ctx context.Context, client *http.Client, source *PageStore, targetBaseURL, pageID string) (PageMetadata, error) {
	if client == nil {
		return PageMetadata{}, fmt.Errorf("page transfer requires an HTTP client")
	}
	reader, metadata, err := source.Open(ctx, pageID)
	if err != nil {
		return PageMetadata{}, err
	}
	defer reader.Close()
	endpoint := strings.TrimRight(targetBaseURL, "/") + "/v1/fabric/logical-page"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, reader)
	if err != nil {
		return PageMetadata{}, fmt.Errorf("create page transfer request: %w", err)
	}
	request.ContentLength = int64(metadata.Bytes)
	request.Header.Set("Content-Type", "application/octet-stream")
	request.Header.Set("X-PAIR-Page-ID", metadata.PageID)
	request.Header.Set("X-PAIR-Page-Bytes", fmt.Sprintf("%d", metadata.Bytes))
	request.Header.Set("X-PAIR-Page-Digest", metadata.Digest)
	response, err := client.Do(request)
	if err != nil {
		return PageMetadata{}, fmt.Errorf("send page %q: %w", pageID, err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return PageMetadata{}, fmt.Errorf("page transfer returned HTTP %d", response.StatusCode)
	}
	var peer PageMetadata
	if err := json.NewDecoder(io.LimitReader(response.Body, 64*1024)).Decode(&peer); err != nil {
		return PageMetadata{}, fmt.Errorf("decode page transfer response: %w", err)
	}
	if peer != metadata {
		return PageMetadata{}, fmt.Errorf("peer page metadata did not match source")
	}
	return peer, nil
}

func executeRemoteLogicalPage(ctx context.Context, client *http.Client, targetBaseURL string, plan fabricwire.LogicalDevicePlan, request fabricwire.LogicalExecuteRequest) (fabricwire.LogicalExecuteResult, error) {
	body, err := json.Marshal(logicalRemoteExecuteRequest{Plan: plan, Request: request})
	if err != nil {
		return fabricwire.LogicalExecuteResult{}, fmt.Errorf("encode remote logical execution request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(targetBaseURL, "/")+"/v1/fabric/logical-page/execute", bytes.NewReader(body))
	if err != nil {
		return fabricwire.LogicalExecuteResult{}, fmt.Errorf("create remote logical execution request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	response, err := client.Do(httpRequest)
	if err != nil {
		return fabricwire.LogicalExecuteResult{}, fmt.Errorf("remote logical execution: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return fabricwire.LogicalExecuteResult{}, fmt.Errorf("remote logical execution returned HTTP %d", response.StatusCode)
	}
	var result fabricwire.LogicalExecuteResult
	if err := json.NewDecoder(io.LimitReader(response.Body, 64*1024)).Decode(&result); err != nil {
		return fabricwire.LogicalExecuteResult{}, fmt.Errorf("decode remote logical execution response: %w", err)
	}
	return result, nil
}

func receivePage(ctx context.Context, client *http.Client, target *PageStore, targetBaseURL, pageID string, expected fabricwire.LogicalPageMetadata) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(targetBaseURL, "/")+"/v1/fabric/logical-page?pageId="+url.QueryEscape(pageID), nil)
	if err != nil {
		return fmt.Errorf("create page receive request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("receive page %q: %w", pageID, err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("page receive returned HTTP %d", response.StatusCode)
	}
	metadata, err := target.Put(ctx, pageID, expected.Bytes, expected.Digest, response.Body)
	if err != nil {
		return err
	}
	if metadata.PageID != expected.PageID || metadata.Bytes != expected.Bytes || metadata.Digest != expected.Digest {
		return fmt.Errorf("received page metadata did not match remote execution result")
	}
	return nil
}
