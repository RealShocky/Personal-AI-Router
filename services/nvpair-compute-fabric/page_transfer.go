// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

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
