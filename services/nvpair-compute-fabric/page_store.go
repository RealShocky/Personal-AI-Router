// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type PageMetadata struct {
	PageID string `json:"pageId"`
	Bytes  uint64 `json:"bytes"`
	Digest string `json:"digest"`
}

type PageStore struct {
	root        string
	maxPageSize uint64
}

func NewPageStore(root string, maxPageSize uint64) (*PageStore, error) {
	if strings.TrimSpace(root) == "" {
		return nil, fmt.Errorf("page store requires a root directory")
	}
	if maxPageSize == 0 {
		return nil, fmt.Errorf("page store requires a positive maximum page size")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, fmt.Errorf("create page store: %w", err)
	}
	return &PageStore{root: root, maxPageSize: maxPageSize}, nil
}

func (s *PageStore) Put(ctx context.Context, pageID string, size uint64, expectedDigest string, source io.Reader) (PageMetadata, error) {
	if err := validatePageID(pageID); err != nil {
		return PageMetadata{}, err
	}
	if size == 0 || size > s.maxPageSize {
		return PageMetadata{}, fmt.Errorf("page %q size %d exceeds store limit %d", pageID, size, s.maxPageSize)
	}
	if expectedDigest == "" || source == nil {
		return PageMetadata{}, fmt.Errorf("page %q requires a digest and source", pageID)
	}
	if err := ctx.Err(); err != nil {
		return PageMetadata{}, err
	}
	dataPath, metadataPath := s.paths(pageID)
	temporary, err := os.CreateTemp(s.root, ".pair-page-*")
	if err != nil {
		return PageMetadata{}, fmt.Errorf("create page temporary file: %w", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	hasher := sha256.New()
	limited := io.LimitReader(source, int64(size)+1)
	written, copyErr := io.Copy(io.MultiWriter(temporary, hasher), limited)
	closeErr := temporary.Close()
	if copyErr != nil {
		return PageMetadata{}, fmt.Errorf("write page %q: %w", pageID, copyErr)
	}
	if closeErr != nil {
		return PageMetadata{}, fmt.Errorf("close page %q: %w", pageID, closeErr)
	}
	if err := ctx.Err(); err != nil {
		return PageMetadata{}, err
	}
	if uint64(written) != size {
		return PageMetadata{}, fmt.Errorf("page %q has %d bytes, expected %d", pageID, written, size)
	}
	actualDigest := "sha256:" + hex.EncodeToString(hasher.Sum(nil))
	if actualDigest != expectedDigest {
		return PageMetadata{}, fmt.Errorf("page %q digest mismatch", pageID)
	}
	metadata := PageMetadata{PageID: pageID, Bytes: size, Digest: actualDigest}
	metadataTemporary, err := os.CreateTemp(s.root, ".pair-page-meta-*")
	if err != nil {
		return PageMetadata{}, fmt.Errorf("create page metadata temporary file: %w", err)
	}
	metadataTemporaryPath := metadataTemporary.Name()
	defer os.Remove(metadataTemporaryPath)
	encoded, err := json.Marshal(metadata)
	if err != nil {
		_ = metadataTemporary.Close()
		return PageMetadata{}, fmt.Errorf("encode page metadata: %w", err)
	}
	if _, err := metadataTemporary.Write(encoded); err != nil {
		_ = metadataTemporary.Close()
		return PageMetadata{}, fmt.Errorf("write page metadata: %w", err)
	}
	if err := metadataTemporary.Close(); err != nil {
		return PageMetadata{}, fmt.Errorf("close page metadata: %w", err)
	}
	if err := os.Rename(temporaryPath, dataPath); err != nil {
		return PageMetadata{}, fmt.Errorf("publish page %q: %w", pageID, err)
	}
	if err := os.Rename(metadataTemporaryPath, metadataPath); err != nil {
		_ = os.Remove(dataPath)
		return PageMetadata{}, fmt.Errorf("publish page metadata %q: %w", pageID, err)
	}
	return metadata, nil
}

func (s *PageStore) Open(ctx context.Context, pageID string) (io.ReadCloser, PageMetadata, error) {
	if err := validatePageID(pageID); err != nil {
		return nil, PageMetadata{}, err
	}
	if err := ctx.Err(); err != nil {
		return nil, PageMetadata{}, err
	}
	_, metadataPath := s.paths(pageID)
	encoded, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, PageMetadata{}, fmt.Errorf("read page metadata: %w", err)
	}
	var metadata PageMetadata
	if err := json.Unmarshal(encoded, &metadata); err != nil {
		return nil, PageMetadata{}, fmt.Errorf("decode page metadata: %w", err)
	}
	if metadata.PageID != pageID || metadata.Bytes == 0 || metadata.Bytes > s.maxPageSize || metadata.Digest == "" {
		return nil, PageMetadata{}, fmt.Errorf("invalid metadata for page %q", pageID)
	}
	dataPath, _ := s.paths(pageID)
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, PageMetadata{}, fmt.Errorf("open page: %w", err)
	}
	info, err := file.Stat()
	if err != nil || uint64(info.Size()) != metadata.Bytes {
		_ = file.Close()
		if err != nil {
			return nil, PageMetadata{}, fmt.Errorf("stat page: %w", err)
		}
		return nil, PageMetadata{}, fmt.Errorf("page %q size does not match metadata", pageID)
	}
	return file, metadata, nil
}

func (s *PageStore) paths(pageID string) (string, string) {
	key := sha256.Sum256([]byte(pageID))
	name := hex.EncodeToString(key[:])
	return filepath.Join(s.root, name+".page"), filepath.Join(s.root, name+".json")
}

func pageDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func validatePageID(pageID string) error {
	if pageID == "" || filepath.IsAbs(pageID) || strings.ContainsRune(pageID, '\x00') || strings.Contains(pageID, "\\") {
		return fmt.Errorf("unsafe page ID")
	}
	for _, part := range strings.Split(pageID, "/") {
		if part == "" || part == "." || part == ".." {
			return fmt.Errorf("unsafe page ID")
		}
	}
	return nil
}
