// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"nvpair-shared/fabricwire"
)

type LogicalDeviceManager struct {
	mu       sync.RWMutex
	workers  *Manager
	plans    map[string]fabricwire.LogicalDevicePlan
	requests map[string]fabricwire.LogicalDevicePlan
}

func NewLogicalDeviceManager(workers *Manager) *LogicalDeviceManager {
	return &LogicalDeviceManager{
		workers:  workers,
		plans:    make(map[string]fabricwire.LogicalDevicePlan),
		requests: make(map[string]fabricwire.LogicalDevicePlan),
	}
}

func (m *LogicalDeviceManager) Plan(request fabricwire.LogicalDevicePlanRequest) (fabricwire.LogicalDevicePlan, error) {
	if request.ProtocolVersion != fabricwire.LogicalDeviceProtocolVersion {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("unsupported logical device protocol version %d", request.ProtocolVersion)
	}
	if request.RequestID == "" || request.Runtime == "" || request.ModelDigest == "" || request.WorkerGoal == 0 {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("logical device plan request requires requestId, runtime, modelDigest, and positive workerGoal")
	}
	if request.ShardStrategy != fabricwire.ShardReplicated && request.ShardStrategy != fabricwire.ShardTensor && request.ShardStrategy != fabricwire.ShardPipeline {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("unsupported shard strategy %q", request.ShardStrategy)
	}
	for _, page := range request.Pages {
		if page.PageID == "" || page.Bytes == 0 {
			return fabricwire.LogicalDevicePlan{}, fmt.Errorf("logical device page requires pageId and positive bytes")
		}
	}

	m.mu.RLock()
	if plan, ok := m.requests[request.RequestID]; ok {
		m.mu.RUnlock()
		return plan, nil
	}
	m.mu.RUnlock()

	wanted := make(map[string]bool, len(request.Providers))
	for _, provider := range request.Providers {
		wanted[string(provider)] = true
	}
	candidates := make([]WorkerRecord, 0)
	for _, worker := range m.workers.Workers() {
		if worker.State != fabricwire.WorkerReady || !runtimeMatches(request.Runtime, worker.Heartbeat) {
			continue
		}
		if len(wanted) > 0 && !hasBackend(worker.Heartbeat.Backends, wanted) {
			continue
		}
		candidates = append(candidates, worker)
	}
	if uint32(len(candidates)) < request.WorkerGoal {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("only %d eligible workers available, need %d", len(candidates), request.WorkerGoal)
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Heartbeat.WorkerID < candidates[j].Heartbeat.WorkerID
	})

	epoch := uint64(time.Now().UnixNano())
	if epoch == 0 {
		epoch = 1
	}
	plan := fabricwire.LogicalDevicePlan{
		Version:       fabricwire.LogicalDeviceProtocolVersion,
		PlanID:        request.RequestID,
		Epoch:         epoch,
		Runtime:       request.Runtime,
		ModelDigest:   request.ModelDigest,
		ShardStrategy: request.ShardStrategy,
		Workers:       make([]fabricwire.WorkerAssignment, 0, request.WorkerGoal),
		Pages:         make([]fabricwire.PagePlacement, 0, len(request.Pages)),
	}
	selected := candidates[:request.WorkerGoal]
	for index, worker := range selected {
		plan.Workers = append(plan.Workers, fabricwire.WorkerAssignment{
			WorkerID:           worker.Heartbeat.WorkerID,
			PeerID:             workerPeerID(worker.Heartbeat),
			Endpoint:           worker.Heartbeat.Endpoint,
			ShardIndex:         uint32(index),
			MemoryBudgetBytes:  worker.Heartbeat.MemoryFree,
			GPUVramBudgetBytes: worker.Heartbeat.GPUVramFree,
		})
	}
	for index, page := range request.Pages {
		worker := selected[index%len(selected)]
		tierID := "host"
		if worker.Heartbeat.GPUCount > 0 && worker.Heartbeat.GPUVramFree >= page.Bytes {
			tierID = "gpu"
		}
		plan.Pages = append(plan.Pages, fabricwire.PagePlacement{
			PageID: page.PageID, WorkerID: worker.Heartbeat.WorkerID, TierID: tierID,
			Bytes: page.Bytes, DType: page.DType, Layout: page.Layout, Epoch: epoch,
		})
	}
	if err := plan.Validate(); err != nil {
		return fabricwire.LogicalDevicePlan{}, err
	}
	m.mu.Lock()
	if existing, ok := m.requests[request.RequestID]; ok {
		m.mu.Unlock()
		return existing, nil
	}
	m.requests[request.RequestID] = plan
	m.plans[plan.PlanID] = plan
	m.mu.Unlock()
	return plan, nil
}
