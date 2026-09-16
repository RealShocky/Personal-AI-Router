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
	mu        sync.RWMutex
	workers   *Manager
	plans     map[string]fabricwire.LogicalDevicePlan
	requests  map[string]fabricwire.LogicalDevicePlan
	states    map[string]fabricwire.LogicalDeviceState
	transfers map[string]fabricwire.TransferStatus
}

func NewLogicalDeviceManager(workers *Manager) *LogicalDeviceManager {
	return &LogicalDeviceManager{
		workers:   workers,
		plans:     make(map[string]fabricwire.LogicalDevicePlan),
		requests:  make(map[string]fabricwire.LogicalDevicePlan),
		states:    make(map[string]fabricwire.LogicalDeviceState),
		transfers: make(map[string]fabricwire.TransferStatus),
	}
}

func (m *LogicalDeviceManager) Describe() fabricwire.LogicalDeviceDescribe {
	describe := fabricwire.LogicalDeviceDescribe{
		ProtocolVersion: fabricwire.LogicalDeviceProtocolVersion,
		WorkerID:        "coordinator",
		Providers:       make([]fabricwire.ProviderCapability, 0),
	}
	workers := m.workers.Workers()
	sort.Slice(workers, func(i, j int) bool {
		return workers[i].Heartbeat.WorkerID < workers[j].Heartbeat.WorkerID
	})
	for _, worker := range workers {
		for _, backend := range worker.Heartbeat.Backends {
			provider := fabricwire.Provider(backend)
			if provider != fabricwire.ProviderCPU && provider != fabricwire.ProviderCUDA && provider != fabricwire.ProviderMetal {
				continue
			}
			tiers := []fabricwire.MemoryTier{{TierID: "host", Kind: fabricwire.MemoryTierHost, CapacityBytes: worker.Heartbeat.MemoryFree, FreeBytes: worker.Heartbeat.MemoryFree, Local: true}}
			if provider == fabricwire.ProviderCUDA || provider == fabricwire.ProviderMetal {
				tiers = append(tiers, fabricwire.MemoryTier{TierID: "gpu", Kind: fabricwire.MemoryTierGPU, CapacityBytes: worker.Heartbeat.GPUVramTotal, FreeBytes: worker.Heartbeat.GPUVramFree, Local: true})
			}
			describe.Providers = append(describe.Providers, fabricwire.ProviderCapability{
				Provider: provider, DeviceID: worker.Heartbeat.WorkerID, SupportsExecution: worker.State == fabricwire.WorkerReady,
				MemoryTiers: tiers,
			})
		}
	}
	return describe
}

func (m *LogicalDeviceManager) Status(planID string) (fabricwire.LogicalDeviceStatus, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	plan, ok := m.plans[planID]
	state := m.states[planID]
	if !ok {
		return fabricwire.LogicalDeviceStatus{}, false
	}
	if state == "" {
		state = fabricwire.LogicalDevicePlanned
	}
	workers := make([]string, 0, len(plan.Workers))
	for _, worker := range plan.Workers {
		workers = append(workers, worker.WorkerID)
	}
	transferIDs := make([]string, 0)
	for id, transfer := range m.transfers {
		if transfer.PlanID == plan.PlanID {
			transferIDs = append(transferIDs, id)
		}
	}
	sort.Strings(transferIDs)
	return fabricwire.LogicalDeviceStatus{PlanID: plan.PlanID, Epoch: plan.Epoch, State: state, Workers: workers, Pages: append([]fabricwire.PagePlacement(nil), plan.Pages...), TransferIDs: transferIDs, UpdatedAtMS: time.Now().UnixMilli()}, true
}

func (m *LogicalDeviceManager) Reconcile(_ ...time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for planID, plan := range m.plans {
		state := m.states[planID]
		if state == fabricwire.LogicalDeviceCancelled || state == fabricwire.LogicalDeviceCompleted {
			continue
		}
		for _, worker := range plan.Workers {
			if !m.workers.Assignable(worker.WorkerID) {
				m.states[planID] = fabricwire.LogicalDeviceDegraded
				break
			}
		}
	}
}

func (m *LogicalDeviceManager) Recover(planID string, replacementWorkerIDs []string) (fabricwire.LogicalDevicePlan, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	plan, ok := m.plans[planID]
	if !ok {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("logical device plan %q not found", planID)
	}
	if m.states[planID] != fabricwire.LogicalDeviceDegraded {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("logical device plan %q is not degraded", planID)
	}
	if len(replacementWorkerIDs) != len(plan.Workers) {
		return fabricwire.LogicalDevicePlan{}, fmt.Errorf("recovery requires %d replacement workers", len(plan.Workers))
	}
	seen := make(map[string]bool, len(replacementWorkerIDs))
	replacements := make([]WorkerRecord, 0, len(replacementWorkerIDs))
	for _, workerID := range replacementWorkerIDs {
		if seen[workerID] {
			return fabricwire.LogicalDevicePlan{}, fmt.Errorf("recovery repeats worker %q", workerID)
		}
		seen[workerID] = true
		worker, ready := m.workers.Worker(workerID)
		if !ready || worker.State != fabricwire.WorkerReady {
			return fabricwire.LogicalDevicePlan{}, fmt.Errorf("replacement worker %q is not ready", workerID)
		}
		replacements = append(replacements, worker)
	}
	epoch := uint64(time.Now().UnixNano())
	if epoch <= plan.Epoch {
		epoch = plan.Epoch + 1
	}
	plan.Epoch = epoch
	plan.Workers = make([]fabricwire.WorkerAssignment, 0, len(replacements))
	for index, worker := range replacements {
		plan.Workers = append(plan.Workers, fabricwire.WorkerAssignment{
			WorkerID: worker.Heartbeat.WorkerID, PeerID: workerPeerID(worker.Heartbeat), Endpoint: worker.Heartbeat.Endpoint,
			ShardIndex: uint32(index), MemoryBudgetBytes: worker.Heartbeat.MemoryFree, GPUVramBudgetBytes: worker.Heartbeat.GPUVramFree,
		})
	}
	for index, page := range plan.Pages {
		page.WorkerID = replacements[index%len(replacements)].Heartbeat.WorkerID
		page.Epoch = epoch
		plan.Pages[index] = page
	}
	if err := plan.Validate(); err != nil {
		return fabricwire.LogicalDevicePlan{}, err
	}
	m.plans[planID] = plan
	m.requests[planID] = plan
	m.states[planID] = fabricwire.LogicalDeviceRecovering
	return plan, nil
}

func (m *LogicalDeviceManager) Transfer(request fabricwire.TransferRequest) (fabricwire.TransferStatus, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	plan, ok := m.plans[request.PlanID]
	if !ok {
		return fabricwire.TransferStatus{}, fmt.Errorf("logical device plan %q not found", request.PlanID)
	}
	if err := request.ValidateForEpoch(plan.Epoch); err != nil {
		return fabricwire.TransferStatus{}, err
	}
	if existing, ok := m.transfers[request.RequestID]; ok {
		return existing, nil
	}
	if request.PageID == "" || request.TargetWorkerID == "" || request.TargetTierID == "" || request.ExpectedDigest == "" {
		return fabricwire.TransferStatus{}, fmt.Errorf("transfer requires pageId, target worker and tier, and expected digest")
	}
	var page fabricwire.PagePlacement
	found := false
	for _, candidate := range plan.Pages {
		if candidate.PageID == request.PageID && !candidate.Replica {
			page = candidate
			found = true
			break
		}
	}
	if !found {
		return fabricwire.TransferStatus{}, fmt.Errorf("page %q not found in logical device plan", request.PageID)
	}
	worker, ok := m.workers.Worker(request.TargetWorkerID)
	if !ok || worker.State != fabricwire.WorkerReady {
		return fabricwire.TransferStatus{}, fmt.Errorf("target worker %q is not ready", request.TargetWorkerID)
	}
	transfer := fabricwire.TransferStatus{TransferID: request.RequestID, PlanID: plan.PlanID, Epoch: plan.Epoch, PageID: page.PageID, SourceWorkerID: page.WorkerID, TargetWorkerID: request.TargetWorkerID, TargetTierID: request.TargetTierID, ExpectedDigest: request.ExpectedDigest, State: fabricwire.TransferQueued, Bytes: page.Bytes}
	m.transfers[transfer.TransferID] = transfer
	return transfer, nil
}

func (m *LogicalDeviceManager) CompleteTransfer(transferID, digest string) (fabricwire.TransferStatus, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	transfer, ok := m.transfers[transferID]
	if !ok {
		return fabricwire.TransferStatus{}, fmt.Errorf("transfer %q not found", transferID)
	}
	if transfer.State != fabricwire.TransferQueued {
		return transfer, fmt.Errorf("transfer %q is not queued", transferID)
	}
	if digest != transfer.ExpectedDigest {
		transfer.State = fabricwire.TransferFailed
		transfer.Error = "transfer digest mismatch"
		m.transfers[transferID] = transfer
		return transfer, fmt.Errorf("transfer digest mismatch")
	}
	for _, next := range []fabricwire.TransferState{fabricwire.TransferAdmitted, fabricwire.TransferCopying, fabricwire.TransferVerified} {
		if !fabricwire.CanTransitionTransfer(transfer.State, next) {
			return transfer, fmt.Errorf("invalid transfer transition %q to %q", transfer.State, next)
		}
		transfer.State = next
	}
	plan := m.plans[transfer.PlanID]
	for index, page := range plan.Pages {
		if page.PageID == transfer.PageID && !page.Replica {
			page.WorkerID = transfer.TargetWorkerID
			page.TierID = transfer.TargetTierID
			page.Digest = digest
			plan.Pages[index] = page
			break
		}
	}
	m.plans[transfer.PlanID] = plan
	m.requests[plan.PlanID] = plan
	m.transfers[transferID] = transfer
	return transfer, nil
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
	m.states[plan.PlanID] = fabricwire.LogicalDevicePlanned
	m.mu.Unlock()
	return plan, nil
}
