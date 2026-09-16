// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type FabricWorkerState = 'ready' | 'suspect' | 'quarantined' | 'draining' | 'failed'

export interface FabricWorker {
    workerId: string
    nodeId: string
    peerId?: string
    endpoint: string
    state: FabricWorkerState
    runtime: string
    backends: string[]
    modelDigests: string[]
    checkpointSupport: boolean
    rpcSupport: boolean
    probeLatencyMillis: number
    memoryFree: number
    gpuVramTotal: number
    gpuVramFree: number
    gpuCount: number
}

export interface FabricJob {
    jobId: string
    groupId: string
    modelDigest: string
    epoch: number
    state: string
}

export interface FabricExecution {
    jobId: string
    pid: number
    state: string
    phase: string
    message: string
    workers: string[]
    epoch: number
    recoveryAttempts: number
    checkpointFile: string
    checkpointStage: number
    updatedAt: string
    rpcPeers: number
    httpPort: number
}

export interface FabricGroupRequest {
    groupId: string
    runtime: string
    backends: string[]
    workerGoal: number
    allowMixed: boolean
    modelDigest?: string
    requireCheckpoint?: boolean
    maxProbeLatencyMillis?: number
    minMemoryFreeBytes?: number
    minGpuVramTotalBytes?: number
    minGpuVramFreeBytes?: number
    minAggregateMemoryFreeBytes?: number
    minAggregateGpuVramTotalBytes?: number
    minAggregateGpuVramFreeBytes?: number
}

export interface FabricGroupPlan {
    groupId: string
    runtime: string
    workers: string[]
    endpoints: Record<string, string>
    epoch: number
    memoryFree: number
    gpuVramTotal: number
    gpuVramFree: number
    gpuCount: number
}

export interface FabricInferenceRequest {
    jobId: string
    modelDigest: string
    serverPath: string
    serverPrefixArgs?: string[]
    modelPath: string
    httpPort: number
    slotSavePath?: string
    checkpointFile?: string
    checkpointIntervalSeconds?: number
    group: FabricGroupRequest
}

export interface FabricCapacity {
    workers: number
    memoryFree: number
    gpuVramTotal: number
    gpuVramFree: number
    gpuCount: number
}

export type FabricProvider = 'cpu' | 'cuda' | 'metal'

export interface FabricMemoryTier {
    tierId: string
    kind: string
    capacityBytes: number
    freeBytes: number
    bandwidthBytesPerSec?: number
    latencyMicros?: number
    local: boolean
}

export interface FabricProviderCapability {
    provider: FabricProvider
    deviceId: string
    supportsExecution: boolean
    supportsCollectives: string[]
    memoryTiers: FabricMemoryTier[]
    maxPageBytes: number
}

export interface FabricLogicalDeviceDescribe {
    protocolVersion: number
    workerId: string
    providers: FabricProviderCapability[]
}

export interface FabricPageSpec {
    pageId: string
    bytes: number
    dtype?: string
    layout?: string
}

export interface FabricLogicalDeviceRequest {
    protocolVersion: number
    requestId: string
    planId?: string
    epoch?: number
    deadlineUnixMs?: number
}

export interface FabricLogicalDevicePlanRequest extends FabricLogicalDeviceRequest {
    runtime: string
    modelDigest: string
    shardStrategy: 'replicated' | 'tensor' | 'pipeline'
    providers?: FabricProvider[]
    workerGoal: number
    pages: FabricPageSpec[]
}

export interface FabricPagePlacement extends FabricPageSpec {
    workerId: string
    tierId: string
    digest?: string
    epoch: number
    replica?: boolean
}

export interface FabricLogicalDevicePlan {
    version: number
    planId: string
    epoch: number
    runtime: string
    modelDigest: string
    shardStrategy: 'replicated' | 'tensor' | 'pipeline'
    workers: Array<{ workerId: string; peerId?: string; endpoint?: string; shardIndex: number; memoryBudgetBytes: number; gpuVramBudgetBytes?: number }>
    pages: FabricPagePlacement[]
}

export interface FabricTransferRequest extends FabricLogicalDeviceRequest {
    planId: string
    epoch: number
    pageId: string
    targetWorkerId: string
    targetTierId: string
    expectedDigest: string
}

export interface FabricTransferStatus {
    transferId: string
    planId: string
    epoch: number
    pageId: string
    sourceWorkerId: string
    targetWorkerId: string
    targetTierId: string
    expectedDigest: string
    state: 'queued' | 'admitted' | 'copying' | 'verified' | 'failed' | 'cancelled'
    bytes: number
    error?: string
}

export interface FabricLogicalDeviceStatus {
    planId: string
    epoch: number
    state: string
    workers: string[]
    pages: FabricPagePlacement[]
    transferIds: string[]
    updatedAtMs: number
}

export interface FabricStatus {
    workers: FabricWorker[]
    capacity: FabricCapacity
    jobs: FabricJob[]
    executions: FabricExecution[]
    logicalDevice?: FabricLogicalDeviceDescribe
}

export type FabricTrainingParallelism = 'data' | 'fsdp'

export interface FabricTrainingNode {
    workerId: string
    address: string
    backend: 'cpu' | 'cuda' | 'metal'
    gpuCount?: number
}

export interface FabricTrainingRequest {
    jobId: string
    modelDigest: string
    trainerPath: string
    modelPath: string
    datasetPath: string
    checkpointDirectory: string
    resumeCheckpoint?: string
    rendezvousEndpoint: string
    parallelism: FabricTrainingParallelism
    processesPerNode: number
    checkpointIntervalSteps: number
    nodes: FabricTrainingNode[]
}

export interface FabricTrainingExecution {
    jobId: string
    nodeRank: number
    pid: number
    state: string
    checkpoint?: FabricCheckpoint
}

export interface FabricCheckpoint {
    jobId: string
    groupId: string
    epoch: number
    stage: number
    slotId?: number
    filename: string
}

export interface FabricTrainingGroupStatus {
    jobId: string
    state: string
    epoch: number
    executions: FabricTrainingExecution[]
    checkpoint: FabricCheckpoint
    recoveryAttempts: number
}
