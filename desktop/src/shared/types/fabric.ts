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

export interface FabricStatus {
    workers: FabricWorker[]
    capacity: FabricCapacity
    jobs: FabricJob[]
    executions: FabricExecution[]
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
