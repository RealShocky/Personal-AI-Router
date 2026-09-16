// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type FabricWorkerState = 'ready' | 'suspect' | 'quarantined' | 'draining' | 'failed'

export interface FabricWorker {
    workerId: string
    nodeId: string
    state: FabricWorkerState
    runtime: string
    backends: string[]
    modelDigests: string[]
    checkpointSupport: boolean
    probeLatencyMillis: number
    memoryFree: number
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
    rpcPeers: number
    httpPort: number
}

export interface FabricStatus {
    workers: FabricWorker[]
    jobs: FabricJob[]
    executions: FabricExecution[]
}
