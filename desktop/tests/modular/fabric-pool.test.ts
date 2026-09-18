// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { FabricStatus } from '@/shared/types/fabric'
import { summarizeFabricPool } from '@/ui/components/NodeList/fabric-pool'

const status: FabricStatus = {
    workers: [
        { workerId: 'dgx', nodeId: 'dgx', endpoint: '', state: 'ready', runtime: 'cuda', backends: ['cuda'], modelDigests: [], checkpointSupport: true, rpcSupport: true, probeLatencyMillis: 2, memoryFree: 10, gpuVramTotal: 80, gpuVramFree: 60, gpuCount: 1 },
        { workerId: 'aws', nodeId: 'aws', endpoint: '', state: 'quarantined', runtime: 'cpu', backends: ['cpu'], modelDigests: [], checkpointSupport: false, rpcSupport: false, probeLatencyMillis: 40, memoryFree: 20, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 },
        { workerId: 'mac', nodeId: 'mac', endpoint: '', state: 'ready', runtime: 'metal', backends: ['metal'], modelDigests: [], checkpointSupport: true, rpcSupport: false, probeLatencyMillis: 8, memoryFree: 30, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 1 }
    ],
    capacity: { workers: 2, memoryFree: 40, gpuVramTotal: 80, gpuVramFree: 60, gpuCount: 2 },
    jobs: [{ jobId: 'job-1', groupId: 'group-1', modelDigest: 'sha256:model', epoch: 1, state: 'running' }],
    executions: []
}

describe('summarizeFabricPool', () => {
    it('counts ready providers while preserving aggregate schedulable capacity', () => {
        expect(summarizeFabricPool(status)).toEqual({
            readyWorkers: 2,
            pairedWorkers: 3,
            activeJobs: 1,
            activeExecutions: 0,
            recoveringWorkers: 1,
            cpuWorkers: 0,
            cudaWorkers: 1,
            metalWorkers: 1,
            memoryFree: 40,
            gpuVramFree: 60,
            gpuVramTotal: 80,
            gpuCount: 2
        })
    })
})
