// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { FabricStatus } from '@/shared/types/fabric'
import { compatibleTrainingNodes } from '@/ui/components/NodeList/fabric-training'

describe('compatibleTrainingNodes', () => {
    it('selects ready homogeneous CPU or CUDA workers with endpoints', () => {
        const status = {
            workers: [
                { workerId: 'cuda-a', nodeId: 'a', endpoint: 'https://a', state: 'ready', runtime: 'cuda', backends: ['cuda'], modelDigests: [], checkpointSupport: true, probeLatencyMillis: 1, memoryFree: 1, gpuVramTotal: 1, gpuVramFree: 1, gpuCount: 1 },
                { workerId: 'metal', nodeId: 'm', endpoint: 'https://m', state: 'ready', runtime: 'metal', backends: ['metal'], modelDigests: [], checkpointSupport: true, probeLatencyMillis: 1, memoryFree: 1, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 1 },
                { workerId: 'offline', nodeId: 'o', endpoint: '', state: 'suspect', runtime: 'cpu', backends: ['cpu'], modelDigests: [], checkpointSupport: true, probeLatencyMillis: 1, memoryFree: 1, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 }
            ],
            capacity: { workers: 1, memoryFree: 1, gpuVramTotal: 1, gpuVramFree: 1, gpuCount: 1 },
            jobs: [],
            executions: []
        } satisfies FabricStatus
        expect(compatibleTrainingNodes(status)).toEqual([{ workerId: 'cuda-a', address: 'https://a', backend: 'cuda', gpuCount: 1 }])
    })
})
