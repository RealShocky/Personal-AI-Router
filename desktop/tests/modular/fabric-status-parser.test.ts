// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { parseFabricGroupPlan, parseFabricStatus } from '@/electron/service-bridge/empty-handlers'

describe('fabric status parser', () => {
    test('normalizes coordinator worker records with nested heartbeats', () => {
        const status = parseFabricStatus({
            capacity: { workers: 1, memoryFreeBytes: 4096, gpuVramTotalBytes: 0, gpuVramFreeBytes: 0, gpuCount: 1 },
            workers: [
                {
                    Heartbeat: {
                        workerId: 'dgx-spark-gb10',
                        nodeId: 'spark-b57c',
                        peerId: 'cluster-peer-dgx',
                        endpoint: 'https://192.168.50.223:14324',
                        state: 'ready',
                        runtime: 'cuda',
                        backends: ['cuda'],
                        memoryFreeBytes: 4096,
                        gpuCount: 1
                    },
                    State: 'ready'
                }
            ],
            jobs: [],
            executions: []
        })

        expect(status.workers[0]).toMatchObject({
            workerId: 'dgx-spark-gb10',
            nodeId: 'spark-b57c',
            peerId: 'cluster-peer-dgx',
            endpoint: 'https://192.168.50.223:14324',
            state: 'ready',
            runtime: 'cuda',
            memoryFree: 4096,
            gpuCount: 1
        })
    })

    test('normalizes a dry-run group plan for operator preview', () => {
        const plan = parseFabricGroupPlan({
            groupId: 'pair-inference-1',
            runtime: 'llama.cpp',
            workers: ['wsl-5060', 'dgx-spark-gb10'],
            endpoints: { 'wsl-5060': 'https://wsl', 'dgx-spark-gb10': 'https://dgx' },
            epoch: 42,
            memoryFree: 0,
            gpuVramTotal: 0,
            gpuVramFree: 0,
            gpuCount: 0
        })

        expect(plan).toEqual({
            groupId: 'pair-inference-1',
            runtime: 'llama.cpp',
            workers: ['wsl-5060', 'dgx-spark-gb10'],
            endpoints: { 'wsl-5060': 'https://wsl', 'dgx-spark-gb10': 'https://dgx' },
            epoch: 42,
            memoryFree: 0,
            gpuVramTotal: 0,
            gpuVramFree: 0,
            gpuCount: 0
        })
    })
})
