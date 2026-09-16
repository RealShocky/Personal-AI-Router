// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { formatFabricGPUCapacity, formatFabricMemory, formatFabricWorkerSummary } from '@/ui/components/NodeList/fabric-status-format'

describe('fabric status formatting', () => {
    test('formats worker resources for the operator card', () => {
        expect(formatFabricMemory(24521306112)).toBe('22.8 GiB free')
        expect(
            formatFabricWorkerSummary({
                runtime: 'cuda',
                backends: ['cuda'],
                memoryFree: 24521306112,
                probeLatencyMillis: 4,
                checkpointSupport: true
            })
        ).toBe('CUDA · 22.8 GiB free · 4 ms · cuda · checkpoints')
    })

    test('formats GPU VRAM capacity and identifies CPU-only workers', () => {
        expect(formatFabricGPUCapacity({ backends: ['cuda'], gpuCount: 1, gpuVramFree: 6 * 1024 ** 3, gpuVramTotal: 8 * 1024 ** 3 })).toBe('1 GPU · 6.0/8.0 GiB VRAM free')
        expect(formatFabricGPUCapacity({ backends: ['cpu'], gpuCount: 0, gpuVramFree: 0, gpuVramTotal: 0 })).toBe('CPU-only')
        expect(formatFabricGPUCapacity({ backends: ['metal'], gpuCount: 0, gpuVramFree: 0, gpuVramTotal: 0 })).toBe('GPU telemetry unavailable')
    })
})
