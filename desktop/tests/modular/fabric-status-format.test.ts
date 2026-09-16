// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { formatFabricMemory, formatFabricWorkerSummary } from '@/ui/components/NodeList/fabric-status-format'

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
})
