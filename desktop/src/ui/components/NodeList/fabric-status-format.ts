// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FabricWorker } from '@/shared/types/fabric'

export function formatFabricMemory(bytes: number): string {
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB free`
}

export function formatFabricWorkerSummary(worker: Pick<FabricWorker, 'runtime' | 'backends' | 'memoryFree' | 'probeLatencyMillis' | 'checkpointSupport'>): string {
    const runtime = worker.runtime.toUpperCase()
    const backends = worker.backends.length > 0 ? worker.backends.join(', ') : runtime.toLowerCase()
    const checkpoint = worker.checkpointSupport ? ' · checkpoints' : ''
    return `${runtime} · ${formatFabricMemory(worker.memoryFree)} · ${worker.probeLatencyMillis} ms · ${backends}${checkpoint}`
}

export function formatFabricGPUCapacity(worker: Pick<FabricWorker, 'gpuCount' | 'gpuVramTotal' | 'gpuVramFree' | 'backends'>): string {
    if (worker.gpuCount === 0 || worker.gpuVramTotal === 0) {
        return worker.backends.some(backend => backend === 'cuda' || backend === 'metal') ? 'GPU telemetry unavailable' : 'CPU-only'
    }
    const freeGiB = worker.gpuVramFree / 1024 ** 3
    const totalGiB = worker.gpuVramTotal / 1024 ** 3
    return `${worker.gpuCount} GPU${worker.gpuCount === 1 ? '' : 's'} · ${freeGiB.toFixed(1)}/${totalGiB.toFixed(1)} GiB VRAM free`
}
