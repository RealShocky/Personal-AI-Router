// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FabricStatus } from '@/shared/types/fabric'

interface FabricPoolSummary {
    readyWorkers: number
    pairedWorkers: number
    activeJobs: number
    activeExecutions: number
    recoveringWorkers: number
    cpuWorkers: number
    cudaWorkers: number
    metalWorkers: number
    memoryFree: number
    gpuVramFree: number
    gpuVramTotal: number
    gpuCount: number
}

export function summarizeFabricPool(status: FabricStatus): FabricPoolSummary {
    const ready = status.workers.filter(worker => worker.state === 'ready')
    const hasBackend = (backend: string) => ready.filter(worker => worker.backends.includes(backend)).length
    return {
        readyWorkers: ready.length,
        pairedWorkers: status.workers.length,
        activeJobs: status.jobs.filter(job => job.state === 'running' || job.state === 'starting').length,
        activeExecutions: status.executions.filter(execution => execution.state === 'running').length,
        recoveringWorkers: status.workers.filter(worker => worker.state === 'suspect' || worker.state === 'quarantined' || worker.state === 'draining').length,
        cpuWorkers: hasBackend('cpu'),
        cudaWorkers: hasBackend('cuda'),
        metalWorkers: hasBackend('metal'),
        memoryFree: status.capacity.memoryFree,
        gpuVramFree: status.capacity.gpuVramFree,
        gpuVramTotal: status.capacity.gpuVramTotal,
        gpuCount: status.capacity.gpuCount
    }
}
