// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FabricStatus, FabricTrainingNode } from '@/shared/types/fabric'

type TrainingBackend = FabricTrainingNode['backend']

function trainingBackend(worker: FabricStatus['workers'][number]): TrainingBackend | null {
    if (worker.runtime === 'cuda' && worker.backends.includes('cuda')) return 'cuda'
    if (worker.runtime === 'cpu' && worker.backends.includes('cpu')) return 'cpu'
    return null
}

export function compatibleTrainingNodes(status: FabricStatus): FabricTrainingNode[] {
    const candidates = status.workers
        .filter(worker => worker.state === 'ready' && worker.endpoint !== '')
        .map(worker => ({ worker, backend: trainingBackend(worker) }))
        .filter((item): item is { worker: FabricStatus['workers'][number]; backend: TrainingBackend } => item.backend !== null)
    const selectedBackend = candidates[0]?.backend
    if (!selectedBackend) return []
    return candidates
        .filter(item => item.backend === selectedBackend)
        .map(item => ({ workerId: item.worker.workerId, address: item.worker.endpoint, backend: item.backend, ...(item.worker.gpuCount > 0 ? { gpuCount: item.worker.gpuCount } : {}) }))
}
