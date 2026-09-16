// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { FabricStatus } from '@/shared/types/fabric'
import { formatFabricGPUCapacity, formatFabricWorkerSummary } from './fabric-status-format'

const EMPTY_STATUS: FabricStatus = { workers: [], capacity: { workers: 0, memoryFree: 0, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 }, jobs: [], executions: [] }

function stateLabel(status: FabricStatus): string {
    if (status.workers.some(worker => worker.state === 'quarantined')) return 'DEGRADED'
    if (status.workers.some(worker => worker.state === 'suspect')) return 'RECOVERING'
    if (status.executions.some(execution => execution.phase === 'recovering')) return 'RECOVERING'
    if (status.executions.some(execution => execution.phase === 'failed')) return 'JOB FAILED'
    if (status.executions.some(execution => execution.state === 'running')) return 'RUNNING'
    if (status.workers.length > 0) return 'READY'
    return 'WAITING FOR WORKERS'
}

export default function FabricStatusCard() {
    const [status, setStatus] = useState<FabricStatus>(EMPTY_STATUS)

    useEffect(() => {
        let active = true
        const refresh = () => {
            void window.pairApi.fabric
                .getStatus()
                .then(next => {
                    if (active) setStatus(next)
                })
                .catch(() => {
                    if (active) setStatus(EMPTY_STATUS)
                })
        }
        refresh()
        const timer = window.setInterval(refresh, 5000)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [])

    const ready = status.workers.filter(worker => worker.state === 'ready').length
    const liveExecutions = status.executions.filter(execution => execution.state === 'running').length
    const logicalProviders = status.logicalDevice?.providers ?? []
    const providerSummary = logicalProviders.length === 0
        ? 'no provider capabilities reported'
        : logicalProviders.map(provider => `${provider.provider}:${provider.deviceId}`).join(' · ')

    return (
        <div className="node-card pair-paper mb-3" data-fabric-status>
            <Stack gap="1">
                <Flex align="center" justify="between" gap="2">
                    <Text kind="body/semibold/sm" className="uppercase">
                        Distributed inference fabric
                    </Text>
                    <Text kind="body/semibold/sm" className="text-subtle-color">
                        {stateLabel(status)}
                    </Text>
                </Flex>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {ready} ready worker{ready === 1 ? '' : 's'} · {status.workers.length} paired ·{' '}
                    {liveExecutions} active job{liveExecutions === 1 ? '' : 's'}
                </Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Logical ready capacity: {status.capacity.workers} worker{status.capacity.workers === 1 ? '' : 's'} ·{' '}
                    {(status.capacity.memoryFree / 1024 ** 3).toFixed(1)} GiB host RAM ·{' '}
                    {status.capacity.gpuCount > 0 ? `${(status.capacity.gpuVramFree / 1024 ** 3).toFixed(1)}/${(status.capacity.gpuVramTotal / 1024 ** 3).toFixed(1)} GiB VRAM free` : 'no reported NVIDIA VRAM'}
                </Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Logical providers: {providerSummary}. Memory tiers are explicit; capacity is schedulable, not contiguous VRAM.
                </Text>
                {status.workers.map(worker => (
                    <Flex key={worker.workerId} align="center" justify="between" gap="2">
                        <Text kind="body/regular/sm">{worker.nodeId}</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {worker.state} · {formatFabricWorkerSummary(worker)} · {formatFabricGPUCapacity(worker)}
                        </Text>
                    </Flex>
                ))}
                {status.jobs.length > 0 && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Jobs: {status.jobs.map(job => `${job.jobId} (${job.state})`).join(' · ')}
                    </Text>
                )}
                {status.executions.length > 0 && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        Executions: {status.executions.map(execution => `${execution.jobId} (${execution.phase || execution.state}${execution.workers.length > 0 ? ` on ${execution.workers.join(', ')}` : ''})`).join(' · ')}
                    </Text>
                )}
            </Stack>
        </div>
    )
}
