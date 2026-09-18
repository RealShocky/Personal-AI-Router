// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { FabricStatus } from '@/shared/types/fabric'
import { formatFabricGPUCapacity, formatFabricWorkerSummary } from './fabric-status-format'
import { summarizeFabricPool } from './fabric-pool'

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
    const pool = summarizeFabricPool(status)
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
                <div className="fabric-pool-grid" data-fabric-pool-summary>
                    <div className="fabric-pool-metric fabric-pool-metric-primary">
                        <Text kind="body/regular/sm" className="text-subtle-color">PAIR compute pool</Text>
                        <Text kind="body/semibold/lg">{pool.readyWorkers} ready worker{pool.readyWorkers === 1 ? '' : 's'}</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">{pool.pairedWorkers} paired · {pool.activeJobs} active job{pool.activeJobs === 1 ? '' : 's'} · {pool.activeExecutions} execution{pool.activeExecutions === 1 ? '' : 's'}</Text>
                    </div>
                    <div className="fabric-pool-metric">
                        <Text kind="body/regular/sm" className="text-subtle-color">Schedulable host RAM</Text>
                        <Text kind="body/semibold/lg">{(pool.memoryFree / 1024 ** 3).toFixed(1)} GiB</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">offload and cache tier</Text>
                    </div>
                    <div className="fabric-pool-metric">
                        <Text kind="body/regular/sm" className="text-subtle-color">GPU capacity</Text>
                        <Text kind="body/semibold/lg">{pool.gpuCount} GPU{pool.gpuCount === 1 ? '' : 's'}</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">{pool.gpuVramTotal > 0 ? `${(pool.gpuVramFree / 1024 ** 3).toFixed(1)} / ${(pool.gpuVramTotal / 1024 ** 3).toFixed(1)} GiB VRAM free` : 'telemetry pending'}</Text>
                    </div>
                    <div className="fabric-pool-metric">
                        <Text kind="body/regular/sm" className="text-subtle-color">Ready providers</Text>
                        <Text kind="body/semibold/lg">{pool.cpuWorkers + pool.cudaWorkers + pool.metalWorkers}</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">{pool.cpuWorkers} CPU · {pool.cudaWorkers} CUDA · {pool.metalWorkers} Metal</Text>
                    </div>
                </div>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    The pool is PAIR’s logical scheduling view. Pages and model stages can move between workers; physical VRAM remains local to each device.
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
                {pool.recoveringWorkers > 0 && (
                    <Text kind="body/regular/sm" className="text-warning-color">
                        {pool.recoveringWorkers} worker{pool.recoveringWorkers === 1 ? '' : 's'} recovering or quarantined; PAIR will re-admit it after a healthy handshake.
                    </Text>
                )}
            </Stack>
        </div>
    )
}
