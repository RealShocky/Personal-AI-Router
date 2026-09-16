// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { FabricStatus } from '@/shared/types/fabric'
import { formatFabricWorkerSummary } from './fabric-status-format'

const EMPTY_STATUS: FabricStatus = { workers: [], jobs: [], executions: [] }

function stateLabel(status: FabricStatus): string {
    if (status.workers.some(worker => worker.state === 'quarantined')) return 'DEGRADED'
    if (status.workers.some(worker => worker.state === 'suspect')) return 'RECOVERING'
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
                {status.workers.map(worker => (
                    <Flex key={worker.workerId} align="center" justify="between" gap="2">
                        <Text kind="body/regular/sm">{worker.nodeId}</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {worker.state} · {formatFabricWorkerSummary(worker)}
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
                        Executions: {status.executions.map(execution => `${execution.jobId} (${execution.state})`).join(' · ')}
                    </Text>
                )}
            </Stack>
        </div>
    )
}
