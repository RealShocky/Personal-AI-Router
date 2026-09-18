// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { FabricLogicalDeviceDescribe, FabricLogicalDevicePlan, FabricLogicalDeviceStatus, FabricStatus } from '@/shared/types/fabric'

const EMPTY_STATUS: FabricStatus = { workers: [], capacity: { workers: 0, memoryFree: 0, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 }, jobs: [], executions: [] }
const EMPTY_DESCRIBE: FabricLogicalDeviceDescribe = { protocolVersion: 1, workerId: 'coordinator', providers: [] }

function formatBytes(bytes: number): string {
    if (bytes < 1024 ** 2) return `${Math.round(bytes / 1024)} KiB`
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`
}

export default function FabricLogicalDeviceCard() {
    const [fabric, setFabric] = useState<FabricStatus>(EMPTY_STATUS)
    const [describe, setDescribe] = useState<FabricLogicalDeviceDescribe>(EMPTY_DESCRIBE)
    const [plan, setPlan] = useState<FabricLogicalDevicePlan | null>(null)
    const [planStatus, setPlanStatus] = useState<FabricLogicalDeviceStatus | null>(null)
    const [workerGoal, setWorkerGoal] = useState('2')
    const [pageBytes, setPageBytes] = useState(String(1 << 20))
    const [modelDigest, setModelDigest] = useState('sha256:logical-device-preview')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState('')

    const refresh = async () => {
        const [nextFabric, nextDescribe] = await Promise.all([
            window.pairApi.fabric.getStatus(),
            window.pairApi.fabric.describeLogicalDevice()
        ])
        setFabric(nextFabric)
        setDescribe(nextDescribe)
        if (plan) {
            setPlanStatus(await window.pairApi.fabric.getLogicalDeviceStatus(plan.planId))
        }
    }

    useEffect(() => {
        let active = true
        const load = async () => {
            try {
                if (active) await refresh()
            } catch {
                if (active) setError('Logical-device status is unavailable')
            }
        }
        void load()
        const timer = window.setInterval(() => void load(), 5000)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [plan])

    const readyCuda = fabric.workers.filter(worker => worker.state === 'ready' && worker.backends.includes('cuda')).length
    const cudaProviders = describe.providers.filter(provider => provider.provider === 'cuda')

    const preview = async () => {
        setBusy(true)
        setError('')
        try {
            const bytes = Number(pageBytes)
            const goal = Number(workerGoal)
            if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error('Page size must be a positive number of bytes')
            if (!Number.isSafeInteger(goal) || goal <= 0) throw new Error('Worker goal must be a positive number')
            const pageId = `ui-logical-page-${Date.now()}`
            const nextPlan = await window.pairApi.fabric.planLogicalDevice({
                protocolVersion: 1,
                requestId: `ui-logical-plan-${Date.now()}`,
                runtime: 'pair',
                modelDigest: modelDigest.trim(),
                shardStrategy: 'pipeline',
                providers: ['cuda'],
                workerGoal: goal,
                pages: [{ pageId, bytes, dtype: 'u8', layout: 'flat' }]
            })
            setPlan(nextPlan)
            setPlanStatus(await window.pairApi.fabric.getLogicalDeviceStatus(nextPlan.planId))
        } catch (cause) {
            setError(cause instanceof Error ? cause.message : 'Logical-device planning failed')
        } finally {
            setBusy(false)
        }
    }

    return (
        <div className="node-card pair-paper mb-3" data-fabric-logical-device>
            <Stack gap="2">
                <Flex align="center" justify="between" gap="2">
                    <Text kind="body/semibold/sm">Logical device fabric</Text>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {readyCuda} CUDA worker{readyCuda === 1 ? '' : 's'} ready
                    </Text>
                </Flex>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Plans work across selected workers. This is schedulable fabric capacity, not one Windows GPU or contiguous VRAM pool.
                </Text>
                <Flex gap="2" wrap="wrap">
                    <FormField slotLabel="Model digest"><TextInput value={modelDigest} onValueChange={setModelDigest} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="CUDA workers"><TextInput value={workerGoal} onValueChange={setWorkerGoal} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Page bytes"><TextInput value={pageBytes} onValueChange={setPageBytes} disabled={busy} size="small" /></FormField>
                </Flex>
                <Flex gap="2" align="center" wrap="wrap">
                    <Button kind="primary" color="brand" size="small" onClick={() => void preview()} disabled={busy || readyCuda < Number(workerGoal)}>
                        Preview logical plan
                    </Button>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {cudaProviders.length} CUDA provider{cudaProviders.length === 1 ? '' : 's'} advertised
                    </Text>
                </Flex>
                {plan && <Text kind="body/regular/sm" className="text-subtle-color">
                    Plan {plan.planId}: {plan.workers.map(worker => worker.workerId).join(', ')} · page on {plan.pages[0]?.workerId} ({plan.pages[0]?.tierId}) · {formatBytes(plan.pages[0]?.bytes ?? 0)}
                    {planStatus ? ` · ${planStatus.state}` : ''}
                </Text>}
                {error && <Text kind="body/regular/sm" className="text-error-color">{error}</Text>}
            </Stack>
        </div>
    )
}
