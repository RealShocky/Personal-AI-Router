// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Checkbox, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { FabricExecution, FabricGroupPlan, FabricInferenceRequest, FabricStatus } from '@/shared/types/fabric'

const EMPTY_STATUS: FabricStatus = { workers: [], capacity: { workers: 0, memoryFree: 0, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 }, jobs: [], executions: [] }

export default function FabricInferenceCard() {
    const [fabric, setFabric] = useState<FabricStatus>(EMPTY_STATUS)
    const [execution, setExecution] = useState<FabricExecution | null>(null)
    const [plan, setPlan] = useState<FabricGroupPlan | null>(null)
    const [jobId, setJobId] = useState('pair-inference-1')
    const [modelDigest, setModelDigest] = useState('')
    const [serverPath, setServerPath] = useState('llama-server')
    const [runThroughWSL, setRunThroughWSL] = useState(false)
    const [wslDistro, setWslDistro] = useState('Ubuntu')
    const [wslServerPath, setWslServerPath] = useState('/opt/llama-b8487/build-cuda128/bin/llama-server')
    const [modelPath, setModelPath] = useState('')
    const [httpPort, setHttpPort] = useState('19090')
    const [workerGoal, setWorkerGoal] = useState('2')
    const [slotSavePath, setSlotSavePath] = useState('')
    const [checkpointFile, setCheckpointFile] = useState('')
    const [checkpointInterval, setCheckpointInterval] = useState('0')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        let active = true
        const refresh = async () => {
            try {
                const next = await window.pairApi.fabric.getStatus()
                if (active) setFabric(next)
                if (jobId.trim()) {
                    const running = await window.pairApi.fabric.getInferenceStatus(jobId.trim())
                    if (active) setExecution(running)
                }
            } catch {
                if (active) setExecution(null)
            }
        }
        void refresh()
        const timer = window.setInterval(() => void refresh(), 5000)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [jobId])

    const readyRPC = fabric.workers.filter(worker => worker.state === 'ready' && worker.rpcSupport && (worker.backends.includes('cpu') || worker.backends.includes('cuda'))).length
    const run = async (operation: () => Promise<void>) => {
        setBusy(true)
        setError('')
        try {
            await operation()
        } catch (cause) {
            setError(cause instanceof Error ? cause.message : 'Inference operation failed')
        } finally {
            setBusy(false)
        }
    }

    const start = () =>
        run(async () => {
            const goal = Number(workerGoal)
            const request: FabricInferenceRequest = {
                jobId: jobId.trim(),
                modelDigest: modelDigest.trim(),
                serverPath: runThroughWSL ? 'wsl.exe' : serverPath.trim(),
                ...(runThroughWSL ? { serverPrefixArgs: ['-d', wslDistro.trim(), '--', wslServerPath.trim()] } : {}),
                modelPath: modelPath.trim(),
                httpPort: Number(httpPort),
                ...(slotSavePath.trim() === '' ? {} : { slotSavePath: slotSavePath.trim() }),
                ...(checkpointFile.trim() === '' ? {} : { checkpointFile: checkpointFile.trim() }),
                ...(Number(checkpointInterval) > 0 ? { checkpointIntervalSeconds: Number(checkpointInterval) } : {}),
                group: { groupId: jobId.trim(), runtime: 'llama.cpp', backends: ['cpu', 'cuda'], workerGoal: goal, allowMixed: true }
            }
            setExecution(await window.pairApi.fabric.startInference(request))
        })

    const dryRun = () =>
        run(async () => {
            const goal = Number(workerGoal)
            const request = {
                groupId: jobId.trim(),
                runtime: 'llama.cpp',
                backends: ['cpu', 'cuda'],
                workerGoal: goal,
                allowMixed: true,
            }
            setPlan(await window.pairApi.fabric.planGroup(request))
        })

    const stop = () => run(async () => {
        await window.pairApi.fabric.stopInference(jobId.trim())
        setExecution(null)
    })

    return (
        <div className="node-card pair-paper mb-3" data-fabric-inference>
            <Stack gap="2">
                <Flex align="center" justify="between" gap="2">
                    <Text kind="body/semibold/sm">Distributed inference</Text>
                    <Text kind="body/regular/sm" className="text-subtle-color">{readyRPC} CPU/CUDA RPC workers ready</Text>
                </Flex>
                <Text kind="body/regular/sm" className="text-subtle-color">Launches one llama.cpp service with authenticated RPC placement across the selected workers.</Text>
                <Flex gap="2" wrap="wrap">
                    <FormField slotLabel="Job ID"><TextInput value={jobId} onValueChange={setJobId} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Model digest"><TextInput value={modelDigest} onValueChange={setModelDigest} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="llama-server"><TextInput value={serverPath} onValueChange={setServerPath} disabled={busy} size="small" /></FormField>
                    <Flex align="center" gap="1">
                        <Checkbox checked={runThroughWSL} onCheckedChange={checked => setRunThroughWSL(checked === true)} disabled={busy} aria-label="Run through WSL" />
                        <Text kind="body/regular/sm">Run through WSL</Text>
                    </Flex>
                    {runThroughWSL && <FormField slotLabel="WSL distro"><TextInput value={wslDistro} onValueChange={setWslDistro} disabled={busy} size="small" />
                    </FormField>}
                    {runThroughWSL && <FormField slotLabel="WSL llama-server"><TextInput value={wslServerPath} onValueChange={setWslServerPath} disabled={busy} size="small" />
                    </FormField>}
                    <FormField slotLabel="Model path"><TextInput value={modelPath} onValueChange={setModelPath} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="HTTP port"><TextInput value={httpPort} onValueChange={setHttpPort} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Workers"><TextInput value={workerGoal} onValueChange={setWorkerGoal} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Slot save path"><TextInput value={slotSavePath} onValueChange={setSlotSavePath} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Checkpoint file"><TextInput value={checkpointFile} onValueChange={setCheckpointFile} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Checkpoint seconds"><TextInput value={checkpointInterval} onValueChange={setCheckpointInterval} disabled={busy} size="small" /></FormField>
                </Flex>
                <Flex gap="2">
                    <Button kind="secondary" size="small" onClick={dryRun} disabled={busy || Number(workerGoal) <= 0}>Preview admission</Button>
                    <Button kind="primary" color="brand" size="small" onClick={start} disabled={busy || readyRPC < Number(workerGoal)}>Start distributed inference</Button>
                    <Button kind="secondary" size="small" onClick={stop} disabled={busy || execution?.state !== 'running'}>Stop inference</Button>
                </Flex>
                {execution && <Text kind="body/regular/sm" className="text-subtle-color">{execution.jobId}: {execution.state} · {execution.rpcPeers} remote RPC peer{execution.rpcPeers === 1 ? '' : 's'} · port {execution.httpPort}</Text>}
                {plan && <Text kind="body/regular/sm" className="text-subtle-color">Admission ready: {plan.workers.join(', ')} · epoch {plan.epoch}</Text>}
                {error && <Text kind="body/regular/sm" className="text-error-color">{error}</Text>}
            </Stack>
        </div>
    )
}
