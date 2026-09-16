// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { FabricStatus, FabricTrainingGroupStatus, FabricTrainingParallelism } from '@/shared/types/fabric'
import { compatibleTrainingNodes } from './fabric-training'

const EMPTY_STATUS: FabricStatus = { workers: [], capacity: { workers: 0, memoryFree: 0, gpuVramTotal: 0, gpuVramFree: 0, gpuCount: 0 }, jobs: [], executions: [] }

export default function FabricTrainingCard() {
    const [fabric, setFabric] = useState<FabricStatus>(EMPTY_STATUS)
    const [training, setTraining] = useState<FabricTrainingGroupStatus | null>(null)
    const [jobId, setJobId] = useState('pair-training-1')
    const [modelDigest, setModelDigest] = useState('')
    const [trainerPath, setTrainerPath] = useState('train.py')
    const [modelPath, setModelPath] = useState('')
    const [datasetPath, setDatasetPath] = useState('')
    const [checkpointDirectory, setCheckpointDirectory] = useState('checkpoints')
    const [rendezvousEndpoint, setRendezvousEndpoint] = useState('')
    const [parallelism, setParallelism] = useState<FabricTrainingParallelism>('fsdp')
    const [processesPerNode, setProcessesPerNode] = useState('1')
    const [checkpointInterval, setCheckpointInterval] = useState('100')
    const [checkpointFile, setCheckpointFile] = useState('')
    const [checkpointStage, setCheckpointStage] = useState('1')
    const [busy, setBusy] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        let active = true
        const refresh = async () => {
            try {
                const nextFabric = await window.pairApi.fabric.getStatus()
                if (active) setFabric(nextFabric)
                if (jobId.trim()) {
                    const nextTraining = await window.pairApi.fabric.getTrainingStatus(jobId.trim())
                    if (active) setTraining(nextTraining)
                }
            } catch {
                if (active) setTraining(null)
            }
        }
        void refresh()
        const timer = window.setInterval(() => void refresh(), 5000)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [jobId])

    const nodes = compatibleTrainingNodes(fabric)
    const run = async (operation: () => Promise<void>) => {
        setBusy(true)
        setError('')
        try {
            await operation()
        } catch (cause) {
            setError(cause instanceof Error ? cause.message : 'Training operation failed')
        } finally {
            setBusy(false)
        }
    }

    const start = () =>
        run(async () => {
            const next = await window.pairApi.fabric.startTraining({
                jobId: jobId.trim(),
                modelDigest: modelDigest.trim(),
                trainerPath: trainerPath.trim(),
                modelPath: modelPath.trim(),
                datasetPath: datasetPath.trim(),
                checkpointDirectory: checkpointDirectory.trim(),
                rendezvousEndpoint: rendezvousEndpoint.trim(),
                parallelism,
                processesPerNode: Number(processesPerNode),
                checkpointIntervalSteps: Number(checkpointInterval),
                nodes
            })
            setTraining({ jobId: jobId.trim(), state: 'running', epoch: 0, executions: next, checkpoint: { jobId: jobId.trim(), groupId: jobId.trim(), epoch: 0, stage: 0, filename: '' }, recoveryAttempts: 0 })
        })

    const stop = () => run(async () => { await window.pairApi.fabric.stopTraining(jobId.trim()) })

    const saveCheckpoint = () =>
        run(async () => {
            if (!training) throw new Error('No active training group')
            const next = await window.pairApi.fabric.saveTrainingCheckpoint({ jobId: jobId.trim(), groupId: jobId.trim(), epoch: training.epoch, stage: Number(checkpointStage), filename: checkpointFile.trim() })
            setTraining({ ...training, checkpoint: next })
        })

    const recover = () =>
        run(async () => {
            if (!training?.checkpoint.filename) throw new Error('Save a verified checkpoint before recovery')
            const next = await window.pairApi.fabric.recoverTraining(jobId.trim(), nodes, rendezvousEndpoint.trim())
            setTraining({ ...training, state: 'running', executions: next, recoveryAttempts: training.recoveryAttempts + 1 })
        })

    return (
        <div className="node-card pair-paper mb-3" data-fabric-training>
            <Stack gap="2">
                <Flex align="center" justify="between" gap="2">
                    <Text kind="body/semibold/sm">Distributed training</Text>
                    <Text kind="body/regular/sm" className="text-subtle-color">{nodes.length} compatible worker{nodes.length === 1 ? '' : 's'}</Text>
                </Flex>
                <Text kind="body/regular/sm" className="text-subtle-color">Runs an explicit torchrun world; it does not create one OS-level GPU.</Text>
                <Flex gap="2" wrap="wrap">
                    <FormField slotLabel="Job ID"><TextInput value={jobId} onValueChange={setJobId} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Model digest"><TextInput value={modelDigest} onValueChange={setModelDigest} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Trainer"><TextInput value={trainerPath} onValueChange={setTrainerPath} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Model path"><TextInput value={modelPath} onValueChange={setModelPath} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Dataset path"><TextInput value={datasetPath} onValueChange={setDatasetPath} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Checkpoint directory"><TextInput value={checkpointDirectory} onValueChange={setCheckpointDirectory} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Rendezvous"><TextInput value={rendezvousEndpoint} onValueChange={setRendezvousEndpoint} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Parallelism"><TextInput value={parallelism} onValueChange={value => setParallelism(value === 'data' ? 'data' : 'fsdp')} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Processes/node"><TextInput value={processesPerNode} onValueChange={setProcessesPerNode} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Checkpoint steps"><TextInput value={checkpointInterval} onValueChange={setCheckpointInterval} disabled={busy} size="small" /></FormField>
                </Flex>
                <Flex gap="2" wrap="wrap">
                    <Button kind="primary" color="brand" size="small" onClick={start} disabled={busy || nodes.length === 0}>Start training</Button>
                    <Button kind="secondary" size="small" onClick={stop} disabled={busy || training?.state !== 'running'}>Stop group</Button>
                    <Button kind="secondary" size="small" onClick={recover} disabled={busy || !training?.checkpoint.filename || nodes.length === 0}>Recover from checkpoint</Button>
                </Flex>
                <Flex gap="2" wrap="wrap">
                    <FormField slotLabel="Checkpoint file"><TextInput value={checkpointFile} onValueChange={setCheckpointFile} disabled={busy} size="small" /></FormField>
                    <FormField slotLabel="Checkpoint stage"><TextInput value={checkpointStage} onValueChange={setCheckpointStage} disabled={busy} size="small" /></FormField>
                    <Button kind="tertiary" size="small" onClick={saveCheckpoint} disabled={busy || !training}>Save checkpoint</Button>
                </Flex>
                {training && <Text kind="body/regular/sm" className="text-subtle-color">{training.jobId}: {training.state} · epoch {training.epoch} · {training.executions.length} rank{training.executions.length === 1 ? '' : 's'} · recovery attempts {training.recoveryAttempts}{training.checkpoint.filename ? ` · checkpoint ${training.checkpoint.filename}` : ''}</Text>}
                {error && <Text kind="body/regular/sm" className="text-error-color">{error}</Text>}
            </Stack>
        </div>
    )
}
