// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Strongly-typed contract for every logical service message exchanged between
 * the UI and the Electron preload service bridge.
 *
 * - `WsInvokeChannelMap`: request/response shape for each logical invoke channel.
 *   The UI invokes `window.pairApi`, preload forwards over IPC, and Electron
 *   main replies with either modular subprocess data or an explicit empty
 *   payload.
 *
 * - `WsPushChannelMap`: payload shape for every service-to-renderer push event.
 *
 * Both maps are the single source of truth; the preload transport and service
 * bridge handlers are generic over these keys so TypeScript catches a
 * mismatched channel or payload at compile time.
 *
 * Discipline:
 *   - One-off payloads are inlined, not named. Named shapes are only lifted
 *     when (a) a second file imports them or (b) inlining would duplicate the
 *     shape twice inside this file.
 *   - `export` is reserved for the handful of symbols consumed by the
 *     service bridge generics â€” nothing else.
 */
import type {
    EngineCommandPayload,
    EngineHubSearchResponse,
    EngineInitialState,
    EngineStatePatch
} from '@/shared/types/engine-api'
import type { EngineProgress, EngineType } from '@/shared/types/engines'
import type { AppInitialSnapshot, ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ServiceError } from '@/shared/types/errors'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodeItemMetrics } from '@/shared/types/metrics'
import type { Workload } from '@/shared/types/workloads'
import type { FabricStatus, FabricTrainingRequest, FabricTrainingExecution, FabricTrainingGroupStatus, FabricCheckpoint, FabricTrainingNode, FabricInferenceRequest, FabricExecution, FabricGroupRequest, FabricGroupPlan, FabricLogicalDeviceDescribe, FabricLogicalDevicePlanRequest, FabricLogicalDevicePlan, FabricLogicalDeviceStatus, FabricTransferRequest, FabricTransferStatus, FabricLogicalDeviceCommitRequest, FabricLogicalExecuteRequest, FabricLogicalExecuteResult, FabricLogicalDeviceSeedRequest } from '@/shared/types/fabric'
import type {
    AvailableNode,
    ClusterIdentityPayload,
    ClusterNode,
    Invite
} from '@/shared/types/cluster'

// -----------------------------------------------------------------------------
// Invoke channels â€” (request, response) pairs
// -----------------------------------------------------------------------------

/**
 * Snapshot returned by `nodes:get-initial`. Plain mutable types on the wire â€”
 * the server holds `DeepReadonly<T>` internally but JSON transport deep-clones
 * into plain objects on the client, so the wire contract is mutable.
 */
interface NodesInitialResponse {
    nodes: Record<string, NodeItem>
    fetchedNodes: boolean
}

export interface WsInvokeChannelMap {
    // App bootstrap
    'app:get-initial': { request: void; response: AppInitialSnapshot }

    // Nodes
    'nodes:get-initial': { request: void; response: NodesInitialResponse }
    // Remove a node from the cluster (cluster-manager `nodes:remove`) and drop any
    // local manual discovery entry for it in the same call. Distinct from the
    // `nodes:remove` push (state update broadcast).
    'nodes:remove-member': {
        request: { nodeId: string }
        response: { nodeId: string; removed: boolean }
    }

    // Discovery
    'discovery:get-nodes': { request: void; response: AvailableNode[] }

    // Cluster â€” PIN-pairing handshake (nvpair-cluster-manager)
    'cluster:get-initial': { request: void; response: ClusterInitialSnapshot }
    'cluster:invite-node': { request: { ipAddress: string }; response: Invite }
    'cluster:invite-status': { request: { inviteId: string }; response: Invite }
    'cluster:respond-to-invite': {
        request: { inviteId: string; accept: boolean; pin?: string }
        response: Invite
    }
    // Abort a still-pending outbound invite (the inviter's counterpart to a
    // joiner decline): tears down the pairing session, invalidates the PIN, and
    // best-effort notifies the joiner. Returns the updated `Invite` (`canceled`).
    'cluster:cancel-invite': { request: { inviteId: string }; response: Invite }
    // Dissolve a cluster that was auto-created solely to back an invite, when
    // that pairing failed and no peer joined (no-op otherwise).
    'cluster:abandon-if-solo': { request: void; response: null }

    // Distributed compute fabric
    'fabric:get-status': { request: void; response: FabricStatus }
    'fabric:logical-device-describe': { request: void; response: FabricLogicalDeviceDescribe }
    'fabric:logical-device-plan': { request: FabricLogicalDevicePlanRequest; response: FabricLogicalDevicePlan }
    'fabric:logical-device-status': { request: { planId: string }; response: FabricLogicalDeviceStatus }
    'fabric:logical-device-transfer': { request: FabricTransferRequest; response: FabricTransferStatus }
    'fabric:logical-device-recover': { request: { planId: string; workers: string[] }; response: FabricLogicalDevicePlan }
    'fabric:logical-device-commit': { request: FabricLogicalDeviceCommitRequest; response: FabricLogicalDeviceStatus }
    'fabric:logical-device-seed': { request: FabricLogicalDeviceSeedRequest; response: { pageId: string; bytes: number; digest: string } }
    'fabric:logical-device-execute': { request: FabricLogicalExecuteRequest; response: FabricLogicalExecuteResult }
    'fabric:plan': { request: FabricGroupRequest; response: FabricGroupPlan }
    'fabric:inference-start': { request: FabricInferenceRequest; response: FabricExecution }
    'fabric:inference-status': { request: { jobId: string }; response: FabricExecution }
    'fabric:inference-stop': { request: { jobId: string }; response: { stopped: boolean } }
    'fabric:training-start': { request: FabricTrainingRequest; response: FabricTrainingExecution[] }
    'fabric:training-status': { request: { jobId: string }; response: FabricTrainingGroupStatus }
    'fabric:training-checkpoint': { request: FabricCheckpoint; response: FabricCheckpoint }
    'fabric:training-recover': { request: { jobId: string; nodes: FabricTrainingNode[]; rendezvousEndpoint: string }; response: FabricTrainingExecution[] }
    'fabric:training-stop': { request: { jobId: string }; response: { stopped: boolean } }

    // Engines
    'engines:get-initial': { request: void; response: EngineInitialState }
    'engine:command': { request: EngineCommandPayload; response: null }
    'engine:search-hub': { request: { engineType: EngineType }; response: EngineHubSearchResponse }

    // Errors
    'errors:get-initial': { request: void; response: ServiceError[] }
    'errors:clear': { request: string; response: null }

    // Workloads
    'workloads:get-initial': { request: void; response: Record<string, Workload> }
}

export type WsInvokeChannel = keyof WsInvokeChannelMap
export type WsInvokeRequest<C extends WsInvokeChannel> = WsInvokeChannelMap[C]['request']
export type WsInvokeResponse<C extends WsInvokeChannel> = WsInvokeChannelMap[C]['response']

// -----------------------------------------------------------------------------
// Push channels â€” server â†’ client broadcast events (single payload per event)
// -----------------------------------------------------------------------------

export interface WsPushChannelMap {
    // Nodes / cluster state
    'nodes:upsert': NodeItem
    'nodes:remove': string
    /** Full cluster membership snapshot, pushed on every change (cluster-manager `nodes:changed`). */
    'nodes:changed': ClusterNode[]

    // Engines
    'engines:state-changed': EngineStatePatch
    'engines:progress-changed': EngineProgress
    'engines:progress-cleared': { key: string }

    // Metrics
    'metrics:update': NodeItemMetrics

    // Workloads. Removal carries the origin node (`originatedFrom`) because
    // workload ids are a per-node proxy counter (the catalog is keyed by the
    // (originatedFrom, id) pair).
    'workloads:upsert': Workload
    'workloads:remove': { workloadId: string; originatedFrom: string | null }

    // Errors
    'errors:update': ServiceError[]

    // Cluster flow â€” an inbound pairing arrived; prompt the user for the PIN.
    'cluster:invite-received': Invite
    // Full authoritative snapshot of every live inbound invite awaiting a PIN,
    // re-emitted by Electron main on every add/prune (arrival, paired, decline,
    // status-poll terminal state, or client TTL expiry).
    'cluster:pending-invites-changed': Invite[]

    // Discovery
    'discovery:nodes-changed': AvailableNode[]

    // Connection lifecycle
    'connection:cluster-identity': ClusterIdentityPayload
    'state:request-refresh': void
}

export type WsPushChannel = keyof WsPushChannelMap
export type WsPushPayload<C extends WsPushChannel> = WsPushChannelMap[C]
