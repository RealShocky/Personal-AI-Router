<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Windows Distributed Hive Design

## Objective

Make a Windows PAIR node operate as the local control plane for a trusted,
heterogeneous compute fabric containing Windows, Linux/DGX, macOS, CUDA, Metal,
and CPU workers. A distributed inference job should see the fabric as one
logical runtime while ordinary operating-system applications continue to see
their local devices normally.

## Scope and reality boundary

The fabric is a runtime-level logical device, not an OS-level virtual GPU or
NUMA CPU. A kernel driver, CUDA DLL, or WinUI frontend cannot make arbitrary
Windows applications transparently use remote Ethernet devices. Custom native
code is appropriate inside the distributed runtime only, where it can explicitly
place tensors, exchange activations, and recover from worker loss.

WinUI 3 is not required for this boundary. Electron already provides the
cross-platform operator UI and typed service bridge. A Windows-native service
and optional tray integration can be added without replacing the renderer.

## Components

- `nvpair-compute-fabric`: broker-supervised coordinator/worker service. It
  owns leases, worker capability snapshots, group membership, job epochs, and
  recovery state.
- llama.cpp RPC/GGUF: the first managed runtime adapter. PAIR keeps
  `ggml-rpc-server` loopback-only and wraps its stream in the cluster-mTLS
  fabric CONNECT tunnel; later adapters may target another runtime.
- Existing `nvpair-cluster-manager`: supplies node identity and mutual TLS.
- Existing `nvpair-node-info`: supplies hardware facts; the fabric verifies
  runtime readiness separately from hardware presence.
- Existing Electron UI: displays fabric state and starts/stops explicit groups;
  it does not implement scheduling or tensor movement.

## Worker lifecycle

```text
discovered -> authenticated -> probing -> ready
ready -> leased -> serving
serving -> suspect -> quarantined
quarantined -> probing -> ready
```

Workers send signed/mTLS heartbeats containing node UUID, worker UUID, runtime
version, backend, model/shard inventory, free memory, queue depth, and epoch.
The coordinator expires a lease after missed heartbeats, stops assigning new
work, and marks the worker suspect before quarantine. Rejoining requires a fresh
handshake and capability probe; stale worker epochs cannot re-enter a running
job.

## Job recovery

Distributed jobs are split into resumable stages with an epoch, group ID, model
digest, and stage checkpoint. A worker loss causes the coordinator to stop the
affected stage, quarantine the worker, rebuild the group from ready members, and
resume from the last valid checkpoint when the runtime supports it. If the
runtime cannot restore a stage safely, the job fails explicitly; it is never
silently replayed with potentially duplicated output.

## Heterogeneous placement

The coordinator treats CUDA, Metal, and CPU as different backend capabilities.
Mixed groups are opt-in and must pass a compatibility probe. A default group is
homogeneous because tensor-parallel communication across a DGX, RTX 5060, and
MacBook can be slower than local inference. The scheduler considers memory,
backend, measured bandwidth, latency, and reliability rather than GPU names
alone.

## Native runtime boundary

No custom kernel or DLL is added until a measured bottleneck requires it. The
first implementation uses an upstream runtime adapter. If profiling proves a
missing operation or transport is the bottleneck, native code will be isolated
behind a versioned adapter with Windows x64 builds, Linux arm64/x64 builds where
needed, explicit ABI tests, and no direct exposure of unauthenticated RPC.

## Windows operation

The Windows node runs the coordinator as a PAIR-supervised service. Linux and
macOS nodes run workers or coordinators using the same protocol. The Windows
desktop remains the operator surface; a future WinUI 3 companion is optional and
must not become a second control plane.

## Delivery stages

1. Define the fabric wire contract and worker state machine.
2. Add the broker-supervised fabric service with authenticated heartbeat,
   quarantine, and rejoin behavior, using synthetic workers in tests.
3. Add real capability probes and the llama.cpp RPC adapter lifecycle.
4. Add explicit homogeneous distributed groups and checkpointed stages.
5. Add measured mixed CUDA/Metal/CPU experiments.
6. Add batch fan-out and broad 50-node CPU fleet utilization.

## Non-goals

- transparent remote memory for arbitrary Windows programs;
- replacing Electron with WinUI 3;
- claiming zero latency during worker failure;
- automatically mixing incompatible devices into a production group;
- writing a new neural-network framework before profiling proves it necessary.
