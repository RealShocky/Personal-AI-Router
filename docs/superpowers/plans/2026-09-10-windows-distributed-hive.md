<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Windows Distributed Hive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows-first, authenticated distributed compute fabric with worker health, automatic quarantine/rejoin, and a runtime adapter boundary for pooled inference.

**Architecture:** Add a broker-supervised Go fabric service that coordinates explicit worker leases and resumable job epochs. Keep Electron as the operator UI, use the existing cluster mTLS identity, and integrate llama.cpp behind an adapter rather than writing a kernel prematurely.

**Tech Stack:** Go 1.25, existing PAIR JSON-RPC/stdio supervision, existing cluster mTLS, existing node-info capability data, llama.cpp adapter process, Electron/TypeScript bridge.

**Spec:** `docs/superpowers/specs/2026-09-10-windows-distributed-hive-design.md`

## Global Constraints

- The fabric is a runtime-level logical device, never a transparent OS-level shared GPU or CPU.
- Existing Ollama/LM Studio routing remains compatible.
- No unauthenticated distributed-runtime listener is exposed.
- Worker loss stops new assignments before quarantine and never silently duplicates output.
- Mixed CUDA/Metal/CPU groups are opt-in and require a compatibility probe.
- No custom kernel or DLL is introduced without a measured runtime bottleneck.
- All changed files retain SPDX headers and changed service binaries receive version bumps.

---

### Task 1: Define fabric protocol and state machine

**Files:**
- Create: `services/shared/fabricwire/fabricwire.go`
- Create: `services/shared/fabricwire/fabricwire_test.go`
- Create: `services/nvpair-compute-fabric/README.md`

- [x] Define worker states, heartbeat, capability, lease, group, epoch, and checkpoint payloads.
- [x] Test JSON round trips, stale epoch rejection, and legal/illegal state transitions.
- [x] Document the protocol and security assumptions.

### Task 2: Add coordinator service and broker supervision

**Files:**
- Create: `services/nvpair-compute-fabric/*.go`
- Modify: `services/nvpair-ui-broker/broker.go`
- Modify: `desktop/src/shared/constants/modular-binaries.ts`
- Modify: `services/versions.json`
- Test: coordinator unit and broker supervision tests

- [ ] Add failing tests for lease expiry, suspect/quarantine, fresh-epoch rejoin, and no-new-work assignment after failure.
- [x] Add the minimal coordinator state machine and stdio JSON-RPC surface.
- [x] Register the service as a broker-supervised optional worker.
- [x] Verify broker startup remains non-fatal when the fabric service is unavailable.

### Task 3: Add authenticated worker handshake and rejoin

**Files:**
- Modify: `services/nvpair-compute-fabric/*.go`
- Modify: `services/nvpair-cluster-manager` only for a narrowly scoped fabric identity/pin relay if required
- Test: `services/tests/fabric_interop_test.go`

- [x] Add the cluster-mTLS endpoint and worker heartbeat/rejoin protocol.
- [x] Reject unpaired clients at the fabric trust gate.
- [x] Test missed heartbeats, quarantine, fresh epoch rejoin, and ready-pool re-entry.
- [x] Test that a stale worker epoch cannot resume an old job.

### Task 4: Add capability-aware group formation

**Files:**
- Modify: `services/nvpair-compute-fabric/*.go`
- Modify: `services/nvpair-node-info` capability contract if probe data is missing
- Modify: desktop bridge types and state
- Test: group compatibility and placement tests

- [x] Test capability-aware homogeneous group formation and CPU-only eligibility.
- [ ] Add measured Metal/CUDA runtime probes and network-performance admission.
- [x] Reject incompatible groups by default.
- [ ] Include measured memory, latency, bandwidth, and reliability in placement decisions.

### Task 5: Add llama.cpp adapter boundary

**Files:**
- Create: `services/fabric-llama-adapter/README.md`
- Create: `services/fabric-llama-adapter/*`
- Modify: broker/fabric service registration files
- Test: adapter process and protocol tests

- [x] Require explicitly configured llama.cpp binaries rather than downloading arbitrary executables.
- [x] Start and stop supervised `ggml-rpc-server` and `llama-server` processes with explicit arguments.
- [x] Keep raw RPC ports loopback-only behind the cluster-mTLS CONNECT tunnel.
- [x] Probe model digest, backend, memory, and checkpoint support before group admission.

### Task 6: Add resumable distributed jobs

**Files:**
- Modify: `services/nvpair-compute-fabric/*`
- Modify: `services/shared/fabricwire/*`
- Modify: desktop API/bridge and workload types
- Test: checkpoint/resume and worker-loss tests

- [x] Add durable stage checkpoint metadata and epoch validation.
- [x] Stop assignment eligibility after worker loss/quarantine.
- [x] Add coordinator recovery records for a new group and epoch.
- [x] Connect recovery records to llama.cpp slot restoration and mark unexpected runtime exits failed.

### Task 7: Update operator documentation

**Files:**
- Modify: `docs/overview.mdx`
- Modify: `docs/architecture.mdx`
- Modify: `docs/getting-started.mdx`
- Modify: `docs/building.mdx`
- Modify: `docs/troubleshooting.mdx`
- Create: `docs/distributed-fabric.mdx`

- [ ] Document Windows coordinator setup and Linux/macOS worker setup.
- [ ] Document pairing, firewall requirements, health states, rejoin behavior, and limitations.
- [ ] Document the difference between ordinary routing, batch fan-out, and distributed model execution.
- [ ] Document that WinUI 3 and custom kernels are not prerequisites.
