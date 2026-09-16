<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Distributed Hardware Fabric Design

## Goal

Extend PAIR from node-level telemetry into a capability-aware hardware fabric
that can eventually coordinate CPU, NVIDIA CUDA, and Apple Metal workers for
distributed inference, while keeping ordinary Ollama and LM Studio routing
unchanged until a specialized runtime is explicitly enabled.

## Reality boundary

The fabric represents a distributed logical compute group; it does not claim to
create one transparent operating-system CPU, GPU, or VRAM device. Same-host
unified memory and vendor fabrics remain runtime-specific. Cross-host work must
use an inference runtime that explicitly supports remote execution.

## Phase 1 scope

Phase 1 adds a versioned hardware-capability description to `nvpair-node-info`
and carries it through the existing node-info polling path. It reports CPU,
RAM, GPU vendor/backend, accelerator memory, and conservative capability labels.
It does not add llama.cpp, distributed model execution, new network listeners,
or changes to request placement.

The capability contract must distinguish:

- `cpu` — CPU execution is available;
- `cuda` — NVIDIA CUDA execution is available and locally detected;
- `metal` — Apple Metal execution is available and locally detected;
- `distributed_worker` — the node is eligible to be enrolled later as a worker;
- `distributed_coordinator` — the node can host a future coordinator.

Unknown capability is omitted rather than guessed. GPU presence alone must not
be treated as proof that a usable inference backend is installed.

## Data flow

`nvpair-node-info` detects static hardware once and merges dynamic memory and
utilization telemetry into the existing response. The broker's existing
node-info poller parses and stores the new optional fields. Existing clients that
ignore the fields remain compatible. The desktop can display the fields after
the bridge types are extended, but routing continues to use the existing model
inventory and scheduler.

## Later phases

1. Add capability-aware eligibility and scheduling without distributed pooling.
2. Add llama.cpp as an explicitly managed engine with local CPU/CUDA/Metal modes.
3. Add authenticated distributed worker registration and coordinator lifecycle.
4. Add distributed model groups, beginning with a small homogeneous CUDA group,
   then mixed CUDA/Metal/CPU experiments behind an opt-in feature flag.
5. Add batch fan-out across the larger CPU-only fleet.

Each later phase requires real hardware and network measurements before its
capability is advertised as production-ready.

## Security and failure rules

Raw distributed-runtime ports are never exposed to the LAN without PAIR trust.
Worker membership is explicit and mutual-TLS protected. A worker that becomes
stale, incompatible, or overloaded is removed from future groups; in-flight
distributed work is reported failed rather than silently re-routed.

## Success criteria for Phase 1

- CPU-only, CUDA, and Metal-capable nodes produce distinguishable reports.
- A no-GPU mini-PC reports CPU capability without fabricated VRAM.
- GPU inventory retains existing telemetry behavior.
- Old node-info consumers continue to parse responses.
- Unit tests cover capability classification and JSON compatibility.
- `go test ./...` passes in `services/nvpair-node-info` and the relevant
  cross-process contract checks remain green.
