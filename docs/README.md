<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PAIR documentation map

Use this page as the starting point for the Windows-first PAIR system and its
optional distributed compute fabric.

## Operators

- [Getting started](getting-started.mdx) — install PAIR, launch the desktop
  application, pair ordinary PAIR nodes, and send a first request.
- [Fabric operations](fabric-operations.mdx) — pair and deploy CPU, CUDA, and
  Metal workers, use the dashboard, package portable workers, and recover nodes.
- [Troubleshooting](troubleshooting.mdx) — diagnose discovery, pairing, engine,
  network, and fabric failures.
- [Log collection](log-collection.mdx) — collect support evidence without
  exposing prompts, credentials, PINs, or private keys.

## Architecture and development

- [Overview](overview.mdx) — product vocabulary and system boundaries.
- [Distributed fabric](distributed-fabric.mdx) — coordinator, worker, runtime
  adapters, logical pooling, and current verified paths.
- [Logical device architecture](logical-device.mdx) — deep boundary analysis
  and the PAIR tensor-device design for explicit heterogeneous memory sharing.
- [Architecture](architecture.mdx) — process model, trust boundaries, and ports.
- [Building and running](building.mdx) — prerequisites and source builds.
- [Developer guide](developing.mdx) — repository tour and contribution workflow.
- [Terminal interface](terminal-interface.mdx) — headless operation on Linux,
  DGX, AWS, and mini-PCs.

## The key boundary

PAIR can make supported inference work appear as one logical service and can
schedule work across heterogeneous workers. It cannot make Windows, Linux,
CUDA, or Metal expose one physical pooled CPU/GPU/VRAM device to the operating
system. CPU RAM can be used for model storage or offload, but it is not VRAM.
