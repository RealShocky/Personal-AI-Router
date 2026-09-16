<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-compute-fabric

This module contains the coordinator and worker heartbeat implementation for
PAIR's distributed compute fabric. The Windows broker supervises it as an
optional worker and restarts it if it exits. The current milestone validates
lease expiry, quarantine, assignment gating, fresh-epoch rejoin, and an
authenticated cluster-mTLS HTTP endpoint.

Worker mode computes a SHA-256 model digest from `--llama-model` (or accepts `--model-digest`), reports checkpoint support, and measures coordinator round-trip latency for admission. The fabric is a runtime-level logical compute group. It does not create a
transparent operating-system CPU, GPU, or VRAM device. The coordinator endpoint
is advertised as service `cf` on port `14324` and accepts only requests from a
currently trusted cluster member. A node can also run this binary in worker
mode with `--coordinator-url`, `--worker-id`, `--node-id`, `--advertise-url`, and `--cluster-dir`;
it sends heartbeats and performs a fresh-epoch rejoin after quarantine.

When `nvidia-smi` is available, worker heartbeats also report aggregate NVIDIA
GPU count and free/total VRAM. These are telemetry and scheduling hints; they
do not combine physical adapters into one operating-system device.

When the fabric process runs outside the broker, pass `--daemon`. This keeps
the HTTP worker, relay, and child-process supervisors alive without requiring a
JSON-RPC stdin session; the fleet installer sets this automatically.

Group admission can require an exact model digest, checkpoint support, and a maximum measured probe latency and minimum free host memory. The coordinator also exposes explicit group planning and durable job controls
over its broker JSON-RPC surface: `fabric:plan-group`, `fabric:job-submit`,
`fabric:checkpoint`, `fabric:recover`, `fabric:job-start`, `fabric:job-stop`, `fabric:job-status`, `fabric:get-status`, and `fabric:build-execution-plan`. The latter returns persisted jobs and live executions for the operator UI. `fabric:build-execution-plan` converts an admitted group into a versioned plan with explicit shard strategy, transport, per-worker memory budgets, checkpoint policy, and failover policy. When a checkpoint includes a slot and filename, the adapter uses llama.cpp slot save/restore endpoints and `--slot-save-path`. Checkpoints are monotonic within an
epoch and are persisted under the broker's per-user `fabric/` state directory.

CUDA and Metal are capability labels for group planning. `fabric:job-start` creates one loopback relay per advertised worker and launches a single logical llama.cpp server with its `--rpc` list. The llama.cpp RPC adapter is the first actual execution path; other runtimes still require
their own adapter and checkpoint semantics.

For a real llama.cpp group, run a loopback `ggml-rpc-server` on each worker,
start this service there with `--rpc-target 127.0.0.1:<rpc-port>`, and start a
local relay on the coordinator for each worker:

```text
nvpair-compute-fabric.exe --cluster-dir C:\...\cluster --http-port 14324 \
  --rpc-target 127.0.0.1:50052
nvpair-compute-fabric.exe --cluster-dir C:\...\cluster \
  --rpc-relay-listen 127.0.0.1:51001 \
  --rpc-relay-url https://worker-address:14324/v1/fabric/rpc
  --rpc-relay-peer-id worker-uuid
nvpair-compute-fabric.exe --llama-server-path C:\...\llama-server.exe \
  --llama-model C:\...\model.gguf --llama-port 11450 \
  --llama-rpc 127.0.0.1:51001
```

The first command is the worker-side mTLS gate, the second is the
coordinator-side local raw-RPC relay, and the third starts the actual
OpenAI-compatible llama.cpp server. The RPC server and relay are loopback/raw
only; only PAIR's pinned mTLS endpoint crosses the network.

For several workers, use one process with
`--rpc-relay-specs "127.0.0.1:51001|https://worker-a:14324/v1/fabric/rpc;127.0.0.1:51002|https://worker-b:14324/v1/fabric/rpc"`
and pass both local relay addresses in `--llama-rpc`.
The explicit `--rpc-relay-peer-id` form is preferred for a standalone relay when the cluster contains more than one pinned peer.

## Test

```bash
go test ./...
```
