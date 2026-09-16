<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvpair-compute-fabric

Current worker heartbeats include `peerId`, the node's cluster-certificate
principal used for exact RPC relay pinning. Older workers may omit it; the
coordinator temporarily falls back to the display `nodeId` compatibility path
while still requiring a certificate pinned in the current cluster. New worker
installations advertise the explicit principal automatically.

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

When `nvidia-smi` exposes memory accounting, worker heartbeats also report
aggregate NVIDIA GPU count and free/total VRAM. If the device is visible but
memory is `Not Supported`, PAIR still counts the GPU and leaves VRAM unknown;
that worker cannot satisfy a minimum-VRAM placement constraint. These are
telemetry and scheduling hints; they do not combine physical adapters into one
operating-system device.

When the fabric process runs outside the broker, pass `--daemon`. This keeps
the HTTP worker, relay, and child-process supervisors alive without requiring a
JSON-RPC stdin session; the fleet installer sets this automatically.

Group admission can require an exact model digest, checkpoint support, and a maximum measured probe latency and minimum free host memory. The coordinator also exposes explicit group planning and durable job controls
over its broker JSON-RPC surface: `fabric:plan-group`, `fabric:job-submit`,
`fabric:checkpoint`, `fabric:recover`, `fabric:job-start`, `fabric:job-stop`, `fabric:job-status`, `fabric:get-status`, and `fabric:build-execution-plan`. The latter returns persisted jobs and live executions for the operator UI. `fabric:build-execution-plan` converts an admitted group into a versioned plan with explicit shard strategy, transport, per-worker memory budgets, checkpoint policy, and failover policy. When a checkpoint includes a slot and filename, the adapter uses llama.cpp slot save/restore endpoints and `--slot-save-path`. Checkpoints are monotonic within an
epoch and are persisted under the broker's per-user `fabric/` state directory.

The training adapter exposes `fabric:build-training-command`. It validates a
`TrainingRequest` and builds a `torchrun` command using static c10d/TCPStore
rendezvous, explicit node rank, FSDP or data parallelism, model digest, dataset, and
checkpoint arguments. The current adapter admits homogeneous CPU or CUDA
worlds. Metal and mixed-backend training are rejected until a collective
runtime with those semantics is installed and probed; Metal remains available
for inference adapters.

Worker-mode HTTP endpoints `/v1/fabric/training/start`,
`/v1/fabric/training/status`, and `/v1/fabric/training/stop` supervise the
validated `torchrun` process over the same pinned mTLS fabric. The endpoint
does not execute a caller-provided shell string: the service constructs the
argument vector and invokes `torchrun` directly. Set the PAIR-owned
`PAIR_TORCHRUN_PATH` environment variable when a node needs a wrapper, such
as a containerized ARM64 launcher on DGX; the wrapper receives the same
structured argument vector and must not evaluate it through a shell.

Training status includes an optional `checkpoint` execution record. A trainer
publishes a JSON manifest named `pair-canary-manifest.json` in the configured
checkpoint directory with a numeric `step` and relative `checkpoint` filename.
The worker only reports the checkpoint when the filename is safe and the file
exists. The coordinator uses this record to resume a replacement training
world after a rank failure.

The coordinator JSON-RPC method `fabric:training-start` fans a validated
request out to every listed rank concurrently. If one rank fails to start, it
stops the ranks that did start and returns an error; it never leaves a partial
training world running.

Before that fan-out, the coordinator checks each listed worker against its
current ready heartbeat, advertised backend, endpoint, and GPU count. When a
worker reports a non-empty model inventory, the requested model digest must be
present there as well. This prevents a stale endpoint or incompatible worker
from entering a collective; an empty model inventory remains an explicitly
unknown capability rather than a false mismatch.

For coordinator-launched inference, `modelDigest` identifies the model in the
execution plan but is not a requirement that every worker advertise a local
copy. The coordinator owns the model path and llama.cpp's RPC layer streams
the required model data to selected workers. Exact local model-digest matching
continues to apply to distributed training, where every rank must load the
same training artifact.

Inference placement also accepts `minAggregateMemoryFreeBytes`,
`minAggregateGpuVramTotalBytes`, and `minAggregateGpuVramFreeBytes`. These
requirements are checked against the exact selected workers, and the returned
group plan includes the selected aggregate capacity for operator inspection.
They are logical scheduling totals, not a new operating-system memory device.

The coordinator also exposes the inference control endpoints
`POST /v1/fabric/inference/start`, `GET /v1/fabric/inference/status?jobId=...`,
and `POST /v1/fabric/inference/stop?jobId=...`. They require the same pinned
cluster mTLS identity as the worker endpoints and call the same admission,
checkpoint, relay, and supervision path as `fabric:job-start`. This makes the
fabric usable by a headless operator without duplicating scheduler logic.

`fabric:training-status` refreshes each assigned rank over pinned mTLS before
returning the rank list and group state, while
`fabric:training-stop` stops every rank in the group. Group state is held by the
coordinator and is intentionally separate from the operating-system device
inventory. `fabric:training-checkpoint` persists a monotonic checkpoint
manifest tied to the group epoch; after a coordinator restart, a running group
is loaded as `recoverable` rather than being falsely reported as active.
`fabric:training-recover` accepts replacement workers and a fresh rendezvous,
requires that checkpoint, and relaunches with `--resume` under a new epoch.
Recovery stops after three failed attempts instead of looping forever.
The coordinator can automatically perform that recovery after a status refresh
when a recovery-node provider is configured. It selects ready workers with the
same backend, excludes the failed group's current workers, and requires the
replacement count to match the original world. Without a verified checkpoint
or a complete replacement set, the group remains `recoverable` for an operator
to inspect; uncheckpointed work is never silently replayed.

CUDA and Metal are capability labels for group planning. `fabric:job-start` first creates and validates a tensor-sharding execution plan, then creates one loopback relay per advertised worker and launches a single logical llama.cpp server with its `--rpc` list. The llama.cpp RPC adapter is the first actual execution path; other runtimes still require
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

When the server is launched through Windows `wsl.exe`, the service discovers
the WSL default-route gateway and binds dynamic relays so the Linux child can
reach them. Dynamic relays run their accept loop in the execution manager and
pin the outgoing cluster certificate to the selected worker identity. Failed
inference recovery is limited to three attempts; after that the job is marked
failed and must be inspected or restarted by an operator.

For several workers, use one process with
`--rpc-relay-specs "127.0.0.1:51001|https://worker-a:14324/v1/fabric/rpc;127.0.0.1:51002|https://worker-b:14324/v1/fabric/rpc"`
and pass both local relay addresses in `--llama-rpc`.

Inference status includes an operator-facing lifecycle record. `phase` is
`starting` until the supervised HTTP `/health` endpoint answers successfully,
then becomes `serving`; a worker loss changes it to `recovering` while a fresh
group epoch is planned. Terminal process outcomes are `failed` or `stopped`.
The record also includes the selected worker IDs, plan epoch, recovery attempt
count, latest checkpoint stage/file, and an RFC3339 `updatedAt` timestamp. The
`message` field contains short operational text only; prompt, response,
credential, and key material are never included.
The explicit `--rpc-relay-peer-id` form is preferred for a standalone relay when the cluster contains more than one pinned peer.

## Logical device planning

`fabric:logical-device-plan` creates an explicit PAIR logical-device plan from
ready worker capabilities. The request includes `protocolVersion`, a unique
`requestId`, `runtime`, `modelDigest`, `shardStrategy`, `workerGoal`, optional
`providers`, and page specifications. The response contains an epoch, worker
assignments, and page placements. A placement identifies a PAIR worker and
memory tier; it is not a CUDA or Metal pointer and does not create contiguous
VRAM.

Example request:

```json
{
  "protocolVersion": 1,
  "requestId": "plan-request-1",
  "runtime": "pair",
  "modelDigest": "sha256:model",
  "shardStrategy": "pipeline",
  "providers": ["cpu", "cuda", "metal"],
  "workerGoal": 2,
  "pages": [{"pageId": "weights-0", "bytes": 1048576, "dtype": "bf16", "layout": "row-major"}]
}
```

The same `requestId` is idempotent and returns the original plan. Later logical
device methods operate on that plan's ID and epoch; an old epoch is rejected
rather than applied to a newer plan.

`fabric:logical-device-describe` returns the currently advertised CPU, CUDA,
and Metal provider capabilities. `fabric:logical-device-status` returns the
authoritative plan epoch, page owners, and transfer IDs. A
`fabric:logical-device-transfer` admits a transfer. When the coordinator has a
configured cluster directory and pinned client identity, it streams the page
to the target worker and returns only after peer metadata verifies the digest;
without that runtime configuration it remains visibly queued for a provider
adapter to execute.

`fabric:logical-device-recover` accepts a degraded plan ID and an exact
replacement worker list. It creates a newer epoch and moves page ownership to
the replacement set only after those workers are ready. Provider execution and
checkpoint verification must still complete before the plan is considered
running.

Provider adapters acknowledge a completed page with
`fabric:logical-device-transfer-complete`, supplying the transfer ID and the
destination digest. The coordinator verifies the digest before publishing the
new owner; this endpoint carries control metadata and never accepts a raw
provider pointer.

The binary page channel is the authenticated `POST /v1/fabric/logical-page`
endpoint. The sender supplies `X-PAIR-Page-ID`, `X-PAIR-Page-Bytes`, and
`X-PAIR-Page-Digest` headers with an `application/octet-stream` body. The peer
streams into its local page store, verifies the exact size and SHA-256 digest,
atomically publishes the page, and returns bounded JSON metadata. `GET` on the
same endpoint retrieves a verified page by `pageId`; both methods require the
existing pinned cluster mTLS identity.

## Test

```bash
go test ./...
```
