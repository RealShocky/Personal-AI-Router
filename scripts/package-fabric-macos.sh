#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
BINARY="${PAIR_FABRIC_BINARY:-$ROOT/services/nvpair-compute-fabric/nvpair-compute-fabric}"
CLUSTER_MANAGER="${PAIR_CLUSTER_MANAGER_BINARY:-$ROOT/services/nvpair-cluster-manager/nvpair-cluster-manager}"
[[ "$(uname -s)" == Darwin ]] || { echo "Run this on macOS or set PAIR_FABRIC_BINARY to a macOS build" >&2; exit 1; }
[[ -f "$BINARY" ]] || { echo "Build macOS first or set PAIR_FABRIC_BINARY" >&2; exit 1; }
command -v zip >/dev/null || { echo "zip is required" >&2; exit 1; }
STAGE="$OUT/fabric-macos-$(uname -m)"; ZIPFILE="$STAGE.zip"
rm -rf "$STAGE" "$ZIPFILE"; mkdir -p "$STAGE"
cp "$BINARY" "$STAGE/nvpair-compute-fabric"; cp "$ROOT/scripts/install-fabric-worker-macos.sh" "$STAGE/"
if [[ -f "$CLUSTER_MANAGER" ]]; then cp "$CLUSTER_MANAGER" "$STAGE/nvpair-cluster-manager"; fi
cp "$ROOT/scripts/pair-training-canary.py" "$STAGE/"
cp "$ROOT/docs/distributed-fabric.mdx" "$STAGE/"
cp "$ROOT/docs/fabric-operations.mdx" "$STAGE/"
cp "$ROOT/docs/README.md" "$STAGE/PAIR-documentation.md"
printf '%s\n' 'PAIR Fabric portable macOS worker' 'Run install-fabric-worker-macos.sh after setting PAIR_* variables.' 'An optional nvpair-cluster-manager binary supports first-time pairing on a headless node.' 'No identities, keys, or models are included.' > "$STAGE/README.txt"
printf '%s\n' 'PAIR_COORDINATOR_URL=https://coordinator.example:14324' 'PAIR_WORKER_ID=mac-worker-01' 'PAIR_NODE_ID=MAC-01' 'PAIR_CLUSTER_DIR=$HOME/Library/Application Support/PAIR/cluster' 'PAIR_RUNTIME=auto' 'PAIR_BACKEND=' > "$STAGE/fabric-worker.env.example"
chmod +x "$STAGE/nvpair-compute-fabric" "$STAGE/install-fabric-worker-macos.sh"
if [[ -f "$STAGE/nvpair-cluster-manager" ]]; then
  (cd "$STAGE" && shasum -a 256 nvpair-compute-fabric nvpair-cluster-manager > SHA256SUMS)
else
  (cd "$STAGE" && shasum -a 256 nvpair-compute-fabric > SHA256SUMS)
fi
(cd "$OUT" && zip -q -r "$(basename "$ZIPFILE")" "$(basename "$STAGE")")
echo "Created $ZIPFILE"
