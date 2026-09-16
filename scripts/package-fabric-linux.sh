#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
BINARY="${PAIR_FABRIC_BINARY:-$ROOT/services/nvpair-compute-fabric/nvpair-compute-fabric}"
ARCH="${PAIR_FABRIC_ARCH:-$(uname -m)}"
CLUSTER_MANAGER="${PAIR_CLUSTER_MANAGER_BINARY:-$ROOT/services/nvpair-cluster-manager/nvpair-cluster-manager}"
[[ -f "$BINARY" ]] || { echo "Build Linux first or set PAIR_FABRIC_BINARY" >&2; exit 1; }
command -v zip >/dev/null || { echo "zip is required" >&2; exit 1; }
case "$ARCH" in x86_64|amd64) ARCH_LABEL=x64 ;; aarch64|arm64) ARCH_LABEL=arm64 ;; *) echo "Unsupported PAIR_FABRIC_ARCH: $ARCH" >&2; exit 2 ;; esac
STAGE="$OUT/fabric-linux-$ARCH_LABEL"
ZIPFILE="$OUT/fabric-linux-$ARCH_LABEL.zip"
rm -rf "$STAGE" "$ZIPFILE"
mkdir -p "$STAGE"
cp "$BINARY" "$STAGE/nvpair-compute-fabric"
if [[ -f "$CLUSTER_MANAGER" ]]; then cp "$CLUSTER_MANAGER" "$STAGE/nvpair-cluster-manager"; fi
cp "$ROOT/scripts/install-fabric-worker.sh" "$STAGE/"
cp "$ROOT/scripts/pair-training-canary.py" "$STAGE/"
cp "$ROOT/scripts/install-pair-training-wsl.sh" "$STAGE/"
cp "$ROOT/scripts/pair-torchrun-dgx.sh" "$STAGE/"
cp "$ROOT/scripts/configure-nccl-ports.sh" "$STAGE/"
cp "$ROOT/docs/distributed-fabric.mdx" "$STAGE/"
cp "$ROOT/docs/fabric-operations.mdx" "$STAGE/"
cp "$ROOT/docs/README.md" "$STAGE/PAIR-documentation.md"
cat > "$STAGE/README.txt" <<EOF
PAIR Fabric portable Linux worker ($ARCH_LABEL)
Run install-fabric-worker.sh after setting PAIR_* variables. The optional
nvpair-cluster-manager binary supports first-time pairing on a headless node.
This archive contains no cluster identity, certificates, private keys, or models.
EOF
cat > "$STAGE/fabric-worker.env.example" <<'EOF'
PAIR_COORDINATOR_URL=https://coordinator.example:14324
PAIR_WORKER_ID=linux-worker-01
PAIR_NODE_ID=LINUX-01
PAIR_CLUSTER_DIR=/var/lib/nvpair/cluster
PAIR_RUNTIME=cpu
PAIR_BACKEND=cpu
PAIR_TRAINING_HOST_ROOT=
EOF
chmod +x "$STAGE/nvpair-compute-fabric" "$STAGE/install-fabric-worker.sh" "$STAGE/configure-nccl-ports.sh"
if [[ -f "$STAGE/nvpair-cluster-manager" ]]; then
  (cd "$STAGE" && sha256sum nvpair-compute-fabric nvpair-cluster-manager > SHA256SUMS)
else
  (cd "$STAGE" && sha256sum nvpair-compute-fabric > SHA256SUMS)
fi
(cd "$OUT" && zip -q -r "$(basename "$ZIPFILE")" "$(basename "$STAGE")")
echo "Created $ZIPFILE"
