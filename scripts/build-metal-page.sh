#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/pair-metal-page}"
[[ "$(uname -s)" == Darwin ]] || { echo "Run this on macOS" >&2; exit 1; }
command -v swiftc >/dev/null || { echo "swiftc is required" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
swiftc "$ROOT/services/nvpair-compute-fabric/metal_page.swift" -framework Metal -O -o "$OUT"
chmod 0755 "$OUT"
echo "Created $OUT"
