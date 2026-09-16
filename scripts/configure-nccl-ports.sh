#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

low="${1:-29401}"
high="${2:-29500}"
if ! [[ "$low" =~ ^[0-9]+$ && "$high" =~ ^[0-9]+$ && "$low" -ge 1024 && "$low" -lt "$high" && "$high" -le 65535 ]]; then
  echo "usage: $0 [low-port high-port]" >&2
  exit 2
fi

sudo sysctl -w "net.ipv4.ip_local_port_range=$low $high"
if [[ "${PAIR_PERSIST_NCCL_PORTS:-0}" == "1" ]]; then
  printf 'net.ipv4.ip_local_port_range = %s %s\n' "$low" "$high" | sudo tee /etc/sysctl.d/99-nvpair-nccl-ports.conf >/dev/null
  sudo sysctl --system >/dev/null
fi
echo "PAIR NCCL ephemeral ports configured as $low-$high."
