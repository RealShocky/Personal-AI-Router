#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
nccl_socket_ifname="${PAIR_NCCL_SOCKET_IFNAME-eth1}"
export NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
export GLOO_SOCKET_IFNAME="$nccl_socket_ifname"
export NCCL_IB_DISABLE=1
exec /opt/nvpair/training-venv/bin/torchrun "$@"
