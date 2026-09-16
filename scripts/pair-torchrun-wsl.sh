#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
nccl_socket_ifname="${PAIR_NCCL_SOCKET_IFNAME-eth1}"
export NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
export GLOO_SOCKET_IFNAME="$nccl_socket_ifname"
export NCCL_IB_DISABLE=1
if [[ -n "${PAIR_NCCL_DEBUG-}" ]]; then export NCCL_DEBUG="$PAIR_NCCL_DEBUG"; fi
if [[ -n "${PAIR_NCCL_DEBUG_SUBSYS-}" ]]; then export NCCL_DEBUG_SUBSYS="$PAIR_NCCL_DEBUG_SUBSYS"; fi
if [[ -n "${PAIR_NCCL_P2P_DISABLE-}" ]]; then export NCCL_P2P_DISABLE="$PAIR_NCCL_P2P_DISABLE"; fi
exec /opt/nvpair/training-venv/bin/torchrun "$@"
