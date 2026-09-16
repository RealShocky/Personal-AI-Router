#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
image="${PAIR_TRAINING_IMAGE:-lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1}"
cuda_visible_devices="${PAIR_TRAINING_CUDA_VISIBLE_DEVICES-}"
nccl_socket_ifname="${PAIR_NCCL_SOCKET_IFNAME-enP7s7}"
docker_args=(--rm --gpus all --network host --ipc host)
if [[ -n "$cuda_visible_devices" && "$cuda_visible_devices" != "all" ]]; then
  docker_args+=( -e "CUDA_VISIBLE_DEVICES=$cuda_visible_devices" )
fi
exec docker run "${docker_args[@]}" \
  -e NCCL_SOCKET_IFNAME="$nccl_socket_ifname" \
  -e GLOO_SOCKET_IFNAME="$nccl_socket_ifname" \
  -e NCCL_IB_DISABLE=1 \
  -v /home/admin/pair-fabric:/opt/nvpair/training \
  "$image" torchrun "$@"
