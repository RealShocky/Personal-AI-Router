#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

venv_dir="${PAIR_TRAINING_VENV:-/opt/nvpair/training-venv}"
python_bin="${venv_dir}/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  python3 -m venv "${venv_dir}"
fi

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install --upgrade --force-reinstall --no-cache-dir \
  torch==2.13.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130 \
  --timeout 60 --retries 2
"${python_bin}" -m pip install --upgrade \
  'cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3' \
  cuda-bindings==13.4.1 \
  nvidia-cudnn-cu13==9.20.0.48 \
  nvidia-cusparselt-cu13==0.8.1 \
  nvidia-nccl-cu13==2.29.7 \
  nvidia-nvshmem-cu13==3.4.5 \
  triton==3.7.1 \
  numpy \
  --timeout 60 --retries 2

"${python_bin}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit('PAIR training runtime has no CUDA device')
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY
