<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param(
    [string]$Nvcc = 'nvcc',
    [string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) 'desktop\cli-bin\pair-cuda-page.exe')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root 'services\nvpair-compute-fabric\cuda_page.cu'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "CUDA helper source not found: $source" }
& $Nvcc $source '-O2' '-std=c++17' '-o' $Output
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Host "Created $Output"
