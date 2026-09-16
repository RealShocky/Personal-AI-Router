<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param([string]$OutputDir)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $root 'dist' }
$binary = Join-Path $root 'services\build\bin\nvpair-compute-fabric.exe'
if (-not (Test-Path -LiteralPath $binary)) { throw "Build first: $binary" }
$stage = Join-Path $OutputDir 'fabric-windows-x64'
$zip = Join-Path $OutputDir 'fabric-windows-x64.zip'
Remove-Item -LiteralPath $stage -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $binary (Join-Path $stage 'nvpair-compute-fabric.exe')
Copy-Item (Join-Path $PSScriptRoot 'run-fabric-worker.ps1') $stage
Copy-Item (Join-Path $PSScriptRoot 'install-fabric-worker-windows.ps1') $stage
Copy-Item (Join-Path $root 'docs\distributed-fabric.mdx') (Join-Path $stage 'distributed-fabric.mdx')
Copy-Item (Join-Path $root 'docs\fabric-operations.mdx') (Join-Path $stage 'fabric-operations.mdx')
Copy-Item (Join-Path $root 'docs\README.md') (Join-Path $stage 'PAIR-documentation.md')
@('PAIR Fabric portable Windows worker','Run .\run-fabric-worker.ps1 after setting the required PAIR_* environment variables.','This bundle contains no cluster identity, certificates, private keys, or models.') | Set-Content (Join-Path $stage 'README.txt') -Encoding utf8
@('PAIR_COORDINATOR_URL=https://coordinator.example:14324','PAIR_WORKER_ID=win-worker-01','PAIR_NODE_ID=WIN-01','PAIR_CLUSTER_DIR=C:\Users\you\AppData\Local\PAIR\cluster','PAIR_RUNTIME=cpu','PAIR_BACKEND=cpu') | Set-Content (Join-Path $stage 'fabric-worker.env.example') -Encoding ascii
Get-FileHash (Join-Path $stage 'nvpair-compute-fabric.exe') -Algorithm SHA256 | ForEach-Object { "$($_.Hash)  nvpair-compute-fabric.exe" } | Set-Content (Join-Path $stage 'SHA256SUMS') -Encoding ascii
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Write-Host "Created $zip"
