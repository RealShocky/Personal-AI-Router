<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Binary,
    [ValidateSet('arm64', 'x64')][string]$Architecture = 'arm64',
    [string]$ClusterManagerBinary,
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'dist')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $OutputDir "fabric-macos-$Architecture"
$zip = Join-Path $OutputDir "fabric-macos-$Architecture.zip"

if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
    throw "macOS fabric binary was not found: $Binary"
}

Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath $Binary -Destination (Join-Path $stage 'nvpair-compute-fabric')
if (-not [string]::IsNullOrWhiteSpace($ClusterManagerBinary)) {
    if (-not (Test-Path -LiteralPath $ClusterManagerBinary -PathType Leaf)) {
        throw "macOS cluster manager binary was not found: $ClusterManagerBinary"
    }
    Copy-Item -LiteralPath $ClusterManagerBinary -Destination (Join-Path $stage 'nvpair-cluster-manager')
}
Copy-Item -LiteralPath (Join-Path $root 'scripts\install-fabric-worker-macos.sh') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\distributed-fabric.mdx') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\fabric-operations.mdx') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\README.md') -Destination (Join-Path $stage 'PAIR-documentation.md')
@(
    'PAIR Fabric portable macOS worker',
    "Target architecture: $Architecture",
    'Run install-fabric-worker-macos.sh after pairing.',
    'An optional nvpair-cluster-manager binary supports first-time pairing on a headless node.',
    'This archive contains no identities, keys, or models.'
) | Set-Content -LiteralPath (Join-Path $stage 'README.txt') -Encoding utf8

Get-ChildItem -LiteralPath $stage -File | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($_.Name)"
} | Set-Content -LiteralPath (Join-Path $stage 'SHA256SUMS') -Encoding ascii

Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Write-Host "Created $zip"
