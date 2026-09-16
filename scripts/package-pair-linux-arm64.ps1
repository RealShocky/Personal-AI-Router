<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>
[CmdletBinding()]
param([string]$BinaryDir, [string]$OutputDir)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BinaryDir)) { $BinaryDir = Join-Path $root '.tmp\pair-linux-arm64' }
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $root 'dist' }
$stage = Join-Path $OutputDir 'pair-linux-arm64'
$zip = Join-Path $OutputDir 'pair-linux-arm64.zip'
Remove-Item -LiteralPath $stage -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Get-ChildItem -LiteralPath $BinaryDir -File | Copy-Item -Destination $stage -Force
Copy-Item -LiteralPath (Join-Path $root 'scripts\install-fabric-worker.sh') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\fabric-operations.mdx') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\distributed-fabric.mdx') -Destination $stage
Copy-Item -LiteralPath (Join-Path $root 'docs\README.md') -Destination (Join-Path $stage 'PAIR-documentation.md')
@('PAIR headless Linux ARM64 bundle','Run ./nvpair-tui to start the broker and pairing UI over SSH.','Run install-fabric-worker.sh only after pairing; it installs the fabric worker as systemd.','This bundle contains no identities, certificates, private keys, SSH keys, or models.') | Set-Content (Join-Path $stage 'README.txt') -Encoding utf8
$hashes = Get-ChildItem -LiteralPath $stage -File | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($_.Name)"
}
$hashes | Set-Content -LiteralPath (Join-Path $stage 'SHA256SUMS') -Encoding ascii
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Write-Host "Created $zip"
