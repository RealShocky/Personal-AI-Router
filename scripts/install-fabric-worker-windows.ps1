<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param([string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'PAIR\fabric-worker'), [switch]$Task)
$ErrorActionPreference = 'Stop'
$source = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'nvpair-compute-fabric.exe') -Destination $InstallDir -Force
Copy-Item -LiteralPath (Join-Path $source 'run-fabric-worker.ps1') -Destination $InstallDir -Force
Write-Host "Installed portable worker files in $InstallDir"
Write-Host "Set PAIR_COORDINATOR_URL, PAIR_WORKER_ID, PAIR_NODE_ID, and PAIR_CLUSTER_DIR before running run-fabric-worker.ps1."
if ($Task) {
    $action = New-ScheduledTaskAction -Execute 'pwsh.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\run-fabric-worker.ps1`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    Register-ScheduledTask -TaskName 'PAIR Fabric Worker' -Action $action -Trigger $trigger -Force | Out-Null
    Write-Host 'Registered the worker as a per-user logon task.'
}
