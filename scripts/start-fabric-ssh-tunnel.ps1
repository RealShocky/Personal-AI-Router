# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SshKey,
    [Parameter(Mandatory = $true)][string]$RemoteUser,
    [Parameter(Mandatory = $true)][string]$RemoteHost,
    [Parameter(Mandatory = $true)][string]$CoordinatorAddress,
    [int]$CoordinatorPort = 14324,
    [string]$LocalBindAddress = '127.0.0.1',
    [int]$LocalWorkerPort = 15428,
    [int]$RemoteWorkerPort = 14324,
    [int]$RemoteCoordinatorPort = 15430,
    [int]$SshPort = 22
)

$ErrorActionPreference = 'Stop'
$keyPath = (Resolve-Path -LiteralPath $SshKey -ErrorAction Stop).Path
$ssh = Get-Command ssh.exe -ErrorAction Stop

$arguments = @(
    '-N',
    '-T',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3',
    '-i', $keyPath,
    '-p', "$SshPort",
    '-L', "${LocalBindAddress}:${LocalWorkerPort}:127.0.0.1:${RemoteWorkerPort}",
    '-R', "${RemoteCoordinatorPort}:${CoordinatorAddress}:${CoordinatorPort}",
    "${RemoteUser}@${RemoteHost}"
)

Write-Host "PAIR tunnel: local ${LocalBindAddress}:${LocalWorkerPort} -> ${RemoteHost}:127.0.0.1:${RemoteWorkerPort}"
Write-Host "PAIR tunnel: remote 127.0.0.1:${RemoteCoordinatorPort} -> ${CoordinatorAddress}:${CoordinatorPort}"
Write-Host 'Keep this process running while the routed worker is admitted.'
& $ssh.Source @arguments
exit $LASTEXITCODE
