<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param(
    [string]$CoordinatorUrl = $env:PAIR_COORDINATOR_URL,
    [string]$WorkerId = $env:PAIR_WORKER_ID,
    [string]$NodeId = $env:PAIR_NODE_ID,
    [string]$ClusterDir = $env:PAIR_CLUSTER_DIR,
    [string]$AdvertiseUrl = $env:PAIR_ADVERTISE_URL,
    [ValidateSet('cpu','cuda','metal')][string]$Runtime = $(if ($env:PAIR_RUNTIME) { $env:PAIR_RUNTIME } else { 'cpu' }),
    [string]$Backend = $(if ($env:PAIR_BACKEND) { $env:PAIR_BACKEND } else { 'cpu' }),
    [int]$HttpPort = $(if ($env:PAIR_HTTP_PORT) { [int]$env:PAIR_HTTP_PORT } else { 14324 }),
    [int]$HeartbeatTimeout = $(if ($env:PAIR_HEARTBEAT_TIMEOUT) { [int]$env:PAIR_HEARTBEAT_TIMEOUT } else { 10 }),
    [string]$Binary = $(Join-Path $PSScriptRoot 'nvpair-compute-fabric.exe')
)

$ErrorActionPreference = 'Stop'
foreach ($item in @{'CoordinatorUrl'=$CoordinatorUrl;'WorkerId'=$WorkerId;'NodeId'=$NodeId;'ClusterDir'=$ClusterDir}) {
    if ([string]::IsNullOrWhiteSpace($item.Value)) { throw "Missing required setting: $($item.Key)" }
}
if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) { throw "Fabric binary not found: $Binary" }
if (-not $AdvertiseUrl) { $AdvertiseUrl = "https://$([System.Net.Dns]::GetHostName()):$HttpPort" }

$arguments = @('--daemon','--cluster-dir',$ClusterDir,'--coordinator-url',$CoordinatorUrl,
    '--worker-id',$WorkerId,'--node-id',$NodeId,'--advertise-url',$AdvertiseUrl,
    '--runtime',$Runtime,'--backend',$Backend,'--http-port',$HttpPort,
    '--heartbeat-timeout',"${HeartbeatTimeout}s")
if ($env:PAIR_RPC_TARGET) { $arguments += @('--rpc-target',$env:PAIR_RPC_TARGET) }
if ($env:PAIR_RPC_SERVER_PATH) { $arguments += @('--rpc-server-path',$env:PAIR_RPC_SERVER_PATH) }
if ($env:PAIR_RPC_PORT) { $arguments += @('--rpc-port',$env:PAIR_RPC_PORT) }
if ($env:PAIR_MODEL_DIGEST) { $arguments += @('--model-digest',$env:PAIR_MODEL_DIGEST) }

& $Binary @arguments
exit $LASTEXITCODE
