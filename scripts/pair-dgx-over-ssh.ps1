<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DgxAddress,
    [string]$DgxUser = 'admin',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\id_ed25519'),
    [string]$CoordinatorConfig = (Join-Path (Split-Path -Parent $PSScriptRoot) '.tmp\dgx-pair-coordinator'),
    [string]$DgxConfig = '/home/admin/pair-dgx',
    [string]$DgxBundle = '/home/admin/pair-linux-arm64'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$manager = Join-Path $root 'services\build\bin\nvpair-cluster-manager.exe'
if (-not (Test-Path -LiteralPath $manager)) { throw "Build nvpair-cluster-manager first: $manager" }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "SSH key not found: $KeyPath" }
New-Item -ItemType Directory -Force -Path $CoordinatorConfig | Out-Null

function Start-RpcProcess([string]$FilePath, [string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $quoted = foreach ($argument in $Arguments) { '"' + $argument.Replace('"','\\"') + '"' }
    $info.Arguments = $quoted -join ' '
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $info
    if (-not $process.Start()) { throw "Could not start $FilePath" }
    return $process
}

function Send-Rpc($Process, [int]$Id, [string]$Method, [hashtable]$Params) {
    $request = @{ jsonrpc='2.0'; id=$Id; method=$Method; params=$Params } | ConvertTo-Json -Compress
    $Process.StandardInput.WriteLine($request)
    $Process.StandardInput.Flush()
}

function Wait-Rpc($Process, [int]$Id, [int]$TimeoutSeconds = 30) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $line = $Process.StandardOutput.ReadLine()
        if ($null -eq $line) {
            $errorText = $Process.StandardError.ReadToEnd()
            throw "RPC process exited before response id ${Id}: $errorText"
        }
        try { $message = $line | ConvertFrom-Json } catch { continue }
        if ($message.id -eq $Id) { return $message }
    }
    throw "Timed out waiting for RPC response id $Id"
}

$sshArgs = @('-o','BatchMode=yes','-o','StrictHostKeyChecking=no','-i',$KeyPath,"$DgxUser@$DgxAddress", "$DgxBundle/nvpair-cluster-manager --config-dir $DgxConfig --port 14321")
$remote = Start-RpcProcess 'ssh.exe' $sshArgs
$local = Start-RpcProcess $manager @('--config-dir',$CoordinatorConfig,'--port','14321')
try {
    Start-Sleep -Milliseconds 800
    Send-Rpc $local 1 'cluster:get-node-id' @{}
    Send-Rpc $remote 2 'cluster:get-node-id' @{}
    $localNode = Wait-Rpc $local 1
    $remoteNode = Wait-Rpc $remote 2
    $remoteNodeId = [string]$remoteNode.result.nodeId
    Send-Rpc $local 3 'cluster:invite-node' @{ address=$DgxAddress; port=14321; nodeId=$remoteNodeId }
    $invite = Wait-Rpc $local 3 30
    $pin = [string]$invite.result.pin
    $inviteId = [string]$invite.result.inviteId
    if ([string]::IsNullOrWhiteSpace($pin) -or [string]::IsNullOrWhiteSpace($inviteId)) { throw 'DGX invite did not return a PIN/invite ID' }
    Send-Rpc $remote 4 'cluster:respond-to-invite' @{ inviteId=$inviteId; accept=$true; pin=$pin }
    $joined = Wait-Rpc $remote 4 30
    if ([string]$joined.result.state -ne 'paired') { throw "DGX pairing did not complete: $($joined | ConvertTo-Json -Compress)" }
    Write-Host "DGX paired successfully. Coordinator node: $($localNode.result.nodeId); DGX node: $remoteNodeId"
}
finally {
    foreach ($process in @($local,$remote)) {
        if ($process -and -not $process.HasExited) { $process.StandardInput.Close(); $process.WaitForExit(5000) | Out-Null; if (-not $process.HasExited) { $process.Kill() } }
        if ($process) { $process.Dispose() }
    }
}
