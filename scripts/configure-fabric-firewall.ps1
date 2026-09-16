<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param(
    [string]$RemoteSubnet = '192.168.50.0/24',
    [string]$PortRange = '14324,15425,29401-29500'
)

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated PowerShell window.'
}

$ruleName = 'PAIR distributed fabric LAN'
$existing = Get-NetFirewallRule -Name 'PAIR-Fabric-LAN' -ErrorAction SilentlyContinue
if ($null -eq $existing) {
    New-NetFirewallRule -Name 'PAIR-Fabric-LAN' -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $PortRange -RemoteAddress $RemoteSubnet -Profile Private | Out-Null
} else {
    Set-NetFirewallRule -Name 'PAIR-Fabric-LAN' -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $PortRange -RemoteAddress $RemoteSubnet -Profile Private | Out-Null
}

Write-Host "Configured $ruleName for TCP $PortRange from $RemoteSubnet."
