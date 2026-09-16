<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>
[CmdletBinding()]
param([string]$OutputDir)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $root '.tmp\pair-linux-arm64' }
$versions = Get-Content (Join-Path $root 'services\versions.json') | ConvertFrom-Json
$components = @('ollama-proxy','lmstudio-proxy','nvpair-node-info','nvpair-node-scanner','nvpair-manual-nodes','nvpair-workload-manager','nvpair-errors','nvpair-engine-manager','nvpair-node-settings','nvpair-ui-broker','nvpair-cluster-manager','nvpair-job-scheduler','nvpair-tui','nvpair-compute-fabric')
Remove-Item -LiteralPath $OutputDir -Force -Recurse -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$oldOs = $env:GOOS; $oldArch = $env:GOARCH
try {
    $env:GOOS = 'linux'; $env:GOARCH = 'arm64'
    foreach ($component in $components) {
        $module = Join-Path $root "services\$component"
        $output = Join-Path $OutputDir $component
        Push-Location $module
        try { go build -ldflags "-X main.Version=$($versions.components.$component)" -o $output . }
        finally { Pop-Location }
        Write-Host "Built $component"
    }
}
finally { $env:GOOS = $oldOs; $env:GOARCH = $oldArch }
Write-Host "Created Linux ARM64 service set in $OutputDir"
