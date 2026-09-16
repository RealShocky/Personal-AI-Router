<#
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
#>

[CmdletBinding()]
param(
    [ValidateSet('Install','Remove','Start','Stop','Status')]
    [string]$Action = 'Install',
    [string]$Distro = $(if ($env:PAIR_WSL_DISTRO) { $env:PAIR_WSL_DISTRO } else { 'Ubuntu' }),
    [string]$TaskName = 'PAIR WSL Fabric Keepalive'
)

$ErrorActionPreference = 'Stop'
$wsl = Join-Path $env:WINDIR 'System32\wsl.exe'
if (-not (Test-Path -LiteralPath $wsl -PathType Leaf)) { throw "WSL executable not found: $wsl" }

$taskArgs = "-d `"$Distro`" -- sleep infinity"
$taskAction = New-ScheduledTaskAction -Execute $wsl -Argument $taskArgs
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn
$taskSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$startupShortcut = Join-Path ([Environment]::GetFolderPath('Startup')) (($TaskName -replace '[^A-Za-z0-9 ._-]', '_') + '.lnk')

switch ($Action) {
    'Install' {
        try {
            Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Description "Keeps the selected WSL distro alive so its PAIR worker can heartbeat and auto-rejoin." -Force | Out-Null
            Start-ScheduledTask -TaskName $TaskName
            Write-Host "Installed and started '$TaskName' as a scheduled task for WSL distro '$Distro'."
        } catch [System.Exception] {
            $shell = New-Object -ComObject WScript.Shell
            $shortcut = $shell.CreateShortcut($startupShortcut)
            $shortcut.TargetPath = $wsl
            $shortcut.Arguments = $taskArgs
            $shortcut.WorkingDirectory = $env:USERPROFILE
            $shortcut.Save()
            Start-Process -FilePath $wsl -ArgumentList @('-d',$Distro,'--','sleep','infinity') -WindowStyle Hidden
            Write-Warning "Scheduled Task registration was denied; installed the no-admin Startup-folder fallback at $startupShortcut."
        }
    }
    'Remove' {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $startupShortcut -Force -ErrorAction SilentlyContinue
        Write-Host "Removed '$TaskName' and its optional Startup shortcut. This does not stop or modify PAIR workers or other WSL projects."
    }
    'Start' { Start-ScheduledTask -TaskName $TaskName; Write-Host "Started '$TaskName'." }
    'Stop' { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue; Write-Host "Stopped '$TaskName'." }
    'Status' {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($null -ne $task) { $task | Get-ScheduledTaskInfo | Select-Object TaskName,LastRunTime,LastTaskResult,TaskState }
        elseif (Test-Path -LiteralPath $startupShortcut) { [pscustomobject]@{ TaskName = $TaskName; LastRunTime = $null; LastTaskResult = $null; TaskState = 'StartupFolderFallback'; Path = $startupShortcut } }
        else { throw "No keepalive task or Startup shortcut found for '$TaskName'." }
    }
}
