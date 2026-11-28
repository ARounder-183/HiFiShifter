#!/usr/bin/env pwsh
# Import the MSVC environment into the CURRENT process and republish it to $GITHUB_ENV,
# so a later *separate* process (GitHub Actions steps, tauri-action, ...) inherits the
# same toolchain.
#
# [Why this file exists] scripts/env/msvc-env.ps1 only affects the process that dot-sources
# it. That is enough for build.ps1 (it runs cargo in-process), but NOT for the Windows arm64
# CI leg, which uses tauri-action: that action is a separate Node process, so the pinned
# CMake generator (NMake Makefiles) found no nmake.exe and hifishifter-kernel's build script
# died inside cmake configure (run 45953527512).
#
# Usage:
#   ./scripts/setup/export-msvc-env.ps1                 # x64 (default)
#   ./scripts/setup/export-msvc-env.ps1 -Arch arm64     # cross/native arm64
#
# Locally (no $GITHUB_ENV) it still imports the environment and prints a summary; the
# export step is simply skipped.
param(
    [ValidateSet('x64', 'arm64')][string]$Arch = 'x64'
)

$ErrorActionPreference = 'Stop'

# [Why a snapshot instead of an explicit variable list] vcvars sets a dozen variables whose
# names differ between VS versions (VSCMD_*, UCRTVersion, WindowsSDKVersion, ...). Listing
# them by hand means the next VS upgrade silently drops one; diffing the environment cannot.
function Get-EnvironmentSnapshot {
    $snapshot = @{}
    foreach ($item in Get-ChildItem Env:) { $snapshot[$item.Name] = [string]$item.Value }
    return $snapshot
}

function Test-ToolOnPath {
    param([string]$Name)
    return [bool](Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
}

$hadNode = Test-ToolOnPath 'node'
$hadPython = Test-ToolOnPath 'python'
$before = Get-EnvironmentSnapshot

# [Why the env var and not a parameter] msvc-env.ps1 is dot-sourced by many callers and may
# only leave $env:* behind (see scripts/checks/check-env-scripts.ps1); the target
# architecture therefore travels in HFS_MSVC_TARGET_ARCH.
$env:HFS_MSVC_TARGET_ARCH = $Arch
. (Join-Path $PSScriptRoot '..\env\msvc-env.ps1')

$cl = Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue
$nmake = Get-Command nmake.exe -CommandType Application -ErrorAction SilentlyContinue
if (!$cl -or !$nmake) {
    throw ('MSVC environment is incomplete after import (target {0}): cl.exe={1}, nmake.exe={2}. ' -f
        $Arch,
        $(if ($cl) { $cl.Source } else { 'missing' }),
        $(if ($nmake) { $nmake.Source } else { 'missing' }))
}

# [Why a denylist] vcvars also sets a few generic / bookkeeping variables. Platform is an
# MSBuild property name: publishing it would apply to every later step and could silently
# retarget an unrelated MSBuild invocation. The __DOTNET_* / __VSCMD_PREINIT_PATH entries are
# vcvars' own bookkeeping and mean nothing outside that process.
$skipNames = @(
    'Platform',
    'CommandPromptType',
    'DevEnvDir',
    '__DOTNET_ADD_64BIT',
    '__DOTNET_PREFERRED_BITNESS',
    '__VSCMD_PREINIT_PATH'
)

$after = Get-EnvironmentSnapshot
$changed = New-Object System.Collections.Generic.List[string]
foreach ($name in $after.Keys) {
    if ($before.ContainsKey($name) -and $before[$name] -ceq $after[$name]) { continue }
    if ($skipNames -contains $name) { continue }
    $value = $after[$name]
    if ($value -match "[`r`n]") {
        Write-Warning ("skipping multi-line environment variable {0}" -f $name)
        continue
    }
    $changed.Add($name)
}

if ($env:GITHUB_ENV) {
    foreach ($name in $changed) {
        # Windows environment variable names are case-insensitive, but the runner applies
        # GITHUB_ENV by name: publishing 'Path' must be spelled PATH so the PATH update of
        # the following steps is the one vcvars produced.
        $exportName = if ($name -ieq 'path') { 'PATH' } else { $name }
        Add-Content -Path $env:GITHUB_ENV -Value ('{0}={1}' -f $exportName, $after[$name])
    }
}

Write-Host ('[msvc-env] target {0} | CMAKE_GENERATOR={1} | HFS_NATIVE_GENERATOR={2}' -f
    $Arch, $env:CMAKE_GENERATOR, $env:HFS_NATIVE_GENERATOR)
Write-Host ('[msvc-env] cl.exe   = {0}' -f $cl.Source)
Write-Host ('[msvc-env] nmake.exe= {0}' -f $nmake.Source)
if ($env:GITHUB_ENV) {
    Write-Host ('[msvc-env] exported {0} variables to GITHUB_ENV' -f $changed.Count)
}
else {
    Write-Host ('[msvc-env] no GITHUB_ENV in this shell; {0} variables changed in-process only' -f $changed.Count)
}

# [Why this is checked at all] Publishing PATH replaces the PATH of every later step. If vcvars
# had dropped a tool that the pipeline needs, the failure would surface much later and in an
# unrelated step; naming it here keeps the diagnosis local.
if ($hadNode -and !(Test-ToolOnPath 'node')) { Write-Warning 'node is no longer on PATH after importing the MSVC environment' }
if ($hadPython -and !(Test-ToolOnPath 'python')) { Write-Warning 'python is no longer on PATH after importing the MSVC environment' }
