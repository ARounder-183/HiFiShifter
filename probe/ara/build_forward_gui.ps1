# 正向ARA开发包构建入口；仅此worktree，原GUI与插件分开构建，避免vslib特性污染插件。
param([switch]$SkipFrontend)
$ErrorActionPreference = 'Stop'
$araForwardRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$araForwardExe = Join-Path $araForwardRoot 'backend\target\debug\HiFiShifter.exe'
$araForwardRunning = Get-CimInstance Win32_Process -Filter "Name = 'HiFiShifter.exe'" |
    Where-Object { $_.ExecutablePath -eq $araForwardExe }
if ($araForwardRunning) { throw 'This worktree GUI is running. Save your current edits and close it normally before rebuilding; this script never terminates it.' }
Push-Location $araForwardRoot
try {
    . .\tools\msvc-env.ps1
    $araForwardTemp = Join-Path $araForwardRoot ('.build-tmp\forward-build-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $araForwardTemp | Out-Null
    # vcvars会重设TEMP/TMP，所以必须在它之后设置；不修改系统环境。
    $env:TEMP = $araForwardTemp
    $env:TMP = $araForwardTemp
    $env:ARA_VST3_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\vst3sdk'
    $env:ARA_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\ARA_SDK'
    if (!$SkipFrontend) {
        Push-Location frontend
        try {
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "frontend build failed: $LASTEXITCODE" }
        } finally { Pop-Location }
    }
    & cargo build --manifest-path backend\Cargo.toml --offline --jobs 1 -p HiFiShifter --features custom-protocol
    if ($LASTEXITCODE -ne 0) { throw "embedded GUI build failed: $LASTEXITCODE" }
    # 不把两个包放同一次cargo调用：app默认vslib会统一到kernel并污染插件导入表。
    & cargo build --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-plugin
    if ($LASTEXITCODE -ne 0) { throw "standalone plugin build failed: $LASTEXITCODE" }
    & dumpbin /dependents backend\target\debug\hifishifter_plugin.dll
    if ($LASTEXITCODE -ne 0) { throw "plugin dependency inspection failed: $LASTEXITCODE" }
    Write-Output "Built GUI and plugin under $araForwardRoot\backend\target\debug; not deployed, no running project changed."
} finally { Pop-Location }
