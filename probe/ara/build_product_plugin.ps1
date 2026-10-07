# 产品插件的隔离构建入口；每次加载 MSVC 后重设临时目录，不修改 SDK 检出。
param([switch]$Test, [string]$Filter = '')
$ErrorActionPreference = 'Stop'
$araProductRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Push-Location $araProductRoot
try {
    . .\tools\msvc-env.ps1
    $araBuildTemp = Join-Path $araProductRoot '.build-tmp\cl'
    New-Item -ItemType Directory -Force $araBuildTemp | Out-Null
    $env:TEMP = $araBuildTemp
    $env:TMP = $araBuildTemp
    $env:ARA_VST3_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\vst3sdk'
    $env:ARA_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\ARA_SDK'
    if ($Test) {
        $araTestArgs = @('test', '--manifest-path', 'backend\Cargo.toml', '-p', 'hifishifter-plugin', '--offline', '--jobs', '1')
        if ($Filter) { $araTestArgs += $Filter }
        & cargo @araTestArgs
    } else {
        & cargo build --manifest-path backend\Cargo.toml -p hifishifter-plugin --offline --jobs 1
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
