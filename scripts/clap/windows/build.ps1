# CLAP Windows 构建：cdylib -> HiFiShifter.clap -> 自检。
param([switch]$SkipVerify)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
. (Join-Path $root "scripts\env\msvc-env.ps1")
. (Join-Path $root "scripts\env\sdk-env.ps1"); Set-SdkEnvironment
Push-Location (Join-Path $root "backend")
try { cargo build -p hifishifter-clap --release --locked } finally { Pop-Location }
$module = Join-Path $root "backend\target\release\hifishifter_clap.dll"
$outDir = Join-Path $root "dist\clap"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item $module (Join-Path $outDir "HiFiShifter.clap") -Force
if (!$SkipVerify) { & (Join-Path $PSScriptRoot "verify.ps1") -ModulePath (Join-Path $outDir "HiFiShifter.clap") }
