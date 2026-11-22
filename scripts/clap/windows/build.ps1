# CLAP Windows build: cdylib -> HiFiShifter.clap -> self-check.
param([switch]$SkipVerify)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
. (Join-Path $root "scripts\env\msvc-env.ps1")
. (Join-Path $root "scripts\env\sdk-env.ps1"); Set-SdkEnvironment
# The env scripts are dot-sourced (incident record: scripts/build/build.ps1), so recompute
# the repo root; never let the same-named variables they leave behind swap it out.
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
# The target directory matches the sibling convention (scripts/build/build.ps1:147,
# clap/unix/build.sh:73): respect CARGO_TARGET_DIR. Relative values are resolved the way
# cargo does it — this script runs cargo inside backend/, so a relative CARGO_TARGET_DIR
# resolves against backend; write the absolute value back so the artifact path taken below
# always matches cargo's actual output location. The old hard-coded backend\target either
# failed the copy on machines with CARGO_TARGET_DIR set, or something worse: silently
# packaging the **previous build's stale DLL** from default-target as the "new" plugin.
$backendDir = Join-Path $root "backend"
if ($env:CARGO_TARGET_DIR) {
    if ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
        $cargoTarget = [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
    }
    else {
        $cargoTarget = [IO.Path]::GetFullPath((Join-Path $backendDir $env:CARGO_TARGET_DIR))
    }
    $env:CARGO_TARGET_DIR = $cargoTarget
}
else {
    $cargoTarget = Join-Path $backendDir "target"
}
Push-Location $backendDir
try { cargo build -p hifishifter-clap --release --locked } finally { Pop-Location }
# PowerShell does not throw on a native command's non-zero exit ($ErrorActionPreference
# only covers cmdlets); without checking $LASTEXITCODE, a failed build lets the Copy-Item
# below silently package the **previous build's stale hifishifter_clap.dll** as the "new"
# plugin (verify.ps1 accepts any valid old build) — same rationale as build.ps1's
# Assert-Ok discipline.
if ($LASTEXITCODE -ne 0) { throw "cargo build -p hifishifter-clap failed (exit $LASTEXITCODE)" }
$module = Join-Path $cargoTarget "release\hifishifter_clap.dll"
$outDir = Join-Path $root "dist\clap"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item $module (Join-Path $outDir "HiFiShifter.clap") -Force

# A single-file CLAP's frontend and model manifest must sit beside the module
# (ResourceLayout::Sibling). With only the .clap, hosts can scan the plugin but
# frontend_dir() cannot find plugin.html — the symptom is the blank window users see.
# CI's frontend-dist artifact is installed here as well.
$resources = Join-Path $outDir "HiFiShifter-resources"
$frontend = Join-Path $resources "frontend"
$models = Join-Path $resources "models"
# The same shared assets as the VST3 bundle: the host plugin and the floating editor share
# chunks, and the SDK type declarations are not runtime payload. Copying only plugin.html
# would 404 detached.html inside the plugin window.
New-Item -ItemType Directory -Force -Path $frontend, $models | Out-Null
Get-ChildItem -LiteralPath (Join-Path $root "frontend\dist") |
    Where-Object { $_.Name -ne 'sdk' } |
    Copy-Item -Destination $frontend -Recurse -Force
Copy-Item (Join-Path $root "backend\src-tauri\resources\models\models.json") $models -Force
# The WebView2 profile is runtime state and must never travel with the resources
# (same criterion as scripts/build/build.ps1).
Remove-Item -LiteralPath (Join-Path $resources "webview") -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "CLAP resources: $resources" -ForegroundColor Green

if (!$SkipVerify) { & (Join-Path $PSScriptRoot "verify.ps1") -ModulePath (Join-Path $outDir "HiFiShifter.clap") }
