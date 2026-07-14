<#
.SYNOPSIS
    Build or run HiFiShifter with CUDA/GPU acceleration on Windows.

.DESCRIPTION
    Default (no flags): fast release build - compiles the binary and stages
                        GPU DLLs, but does NOT create an NSIS installer.

    With -Bundle     : full release build - binary + NSIS installer.
                        Takes significantly longer because of the NSIS
                        packaging step (~2 GB of GPU DLLs to compress).

    With -Dev        : development mode with hot reload.

    The SINGLE canonical location for GPU DLLs is:
        backend/src-tauri/third_party/ort-bundle/

    Populated by setup-windows.ps1.  No env var needed.

.PARAMETER Bundle
    Also create an NSIS installer (slow - ~2 GB of DLLs to package).

.PARAMETER Log
    Enable file logging (log.txt next to the binary, with timestamps).

.PARAMETER Dev
    Run in development mode (cargo tauri dev) with hot reload.

.EXAMPLE
    .\scripts\build-gpu.ps1              # Fast: binary only
    .\scripts\build-gpu.ps1 -Bundle      # Full: binary + NSIS installer
    .\scripts\build-gpu.ps1 -Log         # Binary + file logging enabled
    .\scripts\build-gpu.ps1 -Dev         # Dev server with hot reload
#>

[CmdletBinding()]
param(
    [switch]$Bundle,
    [switch]$Log,
    [switch]$Dev
)

# Build the feature list - `logging` is opt-in.
$Features = "onnx,cuda,tensorrt"
if ($Log) { $Features += ",logging" }

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SrcTauri    = Join-Path $ProjectRoot "backend\src-tauri"
$BundleDir   = Join-Path $SrcTauri "third_party\ort-bundle"
$ModeLabel   = if ($Dev) { "dev" } elseif ($Bundle) { "build + NSIS" } else { "build (binary only)" }
if ($Log)   { $ModeLabel += " + log" }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter GPU - $ModeLabel" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# 1. Rust Environment
# ============================================================
$RustDir = Join-Path $ProjectRoot ".rust"
$CargoHome = Join-Path $RustDir "cargo"
$RustupHome = Join-Path $RustDir "rustup"
$CargoBin = Join-Path $CargoHome "bin\cargo.exe"

if (Test-Path $CargoBin) {
    Write-Host "[build-gpu] Using project-local Rust at $RustDir" -ForegroundColor Green
    $env:CARGO_HOME  = $CargoHome
    $env:RUSTUP_HOME = $RustupHome
    $binPath = Join-Path $CargoHome "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }
    cargo --version
} else {
    Write-Host "[build-gpu] No local Rust found; using system Rust (if available)"
}

# ============================================================
# 2. Verify GPU DLLs exist
# ============================================================
$primaryDll = Join-Path $BundleDir "onnxruntime.dll"
if (-not (Test-Path $primaryDll)) {
    Write-Host ""
    Write-Host "[build-gpu] GPU DLLs not found in $BundleDir" -ForegroundColor Red
    Write-Host "[build-gpu] Run: .\scripts\setup-windows.ps1" -ForegroundColor Yellow
    Write-Host ""
    if (-not $Dev) {
        Write-Error "Cannot build without GPU DLLs. Exiting."
        exit 1
    }
    # Dev mode: try auto-download
    Write-Host "[build-gpu] Attempting auto-download..." -ForegroundColor Cyan
    $setupScript = Join-Path $PSScriptRoot "setup-windows.ps1"
    if (Test-Path $setupScript) {
        & $setupScript -SkipRust -SkipFrontend
        if (-not (Test-Path $primaryDll)) {
            Write-Error "Auto-download failed."
            exit 1
        }
    } else {
        Write-Error "setup-windows.ps1 not found - cannot auto-download."
        exit 1
    }
}

$env:ORT_USE_CUDA = "1"
if ($Dev) {
    $env:ORT_PREFER_DYNAMIC_LINK = "1"
    $env:HIFISHIFTER_DEBUG_COMMANDS = "1"
}

# Count what we have
$dllCount = @(Get-ChildItem "$BundleDir\*.dll" -ErrorAction SilentlyContinue).Count
Write-Host "[build-gpu] GPU DLLs in ort-bundle/: $dllCount" -ForegroundColor Cyan

# ============================================================
# 3. Dev mode: stage DLLs to debug dir for runtime
# ============================================================
if ($Dev) {
    $debugDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\debug"
    New-Item -ItemType Directory -Path $debugDir -Force | Out-Null
    $staged = 0
    Get-ChildItem "$BundleDir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        $dst = Join-Path $debugDir $_.Name
        if (-not (Test-Path $dst) -or ($_.LastWriteTime -gt (Get-Item $dst).LastWriteTime)) {
            Copy-Item $_.FullName $dst -Force
            $staged++
        }
    }
    Write-Host "[build-gpu] Staged $staged DLL(s) to debug dir"
}

# ============================================================
# 4. Build / Bundle / Dev
# ============================================================
Write-Host "[build-gpu] Starting GPU $ModeLabel..." -ForegroundColor Cyan
Write-Host ""

Push-Location $ProjectRoot
try {
    $releaseDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\release"

    if ($Dev) {
        # --- Dev mode: hot-reload dev server ---------------------------------
        cargo tauri dev --features $Features
    }
    elseif ($Bundle) {
        # --- Full build: binary + NSIS installer (SLOW - large DLLs) ---------
        Write-Host "[build-gpu] Generating GPU resource config for NSIS..." -ForegroundColor Cyan
        $stageScript = Join-Path $PSScriptRoot "stage-tauri-resources.ps1"
        if (-not (Test-Path $stageScript)) {
            Write-Error "[build-gpu] stage-tauri-resources.ps1 not found."
            exit 1
        }
        & $stageScript -ProjectRoot $ProjectRoot
        if (-not $?) {
            Write-Error "[build-gpu] Resource config generation failed."
            exit 1
        }

        cargo tauri build --features $Features

        # Verify
        $required = @("cudart64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical CUDA DLLs verified in release dir." -ForegroundColor Green
        }

        # Clean up generated config
        $configPath = Join-Path $SrcTauri "tauri.windows.conf.json"
        if (Test-Path $configPath) {
            Remove-Item $configPath -Force
            Write-Host "[build-gpu] Cleaned up $configPath" -ForegroundColor DarkGray
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] NSIS installer : $releaseDir\bundle\nsis" -ForegroundColor Cyan
    }
    else {
        # --- Fast build: binary only, no NSIS ---------------------------------
        Push-Location $SrcTauri
        cargo build --release --features $Features
        Pop-Location

        # build.rs (with `cuda` feature) copies DLLs from ort-bundle/
        # to target/release/ automatically.  Verify they landed.
        $required = @("cudart64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical CUDA DLLs verified in release dir." -ForegroundColor Green
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] To create an NSIS installer, re-run with -Bundle" -ForegroundColor DarkGray
        Write-Host "[build-gpu] To create a portable ZIP, run: .\scripts\pack-portable.ps1 -SkipBuild" -ForegroundColor DarkGray
    }
} finally {
    Pop-Location
}
