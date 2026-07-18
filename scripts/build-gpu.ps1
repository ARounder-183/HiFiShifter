<#
.SYNOPSIS
    Build or run HiFiShifter with GPU acceleration on Windows.

.DESCRIPTION
    Builds with CPU + DirectML (DirectX 12) GPU support.
    DirectML works on any DirectX 12 GPU (NVIDIA, AMD, Intel Arc).

    With -Bundle : full release build - binary + NSIS installer.
    With -Dev    : development mode with hot reload.

    The SINGLE canonical location for GPU DLLs is:
        backend/src-tauri/third_party/ort-bundle/

    Populated by setup-windows.ps1.  No env var needed.

.PARAMETER Bundle
    Also create an NSIS installer.

.PARAMETER Log
    Enable file logging (log.txt next to the binary, with timestamps).

.PARAMETER Dev
    Run in development mode (cargo tauri dev) with hot reload.

.EXAMPLE
    .\scripts\build-gpu.ps1              # CPU + DirectML build, binary only
    .\scripts\build-gpu.ps1 -Bundle      # CPU + DirectML build + NSIS installer
    .\scripts\build-gpu.ps1 -Dev         # Dev server with hot reload
#>

[CmdletBinding()]
param(
    [switch]$Bundle,
    [switch]$Log,
    [switch]$Dev
)

$Features = "onnx,vslib,directml"
if ($Log) { $Features += ",logging" }
$GpuLabel = "DirectML"

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SrcTauri    = Join-Path $ProjectRoot "backend\src-tauri"
$BundleDir   = Join-Path $SrcTauri "third_party\ort-bundle"
$ModeLabel   = if ($Dev) { "dev" } elseif ($Bundle) { "build + NSIS" } else { "build (binary only)" }
if ($Log)   { $ModeLabel += " + log" }
$FullLabel   = "$GpuLabel $ModeLabel"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter GPU ($GpuLabel) - $ModeLabel" -ForegroundColor Cyan
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
    Write-Host "[build-gpu] Attempting auto-download..." -ForegroundColor Cyan
    $setupScript = Join-Path $PSScriptRoot "setup-windows.ps1"
    if (Test-Path $setupScript) {
        & $setupScript -SkipFrontend
        if (-not (Test-Path $primaryDll)) {
            Write-Error "Auto-download failed."
            exit 1
        }
    } else {
        Write-Error "setup-windows.ps1 not found - cannot auto-download."
        exit 1
    }
}

$env:ORT_LIB_LOCATION = $BundleDir

if ($Dev) {
    $env:ORT_PREFER_DYNAMIC_LINK = "1"
    $env:HIFISHIFTER_DEBUG_COMMANDS = "1"
}

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
Write-Host "[build-gpu] Starting $FullLabel..." -ForegroundColor Cyan
Write-Host ""

Push-Location $ProjectRoot
try {
    $releaseDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\release"

    if ($Dev) {
        cargo tauri dev --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri dev failed (exit code $LASTEXITCODE)"
        }
    }
    elseif ($Bundle) {
        cargo tauri build --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        $required = @("onnxruntime.dll", "onnxruntime_providers_shared.dll")
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical DLLs verified in release dir." -ForegroundColor Green
        }

        $injectScript = Join-Path $PSScriptRoot "inject-gpu-dlls.ps1"
        if (Test-Path $injectScript) {
            Write-Host "[build-gpu] Injecting DLLs into NSIS installer..." -ForegroundColor Cyan
            & $injectScript -TargetTriple "x86_64-pc-windows-msvc"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[build-gpu] WARN: DLL injection failed (exit code $LASTEXITCODE)" -ForegroundColor Yellow
            }
        }
        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] NSIS installer : $releaseDir\bundle\nsis" -ForegroundColor Cyan
    }
    else {
        cargo tauri build --no-bundle --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        $required = @("onnxruntime.dll", "onnxruntime_providers_shared.dll")
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical DLLs verified in release dir." -ForegroundColor Green
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] To create an NSIS installer, re-run with -Bundle" -ForegroundColor DarkGray
        Write-Host "[build-gpu] To create a portable ZIP, run: .\scripts\pack-portable.ps1 -SkipBuild" -ForegroundColor DarkGray
    }
} finally {
    Pop-Location
    Remove-Item Env:ORT_LIB_LOCATION -ErrorAction SilentlyContinue
}
