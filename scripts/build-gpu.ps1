<#
.SYNOPSIS
    Build or run HiFiShifter with GPU acceleration on Windows.

.DESCRIPTION
    Default (no flags): DirectML GPU build - uses any DirectX 12-capable
                        GPU (NVIDIA, AMD, Intel Arc). No CUDA SDK needed.

    With -CUDA       : CUDA + TensorRT GPU build. Requires NVIDIA GPU and
                       CUDA runtime DLLs (via setup-windows.ps1).

    With -Bundle     : full release build - binary + NSIS installer.

    With -Dev        : development mode with hot reload.

    The SINGLE canonical location for GPU DLLs is:
        backend/src-tauri/third_party/ort-bundle/

    Populated by setup-windows.ps1.  No env var needed.

.PARAMETER Bundle
    Also create an NSIS installer.

.PARAMETER Log
    Enable file logging (log.txt next to the binary, with timestamps).

.PARAMETER Dev
    Run in development mode (cargo tauri dev) with hot reload.

.PARAMETER CUDA
    Build with CUDA + TensorRT instead of the default DirectML.
    Requires NVIDIA GPU and CUDA runtime DLLs.

.PARAMETER DirectML
    (Default - flag kept for backward compatibility.)
    Build with DirectML (DirectX 12) GPU support.

.EXAMPLE
    .\scripts\build-gpu.ps1                    # DirectML GPU build, binary only
    .\scripts\build-gpu.ps1 -CUDA              # CUDA GPU build
    .\scripts\build-gpu.ps1 -CUDA -Bundle      # CUDA GPU build + NSIS installer
    .\scripts\build-gpu.ps1 -Bundle            # DirectML GPU build + NSIS
    .\scripts\build-gpu.ps1 -Dev               # DirectML dev server
    .\scripts\build-gpu.ps1 -CUDA -Dev         # CUDA dev server
#>

[CmdletBinding()]
param(
    [switch]$Bundle,
    [switch]$Log,
    [switch]$Dev,
    [switch]$CUDA,
    [switch]$DirectML
)

# DirectML is the default.  -CUDA switches to CUDA + TensorRT.
if ($CUDA) {
    $Features = "onnx,cuda,tensorrt"
    $GpuLabel = "CUDA"
} else {
    $Features = "onnx,vslib"
    $GpuLabel = "DirectML"
}
if ($Log) { $Features += ",logging" }

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
    if ($CUDA) {
        Write-Host "[build-gpu]   for CUDA: download-ort.ps1 downloads GPU package by default" -ForegroundColor Yellow
        Write-Host "[build-gpu]   then run: .\scripts\download-cuda-runtime.ps1" -ForegroundColor Yellow
    }
    Write-Host ""
    if (-not $Dev) {
        Write-Error "Cannot build without GPU DLLs. Exiting."
        exit 1
    }
    # Dev mode: try auto-download
    Write-Host "[build-gpu] Attempting auto-download..." -ForegroundColor Cyan
    $setupScript = Join-Path $PSScriptRoot "setup-windows.ps1"
    if (Test-Path $setupScript) {
        if ($CUDA) {
            & $setupScript -SkipFrontend
        } else {
            & $setupScript -DirectML -SkipFrontend
        }
        if (-not (Test-Path $primaryDll)) {
            Write-Error "Auto-download failed."
            exit 1
        }
    } else {
        Write-Error "setup-windows.ps1 not found - cannot auto-download."
        exit 1
    }
}

# CUDA builds need the CUDA runtime env flag for ort-sys download-binaries fallback.
# DirectML builds do not - the ORT GPU package already includes DirectML.
if ($CUDA) {
    $env:ORT_USE_CUDA = "1"
}

# Tell ort-sys to link against the SAME ONNX Runtime that will be staged
# to the binary directory by build.rs.
$env:ORT_LIB_LOCATION = $BundleDir

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

        # Verify critical DLLs
        if ($CUDA) {
            $required = @(
                "cudart64_12.dll",
                "cublas64_12.dll",
                "cufft64_11.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll"
            )
        } else {
            $required = @(
                "onnxruntime.dll",
                "onnxruntime_providers_shared.dll"
            )
        }
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical $GpuLabel DLLs verified in release dir." -ForegroundColor Green
        }

        # Inject GPU DLLs into NSIS installer
        $injectScript = Join-Path $PSScriptRoot "inject-gpu-dlls.ps1"
        if (Test-Path $injectScript) {
            Write-Host "[build-gpu] Injecting $GpuLabel DLLs into NSIS installer..." -ForegroundColor Cyan
            & $injectScript -TargetTriple "x86_64-pc-windows-msvc"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[build-gpu] WARN: DLL injection failed (exit code $LASTEXITCODE)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[build-gpu] WARN: inject-gpu-dlls.ps1 not found - NSIS will lack $GpuLabel DLLs" -ForegroundColor Yellow
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] NSIS installer : $releaseDir\bundle\nsis" -ForegroundColor Cyan
    }
    else {
        # Fast build: binary only, no NSIS
        cargo tauri build --no-bundle --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        if ($CUDA) {
            $required = @(
                "cudart64_12.dll",
                "cublas64_12.dll",
                "cufft64_11.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll"
            )
        } else {
            $required = @(
                "onnxruntime.dll",
                "onnxruntime_providers_shared.dll",
                "onnxruntime_providers_dml.dll"
            )
        }
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical $GpuLabel DLLs verified in release dir." -ForegroundColor Green
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] To create an NSIS installer, re-run with -Bundle" -ForegroundColor DarkGray
        Write-Host "[build-gpu] To create a portable ZIP, run: .\scripts\pack-portable.ps1 -SkipBuild" -ForegroundColor DarkGray
    }
} finally {
    Pop-Location
    Remove-Item Env:ORT_LIB_LOCATION -ErrorAction SilentlyContinue
}
