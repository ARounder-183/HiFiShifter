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

$env:ORT_USE_CUDA = "1"

# Tell ort-sys to link against the SAME ONNX Runtime that will be staged
# to the binary directory by build.rs.  Without this, ort-sys falls back to
# its `download-binaries` feature and may fetch a different ORT version,
# causing an FFI-version mismatch that manifests as a main-thread hang
# during CUDA EP initialization.
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
Write-Host "[build-gpu] Starting GPU $ModeLabel..." -ForegroundColor Cyan
Write-Host ""

Push-Location $ProjectRoot
try {
    $releaseDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\release"

    if ($Dev) {
        # --- Dev mode: hot-reload dev server ---------------------------------
        cargo tauri dev --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri dev failed (exit code $LASTEXITCODE)"
        }
    }
    elseif ($Bundle) {
        # --- Full build: binary + NSIS installer (SLOW - large DLLs) ---------

        # tauri.windows.conf.json is a COMMITTED static file containing only
        # vslib_x64.dll + SoundTouchDLL.dll.  GPU DLLs are NOT listed there —
        # they are injected post-build by inject-gpu-dlls.ps1 (see below).
        # Nothing modifies the committed config during the build.  Ctrl+C safe.
        cargo tauri build --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        # Verify
        $required = @(
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cufft64_11.dll",
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll"
        )
        $missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
        if ($missing) {
            Write-Host "[build-gpu] WARN: Missing DLLs in release: $($missing -join ', ')" -ForegroundColor Yellow
        } else {
            Write-Host "[build-gpu] All critical CUDA DLLs verified in release dir." -ForegroundColor Green
        }

        # Inject ALL GPU DLLs (ort-bundle/) into the NSIS installer.
        # These are NOT in tauri.windows.conf.json — the committed config is
        # never modified.  inject-gpu-dlls.ps1 patches the generated NSIS
        # script and re-runs makensis.
        $injectScript = Join-Path $PSScriptRoot "inject-gpu-dlls.ps1"
        if (Test-Path $injectScript) {
            Write-Host "[build-gpu] Injecting GPU DLLs into NSIS installer..." -ForegroundColor Cyan
            & $injectScript -TargetTriple "x86_64-pc-windows-msvc"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[build-gpu] WARN: GPU DLL injection failed (exit code $LASTEXITCODE)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[build-gpu] WARN: inject-gpu-dlls.ps1 not found — NSIS will lack GPU DLLs" -ForegroundColor Yellow
        }

        Write-Host "[build-gpu] Release binary : $releaseDir\HiFiShifter.exe" -ForegroundColor Cyan
        Write-Host "[build-gpu] NSIS installer : $releaseDir\bundle\nsis" -ForegroundColor Cyan
    }
    else {
        # --- Fast build: binary only, no NSIS ------------------------------
        # Use --no-bundle to skip NSIS without touching any config file.
        cargo tauri build --no-bundle --features $Features
        if ($LASTEXITCODE -ne 0) {
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        # Verify DLLs
        $required = @(
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cufft64_11.dll",
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll"
        )
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
    # Don't let ORT_LIB_LOCATION leak into the caller's session -
    # it would break plain CPU builds (cargo tauri build without CUDA).
    Remove-Item Env:ORT_LIB_LOCATION -ErrorAction SilentlyContinue
}
