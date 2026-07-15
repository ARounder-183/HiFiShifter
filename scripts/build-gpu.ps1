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
        Write-Host "[build-gpu] Generating GPU resource config for NSIS..." -ForegroundColor Cyan
        $stageScript = Join-Path $PSScriptRoot "stage-tauri-resources.ps1"
        if (-not (Test-Path $stageScript)) {
            throw "[build-gpu] stage-tauri-resources.ps1 not found."
        }
        & $stageScript -ProjectRoot $ProjectRoot
        if (-not $?) {
            throw "[build-gpu] Resource config generation failed."
        }

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

        # Inject the precompiled engine DLL into the NSIS installer.
        # This DLL is excluded from tauri.windows.conf.json by stage-tauri-resources.ps1
        # because NSIS (32-bit) crashes on mmap when solid-LZMA-compressing such a large file.
        # inject-nsis-large-dll.ps1 adds it uncompressed and re-runs makensis.
        $injectScript = Join-Path $PSScriptRoot "inject-nsis-large-dll.ps1"
        $precompiledDll = Join-Path $SrcTauri "third_party\ort-bundle\cudnn_engines_precompiled64_9.dll"
        if (Test-Path $injectScript) {
            if (Test-Path $precompiledDll) {
                Write-Host "[build-gpu] Injecting precompiled engine DLL into NSIS installer..." -ForegroundColor Cyan
                & $injectScript -TargetTriple "x86_64-pc-windows-msvc"
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[build-gpu] WARN: NSIS DLL injection failed (exit code $LASTEXITCODE)" -ForegroundColor Yellow
                    Write-Host "[build-gpu] The installer is usable - cuDNN will fall back to runtime-compiled engines." -ForegroundColor DarkGray
                }
            } else {
                Write-Host "[build-gpu] Precompiled engine DLL not found - skipping NSIS injection" -ForegroundColor DarkYellow
                Write-Host "[build-gpu] cuDNN will use runtime-compiled engines on first launch." -ForegroundColor DarkGray
            }
        } else {
            Write-Host "[build-gpu] inject-nsis-large-dll.ps1 not found - skipping NSIS injection" -ForegroundColor DarkYellow
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
        # --- Fast build: use cargo tauri build with empty targets ---------------
        # `cargo build` does not set up the Tauri context the same way the CLI
        # does - binaries built without the CLI pipeline serve `devUrl` at
        # runtime (localhost:5173).  Workaround: let the CLI build, but override
        # `bundle.targets` to an empty list so NSIS is skipped.
        Write-Host "[build-gpu] Configuring Tauri to skip NSIS for fast build..." -ForegroundColor Cyan
        $noTargets = @{ bundle = @{ targets = @() } }
        $noJson = $noTargets | ConvertTo-Json -Depth 2 -Compress
        $tmpConf = Join-Path $SrcTauri "tauri.windows.conf.json"
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($tmpConf, $noJson, $utf8)

        cargo tauri build --features $Features
        if ($LASTEXITCODE -ne 0) {
            Remove-Item $tmpConf -Force -ErrorAction SilentlyContinue
            throw "[build-gpu] cargo tauri build failed (exit code $LASTEXITCODE)"
        }

        Remove-Item $tmpConf -Force -ErrorAction SilentlyContinue

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
