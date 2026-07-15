<#
.SYNOPSIS
    One-stop Windows development environment setup for HiFiShifter.

.DESCRIPTION
    Installs and configures everything needed for HiFiShifter development on Windows:
    - Portable Rust toolchain (opt-in: -InstallRust, project-local at .rust/)
    - ONNX Runtime GPU (downloaded from GitHub, or from a local source)
    - CUDA runtime DLLs (cuBLAS, cuDNN, etc.)
    - Frontend npm dependencies

    Rust installation is DISABLED by default - the script assumes you already
    have a system-wide Rust installation.  Pass -InstallRust to set up a
    project-local portable toolchain instead.

    This script is designed to be safe to re-run - already-installed components
    are detected and skipped. Also supports being dot-sourced to just load the
    Rust environment into the current shell:
        . .\scripts\setup-windows.ps1 -LoadEnv

    Replaces the old root-level scripts:
        setup-rust-env.ps1, env-rust.ps1, setup-gpu-deps.ps1

.PARAMETER InstallRust
    Install a project-local portable Rust toolchain into .rust/.
    Disabled by default - assumes a system-wide Rust installation is available.

.PARAMETER SkipOrt
    Skip download of ONNX Runtime GPU binaries.

.PARAMETER SkipCudaRuntime
    Skip download of CUDA runtime DLLs (cuBLAS, cuDNN, etc.).

.PARAMETER SkipFrontend
    Skip npm ci for frontend dependencies.

.PARAMETER LoadEnv
    Only load environment variables into the current shell (dot-source mode).
    Sets CARGO_HOME, RUSTUP_HOME, and updates PATH.

.PARAMETER LocalOrtDir
    Path to a pre-extracted ONNX Runtime installation (must contain lib/ and
    include/ subdirectories).  Files are copied locally - no network access.
    Equivalent to the old ORT_LIB_LOCATION workflow.

.PARAMETER LocalPackage
    Path to a locally-downloaded ONNX Runtime ZIP archive.  Extracted and
    copied locally - no network access required.

.EXAMPLE
    .\scripts\setup-windows.ps1 -InstallRust
    # Full setup including project-local Rust toolchain

.EXAMPLE
    .\scripts\setup-windows.ps1 -SkipCudaRuntime
    # Only download ORT and install frontend deps

.EXAMPLE
    .\scripts\setup-windows.ps1 -LocalOrtDir "D:\ort\onnxruntime-win-x64-gpu-1.24.1"
    # Use a pre-extracted local ORT installation (no GitHub access needed)

.EXAMPLE
    .\scripts\setup-windows.ps1 -LocalPackage "D:\Downloads\onnxruntime-win-x64-gpu-1.24.1.zip" -SkipCudaRuntime
    # Use a locally-downloaded ORT ZIP archive

.EXAMPLE
    $env:ORT_MIRROR = "https://ghproxy.com/https://github.com"
    .\scripts\setup-windows.ps1
    # Download through a mirror for faster access

.EXAMPLE
    . .\scripts\setup-windows.ps1 -LoadEnv
    # Just load environment variables into current shell
#>

[CmdletBinding()]
param(
    [Parameter()][switch]$InstallRust,
    [Parameter()][switch]$SkipOrt,
    [Parameter()][switch]$SkipCudaRuntime,
    [Parameter()][switch]$SkipFrontend,
    [Parameter()][switch]$LoadEnv,
    [Parameter()][string]$LocalOrtDir,
    [Parameter()][string]$LocalPackage
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RustDir   = Join-Path $ProjectRoot ".rust"
$CargoHome = Join-Path $RustDir "cargo"
$RustupHome = Join-Path $RustDir "rustup"
$CargoBin  = Join-Path $CargoHome "bin\cargo.exe"
# Single canonical location for GPU DLLs - project-local, gitignored.
# build.rs reads from here; stage-tauri-resources.ps1 reads from here.
$OrtBundleDir = Join-Path $ProjectRoot "backend\src-tauri\third_party\ort-bundle"

# ============================================================
# LoadEnv mode: just configure environment and exit
# ============================================================
if ($LoadEnv) {
    if (Test-Path $RustDir) {
        $env:CARGO_HOME  = $CargoHome
        $env:RUSTUP_HOME = $RustupHome
        $binPath = Join-Path $CargoHome "bin"
        if ($env:PATH -notlike "*$binPath*") {
            $env:PATH = "$binPath;$env:PATH"
        }
        Write-Host "[setup-windows] Standalone Rust environment loaded." -ForegroundColor Green
        Write-Host "  CARGO_HOME  = $env:CARGO_HOME"
        Write-Host "  RUSTUP_HOME = $env:RUSTUP_HOME"
        cargo --version 2>$null
    } else {
        Write-Warning "[setup-windows] No local Rust installation found at .rust\"
    }
    if (Test-Path $OrtBundleDir) {
        Write-Host "[setup-windows] GPU DLL dir = $OrtBundleDir" -ForegroundColor Green
    }
    return
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter - Windows Development Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Step 1: Portable Rust Toolchain
# ============================================================
if ($InstallRust) {
    Write-Host "-- [1/4] Portable Rust Toolchain --" -ForegroundColor Cyan

    $env:CARGO_HOME  = $CargoHome
    $env:RUSTUP_HOME = $RustupHome

    if (Test-Path $CargoBin) {
        Write-Host "  Status: Already installed" -ForegroundColor Green
        & $CargoBin --version
    } else {
        Write-Host "  Installing Rust into $RustDir ..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $RustDir -Force | Out-Null

        $InstallerPath = Join-Path $RustDir "rustup-init.exe"
        try {
            Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $InstallerPath
        } catch {
            Write-Error "Failed to download rustup-init.exe. Check your internet connection."
            exit 1
        }

        $proc = Start-Process -FilePath $InstallerPath `
            -ArgumentList "-y", "--no-modify-path", "--default-toolchain", "stable" `
            -Wait -NoNewWindow -PassThru
        Remove-Item -Force $InstallerPath

        if (-not (Test-Path $CargoBin)) {
            Write-Error "Rust installation completed but cargo.exe was not found."
            exit 1
        }
        Write-Host "  Installed:" -ForegroundColor Green
        & $CargoBin --version
    }

    # Add cargo/bin to PATH for this session
    $binPath = Join-Path $CargoHome "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }

    # Install tauri-cli
    $CargoTauri = Join-Path $CargoHome "bin\cargo-tauri.exe"
    if (Test-Path $CargoTauri) {
        Write-Host "  tauri-cli:  Already installed" -ForegroundColor Green
        & $CargoTauri --version
    } else {
        Write-Host "  Installing tauri-cli..." -ForegroundColor Yellow
        & $CargoBin install tauri-cli --locked
        if ($LASTEXITCODE -ne 0) {
            Write-Error "tauri-cli installation failed."
            exit 1
        }
        Write-Host "  tauri-cli installed." -ForegroundColor Green
    }

    Write-Host ""
} else {
    Write-Host "-- [1/4] Portable Rust Toolchain (skipped - use -InstallRust to enable) --" -ForegroundColor DarkGray
}

# ============================================================
# Step 2: ONNX Runtime GPU
# ============================================================
if (-not $SkipOrt) {
    Write-Host "-- [2/4] ONNX Runtime GPU --" -ForegroundColor Cyan

    $downloadOrtScript = Join-Path $PSScriptRoot "download-ort.ps1"
    if (Test-Path $downloadOrtScript) {
        # Use direct named-parameter invocation rather than splatting (@args)
        # to avoid a PowerShell parameter-binding bug with array-based splatting.
        if ($LocalOrtDir) {
            & $downloadOrtScript -LocalOrtDir $LocalOrtDir -DestDir $OrtBundleDir
        } elseif ($LocalPackage) {
            & $downloadOrtScript -LocalPackage $LocalPackage -DestDir $OrtBundleDir
        } else {
            & $downloadOrtScript -Gpu -DestDir $OrtBundleDir
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Error "ONNX Runtime download failed."
            exit 1
        }
    } else {
        Write-Error "download-ort.ps1 not found at $downloadOrtScript"
        exit 1
    }

    Write-Host ""
} else {
    Write-Host "-- [2/4] ONNX Runtime GPU (skipped) --" -ForegroundColor DarkGray
}

# ============================================================
# Step 3: CUDA Runtime DLLs
# ============================================================
if (-not $SkipCudaRuntime) {
    Write-Host "-- [3/4] CUDA Runtime DLLs (cuBLAS + cuDNN) --" -ForegroundColor Cyan

    # Resolve destination (same as ORT lib dir by default)
    $cudaDestDir = $OrtBundleDir

    $downloadCudaScript = Join-Path $PSScriptRoot "download-cuda-runtime.ps1"
    if (Test-Path $downloadCudaScript) {
        & $downloadCudaScript -DestDir $cudaDestDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: CUDA runtime download had errors. GPU acceleration may not work." -ForegroundColor Yellow
        }
    } else {
        Write-Host "WARNING: download-cuda-runtime.ps1 not found. GPU acceleration will NOT work." -ForegroundColor Yellow
        Write-Host "  CUDA runtime DLLs (cuBLAS 12 + cuDNN 9) are REQUIRED for GPU inference." -ForegroundColor Yellow
    }

    Write-Host ""
} else {
    Write-Host "-- [3/4] CUDA Runtime DLLs (skipped) --" -ForegroundColor DarkGray
}

# ============================================================
# Step 4: Frontend Dependencies
# ============================================================
if (-not $SkipFrontend) {
    Write-Host "-- [4/4] Frontend Dependencies (npm) --" -ForegroundColor Cyan
    Push-Location (Join-Path $ProjectRoot "frontend")
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "npm ci failed; trying npm install..."
            npm install
        }
    } finally {
        Pop-Location
    }
    Write-Host "  Frontend dependencies installed." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "-- [4/4] Frontend Dependencies (skipped) --" -ForegroundColor DarkGray
}

# ============================================================
# Summary
# ============================================================
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    .\scripts\build-gpu.ps1           - Build release with GPU" -ForegroundColor White
Write-Host "    .\scripts\build-gpu.ps1 -Dev      - Run dev server with GPU" -ForegroundColor White
Write-Host "    .\scripts\pack-portable.ps1       - Create portable ZIP" -ForegroundColor White
Write-Host ""
