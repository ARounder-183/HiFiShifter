<#
.SYNOPSIS
    One-stop Windows development environment setup for HiFiShifter.

.DESCRIPTION
    Installs and configures everything needed for HiFiShifter development on Windows:
    - Portable Rust toolchain (opt-in: -InstallRust, project-local at .rust/)
    - ONNX Runtime GPU package (DirectML, DirectX 12)
    - Frontend npm dependencies

    DirectML works on any DirectX 12 GPU (NVIDIA, AMD, Intel Arc) and is always
    available on Windows.  OpenCL provides cross-platform GPU acceleration.

    Rust installation is DISABLED by default - the script assumes a system-wide
    Rust installation.  Pass -InstallRust for a project-local portable toolchain.

    Safe to re-run - already-installed components are detected and skipped.
    To just load the Rust environment without installing anything:
        . .\scripts\setup-windows.ps1 -LoadEnv

.PARAMETER InstallRust
    Install a project-local portable Rust toolchain into .rust/.
    Disabled by default - assumes a system-wide Rust installation is available.

.PARAMETER SkipOrt
    Skip download of ONNX Runtime binaries.

.PARAMETER SkipFrontend
    Skip npm ci for frontend dependencies.

.PARAMETER LoadEnv
    Only load environment variables into the current shell (dot-source mode).
    Sets CARGO_HOME, RUSTUP_HOME, and updates PATH.

.PARAMETER LocalOrtDir
    Path to a pre-extracted ONNX Runtime installation (must contain lib/ and
    include/ subdirectories).

.PARAMETER LocalPackage
    Path to a locally-downloaded ONNX Runtime ZIP archive.

.EXAMPLE
    .\scripts\setup-windows.ps1
    # Default: ORT GPU package (DirectML + OpenCL baseline)

.EXAMPLE
    .\scripts\setup-windows.ps1 -InstallRust
    # Full setup including project-local Rust

.EXAMPLE
    .\scripts\setup-windows.ps1 -LocalOrtDir "D:\ort\onnxruntime-win-x64-gpu-1.24.1"
    # Use a pre-extracted local ORT installation (no network)

.EXAMPLE
    . .\scripts\setup-windows.ps1 -LoadEnv
    # Just load Rust environment variables into current shell
#>

[CmdletBinding()]
param(
    [Parameter()][switch]$InstallRust,
    [Parameter()][switch]$SkipOrt,
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
    Write-Host "-- [1/3] Portable Rust Toolchain --" -ForegroundColor Cyan

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

    $binPath = Join-Path $CargoHome "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }

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
    Write-Host "-- [1/3] Portable Rust Toolchain (skipped - use -InstallRust to enable) --" -ForegroundColor DarkGray
}

# ============================================================
# Step 2: ONNX Runtime (GPU package is the default)
# ============================================================
if (-not $SkipOrt) {
    Write-Host "-- [2/3] ONNX Runtime GPU (DirectML, DirectX 12) --" -ForegroundColor Cyan

    $downloadOrtScript = Join-Path $PSScriptRoot "download-ort.ps1"
    if (Test-Path $downloadOrtScript) {
        if ($LocalOrtDir) {
            & $downloadOrtScript -LocalOrtDir $LocalOrtDir -DestDir $OrtBundleDir
        } elseif ($LocalPackage) {
            & $downloadOrtScript -LocalPackage $LocalPackage -DestDir $OrtBundleDir
        } else {
            & $downloadOrtScript -DestDir $OrtBundleDir
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
    Write-Host "-- [2/3] ONNX Runtime (skipped) --" -ForegroundColor DarkGray
}

# ============================================================
# Step 3: Frontend Dependencies
# ============================================================
if (-not $SkipFrontend) {
    Write-Host "-- [3/3] Frontend Dependencies (npm) --" -ForegroundColor Cyan
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
    Write-Host "-- [3/3] Frontend Dependencies (skipped) --" -ForegroundColor DarkGray
}

# ============================================================
# Summary
# ============================================================
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    .\scripts\build-gpu.ps1              - DirectML GPU build" -ForegroundColor White
Write-Host "    .\scripts\build-gpu.ps1 -Dev         - Dev server with hot reload" -ForegroundColor White
Write-Host "    .\scripts\pack-portable.ps1          - Create portable ZIP" -ForegroundColor White
Write-Host ""
