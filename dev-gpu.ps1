# dev-gpu.ps1
# Run the HiFiShifter dev server with CUDA/GPU support enabled.
# Automatically detects and configures the project-local standalone Rust environment if present.

param(
    [string]$OrtLibDir = $env:ORT_LIB_LOCATION
)

$SrcTauri = Join-Path $PSScriptRoot "backend\src-tauri"

# 1. Setup Standalone Rust Environment if detected locally
$localRustCargo = Join-Path $PSScriptRoot ".rust\cargo"
$localRustup   = Join-Path $PSScriptRoot ".rust\rustup"

if (Test-Path $localRustCargo) {
    Write-Host "[dev-gpu] Standalone Rust environment detected at $localRustCargo" -ForegroundColor Green
    $env:CARGO_HOME  = $localRustCargo
    $env:RUSTUP_HOME = $localRustup
    $binPath = Join-Path $localRustCargo "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }
}

# 2. Setup ONNX Runtime GPU Library Location
if (-not $OrtLibDir) {
    Write-Host "[dev-gpu] ORT_LIB_LOCATION not set. Trying to auto-detect..." -ForegroundColor Yellow
    $candidate = Join-Path $env:USERPROFILE ".cache\ort\lib"
    if (Test-Path $candidate) {
        $OrtLibDir = $candidate
        Write-Host "[dev-gpu] Found: $OrtLibDir" -ForegroundColor Green
    } else {
        Write-Error "[dev-gpu] Cannot find ONNX Runtime GPU library. Run .\setup-gpu-deps.ps1 first."
        exit 1
    }
}

$env:ORT_LIB_LOCATION      = $OrtLibDir
$env:ORT_PREFER_DYNAMIC_LINK = "1"
$env:HIFISHIFTER_DEBUG_COMMANDS = "1"

Write-Host "[dev-gpu] ORT_LIB_LOCATION = $env:ORT_LIB_LOCATION" -ForegroundColor Cyan

# 3. Pre-stage ORT runtime DLLs into the cargo debug output dir so the exe can
#    find them at runtime (cargo does not copy them automatically).
$DebugDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\debug"
New-Item -ItemType Directory -Path $DebugDir -Force | Out-Null

$OrtDlls = @(
    "onnxruntime.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll"
)

foreach ($dll in $OrtDlls) {
    $src = Join-Path $OrtLibDir $dll
    $dst = Join-Path $DebugDir $dll
    if (Test-Path $src) {
        if (-not (Test-Path $dst)) {
            Copy-Item $src $dst -Force
            Write-Host "[dev-gpu] Staged $dll -> debug dir" -ForegroundColor DarkGray
        }
    } else {
        Write-Warning "[dev-gpu] ORT DLL not found: $src"
    }
}

Write-Host "[dev-gpu] Starting Tauri dev with GPU support..." -ForegroundColor Cyan

Set-Location $PSScriptRoot
cargo tauri dev --features onnx
