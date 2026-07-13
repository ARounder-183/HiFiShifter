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
$env:ORT_USE_CUDA           = "1"
$env:HIFISHIFTER_DEBUG_COMMANDS = "1"

Write-Host "[dev-gpu] ORT_LIB_LOCATION = $env:ORT_LIB_LOCATION" -ForegroundColor Cyan

# 3. Pre-stage ORT runtime DLLs into the cargo debug output dir so the exe can
#    find them at runtime (cargo does not copy them automatically).
$DebugDir = Join-Path $SrcTauri "target\x86_64-pc-windows-msvc\debug"
New-Item -ItemType Directory -Path $DebugDir -Force | Out-Null

function Copy-DllIfNewer($SourceDir, $Pattern, $Label = "") {
    $copied = 0
    $files = Get-ChildItem (Join-Path $SourceDir $Pattern) -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $dst = Join-Path $DebugDir $f.Name
        if (-not (Test-Path $dst) -or ($f.LastWriteTime -gt (Get-Item $dst).LastWriteTime)) {
            Copy-Item $f.FullName $dst -Force
            Write-Host "[dev-gpu] Staged $($f.Name) -> debug dir ($Label)" -ForegroundColor DarkGray
            $copied++
        }
    }
    return $copied
}

# 3a. ORT core + CUDA provider DLLs (must be present)
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

# 3b. CUDA runtime DLLs (cuBLAS, cuDNN, cuFFT, cuRAND, cuDART)
# These are REQUIRED at runtime for CUDA EP to actually execute on GPU.
# Without them, CUDA EP loads but silently falls back to CPU (50x slowdown).
$totalStaged = 0

# Priority 1: ORT_LIB_LOCATION (may contain cuDNN)
$totalStaged += Copy-DllIfNewer $OrtLibDir "*.dll" "ORT dir"

# Priority 2: Sibling dev ORT lib dir
$devOrtDir = Join-Path $PSScriptRoot "..\..\dev\onnxruntime\onnxruntime-win-x64-gpu-1.24.1\lib"
if (Test-Path $devOrtDir) {
    $totalStaged += Copy-DllIfNewer $devOrtDir "*.dll" "dev ORT dir"
}

# Priority 3: System CUDA Toolkit
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaBase) {
    $latest = Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latest) {
        $totalStaged += Copy-DllIfNewer (Join-Path $latest.FullName "bin") "*.dll" "CUDA toolkit"
    }
}

# Priority 4: Check for CUDA DLLs in PATH (common with cuDNN standalone installs)
$pathDirs = $env:PATH -split ';' | Where-Object { $_ -and (Test-Path $_) }
foreach ($dir in $pathDirs) {
    $totalStaged += Copy-DllIfNewer $dir "cudnn64_9.dll" "PATH"
    if ((Get-ChildItem (Join-Path $dir "cudnn64_9.dll") -ErrorAction SilentlyContinue)) {
        break
    }
}

# Verify critical CUDA DLLs — if missing, try auto-download
$criticalDlls = @("cudart64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")
$missing = $criticalDlls | Where-Object { -not (Test-Path (Join-Path $DebugDir $_)) }
if ($missing) {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║  MISSING CUDA RUNTIME DLLs — GPU will NOT work!          ║" -ForegroundColor Red
    Write-Host "║  CUDA EP will register but silently fall back to CPU.     ║" -ForegroundColor Red
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "[dev-gpu] Missing: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "[dev-gpu] Attempting auto-download via download-cuda-runtime.ps1..." -ForegroundColor Cyan
    Write-Host ""

    $downloadScript = Join-Path $PSScriptRoot "scripts\download-cuda-runtime.ps1"
    if (Test-Path $downloadScript) {
        & $downloadScript -DestDir $OrtLibDir
        if ($LASTEXITCODE -eq 0) {
            # Re-stage the newly downloaded DLLs
            $totalStaged += Copy-DllIfNewer $OrtLibDir "*.dll" "fresh download"
            $missing = $criticalDlls | Where-Object { -not (Test-Path (Join-Path $DebugDir $_)) }
            if (-not $missing) {
                Write-Host "[dev-gpu] CUDA runtime DLLs downloaded and staged successfully." -ForegroundColor Green
            }
        }
    } else {
        Write-Host "[dev-gpu] download-cuda-runtime.ps1 not found." -ForegroundColor Red
    }

    # Final check
    $missing = $criticalDlls | Where-Object { -not (Test-Path (Join-Path $DebugDir $_)) }
    if ($missing) {
        Write-Host ""
        Write-Host "[dev-gpu] STILL MISSING: $($missing -join ', ')" -ForegroundColor Red
        Write-Host "[dev-gpu] GPU acceleration WILL NOT WORK. Fix:" -ForegroundColor Red
        Write-Host "         1. Run: .\setup-gpu-deps.ps1" -ForegroundColor White
        Write-Host "         2. Then: .\dev-gpu.ps1" -ForegroundColor White
    }
} else {
    Write-Host "[dev-gpu] All critical CUDA runtime DLLs present - GPU acceleration ready." -ForegroundColor Green
}

# Also stage CUDA DLLs from ORT_LIB_LOCATION to debug dir (in case they were
# placed there after the initial copy above)
$totalStaged += Copy-DllIfNewer $OrtLibDir "cublas*.dll" "ORT lib"
$totalStaged += Copy-DllIfNewer $OrtLibDir "cudnn*.dll" "ORT lib"

Write-Host "[dev-gpu] Starting Tauri dev with GPU support..." -ForegroundColor Cyan

Set-Location $PSScriptRoot
cargo tauri dev --features onnx,cuda,tensorrt
