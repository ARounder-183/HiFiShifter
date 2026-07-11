# build-gpu.ps1
# Build the HiFiShifter release executable with CUDA/GPU support enabled.
# Automatically detects and configures the project-local standalone Rust environment if present.

param(
    [string]$OrtLibDir = $env:ORT_LIB_LOCATION
)

# 1. Setup Standalone Rust Environment if detected locally
$localRustCargo = Join-Path $PSScriptRoot ".rust\cargo"
$localRustup = Join-Path $PSScriptRoot ".rust\rustup"

if (Test-Path $localRustCargo) {
    Write-Host "[build-gpu] Standalone Rust environment detected at $localRustCargo" -ForegroundColor Green
    $env:CARGO_HOME = $localRustCargo
    $env:RUSTUP_HOME = $localRustup
    $binPath = Join-Path $localRustCargo "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }
}

# 2. Setup ONNX Runtime GPU Library Location
if (-not $OrtLibDir) {
    Write-Host "[build-gpu] ORT_LIB_LOCATION not set. Trying to auto-detect..." -ForegroundColor Yellow
    $candidate = Join-Path $env:USERPROFILE ".cache\ort\lib"
    if (Test-Path $candidate) {
        $OrtLibDir = $candidate
        Write-Host "[build-gpu] Found: $OrtLibDir" -ForegroundColor Green
    } else {
        Write-Error "[build-gpu] Cannot find ONNX Runtime GPU library. Run .\setup-gpu-deps.ps1 first."
        exit 1
    }
}

$env:ORT_LIB_LOCATION = $OrtLibDir
$env:ORT_USE_CUDA = "1"

Write-Host "[build-gpu] ORT_LIB_LOCATION = $env:ORT_LIB_LOCATION" -ForegroundColor Cyan
Write-Host "[build-gpu] Starting Tauri build with GPU support..." -ForegroundColor Cyan

Set-Location $PSScriptRoot
cargo tauri build --features onnx

# ── Post-build: stage CUDA runtime DLLs into the release directory ──────────────
# Without these, ORT reports ep=cuda but silently executes on CPU at runtime,
# causing 50x slowdown. onnxruntime_providers_cuda.dll loads fine, but
# cublas/cudnn are resolved lazily at session.run() time.
$releaseDir = Join-Path $PSScriptRoot "backend\src-tauri\target\x86_64-pc-windows-msvc\release"

$cudaDllPatterns = @("cublas*.dll","cublasLt*.dll","cudnn*.dll","cufft*.dll","cufftw*.dll","curand*.dll")

function Copy-CudaDlls($sourceDir, $label) {
    $staged = 0
    foreach ($pattern in $cudaDllPatterns) {
        $files = Get-ChildItem (Join-Path $sourceDir $pattern) -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            $dst = Join-Path $releaseDir $f.Name
            if (-not (Test-Path $dst)) {
                Copy-Item $f.FullName $dst -Force
                Write-Host "[build-gpu] Staged ($label): $($f.Name)" -ForegroundColor Green
                $staged++
            }
        }
    }
    return $staged
}

# Priority 1: ORT_LIB_LOCATION — may already contain cuDNN if populated from dev branch
$staged = Copy-CudaDlls $OrtLibDir "ORT dir"

# Priority 2: Sibling dev repo ORT lib dir — dev branch ships cuDNN 9 DLLs there
$devOrtDir = Join-Path $PSScriptRoot "..\..\dev\onnxruntime\onnxruntime-win-x64-gpu-1.24.1\lib"
if (Test-Path $devOrtDir) {
    $staged += Copy-CudaDlls $devOrtDir "dev ORT dir"
}

# Priority 3: System CUDA Toolkit install
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaBase) {
    $latest = Get-ChildItem $cudaBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latest) {
        $staged += Copy-CudaDlls (Join-Path $latest.FullName "bin") "CUDA toolkit"
    }
}

# Verify the critical DLLs are present
$required = @("cublas64_12.dll", "cudnn64_9.dll")
$missing = $required | Where-Object { -not (Test-Path (Join-Path $releaseDir $_)) }
if ($missing) {
    Write-Host ""
    Write-Host "[build-gpu] WARN: Missing critical CUDA DLLs: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "[build-gpu]       GPU inference will silently fall back to CPU (~50x slower)." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[build-gpu] Run the following to download them from NVIDIA CDN:" -ForegroundColor Cyan
    Write-Host "            .\scripts\download-cuda-runtime.ps1" -ForegroundColor White
    Write-Host "[build-gpu] Then re-run: .\build-gpu.ps1" -ForegroundColor Cyan
} else {
    Write-Host "[build-gpu] All critical CUDA DLLs present - GPU acceleration ready." -ForegroundColor Green
}

Write-Host "[build-gpu] Release binary: $(Join-Path $releaseDir 'HiFiShifter.exe')" -ForegroundColor Cyan
