<#
.SYNOPSIS
    Verify that critical CUDA/ORT DLLs are present in the release directory.

.DESCRIPTION
    Checks the release and NSIS bundle directories for the required DLLs
    needed for GPU-accelerated inference. Used by CI as a post-build gate.

.PARAMETER ReleaseDir
    Path to the release directory (cargo target/<triple>/release).

.EXAMPLE
    .\scripts\verify-cuda-dlls.ps1 -ReleaseDir "backend\src-tauri\target\x86_64-pc-windows-msvc\release"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseDir
)

$ErrorActionPreference = "Stop"

$required = @(
    # ONNX Runtime
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
    # CUDA Runtime
    "cudart64_12.dll",
    "curand64_10.dll",
    # cuBLAS
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    # cuFFT
    "cufft64_11.dll",
    "cufftw64_11.dll",
    # cuDNN 9.x
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll"
)

Write-Host "=== Release directory contents ==="
Get-ChildItem "$ReleaseDir\*.dll" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name) ($sizeMB MB)"
}
Write-Host ""

$missing = $required | Where-Object { -not (Test-Path (Join-Path $ReleaseDir $_)) }
if ($missing) {
    Write-Host "ERROR: Missing critical DLLs: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All $($required.Count) critical CUDA DLLs verified in release directory." -ForegroundColor Green

# Also check the NSIS bundle directory
$nsisDir = Join-Path $ReleaseDir "bundle\nsis"
if (Test-Path $nsisDir) {
    Write-Host "=== NSIS bundle DLLs ==="
    Get-ChildItem "$nsisDir\*.dll" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.FullName)"
    }
}
