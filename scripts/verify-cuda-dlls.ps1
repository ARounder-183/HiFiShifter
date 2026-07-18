<#
.SYNOPSIS
    Verify that critical GPU (CUDA/DirectML) + ORT DLLs are present in the
    release directory.

.DESCRIPTION
    Checks the release and NSIS bundle directories for the required DLLs
    needed for GPU-accelerated inference. Used by CI as a post-build gate.

    Default mode: verifies CUDA DLLs (cuBLAS, cuDNN, cuFFT, cuRAND).
    -DirectML mode: verifies DirectML DLLs (onnxruntime_providers_dml.dll).

.PARAMETER ReleaseDir
    Path to the release directory (cargo target/<triple>/release).

.PARAMETER DirectML
    Verify DirectML (DX12 GPU) DLLs instead of CUDA DLLs.

.EXAMPLE
    .\scripts\verify-cuda-dlls.ps1 -ReleaseDir "backend\src-tauri\target\x86_64-pc-windows-msvc\release"

.EXAMPLE
    .\scripts\verify-cuda-dlls.ps1 -DirectML -ReleaseDir "backend\src-tauri\target\x86_64-pc-windows-msvc\release"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseDir,
    [switch]$DirectML
)

$ErrorActionPreference = "Stop"

if ($DirectML) {
    $label = "DirectML"
    $required = @(
        # ONNX Runtime core (DirectML is compiled in, no separate provider DLL)
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll"
    )
} else {
    $label = "CUDA"
    $required = @(
        # ONNX Runtime core
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        # CUDA/TensorRT providers
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
}

Write-Host "=== Release directory contents ($label mode) ==="
Get-ChildItem "$ReleaseDir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name) ($sizeMB MB)"
}
Write-Host ""

$missing = $required | Where-Object { -not (Test-Path (Join-Path $ReleaseDir $_)) }
if ($missing) {
    Write-Host "ERROR: Missing critical $label DLLs: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All $($required.Count) critical $label DLLs verified in release directory." -ForegroundColor Green

# Also check the NSIS bundle directory
$nsisDir = Join-Path $ReleaseDir "bundle\nsis"
if (Test-Path $nsisDir) {
    Write-Host "=== NSIS bundle DLLs ==="
    Get-ChildItem "$nsisDir\*.dll" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.FullName)"
    }
}
