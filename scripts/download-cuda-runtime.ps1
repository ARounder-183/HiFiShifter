<#
.SYNOPSIS
    Download CUDA runtime DLLs required for GPU acceleration in HiFiShifter.

.DESCRIPTION
    Downloads cuBLAS 12 and cuDNN 9 from NVIDIA's official redistribution CDN and
    extracts the required DLLs into ORT_LIB_LOCATION (or the auto-detected ORT
    cache directory). build-gpu.ps1 then stages them alongside the binary automatically.

    DLLs installed (~1.1 GB extracted):
      cublas64_12.dll / cublasLt64_12.dll    (cuBLAS 12.6.4.1)
      cudnn64_9.dll                           (cuDNN 9.8.0 stub loader)
      cudnn_ops64_9.dll / cudnn_cnn64_9.dll / cudnn_adv64_9.dll
      cudnn_engines_precompiled64_9.dll / cudnn_engines_runtime_compiled64_9.dll
      cudnn_graph64_9.dll / cudnn_heuristic64_9.dll

    Re-run to refresh; already-present DLLs are skipped unless -Force is used.

.PARAMETER DestDir
    Directory to place the extracted DLLs.
    Defaults to ORT_LIB_LOCATION env var, then %USERPROFILE%\.cache\ort\lib.

.PARAMETER Force
    Re-download and overwrite even if DLLs already exist.

.EXAMPLE
    .\scripts\download-cuda-runtime.ps1
    .\scripts\download-cuda-runtime.ps1 -DestDir "C:\my\ort\lib"
    .\scripts\download-cuda-runtime.ps1 -Force
#>

param(
    [string]$DestDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Resolve destination directory
if (-not $DestDir) {
    if ($env:ORT_LIB_LOCATION -and (Test-Path $env:ORT_LIB_LOCATION)) {
        $DestDir = $env:ORT_LIB_LOCATION
    } else {
        $DestDir = Join-Path $env:USERPROFILE ".cache\ort\lib"
    }
}
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter CUDA Runtime Downloader" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Output: $DestDir" -ForegroundColor White
Write-Host ""

# Package definitions (verified NVIDIA CDN URLs, June 2026)
# libcublas index:  https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/
# cuDNN manifest:   https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.8.0.json
$Packages = @(
    @{
        Name = "cuBLAS 12.6.4.1"
        Url  = "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.6.4.1-archive.zip"
        Dlls = @("cublas64_12.dll", "cublasLt64_12.dll")
    },
    @{
        Name = "cuDNN 9.8.0 for CUDA 12"
        Url  = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip"
        Dlls = @(
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll"
        )
    }
)

function Expand-WithBestTool([string]$ZipPath, [string]$DestPath) {
    $sevenZip = @(
        "${env:ProgramFiles}\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($sevenZip) {
        & $sevenZip x "-o$DestPath" $ZipPath -y | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
    }
    Write-Host "    (7-Zip not found; using Expand-Archive - may be slow for large files)" -ForegroundColor DarkGray
    Expand-Archive -Path $ZipPath -DestinationPath $DestPath -Force
}

# Main download loop
$TempDir = Join-Path $env:TEMP "hifishifter-cuda-dl-$PID"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$totalStaged = 0
$failed = @()

foreach ($pkg in $Packages) {
    $name = $pkg.Name
    $url  = $pkg.Url
    $dlls = $pkg.Dlls

    # Skip if all DLLs already present (unless -Force)
    $allPresent = ($dlls | Where-Object { -not (Test-Path (Join-Path $DestDir $_)) }).Count -eq 0
    if ($allPresent -and -not $Force) {
        Write-Host "[$name] Already installed - skipping (use -Force to re-download)" -ForegroundColor DarkGreen
        $totalStaged += $dlls.Count
        continue
    }

    Write-Host "[$name] Downloading..." -ForegroundColor Yellow
    Write-Host "  $url" -ForegroundColor DarkGray

    $zipFile    = Join-Path $TempDir ([System.IO.Path]::GetFileName($url))
    $extractDir = Join-Path $TempDir ($name -replace '[^a-zA-Z0-9]', '_')

    try {
        Invoke-WebRequest -Uri $url -OutFile $zipFile
        $sizeMB = [math]::Round((Get-Item $zipFile).Length / 1MB, 1)
        Write-Host "  Downloaded: $sizeMB MB" -ForegroundColor DarkGray
    } catch {
        Write-Host "  ERROR downloading ${name}: $_" -ForegroundColor Red
        $failed += $name
        continue
    }

    Write-Host "  Extracting..." -ForegroundColor DarkGray
    try {
        Expand-WithBestTool -ZipPath $zipFile -DestPath $extractDir
    } catch {
        Write-Host "  ERROR extracting ${name}: $_" -ForegroundColor Red
        $failed += $name
        continue
    }

    # Find and copy each DLL (NVIDIA zip internal structure varies by version)
    $pkgFailed = $false
    foreach ($dll in $dlls) {
        $src = Get-ChildItem -Path $extractDir -Recurse -Filter $dll -ErrorAction SilentlyContinue |
               Select-Object -First 1
        if ($src) {
            $dst = Join-Path $DestDir $dll
            Copy-Item $src.FullName $dst -Force
            $sizeMB = [math]::Round($src.Length / 1MB, 1)
            Write-Host "  + $dll ($sizeMB MB)" -ForegroundColor Green
            $totalStaged++
        } else {
            Write-Host "  ! $dll not found in archive" -ForegroundColor Yellow
            $pkgFailed = $true
        }
    }
    if ($pkgFailed) { $failed += $name }

    # Clean up zip to save disk space between packages
    Remove-Item $zipFile -Force -ErrorAction SilentlyContinue
}

Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "  Done! $totalStaged DLL(s) ready." -ForegroundColor Green
} else {
    Write-Host "  Completed with errors: $($failed -join ', ')" -ForegroundColor Yellow
}

$totalMB = [math]::Round(
    (Get-ChildItem $DestDir -Filter "*.dll" -ErrorAction SilentlyContinue |
     Measure-Object Length -Sum).Sum / 1MB, 1
)
Write-Host "  Dest: $DestDir ($totalMB MB total DLLs)" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: run .\build-gpu.ps1 to build with GPU support." -ForegroundColor Cyan
