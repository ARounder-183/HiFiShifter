<#
.SYNOPSIS
    下载 CUDA Runtime 和 cuDNN 的 redistributable DLL，用于打包到 HiFiShifter 发行版中。

.DESCRIPTION
    从 NVIDIA 官方 CDN 下载 ONNX Runtime CUDA EP 所需的最小运行时 DLL 集合，
    放置到 backend/src-tauri/third_party/cuda-runtime/ 目录。
    build.rs 会在构建时自动将这些 DLL 拷贝到产物目录，
    pack-portable.ps1 会将其打包进便携版 ZIP。

    下载的 DLL 清单（约 1.2 GB 解压后）：
      cudart64_12.dll         CUDA Runtime
      cublas64_12.dll         cuBLAS
      cublasLt64_12.dll       cuBLAS Lt
      cudnn64_9.dll           cuDNN
      cudnn_ops64_9.dll       cuDNN ops
      cudnn_cnn64_9.dll       cuDNN CNN
      cudnn_adv64_9.dll       cuDNN adv (optional)
      cufft64_12.dll          cuFFT (optional)
      cufftw64_12.dll         cuFFTW (optional)

    如需重新下载，删除目标目录后重新运行本脚本。

.PARAMETER CudaVersion
    CUDA 主版本号，默认 12.6

.PARAMETER CudnnVersion
    cuDNN 版本号，默认 9.8.0

.EXAMPLE
    .\scripts\download-cuda-runtime.ps1
    # 使用默认 CUDA 12.6 + cuDNN 9.8

.EXAMPLE
    .\scripts\download-cuda-runtime.ps1 -CudaVersion 12.8 -CudnnVersion 9.10
    # 指定版本
#>

param(
    [string]$CudaVersion = "12.6",
    [string]$CudnnVersion = "9.8"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$OutputDir = Join-Path $ProjectRoot "backend\src-tauri\third_party\cuda-runtime"

# ===== CUDA Redistributable 映射 =====
# 每个条目: (redist-name, dll-pattern)
# NVIDIA redist URL 格式:
#   https://developer.download.nvidia.com/compute/cuda/redist/{name}/windows-x86_64/{name}-windows-x86_64-{version}-archive.zip
# 版本号需查询: https://developer.download.nvidia.com/compute/cuda/redist/{name}/windows-x86_64/
#
# 以下使用 12.6 系列已验证可用的版本号

$CudaPackages = @(
    @{ Name = "cuda_cudart";     Version = "12.6.77";  Dlls = @("cudart64_12.dll") },
    @{ Name = "cuda_cublas";     Version = "12.6.4.1"; Dlls = @("cublas64_12.dll", "cublasLt64_12.dll") },
    @{ Name = "cuda_cufft";      Version = "11.3.0.1"; Dlls = @("cufft64_12.dll", "cufftw64_12.dll") },
    @{ Name = "cuda_curand";     Version = "10.3.7.77"; Dlls = @("curand64_12.dll") },
    @{ Name = "cuda_cusparse";   Version = "12.5.4.2"; Dlls = @("cusparse64_12.dll") },
    @{ Name = "cuda_nvrtc";      Version = "12.6.77";  Dlls = @("nvrtc64_12.dll", "nvrtc-builtins64_12.dll") },
    @{ Name = "cuda_cusolver";   Version = "11.7.1.2"; Dlls = @("cusolver64_12.dll", "cusolverMg64_12.dll") }
)

# cuDNN redist URL:
#   https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-{version}_cuda{cuver}-archive.zip

$CudnnPackages = @(
    @{
        Name    = "cudnn"
        Version = "${CudnnVersion}.0"
        CudaVer = "12"
        Dlls    = @("cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll")
    }
)

# ===== 工具函数 =====

function Test-Admin {
    $wid = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $prp = New-Object System.Security.Principal.WindowsPrincipal($wid)
    return $prp.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Expand-ZipWith7z {
    param([string]$ZipPath, [string]$DestDir)
    # 尝试 7z > Expand-Archive（7z 对大文件更稳定）
    $sevenZip = @(
        "${env:ProgramFiles}\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($sevenZip) {
        & $sevenZip x "-o$DestDir" $ZipPath -y | Out-Null
        if ($LASTEXITCODE -eq 0) { return $true }
    }
    # Fallback
    Expand-Archive -Path $ZipPath -DestinationPath $DestDir -Force
    return $true
}

# ===== 主流程 =====

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter CUDA Runtime 下载工具" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  CUDA:    $CudaVersion" -ForegroundColor White
Write-Host "  cuDNN:   $CudnnVersion" -ForegroundColor White
Write-Host "  输出:    $OutputDir" -ForegroundColor White
Write-Host ""

# 确保输出目录存在
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$TempDir = Join-Path $env:TEMP "hifishifter-cuda-dl-$pid"
if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$Failed = @()
$AllExtractedDlls = @()

# ── 下载 CUDA redist packages ──
Write-Host "[CUDA] 下载 redistributable packages..." -ForegroundColor Yellow

$CudaBaseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"

foreach ($pkg in $CudaPackages) {
    $name = $pkg.Name
    $ver = $pkg.Version
    $url = "$CudaBaseUrl/$name/windows-x86_64/${name}-windows-x86_64-${ver}-archive.zip"
    $zip = Join-Path $TempDir "$name.zip"
    $extractDir = Join-Path $TempDir $name

    Write-Host "  [$name] 下载中... $url" -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -ErrorAction Stop
    }
    catch {
        Write-Host "  [$name] 下载失败: $_" -ForegroundColor Red
        $Failed += $name
        continue
    }

    Write-Host "  [$name] 解压中..." -ForegroundColor Gray
    Expand-ZipWith7z -ZipPath $zip -DestDir $extractDir

    # 搜索 DLL 并复制
    $found = $false
    foreach ($dll in $pkg.Dlls) {
        $src = Get-ChildItem -Path $extractDir -Recurse -Filter $dll -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($src) {
            Copy-Item $src.FullName -Destination $OutputDir -Force
            Write-Host "    ✓ $dll" -ForegroundColor DarkGreen
            $AllExtractedDlls += $dll
            $found = $true
        }
        else {
            Write-Host "    ⚠ $dll (未在包中找到)" -ForegroundColor DarkYellow
        }
    }
    if (-not $found) {
        $Failed += $name
    }
}

# ── 下载 cuDNN redist packages ──
Write-Host "[cuDNN] 下载 redistributable packages..." -ForegroundColor Yellow

$CudnnBaseUrl = "https://developer.download.nvidia.com/compute/cudnn/redist"

foreach ($pkg in $CudnnPackages) {
    $name = $pkg.Name
    $ver = $pkg.Version
    $cuver = $pkg.CudaVer
    $url = "$CudnnBaseUrl/$name/windows-x86_64/${name}-windows-x86_64-${ver}_cuda${cuver}-archive.zip"
    $zip = Join-Path $TempDir "$name.zip"
    $extractDir = Join-Path $TempDir $name

    Write-Host "  [$name] 下载中... $url" -ForegroundColor Gray
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -ErrorAction Stop
    }
    catch {
        Write-Host "  [$name] 下载失败 (可能需要接受 NVIDIA Developer EULA): $_" -ForegroundColor Red
        Write-Host "  请手动访问 https://developer.nvidia.com/cudnn 下载 cuDNN ZIP，" -ForegroundColor Yellow
        Write-Host "  将 bin/ 目录下的 *.dll 复制到: $OutputDir" -ForegroundColor Yellow
        $Failed += $name
        continue
    }

    Write-Host "  [$name] 解压中..." -ForegroundColor Gray
    Expand-ZipWith7z -ZipPath $zip -DestDir $extractDir

    $found = $false
    foreach ($dll in $pkg.Dlls) {
        $src = Get-ChildItem -Path $extractDir -Recurse -Filter $dll -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($src) {
            Copy-Item $src.FullName -Destination $OutputDir -Force
            Write-Host "    ✓ $dll" -ForegroundColor DarkGreen
            $AllExtractedDlls += $dll
            $found = $true
        }
        else {
            Write-Host "    ⚠ $dll (未在包中找到)" -ForegroundColor DarkYellow
        }
    }
    if (-not $found) {
        $Failed += $name
    }
}

# ── 清理临时文件 ──
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

# ── 报告 ──
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
if ($Failed.Count -eq 0) {
    Write-Host "  CUDA Runtime 下载完成 ✓" -ForegroundColor Green
}
else {
    Write-Host "  部分包下载失败: $($Failed -join ', ')" -ForegroundColor Yellow
    Write-Host "  请检查网络连接或手动下载后放入: $OutputDir" -ForegroundColor Yellow
}
Write-Host "  输出目录: $OutputDir" -ForegroundColor White
Write-Host "  DLL 数量: $($AllExtractedDlls.Count)" -ForegroundColor White

# 计算大小
$totalSize = (Get-ChildItem -Path $OutputDir -Filter "*.dll" | Measure-Object -Property Length -Sum).Sum
$totalSizeMB = [math]::Round($totalSize / 1MB, 1)
Write-Host "  总大小:   $totalSizeMB MB" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
