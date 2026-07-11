# setup-gpu-deps.ps1
# Download and install precompiled ONNX Runtime GPU (CUDA) binaries for Windows.
# Extracts them to the default User Cache directory (~/.cache/ort/lib).

$OrtVersion = "1.24.1"
$ZipUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-gpu-$OrtVersion.zip"

$CacheDir = Join-Path $env:USERPROFILE ".cache\ort\lib"
$TempDir = Join-Path $PSScriptRoot "temp_ort_setup"
$ZipPath = Join-Path $TempDir "onnxruntime-gpu.zip"

Write-Host "=== Setting up ONNX Runtime GPU (CUDA) v$OrtVersion ===" -ForegroundColor Cyan
Write-Host "Target directory: $CacheDir" -ForegroundColor Cyan

# 1. Create Directories
if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
}
if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir | Out-Null
}
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# 2. Download ONNX Runtime GPU package
Write-Host "Downloading $ZipUrl ..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath -UserAgent "Mozilla/5.0"
} catch {
    Write-Error "Failed to download ONNX Runtime GPU zip file. Check your internet connection."
    exit 1
}

# 3. Extract Zip
Write-Host "Extracting package..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $ZipPath -DestinationPath $TempDir -Force
} catch {
    Write-Error "Extraction failed. The zip file might be corrupted."
    exit 1
}

# 4. Locate extracted files and copy them to ~/.cache/ort/lib
$ExtractedFolder = Get-ChildItem -Path $TempDir -Directory | Select-Object -First 1
if (-not $ExtractedFolder) {
    Write-Error "Could not locate extracted directory."
    exit 1
}

$LibSourceDir = Join-Path $ExtractedFolder.FullName "lib"
Write-Host "Copying libraries from $LibSourceDir to $CacheDir..." -ForegroundColor Yellow

$FilesToCopy = Get-ChildItem -Path $LibSourceDir -File
foreach ($File in $FilesToCopy) {
    $DestPath = Join-Path $CacheDir $File.Name
    Copy-Item -Path $File.FullName -Destination $DestPath -Force
    Write-Host "  Copied: $($File.Name)" -ForegroundColor Green
}

# 5. Clean up temporary files
Write-Host "Cleaning up temp files..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $TempDir | Out-Null

# 6. Auto-download CUDA runtime DLLs (cuBLAS + cuDNN)
# These are REQUIRED at runtime — without them CUDA EP silently falls back to CPU.
Write-Host ""
Write-Host "=== Downloading CUDA runtime DLLs (cuBLAS + cuDNN) ===" -ForegroundColor Cyan
$downloadScript = Join-Path $PSScriptRoot "scripts\download-cuda-runtime.ps1"
if (Test-Path $downloadScript) {
    & $downloadScript -DestDir $CacheDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "WARNING: CUDA runtime download had errors. GPU acceleration may not work." -ForegroundColor Yellow
        Write-Host "  Run manually: .\scripts\download-cuda-runtime.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING: download-cuda-runtime.ps1 not found. GPU acceleration will NOT work." -ForegroundColor Yellow
    Write-Host "  Download cuBLAS 12 + cuDNN 9 manually from NVIDIA and place in:" -ForegroundColor Yellow
    Write-Host "  $CacheDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Completed Successfully! ===" -ForegroundColor Green
Write-Host "ONNX Runtime GPU + CUDA runtime libraries are now placed in $CacheDir."
Write-Host "You can now run: .\dev-gpu.ps1" -ForegroundColor Green
