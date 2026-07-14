<#
.SYNOPSIS
    Download and extract ONNX Runtime prebuilt binaries for Windows.

.DESCRIPTION
    Downloads the ONNX Runtime redistributable package from GitHub Releases and
    extracts the library files (DLLs, .lib) to the specified destination directory.

    Used by both local dev setup (setup-windows.ps1) and CI workflows.

.PARAMETER Gpu
    Download the GPU (CUDA) variant instead of the CPU variant.

.PARAMETER DestDir
    Directory to place the extracted library files.
    Defaults to backend/src-tauri/third_party/ort-bundle/ in the project root.

.PARAMETER Version
    ONNX Runtime version to download. Default: "1.24.1"

.EXAMPLE
    .\scripts\download-ort.ps1 -Gpu
    .\scripts\download-ort.ps1 -Gpu -DestDir "C:\ort\lib"
    .\scripts\download-ort.ps1 -Version "1.24.0"
#>

[CmdletBinding()]
param(
    [switch]$Gpu,
    [string]$DestDir = "",
    [string]$Version = "1.24.1"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ---- main ----

# Resolve destination directory - default to project-local ort-bundle/
if (-not $DestDir) {
    $DestDir = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "backend\src-tauri\third_party\ort-bundle"
}
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

$variant = if ($Gpu) { "gpu" } else { "cpu" }
$archiveName = "onnxruntime-win-x64-$variant-$Version.zip"
$Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$archiveName"

Write-Host "=== ONNX Runtime Download ===" -ForegroundColor Cyan
Write-Host "  Version:   $Version ($variant)" -ForegroundColor White
Write-Host "  URL:       $Url" -ForegroundColor DarkGray
Write-Host "  Dest:      $DestDir" -ForegroundColor White

# Skip if already present
$primaryDll = Join-Path $DestDir "onnxruntime.dll"
if (Test-Path $primaryDll) {
    Write-Host "  Status:    Already present - skipping" -ForegroundColor Green
    Write-Host "  To re-download, delete $primaryDll and re-run." -ForegroundColor DarkGray
    return
}

$TempDir = Join-Path $env:TEMP "hifishifter-ort-dl-$PID"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$ZipFile = Join-Path $TempDir $archiveName

try {
    Write-Host "  Downloading... ($([math]::Round((Invoke-WebRequest -Uri $Url -Method Head).ContentLength / 1MB, 1)) MB)" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $ZipFile
    $sizeMB = [math]::Round((Get-Item $ZipFile).Length / 1MB, 1)
    Write-Host "  Downloaded:  $sizeMB MB" -ForegroundColor Green
} catch {
    Write-Error "Failed to download ONNX Runtime: $_"
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "  Extracting..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $ZipFile -DestinationPath $TempDir -Force
} catch {
    Write-Error "Failed to extract archive: $_"
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

# Locate the extracted lib directory
$ExtractedDir = Get-ChildItem -Path $TempDir -Directory | Select-Object -First 1
if (-not $ExtractedDir) {
    Write-Error "Could not locate extracted directory."
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

$LibSourceDir = Join-Path $ExtractedDir.FullName "lib"
if (-not (Test-Path $LibSourceDir)) {
    Write-Error "No 'lib' directory found in extracted ONNX Runtime package."
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "  Copying libraries to $DestDir..." -ForegroundColor Yellow
$copied = 0
Get-ChildItem -Path $LibSourceDir -File | ForEach-Object {
    $dst = Join-Path $DestDir $_.Name
    Copy-Item $_.FullName -Destination $dst -Force
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "    + $($_.Name) ($sizeMB MB)" -ForegroundColor DarkGreen
    $copied++
}

Write-Host "  Copied $copied file(s)" -ForegroundColor Green

# Cleanup
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "  Done!" -ForegroundColor Green
