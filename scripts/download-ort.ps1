<#
.SYNOPSIS
    Acquire ONNX Runtime prebuilt binaries for Windows.

.DESCRIPTION
    Populates the project-local ort-bundle/ directory with ONNX Runtime
    libraries (DLLs, .lib) and C headers (include/) so that both ort-sys
    (linking) and build.rs (runtime staging) use the same ORT build.

    Three acquisition modes, tried in order:

    1. -LocalOrtDir <path>   Copy from a pre-extracted ORT installation.
                              The directory must contain lib/ and include/
                              subdirectories (matching the ORT package layout).

    2. -LocalPackage <path>  Extract from a locally-downloaded ZIP archive.

    3. Default (online)      Download from GitHub Releases.  Set the $env:ORT_MIRROR
                              variable to a mirror base URL to override the default.

.PARAMETER Gpu
    Download the GPU (CUDA) variant instead of the CPU variant.
    Ignored when -LocalPackage or -LocalOrtDir is used.

.PARAMETER DestDir
    Directory to place the extracted library files.
    Defaults to backend/src-tauri/third_party/ort-bundle/ in the project root.

.PARAMETER Version
    ONNX Runtime version to download.  Default: "1.24.1".
    Ignored when -LocalPackage or -LocalOrtDir is used.

.PARAMETER LocalPackage
    Path to a locally-downloaded ONNX Runtime ZIP archive.  The archive is
    extracted and its lib/ and include/ contents are copied to DestDir.
    No network access required.

.PARAMETER LocalOrtDir
    Path to a pre-extracted ONNX Runtime installation directory.  The
    directory must contain a lib/ subdirectory (with DLLs and .lib files)
    and an include/ subdirectory (with C headers).  All needed files are
    copied to DestDir without any network access.
    This is equivalent to the old ORT_LIB_LOCATION workflow.

.EXAMPLE
    # Default: download GPU build from GitHub
    .\scripts\download-ort.ps1 -Gpu

.EXAMPLE
    # Use a mirror for faster downloads in China
    $env:ORT_MIRROR = "https://ghproxy.com/https://github.com"
    .\scripts\download-ort.ps1 -Gpu

.EXAMPLE
    # Use a pre-downloaded ZIP (no network)
    .\scripts\download-ort.ps1 -Gpu -LocalPackage "D:\Downloads\onnxruntime-win-x64-gpu-1.24.1.zip"

.EXAMPLE
    # Copy from a pre-extracted ORT installation (no network)
    .\scripts\download-ort.ps1 -LocalOrtDir "D:\ort\onnxruntime-win-x64-gpu-1.24.1"
#>

[CmdletBinding()]
param(
    [switch]$Gpu,
    [string]$DestDir = "",
    [string]$Version = "1.24.1",
    [string]$LocalPackage,
    [string]$LocalOrtDir
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ---- helpers ----

function Copy-OrtFiles {
    param(
        [string]$SourceRoot,   # root of the extracted ORT package (contains lib/ and include/)
        [string]$Destination   # ort-bundle/ directory
    )

    # -- lib/ files --------------------------------------------------
    $libDir = Join-Path $SourceRoot "lib"
    if (-not (Test-Path $libDir)) {
        Write-Error "No 'lib' directory found in $SourceRoot"
        exit 1
    }

    Write-Host "  Copying libraries from $libDir ..." -ForegroundColor Yellow
    $copied = 0
    Get-ChildItem -Path $libDir -File | ForEach-Object {
        $dst = Join-Path $Destination $_.Name
        Copy-Item $_.FullName -Destination $dst -Force
        $sizeMB = [math]::Round($_.Length / 1MB, 1)
        Write-Host "    + $($_.Name) ($sizeMB MB)" -ForegroundColor DarkGreen
        $copied++
    }
    Write-Host "  Copied $copied library file(s)" -ForegroundColor Green

    # -- include/ files -----------------------------------------------
    $includeDir = Join-Path $SourceRoot "include"
    if (Test-Path $includeDir) {
        $includeDest = Join-Path $Destination "include"
        if (Test-Path $includeDest) {
            Remove-Item $includeDest -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Path $includeDest -Force | Out-Null
        $headerCount = 0
        Get-ChildItem -Path $includeDir -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($includeDir.Length + 1)
            $headerDst = Join-Path $includeDest $relative
            $parentDir = Split-Path $headerDst -Parent
            if (-not (Test-Path $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }
            Copy-Item $_.FullName -Destination $headerDst -Force
            $headerCount++
        }
        Write-Host "  Copied $headerCount header file(s) to include/" -ForegroundColor DarkGreen
    } else {
        Write-Host "  WARNING: No include/ directory found - ort-sys may reject this ORT installation" -ForegroundColor Yellow
    }
}

# ---- main ----

# Resolve destination directory - default to project-local ort-bundle/
if (-not $DestDir) {
    $DestDir = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "backend\src-tauri\third_party\ort-bundle"
}
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

# Check if already fully present
$primaryDll = Join-Path $DestDir "onnxruntime.dll"
$headerFile = Join-Path $DestDir "include\onnxruntime_c_api.h"
if ((Test-Path $primaryDll) -and (Test-Path $headerFile)) {
    Write-Host "=== ONNX Runtime ===" -ForegroundColor Cyan
    Write-Host "  Dest:      $DestDir" -ForegroundColor White
    Write-Host "  Status:    Already present - skipping" -ForegroundColor Green
    Write-Host "  To re-download, delete $primaryDll and re-run." -ForegroundColor DarkGray
    exit 0
}

# =====================================================================
# Mode 1: LocalOrtDir - copy from a pre-extracted ORT installation
# =====================================================================
if ($LocalOrtDir) {
    Write-Host "=== ONNX Runtime (local directory) ===" -ForegroundColor Cyan
    Write-Host "  Source:    $LocalOrtDir" -ForegroundColor White
    Write-Host "  Dest:      $DestDir" -ForegroundColor White

    if (-not (Test-Path $LocalOrtDir)) {
        Write-Error "LocalOrtDir not found: $LocalOrtDir"
        exit 1
    }

    # Accept the directory as-is if it contains lib/; otherwise treat it as a
    # package root that has a single subdirectory containing lib/ (common after
    # extracting an ORT release archive).
    $sourceRoot = $LocalOrtDir
    if (-not (Test-Path (Join-Path $sourceRoot "lib"))) {
        $child = Get-ChildItem -Path $LocalOrtDir -Directory | Select-Object -First 1
        if ($child -and (Test-Path (Join-Path $child.FullName "lib"))) {
            $sourceRoot = $child.FullName
        } else {
            Write-Error "LocalOrtDir must contain a lib/ subdirectory.  Found neither in $LocalOrtDir nor in a single child directory."
            exit 1
        }
    }

    Copy-OrtFiles -SourceRoot $sourceRoot -Destination $DestDir
    Write-Host "  Done!" -ForegroundColor Green
    exit 0
}

# =====================================================================
# Mode 2: LocalPackage - extract from a local ZIP file
# =====================================================================
if ($LocalPackage) {
    Write-Host "=== ONNX Runtime (local archive) ===" -ForegroundColor Cyan
    Write-Host "  Archive:   $LocalPackage" -ForegroundColor White
    Write-Host "  Dest:      $DestDir" -ForegroundColor White

    if (-not (Test-Path $LocalPackage)) {
        Write-Error "LocalPackage not found: $LocalPackage"
        exit 1
    }

    $TempDir = Join-Path $env:TEMP "hifishifter-ort-local-$PID"
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

    try {
        Write-Host "  Extracting $([math]::Round((Get-Item $LocalPackage).Length / 1MB, 1)) MB..." -ForegroundColor Yellow
        Expand-Archive -Path $LocalPackage -DestinationPath $TempDir -Force

        $ExtractedDir = Get-ChildItem -Path $TempDir -Directory | Select-Object -First 1
        if (-not $ExtractedDir) {
            Write-Error "Could not locate extracted directory in archive."
            Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
            exit 1
        }

        Copy-OrtFiles -SourceRoot $ExtractedDir.FullName -Destination $DestDir
    } finally {
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Done!" -ForegroundColor Green
    exit 0
}

# =====================================================================
# Mode 3: Download - fetch from GitHub Releases (or mirror)
# =====================================================================
$variant = if ($Gpu) { "gpu" } else { "cpu" }
if ($Gpu) {
    $archiveName = "onnxruntime-win-x64-gpu-$Version.zip"
} else {
    $archiveName = "onnxruntime-win-x64-$Version.zip"
}

$BaseUrl = if ($env:ORT_MIRROR) {
    $env:ORT_MIRROR.TrimEnd('/')
} else {
    "https://github.com"
}
$Url = "$BaseUrl/microsoft/onnxruntime/releases/download/v$Version/$archiveName"

Write-Host "=== ONNX Runtime Download ===" -ForegroundColor Cyan
Write-Host "  Version:   $Version ($variant)" -ForegroundColor White
Write-Host "  URL:       $Url" -ForegroundColor DarkGray
Write-Host "  Dest:      $DestDir" -ForegroundColor White

if ((Test-Path $primaryDll) -and -not (Test-Path $headerFile)) {
    Write-Host "  Status:    DLLs present but include/ headers missing - re-downloading" -ForegroundColor Yellow
}

$TempDir = Join-Path $env:TEMP "hifishifter-ort-dl-$PID"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$ZipFile = Join-Path $TempDir $archiveName

try {
    Write-Host "  Downloading... ($([math]::Round((Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing).ContentLength / 1MB, 1)) MB)" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $ZipFile -UseBasicParsing
    $sizeMB = [math]::Round((Get-Item $ZipFile).Length / 1MB, 1)
    Write-Host "  Downloaded:  $sizeMB MB" -ForegroundColor Green
} catch {
    Write-Error "Failed to download ONNX Runtime: $_"
    if ($env:ORT_MIRROR) {
        Write-Host "  Hint: the mirror at '$BaseUrl' may be down.  Try without ORT_MIRROR, or use -LocalPackage." -ForegroundColor Yellow
    } else {
        Write-Host "  Hint: set ORT_MIRROR to a mirror base URL, or use -LocalPackage <zip> / -LocalOrtDir <dir>." -ForegroundColor Yellow
    }
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

$ExtractedDir = Get-ChildItem -Path $TempDir -Directory | Select-Object -First 1
if (-not $ExtractedDir) {
    Write-Error "Could not locate extracted directory."
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

Copy-OrtFiles -SourceRoot $ExtractedDir.FullName -Destination $DestDir

Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "  Done!" -ForegroundColor Green
exit 0
