<#
.SYNOPSIS
    Pack HiFiShifter into a portable ZIP archive

.DESCRIPTION
    After a successful `cargo tauri build`, this script collects the exe,
    resource files, models, and GPU-dependent DLLs (OpenCL/DirectML) from
    the build output directory and packages them into a portable .zip file.
    All DLLs located in the release directory are automatically collected.

.PARAMETER SkipBuild
    Skip the build step and package from existing artifacts (useful when a
    build has already been performed).

.PARAMETER OutputDir
    Output directory, defaults to the dist folder under the project root.

.EXAMPLE
    .\scripts\pack-portable.ps1
    # Full build + packaging

.EXAMPLE
    .\scripts\pack-portable.ps1 -SkipBuild
    # Skip build, package existing artifacts

.EXAMPLE
    .\scripts\pack-portable.ps1 -OutputDir "C:\output"
    # Specify output directory
#>

param(
    [switch]$SkipBuild,
    [switch]$NoZip,
    [string]$OutputDir,
    [string]$Version
)

$ErrorActionPreference = "Stop"

# ===== Path definitions =====
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$TauriDir = Join-Path $ProjectRoot "backend\src-tauri"
$TauriTargetRoot = Join-Path $TauriDir "target"
$SetVersionScript = Join-Path $ProjectRoot "scripts\set-version.ps1"

# If -Version is provided, update the version number first; subsequent build
# and packaging will use that version.
if ($Version) {
    if (-not (Test-Path $SetVersionScript)) {
        throw "Version script not found: $SetVersionScript"
    }
    Write-Host "[Preprocessing] Applying version: $Version" -ForegroundColor Yellow
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SetVersionScript -Version $Version
    if ($LASTEXITCODE -ne 0) {
        throw "Version update failed, exit code: $LASTEXITCODE"
    }
    Write-Host "[Preprocessing] Version update completed ✓" -ForegroundColor Green
}

# Detect target triple: prefer x86_64 but fall back to aarch64 if present.
$DetectedTriple = $null
$PossibleTriples = @("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc")
foreach ($t in $PossibleTriples) {
    $p = Join-Path $TauriTargetRoot (Join-Path $t "release")
    if (Test-Path $p) {
        $DetectedTriple = $t
        $TargetRelease = $p
        break
    }
}

# If no triple-specific release directory exists yet, default to x86_64 path
# (build may create it later).
if (-not $DetectedTriple) {
    $DetectedTriple = "x86_64-pc-windows-msvc"
    $TargetRelease = Join-Path $TauriTargetRoot "x86_64-pc-windows-msvc\release"
}

# Read version and product name from tauri.conf.json
$TauriConf = Get-Content (Join-Path $TauriDir "tauri.conf.json") -Raw | ConvertFrom-Json
$ProductName = $TauriConf.productName
$Version = $TauriConf.version

# Output directory
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "dist"
}

$PortableDirName = "$ProductName"
$TempDir = Join-Path $OutputDir $PortableDirName

# Determine arch short name for filenames
if ($DetectedTriple -like "*aarch64*") {
    $ArchShort = "arm64"
}
else {
    $ArchShort = "x64"
}

$ZipName = "$ProductName-v$Version-portable-win-$ArchShort.zip"
$ZipPath = Join-Path $OutputDir $ZipName

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HiFiShifter Portable Packaging Tool" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Product Name: $ProductName"
Write-Host "  Version:      $Version"
Write-Host "  Output Path:  $ZipPath"
Write-Host ""

# ===== Interactive choice (when -SkipBuild is not specified) =====
if (-not $SkipBuild) {
    Write-Host "Please select an action:" -ForegroundColor White
    Write-Host "  [1] Full build + packaging" -ForegroundColor Yellow
    Write-Host "  [2] Skip build, package directly (use existing artifacts)" -ForegroundColor Yellow
    Write-Host ""
    do {
        $choice = Read-Host "Enter option (1/2)"
        if ($choice -eq "2") {
            $SkipBuild = $true
            Write-Host ""
            break
        }
        elseif ($choice -eq "1") {
            Write-Host ""
            break
        }
        else {
            Write-Host "Invalid input, please enter 1 or 2" -ForegroundColor Red
        }
    } while ($true)
}

# ===== Step 1: Build (optional) =====
if (-not $SkipBuild) {
    Write-Host "[1/5] Building Release version..." -ForegroundColor Yellow
    Push-Location $TauriDir
    try {
        cargo tauri build
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed, exit code: $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-Host "[1/5] Build completed ✓" -ForegroundColor Green
}
else {
    Write-Host "[1/5] Skipping build step (-SkipBuild)" -ForegroundColor DarkGray
}

# ===== Step 2: Check artifacts =====
Write-Host "[2/5] Checking build artifacts..." -ForegroundColor Yellow

$ExePath = Join-Path $TargetRelease "$ProductName.exe"
if (-not (Test-Path $ExePath)) {
    throw "Cannot find exe: $ExePath`nPlease run 'cargo tauri build' first or remove the -SkipBuild parameter"
}

# Define resource files to collect (source path -> target relative path)
$Resources = @(
    @{ Src = Join-Path $TauriDir "resources\models\nsf_hifigan\pc_nsf_hifigan.onnx"; Dst = "models\nsf_hifigan\pc_nsf_hifigan.onnx" },
    @{ Src = Join-Path $TauriDir "resources\models\nsf_hifigan\config.json";          Dst = "models\nsf_hifigan\config.json" },
    @{ Src = Join-Path $TauriDir "resources\models\hnsep\hnsep.onnx";                 Dst = "models\hnsep\hnsep.onnx" },
    @{ Src = Join-Path $TauriDir "resources\models\hnsep\config.yaml";                Dst = "models\hnsep\config.yaml" },
    @{ Src = Join-Path $TauriDir "resources\models\fcpe\fcpe.onnx";                   Dst = "models\fcpe\fcpe.onnx" }
)

if ($ArchShort -eq "x64") {
    $Resources += @{ Src = Join-Path $TauriDir "third_party\vslib\vslib_x64.dll"; Dst = "vslib_x64.dll" }
}

# Check that all resource files exist
$Missing = @()
foreach ($res in $Resources) {
    if (-not (Test-Path $res.Src)) {
        $Missing += $res.Src
    }
}
if ($Missing.Count -gt 0) {
    Write-Host "The following resource files are missing:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    throw "Resource files are incomplete, cannot package."
}

Write-Host "[2/5] Artifacts check passed ✓" -ForegroundColor Green

# ===== Step 3: Assemble directory =====
Write-Host "[3/5] Assembling portable package directory..." -ForegroundColor Yellow

# Clean up old temporary directory and zip
if (Test-Path $TempDir) {
    Remove-Item $TempDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

# Create output directory
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Copy exe
Copy-Item $ExePath -Destination $TempDir
Write-Host "  ✓ $ProductName.exe" -ForegroundColor DarkGreen

# Copy resource files
foreach ($res in $Resources) {
    $DstFull = Join-Path $TempDir $res.Dst
    $DstDir = Split-Path $DstFull -Parent
    if (-not (Test-Path $DstDir)) {
        New-Item -ItemType Directory -Path $DstDir -Force | Out-Null
    }
    Copy-Item $res.Src -Destination $DstFull
    Write-Host "  ✓ $($res.Dst)" -ForegroundColor DarkGreen
}

# Copy LICENSE
$LicensePath = Join-Path $ProjectRoot "LICENSE"
if (Test-Path $LicensePath) {
    Copy-Item $LicensePath -Destination $TempDir
    Write-Host "  ✓ LICENSE" -ForegroundColor DarkGreen
}

# Copy any DLLs from the release directory (SoundTouchDLL, ORT, etc.).
# Exclude vslib_x64.dll which is handled separately above from the third_party source.
Get-ChildItem -Path $TargetRelease -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Name -ne "vslib_x64.dll") {
        Copy-Item $_.FullName -Destination $TempDir
        Write-Host "  ✓ $($_.Name)" -ForegroundColor DarkGreen
    }
}

# Check WebView2Loader.dll (may be needed by Tauri)
$Wv2Dll = Join-Path $TargetRelease "WebView2Loader.dll"
if (Test-Path $Wv2Dll) {
    Copy-Item $Wv2Dll -Destination $TempDir
    Write-Host "  ✓ WebView2Loader.dll" -ForegroundColor DarkGreen
}

Write-Host "[3/5] Directory assembly completed ✓" -ForegroundColor Green

# ===== Step 4: Compress =====
if (-not $NoZip) {
    Write-Host "[4/5] Compressing to ZIP..." -ForegroundColor Yellow

    Compress-Archive -Path $TempDir -DestinationPath $ZipPath -CompressionLevel Optimal

    # Clean up temporary directory
    Remove-Item $TempDir -Recurse -Force

    $ZipSize = (Get-Item $ZipPath).Length
    $ZipSizeMB = [math]::Round($ZipSize / 1MB, 2)

    Write-Host "[4/5] Compression completed ✓" -ForegroundColor Green
}
else {
    Write-Host "[4/5] Skipping ZIP compression (-NoZip)" -ForegroundColor DarkGray
    Write-Host "       Portable dir staged at: $TempDir" -ForegroundColor Green
}

# ===== Step 5: Copy NSIS installer =====
if (-not $NoZip) {
    Write-Host "[5/5] Copying NSIS installer to dist..." -ForegroundColor Yellow

    # NSIS installer path: look under the detected triple's bundle dir
    $NsisDir = Join-Path $ProjectRoot (Join-Path "backend\src-tauri\target" (Join-Path $DetectedTriple "release\bundle\nsis"))
    if ($ArchShort -eq "x64") {
        $NsisPattern = "${ProductName}_${Version}_x64-setup.exe"
    }
    else {
        $NsisPattern = "${ProductName}_${Version}_arm64-setup.exe"
    }
    $NsisExePath = Join-Path $NsisDir $NsisPattern

    if (Test-Path $NsisExePath) {
        Copy-Item $NsisExePath -Destination $OutputDir
        $NsisSize = (Get-Item (Join-Path $OutputDir $NsisPattern)).Length
        $NsisSizeMB = [math]::Round($NsisSize / 1MB, 2)
        Write-Host "[5/5] NSIS installer copied ✓ ($NsisSizeMB MB)" -ForegroundColor Green
    }
    else {
        Write-Host "[5/5] NSIS installer not found, skipping (path: $NsisExePath)" -ForegroundColor DarkGray
    }
}
else {
    Write-Host "[5/5] Skipping NSIS copy (-NoZip)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
if (-not $NoZip) {
    Write-Host "  Packaging successful!" -ForegroundColor Green
    Write-Host "  Portable: $ZipPath" -ForegroundColor Green
    Write-Host "  Size:     $ZipSizeMB MB" -ForegroundColor Green
    if (Test-Path (Join-Path $OutputDir $NsisPattern)) {
        Write-Host "  Installer: $(Join-Path $OutputDir $NsisPattern)" -ForegroundColor Green
        Write-Host "  Size:      $NsisSizeMB MB" -ForegroundColor Green
    }
}
else {
    Write-Host "  Portable directory staged successfully!" -ForegroundColor Green
    Write-Host "  Location: $TempDir" -ForegroundColor Green
}
Write-Host "============================================" -ForegroundColor Cyan
