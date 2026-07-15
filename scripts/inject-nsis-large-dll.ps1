<#
.SYNOPSIS
    Inject cudnn_engines_precompiled64_9.dll into a Tauri-generated NSIS
    installer script, then re-run makensis to produce the final installer.

.DESCRIPTION
    This is a POST-BUILD step that runs AFTER `cargo tauri build` has
    successfully produced the NSIS installer (without the precompiled
    engine DLL - it is intentionally excluded from Tauri resources).

    The DLL (~500–800 MB) is TOO LARGE for NSIS's 32-bit solid LZMA
    compressor (makensis crashes with "Internal compiler error #12345:
    error mmapping datablock").  This script works around the limitation
    by storing the DLL UNCOMPRESSED (SetCompress off), which avoids the
    memory-mapping failure entirely.

    The precompiled engine DLL is an optimisation cache - cuDNN 9.x
    falls back to runtime compilation when it is absent.  Including it
    uncompressed adds ~500–800 MB to the installer but gives users the
    fastest cold-start experience.

.DEPENDENCY
    makensis - Tauri downloads and installs its own NSIS to
    %LOCALAPPDATA%\tauri\NSIS\ during the first `cargo tauri build`.
    This script locates it automatically; no manual NSIS installation
    is needed.

.PARAMETER TargetTriple
    The Rust target triple (e.g., x86_64-pc-windows-msvc).

.PARAMETER ProductName
    The product name from tauri.conf.json (auto-detected if omitted).

.PARAMETER Version
    The version string from tauri.conf.json (auto-detected if omitted).

.EXAMPLE
    .\scripts\inject-nsis-large-dll.ps1
    .\scripts\inject-nsis-large-dll.ps1 -TargetTriple aarch64-pc-windows-msvc
#>

[CmdletBinding()]
param(
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$ProductName = "",
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SrcTauri    = Join-Path $ProjectRoot "backend\src-tauri"
$ReleaseDir  = Join-Path $SrcTauri "target\$TargetTriple\release"

# =========================================================================
# 0. Read product name and version from tauri.conf.json
# =========================================================================
# IMPORTANT: use UTF-8 explicitly - PowerShell's Get-Content defaults to
# the system ANSI code page (e.g. GBK on Chinese Windows), which corrupts
# non-ASCII characters in UTF-8 files.

if (-not $ProductName -or -not $Version) {
    $tauriConfPath = Join-Path $SrcTauri "tauri.conf.json"
    $tauriConfRaw  = [System.IO.File]::ReadAllText($tauriConfPath, [System.Text.Encoding]::UTF8)
    $TauriConf     = $tauriConfRaw | ConvertFrom-Json
    if (-not $ProductName) { $ProductName = $TauriConf.productName }
    if (-not $Version)     { $Version     = $TauriConf.version }
}

# =========================================================================
# 1. Locate the generated NSIS script and the large DLL
# =========================================================================

$NsisDir    = Join-Path $ReleaseDir "nsis\x64"
$NsisScript = Join-Path $NsisDir "installer.nsi"
$LargeDll   = Join-Path $SrcTauri "third_party\ort-bundle\cudnn_engines_precompiled64_9.dll"

if (-not (Test-Path $NsisScript)) {
    Write-Error "NSIS script not found: $NsisScript"
    Write-Error "Make sure 'cargo tauri build' has been run first."
    exit 1
}

if (-not (Test-Path $LargeDll)) {
    Write-Error "Precompiled engine DLL not found: $LargeDll"
    Write-Error "Run download-cuda-runtime.ps1 first."
    exit 1
}

$dllSizeMB = [math]::Round((Get-Item $LargeDll).Length / 1MB, 1)
Write-Host "=== inject-nsis-large-dll ===" -ForegroundColor Cyan
Write-Host "  Target:      $TargetTriple"
Write-Host "  Product:     $ProductName v$Version"
Write-Host "  NSIS script: $NsisScript"
Write-Host "  Large DLL:   cudnn_engines_precompiled64_9.dll ($dllSizeMB MB)"
Write-Host ""

# =========================================================================
# 2. Patch the NSIS script to add the DLL as uncompressed
# =========================================================================

Write-Host "[1/3] Patching NSIS script..." -ForegroundColor Yellow

# CRITICAL: Tauri writes the NSIS script as UTF-8.  PowerShell's default
# Get-Content uses the system ANSI code page (GBK on Chinese Windows,
# Windows-1252 on English Windows), which DESTROYS non-ASCII bytes in UTF-8
# files.  Use [System.IO.File] with explicit UTF-8 to preserve all bytes.
$script = [System.IO.File]::ReadAllText($NsisScript, [System.Text.UTF8Encoding]::new($false))

# ---- 2a. Switch from solid to non-solid compression ----------------------
# NSIS's "solid" (/SOLID) mode compresses ALL files as one block.  This
# requires memory-mapping the entire compressed payload, which overflows
# NSIS's 32-bit address space when a ~500+ MB DLL is included (even if
# stored uncompressed - solid mode forces the mmap regardless).
#
# Switching to non-solid compression makes NSIS compress each file
# independently, and critically, respects per-file SetCompress off so
# that the giant DLL can be stored raw without any mmap at all.
$solidMarker  = 'SetCompressor /SOLID "lzma"'
$nonSolidLine = 'SetCompressor "lzma"'
if ($script.IndexOf($solidMarker) -ge 0) {
    $script = $script.Replace($solidMarker, $nonSolidLine)
    Write-Host "  Switched: /SOLID lzma -> lzma (non-solid, per-file compression)"
} else {
    Write-Host "  NOTE: /SOLID marker not found - script may already be non-solid"
}

# ---- 2b. Inject the large DLL (uncompressed) into the Install section --
# The marker "; Copy external binaries" is inside the Install Section.
$installMarker = "; Copy external binaries"

if ($script.IndexOf($installMarker) -lt 0) {
    Write-Error "Cannot find install marker '$installMarker' in NSIS script."
    exit 1
}

# Build the injection block with CRLF line endings (NSIS expects Windows line endings).
# The large DLL path is embedded as an absolute path - NSIS handles this fine.
$installInjection = @"

  ; cudnn_engines_precompiled64_9.dll -- stored UNCOMPRESSED because
  ; NSIS (32-bit) cannot memory-map this file for solid LZMA compression.
  ; Injected by scripts/inject-nsis-large-dll.ps1.
  SetCompress off
  File /a "/oname=cudnn_engines_precompiled64_9.dll" "$LargeDll"
  SetCompress auto
"@

$script = $script.Replace($installMarker, "$installMarker$installInjection")

# ---- 2c. Add Delete in the Uninstall section ---------------------------
$uninstallMarker = "; Delete external binaries"

if ($script.IndexOf($uninstallMarker) -lt 0) {
    Write-Error "Cannot find uninstall marker '$uninstallMarker' in NSIS script."
    exit 1
}

$uninstallInjection = @"

  Delete "`$INSTDIR\cudnn_engines_precompiled64_9.dll"
"@

$script = $script.Replace($uninstallMarker, "$uninstallInjection$uninstallMarker")

# ---- 2d. Write back -----------------------------------------------------
# Write as UTF-8 WITHOUT BOM.  NSIS handles this: it checks for
# UTF-16LE BOM first, then falls back to ANSI/UTF-8 detection.
# Tauri's original script is UTF-8 without BOM, so we match that.
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($NsisScript, $script, $utf8NoBom)
Write-Host "  Patched: added File + Delete for cudnn_engines_precompiled64_9.dll"

# =========================================================================
# 3. Locate makensis (Tauri bundles its own - no system install needed)
# =========================================================================

Write-Host "[2/3] Locating makensis..." -ForegroundColor Yellow

function Find-Makensis {
    # Tauri downloads NSIS to %LOCALAPPDATA%\tauri\NSIS\ during the first
    # `cargo tauri build`.  Search there first - it's always available
    # after a successful build with NSIS bundling.
    $tauriNsisBase = Join-Path $env:LOCALAPPDATA "tauri\NSIS"
    if (Test-Path $tauriNsisBase) {
        # The exact layout varies by Tauri version:
        #   v1.x / early v2:  tauri\NSIS\makensis.exe
        #   later v2:         tauri\NSIS\Bin\makensis.exe
        $tauriCandidates = @(
            (Join-Path $tauriNsisBase "Bin\makensis.exe"),
            (Join-Path $tauriNsisBase "makensis.exe")
        )
        foreach ($c in $tauriCandidates) {
            if (Test-Path $c) {
                Write-Host "  Found Tauri-bundled NSIS: $c"
                return $c
            }
        }
        # Recursive search as last resort (one level deep)
        $found = Get-ChildItem -Path $tauriNsisBase -Recurse -Depth 2 -Filter "makensis.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Write-Host "  Found Tauri-bundled NSIS: $($found.FullName)"
            return $found.FullName
        }
    }

    # System-installed NSIS (user or CI pre-installed)
    $systemCandidates = @(
        "C:\Program Files (x86)\NSIS\Bin\makensis.exe",
        "C:\Program Files (x86)\NSIS\makensis.exe",
        "C:\Program Files\NSIS\Bin\makensis.exe",
        "C:\Program Files\NSIS\makensis.exe"
    )
    foreach ($c in $systemCandidates) {
        if (Test-Path $c) {
            Write-Host "  Found system NSIS: $c"
            return $c
        }
    }

    # PATH lookup
    $fromPath = Get-Command "makensis.exe" -ErrorAction SilentlyContinue
    if ($fromPath) {
        Write-Host "  Found via PATH: $($fromPath.Source)"
        return $fromPath.Source
    }

    return $null
}

$makensis = Find-Makensis
if (-not $makensis) {
    Write-Error @"
makensis not found!

Tauri downloads its own NSIS to %LOCALAPPDATA%\tauri\NSIS\ during the first
`cargo tauri build` that produces an NSIS installer.  If you are seeing this
error, either:

  1. `cargo tauri build` has NOT been run on this machine yet.
     → Run it once (even a CPU build) to trigger the NSIS download, then
       re-run this script.

  2. The NSIS download was cleaned up.
     → Delete %LOCALAPPDATA%\tauri\NSIS\ and re-run `cargo tauri build`.
       Tauri will re-download it.

  3. You are running in an environment without LOCALAPPDATA.
     → Set LOCALAPPDATA to a writable directory before running.
"@
    exit 1
}

# =========================================================================
# 4. Run makensis
# =========================================================================

Write-Host "[3/4] Running makensis..." -ForegroundColor Yellow

Push-Location $NsisDir
try {
    & $makensis "installer.nsi"
    if ($LASTEXITCODE -ne 0) {
        throw "makensis failed with exit code $LASTEXITCODE"
    }
    Write-Host "  makensis completed successfully" -ForegroundColor Green
} finally {
    Pop-Location
}

# =========================================================================
# 5. Move output to the bundle directory
# =========================================================================

Write-Host "[4/4] Copying installer to bundle directory..." -ForegroundColor Yellow

$generatedExe = Join-Path $NsisDir "nsis-output.exe"
if (-not (Test-Path $generatedExe)) {
    Write-Error "makensis output not found: $generatedExe"
    exit 1
}

# Determine arch suffix
if ($TargetTriple -like "*aarch64*") {
    $arch = "arm64"
} else {
    $arch = "x64"
}

$expectedName = "${ProductName}_${Version}_${arch}-setup.exe"
$bundleDir   = Join-Path $ReleaseDir "bundle\nsis"
$finalPath   = Join-Path $bundleDir $expectedName

New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

# Remove any previous installer (Tauri may have left one)
if (Test-Path $finalPath) {
    Remove-Item $finalPath -Force
}

Move-Item $generatedExe -Destination $finalPath -Force

$finalSizeMB = [math]::Round((Get-Item $finalPath).Length / 1MB, 1)
Write-Host "  Done: $finalPath ($finalSizeMB MB)" -ForegroundColor Green
Write-Host ""
Write-Host "=== inject-nsis-large-dll: SUCCESS ===" -ForegroundColor Green
