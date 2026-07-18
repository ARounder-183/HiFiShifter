<#
.SYNOPSIS
    Inject ALL GPU DLLs (CUDA or DirectML) from ort-bundle/ into a
    Tauri-generated NSIS installer script, then re-run makensis.

.DESCRIPTION
    This is a POST-BUILD step that runs AFTER `cargo tauri build`.

    GPU DLLs are NOT listed in tauri.windows.conf.json - that file is
    committed to git and NEVER modified by any build step.  Instead,
    this script injects every DLL from ort-bundle/ directly into the
    generated NSIS script.

    Works for both CUDA and DirectML builds:
      - CUDA: injects onnxruntime + CUDA + cuDNN + cuBLAS + DirectML DLLs
      - DirectML: injects onnxruntime + DirectML DLLs (no CUDA runtime)

    Why not use Tauri resources for GPU DLLs?
      1. ort-bundle/ DLLs only exist after a GPU dependency download.
      2. They must not appear in the committed config (validation fails
         on fresh clones / CPU builds).
      3. NSIS injection is Ctrl+C safe - the NSIS script lives in
         target/, not in the git working tree.
      4. The precompiled engine DLL (~500 MB) needs SetCompress off
         to avoid NSIS 32-bit mmap crashes, which requires post-hoc
         NSIS script patching anyway.

.DEPENDENCY
    makensis - Tauri downloads its own NSIS to %LOCALAPPDATA%\tauri\NSIS\
    during the first `cargo tauri build`.  This script locates it
    automatically; no manual NSIS installation is needed.

.PARAMETER TargetTriple
    The Rust target triple (e.g., x86_64-pc-windows-msvc).

.EXAMPLE
    .\scripts\inject-gpu-dlls.ps1                           # CUDA installer
    .\scripts\inject-gpu-dlls.ps1 -TargetTriple aarch64-pc-windows-msvc

    # For DirectML: run build-gpu.ps1 -DirectML -Bundle (which calls this script)
#>

[CmdletBinding()]
param(
    [string]$TargetTriple = "x86_64-pc-windows-msvc"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SrcTauri    = Join-Path $ProjectRoot "backend\src-tauri"
$ReleaseDir  = Join-Path $SrcTauri "target\$TargetTriple\release"
$BundleDir   = Join-Path $SrcTauri "third_party\ort-bundle"

# =========================================================================
# 0. Read product info
# =========================================================================
$tauriConfPath = Join-Path $SrcTauri "tauri.conf.json"
$tauriConfRaw  = [System.IO.File]::ReadAllText($tauriConfPath, [System.Text.UTF8Encoding]::new($false))
$TauriConf     = $tauriConfRaw | ConvertFrom-Json
$ProductName   = $TauriConf.productName
$Version       = $TauriConf.version

# =========================================================================
# 1. Locate the generated NSIS script and the GPU DLLs
# =========================================================================

$NsisDir    = Join-Path $ReleaseDir "nsis\x64"
$NsisScript = Join-Path $NsisDir "installer.nsi"

if (-not (Test-Path $NsisScript)) {
    Write-Error "NSIS script not found: $NsisScript"
    Write-Error "Make sure 'cargo tauri build' has been run first."
    exit 1
}

if (-not (Test-Path $BundleDir)) {
    Write-Error "ort-bundle/ directory not found: $BundleDir"
    Write-Error "Run setup-windows.ps1 (CUDA) or setup-windows.ps1 -DirectML first."
    exit 1
}

$allDlls = @(Get-ChildItem "$BundleDir\*.dll" -ErrorAction SilentlyContinue)
if ($allDlls.Count -eq 0) {
    Write-Error "No DLLs found in $BundleDir"
    Write-Error "Run setup-windows.ps1 to populate ort-bundle/."
    exit 1
}

Write-Host "=== inject-gpu-dlls ===" -ForegroundColor Cyan
Write-Host "  Target:      $TargetTriple"
Write-Host "  Product:     $ProductName v$Version"
Write-Host "  NSIS script: $NsisScript"
Write-Host "  DLLs:        $($allDlls.Count) found in ort-bundle/"
foreach ($d in $allDlls) {
    $sz = [math]::Round($d.Length / 1MB, 1)
    Write-Host "    $($d.Name) ($sz MB)"
}
Write-Host ""

# =========================================================================
# 2. Read and patch the NSIS script
# =========================================================================

Write-Host "[1/4] Patching NSIS script..." -ForegroundColor Yellow

# Read as UTF-8 (Tauri writes NSIS scripts as UTF-8)
$script = [System.IO.File]::ReadAllText($NsisScript, [System.Text.UTF8Encoding]::new($false))

# ---- 2a. Switch from solid to non-solid compression ----------------------
# Solid mode compressors mmap the entire output data block → overflows
# NSIS's 32-bit address space when large files are included.  Non-solid
# compresses each file independently and respects SetCompress off.
$solidMarker  = 'SetCompressor /SOLID "lzma"'
$nonSolidLine = 'SetCompressor "lzma"'
if ($script.IndexOf($solidMarker) -ge 0) {
    $script = $script.Replace($solidMarker, $nonSolidLine)
    Write-Host "  Switched: /SOLID lzma -> lzma (non-solid)"
} else {
    Write-Host "  NOTE: /SOLID marker not found - may already be non-solid"
}

# ---- 2b. Build injection strings ----------------------------------------
# Install section: File commands for each GPU DLL
# Uninstall section: Delete for each GPU DLL
$installLines   = ""
$uninstallLines = ""
$precompiledName = "cudnn_engines_precompiled64_9.dll"

foreach ($dll in $allDlls) {
    $absPath = $dll.FullName
    if ($dll.Name -eq $precompiledName) {
        # Store UNCOMPRESSED - NSIS 32-bit cannot mmap this 500+ MB file
        $installLines += @"

  ; $($dll.Name) - stored UNCOMPRESSED (NSIS 32-bit mmap limit)
  SetCompress off
  File /a "/oname=$($dll.Name)" "$absPath"
  SetCompress auto
"@
    } else {
        # Normal LZMA compression (non-solid, per-file)
        $installLines += @"

  File /a "/oname=$($dll.Name)" "$absPath"
"@
    }
    $uninstallLines += @"

  Delete "`$INSTDIR\$($dll.Name)"
"@
}

# ---- 2c. Inject into Install section -------------------------------------
$installMarker = "; Copy external binaries"
if ($script.IndexOf($installMarker) -lt 0) {
    Write-Error "Cannot find install marker '$installMarker' in NSIS script."
    exit 1
}
$script = $script.Replace($installMarker, "$installMarker$installLines")

# ---- 2d. Inject into Uninstall section -----------------------------------
$uninstallMarker = "; Delete external binaries"
if ($script.IndexOf($uninstallMarker) -lt 0) {
    Write-Error "Cannot find uninstall marker '$uninstallMarker' in NSIS script."
    exit 1
}
$script = $script.Replace($uninstallMarker, "$uninstallLines$uninstallMarker")

# ---- 2e. Write back (UTF-8, no BOM) --------------------------------------
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($NsisScript, $script, $utf8NoBom)
Write-Host "  Patched: $($allDlls.Count) GPU DLL(s) injected into NSIS script"

# =========================================================================
# 3. Locate makensis (Tauri bundles its own)
# =========================================================================

Write-Host "[2/4] Locating makensis..." -ForegroundColor Yellow

function Find-Makensis {
    $tauriBase = Join-Path $env:LOCALAPPDATA "tauri\NSIS"
    if (Test-Path $tauriBase) {
        $tauriCandidates = @(
            (Join-Path $tauriBase "Bin\makensis.exe"),
            (Join-Path $tauriBase "makensis.exe")
        )
        foreach ($c in $tauriCandidates) {
            if (Test-Path $c) {
                Write-Host "  Found Tauri-bundled NSIS: $c"
                return $c
            }
        }
        $found = Get-ChildItem -Path $tauriBase -Recurse -Depth 2 -Filter "makensis.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Write-Host "  Found Tauri-bundled NSIS: $($found.FullName)"
            return $found.FullName
        }
    }
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
    $fromPath = Get-Command "makensis.exe" -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    return $null
}

$makensis = Find-Makensis
if (-not $makensis) {
    Write-Error @"
makensis not found.  Tauri auto-downloads NSIS to %LOCALAPPDATA%\tauri\NSIS\
during the first `cargo tauri build`.  Run a build once to trigger the download.
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

$arch = if ($TargetTriple -like "*aarch64*") { "arm64" } else { "x64" }
$expectedName = "${ProductName}_${Version}_${arch}-setup.exe"
$bundleDir   = Join-Path $ReleaseDir "bundle\nsis"
$finalPath   = Join-Path $bundleDir $expectedName

New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
if (Test-Path $finalPath) { Remove-Item $finalPath -Force }
Move-Item $generatedExe -Destination $finalPath -Force

$finalSizeMB = [math]::Round((Get-Item $finalPath).Length / 1MB, 1)
Write-Host "  Done: $finalPath ($finalSizeMB MB)" -ForegroundColor Green
Write-Host ""
Write-Host "=== inject-gpu-dlls: SUCCESS ===" -ForegroundColor Green
