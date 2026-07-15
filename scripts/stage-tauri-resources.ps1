<#
.SYNOPSIS
    Generate tauri.windows.conf.json for Windows GPU builds.

.DESCRIPTION
    Reads the DLLs present in the project-local ort-bundle directory and
    writes a tauri.windows.conf.json resource map for Tauri/NSIS bundling.

    Only DLLs that actually exist on disk are included - this prevents
    tauri_build::build() from panicking during resource validation.

    ort-bundle/ is the *single canonical location* for GPU DLLs.  It is
    populated by setup-windows.ps1 (local dev) or the CI download steps.
    No ORT_LIB_LOCATION env var is needed.

.PARAMETER ProjectRoot
    Project root directory. Defaults to two levels above this script.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Continue"

try {
    if (-not $ProjectRoot) {
        $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    }

    $BundleDir   = Join-Path $ProjectRoot "backend\src-tauri\third_party\ort-bundle"
    $ConfigPath  = Join-Path $ProjectRoot "backend\src-tauri\tauri.windows.conf.json"

    Write-Host "=== stage-tauri-resources ===" -ForegroundColor Cyan
    Write-Host "  ProjectRoot : $ProjectRoot"
    Write-Host "  BundleDir   : $BundleDir"

    # ---- 1. Build resource map - only files that exist on disk ------------
    $resources = [ordered]@{}

    # Non-GPU resources (always present, committed to git)
    $vslib = Join-Path $ProjectRoot "backend\src-tauri\third_party\vslib\vslib_x64.dll"
    if (Test-Path $vslib) {
        $resources["third_party/vslib/vslib_x64.dll"] = "vslib_x64.dll"
    }

    $stDll = Join-Path $ProjectRoot "backend\src-tauri\third_party\soundtouch-static\soundtouch\SoundTouchDLL.dll"
    if (Test-Path $stDll) {
        $resources["third_party/soundtouch-static/soundtouch/SoundTouchDLL.dll"] = "SoundTouchDLL.dll"
    } else {
        Write-Host "  NOTE: SoundTouchDLL.dll not yet built - will be added at build time"
    }

    # GPU DLLs from the canonical ort-bundle directory
    $gpuCount = 0
    if (Test-Path $BundleDir) {
        $dlls = @(Get-ChildItem "$BundleDir\*.dll" -ErrorAction SilentlyContinue)
        Write-Host "  GPU DLLs found: $($dlls.Count)"
        foreach ($dll in $dlls) {
            # cudnn_engines_precompiled64_9.dll is the cuDNN 9.x precompiled
            # engine cache - typically 500–800 MB.  Including it in the NSIS
            # installer causes a makensis INTERNAL COMPILER ERROR:
            #   "Internal compiler error #12345: error mmapping datablock to …"
            #
            # NSIS is a 32-bit process; its solid LZMA compressor needs to
            # memory-map the entire compressed data block, and a multi-GB
            # payload overflows the 32-bit virtual address space.
            #
            # cuDNN 9.x gracefully falls back to RUNTIME compilation via
            # cudnn_engines_runtime_compiled64_9.dll (included below), so
            # CUDA functionality is not affected - only the very first
            # convolution operation incurs a small one-time compilation delay.
            #
            # The precompiled engine DLL IS included in the portable ZIP
            # (see pack-portable.ps1), which is the recommended distribution
            # format for GPU users who want the fastest cold-start experience.
            if ($dll.Name -eq "cudnn_engines_precompiled64_9.dll") {
                Write-Host "    SKIP (NSIS 32-bit mmap limit): $($dll.Name)" -ForegroundColor DarkYellow
                Write-Host "          cuDNN will use runtime-compiled engines as fallback." -ForegroundColor DarkGray
                continue
            }
            $resources["third_party/ort-bundle/$($dll.Name)"] = $dll.Name
            $gpuCount++
        }
    }

    if ($gpuCount -gt 0) {
        Write-Host "  GPU DLLs in resource map: $gpuCount" -ForegroundColor Green
    } else {
        Write-Warning "No GPU DLLs found in $BundleDir - installer will lack GPU support.`nRun .\scripts\setup-windows.ps1 first, then rebuild."
    }

    # ---- 2. Write config (UTF-8 WITHOUT BOM - BOM breaks tauri_build) ----
    $config = @{ bundle = @{ resources = $resources } }
    $json = $config | ConvertTo-Json -Depth 3 -Compress
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ConfigPath, $json, $utf8NoBom)
    Write-Host "  Wrote $($resources.Count) resource(s) to $ConfigPath"
    Write-Host "Done." -ForegroundColor Green
}
catch {
    Write-Error "stage-tauri-resources.ps1 failed:`n  $_"
    exit 1
}
