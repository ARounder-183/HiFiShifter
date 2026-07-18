<#
.SYNOPSIS
    Verify that ORT + DirectML DLLs are present in the release directory.

.DESCRIPTION
    Checks the release directory for the required DLLs needed for DirectML
    GPU-accelerated inference. Used by CI as a post-build gate.

    NOTE: DirectML is compiled into onnxruntime.dll — there is no separate
    onnxruntime_providers_dml.dll file.

.PARAMETER ReleaseDir
    Path to the release directory (cargo target/<triple>/release).

.EXAMPLE
    .\scripts\verify-gpu-dlls.ps1 -ReleaseDir "backend\src-tauri\target\x86_64-pc-windows-msvc\release"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseDir,
    [switch]$OpenCL,
    [switch]$DirectML
)

$ErrorActionPreference = "Stop"

$required = @("onnxruntime.dll", "onnxruntime_providers_shared.dll")

Write-Host "=== Release directory contents ==="
Get-ChildItem "$ReleaseDir\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name) ($sizeMB MB)"
}
Write-Host ""

$missing = $required | Where-Object { -not (Test-Path (Join-Path $ReleaseDir $_)) }
if ($missing) {
    Write-Host "ERROR: Missing DLLs: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All $($required.Count) DLLs verified." -ForegroundColor Green

$nsisDir = Join-Path $ReleaseDir "bundle\nsis"
if (Test-Path $nsisDir) {
    Write-Host "=== NSIS bundle DLLs ==="
    Get-ChildItem "$nsisDir\*.dll" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.FullName)"
    }
}
