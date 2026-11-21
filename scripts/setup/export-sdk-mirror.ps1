# Export the local SDK mirror: copy the verified pinned checkouts (ARA_SDK / vst3sdk / clap, plus
# SoundTouch) wholesale into the mirror root for offline machines to fetch sources from. The
# mirror is just "another copyable local cache" serving the same developer's builds in a
# no-network environment, not a redistribution channel (licensing notes in README 'Third-party
# Sources').
#
# Usage (run once on a networked machine):
#   .\scripts\setup\export-sdk-mirror.ps1
#   .\scripts\setup\export-sdk-mirror.ps1 -WithTestkit                          # also export the ARA testkit submodules
#   .\scripts\setup\export-sdk-mirror.ps1 -MirrorDirectory E:\hfs-sdk-mirror    # write straight to a USB drive / LAN share
#
# Then, on the offline machine:
#   $env:HIFISHIFTER_SDK_MIRROR = 'E:\hfs-sdk-mirror'
#   $env:HIFISHIFTER_OFFLINE = '1'
#   .\build.bat
[CmdletBinding()]
param([string]$MirrorDirectory, [switch]$WithTestkit)
$ErrorActionPreference = 'Stop'
$exportRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
# The mirror root and cache root are defined in one place by scripts/env/sdk-env.ps1 (single source of truth).
. (Join-Path $PSScriptRoot '..\env\sdk-env.ps1')
if ($MirrorDirectory) { $exportMirrorRoot = [IO.Path]::GetFullPath($MirrorDirectory) }
else { $exportMirrorRoot = Get-SdkMirrorRoot }
$exportCacheRoot = Get-SdkCacheRoot
$exportLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\env\plugin-sdks.json') -Raw | ConvertFrom-Json

function Copy-IntoMirror($Source, $Name) {
    $exportTarget = Join-Path $exportMirrorRoot $Name
    if (Test-Path -LiteralPath $exportTarget) { Remove-Item -LiteralPath $exportTarget -Recurse -Force }
    Copy-Item -LiteralPath $Source -Destination $exportTarget -Recurse -Force
    Write-Output "Exported $Name -> $exportTarget"
}

# First run the same verification to ensure the cache holds the pinned copy (this step
# fills it in while online; once ready, it never touches the network).
$exportPrepareArgs = @{}
if ($WithTestkit) { $exportPrepareArgs['WithTestkit'] = $true }
& (Join-Path $PSScriptRoot 'prepare-plugin-sdks.ps1') @exportPrepareArgs

New-Item -ItemType Directory -Path $exportMirrorRoot -Force | Out-Null
# Keep this in sync with prepare-plugin-sdks.ps1's checkout list (ARA_SDK / vst3sdk /
# clap): the prepare side always prepares clap (HIFISHIFTER_CLAP_SDK_DIR, used by the
# ABI oracle); if the mirror lacks a copy, the offline machine's
# `prepare-plugin-sdks.ps1 -Offline` fails outright over the missing clap.
foreach ($exportSdk in @('ARA_SDK', 'vst3sdk', 'clap')) {
    Copy-IntoMirror (Join-Path $exportCacheRoot $exportSdk) $exportSdk
}

# SoundTouch is cloned automatically by the kernel build.rs into
# src-tauri/third_party/soundtouch-static/soundtouch; keep a copy in the mirror so
# build.rs can take it offline (see build.rs::build_soundtouch). Verify HEAD against
# the pin before exporting -- a mirror must not contain a source of unknown provenance.
$exportSoundtouch = Join-Path $exportRoot 'backend\src-tauri\third_party\soundtouch-static\soundtouch'
if (Test-Path -LiteralPath (Join-Path $exportSoundtouch 'CMakeLists.txt')) {
    $exportSoundtouchHead = & git -c safe.directory=* -C $exportSoundtouch rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or "$exportSoundtouchHead".Trim() -ne $exportLock.soundtouch.commit) {
        throw "SoundTouch HEAD '$("$exportSoundtouchHead".Trim())' does not match the pin $($exportLock.soundtouch.commit); refusing to export it."
    }
    Copy-IntoMirror $exportSoundtouch 'soundtouch'
}
else {
    Write-Output 'SoundTouch source not present; skipping its mirror entry (run a build once to clone it).'
}

Write-Output ''
Write-Output "Mirror ready: $exportMirrorRoot"
Write-Output 'On an offline machine, set HIFISHIFTER_SDK_MIRROR to this directory (or copy it to the'
Write-Output 'default mirror location), then run scripts/setup/prepare-plugin-sdks.ps1 -Offline. See README ''Third-party Sources''.'
