# Single source of truth for the plugin SDK (ARA_SDK / vst3sdk) checkouts.
#
# This file answers only three questions: where the checkouts live, where the local mirror
# is, and which environment variables consumers must set.
#   - Which repository / commit / tree is pinned by scripts/env/plugin-sdks.json;
#   - Downloading and verification are handled by scripts/setup/prepare-plugin-sdks.ps1;
#   - The local mirror is produced by scripts/setup/export-sdk-mirror.ps1.
#
# Usage (call after dot-sourcing):
#   . .\scripts\env\sdk-env.ps1
#   Set-SdkEnvironment            # sets ARA_SDK_DIR / ARA_VST3_SDK_DIR
#   Get-SdkCacheRoot              # returns the cache root
#   Get-SdkMirrorRoot             # returns the local mirror root (HIFISHIFTER_SDK_MIRROR overrides)
#   Test-SdkOfflineRequested      # whether fetching sources over the network is forbidden (HIFISHIFTER_OFFLINE)

function Get-SdkCacheRoot {
    # Per the pins, downloaded into the gitignored third_party/sdk inside the repository:
    # keeps the repo lean and fulfills the license attribution obligation.
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\third_party\sdk'))
}

function Get-SdkMirrorRoot {
    # A copyable local source cache shaped like the cache root (containing ARA_SDK/,
    # vst3sdk/, soundtouch/). Exported on a networked machine and copied to the offline
    # one; it never touches the network.
    if ($env:HIFISHIFTER_SDK_MIRROR) {
        return [IO.Path]::GetFullPath($env:HIFISHIFTER_SDK_MIRROR)
    }
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\third_party\sdk-mirror'))
}

function Test-SdkOfflineRequested {
    "$env:HIFISHIFTER_OFFLINE".Trim() -in @('1', 'true', 'yes', 'on')
}

function Set-SdkEnvironment {
    # Both ara2-bridge's build.rs and audio_abi's ABI oracle read these variables.
    $sdkRoot = Get-SdkCacheRoot
    $env:ARA_SDK_DIR = Join-Path $sdkRoot 'ARA_SDK'
    $env:ARA_VST3_SDK_DIR = Join-Path $sdkRoot 'vst3sdk'
    $env:HIFISHIFTER_CLAP_SDK_DIR = Join-Path $sdkRoot 'clap'
}
