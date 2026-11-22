# Prepare the ARA / VST3 SDK checkouts at the revisions pinned by scripts/env/plugin-sdks.json.
#
# Order: the cache already holds the pinned copy -> local mirror (works offline) ->
# shallow clone over the network.
# "Holds the pinned copy" = commit, tree, clean worktree, and submodule identity all match.
[CmdletBinding()]
param([switch]$WithTestkit, [switch]$Offline)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env\sdk-env.ps1')
$sdkCache = Get-SdkCacheRoot
$sdkMirror = Get-SdkMirrorRoot
$sdkOffline = $Offline -or (Test-SdkOfflineRequested)
$sdkPins = Get-Content (Join-Path $PSScriptRoot '..\env\plugin-sdks.json') -Raw | ConvertFrom-Json

function Test-SdkIdentity($path, $commit, $tree) {
    if (!(Test-Path -LiteralPath $path)) { return $false }
    if ((& git -C $path rev-parse HEAD 2>$null) -ne $commit) { return $false }
    if ((& git -C $path rev-parse 'HEAD^{tree}' 2>$null) -ne $tree) { return $false }
    return -not (& git -C $path status --porcelain --untracked-files=all 2>$null)
}

function Test-SdkPin($path, $pin) {
    if (!(Test-SdkIdentity $path $pin.commit $pin.tree)) { return $false }
    # CLAP is a header-only SDK with no submodules; when the field is absent, only the main checkout is verified.
    if (!$pin.submodule) { return $true }
    Test-SdkIdentity (Join-Path $path $pin.submodule) $pin.submoduleCommit $pin.submoduleTree
}

function Invoke-Git($what, $arguments) {
    & git @arguments
    if ($LASTEXITCODE) { throw "$what failed" }
}

function Get-SdkCheckout($name, $pin) {
    $target = Join-Path $sdkCache $name
    if (Test-SdkPin $target $pin) {
        Write-Host "SDK ready (cache): $name"
        return
    }

    $mirrored = Join-Path $sdkMirror $name
    if (Test-SdkPin $mirrored $pin) {
        Write-Host "SDK imported from local mirror: $mirrored"
        Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
        Copy-Item $mirrored $target -Recurse
        if (!(Test-SdkPin $target $pin)) { throw "Mirror import does not match the pin: $name" }
        return
    }

    if ($sdkOffline) {
        throw "No usable ${name} checkout in offline mode: neither the cache nor the mirror holds the pinned copy (see README 'Third-party Sources')."
    }

    Write-Host "Downloading SDK: $name @ $($pin.commit)"
    Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
    Invoke-Git 'git init' @('init', $target)
    Invoke-Git 'git remote add' @('-C', $target, 'remote', 'add', 'origin', $pin.repository)
    Invoke-Git 'git fetch' @('-C', $target, 'fetch', '--depth', '1', 'origin', $pin.commit)
    Invoke-Git 'git checkout' @('-C', $target, '-c', 'core.autocrlf=false', 'checkout', '--detach', 'FETCH_HEAD')
    if ($pin.submodule) {
        Invoke-Git 'git submodule' @('-C', $target, '-c', 'core.autocrlf=false', 'submodule', 'update', '--init', '--depth', '1', '--', $pin.submodule)
    }
    if (!(Test-SdkPin $target $pin)) { throw "Downloaded checkout does not match the pin: $name" }
}

foreach ($entry in @(@('ARA_SDK', $sdkPins.ara), @('vst3sdk', $sdkPins.vst3), @('clap', $sdkPins.clap))) {
    Get-SdkCheckout $entry[0] $entry[1]
}

# Only ara2-bridge-testkit's cpp-interop needs ARA_Library / ARA_Examples;
# product builds do not, so they are not fetched by default. --recursive also
# pulls in ARA_Examples' nested 3rdParty.
if ($WithTestkit) {
    $araPath = Join-Path $sdkCache 'ARA_SDK'
    $araPin = $sdkPins.ara
    if ($sdkOffline) {
        foreach ($sub in $araPin.testkitSubmodules) {
            if (!(Test-SdkIdentity (Join-Path $araPath $sub.path) $sub.commit $sub.tree)) {
                throw "Offline mode: ARA testkit submodule is missing or off its pin: $($sub.path) (export the mirror with -WithTestkit)"
            }
        }
    }
    else {
        $roots = @($araPin.testkitSubmodules | Where-Object { $_.path -notmatch '/' } | ForEach-Object { $_.path })
        $updateArgs = @('-C', $araPath, '-c', 'core.autocrlf=false', 'submodule', 'update', '--init', '--depth', '1', '--recursive', '--') + $roots
        Invoke-Git 'testkit submodule update' $updateArgs
        foreach ($sub in $araPin.testkitSubmodules) {
            if (!(Test-SdkIdentity (Join-Path $araPath $sub.path) $sub.commit $sub.tree)) {
                throw "ARA testkit submodule does not match the pin: $($sub.path)"
            }
        }
    }
}
