# Import the Android toolchain environment into the current session (dot-source it, do not execute):
#   . .\scripts\env\android-env.ps1
#   Set-AndroidEnvironment
#
# [Why this file exists] Four consumers read four different things, and every failure mode
# names something other than the real cause:
#
#   - `cc` (WORLD, Signalsmith Stretch and fdk-aac are all compiled by it) finds the NDK
#     compiler **only through PATH**. cc-rs detects Android by probing for the standalone
#     wrapper (`aarch64-linux-android21-clang`) with `Command::new(..).output().is_ok()`
#     and never reads ANDROID_NDK_HOME. With the NDK absent from PATH, the first C++ crate
#     dies with "fatal error: 'stdlib.h' file not found" -- which reads like a broken host
#     toolchain, not a missing environment variable.
#   - tauri-cli reads ANDROID_NDK_HOME / NDK_HOME to pick the toolchain and JAVA_HOME to
#     drive gradle.
#   - gradle reads the SDK through ANDROID_HOME **or** ANDROID_SDK_ROOT; the two tools do
#     not agree on which one, so both are set here.
#   - `cargo check --target aarch64-linux-android` does not link, so it needs no linker
#     configuration -- but its build scripts still need a working compiler.
#
# [Why discovery instead of a hardcoded path] Same reasoning as scripts/env/msvc-env.ps1:
# the SDK may live on any drive, the NDK is installed side by side (several versions), and
# the JDK is either a portable extract or a system install. Explicit overrides are honoured
# first (HIFISHIFTER_ANDROID_SDK / HIFISHIFTER_ANDROID_NDK / HIFISHIFTER_JDK), then the
# conventional locations.
#
# [Why everything lives inside functions] This file is **dot-sourced**: top-level variables
# would land in the caller's scope and overwrite its variables (msvc-env.ps1 lost the repo
# root this way once, which surfaced as "models.json not found"). Only `$env:*` leaves the
# process environment behind, which is exactly the one thing the importer should leave.

$ErrorActionPreference = 'Stop'

# [Why the generator goes through the shared module] The NDK toolchain file cannot drive the
# Visual Studio generator, so Android must use Ninja -- but the assignment used to be a bare
# "$env:CMAKE_GENERATOR = 'Ninja'" here, which (because this file is dot-sourced) survived into
# whatever ran next in the same shell, desktop builds included. Set-NativeToolchain writes the
# same variable while also recording it in HFS_NATIVE_GENERATOR, so the next env script can see
# what was decided instead of re-guessing.
. (Join-Path $PSScriptRoot 'native-toolchain.ps1')

function Get-AndroidSdkRoot {
    # An override wins even when it is unconventional: the caller knows its own layout.
    foreach ($name in @('HIFISHIFTER_ANDROID_SDK', 'ANDROID_SDK_ROOT', 'ANDROID_HOME')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value -and (Test-Path -LiteralPath (Join-Path $value 'platform-tools'))) {
            return [IO.Path]::GetFullPath($value)
        }
    }

    # platform-tools is the cheapest reliable marker: every usable SDK has it, and it is
    # what `adb` comes from. Testing for "cmdline-tools" instead would reject SDKs that
    # were installed by Android Studio alone.
    $candidates = @()
    $localAppData = [Environment]::GetFolderPath('LocalApplicationData')
    if ($localAppData) { $candidates += (Join-Path $localAppData 'Android\Sdk') }
    $programFiles = [Environment]::GetFolderPath('ProgramFiles')
    if ($programFiles) { $candidates += (Join-Path $programFiles 'Android\Sdk') }

    # The SDK is regularly kept off the system partition (it is several GB); such an
    # install has no environment variable at all, so scan drive roots for the two
    # conventional names.
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        if (-not $drive.Root) { continue }
        $candidates += (Join-Path $drive.Root 'Android\Sdk')
        $candidates += (Join-Path $drive.Root 'Android\android-sdk')
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'platform-tools')) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    return $null
}

function Get-AndroidNdkRoot {
    foreach ($name in @('HIFISHIFTER_ANDROID_NDK', 'ANDROID_NDK_HOME', 'NDK_HOME', 'ANDROID_NDK_ROOT')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        # source.properties is the NDK's own manifest; requiring it rejects a directory
        # that merely has the right name.
        if ($value -and (Test-Path -LiteralPath (Join-Path $value 'source.properties'))) {
            return [IO.Path]::GetFullPath($value)
        }
    }

    $sdk = Get-AndroidSdkRoot
    if (-not $sdk) { return $null }
    $ndkRoot = Join-Path $sdk 'ndk'
    if (-not (Test-Path -LiteralPath $ndkRoot)) { return $null }

    # NDK installs are version-named directories, so a string sort is wrong ("9.x" would
    # beat "27.x"). Parse the name as a version instead.
    $newest = Get-ChildItem -LiteralPath $ndkRoot -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'source.properties') } |
        Sort-Object {
            $parsed = [version]'0.0'
            [void][version]::TryParse($_.Name, [ref]$parsed)
            $parsed
        } -Descending |
        Select-Object -First 1
    if ($newest) { return $newest.FullName }
    return $null
}

function Get-AndroidJdkRoot {
    foreach ($name in @('HIFISHIFTER_JDK', 'JAVA_HOME')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value -and (Test-Path -LiteralPath (Join-Path $value 'bin\java.exe'))) {
            return [IO.Path]::GetFullPath($value)
        }
    }

    $candidates = @()

    # A portable extract next to the SDK (the layout this repository's setup documents).
    $sdk = Get-AndroidSdkRoot
    if ($sdk) {
        $sdkParent = Split-Path -Parent $sdk
        foreach ($name in @('jdk17', 'jdk-17', 'jdk', 'jdk21', 'jdk-21')) {
            $candidates += (Join-Path $sdkParent $name)
        }
    }

    # System installs, by vendor. The product directory carries the version, so each
    # vendor is globbed rather than guessed.
    $roots = @()
    $programFiles = [Environment]::GetFolderPath('ProgramFiles')
    if ($programFiles) {
        $roots += (Join-Path $programFiles 'Eclipse Adoptium')
        $roots += (Join-Path $programFiles 'Java')
        $roots += (Join-Path $programFiles 'Microsoft')
        $roots += (Join-Path $programFiles 'Amazon Corretto')
        $roots += (Join-Path $programFiles 'Zulu')
        $roots += (Join-Path $programFiles 'BellSoft')
    }
    $localAppData = [Environment]::GetFolderPath('LocalApplicationData')
    if ($localAppData) {
        $roots += (Join-Path $localAppData 'Programs\Eclipse Adoptium')
        $roots += (Join-Path $localAppData 'Programs\Microsoft')
    }
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $candidates += @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^(jdk|jdk-)17' -or $_.Name -match '^(jdk|jdk-)21' } |
            ForEach-Object { $_.FullName })
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'bin\java.exe')) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    return $null
}

function Get-AndroidNdkHostTag {
    # The NDK ships one prebuilt tree per host (windows-x86_64, linux-x86_64, darwin-x86_64).
    # Reading it from disk keeps the same script working in WSL and on a Linux runner.
    param([Parameter(Mandatory = $true)][string]$NdkRoot)
    $prebuilt = Join-Path $NdkRoot 'toolchains\llvm\prebuilt'
    if (-not (Test-Path -LiteralPath $prebuilt)) { return $null }
    $tag = Get-ChildItem -LiteralPath $prebuilt -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($tag) { return $tag.FullName }
    return $null
}

function Get-AndroidSdkCmakeBin {
    <#
    .SYNOPSIS
    Returns the bin directory of the newest cmake the SDK ships, or $null.
    .DESCRIPTION
    [Why the SDK's cmake and not the system one] Two reasons, both empirical:
      - `cmake-rs` uses the FIRST `cmake` on PATH. A system CMake 4.x removes
        compatibility with `cmake_minimum_required(VERSION <3.5)`, which several
        audio libraries still declare; the SDK's 3.31 still accepts them.
      - The SDK also ships `ninja`, which the NDK toolchain file needs (the Visual
        Studio generator cannot drive it).
    #>
    param([Parameter(Mandatory = $true)][string]$SdkRoot)
    $cmakeRoot = Join-Path $SdkRoot 'cmake'
    if (-not (Test-Path -LiteralPath $cmakeRoot)) { return $null }
    $newest = Get-ChildItem -LiteralPath $cmakeRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\cmake.exe') } |
        Sort-Object {
            $parsed = [version]'0.0'
            [void][version]::TryParse($_.Name, [ref]$parsed)
            $parsed
        } -Descending |
        Select-Object -First 1
    if (-not $newest) { return $null }
    return (Join-Path $newest.FullName 'bin')
}

function Get-AndroidLogShimRoot {
    <#
    .SYNOPSIS
    Returns backend/build-support/android-log-shim (the fdk-aac header shim).
    .DESCRIPTION
    [Why the include path is injected through CPATH and not through CFLAGS] CPATH is a
    **path list**, so a repository path containing spaces needs no quoting and no
    `CC_SHELL_ESCAPED_FLAGS`. clang treats each entry as if it had been passed with -I,
    for both C and C++, which is exactly what fdk-aac's `#include "log/log.h"` needs.
    See backend/build-support/android-log-shim/log/log.h for why the shim exists.
    #>
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $shim = Join-Path $repoRoot 'backend\build-support\android-log-shim'
    if (-not (Test-Path -LiteralPath (Join-Path $shim 'log\log.h'))) { return $null }
    return $shim
}

function Get-AndroidMinSdk {
    <#
    .SYNOPSIS
    Reads minSdkVersion from the generated Android project; 26 when it is not there yet.
    .DESCRIPTION
    [Why the toolchain has to agree with it] The NDK only ships libaaudio.so under
    <sysroot>/usr/lib/<abi>/<api>/ for api >= 26. Linking with an API-24 wrapper makes the
    -laaudio emitted by cpal's AAudio backend unresolvable, and the error is
    "ld.lld: error: unable to find library -laaudio" -- which names a library, not the API
    level that is too low. Compiling at exactly the app's minSdk is also what makes the
    compiler reject calls to APIs that do not exist on the oldest supported device.
    #>
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $gradle = Join-Path $repoRoot 'backend\src-tauri\gen\android\app\build.gradle.kts'
    if (Test-Path -LiteralPath $gradle) {
        $match = [regex]::Match((Get-Content -LiteralPath $gradle -Raw), 'minSdk\s*=\s*(\d+)')
        if ($match.Success) { return [int]$match.Groups[1].Value }
    }
    # Same value as backend/src-tauri/tauri.conf.json's bundle.android.minSdkVersion.
    return 26
}

function Get-AndroidNdkWrappers {
    <#
    .SYNOPSIS
    Resolves the NDK compiler wrappers for one Rust android triple.
    .DESCRIPTION
    Returns $null when the ABI has no wrapper in this NDK (for example an old NDK that
    predates 64-bit-only ABIs).
    #>
    param(
        [Parameter(Mandatory = $true)][string]$NdkBin,
        [Parameter(Mandatory = $true)][string]$NdkPrefix,
        # Lowest API level the wrappers may target; see Get-AndroidMinSdk.
        [int]$MinApi = 26
    )

    # The NDK ships one wrapper per (ABI, API level): <prefix><api>-clang[++][.cmd].
    # The **lowest** API level is the right pick. Compiling against a higher one lets the
    # compiler emit calls that do not exist on a device whose minSdkVersion is lower, and
    # that failure appears as an unresolved symbol when the app loads -- not at build time.
    #
    # [Why the .cmd form wins on Windows] Windows also ships an extensionless shell script
    # of the same name for MSYS/Git-Bash. CreateProcess cannot execute it, and `cc` probes
    # candidates with `Command::new(..)`, so handing it the script yields "program not
    # found" even though the file is right there.
    $pattern = '^' + [regex]::Escape($NdkPrefix) + '(\d+)-clang(\+\+)?(\.cmd|\.exe)?$'
    $byLevel = @{}
    foreach ($file in (Get-ChildItem -LiteralPath $NdkBin -File -ErrorAction SilentlyContinue)) {
        $match = [regex]::Match($file.Name, $pattern)
        if (-not $match.Success) { continue }
        $api = [int]$match.Groups[1].Value
        $isCxx = $match.Groups[2].Success
        $isWindowsForm = $match.Groups[3].Success
        if (-not $byLevel.ContainsKey($api)) { $byLevel[$api] = @{} }
        $slot = if ($isCxx) { 'Cxx' } else { 'Cc' }
        # Prefer the .cmd/.exe form; keep the first one seen otherwise.
        if (-not $byLevel[$api].ContainsKey($slot) -or $isWindowsForm) {
            $byLevel[$api][$slot] = $file.FullName
        }
    }
    if ($byLevel.Count -eq 0) { return $null }

    $api = ($byLevel.Keys | Where-Object { $_ -ge $MinApi } | Sort-Object)[0]
    if (-not $api) { return $null }
    $entry = $byLevel[$api]
    if (-not $entry.ContainsKey('Cc')) { return $null }

    $ar = Join-Path $NdkBin 'llvm-ar.exe'
    if (-not (Test-Path -LiteralPath $ar)) { $ar = Join-Path $NdkBin 'llvm-ar' }

    return [pscustomobject]@{
        Api = $api
        Cc = $entry['Cc']
        Cxx = $entry['Cxx']
        Ar = $ar
    }
}

function Set-AndroidEnvironment {
    <#
    .SYNOPSIS
    Exports the Android toolchain environment and prepends the tools to PATH.
    .DESCRIPTION
    Throws when a piece is missing: a half-configured environment produces build errors
    that point at unrelated crates, which is far more expensive than one clear message.
    #>
    $sdk = Get-AndroidSdkRoot
    if (-not $sdk) {
        throw ('Android SDK not found. Install it (or set HIFISHIFTER_ANDROID_SDK / ' +
            'ANDROID_SDK_ROOT) so that <sdk>\platform-tools exists.')
    }
    $ndk = Get-AndroidNdkRoot
    if (-not $ndk) {
        throw ("No Android NDK under $(Join-Path $sdk 'ndk'). Install one with: sdkmanager 'ndk;27.0.12077973'")
    }
    $jdk = Get-AndroidJdkRoot
    if (-not $jdk) {
        throw ('No JDK found (gradle needs 17+). Install one, or point HIFISHIFTER_JDK at it.')
    }
    $ndkBin = Get-AndroidNdkHostTag -NdkRoot $ndk
    if (-not $ndkBin) {
        throw ("NDK at $ndk has no toolchains\llvm\prebuilt tree; the install is incomplete.")
    }
    $ndkBin = Join-Path $ndkBin 'bin'

    $env:ANDROID_HOME = $sdk
    $env:ANDROID_SDK_ROOT = $sdk
    $env:ANDROID_NDK_HOME = $ndk
    $env:NDK_HOME = $ndk
    $env:JAVA_HOME = $jdk

    # ── Per-target compiler/linker overrides ─────────────────────────────────────
    #
    # [Why these are set explicitly and are not redundant with PATH] cc-rs autodetects
    # Android by probing the **standalone wrapper names** (`aarch64-linux-android21-clang`
    # etc.) with `Command::new(..).output()`. On Windows the wrapper only exists as a
    # `.cmd`, which CreateProcess cannot launch by its bare name, so the probe fails and
    # cc-rs falls back to a bare `clang --target=aarch64-linux-android` -- **without an API
    # level and without --sysroot**. That clang then cannot find `<stdlib.h>` or
    # `<android/log.h>`, and fdk-aac / opus fail with errors that look like a broken host
    # toolchain. Naming the compiler per target is the documented cc-rs escape hatch and
    # is correct on every host.
    #
    # [Why the linker override] `cargo build` (not `cargo check`) needs a linker for the
    # target; rustc's default for `aarch64-linux-android` is the GNU `-gcc` name, which no
    # current NDK ships.
    $triples = [ordered]@{
        'aarch64-linux-android'   = 'aarch64-linux-android'
        'armv7-linux-androideabi' = 'armv7a-linux-androideabi'
        'i686-linux-android'      = 'i686-linux-android'
        'x86_64-linux-android'    = 'x86_64-linux-android'
    }
    $minSdk = Get-AndroidMinSdk
    $toolchains = [ordered]@{}
    foreach ($triple in $triples.Keys) {
        $wrappers = Get-AndroidNdkWrappers -NdkBin $ndkBin -NdkPrefix $triples[$triple] -MinApi $minSdk
        if (-not $wrappers) { continue }
        $underscored = $triple.Replace('-', '_')
        $upper = $triple.Replace('-', '_').ToUpperInvariant()
        Set-Item -Path ("Env:CC_" + $underscored) -Value $wrappers.Cc
        if ($wrappers.Cxx) { Set-Item -Path ("Env:CXX_" + $underscored) -Value $wrappers.Cxx }
        Set-Item -Path ("Env:AR_" + $underscored) -Value $wrappers.Ar
        Set-Item -Path ("Env:CARGO_TARGET_" + $upper + "_LINKER") -Value $wrappers.Cc
        $toolchains[$triple] = $wrappers
    }
    if ($toolchains.Count -eq 0) {
        throw ("NDK at $ndk has no compiler wrapper for any Android ABI; the install is incomplete.")
    }

    # Order matters: the NDK bin must come before any host clang that may be on PATH
    # (LLVM ships one with Visual Studio), otherwise cc-rs resolves `clang` to the host
    # compiler and every Android C++ translation unit fails on missing system headers.
    # cmake + ninja from the SDK go first: the NDK toolchain file cannot drive the Visual
    # Studio generator, and a system CMake 4.x rejects the `cmake_minimum_required`
    # versions that libopus still declares. See Get-AndroidSdkCmakeBin.
    $cmakeBin = Get-AndroidSdkCmakeBin -SdkRoot $sdk

    $additions = @(
        (Join-Path $jdk 'bin'),
        $ndkBin,
        (Join-Path $sdk 'platform-tools'),
        (Join-Path $sdk 'cmdline-tools\latest\bin'),
        $cmakeBin
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    # Idempotent: dot-sourcing twice must not stack PATH (the environment block is
    # limited to 8191 characters and long builds add to it already).
    $segments = @($env:PATH -split ';' | Where-Object { $_ })
    $missing = @($additions | Where-Object { $segments -notcontains $_ })
    if ($missing.Count -gt 0) {
        $env:PATH = (($missing + $segments) -join ';')
    }

    # [Why the generator is pinned] cmake-rs leaves the generator to CMake's default, which
    # on Windows is "Visual Studio <version>". The NDK's android.toolchain.cmake cannot be
    # used with that generator: MSBuild then compiles for the host and fails with
    # "VCTargetsPath.vcxproj" errors that name no Android concept at all. Ninja is the
    # generator the NDK documents and the one the SDK ships a binary for.
    if ($cmakeBin) {
        # The SDK's cmake directory also carries the ninja binary the NDK toolchain needs, and
        # it is already first on PATH at this point, so the probe inside Set-NativeToolchain
        # finds it. Passing -Generator explicitly makes the choice independent of PATH order.
        Set-NativeToolchain -Generator Ninja
    }

    # [Why CPATH carries the fdk-aac shim] fdk-aac-sys includes AOSP's internal log/log.h
    # under `#ifdef __ANDROID__`, which the NDK does not ship. The shim provides the one
    # symbol it needs; the reasoning and the exact upstream lines are in
    # backend/build-support/android-log-shim/log/log.h.
    $logShim = Get-AndroidLogShimRoot
    if ($logShim) {
        $separator = [IO.Path]::PathSeparator
        $entries = @($env:CPATH -split [regex]::Escape($separator) | Where-Object { $_ })
        if ($entries -notcontains $logShim) {
            $env:CPATH = (@($logShim) + $entries) -join $separator
        }
    }

    return [pscustomobject]@{
        SdkRoot = $sdk
        NdkRoot = $ndk
        NdkBin = $ndkBin
        JdkRoot = $jdk
        MinSdk = $minSdk
        CmakeBin = $cmakeBin
        LogShim = $logShim
        Toolchains = $toolchains
    }
}

function Test-AndroidToolchain {
    <#
    .SYNOPSIS
    Reports which SDK packages are installed and whether the toolchain can compile.
    .DESCRIPTION
    Called by scripts/build/android.ps1 before cargo runs: "the SDK is installed but no
    platform is" otherwise surfaces as a gradle error that names a file path.
    #>
    $sdk = Get-AndroidSdkRoot
    if (-not $sdk) {
        return [pscustomobject]@{ Ready = $false; Problems = @('Android SDK not found') }
    }
    $problems = @()

    $platforms = @(Get-ChildItem -LiteralPath (Join-Path $sdk 'platforms') -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'android.jar') } |
        ForEach-Object { $_.Name })
    if ($platforms.Count -eq 0) { $problems += "no SDK platform installed under $(Join-Path $sdk 'platforms')" }

    $buildTools = @(Get-ChildItem -LiteralPath (Join-Path $sdk 'build-tools') -Directory -ErrorAction SilentlyContinue |
        Where-Object { (Test-Path -LiteralPath (Join-Path $_.FullName 'aapt2.exe')) -or (Test-Path -LiteralPath (Join-Path $_.FullName 'aapt2')) } |
        ForEach-Object { $_.Name })
    if ($buildTools.Count -eq 0) { $problems += "no build-tools installed under $(Join-Path $sdk 'build-tools')" }

    $ndk = Get-AndroidNdkRoot
    if (-not $ndk) { $problems += 'no Android NDK (see Get-AndroidNdkRoot)' }

    $jdk = Get-AndroidJdkRoot
    if (-not $jdk) { $problems += 'no JDK' }

    return [pscustomobject]@{
        Ready = ($problems.Count -eq 0)
        SdkRoot = $sdk
        NdkRoot = $ndk
        JdkRoot = $jdk
        Platforms = $platforms
        BuildTools = $buildTools
        Problems = $problems
    }
}
