# Build the Android package.
#
#   .\scripts\build\android.ps1                 # APK (side-load)
#   .\scripts\build\android.ps1 -Bundle         # AAB (Play)
#   .\scripts\build\android.ps1 -DebugBuild     # debug APK, no signing
#
# [Why the switch is called -DebugBuild and not -Debug] `[CmdletBinding()]` already gives
# the script a common `-Debug` parameter (it sets `$DebugPreference`). Declaring our own
# `-Debug` therefore fails to load the script at all ("parameter Debug defined multiple
# times"); and if the common one were used instead, the build would silently produce a
# release APK. -DebugBuild is the explicit, unshadowed spelling.
#
# [Why an explicit ABI and not "let tauri decide"] `tauri android build` compiles four
# ABIs by default (arm64-v8a, armeabi-v7a, x86, x86_64). ONNX Runtime ships a prebuilt
# distribution only for aarch64-linux-android -- the other three fail at link time with
# an error that names a crate nobody in this project has heard of. Restricting to
# arm64-v8a is not an optimisation, it is the only combination that links.
#
# [Why --target takes "aarch64" and not the Rust triple] `tauri android build --target`
# is the **ABI short name** (aarch64 / armv7 / i686 / x86_64). Passing the triple
# (`aarch64-linux-android`) is rejected by the CLI's value parser before anything is
# built, so the two spellings are not interchangeable.
#
# [Why the toolchain comes from scripts/env/android-env.ps1] Two failure modes live in
# the environment rather than in this script, and both name something other than the
# cause:
#   - cc-rs discovers the NDK compiler by probing standalone wrapper names with
#     `Command::new(..)`. On Windows the wrapper is a `.cmd`, which CreateProcess cannot
#     launch by its bare name, so the probe fails and cc-rs falls back to a bare
#     `clang --target=aarch64-linux-android` -- with no API level and no sysroot. fdk-aac
#     then dies on "'stdlib.h' file not found".
#   - gradle reads ANDROID_HOME **or** ANDROID_SDK_ROOT while tauri-cli reads
#     ANDROID_NDK_HOME / NDK_HOME; a half-set environment fails inside gradle.
# The env script owns both, so this script only has to call it.
#
# [Why the CLI is resolved instead of assumed] `cargo tauri` is what this repository
# uses for desktop builds (see scripts/build/build.ps1), but a clean CI runner only has
# npm. `npx tauri` is not a third option: there is no such executable on npm, and it
# fails with "could not determine executable to run", which says nothing about the real
# cause.
[CmdletBinding()]
param(
    [switch]$Bundle,
    # Not `-Debug`: that name belongs to CmdletBinding's common parameters (see the header).
    [switch]$DebugBuild,
    [string]$Target = "aarch64"
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $root

. (Join-Path $PSScriptRoot "..\env\android-env.ps1")
$toolchain = Set-AndroidEnvironment
Write-Host "== Android SDK: $($toolchain.SdkRoot)" -ForegroundColor Cyan
Write-Host "== Android NDK: $($toolchain.NdkRoot)" -ForegroundColor Cyan
Write-Host "== JDK:         $($toolchain.JdkRoot)" -ForegroundColor Cyan

# ── Windows symlink privilege ─────────────────────────────────────────────────
#
# [Why this preflight exists] tauri-cli stages the built library into the generated project
# with a **symbolic link** (cargo-mobile2's `os::windows::ln::force_symlink`, which reports
# ERROR_PRIVILEGE_NOT_HELD as "Creation symbolic link is not allowed for this system"). A
# non-elevated Windows token only carries SeCreateSymbolicLinkPrivilege when Developer Mode
# is on, so on a stock machine the build compiles for minutes and then stops at a message
# that reads like a toolchain problem. Checking first costs two seconds.
#
# [Why the remedy is not applied automatically] Both options need administrator rights
# (Developer Mode writes HKLM; granting the privilege is a local security policy change), so
# a build script must not attempt either silently.
function Test-CanCreateSymlink {
    $probe = Join-Path ([IO.Path]::GetTempPath()) ('hfs-symlink-probe-' + [guid]::NewGuid().ToString('N'))
    try {
        New-Item -ItemType Directory -Force -Path $probe | Out-Null
        $target = Join-Path $probe 'target.txt'
        Set-Content -LiteralPath $target -Value 'probe'
        New-Item -ItemType SymbolicLink -Path (Join-Path $probe 'link.txt') -Target $target -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
    finally {
        Remove-Item -LiteralPath $probe -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-CanCreateSymlink)) {
    throw @'
tauri-cli cannot stage the Android library: this Windows account may not create symbolic
links, and the staging step uses one.

  Remedies (either one; both need administrator rights):
    1. Settings -> System -> For developers -> Developer Mode: On
    2. Run this script from an elevated PowerShell (SeCreateSymbolicLinkPrivilege)

  What still works without it -- and is where code errors actually surface:
    . .\scripts\env\android-env.ps1; Set-AndroidEnvironment
    cargo build --target aarch64-linux-android --features onnx
  That compiles and links every Android source file; only packaging needs the symlink.
'@
}

if ($Target -ne "aarch64") {
    throw "Only -Target aarch64 is supported: ONNX Runtime ships no prebuilt for '$Target' (see the file header)."
}

$cli = @()
if (Get-Command cargo-tauri -ErrorAction SilentlyContinue) {
    $cli = @("cargo", "tauri")
}
else {
    $cli = @("npx", "--yes", "--package", "@tauri-apps/cli@2", "tauri")
}
$cliPrefix = @($cli | Select-Object -Skip 1)

Push-Location "backend"
try {
    rustup target add aarch64-linux-android | Out-Null

    # gen/android is committed, so this only runs on a checkout that was never
    # initialised (or after someone deleted it). Without it, "android build" stops with
    # "Android project not found" -- an error that does not say how to create one.
    $androidProject = Join-Path $root "backend\src-tauri\gen\android"
    if (-not (Test-Path -LiteralPath $androidProject)) {
        Write-Host "== gen/android is missing; initialising the Android project" -ForegroundColor Yellow
        $initArgs = $cliPrefix + @("android", "init", "--ci")
        & $cli[0] @initArgs
        if ($LASTEXITCODE) { throw "tauri android init failed (exit $LASTEXITCODE)" }
    }

    # ── C++ runtime ──────────────────────────────────────────────────────────
    #
    # [Why this copy exists] WORLD, Signalsmith, fdk-aac and libopus are all C++ and are
    # compiled by cc-rs, which links the **shared** C++ runtime on Android
    # (`cargo:rustc-link-lib=c++_shared`). The resulting libbackend_lib.so therefore carries
    #     NEEDED libc++_shared.so
    # and Android ships no such system library: an app that does not bundle it dies at
    # System.loadLibrary with 'dlopen failed: library "libc++_shared.so" not found'.
    # tauri-cli stages only libbackend_lib.so, so the copy has to happen here.
    #
    # [Why shared and not c++_static] Static linking also needs -lc++abi, which can only be
    # injected through per-target rustflags -- and those are baked into the fingerprint of
    # every dependency crate (see the note in .cargo/config.toml). Shipping the ~2 MB
    # runtime is the cheaper cost, and it is what Android's own documentation recommends for
    # an app with C++ code.
    $abiMap = @(
        @{ Jni = "arm64-v8a"; Triple = "aarch64-linux-android" },
        @{ Jni = "armeabi-v7a"; Triple = "armv7-linux-androideabi" },
        @{ Jni = "x86"; Triple = "i686-linux-android" },
        @{ Jni = "x86_64"; Triple = "x86_64-linux-android" }
    )
    $prebuilt = Split-Path -Parent $toolchain.NdkBin
    $jniRoot = Join-Path $androidProject "app\src\main\jniLibs"
    foreach ($abi in $abiMap) {
        $source = Join-Path $prebuilt ("sysroot\usr\lib\" + $abi.Triple + "\libc++_shared.so")
        if (-not (Test-Path -LiteralPath $source)) { continue }
        $abiDir = Join-Path $jniRoot $abi.Jni
        New-Item -ItemType Directory -Force -Path $abiDir | Out-Null
        Copy-Item -LiteralPath $source -Destination (Join-Path $abiDir "libc++_shared.so") -Force
        Write-Host "== bundled libc++_shared.so for $($abi.Jni)" -ForegroundColor DarkGray
    }

    $buildArgs = @(
        "android", "build",
        "--target", $Target,
        # `--features onnx` is the default; spelling it out keeps the build
        # reproducible if the default ever changes.
        "--features", "onnx"
    )
    if ($Bundle) { $buildArgs += "--aab" } else { $buildArgs += "--apk" }
    if ($DebugBuild) { $buildArgs += "--debug" }

    # ── Stale package output ─────────────────────────────────────────────────
    #
    # [Why this is here as well as in build.gradle.kts] The gradle script deletes
    # the package output before writing it, but that only runs when the packaging
    # task is *out of date*. An APK that already carries the packager's holes is a
    # perfectly valid task output, so gradle reports the task UP-TO-DATE and ships
    # the same bloated file again. Removing it first costs nothing and makes "the
    # APK on disk is the APK this run produced" true.
    #
    # [Why only the package outputs] Everything else under app/build (dex,
    # resources.arsc, merged jniLibs) is still valid; deleting the whole build
    # directory would turn an 8 s package into a several-minute rebuild.
    foreach ($dir in @("apk", "bundle")) {
        $stale = Join-Path $androidProject "app\build\outputs\$dir"
        if (Test-Path -LiteralPath $stale) { Remove-Item -LiteralPath $stale -Recurse -Force }
    }

    Write-Host ("== " + (($cli + $buildArgs) -join " ")) -ForegroundColor Cyan
    & $cli[0] @($cliPrefix + $buildArgs)
    if ($LASTEXITCODE) { throw "tauri android build failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

# -- Artefact integrity -------------------------------------------------------
#
# [Why check the .so files] ort-sys stages its binaries as *symlinks* into
# target/<triple>/<profile>/. A packaging step that copies the link instead of its
# target produces a 0-byte library inside the APK -- and the app then dies at
# System.loadLibrary with "not an ELF file", long after the build said OK.
$jniLibs = Join-Path $root "backend\src-tauri\gen\android\app\src\main\jniLibs"
if (Test-Path -LiteralPath $jniLibs) {
    $broken = @()
    Get-ChildItem -LiteralPath $jniLibs -Recurse -Filter "*.so" | ForEach-Object {
        $bytes = [IO.File]::ReadAllBytes($_.FullName)
        $isElf = ($bytes.Length -ge 4) -and ($bytes[0] -eq 0x7F) -and ($bytes[1] -eq 0x45) -and
                 ($bytes[2] -eq 0x4C) -and ($bytes[3] -eq 0x46)
        if (-not $isElf) { $broken += $_.FullName }
    }
    if ($broken.Count -gt 0) {
        throw ("Not an ELF shared object (staged symlink copied as a file?):" + [Environment]::NewLine + ($broken -join [Environment]::NewLine))
    }
    Write-Host "== jniLibs .so files are real ELF objects" -ForegroundColor Green
}
else {
    Write-Host "== $jniLibs not found; skipping artefact integrity check" -ForegroundColor Yellow
}

# -- Package size sanity ------------------------------------------------------
#
# [Why] See the comment in gen/android/app/build.gradle.kts: the packager updates
# the archive in place and can leave hundreds of megabytes of dead bytes between
# entries *without* invalidating it. Nothing downstream would notice -- the APK
# installs and a store would happily accept a file twice the size it should be.
# Comparing the file length with the sum of its compressed entries catches that
# (and any future cause) with one subtraction.
$outputs = Join-Path $root "backend\src-tauri\gen\android\app\build\outputs"
Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
$packages = @()
foreach ($pattern in @("*.apk", "*.aab")) {
    $packages += @(Get-ChildItem -LiteralPath $outputs -Recurse -Filter $pattern -File -ErrorAction SilentlyContinue)
}
foreach ($package in $packages) {
    $zip = [IO.Compression.ZipFile]::OpenRead($package.FullName)
    try {
        $payload = ($zip.Entries | Measure-Object -Property CompressedLength -Sum).Sum
    }
    finally { $zip.Dispose() }
    if (-not $payload) { $payload = 0 }
    $slack = $package.Length - $payload
    Write-Host ("== {0}: {1:N1} MiB of entries, {2:N0} B of slack" -f $package.Name, ($payload / 1MB), $slack) -ForegroundColor Green
    if ($slack -gt 16MB) {
        throw @"
$($package.FullName)
  is $([math]::Round($package.Length / 1MB, 1)) MiB but only carries $([math]::Round($payload / 1MB, 1)) MiB of entries.

  That gap is the Android packager reusing an older archive instead of writing a
  fresh one (see the comment in gen/android/app/build.gradle.kts). Delete
  backend\src-tauri\gen\android\app\build and build again; do not ship this file.
"@
    }
}

