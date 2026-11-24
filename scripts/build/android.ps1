# Build the Android package.
#
#   .\scripts\build\android.ps1                   # release APK (standalone)
#   .\scripts\build\android.ps1 -Bundle           # AAB (Play)
#   .\scripts\build\android.ps1 -DebugBuild       # debug APK (dev server required)
#   .\scripts\build\android.ps1 -DebugBuild -Run  # ...and install + launch it
#
# [Why the switch is called -DebugBuild and not -Debug] `[CmdletBinding()]` already gives
# the script a common `-Debug` parameter (it sets `$DebugPreference`). Declaring our own
# `-Debug` therefore fails to load the script at all ("parameter Debug defined multiple
# times"); and if the common one were used instead, the build would silently produce a
# release APK. -DebugBuild is the explicit, unshadowed spelling.
#
# [Why a debug APK is not a standalone app] Tauri compiles `dev` into the binary from the
# cargo profile, and on mobile `dev` means "proxy every request to the dev server" (see
# tauri/src/protocol/tauri.rs, `#[cfg(all(dev, mobile))]`). A debug APK therefore shows
#     Failed to request http://localhost:5173/: error sending request for url (...)
# until something answers on that port *from the device*. Two things make it answer:
# `adb reverse tcp:5173 tcp:5173` (which -Run does) or a dev URL rewritten to this
# machine's LAN address (which `tauri android dev` does). A release APK embeds the built
# frontend instead and needs neither -- that is the one to hand to a phone.
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
    [string]$Target = "aarch64",
    # Install the package on the attached device and launch it.
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $root

. (Join-Path $PSScriptRoot "..\env\android-env.ps1")
$toolchain = Set-AndroidEnvironment
Write-Host "== Android SDK: $($toolchain.SdkRoot)" -ForegroundColor Cyan
Write-Host "== Android NDK: $($toolchain.NdkRoot)" -ForegroundColor Cyan
Write-Host "== JDK:         $($toolchain.JdkRoot)" -ForegroundColor Cyan

$androidProject = Join-Path $root "backend\src-tauri\gen\android"
$profile = if ($DebugBuild) { "Debug" } else { "Release" }
$cargoProfile = if ($DebugBuild) { "debug" } else { "release" }

# ── Windows symlink privilege ─────────────────────────────────────────────────
#
# [Why this probe exists] tauri-cli stages the built library into the generated project
# with a **symbolic link** (cargo-mobile2's `os::windows::ln::force_symlink`, which reports
# ERROR_PRIVILEGE_NOT_HELD as "Creation symbolic link is not allowed for this system"). A
# non-elevated Windows token only carries SeCreateSymbolicLinkPrivilege when Developer Mode
# is on, so on a stock machine the build compiles for minutes and then stops at a message
# that reads like a toolchain problem.
#
# [Why this is no longer fatal] The link exists only to put libbackend_lib.so where gradle
# looks for it; a plain copy is indistinguishable to the packager, and the resource copy
# the CLI does afterwards is a handful of Copy-Item calls. The CLI is still the thing that
# generates the gradle glue (tauri.settings.gradle, tauri.build.gradle.kts,
# tauri.properties) and that compiles the library, so it is always run first -- only the
# last two steps are redone by hand when it trips over the privilege.
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
    Write-Host "== this account cannot create symbolic links; the CLI's staging step will fail and be redone by hand" -ForegroundColor Yellow
}

# ── Staging that does not need a symlink ─────────────────────────────────────
#
# [Why the resource copy matters] platform/resources.rs materialises assets/models/** out
# of the installed APK on first launch, because resource_dir() on Android is the
# "asset://localhost/" pseudo-URL and not a path. If the resources are not in the APK the
# app starts with "no bundled models found" and every model-backed feature is dead.
#
# [Why the target directory is replaced and not merged] tauri-cli never removes resources
# that were dropped from the manifest, so a stale models/ survives a manifest change and
# the APK keeps shipping a model nobody asked for. Replacing the directory makes the APK
# match bundle.resources exactly.
function Sync-BundleResources {
    param([string]$TauriDir, [string]$AssetsRoot)

    $configPath = Join-Path $TauriDir 'tauri.conf.json'
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $resources = $config.bundle.resources
    if (-not $resources) { return 0 }

    $copied = 0
    foreach ($entry in $resources.PSObject.Properties) {
        $source = [IO.Path]::GetFullPath((Join-Path $TauriDir $entry.Name))
        $target = [IO.Path]::GetFullPath((Join-Path $AssetsRoot $entry.Value))
        if (Test-Path -LiteralPath $source -PathType Container) {
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
        }
        else {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $source -Destination $target -Force
        }
        $copied++
    }
    return $copied
}

function Invoke-SelfStagedPackage {
    param([string]$Profile, [string]$CargoProfile)

    $built = Join-Path $root "backend\target\aarch64-linux-android\$CargoProfile\libbackend_lib.so"
    if (-not (Test-Path -LiteralPath $built)) {
        throw "The Rust library was not built: $built is missing. Fix the cargo error above first."
    }

    $jniDir = Join-Path $androidProject "app\src\main\jniLibs\arm64-v8a"
    New-Item -ItemType Directory -Force -Path $jniDir | Out-Null
    Copy-Item -LiteralPath $built -Destination (Join-Path $jniDir "libbackend_lib.so") -Force
    Write-Host ("== staged libbackend_lib.so ({0:N0} bytes)" -f (Get-Item $built).Length) -ForegroundColor DarkGray

    $assets = Join-Path $androidProject "app\src\main\assets"
    $copied = Sync-BundleResources -TauriDir (Join-Path $root "backend\src-tauri") -AssetsRoot $assets
    Write-Host "== synced $copied bundle resource(s) into app/src/main/assets" -ForegroundColor DarkGray

    Push-Location $androidProject
    try {
        & ".\gradlew.bat" ":app:assembleUniversal$Profile" "-x" "rustBuildUniversal$Profile" --console=plain
        if ($LASTEXITCODE) { throw "gradle assembleUniversal$Profile failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }
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

$buildLog = Join-Path ([IO.Path]::GetTempPath()) 'hfs-android-build.log'
$buildExit = 0

Push-Location "backend"
try {
    rustup target add aarch64-linux-android | Out-Null

    # gen/android is committed, so this only runs on a checkout that was never
    # initialised (or after someone deleted it). Without it, "android build" stops with
    # "Android project not found" -- an error that does not say how to create one.
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
    # (cargo:rustc-link-lib=c++_shared). The resulting libbackend_lib.so therefore carries
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
    # [Why only the ABI we actually build] jniLibs is exactly what the packager copies, and
    # the "universal" flavor accepts whatever ABI directories it finds. Staging the C++
    # runtime for x86 as well made the APK advertise
    #     native-code: 'arm64-v8a' 'x86' 'x86_64'
    # while libbackend_lib.so exists only for arm64 -- on an x86 device the installer would
    # pick the x86 directory and System.loadLibrary would fail with "not an ELF file".
    # Ship one ABI: the one ONNX Runtime has a prebuilt for (see the file header).
    $abiMap = @{
        "aarch64" = @{ Jni = "arm64-v8a"; Triple = "aarch64-linux-android" }
    }
    $abi = $abiMap[$Target]
    if (-not $abi) { throw "No ABI mapping for target '$Target'." }
    $prebuilt = Split-Path -Parent $toolchain.NdkBin
    $jniRoot = Join-Path $androidProject "app\src\main\jniLibs"

    # Directories left behind by an older build would keep the foreign ABI in the APK.
    if (Test-Path -LiteralPath $jniRoot) {
        Get-ChildItem -LiteralPath $jniRoot -Directory |
            Where-Object { $_.Name -ne $abi.Jni } |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    }

    $runtime = Join-Path $prebuilt ("sysroot\usr\lib\" + $abi.Triple + "\libc++_shared.so")
    if (-not (Test-Path -LiteralPath $runtime)) { throw "libc++_shared.so not found at $runtime" }
    $abiDir = Join-Path $jniRoot $abi.Jni
    New-Item -ItemType Directory -Force -Path $abiDir | Out-Null
    Copy-Item -LiteralPath $runtime -Destination (Join-Path $abiDir "libc++_shared.so") -Force
    Write-Host "== bundled libc++_shared.so for $($abi.Jni)" -ForegroundColor DarkGray

    $buildArgs = @(
        "android", "build",
        "--target", $Target,
        # "--features onnx" is the default; spelling it out keeps the build
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
    # [Why the preference is relaxed around this call] `2>&1` on a native command turns
    # every stderr line into an ErrorRecord under Windows PowerShell, and this script runs
    # with `$ErrorActionPreference = "Stop"` -- so the first cargo warning ("Info Using
    # installed NDK: ...") aborted the build before it started. The exit code still arrives
    # in `$LASTEXITCODE`, which is what actually matters here.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $cli[0] @($cliPrefix + $buildArgs) 2>&1 | Tee-Object -FilePath $buildLog
        $buildExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}
finally {
    Pop-Location
}

if ($buildExit) {
    $stagingFailed = (Test-Path -LiteralPath $buildLog) -and
        (Select-String -LiteralPath $buildLog -Pattern 'symbolic link' -Quiet)
    if (-not $stagingFailed) { throw "tauri android build failed (exit $buildExit); see $buildLog" }

    if ($Bundle) {
        throw @'
-An AAB needs tauri-cli's own packaging (it writes the bundle's asset layout), and that
step is the one that needs a symbolic link. Enable Developer Mode
(Settings -> System -> For developers) or run this script from an elevated PowerShell,
then run with -Bundle again.
'@
    }

    Write-Host "== tauri-cli could not create the staging symlink; packaging from the built library instead" -ForegroundColor Yellow
    Invoke-SelfStagedPackage -Profile $profile -CargoProfile $cargoProfile
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

# -- Run on the device --------------------------------------------------------
#
# [Why a reverse port forward] A debug APK is a dev build, so every request it makes is
# proxied to the dev URL baked in at compile time -- http://localhost:5173 for anything
# built outside "tauri android dev". "localhost" on the phone is the phone, so without the
# forward the first request fails and the window shows
#     Failed to request http://localhost:5173/: error sending request for url (...)
# The forward makes the phone's port 5173 reach this machine's dev server. A release APK
# needs neither the forward nor a dev server: it carries the built frontend in the binary.
if ($Run) {
    $adb = Join-Path $toolchain.SdkRoot "platform-tools\adb.exe"
    if (-not (Test-Path -LiteralPath $adb)) { throw "adb not found at $adb" }
    $attached = @(& $adb devices | Where-Object { $_ -match "`tdevice$" })
    if ($attached.Count -eq 0) {
        throw "No device is attached. Connect one (or start an emulator), check 'adb devices', then run again."
    }
    $serial = ($attached[0] -split "`t")[0]
    Write-Host "== device: $serial" -ForegroundColor Cyan

    $apk = Get-ChildItem -LiteralPath $outputs -Recurse -Filter "*.apk" -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $apk) { throw "No APK was produced under $outputs" }

    if ($apk.Name -like '*-unsigned.apk') {
        # [Why the debug keystore] A release APK is unsigned until a store signs it, and an
        # unsigned package cannot be installed at all. For side-loading the debug key that
        # already exists on every machine is the right one: it is disposable, it is not a
        # distribution identity, and using it keeps a real signing config (key.properties)
        # out of the local loop.
        $keystore = Join-Path $env:USERPROFILE ".android\debug.keystore"
        if (-not (Test-Path -LiteralPath $keystore)) { throw "No debug keystore at $keystore" }
        $buildTools = Get-ChildItem (Join-Path $toolchain.SdkRoot "build-tools") -Directory |
            Sort-Object Name -Descending | Select-Object -First 1
        $apksigner = Join-Path $buildTools.FullName "apksigner.bat"
        $signedPath = $apk.FullName -replace '-unsigned\.apk$', '-signed.apk'
        Write-Host "== signing with the debug keystore (side-load only)" -ForegroundColor DarkGray
        & $apksigner sign --ks $keystore --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey --out $signedPath $apk.FullName
        if ($LASTEXITCODE) { throw "apksigner failed (exit $LASTEXITCODE)" }
        $apk = Get-Item -LiteralPath $signedPath
    }

    if ($DebugBuild) {
        # [Why the dev server is started here] Without it the APK installs and then shows the
        # proxy error above; that is the single most confusing failure in this flow.
        $devUrl = "http://127.0.0.1:5173/"
        $up = $false
        try { $null = Invoke-WebRequest -Uri $devUrl -TimeoutSec 2 -UseBasicParsing; $up = $true } catch { $up = $false }

        if (-not $up) {
            Write-Host "== starting the dev server in a new window" -ForegroundColor Cyan
            $frontend = Join-Path $root "frontend"
            # [Why the shell is probed] `pwsh` only exists where PowerShell 7 is installed;
            # a stock Windows box still has `powershell`, and npm does not care which one
            # hosts it. Hardcoding either name makes -Run fail on half the machines.
            $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }
            Start-Process -FilePath $shell -ArgumentList @(
                "-NoExit", "-Command",
                "Set-Location '$frontend'; npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"
            ) | Out-Null
            for ($i = 0; $i -lt 60 -and -not $up; $i++) {
                Start-Sleep -Milliseconds 500
                try { $null = Invoke-WebRequest -Uri $devUrl -TimeoutSec 2 -UseBasicParsing; $up = $true } catch { $up = $false }
            }
        }
        if (-not $up) { throw "The dev server did not come up at $devUrl; start it yourself and run again." }

        & $adb -s $serial reverse "tcp:5173" "tcp:5173"
        if ($LASTEXITCODE) { throw "adb reverse failed (exit $LASTEXITCODE)" }
        Write-Host "== device localhost:5173 -> this machine's dev server" -ForegroundColor DarkGray
    }

    & $adb -s $serial install -r $apk.FullName
    if ($LASTEXITCODE) { throw "adb install failed (exit $LASTEXITCODE)" }
    & $adb -s $serial shell am start -n "com.arounder.hifishifter/.MainActivity"
    if ($LASTEXITCODE) { throw "am start failed (exit $LASTEXITCODE)" }
    Write-Host "== launched" -ForegroundColor Green
}
