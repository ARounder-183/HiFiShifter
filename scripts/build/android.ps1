# Build the Android package.
#
#   .\scripts\build\android.ps1            # APK (side-load)
#   .\scripts\build\android.ps1 -Bundle    # AAB (Play)
#   .\scripts\build\android.ps1 -Debug     # debug APK, no signing
#
# [Why an explicit ABI and not "let tauri decide"] `tauri android build` compiles
# four ABIs by default (arm64-v8a, armeabi-v7a, x86, x86_64). ONNX Runtime ships a
# prebuilt distribution only for aarch64-linux-android -- the other three fail at
# link time with an error that names a crate nobody in this project has heard of.
# Restricting to arm64-v8a is not an optimisation, it is the only combination that
# links.
#
# [Why the NDK is discovered instead of hardcoded] The repository may live on any
# drive and the NDK under any SDK root; same reasoning as scripts/env/sdk-env.ps1.
# ANDROID_NDK_HOME / NDK_HOME are honoured first, then the newest side-by-side
# install under ANDROID_SDK_ROOT.
[CmdletBinding()]
param(
    [switch]$Bundle,
    [switch]$Debug,
    [string]$Target = "aarch64"
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $root

function Resolve-AndroidNdk {
    foreach ($name in @("ANDROID_NDK_HOME", "NDK_HOME")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value -and (Test-Path -LiteralPath $value)) { return $value }
    }
    $sdk = [Environment]::GetEnvironmentVariable("ANDROID_SDK_ROOT")
    if (-not $sdk) { $sdk = [Environment]::GetEnvironmentVariable("ANDROID_HOME") }
    if (-not $sdk) { throw "ANDROID_SDK_ROOT / ANDROID_HOME is not set; install the Android SDK first." }
    $ndkRoot = Join-Path $sdk "ndk"
    if (-not (Test-Path -LiteralPath $ndkRoot)) {
        throw "No NDK under $ndkRoot. Install one with: sdkmanager ndk;27.0.12077973"
    }
    # Newest side-by-side install wins. The directory names are versions, so a
    # string sort is wrong ("9.x" > "27.x"); sort by parsed version instead.
    $candidate = Get-ChildItem -LiteralPath $ndkRoot -Directory |
        Sort-Object { [version]($_.Name -replace "[^0-9.].*$", "") } -Descending |
        Select-Object -First 1
    if (-not $candidate) { throw "No NDK install found under $ndkRoot" }
    return $candidate.FullName
}

$ndk = Resolve-AndroidNdk
Write-Host "== Android NDK: $ndk" -ForegroundColor Cyan

# tauri-cli reads these to pick the NDK toolchain. Setting them here (rather than
# asking the user to export them) keeps "one command" true.
$env:ANDROID_NDK_HOME = $ndk
$env:NDK_HOME = $ndk

if ($Target -ne "aarch64") {
    throw "Only -Target aarch64 is supported: ONNX Runtime ships no prebuilt for $Target (see the file header)."
}
$rustTarget = "aarch64-linux-android"

Push-Location "backend"
try {
    rustup target add $rustTarget | Out-Null

    $arguments = @(
        "tauri", "android", "build",
        "--target", $rustTarget,
        # `--features onnx` is the default; spelling it out keeps the build
        # reproducible if the default ever changes.
        "--features", "onnx"
    )
    if ($Bundle) { $arguments += "--aab" } else { $arguments += "--apk" }
    if ($Debug) { $arguments += "--debug" }

    Write-Host "== npx $($arguments -join [char]32)" -ForegroundColor Cyan
    & npx @arguments
    if ($LASTEXITCODE) { throw "tauri android build failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

# ── Artefact integrity ─────────────────────────────────────────────────────
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
