# Run this project's Android build on an emulator on this machine.
#
#   .\scripts\dev\android-emulator.ps1 -Action start      # boot the AVD (headless)
#   .\scripts\dev\android-emulator.ps1 -Action build      # x86_64 debug APK
#   .\scripts\dev\android-emulator.ps1 -Action install    # push + start it
#   .\scripts\dev\android-emulator.ps1 -Action check      # assertions over the DevTools protocol
#   .\scripts\dev\android-emulator.ps1 -Action logs
#   .\scripts\dev\android-emulator.ps1 -Action screenshot -Out shot.png
#   .\scripts\dev\android-emulator.ps1 -Action stop
#
# [Why this exists] An APK can only be judged on Android: the safe-area numbers, the
# system-picker sheets, the WebView's own quirks. Without a device every change in this
# area is a guess, and "it compiles" has already shipped defects that only showed up on a
# phone. The emulator is the cheapest honest answer -- but it needs three things that are
# not obvious, and this script owns all three:
#
#   1. **x86_64 only.** The emulator on an x86_64 host ships a single qemu target
#      (emulator/qemu/windows-x86_64), so an arm64 image cannot be booted even though the
#      SDK will happily download one. The app must therefore be built for x86_64, which is
#      why Cargo.toml links ort-tract for that target -- see the note there.
#   2. **Acceleration.** WHPX (Windows Hypervisor Platform) is required. Without it the
#      image runs in pure emulation and takes minutes per boot, if it boots at all.
#   3. **The dev server.** A debug APK is a dev build: every request is proxied to the dev
#      URL baked in at compile time (http://localhost:5173). "localhost" on the emulator is
#      the emulator, so the port has to be forwarded with adb reverse before the app starts,
#      or the window stays blank.
#
# The app is inspected through scripts/dev/android-webview.mjs, which speaks the Chrome
# DevTools protocol to the WebView -- console, DOM, screenshots, invokes to Rust.
[CmdletBinding()]
param(
    [ValidateSet("start", "stop", "status", "build", "install", "launch", "check", "logs", "screenshot", "devtools")]
    [string]$Action = "status",
    [string]$Avd = "hfs-debug",
    [int]$Api = 35,
    [string]$Tag = "google_apis",
    [string]$Out = "",
    [int]$Lines = 120,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $root

. (Join-Path $PSScriptRoot "..\env\android-env.ps1")
$toolchain = Set-AndroidEnvironment

$sdk = $toolchain.SdkRoot
$adb = Join-Path $sdk "platform-tools\adb.exe"
$emulator = Join-Path $sdk "emulator\emulator.exe"
$avdmanager = Join-Path $sdk "cmdline-tools\latest\bin\avdmanager.bat"
$sdkmanager = Join-Path $sdk "cmdline-tools\latest\bin\sdkmanager.bat"
$devtoolsPort = 9222
$packageName = "com.arounder.hifishifter"
function Get-DeviceSerial {
    $attached = @(& $adb devices | Where-Object { $_ -match "\tdevice$" })
    if ($attached.Count -eq 0) { return $null }
    return ($attached[0] -split "\t")[0]
}

function Wait-ForBoot {
    param([int]$TimeoutSeconds = 300)

    & $adb wait-for-device | Out-Null
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $booted = (& $adb shell getprop sys.boot_completed 2>$null) -join ""
        if ($booted.Trim() -eq "1") { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Start-Emulator {
    $serial = Get-DeviceSerial
    if ($serial) {
        Write-Host "== an emulator is already running: $serial" -ForegroundColor Green
        return
    }

    $image = "system-images;android-$Api;$Tag;x86_64"
    $imageDir = Join-Path $sdk "system-images\android-$Api\$Tag\x86_64"

    if (-not (Test-Path -LiteralPath $emulator)) {
        Write-Host "== installing the emulator package" -ForegroundColor Yellow
        & $sdkmanager --sdk_root=$sdk "emulator" | Out-Null
    }
    if (-not (Test-Path -LiteralPath $imageDir)) {
        Write-Host "== installing $image (about 1.5 GB)" -ForegroundColor Yellow
        ( @("y") * 20 ) | & $sdkmanager --sdk_root=$sdk $image | Out-Null
    }

    $avds = @(& $emulator -list-avds 2>$null)
    if ($avds -notcontains $Avd) {
        Write-Host "== creating AVD $Avd ($image)" -ForegroundColor Yellow
        ( @("no") * 5 ) | & $avdmanager create avd -n $Avd -k $image -d pixel_6 --force | Out-Null
    }

    $accel = (& $emulator -accel-check 2>&1) -join " "
    if ($accel -notmatch "is installed and usable") {
        throw "The emulator cannot use hardware acceleration: $accel"
    }

    Write-Host "== booting $Avd (headless)" -ForegroundColor Cyan
    # -no-window: nothing to show on a box that may not have a desktop session; the UI is
    # reachable through screenshots and the DevTools protocol anyway.
    # -no-snapshot: a stale snapshot would restore the *previous* build of the app.
    # -gpu swiftshader_indirect: the WebGL canvas needs a GL implementation, and the host
    # GPU is not passed through when the emulator runs headless.
    $arguments = @(
        "-avd", $Avd,
        "-no-window", "-no-audio", "-no-boot-anim", "-no-snapshot",
        "-gpu", "swiftshader_indirect",
        "-memory", "3072", "-cores", "4"
    )
    Start-Process -FilePath $emulator -ArgumentList $arguments -WindowStyle Hidden | Out-Null

    if (-not (Wait-ForBoot)) { throw "The emulator did not finish booting within 5 minutes." }
    Write-Host "== booted" -ForegroundColor Green
}

function Stop-Emulator {
    $serial = Get-DeviceSerial
    if (-not $serial) { Write-Host "== no emulator running" -ForegroundColor Yellow; return }
    & $adb -s $serial emu kill | Out-Null
    Write-Host "== stopped $serial" -ForegroundColor Green
}

function Invoke-Build {
    # The emulator runs x86_64, so the app is built for x86_64 and debug: the frontend is
    # served by vite, which is what makes console errors and hot reload available.
    & (Join-Path $PSScriptRoot "..\build\android.ps1") -DebugBuild -Target x86_64
    if ($LASTEXITCODE) { throw "android.ps1 failed (exit $LASTEXITCODE)" }
}

function Install-App {
    $serial = Get-DeviceSerial
    if (-not $serial) { throw "No emulator running; use -Action start." }

    # [Why the whole outputs tree] The "universal" flavor is what the build script packages
    # (it stages a single ABI's .so, so the universal APK is an x86_64 APK), while a plain
    # gradlew run may leave an APK under x86_64/. Newest wins; the flavor name is not a
    # contract worth depending on.
    $apkDir = Join-Path $root "backend\src-tauri\gen\android\app\build\outputs\apk"
    $apk = Get-ChildItem $apkDir -Recurse -Filter "*.apk" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $apk) { throw "No x86_64 APK found; run -Action build first." }

    Write-Host ("== installing {0} ({1:N0} MB)" -f $apk.Name, ($apk.Length / 1MB)) -ForegroundColor Cyan
    & $adb -s $serial install -r $apk.FullName
    if ($LASTEXITCODE) { throw "adb install failed (exit $LASTEXITCODE)" }
}

function Start-App {
    $serial = Get-DeviceSerial
    if (-not $serial) { throw "No emulator running; use -Action start." }

    # [Why the forward] The debug build proxies every WebView request to
    # http://localhost:5173. Without this the page is blank and the only clue is a line in
    # logcat saying the request failed.
    & $adb -s $serial reverse "tcp:5173" "tcp:5173" | Out-Null
    & $adb -s $serial shell am force-stop $packageName | Out-Null
    & $adb -s $serial shell am start -n "$packageName/.MainActivity" | Out-Null
    Write-Host "== launched $packageName (device port 5173 forwarded to the dev server)" -ForegroundColor Green
    Start-Sleep -Seconds 6
    Set-DevToolsPort
}

function Set-DevToolsPort {
    $serial = Get-DeviceSerial
    if (-not $serial) { throw "No emulator running; use -Action start." }
    $appPid = ((& $adb -s $serial shell pidof $packageName 2>$null) -join "").Trim()
    if (-not $appPid) { throw "The app is not running; use -Action launch." }
    # The socket name embeds the pid, so the forward has to be redone on every launch.
    & $adb -s $serial forward "tcp:$devtoolsPort" "localabstract:webview_devtools_remote_$appPid" | Out-Null
    Write-Host "== devtools forwarded to 127.0.0.1:$devtoolsPort" -ForegroundColor Green
}

function Show-Logs {
    $serial = Get-DeviceSerial
    if (-not $serial) { throw "No emulator running; use -Action start." }
    if ($Full) { & $adb -s $serial logcat -d -t $Lines; return }
    $filter = "hifishifter|RustStdout|AndroidRuntime|libbackend|dlopen|FATAL"
    & $adb -s $serial logcat -d -t 4000 | Select-String -Pattern $filter |
        Select-Object -Last $Lines | ForEach-Object { $_.Line }
}

function Save-Screenshot {
    $serial = Get-DeviceSerial
    if (-not $serial) { throw "No emulator running; use -Action start." }
    $path = if ($Out) { $Out } else { Join-Path $env:TEMP "hifishifter-emulator.png" }
    # [Why not "exec-out ... > file"] PowerShell's redirection is text-oriented: it rewrites
    # newlines in the byte stream, and the PNG arrives truncated ("unexpected end of file"
    # from any decoder). Capturing on the device and pulling the file keeps it binary-safe.
    $remote = "/data/local/tmp/hfs-screenshot.png"
    & $adb -s $serial shell screencap -p $remote | Out-Null
    & $adb -s $serial pull $remote $path | Out-Null
    & $adb -s $serial shell rm -f $remote | Out-Null
    Write-Host ("== screenshot: {0} ({1:N0} bytes)" -f $path, (Get-Item $path).Length) -ForegroundColor Green
}

function Invoke-Check {
    Set-DevToolsPort
    & node (Join-Path $PSScriptRoot "android-webview.mjs") --check
    if ($LASTEXITCODE) { throw "The WebView check failed (exit $LASTEXITCODE)." }
}
switch ($Action) {
    "start" { Start-Emulator }
    "stop" { Stop-Emulator }
    "status" {
        $serial = Get-DeviceSerial
        if ($serial) {
            $booted = ((& $adb shell getprop sys.boot_completed 2>$null) -join "").Trim()
            $appPid = ((& $adb -s $serial shell pidof $packageName 2>$null) -join "").Trim()
            Write-Host "== device:  $serial (boot_completed=$booted)" -ForegroundColor Green
            Write-Host "== app pid: $(if ($appPid) { $appPid } else { "not running" })"
        }
        else {
            Write-Host "== no emulator running" -ForegroundColor Yellow
        }
    }
    "build" { Invoke-Build }
    "install" { Install-App }
    "launch" {
        Start-App
        Start-Sleep -Seconds 6
        Invoke-Check
    }
    "check" { Invoke-Check }
    "logs" { Show-Logs }
    "screenshot" { Save-Screenshot }
    "devtools" { Set-DevToolsPort }
}