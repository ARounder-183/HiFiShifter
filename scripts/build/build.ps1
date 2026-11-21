# Product build: frontend → App → VST3 bundle → installer staging → merged installer / portable zip.
#
# There is exactly one output location: dist\. No intermediate delivery directories, no build
# manifests, no source fingerprints — an artifact's identity is decided by the git commit it came
# from; there is no need to also write a JSON file to prove it.
#
#   .\scripts\build\build.ps1                     # App + VST3 + merged installer + portable zip
#   .\scripts\build\build.ps1 -Target App         # App executable only (no installer)
#   .\scripts\build\build.ps1 -Target Plugin      # VST3 bundle only
#   .\scripts\build\build.ps1 -Target Installer   # re-run the installer only (reuses dist\_stage)
#   .\scripts\build\build.ps1 -SkipFrontend       # frontend\dist is already up to date
#   .\scripts\build\build.ps1 -SkipVerify         # CI build-only: skip plugin black-box self-checks
#   .\scripts\build\build.ps1 -Configuration Debug   # debug artifacts, no packaging
#
# [Why Windows has a single installer] The standalone App and the VST3 plugin serve the same
# users, and both use the same 150 MB of models. Splitting into two installers has only two
# outcomes: the user pays for identical weights twice on disk, or the two packages each write
# half the logic to guess whether the other is installed. Merged, the models are written once
# and uninstall has a single entry point.
#
# Requires: MSVC (cl.exe), cargo, npm, NSIS. See scripts/setup/install_deps_windows.ps1.
[CmdletBinding()]
param(
    [ValidateSet('All', 'App', 'Plugin', 'Clap', 'Installer')][string]$Target = 'All',
    [switch]$SkipFrontend,
    # CI build-only: skip the plugin black-box self-checks (the CLAP one runs cargo test ABI oracle).
    [switch]$SkipVerify,
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    # Offline / controlled environments can point at an already downloaded WebView2 bootstrapper (see Get-WebView2Bootstrapper).
    [string]$WebView2Bootstrapper = ''
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

. .\scripts\env\msvc-env.ps1
. .\scripts\env\sdk-env.ps1
Set-SdkEnvironment

# [Why $root is computed again] The two env scripts above are **dot-sourced**: their top-level
# variables land in this scope. Real incident: msvc-env.ps1 had a `foreach ($root in ...)` that
# overwrote this `$root` (the repo root) with the Visual Studio install directory, so
# `Join-Path $root 'backend\...'` pointed at
# `C:\Program Files\Microsoft Visual Studio\2022\Community\backend\...`,
# and the build failed at "models.json not found". The env scripts have since become functions
# (no more leaking variables); computing $root again here is the second gate — one line of cost,
# in exchange for this class of regression never again writing artifacts outside the repo.
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Assert-Ok($what) {
    if ($LASTEXITCODE) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# [Why compare byte-for-byte once more after assembly] What is "copied into the payload" must be
# exactly the bytes that were "just built". This check does not catch compile errors; it catches
# **ordering bugs**: if any other cargo call touches the same target directory after assembly,
# the payload silently swaps in a different engine build while the build looks fine and the
# artifact still loads — only a post-hoc hash comparison reveals the mismatch, when it is no
# longer traceable which build it came from.
function Assert-SameBytes([string]$Expected, [string]$Actual) {
    if (!(Test-Path -LiteralPath $Expected)) { throw "Missing build artifact: $Expected" }
    if (!(Test-Path -LiteralPath $Actual)) { throw "Missing assembled payload: $Actual" }
    $wanted = (Get-FileHash -LiteralPath $Expected -Algorithm SHA256).Hash
    $shipped = (Get-FileHash -LiteralPath $Actual -Algorithm SHA256).Hash
    if ($wanted -ne $shipped) {
        throw ("Payload does not match the build artifact: {0} (build {1}) vs {2} (payload {3});" +
            " some other build rewrote the target directory after assembly") -f $Expected, $wanted, $Actual, $shipped
    }
}

# External tools the build needs: when missing, give an actionable error instead of letting
# PowerShell throw "The term 'xxx' is not recognized" (which gives a fresh clone no direction).
foreach ($tool in 'cargo', 'npm') {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Cannot find $tool. Run scripts\setup\install_deps_windows.ps1 first to install the toolchain, or add its directory to PATH."
    }
}

# [Why not just call `python`] Three common shapes exist on Windows: `python.exe` on PATH, only
# the `py` launcher, and the Microsoft Store `python.exe` alias (present, but running it opens
# the Store). The test is "it actually runs and is 3.x", not "a file named python is on PATH".
function Resolve-Python {
    $candidates = @(
        @{ Exe = 'python'; Prefix = @() },
        @{ Exe = 'py'; Prefix = @('-3') },
        @{ Exe = 'python3'; Prefix = @() }
    )
    foreach ($candidate in $candidates) {
        $found = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (!$found) { continue }
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $major = & $found.Source @($candidate.Prefix) -c 'import sys; print(sys.version_info[0])' 2>$null
        }
        catch { $major = $null }
        finally { $ErrorActionPreference = $previous }
        if ("$major".Trim() -eq '3') {
            return @{ Exe = $found.Source; Prefix = @($candidate.Prefix) }
        }
    }
    throw 'No usable Python 3 found: install it, make sure python.exe (or the py launcher) is on PATH; see scripts\setup\install_deps_windows.ps1'
}

function Reset-Directory([string]$path) {
    $resolved = [IO.Path]::GetFullPath($path)
    $outputRoot = [IO.Path]::GetFullPath((Join-Path $root 'dist')).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($outputRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Unexpected build cleanup path: $resolved" }
    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $resolved | Out-Null
}

# [Which DLLs are not part of the standalone App's runtime payload] `backend\target\<profile>`
# holds artifacts for three formats at once: the standalone App, the VST3 engine
# (`hifishifter_vst3.dll`) and the CLAP module (`hifishifter_clap.dll`), plus the
# `backend_lib.dll` that the App crate builds itself. They share one directory with the generic
# runtime DLLs, so a blanket `Get-ChildItem *.dll` would sweep these modules — neither runtime
# nor App-required — into the installer and the portable zip. Plugins ship only through two
# dedicated paths: the VST3 bundle (see Write-Vst3Stage) and the CLAP directory (see
# Write-ClapStage); the App payload has no overlap with them.
$pluginModuleDlls = @(
    'hifishifter_vst3.dll',
    'hifishifter_clap.dll',
    'backend_lib.dll'
)

function Copy-AppRuntimeDlls([string]$Destination) {
    # Zero-byte DLLs are arm64 placeholders (vslib is x86_64-only); they do not ship.
    Get-ChildItem $built -Filter *.dll |
        Where-Object { $_.Length -gt 0 -and $pluginModuleDlls -notcontains $_.Name } |
        Copy-Item -Destination $Destination
    # [Why check again after copying] The filter above is "exclusion-style": when a new plugin
    # format is added later (say AUv2), changing `$pluginModuleDlls` in one place is enough; but
    # if anyone writes it wrong, or swaps `Get-ChildItem` for some other approach, the artifact
    # silently grows 40 MB while the build still succeeds. With the check here, the mistake
    # becomes a readable failure at build time.
    foreach ($module in $pluginModuleDlls) {
        if (Test-Path (Join-Path $Destination $module)) {
            throw "App payload picked up a host plugin module: $(Join-Path $Destination $module)"
        }
    }
}

# Model weights are not in the repo (see scripts/setup/fetch-models.py): fetch if missing, and a
# run over existing files is just a directory scan. CI caches the extracted result in
# actions/cache keyed by the manifest hash, so this usually does nothing here.
$python = Resolve-Python
& $python.Exe @($python.Prefix) (Join-Path $PSScriptRoot '..\setup\fetch-models.py')
Assert-Ok 'model weights'
& $python.Exe @($python.Prefix) (Join-Path $PSScriptRoot '..\setup\fetch-models.py') --fetch-game
Assert-Ok 'GAME model weights'

$profile = $Configuration.ToLowerInvariant()
$cargoTarget = Join-Path $root 'backend\target'
if ($env:CARGO_TARGET_DIR) {
    $cargoTarget = [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
    $env:CARGO_TARGET_DIR = $cargoTarget
}
$built = Join-Path $cargoTarget $profile
$dist = Join-Path $root 'dist'
$stage = Join-Path $dist '_stage\installer'
$version = (Get-Content backend\src-tauri\tauri.conf.json -Raw | ConvertFrom-Json).version
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
# Licenses shipped with the artifacts: the root LICENSE (MIT + third-party list) and the
# licenses\ directory (full texts and notices of each license). One list only, shared by App and
# plugins, so the two packaging paths never keep separate copies.
$licenses = @('LICENSE', 'licenses')
$modelsSource = Join-Path $root 'backend\src-tauri\resources\models'
$modelsManifest = Get-Content (Join-Path $modelsSource 'models.json') -Raw | ConvertFrom-Json
$modelsVersion = $modelsManifest.version
New-Item -ItemType Directory -Force $dist | Out-Null

# ── Frontend ────────────────────────────────────────────────────────────────
# App and plugins share frontend\dist; build it once. tauri's beforeBuildCommand would run the
# build again, so the code below blanks it with an empty config — inline JSON gets its inner
# quotes stripped when Windows PowerShell passes it to a native process, so it must be written
# to a file and passed as a path.
if ($Target -ne 'Installer' -and !$SkipFrontend) {
    npm --prefix frontend run build
    Assert-Ok 'frontend build'
}
if ($Target -ne 'Installer' -and !(Test-Path frontend\dist\plugin.html)) {
    throw 'frontend\dist has no plugin.html: rebuild without -SkipFrontend'
}

# ── App ─────────────────────────────────────────────────────────────────────
# [Why --no-bundle] The installer is built by us (scripts/release/merged-installer.nsi): it
# installs two locations and takes exclusive ownership of the model store, which Tauri's bundler
# cannot provide. --no-bundle only compiles the exe and the DLLs it depends on.
if ($Target -eq 'All' -or $Target -eq 'App') {
    $blank = Join-Path ([IO.Path]::GetTempPath()) ("hfs-build-" + [Guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::WriteAllText($blank, '{"build":{"beforeBuildCommand":""}}')
    Push-Location backend
    try {
        $tauriArgs = @('tauri', 'build', '--ci', '--no-bundle', '--config', $blank)
        if ($Configuration -eq 'Debug') { $tauriArgs += '--debug' }
        cargo @tauriArgs
        Assert-Ok 'App build'
    }
    finally {
        Pop-Location
        Remove-Item $blank -Force
    }
}

# ── Plugin engine and CLAP module: one compile, two assemblies ──────────────
#
# [Why both shells must finish compiling in the **same** cargo call, then be assembled separately]
# The two shells share one target directory; within a single call, the `ara2-bridge` features are
# unioned (`vst3 + clap`, a union within the contract — see the format_feature_boundary notes).
# Split into two calls, the second rebuilds the dependency tree with a different feature plan,
# and the two artifacts may come from different build plans — post-hoc hash comparison and
# provenance tracing both break. Merged into one call, order no longer affects the result; the
# features of `hifishifter-plugin` (rlib-only) and `hifishifter-kernel` are also resolved only
# once, so CLAP/VST3 still compile the same kernel code.
if ($Target -eq 'All' -or $Target -eq 'Plugin' -or $Target -eq 'Clap') {
    $pluginBuild = @(
        'build', '--manifest-path', 'backend\Cargo.toml',
        '-p', 'hifishifter-vst3', '-p', 'hifishifter-clap'
    )
    if ($Configuration -eq 'Release') { $pluginBuild += '--release' }
    cargo @pluginBuild
    Assert-Ok 'plugin engine and CLAP module build'
}

# ── VST3 ────────────────────────────────────────────────────────────────────
if ($Target -eq 'All' -or $Target -eq 'Plugin') {
    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $module = Join-Path $bundle 'Contents\x86_64-win'
    $resources = Join-Path $bundle 'Contents\Resources'
    # [Why both directories are rebuilt wholesale instead of updated in place] Two reasons stack:
    #   1. Vite chunks carry content hashes and Copy-Item only writes, never deletes — without
    #      clearing, chunks from the previous round linger and the bundle grows monotonically
    #      with every build (exactly what the installer does on the user's machine);
    #   2. Copy-Item's semantic trap: when the destination directory already exists, it copies
    #      **the source directory itself** into it, so the second build yields assets\assets and
    #      the third assets\assets\assets — or fails outright with "item already exists".
    # Reset first, then copy contents; both problems disappear and the build is idempotent.
    Reset-Directory $module
    Reset-Directory "$resources\frontend"
    Reset-Directory "$resources\models"

    Copy-Item "$built\hifishifter_vst3.dll" "$module\HiFiShifterEngine.dll"
    Assert-SameBytes "$built\hifishifter_vst3.dll" "$module\HiFiShifterEngine.dll"
    # The DLL whitelist deliberately excludes vslib: it is the standalone App's closed-source
    # runtime and is not distributed with plugins.
    foreach ($dll in 'onnxruntime.dll', 'DirectML.dll', 'SoundTouchDLL.dll') {
        if (Test-Path "$built\$dll") { Copy-Item "$built\$dll" $module }
    }

    # Static entries and shared assets are copied together; SDK type declarations are not runtime payload.
    Get-ChildItem -LiteralPath frontend\dist |
        Where-Object { $_.Name -ne 'sdk' } |
        Copy-Item -Destination "$resources\frontend" -Recurse
    Copy-Item backend\src-tauri\icons\icon.ico "$resources\frontend"

    # [Why the bundle holds only the manifest, not the weights] The installer writes the weights
    # to %PROGRAMDATA%\HiFiShifter\models\<version>, and both the standalone App and the plugin
    # read that one copy (see backend/hifishifter-kernel/src/model_store.rs). Keeping another
    # copy in the bundle would charge the user's disk a second time for the same 152 MB. For
    # manual-copy installs, use the portable zip instead.
    Copy-Item "$modelsSource\models.json" "$resources\models"

    # Native loader: the host loads this thin DLL, which in turn pulls up the adjacent Rust engine by absolute path.
    $objDir = Join-Path ([IO.Path]::GetTempPath()) ("hfs-loader-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force $objDir | Out-Null
    cl.exe /nologo /utf-8 /std:c++17 /LD /MT /EHsc /O2 backend\hifishifter-plugin\native\module_loader.cpp "/Fo:$objDir\module_loader.obj" /link "/OUT:$module\HiFiShifter.vst3" "/IMPLIB:$objDir\HiFiShifterLoader.lib"
    Assert-Ok 'native VST3 loader build'
    $resolvedObjects = [IO.Path]::GetFullPath($objDir)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolvedObjects.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected loader cleanup path' }
    Remove-Item -LiteralPath $resolvedObjects -Recurse -Force

    # Compiling successfully is not the same as being loadable by a host. Local builds run a
    # black-box gate by default; CI build-only runs skip it.
    if (!$SkipVerify) { & .\scripts\vst3\windows\verify.ps1 -BundleDirectory $bundle }
}

if ($Configuration -eq 'Debug') {
    Write-Host "debug artifacts ready: $dist" -ForegroundColor Green
    return
}

if ($Target -eq 'All' -or $Target -eq 'Plugin' -or $Target -eq 'Clap') {
    $clapModule = Join-Path $built 'hifishifter_clap.dll'
    if (!(Test-Path $clapModule)) { throw "Missing $clapModule" }
    Copy-Item $clapModule (Join-Path $dist 'HiFiShifter.clap') -Force
    Assert-SameBytes $clapModule (Join-Path $dist 'HiFiShifter.clap')
    # The single-file CLAP's resources must sit next to the module (ResourceLayout::Sibling):
    # frontend output + model manifest. Missing either one, the host scans the plugin but only
    # ever gets a blank window.
    $clapRes = Join-Path $dist 'HiFiShifter-resources'
    Reset-Directory (Join-Path $clapRes 'frontend')
    Reset-Directory (Join-Path $clapRes 'models')
    # The same shared assets as the VST3 bundle: the host plugin and the floating editor share
    # chunks, and SDK type declarations are not runtime payload. Copying only plugin.html would
    # 404 detached.html inside the plugin window.
    Get-ChildItem -LiteralPath frontend\dist |
        Where-Object { $_.Name -ne 'sdk' } |
        Copy-Item -Destination (Join-Path $clapRes 'frontend') -Recurse
    Copy-Item (Join-Path $modelsSource 'models.json') (Join-Path $clapRes 'models')
    # [Why SoundTouchDLL.dll also goes into the CLAP resource directory] All three formats now
    # compile the same kernel: SoundTouch switched from `#[link]` to runtime libloading, whose
    # candidate paths include next to the module and the sibling `<stem>-resources`. VST3 puts
    # the DLL next to the engine and the App next to the exe; CLAP is a single-file module in a
    # public directory where a generically named DLL must not be dropped, so it goes into its own
    # private resource directory. Missing this DLL is not an error (is_available()=false → the
    # existing Signalsmith fallback), but shipping it keeps behavior consistent with VST3/App —
    # the user's time-stretch algorithm setting must not change with the plugin format.
    $soundtouch = Join-Path $built 'SoundTouchDLL.dll'
    if (Test-Path $soundtouch) { Copy-Item $soundtouch (Join-Path $clapRes 'SoundTouchDLL.dll') -Force }
    # [Why webview is deleted explicitly] The WebView2 user data directory is **runtime state**
    # (cache, lock files, a local profile); it once landed inside the resource directory and got
    # packed into an installer, and it would hand the next user an occupied profile. The resource
    # directory may contain only frontend, models and the optionally loaded-at-runtime
    # SoundTouchDLL.dll — never WebView2 user data.
    Remove-Item -LiteralPath (Join-Path $clapRes 'webview') -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "CLAP resources assembled: $clapRes" -ForegroundColor Green
    if (!$SkipVerify) { & .\scripts\clap\windows\verify.ps1 -ModulePath (Join-Path $dist 'HiFiShifter.clap') }
}

# ── Installer staging ───────────────────────────────────────────────────────
#
# [Why the staging directory is "reusable"] -Target Installer only re-runs the installer (one
# string change, a different WebView2 bootstrapper) and must not recompile Rust. Convention:
# dist\_stage\installer holds
# app\ / vst3\ / models\ / LICENSE / license-page.txt / licenses\ / models-verify.nsh, all produced by this script.

function Write-AppPayload {
    $app = Join-Path $stage 'app'
    Reset-Directory $app
    Copy-Item "$built\HiFiShifter.exe" $app
    # Runtime DLLs and "which modules belong to the plugin payload" are decided in one place,
    # Copy-AppRuntimeDlls, shared with the portable zip (previously written in two places; adding
    # CLAP only updated one of them).
    Copy-AppRuntimeDlls $app
    New-Item -ItemType Directory -Force "$app\models" | Out-Null
    Copy-Item "$modelsSource\models.json" "$app\models"
    Write-Host "App payload: $app" -ForegroundColor Green
}

function Write-Vst3Stage {
    $vst3 = Join-Path $stage 'vst3'
    Reset-Directory $vst3
    # [Why copy the bundle's contents rather than the HiFiShifter.vst3 directory itself] The
    # installer's landing spot is $HfsVst3Dir\${VST3_SUBDIR}, then File /r "${VST3_DIR}\*" lays
    # down what is in staging. If the whole directory went into staging, the user's machine would
    # gain an extra layer
    # (…\VST3\HiFiShifter.vst3\HiFiShifter.vst3\Contents), while the host only recognizes
    # <name>.vst3\Contents\<arch>\<name>.vst3 — the plugin would quietly "install but never be scanned".
    #
    # [Why copy by per-architecture whitelist instead of Copy-Item * -Recurse] dist\HiFiShifter.vst3
    # is a bundle shared across platforms: Linux/macOS builds write Contents\x86_64-linux or
    # Contents\MacOS, Windows builds write Contents\x86_64-win, and both ends share the same
    # Contents\Resources. A bare -Recurse would sweep the entire architecture directory left by
    # the previous other-platform build (about 57 MB of .so/.dylib) into the Windows installer,
    # and the host would see a pile of unloadable binaries during its scan. Take only the two
    # directories this platform needs; nothing left behind by any other architecture reaches staging.
    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $contents = Join-Path $bundle 'Contents'
    $winModule = Join-Path $contents 'x86_64-win'
    $resources = Join-Path $contents 'Resources'
    foreach ($required in $winModule, $resources) {
        if (!(Test-Path -LiteralPath $required)) {
            throw "VST3 bundle is missing $required: run .\scripts\build\build.ps1 -Target Plugin first"
        }
    }
    New-Item -ItemType Directory -Force (Join-Path $vst3 'Contents') | Out-Null
    Copy-Item $winModule (Join-Path $vst3 'Contents\x86_64-win') -Recurse
    Copy-Item $resources (Join-Path $vst3 'Contents\Resources') -Recurse
    # Everything the installer ships comes from these two directories: compare bytes once more at the payload boundary.
    Assert-SameBytes "$built\hifishifter_vst3.dll" (Join-Path $vst3 'Contents\x86_64-win\HiFiShifterEngine.dll')

    # Backstop: only x86_64-win may appear as an architecture directory in staging. If another
    # platform-native architecture is added later, whitelist it explicitly here instead of
    # restoring Copy-Item *.
    $foreign = @(Get-ChildItem (Join-Path $vst3 'Contents') -Directory |
        Where-Object { $_.Name -ne 'Resources' -and $_.Name -ne 'x86_64-win' })
    if ($foreign.Count) {
        throw "VST3 staging picked up a non-Windows architecture directory: $($foreign.Name -join ', ')"
    }
    Write-Host "VST3 payload: $vst3" -ForegroundColor Green
}

function Write-ClapStage {
    $clap = Join-Path $stage 'HiFiShifter.clap'
    Copy-Item (Join-Path $dist 'HiFiShifter.clap') $clap -Force
    Assert-SameBytes (Join-Path $built 'hifishifter_clap.dll') $clap
    Write-Host "CLAP payload: $clap" -ForegroundColor Green
    # [Why the resource directory must be staged along with it] On Windows, CLAP is a single-file
    # module; the frontend and the model manifest sit in the sibling HiFiShifter-resources per
    # ResourceLayout::Sibling. Installing only the .clap makes frontend_dir() fail and the host
    # sees a blank window.
    $resources = Join-Path $dist 'HiFiShifter-resources'
    if (!(Test-Path $resources)) { throw "Missing CLAP resource directory: $resources" }
    # One more gate before staging: runtime state must never enter the installer (same criterion as the assembly section of build.ps1).
    if (Test-Path (Join-Path $resources 'webview')) {
        throw "CLAP resource directory picked up a WebView2 runtime directory: $(Join-Path $resources 'webview')"
    }
    $staged = Join-Path $stage 'HiFiShifter-resources'
    Reset-Directory $staged
    Copy-Item "$resources\*" $staged -Recurse -Force
    if (!(Test-Path (Join-Path $staged 'frontend\plugin.html'))) {
        throw "CLAP resource directory is missing frontend\plugin.html"
    }
    Write-Host "CLAP resources: $staged" -ForegroundColor Green
}

function Write-ModelsStage {
    $models = Join-Path $stage 'models'
    Reset-Directory $models
    Copy-Item "$modelsSource\*" $models -Recurse
    Write-Host "Model payload ($modelsVersion): $models" -ForegroundColor Green
}

function Write-LicenseStage {
    $license = Join-Path $stage 'LICENSE'
    if (Test-Path $license) { Remove-Item $license -Force }
    Copy-Item LICENSE $license
    $dir = Join-Path $stage 'licenses'
    Reset-Directory $dir
    # Copy the contents, not the directory: Reset-Directory already created $dir, and copying the
    # licenses directory itself would produce licenses\licenses\README.md — while LICENSE
    # references the path licenses/README.md.
    Copy-Item 'licenses\*' $dir -Recurse

    # [Why the license page needs its own BOM-carrying copy] NSIS's LicenseData reads the file
    # into UTF-16 **at compile time**: with a BOM it decodes as UTF-8, without one it decodes with
    # the ANSI code page of the machine doing the compile. LICENSE is BOM-less UTF-8, so an
    # installer built on a Chinese-locale Windows would display θ as a garbled hanzi and drop
    # whole runs of Japanese kana — with no compile error (it is only visible once the license
    # page opens). The shipped LICENSE must stay byte-identical to the repo (users and tools both
    # compare it), so this few-KB copy feeds only the license page: read with strict UTF-8 (fails
    # right here if LICENSE is not valid UTF-8), written with a BOM.
    $page = Join-Path $stage 'license-page.txt'
    $strictUtf8 = New-Object Text.UTF8Encoding($false, $true)
    [IO.File]::WriteAllText($page, [IO.File]::ReadAllText((Join-Path $root 'LICENSE'), $strictUtf8),
        (New-Object Text.UTF8Encoding($true)))

    Write-Host "License payload: $stage" -ForegroundColor Green
}

# The installer's integrity checklist: relative path and byte size for every file.
#
# [Why generate it instead of parsing JSON inside NSIS] NSIS has no JSON parser, and a "the
# directory exists" criterion would disagree with the runtime model_store::looks_complete —
# producing "installer says installed, runtime says unavailable", where the user only sees
# "inference unavailable". Here the manifest is flattened into NSIS statements so both sides use
# the same data and the same criterion.
function Write-ModelsVerifyInclude {
    $path = Join-Path $stage 'models-verify.nsh'
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('; Generated by scripts/build/build.ps1 — do not edit by hand.')
    $lines.Add('; Lists the relative path and byte size of every file in the model manifest: the installer uses')
    $lines.Add('; it to decide whether the store is complete, matching runtime model_store::looks_complete exactly (file exists + byte count equal).')
    $lines.Add('Function HfsVerifyModelFiles')
    foreach ($file in $modelsManifest.files) {
        $relative = ($file.path -replace '/', '\')
        $lines.Add(('  StrCpy $R0 "{0}"' -f $relative))
        $lines.Add(('  StrCpy $R1 "{0}"' -f $file.bytes))
        $lines.Add('  Call HfsCheckModelFile')
        # $R2 = 0 means this entry failed: return immediately (the remaining checks are pointless).
        $lines.Add('  StrCmp $R2 0 0 +2')
        $lines.Add('  Return')
    }
    $lines.Add('FunctionEnd')
    [IO.File]::WriteAllText($path, ($lines -join "`r`n"), (New-Object Text.UTF8Encoding($true)))
}

function Get-WebView2Bootstrapper {
    if ($WebView2Bootstrapper) {
        if (!(Test-Path $WebView2Bootstrapper)) { throw "WebView2 bootstrapper not found: $WebView2Bootstrapper" }
        return (Resolve-Path $WebView2Bootstrapper).Path
    }
    $cache = Join-Path $dist '_cache'
    New-Item -ItemType Directory -Force $cache | Out-Null
    $target = Join-Path $cache 'MicrosoftEdgeWebview2Setup.exe'
    if (Test-Path $target) {
        # Cache hits get the same minimal-size check: a download interrupted halfway leaves a
        # truncated file (-OutFile does not clean up), and trusting the cache unconditionally
        # packs a broken bootstrapper into the installer.
        $cachedSize = (Get-Item $target).Length
        if ($cachedSize -lt 1MB) {
            Remove-Item $target -Force
            Write-Host "Cached WebView2 bootstrapper is incomplete (only $cachedSize bytes); deleted and re-downloading." -ForegroundColor Yellow
        } else {
            return $target
        }
    }
    Write-Host 'Downloading WebView2 bootstrapper (about 1.7 MB)...' -ForegroundColor Cyan
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' -OutFile $target -UseBasicParsing
    $size = (Get-Item $target).Length
    if ($size -lt 1MB) { throw "WebView2 bootstrapper download looks wrong (only $size bytes)" }
    return $target
}

if ($Target -eq 'All' -or $Target -eq 'App') { Write-AppPayload }
if ($Target -eq 'All' -or $Target -eq 'Plugin') { Write-Vst3Stage }
if ($Target -eq 'All' -or $Target -eq 'Plugin' -or $Target -eq 'Clap') { Write-ClapStage }

# Models / licenses go only into the installer staging area: single-target artifacts (the App
# executable / VST3 bundle / CLAP file) do not consume them. They used to run unconditionally for
# -Target App/Plugin/Clap as well, copying hundreds of MB for nothing and — more importantly —
# making the "installer section" below look valid for every target.
if ($Target -eq 'All') {
    Write-ModelsStage
    Write-LicenseStage
    Write-ModelsVerifyInclude
}

if ($Target -eq 'Installer') {
    foreach ($required in 'app', 'vst3', 'HiFiShifter.clap', 'HiFiShifter-resources', 'models', 'licenses', 'LICENSE', 'license-page.txt', 'models-verify.nsh') {
        if (!(Test-Path (Join-Path $stage $required))) {
            throw "dist\_stage\installer is missing $required: run .\scripts\build\build.ps1 -Target All first"
        }
    }
}

# [Why the installer section is gated by target] The file header promises -Target App "builds no
# installer", likewise -Target Plugin / Clap, but the byte comparison and makensis used to run
# unconditionally for **every** target: a single target never assembles a full payload (-Target
# App has no vst3\, -Target Clap has no app\), so stale files left in staging made the byte
# comparison throw on the spot, or makensis reported NSIS compile errors over directories that
# do not exist — both documented entry points failed the moment they ran. Only All / Installer
# actually produce an installer.
if ($Target -eq 'All' -or $Target -eq 'Installer') {
    # [The last gate before the installer: the staged plugin payload must be the exact bytes in target]
    # This check specifically covers the "rebuilt only one format" usage: even a bare -Target
    # Clap run compares here as long as the engine exists in target — an ordering bug can no
    # longer slip through as "the installer carries the previous engine". (-Target Installer
    # intentionally reuses the existing staging area; target may be empty then, and the
    # comparison skips itself.)
    foreach ($payload in @(
        @{ Built = (Join-Path $built 'hifishifter_vst3.dll'); Staged = (Join-Path $stage 'vst3\Contents\x86_64-win\HiFiShifterEngine.dll') },
        @{ Built = (Join-Path $built 'hifishifter_clap.dll'); Staged = (Join-Path $stage 'HiFiShifter.clap') }
    )) {
        if (Test-Path -LiteralPath $payload.Built) {
            Assert-SameBytes $payload.Built $payload.Staged
        }
    }

    # ── Installer ───────────────────────────────────────────────────────────────
    $makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
    if (!$makensis) {
        # [Why not read the `ProgramFiles(x86)` environment variable] It does not exist in some
        # hosts at all, and `msvc-env.ps1` writes empty values back when importing the vcvars
        # environment (.NET treats an empty value as "delete"), so the tail of the build would
        # fail with an error unrelated to NSIS: "Cannot bind argument to parameter 'Path'".
        # The directory can only be asked of the system API (same source as `Get-ProgramFilesRoots`
        # in `scripts/env/msvc-env.ps1`).
        foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
            $base = [Environment]::GetFolderPath($kind)
            if (!$base) { continue }
            $candidate = Join-Path $base 'NSIS\makensis.exe'
            if (Test-Path -LiteralPath $candidate) { $makensis = $candidate; break }
        }
    }
    if (!$makensis) {
        throw 'makensis not found: run scripts\setup\install_deps_windows.ps1 first (Windows ships only this one installer; without it there is no artifact)'
    }

    # Installer strings are generated from the five-language table; regenerate first so a missing
    # translation fails here.
    node (Join-Path $PSScriptRoot '..\release\build-installer-strings.mjs')
    Assert-Ok 'installer string generation'

    $bootstrapper = Get-WebView2Bootstrapper
    $setup = Join-Path $dist ('HiFiShifter_v' + $version + '_windows-' + $arch + '-setup.exe')
    Remove-Item $setup -Force -ErrorAction SilentlyContinue

    # Every path is passed as a /D define; the script never guesses relative paths (values are
    # quoted: paths may contain spaces).
    & $makensis /INPUTCHARSET UTF8 `
        "/DAPP_DIR=$(Join-Path $stage 'app')" `
        "/DVST3_DIR=$(Join-Path $stage 'vst3')" `
        "/DCLAP_FILE=$(Join-Path $stage 'HiFiShifter.clap')" `
        "/DCLAP_RES_DIR=$(Join-Path $stage 'HiFiShifter-resources')" `
        "/DMODELS_DIR=$(Join-Path $stage 'models')" `
        "/DMODELS_VERSION=$modelsVersion" `
        "/DMODELS_VERIFY=$(Join-Path $stage 'models-verify.nsh')" `
        "/DHFS_LICENSE=$(Join-Path $stage 'LICENSE')" `
        "/DHFS_LICENSE_PAGE=$(Join-Path $stage 'license-page.txt')" `
        "/DHFS_LICENSES=$(Join-Path $stage 'licenses')" `
        "/DOUTPUT_FILE=$setup" `
        "/DAPP_VERSION=$version" `
        "/DWEBVIEW2_BOOTSTRAPPER=$bootstrapper" `
        "/DHFS_ICON=$(Join-Path $root 'backend\src-tauri\icons\icon.ico')" `
        (Join-Path $PSScriptRoot '..\release\merged-installer.nsi')
    Assert-Ok 'installer compile'
    Write-Host "Installer: $setup" -ForegroundColor Green
}

# ── Portable zip ────────────────────────────────────────────────────────────
# The portable zip is **self-contained**: the full models ship inside it, unpack and run, with no
# dependency on the installer or ProgramData.
if ($Target -ne 'All') {
    return
}

$portable = Join-Path $dist 'HiFiShifter'
Reset-Directory $portable
Copy-Item "$built\HiFiShifter.exe" $portable
# Same criterion as the installer's App payload: plugin modules do not ship with the standalone App.
Copy-AppRuntimeDlls $portable
Copy-Item $modelsSource "$portable\models" -Recurse
Copy-Item $licenses $portable -Recurse

$portableZip = Join-Path $dist "HiFiShifter-v$version-portable-win-$arch.zip"
Remove-Item $portableZip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $portable -DestinationPath $portableZip -CompressionLevel Optimal
Remove-Item $portable -Recurse -Force
Write-Host "Portable zip: $portableZip" -ForegroundColor Green
