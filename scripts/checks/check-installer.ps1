# Installer closed-loop gate: install → inspect → uninstall → inspect.
#
# [Why CI must run a real install] Almost none of the installer's failure modes raise an error: the
# model bundle is missing a file (the runtime just reports "inference unavailable"), a copy of the
# weights sneaks back into the bundle (disk quietly grows by a full second copy), or uninstall leaves files
# behind (the user sees two copies after reinstalling). These are only visible by actually running
# the installer and asserting each item one by one.
#
# [Why the "test mode" installer is what gets installed] The release branch requires
# RequestExecutionLevel admin and writes HKLM / Program Files, while elevation on a CI runner gets
# the script stuck at UAC. HFS_TEST_MODE only swaps the install location and registry root for the
# current user; **everything else is verbatim identical** (pages, directory reconciliation, models
# section, uninstall section). The release branch stays admin / HKLM / Common Files — that part is
# pinned statically by check-product-consistency.ps1.
#
#   .\scripts\checks\check-installer.ps1
#   .\scripts\checks\check-installer.ps1 -ReleaseInstallerPath dist\HiFiShifter_v..._setup.exe
[CmdletBinding()]
param(
    # Release installer (used for the size assertion). Defaults to the newest dist\*-setup.exe.
    [string]$ReleaseInstallerPath = '',
    # Installer size cap (MB). The old App setup 169.8 MB + VST3 setup 166.7 MB were two artifacts
    # each carrying its own weights; after the merge the weights go into the installer only once,
    # so 230 MB is the "there really is only one copy" criterion.
    [int]$MaxSizeMb = 230
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$stage = Join-Path $root 'dist\_stage\installer'
$failed = [System.Collections.Generic.List[string]]::new()

function Assert-That($name, [scriptblock]$body) {
    try {
        & $body
        Write-Host "  [OK]   $name" -ForegroundColor Green
    }
    catch {
        Write-Host "  [FAIL] $name" -ForegroundColor Red
        Write-Host "         $($_.Exception.Message)" -ForegroundColor Red
        $failed.Add($name)
    }
}

# [Why not Start-Process -Wait] If install / reinstall / uninstall gets stuck at UAC or deadlocks
# inside the installer, `-Wait` hangs until the job-level timeout with no clue in the logs. Here we
# wait a bounded time, kill on timeout, and throw a locatable error (same discipline as the load
# probe in vst3/windows/verify.ps1).
function Invoke-InstallerProcess([string]$path, [string[]]$arguments, [int]$timeoutSeconds = 180) {
    $proc = Start-Process -FilePath $path -ArgumentList $arguments -WindowStyle Hidden -PassThru
    if (!$proc.WaitForExit($timeoutSeconds * 1000)) {
        $proc.Kill()
        throw "$path did not return within $timeoutSeconds seconds (UAC / installer deadlock); killed"
    }
    return $proc
}

function Get-Makensis {
    $cmd = (Get-Command makensis -ErrorAction SilentlyContinue).Source
    if ($cmd) { return $cmd }
    # [Why not read the `ProgramFiles(x86)` env var] See the same spot in build.ps1: it may be absent,
    # or deleted by the vcvars import (empty value = deletion), while the error points at the `-Path`
    # parameter.
    foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
        $base = [Environment]::GetFolderPath($kind)
        if (!$base) { continue }
        $candidate = Join-Path $base 'NSIS\makensis.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw 'makensis not found: run scripts\setup\install_deps_windows.ps1 first'
}

function Remove-GateDirectory([string]$path) {
    $resolved = [IO.Path]::GetFullPath($path)
    $allowed = @(
        [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'HiFiShifterInstallerTest')),
        [IO.Path]::GetFullPath((Join-Path $env:APPDATA 'HiFiShifterTest')),
        # Simulated directories covering the overwrite-install "legacy model root" case: they must be
        # fixed names and exist only under $env:APPDATA.
        [IO.Path]::GetFullPath((Join-Path $env:APPDATA 'HiFiShifterTestLegacy')),
        [IO.Path]::GetFullPath((Join-Path $env:APPDATA 'HiFiShifterTestLegacy2'))
    )
    if ($allowed -notcontains $resolved) { throw "Unexpected installer test cleanup path: $resolved" }
    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host 'Installer gate' -ForegroundColor Cyan

if (!$ReleaseInstallerPath) {
    $found = @(Get-ChildItem (Join-Path $root 'dist') -Filter '*-setup.exe' -ErrorAction SilentlyContinue)
    if ($found.Count -eq 0) { throw 'no installer in dist: run scripts\build\build.ps1 -Target All first' }
    $ReleaseInstallerPath = ($found | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
$releaseInstaller = (Resolve-Path $ReleaseInstallerPath).Path
$sizeMb = [math]::Round((Get-Item $releaseInstaller).Length / 1MB, 1)
Assert-That "installer size below $MaxSizeMb MB (weights installed only once)" {
    if ($sizeMb -ge $MaxSizeMb) { throw "installer is $sizeMb MB ≥ $MaxSizeMb MB" }
    "$sizeMb MB"
}

# ── Compile the test-mode installer (same payload)
foreach ($required in 'app', 'vst3', 'HiFiShifter.clap', 'HiFiShifter-resources', 'models', 'licenses', 'LICENSE', 'license-page.txt', 'models-verify.nsh') {
    if (!(Test-Path (Join-Path $stage $required))) { throw "dist\_stage\installer missing $required" }
}

# [Why the architecture whitelist is pinned] dist\HiFiShifter.vst3 is a bundle shared across
# platforms; Linux/macOS builds leave x86_64-linux / MacOS and similar directories inside Contents.
# If Windows packaging used Copy-Item * -Recurse, those other platforms' .so/.dylib files (about
# 57 MB) would end up in the installer via NSIS too. Check the staging area before compiling: any
# architecture directory outside the whitelist fails right away, rather than being discovered on a
# user's machine.
Assert-That 'VST3 staging area contains only Windows architectures' {
    $contents = Join-Path $stage 'vst3\Contents'
    if (!(Test-Path -LiteralPath $contents)) { throw "missing $contents" }
    $allowed = @('Resources', 'x86_64-win')
    $foreign = @(Get-ChildItem -LiteralPath $contents -Directory |
        Where-Object { $allowed -notcontains $_.Name })
    if ($foreign.Count) { throw "staging area picked up non-Windows architecture directories: $($foreign.Name -join ', ')" }
    $allowed -join ', '
}
$manifest = Get-Content (Join-Path $root 'backend\src-tauri\resources\models\models.json') -Raw | ConvertFrom-Json
$bootstrapper = Join-Path $root 'dist\_cache\MicrosoftEdgeWebview2Setup.exe'
if (!(Test-Path $bootstrapper)) { throw "missing WebView2 bootstrapper: $bootstrapper (run build.ps1 first)" }

$testInstaller = Join-Path $env:TEMP ('hfs-installer-gate-' + [Guid]::NewGuid().ToString('N') + '.exe')
$makensis = Get-Makensis

# ── License page encoding ───────────────────────────────────────────────
#
# [Why the criterion is "bytes inside the compiled artifact"] NSIS LicenseData reads the file into
# UTF-16 **at compile time** and stores it in the installer: with a BOM the file is decoded as
# UTF-8, without one it is decoded using **the ANSI code page of the machine doing the compile**.
# The latter compiles with zero warnings and opens fine after install — it just turns θ in the
# license page into mojibake and drops whole runs of Japanese kana — and only taking the artifact
# apart can catch it, so an uncompressed empty installer below reproduces this path.
$licensePage = Join-Path $stage 'license-page.txt'

Assert-That 'license page uses the BOM-carrying copy, not the bundled LICENSE' {
    $bom = [IO.File]::ReadAllBytes($licensePage)
    if ($bom.Length -lt 3 -or $bom[0] -ne 0xEF -or $bom[1] -ne 0xBB -or $bom[2] -ne 0xBF) {
        throw 'license-page.txt has no UTF-8 BOM: NSIS would decode it as ANSI and every non-ASCII character would be lost'
    }
    $nsi = [IO.File]::ReadAllText((Join-Path $root 'scripts\release\merged-installer.nsi'),
        (New-Object Text.UTF8Encoding($false)))
    if (!$nsi.Contains('MUI_PAGE_LICENSE "${HFS_LICENSE_PAGE}"')) {
        throw 'merged-installer.nsi license page no longer uses ${HFS_LICENSE_PAGE}'
    }
    'BOM + HFS_LICENSE_PAGE'
}

Assert-That 'license page embeds LICENSE verbatim as UTF-16 (not one character lost)' {
    $text = [IO.File]::ReadAllText($licensePage, (New-Object Text.UTF8Encoding($false)))
    $probe = Join-Path $env:TEMP ('hfs-license-probe-' + [Guid]::NewGuid().ToString('N'))
    $probeNsi = $probe + '.nsi'
    $probeExe = $probe + '.exe'
    # Probe: license page only, uncompressed (compressed data blocks cannot be searched byte-wise),
    # zero payload, a few hundred milliseconds.
    $probeScript = @(
        'Unicode true'
        'SetCompress off'
        '!include "MUI2.nsh"'
        'Name "hfs-license-probe"'
        ('OutFile "{0}"' -f $probeExe)
        'InstallDir "$TEMP\hfs-license-probe"'
        'RequestExecutionLevel user'
        ('!insertmacro MUI_PAGE_LICENSE "{0}"' -f $licensePage)
        '!insertmacro MUI_PAGE_INSTFILES'
        '!insertmacro MUI_LANGUAGE "English"'
        'Section'
        'SectionEnd'
    ) -join "`r`n"
    [IO.File]::WriteAllText($probeNsi, $probeScript, (New-Object Text.UTF8Encoding($true)))
    try {
        & $makensis /V1 $probeNsi | Out-Null
        if ($LASTEXITCODE) { throw "license-page probe failed to compile (exit code $LASTEXITCODE)" }
        $bytes = [IO.File]::ReadAllBytes($probeExe)
        # The text block may legally sit at an odd offset; decode both alignments.
        $oddLength = ($bytes.Length - 1) - (($bytes.Length - 1) % 2)
        $views = @(
            [Text.Encoding]::Unicode.GetString($bytes),
            [Text.Encoding]::Unicode.GetString($bytes, 1, $oddLength)
        )
        $checked = 0
        foreach ($line in ($text -split "`r?`n")) {
            $needle = $line.Trim()
            if ($needle -notmatch '[^\x00-\x7f]') { continue }
            $checked++
            $found = $false
            foreach ($view in $views) {
                if ($view.IndexOf($needle, [StringComparison]::Ordinal) -ge 0) { $found = $true; break }
            }
            if (!$found) { throw "this line was lost inside the installer (decoded as ANSI at compile time): $needle" }
        }
        "$checked non-ASCII lines embedded verbatim"
    }
    finally {
        Remove-Item $probeNsi, $probeExe -Force -ErrorAction SilentlyContinue
    }
}

# [Why an argument array instead of line continuations] PowerShell's continuation character is a
# backtick; the more of them a script contains, the easier it is for an editor / formatter to eat
# one space and turn it into "the command silently ends at end of line".
$nsisArgs = @(
    '/INPUTCHARSET', 'UTF8', '/DHFS_TEST_MODE',
    "/DAPP_DIR=$(Join-Path $stage 'app')",
    "/DVST3_DIR=$(Join-Path $stage 'vst3')",
    "/DCLAP_FILE=$(Join-Path $stage 'HiFiShifter.clap')",
    "/DCLAP_RES_DIR=$(Join-Path $stage 'HiFiShifter-resources')",
    "/DMODELS_DIR=$(Join-Path $stage 'models')",
    "/DMODELS_VERSION=$($manifest.version)",
    "/DMODELS_VERIFY=$(Join-Path $stage 'models-verify.nsh')",
    "/DHFS_LICENSE=$(Join-Path $stage 'LICENSE')",
    "/DHFS_LICENSE_PAGE=$(Join-Path $stage 'license-page.txt')",
    "/DHFS_LICENSES=$(Join-Path $stage 'licenses')",
    "/DOUTPUT_FILE=$testInstaller",
    '/DAPP_VERSION=0.0.0-installer-gate',
    "/DWEBVIEW2_BOOTSTRAPPER=$bootstrapper",
    "/DHFS_ICON=$(Join-Path $root 'backend\src-tauri\icons\icon.ico')",
    (Join-Path $root 'scripts\release\merged-installer.nsi')
)
& $makensis @nsisArgs
if ($LASTEXITCODE) { throw "test-mode installer failed to compile (exit code $LASTEXITCODE)" }

# Test-mode install root: InstallDir = %LOCALAPPDATA%\HiFiShifterInstallerTest\App
$testRoot = Join-Path $env:LOCALAPPDATA 'HiFiShifterInstallerTest'
$installDir = Join-Path $testRoot 'App'
$vst3Dir = Join-Path $testRoot 'VST3'
$clapDir = Join-Path $testRoot 'CLAP'
# Plugin resources do not go into the public CLAP directory (only the .clap itself lives there);
# in test mode they land in the product's sibling directory.
$clapResDir = Join-Path $testRoot 'CLAPResources'
# The test-mode model root is set via HFS_MODEL_ROOT (user privileges do not trigger UAC
# virtualization); release mode still uses %ProgramData%\HiFiShifter\models, written by the admin
# installer.
$modelRoot = Join-Path $env:APPDATA 'HiFiShifterTest\models'
$modelDir = Join-Path $modelRoot $manifest.version
$arpKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter'

# ── File association save / restore ──────────────────────────────
#
# Test mode writes .hshp/.hsp under HKCU\Software\Classes. A dev machine may already carry real
# associations, and the gate must not overwrite them without restoring them. Here reg.exe takes a
# **whole-key snapshot** of the four keys: more complete than per-value restore (OpenWithProgids,
# Content Type and other user values come back verbatim), and it leaves no empty keys created
# during the test behind in the system.
$classesRoot = 'HKCU:\Software\Classes'
$classesRegPath = 'HKCU\Software\Classes'
# Use a unique ProgID per run so same-named test keys on the dev machine cannot clobber each other.
$foreignProgId = 'HfsInstallerGateForeign.' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$assocKeys = @('.hshp', '.hsp', 'HiFiShifter.Project', 'Applications\HiFiShifter.exe')
$assocBackupDir = Join-Path $env:TEMP ('hfs-assoc-backup-' + [Guid]::NewGuid().ToString('N'))
$assocBackup = @{}
New-Item -ItemType Directory -Force $assocBackupDir | Out-Null
foreach ($rel in $assocKeys) {
    $full = Join-Path $classesRoot $rel
    $file = Join-Path $assocBackupDir (($rel -replace '[\\/]', '_') + '.reg')
    $existed = Test-Path -LiteralPath $full
    if ($existed) {
        & reg.exe export "HKCU\Software\Classes\$rel" $file /y | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "cannot back up registry key $full (reg export exit code $LASTEXITCODE)" }
    }
    $assocBackup[$rel] = [pscustomobject]@{ Existed = $existed; File = $file }
}

function Get-RegDefault([string]$path) {
    $key = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $key) { return $null }
    return $key.GetValue('')
}

function Get-RegValue([string]$path, [string]$name) {
    $key = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $key) { return $null }
    return $key.GetValue($name)
}

function Restore-AssociationBackups {
    if (!$assocBackupDir -or !(Test-Path -LiteralPath $assocBackupDir)) { return }
    foreach ($rel in $assocKeys) {
        $full = Join-Path $classesRoot $rel
        Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction SilentlyContinue
        $entry = $assocBackup[$rel]
        if ($entry -and $entry.Existed -and (Test-Path -LiteralPath $entry.File)) {
            & reg.exe import $entry.File | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "failed to restore file-association registry key: $full"
            }
        }
    }
    Remove-Item -LiteralPath $assocBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    $script:assocBackupDir = $null
}

# Test markers for the uninstall default that keeps cache/user data: create only two files named
# uniquely for this run, never delete the user's whole directories. One goes in the derivatives
# directory (logs), one in the user data directory (unsaved); the silent uninstall default only
# checks "Delete models", so both should survive. The model library is verified separately by the
# marker below for "deleted by default"; the "kept only when unchecked" branch is verified by a
# dedicated uninstall-option probe.
$userProfileRoot = Join-Path $env:LOCALAPPDATA 'HiFiShifter'
$gateTag = 'hfs-uninstall-gate-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)
$disposableMarker = Join-Path $userProfileRoot "logs\$gateTag.txt"
$userDataMarker = Join-Path $userProfileRoot "unsaved\$gateTag.txt"
# Model library marker: asserts the silent uninstall deletes the shared model library by default
# (independent of whether the install phase managed to build it).
$modelMarker = Join-Path $modelDir "$gateTag.txt"

Remove-GateDirectory $testRoot
# A previous failed run may leave a stale key with "only the ARP key, no uninstall.exe"; clear it so
# the gate starts from a genuinely fresh install every time (the installer handles such stale keys
# itself; this is a safety net that does not rely on it).
Remove-Item -LiteralPath $arpKey -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $disposableMarker, $userDataMarker -Force -ErrorAction SilentlyContinue
# Test-mode-only model root: every run starts from a clean state so a leftover .staging cannot make
# the installer misjudge "already complete" and skip writing the library. Never touches the release
# mode's %ProgramData%\HiFiShifter.
if (Test-Path (Join-Path $env:APPDATA 'HiFiShifterTest')) {
    Remove-GateDirectory (Join-Path $env:APPDATA 'HiFiShifterTest')
}
# Simulated directories for the legacy model root (the overwrite-install scenario "the old version
# wrote the shared library somewhere else"): likewise only these two fixed gate-owned names.
foreach ($stale in 'HiFiShifterTestLegacy', 'HiFiShifterTestLegacy2') {
    $path = Join-Path $env:APPDATA $stale
    if (Test-Path $path) { Remove-GateDirectory $path }
}

try {
    Write-Host '  Installing (silent)...' -ForegroundColor DarkGray
    $proc = Invoke-InstallerProcess $testInstaller @('/S')
    if ($proc.ExitCode -ne 0) { throw "installer exit code $($proc.ExitCode)" }

    Assert-That 'ARP registers an uninstall entry' {
        if (!(Test-Path $arpKey)) { throw "missing registry key $arpKey" }
        (Get-ItemProperty $arpKey).DisplayVersion
    }

    Assert-That 'file associations point at the installed main executable' {
        $hshp = Get-RegDefault "$classesRoot\.hshp"
        if ($hshp -ne 'HiFiShifter.Project') { throw ".hshp default ProgID = $hshp" }
        $command = Get-RegDefault "$classesRoot\HiFiShifter.Project\shell\open\command"
        if (!$command -or !$command.Contains("$installDir\HiFiShifter.exe")) {
            throw "open command does not point at ${installDir}: $command"
        }
        $hsp = Get-RegDefault "$classesRoot\.hsp"
        if ($hsp -ne 'HiFiShifter.Project') { throw ".hsp default ProgID = $hsp" }
        if ($null -eq (Get-RegValue "$classesRoot\.hshp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hshp\OpenWithProgids does not register HiFiShifter.Project'
        }
        if ($null -eq (Get-RegValue "$classesRoot\Applications\HiFiShifter.exe\SupportedTypes" '.hshp')) {
            throw 'Applications\HiFiShifter.exe\SupportedTypes does not register .hshp'
        }
        'HiFiShifter.Project → HiFiShifter.exe'
    }

    Assert-That 'no model weights in the App directory (manifest only)' {
        $onnx = @(Get-ChildItem $installDir -Recurse -Filter *.onnx -ErrorAction SilentlyContinue)
        if ($onnx.Count -gt 0) { throw "install directory still contains $($onnx.Count) .onnx file(s)" }
        if (!(Test-Path (Join-Path $installDir 'models\models.json'))) { throw 'missing models\models.json' }
        'no .onnx under INSTDIR'
    }

    # [Why even "no plugin modules in the App directory" is asserted] The plugins and the standalone
    # App build into the same `backend\target\<profile>` directory, and the packaging script separates
    # them with a single exclusion criterion. When that criterion is wrong, the installer is merely
    # "40-80 MB bigger" and everything works normally after install — you only notice by looking at
    # the installed directory. So the criterion must land on the real post-install location, not on a
    # comment inside build.ps1.
    Assert-That 'no host plugin modules in the App directory' {
        foreach ($module in 'hifishifter_vst3.dll', 'hifishifter_clap.dll') {
            if (Test-Path (Join-Path $installDir $module)) {
                throw "plugin module leaked into the App install directory: $module"
            }
        }
        'no hifishifter_vst3.dll / hifishifter_clap.dll under INSTDIR'
    }

    Assert-That 'license texts and notices land at the paths written in LICENSE' {
        foreach ($rel in 'LICENSE', 'licenses\README.md', 'licenses\COPYING.LGPL-2.1', 'licenses\Apache-2.0.txt') {
            if (!(Test-Path (Join-Path $installDir $rel))) { throw "App missing $rel" }
        }
        $vst3Resources = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents\Resources'
        foreach ($rel in 'LICENSE', 'licenses\README.md') {
            if (!(Test-Path (Join-Path $vst3Resources $rel))) { throw "VST3 missing $rel" }
        }
        if (Test-Path (Join-Path $installDir 'licenses\licenses')) { throw 'license directory gained an extra level: licenses\licenses' }
        'LICENSE + licenses\ present in both App and VST3'
    }

    Assert-That 'CLAP directory holds only the plugin module, no resources beside it' {
        if (!(Test-Path (Join-Path $clapDir 'HiFiShifter.clap'))) { throw 'CLAP module not installed into the chosen directory' }
        # Spec: the public CLAP directory holds only the .clap file itself. One more generically named
        # resource folder and users can neither tell ownership apart nor dare delete it; pinned here as
        # the "exactly one file" criterion.
        $stray = @(Get-ChildItem -LiteralPath $clapDir -Force | Where-Object { $_.Name -ne 'HiFiShifter.clap' })
        if ($stray.Count) { throw "non-plugin files leaked into the CLAP directory: $($stray.Name -join ', ')" }
        'only HiFiShifter.clap'
    }
    Assert-That 'CLAP frontend and model manifest land in the shared resource directory' {
        $clapFrontend = Join-Path $clapResDir 'frontend\plugin.html'
        if (!(Test-Path $clapFrontend)) { throw "CLAP frontend resource missing: $clapFrontend" }
        $clapManifest = Join-Path $clapResDir 'models\models.json'
        if (!(Test-Path $clapManifest)) { throw "CLAP model manifest missing: $clapManifest" }
        'frontend + manifest in shared dir'
    }
    # If the installer really shipped other platforms' binaries, this catches it too (staging checked
    # before compile, installed location checked after).
    Assert-That 'installed VST3 location contains only Windows architectures' {
        $installedContents = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents'
        $allowed = @('Resources', 'x86_64-win')
        $foreign = @(Get-ChildItem -LiteralPath $installedContents -Directory |
            Where-Object { $allowed -notcontains $_.Name })
        if ($foreign.Count) { throw "installed directory picked up non-Windows architecture directories: $($foreign.Name -join ', ')" }
        $allowed -join ', '
    }
    Assert-That 'VST3 Resources\models contains only models.json' {
        $models = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents\Resources\models'
        if (!(Test-Path $models)) { throw "missing $models" }
        $entries = @(Get-ChildItem $models -Recurse -File)
        if ($entries.Count -ne 1) { throw "expected exactly 1 file, found $($entries.Count)" }
        $entries[0].Name
    }

    Assert-That 'shared model library complete per the manifest' {
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw "missing $modelDir\models.json" }
        foreach ($file in $manifest.files) {
            $path = Join-Path $modelDir ($file.path -replace '/', '\')
            if (!(Test-Path $path)) { throw "missing $($file.path)" }
            if ((Get-Item $path).Length -ne $file.bytes) { throw "$($file.path) byte count mismatch" }
        }
        "$($manifest.files.Count) files"
    }

    # Overwrite install: remove the previous version first, then install again. Removal is done
    # in-process by the installer itself (the old uninstall.exe is not executed, which would pop a
    # second uninstall window). Options match the uninstall page (Delete models checked by default,
    # Delete cache unchecked by default); the two switches verify separately that "the choice really
    # takes effect". Cache markers sit in both logs (inside the derivatives whitelist) and unsaved
    # (deliberately kept) so only the regenerable ones get cleaned.
    New-Item -ItemType Directory -Force (Split-Path $modelMarker -Parent) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $disposableMarker -Parent) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $userDataMarker -Parent) | Out-Null
    Set-Content -LiteralPath $modelMarker -Value 'old'
    Set-Content -LiteralPath $disposableMarker -Value 'old'
    Set-Content -LiteralPath $userDataMarker -Value 'keep'
    Assert-That 'overwrite install uninstalls the previous version first (models deleted, cache kept by default)' {
        $before = Get-Item -LiteralPath (Join-Path $installDir 'HiFiShifter.exe')
        $again = Invoke-InstallerProcess $testInstaller @('/S')
        if ($again.ExitCode -ne 0) { throw "overwrite install exit code $($again.ExitCode)" }
        if (!(Test-Path (Join-Path $installDir 'HiFiShifter.exe'))) { throw 'main executable missing after reinstall' }
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw 'model library not rewritten after reinstall' }
        if (Test-Path -LiteralPath $modelMarker) { throw 'uninstaller did not delete the old model library as the default checkbox dictates' }
        if (!(Test-Path -LiteralPath $disposableMarker)) { throw 'cache and logs must not be cleaned by default' }
        if (!(Test-Path -LiteralPath $userDataMarker)) { throw 'user data was wrongly deleted' }
        $mtime = (Get-Item -LiteralPath (Join-Path $modelDir 'models.json')).LastWriteTimeUtc
        "rewritten at $mtime (old exe $($before.LastWriteTimeUtc))"
    }

    # Rebuild markers to verify the two branches "keep models unchecked + clean cache checked".
    New-Item -ItemType Directory -Force (Split-Path $modelMarker -Parent) | Out-Null
    Set-Content -LiteralPath $modelMarker -Value 'keep'
    Set-Content -LiteralPath $disposableMarker -Value 'old'
    Assert-That 'overwrite install can keep models and clean cache per the choice' {
        $keep = Invoke-InstallerProcess $testInstaller @('/S', '/DeleteModels=0', '/DeleteCache=1')
        if ($keep.ExitCode -ne 0) { throw "overwrite install exit code $($keep.ExitCode)" }
        if (!(Test-Path -LiteralPath $modelMarker)) { throw 'user chose to keep, yet the model library marker was deleted' }
        if (!(Test-Path (Join-Path $installDir 'HiFiShifter.exe'))) { throw 'main executable missing after reinstall' }
        if (Test-Path -LiteralPath $disposableMarker) { throw 'user checked clean cache, yet the logs marker is still there' }
        if (!(Test-Path -LiteralPath $userDataMarker)) { throw 'cache cleanup wrongly deleted the unsaved user data' }
        'models kept / cache cleaned / user data kept'
    }
    # ── Shared model library "legacy root" (version drift)
    # The shared library's location changes with the installer version. The installer writes the root
    # into the uninstall entry's ModelRoot; on upgrade the recorded root must be handled: keep → merge
    # into the new root (drop the old copy when the version duplicates); delete → clear the legacy
    # root too. Otherwise a weights copy the runtime never reads stays on disk.
    $legacyModelRoot = Join-Path $env:APPDATA 'HiFiShifterTestLegacy\models'
    $legacyModelRoot2 = Join-Path $env:APPDATA 'HiFiShifterTestLegacy2\models'
    $arpRegPath = 'HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter'
    New-Item -ItemType Directory -Force (Join-Path $legacyModelRoot $manifest.version) | Out-Null
    Set-Content -LiteralPath (Join-Path $legacyModelRoot "$($manifest.version)\models.json") -Value '{}'
    & reg.exe add $arpRegPath /v ModelRoot /t REG_SZ /d $legacyModelRoot /f | Out-Null
    Assert-That 'overwrite install with "keep models" clears the duplicate copy in the legacy root' {
        $keep2 = Invoke-InstallerProcess $testInstaller @('/S', '/DeleteModels=0')
        if ($keep2.ExitCode -ne 0) { throw "overwrite install exit code $($keep2.ExitCode)" }
        if (Test-Path -LiteralPath $legacyModelRoot) { throw "legacy model root still exists: $legacyModelRoot" }
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw 'model library was not kept' }
        'legacy root merged and de-duplicated'
    }

    New-Item -ItemType Directory -Force (Join-Path $legacyModelRoot2 $manifest.version) | Out-Null
    Set-Content -LiteralPath (Join-Path $legacyModelRoot2 "$($manifest.version)\models.json") -Value '{}'
    & reg.exe add $arpRegPath /v ModelRoot /t REG_SZ /d $legacyModelRoot2 /f | Out-Null
    Assert-That 'overwrite install with "delete models" clears the legacy model root too' {
        $purge = Invoke-InstallerProcess $testInstaller @('/S', '/DeleteModels=1')
        if ($purge.ExitCode -ne 0) { throw "overwrite install exit code $($purge.ExitCode)" }
        if (Test-Path -LiteralPath $legacyModelRoot2) { throw "legacy model root still exists: $legacyModelRoot2" }
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw 'model library was not rewritten' }
        'legacy root purged'
    }

    # The final standalone uninstall also uses fresh markers to verify "delete models, keep cache"
    # defaults. /DeleteCache=1 may have emptied the logs directory; recreate the parent directories
    # before writing markers.
    New-Item -ItemType Directory -Force (Split-Path $modelMarker -Parent) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $disposableMarker -Parent) | Out-Null
    Set-Content -LiteralPath $modelMarker -Value 'old'
    Set-Content -LiteralPath $disposableMarker -Value 'old'

    Write-Host '  Uninstalling (silent)...' -ForegroundColor DarkGray
    # Pretend .hsp belongs to another program: the uninstaller may only clear .hshp entries still
    # pointing at this install, and must not touch it. Use reg.exe instead of RegistryKey.SetValue:
    # keys returned by Get-Item are read-only handles.
    & reg.exe add "$classesRegPath\$foreignProgId\shell\open\command" /ve /d '"C:\Other\Other.exe" "%1"' /f | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "cannot create the foreign ProgID (reg add exit code $LASTEXITCODE)" }
    & reg.exe add "$classesRegPath\.hsp" /ve /d $foreignProgId /f | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "cannot write the foreign .hsp association (reg add exit code $LASTEXITCODE)" }

    New-Item -ItemType Directory -Force (Split-Path $disposableMarker -Parent) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $userDataMarker -Parent) | Out-Null
    Set-Content -LiteralPath $disposableMarker -Value 'disposable'
    Set-Content -LiteralPath $userDataMarker -Value 'keep'
    $uninstaller = Join-Path $installDir 'uninstall.exe'
    if (!(Test-Path $uninstaller)) { throw 'missing uninstall.exe' }
    Invoke-InstallerProcess $uninstaller @('/S') | Out-Null
    # The uninstaller copies itself to a temp directory before running, so its exit code is not
    # reliable under -Wait; judge by directory and registry state instead.
    # [Why poll instead of a fixed 3-second sleep] On slow machines / under real-time antivirus
    # scanning, 3 seconds may only get halfway through the uninstall, and the later assertions would
    # misread "not finished deleting yet" as "not fully deleted". Poll until both the App directory
    # and the ARP key are gone (60-second cap) before continuing; on timeout, do not throw — the
    # itemized assertions below produce a failure precise to the entry.
    $uninstallDone = $false
    for ($i = 0; $i -lt 120; $i++) {
        if (!(Test-Path $installDir) -and !(Test-Path $arpKey)) { $uninstallDone = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (!$uninstallDone) {
        Write-Warning 'uninstall still left residue after 60 seconds (App directory / ARP key); continuing with assertions to locate the specific items.'
    }

    Assert-That 'App directory gone after uninstall' {
        if (Test-Path $installDir) { throw "directory still exists: $installDir" }
        'gone'
    }
    Assert-That 'CLAP files gone after uninstall' {
        if (Test-Path (Join-Path $clapDir 'HiFiShifter.clap')) { throw 'HiFiShifter.clap still exists' }
        if (Test-Path $clapResDir) { throw "plugin resource directory still exists: $clapResDir" }
        'module + shared resources gone'
    }
    Assert-That 'VST3 directory gone after uninstall' {
        if (Test-Path (Join-Path $vst3Dir 'HiFiShifter.vst3')) { throw 'plugin directory still exists' }
        'gone'
    }
    Assert-That 'ARP key gone after uninstall' {
        if (Test-Path $arpKey) { throw "registry key still exists: $arpKey" }
        'gone'
    }
    Assert-That 'uninstall removes only its own associations, keeping foreign default programs' {
        $hshp = Get-RegDefault "$classesRoot\.hshp"
        if ($hshp -eq 'HiFiShifter.Project') { throw '.hshp still bound to HiFiShifter.Project' }
        if (Test-Path -LiteralPath "$classesRoot\HiFiShifter.Project") {
            throw 'HiFiShifter.Project ProgID still exists'
        }
        if (Test-Path -LiteralPath "$classesRoot\Applications\HiFiShifter.exe") {
            throw 'Applications\HiFiShifter.exe still exists'
        }
        $hsp = Get-RegDefault "$classesRoot\.hsp"
        if ($hsp -ne $foreignProgId) { throw ".hsp wrongly modified: $hsp" }
        if (!(Test-Path -LiteralPath "$classesRoot\$foreignProgId")) {
            throw "$foreignProgId wrongly deleted"
        }
        if ($null -ne (Get-RegValue "$classesRoot\.hshp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hshp\OpenWithProgids still contains HiFiShifter.Project'
        }
        if ($null -ne (Get-RegValue "$classesRoot\.hsp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hsp\OpenWithProgids still contains HiFiShifter.Project'
        }
        'own removed, foreign kept'
    }
    # [Why the model library default assertion is "deleted"] "Delete models" is checked by default:
    # the model library is written only by this product, and it keeping occupying disk after uninstall
    # defies user expectations. Cache and logs stay kept by default (a reinstall reuses the cache
    # while troubleshooting, and the logs are the only clue for locating the last failure).
    Assert-That 'uninstall deletes the shared model library by default (kept only when unchecked)' {
        if (Test-Path -LiteralPath $modelMarker) { throw "model library marker was not deleted: $modelMarker" }
        'deleted'
    }
    Assert-That 'uninstall keeps cache and logs by default' {
        if (!(Test-Path -LiteralPath $disposableMarker)) { throw "derivatives marker wrongly deleted: $disposableMarker" }
        if (!(Test-Path -LiteralPath $userDataMarker)) { throw "user data marker wrongly deleted: $userDataMarker" }
        'kept'
    }
}
finally {
    Remove-Item -LiteralPath "$classesRoot\$foreignProgId" -Recurse -Force -ErrorAction SilentlyContinue
    Restore-AssociationBackups
    Remove-GateDirectory $testRoot
    Remove-Item $testInstaller -Force -ErrorAction SilentlyContinue
    Remove-Item $disposableMarker, $userDataMarker -Force -ErrorAction SilentlyContinue
    # The test-mode model root is gate-only; clear the whole directory, leaving no .staging.
    Remove-GateDirectory (Join-Path $env:APPDATA 'HiFiShifterTest')
}

Write-Host ''
if ($failed.Count) {
    Write-Host "$($failed.Count) installer check(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host 'Installer gate passed.' -ForegroundColor Green
