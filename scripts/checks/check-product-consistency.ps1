# Product consistency guard: cross-file conventions that "must all hold at once" and that have no
# compile-time protection.
#
# When any of them breaks, nothing reports an error — the user just sees "wrong version shown",
# "models loaded the old ones", "controls do nothing when clicked". This script moves them into CI.
#
#   .\scripts\checks\check-product-consistency.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$failed = [System.Collections.Generic.List[string]]::new()

function Test-Invariant($name, [scriptblock]$body) {
    try {
        $detail = & $body
        Write-Host "  [OK]   $name" -ForegroundColor Green
        if ($detail) { Write-Host "         $detail" -ForegroundColor DarkGray }
    }
    catch {
        Write-Host "  [FAIL] $name" -ForegroundColor Red
        Write-Host "         $($_.Exception.Message)" -ForegroundColor Red
        $failed.Add($name)
    }
}

function Get-RepoText($relative) {
    Get-Content -LiteralPath (Join-Path $root $relative) -Raw
}

# Unified external tool invocation: when the tool is not installed, give one actionable sentence
# (instead of a raw error), and move the exit-code check here too — previously every call site wrote
# its own `if ($LASTEXITCODE)`, and missing one meant a silent pass.
#
# [Why -RequireOutput] On Windows `python` may be the Microsoft Store App Execution Alias: the
# command exists, exits 0, and prints nothing (it just pops the Store). Then `$output` is empty,
# `Test-Invariant`'s `($output | Select-Object -Last 1)` gets $null, the UI still prints [OK],
# and every python criterion "passes vacuously". Validation scripts must leave a conclusion line;
# no output counts as failure.
function Invoke-Tool([string]$tool, [string[]]$arguments, [switch]$RequireOutput) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if (!$command) {
        throw "$tool not found (see scripts/setup/install_deps_windows.ps1 / install_deps_macos.sh / install_deps_linux.sh)"
    }
    $output = & $command.Source @arguments 2>&1
    if ($LASTEXITCODE) {
        throw "$tool failed (exit code $LASTEXITCODE): $($output -join ' | ')"
    }
    if ($RequireOutput -and !($output | Where-Object { "$_".Trim() })) {
        throw "$tool produced no conclusion (possibly the Microsoft Store python alias; the script never really ran)"
    }
    $output
}

Write-Host 'Product consistency checks' -ForegroundColor Cyan

# Version numbers are scattered across several files (see scripts/release/version-sources.json);
# missing one does not break the build, it just makes some UI report a different version — or makes
# the plugin show another version number in "Programs and Features". Writes and checks share
# scripts/release/version.py, comparing each occurrence listed in the manifest: the second spot in
# package-lock and the two entries in Cargo.lock all count.
Test-Invariant 'product version consistent across all sources' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/release/version.py'), '--check') -RequireOutput
    ($output | Where-Object { $_ } | Select-Object -Last 1)
}

# models.json's version is derived from the model contents; at runtime it decides whether the copy
# in the shared model library is usable. When the weights are not in a clean clone, --check-manifest
# prints "check skipped" and returns 0 — that is the normal state, not a regression (reconciling the
# manifest with the weights is only possible once the weights have been fetched).
Test-Invariant 'model manifest matches the real model files' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/setup/fetch-models.py'), '--check-manifest') -RequireOutput
    ($output | Where-Object { $_ } | Select-Object -Last 1)
}

Test-Invariant 'installer strings match the five-language glossary' {
    Invoke-Tool 'node' @((Join-Path $root 'scripts/release/build-installer-strings.mjs'), '--check') | Out-Null
    'scripts/release/installer/installer_strings.nsh'
}

# The license texts are the "missing one and nothing errors" kind: the packaging script just copies
# a directory, and the file names appear only in the README. A rename or a missing file means some
# artifact silently drops a required attribution — so the manifest is pinned here.
Test-Invariant 'bundled license manifest complete (LICENSE + licenses\ directory)' {
    $required = @(
        'LICENSE',
        'licenses\README.md',
        'licenses\COPYING.LGPL-2.1',
        'licenses\Apache-2.0.txt',
        'licenses\NSF-HiGAN-vocoder-NOTICE.txt',
        'licenses\GAME-NOTICE.txt'
    )
    foreach ($file in $required) {
        if (!(Test-Path -LiteralPath (Join-Path $root $file))) { throw "missing bundled license file: $file" }
    }
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    if ($nsi -notmatch '(?m)^\s*File\s+/r\s+"\$\{HFS_LICENSES\}\\\*"\s*$') {
        throw 'merged-installer.nsi no longer installs the full licenses\ directory'
    }
    $license = Get-RepoText 'LICENSE'
    if ($license -notmatch 'CLAP \(CLever Audio Plugin\) headers') { throw 'LICENSE does not credit the CLAP SDK third-party license' }
    if ($license -notmatch 'GAME - Generative Adaptive MIDI Extractor') {
        throw 'GAME model license not credited in LICENSE'
    }
    'LICENSE + licenses\ (LGPL-2.1 / Apache-2.0 / CC BY-NC-SA notices)'
}

# These are all places where hand-reverting silently drops a user-visible capability: no uninstall
# entry, repeated installs piling up assets, models going back to one copy per bundle.
# An overwrite install must **never** run the old uninstaller again: the old uninstall.exe is a
# binary built by the previous version, and whether it is truly silent under /S is not for this
# script to decide (the Tauri uninstaller pops a language picker, and the host-in-use prompt is a
# MessageBox without /SD). Running it = the user sees an extra "HiFiShifter uninstall" window during
# install. The uninstall action must be done in-process by this installer (HfsRemovePreviousInstall);
# no child-process window may appear anywhere on the overwrite-install path.
Test-Invariant 'overwrite install never runs the old uninstaller (no second uninstall window)' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    # Only non-comment lines: the comment mentioning ExecWait is explaining "why this must not be done".
    $code = @(($nsi -split "`n") | Where-Object { $_ -notmatch '^\s*;' })
    $launchers = @($code | Where-Object { $_ -match '^\s*(ExecWait|ExecShell|Exec)\b.*(uninstall|Uninstaller)' })
    if ($launchers.Count) {
        throw "merged-installer.nsi still launches the old uninstaller: $($launchers[0].Trim())"
    }
    foreach ($forbidden in 'HfsOldUninstaller', 'HfsParseUninstallString') {
        if ($nsi -match $forbidden) { throw "merged-installer.nsi still references $forbidden" }
    }
    # The shared model library's location changes with the installer version: the root must be written
    # into the uninstall entry and the uninstaller must read it back, otherwise "uninstalled but the
    # models are still there" or "the kept models are never read by the new version".
    if ($nsi -notmatch '"ModelRoot"') {
        throw 'merged-installer.nsi does not write the shared model library root into the uninstall entry (ModelRoot)'
    }
    if ($nsi -notmatch 'ReadRegStr \$HfsModelRoot') {
        throw 'merged-installer.nsi uninstaller does not read the shared model library root back from the uninstall entry'
    }
    if ($nsi -notmatch 'Function HfsRemovePreviousInstall') {
        throw 'merged-installer.nsi lacks HfsRemovePreviousInstall, the in-process removal of the previous version'
    }
    if ($nsi -match 'HFS_REINSTALL_CONFIRM|MB_YESNO') {
        throw 'merged-installer.nsi still pops a confirmation box outside the overwrite-install options page'
    }
    # Every MessageBox in silent install/uninstall must carry an /SD default, otherwise NSIS stops on
    # an invisible dialog and it shows up as "installer/uninstaller hangs".
    $boxes = @(($nsi -split "`n") | Where-Object { $_ -match '^\s*MessageBox' })
    $unguarded = @($boxes | Where-Object { $_ -notmatch '/SD' })
    if ($unguarded.Count) { throw "merged-installer.nsi has $($unguarded.Count) MessageBox(es) missing /SD" }
    'in-process removal, no confirmation box, all MessageBoxes carry /SD'
}

# CLAP's public plugin directory (Common Files\CLAP) is machine-wide: the spec says only the .clap
# file itself should appear there. Putting frontend/ and models.json next to it under generic names
# means users cannot tell ownership apart and dare not delete them — and they mix with other
# people's plugins in the same directory. Resources must land in the product's own shared directory.
Test-Invariant 'CLAP public directory holds only the plugin module' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    if ($nsi -notmatch 'HFS_CLAP_RES_DIR') {
        throw 'merged-installer.nsi does not point plugin resources at the product-shared directory (HFS_CLAP_RES_DIR)'
    }
    if ($nsi -match 'SetOutPath "\$HfsClapDir\\\$\{CLAP_RESOURCES\}"') {
        throw 'merged-installer.nsi still installs resources into the public CLAP directory'
    }
    if ($nsi -notmatch 'RMDir /r "\$HfsClapDir\\\$\{CLAP_RESOURCES\}"') {
        throw 'merged-installer.nsi does not clean up the resource folder the old version left in the public CLAP directory on upgrade'
    }
    'only HiFiShifter.clap in Common Files\CLAP'
}

Test-Invariant 'merged installer registers the uninstall entry, cleans managed directories, models use the shared library' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    foreach ($required in 'WriteUninstaller', 'UninstallString', 'InstallLocation', 'ModelRoot', 'MODELS_VERSION', 'APP_VERSION', 'FILEASSOC_PROGID', 'Software\Classes') {
        if ($nsi -notmatch [regex]::Escape($required)) { throw "merged-installer.nsi missing $required" }
    }
    if ($nsi -notmatch 'RMDir /r .*Resources\\frontend') {
        throw 'merged-installer.nsi no longer cleans Contents\Resources\frontend — repeated installs pile up assets again'
    }
    if ($nsi -notmatch 'RMDir /r "\$INSTDIR\\models"') {
        throw 'merged-installer.nsi no longer cleans models\ inside the install directory — upgrading from the old installer permanently keeps a second weights copy'
    }
    'WriteUninstaller / UninstallString / directory reconciliation / shared model library'
}

# The merged installer's default branch must be "elevated + HKLM + Common Files": the HFS_TEST_MODE
# branch exists for the test-mode closed loop (installs to the current user, ARP goes to HKCU). If
# anyone makes the test branch the default, the release artifact installs into the user profile and
# cannot be found in "Programs and Features".
Test-Invariant 'merged installer defaults to per-machine install' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    $ifdef = $nsi.IndexOf('!ifdef HFS_TEST_MODE')
    $else = $nsi.IndexOf('!else', $ifdef)
    $endif = $nsi.IndexOf('!endif', $else)
    if ($ifdef -lt 0 -or $else -lt 0 -or $endif -lt 0) { throw 'merged-installer.nsi HFS_TEST_MODE branch is incomplete' }
    $release = $nsi.Substring($else, $endif - $else)
    foreach ($required in 'RequestExecutionLevel admin', '!define HFS_ARP_ROOT HKLM', '!define HFS_CLASSES_ROOT HKLM', 'COMMONFILES64') {
        if ($release -notmatch [regex]::Escape($required)) { throw "release branch missing $required" }
    }
    'admin / HKLM / CommonFiles64 / Classes'
}

# Windows ships a single installer: the build script must invoke the installer itself with
# --no-bundle, and the bundle carries only models.json (the installer writes the weights into
# ProgramData). If either condition breaks, the user pays for the same weights twice on
# disk — while the artifact looks perfectly normal.
Test-Invariant 'Windows has one installer and the models install only once' {
    $build = Get-RepoText 'scripts/build/build.ps1'
    if ($build -notmatch '--no-bundle') { throw 'build.ps1 no longer uses --no-bundle (the Tauri bundler would push the models back into the App directory)' }
    if ($build -notmatch 'merged-installer\.nsi') { throw 'build.ps1 no longer calls merged-installer.nsi' }
    if ($build -match 'Copy-Item\s+"\$modelsSource\\\$(model|\*)') {
        throw 'build.ps1 is copying model weights into the bundle again'
    }
    'build.ps1 → merged-installer.nsi (--no-bundle)'
}

# Criteria are allowed to live only in scripts. Seeing --skip / --test-threads / --offline in
# workflow YAML means someone is "patching in place" again — which is exactly where skip drift
# starts (the same test skipped on one pipeline and not on another, with nobody knowing which one
# counts). `--skip` must stand alone: build-tool flags like build.sh's `--skip-frontend` are not
# test criteria and were false-positives here since 3ae1e1048 introduced them.
Test-Invariant 'workflows contain no inline test criteria' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root '.github\workflows') -Filter *.yml -File) {
        foreach ($line in Get-Content -LiteralPath $file.FullName) {
            # Mentioning these words in comments is allowed (explaining history); the criterion only
            # looks at code.
            $code = ($line -split '#', 2)[0]
            if ($code -match '--skip(?!-)|--test-threads|--offline') { $offenders += "$($file.Name): $($line.Trim())" }
        }
    }
    if ($offenders.Count) { throw ('criteria belong in scripts/checks/: ' + ($offenders -join ' | ')) }
    '.github/workflows'
}

# GitHub expressions only accept single-quoted strings. Written as `if: runner.os == "Windows"`,
# the **entire workflow file is judged invalid**: triggering it yields only a 0-second failure
# record whose name is the file path, with no step ever run. This pit has been hit three times
# (build.yml / vst3-unix.yml / clap.yml), so the criterion is "no double quotes on if: lines" —
# the conditions never need them anyway.
Test-Invariant 'workflow conditions use single-quoted strings' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root '.github\workflows') -Filter *.yml -File) {
        foreach ($line in Get-Content -LiteralPath $file.FullName) {
            if ($line -notmatch '^\s*if:') { continue }
            if ($line.Contains('"')) { $offenders += "$($file.Name): $($line.Trim())" }
        }
    }
    if ($offenders.Count) { throw ('strings on if: lines must be single-quoted: ' + ($offenders -join ' | ')) }
    '.github/workflows'
}


# The frontend invoke.ts switch is the full set of method names the frontend can send. If the
# backend on either side fails to dispatch any one of them, the symptom is "the control is
# clickable but nothing happens" — silent breakage, with no compile error either.
Test-Invariant 'every frontend invoke surface is backed by a backend command' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/checks/check-command-gating.py')) -RequireOutput
    ($output | Select-Object -First 1)
}

# Windows PowerShell 5.1 decodes BOM-less UTF-8 files as ANSI: the last byte of a Chinese comment
# pairs with the trailing newline into a double-byte character — the newline is swallowed, and the
# next line of code becomes part of the comment. The symptom is "some variable is mysteriously
# null", with the error pointing one line further down. Pure-ASCII scripts do not need a BOM, so
# the criterion is "contains non-ASCII and has no BOM".
Test-Invariant 'PowerShell scripts containing non-ASCII all carry a UTF-8 BOM' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root 'scripts') -Recurse -File -Include *.ps1, *.psm1, *.psd1) {
        $bytes = [IO.File]::ReadAllBytes($file.FullName)
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { continue }
        $nonAscii = $false
        foreach ($byte in $bytes) { if ($byte -gt 0x7F) { $nonAscii = $true; break } }
        if ($nonAscii) { $offenders += $file.FullName.Substring($root.Length + 1) }
    }
    if ($offenders.Count) { throw ('these scripts contain non-ASCII without a UTF-8 BOM: ' + ($offenders -join ' | ')) }
    'scripts/'
}

# Python's stdout on Windows follows the ANSI code page (cp1252 on English runners, GBK on Chinese
# systems). Printing one Chinese sentence kills the process with UnicodeEncodeError, and not a
# single real conclusion reaches the log — the gate goes red over encoding, not over the rule it
# guards. This pit has been hit twice (the rate-limit log gate, the version gate), so the criterion
# is written as "any non-ASCII in the source means stdout must be pinned to UTF-8", instead of
# remembering it per script.
Test-Invariant 'Python scripts containing non-ASCII pin stdout to UTF-8' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root 'scripts') -Recurse -File -Include *.py) {
        $text = [IO.File]::ReadAllText($file.FullName, [Text.Encoding]::UTF8)
        if ($text -notmatch '[^\x00-\x7f]') { continue }
        # [Why both quote styles count] The invariant is "stdout pinned to UTF-8", not "which quote
        # style was used to pin it": both styles coexist in the repo (check-ffi-guards.py single
        # quotes, check-dialog-filters.py double quotes); accepting only single quotes would
        # false-positive an already-pinned script as unpinned — the gate would go red over quotes,
        # not encoding.
        if ($text -notmatch 'reconfigure\(encoding=[''"]utf-8[''"]') { $offenders += $file.FullName.Substring($root.Length + 1) }
    }
    if ($offenders.Count) { throw ('these scripts contain non-ASCII without pinning stdout to UTF-8: ' + ($offenders -join ' | ')) }
    'scripts/**/*.py'
}

Write-Host ''
if ($failed.Count) {
    Write-Host "$($failed.Count) check(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host 'All product consistency checks passed.' -ForegroundColor Green
