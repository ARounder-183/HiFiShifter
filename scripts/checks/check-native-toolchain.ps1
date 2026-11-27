# One CMake generator per environment: the invariant scripts/env/native-toolchain.ps1 exists for.
#
# [Why a gate] The failure it prevents is a *build-time* error that names neither the cause nor
# the file that caused it:
#
#   CMake Error: Error: generator : Ninja
#   Does not match the generator used previously: NMake Makefiles
#
# It is reproducible only on a machine whose PATH differs from the author's, and it costs a full
# rebuild to discover. Three separate things had to be true at once for it to happen (opusic-sys
# probing ninja, msvc-env.ps1 skipping its assignment on the already-imported branch, and a cargo
# entry point that imported no env script at all), so no single code review would have caught it.
#
# Three assertions, each pinning one of those three:
#   1. both branches of msvc-env.ps1 leave CMAKE_GENERATOR set, and equal to HFS_NATIVE_GENERATOR;
#   2. every script that can start a cargo build imports a toolchain env script;
#   3. Clear-StaleCmakeCaches really drops a tree recorded under another generator, and only that
#      tree (out/lib beside it must survive).
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))

if ([IO.Path]::DirectorySeparatorChar -ne '\') {
    Write-Host 'SKIP native toolchain gate: not a Windows host (no MSVC / NMake involved)'
    return
}

$failures = New-Object System.Collections.Generic.List[string]

function Assert-True([bool]$condition, [string]$message) {
    if (!$condition) { $script:failures.Add($message) }
}

# ── 1. Both branches of msvc-env.ps1 pin the same generator ─────────────────
#
# The probe deliberately clears the "MSVC already imported" evidence before the first import so
# the real vcvars branch runs, then imports a second time to exercise the early-return branch.
# Both must end at the same answer: the early return is the branch a developer hits every day
# (x64 Native Tools prompt) and the one that used to skip the assignment entirely.
$msvcEnv = Join-Path $root 'scripts\env\msvc-env.ps1'
$probe = Join-Path ([IO.Path]::GetTempPath()) ('hfs-toolchain-' + [Guid]::NewGuid().ToString('N') + '.ps1')
$body = @(
    '$ErrorActionPreference = "Stop"'
    '$env:VSCMD_VER = ""'
    '$env:INCLUDE = ""'
    '$env:LIB = ""'
    '$env:LIBPATH = ""'
    # An explicit choice must win over the PATH probe; this is the documented override and the
    # rollback switch, so it is asserted rather than assumed.
    '$env:HFS_NATIVE_GENERATOR = "NMake"'
    ('. "' + $msvcEnv + '"')
    'Write-Output ("FRESH=" + $env:CMAKE_GENERATOR + "|" + $env:HFS_NATIVE_GENERATOR)'
    ('. "' + $msvcEnv + '"')
    'Write-Output ("REPEAT=" + $env:CMAKE_GENERATOR + "|" + $env:HFS_NATIVE_GENERATOR)'
    '$ninja = Get-Command ninja -CommandType Application -ErrorAction SilentlyContinue'
    'Write-Output ("NINJA=" + [bool]$ninja)'
)
[IO.File]::WriteAllLines($probe, $body, [Text.Encoding]::UTF8)
try {
    $pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (!$pwsh) { $pwsh = (Get-Process -Id $PID).Path }
    $output = @(& $pwsh -NoProfile -ExecutionPolicy Bypass -File $probe 2>&1)
    if ($LASTEXITCODE) { throw "toolchain probe failed (exit code $LASTEXITCODE): $($output -join ' ')" }
}
finally {
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
}

$readings = @{}
foreach ($line in $output) {
    if ($line -match '^(FRESH|REPEAT|NINJA)=(.*)$') { $readings[$Matches[1]] = $Matches[2] }
}
foreach ($key in 'FRESH', 'REPEAT') {
    Assert-True ($readings.ContainsKey($key)) "msvc-env.ps1 probe printed no $key reading"
}
if ($readings.ContainsKey('FRESH')) {
    $fresh = $readings['FRESH'] -split '\|'
    Assert-True ($fresh[0] -eq 'NMake Makefiles') ("msvc-env.ps1 (vcvars branch) left CMAKE_GENERATOR='" + $fresh[0] + "', expected 'NMake Makefiles'")
    Assert-True ($fresh[1] -eq 'NMake') ("msvc-env.ps1 (vcvars branch) left HFS_NATIVE_GENERATOR='" + $fresh[1] + "', expected 'NMake'")
}
if ($readings.ContainsKey('REPEAT')) {
    $repeat = $readings['REPEAT'] -split '\|'
    Assert-True ($repeat[0] -eq 'NMake Makefiles') ("msvc-env.ps1 (already-imported branch) left CMAKE_GENERATOR='" + $repeat[0] + "' -- this is the branch an x64 Native Tools prompt takes, and it used to set nothing at all")
    Assert-True ($repeat[1] -eq 'NMake') ("msvc-env.ps1 (already-imported branch) left HFS_NATIVE_GENERATOR='" + $repeat[1] + "'")
}
if ($readings.ContainsKey('NINJA')) {
    # With NMake pinned, opusic-sys' "ninja --version" probe must fail, otherwise it would
    # override CMAKE_GENERATOR with Ninja and split the build across two generators again.
    Assert-True ($readings['NINJA'] -eq 'False') 'ninja is still reachable after pinning the NMake generator: opusic-sys would override it'
}

# ── 1b. The target architecture decides which vcvars entry point runs ───────
#
# [Why source-level] Which arm64 host/target combination exists depends on the VS payload of
# the machine, so a runtime assertion would be machine-dependent (this dev box has no arm64
# toolset at all). What must hold everywhere is the *decision*: x64 keeps calling
# vcvars64.bat (byte-compatible with the old behaviour), and arm64 goes through vcvarsall
# with the three documented fallbacks. Without that, the Windows arm64 CI leg (tauri-action,
# no build.ps1) pins NMake Makefiles and finds no nmake.exe -- run 45953527512.
$msvcText = [IO.File]::ReadAllText($msvcEnv)
Assert-True ($msvcText.Contains('HFS_MSVC_TARGET_ARCH')) 'msvc-env.ps1 no longer honours HFS_MSVC_TARGET_ARCH, so the Windows arm64 leg cannot select its toolset'
Assert-True ($msvcText.Contains('VC\Auxiliary\Build\vcvarsall.bat')) 'msvc-env.ps1 must discover VC\Auxiliary\Build\vcvarsall.bat: vcvars64.bat can only target x64'
foreach ($arch in 'arm64', 'amd64_arm64', 'x86_arm64') {
    Assert-True ($msvcText.Contains("'$arch'")) ("msvc-env.ps1 lost the '$arch' vcvarsall candidate; the arm64 leg would have nothing to fall back to")
}
Assert-True ($msvcText.Contains('Join-Path $vcvarsDir ''vcvars64.bat''')) 'msvc-env.ps1 must keep calling vcvars64.bat for x64 so that machine behaviour is unchanged'
Assert-True ($msvcText.Contains('VSCMD_ARG_TGT_ARCH')) 'msvc-env.ps1 must verify the selected target: vcvarsall can exit 0 while leaving the x64 toolset on PATH'

# ── 2. Every cargo entry point imports a toolchain env script ───────────────
#
# Criterion: a script that mentions cargo outside a comment must import msvc-env.ps1 or
# android-env.ps1 (both of which end in Set-NativeToolchain). Files under scripts\env\ ARE the
# toolchain; install_deps_windows.ps1 only runs "cargo install", which compiles nothing from this
# repository, so neither needs the import.
$exempt = @(
    (Join-Path $root 'scripts\env'),
    (Join-Path $root 'scripts\setup\install_deps_windows.ps1')
)
foreach ($script in (Get-ChildItem -Path (Join-Path $root 'scripts') -Recurse -Filter '*.ps1' -File)) {
    $skip = $false
    foreach ($prefix in $exempt) {
        if ($script.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { $skip = $true; break }
    }
    if ($skip) { continue }
    $mentionsCargo = $false
    foreach ($line in [IO.File]::ReadLines($script.FullName)) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('#')) { continue }
        if ($trimmed -match '\bcargo\b') { $mentionsCargo = $true; break }
    }
    if (!$mentionsCargo) { continue }
    $text = [IO.File]::ReadAllText($script.FullName)
    $importsToolchain = $text.Contains('msvc-env.ps1') -or $text.Contains('android-env.ps1')
    Assert-True $importsToolchain ($script.FullName.Substring($root.Length + 1) + ' runs cargo but imports neither msvc-env.ps1 nor android-env.ps1, so its CMake generator is whatever the caller''s shell happens to have')
}

# ── 3. The stale-cache sweep is real ───────────────────────────────────────
#
# Built in a temp tree, never against the repository's target directory: a gate that deletes
# build outputs to prove it works would be a gate nobody dares run.
$work = Join-Path ([IO.Path]::GetTempPath()) ('hfs-toolchain-sweep-' + [Guid]::NewGuid().ToString('N'))
try {
    $stale = Join-Path $work 'debug\build\opusic-sys-0123456789abcdef\out\build'
    $fresh = Join-Path $work 'debug\build\soundtouch-0123456789abcdef\out\soundtouch_build'
    $payload = Join-Path $work 'debug\build\opusic-sys-0123456789abcdef\out\lib'
    New-Item -ItemType Directory -Force -Path $stale, $fresh, $payload | Out-Null
    # Double-quoted so the `n is a real newline: a single-quoted '`n' would leave a literal
    # backtick-n glued to the generator name and make the sweep look broken.
    [IO.File]::WriteAllText((Join-Path $stale 'CMakeCache.txt'), "CMAKE_GENERATOR:INTERNAL=Ninja`n")
    [IO.File]::WriteAllText((Join-Path $fresh 'CMakeCache.txt'), "CMAKE_GENERATOR:INTERNAL=NMake Makefiles`n")
    # out/lib stands for the installed headers/libs several crates keep beside the build tree; the
    # sweep must not take them with it.
    [IO.File]::WriteAllText((Join-Path $payload 'libopus.a'), 'keep-me')

    . (Join-Path $root 'scripts\env\native-toolchain.ps1')
    $dropped = Clear-StaleCmakeCaches -TargetDir $work -Generator NMake
    Assert-True ($dropped -eq 1) "Clear-StaleCmakeCaches dropped $dropped trees, expected exactly 1 (the Ninja one)"
    Assert-True (!(Test-Path -LiteralPath $stale)) 'the stale (Ninja) build tree survived the sweep'
    Assert-True (Test-Path -LiteralPath $fresh) 'the matching (NMake) build tree was dropped'
    Assert-True (Test-Path -LiteralPath (Join-Path $payload 'libopus.a')) 'the sweep deleted out/lib beside the build tree'
}
finally {
    $resolved = [IO.Path]::GetFullPath($work)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if ($resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($failures.Count -gt 0) {
    throw ("native toolchain gate failed:`n  - " + ($failures -join "`n  - "))
}
Write-Host 'native toolchain gate OK: one generator per environment, and stale caches self-heal'
