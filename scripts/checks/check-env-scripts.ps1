# Dot-sourced env scripts are only allowed to touch $env:* — they must not leave ordinary
# variables behind in the caller's scope.
#
# [Why this gate exists] Real incident: scripts/env/msvc-env.ps1 had a
# `foreach ($root in Get-VsInstallRoots)`, and it is **dot-sourced** — so the caller's (the build
# script's) `$root` (the repo root) was overwritten with the Visual Studio install directory, and
# every later `Join-Path $root ...` resolved under
# `C:\Program Files\Microsoft Visual Studio\...`. The symptom was the build failing at "models.json
# not found", with only that absurd path in the error — no hint that "the variable had been
# rewritten by a previously dot-sourced script".
#
# [Criterion] Set sentinel values for a group of common names in a child process, dot-source the
# env scripts, then read them back. If any of them changed, the script has written temporary
# variables into the caller's scope again.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$probe = Join-Path ([IO.Path]::GetTempPath()) ('hfs-env-scope-' + [Guid]::NewGuid().ToString('N') + '.ps1')
$names = @('root', 'lines', 'dump', 'vcvars', 'dist', 'profile', 'stage', 'candidate', 'found')
# The sentinels and the verification loop must both be generated from $names: previously the two
# places each hardcoded their own copy and $names itself was never read — adding a new sentinel
# name to $names had no effect, and the gate would quietly test one variable fewer.
# (Statements inside @() are newline-separated: elements coming out of a pipeline get flattened
# into the array, while comma separation would keep the nested array as a single element —
# WriteAllLines would then join it into one line.)
$body = @(
    $names | ForEach-Object { ('$' + $_ + ' = "SENTINEL-' + $_ + '"') }
    # Clear the "MSVC already imported" evidence so the importer's real branch runs (otherwise it
    # exits early and the leak is never tested).
    '$env:INCLUDE = ""'
    '$env:LIB = ""'
    '$env:LIBPATH = ""'
    '$env:VSCMD_VER = ""'
    ('$ErrorActionPreference = "Stop"')
    ('. "' + (Join-Path $root 'scripts\env\msvc-env.ps1') + '"')
    ('. "' + (Join-Path $root 'scripts\env\sdk-env.ps1') + '"')
    'Set-SdkEnvironment'
    ('foreach ($name in ' + (($names | ForEach-Object { '"' + $_ + '"' }) -join ',') + ') {')
    '  $value = Get-Variable -Name $name -ValueOnly -ErrorAction SilentlyContinue'
    '  Write-Output ($name + "=" + $value)'
    '}'
)
[IO.File]::WriteAllLines($probe, $body, [Text.Encoding]::UTF8)
try {
    $pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (!$pwsh) { $pwsh = (Get-Process -Id $PID).Path }
    $output = & $pwsh -NoProfile -ExecutionPolicy Bypass -File $probe 2>&1
    if ($LASTEXITCODE) { throw "probe process failed (exit code $LASTEXITCODE): $($output -join ' ')" }
}
finally {
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
}
$leaked = @()
foreach ($line in $output) {
    if ($line -notmatch '^([a-z]+)=(.*)$') { continue }
    $name = $Matches[1]
    $value = $Matches[2]
    if ($value -ne "SENTINEL-$name") { $leaked += "$name=$value" }
}
if ($leaked.Count -gt 0) {
    throw ("these variables were rewritten by dot-sourced env scripts (they must stay as function-local variables): " + ($leaked -join ', '))
}
Write-Host 'env scope gate OK: dot-sourced scripts only touch env vars'
