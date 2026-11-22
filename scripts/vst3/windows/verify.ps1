# Black-box load gate: prove this bundle can really be loaded by a VST3 host.
#
# Three checks; any failure means the host would see "0 classes" or hang outright:
#   1. the shim exports GetPluginFactory / InitDll / ExitDll;
#   2. the shim depends on KERNEL32 only (CRT statically linked, so no runtime DLLs are
#      dragged into the host process);
#   3. a separate process performs a real dlopen and calls GetPluginFactory — a deadlock
#      must fail here, not hang the build or CI.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BundleDirectory,
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'verify-frontend.ps1') -FrontendDirectory (Join-Path ([IO.Path]::GetFullPath($BundleDirectory)) 'Contents\Resources\frontend')
$module = Join-Path ([IO.Path]::GetFullPath($BundleDirectory)) 'Contents\x86_64-win'
$shim = Join-Path $module 'HiFiShifter.vst3'
foreach ($required in $shim, "$module\HiFiShifterEngine.dll", "$module\SoundTouchDLL.dll", "$module\DirectML.dll") {
    if (!(Test-Path -LiteralPath $required)) { throw "bundle is missing $required" }
}
foreach ($tool in 'dumpbin.exe', 'cl.exe') {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is not on PATH: dot-source . .\scripts\env\msvc-env.ps1 first"
    }
}

$exports = & dumpbin.exe /nologo /exports $shim
if ($LASTEXITCODE) { throw "dumpbin /exports failed: $shim" }
foreach ($entry in 'GetPluginFactory', 'InitDll', 'ExitDll') {
    if (!($exports | Select-String -Pattern $entry -SimpleMatch -Quiet)) { throw "shim is missing export: $entry" }
}

$dependents = & dumpbin.exe /nologo /dependents $shim
if ($LASTEXITCODE) { throw "dumpbin /dependents failed: $shim" }
$unexpected = @($dependents | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } |
    ForEach-Object { $_.Trim() } | Where-Object { $_ -ne 'KERNEL32.dll' })
if ($unexpected.Count) { throw "shim pulls in unexpected runtime dependencies: $($unexpected -join ', ')" }

$work = Join-Path ([IO.Path]::GetTempPath()) ("hfs-probe-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $work | Out-Null
try {
    $exe = Join-Path $work 'load_probe.exe'
    & cl.exe /nologo /utf-8 /std:c++17 /EHsc /O2 /MT (Join-Path $PSScriptRoot 'load_probe.cpp') "/Fo:$work\load_probe.obj" "/Fe:$exe"
    if ($LASTEXITCODE) { throw 'load_probe.cpp compilation failed' }

    # Output is redirected to files instead of a pipe: once a pipe's buffer fills, the child
    # blocks on write, and that "fake timeout" would be misdiagnosed as a GetPluginFactory
    # deadlock.
    $stdout = Join-Path $work 'out.txt'
    $stderr = Join-Path $work 'err.txt'
    $probe = Start-Process $exe -ArgumentList ('"' + $shim + '"') -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if (!$probe.WaitForExit($TimeoutSeconds * 1000)) {
        $probe.Kill()
        throw "load probe did not return within $TimeoutSeconds seconds; GetPluginFactory very likely deadlocked (log: $stdout)"
    }
    if ($probe.ExitCode) {
        throw "load probe failed (exit code $($probe.ExitCode)): $((Get-Content $stdout, $stderr -Raw -ErrorAction SilentlyContinue) -join ' ')"
    }
}
finally {
    $resolvedWork = [IO.Path]::GetFullPath($work)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolvedWork.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected probe cleanup path' }
    Remove-Item -LiteralPath $resolvedWork -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "VST3 load gate passed: $shim" -ForegroundColor Green
