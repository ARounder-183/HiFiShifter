[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BundleDirectory,
    [int]$TimeoutSeconds = 20,
    [string]$ResultPath = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$bundle = [IO.Path]::GetFullPath($BundleDirectory)
$moduleDir = Join-Path $bundle 'Contents\x86_64-win'
$shim = Join-Path $moduleDir 'HiFiShifter.vst3'
$engine = Join-Path $moduleDir 'HiFiShifterEngine.dll'
$soundTouch = Join-Path $moduleDir 'SoundTouchDLL.dll'
$directMl = Join-Path $moduleDir 'DirectML.dll'

foreach ($required in @($shim, $engine, $soundTouch, $directMl)) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required VST3 bundle file: $required"
    }
}

$dumpbin = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
if (!$dumpbin) {
    throw 'dumpbin.exe not found. Run . .\scripts\env\msvc-env.ps1 in this PowerShell session first.'
}
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (!$cl) {
    throw 'cl.exe not found. Run . .\scripts\env\msvc-env.ps1 in this PowerShell session first.'
}

# 1. Static export check: the shim must expose the VST3 module entry points.
$exports = & $dumpbin.Source /nologo /exports $shim
if ($LASTEXITCODE -ne 0) { throw "dumpbin /exports failed for $shim" }
foreach ($name in @('GetPluginFactory', 'InitDll', 'ExitDll')) {
    if (-not ($exports | Select-String -Pattern $name -SimpleMatch -Quiet)) {
        throw "Native VST3 loader is missing export: $name"
    }
}

# 2. The shim itself must stay CRT-static; only KERNEL32 is expected.
$dependents = & $dumpbin.Source /nologo /dependents $shim
if ($LASTEXITCODE -ne 0) { throw "dumpbin /dependents failed for $shim" }
$shimDependencies = @(
    $dependents |
        Where-Object { $_ -match '^\s+\S+\.dll\s*$' } |
        ForEach-Object { $_.Trim() }
)
$unexpected = @($shimDependencies | Where-Object { $_ -notmatch '^KERNEL32\.dll$' })
if ($unexpected.Count -gt 0) {
    throw ("Native VST3 loader has unexpected runtime dependencies: " + ($unexpected -join ', '))
}

# 3. Build the black-box probe into a private temp directory.
$tempDir = Join-Path ([IO.Path]::GetTempPath()) ('hfs-load-probe-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
$probeExe = Join-Path $tempDir 'load_probe.exe'
$probeObj = Join-Path $tempDir 'load_probe.obj'
try {
    & $cl.Source /nologo /utf-8 /std:c++17 /EHsc /O2 /MT (Join-Path $PSScriptRoot 'load_probe.cpp') "/Fo:$probeObj" "/Fe:$probeExe"
    if ($LASTEXITCODE -ne 0) { throw 'load_probe.cpp failed to compile' }

    # 4. Run the probe with a hard timeout. A deadlocked GetPluginFactory must
    #    fail this gate, not hang the build or CI job.
    $logDir = Join-Path $repoRoot '.build-tmp'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stdoutPath = Join-Path $logDir 'vst3-load-gate.log'
    $stderrPath = Join-Path $logDir 'vst3-load-gate.err.log'

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $probeExe
    $psi.Arguments = '"' + $shim + '"'
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    if (-not $process.Start()) { throw 'Failed to start load_probe.exe' }

    # 【为什么先起异步读再等退出】管道缓冲只有 ~4KB：先 WaitForExit 再 ReadToEnd
    # 的顺序里，子进程一旦写满缓冲就会阻塞在写上，WaitForExit 必然超时 ——
    # 而这个假超时会被下面的报错误诊成 "GetPluginFactory deadlocked"。
    # 读与等必须并行；Kill 之后 ReadToEndAsync 也总能完成（管道随进程终止而关）。
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
    if ($timedOut) {
        try { $process.Kill() } catch { }
        $process.WaitForExit()
    }

    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($stdoutPath, $stdout, $utf8)
    [IO.File]::WriteAllText($stderrPath, $stderr, $utf8)

    if ($timedOut) {
        throw "VST3 load probe timed out after $TimeoutSeconds s. This usually means GetPluginFactory deadlocked. See $stdoutPath and $stderrPath"
    }
    if ($process.ExitCode -ne 0) {
        throw "VST3 load probe failed with exit code $($process.ExitCode). See $stdoutPath and $stderrPath"
    }

    if (!$ResultPath) {
        $ResultPath = Join-Path $logDir 'native-load-gate.json'
    }
    $ResultPath = [IO.Path]::GetFullPath($ResultPath)
    New-Item -ItemType Directory -Path (Split-Path -Parent $ResultPath) -Force | Out-Null
    $result = [ordered]@{
        passed         = $true
        timestamp      = [DateTime]::UtcNow.ToString('o')
        probeVersion   = 1
        bundlePath     = $bundle
        shim           = $shim
        timeoutSeconds = $TimeoutSeconds
        probeExitCode  = $process.ExitCode
        logPath        = $stdoutPath
        stderrPath     = $stderrPath
    }
    [IO.File]::WriteAllText($ResultPath, ($result | ConvertTo-Json -Depth 4), $utf8)
    Write-Output "VST3 load gate passed: $shim"
}
finally {
    if (Test-Path -LiteralPath $tempDir) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
