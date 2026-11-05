# 黑盒加载门禁：证明这个 bundle 真的能被 VST3 宿主加载。
#
# 三条，任一条不过都说明宿主会看到"0 个类"或干脆卡死：
#   1. shim 导出 GetPluginFactory / InitDll / ExitDll；
#   2. shim 只依赖 KERNEL32（CRT 静态链接，不把运行时 DLL 拖进宿主进程）；
#   3. 起一个独立进程真的 dlopen 一次并调用 GetPluginFactory —— 死锁必须在这里失败，
#      而不是把构建或 CI 挂住。
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BundleDirectory,
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = 'Stop'
$module = Join-Path ([IO.Path]::GetFullPath($BundleDirectory)) 'Contents\x86_64-win'
$shim = Join-Path $module 'HiFiShifter.vst3'
foreach ($required in $shim, "$module\HiFiShifterEngine.dll", "$module\SoundTouchDLL.dll", "$module\DirectML.dll") {
    if (!(Test-Path -LiteralPath $required)) { throw "bundle 缺少 $required" }
}
foreach ($tool in 'dumpbin.exe', 'cl.exe') {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool 不在 PATH 上：先点源 . .\scripts\env\msvc-env.ps1"
    }
}

$exports = & dumpbin.exe /nologo /exports $shim
if ($LASTEXITCODE) { throw "dumpbin /exports 失败：$shim" }
foreach ($entry in 'GetPluginFactory', 'InitDll', 'ExitDll') {
    if (!($exports | Select-String -Pattern $entry -SimpleMatch -Quiet)) { throw "shim 缺少导出：$entry" }
}

$dependents = & dumpbin.exe /nologo /dependents $shim
if ($LASTEXITCODE) { throw "dumpbin /dependents 失败：$shim" }
$unexpected = @($dependents | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } |
    ForEach-Object { $_.Trim() } | Where-Object { $_ -ne 'KERNEL32.dll' })
if ($unexpected.Count) { throw "shim 引入了意外的运行时依赖：$($unexpected -join ', ')" }

$work = Join-Path ([IO.Path]::GetTempPath()) ("hfs-probe-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $work | Out-Null
try {
    $exe = Join-Path $work 'load_probe.exe'
    & cl.exe /nologo /utf-8 /std:c++17 /EHsc /O2 /MT (Join-Path $PSScriptRoot 'load_probe.cpp') "/Fo:$work\load_probe.obj" "/Fe:$exe"
    if ($LASTEXITCODE) { throw 'load_probe.cpp 编译失败' }

    # 输出重定向到文件而不是管道：管道缓冲满了子进程会阻塞在写上，
    # 那种"假超时"会被误诊成 GetPluginFactory 死锁。
    $stdout = Join-Path $work 'out.txt'
    $stderr = Join-Path $work 'err.txt'
    $probe = Start-Process $exe -ArgumentList ('"' + $shim + '"') -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if (!$probe.WaitForExit($TimeoutSeconds * 1000)) {
        $probe.Kill()
        throw "加载探针 $TimeoutSeconds 秒未返回，GetPluginFactory 很可能死锁（日志：$stdout）"
    }
    if ($probe.ExitCode) {
        throw "加载探针失败（退出码 $($probe.ExitCode)）：$((Get-Content $stdout, $stderr -Raw -ErrorAction SilentlyContinue) -join ' ')"
    }
}
finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "VST3 加载门禁通过：$shim" -ForegroundColor Green
