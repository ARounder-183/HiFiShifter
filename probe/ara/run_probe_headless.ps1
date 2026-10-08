<#
无头跑一个 REAPER 探针脚本，并把结果写到你指定的 JSON 路径。

【为什么需要它】REAPER 的命令行**没有**运行脚本的开关（实测，见 README「命令行不能跑
脚本」）。但它有一个启动钩子：资源目录下的 `Scripts\__startup.eel` 会在启动时执行，而
EEL 里可以调 `AddRemoveReaScript` 把任意 .lua 注册成 action、再用 `Main_OnCommand` 跑它。
于是"隔离 profile + 启动钩子"就能让探针在无人值守的情况下跑完——不必有人在 GUI 里点
Actions → Load → Run。

【隔离】所有写入都落在一个临时资源目录（`-cfgfile`），不碰 `%APPDATA%\REAPER`。临时目录
会先从用户资源目录**复制**一份 ini（不复制的话 REAPER 会认为这是全新便携安装，弹
"Would you like to scan system VST/CLAP/LV2 paths?" 模态框，启动钩子就永远轮不到执行）。

【注意】EEL 里的 REAPER API 是**不带 `reaper.` 前缀**的（JSFX 风格）；写 `reaper.X` 会编译
失败，而编译失败是静默的（钩子只是不执行）。

【-SetupEel】可选的 EEL 片段，在注册探针**之前**执行，用来搭宿主夹具（建轨、插媒体、挂
插件）。有些探针问的是宿主行为，得先有那样的工程状态才问得出来。只给 `-SetupEel` 而不给
`-Probe` 也合法：那就是"只搭夹具"，探针逻辑本身在插件里（例如 F-2 的
`probe_host_musical_content` 只能从插件侧读，见 README）。
#>
param(
    [string]$Probe = "",
    [string]$Out = "",
    [string]$SetupEel = "",
    [string[]]$Env = @(),
    [string]$Project = "",
    [string]$Reaper = "C:\Program Files\REAPER (x64)\reaper.exe",
    [int]$TimeoutSeconds = 90,
    # 哨兵出现后再等这么多秒才杀 REAPER。搭夹具时必须给，理由见下。
    [int]$SettleSeconds = 0
)

$ErrorActionPreference = "Stop"

if ($Probe -eq "" -and $SetupEel -eq "") { throw "give at least one of -Probe / -SetupEel" }
if ($Probe -ne "" -and -not (Test-Path -LiteralPath $Probe)) { throw "probe script not found: $Probe" }
if ($SetupEel -ne "" -and -not (Test-Path -LiteralPath $SetupEel)) { throw "setup eel not found: $SetupEel" }
if (-not (Test-Path -LiteralPath $Reaper)) { throw "reaper.exe not found: $Reaper" }
# 只搭夹具时没有探针产物，用钩子日志当"做完了"的信号。
$waitsForHookLog = ($Probe -eq "")

$scratch = Join-Path (Split-Path -Parent $PSScriptRoot) "..\.build-tmp\reaper-probe"
$scratch = [System.IO.Path]::GetFullPath($scratch)
if (Test-Path -LiteralPath $scratch) { Remove-Item -LiteralPath $scratch -Recurse -Force }
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $scratch "Scripts") -Force | Out-Null

# 复制用户的 ini：让隔离 profile 看起来像"已装好的安装"，避免便携安装的模态提问。
$userResource = [Environment]::GetFolderPath('ApplicationData') + "\REAPER"
foreach ($name in @("REAPER.ini", "reaper-vstplugins64.ini", "reaper-fxtags.ini", "reaper-mouse.ini",
                    "reaper-midihw.ini", "reaper-jsfx.ini", "reaper-recentfx.ini", "reaper-reginfo2.ini",
                    "reaper-install-rev.txt", "REAPER-wndpos.ini")) {
    $src = Join-Path $userResource $name
    if (Test-Path -LiteralPath $src) { Copy-Item -LiteralPath $src -Destination (Join-Path $scratch $name) -Force }
}

$iniPath = Join-Path $scratch "REAPER.ini"
$outPath = ""
if ($Probe -ne "") {
    if ($Out -eq "") { throw "-Probe needs -Out (where the probe writes its JSON)" }
    $outPath = [System.IO.Path]::GetFullPath($Out)
    # 探针用 io.open(path,"w") 写结果；父目录不存在时它会直接 assert 失败，脚本静默中止。
    New-Item -ItemType Directory -Path (Split-Path -Parent $outPath) -Force | Out-Null
    if (Test-Path -LiteralPath $outPath) { Remove-Item -LiteralPath $outPath -Force }
}

# EEL 启动钩子：先跑夹具片段，再把探针注册成 action 并运行它。
# 路径用正斜杠，免得 EEL 字符串里转义反斜杠。
$logEel = (Join-Path $scratch "hook.log").Replace('\', '/')
$setup = ""
if ($SetupEel -ne "") {
    # 【必须显式 -Encoding UTF8】Windows PowerShell 5.1 读无 BOM 文件时按 ANSI（中文系统
    # 上是 CP936）；夹具里的中文注释会被误解码，而 CP936 的前导字节在行尾会**吃掉换行**，
    # 于是下一行代码被并进注释里、整段静默不执行。仓库里 `tools/*.ps1` 的 BOM 规则是同一个
    # 坑的另一面（见 scripts/check-product-consistency.ps1 第 3 节）。
    $setup = (Get-Content -LiteralPath $SetupEel -Raw -Encoding UTF8)
    # EEL 没有取环境变量的办法（`os.getenv` 是 Lua 的，EEL 侧没有等价物），
    # 所以夹具里的仓库根路径用占位符写，由这里替换成正斜杠的绝对路径。
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path.Replace('\', '/')
    $setup = $setup.Replace('__REPO_ROOT__', $repoRoot)
}
$register = ""
if ($Probe -ne "") {
    $probeEel = $Probe.Replace('\', '/')
    $register = @"
id = AddRemoveReaScript(1, 0, "$probeEel", 1);
fprintf(f, "id=%.0f\n", id);
fclose(f);
Main_OnCommand(id, 0);
"@
} else {
    $register = "fclose(f);"
}
$eel = @"
$setup
f = fopen("$logEel", "w");
$register
"@
# 写成无 BOM 的 UTF-8：EEL 只需要字节，而注释里的中文不该被 `-Encoding ascii` 变成 `?`。
[System.IO.File]::WriteAllText((Join-Path $scratch "Scripts\__startup.eel"), $eel, (New-Object System.Text.UTF8Encoding $false))

foreach ($pair in $Env) {
    $eq = $pair.IndexOf("=")
    if ($eq -lt 1) { throw "env entry must look like NAME=VALUE: $pair" }
    Set-Item -Path ("Env:" + $pair.Substring(0, $eq)) -Value $pair.Substring($eq + 1)
}

$reaperArgs = @("-cfgfile", $iniPath, "-nosplash")
if ($Project -ne "") { $reaperArgs += $Project } else { $reaperArgs += "-new" }

Write-Host "[probe] launching $Reaper with resource dir $scratch"
$proc = Start-Process -FilePath $Reaper -ArgumentList $reaperArgs -PassThru

try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $hookLog = Join-Path $scratch "hook.log"
    $sentinel = if ($waitsForHookLog) { $hookLog } else { $outPath }
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 1000
        if (Test-Path -LiteralPath $sentinel) { break }
        if ($proc.HasExited) { break }
    }
    # 【为什么哨兵之后还要等】`TrackFX_AddByName` 只是把插件插进去；REAPER 建 ARA 文档、
    # 推模型、发播放区域都是**之后**异步发生的。夹具一写完哨兵就杀进程的话，插件日志里
    # 只会看到一半流程（实测：有时 `regionSequences=1` 都还没出现），于是"宿主没给
    # 音乐上下文"这种结论就分不清是真没有还是没跑到。
    if ($SettleSeconds -gt 0 -and -not $proc.HasExited) {
        Write-Host "[probe] settling for $SettleSeconds s before shutdown"
        Start-Sleep -Seconds $SettleSeconds
    }
}
finally {
    Get-Process -Name reaper -ErrorAction SilentlyContinue | Stop-Process -Force
}

if ($waitsForHookLog) {
    if (Test-Path -LiteralPath $hookLog) {
        Write-Host "[probe] fixture built (hook.log):"
        Get-Content -LiteralPath $hookLog
        exit 0
    }
    Write-Host "[probe] startup hook did not run (no hook.log)"
    exit 1
}

if (Test-Path -LiteralPath $outPath) {
    Write-Host "[probe] output: $outPath"
    Get-Content -LiteralPath $outPath
    exit 0
}

if (Test-Path -LiteralPath $hookLog) {
    Write-Host "[probe] hook fired but the probe produced no output:"
    Get-Content -LiteralPath $hookLog
} else {
    Write-Host "[probe] startup hook did not run (no hook.log)"
}
exit 1
