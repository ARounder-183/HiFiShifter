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
#>
param(
    [Parameter(Mandatory = $true)][string]$Probe,
    [Parameter(Mandatory = $true)][string]$Out,
    [string[]]$Env = @(),
    [string]$Project = "",
    [string]$Reaper = "C:\Program Files\REAPER (x64)\reaper.exe",
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Probe)) { throw "probe script not found: $Probe" }
if (-not (Test-Path -LiteralPath $Reaper)) { throw "reaper.exe not found: $Reaper" }

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
$outPath = [System.IO.Path]::GetFullPath($Out)
# 探针用 io.open(path,"w") 写结果；父目录不存在时它会直接 assert 失败，脚本静默中止。
New-Item -ItemType Directory -Path (Split-Path -Parent $outPath) -Force | Out-Null
if (Test-Path -LiteralPath $outPath) { Remove-Item -LiteralPath $outPath -Force }

# EEL 启动钩子：注册探针为 action，然后运行它。路径用正斜杠，免得 EEL 字符串里转义反斜杠。
$probeEel = $Probe.Replace('\', '/')
$logEel = (Join-Path $scratch "hook.log").Replace('\', '/')
$eel = @"
f = fopen("$logEel", "w");
id = AddRemoveReaScript(1, 0, "$probeEel", 1);
fprintf(f, "id=%.0f\n", id);
fclose(f);
Main_OnCommand(id, 0);
"@
Set-Content -LiteralPath (Join-Path $scratch "Scripts\__startup.eel") -Value $eel -Encoding ascii

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
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 1000
        if (Test-Path -LiteralPath $outPath) { break }
        if ($proc.HasExited) { break }
    }
}
finally {
    Get-Process -Name reaper -ErrorAction SilentlyContinue | Stop-Process -Force
}

if (Test-Path -LiteralPath $outPath) {
    Write-Host "[probe] output: $outPath"
    Get-Content -LiteralPath $outPath
    exit 0
}

$hookLog = Join-Path $scratch "hook.log"
if (Test-Path -LiteralPath $hookLog) {
    Write-Host "[probe] hook fired but the probe produced no output:"
    Get-Content -LiteralPath $hookLog
} else {
    Write-Host "[probe] startup hook did not run (no hook.log)"
}
exit 1
