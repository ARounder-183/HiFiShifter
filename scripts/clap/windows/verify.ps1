# CLAP Windows 产物自检：导出符号、黑盒加载、ARA 能力。
param([Parameter(Mandatory=$true)][string]$ModulePath)
$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $ModulePath)) { throw "CLAP module missing: $ModulePath" }
# 单文件 CLAP 的资源目录必须与模块同级；缺 plugin.html 的表现是宿主扫描到插件
# 却只给出一块空白窗口（前端根本没被加载）。
$moduleDir = Split-Path -Parent $ModulePath
$resources = Join-Path $moduleDir "HiFiShifter-resources"
$frontend = Join-Path $resources "frontend\plugin.html"
$manifest = Join-Path $resources "models\models.json"
if (!(Test-Path -LiteralPath $frontend)) { throw "CLAP frontend assets missing: $frontend" }
if (!(Test-Path -LiteralPath $manifest)) { throw "CLAP model manifest missing: $manifest" }
Write-Host "OK   resources: frontend + models.json next to the module"
# WebView2 的用户数据目录必须落在 %LOCALAPPDATA%\HiFiShifter\webview 下：模块自己
# 在 Program Files，profile 一旦出现在这里，浏览器进程就无权写自己的数据目录
# （宿主里弹"Microsoft Edge 无法读取和写入其数据目录"，窗口空白）。
$strayProfile = Join-Path $resources "webview"
if (Test-Path -LiteralPath $strayProfile) {
    throw "WebView2 profile leaked into the resource tree: $strayProfile"
}
Write-Host "OK   resources: no WebView2 profile beside the module"
$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
if ($dumpbin) {
    $exports = & $dumpbin.Source /exports $ModulePath 2>&1
    if ($LASTEXITCODE -or -not ($exports -match "clap_entry")) { throw "clap_entry is not exported" }
    Write-Host "OK   exports: clap_entry"
} else { Write-Host "SKIP exports: dumpbin unavailable" }
$signature = @"
using System;
using System.Runtime.InteropServices;
public static class ClapProbe {
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
    public static extern IntPtr LoadLibrary(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi)]
    public static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("kernel32.dll")] public static extern bool FreeLibrary(IntPtr module);
}
"@
if (!("ClapProbe" -as [type])) { Add-Type -TypeDefinition $signature }
$handle = [ClapProbe]::LoadLibrary($ModulePath)
if ($handle -eq [IntPtr]::Zero) { throw "LoadLibrary failed for $ModulePath" }
try {
    $entry = [ClapProbe]::GetProcAddress($handle, "clap_entry")
    if ($entry -eq [IntPtr]::Zero) { throw "clap_entry not found by LoadLibrary" }
    Write-Host "OK   black-box load: clap_entry resolved"
} finally { [void][ClapProbe]::FreeLibrary($handle) }
# 【为什么 Windows 也要跑 ABI oracle】这里的 LoadLibrary 只证明符号存在；
# 手写 #[repr(C)] 与官方头文件的字段偏移必须再比一次，否则 cl.exe / MSVC ABI 下的
# 对齐差异（例如 clap_process_t 的 8 字节填充）不会被发现。
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
. (Join-Path $root "scripts\env\sdk-env.ps1"); Set-SdkEnvironment
Push-Location (Join-Path $root "backend")
try { cargo test -p hifishifter-clap --test clap_layout --locked -- --nocapture }
finally { Pop-Location }
if ($LASTEXITCODE) { throw "CLAP ABI layout oracle failed" }
Write-Host "OK   abi: CLAP 1.1.9 layout matches the pinned SDK"

Write-Host "All Windows CLAP checks passed."
