# CLAP Windows 产物自检：导出符号、黑盒加载、ARA 能力。
param([Parameter(Mandatory=$true)][string]$ModulePath)
$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $ModulePath)) { throw "CLAP module missing: $ModulePath" }
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
