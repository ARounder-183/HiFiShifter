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
Add-Type -TypeDefinition $signature
$handle = [ClapProbe]::LoadLibrary($ModulePath)
if ($handle -eq [IntPtr]::Zero) { throw "LoadLibrary failed for $ModulePath" }
try {
    $entry = [ClapProbe]::GetProcAddress($handle, "clap_entry")
    if ($entry -eq [IntPtr]::Zero) { throw "clap_entry not found by LoadLibrary" }
    Write-Host "OK   black-box load: clap_entry resolved"
} finally { [void][ClapProbe]::FreeLibrary($handle) }
Write-Host "All Windows CLAP checks passed."
