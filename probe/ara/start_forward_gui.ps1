# 正向ARA GUI本地启动入口；只启动隔离REAPER和本worktree的HiFiShifter，不安装到系统。
param([switch]$Reopen, [switch]$NoGui)
$ErrorActionPreference = 'Stop'
$araGuiRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$araGuiBuilds = @(
    (Join-Path $araGuiRoot 'backend\target\debug\HiFiShifter.exe'),
    (Join-Path $araGuiRoot 'backend\target\ara-fix-app\debug\HiFiShifter.exe')
)
# 独立target允许保留用户旧窗口完成新构建；关闭后选最新已完成的本地GUI产物。
$araGuiExe = $araGuiBuilds | Where-Object { Test-Path -LiteralPath $_ } | Get-Item |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$NoGui -and !$araGuiExe) { throw 'Build the HiFiShifter app before launching REAPER.' }
if (!$NoGui) {
    $araGuiRunning = Get-CimInstance Win32_Process -Filter "Name = 'HiFiShifter.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($araGuiRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) }
    if ($araGuiRunning) { throw 'This worktree GUI is running; preserve your edits and close it normally before switching builds.' }
}
if (Get-Process reaper -ErrorAction SilentlyContinue) { throw 'REAPER is running; close it normally before starting the isolated profile.' }
$araGuiScratch = Join-Path $araGuiRoot '.build-tmp\gui-probe'
New-Item -ItemType Directory -Force $araGuiScratch | Out-Null
$env:HIFISHIFTER_ARA_INSTANCE_DIR = Join-Path $araGuiScratch 'instances'
$env:HIFISHIFTER_ARA_LOG = Join-Path $PSScriptRoot 'captures\forward-gui-plugin.log'
$env:WEBVIEW2_USER_DATA_FOLDER = Join-Path $araGuiScratch 'webview'
$araGuiVst = Join-Path $PSScriptRoot 'vst3'
Copy-Item -LiteralPath (Join-Path $araGuiRoot 'backend\target\debug\hifishifter_plugin.dll') -Destination (Join-Path $araGuiVst 'HiFiShifter.vst3') -Force
foreach ($araGuiDll in @('onnxruntime.dll','DirectML.dll','SoundTouchDLL.dll')) {
    $araGuiDllSource = Join-Path $araGuiRoot "backend\target\debug\$araGuiDll"
    if (Test-Path -LiteralPath $araGuiDllSource) { Copy-Item -LiteralPath $araGuiDllSource -Destination (Join-Path $araGuiVst $araGuiDll) -Force }
}
$araGuiArgs = @('-cfgfile', (Join-Path $PSScriptRoot 'reaper-profile\task10-clean\REAPER.ini'))
if ($Reopen) { $araGuiArgs += (Join-Path $araGuiScratch 'forward-gui.RPP') } else { $araGuiArgs += '-new' }
$araGuiArgs += (Join-Path $PSScriptRoot 'build_forward_gui_probe.lua')
# 本轮明确需要用户可见的交互窗口，其余后台构建仍使用Hidden。
Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' -WindowStyle Normal -ArgumentList $araGuiArgs -WorkingDirectory $araGuiVst
if (!$NoGui) {
    Start-Process -FilePath $araGuiExe.FullName -WindowStyle Normal -WorkingDirectory $araGuiExe.DirectoryName `
        -ArgumentList @("--log-file=$(Join-Path $PSScriptRoot 'captures\forward-gui-app.log')")
}
