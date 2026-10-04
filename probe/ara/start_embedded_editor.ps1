# 内嵌GUI隔离启动；绝不启动HiFiShifter.exe，不改系统PATH，也不向已有REAPER送脚本。
param([switch]$Reopen)
$ErrorActionPreference = 'Stop'
$araEmbedRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (Get-Process reaper -ErrorAction SilentlyContinue) { throw 'REAPER is running; preserve the project and close normally before this isolated launch.' }
$araEmbedVst = Join-Path $araEmbedRoot '.build-tmp\embedded-vst3'
$araEmbedBundle = Join-Path $araEmbedVst 'HiFiShifter.vst3\Contents\x86_64-win\HiFiShifter.vst3'
if (!(Test-Path -LiteralPath $araEmbedBundle)) { throw 'Build the embedded bundle first.' }
$araEmbedScratch = Join-Path $araEmbedRoot '.build-tmp\embedded-probe'
$araEmbedProfile = Join-Path $araEmbedScratch 'profile'
New-Item -ItemType Directory -Force -Path $araEmbedProfile | Out-Null
$araEmbedIni = Join-Path $araEmbedProfile 'REAPER.ini'
if (!(Test-Path -LiteralPath $araEmbedIni)) {
    # 复用一次性profile的扫描缓存以避免第三方激活弹窗；不复制用户profile或大量主题。
    foreach ($araEmbedCache in @('reaper-vstplugins64.ini','reaper-vstplugins.ini','reaper-vst3.ini')) {
        $araEmbedSource = Join-Path $PSScriptRoot "reaper-profile\task10-clean\$araEmbedCache"
        if (Test-Path -LiteralPath $araEmbedSource) { Copy-Item -LiteralPath $araEmbedSource -Destination (Join-Path $araEmbedProfile $araEmbedCache) }
    }
    $araEmbedText = "[REAPER]`r`nvstpath64=$araEmbedVst`r`nrenderclosewhendone=4`r`nvstfullstate=49989`r`n"
    [IO.File]::WriteAllText($araEmbedIni,$araEmbedText,[Text.UTF8Encoding]::new($false))
}
$env:HIFISHIFTER_ARA_INSTANCE_DIR = Join-Path $araEmbedScratch 'instances'
$env:HIFISHIFTER_ARA_LOG = Join-Path $PSScriptRoot 'captures\embedded-editor-plugin.log'
# 构建时绝对assets覆盖不能漏入验收，必须从实际模块bundle查找frontend。
Remove-Item Env:HIFISHIFTER_ARA_EDITOR_ASSETS -ErrorAction SilentlyContinue
$araEmbedArgs = @('-cfgfile', $araEmbedIni)
if ($Reopen) {
    $araEmbedProject = Join-Path $araEmbedScratch 'embedded-editor.RPP'
    if (!(Test-Path -LiteralPath $araEmbedProject)) { throw 'Save the disposable embedded project before reopening.' }
    $araEmbedArgs += $araEmbedProject
} else { $araEmbedArgs += '-new' }
$araEmbedArgs += (Join-Path $PSScriptRoot 'build_embedded_editor_probe.lua')
# 用户目标是可见FX内原GUI；这是明确可见的交互验收窗口，不是后台辅助服务。
Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' -WindowStyle Normal -ArgumentList $araEmbedArgs -WorkingDirectory $araEmbedScratch
