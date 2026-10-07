# 中文一次性原生复现：全新profile/工程/日志，绝不-nonewinst或向已有REAPER发送脚本。
param([Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9-]{1,64}$')][string]$ScratchName,
      [ValidateRange(1,2000)][int]$Count=100,
      [switch]$EmptyFolder,
      [switch]$UnicodeSources,[switch]$Stereo,
      [string]$ProjectFile,
      [string]$BundleParent='.build-tmp\deliveries\release-delivery-02')
$ErrorActionPreference='Stop'
$manyRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (Get-CimInstance Win32_Process -Filter "Name='reaper.exe'") {throw 'Preserve running REAPER; no script sent.'}
$manyBundle=[IO.Path]::GetFullPath((Join-Path $manyRoot $BundleParent))
if (!(Test-Path -LiteralPath (Join-Path $manyBundle 'HiFiShifter.vst3\Contents\x86_64-win\HiFiShifter.vst3'))) {throw 'Missing bundle'}
$manyScratch=Join-Path $manyRoot ".build-tmp\$ScratchName"
if (Test-Path -LiteralPath $manyScratch) {throw 'Use a fresh scratch; never overwrite a prior project.'}
$manyProfile=Join-Path $manyScratch 'profile';New-Item -ItemType Directory -Path $manyProfile | Out-Null
foreach ($manyCache in @('reaper-vstplugins64.ini','reaper-vstplugins.ini','reaper-vst3.ini')) {
    $manyCacheSource=Join-Path $PSScriptRoot "reaper-profile\task10-clean\$manyCache"
    if (Test-Path -LiteralPath $manyCacheSource) {Copy-Item -LiteralPath $manyCacheSource -Destination $manyProfile}
}
$manyIni=Join-Path $manyProfile 'REAPER.ini'
[IO.File]::WriteAllText($manyIni,"[REAPER]`r`nvstpath64=$manyBundle`r`nvstfullstate=49989`r`n",[Text.UTF8Encoding]::new($false))
$env:HIFISHIFTER_MANY_CLIPS_DIR=$manyScratch;$env:HIFISHIFTER_MANY_CLIPS_COUNT="$Count"
$env:HIFISHIFTER_MANY_EMPTY_FOLDER=if ($EmptyFolder) {'1'} else {'0'}
$env:HIFISHIFTER_MANY_UNICODE=if ($UnicodeSources) {'1'} else {'0'}
$env:HIFISHIFTER_MANY_STEREO=if ($Stereo) {'1'} else {'0'}
$env:HIFISHIFTER_ARA_LOG=Join-Path $manyScratch 'plugin.log';$env:HIFISHIFTER_ARA_INSTANCE_DIR=Join-Path $manyScratch 'instances'
Remove-Item Env:HIFISHIFTER_ARA_EDITOR_ASSETS -ErrorAction SilentlyContinue
$manyArguments=@('-cfgfile',$manyIni)
if ($ProjectFile) {
    $manyCopy=Join-Path $manyScratch 'many-clips.RPP'
    Copy-Item -LiteralPath $ProjectFile -Destination $manyCopy
    $manyArguments+=@($manyCopy,(Join-Path $PSScriptRoot 'move_parent_probe.lua'))
} else {$manyArguments+=@('-new',(Join-Path $PSScriptRoot 'many_clips_probe.lua'))}
$manyProcess=Start-Process -FilePath 'D:\Softwares\REAPER (x64)\reaper.exe' -WindowStyle Normal -PassThru -WorkingDirectory $manyScratch -ArgumentList $manyArguments
Write-Output "PID=$($manyProcess.Id) SCRATCH=$manyScratch CLIPS=$Count"
