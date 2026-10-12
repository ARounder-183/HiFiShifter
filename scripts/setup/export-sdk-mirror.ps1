# 导出本地 SDK 镜像：把已校验的 pin 检出（ARA_SDK / vst3sdk，以及 SoundTouch）整体拷进
# 镜像根，供离网机器取源。镜像只是"另一份可整体拷贝的本地缓存"，服务于同一开发者在无网络
# 环境下的构建，不是再分发渠道（许可说明见 README '第三方源码'）。
#
# 用法（联网机器上跑一次）：
#   .\scripts\setup\export-sdk-mirror.ps1
#   .\scripts\setup\export-sdk-mirror.ps1 -WithTestkit                          # 连 ARA testkit 子模块一起导出
#   .\scripts\setup\export-sdk-mirror.ps1 -MirrorDirectory E:\hfs-sdk-mirror    # 直接写到 U 盘 / 局域网共享
#
# 之后在离网机器上：
#   $env:HIFISHIFTER_SDK_MIRROR = 'E:\hfs-sdk-mirror'
#   .\scripts\setup\prepare-plugin-sdks.ps1 -Offline
#   .\pack-portable.bat -Offline
[CmdletBinding()]
param([string]$MirrorDirectory, [switch]$WithTestkit)
$ErrorActionPreference = 'Stop'
$exportRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
# 镜像根与缓存根由 scripts/env/sdk-env.ps1 统一定义（唯一事实来源）。
. (Join-Path $PSScriptRoot '..\env\sdk-env.ps1')
if ($MirrorDirectory) { $exportMirrorRoot = [IO.Path]::GetFullPath($MirrorDirectory) }
else { $exportMirrorRoot = Get-SdkMirrorRoot }
$exportCacheRoot = Get-SdkCacheRoot
$exportLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\env\plugin-sdks.json') -Raw | ConvertFrom-Json

function Copy-IntoMirror($Source, $Name) {
    $exportTarget = Join-Path $exportMirrorRoot $Name
    if (Test-Path -LiteralPath $exportTarget) { Remove-Item -LiteralPath $exportTarget -Recurse -Force }
    Copy-Item -LiteralPath $Source -Destination $exportTarget -Recurse -Force
    Write-Output "Exported $Name -> $exportTarget"
}

# 先用同一套校验确保缓存就是 pin 的那份（联网时该步会补齐；已就绪时不触网）。
$exportPrepareArgs = @{}
if ($WithTestkit) { $exportPrepareArgs['WithTestkit'] = $true }
& (Join-Path $PSScriptRoot 'prepare-plugin-sdks.ps1') @exportPrepareArgs

New-Item -ItemType Directory -Path $exportMirrorRoot -Force | Out-Null
foreach ($exportSdk in @('ARA_SDK', 'vst3sdk')) {
    Copy-IntoMirror (Join-Path $exportCacheRoot $exportSdk) $exportSdk
}

# SoundTouch 由内核 build.rs 自动克隆到 src-tauri/third_party/soundtouch-static/soundtouch；
# 镜像里放一份，离网时 build.rs 直接取用（见 build.rs::build_soundtouch）。导出前校验 HEAD
# 与 pin 一致 —— 镜像里不该出现一份来路不明的源。
$exportSoundtouch = Join-Path $exportRoot 'backend\src-tauri\third_party\soundtouch-static\soundtouch'
if (Test-Path -LiteralPath (Join-Path $exportSoundtouch 'CMakeLists.txt')) {
    $exportSoundtouchHead = & git -c safe.directory=* -C $exportSoundtouch rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or "$exportSoundtouchHead".Trim() -ne $exportLock.soundtouch.commit) {
        throw "SoundTouch HEAD '$("$exportSoundtouchHead".Trim())' does not match the pin $($exportLock.soundtouch.commit); refusing to export it."
    }
    Copy-IntoMirror $exportSoundtouch 'soundtouch'
}
else {
    Write-Output 'SoundTouch source not present; skipping its mirror entry (run a build once to clone it).'
}

Write-Output ''
Write-Output "Mirror ready: $exportMirrorRoot"
Write-Output 'On an offline machine, set HIFISHIFTER_SDK_MIRROR to this directory (or copy it to the'
Write-Output 'default mirror location), then run scripts/setup/prepare-plugin-sdks.ps1 -Offline. See README 第三方源码.'
