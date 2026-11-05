# 插件 SDK（ARA_SDK / vst3sdk）检出的唯一事实来源。
#
# 这里只回答三个问题：检出放在哪、本地镜像在哪、消费方要设哪两个环境变量。
#   - 拉哪个仓库 / 哪个 commit / 哪个 tree 由 scripts/env/plugin-sdks.json 钉死；
#   - 下载与校验由 scripts/setup/prepare-plugin-sdks.ps1 完成；
#   - 本地镜像由 scripts/setup/export-sdk-mirror.ps1 生成。
#
# 用法（点源后调用）：
#   . .\scripts\env\sdk-env.ps1
#   Set-SdkEnvironment            # 设置 ARA_SDK_DIR / ARA_VST3_SDK_DIR
#   Get-SdkCacheRoot              # 取缓存根
#   Get-SdkMirrorRoot             # 取本地镜像根（HIFISHIFTER_SDK_MIRROR 可覆盖）
#   Test-SdkOfflineRequested      # 是否禁止为取源联网（HIFISHIFTER_OFFLINE）

function Get-SdkCacheRoot {
    # 按 pin 下载到仓库内、被 gitignore 的 third_party/sdk：控体积，并履行许可声明义务。
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\third_party\sdk'))
}

function Get-SdkMirrorRoot {
    # 可整体拷贝的本地源缓存，形状与缓存根相同（含 ARA_SDK/、vst3sdk/、soundtouch/）。
    # 由能联网的机器导出后拷到离网机器，全程不触网。
    if ($env:HIFISHIFTER_SDK_MIRROR) {
        return [IO.Path]::GetFullPath($env:HIFISHIFTER_SDK_MIRROR)
    }
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\third_party\sdk-mirror'))
}

function Test-SdkOfflineRequested {
    "$env:HIFISHIFTER_OFFLINE".Trim() -in @('1', 'true', 'yes', 'on')
}

function Set-SdkEnvironment {
    # ara2-bridge 的 build.rs 与 audio_abi 的 ABI oracle 都读这两个变量。
    $sdkRoot = Get-SdkCacheRoot
    $env:ARA_SDK_DIR = Join-Path $sdkRoot 'ARA_SDK'
    $env:ARA_VST3_SDK_DIR = Join-Path $sdkRoot 'vst3sdk'
}
