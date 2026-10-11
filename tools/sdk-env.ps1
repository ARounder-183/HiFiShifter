# tools/sdk-env.ps1
#
# 插件 SDK（ARA_SDK / vst3sdk）检出的唯一事实来源。
#
# 这里只回答两个问题：检出放在哪、消费方要设哪两个环境变量。
#   - 「拉哪个仓库 / 哪个 commit / 哪个 tree」由 `tools/plugin-sdks.json` 钉死；
#   - 下载与校验由 `tools/prepare-plugin-sdks.ps1` 完成。
#
# 所有构建脚本与 CI workflow 都从这里取路径，避免同一路径散落在十来个文件里
# （此前散在 3 个脚本 + 2 个 workflow + 一致性守卫 + 文档里）。
#
# 用法（点源后调用）：
#   . .\tools\sdk-env.ps1
#   Set-SdkEnvironment            # 设置 ARA_SDK_DIR / ARA_VST3_SDK_DIR
#   $root = Get-SdkCacheRoot      # 取缓存根目录

# 缓存根：仓库根下的 third_party/sdk。
#
# 为什么整目录 gitignore、构建期才拉取：按 tools/plugin-sdks.json 钉死的 revision
# 下载以控仓库体积，并履行许可声明义务（见根 LICENSE 的 Third-Party Notices：
# ARA SDK 为 Apache-2.0；VST3 SDK 自 3.8.0 起为 MIT，本仓 pin 的正是 3.8.0）。
# 检出放在这里、按需下载。
function Get-SdkCacheRoot {
    # SDK 缓存根目录（绝对路径）。
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\third_party\sdk'))
}

function Get-SdkEnvironmentMap {
    # ara2-bridge 的 build.rs 读取的环境变量名 -> 缓存下的子目录名。
    [ordered]@{
        'ARA_SDK_DIR'      = 'ARA_SDK'
        'ARA_VST3_SDK_DIR' = 'vst3sdk'
    }
}

function Set-SdkEnvironment {
    <#
    .SYNOPSIS
        在当前进程设置 ARA_SDK_DIR / ARA_VST3_SDK_DIR，供 cargo build.rs 使用。
    .PARAMETER Force
        覆盖已存在的同名环境变量。默认保留调用方已设置的值（便于 CI 与开发者覆盖）。
    #>
    param([switch]$Force)
    $sdkCacheRoot = Get-SdkCacheRoot
    $sdkEnvironmentMap = Get-SdkEnvironmentMap
    foreach ($sdkName in $sdkEnvironmentMap.Keys) {
        if ($Force -or !(Test-Path "Env:$sdkName")) {
            Set-Item -Path "Env:$sdkName" -Value (Join-Path $sdkCacheRoot $sdkEnvironmentMap[$sdkName])
        }
    }
}

function Export-SdkEnvironment {
    <#
    .SYNOPSIS
        把同样的赋值写入 $GITHUB_ENV，供 CI 后续 step 继承（GITHUB_ENV 只对后续 step 生效）。
    #>
    if (!$env:GITHUB_ENV) { throw 'Export-SdkEnvironment 只能在 CI 中调用（未设置 GITHUB_ENV）。' }
    $sdkCacheRoot = Get-SdkCacheRoot
    $sdkEnvironmentMap = Get-SdkEnvironmentMap
    foreach ($sdkName in $sdkEnvironmentMap.Keys) {
        Add-Content -LiteralPath $env:GITHUB_ENV -Value "$sdkName=$(Join-Path $sdkCacheRoot $sdkEnvironmentMap[$sdkName])"
    }
}
