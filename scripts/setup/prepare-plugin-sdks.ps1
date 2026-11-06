# 按 scripts/env/plugin-sdks.json 钉死的 revision 准备 ARA / VST3 SDK 检出。
#
# 顺序：缓存里已是 pin 的那份 → 本地镜像（离网可用）→ 联网浅克隆。
# 「是 pin 的那份」= commit、tree、工作区干净、子模块身份全部一致。
[CmdletBinding()]
param([switch]$WithTestkit, [switch]$Offline)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\env\sdk-env.ps1')
$sdkCache = Get-SdkCacheRoot
$sdkMirror = Get-SdkMirrorRoot
$sdkOffline = $Offline -or (Test-SdkOfflineRequested)
$sdkPins = Get-Content (Join-Path $PSScriptRoot '..\env\plugin-sdks.json') -Raw | ConvertFrom-Json

function Test-SdkIdentity($path, $commit, $tree) {
    if (!(Test-Path -LiteralPath $path)) { return $false }
    if ((& git -C $path rev-parse HEAD 2>$null) -ne $commit) { return $false }
    if ((& git -C $path rev-parse 'HEAD^{tree}' 2>$null) -ne $tree) { return $false }
    return -not (& git -C $path status --porcelain --untracked-files=all 2>$null)
}

function Test-SdkPin($path, $pin) {
    if (!(Test-SdkIdentity $path $pin.commit $pin.tree)) { return $false }
    # CLAP 是纯头文件 SDK，没有子模块；缺该字段时只校验主检出。
    if (!$pin.submodule) { return $true }
    Test-SdkIdentity (Join-Path $path $pin.submodule) $pin.submoduleCommit $pin.submoduleTree
}

function Invoke-Git($what, $arguments) {
    & git @arguments
    if ($LASTEXITCODE) { throw "$what 失败" }
}

function Get-SdkCheckout($name, $pin) {
    $target = Join-Path $sdkCache $name
    if (Test-SdkPin $target $pin) {
        Write-Host "SDK 就绪（缓存）：$name"
        return
    }

    $mirrored = Join-Path $sdkMirror $name
    if (Test-SdkPin $mirrored $pin) {
        Write-Host "SDK 从本地镜像导入：$mirrored"
        Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
        Copy-Item $mirrored $target -Recurse
        if (!(Test-SdkPin $target $pin)) { throw "镜像导入后与 pin 不一致：$name" }
        return
    }

    if ($sdkOffline) {
        throw "离网模式下没有 $name 的可用检出：缓存与镜像里都没有 pin 的那一份（见 README「第三方源码」）。"
    }

    Write-Host "SDK 下载：$name @ $($pin.commit)"
    Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
    Invoke-Git 'git init' @('init', $target)
    Invoke-Git 'git remote add' @('-C', $target, 'remote', 'add', 'origin', $pin.repository)
    Invoke-Git 'git fetch' @('-C', $target, 'fetch', '--depth', '1', 'origin', $pin.commit)
    Invoke-Git 'git checkout' @('-C', $target, '-c', 'core.autocrlf=false', 'checkout', '--detach', 'FETCH_HEAD')
    if ($pin.submodule) {
        Invoke-Git 'git submodule' @('-C', $target, '-c', 'core.autocrlf=false', 'submodule', 'update', '--init', '--depth', '1', '--', $pin.submodule)
    }
    if (!(Test-SdkPin $target $pin)) { throw "下载的检出与 pin 不一致：$name" }
}

foreach ($entry in @(@('ARA_SDK', $sdkPins.ara), @('vst3sdk', $sdkPins.vst3), @('clap', $sdkPins.clap))) {
    Get-SdkCheckout $entry[0] $entry[1]
}

# 只有 ara2-bridge-testkit 的 cpp-interop 需要 ARA_Library / ARA_Examples；
# 产品构建不需要，所以默认不拉。--recursive 连带 ARA_Examples 嵌套的 3rdParty。
if ($WithTestkit) {
    $araPath = Join-Path $sdkCache 'ARA_SDK'
    $araPin = $sdkPins.ara
    if ($sdkOffline) {
        foreach ($sub in $araPin.testkitSubmodules) {
            if (!(Test-SdkIdentity (Join-Path $araPath $sub.path) $sub.commit $sub.tree)) {
                throw "离网模式下 ARA testkit 子模块缺失或不在 pin 上：$($sub.path)（用 -WithTestkit 导出镜像）"
            }
        }
    }
    else {
        $roots = @($araPin.testkitSubmodules | Where-Object { $_.path -notmatch '/' } | ForEach-Object { $_.path })
        $updateArgs = @('-C', $araPath, '-c', 'core.autocrlf=false', 'submodule', 'update', '--init', '--depth', '1', '--recursive', '--') + $roots
        Invoke-Git 'testkit 子模块下载' $updateArgs
        foreach ($sub in $araPin.testkitSubmodules) {
            if (!(Test-SdkIdentity (Join-Path $araPath $sub.path) $sub.commit $sub.tree)) {
                throw "ARA testkit 子模块与 pin 不一致：$($sub.path)"
            }
        }
    }
}
