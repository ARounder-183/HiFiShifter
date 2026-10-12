# 准备 ARA / VST3 SDK 检出，落地到 scripts/env/sdk-env.ps1 定义的缓存根（third_party/sdk）。
# 身份校验与 ara2-bridge-companion 0.3.0 保持一致；只下载构建所需子模块。
#
# 取源优先级（每个仓库各自判断）：
#   1. 缓存里已有一份与 pin 一致的干净检出 —— 直接用，不触网；
#   2. 缓存缺失或不是 pin 的那份，但本地镜像（scripts/setup/export-sdk-mirror.ps1 产物）里有 —— 从镜像拷贝；
#   3. 否则联网浅克隆。`-Offline`（或 HIFISHIFTER_OFFLINE=1）禁止第 3 步，快速失败并给出取源指引。
#
# 【为什么要有镜像】GitHub 不可达时构建不该"完全无法进行"：镜像由能联网的机器导出，
# 拷到离网机器后取源全程不触网。缓存损坏（脏工作区 / 换过 origin）也只从镜像修复，不联网。
#
# 默认只拉产品构建必需的子模块（ARA_API / pluginterfaces）。`-WithTestkit` 额外拉
# ARA_Library / ARA_Examples（及其嵌套 3rdParty）—— 只有 ara2-bridge-testkit 的
# cpp-interop 需要它们；产品构建不需要，故保持默认最小化。
[CmdletBinding()]
param([string]$SdkDirectory, [switch]$WithTestkit, [switch]$Offline, [string]$MirrorDirectory)
$ErrorActionPreference = 'Stop'
$sdkTaskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
# 缓存根与镜像根由 scripts/env/sdk-env.ps1 统一定义，避免路径散落在多个脚本与 CI workflow 里。
. (Join-Path $PSScriptRoot '..\env\sdk-env.ps1')
if (!$SdkDirectory) { $SdkDirectory = Get-SdkCacheRoot }
$script:sdkDirectory = [IO.Path]::GetFullPath($SdkDirectory)
if (!$script:sdkDirectory.StartsWith($sdkTaskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'SDK directory must be inside this checkout.'
}
if ($MirrorDirectory) { $script:sdkMirrorRoot = [IO.Path]::GetFullPath($MirrorDirectory) }
else { $script:sdkMirrorRoot = Get-SdkMirrorRoot }
$script:sdkOffline = $Offline -or (Test-SdkOfflineRequested)
$sdkTaskLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\env\plugin-sdks.json') -Raw | ConvertFrom-Json
function Assert-SdkIdentity($Path, $Commit, $Tree) {
    $sdkIdentityCommit = & git -C $Path rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $sdkIdentityCommit -ne $Commit) { throw "SDK commit mismatch: $Path" }
    $sdkIdentityTree = & git -C $Path rev-parse 'HEAD^{tree}'
    if ($LASTEXITCODE -ne 0 -or $sdkIdentityTree -ne $Tree) { throw "SDK tree mismatch: $Path" }
    $sdkIdentityChanges = & git -C $Path status --porcelain --untracked-files=all
    if ($LASTEXITCODE -ne 0 -or $sdkIdentityChanges) { throw "SDK checkout is dirty: $Path" }
}
function Test-SdkCheckout($Path) {
    # 目录本身必须被识别为仓库根：.git 受损时 git 会向上回退到外层仓库，
    # --show-toplevel 于是指向的不是 SDK 目录，这种情况要重建而不是跳过下载。
    if (!(Test-Path -LiteralPath $Path)) { return $false }
    $sdkTopLevel = & git -C $Path rev-parse --show-toplevel 2>$null | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or !$sdkTopLevel) { return $false }
    return ([IO.Path]::GetFullPath("$sdkTopLevel".Trim()) -eq [IO.Path]::GetFullPath($Path))
}
function Test-SdkIdentityOnly($Path, $Commit, $Tree) {
    # 单个目录是否就是 pin 的那份干净检出（不含子模块）。用于测试子模块与普通检出。
    # 先判存在：缺失的目录若直接交给 git，会因向上回退到外层仓库而打印一行 fatal，
    # 那只是噪音（结果同样是"不匹配"）。
    if (!(Test-Path -LiteralPath $Path)) { return $false }
    try { Assert-SdkIdentity $Path $Commit $Tree; return $true }
    catch { return $false }
}
function Test-SdkIdentityMatches($Path, $Pin) {
    # commit + tree + 工作区干净 + 子模块身份全部一致才算匹配。任一项不符返回 $false，
    # 由调用方决定是修远端、从镜像导入还是联网重建。
    if (!(Test-SdkIdentityOnly $Path $Pin.commit $Pin.tree)) { return $false }
    return Test-SdkIdentityOnly (Join-Path $Path $Pin.submodule) $Pin.submoduleCommit $Pin.submoduleTree
}
function Test-SdkRemoteMatches($Path, $Pin) {
    # origin 指向 pin 的仓库才算"可继续使用"：否则后续 fetch 会拉错源。
    $sdkRemote = & git -C $Path remote get-url origin 2>$null | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or !$sdkRemote) { return $false }
    return ($sdkRemote.TrimEnd('/').Replace('.git', '') -eq $Pin.repository.Replace('.git', ''))
}
function Set-SdkRemote($Path, $Pin) {
    # 内容已是 pin 的那份、只是 origin 不同（镜像 / 代理克隆）时，改写远端即可，不必重新下载。
    & git -C $Path remote set-url origin $Pin.repository
    if ($LASTEXITCODE -ne 0) { throw "Failed to point the checkout at $($Pin.repository)." }
}
function Import-SdkFromMirror($Name, $Target) {
    # 从本地镜像整体拷贝一份检出（含 .git），替代联网下载。拷贝后仍走同一套
    # Assert-SdkIdentity 校验 —— 镜像内容不对会被拦下，而不是悄悄参与构建。
    $sdkMirrorEntry = Join-Path $script:sdkMirrorRoot $Name
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    New-Item -ItemType Directory -Path $script:sdkDirectory -Force | Out-Null
    Copy-Item -LiteralPath $sdkMirrorEntry -Destination $Target -Recurse -Force
}
function Initialize-SdkCheckout($Path, $Pin) {
    # 网络步骤偶发失败（浅克隆子模块时遇到过）会中断整个构建；整体重建是幂等的，
    # 失败后清理残留再重试一次。
    for ($sdkAttempt = 1; ; $sdkAttempt++) {
        try { New-SdkCheckout $Path $Pin; return }
        catch {
            if ($sdkAttempt -ge 2) { throw }
            Write-Output "SDK checkout failed ($($_.Exception.Message)); retrying..."
            Start-Sleep -Seconds 3
        }
    }
}
function New-SdkCheckout($Path, $Pin) {
    # 按 pin 重建 checkout；目录可能只是残留（空目录或损坏的 .git），先清干净。
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
    & git init $Path
    if ($LASTEXITCODE -ne 0) { throw 'SDK initialization failed.' }
    & git -C $Path remote add origin $Pin.repository
    if ($LASTEXITCODE -ne 0) { throw 'SDK remote setup failed.' }
    & git -C $Path fetch --depth 1 origin $Pin.commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned SDK download failed.' }
    & git -C $Path -c core.autocrlf=false checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Pinned SDK checkout failed.' }
    & git -C $Path -c core.autocrlf=false submodule update --init --depth 1 -- $Pin.submodule
    if ($LASTEXITCODE -ne 0) { throw 'SDK submodule download failed.' }
}
function Ensure-TestkitSubmodules($Path, $Pin) {
    # 只有 ara2-bridge-testkit 的 cpp-interop 需要 ARA_Library / ARA_Examples。
    # 拉取是幂等的：已初始化且提交正确时是 no-op；缺失（含 ARA_Examples 嵌套的
    # 3rdParty）时补齐。`--recursive` 覆盖嵌套子模块；随后逐个校验 commit/tree，
    # 与产品 SDK 用同一套身份判据。
    if (!$Pin.testkitSubmodules) { return }
    if ($script:sdkOffline) {
        # 离线：不做任何网络操作，只校验镜像/缓存里是否已带齐（导出镜像时用 -WithTestkit）。
        # 按身份（commit/tree/干净）判断，而不是"目录存在"——空占位目录骗不过校验。
        $sdkTestkitMissing = @($Pin.testkitSubmodules | Where-Object {
                !(Test-SdkIdentityOnly (Join-Path $Path $_.path) $_.commit $_.tree)
            } | ForEach-Object { $_.path })
        if ($sdkTestkitMissing.Count) {
            throw (@(
                "Offline mode: ARA testkit submodules are missing or not at the pin: $($sdkTestkitMissing -join ', ').",
                "  Export the mirror with -WithTestkit on a networked machine (scripts/setup/export-sdk-mirror.ps1 -WithTestkit)."
            ) -join "`n")
        }
        return
    }
    # `git submodule update` 只认超项目（ARA_SDK）直接登记的子模块路径；嵌套的
    # 3rdParty 归 ARA_Examples 管，传进来会报 "pathspec did not match"。因此只把
    # 顶层路径交给 update，靠 `--recursive` 连带拉取嵌套子模块，随后逐个校验身份。
    $sdkTestkitRoots = @($Pin.testkitSubmodules | Where-Object { $_.path -notmatch '/' } | ForEach-Object { $_.path })
    & git -C $Path -c core.autocrlf=false submodule update --init --depth 1 --recursive -- $sdkTestkitRoots
    if ($LASTEXITCODE -ne 0) { throw 'Testkit submodule download failed.' }
    foreach ($sdkTestkitSubmodule in $Pin.testkitSubmodules) {
        Assert-SdkIdentity (Join-Path $Path $sdkTestkitSubmodule.path) $sdkTestkitSubmodule.commit $sdkTestkitSubmodule.tree
    }
}
function Resolve-SdkCheckout($Name, $Pin) {
    # 把一个仓库解析成"缓存里与 pin 一致的干净检出"：缓存 → 本地镜像 → 联网；离线时禁网。
    $sdkTarget = Join-Path $script:sdkDirectory $Name
    if (Test-SdkCheckout $sdkTarget) {
        if (Test-SdkIdentityMatches $sdkTarget $Pin) {
            if (!(Test-SdkRemoteMatches $sdkTarget $Pin)) { Set-SdkRemote $sdkTarget $Pin }
            Write-Output "Verified SDK: $Name at $($Pin.commit) (cache)"
            return
        }
        Write-Output "Cache for $Name does not match the pin; repairing."
    }
    $sdkMirrorEntry = Join-Path $script:sdkMirrorRoot $Name
    if (Test-SdkCheckout $sdkMirrorEntry) {
        if (!(Test-SdkIdentityMatches $sdkMirrorEntry $Pin)) {
            throw "SDK mirror entry does not match the pin: $sdkMirrorEntry (refresh it with scripts/setup/export-sdk-mirror.ps1)."
        }
        Write-Output "Importing $Name from the local mirror: $sdkMirrorEntry"
        Import-SdkFromMirror $Name $sdkTarget
        if (!(Test-SdkIdentityMatches $sdkTarget $Pin)) { throw "Mirror import did not reproduce the pinned $Name checkout." }
        if (!(Test-SdkRemoteMatches $sdkTarget $Pin)) { Set-SdkRemote $sdkTarget $Pin }
        Write-Output "Verified SDK: $Name at $($Pin.commit) (mirror)"
        return
    }
    if ($script:sdkOffline) {
        throw (@(
            "Offline mode: $Name is neither in the SDK cache nor in a local mirror.",
            "  Point HIFISHIFTER_SDK_MIRROR (or -MirrorDirectory) at a directory produced by",
            "  scripts/setup/export-sdk-mirror.ps1 on a networked machine, then retry. See README '第三方源码'."
        ) -join "`n")
    }
    Initialize-SdkCheckout $sdkTarget $Pin
    Assert-SdkIdentity $sdkTarget $Pin.commit $Pin.tree
    Assert-SdkIdentity (Join-Path $sdkTarget $Pin.submodule) $Pin.submoduleCommit $Pin.submoduleTree
    Write-Output "Verified SDK: $Name at $($Pin.commit) (downloaded)"
}
foreach ($sdkTaskEntry in @(@('ARA_SDK', $sdkTaskLock.ara), @('vst3sdk', $sdkTaskLock.vst3))) {
    Resolve-SdkCheckout $sdkTaskEntry[0] $sdkTaskEntry[1]
    if ($WithTestkit) {
        Ensure-TestkitSubmodules (Join-Path $script:sdkDirectory $sdkTaskEntry[0]) $sdkTaskEntry[1]
    }
}
