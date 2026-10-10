# 下载并校验 ARA / VST3 SDK 检出，落地到 tools/sdk-env.ps1 定义的缓存根（third_party/sdk）。
# 身份校验与 ara2-bridge-companion 0.3.0 保持一致；只下载构建所需子模块。
[CmdletBinding()]
param([string]$SdkDirectory)
$ErrorActionPreference = 'Stop'
$sdkTaskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
# 缓存根由 tools/sdk-env.ps1 统一定义，避免路径散落在多个脚本与 CI workflow 里。
. (Join-Path $PSScriptRoot 'sdk-env.ps1')
if (!$SdkDirectory) { $SdkDirectory = Get-SdkCacheRoot }
$sdkTaskDirectory = [IO.Path]::GetFullPath($SdkDirectory)
if (!$sdkTaskDirectory.StartsWith($sdkTaskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'SDK directory must be inside this checkout.'
}
$sdkTaskLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'plugin-sdks.json') -Raw | ConvertFrom-Json
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
foreach ($sdkTaskEntry in @(@('ARA_SDK', $sdkTaskLock.ara), @('vst3sdk', $sdkTaskLock.vst3))) {
    $sdkTaskPath = Join-Path $sdkTaskDirectory $sdkTaskEntry[0]
    $sdkTaskPin = $sdkTaskEntry[1]
    if (!(Test-SdkCheckout $sdkTaskPath)) {
        Initialize-SdkCheckout $sdkTaskPath $sdkTaskPin
    }
    $sdkTaskRemote = & git -C $sdkTaskPath remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $sdkTaskRemote.TrimEnd('/').Replace('.git', '') -ne $sdkTaskPin.repository.Replace('.git', '')) {
        Initialize-SdkCheckout $sdkTaskPath $sdkTaskPin
    }
    # 目录被外部清理破坏（例如空的 .git/refs 被删除、未初始化的子模块目录丢失）时，
    # pin 或工作区会不匹配；重建一次再校验，而不是直接失败。
    try {
        Assert-SdkIdentity $sdkTaskPath $sdkTaskPin.commit $sdkTaskPin.tree
        Assert-SdkIdentity (Join-Path $sdkTaskPath $sdkTaskPin.submodule) $sdkTaskPin.submoduleCommit $sdkTaskPin.submoduleTree
    } catch {
        Write-Output "Repairing $($sdkTaskEntry[0]): $($_.Exception.Message)"
        Initialize-SdkCheckout $sdkTaskPath $sdkTaskPin
        Assert-SdkIdentity $sdkTaskPath $sdkTaskPin.commit $sdkTaskPin.tree
        Assert-SdkIdentity (Join-Path $sdkTaskPath $sdkTaskPin.submodule) $sdkTaskPin.submoduleCommit $sdkTaskPin.submoduleTree
    }
    Write-Output "Verified SDK: $($sdkTaskEntry[0]) at $($sdkTaskPin.commit)"
}
