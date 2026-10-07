# 与 ara2-bridge-companion 0.3.0 的 SDK 身份校验保持一致；只下载构建所需子模块。
[CmdletBinding()]
param([string]$SdkDirectory)
$ErrorActionPreference = 'Stop'
$sdkTaskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (!$SdkDirectory) { $SdkDirectory = Join-Path $sdkTaskRoot 'probe\ara\rust-path\.third-party' }
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
foreach ($sdkTaskEntry in @(@('ARA_SDK', $sdkTaskLock.ara), @('vst3sdk', $sdkTaskLock.vst3))) {
    $sdkTaskPath = Join-Path $sdkTaskDirectory $sdkTaskEntry[0]
    $sdkTaskPin = $sdkTaskEntry[1]
    if (!(Test-Path -LiteralPath $sdkTaskPath)) {
        & git init $sdkTaskPath
        if ($LASTEXITCODE -ne 0) { throw 'SDK initialization failed.' }
        & git -C $sdkTaskPath remote add origin $sdkTaskPin.repository
        if ($LASTEXITCODE -ne 0) { throw 'SDK remote setup failed.' }
        & git -C $sdkTaskPath fetch --depth 1 origin $sdkTaskPin.commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned SDK download failed.' }
        & git -C $sdkTaskPath -c core.autocrlf=false checkout --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { throw 'Pinned SDK checkout failed.' }
        & git -C $sdkTaskPath -c core.autocrlf=false submodule update --init --depth 1 -- $sdkTaskPin.submodule
        if ($LASTEXITCODE -ne 0) { throw 'SDK submodule download failed.' }
    }
    $sdkTaskRemote = & git -C $sdkTaskPath remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $sdkTaskRemote.TrimEnd('/').Replace('.git', '') -ne $sdkTaskPin.repository.Replace('.git', '')) {
        throw "SDK repository mismatch: $sdkTaskPath"
    }
    Assert-SdkIdentity $sdkTaskPath $sdkTaskPin.commit $sdkTaskPin.tree
    Assert-SdkIdentity (Join-Path $sdkTaskPath $sdkTaskPin.submodule) $sdkTaskPin.submoduleCommit $sdkTaskPin.submoduleTree
    Write-Output "Verified SDK: $($sdkTaskEntry[0]) at $($sdkTaskPin.commit)"
}
