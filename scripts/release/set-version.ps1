# 同步版本号：一处输入，按清单落盘（清单见同目录的 version-sources.json）。
#
# 【为什么清单是数据而不是代码】写入（本脚本）与校验
# （scripts/checks/check-product-consistency.ps1）必须看同一份文件列表。此前两边各写一份：
# 加一个 Cargo.toml 时只在一处加，另一处就悄悄漏检 —— 插件版本长期停在 0.1.0 而无人
# 发现，正是这么发生的。正则也住在数据里，两边不必各解释一遍。
#
# 支持完整版本（0.1.0-beta.9），也支持 beta9 / beta.9 —— 后者从 canonical 文件
# （tauri.conf.json）取 major.minor.patch。插件 crate 的版本号是用户可见的（编辑器
# About 对话框与「程序和功能」），所以必须和独立 App 一起改。
#
#   .\scripts\release\set-version.ps1 -Version 0.1.0-beta.16
#   .\scripts\release\set-version.ps1 -Version beta16
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Version)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifestPath = Join-Path $PSScriptRoot 'version-sources.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json

function Get-RepoText([string]$relative) {
    Get-Content -LiteralPath (Join-Path $root $relative) -Raw -Encoding utf8
}

function Find-Version([string]$relative, [string]$pattern) {
    $match = [regex]::Match((Get-RepoText $relative), $pattern)
    if (!$match.Success) { throw "$relative 里找不到版本字段" }
    $match.Groups[2].Value
}

$canonical = $manifest.sources | Where-Object { $_.path -eq $manifest.canonical } | Select-Object -First 1
if (!$canonical) { throw "version-sources.json 的 canonical 不在 sources 里：$($manifest.canonical)" }
$current = Find-Version $canonical.path $canonical.pattern

# 【为什么先取出两组再拼】$Matches 是自动变量，会被下一次匹配覆盖。早先的写法在第二个
# 正则之后仍去读 $Matches[2]，而那个正则在第二个位置上是空的 —— 结果拼出来是
# "0.1.0-beta."，看起来像"版本号少了一截"。这里先取序号、再独立求前缀。
if ($Version -match '^(?i)beta[.\-]?(\d+)$') {
    $ordinal = $Matches[1]
    $prefix = [regex]::Match($current, '^(\d+\.\d+\.\d+)-').Groups[1].Value
    if (!$prefix) { throw "当前版本 $current 无法推导 beta 目标版本" }
    $Version = "$prefix-beta.$ordinal"
}

foreach ($source in $manifest.sources) {
    $path = Join-Path $root $source.path
    $text = Get-Content -LiteralPath $path -Raw -Encoding utf8
    $regex = [regex]::new($source.pattern)
    $occurrences = [int]$source.occurrences
    $found = $regex.Matches($text).Count
    if ($found -lt $occurrences) {
        throw "$($source.path) 里只找到 $found 处版本字段，version-sources.json 要求 $occurrences 处"
    }
    $updated = $regex.Replace($text, { param($m) $m.Groups[1].Value + $Version + $m.Groups[3].Value }, $occurrences)
    [IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
}

Write-Host "版本：$current -> $Version" -ForegroundColor Green
