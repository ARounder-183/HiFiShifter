# 同步版本号：一处输入，七处落盘。
#
# 支持完整版本（0.1.0-beta.9），也支持 beta9 / beta.9 —— 后者从 tauri.conf.json 取
# major.minor.patch。插件 crate 的版本号是用户可见的（编辑器 About 对话框与「程序和功能」），
# 所以必须和独立 App 一起改。
#
#   .\scripts\release\set-version.ps1 -Version 0.1.0-beta.16
#   .\scripts\release\set-version.ps1 -Version beta16
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Version)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))

$confPath = Join-Path $root 'backend\src-tauri\tauri.conf.json'
$current = (Get-Content $confPath -Raw | ConvertFrom-Json).version
if ($Version -match '^(?i)beta[.\-]?(\d+)$') {
    if ($current -notmatch '^(\d+\.\d+\.\d+)-') { throw "当前版本 $current 无法推导 beta 目标版本" }
    $Version = $Matches[1] + '-beta.' + $Matches[2]
}

# JSON 只换第一处（package-lock.json 要前两处：顶层版本与根包条目，其余属于依赖）。
$jsonPattern = '(?m)("version"\s*:\s*")([^"]+)(")'
# Cargo.toml 只认 [package] 段内的 version。
$cargoPattern = '(?ms)(\[package\].*?^\s*version\s*=\s*")([^"]+)(")'

function Set-VersionField($relative, $pattern, [int]$count = 1) {
    $path = Join-Path $root $relative
    $text = Get-Content $path -Raw -Encoding utf8
    $regex = [regex]::new($pattern)
    if (!$regex.IsMatch($text)) { throw "$relative 里找不到版本字段" }
    $updated = $regex.Replace($text, { param($m) $m.Groups[1].Value + $Version + $m.Groups[3].Value }, $count)
    [IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
}

Set-VersionField 'frontend\package.json' $jsonPattern
Set-VersionField 'frontend\package-lock.json' $jsonPattern 2
Set-VersionField 'backend\src-tauri\tauri.conf.json' $jsonPattern
Set-VersionField 'backend\src-tauri\tauri.conf.dist-dev.json' $jsonPattern
Set-VersionField 'backend\src-tauri\Cargo.toml' $cargoPattern
Set-VersionField 'backend\hifishifter-plugin\Cargo.toml' $cargoPattern
foreach ($package in 'HiFiShifter', 'hifishifter-plugin') {
    $lockPattern = '(?ms)(\[\[package\]\]\s*name\s*=\s*"' + [regex]::Escape($package) + '"\s*version\s*=\s*")([^"]+)(")'
    Set-VersionField 'backend\Cargo.lock' $lockPattern
}

Write-Host "版本：$current -> $Version" -ForegroundColor Green
