<#
.SYNOPSIS
    统一更新项目版本号（frontend + tauri + cargo）。

.DESCRIPTION
    将以下文件中的主版本号同步为同一值：
    - frontend\package.json
    - backend\src-tauri\Cargo.toml ([package] 内 version)
    - backend\src-tauri\tauri.conf.json
    - backend\src-tauri\tauri.conf.dist-dev.json

    支持两种输入：
    1) 完整版本号：0.1.0-beta.8
    2) beta 后缀：beta.8 / beta8（会基于当前 tauri.conf.json 的主版本前缀生成完整版本）

.PARAMETER Version
    目标版本号（完整版本或 beta 后缀）。

.EXAMPLE
    .\scripts\set-version.ps1 -Version "0.1.0-beta.8"

.EXAMPLE
    .\scripts\set-version.ps1 -Version "beta8"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

function Resolve-TargetVersion {
    param(
        [string]$InputVersion,
        [string]$CurrentVersion
    )

    # 允许 beta8 / beta.8 / beta-8
    if ($InputVersion -match '^(?i)beta[.\-]?(\d+)$') {
        $betaNum = $matches[1]
        if ($CurrentVersion -match '^(\d+\.\d+\.\d+)-') {
            return "$($matches[1])-beta.$betaNum"
        }
        throw "当前版本 '$CurrentVersion' 不是可推导的语义版本（x.y.z-xxx）"
    }

    # 直接使用完整版本号
    return $InputVersion
}

function Replace-JsonVersion {
    param(
        [string]$FilePath,
        [string]$TargetVersion
    )
    $raw = Get-Content $FilePath -Raw
    $updated = [regex]::Replace(
        $raw,
        '(?m)("version"\s*:\s*")([^"]+)(")',
        ('$1' + [regex]::Escape($TargetVersion).Replace('\', '') + '$3'),
        1
    )
    if ($updated -eq $raw) {
        throw "未在 $FilePath 中找到可替换的 JSON version 字段"
    }
    Set-Content -Path $FilePath -Value $updated -NoNewline -Encoding UTF8
}

function Replace-CargoPackageVersion {
    param(
        [string]$FilePath,
        [string]$TargetVersion
    )
    $raw = Get-Content $FilePath -Raw
    $pattern = '(?ms)(\[package\].*?^\s*version\s*=\s*")([^"]+)(")'
    $updated = [regex]::Replace(
        $raw,
        $pattern,
        ('$1' + [regex]::Escape($TargetVersion).Replace('\', '') + '$3'),
        1
    )
    if ($updated -eq $raw) {
        throw "未在 $FilePath 的 [package] 段中找到可替换的 version"
    }
    Set-Content -Path $FilePath -Value $updated -NoNewline -Encoding UTF8
}

$projectRoot = Resolve-Path "$PSScriptRoot\.."
$tauriConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.json"
$tauriDistConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.dist-dev.json"
$cargoTomlPath = Join-Path $projectRoot "backend\src-tauri\Cargo.toml"
$cargoLockPath = Join-Path $projectRoot "backend\src-tauri\Cargo.lock"
$frontendPackagePath = Join-Path $projectRoot "frontend\package.json"
$frontendLockPath = Join-Path $projectRoot "frontend\package-lock.json"

$currentTauri = (Get-Content $tauriConfPath -Raw | ConvertFrom-Json).version
$targetVersion = Resolve-TargetVersion -InputVersion $Version -CurrentVersion $currentTauri

Write-Host "版本更新: $currentTauri -> $targetVersion" -ForegroundColor Cyan

Replace-JsonVersion -FilePath $frontendPackagePath -TargetVersion $targetVersion
Replace-CargoPackageVersion -FilePath $cargoTomlPath -TargetVersion $targetVersion
Replace-JsonVersion -FilePath $tauriConfPath -TargetVersion $targetVersion
Replace-JsonVersion -FilePath $tauriDistConfPath -TargetVersion $targetVersion

if (Test-Path $frontendLockPath) {
    $frontendLockRaw = Get-Content $frontendLockPath -Raw
    $frontendLockUpdated = $frontendLockRaw `
        -replace '(?m)^(\s*"version"\s*:\s*")([^"]+)(",?)$', ('$1' + $targetVersion + '$3')
    Set-Content -Path $frontendLockPath -Value $frontendLockUpdated -NoNewline -Encoding UTF8
}

if (Test-Path $cargoLockPath) {
    $cargoLockRaw = Get-Content $cargoLockPath -Raw
    $cargoLockUpdated = [regex]::Replace(
        $cargoLockRaw,
        '(?ms)(\[\[package\]\]\s*name\s*=\s*"HiFiShifter"\s*version\s*=\s*")([^"]+)(")',
        ('$1' + $targetVersion + '$3'),
        1
    )
    Set-Content -Path $cargoLockPath -Value $cargoLockUpdated -NoNewline -Encoding UTF8
}

Write-Host "已更新版本文件：" -ForegroundColor Green
Write-Host "  - frontend\package.json" -ForegroundColor Green
Write-Host "  - backend\src-tauri\Cargo.toml" -ForegroundColor Green
Write-Host "  - backend\src-tauri\tauri.conf.json" -ForegroundColor Green
Write-Host "  - backend\src-tauri\tauri.conf.dist-dev.json" -ForegroundColor Green
if (Test-Path $frontendLockPath) {
    Write-Host "  - frontend\package-lock.json" -ForegroundColor Green
}
if (Test-Path $cargoLockPath) {
    Write-Host "  - backend\src-tauri\Cargo.lock (HiFiShifter package)" -ForegroundColor Green
}
