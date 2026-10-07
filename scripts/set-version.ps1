<#
.SYNOPSIS
    Synchronize project version across frontend and Tauri backend files.

.DESCRIPTION
    Updates version fields in:
    - frontend\package.json
    - backend\src-tauri\Cargo.toml ([package] section version)
    - backend\src-tauri\tauri.conf.json
    - backend\src-tauri\tauri.conf.dist-dev.json
    - backend\hifishifter-plugin\Cargo.toml ([package] section version)

    Optional updates:
    - frontend\package-lock.json (all JSON "version" lines)
    - backend\Cargo.lock (HiFiShifter and hifishifter-plugin entries)

    The plugin crate is included because its version is user-visible: the ARA
    editor's About dialog reports it, and the installer writes it into the
    Add/Remove Programs entry. Leaving it at a stale value made the plug-in
    claim a different version than the standalone App (see
    docs/plans/2026-10-07-vst3-ara-pipeline-overhaul.md, P1).

    Supported inputs:
    1) Full version, for example: 0.1.0-beta.9
    2) Beta suffix, for example: beta9 / beta.9 / beta-9
       The beta form derives major.minor.patch from tauri.conf.json.

.PARAMETER Version
    Target version string.

.EXAMPLE
    .\scripts\set-version.ps1 -Version "0.1.0-beta.9"

.EXAMPLE
    .\scripts\set-version.ps1 -Version "beta9"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-TextFile {
    param(
        [string]$FilePath
    )

    return Get-Content -Path $FilePath -Raw -Encoding UTF8
}

function Write-TextFile {
    param(
        [string]$FilePath,
        [string]$Content
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($FilePath, $Content, $utf8NoBom)
}

function Resolve-TargetVersion {
    param(
        [string]$InputVersion,
        [string]$CurrentVersion
    )

    if ($InputVersion -match '^(?i)beta[.\-]?(\d+)$') {
        $betaNumber = $matches[1]
        if ($CurrentVersion -match '^(\d+\.\d+\.\d+)-') {
            return "$($matches[1])-beta.$betaNumber"
        }

        throw "Current version '$CurrentVersion' cannot derive beta target. Expected format x.y.z-suffix."
    }

    return $InputVersion
}

function Set-JsonVersion {
    param(
        [string]$FilePath,
        [string]$TargetVersion
    )

    $raw = Read-TextFile -FilePath $FilePath
    $pattern = '(?m)("version"\s*:\s*")([^"]+)(")'

    if (-not [regex]::IsMatch($raw, $pattern)) {
        throw "Could not find a JSON version field in $FilePath."
    }

    $regex = [regex]::new($pattern)
    $updated = $regex.Replace(
        $raw,
        { param($m) $m.Groups[1].Value + $TargetVersion + $m.Groups[3].Value },
        1
    )

    if ($updated -ne $raw) {
        Write-TextFile -FilePath $FilePath -Content $updated
    }
}

function Set-CargoPackageVersion {
    param(
        [string]$FilePath,
        [string]$TargetVersion
    )

    $raw = Read-TextFile -FilePath $FilePath
    $pattern = '(?ms)(\[package\].*?^\s*version\s*=\s*")([^"]+)(")'

    if (-not [regex]::IsMatch($raw, $pattern)) {
        throw "Could not find [package] version in $FilePath."
    }

    $updated = [regex]::Replace(
        $raw,
        $pattern,
        { param($m) $m.Groups[1].Value + $TargetVersion + $m.Groups[3].Value },
        1
    )

    if ($updated -ne $raw) {
        Write-TextFile -FilePath $FilePath -Content $updated
    }
}

function Set-PackageLockVersions {
    param(
        [string]$FilePath,
        [string]$TargetVersion
    )

    $raw = Read-TextFile -FilePath $FilePath
    # Replace the first TWO "version" fields only.  In package-lock.json
    # these are always:
    #   1) the top-level project version
    #   2) the root package entry under "packages".""
    # Every subsequent "version" belongs to a dependency and must stay.
    #
    # IMPORTANT: the static [regex]::Replace(input, pattern, evaluator, N)
    # treats N as RegexOptions (where 2 = Multiline).  To limit count we
    # must use an INSTANCE method on a Regex object.
    $regex = [regex]::new('(?m)("version"\s*:\s*")([^"]+)(")')
    $updated = $regex.Replace(
        $raw,
        { param($m) $m.Groups[1].Value + $TargetVersion + $m.Groups[3].Value },
        2
    )

    if ($updated -ne $raw) {
        Write-TextFile -FilePath $FilePath -Content $updated
    }
}

function Set-CargoLockVersion {
    param(
        [string]$FilePath,
        [string[]]$PackageNames,
        [string]$TargetVersion
    )

    $raw = Read-TextFile -FilePath $FilePath
    foreach ($packageName in $PackageNames) {
        # `[[package]]` 块内 name/version 的相邻关系由 cargo 生成，格式稳定。
        $pattern = '(?ms)(\[\[package\]\]\s*name\s*=\s*"' + [regex]::Escape($packageName) +
            '"\s*version\s*=\s*")([^"]+)(")'
        if (-not [regex]::IsMatch($raw, $pattern)) {
            throw "Could not find [[package]] $packageName in $FilePath."
        }
        $raw = [regex]::Replace(
            $raw,
            $pattern,
            { param($m) $m.Groups[1].Value + $TargetVersion + $m.Groups[3].Value },
            1
        )
    }

    Write-TextFile -FilePath $FilePath -Content $raw
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tauriConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.json"
$tauriDistConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.dist-dev.json"
$cargoTomlPath = Join-Path $projectRoot "backend\src-tauri\Cargo.toml"
$pluginCargoTomlPath = Join-Path $projectRoot "backend\hifishifter-plugin\Cargo.toml"
# workspace 合并后 lock 只有一份，位于 backend\ 而不是 src-tauri\。
$cargoLockPath = Join-Path $projectRoot "backend\Cargo.lock"
$frontendPackagePath = Join-Path $projectRoot "frontend\package.json"
$frontendLockPath = Join-Path $projectRoot "frontend\package-lock.json"

$currentTauriVersion = (Read-TextFile -FilePath $tauriConfPath | ConvertFrom-Json).version
$targetVersion = Resolve-TargetVersion -InputVersion $Version -CurrentVersion $currentTauriVersion

Write-Host "Version update: $currentTauriVersion -> $targetVersion" -ForegroundColor Cyan

Set-JsonVersion -FilePath $frontendPackagePath -TargetVersion $targetVersion
Set-CargoPackageVersion -FilePath $cargoTomlPath -TargetVersion $targetVersion
Set-JsonVersion -FilePath $tauriConfPath -TargetVersion $targetVersion
Set-JsonVersion -FilePath $tauriDistConfPath -TargetVersion $targetVersion
Set-CargoPackageVersion -FilePath $pluginCargoTomlPath -TargetVersion $targetVersion

if (Test-Path $frontendLockPath) {
    Set-PackageLockVersions -FilePath $frontendLockPath -TargetVersion $targetVersion
}

if (Test-Path $cargoLockPath) {
    Set-CargoLockVersion -FilePath $cargoLockPath -PackageNames @('HiFiShifter', 'hifishifter-plugin') -TargetVersion $targetVersion
}

Write-Host "Updated version files:" -ForegroundColor Green
Write-Host "  - frontend\package.json" -ForegroundColor Green
Write-Host "  - backend\src-tauri\Cargo.toml" -ForegroundColor Green
Write-Host "  - backend\src-tauri\tauri.conf.json" -ForegroundColor Green
Write-Host "  - backend\src-tauri\tauri.conf.dist-dev.json" -ForegroundColor Green
Write-Host "  - backend\hifishifter-plugin\Cargo.toml" -ForegroundColor Green
if (Test-Path $frontendLockPath) {
    Write-Host "  - frontend\package-lock.json" -ForegroundColor Green
}
if (Test-Path $cargoLockPath) {
    Write-Host "  - backend\Cargo.lock (HiFiShifter, hifishifter-plugin)" -ForegroundColor Green
}

# 版本来源表：`scripts\check-product-consistency.ps1` 校验同一组文件，这里打印
# 一份便于人工核对 —— 漏改一处时两边的输出会不一致。
Write-Host ""
Write-Host "Version sources:" -ForegroundColor Cyan
$versionSources = [ordered]@{
    'frontend\package.json'                  = (Read-TextFile -FilePath $frontendPackagePath | ConvertFrom-Json).version
    'backend\src-tauri\tauri.conf.json'      = (Read-TextFile -FilePath $tauriConfPath | ConvertFrom-Json).version
    'backend\src-tauri\tauri.conf.dist-dev.json' = (Read-TextFile -FilePath $tauriDistConfPath | ConvertFrom-Json).version
    'backend\src-tauri\Cargo.toml'           = (Select-String -Path $cargoTomlPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
    'backend\hifishifter-plugin\Cargo.toml'  = (Select-String -Path $pluginCargoTomlPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
}
foreach ($source in $versionSources.GetEnumerator()) {
    $marker = if ($source.Value -eq $targetVersion) { 'OK  ' } else { 'DIFF' }
    $color = if ($source.Value -eq $targetVersion) { 'Green' } else { 'Red' }
    Write-Host ("  [{0}] {1} = {2}" -f $marker, $source.Key, $source.Value) -ForegroundColor $color
}