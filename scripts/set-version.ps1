<#
.SYNOPSIS
    Synchronize project version across frontend and Tauri backend files.

.DESCRIPTION
    Updates version fields in:
    - frontend\package.json
    - backend\src-tauri\Cargo.toml ([package] section version)
    - backend\src-tauri\tauri.conf.json
    - backend\src-tauri\tauri.conf.dist-dev.json

    Optional updates:
    - frontend\package-lock.json (all JSON "version" lines)
    - backend\src-tauri\Cargo.lock (HiFiShifter package entry)

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
        [string]$TargetVersion
    )

    $raw = Read-TextFile -FilePath $FilePath
    $pattern = '(?ms)(\[\[package\]\]\s*name\s*=\s*"HiFiShifter"\s*version\s*=\s*")([^"]+)(")'
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

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tauriConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.json"
$tauriDistConfPath = Join-Path $projectRoot "backend\src-tauri\tauri.conf.dist-dev.json"
$cargoTomlPath = Join-Path $projectRoot "backend\src-tauri\Cargo.toml"
$cargoLockPath = Join-Path $projectRoot "backend\src-tauri\Cargo.lock"
$frontendPackagePath = Join-Path $projectRoot "frontend\package.json"
$frontendLockPath = Join-Path $projectRoot "frontend\package-lock.json"

$currentTauriVersion = (Read-TextFile -FilePath $tauriConfPath | ConvertFrom-Json).version
$targetVersion = Resolve-TargetVersion -InputVersion $Version -CurrentVersion $currentTauriVersion

Write-Host "Version update: $currentTauriVersion -> $targetVersion" -ForegroundColor Cyan

Set-JsonVersion -FilePath $frontendPackagePath -TargetVersion $targetVersion
Set-CargoPackageVersion -FilePath $cargoTomlPath -TargetVersion $targetVersion
Set-JsonVersion -FilePath $tauriConfPath -TargetVersion $targetVersion
Set-JsonVersion -FilePath $tauriDistConfPath -TargetVersion $targetVersion

if (Test-Path $frontendLockPath) {
    Set-PackageLockVersions -FilePath $frontendLockPath -TargetVersion $targetVersion
}

if (Test-Path $cargoLockPath) {
    Set-CargoLockVersion -FilePath $cargoLockPath -TargetVersion $targetVersion
}

Write-Host "Updated version files:" -ForegroundColor Green
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