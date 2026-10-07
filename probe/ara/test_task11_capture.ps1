# 一次性验证脚本回归：有效采集必须通过，变异证据不得生成错误的 PASS。
$ErrorActionPreference = 'Stop'
$verifier = Join-Path $PSScriptRoot 'verify_task11_capture.ps1'
$original = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'captures\task11-plugin.log')
$scratch = Join-Path $PSScriptRoot '..\..\.build-tmp\task11-capture-tests'
New-Item -ItemType Directory -Force $scratch | Out-Null

$mutations = @(
    @{ name = 'valid'; expected = $null; change = { param($line) $line } },
    @{ name = 'stretch-duration'; expected = 'duration ratio 2/1'; change = {
        param($line)
        if ($line -match 'playback_region #1:') { $line -replace 'durationPlay=1.000000', 'durationPlay=4.000000' } else { $line }
    } },
    @{ name = 'stretch-flag'; expected = 'duration ratio 2/1'; change = {
        param($line)
        if ($line -match 'playback_region #1:') { $line -replace 'flags=0x1', 'flags=0x0' } else { $line }
    } },
    @{ name = 'reverse-geometry'; expected = 'Reverse region differs'; change = {
        param($line)
        if ($line -match 'playback_region #2:') { $line -replace 'startMod=0.000000', 'startMod=0.500000' } else { $line }
    } },
    @{ name = 'wrong-identity'; expected = 'Unexpected region identity'; change = {
        param($line)
        if ($line -match 'playback_region #2:') { $line -replace 'playback_region #2:', 'playback_region #9:' } else { $line }
    } },
    @{ name = 'unverified-reverse'; expected = 'did not verify'; change = {
        param($line)
        $line -replace '^reverse verified: section=true reversed=true', 'reverse verified: section=true reversed=false'
    } }
)
foreach ($case in $mutations) {
    $log = Join-Path $scratch ($case.name + '.log')
    $output = Join-Path $scratch ($case.name + '.json')
    @($original | ForEach-Object { & $case.change $_ }) | Set-Content -LiteralPath $log -Encoding utf8
    $failure = $null
    try { & $verifier -LogPath $log -OutputPath $output | Out-Null } catch { $failure = $_.Exception.Message }
    if ($null -eq $case.expected) {
        if ($failure) { throw "Valid capture failed: $failure" }
    } elseif (-not $failure -or -not $failure.Contains($case.expected)) {
        throw "Mutation $($case.name) was not rejected for the intended reason: $failure"
    }
    Write-Output "PASS $($case.name)"
}
