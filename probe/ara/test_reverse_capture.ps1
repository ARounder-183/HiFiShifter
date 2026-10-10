# test_reverse_capture.ps1 —— verify_reverse_capture.ps1 的本地回归。
#
# F-4 的真实采集要在 REAPER 里跑（build_reverse_probe.lua），本机跑不了。但**分类器**
# 的正确性可以完全本地验证：喂合成的日志，断言它把"正向坐标""镜像坐标"分别判对，
# 并把缺证据 / 几何不符 / 第三种坐标判成失败（而不是悄悄归入某一类）。
#
# 这与 `test_task11_capture.ps1` / `test_phase3a_output.ps1` 同一条纪律：验证器的
# 正确性由变异回归保证，与真实宿主采集是两件事。
$ErrorActionPreference = 'Stop'
$verifier = Join-Path $PSScriptRoot 'verify_reverse_capture.ps1'
$fixture = Join-Path $PSScriptRoot 'fixtures\phase3a-asymmetric.wav'
$scratch = Join-Path $PSScriptRoot '..\..\.build-tmp\reverse-capture-tests'
if (Test-Path -LiteralPath $scratch) { Remove-Item -LiteralPath $scratch -Recurse -Force }
New-Item -ItemType Directory -Path $scratch -Force | Out-Null

function Write-Logs([string]$name, [string]$scriptBody, [string]$pluginBody) {
    $scriptLog = Join-Path $scratch ($name + '.script.log')
    $pluginLog = Join-Path $scratch ($name + '.plugin.log')
    Set-Content -LiteralPath $scriptLog -Value $scriptBody -Encoding utf8
    Set-Content -LiteralPath $pluginLog -Value $pluginBody -Encoding utf8
    return @($scriptLog, $pluginLog)
}

$scriptHeader = @(
    'take facts: forward startoffs=0.250000 playrate=1.000000 length=1.000000',
    'reverse verified: forward section=false reversed=false offset=0.000000 length=2.000000',
    'take facts: reversed startoffs=0.750000 playrate=1.000000 length=1.000000',
    'reverse verified: reversed section=true reversed=true offset=0.000000 length=2.000000',
    'TrackFX_AddByName -> 0'
)
$regionHeader = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #0: source=fixtures/phase3a-asymmetric.wav startMod=0.250000 durationMod=1.000000 startPlay=0.000000 durationPlay=1.000000 flags=0x1'
$regionForwardCoord = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #1: source=fixtures/phase3a-asymmetric.wav startMod=0.250000 durationMod=1.000000 startPlay=3.000000 durationPlay=1.000000 flags=0x1'
$regionMirroredCoord = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #1: source=fixtures/phase3a-asymmetric.wav startMod=0.750000 durationMod=1.000000 startPlay=3.000000 durationPlay=1.000000 flags=0x1'
$regionThirdCoord = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #1: source=fixtures/phase3a-asymmetric.wav startMod=0.500000 durationMod=1.000000 startPlay=3.000000 durationPlay=1.000000 flags=0x1'
$regionWrongStart = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #0: source=fixtures/phase3a-asymmetric.wav startMod=0.100000 durationMod=1.000000 startPlay=0.000000 durationPlay=1.000000 flags=0x1'
$regionWrongDuration = '[INFO] hifishifter_plugin::ara::model: [ara] playback_region #1: source=fixtures/phase3a-asymmetric.wav startMod=0.750000 durationMod=0.500000 startPlay=3.000000 durationPlay=1.000000 flags=0x1'

function Expect-Success {
    param([string]$Name, [string]$ReverseRegion)
    $paths = Write-Logs $Name ($scriptHeader -join "`n") (@($regionHeader, $ReverseRegion) -join "`n")
    $output = Join-Path $scratch ($Name + '.json')
    & $verifier -LogPath $paths[0] -PluginLogPath $paths[1] -FixturePath $fixture -OutputPath $output | Out-Null
    $json = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    Write-Host "PASS $Name (coordinateSystem=$($json.conclusion.coordinateSystem))"
    return $json.conclusion.coordinateSystem
}

function Expect-Failure {
    param([string]$Name, [string]$ScriptBody, [string]$PluginBody, [string]$Reason)
    $paths = Write-Logs $Name $ScriptBody $PluginBody
    $output = Join-Path $scratch ($Name + '.json')
    $failure = $null
    try {
        & $verifier -LogPath $paths[0] -PluginLogPath $paths[1] -FixturePath $fixture -OutputPath $output | Out-Null
    } catch {
        $failure = $_.Exception.Message
    }
    if (-not $failure -or -not $failure.Contains($Reason)) {
        throw "case $Name should have failed for '$Reason', got: $failure"
    }
    Write-Host "PASS $Name (rejected: $Reason)"
}

# 正向坐标：倒放 region 的 startMod 与正放一致（= 裁切起点 0.25）。
$forwardResult = Expect-Success 'forward-coordinates' $regionForwardCoord
if ($forwardResult -ne 'forward') { throw "expected forward classification, got $forwardResult" }

# 镜像坐标：倒放 region 的 startMod = 2.0 - 0.25 - 1.0 = 0.75。
$mirroredResult = Expect-Success 'mirrored-coordinates' $regionMirroredCoord
if ($mirroredResult -ne 'mirrored') { throw "expected mirrored classification, got $mirroredResult" }

# 变异：宿主没把倒放位读出来。
$notReversed = $scriptHeader -replace 'reversed section=true reversed=true', 'reversed section=true reversed=false'
Expect-Failure 'reverse-not-verified' ($notReversed -join "`n") (@($regionHeader, $regionMirroredCoord) -join "`n") 'did not verify as reversed'

# 变异：第三种坐标 —— 必须人工看，不能被归入任一结论。
Expect-Failure 'third-coordinate' ($scriptHeader -join "`n") (@($regionHeader, $regionThirdCoord) -join "`n") 'needs a human'

# 变异：region 数量不是 2。
Expect-Failure 'region-count' ($scriptHeader -join "`n") $regionHeader 'expected exactly two regions'

# 变异：正放 region 的起点不是裁切起点（探针几何本身错了）。
Expect-Failure 'wrong-forward-start' ($scriptHeader -join "`n") (@($regionWrongStart, $regionMirroredCoord) -join "`n") 'is not the trim offset'

# 变异：倒放 region 的时长不是 item 时长。
Expect-Failure 'wrong-reversed-duration' ($scriptHeader -join "`n") (@($regionHeader, $regionWrongDuration) -join "`n") 'is not the item length'

# 变异：缺倒放 item 的宿主核实。
$noReverse = $scriptHeader | Where-Object { $_ -notmatch '^reverse verified: reversed' }
Expect-Failure 'missing-reversed-verification' ($noReverse -join "`n") (@($regionHeader, $regionMirroredCoord) -join "`n") 'missing host verification for the reversed item'

Write-Host 'reverse capture verifier: all cases pass'
