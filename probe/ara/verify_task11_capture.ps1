# 一次性 Task 11 采集核验：读取宿主日志并与夹具 WAV 的原始 PCM 比较，生成可审计 JSON。
param(
    [string]$LogPath = (Join-Path $PSScriptRoot 'captures\task11-plugin.log'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'captures\task11-stretch-reverse.json')
)
$ErrorActionPreference = 'Stop'
$captureDir = Join-Path $PSScriptRoot 'captures'
$lines = Get-Content -LiteralPath $LogPath
if (-not ($lines -match '^reverse verified: section=true reversed=true')) {
    throw 'REAPER did not verify a reversed take source'
}

$regionPattern = 'playback_region #(?<index>\d+): source=(?<source>.+) startMod=(?<startMod>[\d.-]+) durationMod=(?<durMod>[\d.-]+) startPlay=(?<startPlay>[\d.-]+) durationPlay=(?<durPlay>[\d.-]+) flags=0x(?<flags>[0-9A-F]+)'
$regions = @($lines | ForEach-Object {
    if ($_ -match $regionPattern) {
        [ordered]@{
            index = [int]$Matches.index
            audioSourcePersistentID = $Matches.source
            startInModificationTime = [double]::Parse($Matches.startMod, [cultureinfo]::InvariantCulture)
            durationInModificationTime = [double]::Parse($Matches.durMod, [cultureinfo]::InvariantCulture)
            startInPlaybackTime = [double]::Parse($Matches.startPlay, [cultureinfo]::InvariantCulture)
            durationInPlaybackTime = [double]::Parse($Matches.durPlay, [cultureinfo]::InvariantCulture)
            transformationFlags = [Convert]::ToInt32($Matches.flags, 16)
        }
    }
})
if ($regions.Count -ne 3) { throw 'Expected exactly three initial regions' }
for ($index = 0; $index -lt 3; $index++) {
    if ($regions[$index].index -ne $index -or $regions[$index].startInPlaybackTime -ne (3 * $index)) {
        throw 'Unexpected region identity or fixture placement'
    }
}
if ($regions[1].durationInModificationTime -ne 2.0 -or
    $regions[1].durationInPlaybackTime -ne 1.0 -or
    ($regions[1].transformationFlags -band 1) -eq 0) {
    throw 'Expected a timestretch region with duration ratio 2/1'
}
foreach ($field in @('startInModificationTime', 'durationInModificationTime',
    'durationInPlaybackTime', 'transformationFlags')) {
    if ($regions[2][$field] -ne $regions[0][$field]) {
        throw "Reverse region differs from forward region: $field"
    }
}
if (@($regions.audioSourcePersistentID | Select-Object -Unique).Count -ne 1) {
    throw 'Expected one shared audio source for this reverse experiment'
}

$pcmLine = $lines | Where-Object { $_ -match 'source PCM #0: first16=' } | Select-Object -First 1
if (-not $pcmLine) { throw 'No host PCM evidence found' }
$hostPcm = @(($pcmLine -replace '^.*first16=', '') | ConvertFrom-Json)
$wavPath = Join-Path $PSScriptRoot 'fixtures\tone44100.wav'
$stream = [System.IO.File]::OpenRead($wavPath)
$reader = [System.IO.BinaryReader]::new($stream)
try {
    if ([text.encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'RIFF') { throw 'Expected RIFF' }
    $null = $reader.ReadUInt32()
    if ([text.encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'WAVE') { throw 'Expected WAVE' }
    $dataOffset = $null
    while ($stream.Position -lt $stream.Length) {
        $chunk = [text.encoding]::ASCII.GetString($reader.ReadBytes(4))
        $length = $reader.ReadUInt32()
        $next = $stream.Position + $length + ($length % 2)
        if ($chunk -eq 'fmt ') {
            $format = $reader.ReadUInt16()
            $channels = $reader.ReadUInt16()
            $sampleRate = $reader.ReadUInt32()
            $null = $reader.ReadUInt32()
            $null = $reader.ReadUInt16()
            $bits = $reader.ReadUInt16()
        } elseif ($chunk -eq 'data') {
            $dataOffset = $stream.Position
        }
        $stream.Position = $next
    }
    if ($null -eq $dataOffset -or $format -ne 1 -or $channels -ne 1 -or $bits -ne 16) {
        throw 'Expected the mono PCM16 fixture'
    }
    $stream.Position = $dataOffset
    $normalPcm = @(0..15 | ForEach-Object { $reader.ReadInt16() / 32768.0 })
} finally {
    $reader.Dispose()
    $stream.Dispose()
}
if ($hostPcm.Count -ne 16) { throw 'Expected sixteen source samples' }
$diffs = @(0..15 | ForEach-Object { [Math]::Abs($hostPcm[$_] - $normalPcm[$_]) })
$maxDifference = ($diffs | Measure-Object -Maximum).Maximum
if ($maxDifference -gt 1e-6) { throw "Host source differs from forward PCM: $maxDifference" }

$result = [ordered]@{
    provenance = 'Parsed from task11-plugin.log and fixtures/tone44100.wav; not a full ARA graph dump'
    host = 'REAPER 7.81'
    reverseAction = 41051
    reverseVerifiedByHost = $true
    regions = $regions
    sourcePCM = [ordered]@{
        sampleRate = $sampleRate
        hostFirst16 = $hostPcm
        fileFirst16 = $normalPcm
        maxAbsoluteDifference = $maxDifference
    }
    conclusions = [ordered]@{
        U2 = 'pass: duration ratio 2/1'
        U1 = 'unsupported by the observed ARA-to-TimelineState mapping: same source PCM, no region direction'
    }
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "Task11: U2 PASS; source PCM max difference=$maxDifference; U1 has no mapped direction"
