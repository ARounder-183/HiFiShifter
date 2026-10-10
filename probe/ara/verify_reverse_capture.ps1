# verify_reverse_capture.ps1 —— F-4 分类器：判定倒放 take 的 ARA region 时间坐标系。
#
# 读 build_reverse_probe.lua 跑出来的两份日志（脚本日志 + 插件日志），把两个非对称
# 裁切 item 的 region 坐标对起来，回答一个**二选一**的问题：
#
#   * 正向坐标  —— region 的 startMod 与正放 item 相同（= 源内裁切起点）
#   * 已镜像坐标 —— region 的 startMod = 源长 - 裁切起点 - 时长
#
# 本脚本是**分类器**，不是断言器：两个结论都算"成功"，它只负责如实报出是哪一个。
# 无法归入两者（几何对不上、缺证据、坐标是第三种值）才失败 —— 那种情况必须人工看。
#
# 注意它**只**看 region 坐标，判不出"倒放 take 实际播的是哪一段" —— 那由
# `verify_reverse_window.py`（渲染那一个倒放 item 逐样本比对）独立给出。两个一起才
# 构成 F-4 的完整证据：坐标系 + 被覆盖的源窗口。
param(
    [string]$LogPath = (Join-Path $PSScriptRoot 'captures\reverse-script.log'),
    [string]$PluginLogPath = (Join-Path $PSScriptRoot 'captures\reverse-plugin.log'),
    [string]$FixturePath = (Join-Path $PSScriptRoot 'fixtures\phase3a-asymmetric.wav'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'captures\reverse-coordinates.json'),
    [double]$SourceOffset = 0.25,
    [double]$ItemLength = 1.0,
    [double]$ForwardPosition = 0.0,
    [double]$ReversePosition = 3.0
)
$ErrorActionPreference = 'Stop'

function Parse-Invariant([string]$text) {
    return [double]::Parse($text, [cultureinfo]::InvariantCulture)
}
function Close-Enough([double]$a, [double]$b) {
    return [Math]::Abs($a - $b) -le (1e-6 + 1e-9 * [Math]::Max([Math]::Abs($a), [Math]::Abs($b)))
}

$lines = Get-Content -LiteralPath $LogPath

# 1) 宿主方向位核实：倒放 item 必须真的被宿主判为 reversed。`B_REVERSED` setter 是无效
#    证据（EXECUTION-LEDGER），所以只认 `PCM_Source_GetSectionInfo` 的 revOut。
$reverseLine = $lines | Where-Object { $_ -match '^reverse verified: reversed ' } | Select-Object -First 1
if (-not $reverseLine) { throw 'missing host verification for the reversed item' }
$sectionPattern = '^reverse verified: reversed section=(?<ok>\w+) reversed=(?<rev>\w+) offset=(?<offset>[\d.-]+) length=(?<length>[\d.-]+)$'
if ($reverseLine -notmatch $sectionPattern) { throw "unparsable reversed verification: $reverseLine" }
if ($Matches.ok -ne 'true' -or $Matches.rev -ne 'true') {
    throw "reversed item did not verify as reversed: $reverseLine"
}

# 2) take 事实（D_STARTOFFS / D_LENGTH）——佐证，不参与判据。
$factsPattern = '^take facts: (?<label>forward|reversed) startoffs=(?<startoffs>[\d.-]+) playrate=(?<playrate>[\d.-]+) length=(?<length>[\d.-]+)$'
$takeFacts = @{}
foreach ($line in $lines) {
    if ($line -match $factsPattern) {
        $takeFacts[$Matches.label] = [ordered]@{
            startoffs = Parse-Invariant $Matches.startoffs
            playrate = Parse-Invariant $Matches.playrate
            length = Parse-Invariant $Matches.length
        }
    }
}

# 3) 插件日志里的两条 region。
$pluginLines = Get-Content -LiteralPath $PluginLogPath
$regionPattern = 'playback_region #(?<index>\d+): source=(?<source>.+) startMod=(?<startMod>[\d.-]+) durationMod=(?<durMod>[\d.-]+) startPlay=(?<startPlay>[\d.-]+) durationPlay=(?<durPlay>[\d.-]+) flags=0x(?<flags>[0-9A-Fa-f]+)'
$regions = @($pluginLines | ForEach-Object {
    if ($_ -match $regionPattern) {
        [ordered]@{
            index = [int]$Matches.index
            audioSourcePersistentID = $Matches.source
            startInModificationTime = Parse-Invariant $Matches.startMod
            durationInModificationTime = Parse-Invariant $Matches.durMod
            startInPlaybackTime = Parse-Invariant $Matches.startPlay
            durationInPlaybackTime = Parse-Invariant $Matches.durPlay
            transformationFlags = [Convert]::ToInt32($Matches.flags, 16)
        }
    }
})
if ($regions.Count -ne 2) { throw "expected exactly two regions, saw $($regions.Count)" }

$forwardRegion = $regions | Where-Object { Close-Enough $_.startInPlaybackTime $ForwardPosition } | Select-Object -First 1
$reverseRegion = $regions | Where-Object { Close-Enough $_.startInPlaybackTime $ReversePosition } | Select-Object -First 1
if (-not $forwardRegion) { throw "no region at the forward item position $ForwardPosition" }
if (-not $reverseRegion) { throw "no region at the reversed item position $ReversePosition" }
if (@($regions.audioSourcePersistentID | Select-Object -Unique).Count -ne 1) {
    throw 'expected one shared audio source for this reverse experiment'
}

# 4) 源长（秒）——从夹具 WAV 头取。F-4 的镜像公式需要它。
$stream = [System.IO.File]::OpenRead($FixturePath)
$reader = [System.IO.BinaryReader]::new($stream)
try {
    if ([text.encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'RIFF') { throw 'fixture is not RIFF' }
    $null = $reader.ReadUInt32()
    if ([text.encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'WAVE') { throw 'fixture is not WAVE' }
    $sampleRate = 0; $channels = 0; $bits = 0; $frames = 0
    while ($stream.Position -lt $stream.Length) {
        $chunk = [text.encoding]::ASCII.GetString($reader.ReadBytes(4))
        $length = $reader.ReadUInt32()
        $next = $stream.Position + $length + ($length % 2)
        if ($chunk -eq 'fmt ') {
            $null = $reader.ReadUInt16()          # format tag
            $channels = $reader.ReadUInt16()
            $sampleRate = $reader.ReadUInt32()
            $null = $reader.ReadUInt32()
            $null = $reader.ReadUInt16()
            $bits = $reader.ReadUInt16()
        } elseif ($chunk -eq 'data') {
            $frames = [int]($length / ($bits / 8 * $channels))
        }
        $stream.Position = $next
    }
} finally {
    $reader.Dispose()
    $stream.Dispose()
}
if ($sampleRate -le 0 -or $frames -le 0) { throw 'fixture header is incomplete' }
$sourceSeconds = $frames / $sampleRate

# 5) 分类。
$forwardStart = $forwardRegion.startInModificationTime
$expectedMirroredStart = $sourceSeconds - $forwardStart - $forwardRegion.durationInModificationTime
$actual = $reverseRegion.startInModificationTime

# 正放 region 的起点必须就是裁切起点 —— 否则"对照"本身站不住。
if (-not (Close-Enough $forwardStart $SourceOffset)) {
    throw "forward region startMod ($forwardStart) is not the trim offset ($SourceOffset); probe geometry is wrong"
}
if (-not (Close-Enough $reverseRegion.durationInModificationTime $ItemLength)) {
    throw "reversed region durationMod ($($reverseRegion.durationInModificationTime)) is not the item length ($ItemLength)"
}

$coordinateSystem = $null
if (Close-Enough $actual $forwardStart) {
    $coordinateSystem = 'forward'
} elseif (Close-Enough $actual $expectedMirroredStart) {
    $coordinateSystem = 'mirrored'
} else {
    throw ("reverse region startMod ($actual) is neither the forward value ($forwardStart) " +
        "nor the mirrored value ($expectedMirroredStart); F-4 needs a human")
}

$result = [ordered]@{
    provenance = 'Classified from captures/reverse-script.log and captures/reverse-plugin.log against fixtures/phase3a-asymmetric.wav'
    host = 'REAPER (record the version on the run that produced the capture)'
    reverseAction = 41051
    reverseVerifiedByHost = $true
    fixture = [ordered]@{
        path = $FixturePath
        sampleRate = $sampleRate
        frames = $frames
        durationSeconds = $sourceSeconds
    }
    probeGeometry = [ordered]@{
        sourceOffset = $SourceOffset
        itemLength = $ItemLength
        forwardPosition = $ForwardPosition
        reversePosition = $ReversePosition
    }
    takeFacts = $takeFacts
    forwardRegion = $forwardRegion
    reverseRegion = $reverseRegion
    expected = [ordered]@{
        forwardStartInModificationTime = $forwardStart
        mirroredStartInModificationTime = $expectedMirroredStart
    }
    conclusion = [ordered]@{
        coordinateSystem = $coordinateSystem
        note = 'Region coordinates only; pair with verify_reverse_window.py for the covered source window.'
    }
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "F-4 region coordinates = $coordinateSystem (reversed startMod=$actual; forward=$forwardStart mirrored=$expectedMirroredStart)"
