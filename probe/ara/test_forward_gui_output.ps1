# 一次性正向GUI输出验证器回归；用真实音频数组证明不能把音量变化冒充音高验收。
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\verify_forward_gui_output.ps1" -LibraryOnly

function New-ForwardWave([double]$Frequency, [double]$Gain = 0.2) {
    $wave = [AraForwardGuiOutput+Wave]::new()
    $wave.Rate = 44100
    $wave.Channels = 2
    $wave.Samples = [float[]]::new(44100 * 5 * 2)
    for ($frame = 0; $frame -lt 44100 * 5; $frame++) {
        if ($frame -lt 88200 -or ($frame -ge 132300 -and $frame -lt 176400)) {
            $sample = [float]($Gain * [Math]::Sin(2 * [Math]::PI * $Frequency * $frame / 44100))
            $wave.Samples[$frame * 2] = $sample
            $wave.Samples[$frame * 2 + 1] = $sample
        }
    }
    return $wave
}

function Assert-ForwardRejects([scriptblock]$Operation, [string]$Label) {
    $rejected = $false
    try { & $Operation | Out-Null } catch { $rejected = $true }
    if (!$rejected) { throw "Validator accepted $Label" }
}

$baseline = New-ForwardWave 220
$edited = New-ForwardWave 330
$result = [AraForwardGuiOutput]::Evaluate($baseline, $edited, $edited)
if ($result.PitchChangedWindows -lt 1 -or $result.ReopenMaxDifference -ne 0 -or $result.EditedRms -le 0.01) { throw 'Actual pitch shift or exact reopen was not recognized' }
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($baseline, $baseline, $null) } 'unchanged output'
$gainOnly = New-ForwardWave 220 0.1
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($baseline, $gainOnly, $null) } 'gain-only output'
$gap = New-ForwardWave 330
$gap.Samples[110250 * 2] = 0.2
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($baseline, $gap, $null) } 'nonzero gap'
$wrongReopen = New-ForwardWave 440
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($baseline, $edited, $wrongReopen) } 'changed reopened output'
$bad = New-ForwardWave 330
$bad.Samples[0] = [float]::NaN
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($baseline, $bad, $null) } 'nonfinite samples'
$actualBaseline = [AraForwardGuiOutput]::Read("$PSScriptRoot\captures\forward-gui-baseline.wav")
$actualSource = [AraForwardGuiOutput]::Read("$PSScriptRoot\fixtures\forward-gui-voice.wav")
$baselineError = [AraForwardGuiOutput]::CheckBaseline($actualSource, $actualBaseline)
"6 forward GUI validator cases plus real WAV source/crop/gap oracle passed; baseline max_abs=$baselineError"

# 本轮真实GUI把第一段移到1秒；新增布局必须验证原PCM，不重排音频掩盖错位。
$movedBaseline=[AraForwardGuiOutput]::Read("$PSScriptRoot\captures\gui-keyboard-baseline.wav")
$movedEdited=[AraForwardGuiOutput]::Read("$PSScriptRoot\captures\gui-keyboard-edited.wav")
$movedSource=[AraForwardGuiOutput]::Read("$PSScriptRoot\fixtures\embedded-editor-voice.wav")
$movedError=[AraForwardGuiOutput]::CheckBaseline($movedSource,$movedBaseline,1)
$movedResult=[AraForwardGuiOutput]::Evaluate($movedBaseline,$movedEdited,$movedEdited,1)
if($movedError -gt 1e-6 -or $movedResult.PitchChangedWindows -ne 4 -or
    ($movedResult.EditedHz | Where-Object { [Math]::Abs($_ - 329.63) -gt 1 }).Count -ne 0) {throw 'Moved real GUI pitch fixture was not recognized'}
Assert-ForwardRejects { [AraForwardGuiOutput]::CheckBaseline($movedSource,$movedBaseline,0) } 'wrong first clip placement'
$movedEdited.Samples[1000*2]=0.2
Assert-ForwardRejects { [AraForwardGuiOutput]::Evaluate($movedBaseline,$movedEdited,$null,1) } 'nonzero leading gap after host move'
'3 relocated real GUI fixture cases passed'
