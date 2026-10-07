# 一次性输出验证器回归：可通过的反向参考与五种实际错输出变异，不能伪报 PASS。
$ErrorActionPreference='Stop'
$verifier=Join-Path $PSScriptRoot 'verify_phase3a_output.ps1'
$scratch=Join-Path $PSScriptRoot '..\..\.build-tmp\phase3a-verifier-tests'
New-Item -ItemType Directory -Force $scratch | Out-Null
$sourcePath=Join-Path $PSScriptRoot 'fixtures\phase3a-asymmetric.wav'
$capturePath=Join-Path $PSScriptRoot 'captures\phase3a-output.wav'
# 首次调用加载实际解析/比较器；真实采集目前倒放失败，这是预期不是忽略错误。
try { & $verifier | Out-Null } catch { if (-not $_.Exception.Message.Contains('Reverse output was forward')) { throw } }
$source=[AraOutputProbe]::Read($sourcePath)
foreach($case in @(
    @{name='valid-reversed'; expected=$null},
    @{name='duplicate-gain'; expected='Normal/crop/gap'},
    @{name='silent'; expected='Normal/crop/gap'},
    @{name='forward-reverse'; expected='Reverse output was forward'},
    @{name='wrong-seek'; expected='Normal/crop/gap'},
    @{name='truncated'; expected='unexpected capture geometry'}
)) {
    $output=[AraOutputProbe]::Read($capturePath)
    [AraOutputProbe]::Mutate($source,$output,$case.name)
    $path=Join-Path $scratch ($case.name+'.wav')
    [AraOutputProbe]::Write($path,$output)
    $failure=$null
    try { & $verifier -OutputPath $path -ReportPath (Join-Path $scratch ($case.name+'.json')) | Out-Null } catch { $failure=$_.Exception.Message }
    if ($null -eq $case.expected) {
        if($failure) { throw "Valid reversed oracle failed: $failure" }
    } elseif (-not $failure -or -not $failure.Contains($case.expected)) { throw "Mutation $($case.name) was not rejected correctly: $failure" }
    Write-Output "PASS $($case.name)"
}
