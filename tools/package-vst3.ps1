[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$DeliveryDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [switch]$NoZip,
    [switch]$Installer
)
$ErrorActionPreference = 'Stop'
$packageTaskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$packageTaskDelivery = [IO.Path]::GetFullPath($DeliveryDirectory)
$packageTaskOutput = [IO.Path]::GetFullPath($OutputDirectory)
foreach ($packageTaskPath in @($packageTaskDelivery, $packageTaskOutput)) {
    if (!$packageTaskPath.StartsWith($packageTaskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Package paths must remain inside this checkout.'
    }
}
$packageTaskManifest = Get-Content -LiteralPath (Join-Path $packageTaskDelivery 'build-manifest.json') -Raw | ConvertFrom-Json
$packageTaskBundle = Join-Path $packageTaskDelivery 'HiFiShifter.vst3'
$packageTaskFiles = @($packageTaskManifest.files | Where-Object { $_.path.StartsWith('HiFiShifter.vst3\') })
if (!$packageTaskFiles.Count -or $packageTaskManifest.configuration -ne 'Release') { throw 'A complete Release delivery is required.' }
if (@(Get-ChildItem -LiteralPath $packageTaskBundle -Recurse -File).Count -ne $packageTaskFiles.Count) { throw 'Plugin file count differs from manifest.' }
foreach ($packageTaskFile in $packageTaskFiles) {
    if ((Get-FileHash -LiteralPath (Join-Path $packageTaskDelivery $packageTaskFile.path) -Algorithm SHA256).Hash -ne $packageTaskFile.sha256) {
        throw "Plugin hash mismatch: $($packageTaskFile.path)"
    }
}
foreach ($packageTaskRequired in @('Contents\x86_64-win\HiFiShifter.vst3', 'Contents\x86_64-win\HiFiShifterEngine.dll',
    'Contents\x86_64-win\SoundTouchDLL.dll', 'Contents\Resources\frontend\plugin.html',
    'Contents\Resources\models\fcpe\fcpe.onnx', 'Contents\Resources\models\hnsep\hnsep.onnx',
    'Contents\Resources\models\nsf_hifigan\pc_nsf_hifigan.onnx')) {
    if (!(Test-Path -LiteralPath (Join-Path $packageTaskBundle $packageTaskRequired))) { throw "Missing package resource: $packageTaskRequired" }
}
if (Get-ChildItem -LiteralPath $packageTaskBundle -Recurse -File -Filter '*vslib*') { throw 'VST3 package must not include the App-only vslib runtime.' }
$packageTaskVersion = (Get-Content -LiteralPath (Join-Path $packageTaskRoot 'frontend\package.json') -Raw | ConvertFrom-Json).version
if ($packageTaskVersion -notmatch '^[a-zA-Z0-9.+-]+$') { throw 'Invalid package version.' }
$packageTaskName = "HiFiShifter_v${packageTaskVersion}_windows-x86_64-vst3"
$packageTaskStage = Join-Path $packageTaskOutput ($packageTaskName + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $packageTaskStage -Force | Out-Null
Copy-Item -LiteralPath $packageTaskBundle -Destination $packageTaskStage -Recurse
Copy-Item -LiteralPath (Join-Path $packageTaskRoot 'LICENSE') -Destination $packageTaskStage
$packageTaskManifest.target = 'Plugin'
$packageTaskManifest.files = $packageTaskFiles
$packageTaskManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $packageTaskStage 'build-manifest.json') -Encoding utf8
$packageTaskArchive = Join-Path $packageTaskOutput ($packageTaskName + '.zip')
if (!$NoZip) {
Compress-Archive -LiteralPath (Join-Path $packageTaskStage 'HiFiShifter.vst3'), (Join-Path $packageTaskStage 'LICENSE'),
    (Join-Path $packageTaskStage 'build-manifest.json') -DestinationPath $packageTaskArchive -Force
Write-Output "Packaged: $packageTaskArchive"
}
if ($Installer) {
    $packageTaskNsisCommand = Get-Command makensis -ErrorAction SilentlyContinue
    $packageTaskNsis = if ($packageTaskNsisCommand) { $packageTaskNsisCommand.Source } else { $null }
    if (!$packageTaskNsis) {
        foreach ($packageTaskNsisCandidate in @((Join-Path ${env:ProgramFiles(x86)} 'NSIS\makensis.exe'),
            (Join-Path $env:LOCALAPPDATA 'tauri\NSIS\makensis.exe'))) {
            if (Test-Path -LiteralPath $packageTaskNsisCandidate) { $packageTaskNsis = $packageTaskNsisCandidate; break }
        }
    }
    if (!$packageTaskNsis) { throw 'NSIS is required for -Installer. Install NSIS or package ZIP without that switch.' }
    # 安装器文案由五语词表生成：先重新生成一次，保证打出来的安装器与当前词表一致
    # （漏翻译会在这里直接失败，而不是让用户看到半截翻译的向导）。
    & node (Join-Path $PSScriptRoot 'build-installer-strings.mjs')
    if ($LASTEXITCODE -ne 0) { throw 'Installer string generation failed.' }
    $packageTaskSetup = Join-Path $packageTaskOutput ($packageTaskName + '-setup.exe')
    # LICENSE 用绝对路径传入：`vst3-installer.nsi` 里的默认值 `..\LICENSE` 只在从
    # 仓库的 tools\ 目录编译时成立。
    & $packageTaskNsis /INPUTCHARSET UTF8 "/DPLUGIN_BUNDLE=$packageTaskBundle" "/DPLUGIN_VERSION=$packageTaskVersion" "/DOUTPUT_FILE=$packageTaskSetup" "/DHFS_LICENSE=$(Join-Path $packageTaskRoot 'LICENSE')" (Join-Path $PSScriptRoot 'vst3-installer.nsi')
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $packageTaskSetup)) { throw 'VST3 installer compilation failed.' }
    Write-Output "Installer: $packageTaskSetup"
}
if ($NoZip) { Write-Output "Staged VST3 package: $packageTaskStage" }
else {
    $packageTaskResolvedStage = [IO.Path]::GetFullPath($packageTaskStage)
    if (!$packageTaskResolvedStage.StartsWith($packageTaskOutput + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Package scratch directory escaped output directory.'
    }
    Remove-Item -LiteralPath $packageTaskResolvedStage -Recurse -Force
}
