# Windows统一产品构建流：同一frontend/kernel分别生成独立App和原GUI插件，绝不自动安装或启动。
[CmdletBinding()]
param(
    [ValidateSet('All','App','Plugin')][string]$Target='All',
    [ValidateSet('Debug','Release')][string]$Configuration='Release',
    [Alias('Name')][ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$BuildName,
    [switch]$PlanOnly,
    [switch]$Verify
)
$ErrorActionPreference='Stop'
$buildTaskRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (!$BuildName) {$BuildName='hfs-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8)}
$buildTaskDelivery=Join-Path $buildTaskRoot ".build-tmp\deliveries\$BuildName"
$buildTaskBundleName="$BuildName-vst3"
if ($buildTaskBundleName.Length -gt 64) {throw 'Name is too long for the isolated plugin bundle (maximum 59 characters).'}
if (Test-Path -LiteralPath $buildTaskDelivery) {throw "Delivery already exists; choose a fresh Name: $buildTaskDelivery"}
$buildTaskIncludesApp=$Target -in @('All','App')
$buildTaskIncludesPlugin=$Target -in @('All','Plugin')
if ($buildTaskIncludesPlugin -and (Test-Path -LiteralPath (Join-Path $buildTaskRoot ".build-tmp\$buildTaskBundleName"))) {
    throw 'Plugin bundle already exists; immutable build requires a fresh Name.'
}
foreach ($buildTaskRequired in @('frontend\package-lock.json','backend\Cargo.lock','tools\msvc-env.ps1',
    'backend\src-tauri\resources\models\fcpe\fcpe.onnx','backend\src-tauri\resources\models\hnsep\hnsep.onnx',
    'backend\src-tauri\resources\models\nsf_hifigan\pc_nsf_hifigan.onnx')) {
    if (!(Test-Path -LiteralPath (Join-Path $buildTaskRoot $buildTaskRequired))) {throw "Missing input: $buildTaskRequired"}
}
if ($buildTaskIncludesPlugin) {
    foreach ($buildTaskSdk in @('ARA_SDK','vst3sdk')) {
        if (!(Test-Path -LiteralPath (Join-Path $buildTaskRoot "probe\ara\rust-path\.third-party\$buildTaskSdk\.git"))) {
            throw "Missing locked SDK checkout: $buildTaskSdk; this flow never clones or edits SDKs."
        }
    }
}
Write-Output "Plan: frontend once -> $Target ($Configuration), separate App/plugin Cargo invocations, output=$buildTaskDelivery"
if ($PlanOnly) {return}
if (!(Get-Command cargo -ErrorAction SilentlyContinue) -or !(Get-Command npm -ErrorAction SilentlyContinue)) {throw 'cargo and npm are required.'}

# 中文：跟踪及未跟踪源码都参与构建身份；模型虽被git忽略仍是实际产品输入。
function Get-BuildSourceFingerprint {
    $buildFingerprintPaths=@(& git ls-files --cached --others --exclude-standard -- backend frontend tools)
    if ($LASTEXITCODE -ne 0) {throw 'Cannot enumerate product source files.'}
    $buildFingerprintPaths+=@(Get-ChildItem -LiteralPath (Join-Path $buildTaskRoot 'backend\src-tauri\resources\models') -Recurse -File |
        ForEach-Object {$_.FullName.Substring($buildTaskRoot.Length+1)})
    $buildFingerprintRows=@($buildFingerprintPaths | Sort-Object -Unique | ForEach-Object {
        $buildFingerprintPath=Join-Path $buildTaskRoot $_
        if (Test-Path -LiteralPath $buildFingerprintPath -PathType Leaf) {"$_ $((Get-FileHash -LiteralPath $buildFingerprintPath -Algorithm SHA256).Hash)"}
    })
    $buildFingerprintHasher=[Security.Cryptography.SHA256]::Create()
    try {return ([BitConverter]::ToString($buildFingerprintHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes(($buildFingerprintRows -join "`n"))))).Replace('-','')}
    finally {$buildFingerprintHasher.Dispose()}
}

Push-Location $buildTaskRoot
try {
    # 记录调用起点，构建中工作树若被修改则失败；交付不能声称对应某个不存在的源码快照。
    $buildTaskCommit=& git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {throw 'Cannot resolve source commit.'}
    $buildTaskState=& git status --porcelain
    if ($LASTEXITCODE -ne 0) {throw 'Cannot inspect working changes.'}
    $buildTaskFingerprint=Get-BuildSourceFingerprint
    . .\tools\msvc-env.ps1
    # vcvars会重设TEMP/TMP，必须在其返回后设置新的工作树私有目录。
    $buildTaskTemp=Join-Path $buildTaskRoot ('.build-tmp\product-build-'+[Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $buildTaskTemp | Out-Null
    $env:TEMP=$buildTaskTemp;$env:TMP=$buildTaskTemp
    $env:ARA_SDK_DIR=Join-Path $buildTaskRoot 'probe\ara\rust-path\.third-party\ARA_SDK'
    $env:ARA_VST3_SDK_DIR=Join-Path $buildTaskRoot 'probe\ara\rust-path\.third-party\vst3sdk'
    if ($Verify) {
        & npm --prefix frontend test
        if ($LASTEXITCODE -ne 0) {throw 'Frontend regression failed.'}
        # App默认feature与插件无vslib配置分开，防止Cargo特性统一把闭源功能带进插件。
        & cargo test --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-kernel --lib -- --test-threads=1
        if ($LASTEXITCODE -ne 0) {throw 'Shared kernel/App feature regression failed.'}
        & cargo test --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-kernel --no-default-features --features onnx --lib -- --test-threads=1
        if ($LASTEXITCODE -ne 0) {throw 'Shared kernel/plugin feature regression failed.'}
        if ($buildTaskIncludesApp) {
            & cargo test --manifest-path backend\Cargo.toml --offline --jobs 1 -p HiFiShifter --lib -- --test-threads=1
            if ($LASTEXITCODE -ne 0) {throw 'Standalone App regression failed.'}
        }
        if ($buildTaskIncludesPlugin) {
            & cargo test --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-plugin --lib -- --test-threads=1
            if ($LASTEXITCODE -ne 0) {throw 'Plugin/ABI regression failed.'}
        }
    }
    & npm --prefix frontend run build
    if ($LASTEXITCODE -ne 0) {throw 'Shared frontend build failed.'}
    New-Item -ItemType Directory -Path $buildTaskDelivery | Out-Null
    $buildTaskProfile=$Configuration.ToLowerInvariant()
    if ($buildTaskProfile -eq 'debug') {$buildTaskProfile='debug'}
    if ($buildTaskIncludesApp) {
        Push-Location backend
        try {
            # 已核对cargo tauri build --help：--config合并仅跳过刚执行过的前端构建。
            $buildTaskAppArgs=@('tauri','build','--ci','--no-bundle','--config','{"build":{"beforeBuildCommand":""}}')
            if ($Configuration -eq 'Debug') {$buildTaskAppArgs+='--debug'}
            $buildTaskAppArgs+=@('--','--offline','--jobs','1')
            & cargo @buildTaskAppArgs
            if ($LASTEXITCODE -ne 0) {throw 'Standalone App build failed.'}
        } finally {Pop-Location}
        $buildTaskApp=Join-Path $buildTaskDelivery 'app'
        New-Item -ItemType Directory -Path $buildTaskApp | Out-Null
        Copy-Item -LiteralPath "backend\target\$buildTaskProfile\HiFiShifter.exe" -Destination $buildTaskApp
        # App保留其默认vslib能力；插件打包 helper的DLL白名单不包含它。
        foreach ($buildTaskDll in @('onnxruntime.dll','DirectML.dll','SoundTouchDLL.dll','vslib_x64.dll')) {
            $buildTaskDllSource=Join-Path $buildTaskRoot "backend\target\$buildTaskProfile\$buildTaskDll"
            if (Test-Path -LiteralPath $buildTaskDllSource) {Copy-Item -LiteralPath $buildTaskDllSource -Destination $buildTaskApp}
        }
        Copy-Item -LiteralPath backend\src-tauri\resources\models -Destination (Join-Path $buildTaskApp 'models') -Recurse
    }
    if ($buildTaskIncludesPlugin) {
        $buildTaskPluginArgs=@{SkipFrontend=$true;BundleDirectory=$buildTaskBundleName}
        if ($Configuration -eq 'Release') {$buildTaskPluginArgs.Release=$true}
        # 复用已实测的规范loader/resources打包，所有源码仍唯一；以后可将helper提升到tools。
        & .\probe\ara\build_embedded_editor.ps1 @buildTaskPluginArgs
        $buildTaskPlugin=Join-Path $buildTaskRoot ".build-tmp\$buildTaskBundleName\HiFiShifter.vst3"
        $buildTaskPluginTarget=[IO.Path]::GetFullPath((Join-Path $buildTaskDelivery 'HiFiShifter.vst3'))
        $buildTaskPluginSource=[IO.Path]::GetFullPath($buildTaskPlugin)
        $buildTaskAllowedPrefix=[IO.Path]::GetFullPath((Join-Path $buildTaskRoot '.build-tmp'))+[IO.Path]::DirectorySeparatorChar
        if (!$buildTaskPluginSource.StartsWith($buildTaskAllowedPrefix,[StringComparison]::OrdinalIgnoreCase) -or
            !$buildTaskPluginTarget.StartsWith($buildTaskAllowedPrefix,[StringComparison]::OrdinalIgnoreCase)) {throw 'Plugin move escaped the intended build workspace.'}
        Move-Item -LiteralPath $buildTaskPluginSource -Destination $buildTaskPluginTarget
    }
    if ($Verify -and $buildTaskIncludesPlugin) {
        # 原生导出测试明确读取debug cdylib，即使交付是release也构建其独立测试前置。
        & cargo build --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-plugin
        if ($LASTEXITCODE -ne 0) {throw 'Plugin ABI regression prerequisite build failed.'}
        foreach ($buildTaskContract in @('ara_mapping','renderer_assignments','vst3_exports','no_tauri_in_dependency_tree')) {
            & cargo test --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-plugin --test $buildTaskContract -- --test-threads=1
            if ($LASTEXITCODE -ne 0) {throw "Plugin integration contract failed: $buildTaskContract"}
        }
    }
    $buildTaskAfterCommit=& git rev-parse HEAD
    if ($buildTaskAfterCommit -ne $buildTaskCommit -or (Get-BuildSourceFingerprint) -ne $buildTaskFingerprint) {
        throw 'Product source/model changed while building; output is incomplete evidence, rebuild using a fresh Name.'
    }
    $buildTaskFiles=@(Get-ChildItem -LiteralPath $buildTaskDelivery -Recurse -File | ForEach-Object {
        [ordered]@{path=$_.FullName.Substring($buildTaskDelivery.Length+1);bytes=$_.Length;
            sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
    })
    $buildTaskManifest=[ordered]@{schema=1;sourceCommit=$buildTaskCommit;sourceFingerprint=$buildTaskFingerprint;workingTreeModified=[bool]$buildTaskState;
        target=$Target;configuration=$Configuration;verificationRequested=[bool]$Verify;nativeAcceptance=$false;
        createdUtc=[DateTime]::UtcNow.ToString('o');files=$buildTaskFiles}
    [IO.File]::WriteAllText((Join-Path $buildTaskDelivery 'build-manifest.json'),($buildTaskManifest|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
    Write-Output "Built: $buildTaskDelivery. No installation, REAPER launch or native acceptance was performed."
} finally {Pop-Location}
