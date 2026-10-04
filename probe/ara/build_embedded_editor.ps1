# 内嵌原GUI开发bundle构建；只在当前worktree生成，不安装到系统VST目录，不运行测试。
param([switch]$SkipFrontend,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$BundleDirectory='embedded-vst3')
$ErrorActionPreference = 'Stop'
$araEmbeddedRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$araEmbeddedOutput = Join-Path $araEmbeddedRoot ".build-tmp\$BundleDirectory"
if ((Get-Process reaper -ErrorAction SilentlyContinue) -and
    ($BundleDirectory -eq 'embedded-vst3' -or (Test-Path -LiteralPath $araEmbeddedOutput))) {
    throw 'REAPER is running: only a new, non-existing bundle directory may be built; never replace a loaded bundle.'
}
$araEmbeddedBundle = Join-Path $araEmbeddedOutput 'HiFiShifter.vst3'
$araEmbeddedModule = Join-Path $araEmbeddedBundle 'Contents\x86_64-win'
$araEmbeddedResources = Join-Path $araEmbeddedBundle 'Contents\Resources\frontend'
Push-Location $araEmbeddedRoot
try {
    . .\tools\msvc-env.ps1
    $araEmbeddedTemp = Join-Path $araEmbeddedRoot ('.build-tmp\embedded-build-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $araEmbeddedTemp -Force | Out-Null
    $env:TEMP = $araEmbeddedTemp
    $env:TMP = $araEmbeddedTemp
    $env:ARA_VST3_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\vst3sdk'
    $env:ARA_SDK_DIR = Join-Path $PSScriptRoot 'rust-path\.third-party\ARA_SDK'
    if (!$SkipFrontend) {
        Push-Location frontend
        try { & npm run build; if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' } }
        finally { Pop-Location }
    }
    if (!(Test-Path -LiteralPath 'frontend\dist\plugin.html')) { throw 'Build the plugin.html frontend entry first.' }
    # 不把app和plugin放在同一次cargo调用，避免默认vslib feature合并进插件。
    & cargo build --manifest-path backend\Cargo.toml --offline --jobs 1 -p hifishifter-plugin
    if ($LASTEXITCODE -ne 0) { throw 'plugin engine build failed' }
    New-Item -ItemType Directory -Force -Path $araEmbeddedModule,$araEmbeddedResources | Out-Null
    Copy-Item -LiteralPath 'backend\target\debug\hifishifter_plugin.dll' -Destination (Join-Path $araEmbeddedModule 'HiFiShifterEngine.dll') -Force
    foreach ($araEmbeddedDll in @('onnxruntime.dll','DirectML.dll','SoundTouchDLL.dll')) {
        $araEmbeddedDllPath = Join-Path $araEmbeddedRoot "backend\target\debug\$araEmbeddedDll"
        if (Test-Path -LiteralPath $araEmbeddedDllPath) { Copy-Item -LiteralPath $araEmbeddedDllPath -Destination (Join-Path $araEmbeddedModule $araEmbeddedDll) -Force }
    }
    Get-ChildItem -LiteralPath 'frontend\dist' | Copy-Item -Destination $araEmbeddedResources -Recurse -Force
    # 原GUI分析管线需要FCPE，WORLD声码器不能替代它；模型沿用独立app权威副本。
    $araEmbeddedModels = Join-Path $araEmbeddedBundle 'Contents\Resources\models'
    New-Item -ItemType Directory -Force -Path $araEmbeddedModels | Out-Null
    foreach ($araEmbeddedModel in @('fcpe','nsf_hifigan','hnsep')) {
        $araEmbeddedModelSource = Join-Path $araEmbeddedRoot "backend\src-tauri\resources\models\$araEmbeddedModel"
        if (!(Test-Path -LiteralPath $araEmbeddedModelSource)) {throw "Missing original model resources: $araEmbeddedModel"}
        Copy-Item -LiteralPath $araEmbeddedModelSource -Destination $araEmbeddedModels -Recurse -Force
    }
    & cl.exe /nologo /utf-8 /std:c++17 /LD /MT /EHsc /O2 'backend\hifishifter-plugin\native\module_loader.cpp' "/Fo:$araEmbeddedTemp\module_loader.obj" /link "/OUT:$araEmbeddedModule\HiFiShifter.vst3" "/IMPLIB:$araEmbeddedTemp\HiFiShifterLoader.lib"
    if ($LASTEXITCODE -ne 0) { throw 'native VST3 loader build failed' }
    Write-Output "Built isolated bundle: $araEmbeddedBundle (not installed; tests and REAPER acceptance pending)."
} finally { Pop-Location }
