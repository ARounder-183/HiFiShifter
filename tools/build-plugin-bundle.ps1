# 插件 bundle 构建：把 Rust 引擎、原生 loader、原 GUI 产物与模型组装成一个可加载的
# `HiFiShifter.vst3` 目录。只在当前 checkout 生成，不安装到系统 VST 目录、不运行测试。
#
# 【为什么从 probe/ara 搬到这里】本脚本原先叫 `probe/ara/build_embedded_editor.ps1`，
# 而 `probe/ara/README.md` 明写该目录"是一次性实验产物，探针结束后应整体删除，或至少
# 不得被 backend/、frontend/ 引用"。可产品构建（`tools/build-hifishifter.ps1`）却直接
# 调用它 —— 一个声明为可丢弃的目录成了交付链路的必经环节。搬进 `tools/` 之后，
# `probe/` 可以按它自己的说明被删掉而不影响交付。
#
# 本脚本与 `probe/ara/build_embedded_editor.ps1` 的构建语义**逐字一致**（cargo 参数、
# DLL 白名单、模型来源、loader 编译），只有两处有意改动：
#   1. SDK 目录默认值改为仓库相对路径（仍允许环境变量覆盖）；
#   2. 前端产物只拷插件真正需要的 `plugin.html` 与 `assets/`，不再把整套 dist
#      （`index.html` / `detached.html` / `waveform-test.html`）塞进插件 bundle。
[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$Release,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$BundleDirectory = 'embedded-vst3'
)
$ErrorActionPreference = 'Stop'
$pluginBundleRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pluginBundleOutput = Join-Path $pluginBundleRoot ".build-tmp\$BundleDirectory"
# REAPER 会锁住已加载的 DLL：只允许构建一个全新的目录，绝不覆盖正在使用的 bundle。
if ((Get-Process reaper -ErrorAction SilentlyContinue) -and
    ($BundleDirectory -eq 'embedded-vst3' -or (Test-Path -LiteralPath $pluginBundleOutput))) {
    throw 'REAPER is running: only a new, non-existing bundle directory may be built; never replace a loaded bundle.'
}
$pluginBundleBundle = Join-Path $pluginBundleOutput 'HiFiShifter.vst3'
$pluginBundleModule = Join-Path $pluginBundleBundle 'Contents\x86_64-win'
$pluginBundleResources = Join-Path $pluginBundleBundle 'Contents\Resources\frontend'
$pluginBundleProfile = if ($Release) { 'release' } else { 'debug' }
Push-Location $pluginBundleRoot
try {
    . .\tools\msvc-env.ps1
    # vcvars 会重设 TEMP/TMP，必须在其返回后设置工作树私有的临时目录。
    $pluginBundleTemp = Join-Path $pluginBundleRoot ('.build-tmp\plugin-bundle-build-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $pluginBundleTemp -Force | Out-Null
    $env:TEMP = $pluginBundleTemp
    $env:TMP = $pluginBundleTemp
    # SDK 默认位置仍在 probe/ara 下（那是 `tools/prepare-plugin-sdks.ps1` 的落地目录，
    # 由 `tools/plugin-sdks.json` 钉死 commit 与 tree）；CI 通过环境变量覆盖。
    if (!$env:ARA_VST3_SDK_DIR) { $env:ARA_VST3_SDK_DIR = Join-Path $pluginBundleRoot 'probe\ara\rust-path\.third-party\vst3sdk' }
    if (!$env:ARA_SDK_DIR) { $env:ARA_SDK_DIR = Join-Path $pluginBundleRoot 'probe\ara\rust-path\.third-party\ARA_SDK' }

    if (!$SkipFrontend) {
        Push-Location frontend
        try { & npm run build; if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' } }
        finally { Pop-Location }
    }
    if (!(Test-Path -LiteralPath 'frontend\dist\plugin.html')) { throw 'Build the plugin.html frontend entry first.' }
    if (!(Test-Path -LiteralPath 'frontend\dist\assets')) { throw 'frontend\dist\assets is missing; the plugin entry has no scripts to load.' }

    # 不把 app 和 plugin 放在同一次 cargo 调用，避免默认 vslib feature 合并进插件。
    $pluginBundleCargoArgs = @('build', '--manifest-path', 'backend\Cargo.toml', '--offline', '--jobs', '1', '-p', 'hifishifter-plugin')
    if ($Release) { $pluginBundleCargoArgs += '--release' }
    & cargo @pluginBundleCargoArgs
    if ($LASTEXITCODE -ne 0) { throw 'plugin engine build failed' }

    New-Item -ItemType Directory -Force -Path $pluginBundleModule, $pluginBundleResources | Out-Null
    Copy-Item -LiteralPath "backend\target\$pluginBundleProfile\hifishifter_plugin.dll" -Destination (Join-Path $pluginBundleModule 'HiFiShifterEngine.dll') -Force
    # DLL 白名单刻意不含 vslib：它是独立 App 的闭源运行时，插件不分发。
    foreach ($pluginBundleDll in @('onnxruntime.dll', 'DirectML.dll', 'SoundTouchDLL.dll')) {
        $pluginBundleDllPath = Join-Path $pluginBundleRoot "backend\target\$pluginBundleProfile\$pluginBundleDll"
        if (Test-Path -LiteralPath $pluginBundleDllPath) { Copy-Item -LiteralPath $pluginBundleDllPath -Destination (Join-Path $pluginBundleModule $pluginBundleDll) -Force }
    }

    # 只拷插件入口与它引用的 chunk。整套 dist 里的 `index.html` / `detached.html` /
    # `waveform-test.html` 属于独立 App 与开发用页面，进了插件 bundle 既是死重量，
    # 也会让"这个目录里该有什么"变得没有明确清单可对账。
    Copy-Item -LiteralPath 'frontend\dist\plugin.html' -Destination $pluginBundleResources -Force
    Copy-Item -LiteralPath 'frontend\dist\assets' -Destination $pluginBundleResources -Recurse -Force

    # 原 GUI 分析管线需要 FCPE，WORLD 声码器不能替代它；模型沿用独立 App 的权威副本。
    $pluginBundleModels = Join-Path $pluginBundleBundle 'Contents\Resources\models'
    New-Item -ItemType Directory -Force -Path $pluginBundleModels | Out-Null
    foreach ($pluginBundleModel in @('fcpe', 'nsf_hifigan', 'hnsep')) {
        $pluginBundleModelSource = Join-Path $pluginBundleRoot "backend\src-tauri\resources\models\$pluginBundleModel"
        if (!(Test-Path -LiteralPath $pluginBundleModelSource)) { throw "Missing original model resources: $pluginBundleModel" }
        Copy-Item -LiteralPath $pluginBundleModelSource -Destination $pluginBundleModels -Recurse -Force
    }

    # 原生 loader：VST3 宿主加载的是这个薄 DLL，它再按绝对路径拉起邻接的 Rust 引擎。
    & cl.exe /nologo /utf-8 /std:c++17 /LD /MT /EHsc /O2 'backend\hifishifter-plugin\native\module_loader.cpp' "/Fo:$pluginBundleTemp\module_loader.obj" /link "/OUT:$pluginBundleModule\HiFiShifter.vst3" "/IMPLIB:$pluginBundleTemp\HiFiShifterLoader.lib"
    if ($LASTEXITCODE -ne 0) { throw 'native VST3 loader build failed' }
    Write-Output "Built isolated $pluginBundleProfile bundle: $pluginBundleBundle (not installed; tests and REAPER acceptance pending)."
} finally { Pop-Location }
