# 产品构建：前端 → App → VST3 bundle → dist\ 下的 zip 与安装器。
#
# 产物只有一处：dist\。没有中间交付目录、没有构建清单、没有源码指纹 —— 构建产物的身份
# 由它来自的那个 git 提交决定，不需要再写一份 JSON 去证明。
#
#   .\scripts\build\build.ps1                   # App + VST3，Release
#   .\scripts\build\build.ps1 -Target Plugin    # 只出 VST3
#   .\scripts\build\build.ps1 -Target App       # 只出 App
#   .\scripts\build\build.ps1 -SkipFrontend     # frontend\dist 已是最新
#   .\scripts\build\build.ps1 -Configuration Debug   # debug 产物，不打 zip / 安装器
#
# 需要：MSVC（cl.exe）、cargo、npm、NSIS。见 scripts/setup/install_deps_windows.ps1。
[CmdletBinding()]
param(
    [ValidateSet('All', 'App', 'Plugin')][string]$Target = 'All',
    [switch]$SkipFrontend,
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

. .\scripts\env\msvc-env.ps1
. .\scripts\env\sdk-env.ps1
Set-SdkEnvironment

function Assert-Ok($what) {
    if ($LASTEXITCODE) { throw "$what 失败（退出码 $LASTEXITCODE）" }
}

$profile = $Configuration.ToLowerInvariant()
$built = Join-Path $root "backend\target\$profile"
$dist = Join-Path $root 'dist'
$version = (Get-Content backend\src-tauri\tauri.conf.json -Raw | ConvertFrom-Json).version
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
# 发行物随附的许可：根 LICENSE（MIT + 第三方清单）与 licenses\ 目录（各许可全文与声明）。
# 只有一处清单，App 与插件共用，避免两套打包路径各记一份。
$licenses = @('LICENSE', 'licenses')
New-Item -ItemType Directory -Force $dist | Out-Null

# ── 前端 ────────────────────────────────────────────────────────────────────
# App 与插件共用 frontend\dist，只构建一次。tauri 的 beforeBuildCommand 会再跑一遍，
# 所以下面用一份空配置把它置空 —— 内联 JSON 在 Windows PowerShell 传给原生进程时会被
# 剥掉内层引号，只能写成文件传路径。
if (!$SkipFrontend) {
    npm --prefix frontend run build
    Assert-Ok '前端构建'
}
if (!(Test-Path frontend\dist\plugin.html)) {
    throw 'frontend\dist 里没有 plugin.html：去掉 -SkipFrontend 重新构建'
}

# ── App ─────────────────────────────────────────────────────────────────────
if ($Target -ne 'Plugin') {
    $blank = Join-Path ([IO.Path]::GetTempPath()) ("hfs-build-" + [Guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::WriteAllText($blank, '{"build":{"beforeBuildCommand":""}}')
    Push-Location backend
    try {
        $tauriArgs = @('tauri', 'build', '--ci', '--bundles', 'nsis', '--config', $blank)
        if ($Configuration -eq 'Debug') { $tauriArgs += '--debug' }
        cargo @tauriArgs
        Assert-Ok 'App 构建'
    }
    finally {
        Pop-Location
        Remove-Item $blank -Force
    }
}

# ── VST3 ────────────────────────────────────────────────────────────────────
if ($Target -ne 'App') {
    $pluginArgs = @('build', '--manifest-path', 'backend\Cargo.toml', '-p', 'hifishifter-plugin')
    if ($Configuration -eq 'Release') { $pluginArgs += '--release' }
    cargo @pluginArgs
    Assert-Ok '插件引擎构建'

    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $module = Join-Path $bundle 'Contents\x86_64-win'
    $resources = Join-Path $bundle 'Contents\Resources'
    Remove-Item $bundle -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $module, "$resources\frontend", "$resources\models" | Out-Null

    Copy-Item "$built\hifishifter_plugin.dll" "$module\HiFiShifterEngine.dll"
    # DLL 白名单刻意不含 vslib：它是独立 App 的闭源运行时，插件不分发。
    foreach ($dll in 'onnxruntime.dll', 'DirectML.dll', 'SoundTouchDLL.dll') {
        if (Test-Path "$built\$dll") { Copy-Item "$built\$dll" $module }
    }

    # 只拷插件入口与它引用的 chunk。index.html / detached.html / waveform-test.html
    # 属于独立 App 与开发页面，进了插件 bundle 既是死重量也让目录清单没法对账。
    Copy-Item frontend\dist\plugin.html "$resources\frontend"
    Copy-Item frontend\dist\assets "$resources\frontend\assets" -Recurse
    Copy-Item backend\src-tauri\icons\icon.ico "$resources\frontend"

    # 原 GUI 的分析管线需要 FCPE，WORLD 声码器不能替代它；模型沿用独立 App 的权威副本。
    # models.json 必须一起进 bundle：运行时靠它判断共享模型库里的那一份是否可用。
    foreach ($model in 'fcpe', 'nsf_hifigan', 'hnsep') {
        Copy-Item "backend\src-tauri\resources\models\$model" "$resources\models\$model" -Recurse
    }
    Copy-Item backend\src-tauri\resources\models\models.json "$resources\models"

    # 原生 loader：宿主加载的是这个薄 DLL，它再按绝对路径拉起邻接的 Rust 引擎。
    $objDir = Join-Path ([IO.Path]::GetTempPath()) ("hfs-loader-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force $objDir | Out-Null
    cl.exe /nologo /utf-8 /std:c++17 /LD /MT /EHsc /O2 backend\hifishifter-plugin\native\module_loader.cpp "/Fo:$objDir\module_loader.obj" /link "/OUT:$module\HiFiShifter.vst3" "/IMPLIB:$objDir\HiFiShifterLoader.lib"
    Assert-Ok '原生 VST3 loader 构建'
    Remove-Item $objDir -Recurse -Force

    # 编译成功不等于能被宿主加载。打包前跑一次黑盒门禁，别让加载不了的产物进 zip。
    & .\scripts\vst3\windows\verify.ps1 -BundleDirectory $bundle
}

if ($Configuration -eq 'Debug') {
    Write-Host "debug 产物已就绪：$dist" -ForegroundColor Green
    return
}

# ── 打包 ────────────────────────────────────────────────────────────────────
if ($Target -ne 'Plugin') {
    $portable = Join-Path $dist 'HiFiShifter'
    Remove-Item $portable -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $portable | Out-Null
    Copy-Item "$built\HiFiShifter.exe" $portable
    foreach ($dll in 'onnxruntime.dll', 'DirectML.dll', 'SoundTouchDLL.dll', 'vslib_x64.dll') {
        if (Test-Path "$built\$dll") { Copy-Item "$built\$dll" $portable }
    }
    Copy-Item backend\src-tauri\resources\models "$portable\models" -Recurse
    Copy-Item $licenses $portable -Recurse

    $portableZip = Join-Path $dist "HiFiShifter-v$version-portable-win-$arch.zip"
    Remove-Item $portableZip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path $portable -DestinationPath $portableZip -CompressionLevel Optimal
    Remove-Item $portable -Recurse -Force

    # App 的 NSIS 安装器由 tauri 产出。安装器是产品产物，缺失即构建失败。
    $appInstaller = Get-ChildItem "$built\bundle\nsis" -Filter '*-setup.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (!$appInstaller) { throw "找不到 App 安装器：$built\bundle\nsis（先跑 scripts/setup/install_deps_windows.ps1 装 NSIS）" }
    Copy-Item $appInstaller.FullName $dist
    Write-Host "便携包：$portableZip" -ForegroundColor Green
}

if ($Target -ne 'App') {
    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $vst3Zip = Join-Path $dist ('HiFiShifter_v' + $version + '_windows-x86_64-vst3.zip')
    Remove-Item $vst3Zip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path (@($bundle) + $licenses) -DestinationPath $vst3Zip
    Write-Host "VST3 包：$vst3Zip" -ForegroundColor Green

    $makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
    if (!$makensis) {
        $candidate = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'NSIS\makensis.exe'
        if (Test-Path $candidate) { $makensis = $candidate }
    }
    if (!$makensis) {
        Write-Host '未安装 NSIS，跳过插件安装器（zip 已生成）' -ForegroundColor DarkGray
        return
    }

    # 安装器版把模型装进全机共享库，bundle 里不留副本：独立 App 也读同一份，
    # 两个形态各带一份就是两份完全相同的 150 MB。ZIP 版仍然自包含。
    $stage = Join-Path ([IO.Path]::GetTempPath()) ("hfs-nsis-" + [Guid]::NewGuid().ToString('N'))
    $stagedBundle = Join-Path $stage 'HiFiShifter.vst3'
    New-Item -ItemType Directory -Force $stage | Out-Null
    Copy-Item $bundle $stagedBundle -Recurse
    $models = Join-Path $stagedBundle 'Contents\Resources\models'
    Move-Item $models (Join-Path $stage 'models')
    $modelsVersion = (Get-Content (Join-Path $stage 'models\models.json') -Raw | ConvertFrom-Json).version

    # 安装器文案由五语词表生成；先重新生成一次，漏翻译在这里就失败。
    node (Join-Path $PSScriptRoot '..\release\build-installer-strings.mjs')
    Assert-Ok '安装器文案生成'

    $setup = Join-Path $dist ('HiFiShifter_v' + $version + '_windows-x86_64-vst3-setup.exe')
    & $makensis /INPUTCHARSET UTF8 "/DPLUGIN_BUNDLE=$stagedBundle" "/DPLUGIN_VERSION=$version" "/DOUTPUT_FILE=$setup" "/DHFS_LICENSE=$(Join-Path $root 'LICENSE')" "/DHFS_LICENSES=$(Join-Path $root 'licenses')" "/DHFS_MODELS=$(Join-Path $stage 'models')" "/DHFS_MODELS_VERSION=$modelsVersion" (Join-Path $PSScriptRoot '..\release\vst3-installer.nsi')
    Assert-Ok '插件安装器编译'
    Remove-Item $stage -Recurse -Force
    Write-Host "插件安装器：$setup" -ForegroundColor Green
}
