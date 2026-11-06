# 产品构建：前端 → App → VST3 bundle → 安装器暂存 → 合并安装器 / 便携包。
#
# 产物只有一处：dist\。没有中间交付目录、没有构建清单、没有源码指纹 —— 构建产物的身份
# 由它来自的那个 git 提交决定，不需要再写一份 JSON 去证明。
#
#   .\scripts\build\build.ps1                     # App + VST3 + 合并安装器 + 便携包
#   .\scripts\build\build.ps1 -Target App         # 只出 App 可执行文件（不打安装器）
#   .\scripts\build\build.ps1 -Target Plugin      # 只出 VST3 bundle
#   .\scripts\build\build.ps1 -Target Installer   # 只重打安装器（复用 dist\_stage）
#   .\scripts\build\build.ps1 -SkipFrontend       # frontend\dist 已是最新
#   .\scripts\build\build.ps1 -Configuration Debug   # debug 产物，不打包
#
# 【为什么 Windows 只有一个安装器】独立 App 与 VST3 插件的用户是同一批人，而它们要用的
# 是同一批 150 MB 模型。分成两个安装包就只有两种结局：要么用户为完全相同的权重付两次
# 磁盘，要么两个包各写半套逻辑去猜对方装没装。合并之后模型只写一次，卸载也只有一个入口。
#
# 需要：MSVC（cl.exe）、cargo、npm、NSIS。见 scripts/setup/install_deps_windows.ps1。
[CmdletBinding()]
param(
    [ValidateSet('All', 'App', 'Plugin', 'Installer')][string]$Target = 'All',
    [switch]$SkipFrontend,
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    # 离线 / 受控环境可以显式指定已下载好的 WebView2 bootstrapper（见 Get-WebView2Bootstrapper）。
    [string]$WebView2Bootstrapper = ''
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

function Reset-Directory([string]$path) {
    Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $path | Out-Null
}

# 模型权重不在仓库里（见 scripts/setup/fetch-models.py）：缺了就取，已有则是一次
# 目录扫描。CI 用 actions/cache 以清单哈希为键缓存解压结果，因此这里通常什么都不做。
python (Join-Path $PSScriptRoot '..\setup\fetch-models.py')
python (Join-Path $PSScriptRoot '..\setup\fetch-models.py') --fetch-game
Assert-Ok '模型权重'

$profile = $Configuration.ToLowerInvariant()
$built = Join-Path $root "backend\target\$profile"
$dist = Join-Path $root 'dist'
$stage = Join-Path $dist '_stage\installer'
$version = (Get-Content backend\src-tauri\tauri.conf.json -Raw | ConvertFrom-Json).version
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
# 发行物随附的许可：根 LICENSE（MIT + 第三方清单）与 licenses\ 目录（各许可全文与声明）。
# 只有一处清单，App 与插件共用，避免两套打包路径各记一份。
$licenses = @('LICENSE', 'licenses')
$modelsSource = Join-Path $root 'backend\src-tauri\resources\models'
$modelsManifest = Get-Content (Join-Path $modelsSource 'models.json') -Raw | ConvertFrom-Json
$modelsVersion = $modelsManifest.version
New-Item -ItemType Directory -Force $dist | Out-Null

# ── 前端 ────────────────────────────────────────────────────────────────────
# App 与插件共用 frontend\dist，只构建一次。tauri 的 beforeBuildCommand 会再跑一遍，
# 所以下面用一份空配置把它置空 —— 内联 JSON 在 Windows PowerShell 传给原生进程时会被
# 剥掉内层引号，只能写成文件传路径。
if ($Target -ne 'Installer' -and !$SkipFrontend) {
    npm --prefix frontend run build
    Assert-Ok '前端构建'
}
if ($Target -ne 'Installer' -and !(Test-Path frontend\dist\plugin.html)) {
    throw 'frontend\dist 里没有 plugin.html：去掉 -SkipFrontend 重新构建'
}

# ── App ─────────────────────────────────────────────────────────────────────
# 【为什么是 --no-bundle】安装器由我们自己打（scripts/release/merged-installer.nsi）：
# 它要装两个位置、要独占模型库，Tauri 的打包器给不了这些。--no-bundle 只负责把 exe
# 与它依赖的 DLL 编出来。
if ($Target -eq 'All' -or $Target -eq 'App') {
    $blank = Join-Path ([IO.Path]::GetTempPath()) ("hfs-build-" + [Guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::WriteAllText($blank, '{"build":{"beforeBuildCommand":""}}')
    Push-Location backend
    try {
        $tauriArgs = @('tauri', 'build', '--ci', '--no-bundle', '--config', $blank)
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
if ($Target -eq 'All' -or $Target -eq 'Plugin') {
    $pluginArgs = @('build', '--manifest-path', 'backend\Cargo.toml', '-p', 'hifishifter-plugin')
    if ($Configuration -eq 'Release') { $pluginArgs += '--release' }
    cargo @pluginArgs
    Assert-Ok '插件引擎构建'

    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $module = Join-Path $bundle 'Contents\x86_64-win'
    $resources = Join-Path $bundle 'Contents\Resources'
    # 【为什么两个目录都要整个重建，而不是往里补】两条原因叠在一起：
    #   1. Vite 的 chunk 带内容哈希，Copy-Item 只写不删 —— 不清空就会把上一轮的 chunk
    #      再留一份，bundle 随每次构建单调增长（安装器在用户机器上做的正是同一件事）；
    #   2. Copy-Item 的语义坑：目标目录已存在时，它把**源目录本身**拷进去，于是第二次
    #      构建得到 assets\assets，第三次得到 assets\assets\assets —— 直接报"项已经存在"。
    # 先 Reset 再拷内容，两个问题一起消失，构建因此是幂等的。
    Reset-Directory $module
    Reset-Directory "$resources\frontend"
    Reset-Directory "$resources\models"
    New-Item -ItemType Directory -Force "$resources\frontend\assets" | Out-Null

    Copy-Item "$built\hifishifter_plugin.dll" "$module\HiFiShifterEngine.dll"
    # DLL 白名单刻意不含 vslib：它是独立 App 的闭源运行时，插件不分发。
    foreach ($dll in 'onnxruntime.dll', 'DirectML.dll', 'SoundTouchDLL.dll') {
        if (Test-Path "$built\$dll") { Copy-Item "$built\$dll" $module }
    }

    # 只拷插件入口与它引用的 chunk。index.html / detached.html / waveform-test.html
    # 属于独立 App 与开发页面，进了插件 bundle 既是死重量也让目录清单没法对账。
    Copy-Item frontend\dist\plugin.html "$resources\frontend"
    Copy-Item frontend\dist\assets\* "$resources\frontend\assets" -Recurse
    Copy-Item backend\src-tauri\icons\icon.ico "$resources\frontend"

    # 【为什么 bundle 里只有清单、没有权重】安装器把权重写进
    # %PROGRAMDATA%\HiFiShifter\models\<版本>，独立 App 与插件都从那一份读
    # （见 backend/hifishifter-kernel/src/model_store.rs）。bundle 里再留一份，
    # 用户就要为完全相同的 152 MB 付第二次磁盘。手动拷贝安装时请改用便携包。
    Copy-Item "$modelsSource\models.json" "$resources\models"

    # 原生 loader：宿主加载的是这个薄 DLL，它再按绝对路径拉起邻接的 Rust 引擎。
    $objDir = Join-Path ([IO.Path]::GetTempPath()) ("hfs-loader-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force $objDir | Out-Null
    cl.exe /nologo /utf-8 /std:c++17 /LD /MT /EHsc /O2 backend\hifishifter-plugin\native\module_loader.cpp "/Fo:$objDir\module_loader.obj" /link "/OUT:$module\HiFiShifter.vst3" "/IMPLIB:$objDir\HiFiShifterLoader.lib"
    Assert-Ok '原生 VST3 loader 构建'
    Remove-Item $objDir -Recurse -Force

    # 编译成功不等于能被宿主加载。打包前跑一次黑盒门禁，别让加载不了的产物进安装器。
    & .\scripts\vst3\windows\verify.ps1 -BundleDirectory $bundle
}

if ($Configuration -eq 'Debug') {
    Write-Host "debug 产物已就绪：$dist" -ForegroundColor Green
    return
}

if ($Target -eq 'All' -or $Target -eq 'Plugin' -or $Target -eq 'Clap') {
    $clapArgs = @('build', '--manifest-path', 'backend\Cargo.toml', '-p', 'hifishifter-clap')
    if ($Configuration -eq 'Release') { $clapArgs += '--release' }
    cargo @clapArgs
    Assert-Ok 'CLAP 模块构建'
    $clapModule = Join-Path $built 'hifishifter_clap.dll'
    if (!(Test-Path $clapModule)) { throw "缺少 $clapModule" }
    Copy-Item $clapModule (Join-Path $dist 'HiFiShifter.clap') -Force
    & .\scripts\clap\windows\verify.ps1 -ModulePath (Join-Path $dist 'HiFiShifter.clap')
}

# ── 安装器暂存 ──────────────────────────────────────────────────────────────
#
# 【为什么暂存目录是"可复用"的】-Target Installer 只重打安装器（改一句文案、换一个
# WebView2 bootstrapper）时不该重新编译 Rust。约定：dist\_stage\installer 里放
# app\ / vst3\ / models\ / LICENSE / license-page.txt / licenses\ / models-verify.nsh，全部由本脚本产出。

function Write-AppPayload {
    $app = Join-Path $stage 'app'
    Reset-Directory $app
    Copy-Item "$built\HiFiShifter.exe" $app
    # 0 字节的 DLL 是 arm64 上的占位符（vslib 是 x86_64 专用），不进产物。
    # hifishifter_plugin.dll 属于 VST3 载荷，装进 App 目录只会多一份 40 MB。
    Get-ChildItem $built -Filter *.dll |
        Where-Object { $_.Length -gt 0 -and $_.Name -ne 'hifishifter_plugin.dll' } |
        Copy-Item -Destination $app
    New-Item -ItemType Directory -Force "$app\models" | Out-Null
    Copy-Item "$modelsSource\models.json" "$app\models"
    Write-Host "App 载荷：$app" -ForegroundColor Green
}

function Write-Vst3Stage {
    $vst3 = Join-Path $stage 'vst3'
    Reset-Directory $vst3
    # 【为什么拷的是 bundle 的内容，而不是 HiFiShifter.vst3 这个目录】安装器的落点是
    # $HfsVst3Dir\${VST3_SUBDIR}，再用 File /r "${VST3_DIR}\*" 把暂存区的东西铺进去。
    # 若整个目录进暂存区，用户机器上就会多套一层
    # （…\VST3\HiFiShifter.vst3\HiFiShifter.vst3\Contents），而宿主只认
    # <名字>.vst3\Contents\<架构>\<名字>.vst3 —— 插件会安静地"装好了但扫不到"。
    Copy-Item "$dist\HiFiShifter.vst3\*" $vst3 -Recurse
    Write-Host "VST3 载荷：$vst3" -ForegroundColor Green
}

function Write-ClapStage {
    $clap = Join-Path $stage 'HiFiShifter.clap'
    Copy-Item (Join-Path $dist 'HiFiShifter.clap') $clap -Force
    Write-Host "CLAP 载荷：$clap" -ForegroundColor Green
}

function Write-ModelsStage {
    $models = Join-Path $stage 'models'
    Reset-Directory $models
    Copy-Item "$modelsSource\*" $models -Recurse
    Write-Host "模型载荷（$modelsVersion）：$models" -ForegroundColor Green
}

function Write-LicenseStage {
    $license = Join-Path $stage 'LICENSE'
    if (Test-Path $license) { Remove-Item $license -Force }
    Copy-Item LICENSE $license
    $dir = Join-Path $stage 'licenses'
    Reset-Directory $dir
    # 拷内容而不是目录：Reset-Directory 已经把 $dir 建好了，再拷 licenses 目录本身
    # 会变成 licenses\licenses\README.md —— 而 LICENSE 里写的路径是 licenses/README.md。
    Copy-Item 'licenses\*' $dir -Recurse

    # 【为什么许可页要单独一份带 BOM 的副本】NSIS 的 LicenseData 在**编译期**把文件读成
    # UTF-16 存进安装器：文件带 BOM 才按 UTF-8 解码，不带就按编译所在机器的 ANSI 代码页
    # 解码。LICENSE 是无 BOM 的 UTF-8，于是中文 Windows 上编出来的安装包会把 θ 显示成
    # "胃"、把日文假名整段丢掉，而编译不报错（只有打开许可页才看得见）。
    # 随包的 LICENSE 必须与仓库逐字节相同（用户和工具都会比对），所以这几 K 字节的副本
    # 只喂给许可页：读取用严格 UTF-8（LICENSE 若不是合法 UTF-8，这里当场失败），写出带 BOM。
    $page = Join-Path $stage 'license-page.txt'
    $strictUtf8 = New-Object Text.UTF8Encoding($false, $true)
    [IO.File]::WriteAllText($page, [IO.File]::ReadAllText((Join-Path $root 'LICENSE'), $strictUtf8),
        (New-Object Text.UTF8Encoding($true)))

    Write-Host "许可载荷：$stage" -ForegroundColor Green
}

# 安装器的完整性校验清单：逐文件列出相对路径与字节数。
#
# 【为什么要生成而不是在 NSIS 里解析 JSON】NSIS 没有 JSON 解析器，而"只看目录存在"
# 的判据与运行时 model_store::looks_complete 不一致 —— 那会造出"安装器说装好了、
# 运行时说不可用"，用户只看到"推理不可用"。这里把清单拍平成 NSIS 语句，
# 两边用的是同一份数据、同一个判据。
function Write-ModelsVerifyInclude {
    $path = Join-Path $stage 'models-verify.nsh'
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('; 本文件由 scripts/build/build.ps1 生成 —— 不要手工编辑。')
    $lines.Add('; 逐文件列出模型清单里的相对路径与字节数：安装器用它判断"库是否已完整"，')
    $lines.Add('; 判据与运行时 model_store::looks_complete 完全一致（文件存在 + 字节数相等）。')
    $lines.Add('Function HfsVerifyModelFiles')
    foreach ($file in $modelsManifest.files) {
        $relative = ($file.path -replace '/', '\')
        $lines.Add(('  StrCpy $R0 "{0}"' -f $relative))
        $lines.Add(('  StrCpy $R1 "{0}"' -f $file.bytes))
        $lines.Add('  Call HfsCheckModelFile')
        # $R2 = 0 表示这一项不符：立刻返回（后面的检查没有意义）。
        $lines.Add('  StrCmp $R2 0 0 +2')
        $lines.Add('  Return')
    }
    $lines.Add('FunctionEnd')
    [IO.File]::WriteAllText($path, ($lines -join "`r`n"), (New-Object Text.UTF8Encoding($true)))
}

function Get-WebView2Bootstrapper {
    if ($WebView2Bootstrapper) {
        if (!(Test-Path $WebView2Bootstrapper)) { throw "找不到 WebView2 bootstrapper：$WebView2Bootstrapper" }
        return (Resolve-Path $WebView2Bootstrapper).Path
    }
    $cache = Join-Path $dist '_cache'
    New-Item -ItemType Directory -Force $cache | Out-Null
    $target = Join-Path $cache 'MicrosoftEdgeWebview2Setup.exe'
    if (Test-Path $target) { return $target }
    Write-Host '下载 WebView2 bootstrapper（约 1.7 MB）...' -ForegroundColor Cyan
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' -OutFile $target -UseBasicParsing
    $size = (Get-Item $target).Length
    if ($size -lt 1MB) { throw "WebView2 bootstrapper 下载异常（只有 $size 字节）" }
    return $target
}

if ($Target -eq 'All' -or $Target -eq 'App') { Write-AppPayload }
if ($Target -eq 'All' -or $Target -eq 'Plugin') { Write-Vst3Stage }
if ($Target -eq 'All' -or $Target -eq 'Plugin' -or $Target -eq 'Clap') { Write-ClapStage }

if ($Target -ne 'Installer') {
    Write-ModelsStage
    Write-LicenseStage
    Write-ModelsVerifyInclude
}

if ($Target -eq 'Installer') {
    foreach ($required in 'app', 'vst3', 'HiFiShifter.clap', 'models', 'licenses', 'LICENSE', 'license-page.txt', 'models-verify.nsh') {
        if (!(Test-Path (Join-Path $stage $required))) {
            throw "dist\_stage\installer 缺少 $required：先跑 .\scripts\build\build.ps1 -Target All"
        }
    }
}

# ── 安装器 ──────────────────────────────────────────────────────────────────
$makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
if (!$makensis) {
    $candidate = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'NSIS\makensis.exe'
    if (Test-Path $candidate) { $makensis = $candidate }
}
if (!$makensis) {
    throw '找不到 makensis：先跑 scripts\setup\install_deps_windows.ps1（Windows 只交付这一个安装器，缺它等于没有产物）'
}

# 安装器文案由五语词表生成；先重新生成一次，漏翻译在这里就失败。
node (Join-Path $PSScriptRoot '..\release\build-installer-strings.mjs')
Assert-Ok '安装器文案生成'

$bootstrapper = Get-WebView2Bootstrapper
$setup = Join-Path $dist ('HiFiShifter_v' + $version + '_windows-' + $arch + '-setup.exe')
Remove-Item $setup -Force -ErrorAction SilentlyContinue

# 每一个路径都以 /D 传入，脚本里不出现任何相对路径猜测（值带引号：路径里可能有空格）。
& $makensis /INPUTCHARSET UTF8 `
    "/DAPP_DIR=$(Join-Path $stage 'app')" `
    "/DVST3_DIR=$(Join-Path $stage 'vst3')" `
    "/DCLAP_FILE=$(Join-Path $stage 'HiFiShifter.clap')" `
    "/DMODELS_DIR=$(Join-Path $stage 'models')" `
    "/DMODELS_VERSION=$modelsVersion" `
    "/DMODELS_VERIFY=$(Join-Path $stage 'models-verify.nsh')" `
    "/DHFS_LICENSE=$(Join-Path $stage 'LICENSE')" `
    "/DHFS_LICENSE_PAGE=$(Join-Path $stage 'license-page.txt')" `
    "/DHFS_LICENSES=$(Join-Path $stage 'licenses')" `
    "/DOUTPUT_FILE=$setup" `
    "/DAPP_VERSION=$version" `
    "/DWEBVIEW2_BOOTSTRAPPER=$bootstrapper" `
    (Join-Path $PSScriptRoot '..\release\merged-installer.nsi')
Assert-Ok '安装器编译'
Write-Host "安装器：$setup" -ForegroundColor Green

# ── 便携包 ──────────────────────────────────────────────────────────────────
# 便携包**自包含**：模型全量随包，解压即用，不依赖安装器与 ProgramData。
if ($Target -ne 'All') {
    return
}

$portable = Join-Path $dist 'HiFiShifter'
Reset-Directory $portable
Copy-Item "$built\HiFiShifter.exe" $portable
Get-ChildItem $built -Filter *.dll |
    Where-Object { $_.Length -gt 0 -and $_.Name -ne 'hifishifter_plugin.dll' } |
    Copy-Item -Destination $portable
Copy-Item $modelsSource "$portable\models" -Recurse
Copy-Item $licenses $portable -Recurse

$portableZip = Join-Path $dist "HiFiShifter-v$version-portable-win-$arch.zip"
Remove-Item $portableZip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $portable -DestinationPath $portableZip -CompressionLevel Optimal
Remove-Item $portable -Recurse -Force
Write-Host "便携包：$portableZip" -ForegroundColor Green
