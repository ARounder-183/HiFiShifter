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
    [ValidateSet('All', 'App', 'Plugin', 'Clap', 'Installer')][string]$Target = 'All',
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

# 【为什么再算一次 $root】上面两份 env 脚本是**点源**的：它们的顶层变量会落在这个作用域里。
# 真实事故：msvc-env.ps1 里有一个 `foreach ($root in ...)`，把这里的 `$root`（仓库根）
# 覆盖成 Visual Studio 安装目录，于是 `Join-Path $root 'backend\...'` 指向
# `C:\Program Files\Microsoft Visual Studio\2022\Community\backend\...`，
# 构建在"找不到 models.json"处失败。env 脚本已改成函数（不再泄漏变量），这里再算一次是
# 第二道闸 —— 代价一行，收益是同类回归再也不会把产物写到仓库外。
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Assert-Ok($what) {
    if ($LASTEXITCODE) { throw "$what 失败（退出码 $LASTEXITCODE）" }
}

# 构建需要的外部工具：缺失时给一句能照做的错误，而不是让 PowerShell 抛
# "无法将 xxx 项识别为 cmdlet"（那对新克隆仓库的人没有任何指向性）。
foreach ($tool in 'cargo', 'npm') {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "找不到 $tool。先跑 scripts\setup\install_deps_windows.ps1 装齐工具链，或把它的目录加进 PATH。"
    }
}

# 【为什么不能直接调 `python`】Windows 上有三种常见形态：`python.exe` 在 PATH、只有
# `py` 启动器、以及 Microsoft Store 的 `python.exe` 别名（存在但一运行就弹应用商店）。
# 判据是"能真的跑起来并且是 3.x"，而不是"PATH 上有个叫 python 的文件"。
function Resolve-Python {
    $candidates = @(
        @{ Exe = 'python'; Prefix = @() },
        @{ Exe = 'py'; Prefix = @('-3') },
        @{ Exe = 'python3'; Prefix = @() }
    )
    foreach ($candidate in $candidates) {
        $found = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (!$found) { continue }
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $major = & $found.Source @($candidate.Prefix) -c 'import sys; print(sys.version_info[0])' 2>$null
        }
        catch { $major = $null }
        finally { $ErrorActionPreference = $previous }
        if ("$major".Trim() -eq '3') {
            return @{ Exe = $found.Source; Prefix = @($candidate.Prefix) }
        }
    }
    throw '找不到可用的 Python 3：装好后确保 python.exe（或 py 启动器）在 PATH 上，见 scripts\setup\install_deps_windows.ps1'
}

function Reset-Directory([string]$path) {
    Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $path | Out-Null
}

# 模型权重不在仓库里（见 scripts/setup/fetch-models.py）：缺了就取，已有则是一次
# 目录扫描。CI 用 actions/cache 以清单哈希为键缓存解压结果，因此这里通常什么都不做。
$python = Resolve-Python
& $python.Exe @($python.Prefix) (Join-Path $PSScriptRoot '..\setup\fetch-models.py')
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
    # 单文件 CLAP 的资源必须与模块同级（ResourceLayout::Sibling）：前端产物 +
    # 模型清单。少了任何一个，宿主能扫描到插件却只会得到一块空白窗口。
    $clapRes = Join-Path $dist 'HiFiShifter-resources'
    Reset-Directory (Join-Path $clapRes 'frontend')
    Reset-Directory (Join-Path $clapRes 'models')
    New-Item -ItemType Directory -Force (Join-Path $clapRes 'frontend\assets') | Out-Null
    Copy-Item 'frontend\dist\plugin.html' (Join-Path $clapRes 'frontend')
    Copy-Item 'frontend\dist\assets\*' (Join-Path $clapRes 'frontend\assets') -Recurse
    Copy-Item (Join-Path $modelsSource 'models.json') (Join-Path $clapRes 'models')
    # 【为什么 CLAP 资源目录里也要放 SoundTouchDLL.dll】三种格式现在编译同一份内核：
    # SoundTouch 由 `#[link]` 改成运行期 libloading，候选路径包含模块同级与同级的
    # `<stem>-resources`。VST3 把 DLL 放在引擎旁、App 放在 exe 旁；CLAP 是公共目录里的
    # 单文件模块，不能往那儿塞通用名 DLL，因此放进自己的私有资源目录。缺这份 DLL
    # 不算错误（is_available()=false → 已有的 Signalsmith 降级），但带上它才能与
    # VST3/App 行为一致 —— 用户的拉伸算法设置不该因插件格式而变。
    $soundtouch = Join-Path $built 'SoundTouchDLL.dll'
    if (Test-Path $soundtouch) { Copy-Item $soundtouch (Join-Path $clapRes 'SoundTouchDLL.dll') -Force }
    # 【为什么要显式删掉 webview】WebView2 的用户数据目录是**运行期状态**（缓存、
    # 锁文件、本机 profile），它曾经因为落在资源目录里而被打进安装包，也会让下一个
    # 用户拿到一份被占用的 profile。资源目录里只允许有 frontend、models 与
    # 运行期可选加载的 SoundTouchDLL.dll —— 绝不留下 WebView2 用户数据。
    Remove-Item -LiteralPath (Join-Path $clapRes 'webview') -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "CLAP 资源已组装：$clapRes" -ForegroundColor Green
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
    #
    # 【为什么按架构白名单拷，而不是 Copy-Item * -Recurse】dist\HiFiShifter.vst3 是
    # 跨平台共用的 bundle：Linux/macOS 构建往 Contents\x86_64-linux 或 Contents\MacOS
    # 写，Windows 构建往 Contents\x86_64-win 写，两端都共用同一个 Contents\Resources。
    # 直接 -Recurse 会把上一次别平台留下的整个架构目录（约 57 MB 的 .so/.dylib）一起
    # 拷进 Windows 安装器，宿主扫描时看到的是一堆无法加载的二进制。只取本平台需要的
    # 两个目录，别的架构无论谁留下的都进不了暂存区。
    $bundle = Join-Path $dist 'HiFiShifter.vst3'
    $contents = Join-Path $bundle 'Contents'
    $winModule = Join-Path $contents 'x86_64-win'
    $resources = Join-Path $contents 'Resources'
    foreach ($required in $winModule, $resources) {
        if (!(Test-Path -LiteralPath $required)) {
            throw "VST3 bundle 缺少 $required：先跑 .\scripts\build\build.ps1 -Target Plugin"
        }
    }
    New-Item -ItemType Directory -Force (Join-Path $vst3 'Contents') | Out-Null
    Copy-Item $winModule (Join-Path $vst3 'Contents\x86_64-win') -Recurse
    Copy-Item $resources (Join-Path $vst3 'Contents\Resources') -Recurse

    # 兜底：暂存区里只允许出现 x86_64-win 这一个架构目录。若将来新增别的本平台架构，
    # 要在这里显式加入白名单，而不是把 Copy-Item * 放回来。
    $foreign = @(Get-ChildItem (Join-Path $vst3 'Contents') -Directory |
        Where-Object { $_.Name -ne 'Resources' -and $_.Name -ne 'x86_64-win' })
    if ($foreign.Count) {
        throw "VST3 暂存区混入了非 Windows 架构目录：$($foreign.Name -join ', ')"
    }
    Write-Host "VST3 载荷：$vst3" -ForegroundColor Green
}

function Write-ClapStage {
    $clap = Join-Path $stage 'HiFiShifter.clap'
    Copy-Item (Join-Path $dist 'HiFiShifter.clap') $clap -Force
    Write-Host "CLAP 载荷：$clap" -ForegroundColor Green
    # 【为什么必须连资源目录一起暂存】CLAP 在 Windows 上是单文件模块，前端与
    # 模型清单按 ResourceLayout::Sibling 放在同级的 HiFiShifter-resources。
    # 只装 .clap 时 frontend_dir() 必然失败，宿主看到的是一块空白窗口。
    $resources = Join-Path $dist 'HiFiShifter-resources'
    if (!(Test-Path $resources)) { throw "缺少 CLAP 资源目录：$resources" }
    # 暂存前再挡一次：运行期状态绝不能进安装包（判据同 build.ps1 的组装段）。
    if (Test-Path (Join-Path $resources 'webview')) {
        throw "CLAP 资源目录里混进了 WebView2 运行期目录：$(Join-Path $resources 'webview')"
    }
    $staged = Join-Path $stage 'HiFiShifter-resources'
    Reset-Directory $staged
    Copy-Item "$resources\*" $staged -Recurse -Force
    if (!(Test-Path (Join-Path $staged 'frontend\plugin.html'))) {
        throw "CLAP 资源目录里缺少 frontend\plugin.html"
    }
    Write-Host "CLAP 资源：$staged" -ForegroundColor Green
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
    foreach ($required in 'app', 'vst3', 'HiFiShifter.clap', 'HiFiShifter-resources', 'models', 'licenses', 'LICENSE', 'license-page.txt', 'models-verify.nsh') {
        if (!(Test-Path (Join-Path $stage $required))) {
            throw "dist\_stage\installer 缺少 $required：先跑 .\scripts\build\build.ps1 -Target All"
        }
    }
}

# ── 安装器 ──────────────────────────────────────────────────────────────────
$makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
if (!$makensis) {
    # 【为什么不读 `ProgramFiles(x86)` 环境变量】它在部分宿主里根本不存在，而
    # `msvc-env.ps1` 导入 vcvars 环境时会把空值写回（.NET 把空值当"删除"），于是
    # 构建尾部会以一个和 NSIS 毫无关系的报错失败："无法将自变量绑定到参数 'Path'"。
    # 目录只能问系统 API（与 `scripts/env/msvc-env.ps1` 的 `Get-ProgramFilesRoots` 同源）。
    foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
        $base = [Environment]::GetFolderPath($kind)
        if (!$base) { continue }
        $candidate = Join-Path $base 'NSIS\makensis.exe'
        if (Test-Path -LiteralPath $candidate) { $makensis = $candidate; break }
    }
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
    "/DCLAP_RES_DIR=$(Join-Path $stage 'HiFiShifter-resources')" `
    "/DMODELS_DIR=$(Join-Path $stage 'models')" `
    "/DMODELS_VERSION=$modelsVersion" `
    "/DMODELS_VERIFY=$(Join-Path $stage 'models-verify.nsh')" `
    "/DHFS_LICENSE=$(Join-Path $stage 'LICENSE')" `
    "/DHFS_LICENSE_PAGE=$(Join-Path $stage 'license-page.txt')" `
    "/DHFS_LICENSES=$(Join-Path $stage 'licenses')" `
    "/DOUTPUT_FILE=$setup" `
    "/DAPP_VERSION=$version" `
    "/DWEBVIEW2_BOOTSTRAPPER=$bootstrapper" `
    "/DHFS_ICON=$(Join-Path $root 'backend\src-tauri\icons\icon.ico')" `
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
