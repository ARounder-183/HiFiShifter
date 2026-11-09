# 安装器闭环门禁：装 → 查 → 卸载 → 查。
#
# 【为什么必须在 CI 里真装一遍】安装器的失败方式几乎都不报错：模型少一个文件
# （运行时只是"推理不可用"）、bundle 里又混进一份权重（磁盘悄悄多 150 MB）、
# 卸载没删干净（用户重装后看到两套文件）。这些只有真的跑一遍再逐条断言才看得见。
#
# 【为什么装的是"测试模式"的安装器】发布分支要 RequestExecutionLevel admin 并写
# HKLM / Program Files，而 CI runner 上的提权会把脚本卡在 UAC 上。HFS_TEST_MODE 只把
# 安装位置与注册表根换成当前用户，**其余逻辑逐字相同**（页面、目录对账、模型段、
# 卸载段）。发布分支仍是 admin / HKLM / Common Files —— 那一条由
# check-product-consistency.ps1 静态钉住。
#
#   .\scripts\checks\check-installer.ps1
#   .\scripts\checks\check-installer.ps1 -ReleaseInstallerPath dist\HiFiShifter_v..._setup.exe
[CmdletBinding()]
param(
    # 发布安装器（体积断言用）。缺省取 dist\*-setup.exe 里最新的一个。
    [string]$ReleaseInstallerPath = '',
    # 安装器体积上限（MB）。旧的 App setup 169.8 MB + VST3 setup 166.7 MB 是两份
    # 各自带权重的产物；合并后权重只进安装器一次，230 MB 是"确实只有一份"的判据。
    [int]$MaxSizeMb = 230
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$stage = Join-Path $root 'dist\_stage\installer'
$failed = [System.Collections.Generic.List[string]]::new()

function Assert-That($name, [scriptblock]$body) {
    try {
        & $body
        Write-Host "  [OK]   $name" -ForegroundColor Green
    }
    catch {
        Write-Host "  [FAIL] $name" -ForegroundColor Red
        Write-Host "         $($_.Exception.Message)" -ForegroundColor Red
        $failed.Add($name)
    }
}

function Get-Makensis {
    $cmd = (Get-Command makensis -ErrorAction SilentlyContinue).Source
    if ($cmd) { return $cmd }
    # 【为什么不读 `ProgramFiles(x86)` 环境变量】见 build.ps1 同名处：它可能不存在，
    # 也可能被 vcvars 的导入删掉（空值 = 删除），报错却指向 `-Path` 参数。
    foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
        $base = [Environment]::GetFolderPath($kind)
        if (!$base) { continue }
        $candidate = Join-Path $base 'NSIS\makensis.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw '找不到 makensis：先跑 scripts\setup\install_deps_windows.ps1'
}

Write-Host 'Installer gate' -ForegroundColor Cyan

if (!$ReleaseInstallerPath) {
    $found = @(Get-ChildItem (Join-Path $root 'dist') -Filter '*-setup.exe' -ErrorAction SilentlyContinue)
    if ($found.Count -eq 0) { throw 'dist 下没有安装器：先跑 scripts\build\build.ps1 -Target All' }
    $ReleaseInstallerPath = ($found | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
$releaseInstaller = (Resolve-Path $ReleaseInstallerPath).Path
$sizeMb = [math]::Round((Get-Item $releaseInstaller).Length / 1MB, 1)
Assert-That "安装器体积小于 $MaxSizeMb MB（模型只装一份）" {
    if ($sizeMb -ge $MaxSizeMb) { throw "安装器 $sizeMb MB ≥ $MaxSizeMb MB" }
    "$sizeMb MB"
}

# ── 编译测试模式安装器（与发布安装器同一份脚本、同一份载荷）─────────────────
foreach ($required in 'app', 'vst3', 'HiFiShifter.clap', 'HiFiShifter-resources', 'models', 'licenses', 'LICENSE', 'license-page.txt', 'models-verify.nsh') {
    if (!(Test-Path (Join-Path $stage $required))) { throw "dist\_stage\installer 缺少 $required" }
}

# 【为什么钉死架构白名单】dist\HiFiShifter.vst3 是跨平台共用的 bundle，Linux/macOS
# 构建会往 Contents 里留下 x86_64-linux / MacOS 等目录。Windows 打包若用 Copy-Item *
# -Recurse，这些别平台的 .so/.dylib（约 57 MB）就会被 NSIS 一起打进安装器。
# 编译前先查暂存区：只要出现白名单以外的架构目录就失败，别等装到用户机器上才发现。
Assert-That 'VST3 暂存区只含 Windows 架构' {
    $contents = Join-Path $stage 'vst3\Contents'
    if (!(Test-Path -LiteralPath $contents)) { throw "缺少 $contents" }
    $allowed = @('Resources', 'x86_64-win')
    $foreign = @(Get-ChildItem -LiteralPath $contents -Directory |
        Where-Object { $allowed -notcontains $_.Name })
    if ($foreign.Count) { throw "暂存区混入非 Windows 架构目录：$($foreign.Name -join ', ')" }
    $allowed -join ', '
}
$manifest = Get-Content (Join-Path $root 'backend\src-tauri\resources\models\models.json') -Raw | ConvertFrom-Json
$bootstrapper = Join-Path $root 'dist\_cache\MicrosoftEdgeWebview2Setup.exe'
if (!(Test-Path $bootstrapper)) { throw "缺少 WebView2 bootstrapper：$bootstrapper（先跑 build.ps1）" }

$testInstaller = Join-Path $env:TEMP ('hfs-installer-gate-' + [Guid]::NewGuid().ToString('N') + '.exe')
$makensis = Get-Makensis

# ── 许可页的编码 ────────────────────────────────────────────────────────────
#
# 【为什么这条判据是"编译产物里的字节"】NSIS 的 LicenseData 在**编译期**把文件读成
# UTF-16 存进安装器：文件带 BOM 才按 UTF-8 解码，不带就按**编译所在机器的 ANSI 代码页**
# 解码。后者编译零告警、装完也能打开，只是许可页里 θ 变成"胃"、日文假名整段丢失 ——
# 只有真的把产物拆开看才拦得住，所以下面用一个不压缩的空安装器复现这条路径。
$licensePage = Join-Path $stage 'license-page.txt'

Assert-That '许可页用的是带 BOM 的副本，而不是随包的 LICENSE' {
    $bom = [IO.File]::ReadAllBytes($licensePage)
    if ($bom.Length -lt 3 -or $bom[0] -ne 0xEF -or $bom[1] -ne 0xBB -or $bom[2] -ne 0xBF) {
        throw 'license-page.txt 没有 UTF-8 BOM：NSIS 会按 ANSI 解码，非 ASCII 字符全部丢失'
    }
    $nsi = [IO.File]::ReadAllText((Join-Path $root 'scripts\release\merged-installer.nsi'),
        (New-Object Text.UTF8Encoding($false)))
    if (!$nsi.Contains('MUI_PAGE_LICENSE "${HFS_LICENSE_PAGE}"')) {
        throw 'merged-installer.nsi 的许可页不再使用 ${HFS_LICENSE_PAGE}'
    }
    'BOM + HFS_LICENSE_PAGE'
}

Assert-That '许可页把 LICENSE 原样嵌成 UTF-16（一个字符都不丢）' {
    $text = [IO.File]::ReadAllText($licensePage, (New-Object Text.UTF8Encoding($false)))
    $probe = Join-Path $env:TEMP ('hfs-license-probe-' + [Guid]::NewGuid().ToString('N'))
    $probeNsi = $probe + '.nsi'
    $probeExe = $probe + '.exe'
    # 探针：只有许可页、不压缩（压缩后的数据块没法按字节搜）、零载荷，几百毫秒。
    $probeScript = @(
        'Unicode true'
        'SetCompress off'
        '!include "MUI2.nsh"'
        'Name "hfs-license-probe"'
        ('OutFile "{0}"' -f $probeExe)
        'InstallDir "$TEMP\hfs-license-probe"'
        'RequestExecutionLevel user'
        ('!insertmacro MUI_PAGE_LICENSE "{0}"' -f $licensePage)
        '!insertmacro MUI_PAGE_INSTFILES'
        '!insertmacro MUI_LANGUAGE "English"'
        'Section'
        'SectionEnd'
    ) -join "`r`n"
    [IO.File]::WriteAllText($probeNsi, $probeScript, (New-Object Text.UTF8Encoding($true)))
    try {
        & $makensis /V1 $probeNsi | Out-Null
        if ($LASTEXITCODE) { throw "许可页探针编译失败（退出码 $LASTEXITCODE）" }
        $bytes = [IO.File]::ReadAllBytes($probeExe)
        # 文本块落在奇数偏移上也是合法的，两种对齐各解一遍。
        $oddLength = ($bytes.Length - 1) - (($bytes.Length - 1) % 2)
        $views = @(
            [Text.Encoding]::Unicode.GetString($bytes),
            [Text.Encoding]::Unicode.GetString($bytes, 1, $oddLength)
        )
        $checked = 0
        foreach ($line in ($text -split "`r?`n")) {
            $needle = $line.Trim()
            if ($needle -notmatch '[^\x00-\x7f]') { continue }
            $checked++
            $found = $false
            foreach ($view in $views) {
                if ($view.IndexOf($needle, [StringComparison]::Ordinal) -ge 0) { $found = $true; break }
            }
            if (!$found) { throw "安装器里丢了这一行（编译期按 ANSI 解码了）：$needle" }
        }
        "$checked 行含非 ASCII 的文本原样嵌入"
    }
    finally {
        Remove-Item $probeNsi, $probeExe -Force -ErrorAction SilentlyContinue
    }
}

# 【为什么用参数数组而不是续行符】PowerShell 的续行符是反引号，脚本里出现得越多，
# 就越容易被编辑器 / 格式化工具吃掉一个空格而变成"命令在行尾无声结束"。
$nsisArgs = @(
    '/INPUTCHARSET', 'UTF8', '/DHFS_TEST_MODE',
    "/DAPP_DIR=$(Join-Path $stage 'app')",
    "/DVST3_DIR=$(Join-Path $stage 'vst3')",
    "/DCLAP_FILE=$(Join-Path $stage 'HiFiShifter.clap')",
    "/DCLAP_RES_DIR=$(Join-Path $stage 'HiFiShifter-resources')",
    "/DMODELS_DIR=$(Join-Path $stage 'models')",
    "/DMODELS_VERSION=$($manifest.version)",
    "/DMODELS_VERIFY=$(Join-Path $stage 'models-verify.nsh')",
    "/DHFS_LICENSE=$(Join-Path $stage 'LICENSE')",
    "/DHFS_LICENSE_PAGE=$(Join-Path $stage 'license-page.txt')",
    "/DHFS_LICENSES=$(Join-Path $stage 'licenses')",
    "/DOUTPUT_FILE=$testInstaller",
    '/DAPP_VERSION=0.0.0-installer-gate',
    "/DWEBVIEW2_BOOTSTRAPPER=$bootstrapper",
    "/DHFS_ICON=$(Join-Path $root 'backend\src-tauri\icons\icon.ico')",
    (Join-Path $root 'scripts\release\merged-installer.nsi')
)
& $makensis @nsisArgs
if ($LASTEXITCODE) { throw "测试模式安装器编译失败（退出码 $LASTEXITCODE）" }

# 测试模式的安装根：InstallDir = %LOCALAPPDATA%\HiFiShifterInstallerTest\App
$testRoot = Join-Path $env:LOCALAPPDATA 'HiFiShifterInstallerTest'
$installDir = Join-Path $testRoot 'App'
$vst3Dir = Join-Path $testRoot 'VST3'
$clapDir = Join-Path $testRoot 'CLAP'
# 测试模式的模型根由 HFS_MODEL_ROOT 指定（user 权限不触发 UAC 虚拟化）；
# 发布模式仍是 %ProgramData%\HiFiShifter\models，由 admin 安装器写入。
$modelRoot = Join-Path $env:APPDATA 'HiFiShifterTest\models'
$modelDir = Join-Path $modelRoot $manifest.version
$arpKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter'

# ── 文件关联的保存 / 恢复 ───────────────────────────────────────────────
#
# 测试模式把 .hshp/.hsp 写到 HKCU\Software\Classes。开发机上可能已经有真实
# 关联，门禁不能把它们覆盖后不还原。这里用 reg.exe 对四个键做**整键快照**：
# 比逐值恢复更完整（OpenWithProgids、Content Type 等用户值都原样回来），也
# 不会把测试期间新建的空键留在系统里。
$classesRoot = 'HKCU:\Software\Classes'
$classesRegPath = 'HKCU\Software\Classes'
# 每次运行用唯一 ProgID，避免与开发机上同名测试键互相覆盖。
$foreignProgId = 'HfsInstallerGateForeign.' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$assocKeys = @('.hshp', '.hsp', 'HiFiShifter.Project', 'Applications\HiFiShifter.exe')
$assocBackupDir = Join-Path $env:TEMP ('hfs-assoc-backup-' + [Guid]::NewGuid().ToString('N'))
$assocBackup = @{}
New-Item -ItemType Directory -Force $assocBackupDir | Out-Null
foreach ($rel in $assocKeys) {
    $full = Join-Path $classesRoot $rel
    $file = Join-Path $assocBackupDir (($rel -replace '[\\/]', '_') + '.reg')
    $existed = Test-Path -LiteralPath $full
    if ($existed) {
        & reg.exe export "HKCU\Software\Classes\$rel" $file /y | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "无法备份注册表键 $full（reg export 退出码 $LASTEXITCODE）" }
    }
    $assocBackup[$rel] = [pscustomobject]@{ Existed = $existed; File = $file }
}

function Get-RegDefault([string]$path) {
    $key = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $key) { return $null }
    return $key.GetValue('')
}

function Get-RegValue([string]$path, [string]$name) {
    $key = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -eq $key) { return $null }
    return $key.GetValue($name)
}

function Restore-AssociationBackups {
    if (!$assocBackupDir -or !(Test-Path -LiteralPath $assocBackupDir)) { return }
    foreach ($rel in $assocKeys) {
        $full = Join-Path $classesRoot $rel
        Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction SilentlyContinue
        $entry = $assocBackup[$rel]
        if ($entry -and $entry.Existed -and (Test-Path -LiteralPath $entry.File)) {
            & reg.exe import $entry.File | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "恢复文件关联注册表键失败：$full"
            }
        }
    }
    Remove-Item -LiteralPath $assocBackupDir -Recurse -Force -ErrorAction SilentlyContinue
    $script:assocBackupDir = $null
}

# 默认卸载保留一切用户数据的测试标记：只创建两个以本次 run 唯一命名的文件，
# 绝不删用户的整个目录。一个放派生物目录（logs）、一个放用户数据目录（unsaved），
# 静默卸载默认不勾选任何清理选项，两者都应保留。"勾选才清理"的分支由独立的
# 卸载选项探针验证（un.onInit 默认值 + 选项页 leave 回调）。
$userProfileRoot = Join-Path $env:LOCALAPPDATA 'HiFiShifter'
$gateTag = 'hfs-uninstall-gate-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)
$disposableMarker = Join-Path $userProfileRoot "logs\$gateTag.txt"
$userDataMarker = Join-Path $userProfileRoot "unsaved\$gateTag.txt"
# 模型库标记：断言只验证"卸载器不删模型库里的东西"，不依赖安装阶段是否成功建库。
$modelMarker = Join-Path $modelDir "$gateTag.txt"

Remove-Item $testRoot -Recurse -Force -ErrorAction SilentlyContinue
# 测试模式专用的模型根：每次跑都从干净状态开始，避免上一轮的 .staging 让安装器
# 误判"已完整"而跳过写库。绝不碰发布模式的 %ProgramData%\HiFiShifter。
if (Test-Path (Join-Path $env:APPDATA 'HiFiShifterTest')) {
    Remove-Item (Join-Path $env:APPDATA 'HiFiShifterTest') -Recurse -Force -ErrorAction SilentlyContinue
}

try {
    Write-Host '  安装（静默）...' -ForegroundColor DarkGray
    $proc = Start-Process -FilePath $testInstaller -ArgumentList '/S' -Wait -PassThru
    if ($proc.ExitCode -ne 0) { throw "安装器退出码 $($proc.ExitCode)" }

    Assert-That 'ARP 注册了卸载项' {
        if (!(Test-Path $arpKey)) { throw "缺少注册表键 $arpKey" }
        (Get-ItemProperty $arpKey).DisplayVersion
    }

    Assert-That '文件关联指向安装的主程序' {
        $hshp = Get-RegDefault "$classesRoot\.hshp"
        if ($hshp -ne 'HiFiShifter.Project') { throw ".hshp 默认 ProgID = $hshp" }
        $command = Get-RegDefault "$classesRoot\HiFiShifter.Project\shell\open\command"
        if (!$command -or !$command.Contains("$installDir\HiFiShifter.exe")) {
            throw "打开命令未指向 $installDir：$command"
        }
        $hsp = Get-RegDefault "$classesRoot\.hsp"
        if ($hsp -ne 'HiFiShifter.Project') { throw ".hsp 默认 ProgID = $hsp" }
        if ($null -eq (Get-RegValue "$classesRoot\.hshp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hshp\OpenWithProgids 没有登记 HiFiShifter.Project'
        }
        if ($null -eq (Get-RegValue "$classesRoot\Applications\HiFiShifter.exe\SupportedTypes" '.hshp')) {
            throw 'Applications\HiFiShifter.exe\SupportedTypes 没有登记 .hshp'
        }
        'HiFiShifter.Project → HiFiShifter.exe'
    }

    Assert-That 'App 目录里没有模型权重（只有清单）' {
        $onnx = @(Get-ChildItem $installDir -Recurse -Filter *.onnx -ErrorAction SilentlyContinue)
        if ($onnx.Count -gt 0) { throw "安装目录里还有 $($onnx.Count) 个 .onnx 文件" }
        if (!(Test-Path (Join-Path $installDir 'models\models.json'))) { throw '缺少 models\models.json' }
        'no .onnx under INSTDIR'
    }

    # 【为什么连"插件模块不在 App 目录"也要断言】插件与独立 App 编译在同一个
    # `backend\target\<profile>` 目录里，打包脚本靠一条排除判据把它们分开。判据写错时
    # 安装包只是"大了 40~80 MB"，装完一切照常工作 —— 只有对着装出来的目录看才认得出来。
    # 所以判据必须落在安装后的真实落地点上，而不是只信 build.ps1 里那句注释。
    Assert-That 'App 目录里没有宿主插件模块' {
        foreach ($module in 'hifishifter_plugin.dll', 'hifishifter_clap.dll') {
            if (Test-Path (Join-Path $installDir $module)) {
                throw "App 安装目录混入了插件模块：$module"
            }
        }
        'no hifishifter_plugin.dll / hifishifter_clap.dll under INSTDIR'
    }

    Assert-That '许可全文与声明落到 LICENSE 里写的路径' {
        foreach ($rel in 'LICENSE', 'licenses\README.md', 'licenses\COPYING.LGPL-2.1', 'licenses\Apache-2.0.txt') {
            if (!(Test-Path (Join-Path $installDir $rel))) { throw "App 缺少 $rel" }
        }
        $vst3Resources = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents\Resources'
        foreach ($rel in 'LICENSE', 'licenses\README.md') {
            if (!(Test-Path (Join-Path $vst3Resources $rel))) { throw "VST3 缺少 $rel" }
        }
        if (Test-Path (Join-Path $installDir 'licenses\licenses')) { throw '许可目录多套了一层：licenses\licenses' }
        'App 与 VST3 各一份 LICENSE + licenses\'
    }

    Assert-That 'CLAP 组件装到自己的目录' {
        if (!(Test-Path (Join-Path $clapDir 'HiFiShifter.clap'))) { throw 'CLAP 模块没有安装到选定目录' }
        $clapFrontend = Join-Path $clapDir 'HiFiShifter-resources\frontend\plugin.html'
        if (!(Test-Path $clapFrontend)) { throw "CLAP 前端资源没有随模块落地：$clapFrontend" }
        $clapManifest = Join-Path $clapDir 'HiFiShifter-resources\models\models.json'
        if (!(Test-Path $clapManifest)) { throw "CLAP 模型清单没有随模块落地：$clapManifest" }
        'module + frontend + manifest'
    }
    # 安装器真的把别平台二进制装进来了也会在这里被抓到（编译前查暂存区、装完查落地）。
    Assert-That 'VST3 落地点只含 Windows 架构' {
        $installedContents = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents'
        $allowed = @('Resources', 'x86_64-win')
        $foreign = @(Get-ChildItem -LiteralPath $installedContents -Directory |
            Where-Object { $allowed -notcontains $_.Name })
        if ($foreign.Count) { throw "落地目录混入非 Windows 架构目录：$($foreign.Name -join ', ')" }
        $allowed -join ', '
    }
    Assert-That 'VST3 的 Resources\models 只有 models.json' {
        $models = Join-Path $vst3Dir 'HiFiShifter.vst3\Contents\Resources\models'
        if (!(Test-Path $models)) { throw "缺少 $models" }
        $entries = @(Get-ChildItem $models -Recurse -File)
        if ($entries.Count -ne 1) { throw "期望只有 1 个文件，实际 $($entries.Count) 个" }
        $entries[0].Name
    }

    Assert-That '共享模型库按清单齐全' {
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw "缺少 $modelDir\models.json" }
        foreach ($file in $manifest.files) {
            $path = Join-Path $modelDir ($file.path -replace '/', '\')
            if (!(Test-Path $path)) { throw "缺少 $($file.path)" }
            if ((Get-Item $path).Length -ne $file.bytes) { throw "$($file.path) 字节数不符" }
        }
        "$($manifest.files.Count) 个文件"
    }

    # 幂等：再装一次不该重写模型（判定在安装器的 HfsVerifyModels 里；这里验
    # "还能装成功、模型仍在"）。
    Assert-That '重装（幂等）' {
        $again = Start-Process -FilePath $testInstaller -ArgumentList '/S' -Wait -PassThru
        if ($again.ExitCode -ne 0) { throw "重装退出码 $($again.ExitCode)" }
        if (!(Test-Path (Join-Path $modelDir 'models.json'))) { throw '重装后模型库不见了' }
        'ok'
    }

    Write-Host '  卸载（静默）...' -ForegroundColor DarkGray
    # 把 .hsp 模拟成别的程序：卸载器只能清掉仍指向本安装的 .hshp，不能碰它。
    # 用 reg.exe 而不是 RegistryKey.SetValue：Get-Item 返回的键是只读句柄。
    & reg.exe add "$classesRegPath\$foreignProgId\shell\open\command" /ve /d '"C:\Other\Other.exe" "%1"' /f | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "无法创建外来 ProgID（reg add 退出码 $LASTEXITCODE）" }
    & reg.exe add "$classesRegPath\.hsp" /ve /d $foreignProgId /f | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "无法写入 .hsp 外来关联（reg add 退出码 $LASTEXITCODE）" }

    New-Item -ItemType Directory -Force (Split-Path $disposableMarker -Parent) | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $userDataMarker -Parent) | Out-Null
    Set-Content -LiteralPath $disposableMarker -Value 'disposable'
    Set-Content -LiteralPath $userDataMarker -Value 'keep'
    New-Item -ItemType Directory -Force (Split-Path $modelMarker -Parent) | Out-Null
    Set-Content -LiteralPath $modelMarker -Value 'keep'
    $uninstaller = Join-Path $installDir 'uninstall.exe'
    if (!(Test-Path $uninstaller)) { throw '缺少 uninstall.exe' }
    Start-Process -FilePath $uninstaller -ArgumentList '/S' -Wait | Out-Null
    # 卸载器把自己复制到临时目录再运行，退出码在 -Wait 下并不可靠；用目录与注册表判定。
    Start-Sleep -Seconds 3

    Assert-That '卸载后 App 目录消失' {
        if (Test-Path $installDir) { throw "目录仍存在：$installDir" }
        'gone'
    }
    Assert-That '卸载后 CLAP 文件消失' {
        if (Test-Path (Join-Path $clapDir 'HiFiShifter.clap')) { throw 'HiFiShifter.clap 仍存在' }
        if (Test-Path (Join-Path $clapDir 'HiFiShifter-resources')) { throw 'HiFiShifter-resources 仍存在' }
        'module + resources gone'
    }
    Assert-That '卸载后 VST3 目录消失' {
        if (Test-Path (Join-Path $vst3Dir 'HiFiShifter.vst3')) { throw '插件目录仍存在' }
        'gone'
    }
    Assert-That '卸载后 ARP 键消失' {
        if (Test-Path $arpKey) { throw "注册表键仍存在：$arpKey" }
        'gone'
    }
    Assert-That '卸载只解除自己的关联，保留别人的默认程序' {
        $hshp = Get-RegDefault "$classesRoot\.hshp"
        if ($hshp -eq 'HiFiShifter.Project') { throw '.hshp 仍绑定 HiFiShifter.Project' }
        if (Test-Path -LiteralPath "$classesRoot\HiFiShifter.Project") {
            throw 'HiFiShifter.Project ProgID 仍存在'
        }
        if (Test-Path -LiteralPath "$classesRoot\Applications\HiFiShifter.exe") {
            throw 'Applications\HiFiShifter.exe 仍存在'
        }
        $hsp = Get-RegDefault "$classesRoot\.hsp"
        if ($hsp -ne $foreignProgId) { throw ".hsp 被误改：$hsp" }
        if (!(Test-Path -LiteralPath "$classesRoot\$foreignProgId")) {
            throw "$foreignProgId 被误删"
        }
        if ($null -ne (Get-RegValue "$classesRoot\.hshp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hshp\OpenWithProgids 仍残留 HiFiShifter.Project'
        }
        if ($null -ne (Get-RegValue "$classesRoot\.hsp\OpenWithProgids" 'HiFiShifter.Project')) {
            throw '.hsp\OpenWithProgids 仍残留 HiFiShifter.Project'
        }
        'own removed, foreign kept'
    }
    # 【为什么默认断言是"保留"】卸载选项默认都不勾选：卸载只是移除程序，
    # 不该抹掉渲染缓存（重装排障时要复用）与日志（定位上次故障的唯一线索）。
    # 模型库同理，保留后重装同版本可直接复用，装新版由 HfsModelsGc 自己 GC。
    Assert-That '卸载默认保留共享模型库（勾选才清理）' {
        if (!(Test-Path -LiteralPath $modelMarker)) { throw "模型库标记被误删：$modelMarker" }
        'kept'
    }
    Assert-That '卸载默认保留缓存与日志' {
        if (!(Test-Path -LiteralPath $disposableMarker)) { throw "派生物标记被误删：$disposableMarker" }
        if (!(Test-Path -LiteralPath $userDataMarker)) { throw "用户数据标记被误删：$userDataMarker" }
        'kept'
    }
}
finally {
    Remove-Item -LiteralPath "$classesRoot\$foreignProgId" -Recurse -Force -ErrorAction SilentlyContinue
    Restore-AssociationBackups
    Remove-Item $testRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $testInstaller -Force -ErrorAction SilentlyContinue
    Remove-Item $disposableMarker, $userDataMarker -Force -ErrorAction SilentlyContinue
    # 测试模式模型根是门禁专用的，整目录清掉，不留 .staging。
    Remove-Item (Join-Path $env:APPDATA 'HiFiShifterTest') -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
if ($failed.Count) {
    Write-Host "$($failed.Count) installer check(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host 'Installer gate passed.' -ForegroundColor Green
