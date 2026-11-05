# 产品一致性守卫：几条"必须同时成立"、又没有编译期保护的跨文件约定。
#
# 每一条被破坏时都不会报错，只会在用户那里表现为"版本显示不对""模型读了旧的"
# "控件点了没反应"。这里把它们搬到 CI。
#
#   .\scripts\checks\check-product-consistency.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$failed = [System.Collections.Generic.List[string]]::new()

function Test-Invariant($name, [scriptblock]$body) {
    try {
        $detail = & $body
        Write-Host "  [OK]   $name" -ForegroundColor Green
        if ($detail) { Write-Host "         $detail" -ForegroundColor DarkGray }
    }
    catch {
        Write-Host "  [FAIL] $name" -ForegroundColor Red
        Write-Host "         $($_.Exception.Message)" -ForegroundColor Red
        $failed.Add($name)
    }
}

function Get-RepoText($relative) {
    Get-Content -LiteralPath (Join-Path $root $relative) -Raw
}

# 统一的外部工具调用：工具没装时给一句可行动的话（而不是一条原生报错），退出码检查
# 也搬到这里 —— 此前每个调用点各写一遍 `if ($LASTEXITCODE)`，漏一处就是静默通过。
function Invoke-Tool([string]$tool, [string[]]$arguments) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if (!$command) {
        throw "找不到 $tool（见 scripts/setup/install_deps_windows.ps1 / install_deps_macos.sh / install_deps_linux.sh）"
    }
    $output = & $command.Source @arguments 2>&1
    if ($LASTEXITCODE) {
        throw "$tool 失败（退出码 $LASTEXITCODE）：$($output -join ' | ')"
    }
    $output
}

Write-Host 'Product consistency checks' -ForegroundColor Cyan

# 版本号散在多个文件里（清单见 scripts/release/version-sources.json），漏改一处不会编译
# 失败，只会让某个界面报出别的版本 —— 或者让插件在「程序和功能」里显示另一个版本号。
# 写入与校验共用 scripts/release/version.py，按清单里的 occurrences 逐处比对：
# package-lock 的第二处、Cargo.lock 的两条都算数。
Test-Invariant '产品版本在全部来源中一致' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/release/version.py'), '--check')
    ($output | Where-Object { $_ } | Select-Object -Last 1)
}

# models.json 的版本号由模型内容派生，运行时靠它判断共享模型库里的那一份能不能用。
# 权重不在干净 clone 里时，--check-manifest 会打印"检查跳过"并返回 0 —— 那是正常状态，
# 不是回归（清单与权重的对账只在取到权重之后才可能）。
Test-Invariant '模型清单与真实模型文件一致' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/setup/fetch-models.py'), '--check-manifest')
    ($output | Where-Object { $_ } | Select-Object -Last 1)
}

Test-Invariant '安装器文案与五语词表一致' {
    Invoke-Tool 'node' @((Join-Path $root 'scripts/release/build-installer-strings.mjs'), '--check') | Out-Null
    'scripts/release/installer/installer_strings.nsh'
}

# 许可全文是"少一份也不报错"的那类东西：打包脚本只拷目录，文件名只在 README 里出现。
# 改名或漏文件的结果是某个发行物静默少一份义务披露 —— 因此在这里钉死清单。
Test-Invariant '随包许可清单齐全（LICENSE + licenses\ 目录）' {
    $required = @(
        'LICENSE',
        'licenses\README.md',
        'licenses\COPYING.LGPL-2.1',
        'licenses\Apache-2.0.txt',
        'licenses\NSF-HiGAN-vocoder-NOTICE.txt'
    )
    foreach ($file in $required) {
        if (!(Test-Path -LiteralPath (Join-Path $root $file))) { throw "缺少随包许可文件：$file" }
    }
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    if ($nsi -notmatch 'HFS_LICENSES') { throw 'merged-installer.nsi 不再安装 licenses\ 目录' }
    $license = Get-RepoText 'LICENSE'
    if ($license -notmatch 'CLAP \(CLever Audio Plugin\) headers') { throw 'LICENSE 未登记 CLAP SDK 的第三方许可' }
    'LICENSE + licenses\（LGPL-2.1 / Apache-2.0 / CC BY-NC-SA 声明）'
}

# 这几条都是"手改回去就静默丢一个用户可见能力"的地方：没有卸载项、重复安装堆积资产、
# 模型又回到 bundle 里各存一份。
Test-Invariant '合并安装器注册卸载项、清理受管目录、模型走共享库' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    foreach ($required in 'WriteUninstaller', 'UninstallString', 'InstallLocation', 'MODELS_VERSION', 'APP_VERSION') {
        if ($nsi -notmatch [regex]::Escape($required)) { throw "merged-installer.nsi 缺少 $required" }
    }
    if ($nsi -notmatch 'RMDir /r .*Resources\\frontend') {
        throw 'merged-installer.nsi 不再清理 Contents\Resources\frontend —— 重复安装会重新堆积资产'
    }
    if ($nsi -notmatch 'RMDir /r "\$INSTDIR\\models"') {
        throw 'merged-installer.nsi 不再清理安装目录里的 models\ —— 从旧安装器升级会永久多占 150 MB'
    }
    'WriteUninstaller / UninstallString / 目录对账 / 共享模型库'
}

# 合并安装器的默认分支必须是"提权 + HKLM + Common Files"：HFS_TEST_MODE 那一段是给
# 测试模式跑闭环用的（装到当前用户、ARP 写 HKCU）。谁把测试分支写成默认，发布产物就会
# 装到用户目录、且在"程序和功能"里查不到。
Test-Invariant '合并安装器默认走 per-machine 安装' {
    $nsi = Get-RepoText 'scripts/release/merged-installer.nsi'
    $ifdef = $nsi.IndexOf('!ifdef HFS_TEST_MODE')
    $else = $nsi.IndexOf('!else', $ifdef)
    $endif = $nsi.IndexOf('!endif', $else)
    if ($ifdef -lt 0 -or $else -lt 0 -or $endif -lt 0) { throw 'merged-installer.nsi 的 HFS_TEST_MODE 分支不完整' }
    $release = $nsi.Substring($else, $endif - $else)
    foreach ($required in 'RequestExecutionLevel admin', '!define HFS_ARP_ROOT HKLM', 'COMMONFILES64') {
        if ($release -notmatch [regex]::Escape($required)) { throw "发布分支缺少 $required" }
    }
    'admin / HKLM / CommonFiles64'
}

# Windows 只交付一个安装器：构建脚本必须用 --no-bundle 自打安装器，且 bundle 里只带
# models.json（权重由安装器写进 ProgramData）。这两个条件任一破掉，用户就会为同一份
# 152 MB 权重付第二次磁盘 —— 而产物看起来完全正常。
Test-Invariant 'Windows 只有一个安装器且模型只装一份' {
    $build = Get-RepoText 'scripts/build/build.ps1'
    if ($build -notmatch '--no-bundle') { throw 'build.ps1 不再用 --no-bundle（Tauri 打包器会把模型塞回 App 目录）' }
    if ($build -notmatch 'merged-installer\.nsi') { throw 'build.ps1 不再调用 merged-installer.nsi' }
    if ($build -match 'Copy-Item\s+"\$modelsSource\\\$(model|\*)') {
        throw 'build.ps1 又把模型权重拷进 bundle 了'
    }
    'build.ps1 → merged-installer.nsi（--no-bundle）'
}

# 判据只允许存在于脚本里。工作流的 YAML 里出现 --skip / --test-threads / --offline，
# 就说明有人又在那里"就地打补丁" —— 而那正是跳过失控的起点（同一个测试在一条流水线
# 被跳过、在另一条没被跳过，而没人知道哪条算数）。
Test-Invariant '工作流里没有内联的测试判据' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root '.github\workflows') -Filter *.yml -File) {
        foreach ($line in Get-Content -LiteralPath $file.FullName) {
            # 注释里提到这些词是允许的（说明历史原因），判据只看代码。
            $code = ($line -split '#', 2)[0]
            if ($code -match '--skip|--test-threads|--offline') { $offenders += "$($file.Name): $($line.Trim())" }
        }
    }
    if ($offenders.Count) { throw ('判据必须写进 scripts/checks/：' + ($offenders -join ' | ')) }
    '.github/workflows'
}

# GitHub 的表达式只认单引号字符串。写成 `if: runner.os == "Windows"` 时**整个工作流
# 文件被判无效**：触发时只得到一条 0 秒、名字显示成文件路径的失败记录，任何 step 都没跑。
# 这个坑已经踩过三次（build.yml / vst3-unix.yml / clap.yml），所以判据是"if: 行里不许出现
# 双引号"—— 条件里本来也没有需要双引号的场合。
Test-Invariant '工作流条件用单引号字符串' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root '.github\workflows') -Filter *.yml -File) {
        foreach ($line in Get-Content -LiteralPath $file.FullName) {
            if ($line -notmatch '^\s*if:') { continue }
            if ($line.Contains('"')) { $offenders += "$($file.Name): $($line.Trim())" }
        }
    }
    if ($offenders.Count) { throw ('if: 里的字符串必须用单引号：' + ($offenders -join ' | ')) }
    '.github/workflows'
}


# 前端 invoke.ts 的 switch 是前端能发出的全部方法名。其中任何一个若两边后端都不派发，
# 症状是"控件能点、点了没反应"——静默失效，且不会有编译错误。
Test-Invariant '前端 invoke 面都有后端命令承接' {
    $output = Invoke-Tool 'python' @((Join-Path $root 'scripts/checks/check-command-gating.py'))
    ($output | Select-Object -First 1)
}

# Windows PowerShell 5.1 读无 BOM 的 UTF-8 文件时按 ANSI 解码：中文注释的最后一个字节会和
# 行尾换行配成双字节字符 —— 换行被吃掉，下一行代码变成注释的一部分。症状是"某个变量莫名
# 奇妙是 null"，报错指向再下一行。纯 ASCII 的脚本不需要 BOM，所以判据是"含非 ASCII 且无 BOM"。
Test-Invariant '含非 ASCII 的 PowerShell 脚本都带 UTF-8 BOM' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root 'scripts') -Recurse -File -Include *.ps1, *.psm1, *.psd1) {
        $bytes = [IO.File]::ReadAllBytes($file.FullName)
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { continue }
        $nonAscii = $false
        foreach ($byte in $bytes) { if ($byte -gt 0x7F) { $nonAscii = $true; break } }
        if ($nonAscii) { $offenders += $file.FullName.Substring($root.Length + 1) }
    }
    if ($offenders.Count) { throw ('以下脚本含非 ASCII 却没有 UTF-8 BOM：' + ($offenders -join ' | ')) }
    'scripts/'
}

# Python 在 Windows 上的 stdout 跟随 ANSI 代码页（英文 runner 是 cp1252、中文系统是 GBK）。
# 打印一句中文就会 UnicodeEncodeError 打死进程，真正的结论一条也进不了日志 —— 门禁红的是
# 编码，不是它要守的规则。这个坑已经踩过两次（限流日志门禁、版本号门禁），判据因此写成
# "源码里有非 ASCII 就必须把 stdout 钉成 UTF-8"，而不是逐个脚本去记。
Test-Invariant '含非 ASCII 的 Python 脚本把 stdout 钉成 UTF-8' {
    $offenders = @()
    foreach ($file in Get-ChildItem (Join-Path $root 'scripts') -Recurse -File -Include *.py) {
        $text = [IO.File]::ReadAllText($file.FullName, [Text.Encoding]::UTF8)
        if ($text -notmatch '[^\x00-\x7f]') { continue }
        if ($text -notmatch "reconfigure\(encoding='utf-8'") { $offenders += $file.FullName.Substring($root.Length + 1) }
    }
    if ($offenders.Count) { throw ('以下脚本含非 ASCII 却没有把 stdout 钉成 UTF-8：' + ($offenders -join ' | ')) }
    'scripts/**/*.py'
}

Write-Host ''
if ($failed.Count) {
    Write-Host "$($failed.Count) check(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host 'All product consistency checks passed.' -ForegroundColor Green
