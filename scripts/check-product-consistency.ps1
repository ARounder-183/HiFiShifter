# 产品一致性守卫：把"必须同时成立"的几条不变量钉在一次检查里。
#
# 【为什么需要它】本仓库有若干**跨文件**约定，它们各自都没有编译期保护：
# 版本号分散在四个文件、安装器文案由词表生成、模型清单由脚本生成、交付链路不该依赖
# 一次性探针目录……任何一条被破坏都不会报错，只会在某个用户那里表现为"版本显示不对"
# "设置丢了""多占了 150 MB"。这些恰好是最难从现象反推原因的一类缺陷。
#
# 用法：pwsh -File scripts/check-product-consistency.ps1
# 退出码非 0 表示有检查未通过；每条检查都会打印它验证的是什么。
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$checkRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checkFailures = [System.Collections.Generic.List[string]]::new()

function Test-Consistency {
    param([string]$Name, [scriptblock]$Body)
    try {
        $detail = & $Body
        Write-Host "  [OK]   $Name" -ForegroundColor Green
        if ($detail) { Write-Host "         $detail" -ForegroundColor DarkGray }
    } catch {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        Write-Host "         $($_.Exception.Message)" -ForegroundColor Red
        $checkFailures.Add($Name)
    }
}

function Get-RepoText {
    param([string]$Relative)
    Get-Content -LiteralPath (Join-Path $checkRoot $Relative) -Raw
}

Write-Host "Product consistency checks" -ForegroundColor Cyan

# ── 1. 交付链路不依赖一次性探针目录 ────────────────────────────────────
#
# `probe/` 是一次性验证工程，可以整体删除。产品交付链（脚本 / CI / 前端）不得引用
# 它。插件 SDK 缓存也已移出 probe/，改由 `tools/sdk-env.ps1` 统一定义（third_party/sdk），
# 因此这里不再需要任何例外。
Test-Consistency '交付链路不引用一次性探针产物' {
    $scanRoots = @('tools', 'frontend/src', 'backend/build-support', '.github/workflows')
    $offenders = @()
    foreach ($scanRoot in $scanRoots) {
        $full = Join-Path $checkRoot $scanRoot
        if (!(Test-Path -LiteralPath $full)) { continue }
        foreach ($file in Get-ChildItem -LiteralPath $full -Recurse -File |
            Where-Object { $_.Extension -in @('.ps1', '.nsi', '.nsh', '.mjs', '.js', '.ts', '.tsx', '.yml', '.json') }) {
            $matches = Select-String -LiteralPath $file.FullName -Pattern 'probe[/\\]ara' -AllMatches -ErrorAction SilentlyContinue
            foreach ($match in $matches) {
                # 注释里解释"这段代码原先借用过探针脚本"是正当的，不是依赖。
                if ($match.Line.Trim() -match '^(#|//|;|\*|--)') { continue }
                $offenders += "$($file.FullName.Substring($checkRoot.Length + 1)): $($match.Line.Trim())"
            }
        }
    }
    # Rust 源码同样检查（含测试）：夹具已经搬进 crate 自己的 tests/fixtures。
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $checkRoot 'backend') -Recurse -File -Filter '*.rs' |
        Where-Object { $_.FullName -notlike '*\target\*' -and $_.FullName -notlike '*\probe\*' }) {
        $matches = Select-String -LiteralPath $file.FullName -Pattern 'probe[/\\]ara' -AllMatches -ErrorAction SilentlyContinue
        foreach ($match in $matches) {
            if ($match.Line.Trim() -match '^(//|\*|/\*)') { continue }
            $offenders += "$($file.FullName.Substring($checkRoot.Length + 1)): $($match.Line.Trim())"
        }
    }
    if ($offenders.Count) { throw ("以下位置仍引用 probe/ara 的探针产物：`n         " + ($offenders -join "`n         ")) }
    '扫描了 tools/ frontend/ backend/ .github/'
}

# ── 2. 版本号四处一致 ──────────────────────────────────────────────────
Test-Consistency '产品版本在全部来源中一致' {
    function Get-FirstVersion([string]$Text) {
        foreach ($line in $Text -split "`n") {
            $trimmed = $line.Trim().TrimStart('"')
            if (!$trimmed.StartsWith('version')) { continue }
            $rest = $trimmed.Substring(7).Trim().TrimStart('"').Trim()
            if (!$rest.StartsWith('=') -and !$rest.StartsWith(':')) { continue }
            $value = $rest.Substring(1).Trim().TrimEnd(',').Trim().Trim('"')
            if ($value) { return $value }
        }
        return $null
    }
    $sources = [ordered]@{
        'frontend/package.json'                 = Get-FirstVersion (Get-RepoText 'frontend/package.json')
        'backend/src-tauri/tauri.conf.json'     = Get-FirstVersion (Get-RepoText 'backend/src-tauri/tauri.conf.json')
        'backend/src-tauri/Cargo.toml'          = Get-FirstVersion (Get-RepoText 'backend/src-tauri/Cargo.toml')
        'backend/hifishifter-plugin/Cargo.toml' = Get-FirstVersion (Get-RepoText 'backend/hifishifter-plugin/Cargo.toml')
    }
    $distinct = @($sources.Values | Select-Object -Unique)
    if ($distinct.Count -ne 1) {
        $report = ($sources.GetEnumerator() | ForEach-Object { "$($_.Key) = $($_.Value)" }) -join "`n         "
        throw "版本号不一致（跑 scripts/set-version.ps1）：`n         $report"
    }
    "全部来源 = $($distinct[0])"
}

# ── 3. 含非 ASCII 的 PowerShell 脚本必须带 UTF-8 BOM ───────────────────
#
# 【为什么这条最要紧】Windows PowerShell 5.1 读无 BOM 的文件时按 ANSI（中文系统上是
# GBK）解码。UTF-8 的中文注释被当成 GBK 之后，最后一个字节可能与行尾的换行符配成一个
# 双字节字符 —— **换行被吃掉，下一行代码变成注释的一部分**。症状是"某个变量莫名其妙
# 是 null"，报错指向的是再下一行，与真正的原因隔着两层。
#
# 这个坑本仓库已经踩过两次（一次是 `ff432086`，一次是给 `tools/package-vst3.ps1` 加中文
# 注释却忘了补 BOM）。纯 ASCII 的脚本不需要 BOM，因此判据是"含非 ASCII 且没有 BOM"。
Test-Consistency '含非 ASCII 的 PowerShell 脚本都带 UTF-8 BOM' {
    $offenders = @()
    foreach ($scanRoot in @('tools', 'scripts', 'probe')) {
        $full = Join-Path $checkRoot $scanRoot
        if (!(Test-Path -LiteralPath $full)) { continue }
        foreach ($file in Get-ChildItem -LiteralPath $full -Recurse -File |
            Where-Object { $_.Extension -in @('.ps1', '.psm1', '.psd1') }) {
            $bytes = [IO.File]::ReadAllBytes($file.FullName)
            if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { continue }
            # 无 BOM：只要有一个字节 > 0x7F 就会被 PowerShell 5.1 误读。
            $hasNonAscii = $false
            foreach ($byte in $bytes) { if ($byte -gt 0x7F) { $hasNonAscii = $true; break } }
            if ($hasNonAscii) { $offenders += $file.FullName.Substring($checkRoot.Length + 1) }
        }
    }
    if ($offenders.Count) {
        throw ("以下脚本含非 ASCII 却没有 UTF-8 BOM，PowerShell 5.1 会吃掉换行把下一行变成注释：`n         " +
            ($offenders -join "`n         "))
    }
    'tools/ scripts/ probe/'
}

# ── 4. 安装器文案与词表一致 ────────────────────────────────────────────
Test-Consistency '安装器文案与五语词表一致' {
    & node (Join-Path $checkRoot 'tools/build-installer-strings.mjs') --check
    if ($LASTEXITCODE -ne 0) { throw 'installer_strings.nsh 与词表不一致（跑 node tools/build-installer-strings.mjs）' }
    'tools/installer/installer_strings.nsh'
}

# ── 4. 模型清单与真实模型文件一致 ──────────────────────────────────────
Test-Consistency '模型清单与真实模型文件一致' {
    # 不 spawn `pwsh`：开发机上常常只有 Windows PowerShell 5.1（CI 有 pwsh）。
    # 同一会话直接调用那份脚本，它自带 -Check 模式、只读不写。
    & (Join-Path $checkRoot 'tools/write-models-manifest.ps1') -Check
    if (!$?) { throw 'models.json 过期（跑 tools/write-models-manifest.ps1）' }
    'backend/src-tauri/resources/models/models.json'
}

# ── 5. 安装器的三条不变量 ──────────────────────────────────────────────
#
# 这三条都是"有人手改回去就会静默丢失一个用户可见能力"的地方：没有卸载项、重复安装
# 堆积资产、模型又回到 bundle 里各存一份。
Test-Consistency '安装器注册卸载项、清理受管目录、模型走共享库' {
    $nsi = Get-RepoText 'tools/vst3-installer.nsi'
    foreach ($required in @('WriteUninstaller', 'UninstallString', 'InstallLocation')) {
        if ($nsi -notmatch [regex]::Escape($required)) { throw "vst3-installer.nsi 缺少 $required" }
    }
    # 目录对账：写入前必须清掉受管目录，否则 Vite 的内容哈希文件名会让目录单调增长。
    if ($nsi -notmatch 'RMDir /r "\$INSTDIR\\\$\{PLUGIN_SUBDIR\}\\Contents\\Resources\\frontend"') {
        throw 'vst3-installer.nsi 不再清理 Contents\Resources\frontend —— 重复安装会重新堆积资产'
    }
    if ($nsi -notmatch 'HFS_MODELS_VERSION') { throw 'vst3-installer.nsi 不再安装共享模型库' }
    'WriteUninstaller / UninstallString / 目录对账 / 共享模型库'
}

# ── 6. 安装器的发布分支没被测试开关顶掉 ────────────────────────────────
Test-Consistency '安装器默认分支仍是 admin + HKLM + Common Files' {
    $nsi = Get-RepoText 'tools/vst3-installer.nsi'
    foreach ($required in @('RequestExecutionLevel admin', 'InstallDir "$COMMONFILES64\VST3"', '!define HFS_ARP_ROOT HKLM')) {
        if ($nsi -notmatch [regex]::Escape($required)) { throw "vst3-installer.nsi 的发布分支缺少：$required" }
    }
    # 测试开关只能出现在 !ifdef HFS_TEST_MODE 分支里。
    if ($nsi -match 'RequestExecutionLevel user' -and $nsi -notmatch '(?s)!ifdef HFS_TEST_MODE.*RequestExecutionLevel user.*!else') {
        throw 'RequestExecutionLevel user 出现在测试分支之外'
    }
    'HFS_TEST_MODE 只影响测试分支'
}

# ── 7. 插件 bundle 只带插件入口，不塞整套 dist ─────────────────────────
Test-Consistency '插件 bundle 只拷 plugin.html 与 assets' {
    $bundle = Get-RepoText 'tools/build-plugin-bundle.ps1'
    if ($bundle -match "Get-ChildItem -LiteralPath 'frontend\\dist'") {
        throw 'build-plugin-bundle.ps1 又把整套 frontend/dist 拷进插件了（index.html / detached.html / waveform-test.html 都是死重量）'
    }
    foreach ($required in @('plugin.html', 'assets')) {
        if ($bundle -notmatch [regex]::Escape($required)) { throw "build-plugin-bundle.ps1 不再拷贝 $required" }
    }
    'plugin.html + assets'
}

# ── 8. 两个 workflow 的公共策略一致 ───────────────────────────────────
Test-Consistency '两个 workflow 都声明 concurrency 与 permissions' {
    foreach ($workflow in @('.github/workflows/build.yml', '.github/workflows/vst3.yml')) {
        $text = Get-RepoText $workflow
        foreach ($required in @('concurrency:', 'permissions:')) {
            if ($text -notmatch [regex]::Escape($required)) { throw "$workflow 缺少 $required" }
        }
    }
    'build.yml / vst3.yml'
}

Write-Host ""
if ($checkFailures.Count) {
    Write-Host "$($checkFailures.Count) check(s) failed:" -ForegroundColor Red
    $checkFailures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "All product consistency checks passed." -ForegroundColor Green
