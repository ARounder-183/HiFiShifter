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

Write-Host 'Product consistency checks' -ForegroundColor Cyan

# 版本号散在五个文件里，漏改一处不会编译失败，只会让某个界面报出别的版本。
Test-Invariant '产品版本在全部来源中一致' {
    $json = '(?m)("version"\s*:\s*")([^"]+)(")'
    $cargo = '(?ms)(\[package\].*?^\s*version\s*=\s*")([^"]+)(")'
    $sources = [ordered]@{
        'frontend/package.json'                      = @('frontend\package.json', $json)
        'backend/src-tauri/tauri.conf.json'          = @('backend\src-tauri\tauri.conf.json', $json)
        'backend/src-tauri/tauri.conf.dist-dev.json' = @('backend\src-tauri\tauri.conf.dist-dev.json', $json)
        'backend/src-tauri/Cargo.toml'               = @('backend\src-tauri\Cargo.toml', $cargo)
        'backend/hifishifter-plugin/Cargo.toml'      = @('backend\hifishifter-plugin\Cargo.toml', $cargo)
    }
    $found = [ordered]@{}
    foreach ($source in $sources.GetEnumerator()) {
        $match = [regex]::Match((Get-RepoText $source.Value[0]), $source.Value[1])
        if (!$match.Success) { throw "$($source.Key) 里找不到版本字段" }
        $found[$source.Key] = $match.Groups[2].Value
    }
    $distinct = @($found.Values | Select-Object -Unique)
    if ($distinct.Count -ne 1) {
        $report = ($found.GetEnumerator() | ForEach-Object { "$($_.Key) = $($_.Value)" }) -join ' | '
        throw "版本号不一致（跑 scripts/release/set-version.ps1）：$report"
    }
    "全部来源 = $($distinct[0])"
}

# models.json 的版本号由模型内容派生，运行时靠它判断共享模型库里的那一份能不能用。
Test-Invariant '模型清单与真实模型文件一致' {
    & (Join-Path $root 'scripts\build\write-models-manifest.ps1') -Check
    if (!$?) { throw 'models.json 过期（跑 scripts/build/write-models-manifest.ps1）' }
    'backend/src-tauri/resources/models/models.json'
}

Test-Invariant '安装器文案与五语词表一致' {
    & node (Join-Path $root 'scripts/release/build-installer-strings.mjs') --check
    if ($LASTEXITCODE) { throw 'installer_strings.nsh 与词表不一致（跑 node scripts/release/build-installer-strings.mjs）' }
    'scripts/release/installer/installer_strings.nsh'
}

# 这几条都是"手改回去就静默丢一个用户可见能力"的地方：没有卸载项、重复安装堆积资产、
# 模型又回到 bundle 里各存一份。
Test-Invariant '安装器注册卸载项、清理受管目录、模型走共享库' {
    $nsi = Get-RepoText 'scripts/release/vst3-installer.nsi'
    foreach ($required in 'WriteUninstaller', 'UninstallString', 'InstallLocation', 'HFS_MODELS_VERSION') {
        if ($nsi -notmatch [regex]::Escape($required)) { throw "vst3-installer.nsi 缺少 $required" }
    }
    if ($nsi -notmatch 'RMDir /r .*Resources\\frontend') {
        throw 'vst3-installer.nsi 不再清理 Contents\Resources\frontend —— 重复安装会重新堆积资产'
    }
    'WriteUninstaller / UninstallString / 目录对账 / 共享模型库'
}

# 前端 invoke.ts 的 switch 是前端能发出的全部方法名。其中任何一个若两边后端都不派发，
# 症状是"控件能点、点了没反应"——静默失效，且不会有编译错误。
Test-Invariant '前端 invoke 面都有后端命令承接' {
    $output = & python (Join-Path $root 'scripts/checks/check-command-gating.py') 2>&1
    if ($LASTEXITCODE) { throw ($output -join ' | ') }
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
Write-Host ''
if ($failed.Count) {
    Write-Host "$($failed.Count) check(s) failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host 'All product consistency checks passed.' -ForegroundColor Green
