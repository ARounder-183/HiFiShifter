# CLAP Windows 构建：cdylib -> HiFiShifter.clap -> 自检。
param([switch]$SkipVerify)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
. (Join-Path $root "scripts\env\msvc-env.ps1")
. (Join-Path $root "scripts\env\sdk-env.ps1"); Set-SdkEnvironment
# env 脚本是点源的（事故记录见 scripts/build/build.ps1）：再算一次，绝不让它们留下的
# 同名变量把仓库根换掉。
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
# 目标目录与 sibling 口径一致（scripts/build/build.ps1:147、clap/unix/build.sh:73）：
# 尊重 CARGO_TARGET_DIR。相对值按 cargo 的解析规则折算 —— 本脚本在 backend/ 里
# 跑 cargo，相对 CARGO_TARGET_DIR 就相对 backend 解析；写回绝对值，让下面取
# 产物路径与 cargo 的实际落盘位置永远一致。硬编码 backend\target 的旧写法在设置
# 了 CARGO_TARGET_DIR 的机器上要么拷贝失败，要么更糟：把 default-target 里
# **上一次构建的旧 DLL** 静默打包成"新"插件。
$backendDir = Join-Path $root "backend"
if ($env:CARGO_TARGET_DIR) {
    if ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
        $cargoTarget = [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
    }
    else {
        $cargoTarget = [IO.Path]::GetFullPath((Join-Path $backendDir $env:CARGO_TARGET_DIR))
    }
    $env:CARGO_TARGET_DIR = $cargoTarget
}
else {
    $cargoTarget = Join-Path $backendDir "target"
}
Push-Location $backendDir
try { cargo build -p hifishifter-clap --release --locked } finally { Pop-Location }
# PowerShell 不因原生命令的非零退出而抛错（$ErrorActionPreference 只管 cmdlet）；
# 不检查 $LASTEXITCODE 的话，编译失败时下面的 Copy-Item 会把**上一次构建的旧
# hifishifter_clap.dll** 静默打包成"新"插件（verify.ps1 对任何有效旧构建都放行）
# —— 与 build.ps1 的 Assert-Ok 纪律同一理由。
if ($LASTEXITCODE -ne 0) { throw "cargo build -p hifishifter-clap failed (exit $LASTEXITCODE)" }
$module = Join-Path $cargoTarget "release\hifishifter_clap.dll"
$outDir = Join-Path $root "dist\clap"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item $module (Join-Path $outDir "HiFiShifter.clap") -Force

# 单文件 CLAP 的前端与模型清单必须与模块同级（ResourceLayout::Sibling）。
# 只放 .clap 时宿主能扫描到插件，但 frontend_dir() 找不到 plugin.html ——
# 表现就是用户看到的空白窗口。CI 的 frontend-dist 产物在这里一并装好。
$resources = Join-Path $outDir "HiFiShifter-resources"
$frontend = Join-Path $resources "frontend"
$models = Join-Path $resources "models"
# 与 VST3 bundle 同一份公共资源：宿主插件与浮动编辑器共用 chunk，SDK 类型声明
# 不是运行时载荷。只拷 plugin.html 会让 detached.html 在插件窗口里 404。
New-Item -ItemType Directory -Force -Path $frontend, $models | Out-Null
Get-ChildItem -LiteralPath (Join-Path $root "frontend\dist") |
    Where-Object { $_.Name -ne 'sdk' } |
    Copy-Item -Destination $frontend -Recurse -Force
Copy-Item (Join-Path $root "backend\src-tauri\resources\models\models.json") $models -Force
# WebView2 的 profile 属于运行期状态，绝不跟着资源走（判据同 scripts/build/build.ps1）。
Remove-Item -LiteralPath (Join-Path $resources "webview") -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "CLAP resources: $resources" -ForegroundColor Green

if (!$SkipVerify) { & (Join-Path $PSScriptRoot "verify.ps1") -ModulePath (Join-Path $outDir "HiFiShifter.clap") }
