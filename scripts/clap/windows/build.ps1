# CLAP Windows 构建：cdylib -> HiFiShifter.clap -> 自检。
param([switch]$SkipVerify)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
. (Join-Path $root "scripts\env\msvc-env.ps1")
. (Join-Path $root "scripts\env\sdk-env.ps1"); Set-SdkEnvironment
# env 脚本是点源的（事故记录见 scripts/build/build.ps1）：再算一次，绝不让它们留下的
# 同名变量把仓库根换掉。
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
Push-Location (Join-Path $root "backend")
try { cargo build -p hifishifter-clap --release --locked } finally { Pop-Location }
$module = Join-Path $root "backend\target\release\hifishifter_clap.dll"
$outDir = Join-Path $root "dist\clap"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item $module (Join-Path $outDir "HiFiShifter.clap") -Force

# 单文件 CLAP 的前端与模型清单必须与模块同级（ResourceLayout::Sibling）。
# 只放 .clap 时宿主能扫描到插件，但 frontend_dir() 找不到 plugin.html ——
# 表现就是用户看到的空白窗口。CI 的 frontend-dist 产物在这里一并装好。
$resources = Join-Path $outDir "HiFiShifter-resources"
$frontend = Join-Path $resources "frontend"
$models = Join-Path $resources "models"
New-Item -ItemType Directory -Force -Path $frontend, (Join-Path $frontend "assets"), $models | Out-Null
Copy-Item (Join-Path $root "frontend\dist\plugin.html") $frontend -Force
Copy-Item (Join-Path $root "frontend\dist\assets\*") (Join-Path $frontend "assets") -Recurse -Force
Copy-Item (Join-Path $root "backend\src-tauri\resources\models\models.json") $models -Force
# WebView2 的 profile 属于运行期状态，绝不跟着资源走（判据同 scripts/build/build.ps1）。
Remove-Item -LiteralPath (Join-Path $resources "webview") -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "CLAP resources: $resources" -ForegroundColor Green

if (!$SkipVerify) { & (Join-Path $PSScriptRoot "verify.ps1") -ModulePath (Join-Path $outDir "HiFiShifter.clap") }
