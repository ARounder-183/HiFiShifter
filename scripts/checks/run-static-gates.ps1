# 静态门禁：不编译、不联网、几十秒跑完的那一组跨文件约定。
#
# 每一条被破坏时都不会报错，只在用户那里表现为"FFI 崩了宿主""日志里什么都没有"
# "控件点了没反应""版本显示不对"。它们必须在每次 PR 上都跑，而且必须只有一个入口 ——
# 此前这三条分别写在三条工作流里，其中一条漏掉过（见 vst3.yml 的历史注释）。
#
#   .\scripts\checks\run-static-gates.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Invoke-Gate([string]$name, [scriptblock]$body) {
    Write-Host "== $name" -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE) { throw "$name failed (exit $LASTEXITCODE)" }
}

# 每个被宿主调用的 extern "system" 入口都必须自带 panic 守卫
# （panic 越过 FFI 边界 = abort = DAW 崩溃）。纪律见 docs/thread-model.md。
Invoke-Gate 'FFI panic-guard gate' { python scripts/checks/check-ffi-guards.py }

# 限流日志只允许 Warn / Error。Info/Debug 套上限流会在默认日志级别下变成静默黑洞：
# 代码看起来有诊断，实际一条都不输出。
Invoke-Gate 'Log level gate' { python scripts/checks/check-log-levels.py }

# 四档日志详细程度：禁止遗留环境变量 / 生产 console.* / 过时的默认级别注释回归。
Invoke-Gate 'Log verbosity gate' { python scripts/checks/check-log-verbosity.py }

# 前端 invoke.ts 的 switch 是前端能发出的全部方法名。其中任何一个若两边后端都不派发，
# 症状是"控件能点、点了没反应" —— 静默失效，且不会有编译错误。
Invoke-Gate 'Command gating gate' { python scripts/checks/check-command-gating.py }

# 版本号、安装器文案、模型清单、许可清单、安装器形态：都是"少一处也不报错"的约定。
Invoke-Gate 'Product consistency gate' { & (Join-Path $PSScriptRoot 'check-product-consistency.ps1') }

# 点源的 env 脚本只允许改 $env:*。真实事故：msvc-env.ps1 的一个 `foreach ($root in ...)`
# 把构建脚本的 `$root`（仓库根）改写成了 Visual Studio 安装目录，构建在
# "找不到 backend\src-tauri\resources\models\models.json" 处失败。
Invoke-Gate 'Env script scope gate' { & (Join-Path $PSScriptRoot 'check-env-scripts.ps1') }

# 源码注释里的半截 UTF-8 字节不会让任何编译器报错，但会把中文注释变成乱码，
# 且此后每次 diff 都带着它。第三方树沿用上游原始编码，不在本门禁管辖内。
Invoke-Gate 'Source UTF-8 gate' { python scripts/checks/check-utf8.py }

# SoundTouch 构建生成器与缓存自愈。
Invoke-Gate 'SoundTouch generator gate' { python scripts/checks/check-soundtouch-generator.py }

# 跳过清单里的测试名仍须在 Rust 源码中可解析。
Invoke-Gate 'Test skip gate' { python scripts/checks/check-test-skips.py }

Write-Host 'All static gates passed.' -ForegroundColor Green
