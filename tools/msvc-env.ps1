# tools/msvc-env.ps1
#
# 主要内容：把 MSVC（Visual Studio C++ 工具链）的环境变量导入当前 PowerShell 会话。
#
# 作用：HiFiShifter 的 Rust 构建会通过 cc / cmake 两个 crate 直接调用 cl.exe 编译
#       C/C++ 源码（third_party/world-static、soundtouch、opus 等）。cc crate 不会自己
#       初始化 MSVC 环境：它只在 PATH 上找 cl.exe。若当前会话没有 INCLUDE / LIB / PATH
#       等变量，cl.exe 会在启动编译器后端时报
#           cl : Command line error D8050 : cannot execute '...\c1xx.dll'
#       表现为"cl.exe 存在但编译全部失败"。
#
# 用法（在 PowerShell 中，先从仓库根目录执行）：
#       . .\tools\msvc-env.ps1
#       cd backend\src-tauri; cargo test
#
# 特殊说明：
#   - 必须用点源（`. .\tools\msvc-env.ps1`）而不是 `.\tools\msvc-env.ps1`，
#     否则环境变量只存在于子进程里，对后续 cargo 无效。
#   - 若执行策略禁止未签名脚本（报 "无法加载文件 ... 未对文件进行数字签名"），
#     用子进程绕过，或在允许的会话里先 `Set-ExecutionPolicy -Scope Process Bypass`。
#   - 本脚本刻意**不使用反引号续行**：以 `Invoke-Expression` 内联执行时，
#     反引号会被折叠成字面量，导致 "-requires" 被当成独立语句报错。
#   - 找不到 vswhere 或 vcvars64.bat 时会明确报错退出，不会静默降级
#     （静默降级正是 D8050 难以定位的原因）。

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at $vswhere -- install Visual Studio with the C++ workload"
}

# 只要装了「使用 C++ 的桌面开发」工作负载的实例。
$vsArgs = @(
    '-latest', '-products', '*',
    '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    '-property', 'installationPath'
)
$vsPath = & $vswhere @vsArgs
if (-not $vsPath) {
    throw "No Visual Studio instance with VC.Tools.x86.x64 -- install the 'Desktop development with C++' workload"
}
$vsPath = $vsPath.Trim()

$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

# 技巧：让 cmd 先跑 vcvars64.bat，然后 `set` 转储整个环境；
# 用 NUL 分隔（-s）便于逐条解析，避免值里含 '=' 时被截断。
$dump = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set" 2>$null

foreach ($line in $dump) {
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { continue }
    $name = $line.Substring(0, $idx)
    $value = $line.Substring($idx + 1)
    Set-Item -Force -Path "Env:$name" -Value $value
}

# 校验：这几个变量是 cl.exe 能正常工作的最小充分集合。
$required = @('INCLUDE', 'LIB', 'VCINSTALLDIR', 'WindowsSdkDir', 'VCToolsVersion')
$missing = $required | Where-Object { -not (Test-Path "Env:$_") }
if ($missing) {
    throw "MSVC env incomplete after vcvars64: missing $($missing -join ', ')"
}

Write-Host "[msvc-env] VS      : $vsPath"
Write-Host "[msvc-env] VCTools : $env:VCToolsVersion"
Write-Host "[msvc-env] SDK     : $env:WindowsSDKVersion"
Write-Host "[msvc-env] cl.exe  : $((Get-Command cl.exe -ErrorAction SilentlyContinue).Source)"
