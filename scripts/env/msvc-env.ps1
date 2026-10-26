# scripts/env/msvc-env.ps1
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
#       . .\scripts\env\msvc-env.ps1
#       cd backend\src-tauri; cargo test
#
# 插件 / 内核的测试还要再点源一次 SDK 环境（`ara2-bridge` 的 build.rs 与
# `audio_abi::layout_tests` 的 ABI oracle 都要求 `ARA_SDK_DIR` / `ARA_VST3_SDK_DIR`）：
#       . .\scripts\env\sdk-env.ps1; Set-SdkEnvironment
#       cargo test --manifest-path backend\Cargo.toml -p hifishifter-plugin --lib
# 少点这一句时，oracle 会以 `ORACLE_ENV_HELP` 的文案失败并直接列出上面两条命令。
#
# 特殊说明：
#   - 必须用点源（`. .\scripts\env\msvc-env.ps1`）而不是 `.\scripts\env\msvc-env.ps1`，
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
if ($vsPath) { $vsPath = $vsPath.Trim() }

# 【为什么还要兜底】vswhere 只认 Installer 登记的实例。实测存在这样一种机器：
# vcvars64.bat 能正常跑（cl.exe / link.exe 都在），但 vswhere 返回空 —— 于是本脚本
# 报「没装 C++ 工作负载」，而 cmake 的 "Visual Studio 17 2022" 生成器也找不到实例
# （它同样走 vswhere / 注册表），第三方 C 库（opus）的构建就卡在
#   CMake Error: could not find any instance of Visual Studio.
# 按已知安装位置再找一遍，找不到才报错。
if (-not $vsPath) {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Community'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Professional'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\Enterprise'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\2022\BuildTools'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\BuildTools')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'VC\Auxiliary\Build\vcvars64.bat')) { $vsPath = $candidate; break }
    }
}
if (-not $vsPath) {
    throw "No Visual Studio instance with VC.Tools.x86.x64 -- install the 'Desktop development with C++' workload"
}

$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

# 技巧：让 cmd 先跑 vcvars64.bat，然后 `set` 转储整个环境；
# 逐行解析时按**第一个** '=' 切分（值本身可以含 '='，见下方 IndexOf 用法）。
$dump = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set" 2>$null

foreach ($line in $dump) {
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { continue }
    $name = $line.Substring(0, $idx)
    $value = $line.Substring($idx + 1)
    Set-Item -Force -Path "Env:$name" -Value $value
}

# 【为什么把 cmake 生成器钉成 NMake Makefiles】cmake 的 "Visual Studio 17 2022" 生成器
# 靠 vswhere / 注册表找实例。上面那种「vcvars 能用但 vswhere 看不见」的机器上，它一定失败，
# 而 opusic-sys 的 build.rs 显式请求这个生成器 —— 表现为 `cargo test` 在 opusic-sys 处挂掉，
# 所有 Rust 测试都跑不起来（`cargo check` 不受影响，因为它不构建 C 依赖）。
# NMake 只要 PATH 上有 cl.exe 与 nmake.exe，两个都是 vcvars 刚导进来的。
$env:CMAKE_GENERATOR = 'NMake Makefiles'

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
