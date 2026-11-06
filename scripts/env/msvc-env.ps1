# 把 MSVC 环境导入当前会话（点源，不是执行）：. .\scripts\env\msvc-env.ps1
#
# cc / cmake 两个 crate 直接调 cl.exe，它们不会自己初始化 MSVC 环境。没有 INCLUDE / LIB /
# PATH 时 cl.exe 会在启动编译器后端时报 D8050（"cl.exe 存在但编译全部失败"），所以这一步
# 不能静默跳过。

$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path $vswhere)) { throw "找不到 vswhere.exe：$vswhere（需要带「使用 C++ 的桌面开发」工作负载的 Visual Studio）" }

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsPath) { throw '找不到带「使用 C++ 的桌面开发」工作负载的 Visual Studio 实例' }
$vcvars = Join-Path $vsPath.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
if (!(Test-Path $vcvars)) { throw "找不到 $vcvars" }

# 让 cmd 先跑 vcvars64.bat 再 `set` 转储整个环境；逐行按第一个 '=' 切分（值本身可以含 '='）。
foreach ($line in (& cmd.exe /c "`"$vcvars`" >nul 2>&1 && set")) {
    $split = $line.IndexOf('=')
    if ($split -lt 1) { continue }
    Set-Item -Path "Env:$($line.Substring(0, $split))" -Value $line.Substring($split + 1)
}

# cmake 的 "Visual Studio 17 2022" 生成器靠注册表找实例，而 opusic-sys 显式请求它；
# 在「vcvars 能用但那个生成器找不到实例」的机器上构建会挂在 opusic-sys。NMake 只要求
# PATH 上有 cl.exe 与 nmake.exe —— 两者都是 vcvars 刚导进来的。
$env:CMAKE_GENERATOR = 'NMake Makefiles'
