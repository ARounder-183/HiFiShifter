# 把 MSVC 环境导入当前会话（点源，不是执行）：. .\scripts\env\msvc-env.ps1
#
# cc / cmake 两个 crate 直接调 cl.exe，它们不会自己初始化 MSVC 环境。没有 INCLUDE / LIB /
# PATH 时 cl.exe 会在启动编译器后端时报 D8050（"cl.exe 存在但编译全部失败"），所以这一步
# 不能静默跳过。
#
# 【为什么整份实现包在一个函数里】本文件是**点源**的：它的顶层变量全部落进调用者的作用域。
# 此前顶层有一个 `foreach ($root in ...)`，于是调用者的 `$root`（仓库根）被覆盖成
# Visual Studio 的安装目录 —— 后续每个 `Join-Path $root ...` 都拼到
# `C:\Program Files\Microsoft Visual Studio\2022\Community\` 下面，表现是
# "找不到 backend\src-tauri\resources\models\models.json"。函数体内的变量是局部的，
# 导入器只把 `$env:*` 留在进程里，那正是它唯一该留下的东西。
#
# 【为什么不再只认 vswhere】vswhere 只是**一条**发现路径，不是唯一事实：实测某些机器的
# vswhere 对 `-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` 返回空
# （工作负载装在别的产品里、或 vswhere 版本过旧），而 VS 本身完全可用。现在按**可执行
# 证据**发现：谁下面真的有 `VC\Auxiliary\Build\vcvars64.bat` 就用谁，顺序是
# 已导入的环境变量 → vswhere → 注册表 → 安装目录扫描。
#
# 【为什么不写死盘符或目录】仓库可能在任何盘、任何目录；VS 可能在 Program Files 也可能
# 在 Program Files (x86)。所有位置都由系统 API / 环境变量推出，不出现字面路径。

$ErrorActionPreference = 'Stop'

function Import-MsvcEnvironment {
    # 幂等：在 VS 开发者命令提示符里点源，或同一次构建里被点源两次时，不要重复导入。
    # 重复导入会把 vcvars 的 PATH 再叠一遍，最终超过 8191 字符的环境块上限。
    if ($env:VSCMD_VER -and $env:INCLUDE -and $env:LIB -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        return
    }

    function Get-ProgramFilesRoots {
        # 用系统 API 取，而不是读环境变量：$env:ProgramFiles(x86) 在某些宿主里不存在。
        $programFiles = @()
        foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
            $path = [Environment]::GetFolderPath($kind)
            if ($path) { $programFiles += $path }
        }
        return $programFiles | Select-Object -Unique
    }

    function Find-VsWhere {
        foreach ($programFiles in Get-ProgramFilesRoots) {
            $candidate = Join-Path $programFiles 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
        $onPath = Get-Command vswhere.exe -ErrorAction SilentlyContinue
        if ($onPath) { return $onPath.Source }
        return $null
    }

    function Get-VsInstallRoots {
        $found = @()
        if ($env:VCINSTALLDIR) { $found += (Split-Path -Parent (Split-Path -Parent $env:VCINSTALLDIR)) }
        if ($env:VSINSTALLDIR) { $found += $env:VSINSTALLDIR }

        $vswhere = Find-VsWhere
        if ($vswhere) {
            # 先按组件要求找；找不到再放宽（组件 ID 随工作负载而异，而"有 vcvars64.bat"才是硬判据）。
            foreach ($requires in @('Microsoft.VisualStudio.Component.VC.Tools.x86.x64', $null)) {
                if ($requires) {
                    $hits = & $vswhere -latest -products * -requires $requires -property installationPath 2>$null
                } else {
                    $hits = & $vswhere -latest -products * -property installationPath 2>$null
                }
                foreach ($hit in @($hits)) {
                    if ($hit -and $hit.Trim()) { $found += $hit.Trim() }
                }
                if ($found.Count -gt 0) { break }
            }
        }

        foreach ($key in 'HKLM:\SOFTWARE\Microsoft\VisualStudio\SxS\VS7', 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7') {
            if (!(Test-Path $key)) { continue }
            foreach ($value in (Get-ItemProperty $key).PSObject.Properties) {
                if ($value.Name -match '^\d+\.\d+$' -and $value.Value) { $found += [string]$value.Value }
            }
        }

        foreach ($programFiles in Get-ProgramFilesRoots) {
            $base = Join-Path $programFiles 'Microsoft Visual Studio'
            if (!(Test-Path -LiteralPath $base)) { continue }
            $found += Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue } |
                ForEach-Object { $_.FullName }
        }
        return $found | Where-Object { $_ } | Select-Object -Unique
    }

    $vcvars = $null
    foreach ($candidateRoot in Get-VsInstallRoots) {
        $candidate = Join-Path $candidateRoot.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
        if (Test-Path -LiteralPath $candidate) { $vcvars = $candidate; break }
    }
    if (!$vcvars) {
        throw ('找不到 VC\Auxiliary\Build\vcvars64.bat。需要带「使用 C++ 的桌面开发」工作负载的 Visual Studio；' +
            '也可以先在「x64 Native Tools Command Prompt for VS」里启动 PowerShell 再跑构建。')
    }

    # 【为什么写一个临时 .cmd 而不是 cmd /c "call ..."】Windows 的命令行没有"带引号路径"
    # 的统一转义规则：从 PowerShell / Node 把 call "C:\Program Files\..." 拼成一条命令行时，
    # 引号会被宿主吃掉或转义成 \\"，cmd 于是把整串当成一个不存在的命令名。写进 .cmd 文件
    # 之后由 cmd 自己解析，路径里有空格、中文、任何字符都不需要再转义。
    $transcript = Join-Path ([IO.Path]::GetTempPath()) ("hfs-vcvars-" + [Guid]::NewGuid().ToString('N') + '.cmd')
    $lines = @(
        '@echo off',
        ('call "{0}" >nul 2>&1' -f $vcvars),
        'if errorlevel 1 exit /b 1',
        'set'
    )
    [IO.File]::WriteAllLines($transcript, $lines, [Text.Encoding]::Default)
    try {
        # 【为什么不用 $env:ComSpec】极简宿主（CI 容器、按需构造的环境）里它可能是空的。
        $comspec = if ($env:ComSpec) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }
        $dump = & $comspec /c $transcript
        if ($LASTEXITCODE) { throw "vcvars64.bat 执行失败（退出码 $LASTEXITCODE）：$vcvars" }
    } finally {
        Remove-Item -LiteralPath $transcript -Force -ErrorAction SilentlyContinue
    }

    # 逐行按第一个 '=' 切分（值本身可以含 '='）。
    foreach ($line in $dump) {
        $split = $line.IndexOf('=')
        if ($split -lt 1) { continue }
        Set-Item -Path "Env:$($line.Substring(0, $split))" -Value $line.Substring($split + 1)
    }
    if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "vcvars64.bat 已执行但 PATH 上仍然没有 cl.exe：$vcvars"
    }

    # cmake 的 "Visual Studio 17 2022" 生成器靠注册表找实例，而 opusic-sys 显式请求它；
    # 在「vcvars 能用但那个生成器找不到实例」的机器上构建会挂在 opusic-sys。NMake 只要求
    # PATH 上有 cl.exe 与 nmake.exe —— 两者都是 vcvars 刚导进来的。
    $env:CMAKE_GENERATOR = 'NMake Makefiles'
}

Import-MsvcEnvironment
