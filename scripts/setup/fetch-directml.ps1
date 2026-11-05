# 确保 backend/src-tauri/resources/DirectML.dll 存在。
#
# 【为什么需要它】tauri.windows.conf.json 把 resources/DirectML.dll 列进 bundle.resources，
# tauri-build 会在 **build.rs 里**校验这份清单：文件不在，整个 crate 连 `cargo clippy` 都
# 过不去。而这份 DLL 不是源码 —— 它是 ort 解包 DirectML 版 ONNX Runtime 时的副产物，
# 干净检出上本来就没有。完整的 build.ps1 会先 `cargo build` 出它，门禁流水线只跑
# clippy / test，于是每次都在这里红，报的还是"build script 失败"，看起来像代码坏了。
#
# 顺序：已有 → 复用 target 里的那份 → 按 ort 用的版本从 NuGet 取（1.15.4）。
# 只在 Windows 上有意义；其他平台直接返回。
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$dst = Join-Path $root 'backend\src-tauri\resources\DirectML.dll'

if ([IO.Path]::DirectorySeparatorChar -ne '\') { Write-Host 'DirectML 只在 Windows 上需要，跳过'; return }
if (Test-Path -LiteralPath $dst) { Write-Host "DirectML.dll 已就绪：$dst"; return }

$cached = Get-ChildItem (Join-Path $root 'backend\target') -Recurse -Filter DirectML.dll -File -ErrorAction SilentlyContinue |
    Sort-Object Length -Descending | Select-Object -First 1
if ($cached) {
    Copy-Item $cached.FullName $dst
    Write-Host "DirectML.dll 取自构建缓存：$($cached.FullName)"
    return
}

# 版本必须与 ort 解包的那份一致（见 Microsoft.AI.DirectML NuGet）。
$version = '1.15.4'
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("directml-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $nupkg = Join-Path $tmp 'directml.zip'
    Write-Host "下载 Microsoft.AI.DirectML $version ..."
    Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$version" -OutFile $nupkg -UseBasicParsing
    Expand-Archive -Path $nupkg -DestinationPath (Join-Path $tmp 'pkg') -Force
    $dll = Get-ChildItem (Join-Path $tmp 'pkg') -Recurse -Filter DirectML.dll -File | Select-Object -First 1
    if (!$dll) { throw 'NuGet 包里没有 DirectML.dll' }
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item $dll.FullName $dst
    Write-Host "DirectML.dll 取自 NuGet：$($dll.FullName)"
}
finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
