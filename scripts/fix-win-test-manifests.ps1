# fix-win-test-manifests.ps1 — 为 Windows 上的 cargo test 修复 lib 单元测试 harness 启动失败。
#
# 背景：
#   依赖树（winit / tauri dialog）静态导入了 comctl32.dll!TaskDialogIndirect，
#   该函数只存在于 v6 side-by-side 程序集。主 bin 与 tests/ 集成测试目标由
#   tauri_build / build.rs（rustc-link-arg-tests）嵌入 ComCtl32 v6 清单；
#   但 cargo 没有向 lib 单元测试 harness exe 传 link-arg 的通道（cargo 限制，
#   见 backend/src-tauri/tests/smoke.rs 说明），于是 harness exe 无清单，
#   加载器把 comctl32 导入绑定到 System32 的 v5 副本，进程初始化立即
#   STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139) 失败，cargo test --lib 无法启动。
#
# 修复方式：
#   用 Windows SDK 的 mt.exe 把 v6 清单合并进缺失清单的测试 harness exe。
#
# 用法（在仓库根目录执行）：
#   cargo test --no-run [--features __test-internals]   # 先产出 harness exe
#   powershell -ExecutionPolicy Bypass -File scripts/fix-win-test-manifests.ps1
#   cargo test                                          # 直接复用已构建产物
#
# 重复运行安全：已含 v6 清单的 exe 自动跳过。

$ErrorActionPreference = "Stop"

# 定位 mt.exe（Windows SDK 各版本/架构路径）
$mtCandidates = @(
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\mt.exe",
    "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64\mt.exe"
)
$mt = $mtCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $mt) {
    $mt = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter mt.exe -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $mt) {
    # VS 自带的 mt.exe 兜底
    $mt = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022" -Recurse -Filter mt.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $mt) {
    Write-Error "mt.exe 未找到（需要 Windows SDK 或 Visual Studio）。无法嵌入 ComCtl32 v6 清单。"
}

$v6Manifest = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="*"
        publicKeyToken="6595b64144ccf1df"
        language="*"
      />
    </dependentAssembly>
  </dependency>
</assembly>
'@
$manifestFile = Join-Path $env:TEMP "hifishifter_v6_test.manifest"
$v6Manifest | Set-Content -Path $manifestFile -Encoding UTF8

$depsDir = Join-Path $PSScriptRoot "..\backend\src-tauri\target\debug\deps"
$exes = Get-ChildItem $depsDir -Filter "*.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^(backend_lib|HiFiShifter)-" }

$fixed = 0
$skipped = 0
foreach ($exe in $exes) {
    # 已有 v6 清单的 exe（manifest 资源已含 Common-Controls 6.0）跳过
    $hasV6 = & $mt -nologo "-inputresource:$($exe.FullName);1" 2>$null |
        Select-String -Pattern "Common-Controls.*6\.0" -Quiet
    if ($hasV6) {
        $skipped++
        continue
    }
    & $mt -nologo -manifest $manifestFile -outputresource:"$($exe.FullName);1" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "mt.exe 失败: $($exe.Name)"
        continue
    }
    Write-Host "[fix] embedded ComCtl32 v6 manifest -> $($exe.Name)"
    $fixed++
}

Write-Host "完成：修复 $fixed 个，跳过 $skipped 个（已含 v6 清单）。"
if ($fixed -eq 0) {
    Write-Host "提示：若缺少 harness exe，请先运行 cargo test --no-run。"
}