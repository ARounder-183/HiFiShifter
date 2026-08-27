# test-win.ps1 — 在 Windows 上运行后端（src-tauri）测试。
#
# 背景：
#   1. vslib_x64.dll / SoundTouchDLL.dll 位于 target/debug/（主 bin 所在），
#      lib 测试 harness 在 target/debug/deps/ 下，加载器找不到 → DLL_NOT_FOUND。
#      build.rs 已把这两个 DLL 同步复制到 deps/（见 build_vslib/build_soundtouch）。
#   2. 依赖树静态导入了 comctl32.dll!TaskDialogIndirect（仅 v6 SxS 程序集有）。
#      tauri_build 只给 bin 嵌清单；cargo 的 rustc-link-arg-tests 只到 tests/
#      集成测试；lib 单元测试 harness 无任何链接期通道（cargo 限制）→ 无清单
#      时绑定到 v5 副本 → 进程初始化即 STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139)。
#      解决：给链接步骤注入 /MANIFEST:EMBED（本脚本通过 RUSTFLAGS 实现）。
#   3. RUSTFLAGS 会同时作用于主 bin —— 与 tauri_build 的 resource.lib 内嵌
#      清单重复 → CVT1100 duplicate resource。因此本脚本只跑 lib/集成测试/
#      benches/examples 目标，不构建主 bin（main.rs 无单测，无损失）。
#
# 用法（仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts/test-win.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/test-win.ps1 -- --nocapture
# 额外参数原样传给 cargo test；CI/Linux 不受影响（仍用 cargo test --all-targets）。

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$crateDir = Join-Path $root "backend\src-tauri"

$manifestFile = Join-Path $env:TEMP "hifishifter_v6_test.manifest"
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
$v6Manifest | Set-Content -Path $manifestFile -Encoding UTF8

Push-Location $crateDir
try {
    # --lib：单元测试（cargo 无链接期通道，需 RUSTFLAGS 注入清单）。
    $env:RUSTFLAGS = "-C link-arg=/MANIFEST:EMBED -C link-arg=/MANIFESTINPUT:$manifestFile -C link-arg=/MANIFESTUAC:NO"
    & cargo test --locked --features __test-internals --lib @args
    $libExit = $LASTEXITCODE
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    if ($libExit -ne 0) { exit $libExit }

    # 集成测试目标（tests/smoke.rs）：build.rs 已通过 rustc-link-arg-tests 嵌入
    # 清单，无需 RUSTFLAGS；且 bin 的 test 目标会链接 tauri_build 的
    # resource.lib（内嵌清单），再加 /MANIFEST:EMBED 会 CVT1100 duplicate
    # resource —— 必须保持 RUSTFLAGS 关闭。注意 --tests 会连带 lib 单元测试
    # （无清单 harness），因此显式指定 `--test smoke`。
    & cargo test --locked --features __test-internals --test smoke @args
    exit $LASTEXITCODE
} finally {
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    Pop-Location
}