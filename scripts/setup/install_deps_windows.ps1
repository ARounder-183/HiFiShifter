param(
    [switch]$SkipFrontend
)

# 【choco 失败必须喊出来】保持非致命（runner 常预装这些工具），但失败只写
# Write-Host 会在日志里淹没 —— NSIS 没有别的安装路径，静默失败的症状是几十分钟
# 后 build.ps1 报"找不到 makensis"，与依赖毫无关系。Write-Warning 走 stderr 并带
# [warning] 前缀，CI 摘要与本地终端都能一眼看到。choco 本身不存在时单独提示
# （那条"failed or not available"从此有了明确的归属）。
$chocoCommand = Get-Command choco.exe -ErrorAction SilentlyContinue
if ($null -eq $chocoCommand) {
    Write-Warning "choco.exe not found on PATH; skipping NSIS / CMake / LLVM installs. NSIS in particular has no fallback installer here - 'build.ps1' will fail with '找不到 makensis' when packaging."
}
else {
    foreach ($tool in @('nsis', 'cmake', 'llvm')) {
        Write-Host "[install_deps_windows] Installing $tool via chocolatey"
        choco install $tool -y
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "choco install $tool failed (exit $LASTEXITCODE); continuing - later steps will report the missing tool explicitly."
        }
    }
}

Write-Host "[install_deps_windows] Installing tauri-cli via cargo"
# try/catch 捕不到原生命令的非零退出（那只写 $LASTEXITCODE）—— 必须显式检查。
# 保持非致命：runner 预装链齐全时缺 tauri-cli 会由后续 tauri build 的显式报错暴露。
cargo install tauri-cli --version "^2" --locked -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "tauri-cli install failed (exit $LASTEXITCODE) or already installed; tauri build will report it if actually needed"
}

if (-not $SkipFrontend) {
    Write-Host "[install_deps_windows] Installing frontend deps (npm)"
    # 【为什么这一步必须判死】上面的 choco / cargo install 刻意保持非致命
    # （runner 常预装），但 npm ci 失败会留下半套 node_modules：后续 vite /
    # vitest 报的错与"依赖没装齐"毫无关系。原生命令的非零退出只写
    # $LASTEXITCODE，try/catch 捕不到，必须显式检查。
    npm --prefix frontend ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed (exit $LASTEXITCODE)" }
}

Write-Host "[install_deps_windows] Done"
