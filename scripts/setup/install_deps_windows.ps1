param(
    [switch]$SkipFrontend
)

Write-Host "[install_deps_windows] Installing NSIS via chocolatey (if available)"
choco install nsis -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "choco or NSIS install failed or not available"
}

Write-Host "[install_deps_windows] Installing CMake via chocolatey (if available)"
choco install cmake -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "choco or cmake install failed or not available"
}

Write-Host "[install_deps_windows] Installing LLVM/Clang via chocolatey (if available)"
choco install llvm -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "choco or llvm install failed or not available"
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
