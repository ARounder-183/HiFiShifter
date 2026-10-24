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
    npm --prefix frontend ci
}

Write-Host "[install_deps_windows] Done"
