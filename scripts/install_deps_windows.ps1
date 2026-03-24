param(
    [switch]$SkipFrontend
)

Write-Host "[install_deps_windows] Installing NSIS via chocolatey (if available)"
choco install nsis -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "choco or NSIS install failed or not available"
}

Write-Host "[install_deps_windows] Installing LLVM/Clang via chocolatey (if available)"
choco install llvm -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "choco or llvm install failed or not available"
}

Write-Host "[install_deps_windows] Installing tauri-cli via cargo"
try {
    cargo install tauri-cli --version "^2" --locked -q
} catch {
    Write-Host "tauri-cli install failed or already installed: $_"
}

if (-not $SkipFrontend) {
    Write-Host "[install_deps_windows] Installing frontend deps (npm)"
    npm --prefix frontend ci
}

Write-Host "[install_deps_windows] Done"
