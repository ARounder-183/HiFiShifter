param(
    [switch]$SkipFrontend
)

# [A choco failure must be loud] Keep this non-fatal (runners often preinstall
# these tools), but reporting a failure via Write-Host alone drowns in the log --
# NSIS has no other install path, and a silent failure shows up tens of minutes
# later as build.ps1 reporting "makensis not found", unrelated to dependencies.
# Write-Warning goes to stderr with a [warning] prefix, visible at a glance in
# both the CI summary and a local terminal. When choco itself is missing, call
# that out separately (giving the "failed or not available" case a clear home).
$chocoCommand = Get-Command choco.exe -ErrorAction SilentlyContinue
if ($null -eq $chocoCommand) {
    Write-Warning "choco.exe not found on PATH; skipping NSIS / CMake / LLVM installs. NSIS in particular has no fallback installer here - 'build.ps1' will fail with 'makensis not found' when packaging."
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
# try/catch does not catch a native command's nonzero exit (it only sets
# $LASTEXITCODE) -- the check must be explicit. Keep this non-fatal: when the
# runner's toolchain is complete, a missing tauri-cli is exposed by the explicit
# error from the later tauri build.
cargo install tauri-cli --version "^2" --locked -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "tauri-cli install failed (exit $LASTEXITCODE) or already installed; tauri build will report it if actually needed"
}

if (-not $SkipFrontend) {
    Write-Host "[install_deps_windows] Installing frontend deps (npm)"
    # [Why this step must be fatal] The choco / cargo installs above are deliberately
    # non-fatal (runners often preinstall them), but a failed npm ci leaves half a
    # node_modules behind: the errors vite / vitest report later have nothing to do
    # with "dependencies not fully installed". A native command's nonzero exit only
    # sets $LASTEXITCODE; try/catch does not catch it, so the check must be explicit.
    npm --prefix frontend ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed (exit $LASTEXITCODE)" }
}

Write-Host "[install_deps_windows] Done"
