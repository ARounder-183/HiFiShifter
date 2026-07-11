# env-rust.ps1
# Setup the standalone Rust environment in the current PowerShell session.
# Automatically detects local project-relative environment at .rust
#
# Usage: . .\env-rust.ps1   (Note the dot space dot slash style to run in current scope)

$localRustCargo = Join-Path $PSScriptRoot ".rust\cargo"
$localRustup = Join-Path $PSScriptRoot ".rust\rustup"

if (Test-Path $localRustCargo) {
    $env:CARGO_HOME = $localRustCargo
    $env:RUSTUP_HOME = $localRustup
    $binPath = Join-Path $localRustCargo "bin"
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }
    Write-Host "[env-rust] Standalone Rust environment configured!" -ForegroundColor Green
    Write-Host "  CARGO_HOME  = $env:CARGO_HOME"
    Write-Host "  RUSTUP_HOME = $env:RUSTUP_HOME"
    Write-Host "  PATH        = (Includes cargo/bin)"
    cargo --version
} else {
    Write-Error "[env-rust] Standalone Rust environment not found. Please run .\setup-rust-env.ps1 first."
}
