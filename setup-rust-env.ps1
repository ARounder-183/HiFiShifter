# setup-rust-env.ps1
# Download and install a portable, self-contained Rust environment inside the
# project directory at .rust/  — completely isolated from any system-wide Rust.
#
# After running this, you can:
#   . .\env-rust.ps1       — load env into your current shell
#   .\dev-gpu.ps1          — start GPU dev mode (env is configured automatically)
#   .\build-gpu.ps1        — build GPU release (env is configured automatically)

$RustDir   = Join-Path $PSScriptRoot ".rust"
$CargoHome = Join-Path $RustDir "cargo"
$RustupHome = Join-Path $RustDir "rustup"
$CargoBin  = Join-Path $CargoHome "bin\cargo.exe"

Write-Host "=== Setting up Portable Rust Environment ===" -ForegroundColor Cyan
Write-Host "Target: $RustDir" -ForegroundColor Cyan

# Point rustup / cargo at the local directory before doing anything
$env:CARGO_HOME  = $CargoHome
$env:RUSTUP_HOME = $RustupHome

# 1. Check if Rust is already installed locally
if (Test-Path $CargoBin) {
    Write-Host "Portable Rust already installed:" -ForegroundColor Green
    & $CargoBin --version
} else {
    # 2. Create directories
    New-Item -ItemType Directory -Path $RustDir -Force | Out-Null

    # 3. Download rustup-init.exe
    $InstallerPath = Join-Path $RustDir "rustup-init.exe"
    Write-Host "Downloading rustup-init.exe..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $InstallerPath -UserAgent "Mozilla/5.0"
    } catch {
        Write-Error "Download failed. Check your internet connection."
        exit 1
    }

    # 4. Install Rust silently into the local .rust directory
    Write-Host "Installing Rust (this may take a few minutes)..." -ForegroundColor Yellow
    $proc = Start-Process -FilePath $InstallerPath `
        -ArgumentList "-y", "--no-modify-path", "--default-toolchain", "stable" `
        -Wait -NoNewWindow -PassThru
    Remove-Item -Force $InstallerPath

    if (-not (Test-Path $CargoBin)) {
        Write-Error "Rust installation finished but cargo.exe was not found. Something went wrong."
        exit 1
    }
    Write-Host "Rust installed successfully:" -ForegroundColor Green
    & $CargoBin --version
}

# 5. Add cargo/bin to PATH for this session
$binPath = Join-Path $CargoHome "bin"
if ($env:PATH -notlike "*$binPath*") {
    $env:PATH = "$binPath;$env:PATH"
}

# 6. Install tauri-cli if not already present
$CargoTauri = Join-Path $CargoHome "bin\cargo-tauri.exe"
if (Test-Path $CargoTauri) {
    Write-Host "cargo-tauri already installed:" -ForegroundColor Green
    & $CargoTauri --version
} else {
    Write-Host "Installing tauri-cli..." -ForegroundColor Yellow
    & $CargoBin install tauri-cli --locked
    if ($LASTEXITCODE -ne 0) {
        Write-Error "tauri-cli installation failed."
        exit 1
    }
    Write-Host "tauri-cli installed." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host "To load this environment in your current shell:" -ForegroundColor Cyan
Write-Host "  . .\env-rust.ps1" -ForegroundColor White
Write-Host "To start the GPU dev server:" -ForegroundColor Cyan
Write-Host "  .\dev-gpu.ps1" -ForegroundColor White
