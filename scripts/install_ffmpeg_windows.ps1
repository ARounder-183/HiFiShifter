param(
    [ValidateSet("x64", "arm64")]
    [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$TauriDir = Join-Path $ProjectRoot "backend\src-tauri"
$ProvisionDir = Join-Path $TauriDir "third_party\ffmpeg"
$ResourceDir = Join-Path $TauriDir "resources\ffmpeg"

$ArchToken = if ($Arch -eq "arm64") { "winarm64" } else { "win64" }
$AssetName = "ffmpeg-n8.1-latest-$ArchToken-lgpl-shared-8.1.zip"
$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$AssetName"
$Archive = Join-Path $ProvisionDir $AssetName

New-Item -ItemType Directory -Force -Path $ProvisionDir, $ResourceDir | Out-Null

# Reuse only an already provisioned FFmpeg **8.1** tree (avoid picking an
# older major version left by a previous local build).
$Existing = Get-ChildItem -Path $ProvisionDir -Directory |
    Where-Object {
        -not ($_.Name -eq "current") -and
        ($Header = Join-Path $_.FullName "include\libavcodec\version_major.h") -and
        (Test-Path $Header) -and ((Get-Content -Raw $Header) -match "LIBAVCODEC_VERSION_MAJOR\s+62")
    } |
    Select-Object -First 1
if ($null -eq $Existing) {
    if (-not (Test-Path $Archive)) {
        Write-Host "[ffmpeg] Downloading LGPL shared FFmpeg 8.1 ($Arch): $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Archive
    }
    Write-Host "[ffmpeg] Extracting $AssetName"
    Expand-Archive -Path $Archive -DestinationPath $ProvisionDir -Force
}

$FfmpegDir = Get-ChildItem -Path $ProvisionDir -Directory |
    Where-Object {
        -not ($_.Name -eq "current") -and
        ($Header = Join-Path $_.FullName "include\libavcodec\version_major.h") -and
        (Test-Path $Header) -and ((Get-Content -Raw $Header) -match "LIBAVCODEC_VERSION_MAJOR\s+62")
    } |
    Select-Object -First 1
if ($null -eq $FfmpegDir) {
    throw "FFmpeg provisioning failed: include/libavformat/avformat.h not found under $ProvisionDir"
}

Write-Host "[ffmpeg] Using FFMPEG_DIR=$($FfmpegDir.FullName)"

# Copy the four runtime DLLs (avcodec already depends on swresample) and the
# LGPL license into Tauri resources so the NSIS/portable bundles are complete.
foreach ($Dll in @("avcodec-62.dll", "avformat-62.dll", "avutil-60.dll", "swresample-6.dll")) {
    $Src = Join-Path $FfmpegDir.FullName "bin\$Dll"
    if (Test-Path $Src) {
        Copy-Item -Force $Src (Join-Path $ResourceDir $Dll)
        Write-Host "[ffmpeg] staged $Dll"
    }
    else {
        throw "Missing FFmpeg runtime DLL: $Src"
    }
}
foreach ($Stale in @("avcodec-61.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll")) {
    Remove-Item -Force (Join-Path $ResourceDir $Stale) -ErrorAction SilentlyContinue
}
$License = Join-Path $FfmpegDir.FullName "LICENSE.txt"
if (Test-Path $License) {
    Copy-Item -Force $License (Join-Path $ResourceDir "LICENSE.txt")
}

# Expose the selected SDK through a fixed "current" junction. The Cargo
# [env] table forces FFMPEG_DIR to this path, so switching FFmpeg majors can
# never leave a stale import library in target/release.
$CurrentLink = Join-Path $ProvisionDir "current"
if (Test-Path $CurrentLink) {
    # Remove the junction itself without recursing into its target.
    cmd /c rmdir "$CurrentLink" 2>$null | Out-Null
}
New-Item -ItemType Junction -Path $CurrentLink -Target $FfmpegDir.FullName | Out-Null
Write-Host "[ffmpeg] current -> $($FfmpegDir.FullName)"

# bindgen needs libclang.dll. Chocolatey's LLVM package usually installs it
# under Program Files\LLVM\bin; export LIBCLANG_PATH for the Cargo build step.
$LibclangCandidates = @(
    (Join-Path $env:ProgramFiles "LLVM\bin"),
    (Join-Path ${env:ProgramFiles(x86)} "LLVM\bin"),
    (Join-Path $FfmpegDir.FullName "bin")
)
$LibclangDir = $LibclangCandidates |
    Where-Object { $_ -and (Test-Path (Join-Path $_ "libclang.dll")) } |
    Select-Object -First 1
if (-not $LibclangDir) {
    $Found = Get-ChildItem -Path "$env:ProgramFiles\LLVM", "${env:ProgramFiles(x86)}\LLVM" -Filter "libclang.dll" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($Found) { $LibclangDir = $Found.DirectoryName }
}
if (-not $LibclangDir) {
    throw "libclang.dll not found. Install LLVM (for example: choco install llvm -y) and retry."
}

# Export for subsequent build steps.
if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "LIBCLANG_PATH=$LibclangDir"
    Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $FfmpegDir.FullName "bin")
}
$env:LIBCLANG_PATH = $LibclangDir
$env:PATH = "$(Join-Path $FfmpegDir.FullName 'bin');$env:PATH"

Write-Host "[ffmpeg] Windows provisioning complete"
