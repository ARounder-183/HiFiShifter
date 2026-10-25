param(
    [string]$ToolchainRoot = 'G:\hjs-tc',
    [string]$BuildDirectory,
    [ValidateRange(1, 64)]
    [int]$Jobs = 4,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$env:HACHI_TC_ROOT = $ToolchainRoot
$env:TEMP = Join-Path $ToolchainRoot 'tmp'
$env:TMP = $env:TEMP
$tools = Join-Path $ToolchainRoot 'packages\clang64\bin'
$env:PATH = "$tools;$env:PATH"
# JUCE bootstraps juceaide during configure; bound that nested build as well.
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$cmake = Join-Path $tools 'cmake.exe'
$source = Join-Path $PSScriptRoot 'juce'
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $ToolchainRoot 'build\hachishifter-integrated-otto'
}
$build = [System.IO.Path]::GetFullPath($BuildDirectory)
$cache = Join-Path $ToolchainRoot 'fetch'
$toolchain = Join-Path $source 'cmake\windows-msys2-xwin-native.cmake'
# Never reuse another source tree's CMake cache (including the old release build).
$cmakeCache = Join-Path $build 'CMakeCache.txt'
if (Test-Path $cmakeCache) {
    $homeEntry = Select-String -Path $cmakeCache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$'
    if ($homeEntry) {
        $cachedSource = [System.IO.Path]::GetFullPath($homeEntry.Matches[0].Groups[1].Value)
        if ($cachedSource -ne [System.IO.Path]::GetFullPath($source)) {
            throw "Build directory belongs to another source tree: $cachedSource"
        }
    }
}
$deps = Join-Path $build '_deps'
New-Item -ItemType Directory -Force $deps | Out-Null
$configureArgs = @('-S', $source, '-B', $build, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DFETCHCONTENT_BASE_DIR=$deps", '-DHACHI_ENABLE_ONNX_ANALYSIS=ON')
# Shared dependency build/subbuild caches contain absolute paths from the old
# E: toolchain. Reuse downloaded sources only, with build state under this build.
foreach ($dependency in @('directml_runtime', 'onnxruntime')) {
    $cachedDependency = Join-Path $cache "$dependency-src"
    if (Test-Path $cachedDependency) {
        $configureArgs += "-DFETCHCONTENT_SOURCE_DIR_$($dependency.ToUpperInvariant())=$cachedDependency"
    }
}
# Project CMake patches JUCE in place. Keep the shared checkout untouched and
# repair its pre-existing juceaide toolchain path only in our private copy.
$cachedJuce = Join-Path $cache 'juce-src'
$privateJuce = Join-Path $deps 'juce-src'
if (Test-Path $cachedJuce) {
    if (-not (Test-Path (Join-Path $privateJuce 'CMakeLists.txt'))) {
        New-Item -ItemType Directory -Force $privateJuce | Out-Null
        Get-ChildItem -LiteralPath $cachedJuce -Force |
            Where-Object { $_.Name -ne '.git' } |
            Copy-Item -Destination $privateJuce -Recurse -Force
    }
    $juceaideFile = Join-Path $privateJuce 'extras\Build\juceaide\CMakeLists.txt'
    $juceaide = [System.IO.File]::ReadAllText($juceaideFile)
    $nativeToolchain = $toolchain.Replace('\', '/')
    $repaired = [regex]::Replace($juceaide,
        '"-DCMAKE_TOOLCHAIN_FILE=[^"\r\n]*"(?= # HachiShifter passes the native toolchain)',
        "`"-DCMAKE_TOOLCHAIN_FILE=$nativeToolchain`"")
    if ($repaired -ne $juceaide) {
        [System.IO.File]::WriteAllText($juceaideFile, $repaired)
    }
    $configureArgs += "-DFETCHCONTENT_SOURCE_DIR_JUCE=$privateJuce"
}
Write-Host "Configuring integrated source: $source"
Write-Host "Dedicated build directory: $build"
& $cmake @configureArgs
$configureExit = $LASTEXITCODE
if ($configureExit -ne 0) { throw "CMake configure failed ($configureExit)" }
if (-not $ConfigureOnly) {
    & $cmake --build $build --target HachiShifterNext --parallel $Jobs
    $buildExit = $LASTEXITCODE
    if ($buildExit -ne 0) { throw "Build failed ($buildExit)" }
}
