# Import the MSVC environment into the current session (dot-source it, do not execute): . .\scripts\env\msvc-env.ps1
#
# The cc / cmake crates call cl.exe directly and do not initialize the MSVC environment
# themselves. Without INCLUDE / LIB / PATH, cl.exe reports D8050 once the compiler backend
# starts ("cl.exe exists but every compilation fails"), so this step cannot be silently
# skipped.
#
# [Why the whole implementation lives inside one function] This file is **dot-sourced**:
# its top-level variables all land in the caller's scope. There used to be a top-level
# `foreach ($root in ...)`, which overwrote the caller's `$root` (the repo root) with the
# Visual Studio install directory -- every later `Join-Path $root ...` then resolved under
# `C:\Program Files\Microsoft Visual Studio\2022\Community\`, presenting as
# "backend\src-tauri\resources\models\models.json not found". Variables inside a function
# are local; the importer leaves only `$env:*` behind in the process, which is exactly the
# one thing it should leave behind.
#
# [Why vswhere alone is no longer trusted] vswhere is **one** discovery path, not the only
# truth: on some machines vswhere returns nothing for
# `-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` (the workload lives in
# another product, or the vswhere version is too old), while VS itself works fine.
# Discovery now goes by **executable evidence**: whichever root actually contains
# `VC\Auxiliary\Build\vcvars64.bat` wins, in the order already-imported env vars ->
# vswhere -> registry -> install-directory scan.
#
# [Why no drive letter or directory is hardcoded] The repository may live on any drive, in
# any directory; VS may be under Program Files or Program Files (x86). Every location is
# derived from system APIs / environment variables; no literal paths appear.

$ErrorActionPreference = 'Stop'

# [Why the generator is decided here, by a shared module] This file is the entry point for
# every Windows build, so it is the one place where "which CMake generator does this build
# use" can be answered once for the whole process. The answer lives in
# scripts/env/native-toolchain.ps1 because android-env.ps1 needs the same answer (with a
# different value) and because opusic-sys' own ninja probe has to be neutralised, which is
# not something a single assignment can do. See that file's header for the full incident.
. (Join-Path $PSScriptRoot 'native-toolchain.ps1')

function Import-MsvcEnvironment {
    # Idempotent: when dot-sourced inside a VS developer command prompt, or dot-sourced
    # twice within one build, do not import again. A repeated import stacks vcvars' PATH
    # yet again and eventually exceeds the 8191-character environment block limit.
    if ($env:VSCMD_VER -and $env:INCLUDE -and $env:LIB -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        # [Why the toolchain is still pinned on this branch] Inside an "x64 Native Tools
        # Command Prompt" the environment is already imported, so this early return used to
        # skip the ONLY line that ever set CMAKE_GENERATOR -- the build then inherited
        # whatever the shell happened to have (often nothing, and after an Android build in
        # the same session, Ninja). "Already imported" says nothing about the generator.
        Set-NativeToolchain
        return
    }

    function Get-ProgramFilesRoots {
        # Use the system API instead of reading env vars: $env:ProgramFiles(x86) does not exist in some hosts.
        $programFiles = @()
        foreach ($kind in 'ProgramFilesX86', 'ProgramFiles') {
            $path = [Environment]::GetFolderPath($kind)
            if ($path) { $programFiles += $path }
        }
        return $programFiles | Select-Object -Unique
    }

    function Find-VsWhere {
        foreach ($programFiles in Get-ProgramFilesRoots) {
            $candidate = Join-Path $programFiles 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
        $onPath = Get-Command vswhere.exe -ErrorAction SilentlyContinue
        if ($onPath) { return $onPath.Source }
        return $null
    }

    function Get-VsInstallRoots {
        $found = @()
        if ($env:VCINSTALLDIR) { $found += (Split-Path -Parent (Split-Path -Parent $env:VCINSTALLDIR)) }
        if ($env:VSINSTALLDIR) { $found += $env:VSINSTALLDIR }

        $vswhere = Find-VsWhere
        if ($vswhere) {
            # Search with the component requirement first, then relax (component IDs vary by
            # workload; "has vcvars64.bat" is the hard criterion).
            foreach ($requires in @('Microsoft.VisualStudio.Component.VC.Tools.x86.x64', $null)) {
                if ($requires) {
                    $hits = & $vswhere -latest -products * -requires $requires -property installationPath 2>$null
                } else {
                    $hits = & $vswhere -latest -products * -property installationPath 2>$null
                }
                foreach ($hit in @($hits)) {
                    if ($hit -and $hit.Trim()) { $found += $hit.Trim() }
                }
                if ($found.Count -gt 0) { break }
            }
        }

        foreach ($key in 'HKLM:\SOFTWARE\Microsoft\VisualStudio\SxS\VS7', 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7') {
            if (!(Test-Path $key)) { continue }
            foreach ($value in (Get-ItemProperty $key).PSObject.Properties) {
                if ($value.Name -match '^\d+\.\d+$' -and $value.Value) { $found += [string]$value.Value }
            }
        }

        foreach ($programFiles in Get-ProgramFilesRoots) {
            $base = Join-Path $programFiles 'Microsoft Visual Studio'
            if (!(Test-Path -LiteralPath $base)) { continue }
            $found += Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue } |
                ForEach-Object { $_.FullName }
        }
        return $found | Where-Object { $_ } | Select-Object -Unique
    }

    $vcvars = $null
    foreach ($candidateRoot in Get-VsInstallRoots) {
        $candidate = Join-Path $candidateRoot.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
        if (Test-Path -LiteralPath $candidate) { $vcvars = $candidate; break }
    }
    if (!$vcvars) {
        throw ('VC\Auxiliary\Build\vcvars64.bat not found. Visual Studio with the "Desktop development with C++" workload is required; ' +
            'alternatively, start PowerShell from the "x64 Native Tools Command Prompt for VS" and run the build there.')
    }

    # [Why a temporary .cmd file instead of cmd /c "call ..."] Windows command lines have
    # no universal escaping rule for quoted paths: when PowerShell / Node assemble
    # call "C:\Program Files\..." into one command line, the quotes get eaten by the host
    # or escaped into \\", and cmd then treats the whole string as a nonexistent command
    # name. Once written into a .cmd file, cmd parses it itself; spaces, CJK characters,
    # anything in the path needs no further escaping.
    $transcript = Join-Path ([IO.Path]::GetTempPath()) ("hfs-vcvars-" + [Guid]::NewGuid().ToString('N') + '.cmd')
    $lines = @(
        '@echo off',
        ('call "{0}" >nul 2>&1' -f $vcvars),
        'if errorlevel 1 exit /b 1',
        'set'
    )
    [IO.File]::WriteAllLines($transcript, $lines, [Text.Encoding]::Default)
    try {
        # [Why not rely on $env:ComSpec] In minimal hosts (CI containers, environments built on demand) it can be empty.
        $comspec = if ($env:ComSpec) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }
        $dump = & $comspec /c $transcript
        if ($LASTEXITCODE) { throw "vcvars64.bat failed (exit code $LASTEXITCODE): $vcvars" }
    } finally {
        Remove-Item -LiteralPath $transcript -Force -ErrorAction SilentlyContinue
    }

    # Split each line at the first '=' (values themselves may contain '=').
    foreach ($line in $dump) {
        $split = $line.IndexOf('=')
        if ($split -lt 1) { continue }
        Set-Item -Path "Env:$($line.Substring(0, $split))" -Value $line.Substring($split + 1)
    }
    if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "vcvars64.bat ran but cl.exe is still not on PATH: $vcvars"
    }

    # [Why the generator is pinned after vcvars, not before] The probe that decides between
    # Ninja and NMake looks at PATH, and vcvars is what puts cl.exe / nmake.exe there. Pinning
    # before the import would answer a question about a different environment than the one
    # cargo will actually run in.
    #
    # cmake's "Visual Studio 17 2022" generator locates instances via the registry, and
    # opusic-sys explicitly requests it; on machines where "vcvars works but that generator
    # cannot find an instance" the build hangs inside opusic-sys. Both remaining choices only
    # require tools vcvars just put on PATH.
    Set-NativeToolchain
}

Import-MsvcEnvironment
