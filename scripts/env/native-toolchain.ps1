# One decision point for the native CMake generator (dot-source it, do not execute).
#
#   . .\scripts\env\native-toolchain.ps1
#   Set-NativeToolchain                 # honours HFS_NATIVE_GENERATOR, else probes ninja
#   Clear-StaleCmakeCaches -TargetDir .\backend\target
#
# [Why this file exists] The generator used to be decided by *whoever happened to look
# first*, and the three deciders disagreed:
#
#   1. opusic-sys 0.7.5's build.rs (lines 232-235) probes "ninja --version" and, when it
#      succeeds, calls cmake.generator("Ninja"). An explicit .generator(..) call
#      **overrides** CMAKE_GENERATOR, so the answer depends on whether ninja happens to be
#      on PATH in *that* cargo process.
#   2. scripts/env/msvc-env.ps1 only set CMAKE_GENERATOR = 'NMake Makefiles' on the branch
#      that actually imports vcvars -- inside an "x64 Native Tools" prompt the early return
#      skipped it, so the variable was never set at all.
#   3. scripts/env/android-env.ps1 sets CMAKE_GENERATOR = 'Ninja' and puts the NDK's ninja
#      on PATH. Both env scripts are **dot-sourced**, so one session that built for Android
#      and then for Windows carried Ninja (and the ninja directory) into the desktop build.
#
# The result was the same machine holding NMake Makefiles and Ninja caches side by side
# under one target directory, and CMake then refusing to reconfigure either of them:
#
#   CMake Error: Error: generator : Ninja
#   Does not match the generator used previously: NMake Makefiles
#
# [The invariant this file implements] **One environment, one generator.** Every consumer
# in a build (cmake-rs for soundtouch, opusic-sys' own probe, anything added later) must
# reach the same answer, and a cache recorded under a different answer is by definition
# stale and gets deleted -- exactly the discipline backend/hifishifter-kernel/build.rs
# already applies to soundtouch (lines 683-726), lifted to the environment layer because a
# third-party crate's OUT_DIR cannot be reached from there.
#
# [Why functions and no top-level variables] This file is **dot-sourced**, so a top-level
# "$foo = ..." would land in the caller's scope and could overwrite it (msvc-env.ps1 lost
# the repo root exactly that way once). Only $env:* and function definitions may escape;
# the rule is enforced by scripts/checks/check-env-scripts.ps1.

$ErrorActionPreference = 'Stop'

function ConvertTo-NativeCmakeGenerator {
    <#
    .SYNOPSIS
    Normalises a generator spelling to exactly two values: 'Ninja' or 'NMake Makefiles'.
    .DESCRIPTION
    Accepts 'Auto' as a third, pass-through value so callers can say "no opinion" without
    inventing a sentinel. Anything else is rejected loudly: a typo silently falling back to
    a default would reproduce the very bug this file exists to prevent.
    #>
    param([string]$Value)
    if (!$Value) { return $null }
    $trimmed = $Value.Trim()
    if ($trimmed -ieq 'Auto') { return 'Auto' }
    if ($trimmed -ieq 'Ninja') { return 'Ninja' }
    if ($trimmed -ieq 'NMake' -or $trimmed -ieq 'NMake Makefiles') { return 'NMake Makefiles' }
    throw ("Unsupported CMake generator '$Value'. Use HFS_NATIVE_GENERATOR=Ninja or =NMake.")
}

function Test-NativeNinjaAvailable {
    <#
    .SYNOPSIS
    True when a ninja executable would be found by a child process (the same probe
    opusic-sys performs with Command::new("ninja")).
    #>
    return [bool](Get-Command ninja -CommandType Application -ErrorAction SilentlyContinue)
}

function Resolve-NativeCmakeGenerator {
    <#
    .SYNOPSIS
    Resolves the generator for this build: explicit choice > PATH probe > NMake.
    .DESCRIPTION
    The precedence is not arbitrary. opusic-sys overrides CMAKE_GENERATOR with its own
    probe, so an environment where ninja is on PATH **will** build opus with Ninja no matter
    what this file says. Answering 'Ninja' in that case is therefore the only self-consistent
    answer; answering 'NMake Makefiles' would just move the mismatch into the opusic-sys
    cache. When ninja is absent the probe can never fire, so NMake is the constant.
    Set-NativeToolchain additionally hides ninja from PATH for the NMake case, which keeps
    the two answers aligned even when the user asked for NMake explicitly on a machine that
    does have ninja.
    #>
    param([string]$Generator = 'Auto')
    $requested = ConvertTo-NativeCmakeGenerator $Generator
    if (!$requested -or $requested -eq 'Auto') {
        $requested = ConvertTo-NativeCmakeGenerator $env:HFS_NATIVE_GENERATOR
    }
    if ($requested -and $requested -ne 'Auto') { return $requested }
    if (Test-NativeNinjaAvailable) { return 'Ninja' }
    return 'NMake Makefiles'
}

function Hide-NativeNinjaFromPath {
    <#
    .SYNOPSIS
    Removes every PATH directory that provides a ninja executable; returns them.
    .DESCRIPTION
    [Why masking PATH instead of only setting CMAKE_GENERATOR] See Resolve-NativeCmakeGenerator:
    opusic-sys' explicit .generator("Ninja") outranks the environment variable, so the only
    way to make its probe fail is for the executable to be unreachable. The removal is
    process-local (the caller's own environment block), so it dies with the build.
    [Why it is announced] A silently mangled PATH is indistinguishable from a broken toolchain
    the next time somebody debugs a build; Set-NativeToolchain prints what it removed.
    #>
    $segments = @($env:PATH -split ';' | Where-Object { $_ })
    $keep = New-Object System.Collections.Generic.List[string]
    $hidden = New-Object System.Collections.Generic.List[string]
    foreach ($segment in $segments) {
        $found = $false
        foreach ($name in 'ninja.exe', 'ninja.cmd', 'ninja.bat', 'ninja') {
            if (Test-Path -LiteralPath (Join-Path $segment $name) -PathType Leaf) { $found = $true; break }
        }
        if ($found) { $hidden.Add($segment) } else { $keep.Add($segment) }
    }
    if ($hidden.Count -gt 0) { $env:PATH = ($keep -join ';') }
    return $hidden
}

function Set-NativeToolchain {
    <#
    .SYNOPSIS
    Pins CMAKE_GENERATOR for this build and records the answer in HFS_NATIVE_GENERATOR.
    .DESCRIPTION
    Idempotent: dot-sourcing msvc-env.ps1 twice, or calling this again with the same
    argument, changes nothing (hiding an already-hidden ninja removes nothing).

    [Why HFS_NATIVE_GENERATOR is both input and output] The value has to survive from the env
    script into every child cargo process, and it has to stay readable afterwards for
    diagnostics -- $env:* is the only channel a dot-sourced script may use. It is also the
    documented override (HFS_NATIVE_GENERATOR=Ninja|NMake), so honouring it on the next call
    is what makes "explicit choice > probe" work across the two env scripts. The consequence
    -- the last explicit choice sticks for the session -- is exactly the invariant: one
    environment, one generator.
    #>
    param([ValidateSet('Auto', 'Ninja', 'NMake')][string]$Generator = 'Auto')
    $resolved = Resolve-NativeCmakeGenerator -Generator $Generator
    if ($resolved -eq 'Ninja') {
        if (!(Test-NativeNinjaAvailable)) {
            throw ('HFS_NATIVE_GENERATOR asks for the Ninja CMake generator but no ninja executable is on PATH. ' +
                'Install ninja and put it on PATH, or set HFS_NATIVE_GENERATOR=NMake.')
        }
        # Nothing to mask: opusic-sys' probe reaches the same answer on its own.
        $env:CMAKE_GENERATOR = 'Ninja'
        $env:HFS_NATIVE_GENERATOR = 'Ninja'
        Write-Host '[toolchain] cmake generator: Ninja (ninja found on PATH)'
        return
    }
    $hidden = @(Hide-NativeNinjaFromPath)
    $env:CMAKE_GENERATOR = 'NMake Makefiles'
    $env:HFS_NATIVE_GENERATOR = 'NMake'
    if ($hidden.Count -gt 0) {
        Write-Host ('[toolchain] ninja hidden from PATH: ' + ($hidden -join '; '))
    }
    Write-Host '[toolchain] cmake generator: NMake Makefiles (ninja not reachable)'
}

function Clear-StaleCmakeCaches {
    <#
    .SYNOPSIS
    Deletes CMake build trees whose recorded generator differs from the wanted one.
    .DESCRIPTION
    A CMake cache pins its generator; CMake then refuses to configure that directory with
    any other generator ("does not match the generator used previously"). The directory is
    unrecoverable by any other means, so "generator changed => this cache is void" is a fact,
    not a heuristic.

    Scope is deliberately narrow: only target/**/build/<crate>/out/<cmake build dir> is
    considered, and only that innermost directory is deleted. out/ itself is never touched --
    several crates keep their installed lib/ + include/ there, and deleting it would turn a
    20-second reconfigure into a multi-minute rebuild.

    @param TargetDir cargo's target directory (callers resolve CARGO_TARGET_DIR the way cargo
           does; see Get-NativeCargoTargetDir).
    @param Generator wanted generator; defaults to the one this environment resolved.
    @returns the number of dropped build trees.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$TargetDir,
        [string]$Generator = ''
    )
    $wanted = if ($Generator) { ConvertTo-NativeCmakeGenerator $Generator } else { Resolve-NativeCmakeGenerator }
    if (!$wanted -or $wanted -eq 'Auto') { $wanted = Resolve-NativeCmakeGenerator }
    if (!(Test-Path -LiteralPath $TargetDir)) { return 0 }

    # target/[<triple>/]<profile>/build -- bounded depth so this stays a cheap stat walk
    # instead of a full-tree recursion over a target directory that can hold 100k files.
    $buildRoots = @(Get-ChildItem -LiteralPath $TargetDir -Directory -Recurse -Depth 3 -Filter 'build' -ErrorAction SilentlyContinue |
        Where-Object { $_.Parent -and ($_.Parent.Name -eq 'debug' -or $_.Parent.Name -eq 'release') })
    $dropped = 0
    foreach ($buildRoot in $buildRoots) {
        foreach ($crateDir in @(Get-ChildItem -LiteralPath $buildRoot.FullName -Directory -ErrorAction SilentlyContinue)) {
            $outDir = Join-Path $crateDir.FullName 'out'
            if (!(Test-Path -LiteralPath $outDir -PathType Container)) { continue }
            foreach ($buildDir in @(Get-ChildItem -LiteralPath $outDir -Directory -ErrorAction SilentlyContinue)) {
                $cache = Join-Path $buildDir.FullName 'CMakeCache.txt'
                if (!(Test-Path -LiteralPath $cache -PathType Leaf)) { continue }
                # [Why an explicit reader with try/finally] [IO.File]::ReadLines is lazy and its
                # enumerator is not disposed when the loop breaks out early, so the handle stayed
                # open and the Remove-Item below failed with "used by another process" -- silently,
                # because of -ErrorAction SilentlyContinue. The sweep then reported success while
                # the stale tree survived, which is worse than not sweeping at all.
                $recorded = $null
                $reader = [IO.File]::OpenText($cache)
                try {
                    while ($null -ne ($line = $reader.ReadLine())) {
                        if ($line.StartsWith('CMAKE_GENERATOR:INTERNAL=', [StringComparison]::Ordinal)) {
                            $recorded = $line.Substring(25).Trim()
                            break
                        }
                    }
                }
                finally {
                    $reader.Dispose()
                }
                if (!$recorded -or $recorded -eq $wanted) { continue }
                Write-Host ('[toolchain] dropped stale cmake cache ({0}) -- recorded {1}, wanted {2}' -f $buildDir.FullName, $recorded, $wanted)
                # [Why rename first, then delete] Remove-Item on a tree that another process holds
                # open deletes *some* of it and then gives up -- leaving a directory that still has
                # Makefile/Makefile2 and no CMakeCache.txt. CMake then refuses to regenerate it
                # (Makefile present) and nmake fails on a missing CMakeFiles/*/build.make:
                #   NMAKE : fatal error U1073: don't know how to make 'CMakeFiles\soundstretch.dir\build.make'
                # After that the machine cannot build at all, and nothing in the message mentions a
                # cache. A rename either succeeds completely or leaves the tree untouched, so the
                # original path is always either a *valid* tree or gone.
                $tombstone = $buildDir.FullName + '.stale-' + [Guid]::NewGuid().ToString('N')
                $moved = $false
                try { Move-Item -LiteralPath $buildDir.FullName -Destination $tombstone -ErrorAction Stop; $moved = $true }
                catch { Write-Host ('[toolchain] could not drop {0}: {1} (the next configure will retry)' -f $buildDir.FullName, $_.Exception.Message) -ForegroundColor Yellow }
                if ($moved) {
                    Remove-Item -LiteralPath $tombstone -Recurse -Force -ErrorAction SilentlyContinue
                    $dropped += 1
                }
            }
        }
    }
    return $dropped
}

function Get-NativeCargoTargetDir {
    <#
    .SYNOPSIS
    Resolves cargo's target directory the way cargo does, given the backend directory.
    .DESCRIPTION
    [Why one implementation] scripts/clap/windows/build.ps1 and scripts/build/build.ps1 each
    carried their own copy of this rule. A relative CARGO_TARGET_DIR resolves against the
    directory cargo runs in (backend/), not against the repository root, and getting that
    wrong makes a script look at -- or worse, package -- a previous build's artefact.
    #>
    param([Parameter(Mandatory = $true)][string]$BackendDir)
    if ($env:CARGO_TARGET_DIR) {
        if ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
            return [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
        }
        return [IO.Path]::GetFullPath((Join-Path $BackendDir $env:CARGO_TARGET_DIR))
    }
    return (Join-Path $BackendDir 'target')
}
