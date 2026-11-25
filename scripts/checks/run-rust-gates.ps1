# The single entry point for the Rust gates: fmt / clippy / test, plus the one skip manifest.
#
# [Why this script exists] The same set of cargo commands used to be written out three
# times across workflows: build.yml, vst3.yml, vst3-unix.yml. Their arguments
# (--locked / --offline / --jobs / --skip / --test-threads) all differed, and nothing
# guaranteed their equivalence — so a test could be green on one pipeline and red on
# another, with no one able to say which one was the criterion. Workflows now call only
# this script: **the criterion lives in exactly one place**, and -PrintCommand prints the
# command line it resolves to.
#
#   .\scripts\checks\run-rust-gates.ps1                     # pr: fmt + clippy + test
#   .\scripts\checks\run-rust-gates.ps1 -Only fmt           # formatting only
#   .\scripts\checks\run-rust-gates.ps1 -Profile full        # adds no-default-features / concurrent tests
#   .\scripts\checks\run-rust-gates.ps1 -PrintCommand        # print command lines only (local vs CI diff)
[CmdletBinding()]
param(
    [ValidateSet('pr', 'full')][string]$Profile = 'pr',
    [ValidateSet('all', 'fmt', 'clippy', 'test')][string]$Only = 'all',
    # Empty = the whole workspace; a package name runs only that crate (for offline verification).
    [string]$Package = '',
    [ValidateSet('serial', 'parallel')][string]$TestMode = 'serial',
    # Explicit platform override (default is derived from the runner): the skip manifest has [unix]-only entries.
    [ValidateSet('', 'windows', 'unix')][string]$Platform = '',
    [switch]$PrintCommand
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if (!$Platform) {
    if ($env:RUNNER_OS -eq 'Windows') { $Platform = 'windows' }
    elseif ($env:RUNNER_OS) { $Platform = 'unix' }
    elseif ([IO.Path]::DirectorySeparatorChar -eq '\') { $Platform = 'windows' }
    else { $Platform = 'unix' }
}

# ── Skip manifest ─────────────────────────────────────────────────────────
# The single source is scripts/checks/test-skips.txt: no --skip appears in workflows anymore.
function Get-Skips {
    $file = Join-Path $PSScriptRoot 'test-skips.txt'
    if (!(Test-Path $file)) { throw "Missing skip manifest: $file" }
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($line in Get-Content -LiteralPath $file -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (!$trimmed) { continue }
        if ($trimmed.StartsWith('#')) { continue }
        $entryPlatform = ''
        if ($trimmed.StartsWith('[unix]')) { $entryPlatform = 'unix'; $trimmed = $trimmed.Substring(6).Trim() }
        elseif ($trimmed.StartsWith('[windows]')) { $entryPlatform = 'windows'; $trimmed = $trimmed.Substring(9).Trim() }
        if ($entryPlatform -and $entryPlatform -ne $Platform) { continue }
        # Entries look like "<test name> — <tracking location> — <reason>": take the first field only.
        $name = ($trimmed -split ' — ')[0].Trim()
        if ($name) { $names.Add($name) }
    }
    return $names
}

$manifest = 'backend/Cargo.toml'
$skipArgs = @()
foreach ($name in Get-Skips) { $skipArgs += @('--skip', $name) }

# $Threads overrides TestMode's thread count: the concurrent rerun must reuse the same command with only this one parameter changed.
function Get-CargoCommand([string]$kind, [int]$Threads = 0) {
    # [Why Write-Output per line instead of return @(...)] PowerShell's return packs the
    # array into the output stream as a single object, so the caller's foreach receives
    # "one array" (printed as a space-joined single line). Emitting line by line lets the
    # pipeline naturally expand into multiple commands.
    # [Why every cargo command first sets the environment] ara2-bridge's build.rs panics
    # outright when ARA_SDK_DIR / ARA_VST3_SDK_DIR are missing. The only source of these two
    # variables is scripts/env/sdk-env.ps1; without setting them here, the failure happens
    # "mid-compile, thrown by the build script" and looks like broken source code. That is
    # exactly how CI's old plugin unit-test job went red on its first real run.
    $msvc = Join-Path $root 'scripts\env\msvc-env.ps1'
    $sdkEnv = Join-Path $root 'scripts/env/sdk-env.ps1'
    $sdkRoot = Join-Path $root 'third_party/sdk'
    $backendDir = Join-Path $root 'backend'
    $prefix = ''
    # [Why the platform check] msvc-env.ps1 reads ${env:ProgramFiles(x86)}, which is null on
    # Linux, and Join-Path then throws "Cannot bind argument to parameter Path because it is
    # null" on the spot. PowerShell on Unix also treats \ as a separator, so Test-Path would
    # hit anyway, and the fmt tier (Linux, compiles nothing) would be killed outright by that
    # dot-import.
    $onWindows = [IO.Path]::DirectorySeparatorChar -eq '\'
    # [Why the cache sweep rides along with the environment import] This script is the single
    # entry point for every cargo command in CI, and the native crates (opusic-sys, soundtouch)
    # configure CMake on the first run. A cache recorded under another generator would fail that
    # build script with "generator ... does not match the generator used previously" -- a message
    # that names neither this script nor the generator decision. The sweep makes the environment
    # self-healing; see scripts/env/native-toolchain.ps1.
    if ($onWindows -and (Test-Path $msvc)) {
        $prefix += ". '$msvc';"
        $prefix += " $null = Clear-StaleCmakeCaches -TargetDir (Get-NativeCargoTargetDir -BackendDir '$backendDir');"
    }
    # [Why also check whether the SDK directory exists] The fmt tier (Linux) installs no SDK
    # and compiles nothing; Set-SdkEnvironment would then push two empty paths into the env
    # vars, turning a pure formatting check into "Cannot bind argument to parameter Path
    # because it is null". When the SDK is not ready, leave the env unset — clippy / test,
    # which need it, get a more accurate error from the build script.
    if ((Test-Path $sdkEnv) -and (Test-Path $sdkRoot)) { $prefix += ". '$sdkEnv'; Set-SdkEnvironment;" }
    switch ($kind) {
        'fmt' {
            # hfs-gui (the VST3 standalone render process) lives outside the backend workspace,
            # so --all does not cover it; it accumulated rustfmt drift that way, and "it
            # compiles" does not reveal that kind of decay.
            Write-Output "$prefix cargo fmt --manifest-path $manifest --all --check"
            Write-Output "$prefix cargo fmt --manifest-path backend/hfs-gui/Cargo.toml --all --check"
        }
        'clippy' {
            $target = if ($Package) { "-p $Package" } else { '--workspace' }
            Write-Output "$prefix cargo clippy --locked --manifest-path $manifest $target --all-targets -- -D warnings"
            if ($Profile -eq 'full' -and !$Package) {
                # The pure-Rust shape (no onnx / vslib) has never been compiled: when a lighter
                # shape or a cfg-gate reshuffle comes along, breakage would surface first at
                # users'. Minutes of cost, pinned separately.
                Write-Output "$prefix cargo clippy --locked --manifest-path $manifest -p hifishifter-kernel --no-default-features --all-targets -- -D warnings"
            }
        }
        'test' {
            $threads = if ($Threads -gt 0) { $Threads } elseif ($TestMode -eq 'parallel') { 4 } else { 1 }
            # [Why plugin tests are split into one process per module] The plugin's lib tests
            # are **800+ cases in one process**, and process-level global state keeps
            # accumulating along the way (render cache, pitch cache, host marshal queue, model
            # sessions). On a 4-core runner, reaching the host::reaper::host_edit section
            # occasionally deadlocked: the log stopped right after
            # host_edit::tests::a_poisoned_editor_timeline_refuses_host_edits and hung until the
            # 60-minute cap (the same code ran all-green in 168 seconds on a local 16-core box).
            # One process per module leaves only that module's inherited state behind, and a
            # hang immediately shows which module is stuck.
            # [Why vst3:: is not in the list] The VST3 shell and its black-box tests moved
            # wholesale into `hifishifter-vst3`;
            # the shell's tests run as "one whole crate per process" (the same order of
            # magnitude as a single module).
            $pluginChunks = @('ara::', 'capture::', 'editor::', 'host::', 'render::')
            $skips = if ($skipArgs.Count -gt 0) { ' ' + ($skipArgs -join ' ') } else { '' }
            $pluginRun = (!$Package -or $Package -eq 'hifishifter-plugin')

            # [Why build once before testing] tests/vst3_exports.rs dlopens the produced cdylib
            # to assert the export table, while `cargo test` only builds the test rlib and does
            # not produce a .dll — on a clean checkout that test cannot find the file (its own
            # error message says "run cargo build first").
            if ($pluginRun -and $threads -le 1) {
                # Both shells' export-table tests dlopen the produced cdylib
                # (hifishifter_vst3.dll / hifishifter_clap.dll), while cargo test only builds
                # the test rlib — the same trap as vst3_exports / clap_exports, so build once first.
                Write-Output "$prefix cargo build --locked --manifest-path $manifest -p hifishifter-vst3"
                Write-Output "$prefix cargo build --locked --manifest-path $manifest -p hifishifter-clap"
            }

            if ($pluginRun -and $threads -le 1) {
                # [Why CLAP is not inside --workspace] Cargo unions features across the build
                # graph: src-tauri uses kernel's default features (including vslib), so a
                # --workspace run shared with CLAP compiles kernel with vslib, and
                # #[link(name="vslib_x64")] then leaks into hifishifter_clap.dll's import
                # table — the clap_exports test "a single-file plugin must not import
                # non-system DLLs" goes red, even though CLAP's own Cargo.toml says
                # default-features = false. Resolved with a lone -p, the dependency graph has
                # no src-tauri and the artifact is the release shape. The criterion is
                # unchanged; only the checked target becomes the one shaped like the artifact.
                # [Why the exclude list is the two shells] The shells' build.rs still need
                # SDK/build identity (the vst3 shell needs the ARA/VST3 SDK), and Cargo's
                # feature union would pull src-tauri's vslib into CLAP's import-table
                # criterion — the same treatment hifishifter-plugin got before, one -p each.
                # The shared layer (plugin) needs zero SDKs and has no cdylib, so it returns to
                # the full workspace run.
                $shellExclude = ' --exclude hifishifter-vst3 --exclude hifishifter-clap'
                if (!$Package) {
                    Write-Output "$prefix cargo test --locked --manifest-path $manifest --workspace $shellExclude -- --test-threads=1$skips"
                    Write-Output "$prefix cargo test --locked --manifest-path $manifest -p hifishifter-vst3 -- --test-threads=1$skips"
                    Write-Output "$prefix cargo test --locked --manifest-path $manifest -p hifishifter-clap -- --test-threads=1$skips"
                }
                # [Why --no-run once first] When splitting into per-module processes, every
                # cargo test re-evaluates the build scripts; on macOS arm64 the second chunk was
                # measured stuck on cmake for ten minutes. Compiling all test targets once up
                # front turns every later command into just launching ready-made binaries.
                Write-Output "$prefix cargo test --locked --manifest-path $manifest -p hifishifter-plugin --no-run"
                foreach ($chunk in $pluginChunks) {
                    Write-Output "$prefix cargo test --locked --manifest-path $manifest -p hifishifter-plugin -- $chunk --test-threads=1$skips"
                }
                # Fallback: cases not under any module prefix (tests:: at the crate root) and
                # integration-test targets.
                $chunkSkips = ($pluginChunks | ForEach-Object { "--skip $_" }) -join ' '
                Write-Output "$prefix cargo test --locked --manifest-path $manifest -p hifishifter-plugin -- --test-threads=1 $chunkSkips$skips"
            }
            else {
                $target = if ($Package) { "-p $Package" } else { '--workspace' }
                $exclude = ''
                # Concurrency-tier-specific skips: the App's measure_peak_alloc counter is
                # **process-level**, so allocations by other tests in the same process pollute
                # the peak. That test's own comment says it "must run serially"; the serial tier
                # already runs it in full, so only the concurrent rerun skips it; kernel's
                # pitch_memory sibling is already #[ignore] and needs no extra skip.
                $concurrentSkips = ''
                if (!$Package -and $threads -gt 1) {
                    # Same as above: the concurrent rerun must not union src-tauri's vslib into CLAP's import table either.
                    $exclude = ' --exclude hifishifter-vst3 --exclude hifishifter-clap'
                    $concurrentSkips = ' --skip mixing_a_faded_clip_on_the_audio_thread_does_not_allocate'
                }
                Write-Output "$prefix cargo test --locked --manifest-path $manifest $target$exclude -- --test-threads=$threads$skips$concurrentSkips"
            }
        }
    }
}

$kinds = switch ($Only) {
    'all' { @('fmt', 'clippy', 'test') }
    default { @($Only) }
}

if ($PrintCommand) {
    foreach ($kind in $kinds) {
        # [Why @(...)] PowerShell "downgrades" a single-element array into a string: a bare
        # foreach would print one line and [0] would take the first character — neither
        # errors, they just make the self-verification output look like "a few gates are
        # missing".
        foreach ($command in @(Get-CargoCommand $kind)) { Write-Output $command }
    }
    if ($Profile -eq 'full' -and ($Only -eq 'all' -or $Only -eq 'test') -and !$Package) {
        # The full tier adds a concurrent rerun: --test-threads=1 masks races, and the kernel's
        # most fragile spots (recompute vs auto-apply version races, snapshot publication vs
        # reader reclamation) are all concurrency-sensitive.
        Write-Output (Get-CargoCommand 'test' -Threads 4)
    }
    exit 0
}

foreach ($kind in $kinds) {
    foreach ($command in @(Get-CargoCommand $kind)) {
        Write-Host "== $kind" -ForegroundColor Cyan
        Write-Host "   $command" -ForegroundColor DarkGray
        Invoke-Expression $command
        if ($LASTEXITCODE) { throw "cargo $kind failed (exit $LASTEXITCODE)" }
    }
}

if ($Profile -eq 'full' -and ($Only -eq 'all' -or $Only -eq 'test') -and !$Package) {
    Write-Host '== test (concurrent)' -ForegroundColor Cyan
    $concurrent = Get-CargoCommand 'test' -Threads 4
    Write-Host "   $concurrent" -ForegroundColor DarkGray
    Invoke-Expression $concurrent
    if ($LASTEXITCODE) { throw "concurrent cargo test failed (exit $LASTEXITCODE)" }
}

Write-Host 'All Rust gates passed.' -ForegroundColor Green
