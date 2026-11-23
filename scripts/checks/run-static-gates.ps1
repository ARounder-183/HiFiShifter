# Static gates: the set of cross-file conventions that compile nothing, touch no network,
# and finish in tens of seconds.
#
# Breaking any one of them raises no error; it only shows up for users as "FFI crashed the
# host", "the log is empty", "controls do not respond to clicks", or "the version displays
# wrong". They must run on every PR and must have exactly one entry point — these three
# used to live in three separate workflows, and one of them got missed before (see the
# history comments in vst3.yml).
#
#   .\scripts\checks\run-static-gates.ps1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

function Invoke-Gate([string]$name, [scriptblock]$body) {
    Write-Host "== $name" -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE) { throw "$name failed (exit $LASTEXITCODE)" }
}

# Every extern "system" entry point the host calls must carry its own panic guard
# (a panic crossing the FFI boundary = abort = DAW crash). Discipline: docs/thread-model.md.
Invoke-Gate 'FFI panic-guard gate' { python scripts/checks/check-ffi-guards.py }

# Rate-limited logging allows only Warn / Error. Info/Debug under a limiter becomes a silent
# black hole at the default log level: the code looks like it emits diagnostics but prints none.
Invoke-Gate 'Log level gate' { python scripts/checks/check-log-levels.py }

# Four-tier log verbosity: bans regressions of legacy env vars / production console.* / stale default-level comments.
Invoke-Gate 'Log verbosity gate' { python scripts/checks/check-log-verbosity.py }

# The frontend invoke.ts switch is the full set of method names the frontend can send. If any
# one of them is dispatched by neither backend, the symptom is "the control is clickable but
# nothing happens" — silent failure, with no compile error.
Invoke-Gate 'Command gating gate' { python scripts/checks/check-command-gating.py }

# Every system Save As / Open dialog must carry the type filters from the word list. A missing
# one does not error: the type dropdown goes empty and the user can save a .zip diagnostic
# bundle as .txt (has happened in practice).
Invoke-Gate 'Dialog filter gate' { python scripts/checks/check-dialog-filters.py }

# Version number, installer text, model manifest, license manifest, installer form: all
# conventions where "one missing spot errors nothing".
Invoke-Gate 'Product consistency gate' { & (Join-Path $PSScriptRoot 'check-product-consistency.ps1') }

# Dot-sourced env scripts may only touch $env:*. Real incident: a `foreach ($root in ...)`
# in msvc-env.ps1 overwrote the build script's `$root` (repo root) with the Visual Studio
# install directory, and the build failed at "backend\src-tauri\resources\models\models.json
# not found".
Invoke-Gate 'Env script scope gate' { & (Join-Path $PSScriptRoot 'check-env-scripts.ps1') }

# Truncated UTF-8 bytes inside source comments trip no compiler, but they turn the (Chinese)
# comments into mojibake and then ride along in every diff. Third-party trees keep their
# upstream's original encoding and are out of scope for this gate.
Invoke-Gate 'Source UTF-8 gate' { python scripts/checks/check-utf8.py }

# SoundTouch build generator and cache self-healing.
Invoke-Gate 'SoundTouch generator gate' { python scripts/checks/check-soundtouch-generator.py }

# Test names in the skip manifest must still resolve against the Rust sources.
Invoke-Gate 'Test skip gate' { python scripts/checks/check-test-skips.py }

# The mobile adaptation's central rule: cfg(target_os) is allowed in exactly one
# directory (backend/src-tauri/src/platform/). A stray one raises no error on the
# developer's machine; it only shows up as an unrelated failure when somebody
# builds for a target they are not on.
Invoke-Gate 'Platform boundary gate' { python scripts/checks/check-platform-boundary.py }

# And its frontend counterpart: the application form has one answer (appForm).
# A second answer (window width / user agent / pointer media query) is right on
# the developer's device and wrong on somebody else's.
#
# [Why there is no separate "mobile bundle size" gate] frontend/scripts/check-bundle-size.mjs
# already pins the first-paint chunk and the CSS total, and the mobile budget is
# the same budget -- a second file would be a second source of truth for one
# number. Run it with "npm run check:bundle" after a build.
Invoke-Gate 'Mobile form gate' { python scripts/checks/check-mobile-form-gate.py }

Write-Host 'All static gates passed.' -ForegroundColor Green
