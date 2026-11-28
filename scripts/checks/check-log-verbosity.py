#!/usr/bin/env python3
"""Source gate for log verbosity: anti-regression check for the four-level policy.

[Why this is needed] Once the four-level setting shipped, the easiest way to
regress is "an independent switch growing out of the code again": somewhere
re-reading HIFISHIFTER_DEBUG_*, somewhere calling console.* directly again, or
a comment re-stating "because the default level is Warn, use Warn". None of
these fail the build; they only mean the user picks "None" but still gets
output, or picks "Low" but still gets flooded.

Checks (each individually locatable):
  1. Production Rust source must not contain legacy debug env variables;
  2. Production frontend source must not call console.* directly (the only
     escape hatch is services/frontendLog.ts);
  3. Comments must not re-state "the default level is Warn, so use Warn";
  4. The shipped default must be Low, and hosts must not call set_max_level
     inside their own modules;
  5. The log_settings.json filename constant must appear exactly once;
  6. Documentation defaults must agree (no more "release default info").

Excluded: docs/, scripts/, third_party/, target/, cfg(test) blocks, build.rs.
Exit code: 0 = compliant, 1 = violations found.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / 'backend'
FRONTEND_SRC = ROOT / 'frontend' / 'src'
SKIP_PARTS = {'target', 'node_modules', 'third_party', '.git', 'dist'}

LEGACY_ENV = re.compile(
    r'HIFISHIFTER_DEBUG_COMMANDS|HIFISHIFTER_DEBUG_FORMANT|'
    r'HIFISHIFTER_RENDER_CACHE_LOG|HIFISHIFTER_PLUGIN_PCM_PROBE|'
    r'HIFISHIFTER_PLUGIN_TRANSPORT_PROBE|HIFISHIFTER_BUFFER_AUDIT|'
    r'HIFISHIFTER_CLICK_AUDIT|HIFISHIFTER_PLUGIN_PROBE_ANALYSIS'
)
CONSOLE = re.compile(r'console\.(log|debug|info|warn|error|trace|assert|dir|table|group|time)\s*\(')
# The Chinese alternatives match existing Chinese source comments and must stay;
# the English alternatives catch the same stale claims written in English.
STALE_COMMENT = re.compile(
    r'默认日志级别是 Warn|因为默认级别|必须是 WARN 级|'
    r'用 WARN 级|release 默认 Warn|默认级别是 Warn|'
    r'[Dd]efault log level is Warn|because the default level|'
    r'since the default level|must be (at )?WARN level|'
    r'use (the )?WARN level|release defaults? to Warn|'
    r'release default Warn|[Dd]efault level is Warn'
)
DEFAULT_LOW = re.compile(r'pub const DEFAULT_VERBOSITY:[^=]*=\s*LogVerbosity::Low')


def rust_sources() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in BACKEND.rglob('*.rs'):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name == 'build.rs':
            continue
        out.append(path)
    return out


def frontend_sources() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in FRONTEND_SRC.rglob('*'):
        if path.suffix not in {'.ts', '.tsx'} or not path.is_file():
            continue
        if '.test.' in path.name or '.bench.' in path.name:
            continue
        out.append(path)
    return out


def strip_test_blocks(text: str) -> str:
    """Replace cfg(test) mod tests blocks with blank lines so probes inside
    tests do not trigger false positives."""
    out: list[str] = []
    depth = 0
    for line in text.splitlines(keepends=True):
        if depth == 0 and re.match(r'^\s*#\[cfg\(test\)\]', line):
            depth = 1
            out.append('\n')
            continue
        if depth:
            depth += line.count('{') - line.count('}')
            out.append('\n')
            continue
        out.append(line)
    return ''.join(out)


def main() -> int:
    violations: list[str] = []

    rust = rust_sources()
    for path in rust:
        text = strip_test_blocks(path.read_text(encoding='utf-8', errors='replace'))
        rel = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            if LEGACY_ENV.search(line):
                violations.append(f'{rel}:{number}: legacy debug env switch: {line.strip()[:110]}')
            if STALE_COMMENT.search(line):
                violations.append(f'{rel}:{number}: stale default-level comment: {line.strip()[:110]}')
        if (
            'set_max_level(' in text
            and 'log_policy.rs' not in rel
            and not rel.endswith(('_tests.rs', '/tests.rs'))
        ):
            violations.append(f'{rel}: set_max_level must only be called by log_policy::apply')

    policy = (BACKEND / 'hifishifter-kernel' / 'src' / 'log_policy.rs').read_text(encoding='utf-8')
    if not DEFAULT_LOW.search(policy):
        violations.append('log_policy.rs: DEFAULT_VERBOSITY must be Low')

    settings = (BACKEND / 'hifishifter-kernel' / 'src' / 'log_settings.rs').read_text(encoding='utf-8')
    if len(re.findall(r'"log_settings[.]json"', settings)) != 1:
        violations.append('log_settings.rs: log_settings.json filename constant must appear once')

    for path in frontend_sources():
        text = path.read_text(encoding='utf-8', errors='replace')
        rel = path.relative_to(ROOT).as_posix()
        if rel.endswith('services/frontendLog.ts'):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if CONSOLE.search(line):
                violations.append(f'{rel}:{number}: direct console call (use frontendLog): {line.strip()[:110]}')
            if 'hifishifter.debug' in line:
                violations.append(f'{rel}:{number}: standalone localStorage debug switch: {line.strip()[:110]}')

    for doc in [ROOT / 'README.md', *sorted((ROOT / 'docs' / 'i18n').glob('USERMANUAL*.md'))]:
        if not doc.exists():
            continue
        text = doc.read_text(encoding='utf-8', errors='replace')
        if re.search(r'release (版|version|build|builds)?.{0,24}(默认|default).{0,16}info', text, re.I):
            violations.append(f'{doc.relative_to(ROOT).as_posix()}: stale "release default info" text')

    # The Help menu's "Copy diagnostics summary" was removed entirely at the
    # user's request; the submenu keeps only the four levels, with no title
    # Label / "Current: Low" / override note / health status / separator.
    # This kind of "dead copy resurrection" does not fail the build, but it
    # would flatten the menu back into low information density.
    removed_identifiers = (
        'menu_copy_diagnostics_summary',
        'menu_copy_diagnostics_fallback',
        'menu_copy_diagnostics_failed_detail',
        'menu_copy_diagnostics_retry',
        'menu_log_verbosity_current',
        'menu_log_current',
        'menu_log_overridden',
        'menu_log_unavailable',
        'handleCopyDiagnosticsSummary',
        'diagnosticsSummary',
    )
    for path in frontend_sources():
        text = path.read_text(encoding='utf-8')
        for number, line in enumerate(text.splitlines(), 1):
            for identifier in removed_identifiers:
                if identifier in line:
                    violations.append(
                        f'{path.relative_to(ROOT).as_posix()}:{number}: removed log UI identifier: {identifier}'
                    )
    # The submenu is menu-item *data* now (useLogVerbosityMenu.ts returns
    # AppMenuItemSpec[] and the shared menu surface renders it), so it can no longer
    # grow a title row, a "Current: Low" row or a separator by rendering a Radix
    # surface of its own -- which is exactly what the density rule forbids. Assert
    # the two ways that could come back: rendering a menu surface here, and a status
    # string; plus the semantic that must survive the migration: four mutually
    # exclusive radio items, never checkboxes.
    submenu = FRONTEND_SRC / 'components' / 'layout' / 'useLogVerbosityMenu.ts'
    submenu_text = submenu.read_text(encoding='utf-8')
    submenu_rel = submenu.relative_to(ROOT).as_posix()
    for forbidden in ('<DropdownMenu.', 'statusText'):
        if forbidden in submenu_text:
            violations.append(f'{submenu_rel}: low-density submenu element: {forbidden}')
    if 'radio: true' not in submenu_text:
        violations.append(f'{submenu_rel}: the four levels must stay mutually exclusive radio items')

    if violations:
        print('log verbosity gate failed:')
        for item in violations[:120]:
            print('  ' + item)
        print(f'\n{len(violations)} violation(s) in total. See the header of this script for details.')
        return 1
    print(f'log verbosity ok ({len(rust)} Rust, {len(frontend_sources())} frontend files scanned)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
