#!/usr/bin/env python3
"""Throttled logs may use Warn / Error / Debug; Info is forbidden.

[Why Debug is now allowed] Users can pick one of four verbosity levels under
"Help -> Log verbosity" (None / Low / Medium / High, default Low). Per-callback,
per-block trace points are semantically Debug and are only emitted at the
"High" level; but they must still be throttled, otherwise a single debugging
session can burn through the entire 8 MiB x 5 rotation budget. The old rule
"throttling only allows Warn/Error" was built on "the default level is Warn" --
that statement no longer holds; the four levels are user-selected.

[Why Info + throttling is still forbidden] Info is session narrative (opening a
project, starting a render, changing the verbosity level) -- at most a few
dozen entries per session. Throttling it would silently swallow key
breadcrumbs inside the window.

Rules:
  * Fault signals -> Warn / Error + throttling;
  * High-frequency tracing -> Debug + throttling (only emitted at "High");
  * Session narrative -> call log::info! directly, without throttling.

Exit code: 0 = compliant, 1 = violations found.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parents[2] / 'backend'
SKIP_DIRS = {'target', 'third_party', 'node_modules'}

CALL = re.compile(r'emit_limited(?:_key)?\s*\(')
LEVEL = re.compile(r'log::Level::(\w+)')
ALLOWED = {'Warn', 'Error', 'Debug'}


def sources() -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for path in ROOT.rglob('*.rs'):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        found.append(path)
    return found


def main() -> int:
    violations: list[str] = []
    for path in sources():
        text = path.read_text(encoding='utf-8', errors='replace')
        for match in CALL.finditer(text):
            window = text[match.start() : match.start() + 260]
            level = LEVEL.search(window)
            if level is None or level.group(1) in ALLOWED:
                continue
            line = text.count('\n', 0, match.start()) + 1
            relative = path.relative_to(ROOT.parent).as_posix()
            violations.append(f'{relative}:{line}: throttled log at Level::{level.group(1)}')

    if violations:
        print('throttled logs only allow Warn / Error / Debug (see the header of this script):')
        for violation in violations:
            print('  ' + violation)
        print()
        print('Info is session narrative: call log::info! directly and do not throttle it.')
        return 1

    print(f'log levels ok ({len(sources())} files scanned)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
