#!/usr/bin/env python3
"""Every test name in the skip list must be resolvable in the current Rust sources.

`cargo test --skip` is a silent no-op for names that do not exist, so the list
keeps looking like a valid exemption after a test is deleted or renamed. A
full-text search of the sources is the cheapest possible self-check here.

Exit codes: 0 = all names resolve, 1 = dangling entries exist.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKIPS = ROOT / 'scripts' / 'checks' / 'test-skips.txt'
BACKEND = ROOT / 'backend'


def main() -> int:
    names: list[str] = []
    for line in SKIPS.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = line.split(' — ')[0].strip()
        if name.startswith('[unix]') or name.startswith('[windows]'):
            name = name.split(']', 1)[1].strip()
        names.append(name)
    corpus = '\n'.join(
        path.read_text(encoding='utf-8', errors='replace')
        for path in BACKEND.rglob('*.rs')
        if not any(part in {'target', 'third-party', 'third_party', '.tmp'} for part in path.parts)
    )
    missing = [name for name in names if name not in corpus]
    if missing:
        print('test-skip gate FAILED: unresolved test names')
        for name in missing:
            print('  - ' + name)
        return 1
    print(f'test-skip gate OK: {len(names)} entries, all resolvable')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
