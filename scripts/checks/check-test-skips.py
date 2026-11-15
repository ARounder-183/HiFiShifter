#!/usr/bin/env python3
"""跳过清单里的每个测试名都必须在当前 Rust 源码中可解析。

cargo test --skip 对不存在的名字是静默 no-op；清单因此会在测试被删除或重命名后
继续看起来像一条有效豁免。这里用源码全文检索做最低成本的自校验。

退出码：0 = 全部可解析，1 = 存在悬空条目。
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
