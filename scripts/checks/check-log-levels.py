#!/usr/bin/env python3
"""限流日志允许 Warn / Error / Debug，禁止 Info。

【为什么要放宽到 Debug】用户现在可以在「帮助 → 日志详细程度」里选四档
（无 / 低 / 中 / 高，默认低）。逐回调、逐块的跟踪点按语义归 Debug，
只在「高」档输出；但它们仍必须限流，否则一次排障会话就能把 8 MiB × 5
的轮转额度刷满。过去"限流只允许 Warn/Error"的规则建立在
"默认级别是 Warn"之上 —— 那句话已经不成立，四档由用户选择。

【为什么仍然禁止 Info + 限流】Info 是会话叙事（打开工程、开始渲染、档位
变更），一会话最多几十条，套上限流只会在窗口内静默吞掉关键面包屑。

规则：
  * 故障信号 → Warn / Error + 限流；
  * 高频跟踪 → Debug + 限流（「高」档才输出）；
  * 会话叙事 → 直接 log::info!，不限流。

退出码：0 = 合规，1 = 发现违规。
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
        print('限流日志只允许 Warn / Error / Debug（见本脚本头部）：')
        for violation in violations:
            print('  ' + violation)
        print()
        print('Info 是会话叙事，请直接用 log::info! 且不要限流。')
        return 1

    print(f'log levels ok ({len(sources())} files scanned)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
