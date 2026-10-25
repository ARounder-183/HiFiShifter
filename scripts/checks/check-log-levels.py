#!/usr/bin/env python3
"""限流日志只允许 Warn / Error。

【为什么要这条门禁】限流器（`hifishifter_kernel::log_limiter`）解决的是「同一条故障
在一秒钟里被打印五百次」。它按 10 秒窗口抑制，并把被压掉的条数在下一条前汇总 ——
信息不丢，只丢重复。

但把它用在 Info / Debug 上会制造一个**静默的黑洞**：默认日志级别是 Warn，于是
`emit_limited_key(Info, …)` 一条都不会输出，而代码看起来仍然「有诊断」。这个坑
真实发生过：命令到达日志、前端错误上报、视图回调的进出打点，三处都用 Info + 限流，
于是一整轮排查都建立在「这些日志不存在」之上，而结论是反的。

规则：
  * 故障信号 → `Warn` / `Error` + 限流（永不消失，只是不重复）；
  * 正常路径的跟踪信息 → 直接 `log::info!` / `log::debug!`，由日志级别控制，
    不限流（不限流就不会静默消失）。

退出码：0 = 合规，1 = 发现违规。
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2] / 'backend'
SKIP_DIRS = {'target', 'third_party', 'node_modules'}

# `emit_limited(` / `emit_limited_key(` 之后 260 个字符内出现的 `log::Level::X`。
CALL = re.compile(r'emit_limited(?:_key)?\s*\(')
LEVEL = re.compile(r'log::Level::(\w+)')
ALLOWED = {'Warn', 'Error'}


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
        print('限流日志只允许 Warn/Error（见本脚本头部）：')
        for violation in violations:
            print('  ' + violation)
        print()
        print('Info/Debug 的跟踪信息请直接用 log::info! / log::debug! —— 它们由日志级别')
        print('控制，不需要限流；套上限流只会在默认级别下变成静默黑洞。')
        return 1

    print(f'log levels ok ({len(sources())} files scanned)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
