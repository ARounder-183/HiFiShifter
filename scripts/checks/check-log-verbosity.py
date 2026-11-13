#!/usr/bin/env python3
"""日志详细程度的源码门禁：四档策略的不回潮检查。

【为什么需要】四档设置落地之后，最容易退化成"代码里又长出一个独立开关"：
某处重新读 HIFISHIFTER_DEBUG_*、某处重新直接调 console.*、注释里重新写
"因为默认级别是 Warn 所以用 Warn"。这些都不会让编译失败，只会让用户选了
「无」却仍然有输出、选了「低」却仍然刷屏。

检查项（逐条可定位）：
  1. 生产 Rust 源码不得出现遗留调试环境变量；
  2. 生产前端源码不得直接调用 console.*（唯一出口 services/frontendLog.ts）；
  3. 注释不得再陈述"默认级别是 Warn 所以用 Warn"；
  4. 出厂默认必须是 Low，且宿主不在自己模块里调 set_max_level；
  5. log_settings.json 的文件名常量只有一处；
  6. 文档默认值一致（不再写 release 默认 info）。

排除：docs/、scripts/、third_party/、target/、cfg(test) 区间、build.rs。
退出码：0 = 合规，1 = 发现违规。
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
STALE_COMMENT = re.compile(
    r'默认日志级别是 Warn|因为默认级别|必须是 WARN 级|'
    r'用 WARN 级|release 默认 Warn|默认级别是 Warn'
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
    """把 cfg(test) mod tests 区间换成空行，避免测试里的探针误报。"""
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
        if re.search(r'release (版|build|builds).{0,24}(默认|default).{0,16}info', text, re.I):
            violations.append(f'{doc.relative_to(ROOT).as_posix()}: stale "release default info" text')

    if violations:
        print('日志详细程度门禁失败：')
        for item in violations[:120]:
            print('  ' + item)
        print(f'\n共 {len(violations)} 处。见本脚本头部说明。')
        return 1
    print(f'log verbosity ok ({len(rust)} Rust, {len(frontend_sources())} frontend files scanned)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
