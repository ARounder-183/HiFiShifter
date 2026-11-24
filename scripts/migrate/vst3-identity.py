#!/usr/bin/env python3
"""把 REAPER 工程里 HiFiShifter 的 VST3 身份从 macOS 那套文本改回 Windows 那套。

【为什么需要这一步】宿主写进 <> 的类 ID 是**平台文本化**的结果：Windows 走 COM/GUID，
macOS / Linux 走逐字大端。0.2.0 的产物把类 ID 冻结成 GUID 字节，于是同一个插件在两边
得到两个字符串：

    macOS   174F3A9C00008B1E00004D02B7A13C6E
    Windows 9C3A4F1700001E8B00004D02B7A13C6E

插件现在统一发出"按字定义、按平台编码"的身份（见 hifishifter-vst3/src/vst3.rs 常量
上的长注释），Windows 侧字节一个都没变 —— 存量 Windows 工程不需要跑这个脚本。
这个脚本只服务一种文件：**0.2.0 的 macOS 产物保存过的工程**（绝大多数是测试工程）。
把它们里面那一串换成 Windows 那串，宿主就能在新旧任何一端认出插件。

【改什么、不改什么】
  * 只改 <VST "VST3: HiFiShifter …" … > 行里 {…} 内的那 32 个十六进制字符；
  * 模块名不是 HiFiShifter 的 VST 行一律不碰（同一个工程里还有别的插件）；
  * 行尾的 ARA archive id（com.hifishifter.ara.archive）是字符串、与平台无关，不动；
  * {…} 前那个十进制 id 不动 —— REAPER 认的是 {…} 里的字符串（实测：只换这一串，
    同一个工程立刻从"效果不可用"变成正常加载，随后 Ctrl+S 保存回去也没被改写）。

【默认是演练】不带 --write 只打印将要修改的位置，一个字节都不写。真实工程请先关 REAPER。

用法：
    python scripts/migrate/vst3-identity.py <工程或目录> [...]            # 演练
    python scripts/migrate/vst3-identity.py <工程或目录> [...] --write    # 落盘
    python scripts/migrate/vst3-identity.py --self-test                  # 自检

退出码：0 = 成功（含"没有需要改的"），1 = 有文件读不了/写不了。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 两个平台的文本形式。它们只在这里出现一次：任何一处改动都必须同时改测试。
MACOS_UID = "174F3A9C00008B1E00004D02B7A13C6E"
WINDOWS_UID = "9C3A4F1700001E8B00004D02B7A13C6E"

# <VST "VST3: HiFiShifter (HiFiShifter)" HiFiShifter.vst3 0 "" 1375025479{174F…}
#   ↑ 模块名可能是 HiFiShifter.vst3 / HiFiShifterVerify.vst3 / 带路径的写法，
#     所以只要求引号里的显示名以 HiFiShifter 开头，且 {…} 是 32 个十六进制字符。
_VST_LINE = re.compile(
    r'(?P<prefix><VST\s+"VST3:\s*HiFiShifter[^"]*"[^\n{]*\{)(?P<uid>[0-9A-Fa-f]{32})(?P<suffix>\})'
)


def rewrite(text: str, *, source: str = MACOS_UID, target: str = WINDOWS_UID) -> tuple[str, int]:
    """返回 (新文本, 改了几处)。除 {…} 外一个字符都不动。"""

    changed = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal changed
        if match.group("uid").upper() != source.upper():
            return match.group(0)
        changed += 1
        return match.group("prefix") + target + match.group("suffix")

    return _VST_LINE.sub(swap, text), changed


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in {".rpp", ".rpp-bak"}:
                    files.append(candidate)
        elif path.is_file():
            files.append(path)
        else:
            print(f"[skip] 不存在：{path}", file=sys.stderr)
    return files


def self_test() -> int:
    sample = (
        '      <VST "VST3: HiFiShifter (HiFiShifter)" HiFiShifter.vst3 0 "" '
        f"1375025479{{{MACOS_UID}}} com.hifishifter.ara.archive\n"
        '      <VST "VST3: HiFiShifter (HiFiShifter)" HiFiShifter.vst3 0 "" '
        f"1375025479{{{WINDOWS_UID}}} com.hifishifter.ara.archive\n"
        '      <VST "VST3: Melodyne (Celemony)" Melodyne.vst3 0 "" '
        f"1{{{MACOS_UID}}}\n"
    )
    out, changed = rewrite(sample)
    assert changed == 1, f"只该改 HiFiShifter 的那一行，实际 {changed}"
    assert f"{{{WINDOWS_UID}}}" in out
    assert out.count(WINDOWS_UID) == 2, "已经是 Windows 串的那行不能被改坏"
    assert "Melodyne.vst3" in out and out.count(MACOS_UID) == 1, "别的插件的 UID 不许碰"
    # 幂等：再跑一次什么都不改。
    again, changed_again = rewrite(out)
    assert changed_again == 0 and again == out, "第二次运行必须是无操作"
    print("self-test ok")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="改写 REAPER 工程里 HiFiShifter 的 VST3 身份文本")
    parser.add_argument("paths", nargs="*", type=Path, help="工程文件或目录")
    parser.add_argument("--write", action="store_true", help="真的写盘（默认只演练）")
    parser.add_argument("--self-test", action="store_true", help="跑内置自检后退出")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.paths:
        parser.error("至少要给一个工程文件或目录（或 --self-test）")

    files = collect(args.paths)
    if not files:
        print("没有找到工程文件。")
        return 0

    total = 0
    failures = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError as error:
            print(f"[error] 读不了 {path}: {error}", file=sys.stderr)
            failures += 1
            continue
        # REAPER 工程是 UTF-8；用 surrogateescape 保证非 UTF-8 字节原样往返。
        text = raw.decode("utf-8", errors="surrogateescape")
        rewritten, changed = rewrite(text)
        if not changed:
            continue
        total += changed
        print(f"{'[write]' if args.write else '[dry-run]'} {path}: {changed} 处")
        if args.write:
            try:
                path.write_bytes(rewritten.encode("utf-8", errors="surrogateescape"))
            except OSError as error:
                print(f"[error] 写不了 {path}: {error}", file=sys.stderr)
                failures += 1

    print(f"共 {total} 处" + ("（已写盘）" if args.write else "（演练，未写盘；加 --write 生效）"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
