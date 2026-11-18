#!/usr/bin/env python3
"""同步 / 校验产品版本号：一处输入，按清单落盘（清单见同目录的 version-sources.json）。

【为什么是 .py 而不是继续用 .ps1】改版本就是"读文本 → 正则替换 → 写回"，没有任何平台
API；写成 PowerShell 的唯一后果是：macOS/Linux 的贡献者必须先装一个 100 MB 的 pwsh 才能
改版本号（本仓的 install_deps_* 脚本不装它，static 门禁又只跑在 windows-latest，所以这个
缺口从来没人撞上）。Python 在这里已经是三平台硬依赖：fetch-models.py、check-*.py、
vst3/unix/build.sh 都要它，校验方（check-product-consistency.ps1）本来也在调它。

【为什么写入与校验在同一个文件里】两边都必须按 version-sources.json 里的 occurrences
逐处解释"哪几处是产品版本"。分成两种语言写，就是同一份规则两遍实现 —— 而这条规则
漏一处，症状只是某个界面（或「程序和功能」）显示别的版本号。

用法：
    python scripts/release/version.py --write 0.2.0-beta   # 完整版本
    python scripts/release/version.py --write beta16       # 简写：当前 major.minor.patch + -beta.16
    python scripts/release/version.py --write beta         # 简写：当前 major.minor.patch + -beta
    python scripts/release/version.py --check              # 校验全部来源一致（不一致退出码 1）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Windows 的 stdout 跟随 ANSI 代码页（英文系统是 cp1252、中文系统是 GBK），打印中文会
# UnicodeEncodeError 打死进程 —— 真正的结论一条都进不了日志，红的是编码而不是规则。
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = Path(__file__).resolve().parent / "version-sources.json"


def load_sources() -> dict:
    if not SOURCES_PATH.is_file():
        raise SystemExit(f"缺少版本清单：{SOURCES_PATH}")
    with SOURCES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(relative: str) -> str:
    # 【为什么 newline=""】保持文件原有的换行符：Windows 上写成 CRLF、仓库里存 LF 会让
    # 一次版本号改动变成整文件 diff，评审时什么都看不出来。
    with (REPO_ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(relative: str, text: str) -> None:
    # 与仓库其它文本一致：UTF-8 无 BOM。
    with (REPO_ROOT / relative).open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def source_label(source: dict, index: int = 0) -> str:
    name = source.get("id") or source["path"]
    return name if int(source["occurrences"]) == 1 else f"{name} #{index + 1}"


def canonical_version(manifest: dict) -> str:
    for source in manifest["sources"]:
        if source["path"] == manifest["canonical"]:
            match = re.search(source["pattern"], read_text(source["path"]))
            if not match:
                raise SystemExit(f"{source['path']} 里找不到版本字段")
            return match.group(2)
    raise SystemExit(f"version-sources.json 的 canonical 不在 sources 里：{manifest['canonical']}")


def resolve_target(requested: str, current: str) -> str:
    """把 beta / beta16 这类简写补全成完整版本。

    【为什么单独取前缀】早先的 PowerShell 实现在一次匹配之后又去读自动变量里的第二组，
    而那个位置是空的 —— 于是 -Version beta15 会拼出 "0.1.0-beta."。这里先取序号、再独立
    求前缀，两步互不干扰。
    """
    match = re.fullmatch(r"(?i)beta(?:[.\-]?(\d+))?", requested)
    if not match:
        return requested
    prefix_match = re.match(r"^(\d+\.\d+\.\d+)-", current)
    if not prefix_match:
        raise SystemExit(f"当前版本 {current} 无法推导 beta 目标版本")
    ordinal = match.group(1)
    return f"{prefix_match.group(1)}-beta" + (f".{ordinal}" if ordinal else "")


def collect_sources(manifest: dict) -> list[tuple[dict, str, list[re.Match[str]]]]:
    """读出全部来源并校验**前 N 处**产品版本字段存在，返回 (source, text, matches)。

    【为什么是"至少 occurrences"而不是"恰好 occurrences"】清单里的 occurrences
    表示"正则命中的前 N 处是产品版本"，不是文件里只能有 N 处匹配：
    `frontend/package-lock.json` 的正则会连带命中几百个依赖的 version 字段，
    `re.sub(..., count=N)` 只改前两处（顶层 + packages[""]）。校验必须遵守同一
    口径，否则把合法依赖版本当成漂移。
    """
    collected: list[tuple[dict, str, list[re.Match[str]]]] = []
    for source in manifest["sources"]:
        text = read_text(source["path"])
        expected = int(source["occurrences"])
        matches = list(re.finditer(source["pattern"], text))
        if len(matches) < expected:
            raise SystemExit(
                f"{source['path']} 里只找到 {len(matches)} 处版本字段，"
                f"version-sources.json 要求至少 {expected} 处"
            )
        collected.append((source, text, matches))
    return collected


def write_version(requested: str) -> int:
    manifest = load_sources()
    current = canonical_version(manifest)
    version = resolve_target(requested, current)

    # 【两阶段】先校验并算好全部写入内容，再统一落盘。旧实现边校验边写：
    # 若第 5 个来源突然少了一处版本字段，前 4 个文件已经被改成了新版本，
    # 仓库停在半升级状态，而 --check 当时也查不出这种"多/少一处"的漂移。
    pending: list[tuple[str, str]] = []
    for source, text, _matches in collect_sources(manifest):
        occurrences = int(source["occurrences"])
        updated = re.sub(
            source["pattern"],
            lambda match, v=version: match.group(1) + v + match.group(3),
            text,
            count=occurrences,
        )
        pending.append((source["path"], updated))

    for path, updated in pending:
        write_text(path, updated)

    print(f"版本：{current} -> {version}")
    return 0


def check_version() -> int:
    manifest = load_sources()
    try:
        collected = collect_sources(manifest)
    except SystemExit as error:
        print(str(error), file=sys.stderr)
        return 1

    found: list[tuple[str, str]] = []
    for source, _text, matches in collected:
        # 与写入口径一致：只比对前 occurrences 处（其余命中的是依赖版本）。
        for index in range(int(source["occurrences"])):
            found.append((source_label(source, index), matches[index].group(2)))

    distinct = sorted({value for _, value in found})
    if len(distinct) != 1:
        report = " | ".join(f"{label} = {value}" for label, value in found)
        print(f"版本号不一致（跑 python scripts/release/version.py --write <版本>）：{report}", file=sys.stderr)
        return 1
    print(f"全部来源 = {distinct[0]}（{len(found)} 处）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 / 校验 HiFiShifter 的产品版本号")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", metavar="VERSION", help="写入版本（支持 beta / beta16 简写）")
    group.add_argument("--check", action="store_true", help="只校验全部来源是否一致")
    args = parser.parse_args()

    if args.check:
        return check_version()
    return write_version(args.write)


if __name__ == "__main__":
    raise SystemExit(main())
