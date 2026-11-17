#!/usr/bin/env python3
"""Verify every native file dialog carries a type filter from the vocabulary.

The symptoms this prevents are all silent:

  * a save dialog created without SetFileTypes / add_filter shows an empty type
    dropdown, so the user can save a .zip diagnostics bundle as .txt;
  * a filter whose name is a glob ("*.mid") renders as a pattern in the
    dropdown instead of a readable type name;
  * an attachment "save as" with no filter at all accepts any extension.

None of those throw. They look like a working dialog until someone checks the
saved file's extension. This check makes the missing filter a CI failure.

Two surfaces:

  * the plug-in's own pickers (pick_save_path / pick_open_path), which must be
    given a hifishifter_kernel::file_types::FileType;
  * the standalone app's rfd::FileDialog chains, which must call .add_filter
    before terminating (folder pickers excepted - they have no file type).

MAINTENANCE RULE: a new picker entry point (a third helper that ends in a
system dialog) must be added to PICKER_CALLS below. Otherwise this gate reports
"0 call sites" for it and silently stops guarding anything.

Exit code 0 when every dialog is typed, 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

# 插件的两个「问用户要路径」入口。
PICKER_CALLS = ("pick_save_path", "pick_open_path")

# 词表里的类型：调用点必须交出其中之一（或现构造一个 FileType）。
VOCABULARY = (
    "DIAGNOSTICS_ZIP",
    "MIDI",
    "HIFISHIFTER_JSON",
    "NOTEBOOK_MARKDOWN",
    "NOTEBOOK_HTML",
    "HIFISHIFTER_PROJECT",
    "HIFISHIFTER_PROJECT_BACKUP",
    "JSON_PROJECT",
    "REAPER_PROJECT",
    "REAPER_PROJECT_BACKUP",
    "VOCALSHIFTER_PROJECT",
    "WAV_AUDIO",
    "notebook_document",
    "FileType::",
    "file_types::",
)

# rfd 的终止调用：链在这里结束。
RFD_TERMINATORS = (".save_file(", ".pick_file(", ".pick_files(", ".pick_folder(")

SKIP_DIRS = {"target", "node_modules", ".git", "third-party", "gen"}


def rust_sources():
    out = []
    for path in BACKEND.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def chain_of(source, start):
    """从 start 起的 builder 链，到终止调用为止。"""
    for terminator in RFD_TERMINATORS:
        index = source.find(terminator, start)
        if index >= 0:
            return source[start : index + len(terminator)]
    # 没有终止调用（或链写成了变量）—— 截到下一个分号，让缺失照常报出来。
    semi = source.find(";", start)
    return source[start : semi if semi >= 0 else len(source)]


def enclosing_signature(source, index):
    """
    包住 index 的那个函数签名（从最近的 fn 到它的左花括号）。

    【为什么需要】browser_files.rs 里的非 Windows 分支是**转发**：
    platform_services::pick_save_path(file_name, file_type)。它把调用方交进来的
    过滤器原样往下传，本身当然不写词表常量 —— 硬要求它写，只会逼出一个多余的参数。
    只要**它自己**的签名声明了 FileType，这条路就仍然是类型安全的。
    """
    start = source.rfind("fn ", 0, index)
    if start < 0:
        return ""
    brace = source.find("{", start)
    return source[start : brace if brace >= 0 else index]


def call_arguments(source, start):
    """start 处那次调用的参数文本（按括号配对截取）。"""
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return source[start:]


def main():
    violations = []
    checked_pickers = 0
    checked_rfd = 0

    for path in rust_sources():
        source = path.read_text(encoding="utf-8", errors="replace")
        display = path.relative_to(ROOT).as_posix()

        # ── 插件：pick_save_path / pick_open_path 必须拿到词表里的一条 ────────
        for name in PICKER_CALLS:
            for match in re.finditer(r"\b" + name + r"\s*\(", source):
                # 定义处（fn pick_save_path(）不是调用点。
                prefix = source[max(0, match.start() - 12) : match.start()]
                if prefix.rstrip().endswith("fn"):
                    continue
                checked_pickers += 1
                arguments = call_arguments(source, match.end() - 1)
                signature = enclosing_signature(source, match.start())
                if "FileType" in signature and "FileType" not in arguments:
                    # 转发：本函数自己就声明了 FileType 参数，类型链没有断。
                    continue
                if not any(token in arguments for token in VOCABULARY):
                    violations.append(
                        display
                        + ": "
                        + name
                        + " 没有使用内核词表里的类型过滤器\n    "
                        + arguments.strip()[:160]
                    )

        # ── 独立 App：rfd 链必须在终止前带上 add_filter ───────────────────────
        for match in re.finditer(r"rfd::FileDialog::new\(\)", source):
            chain = chain_of(source, match.start())
            if ".pick_folder(" in chain:
                continue
            checked_rfd += 1
            if ".add_filter(" not in chain:
                violations.append(
                    display
                    + ": rfd 对话框没有 add_filter（类型下拉框会是空的）\n    "
                    + chain.strip()[:200]
                )

    print(
        "check-dialog-filters: "
        + str(checked_pickers)
        + " plug-in picker call(s), "
        + str(checked_rfd)
        + " rfd dialog(s)"
    )

    # 自检：扫描必须真的扫到东西，否则「0 处违规」毫无意义。
    if checked_pickers == 0 or checked_rfd == 0:
        print("\n扫描没有命中任何对话框 —— 路径或判据已失效（不是「没有违规」）")
        return 1

    if violations:
        print("\n以下文件对话框没有正确的类型过滤器：")
        for violation in violations:
            print("  - " + violation)
        print("\n类型名与扩展名的唯一来源是 backend/hifishifter-kernel/src/file_types.rs。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
