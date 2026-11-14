#!/usr/bin/env python3
"""Reject invalid UTF-8 bytes in tracked, repository-owned text files.

A half-written CJK comment (a truncated 3-byte sequence) survives JSON/TS/Rust
parsers as a replacement character: nothing fails, but the comment becomes
unreadable and every diff of that line carries mojibake. The repo declares
`text=auto eol=lf` in .gitattributes, so all tracked text must be valid UTF-8.

Third-party trees are vendored upstream sources: their original encodings are
preserved and are intentionally not reformatted by this gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".ts", ".tsx", ".js", ".mjs", ".cjs", ".json", ".css", ".html", ".md",
    ".rs", ".toml", ".yml", ".yaml", ".sh", ".ps1", ".py", ".txt", ".bat",
    ".h", ".hpp", ".cpp", ".c", ".mm", ".nsh", ".nsi", ".in",
}
THIRD_PARTY_PREFIXES = ("third_party/", "backend/src-tauri/third_party/")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    bad: list[str] = []
    for rel in tracked_files():
        if rel.startswith(THIRD_PARTY_PREFIXES):
            continue
        path = ROOT / rel
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            ".gitattributes",
            ".prettierignore",
        }:
            continue
        try:
            data = path.read_bytes()
        except OSError as error:
            bad.append(f"{rel}: unreadable: {error}")
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            bad.append(f"{rel}:{error.start}: invalid UTF-8 byte")
    if bad:
        print("invalid UTF-8 in tracked text files:", file=sys.stderr)
        for entry in bad:
            print("  " + entry, file=sys.stderr)
        return 1
    print("UTF-8 gate OK: all owned tracked text files decode cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
