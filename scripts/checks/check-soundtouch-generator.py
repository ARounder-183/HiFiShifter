#!/usr/bin/env python3
"""SoundTouch 的 CMake 生成器必须被钉住，且能自愈陈旧缓存。

【为什么需要这条门禁】build.rs 把 SoundTouch 配到 OUT_DIR/soundtouch_build。
CMake 一旦用某个生成器生成过 CMakeCache.txt，后续即使 CMAKE_GENERATOR 环境变量
变了也仍复用旧缓存 —— 本机实测 8 份缓存都写着 Visual Studio 17 2022，而当前
msvc-env.ps1 给的是 NMake Makefiles，于是 clippy 门禁死在 build script 里，看起来
像源码坏了。build.rs 必须：显式把生成器传给 -G、声明环境变量变化重跑、并在缓存
生成器不一致时清理重建，同时把 cmake 的输出回传，避免只留一句 exit code。

退出码：0 = 合规，1 = 发现违规。
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_RS = ROOT / "backend" / "hifishifter-kernel" / "build.rs"


def require(text: str, needle: str, reason: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"缺少 {reason}：{needle}")


def main() -> int:
    text = BUILD_RS.read_text(encoding="utf-8")
    failures: list[str] = []
    require(
        text,
        'cargo:rerun-if-env-changed=CMAKE_GENERATOR',
        "生成器环境变量的 rerun 声明",
        failures,
    )
    require(text, '"-G"', "显式的 cmake -G 参数", failures)
    require(text, "CMAKE_GENERATOR:INTERNAL=", "缓存生成器比对", failures)
    require(text, "remove_dir_all(&build_dir)", "不一致缓存的清理", failures)
    require(text, "cmake configure stdout", "configure 失败时的 stdout 回传", failures)
    require(text, "cmake configure stderr", "configure 失败时的 stderr 回传", failures)
    if failures:
        print("SoundTouch generator gate FAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("SoundTouch generator gate OK: generator pinned, cache validated, output surfaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
