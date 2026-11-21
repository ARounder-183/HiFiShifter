#!/usr/bin/env python3
"""SoundTouch's CMake generator must be pinned, and stale caches must self-heal.

[Why this gate exists] build.rs configures SoundTouch into OUT_DIR/soundtouch_build.
Once CMake has generated a CMakeCache.txt with some generator, it keeps reusing that
old cache even after the CMAKE_GENERATOR environment variable changes - measured on
this machine, all 8 caches said Visual Studio 17 2022 while the current
msvc-env.ps1 provides NMake Makefiles, so the clippy gate died inside the build
script and looked like broken source code. build.rs must: pass the generator
explicitly to -G, declare a rerun when the environment variable changes, clean and
rebuild when the cached generator disagrees, and surface cmake's output so a bare
exit code is never all that remains.

Exit codes: 0 = compliant, 1 = violation found.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_RS = ROOT / "backend" / "hifishifter-kernel" / "build.rs"


def require(text: str, needle: str, reason: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"missing {reason}: {needle}")


def main() -> int:
    text = BUILD_RS.read_text(encoding="utf-8")
    failures: list[str] = []
    require(
        text,
        'cargo:rerun-if-env-changed=CMAKE_GENERATOR',
        "rerun declaration for the generator environment variable",
        failures,
    )
    require(text, '"-G"', "explicit cmake -G argument", failures)
    require(text, "CMAKE_GENERATOR:INTERNAL=", "cached-generator comparison", failures)
    require(text, "remove_dir_all(&build_dir)", "cleanup of a mismatched cache", failures)
    require(text, "cmake configure stdout", "stdout surfaced when configure fails", failures)
    require(text, "cmake configure stderr", "stderr surfaced when configure fails", failures)
    if failures:
        print("SoundTouch generator gate FAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("SoundTouch generator gate OK: generator pinned, cache validated, output surfaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
