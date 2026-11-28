#!/usr/bin/env python3
"""Mechanical gate for the CI artifact names and the Windows toolchain import.

The cross-platform workflow builds six legs and uploads one artifact per leg.  Two
things can go wrong without any compiler ever running:

  * the upload glob is built from the matrix entry name, while the file itself is
    named by scripts/build/unix/package.sh from HFS_TAG (scripts/plugin/unix/lib.sh).
    Linux arm64 is spelled `aarch64` there (dpkg spelling, already part of published
    file names), so a leg named `linux-arm64` uploads nothing -- run 45953527512
    produced a 257 MB tarball and then failed with 'No files were found'.
  * a Windows leg that builds through an action instead of build.ps1 never imports
    the MSVC environment, so the pinned CMake generator (NMake Makefiles) has no
    nmake.exe and the kernel build script dies inside cmake configure.
  * the universal macOS leg is the one leg whose tag does not come from `uname`: it
    cross-compiles both architectures, so the packaging step has to preset
    HFS_PLATFORM_ARCH. Without it the artifact is named after the host and the upload
    glob matches nothing -- the same failure as the first bullet, from the other end.

All of them are naming/wiring mistakes, invisible to typecheck and to every unit test.
Exit code 0 when every rule holds.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/build.yml"
LIB_SH = ROOT / "scripts/plugin/unix/lib.sh"
PACKAGE_SH = ROOT / "scripts/build/unix/package.sh"


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-ci-artifact-names: cannot read %s: %s" % (path, error))


def matrix_entries(workflow: str) -> list[dict]:
    """Every `fromJSON(...)` matrix literal in the workflow (PR legs + all legs)."""
    entries: list[dict] = []
    for literal in re.findall(r"'(\[\{.*?\}\])'", workflow, re.DOTALL):
        try:
            parsed = json.loads(literal)
        except json.JSONDecodeError as error:
            raise SystemExit("check-ci-artifact-names: matrix literal is not JSON: %s" % error)
        if isinstance(parsed, list):
            entries.extend(item for item in parsed if isinstance(item, dict))
    if not entries:
        raise SystemExit("check-ci-artifact-names: no matrix literal found in build.yml")
    return entries


def platform_tag(name: str) -> str:
    """The tag scripts/plugin/unix/lib.sh computes for this leg (os + platform arch).

    The `-universal` leg is the exception: lib.sh takes HFS_PLATFORM_ARCH from the environment
    there (it has no single host architecture to describe), so the name is only honest if the
    packaging step presets it -- which rule 2b checks.
    """
    os_name, _, arch = name.partition("-")
    # lib.sh: Linux keeps dpkg's spelling for arm64; everything else uses uname's.
    platform_arch = "aarch64" if (os_name == "linux" and arch == "arm64") else arch
    return "%s-%s" % (os_name, platform_arch)


def step_block(workflow: str, step_name: str) -> str | None:
    """The YAML lines belonging to one step (up to the next `- name:` at the same level)."""
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "- name: " + step_name:
            indent = len(line) - len(line.lstrip())
            block = [line]
            for follow in lines[index + 1 :]:
                stripped = follow.strip()
                if stripped.startswith("- name:") and (len(follow) - len(follow.lstrip())) == indent:
                    break
                block.append(follow)
            return "\n".join(block)
    return None


def main() -> int:
    workflow = read(WORKFLOW)
    lib_sh = read(LIB_SH)
    package_sh = read(PACKAGE_SH)
    findings: list[str] = []

    # ── 1. The naming rule the gate encodes must still be the one the script uses ──
    if 'HFS_PLATFORM_ARCH="aarch64"' not in lib_sh:
        findings.append(
            "scripts/plugin/unix/lib.sh: the linux-arm64 -> aarch64 mapping disappeared; "
            "this gate's platform_tag() is now stale"
        )
    if '"$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-unified"' not in package_sh:
        findings.append(
            "scripts/build/unix/package.sh: the macOS artifact is no longer "
            "HiFiShifter_v<version>_<tag>-unified.*; the upload glob must follow it"
        )
    if '"$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-packages"' not in package_sh:
        findings.append(
            "scripts/build/unix/package.sh: the Linux artifact is no longer "
            "HiFiShifter_v<version>_<tag>-packages/; the upload glob must follow it"
        )

    # ── 2. Every matrix leg name must equal the tag the packager will emit ──
    for entry in matrix_entries(workflow):
        name = str(entry.get("name", ""))
        expected = platform_tag(name)
        if name != expected:
            findings.append(
                "build.yml matrix: leg %r uploads/announces %r, but package.sh names the "
                "artifact with tag %r -- the upload glob would match no file"
                % (name, name, expected)
            )

    # ── 2b. A universal leg's tag cannot come from uname ──
    packaging = step_block(workflow, "Package installers (Unix)") or ""
    for entry in matrix_entries(workflow):
        name = str(entry.get("name", ""))
        if name.endswith("-universal") and "HFS_PLATFORM_ARCH=universal" not in packaging:
            findings.append(
                "build.yml: leg %r is universal, but the packaging step never exports "
                "HFS_PLATFORM_ARCH=universal -- lib.sh would derive the tag from the host "
                "architecture and the upload glob would match nothing" % name
            )

    # ── 3. Each upload must glob the tag and the suffix its platform actually writes ──
    # [Why the two platforms are checked separately] They used to share one step: both produced
    # one "unified" file. Linux now produces a directory of packages, so the shared step would
    # either upload nothing or, worse, upload the macOS glob for a Linux leg.
    uploads = [
        ("Upload unified package (macOS)", "-unified"),
        ("Upload Linux packages", "-packages"),
    ]
    for step_name, suffix in uploads:
        upload = step_block(workflow, step_name)
        if upload is None:
            findings.append("build.yml: step %r is missing" % step_name)
            continue
        if "matrix.name" not in upload:
            findings.append(
                "build.yml: %r must glob ${{ matrix.name }} so it follows the matrix "
                "instead of a copy-pasted arch spelling" % step_name
            )
        if suffix not in upload:
            findings.append(
                "build.yml: %r lost its %r suffix" % (step_name, suffix)
            )

    # ── 4. Windows must import the MSVC environment before tauri-action runs ──
    import_step = step_block(workflow, "Import MSVC environment (Windows)")
    if import_step is None:
        findings.append(
            "build.yml: step 'Import MSVC environment (Windows)' is missing -- the arm64 leg "
            "builds through tauri-action and would find no nmake.exe"
        )
    else:
        if "export-msvc-env.ps1" not in import_step:
            findings.append(
                "build.yml: 'Import MSVC environment (Windows)' must call "
                "scripts/setup/export-msvc-env.ps1 (the only writer of $GITHUB_ENV)"
            )
        import_at = workflow.find("- name: Import MSVC environment (Windows)")
        arm_at = workflow.find("- name: Build and bundle (Windows arm64)")
        if arm_at < 0:
            findings.append("build.yml: step 'Build and bundle (Windows arm64)' is missing")
        elif import_at > arm_at:
            findings.append(
                "build.yml: the MSVC import step is declared AFTER the arm64 build step; "
                "$GITHUB_ENV only reaches later steps"
            )

    if findings:
        for finding in findings:
            print("check-ci-artifact-names: %s" % finding, file=sys.stderr)
        return 1
    print(
        "check-ci-artifact-names: every leg name is its artifact tag, the universal leg presets "
        "its tag, and Windows imports MSVC"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())