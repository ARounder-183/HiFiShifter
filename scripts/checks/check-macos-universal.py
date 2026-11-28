#!/usr/bin/env python3
"""Mechanical gate for the macOS universal leg.

The universal package is the one artifact whose correctness cannot be decided by reading a
single file: it is the product of a matrix entry, a workflow step, three shell scripts and the
Tauri CLI's own lipo. Every one of those can be wrong in a way that still builds:

  * `--target` placed in `cargo_extra` instead of `tauri_extra` lands on the cargo side of the
    `--` boundary; tauri never sees it, builds the host architecture and the run stays green;
  * a leg named `macos-universal` without `HFS_PLATFORM_ARCH=universal` names its artifact
    `macos-arm64-unified.pkg`, so the upload glob matches nothing;
  * dropping the arch assertion leaves "arm64 runs, x86_64 does not" undetectable until a user
    installs it;
  * the merge step is only reachable through flags that nothing else exercises.

Only `lipo -archs` on the produced files can prove the package really is universal -- that is
the workflow's `Assert universal Mach-O` step. This gate proves the wiring that produces it,
and that the assertion is still there.

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
FORMAT_SCRIPTS = {
    "vst3": ROOT / "scripts/vst3/unix/build.sh",
    "clap": ROOT / "scripts/clap/unix/build.sh",
}

UNIVERSAL_TARGET = "universal-apple-darwin"


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-macos-universal: cannot read %s: %s" % (path, error))


def matrices(workflow: str) -> list[list[dict]]:
    """The two `fromJSON(...)` matrix literals, in source order.

    The matrix is one expression -- `github.event_name == 'pull_request' && '<pr>' || '<release>'`
    -- so the PR list is written first and the release list second. The caller checks that
    ordering assumption instead of trusting it.
    """
    out: list[list[dict]] = []
    for literal in re.findall(r"'(\[\{.*?\}\])'", workflow, re.DOTALL):
        try:
            parsed = json.loads(literal)
        except json.JSONDecodeError as error:
            raise SystemExit("check-macos-universal: matrix literal is not JSON: %s" % error)
        if isinstance(parsed, list):
            out.append([item for item in parsed if isinstance(item, dict)])
    return out


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


def macos_legs(entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if str(entry.get("name", "")).startswith("macos-")]


def main() -> int:
    workflow = read(WORKFLOW)
    lib_sh = read(LIB_SH)
    package_sh = read(PACKAGE_SH)
    findings: list[str] = []

    # ── 1. One macOS leg per matrix, and the release one is the universal one ──
    both = matrices(workflow)
    if len(both) != 2:
        findings.append(
            "build.yml: expected two matrix literals (the PR list and the release list), found %d"
            % len(both)
        )
        pr, release = (both + [[], []])[:2]
    else:
        pr, release = both
    if any(str(entry.get("name", "")).startswith("windows-arm64") for entry in pr):
        findings.append(
            "build.yml: the first matrix literal looks like the release list; the PR list must be "
            "written first (the expression is 'pull_request && <pr> || <release>')"
        )
    if not any(str(entry.get("name", "")).startswith("windows-arm64") for entry in release):
        findings.append("build.yml: the second matrix literal looks like the PR list")

    pr_macos = macos_legs(pr)
    release_macos = macos_legs(release)

    if len(release_macos) != 1:
        findings.append(
            "build.yml: release builds must have exactly one macOS leg (found %d: %s) -- two legs "
            "ship two packages and pay the Intel leg's build cost twice"
            % (len(release_macos), ", ".join(str(e.get("name")) for e in release_macos) or "none")
        )
    for entry in release_macos:
        name = str(entry.get("name", ""))
        target = str(entry.get("cargo_target", ""))
        tauri_extra = str(entry.get("tauri_extra", ""))
        cargo_extra = str(entry.get("cargo_extra", ""))
        if target != UNIVERSAL_TARGET:
            findings.append(
                "build.yml: macOS release leg %r has cargo_target %r; it must be %r so that the "
                "release directory and the cargo cache key name the real target"
                % (name, target, UNIVERSAL_TARGET)
            )
        if "--target " + UNIVERSAL_TARGET not in tauri_extra:
            findings.append(
                "build.yml: leg %r must pass --target %s through tauri_extra -- that is the side "
                "of the '--' boundary tauri reads" % (name, UNIVERSAL_TARGET)
            )
        if "--target" in cargo_extra:
            findings.append(
                "build.yml: leg %r puts --target in cargo_extra; everything after '--' goes to "
                "cargo, so tauri would build the host architecture and the run would still be "
                "green" % name
            )
        if not name.endswith("-universal"):
            findings.append(
                "build.yml: the release macOS leg is named %r; the artifact tag is <os>-<arch> and "
                "the merged package is 'universal'" % name
            )

    if len(pr_macos) != 1:
        findings.append(
            "build.yml: PR builds must have exactly one macOS leg (found %d)" % len(pr_macos)
        )
    for entry in pr_macos:
        name = str(entry.get("name", ""))
        if str(entry.get("cargo_target", "")) == UNIVERSAL_TARGET:
            findings.append(
                "build.yml: PR leg %r builds universal; a PR only has to prove that the code "
                "compiles, and the second architecture doubles its wall clock" % name
            )
        if "--target" in str(entry.get("tauri_extra", "")):
            findings.append("build.yml: PR leg %r must stay single-architecture" % name)

    # ── 2. Both targets must be installed before cargo runs ──
    targets_step = step_block(workflow, "Add both macOS targets (universal)")
    if targets_step is None:
        findings.append(
            "build.yml: step 'Add both macOS targets (universal)' is missing -- cargo fails with "
            "'can't find crate for std' when the cross target was never added"
        )
    else:
        if UNIVERSAL_TARGET not in targets_step:
            findings.append(
                "build.yml: 'Add both macOS targets (universal)' must be gated on the universal "
                "leg (%r)" % UNIVERSAL_TARGET
            )
        for triple in ("aarch64-apple-darwin", "x86_64-apple-darwin"):
            if triple not in targets_step:
                findings.append(
                    "build.yml: 'Add both macOS targets (universal)' never runs "
                    "'rustup target add %s'" % triple
                )

    # ── 3. The arch assertion must exist, must be able to fail, and must cover all four rules ──
    assert_step = step_block(workflow, "Assert universal Mach-O")
    if assert_step is None:
        findings.append(
            "build.yml: step 'Assert universal Mach-O' is missing -- without it a package that "
            "only runs on arm64 passes every check"
        )
    else:
        if "continue-on-error" in assert_step:
            findings.append(
                "build.yml: 'Assert universal Mach-O' must not be continue-on-error; an assertion "
                "that cannot fail is documentation"
            )
        for needle, why in (
            ("ARCH_EXCLUSIVE", "the arm64-only allowlist is not checked"),
            ("lipo -archs", "no file's slices are inspected"),
            ("otool -arch x86_64", "the x86_64 slice's dependencies are not inspected"),
            ("codesign --verify", "the re-sign after lipo is not verified"),
        ):
            if needle not in assert_step:
                findings.append("build.yml: 'Assert universal Mach-O': %s (%r missing)" % (why, needle))

    # ── 4. The packaging step must build, merge and name the universal package ──
    packaging = step_block(workflow, "Package installers (Unix)")
    if packaging is None:
        findings.append("build.yml: step 'Package installers (Unix)' is missing")
    else:
        if "HFS_PLATFORM_ARCH=universal" not in packaging:
            findings.append(
                "build.yml: the packaging step must export HFS_PLATFORM_ARCH=universal; without it "
                "the universal package is published under a single-architecture name and the "
                "upload glob matches nothing"
            )
        if "--merge-arch" not in packaging:
            findings.append(
                "build.yml: the packaging step never passes --merge-arch; the plugin bundles would "
                "keep the host architecture"
            )
        for triple in ("aarch64-apple-darwin", "x86_64-apple-darwin"):
            if "--merge-app-arch" not in packaging or triple not in packaging:
                findings.append(
                    "build.yml: the packaging step must pass --merge-app-arch for %s; the App's "
                    "SoundTouch dylibs are single-arch" % triple
                )
                break

    # ── 5. The scripts the step relies on ──
    if 'HFS_PLATFORM_ARCH:-' not in lib_sh:
        findings.append(
            "scripts/plugin/unix/lib.sh: HFS_PLATFORM_ARCH can no longer be preset, so a merged "
            "package cannot be named after both architectures"
        )
    if "hfs_arch_of_triple" not in lib_sh:
        findings.append(
            "scripts/plugin/unix/lib.sh: hfs_arch_of_triple is missing; a cross build would take "
            "the runtime-dependency list and the tag from the host"
        )
    for needle, why in (
        ("merge_macho_pair", "no shared lipo helper"),
        ("merge_app_arch", "the App bundle is never merged"),
        ("hfs_is_universal", "a file Tauri already lipo'd would be merged twice"),
    ):
        if needle not in package_sh:
            findings.append("scripts/build/unix/package.sh: %s (%r missing)" % (why, needle))
    for fmt, path in FORMAT_SCRIPTS.items():
        script = read(path)
        if "TARGET_TRIPLE" not in script:
            findings.append(
                "scripts/%s/unix/build.sh: no --target support; the second architecture's plugin "
                "bundle cannot be built" % fmt
            )
        elif '--target "$TARGET_TRIPLE"' not in script:
            findings.append(
                "scripts/%s/unix/build.sh: TARGET_TRIPLE is parsed but never passed to cargo" % fmt
            )

    if findings:
        for finding in findings:
            print("check-macos-universal: %s" % finding, file=sys.stderr)
        return 1
    print(
        "check-macos-universal: one universal leg, both targets installed, the merge wired and "
        "the arch assertion in place"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
