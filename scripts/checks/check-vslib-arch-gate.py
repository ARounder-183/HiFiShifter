#!/usr/bin/env python3
"""The vslib native backend is x86_64-Windows only, and every gate must say so.

"hifishifter-kernel/src/vslib.rs" carries #[link(name = "vslib_x64")]: a **link-time**
directive, not a runtime dlopen. The import library it names is x64 machine code living in
third_party/vslib, so it can only ever satisfy an x86_64-Windows link, and
"hifishifter-kernel/build.rs" therefore adds that search path on x86_64-Windows and nowhere
else. Whenever the module participates in a build the link directive is emitted, and on any
other target the link step dies with

    LINK : fatal error LNK1181: cannot open input file 'vslib_x64.lib'

Run 46016687058 is that failure: "Build Backend (windows-arm64)" compiled the whole kernel and
failed at the very last link, because lib.rs gated "mod vslib" on target_os = "windows" alone
while renderer/mod.rs -- which imports it -- also required target_arch = "x86_64". Nothing
else in the repo could see the difference: it type-checks, it unit-tests, and it only breaks
on a target nobody builds locally.

Five assertions, in the order a reviewer would ask them:

1. no cfg attribute anywhere pairs feature = "vslib" with target_os = "windows" without also
   naming target_arch = "x86_64" -- the bug was exactly one such attribute;
2. the three items that bring the backend into the build (the FFI module, the processor, the
   app-side re-export) each carry that full predicate set;
3. the kernel build script links the import library only under the same condition, and only
   after the guard;
4. exactly one file declares the link attribute;
5. the app crate forwards its onnx/vslib features to the kernel instead of inheriting the
   kernel's defaults -- otherwise "--no-default-features --features onnx" silently keeps vslib
   compiled in, which is what made the arm64 leg link an x64 import library at all.

Exit code 0 when every rule holds.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
KERNEL = BACKEND / "hifishifter-kernel"
APP = BACKEND / "src-tauri"

KERNEL_LIB = KERNEL / "src" / "lib.rs"
KERNEL_RENDERER = KERNEL / "src" / "renderer" / "mod.rs"
KERNEL_BUILD = KERNEL / "build.rs"
VSLIB_FFI = KERNEL / "src" / "vslib.rs"
APP_LIB = APP / "src" / "lib.rs"
APP_MANIFEST = APP / "Cargo.toml"

# The one predicate set the native backend is allowed to exist under.
REQUIRED = ('feature = "vslib"', 'target_os = "windows"', 'target_arch = "x86_64"')

LINK_ATTR = '#[link(name = "vslib_x64")]'
LINK_DIRECTIVE = "cargo:rustc-link-lib=dylib=vslib_x64"


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-vslib-arch-gate: cannot read %s: %s" % (path, error))


def rel(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_cfg_attrs(text: str):
    """Yield (line_number, source) for every #[cfg(...)] attribute, parens balanced.

    Attributes in this tree wrap over several lines (renderer/mod.rs has a three-line one), so
    a line-oriented scan would read a fragment and conclude the wrong thing.
    """
    for match in re.finditer(r"#\[cfg\(", text):
        depth = 0
        cursor = match.start() + len("#[cfg")
        end = -1
        while cursor < len(text):
            char = text[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = cursor
                    break
            cursor += 1
        if end < 0:
            continue
        if text[end + 1 : end + 2] != "]":
            continue
        yield text.count("\n", 0, match.start()) + 1, text[match.start() : end + 2]


def gate_for(text: str, needle: str, path: pathlib.Path) -> str:
    """The #[cfg(...)] attribute that guards the line containing needle.

    Walks upward from the item, skipping blank lines and // comments, so a documented gate is
    read exactly like a bare one. Returns "" when the item is ungated -- which is itself a
    finding, because an ungated mod vslib compiles everywhere.
    """
    lines = text.splitlines()
    index = next((i for i, line in enumerate(lines) if needle in line), None)
    if index is None:
        raise SystemExit(
            "check-vslib-arch-gate: %s no longer contains %r; this gate is stale"
            % (rel(path), needle)
        )
    collected: list[str] = []
    cursor = index - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if stripped.startswith("//") or not stripped:
            cursor -= 1
            continue
        collected.insert(0, stripped)
        if stripped.startswith("#[cfg"):
            break
        if not stripped.startswith("#["):
            collected.clear()
            break
        cursor -= 1
    return "\n".join(collected)


def check_gate(text: str, needle: str, path: pathlib.Path, what: str, findings: list[str]) -> None:
    gate = gate_for(text, needle, path)
    if not gate:
        findings.append(
            "%s: '%s' (%s) is ungated. It must carry "
            '#[cfg(all(feature = "vslib", target_os = "windows", target_arch = "x86_64"))]'
            % (rel(path), needle, what)
        )
        return
    missing = [token for token in REQUIRED if token not in gate]
    if missing:
        findings.append(
            "%s: the gate on '%s' (%s) is missing %s. On a target that matches it, the "
            "link-time directive in vslib.rs is emitted and the link fails with "
            "LNK1181: cannot open input file 'vslib_x64.lib'"
            % (rel(path), needle, what, ", ".join(missing))
        )


def main() -> int:
    findings: list[str] = []

    # -- 1. no cfg pairs vslib with windows and forgets the architecture ----
    for path in sorted(BACKEND.rglob("*.rs")):
        if "target" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, attr in iter_cfg_attrs(text):
            # A vslib gate that never mentions Windows is not a native-backend gate: the
            # feature alone drives several pure-Rust code paths (pitch edit dispatch, the
            # algorithm enum, the param descriptors).
            if 'feature = "vslib"' not in attr or 'target_os = "windows"' not in attr:
                continue
            missing = [token for token in REQUIRED if token not in attr]
            if missing:
                findings.append(
                    "%s:%d: cfg is missing %s -- %s"
                    % (
                        rel(path),
                        line_number,
                        ", ".join(missing),
                        " ".join(attr.split()),
                    )
                )

    kernel_lib = read(KERNEL_LIB)
    kernel_renderer = read(KERNEL_RENDERER)
    kernel_build = read(KERNEL_BUILD)
    app_lib = read(APP_LIB)
    app_manifest = read(APP_MANIFEST)
    ffi = read(VSLIB_FFI)

    # -- 2. the items that bring the backend into the build ------------------
    check_gate(kernel_lib, "pub mod vslib;", KERNEL_LIB, "the FFI module", findings)
    check_gate(
        kernel_renderer, "pub mod vslib_processor;", KERNEL_RENDERER, "the processor", findings
    )
    check_gate(
        app_lib,
        "pub(crate) use hifishifter_kernel::vslib;",
        APP_LIB,
        "the app-side re-export",
        findings,
    )

    # -- 3. the build script links only under that same condition -----------
    body_start = kernel_build.find("fn build_vslib()")
    if body_start < 0:
        findings.append("backend/hifishifter-kernel/build.rs: fn build_vslib() is gone")
    else:
        body = kernel_build[body_start:]
        windows_at = body.find('contains("windows")')
        arch_at = body.find('contains("x86_64")')
        link_at = body.find(LINK_DIRECTIVE)
        if windows_at < 0 or arch_at < 0:
            findings.append(
                "backend/hifishifter-kernel/build.rs: build_vslib() no longer checks that "
                'TARGET contains "windows" AND "x86_64"; it would hand rustc an x64 import '
                "library on targets that cannot link it"
            )
        elif link_at < 0:
            findings.append(
                "backend/hifishifter-kernel/build.rs: build_vslib() no longer emits "
                "%s; vslib.rs's #[link] then has no search path anywhere" % LINK_DIRECTIVE
            )
        elif link_at < max(windows_at, arch_at):
            findings.append(
                "backend/hifishifter-kernel/build.rs: %s is emitted BEFORE the "
                "windows/x86_64 guard, so every target links it" % LINK_DIRECTIVE
            )

    # -- 4. exactly one file declares the link attribute --------------------
    if LINK_ATTR not in ffi:
        findings.append(
            "backend/hifishifter-kernel/src/vslib.rs: %s disappeared; if the DLL is now "
            "loaded at runtime, this whole gate is obsolete and should be deleted" % LINK_ATTR
        )
    for path in sorted(BACKEND.rglob("*.rs")):
        if "target" in path.parts or path == VSLIB_FFI:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(
            line.strip().startswith(LINK_ATTR) for line in text.splitlines()
        ):
            findings.append(
                "%s: also declares %s. Only hifishifter-kernel/src/vslib.rs may carry the "
                "link-time dependency; a second copy links every target that compiles it"
                % (rel(path), LINK_ATTR)
            )
        if re.search(r"^\s*(pub )?mod vslib;", text, re.MULTILINE) and path != KERNEL_LIB:
            findings.append(
                "%s: declares 'mod vslib;' outside hifishifter-kernel/src/lib.rs, where the "
                "target_arch gate lives" % rel(path)
            )

    # -- 5. the app forwards its features instead of inheriting defaults ---
    dependency = re.search(
        r"^hifishifter-kernel\s*=\s*\{(?P<spec>[^}]*)\}", app_manifest, re.MULTILINE
    )
    if dependency is None:
        findings.append("backend/src-tauri/Cargo.toml: the hifishifter-kernel dependency is gone")
    else:
        spec = dependency.group("spec")
        if "default-features = false" not in spec:
            findings.append(
                "backend/src-tauri/Cargo.toml: hifishifter-kernel still inherits the kernel's "
                'default features (["onnx", "vslib", "soundtouch"]). Then '
                '"tauri build -- --no-default-features --features onnx" only disables the app\'s '
                "vslib while the kernel keeps compiling it, which is how the Windows ARM64 leg "
                "ended up linking an x64 import library"
            )
        if '"soundtouch"' not in spec:
            findings.append(
                "backend/src-tauri/Cargo.toml: with default-features = false the kernel no "
                'longer gets "soundtouch"; add it to the dependency features (it is a runtime '
                "libloading dependency now, on for every format)"
            )
    for feature, forwarded in (
        ("onnx", "hifishifter-kernel/onnx"),
        ("vslib", "hifishifter-kernel/vslib"),
    ):
        match = re.search(
            r"^%s\s*=\s*\[(?P<items>[^\]]*)\]" % feature, app_manifest, re.MULTILINE
        )
        if match is None:
            findings.append(
                "backend/src-tauri/Cargo.toml: the app feature '%s' is gone; this gate is stale"
                % feature
            )
        elif '"%s"' % forwarded not in match.group("items"):
            findings.append(
                "backend/src-tauri/Cargo.toml: the app feature '%s' does not forward '%s', so "
                "the kernel's own default no longer follows the app's switch"
                % (feature, forwarded)
            )

    if findings:
        for finding in findings:
            print("check-vslib-arch-gate: %s" % finding, file=sys.stderr)
        return 1
    print(
        "check-vslib-arch-gate: the vslib backend is gated on x86_64 Windows everywhere, "
        "and the app forwards onnx/vslib to the kernel"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
