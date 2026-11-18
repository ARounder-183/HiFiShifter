#!/usr/bin/env python3
"""Cross-platform fast static gate runner (< 5s, zero compilation, no PowerShell needed).

Because GitHub Actions workflows (`build.yml`, `vst3-unix.yml`, `clap.yml`) only build
artifacts and no longer execute unit or static gate workflows (`docs/ci.md`), every
developer and coding agent MUST run this script before committing or pushing.

Usage:
    python3 scripts/checks/run-fast-gates.py
    python3 scripts/checks/run-fast-gates.py --install-pre-push-hook
"""

import argparse
import os
import stat
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CHECKS_DIR, "..", ".."))

STATIC_GATES = [
    ("FFI panic-guard & baseline gate", "check-ffi-guards.py"),
    ("Log level gate", "check-log-levels.py"),
    ("Log verbosity gate", "check-log-verbosity.py"),
    ("Frontend-backend command gating gate", "check-command-gating.py"),
    ("Native file dialog filter gate", "check-dialog-filters.py"),
    ("Source UTF-8 integrity gate", "check-utf8.py"),
    ("SoundTouch CMake generator gate", "check-soundtouch-generator.py"),
    ("Rust test-skip registry gate", "check-test-skips.py"),
]


def resolve_git_hooks_dir() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    common = proc.stdout.strip()
    if not os.path.isabs(common):
        common = os.path.abspath(os.path.join(REPO_ROOT, common))
    return os.path.join(common, "hooks")


def install_pre_push_hook() -> int:
    hooks_dir = resolve_git_hooks_dir()
    os.makedirs(hooks_dir, exist_ok=True)
    hook_path = os.path.join(hooks_dir, "pre-push")
    script_body = (
        "#!/usr/bin/env sh\n"
        "# Installed by scripts/checks/run-fast-gates.py\n"
        "set -eu\n"
        'ROOT="$(git rev-parse --show-toplevel)"\n'
        'if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi\n'
        '"$PY" "$ROOT/scripts/checks/run-fast-gates.py"\n'
    )
    with open(hook_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script_body)
    st = os.stat(hook_path)
    os.chmod(hook_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed pre-push fast gate hook at {hook_path}")
    return 0


def run_all_gates() -> int:
    t0 = time.monotonic()
    failures = []
    for label, script_name in STATIC_GATES:
        script_path = os.path.join(CHECKS_DIR, script_name)
        print(f"== {label} ({script_name})", flush=True)
        res = subprocess.run([sys.executable, script_path], cwd=REPO_ROOT)
        if res.returncode != 0:
            failures.append((label, script_name, res.returncode))

    elapsed = time.monotonic() - t0
    if failures:
        print(f"\nFast static gates FAILED in {elapsed:.2f}s:", file=sys.stderr)
        for label, script_name, code in failures:
            print(f"  - {label} ({script_name}) exited {code}", file=sys.stderr)
        return 1

    print(f"\nAll {len(STATIC_GATES)} fast static gates passed in {elapsed:.2f}s.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-platform HiFiShifter static gates.")
    parser.add_argument(
        "--install-pre-push-hook",
        action="store_true",
        help="Install a git pre-push hook that runs fast static gates automatically.",
    )
    args = parser.parse_args()
    if args.install_pre_push_hook:
        return install_pre_push_hook()
    return run_all_gates()


if __name__ == "__main__":
    sys.exit(main())
