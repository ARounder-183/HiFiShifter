#!/usr/bin/env python3
"""Change-scoped verification router for human developers and parallel coding agents.

Inspects `git diff` (working tree + index, or `--base <rev>`) and computes the minimal,
feature-isolated verification commands:
  1. Always includes `run-fast-gates.py`.
  2. Maps touched `backend/<crate>/` paths to targeted `cargo test --locked` runs while
     preserving the shell feature-isolation boundary (`hifishifter-clap` and
     `hifishifter-vst3` are NEVER built inside the same cargo graph as `HiFiShifter`
     `src-tauri`, avoiding Cargo `vslib` feature-union contamination).
  3. Applies platform-appropriate `--skip` entries from `scripts/checks/test-skips.txt`.
  4. Maps touched `frontend/` paths to `typecheck` and targeted `vitest` suites.

Usage:
    python3 scripts/checks/verify-changed.py --print-only
    python3 scripts/checks/verify-changed.py --base HEAD~1 --run
"""

import argparse
import os
import shlex
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CHECKS_DIR, "..", ".."))
SKIPS_FILE = os.path.join(CHECKS_DIR, "test-skips.txt")


def current_platform() -> str:
    return "windows" if os.name == "nt" else "unix"


def load_test_skips() -> list[str]:
    plat = current_platform()
    skips: list[str] = []
    if not os.path.isfile(SKIPS_FILE):
        return skips
    with open(SKIPS_FILE, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entry_plat = ""
            if line.startswith("[unix]"):
                entry_plat = "unix"
                line = line[len("[unix]") :].strip()
            elif line.startswith("[windows]"):
                entry_plat = "windows"
                line = line[len("[windows]") :].strip()
            if entry_plat and entry_plat != plat:
                continue
            name = line.split(" — ")[0].strip()
            if name:
                skips.append(name)
    return skips


def changed_files(base: str | None) -> list[str]:
    files: set[str] = set()
    if base:
        proc = subprocess.run(
            ["git", "diff", "--name-only", base],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files.update(l.strip() for l in proc.stdout.splitlines() if l.strip())
    else:
        for cmd in (
            ["git", "diff", "--name-only"],
            ["git", "diff", "--cached", "--name-only"],
            ["git", "ls-files", "--others", "--exclude-standard"],
        ):
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
            files.update(l.strip() for l in proc.stdout.splitlines() if l.strip())
    return sorted(f.replace("\\", "/") for f in files)


def build_plan(files: list[str]) -> list[list[str]]:
    cmds: list[list[str]] = [[sys.executable, "scripts/checks/run-fast-gates.py"]]
    if not files:
        return cmds

    skips = load_test_skips()
    skip_flags: list[str] = []
    for s in skips:
        skip_flags.extend(["--skip", s])

    crates: set[str] = set()
    mcp_only_in_app = True
    touched_app = False
    touched_frontend = False
    frontend_tests: list[str] = []

    crate_map = {
        "backend/hifishifter-mcp/": "hifishifter-mcp",
        "backend/hifishifter-kernel/": "hifishifter-kernel",
        "backend/hifishifter-plugin/": "hifishifter-plugin",
        "backend/hifishifter-plugin-core/": "hifishifter-plugin-core",
        "backend/hifishifter-ara/": "hifishifter-ara",
        "backend/hifishifter-vst3/": "hifishifter-vst3",
        "backend/hifishifter-clap/": "hifishifter-clap",
        "backend/hifishifter-capture/": "hifishifter-capture",
        "backend/hifishifter-host-link/": "hifishifter-host-link",
        "backend/hifishifter-clipboard/": "hifishifter-clipboard",
    }

    for path in files:
        for prefix, pkg in crate_map.items():
            if path.startswith(prefix):
                crates.add(pkg)
        if path.startswith("backend/src-tauri/"):
            touched_app = True
            if not (
                path.startswith("backend/src-tauri/src/mcp/")
                or path in ("backend/src-tauri/build.rs", "backend/src-tauri/src/main.rs", "backend/src-tauri/src/lib.rs")
            ):
                mcp_only_in_app = False
        if path.startswith("frontend/"):
            touched_frontend = True
            rel = path[len("frontend/") :]
            if rel.endswith((".test.ts", ".test.tsx")):
                frontend_tests.append(rel)
            elif rel.startswith("src/") and rel.endswith((".ts", ".tsx")):
                stem = rel.rsplit(".", 1)[0]
                for ext in (".test.ts", ".test.tsx"):
                    cand = os.path.join(REPO_ROOT, "frontend", stem + ext)
                    if os.path.isfile(cand):
                        frontend_tests.append(stem + ext)

    # Run independent library crates first, keeping shells isolated from src-tauri.
    for pkg in sorted(crates - {"hifishifter-vst3", "hifishifter-clap"}):
        cmds.append(
            ["cargo", "test", "--locked", "--manifest-path", "backend/Cargo.toml", "-p", pkg, "--", "--test-threads=1"]
            + skip_flags
        )

    if touched_app:
        if mcp_only_in_app:
            cmds.append(
                [
                    "cargo",
                    "test",
                    "--locked",
                    "--manifest-path",
                    "backend/Cargo.toml",
                    "-p",
                    "HiFiShifter",
                    "--no-default-features",
                    "--lib",
                    "mcp::",
                ]
            )
        else:
            cmds.append(
                [
                    "cargo",
                    "test",
                    "--locked",
                    "--manifest-path",
                    "backend/Cargo.toml",
                    "-p",
                    "HiFiShifter",
                    "--",
                    "--test-threads=1",
                ]
                + skip_flags
            )

    for shell_pkg in ("hifishifter-vst3", "hifishifter-clap"):
        if shell_pkg in crates:
            cmds.append(["cargo", "build", "--locked", "--manifest-path", "backend/Cargo.toml", "-p", shell_pkg])
            cmds.append(
                [
                    "cargo",
                    "test",
                    "--locked",
                    "--manifest-path",
                    "backend/Cargo.toml",
                    "-p",
                    shell_pkg,
                    "--",
                    "--test-threads=1",
                ]
                + skip_flags
            )

    if touched_frontend:
        cmds.append(["npm", "--prefix", "frontend", "run", "typecheck"])
        cmds.append(["npm", "--prefix", "frontend", "run", "format:check"])
        dedup_tests = sorted(set(frontend_tests))
        if dedup_tests:
            cmds.append(["npm", "--prefix", "frontend", "run", "test", "--"] + dedup_tests)
        else:
            cmds.append(["npm", "--prefix", "frontend", "run", "test"])

    return cmds


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute or run change-scoped HiFiShifter verification.")
    parser.add_argument("--base", help="Optional git revision to diff against (default: working tree + index).")
    parser.add_argument("--run", action="store_true", help="Execute the planned commands sequentially.")
    parser.add_argument("--print-only", action="store_true", help="Print planned commands without running.")
    args = parser.parse_args()

    files = changed_files(args.base)
    print(f"# Changed files ({len(files)}):")
    for f in files[:30]:
        print(f"#   {f}")
    if len(files) > 30:
        print(f"#   ... and {len(files) - 30} more")

    plan = build_plan(files)
    print("\n# Planned verification commands:")
    for cmd in plan:
        print(" ".join(shlex.quote(part) for part in cmd))

    if not args.run or args.print_only:
        return 0

    env = os.environ.copy()
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    if os.path.isdir(cargo_bin) and cargo_bin not in env.get("PATH", ""):
        env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")

    for cmd in plan:
        print(f"\n==> Running: {' '.join(shlex.quote(p) for p in cmd)}", flush=True)
        res = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if res.returncode != 0:
            print(f"Command failed with exit code {res.returncode}", file=sys.stderr)
            return res.returncode

    print("\nAll change-scoped verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
