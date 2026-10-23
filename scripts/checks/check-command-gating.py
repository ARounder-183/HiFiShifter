#!/usr/bin/env python3
"""Verify the frontend invoke surface matches a real backend command.

The frontend reaches the backend through `frontend/src/services/invoke.ts`, whose
switch maps a method name to its named arguments. A method that no backend
dispatches is a *silent* dead control: the UI renders, the click produces a
promise rejection nobody surfaces. This check makes that class of drift fail in
CI instead of in a user's session.

Four surfaces exist, and all are legitimate:

  * the standalone app registers its commands in `backend/src-tauri/src/lib.rs`
    (`tauri::generate_handler![...]`);
  * the ARA plugin dispatches its own in `editor/commands.rs` (actor thread),
    `editor/webview.rs` (UI thread, for anything needing an HWND), and
    `capture/commands.rs` (forwarded from webview via `is_capture_command()`).

A frontend method must appear in at least one of them. The reverse direction is
deliberately *not* checked: a backend command with no frontend caller is normal
(plugin-only control paths, diagnostic entry points).

MAINTENANCE RULE: a new dispatch surface (a module with its own
`"command" => ...` match) MUST be registered in the constants below — the same
obligation `check-ffi-guards.py` documents for its SCAN_ROOTS. An unregistered
surface makes this gate report live commands as dead ones, which trains people
to ignore the gate.

Exit code 0 when every method is accounted for, 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
INVOKE = ROOT / "frontend" / "src" / "services" / "invoke.ts"
APP_LIB = ROOT / "backend" / "src-tauri" / "src" / "lib.rs"
PLUGIN_ACTOR = ROOT / "backend" / "hifishifter-plugin" / "src" / "editor" / "commands.rs"
PLUGIN_WEBVIEW = ROOT / "backend" / "hifishifter-plugin" / "src" / "editor" / "webview.rs"
PLUGIN_CAPTURE = ROOT / "backend" / "hifishifter-plugin" / "src" / "capture" / "commands.rs"


def read(path: pathlib.Path) -> str:
    if not path.is_file():
        raise SystemExit(f"check-command-gating: missing {path}")
    return path.read_text(encoding="utf-8")


def frontend_methods(source: str) -> set[str]:
    """Method names the frontend can ask for: switch arms + the no-arg allow-list."""
    methods = set(re.findall(r'^\s*case "([A-Za-z0-9_]+)":', source, re.MULTILINE))
    # `NO_ARG_COMMANDS` is a Set literal of bare string entries.
    block = re.search(
        r"NO_ARG_COMMANDS[^=]*=\s*new Set\(\[(.*?)\]\)", source, re.DOTALL
    )
    if block:
        methods |= set(re.findall(r'"([A-Za-z0-9_]+)"', block.group(1)))
    return methods


def app_commands(source: str) -> set[str]:
    """Commands registered with Tauri's `generate_handler!` macro."""
    block = re.search(r"generate_handler!\[(.*?)\]", source, re.DOTALL)
    if not block:
        return set()
    # Entries look like `commands::name` or `commands::sub::name`.
    return {entry.rsplit("::", 1)[-1] for entry in re.findall(r"([A-Za-z0-9_:]+)\s*,", block.group(1))}


def plugin_commands(*sources: str) -> set[str]:
    """Commands the plugin dispatches on either thread."""
    commands: set[str] = set()
    for source in sources:
        commands |= set(re.findall(r'"([A-Za-z0-9_]+)"\s*=>', source))
        commands |= set(re.findall(r'Some\("([A-Za-z0-9_]+)"\)', source))
        commands |= set(re.findall(r'Some\(command @ \(([^)]*)\)\)', source))
        # The multi-arm forms above only capture the first alternative; collect the
        # quoted names inside every `Some(command @ (...))` group separately.
        for group in re.findall(r"Some\((?:command @ )?\(([^)]*)\)\)", source):
            commands |= set(re.findall(r'"([A-Za-z0-9_]+)"', group))
        commands |= set(re.findall(r'Some\("([A-Za-z0-9_]+)"\)\s*\|', source))
    return commands


def main() -> int:
    methods = frontend_methods(read(INVOKE))
    app = app_commands(read(APP_LIB))
    plugin = plugin_commands(
        read(PLUGIN_ACTOR), read(PLUGIN_WEBVIEW), read(PLUGIN_CAPTURE)
    )
    known = app | plugin

    missing = sorted(methods - known)
    print(
        f"check-command-gating: {len(methods)} frontend method(s); "
        f"app={len(app)} plugin={len(plugin)}"
    )
    if missing:
        print("\nfrontend invokes commands no backend dispatches:")
        for name in missing:
            print(f"  - {name}")
        print(
            "\nEither wire the command into the plugin/app, or stop invoking it. "
            "A control that only rejects at runtime is worse than a disabled one."
        )
        return 1
    print("every frontend command is dispatched by the app or the plugin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
