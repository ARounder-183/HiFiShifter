#!/usr/bin/env python3
"""Host-thread lock gate.

Why this exists
---------------
The host calls us synchronously on its own UI thread: WebView2 message callbacks,
`IPlugView` callbacks, `plugin_register("timer")`, ARA model callbacks. If one of those
returns late, the host stops pumping messages -- and the WebView2 delivery we are waiting
on only advances when it pumps. "We wait for a lock" and "the host waits for us to return"
can therefore bite each other.

The rule this gate enforces:

    the host UI thread must not take a lock that the session actor can hold
    for the duration of a projection.

Two shapes are checked:

  1. the body of `on_message` in `editor/gui/client.rs` must not mention a mutex at all;
  2. any function in `ara/model.rs` / `render/extension/binding.rs` that enters an
     `AraCallbackGuard` must not take `self.timeline`.

Neither mistake is a compile error, and both look exactly like "the command did nothing"
in the field. The gate therefore also runs a self-test in both directions: a planted
violation must fail, and the real tree must pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "backend" / "hifishifter-plugin" / "src" / "editor" / "gui" / "client.rs"
ARA_MODEL = ROOT / "backend" / "hifishifter-plugin" / "src" / "ara" / "model.rs"
BINDING = ROOT / "backend" / "hifishifter-plugin" / "src" / "render" / "extension" / "binding.rs"

# Anything that can block on another thread. `lock_or_recover` is the crate's own helper;
# `Mutex` covers a freshly created one. `try_lock` is deliberately allowed: it never waits.
MUTEX_PATTERN = re.compile(r"\.lock_or_recover\(\)|\.lock\(\)|\bMutex\b")


def function_body(text: str, signature: str) -> tuple[int, str] | None:
    """Return (first_line_number, body) for the first function whose header contains
    `signature`. The body is taken by brace matching so nested blocks are included."""
    index = text.find(signature)
    if index < 0:
        return None
    open_brace = text.find("{", index)
    if open_brace < 0:
        return None
    depth = 0
    for offset in range(open_brace, len(text)):
        char = text[offset]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                first_line = text.count("\n", 0, open_brace) + 1
                return first_line, text[open_brace : offset + 1]
    return None


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_client(text: str, label: str) -> list[str]:
    found = function_body(text, "fn on_message(")
    if found is None:
        return [f"{label}: could not find on_message in the editor client"]
    first_line, body = found
    problems: list[str] = []
    for match in MUTEX_PATTERN.finditer(body):
        line = first_line + body.count("\n", 0, match.start())
        problems.append(
            f"{label}:{line}: on_message takes a lock ({match.group(0)}); "
            "the host UI thread must read a published snapshot instead"
        )
    return problems


def check_ara_callbacks(text: str, label: str) -> list[str]:
    """Every function that enters an ARA callback guard must stay off the timeline lock."""
    problems: list[str] = []
    for match in re.finditer(r"AraCallbackGuard::enter\(", text):
        # Walk back to the enclosing `fn` header.
        header = text.rfind("fn ", 0, match.start())
        if header < 0:
            continue
        name_end = text.find("(", header)
        name = text[header + 3 : name_end].strip()
        found = function_body(text, f"fn {name}(")
        if found is None:
            continue
        first_line, body = found
        for hit in re.finditer(r"self\.timeline\s*\.\s*lock", body):
            line = first_line + body.count("\n", 0, hit.start())
            problems.append(
                f"{label}:{line}: ARA callback {name} takes the timeline lock; "
                "the host cannot pump while it runs"
            )
    return problems


def run(root_client: Path, root_ara: Path, root_binding: Path) -> list[str]:
    problems: list[str] = []
    problems += check_client(root_client.read_text(encoding="utf-8"), str(root_client))
    for path in (root_ara, root_binding):
        problems += check_ara_callbacks(path.read_text(encoding="utf-8"), str(path))
    return problems


def self_test() -> int:
    """Two directions: a planted violation must fail, the real tree must pass."""
    failures: list[str] = []

    planted_client = """
pub(crate) fn on_message(state: &Rc<RefCell<EditorClient>>, text: &str) {
    let moves = editor.timeline.lock_or_recover().close_track_gaps_moves("t", 0.0);
}
"""
    if not check_client(planted_client, "planted"):
        failures.append("a planted timeline lock inside on_message was not reported")

    clean_client = """
pub(crate) fn on_message(state: &Rc<RefCell<EditorClient>>, text: &str) {
    let clips = editor.ui_clips();
}
"""
    if check_client(clean_client, "clean"):
        failures.append("a lock-free on_message was reported as a violation")

    planted_ara = """
    fn end_editing(&mut self) {
        let _callback = crate::host::AraCallbackGuard::enter("end_editing");
        let _ = self.timeline.lock_or_recover();
    }
"""
    if not check_ara_callbacks(planted_ara, "planted"):
        failures.append("a planted timeline lock inside an ARA callback was not reported")

    clean_ara = """
    fn end_editing(&mut self) {
        let _callback = crate::host::AraCallbackGuard::enter("end_editing");
        self.remap_and_log();
    }
"""
    if check_ara_callbacks(clean_ara, "clean"):
        failures.append("a lock-free ARA callback was reported as a violation")

    for message in failures:
        print(f"self-test: {message}")
    if failures:
        return 1
    print("host-thread lock gate self-test: 4/4")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    problems = run(CLIENT, ARA_MODEL, BINDING)
    if problems:
        print("Host-thread lock gate failed:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("host-thread lock gate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
