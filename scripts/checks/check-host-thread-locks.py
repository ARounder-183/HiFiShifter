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
     `AraCallbackGuard` must not take `self.timeline`;
  3. the pending-import reconciler must take **no** table lock at all (`try_lock` only),
     and `present_host_inventory` must drop the `ui_tracks` guard before it calls it.

Shape 3 is the one that cost a host freeze: the inventory pass held `ui_tracks` for its
whole body and the reconciler asked the same map a second time. `std::sync::Mutex` is
not reentrant, so the first inventory pass after an import never returned -- and the
status bar's `plugin_get_apply_state` error was only the 30 s timeout behind it.

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
DOCUMENT = ROOT / "backend" / "hifishifter-plugin" / "src" / "render" / "document.rs"
WORKSPACE = ROOT / "backend" / "hifishifter-plugin" / "src" / "editor" / "workspace.rs"

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


def check_reconciler_takes_no_locks(text: str, label: str) -> list[str]:
    """The reconciler may only *try* its ledger; taking anything else is the freeze again."""
    found = function_body(text, "fn take_resolved_imports(")
    if found is None:
        return [f"{label}: could not find take_resolved_imports (the reconciler)"]
    first_line, body = found
    problems: list[str] = []
    for match in re.finditer(r"\.lock_or_recover\(", body):
        line = first_line + body.count("\n", 0, match.start())
        problems.append(
            f"{label}:{line}: the pending-import reconciler takes a blocking lock; "
            "it runs from the inventory pass and from ARA publish, so it may only try"
        )
    return problems


def check_inventory_drops_the_guard(text: str, label: str) -> list[str]:
    """`present_host_inventory` must release `tracks` before reconciling."""
    found = function_body(text, "fn present_host_inventory(")
    if found is None:
        return [f"{label}: could not find present_host_inventory"]
    first_line, body = found
    call = body.find("publish_pending_imports(")
    if call < 0:
        return [
            f"{label}:{first_line}: present_host_inventory never reconciles the pending "
            "imports; an imported item would stay pending until some other publish"
        ]
    dropped = body.rfind("drop(tracks)", 0, call)
    if dropped < 0:
        line = first_line + body.count("\n", 0, call)
        return [
            f"{label}:{line}: present_host_inventory reconciles while the ui_tracks guard "
            "is still alive; std::sync::Mutex is not reentrant, so this is a host freeze"
        ]
    return []


def run(
    root_client: Path,
    root_ara: Path,
    root_binding: Path,
    root_document: Path = DOCUMENT,
    root_workspace: Path = WORKSPACE,
) -> list[str]:
    problems: list[str] = []
    problems += check_client(root_client.read_text(encoding="utf-8"), str(root_client))
    for path in (root_ara, root_binding):
        problems += check_ara_callbacks(path.read_text(encoding="utf-8"), str(path))
    problems += check_reconciler_takes_no_locks(
        root_document.read_text(encoding="utf-8"), str(root_document)
    )
    problems += check_inventory_drops_the_guard(
        root_workspace.read_text(encoding="utf-8"), str(root_workspace)
    )
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

    planted_reconciler = """
    fn take_resolved_imports(&self, claims: &ImportClaims) -> Option<Vec<ResolvedImport>> {
        let mut imports = self.pending_imports.lock_or_recover();
        None
    }
"""
    if not check_reconciler_takes_no_locks(planted_reconciler, "planted"):
        failures.append("a planted blocking lock in the reconciler was not reported")

    clean_reconciler = """
    fn take_resolved_imports(&self, claims: &ImportClaims) -> Option<Vec<ResolvedImport>> {
        let mut imports = try_lock_or_recover(&self.pending_imports)?;
        None
    }
"""
    if check_reconciler_takes_no_locks(clean_reconciler, "clean"):
        failures.append("a try-lock reconciler was reported as a violation")

    planted_inventory = """
    pub(crate) fn present_host_inventory(&self, timeline: &mut TimelineState, namespace: &str) {
        let tracks = self.ui_tracks.lock_or_recover();
        self.publish_pending_imports(&claims);
    }
"""
    if not check_inventory_drops_the_guard(planted_inventory, "planted"):
        failures.append("an inventory pass that reconciles under its guard was not reported")

    clean_inventory = """
    pub(crate) fn present_host_inventory(&self, timeline: &mut TimelineState, namespace: &str) {
        let tracks = self.ui_tracks.lock_or_recover();
        drop(tracks);
        self.publish_pending_imports(&claims);
    }
"""
    if check_inventory_drops_the_guard(clean_inventory, "clean"):
        failures.append("an inventory pass that drops the guard first was reported")

    for message in failures:
        print(f"self-test: {message}")
    if failures:
        return 1
    print("host-thread lock gate self-test: 8/8")
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
