#!/usr/bin/env python3
"""Verify every native file dialog carries a type filter from the vocabulary.

The symptoms this prevents are all silent:

  * a save dialog created without SetFileTypes / add_filter shows an empty type
    dropdown, so the user can save a .zip diagnostics bundle as .txt;
  * a filter whose name is a glob ("*.mid") renders as a pattern in the
    dropdown instead of a readable type name;
  * an attachment "save as" with no filter at all accepts any extension.

None of those throw. They look like a working dialog until someone checks the
saved file's extension. This check makes the missing filter a CI failure.

Two surfaces:

  * the plug-in's own pickers (pick_save_path / pick_open_path), which must be
    given a hifishifter_kernel::file_types::FileType;
  * the standalone app's rfd::FileDialog chains, which must call .add_filter
    before terminating (folder pickers excepted - they have no file type).

MAINTENANCE RULE: a new picker entry point (a third helper that ends in a
system dialog) must be added to PICKER_CALLS below. Otherwise this gate reports
"0 call sites" for it and silently stops guarding anything.

Exit code 0 when every dialog is typed, 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

# The plug-in's two "ask the user for a path" entry points.
PICKER_CALLS = ("pick_save_path", "pick_open_path")

# Types from the vocabulary: call sites must hand over one of them
# (or construct a FileType on the spot).
VOCABULARY = (
    "DIAGNOSTICS_ZIP",
    "MIDI",
    "HIFISHIFTER_JSON",
    "NOTEBOOK_MARKDOWN",
    "NOTEBOOK_HTML",
    "HIFISHIFTER_PROJECT",
    "HIFISHIFTER_PROJECT_BACKUP",
    "JSON_PROJECT",
    "REAPER_PROJECT",
    "REAPER_PROJECT_BACKUP",
    "VOCALSHIFTER_PROJECT",
    "WAV_AUDIO",
    "notebook_document",
    "FileType::",
    "file_types::",
)

# rfd terminating calls: a builder chain ends here.
RFD_TERMINATORS = (".save_file(", ".pick_file(", ".pick_files(", ".pick_folder(")

SKIP_DIRS = {"target", "node_modules", ".git", "third-party", "gen"}


def rust_sources():
    out = []
    for path in BACKEND.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def chain_of(source, start):
    """The builder chain starting at start, up to the terminating call."""
    for terminator in RFD_TERMINATORS:
        index = source.find(terminator, start)
        if index >= 0:
            return source[start : index + len(terminator)]
    # No terminating call (or the chain was assigned to a variable) -- cut at
    # the next semicolon so the missing filter is still reported.
    semi = source.find(";", start)
    return source[start : semi if semi >= 0 else len(source)]


def enclosing_signature(source, index):
    """
    The signature of the function enclosing index (from the nearest fn to its
    opening brace).

    [Why this is needed] The non-Windows branch in browser_files.rs is a
    **forwarder**: platform_services::pick_save_path(file_name, file_type). It
    passes the caller-supplied filter through untouched, so of course it does
    not spell out a vocabulary constant itself -- forcing it to would only
    push an extra parameter into the call. As long as **its own** signature
    declares a FileType, the path remains type-safe.
    """
    start = source.rfind("fn ", 0, index)
    if start < 0:
        return ""
    brace = source.find("{", start)
    return source[start : brace if brace >= 0 else index]


def call_arguments(source, start):
    """The argument text of the call at start (extracted by matching parentheses)."""
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return source[start:]


def main():
    violations = []
    checked_pickers = 0
    checked_rfd = 0

    for path in rust_sources():
        source = path.read_text(encoding="utf-8", errors="replace")
        display = path.relative_to(ROOT).as_posix()

        # ── Plug-in: pick_save_path / pick_open_path must get a vocabulary type ──
        for name in PICKER_CALLS:
            for match in re.finditer(r"\b" + name + r"\s*\(", source):
                # The definition site (the `fn pick_save_path(` line) is not
                # a call site.
                prefix = source[max(0, match.start() - 12) : match.start()]
                if prefix.rstrip().endswith("fn"):
                    continue
                checked_pickers += 1
                arguments = call_arguments(source, match.end() - 1)
                signature = enclosing_signature(source, match.start())
                if "FileType" in signature and "FileType" not in arguments:
                    # Forwarder: this function itself declares a FileType
                    # parameter, so the type chain is not broken.
                    continue
                if not any(token in arguments for token in VOCABULARY):
                    violations.append(
                        display
                        + ": "
                        + name
                        + " does not use a type filter from the kernel vocabulary\n    "
                        + arguments.strip()[:160]
                    )

        # ── Standalone app: rfd chain must call add_filter before terminating ──
        for match in re.finditer(r"rfd::FileDialog::new\(\)", source):
            chain = chain_of(source, match.start())
            if ".pick_folder(" in chain:
                continue
            checked_rfd += 1
            if ".add_filter(" not in chain:
                violations.append(
                    display
                    + ": rfd dialog has no add_filter (the type dropdown will be empty)\n    "
                    + chain.strip()[:200]
                )

    print(
        "check-dialog-filters: "
        + str(checked_pickers)
        + " plug-in picker call(s), "
        + str(checked_rfd)
        + " rfd dialog(s)"
    )

    # Sanity check: the scan must actually hit something, otherwise
    # "0 violations" is meaningless.
    if checked_pickers == 0 or checked_rfd == 0:
        print(
            "\nthe scan matched no dialogs -- the paths or criteria are stale; "
            "this is not 'zero violations'"
        )
        return 1

    if violations:
        print("\nthe following file dialogs lack a proper type filter:")
        for violation in violations:
            print("  - " + violation)
        print(
            "\ntype names and extensions come from exactly one source: "
            "backend/hifishifter-kernel/src/file_types.rs."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
