#!/usr/bin/env python3
"""Verify every model in the manifest is actually bundled by the app.

The symptom this prevents is silent and total: a file listed in models.json but
absent from tauri.conf.json's bundle.resources ships in no artefact at all, the
runtime logs one line and reports the backend as available anyway, and pitch
synthesis quietly produces nothing.

Exit code 0 when every model is bundled, 1 otherwise.
"""

from __future__ import annotations

import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAURI_DIR = ROOT / "backend" / "src-tauri"
CONFIG = TAURI_DIR / "tauri.conf.json"
MANIFEST = TAURI_DIR / "resources" / "models" / "models.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    problems = []
    if not CONFIG.is_file():
        print("FAIL: missing " + str(CONFIG), file=sys.stderr)
        return 1
    if not MANIFEST.is_file():
        print("FAIL: missing model manifest " + str(MANIFEST), file=sys.stderr)
        return 1

    config = load_json(CONFIG)
    resources = (config.get("bundle") or {}).get("resources") or {}
    manifest = load_json(MANIFEST)

    entries = []
    for key, value in resources.items():
        if not isinstance(value, str):
            continue
        key_norm = key.replace(chr(92), "/")
        value_norm = value.replace(chr(92), "/")
        if key_norm.startswith("resources/models") or value_norm.startswith("models"):
            entries.append((key_norm, value_norm))

    for key, _value in entries:
        if not (TAURI_DIR / key).resolve().exists():
            problems.append("bundle.resources names a missing path: " + key)

    covered = 0
    total = 0
    for item in manifest.get("files", []):
        relative = str(item.get("path", "")).replace(chr(92), "/")
        if not relative:
            continue
        total += 1
        wanted_source = "resources/models/" + relative
        wanted_dest = "models/" + relative

        exact = [(k, v) for k, v in entries if k == wanted_source]
        if exact:
            if exact[0][1] != wanted_dest:
                problems.append(
                    relative + ": bundled as " + exact[0][1] + ", the runtime looks for " + wanted_dest
                )
            covered += 1
            continue

        directory = [(k, v) for k, v in entries if wanted_source.startswith(k + "/")]
        if directory:
            key, dest = directory[0]
            suffix = wanted_source[len(key):]
            expected = dest.rstrip("/") + suffix
            if expected != wanted_dest:
                problems.append(
                    relative + ": directory entry " + key + " maps to " + expected
                    + ", the runtime looks for " + wanted_dest
                )
            covered += 1
            continue

        problems.append(
            relative + " is in models.json but not in bundle.resources (add \""
            + wanted_source + "\": \"" + wanted_dest + "\")"
        )

    if problems:
        for problem in problems:
            print("FAIL: " + problem, file=sys.stderr)
        print("FAIL: " + str(len(problems)) + " problem(s); " + str(covered) + "/" + str(total) + " covered", file=sys.stderr)
        return 1

    print("OK: all " + str(total) + " model file(s) in models.json are bundled by tauri.conf.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
