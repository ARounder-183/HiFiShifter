#!/usr/bin/env python3
"""Every resource a bundle config declares must exist on the target it is built for.

Tauri validates bundle.resources at **build-script** time, not at bundling time: a declared
source that the target does not produce kills the compile of the app crate with

    resource path resources/linux/libwebgpu_dawn.so doesn't exist
    error: failed to run custom build command for HiFiShifter v0.2.0-beta

...and the AppDir is never created, so the failure surfaces in CI as
"ERROR: AppDir not found" -- a message that names neither the file nor the platform. That is
exactly how the 3ac5ed76 release run lost its linux-arm64 and windows-arm64 legs (jobs
138652036166 / 138652036167).

Why the declaration has to be split rather than "just make the file exist": ORT on Linux ARM64
ships no dawn, and aarch64 Windows has no DirectML. A placeholder would ship a zero-byte shared
library that is never loaded, which turns a build-time error into a runtime one the first time
somebody does load it. So the resource moves into an architecture-scoped config file
(tauri.<platform>.<arch>.conf.json) that only the matching leg passes with --config.

Four assertions, in the order a reviewer would ask them:

1. the config files are named and parse as the schema says;
2. every known architecture-specific source lives in exactly its arch config and nowhere else
   (this is the regression that broke the release);
3. every arch config is actually passed by some build entry point -- a config file nobody
   references is dead, and the invariant silently stops holding;
4. the merged resource map per (platform, arch) has no duplicate destinations.

Exit code 0 when all four hold.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_TAURI = ROOT / "backend" / "src-tauri"
PLATFORMS = ("windows", "linux", "macos")
ARCHES = ("x86_64", "aarch64")
ARCH_CONFIG = re.compile(r"^tauri\.(?P<platform>[a-z]+)\.(?P<arch>[a-z0-9_]+)\.conf\.json$")

# Sources that only some targets produce. Key = the path as written in the config (relative to
# backend/src-tauri), value = (platform, arch, why).
#
# [Why resources/DirectML.dll is deliberately NOT in this table] It looks arch-specific -- the
# obvious guess is "DirectML is x64 only" -- but ort-sys' distribution table (build/download/
# dist.tsv in ort-sys 2.0.0-rc.13) has an aarch64-pc-windows-msvc+directml package, and
# Cargo.toml asks for the directml feature on every Windows target, so build.rs stages the DLL on
# arm64 as well. Splitting it out would have meant every plain cargo-tauri build on x64 Windows
# silently stopped bundling it. The criterion here is "the target really does not produce it",
# not "the name sounds like an architecture".
ARCH_SPECIFIC: dict[str, tuple[str, str, str]] = {
    "resources/linux/libwebgpu_dawn.so": (
        "linux",
        "x86_64",
        "ORT's Linux ARM64 package deliberately ships no dawn (there is no prebuilt WebGPU "
        "provider for it), so resources/linux/ stays empty there and declaring the file aborts "
        "tauri-build before the AppDir exists.",
    ),
}

# Entry points that may pass --config. Used by assertion 3.
REFERENCE_GLOBS = (
    ".github/workflows/*.yml",
    "scripts/**/*.sh",
    "scripts/**/*.ps1",
    "docs/*.md",
    "README.md",
)


def load(name: str) -> dict:
    path = SRC_TAURI / name
    if not path.is_file():
        raise SystemExit(f"check-tauri-resources: missing {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"check-tauri-resources: {name} is not valid JSON: {exc}") from exc


def resources_of(config: dict) -> dict[str, str]:
    """Normalises bundle.resources (list or map form) to a {source: destination} map."""
    raw = (config.get("bundle") or {}).get("resources")
    if raw is None:
        return {}
    if isinstance(raw, list):
        # The list form maps each source onto the same relative path inside the bundle.
        return {str(entry): str(entry) for entry in raw}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    raise SystemExit(f"check-tauri-resources: unsupported bundle.resources shape: {type(raw)}")


def main() -> int:
    problems: list[str] = []

    base = load("tauri.conf.json")
    base_resources = resources_of(base)
    platform_configs: dict[str, dict[str, str]] = {}
    arch_configs: dict[tuple[str, str], dict[str, str]] = {}
    arch_config_files: dict[tuple[str, str], str] = {}

    for path in sorted(SRC_TAURI.glob("tauri.*.conf.json")):
        name = path.name
        if name in ("tauri.conf.json", "tauri.conf.dist-dev.json"):
            continue
        match = ARCH_CONFIG.match(name)
        if match:
            platform, arch = match.group("platform"), match.group("arch")
            if platform not in PLATFORMS:
                problems.append(
                    f"{name}: unknown platform '{platform}' (expected one of {', '.join(PLATFORMS)})"
                )
                continue
            if arch not in ARCHES:
                problems.append(
                    f"{name}: unknown architecture '{arch}' (expected one of {', '.join(ARCHES)})"
                )
                continue
            if (platform, arch) in arch_configs:
                problems.append(
                    f"{name}: a second architecture-scoped config for {platform}/{arch}"
                )
                continue
            arch_configs[(platform, arch)] = resources_of(load(name))
            arch_config_files[(platform, arch)] = name
            continue
        # tauri.<platform>.conf.json
        if name.startswith("tauri.") and name.endswith(".conf.json"):
            middle = name[len("tauri.") : -len(".conf.json")]
            if middle in PLATFORMS:
                platform_configs[middle] = resources_of(load(name))
                continue
        problems.append(f"{name}: not a recognised tauri config name")

    # ── 2. architecture-specific sources live in exactly one place ──────────
    for source, (platform, arch, why) in ARCH_SPECIFIC.items():
        owner = arch_configs.get((platform, arch), {})
        if source not in owner:
            expected = arch_config_files.get(
                (platform, arch), f"tauri.{platform}.{arch}.conf.json"
            )
            problems.append(
                f"{source}: must be declared in {expected} -- {why}"
            )
        for other_platform, entries in platform_configs.items():
            if source in entries:
                problems.append(
                    f"{source}: declared in the generic {other_platform} config, so every "
                    f"{other_platform} target must produce it, but it exists on {platform}/{arch} "
                    f"only. {why}"
                )
        for key, entries in arch_configs.items():
            if key != (platform, arch) and source in entries:
                problems.append(
                    f"{source}: also declared in the {key[0]}/{key[1]} config; it belongs to "
                    f"{platform}/{arch} only"
                )

    # ── 3. no dead arch config ─────────────────────────────────────────────
    corpus = ""
    for pattern in REFERENCE_GLOBS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            try:
                corpus += path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    for (platform, arch), name in arch_config_files.items():
        if name not in corpus:
            problems.append(
                f"{name}: no build entry point passes it. Add --config {name} to the "
                f"{platform}/{arch} leg (and to the local script), otherwise the resources it "
                "declares are simply never bundled."
            )

    # ── 4. no duplicate destinations in a merged view ──────────────────────
    merged_report: list[str] = []
    for platform in PLATFORMS:
        for arch in ARCHES:
            merged: dict[str, str] = dict(base_resources)
            merged.update(platform_configs.get(platform, {}))
            merged.update(arch_configs.get((platform, arch), {}))
            destinations: dict[str, str] = {}
            for source, destination in merged.items():
                previous = destinations.get(destination)
                if previous is not None and previous != source:
                    problems.append(
                        f"{platform}/{arch}: {previous} and {source} both bundle to {destination}"
                    )
                destinations[destination] = source
            merged_report.append(f"  {platform}/{arch}: {len(merged)} resource(s)")

    # ── 2b. an arch config may only carry sources the table knows about ────
    for (platform, arch), entries in arch_configs.items():
        for source in entries:
            if source not in ARCH_SPECIFIC:
                problems.append(
                    f"{arch_config_files[(platform, arch)]}: {source} is declared "
                    "per-architecture but is not listed in ARCH_SPECIFIC, so nothing checks that "
                    "the generic config stays free of it. Add it there with the reason it cannot "
                    "be shared."
                )

    if problems:
        print("check-tauri-resources: bundle resource declarations do not match the targets:")
        for line in problems:
            print(f"  - {line}")
        print(
            "\nAn architecture-scoped config is tauri.<platform>.<arch>.conf.json and must be\n"
            "passed with --config by that leg only. Tauri deep-merges it over tauri.conf.json and\n"
            "tauri.<platform>.conf.json."
        )
        return 1

    print("check-tauri-resources: every declared bundle resource is produced by its target")
    for line in merged_report:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
