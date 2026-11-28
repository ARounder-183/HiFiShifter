#!/usr/bin/env python3
"""Fetch / package model weights: three acquisition paths + one publishing path,
each verified file by file.

[Why the weights are not in the repository] The weight package would make a single
checkout huge, and every CI job checks out once (about 11 times per PR). Only the
manifest stays in the repository:
`backend/src-tauri/resources/models/models.json`.

[Why .py instead of .ps1] One implementation must run on three platforms
(build.ps1 / vst3/unix/build.sh / CI). This repo's gate scripts
(scripts/checks/check-*.py) already depend on Python, the runners on all three
platforms preinstall it, and no platform needs an extra install. Two
implementations (ps1 + sh) would only diverge, and the consequence of divergence
is "one platform's build quietly using stale weights".

[Completeness criteria match the runtime] Compare **byte count + sha256** per
file; the source of truth is models.json (the same manifest as
model_store::looks_complete; that one compares only byte counts because it runs
on the per-startup path, while this one can afford the cost of hashing). The
version is content-derived, so a wrongly fetched version is guaranteed to
surface here.

Usage (fetch weights):
    python scripts/setup/fetch-models.py                # fetch what is missing (default: from release assets)
    python scripts/setup/fetch-models.py --check        # check only; exit code 1 if anything is missing
    python scripts/setup/fetch-models.py --source DIR   # copy from a local directory (existing weights)
    python scripts/setup/fetch-models.py --archive ZIP  # extract from a local zip
    python scripts/setup/fetch-models.py --fetch-game    # fetch the GAME small ONNX
    python scripts/setup/fetch-models.py --base-url URL # use another download source (mirror)

Usage (publish new weights, a few times a year; process in docs/models.md):
    python scripts/setup/fetch-models.py --write-manifest      # rewrite models.json from the real files on disk
    python scripts/setup/fetch-models.py --check-manifest      # only check that models.json is current (used by CI)
    python scripts/setup/fetch-models.py --package             # build a reproducible release asset
    python scripts/setup/fetch-models.py --verify-archive ZIP  # verify a zip against this manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

# On Windows, stdout follows the ANSI code page (cp1252 on English systems, GBK on
# Chinese ones); printing unmappable characters kills the process with a
# UnicodeEncodeError -- not a single real finding reaches the log, and what fails
# is the encoding, not the rule.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "backend" / "src-tauri" / "resources" / "models"
MANIFEST = MODELS_DIR / "models.json"
# Assets awaiting release go to dist/ (same place as the other artifacts, see
# scripts/README.md). Builds only reset dist/_stage, so what lives here is not
# wiped by the next build.
ASSET_DIR = REPO_ROOT / "dist" / "models"
# [Why zip instead of tar.zst] All three platforms can unpack it
# (Expand-Archive / zipfile / unzip); no extra zstd binary is needed. For a one-time
# download the compression-ratio difference is negligible.
DEFAULT_BASE_URL = "https://github.com/ARounder-183/HiFiShifter/releases/download"

# GAME automatic note-segmentation weights. Upstream publishes them as a model
# package; fetch and verify them separately, then write them into the same
# models.json as the other models. That avoids committing the weights to the repo.
GAME_RELEASE_URL = (
    "https://github.com/openvpi/GAME/releases/download/v1.0.3/"
    "GAME-1.0.3-small-onnx.zip"
)
GAME_RELEASE_DIR = "GAME-1.0.3-small-onnx"
GAME_FILES = (
    "bd2dur.onnx",
    "config.json",
    "dur2bd.onnx",
    "encoder.onnx",
    "estimator.onnx",
    "segmenter.onnx",
)

# Build artifacts allowed to exist in the directory but excluded from both the
# manifest and the assets: macOS's build.rs rewrites pc_nsf_hifigan.onnx into a
# CoreML variant (backend/src-tauri/build.rs::
# generate_coreml_model_variant), and the output lands in the same resource tree.
# The build generates it automatically -- listing it would mean Windows/Linux
# installing an unusable weight set, and the version would depend on
# "whether the build environment ever compiled the macOS target". Packaging takes
# files from the manifest, so it is naturally excluded; this set only stops it
# from being reported as a "stray file".
GENERATED_ARTIFACTS = frozenset({"nsf_hifigan/pc_nsf_hifigan_coreml.onnx"})

# GAME is an optional model that can be enabled on its own: on a machine that only
# verified GAME, the old models being missing is normal. Release assets must still
# contain every manifest entry; only the "is the local copy usable" check skips it.
OPTIONAL_MODEL_PREFIXES = ("GAME/",)

# Timestamp of every entry inside the asset. Pinned to the smallest value a zip can
# represent: "packaging the same weights twice" is then byte-identical, which is
# what qualifies sha256 as the release artifact's identity.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)

# Files that do not participate in the manifest: this script's own output plus known
# build artifacts. Both are excluded by **file name** -- matching the historical
# (PowerShell) implementation, so the same directory tree yields the same manifest
# here and there.
MANIFEST_EXCLUDED_NAMES = frozenset({"models.json"}) | frozenset(
    entry.rsplit("/", 1)[-1] for entry in GENERATED_ARTIFACTS
)


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"Model manifest missing: {MANIFEST} (it must be in the repository)")
    with MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sort_key(entry: dict) -> str:
    """Manifest sort key: full-path form (the separator is a backslash).

    [Why backslash] The version is the hash of the "path + sha256" lines joined in
    order; a different order is a different version. The historical implementation
    sorted by PowerShell's "full path", and in a full path only the separator is a
    backslash -- aligning with it lets the existing models.json match byte for byte
    across both implementations.
    """
    return entry["path"].replace("/", "\\")


def derive_version(files: list[dict]) -> str:
    """Recompute the version: sort all (path, sha256) pairs, join them, take the first
    12 hex digits of the sha256.

    [Why every path recomputes it] version is the **only field in the manifest that
    must never be hand-written**, and it also decides the release tag and the CI
    cache key. Generation, verification, and packaging all recompute it from
    content, so a hand-edited manifest fails on the spot instead of becoming an
    asset "whose version has nothing to do with its contents" and being published.
    """
    entries = sorted(files, key=manifest_sort_key)
    lines = "\n".join(f"{entry['path']} {entry['sha256']}" for entry in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:12]


def collect_manifest_entries() -> list[dict]:
    """Scan the weights directory and return manifest entries (path / bytes / uppercase
    sha256).

    [Why sha256 is uppercase] The historical manifest was written by PowerShell's
    Get-FileHash, which emits uppercase. Comparisons normalize to lowercase (see
    verify_hashes), but the **stored** form stays as-is so the same weights do not
    diff wholesale between two generations.
    """
    existing = load_manifest()["files"] if MANIFEST.is_file() else []
    entries = [
        {
            "path": path.relative_to(MODELS_DIR).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path).upper(),
        }
        for path in MODELS_DIR.rglob("*")
        if path.is_file() and path.name not in MANIFEST_EXCLUDED_NAMES
    ]
    # A machine often caches only part of the weights (e.g. only GAME was verified).
    # The manifest is the "complete downloadable set", not a snapshot of the current
    # working tree: missing old entries are kept as-is and verified again the next
    # time the weights are fetched.
    present = {entry["path"] for entry in entries}
    entries.extend(entry for entry in existing if entry["path"] not in present)
    entries.sort(key=manifest_sort_key)
    return entries


def write_manifest() -> int:
    """Rewrite models.json from the real files on disk (version = content hash)."""
    entries = collect_manifest_entries()
    if not entries:
        print(
            f"No model files to register in {MODELS_DIR}: run python scripts/setup/fetch-models.py first",
            file=sys.stderr,
        )
        return 1
    version = derive_version(entries)
    text = json.dumps({"version": version, "files": entries}, ensure_ascii=False, indent=4) + "\n"
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"Wrote {MANIFEST} (version {version}, {len(entries)} files)")
    return 0


def check_manifest() -> int:
    """Compare versions only: the version is the hash of the manifest contents, so an
    equal version guarantees an equal manifest.

    [Why missing weights counts as a pass] A clean clone has only models.json --
    that is the normal state, not a regression. Reconciling the manifest with the
    weights is only possible after the weights have been fetched.
    """
    if not MANIFEST.is_file():
        print(f"Model manifest missing: {MANIFEST} (it must be in the repository)", file=sys.stderr)
        return 1
    entries = collect_manifest_entries()
    if not entries:
        print("models.json check skipped: weights not fetched yet (run scripts/setup/fetch-models.py)")
        return 0
    manifest = load_manifest()
    expected = derive_version(entries)
    if manifest["version"] != expected:
        print(
            f"models.json is stale (manifest {manifest['version']}, recomputed {expected}): "
            "run python scripts/setup/fetch-models.py --write-manifest",
            file=sys.stderr,
        )
        return 1
    print(f"models.json is up to date (version {expected})")
    return 0


def missing_files(models_dir: Path, manifest: dict) -> list[str]:
    """Return entries that are incomplete (missing file / byte count mismatch).

    [Why byte counts are compared first] Hashing all the weights takes hundreds of
    milliseconds to a few seconds, and this script runs before every build. A
    byte-count mismatch already blocks the vast majority of failures (unfinished
    extraction, half-copied files, mixed versions); the real hash verification is
    the "after download / extraction" step (see verify_hashes).
    """
    problems: list[str] = []
    for entry in manifest["files"]:
        path = models_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size {entry['path']}: {path.stat().st_size} != {entry['bytes']}")
    return problems


def verify_hashes(models_dir: Path, manifest: dict) -> list[str]:
    problems: list[str] = []
    for entry in manifest["files"]:
        if entry["path"].startswith(OPTIONAL_MODEL_PREFIXES):
            continue
        path = models_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
            continue
        actual = sha256_of(path)
        # [Why compare lowercase] The manifest stores uppercase hex (the historical
        # implementation was PowerShell's Get-FileHash), while hexdigest() emits
        # lowercase. Both sides are the same hash; comparing characters directly
        # would mark every **correct** copy of the weights as "verification failed".
        if actual != entry["sha256"].lower():
            problems.append(f"sha256 {entry['path']}: {actual} != {entry['sha256']}")
    return problems


def verify_optional_game_hashes(manifest: dict) -> list[str]:
    """If the optional GAME files exist, verify them file by file; do not error when they are absent."""
    problems: list[str] = []
    for entry in manifest["files"]:
        if not entry["path"].startswith(OPTIONAL_MODEL_PREFIXES):
            continue
        path = MODELS_DIR / entry["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size {entry['path']}")
            continue
        if sha256_of(path) != entry["sha256"].lower():
            problems.append(f"sha256 {entry['path']}")
    return problems


def stray_files(models_dir: Path, manifest: dict) -> list[str]:
    """Files neither listed in the manifest nor known build artifacts -- usually "a new model file was never added to the manifest"."""
    listed = {entry["path"] for entry in manifest["files"]}
    found: list[str] = []
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(models_dir).as_posix()
        if relative in listed or relative in GENERATED_ARTIFACTS or relative == "models.json":
            continue
        found.append(relative)
    return found


def install_from_directory(source: Path, manifest: dict) -> None:
    """Copy from an existing model directory (offline / local mirror)."""
    for entry in manifest["files"]:
        src = source / entry["path"]
        if not src.is_file():
            raise SystemExit(f"Source directory is missing {entry['path']}: {source}")
        dst = MODELS_DIR / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_extractall(bundle: zipfile.ZipFile, destination: Path) -> None:
    """Extract a zip into destination, rejecting any entry that would land outside
    it (zip-slip).

    [Why not extractall(filter="data")] The filter parameter exists only since
    Python 3.12, and this script's interpreter criterion accepts major==3 only
    (see Resolve-Python in build.ps1); on 3.8-3.11 passing filter raises TypeError
    immediately. Per-entry validation behaves identically on every version.

    [What it defends against] Published zips are third-party content (GitHub
    releases, local mirrors): when an entry name contains `..` segments, absolute
    paths, drive letters, or backslash separators, extractall writes files outside
    destination -- and this runs on a developer machine with user privileges. An
    invalid name terminates immediately; there is no "fallback hack".
    """
    for info in bundle.infolist():
        normalized = info.filename.replace("\\", "/")
        if (
            normalized.startswith("/")
            or (len(normalized) > 1 and normalized[1] == ":")
            or ".." in PurePosixPath(normalized).parts
        ):
            raise SystemExit(f"Entry in {bundle.filename} would escape the destination, refusing to extract: {info.filename}")
    bundle.extractall(destination)


def install_from_zip(archive: Path, manifest: dict) -> None:
    # [Why extract into a temp directory first] Extracting straight into the target
    # directory lets "failure mid-extraction" leave a model directory that looks
    # present but is missing files -- exactly the state that is hardest to diagnose
    # at runtime.
    with tempfile.TemporaryDirectory(prefix="hfs-models-") as tmp:
        with zipfile.ZipFile(archive) as bundle:
            safe_extractall(bundle, Path(tmp))
        root = Path(tmp)
        # The zip may carry a top-level directory (models-<version>/...); adapt to it.
        candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
        first = manifest["files"][0]["path"]
        for candidate in candidates:
            if (candidate / first).is_file():
                install_from_directory(candidate, manifest)
                return
        raise SystemExit(f"{archive} does not contain the files listed in models.json")


def install_game_from_official_zip(archive: Path) -> None:
    """Install OpenVPI's GAME release zip into models/GAME/."""
    target = MODELS_DIR / "GAME"
    with tempfile.TemporaryDirectory(prefix="hfs-game-install-") as tmp:
        with zipfile.ZipFile(archive) as bundle:
            safe_extractall(bundle, Path(tmp))
        root = Path(tmp) / GAME_RELEASE_DIR
        if not root.is_dir():
            root = Path(tmp)
        target.mkdir(parents=True, exist_ok=True)
        for name in GAME_FILES:
            source = root / name
            if not source.is_file():
                raise SystemExit(f"GAME release package is missing {name}: {archive}")
            shutil.copy2(source, target / name)


def fetch_official_game() -> None:
    # [Why "file exists" alone is not enough] If the previous download was truncated,
    # or the user hand-swapped a same-named ONNX, all six files are present but the
    # inference output is wrong / all zeros. `--fetch-game` used to simply return
    # here, and hash verification only ran at the end of the **main download path**,
    # so this entry point was never verified. Verify against the manifest first and
    # re-download when corrupted; when GAME is not registered in the manifest, fall
    # back to an existence check.
    manifest = load_manifest()
    game_entries = [
        entry for entry in manifest["files"] if entry["path"].startswith(OPTIONAL_MODEL_PREFIXES)
    ]
    if game_entries and all((MODELS_DIR / "GAME" / name).is_file() for name in GAME_FILES):
        if not verify_optional_game_hashes(manifest):
            print("GAME model already exists and passed verification; skipping download")
            return
        print("GAME model failed verification; re-downloading", file=sys.stderr)
    elif all((MODELS_DIR / "GAME" / name).is_file() for name in GAME_FILES):
        print("GAME model already exists; skipping download")
        return
    with tempfile.TemporaryDirectory(prefix="hfs-game-dl-") as tmp:
        archive = Path(tmp) / "GAME-1.0.3-small-onnx.zip"
        download(GAME_RELEASE_URL, archive)
        install_game_from_official_zip(archive)


def download(url: str, destination: Path) -> None:
    """Download to `destination`: write a .part first, then replace atomically.

    [Why timeout is mandatory] On a hung connection a model download would
    block CI forever (urlopen without timeout keeps waiting on TCP keepalive).
    [Why a .part file] A mid-download failure / Ctrl-C would leave a truncated zip
    at the destination; the next run sees "file already exists" and skips it, and
    the symptom is "install succeeded but inference is all zeros".
    """
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "hifishifter-fetch-models"})
    part = destination.with_name(destination.name + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle)
            handle.flush()
        part.replace(destination)
    finally:
        if part.exists():
            part.unlink()


def format_mib(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def build_asset(destination: Path, manifest: dict) -> None:
    """Write the files listed in the manifest into a reproducible zip (release asset).

    [Why not right-click "compress to zip"] An asset's identity is its sha256:
    after upload, anyone can repackage it and compare hashes to learn whether they
    hold the same weights. Explorer, however, embeds the timestamp of the moment of
    packaging and an unspecified entry order -- the same weights yield different
    bytes every time, and that hash means nothing.

    [Why store instead of compress] The deflate byte stream varies with the zlib
    version and stops being reproducible across machines; ONNX weights are float
    tensors, and compression only saves single-digit percentages. Trading a few
    dozen MB of download size for "reproducible everywhere" is worth it: this
    package is the only thing that can match "the copy you installed" with "the
    copy I published" while troubleshooting.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    # [Why write .tmp first, then rename] A failure halfway through writing the archive
    # would leave an asset that "looks present" -- and it is a release artifact;
    # the next person uploading it would be a disaster.
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as bundle:
            for entry in sorted(manifest["files"], key=lambda item: item["path"].replace("/", "\\")):
                info = zipfile.ZipInfo(entry["path"], date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                # Without these two lines, packages built on Windows and on Unix differ in
                # their headers, hence in sha256.
                info.create_system = 3  # 0 = MS-DOS, 3 = Unix
                info.external_attr = 0o644 << 16
                with (MODELS_DIR / entry["path"]).open("rb") as reader, bundle.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_digest(archive: Path) -> str:
    """Write a sha256 sidecar (same format as sha256sum -c) so downloaders can verify without installing Python."""
    digest = sha256_of(archive)
    sidecar = archive.with_name(archive.name + ".sha256")
    with sidecar.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return digest


def verify_asset(archive: Path, manifest: dict) -> list[str]:
    """Check a zip as a release artifact: its entries must be **exactly** the ones the
    manifest lists.

    [Why even one extra file is an error] Extra content in an asset has only two
    origins, both accidents: the previous weights got packaged in (the installer
    installs from the manifest, so the extras never reach the shared library and
    just bloat the user's download), or the CoreML variant was mistaken for a model
    file. A missing file is more direct: after installation, "the model is unusable".
    """
    problems: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = {name for name in bundle.namelist() if not name.endswith("/")}
        expected = {entry["path"] for entry in manifest["files"]}
        problems += [f"zip is missing {path}" for path in sorted(expected - names)]
        problems += [f"zip has extra {path}" for path in sorted(names - expected)]
        if problems:
            return problems
        with tempfile.TemporaryDirectory(prefix="hfs-models-verify-") as tmp:
            safe_extractall(bundle, Path(tmp))
            problems += verify_hashes(Path(tmp), manifest)
    sidecar = archive.with_name(archive.name + ".sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="utf-8").split()[0].lower()
        actual = sha256_of(archive)
        if recorded != actual:
            problems.append(f"{sidecar.name} does not match the zip: {actual} != {recorded}")
    return problems


def package_asset(destination: Path, manifest: dict) -> int:
    version = manifest["version"]
    problems = missing_files(MODELS_DIR, manifest)
    if not problems:
        problems = verify_hashes(MODELS_DIR, manifest)
    if not problems:
        derived = derive_version(manifest["files"])
        if derived != version:
            problems.append(f"manifest version {version} does not match the value recomputed from content {derived} (the manifest was hand-edited)")
    if problems:
        print(f"Refusing to package (manifest version {version}):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "  Weights changed: regenerate the manifest with python scripts/setup/fetch-models.py --write-manifest\n"
            "  Weights unchanged: fetch them first with python scripts/setup/fetch-models.py",
            file=sys.stderr,
        )
        return 1
    for stray in stray_files(MODELS_DIR, manifest):
        print(f"Warning: {stray} is not in the manifest and will not go into the asset (if you added new model files, regenerate the manifest first)", file=sys.stderr)

    build_asset(destination, manifest)
    digest = write_digest(destination)
    sidecar = destination.with_name(destination.name + ".sha256")
    print(f"Asset {destination} ({format_mib(destination.stat().st_size)})")
    print(f"  sha256  {digest}")
    print(f"  sidecar {sidecar}")
    print("Next steps:")
    print(f"  1) python scripts/setup/fetch-models.py --verify-archive {destination}")
    print(f"  2) Upload to the tag models-{version}: commands and order in docs/models.md")
    print("  3) Merge the manifest once the published asset is confirmed downloadable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch / package HiFiShifter model weights")
    parser.add_argument("--check", action="store_true", help="check completeness only (exit code 1 if anything is missing)")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="rewrite models.json from the real files on disk (version = content hash)",
    )
    parser.add_argument(
        "--check-manifest",
        action="store_true",
        help="only check that models.json is current (skips when weights are missing, exit code 0)",
    )
    parser.add_argument("--source", type=Path, help="copy from a local model directory")
    parser.add_argument("--archive", type=Path, help="extract from a local zip")
    parser.add_argument(
        "--fetch-game",
        action="store_true",
        help="fetch the GAME small ONNX model from OpenVPI's official release into models/GAME/",
    )
    parser.add_argument(
        "--package",
        nargs="?",
        const="",
        metavar="OUT",
        help="package into a release asset (default dist/models/models-<version>.zip, no network)",
    )
    parser.add_argument(
        "--verify-archive",
        type=Path,
        metavar="ZIP",
        help="verify that a zip asset matches this manifest exactly",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HIFISHIFTER_MODELS_BASE_URL", DEFAULT_BASE_URL),
        help="base URL of the release assets (for mirrors; HIFISHIFTER_MODELS_BASE_URL also works)",
    )
    args = parser.parse_args()

    if args.fetch_game:
        fetch_official_game()
        print("GAME model installed to backend/src-tauri/resources/models/GAME")
        return 0

    # These two modes read / write the manifest themselves and do not need it loaded
    # first (a clean clone has no weights yet).
    if args.write_manifest:
        return write_manifest()
    if args.check_manifest:
        return check_manifest()

    manifest = load_manifest()
    version = manifest["version"]

    if args.verify_archive:
        archive = args.verify_archive.expanduser().resolve()
        if not archive.is_file():
            raise SystemExit(f"Not found: {archive}")
        problems = verify_asset(archive, manifest)
        if problems:
            print(f"{archive.name} is not the version {version} asset:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{archive.name} matches the manifest (version {version}, {len(manifest['files'])} files)")
        return 0

    if args.package is not None:
        destination = (
            Path(args.package).expanduser().resolve()
            if args.package
            else ASSET_DIR / f"models-{version}.zip"
        )
        return package_asset(destination, manifest)

    problems = missing_files(MODELS_DIR, manifest)
    if not problems:
        if args.check or args.source or args.archive:
            print(f"Model weights complete (version {version})")
        return 0
    if args.check:
        print(
            f"Model weights incomplete (version {version}): {'; '.join(problems[:5])}",
            file=sys.stderr,
        )
        return 1

    print(f"Model weights incomplete (version {version}); fetching now...")
    if args.source:
        install_from_directory(args.source.expanduser().resolve(), manifest)
    elif args.archive:
        install_from_zip(args.archive.expanduser().resolve(), manifest)
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hfs-models-dl-") as tmp:
            archive = Path(tmp) / f"models-{version}.zip"
            url = f"{args.base_url.rstrip('/')}/models-{version}/models-{version}.zip"
            try:
                download(url, archive)
            except Exception as error:  # noqa: BLE001 - network errors take too many shapes; give one human-readable message for all
                raise SystemExit(
                    f"Download failed: {error}\n"
                    f"  Asset URL: {url}\n"
                    "  Offline: use --source <existing model directory> or --archive <local zip>."
                ) from error
            install_from_zip(archive, manifest)

    # [Why hash-verify after the download] "Byte count matches but content does not"
    # (a same-named file swapped in, or a truncated download padded back) has really
    # happened: the symptom is inference quality quietly degrading, not an error.
    hash_problems = verify_hashes(MODELS_DIR, manifest)
    hash_problems += verify_optional_game_hashes(manifest)
    if hash_problems:
        print("Model weight verification failed:", file=sys.stderr)
        for problem in hash_problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"Model weights ready (version {version}, {len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
