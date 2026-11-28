#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Linux native packages: deb + rpm for the app, the model library and the plug-in formats.
#
# [Why native packages] Linux users install software from their distribution: double-click a
# .deb in the software centre, or `apt install ./x.deb` / `dnf install ./x.rpm`. The previous
# download was a self-extracting wizard that reimplemented dependency resolution, desktop
# integration and uninstallation inside a bash script, and had to paint its own dialogs on four
# different backends. The package manager already does all of that, and does it per
# distribution. So the wizard is gone and these four packages are the primary path:
#
#   hifishifter         the app, its desktop entry and the model manifest
#   hifishifter-models  the weights, in a shared store both the app and the plug-ins read
#   hifishifter-vst3    the VST3 bundle
#   hifishifter-clap    the CLAP module and its sibling HiFiShifter-resources
#
# [Why nfpm] One declarative config per package produces both formats -- including the aarch64
# rpm, which would otherwise need a target-architecture rpmbuild on an x86_64 runner. Nothing is
# compiled here: this script only maps a prepared tree onto the FHS layout and hands the result
# to nfpm, so the whole Linux packaging step is reproducible on any host.
#
# Usage:
#   scripts/release/linux-packages/build.sh --stage=DIR --out=DIR [options]
#
#   --stage=DIR             prepared payload (see "Stage layout" below)
#   --out=DIR               where the .deb / .rpm files are written
#   --arch=amd64|arm64      package architecture (default: this machine's)
#   --version=X.Y.Z[-pre]   product version (default: backend/src-tauri/tauri.conf.json)
#   --only=app,models,vst3,clap   packages to build (default: all four)
#   --nfpm=PATH             use this nfpm binary instead of the pinned download
#   -h, --help
#
# Stage layout (assembled by scripts/build/unix/package.sh):
#
#   app/hifishifter               the executable        -> /usr/bin/hifishifter
#   app/lib/*.so                  its private libraries-> /usr/lib/HiFiShifter/
#   models/<manifest version>/    the whole model store-> /usr/share/HiFiShifter/models/<version>/
#   vst3/HiFiShifter.vst3/        the VST3 bundle      -> /usr/lib/vst3/HiFiShifter.vst3/
#   clap/HiFiShifter.clap         the CLAP module      -> /usr/lib/clap/HiFiShifter.clap
#   clap/HiFiShifter-resources/   its sibling resources-> /usr/lib/clap/HiFiShifter-resources/
#   icons/{32x32,128x128,256x256}.png                    -> /usr/share/icons/hicolor/...
#   LICENSE, licenses/                                   -> /usr/share/licenses/<package>/
#
# Version mapping. Two ecosystems, one function each; nothing else may re-derive these, and the
# file names below are the only place the mapping is turned into text:
#
#   product 0.2.0-beta  ->  deb 0.2.0~beta   (the ~ sorts the prerelease before 0.2.0)
#                       ->  rpm 0.2.0-0.1.beta
#   model   <manifest>  ->  deb <manifest>   (the manifest version IS the content hash)
#                       ->  rpm <manifest>-1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly CONFIG_DIR="$SCRIPT_DIR/nfpm"
readonly ASSETS_DIR="$SCRIPT_DIR/assets"
readonly SCRIPTLETS_DIR="$SCRIPT_DIR/scripts"
readonly MAINTAINER="ARounder <ARounder@users.noreply.github.com>"
readonly HOMEPAGE="https://github.com/ARounder-183/HiFiShifter"

# [Why the version is pinned] The package metadata is an artifact: a runner that silently picks
# up a newer nfpm produces a different .deb from the same sources. The download below is the
# only place this number appears; `nfpm --version` is checked against it (a 2.x on PATH is
# accepted with a warning, so a local developer is not blocked by a patch release).
readonly NFPM_VERSION="v2.49.0"
readonly TOOL_CACHE="${HIFISHIFTER_TOOL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/hifishifter-tools}"

STAGE=""
OUT=""
ARCH=""
VERSION=""
ONLY="app,models,vst3,clap"
NFPM_BIN=""

log()  { printf '[packages] %s\n' "$*"; }
step() { printf '\n[packages] ── %s\n' "$*"; }
die()  { printf '[packages] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: scripts/release/linux-packages/build.sh --stage=DIR --out=DIR [options]

  --stage=DIR             prepared payload (app/, models/, vst3/, clap/, icons/, licenses/)
  --out=DIR               where the .deb / .rpm files are written
  --arch=amd64|arm64      package architecture (default: this machine's)
  --version=X.Y.Z[-pre]   product version (default: backend/src-tauri/tauri.conf.json)
  --only=app,models,vst3,clap   packages to build (default: all four)
  --nfpm=PATH             use this nfpm binary instead of the pinned download
  -h, --help
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)   STAGE="${1#*=}"; shift ;;
        --stage)     [ $# -ge 2 ] || die "--stage requires a value"; STAGE="$2"; shift 2 ;;
        --out=*)     OUT="${1#*=}"; shift ;;
        --out)       [ $# -ge 2 ] || die "--out requires a value"; OUT="$2"; shift 2 ;;
        --arch=*)    ARCH="${1#*=}"; shift ;;
        --arch)      [ $# -ge 2 ] || die "--arch requires a value"; ARCH="$2"; shift 2 ;;
        --version=*) VERSION="${1#*=}"; shift ;;
        --version)   [ $# -ge 2 ] || die "--version requires a value"; VERSION="$2"; shift 2 ;;
        --only=*)    ONLY="${1#*=}"; shift ;;
        --only)      [ $# -ge 2 ] || die "--only requires a value"; ONLY="$2"; shift 2 ;;
        --nfpm=*)    NFPM_BIN="${1#*=}"; shift ;;
        --nfpm)      [ $# -ge 2 ] || die "--nfpm requires a value"; NFPM_BIN="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           die "unknown option: $1 (see --help)" ;;
    esac
done

# ── version mapping ─────────────────────────────────────────────────────────

# The product version exactly as the artifact names show it: backend/src-tauri/tauri.conf.json.
product_version() {
    if [ -n "$VERSION" ]; then
        printf '%s\n' "${VERSION#v}"
        return 0
    fi
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' \
        "$REPO_ROOT/backend/src-tauri/tauri.conf.json"
}

# 0.2.0-beta -> 0.2.0~beta. Every hyphen becomes a tilde: Debian sorts "~" before the empty
# string, so a prerelease lands after 0.1.9 and before 0.2.0 -- which is what a beta is.
deb_version() {
    printf '%s\n' "$(printf '%s' "$1" | sed 's/-/~/g')"
}

# models.json's version is a hex content hash: it may start with a letter, and Debian requires a
# version to start with a digit. Prefixing "0." keeps the value readable and ordered, and the same
# string is used in the dependency, so the app and the model package can never disagree.
model_pkg_version() {
    case "$1" in
        [0-9]*) printf '%s\n' "$1" ;;
        *)      printf '0.%s\n' "$1" ;;
    esac
}

# 0.2.0-beta -> 0.2.0. The part after the hyphen has to be represented somewhere or two
# prereleases collide, so it moves into Release as 0.1.<part> (0.1.x sorts before 1).
rpm_version() {
    printf '%s\n' "$(printf '%s' "$1" | sed 's/-.*$//')"
}

rpm_release() {
    case "$1" in
        *-*) printf '0.1.%s\n' "$(printf '%s' "${1#*-}" | sed 's/-/./g')" ;;
        *)   printf '1\n' ;;
    esac
}

# ── architecture ───────────────────────────────────────────────────────────

# --arch accepts both spellings: a caller may pass the Rust tag (x86_64/arm64) or the Debian one
# (amd64/arm64). nfpm maps amd64 -> x86_64 for rpm on its own.
resolve_arch() {
    case "${ARCH:-$(uname -m)}" in
        amd64|x86_64)  HFS_DEB_ARCH="amd64" ;;
        arm64|aarch64) HFS_DEB_ARCH="arm64" ;;
        *)             die "unsupported architecture: ${ARCH:-$(uname -m)} (expected amd64 or arm64)" ;;
    esac
}

# <package arch>: the architecture spelling a file name uses. "all" stays "all" for Debian and
# becomes "noarch" for RPM -- writing the machine's architecture into a models package's file name
# is how an arch-independent package ends up looking architecture-specific to a user.
deb_file_arch() {
    case "$1" in
        all) printf 'all\n' ;;
        *)   printf '%s\n' "$HFS_DEB_ARCH" ;;
    esac
}

rpm_file_arch() {
    case "$1" in
        all) printf 'noarch\n' ;;
        amd64) printf 'x86_64\n' ;;
        arm64) printf 'aarch64\n' ;;
        *)     printf '%s\n' "$1" ;;
    esac
}

# ── nfpm ───────────────────────────────────────────────────────────────────

nfpm_goarch() {
    case "$(uname -m)" in
        x86_64|amd64)  printf 'x86_64\n' ;;
        aarch64|arm64) printf 'arm64\n' ;;
        *)             die "no nfpm build for $(uname -m)" ;;
    esac
}

nfpm_reported_version() {
    "$1" --version 2>/dev/null | sed -n 's/^GitVersion: *//p' | head -1
}

# Print a usable nfpm: an explicit --nfpm, an nfpm already on PATH, or the pinned download.
ensure_nfpm() {
    local candidate="" reported tarball
    if [ -n "$NFPM_BIN" ]; then
        [ -x "$NFPM_BIN" ] || die "--nfpm is not executable: $NFPM_BIN"
        candidate="$NFPM_BIN"
    elif command -v nfpm >/dev/null 2>&1; then
        candidate="$(command -v nfpm)"
    else
        candidate="$TOOL_CACHE/nfpm-$NFPM_VERSION/nfpm"
        if [ ! -x "$candidate" ]; then
            # [Why stderr] ensure_nfpm stdout is the binary path ("NFPM=$(ensure_nfpm)"); a
            # progress line on stdout would become part of the command build_package runs.
            step "downloading nfpm $NFPM_VERSION" >&2
            mkdir -p "$(dirname "$candidate")"
            tarball="$TOOL_CACHE/nfpm-$NFPM_VERSION.tar.gz"
            curl -fsSL -o "$tarball" \
                "https://github.com/goreleaser/nfpm/releases/download/$NFPM_VERSION/nfpm_${NFPM_VERSION#v}_Linux_$(nfpm_goarch).tar.gz" \
                || die "failed to download nfpm $NFPM_VERSION (put an nfpm on PATH or pass --nfpm)"
            tar -xzf "$tarball" -C "$(dirname "$candidate")" nfpm || die "failed to unpack nfpm"
            rm -f "$tarball"
            chmod +x "$candidate"
        fi
    fi
    reported="$(nfpm_reported_version "$candidate")"
    [ -n "$reported" ] || die "not a usable nfpm (no GitVersion in --version): $candidate"
    case "$reported" in
        2.*) ;;
        *) die "nfpm 2.x is required, found $reported ($candidate)" ;;
    esac
    if [ "$reported" != "${NFPM_VERSION#v}" ]; then
        log "WARNING: nfpm $reported differs from the pinned ${NFPM_VERSION#v}; package metadata may drift" >&2
    fi
    printf '%s\n' "$candidate"
}

# ── stage checks ───────────────────────────────────────────────────────────

wants() {
    case ",$ONLY," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

require_stage() {
    [ -n "$STAGE" ] || die "--stage is required"
    [ -d "$STAGE" ] || die "--stage is not a directory: $STAGE"
    [ -n "$OUT" ] || die "--out is required"
    mkdir -p "$OUT"
    HFS_PRODUCT_VERSION="$(product_version)"
    [ -n "$HFS_PRODUCT_VERSION" ] || die "could not determine the product version"
    HFS_DEB_VERSION="$(deb_version "$HFS_PRODUCT_VERSION")"
    HFS_RPM_VERSION="$(rpm_version "$HFS_PRODUCT_VERSION")"
    HFS_RPM_RELEASE="$(rpm_release "$HFS_PRODUCT_VERSION")"
    HFS_MODEL_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' \
        "$STAGE/models/models.json")" || die "stage/models/models.json is missing: $STAGE"
    HFS_MODEL_PKG_VERSION="$(model_pkg_version "$HFS_MODEL_VERSION")"
    # [Why only for the models package] A plug-ins-only build stages models.json alone, just to
    # learn the version for the dependency; the weights are the models package's payload.
    if wants models; then verify_models_tree; fi
    log "version $HFS_PRODUCT_VERSION (deb $HFS_DEB_VERSION / rpm $HFS_RPM_VERSION-$HFS_RPM_RELEASE) | models $HFS_MODEL_VERSION | arch $HFS_DEB_ARCH"
}

# ── model store validation ─────────────────────────────────────────────────

# [Why this check exists twice] package.sh already strips the weights out of the plug-in bundles;
# this one proves the opposite thing: that the store which becomes a *system* package is complete
# and matches its manifest byte for byte. It is also what makes "Architecture: all" honest -- a
# tree that validates against a content-derived manifest is byte-identical to any other tree that
# validates against it, so the weights cannot differ between amd64 and arm64.
#
# GAME/ is the one optional sub-tree (see model_store::OPTIONAL_MODEL_DIRS): a machine that only
# fetched the GAME weights has no other weights yet, and a missing GAME is not a broken store.
verify_models_tree() {
    local models="$STAGE/models"
    [ -f "$models/models.json" ] || die "missing model manifest: $models/models.json"
    python3 - "$models" <<'PY' || die "the staged model store does not match models.json"
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "models.json").read_text(encoding="utf-8"))
problems = []
for entry in manifest.get("files", []):
    path = root / entry["path"]
    if entry["path"].startswith("GAME/") and not path.is_file():
        continue
    if not path.is_file():
        problems.append("missing: " + entry["path"])
        continue
    data = path.read_bytes()
    if len(data) != entry["bytes"]:
        problems.append("size mismatch: %s (%d != %d)" % (entry["path"], len(data), entry["bytes"]))
    # models.json stores the digests upper-cased (scripts/setup/fetch-models.py); compare in one
    # case so the check cannot disagree with the manifest about spelling.
    elif hashlib.sha256(data).hexdigest().upper() != entry["sha256"].upper():
        problems.append("sha256 mismatch: " + entry["path"])
if not manifest.get("files"):
    problems.append("models.json lists no files")
if problems:
    for problem in problems:
        print("  " + problem, file=sys.stderr)
    sys.exit(1)
print("model store validated against models.json (%d files)" % len(manifest["files"]))
PY
}

# ── contents ───────────────────────────────────────────────────────────────
#
# [Why the file list is generated and not written in the YAML] nfpm's YAML is the *metadata* of a
# package; the file lists differ by architecture (arm64 has no libwebgpu_dawn.so) and by manifest
# version. Generating them means the tree that is actually staged is what ships -- a hand-written
# list is a second source of truth that goes stale the first time a bundle gains a file.
#
# <src> <dst> <mode>: one YAML entry. Paths are double-quoted so a space in a bundle path does
# not break the document.
emit_file() {
    printf '  - src: "%s"\n    dst: "%s"\n    file_info:\n      mode: %s\n' \
        "$(yaml_escape "$1")" "$(yaml_escape "$2")" "$3" >>"$CONTENTS"
}

yaml_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

# <src-dir> <dst-dir> <mode>: every regular file under src-dir, mirrored below dst-dir.
emit_tree() {
    local src="$1" dst="$2" mode="$3" file rel
    [ -d "$src" ] || die "missing $src"
    while IFS= read -r file; do
        [ -n "$file" ] || continue
        rel="${file#"$src"/}"
        emit_file "$file" "$dst/$rel" "$mode"
    done <<< "$(find "$src" -type f -print | LC_ALL=C sort)"
}

contents_app() {
    local lib icon
    emit_file "$STAGE/app/hifishifter" /usr/bin/hifishifter 0755
    for lib in "$STAGE"/app/lib/*.so; do
        [ -f "$lib" ] || continue
        emit_file "$lib" "/usr/lib/HiFiShifter/$(basename "$lib")" 0755
    done
    emit_file "$STAGE/models/models.json" /usr/lib/HiFiShifter/models/models.json 0644
    emit_file "$ASSETS_DIR/hifishifter.desktop" /usr/share/applications/hifishifter.desktop 0644
    emit_file "$ASSETS_DIR/hifishifter.xml" /usr/share/mime/packages/hifishifter.xml 0644
    for icon in 32x32 128x128 256x256; do
        [ -f "$STAGE/icons/$icon.png" ] || continue
        emit_file "$STAGE/icons/$icon.png" "/usr/share/icons/hicolor/$icon/apps/hifishifter.png" 0644
    done
    emit_file "$STAGE/LICENSE" /usr/share/licenses/hifishifter/LICENSE 0644
    emit_tree "$STAGE/licenses" /usr/share/licenses/hifishifter/licenses 0644
}

contents_models() {
    emit_tree "$STAGE/models" "/usr/share/HiFiShifter/models/$HFS_MODEL_VERSION" 0644
    emit_file "$STAGE/LICENSE" /usr/share/licenses/hifishifter-models/LICENSE 0644
    emit_tree "$STAGE/licenses" /usr/share/licenses/hifishifter-models/licenses 0644
}

contents_vst3() {
    emit_tree "$STAGE/vst3/HiFiShifter.vst3" /usr/lib/vst3/HiFiShifter.vst3 0644
    emit_file "$STAGE/LICENSE" /usr/share/licenses/hifishifter-vst3/LICENSE 0644
}

contents_clap() {
    emit_file "$STAGE/clap/HiFiShifter.clap" /usr/lib/clap/HiFiShifter.clap 0755
    emit_tree "$STAGE/clap/HiFiShifter-resources" /usr/lib/clap/HiFiShifter-resources 0644
    emit_file "$STAGE/LICENSE" /usr/share/licenses/hifishifter-clap/LICENSE 0644
}

# ── package assembly ───────────────────────────────────────────────────────

# [Why each package gets its own generated copy of its YAML] One file produces the deb and the
# rpm with two different versions, so the placeholders are substituted per packager. [Why not
# nfpm's own environment expansion] It applies to some fields only -- a ${HFS_SCRIPTS_DIR} inside
# "scripts:" reaches nfpm verbatim and the build dies on a literal path. Substituting here is one
# rule for every field, and it is testable without nfpm.
#
# <name>: the config with every placeholder replaced by its value.
render_config() {
    sed \
        -e "s|\${HFS_PKG_ARCH}|$pkg_arch|g" \
        -e "s|\${HFS_PKG_VERSION}|$pkg_version|g" \
        -e "s|\${HFS_PKG_RELEASE}|$pkg_release|g" \
        -e "s|\${HFS_PKG_MAINTAINER}|$MAINTAINER|g" \
        -e "s|\${HFS_PKG_HOMEPAGE}|$HOMEPAGE|g" \
        -e "s|\${HFS_MODEL_VERSION}|$HFS_MODEL_PKG_VERSION|g" \
        -e "s|\${HFS_SCRIPTS_DIR}|$SCRIPTLETS_DIR|g" \
        "$CONFIG_DIR/$1.yaml"
}

# <name> <contents-function> <packager> <target-file>
build_package() {
    local name="$1" contents="$2" packager="$3" target="$4" yaml
    yaml="$WORK/$name.$packager.yaml"
    CONTENTS="$WORK/$name.$packager.contents"
    : >"$CONTENTS"
    "$contents"
    [ -s "$CONTENTS" ] || die "$name: no files staged"

    # The deb and the rpm differ only here: Debian puts the prerelease in the version
    # (0.2.0~beta), RPM moves it into the release (0.2.0-0.1.beta).
    case "$packager" in
        deb) pkg_version="$pkg_version_deb"; pkg_release="" ;;
        rpm) pkg_version="$pkg_version_rpm"; pkg_release="$pkg_release_rpm" ;;
        *) die "unknown packager: $packager" ;;
    esac
    {
        render_config "$name"
        printf '\n# Generated by build.sh from the staged tree -- do not edit.\ncontents:\n'
        cat "$CONTENTS"
    } >"$yaml"

    "$NFPM" package --packager "$packager" --config "$yaml" --target "$target"
    log "built $(basename "$target")"
}


# The models package is versioned by its content; every other package by the product version.
set_package_version() {
    case "$1" in
        hifishifter-models)
            pkg_version_deb="$HFS_MODEL_PKG_VERSION"
            pkg_version_rpm="$HFS_MODEL_PKG_VERSION"
            pkg_release_rpm="1"
            # Architecture-independent: see the note in nfpm/hifishifter-models.yaml. nfpm turns
            # "all" into "noarch" for the rpm on its own.
            pkg_arch="all"
            ;;
        *)
            pkg_version_deb="$HFS_DEB_VERSION"
            pkg_version_rpm="$HFS_RPM_VERSION"
            pkg_release_rpm="$HFS_RPM_RELEASE"
            pkg_arch="$HFS_DEB_ARCH"
            ;;
    esac
}

# <name> <contents-function>: both formats, with the file names the artifact listing promises.
build_both() {
    local name="$1" contents="$2"
    set_package_version "$name"
    step "$name $pkg_version_deb (deb) / $pkg_version_rpm-$pkg_release_rpm (rpm)"
    build_package "$name" "$contents" deb \
        "$OUT/${name}_${pkg_version_deb}_$(deb_file_arch "$pkg_arch").deb"
    build_package "$name" "$contents" rpm \
        "$OUT/${name}-${pkg_version_rpm}-${pkg_release_rpm}.$(rpm_file_arch "$pkg_arch").rpm"
}

# ── entry point ────────────────────────────────────────────────────────────

resolve_arch
require_stage
NFPM="$(ensure_nfpm)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if wants app;    then build_both hifishifter        contents_app;    fi
if wants models; then build_both hifishifter-models contents_models; fi
if wants vst3;   then build_both hifishifter-vst3   contents_vst3;   fi
if wants clap;   then build_both hifishifter-clap   contents_clap;   fi

step "packages written to $OUT"
find "$OUT" -maxdepth 1 -type f \( -name '*.deb' -o -name '*.rpm' \) -printf '%f\n' | LC_ALL=C sort
