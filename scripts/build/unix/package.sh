#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Unix distribution entry point: one compile → one user-installable artifact per OS.
#
#     scripts/build/unix/package.sh                    # plugin packages only (VST3 + CLAP)
#     scripts/build/unix/package.sh --formats=vst3     # VST3 plugin package only
#     scripts/build/unix/package.sh --reuse-frontend   # reuse frontend/dist
#     scripts/build/unix/package.sh --appimage=...     # Linux unified (AppImage + plugins + model store)
#     scripts/build/unix/package.sh --app-bundle=...   # macOS unified (App + plugins + model store)
#     HFS_PLATFORM_ARCH=universal scripts/build/unix/package.sh --app-bundle=... \
#         --merge-arch=DIR --merge-app-arch=DIR        # ...and both architectures in one package
#
# [Why builds are separate but distribution is unified] The two formats' **builds** are two
# different jobs: different crates, different bundle layouts, different entry symbols, different
# black-box probes. Jamming them into one script yields an if-branch tree like build.ps1's — and
# the assets-nesting bug lived in exactly one of those branches. But **distribution** must be
# unified: users do not care about formats, and the plugins and the App share the documentation
# format and the model manifest version; separate downloads sooner or later produce
# "new plugin + old App".
#
# So: each format's script builds its own bundle (scripts/<fmt>/unix/build.sh); this script does
# three things — compile the frontend once, invoke per format, and pack the results into one artifact.
#
# Unified mode (--appimage / --app-bundle) packs the App, VST3, CLAP and a shared model store
# into one installer; bundles keep only models.json, and the weights exist once at the top level
# under models/<version>. Without these flags, the original plugins-only behavior is kept.
#
# Artifacts:
#   Linux   dist/HiFiShifter_v<version>_<os>-<arch>-unified.tar.gz (includes install.sh)
#   macOS   dist/HiFiShifter_v<version>_<os>-<arch>-unified.pkg (installs to /Applications and /Library)
#   Plugins dist/HiFiShifter_v<version>_<os>-<arch>-plugins.tar.gz / .pkg
#
# Signing and notarization: once HIFISHIFTER_CODESIGN_IDENTITY is set, macOS signs with Developer
# ID, and HIFISHIFTER_NOTARY_PROFILE enables notarytool notarization + stapler stapling. The
# default is ad-hoc signing — enough for CI self-checks, not for distribution to strangers
# (Gatekeeper will block it).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
readonly MODELS_SRC="$REPO_ROOT/backend/src-tauri/resources/models"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

FORMATS="vst3,clap"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
REUSE_FRONTEND=0
SKIP_VERIFY=0
MERGE_ARCH=""
MERGE_APP_ARCH=""
APPIMAGE=""
APP_BUNDLE=""

log()  { printf '[package] %s\n' "$*"; }
step() { printf '\n[package] ── %s\n' "$*"; }
die()  { printf '[package] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: scripts/build/unix/package.sh [options]

  --formats=vst3,clap   formats to package (both by default)
  --profile=release|dist
  --out=DIR             output directory (default dist)
  --reuse-frontend      reuse an existing frontend/dist (it must already exist)
  --no-verify           skip format self-checks (only for machines where probes cannot run)
  --merge-arch=DIR      merge another architecture's dist into a universal (needs lipo, macOS only)
  --merge-app-arch=DIR  merge the other architecture's .app into a universal App (macOS only)
  --appimage=FILE       Linux unified: install an existing AppImage into the tar.gz too
  --app-bundle=DIR      macOS unified: install an existing .app into the pkg too
  -h, --help
USAGE
}

# [Why both --x=v and --x v are accepted] The workflow uses the space form (more readable), while
# `--profile=dist` is more common on the command line. Accepting only one form cost a CI run that
# said "unknown option: --profile" — exactly what happened the first time package.sh was wired
# into build.yml.
while [ $# -gt 0 ]; do
    case "$1" in
        --formats=*)      FORMATS="${1#*=}"; shift ;;
        --formats)        [ $# -ge 2 ] || die "--formats requires a value"; FORMATS="$2"; shift 2 ;;
        --profile=*)      PROFILE="${1#*=}"; shift ;;
        --profile)        [ $# -ge 2 ] || die "--profile requires a value"; PROFILE="$2"; shift 2 ;;
        --out=*)          OUT_DIR="${1#*=}"; shift ;;
        --out)            [ $# -ge 2 ] || die "--out requires a value"; OUT_DIR="$2"; shift 2 ;;
        --reuse-frontend) REUSE_FRONTEND=1; shift ;;
        --no-verify)      SKIP_VERIFY=1; shift ;;
        --merge-arch=*)   MERGE_ARCH="${1#*=}"; shift ;;
        --merge-arch)     [ $# -ge 2 ] || die "--merge-arch requires a value"; MERGE_ARCH="$2"; shift 2 ;;
        --merge-app-arch=*) MERGE_APP_ARCH="${1#*=}"; shift ;;
        --merge-app-arch) [ $# -ge 2 ] || die "--merge-app-arch requires a value"; MERGE_APP_ARCH="$2"; shift 2 ;;
        --appimage=*)     APPIMAGE="${1#*=}"; shift ;;
        --appimage)       [ $# -ge 2 ] || die "--appimage requires a value"; APPIMAGE="$2"; shift 2 ;;
        --app-bundle=*)   APP_BUNDLE="${1#*=}"; shift ;;
        --app-bundle)     [ $# -ge 2 ] || die "--app-bundle requires a value"; APP_BUNDLE="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "unknown option: $1 (see --help)" ;;
    esac
done

hfs_detect_platform
PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")"
MODEL_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$MODELS_SRC/models.json")"
readonly PRODUCT_VERSION MODEL_VERSION
ARTIFACT_BASE="$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-plugins"
UNIFIED_BASE="$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-unified"

log "Platform ${HFS_TAG} | Version ${PRODUCT_VERSION} | Models ${MODEL_VERSION} | Formats ${FORMATS} | profile $PROFILE"

wants() {
    case ",$FORMATS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# ── Stage 1: frontend (built once; both formats share the same output) ──────
stage_frontend() {
    if [ "$REUSE_FRONTEND" = "1" ]; then
        [ -s "$FRONTEND_DIST/plugin.html" ] || die "--reuse-frontend given but $FRONTEND_DIST/plugin.html does not exist"
        log "reusing existing frontend output: $FRONTEND_DIST"
        return
    fi
    step "building frontend (vite, including the plugin.html entry)"
    command -v node >/dev/null 2>&1 || die "node missing: requires ^20.19.0 or >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci failed (check the network and frontend/package-lock.json)"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "frontend build failed"
    [ -s "$FRONTEND_DIST/plugin.html" ] || die "frontend build finished but plugin.html does not exist"
}

# ── Stage 2: build per format (each format script owns its own layout) ──────
build_format() {
    local fmt="$1" flag
    case "$fmt" in
        vst3) flag="--skip-frontend" ;;
        clap) flag="--reuse-frontend" ;;
        *) die "unknown format: $fmt" ;;
    esac
    step "building $fmt"
    if [ -s "$FRONTEND_DIST/plugin.html" ]; then
        bash "$REPO_ROOT/scripts/$fmt/unix/build.sh" --stage=bundle \
            --profile="$PROFILE" --out="$OUT_DIR" "$flag"
    else
        bash "$REPO_ROOT/scripts/$fmt/unix/build.sh" --stage=bundle \
            --profile="$PROFILE" --out="$OUT_DIR"
    fi
}

verify_format() {
    local fmt="$1"
    [ "$SKIP_VERIFY" = "1" ] && { log "skipping $fmt self-check per --no-verify"; return 0; }
    step "self-check $fmt"
    case "$fmt" in
        vst3) bash "$REPO_ROOT/scripts/vst3/unix/verify.sh" "$OUT_DIR/HiFiShifter.vst3" ;;
        clap) bash "$REPO_ROOT/scripts/clap/verify.sh" "$OUT_DIR/clap/HiFiShifter.clap" ;;
    esac
}

# ── Stage 3: merge universal (optional, macOS only) ─────────────────────────
#
# [Why there are two switches] A universal package has two kinds of single-arch leftovers, and
# their other half lives in two different places: the plugin bundles come from another dist/
# directory, the App bundle comes from Tauri's other target. One switch cannot describe both.
#
# Every merge site has to ask "is it already universal?" first: lipo refuses a second slice of
# an architecture it already has, so merging a file Tauri already lipo'd turns a correct package
# into a red build (and the error names a temp path, not the cause).

# <dst> <src>: lipo the two single-arch slices into <dst>.
# Returns 0 when it merged, 1 when there was nothing to merge (not a Mach-O, already universal,
# or both sides carry the same slice -- that last case cannot be repaired here and is what the
# workflow's arch assertion is for).
merge_macho_pair() {
    local dst="$1" src="$2" dst_archs src_archs
    [ -f "$dst" ] || return 1
    [ -f "$src" ] || return 1
    file -b "$dst" | grep -q 'Mach-O' || return 1
    file -b "$src" | grep -q 'Mach-O' || return 1
    hfs_is_universal "$dst" && return 1
    hfs_is_universal "$src" && return 1
    dst_archs="$(hfs_macho_archs "$dst" | tr '\n' ' ' | sed 's/ $//')"
    src_archs="$(hfs_macho_archs "$src" | tr '\n' ' ' | sed 's/ $//')"
    if [ "$dst_archs" = "$src_archs" ]; then
        log "WARNING: $dst carries only ${dst_archs} and so does its counterpart; nothing to merge"
        return 1
    fi
    lipo -create "$dst" "$src" -output "$dst.universal" || die "lipo merge failed: $dst"
    mv -f "$dst.universal" "$dst"
    return 0
}

merge_arch() {
    local other="$1" rel merged=0
    [ "$HFS_OS" = "macos" ] || die "--merge-arch only makes sense on macOS"
    command -v lipo >/dev/null 2>&1 || die "lipo missing (Xcode command line tools required)"
    [ -d "$other" ] || die "--merge-arch directory does not exist: $other"
    # [Why the tag is checked] The merged artifact is universal whether or not the name says so;
    # letting it keep the host's tag publishes "macos-arm64" for a package that runs on Intel too
    # -- the one mistake a user cannot see until they install it.
    [ "$HFS_PLATFORM_ARCH" = "universal" ] \
        || die "--merge-arch produces a universal package: set HFS_PLATFORM_ARCH=universal so the artifact name says so (current tag: ${HFS_TAG})"
    step "merging the universal plugin bundles (${other})"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        if merge_macho_pair "$OUT_DIR/$rel" "$other/$rel"; then
            log "universal: $rel"
            merged=$((merged + 1))
        fi
    done <<< "$( cd "$OUT_DIR" && find HiFiShifter.vst3 clap -type f 2>/dev/null )"
    # lipo invalidates the signature of every file it touches, and a bundle's signature covers
    # all of them: re-sign after the merge, never before.
    if wants vst3; then
        hfs_sign_bundle "$OUT_DIR/HiFiShifter.vst3" "$OUT_DIR/HiFiShifter.vst3/Contents/MacOS/HiFiShifter"
    fi
    if wants clap; then
        hfs_sign_bundle "$OUT_DIR/clap/HiFiShifter.clap" "$OUT_DIR/clap/HiFiShifter.clap/Contents/MacOS/HiFiShifter"
    fi
    log "plugins: ${merged} file(s) merged into universal binaries"
}

# ── Stage 3b: make the App bundle universal ─────────────────────────────────
#
# Tauri's `--target universal-apple-darwin` lipos the main executable and copies everything else
# from one target's bundle. A file both targets write to the *same* source path therefore stays
# single-arch in the merged bundle, and which architecture wins depends on build order -- that is
# the SoundTouch dylib, whose path tauri.conf.json's `frameworks`/`files` entries fix. Merge
# those against the other target's bundle.
#
# Resources/macos/ is excluded by design: the x86_64 slice links tract instead of onnxruntime, so
# those dylibs have no counterpart and must stay arm64-only (the workflow asserts exactly that).
merge_app_arch() {
    local other="$1" app="$APP_BUNDLE" rel merged=0
    [ "$HFS_OS" = "macos" ] || die "--merge-app-arch only makes sense on macOS"
    command -v lipo >/dev/null 2>&1 || die "lipo missing (Xcode command line tools required)"
    [ -n "$app" ] || die "--merge-app-arch requires --app-bundle"
    [ -d "$app" ] || die "--app-bundle directory does not exist: $app"
    [ -d "$other" ] || die "--merge-app-arch directory does not exist: $other"
    [ "$HFS_PLATFORM_ARCH" = "universal" ] \
        || die "--merge-app-arch produces a universal package: set HFS_PLATFORM_ARCH=universal so the artifact name says so (current tag: ${HFS_TAG})"
    step "merging the universal App bundle (${other})"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        case "$rel" in
            Contents/Resources/macos/*) continue ;;
        esac
        if merge_macho_pair "$app/$rel" "$other/$rel"; then
            log "universal: $rel"
            merged=$((merged + 1))
        fi
    done <<< "$( cd "$app" && find . -type f | sed 's|^\./||' )"
    hfs_sign_bundle "$app" "$app/Contents/MacOS/HiFiShifter"
    log "App: ${merged} file(s) merged into universal binaries"
}

# ── Stage 4: assemble distribution artifacts ────────────────────────────────
package_linux_plugins() {
    local stage="$OUT_DIR/stage-plugins" entries=()
    step "packaging the Linux plugins package"
    rm -rf "$stage"
    mkdir -p "$stage"
    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "missing $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$stage/"
        entries+=("HiFiShifter.vst3")
    fi
    if wants clap; then
        [ -f "$OUT_DIR/clap/HiFiShifter.clap" ] || die "missing $OUT_DIR/clap/HiFiShifter.clap"
        cp "$OUT_DIR/clap/HiFiShifter.clap" "$stage/"
        cp -R "$OUT_DIR/clap/HiFiShifter-resources" "$stage/"
        entries+=("HiFiShifter.clap" "HiFiShifter-resources")
    fi
    # Plugins-only mode stays "self-contained per format": hard links in the staging directory
    # let the two formats share inodes, so tar stores the second copy as a hard-link entry.
    # Unified mode instead lifts the weights to the top-level models/, written into the shared
    # store by install.sh; no hard links here.
    if wants vst3 && wants clap; then
        rm -rf "$stage/HiFiShifter-resources/models"
        cp -al "$stage/HiFiShifter.vst3/Contents/Resources/models" "$stage/HiFiShifter-resources/models"
    fi
    # [Why the emptiness check] macOS ships bash 3.2, where ${arr[@]} on an empty array exits
    # with "unbound variable" (set -u). An empty array can only come from a mistyped --formats;
    # give a human-readable message in that case instead of making the user decode array expansion.
    [ ${#entries[@]} -gt 0 ] || die "no format selected: --formats=$FORMATS"
    cp "$REPO_ROOT/scripts/plugin/unix/install.sh" "$stage/install.sh"
    chmod +x "$stage/install.sh"
    hfs_stage_licenses "$stage" "$REPO_ROOT"
    hfs_tarball "$ARTIFACT_BASE.tar.gz" "$stage" "${entries[@]}" install.sh LICENSE licenses
    rm -rf "$stage"
}

package_macos_plugins() {
    local root="$OUT_DIR/pkgroot-plugins" numeric
    step "packaging the macOS plugins pkg"
    command -v pkgbuild >/dev/null 2>&1 || die "pkgbuild missing (Xcode command line tools required)"
    rm -rf "$root"
    mkdir -p "$root/Library/Audio/Plug-Ins/VST3" "$root/Library/Audio/Plug-Ins/CLAP"
    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "missing $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$root/Library/Audio/Plug-Ins/VST3/"
    fi
    if wants clap; then
        [ -d "$OUT_DIR/clap/HiFiShifter.clap" ] || die "missing $OUT_DIR/clap/HiFiShifter.clap"
        cp -R "$OUT_DIR/clap/HiFiShifter.clap" "$root/Library/Audio/Plug-Ins/CLAP/"
    fi
    numeric="$(hfs_numeric_version "$PRODUCT_VERSION")"
    rm -f "$ARTIFACT_BASE.pkg"
    pkgbuild --root "$root" --identifier com.hifishifter.plugins --version "$numeric" \
        --install-location / --ownership recommended "$ARTIFACT_BASE.pkg" >/dev/null \
        || die "pkgbuild failed"
    rm -rf "$root"
    sign_macos_pkg "$ARTIFACT_BASE.pkg"
}

# Developer ID-signs the pkg (optional notarization) and writes the sha256. Both pkg kinds share
# this function, so the unified and plugins-only packages never diverge in signing/verification.
sign_macos_pkg() {
    local pkg="$1" identity digest
    identity="${HIFISHIFTER_CODESIGN_IDENTITY:--}"
    if [ "$identity" != "-" ]; then
        rm -f "$pkg.signed"
        productsign --sign "$identity" "$pkg" "$pkg.signed" >/dev/null \
            || die "productsign failed"
        mv -f "$pkg.signed" "$pkg"
        log "pkg signed with $identity"
        if [ -n "${HIFISHIFTER_NOTARY_PROFILE:-}" ]; then
            step "notarization (notarytool)"
            xcrun notarytool submit "$pkg" \
                --keychain-profile "$HIFISHIFTER_NOTARY_PROFILE" --wait || die "notarization failed"
            xcrun stapler staple "$pkg" || die "stapler stapling failed"
            log "notarized and stapled"
        else
            log "HIFISHIFTER_NOTARY_PROFILE not set: skipping notarization (Gatekeeper will block the first user load)"
        fi
    else
        log "ad-hoc signing: enough for CI self-checks only; switch to Developer ID before distributing"
    fi
    digest="$(hfs_sha256 "$pkg")"
    printf '%s  %s\n' "$digest" "$(basename "$pkg")" >"$pkg.sha256"
    log "artifact $(du -h "$pkg" | cut -f1): $pkg"
}

# Linux unified: AppImage + VST3 + CLAP + one shared model store + install.sh.
# All bundles keep only models.json; the weights live in the top-level models/<version>,
# installed by install.sh into the XDG data directory (with --system: /usr/local/share/HiFiShifter).
package_linux_unified() {
    local stage="$OUT_DIR/stage-unified" entries=(
        "HiFiShifter.AppImage" "models" "install.sh" "LICENSE" "licenses"
    )
    step "packaging the Linux unified installer"
    [ -f "$APPIMAGE" ] || die "--appimage file does not exist: $APPIMAGE"
    rm -rf "$stage"
    mkdir -p "$stage"
    cp -f "$APPIMAGE" "$stage/HiFiShifter.AppImage"
    chmod +x "$stage/HiFiShifter.AppImage"

    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "missing $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$stage/"
        hfs_strip_models_to_manifest "$stage/HiFiShifter.vst3/Contents/Resources/models"
        entries+=("HiFiShifter.vst3")
    fi
    if wants clap; then
        [ -f "$OUT_DIR/clap/HiFiShifter.clap" ] || die "missing $OUT_DIR/clap/HiFiShifter.clap"
        cp "$OUT_DIR/clap/HiFiShifter.clap" "$stage/"
        cp -R "$OUT_DIR/clap/HiFiShifter-resources" "$stage/"
        hfs_strip_models_to_manifest "$stage/HiFiShifter-resources/models"
        entries+=("HiFiShifter.clap" "HiFiShifter-resources")
    fi

    mkdir -p "$stage/models/$MODEL_VERSION"
    cp -R "$MODELS_SRC/." "$stage/models/$MODEL_VERSION/"
    [ -f "$stage/models/$MODEL_VERSION/models.json" ] \
        || die "models.json missing from the shared model directory"

    cp "$REPO_ROOT/scripts/plugin/unix/install.sh" "$stage/install.sh"
    chmod +x "$stage/install.sh"
    hfs_stage_licenses "$stage" "$REPO_ROOT"
    hfs_tarball "$UNIFIED_BASE.tar.gz" "$stage" "${entries[@]}"
    rm -rf "$stage"
}

# macOS unified: /Applications/HiFiShifter.app + VST3 + CLAP + one shared model store.
# The pkg installs at root /, so the directory structure mirrors the real system; the App and
# plugin bundles keep only models.json, and the weights go to
# /Library/Application Support/HiFiShifter/models/<version>.
package_macos_unified() {
    local root="$OUT_DIR/pkgroot-unified" numeric app
    step "packaging the macOS unified pkg"
    command -v pkgbuild >/dev/null 2>&1 || die "pkgbuild missing (Xcode command line tools required)"
    [ -d "$APP_BUNDLE" ] || die "--app-bundle directory does not exist: $APP_BUNDLE"
    [ "$(basename "$APP_BUNDLE")" = "HiFiShifter.app" ] \
        || die "--app-bundle must be HiFiShifter.app: $APP_BUNDLE"

    rm -rf "$root"
    mkdir -p "$root/Applications" \
             "$root/Library/Audio/Plug-Ins/VST3" \
             "$root/Library/Audio/Plug-Ins/CLAP" \
             "$root/Library/Application Support/HiFiShifter/models/$MODEL_VERSION"

    cp -R "$APP_BUNDLE" "$root/Applications/"
    app="$root/Applications/HiFiShifter.app"
    hfs_strip_models_to_manifest "$app/Contents/Resources/models"
    hfs_sign_bundle "$app" "$app/Contents/MacOS/HiFiShifter"

    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "missing $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$root/Library/Audio/Plug-Ins/VST3/"
        hfs_strip_models_to_manifest \
            "$root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3/Contents/Resources/models"
        hfs_sign_bundle \
            "$root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3" \
            "$root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3/Contents/MacOS/HiFiShifter"
    fi
    if wants clap; then
        [ -d "$OUT_DIR/clap/HiFiShifter.clap" ] || die "missing $OUT_DIR/clap/HiFiShifter.clap"
        cp -R "$OUT_DIR/clap/HiFiShifter.clap" "$root/Library/Audio/Plug-Ins/CLAP/"
        hfs_strip_models_to_manifest \
            "$root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap/Contents/Resources/models"
        hfs_sign_bundle \
            "$root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap" \
            "$root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap/Contents/MacOS/HiFiShifter"
    fi

    cp -R "$MODELS_SRC/." "$root/Library/Application Support/HiFiShifter/models/$MODEL_VERSION/"
    cp -f "$REPO_ROOT/LICENSE" "$root/Library/Application Support/HiFiShifter/LICENSE"
    cp -R "$REPO_ROOT/licenses" "$root/Library/Application Support/HiFiShifter/licenses"

    numeric="$(hfs_numeric_version "$PRODUCT_VERSION")"
    rm -f "$UNIFIED_BASE.pkg"
    pkgbuild --root "$root" --identifier com.hifishifter.unified --version "$numeric" \
        --install-location / --ownership recommended "$UNIFIED_BASE.pkg" >/dev/null \
        || die "pkgbuild failed"
    rm -rf "$root"
    sign_macos_pkg "$UNIFIED_BASE.pkg"
}

# ── Entry point ─────────────────────────────────────────────────────────────
if [ -n "$APPIMAGE" ] && [ "$HFS_OS" != "linux" ]; then
    die "--appimage only makes sense on Linux"
fi
if [ -n "$APP_BUNDLE" ] && [ "$HFS_OS" != "macos" ]; then
    die "--app-bundle only makes sense on macOS"
fi
if [ -n "$APPIMAGE" ] && [ ! -f "$APPIMAGE" ]; then
    die "--appimage file does not exist: $APPIMAGE"
fi
if [ -n "$APP_BUNDLE" ] && [ ! -d "$APP_BUNDLE" ]; then
    die "--app-bundle directory does not exist: $APP_BUNDLE"
fi
if [ -n "$MERGE_APP_ARCH" ] && [ -z "$APP_BUNDLE" ]; then
    die "--merge-app-arch requires --app-bundle: the App bundle is what gets merged"
fi

stage_frontend
if wants vst3; then build_format vst3; verify_format vst3; fi
if wants clap; then build_format clap; verify_format clap; fi
if [ -n "$MERGE_ARCH" ]; then merge_arch "$MERGE_ARCH"; fi
if [ -n "$MERGE_APP_ARCH" ]; then merge_app_arch "$MERGE_APP_ARCH"; fi

if [ "$HFS_OS" = "linux" ]; then
    if [ -n "$APPIMAGE" ]; then package_linux_unified; else package_linux_plugins; fi
else
    if [ -n "$APP_BUNDLE" ]; then package_macos_unified; else package_macos_plugins; fi
fi

log "done"
