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
#         --merge-arch=DIR --merge-app-arch=OUTDIR --merge-app-arch=OUTDIR \
#                                                      # ...and both architectures in one package
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
#   macOS   dist/HiFiShifter_v<version>_<os>-<arch>-unified.pkg
#             a product archive: welcome / readme / licence / conclusion pages, one
#             selectable choice per plug-in format, and the payload split so that "app
#             only" is a real answer. Installs to /Applications and /Library.
#           dist/HiFiShifter_v<version>_<os>-<arch>-uninstaller.pkg
#             removes every file and receipt the installer wrote, system-wide and
#             per-user. A copy also rides in the payload, so it is still on the machine
#             after the download is gone.
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
  --merge-app-arch=DIR  merge the App's single-arch Mach-Os from one architecture's cargo
                        output directory (backend/target/<triple>/<profile>); repeat once per
                        architecture. macOS only
  --appimage=FILE       Linux unified: install an existing AppImage into the tar.gz too
  --app-bundle=DIR      macOS unified: install an existing .app into the pkg too
                        (macOS always also produces the -uninstaller.pkg artifact)
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
        --merge-app-arch=*) MERGE_APP_ARCH="${MERGE_APP_ARCH}${1#*=}"$'\n'; shift ;;
        --merge-app-arch) [ $# -ge 2 ] || die "--merge-app-arch requires a value"; MERGE_APP_ARCH="${MERGE_APP_ARCH}${2}"$'\n'; shift 2 ;;
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
UNINSTALLER_BASE="$OUT_DIR/HiFiShifter_v${PRODUCT_VERSION}_${HFS_TAG}-uninstaller"
# CFBundleVersion and pkg versions only accept dot-separated integers; the full version
# (including -beta) travels in the artifact name and in the bundles' custom Info.plist key.
MACOS_NUMERIC_VERSION="$(hfs_numeric_version "$PRODUCT_VERSION")"

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
# Tauri lipos the main executable (crates/tauri-cli/src/interface/rust/desktop.rs: it runs
# `lipo -create -output <out_dir>/HiFiShifter <aarch64>/HiFiShifter <x86_64>/HiFiShifter` and
# never bundles per target), so the .app is assembled once, from whichever architecture's files
# the shared source paths happen to hold. A file both builds write to the *same* source path
# therefore stays single-arch, and which architecture wins depends on build order -- that is the
# SoundTouch dylib, whose path tauri.conf.json's `frameworks`/`files` entries fix.
#
# [Why the sources are cargo output directories and not another .app] There is no other .app to
# point at. What exists per target is `backend/target/<triple>/<profile>/`, which holds that
# architecture's `libSoundTouchDLL.dylib` (kernel build.rs Step 5 copies it there) next to its
# binary. Matching by **basename** is exact rather than a guess: the bundle copies a fixed
# filename from a fixed path, so the same filename in a target's output directory is the same
# file built for that target.
#
# [Why the App's own copy is not one of the inputs] It is whatever the last build wrote to the
# shared source path -- x86_64, because tauri-cli builds aarch64 first and x86_64 second. Lipoing
# that against the other target would fail ("same architecture"), and lipoing it against itself
# proves nothing. The two target directories are the authoritative per-architecture sources, so
# the merged dylib is a function of the two builds and not of build order.
#
# Resources/macos/ is excluded by design: the x86_64 slice links ort-tract instead of
# onnxruntime, so those dylibs have no counterpart and must stay arm64-only (the workflow
# asserts exactly that).
#
# <dir>... : one directory per architecture (see the option's help text)
merge_app_arch() {
    local app="$APP_BUNDLE" rel base merged=0 dir cand archs seen files
    local -a dirs=() picked=()
    [ "$HFS_OS" = "macos" ] || die "--merge-app-arch only makes sense on macOS"
    command -v lipo >/dev/null 2>&1 || die "lipo missing (Xcode command line tools required)"
    [ -n "$app" ] || die "--merge-app-arch requires --app-bundle"
    [ -d "$app" ] || die "--app-bundle directory does not exist: $app"
    [ "$HFS_PLATFORM_ARCH" = "universal" ] \
        || die "--merge-app-arch produces a universal package: set HFS_PLATFORM_ARCH=universal so the artifact name says so (current tag: ${HFS_TAG})"
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        [ -d "$dir" ] || die "--merge-app-arch directory does not exist: $dir"
        dirs+=("$dir")
    done <<< "$MERGE_APP_ARCH"
    [ ${#dirs[@]} -gt 0 ] || die "--merge-app-arch needs at least one directory"

    step "merging the universal App bundle (${dirs[*]})"
    # [Why the list is materialised first] The loop replaces files in the tree it is walking
    # (lipo writes <file>.universal, then renames it over <file>). Streaming `find` through a
    # process substitution would be reading a directory while it changes; a here-string makes
    # the walk a snapshot.
    files="$( cd "$app" && find . -type f | sed 's|^\./||' )"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        # Architecture-exclusive by design (see the comment above).
        case "$rel" in
            Contents/Resources/macos/*) continue ;;
        esac
        # Tauri already lipo'd the main executable; leave it alone.
        if hfs_is_universal "$app/$rel"; then continue; fi
        base="$(basename "$rel")"
        picked=()
        seen="|"
        for dir in "${dirs[@]}"; do
            cand="$dir/$base"
            [ -f "$cand" ] || continue
            file -b "$cand" | grep -q 'Mach-O' || continue
            archs="$(hfs_macho_archs "$cand" | tr '\n' ',')"
            [ -n "$archs" ] || continue
            case "$seen" in *"|$archs|"*) continue ;; esac
            seen="$seen$archs|"
            picked+=("$cand")
        done
        [ ${#picked[@]} -ge 2 ] || continue
        lipo -create "${picked[@]}" -output "$app/$rel.universal" || die "lipo merge failed: $rel"
        mv -f "$app/$rel.universal" "$app/$rel"
        log "universal: $rel"
        merged=$((merged + 1))
    done <<< "$files"
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

# macOS: component packages + a product archive, and the standalone uninstaller.
#
# [Why the payload is split into component packages] A product archive is assembled from
# component packages, one per choice. Installer records a receipt per component and installs
# only the ones the user left selected, so the split is what makes "I only want the app" a
# real answer -- "pkgbuild --root" over one tree can only produce one all-or-nothing choice,
# which is exactly what this script used to do.
#
# [Why the Distribution XML and its resources are generated] Installer localizes in exactly
# two ways: .lproj folders, and Localizable.strings -- the choice title/description
# attributes are localization keys. Both are derived from the same five-language catalog the
# Windows installer reads, by scripts/release/build-macos-pkg.mjs.
#
# [Why each component package is named after its choice] The Distribution references
# "<choice>.pkg" and this script writes "$pkgs/<choice>.pkg", so there is no third list of
# filenames to keep in step.

# <choice> <root-dir> <output-pkg>
macos_component_pkg() {
    local choice="$1" root="$2" out="$3"
    rm -f "$out"
    pkgbuild --root "$root" \
        --identifier "com.arounder.hifishifter.$choice" \
        --version "$MACOS_NUMERIC_VERSION" \
        --install-location / --ownership recommended "$out" >/dev/null \
        || die "pkgbuild failed for the '$choice' component"
}

# <tool>... : the pkg tools are Xcode command line tools, node runs the layout generator.
# Checked together so the failure names the reason instead of "command not found".
require_macos_pkg_tools() {
    command -v pkgbuild >/dev/null 2>&1 || die "pkgbuild missing (Xcode command line tools required)"
    command -v productbuild >/dev/null 2>&1 || die "productbuild missing (Xcode command line tools required)"
    command -v node >/dev/null 2>&1 || die "node missing: the installer pages are generated from frontend/src/i18n"
}

# The layout generator compiles frontend/src/i18n/*.ts with the esbuild the frontend already
# depends on (see scripts/release/catalog.mjs for why it is compiled rather than parsed).
#
# [Why this installs instead of failing] --reuse-frontend exists precisely to skip the
# frontend build, so it never installs anything: on a fresh checkout there is no esbuild to
# compile the catalog with. The workflow happens to run "npm --prefix frontend ci" before this
# step, which makes the function a no-op there; locally it is the difference between a clear
# wait and "Cannot find module 'esbuild'".
ensure_macos_pkg_node_deps() {
    [ -d "$REPO_ROOT/frontend/node_modules/esbuild" ] && return 0
    log "frontend/node_modules is missing: installing it (the installer pages come from frontend/src/i18n)"
    ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
        || die "npm ci failed: the installer pages are generated from frontend/src/i18n"
}

# The uninstaller: a product archive whose single component package carries no payload, only
# the removal script. Published next to the installer and copied into the installer's payload.
package_macos_uninstaller() {
    local work="$OUT_DIR/pkg-uninstaller"
    step "packaging the macOS uninstaller pkg"
    require_macos_pkg_tools
    ensure_macos_pkg_node_deps
    rm -rf "$work"
    mkdir -p "$work/pkgs" "$work/scripts"
    cp -f "$REPO_ROOT/scripts/release/macos-pkg/uninstall.sh" "$work/scripts/postinstall"
    chmod +x "$work/scripts/postinstall"
    pkgbuild --nopayload --identifier com.arounder.hifishifter.uninstaller \
        --version "$MACOS_NUMERIC_VERSION" --scripts "$work/scripts" \
        "$work/pkgs/uninstaller.pkg" >/dev/null || die "pkgbuild failed (uninstaller)"
    node "$REPO_ROOT/scripts/release/build-macos-pkg.mjs" --kind=uninstall \
        --out="$work/resources" --version="$MACOS_NUMERIC_VERSION" \
        || die "generating the macOS uninstaller layout failed"
    rm -f "$UNINSTALLER_BASE.pkg"
    productbuild --distribution "$work/resources/distribution.xml" \
        --package-path "$work/pkgs" --resources "$work/resources" \
        "$UNINSTALLER_BASE.pkg" >/dev/null || die "productbuild failed (uninstaller)"
    rm -rf "$work"
    sign_macos_pkg "$UNINSTALLER_BASE.pkg"
}

# macOS unified: /Applications/HiFiShifter.app + VST3 + CLAP + one shared model store.
# The pkg installs at root /, so the directory structure mirrors the real system; the App and
# plugin bundles keep only models.json, and the weights go to
# /Library/Application Support/HiFiShifter/models/<version>.
package_macos_unified() {
    local work="$OUT_DIR/pkg-unified" pkgs="$OUT_DIR/pkg-unified/pkgs"
    local app_root="$work/root-app" models_root="$work/root-models"
    local vst3_root="$work/root-vst3" clap_root="$work/root-clap"
    local app shared="$work/root-app/Library/Application Support/HiFiShifter"

    step "packaging the macOS unified pkg"
    require_macos_pkg_tools
    ensure_macos_pkg_node_deps
    [ -d "$APP_BUNDLE" ] || die "--app-bundle directory does not exist: $APP_BUNDLE"
    [ "$(basename "$APP_BUNDLE")" = "HiFiShifter.app" ] \
        || die "--app-bundle must be HiFiShifter.app: $APP_BUNDLE"
    [ -f "$UNINSTALLER_BASE.pkg" ] \
        || die "the uninstaller pkg is missing: $UNINSTALLER_BASE.pkg"

    rm -rf "$work"
    mkdir -p "$pkgs" \
             "$app_root/Applications" \
             "$models_root/Library/Application Support/HiFiShifter/models/$MODEL_VERSION" \
             "$vst3_root/Library/Audio/Plug-Ins/VST3" \
             "$clap_root/Library/Audio/Plug-Ins/CLAP"

    # The App bundle: strip its weights to the manifest (they exist once, in the shared
    # store), then sign -- after the copy, because the signature covers every file inside.
    cp -R "$APP_BUNDLE" "$app_root/Applications/"
    app="$app_root/Applications/HiFiShifter.app"
    hfs_strip_models_to_manifest "$app/Contents/Resources/models"
    hfs_sign_bundle "$app" "$app/Contents/MacOS/HiFiShifter"

    # Licences and the uninstaller ride with the App component: it cannot be deselected, so
    # they are always there, and the removal script deletes them together with it.
    mkdir -p "$shared"
    cp -f "$REPO_ROOT/LICENSE" "$shared/LICENSE"
    cp -R "$REPO_ROOT/licenses" "$shared/licenses"
    cp -f "$UNINSTALLER_BASE.pkg" "$shared/Uninstall HiFiShifter.pkg"

    cp -R "$MODELS_SRC/." \
        "$models_root/Library/Application Support/HiFiShifter/models/$MODEL_VERSION/"

    if wants vst3; then
        [ -d "$OUT_DIR/HiFiShifter.vst3" ] || die "missing $OUT_DIR/HiFiShifter.vst3"
        cp -R "$OUT_DIR/HiFiShifter.vst3" "$vst3_root/Library/Audio/Plug-Ins/VST3/"
        hfs_strip_models_to_manifest \
            "$vst3_root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3/Contents/Resources/models"
        hfs_sign_bundle \
            "$vst3_root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3" \
            "$vst3_root/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3/Contents/MacOS/HiFiShifter"
    fi
    if wants clap; then
        [ -d "$OUT_DIR/clap/HiFiShifter.clap" ] || die "missing $OUT_DIR/clap/HiFiShifter.clap"
        cp -R "$OUT_DIR/clap/HiFiShifter.clap" "$clap_root/Library/Audio/Plug-Ins/CLAP/"
        hfs_strip_models_to_manifest \
            "$clap_root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap/Contents/Resources/models"
        hfs_sign_bundle \
            "$clap_root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap" \
            "$clap_root/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap/Contents/MacOS/HiFiShifter"
    fi

    macos_component_pkg app "$app_root" "$pkgs/app.pkg"
    macos_component_pkg models "$models_root" "$pkgs/models.pkg"
    if wants vst3; then macos_component_pkg vst3 "$vst3_root" "$pkgs/vst3.pkg"; fi
    if wants clap; then macos_component_pkg clap "$clap_root" "$pkgs/clap.pkg"; fi

    node "$REPO_ROOT/scripts/release/build-macos-pkg.mjs" --kind=install \
        --out="$work/resources" --version="$MACOS_NUMERIC_VERSION" --formats="$FORMATS" \
        || die "generating the macOS pkg layout failed"

    rm -f "$UNIFIED_BASE.pkg"
    productbuild --distribution "$work/resources/distribution.xml" \
        --package-path "$pkgs" --resources "$work/resources" \
        "$UNIFIED_BASE.pkg" >/dev/null || die "productbuild failed"
    rm -rf "$work"
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
    # The uninstaller comes first: the unified package embeds a copy of it, and a
    # plugins-only package still needs one published next to it.
    package_macos_uninstaller
    if [ -n "$APP_BUNDLE" ]; then package_macos_unified; else package_macos_plugins; fi
fi

log "done"
