#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter VST3 — macOS / Linux build entry point.
#
#     scripts/vst3/unix/build.sh                  # one command: SDK → frontend → build → bundle → verify → tar.gz
#     scripts/vst3/unix/build.sh --skip-frontend  # skip npm when only Rust changed
#     scripts/vst3/unix/build.sh --install        # also install into ~/.vst3
#
# Artifacts land in dist/: HiFiShifter.vst3, <artifact>.tar.gz and .sha256.
#
# Three hard rules:
#   1. no PowerShell, and no dependence on any Windows script;
#   2. every dynamic dependency in the artifact must resolve inside the bundle ($ORIGIN / @loader_path);
#   3. the final assembly step is a real dlopen — an artifact that never loads is not an artifact.
#
# The platform rules (runtime deps, rpath, frontend, licenses, signing) live in scripts/plugin/unix/lib.sh,
# shared with CLAP; this script only describes VST3's own layout.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly BACKEND_DIR="$REPO_ROOT/backend"
readonly SDK_PINS="$REPO_ROOT/scripts/env/plugin-sdks.json"
readonly API_NAME="HiFiShifter"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
readonly MODELS_SRC="$BACKEND_DIR/src-tauri/resources/models"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

STAGE="all"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
SKIP_VERIFY=0
SKIP_FRONTEND=0
WANT_INSTALL=0

log()  { printf '[vst3] %s\n' "$*"; }
step() { printf '\n[vst3] ── %s\n' "$*"; }
die()  { printf '[vst3] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: scripts/vst3/unix/build.sh [options]

  --stage=all|sdk|frontend|build|bundle|verify|install|package
                                     stop after a given stage (default all)
  --profile=release|dist|dev         cargo profile (default release)
  --out=DIR                          artifact directory (default dist)
  --skip-frontend                    skip the frontend build (dist must already exist)
  --install                          install into ~/.vst3 after assembly (macOS: ~/Library/…/VST3)
  --no-verify                        skip self-checks (only when probes cannot run on this platform)
  -h, --help                         show this help
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)       STAGE="${1#*=}"; shift ;;
        --stage)         STAGE="$2"; shift 2 ;;
        --profile=*)     PROFILE="${1#*=}"; shift ;;
        --out=*)         OUT_DIR="${1#*=}"; shift ;;
        --out)           OUT_DIR="$2"; shift 2 ;;
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --install)       WANT_INSTALL=1; shift ;;
        --no-verify)     SKIP_VERIFY=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown option: $1 (see --help)" ;;
    esac
done

case "$STAGE" in
    all|sdk|frontend|build|bundle|verify|install|package) ;;
    *) die "--stage must be one of all|sdk|frontend|build|bundle|verify|install|package, got: $STAGE" ;;
esac

# ── Platform table ───────────────────────────────────────────────────────────
#
# All platform differences converge here; no platform if should appear further down.
hfs_detect_platform
if [ "$HFS_OS" = "linux" ]; then
    MODULE_SUBDIR="Contents/${HFS_PLATFORM_ARCH}-linux"
    MODULE_FILE="$API_NAME.so"
    RUNTIME_LIB_SUBDIR="$MODULE_SUBDIR"
    CARGO_LIB="libhifishifter_vst3.so"
else
    MODULE_SUBDIR="Contents/MacOS"
    MODULE_FILE="$API_NAME"
    RUNTIME_LIB_SUBDIR="Contents/Frameworks"
    CARGO_LIB="libhifishifter_vst3.dylib"
fi
readonly MODULE_SUBDIR MODULE_FILE RUNTIME_LIB_SUBDIR CARGO_LIB
readonly BUNDLE_TAG="$HFS_TAG"
readonly OUT_DIR
readonly TARGET_DIR="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
PROFILE_DIR="$PROFILE"
if [ "$PROFILE" = "dev" ]; then PROFILE_DIR="debug"; fi
readonly PROFILE_DIR
readonly BUNDLE_DIR="$OUT_DIR/$API_NAME.vst3"
readonly CARGO_LOG="$TARGET_DIR/cargo-build.log"

PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$BACKEND_DIR/src-tauri/tauri.conf.json")"
readonly PRODUCT_VERSION
readonly ARTIFACT_BASE="${API_NAME}_v${PRODUCT_VERSION}_${BUNDLE_TAG}-vst3"

# Read a field of plugin-sdks.json (python3 is the only cross-platform parser guaranteed to exist in CI images).
pin() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SDK_PINS" "$1" "$2"
}

# Optional field: prints an empty line when absent (CLAP is a header-only SDK, so its pin has no submodule).
pin_opt() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]].get(sys.argv[3], ""))' \
        "$SDK_PINS" "$1" "$2"
}

for tool in git python3 tar; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done

log "platform ${BUNDLE_TAG} | version ${PRODUCT_VERSION} | profile ${PROFILE} | output $OUT_DIR"

# ── Stage 1: SDK ─────────────────────────────────────────────────────────────
#
# ara2-bridge's build.rs runs a triple check (repo + commit + tree) on both SDKs and refuses to
# compile once the working tree is dirty. So no "best effort" here: a tree mismatch fails right
# away instead of letting cargo end 20 minutes later in an incomprehensible panic.

# Submodules are verified against the pin, using exactly the same rules as the main checkout
# (commit + tree must both match).
#
# [Why this step cannot be skipped] vst3sdk's pluginterfaces is a **submodule**: the main
# checkout only holds a gitlink, and `git checkout` will not materialize
# pluginterfaces/base/funknown.h. Without it, ara2-bridge-companion's build.rs panics at
# `pluginterfaces/base/funknown.h` — and that error is buried among hundreds of lines of World
# C++ warnings, while cargo's exit code is just 101, so the scene reads as "build failed with
# no error message".
sync_submodule() {
    local name="$1" dir="$2" sub="$3" commit="$4" tree="$5"
    local subdir="$dir/$sub" actual
    [ -n "$commit" ] && [ -n "$tree" ] || die "$name's pin is missing submoduleCommit / submoduleTree"

    # [Why two separate commands] before initialization .git does not exist yet, and only
    # `submodule update --init` can pull it down; once initialized, passing --depth is just
    # pointless network round-trips.
    # No array-based argument building: macOS's stock bash 3.2 aborts outright when expanding
    # an empty array under set -u.
    if [ -e "$subdir/.git" ]; then
        git -C "$dir" submodule update --init --force -- "$sub" >/dev/null 2>&1 \
            || die "${name}: failed to sync submodule $sub"
    else
        log "fetching $name submodule $sub@${commit:0:12}"
        git -C "$dir" submodule update --init --depth 1 --force -- "$sub" >/dev/null 2>&1 \
            || die "${name}: failed to fetch submodule $sub"
    fi

    actual="$(git -C "$subdir" rev-parse HEAD 2>/dev/null || true)"
    [ "$actual" = "$commit" ] || die "$name's submodule $sub does not match the pin: expected ${commit}, got ${actual:-(missing)}"
    actual="$(git -C "$subdir" rev-parse 'HEAD^{tree}')"
    [ "$actual" = "$tree" ] || die "$name's submodule $sub tree does not match the pin: expected ${tree}, got $actual"
    log "$name submodule $sub verified (${commit:0:12})"
}

sync_sdk() {
    local name="$1" subdir="$2"
    local repo commit tree dir actual_tree sub
    repo="$(pin "$name" repository)"
    commit="$(pin "$name" commit)"
    tree="$(pin "$name" tree)"
    dir="$REPO_ROOT/third_party/sdk/$subdir"

    if [ ! -d "$dir/.git" ]; then
        log "initializing $name checkout: $dir"
        mkdir -p "$(dirname "$dir")"
        git init -q "$dir"
        git -C "$dir" remote add origin "$repo"
    fi
    if ! git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        log "fetching $name ${commit:0:12}"
        git -C "$dir" fetch --depth 1 origin "$commit" >/dev/null 2>&1 \
            || git -C "$dir" fetch origin "$commit" >/dev/null 2>&1 \
            || die "${name}: cannot fetch commit $commit from $repo"
    fi
    git -C "$dir" checkout -q --detach --force "$commit"
    actual_tree="$(git -C "$dir" rev-parse "HEAD^{tree}")"
    [ "$actual_tree" = "$tree" ] || die "$name's tree does not match the pin: expected ${tree}, got $actual_tree"
    log "$name verified ($(git -C "$dir" rev-parse --short HEAD))"

    # Only handled when the pin carries a submodule field (header-only SDKs like CLAP have none).
    sub="$(pin_opt "$name" submodule)"
    if [ -n "$sub" ]; then
        sync_submodule "$name" "$dir" "$sub" \
            "$(pin_opt "$name" submoduleCommit)" "$(pin_opt "$name" submoduleTree)"
    fi
}

stage_sdk() {
    step "preparing ARA / VST3 SDK"
    # The two SDKs' submodules (ARA_API / pluginterfaces) are verified against the pin by sync_sdk.
    sync_sdk ara ARA_SDK
    sync_sdk vst3 vst3sdk
    export ARA_SDK_DIR="$REPO_ROOT/third_party/sdk/ARA_SDK"
    export ARA_VST3_SDK_DIR="$REPO_ROOT/third_party/sdk/vst3sdk"
    log "ARA_SDK_DIR=$ARA_SDK_DIR"
    log "ARA_VST3_SDK_DIR=$ARA_VST3_SDK_DIR"
}

# ── Stage 2: frontend artifacts ──────────────────────────────────────────────
#
# The frontend is one single artifact: the bundle's Contents/Resources/frontend is the same
# tree as on Windows. Putting npm into the same pipeline removes the classic trap of "the
# build ran locally, yet the artifact has no frontend".
stage_frontend() {
    if [ "$SKIP_FRONTEND" = "1" ]; then
        [ -f "$FRONTEND_DIST/plugin.html" ] || die "--skip-frontend given, but $FRONTEND_DIST/plugin.html does not exist"
        log "reusing existing frontend artifacts per --skip-frontend"
        return
    fi
    step "building the frontend (vite, with the plugin.html entry)"
    # vite 8 uses rolldown, whose native bindings declare engines: ^20.19.0 || >=22.12.0.
    # The errors a too-old node produces (missing styleText, @rolldown/binding-* not found)
    # all look like frontend code problems, so the real cause is called out right at the entry.
    command -v node >/dev/null 2>&1 || die "node missing: need ^20.19.0 or >=22.12.0"
    [ "$(node -p 'const [a,b]=process.versions.node.split(".").map(Number); (a===20&&b>=19)||a>22||(a===22&&b>=12)?"yes":"no"')" = "yes" ] \
        || die "node $(node -v) is too old: vite 8 needs ^20.19.0 or >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        # --include=dev: npm run build needs tsc (a devDependency), which NODE_ENV=production would skip.
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci failed (check the network and frontend/package-lock.json)"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "frontend build failed"
    [ -f "$FRONTEND_DIST/plugin.html" ] || die "frontend build finished, but $FRONTEND_DIST/plugin.html does not exist"
    log "frontend artifacts ready: $FRONTEND_DIST"
}

# ── Stage 3: compile ─────────────────────────────────────────────────────────
#
# rust-toolchain.toml pins the channel. On a machine that only ever ran `rustup default
# stable`, rustup sees this file in the repo and tries to **download** that exact version —
# offline, this shows up as "the build stage sits there doing nothing". stable is often the
# very same rustc, so reuse it when the version matches; fail loudly otherwise (silently
# swapping toolchains is worse than getting stuck).
ensure_toolchain() {
    if [ -n "${RUSTUP_TOOLCHAIN:-}" ]; then
        log "toolchain selected via RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN"
        return 0
    fi
    command -v rustup >/dev/null 2>&1 || return 0
    local pinned
    pinned="$(sed -n 's/^channel[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$REPO_ROOT/rust-toolchain.toml" 2>/dev/null | head -1)"
    [ -n "$pinned" ] || return 0
    rustup toolchain list 2>/dev/null | grep -qE "^${pinned}([-[:space:]]|$)" && return 0
    if [ "$(RUSTUP_TOOLCHAIN=stable rustc -V 2>/dev/null | awk '{ print $2 }')" = "$pinned" ]; then
        export RUSTUP_TOOLCHAIN=stable
        log "pinned $pinned not installed; reusing stable (same rustc version)"
        return 0
    fi
    die "rustup toolchain ${pinned} not found (pinned by rust-toolchain.toml).
  With network: rustup toolchain install $pinned
  or override for this run: export RUSTUP_TOOLCHAIN=<installed toolchain name>"
}

stage_build() {
    step "building hifishifter-vst3 (profile=${PROFILE})"
    command -v cargo >/dev/null 2>&1 || die "cargo missing"
    ensure_toolchain
    log "$(rustc -V 2>/dev/null || echo 'rustc version unknown')"
    mkdir -p "$OUT_DIR" "$(dirname "$CARGO_LOG")"
    # --locked: the build must prove the "committed lock" compiles, not resolve a fresh
    # dependency tree on the fly.
    # Output goes to a log: on Unix this crate emits nearly two hundred dead_code warnings
    # for Windows-only code; dumping them to the console pushes the real error out of view.
    # [Why CARGO_TERM_COLOR=never is forced] CI sets CARGO_TERM_COLOR=always, so every log
    # line carries escape sequences and "^error" stops matching — the error locating below
    # would break entirely.
    # [Why the exit code is captured separately] when a signal kills cargo it never gets to
    # write an error line; the exit code is the only clue.
    local status=0
    CARGO_TERM_COLOR=never cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" \
        -p hifishifter-vst3 --profile "$PROFILE" --locked >"$CARGO_LOG" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        hfs_report_build_failure "$CARGO_LOG" "$status"
        die "build failed (exit ${status}); full log: $CARGO_LOG"
    fi
    log "build finished ($(grep -c "^warning" "$CARGO_LOG" || true) warnings; log ${CARGO_LOG})"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "build succeeded but the artifact is missing: $lib"
    log "artifact ${lib} ($(du -h "$lib" | cut -f1))"
}

# ── Stage 4: assemble the bundle ─────────────────────────────────────────────

# On Linux the editor runs in a separate process: creating a GTK window inside the host
# process wrecks that process's rendering entirely (measured: xwd output goes from 3.1 MB to
# 0 bytes). The plugin looks for it "in the same directory as the .so".
build_gui() {
    case "$MODULE_SUBDIR" in *-linux) ;; *) return 0 ;; esac
    log "building the renderer process hfs-gui"
    # Explicit `fetch --locked` instead of --offline: the latter assumes "the plugin build
    # happened to download the same crate versions already", and any lock drift turns into a
    # red offline error far from the real cause.
    ( cd "$REPO_ROOT/backend/hfs-gui" && cargo fetch --locked && \
      cargo build --release --locked --target-dir "$TARGET_DIR/hfs-gui" ) || die "hfs-gui build failed"
    cp -f "$TARGET_DIR/hfs-gui/release/hfs-gui" "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui"
    chmod +x "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui"
    log "renderer ready ($(du -h "$BUNDLE_DIR/$MODULE_SUBDIR/hfs-gui" | cut -f1))"
}

# Model weights are not in the repo (see scripts/setup/fetch-models.py): missing ones are
# downloaded, existing ones just trigger a directory scan. CI caches the result with
# actions/cache keyed by the manifest hash, so this stage is usually a no-op.
stage_models_fetch() {
    step "model weights"
    python3 "$REPO_ROOT/scripts/setup/fetch-models.py" || die "failed to fetch model weights"
    python3 "$REPO_ROOT/scripts/setup/fetch-models.py" --fetch-game || die "failed to fetch the GAME model"
}

stage_bundle() {
    step "assembling $API_NAME.vst3 (${BUNDLE_TAG})"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "build artifact missing: ${lib} (run --stage=build first)"

    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR/$MODULE_SUBDIR" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR" \
             "$BUNDLE_DIR/Contents/Resources/models" "$BUNDLE_DIR/Contents/Resources/frontend"
    cp -f "$lib" "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"

    hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR/$RUNTIME_LIB_SUBDIR"

    build_gui

    hfs_stage_frontend "$BUNDLE_DIR/Contents/Resources" "$FRONTEND_DIST"
    hfs_stage_models "$BUNDLE_DIR/Contents/Resources" "$MODELS_SRC"

    if [ "$HFS_OS" = "macos" ]; then
        hfs_write_info_plist "$BUNDLE_DIR/Contents/Info.plist" "$API_NAME" "com.hifishifter.vst3" "$PRODUCT_VERSION" "!hif"
        hfs_sign_bundle "$BUNDLE_DIR" "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"
    fi
    hfs_assert_rpath "$BUNDLE_DIR/$MODULE_SUBDIR/$MODULE_FILE"
    hfs_check_no_dangling "$BUNDLE_DIR"
    log "bundle ready: $BUNDLE_DIR"
}

# ── Stage 5: verify ──────────────────────────────────────────────────────────
stage_verify() {
    step "verifying the bundle"
    [ -d "$BUNDLE_DIR" ] || die "no bundle yet: ${BUNDLE_DIR} (run --stage=bundle first)"
    bash "$SCRIPT_DIR/verify.sh" "$BUNDLE_DIR"
}

# ── Stage 6: install into the user's VST3 directory (dev loop) ────────────
#
# Not executed by default: writing into the user's plugin directory must not be an implicit
# side effect of "build".
stage_install() {
    step "installing into the user's VST3 directory"
    [ -d "$BUNDLE_DIR" ] || die "no bundle yet: ${BUNDLE_DIR} (run --stage=bundle first)"
    local target
    if [ "$HFS_OS" = "macos" ]; then
        target="$HOME/Library/Audio/Plug-Ins/VST3"
    else
        target="$HOME/.vst3"
    fi
    mkdir -p "$target"
    rm -rf "${target:?}/$API_NAME.vst3"
    cp -R "$BUNDLE_DIR" "$target/$API_NAME.vst3"
    log "installed: $target/$API_NAME.vst3 ($(du -sh "$target/$API_NAME.vst3" | cut -f1))"
}

# ── Stage 7: package ─────────────────────────────────────────────────────────
stage_package() {
    step "packaging $ARTIFACT_BASE"
    local stage="$OUT_DIR/stage"
    rm -rf "$stage"
    mkdir -p "$stage"
    cp -R "$BUNDLE_DIR" "$stage/"
    # Licenses travel with the artifact: the root LICENSE (MIT + third-party list) plus the
    # licenses/ directory (full texts and notices). SoundTouch is LGPL-2.1 (dynamically linked,
    # upstream unmodified), the ARA SDK is Apache-2.0, and the vocoder weights are
    # CC BY-NC-SA 4.0 — missing any one of them is a licensing defect.
    hfs_stage_licenses "$stage" "$REPO_ROOT"
    hfs_tarball "$OUT_DIR/$ARTIFACT_BASE.tar.gz" "$stage" "$API_NAME.vst3" LICENSE licenses
    rm -rf "$stage"
}

# ── Entry ────────────────────────────────────────────────────────────────────
case "$STAGE" in
    sdk)      stage_sdk ;;
    frontend) stage_frontend ;;
    build)    stage_sdk; stage_frontend; stage_build ;;
    bundle)   stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle ;;
    verify)   stage_verify ;;
    install)  stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle; stage_install ;;
    package)  stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle; stage_verify; stage_package ;;
    all)
        stage_sdk; stage_frontend; stage_build; stage_models_fetch; stage_bundle
        if [ "$SKIP_VERIFY" = "1" ]; then log "skipping self-checks per --no-verify"; else stage_verify; fi
        stage_package
        ;;
esac

# --install is orthogonal to --stage: it can also install after all/package.
if [ "$WANT_INSTALL" = "1" ] && [ "$STAGE" != "install" ]; then
    stage_install
fi

log "done"
