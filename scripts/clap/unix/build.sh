#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter CLAP — macOS / Linux build entry point.
#
#     scripts/clap/unix/build.sh                  # build → assemble → verify → tar.gz
#     scripts/clap/unix/build.sh --reuse-frontend # skip npm when only Rust changed
#     scripts/clap/unix/build.sh --stage=bundle   # re-assemble the bundle only (no packaging)
#
# Layout (hosts look for files in these spots; it is not our choice):
#   Linux   dist/clap/HiFiShifter.clap              single-file module
#           dist/clap/HiFiShifter-resources/        <stem>-resources, beside the module
#   macOS   dist/clap/HiFiShifter.clap/Contents/…   bundle (Info.plist + MacOS + Frameworks + Resources)
#
# The three hard rules match the VST3 pipeline: no PowerShell; every dynamic dependency in
# the artifact must resolve inside the package ($ORIGIN / @loader_path); an artifact that
# never really dlopened is not an artifact.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly BACKEND_DIR="$REPO_ROOT/backend"
readonly API_NAME="HiFiShifter"
readonly FRONTEND_DIST="$REPO_ROOT/frontend/dist"
readonly MODELS_SRC="$BACKEND_DIR/src-tauri/resources/models"

# shellcheck source=../../plugin/unix/lib.sh
. "$REPO_ROOT/scripts/plugin/unix/lib.sh"

STAGE="all"
PROFILE="release"
OUT_DIR="$REPO_ROOT/dist"
TARGET_TRIPLE=""
REUSE_FRONTEND=0
SKIP_VERIFY=0

log()  { printf '[clap] %s\n' "$*"; }
step() { printf '\n[clap] ── %s\n' "$*"; }
die()  { printf '[clap] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: scripts/clap/unix/build.sh [options]

  --stage=all|build|bundle|verify|package
                                     stop after a given stage (default all)
  --profile=release|dist|dev         cargo profile (default release)
  --out=DIR                          artifact directory (default dist)
  --target=TRIPLE                    cross-compile for TRIPLE (e.g. x86_64-apple-darwin);
                                     the platform rules then describe the target, not the host
  --reuse-frontend                   reuse the existing frontend/dist (it must already exist)
  --no-verify                        skip self-checks
  -h, --help                         show this help
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage=*)        STAGE="${1#*=}"; shift ;;
        --stage)          STAGE="$2"; shift 2 ;;
        --profile=*)      PROFILE="${1#*=}"; shift ;;
        --out=*)          OUT_DIR="${1#*=}"; shift ;;
        --out)            OUT_DIR="$2"; shift 2 ;;
        --target=*)       TARGET_TRIPLE="${1#*=}"; shift ;;
        --target)         [ $# -ge 2 ] || die "--target requires a value"; TARGET_TRIPLE="$2"; shift 2 ;;
        --reuse-frontend) REUSE_FRONTEND=1; shift ;;
        --no-verify)      SKIP_VERIFY=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "unknown option: $1 (see --help)" ;;
    esac
done

case "$STAGE" in
    all|build|bundle|verify|package) ;;
    *) die "--stage must be one of all|build|bundle|verify|package, got: $STAGE" ;;
esac

# [Why HFS_ARCH follows the triple] see the matching note in vst3/unix/build.sh: the
# runtime-dependency list and the artifact tag describe the artifact, not the build machine.
if [ -n "$TARGET_TRIPLE" ]; then
    HFS_ARCH="$(hfs_arch_of_triple "$TARGET_TRIPLE")"
    export HFS_ARCH
fi
hfs_detect_platform
# cargo puts a cross build under <target-root>/<triple>/<profile>.
CARGO_TARGET_ROOT="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
TARGET_DIR="$CARGO_TARGET_ROOT${TARGET_TRIPLE:+/$TARGET_TRIPLE}"
PROFILE_DIR="$PROFILE"
[ "$PROFILE" = "dev" ] && PROFILE_DIR="debug"
PRODUCT_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$BACKEND_DIR/src-tauri/tauri.conf.json")"
readonly TARGET_DIR PROFILE_DIR PRODUCT_VERSION

if [ "$HFS_OS" = "linux" ]; then
    CARGO_LIB="libhifishifter_clap.so"
    BUNDLE_DIR="$OUT_DIR/clap"
    MODULE="$BUNDLE_DIR/$API_NAME.clap"
    RESOURCES="$BUNDLE_DIR/$API_NAME-resources"
    BUNDLE_ID="com.hifishifter.clap"
else
    CARGO_LIB="libhifishifter_clap.dylib"
    BUNDLE_DIR="$OUT_DIR/clap/$API_NAME.clap"
    MODULE="$BUNDLE_DIR/Contents/MacOS/$API_NAME"
    RESOURCES="$BUNDLE_DIR/Contents/Resources"
    BUNDLE_ID="com.hifishifter.clap"
fi
ARTIFACT="$OUT_DIR/${API_NAME}_v${PRODUCT_VERSION}_${HFS_TAG}-clap.tar.gz"

log "platform ${HFS_TAG} | version ${PRODUCT_VERSION} | profile ${PROFILE} | output $OUT_DIR"
if [ -n "$TARGET_TRIPLE" ]; then
    log "cross-compiling for ${TARGET_TRIPLE} (artifacts under ${TARGET_DIR})"
fi

# ── Stage 1: compile ─────────────────────────────────────────────────────────
stage_build() {
    step "building hifishifter-clap (profile=${PROFILE})"
    command -v cargo >/dev/null 2>&1 || die "cargo missing"
    mkdir -p "$OUT_DIR"
    local cargo_log="$TARGET_DIR/cargo-clap.log"
    mkdir -p "$(dirname "$cargo_log")"
    # Exit code captured separately, color forcibly off, failures located by error line:
    # see hfs_report_build_failure in lib.sh and the matching comments in vst3/unix/build.sh.
    local status=0
    local -a cargo_args=(--manifest-path "$BACKEND_DIR/Cargo.toml" -p hifishifter-clap
                         --profile "$PROFILE" --locked)
    # macOS ships bash 3.2, where "${arr[@]}" on an empty array aborts under set -u; appending
    # only when a triple was asked for keeps the array non-empty in the native case.
    [ -n "$TARGET_TRIPLE" ] && cargo_args+=(--target "$TARGET_TRIPLE")
    CARGO_TERM_COLOR=never cargo build "${cargo_args[@]}" >"$cargo_log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        hfs_report_build_failure "$cargo_log" "$status"
        die "build failed (exit ${status}); full log: $cargo_log"
    fi
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "build succeeded but the artifact is missing: $lib"
    log "artifact ${lib} ($(du -h "$lib" | cut -f1))"
}

# ── Stage 2: frontend artifacts ──────────────────────────────────────────────
stage_frontend() {
    if [ "$REUSE_FRONTEND" = "1" ]; then
        [ -s "$FRONTEND_DIST/plugin.html" ] || die "--reuse-frontend given, but $FRONTEND_DIST/plugin.html does not exist"
        log "reusing existing frontend artifacts per --reuse-frontend"
        return
    fi
    step "building the frontend (vite, with the plugin.html entry)"
    command -v node >/dev/null 2>&1 || die "node missing: need ^20.19.0 or >=22.12.0"
    if [ ! -d "$REPO_ROOT/frontend/node_modules" ]; then
        ( cd "$REPO_ROOT/frontend" && npm ci --include=dev --no-audit --no-fund ) \
            || die "npm ci failed (check the network and frontend/package-lock.json)"
    fi
    ( cd "$REPO_ROOT/frontend" && npm run build ) || die "frontend build failed"
    [ -s "$FRONTEND_DIST/plugin.html" ] || die "frontend build finished, but $FRONTEND_DIST/plugin.html does not exist"
    log "frontend artifacts ready: $FRONTEND_DIST"
}

stage_models_fetch() {
    step "model weights"
    python3 "$REPO_ROOT/scripts/setup/fetch-models.py" || die "failed to fetch model weights"
    python3 "$REPO_ROOT/scripts/setup/fetch-models.py" --fetch-game || die "failed to fetch the GAME model"
}

# ── Stage 3: assemble ────────────────────────────────────────────────────────
#
# [Why runtime dependencies must come along] hifishifter-clap/build.rs declares
# -Wl,-rpath,$ORIGIN (macOS: @loader_path/../Frameworks). That means "libraries are allowed
# to be found next to the module", not "the libraries will be there". The old CLAP script
# copied only the module, so ldd reported not found and macOS dlopen failed outright with
# Library not loaded: @rpath/libSoundTouchDLL.dylib — the promise lived in code, nobody
# kept it.
stage_bundle() {
    step "assembling CLAP (${HFS_TAG})"
    local lib="$TARGET_DIR/$PROFILE_DIR/$CARGO_LIB"
    [ -f "$lib" ] || die "build artifact missing: ${lib} (run --stage=build first)"

    rm -rf "$OUT_DIR/clap"
    if [ "$HFS_OS" = "linux" ]; then
        mkdir -p "$BUNDLE_DIR" "$RESOURCES/frontend" "$RESOURCES/models"
        cp -f "$lib" "$MODULE"
        hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR"
    else
        mkdir -p "$BUNDLE_DIR/Contents/MacOS" "$BUNDLE_DIR/Contents/Frameworks" \
                 "$RESOURCES/frontend" "$RESOURCES/models"
        cp -f "$lib" "$MODULE"
        hfs_stage_runtime_deps "$TARGET_DIR/$PROFILE_DIR" "$BUNDLE_DIR/Contents/Frameworks"
    fi

    hfs_stage_frontend "$RESOURCES" "$FRONTEND_DIST"
    hfs_stage_models "$RESOURCES" "$MODELS_SRC"

    if [ "$HFS_OS" = "macos" ]; then
        hfs_write_info_plist "$BUNDLE_DIR/Contents/Info.plist" "$API_NAME" "$BUNDLE_ID" "$PRODUCT_VERSION" "!hfc"
        hfs_sign_bundle "$BUNDLE_DIR" "$MODULE"
    fi
    hfs_assert_rpath "$MODULE"
    hfs_check_no_dangling "$OUT_DIR/clap"
    log "bundle ready: $BUNDLE_DIR"
}

# ── Stage 4: verify ──────────────────────────────────────────────────────────
stage_verify() {
    step "verifying the CLAP artifact"
    # [Why the ABI oracle runs first] verify.sh's black-box probe and Rust's #[repr(C)] are
    # two copies of the same layout assumption: when both are wrong in the same way, dlopen
    # and passthrough still succeed while a real host — feeding in structs built from the
    # official headers — reads them misaligned. Letting the pinned CLAP SDK headers compute
    # size/align/offset themselves is the only evidence that breaks this loop. The CLAP
    # workflow already fetches the SDK into third_party/sdk/clap, and the clap_layout test
    # in cargo test compiles and compares it.
    step "CLAP 1.1.9 ABI layout oracle"
    (cd "$BACKEND_DIR" && cargo test -p hifishifter-clap --test clap_layout --locked -- --nocapture)
    bash "$SCRIPT_DIR/../verify.sh" "$MODULE"
}

# ── Stage 5: package ─────────────────────────────────────────────────────────
stage_package() {
    step "packaging $(basename "$ARTIFACT")"
    local stage="$OUT_DIR/stage-clap"
    rm -rf "$stage"
    mkdir -p "$stage"
    if [ "$HFS_OS" = "linux" ]; then
        cp -f "$MODULE" "$stage/"
        cp -R "$RESOURCES" "$stage/"
        hfs_stage_licenses "$stage" "$REPO_ROOT"
        hfs_tarball "$ARTIFACT" "$stage" "$API_NAME.clap" "$API_NAME-resources" LICENSE licenses
    else
        cp -R "$BUNDLE_DIR" "$stage/"
        hfs_stage_licenses "$stage" "$REPO_ROOT"
        hfs_tarball "$ARTIFACT" "$stage" "$API_NAME.clap" LICENSE licenses
    fi
    rm -rf "$stage"
}

case "$STAGE" in
    build)   stage_build ;;
    bundle)  stage_build; stage_frontend; stage_models_fetch; stage_bundle ;;
    verify)  stage_verify ;;
    package) stage_build; stage_frontend; stage_models_fetch; stage_bundle; stage_verify; stage_package ;;
    all)
        stage_build; stage_frontend; stage_models_fetch; stage_bundle
        if [ "$SKIP_VERIFY" = "1" ]; then log "skipping self-checks per --no-verify"; else stage_verify; fi
        stage_package
        ;;
esac

log "done"
