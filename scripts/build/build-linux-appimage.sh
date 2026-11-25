#!/usr/bin/env bash
set -euo pipefail

# Build the HiFiShifter AppImage on Linux.
#
# Usage: ./scripts/build/build-linux-appimage.sh [--release|--debug]
#
# Prerequisites:
#   1. Rust toolchain (stable)
#   2. System dependencies (see scripts/setup/install_deps_linux.sh)
#   3. appimagetool (if missing, linux-appimage.sh downloads it to ~/.cache itself)
#   4. Frontend already built (cd frontend && npm ci && npm run build)
#   5. SoundTouch sources in place (see build.rs's automatic clone)
#
# This script only does "compile + let Tauri assemble the AppDir". **What actually goes into the
# bundle, and whether it is right**, is owned by scripts/build/linux-appimage.sh; CI calls the
# same script and the same package subcommand: prune (hand the desktop stack back to the host)
# → verify (ldd assertions) → repack → smoke.
#
# [Why one shared implementation] Historically the local script and CI used two different library
# lists: CI overrode tauri.linux.conf.json to inject a hand-written mapping, while the local
# script injected nothing. So "it builds fine in my WSL" never proved the CI artifact was fine —
# that is exactly how the 0.1.0-beta.13 EGL crash escaped. After moving to a single source of
# truth, a local pass means the same logic passes on CI.

PROFILE="${1:---release}"
PROFILE="${PROFILE#--}"  # strip leading --
CARGO_PROFILE="$PROFILE"
TARGET_PROFILE="$PROFILE"
if [ "$PROFILE" = "debug" ] || [ "$PROFILE" = "dev" ]; then
    # Cargo names the default debug profile dev, but the artifact directory is still target/debug/.
    CARGO_PROFILE="dev"
    TARGET_PROFILE="debug"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
BACKEND_DIR="$PROJECT_DIR/backend"

# [Why the Tauri config file is per-architecture] tauri.linux.conf.json declares only what
# **every** Linux target really produces. libwebgpu_dawn.so is an x86_64 artefact -- ORT on Linux
# ARM64 deliberately ships no dawn -- so declaring it unconditionally makes tauri-build abort with
# "resource path resources/linux/libwebgpu_dawn.so doesn't exist" before a single line compiles,
# and the AppDir is never created. The split is the same one Windows makes for DirectML.dll.
# CI passes the same file (see the tauri_extra matrix field in .github/workflows/build.yml), so a
# local pass still means the CI artifact is the same shape.
# An absolute path because this script runs cargo from backend/, while the tauri project directory
# is backend/src-tauri; the two make a relative --config ambiguous.
TAURI_CONFIG_ARG=""
case "$(uname -m)" in
    x86_64 | amd64) TAURI_CONFIG_ARG="--config $BACKEND_DIR/src-tauri/tauri.linux.x86_64.conf.json" ;;
esac

echo "=== HiFiShifter Linux AppImage Build ==="
echo "Project: $PROJECT_DIR"
echo "Profile: $PROFILE"

# Step 1: compile the Rust backend and let Tauri assemble the AppDir.
# Note: the bundling step may fail on WSL2 (linuxdeploy/FUSE), but the AppDir is already fully
# assembled by that point, so failure is tolerated here; linux-appimage.sh repacks afterwards.
#
# [Why record the main binary's timestamp] That "tolerated failure" has a fatal side effect: if
# compilation fails right away and the disk still holds the **previous build's** AppDir, Step 2
# still finds it, and the packaging script happily packs a stale binary into a fresh AppImage,
# printing SUCCESS all along. A timestamp sentinel demanding "this build really produced a new
# binary" is what allows tolerating bundling failure without also missing a compile failure.
echo ""
echo "--- Step 1: Building Rust binary + AppDir ---"
BUILD_STAMP="$(mktemp)"
trap 'rm -f "$BUILD_STAMP"' EXIT
cd "$BACKEND_DIR"
set +e
# shellcheck disable=SC2086  # TAURI_CONFIG_ARG is a deliberately word-split flag pair ("" on arm64).
HIFISHIFTER_SKIP_FRONTEND_BUILD=1 APPIMAGE_EXTRACT_AND_RUN=1 cargo tauri build \
    --bundles appimage \
    $TAURI_CONFIG_ARG \
    -- --locked --no-default-features --features onnx \
    ${CARGO_PROFILE:+--profile "$CARGO_PROFILE"} 2>&1 | tail -5
build_rc=${PIPESTATUS[0]}
set -e
if [ "$build_rc" -ne 0 ]; then
    echo "(Tauri bundling exited $build_rc — fine to continue as long as the AppDir is complete)"
fi

# Step 2: locate the AppDir. After the cargo workspace merge, artifacts land under
# backend/target/; some layouts add one more target-triple level.
echo ""
echo "--- Step 2: Locating AppDir ---"
RELEASE_DIR="$BACKEND_DIR/target/$TARGET_PROFILE"
if [ ! -d "$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
    TRIPLE_DIR="$BACKEND_DIR/target/x86_64-unknown-linux-gnu/$TARGET_PROFILE"
    if [ -d "$TRIPLE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
        RELEASE_DIR="$TRIPLE_DIR"
    fi
fi
if [ ! -d "$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" ]; then
    echo "ERROR: AppDir not found: $RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir" >&2
    echo "The Tauri build most likely failed; check the output above." >&2
    exit 1
fi
APP_BINARY="$RELEASE_DIR/bundle/appimage/HiFiShifter.AppDir/usr/bin/HiFiShifter"
if [ ! -f "$APP_BINARY" ]; then
    echo "ERROR: main binary missing from the AppDir: $APP_BINARY" >&2
    exit 1
fi
if [ ! "$APP_BINARY" -nt "$BUILD_STAMP" ]; then
    echo "ERROR: main binary was not updated by this build (compile failed but the previous AppDir is still there)" >&2
    echo "       binary: $APP_BINARY" >&2
    echo "       sentinel: $BUILD_STAMP (created before the build started)" >&2
    exit 1
fi

# Step 3: prune → verify → repack → smoke (exactly the same logic as CI).
echo ""
echo "--- Step 3: Packaging ---"
bash "$SCRIPT_DIR/linux-appimage.sh" package "$RELEASE_DIR"

echo ""
echo "=== SUCCESS ==="
ls -lh "$RELEASE_DIR"/bundle/appimage/*.AppImage
