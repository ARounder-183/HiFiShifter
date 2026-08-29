#!/usr/bin/env bash
set -euo pipefail

# Build HiFiShifter AppImage for Linux
# Usage: ./scripts/build-linux-appimage.sh [--release|--debug]
#
# Prerequisites:
#   1. Rust toolchain (stable)
#   2. System deps installed (see scripts/install_deps_linux.sh)
#   3. appimagetool on PATH (see scripts/install_deps_linux.sh)
#   4. Frontend built (cd frontend && npm ci && npm run build)
#   5. SoundTouch source cloned (see build.rs auto-clone or manual clone)

PROFILE="${1:---release}"
PROFILE="${PROFILE#--}"  # strip leading --
CARGO_PROFILE="$PROFILE"
TARGET_PROFILE="$PROFILE"
if [ "$PROFILE" = "debug" ] || [ "$PROFILE" = "dev" ]; then
    # Cargo calls the default debug profile "dev", but its output directory
    # is still target/debug/.
    CARGO_PROFILE="dev"
    TARGET_PROFILE="debug"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"

echo "=== HiFiShifter Linux AppImage Build ==="
echo "Project: $PROJECT_DIR"
echo "Profile: $PROFILE"

# Step 1: Build Rust backend and Tauri AppDir
# Note: The bundling step may fail on WSL2 (linuxdeploy/FUSE issues),
# but the AppDir is already assembled correctly before that point.
echo ""
echo "--- Step 1: Building Rust binary + AppDir ---"
cd "$BACKEND_DIR"
set +e  # bundling may fail; AppDir is already created
HIFISHIFTER_SKIP_FRONTEND_BUILD=1 cargo tauri build \
    --bundles appimage \
    -- --no-default-features --features onnx \
    ${CARGO_PROFILE:+-- --profile "$CARGO_PROFILE"} 2>&1 | tail -5
set -e
echo "(Tauri bundling step may have failed — this is OK, AppDir should be ready)"

# Step 2: Determine AppDir and output paths
TARGET_DIR="$BACKEND_DIR/src-tauri/target/release"
if [ "$TARGET_PROFILE" != "release" ] && [ -n "$TARGET_PROFILE" ]; then
    TARGET_DIR="$BACKEND_DIR/src-tauri/target/$TARGET_PROFILE"
fi
APPDIR="$TARGET_DIR/bundle/appimage/HiFiShifter.AppDir"
OUTPUT="$TARGET_DIR/bundle/appimage/HiFiShifter.AppImage"

if [ ! -d "$APPDIR" ]; then
    # Try x86_64 target-specific path
    TARGET_TRIPLE="x86_64-unknown-linux-gnu"
    TARGET_DIR="$BACKEND_DIR/src-tauri/target/$TARGET_TRIPLE/$TARGET_PROFILE"
    APPDIR="$TARGET_DIR/bundle/appimage/HiFiShifter.AppDir"
    OUTPUT="$TARGET_DIR/bundle/appimage/HiFiShifter.AppImage"
fi

if [ ! -f "$APPDIR/usr/bin/HiFiShifter" ]; then
    echo "ERROR: AppDir not found at $APPDIR"
    echo "The Tauri build may have failed. Check the output above."
    exit 1
fi

# Step 3: Create AppImage
echo ""
echo "--- Step 2: Creating AppImage ---"
echo "AppDir:  $APPDIR"
echo "Output:  $OUTPUT"

export APPIMAGE_EXTRACT_AND_RUN=1
ARCH="${ARCH:-x86_64}"

# Clean up any previous extraction artifacts
rm -rf /tmp/squashfs-root 2>/dev/null || true

appimagetool "$APPDIR" "$OUTPUT" 2>&1 | tail -3

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "=== SUCCESS ==="
    ls -lh "$OUTPUT"
    file "$OUTPUT"
else
    echo "ERROR: AppImage was not created"
    exit 1
fi
