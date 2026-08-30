#!/usr/bin/env bash
# build-gpu-linux.sh
# Build the HiFiShifter Linux binary with WebGPU GPU support.
#
# GPU acceleration uses the ONNX Runtime WebGPU Execution Provider
# (Dawn/Vulkan backend) on Linux x86_64. Linux ARM64 builds are CPU-only
# because there is no prebuilt WebGPU ORT binary for that target.
# No extra Cargo features are needed — the `webgpu` feature is enabled
# by default in the x86_64 Linux ort dependency.
#
# Usage:
#   ./scripts/build-gpu-linux.sh
#
# Prerequisites:
#   - Rust toolchain (stable)
#   - System deps installed (see scripts/install_deps_linux.sh)
#   - Frontend built (cd frontend && npm ci && npm run build)
#
# The ort crate's `download-binaries` feature automatically downloads
# the ONNX Runtime prebuilt binaries with WebGPU support at build time
# on x86_64.
# These are statically linked, so no runtime .so deployment is needed.
#
# WebGPU requires a Vulkan-capable GPU with drivers installed.
# On headless/WSL systems without a GPU, the build still succeeds and
# gracefully falls back to CPU inference at runtime.
#   sudo apt-get install -y libvulkan1 mesa-vulkan-drivers

set -euxo pipefail

CARGO_TARGET="${CARGO_TARGET:-x86_64-unknown-linux-gnu}"
SRC_TAURI="backend/src-tauri"
WEBGPU_BUILD=false
if [[ "$CARGO_TARGET" == x86_64-unknown-linux-* ]]; then
    WEBGPU_BUILD=true
fi

echo "=== Build Linux (WebGPU) Binary ==="
echo "  Target:   ${CARGO_TARGET}"
echo "  Features: onnx (includes webgpu via ort dependency)"
if [ "$WEBGPU_BUILD" = true ]; then
    echo "  WebGPU:   enabled (x86_64)"
else
    echo "  WebGPU:   disabled (CPU-only for ${CARGO_TARGET})"
fi

cd "${SRC_TAURI}"
cargo build --release --target "${CARGO_TARGET}" --no-default-features --features onnx

BIN_DIR="target/${CARGO_TARGET}/release"
echo "=== Release directory contents ==="
ls -la "${BIN_DIR}/"
echo ""
echo "=== Build complete ==="
echo "Binary: ${BIN_DIR}/HiFiShifter"
if [ "$WEBGPU_BUILD" = true ]; then
    echo "WebGPU GPU acceleration is compiled in (falls back to CPU if no GPU available)."
else
    echo "WebGPU is not compiled for ${CARGO_TARGET}; this build is CPU-only."
fi
