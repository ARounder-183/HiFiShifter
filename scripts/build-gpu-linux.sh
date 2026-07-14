#!/usr/bin/env bash
# build-gpu-linux.sh
# Build the HiFiShifter Linux CUDA binary and stage shared libraries.
#
# Usage:
#   ./scripts/build-gpu-linux.sh
#
# Environment (required):
#   ORT_LIB_LOCATION  - Path to ONNX Runtime GPU lib directory
#   CUDA_LIB_PATH     - Path to CUDA runtime library directory
#
# Environment (optional):
#   CARGO_TARGET      - Rust target triple (default: x86_64-unknown-linux-gnu)
#   CARGO_FEATURES    - Cargo features (default: onnx,cuda)

set -euxo pipefail

CARGO_TARGET="${CARGO_TARGET:-x86_64-unknown-linux-gnu}"
CARGO_FEATURES="${CARGO_FEATURES:-onnx,cuda}"
SRC_TAURI="backend/src-tauri"

echo "=== Build Linux CUDA Binary ==="
echo "  Target:   ${CARGO_TARGET}"
echo "  Features: ${CARGO_FEATURES}"
echo "  ORT_LIB:  ${ORT_LIB_LOCATION:-<not set>}"
echo "  CUDA_LIB: ${CUDA_LIB_PATH:-<not set>}"

if [ -z "${ORT_LIB_LOCATION:-}" ]; then
    echo "ERROR: ORT_LIB_LOCATION is not set" >&2
    exit 1
fi
if [ -z "${CUDA_LIB_PATH:-}" ]; then
    echo "ERROR: CUDA_LIB_PATH is not set" >&2
    exit 1
fi

# ort-sys v2.0.0-rc.12 defaults to static linking (looks for .a files),
# but the ONNX Runtime GPU build only ships shared libraries (.so).
# ORT_PREFER_DYNAMIC_LINK=1 makes ort-sys emit `rustc-link-lib=onnxruntime`
# instead of `rustc-link-lib=static=onnxruntime`.
export ORT_PREFER_DYNAMIC_LINK=1

# Symlink CUDA runtime libs into ORT_LIB_LOCATION so the linker can
# resolve transitive NEEDED entries from libonnxruntime_providers_cuda.so
for lib in "${CUDA_LIB_PATH}"/libcudart.so* "${CUDA_LIB_PATH}"/libcublas*.so* "${CUDA_LIB_PATH}"/libcufft*.so* "${CUDA_LIB_PATH}"/libcurand*.so*; do
    [ -e "$lib" ] && ln -sf "$lib" "${ORT_LIB_LOCATION}/" 2>/dev/null || true
done

# -rpath,$ORIGIN: runtime linker searches binary's directory for .so files
# -rpath-link,<ORT_LIB_LOCATION>: build-time linker resolves transitive NEEDED deps
export RUSTFLAGS="-Awarnings -C link-args=-Wl,-rpath,\$ORIGIN -Wl,-rpath-link,${ORT_LIB_LOCATION}"
export LIBRARY_PATH="${ORT_LIB_LOCATION}${LIBRARY_PATH:+:}${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ORT_LIB_LOCATION}${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"

echo "=== ORT lib directory (with CUDA symlinks) ==="
ls -la "${ORT_LIB_LOCATION}/"

cd "${SRC_TAURI}"
cargo build --release --target "${CARGO_TARGET}" --no-default-features --features "${CARGO_FEATURES}"

# Copy ORT + cuDNN shared libraries alongside the binary for distribution
BIN_DIR="target/${CARGO_TARGET}/release"
if [ -n "${ORT_LIB_LOCATION:-}" ] && [ -d "${ORT_LIB_LOCATION}" ]; then
    cp -v "${ORT_LIB_LOCATION}"/*.so* "${BIN_DIR}/" 2>/dev/null || true
fi
echo "=== Release directory contents ==="
ls -la "${BIN_DIR}/"
