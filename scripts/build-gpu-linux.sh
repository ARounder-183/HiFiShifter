#!/usr/bin/env bash
# build-gpu-linux.sh
# Build the HiFiShifter Linux GPU binary with OpenCL support.
#
# Usage:
#   ./scripts/build-gpu-linux.sh
#
# Environment (required):
#   ORT_LIB_LOCATION  - Path to ONNX Runtime library directory
#
# Environment (optional):
#   CARGO_TARGET      - Rust target triple (default: x86_64-unknown-linux-gnu)
#   CARGO_FEATURES    - Cargo features (default: onnx,opencl)

set -euxo pipefail

CARGO_TARGET="${CARGO_TARGET:-x86_64-unknown-linux-gnu}"
CARGO_FEATURES="${CARGO_FEATURES:-onnx,opencl}"
SRC_TAURI="backend/src-tauri"

echo "=== Build Linux GPU (OpenCL) Binary ==="
echo "  Target:   ${CARGO_TARGET}"
echo "  Features: ${CARGO_FEATURES}"
echo "  ORT_LIB:  ${ORT_LIB_LOCATION:-<not set>}"

if [ -z "${ORT_LIB_LOCATION:-}" ]; then
    echo "ERROR: ORT_LIB_LOCATION is not set" >&2
    exit 1
fi

# ort-sys v2.0.0-rc.12 defaults to static linking (looks for .a files),
# but the ONNX Runtime build only ships shared libraries (.so).
# ORT_PREFER_DYNAMIC_LINK=1 makes ort-sys emit `rustc-link-lib=onnxruntime`
# instead of `rustc-link-lib=static=onnxruntime`.
export ORT_PREFER_DYNAMIC_LINK=1

# -rpath,$ORIGIN: runtime linker searches binary's directory for .so files
# -rpath-link,<ORT_LIB_LOCATION>: build-time linker resolves transitive NEEDED deps
export RUSTFLAGS="-Awarnings -C link-args=-Wl,-rpath,\$ORIGIN -Wl,-rpath-link,${ORT_LIB_LOCATION}"
export LIBRARY_PATH="${ORT_LIB_LOCATION}${LIBRARY_PATH:+:}${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ORT_LIB_LOCATION}${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"

echo "=== ORT lib directory ==="
ls -la "${ORT_LIB_LOCATION}/"

cd "${SRC_TAURI}"
cargo build --release --target "${CARGO_TARGET}" --no-default-features --features "${CARGO_FEATURES}"

# Copy ORT shared libraries alongside the binary for distribution
BIN_DIR="target/${CARGO_TARGET}/release"
if [ -n "${ORT_LIB_LOCATION:-}" ] && [ -d "${ORT_LIB_LOCATION}" ]; then
    cp -v "${ORT_LIB_LOCATION}"/*.so* "${BIN_DIR}/" 2>/dev/null || true
fi
echo "=== Release directory contents ==="
ls -la "${BIN_DIR}/"
