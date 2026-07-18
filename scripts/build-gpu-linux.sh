#!/usr/bin/env bash
# build-gpu-linux.sh
# Build the HiFiShifter Linux binary with OpenCL GPU support.
#
# GPU acceleration is baked into the Linux build target — no extra
# Cargo features are needed.  OpenCL is accessed at runtime via the
# ONNX Runtime generic execution provider API.
#
# Usage:
#   ./scripts/build-gpu-linux.sh
#
# Prerequisites:
#   ./scripts/download-ort.sh          # download ONNX Runtime GPU package
#   sudo apt-get install -y ocl-icd-opencl-dev ocl-icd-libopencl1
#
# Environment (required):
#   ORT_LIB_LOCATION  - Path to ONNX Runtime library directory
#                       (set automatically by download-ort.sh)
#
# Environment (optional):
#   CARGO_TARGET      - Rust target triple (default: x86_64-unknown-linux-gnu)

set -euxo pipefail

CARGO_TARGET="${CARGO_TARGET:-x86_64-unknown-linux-gnu}"
SRC_TAURI="backend/src-tauri"

echo "=== Build Linux (OpenCL GPU) Binary ==="
echo "  Target:   ${CARGO_TARGET}"
echo "  ORT_LIB:  ${ORT_LIB_LOCATION:-<not set>}"

# Auto-detect ORT lib directory if not set (download-ort.sh extracts to /tmp/ort-gpu/).
if [ -z "${ORT_LIB_LOCATION:-}" ]; then
    ORT_LIB_LOCATION="$(find /tmp/ort-gpu -maxdepth 3 -type d -name lib 2>/dev/null | head -1 || true)"
    if [ -z "${ORT_LIB_LOCATION:-}" ]; then
        echo "ERROR: ORT_LIB_LOCATION is not set and could not auto-detect" >&2
        echo "Run ./scripts/download-ort.sh first, or set ORT_LIB_LOCATION manually." >&2
        exit 1
    fi
    echo "  Auto-detected ORT lib: ${ORT_LIB_LOCATION}"
fi

# ort-sys v2.0.0-rc.12 defaults to static linking (.a), but the ONNX Runtime
# package only ships shared libraries (.so).  Force dynamic linking.
export ORT_PREFER_DYNAMIC_LINK=1

# -rpath,$ORIGIN: runtime linker searches binary's directory for .so files
# -rpath-link,<ORT_LIB_LOCATION>: build-time linker resolves transitive NEEDED deps
export RUSTFLAGS="-Awarnings -C link-args=-Wl,-rpath,\$ORIGIN -Wl,-rpath-link,${ORT_LIB_LOCATION}"
export LIBRARY_PATH="${ORT_LIB_LOCATION}${LIBRARY_PATH:+:}${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ORT_LIB_LOCATION}${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"

echo "=== ORT lib directory ==="
ls -la "${ORT_LIB_LOCATION}/"

cd "${SRC_TAURI}"
cargo build --release --target "${CARGO_TARGET}" --no-default-features --features onnx

# Copy ORT shared libraries alongside the binary for distribution
BIN_DIR="target/${CARGO_TARGET}/release"
if [ -n "${ORT_LIB_LOCATION:-}" ] && [ -d "${ORT_LIB_LOCATION}" ]; then
    cp -v "${ORT_LIB_LOCATION}"/*.so* "${BIN_DIR}/" 2>/dev/null || true
fi
echo "=== Release directory contents ==="
ls -la "${BIN_DIR}/"
