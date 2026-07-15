#!/usr/bin/env bash
# install-cuda-linux.sh
# Install NVIDIA CUDA toolkit and runtime libraries on Ubuntu 24.04.
#
# Usage:
#   ./scripts/install-cuda-linux.sh
#
# Outputs:
#   Adds /usr/local/cuda/bin to PATH (and GITHUB_PATH in CI).
#   Exports CUDA_LIB_PATH=/usr/local/cuda/lib64 (and writes to GITHUB_ENV in CI).

set -euxo pipefail

echo "=== Installing CUDA toolkit + runtime libraries (Linux GPU) ==="

wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y

# nvcc + CUDA runtime libraries.
# libonnxruntime_providers_cuda.so NEEDs libcudart.so.12, libcublas.so.12,
# libcublasLt.so.12, libcufft.so.11, libcurand.so.10 at LINK TIME.
sudo apt-get install -y \
    cuda-compiler-12-6 \
    libcublas-12-6 \
    libcufft-12-6 \
    libcurand-12-6

# Add CUDA to PATH for nvcc discovery
export PATH="/usr/local/cuda/bin:${PATH}"
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "/usr/local/cuda/bin" >> "${GITHUB_PATH}"
    echo "  CI: /usr/local/cuda/bin written to GITHUB_PATH"
fi
nvcc --version

# Export CUDA library path
CUDA_LIB_PATH="/usr/local/cuda/lib64"
export CUDA_LIB_PATH="${CUDA_LIB_PATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "CUDA_LIB_PATH=${CUDA_LIB_PATH}" >> "${GITHUB_ENV}"
    echo "  CI: CUDA_LIB_PATH written to GITHUB_ENV"
fi

echo "=== Installed CUDA shared libraries ==="
find /usr/local/cuda/lib64 -name "libcud*.so*" -o -name "libcublas*.so*" -o -name "libcufft*.so*" -o -name "libcurand*.so*" 2>/dev/null | sort

# Ensure PKG_CONFIG_PATH for alsa
PKG_DIRS=""
for d in /usr/lib/*-linux-gnu/pkgconfig /usr/lib/pkgconfig /usr/share/pkgconfig; do
    if [ -d "$d" ]; then
        if [ -z "$PKG_DIRS" ]; then
            PKG_DIRS="$d"
        else
            PKG_DIRS="$PKG_DIRS:$d"
        fi
    fi
done
export PKG_CONFIG_PATH="$PKG_DIRS"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "PKG_CONFIG_PATH=$PKG_DIRS" >> "${GITHUB_ENV}"
fi
pkg-config --exists alsa
pkg-config --modversion alsa

echo "Done."
