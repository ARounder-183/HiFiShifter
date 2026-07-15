#!/usr/bin/env bash
# download-ort.sh
# Download and extract ONNX Runtime GPU for Linux, including cuDNN.
#
# Usage:
#   ./scripts/download-ort.sh
#
# Environment:
#   ORT_VERSION  - ONNX Runtime version (default: 1.24.1)
#   RUNNER_TEMP  - CI temp directory (default: /tmp)
#
# Outputs:
#   Sets ORT_LIB_LOCATION env var (and writes to GITHUB_ENV in CI).

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.24.1}"
TEMP_DIR="${RUNNER_TEMP:-/tmp}"
OrtUrl="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
OrtDir="${TEMP_DIR}/ort-gpu"
OrtTgz="${TEMP_DIR}/ort-gpu.tgz"

echo "=== ONNX Runtime GPU Download (Linux) ==="
echo "  Version: ${ORT_VERSION}"
echo "  URL:     ${OrtUrl}"

echo "Downloading ONNX Runtime GPU v${ORT_VERSION} for Linux..."
curl -fSL -o "${OrtTgz}" "${OrtUrl}"

echo "Extracting..."
mkdir -p "${OrtDir}"
tar -xzf "${OrtTgz}" -C "${OrtDir}"

LibDir="$(find "${OrtDir}" -maxdepth 2 -type d -name lib | head -1)"
echo "ORT lib dir: ${LibDir}"

# Download cuDNN 9.8.0 redistributable
TmpDl="${TEMP_DIR}/cuda-redist"
mkdir -p "${TmpDl}"

CudnnUrl="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz"
CudnnArchive="${TmpDl}/cudnn.tar.xz"
echo "Downloading cuDNN 9.8.0..."
curl -fSL -o "${CudnnArchive}" "${CudnnUrl}"
tar -xJf "${CudnnArchive}" -C "${TmpDl}"
rm -f "${CudnnArchive}"

# Copy cuDNN .so files into ORT lib dir
while IFS= read -r -d '' found_dir; do
    echo "Copying cuDNN from ${found_dir}..."
    cp -av "${found_dir}/"* "${LibDir}/" 2>/dev/null || true
done < <(find "${TmpDl}" -type d -name lib -print0)
rm -rf "${TmpDl}"

# Export ORT_LIB_LOCATION
export ORT_LIB_LOCATION="${LibDir}"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "ORT_LIB_LOCATION=${LibDir}" >> "${GITHUB_ENV}"
    echo "  CI: ORT_LIB_LOCATION written to GITHUB_ENV"
fi

echo "=== ORT lib contents ==="
ls -la "${LibDir}/"

echo "Done."
