#!/usr/bin/env bash
set -euo pipefail

# Provision an LGPL shared FFmpeg 8.1 build (headers + import libs + shared
# objects) from the official BtbN FFmpeg-Builds release. FFmpeg remains
# dynamically linked to keep the MIT-licensed application compliant.

ARCH="${1:-x64}"
case "$ARCH" in
  x64|x86_64|amd64) TOKEN="linux64" ;;
  arm64|aarch64) TOKEN="linuxarm64" ;;
  *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_DIR="$ROOT/backend/src-tauri"
PROVISION_DIR="$TAURI_DIR/third_party/ffmpeg"
ASSET="ffmpeg-n8.1-latest-${TOKEN}-lgpl-shared-8.1.tar.xz"
URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$ASSET"
ARCHIVE="$PROVISION_DIR/$ASSET"

mkdir -p "$PROVISION_DIR"

EXISTING="$(find "$PROVISION_DIR" -mindepth 1 -maxdepth 5 -type f -path '*/include/libavcodec/version_major.h' -print -quit | while read -r h; do grep -q 'LIBAVCODEC_VERSION_MAJOR 62' "$h" && echo "$h"; done | head -n1)"
if [ -z "$EXISTING" ]; then
  if [ ! -f "$ARCHIVE" ]; then
    echo "[ffmpeg] Downloading LGPL shared FFmpeg 8.1: $URL"
    curl -fL --retry 3 -o "$ARCHIVE" "$URL"
  fi
  echo "[ffmpeg] Extracting $ASSET"
  tar -xf "$ARCHIVE" -C "$PROVISION_DIR"
fi

FFMPEG_DIR="$(find "$PROVISION_DIR" -mindepth 1 -maxdepth 5 -type f -path '*/include/libavcodec/version_major.h' -print -quit | while read -r h; do grep -q 'LIBAVCODEC_VERSION_MAJOR 62' "$h" && sed 's#/include/libavcodec/version_major.h$##' <<<"$h"; done | head -n1)"
if [ -z "$FFMPEG_DIR" ] || [ ! -d "$FFMPEG_DIR/lib" ]; then
  echo "[ffmpeg] provisioning failed under $PROVISION_DIR" >&2
  exit 1
fi

# Cargo's [env] table forces FFMPEG_DIR to this fixed "current" symlink, so
# a changed FFmpeg major always invalidates/re-links ffmpeg-sys-next.
rm -rf "$PROVISION_DIR/current"
ln -s "$FFMPEG_DIR" "$PROVISION_DIR/current"
echo "[ffmpeg] Using FFMPEG_DIR=$FFMPEG_DIR (current -> $PROVISION_DIR/current)"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "LD_LIBRARY_PATH=$FFMPEG_DIR/lib:$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> "$GITHUB_ENV"
fi

echo "[ffmpeg] Linux provisioning complete"
