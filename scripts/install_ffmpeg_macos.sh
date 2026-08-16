#!/usr/bin/env bash
set -euo pipefail

# macOS has no official LGPL FFmpeg shared dev package, so build a minimal
# LGPL FFmpeg 8.1 from source. External codec libraries are disabled, which
# keeps the result compatible with HiFiShifter's MIT license. FFmpeg is linked
# dynamically only.

FFMPEG_VERSION="${FFMPEG_VERSION:-8.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_DIR="$ROOT/backend/src-tauri"
PREFIX="$TAURI_DIR/third_party/ffmpeg/macos-$(uname -m)"
SOURCE_DIR="$PREFIX/src"

if [ ! -f "$PREFIX/include/libavcodec/version_major.h" ] || ! grep -q 'LIBAVCODEC_VERSION_MAJOR 62' "$PREFIX/include/libavcodec/version_major.h"; then
  mkdir -p "$SOURCE_DIR"
  ARCHIVE="$SOURCE_DIR/ffmpeg-$FFMPEG_VERSION.tar.xz"
  if [ ! -f "$ARCHIVE" ]; then
    echo "[ffmpeg] Downloading FFmpeg $FFMPEG_VERSION"
    curl -fL --retry 3 -o "$ARCHIVE" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
  fi
  tar -xf "$ARCHIVE" -C "$SOURCE_DIR" --strip-components=1
  cd "$SOURCE_DIR"

  echo "[ffmpeg] Configuring minimal LGPL shared build"
  ./configure \
    --prefix="$PREFIX" \
    --enable-shared \
    --disable-static \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-autodetect \
    --enable-pic \
    --disable-asm \
    --pkg-config=true

  echo "[ffmpeg] Building FFmpeg"
  make -j"$(sysctl -n hw.ncpu || echo 4)"
  make install
else
  echo "[ffmpeg] Reusing existing build at $PREFIX"
fi

# Normalize install names to @rpath so the linked app can load the libraries
# from the Tauri Resources/ffmpeg directory on any machine.
RESOURCE_DIR="$TAURI_DIR/resources/ffmpeg"
mkdir -p "$RESOURCE_DIR"
for lib in libavcodec libavformat libavutil libswresample; do
  versioned="$(find "$PREFIX/lib" -maxdepth 1 -name "$lib.*.dylib" -type f -print -quit)"
  if [ -z "$versioned" ]; then
    echo "[ffmpeg] WARNING: missing $lib dylib under $PREFIX/lib" >&2
    exit 1
  fi
  name="$(basename "$versioned")"
  major="${name#${lib}.}"
  major="${major%%.*}"
  normalized="${lib}.${major}.dylib"
  install_name_tool -id "@rpath/$normalized" "$versioned"
  for dep in libavcodec libavformat libavutil libswresample; do
    depfile="$(find "$PREFIX/lib" -maxdepth 1 -name "$dep.*.dylib" -type f -print -quit)"
    [ -n "$depfile" ] || continue
    dep_name="$(basename "$depfile")"
    dep_major="${dep_name#${dep}.}"
    dep_major="${dep_major%%.*}"
    dep_normalized="${dep}.${dep_major}.dylib"
    install_name_tool -change "$PREFIX/lib/$dep_name" "@rpath/$dep_normalized" "$versioned" || true
  done
  cp -f "$versioned" "$RESOURCE_DIR/$normalized"
  echo "[ffmpeg] staged $normalized"
done
if [ -f "$SOURCE_DIR/COPYING.LGPLv3" ]; then
  cp -f "$SOURCE_DIR/COPYING.LGPLv3" "$RESOURCE_DIR/LICENSE.txt"
elif [ -f "$SOURCE_DIR/COPYING.LGPLv2.1" ]; then
  cp -f "$SOURCE_DIR/COPYING.LGPLv2.1" "$RESOURCE_DIR/LICENSE.txt"
fi

rm -rf "$TAURI_DIR/third_party/ffmpeg/current"
ln -s "$PREFIX" "$TAURI_DIR/third_party/ffmpeg/current"
echo "[ffmpeg] Using FFMPEG_DIR=$PREFIX (current -> $TAURI_DIR/third_party/ffmpeg/current)"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "DYLD_LIBRARY_PATH=$PREFIX/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" >> "$GITHUB_ENV"
fi

echo "[ffmpeg] macOS provisioning complete"
