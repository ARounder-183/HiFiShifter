#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build-macos-dmg.sh [options]

Options:
  -s, --skip-build   Reuse an existing app/dmg build and only copy it to dist/
  -o, --output DIR   Output directory (default: <repo>/dist)
  -h, --help         Show this help

Environment:
  HIFISHIFTER_SKIP_FRONTEND_BUILD=1 skips the frontend build in tauri.conf.json.
EOF
}

SKIP_BUILD=0
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[mac-dmg] This script must run on macOS" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
  export PATH
fi

if ! command -v cargo >/dev/null 2>&1 || ! cargo tauri --version >/dev/null 2>&1; then
  echo "[mac-dmg] cargo or Tauri CLI is unavailable." >&2
  echo "[mac-dmg] Install deps with: SKIP_FRONTEND=1 bash scripts/install_deps_macos.sh" >&2
  echo "[mac-dmg] Then install: cargo install tauri-cli --version '^2'" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_DIR="$ROOT/backend/src-tauri"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/dist}"
mkdir -p "$OUTPUT_DIR"

PRODUCT_NAME="$(/usr/bin/python3 - "$TAURI_DIR/tauri.conf.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["productName"])
PY
)"
VERSION="$(/usr/bin/python3 - "$TAURI_DIR/tauri.conf.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f)["version"])
PY
)"

ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  ARCH_SUFFIX="arm64"
else
  ARCH_SUFFIX="x64"
fi

DMG_NAME="${PRODUCT_NAME}_${VERSION}_${ARCH_SUFFIX}.dmg"
RELEASE_DIR="$TAURI_DIR/target/release"
DMG_PATH="$RELEASE_DIR/bundle/dmg/$DMG_NAME"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[mac-dmg] Building $PRODUCT_NAME $VERSION ($ARCH_SUFFIX) DMG"
  (
    cd "$TAURI_DIR"
    cargo tauri build --bundles app,dmg
  )
fi

if [[ ! -f "$DMG_PATH" ]]; then
  echo "[mac-dmg] DMG not found: $DMG_PATH" >&2
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    echo "[mac-dmg] Run without --skip-build, or build with:" >&2
    echo "          cd backend/src-tauri && cargo tauri build --bundles app,dmg" >&2
  fi
  exit 1
fi

DEST_DMG="$OUTPUT_DIR/$DMG_NAME"
cp -f "$DMG_PATH" "$DEST_DMG"
SIZE_MB="$(du -m "$DEST_DMG" | cut -f1)"

echo "============================================"
echo "  macOS DMG packaging successful"
echo "============================================"
echo "  DMG: $DEST_DMG"
echo "  Size: ${SIZE_MB} MB"
echo "  Version: $VERSION"
echo "  Arch: $ARCH_SUFFIX"
echo "============================================"
