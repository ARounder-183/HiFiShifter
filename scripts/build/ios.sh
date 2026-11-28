#!/usr/bin/env bash
# Build the iOS package (macOS host only).
#
#   ./scripts/build/ios.sh                 # build for a connected device
#   ./scripts/build/ios.sh --simulator     # simulator build (no signing)
#
# [Why minimumSystemVersion is 15.1 and not Tauri default 15.0] ort-sys prebuilt
# CoreML binaries declare iOS >= 15.1; linking them into a 15.0 deployment target
# produces a dyld failure at launch on 15.0 devices. The value lives in
# backend/src-tauri/tauri.ios.conf.json so a single source decides it.
#
# [Why --simulator exists] The App Store requires an arm64 device build, but the
# only way to iterate on CI without signing certificates is a simulator build.
# Note that the simulator slice is a *separate* ONNX Runtime distribution
# (aarch64-apple-ios-sim+coreml) -- x86_64 simulator slices have none.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

simulator=0
for arg in "$@"; do
  case "$arg" in
    --simulator) simulator=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" != "Darwin" ]; then
  echo "iOS builds require macOS (Xcode toolchain)." >&2
  exit 1
fi

# `cargo tauri` is what this repository uses for desktop builds; a clean CI runner only
# has npm. `npx tauri` is not an option -- no such executable exists on npm.
if command -v cargo-tauri >/dev/null 2>&1; then
  tauri_cli=(cargo tauri)
else
  tauri_cli=(npx --yes --package @tauri-apps/cli@2 tauri)
fi

# gen/apple is produced by `tauri ios init` and cannot be generated on a non-macOS
# host, so unlike gen/android it is not committed. `ios build` stops with "iOS project
# not found" when it is missing, which does not say how to create one.
if [ ! -d "backend/src-tauri/gen/apple" ]; then
  echo "== gen/apple is missing; initialising the iOS project"
  ( cd backend && "${tauri_cli[@]}" ios init --ci )
fi

if [ "$simulator" -eq 1 ]; then
  rust_target="aarch64-apple-ios-sim"
  echo "== simulator build ($rust_target); signing skipped"
  rustup target add "$rust_target" >/dev/null
  ( cd backend && "${tauri_cli[@]}" ios build --target "$rust_target" --features onnx --debug )
  exit $?
fi

if [ -z "${APPLE_DEVELOPMENT_TEAM:-}" ]; then
  echo "APPLE_DEVELOPMENT_TEAM is not set; device builds need a signing identity." >&2
  echo "Set it in the environment or in backend/src-tauri/tauri.ios.conf.json." >&2
  exit 1
fi

rust_target="aarch64-apple-ios"
rustup target add "$rust_target" >/dev/null
echo "== device build ($rust_target), team $APPLE_DEVELOPMENT_TEAM"
( cd backend && "${tauri_cli[@]}" ios build --target "$rust_target" --features onnx )
