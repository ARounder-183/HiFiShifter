#!/usr/bin/env bash
# Build the Android package (Linux / macOS hosts).
#
#   ./scripts/build/android.sh              # APK (side-load)
#   ./scripts/build/android.sh --bundle     # AAB (Play)
#   ./scripts/build/android.sh --debug      # debug APK, no signing
#
# Same two constraints as the PowerShell twin (see scripts/build/android.ps1 for
# the long version):
#   1. arm64-v8a only -- ONNX Runtime ships no prebuilt for the other three ABIs,
#      and `tauri android build` compiles all four by default.
#   2. The NDK is discovered, never hardcoded.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

bundle=0
debug=0
for arg in "$@"; do
  case "$arg" in
    --bundle) bundle=1 ;;
    --debug) debug=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

resolve_ndk() {
  for name in ANDROID_NDK_HOME NDK_HOME; do
    value="${!name:-}"
    if [ -n "$value" ] && [ -d "$value" ]; then echo "$value"; return; fi
  done
  sdk="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
  if [ -z "$sdk" ]; then
    echo "ANDROID_SDK_ROOT / ANDROID_HOME is not set; install the Android SDK first." >&2
    exit 1
  fi
  if [ ! -d "$sdk/ndk" ]; then
    echo "No NDK under $sdk/ndk. Install one with: sdkmanager ndk;27.0.12077973" >&2
    exit 1
  fi
  # Version-aware sort: "9.x" must not beat "27.x" as a string.
  find "$sdk/ndk" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" 2>/dev/null |
    sort -V | tail -n 1 | sed "s|^|$sdk/ndk/|"
}

ndk="$(resolve_ndk)"
echo "== Android NDK: $ndk"
export ANDROID_NDK_HOME="$ndk"
export NDK_HOME="$ndk"

rust_target="aarch64-linux-android"
rustup target add "$rust_target" >/dev/null

args=(tauri android build --target "$rust_target" --features onnx)
if [ "$bundle" -eq 1 ]; then args+=(--aab); else args+=(--apk); fi
if [ "$debug" -eq 1 ]; then args+=(--debug); fi

echo "== npx ${args[*]}"
( cd backend && npx "${args[@]}" )

# Artefact integrity: ort-sys stages symlinks, and copying a link instead of its
# target yields a 0-byte .so that fails at System.loadLibrary, long after the
# build reported success.
jni="backend/src-tauri/gen/android/app/src/main/jniLibs"
if [ -d "$jni" ]; then
  bad=0
  while IFS= read -r -d "" so; do
    if [ "$(head -c 4 "$so" | od -An -tx1 | tr -d " \n")" != "7f454c46" ]; then
      echo "not an ELF shared object: $so" >&2
      bad=1
    fi
  done < <(find "$jni" -name "*.so" -print0)
  [ "$bad" -eq 0 ] && echo "== jniLibs .so files are real ELF objects"
  [ "$bad" -eq 0 ] || exit 1
else
  echo "== $jni not found; skipping artefact integrity check"
fi
