#!/usr/bin/env bash
# Build the Android package (Linux / macOS hosts).
#
#   ./scripts/build/android.sh              # APK (side-load)
#   ./scripts/build/android.sh --bundle     # AAB (Play)
#   ./scripts/build/android.sh --debug      # debug APK, no signing
#
# Same constraints as the PowerShell twin (see scripts/build/android.ps1 for the long
# version):
#   1. arm64-v8a only -- ONNX Runtime ships no prebuilt for the other three ABIs, and
#      `tauri android build` compiles all four by default.
#   2. `--target` is the ABI short name (aarch64), not the Rust triple.
#   3. The NDK is discovered, never hardcoded.
#
# [Why this script exports more than ANDROID_NDK_HOME] cc-rs does not read that variable
# to find the compiler: it probes for the standalone wrapper names
# (aarch64-linux-android21-clang) through PATH and falls back to a bare
# `clang --target=...` with no sysroot when the probe fails. Naming CC/CXX/AR per target
# is the documented escape hatch and is what actually selects the NDK toolchain. The same
# reasoning is spelled out in scripts/env/android-env.ps1, which is the Windows twin of
# the environment block below.
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

# ── Discovery ────────────────────────────────────────────────────────────────
# [Why not `find -printf` / `sort -V`] Neither is POSIX: BSD find has no -printf and
# macOS sort has no -V, so a script that used them would work on the CI runner and fail
# on a developer's Mac. Plain field-wise `sort -n` is portable and enough for a
# three-part version.

resolve_sdk() {
  for name in HIFISHIFTER_ANDROID_SDK ANDROID_SDK_ROOT ANDROID_HOME; do
    value="${!name:-}"
    if [ -n "$value" ] && [ -d "$value/platform-tools" ]; then echo "$value"; return; fi
  done
  for candidate in "$HOME/Library/Android/sdk" "$HOME/Android/Sdk" /usr/local/lib/android/sdk /opt/android-sdk; do
    if [ -d "$candidate/platform-tools" ]; then echo "$candidate"; return; fi
  done
  echo ""
}

resolve_ndk() {
  for name in HIFISHIFTER_ANDROID_NDK ANDROID_NDK_HOME NDK_HOME ANDROID_NDK_ROOT; do
    value="${!name:-}"
    if [ -n "$value" ] && [ -f "$value/source.properties" ]; then echo "$value"; return; fi
  done
  if [ -z "$sdk" ] || [ ! -d "$sdk/ndk" ]; then echo ""; return; fi
  newest="$(ls -1 "$sdk/ndk" 2>/dev/null | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)"
  if [ -n "$newest" ] && [ -f "$sdk/ndk/$newest/source.properties" ]; then
    echo "$sdk/ndk/$newest"
    return
  fi
  echo ""
}

resolve_jdk() {
  for name in HIFISHIFTER_JDK JAVA_HOME; do
    value="${!name:-}"
    if [ -n "$value" ] && [ -x "$value/bin/java" ]; then echo "$value"; return; fi
  done
  # Portable JDK extracts sit next to the SDK; system installs sit under a vendor dir.
  if [ -n "$sdk" ]; then
    for candidate in "$sdk/../jdk17" "$sdk/../jdk"; do
      if [ -x "$candidate/bin/java" ]; then echo "$candidate"; return; fi
    done
  fi
  for candidate in /usr/lib/jvm/*17* /usr/lib/jvm/*21* "$HOME/.jdks"/*17*; do
    if [ -x "$candidate/bin/java" ]; then echo "$candidate"; return; fi
  done
  echo ""
}

resolve_sdk_cmake_bin() {
  if [ -z "$sdk" ] || [ ! -d "$sdk/cmake" ]; then echo ""; return; fi
  newest="$(ls -1 "$sdk/cmake" 2>/dev/null | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)"
  if [ -n "$newest" ] && [ -x "$sdk/cmake/$newest/bin/ninja" ]; then
    echo "$sdk/cmake/$newest/bin"
    return
  fi
  echo ""
}

sdk="$(resolve_sdk)"
if [ -z "$sdk" ]; then
  echo "Android SDK not found. Install it, or set ANDROID_SDK_ROOT (a directory with platform-tools)." >&2
  exit 1
fi

ndk="$(resolve_ndk)"
if [ -z "$ndk" ]; then
  echo "No Android NDK under $sdk/ndk. Install one with: sdkmanager 'ndk;27.0.12077973'" >&2
  exit 1
fi

jdk="$(resolve_jdk)"
if [ -z "$jdk" ]; then
  echo "No JDK found (gradle needs 17+). Install one, or point HIFISHIFTER_JDK at it." >&2
  exit 1
fi

# The NDK ships one prebuilt tree per host; read it from disk instead of guessing.
ndk_prebuilt="$(ls -1d "$ndk"/toolchains/llvm/prebuilt/*/ 2>/dev/null | head -n 1)"
if [ -z "$ndk_prebuilt" ]; then
  echo "NDK at $ndk has no toolchains/llvm/prebuilt tree; the install is incomplete." >&2
  exit 1
fi
ndk_bin="${ndk_prebuilt%/}/bin"
cmake_bin="$(resolve_sdk_cmake_bin)"

echo "== Android SDK: $sdk"
echo "== Android NDK: $ndk"
echo "== JDK:         $jdk"
if [ -n "$cmake_bin" ]; then echo "== CMake/Ninja: $cmake_bin"; fi

export ANDROID_HOME="$sdk"
export ANDROID_SDK_ROOT="$sdk"
export ANDROID_NDK_HOME="$ndk"
export NDK_HOME="$ndk"
export JAVA_HOME="$jdk"

# The NDK bin goes first: a host clang (LLVM ships one with Visual Studio, /usr/bin on
# some distros) would otherwise win and fail on missing Android headers.
path_prefix="$jdk/bin:$ndk_bin:$sdk/platform-tools:$sdk/cmdline-tools/latest/bin"
if [ -n "$cmake_bin" ]; then path_prefix="$cmake_bin:$path_prefix"; fi
export PATH="$path_prefix:$PATH"

# cmake-rs leaves the generator to CMake's default; the NDK toolchain file needs Ninja.
if [ -n "$cmake_bin" ]; then export CMAKE_GENERATOR="Ninja"; fi

# fdk-aac includes AOSP's internal log/log.h under #ifdef __ANDROID__, which the NDK does
# not ship. The shim supplies the single symbol it needs; see
# backend/build-support/android-log-shim/log/log.h.
export CPATH="$root/backend/build-support/android-log-shim${CPATH:+:$CPATH}"

# ── Per-target compiler/linker overrides ─────────────────────────────────────
# [Why the wrapper's API level must be >= minSdk] The NDK only ships libaaudio.so under
# <sysroot>/usr/lib/<abi>/<api>/ for api >= 26, and cpal's Android backend emits
# -laaudio. An API-24 wrapper therefore fails with
#   ld.lld: error: unable to find library -laaudio
# Compiling at exactly minSdk is also what makes the compiler reject calls to APIs that
# do not exist on the oldest supported device, so we take the LOWEST wrapper at or above
# it rather than the newest one available.
min_sdk="$(sed -n 's/.*minSdk *= *\([0-9][0-9]*\).*/\1/p' \
  backend/src-tauri/gen/android/app/build.gradle.kts 2>/dev/null | head -n 1)"
if [ -z "$min_sdk" ]; then
  # Same value as backend/src-tauri/tauri.conf.json's bundle.android.minSdkVersion.
  min_sdk=26
fi

export_android_triple() {
  local triple="$1" prefix="$2"
  local best_api="" best_cc="" best_cxx=""
  local candidate api
  for candidate in "$ndk_bin/$prefix"*-clang; do
    [ -x "$candidate" ] || continue
    api="${candidate#"$ndk_bin/$prefix"}"
    api="${api%%-clang}"
    case "$api" in '' | *[!0-9]*) continue ;; esac
    [ "$api" -ge "$min_sdk" ] || continue
    if [ -z "$best_api" ] || [ "$api" -lt "$best_api" ]; then
      best_api="$api"
      best_cc="$candidate"
      best_cxx="$candidate++"
    fi
  done
  [ -n "$best_cc" ] || return 0

  local underscored="${triple//-/_}"
  local upper
  upper="$(echo "$underscored" | tr '[:lower:]' '[:upper:]')"
  export "CC_$underscored=$best_cc"
  export "CARGO_TARGET_${upper}_LINKER=$best_cc"
  if [ -x "$best_cxx" ]; then export "CXX_$underscored=$best_cxx"; fi
  export "AR_${underscored}=$ndk_bin/llvm-ar"
}

export_android_triple aarch64-linux-android aarch64-linux-android
export_android_triple armv7-linux-androideabi armv7a-linux-androideabi
export_android_triple i686-linux-android i686-linux-android
export_android_triple x86_64-linux-android x86_64-linux-android

# ── Build ────────────────────────────────────────────────────────────────────
# `cargo tauri` is what this repository uses for desktop builds; a clean CI runner only
# has npm. `npx tauri` is not an option -- no such executable exists on npm.
if command -v cargo-tauri >/dev/null 2>&1; then
  tauri_cli=(cargo tauri)
else
  tauri_cli=(npx --yes --package @tauri-apps/cli@2 tauri)
fi

rust_target="aarch64-linux-android"
rustup target add "$rust_target" >/dev/null

if [ ! -d "backend/src-tauri/gen/android" ]; then
  echo "== gen/android is missing; initialising the Android project"
  ( cd backend && "${tauri_cli[@]}" android init --ci )
fi

# ── C++ runtime ──────────────────────────────────────────────────────────────
# WORLD, Signalsmith, fdk-aac and libopus are C++ and cc-rs links the **shared** C++ runtime
# on Android, so libbackend_lib.so carries "NEEDED libc++_shared.so". Android ships no such
# system library; an app that does not bundle it dies at System.loadLibrary with
# 'dlopen failed: library "libc++_shared.so" not found'. tauri-cli stages only
# libbackend_lib.so, so the copy happens here.
# [Why shared and not c++_static] Static linking also needs -lc++abi, injectable only through
# per-target rustflags, which are baked into every dependency crate's fingerprint.
jni_root="backend/src-tauri/gen/android/app/src/main/jniLibs"
prebuilt="${ndk_prebuilt%/}"
for pair in "arm64-v8a:aarch64-linux-android" "armeabi-v7a:armv7-linux-androideabi" "x86:i686-linux-android" "x86_64:x86_64-linux-android"; do
  jni_abi="${pair%%:*}"
  triple="${pair##*:}"
  src="$prebuilt/sysroot/usr/lib/$triple/libc++_shared.so"
  [ -f "$src" ] || continue
  mkdir -p "$jni_root/$jni_abi"
  cp -f "$src" "$jni_root/$jni_abi/libc++_shared.so"
  echo "== bundled libc++_shared.so for $jni_abi"
done

args=(android build --target aarch64 --features onnx)
if [ "$bundle" -eq 1 ]; then args+=(--aab); else args+=(--apk); fi
if [ "$debug" -eq 1 ]; then args+=(--debug); fi

echo "== ${tauri_cli[*]} ${args[*]}"
( cd backend && "${tauri_cli[@]}" "${args[@]}" )

# Artefact integrity: ort-sys stages symlinks, and copying a link instead of its target
# yields a 0-byte .so that fails at System.loadLibrary, long after the build reported
# success.
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
