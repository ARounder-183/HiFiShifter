#!/usr/bin/env bash
# CLAP 的 Unix 构建：编译 cdylib -> 单文件 .clap / macOS bundle -> 自检 -> dist/ tar.gz。
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
stage="${1:-all}"
version="$(python -c "import json; print(json.load(open('backend/src-tauri/tauri.conf.json'))['version'])" 2>/dev/null || echo dev)"
out="$root/dist/clap/HiFiShifter_${version}_$(uname -s | tr A-Z a-z)-$(uname -m)-clap"
artifact="$root/backend/target/release/libhifishifter_clap.so"
[[ "$(uname -s)" == "Darwin" ]] && artifact="$root/backend/target/release/libhifishifter_clap.dylib"

if [[ "$stage" == "all" || "$stage" == "build" ]]; then
  cargo build --manifest-path backend/Cargo.toml -p hifishifter-clap --release --locked
fi
if [[ "$stage" == "all" || "$stage" == "bundle" ]]; then
  rm -rf "$out"; mkdir -p "$(dirname "$out")"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    mkdir -p "$out/Contents/MacOS" "$out/Contents/Resources"
    cp "$artifact" "$out/Contents/MacOS/HiFiShifter"
    cat > "$out/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>com.hifishifter.clap</string>
  <key>CFBundleName</key><string>HiFiShifter</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleExecutable</key><string>HiFiShifter</string>
  <key>CFBundleShortVersionString</key><string>${version}</string>
  <key>CFBundleVersion</key><string>${version}</string>
</dict>
</plist>
PLIST
  else
    mkdir -p "$out"; cp "$artifact" "$out/HiFiShifter.clap"
  fi
fi
if [[ "$stage" == "all" || "$stage" == "verify" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then scripts/clap/verify.sh "$out"; else scripts/clap/verify.sh "$out/HiFiShifter.clap"; fi
fi
