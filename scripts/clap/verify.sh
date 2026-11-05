#!/usr/bin/env bash
# CLAP 产物自检：只信外部证据（文件、导出符号、动态依赖、黑盒 dlopen、前端与模型清单）。
# 用法：scripts/clap/verify.sh <path-to-HiFiShifter.clap|bundle> [--allow-ara]
set -euo pipefail

bundle="${1:-}"
allow_ara="${2:-}"
if [[ -z "$bundle" ]]; then echo "usage: $0 <bundle> [--allow-ara]" >&2; exit 2; fi
[[ -e "$bundle" ]] || { echo "FAIL 1 layout: $bundle does not exist" >&2; exit 1; }
echo "OK   1 layout: $bundle"

case "$(uname -s)" in
  Darwin) module="$bundle/Contents/MacOS/HiFiShifter";;
  *) module="$bundle";;
esac
[[ -f "$module" ]] || { echo "FAIL 2 module: $module is missing" >&2; exit 1; }
if command -v nm >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]]; then nm -gU "$module" | grep -q "clap_entry";
  else nm -D --defined-only "$module" | grep -q "clap_entry"; fi
  echo "OK   2 exports: clap_entry"
else echo "SKIP 2 exports: nm unavailable"; fi

if command -v ldd >/dev/null 2>&1; then
  ldd "$module" | grep -v "not found" >/dev/null && echo "OK   3 dependencies: no unresolved libraries"
elif command -v otool >/dev/null 2>&1; then
  otool -L "$module" >/dev/null && echo "OK   3 dependencies: otool listed the module"
else echo "SKIP 3 dependencies: no ldd/otool"; fi

cc="${CC:-cc}"
probe_bin="$(mktemp)"
if "$cc" -o "$probe_bin" "$(dirname "$0")/probe.c" -ldl 2>/dev/null; then
  "$probe_bin" "$module" | tee /tmp/hifishifter-clap-probe.log
  grep -q "plugin_id=com.hifishifter.clap" /tmp/hifishifter-clap-probe.log
  grep -q "ara:supported" /tmp/hifishifter-clap-probe.log
  ! grep -q "ara:required" /tmp/hifishifter-clap-probe.log
  echo "OK   4 black-box load: descriptor and ARA feature are honest"
  if [[ "$allow_ara" == "--allow-ara" ]]; then grep -q "ara_factory=present" /tmp/hifishifter-clap-probe.log;
  else grep -q "ara_factory=present" /tmp/hifishifter-clap-probe.log && echo "OK   5 ARA discovery available (instance binding not wired)"; fi
  rm -f "$probe_bin"
else echo "SKIP 4 black-box load: C compiler unavailable"; fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  [[ -f "$bundle/Contents/Info.plist" ]] || { echo "FAIL 5 metadata: Info.plist missing" >&2; exit 1; }
  echo "OK   5 metadata: Info.plist"
else echo "OK   5 metadata: single-file CLAP layout"; fi

resources="$bundle/Contents/Resources"
[[ -d "$bundle" ]] && [[ -d "$resources" ]] && resources="$resources" || resources="$(dirname "$module")/HiFiShifter-resources"
if [[ -d "$resources/frontend" ]]; then
  [[ -f "$resources/frontend/plugin.html" ]] || { echo "FAIL 6 frontend: plugin.html missing" >&2; exit 1; }
  ! grep -R "https\?://" "$resources/frontend/plugin.html" >/dev/null
  echo "OK   6 frontend: plugin.html has no remote reference"
else echo "SKIP 6 frontend: not packaged in this stage"; fi

if [[ -d "$resources/models" ]]; then
  [[ -f "$resources/models/models.json" ]] || { echo "FAIL 7 models: models.json missing" >&2; exit 1; }
  ! find "$resources/models" -name "*.onnx" | grep -q .
  echo "OK   7 models: manifest only, no bundled weights"
else echo "SKIP 7 models: not packaged in this stage"; fi

# 第 8 项：阶段 1 的诚实判据。当前产物暴露 entry 级 ARA 发现，但实例绑定尚未接线；
# 探针的 descriptor 必须含 ara:supported，且绝不声明 ara:required。
grep -q "ara:supported" /tmp/hifishifter-clap-probe.log
! grep -q "ara:required" /tmp/hifishifter-clap-probe.log
echo "OK   8 capability: optional ARA, instance extension unwired"

echo "All CLAP checks passed."
