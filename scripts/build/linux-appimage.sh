#!/usr/bin/env bash
set -euo pipefail

# HiFiShifter AppImage packaging: the **single** place that decides "what the bundle carries and
# what the runtime expects the host to provide".
#
# ── Why this file exists ──────────────────────────────────────────────
# AppImage's default behavior is "copy in as many of the executable's dynamic dependencies as
# possible". For a Tauri app, that rule also sweeps in things in the host desktop stack that
# **must come from the host**: WebKitGTK, GStreamer, and the ICU / libsoup / enchant that only
# WebKit uses. The same subsystem then gets split in two — GTK from inside the bundle (the
# Ubuntu 24.04 copy), libEGL / libwayland-client / Mesa from the host. The moment the versions
# on the two sides of that seam disagree, WebKit aborts right at startup:
#
#     Could not create default EGL display: EGL_BAD_PARAMETER. Aborting...
#
# This is not hypothetical: the v0.1.0-beta.13 feedback (Arch/CachyOS + AMD RX560, needing
# LD_PRELOAD of libwayland-client.so to start at all) was a symptom of exactly this seam. The
# "fix" at the time (the two commits of 2026-07-30) collected library files with find -type f,
# but on Ubuntu those SONAMEs are all symlinks, so **not one file** made it into the bundle and
# the problem shipped unchanged.
#
# ── The current strategy: hand the whole desktop stack to the host ────
# The bundle keeps only "the product's own files" plus GTK3 (the existing behavior of Tauri's
# linuxdeploy-plugin-gtk; how AppImages across the whole ecosystem ship, ABI stable):
#
#   * bundled: the HiFiShifter executable, libSoundTouchDLL.so, libwebgpu_dawn.so, models
#   * host-provided: WebKitGTK, GStreamer, ICU, libsoup, libEGL/GLES, libgbm/libdrm,
#     libwayland-*, libxkbcommon
#
# EGL / Wayland / Mesa then exist only as the host's copy, WebKit uses that same copy, and the
# seam is gone.
#
# Cost: the host must have webkit2gtk-4.1 (provided by Arch/CachyOS, Ubuntu 24.04+, Fedora 39+,
# Debian 13+ and openSUSE Tumbleweed; Ubuntu 22.04 only has 4.0 and is no longer supported).
#
# ── The rejected alternative ──────────────────────────────────────────
# "Bundle WebKit together with GL / DRM / GStreamer plugins / DRI drivers" would also remove the
# seam, but it means taking on all of: double the bundle size, libEGL that cannot be distributed
# with the bundle at all for NVIDIA users (proprietary drivers), Mesa tightly coupled to the
# kernel version, and the glvnd vendor JSON redirection. That just keeps manufacturing the same
# class of failure in another form, so it is not used.
#
# ── Usage ─────────────────────────────────────────────────────────────
#   linux-appimage.sh package <release-dir>      # prune → verify → repack → smoke
#   linux-appimage.sh prune   <AppDir>           # prune only
#   linux-appimage.sh verify  <AppDir>           # verify only (modifies nothing)
#   linux-appimage.sh repack  <AppDir> [out]     # repack only
#   linux-appimage.sh smoke   <AppImage>         # smoke only
#
# Environment variables:
#   HIFISHIFTER_SKIP_SMOKE=1        skip smoke (for CI build-only runs)
#   HIFISHIFTER_SMOKE_REQUIRED=1    treat a missing xvfb-run as a failure
#   HIFISHIFTER_SMOKE_TIMEOUT=45    smoke timeout in seconds
#   HIFISHIFTER_EXTERNAL_MODELS=1   drop the weights inside the AppImage, keep only models.json (unified package)
#   HIFISHIFTER_SMOKE_MODELS_DIR=DIR external model directory; for the smoke test with external models
#   ARCH=x86_64|aarch64             target architecture for appimagetool (default uname -m)

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SELF_DIR/../.." && pwd)

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

HFS_APP_ARCH=""
normalize_arch() {
    # [Why write into a variable instead of stdout] The top level calls this function twice; with
    # an invalid ARCH, if die fires inside the command-substitution subshell, the outer shell sees
    # one failure yet keeps running usage, and the error is duplicated. Parse once here in the
    # **current shell** and exit directly; everything after reuses this result.
    case "${ARCH:-$(uname -m)}" in
        x86_64 | amd64) HFS_APP_ARCH=x86_64 ;;
        aarch64 | arm64) HFS_APP_ARCH=aarch64 ;;
        *) die "unsupported architecture: ${ARCH:-$(uname -m)}" ;;
    esac
}
normalize_arch

# Files that must ship in the bundle — every one required (relative to AppDir).
#
# [Why libwebgpu_dawn.so is per-architecture] ort on Linux ARM64 deliberately does not enable the
# webgpu feature (no prebuilt dawn; see the target branches in backend/src-tauri/Cargo.toml; the
# HFS_ARCH=arm64 branch of scripts/plugin/unix/lib.sh reaches the same conclusion) — the artifact
# neither contains nor NEEDs dawn. Listing it unconditionally would fail the linux-arm64 release
# leg in verify at both "file exists" and "NEEDED resolution", and CI's if:failure() backstop
# would upload the AppImage that failed verification anyway.
case "$HFS_APP_ARCH" in
    aarch64)
        REQUIRED_FILES=(
            "usr/bin/HiFiShifter"
            "usr/lib/HiFiShifter/libSoundTouchDLL.so"
            "usr/lib/HiFiShifter/models/models.json"
        )
        ;;
    *)
        REQUIRED_FILES=(
            "usr/bin/HiFiShifter"
            "usr/lib/HiFiShifter/libSoundTouchDLL.so"
            "usr/lib/HiFiShifter/libwebgpu_dawn.so"
            "usr/lib/HiFiShifter/models/models.json"
        )
        ;;
esac

# Libraries the host must provide — not one may remain in the bundle (globs, applied to the top
# level of usr/lib and usr/lib64).
#
# There is only one criterion: **who consumes it**. WebKit / GStreamer / GL / Wayland are
# consumed by the host, so these libraries must come from the host as whole groups; leaving half
# a group behind reproduces the 2026-07 failure.
HOST_OWNED_GLOBS=(
    'libwebkit2gtk-*.so*'
    'libjavascriptcoregtk-*.so*'
    'libgst*.so*'
    'libicu*.so*'
    'libharfbuzz-icu.so*'
    'libenchant-2.so*'
    'libsoup-3.0.so*'
    'libwayland-*.so*'
    'libxkbcommon*.so*'
    'libgbm.so*'
    'libdrm*.so*'
    'libEGL*.so*'
    'libGLESv2.so*'
    'libglapi.so*'
)

# Entries among the executable's direct NEEDED that must "land outside the bundle" — the
# executable-level assertion of the "single graphics stack". libwayland-client is listed
# separately, because the 2026-07 user workaround was precisely preloading it.
MUST_RESOLVE_HOST=(
    libwebkit2gtk-4.1.so.0
    libjavascriptcoregtk-4.1.so.0
    libsoup-3.0.so.0
    libwayland-client.so.0
)

# Entries among the direct NEEDED that must "land inside the bundle" (own libraries + the GTK that Tauri bundles).
#
# [Why libSoundTouchDLL.so is absent] SoundTouch has moved to **runtime dlopen** (libloading with
# absolute paths; see pack_layout_candidates in kernel soundtouch.rs): it is not in DT_NEEDED, so
# ldd naturally cannot resolve it, and rpath would not help either (linuxdeploy replaces the whole
# RUNPATH with patchelf). Bundled presence is asserted by REQUIRED_FILES, and the loader's
# candidate list is pinned by the kernel unit test
# pack_layout_candidates_cover_every_bundled_location — the two lists must mirror each other.
#
# [Why libwebgpu_dawn.so is per-architecture] Same note as REQUIRED_FILES above: ARM64 Linux has
# no dawn, and the NEEDED assertion must skip it on that platform.
case "$HFS_APP_ARCH" in
    aarch64)
        MUST_RESOLVE_BUNDLE=(
            libgtk-3.so.0
            libgdk-3.so.0
        )
        ;;
    *)
        MUST_RESOLVE_BUNDLE=(
            libgtk-3.so.0
            libgdk-3.so.0
            libwebgpu_dawn.so
        )
        ;;
esac

usage() {
    cat <<'EOF'
Usage:
  linux-appimage.sh package <release-dir>   prune → verify → repack → smoke (one entry point for CI and local builds)
  linux-appimage.sh prune   <AppDir>        remove the libraries the host must provide
  linux-appimage.sh verify  <AppDir>        assert what the bundle must and must not contain, and where dynamic resolution lands
  linux-appimage.sh repack  <AppDir> [out]  regenerate the AppImage with appimagetool
  linux-appimage.sh smoke   <AppImage>      start under xvfb, requiring a startup log and no abort
EOF
}

note() {
    printf '\n=== %s ===\n' "$*"
}

require_dir() {
    [ -d "$1" ] || die "not a directory: $1"
}

# Find a usable appimagetool: check PATH first, then the cache, then download.
ensure_appimagetool() {
    local arch="$1"
    if command -v appimagetool >/dev/null 2>&1; then
        command -v appimagetool
        return 0
    fi
    local cache_dir="$HOME/.cache/hifishifter"
    local cached="$cache_dir/appimagetool-$arch.AppImage"
    if [ ! -x "$cached" ]; then
        mkdir -p "$cache_dir"
        printf 'downloading appimagetool (%s)...\n' "$arch" >&2
        curl -fsSL -o "$cached" \
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage" \
            || die "failed to download appimagetool; put it on PATH"
        chmod +x "$cached"
    fi
    printf '%s\n' "$cached"
}

# ── prune ─────────────────────────────────────────────────────────────

cmd_prune() {
    local appdir="$1"
    require_dir "$appdir"

    local before after removed=0
    before=$(du -sm "$appdir" | cut -f1)

    local dir glob path
    for dir in "$appdir/usr/lib" "$appdir/usr/lib64"; do
        [ -d "$dir" ] || continue
        for glob in "${HOST_OWNED_GLOBS[@]}"; do
            while IFS= read -r -d '' path; do
                rm -f -- "$path"
                removed=$((removed + 1))
            done < <(find "$dir" -maxdepth 1 -name "$glob" -print0 2>/dev/null)
        done
    done

    # WebKit's helper processes (WebKitWebProcess / WebKitNetworkProcess / injected bundle)
    # and the GStreamer plugin directory: only WebKit uses the former; the latter is tightly
    # bound to the host's core version.
    while IFS= read -r -d '' path; do
        rm -rf -- "$path"
        removed=$((removed + 1))
    done < <(find "$appdir/usr/lib" -maxdepth 3 -type d \
        \( -name 'webkit2gtk-4.1' -o -name 'webkitgtk-4.1' -o -name 'gstreamer-1.0' \) -print0 2>/dev/null)

    after=$(du -sm "$appdir" | cut -f1)
    printf 'pruned %d entries: %s MiB -> %s MiB\n' "$removed" "$before" "$after"

    if [ "$removed" -eq 0 ]; then
        # Pruning 0 entries usually means the AppDir was pruned before; not an error, but worth saying.
        printf 'note: no host-owned libraries matched (possibly already pruned)\n'
    fi
}

# ── strip models ──────────────────────────────────────────────────────

# The unified package puts the weights in the install directory's models/<version>, and the
# AppImage keeps only the manifest. An AppImage is read-only squashfs; at runtime the weights
# cannot be "shared" out of it to the plugins. If the weights stayed in the AppImage, after
# installing, the same weights would exist in both the AppImage and the shared store.
cmd_strip_models() {
    local appdir="$1" models
    require_dir "$appdir"
    models="$appdir/usr/lib/HiFiShifter/models"
    require_dir "$models"
    [ -f "$models/models.json" ] || die "missing model manifest: $models/models.json"
    find "$models" -mindepth 1 -maxdepth 1 ! -name 'models.json' -exec rm -rf {} +
    printf 'external models: %s (kept only models.json)\n' "$models"
}

# ── verify ────────────────────────────────────────────────────────────

# Take a soname's resolved path from ldd output (empty if unresolvable).
resolve_of() {
    local ldd_output="$1" soname="$2"
    printf '%s\n' "$ldd_output" | awk -v n="$soname" '$1 == n && $2 == "=>" { print $3; exit }'
}

cmd_verify() {
    local appdir="$1"
    require_dir "$appdir"

    local failures=0
    local f path glob name resolved

    note "bundled files"
    for f in "${REQUIRED_FILES[@]}"; do
        if [ -e "$appdir/$f" ]; then
            printf 'ok    %s\n' "$f"
        else
            printf 'FAIL  %s does not exist\n' "$f" >&2
            failures=$((failures + 1))
        fi
    done

    note "host-owned libraries must not be bundled"
    local bundled=0
    for glob in "${HOST_OWNED_GLOBS[@]}"; do
        while IFS= read -r -d '' path; do
            printf 'FAIL  %s must come from the host but was bundled\n' "${path#"$appdir"/}" >&2
            failures=$((failures + 1))
            bundled=$((bundled + 1))
        done < <(find "$appdir/usr/lib" "$appdir/usr/lib64" -maxdepth 1 -name "$glob" -print0 2>/dev/null)
    done
    [ "$bundled" -eq 0 ] && printf 'ok    no GL / Wayland / WebKit / GStreamer libraries inside the bundle\n'

    note "dynamic resolution"
    local ld_path="$appdir/usr/lib:$appdir/usr/lib/HiFiShifter"
    # GTK's modules (gdk-pixbuf / immodules) live in triplet-suffixed subdirectories; add them to the search path too.
    for path in "$appdir"/usr/lib/*-linux-gnu; do
        [ -d "$path" ] && ld_path="$ld_path:$path"
    done

    local ldd_output
    if ! ldd_output=$(LD_LIBRARY_PATH="$ld_path" ldd "$appdir/usr/bin/HiFiShifter" 2>&1); then
        printf '%s\n' "$ldd_output" >&2
        die "ldd could not resolve $appdir/usr/bin/HiFiShifter"
    fi

    if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
        printf '%s\n' "$ldd_output" | grep 'not found' >&2
        die "a dependency is neither in the bundle nor on the host"
    fi

    for name in "${MUST_RESOLVE_HOST[@]}"; do
        resolved=$(resolve_of "$ldd_output" "$name")
        if [ -z "$resolved" ]; then
            printf 'FAIL  %s not in NEEDED — assertion stale, update the script\n' "$name" >&2
            failures=$((failures + 1))
        elif [[ "$resolved" == "$appdir"/* ]]; then
            printf 'FAIL  %s still resolves inside the bundle: %s\n' "$name" "$resolved" >&2
            failures=$((failures + 1))
        else
            printf 'ok    %s -> %s (host)\n' "$name" "$resolved"
        fi
    done

    for name in "${MUST_RESOLVE_BUNDLE[@]}"; do
        resolved=$(resolve_of "$ldd_output" "$name")
        if [ -z "$resolved" ]; then
            printf 'FAIL  %s not in NEEDED — assertion stale, update the script\n' "$name" >&2
            failures=$((failures + 1))
        elif [[ "$resolved" != "$appdir"/* ]]; then
            printf 'FAIL  %s should be bundled but resolves to %s\n' "$name" "$resolved" >&2
            failures=$((failures + 1))
        else
            printf 'ok    %s -> bundled\n' "$name"
        fi
    done

    if [ "$failures" -ne 0 ]; then
        die "$failures check(s) failed"
    fi
    printf '\nall checks passed\n'
}

# ── repack ────────────────────────────────────────────────────────────

cmd_repack() {
    local appdir="$1" output="${2:-}"
    require_dir "$appdir"

    if [ -z "$output" ]; then
        # Reuse an AppImage name already present in the directory; otherwise compose one from product + version + arch.
        output=$(find "$(dirname "$appdir")" -maxdepth 1 -name '*.AppImage' -print -quit 2>/dev/null || true)
        if [ -z "$output" ]; then
            local version product
            version=$(jq -r '.version' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")
            product=$(jq -r '.productName' "$REPO_ROOT/backend/src-tauri/tauri.conf.json")
            output="$(dirname "$appdir")/${product// /_}_${version}_${HFS_APP_ARCH}.AppImage"
        fi
    fi

    local arch tool
    arch="$HFS_APP_ARCH"
    tool=$(ensure_appimagetool "$arch")

    rm -f -- "$output"
    printf 'repacking %s -> %s (%s)\n' "$appdir" "$output" "$arch"
    APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$tool" "$appdir" "$output" >/dev/null
    [ -f "$output" ] || die "appimagetool produced no $output"
    printf 'repacked: %s (%s)\n' "$output" "$(du -h "$output" | cut -f1)"
}

# ── smoke ─────────────────────────────────────────────────────────────

cmd_smoke() {
    if [ "${HIFISHIFTER_SKIP_SMOKE:-0}" = "1" ]; then
        printf 'skipping smoke (HIFISHIFTER_SKIP_SMOKE=1)\n'
        return 0
    fi
    local appimage="$1"
    [ -f "$appimage" ] || die "AppImage not found: $appimage"

    local runner=()
    if command -v xvfb-run >/dev/null 2>&1; then
        runner=(xvfb-run -a)
    elif [ "${HIFISHIFTER_SMOKE_REQUIRED:-0}" = "1" ]; then
        die "xvfb-run is required (install xvfb in CI)"
    else
        printf 'skip: no xvfb-run, skipping smoke (xvfb is optional locally)\n' >&2
        return 0
    fi

    local log rc=0 workdir
    log=$(mktemp)
    # Give this run its own TMPDIR: APPIMAGE_EXTRACT_AND_RUN extracts under fixed names, and
    # reusing the same directory means re-verifying every file on every run (tens of thousands of
    # "file size matches" lines), which slows the smoke test and buries the real startup log.
    workdir=$(mktemp -d)
    local -a smoke_env=(TMPDIR="$workdir" APPIMAGE_EXTRACT_AND_RUN=1)
    if [ -n "${HIFISHIFTER_SMOKE_MODELS_DIR:-}" ]; then
        smoke_env+=("HIFISHIFTER_MODELS_DIR=$HIFISHIFTER_SMOKE_MODELS_DIR")
    fi
    set +e
    env "${smoke_env[@]}" timeout "${HIFISHIFTER_SMOKE_TIMEOUT:-45}" \
        "${runner[@]}" "$appimage" >"$log" 2>&1
    rc=$?
    set -e
    rm -rf "$workdir"

    # 124 = still alive when time ran out (normal: this is a GUI app); 0 / 143 = exited on its own or got TERM.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 143 ]; then
        printf '%s\n' '--- smoke output ---' >&2
        cat "$log" >&2
        rm -f "$log"
        die "AppImage failed to start (exit code ${rc})"
    fi

    local failed=0
    if ! grep -q 'starting; log file' "$log"; then
        printf '%s\n' 'FAIL  no startup log appeared' >&2
        failed=1
    fi
    if grep -qiE 'could not create default egl display|Aborting\.\.\.|error while loading shared libraries' "$log"; then
        printf '%s\n' 'FAIL  fatal error during startup:' >&2
        grep -iE 'could not create default egl display|Aborting\.\.\.|error while loading shared libraries' "$log" >&2
        failed=1
    fi
    if [ "$failed" -ne 0 ]; then
        printf '%s\n' '--- smoke output ---' >&2
        cat "$log" >&2
        rm -f "$log"
        die "smoke test failed"
    fi

    printf 'smoke ok (kept running for %ss, startup log normal)\n' "${HIFISHIFTER_SMOKE_TIMEOUT:-45}"
    rm -f "$log"
}

# ── package ───────────────────────────────────────────────────────────

find_appdir() {
    local release_dir="$1"
    local candidate="$release_dir/bundle/appimage/HiFiShifter.AppDir"
    [ -d "$candidate" ] || return 1
    printf '%s\n' "$candidate"
}

cmd_package() {
    local release_dir="$1"
    require_dir "$release_dir"

    local appdir
    appdir=$(find_appdir "$release_dir") || die "AppDir not found: $release_dir/bundle/appimage/HiFiShifter.AppDir"

    note "prune"
    cmd_prune "$appdir"

    if [ "${HIFISHIFTER_EXTERNAL_MODELS:-0}" = "1" ]; then
        note "strip models"
        cmd_strip_models "$appdir"
    fi

    note "verify"
    cmd_verify "$appdir"

    note "repack"
    cmd_repack "$appdir"

    note "smoke"
    # The unified AppImage carries no weights; default the smoke test to the full models in the build tree.
    if [ "${HIFISHIFTER_EXTERNAL_MODELS:-0}" = "1" ] && [ -z "${HIFISHIFTER_SMOKE_MODELS_DIR:-}" ]; then
        HIFISHIFTER_SMOKE_MODELS_DIR="$REPO_ROOT/backend/src-tauri/resources/models"
    fi
    cmd_smoke "$(find "$(dirname "$appdir")" -maxdepth 1 -name '*.AppImage' -print -quit)"
}

cmd="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
case "$cmd" in
    package) cmd_package "${1:-}" ;;
    prune) cmd_prune "${1:-}" ;;
    verify) cmd_verify "${1:-}" ;;
    repack) cmd_repack "${1:-}" "${2:-}" ;;
    smoke) cmd_smoke "${1:-}" ;;
    help | -h | --help) usage ;;
    *)
        usage
        exit 2
        ;;
esac
