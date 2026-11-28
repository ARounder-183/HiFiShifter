#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Shared rules for Unix plugin packaging (VST3 / CLAP, macOS + Linux).
#
# [Why this layer exists] the two formats' **layouts** differ: VST3 is a
# Contents/<arch>-linux/ bundle even on Linux, while CLAP is a single file on Linux and only
# a bundle on macOS. But the **platform rules are identical**: runtime dependencies must sit
# beside the module, the rpath must be $ORIGIN / @loader_path, the frontend artifact is
# single-sourced, licenses must travel with the package, and the artifact must contain no
# dangling symlinks. These rules used to live only in vst3/unix/build.sh — the CLAP side had
# none of them — so hifishifter-clap/build.rs declared $ORIGIN while packaging never put the
# libraries in, and clap.yml went red on ldd and dyld at the same time.
#
# Only the "format-independent" implementation lives here. Format scripts own their layout,
# entry symbols and black-box probes.
#
# Convention: this file does not set set -e and does not read or write the caller's
# variables; all state lives in HFS_*.
# ─────────────────────────────────────────────────────────────────────────────

hfs_die() { printf '[plugin] ERROR: %s\n' "$*" >&2; exit 1; }
hfs_log() { printf '[plugin] %s\n' "$*"; }

# Architecture tag of a Rust target triple, in the same spelling hfs_detect_platform uses.
hfs_arch_of_triple() {
    case "$1" in
        x86_64-*)          printf 'x86_64\n' ;;
        aarch64-*|arm64-*) printf 'arm64\n' ;;
        *)                 hfs_die "unsupported target triple: $1 (expected x86_64-* or aarch64-*)" ;;
    esac
}

# Platform triple. macOS keeps uname's architecture name (arm64); Linux uses dpkg's spelling
# (aarch64) — this matches the existing artifact naming HiFiShifter_v<version>_<os>-<arch>-*,
# and renaming would break continuity with already-published file names.
#
# [Why the two variables may be preset] A universal build has two architectures and therefore
# no single host answer: the cross-compiled slice presets HFS_ARCH so the runtime-dependency
# list and the artifact tag describe the **target**, and the merged package presets
# HFS_PLATFORM_ARCH so the artifact name matches the matrix leg (macos-universal). Deriving
# both from uname would make a cross build ask for the host's dylibs and ship them under the
# host's name — the package would look right and fail to load on the other architecture.
hfs_detect_platform() {
    case "$(uname -s)" in
        Linux)  HFS_OS="linux" ;;
        Darwin) HFS_OS="macos" ;;
        *)      hfs_die "only Linux and macOS are supported, current: $(uname -s)" ;;
    esac
    case "${HFS_ARCH:-}" in
        x86_64|arm64) ;;
        "")
            case "$(uname -m)" in
                x86_64|amd64)  HFS_ARCH="x86_64" ;;
                arm64|aarch64) HFS_ARCH="arm64" ;;
                *)             hfs_die "unsupported CPU architecture: $(uname -m)" ;;
            esac
            ;;
        *) hfs_die "HFS_ARCH must be x86_64 or arm64, got: $HFS_ARCH" ;;
    esac
    if [ -z "${HFS_PLATFORM_ARCH:-}" ]; then
        if [ "$HFS_OS" = "linux" ] && [ "$HFS_ARCH" = "arm64" ]; then
            HFS_PLATFORM_ARCH="aarch64"
        else
            HFS_PLATFORM_ARCH="$HFS_ARCH"
        fi
    fi
    HFS_TAG="$HFS_OS-$HFS_PLATFORM_ARCH"
    export HFS_OS HFS_ARCH HFS_PLATFORM_ARCH HFS_TAG
}

# Architectures inside a Mach-O file, one per line and sorted, so that two files' slices can be
# compared as strings. Empty for a file lipo cannot read.
hfs_macho_archs() {
    lipo -archs "$1" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort
}

# True when the file already carries both slices (lipo refuses a second merge of the same
# architecture, so every merge site has to ask first).
hfs_is_universal() {
    local archs
    archs="$(hfs_macho_archs "$1")"
    printf '%s\n' "$archs" | grep -qx x86_64 && printf '%s\n' "$archs" | grep -qx arm64
}

hfs_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# CFBundleVersion only accepts dot-separated integers; the full version (including -beta) is
# stored under a custom key.
hfs_numeric_version() {
    local version
    version="$(printf '%s' "${1#v}" | sed -E 's/-.*$//')"
    case "$version" in
        [0-9]*.[0-9]*.[0-9]*) printf '%s\n' "$version" ;;
        *) printf '0.0.0\n' ;;
    esac
}

# Runtime libraries that must sit beside the module in the artifact. ort's copy-dylibs puts
# them under the target directory; Linux ARM64's ort deliberately ships without the webgpu
# feature (no prebuilt dawn), so the artifact neither NEEDs nor contains libwebgpu_dawn.so —
# listing it would make the build fail forever on that platform.
hfs_runtime_deps() {
    if [ "$HFS_OS" = "linux" ]; then
        # [Why HFS_ARCH is tested instead of HFS_PLATFORM_ARCH] the latter is renamed to
        # "aarch64" on Linux/arm64 (dpkg's spelling, kept in artifact names) and therefore
        # never equals "arm64" — this branch would never have hit, and linux-aarch64 would
        # demand a dawn that simply does not exist on that platform.
        if [ "$HFS_ARCH" = "arm64" ]; then
            printf '%s\n' "libSoundTouchDLL.so"
        else
            printf '%s\n' "libSoundTouchDLL.so" "libwebgpu_dawn.so"
        fi
    elif [ "$HFS_ARCH" = "x86_64" ]; then
        # macOS Intel uses ort's alternative-backend (pure-Rust tract, see kernel's
        # Cargo.toml): it links neither libonnxruntime nor dawn. Listing dawn would make
        # this leg fail on "missing runtime dependency" while the artifact is actually fine.
        printf '%s\n' "libSoundTouchDLL.dylib"
    else
        printf '%s\n' "libSoundTouchDLL.dylib" "libwebgpu_dawn.dylib"
    fi
}

# <target-dir> <dest-dir>
hfs_stage_runtime_deps() {
    local target="$1" dest="$2" dep src
    mkdir -p "$dest"
    while IFS= read -r dep; do
        [ -n "$dep" ] || continue
        src="$target/$dep"
        [ -e "$src" ] || hfs_die "missing runtime dependency ${src} (build first: ort's copy-dylibs puts it under the target directory)"
        # -L: ORT's dawn under target/ is a symlink into ~/.cache; a plain cp would leave a
        # dangling link inside the archive (pointing at a path that does not exist on the
        # user's machine).
        cp -Lf "$src" "$dest/$dep"
        hfs_log "dependency ${dep} ($(du -h "$dest/$dep" | cut -f1))"
    done <<< "$(hfs_runtime_deps)"
}

# <resources-dir> [<frontend-dist>]
hfs_stage_frontend() {
    local res="$1" src="${2:-${HFS_FRONTEND_DIST:-}}"
    [ -n "$src" ] || hfs_die "no frontend artifact directory given"
    [ -s "$src/plugin.html" ] || hfs_die "missing frontend artifact: $src/plugin.html"
    [ -d "$src/assets" ] || hfs_die "missing $src/assets: the plugin entry has no script to load"
    mkdir -p "$res/frontend"
    cp -f "$src/plugin.html" "$res/frontend/"
    rm -rf "$res/frontend/assets"
    cp -R "$src/assets" "$res/frontend/"
    hfs_log "frontend assets ($(du -sh "$res/frontend" | cut -f1))"
}

# Models must ship in the package. Windows has an installer-owned shared model store; the
# unified macOS/Linux installer also keeps the weights in a shared store and lets only
# models.json travel with the bundle. The plugin-only package (not going through the unified
# installer) stays self-contained: extract and it works, no external directory.
# <resources-dir> <models-source>
hfs_stage_models() {
    local res="$1" src="$2"
    [ -d "$src" ] || hfs_die "missing model resources directory: $src"
    mkdir -p "$res/models"
    cp -R "$src/." "$res/models/"
    [ -f "$res/models/models.json" ] \
        || hfs_die "model manifest missing: run python3 scripts/setup/fetch-models.py --write-manifest first"
}

# Compress the bundle's models directory to "manifest only" form. The unified installer
# keeps the weights in the shared store, and the standalone app plus every plugin locate
# that one copy via the version recorded in models.json; leaving a full copy in the bundle
# would just double the pkg/tar size.
# <models-dir>
hfs_strip_models_to_manifest() {
    local models="$1"
    [ -f "$models/models.json" ] || hfs_die "missing model manifest: $models/models.json"
    find "$models" -mindepth 1 -maxdepth 1 ! -name 'models.json' -exec rm -rf {} \;
    [ "$(find "$models" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" = "1" ] \
        || hfs_die "models directory was not cleaned up: $models"
}

# <dest-dir> <repo-root>
hfs_stage_licenses() {
    local dest="$1" root="$2" extra
    for extra in LICENSE licenses; do
        [ -e "$root/$extra" ] || hfs_die "missing license: $extra"
        cp -R "$root/$extra" "$dest/"
    done
}

# <plist-path> <executable> <bundle-id> <version> <signature>
hfs_write_info_plist() {
    local path="$1" exec_name="$2" bundle_id="$3" version="$4" signature="$5"
    local numeric
    numeric="$(hfs_numeric_version "$version")"
    {
        printf '<?xml version="1.0" encoding="UTF-8"?>\n'
        printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        printf '<plist version="1.0">\n<dict>\n'
        printf '  <key>CFBundleDevelopmentRegion</key><string>en</string>\n'
        printf '  <key>CFBundleExecutable</key><string>%s</string>\n' "$exec_name"
        printf '  <key>CFBundleIdentifier</key><string>%s</string>\n' "$bundle_id"
        printf '  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\n'
        printf '  <key>CFBundleName</key><string>HiFiShifter</string>\n'
        printf '  <key>CFBundleDisplayName</key><string>HiFiShifter</string>\n'
        printf '  <key>CFBundlePackageType</key><string>BNDL</string>\n'
        printf '  <key>CFBundleSignature</key><string>%s</string>\n' "$signature"
        printf '  <key>CFBundleShortVersionString</key><string>%s</string>\n' "$numeric"
        printf '  <key>CFBundleVersion</key><string>%s</string>\n' "$numeric"
        printf '  <key>NSHighResolutionCapable</key><true/>\n'
        printf '  <key>HiFiShifterVersion</key><string>%s</string>\n' "$version"
        printf '</dict>\n</plist>\n'
    } >"$path"
    hfs_log "Info.plist generated (CFBundleVersion=${numeric})"
}

# Signing identity: ad-hoc by default (an unsigned Mach-O simply fails to load on arm64).
# Release signing and notarization are enabled via HIFISHIFTER_CODESIGN_IDENTITY; the build
# script does not pretend to own a certificate.
# <bundle-dir> <module-file>
hfs_sign_bundle() {
    local bundle="$1" module="$2" identity file
    identity="${HIFISHIFTER_CODESIGN_IDENTITY:--}"
    command -v codesign >/dev/null 2>&1 || hfs_die "codesign missing"
    local -a args=(--force --sign "$identity")
    if [ "$identity" != "-" ]; then
        # Notarization requires hardened runtime + a secure timestamp; with ad-hoc signing
        # these two flags are pointless.
        args+=(--options runtime --timestamp)
    fi
    # The order is a hard constraint: dependencies first, then the module, then the bundle —
    # the bundle's signature covers the whole directory tree, and the other way around would
    # instantly invalidate the first two steps' signatures.
    while IFS= read -r file; do
        [ -n "$file" ] || continue
        codesign "${args[@]}" "$file" >/dev/null 2>&1 || hfs_die "failed to sign dependency: $file"
    done < <(find "$bundle" -type f \( -name '*.dylib' -o -name '*.so' \) -print)
    codesign "${args[@]}" "$module" >/dev/null 2>&1 || hfs_die "failed to sign module: $module"
    codesign "${args[@]}" "$bundle" >/dev/null 2>&1 || hfs_die "failed to sign bundle: $bundle"
    if [ "$identity" = "-" ]; then
        hfs_log "ad-hoc signing done"
    else
        hfs_log "signed with $identity"
    fi
}

# Dependencies placed beside the module require a $ORIGIN / @loader_path search path: the
# user's machine has no LD_LIBRARY_PATH — that is the host process's environment, which we
# have no right to demand.
# <module-file>
hfs_assert_rpath() {
    local module="$1" paths
    if [ "$HFS_OS" = "linux" ]; then
        command -v readelf >/dev/null 2>&1 || { hfs_log "SKIP rpath: no readelf"; return 0; }
        paths="$(readelf -d "$module" 2>/dev/null | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p' | tr '\n' ':')"
        case ":$paths" in
            *':$ORIGIN'*) hfs_log "rpath contains \$ORIGIN (${paths})"; return 0 ;;
        esac
        hfs_die "module has no \$ORIGIN rpath (current: ${paths:-none}): dependencies would fail to resolve on the user's machine"
    else
        command -v otool >/dev/null 2>&1 || { hfs_log "SKIP rpath: no otool"; return 0; }
        otool -l "$module" 2>/dev/null | grep -A2 'LC_RPATH' | grep -q '@loader_path' \
            || hfs_die "module has no @loader_path rpath: dependencies inside the bundle would fail to resolve"
        hfs_log "rpath contains @loader_path"
    fi
}

# <dir>
hfs_check_no_dangling() {
    local dir="$1" dangling
    dangling="$(find "$dir" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
    [ -z "$dangling" ] || hfs_die "artifact contains dangling symlinks: $dangling"
}

# Scene replay when a cargo build fails. **Does not exit**: the caller keeps its own
# [vst3] / [clap] prefix.
# <log-file> <exit-status>
#
# [Why the error line is located first] a build script's panic lands in the **middle** of the
# log: cargo lets its parallel jobs finish before exiting, so hundreds of compile warnings
# from native dependencies (World's C++) follow it. Replaying only the tail produces the
# scene "a pile of warnings + exit code 101" that looks causeless — vst3-unix.yml was once
# misread as OOM for a whole round until the full log was dug out and the real panic showed.
hfs_report_build_failure() {
    local log="$1" status="$2" first_line window
    if [ ! -f "$log" ]; then
        printf '[plugin] no build log: %s\n' "$log" >&2
        return 0
    fi
    if [ "$status" -eq 137 ] || [ "$status" -eq 143 ]; then
        printf '[plugin] process killed by signal (exit %s): the runner is short on memory or disk.\n' "$status" >&2
    fi
    # CARGO_TERM_COLOR=never is the precondition for this criterion (otherwise lines start
    # with escape sequences).
    first_line="$(grep -n -m1 -E '^error(\[|:)' "$log" 2>/dev/null | cut -d: -f1 || true)"
    if [ -n "$first_line" ]; then
        window=$((first_line + 40))
        printf '[plugin] first error in the log (from line %s):\n' "$first_line" >&2
        sed -n "${first_line},${window}p" "$log" >&2
        printf '[plugin] last 20 lines of the log:\n' >&2
        tail -20 "$log" >&2
    else
        printf '[plugin] no error lines in the log; last 120 lines:\n' >&2
        tail -120 "$log" >&2
    fi
}

# <output.tar.gz> <workdir> <entry>...
hfs_tarball() {
    local out="$1" work="$2"
    shift 2
    [ $# -gt 0 ] || hfs_die "no entries given for packaging"
    rm -f "$out"
    ( cd "$work" && tar -czf "$out" "$@" ) || hfs_die "packaging failed: $out"
    local digest
    digest="$(hfs_sha256 "$out")"
    printf '%s  %s\n' "$digest" "$(basename "$out")" >"$out.sha256"
    hfs_log "artifact $(du -h "$out" | cut -f1): $out"
    hfs_log "sha256 $digest"
}
