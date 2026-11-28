#!/usr/bin/env bash
# Shared rules for the portable, per-user Linux install.
#
# [Why this exists at all] The distribution packages (scripts/release/linux-packages) cover
# Debian/Ubuntu and Fedora/RHEL. This is the path for everybody else: no root, offline, an
# unknown distribution, or a machine where the user simply wants the app in their home folder.
# It must therefore never assume a package manager -- and, since the last wizard's four dialog
# backends were the thing that broke, it must never assume a dialog program either.
#
# [Why the installer and the uninstaller share this file] They must agree on every path, and they
# run months apart: the uninstaller copy is installed next to the app, long after the download is
# gone. A second list of paths would drift the first time a directory moves, and the symptom is
# the worst kind -- an "uninstalled" machine that still has a VST3 the DAW keeps loading.
#
# Convention: this file sets no shell options and never calls exit; the entry scripts own both.

hfs_die() { printf 'error: %s\n' "$*" >&2; exit 1; }
hfs_log() { printf '%s\n' "$*"; }

hfs_home() { printf '%s' "${HOME:-}"; }

# <app-dir>: set every path this product owns.
#
# [Why XDG variables are honoured] A user who moved XDG_DATA_HOME expects the menu entry and the
# model library to follow; writing to the literal ~/.local/share would put the files where
# nothing looks. ~/.vst3 and ~/.clap are not XDG paths -- they are the VST3 and CLAP conventions,
# which is exactly why they are used.
hfs_layout() {
    local app_dir="$1" home data
    home="$(hfs_home)"
    [ -n "$home" ] || hfs_die "HOME is not set: a per-user install needs it (use the .deb/.rpm for a system install)"
    data="${XDG_DATA_HOME:-$home/.local/share}"
    HFS_APP_DIR="$app_dir"
    HFS_BIN_DIR="$home/.local/bin"
    HFS_DATA_DIR="$data/HiFiShifter"
    HFS_MODELS_DIR="$HFS_DATA_DIR/models"
    HFS_VST3_DIR="$home/.vst3"
    HFS_CLAP_DIR="$home/.clap"
    HFS_DESKTOP_DIR="$data/applications"
    HFS_ICON_DIR="$data/icons/hicolor"
    HFS_MIME_DIR="$data/mime/packages"
    HFS_MANIFEST="$HFS_DATA_DIR/install-manifest"
}

# <path>: print the path with a trailing slash, for prefix comparisons.
hfs_slash() {
    case "$1" in
        */) printf '%s' "$1" ;;
        *) printf '%s/' "$1" ;;
    esac
}

# <path>: 0 when removing it cannot take anything that is not ours.
#
# [Why this exists at all] The uninstaller removes paths read back from a manifest file. If that
# file were edited (or truncated by a full disk and then hand-repaired), a plain "rm -rf" would
# delete whatever the line says. Every removal therefore has to sit under a root this product is
# allowed to own, and the roots themselves (/, $HOME, ~/.local) are refused even when they are
# prefixes of the paths above them.
hfs_is_safe_remove() {
    local path="$1" root home
    [ -n "$path" ] || return 1
    case "$path" in
        *$'\n'* | *$'\t'*) return 1 ;;
        / | /*/../* | */..) return 1 ;;
    esac
    home="$(hfs_home)"
    case "$path" in
        "" | "$home" | "$home/.local" | "$home/.local/share" | "$home/.local/lib" | "$home/.config" | "$home/.cache" | "$home/.local/state") return 1 ;;
        "/opt" | "/usr" | "/usr/local" | "/usr/share" | "/usr/lib" | "/etc" | "/var") return 1 ;;
    esac
    for root in "$HFS_APP_DIR" "$HFS_DATA_DIR" "$HFS_BIN_DIR" "$HFS_VST3_DIR" "$HFS_CLAP_DIR" "$HFS_DESKTOP_DIR" "$HFS_ICON_DIR" "$HFS_MIME_DIR"; do
        [ -n "$root" ] || continue
        case "$(hfs_slash "$path")" in
            "$(hfs_slash "$root")"*) return 0 ;;
        esac
    done
    return 1
}

# <file>: read the manifest the installer wrote.
#
# Sets HFS_MANIFEST_SCOPE, HFS_MANIFEST_APP_DIR, HFS_MANIFEST_VERSION, and fills
# HFS_MANIFEST_PATHS (files, in the order they were written) and HFS_MANIFEST_DIRS.
hfs_manifest_read() {
    local file="$1" line key value
    HFS_MANIFEST_SCOPE=""
    HFS_MANIFEST_APP_DIR=""
    HFS_MANIFEST_VERSION=""
    HFS_MANIFEST_PATHS=()
    HFS_MANIFEST_DIRS=()
    [ -f "$file" ] || return 1
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            '' | '#'*) continue ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            scope) HFS_MANIFEST_SCOPE="$value" ;;
            app_dir) HFS_MANIFEST_APP_DIR="$value" ;;
            version) HFS_MANIFEST_VERSION="$value" ;;
            path) HFS_MANIFEST_PATHS+=("$value") ;;
            dir) HFS_MANIFEST_DIRS+=("$value") ;;
        esac
    done <"$file"
    return 0
}

# <string>: escape for a Desktop Entry value (backslash and quote).
hfs_desktop_value() {
    local text="$1"
    text="${text//\\/\\\\}"
    text="${text//\"/\\\"}"
    printf '%s' "$text"
}

# <file> <app-image>
#
# [Why the launcher checks for FUSE] A type-2 AppImage needs libfuse.so.2, which Ubuntu 24.04 no
# longer installs by default. Without it the app dies with "dlopen(): error loading libfuse.so.2"
# the first time the user clicks the menu entry -- an install that looks fine and starts nothing.
# The AppImage runtime can extract itself instead: slower, but it works everywhere, so it is used
# exactly when FUSE is absent.
hfs_write_launcher() {
    local file="$1" quoted="$2"
    quoted="${quoted//\\/\\\\}"
    quoted="${quoted//\"/\\\"}"
    {
        printf '#!/bin/sh\n'
        printf '# HiFiShifter launcher, generated by the installer.\n'
        printf 'app="%s"\n' "$quoted"
        printf 'if command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep -q %s; then\n' "'libfuse\\.so\\.2'"
        printf '    # No libfuse2: let the AppImage extract itself instead of failing to mount.\n'
        printf '    APPIMAGE_EXTRACT_AND_RUN=1\n'
        printf '    export APPIMAGE_EXTRACT_AND_RUN\n'
        printf 'fi\n'
        printf 'exec "$app" "$@"\n'
    } >"$file"
    chmod 755 "$file"
}
