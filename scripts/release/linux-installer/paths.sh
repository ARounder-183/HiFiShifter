#!/usr/bin/env bash
# HiFiShifter Linux installer: every path the product owns, computed in exactly one place.
#
# [Why the installer and the uninstaller share this file] They must agree on every path, and
# they run months apart: the uninstaller copy is installed next to the app, long after the
# download is gone. A second list of paths would drift the first time a directory moves, and the
# symptom is the worst kind -- an "uninstalled" machine that still has a VST3 the DAW keeps
# loading. So both sides source this file.
#
# [Why only the app folder is configurable] The plug-in folders, the menu entry and the icon
# theme are conventions a DAW and a desktop already scan (~/.vst3, ~/.clap, XDG_DATA_DIRS).
# Asking about them would produce installs nothing finds. The app folder has no such convention,
# so it is the only thing the wizard asks for -- and the only value the manifest has to record
# beyond "which scope".

# [Why HOME is resolved here and not at every call site] The uninstaller may run from a desktop
# entry, where HOME is always set, or from a root shell, where it is /root. Callers pass the
# scope they mean; this file never guesses.
hfs_home() {
    printf '%s' "${HOME:-}"
}

# <scope>: the default app folder. Kept in one place because both the wizard's directory page and
# the uninstaller's fallback (when the manifest is gone) need the same answer.
hfs_default_app_dir() {
    case "$1" in
        system) printf '%s' "/opt/hifishifter" ;;
        *) printf '%s' "$(hfs_home)/.local/lib/hifishifter" ;;
    esac
}

# <scope> [<app_dir>]: set every path this product owns.
#
# Sets HFS_SCOPE, HFS_APP_DIR, HFS_BIN_DIR, HFS_DATA_DIR, HFS_VST3_DIR, HFS_CLAP_DIR,
# HFS_DESKTOP_DIR, HFS_ICON_DIR, HFS_MIME_DIR, HFS_CACHE_DIR, HFS_STATE_DIR, HFS_CONFIG_DIR.
#
# [Why XDG variables are honoured for a per-user install] A user who moved XDG_DATA_HOME expects
# the menu entry and the model library to follow; writing to the literal ~/.local/share would put
# the files where nothing looks. For a system install the FHS locations are fixed by definition.
hfs_layout() {
    HFS_SCOPE="$1"
    HFS_APP_DIR="${2:-$(hfs_default_app_dir "$1")}"
    case "$HFS_SCOPE" in
        system)
            HFS_BIN_DIR="/usr/local/bin"
            HFS_DATA_DIR="/usr/local/share/HiFiShifter"
            HFS_VST3_DIR="/usr/lib/vst3"
            HFS_CLAP_DIR="/usr/lib/clap"
            HFS_DESKTOP_DIR="/usr/share/applications"
            HFS_ICON_DIR="/usr/share/icons/hicolor"
            HFS_MIME_DIR="/usr/share/mime/packages"
            HFS_CACHE_DIR="/var/cache/hifishifter"
            HFS_STATE_DIR="/var/log/hifishifter"
            HFS_CONFIG_DIR="/etc/hifishifter"
            ;;
        *)
            local home data
            home="$(hfs_home)"
            data="${XDG_DATA_HOME:-$home/.local/share}"
            HFS_BIN_DIR="$home/.local/bin"
            HFS_DATA_DIR="$data/HiFiShifter"
            HFS_VST3_DIR="$home/.vst3"
            HFS_CLAP_DIR="$home/.clap"
            HFS_DESKTOP_DIR="$data/applications"
            HFS_ICON_DIR="$data/icons/hicolor"
            HFS_MIME_DIR="$data/mime/packages"
            HFS_CACHE_DIR="${XDG_CACHE_HOME:-$home/.cache}/HiFiShifter"
            HFS_STATE_DIR="${XDG_STATE_HOME:-$home/.local/state}/HiFiShifter"
            HFS_CONFIG_DIR="${XDG_CONFIG_HOME:-$home/.config}/HiFiShifter"
            ;;
    esac
    HFS_MANIFEST_FILE="$HFS_DATA_DIR/install-manifest"
}

# The WebView profile Tauri derives from the bundle identifier. It is derived data (cookies,
# localStorage), never settings -- the settings live in HFS_CONFIG_DIR and are kept.
hfs_webview_cache_dir() {
    local home
    home="$(hfs_home)"
    printf '%s' "${XDG_CACHE_HOME:-$home/.cache}/com.arounder.hifishifter"
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
        "" | "$home" | "$home/.local" | "$home/.local/share" | "$home/.config" | "$home/.cache" | "$home/.local/state") return 1 ;;
        "/opt" | "/usr" | "/usr/local" | "/usr/share" | "/usr/lib" | "/etc" | "/var") return 1 ;;
    esac
    for root in "$HFS_APP_DIR" "$HFS_DATA_DIR" "$HFS_BIN_DIR" "$HFS_VST3_DIR" "$HFS_CLAP_DIR" "$HFS_DESKTOP_DIR" "$HFS_ICON_DIR" "$HFS_MIME_DIR" "$HFS_CACHE_DIR" "$HFS_STATE_DIR" "$HFS_CONFIG_DIR"; do
        case "$(hfs_slash "$path")" in
            "$(hfs_slash "$root")"*) return 0 ;;
        esac
    done
    return 1
}

# <file>: read the manifest the installer wrote.
#
# Sets HFS_MANIFEST_SCOPE, HFS_MANIFEST_APP_DIR, HFS_MANIFEST_VERSION and fills
# HFS_MANIFEST_PATHS with every "path=" line, in the order it was written.
hfs_manifest_read() {
    local file="$1" line key value
    HFS_MANIFEST_SCOPE=""
    HFS_MANIFEST_APP_DIR=""
    HFS_MANIFEST_VERSION=""
    HFS_MANIFEST_PATHS=()
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
        esac
    done <"$file"
    return 0
}
