#!/usr/bin/env bash
# HiFiShifter portable uninstall.
#
# [Why the install manifest is the source of truth] This script runs months after the download is
# gone, next to an app folder whose contents may have changed. Guessing a layout is how an
# uninstaller leaves a VST3 the DAW keeps loading; the manifest lists exactly what the installer
# wrote, in the order it wrote it.
#
# Usage:
#   ~/.local/bin/hifishifter-uninstall            ask for confirmation
#   ~/.local/bin/hifishifter-uninstall --yes      unattended
#   ./uninstall-user.sh --prefix=DIR             remove an install whose manifest is gone
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"

APP_DIR=""
ASSUME_YES=0

usage() {
    cat <<'USAGE'
Usage: ./uninstall-user.sh [options]

  --yes          remove without asking for confirmation
  --prefix=DIR   the app folder that was installed (default: the one this script sits in)
  -h, --help     show this text

Settings, projects, audio files and models-user are never touched.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y)   ASSUME_YES=1; shift ;;
        --prefix=*) APP_DIR="${1#*=}"; shift ;;
        --prefix)   [ $# -ge 2 ] || hfs_die "--prefix requires a value"; APP_DIR="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          printf 'error: unknown option: %s (see --help)\n' "$1" >&2; exit 2 ;;
    esac
done

# Installed layout: <app>/installer/uninstall-user.sh. Running it from an extracted bundle is the
# other case, where --prefix is the only way to know what to remove.
if [ -z "$APP_DIR" ]; then
    if [ "$(basename -- "$here")" = "installer" ]; then
        APP_DIR="$(dirname -- "$here")"
    else
        hfs_die "this copy is not installed; pass --prefix=DIR of the installation you want to remove"
    fi
fi
hfs_layout "$APP_DIR"

if ! hfs_manifest_read "$HFS_MANIFEST"; then
    cat >&2 <<EOF
error: no install manifest at $HFS_MANIFEST

The app folder is $HFS_APP_DIR. Either the installation was never finished, or it was
installed by something else. If you are sure, remove that folder by hand.
EOF
    exit 1
fi

printf 'HiFiShifter %s\n\n' "${HFS_MANIFEST_VERSION:-unknown}"
printf 'This removes the app, the plug-ins and the shared model library written by the installer.\n'
printf 'Your settings, projects, audio files and models-user are kept.\n\n'

if [ "$ASSUME_YES" != "1" ]; then
    printf 'Remove HiFiShifter? [y/N] '
    read -r answer || answer=""
    case "$answer" in
        y|Y|yes|YES) ;;
        *) printf 'cancelled.\n'; exit 1 ;;
    esac
fi

remove_path() {
    local path="$1"
    if ! hfs_is_safe_remove "$path"; then
        printf '  skipped (outside the install roots): %s\n' "$path" >&2
        return 0
    fi
    [ -e "$path" ] || [ -L "$path" ] || return 0
    rm -rf -- "$path"
    printf '  removed %s\n' "$path"
}

# Reverse order: files were written parents-first, so children go first.
if [ ${#HFS_MANIFEST_PATHS[@]} -gt 0 ]; then
    for ((index = ${#HFS_MANIFEST_PATHS[@]} - 1; index >= 0; index--)); do
        remove_path "${HFS_MANIFEST_PATHS[index]}"
    done
fi

# [Why rmdir and not rm -rf] These are the directories the installer created. rmdir only removes
# an empty one, so anything the user put there -- or a models-user next to the model library --
# survives even if the manifest claims the directory.
cleanup_dir() {
    local path="$1"
    if ! hfs_is_safe_remove "$path"; then
        return 0
    fi
    [ -d "$path" ] || return 0
    rmdir -- "$path" 2>/dev/null || true
}

# [Why this loop runs forward, unlike the file loop] The installer records these directories
# children-first (installer/, then the app folder; models/, then the data folder), so removing in
# the recorded order empties every parent before its turn comes. Reversing it would try the parent
# while the child still exists, and rmdir would silently skip both.
for ((index = 0; index < ${#HFS_MANIFEST_DIRS[@]}; index++)); do
    cleanup_dir "${HFS_MANIFEST_DIRS[index]}"
done

# The manifest is the last thing to go: until here, an interrupted uninstall can be re-run.
remove_path "$HFS_MANIFEST"

if command -v update-desktop-database >/dev/null 2>&1 && [ -d "$HFS_DESKTOP_DIR" ]; then
    update-desktop-database -q "$HFS_DESKTOP_DIR" || true
fi
if command -v update-mime-database >/dev/null 2>&1 && [ -d "$(dirname -- "$HFS_MIME_DIR")" ]; then
    update-mime-database "$(dirname -- "$HFS_MIME_DIR")" || true
fi

printf '\nHiFiShifter was removed.\n'
printf 'Restart your DAW so it drops HiFiShifter from its plug-in list.\n'
printf 'Your settings stay in %s; delete that folder too if you want them gone.\n' "$HFS_DATA_DIR"
