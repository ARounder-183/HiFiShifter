#!/usr/bin/env bash
# HiFiShifter portable install: everything in your own home folder, no root, no dialogs.
#
# [Why plain text] The previous Linux download was a self-extracting wizard with four dialog
# backends (zenity / whiptail / dialog / CLI). Every backend was a different way to fail, and the
# one WSL hit -- whiptail's missing --dselect -- made the whole installer disappear on the page
# after "components". A text installer that asks once and prints what it does has no such surface.
#
# Usage:
#   ./install-user.sh                     ask for confirmation, install everything
#   ./install-user.sh --yes               unattended
#   ./install-user.sh --prefix=DIR        install the app files under DIR
#   ./install-user.sh --components=app,models,vst3,clap,menu,mime
#   ./install-user.sh --help
#
# Defaults: ~/.local/lib/hifishifter (app), ~/.local/bin (commands), ~/.vst3, ~/.clap,
# $XDG_DATA_HOME/HiFiShifter/models/<version> (shared weights), $XDG_DATA_HOME/applications.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"

# [Why the default comes from the bundle] A portable bundle built without the plug-in formats has
# no HiFiShifter.vst3 to copy; a hard-coded default would stop halfway through the install. The
# bundle ships a "components" file that lists exactly what it contains.
COMPONENTS=""
APP_DIR=""
ASSUME_YES=0

usage() {
    cat <<'USAGE'
Usage: ./install-user.sh [options]

  --yes                 install without asking for confirmation
  --prefix=DIR          app files go to DIR (default: ~/.local/lib/hifishifter)
  --components=LIST     any of app,models,vst3,clap,menu,mime (default: all)
  -h, --help            show this text

Everything else has a fixed, conventional location: the hifishifter command in ~/.local/bin,
the plug-ins in ~/.vst3 and ~/.clap, and the shared model library in
$XDG_DATA_HOME/HiFiShifter/models. Nothing outside your home directory is touched.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y)            ASSUME_YES=1; shift ;;
        --prefix=*)          APP_DIR="${1#*=}"; shift ;;
        --prefix)            [ $# -ge 2 ] || hfs_die "--prefix requires a value"; APP_DIR="$2"; shift 2 ;;
        --components=*)      COMPONENTS="${1#*=}"; shift ;;
        --components)        [ $# -ge 2 ] || hfs_die "--components requires a value"; COMPONENTS="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   printf 'error: unknown option: %s (see --help)\n' "$1" >&2; exit 2 ;;
    esac
done

[ -n "$COMPONENTS" ] || COMPONENTS="$(cat "$here/components" 2>/dev/null || true)"
[ -n "$COMPONENTS" ] || COMPONENTS="app,models,menu,mime"
[ -n "$APP_DIR" ] || APP_DIR="${HOME:-}/.local/lib/hifishifter"
hfs_layout "$APP_DIR"

# [Why every component is validated before anything is written] A typo in --components would
# otherwise install a subset and report success; the same typo in the old wizard printed
# "unknown component" only after the user had already answered every page.
for component in ${COMPONENTS//,/ }; do
    case "$component" in
        app|models|vst3|clap|menu|mime) ;;
        *) hfs_die "unknown component: $component (valid: app, models, vst3, clap, menu, mime)" ;;
    esac
done
has_component() {
    case ",$COMPONENTS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

VERSION="$(cat "$here/version" 2>/dev/null || true)"
[ -n "$VERSION" ] || hfs_die "the bundle has no version file: download it again"
# [Why two versions] The product version names the artifact; the model version is the content hash
# recorded in models.json and is what the app looks up in the shared store. They move independently
# -- an app update does not re-download the weights -- so the bundle carries both.
MODEL_VERSION="$(cat "$here/model-version" 2>/dev/null || true)"
[ -n "$MODEL_VERSION" ] || hfs_die "the bundle has no model-version file: download it again"
MODELS_SRC="$here/models/$MODEL_VERSION"

[ -f "$here/HiFiShifter.AppImage" ] || hfs_die "the bundle has no HiFiShifter.AppImage: download it again"

printf 'HiFiShifter %s\n\n' "$VERSION"
printf '  application   %s\n' "$HFS_APP_DIR"
printf '  commands      %s/hifishifter, %s/hifishifter-uninstall\n' "$HFS_BIN_DIR" "$HFS_BIN_DIR"
if has_component models; then
    printf '  model library %s/%s\n' "$HFS_MODELS_DIR" "$MODEL_VERSION"
fi
if has_component vst3; then printf '  VST3 plug-in  %s/HiFiShifter.vst3\n' "$HFS_VST3_DIR"; fi
if has_component clap; then printf '  CLAP plug-in  %s/HiFiShifter.clap\n' "$HFS_CLAP_DIR"; fi
if has_component menu; then printf '  menu entry    %s/hifishifter.desktop\n' "$HFS_DESKTOP_DIR"; fi
printf '\n'

if [ "$ASSUME_YES" != "1" ]; then
    printf 'Install HiFiShifter into your home folder? [Y/n] '
    read -r answer || answer=""
    case "$answer" in
        ""|y|Y|yes|YES) ;;
        *) printf 'cancelled.\n'; exit 1 ;;
    esac
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/hifishifter-install.XXXXXX")"

# Paths this run created (not ones it overwrote): a failed install removes exactly these, so a
# failure leaves the machine as it was -- and an upgrade that fails keeps the previous version
# whole instead of deleting files the old manifest still lists.
NEW_PATHS=()
cleanup() {
    local status=$?
    if [ "$status" -ne 0 ] && [ ${#NEW_PATHS[@]} -gt 0 ]; then
        printf 'installation failed; removing the files this run created\n' >&2
        for ((index = ${#NEW_PATHS[@]} - 1; index >= 0; index--)); do
            rm -rf -- "${NEW_PATHS[index]}" 2>/dev/null || true
        done
    fi
    rm -rf -- "$WORK"
}
trap cleanup EXIT INT TERM

# [Why the old manifest is read before anything is written] Reinstalling must not accumulate:
# choosing "no CLAP" on the second run would otherwise leave the first run's HiFiShifter.clap
# behind, and the DAW would keep finding a plug-in the user just declined. Only paths this
# installer wrote are compared; models-user, settings and projects are never in a manifest.
OLD_PATHS=()
OLD_DIRS=()
if hfs_manifest_read "$HFS_MANIFEST"; then
    OLD_PATHS=("${HFS_MANIFEST_PATHS[@]}")
    OLD_DIRS=("${HFS_MANIFEST_DIRS[@]}")
fi

HFS_PATHS=()
HFS_DIRS=()

# <src> <dst> <mode>
install_file() {
    local src="$1" dst="$2" mode="$3"
    hfs_is_safe_remove "$dst" || hfs_die "refusing to write outside the install roots: $dst"
    [ -f "$src" ] || hfs_die "missing payload file: $src"
    [ -e "$dst" ] || NEW_PATHS+=("$dst")
    mkdir -p -- "$(dirname -- "$dst")" || return 1
    # [Why rm -f first] cp refuses to copy a file onto itself, and a reinstall from the same
    # bundle directory is exactly that when the app folder is the bundle folder.
    rm -f -- "$dst" || return 1
    cp -f -- "$src" "$dst" || return 1
    chmod "$mode" -- "$dst" || return 1
    HFS_PATHS+=("$dst")
    printf '  installed %s\n' "$dst"
}

# <src-dir> <dst-dir>
install_tree() {
    local src="$1" dst="$2"
    hfs_is_safe_remove "$dst" || hfs_die "refusing to write outside the install roots: $dst"
    [ -d "$src" ] || hfs_die "missing payload folder: $src"
    [ -e "$dst" ] || NEW_PATHS+=("$dst")
    rm -rf -- "$dst" || return 1
    mkdir -p -- "$(dirname -- "$dst")" || return 1
    cp -R -- "$src" "$dst" || return 1
    HFS_PATHS+=("$dst")
    printf '  installed %s\n' "$dst"
}

# <file> <name> <comment> <exec> <terminal> <extra>
write_desktop() {
    local file="$1" name="$2" comment="$3" exec_line="$4" terminal="$5" extra="$6"
    {
        printf '[Desktop Entry]\n'
        printf 'Type=Application\n'
        printf 'Version=1.0\n'
        printf 'Name=%s\n' "$name"
        printf 'Comment=%s\n' "$comment"
        printf 'Exec=%s\n' "$exec_line"
        printf 'Icon=hifishifter\n'
        printf 'Terminal=%s\n' "$terminal"
        printf 'Categories=AudioVideo;Audio;AudioVideoEditing;\n'
        [ -n "$extra" ] && printf '%s\n' "$extra"
    } >"$file"
    chmod 644 "$file"
}

write_mime() {
    cat >"$1" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-hifishifter-project">
    <comment>HiFiShifter project</comment>
    <glob pattern="*.hshp"/>
    <glob pattern="*.hsp"/>
  </mime-type>
</mime-info>
EOF
    chmod 644 "$1"
}

printf 'installing\n'

if has_component app; then
    install_file "$here/HiFiShifter.AppImage" "$HFS_APP_DIR/HiFiShifter.AppImage" 755
    # [Why the launcher and not the AppImage] The launcher checks for libfuse2 and sets
    # APPIMAGE_EXTRACT_AND_RUN when it is missing; pointing the menu entry at the image directly
    # would start the app through a path with no such fallback, and on Ubuntu 24.04 nothing would
    # happen when the user clicks it.
    hfs_write_launcher "$WORK/hifishifter" "$HFS_APP_DIR/HiFiShifter.AppImage"
    install_file "$WORK/hifishifter" "$HFS_BIN_DIR/hifishifter" 755

    # The uninstaller is installed next to the app: the download is long gone by the time the
    # user wants the product removed, and there is no "Add or remove programs" on Linux.
    install_file "$here/install-user.sh" "$HFS_APP_DIR/installer/install-user.sh" 755
    install_file "$here/uninstall-user.sh" "$HFS_APP_DIR/installer/uninstall-user.sh" 755
    install_file "$here/lib.sh" "$HFS_APP_DIR/installer/lib.sh" 644
    {
        printf '#!/bin/sh\n'
        printf '# HiFiShifter uninstaller launcher, generated by the installer.\n'
        printf 'exec "%s" "$@"\n' "$HFS_APP_DIR/installer/uninstall-user.sh"
    } >"$WORK/hifishifter-uninstall"
    chmod 755 "$WORK/hifishifter-uninstall"
    install_file "$WORK/hifishifter-uninstall" "$HFS_BIN_DIR/hifishifter-uninstall" 755
fi

if has_component models; then
    install_tree "$MODELS_SRC" "$HFS_MODELS_DIR/$MODEL_VERSION"
fi

if has_component vst3; then
    install_tree "$here/HiFiShifter.vst3" "$HFS_VST3_DIR/HiFiShifter.vst3"
fi

if has_component clap; then
    install_file "$here/HiFiShifter.clap" "$HFS_CLAP_DIR/HiFiShifter.clap" 755
    install_tree "$here/HiFiShifter-resources" "$HFS_CLAP_DIR/HiFiShifter-resources"
fi

if has_component menu; then
    local_exec="\"$(hfs_desktop_value "$HFS_BIN_DIR/hifishifter")\" %F"
    extra=""
    if has_component mime; then extra="MimeType=application/x-hifishifter-project;"; fi
    write_desktop "$WORK/hifishifter.desktop" "HiFiShifter" \
        "Standalone editor for vocal editing and pitch processing" "$local_exec" false "$extra"
    install_file "$WORK/hifishifter.desktop" "$HFS_DESKTOP_DIR/hifishifter.desktop" 644
    for icon in 32x32 128x128 256x256; do
        [ -f "$here/icons/$icon.png" ] || continue
        install_file "$here/icons/$icon.png" "$HFS_ICON_DIR/$icon/apps/hifishifter.png" 644
    done
fi

if has_component mime; then
    write_mime "$WORK/hifishifter.xml"
    install_file "$WORK/hifishifter.xml" "$HFS_MIME_DIR/hifishifter.xml" 644
fi

# Directories that only exist because of us; the uninstaller removes them with rmdir, so a
# directory that still holds something else simply stays.
if has_component app; then
    HFS_DIRS+=("$HFS_APP_DIR/installer" "$HFS_APP_DIR")
fi
if has_component models; then
    HFS_DIRS+=("$HFS_MODELS_DIR" "$HFS_DATA_DIR")
fi
if has_component menu; then
    for icon in 32x32 128x128 256x256; do
        HFS_DIRS+=("$HFS_ICON_DIR/$icon/apps" "$HFS_ICON_DIR/$icon")
    done
fi

# [Why the manifest is written last] Until every file above exists, "installed" is not true. A
# half-finished install therefore leaves the previous manifest (or none) -- and the uninstaller
# refuses to guess, rather than deleting a directory it cannot recognise.
{
    printf '# HiFiShifter install manifest. Written by install-user.sh; read by uninstall-user.sh.\n'
    printf 'version=%s\n' "$VERSION"
    printf 'scope=user\n'
    printf 'app_dir=%s\n' "$HFS_APP_DIR"
    printf '%s\n' "${HFS_PATHS[@]/#/path=}"
    printf '%s\n' "${HFS_DIRS[@]/#/dir=}"
} >"$WORK/manifest"
install_file "$WORK/manifest" "$HFS_MANIFEST" 644

# Reinstalls must not leave the components the user just declined behind.
for old in "${OLD_PATHS[@]:-}"; do
    [ -n "$old" ] || continue
    keep=0
    for path in "${HFS_PATHS[@]}"; do
        [ "$path" = "$old" ] && { keep=1; break; }
    done
    if [ "$keep" = 0 ] && [ -e "$old" ] && hfs_is_safe_remove "$old"; then
        rm -rf -- "$old"
        printf '  removed %s (no longer selected)\n' "$old"
    fi
done

# A directory the previous install created and this one no longer uses (models were declined,
# the menu entry was dropped) is empty by now, so rmdir can take it -- and leaves it alone if it
# is not.
for old in "${OLD_DIRS[@]:-}"; do
    [ -n "$old" ] || continue
    keep=0
    for dir in "${HFS_DIRS[@]}"; do
        [ "$dir" = "$old" ] && { keep=1; break; }
    done
    if [ "$keep" = 0 ] && [ -d "$old" ] && hfs_is_safe_remove "$old"; then
        rmdir -- "$old" 2>/dev/null || true
    fi
done

# Refresh the per-user caches only when the tools exist; a failure here must not undo a
# successful install.
if has_component menu && command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q "$HFS_DESKTOP_DIR" || true
fi
if has_component mime && command -v update-mime-database >/dev/null 2>&1; then
    update-mime-database "$(dirname -- "$HFS_MIME_DIR")" || true
fi

printf '\nHiFiShifter %s is installed.\n' "$VERSION"
printf 'Start it from the application menu, or run: %s/hifishifter\n' "$HFS_BIN_DIR"
printf 'Uninstall with: %s/hifishifter-uninstall\n' "$HFS_BIN_DIR"
printf 'Your settings, projects and own model files stay in %s; nothing here was removed.\n' "$HFS_DATA_DIR"
