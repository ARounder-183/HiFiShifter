#!/usr/bin/env bash
# HiFiShifter Linux install script.
#
#   ./install.sh               current user: ~/.local (AppImage + ~/.vst3 + ~/.clap + XDG model store)
#   sudo ./install.sh --system system-wide: /opt/hifishifter + /usr/lib/vst3 + /usr/lib/clap
#                              + /usr/local/share/HiFiShifter model store
#
# The all-in-one package contains HiFiShifter.AppImage, VST3, CLAP and models/ at the same
# time; the plugin-only package has just the latter two, and the script skips entries that
# are not present. It only touches its own names and never modifies other plugins.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
system=0

for arg in "$@"; do
    case "$arg" in
        --system) system=1 ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

if [ "$system" = "1" ]; then
    if [ "$(id -u)" != "0" ]; then
        printf -- '--system requires root: sudo ./install.sh --system\n' >&2
        exit 1
    fi
    app_dir="/opt/hifishifter"
    bin_dir="/usr/local/bin"
    data_root="/usr/local/share/HiFiShifter"
    vst3_dir="/usr/lib/vst3"
    clap_dir="/usr/lib/clap"
else
    app_dir="$HOME/.local/lib/hifishifter"
    bin_dir="$HOME/.local/bin"
    data_root="${XDG_DATA_HOME:-$HOME/.local/share}/HiFiShifter"
    vst3_dir="$HOME/.vst3"
    clap_dir="$HOME/.clap"
fi

# ── Standalone app ──────────────────────────────────────────────────────────
if [ -f "$here/HiFiShifter.AppImage" ]; then
    mkdir -p "$app_dir" "$bin_dir"
    cp -f "$here/HiFiShifter.AppImage" "$app_dir/HiFiShifter.AppImage"
    chmod 755 "$app_dir/HiFiShifter.AppImage"
    # A thin wrapper instead of a direct symlink: some AppImage runner versions treat
    # argv[0] as the image path and resolve to the wrong location when launched through a
    # symlink.
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$app_dir/HiFiShifter.AppImage" >"$bin_dir/hifishifter"
    chmod 755 "$bin_dir/hifishifter"
    printf 'AppImage → %s/HiFiShifter.AppImage (launch command: %s/hifishifter)\n' "$app_dir" "$bin_dir"
fi

# ── Shared model store ──────────────────────────────────────────────────────
if [ -d "$here/models" ]; then
    # This directory is owned by the installer; replace it wholesale so old-version weights
    # do not keep occupying disk forever after upgrades. Running plugins only read the
    # models; please quit the host before upgrading.
    rm -rf "$data_root/models"
    mkdir -p "$data_root/models"
    for version_dir in "$here/models"/*; do
        [ -d "$version_dir" ] || continue
        version="$(basename "$version_dir")"
        cp -R "$version_dir" "$data_root/models/$version"
        printf 'model store → %s/models/%s\n' "$data_root" "$version"
    done

    # The model store's license is placed next to the shared directory together with the
    # root LICENSE, so it remains findable after uninstalling the plugin.
    [ -f "$here/LICENSE" ] && cp -f "$here/LICENSE" "$data_root/LICENSE"
    if [ -d "$here/licenses" ]; then
        rm -rf "$data_root/licenses"
        cp -R "$here/licenses" "$data_root/licenses"
    fi
fi

# ── VST3 ────────────────────────────────────────────────────────────────────
if [ -d "$here/HiFiShifter.vst3" ]; then
    mkdir -p "$vst3_dir"
    rm -rf "$vst3_dir/HiFiShifter.vst3"
    cp -R "$here/HiFiShifter.vst3" "$vst3_dir/"
    printf 'VST3 → %s/HiFiShifter.vst3\n' "$vst3_dir"
fi

# ── CLAP ────────────────────────────────────────────────────────────────────
if [ -f "$here/HiFiShifter.clap" ]; then
    mkdir -p "$clap_dir"
    rm -f "$clap_dir/HiFiShifter.clap"
    rm -rf "$clap_dir/HiFiShifter-resources"
    cp "$here/HiFiShifter.clap" "$clap_dir/"
    cp -R "$here/HiFiShifter-resources" "$clap_dir/"
    printf 'CLAP → %s/HiFiShifter.clap\n' "$clap_dir"
fi

printf '\nDone. Restart your host (or rescan plugins inside it).\n'
