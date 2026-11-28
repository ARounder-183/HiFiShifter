#!/bin/bash
#
# HiFiShifter macOS uninstaller.
#
# [Why this is a package and not a shell script in a README] macOS has no "Add or Remove
# Programs". An app installed by a pkg is removed with pkgutil and rm by whoever knows the
# paths — which is nobody. A product archive wrapping this one script gives the user what
# Windows users already have: a wizard with a welcome page, a list of exactly what will be
# removed, a conclusion, and the elevation needed to write outside /Applications. It is
# published next to the installer and also installed into the payload directory, so the
# person who deleted the download still has it.
#
# [Why there are no choices] The macOS install is one installation: one app, one plug-in
# pair, one shared model store. "Keep the weights for later" would leave gigabytes behind
# that no app and no plug-in can read.
#
# [Why `set -e` is absent] Every step is "remove it if it exists". A missing path is the
# normal case (a user who never installed CLAP), and aborting halfway would leave a machine
# that is neither installed nor clean.
#
# [Why the receipts are forgotten] A leftover receipt makes `pkgutil --pkgs` claim a
# package is still installed, and the next installer treats the machine as an upgrade. This
# script's own receipt cannot be removed from here — Installer writes it after the script
# returns — so the line below clears it on any later run instead. One invisible receipt is
# the price of using Installer at all.
#
# [Why the payload directory goes last] The published uninstaller is *inside* that
# directory; deleting it before the rest of the work is done means deleting the package
# Installer is currently running from. Installer has already expanded it by now, but the
# order costs nothing and removes the question.

set -u

log() { printf '[HiFiShifter uninstall] %s\n' "$*"; }

remove() {
    [ -e "$1" ] || return 0
    if /bin/rm -rf "$1" 2>/dev/null; then
        log "removed $1"
    else
        log "could not remove $1"
    fi
}

# ── The installed product ───────────────────────────────────────────────────
remove "/Applications/HiFiShifter.app"
remove "/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3"
remove "/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap"

# ── Receipts ────────────────────────────────────────────────────────────────
# The first five are the component packages of the current product archive; the last two are
# the single-package installers of earlier releases, forgotten so an upgrade does not leave
# receipts for packages that no longer exist.
for identifier in \
    com.arounder.hifishifter.app \
    com.arounder.hifishifter.models \
    com.arounder.hifishifter.vst3 \
    com.arounder.hifishifter.clap \
    com.arounder.hifishifter.uninstaller \
    com.hifishifter.unified \
    com.hifishifter.plugins ; do
    /usr/sbin/pkgutil --forget "$identifier" >/dev/null 2>&1 || true
done

# ── Per-user data ───────────────────────────────────────────────────────────
# [Why the console user instead of $HOME] Installer runs as root, so $HOME is /var/root. The
# files to clean belong to the person who is logged in, and /dev/console names them.
# SUDO_USER is the CI path (`sudo installer -pkg ...`), where /dev/console belongs to root.
target_user=""
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    target_user="$SUDO_USER"
else
    target_user="$(/usr/bin/stat -f '%Su' /dev/console 2>/dev/null || true)"
fi
case "$target_user" in
    ""|root|loginwindow|_mbsetupuser) target_user="" ;;
esac

if [ -n "$target_user" ]; then
    home="$(/usr/bin/dscl . -read "/Users/$target_user" NFSHomeDirectory 2>/dev/null | /usr/bin/awk '{print $2}')"
    [ -n "${home:-}" ] || home="/Users/$target_user"
    if [ -d "$home" ]; then
        log "cleaning user data under $home"
        # These names belong to backend/hifishifter-kernel/src/app_paths.rs; the static gate
        # check-macos-pkg.py fails if the two lists drift apart.
        for relative in \
            "Library/Application Support/HiFiShifter" \
            "Library/Caches/HiFiShifter" \
            "Library/Logs/HiFiShifter" \
            "Library/Application Support/com.arounder.hifishifter" \
            "Library/Caches/com.arounder.hifishifter" \
            "Library/Logs/com.arounder.hifishifter" \
            "Library/WebKit/com.arounder.hifishifter" \
            "Library/HTTPStorages/com.arounder.hifishifter" \
            "Library/Saved Application State/com.arounder.hifishifter.savedState" ; do
            remove "$home/$relative"
        done
    fi
else
    log "no logged-in user found; per-user caches and settings were left alone"
fi

# ── The shared payload directory (models, licences, this installer's own copy) ──
remove "/Library/Application Support/HiFiShifter"

exit 0
