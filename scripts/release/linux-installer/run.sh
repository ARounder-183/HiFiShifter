#!/usr/bin/env bash
# HiFiShifter Linux installer / uninstaller.
#
#   ./install.sh                     the wizard (GUI when zenity is available, terminal otherwise)
#   ./install.sh --yes               unattended: current user, every component, default folders
#   sudo ./install.sh --system       system-wide install (or --system as a normal user: the
#                                    wizard elevates only the file operations, via pkexec)
#   ./install.sh --uninstall         remove what this installer wrote
#
# [Why this is a script and not a .deb] The product ships as an AppImage plus two plug-in
# formats plus one shared model library. A .deb is all-or-nothing: it cannot offer "only the
# VST3 plug-in" on the choices page, it cannot install for one user without root, and a
# double-clicked .deb depends on a software centre that may not be installed. This script is the
# same shape as the NSIS installer and the macOS pkg: welcome, license, choices, progress,
# conclusion, and an uninstaller that knows exactly what was written.
#
# [Why the wizard never runs as root] Only the file operations are elevated. The user's answers
# are collected by the unprivileged process, written into a plan file, and executed by a second
# instance ("--apply-plan") under pkexec or sudo. That keeps GTK themes, HOME and XDG paths
# correct, avoids a root-owned ~/.config, and means the privileged half has no dialogs to get
# wrong.
#
# [Why the plan is a file instead of a list of cp commands] It is the same object the uninstaller
# reads back (through the manifest): every destination is written down once, so "what gets
# installed" and "what gets removed" cannot drift. It also makes --apply-plan testable on its
# own, which is how CI installs and uninstalls without a desktop.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="${BASH_SOURCE[0]}"
# shellcheck source=paths.sh
. "$HERE/paths.sh"
# shellcheck source=ui.sh
. "$HERE/ui.sh"
# shellcheck source=strings.sh
. "$HERE/strings.sh"

HFS_MODE=install
HFS_SCOPE=""
HFS_SCOPE_SET=0
HFS_UNATTENDED=0
HFS_UI_CHOICE=auto
HFS_APP_DIR_ARG=""
HFS_COMPONENTS=""
HFS_APPLY_PLAN=""
HFS_LAUNCH=1
HFS_LANG_ARG=""
HFS_WORK=""
HFS_VERSION="unknown"
HFS_REMOVE_MODELS=0
HFS_REMOVE_CACHE=0

# ── copy ───────────────────────────────────────────────────────────────────

hfs_detect_language() {
    local tag="${HIFISHIFTER_INSTALL_LANG:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}"
    case "$tag" in
        zh_CN* | zh_Hans* | zh_SG*) printf '%s' "zh-CN" ;;
        zh_TW* | zh_HK* | zh_Hant* | zh_MO*) printf '%s' "zh-TW" ;;
        ja*) printf '%s' "ja-JP" ;;
        ko*) printf '%s' "ko-KR" ;;
        *) printf '%s' "en-US" ;;
    esac
}

hfs_load_strings() {
    HFS_LANG="${HFS_LANG_ARG:-$(hfs_detect_language)}"
    local dict="HFS_I18N_${HFS_LANG//-/_}"
    if ! declare -p "$dict" >/dev/null 2>&1; then
        HFS_LANG="en-US"
        dict="HFS_I18N_en_US"
    fi
    # [Why a nameref] t() is called a hundred times per run; an indirect lookup through a
    # variable name would need eval or a second expansion at every call site.
    declare -gn HFS_STR="$dict"
}

t() {
    printf '%s' "${HFS_STR[$1]:-}"
}

# <key> [<detail>]: show the message and stop. Every failure the user can cause names its reason.
die() {
    local key="$1" detail="${2:-}"
    printf 'installer: %s%s\n' "$key" "${detail:+ (}${detail}${detail:+)}" >&2
    ui_error "HiFiShifter" "$(t "$key")"
    exit 1
}

# ── argument parsing ───────────────────────────────────────────────────────

hfs_usage() {
    cat <<'EOF'
Usage: install.sh [options]

  --user               install for the current user (the default)
  --system             install for everyone; asks for administrator rights
  --yes, -y            unattended: no pages, every component, default folders
  --prefix=DIR         app install folder (default: ~/.local/lib/hifishifter or /opt/hifishifter)
  --components=LIST    comma separated: app,models,vst3,clap,menu,mime
  --cli                never use a graphical dialog
  --gui                require a graphical dialog
  --lang=LOCALE        en-US, zh-CN, zh-TW, ja-JP or ko-KR (default: from LANG)
  --uninstall          remove an installation instead of creating one
  --no-launch          do not offer to start the app when done
  --apply-plan=FILE    internal: execute an installation plan (used with pkexec/sudo)
  --version
  -h, --help
EOF
}

hfs_parse_args() {
    local arg
    while [ $# -gt 0 ]; do
        arg="$1"
        case "$arg" in
            --uninstall) HFS_MODE=uninstall ;;
            --system) HFS_SCOPE=system; HFS_SCOPE_SET=1 ;;
            --user) HFS_SCOPE=user; HFS_SCOPE_SET=1 ;;
            --yes | -y | --unattended) HFS_UNATTENDED=1 ;;
            --cli) HFS_UI_CHOICE=cli ;;
            --gui) HFS_UI_CHOICE=gui ;;
            --no-launch) HFS_LAUNCH=0 ;;
            --lang=*) HFS_LANG_ARG="${arg#*=}" ;;
            --prefix=*) HFS_APP_DIR_ARG="${arg#*=}" ;;
            --components=*) HFS_COMPONENTS="${arg#*=}" ;;
            --ui=*) HFS_UI_CHOICE="${arg#*=}" ;;
            --apply-plan=*) HFS_APPLY_PLAN="${arg#*=}" ;;
            --lang | --prefix | --components | --ui | --apply-plan)
                [ $# -ge 2 ] || { printf 'missing value for %s\n' "$arg" >&2; exit 2; }
                case "$arg" in
                    --lang) HFS_LANG_ARG="$2" ;;
                    --prefix) HFS_APP_DIR_ARG="$2" ;;
                    --components) HFS_COMPONENTS="$2" ;;
                    --ui) HFS_UI_CHOICE="$2" ;;
                    --apply-plan) HFS_APPLY_PLAN="$2" ;;
                esac
                shift
                ;;
            --version)
                if [ -f "$HFS_PAYLOAD/version" ]; then
                    printf 'HiFiShifter installer %s\n' "$(cat "$HFS_PAYLOAD/version")"
                else
                    printf 'HiFiShifter installer\n'
                fi
                exit 0
                ;;
            -h | --help)
                hfs_usage
                exit 0
                ;;
            *)
                printf 'unknown option: %s (see --help)\n' "$arg" >&2
                exit 2
                ;;
        esac
        shift
    done
}

# ── payload ────────────────────────────────────────────────────────────────

# The payload is the folder the wizard was extracted from: it holds the AppImage, the two
# plug-in bundles, models/, icons/ and the licences. Anything missing is reported before a single
# file is written, because a half-installed product is worse than "download it again".
HFS_PAYLOAD="$(cd -- "$HERE/.." && pwd)"

hfs_have_app() { [ -f "$HFS_PAYLOAD/HiFiShifter.AppImage" ]; }
hfs_have_models() {
    local dir
    for dir in "$HFS_PAYLOAD"/models/*/; do
        [ -f "$dir/models.json" ] && return 0
    done
    return 1
}
hfs_have_vst3() { [ -d "$HFS_PAYLOAD/HiFiShifter.vst3" ]; }
hfs_have_clap() { [ -f "$HFS_PAYLOAD/HiFiShifter.clap" ] && [ -d "$HFS_PAYLOAD/HiFiShifter-resources" ]; }

hfs_require_payload() {
    if ! hfs_have_app && ! hfs_have_vst3 && ! hfs_have_clap; then
        die err_payload
    fi
    [ -r "$HFS_PAYLOAD" ] || die err_payload
}

# ── components ─────────────────────────────────────────────────────────────

HFS_DEFAULT_COMPONENTS="app,models,vst3,clap,menu,mime"

hfs_component_on() {
    case " ${HFS_COMPONENTS} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# Keep only the tags this payload can actually install.
hfs_normalize_components() {
    local list="" tag
    for tag in ${HFS_COMPONENTS//,/ }; do
        case "$tag" in
            app) hfs_have_app && list="$list app" ;;
            models) hfs_have_models && list="$list models" ;;
            vst3) hfs_have_vst3 && list="$list vst3" ;;
            clap) hfs_have_clap && list="$list clap" ;;
            menu | mime) list="$list $tag" ;;
            *)
                # A tag that is not in the list is a command-line typo, not a broken payload:
                # say so instead of blaming the download.
                printf 'unknown component: %s (valid: app,models,vst3,clap,menu,mime)\n' "$tag" >&2
                exit 2
                ;;
        esac
    done
    HFS_COMPONENTS="${list# }"
}

# ── plan building ──────────────────────────────────────────────────────────

# [Why the manifest lists paths instead of describing them] The uninstaller must not guess. It
# removes exactly what is written here, in reverse order, and nothing else -- so a future layout
# change cannot leave a stale VST3 behind just because an uninstaller was not updated.
HFS_PLAN_PATHS=()
HFS_PLAN_SIZE=0

hfs_size_of() {
    if [ -d "$1" ]; then
        du -sb -- "$1" 2>/dev/null | cut -f1
    else
        stat -c '%s' -- "$1" 2>/dev/null || printf '0'
    fi
}

plan_header() {
    printf 'scope\t%s\n' "$HFS_SCOPE" >>"$1"
    printf 'app_dir\t%s\n' "$HFS_APP_DIR" >>"$1"
    printf 'version\t%s\n' "$HFS_VERSION" >>"$1"
}

# <plan> <src> <dst> <mode> [<manifest-path>]
plan_copy() {
    printf 'copy\t%s\t%s\t%s\n' "$2" "$3" "$4" >>"$1"
    HFS_PLAN_SIZE=$((HFS_PLAN_SIZE + $(hfs_size_of "$2")))
    [ -n "${5:-}" ] && HFS_PLAN_PATHS+=("$5")
    return 0
}

# <plan> <src-dir> <dst> [<manifest-path>...]
plan_tree() {
    printf 'tree\t%s\t%s\n' "$2" "$3" >>"$1"
    HFS_PLAN_SIZE=$((HFS_PLAN_SIZE + $(hfs_size_of "$2")))
    shift 3
    HFS_PLAN_PATHS+=("$@")
    return 0
}

# <plan> <staged> <dst> <mode> [<manifest-path>]
plan_write() {
    printf 'write\t%s\t%s\t%s\n' "$2" "$3" "$4" >>"$1"
    [ -n "${5:-}" ] && HFS_PLAN_PATHS+=("$5")
    return 0
}

plan_dir() {
    printf 'dir\t%s\n' "$2" >>"$1"
}

# <string>: escape for a Desktop Entry value (backslash, quote, backtick and dollar).
hfs_desktop_value() {
    local text="$1"
    text="${text//\\/\\\\}"
    text="${text//\"/\\\"}"
    printf '%s' "$text"
}

hfs_write_desktop() {
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

hfs_write_mime() {
    local file="$1"
    cat >"$file" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-hifishifter-project">
    <comment>HiFiShifter project</comment>
    <glob pattern="*.hshp"/>
    <glob pattern="*.hsp"/>
  </mime-type>
</mime-info>
EOF
    chmod 644 "$file"
}

# <plan> <work>: every file the install writes, in execution order.
hfs_build_install_plan() {
    local plan="$1" gen="$2/gen" app_exec="" icon old path keep
    mkdir -p "$gen"
    HFS_PLAN_PATHS=()
    HFS_PLAN_SIZE=0

    plan_header "$plan"

    # 1. Directories that must exist even when a component is skipped.
    plan_dir "$plan" "$HFS_APP_DIR"
    plan_dir "$plan" "$HFS_BIN_DIR"
    plan_dir "$plan" "$HFS_DATA_DIR"
    if hfs_component_on models; then plan_dir "$plan" "$HFS_DATA_DIR/models"; fi

    # 2. The app and its launchers.
    if hfs_component_on app; then
        plan_copy "$plan" "$HFS_PAYLOAD/HiFiShifter.AppImage" "$HFS_APP_DIR/HiFiShifter.AppImage" 755 "$HFS_APP_DIR/HiFiShifter.AppImage"
        # [Why the launcher and not the AppImage] The launcher checks for libfuse2 and sets
        # APPIMAGE_EXTRACT_AND_RUN when it is missing. Pointing the menu entry at the image
        # directly would start the app through a path that has no such fallback -- the user
        # clicks "HiFiShifter" and nothing happens on Ubuntu 24.04.
        app_exec="\"$(hfs_desktop_value "$HFS_BIN_DIR/hifishifter")\" %F"
    fi

    # 3. The uninstaller: a copy of this wizard, next to the app, so it survives the download
    #    being deleted (the same reason the macOS pkg carries its own copy). Installed before
    #    the manifest, so even a half-finished install can be removed.
    plan_dir "$plan" "$HFS_APP_DIR/installer"
    plan_copy "$plan" "$HERE/run.sh" "$HFS_APP_DIR/installer/run.sh" 755 "$HFS_APP_DIR/installer/run.sh"
    plan_copy "$plan" "$HERE/ui.sh" "$HFS_APP_DIR/installer/ui.sh" 644 "$HFS_APP_DIR/installer/ui.sh"
    plan_copy "$plan" "$HERE/paths.sh" "$HFS_APP_DIR/installer/paths.sh" 644 "$HFS_APP_DIR/installer/paths.sh"
    plan_copy "$plan" "$HERE/strings.sh" "$HFS_APP_DIR/installer/strings.sh" 644 "$HFS_APP_DIR/installer/strings.sh"
    plan_copy "$plan" "$HERE/uninstall.sh" "$HFS_APP_DIR/uninstall" 755 "$HFS_APP_DIR/uninstall"

    # 4. The shared model library: one copy, replaced wholesale so an old version does not keep
    #    its weights forever. models-user/ lives beside it and is never touched.
    if hfs_component_on models; then
        plan_tree "$plan" "$HFS_PAYLOAD/models" "$HFS_DATA_DIR/models" "$HFS_DATA_DIR/models"
        plan_copy "$plan" "$HFS_PAYLOAD/LICENSE" "$HFS_DATA_DIR/LICENSE" 644 "$HFS_DATA_DIR/LICENSE"
        plan_tree "$plan" "$HFS_PAYLOAD/licenses" "$HFS_DATA_DIR/licenses" "$HFS_DATA_DIR/licenses"
    fi

    # 5. Plug-ins.
    if hfs_component_on vst3; then
        plan_tree "$plan" "$HFS_PAYLOAD/HiFiShifter.vst3" "$HFS_VST3_DIR/HiFiShifter.vst3" "$HFS_VST3_DIR/HiFiShifter.vst3"
    fi
    if hfs_component_on clap; then
        plan_copy "$plan" "$HFS_PAYLOAD/HiFiShifter.clap" "$HFS_CLAP_DIR/HiFiShifter.clap" 755 "$HFS_CLAP_DIR/HiFiShifter.clap"
        plan_tree "$plan" "$HFS_PAYLOAD/HiFiShifter-resources" "$HFS_CLAP_DIR/HiFiShifter-resources" "$HFS_CLAP_DIR/HiFiShifter-resources"
    fi

    # 6. Desktop integration: a menu entry for the app, one for removal, the icon, and the
    #    project association. The removal entry is why a Linux user can find the uninstaller at
    #    all -- there is no "Add or Remove Programs" to open.
    if hfs_component_on menu; then
        if hfs_component_on app; then
            hfs_write_launcher "$gen/hifishifter" "$HFS_APP_DIR/HiFiShifter.AppImage"
            plan_write "$plan" "$gen/hifishifter" "$HFS_BIN_DIR/hifishifter" 755 "$HFS_BIN_DIR/hifishifter"
            hfs_write_desktop "$gen/hifishifter.desktop" "HiFiShifter" "$(t app_comment)" \
                "$app_exec" false "$(hfs_mime_line)"
            plan_write "$plan" "$gen/hifishifter.desktop" "$HFS_DESKTOP_DIR/hifishifter.desktop" 644 "$HFS_DESKTOP_DIR/hifishifter.desktop"
        fi
        hfs_write_desktop "$gen/hifishifter-uninstall.desktop" "$(t uninstall_title)" "$(t uninstall_comment)" \
            "\"$(hfs_desktop_value "$HFS_APP_DIR/uninstall")\"" true ""
        plan_write "$plan" "$gen/hifishifter-uninstall.desktop" "$HFS_DESKTOP_DIR/hifishifter-uninstall.desktop" 644 "$HFS_DESKTOP_DIR/hifishifter-uninstall.desktop"
        for icon in 32x32 128x128 256x256; do
            [ -f "$HFS_PAYLOAD/icons/$icon.png" ] || continue
            plan_dir "$plan" "$HFS_ICON_DIR/$icon/apps"
            plan_copy "$plan" "$HFS_PAYLOAD/icons/$icon.png" "$HFS_ICON_DIR/$icon/apps/hifishifter.png" 644 "$HFS_ICON_DIR/$icon/apps/hifishifter.png"
        done
    fi
    if hfs_component_on mime; then
        hfs_write_mime "$gen/hifishifter.xml"
        plan_write "$plan" "$gen/hifishifter.xml" "$HFS_MIME_DIR/hifishifter.xml" 644 "$HFS_MIME_DIR/hifishifter.xml"
    fi

    # 7. The launcher without a menu entry (an install that skipped shortcuts still needs a
    #    command to start the app), the uninstaller command, and -- last of all -- the manifest:
    #    until every file above exists, "installed" is not true.
    if hfs_component_on app && ! hfs_component_on menu; then
        hfs_write_launcher "$gen/hifishifter" "$HFS_APP_DIR/HiFiShifter.AppImage"
        plan_write "$plan" "$gen/hifishifter" "$HFS_BIN_DIR/hifishifter" 755 "$HFS_BIN_DIR/hifishifter"
    fi
    if hfs_component_on app; then
        {
            printf '#!/bin/sh\n'
            printf '# HiFiShifter uninstaller launcher, generated by the installer.\n'
            printf 'exec "%s" "$@"\n' "$HFS_APP_DIR/uninstall"
        } >"$gen/hifishifter-uninstall"
        chmod 755 "$gen/hifishifter-uninstall"
        plan_write "$plan" "$gen/hifishifter-uninstall" "$HFS_BIN_DIR/hifishifter-uninstall" 755 "$HFS_BIN_DIR/hifishifter-uninstall"
    fi

    # [Why the previous manifest is read] Reinstalling must not accumulate: choosing "no CLAP"
    # on the second run would otherwise leave the first run's HiFiShifter.clap behind, and the
    # DAW would keep finding a plug-in the user just declined. Only payload paths are compared --
    # models-user, settings and projects are never in a manifest.
    if hfs_manifest_read "$HFS_MANIFEST_FILE"; then
        for old in "${HFS_MANIFEST_PATHS[@]}"; do
            [ -n "$old" ] || continue
            keep=0
            for path in "${HFS_PLAN_PATHS[@]}"; do
                if [ "$path" = "$old" ]; then
                    keep=1
                    break
                fi
            done
            [ "$keep" = 1 ] || printf 'remove\t%s\n' "$old" >>"$plan"
        done
    fi

    {
        printf '# HiFiShifter install manifest. Written by the installer; read by the uninstaller.\n'
        printf 'version=%s\n' "$HFS_VERSION"
        printf 'scope=%s\n' "$HFS_SCOPE"
        printf 'app_dir=%s\n' "$HFS_APP_DIR"
        printf '%s\n' "${HFS_PLAN_PATHS[@]/#/path=}"
    } >"$gen/manifest"
    printf 'write\t%s\t%s\t644\n' "$gen/manifest" "$HFS_MANIFEST_FILE" >>"$plan"
    printf 'total\t%s\n' "$HFS_PLAN_SIZE" >>"$plan"
}

hfs_mime_line() {
    if hfs_component_on mime; then
        printf 'MimeType=application/x-hifishifter-project;'
    fi
}

# ── plan execution ─────────────────────────────────────────────────────────

hfs_emit_progress() {
    local total="$1" done="$2" label="$3" pct=0
    if [ "$total" -gt 0 ]; then pct=$((done * 100 / total)); fi
    [ "$pct" -gt 99 ] && pct=99
    printf '%s\t%s\n' "$pct" "$(basename -- "$label")"
}

# [Why every destination is checked] The plan is a file, and a file can be edited. The
# privileged half therefore refuses any path that is not under a root this product owns (the
# same predicate the uninstaller uses), so a tampered plan cannot turn into "rm -rf HOME".
hfs_do_copy_file() {
    local src="$1" dst="$2" mode="${3:-0644}"
    hfs_is_safe_remove "$dst" || { printf 'refusing to write outside the install roots: %s\n' "$dst" >&2; return 1; }
    [ -f "$src" ] || { printf 'missing source file: %s\n' "$src" >&2; return 1; }
    mkdir -p -- "$(dirname -- "$dst")" || return 1
    # [Why rm -f first] cp refuses to copy a file onto itself, and a reinstall from the same
    # payload path is exactly that when the app folder is the payload folder.
    rm -f -- "$dst" || return 1
    cp -f -- "$src" "$dst" || return 1
    chmod "$mode" -- "$dst" || return 1
}

hfs_do_copy_tree() {
    local src="$1" dst="$2"
    hfs_is_safe_remove "$dst" || { printf 'refusing to write outside the install roots: %s\n' "$dst" >&2; return 1; }
    [ -d "$src" ] || { printf 'missing source folder: %s\n' "$src" >&2; return 1; }
    rm -rf -- "$dst" || return 1
    mkdir -p -- "$(dirname -- "$dst")" || return 1
    cp -R -- "$src" "$dst" || return 1
}

hfs_do_remove() {
    local path="$1"
    if ! hfs_is_safe_remove "$path"; then
        printf 'refusing to remove outside the install roots: %s\n' "$path" >&2
        return 0
    fi
    [ -e "$path" ] || [ -L "$path" ] || return 0
    rm -rf -- "$path" || return 1
}

# <plan>: execute it, printing "percent<TAB>label" on stdout.
hfs_apply_plan() {
    local plan="$1" total=0 done=0 kind a b c
    [ -f "$plan" ] || { printf 'missing plan: %s\n' "$plan" >&2; return 1; }
    total="$(awk -F '\t' '$1 == "total" { print $2; exit }' "$plan")"
    [ -n "$total" ] || total=0

    while IFS=$'\t' read -r kind a b c; do
        case "$kind" in
            '' | '#'* | total | scope | app_dir | version) continue ;;
            dir) mkdir -p -- "$a" || return 1; continue ;;
            copy) hfs_do_copy_file "$a" "$b" "$c" || return 1; done=$((done + $(hfs_size_of "$a"))) ;;
            tree) hfs_do_copy_tree "$a" "$b" || return 1; done=$((done + $(hfs_size_of "$a"))) ;;
            write) hfs_do_copy_file "$a" "$b" "$c" || return 1 ;;
            remove) hfs_do_remove "$a" || return 1; done=$((done + 1)) ;;
            *)
                printf 'unknown plan action: %s\n' "$kind" >&2
                return 1
                ;;
        esac
        hfs_emit_progress "$total" "$done" "$b"
    done <"$plan"
    printf '100\t%s\n' "$HFS_VERSION"
}

# ── elevation ──────────────────────────────────────────────────────────────

# <plan>: run the plan with the rights it needs, keeping the user's answers and HOME.
hfs_apply_plan_privileged() {
    local plan="$1"
    if [ "$HFS_SCOPE" != system ] || [ "$(id -u)" = 0 ]; then
        "$SELF" --apply-plan="$plan"
        return $?
    fi
    if command -v pkexec >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        pkexec env DISPLAY="${DISPLAY:-}" XAUTHORITY="${XAUTHORITY:-}" WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
            "$SELF" --apply-plan="$plan"
        return $?
    fi
    if [ -t 0 ] && command -v sudo >/dev/null 2>&1; then
        sudo "$SELF" --apply-plan="$plan"
        return $?
    fi
    return 126
}

# ── pages ──────────────────────────────────────────────────────────────────

hfs_page_welcome() {
    [ "$HFS_UNATTENDED" = 1 ] && return 0
    ui_msg "$(t welcome_title)" "$(t welcome_body)"
}

hfs_page_license() {
    local file="$HFS_PAYLOAD/LICENSE"
    [ "$HFS_UNATTENDED" = 1 ] && return 0
    [ -f "$file" ] || return 0
    ui_text "$(t license_title)" "$file"
    if ! ui_confirm "$(t license_title)" "$(t license_intro)" "$(t license_accept)"; then
        ui_error "$(t license_title)" "$(t license_decline)"
        exit 1
    fi
}

hfs_page_scope() {
    [ "$HFS_SCOPE_SET" = 1 ] && return 0
    if [ "$HFS_UNATTENDED" = 1 ]; then
        HFS_SCOPE=user
        return 0
    fi
    local choice
    choice="$(ui_radio "$(t scope_title)" "$(t scope_body)" user \
        "user|$(t scope_user)|$(t scope_user_desc)" \
        "system|$(t scope_system)|$(t scope_system_desc)")" || exit 1
    HFS_SCOPE="${choice:-user}"
}

hfs_page_components() {
    if [ -n "$HFS_COMPONENTS" ]; then
        hfs_normalize_components
        return 0
    fi
    HFS_COMPONENTS="$HFS_DEFAULT_COMPONENTS"
    if [ "$HFS_UNATTENDED" = 1 ]; then
        hfs_normalize_components
        return 0
    fi
    # [Why the app and the models are not offered] The app is the product, and the model library
    # has no second source: without it the app and every plug-in fail at the first inference.
    # A checkbox that must stay checked is a lie with extra steps.
    local selected
    selected="$(ui_check "$(t components_title)" "$(t components_body)" \
        "vst3|on|$(t component_vst3)|$(t desc_vst3)" \
        "clap|on|$(t component_clap)|$(t desc_clap)" \
        "menu|on|$(t component_menu)|$(t desc_menu)" \
        "mime|on|$(t component_mime)|$(t desc_mime)")" || exit 1
    HFS_COMPONENTS="app models $selected"
    hfs_normalize_components
}

hfs_page_app_dir() {
    hfs_component_on app || return 0
    [ "$HFS_UNATTENDED" = 1 ] && return 0
    [ -n "$HFS_APP_DIR_ARG" ] && return 0
    local dir
    dir="$(ui_dir "$(t app_dir_title)" "$(t app_dir_body)" "$(hfs_default_app_dir "$HFS_SCOPE")")" || exit 1
    [ -n "$dir" ] && HFS_APP_DIR_ARG="$dir"
    return 0
}

# ── install ────────────────────────────────────────────────────────────────

hfs_read_version() {
    if [ -f "$HFS_PAYLOAD/version" ]; then
        HFS_VERSION="$(cat "$HFS_PAYLOAD/version")"
    fi
}

hfs_complete_install() {
    # Refresh the caches the desktop and the MIME database keep; a missing tool is normal on a
    # minimal system and must not fail an otherwise complete installation.
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$HFS_DESKTOP_DIR" >/dev/null 2>&1 || true
    fi
    if command -v update-mime-database >/dev/null 2>&1 && hfs_component_on mime; then
        update-mime-database "$(dirname -- "$HFS_MIME_DIR")" >/dev/null 2>&1 || true
    fi
    if command -v gtk-update-icon-cache >/dev/null 2>&1 && [ -d "$HFS_ICON_DIR" ]; then
        gtk-update-icon-cache -q -t -f "$HFS_ICON_DIR" >/dev/null 2>&1 || true
    fi
}

install_main() {
    hfs_require_payload
    hfs_read_version

    # [Why the app must be closed] Replacing the model library under a running app is not a
    # crash on Linux (files are replaced by rename), but the session in memory would keep
    # reading the new weights with the old manifest -- a state nobody can debug later.
    if command -v pgrep >/dev/null 2>&1 && pgrep -x HiFiShifter >/dev/null 2>&1; then
        die err_app_running
    fi

    hfs_page_welcome
    hfs_page_license
    hfs_page_scope

    if [ "$HFS_SCOPE" = user ] && [ "$(id -u)" = 0 ]; then
        die err_root_scope
    fi

    hfs_layout "$HFS_SCOPE" "$HFS_APP_DIR_ARG"
    hfs_page_components
    hfs_page_app_dir
    hfs_layout "$HFS_SCOPE" "$HFS_APP_DIR_ARG"

    HFS_WORK="$(mktemp -d "${TMPDIR:-/tmp}/hifishifter-install.XXXXXX")" || die err_write
    trap 'rm -rf -- "$HFS_WORK"' EXIT

    local plan="$HFS_WORK/plan"
    hfs_build_install_plan "$plan" "$HFS_WORK"

    local rc
    ui_run_with_progress "$(t installing)" "" hfs_apply_plan_privileged "$plan"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        if [ "$rc" = 126 ] || [ "$rc" = 127 ]; then
            die err_elevation "exit $rc"
        fi
        die err_write "exit $rc"
    fi

    hfs_complete_install

    if [ "$HFS_UNATTENDED" = 0 ]; then
        ui_msg "$(t finish_title)" "$(t finish_body)"
        if [ "$HFS_LAUNCH" = 1 ] && hfs_component_on app; then
            if ui_confirm "" "$(t finish_launch)"; then
                nohup "$HFS_BIN_DIR/hifishifter" >/dev/null 2>&1 &
            fi
        fi
    fi
    return 0
}

# ── uninstall ──────────────────────────────────────────────────────────────

# The uninstaller lives in <app>/installer/run.sh, so its own location answers "which install",
# and the manifest answers "which files". A missing manifest is not fatal: the fixed names are
# still removable, which is exactly the situation after somebody deleted the data folder by hand.
hfs_resolve_installed_layout() {
    local app_dir="${HFS_APP_DIR_ARG:-}" scope requested="${HFS_SCOPE:-}"
    if [ -z "$app_dir" ]; then
        app_dir="$(cd -- "$HERE/.." && pwd)"
    fi
    # [Why both scopes are probed] "uninstall" may be started from the extracted payload, from
    # the installed copy, or from hifishifter-uninstall in PATH. Only the manifest knows which
    # install is meant; its two possible homes are the two data roots, and reading the wrong one
    # is harmless (the file is simply absent).
    for scope in user system; do
        [ -n "$requested" ] && [ "$requested" != "$scope" ] && continue
        hfs_layout "$scope" "$app_dir"
        if hfs_manifest_read "$HFS_MANIFEST_FILE"; then
            [ -n "$HFS_MANIFEST_SCOPE" ] && scope="$HFS_MANIFEST_SCOPE"
            [ -n "$HFS_MANIFEST_APP_DIR" ] && app_dir="$HFS_MANIFEST_APP_DIR"
            hfs_layout "$scope" "$app_dir"
            return 0
        fi
    done
    # [Why this guard] Without it, running --uninstall on an extracted payload would treat the
    # extraction folder as an installation and delete product-named files out of it. The
    # installed copy always carries "uninstall" beside installer/; the payload does not.
    if [ ! -e "$app_dir/uninstall" ]; then
        printf 'no HiFiShifter installation found (looked for %s)\n' "$app_dir" >&2
        return 2
    fi
    case "$app_dir" in /opt/*) HFS_SCOPE=system ;; *) HFS_SCOPE=user ;; esac
    hfs_layout "$HFS_SCOPE" "$app_dir"
    return 1
}

hfs_build_uninstall_plan() {
    local plan="$1" path i
    local -a removes=()
    printf 'scope\t%s\n' "$HFS_SCOPE" >>"$plan"
    printf 'app_dir\t%s\n' "$HFS_APP_DIR" >>"$plan"
    printf 'version\t%s\n' "${HFS_MANIFEST_VERSION:-$HFS_VERSION}" >>"$plan"

    # [Why reverse order] The manifest lists files in install order; removing them backwards
    # takes the leaves first, so an interrupted uninstall never leaves a half-empty tree that the
    # next run cannot recognise.
    if [ "${#HFS_MANIFEST_PATHS[@]}" -gt 0 ]; then
        for ((i = ${#HFS_MANIFEST_PATHS[@]} - 1; i >= 0; i--)); do
            path="${HFS_MANIFEST_PATHS[$i]}"
            [ "$path" = "$HFS_MANIFEST_FILE" ] && continue
            removes+=("$path")
        done
    else
        removes+=(
            "$HFS_APP_DIR/HiFiShifter.AppImage"
            "$HFS_APP_DIR/uninstall"
            "$HFS_BIN_DIR/hifishifter"
            "$HFS_BIN_DIR/hifishifter-uninstall"
            "$HFS_DESKTOP_DIR/hifishifter.desktop"
            "$HFS_DESKTOP_DIR/hifishifter-uninstall.desktop"
            "$HFS_MIME_DIR/hifishifter.xml"
            "$HFS_VST3_DIR/HiFiShifter.vst3"
            "$HFS_CLAP_DIR/HiFiShifter.clap"
            "$HFS_CLAP_DIR/HiFiShifter-resources"
        )
    fi

    if [ "$HFS_REMOVE_MODELS" = 1 ]; then
        removes+=("$HFS_DATA_DIR/models" "$HFS_DATA_DIR/LICENSE" "$HFS_DATA_DIR/licenses")
    fi
    if [ "$HFS_REMOVE_CACHE" = 1 ]; then
        removes+=("$HFS_CACHE_DIR" "$HFS_STATE_DIR" "$(hfs_webview_cache_dir)")
    fi

    # The installer files are listed one by one, but their folder is not -- and an empty
    # installer/ left behind is exactly what makes the next check claim "still installed".
    removes+=("$HFS_APP_DIR/installer")

    # The manifest goes last: it is what the list above was read from.
    removes+=("$HFS_MANIFEST_FILE")

    for path in "${removes[@]}"; do
        printf 'remove\t%s\n' "$path" >>"$plan"
    done
    # The progress denominator is the number of removals: bytes say nothing about how long it
    # takes to unlink a tree, and a gauge stuck at 0% is indistinguishable from a hang.
    printf 'total\t%s\n' "${#removes[@]}" >>"$plan"
}

uninstall_main() {
    hfs_resolve_installed_layout
    case $? in
        0) ;;
        2) exit 1 ;;
        *) : ;;
    esac
    if [ "$HFS_SCOPE" = user ] && [ "$(id -u)" = 0 ]; then
        die err_root_scope
    fi

    if [ "$HFS_UNATTENDED" = 0 ]; then
        ui_msg "$(t uninstall_title)" "$(t uninstall_body)"
        local selected
        selected="$(ui_check "$(t uninstall_options_title)" "$(t uninstall_options_body)" \
            "models|on|$(t uninstall_opt_models)|$(t uninstall_opt_models_desc)" \
            "cache|off|$(t uninstall_opt_cache)|$(t uninstall_opt_cache_desc)")" || exit 1
        case " $selected " in *" models "*) HFS_REMOVE_MODELS=1 ;; esac
        case " $selected " in *" cache "*) HFS_REMOVE_CACHE=1 ;; esac
    else
        HFS_REMOVE_MODELS=1
        HFS_REMOVE_CACHE=0
    fi

    HFS_WORK="$(mktemp -d "${TMPDIR:-/tmp}/hifishifter-uninstall.XXXXXX")" || die err_write
    trap 'rm -rf -- "$HFS_WORK"' EXIT
    local plan="$HFS_WORK/plan"
    hfs_build_uninstall_plan "$plan"

    local rc
    ui_run_with_progress "$(t uninstall_progress)" "" hfs_apply_plan_privileged "$plan"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        if [ "$rc" = 126 ] || [ "$rc" = 127 ]; then
            die err_elevation "exit $rc"
        fi
        die err_write "exit $rc"
    fi

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$HFS_DESKTOP_DIR" >/dev/null 2>&1 || true
    fi
    if command -v update-mime-database >/dev/null 2>&1; then
        update-mime-database "$(dirname -- "$HFS_MIME_DIR")" >/dev/null 2>&1 || true
    fi
    if command -v gtk-update-icon-cache >/dev/null 2>&1 && [ -d "$HFS_ICON_DIR" ]; then
        gtk-update-icon-cache -q -t -f "$HFS_ICON_DIR" >/dev/null 2>&1 || true
    fi

    # Empty parents are ours only when nothing else lives there; rmdir refuses otherwise.
    for path in "$HFS_ICON_DIR/32x32/apps" "$HFS_ICON_DIR/128x128/apps" "$HFS_ICON_DIR/256x256/apps" \
        "$HFS_ICON_DIR/32x32" "$HFS_ICON_DIR/128x128" "$HFS_ICON_DIR/256x256" \
        "$HFS_VST3_DIR" "$HFS_CLAP_DIR" "$HFS_APP_DIR" "$HFS_DATA_DIR"; do
        [ -d "$path" ] && rmdir --ignore-fail-on-non-empty -- "$path" 2>/dev/null
    done

    if [ "$HFS_UNATTENDED" = 0 ]; then
        ui_msg "$(t uninstall_done_title)" "$(t uninstall_done_body)
$(t kept_note)"
    fi
    return 0
}

# ── entry point ────────────────────────────────────────────────────────────

# [Why the terminal re-exec] A .run file can be double-clicked in a file manager that has no
# "run in terminal" option. With no display and no tty, whiptail cannot draw anything; opening a
# terminal first turns a silent no-op into the wizard the user expected.
hfs_maybe_open_terminal() {
    [ "$HFS_UNATTENDED" = 0 ] || return 0
    # --cli/--gui are explicit answers: never second-guess them.
    [ "$HFS_UI_CHOICE" = auto ] || return 0
    [ -t 0 ] && return 0
    [ -n "${HFS_IN_TERMINAL:-}" ] && return 0
    # [Why a display is required] Without one there is nowhere to open a terminal; trying would
    # hang on a launcher that cannot connect (WSL and headless CI both hit this).
    [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || return 0
    # zenity runs without a terminal, so there is nothing to open.
    command -v zenity >/dev/null 2>&1 && return 0
    local term
    for term in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal xterm; do
        command -v "$term" >/dev/null 2>&1 || continue
        case "$term" in
            gnome-terminal) exec env HFS_IN_TERMINAL=1 "$term" -- "$SELF" "$@" ;;
            *) exec env HFS_IN_TERMINAL=1 "$term" -e "$SELF" "$@" ;;
        esac
    done
    return 0
}

main() {
    hfs_parse_args "$@"
    hfs_load_strings
    ui_detect_backend "$HFS_UI_CHOICE" || die err_ui

    if [ -n "$HFS_APPLY_PLAN" ]; then
        local scope app_dir planned_version
        scope="$(awk -F '\t' '$1 == "scope" { print $2; exit }' "$HFS_APPLY_PLAN")"
        app_dir="$(awk -F '\t' '$1 == "app_dir" { print $2; exit }' "$HFS_APPLY_PLAN")"
        planned_version="$(awk -F '\t' '$1 == "version" { print $2; exit }' "$HFS_APPLY_PLAN")"
        [ -n "$planned_version" ] && HFS_VERSION="$planned_version"
        hfs_layout "${scope:-user}" "$app_dir"
        hfs_apply_plan "$HFS_APPLY_PLAN"
        exit $?
    fi

    hfs_maybe_open_terminal "$@"

    case "$HFS_MODE" in
        uninstall) uninstall_main ;;
        *) install_main ;;
    esac
}

main "$@"
