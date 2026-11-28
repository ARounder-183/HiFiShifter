#!/usr/bin/env bash
# One wizard, several dialog backends.
#
# [Why an abstraction instead of "just use zenity"] There is no dialog program every Linux
# desktop ships. GNOME and most Debian derivatives have zenity; a minimal install often has
# whiptail (newt) and nothing else; a headless machine has neither, and an installer that
# refuses to run there cannot be tested in CI either. The wizard therefore speaks one small
# vocabulary (message / confirm / text / radio / checklist / input / directory / progress) and
# every backend below implements exactly that. The pages themselves never branch on a backend.
#
# [Why the backends are chosen in this order] zenity first: it runs without a terminal, which is
# what a double-clicked .run file needs. whiptail/dialog second: they work over SSH and in CI.
# The plain-text backend last: it is the one that always works, and it is what --cli selects.
#
# [Why every function writes its answer to stdout and its chrome to stderr] The caller does
# 'choice=$(ui_radio ...)'. A backend that printed a friendly "you chose X" line to stdout would
# corrupt that value; progress and diagnostics therefore never touch stdout.

HFS_UI_BACKEND=""

# [What a backend can do] 'gui' backends run without a terminal (zenity), 'tui' backends need
# one. The installer uses this to decide whether it must re-exec itself inside a terminal
# emulator when it was started by a double-click.
ui_nature() {
    case "$HFS_UI_BACKEND" in
        zenity) printf '%s' "gui" ;;
        whiptail | dialog) printf '%s' "tui" ;;
        *) printf '%s' "cli" ;;
    esac
}

# [<requested>] : pick a backend. 'auto' is the default; --cli and --gui narrow the choice.
ui_detect_backend() {
    local requested="${1:-auto}"
    case "$requested" in
        zenity | whiptail | dialog | cli)
            HFS_UI_BACKEND="$requested"
            return 0
            ;;
        gui)
            if ui_has_display && command -v zenity >/dev/null 2>&1; then
                HFS_UI_BACKEND=zenity
                return 0
            fi
            return 1
            ;;
        auto) ;;
        *)
            printf 'unknown UI backend: %s\n' "$requested" >&2
            return 1
            ;;
    esac
    if ui_has_display && command -v zenity >/dev/null 2>&1; then
        HFS_UI_BACKEND=zenity
    elif [ -t 0 ] && [ -t 1 ] && command -v whiptail >/dev/null 2>&1; then
        HFS_UI_BACKEND=whiptail
    elif [ -t 0 ] && [ -t 1 ] && command -v dialog >/dev/null 2>&1; then
        HFS_UI_BACKEND=dialog
    else
        HFS_UI_BACKEND=cli
    fi
    return 0
}

ui_has_display() {
    [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]
}

# ── labels <-> tags ────────────────────────────────────────────────────────
#
# [Why the UI works with labels and the caller with tags] whiptail and zenity both display a
# column of text and return what was displayed. Returning "vst3" would show the user an internal
# identifier; returning the translated label means the caller has to translate back. The arrays
# below are that dictionary, filled by ui_parse_items before the dialog opens.

HFS_UI_TAGS=()
HFS_UI_LABELS=()
HFS_UI_DESCS=()

# <tag|label|description>...
ui_parse_items() {
    local item rest
    HFS_UI_TAGS=()
    HFS_UI_LABELS=()
    HFS_UI_DESCS=()
    for item in "$@"; do
        HFS_UI_TAGS+=("${item%%|*}")
        rest="${item#*|}"
        HFS_UI_LABELS+=("${rest%%|*}")
        HFS_UI_DESCS+=("${rest#*|}")
    done
}

# <label>: print the tag that label belongs to (empty when it is not ours).
ui_tag_of_label() {
    local i
    for i in "${!HFS_UI_LABELS[@]}"; do
        if [ "${HFS_UI_LABELS[$i]}" = "$1" ]; then
            printf '%s' "${HFS_UI_TAGS[$i]}"
            return 0
        fi
    done
    return 1
}

# ── messages ───────────────────────────────────────────────────────────────

# <title> <body>
ui_msg() {
    local title="$1" body="$2"
    case "$HFS_UI_BACKEND" in
        zenity) zenity --info --title="$title" --text="$body" --width=560 >/dev/null 2>&1 ;;
        whiptail | dialog) "$HFS_UI_BACKEND" --title "$title" --msgbox "$body" 20 78 ;;
        *) printf '\n%s\n%s\n' "$title" "$body" ;;
    esac
}

# <title> <body>
ui_error() {
    local title="$1" body="$2"
    case "$HFS_UI_BACKEND" in
        zenity) zenity --error --title="$title" --text="$body" --width=560 >/dev/null 2>&1 ;;
        whiptail | dialog) "$HFS_UI_BACKEND" --title "$title" --msgbox "$body" 20 78 ;;
        *) printf '\n%s\n%s\n' "$title" "$body" >&2 ;;
    esac
}

# <title> <body> [<ok-label>]: 0 = the user agreed.
ui_confirm() {
    local title="$1" body="$2" ok_label="${3:-}" answer=""
    case "$HFS_UI_BACKEND" in
        zenity)
            if [ -n "$ok_label" ]; then
                zenity --question --title="$title" --text="$body" --ok-label="$ok_label" --width=560 >/dev/null 2>&1
            else
                zenity --question --title="$title" --text="$body" --width=560 >/dev/null 2>&1
            fi
            ;;
        whiptail | dialog) "$HFS_UI_BACKEND" --title "$title" --yesno "$body" 20 78 ;;
        *)
            [ -n "$title" ] && printf '\n%s\n' "$title" >&2
            printf '%s\n' "$body" >&2
            printf 'Continue? [Y/n] ' >&2
            # EOF means "no": an unattended stdin must never start the app by accident.
            read -r answer || answer=n
            case "$answer" in "" | y | Y | yes | Yes) return 0 ;; *) return 1 ;; esac
            ;;
    esac
}

# <title> <file>: show a scrollable file (the license).
ui_text() {
    local title="$1" file="$2"
    case "$HFS_UI_BACKEND" in
        zenity) zenity --text-info --title="$title" --filename="$file" --width=760 --height=560 >/dev/null 2>&1 ;;
        whiptail | dialog) "$HFS_UI_BACKEND" --title "$title" --scrolltext --textbox "$file" 24 80 ;;
        *) printf '\n%s\n\n' "$title" >&2; sed -n '1,80p' "$file" >&2; printf '\n' >&2 ;;
    esac
}

# ── choice pages ───────────────────────────────────────────────────────────

# <title> <body> <default-tag> <tag|label|desc>...  -> prints the chosen tag
ui_radio() {
    local title="$1" body="$2" default="$3"
    shift 3
    ui_parse_items "$@"
    local i out label answer chosen
    case "$HFS_UI_BACKEND" in
        zenity)
            local -a args=(--list --radiolist --title="$title" --text="$body" --width=680 --height=420 --print-column=2 --column=" " --column=" " --column=" ")
            for i in "${!HFS_UI_TAGS[@]}"; do
                local state=FALSE
                [ "${HFS_UI_TAGS[$i]}" = "$default" ] && state=TRUE
                args+=("$state" "${HFS_UI_LABELS[$i]}" "${HFS_UI_DESCS[$i]}")
            done
            out="$(zenity "${args[@]}" 2>/dev/null)" || return 1
            ui_tag_of_label "$out"
            ;;
        whiptail | dialog)
            local -a args=(--title "$title" --radiolist "$body" 20 78 4)
            for i in "${!HFS_UI_TAGS[@]}"; do
                local state=OFF
                [ "${HFS_UI_TAGS[$i]}" = "$default" ] && state=ON
                args+=("${HFS_UI_LABELS[$i]}" "${HFS_UI_DESCS[$i]}" "$state")
            done
            out="$("$HFS_UI_BACKEND" "${args[@]}" 3>&1 1>&2 2>&3)" || return 1
            out="$(printf '%s' "$out" | tr -d '"')"
            ui_tag_of_label "$out"
            ;;
        *)
            printf '\n%s\n%s\n\n' "$title" "$body" >&2
            for i in "${!HFS_UI_TAGS[@]}"; do
                label="${HFS_UI_LABELS[$i]}"
                [ "${HFS_UI_TAGS[$i]}" = "$default" ] && label="$label (default)"
                printf '  %d) %s - %s\n' "$((i + 1))" "$label" "${HFS_UI_DESCS[$i]}" >&2
            done
            printf 'Choice [1-%d]: ' "${#HFS_UI_TAGS[@]}" >&2
            read -r answer || true
            case "$answer" in
                '') chosen="$default" ;;
                *[!0-9]*) return 1 ;;
                *)
                    if [ "$answer" -ge 1 ] && [ "$answer" -le "${#HFS_UI_TAGS[@]}" ]; then
                        chosen="${HFS_UI_TAGS[$((answer - 1))]}"
                    else
                        return 1
                    fi
                    ;;
            esac
            printf '%s' "$chosen"
            ;;
    esac
}

# <title> <body> <tag|on|label|desc>...  -> prints the selected tags, space separated
ui_check() {
    local title="$1" body="$2"
    shift 2
    # [Why the arrays are built here] A checklist item carries one more field than a radio item
    # (tag|state|label|description). ui_tag_of_label reads the globals, so they still have to be
    # filled -- with the label and the description, not with the state.
    HFS_UI_TAGS=()
    HFS_UI_LABELS=()
    HFS_UI_DESCS=()
    local -a enabled=()
    local item rest i out selected="" word tag
    for item in "$@"; do
        tag="${item%%|*}"
        rest="${item#*|}"
        enabled+=("${rest%%|*}")
        rest="${rest#*|}"
        HFS_UI_TAGS+=("$tag")
        HFS_UI_LABELS+=("${rest%%|*}")
        HFS_UI_DESCS+=("${rest#*|}")
    done
    case "$HFS_UI_BACKEND" in
        zenity)
            local -a args=(--list --checklist --title="$title" --text="$body" --width=740 --height=460 --separator="|" --print-column=2 --column=" " --column=" " --column=" ")
            for i in "${!HFS_UI_TAGS[@]}"; do
                local state=FALSE
                [ "${enabled[$i]}" = "on" ] && state=TRUE
                args+=("$state" "${HFS_UI_LABELS[$i]}" "${HFS_UI_DESCS[$i]}")
            done
            out="$(zenity "${args[@]}" 2>/dev/null)" || return 1
            # [Why a pipe separator] A label contains spaces ("Application menu entry"), so the
            # default space separation cannot be split again. '|' never appears in a translated
            # label -- the catalog gate rejects it in installer strings.
            local saved_ifs="$IFS"
            IFS='|'
            for word in $out; do
                IFS="$saved_ifs"
                [ -n "$word" ] || continue
                tag="$(ui_tag_of_label "$word")" || continue
                selected="$selected $tag"
            done
            IFS="$saved_ifs"
            printf '%s' "${selected# }"
            ;;
        whiptail | dialog)
            local -a args=(--title "$title" --checklist "$body" 22 80 8)
            for i in "${!HFS_UI_TAGS[@]}"; do
                local state=OFF
                [ "${enabled[$i]}" = "on" ] && state=ON
                args+=("${HFS_UI_LABELS[$i]}" "${HFS_UI_DESCS[$i]}" "$state")
            done
            out="$("$HFS_UI_BACKEND" "${args[@]}" 3>&1 1>&2 2>&3)" || return 1
            # [Why eval set --] whiptail quotes every selected item ("VST3" "Application menu
            # entry"); word splitting would turn one label into two. The items are ours, and the
            # catalog gate rejects a double quote inside installer strings.
            eval "set -- $out"
            for word in "$@"; do
                [ -n "$word" ] || continue
                tag="$(ui_tag_of_label "$word")" || continue
                selected="$selected $tag"
            done
            printf '%s' "${selected# }"
            ;;
        *)
            printf '\n%s\n%s\n\n' "$title" "$body" >&2
            for i in "${!HFS_UI_TAGS[@]}"; do
                local mark="[ ]"
                [ "${enabled[$i]}" = "on" ] && mark="[x]"
                printf '  %d) %s %s - %s\n' "$((i + 1))" "$mark" "${HFS_UI_LABELS[$i]}" "${HFS_UI_DESCS[$i]}" >&2
            done
            local answer=""
            printf 'Toggle by number (space separated), "all" or "none" [keep current]: ' >&2
            read -r answer || true
            case "$answer" in
                '') for i in "${!HFS_UI_TAGS[@]}"; do [ "${enabled[$i]}" = "on" ] && selected="$selected ${HFS_UI_TAGS[$i]}"; done ;;
                all | ALL) selected="${HFS_UI_TAGS[*]}" ;;
                none | NONE) selected="" ;;
                *)
                    for word in $answer; do
                        case "$word" in
                            *[!0-9]*) return 1 ;;
                        esac
                        if [ "$word" -ge 1 ] && [ "$word" -le "${#HFS_UI_TAGS[@]}" ]; then
                            selected="$selected ${HFS_UI_TAGS[$((word - 1))]}"
                        else
                            return 1
                        fi
                    done
                    ;;
            esac
            printf '%s' "${selected# }"
            ;;
    esac
}

# <title> <body> <default>  -> prints the entered text
ui_input() {
    local title="$1" body="$2" default="$3" out=""
    case "$HFS_UI_BACKEND" in
        zenity) out="$(zenity --entry --title="$title" --text="$body" --entry-text="$default" --width=640 2>/dev/null)" || return 1 ;;
        whiptail | dialog) out="$("$HFS_UI_BACKEND" --title "$title" --inputbox "$body" 12 78 "$default" 3>&1 1>&2 2>&3)" || return 1 ;;
        *)
            printf '\n%s\n%s\n' "$title" "$body" >&2
            printf '[%s]: ' "$default" >&2
            read -r out || true
            [ -n "$out" ] || out="$default"
            ;;
    esac
    printf '%s' "$out"
}

# <title> <body> <default>  -> prints the chosen directory
ui_dir() {
    local title="$1" body="$2" default="$3" out="" start=""
    case "$HFS_UI_BACKEND" in
        zenity)
            # The default folder usually does not exist yet; zenity refuses a filename that is
            # not on disk, so the page opens at the nearest existing parent instead.
            start="$default"
            while [ ! -d "$start" ] && [ "$start" != "/" ] && [ "$start" != "." ]; do
                start="$(dirname -- "$start")"
            done
            out="$(zenity --file-selection --directory --title="$title" --filename="$start/" 2>/dev/null)" || return 1
            ;;
        whiptail | dialog) out="$("$HFS_UI_BACKEND" --title "$title" --dselect "$default/" 14 78 3>&1 1>&2 2>&3)" || return 1 ;;
        *) ui_input "$title" "$body" "$default"; return $? ;;
    esac
    [ -n "$out" ] || out="$default"
    printf '%s' "$out"
}

# ── progress ───────────────────────────────────────────────────────────────

# <title> <body>: reads "percent<TAB>message" lines on stdin.
ui_progress() {
    local title="$1" body="$2" pct=0 last=-5 message
    case "$HFS_UI_BACKEND" in
        zenity)
            # zenity's progress reads a bare number per line; a line starting with '#' replaces
            # the label. Both are produced from the same two-field input.
            awk -F '\t' '{ print $1; if ($2 != "") { print "# " $2 } }' |
                zenity --progress --title="$title" --text="$body" --auto-close --no-cancel --width=520
            ;;
        whiptail | dialog)
            awk -F '\t' '{ printf "XXX\n%s\n%s\nXXX\n", $1, $2 }' |
                "$HFS_UI_BACKEND" --title "$title" --gauge "$body" 10 74 0 || true
            ;;
        *)
            while IFS=$'\t' read -r pct message; do
                [ -n "$pct" ] || continue
                if [ "$pct" -ge $((last + 5)) ] || [ "$pct" -eq 100 ]; then
                    printf '%3s%% %s\n' "$pct" "$message" >&2
                    last="$pct"
                fi
            done
            ;;
    esac
}

# <title> <body> <command...>: run a command whose stdout is the progress protocol, and show it.
#
# [Why a FIFO and not a pipe] The progress UI has to be running before the worker starts writing,
# and the worker's exit status has to reach the caller (a pipeline would only report the last
# command's status). The FIFO makes the ordering explicit and keeps $? from the worker.
ui_run_with_progress() {
    local title="$1" body="$2"
    shift 2
    local work fifo status
    work="$(mktemp -d "${TMPDIR:-/tmp}/hifishifter-ui.XXXXXX")" || return 1
    fifo="$work/progress"
    mkfifo "$fifo" || { rm -rf "$work"; return 1; }
    ui_progress "$title" "$body" <"$fifo" &
    local ui_pid=$!
    "$@" >"$fifo"
    status=$?
    wait "$ui_pid" 2>/dev/null || true
    rm -rf "$work"
    return "$status"
}
