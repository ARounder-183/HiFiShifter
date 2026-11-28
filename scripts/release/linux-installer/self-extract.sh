#!/bin/sh
# HiFiShifter self-extracting installer.
#
# [Why a self-extracting file instead of "extract the tar.gz first"] The other two platforms
# hand the user one file that opens a wizard: setup.exe, or a .pkg. On Linux the download used to
# be a tar.gz whose README line was "extract it, then run ./install.sh" -- two steps, a terminal,
# and a folder full of files nobody explains. This stub turns the same payload into one
# executable file: double-click it (or run it) and the wizard starts.
#
# [Why the payload is appended after a marker line instead of being embedded in the script]
# Extraction is "start reading at line N and hand the bytes to tar". Nothing is encoded, so the
# file stays the size of the payload, and --extract gives back the plain tree for anyone who
# wants to inspect it or install by hand.
#
#   ./HiFiShifter_...run                 run the wizard
#   ./HiFiShifter_...run --extract=DIR   unpack the payload into DIR and stop
#   ./HiFiShifter_...run --yes           unattended install
#
set -eu

extract_dir=""
for arg in "$@"; do
    case "$arg" in
        --extract=*) extract_dir="${arg#*=}" ;;
    esac
done

marker='__HFS_PAYLOAD_BELOW__'
line="$(awk -v m="$marker" '$0 == m { print NR + 1; exit }' "$0")"
if [ -z "$line" ]; then
    echo "error: this file is not a HiFiShifter installer (payload marker missing)" >&2
    exit 1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/hifishifter-setup.XXXXXX")"
cleanup() { rm -rf -- "$tmp"; }
trap cleanup EXIT INT TERM

# [Why tail and not dd] tail skips whole lines and then copies raw bytes; GNU, busybox and BSD
# tail all behave the same way here, and the payload never has to be decoded.
tail -n +"$line" "$0" | tar -xzf - -C "$tmp" || {
    echo "error: the installer payload is damaged; download the file again" >&2
    exit 1
}

if [ -n "$extract_dir" ]; then
    mkdir -p -- "$extract_dir"
    cp -a "$tmp/." "$extract_dir/"
    printf 'extracted to %s\n' "$extract_dir"
    exit 0
fi

[ -x "$tmp/install.sh" ] || {
    echo "error: the installer payload has no install.sh; download the file again" >&2
    exit 1
}

exec "$tmp/install.sh" "$@"

__HFS_PAYLOAD_BELOW__
