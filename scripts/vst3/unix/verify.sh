#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter VST3 bundle self-check (macOS / Linux).
#
#     scripts/vst3/unix/verify.sh <path/to/HiFiShifter.vst3>
#
# Criteria (any failure exits non-zero). Design principle: **trust external evidence only** —
# the symbol table, the dynamic section, a real dlopen, a real handshake with the host, and
# "the frontend artifacts really are in the package", not "the build script said it was fine".
#
#   1 Layout  2 Exported symbols  3 Dependency closure  4 dlopen black-box load  5 macOS bundle metadata
#   6 Packaging hygiene  7 Frontend artifacts  8 Host handshake (native extension + IID layout)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# [Why convert to an absolute path first] section 3 uses "$BUNDLE"/* to decide "is the
# dependency inside the package", while ldd reports absolute paths. With a relative path the
# prefix never matches, so two in-package dependencies get reported as leaks — running it
# locally as the README shows turns red, while CI stays green only because it passes an
# absolute path.
readonly BUNDLE="$(cd -- "${1:?usage: verify.sh <path/to/HiFiShifter.vst3>}" && pwd)"

FAILED=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$*"; FAILED=1; }
note() { printf "    %s\n" "$*"; }

# If the plugin entry deadlocks (dlopen / GetPluginFactory / host handshake) the probe never
# returns and the whole packaging pipeline hangs until job timeout. The Windows verify.ps1
# states the requirement explicitly: "a deadlock must fail here, not hang CI"; the Unix side
# keeps the same discipline with coreutils timeout.
#
# [Why the python3 fallback] macOS runners have no GNU coreutils, so a `timeout` command does
# not even exist (the vst3-unix.yml test step was fixed for this once). Without a fallback
# rc=127, the probe is treated as "load failed", and the log is empty — the package was fine
# yet the pipeline went red. python3 is available on both CI legs (explicitly installed on
# Linux, preinstalled on macOS): kill the fresh process group after a time limit, with exit
# semantics aligned with GNU timeout (timeout 124, child killed by signal 128+n).
readonly PROBE_TIMEOUT_SEC="${HFS_PROBE_TIMEOUT_SEC:-30}"
run_probe() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$PROBE_TIMEOUT_SEC" "$@"
        local rc=$?
        if [ "$rc" -eq 124 ]; then
            bad "probe did not return within $PROBE_TIMEOUT_SEC seconds (plugin entry likely deadlocked)"
            return 1
        fi
        return "$rc"
    fi
    python3 -c '
import os, signal, subprocess, sys
sec = int(sys.argv[1])
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    rc = p.wait(timeout=sec)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    sys.exit(124)
sys.exit(128 - rc if rc < 0 else rc)
' "$PROBE_TIMEOUT_SEC" "$@"
    local rc=$?
    if [ "$rc" -eq 124 ]; then
        bad "probe did not return within $PROBE_TIMEOUT_SEC seconds (plugin entry likely deadlocked)"
        return 1
    fi
    return "$rc"
}

[ -d "$BUNDLE" ] || { printf "bundle does not exist: %s\n" "$BUNDLE" >&2; exit 2; }

case "$(uname -s)" in
    Linux)  HOST_OS="linux" ;;
    Darwin) HOST_OS="macos" ;;
    *)      printf "unsupported platform\n" >&2; exit 2 ;;
esac
case "$(uname -m)" in
    x86_64|amd64) HOST_ARCH="x86_64" ;;
    arm64|aarch64) HOST_ARCH="arm64" ;;
    *) printf "unsupported architecture: %s\n" "$(uname -m)" >&2; exit 2 ;;
esac

if [ "$HOST_OS" = "linux" ]; then
    [ "$HOST_ARCH" = "arm64" ] && PLATFORM_ARCH="aarch64" || PLATFORM_ARCH="x86_64"
    MODULE="$BUNDLE/Contents/${PLATFORM_ARCH}-linux/HiFiShifter.so"
    LIB_DIR="$(dirname "$MODULE")"
    REQUIRED_SYMBOLS="GetPluginFactory ModuleEntry ModuleExit"
else
    MODULE="$BUNDLE/Contents/MacOS/HiFiShifter"
    LIB_DIR="$BUNDLE/Contents/Frameworks"
    REQUIRED_SYMBOLS="GetPluginFactory bundleEntry bundleExit"
fi

printf "\n[vst3:verify] %s\n" "$BUNDLE"

# ── 1. Layout ────────────────────────────────────────────────────────────────
printf "\n1) Layout\n"
if [ -f "$MODULE" ]; then
    ok "module at the canonical path: ${MODULE#"$BUNDLE"/}"
else
    bad "module not found: $MODULE"
    printf "\nLayout is wrong; the remaining checks are meaningless.\n" >&2
    exit 1
fi
[ -f "$BUNDLE/Contents/Resources/models/models.json" ] \
    && ok "model manifest present" || bad "missing Contents/Resources/models/models.json"

# ── 2. Exported symbols ──────────────────────────────────────────────────────
printf "\n2) Exported symbols\n"
if [ "$HOST_OS" = "linux" ]; then
    SYMBOLS="$(nm -D --defined-only "$MODULE" 2>/dev/null || true)"
else
    SYMBOLS="$(nm -g "$MODULE" 2>/dev/null || true)"
fi
for symbol in $REQUIRED_SYMBOLS; do
    if printf "%s" "$SYMBOLS" | grep -Eq "(^|[[:space:]])_?${symbol}$"; then
        ok "$symbol"
    else
        bad "missing exported symbol ${symbol} (the host will show 0 classes and not complain)"
    fi
done

# ── 3. Dependency closure ────────────────────────────────────────────────────
#
# Criterion: every dynamic dependency must be either inside the bundle or a **system library**
# (/lib, /usr/lib prefix). Everything else counts as a leak — the build environment's absolute
# paths are guaranteed not to exist on the target system.
printf "\n3) Dynamic dependencies\n"
ESCAPED=()
if [ "$HOST_OS" = "linux" ]; then
    # [Why LD_LIBRARY_PATH is cleared] this check answers "can it resolve on the user's
    # machine": the user has no search paths into the build tree, so dependencies can only
    # come from the module's own $ORIGIN rpath and its in-bundle neighbors. A CI job-level
    # LD_LIBRARY_PATH (needed by linuxdeploy / the packaging step, pointing at
    # backend/target/dist) takes precedence over DT_RUNPATH and hijacks libwebgpu_dawn.so
    # into the build tree — the criterion would be polluted by the build environment, and
    # something present inside the bundle would be misreported as "external".
    LINKAGE="$(env -u LD_LIBRARY_PATH ldd "$MODULE" 2>&1 || true)"
    if printf "%s" "$LINKAGE" | grep -q "not found"; then
        bad "some dependencies cannot be resolved:"
        printf "%s\n" "$LINKAGE" | grep "not found" | sed "s/^/    /"
    else
        ok "ldd resolved everything"
    fi
    while IFS= read -r linkage_line; do
        dep="$(printf "%s" "$linkage_line" | sed -n "s/.*=> \(\/[^ ]*\).*/\1/p")"
        [ -z "$dep" ] && continue
        case "$dep" in
            "$BUNDLE"/*) continue ;;
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$LINKAGE"
else
    LINKAGE="$(otool -L "$MODULE" 2>&1 || true)"
    # [Why the module itself is skipped] otool -L's first "dependency" is really the module's
    # own install name (LC_ID_DYLIB), written at link time as a build-environment absolute
    # path — it is not a dependency, yet it would denounce itself under this "anything
    # pointing outside the package" criterion. otool -D extracts it precisely.
    SELF_ID="$(otool -D "$MODULE" 2>/dev/null | tail -n +2 | head -1 || true)"
    while IFS= read -r linkage_line; do
        dep="$(printf "%s" "$linkage_line" | sed -E "s/^[[:space:]]*//; s/ \(compatibility.*$//")"
        [ -z "$dep" ] && continue
        case "$dep" in
            # otool -L's first entry is the module's own install name (LC_ID_DYLIB), not a dependency.
            "$MODULE") continue ;;
            "$SELF_ID") continue ;;
            /usr/lib/*|/System/*) continue ;;
            @rpath/*|@loader_path/*)
                # An @rpath entry must resolve to a real file inside Frameworks, otherwise loading fails anyway.
                base="$(basename "$dep")"
                [ -f "$LIB_DIR/$base" ] || bad "dependency $base declared as $dep but $LIB_DIR does not contain it"
                ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$(printf "%s\n" "$LINKAGE" | tail -n +2)"
fi
if [ ${#ESCAPED[@]} -eq 0 ]; then
    ok "no non-system dependencies pointing outside the bundle"
else
    bad "the following dependencies point outside the bundle (very likely absent on the user's machine):"
    printf "    %s\n" "${ESCAPED[@]}"
fi

# ── 4. Black-box load ────────────────────────────────────────────────────────
printf "\n4) Black-box load (dlopen → entry → factory)\n"
PROBE_DIR="$(mktemp -d)"
trap "rm -rf \"$PROBE_DIR\"" EXIT
EXTRA_LIBS=""
[ "$HOST_OS" = "linux" ] && EXTRA_LIBS="-ldl"
    if cc -O1 -Wall -o "$PROBE_DIR/probe" "$SCRIPT_DIR/probe.c" $EXTRA_LIBS 2>"$PROBE_DIR/cc.log"; then
        # Same reasoning as above: the black-box load is a fact check of the packaged form,
        # with no search paths from the build tree.
        if run_probe env -u LD_LIBRARY_PATH "$PROBE_DIR/probe" "$MODULE" >"$PROBE_DIR/out.log" 2>&1; then
        ok "module loads, factory usable"
        sed "s/^/    /" "$PROBE_DIR/out.log"
    else
        bad "black-box load failed:"
        sed "s/^/    /" "$PROBE_DIR/out.log"
    fi
else
    bad "probe compilation failed (a C compiler is required):"
    sed "s/^/    /" "$PROBE_DIR/cc.log"
fi

# ── 5. macOS bundle metadata ─────────────────────────────────────────────────
if [ "$HOST_OS" = "macos" ]; then
    printf "\n5) macOS bundle metadata\n"
    PLIST="$BUNDLE/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        ok "Info.plist present"
        EXEC_NAME="$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$PLIST" 2>/dev/null || true)"
        if [ -n "$EXEC_NAME" ] && [ -f "$BUNDLE/Contents/MacOS/$EXEC_NAME" ]; then
            ok "CFBundleExecutable=$EXEC_NAME points at a real file"
        else
            bad "CFBundleExecutable missing or pointing at a non-existent file (hosts will not load this bundle)"
        fi
    else
        bad "missing Contents/Info.plist"
    fi
    if codesign --verify --strict "$BUNDLE" >/dev/null 2>&1; then
        ok "signature valid (ad-hoc)"
    else
        bad "signature verification failed (will not load on arm64)"
    fi
else
    printf "\n5) macOS bundle metadata — not applicable on this platform\n"
fi

# ── 6. Packaging hygiene ─────────────────────────────────────────────────────
printf "\n6) Packaging hygiene\n"
# [Why dangling links are checked] ORT's libwebgpu_dawn under target/ is a **symlink into
# ~/.cache**; cp without -L packs the link itself into the artifact, pointing at a path that
# does not exist on the user's machine.
DANGLING="$(find "$BUNDLE" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
if [ -z "$DANGLING" ]; then
    ok "no dangling symlinks"
else
    bad "dangling symlinks found:"
    printf "%s\n" "$DANGLING" | sed "s/^/    /"
fi
MODULE_KIND="$(file -b "$MODULE" 2>/dev/null || echo unknown)"
if [ "$HOST_OS" = "linux" ]; then
    case "$MODULE_KIND" in
        *ELF*shared\ object*) ok "module is an ELF shared library" ;;
        *) bad "module is not an ELF shared library: $MODULE_KIND" ;;
    esac
else
    case "$MODULE_KIND" in
        *Mach-O*) ok "module is Mach-O" ;;
        *) bad "module is not Mach-O: $MODULE_KIND" ;;
    esac
fi


# ── 7. Frontend artifacts ────────────────────────────────────────────────────
#
# [Why these three checks live here] a bundle missing one asset is the most common packaging
# omission, and its symptom is "the plugin window is completely blank" — the host does not
# complain and no log line appears. The three criteria cover: is there an html, do the
# js/css files the html references exist, and does anything secretly depend on remote resources.
printf "\n7) Frontend artifacts\n"
FRONTEND="$BUNDLE/Contents/Resources/frontend"
if [ -s "$FRONTEND/plugin.html" ]; then
    ok "plugin.html present ($(du -h "$FRONTEND/plugin.html" | cut -f1))"
else
    bad "missing or empty Contents/Resources/frontend/plugin.html"
fi
# Every js/css referenced by the html must really exist.
MISSING_ASSETS=""
if [ -s "$FRONTEND/plugin.html" ]; then
    while IFS= read -r asset; do
        [ -z "$asset" ] && continue
        case "$asset" in
            /*) rel="${asset#/}" ;;
            *)  rel="$asset" ;;
        esac
        [ -f "$FRONTEND/$rel" ] || MISSING_ASSETS="$MISSING_ASSETS $rel"
    done <<< "$(grep -oE '(src|href)="[^"]+[.](js|css)"' "$FRONTEND/plugin.html" \
        | sed -E 's/^(src|href)="//; s/"$//' || true)"
fi
if [ -z "$MISSING_ASSETS" ]; then
    ok "every js/css referenced by the html is inside the bundle"
else
    bad "html references non-existent resources: $MISSING_ASSETS"
fi
# Remote references: the plugin window must work offline.
#
# [Why only html/css **attributes** are checked instead of a repo-wide grep] minified JS is
# full of **namespace strings** like `http://www.w3.org/2000/svg` and links inside license
# text — they load nothing, yet they make a repo-wide grep report piles of false positives
# that everyone eventually ignores. The only things that can trigger a network request are
# the html's src/href and css's url(), so the criteria look at exactly those two places.
REMOTE=""
if [ -d "$FRONTEND" ]; then
    HTML_REMOTE="$(grep -rInE "(src|href)[[:space:]]*=[[:space:]]*[\"']https?://" \
        --include=*.html "$FRONTEND" 2>/dev/null || true)"
    CSS_REMOTE="$(grep -rInE "url\([[:space:]]*[\"']?https?://" \
        --include=*.css "$FRONTEND" 2>/dev/null || true)"
    REMOTE="$(printf "%s\n%s\n" "$HTML_REMOTE" "$CSS_REMOTE" | sed "/^$/d")"
fi
if [ -z "$REMOTE" ]; then
    ok "no remote references in html/css"
else
    note "the following locations reference remote resources:"
    printf "%s\n" "$REMOTE" | sed "s/^/    /"
    bad "remote references in the frontend artifacts (offline opening will fail)"
fi
# ── 8. Host handshake ────────────────────────────────────────────────────────
#
# [Why this belongs in the packaging gate] VST3 IIDs are compared **byte for byte**, and
# `INLINE_UID` expands into two layouts depending on `COM_COMPATIBLE`: Windows uses
# COM/GUID, Linux / macOS use big-endian per 32-bit word. The failure shape of sending the
# wrong layout is "the host answers with a null pointer" — no crash, no error, completely
# invisible on Windows, while on Unix it means ARA gets no item, the editor and processor
# lose their link, and the project never marks itself dirty.
#
# The probe implements `IHostApplication` and `IReaperHostApplication` itself (**using only
# this platform's layout**), walks `setHostContext → createInstance → initialize`, then reads
# the plugin log and asserts three lines. No view is built and X11 is not touched, so it runs
# without a display server.
printf "\n8) Host handshake (native extension + IID layout)\n"
if [ "$HOST_OS" = "linux" ]; then
    if cc -O1 -Wall -o "$PROBE_DIR/host_iid" "$SCRIPT_DIR/host_iid_probe.c" -ldl \
        2>"$PROBE_DIR/host_iid-cc.log"; then
        if run_probe "$PROBE_DIR/host_iid" "$MODULE" "$PROBE_DIR/handshake" >"$PROBE_DIR/host_iid.log" 2>&1; then
            ok "plugin obtained the host's native extension using this platform's layout"
            sed "s/^/    /" "$PROBE_DIR/host_iid.log"
        else
            bad "host handshake failed (plugin cannot obtain IReaperHostApplication):"
            sed "s/^/    /" "$PROBE_DIR/host_iid.log"
        fi
    else
        bad "handshake probe compilation failed (a C compiler is required):"
        sed "s/^/    /" "$PROBE_DIR/host_iid-cc.log"
    fi
else
    printf "    macOS skipped: the probe is written against ELF's loading path; Darwin needs its own pass\n"
fi

if [ "$FAILED" = "0" ]; then
    printf "\n[vst3:verify] all checks passed\n\n"
    exit 0
fi
printf "\n[vst3:verify] some checks failed\n\n" >&2
exit 1
