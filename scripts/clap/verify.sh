#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiFiShifter CLAP artifact self-check (macOS / Linux).
#
#     scripts/clap/verify.sh <path/to/HiFiShifter.clap>
#
# On macOS this path is a bundle directory, on Linux a single-file module.
#
# The criteria share their origin with the VST3 pipeline: **trust external evidence only** —
# the symbol table, the dynamic section, a real dlopen that also completes the process and
# state round-trips, not "the build script said it was fine".
#
#   1 Layout  2 Exported symbols  3 Dependency closure  4 Black-box load  5 macOS bundle metadata
#   6 Packaging hygiene  7 Frontend artifacts  8 Model manifest
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TARGET="${1:?usage: verify.sh <path/to/HiFiShifter.clap>}"

FAILED=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$*"; FAILED=1; }
note() { printf "    %s\n" "$*"; }

# Same discipline as vst3/unix/verify.sh: a deadlock in the plugin entry must fail here,
# never hang the pipeline. The discipline includes the same fallback: macOS runners have no
# GNU coreutils `timeout` (rc=127), python3 is available on both CI legs, and a fresh process
# group is killed after a time limit with exit semantics aligned with timeout
# (timeout 124, child killed by signal 128+n). Per-line rationale: run_probe in vst3/unix/verify.sh.
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

[ -e "$TARGET" ] || { printf "artifact does not exist: %s\n" "$TARGET" >&2; exit 2; }

case "$(uname -s)" in
    Linux)  HOST_OS="linux" ;;
    Darwin) HOST_OS="macos" ;;
    *)      printf "unsupported platform\n" >&2; exit 2 ;;
esac

if [ "$HOST_OS" = "linux" ]; then
    # Linux's CLAP is a single file with a sibling <stem>-resources directory.
    if [ -d "$TARGET" ]; then MODULE="$TARGET/HiFiShifter.clap"; else MODULE="$TARGET"; fi
    BUNDLE_DIR="$(cd -- "$(dirname -- "$MODULE")" && pwd)"
    RESOURCES="$BUNDLE_DIR/HiFiShifter-resources"
else
    # macOS's CLAP is a bundle: Contents/MacOS + Contents/Frameworks + Contents/Resources.
    if [ -d "$TARGET" ]; then BUNDLE_DIR="$(cd -- "$TARGET" && pwd)"; else BUNDLE_DIR="$(cd -- "$(dirname -- "$TARGET")/../.." && pwd)"; fi
    MODULE="$BUNDLE_DIR/Contents/MacOS/HiFiShifter"
    RESOURCES="$BUNDLE_DIR/Contents/Resources"
fi

printf "\n[clap:verify] %s\n" "$MODULE"

# ── 1. Layout ────────────────────────────────────────────────────────────────
printf "\n1) Layout\n"
if [ -f "$MODULE" ]; then
    ok "module present: $MODULE"
else
    bad "module not found: $MODULE"
    printf "\nLayout is wrong; the remaining checks are meaningless.\n" >&2
    exit 1
fi
if [ "$HOST_OS" = "linux" ]; then
    # A single-file CLAP has no Contents/; resources live beside the module as
    # <stem>-resources (see plugin-core's ResourceLayout::Sibling). Without that directory
    # the plugin's frontend_dir() returns an error the first time the editor opens — on the
    # host side that reads as "blank window", with not a single log line.
    if [ -d "$RESOURCES" ]; then ok "resources directory beside the module: $(basename "$RESOURCES")"
    else bad "missing sibling resources directory: $RESOURCES"; fi
else
    if [ -d "$RESOURCES" ]; then ok "bundle has Contents/Resources"
    else bad "missing Contents/Resources"; fi
fi
[ -f "$RESOURCES/models/models.json" ] && ok "model manifest present" || bad "missing models/models.json"

# ── 2. Exported symbols ──────────────────────────────────────────────────────
printf "\n2) Exported symbols\n"
if [ "$HOST_OS" = "linux" ]; then
    SYMBOLS="$(nm -D --defined-only "$MODULE" 2>/dev/null || true)"
else
    SYMBOLS="$(nm -gU "$MODULE" 2>/dev/null || true)"
fi
if printf "%s" "$SYMBOLS" | grep -Eq "(^|[[:space:]])_?clap_entry$"; then
    ok "clap_entry"
else
    bad "missing exported symbol clap_entry (hosts will decide this is not a CLAP plugin)"
fi

# ── 3. Dependency closure ────────────────────────────────────────────────────
#
# Criterion: every dynamic dependency must be either inside the package or a **system
# library** (/lib, /usr/lib prefix). Everything else counts as a leak — the build
# environment's absolute paths are guaranteed not to exist on the target system.
printf "\n3) Dynamic dependencies\n"
ESCAPED=()
if [ "$HOST_OS" = "linux" ]; then
    # [Why LD_LIBRARY_PATH is cleared] this check answers "can it resolve on the user's
    # machine": the user has no search paths into the build tree, so dependencies can only
    # come from the module's own $ORIGIN rpath and packaged copies next to it. A CI
    # job-level LD_LIBRARY_PATH (needed by linuxdeploy / the packaging step, pointing at
    # backend/target/dist) takes precedence over DT_RUNPATH and hijacks libwebgpu_dawn.so
    # into the build tree — the criterion would be polluted by the build environment, and
    # something sitting next to the module would be misreported as "outside the package".
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
            "$BUNDLE_DIR"/*) continue ;;
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$LINKAGE"
    # Dependencies sitting beside the module require a \$ORIGIN search path; otherwise a
    # user's machine can only rely on LD_LIBRARY_PATH — that is the host process's
    # environment, which we have no right to demand.
    RUNPATH="$(readelf -d "$MODULE" 2>/dev/null | sed -n 's/.*(\(RPATH\|RUNPATH\)).*\[\(.*\)\]/\2/p' || true)"
    case ":$RUNPATH:" in
        *':$ORIGIN'*) ok "rpath contains \$ORIGIN (${RUNPATH})" ;;
        *) bad "module has no \$ORIGIN rpath (current: ${RUNPATH:-none})" ;;
    esac
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
            # [Why the module itself is skipped] otool -L's first "dependency" is really the
            # module's own install name (LC_ID_DYLIB), written at link time as a
            # build-environment absolute path. It is not a dependency, but under this
            # "anything pointing outside the package" criterion it would denounce itself.
            "$MODULE") continue ;;
            "$SELF_ID") continue ;;
            /usr/lib/*|/System/*) continue ;;
            @rpath/*|@loader_path/*)
                base="$(basename "$dep")"
                [ -f "$BUNDLE_DIR/Contents/Frameworks/$base" ] \
                    || bad "dependency $base declared as $dep but Contents/Frameworks does not contain it"
                ;;
            *) ESCAPED+=("$dep") ;;
        esac
    done <<< "$(printf "%s\n" "$LINKAGE" | tail -n +2)"
    if otool -l "$MODULE" 2>/dev/null | grep -A2 'LC_RPATH' | grep -q '@loader_path'; then
        ok "rpath contains @loader_path"
    else
        bad "module has no @loader_path rpath"
    fi
fi
if [ ${#ESCAPED[@]} -eq 0 ]; then
    ok "no non-system dependencies pointing outside the package"
else
    bad "the following dependencies point outside the package (very likely absent on the user's machine):"
    printf "    %s\n" "${ESCAPED[@]}"
fi

# ── 4. Black-box load ────────────────────────────────────────────────────────
#
# The probe really walks dlopen → init → factory → instance → activate → process
# (per-sample passthrough comparison) → state save/load round-trip, and cross-checks the
# pairing of the ARA factory table with the descriptor. Any failure means "it compiles" and
# "it works" are still far apart.
printf "\n4) Black-box load (dlopen → entry → factory → instance → process → state)\n"
PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROBE_DIR"' EXIT
EXTRA_LIBS=""
[ "$HOST_OS" = "linux" ] && EXTRA_LIBS="-ldl"
if cc -O1 -Wall -o "$PROBE_DIR/probe" "$SCRIPT_DIR/probe.c" $EXTRA_LIBS 2>"$PROBE_DIR/cc.log"; then
    # Same reasoning as above: the black-box load is a fact check of the packaged form,
    # with no search paths from the build tree.
    if run_probe env -u LD_LIBRARY_PATH "$PROBE_DIR/probe" "$MODULE" >"$PROBE_DIR/out.log" 2>&1; then
        ok "module loads, factory and instance usable"
        sed "s/^/    /" "$PROBE_DIR/out.log"
        grep -q "plugin_id=com.hifishifter.clap" "$PROBE_DIR/out.log" || bad "descriptor id is not com.hifishifter.clap"
        grep -q "process=ok" "$PROBE_DIR/out.log" || bad "process did not run through (passthrough comparison failed)"
        # ── ARA (stage 2) ───────────────────────────────────────────────────
        # The criterion is inverted from "must be absent": the host sees the factory table
        # at scan time and pairs it with the descriptor via get_plugin_id; one character of
        # difference and the plugin is never claimed by ARA, while the host log says
        # nothing at all.
        grep -q "ara_factory=present plugin_id=com.hifishifter.clap" "$PROBE_DIR/out.log" \
            || bad "entry publishes no ARA factory table paired with the descriptor"
        grep -q "ara_plugin_extension=present" "$PROBE_DIR/out.log" \
            || bad "instance has no ARA plugin extension: the host cannot bind the document controller"
        # REAPER's 2022 private draft ID must be **rejected**: its host implementation
        # crashes inside its own code when it actually needs to render (reaper.exe+0x5a677).
        # This assertion keeps it from being "conveniently" re-added.
        grep -q "ara_reaper_legacy=absent" "$PROBE_DIR/out.log" \
            || bad "instance answered REAPER's private draft ARA extension ID (would crash REAPER)"
        grep -q "feature=ara:supported" "$PROBE_DIR/out.log" \
            || bad "descriptor does not declare ara:supported (hosts will not treat this plugin as an ARA candidate)"
        if grep -q "feature=ara:required" "$PROBE_DIR/out.log"; then
            bad "descriptor declares ara:required (hosts without ARA would refuse to create instances outright)"
        fi
        # [Why extensions are asserted one by one] the old criterion had a hole: the probe
        # printed "absent" and still returned 0, so artifacts missing clap.state stayed
        # green while hosts lost every edit on project save. Now "must be present" and
        # "must be absent" are both hard criteria, so capability claims and self-checks
        # cannot wave each other through.
        grep -q "round-trip([0-9]* bytes)" "$PROBE_DIR/out.log" || bad "state round-trip did not run through"
        grep -q "render_extension=present" "$PROBE_DIR/out.log" || bad "clap.render extension missing"
        grep -q "audio_ports_extension=present" "$PROBE_DIR/out.log" || bad "clap.audio-ports extension missing"
        grep -q "timer_support_extension=present" "$PROBE_DIR/out.log" || bad "clap.timer-support extension missing"
        grep -q "gui_extension=present" "$PROBE_DIR/out.log" || bad "clap.gui extension missing: the host cannot create an editor window"
        grep -q "latency_extension=present latency=0" "$PROBE_DIR/out.log" || bad "clap.latency must honestly report 0"
        grep -q "tail_extension=present tail=0" "$PROBE_DIR/out.log" || bad "clap.tail must honestly report 0"
        grep -Eq "host_log_bytes=[1-9][0-9]*" "$PROBE_DIR/out.log" || bad "host received no clap.log output"
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
    PLIST="$BUNDLE_DIR/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        ok "Info.plist present"
        EXEC_NAME="$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$PLIST" 2>/dev/null || true)"
        if [ -n "$EXEC_NAME" ] && [ -f "$BUNDLE_DIR/Contents/MacOS/$EXEC_NAME" ]; then
            ok "CFBundleExecutable=$EXEC_NAME points at a real file"
        else
            bad "CFBundleExecutable missing or pointing at a non-existent file (hosts will not load this bundle)"
        fi
    else
        bad "missing Contents/Info.plist"
    fi
    if codesign --verify --strict "$BUNDLE_DIR" >/dev/null 2>&1; then
        ok "signature valid"
    else
        bad "signature verification failed (will not load on arm64)"
    fi
else
    printf "\n5) macOS bundle metadata — not applicable on this platform\n"
fi

# ── 6. Packaging hygiene ─────────────────────────────────────────────────────
printf "\n6) Packaging hygiene\n"
DANGLING="$(find "$BUNDLE_DIR" -type l ! -exec test -e {} \; -print 2>/dev/null || true)"
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
# A package missing one asset is the most common omission, and its symptom is "the plugin
# window is completely blank" — the host does not complain and no log line appears. Three
# criteria: is there an html, do the js/css files the html references exist, and are there
# remote references.
printf "\n7) Frontend artifacts\n"
FRONTEND="$RESOURCES/frontend"
if [ -s "$FRONTEND/plugin.html" ]; then
    ok "plugin.html present ($(du -h "$FRONTEND/plugin.html" | cut -f1))"
else
    bad "missing or empty frontend/plugin.html"
fi
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
    ok "every js/css referenced by the html is inside the package"
else
    bad "html references non-existent resources: $MISSING_ASSETS"
fi
REMOTE=""
if [ -d "$FRONTEND" ]; then
    HTML_REMOTE="$(grep -rInE "(src|href)[[:space:]]*=[[:space:]]*[\"']https?://" --include=*.html "$FRONTEND" 2>/dev/null || true)"
    CSS_REMOTE="$(grep -rInE "url\([[:space:]]*[\"']?https?://" --include=*.css "$FRONTEND" 2>/dev/null || true)"
    REMOTE="$(printf "%s\n%s\n" "$HTML_REMOTE" "$CSS_REMOTE" | sed "/^$/d")"
fi
if [ -z "$REMOTE" ]; then
    ok "no remote references in html/css"
else
    note "the following locations reference remote resources:"
    printf "%s\n" "$REMOTE" | sed "s/^/    /"
    bad "remote references in the frontend artifacts (offline opening will fail)"
fi

# ── 8. Model self-containment ────────────────────────────────────────────────
#
# Unix has no installer-owned shared model store (see model_store's module docs:
# shared_store_dir returns None on non-Windows), so the plugin uses the copy inside the
# bundle directly. The criterion is therefore not "no weights" but "every file in the
# manifest is present" — a missing one shows up as silent inference degradation, with no
# error from the host or the log.
printf "\n8) Model self-containment\n"
if [ -f "$RESOURCES/models/models.json" ]; then
    ok "models.json present"
    if command -v python3 >/dev/null 2>&1; then
        if python3 - "$RESOURCES/models" <<'PY'
import json, os, sys
root = sys.argv[1]
with open(os.path.join(root, "models.json"), encoding="utf-8") as fh:
    manifest = json.load(fh)
missing = [f["path"] for f in manifest.get("files", []) if not os.path.isfile(os.path.join(root, f["path"]))]
if missing:
    print("missing: " + ", ".join(missing))
    sys.exit(1)
print("files=%d version=%s" % (len(manifest.get("files", [])), manifest.get("version")))
PY
        then
            ok "every file in the manifest is in the package"
        else
            bad "model files referenced by the manifest are missing (inference would degrade silently)"
        fi
    else
        note "SKIP: no python3, cannot cross-check the manifest"
    fi
else
    bad "missing models/models.json"
fi

if [ "$FAILED" = "0" ]; then
    printf "\n[clap:verify] all checks passed\n\n"
    exit 0
fi
printf "\n[clap:verify] some checks failed\n\n" >&2
exit 1
