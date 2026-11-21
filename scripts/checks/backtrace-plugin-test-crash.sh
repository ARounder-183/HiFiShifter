#!/usr/bin/env bash
# Captures a stack when a test process is killed by a signal.
#
#     bash scripts/checks/backtrace-plugin-test-crash.sh [filters...]
#
# With no filters it replays the default two groups: the render path (draw a pitch line →
# vocoder) and the VST3 shell. Each has crashed once on CI, and cargo left only a
# "signal: 11, SIGSEGV".
#
# [Why a separate file] "Test criteria belong only in scripts/checks/" is a hard rule of
# this repo (scripts/checks/check-product-consistency.ps1 intercepts --skip /
# --test-threads / --offline in workflows). This one is a diagnostic, not a criterion, but
# its arguments likewise must not be scattered across YAML — otherwise the next person will
# assume it is part of the gates.
#
# Always exits 0: it is a diagnostic and must not turn an already-red job into a different red.
set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/backend/target}"

if ! command -v gdb >/dev/null 2>&1; then
    echo "no gdb: skipping stack capture (macOS runners do not ship it)"
    exit 0
fi

BIN="$(ls -t "$TARGET_DIR"/debug/deps/hifishifter_vst3-* 2>/dev/null | grep -v '\.d$' | head -1 || true)"
if [ -z "$BIN" ]; then
    echo "no test binary: skipping (run cargo test --no-run first)"
    exit 0
fi
echo "test binary: $BIN"

# [Why whole groups are replayed by default] The crash requires "hundreds of prior cases
# having run in the same process" as a precondition — the suspect case passes on its own
# (measured 71 seconds). Replaying a single case tests nothing.
if [ $# -eq 0 ]; then
    set -- 'editor::' 'vst3::'
fi

for filter in "$@"; do
    echo "::group::gdb $filter"
    # On a crash gdb stops on the signal and bt shows the crash site; without one it is just
    # a normal stack, still useful.
    # The replay must use the same EP criteria as the official tests: this script may be
    # invoked from a local shell or a standalone step where job-level env is absent, and Dawn
    # would then corrupt the heap on this Lavapipe-only machine — so the "capture stack" step
    # would manufacture its own SIGSEGV that looks like the original test crashing again.
    HIFISHIFTER_ORT_EP=cpu \
      gdb -batch -ex run -ex 'thread apply all bt 12' --args "$BIN" "$filter" --test-threads=1 || true
    echo "::endgroup::"
done
exit 0
