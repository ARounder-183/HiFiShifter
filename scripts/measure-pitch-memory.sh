#!/usr/bin/env bash
# measure-pitch-memory.sh
# Measure the peak working set of the pitch-analysis pipeline.
#
# Why this exists: the acceptance criterion for the long-audio memory fix is
# "peak working set is decoupled from source length", and that can only be
# checked by actually measuring allocations. The measurement lives in an
# `#[ignore]`d Rust test so it stays next to the code it guards; this script is
# just the entry point that runs it correctly.
#
# The counter is process-wide, so the test MUST run single-threaded and alone.
# `--ignored` runs only ignored tests and `--test-threads=1` keeps the harness
# from allocating on other threads while the measurement is in flight.
#
# Usage:
#   ./scripts/measure-pitch-memory.sh
#
# Interpreting the output:
#   The test prints peak bytes for a 60 s and a 240 s stereo source. The two
#   numbers should be essentially equal (both ~31 MiB with the default 30 s
#   chunk). If the second is several times the first, the pipeline has regressed
#   to holding buffers proportional to source length.
#
# `HIFISHIFTER_PITCH_CHUNK_SEC=0` switches analysis to the single-pass path,
# which is expected to scale with source length -- useful as a contrast.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/backend/src-tauri"

echo "==> measuring pitch-analysis peak working set"
cargo test --lib -- --ignored --test-threads=1 pitch_memory --nocapture
