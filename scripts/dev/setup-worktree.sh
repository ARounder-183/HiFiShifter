#!/usr/bin/env bash
# Bootstrap an isolated Git worktree for parallel subagent development.
#
# Reuses heavy read-only / cached directories from the primary checkout via symlinks
# (`frontend/node_modules`, `third_party/sdk`, `backend/src-tauri/resources/models/*`)
# so a parallel subagent can run `vitest`, `tsc`, and `cargo test` in seconds without
# redownloading 200 MB ONNX models or reinstalling 400 npm packages.
#
# Usage:
#   bash scripts/dev/setup-worktree.sh <branch-or-worktree-name> [base-ref]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/dev/setup-worktree.sh <worktree-name> [base-ref]" >&2
    exit 1
fi

NAME="$1"
BASE_REF="${2:-HEAD}"

MAIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WT_DIR="$MAIN_ROOT/.worktrees/$NAME"

mkdir -p "$MAIN_ROOT/.worktrees"

if [ ! -d "$WT_DIR" ]; then
    if git -C "$MAIN_ROOT" show-ref --verify --quiet "refs/heads/$NAME"; then
        git -C "$MAIN_ROOT" worktree add "$WT_DIR" "$NAME"
    else
        git -C "$MAIN_ROOT" worktree add -b "$NAME" "$WT_DIR" "$BASE_REF"
    fi
fi

# 1. Share frontend/node_modules if present in main checkout
if [ -d "$MAIN_ROOT/frontend/node_modules" ] && [ ! -e "$WT_DIR/frontend/node_modules" ]; then
    ln -s "$MAIN_ROOT/frontend/node_modules" "$WT_DIR/frontend/node_modules"
    echo "[worktree] linked frontend/node_modules"
fi

# 2. Share pinned ARA / VST3 / CLAP SDKs if present in main checkout
if [ -d "$MAIN_ROOT/third_party/sdk" ] && [ ! -e "$WT_DIR/third_party/sdk" ]; then
    mkdir -p "$WT_DIR/third_party"
    ln -s "$MAIN_ROOT/third_party/sdk" "$WT_DIR/third_party/sdk"
    echo "[worktree] linked third_party/sdk"
fi

# 3. Share downloaded ONNX model subdirectories (preserving tracked models.json)
MAIN_MODELS="$MAIN_ROOT/backend/src-tauri/resources/models"
WT_MODELS="$WT_DIR/backend/src-tauri/resources/models"
if [ -d "$MAIN_MODELS" ] && [ -d "$WT_MODELS" ]; then
    for sub in "$MAIN_MODELS"/*; do
        base_name="$(basename "$sub")"
        if [ "$base_name" = "models.json" ] || [ ! -e "$sub" ]; then
            continue
        fi
        if [ ! -e "$WT_MODELS/$base_name" ]; then
            ln -s "$sub" "$WT_MODELS/$base_name"
            echo "[worktree] linked resources/models/$base_name"
        fi
    done
fi

echo ""
echo "Worktree ready at: $WT_DIR"
echo "Recommended isolated target dir for parallel Cargo builds:"
echo "  export CARGO_TARGET_DIR=\"$WT_DIR/backend/target-agent\""
