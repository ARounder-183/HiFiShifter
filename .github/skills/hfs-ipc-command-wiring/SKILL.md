---
name: hfs-ipc-command-wiring
description: Use when adding, renaming, or modifying any frontend-to-backend IPC command across HiFiShifter's standalone Tauri App and VST3/CLAP plugin editor.
---

# HiFiShifter IPC Command Wiring

HiFiShifter's React frontend (`frontend/src/services/invoke.ts`) runs identically inside two different hosts:
1. **Standalone App** (`backend/src-tauri/src/commands.rs` + `backend/src-tauri/src/lib.rs` `tauri::generate_handler![...]`).
2. **VST3 / CLAP Embedded Plugin Editor** (`backend/hifishifter-plugin/src/editor/commands.rs`).

An unwired command compiles cleanly in both Rust and TypeScript and fails silently at runtime when clicked.

## Mandatory Checklist

1. **Frontend Facade & Router**:
   - Add typed wrapper in `frontend/src/services/api/*.ts` (enforced by `frontend/tests/apiFacadeGates.test.ts` — UI components never call `invoke()` directly).
   - Route the method name in `frontend/src/services/invoke.ts`.
   - If the command is App-only (e.g., `mcp_*`) or requires native DAW extensions (e.g., clip split/move in REAPER), gate the UI control on `isPluginMode()` or `HostCapabilities` (`frontend/src/services/hostCapabilities.ts`).
2. **Standalone App Dispatch (`backend/src-tauri`)**:
   - Wrap synchronous `#[tauri::command]` handlers in `common::guard_command` / `common::guard_json_command` (`docs/thread-model.md`). Move slow media/disk work onto `spawn_blocking`.
   - Register the command in `tauri::generate_handler![...]` in `backend/src-tauri/src/lib.rs`.
3. **Plugin Editor Dispatch (`backend/hifishifter-plugin`)**:
   - If supported in plugin mode, add dispatch arm in `backend/hifishifter-plugin/src/editor/commands.rs`, checking `HostCapabilities` (never host name strings).
   - Return structured error codes, never ad-hoc substrings matched via `error.contains(...)`.
4. **Dialog Filters (if opening native file pickers)**:
   - Every `rfd` or host picker call must include an approved file-type filter (`python3 scripts/checks/check-dialog-filters.py`).
5. **Verification**:
   - Run `python3 scripts/checks/check-command-gating.py` and `npm --prefix frontend run test -- src/services/invoke.wiring.test.ts`.
