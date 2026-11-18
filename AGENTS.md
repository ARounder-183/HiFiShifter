# HiFiShifter — Agent Engineering Guide

HiFiShifter is a cross-platform vocal editing & synthesis workstation ("Jinriki-VOCALOID" arrangement + tuning) shipping as **three host forms** over one shared Rust DSP/state kernel and one React/WebGL2 frontend:

1. **Standalone Desktop App** (`backend/src-tauri` — Tauri 2 + `cpal` audio engine + local Streamable HTTP / stdio MCP server).
2. **VST3 + ARA2 Plugin** (`backend/hifishifter-vst3` shell -> `backend/hifishifter-plugin` shared engine).
3. **CLAP Plugin** (`backend/hifishifter-clap` shell -> `backend/hifishifter-plugin` shared engine, supporting ARA2 final IDs + Capture mode).

---

## 1. Critical Operational Fact: CI Does NOT Run Tests or Static Gates

GitHub Actions workflows (`.github/workflows/build.yml`, `vst3-unix.yml`, `clap.yml`) **only build release artifacts** (see `docs/ci.md`). Unit tests, integration tests, and static gates do **not** run in CI.

**Every agent MUST run local verification before committing or declaring a task complete:**

```bash
# 1. Mandatory <3s cross-platform static gates (FFI panic guards, lock/error baselines,
#    frontend-backend command wiring, native dialog filters, UTF-8, log levels/verbosity):
python3 scripts/checks/run-fast-gates.py

# 2. Change-scoped minimal unit/typecheck router (prints or executes targeted checks):
python3 scripts/checks/verify-changed.py --print-only
python3 scripts/checks/verify-changed.py --run

# 3. Full suite entrypoints (when touching cross-cutting kernel/plugin/UI contracts):
bash scripts/checks/run-frontend-gates.sh
pwsh scripts/checks/run-rust-gates.ps1
```

---

## 2. Backend Workspace Map (`backend/Cargo.toml`)

| Crate | Package Name | Role & Hard Boundaries |
| :--- | :--- | :--- |
| `hifishifter-kernel` | `hifishifter-kernel` | Pure offline core: `TimelineState`, parameter authority, World / PC-NSF-HiFiGAN / VsLib vocoders, Signalsmith / SoundTouch stretch, `mixdown`, `pitch_bridge`. **Never** depends on Tauri, `wry`, or audio device I/O. |
| `src-tauri` | `HiFiShifter` (`backend_lib`) | Standalone Tauri 2 shell, `cpal` playback/recording engine, IPC commands (`commands.rs`), and App MCP adapter (`src/mcp/`). |
| `hifishifter-plugin` | `hifishifter-plugin` | Format-neutral `rlib` shared by VST3 & CLAP: ARA document model, Capture projection, embedded WebView editor stack, REAPER native extension. **No `cdylib` exports here.** |
| `hifishifter-plugin-core` | `hifishifter-plugin-core` | Format-neutral IDs, `MutexExt::lock_or_recover`, resource root resolution, sync primitives. |
| `hifishifter-vst3` | `hifishifter-vst3` | Thin VST3 + ARA2 ABI `cdylib` shell (`GetPluginFactory`). Requires pinned `ARA_SDK_DIR` & `ARA_VST3_SDK_DIR`. |
| `hifishifter-clap` | `hifishifter-clap` | Thin CLAP `cdylib` shell (`clap_entry`). **Never** build in the same `cargo` invocation as `HiFiShifter` (`src-tauri`), or Cargo feature unification pulls `vslib` into CLAP's DLL import table. |
| `hifishifter-ara` | `hifishifter-ara` | ARA2 companion binding helpers. |
| `hifishifter-capture` | `hifishifter-capture` | Real-time insert capture recorder, sidecar persistence, and loop-pass composite clip projection. |
| `hifishifter-host-link` | `hifishifter-host-link` | Lock-free two-file workspace handoff (`manifest.hfslink` + `edits.hfsedit` + sample-hashed float WAVs). |
| `hifishifter-clipboard` | `hifishifter-clipboard` | Native OS + REAPER / VocalShifter / MIDI clipboard codecs. |
| `hifishifter-mcp` | `hifishifter-mcp` | Streamable HTTP (`rmcp`) + `stdio` proxy, instance discovery, unified `voicebank` marks contract, and typed `EditorRequest` schemas (`docs/schemas/`). |

*(Separate out-of-workspace crate: `backend/hfs-gui` — standalone GUI helper process for Linux/Unix plugin embedding; remember to format its `Cargo.toml` when editing it.)*

---

## 3. Non-Negotiable Invariants ("Three Iron Laws" & Coordinate Axes)

1. **Panic Containment (`docs/thread-model.md`)**:
   - Every `extern "system"` / `extern "C"` / `extern "C-unwind"` host callback in `hifishifter-vst3`, `hifishifter-clap`, and `hifishifter-plugin` **must** wrap its body in `guard_ffi`, `boundary`, or audio-specific `catch_unwind`.
   - Every synchronous Tauri `#[tauri::command]` in `src-tauri/src/commands.rs` **must** wrap its body in `common::guard_command` / `guard_json_command`.
   - `[profile.release]` uses unwind (never set `panic = "abort"`).
2. **Poison-Recovering Locks & Structured Error Codes**:
   - Production Rust code **must never** call `.lock().unwrap()` (baseline is `0` in `baseline-lock-unwrap.txt`). Use `MutexExt::lock_or_recover()` or `.unwrap_or_else(|e| e.into_inner())`.
   - Never classify errors via `error.contains("...")` in production code (`baseline-error-contains.txt`). Use structured `code` fields (`McpErrorCode`, `notebook_*`, `hifishifter-host-link` codes).
3. **Host Capability Gating (`docs/host-extension-boundary.md`)**:
   - Never branch on `EditorHost::app_name()` strings. Gate UI and plugin commands strictly on `HostCapabilities` (`HostBackend::Native` for REAPER vs `HostBackend::AraOnly`).
   - Do **not** answer `com.celemony.ara.plugin_extension.draft/1` in CLAP (crashes REAPER 7.85; see `docs/2026-11-08-reaper-clap-ara-crash-analysis.md`).
4. **Time & Parameter Coordinate Axes**:
   - **Timeline clip positions/lengths**: project absolute seconds (`start_sec`, `length_sec`).
   - **Audio source windows**: media file seconds (`source_start_sec`, `source_end_sec`); never derive `playback_rate` by dividing looped source span by clip length!
   - **Parameter frame authority**: grid-cell semantics `[start + i*spf, start + (i+1)*spf)` via `param_authority::window_covers` (`5 ms` frame period). Curves belong to the **root track**, not individual clips.
   - **REAPER Take Pitch Envelope (`pitch_bridge.rs`)**: horizontal axis is `u = (t - item_start) * |rate|` (`u = 0` at `SOFFS`); **never** add `SOFFS` to `u` (`docs/2026-11-17-take-envelope-axis-findings.md`).

---

## 4. Parallel Subagent & Worktree Workflow

- Bootstrap isolated worktrees via `bash scripts/dev/setup-worktree.sh <branch-name>` (shares `frontend/node_modules`, `third_party/sdk`, and downloaded `resources/models/*` via symlinks).
- Set an isolated `CARGO_TARGET_DIR` per parallel agent when running concurrent Rust builds so Cargo file locks do not contend.
- In debug/test builds without the ~200 MB release ONNX weights, `src-tauri/build.rs` automatically overrides `TAURI_CONFIG` `bundle.resources=[]`; release/dist builds still require `python3 scripts/setup/fetch-models.py`.
- After intentionally modifying MCP contracts (`backend/hifishifter-mcp/src/contract.rs` or `voicebank.rs`), regenerate committed JSON Schemas with:
  `HFS_UPDATE_SCHEMAS=1 cargo test --manifest-path backend/Cargo.toml -p hifishifter-mcp --test schema_snapshot`

## 5. Subdirectory Guides & Domain Skills

- Plugin internals: [`backend/hifishifter-plugin/AGENTS.md`](backend/hifishifter-plugin/AGENTS.md)
- Frontend & render kernels: [`frontend/AGENTS.md`](frontend/AGENTS.md)
- App MCP & Voicebank workflow: [`docs/mcp.md`](docs/mcp.md)
- Domain skills in `.github/skills/`: `hfs-ipc-command-wiring`, `hfs-coordinate-axes`, `hfs-plugin-ffi-safety`, `hfs-mcp-vocal-workflow`.
