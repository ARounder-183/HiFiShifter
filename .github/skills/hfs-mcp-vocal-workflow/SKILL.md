---
name: hfs-mcp-vocal-workflow
description: Use when operating HiFiShifter via App MCP tools to arrange vocals, tune curves, audition acoustic previews, or extend backend/hifishifter-mcp and backend/src-tauri/src/mcp.
---

# HiFiShifter MCP Vocal Arrangement & Co-Editing Workflow

HiFiShifter's App MCP (`http://127.0.0.1:39393/mcp` or `HiFiShifter --mcp-stdio`) edits the **live visible project** shared with the human user. Full contract: `docs/mcp.md` and `docs/schemas/mcp-tools.schema.json`.

## Standard Closed-Loop Agent Sequence

1. **Discover Context & Capabilities**:
   - Call `hfs_get_context` to obtain `revision` and `scope_revisions` (`tempo_revision` + per-root-track `track_revisions`).
   - Call `hfs_get_edit_capabilities` to inspect root-track ownership, parameter units (`midi`, `cents`, `scale_steps`, `linear_gain`), min/max bounds, and gates (`compose_disabled`, `harmonic_separation_disabled`).
2. **Choose the Right Concurrency Token for `hfs_edit`**:
   - Co-editing alongside a human? Prefer `expected_scope`:
     - `{"mode": "isolated_new_tracks", "tempo_revision": "..."}` when creating a new candidate root track.
     - `{"mode": "root_track", "root_track_id": "...", "track_revision": "...", "tempo_revision": "..."}` when refining a single root track's clips, takes, or curves.
   - Changing global `set_tempo` or `set_project_length`? Pass `expected_revision` (note: `set_tempo` preserves existing `scale` at matching `position_sec` unless `scale` or `clear_scale: true` is passed).
   - Always supply a unique `idempotency_key` (`1..=128` chars) and human-readable `label`.
3. **Pack Candidates & Shape Expressive Curves**:
   - Use `import_candidate_takes` (`1..=16` candidates from `*.marks.json`) so candidate 0 plays immediately and the human can press `T` / `Shift+T` to A/B alternatives.
   - Wait for source F0 analysis via `hfs_wait_ready({"targets": ["pitch_analysis"], "track_ids": [...]})` before applying `shift_to_note` or `flatten_to_note`.
   - Use declarative `transform_curve` modes (`hermite_transition`, `vibrato`, `smooth`) instead of streaming thousands of raw 5ms floats whenever possible.
4. **Wait for Synthesis & Run Acoustic Self-Check**:
   - Call `hfs_wait_ready({"targets": ["clip_render", "gui_view"], "track_ids": [...]})`.
   - Call `hfs_render_preview` on each arranged phrase (`<= 30s`) and inspect `metrics` (`clipped_samples == 0`, `non_finite_samples == 0`, `internal_silent_dropout_count`, `boundary_checks[].click_suspect`) — fix clicks/dropouts before handing off!
5. **Human Handoff**:
   - Call `hfs_notebook` (`upsert_agent_section`) to post a fenced review checklist (OOV fallbacks, cross-take splices, timestamps) without touching human notes.
   - Call `hfs_editor_view` with `selected_track_id`, `active_param: "pitch"`, `focus_start_sec`, `focus_end_sec`, and `seek_playhead: true` to focus the GUI onto the phrase.

## Extending MCP Contracts

Whenever `backend/hifishifter-mcp/src/contract.rs` or `voicebank.rs` changes:
1. Update `docs/schemas/*.json` via `HFS_UPDATE_SCHEMAS=1 cargo test --manifest-path backend/Cargo.toml -p hifishifter-mcp --test schema_snapshot`.
2. Add headless integration coverage in `backend/src-tauri/src/mcp/editor_tests.rs` (`HeadlessMcpHost`).
