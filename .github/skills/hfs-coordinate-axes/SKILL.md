---
name: hfs-coordinate-axes
description: Use when touching clip geometry, time stretching, crossfades, parameter curves, parameter blocks, ARA projection, or REAPER Take Pitch Envelope conversion.
---

# HiFiShifter Time & Parameter Coordinate Axes

Subtle off-by-one frame or wrong-axis bugs ("off by `SOFFS`", "stretched loop clip squashes curve") are the #1 source of audio/tuning regressions. Verify which coordinate domain every variable belongs to before writing math.

## The Four Coordinate Domains

1. **Project Time (`project_sec`, `f64`)**:
   - `Clip.start_sec`, `Clip.length_sec`, `Clip.fade_in_sec`, `Clip.fade_out_sec`, `TempoPointData.position_sec`.
   - Crossfade width `F` between adjacent clips is always in **project seconds**, never source-handle seconds:
     - Left clip: `start_sec = A + F - L_src`, `fade_out_sec = F`
     - Right clip: `start_sec = A`, `length_sec = D`, `playback_rate = R_src / D`, `fade_in_sec = F`
2. **Media Source Time (`source_sec`, `f64`)**:
   - `Clip.source_start_sec`, `ClipTake.source_start_sec`, `ClipTake.source_end_sec`, and `PhraseMarks` (`*.marks.json`) spans.
   - **Never** compute `playback_rate` as `(source_end - source_start) / length_sec`: on looped items (`B_LOOPSRC=1`), the source window is `[anchor, media_end)` and dividing it misclassifies a split as a 2x stretch (`docs/host-extension-boundary.md` §3). Always read explicit `playback_rate`.
3. **Parameter Curve Grid Cells (`frame_index`, `5 ms` period)**:
   - Root-track owned (`params_by_root_track[root_id]`); child tracks only own delta curves (`child_pitch_offset_cents@<id>`, `child_pitch_offset_degrees@<id>`, `child_formant_offset_cents@<id>`).
   - Sample `i` owns the half-open cell `[start + i*spf, start + (i+1)*spf)` (`param_authority::window_covers`), **not** a point instant.
   - Blank regions on `dyn` display `0.0` (`common_params::dyn_orig_level_at`), while unedited synthesis frames use `DYN_FOLLOW_ORIG` (unity gain).
   - Pitch curve transforms (`flatten_to_note`, `shift_to_note`, `smooth`, `vibrato`) must skip unvoiced frames (`orig[i] <= 0.0`) so consonants are never pitched or dragged toward zero.
4. **REAPER Take Envelope Axis (`u`, `hifishifter-kernel/src/pitch_bridge.rs`)**:
   - `u = (t_project - item_start) * |rate|`, with `u = 0` anchored at the take in-point (`SOFFS`).
   - **Never** add `SOFFS` to `u`, even for reversed (`MODE 3`) or looped takes (`docs/2026-11-17-take-envelope-axis-findings.md`).
   - Always route conversions through `pitch_bridge::offset_to_take_u` / `take_u_to_offset` and force start/end boundary points (REAPER holds endpoint values outside the first/last envelope point).
