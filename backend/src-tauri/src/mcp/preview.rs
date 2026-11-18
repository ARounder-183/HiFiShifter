//! Bounded offline phrase preview mixdown and deterministic acoustic QA metrics (`hfs_render_preview`).
//!
//! Enables multimodal and verification Agents to render a <= 30s time window (optionally
//! isolating specific tracks without mutating the live project), inspect objective audio
//! metrics (peak/RMS dBFS, clipping, dropouts, clip-boundary clicks), and optionally write
//! an atomic `.wav` file for listening.

use super::draft::{self, Failure};
use crate::state::AppState;
use hifishifter_kernel::state::TimelineState;
use hifishifter_mcp::RenderPreviewRequest;
use serde_json::{json, Value};
use std::{collections::HashSet, path::Path, sync::Arc};

const MAX_PREVIEW_DURATION_SEC: f64 = 30.0;

pub(super) fn execute_render_preview(
    state: &AppState,
    revision: &str,
    request: RenderPreviewRequest,
) -> Result<Value, Failure> {
    draft::finite(request.start_sec, 0.0, 86400.0, "start_sec")?;
    draft::finite(request.end_sec, 0.0, 86400.0, "end_sec")?;
    let span = request.end_sec - request.start_sec;
    if span <= 0.0 || span > MAX_PREVIEW_DURATION_SEC {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            format!(
                "preview window (end_sec - start_sec) must be in (0, {MAX_PREVIEW_DURATION_SEC}] seconds"
            ),
        ));
    }
    let sample_rate = request.sample_rate.unwrap_or(44_100);
    if !(8_000..=96_000).contains(&sample_rate) {
        return Err(Failure::new(
            "INVALID_ARGUMENT",
            "sample_rate must be in 8000..=96000",
        ));
    }
    if let Some(out) = &request.output_wav_path {
        let p = Path::new(out);
        if !p.is_absolute()
            || p.extension()
                .and_then(|s| s.to_str())
                .is_none_or(|ext| !ext.eq_ignore_ascii_case("wav"))
        {
            return Err(Failure::new(
                "INVALID_ARGUMENT",
                "output_wav_path must be an absolute .wav path",
            ));
        }
        if p.exists() && request.overwrite != Some(true) {
            return Err(Failure::new(
                "SAVE_FAILED",
                "output_wav_path already exists; pass overwrite=true or choose a new path",
            ));
        }
    }

    let mut preview_tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    if let Some(filter_ids) = &request.track_ids {
        if filter_ids.is_empty() {
            return Err(Failure::new(
                "INVALID_ARGUMENT",
                "track_ids must be non-empty when provided (omit to mix all active tracks)",
            ));
        }
        let mut keep = HashSet::new();
        for id in filter_ids {
            if !preview_tl.tracks.iter().any(|t| t.id == *id) {
                return Err(Failure::new("TRACK_NOT_FOUND", id.clone()));
            }
            keep.insert(id.clone());
            // Include child tracks when a root/parent track is requested.
            for t in &preview_tl.tracks {
                if preview_tl.resolve_root_track_id(&t.id).as_deref() == Some(id.as_str()) {
                    keep.insert(t.id.clone());
                }
            }
            // Keep ancestor chain unmuted so root-track mix routing remains intact.
            let mut cur = preview_tl
                .tracks
                .iter()
                .find(|t| t.id == *id)
                .and_then(|t| t.parent_id.clone());
            while let Some(pid) = cur {
                keep.insert(pid.clone());
                cur = preview_tl
                    .tracks
                    .iter()
                    .find(|t| t.id == pid)
                    .and_then(|t| t.parent_id.clone());
            }
        }
        for track in &mut preview_tl.tracks {
            track.solo = false;
            track.muted = !keep.contains(&track.id);
        }
        // If only a specific child track was requested, mute sibling clips outside `filter_ids`
        // unless its root track itself was explicitly requested.
        let requested_exact: HashSet<&str> = filter_ids.iter().map(String::as_str).collect();
        let any_root_requested = filter_ids.iter().any(|id| {
            preview_tl
                .tracks
                .iter()
                .any(|t| t.id == *id && t.parent_id.is_none())
        });
        if !any_root_requested {
            for clip in &mut preview_tl.clips {
                if !requested_exact.contains(clip.track_id.as_str()) {
                    clip.muted = true;
                }
            }
        }
    }

    let boundary_times = collect_window_boundaries(&preview_tl, request.start_sec, request.end_sec);
    let cache_stats = Arc::new(crate::mixdown::MixdownCacheStats::default());
    let opts = crate::mixdown::MixdownOptions {
        sample_rate,
        start_sec: request.start_sec,
        end_sec: Some(request.end_sec),
        stretch: crate::time_stretch::resolved_external_stretch_algorithm(),
        apply_pitch_edit: request.apply_pitch_edit.unwrap_or(true),
        output: crate::encode::OutputSpec::wav_32f(),
        quality_preset: crate::mixdown::QualityPreset::Realtime,
        cancel_flag: None,
        progress: None,
        cache_stats: Some(cache_stats.clone()),
    };

    let (out_rate, out_channels, duration_sec, interleaved) =
        crate::mixdown::render_mixdown_interleaved(&preview_tl, opts)
            .map_err(|e| Failure::new("PREVIEW_RENDER_FAILED", e))?;

    let metrics = compute_acoustic_metrics(
        &interleaved,
        out_rate,
        out_channels,
        request.start_sec,
        &boundary_times,
    );

    let mut written_bytes: Option<u64> = None;
    if let Some(out_str) = &request.output_wav_path {
        let out_path = Path::new(out_str);
        let parent = out_path
            .parent()
            .ok_or_else(|| Failure::new("SAVE_FAILED", "missing parent directory"))?;
        std::fs::create_dir_all(parent).map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        let temp = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        {
            let spec = hound::WavSpec {
                channels: out_channels,
                sample_rate: out_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::new(std::io::BufWriter::new(temp.as_file()), spec)
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
            for &sample in &interleaved {
                writer
                    .write_sample(if sample.is_finite() { sample } else { 0.0 })
                    .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
            }
            writer
                .finalize()
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        }
        temp.as_file()
            .sync_all()
            .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        if request.overwrite == Some(true) {
            temp.persist(out_path)
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        } else {
            temp.persist_noclobber(out_path)
                .map_err(|e| Failure::new("SAVE_FAILED", e.to_string()))?;
        }
        written_bytes = std::fs::metadata(out_path).ok().map(|m| m.len());
    }

    let (reused, stored) = cache_stats.snapshot();
    Ok(json!({
        "ok": true,
        "revision": revision,
        "start_sec": request.start_sec,
        "end_sec": request.end_sec,
        "duration_sec": duration_sec,
        "sample_rate": out_rate,
        "channels": out_channels,
        "apply_pitch_edit": request.apply_pitch_edit.unwrap_or(true),
        "output_wav_path": request.output_wav_path,
        "bytes_written": written_bytes,
        "cache_stats": { "reused_clips": reused, "stored_clips": stored },
        "metrics": metrics,
    }))
}

fn collect_window_boundaries(
    tl: &TimelineState,
    start_sec: f64,
    end_sec: f64,
) -> Vec<(String, &'static str, f64)> {
    let mut out = Vec::new();
    for clip in &tl.clips {
        if clip.muted {
            continue;
        }
        if clip.start_sec > start_sec + 0.005 && clip.start_sec < end_sec - 0.005 {
            out.push((clip.id.clone(), "clip_start", clip.start_sec));
        }
        let clip_end = clip.start_sec + clip.length_sec;
        if clip_end > start_sec + 0.005 && clip_end < end_sec - 0.005 {
            out.push((clip.id.clone(), "clip_end", clip_end));
        }
    }
    out
}

fn linear_to_dbfs(v: f64) -> f64 {
    if v <= 1e-9 {
        -180.0
    } else {
        20.0 * v.log10()
    }
}

fn compute_acoustic_metrics(
    interleaved: &[f32],
    sample_rate: u32,
    channels: u16,
    window_start_sec: f64,
    boundaries: &[(String, &'static str, f64)],
) -> Value {
    let ch = (channels.max(1)) as usize;
    let total_frames = interleaved.len() / ch;
    let mut peak = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut clipped_samples = 0usize;
    let mut non_finite_samples = 0usize;

    let mut mono = Vec::with_capacity(total_frames);
    for frame in interleaved.chunks_exact(ch) {
        let mut mono_acc = 0.0f64;
        for &s in frame {
            if !s.is_finite() {
                non_finite_samples += 1;
                continue;
            }
            let a = (s as f64).abs();
            if a > peak {
                peak = a;
            }
            if a > 1.0 {
                clipped_samples += 1;
            }
            sum_sq += (s as f64) * (s as f64);
            mono_acc += s as f64;
        }
        mono.push((mono_acc / ch as f64) as f32);
    }

    let rms = if !interleaved.is_empty() {
        (sum_sq / interleaved.len() as f64).sqrt()
    } else {
        0.0
    };

    // 20ms window activity & internal dropout detection (>= 40ms silent hole between active regions).
    let win_frames = ((sample_rate as usize) / 50).max(1);
    let active_thresh = 10f64.powf(-55.0 / 20.0); // -55 dBFS
    let mut win_active = Vec::new();
    for chunk in mono.chunks(win_frames) {
        let w_rms = (chunk.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
            / chunk.len().max(1) as f64)
            .sqrt();
        win_active.push(w_rms >= active_thresh);
    }
    let active_count = win_active.iter().filter(|&&a| a).count();
    let active_ratio = if win_active.is_empty() {
        0.0
    } else {
        active_count as f64 / win_active.len() as f64
    };

    let mut silent_dropouts = 0usize;
    if let (Some(first_act), Some(last_act)) = (
        win_active.iter().position(|&a| a),
        win_active.iter().rposition(|&a| a),
    ) {
        let mut run = 0usize;
        for &is_act in &win_active[first_act..=last_act] {
            if !is_act {
                run += 1;
            } else {
                if run >= 2 {
                    silent_dropouts += 1;
                }
                run = 0;
            }
        }
    }

    let radius = ((sample_rate as f64) * 0.005).round() as usize; // +-5ms click window
    let boundary_checks: Vec<Value> = boundaries
        .iter()
        .map(|(clip_id, kind, sec)| {
            let center = ((*sec - window_start_sec) * sample_rate as f64).round() as isize;
            let lo = (center - radius as isize).max(1) as usize;
            let hi = ((center + radius as isize) as usize).min(mono.len());
            let mut max_step = 0.0f64;
            if hi > lo {
                for idx in lo..hi {
                    let step = (mono[idx] as f64 - mono[idx - 1] as f64).abs();
                    if step > max_step {
                        max_step = step;
                    }
                }
            }
            json!({
                "clip_id": clip_id,
                "kind": kind,
                "position_sec": sec,
                "max_sample_step": max_step,
                "click_suspect": max_step > 0.35,
            })
        })
        .collect();

    json!({
        "peak_linear": peak,
        "peak_dbfs": linear_to_dbfs(peak),
        "rms_dbfs": linear_to_dbfs(rms),
        "clipped_samples": clipped_samples,
        "non_finite_samples": non_finite_samples,
        "active_window_ratio": active_ratio,
        "internal_silent_dropout_count": silent_dropouts,
        "boundary_checks": boundary_checks,
    })
}
