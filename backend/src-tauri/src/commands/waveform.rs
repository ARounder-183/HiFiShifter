use crate::state::AppState;
use crate::waveform;
use tauri::State;

use super::common::guard_waveform_command;

// ===================== waveform peaks =====================

pub(super) fn get_waveform_peaks_segment(
    state: State<'_, AppState>,
    source_path: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> waveform::WaveformPeaksSegmentPayload {
    let hop = 256usize;
    let cols = columns.clamp(16, 8192);

    let peaks = match state.get_or_compute_waveform_peaks(&source_path, hop) {
        Ok(p) => p,
        Err(_) => {
            return waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            }
        }
    };

    waveform::segment_from_cached(peaks.as_ref(), start_sec, duration_sec, cols)
}

pub(super) fn clear_waveform_cache(state: State<'_, AppState>) -> serde_json::Value {
    let stats = state.clear_waveform_cache();
    let dir = {
        state
            .waveform_cache_dir
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .display()
            .to_string()
    };
    serde_json::json!({
        "ok": true,
        "removed_files": stats.removed_files,
        "removed_bytes": stats.removed_bytes,
        "dir": dir,
    })
}

// ===================== root mix waveform peaks =====================

pub(super) fn get_root_mix_waveform_peaks_segment(
    state: State<'_, AppState>,
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> waveform::WaveformPeaksSegmentPayload {
    guard_waveform_command("get_root_mix_waveform_peaks_segment", || {
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            eprintln!(
                "get_root_mix_waveform_peaks_segment(track_id={}, start_sec={:.3}, duration_sec={:.3}, columns={})",
                track_id, start_sec, duration_sec, columns
            );
        }
        let tl0 = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(root) = tl0.resolve_root_track_id(&track_id) else {
            return waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            };
        };

        // Collect root + descendants.
        let mut included: std::collections::HashSet<String> = std::collections::HashSet::new();
        included.insert(root.clone());
        let mut idx = 0usize;
        let mut frontier = vec![root.clone()];
        while idx < frontier.len() {
            let cur = frontier[idx].clone();
            for child in tl0
                .tracks
                .iter()
                .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
            {
                if included.insert(child.clone()) {
                    frontier.push(child);
                }
            }
            idx += 1;
            if idx > 4096 {
                break;
            }
        }

        let mut tl = tl0.clone();
        tl.tracks.retain(|t| included.contains(&t.id));
        tl.clips.retain(|c| included.contains(&c.track_id));

        // Peaks are used as a visual background in the UI; do not hide waveforms
        // due to mixer states (mute/solo) which would otherwise result in a silent
        // mix and an invisible waveform.
        for t in &mut tl.tracks {
            t.muted = false;
            t.solo = false;
        }
        for c in &mut tl.clips {
            c.muted = false;
        }

        let cols = columns.clamp(16, 8192);
        let opts = crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec: Some(start_sec + duration_sec.max(0.0)),
            // Peaks are used as a visual timing reference. Use Signalsmith Stretch so
            // stretched clips line up with the same timing as pitch analysis.
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            // 实时预览使用默认质量（Wav16 + Realtime）。
            export_format: crate::mixdown::ExportFormat::Wav16,
            quality_preset: crate::mixdown::QualityPreset::Realtime,
        };

        let (_sr, ch, _dur, mix) = match crate::mixdown::render_mixdown_interleaved(&tl, opts) {
            Ok(v) => v,
            Err(_) => {
                return waveform::WaveformPeaksSegmentPayload {
                    ok: false,
                    min: vec![],
                    max: vec![],
                }
            }
        };

        let channels = ch.max(1) as usize;
        let frames = mix.len() / channels;
        if frames == 0 {
            return waveform::WaveformPeaksSegmentPayload {
                ok: true,
                min: vec![0.0; cols],
                max: vec![0.0; cols],
            };
        }

        let mut out_min = vec![f32::INFINITY; cols];
        let mut out_max = vec![f32::NEG_INFINITY; cols];
        for x in 0..cols {
            let i0 = (x * frames) / cols;
            let i1 = ((x + 1) * frames) / cols;
            let i1 = i1.max(i0 + 1).min(frames);
            for f in i0..i1 {
                let base = f * channels;
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += mix[base + c];
                }
                let v = sum / channels as f32;
                if v < out_min[x] {
                    out_min[x] = v;
                }
                if v > out_max[x] {
                    out_max[x] = v;
                }
            }
            if !out_min[x].is_finite() {
                out_min[x] = 0.0;
            }
            if !out_max[x].is_finite() {
                out_max[x] = 0.0;
            }
        }

        waveform::WaveformPeaksSegmentPayload {
            ok: true,
            min: out_min,
            max: out_max,
        }
    })
}

// ===================== track subtree mix waveform peaks =====================

pub(super) fn get_track_mix_waveform_peaks_segment(
    state: State<'_, AppState>,
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
) -> waveform::WaveformPeaksSegmentPayload {
    guard_waveform_command("get_track_mix_waveform_peaks_segment", || {
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            eprintln!(
                "get_track_mix_waveform_peaks_segment(track_id={}, start_sec={:.3}, duration_sec={:.3}, columns={})",
                track_id, start_sec, duration_sec, columns
            );
        }
        let tl0 = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if !tl0.tracks.iter().any(|t| t.id == track_id) {
            return waveform::WaveformPeaksSegmentPayload {
                ok: false,
                min: vec![],
                max: vec![],
            };
        }

        // Collect track + descendants.
        let mut included: std::collections::HashSet<String> = std::collections::HashSet::new();
        included.insert(track_id.clone());
        let mut idx = 0usize;
        let mut frontier = vec![track_id.clone()];
        while idx < frontier.len() {
            let cur = frontier[idx].clone();
            for child in tl0
                .tracks
                .iter()
                .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
            {
                if included.insert(child.clone()) {
                    frontier.push(child);
                }
            }
            idx += 1;
            if idx > 4096 {
                break;
            }
        }

        let mut tl = tl0.clone();
        tl.tracks.retain(|t| included.contains(&t.id));
        tl.clips.retain(|c| included.contains(&c.track_id));

        // Peaks are used as a visual background in the UI; do not hide waveforms
        // due to mixer states (mute/solo) which would otherwise result in a silent
        // mix and an invisible waveform.
        for t in &mut tl.tracks {
            t.muted = false;
            t.solo = false;
        }
        for c in &mut tl.clips {
            c.muted = false;
        }

        let cols = columns.clamp(16, 8192);
        let opts = crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec: Some(start_sec + duration_sec.max(0.0)),
            // Peaks are used as a visual timing reference. Use Signalsmith Stretch so
            // stretched clips line up with the same timing as pitch analysis.
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            // 实时预览使用默认质量（Wav16 + Realtime）。
            export_format: crate::mixdown::ExportFormat::Wav16,
            quality_preset: crate::mixdown::QualityPreset::Realtime,
        };

        let (_sr, ch, _dur, mix) = match crate::mixdown::render_mixdown_interleaved(&tl, opts) {
            Ok(v) => v,
            Err(_) => {
                return waveform::WaveformPeaksSegmentPayload {
                    ok: false,
                    min: vec![],
                    max: vec![],
                }
            }
        };

        let channels = ch.max(1) as usize;
        let frames = mix.len() / channels;
        if frames == 0 {
            return waveform::WaveformPeaksSegmentPayload {
                ok: true,
                min: vec![0.0; cols],
                max: vec![0.0; cols],
            };
        }

        let mut out_min = vec![f32::INFINITY; cols];
        let mut out_max = vec![f32::NEG_INFINITY; cols];
        for x in 0..cols {
            let i0 = (x * frames) / cols;
            let i1 = ((x + 1) * frames) / cols;
            let i1 = i1.max(i0 + 1).min(frames);
            for f in i0..i1 {
                let base = f * channels;
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += mix[base + c];
                }
                let v = sum / channels as f32;
                if v < out_min[x] {
                    out_min[x] = v;
                }
                if v > out_max[x] {
                    out_max[x] = v;
                }
            }
            if !out_min[x].is_finite() {
                out_min[x] = 0.0;
            }
            if !out_max[x].is_finite() {
                out_max[x] = 0.0;
            }
        }

        waveform::WaveformPeaksSegmentPayload {
            ok: true,
            min: out_min,
            max: out_max,
        }
    })
}
