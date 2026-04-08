#![allow(dead_code)]

use std::ops::Range;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct PitchAnalysisConfig {
    pub analysis_sr: u32,
    pub silence_rms_threshold: f64,
    pub vad_merge_gap_ms: f64, // Task 4.3: Merge gap threshold
    pub chunk_sec: f64,
    pub chunk_ctx_sec: f64,
}

impl PitchAnalysisConfig {
    pub fn global() -> &'static Self {
        static CFG: OnceLock<PitchAnalysisConfig> = OnceLock::new();
        CFG.get_or_init(|| PitchAnalysisConfig {
            analysis_sr: env_u32("HIFISHIFTER_PITCH_ANALYSIS_SR").unwrap_or(16000),
            // Task 4.6: VAD RMS threshold configurable (default 0.02)
            silence_rms_threshold: env_f64("HIFISHIFTER_VAD_RMS_THRESHOLD").unwrap_or(0.02),
            // Task 4.3: Merge gap threshold (default 50ms)
            vad_merge_gap_ms: env_f64("HIFISHIFTER_VAD_MERGE_GAP_MS").unwrap_or(50.0),
            chunk_sec: env_f64("HIFISHIFTER_PITCH_CHUNK_SEC").unwrap_or(30.0),
            chunk_ctx_sec: env_f64("HIFISHIFTER_PITCH_CHUNK_CTX_SEC").unwrap_or(0.3),
        })
    }
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

pub fn compute_rms_windows(samples: &[f64], window_samples: usize) -> Vec<f64> {
    if window_samples == 0 || samples.is_empty() {
        return vec![];
    }

    let mut out: Vec<f64> = Vec::with_capacity((samples.len() / window_samples) + 1);
    let mut i = 0usize;
    while i < samples.len() {
        let end = (i + window_samples).min(samples.len());
        let mut sum_sq = 0.0f64;
        for &v in &samples[i..end] {
            let vv = v;
            sum_sq += vv * vv;
        }
        let denom = (end - i).max(1) as f64;
        out.push((sum_sq / denom).sqrt());
        i = end;
    }

    out
}

pub fn classify_voiced_ranges(
    rms_windows: &[f64],
    threshold: f64,
    window_samples: usize,
) -> Vec<Range<usize>> {
    if rms_windows.is_empty() || window_samples == 0 {
        return vec![];
    }

    let mut out: Vec<Range<usize>> = Vec::new();
    let mut i = 0usize;
    while i < rms_windows.len() {
        if rms_windows[i].is_finite() && rms_windows[i] > threshold {
            let start_win = i;
            let mut end_win = i + 1;
            while end_win < rms_windows.len()
                && rms_windows[end_win].is_finite()
                && rms_windows[end_win] > threshold
            {
                end_win += 1;
            }
            let start = start_win * window_samples;
            let end = end_win * window_samples;
            if end > start {
                out.push(start..end);
            }
            i = end_win;
        } else {
            i += 1;
        }
    }

    out
}

/// Merge adjacent voiced ranges if gap < merge_threshold_ms (Task 4.3)
pub fn merge_adjacent_voiced_ranges(
    ranges: Vec<Range<usize>>,
    merge_threshold_samples: usize,
) -> Vec<Range<usize>> {
    if ranges.is_empty() {
        return vec![];
    }

    let mut merged: Vec<Range<usize>> = Vec::new();
    let mut current = ranges[0].clone();

    for range in ranges.into_iter().skip(1) {
        let gap = range.start.saturating_sub(current.end);

        if gap <= merge_threshold_samples {
            // Merge: extend current range
            current.end = range.end;
        } else {
            // Gap too large: push current and start new
            merged.push(current);
            current = range;
        }
    }

    // Don't forget the last range
    merged.push(current);
    merged
}

pub fn split_into_chunks(range: Range<usize>, chunk_samples: usize) -> Vec<Range<usize>> {
    if chunk_samples == 0 {
        return vec![range];
    }

    let len = range.end.saturating_sub(range.start);
    let capacity = if len == 0 { 1 } else { (len + chunk_samples - 1) / chunk_samples };
    let mut out = Vec::with_capacity(capacity);
    
    let mut start = range.start;
    let end = range.end;
    while start < end {
        let next = (start + chunk_samples).min(end);
        if next > start {
            out.push(start..next);
        }
        start = next;
    }
    if out.is_empty() {
        out.push(range);
    }
    out
}

pub fn extend_with_context(
    range: Range<usize>,
    ctx_samples: usize,
    total_samples: usize,
) -> Range<usize> {
    if total_samples == 0 {
        return 0..0;
    }
    let start = range.start.saturating_sub(ctx_samples);
    let end = (range.end + ctx_samples).min(total_samples).max(start + 1);
    start..end
}

pub fn apply_crossfade(current: &[f64], next: &[f64], ctx_frames: usize) -> Vec<f64> {
    if ctx_frames == 0 || current.is_empty() || next.is_empty() {
        return vec![];
    }

    let fade = ctx_frames.min(current.len()).min(next.len());
    if fade == 0 {
        return vec![];
    }

    let start = current.len().saturating_sub(fade);
    let mut out = Vec::with_capacity(fade);
    for i in 0..fade {
        let t = if fade <= 1 {
            1.0
        } else {
            i as f64 / (fade as f64 - 1.0)
        };
        let a = current[start + i];
        let b = next[i];
        out.push(a * (1.0 - t) + b * t);
    }
    out
}
