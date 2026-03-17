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
    let mut out = Vec::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rms_windows() {
        // Empty input
        assert_eq!(
            compute_rms_windows(&Vec::<f64>::new(), 100),
            Vec::<f64>::new()
        );

        // Single window
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let rms = compute_rms_windows(&samples, 4);
        assert_eq!(rms.len(), 1);

        // RMS of [0.1, 0.2, 0.3, 0.4] = sqrt((0.01+0.04+0.09+0.16)/4) = sqrt(0.075) ≈ 0.274
        assert!((rms[0] - 0.274).abs() < 0.01);

        // Multiple windows
        let samples = vec![1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let rms = compute_rms_windows(&samples, 2);
        assert_eq!(rms.len(), 3);

        // Window 1: [1.0, 1.0] → RMS = 1.0
        assert!((rms[0] - 1.0).abs() < 0.001);
        // Window 2: [0.0, 0.0] → RMS = 0.0
        assert!((rms[1] - 0.0).abs() < 0.001);
        // Window 3: [0.5, 0.5] → RMS ≈ 0.5
        assert!((rms[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_classify_voiced_ranges() {
        // Threshold = 0.5, window_samples = 2
        let rms_windows = vec![0.1, 0.8, 0.9, 0.2, 0.7, 0.6, 0.3];
        let ranges = classify_voiced_ranges(&rms_windows, 0.5, 2);

        // Windows 1-2 (0.8, 0.9) → samples 2-6
        // Windows 4-5 (0.7, 0.6) → samples 8-12
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 2..6);
        assert_eq!(ranges[1], 8..12);

        // Empty input
        let empty: Vec<f64> = vec![];
        assert_eq!(
            classify_voiced_ranges(&empty, 0.5, 2),
            Vec::<Range<usize>>::new()
        );

        // All below threshold
        let rms_windows = vec![0.1, 0.2, 0.3];
        let ranges = classify_voiced_ranges(&rms_windows, 0.5, 2);
        assert_eq!(ranges.len(), 0);

        // All above threshold
        let rms_windows = vec![0.6, 0.7, 0.8];
        let ranges = classify_voiced_ranges(&rms_windows, 0.5, 2);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..6);
    }

    #[test]
    fn test_merge_adjacent_voiced_ranges() {
        // Gap = 5 samples, merge_threshold = 10
        let ranges = vec![0..10, 15..25, 30..40];
        let merged = merge_adjacent_voiced_ranges(ranges, 10);

        // 0..10 and 15..25 have gap=5 ≤ 10, should merge to 0..25
        // 0..25 and 30..40 have gap=5 ≤ 10, should merge to 0..40
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], 0..40);

        // Gap > threshold
        let ranges = vec![0..10, 25..35, 50..60];
        let merged = merge_adjacent_voiced_ranges(ranges, 10);

        // All gaps > 10, no merging
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0], 0..10);
        assert_eq!(merged[1], 25..35);
        assert_eq!(merged[2], 50..60);

        // Empty input
        let empty: Vec<Range<usize>> = vec![];
        assert_eq!(
            merge_adjacent_voiced_ranges(empty, 10),
            Vec::<Range<usize>>::new()
        );

        // Single range
        let ranges = vec![0..100];
        let merged = merge_adjacent_voiced_ranges(ranges, 10);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], 0..100);
    }

    #[test]
    fn test_split_into_chunks() {
        // Range 0..100, chunk_size = 30
        let chunks = split_into_chunks(0..100, 30);

        // Should produce: [0..30, 30..60, 60..90, 90..100]
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], 0..30);
        assert_eq!(chunks[1], 30..60);
        assert_eq!(chunks[2], 60..90);
        assert_eq!(chunks[3], 90..100);

        // Exact multiple
        let chunks = split_into_chunks(0..90, 30);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], 0..30);
        assert_eq!(chunks[1], 30..60);
        assert_eq!(chunks[2], 60..90);

        // chunk_size = 0 (edge case)
        let chunks = split_into_chunks(0..100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], 0..100);

        // Very small range
        let chunks = split_into_chunks(0..5, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], 0..5);
    }

    #[test]
    fn test_extend_with_context() {
        // Range 100..200, context = 50, total = 1000
        let ext = extend_with_context(100..200, 50, 1000);
        assert_eq!(ext, 50..250);

        // Left boundary clamp
        let ext = extend_with_context(20..100, 50, 1000);
        assert_eq!(ext, 0..150);

        // Right boundary clamp
        let ext = extend_with_context(900..950, 100, 1000);
        assert_eq!(ext, 800..1000);

        // Both boundaries clamp
        let ext = extend_with_context(10..30, 50, 100);
        assert_eq!(ext, 0..80);

        // Empty total_samples (edge case)
        let ext = extend_with_context(10..20, 5, 0);
        assert_eq!(ext, 0..0);
    }

    #[test]
    fn test_apply_crossfade() {
        // Two arrays with 5 frames, crossfade last 3 frames
        let current = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let next = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let blended = apply_crossfade(&current, &next, 3);

        // Should blend last 3 frames of current with first 3 frames of next
        // Frame 0 (from current[2]): t=0.0 → 1.0*(1-0) + 0.0*0 = 1.0
        // Frame 1 (from current[3]): t=0.5 → 1.0*(1-0.5) + 0.0*0.5 = 0.5
        // Frame 2 (from current[4]): t=1.0 → 1.0*(1-1) + 0.0*1 = 0.0
        assert_eq!(blended.len(), 3);
        assert!((blended[0] - 1.0).abs() < 0.001);
        assert!((blended[1] - 0.5).abs() < 0.001);
        assert!((blended[2] - 0.0).abs() < 0.001);

        // Crossfade between different values
        let current = vec![10.0, 10.0, 10.0];
        let next = vec![20.0, 20.0, 20.0];
        let blended = apply_crossfade(&current, &next, 2);

        // Frame 0: t=0.0 → 10.0
        // Frame 1: t=1.0 → 20.0
        assert_eq!(blended.len(), 2);
        assert!((blended[0] - 10.0).abs() < 0.001);
        assert!((blended[1] - 20.0).abs() < 0.001);

        // ctx_frames = 0 (edge case)
        let blended = apply_crossfade(&current, &next, 0);
        assert_eq!(blended.len(), 0);

        // Empty arrays
        let blended = apply_crossfade(&[], &[], 3);
        assert_eq!(blended.len(), 0);

        // ctx_frames > array length
        let current = vec![1.0, 2.0];
        let next = vec![3.0, 4.0];
        let blended = apply_crossfade(&current, &next, 5);

        // Should cap at min(current.len(), next.len()) = 2
        assert_eq!(blended.len(), 2);
    }

    #[test]
    fn test_integration_vad_chunking_crossfade() {
        // Simulate a complete pipeline: VAD → chunking → crossfade

        // 1. Generate synthetic audio with voiced/unvoiced regions
        let sr = 16000;
        let duration_sec = 2.0;
        let total_samples = (sr as f64 * duration_sec) as usize;

        let mut audio = vec![0.0; total_samples];
        // Voiced region: 0.5s to 1.5s
        let voiced_start = (sr as f64 * 0.5) as usize;
        let voiced_end = (sr as f64 * 1.5) as usize;
        for i in voiced_start..voiced_end {
            audio[i] = 0.5 * ((i as f64 * 2.0 * std::f64::consts::PI * 200.0 / sr as f64).sin());
        }

        // 2. VAD
        let window_samples = (sr as f64 * 0.05) as usize; // 50ms
        let rms_windows = compute_rms_windows(&audio, window_samples);
        let threshold = 0.1;
        let voiced_ranges_raw = classify_voiced_ranges(&rms_windows, threshold, window_samples);

        // 3. Merge
        let merge_gap_samples = (sr as f64 * 0.05) as usize; // 50ms
        let voiced_ranges = merge_adjacent_voiced_ranges(voiced_ranges_raw, merge_gap_samples);

        // Should detect roughly the voiced region
        assert!(voiced_ranges.len() > 0);

        // 4. Chunking
        let chunk_samples = (sr as f64 * 0.5) as usize; // 0.5s chunks
        let mut all_chunks = vec![];
        for range in &voiced_ranges {
            let chunks = split_into_chunks(range.clone(), chunk_samples);
            all_chunks.extend(chunks);
        }

        // For 1s of voiced audio with 0.5s chunks, expect 2 chunks
        assert!(all_chunks.len() >= 2);

        // 5. Context extension & crossfade simulation
        let ctx_samples = (sr as f64 * 0.1) as usize; // 0.1s context
        let mut mock_f0_results = vec![];

        for chunk in &all_chunks {
            let ext_range = extend_with_context(chunk.clone(), ctx_samples, audio.len());

            // Simulate F0 extraction (mock: constant 100 Hz)
            let chunk_duration_sec = (ext_range.end - ext_range.start) as f64 / sr as f64;
            let frame_period_ms = 5.0;
            let n_frames = (chunk_duration_sec * 1000.0 / frame_period_ms) as usize + 1;
            mock_f0_results.push(vec![100.0; n_frames]);
        }

        // 6. Crossfade merging
        if mock_f0_results.len() >= 2 {
            let ctx_frames = ctx_samples / ((sr as f64 * 0.005) as usize); // Convert to frame count
            let blended =
                apply_crossfade(&mock_f0_results[0], &mock_f0_results[1], ctx_frames.min(10));

            // Should produce smooth transition
            assert!(blended.len() > 0);
        }
    }

    #[test]
    fn test_vad_silence_detection() {
        // Pure silence should produce no voiced ranges
        let silence = vec![0.0; 10000];
        let window_samples = 100;
        let rms_windows = compute_rms_windows(&silence, window_samples);
        let ranges = classify_voiced_ranges(&rms_windows, 0.01, window_samples);
        assert_eq!(ranges.len(), 0);

        // Pure voice (all above threshold) should produce one range covering all
        let voice = vec![0.5; 10000];
        let rms_windows = compute_rms_windows(&voice, window_samples);
        let ranges = classify_voiced_ranges(&rms_windows, 0.01, window_samples);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, 10000);
    }

    #[test]
    fn test_chunking_correctness() {
        // Verify chunks cover entire range without gaps or overlaps
        let range = 0..1000;
        let chunk_samples = 300;
        let chunks = split_into_chunks(range.clone(), chunk_samples);

        // First chunk starts at range.start
        assert_eq!(chunks[0].start, 0);

        // Last chunk ends at range.end
        assert_eq!(chunks[chunks.len() - 1].end, 1000);

        // No gaps between chunks
        for i in 0..chunks.len() - 1 {
            assert_eq!(chunks[i].end, chunks[i + 1].start);
        }

        // Total coverage equals original range length
        let total: usize = chunks.iter().map(|c| c.end - c.start).sum();
        assert_eq!(total, 1000);
    }
}
