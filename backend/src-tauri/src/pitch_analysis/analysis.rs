// pitch_analysis::analysis — 核心分析流水线
// 包含：快照/diff、单 clip 分析、并行多 clip 分析、音高曲线合成。

#![allow(dead_code)]

use crate::state::{AppState, Clip, PitchAnalysisAlgo, TimelineState};
use std::collections::HashSet;
use std::path::Path;

use super::{
    build_root_pitch_key, hz_to_midi, resample_curve_linear, PitchJob,
};

fn build_root_mix_timeline(tl: &TimelineState, root_track_id: &str) -> TimelineState {
    // Collect root + descendants.
    let mut included: HashSet<String> = HashSet::new();
    included.insert(root_track_id.to_string());
    let mut idx = 0usize;
    let mut frontier = vec![root_track_id.to_string()];
    while idx < frontier.len() {
        let cur = frontier[idx].clone();
        for child in tl
            .tracks
            .iter()
            .filter(|t| t.parent_id.as_deref() == Some(cur.as_str()))
            .map(|t| t.id.clone())
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

    let mut out = tl.clone();
    out.tracks.retain(|t| included.contains(&t.id));
    out.clips.retain(|c| included.contains(&c.track_id));

    // Background waveform ignores mute/solo; pitch analysis should match that.
    for t in &mut out.tracks {
        t.muted = false;
        t.solo = false;
    }
    for c in &mut out.clips {
        c.muted = false;
    }

    // Avoid cloning large curve buffers into the job.
    out.params_by_root_track.clear();
    out
}

/// Build a timeline snapshot for incremental refresh detection
///
/// Generates a snapshot of the current timeline state, capturing:
/// - Cache keys for all clips (to detect parameter changes)
/// - BPM and frame period (to detect global parameter changes)
///
/// This snapshot can be compared with previous snapshots to determine
/// which clips need re-analysis.
fn build_timeline_snapshot(
    clips: &[Clip],
    bpm: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    algo: &PitchAnalysisAlgo,
) -> crate::state::TimelineSnapshot {
    use std::collections::HashMap;

    let mut clip_keys = HashMap::new();

    for clip in clips {
        if let Some(source_path) = &clip.source_path {
            let (file_size, file_mtime) =
                crate::clip_pitch_cache::get_file_signature(std::path::Path::new(source_path));

            let key_data = crate::clip_pitch_cache::ClipCacheKey {
                source_path: source_path.clone(),
                file_size,
                file_mtime,
                algo: match algo {
                    PitchAnalysisAlgo::WorldDll => "world_dll",
                    PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx",
                    PitchAnalysisAlgo::VocalShifterVslib => "vslib",
                    PitchAnalysisAlgo::None => "none",
                    PitchAnalysisAlgo::Unknown => "unknown",
                }
                .to_string(),
                f0_floor: crate::clip_pitch_cache::quantize_f64(f0_floor, 10.0),
                f0_ceil: crate::clip_pitch_cache::quantize_f64(f0_ceil, 10.0),
                version: crate::clip_pitch_cache::CACHE_FORMAT_VERSION,
            };

            let cache_key = crate::clip_pitch_cache::generate_clip_cache_key(&key_data);
            clip_keys.insert(clip.id.clone(), cache_key);
        }
    }

    crate::state::TimelineSnapshot {
        clips: clip_keys,
        bpm,
        frame_period_ms,
    }
}

/// Detected change types for incremental refresh
#[derive(Debug, Clone, PartialEq, Eq)]
enum ClipChangeType {
    Added,     // New clip in timeline
    Modified,  // Existing clip with changed parameters
    Deleted,   // Clip removed from timeline
    Unchanged, // Clip exists with same cache key
}

/// Result of comparing two timeline snapshots
#[derive(Debug)]
struct SnapshotComparison {
    added_clip_ids: Vec<String>,
    modified_clip_ids: Vec<String>,
    deleted_clip_ids: Vec<String>,
    unchanged_clip_ids: Vec<String>,
}

/// Compare two timeline snapshots to detect changes
///
/// Returns a comparison result indicating which clips were added, modified,
/// deleted, or unchanged. This enables incremental refresh by only re-analyzing
/// clips that have actually changed.
///
/// # Change Detection Rules
/// - **Added**: Clip ID exists in new snapshot but not in old
/// - **Modified**: Clip ID exists in both, but cache key differs
/// - **Deleted**: Clip ID exists in old snapshot but not in new
/// - **Unchanged**: Clip ID and cache key are identical in both snapshots
///
/// Note: Position-only changes (start_sec) do NOT affect the cache key,
/// so moving a clip without changing its content will not trigger re-analysis.
fn compare_snapshots(
    old_snapshot: Option<&crate::state::TimelineSnapshot>,
    new_snapshot: &crate::state::TimelineSnapshot,
) -> SnapshotComparison {
    use std::collections::HashSet;

    // If no old snapshot exists, all clips are "added" (first analysis)
    let Some(old) = old_snapshot else {
        return SnapshotComparison {
            added_clip_ids: new_snapshot.clips.keys().cloned().collect(),
            modified_clip_ids: Vec::new(),
            deleted_clip_ids: Vec::new(),
            unchanged_clip_ids: Vec::new(),
        };
    };

    // Check for global parameter changes (BPM or frame period)
    let global_params_changed = (old.bpm - new_snapshot.bpm).abs() > 1e-6
        || (old.frame_period_ms - new_snapshot.frame_period_ms).abs() > 1e-6;

    let old_ids: HashSet<&String> = old.clips.keys().collect();
    let new_ids: HashSet<&String> = new_snapshot.clips.keys().collect();

    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut deleted = Vec::new();
    let mut unchanged = Vec::new();

    // Detect added and modified clips
    for (clip_id, new_key) in &new_snapshot.clips {
        if let Some(old_key) = old.clips.get(clip_id) {
            // Clip exists in both snapshots
            if old_key != new_key || global_params_changed {
                modified.push(clip_id.clone());
            } else {
                unchanged.push(clip_id.clone());
            }
        } else {
            // Clip is new
            added.push(clip_id.clone());
        }
    }

    // Detect deleted clips
    for clip_id in &old_ids {
        if !new_ids.contains(clip_id) {
            deleted.push((*clip_id).clone());
        }
    }

    SnapshotComparison {
        added_clip_ids: added,
        modified_clip_ids: modified,
        deleted_clip_ids: deleted,
        unchanged_clip_ids: unchanged,
    }
}

/// Determine which clips need analysis based on incremental refresh
///
/// This function implements incremental refresh by comparing the current timeline
/// with the last snapshot. Only clips that have been added or modified need re-analysis.
///
/// # Parameters
/// - `clips`: All clips in the current timeline
/// - `old_snapshot`: Previous timeline snapshot (if any)
/// - `new_snapshot`: Current timeline snapshot
///
/// # Returns
/// A tuple of (clips_to_analyze, unchanged_clip_ids):
/// - `clips_to_analyze`: Clips that need analysis (added + modified)
/// - `unchanged_clip_ids`: Clips that can be loaded from cache
fn determine_clips_to_analyze<'a>(
    clips: &'a [Clip],
    old_snapshot: Option<&crate::state::TimelineSnapshot>,
    new_snapshot: &crate::state::TimelineSnapshot,
) -> (Vec<&'a Clip>, Vec<String>) {
    let comparison = compare_snapshots(old_snapshot, new_snapshot);

    // Clips needing analysis: added + modified
    let need_analysis_ids: std::collections::HashSet<String> = comparison
        .added_clip_ids
        .into_iter()
        .chain(comparison.modified_clip_ids.into_iter())
        .collect();

    // Filter clips that need analysis
    let clips_to_analyze: Vec<&Clip> = clips
        .iter()
        .filter(|clip| need_analysis_ids.contains(&clip.id))
        .collect();

    (clips_to_analyze, comparison.unchanged_clip_ids)
}

pub(crate) fn build_pitch_job(tl: &TimelineState, root_track_id: &str) -> Option<PitchJob> {
    let fp = tl.frame_period_ms();
    let target = tl.target_param_frames(fp);

    let (compose_enabled, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::Unknown));
    if !compose_enabled {
        return None;
    }
    if matches!(algo, PitchAnalysisAlgo::None) {
        return None;
    }

    let key = build_root_pitch_key(tl, root_track_id);

    // If already up-to-date, do nothing.
    let is_up_to_date = tl
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.pitch_orig_key.as_deref() == Some(&key) && e.pitch_orig.len() == target)
        .unwrap_or(false);
    if is_up_to_date {
        return None;
    }

    let mix_timeline = build_root_mix_timeline(tl, root_track_id);

    Some(PitchJob {
        root_track_id: root_track_id.to_string(),
        key,
        frame_period_ms: fp,
        target_frames: target,
        algo,
        timeline: mix_timeline,
    })
}

/// Analyze a single clip's pitch curve with caching support
///
/// This function checks the cache first, and only performs expensive F0 analysis
/// if the result is not cached. Results are stored in the cache for future use.
///
/// # Returns
/// - `Ok(Arc<Vec<f32>>)`: MIDI pitch curve (unvoiced frames = 0.0)
/// - `Err(String)`: Error message if analysis fails
#[allow(clippy::too_many_arguments)]
fn analyze_clip_with_cache(
    clip: &Clip,
    _bpm: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    algo: &PitchAnalysisAlgo,
    cache: &std::sync::Arc<std::sync::Mutex<crate::clip_pitch_cache::ClipPitchCache>>,
    debug: bool,
) -> Result<std::sync::Arc<Vec<f32>>, String> {
    let Some(source_path) = clip.source_path.as_ref() else {
        return Err("No source path".to_string());
    };

    eprintln!(
        "[pitch:analyze] clip_id={} source={} duration_sec={:?} fp={:.1}ms algo={:?}",
        clip.id, source_path, clip.duration_sec, frame_period_ms, algo,
    );

    // Generate cache key
    let (file_size, file_mtime) =
        crate::clip_pitch_cache::get_file_signature(Path::new(source_path));

    let algo_str = match algo {
        PitchAnalysisAlgo::WorldDll => "world_dll",
        PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx",
        PitchAnalysisAlgo::VocalShifterVslib => "vslib",
        PitchAnalysisAlgo::None => "none",
        PitchAnalysisAlgo::Unknown => "unknown",
    };

    let key_data = crate::clip_pitch_cache::ClipCacheKey {
        source_path: source_path.clone(),
        file_size,
        file_mtime,
        algo: algo_str.to_string(),
        f0_floor: crate::clip_pitch_cache::quantize_f64(f0_floor, 10.0),
        f0_ceil: crate::clip_pitch_cache::quantize_f64(f0_ceil, 10.0),
        version: crate::clip_pitch_cache::CACHE_FORMAT_VERSION,
    };

    let cache_key = crate::clip_pitch_cache::generate_clip_cache_key(&key_data);

    // Query cache
    {
        let mut cache_guard = cache
            .lock()
            .map_err(|e| format!("Cache lock error: {}", e))?;
        if let Some(cached) = cache_guard.get(&cache_key) {
            if debug {
                eprintln!(
                    "clip_pitch_cache: HIT for clip_id={} key={}",
                    clip.id,
                    &cache_key[..16]
                );
            }
            return Ok(cached);
        }
        if debug {
            eprintln!(
                "clip_pitch_cache: MISS for clip_id={} key={}",
                clip.id,
                &cache_key[..16]
            );
        }
    }

    // Decode audio
    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path))
            .map_err(|e| format!("Failed to decode audio: {}", e))?;

    let in_channels_usize = (in_channels as usize).max(1);
    let in_frames = pcm.len() / in_channels_usize;
    if in_frames < 2 {
        return Err("Audio too short".to_string());
    }

    // 全量分析策略：分析完整源音频，trim/rate 在组装阶段处理
    // Resample 全量 PCM 到 44100 Hz
    let segment =
        crate::mixdown::linear_resample_interleaved(&pcm, in_channels_usize, in_rate, 44100);
    let seg_frames = segment.len() / in_channels_usize;
    if seg_frames < 2 {
        return Err("Resampled audio too short".to_string());
    }

    // Convert to mono
    let mut mono_raw: Vec<f64> = segment
        .chunks_exact(in_channels_usize)
        .map(|chunk| (chunk.iter().sum::<f32>() as f64) / (in_channels_usize as f64))
        .collect();

    // Preprocess: remove DC and normalize
    let mut mean = 0.0f64;
    for &v in &mono_raw {
        mean += v;
    }
    mean /= mono_raw.len().max(1) as f64;

    let mut max_abs = 0.0f64;
    for &v in &mono_raw {
        let a = (v - mean).abs();
        if a.is_finite() && a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs.is_finite() && max_abs > 1.0 {
        (1.0 / max_abs).clamp(0.0, 1.0)
    } else {
        1.0
    };

    for v in &mut mono_raw {
        *v = ((*v - mean) * scale).clamp(-1.0, 1.0);
    }
    let mono = mono_raw;

    // Compute F0 using FCPE ONNX.
    let f0_hz = crate::fcpe_onnx::infer_f0_hz(&mono, 44100, frame_period_ms, f0_floor, f0_ceil)
        .map_err(|e| format!("F0 analysis failed: {e}"))?;

    if f0_hz.len() < 2 {
        return Err("F0 analysis returned too few frames".to_string());
    }

    // Convert Hz to MIDI
    let mut midi: Vec<f32> = Vec::with_capacity(f0_hz.len());
    for hz in f0_hz {
        midi.push(hz_to_midi(hz));
    }

    if debug {
        eprintln!(
            "clip_pitch_cache: ANALYZED clip_id={} midi_len={} key={}",
            clip.id,
            midi.len(),
            &cache_key[..16]
        );
    }

    // Store in cache
    let result = std::sync::Arc::new(midi);
    {
        let mut cache_guard = cache
            .lock()
            .map_err(|e| format!("Cache lock error: {}", e))?;
        cache_guard.put(cache_key, std::sync::Arc::clone(&result));
    }

    Ok(result)
}

/// Parallel analysis result for a single clip
#[derive(Debug, Clone)]
struct ClipAnalysisResult {
    clip_id: String,
    clip_start_sec: f64,
    clip_end_sec: f64,
    pre_silence_sec: f64,
    clip_total_frames: usize,
    midi: std::sync::Arc<Vec<f32>>,
    track_gain_value: f32,
    was_cache_hit: bool,
}

/// Helper function to process a single clip with cache and progress tracking
///
/// This is extracted from compute_pitch_curve_parallel to allow both
/// parallel (ONNX) and serial (WORLD) processing with the same logic.
#[allow(clippy::too_many_arguments)]
fn process_single_clip(
    clip: &Clip,
    tracks_gain: &std::collections::HashMap<String, f32>,
    bpm: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    algo: &PitchAnalysisAlgo,
    cache: &std::sync::Arc<std::sync::Mutex<crate::clip_pitch_cache::ClipPitchCache>>,
    tracker: Option<&std::sync::Arc<crate::pitch_progress::ProgressTracker>>,
    debug: bool,
    duration_sec: f64,
    _bs: f64, // Beat duration in seconds (kept for signature compatibility)
) -> Result<ClipAnalysisResult, String> {
    let clip_start_sec = clip.start_sec.max(0.0);
    let clip_timeline_len_sec = clip.length_sec.max(0.0);
    let clip_end_sec = clip_start_sec + clip_timeline_len_sec;

    let track_gain_value = tracks_gain.get(&clip.track_id).copied().unwrap_or(1.0);

    eprintln!(
        "[pitch:process] clip_id={} start={:.3}s len={:.3}s src_start={:.3}s src_end={:.3}s pr={:.2} track_gain={:.2}",
        clip.id, clip_start_sec, clip_timeline_len_sec,
        clip.source_start_sec, clip.source_end_sec,
        clip.playback_rate, track_gain_value,
    );

    // Check if clip has valid source
    let Some(_source_path) = clip.source_path.as_ref() else {
        if debug {
            eprintln!("  clip {} skipped: no source path", clip.id);
        }
        return Err(format!("Clip {} has no source path", clip.id));
    };

    // Check cache before analysis
    let was_cache_hit = {
        let (file_size, file_mtime) =
            crate::clip_pitch_cache::get_file_signature(std::path::Path::new(_source_path));
        let key_data = crate::clip_pitch_cache::ClipCacheKey {
            source_path: _source_path.clone(),
            file_size,
            file_mtime,
            algo: match algo {
                PitchAnalysisAlgo::WorldDll => "world_dll",
                PitchAnalysisAlgo::NsfHifiganOnnx => "nsf_hifigan_onnx",
                PitchAnalysisAlgo::VocalShifterVslib => "vslib",
                PitchAnalysisAlgo::None => "none",
                PitchAnalysisAlgo::Unknown => "unknown",
            }
            .to_string(),
            f0_floor: crate::clip_pitch_cache::quantize_f64(f0_floor, 10.0),
            f0_ceil: crate::clip_pitch_cache::quantize_f64(f0_ceil, 10.0),
            version: crate::clip_pitch_cache::CACHE_FORMAT_VERSION,
        };
        let cache_key = crate::clip_pitch_cache::generate_clip_cache_key(&key_data);

        if let Ok(mut guard) = cache.lock() {
            guard.get(&cache_key).is_some()
        } else {
            false
        }
    };

    // Analyze clip (with caching)
    // 在分析开始前，通知 tracker 当前正在处理?clip
    if let Some(tracker) = tracker {
        tracker.set_current_clip(Some(clip.name.clone()));
    }

    let midi_result = analyze_clip_with_cache(
        clip,
        bpm,
        frame_period_ms,
        f0_floor,
        f0_ceil,
        algo,
        cache,
        debug,
    );

    // Update progress
    if let Some(tracker) = tracker {
        let progress = tracker.report_clip_completed(duration_sec, was_cache_hit);
        // 分析完成后清除当?clip 名称
        tracker.set_current_clip(None);
        if debug {
            eprintln!(
                "  clip {} completed (cache_hit={}), overall progress={:.1}%",
                clip.id,
                was_cache_hit,
                progress * 100.0
            );
        }
    }

    // Handle result
    match midi_result {
        Ok(full_midi) => {
            // 全量分析策略：缓存中是全量源音频曲线，
            // 需要做 trim+resample 转换为 timeline 对齐的曲线
            let playback_rate = if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
                clip.playback_rate as f64
            } else {
                1.0
            };

            let midi = std::sync::Arc::new(crate::pitch_clip::trim_and_resample_midi(
                &full_midi,
                frame_period_ms,
                clip.source_start_sec,
                clip.source_end_sec,
                playback_rate,
                clip_timeline_len_sec,
            ));

            // Calculate pre_silence_sec for clip placement
            let pre_silence_sec_src = (-clip.source_start_sec).max(0.0);
            let pre_silence_sec = pre_silence_sec_src / playback_rate.max(1e-6);

            // Estimate clip_total_frames (from original audio)
            let clip_total_frames = if let Some(dur) = clip.duration_sec {
                let in_rate = 44100.0; // Assuming standard rate
                (dur * in_rate).round().max(0.0) as usize
            } else {
                // Fallback: use midi length as approximation
                midi.len() * (frame_period_ms / 1000.0 * 44100.0) as usize
            };

            Ok(ClipAnalysisResult {
                clip_id: clip.id.clone(),
                clip_start_sec,
                clip_end_sec,
                pre_silence_sec,
                clip_total_frames,
                midi,
                track_gain_value,
                was_cache_hit,
            })
        }
        Err(e) => {
            if debug {
                eprintln!("  clip {} failed: {}", clip.id, e);
            }
            Err(format!("Clip {}: {}", clip.id, e))
        }
    }
}

/// Parallel pitch analysis for multiple clips
///
/// This function analyzes multiple clips in parallel using rayon, with caching support
/// and progress tracking. It returns results for all successfully analyzed clips,
/// even if some clips fail.
///
/// # Parameters
/// - `clips`: List of clips to analyze
/// - `bpm`: Project BPM
/// - `frame_period_ms`: Analysis frame period in milliseconds
/// - `f0_floor`: F0 floor frequency (Hz)
/// - `f0_ceil`: F0 ceiling frequency (Hz)
/// - `algo`: Analysis algorithm
/// - `cache`: Clip pitch cache
/// - `tracker`: Progress tracker (optional for progress updates)
/// - `debug`: Enable debug logging
///
/// # Returns
/// - `Ok(Vec<ClipAnalysisResult>)`: Successfully analyzed clips (may be partial)
/// - `Err(String)`: Critical failure (>50% clips failed)
#[allow(clippy::too_many_arguments)]
fn compute_pitch_curve_parallel(
    clips: &[Clip],
    tracks_gain: &std::collections::HashMap<String, f32>,
    bpm: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    algo: &PitchAnalysisAlgo,
    cache: &std::sync::Arc<std::sync::Mutex<crate::clip_pitch_cache::ClipPitchCache>>,
    tracker: Option<&std::sync::Arc<crate::pitch_progress::ProgressTracker>>,
    debug: bool,
) -> Result<Vec<ClipAnalysisResult>, String> {
    use rayon::prelude::*;

    if clips.is_empty() {
        return Ok(Vec::new());
    }

    let bs = 60.0 / bpm.max(1e-6);

    // Separate clips by algorithm: WORLD requires serial processing due to world_dll_mutex
    let (world_clips, onnx_clips): (Vec<&Clip>, Vec<&Clip>) = clips
        .iter()
        .partition(|_clip| matches!(algo, PitchAnalysisAlgo::WorldDll));

    if debug {
        eprintln!(
            "compute_pitch_curve_parallel: {} total clips ({} WORLD, {} ONNX/other)",
            clips.len(),
            world_clips.len(),
            onnx_clips.len()
        );
    }

    let mut all_results: Vec<Result<ClipAnalysisResult, String>> = Vec::new();

    // Process ONNX clips in parallel (no locking constraints)
    if !onnx_clips.is_empty() {
        if debug {
            eprintln!(
                "  Processing {} ONNX clips in parallel...",
                onnx_clips.len()
            );
        }

        // Sort by workload descending for better load balancing
        let mut onnx_sorted: Vec<(&Clip, f64)> = onnx_clips
            .iter()
            .map(|clip| {
                let duration_sec = clip.length_sec.max(0.0);
                (*clip, duration_sec)
            })
            .collect();

        onnx_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let onnx_results: Vec<Result<ClipAnalysisResult, String>> = onnx_sorted
            .par_iter()
            .map(|(clip, duration_sec)| {
                process_single_clip(
                    clip,
                    tracks_gain,
                    bpm,
                    frame_period_ms,
                    f0_floor,
                    f0_ceil,
                    algo,
                    cache,
                    tracker,
                    debug,
                    *duration_sec,
                    bs,
                )
            })
            .collect();

        all_results.extend(onnx_results);
    }

    // Process WORLD clips serially (due to world_dll_mutex)
    if !world_clips.is_empty() {
        if debug {
            eprintln!("  Processing {} WORLD clips serially...", world_clips.len());
        }

        // Sort by workload descending (not as critical for serial, but consistent)
        let mut world_sorted: Vec<(&Clip, f64)> = world_clips
            .iter()
            .map(|clip| {
                let duration_sec = clip.length_sec.max(0.0);
                (*clip, duration_sec)
            })
            .collect();
        world_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (clip, duration_sec) in world_sorted {
            let result = process_single_clip(
                &clip,
                tracks_gain,
                bpm,
                frame_period_ms,
                f0_floor,
                f0_ceil,
                algo,
                cache,
                tracker,
                debug,
                duration_sec,
                bs,
            );
            all_results.push(result);
        }
    }

    // Separate successes and failures
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for result in all_results {
        match result {
            Ok(clip_result) => successes.push(clip_result),
            Err(e) => failures.push(e),
        }
    }

    if debug {
        eprintln!(
            "compute_pitch_curve_parallel: {} successes, {} failures",
            successes.len(),
            failures.len()
        );
    }

    // Check failure rate
    let total = successes.len() + failures.len();
    if total > 0 {
        let failure_rate = failures.len() as f64 / total as f64;
        if failure_rate > 0.5 {
            return Err(format!(
                "Critical failure: {}/{} clips failed (>{:.0}%). Errors: {}",
                failures.len(),
                total,
                failure_rate * 100.0,
                failures.join("; ")
            ));
        }
    }

    // Return partial results (even if some clips failed, as long as <50% failed)
    if !failures.is_empty() && debug {
        eprintln!(
            "  Warning: {} clips failed but continuing with {} successes",
            failures.len(),
            successes.len()
        );
    }

    Ok(successes)
}

/// Incremental pitch analysis with caching and parallel processing
///
/// This function implements the full incremental refresh workflow:
/// 1. Query previous snapshot from state
/// 2. Generate current snapshot
/// 3. Compare snapshots to identify changed clips
/// 4. Analyze only changed clips in parallel
/// 5. Load unchanged clips from cache
/// 6. Merge results
/// 7. Update snapshot in state
///
/// # Parameters
/// - `state`: AppState containing cache and snapshot storage
/// - `root_track_id`: Root track being analyzed
/// - `clips`: All clips in the timeline
/// - `tracks_gain`: Track gain values
/// - `bpm`: Project BPM
/// - `frame_period_ms`: Analysis frame period
/// - `f0_floor`: F0 floor frequency
/// - `f0_ceil`: F0 ceiling frequency
/// - `algo`: Analysis algorithm
/// - `debug`: Enable debug logging
///
/// # Returns
/// - `Ok((results, new_snapshot))`: Analysis results and updated snapshot
/// - `Err(String)`: Critical failure
#[allow(clippy::too_many_arguments)]
fn compute_pitch_curve_with_incremental_refresh(
    state: &AppState,
    root_track_id: &str,
    clips: &[Clip],
    tracks_gain: &std::collections::HashMap<String, f32>,
    bpm: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    algo: &PitchAnalysisAlgo,
    debug: bool,
) -> Result<(Vec<ClipAnalysisResult>, crate::state::TimelineSnapshot), String> {
    let cache = &state.clip_pitch_cache;

    // Task 9.1: Query previous snapshot
    let old_snapshot = if let Ok(snapshot_map) = state.pitch_timeline_snapshot.lock() {
        snapshot_map.get(root_track_id).cloned()
    } else {
        None
    };

    // Task 9.2: Generate current snapshot
    let new_snapshot =
        build_timeline_snapshot(clips, bpm, frame_period_ms, f0_floor, f0_ceil, algo);

    // Task 9.3: Compare snapshots to identify changes
    let (clips_to_analyze, unchanged_clip_ids) =
        determine_clips_to_analyze(clips, old_snapshot.as_ref(), &new_snapshot);

    if debug {
        eprintln!(
            "Incremental refresh: {} clips need analysis, {} unchanged (cached)",
            clips_to_analyze.len(),
            unchanged_clip_ids.len()
        );
    }

    let mut all_results = Vec::new();

    // Task 9.4: Analyze only changed clips in parallel
    if !clips_to_analyze.is_empty() {
        // 统一克隆一次，避免将 Clip 结构体重复深拷贝两次引发的卡顿
        let cloned_clips: Vec<Clip> = clips_to_analyze.iter().map(|&c| c.clone()).collect();

        // Create progress tracker for changed clips only
        let tracker = std::sync::Arc::new(crate::pitch_progress::ProgressTracker::new(
            &cloned_clips,
            bpm,
            cache,
        ));

        let analyzed_results = compute_pitch_curve_parallel(
            &cloned_clips,
            tracks_gain,
            bpm,
            frame_period_ms,
            f0_floor,
            f0_ceil,
            algo,
            cache,
            Some(&tracker),
            debug,
        )?;

        all_results.extend(analyzed_results);
    }

    // Task 9.5: Load unchanged clips from cache
    for clip_id in &unchanged_clip_ids {
        let Some(clip) = clips.iter().find(|c| &c.id == clip_id) else {
            continue;
        };

        // Try to load from cache
        let cache_result = analyze_clip_with_cache(
            clip,
            bpm,
            frame_period_ms,
            f0_floor,
            f0_ceil,
            algo,
            cache,
            debug,
        );

        if let Ok(full_midi) = cache_result {
            let clip_start_sec = clip.start_sec.max(0.0);
            let clip_timeline_len_sec = clip.length_sec.max(0.0);
            let clip_end_sec = clip_start_sec + clip_timeline_len_sec;

            let track_gain_value = tracks_gain.get(&clip.track_id).copied().unwrap_or(1.0);

            let pre_silence_sec_src = (-clip.source_start_sec).max(0.0);
            let playback_rate = if clip.playback_rate.is_finite() && clip.playback_rate > 0.0 {
                clip.playback_rate as f64
            } else {
                1.0
            };
            let pre_silence_sec = pre_silence_sec_src / playback_rate.max(1e-6);

            // 全量分析策略：缓存中是全量源音频曲线，做 trim+resample
            let midi = std::sync::Arc::new(crate::pitch_clip::trim_and_resample_midi(
                &full_midi,
                frame_period_ms,
                clip.source_start_sec,
                clip.source_end_sec,
                playback_rate,
                clip_timeline_len_sec,
            ));

            let clip_total_frames = if let Some(dur) = clip.duration_sec {
                let in_rate = 44100.0;
                (dur * in_rate).round().max(0.0) as usize
            } else {
                midi.len() * (frame_period_ms / 1000.0 * 44100.0) as usize
            };

            all_results.push(ClipAnalysisResult {
                clip_id: clip.id.clone(),
                clip_start_sec,
                clip_end_sec,
                pre_silence_sec,
                clip_total_frames,
                midi,
                track_gain_value,
                was_cache_hit: true,
            });
        }
    }

    // Task 9.6: Results are already merged in all_results
    if debug {
        eprintln!(
            "Incremental refresh complete: {} total results ({} analyzed, {} cached)",
            all_results.len(),
            clips_to_analyze.len(),
            unchanged_clip_ids.len()
        );
    }

    // Task 9.7: Update snapshot (will be done in caller)
    Ok((all_results, new_snapshot))
}

// Helper functions for pitch analysis business logic

fn beat_sec(bpm: f64) -> f64 {
    60.0 / bpm.max(1e-6)
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Calculate clip weight at a given frame, accounting for fades and gain
#[allow(clippy::too_many_arguments)]
fn clip_weight_at_frame(
    clip: &Clip,
    bpm: f64,
    sample_rate: u32,
    _clip_start_sec: f64,
    pre_silence_sec: f64,
    clip_total_frames: usize,
    local_in_clip_frames: usize,
    track_gain_value: f32,
) -> f32 {
    let _bs = beat_sec(bpm);
    let gain = (clip.gain.max(0.0) * track_gain_value).clamp(0.0, 4.0);
    if gain <= 0.0 {
        return 0.0;
    }

    let fade_in_frames = (clip.fade_in_sec.max(0.0) * sample_rate as f64)
        .round()
        .max(0.0) as usize;
    let fade_out_frames = (clip.fade_out_sec.max(0.0) * sample_rate as f64)
        .round()
        .max(0.0) as usize;
    let pre_silence_frames = (pre_silence_sec * sample_rate as f64).round().max(0.0) as usize;
    let local_in_clip = pre_silence_frames.saturating_add(local_in_clip_frames);
    if local_in_clip >= clip_total_frames {
        return 0.0;
    }

    let mut g = gain;
    if fade_in_frames > 0 && local_in_clip < fade_in_frames {
        g *= (local_in_clip as f32 / fade_in_frames as f32).clamp(0.0, 1.0);
    }
    if fade_out_frames > 0 && local_in_clip + fade_out_frames > clip_total_frames {
        let remain = clip_total_frames.saturating_sub(local_in_clip);
        g *= (remain as f32 / fade_out_frames as f32).clamp(0.0, 1.0);
    }

    // Also drop weight before the audible segment start (pre_silence).
    if local_in_clip < pre_silence_frames {
        g = 0.0;
    }

    // Prevent pathological values.
    g.clamp(0.0, 4.0)
}

/// Optimized pitch curve fusion with coverage table
///
/// This function fuses multiple clip pitch curves into a single root curve using
/// a coverage table for O(1) frame lookups. Implements winner-take-most algorithm
/// with hysteresis to avoid rapid switching.
///
/// # Optimizations
/// - Pre-builds coverage table (O(N×M)) to identify active clips per frame
/// - Fast-path for empty frames (no clips): write 0.0 directly
/// - Fast-path for single-clip frames: skip weight calculation
/// - Only computes weights for multi-clip overlaps
/// - Maintains hysteresis for smooth transitions
///
/// # Parameters
/// - `clip_results`: Analysis results from parallel/incremental processing
/// - `clips`: Original clip data (for weight calculation)
/// - `target_frames`: Output curve length
/// - `frame_period_ms`: Frame period in milliseconds  
/// - `bpm`: Project BPM
/// - `debug`: Enable debug logging
///
/// # Returns
/// Fused MIDI pitch curve (0.0 for unvoiced frames)
#[allow(clippy::too_many_arguments)]
fn fuse_clip_pitches_optimized(
    clip_results: &[ClipAnalysisResult],
    clips: &[Clip],
    target_frames: usize,
    frame_period_ms: f64,
    bpm: f64,
    debug: bool,
) -> Vec<f32> {
    if clip_results.is_empty() {
        return vec![0.0; target_frames];
    }

    let mut out = vec![0.0f32; target_frames];

    // 预先抽取 Clip 引用，消灭帧循环内部的 O(N) 字符串 ID 查找
    let resolved_clips: Vec<Option<&Clip>> = clip_results
        .iter()
        .map(|r| clips.iter().find(|c| c.id == r.clip_id))
        .collect();

    // 预计算边界，避免每帧重复计算 float 乘除
    struct ClipBounds {
        start_f: usize,
        end_f: usize,
    }
    let clip_bounds: Vec<ClipBounds> = clip_results
        .iter()
        .map(|r| ClipBounds {
            start_f: ((r.clip_start_sec * 1000.0) / frame_period_ms)
                .round()
                .max(0.0) as usize,
            end_f: ((r.clip_end_sec * 1000.0) / frame_period_ms)
                .round()
                .max(0.0) as usize,
        })
        .collect();

    // 使用 usize 索引代替 String 进行堆内存分配
    let mut last_winner_idx: Option<usize> = None;
    let hysteresis_ratio: f32 = 1.10;

    // 砍掉会造成严重内存碎片的 Coverage Table
    for frame_idx in 0..target_frames {
        let abs_time_sec = (frame_idx as f64) * frame_period_ms / 1000.0;

        let mut active_count = 0;
        let mut last_active_idx = 0;

        // 判定当前帧激活的 Clip
        for (clip_idx, bounds) in clip_bounds.iter().enumerate() {
            if frame_idx >= bounds.start_f && frame_idx < bounds.end_f {
                active_count += 1;
                last_active_idx = clip_idx;
            }
        }

        if active_count == 0 {
            out[frame_idx] = 0.0;
            continue;
        }

        // Fast-path for single-clip frames
        if active_count == 1 {
            let clip_idx = last_active_idx;
            let result = &clip_results[clip_idx];
            let local_sec = abs_time_sec - result.clip_start_sec;
            let local_frame = ((local_sec * 1000.0) / frame_period_ms).round().max(0.0) as usize;

            if let Some(&pitch) = result.midi.get(local_frame) {
                if pitch.is_finite() && pitch > 0.0 {
                    out[frame_idx] = pitch;
                    last_winner_idx = Some(clip_idx);
                    continue;
                }
            }
            out[frame_idx] = 0.0;
            continue;
        }

        // Multi-clip overlap - full winner-take-most
        let mut best_idx: Option<usize> = None;
        let mut best_weight: f32 = 0.0;
        let mut best_pitch: f32 = 0.0;

        for (clip_idx, bounds) in clip_bounds.iter().enumerate() {
            if frame_idx < bounds.start_f || frame_idx >= bounds.end_f {
                continue;
            }
            let result = &clip_results[clip_idx];
            let local_sec = abs_time_sec - result.clip_start_sec;
            let local_frame = ((local_sec * 1000.0) / frame_period_ms).round().max(0.0) as usize;
            let p = result.midi.get(local_frame).copied().unwrap_or(0.0);

            if !(p.is_finite() && p > 0.0) {
                continue;
            }

            let Some(clip) = resolved_clips[clip_idx] else {
                continue;
            };

            let local_in_clip_frames = ((local_sec * 44100.0).round().max(0.0)) as usize;
            let w = clip_weight_at_frame(
                clip,
                bpm,
                44100,
                result.clip_start_sec,
                result.pre_silence_sec,
                result.clip_total_frames,
                local_in_clip_frames,
                result.track_gain_value,
            );

            if w > best_weight {
                best_weight = w;
                best_idx = Some(clip_idx);
                best_pitch = p;
            }
        }

        // Apply hysteresis
        if let Some(prev_idx) = last_winner_idx {
            if let Some(now_idx) = best_idx {
                if prev_idx != now_idx {
                    let prev_bounds = &clip_bounds[prev_idx];
                    if frame_idx >= prev_bounds.start_f && frame_idx < prev_bounds.end_f {
                        let result = &clip_results[prev_idx];
                        let local_sec = abs_time_sec - result.clip_start_sec;
                        let local_frame =
                            ((local_sec * 1000.0) / frame_period_ms).round().max(0.0) as usize;
                        let prev_pitch = result.midi.get(local_frame).copied().unwrap_or(0.0);

                        if prev_pitch > 0.0 {
                            if let Some(clip) = resolved_clips[prev_idx] {
                                let local_in_clip_frames =
                                    ((local_sec * 44100.0).round().max(0.0)) as usize;
                                let prev_weight = clip_weight_at_frame(
                                    clip,
                                    bpm,
                                    44100,
                                    result.clip_start_sec,
                                    result.pre_silence_sec,
                                    result.clip_total_frames,
                                    local_in_clip_frames,
                                    result.track_gain_value,
                                );

                                if prev_weight > 0.0 && best_weight < prev_weight * hysteresis_ratio
                                {
                                    out[frame_idx] = prev_pitch;
                                    last_winner_idx = Some(prev_idx);
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Commit winner
        if best_weight > 0.0 {
            out[frame_idx] = best_pitch;
            last_winner_idx = best_idx;
        } else {
            out[frame_idx] = 0.0;
        }
    }

    if debug {
        let nonzero = out.iter().filter(|&&v| v.is_finite() && v > 0.0).count();
        eprintln!(
            "[pitch:fuse] result: {}/{} frames ({:.1}%) have pitch",
            nonzero,
            out.len(),
            (nonzero as f64 / out.len().max(1) as f64) * 100.0
        );
    }

    out
}

pub(crate) fn compute_pitch_curve(job: &PitchJob, mut on_progress: impl FnMut(f32)) -> Vec<f32> {
    use std::sync::Arc;

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

    on_progress(0.02);

    // If FCPE isn't available, return zeros.
    if matches!(
        job.algo,
        PitchAnalysisAlgo::WorldDll
            | PitchAnalysisAlgo::NsfHifiganOnnx
            | PitchAnalysisAlgo::Unknown
    ) && !crate::fcpe_onnx::is_available()
    {
        if debug {
            eprintln!(
                "pitch: FCPE unavailable; return zeros (root_track_id={} key={} frames={})",
                job.root_track_id, job.key, job.target_frames
            );
        }
        return vec![0.0; job.target_frames];
    }

    let mut out = vec![0.0f32; job.target_frames];

    let project_sec = job.timeline.project_duration_sec();
    if project_sec <= 1e-9 {
        return out;
    }

    if debug {
        eprintln!(
            "pitch: start analysis v2 (root_track_id={} key={} clips={} frames={} fp_ms={} algo={:?})",
            job.root_track_id,
            job.key,
            job.timeline.clips.len(),
            job.target_frames,
            job.frame_period_ms,
            job.algo
        );
    }

    // Strategy (v2): analyze per-clip pitch in timeline time, then fuse to a single
    // root curve by choosing the dominant (highest-weight) voiced clip each frame.
    // This avoids WORLD instability on overlap regions.

    // Match python demo defaults (utils/wav2F0.py): f0_min=40, f0_max=1600.
    let f0_floor = 40.0;
    let f0_ceil = 1600.0;
    let frame_period_tl_ms = job.frame_period_ms.max(0.1);

    // Track gains (mute/solo already cleared in build_root_mix_timeline).
    let mut track_gain: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for t in &job.timeline.tracks {
        track_gain.insert(t.id.clone(), clamp01(t.volume));
    }

    let bpm = job.timeline.bpm;
    if !(bpm.is_finite() && bpm > 0.0) {
        return out;
    }

    // We need per-frame candidate pitches + weights.
    // Do per-clip analysis first; keep in memory as MIDI curve in timeline frames.
    struct ClipPitch {
        clip_id: String,
        start_sec: f64,
        end_sec: f64,
        pre_silence_sec: f64,
        clip_total_frames: usize,
        midi: Vec<f32>,
        track_gain_value: f32,
    }

    let mut clip_pitches: Vec<ClipPitch> = Vec::new();

    for clip in &job.timeline.clips {
        let Some(source_path) = clip.source_path.as_ref() else {
            continue;
        };

        // Timeline placement.
        let clip_start_sec = clip.start_sec.max(0.0);
        let clip_timeline_len_sec = clip.length_sec.max(0.0);
        if !(clip_timeline_len_sec.is_finite() && clip_timeline_len_sec > 0.0) {
            continue;
        }
        let clip_end_sec = clip_start_sec + clip_timeline_len_sec;

        // Decode audio.
        let (in_rate, in_channels, pcm) =
            match crate::audio_utils::decode_audio_f32_interleaved(Path::new(source_path)) {
                Ok(v) => v,
                Err(_) => continue,
            };
        let in_channels_usize = (in_channels as usize).max(1);
        let in_frames = pcm.len() / in_channels_usize;
        if in_frames < 2 {
            continue;
        }

        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };

        // Source range (already in sec).
        let source_start_sec = clip.source_start_sec.max(0.0);
        let source_end_sec = clip.source_end_sec;
        let pre_silence_sec = (-clip.source_start_sec).max(0.0) / playback_rate.max(1e-6);

        let total_sec = (in_frames as f64) / (in_rate.max(1) as f64);
        if !(total_sec.is_finite() && total_sec > 0.0) {
            continue;
        }

        let src_end_limit_sec = source_end_sec.min(total_sec).max(source_start_sec);
        if src_end_limit_sec - source_start_sec <= 1e-9 {
            continue;
        }

        let src_i0 = (source_start_sec * in_rate as f64).floor().max(0.0) as usize;
        let src_i1 = (src_end_limit_sec * in_rate as f64)
            .ceil()
            .max(src_i0 as f64) as usize;
        let src_i1 = src_i1.min(in_frames);
        if src_i1 <= src_i0 + 1 {
            continue;
        }

        let segment = &pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)];

        // Resample to analysis rate (44100) and convert to mono.
        let mut segment = crate::mixdown::linear_resample_interleaved(
            segment,
            in_channels_usize,
            in_rate,
            44100,
        );
        if clip.reversed {
            crate::mixdown::reverse_interleaved_frames(&mut segment, in_channels_usize);
        }

        let seg_frames = segment.len() / in_channels_usize;
        if seg_frames < 2 {
            continue;
        }

        let mut mono_raw: Vec<f64> = segment
            .chunks_exact(in_channels_usize)
            .map(|chunk| (chunk.iter().sum::<f32>() as f64) / (in_channels_usize as f64))
            .collect();

        // Preprocess: remove DC and clamp.
        let mut mean = 0.0f64;
        for &v in &mono_raw {
            mean += v;
        }
        mean /= mono_raw.len().max(1) as f64;

        let mut max_abs = 0.0f64;
        for &v in &mono_raw {
            let a = (v - mean).abs();
            if a.is_finite() && a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs.is_finite() && max_abs > 1.0 {
            (1.0 / max_abs).clamp(0.0, 1.0)
        } else {
            1.0
        };

        for v in &mut mono_raw {
            *v = ((*v - mean) * scale).clamp(-1.0, 1.0);
        }
        let mono = mono_raw;

        // Compute f0.
        let f0_hz = match crate::fcpe_onnx::infer_f0_hz(
            &mono,
            44100,
            frame_period_tl_ms,
            f0_floor,
            f0_ceil,
        ) {
            Ok(v) => v,
            Err(_) => Vec::new(),
        };

        if f0_hz.len() < 2 {
            continue;
        }

        // Convert to MIDI, keep unvoiced as 0.
        let mut midi: Vec<f32> = Vec::with_capacity(f0_hz.len());
        for hz in f0_hz {
            midi.push(hz_to_midi(hz));
        }

        // Time-align: analysis output is on the segment timeline. We need it in clip timeline time.
        // For now, resample to the clip's timeline length in frames.

        // DEBUG: Check for time alignment issues that cause pitch curve speed mismatch
        let actual_audio_sec = seg_frames as f64 / 44100.0;
        let clip_frames_from_timeline = ((clip_timeline_len_sec * 1000.0) / frame_period_tl_ms)
            .round()
            .max(1.0) as usize;
        let clip_frames_from_audio = ((actual_audio_sec * 1000.0) / frame_period_tl_ms)
            .round()
            .max(1.0) as usize;
        let ratio = actual_audio_sec / clip_timeline_len_sec.max(1e-9);

        // ?playback_rate != 1 时，actual_audio_sec ?clip_timeline_len_sec 不同是正常的
        // （actual_audio_sec ?clip_timeline_len_sec × playback_rate），不应被当作错?
        if debug {
            eprintln!(
                "pitch: [ALIGNMENT] clip_id={} clip_timeline_len_sec={:.3} actual_audio_sec={:.3} ratio={:.3} playback_rate={:.2}",
                clip.id,
                clip_timeline_len_sec,
                actual_audio_sec,
                ratio,
                playback_rate
            );
            eprintln!(
                "  frames_from_timeline={} frames_from_audio={} midi_len={}",
                clip_frames_from_timeline,
                clip_frames_from_audio,
                midi.len(),
            );
        }

        // 始终使用 timeline 帧数：源时域?F0 曲线需?resample ?timeline 时域
        // 这样 pitch_orig 中每帧对应的就是 timeline 上的 frame_period 步进
        let clip_frames = clip_frames_from_timeline;

        let midi = resample_curve_linear(&midi, clip_frames);

        let tg = track_gain.get(&clip.track_id).copied().unwrap_or(1.0);

        // 始终使用 clip_end_sec（timeline 域），确保融合后曲线长度?clip 显示范围一?
        let adjusted_end_sec = clip_end_sec;

        clip_pitches.push(ClipPitch {
            clip_id: clip.id.clone(),
            start_sec: clip_start_sec,
            end_sec: adjusted_end_sec,
            pre_silence_sec,
            clip_total_frames: ((actual_audio_sec * 44100.0).round().max(1.0)) as usize,
            midi,
            track_gain_value: tg,
        });
    }

    on_progress(0.85);

    // Task 10: Convert ClipPitch to ClipAnalysisResult for fusion optimization
    let clip_results: Vec<ClipAnalysisResult> = clip_pitches
        .into_iter()
        .map(|cp| ClipAnalysisResult {
            clip_id: cp.clip_id,
            clip_start_sec: cp.start_sec,
            clip_end_sec: cp.end_sec,
            pre_silence_sec: cp.pre_silence_sec,
            clip_total_frames: cp.clip_total_frames,
            midi: Arc::new(cp.midi), // Wrap in Arc for ClipAnalysisResult
            track_gain_value: cp.track_gain_value,
            was_cache_hit: false,
        })
        .collect();

    // Task 10.1-10.8: Use optimized fusion algorithm with coverage table
    out = fuse_clip_pitches_optimized(
        &clip_results,
        &job.timeline.clips,
        job.target_frames,
        frame_period_tl_ms,
        bpm,
        debug,
    );

    on_progress(1.0);

    if debug {
        let any_nonzero = out.iter().any(|&v| v.is_finite() && v > 0.0);
        eprintln!(
            "pitch: done analysis v2 (root_track_id={} key={} any_nonzero={})",
            job.root_track_id, job.key, any_nonzero
        );
    }

    out
}
