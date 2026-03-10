// pitch_analysis — 音高分析主模块
// 工具函数、类型定义、公开 API。
// 核心分析流水线见 analysis.rs，调度逻辑见 schedule.rs。

use crate::state::{AppState, Clip, PitchAnalysisAlgo, TimelineState};
use serde::Serialize;
use std::collections::HashSet;
use std::path::Path;
use tauri::Emitter;

pub(crate) mod analysis;
pub(crate) mod schedule;

// 公开 API — 供 crate 内其他模块使用
pub use schedule::maybe_schedule_pitch_orig;
pub(crate) use analysis::{build_pitch_job, compute_pitch_curve};


pub(crate) fn hz_to_midi(hz: f64) -> f32 {
    if !(hz.is_finite() && hz > 1e-6) {
        return 0.0;
    }
    let midi = 69.0 + 12.0 * (hz / 440.0).log2();
    if midi.is_finite() {
        midi as f32
    } else {
        0.0
    }
}

pub(crate) fn quantize_i64(x: f64, scale: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    (x * scale).round() as i64
}

pub(crate) fn quantize_u32(x: f64, scale: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let v = (x * scale).round();
    if v <= 0.0 {
        0
    } else if v > (u32::MAX as f64) {
        u32::MAX
    } else {
        v as u32
    }
}

pub(crate) fn file_sig(path: &Path) -> (u64, u64) {
    // (len_bytes, modified_ms_since_epoch)
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return (0, 0),
    };
    let len = meta.len();
    let mtime_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    (len, mtime_ms)
}

pub(crate) fn build_root_pitch_key(tl: &TimelineState, root_track_id: &str) -> String {
    let bpm = if tl.bpm.is_finite() && tl.bpm > 0.0 {
        tl.bpm
    } else {
        120.0
    };

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pitch_orig_v2_clip_fuse");
    hasher.update(root_track_id.as_bytes());
    hasher.update(&quantize_u32(bpm, 1000.0).to_le_bytes());
    hasher.update(&quantize_u32(tl.frame_period_ms(), 1000.0).to_le_bytes());

    // Include track-level analysis config.
    let (compose, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::Unknown));
    hasher.update(&[if compose { 1 } else { 0 }]);
    hasher.update(match algo {
        PitchAnalysisAlgo::WorldDll => b"world_dll",
        PitchAnalysisAlgo::NsfHifiganOnnx => b"nsf_hifigan_onnx",
        PitchAnalysisAlgo::VocalShifterVslib => b"vslib",
        PitchAnalysisAlgo::ExternalResampler(_) => b"external_resampler",
        PitchAnalysisAlgo::None => b"none",
        PitchAnalysisAlgo::Unknown => b"unknown",
    });

    // If WORLD is selected, include its availability so we can cache the
    // unavailable state but still recompute when the DLL becomes available.
    if matches!(
        algo,
        PitchAnalysisAlgo::WorldDll
            | PitchAnalysisAlgo::NsfHifiganOnnx
            | PitchAnalysisAlgo::Unknown
    ) {
        hasher.update(&[if crate::world::is_available() { 1 } else { 0 }]);
    }

    // Include each clip mapped to this root track.
    // Sort by clip id for stability.
    let mut clips: Vec<&Clip> = tl
        .clips
        .iter()
        .filter(|c| tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id))
        .collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));

    for c in clips {
        hasher.update(c.id.as_bytes());
        hasher.update(&quantize_u32(c.start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.length_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.playback_rate as f64, 10000.0).to_le_bytes());
        hasher.update(&quantize_i64(c.source_start_sec, 1000.0).to_le_bytes());
        hasher.update(&quantize_u32(c.source_end_sec, 1000.0).to_le_bytes());
        if let Some(sp) = c.source_path.as_deref() {
            hasher.update(sp.as_bytes());
            let p = Path::new(sp);
            let (len, mtime) = file_sig(p);
            hasher.update(&len.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());
        } else {
            hasher.update(b"(no_source)");
        }
    }

    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, Clone)]
pub(crate) struct PitchJob {
    pub(crate) root_track_id: String,
    pub(crate) key: String,
    pub(crate) frame_period_ms: f64,
    pub(crate) target_frames: usize,
    pub(crate) algo: PitchAnalysisAlgo,

    /// Root-subtree timeline snapshot used for root-mix analysis.
    /// This matches what the parameter panel background waveform shows.
    pub(crate) timeline: TimelineState,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigUpdatedEvent {
    pub root_track_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigAnalysisStartedEvent {
    pub root_track_id: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchOrigAnalysisProgressEvent {
    pub root_track_id: String,
    pub progress: f32,
    /// 当前正在分析?clip 名称（None 表示未知或已完成?
    pub current_clip_name: Option<String>,
    /// 已完成的 clip 数量
    pub completed_clips: u32,
    /// 需要分析的 clip 总数
    pub total_clips: u32,
}

pub(crate) fn resample_curve_linear(values: &[f32], out_len: usize) -> Vec<f32> {
    if out_len == 0 {
        return vec![];
    }
    if values.is_empty() {
        return vec![0.0; out_len];
    }
    if values.len() == out_len {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; out_len];
    }
    if out_len == 1 {
        return vec![values[0]];
    }

    let in_len = values.len();
    let scale = (in_len - 1) as f64 / (out_len - 1) as f64;
    let mut out = vec![0.0f32; out_len];
    for (of, out_v) in out.iter_mut().enumerate() {
        let t_in = (of as f64) * scale;
        let i0 = t_in.floor() as usize;
        let i1 = (i0 + 1).min(in_len - 1);
        let frac = (t_in - (i0 as f64)) as f32;
        let a = values[i0];
        let b = values[i1];
        *out_v = a + (b - a) * frac;
    }
    out
}


// Task 3.6: PitchProgressPayload for frontend API
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchProgressPayload {
    pub root_track_id: String,
    pub progress: f32,
    pub eta_seconds: Option<f64>,
    /// 当前正在分析?clip 名称
    pub current_clip_name: Option<String>,
    /// 已完成的 clip 数量
    pub completed_clips: u32,
    /// 需要分析的 clip 总数
    pub total_clips: u32,
}

impl From<&PitchOrigAnalysisProgressEvent> for PitchProgressPayload {
    fn from(event: &PitchOrigAnalysisProgressEvent) -> Self {
        Self {
            root_track_id: event.root_track_id.clone(),
            progress: event.progress,
            eta_seconds: None,
            current_clip_name: event.current_clip_name.clone(),
            completed_clips: event.completed_clips,
            total_clips: event.total_clips,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip_pitch_cache::ClipPitchCache;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_analyze_clip_with_cache_missing_source() {
        let clip = Clip {
            id: "test_clip".to_string(),
            track_id: "test_track".to_string(),
            name: "Test Clip".to_string(),
            start_sec: 0.0,
            length_sec: 4.0,
            color: "#ff0000".to_string(),
            source_path: None,
            duration_sec: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 4.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
        };

        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        let result = analyze_clip_with_cache(
            &clip,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            false,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No source path");
    }

    #[test]
    fn test_analyze_clip_with_cache_nonexistent_file() {
        let clip = Clip {
            id: "test_clip".to_string(),
            track_id: "test_track".to_string(),
            name: "Test Clip".to_string(),
            start_sec: 0.0,
            length_sec: 4.0,
            color: "#ff0000".to_string(),
            source_path: Some("/nonexistent/path/to/audio.wav".to_string()),
            duration_sec: Some(2.0),
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,        };

        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        let result = analyze_clip_with_cache(
            &clip,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            false,
        );

        // Should fail with decode error
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to decode audio"));
    }

    #[test]
    fn test_compute_pitch_curve_parallel_empty_clips() {
        let clips: Vec<Clip> = vec![];
        let tracks_gain = std::collections::HashMap::new();
        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        
        let result = compute_pitch_curve_parallel(
            &clips,
            &tracks_gain,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            None,
            false,
        );
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
    
    #[test]
    fn test_compute_pitch_curve_parallel_all_invalid() {
        let clips = vec![
            Clip {
                id: "clip1".to_string(),
                track_id: "track1".to_string(),
                name: "Invalid 1".to_string(),
            start_sec: 0.0,
            length_sec: 4.0,
            color: "#ff0000".to_string(),
            source_path: None,
            duration_sec: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 4.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            },
            Clip {
                id: "clip2".to_string(),
                track_id: "track1".to_string(),
                name: "Invalid 2".to_string(),
                start_sec: 4.0,
                length_sec: 4.0,
                color: "#00ff00".to_string(),
                source_path: None,
                duration_sec: None,
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: 4.0,
                playback_rate: 1.0,
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
            },
        ];
        
        let mut tracks_gain = std::collections::HashMap::new();
        tracks_gain.insert("track1".to_string(), 1.0);
        
        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        
        let result = compute_pitch_curve_parallel(
            &clips,
            &tracks_gain,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            None,
            false,
        );
        
        // Should fail with critical failure (>50% failed)
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Critical failure"));
    }
    
    #[test]
    fn test_compute_pitch_curve_parallel_with_tracker() {
        // Test that progress tracker integration doesn't cause crashes
        let clips = vec![
            Clip {
                id: "clip1".to_string(),
                track_id: "track1".to_string(),
                name: "Test Clip".to_string(),
            start_sec: 0.0,
            length_sec: 4.0,
                color: "#ff0000".to_string(),
                source_path: Some("/nonexistent.wav".to_string()),
                duration_sec: Some(2.0),
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            },
        ];
        
        let mut tracks_gain = std::collections::HashMap::new();
        tracks_gain.insert("track1".to_string(), 1.0);
        
        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        let tracker = Arc::new(crate::pitch_progress::ProgressTracker::new(
            &clips,
            120.0,
            &cache,
        ));
        
        let result = compute_pitch_curve_parallel(
            &clips,
            &tracks_gain,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            Some(&tracker),
            false,
        );
        
        // Should fail (nonexistent file) but not crash
        assert!(result.is_err());
        
        // Progress should be updated (100% since clip completed, even with failure)
        let progress = tracker.get_current_progress();
        assert!(progress >= 0.0 && progress <= 1.0);
    }
    
    #[test]
    fn test_compute_pitch_curve_parallel_mixed_algorithms() {
        // This test verifies that the WORLD/ONNX separation logic works correctly.
        // In a real scenario, WORLD clips would be processed serially and ONNX clips in parallel.
        // Here we just verify the function handles clips with different algorithms without crashing.
        
        // Note: Since we're passing WorldDll as `algo` parameter, all clips will be treated as WORLD clips.
        // In the actual implementation, each clip would have its own algorithm field.
        // For now, this test verifies the separation logic doesn't break the flow.
        
        let clips = vec![
            Clip {
                id: "world1".to_string(),
                track_id: "track1".to_string(),
                name: "WORLD Clip 1".to_string(),
            start_sec: 0.0,
            length_sec: 4.0,
                color: "#ff0000".to_string(),
                source_path: Some("/nonexistent_world.wav".to_string()),
                duration_sec: Some(2.0),
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            },
            Clip {
                id: "world2".to_string(),
                track_id: "track1".to_string(),
                name: "WORLD Clip 2".to_string(),
            start_sec: 4.0,
            length_sec: 4.0,
                color: "#00ff00".to_string(),
                source_path: Some("/nonexistent_world2.wav".to_string()),
                duration_sec: Some(2.0),
                waveform_preview: None,
                pitch_range: None,
                gain: 1.0,
                muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            },
        ];
        
        let mut tracks_gain = std::collections::HashMap::new();
        tracks_gain.insert("track1".to_string(), 1.0);
        
        let cache = Arc::new(Mutex::new(ClipPitchCache::new(10)));
        
        // Test with WORLD algorithm (all clips will be processed serially)
        let result = compute_pitch_curve_parallel(
            &clips,
            &tracks_gain,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::WorldDll,
            &cache,
            None,
            false,
        );
        
        // Should fail (all clips will fail due to nonexistent files, 100% > 50%)
        assert!(result.is_err());
        
        // Test with ONNX algorithm (all clips will be processed in parallel)
        let result_onnx = compute_pitch_curve_parallel(
            &clips,
            &tracks_gain,
            120.0,
            5.0,
            40.0,
            1600.0,
            &PitchAnalysisAlgo::NsfHifiganOnnx,
            &cache,
            None,
            false,
        );
        
        // Should also fail (all clips will fail, 100% > 50%)
        assert!(result_onnx.is_err());
    }
    
    #[test]
    fn test_compare_snapshots_no_old_snapshot() {
        use std::collections::HashMap;
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());
        new_clips.insert("clip2".to_string(), "key2".to_string());
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(None, &new_snapshot);
        
        assert_eq!(comparison.added_clip_ids.len(), 2);
        assert!(comparison.added_clip_ids.contains(&"clip1".to_string()));
        assert!(comparison.added_clip_ids.contains(&"clip2".to_string()));
        assert_eq!(comparison.modified_clip_ids.len(), 0);
        assert_eq!(comparison.deleted_clip_ids.len(), 0);
        assert_eq!(comparison.unchanged_clip_ids.len(), 0);
    }
    
    #[test]
    fn test_compare_snapshots_added_clips() {
        use std::collections::HashMap;
        
        let mut old_clips = HashMap::new();
        old_clips.insert("clip1".to_string(), "key1".to_string());
        
        let old_snapshot = crate::state::TimelineSnapshot {
            clips: old_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());
        new_clips.insert("clip2".to_string(), "key2".to_string());
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(Some(&old_snapshot), &new_snapshot);
        
        assert_eq!(comparison.added_clip_ids, vec!["clip2".to_string()]);
        assert_eq!(comparison.modified_clip_ids.len(), 0);
        assert_eq!(comparison.deleted_clip_ids.len(), 0);
        assert_eq!(comparison.unchanged_clip_ids, vec!["clip1".to_string()]);
    }
    
    #[test]
    fn test_compare_snapshots_modified_clips() {
        use std::collections::HashMap;
        
        let mut old_clips = HashMap::new();
        old_clips.insert("clip1".to_string(), "key1".to_string());
        old_clips.insert("clip2".to_string(), "key2".to_string());
        
        let old_snapshot = crate::state::TimelineSnapshot {
            clips: old_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());
        new_clips.insert("clip2".to_string(), "key2_modified".to_string());
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(Some(&old_snapshot), &new_snapshot);
        
        assert_eq!(comparison.added_clip_ids.len(), 0);
        assert_eq!(comparison.modified_clip_ids, vec!["clip2".to_string()]);
        assert_eq!(comparison.deleted_clip_ids.len(), 0);
        assert_eq!(comparison.unchanged_clip_ids, vec!["clip1".to_string()]);
    }
    
    #[test]
    fn test_compare_snapshots_deleted_clips() {
        use std::collections::HashMap;
        
        let mut old_clips = HashMap::new();
        old_clips.insert("clip1".to_string(), "key1".to_string());
        old_clips.insert("clip2".to_string(), "key2".to_string());
        
        let old_snapshot = crate::state::TimelineSnapshot {
            clips: old_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(Some(&old_snapshot), &new_snapshot);
        
        assert_eq!(comparison.added_clip_ids.len(), 0);
        assert_eq!(comparison.modified_clip_ids.len(), 0);
        assert_eq!(comparison.deleted_clip_ids, vec!["clip2".to_string()]);
        assert_eq!(comparison.unchanged_clip_ids, vec!["clip1".to_string()]);
    }
    
    #[test]
    fn test_compare_snapshots_global_param_change() {
        use std::collections::HashMap;
        
        let mut old_clips = HashMap::new();
        old_clips.insert("clip1".to_string(), "key1".to_string());
        
        let old_snapshot = crate::state::TimelineSnapshot {
            clips: old_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 140.0,  // BPM changed
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(Some(&old_snapshot), &new_snapshot);
        
        // Clip should be marked as modified due to BPM change
        assert_eq!(comparison.added_clip_ids.len(), 0);
        assert_eq!(comparison.modified_clip_ids, vec!["clip1".to_string()]);
        assert_eq!(comparison.deleted_clip_ids.len(), 0);
        assert_eq!(comparison.unchanged_clip_ids.len(), 0);
    }
    
    #[test]
    fn test_compare_snapshots_mixed_changes() {
        use std::collections::HashMap;
        
        let mut old_clips = HashMap::new();
        old_clips.insert("clip1".to_string(), "key1".to_string());
        old_clips.insert("clip2".to_string(), "key2".to_string());
        old_clips.insert("clip3".to_string(), "key3".to_string());
        
        let old_snapshot = crate::state::TimelineSnapshot {
            clips: old_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let mut new_clips = HashMap::new();
        new_clips.insert("clip1".to_string(), "key1".to_string());  // unchanged
        new_clips.insert("clip2".to_string(), "key2_modified".to_string());  // modified
        new_clips.insert("clip4".to_string(), "key4".to_string());  // added
        // clip3 deleted
        
        let new_snapshot = crate::state::TimelineSnapshot {
            clips: new_clips,
            bpm: 120.0,
            frame_period_ms: 5.0,
        };
        
        let comparison = compare_snapshots(Some(&old_snapshot), &new_snapshot);
        
        assert_eq!(comparison.added_clip_ids, vec!["clip4".to_string()]);
        assert_eq!(comparison.modified_clip_ids, vec!["clip2".to_string()]);
        assert_eq!(comparison.deleted_clip_ids, vec!["clip3".to_string()]);
        assert_eq!(comparison.unchanged_clip_ids, vec!["clip1".to_string()]);
    }

    // Note: Full integration tests with actual audio files and cache hit/miss
    // scenarios would require test audio files and more complex setup.
    // These tests verify the basic error handling paths.
}
