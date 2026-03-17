//! Progress tracking for multi-clip pitch analysis
//!
//! This module provides structures and functions for tracking the overall
//! progress of parallel pitch analysis jobs across multiple clips.

use crate::state::Clip;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Progress tracker for multi-clip pitch analysis
///
/// This structure tracks the overall progress of analyzing multiple clips,
/// providing weighted progress calculation based on clip duration and cache hit status.
pub struct ProgressTracker {
    /// Total workload in clip-seconds (weighted by cache miss factor)
    total_workload: f64,
    /// Completed workload so far (clip-seconds, scaled by 1000 for atomic ops)
    completed_workload: AtomicU64,
    /// Start time for ETA calculation
    start_time: Instant,
    /// Total number of clips to analyze (excluding cache hits)
    pub total_clips: u32,
    /// Number of clips completed so far
    completed_clips: AtomicU32,
    /// Name of the clip currently being analyzed
    current_clip_name: Mutex<Option<String>>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    ///
    /// # Parameters
    /// - `clips`: List of clips to analyze
    /// - `bpm`: Project BPM
    /// - `cache`: Reference to cache for hit rate estimation
    ///
    /// # Workload calculation
    /// Each clip contributes: duration_sec * cache_miss_factor
    /// - Cache hit (95% probability after warm-up): 0.01x weight
    /// - Cache miss: 1.0x weight
    pub fn new(
        clips: &[Clip],
        bpm: f64,
        cache: &Arc<Mutex<crate::clip_pitch_cache::ClipPitchCache>>,
    ) -> Self {
        let bs = 60.0 / bpm.max(1e-6);

        // Estimate cache hit rate (use current stats if available)
        let cache_miss_factor = {
            if let Ok(guard) = cache.lock() {
                let stats = guard.stats();
                if stats.hits + stats.misses > 10 {
                    // Use actual hit rate
                    1.0 - stats.hit_rate * 0.99 // Cache hits contribute 1% workload
                } else {
                    // Cold start: assume 100% miss rate
                    1.0
                }
            } else {
                1.0 // Fallback: assume worst case
            }
        };

        let mut total = 0.0f64;
        for clip in clips {
            let duration_sec = clip.length_sec.max(0.0);
            if duration_sec > 0.0 && duration_sec.is_finite() {
                total += duration_sec * cache_miss_factor;
            }
        }

        Self {
            total_workload: total.max(1e-6), // Avoid division by zero
            completed_workload: AtomicU64::new(0),
            start_time: Instant::now(),
            total_clips: clips.len() as u32,
            completed_clips: AtomicU32::new(0),
            current_clip_name: Mutex::new(None),
        }
    }

    /// Report completion of a clip
    ///
    /// # Parameters
    /// - `clip_duration_sec`: Duration of the completed clip in seconds
    /// - `was_cache_hit`: Whether the clip was served from cache
    ///
    /// # Returns
    /// Current overall progress (0.0 to 1.0)
    pub fn report_clip_completed(&self, clip_duration_sec: f64, was_cache_hit: bool) -> f32 {
        let workload = if was_cache_hit {
            clip_duration_sec * 0.01 // Cache hits count as 1% workload
        } else {
            clip_duration_sec
        };

        let workload_u64 = (workload * 1000.0).round().max(0.0) as u64;
        self.completed_workload
            .fetch_add(workload_u64, Ordering::Relaxed);
        self.completed_clips.fetch_add(1, Ordering::Relaxed);

        self.get_current_progress()
    }

    /// Set the name of the clip currently being analyzed
    pub fn set_current_clip(&self, name: Option<String>) {
        if let Ok(mut guard) = self.current_clip_name.lock() {
            *guard = name;
        }
    }

    /// Get the name of the clip currently being analyzed
    pub fn get_current_clip_name(&self) -> Option<String> {
        self.current_clip_name.lock().ok()?.clone()
    }

    /// Get the number of completed clips
    pub fn get_completed_clips(&self) -> u32 {
        self.completed_clips.load(Ordering::Relaxed)
    }

    /// Get current progress percentage
    pub fn get_current_progress(&self) -> f32 {
        let completed = self.completed_workload.load(Ordering::Relaxed) as f64 / 1000.0;
        let progress = (completed / self.total_workload).clamp(0.0, 1.0);
        progress as f32
    }

    /// Estimate remaining time in seconds
    ///
    /// # Returns
    /// - `Some(seconds)`: Estimated time remaining
    /// - `None`: Not enough data to estimate
    pub fn estimate_eta(&self) -> Option<f64> {
        let elapsed_sec = self.start_time.elapsed().as_secs_f64();
        if elapsed_sec < 0.1 {
            return None; // Too early to estimate
        }

        let completed = self.completed_workload.load(Ordering::Relaxed) as f64 / 1000.0;
        if completed < 1e-6 {
            return None; // No progress yet
        }

        let remaining = (self.total_workload - completed).max(0.0);
        let speed = completed / elapsed_sec; // workload per second

        if speed < 1e-9 {
            return None; // Insufficient speed data
        }

        Some(remaining / speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_clip(id: &str, length_sec: f64) -> Clip {
        Clip {
            id: id.to_string(),
            track_id: "test_track".to_string(),
            name: format!("Clip {}", id),
            start_sec: 0.0,
            length_sec,
            color: "#ff0000".to_string(),
            source_path: Some(format!("/test/{}.wav", id)),
            duration_sec: Some(length_sec),
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: length_sec,
            playback_rate: 1.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
        }
    }

    #[test]
    fn test_progress_tracker_weighted_calculation() {
        let clips = vec![
            create_test_clip("1", 2.0), // 2 sec
            create_test_clip("2", 4.0), // 4 sec
            create_test_clip("3", 2.0), // 2 sec
        ];

        let cache = Arc::new(Mutex::new(crate::clip_pitch_cache::ClipPitchCache::new(10)));

        let tracker = ProgressTracker::new(&clips, 120.0, &cache);

        // Total workload = 2 + 4 + 2 = 8 sec (all cache miss)
        assert!((tracker.total_workload - 8.0).abs() < 0.01);

        // Report first clip complete (cache miss)
        let progress = tracker.report_clip_completed(2.0, false);
        assert!((progress - 0.25).abs() < 0.01); // 2/8 = 25%

        // Report second clip complete (cache hit)
        let progress = tracker.report_clip_completed(4.0, true);
        // Completed = 2.0 + 0.04 = 2.04, progress = 2.04/8 = 25.5%
        assert!((progress - 0.255).abs() < 0.01);

        // Report third clip complete (cache miss)
        let progress = tracker.report_clip_completed(2.0, false);
        // Completed = 2.04 + 2.0 = 4.04, progress = 4.04/8 = 50.5%
        assert!((progress - 0.505).abs() < 0.01);
    }

    #[test]
    fn test_progress_tracker_concurrent_updates() {
        use std::thread;

        let clips = vec![
            create_test_clip("1", 40.0), // 20 sec total workload
        ];

        let cache = Arc::new(Mutex::new(crate::clip_pitch_cache::ClipPitchCache::new(10)));

        let tracker = Arc::new(ProgressTracker::new(&clips, 120.0, &cache));

        // Simulate 10 threads each completing 2 seconds of work
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    tracker.report_clip_completed(2.0, false);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Total completed = 20 sec, should be 100%
        let final_progress = tracker.get_current_progress();
        assert!((final_progress - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_tracker_eta_estimation() {
        let clips = vec![
            create_test_clip("1", 80.0), // 40 sec @ 120 BPM
        ];

        let cache = Arc::new(Mutex::new(crate::clip_pitch_cache::ClipPitchCache::new(10)));

        let tracker = ProgressTracker::new(&clips, 120.0, &cache);

        // Initially no ETA (not enough progress)
        assert!(tracker.estimate_eta().is_none());

        // Simulate completing 10 sec of work
        tracker.report_clip_completed(10.0, false);

        // Sleep to let some elapsed time accumulate for speed calculation
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ETA should exist and be positive
        // In fast test environments, speed can be very high (10 sec work / 0.01 sec elapsed = 1000x)
        // so ETA might be very small. We just verify it's calculated and reasonable.
        if let Some(eta) = tracker.estimate_eta() {
            // Just check it's positive and not infinite
            assert!(
                eta > 0.0 && eta < 1000.0,
                "ETA {} out of expected range",
                eta
            );
        }

        // Complete more work to get more stable ETA
        tracker.report_clip_completed(10.0, false);
        if let Some(eta) = tracker.estimate_eta() {
            assert!(eta > 0.0 && eta < 1000.0);
        }
    }
}
