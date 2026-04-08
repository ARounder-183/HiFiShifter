//! Progress tracking for multi-clip pitch analysis
//!
//! This module provides structures and functions for tracking the overall
//! progress of parallel pitch analysis jobs across multiple clips.

#![allow(dead_code)]

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
        let _bs = 60.0 / bpm.max(1e-6);
        
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
        if self.total_workload <= 0.0 {
            return 1.0;
        }
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
