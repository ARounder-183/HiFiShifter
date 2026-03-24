// Pitch cache management commands

use crate::state::AppState;
use serde::Serialize;
use tauri::State;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchCacheStatsPayload {
    pub cached_clips: usize,
    pub total_capacity: usize,
    pub cache_hit_rate: Option<f32>,
}

pub(super) fn clear_pitch_cache(state: State<'_, AppState>) -> serde_json::Value {
    let mut result = serde_json::json!({ "ok": true });

    if let Ok(mut cache_guard) = state.clip_pitch_cache.lock() {
        cache_guard.clear();
        result["message"] =
            serde_json::Value::String("Pitch cache cleared successfully".to_string());
    } else {
        result["ok"] = serde_json::Value::Bool(false);
        result["error"] = serde_json::Value::String("Failed to lock cache".to_string());
    }

    result
}

pub(super) fn get_pitch_cache_stats(state: State<'_, AppState>) -> PitchCacheStatsPayload {
    let mut stats = PitchCacheStatsPayload {
        cached_clips: 0,
        total_capacity: 100, // Default LRU capacity
        cache_hit_rate: None,
    };

    if let Ok(cache_guard) = state.clip_pitch_cache.lock() {
        let cache_stats = cache_guard.stats();
        stats.cached_clips = cache_stats.entries;
        stats.total_capacity = cache_stats.capacity;
        stats.cache_hit_rate = Some(cache_stats.hit_rate as f32);
    }

    stats
}
