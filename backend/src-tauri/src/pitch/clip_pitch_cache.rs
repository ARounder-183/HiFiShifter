//! Clip-level pitch analysis cache
//!
//! This module provides a caching layer for pitch analysis results to avoid
//! redundant expensive F0 analysis operations. The cache uses LRU eviction
//! and generates keys based on all parameters that affect pitch analysis results.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

/// Version number for cache format. Increment this when the cache key format
/// or analysis algorithm changes to invalidate old cache entries.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Default maximum number of cached clip pitch curves
pub const DEFAULT_CACHE_CAPACITY: usize = 100;

/// Cache key components for a clip pitch analysis result
///
/// 全量分析策略：缓存 key 不含 source_start/end/playback_rate，
/// 始终缓存全量源音频的 MIDI 曲线，trim/rate 变化在组装阶段处理。
#[derive(Debug, Clone, PartialEq)]
pub struct ClipCacheKey {
    /// Source audio file path
    pub source_path: String,
    /// File size in bytes
    pub file_size: u64,
    /// File modification time (milliseconds since UNIX_EPOCH)
    pub file_mtime: u64,
    /// Analysis algorithm identifier
    pub algo: String,
    /// F0 floor frequency (Hz)
    pub f0_floor: u64,
    /// F0 ceiling frequency (Hz)
    pub f0_ceil: u64,
    /// Cache format version
    pub version: u32,
}

/// LRU cache for clip pitch analysis results
pub struct ClipPitchCache {
    cache: LruCache<String, Arc<Vec<f32>>>,
    hits: u64,
    misses: u64,
}

impl ClipPitchCache {
    /// Create a new cache with specified capacity
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity)
            .unwrap_or(NonZeroUsize::new(DEFAULT_CACHE_CAPACITY).unwrap());
        Self {
            cache: LruCache::new(cap),
            hits: 0,
            misses: 0,
        }
    }

    /// Query the cache for a clip's pitch curve
    pub fn get(&mut self, key: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(curve) = self.cache.get(key) {
            self.hits += 1;
            Some(Arc::clone(curve))
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a new pitch curve into the cache
    pub fn put(&mut self, key: String, curve: Arc<Vec<f32>>) {
        self.cache.put(key, curve);
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            entries: self.cache.len(),
            capacity: self.cache.cap().get(),
            hits: self.hits,
            misses: self.misses,
            hit_rate,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

/// Quantize a floating-point value to specified precision
///
/// This is used to avoid cache misses due to floating-point rounding errors.
/// The multiplier determines precision: 1000.0 gives 3 decimal places.
pub fn quantize_f64(value: f64, multiplier: f64) -> u64 {
    if !value.is_finite() {
        return 0;
    }
    (value * multiplier).round().max(0.0) as u64
}

/// Quantize a signed floating-point value to specified precision
pub fn quantize_i64(value: f64, multiplier: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    (value * multiplier).round() as i64
}

/// Get file signature (size and modification time)
pub fn get_file_signature(path: &Path) -> (u64, u64) {
    if let Ok(metadata) = std::fs::metadata(path) {
        let len = metadata.len();
        let mtime_ms = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        (len, mtime_ms)
    } else {
        (0, 0)
    }
}

/// Generate a cache key string from clip parameters
pub fn generate_clip_cache_key(key_data: &ClipCacheKey) -> String {
    let mut hasher = blake3::Hasher::new();

    // Add version
    hasher.update(b"clip_pitch_v");
    hasher.update(&key_data.version.to_le_bytes());

    // Add file identity
    hasher.update(key_data.source_path.as_bytes());
    hasher.update(&key_data.file_size.to_le_bytes());
    hasher.update(&key_data.file_mtime.to_le_bytes());

    // 全量分析策略：不含 source_start/end/playback_rate
    // trim/rate 变化不影响缓存 key，在组装阶段处理

    // Add analysis algorithm and parameters
    hasher.update(key_data.algo.as_bytes());
    hasher.update(&key_data.f0_floor.to_le_bytes());
    hasher.update(&key_data.f0_ceil.to_le_bytes());

    hasher.finalize().to_hex().to_string()
}
