//! Clip-level pitch analysis cache
//!
//! This module provides a caching layer for pitch analysis results to avoid
//! redundant expensive F0 analysis operations. The cache uses LRU eviction
//! and generates keys based on all parameters that affect pitch analysis results.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_f64() {
        assert_eq!(quantize_f64(1.2345, 1000.0), 1235);
        assert_eq!(quantize_f64(1.2344, 1000.0), 1234);
        assert_eq!(quantize_f64(-1.5, 1000.0), 0); // Negative clamped to 0
        assert_eq!(quantize_f64(f64::NAN, 1000.0), 0);
        assert_eq!(quantize_f64(f64::INFINITY, 1000.0), 0);
    }

    #[test]
    fn test_quantize_i64() {
        assert_eq!(quantize_i64(1.2345, 1000.0), 1235);
        assert_eq!(quantize_i64(-1.5, 1000.0), -1500);
        assert_eq!(quantize_i64(f64::NAN, 1000.0), 0);
    }

    #[test]
    fn test_cache_key_consistency() {
        let key1 = ClipCacheKey {
            source_path: "/test/audio.wav".to_string(),
            file_size: 1000,
            file_mtime: 123456789,
            algo: "world_dll".to_string(),
            f0_floor: 40,
            f0_ceil: 1600,
            version: CACHE_FORMAT_VERSION,
        };

        let key2 = key1.clone();

        let hash1 = generate_clip_cache_key(&key1);
        let hash2 = generate_clip_cache_key(&key2);

        assert_eq!(
            hash1, hash2,
            "Same parameters should generate same cache key"
        );
    }

    #[test]
    fn test_cache_key_trim_rate_invariance() {
        // 全量分析策略：不同的 trim/rate 应该产生相同的 cache key
        // 因为缓存的是全量源音频曲线，trim/rate 在组装阶段处理
        let key1 = ClipCacheKey {
            source_path: "/test/audio.wav".to_string(),
            file_size: 1000,
            file_mtime: 123456789,
            algo: "world_dll".to_string(),
            f0_floor: 40,
            f0_ceil: 1600,
            version: CACHE_FORMAT_VERSION,
        };

        let key2 = key1.clone();

        let hash1 = generate_clip_cache_key(&key1);
        let hash2 = generate_clip_cache_key(&key2);

        assert_eq!(hash1, hash2, "Trim/rate should not affect cache key");
    }

    #[test]
    fn test_cache_key_parameter_sensitivity() {
        let base_key = ClipCacheKey {
            source_path: "/test/audio.wav".to_string(),
            file_size: 1000,
            file_mtime: 123456789,
            algo: "world_dll".to_string(),
            f0_floor: 40,
            f0_ceil: 1600,
            version: CACHE_FORMAT_VERSION,
        };

        let base_hash = generate_clip_cache_key(&base_key);

        // Different source file should produce different key
        let mut key = base_key.clone();
        key.source_path = "/test/other.wav".to_string();
        assert_ne!(generate_clip_cache_key(&key), base_hash);

        // Different algorithm should produce different key
        let mut key = base_key.clone();
        key.algo = "nsf_hifigan_onnx".to_string();
        assert_ne!(generate_clip_cache_key(&key), base_hash);

        // Different f0 floor should produce different key
        let mut key = base_key.clone();
        key.f0_floor = 80;
        assert_ne!(generate_clip_cache_key(&key), base_hash);
    }

    #[test]
    fn test_clip_pitch_cache_basic() {
        let mut cache = ClipPitchCache::new(2);

        let curve1 = Arc::new(vec![60.0, 61.0, 62.0]);
        let curve2 = Arc::new(vec![65.0, 66.0, 67.0]);

        cache.put("key1".to_string(), curve1.clone());
        cache.put("key2".to_string(), curve2.clone());

        assert_eq!(cache.get("key1").unwrap().as_ref(), curve1.as_ref());
        assert_eq!(cache.get("key2").unwrap().as_ref(), curve2.as_ref());

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_clip_pitch_cache_lru() {
        let mut cache = ClipPitchCache::new(2);

        let curve1 = Arc::new(vec![60.0]);
        let curve2 = Arc::new(vec![61.0]);
        let curve3 = Arc::new(vec![62.0]);

        cache.put("key1".to_string(), curve1);
        cache.put("key2".to_string(), curve2);
        cache.put("key3".to_string(), curve3); // Should evict key1

        assert!(cache.get("key1").is_none()); // Evicted
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_clip_pitch_cache_clear() {
        let mut cache = ClipPitchCache::new(10);

        cache.put("key1".to_string(), Arc::new(vec![60.0]));
        cache.get("key1");
        cache.get("key2"); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }
}
