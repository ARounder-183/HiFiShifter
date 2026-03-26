// 波形命令：Mix 波形 + V2 Mipmap 二进制传输
use crate::state::AppState;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tauri::State;

use super::common::guard_waveform_command;

const WAVEFORM_COLUMNS_MIN: usize = 16;
const WAVEFORM_COLUMNS_MAX: usize = 65_536;

/// Mix 娉㈠舰杩斿洖杞借嵎锛堝師 WaveformPeaksSegmentPayload锛?
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WaveformPeaksSegmentPayload {
    pub ok: bool,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
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
) -> WaveformPeaksSegmentPayload {
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
            return WaveformPeaksSegmentPayload {
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

        let cols = columns.clamp(WAVEFORM_COLUMNS_MIN, WAVEFORM_COLUMNS_MAX);
        let opts = crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec: Some(start_sec + duration_sec.max(0.0)),
            // Peaks are used as a visual timing reference. Use Signalsmith Stretch so
            // stretched clips line up with the same timing as pitch analysis.
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            // 瀹炴椂棰勮浣跨敤榛樿璐ㄩ噺锛圵av16 + Realtime锛夈€?
            export_format: crate::mixdown::ExportFormat::Wav16,
            quality_preset: crate::mixdown::QualityPreset::Realtime,
        };

        let (_sr, ch, _dur, mix) = match crate::mixdown::render_mixdown_interleaved(&tl, opts) {
            Ok(v) => v,
            Err(_) => {
                return WaveformPeaksSegmentPayload {
                    ok: false,
                    min: vec![],
                    max: vec![],
                }
            }
        };

        let channels = ch.max(1) as usize;
        let frames = mix.len() / channels;
        if frames == 0 {
            return WaveformPeaksSegmentPayload {
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

        WaveformPeaksSegmentPayload {
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
) -> WaveformPeaksSegmentPayload {
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
            return WaveformPeaksSegmentPayload {
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

        let cols = columns.clamp(WAVEFORM_COLUMNS_MIN, WAVEFORM_COLUMNS_MAX);
        let opts = crate::mixdown::MixdownOptions {
            sample_rate: 44100,
            start_sec,
            end_sec: Some(start_sec + duration_sec.max(0.0)),
            // Peaks are used as a visual timing reference. Use Signalsmith Stretch so
            // stretched clips line up with the same timing as pitch analysis.
            stretch: crate::time_stretch::StretchAlgorithm::SignalsmithStretch,
            apply_pitch_edit: true,
            // 瀹炴椂棰勮浣跨敤榛樿璐ㄩ噺锛圵av16 + Realtime锛夈€?
            export_format: crate::mixdown::ExportFormat::Wav16,
            quality_preset: crate::mixdown::QualityPreset::Realtime,
        };

        let (_sr, ch, _dur, mix) = match crate::mixdown::render_mixdown_interleaved(&tl, opts) {
            Ok(v) => v,
            Err(_) => {
                return WaveformPeaksSegmentPayload {
                    ok: false,
                    min: vec![],
                    max: vec![],
                }
            }
        };

        let channels = ch.max(1) as usize;
        let frames = mix.len() / channels;
        if frames == 0 {
            return WaveformPeaksSegmentPayload {
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

        WaveformPeaksSegmentPayload {
            ok: true,
            min: out_min,
            max: out_max,
        }
    })
}

// ===================== v2 mipmap 浜岃繘鍒朵紶杈?=====================

/// 鑾峰彇鎸囧畾绾у埆鐨勬尝褰?mipmap 鏁版嵁锛堜簩杩涘埗鏍煎紡锛?
///
/// 杩斿洖 Vec<u8>锛孴auri 浼氫紶杈撲负 number[]锛圝S 渚ч渶杞?ArrayBuffer锛夛紝
/// 鍓嶇閫氳繃 DataView + Float32Array 鐩存帴璇诲彇銆?
///
/// 浜岃繘鍒跺崗璁細[Header 20B] [min f32[]] [max f32[]]
/// 获取指定级别的波形 mipmap 数据（Base64 编码的二进制格式）
///
/// 返回 Base64 编码的 String，避免 Tauri v2 将 Vec<u8> 序列化为 JSON number[]
/// 导致的 3~5 倍传输膨胀。前端通过 atob() 解码后直接创建 Float32Array 视图。
///
/// 二进制协议：[Header 20B] [min f32[]] [max f32[]]
pub(super) fn get_waveform_mipmap_binary(
    state: State<'_, AppState>,
    source_path: String,
    level: u8,
) -> String {
    let level = (level as usize).min(2);
    match state.get_or_compute_waveform_peaks_v2(&source_path) {
        Ok(data) => {
            let bytes = data.to_binary_level(level);
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        }
        Err(_) => String::new(),
    }
}

/// 棰勫姞杞芥墍鏈夌骇鍒殑 mipmap 鏁版嵁锛堥煶棰戝姞杞芥椂璋冪敤锛?
///
/// 瑙﹀彂 mipmap 璁＄畻骞剁紦瀛樺埌鍐呭瓨 + 纾佺洏锛岄伩鍏嶉娆℃覆鏌撴椂鐨勫欢杩熴€?
pub(super) fn preload_waveform_mipmap(
    state: State<'_, AppState>,
    source_path: String,
) -> serde_json::Value {
    match state.get_or_compute_waveform_peaks_v2(&source_path) {
        Ok(_) => serde_json::json!({"ok": true}),
        Err(e) => serde_json::json!({"ok": false, "error": e}),
    }
}

// ===================== batch preload =====================

/// 批量获取多个音频文件的所有 3 级 mipmap 数据（Base64 编码）
///
/// 将 N 个文件 × 3 级 = 3N 次 IPC 合并为 1 次，大幅减少 IPC 往返开销。
/// 返回 HashMap<sourcePath, [L0_base64, L1_base64, L2_base64]>。
/// 若某个文件计算失败，对应值为 3 个空字符串。
pub(super) fn batch_get_waveform_mipmap(
    state: State<'_, AppState>,
    source_paths: Vec<String>,
) -> std::collections::HashMap<String, [String; 3]> {
    let encoder = base64::engine::general_purpose::STANDARD;
    let mut result = std::collections::HashMap::with_capacity(source_paths.len());

    for path in source_paths {
        match state.get_or_compute_waveform_peaks_v2(&path) {
            Ok(data) => {
                let l0 = encoder.encode(data.to_binary_level(0));
                let l1 = encoder.encode(data.to_binary_level(1));
                let l2 = encoder.encode(data.to_binary_level(2));
                result.insert(path, [l0, l1, l2]);
            }
            Err(_) => {
                result.insert(path, [String::new(), String::new(), String::new()]);
            }
        }
    }

    result
}
