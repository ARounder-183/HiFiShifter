// 波形命令：Mix 波形 + V2 Mipmap 二进制传输
use crate::state::AppState;
use base64::Engine as _;
use tauri::State;

pub use hifishifter_kernel::editor::waveform::WaveformPeaksSegmentPayload;

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

/// 原根轨mix波形薄适配，默认设备/文件工作区路径不变。
pub(super) fn get_root_mix_waveform_peaks_segment(state:State<'_,AppState>,track_id:String,
    start_sec:f64,duration_sec:f64,columns:usize)->WaveformPeaksSegmentPayload {
    hifishifter_kernel::editor::waveform::get_root_mix_waveform_peaks_segment(&*state,track_id,start_sec,duration_sec,columns)
}
/// 原轨道mix波形薄适配。
pub(super) fn get_track_mix_waveform_peaks_segment(state:State<'_,AppState>,track_id:String,
    start_sec:f64,duration_sec:f64,columns:usize)->WaveformPeaksSegmentPayload {
    hifishifter_kernel::editor::waveform::get_track_mix_waveform_peaks_segment(&*state,track_id,start_sec,duration_sec,columns)
}
/// 返回 Base64 编码的 String，避免 Tauri v2 将 Vec<u8> 序列化为 JSON number[]
/// 导致的 3~5 倍传输膨胀。前端通过 atob() 解码后直接创建 Float32Array 视图。
///
/// 二进制协议（v2）：[Header 28B: magic"WFPK" | format_version u32 | sample_rate |
/// division_factor | count | level | channels] 后接逐声道 [min f32[]] [max f32[]]
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
///
/// `levels` 为要**编码并传输**的级别白名单（`None` = 全部三级，与旧行为一致；
/// 空列表同样按全部处理）。批量预载只落地 L2，却曾让后端把 L0（单级 ≈ 159MB/小时
/// 素材、base64 后 ≈ 212MB）也编码传输后由前端丢弃——传 `[2]` 即消除这段浪费。
/// 未被请求的级别返回空字符串，返回形状保持不变。
pub(super) fn batch_get_waveform_mipmap(
    state: State<'_, AppState>,
    source_paths: Vec<String>,
    levels: Option<Vec<u8>>,
) -> std::collections::HashMap<String, [String; 3]> {
    let encoder = base64::engine::general_purpose::STANDARD;
    let mut result = std::collections::HashMap::with_capacity(source_paths.len());

    let mut selected = [false; 3];
    match &levels {
        Some(requested) => {
            for &level in requested {
                if (level as usize) < 3 {
                    selected[level as usize] = true;
                }
            }
            // 空白名单视为"全都要"，避免调用方传空数组时拿到全空结果。
            if !selected.iter().any(|&on| on) {
                selected = [true; 3];
            }
        }
        None => selected = [true; 3],
    }

    for path in source_paths {
        match state.get_or_compute_waveform_peaks_v2(&path) {
            Ok(data) => {
                let mut encoded: [String; 3] = Default::default();
                for level in 0..3 {
                    if selected[level] {
                        encoded[level] = encoder.encode(data.to_binary_level(level));
                    }
                }
                result.insert(path, encoded);
            }
            Err(e) => {
                log::warn!("waveform mipmap batch compute failed for {path}: {e}");
                result.insert(path, [String::new(), String::new(), String::new()]);
            }
        }
    }

    result
}
