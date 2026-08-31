/*
 * commands/formant.rs - Clip 源共振峰分析命令。
 *
 * 主要内容：
 * - analyze_clip_formants：按 clip_id 返回源共振峰统计（统计 F1/F2 +
 *   稀疏轨迹 + 浊音占比），供前端共振峰工具窗口做"源点 → 目标点"可视化。
 * - 分析计算与 DSP 主流程共用同一套代码（formant_morph::analysis），
 *   解码与缓存复用 formant_cache.rs，保证显示位置与算法认定一致。
 *
 * 与其他模块的关系：
 * - 由 commands.rs 以 `#[path]` 登记并转发（异步 spawn_blocking，
 *   与 waveform 命令同模式，避免阻塞 UI 主线程）。
 * - 需在 lib.rs 的 invoke_handler 中注册。
 *
 * 维护说明：
 * - 载荷字段命名 camelCase（与前端 TS 类型一一对应）；新增字段需同步
 *   frontend/src/services/api/timeline.ts 与 sessionTypes.ts。
 */

use crate::state::AppState;
use serde::Serialize;
use tauri::State;

/// Clip 源共振峰分析载荷。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipFormantAnalysisPayload {
    pub ok: bool,
    /// 统计源 F1（检出帧中位数，Hz；无检出为 0）。
    pub source_f1_hz: f32,
    /// 统计源 F2（Hz；无检出为 0）。
    pub source_f2_hz: f32,
    /// 稀疏轨迹 [t_norm, f1_hz, f2_hz]，按时间升序，≤64 点。
    pub track: Vec<[f32; 3]>,
    /// 检出候选的分析帧占比 [0,1]，过低时前端提示素材不适合调整。
    pub voiced_ratio: f32,
    /// 诊断消息："source_too_short" / "no_voiced_frames"。
    pub message: Option<String>,
}

/// 计算并返回 clip 的源共振峰分析（同步实现，供异步命令包装调用）。
///
/// 流程：
/// 1. 从 timeline 状态取 clip（不存在 → "clip_not_found"）。
/// 2. 走 formant_cache::get_or_compute_formant_analysis（带缓存）。
/// 3. 转换为 camelCase 载荷。
pub(super) fn analyze_clip_formants(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<ClipFormantAnalysisPayload, String> {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .clone();
    let Some(clip) = tl.clips.iter().find(|c| c.id == clip_id).cloned() else {
        return Err("clip_not_found".to_string());
    };
    let summary = crate::formant_cache::get_or_compute_formant_analysis(&clip)?;
    Ok(ClipFormantAnalysisPayload {
        ok: true,
        source_f1_hz: summary.source_f1_hz,
        source_f2_hz: summary.source_f2_hz,
        track: summary
            .track
            .iter()
            .map(|(t, f1, f2)| [*t, *f1, *f2])
            .collect(),
        voiced_ratio: summary.voiced_ratio,
        message: summary.message.map(|m| m.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_serializes_camel_case() {
        let payload = ClipFormantAnalysisPayload {
            ok: true,
            source_f1_hz: 800.0,
            source_f2_hz: 1200.0,
            track: vec![[0.0, 800.0, 1200.0]],
            voiced_ratio: 0.9,
            message: None,
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("sourceF1Hz").is_some());
        assert!(json.get("sourceF2Hz").is_some());
        assert!(json.get("voicedRatio").is_some());
        assert!(json.get("source_f1_hz").is_none());
    }
}
