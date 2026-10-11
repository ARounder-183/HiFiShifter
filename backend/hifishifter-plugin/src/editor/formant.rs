/*
 * editor/formant.rs - Clip 源共振峰分析命令（插件侧）。
 *
 * 与独立 App 的 `commands/formant.rs` 载荷**逐字段同形**：前端共振峰工具窗口在两种
 * 模式下读同一份 TS 类型（`ClipFormantAnalysisResult`），所以字段名必须保持一致。
 *
 * 实现直接复用 App 的 `formant_cache::get_or_compute_formant_analysis`：插件在加载时
 * 已经把宿主经 ARA 授权的 PCM 物化成 `clip.source_path` 指向的**私有分析副本**
 * （`kernel::editor::host_pcm::materialize`），所以这里解码的是插件自己的 WAV，不是
 * 宿主媒体文件。切片 / 倒放预反转 / 通道平均因此与 App 逐字一致 —— 同一 clip 在两种
 * 模式里显示相同的共振峰位置。
 */

use super::session::EditorSession;
use crate::sync::MutexExt;
use serde::Serialize;
use serde_json::Value;

/// Clip 源共振峰分析载荷（camelCase，与 App 端 `ClipFormantAnalysisPayload` 一致）。
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ClipFormantAnalysisPayload {
    ok: bool,
    source_f1_hz: f32,
    source_f2_hz: f32,
    track: Vec<[f32; 3]>,
    voiced_ratio: f32,
    message: Option<String>,
}

impl EditorSession {
    /// 计算并返回 clip 的源共振峰分析（供 `analyze_clip_formants` 命令调用）。
    pub(super) fn analyze_clip_formants(&self, input: &Value) -> Result<Value, String> {
        let clip_id = input["clipId"].as_str().ok_or("clipId missing")?;
        let clip = self
            .timeline
            .lock_or_recover()
            .clips
            .iter()
            .find(|clip| clip.id == clip_id)
            .cloned()
            .ok_or("clip_not_found")?;
        let summary = hifishifter_kernel::formant_cache::get_or_compute_formant_analysis(&clip)?;
        serde_json::to_value(ClipFormantAnalysisPayload {
            ok: true,
            source_f1_hz: summary.source_f1_hz,
            source_f2_hz: summary.source_f2_hz,
            track: summary
                .track
                .iter()
                .map(|(time, f1, f2)| [*time, *f1, *f2])
                .collect(),
            voiced_ratio: summary.voiced_ratio,
            message: summary.message.map(|message| message.to_string()),
        })
        .map_err(|error| error.to_string())
    }
}
