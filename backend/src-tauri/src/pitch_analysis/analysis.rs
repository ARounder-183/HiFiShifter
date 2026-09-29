// pitch_analysis::analysis — 根轨道音高任务判定。
//
// 【历史】本模块曾包含一整套「增量刷新 + 并行分析 + 音高融合」流水线：
// `build_timeline_snapshot` / `compare_snapshots` / `determine_clips_to_analyze` /
// `analyze_clip_with_cache` / `process_single_clip` / `compute_pitch_curve_parallel` /
// `compute_pitch_curve_with_incremental_refresh` / `fuse_clip_pitches_optimized` /
// `compute_pitch_curve`。音高分析改为「per-clip 全量分析 + 全局缓存 + 组装期按需
// 截取」（`pitch_clip` + `schedule`）之后，那些函数就再没有调用者了，却仍留在
// 文件里，还连带让 `clip_pitch_cache` / `pitch_progress` 两个模块以及
// `AppState` 的 `clip_pitch_cache` / `pitch_timeline_snapshot` 两个字段看起来
// 「有人用」——排查内存问题时它们把注意力引向了错误的缓存。
//
// 整块删除，只保留仍然存活的 `build_pitch_job`。

use crate::state::{PitchAnalysisAlgo, TimelineState};

use super::{build_root_pitch_key, PitchJob};

/// 判定某根轨道当前是否需要（重新）组装 `pitch_orig`；需要时返回本次装配的缓存 key。
///
/// 返回 `None` 表示无需动作，任一条成立即跳过：
/// - 既未开启 Compose、也没有非静音的音高参考块（MIDI clip）、且当前也没有已生效的
///   音高调整（后者用于在 MIDI clip 被静音后仍触发一次组装，以清除标志与数据）；
/// - 算法为 `PitchAnalysisAlgo::None`；
/// - 现有 `pitch_orig` 的 key 与长度都已是最新。
pub(crate) fn build_pitch_job(tl: &TimelineState, root_track_id: &str) -> Option<PitchJob> {
    let fp = tl.frame_period_ms();
    let target = tl.target_param_frames(fp);

    let (compose_enabled, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::Unknown));

    // 检查是否存在非静音的音高参考块（MIDI clip），若存在则即使 compose_enabled 为 false
    // 也需要触发 pitch_orig 组装，确保音高参考块的数据能写入 pitch_edit 并影响渲染。
    let has_active_midi_clip = tl.clips.iter().any(|c| {
        tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id)
            && !c.muted
            && c.midi_note_data.is_some()
    });

    // 若当前 params 中已记录 has_pitch_adjustment_active，则即使所有 MIDI clip 都被静音，
    // 也应触发组装以清除标志和对应的音高数据。
    let currently_has_adjustment = tl
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.has_pitch_adjustment_active)
        .unwrap_or(false);

    if !compose_enabled && !has_active_midi_clip && !currently_has_adjustment {
        return None;
    }
    if matches!(algo, PitchAnalysisAlgo::None) {
        return None;
    }

    let key = build_root_pitch_key(tl, root_track_id);

    // If already up-to-date, do nothing.
    let is_up_to_date = tl
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.pitch_orig_key.as_deref() == Some(&key) && e.pitch_orig.len() == target)
        .unwrap_or(false);
    if is_up_to_date {
        return None;
    }

    Some(PitchJob {
        root_track_id: root_track_id.to_string(),
        key,
    })
}
