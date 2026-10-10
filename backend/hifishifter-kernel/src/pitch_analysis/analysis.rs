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

/// 组装门禁：该根是否**可能**有组装任务（不含"是否已是最新"这一条）。
///
/// 【不变量：门禁必须覆盖渲染门禁】`pitch_editing::does_clip_need_processor_render`
/// 为真表示"渲染要等原线"。如果分析门禁对它为假，就会出现"渲染门禁在等一个永远不会
/// 被调度的任务"——`pitch_orig_key` 永不置位，插件永久 `pitch analysis pending`
/// （用户报障：关掉合成后左下角一直"渲染中"）。因此这里**必须**对渲染门禁的每一条
/// 成因为真；最后一段兜底就是把这条不变量写成代码，而不是靠两处各改各的。
///
/// 【为什么调用方也各自另写判据是错的】此前独立 App 有一份、插件侧又有一份，三份对
/// 同一个根给出相反结论。现在内核这一份是唯一权威，调用方一律引用它。
fn assembly_gate_open(tl: &TimelineState, root_track_id: &str) -> bool {
    // 轨道不存在时按"默认配置"计（compose 关、默认算法）—— 不要用 `Unknown`
    // 当哨兵：它现在是一个有明确执行语义的值（见 `PitchAnalysisAlgo::effective`）。
    let (compose_enabled, algo) = tl
        .tracks
        .iter()
        .find(|t| t.id == root_track_id)
        .map(|t| (t.compose_enabled, t.pitch_analysis_algo.clone()))
        .unwrap_or((false, PitchAnalysisAlgo::default()));

    if matches!(algo, PitchAnalysisAlgo::None) {
        return false;
    }

    // 检查是否存在非静音的音高参考块（MIDI clip），若存在则即使 compose_enabled 为 false
    // 也需要触发 pitch_orig 组装，确保音高参考块的数据能写入 pitch_edit 并影响渲染。
    let has_active_midi_clip = tl.clips.iter().any(|c| {
        tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id)
            && !c.muted
            && c.midi_note_data.is_some()
    });
    if compose_enabled || has_active_midi_clip {
        return true;
    }

    // 兜底：渲染门禁的每一条成因（手绘曲线、已生效的音高调整、子轨共振峰偏移……）都
    // 必须能唤起分析。`does_clip_need_processor_render` 是渲染门禁的逐 clip 判据，
    // 直接复用它即得"分析门禁 ⊇ 渲染门禁"。
    tl.clips.iter().any(|c| {
        tl.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id)
            && crate::pitch_editing::does_clip_need_processor_render(tl, c, c.start_sec)
    })
}

/// 该根轨道是否有**待组装**的原线（`pitch_orig_key` 尚未置位）。
///
/// 【用途】渲染门禁的"要不要等分析"判据：返回 `false` 表示"没有组装任务"，调用方
/// **不应**把渲染判为 pending —— 否则一个永远不会被调度任务满足的等待会把渲染卡死。
/// 与 `build_pitch_job` 共用 [`assembly_gate_open`]，因此"分析器要不要分析"与"门禁
/// 要不要等"永远一致。
pub fn root_pitch_assembly_pending(tl: &TimelineState, root_track_id: &str) -> bool {
    assembly_gate_open(tl, root_track_id)
        && tl
            .params_by_root_track
            .get(root_track_id)
            .is_some_and(|e| e.pitch_orig_key.is_none())
}

/// 判定某根轨道当前是否需要（重新）组装 `pitch_orig`；需要时返回本次装配的缓存 key。
///
/// 返回 `None` 表示无需动作，任一条成立即跳过：
/// - [`assembly_gate_open`] 为假（既未开启 Compose、也没有非静音的音高参考块、
///   且当前没有已生效的音高调整；或算法为 `PitchAnalysisAlgo::None`）；
/// - 现有 `pitch_orig` 的 key 与长度都已是最新。
pub fn build_pitch_job(tl: &TimelineState, root_track_id: &str) -> Option<PitchJob> {
    if !assembly_gate_open(tl, root_track_id) {
        return None;
    }
    let fp = tl.frame_period_ms();
    let target = tl.target_param_frames(fp);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TrackParamsState;

    fn timeline(compose: bool) -> TimelineState {
        let mut tl = TimelineState::default();
        tl.tracks = serde_json::from_value(serde_json::json!([
            {"id":"root","name":"root","order":0,"compose_enabled":compose,
             "pitch_analysis_algo":"nsf_hifigan_onnx"}
        ]))
        .unwrap();
        tl.clips = serde_json::from_value(serde_json::json!([
            {"id":"clip","track_id":"root","name":"source","start_sec":0.0,"length_sec":1.0,
             "takes":[{"id":"take","source_start_sec":0.0,"source_end_sec":1.0}]}
        ]))
        .unwrap();
        tl.params_by_root_track
            .insert("root".into(), TrackParamsState::default());
        tl
    }

    /// **不变量：渲染门禁要等的原线，分析门禁必须能产出。**
    ///
    /// 【为什么这是回归测试】分析器的门禁此前不看"手绘曲线"，而渲染门禁
    /// （`does_clip_need_processor_render`）看。两者对同一个根给出相反结论：渲染门禁
    /// 在等一个永远不会被调度的任务 ⇒ 插件永久 "pitch analysis pending"，以 150 ms
    /// 一轮无限重试（用户报障：关掉合成后左下角一直"渲染中"）。插件曾用"在分析快照
    /// 里强制打开 compose"绕过，但 `build_root_pitch_key` 把 compose 计入键，于是分析
    /// 键与会话键不符、`pitch_orig_key` 被反复清空 —— 绕过本身也是错的。
    /// 现在分析门禁直接复用渲染门禁的逐 clip 判据，覆盖关系**由构造保证**。
    #[test]
    fn the_assembly_gate_covers_every_render_gate_cause() {
        let mut tl = timeline(false);
        // 造一条"确实需要处理器渲染"的 clip：手绘曲线落在 clip 的时间范围内。
        {
            let params = tl.params_by_root_track.get_mut("root").unwrap();
            params.pitch_edit_user_modified = true;
            params.frame_period_ms = 10.0;
            params.pitch_edit = vec![1.0; 200];
        }
        assert!(
            tl.clips
                .iter()
                .any(|c| crate::pitch_editing::does_clip_need_processor_render(
                    &tl,
                    c,
                    c.start_sec
                )),
            "前提：这条 clip 确实需要处理器渲染"
        );
        assert!(
            root_pitch_assembly_pending(&tl, "root"),
            "渲染门禁要等的原线，分析门禁必须能产出 —— 否则渲染会永久 pending"
        );
        assert!(
            build_pitch_job(&tl, "root").is_some(),
            "同一组门禁下，分析器也必须认为有任务"
        );
    }

    /// 组装完成后（`pitch_orig_key` 已置位）不再 pending —— 否则渲染门禁会永久卡住。
    #[test]
    fn a_converged_root_is_not_pending() {
        let mut tl = timeline(true);
        let key = build_root_pitch_key(&tl, "root");
        let target = tl.target_param_frames(tl.frame_period_ms());
        {
            let params = tl.params_by_root_track.get_mut("root").unwrap();
            params.pitch_orig_key = Some(key);
            params.pitch_orig = vec![0.0; target];
        }
        assert!(!root_pitch_assembly_pending(&tl, "root"));
        assert!(build_pitch_job(&tl, "root").is_none());
    }

    /// 算法为 `None` 时不产生任务（纯混音不受门禁阻塞）。
    #[test]
    fn a_disabled_algorithm_is_never_pending() {
        let mut tl = timeline(true);
        tl.tracks[0].pitch_analysis_algo = PitchAnalysisAlgo::None;
        tl.params_by_root_track
            .get_mut("root")
            .unwrap()
            .pitch_edit_user_modified = true;
        assert!(!root_pitch_assembly_pending(&tl, "root"));
    }
}
