use crate::state::AppState;
use tauri::State;

pub(super) fn ping() -> serde_json::Value {
    serde_json::json!({ "ok": true, "message": "pong" })
}

pub(super) fn get_runtime_info(state: State<'_, AppState>) -> crate::models::RuntimeInfoPayload {
    state.runtime_info()
}

/// 关于对话框数据：版本号、commit（非 git 构建为 null）、脏标志与
/// GitHub 仓库链接（上游非 GitHub 时为 null，前端回退固定链接）。
///
/// 形状由 `hifishifter_kernel::build_info::about_payload` 定义，与插件的
/// `get_about_info` **同一份实现** —— 两边各拼一次 JSON 时，插件那边就少报了
/// commit 与仓库链接（前端只当字段为空，不报错），因此不再给第二处可写歪的机会。
pub(super) fn get_about_info() -> serde_json::Value {
    hifishifter_kernel::build_info::about_payload("standalone", crate::build_info::version())
}

pub(super) fn consume_startup_project_path(state: State<'_, AppState>) -> serde_json::Value {
    let path = state.take_pending_startup_project_path();
    serde_json::json!({ "ok": true, "path": path })
}

/// 前端已注册好 hs:external-project-open 监听；此前的 Opened 路径留在 pending。
pub(super) fn mark_frontend_ready(state: State<'_, AppState>) -> serde_json::Value {
    state.mark_frontend_ready();
    serde_json::json!({ "ok": true })
}

pub(super) fn set_ui_locale(state: State<'_, AppState>, locale: String) -> serde_json::Value {
    let locale = locale.trim();
    let lower = locale.to_lowercase();
    let normalized = if locale.eq_ignore_ascii_case("zh-CN")
        || locale.eq_ignore_ascii_case("zh_CN")
        || lower.starts_with("zh")
    {
        "zh-CN".to_string()
    } else if locale.eq_ignore_ascii_case("ja-JP")
        || locale.eq_ignore_ascii_case("ja_JP")
        || lower.starts_with("ja")
    {
        "ja-JP".to_string()
    } else if locale.eq_ignore_ascii_case("ko-KR")
        || locale.eq_ignore_ascii_case("ko_KR")
        || lower.starts_with("ko")
    {
        "ko-KR".to_string()
    } else {
        // Default to en-US for unknown values.
        "en-US".to_string()
    };

    {
        let mut guard = state.ui_locale.write().unwrap_or_else(|e| e.into_inner());
        *guard = normalized.clone();
    }

    serde_json::json!({"ok": true, "locale": normalized})
}

pub(super) fn get_timeline_state(
    state: State<'_, AppState>,
) -> crate::models::TimelineStatePayload {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

pub(crate) fn get_timeline_state_from_ref(state: &AppState) -> crate::models::TimelineStatePayload {
    let tl = state
        .timeline
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let mut payload = tl.to_payload();
    payload.project = Some(state.project_meta_payload());
    payload
}

/// Lightweight timeline state — skips waveform_preview, pitch_range, and
/// midi_note_data to reduce clone+serialize cost.
///
/// 当前前端未使用（仍轮询全量 `get_timeline_state`）；保留作为轮询瘦身入口。
pub(super) fn get_timeline_state_lite(
    state: State<'_, AppState>,
) -> crate::models::TimelineStatePayload {
    // to_payload_lite 接受引用：直接在锁内构建 payload，
    // 省掉整棵 TimelineState（含全部参数曲线）的深克隆。
    let mut payload = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.to_payload_lite()
    };
    payload.project = Some(state.project_meta_payload());
    payload
}

/// 把"这次 set_transport 是否携带定位意图"解析成显式的 seek 目标。
///
/// 【为什么单独成函数】这条判据曾经是一条无条件语句（`seek_sec(tl.playhead_sec)`），
/// 于是"只改 BPM"也会把引擎拽回时间线里的陈旧播放头。把它抽出来之后，"没有意图
/// 就不定位"这件事有了可断言的形式（见文件末尾的测试），而不是埋在命令体的一句
/// 注释里。
///
/// 返回 `None` = 不定位；`Some(sec)` = 定位到该秒（负值钳到 0）。
fn resolve_seek_intent(playhead_sec: Option<f64>) -> Option<f64> {
    playhead_sec
        .filter(|value| value.is_finite())
        .map(|value| value.max(0.0))
}

pub(super) fn set_transport(
    state: State<'_, AppState>,
    playhead_sec: Option<f64>,
    bpm: Option<f64>,
) -> serde_json::Value {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        log::warn!(
            "set_transport(playhead_sec={:?}, bpm={:?})",
            playhead_sec,
            bpm
        );
    }
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let prev_bpm = tl.bpm;
    // 只有**显式**给了 playhead_sec 才动引擎位置。
    //
    // 【为什么必须记下"这次有没有定位意图"】引擎位置是起播的唯一权威，而这
    // 条命令还被用来"只改 BPM"（`updateTransportBpm`）与"只写回锚点"。
    // 无条件 `seek_sec(tl.playhead_sec)` 会在这些路径上把引擎拽回一个**陈旧**
    // 的播放头（时间线里的 playhead_sec 在播放期间并不前进），下一次播放就
    // 从那个旧点起播 —— 这正是"改个 BPM 光标就跳回旧起点"的来源。
    let seek_target = resolve_seek_intent(playhead_sec);
    if let Some(sec) = seek_target {
        tl.playhead_sec = sec;
    }
    if let Some(v) = bpm {
        if v.is_finite() && v > 0.0 {
            // BPM is project-affecting: checkpoint for undo.
            state.checkpoint_timeline(&tl, crate::state::HistoryOp::EditTempo);
            // 与 Tempo Map 规范化/前端 clampBpm 一致：钳制到 10-960，
            // 否则这里可直接把 Tempo Map 初始点 BPM 写出合法范围。
            let clamped = v.clamp(10.0, 960.0);
            tl.bpm = clamped;
            // Tempo Map 存在时，工程 BPM 与 0 位置点保持一致。
            if let Some(points) = tl.tempo_map.as_mut() {
                if let Some(first) = points.first_mut() {
                    first.bpm = clamped;
                }
            }
        }
    }

    // Keep realtime engine transport aligned **only** when this call carried an
    // explicit seek intent. BPM-only updates must not move the transport.
    if let Some(sec) = seek_target {
        state.audio_engine.seek_sec(sec);
    }
    let bpm_changed = (tl.bpm - prev_bpm).abs() > 1e-9;
    if bpm_changed {
        state.audio_engine.update_timeline(tl.clone());
    }
    let playhead = tl.playhead_sec;
    let bpm_now = tl.bpm;

    // BPM 变化会改变节拍器响点表（时间锚定），重建之。其余 seek 场景
    // 响点表不变，无需重建。
    //
    // ⚠ 必须先 `drop(tl)` 再刷新：refresh_metronome_schedule 内部会重新
    // 加 timeline 锁，std Mutex 不可重入，锁内调用 = 自我死锁
    // （症状：修改 BPM 后应用整体未响应）。
    drop(tl);
    if bpm_changed {
        crate::commands::playback::refresh_metronome_schedule(&state);
    }

    serde_json::json!({"ok": true, "playhead_sec": playhead, "bpm": bpm_now })
}

// ===================== undo / redo =====================

pub(super) fn undo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    state.undo_timeline()
}

pub(super) fn redo_timeline(state: State<'_, AppState>) -> crate::models::TimelineStatePayload {
    state.redo_timeline()
}

#[cfg(test)]
mod tests {
    use super::resolve_seek_intent;

    /// 只改 BPM 时**不得**动引擎位置：这是"改个 BPM 光标就跳回旧起点"的直接根因。
    #[test]
    fn a_bpm_only_update_carries_no_seek_intent() {
        assert_eq!(resolve_seek_intent(None), None);
    }

    /// 显式定位必须被采纳，且负值钳到 0（时间线没有负时间）。
    #[test]
    fn an_explicit_playhead_is_clamped_and_forwarded() {
        assert_eq!(resolve_seek_intent(Some(12.5)), Some(12.5));
        assert_eq!(resolve_seek_intent(Some(-3.0)), Some(0.0));
        assert_eq!(resolve_seek_intent(Some(0.0)), Some(0.0));
    }

    /// 非法读数不得变成"定位到 NaN"（会让帧计数锁死在 0 或 UB 级行为差异）。
    #[test]
    fn a_non_finite_playhead_never_reaches_the_engine_as_nan() {
        assert_eq!(resolve_seek_intent(Some(f64::NAN)), None);
        assert_eq!(resolve_seek_intent(Some(f64::INFINITY)), None);
        assert_eq!(resolve_seek_intent(Some(f64::NEG_INFINITY)), None);
    }
}
