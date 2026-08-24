//! Loop（循环源）纯函数回归测试（集成测试目标）。
//!
//! 为什么放在 tests/ 而不是模块内 `#[cfg(test)]`：lib 单元测试 harness 在
//! Windows 上因缺少内嵌 ComCtl32 v6 清单无法启动（tauri_build 只给 bin 目标
//! 嵌清单）。本文件经由 build.rs 的 `/MANIFEST:EMBED` link-arg 获得清单，
//! 可以在 Windows 上直接运行：
//!
//! ```text
//! cargo test --features __test-internals --test loop_semantics
//! ```
//!
//! 同类单元测试也保留在 pitch_clip.rs / state.rs 的 `#[cfg(test)]` 模块中，
//! 在 Linux/CI 等无此限制的环境下照常执行。

use backend_lib::__test_internals::{
    leading_silence_sec, pitch_trim_window_sec, playback_window_sec, trim_and_resample_midi,
    Clip, SplitTransitionDurationUnit, SplitTransitionMode, SplitTransitionOptions, TimelineState,
};

/// 构造一个最小 Clip（仅几何字段参与窗口模型）。
fn make_clip(
    start_src: f64,
    end_src: f64,
    length: f64,
    rate: f32,
    reversed: bool,
    loop_enabled: bool,
) -> Clip {
    let mut tl = TimelineState::default();
    let track_id = tl.tracks[0].id.clone();
    let id = tl.add_clip(Some(track_id), Some("T".into()), Some(0.0), Some(length), None);
    {
        let c = tl.clips.iter_mut().find(|c| c.id == id).unwrap();
        c.source_start_sec = start_src;
        c.source_end_sec = end_src;
        c.playback_rate = rate;
        c.reversed = reversed;
        c.loop_enabled = loop_enabled;
    }
    tl.clips.into_iter().find(|c| c.id == id).unwrap()
}

/// 构造挂载在 TimelineState 上的 clip（供 split 等状态操作测试）。
fn make_state_clip(
    tl: &mut TimelineState,
    length: f64,
    start_src: f64,
    end_src: f64,
    reversed: bool,
) -> String {
    let track_id = tl.tracks[0].id.clone();
    let id = tl.add_clip(Some(track_id), Some("T".into()), Some(0.0), Some(length), None);
    {
        let c = tl.clips.iter_mut().find(|c| c.id == id).unwrap();
        c.source_start_sec = start_src;
        c.source_end_sec = end_src;
        c.playback_rate = 1.0;
        c.reversed = reversed;
        c.loop_enabled = false;
    }
    id
}

/// ── 倒放非循环 Clip 的**分割连续性**回归（用户工程实测形态）──────────
///
/// 倒放消费窗口锚定 se：整条 clip 消费 [se−len·r, se] 降序。分割后：
///   左段 = [se−S·r, se]、右段 = [se−len·r, se−S·r]
/// 两段在切割点共享同一源位置（内容连续），且右段锚点**绝不能被陈旧
/// ss 上钳**（此前 `.max(ss)` 把 1.15 抬回 5.53，窗口几乎全落媒体外）。
#[test]
fn split_reversed_nonloop_is_contiguous_and_anchor_correct() {
    let mut tl = TimelineState::default();
    // 用户工程同构：延伸窗倒放块（ss=5.53、se=9.25、len=10.41，媒体 10s）。
    let id = make_state_clip(&mut tl, 10.40917863594724, 5.526592881691009, 9.253863715024345, true);
    let (orig_ss, orig_se, orig_len) = (5.526592881691009f64, 9.253863715024345f64, 10.40917863594724f64);

    let s = 8.10318f64;
    let right_id = tl.split_clip(&id, s).expect("split should succeed");

    let left = tl.clips.iter().find(|c| c.id == id).unwrap();
    let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

    // 左段：长度 S；窗口 [se−S, se]，锚点保持不变。
    assert!((left.length_sec - s).abs() < 1e-9);
    assert!((left.start_sec - 0.0).abs() < 1e-9);
    assert!((left.source_end_sec - orig_se).abs() < 1e-9);
    assert!((left.source_start_sec - (orig_se - s)).abs() < 1e-9);

    // 右段：长度 R；锚点下移 S；窗口起点 = 原真实窗口起点（se−len）。
    assert!((right.start_sec - s).abs() < 1e-9);
    let right_len = orig_len - s;
    assert!((right.length_sec - right_len).abs() < 1e-9);
    let expect_right_se = orig_se - s;
    assert!((right.source_end_sec - expect_right_se).abs() < 1e-9, "right anchor must advance by S, got {}", right.source_end_sec);
    assert!(
        (right.source_start_sec - (expect_right_se - right_len)).abs() < 1e-9,
        "right window start must equal original true window start"
    );
    assert!((right.source_start_sec - (orig_se - orig_len)).abs() < 1e-9);

    // 内容连续性：左段最后可听源位置 == 右段第一可听源位置 == se−S。
    let (_, left_win_end) = playback_window_sec(left);
    let (right_win_start, right_win_end) = playback_window_sec(right);
    assert!((left_win_end - orig_se).abs() < 1e-9);
    assert!((right_win_end - (orig_se - s)).abs() < 1e-9);
    assert!((right_win_start - (orig_se - orig_len)).abs() < 1e-9);
}

/// 正放非循环分割：左段终点派生（不被陈旧存储 se 钳制）、右段窗口随之
/// 派生 —— 与倒放互为镜像，两方向逻辑完全对称。
#[test]
fn split_forward_nonloop_derives_windows_without_stale_clamp() {
    let mut tl = TimelineState::default();
    // 陈旧 se=3 与 len=6 脱钩（真实窗口 [0,6)，媒体 10s）。
    let id = make_state_clip(&mut tl, 6.0, 0.0, 3.0, false);

    let right_id = tl.split_clip(&id, 4.0).expect("split should succeed");
    let left = tl.clips.iter().find(|c| c.id == id).unwrap();
    let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

    assert!((left.length_sec - 4.0).abs() < 1e-9);
    assert!((left.source_start_sec - 0.0).abs() < 1e-9);
    assert!((left.source_end_sec - 4.0).abs() < 1e-9, "left end must derive from start+len, not clamp to stale se");
    assert!((right.start_sec - 4.0).abs() < 1e-9);
    assert!((right.source_start_sec - 4.0).abs() < 1e-9);
    assert!((right.source_end_sec - 6.0).abs() < 1e-9);
}

/// 分割过渡（ExtendOverlap）作用于倒放非循环：左右各向外延伸 g，
/// 锚点/窗口同步派生，重叠区内容相位保持（左尾与右头衔接同一源位置）。
#[test]
fn split_transition_extend_overlap_keeps_reversed_phase() {
    let mut tl = TimelineState::default();
    let id = make_state_clip(&mut tl, 4.0, 1.0, 5.0, true); // 窗口 [1,5]，len 4

    let opts = SplitTransitionOptions {
        enabled: true,
        mode: SplitTransitionMode::ExtendOverlap,
        duration_unit: SplitTransitionDurationUnit::Seconds,
        duration_sec: 0.5,
        duration_percent: 1.0,
        curve: None,
        overlap_fades: false,
    };
    let right_id = tl
        .split_clip_with_transition(&id, 3.0, &opts)
        .expect("split with transition should succeed");

    let left = tl.clips.iter().find(|c| c.id == id).unwrap();
    let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

    // 左段：len 3+0.5；窗口 [5−3.5, 5]=[1.5,5]（尾部向源下方延伸为静音区）。
    assert!((left.length_sec - 3.5).abs() < 1e-9);
    assert!((left.source_end_sec - 5.0).abs() < 1e-9);
    assert!((left.source_start_sec - 1.5).abs() < 1e-9);
    assert!((left.start_sec - 0.0).abs() < 1e-9);

    // 右段：start 3→2.5、len 1→1.5；锚点（分割后为 2）+0.5 → 2.5；
    // 窗口起点随派生 = 2.5−1.5 = 1.0。
    assert!((right.start_sec - 2.5).abs() < 1e-9);
    assert!((right.length_sec - 1.5).abs() < 1e-9);
    assert!((right.source_end_sec - 2.5).abs() < 1e-9);
    assert!((right.source_start_sec - 1.0).abs() < 1e-9);

    // 相位衔接：重叠区 [2.5, 3.5] 内左段（源位 5−τ）与右段
    // （源位 2.5−(τ−2.5)）逐点相等 —— 两侧听到相同内容。
    let (_, lwe) = playback_window_sec(left);
    let (rws, rwe) = playback_window_sec(right);
    assert!((lwe - 5.0).abs() < 1e-9);
    assert!((rwe - 2.5).abs() < 1e-9 && (rws - 1.0).abs() < 1e-9);
}

/// Loop + 媒体时长未知 + 环绕窗口（start > end，split 产生）：
/// 不得落入"空窗口"提前返回 —— 退化为整条缓存曲线回绕，输出非空且相位与
/// 逐帧 floor_mod 参考一致。
#[test]
fn loop_wrapped_window_without_media_duration_does_not_collapse_to_empty() {
    // 环绕窗口 [3.5, 3.0)（start > end），缓存曲线 100 帧（1s @10ms）。
    let full: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let out = trim_and_resample_midi(
        &full,
        10.0,
        3.5,  // source_start_sec（> end）
        3.0,  // source_end_sec
        1.0,  // playback_rate
        2.0,  // clip_timeline_len_sec → target = 200 帧
        true, // loop_enabled
        None, // 媒体时长未知
        false,
    );
    assert_eq!(out.len(), 200, "curve must cover the clip length");
    assert!(out.iter().any(|&v| v > 0.0), "curve must not be empty");

    // 与逐帧 floor_mod 参考对拍：idx = floor_mod(anchor + i, N)。
    let anchor = ((3.5f64 * 1000.0) / 10.0).round() as i64; // 350
    for (i, v) in out.iter().enumerate() {
        let expect = full[(anchor + i as i64).rem_euclid(100) as usize];
        assert!((v - expect).abs() < 1e-6, "frame {i}: {v} != {expect}");
    }
}

/// Loop + 已知媒体时长：逐帧 floor_mod(anchor ± round(i·rate), N) 映射正确
/// （正放，负锚点环绕），且长度等于 clip 时间线帧数。
#[test]
fn loop_with_media_duration_maps_per_frame_floor_mod() {
    // 媒体 1s（100 帧 @10ms），锚点 -0.25s（负值环绕到尾部一侧）。
    let full: Vec<f32> = (0..100).map(|i| (i * 7) as f32 % 13.0).collect();
    let out = trim_and_resample_midi(&full, 10.0, -0.25, 1.0, 1.0, 1.5, true, Some(1.0), false);
    assert_eq!(out.len(), 150);
    let anchor = ((-0.25f64 * 1000.0) / 10.0).round() as i64; // -25
    for (i, v) in out.iter().enumerate() {
        let idx = (anchor + i as i64).rem_euclid(100);
        assert!((v - full[idx as usize]).abs() < 1e-6, "frame {i}");
    }
}

/// 非 Loop 行为回归：source_end 按 起点+长度×速率 **派生**。
///
/// - 陈旧 source_end（历史工程/循环开关切换残留）必须被自愈：
///   窗口不得被截短到旧值，否则静音区冻结、音频被错误丢弃；
/// - 一致输入下输出恰好铺满 clip 长度，内容按窗口顺序保留。
#[test]
fn non_loop_derives_source_end_from_start_and_length() {
    let full: Vec<f32> = (0..300).map(|i| i as f32).collect();

    // 存储窗口 [0.5, 1.5) 与长度 2s 脱钩（陈旧值）：
    // 派生终点 = 0.5 + 2·1 = 2.5 → 输出 200 帧，内容为源 [0.5, 2.5)。
    let out = trim_and_resample_midi(&full, 10.0, 0.5, 1.5, 1.0, 2.0, false, None, false);
    assert_eq!(out.len(), 200, "derived end must heal the stale window");
    for (i, v) in out.iter().enumerate() {
        assert!((v - full[50 + i]).abs() < 1e-6, "frame {i} content preserved");
    }

    // 一致输入（se == start + len·rate）行为不变。
    let out2 = trim_and_resample_midi(&full, 10.0, 0.5, 2.5, 1.0, 2.0, false, None, false);
    assert_eq!(out2, out);
}

/// ── 倒放非循环 Clip 的**消费窗口模型**回归（本次修复的核心）────────────
///
/// 倒放 Clip 的消费窗口 = [se−len·r, se]（锚定 se，ss 不参与）；
/// se>D → 前导静音；窗口下探 <0 → 尾部静音。
#[test]
fn reversed_nonloop_window_model_anchors_at_source_end() {
    // 正放：win=[ss, ss+len·r)。
    let fwd = make_clip(2.0, 6.0, 4.0, 1.0, false, false);
    assert_eq!(playback_window_sec(&fwd), (2.0, 6.0));
    // 正放 ss<0 → 前导静音 1s。
    let fwd_neg = make_clip(-1.0, 3.0, 4.0, 1.0, false, false);
    assert_eq!(playback_window_sec(&fwd_neg), (-1.0, 3.0));
    assert!((leading_silence_sec(&fwd_neg, Some(10.0)) - 1.0).abs() < 1e-9);

    // 倒放：win=[se−len·r, se)；陈旧/域外的 ss 不得影响窗口。
    let rev = make_clip(-99.0, 6.0, 4.0, 1.0, true, false);
    assert_eq!(playback_window_sec(&rev), (2.0, 6.0));
    // 倒放 ss<0 是尾部静音：不产生前导静音。
    assert!(leading_silence_sec(&rev, Some(10.0)).abs() < 1e-12);

    // 倒放 se 越过媒体末端（trim_left 延伸 / split 过渡可达）→ 前导静音。
    let rev_over = make_clip(0.0, 12.0, 4.0, 1.0, true, false);
    assert_eq!(playback_window_sec(&rev_over), (8.0, 12.0));
    assert!((leading_silence_sec(&rev_over, Some(10.0)) - 2.0).abs() < 1e-9);
    // 媒体时长未知：按无前导静音兜底。
    assert!(leading_silence_sec(&rev_over, None).abs() < 1e-12);

    // Loop：负 ss 是环绕锚点 —— 恒无前导静音。
    let looped = make_clip(-5.0, 5.0, 4.0, 1.0, false, true);
    assert!(leading_silence_sec(&looped, Some(10.0)).abs() < 1e-12);

    // trim 窗口重定向：非 Loop 倒放 → [se−len·r, se]；正放/Loop 透传。
    let (ts, te) = pitch_trim_window_sec(&rev_over);
    assert_eq!((ts, te), (8.0, 12.0));
    let (fs, fe) = pitch_trim_window_sec(&fwd);
    assert_eq!((fs, fe), (2.0, 6.0));
}

/// 非 Loop 正放 Clip 的窗口下探媒体起点之前（slip 左移 / 左延伸可达）：
/// 音高曲线必须以**前导静音**开头，内容对齐到真实消费位置 —— 而不是把
/// 窗口内少量素材拉伸铺满整条 clip。
#[test]
fn nonloop_pitch_curve_places_leading_silence_before_media_start() {
    let full: Vec<f32> = (0..300).map(|i| i as f32).collect();
    // 窗口 [-0.5, 0.5)：前 0.5s 为前导静音，后 0.5s 对应源帧 [0, 50)。
    let clip = make_clip(-0.5, 0.5, 1.0, 1.0, false, false);
    assert_eq!(playback_window_sec(&clip), (-0.5, 0.5));
    assert!((leading_silence_sec(&clip, Some(3.0)) - 0.5).abs() < 1e-9);

    let out = trim_and_resample_midi(&full, 10.0, -0.5, 0.5, 1.0, 1.0, false, None, false);
    assert_eq!(out.len(), 100, "curve must cover the clip length");
    for i in 0..50 {
        assert_eq!(out[i], 0.0, "frame {i} must be leading silence");
    }
    for i in 50..100 {
        assert!(
            (out[i] - full[i - 50]).abs() < 1e-6,
            "frame {i} must be source frame {}",
            i - 50
        );
    }
}

/// 用户工程实测场景：倒放块（窗口 [se−len, se]，尾部为媒体下方静音）在
/// 静音区内分割 —— 右段窗口必须**整体保持在媒体域下方**（纯静音、保持
/// 偏移），左段正常衔接。任何一段被钳回媒体域都会凭空出现波形/声音。
#[test]
fn split_reversed_tail_silence_right_piece_stays_below_media() {
    let mut tl = TimelineState::default();
    // 1_ori 同构：len=10.40918、ss=0、se=D=3.72727 → 消费窗口 [−6.68191, D]。
    let id = make_state_clip(
        &mut tl,
        10.40917863594724,
        0.0,
        3.7272708333333338,
        true,
    );
    let d = 3.7272708333333338f64;
    let orig_len = 10.40917863594724f64;

    let s = 6.5f64; // 分割点位于静音尾部（> D）
    let right_id = tl.split_clip(&id, s).expect("split should succeed");
    let left = tl.clips.iter().find(|c| c.id == id).unwrap();
    let right = tl.clips.iter().find(|c| c.id == right_id).unwrap();

    // 左段：窗口 [se−S, se] —— 媒体 + 尾部静音，偏移保持。
    assert!((left.length_sec - s).abs() < 1e-9);
    assert!((left.source_end_sec - d).abs() < 1e-9);
    assert!((left.source_start_sec - (d - s)).abs() < 1e-9);

    // 右段：窗口 [se−len₀, se−S] 整体 < 0 —— 纯静音，偏移保持。
    assert!((right.start_sec - s).abs() < 1e-9);
    assert!((right.length_sec - (orig_len - s)).abs() < 1e-9);
    let expect_se = d - s;
    assert!((right.source_end_sec - expect_se).abs() < 1e-9, "right anchor = {} must stay below media", right.source_end_sec);
    assert!(right.source_end_sec < 0.0, "right anchor must remain in silent domain");
    let (rws, rwe) = playback_window_sec(right);
    assert!(rwe < 0.0 && rws < rwe, "entire window must stay below media, got [{rws}, {rwe}]");

    // 无前导静音（锚点未越过媒体末端）；静音完全由"窗口在媒体外"表达。
    assert!(leading_silence_sec(right, Some(d)).abs() < 1e-12);
}

/// 倒放非循环 Clip 的音高曲线必须取自真实消费窗口 [se−len·r, se]
/// （此前取存储 [ss, se]，延伸过的窗口把曲线拉伸到错误区域）。
#[test]
fn reversed_nonloop_pitch_curve_consumes_top_of_window() {
    // 缓存曲线 300 帧（3s @10ms），值 = 帧号。
    let full: Vec<f32> = (0..300).map(|i| i as f32).collect();
    // 倒放 Clip：存储窗口 [0.5, 2.5]、长度 1s、rate=1
    // → 真实消费窗口 [se−len·r, se] = [1.5, 2.5)（源帧 150..250）。
    let clip = make_clip(0.5, 2.5, 1.0, 1.0, true, false);
    let (ws, we) = playback_window_sec(&clip);
    assert!((ws - 1.5).abs() < 1e-9 && (we - 2.5).abs() < 1e-9);

    // 模拟调用方：helper 重定向后的 trim 实参（内部按升序处理，输出再翻转）。
    let (ts, te) = pitch_trim_window_sec(&clip);
    let out = trim_and_resample_midi(&full, 10.0, ts, te, 1.0, 1.0, false, None, false);
    assert_eq!(out.len(), 100);
    for (i, v) in out.iter().enumerate() {
        assert!(
            (v - full[150 + i]).abs() < 1e-6,
            "frame {i} must come from source frame {}",
            150 + i
        );
    }
}
