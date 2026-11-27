//! 增益**分层**的契约：item 层与 take 层是两个旋钮，任何一次写入只能动一层。
//!
//! 【为什么单独一个文件】这条不变量跨三个模块（内核的 patch、插件的写回回执、前端的
//! 载荷字段名），而它此前被违反过一次：时间轴的行内编辑把 **item** 层的值发在裸
//! `gain` 上，而后端把裸 `gain` 读成 **take** 层。现场因此出现一对镜像读数
//! （`gain=4.0 expected 1.0` 与 `gain=1.0 expected 4.0`），用户看到的是"旋钮回弹"。

use super::model::{ClipStatePatch, TimelineState};

/// 造一条只有一个 clip 的时间线（active take 与 flat 投影已同步）。
fn timeline_with_clip() -> (TimelineState, String) {
    let mut timeline = TimelineState::default();
    let track_id = timeline.tracks[0].id.clone();
    let clip_id = timeline.add_clip(
        Some(track_id),
        Some("G".into()),
        Some(0.0),
        Some(2.0),
        Some("C:/audio/g.wav".into()),
    );
    if let Some(clip) = timeline.clips.iter_mut().find(|c| c.id == clip_id) {
        clip.sync_take_from_flat();
    }
    (timeline, clip_id)
}

/// **不变量：take 层补丁只改 take 层。**
#[test]
fn a_take_gain_patch_never_changes_the_item_gain() {
    let (mut timeline, clip_id) = timeline_with_clip();
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            take_gain: Some(4.0),
            ..Default::default()
        },
    );
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.gain - 4.0).abs() < 1e-6, "take 层必须改");
    assert!(
        (clip.clip_gain - 1.0).abs() < 1e-6,
        "item 层不得被 take 层补丁带上"
    );
    assert!((clip.takes[0].gain - 4.0).abs() < 1e-6, "权威值在 take 上");
}

/// **不变量：item 层补丁只改 item 层。**
#[test]
fn an_item_gain_patch_never_changes_the_take_gain() {
    let (mut timeline, clip_id) = timeline_with_clip();
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            item_gain: Some(4.0),
            ..Default::default()
        },
    );
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.clip_gain - 4.0).abs() < 1e-6, "item 层必须改");
    assert!(
        (clip.gain - 1.0).abs() < 1e-6,
        "take 层不得被 item 层补丁带上"
    );
    assert!((clip.takes[0].gain - 1.0).abs() < 1e-6);
}

/// **不变量：旧前端的裸 `gain` 被读成 take 层。**
///
/// 【为什么保留这条别名】改名不能把已经保存的载荷读丢。但新代码一律发 `takeGain` ——
/// 别名只服务旧数据，不是"裸 gain 也可以继续用"。
#[test]
fn the_legacy_gain_key_is_read_as_the_take_layer() {
    let patch: ClipStatePatch = serde_json::from_value(serde_json::json!({
        "gain": 2.0,
    }))
    .unwrap();
    assert_eq!(patch.take_gain, Some(2.0));
    assert_eq!(patch.item_gain, None);

    let named: ClipStatePatch = serde_json::from_value(serde_json::json!({
        "takeGain": 3.0,
        "itemGain": 4.0,
    }))
    .unwrap();
    assert_eq!(named.take_gain, Some(3.0));
    assert_eq!(named.item_gain, Some(4.0));
}

/// **不变量：+12 dB 再归零，两层各自回到 1.0。**
///
/// 【为什么这条要端到端跑一遍】用户报障的形状是"先旋钮 +12 dB，再行内编辑回 0"。
/// 单层的用例都能过，而两层的顺序组合才暴露"写错层"：item 停在 4.0、take 变成 1.0。
#[test]
fn a_twelve_db_round_trip_leaves_both_layers_at_unity() {
    let (mut timeline, clip_id) = timeline_with_clip();
    // 旋钮拖拽：item 层 +12 dB。
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            item_gain: Some(4.0),
            ..Default::default()
        },
    );
    // 行内编辑 / 双击复位：item 层回到 0 dB。
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            item_gain: Some(1.0),
            ..Default::default()
        },
    );
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.clip_gain - 1.0).abs() < 1e-6, "item 层必须回到 1.0");
    assert!(
        (clip.gain - 1.0).abs() < 1e-6,
        "take 层全程没被碰过，必须仍是 1.0"
    );
    assert!((clip.takes[0].gain - 1.0).abs() < 1e-6);
}

/// 可听增益是两层的**乘积**：只改一层就只改变一层。
#[test]
fn the_audible_gain_is_the_product_of_both_layers() {
    let (mut timeline, clip_id) = timeline_with_clip();
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            take_gain: Some(2.0),
            item_gain: Some(3.0),
            ..Default::default()
        },
    );
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.audible_gain() - 6.0).abs() < 1e-5);
}

/// **不变量：normalize_takes 只物化投影，绝不在层间搬增益。**
#[test]
fn normalize_takes_never_moves_gain_between_layers() {
    let (mut timeline, clip_id) = timeline_with_clip();
    {
        let clip = timeline.clips.iter_mut().find(|c| c.id == clip_id).unwrap();
        clip.clip_gain = 1.0;
        clip.gain = 0.5;
        clip.sync_take_from_flat();
        clip.normalize_takes();
    }
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.clip_gain - 1.0).abs() < 1e-6, "item 层不得被搬走");
    assert!(
        (clip.takes[0].gain - 0.5).abs() < 1e-6,
        "take 层必须留在原地"
    );
    assert!((clip.gain - 0.5).abs() < 1e-6, "扁平投影随 take");
}

/// **回归：反相 take 不是静音。**
#[test]
fn a_negative_take_gain_is_not_silence() {
    let (mut timeline, clip_id) = timeline_with_clip();
    timeline.patch_clip_state(
        &clip_id,
        ClipStatePatch {
            take_gain: Some(-0.25),
            ..Default::default()
        },
    );
    let clip = timeline.clips.iter().find(|c| c.id == clip_id).unwrap();
    assert!((clip.take_gain() + 0.25).abs() < 1e-6);
    assert!((clip.audible_gain() + 0.25).abs() < 1e-6);
}

/// **跨 Clip 聚合：旧 item × 旧 take 折进新 take，新 item 保持 0 dB。**
#[test]
fn packed_takes_fold_item_gain_and_mute() {
    let (mut timeline, first) = timeline_with_clip();
    let track_id = timeline.tracks[0].id.clone();
    let second = timeline.add_clip(
        Some(track_id),
        Some("H".into()),
        Some(0.0),
        Some(1.0),
        Some("C:/audio/h.wav".into()),
    );
    for (id, take_gain, item_gain, muted) in [
        (&first, 0.5_f32, 2.0_f32, false),
        (&second, 0.5_f32, 2.0_f32, true),
    ] {
        let clip = timeline.clips.iter_mut().find(|c| c.id == *id).unwrap();
        clip.gain = take_gain;
        clip.clip_gain = item_gain;
        clip.muted = muted;
        clip.sync_take_from_flat();
        clip.normalize_takes();
    }
    let packed_id = timeline
        .pack_clips_into_takes(&[first, second])
        .expect("pack succeeds");
    let packed = timeline.clips.iter().find(|c| c.id == packed_id).unwrap();
    assert!(
        (packed.clip_gain - 1.0).abs() < 1e-6,
        "新容器 item 层为 0 dB"
    );
    assert_eq!(packed.takes.len(), 2);
    assert!(
        (packed.takes[0].gain - 1.0).abs() < 1e-6,
        "2.0 × 0.5 折成 1.0"
    );
    assert!((packed.takes[1].gain - 0.0).abs() < 1e-6, "静音源折成静音");
}
