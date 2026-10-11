//! `assigned_timeline` 的保留规则回归。

use super::*;

/// 空 folder 轨 / 只有静音 item 的子轨仍须作为参数根存在。
///
/// 【回归】旧实现先按 region 筛 clip，再丢掉**所有**没有 clip 的轨道 —— 于是一个
/// 被宿主分配了 sequence、当前却没有 clip 的轨道会整条消失，用户看到"轨道没了"。
/// 这与 `add_group_view_tracks` 的既有口径（空父轨也能承载参数组算法）直接矛盾。
#[test]
fn a_sequence_without_clips_still_yields_a_parameter_root() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    {
        let mut timeline = document.timeline.lock_or_recover();
        let timeline = timeline.as_mut().unwrap();
        timeline.tracks.push(
            serde_json::from_value(
                serde_json::json!({"id":"empty","name":"empty folder","order":2}),
            )
            .unwrap(),
        );
        assert_eq!(timeline.tracks.len(), 3);
    }
    let sequence_key = 4242_u64;
    document
        .sequence_regions
        .lock_or_recover()
        .insert(sequence_key, std::collections::HashSet::new());
    document
        .sequence_track_index
        .lock_or_recover()
        .insert(sequence_key, 2);
    owner
        .sequences
        .lock_or_recover()
        .insert(owner.role.load(Ordering::Acquire), vec![sequence_key]);

    let timeline = owner.assigned_timeline(&document).unwrap();
    assert!(
        timeline.tracks.iter().any(|track| track.id == "empty"),
        "被分配 sequence 的空轨道必须留下：{:?}",
        timeline.tracks.iter().map(|t| &t.id).collect::<Vec<_>>()
    );
    assert_eq!(timeline.clips.len(), 1, "空轨道不带来任何 clip");

    // 反向：**没有**被分配 sequence 的空轨道仍然要被剔除，否则等于把整张文档
    // 交给了空 renderer（旧的"零分配只能看零轨道"约束不能被这次改动放宽）。
    document
        .sequence_track_index
        .lock_or_recover()
        .remove(&sequence_key);
    let timeline = owner.assigned_timeline(&document).unwrap();
    assert!(
        !timeline.tracks.iter().any(|track| track.id == "empty"),
        "未分配的空轨道不得留下"
    );
    document.close();
}
