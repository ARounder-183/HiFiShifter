//! 参数块分割 / 伸缩的回归测试：Loop 回绕、周期边界、倒放、跨 Clip、基块衬底。
use hifishifter_kernel::param_blocks::compose_param_block_view;
use hifishifter_kernel::param_blocks::editing::{
    create_block_with_id, delete_blocks, resize_block, restore_game_blocks, split_block,
};
use hifishifter_kernel::param_blocks::{
    BlockGeometry, BlockLabel, ParamBlockEditSet, ProjectedBlockKind, TakeSpan, UserParamBlock,
};
use hifishifter_kernel::state::TimelineState;

fn timeline(clips: serde_json::Value) -> TimelineState {
    let mut timeline: TimelineState = serde_json::from_value(serde_json::json!({
        "tracks":[{"id":"root","name":"Root","order":0}],
        "clips": clips,
        "project_sec":16,"bpm":120
    }))
    .unwrap();
    for clip in &mut timeline.clips {
        clip.normalize_takes();
    }
    timeline
}

fn empty() -> ParamBlockEditSet {
    ParamBlockEditSet::default()
}

fn loop_clip() -> serde_json::Value {
    serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":6.0,
        "loop_enabled":true,"loop_period_sec":2.27,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.266,
            "source_end_sec":2.27,"loop_enabled":true,"duration_sec":2.27,
            "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d",
                "sourceSpansSec":[{"startSec":0.046,"endSec":0.252}]}}]
    }])
}

#[test]
fn loop_wrap_split_and_resize_do_not_swallow_following_blocks() {
    let timeline = timeline(loop_clip());
    let base = compose_param_block_view(&timeline, "root", &empty(), 5.0).unwrap();
    assert_eq!(base.blocks.len(), 2);
    let first = &base.blocks[0];
    let split = split_block(&timeline, &empty(), &first.id, first.start_frame + 10, 5.0).unwrap();
    let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
    let users: Vec<_> = split_view
        .blocks
        .iter()
        .filter(|block| block.kind != ProjectedBlockKind::Base)
        .collect();
    assert_eq!(users.len(), 2);
    assert_eq!(users[0].start_frame, first.start_frame);
    assert_eq!(users[0].end_frame_exclusive, first.start_frame + 10);
    assert_eq!(users[1].end_frame_exclusive, first.end_frame_exclusive);

    let resized = resize_block(
        &timeline,
        &empty(),
        &first.id,
        "right",
        first.end_frame_exclusive + 2,
        5.0,
    )
    .unwrap();
    let resized_view = compose_param_block_view(&timeline, "root", &resized, 5.0).unwrap();
    let target = resized_view
        .blocks
        .iter()
        .find(|block| block.id == first.id)
        .unwrap();
    assert_eq!(target.end_frame_exclusive, first.end_frame_exclusive + 2);
    assert_eq!(resized_view.blocks.len(), 2, "后续周期基块不得被吞并");
}

#[test]
fn reverse_resize_changes_the_correct_source_edge() {
    let timeline = timeline(serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,
            "source_end_sec":2.0,"reversed":true,
            "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d",
                "sourceSpansSec":[{"startSec":0.2,"endSec":0.6}]}}]
    }]));
    let base = compose_param_block_view(&timeline, "root", &empty(), 5.0).unwrap();
    let block = &base.blocks[0];
    assert_eq!((block.start_frame, block.end_frame_exclusive), (280, 360));
    for (edge, target) in [
        ("right", 340u32),
        ("right", 380),
        ("left", 260),
        ("left", 240),
    ] {
        let next = resize_block(&timeline, &empty(), &block.id, edge, target, 5.0).unwrap();
        let next_view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        let target_block = next_view
            .blocks
            .iter()
            .find(|item| item.id == block.id)
            .unwrap();
        if edge == "right" {
            assert_eq!(target_block.end_frame_exclusive, target);
        } else {
            assert_eq!(target_block.start_frame, target);
        }
    }
}

fn cross_clip_timeline() -> TimelineState {
    timeline(serde_json::json!([
        {"id":"a","track_id":"root","name":"a","start_sec":0.0,"length_sec":4.0,
         "takes":[{"id":"a_take","source_path":"a.wav","source_start_sec":10.0,"source_end_sec":14.0}]},
        {"id":"b","track_id":"root","name":"b","start_sec":4.0,"length_sec":4.0,
         "takes":[{"id":"b_take","source_path":"b.wav","source_start_sec":0.0,"source_end_sec":4.0}]}
    ]))
}

fn cross_edits() -> ParamBlockEditSet {
    ParamBlockEditSet {
        schema_version: 1,
        revision: 1,
        blocks: vec![UserParamBlock {
            id: "cross".into(),
            geometry: BlockGeometry::Take {
                members: vec![
                    TakeSpan {
                        clip_id: "a".into(),
                        take_id: "a_take".into(),
                        cycle: 0,
                        source_start_sec: 10.0,
                        source_end_sec: 14.0,
                        syllable_id: None,
                        label: BlockLabel::Manual,
                    },
                    TakeSpan {
                        clip_id: "b".into(),
                        take_id: "b_take".into(),
                        cycle: 0,
                        source_start_sec: 0.0,
                        source_end_sec: 4.0,
                        syllable_id: None,
                        label: BlockLabel::Manual,
                    },
                ],
            },
            deleted: false,
            manual: true,
            value: Some(60.0),
            transition_out_sec: None,
            created_seq: 1,
            base_cover: None,
            grouped_from: None,
        }],
    }
}

#[test]
fn cross_clip_resize_and_split_use_projected_member_order() {
    let timeline = cross_clip_timeline();
    let edits = cross_edits();
    let base = compose_param_block_view(&timeline, "root", &edits, 5.0).unwrap();
    let block = &base.blocks[0];
    assert_eq!((block.start_frame, block.end_frame_exclusive), (0, 1600));
    for (edge, target) in [("right", 1580u32), ("left", 20)] {
        let next = resize_block(&timeline, &edits, &block.id, edge, target, 5.0).unwrap();
        let next_view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        let target_block = next_view
            .blocks
            .iter()
            .find(|item| item.id == "cross")
            .unwrap();
        if edge == "right" {
            assert_eq!(target_block.end_frame_exclusive, target);
        } else {
            assert_eq!(target_block.start_frame, target);
        }
    }
    let split = split_block(&timeline, &edits, &block.id, 400, 5.0).unwrap();
    let split_view = compose_param_block_view(&timeline, "root", &split, 5.0).unwrap();
    let users: Vec<_> = split_view
        .blocks
        .iter()
        .filter(|item| item.kind != ProjectedBlockKind::Base)
        .collect();
    assert_eq!(users.len(), 2);
    let left = users[0];
    let right = users[1];
    assert_eq!(left.members.len(), 1);
    assert_eq!(right.members.len(), 2);
    assert!(left.members.iter().all(|member| member.clip_id == "a"));
    assert!(right.members.iter().any(|member| member.clip_id == "b"));
}

#[test]
fn multi_member_shrink_stops_at_the_target_frame() {
    let timeline = timeline(serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":2.0}]
    }]));
    let edits = ParamBlockEditSet {
        schema_version: 1,
        revision: 1,
        blocks: vec![UserParamBlock {
            id: "user".into(),
            geometry: BlockGeometry::Take {
                members: vec![
                    TakeSpan {
                        clip_id: "clip".into(),
                        take_id: "take".into(),
                        cycle: 0,
                        source_start_sec: 0.0,
                        source_end_sec: 0.5,
                        syllable_id: None,
                        label: BlockLabel::Manual,
                    },
                    TakeSpan {
                        clip_id: "clip".into(),
                        take_id: "take".into(),
                        cycle: 0,
                        source_start_sec: 0.5,
                        source_end_sec: 1.0,
                        syllable_id: None,
                        label: BlockLabel::Manual,
                    },
                ],
            },
            deleted: false,
            manual: true,
            value: Some(60.0),
            transition_out_sec: None,
            created_seq: 1,
            base_cover: None,
            grouped_from: None,
        }],
    };
    for (edge, target) in [
        ("right", 50u32),
        ("right", 75),
        ("left", 150),
        ("left", 125),
    ] {
        let next = resize_block(&timeline, &edits, "user", edge, target, 5.0).unwrap();
        let next_view = compose_param_block_view(&timeline, "root", &next, 5.0).unwrap();
        let block = next_view
            .blocks
            .iter()
            .find(|item| item.id == "user")
            .unwrap();
        if edge == "right" {
            assert_eq!(block.end_frame_exclusive, target);
        } else {
            assert_eq!(block.start_frame, target);
        }
    }
}

#[test]
fn base_underlay_keeps_remainders_and_manual_delete_restores_base() {
    let timeline = timeline(serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":2.0,
            "syllable_analysis":{"algorithm":"test","algorithmVersion":1,"sourceDigest":"d",
                "sourceSpansSec":[{"startSec":0.0,"endSec":1.0}]}}]
    }]));
    let base = compose_param_block_view(&timeline, "root", &empty(), 5.0).unwrap();
    let manual = UserParamBlock {
        id: "manual".into(),
        geometry: BlockGeometry::Take {
            members: vec![TakeSpan {
                clip_id: "clip".into(),
                take_id: "take".into(),
                cycle: 0,
                source_start_sec: 0.4,
                source_end_sec: 0.6,
                syllable_id: None,
                label: BlockLabel::Manual,
            }],
        },
        deleted: false,
        manual: true,
        value: Some(60.0),
        transition_out_sec: None,
        created_seq: 1,
        base_cover: None,
        grouped_from: None,
    };
    let edits = ParamBlockEditSet {
        schema_version: 1,
        revision: 1,
        blocks: vec![manual],
    };
    let view = compose_param_block_view(&timeline, "root", &edits, 5.0).unwrap();
    assert_eq!(view.blocks.len(), 3);
    assert_eq!(view.blocks[0].start_frame, 0);
    assert_eq!(view.blocks[0].end_frame_exclusive, 80);
    assert_eq!(view.blocks[2].start_frame, 120);
    assert_eq!(view.blocks[2].end_frame_exclusive, 200);

    let (deleted, snapshots) = delete_blocks(&timeline, &edits, &view.blocks[1..2], 5.0).unwrap();
    assert!(snapshots[0].block.base_cover.is_none());
    let after = compose_param_block_view(&timeline, "root", &deleted, 5.0).unwrap();
    assert_eq!(after.blocks.len(), 1);
    assert_eq!(
        (
            after.blocks[0].start_frame,
            after.blocks[0].end_frame_exclusive
        ),
        (0, 200)
    );
    assert_eq!(base.blocks.len(), 1);
}

#[test]
fn create_block_uses_caller_id_and_midi_fallback_builds_base() {
    let timeline = timeline(serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":2.0,
            "midi_note_data":[{"startSec":0.2,"endSec":0.6,"note":60,"velocity":100,"channel":0}]}]
    }]));
    let created = create_block_with_id(
        &timeline,
        "root",
        &empty(),
        20,
        80,
        Some(61.0),
        5.0,
        Some("want"),
    )
    .unwrap();
    assert_eq!(created.blocks[0].id, "want");
    let view = compose_param_block_view(&timeline, "root", &empty(), 5.0).unwrap();
    assert_eq!(view.blocks.len(), 1);
    assert_eq!(view.blocks[0].kind, ProjectedBlockKind::Base);
}

#[test]
fn restore_game_trims_only_the_requested_range() {
    let timeline = timeline(serde_json::json!([{
        "id":"clip","track_id":"root","name":"clip","start_sec":0.0,"length_sec":2.0,
        "takes":[{"id":"take","source_path":"a.wav","source_start_sec":0.0,"source_end_sec":2.0,
            "game_block_baseline":{"revision":1,"algorithm":"game","algorithmVersion":1,
                "sourceSpansSec":[{"startSec":0.0,"endSec":1.0}]}}]
    }]));
    let edits = ParamBlockEditSet {
        schema_version: 1,
        revision: 1,
        blocks: vec![UserParamBlock {
            id: "user".into(),
            geometry: BlockGeometry::Take {
                members: vec![TakeSpan {
                    clip_id: "clip".into(),
                    take_id: "take".into(),
                    cycle: 0,
                    source_start_sec: 0.0,
                    source_end_sec: 2.0,
                    syllable_id: None,
                    label: BlockLabel::Manual,
                }],
            },
            deleted: false,
            manual: true,
            value: Some(60.0),
            transition_out_sec: None,
            created_seq: 1,
            base_cover: None,
            grouped_from: None,
        }],
    };
    let next = restore_game_blocks(&timeline, "root", &edits, 50, 100, 5.0).unwrap();
    let user = next.blocks.iter().find(|block| block.id == "user").unwrap();
    let members = user.members();
    assert_eq!(members.len(), 2);
    assert!((members[0].source_start_sec - 0.0).abs() < 1e-6);
    assert!((members[0].source_end_sec - 0.25).abs() < 1e-6);
    assert!((members[1].source_start_sec - 0.5).abs() < 1e-6);
    assert!((members[1].source_end_sec - 2.0).abs() < 1e-6);
    let restored = next
        .blocks
        .iter()
        .find(|block| block.id != "user" && !block.deleted)
        .expect("restored baseline block");
    assert_eq!(restored.members()[0].source_start_sec, 0.25);
    assert_eq!(restored.members()[0].source_end_sec, 0.5);
}
