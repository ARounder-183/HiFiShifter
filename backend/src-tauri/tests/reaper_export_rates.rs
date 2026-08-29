//! REAPER export rate / multi-take collection regressions (integration
//! target). Runs with plain `cargo test` (helpers from
//! `backend_lib::__test_internals`):
//!
//! ```text
//! cargo test --test reaper_export_rates
//! ```
//!
//! Behavioral contract:
//! - RPP has no item-level PLAYRATE; the take PLAYRATE is the effective
//!   rate — the default (active) take must carry the **combined rate**
//!   (clip × take), otherwise rate-changing items lose speed in the
//!   RPP → HiFiShifter → RPP round-trip;
//! - Explicit TAKE blocks list every take except the active one (the item
//!   body carries the active take) and omit SEL — the importer falls back
//!   to the default take when SEL is absent, restoring the active choice.

use backend_lib::__test_internals::{build_reaper_clipboard, parse_clipboard_bytes, TimelineState};

#[test]
fn multi_take_export_preserves_combined_rates_and_nonzero_active_index() {
    let mut timeline = TimelineState::default();
    let track_id = timeline.tracks[0].id.clone();
    let clip_id = timeline.add_clip(
        Some(track_id),
        Some("Rates".to_string()),
        Some(0.0),
        Some(2.0),
        Some("C:/audio/a.wav".to_string()),
    );
    {
        let clip = timeline
            .clips
            .iter_mut()
            .find(|clip| clip.id == clip_id)
            .unwrap();
        // takes = [Rates(1.0), Second(1.0)]；随后把 Clip 级倍率设为 2.0，
        // 并给第二个 take 自身速率 1.5（组合有效速率 3.0），再把 active
        // 切到第二个 take —— 覆盖“active 非首位 + 非 1 速率”的导出路径。
        let mut second = clip.active_take().clone();
        second.id = "test_take_second".to_string();
        second.name = "Second".to_string();
        second.source_path = Some("C:/audio/b.wav".to_string());
        clip.add_take(second);
        clip.clip_playback_rate = 2.0;
        clip.playback_rate = 2.0;
        clip.sync_take_from_flat();
        let second_id = clip.takes[1].id.clone();
        clip.switch_active_take(&second_id)
            .expect("second take exists");
        clip.takes[1].playback_rate = 1.5;
    }

    let export = build_reaper_clipboard(&timeline, &[clip_id]).expect("clipboard export");
    assert_eq!(export.exported_clip_count, 1);
    let parsed = parse_clipboard_bytes(&export.bytes).expect("parse exported clipboard");
    let item = &parsed.tracks[0].items[0];
    // active take（Second）作为 default take：PLAYRATE = clip 级 × take 自身。
    assert_eq!(item.default_take.name, "Second");
    assert!((item.default_take.play_rate[0] - 3.0).abs() < 1e-9);
    // 显式 TAKE 块只含非 active 的第一个 take，且不打 SEL 标记。
    assert_eq!(item.takes.len(), 1);
    assert_eq!(item.takes[0].name, "Rates");
    assert!((item.takes[0].play_rate[0] - 2.0).abs() < 1e-9);
    assert!(!item.takes[0].selected);

    // ── 往返：重新导入后 take 集合、active 选择与组合速率保持一致 ──
    let active_default_rate = item.default_take.play_rate[0];
    let explicit_names: Vec<&str> = item.takes.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(explicit_names, vec!["Rates"]);
    assert!(
        (active_default_rate - 3.0).abs() < 1e-9,
        "default take must carry combined effective rate"
    );
}
