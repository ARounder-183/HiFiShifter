//! 宿主序列集成测试：用真实块序列重放现场日志里的 VEGAS 形状。
//!
//! 这些用例不碰宿主、不碰音频设备；它们只把 §note_stream_geometry§ 与
//! §CaptureStore::on_block_with_context§ 串起来，验证“几何 -> 捕获”这一条链。

use super::*;
use crate::capture::recorder::BlockContext;
use crate::capture::InputBlock;

fn feed_capture(
    owner: &ExtensionOwner,
    samples: &[f32],
    start: i64,
    frames: usize,
    block: &StreamBlock,
    playing: bool,
) {
    let input = InputBlock {
        start,
        frames,
        sample_rate: 48_000.0,
        planes: [Some(samples), Some(samples)],
        playing,
    };
    let mut context = BlockContext::realtime(true);
    context.capture_epoch = owner.capture().capture_epoch();
    context.playing = playing;
    context.phase = block.phase;
    context.playing_rising = playing;
    context.new_pass = block.new_pass;
    context.transport_active = block.transport_active;
    context.start_window = block.start_window;
    context.epoch = block.epoch;
    context.new_epoch = block.new_epoch;
    context.rebase_frames = block.rebase_frames;
    context.advancing = true;
    let _ = owner.capture().on_block_with_context(input, context);
}

/// 现场 21:18 的 Event FX 序列：绝对域首块 -> kPlaying 上升沿 + 相对域。
///
/// 【验收】一个连续 take：第一段 0..32768、第二段 32768..380928，没有 0.68 s
/// 的绝对域首段，也没有 5296 帧假洞。
#[test]
fn vegas_event_fx_sequence_maps_event_start_to_local_zero() {
    let owner = ExtensionOwner::default();
    owner.capture().note_project_rate(48_000.0);
    owner.capture().start().unwrap();

    let first_frames = 32_768_usize;
    let first_samples = vec![0.1_f32; first_frames];
    let first = owner.note_stream_geometry(
        ProcessStream::Realtime,
        906_000,
        first_frames,
        48_000.0,
        StreamTime::default(),
        true,
        false,
        false,
    );
    assert_eq!(first.position, 906_000, "pre-roll 仍在工程绝对域");
    assert!(first.transport_active);
    feed_capture(&owner, &first_samples, 906_000, first_frames, &first, false);

    let second_frames = 348_160_usize;
    let second_samples = vec![0.1_f32; second_frames];
    let second = owner.note_stream_geometry(
        ProcessStream::Realtime,
        32_768,
        second_frames,
        48_000.0,
        StreamTime::default(),
        true,
        false,
        true,
    );
    assert_eq!(second.position, 32_768, "事件起点映射到本地 0 之后");
    assert!(!second.new_pass, "域切换不是新的传输会话");
    assert_eq!(second.rebase_frames, -906_000);
    feed_capture(
        &owner,
        &second_samples,
        32_768,
        second_frames,
        &second,
        true,
    );

    owner.capture().stop().unwrap();
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    assert_eq!(timeline.clips.len(), 1, "整条 take 必须是一个连续片段");
    let clip = &timeline.clips[0];
    assert!(
        clip.start_sec.abs() < 1e-9,
        "片段必须从本地 0 开始（实际 {}）",
        clip.start_sec
    );
    let expected = (first_frames + second_frames) as f64 / 48_000.0;
    assert!(
        (clip.length_sec - expected).abs() < 1e-9,
        "片段长度必须是整条事件：{} vs {}",
        clip.length_sec,
        expected
    );
}
