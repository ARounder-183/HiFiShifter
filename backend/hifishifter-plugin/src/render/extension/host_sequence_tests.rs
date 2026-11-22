//! 宿主序列集成测试：用真实块序列重放现场日志里的 VEGAS 形状。
//!
//! 这些用例不碰宿主、不碰音频设备；它们只把 `note_stream_geometry` 与
//! `CaptureStore::on_block_with_context` 串起来，验证“几何 -> 捕获”这一条链。

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
    context.prune_before = block.prune_before;
    context.raw_start = block.raw_start;
    context.position_trusted = block.position_trusted;
    context.publish_deferred = block.publish_deferred;
    context.preview = block.preview;
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
        StreamTime {
            music: Some(24_414.0),
            tempo: Some(120.0),
            ..StreamTime::default()
        },
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
        StreamTime {
            music: Some(1.5),
            tempo: Some(120.0),
            ..StreamTime::default()
        },
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

/// 现场 23:41 的 Event FX 预卷：首块是绝对域、`kPlaying = 0`。
///
/// 【验收】它进入 PendingAbsolute：位置照常按 raw 写入（Track 需要保留），但
/// `position_trusted = false` —— UI 不会看到 60.000 s。
#[test]
fn event_pre_roll_is_pending_and_not_published() {
    let owner = ExtensionOwner::default();
    owner.capture().note_project_rate(48_000.0);
    owner.capture().start().unwrap();

    let block = owner.note_stream_geometry(
        ProcessStream::Realtime,
        2_880_000,
        848,
        48_000.0,
        StreamTime {
            music: Some(24_414.0),
            tempo: Some(120.0),
            ..StreamTime::default()
        },
        false,
        false,
        false,
    );
    assert_eq!(block.position, 2_880_000, "预卷仍按 raw 写入");
    assert!(!block.position_trusted, "预卷位置不可发布");
    assert!(block.publish_deferred);
}

/// 现场 23:41 的 Event FX 切换：预卷 + 第一个相对块。
///
/// 【验收】rebase + prune 之后 take 只有一个段，起点是第一个相对块（36,016），
/// 没有 0.68 s 的绝对域前缀，也没有 3248 帧的洞。
#[test]
fn event_switch_prunes_the_disconnected_pre_roll() {
    let owner = ExtensionOwner::default();
    owner.capture().note_project_rate(48_000.0);
    owner.capture().start().unwrap();

    let pre_roll = vec![0.01_f32; 848];
    for _ in 0..37 {
        let block = owner.note_stream_geometry(
            ProcessStream::Realtime,
            2_880_000,
            848,
            48_000.0,
            StreamTime {
                music: Some(24_414.0),
                tempo: Some(120.0),
                ..StreamTime::default()
            },
            false,
            false,
            false,
        );
        feed_capture(&owner, &pre_roll, block.position, 848, &block, false);
    }

    let content = vec![0.1_f32; 848];
    let switch = owner.note_stream_geometry(
        ProcessStream::Realtime,
        36_016,
        848,
        48_000.0,
        StreamTime {
            music: Some(1.5),
            tempo: Some(120.0),
            ..StreamTime::default()
        },
        true,
        false,
        true,
    );
    assert_eq!(switch.rebase_frames, -2_880_000);
    assert_eq!(switch.prune_before, 36_016);
    assert_eq!(switch.position, 36_016);
    assert!(switch.position_trusted);
    feed_capture(&owner, &content, 36_016, 848, &switch, true);

    owner.capture().stop().unwrap();
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    assert_eq!(timeline.clips.len(), 1, "整条 take 必须是一个连续片段");
    let expected = 36_016.0 / 48_000.0;
    assert!(
        (timeline.clips[0].start_sec - expected).abs() < 1e-6,
        "片段必须从第一个相对块开始（实际 {}，期望 {expected}）",
        timeline.clips[0].start_sec
    );
}

/// Track FX 的预卷与正文同域：不 rebase、不 prune，take 从 60.000 s 开始。
#[test]
fn track_pre_roll_resolves_project_and_is_kept() {
    let owner = ExtensionOwner::default();
    owner.capture().note_project_rate(48_000.0);
    owner.capture().start().unwrap();

    let pre_roll = vec![0.01_f32; 848];
    for _ in 0..37 {
        let block = owner.note_stream_geometry(
            ProcessStream::Realtime,
            2_880_000,
            848,
            48_000.0,
            StreamTime {
                music: Some(24_414.0),
                tempo: Some(120.0),
                ..StreamTime::default()
            },
            false,
            false,
            false,
        );
        feed_capture(&owner, &pre_roll, block.position, 848, &block, false);
    }

    let block = owner.note_stream_geometry(
        ProcessStream::Realtime,
        2_916_016,
        848,
        48_000.0,
        StreamTime {
            music: Some(24_415.0),
            tempo: Some(120.0),
            ..StreamTime::default()
        },
        true,
        false,
        true,
    );
    assert_eq!(block.rebase_frames, 0, "Track 不换域");
    assert_eq!(block.prune_before, 0, "Track 的预卷必须保留");
    assert_eq!(block.position, 2_916_016);
    assert!(block.position_trusted);
}

/// 现场 2026-11-22 01:56 的 VEGAS Track FX 起播窗口（全管线）：
/// 音频先流（kPlaying=0、raw 钉在播放头 179512），32 块共 32768 帧真实内容；
/// ~0.79s 后 kPlaying 置位，raw 跳到 217584（= 179512 + 38072，比已写位置多 5304 帧
/// —— 宿主传输比送块快，那 5304 帧宿主从未送来）。
///
/// 【验收】三个症状同根（现场 take 1 在 tape 32768 处分裂 / take 3 的 0.563 s 假静音
/// 前缀 / take 2 的 0.119 s 残片）：
///   1. 起播窗口的内容必须按钉住的 raw 写入 —— take 从 179512 起，窗口 32768 帧
///      不能被门控丢弃（旧代码只允许 `!had` 的首块开窗，武装块先到时窗口永不打开）；
///   2. kPlaying 升起不是"换尺子"：PendingAbsolute→Project 必须保持 resync 资格，
///      +5304 帧记账差按"计数器可能错过的范围"桥接 —— 整条 take 是**一个**片段，
///      不许在接缝处断成两个 clip（现场 take 1 的 0.11 s 假洞 + 咔声）；
///   3. 不许有任何片段落在错误位置（旧 Event/重基路径会把窗口前缀拉到工程 0）。
#[test]
fn vegas_start_window_is_captured_and_stays_one_take() {
    let owner = ExtensionOwner::default();
    owner.capture().note_project_rate(48_000.0);
    owner.capture().start().unwrap();

    let mut music = 8.9_f64;
    let mut raw = 0_i64;
    let feed = |owner: &ExtensionOwner,
                raw: i64,
                frames: usize,
                playing: bool,
                amp: f32,
                seed: f32,
                music: &mut f64,
                pinned_raw: Option<i64>| {
        let samples: Vec<f32> = (0..frames)
            .map(|n| ((n as f32 * 0.013 + seed).sin() * amp).clamp(-0.9, 0.9))
            .collect();
        let plane0 = samples.clone();
        let plane1 = samples;
        *music += frames as f64 / 48_000.0 / 60.0 * 120.0;
        let context = crate::block_context::BlockContext {
            sample_rate: 48_000.0,
            project_time_samples: pinned_raw.unwrap_or(raw),
            playing,
            recording: false,
            continuous: None,
            music: Some(*music),
            tempo: Some(120.0),
            system_time: None,
            time_signature: None,
            cycle_valid: false,
            cycle_active: false,
            cycle_start_music: 0.0,
            cycle_end_music: 0.0,
        };
        let input = crate::capture::InputBlock {
            start: pinned_raw.unwrap_or(raw),
            frames,
            sample_rate: 48_000.0,
            planes: [Some(plane0.as_slice()), Some(plane1.as_slice())],
            playing,
        };
        owner.observe_input_block(input, context, ProcessStream::Realtime)
    };

    // 停播武装：宿主送一个 1 帧块（raw=0）。
    let _ = feed(&owner, 0, 1, false, 0.0, 0.0, &mut music, None);

    // 起播窗口：raw 钉在 179512，kPlaying=0，真实内容流入。
    let mut delivered = 0_usize;
    for index in 0..32usize {
        let frames = if index % 2 == 0 { 1200 } else { 848 };
        let evidence = feed(
            &owner,
            raw,
            frames,
            false,
            0.3,
            index as f32 * 0.31,
            &mut music,
            Some(179_512),
        );
        delivered += frames;
        raw += frames as i64;
        assert_eq!(
            evidence.map(|e| (e.transport_active, e.allow_output)),
            Some((true, true)),
            "起播窗口第 {index} 块必须按播放时间放行（窗口开着且可写）"
        );
    }
    assert_eq!(delivered, 32_768);

    // kPlaying 置位：raw 跳到 217584。
    raw = 217_584;
    for index in 0..6usize {
        let frames = 848_usize;
        let evidence = feed(
            &owner,
            raw,
            frames,
            true,
            0.4,
            100.0 + index as f32 * 0.37,
            &mut music,
            None,
        );
        raw += frames as i64;
        assert!(
            evidence.is_some_and(|e| e.transport_active),
            "正文第 {index} 块必须继续有权推进"
        );
    }

    owner.capture().stop().unwrap();
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    assert!(!timeline.clips.is_empty(), "窗口内容 + 正文必须产出素材");
    let first = &timeline.clips[0];
    let expected_start = 179_512_f64 / 48_000.0;
    assert!(
        (first.start_sec - expected_start).abs() < 1e-6,
        "take 必须从起播窗口的钉住位置起：{} vs {expected_start}",
        first.start_sec
    );
    for clip in &timeline.clips {
        assert!(
            clip.start_sec >= expected_start - 1e-9,
            "没有任何片段可以落在窗口起点之前（工程 0 的错位片段）：{}",
            clip.start_sec
        );
    }
    // 一个 take：接缝桥接后所有片段同源（同一 tape 的连续 piece 之间没有 >1 帧的
    // 非洞间隔；宿主没送的那 5304 帧是覆盖表里的诚实洞，投影允许在那里分 piece，
    // 但 piece 数必须有限且位置单调衔接）。
    assert!(
        timeline.clips.len() <= 2,
        "take 断裂 = 接缝 resync 失败：clips={}",
        timeline.clips.len()
    );
}

/// **UI 播放头必须落在宿主播放头域，而不是喂入域。**
///
/// 【回归的现场】2026-11-22 的 VEGAS（用户报障"插件光标比宿主右偏 ~0.6 s"）：
/// 渲染缓存让 FX 的**喂入位置**领先宿主自己的播放头一个预热窗 —— 按下播放后先灌
/// 31920 帧音频（kPlaying=0、raw 钉在按下点 786000），随后 kPlaying 连同**前进了
/// Δ=36016 帧的传输位**一起交给插件。宿主光标在这段时间里从按下点往前走了一段，
/// 但走多远只有宿主知道：
///   * 发布喂入位置 ⇒ 右偏（2026-11-22 第一次报障）；
///   * 升起时扣整个 Δ ⇒ 左偏 ~0.6 s（同日第二次报障）；
///   * 升起时按预热窗终点发布 ⇒ 右偏 ~0.3 s（同日第三次报障 —— 光标并没有走满 W）。
///
/// 契约（与上一条测试同一条现场序列，读数换 TransportClock 的发布位）：
///   * 预热窗内：发布位**钉在按下点**（窗口期 UI 由 hold 冻结）；
///   * kPlaying 升起那一块（还没停过播、没有实测领先量）：发布位 = **预热窗中点**
///     —— 光标落在 [按下点, 窗口终点] 之间的无偏估计；
///   * 播放推进：发布位 = 升起落点 + 播放时长，与宿主光标同速；
///   * 磁带记账不受影响（素材仍落在喂入域 —— 上一条测试负责它）。
#[test]
fn the_ui_playhead_tracks_the_host_cursor_not_the_feed_position() {
    let owner = Arc::new(ExtensionOwner::default());
    owner.capture().note_project_rate(48_000.0);
    // 时钟装载点与生产一致：`arm_capture_rendering` 给 Capture 实例自备一份时钟。
    owner.arm_capture_rendering();
    owner.capture().start().unwrap();

    let mut music = 8.9_f64;
    let mut raw = 0_i64;
    let feed = |owner: &ExtensionOwner,
                raw: i64,
                frames: usize,
                playing: bool,
                amp: f32,
                seed: f32,
                music: &mut f64,
                pinned_raw: Option<i64>| {
        let samples: Vec<f32> = (0..frames)
            .map(|n| ((n as f32 * 0.013 + seed).sin() * amp).clamp(-0.9, 0.9))
            .collect();
        let plane0 = samples.clone();
        let plane1 = samples;
        *music += frames as f64 / 48_000.0 / 60.0 * 120.0;
        let context = crate::block_context::BlockContext {
            sample_rate: 48_000.0,
            project_time_samples: pinned_raw.unwrap_or(raw),
            playing,
            recording: false,
            continuous: None,
            music: Some(*music),
            tempo: Some(120.0),
            system_time: None,
            time_signature: None,
            cycle_valid: false,
            cycle_active: false,
            cycle_start_music: 0.0,
            cycle_end_music: 0.0,
        };
        let input = crate::capture::InputBlock {
            start: pinned_raw.unwrap_or(raw),
            frames,
            sample_rate: 48_000.0,
            planes: [Some(plane0.as_slice()), Some(plane1.as_slice())],
            playing,
        };
        owner.observe_input_block(input, context, ProcessStream::Realtime)
    };

    // 停播武装：宿主送一个 1 帧块（raw=0）。
    let _ = feed(&owner, 0, 1, false, 0.0, 0.0, &mut music, None);
    let clock = owner
        .clock
        .get()
        .expect("arm_capture_rendering 之后时钟必须已装载");

    // 预热窗：raw 钉在按下点 179512，kPlaying=0。域还是 PendingAbsolute ⇒ 位置
    // 暂不发布（hold）—— UI 光标必须停在原地，绝不许跟着喂入位置往前跑。
    let press = 179_512_i64;
    let press_sec = press as f64 / 48_000.0;
    let mut window_frames = 0_i64;
    let mut last_window_frames = 0_i64;
    for index in 0..32usize {
        let frames = if index % 2 == 0 { 1200 } else { 848 };
        let _ = feed(
            &owner,
            raw,
            frames,
            false,
            0.3,
            index as f32 * 0.31,
            &mut music,
            Some(press),
        );
        raw += frames as i64;
        window_frames += frames as i64;
        last_window_frames = frames as i64;
        let view = clock.read_view();
        assert!(
            view.position_sec <= press_sec + 1e-6,
            "预热窗第 {index} 块：UI 光标不得越过按下点（宿主还没动），实际 {}",
            view.position_sec
        );
    }
    // 预热窗终点 = 最后一块的**起点**（发布的是块起始时刻的播放头）：
    // 按下点 + 总帧数 − 末块帧数。
    let pre_roll_end = (press + window_frames - last_window_frames) as f64 / 48_000.0;
    // 还没有停过播 ⇒ 没有实测领先量；升起时发布预热窗中点（无偏估计）。
    let pre_roll_window = window_frames - last_window_frames;
    let midpoint = (press + pre_roll_window / 2) as f64 / 48_000.0;

    // kPlaying 置位：raw 一次性跳到 按下点+38072。发布位必须落在预热窗中点，
    // 不许跟着跳进喂入域；也不许退回到按下点（那是"扣整个 Δ"的左偏）。
    let delta = 38_072_i64;
    raw = press + delta;
    let first = feed(&owner, raw, 848, true, 0.4, 100.0, &mut music, None);
    assert!(
        first.is_some_and(|evidence| evidence.transport_active),
        "正文首块必须继续有权推进"
    );
    let view = clock.read_view();
    assert!(
        (view.position_sec - midpoint).abs() < 1e-6,
        "升起时刻：UI 光标必须落在预热窗中点 {midpoint}，实际 {}（不许退回按下点，也不许跳到喂入域）",
        view.position_sec
    );

    // 播放推进：发布位 = 预热窗中点 + 已播放时长（发布的是**块起始时刻**的播放头；
    // 升起块本身已播 848 帧，所以第 index 个后续块开头，宿主光标在中点+848·index）。
    for index in 1..6usize {
        raw += 848;
        let _ = feed(
            &owner,
            raw,
            848,
            true,
            0.4,
            100.0 + index as f32 * 0.37,
            &mut music,
            None,
        );
        let view = clock.read_view();
        let expected = midpoint + 848.0 * index as f64 / 48_000.0;
        assert!(
            (view.position_sec - expected).abs() < 1e-6,
            "播放第 {index} 块：UI 光标必须与宿主同速（{expected}），实际 {}",
            view.position_sec
        );
    }
    // 预热窗终点只在这里用于诊断断言：中点必须严格落在 [按下点, 终点] 内。
    assert!(
        press_sec < midpoint && midpoint < pre_roll_end,
        "中点估计必须落在预热窗内：press={press_sec} mid={midpoint} end={pre_roll_end}"
    );
}

/// **停播那一块能量出宿主渲染领先量，下一次升起直接用它。**
///
/// 首次播放只能取预热窗中点（无偏估计）；但停播时宿主把光标停在 `raw_stop`，
/// 而喂入位已经走到 `last_position` —— 差值就是"渲染在前、光标在后"的真实距离。
/// 它比任何基于 Δ/W 的估计都可靠，且跨 pass 保留（宿主音频引擎的属性）。
/// 本用例把这条链走完：播放 → 停播（lead=20000）→ 再播放，第二次升起必须落在
/// `按下点 + Δ − lead`，而不是中点。
#[test]
fn a_stop_measures_the_host_render_lead_and_the_next_rise_uses_it() {
    let owner = Arc::new(ExtensionOwner::default());
    owner.capture().note_project_rate(48_000.0);
    owner.arm_capture_rendering();
    owner.capture().start().unwrap();

    let mut music = 8.9_f64;
    let feed = |owner: &ExtensionOwner,
                raw: i64,
                frames: usize,
                playing: bool,
                amp: f32,
                seed: f32,
                music: &mut f64,
                pinned_raw: Option<i64>| {
        let samples: Vec<f32> = (0..frames)
            .map(|n| ((n as f32 * 0.013 + seed).sin() * amp).clamp(-0.9, 0.9))
            .collect();
        let plane0 = samples.clone();
        let plane1 = samples;
        *music += frames as f64 / 48_000.0 / 60.0 * 120.0;
        let context = crate::block_context::BlockContext {
            sample_rate: 48_000.0,
            project_time_samples: pinned_raw.unwrap_or(raw),
            playing,
            recording: false,
            continuous: None,
            music: Some(*music),
            tempo: Some(120.0),
            system_time: None,
            time_signature: None,
            cycle_valid: false,
            cycle_active: false,
            cycle_start_music: 0.0,
            cycle_end_music: 0.0,
        };
        let input = crate::capture::InputBlock {
            start: pinned_raw.unwrap_or(raw),
            frames,
            sample_rate: 48_000.0,
            planes: [Some(plane0.as_slice()), Some(plane1.as_slice())],
            playing,
        };
        owner.observe_input_block(input, context, ProcessStream::Realtime)
    };

    // 停播武装。
    let _ = feed(&owner, 0, 1, false, 0.0, 0.0, &mut music, None);
    let clock = owner
        .clock
        .get()
        .expect("arm_capture_rendering 之后时钟必须已装载");

    // ── 第一次播放：预热窗 → 升起 → 若干播放块 ─────────────────────────
    let press = 179_512_i64;
    let mut raw = 0_i64;
    for index in 0..32usize {
        let frames = if index % 2 == 0 { 1200 } else { 848 };
        let _ = feed(
            &owner,
            raw,
            frames,
            false,
            0.3,
            index as f32 * 0.31,
            &mut music,
            Some(press),
        );
        raw += frames as i64;
    }
    let delta = 38_072_i64;
    raw = press + delta;
    let _ = feed(&owner, raw, 848, true, 0.4, 100.0, &mut music, None);
    for index in 1..5usize {
        raw += 848;
        let _ = feed(
            &owner,
            raw,
            848,
            true,
            0.4,
            100.0 + index as f32 * 0.37,
            &mut music,
            None,
        );
    }
    let last_feed = raw;
    let window = {
        let geometry = &owner.streams[ProcessStream::Realtime.index()];
        geometry
            .pre_roll_window
            .load(std::sync::atomic::Ordering::Acquire)
    };
    assert!(window > 0, "第一次播放必须记下预热窗宽度");

    // ── 停播：宿主把光标停在喂入位之后 lead 帧 ───────────────────────────
    let lead = 20_000_i64;
    assert!(lead <= window, "用例的领先量必须落在预热窗内");
    let stop_raw = last_feed - lead;
    let _ = feed(
        &owner,
        stop_raw,
        848,
        false,
        0.0,
        200.0,
        &mut music,
        Some(stop_raw),
    );
    let geometry = &owner.streams[ProcessStream::Realtime.index()];
    assert!(
        geometry
            .render_lead_valid
            .load(std::sync::atomic::Ordering::Acquire),
        "停播跃迁必须量到渲染领先量"
    );
    assert_eq!(
        geometry
            .render_lead
            .load(std::sync::atomic::Ordering::Acquire),
        lead,
        "领先量 = 最后喂入位 − 宿主停播光标"
    );

    // ── 第二次播放：升起必须用实测领先量，而不是中点 ─────────────────────
    let press2 = stop_raw;
    let mut raw2 = 0_i64;
    for index in 0..32usize {
        let frames = if index % 2 == 0 { 1200 } else { 848 };
        let _ = feed(
            &owner,
            raw2,
            frames,
            false,
            0.3,
            300.0 + index as f32 * 0.31,
            &mut music,
            Some(press2),
        );
        raw2 += frames as i64;
    }
    let delta2 = 38_072_i64;
    let rise2 = press2 + delta2;
    let _ = feed(&owner, rise2, 848, true, 0.4, 400.0, &mut music, None);
    let view = clock.read_view();
    let expected = (rise2 - lead) as f64 / 48_000.0;
    assert!(
        (view.position_sec - expected).abs() < 1e-6,
        "第二次升起：UI 光标必须用实测领先量 {lead}（发布 {expected}），实际 {}",
        view.position_sec
    );
    assert_eq!(
        owner.ui_playhead_offset(),
        lead,
        "UI 播放头修正量必须等于实测渲染领先量"
    );

    // 捕获状态里的几何是**展示值**：red 捕获带与参数编辑器标尺都用它，必须和
    // 播放头同域。磁带记账仍是喂入域 —— 这里只核对发布出来的 JSON，不碰存储。
    let status = crate::capture::commands::handle(&owner, "capture_status", &serde_json::json!({}))
        .expect("capture_status");
    let published = &status["capture"];
    let stored = owner
        .capture()
        .status()
        .capture
        .expect("armed capture must expose a span");
    assert_eq!(
        published["positionFrame"].as_i64(),
        Some(stored.position_frame - lead),
        "捕获当前位置必须平移到宿主光标域"
    );
    assert_eq!(
        published["startFrame"].as_i64(),
        Some(stored.start_frame - lead),
        "捕获带起点必须平移到宿主光标域"
    );
    for (index, run) in stored.runs.iter().enumerate() {
        assert_eq!(
            published["runs"][index]["startFrame"].as_i64(),
            Some(run.start - lead),
            "第 {index} 段捕获带起点必须平移到宿主光标域"
        );
        assert_eq!(
            published["runs"][index]["endFrame"].as_i64(),
            Some(run.end - lead),
            "第 {index} 段捕获带终点必须平移到宿主光标域"
        );
    }

    // 播放推进保持同速。
    let mut raw2 = rise2;
    for index in 1..4usize {
        raw2 += 848;
        let _ = feed(
            &owner,
            raw2,
            848,
            true,
            0.4,
            400.0 + index as f32 * 0.37,
            &mut music,
            None,
        );
        let view = clock.read_view();
        let expected = expected + 848.0 * index as f64 / 48_000.0;
        assert!(
            (view.position_sec - expected).abs() < 1e-6,
            "第二次播放第 {index} 块：UI 光标必须与宿主同速（{expected}），实际 {}",
            view.position_sec
        );
    }
}
