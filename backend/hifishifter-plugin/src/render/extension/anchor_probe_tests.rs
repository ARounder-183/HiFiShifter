//! 内容锚定的集成用例：从「宿主报的位置」到「实测的位置」。
//!
//! 这些用例不碰宿主、不碰音频设备。它们把一张真实的映射表 + 一份真实的磁带探针
//! 塞进 CaptureStore，然后问同一个问题：**这一块被当成工程哪一段**。
//!
//! 【为什么值得单独一个文件】这是本轮唯一改变声音位置的决定，而它在产物上的三种错误
//! 形态（整段平移、按旧光标平移、根本没锚）完全同形 —— 都是「某段音频出现在错误的
//! 时间上」。把它钉在这里，比在日志里读 anchor=playhead 可靠得多。

use super::*;
use hifishifter_capture::anchor_probe::{ProbeBlock, TapeSamples};
use hifishifter_capture::project_map::{AnchorPolicy, AnchorSource, ProjectMapSegment};

const RATE: i64 = 48_000;
/// 片段起点：15.000 s。
const CLIP_START: i64 = 15 * RATE;
/// 现场那个「上一次的光标」：39.396 s。
const CURSOR: i64 = 1_891_012;
const BLOCK: usize = 960;

/// 一份可控的干声素材（确定性的非周期噪声）。
struct TestTape {
    left: Vec<f32>,
    covered: Vec<std::ops::Range<i64>>,
}

impl TapeSamples for TestTape {
    fn origin(&self) -> i64 {
        CLIP_START
    }
    fn left(&self) -> &[f32] {
        &self.left
    }
    fn right(&self) -> &[f32] {
        &[]
    }
    fn covered(&self) -> &[std::ops::Range<i64>] {
        &self.covered
    }
}

fn noise(frames: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..frames)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// 一段 4 秒素材 + 一张「整段对应片段 [15 s, 19 s)」的映射表。
fn fixture() -> (Arc<ExtensionOwner>, Vec<f32>) {
    let owner = Arc::new(ExtensionOwner::default());
    let tape = TestTape {
        left: noise(4 * RATE as usize, 23),
        covered: Vec::new(),
    };
    let samples: Vec<f32> = tape.left.clone();
    owner.capture().publish_project_map_for_test(
        3,
        &[ProjectMapSegment {
            tape_start: 0,
            tape_end: 4 * RATE,
            project_start: CLIP_START,
            take: 1,
        }],
    );
    owner.capture().publish_tape_probe(Arc::new(tape));
    (owner, samples)
}

fn project(owner: &ExtensionOwner, tape_frame: i64, hint: i64, block: &[f32]) -> i64 {
    let block = ProbeBlock {
        left: block,
        right: &[],
        frames_before: 0,
    };
    owner.project_position_for_stream_probed(
        ProcessStream::Realtime,
        tape_frame,
        hint,
        hint,
        Some(block),
    )
}

fn anchor_source(owner: &ExtensionOwner) -> Option<AnchorSource> {
    AnchorSource::from_code(
        owner.streams[ProcessStream::Realtime.index()]
            .anchor_source
            .load(Ordering::Acquire),
    )
}

/// **本轮的主用例：Render As 的锚点从「光标」挪到「内容」。**
///
/// 【现场原形】渲染范围 [15.000, 46.681]，光标停在 39.396 s。宿主首块报
/// projectTimeSamples = 1 891 012，而喂进来的内容是**片段起点**。位置规则给出
/// project_zero = 1 891 012（整条渲染右移 24.396 s），内容判定给出 720 000。
#[test]
fn a_render_anchored_on_the_cursor_is_moved_to_the_content() {
    let (owner, samples) = fixture();
    let block = samples[0..BLOCK].to_vec();
    // 喂入域第 0 帧就是片段起点；宿主读数（光标）落在片段窗口里，位置规则会信它。
    assert_eq!(project(&owner, 0, CURSOR, &block), CLIP_START);
    assert_eq!(
        anchor_source(&owner),
        Some(AnchorSource::Content),
        "锚点必须由内容判定给出，而不是位置规则"
    );
}

/// **不变量：从播放头起播的预览逐帧不变。**
///
/// 【为什么这条必须留】§1.4 的预览现场（raw=720000 tape=23728 project=743728）
/// 无法区分「按播放头锚」与「按片段起点锚」—— 三者同值。新判据必须让这一条保持原样。
#[test]
fn a_preview_that_starts_mid_item_is_unchanged() {
    let (owner, samples) = fixture();
    let playhead = CLIP_START + 2 * RATE;
    let start = (playhead - CLIP_START) as usize;
    let block = samples[start..start + BLOCK].to_vec();
    assert_eq!(
        project(&owner, 0, playhead, &block),
        playhead,
        "播放头在片段内时，内容判定给出的就是播放头"
    );
    assert_eq!(anchor_source(&owner), Some(AnchorSource::Content));
}

/// **不变量：一条喂入流的锚点只赋值一次。**
///
/// 【为什么】中途改锚点等于在产物中间切一刀 —— 前一半按 A 播、后一半按 B 播，而用户
/// 听到的是「某一段突然重复/跳了一段」。
#[test]
fn the_anchor_is_assigned_exactly_once_per_feed() {
    let (owner, samples) = fixture();
    let first = samples[0..BLOCK].to_vec();
    assert_eq!(project(&owner, 0, CURSOR, &first), CLIP_START);
    // 150 块之后，锚点与它记下的首块读数都必须原样不动（150 × 960 = 144 000 帧，
    // 仍在 4 秒素材之内 —— 走出素材之后「没有映射」是**正确**答案，那会把这条
    // 用例变成一条关于素材长度的用例）。
    let mut tape_frame = 0_i64;
    for index in 0..150 {
        tape_frame += BLOCK as i64;
        let offset = (index as usize * BLOCK) % (samples.len() - BLOCK);
        let block = samples[offset..offset + BLOCK].to_vec();
        // 后续块一律报同一个宿主读数（钉住流的真实形状）。
        let position = project(&owner, tape_frame, CURSOR, &block);
        assert_eq!(
            position,
            CLIP_START + tape_frame,
            "锚点一旦定稿，每一块都必须落在同一条时间轴上"
        );
    }
}

/// **不变量：定稿之后内容变了也不重锚。**
///
/// 【为什么】渲染中宿主喂的内容可能与素材不完全一致（上游 FX、速率变化）。那时正确的
/// 行为是**继续按已定的锚点播完**，而不是在中途换一个零点 —— 后者会把整段产物切成
/// 两截，而两截各自「看起来合理」。
#[test]
fn a_decided_anchor_never_changes_mid_feed() {
    let (owner, samples) = fixture();
    let first = samples[0..BLOCK].to_vec();
    assert_eq!(project(&owner, 0, CURSOR, &first), CLIP_START);
    // 换一份完全不相干的内容（另一段噪声），宿主读数不变 —— 同一条喂入流。
    // 【为什么 hint 必须不变】缓存键是 `(映射代次, first_raw)`：hint 一变就是
    // **另一条**喂入流，那时重锚是正确行为。这条用例问的是「同一条流里内容变了
    // 会怎样」。
    let other = noise(BLOCK, 99);
    let position = project(&owner, BLOCK as i64, CURSOR, &other);
    assert_eq!(
        position,
        CLIP_START + BLOCK as i64,
        "定稿之后只按原锚点推进，内容变化不得改零点"
    );
    assert_eq!(anchor_source(&owner), Some(AnchorSource::Content));
}

/// **开关仍然可观测**：强制 clip 时内容判定不参与。
#[test]
fn a_forced_policy_skips_the_content_probe() {
    let (owner, samples) = fixture();
    let block = samples[0..BLOCK].to_vec();
    let probe = ProbeBlock {
        left: &block,
        right: &[],
        frames_before: 0,
    };
    let outcome = owner
        .capture()
        .feed_anchor_probed(CURSOR, AnchorPolicy::Clip, Some(probe));
    assert!(outcome.probe.is_none(), "强制策略下不许跑探测");
    assert_eq!(
        outcome.anchor.map(|anchor| anchor.source),
        Some(AnchorSource::Clip)
    );
}
