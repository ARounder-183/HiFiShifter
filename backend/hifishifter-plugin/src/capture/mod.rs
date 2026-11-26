//! VST3 外壳对捕获运行时的适配层。
//!
//! 【为什么捕获运行时不在这个 crate 里】起播滞后、停播钉住、循环回绕、段封口、
//! sidecar 恢复、时间线投影 —— 这些判定的输入只有"工程位置 + 块 + 是否在播放"，
//! 与宿主是 VST3 还是 CLAP 无关。运行时因此住在 `hifishifter-capture`，两个格式
//! 外壳各写一份**只做转换**的适配层：VST3 把 `ProcessContext` 变成
//! [`recorder::InputBlock`]，CLAP 把 `clap_event_transport_t` 变成同一个形状。
//!
//! 【这份适配层只做三件事】
//! 1. 转发：让 `crate::capture::recorder::…` / `crate::capture::CaptureStore` 这些
//!    既有路径继续成立（渲染侧与编辑器侧有上百处引用）；
//! 2. 把中性形状换成渲染侧的类型：[`CaptureTape`] → `PlaybackSnapshot`、
//!    [`CaptureRegion`] → `AraPlaybackRegion`；
//! 3. 命令分发（`commands.rs`）：它要碰 `ExtensionOwner`（处理代次、合成请求），
//!    那部分确实是 VST3 专属。

pub(crate) mod commands;

pub(crate) use hifishifter_capture::recorder;
pub(crate) use hifishifter_capture::store::*;
pub(crate) use hifishifter_capture::timeline;
pub(crate) use hifishifter_capture::{CaptureRegion, CaptureTape};

use crate::ara::AraPlaybackRegion;
use crate::render::snapshot::PlaybackSnapshot;

/// [`CaptureTape`] → 渲染侧的不可变播放快照。
///
/// 【为什么不能直接用同一个类型】快照的发布器（读者计数、退役缓冲、预算退还）属于
/// 渲染管线；捕获只交出"要发布的样本与覆盖表"。转换是纯字段搬运，预算凭证跟着数据走。
pub(crate) fn snapshot_from_tape(tape: CaptureTape) -> PlaybackSnapshot {
    PlaybackSnapshot {
        sample_rate: tape.sample_rate,
        origin_sample: tape.origin_sample,
        left: tape.left,
        right: tape.right,
        covered: tape.covered,
        _reservation: tape.reservation,
    }
}

/// 让内容探测读到**已发布的那一份**样本（见 `hifishifter_capture::anchor_probe`）。
///
/// 【为什么实现放在这里】`PlaybackSnapshot` 是渲染管线的类型，而 `TapeSamples` 是
/// 捕获运行时的抽象。这份适配层正是两者之间唯一该有的一道桥 —— 与上面
/// `snapshot_from_tape` 是同一条纪律：捕获只交出数据，格式外壳只做转换。
///
/// 【为什么四个字段就够了】包络要的是「起点 + 两个平面 + 覆盖表」；读者计数、预算
/// 凭证、内容指纹都属于发布器，与「这一块在哪」无关。
impl hifishifter_capture::anchor_probe::TapeSamples for PlaybackSnapshot {
    fn origin(&self) -> i64 {
        self.origin_sample
    }

    fn left(&self) -> &[f32] {
        &self.left
    }

    fn right(&self) -> &[f32] {
        &self.right
    }

    fn covered(&self) -> &[std::ops::Range<i64>] {
        &self.covered
    }
}
/// 捕获的 1:1 区间 → ARA 形状的播放区间。
///
/// 【为什么可以逐字段搬】字段语义完全相同（秒、源内起点、播放起点、无拉伸/无淡化），
/// 渲染入口的几何校验因此不需要为捕获分叉。
pub(crate) fn ara_regions(regions: Vec<CaptureRegion>) -> Vec<AraPlaybackRegion> {
    regions
        .into_iter()
        .map(|region| AraPlaybackRegion {
            name: region.name,
            audio_source_persistent_id: region.audio_source_persistent_id,
            audio_modification_persistent_id: region.audio_modification_persistent_id,
            region_sequence_index: region.region_sequence_index,
            sequence_model_ref: region.sequence_model_ref,
            start_in_modification_time: region.start_in_modification_time,
            duration_in_modification_time: region.duration_in_modification_time,
            start_in_playback_time: region.start_in_playback_time,
            duration_in_playback_time: region.duration_in_playback_time,
            is_timestretch_enabled: region.is_timestretch_enabled,
            has_content_based_fade_at_head: region.has_content_based_fade_at_head,
            has_content_based_fade_at_tail: region.has_content_based_fade_at_tail,
        })
        .collect()
}
