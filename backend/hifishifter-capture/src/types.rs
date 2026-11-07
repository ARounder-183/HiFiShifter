//! 捕获运行时交给格式外壳的中性数据形状。
//!
//! 【为什么这些类型在这里而不是各格式里】它们描述的是"我们录到了什么"，与宿主 ABI
//! 无关。VST3 的 `SourcePcm` 是本文件 [`CapturePcm`] 的**类型别名**（同一份数据，
//! 渲染侧一行都不用改）；它的 `PlaybackSnapshot` 由 [`CaptureTape`] 一对一转换，
//! 只多一个内存预算凭证。

use std::ops::Range;

/// 一段素材的源 PCM（每个 take 一条"素材带"，段按录制顺序首尾相接）。
#[derive(Debug)]
pub struct CapturePcm {
    /// 素材自身的采样率。
    pub sample_rate: u32,
    /// 声道平面（1 或 2 条）。
    pub planes: Vec<Vec<f32>>,
    /// 版本号（= take id）：下游据此判断"源变了没有"。
    pub version: u64,
    /// 哪些帧**真的录到了**（源帧域，升序、不重叠）。空 = 整份都算。
    pub covered: Vec<Range<usize>>,
    /// 内存预算凭证：随数据一起释放，绝不单独记账。
    pub _reservation: Option<crate::budget::Reservation>,
}

/// 原始磁带：把捕获到的样本按工程帧铺平，供渲染发布器直接发布。
///
/// 【为什么不是渲染快照本身】发布器在 VST3 crate 里（它还要管退役缓冲与读者计数）。
/// 这里只交出"要发布的样本与覆盖表"，格式外壳把它包成自己的快照类型。
#[derive(Debug)]
pub struct CaptureTape {
    /// 输出采样率。
    pub sample_rate: u32,
    /// 第一个样本对应的工程帧。
    pub origin_sample: i64,
    /// 左平面。
    pub left: Vec<f32>,
    /// 右平面；**空 = 与左平面相同**（单声道素材的存储形态）。
    pub right: Vec<f32>,
    /// 本份磁带真的属于我们的工程帧区间（升序、不重叠）。空 = 整份都算。
    pub covered: Vec<Range<i64>>,
    /// 内存预算凭证。
    pub reservation: Option<crate::budget::Reservation>,
}

impl CaptureTape {
    /// 两个平面逐 bit 相同时合并成单平面，并把多余的额度退回去。
    ///
    /// 返回 `true` 表示发生了合并。语义与渲染侧的快照归并**逐字相同**：有符号零与
    /// 真实立体声绝不合并（`-0.0` 与 `0.0` 的比特不同）。
    pub fn compact_mono(&mut self) -> bool {
        if self.right.is_empty() {
            return false;
        }
        if self.left.len() != self.right.len() {
            return false;
        }
        if !self
            .left
            .iter()
            .zip(&self.right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
        {
            return false;
        }
        let Some(bytes) = self.right.len().checked_mul(4) else {
            return false;
        };
        let refund = match &mut self.reservation {
            Some(reservation) => reservation.split_off(bytes),
            None => None,
        };
        drop(std::mem::take(&mut self.right));
        drop(refund);
        true
    }
}

/// 由时间线合成的一段 1:1 播放区间（无拉伸、无淡化）。
///
/// 【为什么是这套字段】渲染入口的几何校验以 ARA 的区间形状为准；捕获片段与它 1:1
/// 对应，因此合成出来的就是精确描述 —— 整条下游渲染路径不需要为捕获分叉。
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureRegion {
    /// 片段名。
    pub name: Option<String>,
    /// 源 id（`CapturePcm` 的键）。
    pub audio_source_persistent_id: String,
    /// 素材内偏移的所属 id（捕获用 clip id）。
    pub audio_modification_persistent_id: String,
    /// 序列下标。
    pub region_sequence_index: Option<usize>,
    /// 该区间归属的宿主序列（捕获没有，恒 `None`）。
    pub sequence_model_ref: Option<u64>,
    /// 源内起点（秒）。
    pub start_in_modification_time: f64,
    /// 源内长度（秒）。
    pub duration_in_modification_time: f64,
    /// 播放起点（秒）。
    pub start_in_playback_time: f64,
    /// 播放长度（秒）。
    pub duration_in_playback_time: f64,
    /// 是否启用变速（捕获恒 `false`）。
    pub is_timestretch_enabled: bool,
    /// 头部是否有基于内容的淡化。
    pub has_content_based_fade_at_head: bool,
    /// 尾部是否有基于内容的淡化。
    pub has_content_based_fade_at_tail: bool,
}
