//! HFSPeaks v2 多级 Mipmap 峰值格式
//!
//! 参考 Reaper REAPEAKS 格式设计，支持多级分辨率峰值数据，
//! 实现任意缩放级别的快速波形渲染。
//!
//! ## 文件格式布局
//! ```text
//! Header (48 bytes)
//! ├─ Magic: "HFSP" (4 bytes)
//! ├─ Version: u16 (2 bytes)
//! ├─ Channels: u16 (2 bytes)
//! ├─ Sample Rate: u32 (4 bytes)
//! ├─ Total Frames: u64 (8 bytes)
//! ├─ Source File Size: u64 (8 bytes)
//! ├─ Source Modified (ns): u64 (8 bytes)
//! ├─ Mipmap Count: u32 (4 bytes)
//! └─ Reserved: [u8; 8]
//!
//! Mipmap Headers (n × 16 bytes)
//! ├─ Division Factor: u32 (samples per peak)
//! ├─ Peak Count: u32
//! └─ Data Offset: u64
//!
//! Mipmap Data
//! ├─ Level 0: peak_count × channels × 8 bytes (min: f32, max: f32)
//! ├─ Level 1: ...
//! └─ Level n: ...
//! ```

use std::io::{Read, Write};


// ============== 常量定义 ==============

/// 文件魔数
pub const MAGIC: &[u8; 4] = b"HFSP";

/// 当前格式版本
///
/// v3：峰值数据真正按声道存储（`min`/`max` 为 channel-major 平面拼接，
/// 每峰值 `channels` 组极值）。v2 及更早的文件虽然头部声明了 `channels`，
/// 但数据实际只含一组跨声道合并极值（读取端按 channels 倍增读取必然
/// EOF 失败），本版本一并修复；VERSION 递增使旧缓存全部重建。
pub const VERSION: u16 = 3;

/// WFPK IPC 载荷格式版本（v2：28 字节头含 channels，逐声道分块）。
pub const WFPK_FORMAT_VERSION: u32 = 2;

/// 最大 mipmap 级别数
pub const MAX_MIPMAP_LEVELS: usize = 3;

/// 默认 mipmap 除数因子 (针对 44.1kHz 优化)
/// 三级 mipmap 缓存方案：
/// - L0 (div=16):   精细级，近距离对轨，spp ≤ 512
/// - L1 (div=512):  中间级，日常编辑，512 < spp ≤ 1024
/// - L2 (div=4096): 全局级，预览/导航，spp > 1024
pub const DEFAULT_DIVISION_FACTORS: [u32; MAX_MIPMAP_LEVELS] = [
    16,   // Level 0: ~2756 peaks/sec at 44.1kHz (精细级，近距离对轨)
    512,  // Level 1: ~86 peaks/sec at 44.1kHz (中间级，日常编辑)
    4096, // Level 2: ~11 peaks/sec at 44.1kHz (全局级，预览/导航)
];

/// 级别选择的 spp (samples_per_pixel) 阈值
/// spp ≤ SPP_THRESHOLDS[0] → L0
/// SPP_THRESHOLDS[0] < spp ≤ SPP_THRESHOLDS[1] → L1
/// spp > SPP_THRESHOLDS[1] → L2
#[allow(dead_code)]
pub const SPP_THRESHOLDS: [f64; 2] = [512.0, 1024.0];

// ============== 文件头结构 ==============

/// HFSPeaks 文件头 (48 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct HfsPeakHeader {
    /// 魔数 "HFSP"
    pub magic: [u8; 4],
    /// 格式版本
    pub version: u16,
    /// 声道数
    pub channels: u16,
    /// 采样率
    pub sample_rate: u32,
    /// 总帧数
    pub total_frames: u64,
    /// 源文件大小
    pub source_file_size: u64,
    /// 源文件修改时间 (纳秒)
    pub source_modified_ns: u64,
    /// Mipmap 级别数量
    pub mipmap_count: u32,
    /// 保留字段
    pub reserved: [u8; 8],
}

impl Default for HfsPeakHeader {
    fn default() -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            channels: 0,
            sample_rate: 0,
            total_frames: 0,
            source_file_size: 0,
            source_modified_ns: 0,
            mipmap_count: 0,
            reserved: [0; 8],
        }
    }
}

impl HfsPeakHeader {
    /// 头部大小 (bytes)
    pub const SIZE: usize = 48;

    /// 从字节数组解析
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != MAGIC {
            return None;
        }

        Some(Self {
            magic,
            version: u16::from_le_bytes([bytes[4], bytes[5]]),
            channels: u16::from_le_bytes([bytes[6], bytes[7]]),
            sample_rate: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            total_frames: u64::from_le_bytes([
                bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18],
                bytes[19],
            ]),
            source_file_size: u64::from_le_bytes([
                bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26],
                bytes[27],
            ]),
            source_modified_ns: u64::from_le_bytes([
                bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34],
                bytes[35],
            ]),
            mipmap_count: u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]),
            reserved: [
                bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46],
                bytes[47],
            ],
        })
    }

    /// 转换为字节数组
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];

        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.channels.to_le_bytes());
        buf[8..12].copy_from_slice(&self.sample_rate.to_le_bytes());
        buf[12..20].copy_from_slice(&self.total_frames.to_le_bytes());
        buf[20..28].copy_from_slice(&self.source_file_size.to_le_bytes());
        buf[28..36].copy_from_slice(&self.source_modified_ns.to_le_bytes());
        buf[36..40].copy_from_slice(&self.mipmap_count.to_le_bytes());
        buf[40..48].copy_from_slice(&self.reserved);

        buf
    }
}

// ============== Mipmap 头结构 ==============

/// 单个 Mipmap 级别的头部信息 (16 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct MipmapHeader {
    /// 除数因子：每个峰值代表的采样数
    pub division_factor: u32,
    /// 峰值数量
    pub peak_count: u32,
    /// 数据在文件中的偏移量
    pub data_offset: u64,
}

impl Default for MipmapHeader {
    fn default() -> Self {
        Self {
            division_factor: 0,
            peak_count: 0,
            data_offset: 0,
        }
    }
}

impl MipmapHeader {
    /// 头部大小 (bytes)
    pub const SIZE: usize = 16;

    /// 从字节数组解析
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        Some(Self {
            division_factor: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            peak_count: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            data_offset: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
        })
    }

    /// 转换为字节数组
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];

        buf[0..4].copy_from_slice(&self.division_factor.to_le_bytes());
        buf[4..8].copy_from_slice(&self.peak_count.to_le_bytes());
        buf[8..16].copy_from_slice(&self.data_offset.to_le_bytes());

        buf
    }
}

// ============== Mipmap 数据结构 ==============

/// 单个 Mipmap 级别的峰值数据（v3：逐声道存储）。
///
/// `min`/`max` 为 **channel-major 平面拼接**：
/// `min[ch * peak_count + p]` 为第 `ch` 声道第 `p` 个峰值的极小值。
/// 磁盘布局 = min 全部值（按此顺序）+ max 全部值，与 [`MipmapData::write_to`]
/// / [`MipmapData::read_from`] 一一对应。
#[derive(Debug, Clone)]
pub struct MipmapData {
    /// 声道数（数据布局的一部分）。
    pub channels: u16,
    /// 最小值数组（channel-major 拼接，长度 = peak_count × channels）。
    pub min: Vec<f32>,
    /// 最大值数组（channel-major 拼接，长度 = peak_count × channels）。
    pub max: Vec<f32>,
}

impl MipmapData {
    /// 创建空的峰值数据
    pub fn new() -> Self {
        Self {
            channels: 1,
            min: Vec::new(),
            max: Vec::new(),
        }
    }

    /// 每声道峰值数
    pub fn len(&self) -> usize {
        let ch = self.channels.max(1) as usize;
        self.min.len().min(self.max.len()) / ch
    }

    /// 是否为空
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 指定声道的 min 切片（长度 = peak_count）。
    pub fn channel_min(&self, channel: usize) -> &[f32] {
        let ch = self.channels.max(1) as usize;
        let count = self.len();
        let start = channel.min(ch.saturating_sub(1)) * count;
        &self.min[start..start + count]
    }

    /// 指定声道的 max 切片（长度 = peak_count）。
    pub fn channel_max(&self, channel: usize) -> &[f32] {
        let ch = self.channels.max(1) as usize;
        let count = self.len();
        let start = channel.min(ch.saturating_sub(1)) * count;
        &self.max[start..start + count]
    }

    /// 计算数据大小 (bytes)
    #[allow(dead_code)]
    pub fn data_size(&self) -> usize {
        (self.min.len() + self.max.len()) * 4
    }

    /// 写入到 Writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for &v in &self.min {
            writer.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.max {
            writer.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// 从 Reader 读取
    pub fn read_from<R: Read>(
        reader: &mut R,
        peak_count: usize,
        channels: u16,
    ) -> std::io::Result<Self> {
        let total_values = peak_count * channels.max(1) as usize;

        let mut min = vec![0.0f32; total_values];
        let mut max = vec![0.0f32; total_values];

        // 读取 min 值
        for v in &mut min {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }

        // 读取 max 值
        for v in &mut max {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }

        Ok(Self {
            channels: channels.max(1),
            min,
            max,
        })
    }
}

impl Default for MipmapData {
    fn default() -> Self {
        Self::new()
    }
}

// ============== 完整的 HFSPeaks 文件结构 ==============

/// 完整的 HFSPeaks v2 文件数据
#[derive(Debug, Clone)]
pub struct HfsPeakFile {
    /// 文件头
    pub header: HfsPeakHeader,
    /// Mipmap 头部列表
    pub mipmap_headers: Vec<MipmapHeader>,
    /// Mipmap 数据列表
    pub mipmap_data: Vec<MipmapData>,
}

impl HfsPeakFile {
    /// 创建新的 HFSPeaks 文件结构
    pub fn new(
        channels: u16,
        sample_rate: u32,
        total_frames: u64,
        source_file_size: u64,
        source_modified_ns: u64,
    ) -> Self {
        Self {
            header: HfsPeakHeader {
                magic: *MAGIC,
                version: VERSION,
                channels,
                sample_rate,
                total_frames,
                source_file_size,
                source_modified_ns,
                mipmap_count: 0,
                reserved: [0; 8],
            },
            mipmap_headers: Vec::new(),
            mipmap_data: Vec::new(),
        }
    }

    /// 添加一个 mipmap 级别
    pub fn add_mipmap(&mut self, division_factor: u32, data: MipmapData) {
        let peak_count = data.len() as u32;

        // 计算数据偏移量
        let data_offset = if self.mipmap_headers.is_empty() {
            // 第一个 mipmap 数据紧跟在所有头部之后
            let header_size = HfsPeakHeader::SIZE as u64;
            let mipmap_headers_size =
                (self.mipmap_headers.len() + 1) as u64 * MipmapHeader::SIZE as u64;
            header_size + mipmap_headers_size
        } else {
            // 后续 mipmap 数据在前一个 mipmap 数据之后
            let last = self.mipmap_headers.last().unwrap();
            last.data_offset + last.peak_count as u64 * self.header.channels as u64 * 8
        };

        self.mipmap_headers.push(MipmapHeader {
            division_factor,
            peak_count,
            data_offset,
        });
        self.mipmap_data.push(data);
        self.header.mipmap_count = self.mipmap_headers.len() as u32;
    }

    /// 根据缩放级别选择最佳 mipmap 级别
    ///
    /// # 参数
    /// - `samples_per_pixel`: 每像素对应的采样数
    ///
    /// # 返回
    /// 最佳 mipmap 级别索引
    #[allow(dead_code)]
    pub fn select_mipmap_level(&self, samples_per_pixel: f64) -> usize {
        // 根据 spp 阈值选择最佳 mipmap 级别
        // spp ≤ 512 → L0 (div=16, 精细级)
        // 512 < spp ≤ 1024 → L1 (div=512, 中间级)
        // spp > 1024 → L2 (div=4096, 全局级)
        let max_level = self.mipmap_headers.len().saturating_sub(1);

        if samples_per_pixel <= SPP_THRESHOLDS[0] {
            0
        } else if samples_per_pixel <= SPP_THRESHOLDS[1] {
            1.min(max_level)
        } else {
            2.min(max_level)
        }
    }

    /// Estimated in-memory byte cost of all mipmap vectors.
    pub fn estimated_byte_size(&self) -> u64 {
        // min/max 各 (peak_count × channels) 个 f32。
        self.mipmap_data
            .iter()
            .map(|data| (data.min.len() + data.max.len()) as u64)
            .sum::<u64>()
            .saturating_mul(4)
    }

    /// 将指定级别的 mipmap 数据序列化为二进制格式
    ///
    /// 二进制协议格式：
    /// ```text
    /// [Header (20 bytes)] [min_data] [max_data]
    ///
    /// Header:
    ///   bytes 0-3:   magic "WFPK" (4 bytes)
    ///   bytes 4-7:   sample_rate (u32, little-endian)
    ///   bytes 8-11:  division_factor (u32, little-endian)
    ///   bytes 12-15: peak_count (u32, little-endian)
    ///   bytes 16-19: level (u32, little-endian)
    ///
    /// min_data: peak_count × f32 (little-endian)
    /// max_data: peak_count × f32 (little-endian)
    /// ```
    pub fn to_binary_level(&self, level: usize) -> Vec<u8> {
        let level = level.min(self.mipmap_data.len().saturating_sub(1));
        if self.mipmap_data.is_empty() {
            return Vec::new();
        }

        let data = &self.mipmap_data[level];
        let mh = &self.mipmap_headers[level];
        let count = data.len();
        let channels = data.channels.max(1) as usize;
        let mut buf = Vec::with_capacity(28 + count * channels * 8);

        // Header (28 bytes)
        buf.extend_from_slice(b"WFPK");
        buf.extend_from_slice(&WFPK_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.header.sample_rate.to_le_bytes());
        buf.extend_from_slice(&mh.division_factor.to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        buf.extend_from_slice(&(level as u32).to_le_bytes());
        buf.extend_from_slice(&(data.channels as u32).to_le_bytes());

        // 逐声道分块：ch0_min / ch0_max / ch1_min / ch1_max …
        for ch in 0..channels {
            for &v in data.channel_min(ch) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            for &v in data.channel_max(ch) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        buf
    }
}

// ============== API 响应结构 ==============

/// Byte-budgeted in-memory cache for decoded peak files.
///
/// Entries whose Arc is still held outside the cache are treated as pinned;
/// eviction skips them until the external owner drops its reference.
#[derive(Debug)]
pub struct WaveformPeakCache {
    entries: Vec<(String, std::sync::Arc<HfsPeakFile>, u64)>,
    total_bytes: u64,
    budget_bytes: u64,
}

impl Default for WaveformPeakCache {
    fn default() -> Self {
        Self::new(256 * 1024 * 1024)
    }
}

impl WaveformPeakCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
            budget_bytes: budget_bytes.max(1),
        }
    }

    pub fn get(&mut self, source_path: &str) -> Option<std::sync::Arc<HfsPeakFile>> {
        let index = self.entries.iter().position(|(key, _, _)| key == source_path)?;
        let value = self.entries[index].1.clone();
        self.entries.remove(index);
        self.entries.push((source_path.to_string(), value.clone(), value.estimated_byte_size()));
        Some(value)
    }

    pub fn insert(&mut self, source_path: &str, value: std::sync::Arc<HfsPeakFile>) {
        self.remove(source_path);
        let weight = value.estimated_byte_size();
        self.entries.push((source_path.to_string(), value, weight));
        self.total_bytes = self.total_bytes.saturating_add(weight);
        self.enforce_budget();
    }

    pub fn remove(&mut self, source_path: &str) -> bool {
        if let Some(index) = self.entries.iter().position(|(key, _, _)| key == source_path) {
            let (_, _, weight) = self.entries.remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(weight);
            return true;
        }
        false
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    // Cache introspection accessors: used by unit tests and diagnostics;
        // no callers in non-test builds.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    #[allow(dead_code)]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    fn enforce_budget(&mut self) {
        while self.total_bytes > self.budget_bytes && !self.entries.is_empty() {
            let mut evict_index = None;
            for (index, (_, value, _)) in self.entries.iter().enumerate() {
                if std::sync::Arc::strong_count(value) <= 1 {
                    evict_index = Some(index);
                    break;
                }
            }
            let Some(index) = evict_index else { break };
            let (_, _, weight) = self.entries.remove(index);
            self.total_bytes = self.total_bytes.saturating_sub(weight);
        }
    }
}

// ============== 辅助函数 ==============

/// 根据采样率计算 mipmap 除数因子
pub fn calculate_division_factors(sample_rate: u32) -> Vec<u32> {
    // 以 44100Hz 为基准，按比例调整
    let base_rate = 44100.0;
    let scale = sample_rate as f64 / base_rate;

    DEFAULT_DIVISION_FACTORS
        .iter()
        .map(|&d| (d as f64 * scale).round() as u32)
        .collect()
}

// ============== 多级 Mipmap 峰值计算 ==============

use std::path::Path;

/// 单个级别的逐声道累积器
struct LevelAccumulator {
    /// 各声道的运行 min（长度 = channels）
    acc_min: Vec<f32>,
    /// 各声道的运行 max（长度 = channels）
    acc_max: Vec<f32>,
    frame_count: usize,
}

/// 多级 Mipmap 峰值计算器（v3：逐声道）
#[allow(dead_code)]
pub struct MipmapPeakCalculator {
    /// 采样率
    sample_rate: u32,
    /// 声道数
    channels: u16,
    /// 总帧数
    total_frames: u64,
    /// 各级别的除数因子
    division_factors: Vec<u32>,
    /// 各级别的累积器
    accumulators: Vec<LevelAccumulator>,
}

impl MipmapPeakCalculator {
    /// 创建新的计算器
    pub fn new(sample_rate: u32, channels: u16, total_frames: u64) -> Self {
        let division_factors = calculate_division_factors(sample_rate);
        let ch = channels.max(1) as usize;
        let accumulators = division_factors
            .iter()
            .map(|_| LevelAccumulator {
                acc_min: vec![f32::INFINITY; ch],
                acc_max: vec![f32::NEG_INFINITY; ch],
                frame_count: 0,
            })
            .collect();

        Self {
            sample_rate,
            channels,
            total_frames,
            division_factors,
            accumulators,
        }
    }

    /// 处理一帧数据
    ///
    /// # 参数
    /// - `frame_min`: 各声道该帧的最小值（长度 = channels）
    /// - `frame_max`: 各声道该帧的最大值（长度 = channels）
    /// - `output_callback`: 当某级别完成一个峰值时调用
    ///   （参数为 level_idx、各声道 min 切片、各声道 max 切片）
    pub fn process_frame<F: FnMut(usize, &[f32], &[f32])>(
        &mut self,
        frame_min: &[f32],
        frame_max: &[f32],
        output_callback: &mut F,
    ) {
        for (level_idx, acc) in self.accumulators.iter_mut().enumerate() {
            // 更新累积器（逐声道）
            for (c, &v) in frame_min.iter().enumerate() {
                if v < acc.acc_min[c] {
                    acc.acc_min[c] = v;
                }
            }
            for (c, &v) in frame_max.iter().enumerate() {
                if v > acc.acc_max[c] {
                    acc.acc_max[c] = v;
                }
            }
            acc.frame_count += 1;

            // 检查是否需要输出
            let divisor = self.division_factors[level_idx] as usize;
            if acc.frame_count >= divisor {
                for v in acc.acc_min.iter_mut() {
                    *v = if v.is_finite() { *v } else { 0.0 };
                }
                for v in acc.acc_max.iter_mut() {
                    *v = if v.is_finite() { *v } else { 0.0 };
                }
                output_callback(level_idx, &acc.acc_min, &acc.acc_max);

                // 重置累积器
                for v in acc.acc_min.iter_mut() {
                    *v = f32::INFINITY;
                }
                for v in acc.acc_max.iter_mut() {
                    *v = f32::NEG_INFINITY;
                }
                acc.frame_count = 0;
            }
        }
    }

    /// 刷新剩余的累积数据
    pub fn flush<F: FnMut(usize, &[f32], &[f32])>(&mut self, mut output_callback: F) {
        for (level_idx, acc) in self.accumulators.iter_mut().enumerate() {
            if acc.frame_count > 0 {
                for v in acc.acc_min.iter_mut() {
                    *v = if v.is_finite() { *v } else { 0.0 };
                }
                for v in acc.acc_max.iter_mut() {
                    *v = if v.is_finite() { *v } else { 0.0 };
                }
                output_callback(level_idx, &acc.acc_min, &acc.acc_max);

                // 重置
                for v in acc.acc_min.iter_mut() {
                    *v = f32::INFINITY;
                }
                for v in acc.acc_max.iter_mut() {
                    *v = f32::NEG_INFINITY;
                }
                acc.frame_count = 0;
            }
        }
    }
}

/// 逐声道峰值输出缓冲（channel-major 存储，最终构建 [`MipmapData`]）
struct PerChannelPeakBuffer {
    channels: usize,
    /// min[channel][peak]
    min: Vec<Vec<f32>>,
    /// max[channel][peak]
    max: Vec<Vec<f32>>,
}

impl PerChannelPeakBuffer {
    fn new(channels: u16) -> Self {
        let ch = channels.max(1) as usize;
        Self {
            channels: ch,
            min: (0..ch).map(|_| Vec::new()).collect(),
            max: (0..ch).map(|_| Vec::new()).collect(),
        }
    }

    fn push(&mut self, ch_min: &[f32], ch_max: &[f32]) {
        for (c, &v) in ch_min.iter().enumerate().take(self.channels) {
            self.min[c].push(v);
        }
        for (c, &v) in ch_max.iter().enumerate().take(self.channels) {
            self.max[c].push(v);
        }
    }

    fn into_mipmap_data(self) -> MipmapData {
        let peak_count = self.min.first().map(|v| v.len()).unwrap_or(0);
        let mut min = Vec::with_capacity(peak_count * self.channels);
        let mut max = Vec::with_capacity(peak_count * self.channels);
        for ch in 0..self.channels {
            min.extend_from_slice(&self.min[ch]);
            max.extend_from_slice(&self.max[ch]);
        }
        MipmapData {
            channels: self.channels as u16,
            min,
            max,
        }
    }
}

/// 从音频文件计算多级 mipmap 峰值
///
/// 支持 WAV (通过 hound) 和其他格式 (通过 symphonia)
///
/// # 参数
/// - `progress_cb`: 可选的进度回调，参数为 0.0~1.0 的进度值
#[allow(dead_code)]
pub fn compute_mipmap_peaks(path: &Path) -> Result<HfsPeakFile, String> {
    compute_mipmap_peaks_with_progress(path, None::<fn(f32)>)
}

/// 带进度回调的多级 mipmap 峰值计算
pub fn compute_mipmap_peaks_with_progress<F: FnMut(f32)>(
    path: &Path,
    mut progress_cb: Option<F>,
) -> Result<HfsPeakFile, String> {
    // 获取文件元数据
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let source_file_size = meta.len();
    let source_modified_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // 尝试用 hound 处理 WAV 文件
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        if let Ok(peaks) =
            compute_mipmap_peaks_hound(path, source_file_size, source_modified_ns, &mut progress_cb)
        {
            return Ok(peaks);
        }
    }

    // 其他格式（含视频容器中的音轨）统一走 Symphonia 解码峰值计算。
    compute_mipmap_peaks_media(path, source_file_size, source_modified_ns, &mut progress_cb)
}

/// 使用 hound 计算 WAV 文件的多级峰值
fn compute_mipmap_peaks_hound<F: FnMut(f32)>(
    path: &Path,
    source_file_size: u64,
    source_modified_ns: u64,
    progress_cb: &mut Option<F>,
) -> Result<HfsPeakFile, String> {
    use hound::{SampleFormat, WavReader};

    let reader = WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();

    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err("invalid wav spec".to_string());
    }

    let channels = spec.channels as u16;
    let total_frames = reader.duration() as u64;
    let sample_rate = spec.sample_rate;

    // 初始化输出缓冲区（逐声道）
    let division_factors = calculate_division_factors(sample_rate);
    let mut output_buffers: Vec<PerChannelPeakBuffer> = division_factors
        .iter()
        .map(|_| PerChannelPeakBuffer::new(channels))
        .collect();

    // 创建计算器
    let mut calculator = MipmapPeakCalculator::new(sample_rate, channels, total_frames);

    // 重新打开文件读取采样数据
    let mut reader = WavReader::open(path).map_err(|e| e.to_string())?;

    // 输出回调（per-channel 切片）
    let mut output_callback = |level: usize, min: &[f32], max: &[f32]| {
        if level < output_buffers.len() {
            output_buffers[level].push(min, max);
        }
    };

    // 进度跟踪
    let mut frames_processed: u64 = 0;
    let progress_interval = (total_frames / 20).max(1); // 每 ~5% 报告一次

    // 每帧逐声道极值暂存（复用，避免逐帧分配）
    let ch_usize = channels.max(1) as usize;
    let mut frame_min = vec![0.0f32; ch_usize];
    let mut frame_max = vec![0.0f32; ch_usize];

    // 根据格式处理采样
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let mut buf = vec![0i16; ch_usize];
            let mut i = 0usize;
            for s in reader.samples::<i16>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= ch_usize {
                    i = 0;
                    frame_channel_extremes_i16(&buf, &mut frame_min, &mut frame_max);
                    calculator.process_frame(&frame_min, &frame_max, &mut output_callback);
                    frames_processed += 1;
                    if frames_processed % progress_interval == 0 {
                        if let Some(cb) = progress_cb.as_mut() {
                            cb(frames_processed as f32 / total_frames.max(1) as f32);
                        }
                    }
                }
            }
        }
        (SampleFormat::Int, 24) => {
            let denom = (1u32 << 23) as f32;
            let mut buf = vec![0i32; ch_usize];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= ch_usize {
                    i = 0;
                    frame_channel_extremes_i32(&buf, denom, &mut frame_min, &mut frame_max);
                    calculator.process_frame(&frame_min, &frame_max, &mut output_callback);
                    frames_processed += 1;
                    if frames_processed % progress_interval == 0 {
                        if let Some(cb) = progress_cb.as_mut() {
                            cb(frames_processed as f32 / total_frames.max(1) as f32);
                        }
                    }
                }
            }
        }
        (SampleFormat::Int, 32) => {
            let mut buf = vec![0i32; ch_usize];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= ch_usize {
                    i = 0;
                    frame_channel_extremes_i32(&buf, i32::MAX as f32, &mut frame_min, &mut frame_max);
                    calculator.process_frame(&frame_min, &frame_max, &mut output_callback);
                    frames_processed += 1;
                    if frames_processed % progress_interval == 0 {
                        if let Some(cb) = progress_cb.as_mut() {
                            cb(frames_processed as f32 / total_frames.max(1) as f32);
                        }
                    }
                }
            }
        }
        (SampleFormat::Float, 32) => {
            let mut buf = vec![0f32; ch_usize];
            let mut i = 0usize;
            for s in reader.samples::<f32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= ch_usize {
                    i = 0;
                    frame_channel_extremes_f32(&buf, &mut frame_min, &mut frame_max);
                    calculator.process_frame(&frame_min, &frame_max, &mut output_callback);
                    frames_processed += 1;
                    if frames_processed % progress_interval == 0 {
                        if let Some(cb) = progress_cb.as_mut() {
                            cb(frames_processed as f32 / total_frames.max(1) as f32);
                        }
                    }
                }
            }
        }
        _ => return Err("unsupported wav format".to_string()),
    }

    // 刷新剩余数据
    calculator.flush(&mut output_callback);

    // 构建 HfsPeakFile
    let mut file = HfsPeakFile::new(
        channels,
        sample_rate,
        total_frames,
        source_file_size,
        source_modified_ns,
    );

    for (level_idx, buffer) in output_buffers.into_iter().enumerate() {
        file.add_mipmap(division_factors[level_idx], buffer.into_mipmap_data());
    }

    Ok(file)
}

/// 使用 Symphonia 计算非 WAV 媒体（音频与视频容器）的多级峰值。
fn compute_mipmap_peaks_media<F: FnMut(f32)>(
    path: &Path,
    source_file_size: u64,
    source_modified_ns: u64,
    progress_cb: &mut Option<F>,
) -> Result<HfsPeakFile, String> {
    let probe = crate::media::probe_media(path, 0, None)
        .ok_or_else(|| "symphonia media probe failed".to_string())?;
    let sample_rate = probe.sample_rate.max(1);
    let channels = probe.channels.max(1);
    let total_frames = probe.total_frames;

    let division_factors = calculate_division_factors(sample_rate);
    let mut output_buffers: Vec<PerChannelPeakBuffer> = division_factors
        .iter()
        .map(|_| PerChannelPeakBuffer::new(channels))
        .collect();
    let mut calculator = MipmapPeakCalculator::new(sample_rate, channels, total_frames);

    let mut output_callback = |level: usize, min: &[f32], max: &[f32]| {
        if level < output_buffers.len() {
            output_buffers[level].push(min, max);
        }
    };

    let mut frames_processed: u64 = 0;
    let progress_interval = if total_frames > 0 {
        (total_frames / 20).max(1)
    } else {
        44100
    };

    // 每帧逐声道极值暂存（复用，避免逐帧分配）
    let mut frame_min = vec![0.0f32; channels.max(1) as usize];
    let mut frame_max = vec![0.0f32; channels.max(1) as usize];

    crate::media::visit_media_audio_frames(
        path,
        Some(probe.audio_stream_index),
        |frame, _rate, ch| {
            let ch = ch.max(1) as usize;
            let frames = frame.len() / ch;
            for f in 0..frames {
                let base = f * ch;
                for (c, slot) in frame_min.iter_mut().enumerate() {
                    let v = if c < ch {
                        frame.get(base + c).copied().unwrap_or(0.0)
                    } else {
                        // 报告声道数少于缓冲槽位时补零，避免残留上一帧极值。
                        0.0
                    };
                    *slot = v;
                    frame_max[c] = v;
                }
                calculator.process_frame(&frame_min, &frame_max, &mut output_callback);
                frames_processed += 1;
                if frames_processed % progress_interval == 0 {
                    if let Some(cb) = progress_cb.as_mut() {
                        if total_frames > 0 {
                            cb(frames_processed as f32 / total_frames as f32);
                        } else {
                            let estimated_total = source_file_size / ((channels as u64) * 4).max(1);
                            cb((frames_processed as f32 / estimated_total as f32).min(0.99));
                        }
                    }
                }
            }
            Ok(())
        },
    )
    .map_err(|e| e)?;

    calculator.flush(&mut output_callback);

    let mut file = HfsPeakFile::new(
        channels,
        sample_rate,
        total_frames,
        source_file_size,
        source_modified_ns,
    );

    for (level_idx, buffer) in output_buffers.into_iter().enumerate() {
        file.add_mipmap(division_factors[level_idx], buffer.into_mipmap_data());
    }

    Ok(file)
}

// ============== 辅助函数 ==============

/// 计算 i16 帧缓冲的**逐声道**极值（v3）
fn frame_channel_extremes_i16(buf: &[i16], out_min: &mut [f32], out_max: &mut [f32]) {
    for (c, &x) in buf.iter().enumerate().take(out_min.len()) {
        let v = x as f32 / i16::MAX as f32;
        out_min[c] = v;
        out_max[c] = v;
    }
}

/// 计算 i32 帧缓冲的**逐声道**极值（v3）
fn frame_channel_extremes_i32(buf: &[i32], denom: f32, out_min: &mut [f32], out_max: &mut [f32]) {
    for (c, &x) in buf.iter().enumerate().take(out_min.len()) {
        let v = x as f32 / denom;
        out_min[c] = v;
        out_max[c] = v;
    }
}

/// 计算 f32 帧缓冲的**逐声道**极值（v3）
fn frame_channel_extremes_f32(buf: &[f32], out_min: &mut [f32], out_max: &mut [f32]) {
    for (c, &v) in buf.iter().enumerate().take(out_min.len()) {
        out_min[c] = v;
        out_max[c] = v;
    }
}


// ============== 文件存储与加载 ==============

use std::fs::File;
use std::io::{BufReader, BufWriter};

impl HfsPeakFile {
    /// 保存峰值数据到文件
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // 写入临时文件，然后原子性重命名
        let tmp_path = path.with_extension("hfspeaks.tmp");
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        // 写入文件头
        writer.write_all(&self.header.to_bytes())?;

        // 写入 mipmap headers
        for mh in &self.mipmap_headers {
            writer.write_all(&mh.to_bytes())?;
        }

        // 写入 mipmap data
        for data in &self.mipmap_data {
            data.write_to(&mut writer)?;
        }

        writer.flush()?;
        drop(writer);

        // 原子性重命名
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// 从文件加载峰值数据
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 读取文件头
        let mut header_buf = [0u8; HfsPeakHeader::SIZE];
        reader.read_exact(&mut header_buf)?;

        let header = HfsPeakHeader::from_bytes(&header_buf).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid header")
        })?;

        // 验证魔数和版本
        if &header.magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic",
            ));
        }
        if header.version > VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported version",
            ));
        }

        // 读取 mipmap headers
        let mipmap_count = header.mipmap_count as usize;
        let mut mipmap_headers = Vec::with_capacity(mipmap_count);

        for _ in 0..mipmap_count {
            let mut mh_buf = [0u8; MipmapHeader::SIZE];
            reader.read_exact(&mut mh_buf)?;
            let mh = MipmapHeader::from_bytes(&mh_buf).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid mipmap header")
            })?;
            mipmap_headers.push(mh);
        }

        // 读取 mipmap data
        let mut mipmap_data = Vec::with_capacity(mipmap_count);
        for mh in &mipmap_headers {
            let data = MipmapData::read_from(&mut reader, mh.peak_count as usize, header.channels)?;
            mipmap_data.push(data);
        }

        Ok(Self {
            header,
            mipmap_headers,
            mipmap_data,
        })
    }

    /// 从文件加载，仅读取指定级别的 mipmap 数据
    /// 用于按需加载，减少内存占用
    #[allow(dead_code)]
    pub fn load_mipmap_level(
        path: &Path,
        level: usize,
    ) -> std::io::Result<(HfsPeakHeader, MipmapHeader, MipmapData)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 读取文件头
        let mut header_buf = [0u8; HfsPeakHeader::SIZE];
        reader.read_exact(&mut header_buf)?;

        let header = HfsPeakHeader::from_bytes(&header_buf).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid header")
        })?;

        if level >= header.mipmap_count as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "level out of range",
            ));
        }

        // 跳过前面的 mipmap headers
        let mipmap_count = header.mipmap_count as usize;
        let mut mipmap_headers = Vec::with_capacity(mipmap_count);

        for _ in 0..mipmap_count {
            let mut mh_buf = [0u8; MipmapHeader::SIZE];
            reader.read_exact(&mut mh_buf)?;
            let mh = MipmapHeader::from_bytes(&mh_buf).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid mipmap header")
            })?;
            mipmap_headers.push(mh);
        }

        let target_mh = &mipmap_headers[level];

        // 跳到目标数据位置
        let current_pos = HfsPeakHeader::SIZE + mipmap_count * MipmapHeader::SIZE;
        let target_pos = target_mh.data_offset as u64;

        if target_pos > current_pos as u64 {
            // 需要跳过前面级别的数据
            let skip_bytes = target_pos - current_pos as u64;
            std::io::copy(&mut reader.by_ref().take(skip_bytes), &mut std::io::sink())?;
        }

        // 读取目标数据
        let data =
            MipmapData::read_from(&mut reader, target_mh.peak_count as usize, header.channels)?;

        Ok((header, *target_mh, data))
    }
}

// ============== 缓存管理 ==============

use std::path::PathBuf;

/// 缓存清理统计（原 waveform_disk_cache::ClearStats）
#[derive(Debug, Clone, Copy)]
pub struct ClearStats {
    pub removed_files: u64,
    pub removed_bytes: u64,
}

/// 获取默认缓存目录路径（原 waveform_disk_cache::default_cache_dir）
pub fn default_cache_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.join("peaks");
        }
    }
    std::env::temp_dir()
        .join("hifishifter")
        .join("waveform_peaks_cache")
}

/// 确保目录存在（原 waveform_disk_cache::ensure_dir）
pub fn ensure_cache_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())
}

/// 清理缓存目录中的 .hfspeaks 文件（原 waveform_disk_cache::clear_dir）
pub fn clear_cache_dir(dir: &Path) -> ClearStats {
    let mut removed_files = 0u64;
    let mut removed_bytes = 0u64;

    let entries = match std::fs::read_dir(dir) {
        Ok(v) => v,
        Err(_) => {
            return ClearStats {
                removed_files,
                removed_bytes,
            }
        }
    };

    for e in entries.flatten() {
        let p = e.path();
        if p.is_file() {
            let is_peaks = p
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("hfspeaks"))
                .unwrap_or(false);
            if !is_peaks {
                continue;
            }
            if let Ok(meta) = e.metadata() {
                removed_bytes = removed_bytes.saturating_add(meta.len());
            }
            if std::fs::remove_file(&p).is_ok() {
                removed_files = removed_files.saturating_add(1);
            }
        }
    }

    ClearStats {
        removed_files,
        removed_bytes,
    }
}

/// HFSPeaks v2 缓存管理器
pub struct HfsPeaksCache {
    cache_dir: PathBuf,
}

impl HfsPeaksCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// 确保缓存目录存在
    pub fn ensure_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.cache_dir)
    }

    /// 计算缓存文件路径
    ///
    /// 使用文件路径 + 大小 + 修改时间的哈希作为缓存键
    pub fn cache_file_path(&self, source_path: &Path) -> PathBuf {
        let canonical = source_path
            .canonicalize()
            .unwrap_or_else(|_| source_path.to_path_buf());
        let (len, mtime_ns) = get_metadata_fingerprint(&canonical);

        let mut hasher = blake3::Hasher::new();
        hasher.update(canonical.to_string_lossy().as_bytes());
        hasher.update(b"\n");
        hasher.update(&len.to_le_bytes());
        hasher.update(&mtime_ns.to_le_bytes());
        hasher.update(&VERSION.to_le_bytes());

        let hash = hasher.finalize();
        let name = format!("{}.hfspeaks", hash.to_hex());
        self.cache_dir.join(name)
    }

    /// 尝试从缓存加载
    pub fn try_load(&self, source_path: &Path) -> Option<HfsPeakFile> {
        let cache_path = self.cache_file_path(source_path);

        // 验证缓存是否有效
        if !cache_path.exists() {
            return None;
        }

        // 加载并验证
        let file = HfsPeakFile::load(&cache_path).ok()?;

        // 验证源文件指纹
        let (current_len, current_mtime) = get_metadata_fingerprint(source_path);
        if file.header.source_file_size != current_len
            || file.header.source_modified_ns != current_mtime
        {
            // 源文件已更改，缓存无效
            return None;
        }

        Some(file)
    }

    /// 保存到缓存
    pub fn save(&self, source_path: &Path, peaks: &HfsPeakFile) -> std::io::Result<()> {
        self.ensure_dir()?;
        let cache_path = self.cache_file_path(source_path);
        peaks.save(&cache_path)
    }

    /// 获取或计算峰值数据
    ///
    /// 优先从缓存加载，缓存不存在时计算并保存
    #[allow(dead_code)]
    pub fn get_or_compute(&self, source_path: &Path) -> Result<HfsPeakFile, String> {
        // 尝试从缓存加载
        if let Some(cached) = self.try_load(source_path) {
            return Ok(cached);
        }

        // 计算新的峰值数据
        let peaks = compute_mipmap_peaks(source_path)?;

        // 保存到缓存
        if let Err(e) = self.save(source_path, &peaks) {
            log::error!("Warning: failed to save peaks cache: {}", e);
        }

        Ok(peaks)
    }

    /// 清理缓存目录
    #[allow(dead_code)]
    pub fn clear(&self) -> std::io::Result<(u64, u64)> {
        let mut removed_files = 0u64;
        let mut removed_bytes = 0u64;

        if !self.cache_dir.exists() {
            return Ok((removed_files, removed_bytes));
        }

        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let is_peaks = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("hfspeaks"))
                    .unwrap_or(false);

                if is_peaks {
                    if let Ok(meta) = entry.metadata() {
                        removed_bytes += meta.len();
                    }
                    if std::fs::remove_file(&path).is_ok() {
                        removed_files += 1;
                    }
                }
            }
        }

        Ok((removed_files, removed_bytes))
    }
}

/// 获取文件元数据指纹
fn get_metadata_fingerprint(path: &Path) -> (u64, u64) {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return (0, 0),
    };

    let len = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    (len, mtime_ns)
}

#[cfg(test)]
mod waveform_tile_tests {
    use super::*;

    fn fixture() -> HfsPeakFile {
        let mut file = HfsPeakFile::new(1, 4, 5, 12, 34);
        let data = MipmapData {
            channels: 1,
            min: vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            max: vec![1.0, 0.5, 0.0, -0.5, -1.0],
        };
        file.add_mipmap(1, data);
        file
    }

    #[test]
    fn waveform_peak_cache_evicts_by_bytes_and_respects_pins() {
        let mut cache = WaveformPeakCache::new(10);
        let first = std::sync::Arc::new(fixture());
        let second = std::sync::Arc::new(fixture());
        let pinned = first.clone();

        cache.insert("a", first);
        cache.insert("b", second);
        assert!(cache.len() <= 1);

        drop(pinned);
        let third = std::sync::Arc::new(fixture());
        cache.insert("c", third);
        assert!(cache.len() <= 1);
    }

    #[test]
    fn waveform_peak_cache_tracks_weights_and_clear() {
        let mut cache = WaveformPeakCache::new(10);
        let value = std::sync::Arc::new(fixture());
        let weight = value.estimated_byte_size();
        let pinned = value.clone();
        cache.insert("a", value);
        assert!(weight > 0);
        assert_eq!(cache.total_bytes(), weight);
        drop(pinned);
        assert!(cache.total_bytes() > 0);
        cache.clear();
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.is_empty());
    }


    #[test]
    fn mipmap_data_channel_slices_and_envelope() {
        let data = MipmapData {
            channels: 2,
            // channel-major：[ch0 p0, ch0 p1, ch1 p0, ch1 p1]
            min: vec![-1.0, -0.5, -0.2, 0.0],
            max: vec![1.0, 0.5, 0.2, 0.1],
        };
        assert_eq!(data.len(), 2);
        assert_eq!(data.channel_min(0), &[-1.0, -0.5][..]);
        assert_eq!(data.channel_min(1), &[-0.2, 0.0][..]);
        assert_eq!(data.channel_max(0), &[1.0, 0.5][..]);
        assert_eq!(data.channel_max(1), &[0.2, 0.1][..]);
        assert_eq!(data.data_size(), 4 * 4 * 2);
    }

    #[test]
    fn calculator_accumulates_per_channel() {
        let mut calc = MipmapPeakCalculator::new(44100, 2, 16);
        let mut buffers: Vec<PerChannelPeakBuffer> = vec![PerChannelPeakBuffer::new(2)];
        let mut cb = |level: usize, min: &[f32], max: &[f32]| {
            if level == 0 {
                buffers[0].push(min, max);
            }
        };
        // 2 帧一波（div=16 太大不会输出，改为直接验证 flush 输出）。
        // 左声道递减、右声道递增，验证两声道极值互不串扰。
        for f in 0..3u32 {
            let l = 0.5 - f as f32 * 0.1;
            let r = -0.5 + f as f32 * 0.1;
            calc.process_frame(&[l, r], &[l, r], &mut cb);
        }
        calc.flush(&mut cb);
        let data = buffers.pop().unwrap().into_mipmap_data();
        assert_eq!(data.channels, 2);
        assert_eq!(data.len(), 1);
        // L: 0.5, 0.4, 0.3 → min 0.3 / max 0.5；R: -0.5, -0.4, -0.3 → min -0.5 / max -0.3。
        assert_eq!(data.channel_min(0), &[0.3][..]);
        assert_eq!(data.channel_max(0), &[0.5][..]);
        assert_eq!(data.channel_min(1), &[-0.5][..]);
        assert_eq!(data.channel_max(1), &[-0.3][..]);
    }

    #[test]
    fn hsp_roundtrip_preserves_per_channel_peaks() {
        // 回归：v2 时代写入端只写一组跨声道极值，但读取端按 channels 倍增
        // 读取 —— 立体声文件的磁盘缓存永远读不回来。v3 起逐声道写入后
        // save → load 必须无损往返。
        let tmp = std::env::temp_dir().join(format!("hfs_stereo_test_{}.hsp", std::process::id()));
        let mut file = HfsPeakFile::new(2, 44100, 4, 100, 200);
        file.add_mipmap(
            1,
            MipmapData {
                channels: 2,
                min: vec![-1.0, -0.3, -0.5, 0.0],
                max: vec![1.0, 0.3, 0.5, 0.1],
            },
        );
        file.save(&tmp).expect("save hsp");

        let loaded = HfsPeakFile::load(&tmp).expect("load hsp");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!({ let c = loaded.header.channels; c }, 2);
        assert_eq!(loaded.mipmap_data[0].channels, 2);
        assert_eq!(loaded.mipmap_data[0].len(), 2);
        assert_eq!(loaded.mipmap_data[0].channel_min(0), &[-1.0, -0.3][..]);
        assert_eq!(loaded.mipmap_data[0].channel_min(1), &[-0.5, 0.0][..]);
        assert_eq!(loaded.mipmap_data[0].channel_max(0), &[1.0, 0.3][..]);
        assert_eq!(loaded.mipmap_data[0].channel_max(1), &[0.5, 0.1][..]);
    }

    #[test]
    fn wfpk_v2_layout_carries_channels_and_per_channel_blocks() {
        let mut file = HfsPeakFile::new(2, 48000, 2, 10, 20);
        file.add_mipmap(
            1,
            MipmapData {
                channels: 2,
                min: vec![-1.0, -0.25, -0.5, -0.125],
                max: vec![1.0, 0.25, 0.5, 0.125],
            },
        );
        let bytes = file.to_binary_level(0);

        assert_eq!(&bytes[0..4], b"WFPK");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), WFPK_FORMAT_VERSION);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 48000);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 2);

        // ch0_min, ch0_max, ch1_min, ch1_max
        let expect: Vec<Vec<f32>> = vec![
            vec![-1.0, -0.25],
            vec![1.0, 0.25],
            vec![-0.5, -0.125],
            vec![0.5, 0.125],
        ];
        let mut offset = 28;
        for block in &expect {
            for v in block {
                let raw: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
                assert_eq!(f32::from_le_bytes(raw), *v);
                offset += 4;
            }
        }
        assert_eq!(bytes.len(), offset);
    }

    #[test]
    fn compute_mipmap_peaks_wav_is_per_channel() {
        // 写一个 2 声道 wav：L = 递减、R = 递增，验证逐声道峰值互不串扰。
        let path = std::env::temp_dir().join(format!("hfs_ch_test_{}.wav", std::process::id()));
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for f in 0..512i16 {
                let l = 8000 - (f % 256) * 30;
                let r = -8000 + (f % 256) * 30;
                // hound 的 write_sample 按写入顺序交错排列声道。
                writer.write_sample(l).unwrap();
                writer.write_sample(r).unwrap();
            }
            writer.finalize().unwrap();
        }

        let file = compute_mipmap_peaks(&path).expect("compute peaks");
        let _ = std::fs::remove_file(&path);

        assert_eq!({ let c = file.header.channels; c }, 2);
        let l0 = &file.mipmap_data[0];
        assert_eq!(l0.channels, 2);
        // L 声道最大值应为正、R 声道最小值应为负。
        let l_max = l0.channel_max(0).iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let r_min = l0.channel_min(1).iter().fold(f32::INFINITY, |a, &b| a.min(b));
        assert!(l_max > 0.2, "L channel max should be positive, got {l_max}");
        assert!(r_min < -0.2, "R channel min should be negative, got {r_min}");
    }
}
