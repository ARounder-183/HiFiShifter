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

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

// ============== 常量定义 ==============

/// 文件魔数
pub const MAGIC: &[u8; 4] = b"HFSP";

/// 当前格式版本
pub const VERSION: u16 = 2;

/// 最大 mipmap 级别数
pub const MAX_MIPMAP_LEVELS: usize = 3;

/// 默认 mipmap 除数因子 (针对 44.1kHz 优化)
/// 三级 mipmap 缓存方案：
/// - L0 (div=64):   精细级，近距离对轨，spp ≤ 256
/// - L1 (div=512):  中间级，日常编辑，256 < spp ≤ 2048
/// - L2 (div=4096): 全局级，预览/导航，spp > 2048
pub const DEFAULT_DIVISION_FACTORS: [u32; MAX_MIPMAP_LEVELS] = [
    64,   // Level 0: ~689 peaks/sec at 44.1kHz (精细级，近距离对轨)
    512,  // Level 1: ~86 peaks/sec at 44.1kHz (中间级，日常编辑)
    4096, // Level 2: ~11 peaks/sec at 44.1kHz (全局级，预览/导航)
];

/// 级别选择的 spp (samples_per_pixel) 阈值
/// spp ≤ SPP_THRESHOLDS[0] → L0
/// SPP_THRESHOLDS[0] < spp ≤ SPP_THRESHOLDS[1] → L1
/// spp > SPP_THRESHOLDS[1] → L2
pub const SPP_THRESHOLDS: [f64; 2] = [256.0, 2048.0];

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

/// 单个 Mipmap 级别的峰值数据
#[derive(Debug, Clone)]
pub struct MipmapData {
    /// 最小值数组 (interleaved by channel: [ch0_min, ch1_min, ...] per peak)
    pub min: Vec<f32>,
    /// 最大值数组
    pub max: Vec<f32>,
}

impl MipmapData {
    /// 创建空的峰值数据
    pub fn new() -> Self {
        Self {
            min: Vec::new(),
            max: Vec::new(),
        }
    }

    /// 创建指定容量的峰值数据
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            min: Vec::with_capacity(capacity),
            max: Vec::with_capacity(capacity),
        }
    }

    /// 获取峰值数量
    pub fn len(&self) -> usize {
        self.min.len().min(self.max.len())
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 计算数据大小 (bytes)
    pub fn data_size(&self, channels: u16) -> usize {
        self.len() * channels as usize * 8 // min(f32) + max(f32) = 8 bytes per channel per peak
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
        let total_values = peak_count * channels as usize;

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

        Ok(Self { min, max })
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
    pub fn select_mipmap_level(&self, samples_per_pixel: f64) -> usize {
        // 根据 spp 阈值选择最佳 mipmap 级别
        // spp ≤ 256 → L0 (div=64, 精细级)
        // 256 < spp ≤ 2048 → L1 (div=512, 中间级)
        // spp > 2048 → L2 (div=4096, 全局级)
        let max_level = self.mipmap_headers.len().saturating_sub(1);

        if samples_per_pixel <= SPP_THRESHOLDS[0] {
            0
        } else if samples_per_pixel <= SPP_THRESHOLDS[1] {
            1.min(max_level)
        } else {
            2.min(max_level)
        }
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
        let count = data.min.len();
        let mut buf = Vec::with_capacity(20 + count * 8);

        // Header (20 bytes)
        buf.extend_from_slice(b"WFPK");
        buf.extend_from_slice(&self.header.sample_rate.to_le_bytes());
        buf.extend_from_slice(&mh.division_factor.to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        buf.extend_from_slice(&(level as u32).to_le_bytes());

        // min data
        for &v in &data.min {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // max data
        for &v in &data.max {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf
    }

    /// 获取指定级别的峰值数据（可选时间范围裁剪）
    ///
    /// # 参数
    /// - `level`: mipmap 级别
    /// - `start_sec`: 开始时间（秒）
    /// - `duration_sec`: 持续时间（秒）
    /// - `columns`: 输出列数
    ///
    /// # 返回
    /// 裁剪后的峰值数据
    pub fn get_peaks_segment(
        &self,
        level: usize,
        start_sec: f64,
        duration_sec: f64,
        columns: usize,
    ) -> PeaksSegmentResult {
        if level >= self.mipmap_data.len() {
            return PeaksSegmentResult {
                ok: false,
                min: vec![],
                max: vec![],
                level: 0,
                sample_rate: 0,
                division_factor: 0,
                actual_start_sec: 0.0,
                actual_duration_sec: 0.0,
            };
        }

        let data = &self.mipmap_data[level];
        let mh = &self.mipmap_headers[level];

        if columns == 0 || !start_sec.is_finite() || !duration_sec.is_finite() {
            return PeaksSegmentResult {
                ok: false,
                min: vec![],
                max: vec![],
                level: level as u32,
                sample_rate: self.header.sample_rate,
                division_factor: mh.division_factor,
                actual_start_sec: 0.0,
                actual_duration_sec: 0.0,
            };
        }

        let sr = self.header.sample_rate.max(1) as f64;
        let division_factor = mh.division_factor.max(1) as f64;

        // 计算峰值索引范围
        let start_frame = (start_sec.max(0.0) * sr).floor() as i64;
        let frames = (duration_sec.max(0.0) * sr).ceil() as i64;

        if frames <= 0 {
            return PeaksSegmentResult {
                ok: true,
                min: vec![0.0; columns],
                max: vec![0.0; columns],
                level: level as u32,
                sample_rate: self.header.sample_rate,
                division_factor: mh.division_factor,
                actual_start_sec: start_sec.max(0.0),
                actual_duration_sec: 0.0,
            };
        }

        // 每个峰值代表 division_factor 个采样
        let start_peak = ((start_frame as f64) / division_factor).floor() as i64;
        let end_peak = (((start_frame + frames) as f64) / division_factor).ceil() as i64;

        let len = data.len() as i64;
        let i0 = start_peak.max(0).min(len);
        let i1 = end_peak.max(0).min(len);

        // 计算实际覆盖的时间范围（由 floor/ceil 取整后的峰值索引决定）
        let actual_start_sec = (i0 as f64 * division_factor) / sr;
        let actual_duration_sec = ((i1 - i0) as f64 * division_factor) / sr;

        if i1 <= i0 {
            return PeaksSegmentResult {
                ok: true,
                min: vec![0.0; columns],
                max: vec![0.0; columns],
                level: level as u32,
                sample_rate: self.header.sample_rate,
                division_factor: mh.division_factor,
                actual_start_sec,
                actual_duration_sec: 0.0,
            };
        }

        // 将峰值映射到输出列
        let span = (i1 - i0).max(1) as f64;
        let mut out_min = vec![f32::INFINITY; columns];
        let mut out_max = vec![f32::NEG_INFINITY; columns];

        for idx in i0..i1 {
            let rel = (idx - i0) as f64;
            let x = ((rel * columns as f64) / span).floor() as usize;
            if x >= columns {
                continue;
            }

            let mi = data.min[idx as usize];
            let ma = data.max[idx as usize];

            if mi < out_min[x] {
                out_min[x] = mi;
            }
            if ma > out_max[x] {
                out_max[x] = ma;
            }
        }

        // 处理无效值
        for i in 0..columns {
            if !out_min[i].is_finite() {
                out_min[i] = 0.0;
            }
            if !out_max[i].is_finite() {
                out_max[i] = 0.0;
            }
        }

        PeaksSegmentResult {
            ok: true,
            min: out_min,
            max: out_max,
            level: level as u32,
            sample_rate: self.header.sample_rate,
            division_factor: mh.division_factor,
            actual_start_sec,
            actual_duration_sec,
        }
    }
}

// ============== API 响应结构 ==============

/// 峰值片段查询结果
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PeaksSegmentResult {
    /// 是否成功
    pub ok: bool,
    /// 最小值数组
    pub min: Vec<f32>,
    /// 最大值数组
    pub max: Vec<f32>,
    /// 使用的 mipmap 级别
    #[serde(rename = "mipmap_level")]
    pub level: u32,
    /// 采样率
    pub sample_rate: u32,
    /// 除数因子：每个峰值代表的采样数
    pub division_factor: u32,
    /// 返回数据实际覆盖的起始时间（秒），由 floor/ceil 取整后的峰值索引决定
    pub actual_start_sec: f64,
    /// 返回数据实际覆盖的持续时间（秒），由 floor/ceil 取整后的峰值索引决定
    pub actual_duration_sec: f64,
}

/// 多级峰值查询响应
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PeaksResponse {
    /// 是否成功
    pub ok: bool,
    /// 峰值数据
    pub peaks: PeaksSegmentResult,
    /// 采样率
    pub sample_rate: u32,
    /// 总时长（秒）
    pub duration_sec: f64,
    /// 可用的 mipmap 级别数
    pub mipmap_levels: u32,
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

/// 多级 Mipmap 峰值计算器
pub struct MipmapPeakCalculator {
    /// 采样率
    sample_rate: u32,
    /// 声道数
    channels: u16,
    /// 总帧数
    total_frames: u64,
    /// 各级别的除数因子
    division_factors: Vec<u32>,
    /// 各级别的累积器 (min, max, frame_count)
    accumulators: Vec<(f32, f32, usize)>,
}

impl MipmapPeakCalculator {
    /// 创建新的计算器
    pub fn new(sample_rate: u32, channels: u16, total_frames: u64) -> Self {
        let division_factors = calculate_division_factors(sample_rate);
        let accumulators = division_factors
            .iter()
            .map(|_| (f32::INFINITY, f32::NEG_INFINITY, 0usize))
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
    /// - `frame_values`: 各声道的采样值 (取极值后传入)
    /// - `frame_min`: 该帧的最小值
    /// - `frame_max`: 该帧的最大值
    /// - `output_callback`: 当某级别完成一个峰值时调用
    pub fn process_frame<F: FnMut(usize, f32, f32)>(
        &mut self,
        frame_min: f32,
        frame_max: f32,
        output_callback: &mut F,
    ) {
        for (level_idx, (acc_min, acc_max, frame_count)) in self.accumulators.iter_mut().enumerate()
        {
            // 更新累积器
            if frame_min < *acc_min {
                *acc_min = frame_min;
            }
            if frame_max > *acc_max {
                *acc_max = frame_max;
            }
            *frame_count += 1;

            // 检查是否需要输出
            let divisor = self.division_factors[level_idx] as usize;
            if *frame_count >= divisor {
                output_callback(
                    level_idx,
                    if acc_min.is_finite() { *acc_min } else { 0.0 },
                    if acc_max.is_finite() { *acc_max } else { 0.0 },
                );

                // 重置累积器
                *acc_min = f32::INFINITY;
                *acc_max = f32::NEG_INFINITY;
                *frame_count = 0;
            }
        }
    }

    /// 刷新剩余的累积数据
    pub fn flush<F: FnMut(usize, f32, f32)>(&mut self, mut output_callback: F) {
        for (level_idx, (acc_min, acc_max, frame_count)) in self.accumulators.iter_mut().enumerate()
        {
            if *frame_count > 0 {
                output_callback(
                    level_idx,
                    if acc_min.is_finite() { *acc_min } else { 0.0 },
                    if acc_max.is_finite() { *acc_max } else { 0.0 },
                );

                // 重置
                *acc_min = f32::INFINITY;
                *acc_max = f32::NEG_INFINITY;
                *frame_count = 0;
            }
        }
    }
}

/// 从音频文件计算多级 mipmap 峰值
///
/// 支持 WAV (通过 hound) 和其他格式 (通过 symphonia)
///
/// # 参数
/// - `progress_cb`: 可选的进度回调，参数为 0.0~1.0 的进度值
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

    // 回退到 symphonia
    compute_mipmap_peaks_symphonia(path, source_file_size, source_modified_ns, &mut progress_cb)
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

    // 初始化输出缓冲区
    let division_factors = calculate_division_factors(sample_rate);
    let mut output_buffers: Vec<Vec<(f32, f32)>> =
        division_factors.iter().map(|_| Vec::new()).collect();

    // 创建计算器
    let mut calculator = MipmapPeakCalculator::new(sample_rate, channels, total_frames);

    // 重新打开文件读取采样数据
    let mut reader = WavReader::open(path).map_err(|e| e.to_string())?;

    // 输出回调
    let mut output_callback = |level: usize, min: f32, max: f32| {
        if level < output_buffers.len() {
            output_buffers[level].push((min, max));
        }
    };

    // 进度跟踪
    let mut frames_processed: u64 = 0;
    let progress_interval = (total_frames / 20).max(1); // 每 ~5% 报告一次

    // 根据格式处理采样
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let mut buf = vec![0i16; channels as usize];
            let mut i = 0usize;
            for s in reader.samples::<i16>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels as usize {
                    i = 0;
                    let (ch_min, ch_max) = compute_channel_extremes_i16(&buf);
                    calculator.process_frame(ch_min, ch_max, &mut output_callback);
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
            let mut buf = vec![0i32; channels as usize];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels as usize {
                    i = 0;
                    let (ch_min, ch_max) = compute_channel_extremes_i32(&buf, denom);
                    calculator.process_frame(ch_min, ch_max, &mut output_callback);
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
            let mut buf = vec![0i32; channels as usize];
            let mut i = 0usize;
            for s in reader.samples::<i32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels as usize {
                    i = 0;
                    let (ch_min, ch_max) = compute_channel_extremes_i32(&buf, i32::MAX as f32);
                    calculator.process_frame(ch_min, ch_max, &mut output_callback);
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
            let mut buf = vec![0f32; channels as usize];
            let mut i = 0usize;
            for s in reader.samples::<f32>() {
                buf[i] = s.map_err(|e| e.to_string())?;
                i += 1;
                if i >= channels as usize {
                    i = 0;
                    let (ch_min, ch_max) = compute_channel_extremes_f32(&buf);
                    calculator.process_frame(ch_min, ch_max, &mut output_callback);
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
        let data = MipmapData {
            min: buffer.iter().map(|(min, _)| *min).collect(),
            max: buffer.iter().map(|(_, max)| *max).collect(),
        };
        file.add_mipmap(division_factors[level_idx], data);
    }

    Ok(file)
}

/// 使用 symphonia 计算其他格式文件的多级峰值
fn compute_mipmap_peaks_symphonia<F: FnMut(f32)>(
    path: &Path,
    source_file_size: u64,
    source_modified_ns: u64,
    progress_cb: &mut Option<F>,
) -> Result<HfsPeakFile, String> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    // 估算总帧数（可能不精确）
    let total_frames = track.codec_params.n_frames.unwrap_or(0);

    // 初始化输出缓冲区
    let division_factors = calculate_division_factors(sample_rate);
    let mut output_buffers: Vec<Vec<(f32, f32)>> =
        division_factors.iter().map(|_| Vec::new()).collect();

    // 创建计算器
    let mut calculator = MipmapPeakCalculator::new(sample_rate, channels, total_frames);

    // 输出回调
    let mut output_callback = |level: usize, min: f32, max: f32| {
        if level < output_buffers.len() {
            output_buffers[level].push((min, max));
        }
    };

    // 进度跟踪
    let mut frames_processed: u64 = 0;
    let progress_interval = if total_frames > 0 {
        (total_frames / 20).max(1)
    } else {
        44100
    }; // symphonia 可能没有精确 total_frames

    // 解码循环
    let track_id = track.id;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(e) => return Err(e.to_string()),
        };

        // 使用 SampleBuffer 转换为 f32 interleaved
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sbuf = symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);
        sbuf.copy_interleaved_ref(decoded);
        let samples = sbuf.samples();

        // 处理帧
        let frames = samples.len() / channels as usize;
        for f in 0..frames {
            let base = f * channels as usize;
            let mut ch_min = f32::INFINITY;
            let mut ch_max = f32::NEG_INFINITY;
            for ch in 0..channels as usize {
                let v = samples.get(base + ch).copied().unwrap_or(0.0);
                if v < ch_min {
                    ch_min = v;
                }
                if v > ch_max {
                    ch_max = v;
                }
            }
            calculator.process_frame(ch_min, ch_max, &mut output_callback);
            frames_processed += 1;
            if frames_processed % progress_interval == 0 {
                if let Some(cb) = progress_cb.as_mut() {
                    if total_frames > 0 {
                        cb(frames_processed as f32 / total_frames as f32);
                    } else {
                        // total_frames 未知时，基于文件大小估算
                        let estimated_total = source_file_size / ((channels as u64) * 4).max(1);
                        cb((frames_processed as f32 / estimated_total as f32).min(0.99));
                    }
                }
            }
        }
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
        let data = MipmapData {
            min: buffer.iter().map(|(min, _)| *min).collect(),
            max: buffer.iter().map(|(_, max)| *max).collect(),
        };
        file.add_mipmap(division_factors[level_idx], data);
    }

    Ok(file)
}

// ============== 辅助函数 ==============

/// 计算 i16 缓冲区的声道极值
fn compute_channel_extremes_i16(buf: &[i16]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in buf {
        let v = x as f32 / i16::MAX as f32;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// 计算 i32 缓冲区的声道极值
fn compute_channel_extremes_i32(buf: &[i32], denom: f32) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in buf {
        let v = x as f32 / denom;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// 计算 f32 缓冲区的声道极值
fn compute_channel_extremes_f32(buf: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in buf {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    (min, max)
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
    pub fn get_or_compute(&self, source_path: &Path) -> Result<HfsPeakFile, String> {
        // 尝试从缓存加载
        if let Some(cached) = self.try_load(source_path) {
            return Ok(cached);
        }

        // 计算新的峰值数据
        let peaks = compute_mipmap_peaks(source_path)?;

        // 保存到缓存
        if let Err(e) = self.save(source_path, &peaks) {
            eprintln!("Warning: failed to save peaks cache: {}", e);
        }

        Ok(peaks)
    }

    /// 清理缓存目录
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

    /// 尝试从旧版缓存迁移
    ///
    /// 查找旧版缓存文件，如果存在则迁移到 v2 格式
    pub fn try_migrate_from_v1(
        &self,
        source_path: &Path,
        old_cache_path: &Path,
    ) -> Option<HfsPeakFile> {
        // 尝试加载旧格式
        let legacy = try_load_v1_format(old_cache_path)?;

        // 迁移到 v2 格式
        let v2 = migrate_v1_to_v2(legacy, source_path);

        // 保存 v2 格式
        if let Err(e) = self.save(source_path, &v2) {
            eprintln!("Warning: failed to save migrated v2 cache: {}", e);
        } else {
            // 删除旧版缓存文件
            let _ = std::fs::remove_file(old_cache_path);
        }

        Some(v2)
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

// ============== 旧格式兼容与迁移 ==============

/// 旧版 HFSPEAKS v1 格式常量
const V1_MAGIC: &[u8; 8] = b"HFSPEAKS";
const V1_VERSION: u32 = 1;

/// 旧版缓存数据结构（兼容 v1 格式）
pub struct LegacyCachedPeaks {
    pub sample_rate: u32,
    pub hop: usize,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub total_frames: u64,
}

/// 尝试从旧版 v1 格式加载
///
/// v1 格式结构：
/// - MAGIC: 8 bytes "HFSPEAKS"
/// - VERSION: 4 bytes u32 = 1
/// - sample_rate: 4 bytes u32
/// - hop: 4 bytes u32
/// - total_frames: 8 bytes u64
/// - len: 4 bytes u32
/// - min[]: len * 4 bytes f32
/// - max[]: len * 4 bytes f32
pub fn try_load_v1_format(path: &Path) -> Option<LegacyCachedPeaks> {
    use std::io::Read;

    let mut f = File::open(path).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;

    let mut off: usize = 0;
    let mut take = |n: usize| -> Option<&[u8]> {
        if off + n > buf.len() {
            return None;
        }
        let s = &buf[off..off + n];
        off += n;
        Some(s)
    };

    // 验证 MAGIC
    if take(8)? != V1_MAGIC {
        return None;
    }

    // 验证 VERSION
    let ver = u32::from_le_bytes(take(4)?.try_into().ok()?);
    if ver != V1_VERSION {
        return None;
    }

    let sample_rate = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let hop = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let total_frames = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;

    if len == 0 || len > 10_000_000 {
        return None;
    }

    let mut read_f32_vec = |count: usize| -> Option<Vec<f32>> {
        let bytes = take(count * 4)?;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let j = i * 4;
            let v = f32::from_le_bytes(bytes[j..j + 4].try_into().ok()?);
            out.push(v);
        }
        Some(out)
    };

    let min = read_f32_vec(len)?;
    let max = read_f32_vec(len)?;

    Some(LegacyCachedPeaks {
        sample_rate,
        hop,
        min,
        max,
        total_frames,
    })
}

/// 将旧版 v1 格式迁移到 v2 多级 mipmap 格式
///
/// 由于 v1 只有一级峰值，迁移策略：
/// - Level 0: 直接使用 v1 数据（如果 hop 接近 128）
/// - Level 1-3: 从 Level 0 降采样生成
pub fn migrate_v1_to_v2(legacy: LegacyCachedPeaks, source_path: &Path) -> HfsPeakFile {
    let (source_file_size, source_modified_ns) = get_metadata_fingerprint(source_path);

    // 确定最佳 mipmap 分配
    // v1 的 hop 通常是根据请求的 columns 动态计算的
    // 我们需要根据实际 hop 值决定如何分配

    let sample_rate = legacy.sample_rate;
    let total_frames = legacy.total_frames;
    let channels = 1u16; // v1 不存储声道数，假设为单声道

    // 构建 mipmap 数据
    let mut mipmap_headers = Vec::new();
    let mut mipmap_data = Vec::new();
    let mut data_offset: u64 = (HfsPeakHeader::SIZE + 4 * MipmapHeader::SIZE) as u64; // 4 级 mipmap

    // Level 0: 使用原始数据或插值
    let level0_div = DEFAULT_DIVISION_FACTORS[0];
    let level0_min = legacy.min.clone();
    let level0_max = legacy.max.clone();

    let level0_count = level0_min.len() as u32;
    mipmap_headers.push(MipmapHeader {
        division_factor: level0_div,
        peak_count: level0_count,
        data_offset,
    });
    mipmap_data.push(MipmapData {
        min: level0_min,
        max: level0_max,
    });
    data_offset += level0_count as u64 * 8; // 每个 min/max 对 8 bytes

    // Level 1-3: 降采样生成
    for level in 1..4 {
        let div_factor = DEFAULT_DIVISION_FACTORS[level];
        let prev_data = &mipmap_data[level - 1];
        let downsampled = downsample_mipmap(&prev_data.min, &prev_data.max, 4);

        // 计算实际的 peak count
        let peak_count = downsampled.0.len() as u32;

        mipmap_headers.push(MipmapHeader {
            division_factor: div_factor,
            peak_count,
            data_offset,
        });
        mipmap_data.push(MipmapData {
            min: downsampled.0,
            max: downsampled.1,
        });
        data_offset += peak_count as u64 * 8;
    }

    // 构建文件头
    let header = HfsPeakHeader {
        magic: *MAGIC,
        version: VERSION,
        channels,
        sample_rate,
        total_frames,
        source_file_size,
        source_modified_ns,
        mipmap_count: 4,
        reserved: [0; 8],
    };

    HfsPeakFile {
        header,
        mipmap_headers,
        mipmap_data,
    }
}

/// 降采样 mipmap 数据
fn downsample_mipmap(min: &[f32], max: &[f32], factor: usize) -> (Vec<f32>, Vec<f32>) {
    if factor <= 1 || min.is_empty() {
        return (min.to_vec(), max.to_vec());
    }

    let n = min.len();
    let mut result_min = Vec::new();
    let mut result_max = Vec::new();

    for i in (0..n).step_by(factor) {
        let end = (i + factor).min(n);
        let mut w_min = f32::INFINITY;
        let mut w_max = f32::NEG_INFINITY;

        for j in i..end {
            let min_val = min[j];
            let max_val = max[j];
            if min_val < w_min {
                w_min = min_val;
            }
            if max_val > w_max {
                w_max = max_val;
            }
        }

        result_min.push(if w_min.is_infinite() { 0.0 } else { w_min });
        result_max.push(if w_max.is_infinite() { 0.0 } else { w_max });
    }

    (result_min, result_max)
}
