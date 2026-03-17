//! 流式 WORLD 声码器接口
//!
//! 提供两个核心结构体：
//! - [`StreamingWorldAnalyzer`]：将 PCM 样本流式送入，输出 F0 + 频谱帧序列。
//!   内部维护重叠缓冲区，每次 `push_samples` 后可通过 `pull_frames` 取出已就绪的帧。
//! - [`StreamingWorldSynthesizer`]：封装 WORLD 的 `WorldSynthesizer` C API，
//!   通过 `push_frames` 送入参数帧，通过 `pull_samples` 取出合成 PCM。
//!
//! 两者组合可替代 `vocode_one` 中的批量 `Synthesis` 调用，
//! 实现低延迟、低内存占用的流式音高变换。

use crate::world_vocoder::{
    AddParameters, CheapTrick, CheapTrickOption, D4COption, DestroySynthesizer,
    GetFFTSizeForCheapTrick, GetSamplesForHarvest, Harvest, HarvestOption,
    InitializeCheapTrickOption, InitializeD4COption, InitializeHarvestOption,
    InitializeSynthesizer, IsLocked, Synthesis2, WorldSynthesizerRaw, D4C,
};

// ─── StreamingWorldAnalyzer ────────────────────────────────────────────────────

/// 流式 WORLD 分析器。
///
/// 工作原理：
/// 1. 调用方通过 [`push_samples`] 将 PCM 样本（f64，单声道）送入内部缓冲区。
/// 2. 当缓冲区积累了足够的样本（≥ `chunk_samples`）时，
///    [`pull_frames`] 会触发一次 Harvest + CheapTrick + D4C 分析，
///    返回新产生的帧（F0、频谱、非周期性）。
/// 3. 分析窗口之间保留 `overlap_samples` 的重叠，以保证帧连续性。
///
/// # 线程安全
/// 该结构体**不是** `Send`/`Sync`，调用方需在单一线程中使用。
pub struct StreamingWorldAnalyzer {
    /// 采样率（Hz）
    fs: i32,
    /// 帧周期（毫秒）
    frame_period_ms: f64,
    /// F0 下限（Hz）
    f0_floor: f64,
    /// F0 上限（Hz）
    f0_ceil: f64,
    /// FFT 大小（由 CheapTrick 决定）
    fft_size: i32,
    /// 每次分析的块大小（样本数）
    chunk_samples: usize,
    /// 块间重叠（样本数），保证帧边界连续
    overlap_samples: usize,
    /// 内部 PCM 缓冲区
    buffer: Vec<f64>,
    /// 已处理的累计样本数（用于计算绝对时间）
    processed_samples: usize,
    /// 已输出的累计帧数
    output_frames: usize,
}

/// 一批分析帧的输出。
pub struct AnalysisFrames {
    /// F0 曲线（Hz，0 表示无声）
    pub f0: Vec<f64>,
    /// 时间轴（秒）
    pub temporal_positions: Vec<f64>,
    /// 频谱包络，每帧 `fft_size/2 + 1` 个 bin
    pub spectrogram: Vec<Vec<f64>>,
    /// 非周期性，每帧 `fft_size/2 + 1` 个 bin
    pub aperiodicity: Vec<Vec<f64>>,
    /// FFT 大小
    pub fft_size: i32,
    /// 帧周期（毫秒）
    pub frame_period_ms: f64,
}

impl StreamingWorldAnalyzer {
    /// 创建分析器。
    ///
    /// - `chunk_sec`：每次分析的块长度（秒），建议 1.0～4.0
    /// - `overlap_sec`：块间重叠（秒），建议 0.05～0.1
    pub fn new(
        fs: u32,
        frame_period_ms: f64,
        f0_floor: f64,
        f0_ceil: f64,
        chunk_sec: f64,
        overlap_sec: f64,
    ) -> Self {
        let fs_i = fs as i32;
        // 先用默认选项获取 fft_size
        let mut ct_opt = CheapTrickOption {
            q1: -0.15,
            f0_floor: f0_floor.max(20.0),
            fft_size: 0,
        };
        unsafe { InitializeCheapTrickOption(fs_i, &mut ct_opt) };
        ct_opt.f0_floor = f0_floor.max(20.0);
        let fft_size = unsafe { GetFFTSizeForCheapTrick(fs_i, &ct_opt) };

        let chunk_samples = ((chunk_sec * fs as f64).round() as usize).max(1024);
        let overlap_samples = ((overlap_sec * fs as f64).round() as usize).max(64);

        Self {
            fs: fs_i,
            frame_period_ms,
            f0_floor,
            f0_ceil,
            fft_size,
            chunk_samples,
            overlap_samples,
            buffer: Vec::new(),
            processed_samples: 0,
            output_frames: 0,
        }
    }

    /// 送入新的 PCM 样本（f64，单声道）。
    pub fn push_samples(&mut self, samples: &[f64]) {
        self.buffer.extend_from_slice(samples);
    }

    /// 取出已就绪的分析帧。
    ///
    /// 当缓冲区样本数 ≥ `chunk_samples + overlap_samples` 时触发分析；
    /// 否则返回 `None`（需要更多输入）。
    ///
    /// 调用方应循环调用直到返回 `None`。
    pub fn pull_frames(&mut self) -> Option<AnalysisFrames> {
        let needed = self.chunk_samples + self.overlap_samples;
        if self.buffer.len() < needed {
            return None;
        }

        // 取出当前块（含前向重叠）
        let chunk = &self.buffer[..needed];
        let x_len = chunk.len() as i32;

        // Harvest F0 检测
        let fp = self.frame_period_ms;
        let n_frames = unsafe { GetSamplesForHarvest(self.fs, x_len, fp) } as usize;
        if n_frames == 0 {
            // 消耗掉 chunk_samples，保留重叠
            self.advance_buffer();
            return None;
        }

        let mut harvest_opt = HarvestOption {
            f0_floor: self.f0_floor,
            f0_ceil: self.f0_ceil,
            frame_period: fp,
        };
        unsafe { InitializeHarvestOption(&mut harvest_opt) };
        harvest_opt.f0_floor = self.f0_floor;
        harvest_opt.f0_ceil = self.f0_ceil;
        harvest_opt.frame_period = fp;

        let mut temporal_positions = vec![0.0f64; n_frames];
        let mut f0 = vec![0.0f64; n_frames];

        unsafe {
            Harvest(
                chunk.as_ptr(),
                x_len,
                self.fs,
                &harvest_opt,
                temporal_positions.as_mut_ptr(),
                f0.as_mut_ptr(),
            );
        }

        // 将时间轴偏移到绝对时间（减去前向重叠对应的时间）
        let overlap_sec = self.overlap_samples as f64 / self.fs as f64;
        let abs_offset = self.processed_samples as f64 / self.fs as f64 - overlap_sec;
        for t in temporal_positions.iter_mut() {
            *t += abs_offset.max(0.0);
        }

        // CheapTrick 频谱分析
        let spec_bins = (self.fft_size as usize / 2) + 1;
        let mut spectrogram: Vec<Vec<f64>> = vec![vec![0.0f64; spec_bins]; n_frames];
        let mut sp_ptrs: Vec<*mut f64> = spectrogram.iter_mut().map(|r| r.as_mut_ptr()).collect();

        let mut ct_opt = CheapTrickOption {
            q1: -0.15,
            f0_floor: self.f0_floor.max(20.0),
            fft_size: self.fft_size,
        };
        unsafe { InitializeCheapTrickOption(self.fs, &mut ct_opt) };
        ct_opt.f0_floor = self.f0_floor.max(20.0);
        ct_opt.fft_size = self.fft_size;

        // CheapTrick 需要原始时间轴（相对于块起始）
        let mut local_tp: Vec<f64> = temporal_positions
            .iter()
            .map(|&t| (t - abs_offset.max(0.0)).max(0.0))
            .collect();

        unsafe {
            CheapTrick(
                chunk.as_ptr(),
                x_len,
                self.fs,
                local_tp.as_mut_ptr(),
                f0.as_ptr(),
                n_frames as i32,
                &ct_opt,
                sp_ptrs.as_mut_ptr(),
            );
        }

        // D4C 非周期性分析
        let mut d4c_opt = D4COption { threshold: 0.85 };
        unsafe { InitializeD4COption(&mut d4c_opt) };

        let mut aperiodicity: Vec<Vec<f64>> = vec![vec![0.0f64; spec_bins]; n_frames];
        let mut ap_ptrs: Vec<*mut f64> = aperiodicity.iter_mut().map(|r| r.as_mut_ptr()).collect();

        unsafe {
            D4C(
                chunk.as_ptr(),
                x_len,
                self.fs,
                local_tp.as_mut_ptr(),
                f0.as_ptr(),
                n_frames as i32,
                self.fft_size,
                &d4c_opt,
                ap_ptrs.as_mut_ptr(),
            );
        }

        // 只保留属于当前块（非重叠区）的帧
        // 重叠区的帧已在上一块中输出，跳过以避免重复
        let skip_frames = if self.output_frames == 0 {
            0
        } else {
            // 重叠区对应的帧数
            let overlap_ms = self.overlap_samples as f64 * 1000.0 / self.fs as f64;
            ((overlap_ms / fp).ceil() as usize).min(n_frames)
        };

        let valid_f0 = f0[skip_frames..].to_vec();
        let valid_tp = temporal_positions[skip_frames..].to_vec();
        let valid_sp = spectrogram[skip_frames..].to_vec();
        let valid_ap = aperiodicity[skip_frames..].to_vec();

        self.output_frames += valid_f0.len();
        self.advance_buffer();

        if valid_f0.is_empty() {
            return None;
        }

        Some(AnalysisFrames {
            f0: valid_f0,
            temporal_positions: valid_tp,
            spectrogram: valid_sp,
            aperiodicity: valid_ap,
            fft_size: self.fft_size,
            frame_period_ms: self.frame_period_ms,
        })
    }

    /// 强制刷新剩余缓冲区（流结束时调用）。
    ///
    /// 将缓冲区中剩余的所有样本作为最后一块进行分析。
    pub fn flush(&mut self) -> Option<AnalysisFrames> {
        if self.buffer.is_empty() {
            return None;
        }
        // 临时将 chunk_samples 设为缓冲区大小，触发分析
        let orig_chunk = self.chunk_samples;
        self.chunk_samples = self.buffer.len().saturating_sub(self.overlap_samples);
        if self.chunk_samples == 0 {
            self.chunk_samples = self.buffer.len();
        }
        let result = self.pull_frames();
        self.chunk_samples = orig_chunk;
        result
    }

    /// 返回当前 FFT 大小。
    pub fn fft_size(&self) -> i32 {
        self.fft_size
    }

    /// 消耗 `chunk_samples` 个样本，保留 `overlap_samples` 个重叠样本。
    fn advance_buffer(&mut self) {
        let advance = self.chunk_samples.min(self.buffer.len());
        self.processed_samples += advance;
        self.buffer.drain(..advance);
    }
}

// ─── StreamingWorldSynthesizer ─────────────────────────────────────────────────

/// 流式 WORLD 合成器。
///
/// 封装 WORLD 的 `WorldSynthesizer` C API，提供：
/// - [`push_frames`]：送入一批参数帧（F0、频谱、非周期性）
/// - [`pull_samples`]：取出已合成的 PCM 样本
///
/// 内部使用环形缓冲区（`number_of_pointers` 个槽位），
/// 当环形缓冲区满时 `push_frames` 返回 `false`，调用方应先调用 `pull_samples`。
///
/// # 内存安全
/// `WorldSynthesizerRaw` 通过 `Box` 分配在堆上，避免移动后指针失效。
/// `Drop` 实现会调用 `DestroySynthesizer` 释放 C 侧内存。
///
/// WORLD 的 `AddParameters` 只存储指针，不拷贝数据。因此 `pending_data` 队列
/// 持有每次 `push_frames` 传入的 spectrogram/aperiodicity 数据的所有权，
/// 直到 `Synthesis2` 消费完毕后才释放，防止悬空指针导致 ACCESS_VIOLATION。
pub struct StreamingWorldSynthesizer {
    /// C 侧合成器状态（堆分配，避免移动）
    inner: Box<WorldSynthesizerRaw>,
    /// 每次 Synthesis2 输出的样本数
    buffer_size: usize,
    /// 已合成但尚未被取走的 PCM 样本
    output_buf: Vec<f64>,
    /// 是否已初始化
    initialized: bool,
    /// FFT 大小（用于读取 buffer 偏移）
    fft_size: usize,
    /// 持有已送入 AddParameters 的帧数据所有权，防止悬空指针。
    /// 每次 push_frames 成功后追加一条，pull_samples 消费完毕后清理。
    /// 元素：(f0, spectrogram, aperiodicity, sp_ptrs, ap_ptrs)
    /// 注意：sp_ptrs/ap_ptrs 是 WORLD 内部 synth->spectrogram[i] 所指向的指针数组，
    /// 必须与数据一起存活，否则 Synthesis2 解引用时会产生 ACCESS_VIOLATION。
    pending_data: std::collections::VecDeque<(
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<*mut f64>,
        Vec<*mut f64>,
    )>,
    /// 环形缓冲区槽位数（用于判断何时可以安全释放旧数据）
    number_of_pointers: usize,
}

impl StreamingWorldSynthesizer {
    /// 创建合成器。
    ///
    /// - `buffer_size`：每次 `Synthesis2` 输出的样本数，建议 512～2048
    /// - `number_of_pointers`：环形缓冲区槽位数，建议 8～32
    pub fn new(
        fs: u32,
        frame_period_ms: f64,
        fft_size: i32,
        buffer_size: usize,
        number_of_pointers: usize,
    ) -> Self {
        // 零初始化，避免未定义行为
        let mut inner: Box<WorldSynthesizerRaw> = unsafe { Box::new(std::mem::zeroed()) };

        unsafe {
            InitializeSynthesizer(
                fs as i32,
                frame_period_ms,
                fft_size,
                buffer_size as i32,
                number_of_pointers as i32,
                inner.as_mut(),
            );
        }

        Self {
            inner,
            buffer_size,
            output_buf: Vec::new(),
            initialized: true,
            fft_size: fft_size as usize,
            pending_data: std::collections::VecDeque::new(),
            number_of_pointers,
        }
    }

    /// 送入一批参数帧。
    ///
    /// - `f0`：F0 曲线（Hz，0 表示无声）
    /// - `spectrogram`：频谱包络，每帧 `fft_size/2 + 1` 个 bin
    /// - `aperiodicity`：非周期性，每帧 `fft_size/2 + 1` 个 bin
    ///
    /// 返回 `true` 表示成功加入队列，`false` 表示环形缓冲区已满（需先 `pull_samples`）。
    ///
    /// 注意：数据所有权会被转移到内部队列，直到 `Synthesis2` 消费完毕后才释放。
    pub fn push_frames(
        &mut self,
        f0: Vec<f64>,
        spectrogram: Vec<Vec<f64>>,
        aperiodicity: Vec<Vec<f64>>,
    ) -> bool {
        if !self.initialized {
            return false;
        }
        // 将数据移入内部存储，确保在 Synthesis2 消费前数据不被释放。
        // sp_ptrs/ap_ptrs 先用空 Vec 占位，后面填充。
        self.pending_data
            .push_back((f0, spectrogram, aperiodicity, Vec::new(), Vec::new()));
        let entry = self.pending_data.back_mut().unwrap();

        let f0_len = entry.0.len() as i32;
        // 构建指针数组，并存入 entry，确保其生命周期与数据一致。
        // WORLD 的 AddParameters 只存储 sp_ptrs/ap_ptrs 的指针，不拷贝数据，
        // 因此这两个 Vec 必须在 Synthesis2 消费完毕前保持存活。
        entry.3 = entry.1.iter_mut().map(|r| r.as_mut_ptr()).collect();
        entry.4 = entry.2.iter_mut().map(|r| r.as_mut_ptr()).collect();

        let ok = unsafe {
            AddParameters(
                entry.0.as_mut_ptr(),
                f0_len,
                entry.3.as_mut_ptr(),
                entry.4.as_mut_ptr(),
                self.inner.as_mut(),
            )
        };

        if ok == 0 {
            // AddParameters 失败（环形缓冲区满），撤回刚才存入的数据
            self.pending_data.pop_back();
            return false;
        }
        true
    }

    /// 取出已合成的 PCM 样本（f64，单声道）。
    ///
    /// 内部循环调用 `Synthesis2` 直到无法继续合成，
    /// 将所有已合成样本追加到 `output_buf` 后返回。
    ///
    /// 调用方可以在每次 `push_frames` 后调用此函数取走样本。
    pub fn pull_samples(&mut self) -> Vec<f64> {
        if !self.initialized {
            return vec![];
        }

        let mut synthesis_count = 0usize;

        loop {
            // IsLocked 返回 1 表示环形缓冲区既满又无法合成，需要 refresh
            if unsafe { IsLocked(self.inner.as_mut()) } != 0 {
                break;
            }

            let ok = unsafe { Synthesis2(self.inner.as_mut()) };
            if ok == 0 {
                break;
            }
            synthesis_count += 1;

            let buffer_ptr = unsafe { get_synth_buffer(self.inner.as_mut()) };

            if buffer_ptr.is_null() {
                break;
            }

            let samples: Vec<f64> =
                unsafe { std::slice::from_raw_parts(buffer_ptr, self.buffer_size).to_vec() };
            self.output_buf.extend_from_slice(&samples);
        }

        // 每次 Synthesis2 消费约 buffer_size 个样本，对应若干帧。
        // 当 pending_data 队列长度超过 number_of_pointers 时，
        // 说明最早的数据已经被环形缓冲区覆盖，可以安全释放。
        // 保守策略：保留最近 number_of_pointers 条数据，释放更早的。
        while self.pending_data.len() > self.number_of_pointers {
            self.pending_data.pop_front();
        }

        std::mem::take(&mut self.output_buf)
    }

    /// 检查合成器是否被锁定（环形缓冲区满且无法合成）。
    pub fn is_locked(&mut self) -> bool {
        unsafe { IsLocked(self.inner.as_mut()) != 0 }
    }
}

impl Drop for StreamingWorldSynthesizer {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { DestroySynthesizer(self.inner.as_mut()) };
            self.initialized = false;
        }
    }
}

// ─── 辅助：获取 WorldSynthesizer::buffer 指针 ─────────────────────────────────
//
// WorldSynthesizer 结构体的 `buffer` 字段是第 6 个字段（0-indexed: 5），
// 布局（synthesisrealtime.h，x86_64 对齐）：
//   offset  0: fs             (i32,  4 bytes)
//   offset  8: frame_period   (f64,  8 bytes)  ← 8 字节对齐
//   offset 16: buffer_size    (i32,  4 bytes)
//   offset 20: number_of_pointers (i32, 4 bytes)
//   offset 24: fft_size       (i32,  4 bytes)
//   offset 28: _pad           (4 bytes)
//   offset 32: buffer         (*mut f64, 8 bytes)  ← 指针
//
// 通过字节偏移读取 buffer 指针，避免依赖 C 侧额外导出函数。
unsafe fn get_synth_buffer(synth: *mut WorldSynthesizerRaw) -> *mut f64 {
    const BUFFER_PTR_OFFSET: usize = 32;
    let base = synth as *const u8;
    let ptr_ptr = base.add(BUFFER_PTR_OFFSET) as *const *mut f64;
    *ptr_ptr
}

// ─── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证：分块输入与批量输入的 F0 输出在有声区域的均值误差 < 5 Hz。
    ///
    /// 使用 440 Hz 正弦波作为测试信号（标准 A4 音）。
    #[test]
    fn test_streaming_analyzer_f0_consistency() {
        let fs = 44100u32;
        let freq = 440.0f64;
        let duration_sec = 2.0f64;
        let n_samples = (fs as f64 * duration_sec) as usize;

        // 生成 440 Hz 正弦波
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs as f64).sin() * 0.8)
            .collect();

        let frame_period_ms = 5.0;
        let f0_floor = 80.0;
        let f0_ceil = 800.0;

        // 批量分析（参考值）
        let batch_f0 = {
            let x_len = signal.len() as i32;
            let mut opt = HarvestOption {
                f0_floor,
                f0_ceil,
                frame_period: frame_period_ms,
            };
            unsafe { InitializeHarvestOption(&mut opt) };
            opt.f0_floor = f0_floor;
            opt.f0_ceil = f0_ceil;
            opt.frame_period = frame_period_ms;

            let n = unsafe { GetSamplesForHarvest(fs as i32, x_len, frame_period_ms) } as usize;
            let mut tp = vec![0.0f64; n];
            let mut f0 = vec![0.0f64; n];
            unsafe {
                Harvest(
                    signal.as_ptr(),
                    x_len,
                    fs as i32,
                    &opt,
                    tp.as_mut_ptr(),
                    f0.as_mut_ptr(),
                );
            }
            f0
        };

        // 流式分析
        let mut analyzer =
            StreamingWorldAnalyzer::new(fs, frame_period_ms, f0_floor, f0_ceil, 1.0, 0.05);
        let mut streaming_f0: Vec<f64> = Vec::new();

        // 分块送入（每次 4096 样本）
        let chunk_size = 4096;
        let mut pos = 0;
        while pos < signal.len() {
            let end = (pos + chunk_size).min(signal.len());
            analyzer.push_samples(&signal[pos..end]);
            while let Some(frames) = analyzer.pull_frames() {
                streaming_f0.extend_from_slice(&frames.f0);
            }
            pos = end;
        }
        if let Some(frames) = analyzer.flush() {
            streaming_f0.extend_from_slice(&frames.f0);
        }

        // 比较有声区域的 F0 均值
        let batch_voiced: Vec<f64> = batch_f0.iter().copied().filter(|&v| v > 0.0).collect();
        let stream_voiced: Vec<f64> = streaming_f0.iter().copied().filter(|&v| v > 0.0).collect();

        assert!(!batch_voiced.is_empty(), "批量分析应检测到有声帧");
        assert!(!stream_voiced.is_empty(), "流式分析应检测到有声帧");

        let batch_mean = batch_voiced.iter().sum::<f64>() / batch_voiced.len() as f64;
        let stream_mean = stream_voiced.iter().sum::<f64>() / stream_voiced.len() as f64;

        let diff = (batch_mean - stream_mean).abs();
        assert!(
            diff < 5.0,
            "F0 均值误差过大：批量={:.2} Hz，流式={:.2} Hz，差值={:.2} Hz",
            batch_mean,
            stream_mean,
            diff
        );
    }
}
