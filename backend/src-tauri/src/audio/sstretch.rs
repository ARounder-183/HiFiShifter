//! Signalsmith Stretch FFI 封装
//!
//! 基于 Signalsmith Stretch (MIT) 的音频时间拉伸模块。
//!
//! 公共接口：
//! - `SignalsmithRealtimeStretcher` — 实时流式拉伸器
//! - `try_time_stretch_interleaved_offline()` — 离线批量拉伸
//! - `try_time_stretch_interleaved_realtime()` — 实时模式批量拉伸
//! - `is_available()` — 始终返回 true（静态链接）

use std::ffi::c_int;

// ── FFI 声明 ──────────────────────────────────────────────────────
// 对应 sstretch-c.h 中的 C API
type SStretchState = *mut std::ffi::c_void;

extern "C" {
    fn sstretch_new(sample_rate: u32, channels: u32) -> SStretchState;
    fn sstretch_delete(state: SStretchState);
    fn sstretch_reset(state: SStretchState);
    fn sstretch_set_transpose_semitones(state: SStretchState, semitones: f64);
    fn sstretch_set_transpose_factor(state: SStretchState, factor: f64);
    fn sstretch_input_latency(state: SStretchState) -> c_int;
    fn sstretch_output_latency(state: SStretchState) -> c_int;

    fn sstretch_process_interleaved(
        state: SStretchState,
        input_interleaved: *const f32,
        in_frames: u32,
        output_interleaved: *mut f32,
        out_frames: u32,
    ) -> c_int;

    fn sstretch_process_offline(
        state: SStretchState,
        input_interleaved: *const f32,
        in_frames: u32,
        output_interleaved: *mut f32,
        out_frames: u32,
        time_ratio: f64,
    ) -> c_int;

    fn sstretch_flush(state: SStretchState, output_interleaved: *mut f32, out_frames: u32)
        -> c_int;
}

// ── 公共 API ──────────────────────────────────────────────────────

/// Signalsmith Stretch 是否可用。
/// 由于静态链接，始终返回 true。
pub fn is_available() -> bool {
    true
}

/// 实时流式拉伸器。
///
/// 用于 `stretch_stream` 后台 worker 的逐块 process + retrieve 流程。
/// 注意：Signalsmith Stretch 不区分 process/retrieve，而是一次 process() 调用
/// 同时消费输入和产出输出。为保持 API 兼容性，我们在内部缓冲输出。
pub struct SignalsmithRealtimeStretcher {
    state: SStretchState,
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,

    /// 内部输出缓冲区（交错格式）
    /// process_interleaved() 产出的数据暂存于此，
    /// retrieve_interleaved_into() 从中取出。
    out_buffer: Vec<f32>,
    temp_out: Vec<f32>,
}

unsafe impl Send for SignalsmithRealtimeStretcher {}

impl SignalsmithRealtimeStretcher {
    pub fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, String> {
        if channels == 0 {
            return Err("signalsmith stretch: channels == 0".to_string());
        }
        if channels > 2 {
            return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
        }

        let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
            time_ratio
        } else {
            1.0
        };

        let state = unsafe { sstretch_new(sample_rate.max(1), channels as u32) };
        if state.is_null() {
            return Err("sstretch_new returned null".to_string());
        }

        // pitch scale = 1.0（不变调），音高偏移为 0 半音
        unsafe {
            sstretch_set_transpose_semitones(state, 0.0);
        }

        eprintln!(
            "[SignalsmithStretch] Created: sample_rate={}, channels={}, time_ratio={:.6}",
            sample_rate, channels, time_ratio
        );

        Ok(Self {
            state,
            channels,
            sample_rate: sample_rate.max(1),
            time_ratio,
            out_buffer: Vec::with_capacity(4096),
            temp_out: Vec::with_capacity(4096),
        })
    }

    pub fn reset(&mut self, time_ratio: f64) -> Result<(), String> {
        let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
            time_ratio
        } else {
            1.0
        };
        self.time_ratio = time_ratio;
        self.out_buffer.clear();
        unsafe {
            sstretch_reset(self.state);
            sstretch_set_transpose_semitones(self.state, 0.0);
        }
        Ok(())
    }

    /// 输入交错 PCM 进行处理。
    ///
    /// Signalsmith Stretch 的 process() 同时消费输入和产出输出。
    /// 输出帧数 = in_frames * time_ratio（向上取整），结果暂存在内部缓冲区。
    pub fn process_interleaved(
        &mut self,
        input_interleaved: &[f32],
        _final: bool,
    ) -> Result<(), String> {
        if input_interleaved.is_empty() {
            return Ok(());
        }
        let in_frames = input_interleaved.len() / self.channels.max(1);
        if in_frames == 0 {
            return Ok(());
        }

        // 根据 time_ratio 计算对应的输出帧数
        let out_frames = ((in_frames as f64) * self.time_ratio).ceil() as usize;
        if out_frames == 0 {
            return Ok(());
        }

        // 复用结构体自带的 buffer，消除高频内存分配
        self.temp_out.resize(out_frames * self.channels, 0.0);

        let ret = unsafe {
            sstretch_process_interleaved(
                self.state,
                input_interleaved.as_ptr(),
                in_frames as u32,
                self.temp_out.as_mut_ptr(),
                out_frames as u32,
            )
        };

        if ret < 0 {
            return Err("sstretch_process_interleaved failed".to_string());
        }

        // 追加到内部缓冲区
        self.out_buffer.extend_from_slice(&self.temp_out);

        Ok(())
    }

    /// 从内部缓冲区取出最多 `max_frames` 帧的交错 PCM。
    pub fn retrieve_interleaved_into(
        &mut self,
        out_interleaved: &mut Vec<f32>,
        max_frames: usize,
    ) -> Result<usize, String> {
        if self.out_buffer.is_empty() || max_frames == 0 {
            return Ok(0);
        }

        let avail_samples = self.out_buffer.len();
        let avail_frames = avail_samples / self.channels.max(1);
        let take_frames = avail_frames.min(max_frames);
        let take_samples = take_frames * self.channels;

        if take_samples == 0 {
            return Ok(0);
        }

        out_interleaved.extend_from_slice(&self.out_buffer[..take_samples]);

        // 移除已取出的数据
        self.out_buffer.drain(..take_samples);

        Ok(take_frames)
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for SignalsmithRealtimeStretcher {
    fn drop(&mut self) {
        if self.state.is_null() {
            return;
        }
        unsafe {
            sstretch_delete(self.state);
        }
        self.state = std::ptr::null_mut();
    }
}

// ── 离线批量拉伸 ──────────────────────────────────────────────────

/// 使用 Signalsmith Stretch 离线模式完成批量时间拉伸。
///
/// 内部自动处理延迟补偿。
/// pitch_scale 保持 1.0（不变调）。
pub fn try_time_stretch_interleaved_offline(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    if input_interleaved.is_empty() || channels == 0 {
        return Ok(vec![]);
    }
    if channels > 2 {
        return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
    }

    let in_frames = input_interleaved.len() / channels;
    if in_frames < 2 {
        return Ok(input_interleaved.to_vec());
    }
    let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
        time_ratio
    } else {
        1.0
    };

    let out_frames = if out_frames_hint > 0 {
        out_frames_hint
    } else {
        ((in_frames as f64) * time_ratio).ceil() as usize
    };

    unsafe {
        let state = sstretch_new(sample_rate.max(1), channels as u32);
        if state.is_null() {
            return Err("sstretch_new returned null".to_string());
        }

        // pitch_scale = 1.0（不变调）
        sstretch_set_transpose_semitones(state, 0.0);

        let mut output = vec![0.0f32; out_frames * channels];

        let got = sstretch_process_offline(
            state,
            input_interleaved.as_ptr(),
            in_frames as u32,
            output.as_mut_ptr(),
            out_frames as u32,
            time_ratio,
        );

        sstretch_delete(state);

        if got < 0 {
            return Err("sstretch_process_offline failed".to_string());
        }

        // 如果实际输出少于期望，截断
        let actual_frames = got as usize;
        if actual_frames < out_frames {
            output.truncate(actual_frames * channels);
        }

        Ok(output)
    }
}

/// 使用 Signalsmith Stretch 实时模式完成批量时间拉伸。
///
/// 与 `try_time_stretch_interleaved_offline` 功能相同，
/// 但使用流式 process 接口（与 stretch_stream 路径一致）。
pub fn try_time_stretch_interleaved_realtime(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    if input_interleaved.is_empty() || channels == 0 {
        return Ok(vec![]);
    }
    if channels > 2 {
        return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
    }

    let in_frames = input_interleaved.len() / channels;
    if in_frames < 2 {
        return Ok(input_interleaved.to_vec());
    }
    let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
        time_ratio
    } else {
        1.0
    };

    let out_frames = if out_frames_hint > 0 {
        out_frames_hint
    } else {
        ((in_frames as f64) * time_ratio).ceil() as usize
    };

    unsafe {
        let state = sstretch_new(sample_rate.max(1), channels as u32);
        if state.is_null() {
            return Err("sstretch_new returned null".to_string());
        }

        sstretch_set_transpose_semitones(state, 0.0);

        // 分块处理
        const BLOCK: usize = 1024;

        let input_latency = sstretch_input_latency(state) as usize;
        let output_latency = sstretch_output_latency(state) as usize;

        // 总共需要的输出帧数（含 pre-roll）
        let total_out = out_frames + output_latency;
        // 总共需要的输入帧数（含尾部 flush 静音）
        let total_in = in_frames + input_latency;

        let mut all_output: Vec<f32> = Vec::with_capacity(total_out * channels);

        let mut in_cursor: usize = 0;
        let mut out_produced: usize = 0;

        // 提前在循环外分配好最大容量的复用 Buffer，消除循环内开销
        let mut in_block = vec![0.0f32; BLOCK * channels];
        let mut out_block = vec![0.0f32; BLOCK * 4 * channels]; // block_out 最大为 BLOCK * 4

        while in_cursor < total_in || out_produced < total_out {
            let remain_in = total_in.saturating_sub(in_cursor);
            let block_in = remain_in.min(BLOCK);

            let block_out = if total_in > 0 {
                let next_progress = ((in_cursor + block_in) as f64) / (total_in as f64);
                let next_expected = (next_progress * total_out as f64).round() as usize;
                next_expected
                    .saturating_sub(out_produced)
                    .max(1)
                    .min(BLOCK * 4)
            } else {
                0
            };

            if block_in == 0 && block_out == 0 {
                break;
            }

            // 利用底层 memcpy 极速拷贝数据，超出的部分补 0
            let valid_in_frames = in_frames.saturating_sub(in_cursor).min(block_in);
            let valid_in_samples = valid_in_frames * channels;
            let total_in_samples = block_in * channels;

            if valid_in_samples > 0 {
                let start = in_cursor * channels;
                in_block[..valid_in_samples]
                    .copy_from_slice(&input_interleaved[start..start + valid_in_samples]);
            }
            if total_in_samples > valid_in_samples {
                in_block[valid_in_samples..total_in_samples].fill(0.0);
            }

            let ret = sstretch_process_interleaved(
                state,
                in_block.as_ptr(),
                block_in as u32,
                out_block.as_mut_ptr(),
                block_out as u32,
            );

            if ret < 0 {
                sstretch_delete(state);
                return Err("sstretch_process_interleaved failed".to_string());
            }

            // 仅取出当前轮次有效输出的切片
            all_output.extend_from_slice(&out_block[..block_out * channels]);
            in_cursor += block_in;
            out_produced += block_out;
        }

        // flush 残余输出
        if out_produced < total_out {
            let flush_frames = total_out - out_produced;
            let mut flush_buf = vec![0.0f32; flush_frames * channels];
            let _ = sstretch_flush(state, flush_buf.as_mut_ptr(), flush_frames as u32);
            all_output.extend_from_slice(&flush_buf);
        }

        sstretch_delete(state);

        // 跳过 pre-roll，取 out_frames 帧
        let skip = output_latency.min(all_output.len() / channels.max(1));
        let skip_samples = skip * channels;
        let available = all_output.len().saturating_sub(skip_samples);
        let take_samples = (out_frames * channels).min(available);

        if take_samples == 0 {
            return Ok(vec![]);
        }

        let result = all_output[skip_samples..skip_samples + take_samples].to_vec();
        Ok(result)
    }
}
