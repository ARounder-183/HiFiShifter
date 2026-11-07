//! CLAP 缓冲到中性 `AudioBlock` 的适配。
//!
//! 【为什么不照搬 VST3 的 `silence_flags`】CLAP 的 `constant_mask` 是「该声道全块为常量」
//! 的提示，不是「全静音且缓冲仍可读」；把提示当门控会让录音与输出一起静音。这里一律
//! **总是读、总是写满**：只处理 F32，F64 保守静音并返回成功；in-place 时先读后写。

use crate::abi::{clap_audio_buffer_t, clap_process_t, CLAP_RENDER_OFFLINE, CLAP_SECTIME_FACTOR};
use hifishifter_plugin_core::format::{AudioBlock, ChannelPlane, SampleWidth, TransportFrame};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 一个 CLAP 实例的音频时间状态。
///
/// 【为什么不是进程级 static】采样率、`clap.render` 模式与 transport 观察值都随
/// **实例**走：同一个 DAW 进程里两条轨道各插一个本插件，一个 48 kHz 实时、一个
/// 96 kHz 离线 bounce 是完全正常的。此前的全局 `SAMPLE_RATE` / `OFFLINE_RENDER`
/// 会让后 activate 的实例覆盖前一个，结果是错的工程位置换算、错的离线标记，
/// 以及"这条轨明明在实时播放却被标成离线"。状态放在 `Instance` 里，
/// 每个 `process` 只读自己的那一份。
#[derive(Debug, Default)]
pub struct AudioState {
    /// 本实例激活声明的采样率（Hz 的位模式）；0 = 尚未激活。
    sample_rate: AtomicU64,
    /// `clap_plugin_render::set` 给出的当前模式是否为离线。
    offline_render: AtomicBool,
    /// 本实例是否收到过 transport 事件（Capture bootstrap 如实上报）。
    saw_transport: AtomicBool,
    /// 本实例是否收到过离线 render 模式。
    saw_offline: AtomicBool,
    /// 处理过多少块（诊断：宿主到底有没有在跑我们）。
    blocks: AtomicU64,
    /// 见过的 transport 标志位并集（诊断：宿主给的是拍还是秒）。
    transport_flags: AtomicU64,
    /// 是否从 transport 里成功推出过工程位置（录制的硬前提）。
    saw_position: AtomicBool,
}

impl AudioState {
    /// 记录一次激活的采样率。
    pub fn set_sample_rate(&self, rate: f64) {
        self.sample_rate.store(rate.to_bits(), Ordering::Release);
    }

    /// 取本实例的采样率；未激活时是 0（`frame_is_trustworthy` 会拒绝该块）。
    pub fn sample_rate(&self) -> f64 {
        f64::from_bits(self.sample_rate.load(Ordering::Acquire))
    }

    /// 记录一次 render 模式；只把 `CLAP_RENDER_OFFLINE` 当作离线证据。
    /// 传入未定义的模式返回 `false`，不改动状态。
    pub fn set_render_mode(&self, mode: i32) -> bool {
        if mode != crate::abi::CLAP_RENDER_REALTIME && mode != CLAP_RENDER_OFFLINE {
            return false;
        }
        let offline = mode == CLAP_RENDER_OFFLINE;
        self.offline_render.store(offline, Ordering::Release);
        if offline {
            self.saw_offline.store(true, Ordering::Release);
        }
        true
    }

    /// 当前 render 模式是否为离线。
    ///
    /// 【为什么需要读它】共享捕获路径按流分开记账（实时 / 预取 / 离线），而
    /// `clap.render` 的模式是唯一来源；把它留在本结构里，调用方不必自己再存一份。
    pub fn is_offline(&self) -> bool {
        self.offline_render.load(Ordering::Acquire)
    }

    /// Capture bootstrap 的能力读数（是否见过 transport / 是否见过离线 render）。
    /// 诊断读数：@@(处理过的块数, 见过的 transport 标志, 是否推出过位置)@@。
    ///
    /// 【为什么值得单独暴露】"按钮一直是灰的"有两种完全不同的原因：宿主根本没在跑我们
    /// （块数 0），或者宿主跑了但没给可用的时间轴（标志位里只有拍号/速度，没有
    /// seconds/beats）。三个数一起打出来，就不必再猜。
    pub fn capture_evidence(&self) -> (u64, u32, bool) {
        (
            self.blocks.load(Ordering::Acquire),
            self.transport_flags.load(Ordering::Acquire) as u32,
            self.saw_position.load(Ordering::Acquire),
        )
    }

    pub fn capture_observations(&self) -> (bool, bool) {
        (
            self.saw_transport.load(Ordering::Acquire),
            self.saw_offline.load(Ordering::Acquire),
        )
    }

    /// 把 CLAP 的时间读数转换为中性 `TransportFrame`。
    ///
    /// `steady_time` 在停播与重放时仍会前进，因此**绝不**拿它的变化推断播放状态。
    /// 工程位置只从 transport 事件的 `song_pos_seconds` 换算；没有事件就
    /// `position_trusted=false`。采样率与离线标记都取自本实例，不读进程级全局。
    pub fn to_transport_frame(&self, process: &clap_process_t) -> TransportFrame {
        let sample_rate = self.sample_rate();
        let offline = self.offline_render.load(Ordering::Acquire);
        let steady = (process.steady_time >= 0).then_some(process.steady_time);
        self.blocks.fetch_add(1, Ordering::AcqRel);
        let Some(transport) = (unsafe { process.transport.as_ref() }) else {
            return TransportFrame {
                project_time_samples: 0,
                sample_rate,
                playing: None,
                recording: None,
                cycle_active: None,
                tempo: None,
                time_sig_num: None,
                time_sig_den: None,
                steady_time: steady,
                offline,
                position_trusted: false,
            };
        };
        self.saw_transport.store(true, Ordering::Release);
        self.transport_flags
            .fetch_or(u64::from(transport.flags), Ordering::AcqRel);
        // 【为什么秒不是唯一来源】CLAP 的 transport 可以只声明拍时间轴。实测宿主
        // （REAPER 的 CLAP 实现是其中之一）只给 `HAS_BEATS_TIMELINE + HAS_TEMPO` 时，
        // 只认 seconds 的实现会让 position_trusted 永远是 false —— 表现为"捕获按钮
        // 永远灰着"，而插件其实已经被正常处理了。拍是精确的定点值，配上速度就能
        // 换算成秒，因此两种时间轴都接受。
        let position = transport_position(transport, sample_rate);
        if position.is_some() {
            self.saw_position.store(true, Ordering::Release);
        }
        TransportFrame {
            project_time_samples: position.unwrap_or(0),
            sample_rate,
            playing: Some(transport.flags & crate::abi::transport_flags::IS_PLAYING != 0),
            recording: Some(transport.flags & crate::abi::transport_flags::IS_RECORDING != 0),
            cycle_active: Some(transport.flags & crate::abi::transport_flags::IS_LOOP_ACTIVE != 0),
            tempo: (transport.flags & crate::abi::transport_flags::HAS_TEMPO != 0)
                .then_some(transport.tempo),
            time_sig_num: (transport.flags & crate::abi::transport_flags::HAS_TIME_SIGNATURE != 0)
                .then_some(i32::from(transport.tsig_num)),
            time_sig_den: (transport.flags & crate::abi::transport_flags::HAS_TIME_SIGNATURE != 0)
                .then_some(i32::from(transport.tsig_denom)),
            steady_time: steady,
            offline,
            position_trusted: position.is_some(),
        }
    }
}

/// 用渲染结果覆盖输出缓冲；渲染没有覆盖到的帧保持宿主原声（直通）。
///
/// 【为什么在直通之后写】`process_block` 已经把输入抄进输出（in-place 时是同一块
/// 内存）。这里只覆盖"属于素材"的那一段，其余帧原样保留 —— 于是"插件没处理的地方"
/// 听起来与没插插件完全一致，不会出现前 30 秒静音那种断层。
///
/// # Safety
/// `process` 必须是宿主在本次 `process` 回调里给出的有效指针；输出平面在调用期间
/// 有效且至少有 `frames_count` 个样本。
pub unsafe fn overwrite_output(
    process: &clap_process_t,
    tape: &crate::render::RenderTape,
    start_frame: i64,
    passthrough_outside: bool,
) {
    let frames = process.frames_count as usize;
    if frames == 0 || process.audio_outputs.is_null() || process.audio_outputs_count == 0 {
        return;
    }
    let buffer = unsafe { &mut *process.audio_outputs };
    if buffer.data32.is_null() || buffer.channel_count == 0 {
        return;
    }
    for channel in 0..(buffer.channel_count.min(2) as usize) {
        let data = unsafe { *buffer.data32.add(channel) };
        if data.is_null() {
            continue;
        }
        for index in 0..frames {
            let Some(sample) = tape.sample(start_frame + index as i64) else {
                // 【为什么这里可能写 0】`outside_passthrough = false` 是用户的显式
                // 选择："素材之外静音"。默认（true）保持宿主原声 —— 那正是
                // `process_block` 已经写进去的内容。
                if !passthrough_outside {
                    unsafe { *data.add(index) = 0.0 };
                }
                continue;
            };
            unsafe { *data.add(index) = sample[channel.min(1)] };
        }
    }
}

/// 校验并拷贝一个音频块。返回 `false` 表示宿主布局非法（输出必须丢弃）。
///
/// # Safety
/// `process` 必须来自宿主的有效 `clap_process_t`，其缓冲在调用期间有效。
pub unsafe fn process_block(process: &clap_process_t) -> bool {
    let frames = process.frames_count as usize;
    if frames == 0 {
        return true;
    }
    let Some(inputs) = (unsafe { buffers(process.audio_inputs, process.audio_inputs_count) })
    else {
        return false;
    };
    let Some(outputs) =
        (unsafe { buffers_mut(process.audio_outputs, process.audio_outputs_count) })
    else {
        return false;
    };
    let outputs: &[clap_audio_buffer_t] = outputs;
    // F64 不处理：保守地把输出写零，但仍返回成功（与 VST3 kSample64 同口径）。
    if inputs
        .iter()
        .chain(outputs.iter())
        .any(|b| !b.data64.is_null())
    {
        for output in outputs {
            unsafe { clear_f64(output, frames) };
        }
        return true;
    }
    // 固定 2 声道平面数组：音频回调不分配、不拷贝整块 PCM。
    let mut in_planes = [ChannelPlane {
        ptr: std::ptr::null_mut(),
        frames: 0,
    }; 2];
    let mut in_count = 0usize;
    if let Some(buffer) = inputs.first() {
        for channel in 0..(buffer.channel_count as usize).min(2) {
            let Some(ptr) = (unsafe { plane_ptr32(buffer, channel) }) else {
                return false;
            };
            in_planes[in_count] = ChannelPlane {
                ptr: ptr.cast::<c_void>(),
                frames,
            };
            in_count += 1;
        }
    }
    let Some(output) = outputs.first() else {
        return false;
    };
    let channel_count = (output.channel_count as usize).min(2);
    if channel_count == 0 {
        return false;
    }
    let mut out_planes = [ChannelPlane {
        ptr: std::ptr::null_mut(),
        frames: 0,
    }; 2];
    for channel in 0..channel_count {
        let Some(ptr) = (unsafe { plane_ptr32_mut(output, channel) }) else {
            return false;
        };
        out_planes[channel] = ChannelPlane {
            ptr: ptr.cast::<c_void>(),
            frames,
        };
    }
    let in_place = out_planes[..channel_count].iter().any(|out| {
        in_planes[..in_count]
            .iter()
            .any(|input| input.ptr == out.ptr)
    });
    // 【为什么用 AudioBlock 而不是直接操作裸指针】这一层是格式外壳与共享语义的
    // 唯一交界：core 只认识 ChannelPlane，不认识 clap_audio_buffer_t。构造之后
    // 所有搬运都从 block 的平面描述出发，in-place 语义也由 block 自己携带；
    // 把 block 丢在一边另写一套指针循环，等于让中性契约永远没有消费者。
    let mut block = AudioBlock {
        frames,
        sample_width: SampleWidth::F32,
        input: &in_planes[..in_count],
        output: &mut out_planes[..channel_count],
        in_place,
    };
    // 保守直通：逐帧先读两个源样本再写输出；in-place 与声道交换都安全。
    // F64 已在上面单独处理，这里只可能是 F32。
    if block.sample_width != SampleWidth::F32 {
        return false;
    }
    // 【为什么 in_place 不需要单独分支】"读一个源就写一个输出"会在共用地址时
    // 让第二个源读到刚被覆盖的数据。这里每帧都先 read_frame 把两个源读进栈上定长
    // 数组，再 write_frame 统一写全部输出平面，顺序天然满足 in-place 契约。
    // block.in_place 由构造方检测后随 block 传递，供后续共享渲染层判断是否需要
    // 额外的缓冲；本直通实现只依赖"先读齐再写"这一条，不分配堆内存。
    for frame in 0..block.frames {
        let sources = read_frame(&block, frame);
        write_frame(&mut block, frame, &sources);
    }
    true
}

/// 读出一帧两个源声道（缺声道时按既定规则补齐），不触碰任何输出平面。
fn read_frame(block: &AudioBlock<'_>, frame: usize) -> [f32; 3] {
    let left = if !block.input.is_empty() {
        unsafe { *block.input[0].ptr.cast::<f32>().add(frame) }
    } else {
        0.0
    };
    let right = if block.input.len() > 1 {
        unsafe { *block.input[1].ptr.cast::<f32>().add(frame) }
    } else {
        left
    };
    [left, right, (left + right) * 0.5]
}

/// 把一帧写进所有输出平面；单声道输出取两路平均。
fn write_frame(block: &mut AudioBlock<'_>, frame: usize, values: &[f32; 3]) {
    let mono = block.output.len() == 1;
    for (channel, plane) in block.output.iter_mut().enumerate() {
        let target = unsafe { plane.ptr.cast::<f32>().add(frame) };
        let value = if mono { values[2] } else { values[channel] };
        unsafe { target.write(value) };
    }
}

unsafe fn buffers(
    ptr: *const clap_audio_buffer_t,
    count: u32,
) -> Option<&'static [clap_audio_buffer_t]> {
    if count == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, count as usize) })
}
unsafe fn buffers_mut(
    ptr: *mut clap_audio_buffer_t,
    count: u32,
) -> Option<&'static mut [clap_audio_buffer_t]> {
    if count == 0 {
        return Some(&mut []);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, count as usize) })
}
unsafe fn plane_ptr32(buffer: &clap_audio_buffer_t, channel: usize) -> Option<*mut f32> {
    if buffer.data32.is_null() || channel >= buffer.channel_count as usize {
        return None;
    }
    let ptr = unsafe { *buffer.data32.add(channel) };
    (!ptr.is_null()).then_some(ptr)
}
unsafe fn plane_ptr32_mut(buffer: &clap_audio_buffer_t, channel: usize) -> Option<*mut f32> {
    if buffer.data32.is_null() || channel >= buffer.channel_count as usize {
        return None;
    }
    let ptr = unsafe { *buffer.data32.add(channel) };
    (!ptr.is_null()).then_some(ptr)
}
unsafe fn clear_f64(buffer: &clap_audio_buffer_t, frames: usize) {
    if buffer.data64.is_null() {
        return;
    }
    for channel in 0..buffer.channel_count as usize {
        let ptr = unsafe { *buffer.data64.add(channel) };
        if !ptr.is_null() {
            unsafe { std::slice::from_raw_parts_mut(ptr, frames) }.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    include!("audio/tests.rs");
}

/// 从 CLAP transport 推出工程位置（采样帧）；两条时间轴都接受，都没有则 `None`。
///
/// 【为什么抽成函数】它是"宿主时间 → 工程帧"的**唯一**换算：捕获形态的观察用它，
/// ARA 形态喂给共享捕获路径的 `BlockContext` 也用它。两份实现迟早会在
/// "只给拍不给秒"的宿主上分叉（REAPER 的 CLAP 就是其中之一）。
pub fn transport_position(transport: &crate::abi::clap_event_transport_t, sample_rate: f64) -> Option<i64> {
    let has_seconds = transport.flags & crate::abi::transport_flags::HAS_SECONDS_TIMELINE != 0;
    let has_beats = transport.flags & crate::abi::transport_flags::HAS_BEATS_TIMELINE != 0;
    let has_tempo = transport.flags & crate::abi::transport_flags::HAS_TEMPO != 0;
    let seconds = transport.song_pos_seconds as f64 / CLAP_SECTIME_FACTOR;
    let beats = transport.song_pos_beats as f64 / crate::abi::CLAP_BEATTIME_FACTOR;
    if has_seconds && seconds.is_finite() && seconds >= 0.0 {
        Some((seconds * sample_rate).round() as i64)
    } else if has_beats
        && has_tempo
        && beats.is_finite()
        && beats >= 0.0
        && transport.tempo > 0.0
    {
        Some((beats * 60.0 / transport.tempo * sample_rate).round() as i64)
    } else {
        None
    }
}
