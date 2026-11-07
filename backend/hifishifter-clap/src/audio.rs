//! CLAP 缓冲到中性 `AudioBlock` 的适配。
//!
//! 【为什么不照搬 VST3 的 `silence_flags`】CLAP 的 `constant_mask` 是「该声道全块为常量」
//! 的提示，不是「全静音且缓冲仍可读」；把提示当门控会让录音与输出一起静音。这里一律
//! **总是读、总是写满**：只处理 F32，F64 保守静音并返回成功；in-place 时先读后写。

use crate::abi::{clap_audio_buffer_t, clap_process_t, CLAP_RENDER_OFFLINE, CLAP_SECTIME_FACTOR};
use hifishifter_plugin::audio_block::{AudioBlock, InputBus, OutputBus};
use hifishifter_plugin_core::format::TransportFrame;
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
    // 【为什么这里不再有 saw_transport / saw_offline / blocks 这几份读数】它们描述的是
    // “宿主实际给了什么”，而那是**捕获存储**的观察位（~CaptureStore::status~ 里的
    // ~saw_context~ / ~saw_input~ / ~saw_offline~）。同一个事实存两份，界面就会在
    // 某一次改动后读到互相矛盾的答案。本结构只保留**它独有**的两项：声明过的采样率
    // 与 render 模式。
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
        self.offline_render
            .store(mode == CLAP_RENDER_OFFLINE, Ordering::Release);
        true
    }

    /// 当前 render 模式是否为离线。
    ///
    /// 【为什么需要读它】共享捕获路径按流分开记账（实时 / 预取 / 离线），而
    /// `clap.render` 的模式是唯一来源；把它留在本结构里，调用方不必自己再存一份。
    pub fn is_offline(&self) -> bool {
        self.offline_render.load(Ordering::Acquire)
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
        // 【为什么这里不再记读数】“宿主给过什么”由共享捕获路径记在 `CaptureStore` 上
        // （见本文件的字段说明）；这里只负责把这一块的时间换算成中性描述。
        // 【为什么秒不是唯一来源】CLAP 的 transport 可以只声明拍时间轴。实测宿主
        // （REAPER 的 CLAP 实现是其中之一）只给 `HAS_BEATS_TIMELINE + HAS_TEMPO` 时，
        // 只认 seconds 的实现会让 position_trusted 永远是 false —— 表现为"捕获按钮
        // 永远灰着"，而插件其实已经被正常处理了。拍是精确的定点值，配上速度就能
        // 换算成秒，因此两种时间轴都接受。
        let position = transport_position(transport, sample_rate);
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

/// 把 CLAP 的 `process` 数据翻译成共享的中立块。
///
/// 【为什么翻译放在格式外壳】`clap_audio_buffer_t` 的字段布局、`data32`/`data64` 的
/// 联合体语义、`channel_count` 与指针数组的对应关系，都是 **CLAP ABI** 的知识。
/// 共享渲染层刻意不认识它们 —— 它只面对“平面在哪、有几帧、是不是 in-place”。
///
/// 【为什么 in-place 要看“同一批平面指针”】CLAP 没有 VST3 那种“同一份总线结构”的
/// 概念，宿主只保证输入与输出的平面指针**可能相同**。逐声道比对指针是这里唯一可靠的
/// 判据（而它只影响静音掩码这条可选提示）。
///
/// 【返回 `None` 的情形】空块或没有可写输出平面 —— 调用方对这两种情形的既有语义都是
/// “本块没有可处理的内容”。
///
/// # Safety
/// `process` 必须是宿主在本次回调里给出的有效记录；非空平面至少含 `frames_count` 帧。
pub unsafe fn block_from_clap<'a>(
    process: &'a clap_process_t,
    mode: i32,
    silence: &'a mut u64,
) -> Option<AudioBlock<'a>> {
    let frames = process.frames_count as usize;
    if frames == 0 || process.audio_outputs_count == 0 || process.audio_outputs.is_null() {
        return None;
    }
    let output = unsafe { &mut *process.audio_outputs };
    if output.channel_count == 0 || output.data32.is_null() {
        return None;
    }
    let input = if process.audio_inputs_count > 0 && !process.audio_inputs.is_null() {
        let bus = unsafe { &*process.audio_inputs };
        // SAFETY: 宿主声明 `channel_count` 个平面指针，本回调期间有效。
        unsafe { InputBus::new(bus.data32, bus.channel_count as usize) }
    } else {
        InputBus::empty()
    };
    let in_place = if process.audio_inputs_count > 0 && !process.audio_inputs.is_null() {
        let bus = unsafe { &*process.audio_inputs };
        let shared = bus.channel_count.min(output.channel_count) as usize;
        (0..shared).any(|channel| {
            // SAFETY: `channel < min(两个声道路数)`，两侧都声明了对应指针。
            let source = unsafe { *bus.data32.add(channel) };
            let target = unsafe { *output.data32.add(channel) };
            !source.is_null() && source == target
        })
    } else {
        false
    };
    // SAFETY: 上面已确认指针数组非空且声明了 `channel_count` 个平面。
    let pointers: &'a mut [*mut f32] =
        unsafe { std::slice::from_raw_parts_mut(output.data32, output.channel_count as usize) };
    Some(AudioBlock {
        frames,
        mode,
        input,
        output: OutputBus::new(pointers, silence),
        in_place,
    })
}

/// 校验并直通一个音频块。返回 `false` 表示宿主布局非法（输出必须丢弃）。
///
/// 【为什么这里只剩“翻译 + 直通”】in-place 的“先读齐再写”、单声道下混、inactive 声道
/// 跳过 —— 这些结论现在只有一处实现（`AudioBlock::pass_through`）。此前它们是本文件
/// 里的一段独立循环，与 VST3 侧的同名逻辑各写一遍。
///
/// # Safety
/// `process` 必须来自宿主的有效 `clap_process_t`，其缓冲在调用期间有效。
pub unsafe fn process_block(process: &clap_process_t) -> bool {
    let frames = process.frames_count as usize;
    if frames == 0 {
        return true;
    }
    // F64 不处理：保守地把输出写零，但仍返回成功（与 VST3 kSample64 同口径）。
    if let Some(outputs) = unsafe { buffers(process.audio_outputs, process.audio_outputs_count) } {
        if outputs.iter().any(|buffer| !buffer.data64.is_null()) {
            for output in outputs {
                unsafe { clear_f64(output, frames) };
            }
            return true;
        }
    }
    let mut silence = 0_u64;
    let Some(mut block) = (unsafe { block_from_clap(process, 0, &mut silence) }) else {
        return false;
    };
    // SAFETY: 由本函数的调用契约保证。
    unsafe { block.pass_through() }.is_ok()
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
