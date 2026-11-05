//! CLAP 缓冲到中性 `AudioBlock` 的适配。
//!
//! 【为什么不照搬 VST3 的 `silence_flags`】CLAP 的 `constant_mask` 是「该声道全块为常量」
//! 的提示，不是「全静音且缓冲仍可读」；把提示当门控会让录音与输出一起静音。这里一律
//! **总是读、总是写满**：只处理 F32，F64 保守静音并返回成功；in-place 时先读后写。

use crate::abi::{clap_audio_buffer_t, clap_process_t, CLAP_SECTIME_FACTOR};
use hifishifter_plugin_core::format::{AudioBlock, ChannelPlane, SampleWidth, TransportFrame};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 最近一次激活声明的采样率（Hz 的位模式）。
static SAMPLE_RATE: AtomicU64 = AtomicU64::new(0);
/// `clap_plugin_render::set(CLAP_RENDER_OFFLINE)` 的观察结果。
static OFFLINE_RENDER: AtomicBool = AtomicBool::new(false);
/// 「宿主是否给过 transport」的观察结果（Capture bootstrap 如实上报）。
static SAW_TRANSPORT: AtomicBool = AtomicBool::new(false);
/// 「宿主是否给过离线 render」的观察结果。
static SAW_OFFLINE: AtomicBool = AtomicBool::new(false);

/// 记录一次激活的采样率。
pub fn set_sample_rate(rate: f64) {
    SAMPLE_RATE.store(rate.to_bits(), Ordering::Release);
}
/// 记录一次 render 模式；只把 `CLAP_RENDER_OFFLINE` 当作离线证据。
pub fn set_render_mode(mode: i32) {
    let offline = mode == crate::abi::CLAP_RENDER_OFFLINE;
    OFFLINE_RENDER.store(offline, Ordering::Release);
    if offline {
        SAW_OFFLINE.store(true, Ordering::Release);
    }
}
/// Capture bootstrap 的能力读数。
pub fn capture_observations() -> (bool, bool) {
    (
        SAW_TRANSPORT.load(Ordering::Acquire),
        SAW_OFFLINE.load(Ordering::Acquire),
    )
}
/// 本产物观察到过 transport 事件（用于黑盒探针打印，不用于假装播放）。
pub fn saw_transport() -> bool {
    SAW_TRANSPORT.load(Ordering::Acquire)
}
/// 本产物观察到过离线 render 模式。
pub fn saw_offline() -> bool {
    SAW_OFFLINE.load(Ordering::Acquire)
}

/// 把 CLAP 的时间读数转换为中性 `TransportFrame`。
///
/// `steady_time` 在停播与重放时仍会前进，因此**绝不**拿它的变化推断播放状态。
/// 工程位置只从 transport 事件的 `song_pos_seconds` 换算；没有事件就 `position_trusted=false`。
pub fn to_transport_frame(process: &clap_process_t) -> TransportFrame {
    let sample_rate = f64::from_bits(SAMPLE_RATE.load(Ordering::Acquire));
    let offline = OFFLINE_RENDER.load(Ordering::Acquire);
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
    SAW_TRANSPORT.store(true, Ordering::Release);
    let has_seconds = transport.flags & crate::abi::transport_flags::HAS_SECONDS_TIMELINE != 0;
    let seconds = transport.song_pos_seconds as f64 / CLAP_SECTIME_FACTOR;
    let position = (has_seconds && seconds.is_finite() && seconds >= 0.0)
        .then(|| (seconds * sample_rate).round() as i64);
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
    let _block = AudioBlock {
        frames,
        sample_width: SampleWidth::F32,
        input: &in_planes[..in_count],
        output: &mut out_planes[..channel_count],
        in_place,
    };
    // 保守直通：逐帧先读两个源样本再写输出；in-place 与声道交换都安全。
    for frame in 0..frames {
        let left = if in_count > 0 {
            unsafe { *in_planes[0].ptr.cast::<f32>().add(frame) }
        } else {
            0.0
        };
        let right = if in_count > 1 {
            unsafe { *in_planes[1].ptr.cast::<f32>().add(frame) }
        } else {
            left
        };
        let values = [left, right, (left + right) * 0.5];
        for channel in 0..channel_count {
            let target = unsafe { out_planes[channel].ptr.cast::<f32>().add(frame) };
            let value = if channel_count == 1 {
                values[2]
            } else {
                values[channel]
            };
            unsafe { target.write(value) };
        }
    }
    true
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
