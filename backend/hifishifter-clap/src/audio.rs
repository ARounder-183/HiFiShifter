//! CLAP 缓冲到中性 `AudioBlock` 的适配。
//!
//! 【为什么不照搬 VST3 的 `silence_flags`】CLAP 的 `constant_mask` 是「该声道全块为常量」
//! 的提示，不是「全静音且缓冲仍可读」；把提示当门控会让录音与输出一起静音。这里一律
//! **总是读、总是写满**：只处理 F32，F64 保守静音并返回成功；in-place 时先读后写。

use crate::abi::{clap_audio_buffer_t, clap_process_t, CLAP_SECTIME_FACTOR};
use hifishifter_plugin_core::format::{AudioBlock, ChannelPlane, SampleWidth, TransportFrame};
use hifishifter_plugin_core::resource_root;
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
    let inputs = unsafe { buffers(process.audio_inputs, process.audio_inputs_count) };
    let outputs = unsafe { buffers_mut(process.audio_outputs, process.audio_outputs_count) };
    let Some(inputs) = inputs else { return false };
    let Some(outputs) = outputs else { return false };
    let outputs: &[clap_audio_buffer_t] = outputs;
    // F64 不处理：保守地把输出写零，但仍返回成功（与 VST3 kSample64 同口径）。
    let f64_width = inputs
        .iter()
        .chain(outputs.iter())
        .any(|b| !b.data64.is_null());
    if f64_width {
        for output in outputs {
            unsafe { clear_f64(output, frames) };
        }
        return true;
    }
    let mut in_planes = Vec::with_capacity(2);
    if let Some(buffer) = inputs.first() {
        for channel in 0..(buffer.channel_count as usize).min(2) {
            let ptr = unsafe { plane_ptr32(buffer, channel, frames) };
            let Some(ptr) = ptr else { return false };
            in_planes.push(ChannelPlane {
                ptr: ptr.cast::<c_void>(),
                frames,
            });
        }
    }
    let mut out_planes = Vec::with_capacity(2);
    let out_channels = outputs.first().map_or(0, |b| b.channel_count as usize);
    if out_channels == 0 {
        return false;
    }
    for channel in 0..out_channels {
        let ptr = unsafe {
            plane_ptr32_mut(
                &outputs[channel / out_channels.max(1)],
                channel.min(out_channels - 1),
                frames,
            )
        };
        let Some(ptr) = ptr else { return false };
        out_planes.push(ChannelPlane {
            ptr: ptr.cast::<c_void>(),
            frames,
        });
    }
    let in_place = out_planes
        .iter()
        .any(|out| in_planes.iter().any(|input| input.ptr == out.ptr));
    // Core 的中性契约只描述布局；真正的 Capture/render 消费在宿主面接通之前一律透传。
    let _block = AudioBlock {
        frames,
        sample_width: SampleWidth::F32,
        input: &in_planes,
        output: &mut out_planes,
        in_place,
    };
    // 保守直通：先用输入平面构造快照，再写输出，保证 in-place 时先读后写。
    let snapshot: Vec<Vec<f32>> = in_planes
        .iter()
        .map(|p| unsafe { std::slice::from_raw_parts(p.ptr.cast::<f32>(), p.frames).to_vec() })
        .collect();
    for (index, plane) in out_planes.iter_mut().enumerate() {
        let source = snapshot.get(index).or_else(|| snapshot.first());
        let target =
            unsafe { std::slice::from_raw_parts_mut(plane.ptr.cast::<f32>(), plane.frames) };
        if let Some(source) = source {
            if source.len() == target.len() {
                target.copy_from_slice(source);
            } else if source.len() == 1 {
                target.fill(source[0]);
            } else {
                target.fill(0.0);
            }
        } else {
            target.fill(0.0);
        }
    }
    let _ = resource_root::layout();
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
unsafe fn plane_ptr32(
    buffer: &clap_audio_buffer_t,
    channel: usize,
    frames: usize,
) -> Option<*mut f32> {
    let _ = frames;
    if buffer.data32.is_null() || channel >= buffer.channel_count as usize {
        return None;
    }
    let ptr = unsafe { *buffer.data32.add(channel) };
    (!ptr.is_null()).then_some(ptr)
}
unsafe fn plane_ptr32_mut(
    buffer: &clap_audio_buffer_t,
    channel: usize,
    frames: usize,
) -> Option<*mut f32> {
    let _ = frames;
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
