//! VST3 音频缓冲与时间上下文 ABI。以锁定 SDK 的原生布局测试为契约。

use std::ffi::c_void;

/// SDK union 的两个成员都是同尺寸指针；只在 kSample32 下解引用。
#[repr(C)]
pub(crate) struct AudioBusBuffers {
    pub num_channels: i32,
    pub silence_flags: u64,
    pub channel_buffers: *mut *mut f32,
}

/// SDK ProcessData 的完整结构，不把可选指针字段省略为自定义前缀。
#[repr(C)]
#[derive(Default)]
pub(crate) struct ProcessData {
    pub process_mode: i32,
    pub symbolic_sample_size: i32,
    pub num_samples: i32,
    pub num_inputs: i32,
    pub num_outputs: i32,
    pub inputs: *mut AudioBusBuffers,
    pub outputs: *mut AudioBusBuffers,
    pub input_parameter_changes: *mut c_void,
    pub output_parameter_changes: *mut c_void,
    pub input_events: *mut c_void,
    pub output_events: *mut c_void,
    pub process_context: *mut ProcessContext,
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct Chord {
    pub key_note: u8,
    pub root_note: u8,
    pub chord_mask: i16,
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct FrameRate {
    pub frames_per_second: u32,
    pub flags: u32,
}

/// 完整 SDK 时间上下文；project_time_samples 始终有效，不依赖音乐时间有效位。
#[repr(C)]
#[derive(Default)]
pub(crate) struct ProcessContext {
    pub state: u32,
    pub sample_rate: f64,
    pub project_time_samples: i64,
    pub system_time: i64,
    pub continuous_time_samples: i64,
    pub project_time_music: f64,
    pub bar_position_music: f64,
    pub cycle_start_music: f64,
    pub cycle_end_music: f64,
    pub tempo: f64,
    pub time_sig_numerator: i32,
    pub time_sig_denominator: i32,
    pub chord: Chord,
    pub smpte_offset_subframes: i32,
    pub frame_rate: FrameRate,
    pub samples_to_next_clock: i32,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum BufferError {
    InvalidArgument,
    UnsupportedFormat,
}

/// 校验当前支持的总线形状，不读取或写入 plane；inactive plane 的空指针合法。
///
/// 【为什么允许单声道】人声轨单声道极其常见，而此前 1 声道直接被判
/// `UnsupportedFormat` ⇒ `process` 返回失败 ⇒ 宿主**静默旁路**插件。内部渲染仍是
/// 立体声（`PlaybackSnapshot` 始终持左右两轨），单声道只是**输出面**的降级，见
/// `render::snapshot` 的 `copy_block`。
fn validate_bus(bus: &AudioBusBuffers) -> Result<(), BufferError> {
    if bus.num_channels < 0 {
        return Err(BufferError::InvalidArgument);
    }
    if bus.num_channels > 2 {
        return Err(BufferError::UnsupportedFormat);
    }
    if bus.num_channels > 0 && bus.channel_buffers.is_null() {
        return Err(BufferError::InvalidArgument);
    }
    Ok(())
}

/// 先完整校验总线形状，不写输出，也不能破坏in-place的输入。
///
/// # Safety
/// 宿主须提供合法对齐的 SDK 结构、声明数量的总线/通道数组与至少 num_samples 帧的
/// 可写非空 plane。inactive plane 可以为 null。指针只在当前回调使用，不保留。
pub(crate) unsafe fn validate(data: *mut ProcessData) -> Result<(), BufferError> {
    if data.is_null() {
        return Err(BufferError::InvalidArgument);
    }
    // SAFETY: 调用者提供当前回调独占的 SDK 结构。
    let data = unsafe { &*data };
    if data.num_samples < 0
        || data.num_inputs < 0
        || data.num_outputs < 0
        || !(0..=2).contains(&data.process_mode)
    {
        return Err(BufferError::InvalidArgument);
    }
    if data.symbolic_sample_size != 0 || data.num_inputs > 1 || data.num_outputs > 1 {
        return Err(BufferError::UnsupportedFormat);
    }
    // 参数 flush 不要求音频 plane；SDK 允许无输出的 flush/事件调用。
    if data.num_samples == 0 {
        return Ok(());
    }
    if (data.num_inputs > 0 && data.inputs.is_null())
        || (data.num_outputs > 0 && data.outputs.is_null())
    {
        return Err(BufferError::InvalidArgument);
    }
    if data.num_inputs > 0 {
        // SAFETY: 已校验唯一输入总线非空；这里只读形状，不读宿主样本。
        validate_bus(unsafe { &*data.inputs })?;
    }
    if data.num_outputs == 0 {
        return Ok(());
    }
    // SAFETY: 已校验唯一输出总线非空，宿主提供完整结构。
    let output = unsafe { &*data.outputs };
    validate_bus(output)?;
    Ok(())
}

/// 仅在完整校验后清输出；playback renderer失败/停播仍清零，纯editor不走此入口。
/// # Safety
/// 与validate相同，非空output plane必须有至少num_samples可写帧。
pub(crate) unsafe fn clear_outputs(data: *mut ProcessData) -> Result<(), BufferError> {
    unsafe {
        validate(data)?;
    }
    let data = unsafe { &*data };
    if data.num_samples == 0 || data.num_outputs == 0 {
        return Ok(());
    }
    let output = unsafe { &mut *data.outputs };
    if output.num_channels == 0 {
        output.silence_flags = 0;
        return Ok(());
    }
    // 【为什么按 `num_channels` 而不是写死 2】总线现在允许单声道（见 `validate_bus`），
    // 写死 2 会读越界的 `channel_buffers[1]`。
    let channels = output.num_channels.clamp(0, 2) as usize;
    for channel in 0..channels {
        // SAFETY: 宿主提供 `num_channels` 个 plane 指针，非空 plane 至少含指定帧数。
        let plane = unsafe { *output.channel_buffers.add(channel) };
        if !plane.is_null() {
            unsafe { std::ptr::write_bytes(plane, 0, data.num_samples as usize) };
        }
    }
    output.silence_flags = (1_u64 << channels) - 1;
    Ok(())
}

/// 纯editor renderer按SDK透传宿主信号；输入/输出bus或对应plane相同也不能先清零。
/// # Safety
/// 与validate相同，输入非空plane可读、输出可写num_samples帧；不保留任何宿主指针。
pub(crate) unsafe fn pass_through(data: *mut ProcessData) -> Result<(), BufferError> {
    unsafe {
        validate(data)?;
    }
    let data = unsafe { &*data };
    if data.num_samples == 0 || data.num_outputs == 0 {
        return Ok(());
    }
    // 先按值复制header，允许inputs/outputs指向同一个AudioBusBuffers，不制造共享/可变引用别名。
    let input = if data.num_inputs > 0 {
        Some(unsafe { std::ptr::read(data.inputs) })
    } else {
        None
    };
    let output = unsafe { &mut *data.outputs };
    if output.num_channels == 0 {
        output.silence_flags = 0;
        return Ok(());
    }
    // 【为什么按实际声道数展开】总线允许单声道（见 `validate_bus`）。单声道输入喂给
    // 立体声输出时两个输出通道复用同一个源 plane（下面的"每帧先读后写"已经处理别名）；
    // 立体声输入给单声道输出时取平均，而不是丢掉右声道。
    let channels = output.num_channels.clamp(0, 2) as usize;
    let input_channels = input
        .as_ref()
        .map(|bus| bus.num_channels.clamp(0, 2) as usize)
        .unwrap_or(0);
    let sources: [*mut f32; 2] = std::array::from_fn(|channel| match &input {
        Some(bus) if input_channels > 0 => {
            let index = if input_channels == 1 { 0 } else { channel };
            if index >= input_channels || bus.silence_flags & (1 << index) != 0 {
                std::ptr::null_mut()
            } else {
                // SAFETY: `index < input_channels <= 2`，宿主提供对应数量的 plane 指针。
                unsafe { *bus.channel_buffers.add(index) }
            }
        }
        _ => std::ptr::null_mut(),
    });
    let targets: [*mut f32; 2] = std::array::from_fn(|channel| {
        if channel < channels {
            // SAFETY: `channel < channels = output.num_channels`，宿主提供对应数量的指针。
            unsafe { *output.channel_buffers.add(channel) }
        } else {
            std::ptr::null_mut()
        }
    });
    let bytes = data.num_samples as usize * std::mem::size_of::<f32>();
    for source in sources {
        for target in targets {
            if source.is_null() || target.is_null() || source == target {
                continue;
            }
            let a = source as usize;
            let b = target as usize;
            if a < b.saturating_add(bytes) && b < a.saturating_add(bytes) {
                return Err(BufferError::UnsupportedFormat);
            }
        }
    }
    // 支持独立plane或整个plane的in-place/交换；部分偏移重叠明确拒绝且不写数据。
    // 每帧先读两个源再写输出，兼容对应in-place与两个完整plane的交换别名。
    for frame in 0..data.num_samples as usize {
        let values = sources.map(|source| {
            if source.is_null() {
                0.0
            } else {
                unsafe { *source.add(frame) }
            }
        });
        if channels == 1 {
            let mono = if input_channels >= 2 {
                (values[0] + values[1]) * 0.5
            } else {
                values[0]
            };
            if !targets[0].is_null() {
                unsafe { targets[0].add(frame).write(mono) };
            }
        } else {
            for channel in 0..channels {
                if !targets[channel].is_null() {
                    unsafe {
                        targets[channel].add(frame).write(values[channel]);
                    }
                }
            }
        }
    }
    output.silence_flags = (0..channels).fold(0, |flags, channel| {
        if sources[channel].is_null() || targets[channel].is_null() {
            flags | (1 << channel)
        } else {
            flags
        }
    });
    Ok(())
}

#[cfg(all(test, target_os = "windows", target_arch = "x86_64"))]
mod layout_tests {
    use super::*;
    use crate::vst3::ProcessSetup;
    use std::mem::{align_of, offset_of, size_of};
    use std::path::PathBuf;
    use std::process::Command;

    /// 原生 SDK 编译器为布局 oracle；任何字段偏移/对齐漂移都必须阻断测试。
    #[test]
    fn rust_audio_layout_matches_the_locked_native_sdk() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let scratch = manifest.join("../../.build-tmp/vst3-layout");
        std::fs::create_dir_all(&scratch).unwrap();
        let sdk = std::env::var_os("ARA_VST3_SDK_DIR").expect("ARA_VST3_SDK_DIR required");
        let exe = scratch.join("vst3-layout.exe");
        let compile = Command::new("cl.exe")
            .current_dir(&scratch)
            .args(["/nologo", "/std:c++17", "/EHsc", "/utf-8"])
            .arg(format!("/I{}", PathBuf::from(sdk).display()))
            .arg(format!("/Fo{}", scratch.join("vst3-layout.obj").display()))
            .arg(format!("/Fe{}", exe.display()))
            .arg(manifest.join("tests/vst3_layout.cpp"))
            .output()
            .expect("MSVC cl.exe required, load tools/msvc-env.ps1");
        assert!(
            compile.status.success(),
            "native ABI oracle compile failed: {compile:?}"
        );
        let native = Command::new(exe).output().unwrap();
        assert!(
            native.status.success(),
            "native ABI oracle failed: {native:?}"
        );
        let layout: serde_json::Value = serde_json::from_slice(&native.stdout).unwrap();
        macro_rules! check {
            ($name:expr, $value:expr) => {
                assert_eq!(
                    layout[$name].as_u64().unwrap(),
                    $value as u64,
                    "ABI {}",
                    $name
                );
            };
        }
        macro_rules! kind {
            ($ty:ty) => {
                check!(concat!(stringify!($ty), ".size"), size_of::<$ty>());
                check!(concat!(stringify!($ty), ".align"), align_of::<$ty>());
            };
        }
        macro_rules! field {
            ($ty:ty, $rust:ident, $native:ident) => {
                check!(
                    concat!(stringify!($ty), ".", stringify!($native)),
                    offset_of!($ty, $rust)
                );
            };
        }
        kind!(ProcessSetup);
        field!(ProcessSetup, process_mode, processMode);
        field!(ProcessSetup, symbolic_sample_size, symbolicSampleSize);
        field!(ProcessSetup, max_samples_per_block, maxSamplesPerBlock);
        field!(ProcessSetup, sample_rate, sampleRate);
        kind!(AudioBusBuffers);
        field!(AudioBusBuffers, num_channels, numChannels);
        field!(AudioBusBuffers, silence_flags, silenceFlags);
        field!(AudioBusBuffers, channel_buffers, channelBuffers32);
        field!(AudioBusBuffers, channel_buffers, channelBuffers64);
        kind!(ProcessData);
        field!(ProcessData, process_mode, processMode);
        field!(ProcessData, symbolic_sample_size, symbolicSampleSize);
        field!(ProcessData, num_samples, numSamples);
        field!(ProcessData, num_inputs, numInputs);
        field!(ProcessData, num_outputs, numOutputs);
        field!(ProcessData, inputs, inputs);
        field!(ProcessData, outputs, outputs);
        field!(ProcessData, input_parameter_changes, inputParameterChanges);
        field!(
            ProcessData,
            output_parameter_changes,
            outputParameterChanges
        );
        field!(ProcessData, input_events, inputEvents);
        field!(ProcessData, output_events, outputEvents);
        field!(ProcessData, process_context, processContext);
        kind!(ProcessContext);
        field!(ProcessContext, state, state);
        field!(ProcessContext, sample_rate, sampleRate);
        field!(ProcessContext, project_time_samples, projectTimeSamples);
        field!(ProcessContext, system_time, systemTime);
        field!(
            ProcessContext,
            continuous_time_samples,
            continousTimeSamples
        );
        field!(ProcessContext, project_time_music, projectTimeMusic);
        field!(ProcessContext, bar_position_music, barPositionMusic);
        field!(ProcessContext, cycle_start_music, cycleStartMusic);
        field!(ProcessContext, cycle_end_music, cycleEndMusic);
        field!(ProcessContext, tempo, tempo);
        field!(ProcessContext, time_sig_numerator, timeSigNumerator);
        field!(ProcessContext, time_sig_denominator, timeSigDenominator);
        field!(ProcessContext, chord, chord);
        field!(ProcessContext, smpte_offset_subframes, smpteOffsetSubframes);
        field!(ProcessContext, frame_rate, frameRate);
        field!(ProcessContext, samples_to_next_clock, samplesToNextClock);
        kind!(Chord);
        field!(Chord, key_note, keyNote);
        field!(Chord, root_note, rootNote);
        field!(Chord, chord_mask, chordMask);
        kind!(FrameRate);
        field!(FrameRate, frames_per_second, framesPerSecond);
        field!(FrameRate, flags, flags);
    }
}
