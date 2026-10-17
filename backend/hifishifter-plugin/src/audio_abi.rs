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

/// `ProcessContext::state` 的 `kPlaying` 位。
///
/// 【为什么需要它】"宿主**正在**播放"与"宿主给了工程位置"是两件事：后者（
/// `projectTimeSamples`，SDK 标为 always valid）只说明这条通路成立，前者才决定按了录制
/// 之后**有没有东西进来**。面板据此把"按了录制却没有声音"的原因说清楚。
///
/// 取值来自 SDK `ivstprocesscontext.h` 的 `StatesAndFlags::kPlaying = 1 << 1`。
pub(crate) const PROCESS_CONTEXT_PLAYING: u32 = 1 << 1;

/// `ProcessContext::state` 的 `kRecording` 位（SDK `ivstprocesscontext.h:104`）。
///
/// 捕获选举用它优先实时流：录音轨上的真实输入在实时路径上，prefetch 可能只是
/// 预计算。它仍只是提示；缺席时由连续块与非静音证据决定。
pub(crate) const PROCESS_CONTEXT_RECORDING: u32 = 1 << 3;

/// `ProcessContext::state` 的 `kCycleActive` 位（SDK `ivstprocesscontext.h:105`）。
pub(crate) const PROCESS_CONTEXT_CYCLE_ACTIVE: u32 = 1 << 2;

/// `projectTimeMusic` 是否有效（SDK `ivstprocesscontext.h:111`）。
pub(crate) const PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID: u32 = 1 << 9;

/// `tempo` 是否有效（SDK `ivstprocesscontext.h:112`）。
pub(crate) const PROCESS_CONTEXT_TEMPO_VALID: u32 = 1 << 10;

/// `cycleStartMusic` / `cycleEndMusic` 是否有效（SDK `ivstprocesscontext.h:114`）。
pub(crate) const PROCESS_CONTEXT_CYCLE_VALID: u32 = 1 << 12;

/// `continuousTimeSamples` 是否有效（SDK `ivstprocesscontext.h:113`）。
pub(crate) const PROCESS_CONTEXT_CONTINUOUS_VALID: u32 = 1 << 17;

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
    // 【为什么不再拒绝 > 2 声道】此前 5.1 / 7.1 / 任意环绕总线一律
    // `UnsupportedFormat`，而 `setBusArrangements` 也跟着返回 `kResultFalse` ——
    // 多数宿主据此**把插件从插槽里弹出**（用户看到"插件不见了"，且没有提示）。
    // 正确的降级是：接受布局、只处理前两个声道、其余保持静音并在场。
    // 上限 64 只为把静音掩码的移位限定在合法范围（`1u64 << 64` 是 UB）。
    if bus.num_channels > 64 {
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

/// 输入与输出是否**共用同一份总线结构**（in-place 宿主）。
///
/// 【为什么必须知道这件事】`silence_flags` 是**写在总线结构里**的。宿主可以合法地把
/// 同一个 `AudioBusBuffers` 同时当输入和输出（见 `pass_through` 的说明），于是一个插件
/// 给自己输出写的"全是静音"掩码，会在**下一块**被它自己当作"宿主说这个输入是静音的"
/// 读回来 —— 自己把自己录成静音、也把自己直通成静音。掩码只是给宿主省算力的**可选**
/// 提示，in-place 时省掉它没有任何代价。
pub(crate) fn input_output_share_bus(data: &ProcessData) -> bool {
    data.num_inputs > 0
        && data.num_outputs > 0
        && !data.inputs.is_null()
        && data.inputs == data.outputs
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
    // 【为什么按 `num_channels` 而不是写死 2，也不再钳到 2】总线现在接受任意声道数
    // （见 `validate_bus`）：钳到 2 会让第 3..N 个声道**留着上一次回调的残留**——
    // 而 `copy_block` 只写前两个，于是环绕声道输出垃圾。这里必须把所有声明的声道都
    // 清零，后两个之后的部分才真正静音。
    let channels = output.num_channels as usize;
    for channel in 0..channels {
        // SAFETY: 宿主提供 `num_channels` 个 plane 指针，非空 plane 至少含指定帧数。
        let plane = unsafe { *output.channel_buffers.add(channel) };
        if !plane.is_null() {
            unsafe { std::ptr::write_bytes(plane, 0, data.num_samples as usize) };
        }
    }
    // 【为什么 in-place 时不写这个掩码】见 [`input_output_share_bus`]：写进去的
    // "全静音"会在下一块被我们自己的 `capture_view` 读成"宿主没给输入"。
    if !input_output_share_bus(data) {
        output.silence_flags = (1_u64 << channels.min(64)) - 1;
    }
    Ok(())
}

/// 非 32-bit 浮点（`kSample64`）时的**降级静音**：按宿主声明的样本宽度清零输出。
///
/// 【为什么是静音 + OK，而不是返回失败】`validate` 对非 32-bit 浮点判
/// `UnsupportedFormat` ⇒ `process` 返回非 OK ⇒ 多数宿主把插件**整条踢出链路**（用户
/// 看到"插件不见了"，且日志无从查看）。我们确实读不了 f64 平面（`copy_block` 只按
/// `f32` 解引用），所以正确的降级是：不读、按实际宽度写零、返回 OK —— 宿主得到静音
/// 但插件在场，可诊断、可恢复。真正的 64-bit 处理是独立一期。
///
/// 【返回 false 的情形】不是"样本宽度"问题（`symbolic_sample_size == 0`）时返回
/// false，让调用方沿用原有的失败上报 —— 多总线等形状问题不在本降级覆盖范围内。
///
/// # Safety
/// 调用方保证 `data` 指向当前回调独占的 SDK 结构；非空 plane 至少含 `num_samples` 个
/// 宿主声明宽度的样本。
pub(crate) unsafe fn silence_unsupported_sample_size(data: *mut ProcessData) -> bool {
    if data.is_null() {
        return false;
    }
    // SAFETY: 调用者提供当前回调独占的 SDK 结构。
    let data = unsafe { &*data };
    if data.symbolic_sample_size == 0 || data.num_samples < 0 || data.num_outputs < 0 {
        return false;
    }
    if data.num_samples == 0 || data.num_outputs == 0 {
        return true;
    }
    // 只认 kSample64（1）；其它未知宽度交给调用方按失败处理。
    if data.symbolic_sample_size != 1 || data.outputs.is_null() {
        return false;
    }
    for bus in 0..data.num_outputs as usize {
        // SAFETY: 宿主声明 num_outputs 个总线结构；指针数组可索引。
        let output = unsafe { &mut *data.outputs.add(bus) };
        if output.num_channels < 0 || (output.num_channels > 0 && output.channel_buffers.is_null())
        {
            return false;
        }
        for channel in 0..output.num_channels as usize {
            // SAFETY: 宿主声明 num_channels 个 plane 指针。
            let plane = unsafe { *output.channel_buffers.add(channel) };
            if !plane.is_null() {
                // SAFETY: 非空 plane 至少含 num_samples 个 f64 样本（8 字节/样本）。
                unsafe {
                    std::ptr::write_bytes(plane as *mut u8, 0, data.num_samples as usize * 8)
                };
            }
        }
        output.silence_flags = (1_u64 << output.num_channels.min(64)) - 1;
    }
    true
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
            // 【为什么不看 `silence_flags`】与 `capture_view` 同一理由：该位只是可选提示，
            // 而置位时缓冲**仍必须可读**。把它当成"没有输入"会让一条本来有声的轨道
            // 被插件静音掉 —— 用户看到的是"插上插件就没声音了"。
            if index >= input_channels {
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

/// 从输入总线构造捕获视图（Transfer 通路）。
///
/// 【为什么放在本文件而不是 `transfer/`】读输入总线是 **VST3 ABI** 的知识：
/// `AudioBusBuffers` 的字段布局、`silence_flags` 的位序、inactive plane 可以为 null。
/// `transfer::capture` 刻意不认识这些（它只面对 `InputBlock`），这样捕获逻辑可以
/// 在完全不碰宿主 ABI 的情况下单测。
///
/// 【为什么不看 `silence_flags`】SDK 原文：该位"置位表示每个样本都是 0"，
/// **且置位时缓冲仍必须指向有效内存**；它是给插件省算力的**可选**提示
/// （"This flag is optional. A host is free to support it or not."）。
///
/// 此前这里把置位当作"这个声道没有输入"（`planes[channel] = None`），代价是
/// **捕获写零 + 直通写零**：一个把该位当作"未初始化/不关心"的宿主会让插件
/// **录制与输出全静音**，而它其实一直在喂音频。读样本永远是对的（宿主保证可读），
/// 跳过只省一次拷贝 —— 用"可能全哑"换一次拷贝不值得。
///
/// # Safety
/// 与 [`validate`] 相同的前提：`data` 指向本回调独占的合法 SDK 结构，且输入总线的
/// 每个非空 plane 至少有 `num_samples` 帧可读。**返回的切片只在本次回调内有效**，
/// 调用方不得把它们存起来或带出回调。
///
/// `playing` 由调用方从 `ProcessContext::state` 的 `kPlaying` 位给出（见
/// `crate::audio_abi::PROCESS_CONTEXT_PLAYING`）：它与位置、采样率、样本同属**同一个块**，
/// 因此在这里一次性装进视图，而不是让每个消费方各自去读一遍上下文。
pub(crate) unsafe fn capture_view<'a>(
    data: *const ProcessData,
    start: i64,
    sample_rate: f64,
    playing: bool,
) -> crate::transfer::capture::InputBlock<'a> {
    // SAFETY: 调用方保证 data 在本调用期间可读。
    let data = unsafe { &*data };
    let frames = data.num_samples.max(0) as usize;
    let mut planes: [Option<&'a [f32]>; 2] = [None, None];
    if frames > 0 && data.num_inputs > 0 && !data.inputs.is_null() {
        // SAFETY: validate 已确认唯一输入总线非空。
        let bus = unsafe { &*data.inputs };
        if !bus.channel_buffers.is_null() {
            // 只取前两个声道：内部快照恒为立体声（`PlaybackSnapshot` 只有左右），
            // 多声道需要一条显式的下混策略，那是产品决策而不是补一个边界。
            for channel in 0..(bus.num_channels.max(0) as usize).min(2) {
                // SAFETY: channel < min(num_channels, 2) ≤ 声明数量，宿主提供对应指针。
                let plane = unsafe { *bus.channel_buffers.add(channel) };
                // 【唯一算"没有输入"的情形是 plane 为空】那是 SDK 允许的 inactive 声道。
                if plane.is_null() {
                    continue;
                }
                // SAFETY: 宿主声明该 plane 至少有 num_samples 帧（validate 已查形状）。
                planes[channel] = Some(unsafe { std::slice::from_raw_parts(plane, frames) });
            }
        }
    }
    crate::transfer::capture::InputBlock {
        start,
        frames,
        sample_rate,
        planes,
        playing,
    }
}

/// 手写 ABI 结构体的**完整布局清单**（`类型.字段` / `类型.size` / `类型.align` → 字节）。
///
/// 【为什么抽成函数】它是跨平台布局基准的唯一来源：
/// - Windows x86_64 上由 MSVC oracle（`layout_tests`）拿**真 SDK 结构**交叉验证；
/// - 其它平台（Linux / macOS / aarch64）与提交的 JSON 基准逐项比对。
///
/// 此前布局断言被 `cfg(target_os = "windows", target_arch = "x86_64")` 整块挡住 ——
/// 手写 `#[repr(C)]` 结构在 Linux/macOS 上的字段漂移**完全没守**。这些结构只含
/// i32/i64/u32/u64/f64/指针，没有 C `long`，因此 64 位 LP64/LLP64 平台布局一致，
/// 可以共用一份基准。
#[cfg(test)]
fn abi_layout_entries() -> Vec<(&'static str, usize)> {
    use crate::vst3::ProcessSetup;
    use std::mem::{align_of, offset_of, size_of};

    macro_rules! kind {
        ($ty:ty) => {{
            vec![
                (concat!(stringify!($ty), ".size"), size_of::<$ty>()),
                (concat!(stringify!($ty), ".align"), align_of::<$ty>()),
            ]
        }};
    }
    macro_rules! field {
        ($ty:ty, $rust:ident, $native:ident) => {
            (
                concat!(stringify!($ty), ".", stringify!($native)),
                offset_of!($ty, $rust),
            )
        };
    }
    let mut entries: Vec<(&'static str, usize)> = Vec::new();
    entries.extend(kind!(ProcessSetup));
    entries.push(field!(ProcessSetup, process_mode, processMode));
    entries.push(field!(
        ProcessSetup,
        symbolic_sample_size,
        symbolicSampleSize
    ));
    entries.push(field!(
        ProcessSetup,
        max_samples_per_block,
        maxSamplesPerBlock
    ));
    entries.push(field!(ProcessSetup, sample_rate, sampleRate));
    entries.extend(kind!(AudioBusBuffers));
    entries.push(field!(AudioBusBuffers, num_channels, numChannels));
    entries.push(field!(AudioBusBuffers, silence_flags, silenceFlags));
    entries.push(field!(AudioBusBuffers, channel_buffers, channelBuffers32));
    entries.push(field!(AudioBusBuffers, channel_buffers, channelBuffers64));
    entries.extend(kind!(ProcessData));
    entries.push(field!(ProcessData, process_mode, processMode));
    entries.push(field!(
        ProcessData,
        symbolic_sample_size,
        symbolicSampleSize
    ));
    entries.push(field!(ProcessData, num_samples, numSamples));
    entries.push(field!(ProcessData, num_inputs, numInputs));
    entries.push(field!(ProcessData, num_outputs, numOutputs));
    entries.push(field!(ProcessData, inputs, inputs));
    entries.push(field!(ProcessData, outputs, outputs));
    entries.push(field!(
        ProcessData,
        input_parameter_changes,
        inputParameterChanges
    ));
    entries.push(field!(
        ProcessData,
        output_parameter_changes,
        outputParameterChanges
    ));
    entries.push(field!(ProcessData, input_events, inputEvents));
    entries.push(field!(ProcessData, output_events, outputEvents));
    entries.push(field!(ProcessData, process_context, processContext));
    entries.extend(kind!(ProcessContext));
    entries.push(field!(ProcessContext, state, state));
    entries.push(field!(ProcessContext, sample_rate, sampleRate));
    entries.push(field!(
        ProcessContext,
        project_time_samples,
        projectTimeSamples
    ));
    entries.push(field!(ProcessContext, system_time, systemTime));
    entries.push(field!(
        ProcessContext,
        continuous_time_samples,
        continousTimeSamples
    ));
    entries.push(field!(ProcessContext, project_time_music, projectTimeMusic));
    entries.push(field!(ProcessContext, bar_position_music, barPositionMusic));
    entries.push(field!(ProcessContext, cycle_start_music, cycleStartMusic));
    entries.push(field!(ProcessContext, cycle_end_music, cycleEndMusic));
    entries.push(field!(ProcessContext, tempo, tempo));
    entries.push(field!(ProcessContext, time_sig_numerator, timeSigNumerator));
    entries.push(field!(
        ProcessContext,
        time_sig_denominator,
        timeSigDenominator
    ));
    entries.push(field!(ProcessContext, chord, chord));
    entries.push(field!(
        ProcessContext,
        smpte_offset_subframes,
        smpteOffsetSubframes
    ));
    entries.push(field!(ProcessContext, frame_rate, frameRate));
    entries.push(field!(
        ProcessContext,
        samples_to_next_clock,
        samplesToNextClock
    ));
    entries.extend(kind!(Chord));
    entries.push(field!(Chord, key_note, keyNote));
    entries.push(field!(Chord, root_note, rootNote));
    entries.push(field!(Chord, chord_mask, chordMask));
    entries.extend(kind!(FrameRate));
    entries.push(field!(FrameRate, frames_per_second, framesPerSecond));
    entries.push(field!(FrameRate, flags, flags));
    entries
}

/// 跨平台布局门禁：任何 64 位目标上，手写结构的每个字段偏移都必须与提交的基准一致。
///
/// 【为什么要有它，而不是只留 MSVC oracle】oracle 只在 Windows x86_64 跑；Linux/macOS
/// 的手写 ABI 一旦漂移，CI 上没有任何一条断言会失败 —— 直到用户在这些平台上撞到
/// 内存错乱。这份 JSON 基准把"已经验过的布局"固化下来，全平台都能守。
#[cfg(test)]
mod abi_baseline {
    use super::abi_layout_entries;

    fn baseline() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/abi-layout.json"))
            .expect("committed ABI layout baseline must be valid JSON")
    }

    #[test]
    fn the_hand_written_abi_layout_matches_the_committed_baseline() {
        // 32 位目标的指针宽度不同，布局另有一套；这里只守 64 位（CI 的目标平台）。
        if std::mem::size_of::<usize>() != 8 {
            return;
        }
        let baseline = baseline();
        for (name, value) in abi_layout_entries() {
            let expected = baseline[name]
                .as_u64()
                .unwrap_or_else(|| panic!("baseline is missing ABI entry {name}"));
            assert_eq!(
                expected as usize, value,
                "ABI {name} drifted from the committed baseline; if the change is \
                 intentional, re-verify against the MSVC oracle and update \
                 tests/fixtures/abi-layout.json"
            );
        }
    }

    /// 生成基准文件的一次性助手（`HFS_WRITE_ABI_BASELINE=1 cargo test ...`）。
    #[test]
    fn write_abi_baseline_when_requested() {
        if std::env::var_os("HFS_WRITE_ABI_BASELINE").is_none() {
            return;
        }
        let mut map = serde_json::Map::new();
        for (name, value) in abi_layout_entries() {
            map.insert(name.to_string(), serde_json::json!(value));
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/abi-layout.json"
        );
        std::fs::write(path, serde_json::to_string_pretty(&map).unwrap() + "\n").unwrap();
    }
}

#[cfg(all(test, target_os = "windows", target_arch = "x86_64"))]
mod layout_tests {
    use super::*;
    use crate::vst3::ProcessSetup;
    use std::mem::{align_of, offset_of, size_of};
    use std::path::PathBuf;
    use std::process::Command;

    /// oracle 需要"普通 `cargo test` shell 里没有"的两样东西，以及怎么拿到它们。
    ///
    /// 【为什么是硬失败而不是跳过】oracle 是手写 ABI 的唯一真值来源；环境不齐时
    /// 静默跳过 = "绿在没跑"，与 `check-ffi-guards.py` 曾经"门禁恒过"是同一类错误。
    const ORACLE_ENV_HELP: &str = "\
the native ABI oracle needs the MSVC toolchain and the pinned VST3 SDK:
  cl.exe on PATH    <-  . ./scripts/env/msvc-env.ps1
  ARA_VST3_SDK_DIR  <-  . ./scripts/env/sdk-env.ps1; Set-SdkEnvironment
then, from the repo root in PowerShell:
  cargo test --manifest-path backend/Cargo.toml -p hifishifter-plugin --lib
CI sets both before running the tests.";

    /// 原生 SDK 编译器为布局 oracle；任何字段偏移/对齐漂移都必须阻断测试。
    #[test]
    fn rust_audio_layout_matches_the_locked_native_sdk() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let scratch = manifest.join("../../.build-tmp/vst3-layout");
        std::fs::create_dir_all(&scratch).unwrap();
        let sdk =
            std::env::var_os("ARA_VST3_SDK_DIR").unwrap_or_else(|| panic!("{}", ORACLE_ENV_HELP));
        let exe = scratch.join("vst3-layout.exe");
        let compile = Command::new("cl.exe")
            .current_dir(&scratch)
            .args(["/nologo", "/std:c++17", "/EHsc", "/utf-8"])
            .arg(format!("/I{}", PathBuf::from(sdk).display()))
            .arg(format!("/Fo{}", scratch.join("vst3-layout.obj").display()))
            .arg(format!("/Fe{}", exe.display()))
            .arg(manifest.join("tests/vst3_layout.cpp"))
            .output()
            .unwrap_or_else(|e| panic!("{}\n\nspawning cl.exe failed: {}", ORACLE_ENV_HELP, e));
        assert!(
            compile.status.success(),
            "native ABI oracle compile failed: {:?}\n\n{}",
            compile,
            ORACLE_ENV_HELP
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
