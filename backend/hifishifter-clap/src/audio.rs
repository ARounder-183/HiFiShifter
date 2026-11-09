//! CLAP 缓冲到中性 `AudioBlock` 的适配。
//!
//! 【为什么不照搬 VST3 的 `silence_flags`】CLAP 的 `constant_mask` 是「该声道全块为常量」
//! 的提示，不是「全静音且缓冲仍可读」；把提示当门控会让录音与输出一起静音。这里一律
//! **总是读、总是写满**：只处理 F32，F64 保守静音并返回成功；in-place 时先读后写。

use crate::abi::{clap_process_t, transport_flags, CLAP_RENDER_OFFLINE, CLAP_SECTIME_FACTOR};
use hifishifter_plugin::audio_block::{AudioBlock, InputBus, OutputBus};
use hifishifter_plugin::block_context::BlockContext;
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

    /// 取本实例的采样率；未激活时是 0（共享层会按"没有有效采样率"处理该块）。
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

    /// 把 CLAP 的时间读数翻译成**共享的** `BlockContext`；宿主没给 transport 时返回 `None`。
    ///
    /// 【为什么不再有第二个中性时间类型】本文件此前还构造过 `TransportFrame`
    /// （`hifishifter-plugin-core::format`），于是"一个块的时间事实"在仓库里有两份描述：
    /// VST3 用 `BlockContext::from_vst3`，CLAP 用 `TransportFrame` 再转一次 `BlockContext`。
    /// 后者只有 CLAP 一个消费者，却要单独维护 `position_trusted` / `playing: Option<bool>`
    /// 这些前者没有的概念 —— 两个格式因此不可能在"这一块在哪、在不在播"上真正共用判断。
    /// 现在只保留 `BlockContext`：它是渲染与捕获**唯一**的时间输入。
    ///
    /// 【`None` 与"位置 0"是两件事】`clap_process_t::transport` 允许为空，VST3 的
    /// `ProcessContext` 同样可以为空。两边都映射成 `None`，共享层据此按"没有时间事实"
    /// 处理（实时静音、离线报 NotReady）。把空 transport 读成"工程第 0 帧"会让离线块
    /// 从快照开头取音频 —— 那是数据正确性失败，不是降级。
    pub fn block_context(&self, process: &clap_process_t) -> Option<BlockContext> {
        // SAFETY: `process` 由宿主在本次回调内给出；transport 指针要么为空，要么指向
        // 一个在回调期间有效的 `clap_event_transport_t`。
        let transport = unsafe { process.transport.as_ref() }?;
        let sample_rate = self.sample_rate();
        let flags = transport.flags;
        let has_beats = flags & transport_flags::HAS_BEATS_TIMELINE != 0;
        let has_tempo = flags & transport_flags::HAS_TEMPO != 0;
        let has_time_signature = flags & transport_flags::HAS_TIME_SIGNATURE != 0;
        let beats = |value: i64| value as f64 / crate::abi::CLAP_BEATTIME_FACTOR;
        Some(BlockContext {
            sample_rate,
            project_time_samples: transport_position(transport, sample_rate).unwrap_or(0),
            playing: flags & transport_flags::IS_PLAYING != 0,
            recording: flags & transport_flags::IS_RECORDING != 0,
            // 【为什么恒为 None】`steady_time` 是"回调的稳态计数"，不是工程位置；
            // 共享层把它当"距锚点推进了多少帧"用，而宿主可以合法地让它不随块长增长
            // （或干脆不给），那会让位置推进停住。位置推进由 music+tempo 与块计数器
            // 两条既有通路负责，与 VST3 用的是同一段代码。
            continuous: None,
            music: has_beats.then(|| beats(transport.song_pos_beats)),
            tempo: (has_tempo && transport.tempo.is_finite() && transport.tempo > 0.0)
                .then_some(transport.tempo),
            // 【为什么恒为 None】CLAP 没有系统时间字段（见 `BlockContext::system_time`）。
            system_time: None,
            time_signature: (has_time_signature
                && transport.tsig_num > 0
                && transport.tsig_denom > 0)
                .then_some((
                    i32::from(transport.tsig_num),
                    i32::from(transport.tsig_denom),
                )),
            // 【为什么循环的"有效"看拍轴】起止点是拍时间轴上的值；宿主没给拍轴时那两个
            // 数没有意义。循环**正在走**另由 `IS_LOOP_ACTIVE` 回答（`cycle_active`）。
            cycle_valid: has_beats,
            cycle_active: flags & transport_flags::IS_LOOP_ACTIVE != 0,
            cycle_start_music: if has_beats {
                beats(transport.loop_start_beats)
            } else {
                0.0
            },
            cycle_end_music: if has_beats {
                beats(transport.loop_end_beats)
            } else {
                0.0
            },
        })
    }
}

/// 本块样本宽度的权威读数：0 = 32-bit 浮点，非 0 = 本插件读不了的宽度。
///
/// 【为什么按输出判】CLAP 的 `clap_audio_buffer_t` 用 `data32` / `data64` 两个指针
/// 表达宽度：`data32` 为空而 `data64` 非空就是 64-bit。输出是"我们要写的那一面"，
/// 它决定了这一块能不能产出 —— 输入即使给了 32-bit，输出是 64-bit 也写不进去。
/// 与 VST3 的 `symbolic_sample_size` 同一条事实、同一个取值约定。
pub fn sample_precision(process: &clap_process_t) -> i32 {
    if process.audio_outputs_count == 0 || process.audio_outputs.is_null() {
        return 0;
    }
    // SAFETY: 宿主声明了 audio_outputs_count 个总线，第一个在本回调期间有效。
    let output = unsafe { &*process.audio_outputs };
    if output.data32.is_null() && !output.data64.is_null() {
        1
    } else {
        0
    }
}

/// 把输出平面清零：读不了这种宽度（或形状不受支持）时，宿主仍要拿到一块**确定的**静音。
///
/// 【为什么不直接返回错误】与 VST3 的 `silence_unsupported_sample_size` 同一条取舍：
/// 返回失败会让多数宿主把插件整条踢出链路（用户看到"插件不见了"）；静音但在场是
/// 可诊断的降级，界面会通过 `CaptureStore::unsupported_precision` 说清原因。
///
/// 【为什么两种宽度都清】64-bit 块里 `data32` 按定义为空，清它没有任何副作用；而
/// "总线形状不受支持"（`audio_outputs_count > 1`）时两个指针都可能非空 —— 只清其中
/// 一种会把另一种宽度的垃圾留在宿主缓冲里。一个函数、一条判据。
///
/// # Safety
/// `process` 必须是宿主在本次回调里给出的有效记录；非空平面至少含 `frames_count` 帧可写。
pub unsafe fn silence_outputs(process: &clap_process_t) -> bool {
    let frames = process.frames_count as usize;
    if process.audio_outputs_count == 0 || process.audio_outputs.is_null() {
        return false;
    }
    // SAFETY: 宿主声明了 audio_outputs_count 个总线，均在本回调期间有效。
    let outputs = unsafe {
        std::slice::from_raw_parts(process.audio_outputs, process.audio_outputs_count as usize)
    };
    let mut cleared = false;
    for output in outputs {
        for channel in 0..output.channel_count as usize {
            // SAFETY: 非空指针数组至少含 channel_count 个有效指针（CLAP 契约）。
            let f32_plane = unsafe {
                if output.data32.is_null() {
                    std::ptr::null_mut()
                } else {
                    *output.data32.add(channel)
                }
            };
            let f64_plane = unsafe {
                if output.data64.is_null() {
                    std::ptr::null_mut()
                } else {
                    *output.data64.add(channel)
                }
            };
            if !f32_plane.is_null() {
                // SAFETY: 非空平面至少含 frames 帧可写。
                unsafe { std::slice::from_raw_parts_mut(f32_plane, frames) }.fill(0.0);
                cleared = true;
            }
            if !f64_plane.is_null() {
                // SAFETY: 同上。
                unsafe { std::slice::from_raw_parts_mut(f64_plane, frames) }.fill(0.0);
                cleared = true;
            }
        }
    }
    cleared
}

/// 本块的输出总线形状是否是我们声明的那一种（`0` = 没有输出，`1` = 单条立体声输出）。
///
/// 【为什么需要判】端口表只声明一条输出（`crate::ports`），多出来的总线我们既不知道
/// 该写什么、也不会被 `AudioBlock` 覆盖 —— 留在那里的会是宿主缓冲里的残留样本。
/// VST3 的 `audio_abi::validate` 对同一个形状回 `UnsupportedFormat`，两个格式必须
/// 给出同一个结论。
pub fn output_layout_supported(process: &clap_process_t) -> bool {
    process.audio_outputs_count <= 1
}
/// 把 CLAP 的 `process` 数据翻译成共享的中立块。
///
/// 【为什么翻译放在格式外壳】`clap_audio_buffer_t` 的字段布局、`data32`/`data64` 的
/// 联合体语义、`channel_count` 与指针数组的对应关系，都是 **CLAP ABI** 的知识。
/// 共享渲染层刻意不认识它们 —— 它只面对"平面在哪、有几帧、是不是 in-place"。
///
/// 【为什么没有可写输出时也返回一个块】"宿主没接我们的输出"与"这一块该不该录"是
/// 两件无关的事（见 `ExtensionOwner::capture_observe_block` 的说明）。返回 `None`
/// 会让观察整段跳过 ⇒ 录制静音、`canCapture` 永远灰着。这里返回一个**零声道**的
/// 输出视图：观察照常发生，渲染侧写零个声道（`AudioBlock` 的每个写入点都按
/// `output.channels()` 循环）。
///
/// 【为什么 in-place 要看"同一批平面指针"】CLAP 没有 VST3 那种"同一份总线结构"的
/// 概念，宿主只保证输入与输出的平面指针**可能相同**。逐声道比对指针是这里唯一可靠的
/// 判据（而它只影响静音掩码这条可选提示）。
///
/// # Safety
/// `process` 必须是宿主在本次回调里给出的有效记录；非空平面至少含 `frames_count` 帧。
pub unsafe fn block_from_clap<'a>(
    process: &'a clap_process_t,
    mode: i32,
    silence: &'a mut u64,
) -> AudioBlock<'a> {
    let frames = process.frames_count as usize;
    let input = if process.audio_inputs_count > 0 && !process.audio_inputs.is_null() {
        // SAFETY: 宿主声明了 audio_inputs_count 个总线，第一个在本回调期间有效。
        let bus = unsafe { &*process.audio_inputs };
        unsafe { InputBus::new(bus.data32, bus.channel_count as usize) }
    } else {
        InputBus::empty()
    };
    let (pointers, in_place) =
        if process.audio_outputs_count > 0 && !process.audio_outputs.is_null() {
            // SAFETY: 同上。
            let bus = unsafe { &*process.audio_outputs };
            let pointers: &'a mut [*mut f32] = if bus.channel_count > 0 && !bus.data32.is_null() {
                // SAFETY: 宿主声明了 channel_count 个平面指针，本回调期间有效。
                unsafe { std::slice::from_raw_parts_mut(bus.data32, bus.channel_count as usize) }
            } else {
                &mut []
            };
            let shared = if process.audio_inputs_count > 0 && !process.audio_inputs.is_null() {
                // SAFETY: 同上。
                let inputs = unsafe { &*process.audio_inputs };
                let count = (inputs.channel_count as usize).min(bus.channel_count as usize);
                (0..count).any(|channel| {
                    // SAFETY: channel < min(两个声道路数)，两侧都声明了对应指针。
                    let source = unsafe { *inputs.data32.add(channel) };
                    let target = unsafe { *bus.data32.add(channel) };
                    !source.is_null() && source == target
                })
            } else {
                false
            };
            (pointers, shared)
        } else {
            (&mut [][..], false)
        };
    AudioBlock {
        frames,
        mode,
        input,
        output: OutputBus::new(pointers, silence),
        in_place,
    }
}
/// 校验并直通一个音频块。返回 `false` 表示宿主布局非法（输出必须丢弃）。
///
/// 【为什么这里只剩"翻译 + 直通"】in-place 的"先读齐再写"、单声道下混、inactive 声道
/// 跳过 —— 这些结论现在只有一处实现（`AudioBlock::pass_through`）。
///
/// 【它现在只服务两处】F64 宽度由 `silence_outputs` 单独处理；本函数保留给
/// 直通夹具（`audio/tests.rs`）与"没有可处理内容"的保守分支。真正的播放走
/// `ExtensionOwner::process_audio`。
///
/// # Safety
/// `process` 必须来自宿主的有效 `clap_process_t`，其缓冲在调用期间有效。
pub unsafe fn process_block(process: &clap_process_t) -> bool {
    let frames = process.frames_count as usize;
    if frames == 0 {
        return true;
    }
    // F64 不处理：保守地把输出写零，但仍返回成功（与 VST3 kSample64 同口径）。
    // 【为什么按宽度判而不是看 silence_outputs 的返回值】后者现在也服务"总线形状不受
    // 支持"，两种宽度都会清 —— 拿它当判据会把正常的 32-bit 块当成 64-bit 静音掉。
    if sample_precision(process) != 0 {
        unsafe { silence_outputs(process) };
        return true;
    }
    let mut silence = 0_u64;
    // SAFETY: 由本函数的调用契约保证。
    let mut block = unsafe { block_from_clap(process, 0, &mut silence) };
    // SAFETY: 同上。
    unsafe { block.pass_through() }.is_ok()
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
pub fn transport_position(
    transport: &crate::abi::clap_event_transport_t,
    sample_rate: f64,
) -> Option<i64> {
    let has_seconds = transport.flags & crate::abi::transport_flags::HAS_SECONDS_TIMELINE != 0;
    let has_beats = transport.flags & crate::abi::transport_flags::HAS_BEATS_TIMELINE != 0;
    let has_tempo = transport.flags & crate::abi::transport_flags::HAS_TEMPO != 0;
    let seconds = transport.song_pos_seconds as f64 / CLAP_SECTIME_FACTOR;
    let beats = transport.song_pos_beats as f64 / crate::abi::CLAP_BEATTIME_FACTOR;
    if has_seconds && seconds.is_finite() && seconds >= 0.0 {
        Some((seconds * sample_rate).round() as i64)
    } else if has_beats && has_tempo && beats.is_finite() && beats >= 0.0 && transport.tempo > 0.0 {
        Some((beats * 60.0 / transport.tempo * sample_rate).round() as i64)
    } else {
        None
    }
}
