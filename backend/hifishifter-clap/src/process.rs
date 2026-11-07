//! CLAP 音频回调的语义：一块宿主音频 → 共享渲染通路。
//!
//! 【为什么只有这一条通路】“这一块该出什么声”的每一个分支（捕获观察、相位判定、
//! 门控、快照输出、离线降级）都在 `ExtensionOwner::process_audio` 里，与 VST3 同一份。
//! 外壳在这里只做**翻译**：把 `clap_process_t` 的缓冲与 transport 变成
//! `AudioBlock` + `BlockContext`，再把结局映射成 CLAP 的返回码。
//!
//! 【为什么 Capture 形态也走这里】它此前走的是另一套（`CaptureHost::on_process` +
//! 独立的 `RenderEngine`）：同一块音频有两条记账、两份时间线、两个渲染器。那套已经
//! 删除 —— 两种形态的差别只在“有没有 ARA 文档”，不在“音频怎么处理”。

use crate::abi::clap_process_t;
use hifishifter_plugin::render::extension::RenderOutcome;

/// 处理一块宿主音频；返回共享通路给出的结局。
///
/// 【为什么 `AudioState::to_transport_frame` 只在这里调一次】它推进块计数与读数序号，
/// 每块调两次会让诊断读数翻倍、前端乱序判据提前触发（这个 bug 修过一次）。所以
/// 本块的所有时间事实都从它返回的那一份 `TransportFrame` 派生。
///
/// 【为什么循环边界仍读原始 transport】`TransportFrame` 只带位置与状态，不带循环
/// 起止点；那两个值只有 `clap_event_transport_t` 里有。它们在同一个回调内读，不会
/// 与 `to_transport_frame` 的读数分叉。
///
/// # Safety
/// `plugin` 与 `process` 必须是宿主按 CLAP 契约传入的活记录，且 `process` 的输入
/// 平面至少有 `frames_count` 帧可读。
pub unsafe fn process_audio(
    plugin: *const crate::abi::clap_plugin_t,
    process: *const clap_process_t,
) -> RenderOutcome {
    if plugin.is_null() || process.is_null() {
        return RenderOutcome::Invalid;
    }
    // SAFETY: 由调用方保证。
    let Some(instance) = (unsafe { crate::instance(plugin) }) else {
        return RenderOutcome::Invalid;
    };
    let process = unsafe { &*process };
    let frame = instance.audio.to_transport_frame(process);
    let mut context = hifishifter_plugin::block_context::BlockContext {
        sample_rate: frame.sample_rate,
        project_time_samples: frame.project_time_samples,
        playing: frame.is_playing(),
        recording: frame.is_recording(),
        cycle_active: frame.cycle_active == Some(true),
        tempo: frame.tempo.filter(|tempo| tempo.is_finite() && *tempo > 0.0),
        ..Default::default()
    };
    if let Some(transport) = unsafe { process.transport.as_ref() } {
        if transport.flags & crate::abi::transport_flags::HAS_BEATS_TIMELINE != 0 {
            let factor = crate::abi::CLAP_BEATTIME_FACTOR;
            context.music = Some(transport.song_pos_beats as f64 / factor);
            context.cycle_valid = true;
            context.cycle_start_music = transport.loop_start_beats as f64 / factor;
            context.cycle_end_music = transport.loop_end_beats as f64 / factor;
        }
    }
    // 进程形态：CLAP 只有“离线 render 模式”这一个权威来源（见 AudioState::is_offline）。
    let mode = if frame.offline { 2 } else { 0 };
    let stream = if frame.offline {
        hifishifter_capture::recorder::ProcessStream::Offline
    } else {
        hifishifter_capture::recorder::ProcessStream::Realtime
    };
    let mut silence = 0_u64;
    // SAFETY: 宿主在本次回调里给出的有效记录；平面至少含 frames_count 帧。
    let Some(mut block) = (unsafe { crate::audio::block_from_clap(process, mode, &mut silence) })
    else {
        // 没有可写输出平面（或 F64 宽度）：交给同一条翻译的保守分支 —— 它把 F64
        // 输出清零并如实返回成功，与 VST3 的 kSample64 降级同口径。
        return if unsafe { crate::audio::process_block(process) } {
            RenderOutcome::Done
        } else {
            RenderOutcome::Invalid
        };
    };
    // SAFETY: 与 block_from_clap 同一份宿主缓冲，且只在本次回调内使用。
    unsafe { instance.owner.process_audio(&mut block, context, stream, None) }
}
