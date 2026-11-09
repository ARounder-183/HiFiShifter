//! CLAP 音频回调的语义：一块宿主音频 → 共享渲染通路。
//!
//! 【为什么只有这一条通路】“这一块该出什么声”的每一个分支（捕获观察、相位判定、
//! 门控、快照输出、离线降级）都在 `ExtensionOwner::process_audio` 里，与 VST3 同一份。
//! 外壳在这里只做**翻译**：把 `clap_process_t` 的缓冲与 transport 变成
//! `AudioBlock` + `Option<BlockContext>`，再把结局映射成 CLAP 的返回码。
//!
//! 【为什么 Capture 形态也走这里】它此前走的是另一套（`CaptureHost::on_process` +
//! 独立的 `RenderEngine`）：同一块音频有两条记账、两份时间线、两个渲染器。那套已经
//! 删除 —— 两种形态的差别只在“有没有 ARA 文档”，不在“音频怎么处理”。
//!
//! 【为什么时间事实是 `Option`】`clap_process_t::transport` 允许为空。空 transport
//! 与“工程第 0 帧”是两件不同的事：前者是“宿主没告诉我们这一块在哪”，后者是一个
//! 合法位置。共享层按 `None` 处理前者（实时静音、离线报 NotReady），与 VST3 的
//! `process_context == nullptr` 完全同一条判据 —— 把空 transport 读成位置 0 会让
//! 离线块从快照开头取音频。

use crate::abi::clap_process_t;
use hifishifter_plugin::render::extension::RenderOutcome;

/// 处理一块宿主音频；返回共享通路给出的结局。
///
/// 【为什么精度先判】64-bit 平面读不了：`data32` 为空 ⇒ 中立块只有零个声道。那正是
/// “静音但在场”的降级，而精度必须先记账 —— 界面据此说清“为什么是静音”
/// （`CaptureStore::unsupported_precision`）。与 VST3 的
/// `silence_unsupported_sample_size` 是同一条取舍。
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
    let precision = crate::audio::sample_precision(process);
    if precision != 0 {
        instance.owner.begin_process_block(precision);
        // SAFETY: 宿主在本次回调里给出的有效记录；非空平面至少含 frames 帧。
        unsafe { crate::audio::silence_outputs(process) };
        return RenderOutcome::Done;
    }
    // 【为什么多总线要拒绝】见 `audio::output_layout_supported`：没被 `AudioBlock`
    // 覆盖的输出总线会留着宿主缓冲里的残留样本。VST3 对同一形状回
    // `UnsupportedFormat`，这里给出同一个结局。
    if !crate::audio::output_layout_supported(process) {
        // SAFETY: 同上。
        unsafe { crate::audio::silence_outputs(process) };
        return RenderOutcome::Unsupported;
    }
    // 进程形态：CLAP 只有“离线 render 模式”这一个权威来源（见 `AudioState::is_offline`）。
    let mode = if instance.audio.is_offline() { 2 } else { 0 };
    let mut silence = 0_u64;
    // SAFETY: 宿主在本次回调里给出的有效记录；非空平面至少含 frames_count 帧。
    let mut block = unsafe { crate::audio::block_from_clap(process, mode, &mut silence) };
    // SAFETY: 与 block_from_clap 同一份宿主缓冲，且只在本次回调内使用。
    unsafe {
        instance
            .owner
            .process_audio(&mut block, instance.audio.block_context(process), precision)
    }
}
