//! HiFiShifter 的离线音频内核。
//!
//! 这里放**不依赖 Tauri、也不依赖音频设备**的那部分：时间线模型、混音、拉伸、
//! 渲染缓存、声码器、音高分析与编辑。ARA 插件侧（`hifishifter-plugin`）只依赖这一层，
//! 于是插件不会把 WebView2 / Tauri 运行时带进 DAW 进程。
//!
//! ## 边界是怎么定下来的
//!
//! 不是按"依赖闭包"拍的，而是**实测**出来的：从 `mixdown` 出发做 `crate::` 引用闭包、
//! 剥离 `#[cfg(test)]` 里的引用、排除设备层 `audio_engine`，得到 **41 个模块**；
//! 并且这 41 个模块的**生产代码里一处 `tauri::` 都没有**。
//!
//! 定这条边靠两处改造（都在 app 层做掉，见各自的提交说明）：
//! - [`host`]：内核 worker 要触发宿主侧的后台渲染时走回调，而不是直接够
//!   `commands::playback`；
//! - [`events::events`]：UI 事件走进程级出口，调用点不再持有 `tauri::AppHandle`。
//!
//! 留在 app 侧的是：`commands`（IPC 包装）、`audio_engine` 的设备半边
//! （`engine.rs` 的 cpal 流、`snapshot.rs`）、剪贴板/搜索/录制等宿主功能，
//! 以及 `tauri` 本身。
//!
//! ## 与 app 的关系
//!
//! 内核是 app 的依赖。app 侧 `backend/src-tauri/src/lib.rs` 用 `pub use`
//! 把这些模块再接回 crate 根，于是 app 里 `crate::mixdown::…` 这类**既有路径完全不变**。

// ── 出口与基础设施 ──────────────────────────────────────────────────────────
pub mod byte_budget_cache;
pub mod engine_command;
pub mod events;
pub mod host;
pub mod log_limiter;
pub mod model_paths;
pub mod util;

// 模型路径的读取函数直接挂到 crate 根：声码器与 FCPE 模块里写的是
// `crate::nsf_hifigan_model_dir()` 这类调用，挂到根上就不必改那些调用点。
pub use model_paths::{fcpe_onnx_path, hnsep_model_dir, nsf_hifigan_model_dir};

/// 热路径调试日志：经 `log::debug!` 走统一日志管线，release 默认 Info 级别下
/// 不产生格式化开销；需要诊断时用 `HIFISHIFTER_LOG=debug` 临时打开。
///
/// 【为什么内核也要有这份】这些宏原先定义在 app 的 crate 根，而调用点遍布内核模块
/// （渲染器、音高分析、缓存）。宏是文本作用域的，模块搬走后必须跟着搬 —— 或者像
/// [`crate::log_limiter`] 那样把实现放进内核。两个都做了：宏在这里重定义，
/// 限流的实现也搬进了内核。
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        log::debug!($($arg)*);
    }
}

/// 限流警告：同一调用点每 10 秒最多输出一条，窗口内被抑制的条数会在
/// 下一条输出前以 `[throttled]` 汇总行补记。
macro_rules! log_warn_limited {
    ($($arg:tt)*) => {
        $crate::log_limiter::emit_limited(
            log::Level::Warn,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

/// 限流错误：语义同 [`log_warn_limited!`]，级别为 error。
macro_rules! log_error_limited {
    ($($arg:tt)*) => {
        $crate::log_limiter::emit_limited(
            log::Level::Error,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

// ── 时间线模型 ──────────────────────────────────────────────────────────────
pub mod state;
pub mod editor;

// ── 曲线与参数 ──────────────────────────────────────────────────────────────
pub mod fade_curves;
pub mod models;
pub mod pitch_config;
pub mod vibrato;

// ── 时间拉伸（原生构建见 build.rs）────────────────────────────────────────────
pub mod soundtouch;
pub mod sstretch;
pub mod time_stretch;

// ── 解码 / 编码 / 媒体 ──────────────────────────────────────────────────────
pub mod audio_utils;
pub mod encode;
pub mod media;
pub mod midi_import;

// ── 渲染与缓存 ──────────────────────────────────────────────────────────────
pub mod mixdown;
pub mod render_cache;
pub mod render_key;
pub mod renderer;
pub mod synth_clip_cache;

// ── 声码器 ──────────────────────────────────────────────────────────────────
pub mod formant_cache;
pub mod formant_morph;
pub mod glottal_rd;
pub mod hnsep_dsp;
pub mod rd_tension;
pub mod streaming_world;
pub mod world_vocoder;

#[cfg(feature = "onnx")]
pub mod mel_utils;

// `nsf_hifigan_onnx` / `hnsep_onnx` / `fcpe_onnx` 在关闭 `onnx` feature 时用各自的 stub。
// 这里用 `#[path]` 直接顶替模块名（app 原来的写法是另起一个名字再 `use ... as ...`），
// 好处是**调用点完全不需要 cfg**。
#[cfg(feature = "onnx")]
pub mod nsf_hifigan_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "nsf_hifigan_onnx_stub.rs"]
pub mod nsf_hifigan_onnx;

#[cfg(feature = "onnx")]
pub mod hnsep_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "hnsep_onnx_stub.rs"]
pub mod hnsep_onnx;

#[cfg(feature = "onnx")]
pub mod fcpe_onnx;
#[cfg(not(feature = "onnx"))]
#[path = "fcpe_onnx_stub.rs"]
pub mod fcpe_onnx;

#[cfg(feature = "onnx")]
pub mod vocoder_ort_session;

// GPU 信息 / DirectML 适配器：Windows 有真实现，别的平台用 stub。
#[cfg(target_os = "windows")]
pub mod gpu_info;
#[cfg(not(target_os = "windows"))]
#[path = "gpu_info_stub.rs"]
pub mod gpu_info;

#[cfg(target_os = "windows")]
pub mod dml_adapters;
#[cfg(not(target_os = "windows"))]
#[path = "dml_adapters_stub.rs"]
pub mod dml_adapters;

// ── 音高分析 / 编辑 ─────────────────────────────────────────────────────────
pub mod pitch_analysis;
pub mod pitch_clip;
pub mod pitch_editing;
pub mod streaming_pitch;

// ── 通道与声道策略 ──────────────────────────────────────────────────────────
pub mod channel_decision;
pub mod channel_mode;
pub mod channel_policy;
pub mod stereo_detect;

// ── 工程与杂项 ──────────────────────────────────────────────────────────────
pub mod config;
pub mod clip_rendering_state;
pub mod notebook_assets;
pub mod project;
pub mod temp_manager;

// ── 可选的原生声码器（闭源、仅 Windows）────────────────────────────────────
#[cfg(all(feature = "vslib", target_os = "windows"))]
pub mod vslib;

// ── 节拍器（`EngineCommand` 的载荷类型在这里）──────────────────────────────
pub mod metronome;

// ── 测试专用的分配计量 ──────────────────────────────────────────────────────
//
// 【为什么内核也要有一份】`streaming_pitch` 那条"峰值工作集与素材长度解耦"的验收
// 只能靠真的量分配量来验证 —— 比较输出曲线或帧数完全覆盖不到这一点。
// `#[global_allocator]` 是**按 crate 生效**的：内核的测试二进制需要自己的那一份，
// 借不到 app 的。两边实现逐字相同。
//
// 只在测试构建下替换分配器，发布二进制走系统分配器，无任何运行时开销。
#[cfg(test)]
pub mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    pub struct CountingAllocator;

    impl CountingAllocator {
        fn record_add(bytes: usize) {
            let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        fn record_sub(bytes: usize) {
            LIVE.fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                Self::record_add(layout.size());
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            Self::record_sub(layout.size());
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                Self::record_add(layout.size());
            }
            ptr
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                Self::record_sub(layout.size());
                Self::record_add(new_size);
            }
            new_ptr
        }
    }

    /// 测量 `f` 执行期间**新增的峰值常驻分配**（字节）。
    ///
    /// 基线取进入 `f` 时刻的存活字节数，因此结果只反映 `f` 自身的开销。
    /// 【前置条件】计数器是进程级的，调用期间不能有别的线程在分配 —— 使用者必须
    /// 串行运行（见 `streaming_pitch` 里调用处的 `#[ignore]` 说明与 `--test-threads=1`）。
    pub fn measure_peak_alloc<T>(f: impl FnOnce() -> T) -> (T, usize) {
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        let out = f();
        let peak = PEAK.load(Ordering::Relaxed);
        (out, peak.saturating_sub(base))
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: alloc_probe::CountingAllocator = alloc_probe::CountingAllocator;
