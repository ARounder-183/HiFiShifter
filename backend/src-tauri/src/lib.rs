// ── clippy 策略 ─────────────────────────────────────────────────────────────
//
// 【为什么要有这一块】CI 曾用 `RUSTFLAGS: "-Awarnings"` 把**整个 workspace 的
// 告警一起静音**，于是 190 条 clippy 告警长期不可见（其中若干是死代码与恒真断言）。
// 那个做法已移除；取而代之的是把"有意识接受的风格类 lint"逐条写在这里，让
// **缺陷类** lint 保持默认告警级别、不再被噪声淹没。
//
// 每一条都是判断，不是省事：
#![allow(
    // 音频/DSP 的许多函数天然需要一组同源的参数（缓冲、长度、采样率、增益…）。
    // 硬抽成结构体只会把调用点变成构造字面量，不增加任何表达力。
    clippy::too_many_arguments,
    // 同上：闭包/迭代器签名的类型确实复杂，但抽 `type` 别名会让"它到底是什么"
    // 更难读（尤其是 `impl Fn` 嵌套）。逐处抽别名的收益不抵可读性损失。
    clippy::type_complexity,
    // `!(x > 0.0)` 这类写法**是刻意的 NaN 语义**：NaN 参与比较一律为 false，
    // 取反后为 true，正是"非法值按不活跃处理"想要的结果。改成 `partial_cmp`
    // 会把一行直白的表达式变成三段匹配，更容易写错。
    clippy::neg_cmp_op_on_partial_ord,
    // 逐元素数值循环里索引是**语义的一部分**（相邻样本、跨通道步长、原地读写），
    // 改成迭代器往往要引入 `windows()` / `zip()` 反而更难核对边界。
    clippy::needless_range_loop,
    // `let mut x = X::default(); x.a = 1;` 与结构体更新语法等价；后者在字段多、
    // 默认值集中定义时更难读（要来回对照 Default 实现）。
    clippy::field_reassign_with_default,
    // 文档续行缩进是纯排版；本仓注释以中文长句为主，rustfmt/clippy 的续行规则
    // 与中文标点的组合并不总是更易读。
    clippy::doc_lazy_continuation,
    // `if d != 0 { a / d }` 与 `a.checked_div(d)` 等价；前者在数值代码里与
    // 周围的显式边界判断同一风格。
    clippy::manual_checked_ops,
    // `len()` 无 `is_empty()`：这些缓存结构不实现 `Default`/`is_empty` 语义，
    // 补一个 `is_empty` 只是为了满足 lint。
    clippy::len_without_is_empty,
)]

/// 热路径调试日志：经 `log::debug!` 走统一日志管线，release 默认 Info 级别下
/// 不产生格式化开销；需要诊断时用 `HIFISHIFTER_LOG=debug` 临时打开。
/// 定义在 crate 根，供所有子模块（引擎 worker、snapshot、pitch 等）使用。
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        log::debug!($($arg)*);
    }
}

/// 限流错误：语义同 [`log_warn_limited!`]，级别为 error。
macro_rules! log_error_limited {
    ($($arg:tt)*) => {
        $crate::logging::emit_limited(log::Level::Error, file!(), line!(), format_args!($($arg)*))
    };
}

#[cfg(test)]
mod build_git;
// 跨 app / 内核边界的测试：它们要同时看见两侧，所以住在 app。
// 详见各自的文件头注释。
mod build_info;
#[cfg(test)]
mod project_tests;
#[cfg(test)]
mod renderer_cross_checks;
// app 层的事件出口：把 Tauri 的 `AppHandle` 包成内核认识的 `EventSink`，
// 并给 app 层自己留一个 `AppHandle` 出口。内核模块不再直接引用 `tauri::`。
mod app_events;
pub mod logging;
mod zip_util;

mod ara_bridge;
mod audio_engine;
pub(crate) mod commands;
#[path = "audio/hfspeaks_v2.rs"]
mod hfspeaks_v2;
mod launch_args;
mod recording;
mod search;
#[path = "audio/silence_detect.rs"]
mod silence_detect;

// ── 内核模块：接回路径 ──────────────────────────────────────────────────────
//
// 它们已经搬进 `hifishifter-kernel`。这里用 `pub(crate) use` 再接回 crate 根，
// 于是 app 里 `crate::mixdown::…` / `crate::state::Clip` 这类**既有路径完全不变** ——
// 搬迁因此不需要改动任何调用点。可见性也照旧（原本是私有或 `pub(crate)` 的，
// 不会因为搬走而变成对外公开）。
//
// 判据与理由见 `docs/superpowers/specs/2026-10-04-ara-plugin-v1-design.md` §4.2。
pub use hifishifter_kernel::fade_curves;
pub(crate) use hifishifter_kernel::{
    audio_utils, channel_decision, channel_mode, channel_policy, clip_rendering_state, config,
    dml_adapters, encode, formant_cache, gpu_info, media, midi_import, mixdown, models,
    notebook_assets, pitch_analysis, pitch_clip, pitch_editing, project, render_cache, render_key,
    renderer, stereo_detect, synth_clip_cache, temp_manager,
};

// 这几个的可见性跟着自己的 feature / target 走，不能放进上面那个统一的 `use`。
#[cfg(all(feature = "vslib", target_os = "windows"))]
pub(crate) use hifishifter_kernel::vslib;
#[cfg(feature = "onnx")]
pub(crate) use hifishifter_kernel::{fcpe_onnx, hnsep_onnx, nsf_hifigan_onnx};
#[cfg(not(feature = "onnx"))]
pub(crate) use hifishifter_kernel::{fcpe_onnx, hnsep_onnx, nsf_hifigan_onnx};

// ── 测试专用的分配计量 ───────────────────────────────────────────────────────
//
// "导入超长音频内存爆炸"的验收标准是「峰值工作集与素材长度解耦」，这只能靠真的
// 量分配量来验证 —— 比较输出曲线或帧数完全覆盖不到这一点。
//
// 只在测试构建下替换分配器（`#[cfg(test)]`），发布二进制走系统分配器，无任何
// 运行时开销。
#[cfg(test)]
pub(crate) mod alloc_probe {
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
    /// 串行运行（见调用处的 `#[ignore]` 说明与 `--test-threads=1`）。
    ///
    /// 【为什么保留而未删除】它是内存验收的计量设施（见上方 `alloc_probe` 的说明
    /// 与 `docs/` 里的峰值内存验收标准），当前没有调用点是因为依赖它的 `#[ignore]`
    /// 用例尚未回填。删掉它会把这条验收路径一并删掉。
    #[allow(
        dead_code,
        reason = "内存峰值验收的计量设施；调用方（#[ignore] 用例）尚未回填"
    )]
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

#[cfg(target_os = "linux")]
mod linux_clipboard;
mod project_fragment;
#[path = "import/reaper_export.rs"]
mod reaper_export;
#[path = "import/reaper_import.rs"]
mod reaper_import;
#[path = "import/reaper_parser.rs"]
mod reaper_parser;
// `soundtouch` / `sstretch` / `time_stretch` 已迁到 `hifishifter-kernel`。
// 再导出，app 侧 `crate::time_stretch::…` 的路径保持不变。
//
// 【为什么这三个一起走】前两个是后者的原生后端（FFI），而它们的原生构建
// 也已随之内核化 —— 否则插件单独链接内核时会链接失败。设计 §4.9。
pub use hifishifter_kernel::{soundtouch, sstretch, time_stretch};
mod state;
mod system_clipboard;
#[path = "import/vocalshifter_clipboard.rs"]
mod vocalshifter_clipboard;
#[path = "import/vocalshifter_import.rs"]
mod vocalshifter_import;
#[cfg(all(feature = "vslib", target_os = "windows", target_arch = "x86_64"))]
#[cfg(target_os = "windows")]
mod webview2_accelerators;

/// Internal pure-function exports used by integration tests (tests/).
///
/// Kept unconditional (no feature gate): comctl32.dll is delay-loaded via
/// build.rs, so the lib unit-test harness runs natively on Windows and
/// these helpers are only needed by the tests/ integration targets.
#[doc(hidden)]
pub mod __test_internals {
    pub use crate::pitch_clip::trim_and_resample_midi;
    // REAPER export round-trips: rate/multi-take export regressions run via
    // the integration targets (loop_semantics / reaper_export_rates).
    pub use crate::reaper_export::build_reaper_clipboard;
    pub use crate::reaper_parser::parse_clipboard_bytes;
    pub use crate::state::{
        Clip, SplitTransitionDurationUnit, SplitTransitionMode, SplitTransitionOptions,
        TimelineState,
    };

    /// Consumed playback window (forward [ss, ss+len·r) / reverse [se−len·r, se)).
    pub fn playback_window_sec(c: &Clip) -> (f64, f64) {
        crate::state::clip_playback_window_sec(c)
    }

    /// Directional leading silence (forward: window start; reverse: window
    /// end past the media end).
    pub fn leading_silence_sec(c: &Clip, media_total_sec: Option<f64>) -> f64 {
        crate::state::clip_leading_silence_sec(c, media_total_sec)
    }

    /// Window arguments for trim_and_resample_midi (non-loop reverse is
    /// redirected to [se−len·r, se]).
    pub fn pitch_trim_window_sec(c: &Clip) -> (f64, f64) {
        crate::state::clip_pitch_trim_window_sec(c)
    }
}

/// HiFiShifter 离线内核的对外窗口。
///
/// 为什么需要它：ARA 插件（`hifishifter-plugin`）必须**复用**本 crate 已有的
/// 混音 / 拉伸 / 编码内核，而不是另写一套；但 `lib.rs` 里的模块大多是私有的
/// （形如 `#[path = "audio/mixdown.rs"] mod mixdown;`），外部 crate 看不到它们。
///
/// 这个模块**只做再导出**：不改变任何模块的可见性，也不改变任何行为。
/// 列在这里的都是不依赖 Tauri、不依赖 cpal 音频设备的离线入口 —— 与设计文档
/// §5.1「复用 backend_lib 内核（不依赖 Tauri / cpal）」是同一件事。
///
/// 边界：这里**不含**设备边界（`audio_engine` 的 cpal 流）与 IPC 包装
/// （`commands`）；插件侧用宿主回调替代它们。
pub mod kernel {
    pub use crate::encode::OutputSpec;
    pub use crate::mixdown::{render_mixdown_interleaved, MixdownOptions, QualityPreset};
    pub use crate::state::{Clip, TimelineState, Track, TrackParamsState};
    pub use crate::time_stretch::{time_stretch_interleaved, StretchAlgorithm};
}

use tauri::Manager;

// 模型路径的登记处已迁到 `hifishifter-kernel`：声码器与 FCPE 在内核侧，
// 它们用时来读；而"模型装在哪"是宿主才知道的事（app 从 `resource_dir` 找），
// 所以内核只提供只写一次的登记处，不认识 Tauri 的 `path()`。
pub use hifishifter_kernel::model_paths::{
    fcpe_onnx_path, hnsep_model_dir, nsf_hifigan_model_dir, set_fcpe_onnx_path,
    set_hnsep_model_dir, set_nsf_hifigan_model_dir,
};

pub fn nsf_hifigan_onnx_probe() -> Result<String, String> {
    // Probe ONNX model availability.
    #[cfg(feature = "onnx")]
    {
        nsf_hifigan_onnx::probe_load().map(|_| "ok".to_string())
    }
    #[cfg(not(feature = "onnx"))]
    {
        Err("onnx feature disabled".to_string())
    }
}

/// Run the inference-device benchmark and return the serialized results.
/// Used by the in-app benchmark dialog and by the `--benchmark` CLI flag.
pub fn run_vocoder_benchmark_cli() -> Result<String, String> {
    #[cfg(feature = "onnx")]
    {
        let results = nsf_hifigan_onnx::run_benchmark()?;
        serde_json::to_string_pretty(&results)
            .map_err(|e| format!("failed to serialize benchmark results: {e}"))
    }
    #[cfg(not(feature = "onnx"))]
    {
        Err("onnx feature disabled".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(state::AppState::default())
        .manage(ara_bridge::AraBridge::default())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // ── AppImage Mesa/EGL driver path ──────────────────────────
            // When running inside an AppImage, Mesa's libEGL is bundled
            // along with its DRI drivers under usr/lib/dri/.  Tell Mesa
            // where to find them so WebKit2GTK can create its EGL display.
            #[cfg(target_os = "linux")]
            if let Ok(appdir) = std::env::var("APPDIR") {
                let dri_dir = format!("{appdir}/usr/lib/dri");
                if std::path::Path::new(&dri_dir).is_dir() {
                    std::env::set_var("LIBGL_DRIVERS_PATH", &dri_dir);
                    log::warn!("[setup] LIBGL_DRIVERS_PATH={dri_dir}");
                }
            }

            // 打包后的应用：从 resource_dir 查找内嵌的 ONNX 模型
            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("nsf_hifigan");
                let has_model = p.join("pc_nsf_hifigan.onnx").exists()
                    || p.join("pc_nsf_hifigan_coreml.onnx").exists();
                if has_model && p.join("config.json").exists() {
                    let _ = set_nsf_hifigan_model_dir(p);
                }
            }

            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("hnsep");
                if p.join("hnsep.onnx").exists() {
                    let _ = set_hnsep_model_dir(p);
                }
            }

            if let Ok(res_dir) = app.path().resource_dir() {
                let p = res_dir.join("models").join("fcpe").join("fcpe.onnx");
                if p.exists() {
                    let _ = set_fcpe_onnx_path(p);
                }
            }

            let state = app.state::<state::AppState>();

            // 从进程启动参数中解析工程路径（双击文件关联场景）。
            let startup_project_path =
                launch_args::extract_project_path_from_args(std::env::args_os());
            state.set_pending_startup_project_path(startup_project_path);

            // Expose app handle for background workers.
            let _ = state.app_handle.set(app.handle().clone());

            // app 层自己的 AppHandle 出口（窗口/路径/state）+ 内核事件出口。
            app_events::install(app.handle());
            if let Some(sink) = app_events::event_sink() {
                let _ = state.events.set(sink);
            }
            // 内核找宿主的出口（后台渲染开关 / 请求）。必须在任何内核 worker 启动前装配。
            let _ = app_events::install_host_callbacks();

            // 不再把 app_handle 经命令通道送给 audio engine worker ——
            // `EngineCommand` 已是内核的纯数据类型。worker 会在自己的循环里
            // 惰性从 `app_events::app_handle()` 取（见 engine.rs 的说明）。

            // ── 模型会话后台预热（绝不阻塞 UI 线程）────────────────────────
            // 会话构建 + 首次推理烟测在 GPU（DirectML）下需数秒（实测开启
            // DirectML 时启动到可交互 ~4s，CPU 仅 ~1s）。若这些工作发生在 UI
            // 线程 / 前端初始化命令 / 引擎 worker / 快照构建上，前端初始化与
            // 全部 IPC 都会被阻塞。这里只**派发**后台预热线程：UI 立即可交互，
            // 三个模型的会话在后台并行/串行构建，首个渲染或播放到来时通常已
            // 就绪（可用性查询 is_available 为非阻塞乐观语义，不触发构建）。
            crate::nsf_hifigan_onnx::ensure_background_prewarm();
            crate::fcpe_onnx::ensure_background_prewarm();
            crate::hnsep_onnx::ensure_background_prewarm();

            // Prefer the OS-level app cache dir so peaks persist across runs.
            let base = app
                .path()
                .app_cache_dir()
                .unwrap_or_else(|_| hfspeaks_v2::default_cache_dir());
            let dir = base.join("hifishifter").join("waveform_peaks_cache");
            {
                let mut d = state
                    .waveform_cache_dir
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *d = dir.clone();
            }
            let _ = hfspeaks_v2::ensure_cache_dir(&dir);

            // 渲染缓存与波形缓存同源（同一用户缓存根），使"重新打开工程不再
            // 重新合成"具备持久化落点；具体设置（开关/容量/自定义目录）在
            // 应用 UI 设置时下发（见下方 load_ui_settings）。
            let render_cache_base = base.join("hifishifter").join("render_cache");
            crate::render_cache::init(render_cache_base);

            // 加载持久化的最近工程列表。
            //
            // 【为什么不再用 `app.path().app_config_dir()`】路径改由内核统一计算：
            // ARA 插件没有 Tauri（依赖树守卫禁止），它必须自己算出**同一个**目录，
            // 否则两个形态各写一份设置 —— 用户在 App 里调好的外观、快捷键、语言
            // 在插件里全部失效。让内核算一次、两边都调用，就没有第二处可以算错。
            // 算出来的路径与 Tauri 的历史布局逐一对应（见 config_location 的模块注释）。
            if let Ok(cfg_dir) = hifishifter_kernel::config_location::resolve_and_create(None) {
                let recent = crate::config::load_recent(&cfg_dir);
                {
                    let mut p = state.project.lock().unwrap_or_else(|e| e.into_inner());
                    p.recent = recent;
                }
                let _ = state.config_dir.set(cfg_dir);
            }

            // 启动即同步"为新的音频块启用循环"的进程级默认值：拖放导入、
            // 打开 v<4 工程的迁移等都可能在 get_ui_settings 之前发生，
            // 不能假设前端已先拉取过设置。
            if let Some(cfg_dir) = state.config_dir.get() {
                let ui = crate::config::load_ui_settings(cfg_dir);
                crate::config::set_loop_new_clips_default(ui.loop_new_clips);
                crate::config::set_sync_edits_across_takes(ui.sync_edits_across_takes);
                // 启动即同步渲染缓存配置：打开工程发生在 get_ui_settings 之前时
                // （外部文件关联、命令行传工程路径），也必须按用户的开关/容量生效。
                crate::render_cache::apply_settings(&ui.render_cache);
            }

            // 尝试恢复上次运行时保存的窗口状态（非强制性）
            if let Some(cfg_dir) = state.config_dir.get() {
                if let Some(win) = app.get_webview_window("main") {
                    let ws = crate::config::load_window_state(cfg_dir);
                    let scale = win.scale_factor().unwrap_or(1.0);

                    // 尺寸：合法才应用（清洗后 None 时保持 tauri.conf.json 默认值）
                    if let (Some(w), Some(h)) = (ws.width, ws.height) {
                        let _ = win.set_size(tauri::Size::Logical(tauri::LogicalSize {
                            width: w,
                            height: h,
                        }));
                    }

                    // 位置：恢复出的矩形必须与某台已连接显示器有足够的可见
                    // 重叠，否则一律回退主显示器内居中。保存的是逻辑像素，
                    // 按 set_position(Logical) 的同一换算（当前 scale）转回
                    // 物理像素做交集判断 —— 校验的就是实际落点。处理两类
                    // 事故：保存后显示器被拔出/改变布局；历史版本在窗口
                    // 最小化时落盘的停泊坐标（如 -25600）。
                    if let (Some(x), Some(y)) = (ws.x, ws.y) {
                        let probe_w = ws.width.map(|w| w * scale).unwrap_or_else(|| {
                            win.inner_size().map(|s| s.width as f64).unwrap_or(1200.0)
                        });
                        let probe_h = ws.height.map(|h| h * scale).unwrap_or_else(|| {
                            win.inner_size().map(|s| s.height as f64).unwrap_or(800.0)
                        });
                        let probe = crate::config::PhysicalRect {
                            x: x as f64 * scale,
                            y: y as f64 * scale,
                            width: probe_w,
                            height: probe_h,
                        };
                        let monitor_bounds = |m: &tauri::Monitor| crate::config::PhysicalRect {
                            x: m.position().x as f64,
                            y: m.position().y as f64,
                            width: m.size().width as f64,
                            height: m.size().height as f64,
                        };
                        let monitors = win.available_monitors().unwrap_or_default();

                        // 与恢复矩形可见重叠量最大的显示器（若有）
                        let mut anchor: Option<(crate::config::PhysicalRect, f64)> = None;
                        for m in &monitors {
                            let bounds = monitor_bounds(m);
                            let (ow, oh) = crate::config::overlap_size(&probe, &bounds);
                            if ow < crate::config::MIN_VISIBLE_OVERLAP_W
                                || oh < crate::config::MIN_VISIBLE_OVERLAP_H
                            {
                                continue;
                            }
                            let area = ow * oh;
                            let better = match &anchor {
                                Some((_, best_area)) => area > *best_area,
                                None => true,
                            };
                            if better {
                                anchor = Some((bounds, area));
                            }
                        }

                        match anchor {
                            Some((bounds, _)) => {
                                // 位置有效；尺寸超出该显示器时钳制
                                if let (Some(w), Some(h)) = (ws.width, ws.height) {
                                    let (cw, ch) = crate::config::clamp_size_to_bounds(
                                        w * scale,
                                        h * scale,
                                        &bounds,
                                    );
                                    if cw < w * scale - 0.5 || ch < h * scale - 0.5 {
                                        let _ = win.set_size(tauri::Size::Physical(
                                            tauri::PhysicalSize {
                                                width: cw.round() as u32,
                                                height: ch.round() as u32,
                                            },
                                        ));
                                    }
                                }
                                let _ = win.set_position(tauri::Position::Logical(
                                    tauri::LogicalPosition {
                                        x: x as f64,
                                        y: y as f64,
                                    },
                                ));
                            }
                            None => {
                                // 回退：主显示器（缺失时取第一台已连接显示器）
                                // 内居中，尺寸钳制到该显示器。绝不让窗口落在
                                // 屏幕外导致"启动即不可见"。
                                let fallback = win
                                    .primary_monitor()
                                    .ok()
                                    .flatten()
                                    .or_else(|| monitors.first().cloned());
                                if let Some(m) = fallback {
                                    let bounds = monitor_bounds(&m);
                                    let (cw, ch) = crate::config::clamp_size_to_bounds(
                                        probe_w, probe_h, &bounds,
                                    );
                                    let _ =
                                        win.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                                            width: cw.round() as u32,
                                            height: ch.round() as u32,
                                        }));
                                    let (px, py) =
                                        crate::config::centered_position((cw, ch), &bounds);
                                    let _ = win.set_position(tauri::Position::Physical(
                                        tauri::PhysicalPosition { x: px, y: py },
                                    ));
                                }
                            }
                        }
                    }

                    if ws.fullscreen.unwrap_or(false) {
                        let _ = win.set_fullscreen(true);
                    } else if ws.maximized.unwrap_or(false) {
                        let _ = win.maximize();
                    } else {
                        let _ = win.set_fullscreen(false);
                    }
                }
            }

            // 禁用 WebView2 的浏览器级快捷键（Ctrl+Plus/Minus 缩放、Ctrl+F
            // 查找、F5 刷新等）—— 它们不可被页面取消，会抢占应用快捷键
            // （参数线微调 Ctrl+= / Ctrl+- 等）。与窗口状态恢复相互独立，
            // 因此放在 cfg_dir 块之外。
            #[cfg(target_os = "windows")]
            if let Some(win) = app.get_webview_window("main") {
                webview2_accelerators::disable_browser_accelerator_keys(&win);
            }

            // 启动时清理上次遗留的临时文件（后台线程，不阻塞启动）
            temp_manager::cleanup_stale_temp_files();

            Ok(())
        })
        // 在窗口事件中监听 CloseRequested，保存窗口状态到配置目录
        .on_window_event(|win, event| {
            // 主窗口真正销毁（其 CloseRequested 可能被前端的"未保存更改"确认
            // 拦截，因此必须挂 Destroyed 而非 CloseRequested）→ 退出应用。
            // 否则"外观设置"等子窗口会让进程继续存活，出现主窗口已关、
            // 子窗口残留且无法关闭的情况；exit 会先同步关闭全部剩余窗口，
            // 再走 RunEvent::Exit 的引擎/会话清理。
            if let tauri::WindowEvent::Destroyed = event {
                if win.label() == "main" {
                    win.app_handle().exit(0);
                }
                return;
            }

            if let tauri::WindowEvent::CloseRequested { .. } = event {
                // 仅针对主窗口保存状态
                if win.label() != "main" {
                    return;
                }

                let maximized = win.is_maximized().unwrap_or(false);
                let fullscreen = win.is_fullscreen().unwrap_or(false);
                let minimized = win.is_minimized().unwrap_or(false);
                let visible = win.is_visible().unwrap_or(true);
                let mut x_opt = None;
                let mut y_opt = None;
                let mut w_opt = None;
                let mut h_opt = None;
                // 最小化（Windows 会把窗口停泊到 -32000 物理坐标，按 125% 缩放
                // 换算成逻辑像素正是 -25600 的污染来源）、最大化/全屏（此时
                // 几何是显示器边界而非窗口几何）、隐藏：捕获不到有意义的窗口
                // 几何。几何字段留 None → save_window_state 保留上一次的好值，
                // 只更新标志位；下次正常关闭时几何会被刷新。
                if !minimized && !maximized && !fullscreen && visible {
                    // 统一保存为逻辑像素：outer_position()/inner_size() 返回物理
                    // 像素，而恢复端按 Logical 解释；在 125%/150% 缩放屏上若不换算，
                    // 每次重启窗口都会放大 scale 倍并持续漂移。
                    let scale = win.scale_factor().unwrap_or(1.0);
                    if let Ok(pos) = win.outer_position() {
                        let logical: tauri::LogicalPosition<f64> = pos.to_logical(scale);
                        x_opt = Some(logical.x.round() as i32);
                        y_opt = Some(logical.y.round() as i32);
                    }
                    if let Ok(size) = win.inner_size() {
                        let logical: tauri::LogicalSize<f64> = size.to_logical(scale);
                        w_opt = Some(logical.width);
                        h_opt = Some(logical.height);
                    }
                }

                if let Some(cfg_dir) = win.app_handle().state::<state::AppState>().config_dir.get()
                {
                    let ws = crate::config::WindowState {
                        x: x_opt,
                        y: y_opt,
                        width: w_opt,
                        height: h_opt,
                        maximized: Some(maximized),
                        fullscreen: Some(fullscreen),
                    };
                    crate::config::save_window_state(cfg_dir, &ws);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::ara_list_instances,
            commands::ara_connect,
            commands::ara_submit,
            commands::ara_refresh,
            commands::ara_disconnect,
            commands::ping,
            commands::get_about_info,
            commands::analyze_clip_formants,
            commands::get_runtime_info,
            commands::consume_startup_project_path,
            commands::set_ui_locale,
            commands::get_timeline_state,
            commands::get_timeline_state_lite,
            commands::set_transport,
            commands::close_window,
            commands::undo_timeline,
            commands::redo_timeline,
            commands::begin_undo_group,
            commands::end_undo_group,
            commands::get_history_state,
            commands::set_history_position,
            commands::record_param_selection_step,
            commands::set_project_save_undo_history,
            commands::get_project_meta,
            commands::new_project,
            commands::open_project_dialog,
            commands::open_project,
            commands::import_project_dialog,
            commands::import_project,
            commands::set_project_notes,
            commands::seal_project_notes_history,
            commands::notebook_put_asset,
            commands::notebook_read_asset,
            commands::notebook_list_assets,
            commands::notebook_remove_asset,
            commands::notebook_prune_assets,
            commands::notebook_read_file_base64,
            commands::notebook_read_clipboard_payload,
            commands::notebook_write_clipboard_payload,
            commands::notebook_read_clipboard_image,
            commands::notebook_export_document,
            commands::notebook_save_asset_as,
            commands::save_project,
            commands::save_project_as,
            commands::save_project_to_path,
            commands::get_auto_backup_settings,
            commands::save_auto_backup_settings,
            commands::run_timed_auto_backup,
            commands::get_recording_settings,
            commands::save_recording_settings,
            commands::get_recording_devices,
            commands::get_recording_apps,
            commands::start_recording,
            commands::stop_recording,
            commands::get_recording_state,
            commands::set_project_base_scale,
            commands::set_project_custom_scale,
            commands::set_project_stretch_settings,
            commands::set_project_timeline_settings,
            commands::set_timeline_tempo_map,
            commands::open_audio_dialog,
            commands::open_audio_dialog_multi,
            commands::open_audio_dialog_for_source,
            commands::get_media_audio_streams,
            commands::pick_output_path,
            commands::pick_directory,
            commands::open_midi_dialog,
            commands::get_root_mix_waveform_peaks_segment,
            commands::get_track_mix_waveform_peaks_segment,
            commands::clear_waveform_cache,
            commands::get_waveform_mipmap_binary,
            commands::preload_waveform_mipmap,
            commands::batch_get_waveform_mipmap,
            commands::import_audio_item,
            commands::import_audio_bytes,
            commands::add_track,
            commands::add_track_tree,
            commands::remove_track,
            commands::duplicate_track,
            commands::move_track,
            commands::set_track_state,
            commands::select_track,
            commands::set_project_length,
            commands::get_track_summary,
            commands::get_param_frames,
            commands::set_param_frames,
            commands::restore_param_frames,
            commands::convert_mix_param,
            commands::stretch_track_linked_params,
            commands::add_clip,
            commands::create_clips_bulk,
            commands::get_static_param,
            commands::set_static_param,
            commands::remove_clip,
            commands::remove_clips,
            commands::move_clip,
            commands::move_clips,
            commands::get_clip_linked_params,
            commands::apply_clip_linked_params,
            commands::set_clip_state,
            commands::set_clips_state_bulk,
            commands::set_clip_active_take,
            commands::cycle_clip_takes,
            commands::pack_clips_into_takes,
            commands::explode_clip_takes,
            commands::duplicate_clip_take,
            commands::remove_clip_take,
            commands::rename_clip_take,
            commands::set_clip_take_reversed,
            commands::set_clip_take_channel_mode,
            commands::scan_and_convert_fake_stereo,
            commands::add_clip_take_from_media,
            commands::import_media_files_as_takes,
            commands::duplicate_clips_bulk,
            commands::replace_clip_source,
            commands::check_source_files_changed,
            commands::search_source_file_replacements,
            commands::split_clip,
            commands::split_clips_at,
            commands::analyze_clip_silence,
            commands::remove_clip_silence,
            commands::close_track_gaps,
            commands::glue_clips,
            commands::group_clips,
            commands::ungroup_clips,
            commands::toggle_group_disabled,
            commands::convert_clips_to_pitch_reference,
            commands::update_pitch_reference,
            commands::select_clip,
            commands::copy_timeline_clips,
            commands::copy_timeline_tracks,
            commands::paste_timeline_clipboard,
            commands::has_timeline_clipboard,
            commands::clipboard_kind,
            commands::write_system_clipboard_object,
            commands::read_system_clipboard_object,
            commands::load_default_model,
            commands::load_model,
            commands::set_pitch_shift,
            commands::process_audio,
            commands::synthesize,
            commands::save_synthesized,
            commands::save_separated,
            commands::export_audio_advanced,
            commands::cancel_export_audio,
            commands::get_export_audio_defaults,
            commands::preview_export_audio_plan,
            commands::quick_export_selected_clips,
            commands::set_metronome,
            commands::play_original,
            commands::stop_audio,
            commands::get_playback_state,
            commands::start_background_render,
            commands::cancel_background_render,
            commands::get_pitch_analysis_progress,
            commands::open_log_folder,
            commands::pick_diagnostics_output_path,
            commands::export_diagnostics,
            commands::export_layout_json,
            commands::export_theme_json,
            commands::export_vibrato_presets_json,
            commands::log_frontend_error,
            commands::get_onnx_status,
            commands::get_onnx_diagnostic,
            commands::get_vslib_status,
            commands::run_vocoder_benchmark,
            commands::get_gpu_devices,
            commands::get_dml_adapters,
            commands::list_directory,
            commands::stat_paths,
            commands::collect_folder_media,
            commands::create_directory,
            commands::rename_path,
            commands::delete_paths,
            commands::get_audio_file_info,
            commands::read_audio_preview,
            commands::search_files_recursive,
            commands::reveal_paths_in_file_manager,
            commands::open_path_with_default_app,
            commands::transliterate,
            commands::open_vocalshifter_dialog,
            commands::import_vocalshifter_project,
            commands::paste_vocalshifter_clipboard,
            commands::open_reaper_dialog,
            commands::import_reaper_project,
            commands::paste_reaper_clipboard,
            commands::has_reaper_clipboard,
            commands::get_render_cache_stats,
            commands::clear_render_cache,
            commands::open_render_cache_dir,
            commands::reveal_export_paths,
            commands::get_processor_params,
            commands::get_midi_tracks,
            commands::read_midi_clipboard_to_memory,
            commands::import_midi_to_pitch,
            commands::import_midi_as_clip,
            commands::replace_midi_clip_data,
            commands::pick_midi_output_path,
            commands::export_pitch_to_midi,
            commands::get_ui_settings,
            commands::save_ui_settings,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // 退出前把渲染缓存的待写队列（`onExit` / `manual` 模式以及在途
                // 写入）排空；带超时，磁盘异常时也不能卡住退出流程。
                if !crate::render_cache::flush_blocking(std::time::Duration::from_secs(3)) {
                    log::warn!("[render_cache] pending writes did not drain before exit timeout");
                }

                // Shut down audio engine: stop meter thread, send Shutdown to
                // worker threads, and drop the channel sender so all worker
                // threads exit their recv loops.
                let state = app_handle.state::<state::AppState>();
                crate::recording::shutdown(state.inner());
                state.audio_engine.shutdown();

                // Force-drop all ONNX sessions to release GPU memory before exit.
                crate::nsf_hifigan_onnx::drop_shared_session();
                crate::fcpe_onnx::drop_shared_session();
                crate::hnsep_onnx::drop_shared_session();
            }
        });
}
