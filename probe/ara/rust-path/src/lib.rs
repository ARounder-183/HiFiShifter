//! HiFiShifter ARA 探针 / Task 2 Step 3 —— 可被 REAPER 加载的 VST3+ARA 插件。
//!
//! 目的：回答探针计划里唯一的硬判据 —— 一个 Rust 写的 cdylib 能否被 REAPER 当作
//! ARA 插件加载，并把宿主侧的 audioSource / playbackRegion 数量读到日志里。
//!
//! 结构：
//! - `vst3` 是手写的最小 VST3 模块 ABI（工厂 + 组件 + 处理器），导出 `GetPluginFactory`
//!   / `InitDll` / `ExitDll`；
//! - `model` 是一次性的 ARA 文档模型探针，把宿主推来的对象计数并落盘；
//! - 本文件把两者接起来：建一个 ARA 工厂、把它包成 companion 关联，再交给 VST3 侧。
//!
//! 这是一次性探针产物，不属于产品代码；全部放在 `probe/ara/` 下。

mod model;
mod vst3;

use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::core::AraError;
use ara2_bridge::plugin::{Factory, FactoryBuilder, PluginBuilder, PluginEntry};
use model::{ProbeModel, ProbeState};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// VST3 类名，同时必须是 ARA 工厂的 plugInName（ARA 的配对规则要求两者一致）。
pub(crate) const CLASS_NAME: &str = "HiFiShifter ARA Probe";
/// 探针版本号（展示用）。
pub(crate) const VERSION: &str = "0.1.0";
/// ARA 工厂的持久 ID。
const FACTORY_ID: &str = "com.hifishifter.ara.probe";
/// ARA 文档归档 ID。
const ARCHIVE_ID: &str = "com.hifishifter.ara.probe.archive";
/// 未设置 `HIFISHIFTER_ARA_PROBE_LOG` 时的落盘位置（本工作树内，便于采集）。
const FALLBACK_LOG: &str =
    r"E:\code\HiFiShifter\.worktrees\ara-bridge-probe\probe\ara\captures\ara-probe-plugin.log";

/// 把 `Factory` 以只读方式跨线程共享的包装。
///
/// `Factory` 内部是 ARA 回调的常驻 backing，ARA 工厂契约本身就要求它在进程内稳定可读；
/// 探针只读取它（取 generation / 包 companion 关联），不修改，因此这条 `Sync` 是安全的。
pub(crate) struct StaticFactory(&'static Factory);

// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Send for StaticFactory {}
// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Sync for StaticFactory {}

impl StaticFactory {
    /// 借出该工厂的初始化入口（用于读取协商到的 ARA 版本）。
    pub(crate) fn entry(&self) -> &PluginEntry {
        self.0.entry()
    }
}

/// 进程级探针运行时：泄漏到进程结束的 ARA 工厂 + companion 关联。
pub(crate) struct Runtime {
    /// ARA 工厂。
    pub(crate) factory: StaticFactory,
    /// companion 层看到的天生关联（VST3 主工厂与处理器都用它）。
    pub(crate) companion: CompanionFactory<'static>,
}

impl Runtime {
    /// 建立工厂与 companion 关联；日志器在最早时刻就绪，便于记录失败原因。
    fn init() -> Result<Runtime, AraError> {
        let state = Arc::new(ProbeState::new(log_path()));
        state.log.write(&format!(
            "=== HiFiShifter ARA probe {VERSION} (class '{CLASS_NAME}') ==="
        ));
        let logger = Arc::clone(&state);
        let build_state = Arc::clone(&state);
        let factory = FactoryBuilder::new(FACTORY_ID, ARCHIVE_ID)
            .display(CLASS_NAME, "HiFiShifter", "https://example.invalid", VERSION)
            .document_controller(move || {
                let model = ProbeModel {
                    state: Arc::clone(&build_state),
                };
                Ok(PluginBuilder::new(model).build()?)
            })
            .build()?;
        let factory: &'static Factory = Box::leak(Box::new(factory));
        // SAFETY: 工厂被泄漏到进程结束，as_raw 指向的 ARAFactory 与工厂同寿。
        let companion = unsafe { CompanionFactory::from_raw(CLASS_NAME, &*factory.as_raw())? };
        logger
            .log
            .write("ARA factory built; companion association ready");
        Ok(Runtime {
            factory: StaticFactory(factory),
            companion,
        })
    }
}

/// 进程级唯一的探针运行时；首次访问时初始化。
static RUNTIME: OnceLock<Option<Runtime>> = OnceLock::new();

/// 取运行时；初始化失败时返回 `None`（原因写入日志或 stderr）。
pub(crate) fn runtime() -> Option<&'static Runtime> {
    RUNTIME
        .get_or_init(|| match Runtime::init() {
            Ok(runtime) => Some(runtime),
            Err(error) => {
                // 日志器可能尚未建立，此时只能退回 stderr（REAPER 下通常不可见）。
                eprintln!("[HiFiShifter ARA probe] runtime init failed: {error:?}");
                None
            }
        })
        .as_ref()
}

/// 决定日志落盘位置：优先环境变量，其次工作树内的固定回退路径。
fn log_path() -> PathBuf {
    match std::env::var_os("HIFISHIFTER_ARA_PROBE_LOG") {
        Some(path) if !path.is_empty() => PathBuf::from(path),
        _ => PathBuf::from(FALLBACK_LOG),
    }
}

/// 把一行诊断信息写进探针日志；供 VST3 侧回调定位"宿主走到哪一步"。
pub(crate) fn probe_log(message: &str) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
    {
        let _ = writeln!(file, "[vst3] {message}");
    }
}

/// VST3 模块入口：返回 `IPluginFactory*`。
#[no_mangle]
pub extern "system" fn GetPluginFactory() -> *mut c_void {
    let _ = runtime();
    vst3::get_plugin_factory()
}

/// VST3 模块初始化入口（REAPER 在 `LoadLibrary` 之后显式调用）。
#[no_mangle]
pub extern "system" fn InitDll() -> bool {
    // 即使初始化失败也返回 true：让宿主把类列出来，失败原因记在日志里，
    // 否则 REAPER 会直接跳过插件、什么信息都拿不到。
    let _ = runtime();
    true
}

/// VST3 模块卸载入口。
#[no_mangle]
pub extern "system" fn ExitDll() -> bool {
    true
}
