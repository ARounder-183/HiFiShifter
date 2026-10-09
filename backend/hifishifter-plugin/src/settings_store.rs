//! 插件进程级的用户设置与前端偏好：与独立 App 共用同一份配置文件。
//!
//! 【为什么不再是 `EditorSession` 的字段】设置原先挂在编辑器会话上，而会话由
//! `DocumentSession::editor_session()` 惰性创建 —— 也就是说设置是**每个 ARA 文档**
//! 一份，且初始化为 `UiSettings::default()`。换一个 REAPER 工程就是一套全新默认值，
//! 退出再进更是全部归零。设置属于用户，不属于某一个工程。
//!
//! 【为什么是 `Mutex<Option<..>>` 而不是 `OnceLock`】`OnceLock` 一旦初始化就无法
//! 重置，而"重启后设置还在吗"正是这里最需要测的行为。`Option` 让测试可以丢掉状态
//! 再从磁盘重新加载，模拟一次真实的宿主重启。
//!
//! 【为什么目录解析在 `cfg(test)` 下换实现】本模块真的会写盘。如果单元测试写真实
//! 的用户配置，一次 `cargo test` 就会改掉开发者自己的设置，并行测试之间也会互踩。
//! 测试一律落在进程私有的临时目录。

use hifishifter_kernel::config::UiSettings;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// 前端偏好的键名沿用原 `localStorage` 的 `hifishifter.*` 字符串。
///
/// 【为什么不做命名映射】迁移时不必维护一张表，排查时也能把浏览器里看到的键名
/// 与配置文件里的键名直接对上。
pub const LOCALE_KEY: &str = "hifishifter.locale";

struct StoreState {
    /// 配置目录；`None` 表示这台机器上解析不出来 —— 此时设置只在内存里生效，
    /// 不影响本次会话使用，只是重启后丢失。
    dir: Option<PathBuf>,
    settings: UiSettings,
    prefs: BTreeMap<String, String>,
}

impl StoreState {
    fn load() -> Self {
        let dir = resolve_dir();
        let settings = dir
            .as_deref()
            .map(hifishifter_kernel::config::load_ui_settings)
            .unwrap_or_default();
        let prefs = dir
            .as_deref()
            .map(hifishifter_kernel::config::load_frontend_prefs)
            .unwrap_or_default();
        // 启动即下发一次：进程级消费方（ONNX 会话 / 拉伸默认值 / 导入策略）必须与
        // 磁盘上的取值一致。此前插件从不调用这些下发函数，于是"默认拉伸算法"
        // 这类设置在插件里是死的 —— 值存下来了，没有任何人读它。
        hifishifter_kernel::ui_settings_apply::apply(&settings);
        Self {
            dir,
            settings,
            prefs,
        }
    }

    /// 写盘失败只记日志：设置存不下不该让宿主进程出错，本次会话仍按内存中的值走。
    fn persist_settings(&self) {
        let Some(dir) = &self.dir else { return };
        hifishifter_kernel::config::save_ui_settings(dir, &self.settings);
    }

    fn persist_prefs(&self) {
        let Some(dir) = &self.dir else { return };
        hifishifter_kernel::config::save_frontend_prefs(dir, &self.prefs);
    }
}

static STORE: Mutex<Option<StoreState>> = Mutex::new(None);

/// 解析配置目录。测试走进程私有临时目录（见模块注释）。
fn resolve_dir() -> Option<PathBuf> {
    #[cfg(test)]
    {
        use std::sync::OnceLock;
        static TEST_DIR: OnceLock<PathBuf> = OnceLock::new();
        let dir = TEST_DIR.get_or_init(|| {
            let dir = std::env::temp_dir()
                .join(format!("hfs-plugin-settings-test-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            dir
        });
        Some(dir.clone())
    }
    #[cfg(not(test))]
    {
        hifishifter_kernel::config_location::resolve_and_create(None).ok()
    }
}

/// 借出进程级状态，首次访问时从磁盘加载。
fn with_state<T>(f: impl FnOnce(&mut StoreState) -> T) -> T {
    let mut guard = STORE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(StoreState::load());
    }
    f(guard.as_mut().expect("store initialized above"))
}

/// 当前生效的 UI 设置。
pub fn settings() -> UiSettings {
    with_state(|state| state.settings.clone())
}

/// 设置代次：每次写入自增。
///
/// 【为什么需要】热路径（每次 `workspace_timeline_locked`）都要读"插件自有的音乐
/// 上下文"，而 `settings()` 会克隆整份 `UiSettings`（字段很多）。用一个原子代次
/// 做缓存戳，命中时只付一次原子读的代价。
static REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 当前设置代次；与 [`save_settings_patch`] 的写入一一对应。
pub fn revision() -> u64 {
    REVISION.load(std::sync::atomic::Ordering::Acquire)
}

/// 应用一份**部分**设置补丁并落盘。
///
/// 合并语义沿用独立 App 的一层深合并（`hifishifter_kernel::editor::settings::merge`）：
/// 前端一次只发变更字段，按顶层键整体替换会让未发送的兄弟字段回落到默认值 ——
/// 用户改一个开关就会丢掉一批设置。
pub fn save_settings_patch(patch: &serde_json::Value) -> Result<UiSettings, String> {
    let result = with_state(|state| {
        let base = serde_json::to_value(&state.settings).map_err(|e| e.to_string())?;
        let merged = hifishifter_kernel::editor::settings::merge(base, patch);
        let next: UiSettings =
            serde_json::from_value(merged).map_err(|e| format!("invalid ui settings: {e}"))?;
        state.settings = next;
        state.persist_settings();
        // 下发到进程级消费方（ONNX 会话 / 拉伸默认值 / 导入策略）。
        hifishifter_kernel::ui_settings_apply::apply(&state.settings);
        Ok(state.settings.clone())
    });
    if result.is_ok() {
        REVISION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    result
}

/// 全部前端偏好（前端启动时一次性取走，用于灌进内存缓存）。
pub fn frontend_prefs() -> BTreeMap<String, String> {
    with_state(|state| state.prefs.clone())
}

/// 合并写入若干前端偏好，返回写入后的完整快照。
pub fn save_frontend_prefs(patch: BTreeMap<String, String>) -> BTreeMap<String, String> {
    with_state(|state| {
        for (key, value) in patch {
            state.prefs.insert(key, value);
        }
        state.persist_prefs();
        state.prefs.clone()
    })
}

/// 删除若干前端偏好键，返回删除后的完整快照。
pub fn delete_frontend_prefs(keys: &[String]) -> BTreeMap<String, String> {
    with_state(|state| {
        for key in keys {
            state.prefs.remove(key);
        }
        state.persist_prefs();
        state.prefs.clone()
    })
}

/// 记住界面语言。
///
/// 【为什么写进前端偏好而不是新开一个字段】前端本来就把语言存在
/// `hifishifter.locale` 这个键下；写同一个键，两边的取值不会分叉，用户换回
/// 独立 App 时语言也跟着走。插件没有原生对话框，因此不需要再做一遍 App 那边的
/// 语言归一化（那一步只为原生对话框服务）。
pub fn set_locale(locale: &str) -> String {
    let mut patch = BTreeMap::new();
    patch.insert(LOCALE_KEY.to_string(), locale.to_string());
    save_frontend_prefs(patch);
    locale.to_string()
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// 丢掉内存状态，下次访问时重新从磁盘加载 —— 等价于宿主重启一次。
    ///
    /// 这是本模块最关键的测试手段：只有这样才能验证"设置真的落盘了"，
    /// 而不是"还在那个 Mutex 里"。
    pub(crate) fn simulate_restart() {
        let mut guard = STORE.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    /// 测试用临时配置目录（与 `resolve_dir` 在 `cfg(test)` 下返回的同一个）。
    pub(crate) fn test_dir() -> PathBuf {
        resolve_dir().expect("test config dir")
    }

    /// 清空测试目录，让每个测试从干净状态起手。
    pub(crate) fn reset() {
        simulate_restart();
        let dir = test_dir();
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);
    }

    /// 串行化本模块的测试。
    ///
    /// 【为什么必须有】`STORE` 是**进程级**的单一状态，`test_dir()` 也是单一目录，
    /// 而 `reset()` 会把两者一起清空。两个测试并行时，A 的写入会被 B 的 `reset()`
    /// 抹掉 —— 表现为"未参与本次保存的键丢失"这种看起来像真 bug 的失败。
    /// 这不是被测代码的问题（进程里本来就只有一个设置存储），是测试之间需要互斥。
    pub(crate) fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{exclusive, reset, simulate_restart};
    use super::*;

    /// 保存的设置必须在**重新加载之后**还在。
    ///
    /// 【为什么这条是核心】插件此前"保存成功"只是写进了内存里的那个 Mutex，
    /// 返回 `{"ok":true}` 却什么也没落盘 —— 于是每次重启都回到出厂默认。
    /// 只有跨过一次重新加载还能读回，才叫持久化。
    #[test]
    fn saved_settings_survive_a_restart() {
        let _serial = exclusive();
        reset();
        let mut edited = settings();
        edited.ruler_label_spacing_px = 137;
        edited.auto_crossfade = true;
        save_settings_patch(&serde_json::to_value(&edited).unwrap()).expect("save settings");

        simulate_restart();

        let reloaded = settings();
        assert_eq!(reloaded.ruler_label_spacing_px, 137, "重启后设置丢失");
        assert!(reloaded.auto_crossfade, "重启后设置丢失");
    }

    /// 部分保存不得抹掉未发送的字段。
    #[test]
    fn a_partial_patch_keeps_the_other_fields() {
        let _serial = exclusive();
        reset();
        let mut edited = settings();
        edited.ruler_label_spacing_px = 111;
        edited.auto_crossfade = true;
        save_settings_patch(&serde_json::to_value(&edited).unwrap()).unwrap();

        save_settings_patch(&serde_json::json!({ "rulerLabelSpacingPx": 222 })).unwrap();
        simulate_restart();

        let reloaded = settings();
        assert_eq!(reloaded.ruler_label_spacing_px, 222);
        assert!(reloaded.auto_crossfade, "未参与本次保存的字段被抹掉了");
    }

    /// 设置是**每用户**的，不随 ARA 文档（工程）变化。
    ///
    /// 【回归测试】设置原先挂在 `EditorSession` 上，而会话按 ARA 文档惰性创建 ——
    /// 换一个 REAPER 工程就回到出厂默认。现在它挂在进程上，`simulate_restart`
    /// 只重载磁盘，不重建任何会话。
    #[test]
    fn settings_are_not_scoped_to_a_document() {
        let _serial = exclusive();
        reset();
        let mut edited = settings();
        edited.ruler_label_spacing_px = 88;
        save_settings_patch(&serde_json::to_value(&edited).unwrap()).unwrap();

        // 另一次读取（模拟另一个工程里的编辑器窗口首次拉取设置）。
        assert_eq!(settings().ruler_label_spacing_px, 88);
    }

    /// 前端偏好跨重启保留，且合并写入不丢其它键。
    #[test]
    fn frontend_prefs_survive_a_restart_and_merge() {
        let _serial = exclusive();
        reset();
        let mut first = BTreeMap::new();
        first.insert("hifishifter.keybindings".to_string(), "{}".to_string());
        first.insert("hifishifter.pxPerSec".to_string(), "120".to_string());
        save_frontend_prefs(first);

        let mut second = BTreeMap::new();
        second.insert("hifishifter.pxPerSec".to_string(), "240".to_string());
        save_frontend_prefs(second);

        simulate_restart();

        let prefs = frontend_prefs();
        assert_eq!(
            prefs.get("hifishifter.pxPerSec").map(String::as_str),
            Some("240")
        );
        assert_eq!(
            prefs.get("hifishifter.keybindings").map(String::as_str),
            Some("{}"),
            "未参与本次保存的键丢失"
        );
    }

    /// 删除偏好键只删指定的那些。
    #[test]
    fn deleting_prefs_removes_only_the_named_keys() {
        let _serial = exclusive();
        reset();
        let mut patch = BTreeMap::new();
        patch.insert("hifishifter.appearance".to_string(), "dark".to_string());
        patch.insert("hifishifter.customThemes".to_string(), "[]".to_string());
        save_frontend_prefs(patch);

        let after = delete_frontend_prefs(&["hifishifter.appearance".to_string()]);
        assert!(!after.contains_key("hifishifter.appearance"));
        assert!(after.contains_key("hifishifter.customThemes"));
    }

    /// 界面语言写在约定好的键上并跨重启保留。
    #[test]
    fn the_locale_is_remembered() {
        let _serial = exclusive();
        reset();
        set_locale("ja-JP");
        simulate_restart();
        assert_eq!(
            frontend_prefs().get(LOCALE_KEY).map(String::as_str),
            Some("ja-JP")
        );
    }
}
