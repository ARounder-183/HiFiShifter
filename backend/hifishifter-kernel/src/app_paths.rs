//! 用户级目录与共享模型库的唯一计算点：独立 App 与 VST3/CLAP 插件都调它。
//!
//! 【为什么必须有这一处】"用户数据放哪儿"此前分散在三处：App 用 Tauri 的
//! `app_config_dir()` / `app_cache_dir()` / `app_log_dir()`，插件自己拼
//! `com.arounder.hifishifter/HiFiShifter`，内核的 `logfile` 又拼了第三份。代价是
//! 两个形态的目录只要有一处算错，症状就只是"设置不共享""日志不见了"—— 没有报错，
//! 只能靠用户描述猜。现在路径只有这一处计算，**没有第二处可以算错**。
//!
//! ## 目标布局
//!
//! | 用途 | Windows | macOS | Linux |
//! | --- | --- | --- | --- |
//! | 设置（漫游） | `%APPDATA%\HiFiShifter` | `~/Library/Application Support/HiFiShifter` | `$XDG_CONFIG_HOME/HiFiShifter` |
//! | 本地数据 | `%LOCALAPPDATA%\HiFiShifter` | `~/Library/Application Support/HiFiShifter` | `$XDG_DATA_HOME/HiFiShifter` |
//! | 缓存 | `%LOCALAPPDATA%\HiFiShifter\cache` | `~/Library/Caches/HiFiShifter` | `$XDG_CACHE_HOME/HiFiShifter` |
//! | 日志 | `%LOCALAPPDATA%\HiFiShifter\logs` | `~/Library/Logs/HiFiShifter` | `$XDG_STATE_HOME/HiFiShifter/logs` |
//! | WebView profile | `%LOCALAPPDATA%\HiFiShifter\EBWebView` | 跟随 Tauri 默认 | 跟随 Tauri 默认 |
//! | 共享模型库 | `%PROGRAMDATA%\HiFiShifter\models\<版本>` | `/Library/Application Support/HiFiShifter/models/<版本>`（统一 pkg） | `$XDG_DATA_HOME/HiFiShifter/models/<版本>` 或 `/usr/local/share/HiFiShifter/models/<版本>`（统一包） |
//!
//! 【为什么目录名是 `HiFiShifter` 而不是 identifier】`com.arounder.hifishifter` 是
//! **身份**（安装器、签名、更新源、WebView 数据目录归属），它对用户不可见；而用户
//! 要在资源管理器里找自己的东西时，看到的是 `%APPDATA%\HiFiShifter`。历史布局把身份
//! 当成了目录名，于是用户拿到的是一串反向域名 —— 这一层在 v2 布局里消失，**身份本身
//! 一个字都没改**（改它等于换一个产品）。
//!
//! 【为什么 macOS 的设置与本地数据同目录】macOS 没有 Roaming/Local 之分，
//! `Application Support` 就是两者共同的家；硬拆成两个目录只会造出一个用户永远
//! 找不到的第二处。
//!
//! 【旧布局怎么办】见 `crate::app_paths_migrate`：两个宿主在启动最早期各跑一次
//! 一次性迁移，把 `com.arounder.hifishifter\…` 搬到新根。

use std::path::{Component, Path, PathBuf};

/// 产品目录名。
///
/// 与 `bundle.identifier` 无关：identifier 是**身份**，这是给人看的目录名。
pub const PRODUCT_DIR: &str = "HiFiShifter";

/// 布局标记文件：存在于**本地数据根**下，表示 v2 迁移已完成。
pub const LAYOUT_MARKER: &str = ".layout-v2.json";

/// 覆盖配置目录的环境变量。
///
/// 【为什么必须有】插件与内核的单元测试跑在任何一台机器上，如果它们写真实的用户
/// 配置，一次 `cargo test` 就会改掉运行测试那个人的设置；并行测试之间也会互相踩。
/// 有了这个变量，每个测试指向自己的临时目录，落盘逻辑才可测。
pub const CONFIG_DIR_ENV: &str = "HIFISHIFTER_CONFIG_DIR";

/// 覆盖日志目录的环境变量（两个宿主同名同义）。
pub const LOG_DIR_ENV: &str = "HIFISHIFTER_LOG_DIR";

/// 历史布局的一级目录名（= Tauri 的 identifier）。
///
/// 只用于**迁移**与**读时重映射**：新布局里它不再出现在任何路径中。
pub const LEGACY_IDENTIFIER: &str = "com.arounder.hifishifter";

/// 读一个"目录型"环境变量：未设置或只有空白时视为未设置（不能解析成空路径）。
fn env_path(key: &str) -> Option<PathBuf> {
    let raw = std::env::var_os(key)?;
    let raw = raw.to_string_lossy().trim().to_owned();
    if raw.is_empty() {
        None
    } else {
        Some(PathBuf::from(raw))
    }
}

#[cfg(windows)]
fn roaming_base() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

/// Windows 的**本地**数据根。
///
/// `LOCALAPPDATA` 是标准变量；极简环境（CI 容器、测试沙箱）可能没有它，此时按同一
/// 语义回退到 `%USERPROFILE%\AppData\Local`。
#[cfg(windows)]
fn local_base() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| {
            let home = std::env::var_os("USERPROFILE")?;
            Some(PathBuf::from(home).join("AppData").join("Local"))
        })
}

#[cfg(not(windows))]
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// XDG 目录：环境变量优先，缺省 `~/<fallback>`。
#[cfg(all(unix, not(target_os = "macos")))]
fn xdg_base(variable: &str, fallback: &str) -> Option<PathBuf> {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .or_else(|| Some(home_dir()?.join(fallback)))
}

/// 设置目录（漫游）。
pub fn config_dir() -> Option<PathBuf> {
    if let Some(dir) = env_path(CONFIG_DIR_ENV) {
        return Some(dir);
    }
    #[cfg(windows)]
    {
        Some(roaming_base()?.join(PRODUCT_DIR))
    }
    #[cfg(target_os = "macos")]
    {
        Some(
            home_dir()?
                .join("Library")
                .join("Application Support")
                .join(PRODUCT_DIR),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some(xdg_base("XDG_CONFIG_HOME", ".config")?.join(PRODUCT_DIR))
    }
}

/// 本地数据目录（PCM 暂存、捕获、传输、插件记事本…）。
///
/// 【为什么不能用 `%TEMP%`】WebView2 的用户数据目录与渲染用的 PCM 暂存原先都放在
/// 系统临时目录里。那不是"临时"能形容的问题：
/// - Windows 磁盘清理会在**会话进行中**删除 `%TEMP%` 下的内容 —— 正在运行的
///   WebView2 用户数据目录被删掉是崩溃隐患，不只是"设置丢了"；
/// - `%TEMP%` 可能被策略指向内存盘或每次登录重建的目录。
///
/// 放在本地数据目录下，这些文件的生命周期就由我们自己决定。
pub fn local_data_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        Some(local_base()?.join(PRODUCT_DIR))
    }
    #[cfg(target_os = "macos")]
    {
        Some(
            home_dir()?
                .join("Library")
                .join("Application Support")
                .join(PRODUCT_DIR),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some(xdg_base("XDG_DATA_HOME", ".local/share")?.join(PRODUCT_DIR))
    }
}

/// 缓存目录（波形峰值、整段渲染 PCM）。
///
/// 【为什么与本地数据分开】这些文件是**派生物**：删掉只会变慢，不会丢数据。独立成
/// 一层，用户与系统清理工具才能安全地"清缓存"，而不必猜 `HiFiShifter\` 里哪几个
/// 子目录可以动。
pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        Some(local_base()?.join(PRODUCT_DIR).join("cache"))
    }
    #[cfg(target_os = "macos")]
    {
        Some(home_dir()?.join("Library").join("Caches").join(PRODUCT_DIR))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some(xdg_base("XDG_CACHE_HOME", ".cache")?.join(PRODUCT_DIR))
    }
}

/// 日志目录。两个宿主写进**同一个目录**、各自一份文件名。
pub fn log_dir() -> Option<PathBuf> {
    if let Some(dir) = env_path(LOG_DIR_ENV) {
        return Some(dir);
    }
    #[cfg(windows)]
    {
        Some(local_base()?.join(PRODUCT_DIR).join("logs"))
    }
    #[cfg(target_os = "macos")]
    {
        Some(home_dir()?.join("Library").join("Logs").join(PRODUCT_DIR))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Some(
            xdg_base("XDG_STATE_HOME", ".local/state")?
                .join(PRODUCT_DIR)
                .join("logs"),
        )
    }
}

/// **共享**数据根目录的候选列表，按优先级排列。
///
/// 模型库不是"每用户缓存"，而是安装器写入的一份只读载荷：独立 App 与所有插件
/// 都从同一份读取。不同安装形态能写的位置不同，因此这里返回候选而不是单个路径；
/// 运行时按顺序挑选第一个完整可用的模型库。
///
/// - Windows：`%PROGRAMDATA%\HiFiShifter`（合并安装器独占写入）。
/// - macOS：`/Library/Application Support/HiFiShifter`（统一 pkg 写入）；再加
///   一个用户级路径，供不经过 pkg 的手动安装兜底。
/// - Linux：`$XDG_DATA_HOME/HiFiShifter`（用户级统一包默认写入）；再依次尝试
///   `/usr/local/share/HiFiShifter` 与 `/usr/share/HiFiShifter`，供
///   `install.sh --system` 或发行版包管理器安装。
///
/// 【为什么 Windows 只给一个】合并安装器以管理员身份运行，`%PROGRAMDATA%` 对标准
/// 用户可读、对提权安装可写。把模型写进 `%LOCALAPPDATA%` 会在"标准用户 + 管理员
/// 密码"提权时落到管理员的配置目录，真正运行插件的标准用户反而读不到。
pub fn shared_data_dirs() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        // PROGRAMDATA 在正常 Windows 上一定存在；极简测试环境可能没有环境块，
        // 此时按系统盘约定退回 C:\ProgramData，避免共享库路径直接消失。
        let base = std::env::var_os("PROGRAMDATA")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("SystemDrive")
                    .map(|drive| PathBuf::from(drive).join("ProgramData"))
            })
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        vec![base.join(PRODUCT_DIR)]
    }
    #[cfg(target_os = "macos")]
    {
        let mut dirs = vec![PathBuf::from("/Library")
            .join("Application Support")
            .join(PRODUCT_DIR)];
        if let Some(home) = home_dir() {
            dirs.push(
                home.join("Library")
                    .join("Application Support")
                    .join(PRODUCT_DIR),
            );
        }
        dirs
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mut dirs = Vec::new();
        if let Some(base) = xdg_base("XDG_DATA_HOME", ".local/share") {
            dirs.push(base.join(PRODUCT_DIR));
        }
        dirs.push(PathBuf::from("/usr/local/share").join(PRODUCT_DIR));
        dirs.push(PathBuf::from("/usr/share").join(PRODUCT_DIR));
        dirs
    }
}

/// **共享**数据根目录的首选位置。
///
/// 保留这个单值入口给只需要一个位置的调用方；模型解析应使用
/// [`shared_data_dirs`] 逐个尝试，避免用户级与系统级安装形态互相遮蔽。
pub fn shared_data_dir() -> Option<PathBuf> {
    shared_data_dirs().into_iter().next()
}

/// 本地数据目录下的一个子目录；解析不出来时退回系统临时目录。
///
/// 【为什么保留临时目录兜底】缺 `LOCALAPPDATA` 时宁可用临时目录也不能让插件起不来 ——
/// 一个跑得起来但清理不完美的会话，好过一个打不开的窗口。
pub fn local_data_subdir(name: &str) -> PathBuf {
    match local_data_dir() {
        Some(base) => base.join(name),
        None => std::env::temp_dir()
            .join(PRODUCT_DIR.to_lowercase())
            .join(name),
    }
}

/// 共享数据目录下的一个子目录；无法确定共享根目录时为 `None`。
pub fn shared_data_subdir(name: &str) -> Option<PathBuf> {
    Some(shared_data_dir()?.join(name))
}

/// 一条"旧位置 → 新位置"的映射。
///
/// 同时服务两件事：一次性迁移（`app_paths_migrate`）与工程里历史绝对路径的读时
/// 重映射（`project::resolve_source_paths_on_open`）。两件事必须用**同一份**映射 ——
/// 否则"迁移搬到了 A，重映射却指向 B"，症状是老工程打开后素材缺失。
#[derive(Debug, Clone)]
pub struct LegacyRoot {
    /// 历史位置。
    pub from: PathBuf,
    /// 该旧根在新布局下的落点。
    pub to: PathBuf,
    /// 根内子项的重定向（缓存拆分：`hifishifter\waveform_peaks_cache` →
    /// `HiFiShifter\cache\waveform_peaks_cache`）。
    pub redirects: Vec<(String, PathBuf)>,
}

impl LegacyRoot {
    /// 把一条落在本旧根之下的路径映射到新布局；不属于本根时返回 None。
    pub fn remap(&self, path: &Path) -> Option<PathBuf> {
        let rest = strip_prefix_ci(path, &self.from)?;
        let mut components = rest.components();
        if let Some(Component::Normal(first)) = components.next() {
            let first = first.to_string_lossy();
            for (name, dest) in &self.redirects {
                if component_name_eq(&first, name) {
                    return Some(components.fold(dest.clone(), |acc, part| acc.join(part)));
                }
            }
        }
        Some(self.to.join(rest))
    }
}

/// 历史布局的各个根（**迁移**与**读时重映射**的唯一事实来源）。
///
/// 只列出确实存在的形态：环境变量缺失（例如没有 `APPDATA`）时对应项直接不出现，
/// 调用方不需要再判空。
pub fn legacy_roots() -> Vec<LegacyRoot> {
    let mut roots: Vec<LegacyRoot> = Vec::new();

    #[cfg(windows)]
    {
        if let Some(roaming) = roaming_base() {
            if let Some(to) = config_dir() {
                roots.push(LegacyRoot {
                    from: roaming.join(LEGACY_IDENTIFIER).join(PRODUCT_DIR),
                    to,
                    redirects: Vec::new(),
                });
            }
        }
        if let Some(local) = local_base() {
            let legacy_local = local.join(LEGACY_IDENTIFIER);
            if let (Some(local_root), Some(cache_root)) = (local_data_dir(), cache_dir()) {
                roots.push(LegacyRoot {
                    from: legacy_local.join("hifishifter"),
                    to: local_root,
                    redirects: vec![
                        (
                            "waveform_peaks_cache".to_string(),
                            cache_root.join("waveform_peaks_cache"),
                        ),
                        ("render_cache".to_string(), cache_root.join("render_cache")),
                    ],
                });
            }
            if let (Some(log_root), Some(local_root)) = (log_dir(), local_data_dir()) {
                roots.push(LegacyRoot {
                    from: legacy_local.join("logs"),
                    to: log_root,
                    redirects: Vec::new(),
                });
                roots.push(LegacyRoot {
                    from: legacy_local.join("EBWebView"),
                    to: local_root.join("EBWebView"),
                    redirects: Vec::new(),
                });
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = home_dir() {
            if let Some(to) = config_dir() {
                roots.push(LegacyRoot {
                    from: home
                        .join("Library")
                        .join("Application Support")
                        .join(LEGACY_IDENTIFIER)
                        .join(PRODUCT_DIR),
                    to,
                    redirects: Vec::new(),
                });
            }
            if let Some(cache_to) = cache_dir() {
                roots.push(LegacyRoot {
                    from: home
                        .join("Library")
                        .join("Caches")
                        .join(LEGACY_IDENTIFIER)
                        .join(PRODUCT_DIR),
                    to: cache_to,
                    redirects: Vec::new(),
                });
            }
            if let Some(log_to) = log_dir() {
                roots.push(LegacyRoot {
                    from: home.join("Library").join("Logs").join(LEGACY_IDENTIFIER),
                    to: log_to,
                    redirects: Vec::new(),
                });
            }
        }
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(to) = config_dir() {
            roots.push(LegacyRoot {
                from: xdg_base("XDG_CONFIG_HOME", ".config")
                    .unwrap_or_default()
                    .join(LEGACY_IDENTIFIER)
                    .join(PRODUCT_DIR),
                to,
                redirects: Vec::new(),
            });
        }
        if let (Some(local_to), Some(cache_to)) = (local_data_dir(), cache_dir()) {
            roots.push(LegacyRoot {
                from: xdg_base("XDG_CACHE_HOME", ".cache")
                    .unwrap_or_default()
                    .join(LEGACY_IDENTIFIER)
                    .join(PRODUCT_DIR),
                to: local_to,
                redirects: vec![
                    (
                        "waveform_peaks_cache".to_string(),
                        cache_to.join("waveform_peaks_cache"),
                    ),
                    ("render_cache".to_string(), cache_to.join("render_cache")),
                ],
            });
        }
        if let Some(log_to) = log_dir() {
            roots.push(LegacyRoot {
                from: xdg_base("XDG_DATA_HOME", ".local/share")
                    .unwrap_or_default()
                    .join(LEGACY_IDENTIFIER)
                    .join("logs"),
                to: log_to,
                redirects: Vec::new(),
            });
        }
    }

    roots
}

/// 把「应用私有目录」的历史绝对路径映射到当前路径。
///
/// 【为什么需要读时重映射】工程文件里存的是**绝对**路径（`take.source_path`），
/// 迁移把素材搬走之后，旧工程就指向了不存在的位置 —— 用户看到的是"素材缺失"，
/// 而文件明明就在硬盘上。这里只做**读时**映射，不写回：用户下次保存时路径自然
/// 更新，而在此之前打开旧工程的行为与从前完全一致。
///
/// 命中条件与返回：
/// - 路径落在 `legacy_roots()` 的某个旧根之下，且映射后的文件**存在** → 返回新路径；
/// - 命中旧根但新位置没有这个文件 → 返回 None（保持原路径，走既有"缺失文件"流程）；
/// - 不命中任何旧根（用户自己的素材路径）→ 返回 None（一个字都不动）。
pub fn remap_legacy_app_path(path: &str) -> Option<PathBuf> {
    remap_with(&legacy_roots(), path)
}

/// [`remap_legacy_app_path`] 的**纯**实现：旧根集合由调用方给定。
///
/// 单测因此可以在临时目录里构造任意新旧布局，而不必去改进程级环境变量（那会让
/// 并行用例互相踩）。
pub fn remap_with(roots: &[LegacyRoot], path: &str) -> Option<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let original = Path::new(trimmed);
    for root in roots {
        let Some(mapped) = root.remap(original) else {
            continue;
        };
        return if mapped.exists() { Some(mapped) } else { None };
    }
    None
}

/// 逐组件去掉前缀；Windows 上大小写不敏感（用户目录的大小写随系统语言变化）。
///
/// 对 crate 内公开：`media_paths::rebase_path` 用同一套比较规则，否则"迁移认得
/// 的路径"与"搬家认得的路径"会在大小写上分叉。
pub(crate) fn strip_prefix_ci(path: &Path, prefix: &Path) -> Option<PathBuf> {
    let mut path_components = path.components();
    for prefix_component in prefix.components() {
        let component = path_components.next()?;
        if !component_eq(component, prefix_component) {
            return None;
        }
    }
    Some(path_components.collect())
}

fn component_eq(left: Component<'_>, right: Component<'_>) -> bool {
    #[cfg(windows)]
    {
        left.as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

fn component_name_eq(left: &str, right: &str) -> bool {
    #[cfg(windows)]
    {
        left.eq_ignore_ascii_case(right)
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 三个平台的目录都必须能被用户认出来：产品目录名是唯一看得见的那一层。
    #[test]
    fn every_user_directory_names_the_product() {
        for (label, dir) in [
            ("config", config_dir()),
            ("local", local_data_dir()),
            ("cache", cache_dir()),
            ("log", log_dir()),
        ] {
            let Some(dir) = dir else {
                // 环境缺少 HOME/APPDATA 时无从判定，跳过（CI 与本地开发都会命中）。
                continue;
            };
            assert!(
                dir.to_string_lossy().contains(PRODUCT_DIR),
                "{label} 目录 {dir:?} 不含 {PRODUCT_DIR}"
            );
        }
    }

    /// 缓存与日志必须与本地数据分开：`HiFiShifter\cache` / `HiFiShifter\logs` 是
    /// 用户与清理工具能安全识别的两层，混进本地数据就再也分不出来。
    #[test]
    fn caches_and_logs_live_below_the_local_root() {
        #[cfg(windows)]
        {
            let (Some(local), Some(cache), Some(log)) = (local_data_dir(), cache_dir(), log_dir())
            else {
                return;
            };
            assert_eq!(cache, local.join("cache"));
            assert_eq!(log, local.join("logs"));
        }
    }

    /// 三个平台都必须有明确的共享库落点；Windows 只有一个，Unix 允许用户级
    /// 与系统级共存。
    #[test]
    fn every_installer_platform_has_a_shared_store_root() {
        let dirs = shared_data_dirs();
        assert!(!dirs.is_empty());
        assert!(dirs.iter().all(|dir| dir.ends_with(PRODUCT_DIR)));

        #[cfg(windows)]
        assert_eq!(dirs.len(), 1);

        #[cfg(target_os = "macos")]
        assert!(
            dirs[0].starts_with("/Library"),
            "macOS 的首选共享库应来自统一 pkg 的 /Library 落点：{:?}",
            dirs[0]
        );

        #[cfg(all(unix, not(target_os = "macos")))]
        assert!(
            dirs.iter().any(|dir| dir.ends_with(PRODUCT_DIR)),
            "Linux 缺少共享库候选：{dirs:?}"
        );

        assert!(shared_data_subdir("models").is_some());
    }

    /// 旧根的映射必须把子路径整段带过去，且缓存被**拆到**缓存根。
    #[test]
    fn a_legacy_root_remaps_children_and_splits_caches() {
        let root = LegacyRoot {
            from: PathBuf::from("/legacy/com.arounder.hifishifter/hifishifter"),
            to: PathBuf::from("/new/HiFiShifter"),
            redirects: vec![(
                "waveform_peaks_cache".to_string(),
                PathBuf::from("/new/HiFiShifter/cache/waveform_peaks_cache"),
            )],
        };
        assert_eq!(
            root.remap(Path::new(
                "/legacy/com.arounder.hifishifter/hifishifter/pcm/a.raw"
            )),
            Some(PathBuf::from("/new/HiFiShifter/pcm/a.raw"))
        );
        assert_eq!(
            root.remap(Path::new(
                "/legacy/com.arounder.hifishifter/hifishifter/waveform_peaks_cache/x.hfs"
            )),
            Some(PathBuf::from(
                "/new/HiFiShifter/cache/waveform_peaks_cache/x.hfs"
            ))
        );
        // 不在旧根之下的路径一个字都不动。
        assert_eq!(root.remap(Path::new("/home/user/Music/take.wav")), None);
    }
}
