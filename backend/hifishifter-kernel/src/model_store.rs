//! 共享模型库：让独立 App、VST3 与 CLAP 插件共用同一份模型文件。
//!
//! 【为什么需要它】每种形态都有自己的 bundle：App 从 "resource_dir()/models" 读，
//! 插件从 "Contents/Resources/models" 读。若每个 bundle 都带完整权重，同一台
//! 机器上装 App + VST3 + CLAP 就是三份完全相同的权重。统一安装包把权重放到一份
//! 共享库，所有形态只保留 models.json 并从这里读。
//!
//! 【设计】模型有一份"共享库"，所有形态都**优先**用它；库不存在时退回自己的
//! bundle。Windows 的合并安装器、macOS 的统一 pkg、Linux 的三合一 tar.gz 都只
//! 写一份权重；便携版（手动拷贝 ZIP / AppImage）仍然可以自带完整权重，不依赖
//! 外部目录。
//!
//! 解析顺序："HIFISHIFTER_MODELS_DIR" > 用户自备目录（models-user，按件覆盖）>
//! 共享库候选 > bundle。
//!
//! 【为什么用版本号目录】模型换一版就必须重建，否则新代码会读到旧权重 —— 那种错误
//! 不会报错，只会让推理结果悄悄变差。版本号由**内容**派生（见
//! "scripts/setup/fetch-models.py --write-manifest"），因此模型一变版本就变，不需要人工维护。
//!
//! 【共享库落在哪】Windows 是 "%PROGRAMDATA%\\HiFiShifter"（安装器独占写）；
//! macOS 是 "/Library/Application Support/HiFiShifter"（统一 pkg 写）；Linux 默认是
//! "$XDG_DATA_HOME/HiFiShifter"（用户级三合一包），系统安装还会尝试
//! "/usr/local/share/HiFiShifter" 与 "/usr/share/HiFiShifter"。候选顺序见
//! "app_paths::shared_data_dirs"。
//!
//! 【运行时谁写共享库】只有 Windows 的独立 App 会在共享库缺失时从自己的 bundle
//! 建立（同卷硬链接，零额外字节）。macOS/Linux 的共享库由安装器写入，运行时只读；
//! 便携版直接用 bundle，不往用户目录或 /Library 写任何模型。

use std::path::{Path, PathBuf};

/// 可选模型子目录：它们**不参与**“共享模型库是否完整”的判定。
///
/// GAME 是自动音符分块的增量模型；本机只验证 GAME 时，旧版模型尚未下载是正常
/// 状态。把它算进“完整”会让程序拒绝登记共享库，或让构建误以为整个模型树就绪。
pub const OPTIONAL_MODEL_DIRS: &[&str] = &["GAME/"];

/// 覆盖模型目录的环境变量（CI、排障、自定义模型布局）。
pub const MODELS_DIR_ENV: &str = "HIFISHIFTER_MODELS_DIR";

/// 模型清单文件名。两个 bundle 都必须带上它：没有清单就无法判断共享库是否可用。
pub const MANIFEST_FILE: &str = "models.json";

/// 用户自备模型目录的名字（挂在本地数据目录下）。
///
/// 【为什么需要它】共享库与 bundle 都按清单（版本 + 字节数）校验，那是给**安装的**
/// 模型准备的完整性闸，用户换一份自己的权重必然过不了 —— 而且换完连个说法都没有
/// （静默退回 bundle）。`HIFISHIFTER_MODELS_DIR` 是给 CI 与排障的整树覆盖，普通用户
/// 改环境变量不现实。`models-user` 是给"想自己换模型的人"的正式入口：目录下放了哪件
/// 就覆盖哪件（`fcpe/fcpe.onnx`、`nsf_hifigan/pc_nsf_hifigan.onnx`、`hnsep/hnsep.onnx`、
/// `GAME/`），没放的件由共享库/bundle 兜底；校验只有"文件存在"—— 自备权重的正确性
/// 由使用者负责（加载失败会在日志与界面上如实报错）。
pub const USER_MODELS_SUBDIR: &str = "models-user";

/// 用户自备模型目录的完整路径。
pub fn user_models_dir() -> PathBuf {
    crate::app_paths::local_data_subdir(USER_MODELS_SUBDIR)
}

/// 清单里的一个文件。
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelFile {
    /// 相对模型根目录的路径，用 `/` 分隔。
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// 模型清单：版本号 + 文件表。
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelManifest {
    /// 由内容派生的版本号（所有文件 sha256 的汇总）。
    pub version: String,
    pub files: Vec<ModelFile>,
}

/// 模型从哪里来（供日志与诊断展示）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// 环境变量显式指定。
    Override,
    /// 共享库。
    Shared,
    /// 自己 bundle 里的副本。
    Bundle,
}

impl ModelSource {
    /// 稳定标识（写进日志与诊断包；不随界面语言变化）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Override => "override",
            Self::Shared => "shared",
            Self::Bundle => "bundle",
        }
    }
}

/// 解析结果。
#[derive(Debug, Clone)]
pub struct ModelOrigin {
    pub dir: PathBuf,
    pub source: ModelSource,
    /// 共享库是否在本次调用中被建立（首次运行时为 true）。
    pub published: bool,
    /// 清单里的版本号（由模型内容派生）；bundle 不带清单时为 `None`。
    pub version: Option<String>,
    /// 用户自备目录（`models-user`）的按件覆盖是否生效。
    ///
    /// 【为什么是布尔而不是来源】覆盖是**按件**的：换掉声码器的会话里，其余件仍然
    /// 来自 `dir` 指向的安装源。`source` 必须忠实于 `dir`，否则"来源 + 目录"这行
    /// 日志会自相矛盾；覆盖这件事由它自己的位置（`user_overlay`）与启动日志陈述。
    pub user_overlay: bool,
}

impl ModelOrigin {
    /// 一行可读的来源描述：**来源 + 目录 + 版本 + 是否刚发布**。
    ///
    /// 【为什么集中在这里】App 与插件此前各写一份同样的 match，两边的字段与措辞
    /// 稍一改动就分叉，而"模型到底从哪儿来的"正是用户报"推理不可用"时第一个要
    /// 判断的事。诊断包里现在只有一种形状。
    pub fn describe(&self) -> String {
        let version = self.version.as_deref().unwrap_or("unknown");
        format!(
            "{} ({}, version {version}{}{})",
            self.dir.display(),
            self.source.as_str(),
            if self.published {
                ", just published"
            } else {
                ""
            },
            if self.user_overlay {
                ", user-overlay"
            } else {
                ""
            },
        )
    }
}

/// 读取清单；缺失或损坏时返回 `None`（调用方退回旧行为）。
pub fn read_manifest(models_dir: &Path) -> Option<ModelManifest> {
    let raw = std::fs::read_to_string(models_dir.join(MANIFEST_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// 廉价校验：清单里列出的文件都在、且字节数一致。
///
/// 【为什么不算 sha256】这里在每次启动的路径上。对全部权重做哈希要几百毫秒，而字节数
/// 一致已经能挡住绝大多数实际故障（拷贝中断、只复制了一部分、版本混装）。完整校验
/// 留给诊断导出。
pub fn looks_complete(models_dir: &Path, manifest: &ModelManifest) -> bool {
    if manifest.files.is_empty() {
        return false;
    }
    manifest.files.iter().all(|file| {
        if OPTIONAL_MODEL_DIRS
            .iter()
            .any(|prefix| file.path.starts_with(prefix))
        {
            return true;
        }
        std::fs::metadata(models_dir.join(&file.path))
            .map(|meta| meta.is_file() && meta.len() == file.bytes)
            .unwrap_or(false)
    })
}

/// 共享库中该版本的候选位置，按优先级排列。
///
/// 不同安装形态写的位置不同：Windows 合并安装器写 `%PROGRAMDATA%\HiFiShifter`；
/// macOS 统一 pkg 写 `/Library/Application Support/HiFiShifter`；Linux 统一包默认写
/// `$XDG_DATA_HOME/HiFiShifter`，系统安装还可写 `/usr/local/share/HiFiShifter`。
/// 运行时必须逐个尝试，不能假设只有一个位置。
pub fn shared_store_dirs(version: &str) -> Vec<PathBuf> {
    crate::app_paths::shared_data_dirs()
        .into_iter()
        .map(|root| root.join("models").join(version))
        .collect()
}

/// 共享库中该版本的首选位置（保留单值 API；解析时用 [`shared_store_dirs`]）。
pub fn shared_store_dir(version: &str) -> Option<PathBuf> {
    shared_store_dirs(version).into_iter().next()
}
/// 把一个完整可用的模型目录"发布"到共享库。
///
/// 优先硬链接（同卷时零额外字节 —— 这正是"不再重复占用一份权重"的关键），失败则退回
/// 拷贝。先写临时目录再原子改名：中途失败不会留下一个"看起来存在但缺文件"的库，
/// 那会让后续启动误以为库可用。
fn publish(source_dir: &Path, manifest: &ModelManifest, target_dir: &Path) -> Result<(), String> {
    if let Some(parent) = target_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    // 用固定后缀而不是随机名：同卷硬链接失败后残留的目录可以被下一次运行覆盖。
    let staging = target_dir.with_extension("publishing");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("create {}: {e}", staging.display()))?;

    let copy_result = (|| -> Result<(), String> {
        for file in &manifest.files {
            let from = source_dir.join(&file.path);
            let to = staging.join(&file.path);
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            if std::fs::hard_link(&from, &to).is_err() {
                std::fs::copy(&from, &to)
                    .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))?;
            }
        }
        // 清单必须一起发布：没有它，下次启动无法判断这个库是否可用。
        std::fs::copy(source_dir.join(MANIFEST_FILE), staging.join(MANIFEST_FILE))
            .map_err(|e| format!("copy manifest: {e}"))?;
        Ok(())
    })();

    if let Err(error) = copy_result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }

    // 目标已存在时先删（可能是上一次中断留下的残缺库）。
    if target_dir.exists() {
        let _ = std::fs::remove_dir_all(target_dir);
    }
    std::fs::rename(&staging, target_dir).map_err(|e| {
        let _ = std::fs::remove_dir_all(&staging);
        format!("publish {}: {e}", target_dir.display())
    })
}

/// 解析模型目录并登记到 [`crate::model_paths`]。
///
/// `bundle_models` 是调用方自己 bundle 里的模型目录（App 用 `resource_dir()/models`，
/// 插件用 `Contents/Resources/models`）。返回实际使用的来源，供日志与诊断展示。
///
/// **绝不因为共享库不可用而失败**：最坏情况就是退回 bundle —— 那正是今天的行为。
pub fn resolve_and_register(bundle_models: &Path) -> Result<ModelOrigin, String> {
    resolve_models(bundle_models, true, &user_models_dir())
}

/// Plug-in initialization cannot copy a model library on the host UI thread.
/// Prefer an existing shared version, otherwise register the self-contained bundle.
pub fn resolve_and_register_without_publish(bundle_models: &Path) -> Result<ModelOrigin, String> {
    resolve_models(bundle_models, false, &user_models_dir())
}

fn resolve_models(
    bundle_models: &Path,
    allow_publish: bool,
    user_dir: &Path,
) -> Result<ModelOrigin, String> {
    // 1) 显式覆盖：直接用它，不参与共享库的建立（用户可能故意指向别处）。
    if let Some(raw) = std::env::var_os(MODELS_DIR_ENV) {
        let raw = raw.to_string_lossy().trim().to_owned();
        if !raw.is_empty() {
            let dir = PathBuf::from(raw);
            if register(&dir) {
                return Ok(ModelOrigin {
                    version: read_manifest(&dir).map(|manifest| manifest.version),
                    dir,
                    source: ModelSource::Override,
                    published: false,
                    user_overlay: false,
                });
            }
            log::warn!("{MODELS_DIR_ENV} 指向的目录缺少模型文件：{}", dir.display());
        }
    }

    // 2) 用户自备目录：先登记、按件生效。
    //
    // 【为什么是"先登记"而不是"命中即返回"】`crate::model_paths` 的登记处只写一次、
    // 不覆盖 —— 这里先登记就意味着 models-user 里放了的件赢下那一件；没放的件
    // 登记处仍空着，由后面的共享库/bundle 照常补上。于是"只想换声码器"是合法用法：
    // 放一个 `nsf_hifigan/` 子目录就够了，其余模型不受影响。这也是它与
    // `HIFISHIFTER_MODELS_DIR`（整树覆盖，缺件就是缺件）的核心差别。
    //
    // 【为什么不做完整性校验】安装的模型用清单闸（版本 + 字节数）挡拷贝中断；自备
    // 权重没有清单可言，闸只会把人挡在门外 —— 它的"校验"就是推理加载本身，加载
    // 失败会在日志与界面上如实报错（见 `nsf_hifigan_onnx::probe_load` 一族）。
    let user_dir_used = register(user_dir);
    if user_dir_used {
        log::info!(
            "models: 用户自备模型目录已生效（按件覆盖，缺的件用安装的模型兜底）：{}",
            user_dir.display()
        );
    } else if user_dir.is_dir() {
        // 目录存在却一件没认出来：几乎必然是文件名/布局不对。不报错（这不是故障，
        // 安装链照常走），但要让"为什么我的模型没生效"在日志里一眼可答。
        log::warn!(
            "models: 用户自备模型目录 {} 存在，但没有可识别的模型文件（应为 fcpe/fcpe.onnx、nsf_hifigan/pc_nsf_hifigan.onnx、hnsep/hnsep.onnx 或 GAME/）",
            user_dir.display()
        );
    }

    let bundle_manifest = read_manifest(bundle_models);
    let bundle_usable = bundle_manifest
        .as_ref()
        .map(|manifest| looks_complete(bundle_models, manifest))
        .unwrap_or(false);

    // 3) 共享库：只有在清单能对上、且文件齐全时才用。
    //
    // 非 Windows 的共享库由统一安装包写入：macOS 是 /Library/Application Support，
    // Linux 是 XDG 数据目录或 /usr/local/share。运行时按候选顺序逐个检查，
    // 这样用户级安装与系统级安装可以共存而不会互相遮蔽。
    if let Some(manifest) = &bundle_manifest {
        let shared_dirs = shared_store_dirs(&manifest.version);
        for shared in &shared_dirs {
            let stored = read_manifest(shared);
            let shared_usable = stored
                .as_ref()
                .map(|stored| stored.version == manifest.version && looks_complete(shared, stored))
                .unwrap_or(false);
            // 【为什么值得一条日志】清单在但文件对不上 = 有人动过这个库（中断的
            // 更新、手工替换）。此前它静默退回 bundle，"为什么共享库没被用上"
            // 在日志里没有任何线索。指名道姓列出对不上的文件。
            if let Some(stored) = stored.filter(|_| !shared_usable) {
                let mismatched: Vec<&str> = stored
                    .files
                    .iter()
                    .filter(|file| {
                        !OPTIONAL_MODEL_DIRS
                            .iter()
                            .any(|prefix| file.path.starts_with(prefix))
                            && !std::fs::metadata(shared.join(&file.path))
                                .map(|meta| meta.is_file() && meta.len() == file.bytes)
                                .unwrap_or(false)
                    })
                    .map(|file| file.path.as_str())
                    .collect();
                log::warn!(
                    "models: 共享模型库 {} 版本 {} 文件不齐（{}），退回 bundle；如需自备模型请使用 {} 目录",
                    shared.display(),
                    stored.version,
                    if mismatched.is_empty() {
                        "版本不符".to_owned()
                    } else {
                        mismatched.join(", ")
                    },
                    user_dir.display()
                );
            }
            if shared_usable && register(shared) {
                if user_dir_used {
                    log::info!(
                        "models: 其余模型来自共享库 {}（用户自备目录覆盖的件不受影响）",
                        shared.display()
                    );
                }
                return Ok(ModelOrigin {
                    dir: shared.clone(),
                    source: ModelSource::Shared,
                    published: false,
                    version: Some(manifest.version.clone()),
                    user_overlay: user_dir_used,
                });
            }
        }

        // 4) Windows 安装器之外的运行形态不会写共享库：统一包已经把它装好，
        // 便携版则继续用 bundle。只有 Windows 的 App 仍保留"从 bundle 建立共享库"
        // 的首次启动行为（插件侧始终 allow_publish=false）。
        if bundle_usable && allow_publish && cfg!(windows) {
            if let Some(shared) = shared_dirs.first() {
                match publish(bundle_models, manifest, shared) {
                    Ok(()) => {
                        if register(shared) {
                            return Ok(ModelOrigin {
                                dir: shared.clone(),
                                source: ModelSource::Shared,
                                published: true,
                                version: Some(manifest.version.clone()),
                                user_overlay: user_dir_used,
                            });
                        }
                    }
                    Err(error) => {
                        log::warn!("建立共享模型库失败，改用 bundle 内的副本：{error}");
                    }
                }
            }
        }
    }
    // 5) 兜底：bundle 里的副本（便携版、手动安装、以及全部非 Windows 交付形态）。
    if register(bundle_models) {
        return Ok(ModelOrigin {
            dir: bundle_models.to_path_buf(),
            source: ModelSource::Bundle,
            published: false,
            version: bundle_manifest
                .as_ref()
                .map(|manifest| manifest.version.clone()),
            user_overlay: user_dir_used,
        });
    }

    Err(format!(
        "模型不可用：bundle 目录 {} 缺少模型文件或 {MANIFEST_FILE}",
        bundle_models.display()
    ))
}

/// 按目录布局登记三处模型路径；目录里没有任何模型时返回 false。
fn register(dir: &Path) -> bool {
    let fcpe = dir.join("fcpe/fcpe.onnx");
    let nsf = dir.join("nsf_hifigan");
    let hnsep = dir.join("hnsep");
    let game = dir.join("GAME");
    let mut any = false;
    if fcpe.is_file() {
        crate::model_paths::set_fcpe_onnx_path(fcpe);
        any = true;
    }
    if nsf.join("pc_nsf_hifigan.onnx").is_file() {
        crate::model_paths::set_nsf_hifigan_model_dir(nsf);
        any = true;
    }
    if hnsep.join("hnsep.onnx").is_file() {
        crate::model_paths::set_hnsep_model_dir(hnsep);
        any = true;
    }
    if crate::game_onnx::models_present(&game) {
        crate::model_paths::set_game_model_dir(game);
        any = true;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hfs-model-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// 造一个"像真模型"的目录：三个文件 + 清单。
    fn fake_models(root: &Path, marker: u8) {
        for (relative, bytes) in [
            ("fcpe/fcpe.onnx", vec![marker; 32]),
            ("nsf_hifigan/pc_nsf_hifigan.onnx", vec![marker; 48]),
            ("hnsep/hnsep.onnx", vec![marker; 64]),
        ] {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &bytes).unwrap();
        }
        let manifest = ModelManifest {
            version: format!("v{marker}"),
            files: vec![
                ModelFile {
                    path: "fcpe/fcpe.onnx".into(),
                    bytes: 32,
                    sha256: String::new(),
                },
                ModelFile {
                    path: "nsf_hifigan/pc_nsf_hifigan.onnx".into(),
                    bytes: 48,
                    sha256: String::new(),
                },
                ModelFile {
                    path: "hnsep/hnsep.onnx".into(),
                    bytes: 64,
                    sha256: String::new(),
                },
            ],
        };
        std::fs::write(
            root.join(MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    /// 缺文件 / 字节数不符 → 判定为不可用。
    #[test]
    fn completeness_requires_every_file_at_the_right_size() {
        let dir = scratch("complete");
        fake_models(&dir, 1);
        let manifest = read_manifest(&dir).expect("manifest");
        assert!(looks_complete(&dir, &manifest));

        std::fs::remove_file(dir.join("hnsep/hnsep.onnx")).unwrap();
        assert!(!looks_complete(&dir, &manifest), "缺文件却判定为完整");

        fake_models(&dir, 1);
        std::fs::write(dir.join("fcpe/fcpe.onnx"), vec![1u8; 8]).unwrap();
        assert!(!looks_complete(&dir, &manifest), "字节数不符却判定为完整");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 发布必须把文件与清单都放到目标目录，并且**不留残余的临时目录**。
    #[test]
    fn publish_moves_files_and_manifest_without_leftovers() {
        let source = scratch("publish-src");
        fake_models(&source, 7);
        let manifest = read_manifest(&source).unwrap();
        let target = scratch("publish-dst").join("nested").join("v7");

        publish(&source, &manifest, &target).expect("publish");
        assert!(looks_complete(&target, &read_manifest(&target).unwrap()));
        assert!(
            !target.with_extension("publishing").exists(),
            "残留了发布临时目录"
        );

        // 硬链接或拷贝都行：内容必须一致。
        assert_eq!(
            std::fs::read(target.join("fcpe/fcpe.onnx")).unwrap().len(),
            32
        );

        let _ = std::fs::remove_dir_all(&source);
        let _ = std::fs::remove_dir_all(target.parent().unwrap().parent().unwrap());
    }

    /// 覆盖发布：目标已有旧内容时必须被替换，而不是留下混合状态。
    #[test]
    fn publish_replaces_an_existing_store() {
        let source = scratch("publish-replace");
        fake_models(&source, 9);
        let manifest = read_manifest(&source).unwrap();
        let root = scratch("publish-replace-root");
        let target = root.join("v9");

        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("stale.txt"), b"old").unwrap();

        publish(&source, &manifest, &target).expect("publish");
        assert!(!target.join("stale.txt").exists(), "旧内容没有被清掉");
        assert!(looks_complete(&target, &read_manifest(&target).unwrap()));

        let _ = std::fs::remove_dir_all(&source);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 清单缺失 → 不 panic，只是判定为不可用（调用方据此退回 bundle）。
    #[test]
    fn a_missing_manifest_is_not_an_error() {
        let dir = scratch("no-manifest");
        assert!(read_manifest(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 仓库里那份 `models.json` 必须与**真实存在的模型文件**对得上。
    ///
    /// 【为什么这条必须有】清单是生成物（`scripts/setup/fetch-models.py --write-manifest`），而
    /// 运行时要靠它判断共享库是否可用。清单一旦与文件不符（换了模型没重新生成、
    /// 或手工改过），共享库就会被永久判为"不完整"，两个形态每次都退回各自的 bundle
    /// —— 重复占用一份权重，且没有任何报错。这条测试让它变成构建失败。
    #[test]
    fn the_committed_manifest_matches_the_real_model_files() {
        let models =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../src-tauri/resources/models");
        if !models.is_dir() {
            // 模型是 git 跟踪的，正常检出一定在；缺失说明这份检出被裁剪过。
            eprintln!("skipping: {} is missing", models.display());
            return;
        }
        let manifest = read_manifest(&models).expect("resources/models/models.json must exist");
        assert!(
            manifest.version.len() == 12 && manifest.version.chars().all(|c| c.is_ascii_hexdigit()),
            "版本号应由内容派生为 12 位十六进制：{}",
            manifest.version
        );
        // 权重自 v0.2 起不再进仓库（scripts/setup/fetch-models.py）：没取过权重的检出
        // 是**清单在、文件不在**的正常状态，不是回归。取过之后再逐条校验字节数 ——
        // 那时缺失就是真的缺了。
        if manifest
            .files
            .iter()
            .any(|file| !models.join(&file.path).is_file())
        {
            eprintln!("skipping: model weights not fetched (run scripts/setup/fetch-models.py)");
            return;
        }
        assert!(
            looks_complete(&models, &manifest),
            "清单里的字节数与实际文件不符 —— 重新运行 scripts/setup/fetch-models.py --write-manifest"
        );
    }

    /// 共享库路径带版本号：不同版本必须落在不同目录，否则新代码会读到旧权重。
    #[test]
    fn the_shared_store_is_versioned() {
        let a = shared_store_dir("aaa").expect("shared store");
        let b = shared_store_dir("bbb").expect("shared store");
        assert_ne!(a, b);
        assert!(a.ends_with("aaa"));
        assert!(a.parent().unwrap().ends_with("models"));
    }

    /// 三个平台都必须能解析出共享库路径；Unix 还允许用户级与系统级候选共存。
    #[test]
    fn shared_store_is_available_on_every_installer_platform() {
        let dirs = shared_store_dirs("v1");
        assert!(!dirs.is_empty());
        assert!(dirs.iter().all(|dir| dir.ends_with("v1")));
    }

    /// 非 Windows：解析模型绝不能往用户目录写任何东西。
    ///
    /// 【为什么值得单独钉一条】历史实现在这两条平台上退回本地数据目录，结果是每个
    /// 用户首启就往 `~/Library/Caches` / `$XDG_CACHE_HOME` 里多写一份权重 ——
    /// 而那正是系统会清理的目录，AppImage 上还必然退化成真拷贝（只读挂载）。
    ///
    /// 【为什么走 resolve_models 而不是公共入口】公共入口读**真实的**用户自备目录；
    /// 开发机上若存在它，本断言（source == Bundle）就会被它推翻。注入一个不存在的
    /// 路径，把测试与开发机的目录状态解耦。
    #[test]
    fn resolve_and_register_never_writes_outside_windows() {
        if cfg!(windows) {
            // Windows 上共享库是正常路径，由 publish_* 系列测试覆盖。
            return;
        }
        let bundle = scratch("resolve-bundle");
        fake_models(&bundle, 3);
        let cache_root = crate::app_paths::cache_dir();
        let cache_existed = cache_root.as_ref().map(|dir| dir.exists()).unwrap_or(false);

        let origin = resolve_models(
            &bundle,
            false,
            Path::new("hfs-test-no-such-user-models-dir"),
        )
        .expect("bundle models are usable");
        assert_eq!(origin.source, ModelSource::Bundle);
        assert!(!origin.published);
        assert!(!origin.user_overlay);
        assert_eq!(origin.version.as_deref(), Some("v3"));

        if let Some(cache_root) = cache_root {
            if !cache_existed {
                assert!(
                    !cache_root.exists(),
                    "非 Windows 不该建立任何缓存目录：{}",
                    cache_root.display()
                );
            }
        }
        let _ = std::fs::remove_dir_all(&bundle);
    }

    /// 用户自备目录按件覆盖：只放声码器 ⇒ 覆盖生效（`user_overlay`），其余件照常
    /// 由安装链兜底；目录里放什么都不影响"来源目录"的忠实性。
    #[test]
    fn a_user_models_dir_overrides_per_piece() {
        let bundle = scratch("overlay-bundle");
        fake_models(&bundle, 4);
        let user = scratch("overlay-user");
        // 只放一件，且字节数故意与 bundle 清单不符（自备权重没有清单可言）。
        let nsf = user.join("nsf_hifigan");
        std::fs::create_dir_all(&nsf).unwrap();
        std::fs::write(nsf.join("pc_nsf_hifigan.onnx"), vec![9u8; 128]).unwrap();

        let origin = resolve_models(&bundle, false, &user).expect("bundle fallback is usable");
        assert!(origin.user_overlay, "自备目录放了一件就该生效");
        assert_eq!(
            origin.source,
            ModelSource::Bundle,
            "来源目录仍是兜底的 bundle"
        );
        assert_eq!(origin.dir, bundle);

        // 空目录/不存在：不生效，也不报错。
        let empty = scratch("overlay-empty");
        let origin = resolve_models(&bundle, false, &empty).expect("usable");
        assert!(!origin.user_overlay, "空的自备目录不该生效");
        let origin =
            resolve_models(&bundle, false, Path::new("hfs-test-no-such-dir")).expect("usable");
        assert!(!origin.user_overlay, "不存在的自备目录不该生效");

        let _ = std::fs::remove_dir_all(&bundle);
        let _ = std::fs::remove_dir_all(&user);
        let _ = std::fs::remove_dir_all(&empty);
    }
}
