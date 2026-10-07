//! 共享模型库：让独立 App 与 ARA 插件共用同一份 152 MB 的模型文件。
//!
//! 【为什么需要它】两个形态各自从自己的目录读模型：App 从 `resource_dir()/models`，
//! 插件从 `Contents/Resources/models`。同时安装两者就是两份完全相同的 152 MB ——
//! 用户为此付了两倍磁盘，而内容一字不差。
//!
//! 【设计】模型有一份"共享库"，两个形态都**优先**用它；库不存在时从自己的 bundle
//! 建立（同卷用硬链接，零额外字节）。bundle 里的副本因此退化为兜底：
//! - 便携版（手动拷贝 ZIP）仍然自包含，不依赖任何外部目录；
//! - 安装器版可以把模型只装进共享库，bundle 里不留副本。
//!
//! 解析顺序：`HIFISHIFTER_MODELS_DIR` > 共享库 > bundle。
//!
//! 【为什么用版本号目录】模型换一版就必须重建，否则新代码会读到旧权重 —— 那种错误
//! 不会报错，只会让推理结果悄悄变差。版本号由**内容**派生（见
//! `tools/write-models-manifest.ps1`），因此模型一变版本就变，不需要人工维护。
//!
//! 【为什么共享库在全机目录而不是用户目录】安装器以管理员运行。若它把 152 MB 写进
//! `%LOCALAPPDATA%`，"标准用户 + 输入管理员密码"这种提权方式写的就是**管理员**的
//! 配置目录，而运行插件的是标准用户 —— 插件找不到模型，症状只是"推理不可用"。
//! `%PROGRAMDATA%` 谁提权都能写对位置，所有用户都读得到，且默认允许普通用户创建
//! 子目录（所以运行时发布模型也不需要提权）。

use std::path::{Path, PathBuf};

/// 覆盖模型目录的环境变量（CI、排障、自定义模型布局）。
pub const MODELS_DIR_ENV: &str = "HIFISHIFTER_MODELS_DIR";

/// 模型清单文件名。两个 bundle 都必须带上它：没有清单就无法判断共享库是否可用。
pub const MANIFEST_FILE: &str = "models.json";

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

/// 解析结果。
#[derive(Debug, Clone)]
pub struct ModelOrigin {
    pub dir: PathBuf,
    pub source: ModelSource,
    /// 共享库是否在本次调用中被建立（首次运行时为 true）。
    pub published: bool,
}

/// 读取清单；缺失或损坏时返回 `None`（调用方退回旧行为）。
pub fn read_manifest(models_dir: &Path) -> Option<ModelManifest> {
    let raw = std::fs::read_to_string(models_dir.join(MANIFEST_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// 廉价校验：清单里列出的文件都在、且字节数一致。
///
/// 【为什么不算 sha256】这里在每次启动的路径上。152 MB 的哈希要几百毫秒，而字节数
/// 一致已经能挡住绝大多数实际故障（拷贝中断、只复制了一部分、版本混装）。完整校验
/// 留给诊断导出。
pub fn looks_complete(models_dir: &Path, manifest: &ModelManifest) -> bool {
    if manifest.files.is_empty() {
        return false;
    }
    manifest.files.iter().all(|file| {
        std::fs::metadata(models_dir.join(&file.path))
            .map(|meta| meta.is_file() && meta.len() == file.bytes)
            .unwrap_or(false)
    })
}

/// 共享库中该版本的位置。
///
/// 落在**全机共享**目录（Windows 上是 `%PROGRAMDATA%\HiFiShifter\models\<version>`）：
/// 安装器以管理员身份运行，写进用户目录在"标准用户 + 管理员密码"提权时会落到管理员
/// 的配置目录里，而运行插件的是标准用户 —— 见 `config_location::default_shared_data_dir`。
pub fn shared_store_dir(version: &str) -> PathBuf {
    crate::config_location::shared_data_subdir("models").join(version)
}

/// 把一个完整可用的模型目录"发布"到共享库。
///
/// 优先硬链接（同卷时零额外字节 —— 这正是"不再重复占用 152 MB"的关键），失败则退回
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
    // 1) 显式覆盖：直接用它，不参与共享库的建立（用户可能故意指向别处）。
    if let Some(raw) = std::env::var_os(MODELS_DIR_ENV) {
        let raw = raw.to_string_lossy().trim().to_owned();
        if !raw.is_empty() {
            let dir = PathBuf::from(raw);
            if register(&dir) {
                return Ok(ModelOrigin {
                    dir,
                    source: ModelSource::Override,
                    published: false,
                });
            }
            log::warn!("{MODELS_DIR_ENV} 指向的目录缺少模型文件：{}", dir.display());
        }
    }

    let bundle_manifest = read_manifest(bundle_models);
    let bundle_usable = bundle_manifest
        .as_ref()
        .map(|manifest| looks_complete(bundle_models, manifest))
        .unwrap_or(false);

    // 2) 共享库：只有在清单能对上、且文件齐全时才用。
    if let Some(manifest) = &bundle_manifest {
        let shared = shared_store_dir(&manifest.version);
        let shared_usable = read_manifest(&shared)
            .map(|stored| stored.version == manifest.version && looks_complete(&shared, &stored))
            .unwrap_or(false);
        if shared_usable && register(&shared) {
            return Ok(ModelOrigin {
                dir: shared,
                source: ModelSource::Shared,
                published: false,
            });
        }

        // 3) 共享库缺失/不完整 → 从 bundle 建立。失败只是回到 bundle，不影响可用性。
        if bundle_usable {
            match publish(bundle_models, manifest, &shared) {
                Ok(()) => {
                    if register(&shared) {
                        return Ok(ModelOrigin {
                            dir: shared,
                            source: ModelSource::Shared,
                            published: true,
                        });
                    }
                }
                Err(error) => {
                    // 只读卷、权限不足、磁盘满都会走到这里 —— 不是致命错误。
                    log::warn!("建立共享模型库失败，改用 bundle 内的副本：{error}");
                }
            }
        }
    }

    // 4) 兜底：bundle 里的副本（便携版与手动安装走这条路）。
    if register(bundle_models) {
        return Ok(ModelOrigin {
            dir: bundle_models.to_path_buf(),
            source: ModelSource::Bundle,
            published: false,
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
    /// 【为什么这条必须有】清单是生成物（`tools/write-models-manifest.ps1`），而
    /// 运行时要靠它判断共享库是否可用。清单一旦与文件不符（换了模型没重新生成、
    /// 或手工改过），共享库就会被永久判为"不完整"，两个形态每次都退回各自的 bundle
    /// —— 重复占用 152 MB，且没有任何报错。这条测试让它变成构建失败。
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
        let missing: Vec<&str> = manifest
            .files
            .iter()
            .filter(|file| !models.join(&file.path).is_file())
            .map(|file| file.path.as_str())
            .collect();
        assert!(missing.is_empty(), "清单列了不存在的文件：{missing:?}");
        assert!(
            looks_complete(&models, &manifest),
            "清单里的字节数与实际文件不符 —— 重新运行 tools/write-models-manifest.ps1"
        );
    }

    /// 共享库路径带版本号：不同版本必须落在不同目录，否则新代码会读到旧权重。
    #[test]
    fn the_shared_store_is_versioned() {
        let a = shared_store_dir("aaa");
        let b = shared_store_dir("bbb");
        assert_ne!(a, b);
        assert!(a.ends_with("aaa"));
        assert!(a.parent().unwrap().ends_with("models"));
    }
}
