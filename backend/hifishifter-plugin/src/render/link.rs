//! 宿主链接的运行时：导出工作区、监视回写的编辑。
//!
//! # 它与管道时代的那条「通道」差在哪
//!
//! 旧实现是**每个编辑实例一个命名管道服务端**：线程、心跳文件、令牌、SDDL、
//! 发现目录、单客户端上限，全都挂在实例的生命周期上。于是「关掉插件窗口」
//! 就等于「链接消失」，而独立 App 里已经拿到的工程也跟着失去意义。
//!
//! 现在这里只剩**一份进程级状态**：用户导出了哪个工作区、有没有新编辑等着合并。
//! 没有常驻线程（导出在一次性工作线程上），没有心跳，没有令牌，没有发现。
//! 宿主进程退出之后工作区仍在磁盘上，独立 App 照常打开它 —— 这是**特性**，不是缺口。
//!
//! # 为什么导出去工作线程
//!
//! 导出要写音频（可能上百 MB）。命令分发与 actor 循环是同一个线程，在那里同步写完
//! 会让 UI 在整段时间里无响应。所以导出起一个一次性线程，状态由 UI 轮询。

use super::extension::ExtensionOwner;
use hifishifter_host_link::{now_ms, LinkManifest, LinkStatus, Workspace};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// 当前的工作区状态。
///
/// 【为什么把数据放进枚举的变体】「正在进行导出」与「已经有一份工作区」不可能同时成立。
/// 用一个 `Option<Workspace>` 加若干散落的布尔值表达它，就得在每一处读点自己判断组合，
/// 而漏判的那一处会显示上一次导出的摘要 —— 正是最难发现的一类界面缺陷。
enum Phase {
    /// 工作线程正在写。
    Running,
    /// 有一份可用的工作区。
    Ready {
        workspace: Workspace,
        identity: String,
        manifest: Box<LinkManifest>,
        /// 上一次**已经处理过**的编辑文件时间戳。
        ///
        /// 【为什么用 mtime 而不是轮询整个目录】actor 循环每秒醒几次；对一个已知路径
        /// 做一次 `metadata` 是最便宜的「有没有新东西」判据。
        seen_edit_stamp: Option<u64>,
        last_status: Option<LinkStatus>,
    },
    /// 导出失败，附原因。
    Failed { error: String },
}

fn phase() -> &'static Mutex<Option<Phase>> {
    static PHASE: OnceLock<Mutex<Option<Phase>>> = OnceLock::new();
    PHASE.get_or_init(|| Mutex::new(None))
}

fn lock() -> std::sync::MutexGuard<'static, Option<Phase>> {
    phase()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 单调递增，用来作废被后来者取代的导出。
static EXPORT_EPOCH: AtomicU64 = AtomicU64::new(0);

/// 工作区根目录；用户可用 `HIFISHIFTER_LINK_DIR` 覆盖（验收与测试会用到）。
pub(crate) fn root() -> Result<PathBuf, String> {
    hifishifter_host_link::workspace::default_root()
        .ok_or_else(|| "no writable data directory for link workspaces".to_string())
}

/// 开始一次导出。立即返回；结果由 [`status`] 轮询。
///
/// 【为什么返回值不当成结果】导出是重的（写音频 + 序列化整个时间轴）。
/// 把它放在命令分发线程上同步做完，UI 就会在这段时间里卡住 ——
/// 而「点一下导出，然后在独立 App 里编辑」正是这个功能唯一的用法。
pub(crate) fn start_export(owner: Arc<ExtensionOwner>) -> Result<serde_json::Value, String> {
    let root = root()?;
    let epoch = EXPORT_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    // 换工作区时先把旧的读数丢掉：新导出还没完成之前，摘要不该显示上一次的。
    *lock() = Some(Phase::Running);
    std::thread::Builder::new()
        .name("hfs-link-export".into())
        .spawn(move || {
            let outcome = owner.export_link(&root);
            let mut guard = lock();
            // 这次导出已被后来的一次取代：结果没有意义，也不要覆盖新状态。
            if EXPORT_EPOCH.load(Ordering::Acquire) != epoch {
                return;
            }
            match outcome {
                Ok((manifest, workspace)) => {
                    log::info!(
                        "[link] exported {} clips / {} sources",
                        manifest.clip_count(),
                        manifest.sources.len()
                    );
                    *guard = Some(Phase::Ready {
                        workspace,
                        identity: manifest.link_id.clone(),
                        manifest: Box::new(manifest),
                        // 【为什么不是 None 之后的第一次读】导出**刚刚**写的 status 是初值。
                        // 这里把当前编辑文件的时间戳记下来，actor 就不会把上一次会话
                        // 遗留在工作区里的编辑当成新编辑重放一遍。
                        seen_edit_stamp: None,
                        last_status: None,
                    });
                }
                Err(error) => {
                    log::warn!("[link] export failed: {error}");
                    *guard = Some(Phase::Failed { error });
                }
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(status())
}

/// 停止监视（用户点了「断开」/ 换了工作区）。
pub(crate) fn release() -> serde_json::Value {
    *lock() = None;
    status()
}

/// 当前状态的 JSON 形状（插件模式的前端直接消费）。
pub(crate) fn status() -> serde_json::Value {
    let guard = lock();
    match guard.as_ref() {
        None => serde_json::json!({ "attached": false }),
        Some(Phase::Running) => serde_json::json!({ "attached": true, "phase": "running" }),
        Some(Phase::Failed { error }) => {
            serde_json::json!({ "attached": true, "phase": "failed", "error": error })
        }
        Some(Phase::Ready {
            workspace,
            identity,
            manifest,
            last_status,
            ..
        }) => {
            let (complete, partial, external) = manifest.audio_summary();
            serde_json::json!({
                "attached": true,
                "phase": "ready",
                "identity": identity,
                "hostLabel": manifest.host_label,
                "sourceKind": manifest.source_kind.tag(),
                "directory": workspace.root().display().to_string(),
                "exportedAt": manifest.created_unix_ms,
                "clips": manifest.clip_count(),
                "tracks": manifest.track_count(),
                "audio": { "complete": complete, "partial": partial, "extern": external },
                "losses": manifest.losses.len(),
                "acceptsCommit": manifest.grants(hifishifter_host_link::Capability::Commit),
                "diagnostics": manifest.diagnostics,
                "lastEdit": last_status.as_ref().map(|status| serde_json::json!({
                    "ok": status.ok,
                    "code": status.code,
                    "error": status.error,
                    "edits": status.revisions.edits,
                    "model": status.revisions.model,
                    "atUnixMs": status.updated_unix_ms,
                })),
            })
        }
    }
}

/// 监视回写：有新的 `edits.hfsedit` 就合并它，并把回答落盘。
///
/// 【为什么挂在会话 actor 上】与 Capture 的补缓冲同一个理由：actor 由**会话**创建，
/// 与「插件窗口开没开」无关，而且它本来就在做需要串行化的编辑工作。
/// 挂在 WebView 的定时器上会让「关掉窗口」等于「不再接受回写」。
pub(crate) fn poll(owner: &Arc<ExtensionOwner>) {
    let (workspace, identity, previous_stamp) = {
        let guard = lock();
        let Some(Phase::Ready {
            workspace,
            identity,
            seen_edit_stamp,
            ..
        }) = guard.as_ref()
        else {
            return;
        };
        (workspace.clone(), identity.clone(), *seen_edit_stamp)
    };
    let Some(current_stamp) = workspace.edit_stamp_ms() else {
        return;
    };
    if Some(current_stamp) == previous_stamp {
        return;
    }
    // 编辑文件可能正写到一半 —— 原子改名保证读到的要么是旧的、要么是完整的，
    // 所以读失败只意味着「再等一轮」，不是错误。
    let Ok(edit) = workspace.read_edit() else {
        return;
    };
    if edit.link_id != identity {
        // 不属于这份清单的编辑：**不要**消费它的时间戳，否则它会一直躺在那里
        // 却永远不被处理；把它记成一个明确的错误，让 UI 说出来。
        let mut guard = lock();
        if let Some(Phase::Ready { last_status, .. }) = guard.as_mut() {
            *last_status = Some(LinkStatus {
                format: hifishifter_host_link::STATUS_FORMAT.into(),
                format_version: hifishifter_host_link::FORMAT_VERSION,
                link_id: identity.clone(),
                ok: false,
                code: Some(hifishifter_host_link::error_code::LINK_MISMATCH),
                error: Some(format!(
                    "an edit for {} appeared in this workspace",
                    edit.link_id
                )),
                revisions: Default::default(),
                updated_unix_ms: now_ms(),
            });
        }
        return;
    }
    let status = owner.apply_link_edit(&identity, &edit);
    if let Err(error) = workspace.write_status(&status) {
        log::warn!("[link] cannot write the link status: {error}");
    }
    let mut guard = lock();
    if let Some(Phase::Ready {
        identity: current,
        seen_edit_stamp,
        last_status,
        ..
    }) = guard.as_mut()
    {
        // 【为什么这里再判一次身份】导出可能在这段工作期间被换成了另一个工作区。
        // 往新工作区里记一份属于旧工作区的结果，会让 UI 显示一个不属于它的读数。
        if *current == identity {
            *seen_edit_stamp = Some(current_stamp);
            *last_status = Some(status);
        }
    }
}
