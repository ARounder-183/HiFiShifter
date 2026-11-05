//! 导入素材的落点：**工程素材目录**或**未保存会话目录**。
//!
//! 【为什么不再往设置目录里塞】历史实现把拖放导入的文件写进
//! `<设置目录>/imported_media/import_<uuid>.wav`。三个后果：
//!
//! 1. **设置目录里躺着几百 MB 音频**。用户备份 / 同步设置时会把素材一起带上，
//!    而"设置"本该是可以随手删掉重来的东西；
//! 2. 工程搬走（拷到另一台机器、发给别人）之后，素材仍然钉在原机器的设置目录里 ——
//!    工程看起来"自包含"，实际上换台机器就缺文件；
//! 3. 谁也说不清 `imported_media` 里的哪个文件还有用：它不属于任何工程。
//!
//! 现在的规则只有两条，且都能用一句话说清：
//!
//! | 情况 | 落点 |
//! | --- | --- |
//! | 工程**已保存** | `<工程目录>/<工程文件名去扩展名>_media/` |
//! | 工程**未保存** | `<本地数据根>/unsaved/<会话 id>/` |
//!
//! 保存工程时，未保存会话里的文件会被搬进工程素材目录并改写时间线里的路径
//! （见 `crate::commands::project::save_project_to_path_inner`）。于是"工程 + 它的
//! `_media` 目录"就是一个可以整体拷走的单元。
//!
//! 【为什么磁盘文件名仍是 import_<uuid>】导入的两个文件可能同名（`01.wav` 来自两个
//! 不同目录），按原名落盘会互相覆盖。UUID 保证不覆盖，而 clip 的**显示名**仍是用户
//! 看到的原始文件名（见 `import_audio_bytes` 的 `display_name`）。

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// 工程素材目录的后缀：`<工程名>_media`。
pub const PROJECT_MEDIA_SUFFIX: &str = "_media";

/// 未保存会话目录的一级名字（在本地数据根下）。
pub const UNSAVED_DIR: &str = "unsaved";

/// 未保存会话的存活期：超过这个时间的会话目录会在启动时被清掉。
///
/// 【为什么是 7 天而不是"启动即清"】用户可能"今天导入、明天才保存工程"，也可能
/// 几天不关应用。7 天足够覆盖正常的编辑节奏，同时保证崩溃 / 忘记保存的会话不会
/// 永久占着磁盘。
pub const UNSAVED_SESSION_TTL: Duration = Duration::from_secs(7 * 24 * 3600);

/// 工程素材目录：`<工程目录>/<工程文件名去扩展名>_media`。
///
/// 工程路径没有文件名（例如 "/" 或 "C:\"）时返回 `None` —— 那种路径本来也不是
/// 可以保存工程的地方。
pub fn project_media_dir(project_path: &Path) -> Option<PathBuf> {
    let stem = project_path.file_stem()?.to_str()?;
    if stem.is_empty() {
        return None;
    }
    let dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    Some(dir.join(format!("{stem}{PROJECT_MEDIA_SUFFIX}")))
}

/// 未保存会话目录：`<本地数据根>/unsaved/<会话 id>`。
pub fn unsaved_session_dir(local_root: &Path, session: &str) -> PathBuf {
    local_root.join(UNSAVED_DIR).join(session)
}

/// 导入落点：有工程 → 工程素材目录；没有 → 未保存会话目录。
///
/// 两者都不可用（既没有工程路径，也没有本地数据根）时返回 `None`，调用方退回
/// 系统临时目录 —— 那是**最后兜底**，正常路径不会再走到。
pub fn import_target_dir(
    project_path: Option<&Path>,
    local_root: Option<&Path>,
    session: &str,
) -> Option<PathBuf> {
    if let Some(project_path) = project_path {
        if let Some(dir) = project_media_dir(project_path) {
            return Some(dir);
        }
    }
    local_root.map(|root| unsaved_session_dir(root, session))
}

/// 把一条落在 `from` 之下的路径改写到 `to` 之下；不在其下时返回 `None`。
///
/// 【为什么要大小写不敏感地比前缀（仅 Windows）】工程里存的绝对路径可能来自用户
/// 手输或另一台机器的规范化结果，大小写与实际磁盘不一致；Windows 的文件系统本来
/// 也不区分它。
pub fn rebase_path(path: &str, from: &Path, to: &Path) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let rest = crate::app_paths::strip_prefix_ci(Path::new(trimmed), from)?;
    let rebased = to.join(rest);
    Some(rebased.to_string_lossy().into_owned())
}

/// 把一个目录里的文件搬进另一个目录（按相对路径保持结构），返回搬运的文件数。
///
/// 【为什么逐文件而不是整体 rename】目标是 `<工程>_media`：它可能**已经存在**
/// （用户此前保存过一次并导入过素材），整体 rename 会失败。重名时加 `-1` / `-2`
/// 后缀 —— 磁盘名本来就是 UUID，重名只可能来自用户手工放入的文件，那种情况下
/// 覆盖是不可接受的。
pub fn move_directory(from: &Path, to: &Path) -> Result<usize, String> {
    if !from.is_dir() {
        return Ok(0);
    }
    std::fs::create_dir_all(to).map_err(|e| format!("create {}: {e}", to.display()))?;
    let mut moved = 0usize;
    move_children(from, to, &mut moved)?;
    // 搬空的会话目录立刻删掉：它只是"还没保存"的临时家。
    let _ = std::fs::remove_dir(from);
    Ok(moved)
}

fn move_children(from: &Path, to: &Path, moved: &mut usize) -> Result<(), String> {
    let entries = std::fs::read_dir(from).map_err(|e| format!("read {}: {e}", from.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let source = entry.path();
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        let destination = to.join(entry.file_name());
        if file_type.is_dir() {
            std::fs::create_dir_all(&destination)
                .map_err(|e| format!("create {}: {e}", destination.display()))?;
            move_children(&source, &destination, moved)?;
            let _ = std::fs::remove_dir(&source);
            continue;
        }
        let destination = unique_destination(&destination);
        std::fs::rename(&source, &destination)
            .or_else(|_| {
                // 跨卷（工程在另一块盘上）时 rename 必然失败：拷过去再删源。
                std::fs::copy(&source, &destination)
                    .map(|_| ())
                    .and_then(|()| std::fs::remove_file(&source))
            })
            .map_err(|e| {
                format!(
                    "move {} -> {}: {e}",
                    source.display(),
                    destination.display()
                )
            })?;
        *moved += 1;
    }
    Ok(())
}

/// 目标已存在时找一个不重名的名字：`a.wav` → `a-1.wav` → `a-2.wav` …
fn unique_destination(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("import");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    for index in 1..1000 {
        let name = if extension.is_empty() {
            format!("{stem}-{index}")
        } else {
            format!("{stem}-{index}.{extension}")
        };
        let candidate = path.with_file_name(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    path.to_path_buf()
}

/// 清理过期的未保存会话目录，返回 `(删除的会话数, 释放的字节数)`。
///
/// 【为什么按目录的修改时间】会话目录里的文件是自己写的，目录的 mtime 会随每次
/// 导入更新，正好等于"最后一次使用时间"。
pub fn prune_unsaved_sessions(local_root: &Path, max_age: Duration) -> (u64, u64) {
    let root = local_root.join(UNSAVED_DIR);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return (0, 0);
    };
    let now = SystemTime::now();
    let mut removed = 0u64;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(modified) = last_used_at(&path) else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age <= max_age {
            continue;
        }
        bytes += directory_size(&path);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => removed += 1,
            Err(error) => log::warn!(
                "[media] could not remove expired unsaved session {}: {error}",
                path.display()
            ),
        }
    }
    (removed, bytes)
}

/// 会话目录的"最后一次使用时间"：目录里**最新文件**的修改时间。
///
/// 【为什么不是目录自身的 mtime】目录的 mtime 只反映"目录条目变了"，而一次导入
/// 完全可能覆盖已有文件（目录条目没变）。看文件才是"最后一次导入"的准确时刻。
/// 目录为空时退回目录自身的时间 —— 那是它唯一有的信息。
fn last_used_at(path: &Path) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let Ok(modified) = entry.metadata().and_then(|meta| meta.modified()) else {
                continue;
            };
            newest = Some(match newest {
                Some(current) if current > modified => current,
                _ => modified,
            });
        }
    }
    newest.or_else(|| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
    })
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += directory_size(&entry.path());
        } else {
            total += meta.len();
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("hfs-media-paths-{}-{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, body: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    /// 有工程 → 落工程素材目录，且目录名是"工程名 + _media"。
    #[test]
    fn import_target_dir_prefers_project_media_dir() {
        let scratch = Scratch::new("project");
        let project = scratch.path().join("My Song.hshp");
        let local_root = scratch.path().join("local");

        let dir = import_target_dir(Some(&project), Some(&local_root), "session-1")
            .expect("project media dir");
        assert_eq!(dir, scratch.path().join("My Song_media"));
        assert!(!dir.to_string_lossy().contains("unsaved"));
    }

    /// 没有工程 → 落本地数据根下的未保存会话目录。
    #[test]
    fn unsaved_session_dir_is_under_local_root() {
        let scratch = Scratch::new("unsaved");
        let local_root = scratch.path().join("local");
        let dir = import_target_dir(None, Some(&local_root), "20260101-000000-abcd")
            .expect("unsaved session dir");
        assert_eq!(
            dir,
            local_root.join(UNSAVED_DIR).join("20260101-000000-abcd")
        );
        assert_eq!(
            unsaved_session_dir(&local_root, "x"),
            local_root.join("unsaved").join("x")
        );
    }

    /// 两者都不可用 → None（调用方退回临时目录，而不是 panic）。
    #[test]
    fn import_target_dir_is_none_without_project_and_local_root() {
        assert_eq!(import_target_dir(None, None, "s"), None);
    }

    /// 保存工程时：文件搬进工程素材目录，路径按新目录改写。
    #[test]
    fn save_as_moves_unsaved_media_and_rewrites_paths() {
        let scratch = Scratch::new("adopt");
        let session = scratch.path().join("local/unsaved/s1");
        let media = session.join("import_abcd.wav");
        write(&media, b"RIFF");

        let project = scratch.path().join("song.hshp");
        let target = project_media_dir(&project).expect("media dir");
        let moved = move_directory(&session, &target).expect("move");
        assert_eq!(moved, 1);
        assert!(target.join("import_abcd.wav").is_file());
        assert!(!session.exists(), "搬空的会话目录应被删除");

        assert_eq!(
            rebase_path(&media.to_string_lossy(), &session, &target),
            Some(
                target
                    .join("import_abcd.wav")
                    .to_string_lossy()
                    .into_owned()
            )
        );
    }

    /// 工程已有素材目录时**合并**进去（不是整体 rename），重名加后缀。
    #[test]
    fn moving_into_an_existing_media_dir_merges_without_overwriting() {
        let scratch = Scratch::new("merge");
        let session = scratch.path().join("session");
        let target = scratch.path().join("song_media");
        write(&session.join("import_a.wav"), b"new");
        write(&target.join("import_a.wav"), b"old");

        let moved = move_directory(&session, &target).expect("move");
        assert_eq!(moved, 1);
        assert_eq!(std::fs::read(target.join("import_a.wav")).unwrap(), b"old");
        assert_eq!(
            std::fs::read(target.join("import_a-1.wav")).unwrap(),
            b"new"
        );
    }

    /// 不在 `from` 之下的路径一个字都不改。
    #[test]
    fn rebase_path_leaves_foreign_paths_alone() {
        let from = std::env::temp_dir().join("hfs-media-from");
        let to = std::env::temp_dir().join("hfs-media-to");
        let foreign = std::env::temp_dir().join("hfs-media-foreign").join("a.wav");
        assert_eq!(rebase_path(&foreign.to_string_lossy(), &from, &to), None);
        assert_eq!(rebase_path("   ", &from, &to), None);
    }

    /// 过期会话被清掉，最近使用过的留下。
    #[test]
    fn expired_unsaved_sessions_are_pruned() {
        let scratch = Scratch::new("prune");
        let local_root = scratch.path().join("local");
        let old = unsaved_session_dir(&local_root, "old");
        let fresh = unsaved_session_dir(&local_root, "fresh");
        write(&old.join("a.wav"), b"1234");
        write(&fresh.join("b.wav"), b"5678");

        // 把 old 里那个文件的时间往前拨 8 天（"最后一次使用"由它决定）。
        // 【为什么不去改目录的 mtime】Windows 上不能用 OpenOptions 打开目录，
        // 实现也不依赖目录自身的 mtime —— 见 `newest_modified`。
        let eight_days_ago = SystemTime::now() - Duration::from_secs(8 * 24 * 3600);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(old.join("a.wav"))
            .unwrap();
        file.set_modified(eight_days_ago).unwrap();
        drop(file);

        let (removed, bytes) = prune_unsaved_sessions(&local_root, UNSAVED_SESSION_TTL);
        assert_eq!(removed, 1, "应删掉一个过期会话");
        assert_eq!(bytes, 4);
        assert!(!old.exists());
        assert!(fresh.join("b.wav").is_file());
    }
}
