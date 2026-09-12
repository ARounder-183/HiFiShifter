//! 操作记录（撤销历史）的 `-UNDO` 伴生文件读写。
//!
//! - **保存**：`<带扩展名的工程文件名称>-UNDO`（后缀固定大写，如
//!   `1.hshp-UNDO`）；仅在「与工程一起保存操作历史」设置开启时写入。
//! - **读取**：无论设置是否开启都尝试读取；后缀**大小写不敏感**
//!   （`-UNDO` / `-undo` / `-Undo` … 都能识别）。
//! - **失败一律静默**：读不到 / 解析失败就当作没有历史（回到打开工程后的
//!   默认状态），不影响工程本身的打开与保存。
//!
//! 时间戳统一为 **Unix 毫秒（UTC）**：文件里与后端内存里都是 UTC，
//! 只有前端展示时按用户本地时区格式化。

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::state::{AppState, HistoryRecord, TimelineState};

/// 伴生文件后缀（写入时固定大写；读取时大小写不敏感）。
pub const UNDO_FILE_SUFFIX: &str = "-UNDO";
/// 文件格式版本。
const UNDO_FILE_VERSION: u32 = 1;
/// 单文件体积上限：每条记录都含完整时间线快照，超出后从**最旧**的记录开始丢弃。
const MAX_UNDO_FILE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UndoFileRecord {
    /// 语言无关的操作 key；`None` = 初始状态行。
    label: Option<String>,
    /// Unix 毫秒（UTC）—— 展示时由前端按本地时区格式化。
    at_ms: u64,
    /// 该状态快照；`None` 只会出现在「当前位置」的记录上（保存时已用实时
    /// 时间线补齐，故落盘的记录通常都带快照）。
    #[serde(default)]
    state: Option<TimelineState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UndoFile {
    #[serde(default)]
    version: u32,
    /// 写入时刻（Unix 毫秒，UTC）。
    #[serde(default)]
    saved_at_ms: u64,
    /// 历史起点时刻（初始状态行的时间）。
    #[serde(default)]
    started_at_ms: u64,
    #[serde(default)]
    position: usize,
    #[serde(default)]
    records: Vec<UndoFileRecord>,
}

/// `<带扩展名的工程文件名称>-UNDO`
pub fn undo_file_path_for(project_path: &Path) -> Option<PathBuf> {
    let file_name = project_path.file_name()?.to_string_lossy().to_string();
    if file_name.is_empty() {
        return None;
    }
    Some(project_path.with_file_name(format!("{file_name}{UNDO_FILE_SUFFIX}")))
}

/// 大小写不敏感地查找伴生文件：先按标准大写名，再扫描同目录的其它大小写变体。
pub fn find_undo_file(project_path: &Path) -> Option<PathBuf> {
    let exact = undo_file_path_for(project_path)?;
    if exact.is_file() {
        return Some(exact);
    }
    let dir = project_path.parent()?;
    let base = project_path.file_name()?.to_string_lossy().to_string();
    let prefix = format!("{base}-");
    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                return false;
            };
            name.len() == prefix.len() + UNDO_FILE_SUFFIX.len() - 1
                && name.starts_with(&prefix)
                && name[prefix.len()..].eq_ignore_ascii_case("undo")
                && path.is_file()
        })
        .collect();
    // 变体可能不止一个：排序后取最后一个，避免依赖平台枚举顺序。
    candidates.sort();
    candidates.pop()
}

/// 保存操作记录到伴生文件（设置未开启 / 无历史 / 写入失败都静默返回 false）。
pub fn save_undo_history(state: &AppState, project_path: &Path) -> bool {
    let Some(bytes) = serialize_undo_history(state) else {
        return false;
    };
    let Some(target) = undo_file_path_for(project_path) else {
        return false;
    };
    match fs::write(&target, bytes) {
        Ok(()) => true,
        Err(error) => {
            log::warn!("[undo-history] write failed ({}): {error}", target.display());
            false
        }
    }
}

/// 序列化操作记录（不落盘）：工程保存写伴生文件、ZIP 归档写入压缩包内条目
/// 共用这一段。设置未开启 / 无历史 / 序列化失败都返回 `None`。
pub fn serialize_undo_history(state: &AppState) -> Option<Vec<u8>> {
    // 是否随工程保存 UNDO 数据由**工程级开关**决定（全局设置只决定新工程的
    // 初始值）；打开工程时总是尝试读取，与本开关无关。
    if !state
        .project
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .save_undo_history
    {
        return None;
    }

    // 取快照：历史记录 + 实时时间线（补齐「当前位置」记录的快照）。
    let (position, started_at_ms, records) = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let h = state
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut records: Vec<UndoFileRecord> = Vec::with_capacity(h.records.len());
        for (index, record) in h.records.iter().enumerate() {
            let snapshot = if index == h.position {
                Some(tl.clone())
            } else {
                record.state.clone()
            };
            records.push(UndoFileRecord {
                label: record.label.clone(),
                at_ms: record.at_ms,
                state: snapshot,
            });
        }
        (h.position, h.started_at_ms, records)
    };
    if records.is_empty() {
        return None;
    }

    // 从最新往回累加：超出体积上限即停止，更旧的记录被丢弃。
    let total_records = records.len();
    let mut kept: Vec<serde_json::Value> = Vec::with_capacity(records.len());
    let mut total = 0usize;
    let mut oldest_kept_at_ms = started_at_ms;
    for record in records.into_iter().rev() {
        let at_ms = record.at_ms;
        let value = match serde_json::to_value(&record) {
            Ok(value) => value,
            Err(error) => {
                log::warn!("[undo-history] record serialize failed: {error}");
                return None;
            }
        };
        let size = serde_json::to_string(&value).map(|s| s.len()).unwrap_or(0);
        if !kept.is_empty() && total.saturating_add(size) > MAX_UNDO_FILE_BYTES {
            break;
        }
        total = total.saturating_add(size);
        oldest_kept_at_ms = at_ms;
        kept.push(value);
    }
    kept.reverse();
    if kept.is_empty() {
        return None;
    }
    let dropped = total_records.saturating_sub(kept.len());
    let position = position.saturating_sub(dropped).min(kept.len() - 1);
    // 初始状态行被丢弃时，用最旧保留记录的时刻作为历史起点。
    let started_at_ms = if dropped > 0 {
        oldest_kept_at_ms
    } else {
        started_at_ms
    };

    let file = serde_json::json!({
        "version": UNDO_FILE_VERSION,
        "savedAtMs": crate::state::now_unix_ms(),
        "startedAtMs": started_at_ms,
        "position": position,
        "records": kept,
    });
    match serde_json::to_vec(&file) {
        Ok(bytes) => Some(bytes),
        Err(error) => {
            log::warn!("[undo-history] serialize failed: {error}");
            None
        }
    }
}

/// 读取伴生文件恢复操作记录（`open_project` 无条件尝试）。
///
/// 返回是否成功恢复；失败时历史保持调用方的状态（打开工程时已清空）。
pub fn load_undo_history(state: &AppState, project_path: &Path) -> bool {
    let Some(path) = find_undo_file(project_path) else {
        return false;
    };
    let Ok(bytes) = fs::read(&path) else {
        return false;
    };
    // 先按 JSON（本程序写出）解析，再兼容 MessagePack。
    let parsed = serde_json::from_slice::<UndoFile>(&bytes)
        .or_else(|_| rmp_serde::from_slice::<UndoFile>(&bytes));
    let Ok(file) = parsed else {
        log::warn!("[undo-history] unreadable file: {}", path.display());
        return false;
    };
    if file.version != UNDO_FILE_VERSION {
        return false;
    }
    // 每条快照都要走与「打开工程」相同的反序列化整理：磁盘形态里 `Clip` 的
    // 媒体字段只是 active take 的内存投影（`skip_serializing` 不落盘），
    // 少了 finalize 这一步，恢复出来的 Clip 就只有时间位置、没有音频 ——
    // 这正是「读取 -UNDO 后撤销 → Clip 音频内容被清空」的根因。
    let records: Vec<HistoryRecord> = file
        .records
        .into_iter()
        .map(|record| {
            let state = record.state.map(|state| {
                let (finalized, _missing_files) = crate::project::finalize_timeline_for_session(
                    state,
                    project_path,
                    crate::project::CURRENT_PROJECT_FILE_VERSION,
                );
                finalized
            });
            HistoryRecord {
                label: record.label,
                at_ms: record.at_ms,
                state,
            }
        })
        .collect();
    if records.is_empty() {
        return false;
    }
    let position = file.position.min(records.len() - 1);
    {
        let mut h = state
            .timeline_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        h.records = records;
        h.position = position;
        h.started_at_ms = if file.started_at_ms > 0 {
            file.started_at_ms
        } else {
            h.records
                .first()
                .map(|record| record.at_ms)
                .unwrap_or_else(crate::state::now_unix_ms)
        };
    }
    state.emit_history_state();
    true
}

#[cfg(test)]
mod tests {
    use super::{find_undo_file, undo_file_path_for};
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hifishifter_undo_file_test_{}_{}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn project_switch_controls_undo_persistence_not_the_global_default() {
        use crate::state::{AppState, HistoryOp};

        let state = AppState::default();
        {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            state.checkpoint_timeline(&tl, HistoryOp::AddClip);
        }

        // 工程级开关默认关闭（= 全局「新建工程默认值」的新默认）：不写出。
        assert!(
            super::serialize_undo_history(&state).is_none(),
            "默认应不写出 UNDO 数据"
        );

        // 打开工程级开关 → 有历史就有内容可写。
        state.set_project_save_undo_history(true);
        assert!(
            super::serialize_undo_history(&state).is_some(),
            "工程级开关开启后应能序列化 UNDO 数据"
        );

        // 关闭工程级开关 → 不再写出 UNDO 数据。
        state.set_project_save_undo_history(false);
        assert!(
            super::serialize_undo_history(&state).is_none(),
            "工程级开关关闭后不应写出 UNDO 数据"
        );

        // 全局默认只决定新工程的初值，不影响当前工程。
        let mut settings = crate::config::UiSettings::default();
        settings.save_undo_history_by_default = true;
        state.store_ui_settings_cache(&settings);
        assert!(
            super::serialize_undo_history(&state).is_none(),
            "全局默认不应覆盖当前工程的开关"
        );

        // 重新打开工程级开关 → 恢复写出。
        state.set_project_save_undo_history(true);
        assert!(super::serialize_undo_history(&state).is_some());
    }

    #[test]
    fn undo_file_name_keeps_extension_and_uppercases_suffix() {
        let project = PathBuf::from("/tmp/projects/1.hshp");
        assert_eq!(
            undo_file_path_for(&project),
            Some(PathBuf::from("/tmp/projects/1.hshp-UNDO"))
        );
    }

    #[test]
    fn lookup_is_case_insensitive_for_the_undo_suffix() {
        for variant in ["-UNDO", "-undo", "-Undo", "-uNdO"] {
            let dir = temp_dir(variant);
            let project = dir.join("1.hshp");
            std::fs::write(&project, b"{}").expect("write project");
            let companion = dir.join(format!("1.hshp{variant}"));
            std::fs::write(&companion, b"{}").expect("write companion");
            // 大小写不敏感的文件系统（Windows）上，命中的路径名可能与写入时
            // 的大小写不同：只断言“找得到且确实存在”。
            let found = find_undo_file(&project).unwrap_or_else(|| {
                panic!("variant {variant} should be found in {}", dir.display())
            });
            assert!(found.is_file(), "variant {variant}: {found:?} is not a file");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn missing_companion_file_is_not_found() {
        let dir = temp_dir("missing");
        let project = dir.join("2.hshp");
        std::fs::write(&project, b"{}").expect("write project");
        // 同目录下另一个工程的伴生文件不应被误匹配。
        std::fs::write(dir.join("other.hshp-UNDO"), b"{}").expect("write other");
        assert_eq!(find_undo_file(&project), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exact_uppercase_name_wins() {
        let dir = temp_dir("prefer");
        let project = dir.join("3.hshp");
        std::fs::write(&project, b"{}").expect("write project");
        std::fs::write(dir.join("3.hshp-UNDO"), b"{}").expect("write upper");
        std::fs::write(dir.join("3.hshp-undo"), b"{}").expect("write lower");
        assert_eq!(
            find_undo_file(&project),
            Some(dir.join("3.hshp-UNDO")),
            "标准大写名优先"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
