// 管理 UI 设置（主题、推理设备、默认拉伸算法等）的读写命令。
//
// 推理设备（ORT Execution Provider）相关的两个函数是本文件的重点：
// `get_ui_settings`（读路径）和 `save_ui_settings`（写路径）都会把
// `ort_ep` / `ort_device_id` 下发给三个 ONNX 模型模块。下发会销毁并重建
// 全部 ORT 会话，成本很高（CoreML 单个模型重编译就要 0.4~1.2s），因此这里
// 用 `apply_ort_ep_settings()` 做去重，只在取值真正变化时重建。

use crate::config::UiSettings;
use crate::state::AppState;
use std::sync::{Mutex, OnceLock};
use tauri::State;

/// Last `(ort_ep, ort_device_id)` pair pushed down to the ONNX modules.
///
/// `get_ui_settings()` is a *read* path and is invoked often (app start,
/// settings panel mounts).  Re-applying the EP settings unconditionally tore
/// down all three ORT sessions on every read, forcing a full model reload.
static APPLIED_EP: OnceLock<Mutex<Option<(String, Option<i32>)>>> = OnceLock::new();

/// Push the EP settings down to the three ONNX model modules, but only when
/// they differ from what is already applied.
///
/// Returns `true` when the sessions were invalidated (i.e. the caller may need
/// to invalidate render caches).  Dropping a session forces a rebuild; on
/// CoreML that means recompiling the model, so this must not happen on every
/// settings read.
fn apply_ort_ep_settings(ep: &str, device_id: Option<i32>) -> bool {
    let slot = APPLIED_EP.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let key = (ep.to_string(), device_id);
    if guard.as_ref() == Some(&key) {
        return false;
    }
    *guard = Some(key);
    drop(guard);

    crate::nsf_hifigan_onnx::update_ort_ep(ep, device_id);
    crate::hnsep_onnx::update_ort_ep(ep, device_id);
    crate::fcpe_onnx::update_ort_ep(ep, device_id);
    true
}

/// 需要做**深度合并**（逐个嵌套子键覆盖）的顶层键。
///
/// 这些键的值是对象，而前端可能只发送其中一个子字段（如只改声道导入策略的
/// 容差）。若按顶层键做浅替换，未发送的兄弟子键会整块丢失。
///
/// `dock` 尤其关键：行为选项与整份布局同处一个对象，只写选项的部分保存若走
/// 浅替换，会把 `layout` 整个抹掉 —— 用户排了十分钟的界面会在改一个开关后归零。
const DEEP_MERGE_KEYS: &[&str] = &[
    "timelineSnap",
    "renderCache",
    "channelImportPolicy",
    "notebook",
    "dock",
];

/// 以现有设置为基底合并前端发来的部分补丁（纯函数，便于测试）。
///
/// 顶层做逐键覆盖；[`DEEP_MERGE_KEYS`] 中的键再做一层子键合并。
pub(crate) fn merge_ui_settings_patch(
    mut base: serde_json::Value,
    patch: &serde_json::Value,
) -> serde_json::Value {
    if let (serde_json::Value::Object(base_obj), serde_json::Value::Object(patch_obj)) =
        (&mut base, patch)
    {
        for (key, value) in patch_obj {
            if DEEP_MERGE_KEYS.contains(&key.as_str()) {
                match base_obj.get_mut(key.as_str()) {
                    Some(serde_json::Value::Object(base_nested)) => {
                        if let serde_json::Value::Object(patch_nested) = value {
                            for (nested_key, nested_value) in patch_nested {
                                base_nested.insert(nested_key.clone(), nested_value.clone());
                            }
                        }
                    }
                    _ => {
                        base_obj.insert(key.clone(), value.clone());
                    }
                }
            } else {
                base_obj.insert(key.clone(), value.clone());
            }
        }
    }
    base
}

pub(super) fn get_ui_settings(state: State<'_, AppState>) -> UiSettings {
    let mut settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    settings.normalize_split_transition();
    settings.normalize_time_display();
    // 渲染缓存设置同样做规范化回读：手改坏的配置值（越界容量 / 非法枚举）
    // 不应被前端原样展示或写回。
    settings.render_cache = settings.render_cache.normalized();
    settings.channel_import_policy = settings.channel_import_policy.normalized();
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    // Apply EP settings on load — all three ONNX models.  No-op when the
    // values are unchanged, so repeated settings reads do not rebuild sessions.
    apply_ort_ep_settings(&settings.ort_ep, settings.ort_device_id);
    // Sync background render setting
    crate::commands::playback::AUTO_BG_RENDER_ENABLED.store(
        settings.auto_background_render,
        std::sync::atomic::Ordering::Relaxed,
    );
    // Sync "loop for new clips" default (used by importers / legacy project migration)
    crate::config::set_loop_new_clips_default(settings.loop_new_clips);
    crate::config::set_sync_edits_across_takes(settings.sync_edits_across_takes);
    // Sync the import channel policy (used by importers / legacy project migration)
    crate::config::set_channel_import_policy(&settings.channel_import_policy);
    // 同步渲染缓存配置（总开关 / 容量 / 超龄…）：应用设置是幂等的，写入或
    // 缓存目录变化时才会触发目录准备与回收。
    crate::render_cache::apply_settings(&settings.render_cache);
    // 刷新进程内缓存，供拖拽热路径（ripple/split 选项）无盘读取
    state.store_ui_settings_cache(&settings);
    settings
}

pub(super) fn save_ui_settings(
    state: State<'_, AppState>,
    settings_value: serde_json::Value,
) -> serde_json::Value {
    // 前端可能只发送变更字段（部分保存，如单个 MIDI 导入选项）。
    // UiSettings 的所有字段都带 serde default，直接用部分对象反序列化
    // 会把未发送字段重置为默认值，覆盖磁盘上的其他设置 —— 因此以
    // 现有设置为基础做 JSON 级合并，再反序列化应用。
    let prev_settings = if let Some(dir) = state.config_dir.get() {
        crate::config::load_ui_settings(dir)
    } else {
        UiSettings::default()
    };
    let merged_value = match serde_json::to_value(&prev_settings) {
        Ok(base) => merge_ui_settings_patch(base, &settings_value),
        Err(_) => settings_value,
    };
    let mut settings: UiSettings =
        serde_json::from_value(merged_value).unwrap_or_else(|_| prev_settings.clone());

    settings.normalize_split_transition();
    settings.normalize_time_display();
    settings.render_cache = settings.render_cache.normalized();
    settings.channel_import_policy = settings.channel_import_policy.normalized();
    let prev_ep = prev_settings.ort_ep.clone();

    if let Some(dir) = state.config_dir.get() {
        crate::config::save_ui_settings(dir, &settings);
    }
    // 刷新进程内缓存，供拖拽热路径（ripple/split 选项）无盘读取
    state.store_ui_settings_cache(&settings);
    crate::time_stretch::update_global_stretch_defaults(
        settings.default_stretch_algorithm,
        settings.default_hifigan_mel_stretch,
    );
    crate::commands::playback::AUTO_BG_RENDER_ENABLED.store(
        settings.auto_background_render,
        std::sync::atomic::Ordering::Relaxed,
    );
    crate::config::set_loop_new_clips_default(settings.loop_new_clips);
    crate::config::set_sync_edits_across_takes(settings.sync_edits_across_takes);
    crate::config::set_channel_import_policy(&settings.channel_import_policy);
    // 渲染缓存配置变化（开关 / 上限 / 超龄 / 目录）→ 立即生效并触发回收。
    crate::render_cache::apply_settings(&settings.render_cache);

    // Both fields matter: changing only the DirectML device ID must also
    // rebuild the sessions, otherwise the new device is silently ignored.
    let ep_changed =
        prev_ep != settings.ort_ep || prev_settings.ort_device_id != settings.ort_device_id;

    // Changing the global stretch defaults only affects the current project when
    // the project inherits the corresponding setting. Compute the effective value
    // so unrelated global edits (e.g. theme-only saves) do not invalidate renders.
    let effective_stretch_changed = {
        let project = state.project.lock().unwrap_or_else(|e| e.into_inner());
        let algorithm_before = project
            .stretch_algorithm_override
            .unwrap_or(prev_settings.default_stretch_algorithm);
        let algorithm_after = project
            .stretch_algorithm_override
            .unwrap_or(settings.default_stretch_algorithm);
        let mel_before = project
            .hifigan_mel_stretch_override
            .unwrap_or(prev_settings.default_hifigan_mel_stretch);
        let mel_after = project
            .hifigan_mel_stretch_override
            .unwrap_or(settings.default_hifigan_mel_stretch);
        algorithm_before != algorithm_after || mel_before != mel_after
    };

    if ep_changed {
        apply_ort_ep_settings(&settings.ort_ep, settings.ort_device_id);
    }

    if ep_changed || effective_stretch_changed {
        let timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for clip in &timeline.clips {
            crate::synth_clip_cache::invalidate_clip_all_caches(&clip.id);
        }
        state.audio_engine.update_timeline(timeline);
        if let Some(handle) = state.app_handle.get() {
            crate::commands::playback::request_background_render(handle);
        }
    }

    // Swing 参与节拍器响点展开（弱网格线奇数格偏移）：变化时重建响点表，
    // 使节拍器与时间标尺的 Swing 语义保持一致（播放中也实时生效）。
    if prev_settings.timeline_snap.swing_enabled != settings.timeline_snap.swing_enabled
        || prev_settings.timeline_snap.swing_percent != settings.timeline_snap.swing_percent
    {
        crate::commands::playback::refresh_metronome_schedule(&state);
    }

    serde_json::json!({ "ok": true })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deep_merge_keeps_sibling_subkeys_of_channel_import_policy() {
        // 前端只改容差：其余子键必须原样保留（白名单漏了就会整块丢失）。
        let base = json!({
            "channelImportPolicy": {
                "mode": "smart",
                "windowSec": 0.25,
                "windowCount": 12,
                "tolerance": 1e-6,
                "monoTargetMode": 2,
            }
        });
        let patch = json!({ "channelImportPolicy": { "tolerance": 1e-4 } });
        let merged = merge_ui_settings_patch(base, &patch);
        let policy = &merged["channelImportPolicy"];
        assert_eq!(policy["tolerance"], json!(1e-4));
        assert_eq!(policy["mode"], json!("smart"), "兄弟子键不得丢失");
        assert_eq!(policy["windowCount"], json!(12));
        assert_eq!(policy["monoTargetMode"], json!(2));
    }

    #[test]
    fn deep_merge_covers_every_nested_settings_object() {
        // 逐个白名单键验证：只发一个子字段时，另一个子字段必须留存。
        for key in DEEP_MERGE_KEYS {
            let key: &str = key;
            let base = json!({ key: { "kept": 1, "changed": 1 } });
            let patch = json!({ key: { "changed": 2 } });
            let merged = merge_ui_settings_patch(base, &patch);
            assert_eq!(merged[key]["kept"], json!(1), "{key} 的兄弟子键丢失");
            assert_eq!(merged[key]["changed"], json!(2), "{key} 的补丁未生效");
        }
    }

    #[test]
    fn shallow_keys_replace_wholesale() {
        // 非白名单键按整体替换：对象不做子键合并。
        let base = json!({ "paramAxisUnits": { "volume": "db" } });
        let patch = json!({ "paramAxisUnits": { "dyn": "ratio" } });
        let merged = merge_ui_settings_patch(base, &patch);
        assert_eq!(merged["paramAxisUnits"], json!({ "dyn": "ratio" }));
    }

    #[test]
    fn merge_preserves_untouched_top_level_keys() {
        let base = json!({ "autoCrossfade": true, "loopNewClips": true });
        let patch = json!({ "autoCrossfade": false });
        let merged = merge_ui_settings_patch(base, &patch);
        assert_eq!(merged["autoCrossfade"], json!(false));
        assert_eq!(merged["loopNewClips"], json!(true));
    }

    #[test]
    fn merge_tolerates_non_object_base_or_patch() {
        // 基底不是对象时原样返回基底（调用方随后会走反序列化兜底）。
        let merged = merge_ui_settings_patch(json!(null), &json!({ "a": 1 }));
        assert_eq!(merged, json!(null));
        // 补丁不是对象时同样不改动基底。
        let merged = merge_ui_settings_patch(json!({ "a": 1 }), &json!(42));
        assert_eq!(merged, json!({ "a": 1 }));
    }

    #[test]
    fn merge_fills_missing_nested_object_from_patch() {
        // 基底缺该键时退化为整体插入。
        let merged = merge_ui_settings_patch(
            json!({ "autoCrossfade": true }),
            &json!({ "channelImportPolicy": { "mode": "off" } }),
        );
        assert_eq!(merged["channelImportPolicy"]["mode"], json!("off"));
    }
}
