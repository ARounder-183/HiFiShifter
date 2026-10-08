//! 原GUI命令的插件准入与类型适配；使用共享原参数/历史/波形实现，不接收宿主几何变更。
use super::session::EditorSession;
use base64::Engine as _;
use hifishifter_kernel::editor::{history, params, waveform, ParamHost};
use hifishifter_kernel::state::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

pub(super) fn mutates_audio(command: &str) -> bool {
    matches!(
        command,
        "set_param_frames"
            | "restore_param_frames"
            | "set_static_param"
            | "convert_mix_param"
            // 拉伸映射同样改写参数曲线，必须让在途渲染作废（否则拉伸后听到的是
            // 旧曲线渲染出来的音频）。
            | "stretch_track_linked_params"
            | "set_track_state"
            | "move_track"
            | "undo_timeline"
            | "redo_timeline"
            | "undo_parameter_edit"
            | "redo_parameter_edit"
            | "set_history_position"
    )
}

// 回归诊断测试仅在cfg(test)编译；产品分组历史路径由已加载分支实际执行。
#[cfg(test)]
#[path = "commands_tests.rs"]
mod undo_group_diagnostic;
fn value<T: serde::Serialize>(input: T) -> Result<Value, String> {
    serde_json::to_value(input).map_err(|e| e.to_string())
}
fn args<T: serde::de::DeserializeOwned>(input: Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|e| format!("invalid editor arguments: {e}"))
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Frames {
    track_id: String,
    param: String,
    start_frame: u32,
    #[serde(default)]
    frame_count: u32,
    stride: Option<u32>,
    binary: Option<bool>,
    with_sentinel: Option<bool>,
    #[serde(default)]
    values: Vec<f32>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Static {
    track_id: String,
    param: String,
    value: Option<f64>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Mix {
    track_id: String,
    from: String,
    ranges: Vec<hifishifter_kernel::editor::ConvertRange>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StretchLinked {
    track_id: String,
    mappings: Vec<hifishifter_kernel::state::StretchLinkedRangeSec>,
    checkpoint: Option<bool>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Segment {
    track_id: String,
    start_sec: f64,
    duration_sec: f64,
    columns: usize,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrackPatch {
    track_id: String,
    volume: Option<f32>,
    muted: Option<bool>,
    solo: Option<bool>,
    compose_enabled: Option<bool>,
    pitch_analysis_algo: Option<PitchAnalysisAlgo>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PrivateTrackMove {
    track_id: String,
    target_index: usize,
    parent_track_id: Option<String>,
}

/// 构造原payload，包括真实本地dirty和历史深度，不伪造工程文件路径。
pub(super) fn payload(session: &EditorSession, lite: bool) -> Result<Value, String> {
    let (position, _) = session.transport();
    let timeline = session.timeline.lock().unwrap();
    let mut payload = if lite {
        timeline.to_payload_lite()
    } else {
        timeline.to_payload()
    };
    payload.playhead_sec = position.max(0.);
    let (undo, redo) = history_depths_of(&session.history.lock().unwrap());
    payload.undo_depth = Some(undo);
    payload.redo_depth = Some(redo);
    let project = session.project.lock().unwrap().clone();
    payload.project = Some(hifishifter_kernel::models::ProjectMetaPayload {
        name: "REAPER / HiFiShifter".into(),
        path: None,
        dirty: session.generation.load(Ordering::Acquire)
            != session.applied.load(Ordering::Acquire),
        recent: vec![],
        notes_markdown: project.notes_markdown,
        base_scale: project.base_scale,
        use_custom_scale: project.use_custom_scale,
        custom_scale: project.custom_scale,
        beats_per_bar: project.beats_per_bar,
        time_signature_denominator: project.time_signature_denominator,
        grid_size: project.grid_size,
        stretch_algorithm_override: project.stretch_algorithm_override,
        hifigan_mel_stretch_override: project.hifigan_mel_stretch_override,
        save_undo_history: project.save_undo_history,
    });
    drop(timeline);
    let mut result = value(payload)?;
    session.decorate_host_fades(&mut result);
    // 宿主音频读数（语言无关分类）：GUI 据此把"看起来能用、其实没内容"的空白 Clip
    // 说明白（见 `render::extension::HostAudioState`）。
    session.decorate_host_audio(&mut result);
    Ok(result)
}
fn history_state(session: &EditorSession) -> Value {
    let history = session.history.lock().unwrap();
    let (undo, redo) = history_depths_of(&history);
    let records: Vec<_> = history
        .records
        .iter()
        .map(|r| json!({"label":r.label,"atMs":r.at_ms}))
        .collect();
    json!({"ok":true,"position":history.position,"undoDepth":undo,"redoDepth":redo,
        "records":if records.is_empty() {vec![json!({"label":null,"atMs":history.started_at_ms})]} else {records}})
}
fn track_exists(session: &EditorSession, track: &str) -> Result<(), String> {
    if session
        .timeline
        .lock()
        .unwrap()
        .tracks
        .iter()
        .any(|t| t.id == track)
    {
        Ok(())
    } else {
        Err("unknown host track".into())
    }
}
fn budget(start: u32, count: usize) -> Result<(), String> {
    if count > 1_000_000
        || (start as usize)
            .checked_add(count)
            .is_none_or(|n| n > 1_000_000)
    {
        return Err("parameter frame budget exceeded".into());
    }
    Ok(())
}
fn after_write(session: &EditorSession, result: Value) -> Result<Value, String> {
    if result["ok"] == false {
        return Err(result["error"]
            .as_str()
            .unwrap_or("parameter operation rejected")
            .into());
    }
    if let Some(error) = session.error.lock().unwrap().clone() {
        return Err(error);
    }
    session.emit("history_state", history_state(session));
    session.notify_timeline();
    Ok(result)
}
fn peaks(
    session: &EditorSession,
    path: &str,
) -> Result<std::sync::Arc<hifishifter_kernel::hfspeaks_v2::HfsPeakFile>, String> {
    session.check_source(path)?;
    if let Some(found) = session.peaks.lock().unwrap().get(path).cloned() {
        return Ok(found);
    }
    let result = std::sync::Arc::new(hifishifter_kernel::hfspeaks_v2::compute_mipmap_peaks(
        std::path::Path::new(path),
    )?);
    let mut cache = session.peaks.lock().unwrap();
    let total: u64 = cache.values().map(|p| p.estimated_byte_size()).sum();
    if total.saturating_add(result.estimated_byte_size()) > 64 * 1024 * 1024 {
        cache.clear();
    }
    cache.insert(path.into(), result.clone());
    Ok(result)
}

/// 仅actor线程调用。宿主PCM不可用/未知命令/平台能力缺失均明确报错。
pub(super) fn dispatch(
    session: &EditorSession,
    command: &str,
    input: Value,
) -> Result<Value, String> {
    match command {
        "list_directory"
        | "stat_paths"
        | "get_audio_file_info"
        | "search_files_recursive"
        | "create_directory"
        | "rename_path"
        | "delete_paths"
        | "reveal_paths_in_file_manager"
        | "open_path_with_default_app" => return session.browser_command(command, &input),
        // 设置由进程级的 `settings_store` 持有：它属于用户，不属于某一个 ARA 文档，
        // 并且与独立 App 共用同一份配置文件。此前这里是 `session.settings`（每个
        // 文档一份、初始化为出厂默认），保存只改内存、不落盘。
        "get_ui_settings" => return value(crate::settings_store::settings()),
        "save_ui_settings" => {
            return value(crate::settings_store::save_settings_patch(
                &input["settings"],
            )?)
        }
        // 前端偏好（原 localStorage 的 `hifishifter.*` 键）。批量接口：前端启动时
        // 一次取走全部、写入时按去抖批量提交，避免一次设置变更产生多个往返。
        "ui_kv_dump" => return value(crate::settings_store::frontend_prefs()),
        "ui_kv_put" => {
            let patch: std::collections::BTreeMap<String, String> = args(input["patch"].clone())?;
            if patch.len() > 512 {
                return Err("too many preference keys in one write".into());
            }
            return value(crate::settings_store::save_frontend_prefs(patch));
        }
        "ui_kv_delete" => {
            let keys: Vec<String> = args(input["keys"].clone())?;
            if keys.len() > 512 {
                return Err("too many preference keys in one delete".into());
            }
            return value(crate::settings_store::delete_frontend_prefs(&keys));
        }
        "get_about_info" => {
            // 与独立 App **逐字段同形**：前端 `AboutDialog` 只认这一组键。
            // 原先这里返回 `{name, version, host}`，于是插件里版本号显示错、
            // commit 恒空、仓库链接恒走前端兜底值。
            return Ok(hifishifter_kernel::build_info::about_payload(
                "ARA plugin",
                crate::VERSION,
            ));
        }
        "plugin_get_apply_state" => return Ok(session.state()),
        // Help 菜单的「打开日志目录」。
        //
        // 【为什么同时回报路径】打开资源管理器是尽力而为（[`reveal_directory`]）；
        // 无论成败都要把目录与文件路径交出去，前端才能在失败时让用户**复制**它，
        // 而不是逼他手抄。
        "open_log_folder" => {
            let dir = crate::diagnostics::log_directory();
            let file = crate::diagnostics::log_file();
            let opened = super::browser_files::reveal_directory(&dir);
            return Ok(json!({
                "ok": true,
                "path": dir.to_string_lossy(),
                "file": file.to_string_lossy(),
                "opened": opened.is_ok(),
            }));
        }
        // 诊断导出与基准测试：导出走 UI 线程的精简实现（见 `webview.rs`，命令名与
        // 独立 App **相同**，前端不必按模式分支），基准测试在插件里**保持拒绝** ——
        // `src-tauri` 的完整实现会初始化 ORT 会话，在 GPU/驱动异常的环境下可能硬崩，
        // 而在 DAW 进程里崩会带走用户的整个会话。
        "run_vocoder_benchmark" => {
            return Err("the vocoder benchmark is not available in ARA plugin mode".into())
        }
        "plugin_history_barrier" => {
            session.suppress_history.store(false, Ordering::Release);
            return Ok(json!({"ok":true}));
        }
        "plugin_editor_barrier" => return Ok(json!({"ok":true})),
        "get_timeline_inventory" => {
            // 媒体写入回执只取结构，不为显示新GUID先物化整份宿主PCM或跑分析。
            if let Some(document) = session.document.upgrade() {
                session.refresh_ui_geometry(&document);
            }
            return payload(session, false);
        }
        "select_track" | "select_clip" => {
            // 选择已确认的GUI对象不必等PCM；新粘贴片段先选中，再由后台补音频/波形。
            if let Some(document) = session.document.upgrade() {
                session.refresh_ui_geometry(&document);
            }
            if command == "select_track" {
                let id = input["trackId"].as_str().ok_or("trackId missing")?;
                track_exists(session, id)?;
                session.timeline.lock().unwrap().select_track(id);
            } else {
                let id = input["clipId"].as_str().map(str::to_owned);
                let mut timeline = session.timeline.lock().unwrap();
                if let Some(id) = &id {
                    if !timeline.clips.iter().any(|clip| &clip.id == id) {
                        return Err("unknown host clip".into());
                    }
                }
                timeline.select_clip(id);
            }
            // 只有已授权clip才切换源曲线，未就绪的显示占位绝不读本机原文件。
            let authorized = {
                let timeline = session.timeline.lock().unwrap();
                timeline.selected_clip_id.as_ref().is_none_or(|id| {
                    timeline
                        .clips
                        .iter()
                        .any(|clip| &clip.id == id && clip.source_path.is_some())
                })
            };
            if authorized {
                session.select_source_projection()?;
            }
            session.notify_timeline();
            return payload(session, false);
        }
        "get_playback_state" => {
            // 宿主播放态不依赖曲线载入；Unsupported/Conflict也必须还能观察播放并暂停。
            return Ok(session.playback_state());
        }
        "plugin_refresh" => {
            session.ensure_loaded(input["force"].as_bool().unwrap_or(false))?;
            return payload(session, false);
        }
        // 语言原先只是原样回显（"返回 ok 但什么也没做"）。现在记进与独立 App
        // 共用的前端偏好，用户换回 App 时语言也跟着走。
        "set_ui_locale" => {
            let locale = input["locale"].as_str().ok_or("locale missing")?;
            if locale.len() > 32 {
                return Err("locale too long".into());
            }
            return Ok(json!({"ok":true,"locale":crate::settings_store::set_locale(locale)}));
        }
        "consume_startup_project_path" => return Ok(Value::Null),
        "get_processor_params" => {
            return value(
                hifishifter_kernel::editor::capabilities::get_processor_params(
                    input["algo"].as_str().ok_or("algo missing")?.into(),
                ),
            )
        }
        "transliterate" => {
            let texts: Vec<String> = args(input["texts"].clone())?;
            if texts.len() > 5000 || texts.iter().map(String::len).sum::<usize>() > 1024 * 1024 {
                return Err("text index budget exceeded".into());
            }
            let options: hifishifter_kernel::search::SearchOptions = if input["options"].is_null() {
                Default::default()
            } else {
                args(input["options"].clone())?
            };
            return value(hifishifter_kernel::search::transliterate_batch(
                &texts, &options,
            ));
        }
        "read_system_clipboard_object" => {
            return match hifishifter_clipboard::read_bytes()? {
                Some(bytes) => match String::from_utf8(bytes) {
                    Ok(payload) => Ok(json!({"ok":true,"available":true,"payload":payload})),
                    Err(_) => Ok(json!({"ok":true,"available":false})),
                },
                None => Ok(json!({"ok":true,"available":false})),
            };
        }
        "write_system_clipboard_object" => {
            let payload = input["payload"]
                .as_str()
                .ok_or("clipboard payload required")?;
            if payload.len() > 8 * 1024 * 1024 {
                return Err("parameter clipboard budget exceeded".into());
            }
            let decoded: Value = args(serde_json::from_str(payload).map_err(|e| e.to_string())?)?;
            if decoded["kind"] != "param" {
                return Err("timeline clipboard geometry is controlled by host".into());
            }
            hifishifter_clipboard::write_bytes(
                payload.as_bytes(),
                input["textSummary"]
                    .as_str()
                    .unwrap_or("HiFiShifter parameter data copied."),
            )?;
            return Ok(json!({"ok":true}));
        }
        "clipboard_kind" => {
            let kind = hifishifter_clipboard::read_bytes()?
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|payload| super::host_clipboard::clipboard_kind(&payload));
            return Ok(json!({"ok":true,"kind":kind}));
        }
        "emit_ui_event" => {
            let event = input["event"].as_str().ok_or("event missing")?;
            session.emit(event, input["payload"].clone());
            return Ok(Value::Null);
        }
        _ => {}
    }
    if let Err(error) = session.ensure_loaded(false) {
        if matches!(command, "get_timeline_state" | "get_timeline_state_lite") {
            if let Some(document) = session.document.upgrade() {
                if !document.ui_tracks.lock().unwrap().is_empty() {
                    {
                        let mut timeline = session.timeline.lock().unwrap();
                        document.present_host_inventory(&mut timeline, &session.namespace);
                    }
                    return payload(session, command == "get_timeline_state_lite");
                }
            }
        }
        return Err(error);
    }
    match command {
        "get_timeline_state" => payload(session, false),
        "move_track" => {
            let request: PrivateTrackMove = args(input)?;
            session.move_private_track(
                &request.track_id,
                request.target_index,
                request.parent_track_id,
            )?;
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 工程级拉伸覆盖（"拉伸算法 / 声码器 Mel 拉伸"子菜单）。
        //
        // 【为什么插件要支持它，而不是像缓存清理那样禁用】它控制的是**插件自己的**
        // 音频处理（拉伸算法、HiFiGAN 的 mel 拉伸），不是宿主的属性 —— 在插件里
        // 它同样有意义。此前这条命令落到 `Command unavailable`，于是整个子菜单点了
        // 没有任何反应。
        //
        // 【为什么不能只写 ProjectState】真正的效果来自进程级的拉伸配置
        // （`update_project_stretch_overrides`）；只改 `ProjectState` 会让界面上的
        // 勾选要等下一次渲染才可能被读到。
        "set_project_stretch_settings" => {
            let algorithm: Option<hifishifter_kernel::time_stretch::UserStretchAlgorithm> =
                serde_json::from_value(input["stretchAlgorithmOverride"].clone())
                    .map_err(|_| "invalid stretch algorithm override".to_string())?;
            let hifigan: Option<bool> =
                serde_json::from_value(input["hifiganMelStretchOverride"].clone())
                    .map_err(|_| "invalid hifigan mel stretch override".to_string())?;
            {
                let mut project = session.project.lock().unwrap();
                project.stretch_algorithm_override = algorithm;
                project.hifigan_mel_stretch_override = hifigan;
            }
            hifishifter_kernel::time_stretch::update_project_stretch_overrides(algorithm, hifigan);
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 工程基准音阶。
        //
        // 【为什么插件必须支持它】REAPER **没有工程调号概念**（音阶只能是
        // HiFiShifter 自有的设置），而 HiFiShifter 的音高吸附、级数渲染与渲染缓存键
        // 全都锚定它。此前这条命令落到 `Command unavailable`，于是插件里音阶选择器
        // 是灰的 —— 用户拿不到一整套依赖音阶的功能。
        //
        // 【为什么要落盘】`ProjectState` 是 per-ARA-document 且从不写的；只改它
        // 会让用户选的音阶在换工程/重启后归零。真正的家是 `UiSettings`。
        "set_project_base_scale" => {
            let requested = input["baseScale"].as_str().unwrap_or_default();
            let scale = hifishifter_kernel::state::model::normalize_scale_key(requested);
            {
                let mut project = session.project.lock().unwrap();
                project.base_scale = scale.clone();
                project.use_custom_scale = false;
            }
            crate::settings_store::save_settings_patch(&json!({
                "pluginMusicalContext": {
                    "baseScale": scale,
                    "useCustomScale": false,
                }
            }))?;
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        "set_project_custom_scale" => {
            let custom: hifishifter_kernel::project::CustomScale =
                serde_json::from_value(input["customScale"].clone())
                    .map_err(|_| "invalid custom scale".to_string())?;
            let normalized = custom.normalized();
            {
                let mut project = session.project.lock().unwrap();
                project.custom_scale = Some(normalized.clone());
                project.use_custom_scale = true;
            }
            crate::settings_store::save_settings_patch(&json!({
                "pluginMusicalContext": {
                    "useCustomScale": true,
                    "customScale": normalized,
                }
            }))?;
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        "get_timeline_state_lite" => payload(session, true),
        "get_project_meta" => Ok(payload(session, true)?["project"].clone()),
        // vslib 是闭源库，插件包**刻意不随附**（见 `hifishifter-plugin/Cargo.toml` 的
        // 说明）。此前这条落到 `Command unavailable`，于是每次启动都有一次失败
        // invoke，前端只能把 vslib 显示成"未知"。如实回答"没编进来 + 为什么"，
        // 算法列表才能按能力过滤掉它。
        "get_vslib_status" => Ok(json!({
            "compiled": false,
            "available": false,
            "version": serde_json::Value::Null,
            "error": "vslib is not bundled with the ARA plugin",
        })),
        // 记事本正文：独立 App 把它写进工程文件；插件没有工程文件，写进插件自己的
        // 数据目录（见 `editor/notebook.rs`）。此前这条落到 `Command unavailable`，
        // 而前端把失败吞掉 —— 症状是"在插件里写的笔记重载即消失"。
        //
        // 【为什么不登记为撤销步】插件的撤销栈只记时间轴与参数；把记事本塞进去会
        // 让"撤销"在两套权威之间跳。用户要撤销的是编辑动作，不是打字。
        "set_project_notes" => {
            let markdown = input["notesMarkdown"].as_str().unwrap_or_default();
            super::notebook::store().set_notes(markdown)?;
            session.project.lock().unwrap().notes_markdown = markdown.to_owned();
            payload(session, true)
        }
        "seal_project_notes_history" => Ok(super::notebook::seal_notes_history()),
        "notebook_put_asset" => {
            let result = super::notebook::store().put_asset(
                input["assetId"].as_str().unwrap_or_default(),
                input["kind"].as_str().unwrap_or("image"),
                input["ext"].as_str().unwrap_or("bin"),
                input["mime"].as_str(),
                input["dataBase64"].as_str().unwrap_or_default(),
                input.get("meta").cloned().filter(|v| !v.is_null()),
            )?;
            Ok(result)
        }
        "notebook_read_asset" => {
            Ok(super::notebook::store().read_asset(input["assetId"].as_str().unwrap_or_default()))
        }
        "notebook_list_assets" => Ok(super::notebook::store().list_assets()),
        "notebook_remove_asset" => Ok(
            super::notebook::store().remove_asset(input["assetId"].as_str().unwrap_or_default())
        ),
        "notebook_prune_assets" => Ok(super::notebook::store().prune_assets()),
        // 纯文件读取：不碰工程状态，也不需要宿主。
        "notebook_read_file_base64" => Ok(super::notebook::read_file_base64(
            input["path"].as_str().unwrap_or_default(),
            input["maxBytes"].as_u64(),
        )),
        "notebook_read_clipboard_image" => Ok(super::notebook::read_clipboard_image()),
        "get_runtime_info" => {
            let timeline = payload(session, true)?;
            let (_, playing) = session.transport();
            Ok(
                json!({"ok":true,"device":"REAPER / ARA","model_loaded":hifishifter_kernel::world_vocoder::is_available(),
                "audio_loaded":!session.timeline.lock().unwrap().clips.is_empty(),"has_synthesized":session.applied.load(Ordering::Acquire)>0,
                "is_playing":playing,"playback_target":if playing {Some("synthesized")} else {None},"gpu_backend":"","timeline":timeline}),
            )
        }
        "get_history_state" => Ok(history_state(session)),
        "begin_undo_group" => {
            let timeline = session.timeline.lock().unwrap();
            history::checkpoint(
                &mut session.history.lock().unwrap(),
                &timeline,
                input["label"].as_str().unwrap_or("batch").into(),
                || None,
            );
            session.suppress_history.store(true, Ordering::Release);
            drop(timeline);
            session.emit("history_state", history_state(session));
            payload(session, false)
        }
        "end_undo_group" => {
            session.suppress_history.store(false, Ordering::Release);
            Ok(json!({"ok":true}))
        }
        "set_transport" => {
            if input["bpm"].is_number() {
                return Err("tempo is controlled by REAPER".into());
            }
            if let Some(position) = input["playheadSec"].as_f64() {
                if position.is_finite() {
                    session.timeline.lock().unwrap().playhead_sec = position.max(0.);
                }
            }
            payload(session, true)
        }
        // 【为什么必须支持它】ActionBar 的网格下拉与"吸附/网格设置"对话框都走这条
        // 命令；插件此前对它回 `Command unavailable`，于是那些控件**看起来能用、
        // 点了什么也不会发生**。
        //
        // 【为什么拍号在这里被忽略】拍号是宿主权威（`render::transport` 从 VST3
        // 进程上下文读，见 `time_signature`），插件不写它。但前端在改网格时会把
        // **当前**拍号一并放进同一个补丁，所以这里不能因为看到拍号就报错 ——
        // 只接受网格那部分。真正要改拍号时 UI 是灰的（`ActionBar` 里按插件模式禁用）。
        "set_project_timeline_settings" => {
            let grid = input["gridSize"]
                .as_str()
                .map(hifishifter_kernel::config::TimelineSnapSettings::normalize_grid_size);
            if let Some(grid) = grid {
                session.project.lock().unwrap().grid_size = grid.clone();
                // 网格是 HiFiShifter 自有的设置（宿主没有对应概念），因此要**落盘**：
                // 只写进 per-document 的 `ProjectState` 会在换工程/重启后归零。
                crate::settings_store::save_settings_patch(&json!({ "gridSize": grid }))?;
            }
            after_write(session, json!({"ok":true}))?;
            payload(session, true)
        }
        // 长分析的轮询兜底。事件通道（`pitch_orig_analysis_progress`）在正常情况下
        // 就够了，但事件可能在窗口尚未挂载时发出 —— 那时前端只能靠轮询这条把
        // "正在分析第 3/8 个片段"补回来。此前这里恒回 `null`，于是**首次**分析
        // 期间界面完全没有进度，看起来像卡死。
        "get_pitch_analysis_progress" => Ok(
            match hifishifter_kernel::pitch_clip::get_clip_pitch_batch_progress() {
                Some(batch) => json!({
                    "rootTrackId": "",
                    "progress": batch.progress,
                    "currentClipName": batch.current_clip_name,
                    "completedClips": batch.completed_clips,
                    "totalClips": batch.total_clips,
                }),
                None => Value::Null,
            },
        ),
        "get_track_summary" => {
            let timeline = session.timeline.lock().unwrap();
            Ok(
                json!({"ok":true,"track_id":input["trackId"].as_str().map(str::to_owned).or_else(||timeline.selected_track_id.clone()),
                "waveform_preview":[],"pitch_range":{"min":-24,"max":24}}),
            )
        }
        "get_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.frame_count as usize)?;
            value(params::get_param_frames(
                session,
                a.track_id,
                a.param,
                a.start_frame,
                a.frame_count,
                a.stride,
                a.binary,
                a.with_sentinel,
            ))
        }
        "set_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.values.len())?;
            after_write(
                session,
                params::set_param_frames(
                    session,
                    a.track_id,
                    a.param,
                    a.start_frame,
                    a.values,
                    a.checkpoint,
                ),
            )
        }
        "restore_param_frames" => {
            let a: Frames = args(input)?;
            track_exists(session, &a.track_id)?;
            budget(a.start_frame, a.frame_count as usize)?;
            after_write(
                session,
                params::restore_param_frames(
                    session,
                    a.track_id,
                    a.param,
                    a.start_frame,
                    a.frame_count,
                    a.checkpoint,
                ),
            )
        }
        "get_static_param" => {
            let a: Static = args(input)?;
            track_exists(session, &a.track_id)?;
            value(params::get_static_param(session, a.track_id, a.param))
        }
        "set_static_param" => {
            let a: Static = args(input)?;
            track_exists(session, &a.track_id)?;
            let number = a
                .value
                .filter(|n| n.is_finite())
                .ok_or("finite static parameter required")?;
            after_write(
                session,
                params::set_static_param(session, a.track_id, a.param, number, a.checkpoint),
            )
        }
        "convert_mix_param" => {
            let a: Mix = args(input)?;
            track_exists(session, &a.track_id)?;
            for range in &a.ranges {
                budget(range.start_frame, range.frame_count as usize)?;
            }
            after_write(
                session,
                params::convert_mix_param(session, a.track_id, a.from, a.ranges),
            )
        }
        // 「拉伸时锁定参数线」的时域映射。
        //
        // 【为什么必须有】它是**无条件**被调用的：前端在 clip 拉伸手势收尾时总会
        // 发它（见 `params.ts` 的调用点）。此前这条落到 `Command unavailable`，
        // 于是拉伸后参数线与片段长度对不上 —— 而失败只留一行日志，用户看到的是
        // "拉伸坏了"。它只改 HFS 自己的参数曲线、不碰宿主几何，所以插件里完全可用。
        "stretch_track_linked_params" => {
            let a: StretchLinked = args(input)?;
            track_exists(session, &a.track_id)?;
            if a.mappings.len() > 4096 {
                return Err("stretch mapping budget exceeded".into());
            }
            after_write(
                session,
                params::stretch_track_linked_params(session, a.track_id, a.mappings, a.checkpoint),
            )
        }
        "set_track_state" => {
            let object = input.as_object().ok_or("track arguments required")?;
            for (key, value) in object {
                if !matches!(
                    key.as_str(),
                    "trackId"
                        | "muted"
                        | "solo"
                        | "volume"
                        | "composeEnabled"
                        | "pitchAnalysisAlgo"
                ) && !value.is_null()
                {
                    return Err(format!("track property controlled by host: {key}"));
                }
            }
            let patch: TrackPatch = args(input.clone())?;
            // 工程反序列化为前向兼容保留Unknown；实时GUI命令不能把它当可执行算法。
            if patch.pitch_analysis_algo.as_ref().is_some_and(|algo| {
                matches!(
                    algo,
                    PitchAnalysisAlgo::Unknown | PitchAnalysisAlgo::VocalShifterVslib
                )
            }) {
                return Err("algorithm unavailable in plugin mode".into());
            }
            let id = patch.track_id.as_str();
            track_exists(session, id)?;
            if patch
                .volume
                .is_some_and(|n| !n.is_finite() || !(0.0..=4.0).contains(&n))
            {
                return Err("track volume out of range".into());
            }
            let mut timeline = session.timeline.lock().unwrap();
            let mut candidate = timeline.clone();
            let track = candidate.tracks.iter_mut().find(|t| t.id == id).unwrap();
            if let Some(volume) = patch.volume {
                track.volume = volume;
            }
            if let Some(value) = patch.muted {
                track.muted = value;
            }
            if let Some(value) = patch.solo {
                track.solo = value;
            }
            if let Some(value) = patch.compose_enabled {
                track.compose_enabled = value;
            }
            if let Some(value) = patch.pitch_analysis_algo {
                track.pitch_analysis_algo = value;
            }
            session.checkpoint_timeline(&timeline, HistoryOp::EditTrack);
            *timeline = candidate;
            session.mark_dirty();
            session.publish_timeline(timeline.clone());
            drop(timeline);
            after_write(session, json!({"ok":true}))?;
            payload(session, false)
        }
        "undo_timeline"
        | "redo_timeline"
        | "undo_parameter_edit"
        | "redo_parameter_edit"
        | "set_history_position" => {
            let mut timeline = session.timeline.lock().unwrap();
            let mut recorded = session.history.lock().unwrap();
            let (target, intent) = match command {
                "undo_timeline" | "undo_parameter_edit" => {
                    (recorded.position.saturating_sub(1), HistoryJumpIntent::Undo)
                }
                "redo_timeline" | "redo_parameter_edit" => {
                    (recorded.position.saturating_add(1), HistoryJumpIntent::Redo)
                }
                _ => (
                    input["position"]
                        .as_u64()
                        .ok_or("history position missing")? as usize,
                    HistoryJumpIntent::Jump,
                ),
            };
            if let Some((next, _, selection)) =
                history::jump(&mut recorded, &timeline, target, intent, None)
            {
                if matches!(command, "undo_parameter_edit" | "redo_parameter_edit") {
                    // 参数快照只恢复控制数据，绝不把旧clip位置/源窗口写回或覆盖当前宿主几何。
                    timeline.params_by_root_track = next.params_by_root_track;
                    for track in &mut timeline.tracks {
                        if let Some(old) = next.tracks.iter().find(|old| old.id == track.id) {
                            track.volume = old.volume;
                            track.muted = old.muted;
                            track.solo = old.solo;
                            track.compose_enabled = old.compose_enabled;
                            track.pitch_analysis_algo = old.pitch_analysis_algo.clone();
                        }
                    }
                } else {
                    *timeline = next;
                }
                drop(recorded);
                session.mark_dirty();
                session.publish_timeline(timeline.clone());
                drop(timeline);
                session.emit("history_state", history_state(session));
                session.notify_timeline();
                let mut payload = payload(session, false)?;
                if let Some(selection) = selection {
                    payload["param_selection_restore"] = json!(selection);
                }
                if let Some(error) = session.error.lock().unwrap().clone() {
                    return Err(error);
                }
                Ok(payload)
            } else {
                drop(recorded);
                drop(timeline);
                payload(session, false)
            }
        }
        "get_waveform_mipmap_binary" => {
            let path = input["sourcePath"].as_str().ok_or("sourcePath missing")?;
            let level = input["level"].as_u64().unwrap_or(2).min(2) as usize;
            Ok(Value::String(
                base64::engine::general_purpose::STANDARD
                    .encode(peaks(session, path)?.to_binary_level(level)),
            ))
        }
        "preload_waveform_mipmap" => {
            peaks(
                session,
                input["sourcePath"].as_str().ok_or("sourcePath missing")?,
            )?;
            Ok(json!({"ok":true}))
        }
        "batch_get_waveform_mipmap" => {
            let paths: Vec<String> = args(input["sourcePaths"].clone())?;
            let levels: Option<Vec<usize>> = args(input["levels"].clone())?;
            if paths.len() > 64 {
                return Err("too many waveform sources".into());
            }
            let mut result = serde_json::Map::new();
            for path in paths {
                let data = peaks(session, &path)?;
                let encoded: Vec<_> = (0..3)
                    .map(|level| {
                        if levels
                            .as_ref()
                            .is_none_or(|l| l.is_empty() || l.contains(&level))
                        {
                            base64::engine::general_purpose::STANDARD
                                .encode(data.to_binary_level(level))
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                result.insert(path, json!(encoded));
            }
            Ok(Value::Object(result))
        }
        "get_root_mix_waveform_peaks_segment" | "get_track_mix_waveform_peaks_segment" => {
            let a: Segment = args(input)?;
            track_exists(session, &a.track_id)?;
            if !a.start_sec.is_finite() || !a.duration_sec.is_finite() || a.duration_sec <= 0. {
                return Err("invalid waveform range".into());
            }
            if a.duration_sec * 48000.0 * 2.0 * 4.0 > 64.0 * 1024.0 * 1024.0 {
                return Err("waveform mix buffer budget exceeded".into());
            }
            if command == "get_root_mix_waveform_peaks_segment" {
                value(waveform::get_root_mix_waveform_peaks_segment(
                    session,
                    a.track_id,
                    a.start_sec,
                    a.duration_sec,
                    a.columns,
                ))
            } else {
                value(waveform::get_track_mix_waveform_peaks_segment(
                    session,
                    a.track_id,
                    a.start_sec,
                    a.duration_sec,
                    a.columns,
                ))
            }
        }
        _ => Err(format!("Command unavailable in ARA plugin mode: {command}")),
    }
}
