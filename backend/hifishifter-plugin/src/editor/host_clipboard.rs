//! 原GUI媒体剪贴板的宿主执行层：保存真实item状态和源域曲线，不保存宿主指针。
use super::parameter_atlas::{ParameterAtlas, RegionGeometry, RegionParameters};
use super::session::EditorSession;
use crate::host::geometry::HostClipGeometry;
use crate::host::reaper::{HostClipTarget, HostTrackTarget, ReaperHost, RewrittenItem};
use crate::render::document::DocumentSession;
use crate::render::extension::ExtensionOwner;
use hifishifter_kernel::state::Clip;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::Arc;

const FORMAT: &str = "hifishifter-reaper-items-1";
const MAX_CLIPS: usize = 512;
const MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct ClipboardTrack {
    name: String,
}
#[derive(Serialize, Deserialize)]
struct ClipboardItem {
    track: usize,
    chunk: String,
    before: Clip,
    seed: Option<RegionParameters>,
}
#[derive(Serialize, Deserialize)]
struct ClipboardItems {
    format: String,
    version: u32,
    kind: String,
    origin_sec: f64,
    tracks: Vec<ClipboardTrack>,
    items: Vec<ClipboardItem>,
}

struct PasteItem {
    slot: usize,
    state: RewrittenItem,
    before: Clip,
    start_sec: f64,
    seed: Option<RegionParameters>,
}
pub(super) struct PastePlan {
    host: Arc<ReaperHost>,
    tracks: Vec<(String, Option<HostTrackTarget>)>,
    items: Vec<PasteItem>,
}
pub(super) struct DeletePlan {
    targets: Vec<(HostClipTarget, String)>,
}
pub(super) struct MediaReceipt {
    created: Vec<(String, HostClipGeometry)>,
    deleted: Vec<(String, String)>,
}

/// 正常GUI已有已加载状态；不在native主线程materialize素材或等待DSP。
fn selected_clips(
    editor: &EditorSession,
    input: &Value,
    single: bool,
) -> Result<Vec<Clip>, String> {
    let ids = if single {
        vec![input["clipId"].as_str().ok_or("clipId missing")?.to_owned()]
    } else {
        input["clipIds"]
            .as_array()
            .ok_or("clipIds missing")?
            .iter()
            .map(|id| id.as_str().map(str::to_owned).ok_or("invalid clipId"))
            .collect::<Result<Vec<_>, _>>()?
    };
    if ids.is_empty() || ids.len() > MAX_CLIPS {
        return Err("clip operation batch must contain 1..512 items".into());
    }
    let timeline = editor.timeline.lock().unwrap();
    let mut seen = BTreeSet::new();
    ids.into_iter()
        .map(|id| {
            if !seen.insert(id.clone()) {
                return Err("duplicate selected clip".into());
            }
            if !id.starts_with(&editor.namespace) {
                return Err("clip belongs to another editor".into());
            }
            timeline
                .clips
                .iter()
                .find(|clip| clip.id == id)
                .cloned()
                .ok_or("unknown GUI clip".into())
        })
        .collect()
}

/// 源窗口与有效倍率也预检，避免复制刚被宿主替换但GUI尚未回流的错误曲线。
fn target_for(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    clip: &Clip,
    allowed: &impl Fn() -> bool,
) -> Result<HostClipTarget, String> {
    let native = clip
        .id
        .strip_prefix(&editor.namespace)
        .ok_or("foreign editor clip")?;
    let target = owner.host_edit_target(native)?.current(allowed)?;
    let g = &target.geometry;
    if clip.reversed || !g.markers.is_empty() {
        return Err("reverse/nonlinear clip clipboard editing is unsupported".into());
    }
    for (actual, expected) in [
        (g.start_sec, clip.start_sec),
        (g.duration_sec, clip.length_sec),
        (g.source_start_sec, clip.source_start_sec),
        (g.playback_rate, clip.playback_rate as f64),
    ] {
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > 1e-6 {
            return Err("host clip changed before media operation; refresh required".into());
        }
    }
    Ok(target)
}

/// 必须在actor写入屏障之后调用；剪切前先完成系统剪贴板写入，失败不删除原item。
fn capture(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    input: &Value,
    allowed: &impl Fn() -> bool,
) -> Result<ClipboardItems, String> {
    let document = owner.editor_document()?;
    let mut clips = selected_clips(editor, input, false)?;
    let mut tracks = editor.timeline.lock().unwrap().tracks.clone();
    tracks.sort_by_key(|track| track.order);
    tracks.retain(|track| clips.iter().any(|clip| clip.track_id == track.id));
    let origin_sec = clips
        .iter()
        .map(|clip| clip.start_sec)
        .fold(f64::INFINITY, f64::min);
    clips.sort_by(|a, b| a.start_sec.total_cmp(&b.start_sec).then(a.id.cmp(&b.id)));
    let mut items = Vec::new();
    let mut bytes = 0_usize;
    let mut first: Option<HostClipTarget> = None;
    for clip in clips {
        let target = target_for(owner, editor, &clip, allowed)?;
        if first
            .as_ref()
            .is_some_and(|first| !first.same_project(&target))
        {
            return Err("clipboard selection spans multiple projects".into());
        }
        if first.is_none() {
            first = Some(target.clone());
        }
        let chunk = target.capture_item_state(allowed)?;
        bytes = bytes
            .checked_add(chunk.len())
            .ok_or("clipboard byte overflow")?;
        if bytes > MAX_BYTES {
            return Err("clip clipboard item state budget exceeded".into());
        }
        let track = tracks
            .iter()
            .position(|track| track.id == clip.track_id)
            .ok_or("clipboard source track missing")?;
        let seed = {
            let _transaction = document.transaction.lock().unwrap();
            document
                .edits
                .lock()
                .unwrap()
                .atlas
                .split_seed(&target.geometry.item_id)
        };
        items.push(ClipboardItem {
            track,
            chunk,
            before: clip,
            seed,
        });
    }
    if !allowed() {
        return Err("clipboard editor lease changed".into());
    }
    let object = ClipboardItems {
        format: FORMAT.into(),
        version: 1,
        kind: "clips".into(),
        origin_sec,
        tracks: tracks
            .into_iter()
            .map(|track| ClipboardTrack { name: track.name })
            .collect(),
        items,
    };
    Ok(object)
}
/// 系统复制和Ctrl拖动共享宿主捕获，但Ctrl拖动不能覆盖用户已有参数/片段剪贴板。
pub(super) fn copy(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    input: &Value,
    allowed: &impl Fn() -> bool,
) -> Result<Value, String> {
    let object = capture(owner, editor, input, allowed)?;
    let bytes = serde_json::to_vec(&object).map_err(|error| error.to_string())?;
    if bytes.len() > MAX_BYTES {
        return Err("clip clipboard state and curve budget exceeded".into());
    }
    hifishifter_clipboard::write_bytes(
        &bytes,
        &format!("HiFiShifter: {} REAPER clips copied.", object.items.len()),
    )?;
    Ok(json!({"ok":true,"kind":"clips"}))
}

/// 只接受本原生剪贴板格式；独立App/REAPER的其它格式不伪装成已实现的宿主粘贴。
fn decode(bytes: &[u8]) -> Result<ClipboardItems, String> {
    if bytes.len() > MAX_BYTES {
        return Err("clip clipboard budget exceeded".into());
    }
    let mut data: ClipboardItems = serde_json::from_slice(bytes)
        .map_err(|_| "clipboard does not contain native HiFiShifter clips")?;
    if data.format != FORMAT
        || data.version != 1
        || data.kind != "clips"
        || data.items.is_empty()
        || data.items.len() > MAX_CLIPS
        || data.tracks.is_empty()
        || data.tracks.len() > MAX_CLIPS
        || !data.origin_sec.is_finite()
        || data.origin_sec < 0.0
    {
        return Err("invalid native clip clipboard".into());
    }
    for track in &data.tracks {
        if track.name.len() > 4096 || track.name.contains('\0') {
            return Err("invalid clipboard track name".into());
        }
    }
    for item in &mut data.items {
        // Clip媒体投影字段在存储中省略，源窗口/倍率的权威是takes，不能拿默认0/1去核对宿主。
        item.before.normalize_takes();
        if item.track >= data.tracks.len()
            || !item.before.start_sec.is_finite()
            || item.before.start_sec < data.origin_sec
            || !item.before.length_sec.is_finite()
            || item.before.length_sec <= 0.0
            || !item.before.source_start_sec.is_finite()
            || item.before.source_start_sec < 0.0
            || !item.before.playback_rate.is_finite()
            || item.before.playback_rate <= 0.0
            || item.before.reversed
        {
            return Err("invalid clipboard clip geometry".into());
        }
        if let Some(seed) = item.seed.take() {
            item.seed = Some(ParameterAtlas::reserve_copy_seed(seed)?);
        }
    }
    if data
        .items
        .iter()
        .map(|item| item.track)
        .collect::<BTreeSet<_>>()
        .len()
        != data.tracks.len()
    {
        return Err("clipboard contains unused track slots".into());
    }
    Ok(data)
}

/// 同一个结构化剪贴板协议同时服务参数与clip路由，不能将clip类型过滤成null。
pub(super) fn clipboard_kind(payload: &Value) -> Option<&'static str> {
    match payload["kind"].as_str() {
        Some("param") => Some("param"),
        Some("clips") if payload["format"] == FORMAT && payload["version"] == 1 => Some("clips"),
        _ => None,
    }
}

/// Ctrl拖动复制直接捕获当前宿主item，目标来自明确GUI映射，不经系统剪贴板或全局粘贴action。
pub(super) fn plan_duplicate(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    input: &Value,
    allowed: &impl Fn() -> bool,
) -> Result<PastePlan, String> {
    let input = input.get("payload").unwrap_or(input);
    let delta = input["deltaSec"]
        .as_f64()
        .filter(|delta| delta.is_finite())
        .ok_or("finite duplicate delta required")?;
    let mut data = capture(
        owner,
        editor,
        &json!({"clipIds":input["sourceClipIds"]}),
        allowed,
    )?;
    let host = owner
        .project_history_host()
        .filter(|host| host.can_clipboard_items())
        .ok_or("host duplicate capability unavailable")?;
    let mode = input["trackMode"]["kind"].as_str().unwrap_or("same_track");
    if !["same_track", "explicit_mapping", "new_tracks"].contains(&mode) {
        return Err("unsupported host duplicate track mode".into());
    }
    let mut slots = Vec::<String>::new();
    let mut tracks = Vec::new();
    let mut items = Vec::new();
    let mut generated = BTreeSet::new();
    if mode == "new_tracks" {
        let span = input["trackMode"]["span"]
            .as_u64()
            .filter(|span| (1..=512).contains(span))
            .ok_or("invalid duplicate new-track span")? as usize;
        if !host.can_create_audio_track() {
            return Err("host new-track duplicate unavailable".into());
        }
        tracks = (0..span)
            .map(|index| (format!("Copied audio {}", index + 1), None))
            .collect();
    }
    for item in &mut data.items {
        let destination = if mode == "same_track" || mode == "new_tracks" {
            item.before.track_id.as_str()
        } else {
            input["trackMode"]["mapping"][&item.before.track_id]
                .as_str()
                .ok_or("duplicate destination mapping missing")?
        };
        let slot = if mode == "new_tracks" {
            input["trackMode"]["mapping"][&item.before.track_id]
                .as_u64()
                .filter(|slot| (*slot as usize) < tracks.len())
                .ok_or("new-track duplicate mapping missing")? as usize
        } else if let Some(slot) = slots.iter().position(|id| id == destination) {
            slot
        } else {
            let native = editor.host_track_id(destination)?;
            let target = owner.audio_import_target(Some(&native), allowed)?;
            let slot = slots.len();
            slots.push(destination.to_owned());
            tracks.push((data.tracks[item.track].name.clone(), Some(target)));
            slot
        };
        let start_sec = (item.before.start_sec + delta).max(0.);
        let state = host.prepare_copied_item(&item.chunk, start_sec, allowed)?;
        if state
            .generated_guids
            .iter()
            .any(|guid| !generated.insert(guid.clone()))
        {
            return Err("duplicate identity across drag copy batch".into());
        }
        let seed = if input["copyLinkedParams"] == false {
            None
        } else {
            item.seed.take()
        };
        items.push(PasteItem {
            slot,
            state,
            before: item.before.clone(),
            start_sec,
            seed,
        });
    }
    Ok(PastePlan {
        host,
        tracks,
        items,
    })
}

/// 可用性查询不创建/删除任何item，失败明确为空，不把参数剪贴板当片段剪贴板。
pub(super) fn available() -> Result<Value, String> {
    let data = hifishifter_clipboard::read_bytes()?.and_then(|bytes| decode(&bytes).ok());
    Ok(
        json!({"ok":true,"available":data.is_some(),"kind":data.as_ref().map(|_|"clips"),
        "clipCount":data.as_ref().map(|data|data.items.len()).unwrap_or(0)}),
    )
}

/// 所有数据、目标与GUID先预检，再开始原生Undo块；默认粘贴保留轨道相对关系。
pub(super) fn plan_paste(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    input: &Value,
    allowed: &impl Fn() -> bool,
) -> Result<PastePlan, String> {
    let bytes = hifishifter_clipboard::read_bytes()?.ok_or("no clip clipboard")?;
    let data = decode(&bytes)?;
    let mode = input["mode"].as_str().unwrap_or("selected");
    if !["selected", "new_tracks"].contains(&mode) {
        return Err("unknown native paste mode".into());
    }
    let host = owner
        .project_history_host()
        .filter(|host| host.can_clipboard_items())
        .ok_or("host clipboard capability unavailable")?;
    let (mut current, selected) = {
        let timeline = editor.timeline.lock().unwrap();
        (timeline.tracks.clone(), timeline.selected_track_id.clone())
    };
    current.sort_by_key(|track| track.order);
    let base = if mode == "selected" {
        let id = selected.ok_or("select a target track before pasting")?;
        current
            .iter()
            .position(|track| track.id == id)
            .ok_or("selected target track missing")?
    } else {
        current.len()
    };
    let mut tracks = Vec::new();
    for (slot, source) in data.tracks.into_iter().enumerate() {
        let target = if let Some(track) = current.get(base + slot) {
            let id = editor.host_track_id(&track.id)?;
            Some(owner.audio_import_target(Some(&id), allowed)?)
        } else {
            if !host.can_create_audio_track() {
                return Err("paste requires more host tracks; new-track API unavailable".into());
            }
            None
        };
        tracks.push((source.name, target));
    }
    owner.refresh_reaper_transport();
    let anchor = owner
        .clock
        .get()
        .map(|clock| clock.read().0)
        .ok_or("host playhead unavailable")?;
    let mut items = Vec::new();
    let mut generated = BTreeSet::new();
    for item in data.items {
        let start_sec = anchor + item.before.start_sec - data.origin_sec;
        let state = host.prepare_copied_item(&item.chunk, start_sec, allowed)?;
        if state
            .generated_guids
            .iter()
            .any(|guid| !generated.insert(guid.clone()))
        {
            return Err("duplicate generated identity across paste batch".into());
        }
        items.push(PasteItem {
            slot: item.track,
            state,
            before: item.before,
            start_sec,
            seed: item.seed,
        });
    }
    Ok(PastePlan {
        host,
        tracks,
        items,
    })
}

/// 真实新GUID在state loader重入前登记seed；回流不重发创建，部分失败保留宿主Undo。
pub(super) fn execute_paste(
    owner: &Arc<ExtensionOwner>,
    plan: PastePlan,
    allowed: &impl Fn() -> bool,
) -> Result<MediaReceipt, String> {
    let document = owner.editor_document()?;
    let mut tracks = Vec::new();
    let mut new_tracks = Vec::new();
    for (name, target) in plan.tracks {
        let target = if let Some(target) = target {
            target
        } else {
            let track = plan.host.create_audio_track(&name, None, allowed)?;
            let target = track.target.clone();
            new_tracks.push(track);
            target
        };
        tracks.push(target);
    }
    let mut created = Vec::new();
    for item in plan.items {
        let target = &tracks[item.slot];
        owner.refresh_reaper_transport();
        let root = {
            let tracks = document.ui_tracks.lock().unwrap();
            tracks
                .get(target.inventory_guid())
                .map(|track| track.id.clone())
        }
        .ok_or("new paste track has not entered host inventory")?;
        let geometry = RegionGeometry {
            project_start: item.start_sec,
            project_duration: item.before.length_sec,
            source_start: item.before.source_start_sec,
            source_duration: item.before.length_sec * item.before.playback_rate as f64,
        };
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut edits = document.edits.lock().unwrap();
            edits
                .atlas
                .register_copy(&item.state.item_guid, &root, geometry, item.seed)?;
            edits.revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
        }
        let outcome = target.create_copied_item(&item.state, allowed);
        let actual = match outcome {
            Ok(actual) => actual,
            Err(error) => {
                let _transaction = document.transaction.lock().unwrap();
                let mut edits = document.edits.lock().unwrap();
                edits.atlas.copy_seeds.remove(&item.state.item_guid);
                edits.revision = edits
                    .revision
                    .checked_add(1)
                    .ok_or("edit revision exhausted")?;
                return Err(format!(
                    "paste stopped after {} item(s): {error}; use host Undo to cancel",
                    created.len()
                ));
            }
        };
        if [
            (actual.start_sec, item.start_sec),
            (actual.duration_sec, item.before.length_sec),
            (actual.source_start_sec, item.before.source_start_sec),
            (actual.playback_rate, item.before.playback_rate as f64),
        ]
        .iter()
        .any(|(actual, expected)| (actual - expected).abs() > 1e-6)
        {
            return Err(
                "pasted item geometry differs from copied data; host Undo remains available".into(),
            );
        }
        created.push((target.inventory_guid().to_owned(), actual));
    }
    // 原生setter在Undo块内不保证立即推进项目change计数；本批写完必须重新采集清单。
    *document.ui_inventory_stamp.lock().unwrap() = None;
    owner.refresh_reaper_transport();
    for track in &mut new_tracks {
        track.commit();
    }
    for renderer in document.renderer_owners() {
        renderer.prepare();
    }
    Ok(MediaReceipt {
        created,
        deleted: Vec::new(),
    })
}

/// 整批对象预检后删除；不存在的GUI残留、重复item或跨工程选择均不开始写入。
pub(super) fn plan_delete(
    owner: &Arc<ExtensionOwner>,
    editor: &EditorSession,
    input: &Value,
    single: bool,
    allowed: &impl Fn() -> bool,
) -> Result<DeletePlan, String> {
    let mut targets = Vec::<(HostClipTarget, String)>::new();
    for clip in selected_clips(editor, input, single)? {
        let target = target_for(owner, editor, &clip, allowed)?;
        if targets
            .iter()
            .any(|(other, _)| other.geometry.item_id == target.geometry.item_id)
            || targets
                .first()
                .is_some_and(|(first, _)| !first.same_project(&target))
        {
            return Err("duplicate item or cross-project delete batch".into());
        }
        targets.push((target, clip.id));
    }
    Ok(DeletePlan { targets })
}

/// 返回真实已删除GUID集合，后续等模型与GUI均不存在这些对象再回执。
pub(super) fn execute_delete(
    owner: &Arc<ExtensionOwner>,
    plan: DeletePlan,
    allowed: &impl Fn() -> bool,
) -> Result<MediaReceipt, String> {
    let mut deleted = Vec::new();
    for (target, id) in plan.targets {
        if let Err(error) = target.delete_item(allowed) {
            return Err(format!(
                "delete stopped after {} item(s): {error}; host Undo remains available",
                deleted.len()
            ));
        }
        deleted.push((target.geometry.item_id.clone(), id));
    }
    let document = owner.editor_document()?;
    *document.ui_inventory_stamp.lock().unwrap() = None;
    owner.refresh_reaper_transport();
    Ok(MediaReceipt {
        created: Vec::new(),
        deleted,
    })
}

impl MediaReceipt {
    /// 媒体创建/删除按真实清单GUID、几何和目标轨道回执；音频就绪另行标记，不阻塞显示。
    pub(super) fn matches(
        &self,
        document: &DocumentSession,
        namespace: &str,
        payload: &mut Value,
    ) -> bool {
        let Some(clips) = payload["clips"].as_array() else {
            return false;
        };
        for (item, old_id) in &self.deleted {
            if document.gui_clip_for_host_item(item).is_some() {
                return false;
            }
            if clips.iter().any(|clip| {
                clip["id"] == *old_id || clip["id"] == format!("{namespace}ara-item-{item}")
            }) {
                return false;
            }
        }
        let mut created_ids = Vec::new();
        let mut audio_pending = false;
        for (track_guid, g) in &self.created {
            let Some(native) = document.gui_clip_for_host_item(&g.item_id) else {
                return false;
            };
            let id = format!("{namespace}{native}");
            let Some(clip) = clips.iter().find(|clip| clip["id"] == id) else {
                return false;
            };
            // 这是成功的宿主媒体写入，不是合成成功；没有授权PCM时仍禁止私读源文件。
            audio_pending |= clip["muted"] != true
                && (document.clip_for_host_item(&g.item_id).is_none()
                    || clip["source_path"].as_str().is_none_or(str::is_empty));
            if [
                ("start_sec", g.start_sec),
                ("length_sec", g.duration_sec),
                ("source_start_sec", g.source_start_sec),
                ("playback_rate", g.playback_rate),
            ]
            .iter()
            .any(|(key, expected)| {
                clip[*key]
                    .as_f64()
                    .is_none_or(|actual| (actual - expected).abs() > 1e-6)
            }) {
                return false;
            }
            let track = document
                .ui_tracks
                .lock()
                .unwrap()
                .get(track_guid)
                .map(|track| track.id.clone());
            if track.is_none_or(|track| clip["track_id"] != format!("{namespace}{track}")) {
                return false;
            }
            created_ids.push(id);
        }
        payload["created_clip_ids"] = json!(created_ids);
        payload["host_audio_pending"] = json!(audio_pending);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确切宿主创建已回流时不等神经PCM；来源未授权仍显式pending且没有路径可读。
    #[test]
    fn goal_feedback_media_receipt_returns_before_audio_is_ready() {
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_media();
        host.clear_markers();
        host.inventory_enabled.set(true);
        let api = Arc::new(host.client());
        let track = api.ui_track(&|| true).unwrap();
        let geometry = track.items[0].geometry.clone();
        let document = DocumentSession::new(987654);
        document
            .ui_tracks
            .lock()
            .unwrap()
            .insert(track.guid.clone(), track.clone());
        let mut timeline = hifishifter_kernel::state::TimelineState::default();
        document.present_host_inventory(&mut timeline, "fast-");
        let mut payload = serde_json::to_value(timeline.to_payload()).unwrap();
        let receipt = MediaReceipt {
            created: vec![(track.guid.clone(), geometry)],
            deleted: vec![],
        };
        assert!(receipt.matches(&document, "fast-", &mut payload));
        assert_eq!(payload["host_audio_pending"], true);
        assert!(payload["clips"][0]["source_path"].is_null());
        payload["clips"][0]["start_sec"] = json!(99.);
        assert!(!receipt.matches(&document, "fast-", &mut payload));
        document.close();
    }

    fn clipboard(count: usize) -> ClipboardItems {
        let mut before:Clip=serde_json::from_value(json!({"id":"clip","track_id":"track","name":"ka","start_sec":1.,"length_sec":0.1,
            "takes":[{"id":"take","source_path":"source","source_start_sec":0.,"source_end_sec":0.1}]})).unwrap();
        before.normalize_takes();
        ClipboardItems {
            format: FORMAT.into(),
            version: 1,
            kind: "clips".into(),
            origin_sec: 1.,
            tracks: vec![ClipboardTrack {
                name: "人力测试".into(),
            }],
            items: (0..count)
                .map(|index| {
                    let mut before = before.clone();
                    before.start_sec += index as f64 * 0.11;
                    ClipboardItem {
                        track: 0,
                        chunk: "<ITEM>".into(),
                        before,
                        seed: None,
                    }
                })
                .collect(),
        }
    }

    /// 先独立保存完整数据，再解码；不引用活item，百个短clip的相对位置不会因剪切丢失。
    #[test]
    fn native_clipboard_decode_keeps_hundred_short_clip_offsets() {
        let bytes = serde_json::to_vec(&clipboard(100)).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.items.len(), 100);
        assert_eq!(decoded.tracks[0].name, "人力测试");
        assert!((decoded.items[99].before.start_sec - 1. - 99. * 0.11).abs() < 1e-12);
        assert_eq!(decoded.items[0].before.length_sec, 0.1);
    }
    /// 真剪辑不是默认源起点/速率：存储省略Clip投影后必须从Take恢复，不能只测0/1样例。
    #[test]
    fn native_clipboard_decode_restores_slipped_and_stretched_take_geometry() {
        let mut object = clipboard(1);
        let before = &mut object.items[0].before;
        before.takes[0].source_start_sec = 2.75;
        before.takes[0].source_end_sec = 3.;
        before.takes[0].playback_rate = 2.5;
        before.normalize_takes();
        let decoded = decode(&serde_json::to_vec(&object).unwrap()).unwrap();
        let copied = &decoded.items[0].before;
        assert_eq!(copied.source_start_sec, 2.75);
        assert_eq!(copied.playback_rate, 2.5);
        assert_eq!(copied.length_sec, 0.1);
    }
    /// Param与clip都参加快捷键路由，拒绝不属于本原生协议的clips数据。
    #[test]
    fn native_clipboard_kind_routes_clips_without_breaking_parameter_paste() {
        assert_eq!(
            clipboard_kind(&json!({"kind":"param","version":2})),
            Some("param")
        );
        assert_eq!(
            clipboard_kind(&json!({"kind":"clips","format":FORMAT,"version":1})),
            Some("clips")
        );
        assert_eq!(
            clipboard_kind(&json!({"kind":"clips","format":"foreign","version":1})),
            None
        );
    }

    /// 外部剪贴板的轨道下标、几何、类型和版本必须先拒绝，不能触发宿主创建。
    #[test]
    fn native_clipboard_decode_rejects_invalid_slots_geometry_and_foreign_formats() {
        for variant in 0..5 {
            let mut data = clipboard(1);
            match variant {
                0 => data.items[0].track = 99,
                1 => data.items[0].before.length_sec = -1.,
                2 => data.origin_sec = 2.,
                3 => data.version = 99,
                _ => data.format = "foreign".into(),
            }
            assert!(decode(&serde_json::to_vec(&data).unwrap()).is_err());
        }
        let mut unused = clipboard(1);
        unused.tracks.push(ClipboardTrack {
            name: "unused".into(),
        });
        assert!(decode(&serde_json::to_vec(&unused).unwrap()).is_err());
        assert!(decode(&serde_json::to_vec(&clipboard(MAX_CLIPS + 1)).unwrap()).is_err());
    }
}
