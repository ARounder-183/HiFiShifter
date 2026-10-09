//! 原GUI几何命令的纯规划与真实宿主写入；不修改actor私有timeline冒充REAPER成功。
use super::session::EditorSession;
use crate::host::reaper::HostClipTarget;
use crate::render::extension::ExtensionOwner;
use hifishifter_kernel::state::{Clip, ClipStatePatch, TimelineState};
use serde_json::Value;
use std::sync::Arc;

pub(crate) struct ClipEdit {
    pub native_id: String,
    pub source_track: String,
    pub destination_track: String,
    pub before: Clip,
    pub after: Clip,
    pub patch: ClipStatePatch,
}
pub(crate) struct HostEditPlan {
    pub edits: Vec<ClipEdit>,
}

/// 宿主setter完成后只等待本次确切clip的几何回流，不把旧的可读timeline当成功。
#[derive(Debug)]
pub(crate) struct HostEditReceipt {
    clips: Vec<(String, Vec<(&'static str, Value)>)>,
}
impl HostEditPlan {
    /// 冻结本次确切clip及预期宿主几何，单独的fade宽度也必须进入完成门。
    pub(crate) fn receipt(&self, namespace: &str) -> HostEditReceipt {
        HostEditReceipt {
            clips: self
                .edits
                .iter()
                .map(|edit| {
                    let after = &edit.after;
                    let mut fields = vec![
                        ("start_sec", serde_json::json!(after.start_sec)),
                        ("length_sec", serde_json::json!(after.length_sec)),
                        (
                            "source_start_sec",
                            serde_json::json!(after.source_start_sec),
                        ),
                        ("playback_rate", serde_json::json!(after.playback_rate)),
                        (
                            "track_id",
                            serde_json::json!(format!("{namespace}{}", edit.destination_track)),
                        ),
                    ];
                    for (value, key) in [
                        (edit.patch.fade_in_sec, "fade_in_sec"),
                        (edit.patch.fade_out_sec, "fade_out_sec"),
                        (edit.patch.auto_fade_in_sec, "auto_fade_in_sec"),
                        (edit.patch.auto_fade_out_sec, "auto_fade_out_sec"),
                        (edit.patch.snap_offset_sec, "snap_offset_sec"),
                        (edit.patch.gain.map(f64::from), "gain"),
                    ] {
                        if let Some(value) = value {
                            fields.push((key, serde_json::json!(value)));
                        }
                    }
                    if let Some(group) = edit.patch.host_group_id {
                        fields.push((
                            "group_id",
                            if group == 0 {
                                serde_json::Value::Null
                            } else {
                                serde_json::json!(format!("reaper-group-{group}"))
                            },
                        ));
                    }
                    (edit.before.id.clone(), fields)
                })
                .collect(),
        }
    }
}
impl HostEditReceipt {
    /// 仅验证回流快照，不从GUI乐观状态或路径/位置猜clip身份。
    pub(crate) fn matches(&self, payload: &Value) -> bool {
        let Some(clips) = payload["clips"].as_array() else {
            return false;
        };
        self.clips.iter().all(|(id, fields)| {
            clips
                .iter()
                .find(|clip| clip["id"] == *id)
                .is_some_and(|clip| {
                    fields
                        .iter()
                        .all(|(key, value)| match (clip[*key].as_f64(), value.as_f64()) {
                            (Some(actual), Some(expected)) => (actual - expected).abs() <= 1e-6,
                            _ => clip[*key] == *value,
                        })
                })
        })
    }
}

/// 只拦截已实现的宿主命令；未知命令仍走原actor的明确错误路径。
pub(super) fn is_clip_edit(command: &str) -> bool {
    matches!(
        command,
        "move_clip" | "move_clips" | "set_clip_state" | "set_clips_state_bulk"
    )
}

/// 请求里是否包含淡变轴字段。
fn requests_fade_axes(patch: &ClipStatePatch) -> bool {
    patch.fade_in_shape.is_some()
        || patch.fade_out_shape.is_some()
        || patch.fade_in_dir.is_some()
        || patch.fade_out_dir.is_some()
        || patch.fade_in_s.is_some()
        || patch.fade_out_s.is_some()
}

/// 淡变轴写入的合法性 —— **纯函数**，因为"哪些字段落在哪套轴上"是领域知识，
/// 不该埋在一个 200 行的循环里（那里既难读也难测）。
///
/// 官方头文件把两套轴标成互补区间（`sdk/reaper_plugin_functions.h`）：
///   `C_FADE*SHAPE` / `D_FADE*DIR`        —— v7.80 and earlier
///   `D_FADE*DIR_NEW` / `D_FADE*DIR2_NEW` —— v7.81 and later
///
/// 因此：
/// - 版本读不出来（`None`）→ 拒绝，不猜；
/// - 旧轴宿主 + S 参数 → 拒绝（S 只存在于新轴，做不到的事不静默丢弃）；
/// - 新轴宿主 + 形状预设 → **接受**。这里此前是拒绝的，理由是"预设 → (curvature, S)
///   的映射尚未校准"；该映射现已实测落地（[`hifishifter_kernel::fade_axes`]，
///   证据 `probe/ara/FADE-AXIS-FINDINGS.md`），所以预设按钮在新轴宿主上也能用。
///   表外的形状号（越界或小数变体）仍然拒绝 —— 宿主会把它静默变成别的形状
///   （实测写 `5.1` 读回 `SHAPE=7`）。
fn validate_fade_axes(axes_new: Option<bool>, patch: &ClipStatePatch) -> Result<(), String> {
    if !requests_fade_axes(patch) {
        return Ok(());
    }
    match axes_new {
        None => Err("host fade axis version unavailable; cannot write the fade shape".into()),
        Some(false) if patch.fade_in_s.is_some() || patch.fade_out_s.is_some() => {
            Err("fade S parameter requires REAPER 7.81 or later".into())
        }
        Some(true) => {
            for shape in [patch.fade_in_shape, patch.fade_out_shape]
                .into_iter()
                .flatten()
            {
                if hifishifter_kernel::fade_axes::host_fade_preset_axes(shape).is_none() {
                    return Err(format!(
                        "unknown fade shape {shape}: REAPER 7.81+ has no preset at that index"
                    ));
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// 未知非空字段必须失败，不能用户改shape却被当成成功的长度修改。
fn patch(input: &Value) -> Result<ClipStatePatch, String> {
    let object = input.as_object().ok_or("clip patch must be an object")?;
    for (key, value) in object {
        if value.is_null() {
            continue;
        }
        if !matches!(
            key.as_str(),
            "clipId"
                | "checkpoint"
                | "name"
                | "startSec"
                | "lengthSec"
                | "sourceStartSec"
                | "sourceEndSec"
                | "playbackRate"
                | "clipPlaybackRate"
                | "gain"
                | "muted"
                | "snapOffsetSec"
                | "fadeInSec"
                | "fadeOutSec"
                | "autoFadeInSec"
                | "autoFadeOutSec"
                | "fadeInShape"
                | "fadeOutShape"
                | "fadeInDir"
                | "fadeOutDir"
                | "fadeInS"
                | "fadeOutS"
                | "reversed"
                | "loopEnabled"
                | "channelMode"
                | "hostGroupId"
        ) {
            return Err(format!("host clip property is not yet writable: {key}"));
        }
        if key == "name" {
            if value
                .as_str()
                .is_none_or(|name| name.len() > 8192 || name.contains('\0'))
            {
                return Err("invalid take name".into());
            }
            continue;
        }
        if !matches!(
            key.as_str(),
            "clipId" | "checkpoint" | "muted" | "reversed" | "loopEnabled"
        ) {
            let number = value
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("finite numeric clip edit required")?;
            if matches!(
                key.as_str(),
                "fadeInDir" | "fadeOutDir" | "fadeInS" | "fadeOutS"
            ) {
                // 官方头文件把两个新轴都标为 -1..1（curvature / S parameter）。
                if !(-1.0..=1.0).contains(&number) {
                    return Err("fade curvature/S outside -1..1".into());
                }
            } else if !(0. ..=1_000_000.).contains(&number) {
                return Err("clip edit outside supported finite range".into());
            }
            if matches!(key.as_str(), "fadeInShape" | "fadeOutShape") && number >= 7. {
                return Err("unknown fade shape".into());
            }
            if matches!(
                key.as_str(),
                "lengthSec" | "playbackRate" | "clipPlaybackRate"
            ) && number <= 0.
            {
                return Err("positive clip length/rate required".into());
            }
            if key == "channelMode" && (number.fract() != 0. || number > 4.) {
                return Err("invalid host channel mode".into());
            }
            if key == "gain" && number > 4. {
                return Err("clip volume outside original GUI range 0..4".into());
            }
            if key == "hostGroupId"
                && (!(0.0..=2_000_000_000.0).contains(&number) || number.fract() != 0.)
            {
                return Err("invalid host item group id".into());
            }
        }
    }
    let patch: ClipStatePatch = serde_json::from_value(input.clone()).map_err(|e| e.to_string())?;
    // 倒放仍然拒绝：ARA 不给反向 PCM，插件渲染不出正确的反向内容（见 `probe/ara/` 的
    // 倒放取证）。循环源**可以写** —— REAPER 的 `B_LOOPSRC` 是纯 item 属性，内核的
    // `loop_enabled` 语义（对整份媒体回绕）与它一致，写回不会让两边不一致。
    if patch.reversed == Some(true) {
        return Err("reverse source editing is not supported".into());
    }
    Ok(patch)
}

impl EditorSession {
    /// UI只短暂读取当前clip身份/几何；源参数大数组不复制，不在这里materialize或分析。
    pub(crate) fn plan_host_edit(
        &self,
        command: &str,
        input: &Value,
    ) -> Result<HostEditPlan, String> {
        let requests = match command {
            "move_clip" | "set_clip_state" => vec![input.clone()],
            "move_clips" => input["moves"]
                .as_array()
                .ok_or("move list missing")?
                .clone(),
            "set_clips_state_bulk" => input["updates"]
                .as_array()
                .ok_or("patch list missing")?
                .clone(),
            _ => return Err("unknown host clip command".into()),
        };
        if requests.is_empty() || requests.len() > 512 {
            return Err("host edit batch must contain 1..512 clips".into());
        }
        let timeline = self.timeline.lock().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut edits = Vec::new();
        for request in requests {
            let id = request["clipId"].as_str().ok_or("clipId missing")?;
            if !seen.insert(id.to_owned()) {
                return Err("duplicate clip in host edit batch".into());
            }
            let before = timeline
                .clips
                .iter()
                .find(|clip| clip.id == id)
                .cloned()
                .ok_or("unknown editor clip")?;
            let destination = request["trackId"].as_str().unwrap_or(&before.track_id);
            if !timeline.tracks.iter().any(|track| track.id == destination) {
                return Err("unknown target host track".into());
            }
            let mut changes = request.clone();
            changes
                .as_object_mut()
                .ok_or("clip request missing")?
                .remove("trackId");
            changes.as_object_mut().unwrap().remove("moveLinkedParams");
            let mut patch = patch(&changes)?;
            // 整体宽度/交叉点patch会带未改的shape；不能把它误判为自定义渐变请求。
            if patch.fade_in_shape == Some(before.fade_in_shape) {
                patch.fade_in_shape = None;
            }
            if patch.fade_out_shape == Some(before.fade_out_shape) {
                patch.fade_out_shape = None;
            }
            if patch.fade_in_dir == Some(before.fade_in_dir) {
                patch.fade_in_dir = None;
            }
            if patch.fade_out_dir == Some(before.fade_out_dir) {
                patch.fade_out_dir = None;
            }
            // 复用原Clip级乘数/Take有效倍率和裁切语义，而不是把clipPlaybackRate当take速率。
            let mut view = TimelineState {
                clips: vec![before.clone()],
                ..Default::default()
            };
            view.patch_clip_state(id, patch.clone());
            let mut after = view.clips.remove(0);
            if let Some(end) = patch.source_end_sec {
                if patch.length_sec.is_none() {
                    after.length_sec =
                        (end - after.source_start_sec) / (after.playback_rate as f64);
                    if after.length_sec <= 0. {
                        return Err("source trim window must have positive length".into());
                    }
                }
            }
            let native_id = id
                .strip_prefix(&self.namespace)
                .ok_or("clip belongs to another editor session")?
                .to_owned();
            let destination_track = destination
                .strip_prefix(&self.namespace)
                .ok_or("target track belongs to another editor session")?
                .to_owned();
            let source_track = before
                .track_id
                .strip_prefix(&self.namespace)
                .ok_or("source track belongs to another editor session")?
                .to_owned();
            edits.push(ClipEdit {
                native_id,
                source_track,
                destination_track,
                before,
                after,
                patch,
            });
        }
        Ok(HostEditPlan { edits })
    }
}

/// 先冻结并验证整批对象，再开始Undo和setter；任一预检失败都不写入任何宿主item。
// 保留无undo上下文的一次性执行入口，当前调用方统一走execute_managed。
#[allow(dead_code)]
pub(crate) fn execute(
    owner: &Arc<ExtensionOwner>,
    plan: HostEditPlan,
    authorized: impl Fn() -> bool,
) -> Result<(), String> {
    execute_managed(owner, plan, authorized, false)
}
pub(super) fn execute_managed(
    owner: &Arc<ExtensionOwner>,
    plan: HostEditPlan,
    authorized: impl Fn() -> bool,
    managed: bool,
) -> Result<(), String> {
    let document = owner.editor_document()?;
    let mut targets: Vec<(ClipEdit, HostClipTarget, Option<HostClipTarget>)> = Vec::new();
    let mut styles = std::collections::BTreeMap::new();
    for edit in plan.edits {
        if !authorized() {
            return Err("host editor lease revoked".into());
        }
        let target = owner.host_edit_target(&edit.native_id)?;
        if !target.geometry.markers.is_empty()
            && (edit.patch.length_sec.is_some()
                || edit.patch.source_start_sec.is_some()
                || edit.patch.source_end_sec.is_some()
                || edit.patch.playback_rate.is_some()
                || edit.patch.clip_playback_rate.is_some())
        {
            return Err("nonlinear host stretch-marker editing is not supported".into());
        }
        if (target.geometry.start_sec - edit.before.start_sec).abs() > 1e-6
            || (target.geometry.duration_sec - edit.before.length_sec).abs() > 1e-6
        {
            return Err("host clip changed before GUI commit; refresh required".into());
        }
        if (edit.patch.source_start_sec.is_some()
            || edit.patch.source_end_sec.is_some()
            || edit.patch.length_sec.is_some()
            || edit.patch.playback_rate.is_some()
            || edit.patch.clip_playback_rate.is_some())
            && ((target.geometry.source_start_sec - edit.before.source_start_sec).abs() > 1e-6
                || (target.geometry.playback_rate - edit.before.playback_rate as f64).abs() > 1e-6)
        {
            return Err(
                "host source window/rate changed before GUI commit; refresh required".into(),
            );
        }
        let destination = if edit.source_track == edit.destination_track {
            None
        } else {
            let clip = {
                let timeline = document.timeline.lock().unwrap();
                timeline
                    .as_ref()
                    .ok_or("host timeline unavailable")?
                    .clips
                    .iter()
                    .find(|clip| clip.track_id == edit.destination_track)
                    .map(|clip| clip.id.clone())
                    .ok_or("target track has no directly bound ARA item")?
            };
            Some(owner.host_edit_target(&clip)?)
        };
        if targets
            .first()
            .is_some_and(|(_, first, _)| !first.same_project(&target))
        {
            return Err("host batch spans multiple projects".into());
        }
        if destination
            .as_ref()
            .is_some_and(|dest| !target.same_project(dest))
        {
            return Err("target track belongs to another project".into());
        }
        // ── 淡变形状 / 曲率 / S 参数 ────────────────────────────────────────
        //
        // 【落到哪个宿主字段取决于宿主版本】官方头文件把两套轴标成互补区间
        // （`sdk/reaper_plugin_functions.h`）：
        //   `C_FADE*SHAPE` / `D_FADE*DIR`      —— v7.80 and earlier
        //   `D_FADE*DIR_NEW` / `D_FADE*DIR2_NEW` —— v7.81 and later
        // 所以这里**必须**按 `fade_axes_new` 分流，而不是按模式或猜。
        if requests_fade_axes(&edit.patch) {
            // 委托（宿主把边界包络交给 HFS 渲染）时才存 HFS 自有样式；
            // 否则直接写宿主自己的轴。此前这里**无条件**要求委托 ——
            // 而 REAPER 从不委托，于是整条形状编辑路径是死的。
            let delegated = {
                let key = {
                    let ids = document.clip_ids.lock().unwrap();
                    ids.iter()
                        .find(|(_, id)| id.as_str() == edit.native_id)
                        .map(|(key, _)| *key)
                        .ok_or("fade region identity missing")?
                };
                document
                    .regions
                    .lock()
                    .unwrap()
                    .get(&key)
                    .is_some_and(|region| {
                        region.has_content_based_fade_at_head
                            || region.has_content_based_fade_at_tail
                    })
            };
            if delegated {
                // 【为什么委托时不校验轴语义】委托时形状由 **HFS 自己**渲染
                // （`FadeStyle` 进插件状态），宿主轴根本不参与 —— 拿宿主版本去
                // 拦它会把一条本来能用的路径堵死。但 S 参数在 `FadeStyle` 里没有
                // 位置，所以仍然拒绝，而不是静默丢弃。
                if edit.patch.fade_in_s.is_some() || edit.patch.fade_out_s.is_some() {
                    return Err("fade S parameter is not part of the HiFiShifter fade style".into());
                }
                let mut style = document
                    .edits
                    .lock()
                    .unwrap()
                    .fades
                    .get(&target.geometry.item_id)
                    .cloned()
                    .unwrap_or_default();
                if let Some(value) = edit.patch.fade_in_shape {
                    style.in_shape = value;
                }
                if let Some(value) = edit.patch.fade_out_shape {
                    style.out_shape = value;
                }
                if let Some(value) = edit.patch.fade_in_dir {
                    style.in_dir = value;
                }
                if let Some(value) = edit.patch.fade_out_dir {
                    style.out_dir = value;
                }
                style.validate()?;
                styles.insert(target.geometry.item_id.clone(), style);
            } else {
                // 直接写宿主轴：这里才需要轴语义校验（版本未知/新轴不接受预设）。
                validate_fade_axes(target.geometry.fade_axes_new, &edit.patch)?;
            }
        }
        targets.push((edit, target, destination));
    }
    let undo = if managed {
        None
    } else {
        Some(targets[0].1.begin_undo(&authorized)?)
    };
    let result: Result<(), String> = (|| {
        for (edit, target, destination) in &targets {
            let before = &edit.before;
            let after = &edit.after;
            // 写速率时明确保调；普通移动/裁切不擅自改变用户B_PPITCH。
            if (before.playback_rate - after.playback_rate).abs() > 1e-6 {
                target.set_take(c"B_PPITCH", 1., &authorized)?;
                target.set_take(c"D_PLAYRATE", after.playback_rate as f64, &authorized)?;
            }
            if (before.source_start_sec - after.source_start_sec).abs() > 1e-9 {
                target.set_take(c"D_STARTOFFS", after.source_start_sec, &authorized)?;
            }
            if (before.start_sec - after.start_sec).abs() > 1e-9 {
                target.set_item(c"D_POSITION", after.start_sec, &authorized)?;
            }
            if (before.length_sec - after.length_sec).abs() > 1e-9 {
                target.set_item(c"D_LENGTH", after.length_sec, &authorized)?;
            }
            if let Some(value) = edit.patch.muted {
                target.set_item(c"B_MUTE", if value { 1. } else { 0. }, &authorized)?;
            }
            if let Some(value) = edit.patch.gain {
                target.set_item(c"D_VOL", value as f64, &authorized)?;
            }
            if let Some(enabled) = edit.patch.loop_enabled {
                // 双向：`B_LOOPSRC` 是 item 属性，读回来的就是它（见 `HostClipGeometry`）。
                target.set_item(c"B_LOOPSRC", if enabled { 1. } else { 0. }, &authorized)?;
            }
            if let Some(value) = edit.patch.channel_mode {
                if !(0..=4).contains(&value) {
                    return Err("invalid host channel mode".into());
                }
                target.set_take(c"I_CHANMODE", value as f64, &authorized)?;
            }
            if let Some(value) = edit.patch.host_group_id {
                if !(0..=2_000_000_000).contains(&value) {
                    return Err("invalid host item group id".into());
                }
                target.set_item(c"I_GROUPID", value as f64, &authorized)?;
            }
            if let Some(value) = edit
                .patch
                .name
                .as_deref()
                .filter(|value| *value != before.name)
            {
                target.set_take_name(value, &authorized)?;
            }
            if let Some(value) = edit.patch.snap_offset_sec {
                target.set_item(c"D_SNAPOFFSET", value, &authorized)?;
            }
            for (value, name) in [
                (edit.patch.fade_in_sec, c"D_FADEINLEN"),
                (edit.patch.fade_out_sec, c"D_FADEOUTLEN"),
                (edit.patch.auto_fade_in_sec, c"D_FADEINLEN_AUTO"),
                (edit.patch.auto_fade_out_sec, c"D_FADEOUTLEN_AUTO"),
            ] {
                if let Some(value) = value {
                    target.set_item(name, value, &authorized)?;
                }
            }
            if let Some(destination) = destination {
                target.move_to(destination, &authorized)?;
            }
            // 【为什么分两种情形】委托时（宿主把边界包络交给 HFS 渲染）宿主只留默认
            // 样式，HFS 的 shape/dir 进插件状态；**不委托时直接写宿主自己的轴** ——
            // 那才是让用户"能调淡变类型和曲率"的那条路。
            if styles.contains_key(&target.geometry.item_id) {
                // REAPER只留默认样式；自定义shape/dir进入HFS状态，而不是宿主c/S轴。
                match target.geometry.fade_axes_new {
                    Some(true) => {
                        for (name, value) in [
                            (c"D_FADEINDIR_NEW", target.geometry.fade_in_dir_new),
                            (c"D_FADEOUTDIR_NEW", target.geometry.fade_out_dir_new),
                            (c"D_FADEINDIR2_NEW", target.geometry.fade_in_dir2_new),
                            (c"D_FADEOUTDIR2_NEW", target.geometry.fade_out_dir2_new),
                        ] {
                            if value != 0. {
                                target.set_item(name, 0., &authorized)?;
                            }
                        }
                    }
                    Some(false) => {
                        for (name, value) in [
                            (c"C_FADEINSHAPE", target.geometry.fade_in_shape),
                            (c"C_FADEOUTSHAPE", target.geometry.fade_out_shape),
                            (c"D_FADEINDIR", target.geometry.fade_in_dir),
                            (c"D_FADEOUTDIR", target.geometry.fade_out_dir),
                        ] {
                            if value != 0. {
                                target.set_item(name, 0., &authorized)?;
                            }
                        }
                    }
                    None => {
                        return Err(
                            "host fade version unavailable; cannot normalize default style".into(),
                        )
                    }
                }
            } else if let Some(axes_new) = target.geometry.fade_axes_new {
                // 直接写宿主轴。`fade_axes_new` 已在上面的校验里排除 `None`，
                // 这里再取一次是为了拿到值（校验阶段只做了拒绝）。
                if axes_new {
                    // 【形状预设要写成**一对**分量】≥7.81 没有"形状号"这个可写状态，
                    // 形状由 `(curvature, S)` 决定；七个预设各自对应一组确定的
                    // 坐标（实测表见 `hifishifter_kernel::fade_axes`）。所以写形状
                    // 就是同时写两个分量。
                    //
                    // 【同一次补丁里形状压过 dir】界面切换预设时会**同时**送
                    // `fadeInShape` 与该形状的默认 `fadeInDir`（HFS 自己的曲率轴，
                    // `DEFAULT_FADE_DIR_BY_SHAPE`）。但两套轴不是同一套参数化 ——
                    // 实测旧 `D_FADEINDIR` 与新 `D_FADEINDIR_NEW` 互相读回都是被重
                    // 映射过的值（`probe/ara/FADE-AXIS-FINDINGS.md`）。若让 HFS 的 dir
                    // 覆盖预设的 curvature，用户点了"轻微凸"会得到线性
                    // （预设 1 = `c 0.5`，而 HFS 给该形状的默认 dir 是 0）。
                    // 曲率滑杆走的是**只带 dir、不带 shape** 的补丁，不受此规则影响。
                    for (shape, dir, s, dir_key, s_key) in [
                        (
                            edit.patch.fade_in_shape,
                            edit.patch.fade_in_dir,
                            edit.patch.fade_in_s,
                            c"D_FADEINDIR_NEW",
                            c"D_FADEINDIR2_NEW",
                        ),
                        (
                            edit.patch.fade_out_shape,
                            edit.patch.fade_out_dir,
                            edit.patch.fade_out_s,
                            c"D_FADEOUTDIR_NEW",
                            c"D_FADEOUTDIR2_NEW",
                        ),
                    ] {
                        if let Some(shape) = shape {
                            let (curvature, s_value) =
                                hifishifter_kernel::fade_axes::host_fade_preset_axes(shape)
                                    .ok_or_else(|| format!("unknown fade shape {shape}"))?;
                            target.set_item(dir_key, curvature, &authorized)?;
                            target.set_item(s_key, s_value, &authorized)?;
                            continue;
                        }
                        if let Some(value) = dir {
                            target.set_item(dir_key, value, &authorized)?;
                        }
                        if let Some(value) = s {
                            target.set_item(s_key, value, &authorized)?;
                        }
                    }
                } else {
                    for (value, name) in [
                        (edit.patch.fade_in_shape, c"C_FADEINSHAPE"),
                        (edit.patch.fade_out_shape, c"C_FADEOUTSHAPE"),
                        (edit.patch.fade_in_dir, c"D_FADEINDIR"),
                        (edit.patch.fade_out_dir, c"D_FADEOUTDIR"),
                    ] {
                        if let Some(value) = value {
                            target.set_item(name, value, &authorized)?;
                        }
                    }
                }
            }
            target.update(&authorized)?;
        }
        Ok(())
    })();
    if result.is_ok() && !styles.is_empty() {
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut edits = document.edits.lock().unwrap();
            edits.fades.extend(styles);
            edits.revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
        }
        for owner in document.renderer_owners() {
            owner.prepare();
        }
    }
    drop(undo);
    result.map_err(|error| format!("host clip edit failed (Undo block retained): {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    /// clip音量写真实item D_VOL，支持零；不能写take音量/极性或靠本地gain冒充成功。
    #[test]
    fn host_clip_gain_writes_item_volume_and_preserves_take_polarity() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        host.set_take_volume(-0.25);
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        for gain in [0.5, 0.] {
            let plan = editor
                .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"gain":gain}))
                .unwrap();
            let receipt = plan.receipt(&editor.namespace);
            host.reset();
            execute(&owner, plan, || document.is_alive()).unwrap();
            let native = clip.id.strip_prefix(&editor.namespace).unwrap();
            let g = owner.host_edit_target(native).unwrap().geometry;
            assert_eq!(g.item_gain, gain);
            assert_eq!(g.take_gain, -0.25);
            assert!(host.calls().contains(&"write-item:D_VOL".into()));
            assert!(!host.calls().iter().any(|call| call == "write-take:D_VOL"));
            let payload = json!({"clips":[{"id":clip.id,"start_sec":clip.start_sec,"length_sec":clip.length_sec,
                "source_start_sec":clip.source_start_sec,"playback_rate":clip.playback_rate,"track_id":clip.track_id,"gain":gain}]});
            assert!(receipt.matches(&payload));
            let mut stale = payload;
            stale["clips"][0]["gain"] = json!(1.);
            assert!(!receipt.matches(&stale));
        }
        assert!(editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"gain":-1.}))
            .is_err());
        assert!(editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"gain":5.}))
            .is_err());
        document.close();
    }
    /// 规划复用原Clip×Take倍率，且未开始宿主写之前整批拒绝非法/未知字段。
    #[test]
    fn host_edit_planning_preserves_effective_rate_and_rejects_unknown_batch_fields() {
        let (_model, owner, _id) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"lengthSec":1.,"clipPlaybackRate":2.}),
            )
            .unwrap();
        assert_eq!(plan.edits[0].after.playback_rate, 2.);
        assert_eq!(plan.edits[0].after.length_sec, 1.);
        assert!(editor
            .plan_host_edit("set_clip_state", &json!({"clipId":clip.id,"reversed":true}))
            .is_err());
        // 循环源**可以**规划：`B_LOOPSRC` 是纯 item 属性，内核语义与它一致。
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"loopEnabled":true})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"loopEnabled":false})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"hostGroupId":1234})
            )
            .is_ok());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"name":"renamed active take"})
            )
            .is_ok());
        assert!(editor.plan_host_edit("set_clips_state_bulk",&json!({"updates":[{"clipId":clip.id,"startSec":1.},{"clipId":"other","lengthSec":-1.}]})).is_err());
        assert!(editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":99.})
            )
            .is_err());
        assert_eq!(
            editor.timeline.lock().unwrap().clips[0].start_sec,
            clip.start_sec,
            "纯规划不得修改actor私有几何"
        );
        editor.close();
    }
    /// 使用生产clip路由、官方typed getter/setter与Undo块；没有把私有timeline赋值当同步。
    #[test]
    fn host_edit_writes_directly_bound_item_under_one_project_undo_block() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan=editor.plan_host_edit("set_clip_state",&json!({"clipId":clip.id,"startSec":3.,"lengthSec":8./44100.,"clipPlaybackRate":0.5})).unwrap();
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert_eq!(
            calls
                .iter()
                .filter(|name| name.as_str() == "undo-begin")
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|name| name.as_str() == "undo-end")
                .count(),
            1
        );
        assert!(calls.contains(&"write-item:D_POSITION".into()));
        assert!(calls.contains(&"write-take:D_PLAYRATE".into()));
        let geometry = owner.reaper_geometry().unwrap_err();
        assert!(
            geometry.contains("incompatible"),
            "未模拟ARA回流，不能伪造已同步模型"
        );
        assert_eq!(editor.timeline.lock().unwrap().clips[0].start_sec, 0.);
        document.close();
    }
    /// 旧快照仍ok也不能结束移动；长度/倍率/淡变宽度均须准确回流，且缺失clip不可成功。
    #[test]
    fn host_edit_receipt_rejects_readable_old_geometry_and_waits_for_widths() {
        let (_model, owner, _id) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"startSec":3.,"fadeInSec":0.2}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace);
        let mut payload = super::super::commands::payload(&editor, false).unwrap();
        assert!(!receipt.matches(&payload));
        payload["clips"][0]["start_sec"] = json!(3.);
        assert!(!receipt.matches(&payload));
        payload["clips"][0]["fade_in_sec"] = json!(0.2);
        assert!(receipt.matches(&payload), "{receipt:?} vs {payload}");
        payload["clips"][0]["id"] = json!("other");
        assert!(!receipt.matches(&payload));
        editor.close();
    }
    /// 宽度patch冗余带着原shape不要求委托；实际改shape仍保持明确的责任门。
    #[test]
    fn host_edit_fade_width_writes_reaper_without_delegation_or_redundant_shape() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan=editor.plan_host_edit("set_clip_state",&json!({"clipId":clip.id,"fadeInSec":0.00002,"fadeOutSec":0.00003,
            "autoFadeInSec":0.,"autoFadeOutSec":0.,"fadeInShape":clip.fade_in_shape,"fadeOutShape":clip.fade_out_shape,
            "fadeInDir":clip.fade_in_dir,"fadeOutDir":clip.fade_out_dir})).unwrap();
        assert!(plan.edits[0].patch.fade_in_shape.is_none());
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        for field in [
            "D_FADEINLEN",
            "D_FADEOUTLEN",
            "D_FADEINLEN_AUTO",
            "D_FADEOUTLEN_AUTO",
        ] {
            assert!(calls.contains(&format!("write-item:{field}")));
        }
        assert!(!calls
            .iter()
            .any(|call| call.starts_with("write-item:D_FADEINDIR")
                || call.starts_with("write-item:C_FADE")));
        // 曲率与 S 落到新轴；形状预设现在也能写，落成 (curvature, S) 一对分量。
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInDir":0.5,"fadeInS":-0.25}),
            )
            .unwrap();
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert!(calls.contains(&"write-item:D_FADEINDIR_NEW".to_string()));
        assert!(calls.contains(&"write-item:D_FADEINDIR2_NEW".to_string()));
        assert_eq!(host.item_value("D_FADEINDIR_NEW"), Some(0.5));
        assert_eq!(host.item_value("D_FADEINDIR2_NEW"), Some(-0.25));
        // 旧轴不得被顺手改掉 —— 它不是权威值。
        assert!(!calls.contains(&"write-item:D_FADEINDIR".to_string()));
        assert!(!calls.contains(&"write-item:C_FADEINSHAPE".to_string()));

        // 形状预设：写 `fadeInShape` 就是同时写两个新轴分量，值取自实测表
        // （预设 3「陡峭凸」= `(curvature 1, S 0)`）。同一次补丁里带的 `fadeInDir`
        // 不参与 —— 界面切预设时会顺手送 HFS 自己的默认 dir，那是**另一套参数化**，
        // 让它覆盖 curvature 会让"陡峭凸"变成"陡峭凹"（预设 4 = `(-1, 0)`）。
        host.set_value("D_FADEINDIR_NEW", -0.2);
        host.set_value("D_FADEINDIR2_NEW", -0.3);
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"fadeInShape":3.,"fadeInDir":-1.}),
            )
            .unwrap();
        assert_eq!(plan.edits[0].patch.fade_in_shape, Some(3.));
        host.reset();
        execute(&owner, plan, || document.is_alive()).unwrap();
        let calls = host.calls();
        assert!(calls.contains(&"write-item:D_FADEINDIR_NEW".to_string()));
        assert!(calls.contains(&"write-item:D_FADEINDIR2_NEW".to_string()));
        assert_eq!(host.item_value("D_FADEINDIR_NEW"), Some(1.));
        assert_eq!(host.item_value("D_FADEINDIR2_NEW"), Some(0.));
        assert!(!calls.contains(&"write-item:D_FADEINDIR".to_string()));
        assert!(!calls.contains(&"write-item:C_FADEINSHAPE".to_string()));
        document.close();
    }

    /// 吸附偏移经真实typed setter和UI元数据回流，不能写完后永远等默认零值。
    #[test]
    fn host_edit_snap_offset_receipt_uses_current_host_metadata() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"snapOffsetSec":2./44100.}),
            )
            .unwrap();
        let receipt = plan.receipt(&editor.namespace);
        assert!(!receipt.matches(&super::super::commands::payload(&editor, false).unwrap()));
        execute(&owner, plan, || document.is_alive()).unwrap();
        owner.refresh_reaper_transport();
        assert_eq!(
            owner.host_geometry_metadata().unwrap()[0]
                .geometry
                .snap_offset_sec,
            2. / 44100.
        );
        let payload = super::super::commands::payload(&editor, false).unwrap();
        assert!(receipt.matches(&payload));
        // 官方只说明秒域，不因用户已有负偏移使整个host几何失效；GUI自行按原约定绘制。
        host.set_value("D_SNAPOFFSET", -1. / 44100.);
        owner.refresh_reaper_transport();
        assert_eq!(
            owner.host_geometry_metadata().unwrap()[0]
                .geometry
                .snap_offset_sec,
            -1. / 44100.
        );
        document.close();
    }
    /// 起点/时长未变但宿主已经改源窗口时，裁切预检必须在Undo/写API之前拒绝。
    #[test]
    fn host_edit_source_trim_rejects_stale_source_offset_before_any_write() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_writer();
        host.clear_markers();
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan = editor
            .plan_host_edit(
                "set_clip_state",
                &json!({"clipId":clip.id,"sourceStartSec":1./44100.}),
            )
            .unwrap();
        host.set_value("D_STARTOFFS", 2. / 44100.);
        host.reset();
        assert!(execute(&owner, plan, || document.is_alive())
            .unwrap_err()
            .contains("source window/rate changed"));
        assert!(!host
            .calls()
            .iter()
            .any(|call| call == "undo-begin" || call.starts_with("write-")));
        document.close();
    }
}

#[cfg(test)]
mod fade_axis_validation_tests {
    use super::*;

    fn shape(shape: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_shape: Some(shape),
            ..Default::default()
        }
    }
    fn curvature(dir: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_dir: Some(dir),
            ..Default::default()
        }
    }
    fn s_param(value: f64) -> ClipStatePatch {
        ClipStatePatch {
            fade_in_s: Some(value),
            ..Default::default()
        }
    }

    /// 不含淡变轴字段的补丁永远合法 —— 长度/位置编辑不该被轴语义拦住。
    #[test]
    fn patches_without_fade_axes_are_always_accepted() {
        let patch = ClipStatePatch {
            length_sec: Some(1.),
            ..Default::default()
        };
        for axes in [None, Some(true), Some(false)] {
            assert!(validate_fade_axes(axes, &patch).is_ok(), "{axes:?}");
        }
    }

    /// 版本读不出来时一律拒绝：不猜哪套轴是权威的。
    #[test]
    fn an_unknown_axis_set_rejects_every_fade_axis_write() {
        for patch in [shape(3.), curvature(0.5), s_param(0.2)] {
            assert!(validate_fade_axes(None, &patch).is_err());
        }
    }

    /// 旧轴宿主（≤7.80）：形状与曲率可以写，S 参数不行（它只存在于新轴）。
    #[test]
    fn legacy_hosts_take_the_shape_and_curvature_but_not_the_s_parameter() {
        assert!(validate_fade_axes(Some(false), &shape(3.)).is_ok());
        assert!(validate_fade_axes(Some(false), &curvature(0.5)).is_ok());
        let error = validate_fade_axes(Some(false), &s_param(0.2)).expect_err("S 必须被拒绝");
        assert!(error.contains("7.81"), "{error}");
    }

    /// 新轴宿主（≥7.81）：曲率、S、以及**形状预设**都可以写。
    ///
    /// 形状此前被拒绝，理由是"预设 → (curvature, S) 的映射尚未校准"。该映射现已实测
    /// 落地（`hifishifter_kernel::fade_axes`），所以预设按钮在新轴宿主上也能用。
    #[test]
    fn continuous_axis_hosts_take_curvature_s_and_shape_presets() {
        assert!(validate_fade_axes(Some(true), &curvature(0.5)).is_ok());
        assert!(validate_fade_axes(Some(true), &s_param(-0.25)).is_ok());
        for preset in 0..7 {
            assert!(
                validate_fade_axes(Some(true), &shape(preset as f64)).is_ok(),
                "preset {preset}"
            );
        }
    }

    /// 表外的形状号必须拒绝。宿主对越界值不会报错，而是静默变成别的形状 ——
    /// 实测写 `5.1` 读回 `SHAPE=7`（见 `probe/ara/FADE-AXIS-FINDINGS.md`）。
    /// 宁可报错，也不要用户点了 A 得到 B。
    #[test]
    fn continuous_axis_hosts_reject_shapes_outside_the_measured_table() {
        for value in [-1.0, 7.0, 1.1, 5.1, 6.5] {
            let error =
                validate_fade_axes(Some(true), &shape(value)).expect_err("表外形状必须被拒绝");
            assert!(error.contains("unknown fade shape"), "{error}");
        }
    }

    /// 旧轴宿主上形状仍是整数预设号，不走那张表 —— 旧轴宿主本来就有形状号。
    #[test]
    fn legacy_hosts_keep_accepting_plain_shape_indices() {
        assert!(validate_fade_axes(Some(false), &shape(6.)).is_ok());
    }

    /// 任一淡变轴字段都会触发校验（不能只检查 `fade_in_*`）。
    #[test]
    fn every_fade_axis_field_triggers_validation() {
        for patch in [
            ClipStatePatch {
                fade_out_shape: Some(4.),
                ..Default::default()
            },
            ClipStatePatch {
                fade_out_dir: Some(0.1),
                ..Default::default()
            },
            ClipStatePatch {
                fade_out_s: Some(0.1),
                ..Default::default()
            },
        ] {
            assert!(validate_fade_axes(None, &patch).is_err(), "{patch:?}");
        }
    }
}
