//! 原GUI分割命令只写所属REAPER item；整批预检、宿主Undo、真实两段回流共同决定成功。
use super::session::EditorSession;
use crate::{
    host::{geometry::HostClipGeometry, reaper::HostClipTarget},
    render::{document::DocumentSession, extension::ExtensionOwner},
};
use hifishifter_kernel::state::Clip;
use serde_json::{json, Value};
use std::sync::Arc;

pub(crate) struct SplitPlan {
    clips: Vec<(String, Clip)>,
    position: f64,
}
pub(crate) struct SplitReceipt {
    pairs: Vec<(HostClipGeometry, HostClipGeometry)>,
}

impl EditorSession {
    /// 使用当前GUI身份规划；不在私有timeline中生成与宿主无关的UUID。
    pub(crate) fn plan_host_split(
        &self,
        command: &str,
        input: &Value,
    ) -> Result<SplitPlan, String> {
        let position = input["splitSec"]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("finite splitSec required")?;
        let ids = if command == "split_clip" {
            vec![input["clipId"].as_str().ok_or("clipId missing")?.to_owned()]
        } else {
            input["clipIds"]
                .as_array()
                .ok_or("clipIds missing")?
                .iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or("invalid clipId"))
                .collect::<Result<Vec<_>, _>>()?
        };
        if ids.is_empty() || ids.len() > 512 {
            return Err("split batch must contain 1..512 clips".into());
        }
        let timeline = self.timeline.lock().unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut clips = Vec::new();
        for id in ids {
            if !seen.insert(id.clone()) {
                return Err("duplicate clip in split batch".into());
            }
            let clip = timeline
                .clips
                .iter()
                .find(|clip| clip.id == id)
                .ok_or("unknown GUI clip")?;
            if position <= clip.start_sec + 1e-6
                || position >= clip.start_sec + clip.length_sec - 1e-6
            {
                return Err("split position must be inside every selected clip".into());
            }
            let native = id
                .strip_prefix(&self.namespace)
                .ok_or("clip belongs to another editor session")?
                .to_owned();
            clips.push((native, clip.clone()));
        }
        Ok(SplitPlan { clips, position })
    }
}

/// 整批当前源窗口/倍率均吻合才开始分割；失败保留真实宿主Undo而不伪造成功。
pub(crate) fn execute(
    owner: &Arc<ExtensionOwner>,
    plan: SplitPlan,
    authorized: &impl Fn() -> bool,
) -> Result<SplitReceipt, String> {
    let document = owner.editor_document()?;
    if !owner
        .project_history_host()
        .is_some_and(|host| host.can_split_clips())
    {
        return Err("host split capability unavailable".into());
    }
    let mut targets: Vec<HostClipTarget> = Vec::new();
    for (native, before) in &plan.clips {
        if !authorized() {
            return Err("host split lease revoked".into());
        }
        let target = owner.host_edit_target(native)?.current(authorized)?;
        let g = &target.geometry;
        if !g.markers.is_empty() {
            return Err("nonlinear host split is not supported".into());
        }
        if [
            (g.start_sec, before.start_sec),
            (g.duration_sec, before.length_sec),
            (g.source_start_sec, before.source_start_sec),
            (g.playback_rate, before.playback_rate as f64),
        ]
        .iter()
        .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            return Err("host clip changed before split; refresh required".into());
        }
        if targets
            .first()
            .is_some_and(|first| !first.same_project(&target))
        {
            return Err("split batch spans multiple projects".into());
        }
        if targets
            .iter()
            .any(|other| other.geometry.item_id == g.item_id)
        {
            return Err("duplicate host item in split batch".into());
        }
        targets.push(target);
    }
    let mut pairs = Vec::new();
    for target in targets {
        let seed = {
            let _transaction = document.transaction.lock().unwrap();
            document
                .edits
                .lock()
                .unwrap()
                .atlas
                .split_seed(&target.geometry.item_id)
        };
        let pair = target.split_at(plan.position, authorized)?;
        // 确切API回执建立右段→父item关系，重叠同源片段不走模糊祖先匹配。
        {
            let _transaction = document.transaction.lock().unwrap();
            let mut edits = document.edits.lock().unwrap();
            edits
                .atlas
                .register_split(&pair.0.item_id, &pair.1.item_id, seed)?;
            edits.revision = edits
                .revision
                .checked_add(1)
                .ok_or("edit revision exhausted")?;
        }
        pairs.push(pair);
    }
    owner.refresh_reaper_transport();
    for owner in document.renderer_owners() {
        owner.prepare();
    }
    Ok(SplitReceipt { pairs })
}

impl SplitReceipt {
    /// 左右两段均按真实GUID解析，并确认源窗口/长度回流后，才填原GUI的选中右段回执。
    pub(crate) fn matches(
        &self,
        document: &DocumentSession,
        namespace: &str,
        payload: &mut Value,
    ) -> bool {
        let mut created = Vec::new();
        for (left, right) in &self.pairs {
            for (is_right, g) in [(false, left), (true, right)] {
                let Some(native) = document.gui_clip_for_host_item(&g.item_id) else {
                    return false;
                };
                let id = format!("{namespace}{native}");
                let Some(clip) = payload["clips"]
                    .as_array()
                    .and_then(|clips| clips.iter().find(|clip| clip["id"] == id))
                else {
                    return false;
                };
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
                if is_right {
                    created.push(id);
                }
            }
        }
        payload["created_clip_ids"] = json!(created);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 边界、重复、外会话与旧源窗口均在写入前失败，规划不能修改私有模型。
    #[test]
    fn host_split_planning_rejects_invalid_batches_without_mutation() {
        let (_model, owner, _id) = super::super::session::tests::fixture();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let mid = clip.start_sec + clip.length_sec / 2.;
        assert!(editor
            .plan_host_split("split_clip", &json!({"clipId":clip.id,"splitSec":mid}))
            .is_ok());
        for position in [clip.start_sec, clip.start_sec + clip.length_sec] {
            assert!(editor
                .plan_host_split("split_clip", &json!({"clipId":clip.id,"splitSec":position}))
                .is_err());
        }
        assert!(editor
            .plan_host_split(
                "split_clips_at",
                &json!({"clipIds":[clip.id,clip.id],"splitSec":mid})
            )
            .is_err());
        assert_eq!(editor.timeline.lock().unwrap().clips.len(), 1);
        editor.close();
    }
    /// 真正经过raw SplitMediaItem绑定取得新GUID；源起点按倍率推进，不改私有timeline冒充回流。
    #[test]
    fn host_split_native_returns_distinct_item_guids_and_keeps_source_window() {
        let (model, owner, _id) = super::super::session::tests::fixture();
        let document = model.session();
        let host = crate::host::reaper::ReaperFixture::new();
        host.enable_split();
        host.clear_markers();
        host.set_value("D_POSITION", 0.);
        host.set_value("D_LENGTH", 4. / 44100.);
        host.set_value("D_PLAYRATE", 1.);
        host.set_value("D_STARTOFFS", 0.);
        unsafe {
            owner.bind_reaper_host(host.context());
        }
        owner.refresh_reaper_transport();
        let editor = owner.editor_session().unwrap();
        editor.ensure_loaded(false).unwrap();
        let clip = editor.timeline.lock().unwrap().clips[0].clone();
        let plan = editor
            .plan_host_split(
                "split_clip",
                &json!({"clipId":clip.id,"splitSec":2./44100.}),
            )
            .unwrap();
        host.reset();
        let receipt = execute(&owner, plan, &|| document.is_alive()).unwrap();
        let (left, right) = &receipt.pairs[0];
        assert_ne!(left.item_id, right.item_id);
        assert!((left.duration_sec - 2. / 44100.).abs() < 1e-10);
        assert!((right.source_start_sec - 2. / 44100.).abs() < 1e-10);
        assert_eq!(
            host.calls()
                .iter()
                .filter(|name| *name == "split-item")
                .count(),
            1
        );
        assert_eq!(
            editor.timeline.lock().unwrap().clips.len(),
            1,
            "fixture未模拟ARA创建，不能假装已看见两段"
        );
        document.close();
    }
}
