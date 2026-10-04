//! 插件侧编辑状态；客户端只能提交参数，宿主的clip几何始终权威。
use hifishifter_kernel::state::{TimelineState, Track, TrackParamsState};
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

#[derive(Default, Clone, Serialize, Deserialize)]
pub(crate) struct EditState {
    pub revision: u64,
    pub params: BTreeMap<String, TrackParamsState>,
    pub tracks: Vec<Track>,
}

impl EditState {
    /// 只复制用户曲线与已有轨道参数，不接受客户端移动/替换clip。
    pub fn merge(&self, host: &TimelineState, client: &TimelineState, base_revision: u64) -> Result<Self, String> {
        if base_revision != self.revision { return Err("Conflict: edit revision changed".into()); }
        if client.tracks.len() != host.tracks.len() || client.tracks.iter().any(|track|
            !host.tracks.iter().any(|known| known.id == track.id) || !track.volume.is_finite() || !(0.0..=4.0).contains(&track.volume)) {
            return Err("invalid or unknown track".into());
        }
        for (id, params) in &client.params_by_root_track {
            if !host.tracks.iter().any(|track| track.id == *id) || !params.frame_period_ms.is_finite()
                || !(0.1..=100.0).contains(&params.frame_period_ms) {
                return Err("invalid curve track or frame period".into());
            }
            for curve in [&params.pitch_orig, &params.pitch_edit, &params.tension_orig, &params.tension_edit]
                .into_iter().chain(params.extra_curves.values()) {
                if curve.len() > 1_000_000 || curve.iter().any(|value| !value.is_finite() || value.abs() > 10000.0) {
                    return Err("invalid parameter curve".into());
                }
            }
            if params.extra_params.values().any(|value| !value.is_finite()) { return Err("invalid static parameter".into()); }
        }
        Ok(Self { revision: self.revision.checked_add(1).ok_or("revision exhausted")?,
            params: client.params_by_root_track.clone(), tracks: client.tracks.clone() })
    }
    /// 当前宿主几何上叠加持久化参数，旧轨道记录不会创造虚构轨道。
    pub fn apply(&self, timeline: &mut TimelineState) {
        timeline.params_by_root_track = self.params.iter().filter(|(id, _)| timeline.tracks.iter().any(|t| t.id == **id))
            .map(|(id, p)| (id.clone(), p.clone())).collect();
        for track in &mut timeline.tracks {
            if let Some(edited) = self.tracks.iter().find(|t| t.id == track.id) {
                track.compose_enabled = edited.compose_enabled;
                track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
                track.volume = edited.volume;
                track.muted = edited.muted;
                track.solo = edited.solo;
            }
        }
    }

    /// 有版本且有界的组件state；旧空state保持默认编辑。
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&serde_json::json!({"version":1,"edits":self})).map_err(|e| e.to_string())
    }
    /// 恢复使乐观并发revision前进，防止旧GUI再次覆盖宿主undo/恢复。
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() { return Ok(()); }
        if bytes.len() > hifishifter_ara_ipc::MAX_FRAME { return Err("state too large".into()); }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if value["version"] != 1 { return Err("unsupported state version".into()); }
        let mut restored: Self = serde_json::from_value(value["edits"].clone()).map_err(|e| e.to_string())?;
        restored.revision = self.revision.max(restored.revision).checked_add(1).ok_or("revision exhausted")?;
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn host() -> TimelineState {
        serde_json::from_value(serde_json::json!({"tracks":[{"id":"track","name":"host","order":0}],"clips":[],"bpm":120,"project_sec":1})).unwrap()
    }
    #[test]
    fn edits_preserve_host_geometry_and_reject_stale_writer() {
        let mut client = host();
        client.tracks[0].volume = 0.5;
        client.project_sec = 400.;
        let state = EditState::default().merge(&host(), &client, 0).unwrap();
        let mut rendered = host();
        state.apply(&mut rendered);
        assert_eq!(rendered.tracks[0].volume, 0.5);
        assert_eq!(rendered.project_sec, 1.);
        assert!(state.merge(&host(), &client, 0).is_err());
    }
    #[test]
    fn invalid_curve_and_unknown_track_are_rejected() {
        let mut client = host();
        client.params_by_root_track.insert("track".into(), TrackParamsState { pitch_edit: vec![f32::NAN], ..Default::default() });
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
        client.params_by_root_track.clear();
        client.tracks[0].id = "foreign".into();
        assert!(EditState::default().merge(&host(), &client, 0).is_err());
    }
}
