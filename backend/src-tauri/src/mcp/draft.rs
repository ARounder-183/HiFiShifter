//! Validate on an isolated timeline, then commit with compare-and-swap under the live timeline lock.
use hifishifter_kernel::{
    editor::{params, ParamHost},
    state::*,
};
use hifishifter_mcp::{CurveTransform, Operation};
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path, sync::Mutex};

#[derive(Debug)]
pub(super) struct Failure {
    pub code: &'static str,
    pub message: String,
}
impl Failure {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn value(&self) -> Value {
        json!({"ok":false,"code":self.code,"message":self.message})
    }
}
type Result<T> = std::result::Result<T, Failure>;
fn invalid(message: impl Into<String>) -> Failure {
    Failure::new("INVALID_ARGUMENT", message)
}
pub(super) fn finite(value: f64, min: f64, max: f64, field: &str) -> Result<()> {
    if value.is_finite() && value >= min && value <= max {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} must be finite and in [{min}, {max}]"
        )))
    }
}

pub(super) struct Draft {
    pub timeline: Mutex<TimelineState>,
}
impl ParamHost for Draft {
    fn timeline(&self) -> &Mutex<TimelineState> {
        &self.timeline
    }
    fn checkpoint_timeline(&self, _: &TimelineState, _: HistoryOp) {}
    fn mark_dirty(&self) {}
    fn publish_timeline(&self, _: TimelineState) {}
}
impl Draft {
    pub fn new(timeline: TimelineState) -> Self {
        Self {
            timeline: Mutex::new(timeline),
        }
    }
}
fn lock(draft: &Draft) -> std::sync::MutexGuard<'_, TimelineState> {
    draft.timeline.lock().unwrap_or_else(|e| e.into_inner())
}
fn track(tl: &TimelineState, id: &str) -> Result<()> {
    if tl.tracks.iter().any(|t| t.id == id) {
        Ok(())
    } else {
        Err(Failure::new("TRACK_NOT_FOUND", id))
    }
}
fn clip(tl: &TimelineState, id: &str) -> Result<()> {
    if tl.clips.iter().any(|c| c.id == id) {
        Ok(())
    } else {
        Err(Failure::new("CLIP_NOT_FOUND", id))
    }
}
fn resolve(ids: &HashMap<String, String>, id: &str) -> String {
    ids.get(id).cloned().unwrap_or_else(|| id.into())
}
fn alias(ids: &mut HashMap<String, String>, name: &str, id: String) -> Result<()> {
    if name.is_empty() || name.len() > 128 || ids.contains_key(name) {
        return Err(invalid(
            "client_id must be nonempty and unique within the batch",
        ));
    }
    ids.insert(name.into(), id);
    Ok(())
}
fn check_reply(reply: Value) -> Result<()> {
    if reply["ok"] == true {
        Ok(())
    } else {
        Err(invalid(
            reply["error"].as_str().unwrap_or("parameter edit rejected"),
        ))
    }
}

/// Parameter availability is shared by discovery and every read/write path.
pub(super) fn parameter(
    tl: &TimelineState,
    track_id: &str,
    param: &str,
    writing: bool,
) -> Result<Value> {
    track(tl, track_id)?;
    let root = tl
        .resolve_root_track_id(track_id)
        .ok_or_else(|| invalid("invalid track hierarchy"))?;
    let owner = tl
        .tracks
        .iter()
        .find(|t| t.id == root)
        .ok_or_else(|| invalid("missing root"))?;
    let mut descriptor = if param == "pitch" {
        json!({"id":"pitch","kind":"curve","unit":"midi","min":1.0,"max":127.0,"default":null})
    } else if let Some((prefix, min, max, unit)) = [
        ("child_pitch_offset_cents@", -2400.0, 2400.0, "cents"),
        ("child_pitch_offset_degrees@", -14.0, 14.0, "scale_steps"),
        ("child_formant_offset_cents@", -2400.0, 2400.0, "cents"),
    ]
    .into_iter()
    .find(|(prefix, _, _, _)| param.starts_with(prefix))
    {
        let child = &param[prefix.len()..];
        if child != track_id
            || !tl
                .tracks
                .iter()
                .any(|t| t.id == child && t.parent_id.is_some())
        {
            return Err(invalid("child offset must target its own child track"));
        }
        json!({"id":param,"kind":"curve","unit":unit,"min":min,"max":max,"default":0})
    } else {
        let kind = SynthPipelineKind::from_track_algo(&owner.pitch_analysis_algo);
        let p = hifishifter_kernel::renderer::all_param_descriptors(kind)
            .into_iter()
            .find(|p| p.id == param)
            .ok_or_else(|| Failure::new("PARAM_NOT_SUPPORTED", param))?;
        match p.kind {
            hifishifter_kernel::renderer::ParamKind::AutomationCurve {
                unit,
                default_value,
                min_value,
                max_value,
            } => {
                json!({"id":p.id,"kind":"curve","unit":unit,"default":default_value,"min":min_value,"max":max_value})
            }
            hifishifter_kernel::renderer::ParamKind::StaticEnum {
                options,
                default_value,
            } => json!({"id":p.id,"kind":"static","default":default_value,"options":options}),
        }
    };
    let is_mix = matches!(param, "volume" | "pan" | "dyn");
    let separation = tl
        .params_by_root_track
        .get(&root)
        .and_then(|p| p.extra_params.get("breath_enabled"))
        .copied()
        .unwrap_or(0.0)
        > 0.5;
    let disabled = if !is_mix && !owner.compose_enabled {
        Some("compose_disabled")
    } else if matches!(param, "breath_gain" | "hifigan_tension" | "hifigan_growl") && !separation {
        Some("harmonic_separation_disabled")
    } else {
        None
    };
    descriptor["root_track_id"] = json!(root);
    descriptor["owner_track_id"] = json!(root);
    descriptor["target_track_id"] = json!(track_id);
    descriptor["scope"] = json!("root_track");
    descriptor["readable"] = json!(true);
    descriptor["active"] = json!(disabled.is_none());
    descriptor["writable"] = json!(disabled.is_none() || descriptor["kind"] == "static");
    descriptor["disabled_reason"] = json!(disabled);
    if writing && disabled.is_some() && descriptor["kind"] != "static" {
        return Err(Failure::new("PARAM_DISABLED", disabled.unwrap_or_default()));
    }
    Ok(descriptor)
}

fn values_allowed(descriptor: &Value, values: &[f32]) -> Result<()> {
    if descriptor["kind"] != "curve" {
        return Err(invalid("parameter is static, use set_static"));
    }
    let min = descriptor["min"].as_f64().unwrap_or(0.0);
    let max = descriptor["max"].as_f64().unwrap_or(0.0);
    for &v in values {
        finite(v as f64, min, max, "curve value")?;
    }
    Ok(())
}

pub(super) fn apply(
    timeline: TimelineState,
    operations: &[Operation],
    move_linked: bool,
) -> Result<(TimelineState, Value)> {
    if operations.is_empty() || operations.len() > 256 {
        return Err(invalid("operations must contain 1..256 entries"));
    }
    let fp = timeline.frame_period_ms();
    let total: usize = operations
        .iter()
        .map(|op| match op {
            Operation::WriteCurve { values, .. } => values.len(),
            Operation::RestoreCurve { frame_count, .. } => *frame_count as usize,
            Operation::TransformCurve {
                start_sec, end_sec, ..
            } => (((end_sec - start_sec).max(0.0) * 1000.0 / fp).ceil() as usize).min(200_001),
            _ => 0,
        })
        .sum();
    if total > 200_000 {
        return Err(invalid("curve write budget exceeded (200000 samples)"));
    }
    let draft = Draft::new(timeline);
    let mut ids = HashMap::new();
    for operation in operations {
        match operation {
            Operation::CreateTrack {
                client_id,
                name,
                parent_id,
                index,
            } => {
                let mut tl = lock(&draft);
                let parent = parent_id.as_ref().map(|id| resolve(&ids, id));
                if let Some(id) = &parent {
                    track(&tl, id)?;
                }
                if name.is_empty() || name.len() > 512 {
                    return Err(invalid("invalid track name"));
                }
                let id = tl.add_track(Some(name.clone()), parent, *index);
                alias(&mut ids, client_id, id)?;
            }
            Operation::PatchTrack { track_id, patch } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, track_id);
                track(&tl, &id)?;
                if let Some(v) = patch.volume {
                    finite(
                        v as f64,
                        0.0,
                        hifishifter_kernel::gain::GAIN_MAX as f64,
                        "track volume",
                    )?;
                }
                if let Some(a) = &patch.algorithm {
                    if !matches!(
                        a.as_str(),
                        "nsf_hifigan_onnx" | "world_dll" | "vslib" | "none"
                    ) {
                        return Err(invalid("unknown algorithm ID"));
                    }
                }
                tl.set_track_state(
                    &id,
                    patch.muted,
                    patch.solo,
                    patch.volume,
                    patch.compose_enabled,
                    patch.algorithm.as_deref().map(PitchAnalysisAlgo::from_id),
                    patch.color.clone(),
                    patch.name.clone(),
                );
            }
            Operation::MoveTrack {
                track_id,
                parent_id,
                index,
            } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, track_id);
                track(&tl, &id)?;
                let parent = parent_id.as_ref().map(|id| resolve(&ids, id));
                let mut ancestor = parent.clone();
                while let Some(a) = ancestor {
                    track(&tl, &a)?;
                    if a == id {
                        return Err(invalid("track hierarchy cycle"));
                    }
                    ancestor = tl
                        .tracks
                        .iter()
                        .find(|t| t.id == a)
                        .and_then(|t| t.parent_id.clone());
                }
                tl.move_track(&id, *index, parent);
            }
            Operation::DuplicateTrack { track_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, track_id);
                track(&tl, &id)?;
                tl.duplicate_track(&id);
            }
            Operation::RemoveTrack { track_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, track_id);
                track(&tl, &id)?;
                tl.remove_track(&id);
            }
            Operation::ImportClip {
                client_id,
                track_id,
                path,
                start_sec,
                source_start_sec,
                length_sec,
                playback_rate,
            } => {
                finite(*start_sec, 0.0, 86400.0, "start_sec")?;
                let path = Path::new(path);
                if !path.is_absolute() || !path.is_file() {
                    return Err(invalid("audio path must be an existing absolute file"));
                }
                let info = hifishifter_kernel::audio_utils::try_read_audio_header_only(path)
                    .ok_or_else(|| invalid("audio file cannot be decoded"))?;
                let source = source_start_sec.unwrap_or(0.0);
                finite(source, 0.0, info.duration_sec, "source_start_sec")?;
                let rate = playback_rate.unwrap_or(1.0);
                finite(rate as f64, 0.05, 20.0, "playback_rate")?;
                let length = length_sec.unwrap_or((info.duration_sec - source) / rate as f64);
                finite(length, 0.000001, 86400.0, "length_sec")?;
                if source + length * rate as f64 > info.duration_sec + 1e-6 {
                    return Err(invalid("source window exceeds file duration"));
                }
                let mut tl = lock(&draft);
                let target = resolve(&ids, track_id);
                track(&tl, &target)?;
                let id = tl.add_clip(
                    Some(target),
                    None,
                    Some(*start_sec),
                    Some(length),
                    Some(path.to_string_lossy().into_owned()),
                );
                tl.patch_clip_state(
                    &id,
                    ClipStatePatch {
                        length_sec: Some(length),
                        source_start_sec: Some(source),
                        playback_rate: Some(rate),
                        loop_enabled: Some(false),
                        ..Default::default()
                    },
                );
                alias(&mut ids, client_id, id)?;
            }
            Operation::PatchClip { clip_id, patch } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                for (v, name) in [
                    (patch.start_sec, "start_sec"),
                    (patch.source_start_sec, "source_start_sec"),
                    (patch.fade_in_sec, "fade_in_sec"),
                    (patch.fade_out_sec, "fade_out_sec"),
                    (patch.auto_fade_in_sec, "auto_fade_in_sec"),
                    (patch.auto_fade_out_sec, "auto_fade_out_sec"),
                ] {
                    if let Some(v) = v {
                        finite(v, 0.0, 86400.0, name)?;
                    }
                }
                if let Some(v) = patch.length_sec {
                    finite(v, 0.000001, 86400.0, "length_sec")?;
                }
                if let Some(v) = patch.playback_rate {
                    finite(v as f64, 0.05, 20.0, "playback_rate")?;
                }
                for v in [patch.gain, patch.item_gain].into_iter().flatten() {
                    finite(
                        v as f64,
                        0.0,
                        hifishifter_kernel::gain::GAIN_MAX as f64,
                        "gain",
                    )?;
                }
                tl.patch_clip_state(
                    &id,
                    ClipStatePatch {
                        name: patch.name.clone(),
                        start_sec: patch.start_sec,
                        length_sec: patch.length_sec,
                        source_start_sec: patch.source_start_sec,
                        playback_rate: patch.playback_rate,
                        gain: patch.gain,
                        item_gain: patch.item_gain,
                        muted: patch.muted,
                        reversed: patch.reversed,
                        loop_enabled: patch.loop_enabled,
                        color: patch.color.clone(),
                        fade_in_sec: patch.fade_in_sec,
                        fade_out_sec: patch.fade_out_sec,
                        fade_in_shape: patch.fade_in_shape,
                        fade_out_shape: patch.fade_out_shape,
                        fade_in_dir: patch.fade_in_dir,
                        fade_out_dir: patch.fade_out_dir,
                        auto_fade_in_sec: patch.auto_fade_in_sec,
                        auto_fade_out_sec: patch.auto_fade_out_sec,
                        ..Default::default()
                    },
                );
            }
            Operation::MoveClip {
                clip_id,
                track_id,
                start_sec,
            } => {
                finite(*start_sec, 0.0, 86400.0, "start_sec")?;
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                let target = track_id.as_ref().map(|t| resolve(&ids, t));
                if let Some(t) = &target {
                    track(&tl, t)?;
                }
                tl.move_clips(
                    &[MoveClipPayload {
                        clip_id: id,
                        start_sec: *start_sec,
                        track_id: target,
                    }],
                    move_linked,
                );
            }
            Operation::SplitClip { clip_id, at_sec } => {
                finite(*at_sec, 0.0, 86400.0, "at_sec")?;
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                if tl.split_clip(&id, *at_sec).is_none() {
                    return Err(invalid("split must lie inside the clip"));
                }
            }
            Operation::RemoveClip { clip_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                tl.remove_clips(&[id]);
            }
            Operation::SetActiveTake { clip_id, take_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                let c = tl
                    .clips
                    .iter_mut()
                    .find(|c| c.id == id)
                    .ok_or_else(|| invalid("missing clip"))?;
                if !c.takes.iter().any(|t| t.id == *take_id) {
                    return Err(invalid("unknown take ID"));
                }
                c.switch_active_take(take_id).map_err(invalid)?;
            }
            Operation::RenameTake {
                clip_id,
                take_id,
                name,
            } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                tl.clips
                    .iter_mut()
                    .find(|c| c.id == id)
                    .ok_or_else(|| invalid("missing clip"))?
                    .rename_take(take_id, name)
                    .map_err(invalid)?;
            }
            Operation::DuplicateTake { clip_id, take_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                let c = tl
                    .clips
                    .iter_mut()
                    .find(|c| c.id == id)
                    .ok_or_else(|| invalid("missing clip"))?;
                c.sync_take_from_flat();
                let mut take = c
                    .take(take_id)
                    .cloned()
                    .ok_or_else(|| invalid("unknown take"))?;
                take.id = uuid::Uuid::new_v4().to_string();
                c.add_take(take);
            }
            Operation::RemoveTake { clip_id, take_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                tl.clips
                    .iter_mut()
                    .find(|c| c.id == id)
                    .ok_or_else(|| invalid("missing clip"))?
                    .remove_take(take_id)
                    .map_err(invalid)?;
            }
            Operation::PackTakes { clip_ids } => {
                let mut tl = lock(&draft);
                let targets: Vec<String> = clip_ids.iter().map(|id| resolve(&ids, id)).collect();
                for id in &targets {
                    clip(&tl, id)?;
                }
                if targets.len() < 2 || tl.pack_clips_into_takes(&targets).is_none() {
                    return Err(invalid("clips cannot be packed into takes"));
                }
            }
            Operation::ExplodeTakes { clip_id } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, clip_id);
                clip(&tl, &id)?;
                tl.explode_clip_takes(&id);
            }
            Operation::DuplicateClips {
                clip_ids,
                delta_sec,
            } => {
                finite(*delta_sec, -86400.0, 86400.0, "delta_sec")?;
                let mut tl = lock(&draft);
                let targets: Vec<String> = clip_ids.iter().map(|id| resolve(&ids, id)).collect();
                if targets.is_empty() || targets.len() > 1000 {
                    return Err(invalid("clip_ids must contain 1..1000 items"));
                }
                for id in &targets {
                    clip(&tl, id)?;
                    if tl
                        .clips
                        .iter()
                        .find(|c| c.id == *id)
                        .is_some_and(|c| c.start_sec + delta_sec < 0.0)
                    {
                        return Err(invalid("duplicate would start before zero"));
                    }
                }
                tl.duplicate_clips_bulk(&DuplicateClipsBulkPayload {
                    source_clip_ids: targets,
                    delta_sec: *delta_sec,
                    track_mode: DuplicateClipsTrackMode::SameTrack,
                    copy_linked_params: true,
                    select_created_clips: false,
                    apply_auto_crossfade: false,
                    place_on_selected_track: false,
                    rename_copies: Some(false),
                });
            }
            Operation::GroupClips { clip_ids, grouped } => {
                let mut tl = lock(&draft);
                let targets: Vec<String> = clip_ids.iter().map(|id| resolve(&ids, id)).collect();
                for id in &targets {
                    clip(&tl, id)?;
                }
                if *grouped {
                    tl.group_clips(&targets);
                } else {
                    tl.ungroup_clips(&targets);
                }
            }
            Operation::WriteCurve {
                track_id,
                param,
                start_frame,
                values,
            } => {
                let id = resolve(&ids, track_id);
                let d = parameter(&lock(&draft), &id, param, true)?;
                values_allowed(&d, values)?;
                curve_bounds(&draft, *start_frame, values.len())?;
                check_reply(params::set_param_frames_with_present(
                    &draft,
                    id,
                    param.clone(),
                    *start_frame,
                    values.clone(),
                    Some(vec![true; values.len()]),
                    Some(false),
                ))?;
            }
            Operation::RestoreCurve {
                track_id,
                param,
                start_frame,
                frame_count,
            } => {
                let id = resolve(&ids, track_id);
                parameter(&lock(&draft), &id, param, false)?;
                curve_bounds(&draft, *start_frame, *frame_count as usize)?;
                check_reply(params::restore_param_frames(
                    &draft,
                    id,
                    param.clone(),
                    *start_frame,
                    *frame_count,
                    Some(false),
                ))?;
            }
            Operation::SetStatic {
                track_id,
                param,
                value,
            } => {
                let id = resolve(&ids, track_id);
                let d = parameter(&lock(&draft), &id, param, true)?;
                if d["kind"] != "static"
                    || !d["options"].as_array().is_some_and(|options| {
                        options.iter().any(|o| o[1].as_f64() == Some(*value))
                    })
                {
                    return Err(invalid("invalid static enum option"));
                }
                check_reply(params::set_static_param(
                    &draft,
                    id,
                    param.clone(),
                    *value,
                    Some(false),
                ))?;
            }
            Operation::TransformCurve {
                track_id,
                param,
                start_sec,
                end_sec,
                transform,
            } => transform_curve(
                &draft,
                &resolve(&ids, track_id),
                param,
                *start_sec,
                *end_sec,
                transform,
            )?,
            Operation::SetTempo { points } => {
                if points.is_empty() || points.len() > 10000 || points[0].position_sec != 0.0 {
                    return Err(invalid("tempo points must start at second 0"));
                }
                let mut previous = -1.0;
                for p in points {
                    finite(p.position_sec, 0.0, 86400.0, "tempo position")?;
                    finite(p.bpm, 10.0, 960.0, "bpm")?;
                    if p.position_sec <= previous {
                        return Err(invalid("tempo points must be strictly ordered"));
                    }
                    previous = p.position_sec;
                    if p.numerator.is_some_and(|n| n == 0 || n > 32)
                        || p.denominator
                            .is_some_and(|d| ![1, 2, 4, 8, 16, 32].contains(&d))
                    {
                        return Err(invalid("invalid time signature"));
                    }
                }
                let mut tl = lock(&draft);
                tl.tempo_map = Some(
                    points
                        .iter()
                        .map(|p| TempoPointData {
                            id: uuid::Uuid::new_v4().to_string(),
                            position_sec: p.position_sec,
                            bpm: p.bpm,
                            numerator: p.numerator,
                            denominator: p.denominator,
                            scale: None,
                        })
                        .collect(),
                );
                tl.bpm = points[0].bpm;
                tl.normalize_tempo_map();
            }
            Operation::SetProjectLength { seconds } => {
                finite(*seconds, 0.01, 86400.0, "seconds")?;
                lock(&draft).set_project_length(*seconds);
            }
            Operation::SetParamBlocks { track_id, layout } => {
                let mut tl = lock(&draft);
                let id = resolve(&ids, track_id);
                track(&tl, &id)?;
                let root = tl
                    .resolve_root_track_id(&id)
                    .ok_or_else(|| invalid("invalid root"))?;
                let next: hifishifter_kernel::param_line_blocks::ParamLineBlockSet =
                    serde_json::from_value(layout.clone()).map_err(|e| invalid(e.to_string()))?;
                next.validate().map_err(invalid)?;
                if (next.frame_period_ms - tl.frame_period_ms()).abs() > 1e-9 {
                    return Err(invalid(
                        "parameter block frame period does not match project",
                    ));
                }
                let entry = tl.params_by_root_track.entry(root).or_default();
                let revision = entry.param_line_blocks.as_ref().map_or(0, |b| b.revision);
                let mut next = next;
                next.revision = revision.saturating_add(1);
                entry.param_line_blocks = Some(next);
            }
        }
    }
    let result = draft
        .timeline
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    Ok((result, json!(ids)))
}

fn curve_bounds(draft: &Draft, start: u32, count: usize) -> Result<()> {
    let tl = lock(draft);
    let max = tl.target_param_frames(tl.frame_period_ms());
    if count == 0
        || count > 200_000
        || (start as usize)
            .checked_add(count)
            .is_none_or(|end| end > max)
    {
        return Err(invalid("curve range exceeds project or sample budget"));
    }
    Ok(())
}
fn transform_curve(
    draft: &Draft,
    id: &str,
    param: &str,
    start: f64,
    end: f64,
    transform: &CurveTransform,
) -> Result<()> {
    finite(start, 0.0, 86400.0, "start_sec")?;
    finite(end, 0.0, 86400.0, "end_sec")?;
    if end <= start {
        return Err(invalid("end_sec must exceed start_sec"));
    }
    let descriptor = parameter(&lock(draft), id, param, true)?;
    let fp = lock(draft).frame_period_ms();
    let first = (start * 1000.0 / fp).ceil() as u32;
    let last = (end * 1000.0 / fp).ceil() as u32;
    let count = last.saturating_sub(first);
    curve_bounds(draft, first, count as usize)?;
    let frames = params::get_param_frames(
        draft,
        id.into(),
        param.into(),
        first,
        count,
        None,
        Some(false),
        Some(true),
    );
    if frames.analysis_pending == Some(true) {
        return Err(Failure::new(
            "ANALYSIS_PENDING",
            "source analysis has not finished; query the curve again after App analysis completes",
        ));
    }
    if !frames.ok || frames.edit.len() != count as usize {
        return Err(Failure::new("ANALYSIS_PENDING", "curve data is not ready"));
    }
    let pitch = param == "pitch";
    let mut values = frames.edit.clone();
    let mut present = vec![true; values.len()];
    let mut voiced: Vec<f32> = frames
        .orig
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    voiced.sort_by(f32::total_cmp);
    let median = voiced.get(voiced.len() / 2).copied();
    for i in 0..values.len() {
        if pitch && frames.orig.get(i).copied().unwrap_or(0.0) <= 0.0 {
            values[i] = 0.0;
            present[i] = false;
            continue;
        }
        values[i] = match transform {
            CurveTransform::Set { value } => *value,
            CurveTransform::Ramp { from, to } => {
                from + (to - from) * i as f32 / (values.len().saturating_sub(1).max(1) as f32)
            }
            CurveTransform::Shift { amount } => values[i] + amount,
            CurveTransform::ShiftToNote { target_midi } => {
                if !pitch {
                    return Err(invalid("shift_to_note requires pitch"));
                }
                frames.orig[i] + target_midi
                    - median.ok_or_else(|| {
                        Failure::new("PITCH_UNAVAILABLE", "no voiced source pitch in range")
                    })?
            }
            CurveTransform::FlattenToNote { target_midi } => {
                if !pitch {
                    return Err(invalid("flatten_to_note requires pitch"));
                }
                *target_midi
            }
        };
    }
    if pitch && voiced.is_empty() {
        return Err(Failure::new(
            "PITCH_UNAVAILABLE",
            "no voiced source pitch in range",
        ));
    }
    let checked: Vec<f32> = values
        .iter()
        .zip(&present)
        .filter_map(|(v, p)| p.then_some(*v))
        .collect();
    values_allowed(&descriptor, &checked)?;
    check_reply(params::set_param_frames_with_present(
        draft,
        id.into(),
        param.into(),
        first,
        values,
        Some(present),
        Some(false),
    ))
}

#[cfg(test)]
#[path = "draft_tests.rs"]
mod tests;
