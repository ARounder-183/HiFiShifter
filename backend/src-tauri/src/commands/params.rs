use crate::state::AppState;
use tauri::State;

// ===================== param curves =====================

pub(super) fn resolve_extra_curve_default_value(
    kind: &crate::state::SynthPipelineKind,
    param: &str,
) -> f32 {
    crate::renderer::automation_curve_default_value(kind, param).unwrap_or(0.0)
}

fn resolve_param_reference_value(kind: &crate::state::SynthPipelineKind, param: &str) -> f32 {
    match param {
        "pitch" => 0.0,
        "tension" => 0.0,
        _ => resolve_extra_curve_default_value(kind, param),
    }
}

fn resolve_param_reference_kind(param: &str) -> crate::models::ParamReferenceKind {
    match param {
        "pitch" => crate::models::ParamReferenceKind::SourceCurve,
        _ => crate::models::ParamReferenceKind::DefaultValue,
    }
}

fn resolve_extra_curve_frame_pair(
    curve: Option<&Vec<f32>>,
    default_value: f32,
    idx: usize,
) -> (f32, f32) {
    let edit_value = curve
        .and_then(|values| values.get(idx))
        .copied()
        .unwrap_or(default_value);
    (default_value, edit_value)
}

fn resolve_static_param_default_value(
    kind: &crate::state::SynthPipelineKind,
    param: &str,
) -> f64 {
    crate::renderer::static_enum_default_value(kind, param)
        .map(|value| value as f64)
        .unwrap_or(0.0)
}




pub(super) fn get_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    stride: Option<u32>,
) -> crate::models::ParamFramesPayload {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "get_param_frames(track_id={}, param={}, start_frame={}, frame_count={}, stride={:?})",
            track_id, param, start_frame, frame_count, stride
        );
    }
    let (root, fp, entry, compose_enabled, pitch_algo) = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        let root = match tl.resolve_root_track_id(&track_id) {
            Some(id) => id,
            None => {
                return crate::models::ParamFramesPayload {
                    ok: false,
                    root_track_id: "".to_string(),
                    param,
                    frame_period_ms: tl.frame_period_ms(),
                    start_frame,
                    orig: vec![],
                    edit: vec![],
                    reference_kind: resolve_param_reference_kind("pitch"),
                    analysis_pending: None,
                    analysis_progress: None,
                    pitch_edit_user_modified: None,
                    pitch_edit_backend_available: None,
                }
            }
        };

        tl.ensure_params_for_root(&root);
        let fp = tl.frame_period_ms();

        let track = tl.tracks.iter().find(|t| t.id == root);
        let compose_enabled = track.map(|t| t.compose_enabled).unwrap_or(false);
        let pitch_algo = track
            .map(|t| t.pitch_analysis_algo.clone())
            .unwrap_or_default();

        let entry = tl
            .params_by_root_track
            .get(&root)
            .cloned()
            .unwrap_or_default();

        (root, fp, entry, compose_enabled, pitch_algo)
    };

    let pitch_edit_user_modified =
        (param == "pitch").then_some(entry.pitch_edit_user_modified);

    let pitch_edit_backend_available = if param == "pitch" {
        let algo = crate::pitch_editing::PitchEditAlgorithm::from_track_algo(&pitch_algo);
        let available = match algo {
            crate::pitch_editing::PitchEditAlgorithm::WorldVocoder => {
                crate::world_vocoder::is_available()
            }
            crate::pitch_editing::PitchEditAlgorithm::NsfHifiganOnnx => {
                crate::nsf_hifigan_onnx::is_available()
            }
            #[cfg(feature = "vslib")]
            crate::pitch_editing::PitchEditAlgorithm::VocalShifterVslib => true,
            crate::pitch_editing::PitchEditAlgorithm::Bypass => true,
        };
        Some(available)
    } else {
        None
    };

    if param == "pitch" && !compose_enabled {
        return crate::models::ParamFramesPayload {
            ok: true,
            root_track_id: root,
            param,
            frame_period_ms: fp,
            start_frame,
            orig: vec![],
            edit: vec![],
            reference_kind: resolve_param_reference_kind("pitch"),
            analysis_pending: None,
            analysis_progress: None,
            pitch_edit_user_modified,
            pitch_edit_backend_available,
        };
    }

    // Schedule pitch_orig analysis in background; return current cached curve immediately.
    let analysis_pending = if param == "pitch" {
        Some(crate::pitch_analysis::maybe_schedule_pitch_orig(&state, &root))
    } else {
        None
    };

    let start = start_frame as usize;
    let count = (frame_count as usize).max(1);
    let step = (stride.unwrap_or(1).max(1)) as usize;
    let kind = crate::state::SynthPipelineKind::from_track_algo(&pitch_algo);

    let mut orig = Vec::with_capacity(count);
    let mut edit = Vec::with_capacity(count);

    match param.as_str() {
        "pitch" => {
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let o = entry.pitch_orig.get(idx).copied().unwrap_or(0.0);
                let e_raw = entry.pitch_edit.get(idx).copied().unwrap_or(0.0);
                // Treat 0 as "unset" and fall back to orig.
                let e = if e_raw == 0.0 && o != 0.0 { o } else { e_raw };
                orig.push(o);
                edit.push(e);
            }
        }
        "tension" => {
            let reference_value = resolve_param_reference_value(&kind, &param);
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let e = entry
                    .tension_edit
                    .get(idx)
                    .copied()
                    .unwrap_or(reference_value);
                orig.push(reference_value);
                edit.push(e);
            }
        }
        _ => {
            // Extra automation curve: dashed orig should stay at the processor default,
            // while solid edit reflects the user-authored curve.
            let curve = entry.extra_curves.get(&param);
            let default_value = resolve_param_reference_value(&kind, &param);
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let (o, e) = resolve_extra_curve_frame_pair(curve, default_value, idx);
                orig.push(o);
                edit.push(e);
            }
        }
    }

    crate::models::ParamFramesPayload {
        ok: true,
        root_track_id: root,
        param: param.clone(),
        frame_period_ms: fp,
        start_frame,
        orig,
        edit,
        reference_kind: resolve_param_reference_kind(&param),
        analysis_pending,
        analysis_progress: None,
        pitch_edit_user_modified,
        pitch_edit_backend_available,
    }
}




pub(super) fn set_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    values: Vec<f32>,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);

    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    // For extra_curves we need separate handling; batch into known vs extra below.
    let is_extra_curve = !matches!(param.as_str(), "pitch" | "tension");
    if is_extra_curve {
        // Ensure the curve vector exists and is long enough.
        let curve = entry.extra_curves.entry(param.clone()).or_insert_with(Vec::new);
        let needed = start_frame as usize + values.len();
        let default_value = resolve_extra_curve_default_value(&kind, &param);
        if curve.len() < needed {
            curve.resize(needed, default_value);
        }
    }

    let dst = match param.as_str() {
        "pitch" => &mut entry.pitch_edit,
        "tension" => &mut entry.tension_edit,
        _ => {
            // Safety: we ensured extra_curves[&param] exists above.
            entry.extra_curves.get_mut(&param).unwrap()
        }
    };

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
    let extra_curve_default = is_extra_curve
        .then(|| resolve_extra_curve_default_value(&kind, &param))
        .unwrap_or(0.0);

    let start = start_frame as usize;
    let mut written = 0usize;
    let mut non_finite = 0usize;
    let mut clamped = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut max_delta = 0.0f32;
    let mut prev_v: Option<f32> = None;
    for (i, v) in values.into_iter().enumerate() {
        let idx = start.saturating_add(i);
        if idx >= dst.len() {
            break;
        }

        let mut v = if v.is_finite() {
            v
        } else {
            non_finite += 1;
            if is_extra_curve { extra_curve_default } else { 0.0 }
        };

        match param.as_str() {
            "pitch" => {
                // MIDI pitch. Keep 0 as "unset"; otherwise clamp into a reasonable range.
                if v != 0.0 {
                    let vv = v.clamp(1.0, 127.0);
                    if vv != v {
                        clamped += 1;
                    }
                    v = vv;
                }
            }
            "tension" => {
                // Tension is a UI parameter in [-100, 100].
                let vv = v.clamp(-100.0, 100.0);
                if vv != v {
                    clamped += 1;
                }
                v = vv;
            }
            _ => {}
        }

        min_v = min_v.min(v);
        max_v = max_v.max(v);
        if let Some(p) = prev_v {
            max_delta = max_delta.max((v - p).abs());
        }
        prev_v = Some(v);

        dst[idx] = v;
        written += 1;
    }

    if debug {
        // This helps diagnose whether the frontend is sending invalid / extreme curves.
        eprintln!(
            "set_param_frames(param={param}, start_frame={start_frame}, len={}): non_finite={non_finite} clamped={clamped} min={min_v:.3} max={max_v:.3} max_delta={max_delta:.3}",
            written
        );
    }

    if param == "pitch" {
        entry.pitch_edit_user_modified = true;
    }

    // Ensure realtime playback reflects edits immediately.
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}




pub(super) fn restore_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);
    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    let start = start_frame as usize;
    let count = (frame_count as usize).max(1);

    match param.as_str() {
        "pitch" => {
            for i in 0..count {
                let idx = start.saturating_add(i);
                if idx >= entry.pitch_edit.len() {
                    break;
                }
                let o = entry.pitch_orig.get(idx).copied().unwrap_or(0.0);
                entry.pitch_edit[idx] = o;
            }

            // If the curve fully matches orig now, clear the user-modified flag.
            let len = entry.pitch_orig.len().min(entry.pitch_edit.len());
            entry.pitch_edit_user_modified = false;
            for i in 0..len {
                let o = entry.pitch_orig[i];
                let e = entry.pitch_edit[i];
                if (e.is_finite() && e > 0.0)
                    && (!(o.is_finite() && o > 0.0) || (e - o).abs() > 1e-3)
                {
                    entry.pitch_edit_user_modified = true;
                    break;
                }
            }
        }
        "tension" => {
            let reference_value = resolve_param_reference_value(&kind, &param);
            for i in 0..count {
                let idx = start.saturating_add(i);
                if idx >= entry.tension_edit.len() {
                    break;
                }
                entry.tension_edit[idx] = reference_value;
            }
        }
        _ => {
            let default_value = resolve_param_reference_value(&kind, &param);
            if let Some(curve) = entry.extra_curves.get_mut(&param) {
                for i in 0..count {
                    let idx = start.saturating_add(i);
                    if idx >= curve.len() {
                        break;
                    }
                    curve[idx] = default_value;
                }
            }
        }
    }

    // Ensure realtime playback reflects edits immediately.
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}

pub(super) fn get_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
) -> crate::models::StaticParamValuePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let root = match tl.resolve_root_track_id(&track_id) {
        Some(id) => id,
        None => {
            return crate::models::StaticParamValuePayload {
                ok: false,
                root_track_id: String::new(),
                param,
                value: 0.0,
            }
        }
    };

    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);
    let value = tl
        .params_by_root_track
        .get(&root)
        .and_then(|entry| entry.extra_params.get(&param).copied())
        .unwrap_or_else(|| resolve_static_param_default_value(&kind, &param));

    crate::models::StaticParamValuePayload {
        ok: true,
        root_track_id: root,
        param,
        value,
    }
}

pub(super) fn set_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    value: f64,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);

    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    entry.extra_params.insert(param, value);
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_extra_curve_default_value, resolve_extra_curve_frame_pair,
        resolve_param_reference_value,
    };
    use crate::state::SynthPipelineKind;

    #[test]
    fn breath_gain_uses_processor_default_value() {
        let value = resolve_extra_curve_default_value(
            &SynthPipelineKind::NsfHifiganOnnx,
            "breath_gain",
        );
        assert!((value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unknown_extra_curve_falls_back_to_zero() {
        let value = resolve_extra_curve_default_value(
            &SynthPipelineKind::NsfHifiganOnnx,
            "not_existing_param",
        );
        assert!(value.abs() < 1e-6);
    }

    #[test]
    fn non_pitch_reference_curve_uses_backend_default_value() {
        let tension = resolve_param_reference_value(&SynthPipelineKind::NsfHifiganOnnx, "tension");
        let breath_gain =
            resolve_param_reference_value(&SynthPipelineKind::NsfHifiganOnnx, "breath_gain");

        assert!(tension.abs() < 1e-6);
        assert!((breath_gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn extra_curve_orig_stays_at_default_when_edit_exists() {
        let curve = vec![1.25, 1.5, 0.75];

        let (orig, edit) = resolve_extra_curve_frame_pair(Some(&curve), 1.0, 1);

        assert!((orig - 1.0).abs() < 1e-6);
        assert!((edit - 1.5).abs() < 1e-6);
    }

    #[test]
    fn extra_curve_orig_and_edit_fall_back_to_default_when_missing() {
        let (orig, edit) = resolve_extra_curve_frame_pair(None, 1.0, 12);

        assert!((orig - 1.0).abs() < 1e-6);
        assert!((edit - 1.0).abs() < 1e-6);
    }
}
