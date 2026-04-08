use crate::state::{PitchAnalysisAlgo, SynthPipelineKind, TimelineState};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static MONO_SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn pitch_edit_algo_from_env() -> Option<String> {
    std::env::var("HIFISHIFTER_PITCH_EDIT_ALGO")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchEditAlgorithm {
    WorldVocoder,
    NsfHifiganOnnx,
    #[cfg(feature = "vslib")]
    VocalShifterVslib,
    Bypass,
}

#[derive(Debug, Clone)]
pub(crate) struct PitchCurvesSnapshot<'a> {
    pub frame_period_ms: f64,
    pub pitch_orig: &'a [f32],
    pub pitch_edit: &'a [f32],
}

impl<'a> PitchCurvesSnapshot<'a> {
    #[allow(dead_code)]
    pub fn midi_at_time(&self, abs_time_sec: f64) -> f64 {
        if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
            return 0.0;
        }

        let inv_fp = 1000.0 / self.frame_period_ms.max(0.1);
        let idx_f = abs_time_sec * inv_fp;
        if !(idx_f.is_finite() && idx_f >= 0.0) {
            return 0.0;
        }
        let i0 = idx_f.floor() as isize;
        if i0 < 0 {
            return 0.0;
        }
        let i0 = i0 as usize;
        let len = self.pitch_orig.len().min(self.pitch_edit.len().max(1));
        if i0 >= len {
            return 0.0;
        }
        let i1 = (i0 + 1).min(len.saturating_sub(1));
        let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

        let orig0 = self.pitch_orig.get(i0).copied().unwrap_or(0.0) as f64;
        let orig1 = self.pitch_orig.get(i1).copied().unwrap_or(0.0) as f64;
        let edit0 = self.pitch_edit.get(i0).copied().unwrap_or(0.0) as f64;
        let edit1 = self.pitch_edit.get(i1).copied().unwrap_or(0.0) as f64;

        // For ONNX, `pitch_edit` is treated as an absolute target MIDI curve.
        // Allow it to work even when `pitch_orig` is missing (all zeros).
        let mut base0 = if edit0.is_finite() && edit0 > 0.0 {
            edit0
        } else {
            orig0
        };
        let mut base1 = if edit1.is_finite() && edit1 > 0.0 {
            edit1
        } else {
            orig1
        };

        if !(base0.is_finite() && base0 > 0.0) && (base1.is_finite() && base1 > 0.0) {
            base0 = base1;
        }
        if !(base1.is_finite() && base1 > 0.0) && (base0.is_finite() && base0 > 0.0) {
            base1 = base0;
        }
        if !(base0.is_finite() && base0 > 0.0 && base1.is_finite() && base1 > 0.0) {
            return 0.0;
        }

        let v = base0 + (base1 - base0) * frac;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub fn is_voiced_at_time(&self, abs_time_sec: f64) -> bool {
        let fp = self.frame_period_ms.max(0.1);
        let idx = ((abs_time_sec.max(0.0) * 1000.0) / fp).round().max(0.0) as usize;
        let orig = self.pitch_orig.get(idx).copied().unwrap_or(0.0);
        let edit = self.pitch_edit.get(idx).copied().unwrap_or(0.0);
        (orig.is_finite() && orig > 0.0) || (edit.is_finite() && edit > 0.0)
    }
}

#[allow(dead_code)]
pub(crate) fn selected_pitch_curves_snapshot<'a>(
    timeline: &'a TimelineState,
) -> Option<PitchCurvesSnapshot<'a>> {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let root = timeline.resolve_root_track_id(&selected)?;

    let entry = timeline.params_by_root_track.get(&root)?;
    Some(PitchCurvesSnapshot {
        frame_period_ms: entry.frame_period_ms.max(0.1),
        pitch_orig: &entry.pitch_orig,
        pitch_edit: &entry.pitch_edit,
    })
}

fn pitch_edit_backend_available_for_track(track: &crate::state::Track) -> bool {
    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    match algo {
        PitchEditAlgorithm::WorldVocoder => crate::world_vocoder::is_available(),
        PitchEditAlgorithm::NsfHifiganOnnx => crate::nsf_hifigan_onnx::is_available(),
        #[cfg(feature = "vslib")]
        PitchEditAlgorithm::VocalShifterVslib => true,
        PitchEditAlgorithm::Bypass => true,
    }
}

pub(crate) fn extra_param_enabled(extra_params: &HashMap<String, f64>, key: &str) -> bool {
    extra_params.get(key).copied().unwrap_or(0.0) >= 0.5
}

fn curve_differs_from_default_in_range(
    curve: Option<&Vec<f32>>,
    frame_period_ms: f64,
    start_sec: f64,
    end_sec: f64,
    default_value: f32,
) -> bool {
    let Some(curve) = curve else {
        return false;
    };
    if curve.is_empty() {
        return false;
    }

    let fp = frame_period_ms.max(0.1);
    let start_idx = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_idx = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let lo = start_idx.min(curve.len());
    let hi = end_idx.min(curve.len());
    curve[lo..hi]
        .iter()
        .any(|value| (value - default_value).abs() > 1e-3)
}

pub(crate) fn hifigan_tension_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a Vec<f32>> {
    clip.extra_curves
        .as_ref()
        .and_then(|curves| curves.get("hifigan_tension"))
        .or_else(|| entry.extra_curves.get("hifigan_tension"))
}

pub(crate) fn hifigan_tension_active_for_clip(
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let curve = hifigan_tension_curve_for_clip(entry, clip);
    curve_differs_from_default_in_range(
        curve,
        entry.frame_period_ms.max(0.1),
        clip_start_sec,
        clip_start_sec + clip.length_sec.max(0.0),
        0.0,
    )
}

pub(crate) fn hifigan_formant_shift_curve_for_clip<'a>(
    entry: &'a crate::state::TrackParamsState,
    clip: &'a crate::state::Clip,
) -> Option<&'a Vec<f32>> {
    clip.extra_curves
        .as_ref()
        .and_then(|curves| curves.get("formant_shift_cents"))
        .or_else(|| entry.extra_curves.get("formant_shift_cents"))
}

pub(crate) fn hifigan_formant_shift_active_for_clip(
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let curve = hifigan_formant_shift_curve_for_clip(entry, clip);
    curve_differs_from_default_in_range(
        curve,
        entry.frame_period_ms.max(0.1),
        clip_start_sec,
        clip_start_sec + clip.length_sec.max(0.0),
        0.0,
    )
}

fn track_requests_extra_processing(
    algo: PitchEditAlgorithm,
    entry: &crate::state::TrackParamsState,
    clip: &crate::state::Clip,
) -> bool {
    let extra_params = clip.extra_params.as_ref().unwrap_or(&entry.extra_params);
    matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && extra_param_enabled(extra_params, "breath_enabled")
}

impl PitchEditAlgorithm {
    pub fn from_track_algo(algo: &PitchAnalysisAlgo) -> Self {
        if let Some(v) = pitch_edit_algo_from_env() {
            if matches!(v.as_str(), "nsf_hifigan" | "nsf_hifigan_onnx" | "onnx") {
                return Self::NsfHifiganOnnx;
            }
            if matches!(v.as_str(), "world" | "world_vocoder") {
                // fall through to track algo below
            }
        }
        match algo {
            PitchAnalysisAlgo::WorldDll | PitchAnalysisAlgo::Unknown => Self::WorldVocoder,
            PitchAnalysisAlgo::NsfHifiganOnnx => Self::NsfHifiganOnnx,
            #[cfg(feature = "vslib")]
            PitchAnalysisAlgo::VocalShifterVslib => Self::VocalShifterVslib,
            #[cfg(not(feature = "vslib"))]
            PitchAnalysisAlgo::VocalShifterVslib => Self::Bypass,
            PitchAnalysisAlgo::None => Self::Bypass,
        }
    }
}

#[allow(dead_code)]
pub fn selected_pitch_edit_algorithm(timeline: &TimelineState) -> PitchEditAlgorithm {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return PitchEditAlgorithm::Bypass;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return PitchEditAlgorithm::Bypass;
    };

    PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo)
}

fn semitone_ratio(semitones: f64) -> f64 {
    (2.0f64).powf(semitones / 12.0)
}

#[derive(Debug, Clone, Copy)]
enum ChildPitchOffsetParamMode {
    Cents,
    Degrees,
}

#[derive(Debug, Clone, Copy)]
struct ChildPitchOffsetLayer<'a> {
    cents: f64,
    degree_steps: f64,
    cents_curve: Option<&'a Vec<f32>>,
    degree_steps_curve: Option<&'a Vec<f32>>,
}

#[derive(Debug, Clone)]
struct ChildPitchOffsetConfig<'a> {
    layers: Vec<ChildPitchOffsetLayer<'a>>,
}

const CHILD_PITCH_OFFSET_CENTS_PREFIX: &str = "child_pitch_offset_cents@";
const CHILD_PITCH_OFFSET_DEGREES_PREFIX: &str = "child_pitch_offset_degrees@";
const CHILD_PITCH_OFFSET_CENTS_DEFAULT: f64 = 0.0;
const CHILD_PITCH_OFFSET_DEGREES_DEFAULT: f64 = 0.0;

fn child_pitch_offset_curve_key(mode: ChildPitchOffsetParamMode, track_id: &str) -> String {
    match mode {
        ChildPitchOffsetParamMode::Cents => {
            format!("{CHILD_PITCH_OFFSET_CENTS_PREFIX}{track_id}")
        }
        ChildPitchOffsetParamMode::Degrees => {
            format!("{CHILD_PITCH_OFFSET_DEGREES_PREFIX}{track_id}")
        }
    }
}

fn ordered_scale_semitone_offsets(scale_notes: &[u8]) -> Vec<i32> {
    if scale_notes.is_empty() {
        return vec![0, 2, 4, 5, 7, 9, 11];
    }
    let mut normalized: Vec<i32> = scale_notes.iter().map(|v| (v % 12) as i32).collect();
    normalized.sort_unstable();
    normalized.dedup();
    if normalized.is_empty() {
        return vec![0, 2, 4, 5, 7, 9, 11];
    }

    let mut out = Vec::with_capacity(normalized.len());
    let mut prev = i32::MIN;
    for mut value in normalized {
        while value <= prev {
            value += 12;
        }
        out.push(value);
        prev = value;
    }
    out
}

fn scale_degree_to_midi_integer(abs_degree: i32, offsets: &[i32]) -> f64 {
    let degree_count = offsets.len() as i32;
    if degree_count <= 0 {
        return 0.0;
    }
    let oct = abs_degree.div_euclid(degree_count);
    let idx = abs_degree.rem_euclid(degree_count) as usize;
    (oct * 12 + offsets[idx]) as f64
}

fn scale_degree_to_midi(abs_degree: f64, offsets: &[i32]) -> f64 {
    if !abs_degree.is_finite() {
        return 0.0;
    }
    let lower_degree = abs_degree.floor() as i32;
    let frac = abs_degree - lower_degree as f64;
    let lower = scale_degree_to_midi_integer(lower_degree, offsets);
    if frac <= 1e-9 {
        return lower;
    }
    let upper = scale_degree_to_midi_integer(lower_degree + 1, offsets);
    lower + (upper - lower) * frac
}

fn transpose_midi_by_scale_steps(midi: f64, degree_steps: f64, scale_notes: &[u8]) -> f64 {
    if !midi.is_finite() || degree_steps.abs() <= 1e-9 {
        return midi;
    }
    let offsets = ordered_scale_semitone_offsets(scale_notes);
    if offsets.is_empty() {
        return midi;
    }

    let degree_count = offsets.len() as i32;
    let base_oct = (midi / 12.0).floor() as i32;

    let mut lower: Option<(i32, f64)> = None;
    let mut upper: Option<(i32, f64)> = None;
    for oct in (base_oct - 3)..=(base_oct + 3) {
        for (idx, offset) in offsets.iter().enumerate() {
            let candidate_midi = (oct * 12 + *offset) as f64;
            let abs_degree = oct * degree_count + idx as i32;
            if candidate_midi <= midi {
                if lower.map(|(_, v)| candidate_midi > v).unwrap_or(true) {
                    lower = Some((abs_degree, candidate_midi));
                }
            }
            if candidate_midi >= midi {
                if upper.map(|(_, v)| candidate_midi < v).unwrap_or(true) {
                    upper = Some((abs_degree, candidate_midi));
                }
            }
        }
    }

    let (lower_degree, lower_midi) = lower.unwrap_or((0, midi));
    let (upper_degree, upper_midi) = upper.unwrap_or((lower_degree, lower_midi));
    let span = upper_midi - lower_midi;
    let ratio = if span.abs() <= 1e-9 {
        0.0
    } else {
        ((midi - lower_midi) / span).clamp(0.0, 1.0)
    };

    let target_lower = scale_degree_to_midi(lower_degree as f64 + degree_steps, &offsets);
    let target_upper = scale_degree_to_midi(upper_degree as f64 + degree_steps, &offsets);
    target_lower + (target_upper - target_lower) * ratio
}

fn active_child_pitch_offset_config<'a>(
    timeline: &'a TimelineState,
    clip_track_id: &str,
) -> Option<ChildPitchOffsetConfig<'a>> {
    let track = timeline.tracks.iter().find(|track| track.id == clip_track_id)?;
    if track.parent_id.is_none() {
        return None;
    }

    let root_track_id = timeline.resolve_root_track_id(clip_track_id)?;
    let entry = timeline.params_by_root_track.get(&root_track_id);

    let mut lineage_child_ids: Vec<&str> = Vec::new();
    let mut cursor = Some(clip_track_id);
    let mut safety = 0usize;
    while let Some(track_id) = cursor {
        let Some(node) = timeline.tracks.iter().find(|track| track.id == track_id) else {
            break;
        };
        if node.parent_id.is_none() {
            break;
        }
        lineage_child_ids.push(track_id);
        cursor = node.parent_id.as_deref();
        safety += 1;
        if safety > timeline.tracks.len() + 2 {
            break;
        }
    }

    if lineage_child_ids.is_empty() {
        return None;
    }

    lineage_child_ids.reverse();

    let frame_period_ms = entry.map(|state| state.frame_period_ms).unwrap_or(5.0);
    let mut has_effective = false;
    let mut layers: Vec<ChildPitchOffsetLayer<'a>> = Vec::with_capacity(lineage_child_ids.len());

    for track_id in lineage_child_ids {
        let cents_curve = entry.and_then(|state| {
            state
                .extra_curves
                .get(&child_pitch_offset_curve_key(
                    ChildPitchOffsetParamMode::Cents,
                    track_id,
                ))
        });
        let degree_steps_curve = entry.and_then(|state| {
            state
                .extra_curves
                .get(&child_pitch_offset_curve_key(
                    ChildPitchOffsetParamMode::Degrees,
                    track_id,
                ))
        });

        let static_cents = CHILD_PITCH_OFFSET_CENTS_DEFAULT;
        let static_degree_steps = CHILD_PITCH_OFFSET_DEGREES_DEFAULT;

        let has_cents_curve = curve_differs_from_default_in_range(
            cents_curve,
            frame_period_ms,
            0.0,
            f64::MAX,
            static_cents as f32,
        );
        let has_degree_curve = curve_differs_from_default_in_range(
            degree_steps_curve,
            frame_period_ms,
            0.0,
            f64::MAX,
            static_degree_steps as f32,
        );
        has_effective = has_effective || has_cents_curve || has_degree_curve;

        layers.push(ChildPitchOffsetLayer {
            cents: static_cents,
            degree_steps: static_degree_steps,
            cents_curve,
            degree_steps_curve,
        });
    }

    if !has_effective {
        return None;
    }

    Some(ChildPitchOffsetConfig { layers })
}

fn sample_child_offset_cents(layer: &ChildPitchOffsetLayer<'_>, frame_idx: usize) -> f64 {
    layer
        .cents_curve
        .and_then(|curve| curve.get(frame_idx).copied())
        .filter(|value| value.is_finite())
        .map(|value| value as f64)
        .unwrap_or(layer.cents)
}

fn sample_child_offset_degree_steps(layer: &ChildPitchOffsetLayer<'_>, frame_idx: usize) -> f64 {
    layer
        .degree_steps_curve
        .and_then(|curve| curve.get(frame_idx).copied())
        .filter(|value| value.is_finite())
        .map(|value| value as f64)
        .unwrap_or(layer.degree_steps)
}

fn apply_child_pitch_offset_to_midi(
    midi: f64,
    cfg: &ChildPitchOffsetConfig<'_>,
    frame_idx: usize,
    scale_notes: &[u8],
) -> f64 {
    if !(midi.is_finite() && midi > 0.0) {
        return 0.0;
    }

    let mut current = midi;
    for layer in &cfg.layers {
        let steps = sample_child_offset_degree_steps(layer, frame_idx);
        let cents = sample_child_offset_cents(layer, frame_idx);

        if steps.abs() > 1e-9 {
            current = transpose_midi_by_scale_steps(current, steps, scale_notes);
        }
        if cents.abs() > 1e-9 {
            current += cents / 100.0;
        }

        if !(current.is_finite() && current > 0.0) {
            return 0.0;
        }
    }

    current
}

pub(crate) fn build_clip_input_pitch_curve(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
    frame_period_ms: f64,
    clip_playback_rate: f64,
    is_vslib: bool,
) -> Option<Vec<f32>> {
    let child_offset_cfg = active_child_pitch_offset_config(timeline, &clip.track_id);

    let timeline_midi_raw: Vec<f32> = if is_vslib {
        Vec::new()
    } else {
        let clip_root = clip_root_track_id(timeline, clip)?;
        let clip_pitch = crate::pitch_clip::get_or_compute_clip_pitch_midi_global(
            timeline,
            clip,
            &clip_root,
            frame_period_ms,
        )?;

        let tm = crate::pitch_clip::trim_and_resample_midi(
            &clip_pitch.midi,
            frame_period_ms,
            clip.source_start_sec,
            clip.source_end_sec,
            clip_playback_rate,
            clip.length_sec.max(0.0),
        );
        if tm.is_empty() {
            return None;
        }
        tm
    };

    let timeline_midi = if let Some(ref cfg) = child_offset_cfg {
        let fp = frame_period_ms.max(0.1);
        let start_idx = ((clip_start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
        timeline_midi_raw
            .iter()
            .enumerate()
            .map(|(local_idx, &midi)| {
                let frame_idx = start_idx.saturating_add(local_idx);
                apply_child_pitch_offset_to_midi(
                    midi as f64,
                    cfg,
                    frame_idx,
                    &timeline.project_scale_notes,
                ) as f32
            })
            .collect()
    } else {
        timeline_midi_raw
    };

    Some(timeline_midi)
}

fn root_pitch_edit_state<'a>(
    timeline: &'a TimelineState,
    root_track_id: &str,
) -> Option<(&'a crate::state::Track, &'a crate::state::TrackParamsState)> {
    let track = timeline
        .tracks
        .iter()
        .find(|track| track.id == root_track_id)?;
    let entry = timeline.params_by_root_track.get(root_track_id)?;
    Some((track, entry))
}

fn clip_pitch_edit_state<'a>(
    timeline: &'a TimelineState,
    clip: &crate::state::Clip,
) -> Option<(&'a crate::state::Track, &'a crate::state::TrackParamsState)> {
    let clip_root = timeline.resolve_root_track_id(&clip.track_id)?;
    root_pitch_edit_state(timeline, &clip_root)
}

fn clip_root_track_id(timeline: &TimelineState, clip: &crate::state::Clip) -> Option<String> {
    timeline.resolve_root_track_id(&clip.track_id)
}

fn edit_midi_at_time_or_none(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    abs_time_sec: f64,
) -> Option<f64> {
    if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
        return None;
    }

    let inv_fp = 1000.0 / frame_period_ms.max(0.1);
    let idx_f = abs_time_sec * inv_fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return None;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return None;
    }
    let i0 = i0 as usize;
    if i0 >= pitch_edit.len() {
        return None;
    }
    let i1 = (i0 + 1).min(pitch_edit.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let e0 = pitch_edit.get(i0).copied().unwrap_or(0.0) as f64;
    let e1 = pitch_edit.get(i1).copied().unwrap_or(0.0) as f64;

    let e0 = if e0.is_finite() && e0 > 0.0 {
        Some(e0)
    } else {
        None
    };
    let e1 = if e1.is_finite() && e1 > 0.0 {
        Some(e1)
    } else {
        None
    };

    match (e0, e1) {
        (None, None) => None,
        (Some(v), None) => Some(v),
        (None, Some(v)) => Some(v),
        (Some(a), Some(b)) => {
            let v = a + (b - a) * frac;
            if v.is_finite() && v > 0.0 {
                Some(v)
            } else {
                None
            }
        }
    }
}

fn clip_midi_at_time(
    frame_period_ms: f64,
    clip_start_sec: f64,
    clip_midi: &[f32],
    abs_time_sec: f64,
) -> f64 {
    if !(abs_time_sec.is_finite() && abs_time_sec >= clip_start_sec) {
        return 0.0;
    }

    let local_sec = abs_time_sec - clip_start_sec;
    let inv_fp = 1000.0 / frame_period_ms.max(0.1);
    let idx_f = local_sec * inv_fp;
    if !(idx_f.is_finite() && idx_f >= 0.0) {
        return 0.0;
    }
    let i0 = idx_f.floor() as isize;
    if i0 < 0 {
        return 0.0;
    }
    let i0 = i0 as usize;
    if i0 >= clip_midi.len() {
        return 0.0;
    }
    let i1 = (i0 + 1).min(clip_midi.len().saturating_sub(1));
    let frac = (idx_f - (i0 as f64)).clamp(0.0, 1.0);

    let a = clip_midi.get(i0).copied().unwrap_or(0.0) as f64;
    let b = clip_midi.get(i1).copied().unwrap_or(0.0) as f64;

    let mut a = if a.is_finite() && a > 0.0 { a } else { 0.0 };
    let mut b = if b.is_finite() && b > 0.0 { b } else { 0.0 };
    if a <= 0.0 && b > 0.0 {
        a = b;
    }
    if b <= 0.0 && a > 0.0 {
        b = a;
    }
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }

    let v = a + (b - a) * frac;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn any_user_edit_in_range(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    start_sec: f64,
    end_sec: f64,
) -> bool {
    let fp = frame_period_ms.max(0.1);
    let start_f = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_f = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let end_f = end_f.min(pitch_edit.len());
    if start_f >= end_f {
        return false;
    }

    // 短片段必须密集采样，否则会漏掉尾部编辑点（导致短 clip 被误判为“无编辑”）。
    let span = end_f.saturating_sub(start_f);
    let stride = if span <= 256 {
        1
    } else {
        ((20.0 / fp).round() as usize).max(1) // ~20ms for long regions
    };
    let mut i = start_f;
    let mut last_checked = start_f;
    while i < end_f {
        let v = pitch_edit.get(i).copied().unwrap_or(0.0);
        if v.is_finite() && v > 0.0 {
            return true;
        }
        last_checked = i;
        i += stride;
    }

    // Ensure tail frame is always checked even when stride skips it.
    let tail = end_f.saturating_sub(1);
    if tail != last_checked {
        let v = pitch_edit.get(tail).copied().unwrap_or(0.0);
        if v.is_finite() && v > 0.0 {
            return true;
        }
    }
    false
}

fn any_effective_pitch_change_in_range(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    clip_start_sec: f64,
    clip_midi: &[f32],
    start_sec: f64,
    end_sec: f64,
) -> bool {
    let fp = frame_period_ms.max(0.1);
    let start_f = ((start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
    let end_f = ((end_sec.max(start_sec) * 1000.0) / fp).ceil().max(0.0) as usize;
    let end_f = end_f.min(pitch_edit.len());
    if start_f >= end_f {
        return false;
    }

    // ~100ms sampling is enough to avoid wasting expensive inference.
    // Use a small epsilon to ignore tiny float noise in MIDI curves.
    let eps_semitones = 0.10f64;
    let span = end_f.saturating_sub(start_f);
    let stride = if span <= 256 {
        1
    } else {
        ((20.0 / fp).round() as usize).max(1)
    };

    let mut i = start_f;
    let mut last_checked = start_f;
    while i < end_f {
        let abs_time_sec = (i as f64) * fp / 1000.0;

        let orig = clip_midi_at_time(frame_period_ms, clip_start_sec, clip_midi, abs_time_sec);
        if !(orig.is_finite() && orig > 0.0) {
            i += stride;
            continue;
        }

        let Some(target) = edit_midi_at_time_or_none(frame_period_ms, pitch_edit, abs_time_sec)
        else {
            i += stride;
            continue;
        };

        if !(target.is_finite() && target > 0.0) {
            i += stride;
            continue;
        }

        if (target - orig).abs() > eps_semitones {
            return true;
        }

        last_checked = i;
        i += stride;
    }

    // Ensure tail frame is always checked even when stride skips it.
    let tail = end_f.saturating_sub(1);
    if tail != last_checked {
        let abs_time_sec = (tail as f64) * fp / 1000.0;
        let orig = clip_midi_at_time(frame_period_ms, clip_start_sec, clip_midi, abs_time_sec);
        if orig.is_finite() && orig > 0.0 {
            if let Some(target) =
                edit_midi_at_time_or_none(frame_period_ms, pitch_edit, abs_time_sec)
            {
                if target.is_finite() && target > 0.0 && (target - orig).abs() > eps_semitones {
                    return true;
                }
            }
        }
    }

    false
}

/// v2: Apply pitch edit to a single clip's stereo segment in-place.
///
/// Semantics:
/// - `pitch_edit[t] > 0`: target is absolute MIDI (user-set)
/// - `pitch_edit[t] == 0`: target is the clip's own original MIDI at that time (no change)
///
/// Returns whether processing was applied.
pub fn maybe_apply_pitch_edit_to_clip_segment(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
    seg_start_sec: f64,
    sample_rate: u32,
    pcm_stereo: &mut Vec<f32>,
) -> Result<bool, String> {
    if pcm_stereo.len() < 2 {
        return Ok(false);
    }

    let Some((track, entry)) = clip_pitch_edit_state(timeline, clip) else {
        return Ok(false);
    };
    let child_offset_cfg = active_child_pitch_offset_config(timeline, &clip.track_id);
    let has_child_pitch_offset = child_offset_cfg.is_some();
    if !track.compose_enabled {
        return Ok(false);
    }

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return Ok(false);
    }

    let extra_processing = track_requests_extra_processing(algo, entry, clip);
    let tension_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && hifigan_tension_active_for_clip(entry, clip, clip_start_sec);
    let formant_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && hifigan_formant_shift_active_for_clip(entry, clip, clip_start_sec);

    // 当处理器声明 handles_time_stretch 且 playback_rate != 1.0 时，
    // 即使用户没有编辑音高/张力/共振峰，也需要触发处理器渲染以执行 mel 域拉伸。
    let needs_processor_stretch = {
        let kind = SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let handles = crate::renderer::get_processor(kind)
            .capabilities()
            .handles_time_stretch;
        let rate = (clip.playback_rate as f64).max(1e-6);
        handles && (rate - 1.0).abs() > 1e-6
    };

    // v2 semantics: do nothing until the user actually modified the edit curve.
    // This avoids treating auto-synced `pitch_edit` (e.g. copied from pitch_orig) as an edit.
    // 例外：needs_processor_stretch 时必须进入处理器以执行 mel 拉伸。
    if !entry.pitch_edit_user_modified
        && !extra_processing
        && !tension_processing
        && !formant_processing
        && !has_child_pitch_offset
        && !needs_processor_stretch
    {
        return Ok(false);
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // 预计算处理器能力：决定 seg_end_sec 的时间轴计算方式。
    // - handles_time_stretch=true（如 World/HiFiGAN chain、vslib）：
    //     输入 PCM 为源速率，输出 = 源帧数 / playback_rate（时间轴帧数）
    // - handles_time_stretch=false：输入 PCM 已由外部 Signalsmith Stretch 预拉伸，帧数 = 时间轴帧数
    let kind = SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
    let clip_playback_rate = (clip.playback_rate as f64).max(1e-6);
    let processor_handles_stretch = crate::renderer::get_processor(kind)
        .capabilities()
        .handles_time_stretch;

    // Quick skip when user never set a target in this segment window.
    let seg_frames = pcm_stereo.len() / 2;
    // 输出帧数（时间轴帧数）：内部拉伸时需折算，外部预拉伸时 seg_frames 已是时间轴帧数
    let expected_out_frames = if processor_handles_stretch {
        ((seg_frames as f64) / clip_playback_rate).round().max(2.0) as usize
    } else {
        seg_frames
    };
    // seg_end_sec 始终以时间轴坐标（输出帧）计，确保音高编辑范围检测与声码器上下文一致
    let seg_end_sec = seg_start_sec + (expected_out_frames as f64) / (sample_rate.max(1) as f64);
    let has_pitch_user_edit = any_user_edit_in_range(
        frame_period_ms,
        pitch_edit,
        seg_start_sec,
        seg_end_sec,
    ) || has_child_pitch_offset;
    if !has_pitch_user_edit
        && !extra_processing
        && !tension_processing
        && !formant_processing
        && !has_child_pitch_offset
        && !needs_processor_stretch
    {
        return Ok(false);
    }

    eprintln!(
        "[pitch_edit] clip_id={} algo={:?} seg=[{:.3},{:.3}) compose_enabled={} user_modified={}",
        clip.id,
        algo,
        seg_start_sec,
        seg_end_sec,
        track.compose_enabled,
        entry.pitch_edit_user_modified
    );

    // vslib 使用自身内部分析（ANALYZE_OPTION_VOCAL_SHIFTER），不依赖 WORLD 音高轮廓；
    // 向 VslibSetPitchArray 传递绝对目标音高，不需要原始 MIDI 曲线。
    // 因此对 vslib 跳过 get_or_compute_clip_pitch_midi_global 和 any_effective_pitch_change_in_range，
    // 仅凭 any_user_edit_in_range（已在上方通过）即可触发合成。
    #[cfg(feature = "vslib")]
    let is_vslib = matches!(algo, PitchEditAlgorithm::VocalShifterVslib);
    #[cfg(not(feature = "vslib"))]
    let is_vslib = false;

    // 在渲染前统一构建 clip 输入 pitch 曲线（根轨对应曲线 + 子轨偏移变换）。
    let Some(timeline_midi) = build_clip_input_pitch_curve(
        timeline,
        clip,
        clip_start_sec,
        frame_period_ms,
        clip_playback_rate,
        is_vslib,
    ) else {
        return Ok(false);
    };

    if has_pitch_user_edit && !has_child_pitch_offset && !is_vslib {
        // 若音高编辑值与原始音高完全一致（无实际变化），且不需要其他效果处理，则跳过。
        let has_effective_pitch_change = any_effective_pitch_change_in_range(
            frame_period_ms,
            pitch_edit,
            clip_start_sec,
            &timeline_midi,
            seg_start_sec,
            seg_end_sec,
        );
        if !has_effective_pitch_change
            && !(extra_processing || tension_processing || formant_processing || needs_processor_stretch)
        {
            return Ok(false);
        }
    }

    let mut effective_pitch_edit: Vec<f32> = pitch_edit.to_vec();
    if let Some(ref cfg) = child_offset_cfg {
        for (frame_idx, value) in effective_pitch_edit.iter_mut().enumerate() {
            if *value > 0.0 {
                *value = apply_child_pitch_offset_to_midi(
                    *value as f64,
                    cfg,
                    frame_idx,
                    &timeline.project_scale_notes,
                ) as f32;
            }
        }

        if is_vslib && !timeline_midi.is_empty() {
            let fp = frame_period_ms.max(0.1);
            let start_idx = ((clip_start_sec.max(0.0) * 1000.0) / fp).floor().max(0.0) as usize;
            for (local_idx, midi) in timeline_midi.iter().copied().enumerate() {
                if !(midi.is_finite() && midi > 0.0) {
                    continue;
                }
                let abs_idx = start_idx.saturating_add(local_idx);
                if abs_idx >= effective_pitch_edit.len() {
                    break;
                }
                if effective_pitch_edit[abs_idx] <= 0.0 {
                    effective_pitch_edit[abs_idx] = midi;
                }
            }
        }
    }
    let pitch_edit_for_ctx = effective_pitch_edit.as_slice();

    // stereo -> mono (we don't preserve stereo; use left channel for cheaper conversion)
    let frames = seg_frames;
    // kind / clip_playback_rate / processor_handles_stretch / expected_out_frames 已在函数上方计算

    let processed: Option<Vec<f32>> = MONO_SCRATCH.with(|buf| -> Result<Option<Vec<f32>>, String> {
        let mut mono = buf.borrow_mut();
        mono.clear();
        mono.reserve(frames); // 预分配内存
        // 跨步读取左声道，消除 memset 和越界检查
        mono.extend(pcm_stereo.iter().step_by(2).take(frames).copied());

        // 通过 ClipProcessor trait 调用，解耦合成链路（含音高合成）。
        let processor = crate::renderer::get_processor(kind);
        if !processor.is_available() {
            return Ok(None);
        }

        // 从 TrackParamsState 读取声码器专属曲线/参数（Phase 5 新增字段）
        let extra_curves = &entry.extra_curves;
        let extra_params = &entry.extra_params;

        // 若 Clip 有 clip 级别覆盖，优先使用；否则 fall back 到 track 级别
        let extra_curves: &std::collections::HashMap<String, Vec<f32>> =
            clip.extra_curves.as_ref().unwrap_or(extra_curves);
        let extra_params: &std::collections::HashMap<String, f64> =
            clip.extra_params.as_ref().unwrap_or(extra_params);

        // 若处理器自己处理时间拉伸（如 vslib 使用 Timing 控制点），传递实际 playback_rate；
        // 否则 PCM 已由外部 Signalsmith Stretch 预处理，rate=1.0。
        let ctx_playback_rate = if processor_handles_stretch { clip_playback_rate } else { 1.0 };

        let ctx = crate::renderer::ClipProcessContext {
            mono_pcm: mono.as_slice(),
            sample_rate,
            clip_start_sec,
            seg_start_sec,
            seg_end_sec: seg_start_sec + (expected_out_frames as f64) / (sample_rate.max(1) as f64),
            frame_period_ms,
            pitch_edit: pitch_edit_for_ctx,
            clip_midi: &timeline_midi,
            playback_rate: ctx_playback_rate,
            out_frames: expected_out_frames,
            clip_id: &clip.id,
            extra_curves,
            extra_params,
        };
        if is_vslib {
            eprintln!(
                "[pitch_edit:vslib] dispatch clip_id={} processor={} available={} handles_stretch={} in_frames={} out_frames={} seg=[{:.3},{:.3}) rate={:.3}",
                clip.id,
                processor.id(),
                processor.is_available(),
                processor.capabilities().handles_time_stretch,
                mono.len(),
                expected_out_frames,
                seg_start_sec,
                ctx.seg_end_sec,
                ctx.playback_rate,
            );
        }
        let out = processor.process(&ctx)?;
        if is_vslib {
            let nonzero = out.iter().filter(|&&v| v.abs() > 1e-6).count();
            let peak = out.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            eprintln!(
                "[pitch_edit:vslib] result clip_id={} out_frames={} nonzero={} peak={:.6}",
                clip.id,
                out.len(),
                nonzero,
                peak,
            );
        }
        Ok(Some(out))
    })?;

    let Some(processed) = processed else {
        return Ok(false);
    };

    if processed.len() != expected_out_frames {
        return Err(format!(
            "pitch_edit: output length mismatch (got {}, expected {})",
            processed.len(),
            expected_out_frames
        ));
    }

    // 若输出尺寸与输入不同，调整 Vec 大小并写入
    let stereo_out = expected_out_frames * 2;
    pcm_stereo.clear();
    pcm_stereo.reserve(stereo_out);
    // 消除索引越界检查，批量写入双声道
    for &v in processed.iter().take(expected_out_frames) {
        pcm_stereo.push(v);
        pcm_stereo.push(v);
    }

    Ok(true)
}

#[allow(dead_code)]
pub fn is_pitch_edit_active(timeline: &TimelineState) -> bool {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return false;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return false;
    };
    if !track.compose_enabled {
        return false;
    }

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return false;
    }

    let entry = timeline.params_by_root_track.get(&root);
    let Some(entry) = entry else {
        return false;
    };

    // v2 semantics: pitch edit is considered active only after the user modifies the edit curve.
    entry.pitch_edit_user_modified
}

#[allow(dead_code)]
pub fn is_pitch_edit_backend_available(timeline: &TimelineState) -> bool {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return false;
    };

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return false;
    };

    pitch_edit_backend_available_for_track(track)
}

pub fn semitone_to_ratio(semitones: f64) -> f64 {
    semitone_ratio(semitones)
}

/// 检测指定clip是否需要pitch edit
/// 返回true表示该clip需要pitch edit处理
#[allow(dead_code)]
pub fn does_clip_need_pitch_edit(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    does_clip_need_processor_render(timeline, clip, clip_start_sec)
}

pub fn does_clip_need_processor_render(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let has_child_pitch_offset = active_child_pitch_offset_config(timeline, &clip.track_id).is_some();
    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };

    let Some((track, entry)) = root_pitch_edit_state(timeline, &clip_root) else {
        return false;
    };
    if !track.compose_enabled {
        return false;
    }
    if !pitch_edit_backend_available_for_track(track) {
        return false;
    }

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return false;
    }

    let extra_processing = track_requests_extra_processing(algo, entry, clip);
    let tension_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && hifigan_tension_active_for_clip(entry, clip, clip_start_sec);
    let formant_processing = matches!(algo, PitchEditAlgorithm::NsfHifiganOnnx)
        && hifigan_formant_shift_active_for_clip(entry, clip, clip_start_sec);

    // 当处理器声明 handles_time_stretch 且 playback_rate != 1.0 时，
    // 即使用户没有编辑音高，也需要触发处理器预渲染以执行 mel 域拉伸。
    let needs_processor_stretch = {
        let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let handles = crate::renderer::get_processor(kind)
            .capabilities()
            .handles_time_stretch;
        let rate = (clip.playback_rate as f64).max(1e-6);
        handles && (rate - 1.0).abs() > 1e-6
    };

    // v2 semantics: only treat pitch edit as active after the user modified the edit curve.
    // Otherwise `pitch_edit` may be auto-synced to `pitch_orig` and contain non-zero MIDI values,
    // which should NOT trigger synthesis / prerender.
    // 例外：needs_processor_stretch 时必须触发预渲染以执行 mel 拉伸。
    if !entry.pitch_edit_user_modified
        && !extra_processing
        && !tension_processing
        && !formant_processing
        && !has_child_pitch_offset
        && !needs_processor_stretch
    {
        return false;
    }

    if extra_processing
        || tension_processing
        || formant_processing
        || has_child_pitch_offset
        || needs_processor_stretch
    {
        return true;
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // 检查clip时间范围内是否有用户设置的pitch edit
    // 注意：这里必须使用 clip 在时间线上的可见长度（length_sec），而不是源文件时长（duration_sec）。
    // 否则当 playback_rate < 1（减速拉伸）时，clip 时间线长度会变长，后半段的编辑将不会触发合成。
    let clip_end_sec = clip_start_sec + clip.length_sec.max(0.0);
    any_user_edit_in_range(frame_period_ms, pitch_edit, clip_start_sec, clip_end_sec)
}
