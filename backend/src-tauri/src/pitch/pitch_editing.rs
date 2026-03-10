use crate::state::{PitchAnalysisAlgo, SynthPipelineKind, TimelineState};
use std::cell::RefCell;

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
    Bypass,
}

#[derive(Debug, Clone)]
pub(crate) struct PitchCurvesSnapshot {
    pub frame_period_ms: f64,
    pub pitch_orig: Vec<f32>,
    pub pitch_edit: Vec<f32>,
}

impl PitchCurvesSnapshot {
    #[allow(dead_code)]
    pub fn midi_at_time(&self, abs_time_sec: f64) -> f64 {
        if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
            return 0.0;
        }

        let fp = self.frame_period_ms.max(0.1);
        let idx_f = (abs_time_sec * 1000.0) / fp;
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

pub(crate) fn selected_pitch_curves_snapshot(
    timeline: &TimelineState,
) -> Option<PitchCurvesSnapshot> {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let root = timeline.resolve_root_track_id(&selected)?;

    let entry = timeline.params_by_root_track.get(&root)?;
    Some(PitchCurvesSnapshot {
        frame_period_ms: entry.frame_period_ms.max(0.1),
        pitch_orig: entry.pitch_orig.clone(),
        pitch_edit: entry.pitch_edit.clone(),
    })
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
            PitchAnalysisAlgo::VocalShifterVslib => Self::WorldVocoder,
            PitchAnalysisAlgo::ExternalResampler(_) => Self::Bypass,
            PitchAnalysisAlgo::None => Self::Bypass,
        }
    }
}

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

fn edit_midi_at_time_or_none(
    frame_period_ms: f64,
    pitch_edit: &[f32],
    abs_time_sec: f64,
) -> Option<f64> {
    if !(abs_time_sec.is_finite() && abs_time_sec >= 0.0) {
        return None;
    }

    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_time_sec * 1000.0) / fp;
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
    let fp = frame_period_ms.max(0.1);
    let idx_f = (local_sec * 1000.0) / fp;
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

    let stride = ((100.0 / fp).round() as usize).max(1); // ~100ms
    let mut i = start_f;
    while i < end_f {
        let v = pitch_edit.get(i).copied().unwrap_or(0.0);
        if v.is_finite() && v > 0.0 {
            return true;
        }
        i += stride;
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
    let stride = ((100.0 / fp).round() as usize).max(1);

    let mut i = start_f;
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

        i += stride;
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
    pcm_stereo: &mut [f32],
) -> Result<bool, String> {
    if pcm_stereo.len() < 32 {
        return Ok(false);
    }

    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return Ok(false);
    };

    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return Ok(false);
    };
    if clip_root != root {
        return Ok(false);
    }

    let track = timeline.tracks.iter().find(|t| t.id == root);
    let Some(track) = track else {
        return Ok(false);
    };
    if !track.compose_enabled {
        return Ok(false);
    }

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    if matches!(algo, PitchEditAlgorithm::Bypass) {
        return Ok(false);
    }

    let entry = timeline.params_by_root_track.get(&root);
    let Some(entry) = entry else {
        return Ok(false);
    };

    // v2 semantics: do nothing until the user actually modified the edit curve.
    // This avoids treating auto-synced `pitch_edit` (e.g. copied from pitch_orig) as an edit.
    if !entry.pitch_edit_user_modified {
        return Ok(false);
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // Quick skip when user never set a target in this segment window.
    let seg_frames = pcm_stereo.len() / 2;
    let seg_end_sec = seg_start_sec + (seg_frames as f64) / (sample_rate.max(1) as f64);
    if !any_user_edit_in_range(frame_period_ms, pitch_edit, seg_start_sec, seg_end_sec) {
        return Ok(false);
    }

    // Get per-clip original MIDI curve (full source, source-time indexed).
    let clip_pitch = crate::pitch_clip::get_or_compute_clip_pitch_midi_global(
        timeline,
        clip,
        &root,
        frame_period_ms,
    );
    let Some(clip_pitch) = clip_pitch else {
        return Ok(false);
    };

    // 将源时间索引的 MIDI 曲线 trim+resample 为时间轴对齐的曲线：
    // timeline_midi[0] 对应 clip_start_sec，每帧 frame_period_ms，按 playback_rate 拉伸后的时间轴坐标。
    // 这与前端显示所用的曲线变换完全一致（见 emit_clip_pitch_data_for_clip）。
    let playback_rate = (clip.playback_rate as f64).max(1e-6);
    let timeline_midi = crate::pitch_clip::trim_and_resample_midi(
        &clip_pitch.midi,
        frame_period_ms,
        clip.source_start_sec,
        clip.source_end_sec,
        playback_rate,
        clip.length_sec.max(0.0),
    );
    if timeline_midi.is_empty() {
        return Ok(false);
    }

    // Skip expensive processing if the edit curve does not actually change pitch vs clip's original MIDI.
    if !any_effective_pitch_change_in_range(
        frame_period_ms,
        pitch_edit,
        clip_start_sec,
        &timeline_midi,
        seg_start_sec,
        seg_end_sec,
    ) {
        return Ok(false);
    }

    // stereo -> mono (we don't preserve stereo; use left channel for cheaper conversion)
    let frames = seg_frames;
    let processed: Option<Vec<f32>> = MONO_SCRATCH.with(|buf| -> Result<Option<Vec<f32>>, String> {
        let mut mono = buf.borrow_mut();
        mono.clear();
        mono.resize(frames, 0.0);
        for f in 0..frames {
            mono[f] = pcm_stereo[f * 2];
        }

        // 通过 Renderer trait 调用，解耦合成链路。
        let kind = SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let renderer = crate::renderer::get_renderer(&kind);
        if !renderer.is_available() {
            return Ok(None);
        }

        let ctx = crate::renderer::RenderContext {
            mono_pcm: mono.as_slice(),
            sample_rate,
            seg_start_sec,
            seg_end_sec: seg_start_sec + (frames as f64) / (sample_rate.max(1) as f64),
            clip_start_sec,
            frame_period_ms,
            pitch_edit,
            clip_midi: &timeline_midi,
            clip_id: &clip.id,
        };
        let out = renderer.render(&ctx)?;
        Ok(Some(out))
    })?;

    let Some(processed) = processed else {
        return Ok(false);
    };

    if processed.len() != frames {
        return Err("pitch_edit: output length mismatch".to_string());
    }

    for f in 0..frames {
        let v = processed[f];
        pcm_stereo[f * 2] = v;
        pcm_stereo[f * 2 + 1] = v;
    }

    Ok(true)
}



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

    let algo = PitchEditAlgorithm::from_track_algo(&track.pitch_analysis_algo);
    match algo {
        PitchEditAlgorithm::WorldVocoder => crate::world_vocoder::is_available(),
        PitchEditAlgorithm::NsfHifiganOnnx => crate::nsf_hifigan_onnx::is_available(),
        PitchEditAlgorithm::Bypass => true,
    }
}

pub fn semitone_to_ratio(semitones: f64) -> f64 {
    semitone_ratio(semitones)
}

/// 检测指定clip是否需要pitch edit
/// 返回true表示该clip需要pitch edit处理
pub fn does_clip_need_pitch_edit(
    timeline: &TimelineState,
    clip: &crate::state::Clip,
    clip_start_sec: f64,
) -> bool {
    let selected = timeline
        .selected_track_id
        .clone()
        .or_else(|| timeline.tracks.first().map(|t| t.id.clone()))
        .unwrap_or_default();
    let Some(root) = timeline.resolve_root_track_id(&selected) else {
        return false;
    };

    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };
    if clip_root != root {
        return false;
    }

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

    // v2 semantics: only treat pitch edit as active after the user modified the edit curve.
    // Otherwise `pitch_edit` may be auto-synced to `pitch_orig` and contain non-zero MIDI values,
    // which should NOT trigger synthesis / prerender.
    if !entry.pitch_edit_user_modified {
        return false;
    }

    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let pitch_edit = entry.pitch_edit.as_slice();

    // 检查clip时间范围内是否有用户设置的pitch edit
    // 注意：这里必须使用 clip 在时间线上的可见长度（length_sec），而不是源文件时长（duration_sec）。
    // 否则当 playback_rate < 1（减速拉伸）时，clip 时间线长度会变长，后半段的编辑将不会触发合成。
    let clip_end_sec = clip_start_sec + clip.length_sec.max(0.0);
    any_user_edit_in_range(frame_period_ms, pitch_edit, clip_start_sec, clip_end_sec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{PitchAnalysisAlgo, TimelineState, Track, TrackParamsState};

    fn make_timeline_with_pitch_edit(
        frame_period_ms: f64,
        pitch_edit_user_modified: bool,
        pitch_edit_len: usize,
        edited_times_sec: &[f64],
    ) -> TimelineState {
        let mut pitch_edit = vec![0.0f32; pitch_edit_len];
        for &t in edited_times_sec {
            let i = ((t.max(0.0) * 1000.0) / frame_period_ms.max(0.1)).round().max(0.0) as usize;
            if i < pitch_edit.len() {
                // any positive value counts as a user-set target
                pitch_edit[i] = 60.0;
            }
        }

        let root_id = "track_root".to_string();
        let track = Track {
            id: root_id.clone(),
            name: "Root".to_string(),
            parent_id: None,
            order: 0,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: true,
            pitch_analysis_algo: PitchAnalysisAlgo::World,
            color: String::new(),
        };

        let mut tl = TimelineState {
            tracks: vec![track],
            clips: vec![],
            selected_track_id: Some(root_id.clone()),
            selected_clip_id: None,
            bpm: 120.0,
            playhead_sec: 0.0,
            project_sec: 60.0,
            params_by_root_track: Default::default(),
            next_track_order: 1,
        };

        tl.params_by_root_track.insert(
            root_id,
            TrackParamsState {
                pitch_edit,
                pitch_edit_user_modified,
                frame_period_ms,
                ..Default::default()
            },
        );
        tl
    }

    #[test]
    fn does_clip_need_pitch_edit_uses_timeline_length_not_source_duration() {
        let frame_period_ms = 5.0;

        // Create an edit at t=6s (in the stretched tail).
        let mut tl = make_timeline_with_pitch_edit(frame_period_ms, true, 20_000, &[6.0]);

        let clip = crate::state::Clip {
            id: "clip1".to_string(),
            track_id: "track_root".to_string(),
            name: "c".to_string(),
            start_sec: 0.0,
            length_sec: 8.0, // stretched on timeline
            color: String::new(),
            source_path: None,
            duration_sec: Some(4.0), // original source duration (compat)
            duration_frames: None,
            source_sample_rate: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 4.0,
            playback_rate: 0.5,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
        };

        // Place clip into timeline so root resolution works.
        tl.clips.push(clip.clone());

        assert!(does_clip_need_pitch_edit(&tl, &clip, 0.0));
    }
}
