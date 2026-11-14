//! Capture host-link edit merging, kept out of the large session actor module.
use super::*;

pub(super) fn apply(
    session: &EditorSession,
    params: &std::collections::BTreeMap<String, TrackParamsState>,
    tracks: &[Track],
) -> Result<(), String> {
    if !session.is_capture() {
        return Err("host-link edits only apply to Capture sessions".into());
    }
    let known: std::collections::BTreeSet<String> = {
        let timeline = session.timeline.lock_or_recover();
        timeline
            .tracks
            .iter()
            .map(|track| track.id.clone())
            .collect()
    };
    if tracks.len() != known.len()
        || tracks.iter().any(|track| !known.contains(&track.id))
        || params.keys().any(|root| !known.contains(root))
    {
        return Err("invalid or unknown track".into());
    }
    for value in params.values() {
        if !value.frame_period_ms.is_finite() || !(0.1..=100.0).contains(&value.frame_period_ms) {
            return Err("invalid curve track or frame period".into());
        }
    }
    let mut timeline = session.timeline.lock_or_recover().clone();
    timeline.params_by_root_track = params.clone();
    for edited in tracks {
        let Some(track) = timeline
            .tracks
            .iter_mut()
            .find(|track| track.id == edited.id)
        else {
            return Err("invalid or unknown track".into());
        };
        track.volume = edited.volume;
        track.muted = edited.muted;
        track.solo = edited.solo;
        track.compose_enabled = edited.compose_enabled;
        track.pitch_analysis_algo = edited.pitch_analysis_algo.clone();
    }
    *session.timeline.lock_or_recover() = timeline;
    session.request_reload();
    Ok(())
}
