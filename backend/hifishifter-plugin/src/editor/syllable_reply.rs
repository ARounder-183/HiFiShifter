use hifishifter_kernel::pitch_analysis::{
    TakeSyllableAnalysis, SYLLABLE_ALGORITHM, SYLLABLE_ALGORITHM_VERSION,
};
use serde_json::{json, Value};

/// Every analysis reply carries a valid provenance window, even when no cached
/// analysis exists. The frontend reducer is strict and rejects null windows.
pub fn syllable_reply_payload(
    clip_id: &str,
    take_id: &str,
    analysis: Option<&TakeSyllableAnalysis>,
    window: (f64, f64),
) -> Value {
    json!({
        "ok": true,
        "clipId": clip_id,
        "takeId": take_id,
        "sourceSpansSec": analysis.map_or(Vec::new(), |a| a.source_spans_sec.iter()
            .map(|s| json!({"startSec": s.start_sec, "endSec": s.end_sec, "midiNote": s.midi_note})).collect()),
        "sourceNotes": analysis.map_or(Vec::new(), |a| a.source_spans_sec.iter().map(|s| s.midi_note).collect()),
        "analyzedWindowSec": analysis.map_or([window.0.min(window.1), window.0.max(window.1)], |a| a.analyzed_window_sec),
        "algorithm": analysis.map(|a| a.algorithm.clone()).unwrap_or_else(|| SYLLABLE_ALGORITHM.to_string()),
        "algorithmVersion": analysis.map(|a| a.algorithm_version).unwrap_or(SYLLABLE_ALGORITHM_VERSION),
        "analyzed": analysis.is_some(),
    })
}
