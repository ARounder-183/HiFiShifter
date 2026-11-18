//! Unified phrase-first voicebank & companion `*.marks.json` contract.
//!
//! Shared across:
//! 1. External voicebank preparation / forced-alignment CLIs (`auto_draft` / `needs_review`),
//! 2. HiFiShifter's built-in Voicebank Calibration Workspace (`confirmed` / `disabled` + tags),
//! 3. `Ctrl+F` quick vowel-first / VC-transition search, and
//! 4. Full-phrase Agent arrangement planners feeding `hfs_edit` (`ImportClip` / `ImportCandidateTakes`).
//!
//! Single-sample legacy voicebanks (`ka.wav`) are represented identically as a
//! degenerate 1-syllable `PhraseMarks` file (`ka.marks.json`), eliminating any
//! split between single-note JSONL indices and phrase-level mark sidecars.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const VOICEBANK_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReviewState {
    #[default]
    AutoDraft,
    NeedsReview,
    Confirmed,
    Disabled,
}

impl ReviewState {
    /// Whether this item is protected from being overwritten by a re-run of automated alignment.
    pub fn is_human_locked(self) -> bool {
        matches!(self, Self::Confirmed | Self::Disabled)
    }

    /// Whether ordinary candidate search (`Ctrl+F` or Agent planner) may select this item.
    pub fn is_selectable(self, allow_unconfirmed_auto: bool) -> bool {
        match self {
            Self::Confirmed => true,
            Self::AutoDraft => allow_unconfirmed_auto,
            Self::NeedsReview | Self::Disabled => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum MarkProvenance {
    #[default]
    Auto,
    Manual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PhoneRole {
    Onset,
    Nucleus,
    Coda,
    Silence,
    Breath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    /// Natural cross-syllable VC transition (e.g. `a -> t` inside `ka ta`).
    CrossSyllableVc,
    /// Intra-syllable vowel-to-coda transition (e.g. `a -> n` inside `kan`).
    IntraSyllableVc,
    /// Vowel-to-vowel continuous transition (e.g. `a -> i`).
    Vv,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VoicebankManifest {
    pub schema_version: u32,
    pub library_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub singer: Option<String>,
    pub default_language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aligner_revision: Option<String>,
    pub phrases: Vec<PhraseIndexEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhraseIndexEntry {
    pub phrase_id: String,
    pub wav_rel_path: String,
    pub marks_rel_path: String,
    pub content_sha256: String,
    pub duration_sec: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyric_summary: Option<String>,
    #[serde(default)]
    pub review_state: ReviewState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhraseMarks {
    pub schema_version: u32,
    pub phrase_id: String,
    pub wav_rel_path: String,
    pub content_sha256: String,
    pub sample_rate: u32,
    pub duration_sec: f64,
    pub language: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyric: Option<String>,
    #[serde(default)]
    pub review_state: ReviewState,
    #[serde(default)]
    pub phones: Vec<PhoneSegment>,
    #[serde(default)]
    pub syllables: Vec<SyllableSegment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<TransitionSegment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<ScopedTag>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhoneSegment {
    pub id: String,
    pub label: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub role: PhoneRole,
    #[serde(default)]
    pub provenance: MarkProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SyllableSegment {
    pub id: String,
    pub reading: String,
    pub start_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub phone_ids: Vec<String>,
    /// Vowel anchor (`A`) used for beat alignment and vowel-first crossfades.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vowel_onset_sec: Option<f64>,
    /// Safe time-stretch sub-spans inside the vowel nucleus (`[start_sec, end_sec]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stable_vowel_regions_sec: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f0_median_midi: Option<f32>,
    #[serde(default)]
    pub review_state: ReviewState,
    #[serde(default)]
    pub provenance: MarkProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TransitionSegment {
    pub id: String,
    /// Searchable transition token such as `"at"` (from `ka` -> `ta`).
    pub reading: String,
    pub kind: TransitionKind,
    pub from_syllable_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_syllable_id: Option<String>,
    pub start_sec: f64,
    /// Acoustic handoff anchor (e.g. consonant closure / onset start).
    pub boundary_sec: f64,
    pub end_sec: f64,
    #[serde(default)]
    pub review_state: ReviewState,
    #[serde(default)]
    pub provenance: MarkProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScopedTag {
    pub tag: String,
    pub scope: TagScope,
    #[serde(default)]
    pub provenance: MarkProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TagScope {
    Phrase,
    Syllable { syllable_id: String },
    Span { start_sec: f64, end_sec: f64 },
}

fn check_span(start: f64, end: f64, duration: f64, label: &str) -> Result<(), String> {
    if !start.is_finite()
        || !end.is_finite()
        || start < 0.0
        || end <= start
        || end > duration + 1e-6
    {
        return Err(format!(
            "{label} span [{start}, {end}) must be finite and within [0, {duration}]"
        ));
    }
    Ok(())
}

impl VoicebankManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != VOICEBANK_SCHEMA_VERSION {
            return Err(format!(
                "unsupported voicebank schema_version {} (expected {})",
                self.schema_version, VOICEBANK_SCHEMA_VERSION
            ));
        }
        if self.library_id.trim().is_empty() || self.name.trim().is_empty() {
            return Err("library_id and name must be non-empty".into());
        }
        let mut seen = HashSet::new();
        for entry in &self.phrases {
            if entry.phrase_id.trim().is_empty() || !seen.insert(entry.phrase_id.as_str()) {
                return Err(format!(
                    "duplicate or empty phrase_id '{}'",
                    entry.phrase_id
                ));
            }
            if !entry.duration_sec.is_finite() || entry.duration_sec <= 0.0 {
                return Err(format!("invalid duration_sec for '{}'", entry.phrase_id));
            }
        }
        Ok(())
    }
}

impl PhraseMarks {
    /// Structural and coordinate-domain validation against the phrase WAV duration.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != VOICEBANK_SCHEMA_VERSION {
            return Err(format!(
                "unsupported marks schema_version {} (expected {})",
                self.schema_version, VOICEBANK_SCHEMA_VERSION
            ));
        }
        if self.sample_rate == 0 || !self.duration_sec.is_finite() || self.duration_sec <= 0.0 {
            return Err("sample_rate and duration_sec must be positive".into());
        }
        let mut phone_ids = HashSet::new();
        let mut prev_phone_end = 0.0f64;
        for phone in &self.phones {
            if phone.id.trim().is_empty() || !phone_ids.insert(phone.id.as_str()) {
                return Err(format!("duplicate or empty phone id '{}'", phone.id));
            }
            check_span(phone.start_sec, phone.end_sec, self.duration_sec, &phone.id)?;
            if phone.start_sec + 1e-6 < prev_phone_end {
                return Err(format!("overlapping phone '{}'", phone.id));
            }
            prev_phone_end = phone.end_sec;
        }
        let mut syllable_ids = HashSet::new();
        for syl in &self.syllables {
            if syl.id.trim().is_empty() || !syllable_ids.insert(syl.id.as_str()) {
                return Err(format!("duplicate or empty syllable id '{}'", syl.id));
            }
            check_span(syl.start_sec, syl.end_sec, self.duration_sec, &syl.id)?;
            for pid in &syl.phone_ids {
                if !phone_ids.contains(pid.as_str()) {
                    return Err(format!(
                        "syllable '{}' references unknown phone_id '{}'",
                        syl.id, pid
                    ));
                }
            }
            if let Some(vo) = syl.vowel_onset_sec {
                if !vo.is_finite() || vo < syl.start_sec - 1e-6 || vo > syl.end_sec + 1e-6 {
                    return Err(format!(
                        "vowel_onset_sec {vo} lies outside syllable '{}'",
                        syl.id
                    ));
                }
            }
            for &[rs, re] in &syl.stable_vowel_regions_sec {
                check_span(rs, re, self.duration_sec, "stable_vowel_region")?;
                if rs < syl.start_sec - 1e-6 || re > syl.end_sec + 1e-6 {
                    return Err(format!(
                        "stable_vowel_region [{rs}, {re}) exceeds syllable '{}'",
                        syl.id
                    ));
                }
            }
        }
        for tr in &self.transitions {
            check_span(tr.start_sec, tr.end_sec, self.duration_sec, &tr.id)?;
            if !tr.boundary_sec.is_finite()
                || tr.boundary_sec < tr.start_sec
                || tr.boundary_sec > tr.end_sec
            {
                return Err(format!("transition '{}' boundary_sec out of range", tr.id));
            }
            if !syllable_ids.contains(tr.from_syllable_id.as_str()) {
                return Err(format!(
                    "transition '{}' references unknown from_syllable_id '{}'",
                    tr.id, tr.from_syllable_id
                ));
            }
            if let Some(to_id) = &tr.to_syllable_id {
                if !syllable_ids.contains(to_id.as_str()) {
                    return Err(format!(
                        "transition '{}' references unknown to_syllable_id '{}'",
                        tr.id, to_id
                    ));
                }
            }
        }
        for tag in &self.tags {
            if tag.tag.trim().is_empty() {
                return Err("tag name must be non-empty".into());
            }
            match &tag.scope {
                TagScope::Phrase => {}
                TagScope::Syllable { syllable_id } => {
                    if !syllable_ids.contains(syllable_id.as_str()) {
                        return Err(format!("tag references unknown syllable '{syllable_id}'"));
                    }
                }
                TagScope::Span { start_sec, end_sec } => {
                    check_span(*start_sec, *end_sec, self.duration_sec, "tag span")?;
                }
            }
        }
        Ok(())
    }

    /// Re-running automated alignment must NEVER overwrite human-confirmed,
    /// human-disabled, or manually edited boundaries and tags.
    pub fn merge_auto_draft(&self, mut fresh_auto: PhraseMarks) -> PhraseMarks {
        if self.review_state.is_human_locked() || self.content_sha256 != fresh_auto.content_sha256 {
            return self.clone();
        }
        let manual_phones: HashMap<&str, &PhoneSegment> = self
            .phones
            .iter()
            .filter(|p| p.provenance == MarkProvenance::Manual)
            .map(|p| (p.id.as_str(), p))
            .collect();
        for phone in &mut fresh_auto.phones {
            if let Some(existing) = manual_phones.get(phone.id.as_str()) {
                *phone = (*existing).clone();
            }
        }

        let locked_syllables: HashMap<&str, &SyllableSegment> = self
            .syllables
            .iter()
            .filter(|s| s.provenance == MarkProvenance::Manual || s.review_state.is_human_locked())
            .map(|s| (s.id.as_str(), s))
            .collect();
        for syl in &mut fresh_auto.syllables {
            if let Some(existing) = locked_syllables.get(syl.id.as_str()) {
                *syl = (*existing).clone();
            }
        }

        let locked_transitions: HashMap<&str, &TransitionSegment> = self
            .transitions
            .iter()
            .filter(|t| t.provenance == MarkProvenance::Manual || t.review_state.is_human_locked())
            .map(|t| (t.id.as_str(), t))
            .collect();
        for tr in &mut fresh_auto.transitions {
            if let Some(existing) = locked_transitions.get(tr.id.as_str()) {
                *tr = (*existing).clone();
            }
        }

        // Preserve all manual tags without duplication.
        for tag in &self.tags {
            if tag.provenance == MarkProvenance::Manual && !fresh_auto.tags.contains(tag) {
                fresh_auto.tags.push(tag.clone());
            }
        }
        if self.lyric.is_some() {
            fresh_auto.lyric = self.lyric.clone();
        }
        fresh_auto
    }
}
