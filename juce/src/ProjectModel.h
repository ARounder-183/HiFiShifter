#pragma once

#include <juce_data_structures/juce_data_structures.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include "backend/UtauOtoOverride.h"
#include "backend/TailFadeSettings.h"
#include "backend/NativeEnvelope.h"
#include "backend/CvvcEnvelope.h"
#include "backend/CvvcTiming.h"
#include "PronunciationTiming.h"
#include "ProjectMaterialRegistrationTypes.h"
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace hachi
{
inline constexpr int inheritedUtauConsonantVelocity =
    std::numeric_limits<int>::min();

struct SampleRegionSetting;
struct VoicebankOtoEntry;

// Native annotation vocabulary shared by every import path.  Input formats
// may retain provenance, but downstream editing does not branch on it.
enum class NativeSegmentRole
{
    unknown,
    consonant,
    vowel,
    transition,
    silence,
    breath,
    noise,
    ending
};
[[nodiscard]] juce::String nativeSegmentRoleName(NativeSegmentRole role);
[[nodiscard]] NativeSegmentRole parseNativeSegmentRole(const juce::String& value);

struct NativeSegment
{
    juce::String id;
    juce::String alias = "-";
    NativeSegmentRole role = NativeSegmentRole::unknown;
    double sourceStartSeconds = 0.0;
    double sourceEndSeconds = 0.0;
    // Native data only: imported/estimated/user.  This is provenance, not an
    // execution mode, and does not select a renderer or an editor layout.
    juce::String provenance = "estimated";
    float confidence = 0.0f;
    double alignmentSeconds = 0.0;
    double overlapSeconds = 0.0;
    bool stretchable = true;
    double stretchWeight = 1.0;
};

struct NativeMaterialAnnotation
{
    int version = 2;
    juce::String materialId;
    juce::File sourceFile;
    juce::File annotationFile;
    std::vector<NativeSegment> segments;
};
// The UTAU synthesis modes.  A track in any of them is PitchAlgorithm::utau,
// so every UTAU affordance is inherited rather than reimplemented; the mode
// only says how a sample is carved up and how a syllable reaches the engine.
//
//   classic  plain UTAU: one sample, the oto's own consonant and stretch
//   jie 界   four regions per sample, each with its own stretch policy
//   mou 谋   the four-region CV of 界, plus a two-region VC -- one written
//            syllable becomes a CV note and a VC note, as UTAU tuning has
//            always done it
enum class UtauMode { classic, jie, mou };

// Four-region data is in play for 界 and 谋 alike: 谋 keeps the four-region
// CV and only adds the VC beside it, so everything that reads oto4 reads it
// for both.  Only the labels tell the two apart.
[[nodiscard]] constexpr bool utauModeUsesRegions(UtauMode mode)
{
    return mode != UtauMode::classic;
}

// 线性flag -- a flag drawn as a curve along the note instead of written once in
// the Flags box -- is available in every UTAU mode. WCSNDM curves do not
// require oto4 data; DiffSinger uses its separate parameter namespace.
[[nodiscard]] constexpr bool utauModeUsesFlagCurves(UtauMode mode)
{
    return mode == UtauMode::classic || mode == UtauMode::jie || mode == UtauMode::mou;
}
[[nodiscard]] juce::String utauModeLabel(UtauMode mode);
// The word written into a project file and accepted over MCP.  "utau4" is
// what 界 has always been called there and still reads.
[[nodiscard]] juce::String utauModeKey(UtauMode mode);
[[nodiscard]] UtauMode parseUtauMode(const juce::String& text);
// Which item of the pitch-algorithm picker a mode is, and back.  Kept in one
// place: the id used to be written out by hand wherever it was needed, and
// adding a third mode left one of those spellings behind, so 谋 came up
// wearing the generic parameter set instead of the UTAU toolbar.
// utauModeForPickerItem answers nothing for an item that is not a UTAU mode.
[[nodiscard]] int utauModePickerItem(UtauMode mode);
[[nodiscard]] std::optional<UtauMode> utauModeForPickerItem(int itemId);

enum class PitchAlgorithm
{
    mld5,
    mld3,
    nsfHifigan,
    world,
    vocalShifter,
    llsm2,
    utau
};

// The two NSF-exclusive variable-hop Mel stretch orders.  Melodyne5 applies
// its pitch/frequency-domain mask to each element before the time-domain
// Catmull-Rom stretch/splice (pitch first, then splice), so the shift-first
// path matches the original algorithm; the splice-first path is the HachiShifter
// variable-mel-hop default and keeps the join blended before formant shifting.
[[nodiscard]] PitchAlgorithm defaultPitchAlgorithm(const juce::File& modelDirectory = {});

enum class StretchAlgorithm
{
    melodyneHybrid,
    variableMelHop,
    loop,
    soundTouch,
    nsfShiftThenSplice,
    hifiShifterMel,
    wcsNonLoop
};

// Where the splice sits relative to the pitch work, for a track that composes.
//  - processThenSplice: every clip is rendered on its own and the results are
//    laid end to end.  Each render is independent, so it is what the cache,
//    cancellation and the per-clip fades are all shaped around.
//  - stretchSpliceThenPitch (the DAW/Melodyne order): clips joined by a
//    Melodyne pitch join are assembled into one phrase first -- their time maps
//    stitched into one continuous map, their pitch lines glided across each
//    seam -- decoded in a single NSF-HiFiGAN pass, and only then cut back into
//    per-clip buffers.  Nothing is spliced after the model has run, so phase,
//    F0 and mel stay continuous through every internal seam.
enum class RenderOrder
{
    processThenSplice,
    stretchSpliceThenPitch
};

struct PitchPoint
{
    double timeSeconds = 0.0;
    float relativeCents = 0.0f;
    float withoutVibratoCents = 0.0f;
    bool voiced = true;
    float manualTargetCents = 0.0f;
    bool hasManualTarget = false;
};

enum class PitchCurveShape
{
    natural,
    linear,
    smooth,
    easeIn,
    easeOut,
    customBezier
};

struct PitchCurveEditPoint
{
    double timeSeconds = 0.0;
    float targetMidi = 60.0f;
    // The shape of the incoming segment: previous point -> this point.
    // The first point's value is intentionally ignored.
    PitchCurveShape shape = PitchCurveShape::natural;
    // Normalised cubic Bezier handles for the incoming segment. X is clamped
    // to [0, 1] so time remains single-valued; Y may leave that range to make
    // controlled pitch overshoot possible.
    float bezierX1 = 0.33f;
    float bezierY1 = 0.0f;
    float bezierX2 = 0.67f;
    float bezierY2 = 1.0f;
    // Internal DS prediction/restoration samples do not expose editable handles.
    bool diffSingerRestoreSupport = false;
    // Exact newly cropped cubics opt into double inversion/evaluation. Legacy
    // and default curves keep their historical float evaluator unchanged.
    bool highPrecisionBezier = false;
};

// Per-note amplitude automation used after the UTAU resampler and before the
// internal wavtool-style overlap mixer.  Times are relative to the nominal
// note start and may be negative to cover the oto.ini preutterance region.
struct AmplitudeEnvelopePoint
{
    double timeSeconds = 0.0;
    float gainDb = 0.0f;
    // The stretch from this point to the next runs straight in amplitude, the
    // way UTAU reads an envelope, instead of straight in dB.  Only presets ask
    // for it; everything drawn by hand, imported or saved before keeps dB.
    bool linearToNext = false;
    // Explicit automation on a linked native cut, rather than a synthesized
    // fragment endpoint. Retained when rebuilding the group's shared curve.
    bool nativeSeamAnchor = false;
};

// The envelope scaled by a base value, in linear percent -- the way UTAU
// counts an envelope, so 200 really is twice as loud and not twice the dB.
// A point at silence stays at silence: nothing multiplied by anything is
// still nothing, and the two ends of a UST envelope are there to be silent.
[[nodiscard]] float scaledEnvelopeGainDb(float gainDb, double factor);
[[nodiscard]] std::vector<AmplitudeEnvelopePoint> scaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent);
// And back again, for writing an edited shape into a note that carries a base.
[[nodiscard]] std::vector<AmplitudeEnvelopePoint> unscaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent);

// A handle on a per-frame flag curve.  Times are relative to the nominal note
// start, exactly like an amplitude envelope point, and may be negative to
// reach into the preutterance; the value is in the flag's own units.  One
// point is a legitimate curve -- it holds that value for the whole note.
struct FlagCurvePoint
{
    double timeSeconds = 0.0;
    float value = 0.0f;
    // The shape of the segment arriving at this point, as on the pitch line,
    // so the first point never has its own shape consulted.  A flag curve is
    // straight lines to begin with; "natural" is a contour-wide shape, is not
    // offered here, and reads as linear if it ever turns up.
    PitchCurveShape shape = PitchCurveShape::linear;
    float bezierX1 = 0.33f;
    float bezierY1 = 0.0f;
    float bezierX2 = 0.67f;
    float bezierY2 = 1.0f;
};

// A shape by name, and back.  Used by the project file, the MCP layer and
// anything else that has to say which shape a segment has.
[[nodiscard]] juce::String pitchCurveShapeName(PitchCurveShape value);
[[nodiscard]] PitchCurveShape parsePitchCurveShape(const juce::String& value);
// The eased progress along one segment, 0..1.  Shared by the pitch line and
// the flag curves, so a shape means the same thing in both.
[[nodiscard]] float shapedSegmentProgress(PitchCurveShape shape, float u,
                                          float bezierX1, float bezierY1,
                                          float bezierX2, float bezierY2);
// A flag curve read at one instant, held flat outside its own handles.
[[nodiscard]] float flagCurveValueAt(const std::vector<FlagCurvePoint>& points,
                                     double timeSeconds);

// One flag's curve on one note.
struct FlagCurve
{
    juce::String flag;
    std::vector<FlagCurvePoint> points;
};

// Which flags can be drawn as a curve, and the range each is drawn on.  The
// ranges are the engine's own clamps, so a handle cannot be put somewhere that
// would not be rendered.  Only flags whose effect is decided frame by frame
// are here: a kernel choice or a whole-note normalisation has no per-frame
// meaning, and the engine says so and ignores it.  The one exception is MY,
// the scream: the engine runs it on the finished mix rather than frame by
// frame, and reads the curve there for its depth.
struct FlagCurveKind
{
    const char* flag;
    const char* label;
    float minimum;
    float maximum;
    // b and bh only ever act on the onset: bh is gated to the onset region by
    // the engine outright, and b acts on unvoiced frames, which a rendered
    // note keeps inside its onset.  Measured on si-/shi-/ban, both leave the
    // vowel within 0.1% (bh) and 4% (b) while moving the onset by 8-99%; the
    // flag that does work across the whole note is Mb, not these two.  So
    // their curves are drawn and edited over the onset alone.
    bool onsetOnly = false;
    float defaultValue = 0.0f;
};
[[nodiscard]] const std::vector<FlagCurveKind>& flagCurveKinds();
[[nodiscard]] const std::vector<FlagCurveKind>& hifisamplerFlagCurveKinds();
[[nodiscard]] const std::vector<FlagCurveKind>& diffSingerFlagCurveKinds();
[[nodiscard]] const std::vector<FlagCurveKind>& diffSingerParameterKinds();
[[nodiscard]] const FlagCurveKind& flagCurveKindFor(const juce::String& flag);
struct NoteData;
// The points of one flag's curve on a note, empty when it has none.
[[nodiscard]] std::vector<FlagCurvePoint> flagCurvePointsFor(const NoteData& note,
                                                             const juce::String& flag);
[[nodiscard]] std::vector<std::pair<juce::String, std::vector<std::pair<double, double>>>>
    sampleDiffSingerFlagCurves(const NoteData& note);

[[nodiscard]] float evaluatePitchCurve(
    const std::vector<PitchCurveEditPoint>& points, double timeSeconds);
// Pure exact cubic crop, with output times local to from. No sampling or
// linearization of shaped segments; false if a singular crop is unrepresentable.
[[nodiscard]] bool sliceNativePitchCurve(const std::vector<PitchCurveEditPoint>&,
    double from, double to, std::vector<PitchCurveEditPoint>& result);
// Compose a local polyline stroke with <=1ms/quarter-span INSIDE edge guards.
// Old boundary values and exact cubic prefix/suffix are preserved. Sparse notes
// return a complete curve/store=true; derived notes return only the stroke
// window/store=false so existing source/target facts outside it are untouched.
// False preserve flags are ONLY artificial clips at the body/authored union
// domain edge, with no old authored support beyond. They use the user's exact
// endpoint Y, so a continuous global stroke never resets at each material seam.
[[nodiscard]] bool composeNativePitchStroke(const NoteData&,
    const std::vector<PitchCurveEditPoint>& localStroke, std::vector<PitchCurveEditPoint>& out,
    bool& storePitchControlPoints, juce::String& error,
    bool preserveLeft = true, bool preserveRight = true);

// A source-time anchor imported from an external editor.  Both values are
// relative to the beginning of the clip: targetSeconds is the position on the
// HachiShifter timeline and sourceSeconds is the corresponding position in the
// selected source-media range.  Keeping these anchors in the native project is
// essential for preserving nonlinear Attack/vowel timing instead of reducing
// every imported element to one linear duration ratio.
struct SourceTimePoint
{
    double targetSeconds = 0.0;
    double sourceSeconds = 0.0;
};

struct NoteData
{
    juce::String id;
    // Original source inside a merged timeline clip; note editing stays flat.
    juce::String clipPartId;
    juce::String label;
    // The native editor consumes these fields for every note, regardless of
    // whether the note came from UST, MPD, MIDI or a hand-created placement.
    // Segment times are note-local after an HJM region is bound to a note.
    NativeSegmentRole nativeRole = NativeSegmentRole::unknown;
    juce::String nativeProvenance = "estimated";
    float nativeConfidence = 0.0f;
    double nativeSourceStartSeconds = -1.0;
    double nativeSourceEndSeconds = -1.0;
    std::vector<NativeSegment> nativeSegments;
    // Identity is (owning clip/part sourceFile, audio ID, region ID). No
    // redundant absolute material root: sourceFile already relocates with media.
    juce::String materialAudioId, materialRegionId;
    // Alias selection is not identity, and a project display label is not a
    // source selection. Primary-follow intent avoids copying a whole old row.
    juce::String materialBoundAlias;
    bool materialAliasFollowsPrimary = true;
    bool materialLabelFollowsAlias = true;
    // Explicit source replacement locks the existing musical score while new
    // acoustic F0 is pending. Also provides a native singleton shared curve.
    bool materialTargetPitchLocked = false;
    // Last complete pronunciation bounds, separate from a split/cropped
    // note's actual source slice. Copied with the reference, not rebound by trims.
    double materialSourceStartSeconds = -1.0, materialSourceEndSeconds = -1.0;
    bool materialLocalOverride = false;
    bool materialReferenceUnresolved = false; // retain last usable snapshot
    juce::StringArray materialAdditionalAliases;
    juce::String materialConnectedNextRegionId;
    PronunciationKind pronunciationKind = PronunciationKind::normal;
    bool simplifiedConsonant = false;
    bool vowelSplitsExplicit = false; // named authoring boundaries, never analysis epochs
    // Passed verbatim as the UTAU resampler's flags argument.  Interpretation
    // belongs to the selected resampler because flag dialects are not uniform.
    // DS phoneme onset overrides; immutable JSON, note-local seconds.
    juce::String diffSingerTiming;
    juce::String diffSingerPronunciation;
    juce::String utauFlags;
    // Every signed integer, including negative values, is a valid resampler
    // velocity.  INT_MIN is reserved internally for the track-wide value.
    int utauConsonantVelocity = inheritedUtauConsonantVelocity;
    // Optional per-note UTAU timing.  Preutterance is the final lead-in after
    // consonant velocity has been applied; overlap is signed so a negative
    // value creates a real gap before this note.
    bool utauPreutteranceOverrideEnabled = false;
    double utauPreutteranceSeconds = 0.0;
    bool utauOverlapOverrideEnabled = false;
    double utauOverlapSeconds = 0.0;
    // STP: how far the whole oto entry is moved inside the recording before a
    // note of it is read.  Positive reads later in the file, negative earlier.
    // Every boundary the entry holds -- offset, overlap, preutterance,
    // consonant, cutoff and the four region lengths -- is measured from the
    // offset, so shifting the offset carries all of them along together.  It
    // changes which audio is played, never where the note sits in the piece.
    double utauStpSeconds = 0.0;
    double utauModulationPercent = 0.0;
    juce::String ustSourceSection, ustBaseline;
    int ustSourceSectionIndex = -1;
    // This note's own oto entry, edited for it alone through 单独OTO编辑.  When
    // enabled the note renders from it instead of its voicebank entry, and the
    // oto files are never touched.  It describes one recording, so a new lyric
    // takes it away.
    backend::UtauOtoOverride utauOto;
    // The note's own oto for a timing lookup, or nothing when it has none.
    [[nodiscard]] const backend::UtauOtoOverride* ownOto() const
    {
        return utauOto.enabled ? &utauOto : nullptr;
    }
    // Manual four-region split, as three cumulative fractions of the note's
    // sounding span (onset|glide, glide|nucleus, nucleus|coda).  Unset means
    // the regions are allocated by weight from the sample's own lengths.
    // Synthetic vibrato, following UTAU's seven VBR parameters.  It is a layer
    // on top of whatever pitch curve the note already has: the curve stays the
    // baseline and the vibrato swings around it.
    bool vibratoEnabled = false;
    double vibratoLengthPercent = 65.0;    // share of the note, measured back from its end
    double vibratoCycleMs = 180.0;         // one full swing
    double vibratoDepthCents = 35.0;
    double vibratoFadeInPercent = 20.0;    // of the vibrato span
    double vibratoFadeOutPercent = 20.0;
    double vibratoPhasePercent = 0.0;      // 100 = one whole cycle
    double vibratoOffsetPercent = 0.0;     // shifts the swing centre, in depths
    // Where the swing stops, as a share of the note from its start.  UTAU's
    // always stops at the note's end, which is 100 and what a UST brings; the
    // length is then measured back from here rather than from the note's end.
    double vibratoEndPercent = 100.0;
    // A crop keeps the existing vibrato clock, including phase and fades.
    double vibratoReferenceDurationSeconds = 0.0;
    double vibratoTimeOffsetSeconds = 0.0;
    // Display only: draw the pitch line with the vibrato already folded in,
    // instead of a flat pitch line plus a separate swing.  Purely how the
    // note is shown; what is rendered is the same either way.
    bool vibratoRealLine = false;
    // Per-region flags for the four-region mode.  Empty entries fall back to
    // the note and track flags, so a split note keeps its ordinary flags
    // wherever it has nothing region-specific to say.
    bool utauFlagSplit = false;
    // Crossfade this note into the one before it at mix time.  Deliberately
    // not a timing override: splicing must not touch preutterance or
    // overlap, which belong to the voicebank and drive the drawn spans.
    bool utauSplice = false;
    // Whether this note and a note that abuts it are joined by the editor's
    // own short S transition.  On for notes made here.  Off for notes read
    // from a UST, whose pitch is what the file says: a note with no bend of
    // its own sits on its own pitch to its edges, as UTAU sings it, and a bend
    // it does carry is sung as written rather than having its ends replaced.
    bool utauAutoPitchTransition = true;
    // Flattening a native note makes its pitch independent, while its source
    // timing/link and shared loudness envelope remain intact.
    bool nativeIndependentPitch = false;
    // Source audio with no detected/authored pitched note. midiNote only
    // positions the editing object; the renderer must not manufacture F0.
    bool nativeUnpitched = false;
    // Once edited, pitch handles (including endpoints) are literal positions.
    // Legacy/just-flattened notes may still derive their initial seam handles.
    bool nativePitchHandlesPlaced = false;
    juce::String utauRegionFlags1, utauRegionFlags2,
                 utauRegionFlags3, utauRegionFlags4;
    // Per-frame ("linear") flags for this note.  While this is on, g comes
    // from the curve below and the g written in the Flags text is ignored;
    // every other flag in that text goes on working as before.
    bool utauFlagCurveEnabled = false;
    std::vector<FlagCurve> utauFlagCurves;
    bool utauJieSplitSet = false;
    double utauJieSplit1 = 0.0;
    double utauJieSplit2 = 0.0;
    double utauJieSplit3 = 0.0;
    double startSeconds = 0.0;
    double durationSeconds = 0.25;
    double consonantSeconds = 0.04;
    // Melodyne may keep an unpitched onset as a complete preceding element.
    // During UTAU conversion it belongs to the following vowel and contributes
    // to that vowel's preutterance rather than becoming an independent note.
    bool melodyneConsonantCandidate = false;
    juce::String melodyneVowelNoteId;
    float midiNote = 60.0f;
    float sourceMidiCenter = -1.0f;
    // Unlike a hand-created target pitch, this centre/contour was measured
    // from the source recording and can prove that a native render is a no-op.
    bool sourcePitchMeasured = false;
    float modulation = 1.0f;
    float drift = 1.0f;
    float tension = 0.0f;
    float breath = 0.0f;
    float formantSemitones = 0.0f;
    float gain = 1.0f;
    float attackSpeed = 1.0f;
    // Melodyne's Robust Pitch Curve is stored by the detector/source.  Hachi
    // exposes the same behaviour per note so difficult notes can opt in
    // without changing the rest of the source analysis.
    bool robustPitchCurve = false;
    bool connectedToPrevious = false;
    bool connectedToNext = false;
    std::vector<PitchPoint> contour;
    // Sparse, user-visible pitch handles used by the UTAU point editor.
    // These are stored separately from the dense 5 ms render contour so a
    // deliberately added collinear handle survives simplification and save/load.
    std::vector<PitchCurveEditPoint> pitchControlPoints;
    // Explicit musical support intent. Geometry/import history alone never
    // opts legacy PBW into extended playback. Extents come from this curve.
    bool pitchSupportAuthored = false;
    // Last accepted DiffSinger prediction, for visual comparison only. Manual
    // pitch edits never write this field; it is not part of synthesis/cache keys.
    std::vector<PitchCurveEditPoint> diffSingerPitchReference;
    // Independent additive semitone curve. Empty means a flat zero offset.
    std::vector<PitchCurveEditPoint> diffSingerPitchOffset;
    // An older file may only contain its current pitch, not a DS prediction.
    bool diffSingerPitchReferenceFromSavedPitch = false;
    std::vector<AmplitudeEnvelopePoint> amplitudeEnvelope;
    backend::CvvcEnvelopeEdits cvvcEnvelopes;
    backend::CvvcBoundaryPositions cvvcBoundaries;
    // Derived phoneme layout, immutable JSON. Saved, but excluded from dirty
    // fingerprints and render keys; never a second pronunciation authority.
    juce::String cvvcPhonemeCache;
    // The envelope's height as a whole, in UTAU's own linear percent: 100 is
    // the envelope as drawn, 200 twice as tall, 0 silence.  A note with no
    // envelope has a flat 100% one, so this raises it just the same.
    float amplitudeEnvelopeBasePercent = 100.0f;
    // OTO-linked amplitude effect: 0 off, 1 linear, 2 smooth curve.
    int utauTailFadeMode = 0;
    backend::TailFadeSettings utauTailFade;
    backend::NativeEnvelopeSettings nativeEnvelope;
    std::vector<double> sibilantMarkers;
};

// Scales an amplitude envelope by a base percent (100 = unchanged), in the
// dB domain, clamped to the same [-60, +12] dB range the UST importer uses.
[[nodiscard]] std::vector<AmplitudeEnvelopePoint> scaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent);
// The inverse: recover the drawn envelope from a scaled one.
[[nodiscard]] std::vector<AmplitudeEnvelopePoint> unscaledAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& points, float basePercent);

[[nodiscard]] float renderedPitchCents(const NoteData& note, const PitchPoint& point);

struct GainEnvelopePoint
{
    double timeSeconds = 0.0; // Relative to the owning region, in seconds.
    float gainDb = 0.0f;
};
using TrackGainPoint = GainEnvelopePoint;

struct NativeSourcePitchPoint
{
    double sourceSeconds = 0;
    float midi = 0, withoutVibratoMidi = 0;
    bool voiced = false;
};

struct ClipData
{
    juce::String id;
    juce::File sourceFile;
    double startSeconds = 0.0;
    double sourceOffsetSeconds = 0.0;
    double sourceDurationSeconds = 0.0;
    double durationSeconds = 1.0;
    // Audio occupies only this part of the clip; the remainder is empty space.
    // -1 preserves the full-clip behaviour of older projects.
    double audioStartSeconds = 0.0;
    double audioDurationSeconds = -1.0;
    [[nodiscard]] double audioLength() const
    { return audioDurationSeconds < 0.0 ? durationSeconds : audioDurationSeconds; }
    double fadeInSeconds = 0.0;
    double fadeOutSeconds = 0.0;
    // Melodyne successive-join amplitude transitions.  These are the actual
    // crossfade mechanism in a Melodyne project: MUSuccessiveJoin.amplitudeTransitionDuration
    // with joinsAmplitudes=true, applied as a LINEAR complementary fade at the
    // join boundary (element shape powers are 1.0).  The legacy fadeInTime/
    // fadeOutTime fields are always NaN in saved projects.
    double crossfadeInSeconds = 0.0;
    double crossfadeOutSeconds = 0.0;
    float gain = 1.0f;
    // Empty means 0 dB. Curves move with this region, independently of its neighbours.
    std::vector<GainEnvelopePoint> gainEnvelope;
    // Retain source envelopes when flattening merged regions. These multiply
    // the editable envelope, preserving overlapping sources without resampling.
    std::vector<std::vector<GainEnvelopePoint>> inheritedGainEnvelopes;
    bool muted = false;
    // Visual-only reference blocks while another region is being edited.
    bool showNoteHints = false;
    // Keep this region fully editable when another region on its track is focused.
    bool showNormalDisplay = false;
    // A Melodyne glide split: two elements joined by followingJoin.joinsPitches.
    // The recording runs continuously through the seam while the two halves are
    // pitched and stretched separately, so under stretchSpliceThenPitch such a
    // chain is rendered as one phrase rather than spliced afterwards.
    bool glideConnectedToNext = false;
    bool glideConnectedFromPrevious = false;
    std::vector<SourceTimePoint> sourceTimeMap;
    // Immutable acoustic reference in full-file coordinates; cropping does
    // not discard hidden source F0. The clock may extend outside this region.
    std::shared_ptr<const std::vector<NativeSourcePitchPoint>> nativeSourcePitch;
    bool nativeSourcePitchComplete = false;
    bool nativeSourcePitchPending = false;
    // Full-file byte fingerprint for explicit per-note source assignment.
    // Pending owners are muted until guarded acoustic facts arrive; resumeMuted
    // stores ONLY the child's own user mute, never an inherited parent mute.
    juce::String materialSourceFingerprint;
    // Actual imported source-use component; registration provenance only.
    // Never inferred from neighbouring times, filenames or pitch-join graphs.
    juce::String materialSourceUsageGroupId;
    bool materialSourceAnalysisPending = false;
    bool materialSourceResumeMuted = false;
    std::vector<SourceTimePoint> nativeTrimClock;
    std::vector<NoteData> notes;
    // Source-only children, with starts relative to this clip. Notes are owned
    // by the parent and refer to these stable ids through clipPartId.
    std::vector<ClipData> parts;
    // Explicitly linked native sources share one editing time warp.
    bool nativeAudioLinked = false;
    // Transient source-expansion metadata; reconstructed from the parent on
    // every expansion, never saved as a second link relationship.
    juce::String nativePitchGroupId;
};

enum class UtauOutputEngine { inherit = 0, resampler = 1, pcNsfHifigan = 2 };
inline juce::String utauOutputEngineKey(UtauOutputEngine engine)
{
    return engine == UtauOutputEngine::pcNsfHifigan ? "pc-nsf-hifigan"
        : engine == UtauOutputEngine::resampler ? "resampler" : "inherit";
}
inline UtauOutputEngine parseUtauOutputEngine(const juce::String& key)
{
    return key == "pc-nsf-hifigan" ? UtauOutputEngine::pcNsfHifigan
        : key == "resampler" ? UtauOutputEngine::resampler : UtauOutputEngine::inherit;
}

enum class TrackType { hachi, utau, diffSinger };
[[nodiscard]] juce::String trackTypeName(TrackType);
[[nodiscard]] std::optional<TrackType> parseTrackType(const juce::String&);

struct TrackData
{
    juce::String id;
    juce::String name;
    bool compose = true;
    bool muted = false;
    bool solo = false;
    // A material track: something to work against rather than part of the
    // piece.  It is heard only while it is the track being worked on, and is
    // silent whenever anything else is playing -- so a reference vocal or a
    // backing take can sit in the project without ever being mixed into it.
    bool referenceOnly = false;
    float volume = 1.0f;
    float pan = 0.0f;
    bool smoothOverlaps = false;
    bool allowNativeAudioOverlap = false;
    bool normalizeVolume = false;
    bool nsfSmoothPitchTransitions = true;
    // Native NSF: transpose harmonics, map source noise and restore clear UV PCM.
    bool nsfNoiseProtection = true;
    // UTAU tracks select samples from this directory instead of using the
    // MIDI clip's sourceFile as audio.  The path is stored per track so one
    // project can use several independent voicebanks.
    juce::File voicebankDirectory;
    juce::String diffSingerLanguage = "zh";
    juce::String diffSingerSpeaker;
    juce::String diffSingerDictionary;
    // UTAU resampler positional parameter 4.  100 preserves the voicebank's
    // original consonant duration; larger values make consonants faster.
    int utauConsonantVelocity = 100;
    juce::String utauGlobalFlags;
    juce::String ustSourceDocument, ustSourceEncoding, ustSourceBytes, ustBaseline;
    bool ustSourceBom = false;
    // The four-region ("Jie/UTAU") variant of the UTAU mode.  It is a flag on
    // the existing utau algorithm rather than a separate PitchAlgorithm value
    // because "is this a UTAU track?" is asked in ~27 places across the piano
    // roll, track list, engine and project model; a second enum value would
    // have to be added to every one of them, and any missed site would
    // silently drop an affordance from the new mode.
    UtauMode utauMode = UtauMode::classic;
    // Opt-in lyric phonemizer; old projects retain literal OTO aliases.
    bool chineseCvvc = false;
    bool utauPianoPreview = true;
    bool cvvcCurveCrossfade = false;
    // MLD5/MLD3 remain loadable for old projects. New native tracks default
    // explicitly to NSF-HiFiGAN; a missing model is reported, not substituted.
    PitchAlgorithm pitchAlgorithm = defaultPitchAlgorithm();
    // Explicit native picker selection must not reactivate legacy NSF+OTO
    // synthesis merely because a previously selected voicebank is remembered.
    bool nativeNsfAudio = false;
    StretchAlgorithm stretchAlgorithm = StretchAlgorithm::melodyneHybrid;
    // Kept at processThenSplice so an existing project, and every track that
    // predates this field, renders exactly as it did before.
    RenderOrder renderOrder = RenderOrder::processThenSplice;
    // Optional same-source decode continuity, also valid in process order.
    bool mergeContiguousSource = false;
    std::vector<ClipData> clips;
    // Backing audio always uses the source reader; engine fields are inert.
    bool accompaniment = false;
    UtauOutputEngine outputEngine = UtauOutputEngine::inherit;
    // Empty per-track paths use global settings, then bundled resampler / internal mixer.
    juce::File outputResampler, outputWavtool;
    // Null exists only for unadopted legacy/in-memory input. Creation, load and
    // replace resolve it once; persisted model tracks always carry one of 3 types.
    std::optional<TrackType> trackType;
};
[[nodiscard]] TrackType effectiveTrackType(const TrackData&);

// One pitch line for UTAU notes whose pitch points reach into one another.
//
// A note's points used to be its own and nothing else's: the first point of a
// UST bend sits before its note, over the tail of the one before, and dragging
// it reshaped only its own note's consonant -- the tail it visibly lay on sang
// on unchanged.  Now, where the next note's bend reaches back into a note, the
// two sing one line, and what decides the pitch at any moment is the points
// drawn there, whichever note they were placed on.
//
// At each moment the line belongs to the latest note whose first point has
// been reached.  Within that stretch the note's own curve is sung exactly as
// it stands, up to its last point there; from that point the line runs
// smoothly into the next note's first point, and on along the next note's
// curve.  Both ends of that join are points on screen: nothing is laid in
// between, since anything worked out from the points either side moves when
// either of them is dragged, and the line never stands still only to drop
// straight down at the hand-over.  A note's points past the
// moment the next one takes over decide nothing any more -- a UST bend often
// runs on for hundreds of milliseconds past its note -- so they are neither
// drawn nor offered, rather than left there to be dragged in vain.
//
// Unedited notes that only touch keep their own lines. Explicit edits meeting
// a boundary, or reaching across one or more adjacent notes, are read by every
// affected note; a receiving note need not own any control points. Rests, gaps
// and independent overlapping voices separate lines -- except a rest shorter
// than 80 ms (a UST's rests come in as gaps) that a bend drawn on both sides
// reaches over.
struct SharedPitchLine
{
    // A stretch of the line: a note's whole curve, over the stretch it owns,
    // or the join from its last point to the next note's first.  Before the
    // first stretch and after the last, the ends are held.
    struct Piece
    {
        double from = 0.0;   // absolute seconds
        std::vector<PitchCurveEditPoint> points;   // absolute seconds, absolute MIDI
    };
    std::vector<Piece> pieces;   // in time order, each running to the next's start
    [[nodiscard]] float midiAt(double absoluteSeconds) const;
    // Where the line has corners, absolute: every point of every piece within
    // its stretch, and every stretch's start.
    [[nodiscard]] std::vector<double> cornersBetween(double from, double to) const;
};
struct SharedPitchLineMember
{
    std::shared_ptr<const SharedPitchLine> line;
    // The stretch this note's own points decide, absolute.  Points outside it
    // are overruled by a later note.  Empty when a later note's bend reaches
    // back past all of this one.
    double ownFrom = -std::numeric_limits<double>::infinity();
    double ownTo = std::numeric_limits<double>::infinity();
    // Where its part of the line is drawn: its own stretch, up to where the
    // next note takes over, so the notes of a line draw it end to end.
    double drawFrom = 0.0;
    double drawTo = 0.0;
    // Where the next note takes over: its first point.
    double takeover = std::numeric_limits<double>::infinity();
    bool joinsPrevious = false;
    bool joinsNext = false;
    bool nativeSharedCurve = false;
    bool renderSharedCurve = false;
    std::vector<double> nativeHandleTimes;
    // Independent native bodies can meet through a 1 ms boundary bridge.
    // These effective handles agree with the rendered piecewise curve.
    std::vector<PitchCurveEditPoint> nativeBoundaryAnchors;
    std::vector<juce::Range<double>> connectionPitchWindows; // derived graph ownership
    // Native request overlay only: when its old base did not render a shared
    // line, consumers must retain the old request outside these graph windows.
    bool connectionPitchOnly = false;
    [[nodiscard]] bool owns(double absoluteSeconds) const
    {
        for(const auto& window:connectionPitchWindows)
            if(absoluteSeconds>=window.getStart()-1.e-9&&absoluteSeconds<=window.getEnd()+1.e-9)return false;
        if (nativeSharedCurve)
            return std::any_of(nativeHandleTimes.begin(), nativeHandleTimes.end(),
                [&](double time) { return std::abs(time - absoluteSeconds) < 1.0e-7; });
        return ownTo > ownFrom && absoluteSeconds >= ownFrom - 1.0e-9
            && absoluteSeconds <= ownTo + 1.0e-9;
    }
};
struct SharedPitchLines
{
    std::map<juce::String, SharedPitchLineMember> byNote;
    [[nodiscard]] const SharedPitchLineMember* memberFor(const juce::String& noteId) const
    {
        const auto found = byNote.find(noteId);
        return found != byNote.end() ? &found->second : nullptr;
    }
};
// The shared lines of one track.  replacedNoteId/replacedPoints stand in for
// that note's placed points, so a point being dragged is seen before it lands.
[[nodiscard]] SharedPitchLines sharedPitchLines(
    const TrackData& track, const juce::String& replacedNoteId = {},
    const std::vector<PitchCurveEditPoint>* replacedPoints = nullptr,
    bool nativeEnvelopeGrouping = false);
// A note's own pitch points, relative to its start: the ones placed on it, or
// the contour it is sung along when it has none.
[[nodiscard]] std::vector<PitchCurveEditPoint> ownPitchPoints(const NoteData& note);
[[nodiscard]] float diffSingerPitchOffsetAt(const NoteData& note, double time);

// Shared eligibility for the model and curve editor: all UTAU modes support
// curves; the voicebank chooses WCSNDM or DiffSinger parameter names.
[[nodiscard]] inline bool trackIsDiffSinger(const TrackData& track)
{
    return !track.accompaniment && effectiveTrackType(track)==TrackType::diffSinger;
}
// OTO timing and regional editing are shared by classic resamplers and native NSF.
// FLAG DSP remains a separate capability. DiffSinger keeps its own renderer.
[[nodiscard]] inline bool trackUsesVoicebankSynthesis(const TrackData& track)
{
    return !track.accompaniment && effectiveTrackType(track)!=TrackType::hachi;
}
// Native recordings are edited as one track-wide timeline. Voicebank and
// source-file editors retain their selected-region scope.
[[nodiscard]] inline bool trackShowsAllNativeRegions(const TrackData& track)
{
    return track.compose && !track.accompaniment && effectiveTrackType(track)==TrackType::hachi;
}
[[nodiscard]] inline UtauOutputEngine effectiveUtauOutputEngine(const TrackData& track,
    UtauOutputEngine defaultEngine = UtauOutputEngine::resampler)
{
    if (trackIsDiffSinger(track)) return UtauOutputEngine::resampler;
    if (track.outputEngine != UtauOutputEngine::inherit) return track.outputEngine;
    // Compatibility for legacy in-memory requests; old project files migrate on load.
    if (track.pitchAlgorithm == PitchAlgorithm::nsfHifigan && trackUsesVoicebankSynthesis(track))
        return UtauOutputEngine::pcNsfHifigan;
    return defaultEngine == UtauOutputEngine::pcNsfHifigan ? defaultEngine : UtauOutputEngine::resampler;
}
[[nodiscard]] inline juce::File effectiveUtauResampler(const TrackData& track, const juce::File& fallback)
{
    return track.outputEngine == UtauOutputEngine::inherit || track.outputResampler == juce::File{} ? fallback : track.outputResampler;
}
[[nodiscard]] inline juce::File effectiveUtauWavtool(const TrackData& track, const juce::File& fallback)
{
    return track.outputEngine == UtauOutputEngine::inherit || track.outputWavtool == juce::File{} ? fallback : track.outputWavtool;
}

[[nodiscard]] inline bool trackTakesFlagCurves(const TrackData& track)
{
    return !track.accompaniment && track.pitchAlgorithm == PitchAlgorithm::utau
        && utauModeUsesFlagCurves(track.utauMode);
}
// A native connection is explicit and can cross source files or HJM regions.
// The legacy note booleans remain as a compatibility projection for existing
// renderers; new editing code can identify both endpoints without guessing
// from their order on a track.
struct NativeConnection
{
    juce::String id;
    juce::String leftNoteId;
    juce::String rightNoteId;
    juce::String type = "pitch-and-amplitude";
    double boundarySeconds = 0.0;
    std::vector<PitchCurveEditPoint> pitchCurve;
    std::vector<AmplitudeEnvelopePoint> amplitudeCurve;
    // Optional forced-portamento encoding. pitchCurve is the SOLE authority:
    // normalized time 0..1, MIDI residual from the live endpoint chord. Older
    // absolute graph curves retain their original units when this flag is false.
    bool pitchCurveEndpointRelative = false;
    double pitchStartOffsetSeconds = -0.05; // from live LEFT note end
    double pitchEndOffsetSeconds = 0.05;    // from live RIGHT musical P anchor
};

struct ResolvedConnectionPitchCurve
{
    juce::String connectionId, leftNoteId, rightNoteId;
    bool active = false;
    juce::String issue;
    double fromSeconds = 0, toSeconds = 0, anchorSeconds = 0;
    std::vector<PitchCurveEditPoint> points; // derived absolute time/MIDI, not stored
};
// Graphless sharedPitchLines remains the legacy oracle. Render AND editor must
// explicitly use this overload to hear/draw forced curves, including other files.
[[nodiscard]] SharedPitchLines sharedPitchLines(const TrackData&,
    const std::vector<NativeConnection>&, const juce::String& replacedNoteId = {},
    const std::vector<PitchCurveEditPoint>* replacedPoints = nullptr);
[[nodiscard]] std::vector<ResolvedConnectionPitchCurve> resolvedConnectionPitchCurves(
    const TrackData&, const std::vector<NativeConnection>&,
    const juce::String& replacedNoteId = {}, const std::vector<PitchCurveEditPoint>* replacedPoints = nullptr);
// Absolute graph stroke -> complete absolute graph controls using the SAME
// exact cubic splice kernel as note strokes. No fake NoteData curve carrier.
// Intersects the live bridge window; outer graph pitches remain score-attached.
// False with empty error is a harmless off-window/tiny no-op.
[[nodiscard]] bool composeConnectionPitchStroke(const TrackData&,
    const std::vector<NativeConnection>&, const juce::String& connectionId,
    const std::vector<PitchCurveEditPoint>& absoluteStroke,
    std::vector<PitchCurveEditPoint>& out, juce::String& error);
// Empty when no explicit incident graph pitch curve exists. Includes live
// anchors/base values/validity so move/transpose invalidates only affected audio.
[[nodiscard]] juce::String connectionPitchDigest(const TrackData&,
    const std::vector<NativeConnection>&, const ClipData& affectedClip);

// Vibrato offset in cents at a time inside the note.  Shared by the piano roll
// and the renderer so what is drawn is what is heard.
double vibratoCentsAt(const NoteData& note, double localSeconds);
// Where a note's vibrato runs, in seconds from the note's start: it stops at
// vibratoEndPercent of the note and reaches back vibratoLengthPercent of the
// note from there, never before the note begins.
struct VibratoSpan { double start = 0.0; double end = 0.0; };
[[nodiscard]] VibratoSpan vibratoSpanOf(const NoteData& note);

struct TempoChange
{
    // Musical position measured in quarter notes from beatOriginSeconds.
    double quarterPosition = 0.0;
    double bpm = 120.0;
};

struct ProjectData
{
    juce::String name = "Untitled";
    double bpm = 120.0;
    double beatOriginSeconds = 0.0;
    int numerator = 4;
    int denominator = 4;
    juce::String gridDivision = "1/16";
    // Smallest unit a note can be stretched or moved by, as 1/N of a beat.
    int noteEditDivision = 64;
    juce::String baseScale = "C";
    bool assetManagerVisible = false;
    std::vector<TempoChange> tempoChanges;
    std::vector<NativeConnection> nativeConnections;
    std::vector<TrackData> tracks;
    // Immutable JSON avoids sharing mutable metadata between undo snapshots.
    juce::String hamoodState;

    [[nodiscard]] double durationSeconds() const;
    [[nodiscard]] double secondsForQuarterPosition(double quarterPosition) const;
    [[nodiscard]] double quarterPositionForSeconds(double seconds) const;
    [[nodiscard]] double tempoAtQuarterPosition(double quarterPosition) const;
    [[nodiscard]] double tempoAtSeconds(double seconds) const;
};

class ProjectModel final : public juce::ChangeBroadcaster
{
public:
    ProjectModel();

    [[nodiscard]] ProjectData snapshot() const;
    [[nodiscard]] std::uint64_t revisionNumber() const;
    void replace(ProjectData replacement);
    // Opening/new documents never share undo history with the previous document.
    void resetDocument(ProjectData replacement = {});
    void clear();
    // Saved document UI state; does not enter musical undo or trigger audio sync.
    void setAssetManagerVisible(bool visible);
    bool setHamoodState(const juce::String& state, juce::String& error);
    bool undo();
    bool redo();
    [[nodiscard]] bool canUndo() const;
    [[nodiscard]] bool canRedo() const;

    [[nodiscard]] juce::String addAudioFile(const juce::File& file, double durationSeconds,
                                            double startSeconds = 0.0,
                                            const juce::String& targetTrackId = {},
                                            const VoicebankOtoEntry* oto = nullptr,
                                            UtauMode otoMode = UtauMode::classic);
    // Insert an authoritative ID selection, not aliases or row indices. The
    // earliest selected source edge lands at absoluteStart (subject to the
    // native collision-placement service). Exterior recording padding is not
    // imported. Source gaps retain their timeline length but are silent: each
    // discontinuous selected run owns an independent source-only child.
    // Overlapping selected source intervals are rejected, never flattened.
    // COMPLETE rows must have stable IDs and valid explicit OTTO links.
    [[nodiscard]] juce::String addMaterialPronunciations(
        const juce::File& audio, const std::vector<SampleRegionSetting>& completeRows,
        const std::vector<juce::String>& selectedRegionIds, double absoluteStart,
        const juce::String& targetTrackId, juce::String& error,
        const juce::String& chosenAlias = {});
    [[nodiscard]] juce::String addTypedTrack(TrackType type = TrackType::hachi,
                                            const juce::String& name = {});
    [[nodiscard]] juce::String addTrack(const juce::String& name, bool compose = true,
                                       bool referenceOnly = false, bool accompaniment = false,
                                       TrackType type = TrackType::hachi);
    void setTrackReferenceOnly(const juce::String& trackId, bool referenceOnly);
    void setTrackName(const juce::String& trackId, const juce::String& name);
    bool setClipNotesIfEmpty(const juce::String& clipId, std::vector<NoteData> notes);
    std::size_t markNativeUnpitchedRegions(const juce::String& clipId);
    bool setNativeTrimSourceAnalysis(const juce::File& file, const std::vector<NoteData>& sourceNotes,
                                     const juce::String& expectedSourceFingerprint = {});
    bool setClipAudioAnalysis(const juce::String& clipId, std::vector<NoteData> notes,
                              const ClipData& importedClip = ClipData{});
    bool addMidiFile(const juce::File& file, juce::String& error);
    // One track of a MIDI file, as the import offers it.
    struct MidiTrackChoice
    {
        int index = 0;         // its place in the file, from 0
        juce::String name;     // the name it gives itself, when it gives one
        int noteCount = 0;
    };
    // The tracks of a MIDI file that have notes on them, in file order.  A
    // conductor track, or any other with nothing to sing, is not one of them.
    [[nodiscard]] static std::vector<MidiTrackChoice> midiTrackChoices(const juce::File& file,
                                                                       juce::String& error);
    // One of them as a new track, the way 新建轨道 makes one, with its notes
    // and their lyrics.  The notes land on the project's beats -- as a UST's
    // do -- and the file's tempo, tempo changes and time signature are taken
    // only when the project has no tracks yet.  One undo takes it away again.
    // With targetTrackId, append a region to that existing melodic track at
    // startSeconds, preserving its settings and the project's tempo map.
    // Returns the destination track's id, or nothing with error set.
    [[nodiscard]] juce::String addMidiTrack(const juce::File& file, int trackIndex,
                                            juce::String& error,
                                            const juce::String& targetTrackId = {}, double startSeconds = 0.0);
    // The same song written back out: every track that has notes, with the
    // tempo map, the time signature and the lyrics, as a type 1 file at 480
    // ticks to the quarter.  Musical time, not seconds -- a project read back
    // into this editor, or opened in anything else, lands on the same beats.
    // Empty trackId exports the whole score. Muted tracks/clips are excluded
    // unless requested; reserved rests and unpitched native regions never sing.
    static bool writeMidiFile(const ProjectData& data, const juce::File& file,
                              juce::String& error, const juce::String& trackId = {},
                              bool includeMuted = false);

    // UST/USTX musical data, imported into an explicit destination family
    // (Hachi by default). Lyrics, timing and source renderer parameters remain
    // preserved independently of the selected destination interface.
    // warnings collects what could not be carried over, which is worth saying
    // out loud rather than leaving it to be noticed missing.
    //
    // A UST is a song, so importing one is opening it: replaceProject leaves
    // nothing of what was open, which is what lets the new song own the tempo
    // map and the project's name.  addTrack lays it beside what is already
    // there instead -- a harmony part beside a lead -- and is what the MCP
    // tool asks for.  Either way one undo brings the previous project back
    // whole.  importedTrackId, when given, comes back naming the new track,
    // because that is the one worth showing afterwards.
    enum class UstImportMode { addTrack, replaceProject };
    bool addUstFile(const juce::File& file, juce::String& error,
                    juce::StringArray& warnings,
                    UstImportMode mode = UstImportMode::addTrack,
                    juce::String* importedTrackId = nullptr, int encoding = 0,
                    TrackType destinationType = TrackType::hachi);
    bool exportUst(const juce::File& file, const juce::String& trackId,
                   juce::String& error, juce::StringArray& warnings, int encoding = 0) const;
    void setTempo(double bpm, int numerator, int denominator = 4);
    // Keep the existing musical-time remapping by default. With false, only
    // the tempo map changes; clips, notes and their curves keep their seconds.
    void setTempoChange(double quarterPosition, double bpm, bool synchronizeNotes = true);
    // Digest of everything the project serialiser writes.  Use this to ask
    // "has anything changed?" -- it cannot miss a field the way a
    // hand-maintained projection of the project can.
    [[nodiscard]] juce::int64 contentFingerprint() const;
    // Library adoption changes references only; all source clocks and music stay exact.
    [[nodiscard]] static juce::String registrationMusicSourceFingerprint(const ProjectData&);
    bool adoptRegisteredMaterialBindings(std::uint64_t expectedRevision,
        const juce::String& expectedFingerprint,
        const std::vector<ProjectMaterialRegistrationOwnerBinding>& owners,
        const std::vector<ProjectMaterialRegistrationNoteBinding>& notes,
        juce::String& error);
    void setGridDivision(const juce::String& division);
    void setNoteEditDivision(int division);
    void setBaseScale(const juce::String& scale);
    void setTrackCompose(const juce::String& trackId, bool enabled);
    void setTrackMuted(const juce::String& trackId, bool muted);
    void setTrackSolo(const juce::String& trackId, bool solo);
    void setTrackVolume(const juce::String& trackId, float volume);
    void setClipGainEnvelope(const juce::String& clipId, std::vector<GainEnvelopePoint> points);
    void setTrackPan(const juce::String& trackId, float pan);
    void setTrackSmoothOverlaps(const juce::String& trackId, bool enabled);
    void setTrackNormalizeVolume(const juce::String& trackId, bool enabled);
    void setTrackNsfSmoothPitchTransitions(const juce::String& trackId, bool enabled);
    void setTrackNsfNoiseProtection(const juce::String& trackId, bool enabled);
    void setTrackVoicebankDirectory(const juce::String& trackId,
                                    const juce::File& directory);
    void setTrackUtauConsonantVelocity(const juce::String& trackId, int velocity);
    void setTrackUtauGlobalFlags(const juce::String& trackId, const juce::String& flags);
    void setUtauMode(UtauMode mode);
    void setTrackUtauMode(const juce::String& trackId, UtauMode mode);
    void setTrackChineseCvvc(const juce::String& trackId, bool enabled);
    void setTrackPianoPreview(const juce::String& trackId, bool enabled);
    void setTrackCvvcCurveCrossfade(const juce::String& trackId, bool curved);
    void setTrackOutputEngine(const juce::String& trackId, UtauOutputEngine engine,
                              const juce::File& resampler = {}, const juce::File& wavtool = {});
    void setNotesVibrato(const std::vector<juce::String>& noteIds,
                         const NoteData& parameters, bool enabled,
                         std::optional<bool> realLine = std::nullopt);
    void setNotesVibratoRealLine(const std::vector<juce::String>& noteIds, bool enabled);
    // Writes the swing into the note as ordinary pitch control points and
    // switches the vibrato off.  One undo step; once the resulting curve has
    // been edited by hand there is no way back to the parametric form.
    bool bakeNoteVibratoIntoPitch(const juce::String& noteId);
    void setNotesUtauSplice(const std::vector<juce::String>& noteIds, bool enabled);
    // All UTAU modes accept curves. Turning them off is allowed on any track,
    // and preserves the stored points for re-enabling or switching backends.
    // One user gesture, one undo entry; delegates parameter normalization to the existing setters.
    struct PointCurveEdit
    {
        enum Kind { pitch, amplitude, flag, cvvc, connectionPitch, connectionPitchStroke };
        Kind kind=pitch; juce::String noteId,key; int part=0;
        std::vector<PitchCurveEditPoint> pitchPoints;
        std::vector<AmplitudeEnvelopePoint> amplitudePoints;
        std::vector<FlagCurvePoint> flagPoints;
        backend::CvvcEnvelopeCurve cvvcPoints;
        bool storePitchControlPoints = true;
        juce::String connectionId; // graph commands only
        // Whole graph-handle batches may explicitly follow new proposed note
        // endpoint pitches. Default remains strict; interior MIDI stays literal.
        bool reanchorConnectionEndpoints = false;
        bool authorPitchSupport = false; // command intent, never inferred from coordinates
    };
    struct PointCurvePlanPreview
    {
        bool changed = false;
        std::vector<TrackData> tracks; // affected real tracks only, original order
        std::vector<NativeConnection> connections;
    };
    // Notes are proposed first, graph commands resolve against that same new
    // score. Any invalid late command rejects ALL edits. One live undo/change.
    bool setPointCurves(std::vector<PointCurveEdit> edits, std::uint64_t expectedRevision,
                        juce::String* error = nullptr);
    bool previewPointCurves(const std::vector<PointCurveEdit>& edits, std::uint64_t expectedRevision,
                           PointCurvePlanPreview&, juce::String& error) const;
    struct CvvcEnvelopeEdit { juce::String noteId; int part; juce::String flag; backend::CvvcEnvelopeCurve points; };
    bool resetCvvcBoundaries(const std::vector<juce::String>& noteIds, std::uint64_t expectedRevision);
    bool setCvvcBoundary(const juce::String& noteId, int rightPart, double position, std::uint64_t expectedRevision);
    bool setCvvcEnvelopes(std::vector<CvvcEnvelopeEdit>, std::uint64_t expectedRevision);
    bool setCvvcEnvelope(const juce::String& noteId, int part, const juce::String& flag,
                         backend::CvvcEnvelopeCurve points, std::uint64_t expectedRevision);
    void setNotesUtauFlagCurveEnabled(const std::vector<juce::String>& noteIds,
                                      bool enabled);
    // Drop every flag curve on the given notes, back to what they looked like
    // the moment the switch went on.  Notes with the switch off are left
    // alone, curves and all: they are not showing a curve to be reset, and a
    // mixed selection should not quietly lose what it is holding for later.
    // Answers whether anything was actually dropped.
    bool resetNotesUtauFlagCurves(const std::vector<juce::String>& noteIds);
    // The same, for one flag: the notes' other curves are left as they are.
    bool resetNotesUtauFlagCurve(const std::vector<juce::String>& noteIds,
                                 const juce::String& flag);
    // An empty list drops that flag's curve; the others are left alone.
    // Points land on 界 and 谋 tracks only; dropping works on any of them.
    bool setNoteUtauFlagCurves(const juce::String& flag,
        const std::map<juce::String, std::vector<FlagCurvePoint>>& edits,
        std::optional<std::uint64_t> expectedRevision = std::nullopt, bool enabledOnly = true);
    bool setNoteUtauFlagCurve(const juce::String& noteId, const juce::String& flag,
                              std::vector<FlagCurvePoint> points);
    void setNotesRegionFlags(const std::vector<juce::String>& noteIds, bool split,
                             const juce::String& first, const juce::String& second,
                             const juce::String& third, const juce::String& fourth);
    // Hands the four regions back to the oto, forgetting a hand-placed split.
    void setNotesUtauJieSplitCleared(const std::vector<juce::String>& noteIds);
    void setNotesUtauJieSplit(const std::vector<juce::String>& noteIds,
                              double first, double second, double third);
    void clearNotesUtauJieSplit(const std::vector<juce::String>& noteIds);
    void setTrackPitchAlgorithm(const juce::String& trackId, PitchAlgorithm algorithm);
    void setTrackStretchAlgorithm(const juce::String& trackId, StretchAlgorithm algorithm);
    void setPitchAlgorithm(PitchAlgorithm algorithm);
    void setStretchAlgorithm(StretchAlgorithm algorithm);
    void setTrackRenderOrder(const juce::String& trackId, RenderOrder order,
                             std::optional<bool> mergeContiguous = {});
    void setRenderOrder(RenderOrder order);
    void moveClip(const juce::String& clipId, double startSeconds);
    [[nodiscard]] bool canSplitClip(const juce::String& clipId, double absoluteSeconds) const;
    // Cut in place. Note tracks snap to the nearest gap without cutting notes.
    // Returns the right piece; both pieces stay on the same track.
    [[nodiscard]] juce::String splitClip(const juce::String& clipId, double absoluteSeconds);
    [[nodiscard]] juce::String duplicateClip(const juce::String& clipId,
                                             double startSeconds = -1.0,
                                             const juce::String& targetTrackId = {});
    void resizeClip(const juce::String& clipId, double startSeconds,
                    double durationSeconds);
    bool trimClip(const juce::String& clipId, double startSeconds, double durationSeconds);
    bool setClipEditableLength(const juce::String& clipId, double durationSeconds);
    void setClipGain(const juce::String& clipId, float gain);
    void setClipFades(const juce::String& clipId, double fadeInSeconds,
                      double fadeOutSeconds);
    void setClipMuted(const juce::String& clipId, bool muted);
    void setClipNoteHints(const juce::String& clipId, bool enabled);
    void setClipNormalDisplay(const juce::String& clipId, bool enabled);
    void removeClip(const juce::String& clipId);
    void removeClips(const std::vector<juce::String>& clipIds);
    [[nodiscard]] bool canMergeClips(const std::vector<juce::String>& clipIds) const;
    [[nodiscard]] juce::String mergeClips(const std::vector<juce::String>& clipIds);
    void moveClips(const std::vector<juce::String>& clipIds, double deltaSeconds, int trackDelta = 0);
    void removeTrack(const juce::String& trackId);
    // Empty beforeTrackId places the track last. IDs remain stable.
    bool moveTrackBefore(const juce::String& trackId, const juce::String& beforeTrackId);
    void transposeNote(const juce::String& noteId, float semitones);
    void transposeNotes(const std::vector<juce::String>& noteIds, float semitones);
    // Move a group of UTAU notes as one phrase.  When the destination phrase
    // overlaps an existing note, the phrase is inserted immediately before
    // the first collision and the following notes are shifted to the right.
    bool moveUtauNotes(const std::vector<juce::String>& noteIds,
                       double deltaSeconds, float semitones, juce::String* error = nullptr);
    // Native timing keeps shared boundaries connected and warps the source
    // clock; an outer note moves rigidly while its neighbour stretches.
    bool moveNativeNotes(const std::vector<juce::String>& noteIds,
                         double deltaSeconds, float semitones, juce::String* error = nullptr);
    bool trimNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge);
    bool resizeNativeNoteEdge(const juce::String& noteId, double deltaSeconds, bool leftEdge);
    // Align the physical preutterance marker (or score onset) to the nearest
    // feasible musical grid, preserving pitch data and committing one undo.
    bool alignNotePreutteranceToGrid(const juce::String& noteId, double quarterStep, juce::String& error);
    void setNotesMidi(const std::vector<juce::String>& noteIds, float midiNote);
    void averageNotesMidi(const std::vector<juce::String>& noteIds);
    void quantizeNotesMidi(const std::vector<juce::String>& noteIds, float stepSemitones = 1.0f);
    // Which notes of a clip a stretch of the timeline lands on.  Pasting asks
    // this before it writes anything, so it can say what it is about to
    // replace.
    [[nodiscard]] std::vector<juce::String> notesOverlapping(
        const juce::String& clipId, double fromSeconds, double toSeconds) const;
    [[nodiscard]] std::vector<juce::String> insertNotes(
        const juce::String& clipId, const std::vector<NoteData>& noteTemplates,
        double absoluteStartSeconds);
    [[nodiscard]] std::vector<juce::String> insertNativeAudioClips(
        const juce::String& trackId, const std::vector<ClipData>& templates,
        const std::vector<NativeConnection>& connections, double atSeconds);
    [[nodiscard]] std::vector<juce::String> duplicateNotes(
        const std::vector<juce::String>& noteIds, const juce::String& targetClipId,
        double absoluteStartSeconds);
    void resizeNote(const juce::String& noteId, double newStart, double newDuration);
    [[nodiscard]] juce::String splitNote(const juce::String& noteId,
                                         double localSeconds);
    [[nodiscard]] juce::String mergeNotes(const std::vector<juce::String>& noteIds);
    [[nodiscard]] bool canJoinNativeNotes(const juce::String&,const juce::String&) const;
    [[nodiscard]] juce::String joinNativeNotes(const juce::String&,const juce::String&);
    // Explicitly connect selected notes without destructively merging their
    // editable data. Pairs may cross clips and source-material regions.
    void setNotesConnection(const std::vector<juce::String>& noteIds, bool enabled);
    // Pitch-only forced connection, NEVER a source join/amplitude splice. Uses
    // existing nativeConnections directional authority and one undo transaction.
    bool setNotesPortamento(const std::vector<juce::String>& noteIds, bool enabled,
                            juce::String& error, double durationSeconds = 0.1);
    // Editable absolute controls from resolvedConnectionPitchCurves. Outer MIDI
    // values stay attached to current note scores; endpoint TIME/curvature and
    // interior MIDI are editable. Mixed PointCurveEdit batches stage real note
    // edits first, then use this validator against the proposed endpoint score.
    bool setConnectionPitchCurve(const juce::String& connectionId,
        std::vector<PitchCurveEditPoint> absolutePoints, juce::String& error);
    bool setConnectionPitchPoint(const juce::String& connectionId, int pointIndex,
        PitchCurveEditPoint absolutePoint, juce::String& error);
    bool disconnectNativeAudio(const std::vector<juce::String>& noteIds);
    bool linkNativeAudio(const std::vector<juce::String>& noteIds);
    bool setNativeAudioOverlap(const juce::String& trackId, bool enabled);
    // Inserts a zero-length lead-in note in front of this one carrying the same
    // lyric, and pins the following note's preutterance to 0 with the given
    // overlap, so a consonant can be sung ahead of the beat.  Returns the new
    // note's id, or empty when the note is already at the clip start.
    [[nodiscard]] juce::String insertPrefixNote(const juce::String& noteId,
                                                double targetOverlapSeconds);
    void setNoteModulation(const juce::String& noteId, float modulation);
    void setNoteDrift(const juce::String& noteId, float drift);
    void setNoteTension(const juce::String& noteId, float tension);
    void setNoteBreath(const juce::String& noteId, float breath);
    void setNoteFormant(const juce::String& noteId, float semitones);
    void setNoteGain(const juce::String& noteId, float gain);
    bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode);
    bool setNotesNativeEnvelope(const std::vector<juce::String>& noteIds, const backend::NativeEnvelopeSettings& settings);
    bool setNotesTailFade(const std::vector<juce::String>& noteIds, int mode,
                          const backend::TailFadeSettings& settings);
    void setNotesAmplitudeEnvelopeBase(const std::vector<juce::String>& noteIds,
                                       float basePercent);
    void setNoteAttack(const juce::String& noteId, double consonantSeconds, float attackSpeed);
    void setNoteAttackSpeed(const juce::String& noteId, float attackSpeed);
    void setNoteLabel(const juce::String& noteId, const juce::String& label);
    // Every Chinese character in the lyrics of a UTAU track turned into its
    // pinyin (see lyricInPinyin), in one undoable step.  Nothing on a track
    // that is not UTAU.  How many notes changed.
    int convertTrackLyricsToPinyin(const juce::String& trackId);
    // Several at once, as one undoable step.  Typing a line of lyrics is one
    // action to the person doing it, so one Ctrl+Z should take it back rather
    // than walking the phrase backwards a syllable at a time.
    void setNoteLabels(const std::vector<std::pair<juce::String, juce::String>>& labels);
    // Convert Chinese lyrics without tying the command to an import source or
    // renderer; callers decide which track/selection to apply it to.

    void setNoteUtauFlags(const juce::String& noteId, const juce::String& flags);
    void setNotesUtauFlags(const std::vector<juce::String>& noteIds,
                           const juce::String& flags);
    void setNotesUtauConsonantVelocity(const std::vector<juce::String>& noteIds,
                                       int velocity);
    // Puts a note of no length immediately in front of this one: 拼字.
    //
    // A note with no body of its own sounds only its lead-in, and the lead-in
    // is the consonant -- the first region.  That is what it is for: a
    // consonant of its own in front of a note that is to be sung as a vowel.
    //
    // It sits on the same beat as the note it leads into and takes that note's
    // pitch and lyric, so it renders something rather than the preview tone
    // until a lyric of its own is typed.  Listed in front of it, and on the
    // same beat the shorter note leads, so the two agree about the order.
    //
    // The note it leads into has its preutterance pinned to nothing, keeping
    // the overlap it had, so its vowel starts on its own beat and the
    // consonant is heard right up to it.  Left as it was, that note went on
    // reaching back over the consonant with its own lead-in and cut it off
    // early -- and its handle sat exactly on top of the new note's, so it could
    // not be taken hold of to change that.  Pinned, it can be dragged back out
    // over the consonant by hand, and crosses it only then.
    //
    // One undoable step for both.  Empty when the note is not found, or when
    // it begins at the very start of its clip -- a lead-in reaches back before
    // the beat, and there is nothing to reach back into there.

    // Gives one note its own oto entry, or takes it away again with one whose
    // enabled is false.  One undoable step.
    void setNoteUtauOto(const juce::String& noteId, const backend::UtauOtoOverride& oto);
    // Classic note-local callback commit: revision, identity and capability
    // checks share the mutation lock. Success may be an honest no-op.
    bool setNoteUtauOtoChecked(const juce::String& noteId, std::uint64_t expectedRevision,
        const backend::UtauOtoOverride& value, std::uint64_t& resultingRevision, juce::String& error);
    // 恢复为音源OTO: these notes go back to their voicebank entries.  Notes with
    // no oto of their own are left alone.  One undoable step, and none at all
    // when there was nothing to restore.
    void clearNotesUtauOto(const std::vector<juce::String>& noteIds);
    // Sets the STP of these notes, in seconds.  One undoable step.
    void setNotesUtauStp(const std::vector<juce::String>& noteIds, double seconds);
    void setNoteUtauTimingOverrides(const juce::String& noteId, bool enabled,
                                    double preutteranceSeconds,
                                    double overlapSeconds);
    void setNoteRobustPitchCurve(const juce::String& noteId, bool enabled);
    bool rememberDiffSingerParameters(const std::map<juce::String, std::vector<FlagCurve>>& curves);
    bool applyDiffSingerParameters(std::uint64_t expectedRevision,
        const std::map<juce::String, std::vector<FlagCurve>>& curves);
    bool setDiffSingerParameterCurve(const juce::String& id, const juce::String& key,
        std::vector<FlagCurvePoint> points);
    bool setDiffSingerPitchOffset(const juce::String& noteId,
                                 std::vector<PitchCurveEditPoint> points);
    bool setNotePitchCurve(const juce::String& noteId,
                           std::vector<PitchCurveEditPoint> points,
                           bool storeControlPoints = false, bool authorPitchSupport = false);
    bool applyDiffSingerTiming(std::uint64_t expectedRevision,
        const std::vector<std::pair<juce::String, juce::String>>& timings);
    bool applyDiffSingerPronunciation(std::uint64_t expectedRevision, const juce::String& trackId,
        const std::vector<std::pair<juce::String, juce::String>>& readings, const juce::String& dictionary);
    bool restoreDiffSingerPitchPoint(std::uint64_t expectedRevision, const juce::String& noteId,
        const std::vector<PitchCurveEditPoint>& displayedPoints, int index);
    bool applyDiffSingerPitch(std::uint64_t expectedRevision,
        const std::vector<std::pair<juce::String, std::vector<PitchCurveEditPoint>>>& curves,
                              bool preserveOutsideNoteHandles = false);
    void setDiffSingerOptions(const juce::String& trackId, const juce::String& language,
                              const juce::String& speaker);
    bool setNoteAmplitudeEnvelope(const juce::String& noteId,
                                  std::vector<AmplitudeEnvelopePoint> points);
    // Resolved sounding spans from the editor; points use absolute seconds,
    // before base gain and automatic OTO mixing. part=0 is a whole note.
    struct CrossfadeSpan
    {
        juce::String noteId;
        int part = 0;
        double from = 0, to = 0;
        std::vector<AmplitudeEnvelopePoint> points;
        double soundingEnd = 0; // optional OTO-mix cutoff, without changing part coordinates
    };
    int crossfadeNotes(std::vector<CrossfadeSpan> spans, std::uint64_t expectedRevision, bool curved = false);
    bool rememberCvvcPhonemes(const std::vector<std::pair<juce::String,juce::String>>& caches,
                             std::uint64_t expectedRevision);
    bool setNotesAmplitudeEnvelopes(
        std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> envelopes);
    // Scales the whole amplitude envelope of these notes by a base percent
    // (100 = unchanged, 200 = twice as loud, 0 = silence) without reshaping it.

    // Where a new note goes, and whether there is room for one at all.
    //
    // Notes on one track are a sequence, not a chord: the UTAU modes splice
    // them end to end and every other mode reads them the same way.  Two that
    // share a moment are not a thicker sound, they are a phrase with no
    // defined order.  So a request landing inside an existing note is
    // refused, and one with room is trimmed to stop where the next begins.
    //
    // It lives here rather than in the tool that calls it because only here
    // are the notes certainly current.  A view holds a snapshot that refreshes
    // on a message, so two quick clicks can both read a list that is one note
    // out of date -- which is one of the ways overlapping notes were made.
    struct NoteSpan { double startSeconds = 0.0; double durationSeconds = 0.0; };
    struct PlannedNote
    {
        bool create = false;
        double startSeconds = 0.0;
        double durationSeconds = 0.0;
    };
    // All times clip-local.  Pure, and public so a check reads the same rule
    // the model applies.
    [[nodiscard]] static PlannedNote plannedNoteFor(double startSeconds,
                                                    double requestedSeconds,
                                                    double clipSeconds,
                                                    const std::vector<NoteSpan>& occupied);
    // An empty stretch of timeline on a melodic track, for notes to be drawn
    // into.  A track starts life with no clips and nothing else in the
    // interface makes one, so without this a new track cannot be drawn on at
    // all.  Returns an empty id for an audio track or an unknown one.
    [[nodiscard]] juce::String addClip(const juce::String& trackId,
                                       double startSeconds, double durationSeconds);
    [[nodiscard]] bool canAddEmptyTuningClip(const juce::String& trackId, double startSeconds) const;
    // Explicitly enables note editing on a non-accompaniment track, in one undo step.
    [[nodiscard]] juce::String addEmptyTuningClip(const juce::String& trackId,
                                                  double startSeconds, double durationSeconds = 10.0);
    // Which of a recording's sidecar regions become notes.
    //
    // One recording usually serves several oto entries: New Geping's a.wav is
    // 0.41 seconds of one syllable with three rows against it -- "a", "a -"
    // and "a" again -- each a different alias over the same sound.  Taken one
    // per row they arrive as three notes stacked on top of each other, which
    // reads as a syllable mysteriously split and, on a track that can only
    // sing one note at a time, leaves all but one of them silent.
    //
    // So overlapping rows are alternatives, not a sequence: the fullest one
    // wins and the rest are dropped.  Rows that do not overlap are separate
    // sounds in one file and are all kept.  Returns the indices to keep, in
    // the order given.  Pure, and public so a check reads the same rule the
    // importer acts on.
    struct RegionSpan { double startSeconds = 0.0; double endSeconds = 0.0; };
    [[nodiscard]] static std::vector<std::size_t> regionsToImport(
        const std::vector<RegionSpan>& regions);
    // The first moment at or after the one asked for that is not already
    // sounding.  Inside a note there is no room for another, so the answer is
    // where that note ends -- and where notes abut, after the last of them.
    // Pure, and public so a check reads the same rule the tool acts on.
    // Lays the pitch line flat on each note's own pitch: every frame
    // targeted at the note -- the unvoiced ones too, so a stretch where
    // nothing was detected moves with the rest instead of keeping whatever it
    // had -- drift and modulation off, and two anchors left.  One undoable
    // step for the lot.
    //
    // This sets where the note is asked to go.  It does not touch the
    // measured curve it is moved away from, and under mld5 the sound is
    // shifted by the difference between the two -- so an octave error in the
    // measurement still comes out as an octave error in the result.
    void flattenNotePitch(const std::vector<juce::String>& noteIds, bool snapNativeToSemitone = false);
    bool restoreNativeSourcePitch(const std::vector<juce::String>& noteIds);
    [[nodiscard]] static double firstFreeStartFrom(double startSeconds,
                                                   const std::vector<NoteSpan>& occupied);
    // Returns an empty id when there is no room, rather than making an overlap.
    [[nodiscard]] juce::String addNote(const juce::String& preferredClipId,
                                       double absoluteStart, double duration, float midiNote);
    // Draws a note starting at the first free moment from here, no longer than
    // asked and no longer than the room before whatever follows.
    //
    // Separate from addNote because the search has to happen under the lock:
    // a view works from a snapshot that refreshes on a message, so it can be a
    // note out of date and would pick a start that is no longer free.
    [[nodiscard]] juce::String addNoteFrom(const juce::String& preferredClipId,
                                           double absoluteFromSeconds,
                                           double maximumDuration, float midiNote);
    void removeNote(const juce::String& noteId);
    void removeNotes(const std::vector<juce::String>& noteIds);
    // Removes the notes and the stretch of time they occupied: whatever
    // followed moves back to meet whatever came before, and the clip loses
    // that much length.  The inverse of pasting a phrase in.
    void removeNotesRippling(const std::vector<juce::String>& noteIds);
    // Open a silence in front of a note: everything from that note on, all
    // through its track, moves later by that much and the track grows by it.
    // The inverse of removeNotesRippling, which closes one up.
    void insertGapBeforeNote(const juce::String& noteId, double seconds);
    // Close a silence in front of a note: that note and everything after it,
    // all through its track, move earlier by that much and the track loses it.
    void closeGapBeforeNote(const juce::String& noteId, double seconds);
    void toggleNoteConnection(const juce::String& noteId);
    struct MaterialUpdateReport
    {
        bool rejected = false;
        int updatedNotes = 0, boundLegacyNotes = 0, unresolvedNotes = 0;
        int skippedLocalOverrides = 0, ambiguousLegacyNotes = 0;
        juce::StringArray diagnostics;
        [[nodiscard]] bool changed() const { return updatedNotes > 0; }
    };
    // rows is the complete current annotation for this source (empty means
    // deleted). Legacy matching uses both source edges within 10 microseconds,
    // never row order or alias. Rejection leaves ALL placements unchanged.
    // New-ID topology is partitioned only for an intact single-source native
    // shared owner and an explicit contiguous chain covering the old interval.
    // The old source clock/coverage remains unchanged; cubic target automation
    // and vibrato phase are sliced exactly. Fully derived pitch stays read-only
    // contour data (no promoted source-sample handles). Mixed authored/derived,
    // local/cropped/multi-source or advanced automation is rejected before publication.
    // apply=false is a publication preflight: identical validation/report,
    // but no project mutation, undo, revision, or notification. changed() then
    // describes the prospective update, not a committed one.
    [[nodiscard]] MaterialUpdateReport applySourceSettingsWithReport(
        const juce::File& source, const std::vector<SampleRegionSetting>& rows,
        bool apply = true);
    void applySourceSettings(const juce::File& source,
                             const std::vector<SampleRegionSetting>& rows);
    void setNoteMaterialLocalOverride(const juce::String& noteId, bool enabled);
    struct NoteMaterialLocalBinding
    {
        juce::String noteId, parentClipId, sourcePartId, audioId, regionId;
        juce::File sourceFile;
        juce::String sourceFingerprint;
        std::uint64_t projectRevision = 0;
        bool bindingResolved = false, nativeFullRow = false, utauScalarTiming = false;
        juce::String diagnostic;
    };
    // Transient absolute-source row projection, never another persisted row.
    // An unbound bank note returns capability/diagnostic, not an alias guess.
    bool getNoteMaterialLocalBinding(const juce::String& noteId,
        NoteMaterialLocalBinding&, SampleRegionSetting& row, juce::String& error) const;
    bool applyNoteMaterialLocalRow(const NoteMaterialLocalBinding&, const SampleRegionSetting&,
        std::uint64_t& resultingRevision, juce::String& error);
    bool restoreNoteMaterialLocalRow(const NoteMaterialLocalBinding&, SampleRegionSetting& restored,
        std::uint64_t& resultingRevision, juce::String& error);
    // Pure representability adapter; no source binding or manifest writes.
    static bool makeClassicNoteMaterialOto(const SampleRegionSetting&,
        backend::UtauOtoOverride&, juce::String& error);
    struct MaterialSourceBindingTicket
    {
        juce::String noteId, parentClipId, sourcePartId;
        juce::File sourceFile;
        juce::String sourceFingerprint;
    };
    struct MaterialSourceReplacementReport
    {
        bool rejected = false, changed = false, pendingAnalysis = false;
        juce::String clipId, sourcePartId;
        MaterialSourceBindingTicket ticket;
        juce::StringArray diagnostics;
    };
    // Explicitly replace ONLY this note's physical source, preserving its ID,
    // score time/pitch/expression and track renderer. apply=false is pure
    // preflight; only a committed report's ticket may be queued for analysis.
    // Native physical-file renderers are supported; voicebank/DS adapters must
    // explicitly implement this capability before they may accept the action.
    [[nodiscard]] MaterialSourceReplacementReport replaceNoteMaterialSource(
        const juce::String& noteId, const juce::File& audio,
        const std::vector<SampleRegionSetting>& completeRows,
        const juce::String& audioId, const juce::String& regionId,
        const juce::String& chosenAlias, juce::String& error, bool apply = true);
    [[nodiscard]] std::vector<MaterialSourceBindingTicket> pendingMaterialSourceAnalyses() const;
    // Full-file acoustic facts only, never old target snapshots. Empty facts
    // reuse the bounded immutable fingerprint cache (e.g. redo after stale B).
    // A stale ticket can warm the cache but cannot resurrect its old binding;
    // only CURRENT pending owners of the same file/fingerprint are updated.
    bool completeMaterialSourceAnalysis(const MaterialSourceBindingTicket& ticket,
        const std::vector<NoteData>& fullFileSourceNotes, juce::String& error);

    bool save(const juce::File& file, juce::String& error) const;
    bool load(const juce::File& file, juce::String& error);
    static bool saveRecoverySnapshot(ProjectData data, const juce::File& recoveryFile,
                                     const juce::File& originalFile, juce::String& error);

private:
    [[nodiscard]] std::optional<double> clipSplitPositionLocked(
        const juce::String& clipId, double absoluteSeconds) const;
    // Exact cuts are internal to trimming; the Split action snaps to note gaps.
    [[nodiscard]] bool canSplitClipAt(const juce::String& clipId, double absoluteSeconds) const;
    [[nodiscard]] juce::String splitClipAt(const juce::String& clipId, double absoluteSeconds);
    void pushUndoLocked();
    bool setNoteUtauOtoLocked(const juce::String& noteId, const backend::UtauOtoOverride& value);
    bool applyNoteMaterialLocalRowInternal(const NoteMaterialLocalBinding&, const SampleRegionSetting&,
        bool restoringSource, std::uint64_t& resultingRevision, juce::String& error,
        SampleRegionSetting* committedRow = nullptr);
    bool planPointCurvesLocked(const std::vector<PointCurveEdit>&, PointCurvePlanPreview&,
                               juce::String& error) const;
    void replaceInternal(ProjectData replacement, bool newDocument);
    // Whether a clip's source is a recording -- audio the engine plays
    // straight out when there are no notes left to play it through.
    //
    // Not simply "has a source file": a UST clip keeps the .ust as its
    // source too, and a score file makes no sound of its own.  Named the
    // other way round -- everything that is not one of the score formats is
    // audio -- so that a new audio format the importer learns is treated
    // correctly without anyone remembering to come back here.
    [[nodiscard]] static bool clipIsRecording(const ClipData& clip);
    // Drops any clip that is a recording and has just lost its last note.
    //
    // The notes are what a recording is played through; with none left the
    // clip renders its source straight out, so deleting every note left the
    // material sounding at full volume with its waveform still drawn --
    // measured at 0.1996 RMS against 0 for no clip at all.  Deleting the
    // notes is how the sound is meant to go away, so the recording goes with
    // them.  A composed clip has nothing to sound and is kept: with no notes
    // it is a silent span of timeline, which is what a new track is drawn
    // into.  Call with the lock, inside an operation that has already pushed
    // an undo step, so the notes and the clip come back together.
    void dropEmptyRecordedClipsLocked();
    // The one place a drawn note is built, so the two entry points cannot
    // drift apart over what a new note starts life as.  Call with the lock.
    [[nodiscard]] juce::String addNoteLocked(ClipData& destination,
                                             const PlannedNote& planned, float midiNote);
    juce::ValueTree toValueTree(const juce::File& projectFile, bool includeDerived = true) const;
    static ProjectData fromValueTree(const juce::ValueTree& tree,
                                     const juce::File& projectFile);
    static juce::String makeId(const char* prefix);

    bool attachSourceAnalysisLocked(const juce::File&, const std::vector<NoteData>&,
                                    const juce::String& guardedFingerprint);
    std::map<juce::String, std::shared_ptr<const std::vector<NoteData>>> materialSourceAnalysisCache;
    mutable juce::CriticalSection lock;
    ProjectData project;
    std::vector<ProjectData> undoHistory;
    std::vector<ProjectData> redoHistory;
    std::uint64_t revision = 0;
    bool suppressNestedUndo = false;
    // Keep large Melodyne projects bounded: snapshots are full native project
    // states, so a short history is preferable to unbounded contour copies.
    static constexpr std::size_t maxHistory = 24;
};
}
