#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include "UtauOtoOverride.h"
#include "TailFadeSettings.h"
#include "CvvcEnvelope.h"
#include "CvvcTiming.h"
#include "UtauMixEnvelope.h"
#include "../WavExportOptions.h"
#include <array>
#include <functional>
#include <optional>
#include <vector>
#include <memory>

namespace hachi { struct VoicebankOtoEntry; }

namespace hachi::backend
{
struct UtauPitchPoint
{
    double timeSeconds = 0.0;
    float cents = 0.0f;
};

struct UtauAmplitudePoint
{
    double timeSeconds = 0.0;
    float gainDb = 0.0f;
    // As AmplitudeEnvelopePoint::linearToNext.
    bool linearToNext = false;
    std::optional<EnvelopeBezier> bezier;
};

struct UtauNoteRenderSpec
{
    juce::String alias;
    juce::String flags;
    double startSeconds = 0.0;
    double durationSeconds = 0.25;
    float midiNote = 60.0f;
    float gain = 1.0f;
    std::vector<UtauPitchPoint> pitchCurve;
    std::vector<UtauAmplitudePoint> amplitudeEnvelope;
    int consonantVelocity = 100;
    bool preutteranceOverrideEnabled = false;
    double preutteranceSeconds = 0.0;
    bool overlapOverrideEnabled = false;
    double overlapSeconds = 0.0;
    // Local tempo at this note's musical start.  Zero uses request.bpm for
    // compatibility with older callers and smoke fixtures.
    double bpm = 0.0;
    // Manual four-region split for this note, as three cumulative fractions
    // of the sounding span.  Unset lets the regions be allocated by weight.
    // Declared last: several callers build this struct positionally.
    bool jieSplitSet = false;
    std::array<double, 3> jieSplit {};
    // Per-region flags; empty entries mean the region uses the note flags.
    bool flagSplit = false;
    std::array<juce::String, 4> regionFlags {};
    // A per-frame curve for g.  Times are relative to the nominal note start,
    // like the amplitude envelope; the engine measures from the start of the
    // rendered segment, which is a preutterance earlier.  While this is on the
    // g written in the flags text is ignored.
    bool flagCurve = false;
    // Already sampled: a shaped segment is handed over as enough plain points
    // for the engine's straight-line reading to follow it.  One entry per flag
    // that has a curve, named as the engine names it.
    std::vector<std::pair<juce::String, std::vector<std::pair<double, double>>>>
        flagCurves;
    // Crossfade into the note before this one at mix time.  Affects only
    // the mix, never the render, so it is not part of a note's cache key.
    bool splice = false;
    // STP: the whole oto entry is moved this far inside the recording before
    // anything is read from it.  Positive reads later in the file, negative
    // earlier.  Declared last, like the fields above it, because several
    // callers build this struct positionally.
    double stpSeconds = 0.0;
    // This note's own oto entry, when it has one: rendered from in place of
    // the voicebank's.  The STP above moves whichever entry that is.
    UtauOtoOverride oto;
    // Immutable timeline evaluator. Ownership of editing handles must not
    // restrict which sounding note can read this curve. Local time is relative
    // to the nominal note start, including negative lead-in and overlap tails.
    std::function<float(double)> timelinePitchCents;
    juce::String diffSingerTiming;
    juce::String diffSingerPronunciation;
    juce::String diffSingerContext;
    int tailFadeMode = 0;
    TailFadeSettings tailFadeSettings;
    double modulationPercent = 0.0;
    // Generated phonemes retain the parent note's envelope clock.
    bool preserveEnvelopeTiming = false;
    bool aliasIsResolved = false;
    CvvcEnvelopeEdits cvvcEnvelopes;
    float cvvcDefaultEnvelopeGain = 1.0f; // base volume when no user envelope exists
    bool cvvcStandardEnvelope = false;
    CvvcBoundaryPositions cvvcBoundaries;
    // Saved native note controls, in their native units (not FLAG g/Ht/Hb).
    // Consumed by the native NSF voicebank backend; external UTAU engines do
    // not implement these editor controls. Append to preserve aggregate callers.
    float nativeFormantSemitones = 0.0f;
    float nativeTension = 0.0f;
    float nativeBreath = 0.0f;
};

struct UtauPhonemeSpan
{
    juce::String token, kind;
    double startSeconds = 0, endSeconds = 0; // relative to the MIDI note
    int envelopePart = 0;
    double leadInSeconds = 0;
    UtauMixEnvelope mixEnvelope;
};

// legacyAuto is only for historical direct backend callers. Host score tracks
// carry their chosen family; bank files validate capability, never select it.
enum class RequestedVoicebankFamily { legacyAuto, utau, diffSinger };

struct UtauRenderRequest
{
    // Preview may audition unlabelled notes; exports must leave them silent.
    bool silenceEmptyLyrics = false;
    bool pianoPreviewEnabled = true;
    bool chineseCvvc = false;
    bool cvvcCurveCrossfade = false;
    bool cvvcPhonemes = false; // set by the CVVC adapter after expanding score notes
    juce::File voicebankDirectory;
    juce::String diffSingerLanguage = "zh";
    juce::String diffSingerSpeaker;
    juce::String diffSingerDictionary;
    juce::String diffSingerCacheSession;
    juce::var diffSingerInference; // immutable settings snapshot for queued renders
    std::function<bool()> cancelled;
    juce::File resamplerExecutable;
    juce::File wavtoolExecutable;
    bool requireExternalResampler = false;
    bool fourRegion = false;
    // 谋•UTAU: read 谋•OTO and hand the engine the per-region classes.  Off
    // for UTAU and 界•UTAU, which have no such annotation -- with it off the
    // engine is sent exactly what it was sent before.
    bool consonantClasses = false;
    double targetDurationSeconds = 0.0;
    double bpm = 120.0;
    std::vector<UtauNoteRenderSpec> notes;
    std::function<void(double)> progress;
    // Each note's own audio, as the engine made it and before it is mixed
    // with its neighbours: which note of notes it is, the piece, its rate,
    // and how far before the note's start the piece begins -- its first
    // sample is one preutterance early, which is where the consonant is.
    //
    // A note drawn from the mix instead is drawn from a stretch that holds
    // the note before it as well, and its own consonant lands in that note's
    // row rather than its own.
    // The last two read, at a time measured from the note's start, what the
    // mix multiplies this note's samples by: everything, and everything but
    // the envelope.  Everything means the envelope and the fades the mix puts
    // on this note's own samples -- in across its overlap, out where the next
    // note takes over.  A piece runs on well past that hand-over, and drawn
    // without the fade it showed sound where the envelope had already ended,
    // sound nobody hears.  The fades are this note's samples fading and
    // nothing of the neighbour's, so the picture still holds one note.
    std::function<void(std::size_t, const juce::var&)> noteParameters;
    std::function<void(std::size_t, const std::vector<UtauPhonemeSpan>&)> notePhonemes;
    std::function<void(std::size_t, const UtauMixEnvelope&)> noteMixEnvelope;
    std::function<void(std::size_t, const juce::AudioBuffer<float>&, double, double,
                       const std::function<float(double)>&,
                       const std::function<float(double)>&)> notePiece;
    WavExportComponent exportComponent = WavExportComponent::full;
    // Scheduling only; excluded from synthesis/cache keys. The full request
    // stays in musical order, including neighbours needed for crossfades.
    double timelineStartSeconds = 0.0;
    std::function<double()> priorityPosition;
    RequestedVoicebankFamily requestedFamily = RequestedVoicebankFamily::legacyAuto;
};

struct VoicebankFamilyResolution
{
    RequestedVoicebankFamily family = RequestedVoicebankFamily::utau;
    juce::String warning;
    // Empty for all matching legacy routes: preserve historical cache bytes.
    juce::String cacheDiscriminator;
};
// Scheduling/cache call false (no bank index loading). Render workers call true
// before CVVC expansion, model work or piano/native fallback. One shared policy.
[[nodiscard]] VoicebankFamilyResolution resolveRequestedVoicebankFamily(
    const UtauRenderRequest& request, bool validateUtauSamples = false);

struct UtauRenderResult
{
    juce::AudioBuffer<float> buffer;
    double sampleRate = 0.0;
    juce::String backend;
    juce::String warning;
};

// The four-region split of one note, in seconds, onset/glide/nucleus/coda.
struct UtauRegionSplit
{
    bool valid = false;
    std::array<double, 4> seconds {};
};

// A note whose lyric is this is a rest: it keeps its place in the phrase and
// sounds nothing.  Two letters rather than one, because a single "R" is a real
// alias in the Chinese CVVC banks -- they record the release of every vowel as
// "a R", "ai R" and so on -- and a lyric of "R" should go on reaching it.
// Not the same as an empty lyric, which renders the piano preview tone.
[[nodiscard]] bool isRestLyric(const juce::String& lyric);

struct UtauSampleTiming
{
    double preutteranceSeconds = 0.0;
    double consonantSeconds = 0.0;
    double overlapSeconds = 0.0;
    // Source lengths of the four regions, from the voicebank's oto4.ini.
    // Empty unless the sample has been extended to four regions.
    bool hasRegions = false;
    std::array<double, 4> regionSeconds {};
    // One letter per region from 谋-OTO; empty in every other mode.
    juce::String mouClasses;
    // Offset to cutoff: all the audio this entry has, and so the furthest a
    // lead-in can reach back before the beat.
    double sampleSeconds = 0.0;
    // The oto alias the lyric actually resolved to.  With a prefix.map that
    // is the lyric wrapped in the prefix and suffix its pitch calls for, so
    // it says which pitch bank of a multi-bank voicebank was reached.
    juce::String resolvedAlias;
};

struct UtauTailFadeSpan {double startSeconds=0,endSeconds=0;};

inline double consonantVelocityScale(int velocity)
{
    return std::exp2(juce::jlimit(-1022.0, 1023.0, 1.0 - static_cast<double>(velocity) / 100.0));
}

inline double scaledUtauPreutterance(double pre, double consonant, double available, int velocity)
{
    return juce::jlimit(0.0, available,
        std::min(pre, consonant) * consonantVelocityScale(velocity) + std::max(0.0, pre - consonant));
}

inline double scaledUtauPreutterance(const UtauSampleTiming& timing, int velocity)
{
    return scaledUtauPreutterance(timing.preutteranceSeconds, timing.consonantSeconds, timing.sampleSeconds, velocity);
}

class UtauRenderer final
{
public:
    // Strict OTO lookup for phonemizers: never use the one-recording/file-name fallback.
    static std::optional<UtauSampleTiming> mappedAliasTiming(
        const juce::File& directory, const juce::String& alias, float midiNote);
    // WCSNDM 0.0803 exposes Mx1/Mx2 only on pure K2. Its other kernels
    // either ignore Mx or mix an unfiltered L1 component back into the result.
    [[nodiscard]] static bool supportsComponentExport(const juce::File& engine,
                                                      const juce::String& effectiveFlags);
    [[nodiscard]] static juce::String diagnosticPitchbend(
        const UtauNoteRenderSpec& note, double bpm, double preutterance, double outputSeconds);
    // Fit existing attack/release ramps to the actual sounding span. Shared by
    // resampler and neural voicebank mixers; stored editing points stay intact.
    [[nodiscard]] static std::vector<UtauAmplitudePoint> fitAmplitudeEnvelope(
        const std::vector<UtauAmplitudePoint>& drawn, double preutteranceSeconds,
        double soundingEndSeconds);
    static void invalidateVoicebankCache();
    // The HF vocoder daemon an engine talks to, started ahead of the first
    // render: it takes 10-20 s to load its model, which the first HF note used
    // to wait for.  Nothing happens unless the engine has an hf_backend beside
    // it, and nothing is started when a daemon is already listening.  Runs on
    // a background thread.
    static void prewarmHfDaemon(const juce::File& resamplerExecutable);
    // The same, on the calling thread, and whether a daemon was started.  The
    // port is the daemon's own (51765) except in a check.
    static bool startHfDaemonIfNeeded(const juce::File& resamplerExecutable,
                                      int port = 51765);
    // Stop only HF daemon processes this editor started. A pre-existing
    // daemon may belong to another editor and is never adopted or stopped.
    // Permanent for this application lifetime: no prewarm can restart one.
    static void shutdownHfDaemon();
    // The interpreter the engine would start the daemon with: the first line of
    // hf_backend/python.txt, a relative one taken from the engine's folder, and
    // "pythonw" when there is no such file.  Read exactly as the engine reads
    // it, so the daemon started here is the one the engine would have started.
    [[nodiscard]] static juce::String hfDaemonInterpreter(const juce::File& engineDirectory);
    // Whether sampleTiming can answer for this voicebank without first
    // reading the whole of it.  When it cannot, the reading is started on a
    // background thread -- once, however often this is asked -- and whenReady
    // runs on the message thread after it is done.  For the roll, which asks
    // on every layout and must not stop the window while a bank is read.
    [[nodiscard]] static bool voicebankIndexReady(const juce::File& voicebankDirectory,
                                                  bool fourRegion, bool consonantClasses,
                                                  std::function<void()> whenReady);
    // Test seams.  How many times a voicebank index has been read from disk;
    // making every cached index re-check its files on the next look instead
    // of within the second; and which sample an alias resolves to, through the
    // index's tables or through the plain scan they replaced, described as
    // "file|alias|offset" so the two can be compared.
    [[nodiscard]] static int diagnosticIndexBuilds();
    // How often the message thread had to read a bank itself or wait for one
    // being read -- the stall the roll's background reading exists to avoid.
    [[nodiscard]] static int diagnosticMessageThreadWaits();
    static void diagnosticRecheckVoicebankFiles();
    // Refresh at the next worker/background lookup; no bank load on the UI thread.
    static void recheckVoicebankFiles(const juce::File& root);
    [[nodiscard]] static juce::String diagnosticResolve(const juce::File& voicebankDirectory,
                                                        const juce::String& alias, float midiNote,
                                                        bool fourRegion, bool consonantClasses,
                                                        bool byScan);
    // Where the crossfade into the next note finishes, which is where the
    // note before it stops sounding.
    //
    // The next note begins sounding a preutterance before its beat and fades
    // in across its overlap; the note before it fades out over the same
    // stretch.  Never past the next note's own end: beyond there the note
    // after it owns the seam, and an overlap typed longer than the note it
    // belongs to would leave the previous syllable droning under the phrase.
    [[nodiscard]] static double crossfadeEnd(double soundStart, double overlap,
                                             double nextNoteEnd);
    // Whether there is a crossfade between them at all.
    //
    // Contiguous counts.  A preutterance of zero puts the next note's sounding
    // start exactly on this note's end, and asking whether it starts *before*
    // that end answered no: the overlap was ignored for every note with no
    // lead-in, which is most of a plain CV bank.  Only a real gap -- the next
    // note beginning to sound after this one has finished -- has no seam.
    [[nodiscard]] static bool crossfadesInto(double soundStart, double noteEnd);
    // 界: a 拼字 note -- one with no length of its own -- reads its entry's
    // first two regions, the onset and the glide, and nothing after them, as
    // though the entry had only two.  A four-region entry is cut where its
    // second region ends.  Not 谋, whose entries say for themselves how many
    // regions they have, and not plain UTAU, which has none.
    //
    // Both the renderer and the roll read this, so the lines drawn in such a
    // note are where its regions really end up.
    // WCSNDM's 16th argument carries curves independently of oto4 / mode.
    // Disabled curves stay stored in the project but are not sent.
    [[nodiscard]] static bool sendsFlagCurves(bool noteFlagCurve);
    [[nodiscard]] static bool readsOnlyFirstTwoRegions(bool fourRegion,
                                                       bool consonantClasses,
                                                       double durationSeconds);
    [[nodiscard]] static std::array<double, 4> firstTwoRegions(
        const std::array<double, 4>& regionSeconds);
    [[nodiscard]] static std::optional<UtauSampleTiming> sampleTiming(
        const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
        int consonantVelocity = 100, bool fourRegion = false,
        bool consonantClasses = false);
    // The timing one note sings with: its own oto when it has one enabled,
    // otherwise its entry's.  Everything that draws or measures a note reads
    // this, so the roll shows the entry the note will really be rendered from.
    [[nodiscard]] static std::optional<UtauSampleTiming> sampleTiming(
        const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
        int consonantVelocity, bool fourRegion, bool consonantClasses,
        const UtauOtoOverride* noteOto);
    // The recording an alias resolves to, plus its sounding region and OTO
    // timing after consonant-velocity scaling and STP.  Voicebank loading is
    // shared native material code; exposing the resolved sample lets a native
    // renderer (NSF-HiFiGAN, etc.) read the same audio/OTO without the classic
    // resampler.  found is false when nothing matched.
    struct ResolvedSample
    {
        bool found = false;
        juce::File file;
        double offsetSeconds = 0.0;      // region start in the recording (STP-shifted)
        double endSeconds = 0.0;         // region end
        double fileSeconds = 0.0;        // whole recording length
        double preutteranceSeconds = 0.0;// velocity-adjusted lead-in
        double consonantSeconds = 0.0;   // unscaled fixed consonant length
        double overlapSeconds = 0.0;
        float sourceMidi = 60.0f;
        bool hasRegions = false;
        std::array<double, 4> regionSeconds {};
        juce::String mouClasses;
        // The actual canonical lookup's immutable, unmodified physical OTO row.
        // Null for native OTTO/HJM, unlisted/derived audio, or no resolved sample.
        // Consumers must revalidate sourceFingerprint before local application;
        // never rediscover a row by alias/file or treat this as a native ID.
        // Classic mode only; null in Jie/Mou because their auxiliary files
        // are not pinned by the base row's fingerprint.
        std::shared_ptr<const hachi::VoicebankOtoEntry> sourceOto;
        juce::String resolvedAlias;
    };
    [[nodiscard]] static ResolvedSample resolveVoiceSample(
        const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
        int consonantVelocity = 100, bool fourRegion = false,
        bool consonantClasses = false, double stpSeconds = 0.0,
        bool preutteranceOverrideEnabled = false, double preutteranceSeconds = 0.0,
        bool overlapOverrideEnabled = false, double overlapSeconds = 0.0,
        const UtauOtoOverride* noteOto = nullptr, bool aliasIsResolved = false);
    static UtauRenderResult render(const UtauRenderRequest& request);
    static juce::AudioBuffer<float> pianoPreview(const UtauNoteRenderSpec& note, double sampleRate = 44100.0);
    static float amplitudeGain(const std::vector<UtauAmplitudePoint>& envelope, double localSeconds);
    // The last existing OTO region, on the same sounding span shown in the roll.
    [[nodiscard]] static std::optional<UtauTailFadeSpan> tailFadeSpan(
        const UtauSampleTiming& timing,double soundStart,double soundEnd,
        double nominalDuration,int velocity,bool fourRegion,bool mou,
        const std::array<double,3>* manual=nullptr);
    [[nodiscard]] static std::optional<UtauTailFadeSpan> headEnvelopeSpan(
        const UtauSampleTiming& timing,double soundStart,double soundEnd,
        double nominalDuration,int velocity,bool fourRegion,bool mou,
        const std::array<double,3>* manual=nullptr);
    // How one note's four regions divide its output.  With no manual split
    // this reproduces the engine's own weight allocation, so what the piano
    // roll draws is what the resampler is asked for.
    // leadInSeconds is how far the sample reaches back before the note
    // begins.  The onset is the consonant, and the consonant is what is sung
    // before the beat, so the onset ends where the note starts: that is the
    // whole meaning of a preutterance.  Its own length follows from that.
    [[nodiscard]] static UtauRegionSplit regionSplit(
        const std::array<double, 4>& sourceSeconds, double outputSeconds,
        int consonantVelocity, double leadInSeconds,
        const std::array<double, 3>* manualFractions,
        // 谋-OTO classes, one letter per region.  Null in every other
        // mode, and then this behaves exactly as it always did.
        const juce::String* classes = nullptr);
};
}
