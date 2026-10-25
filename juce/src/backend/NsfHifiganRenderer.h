#pragma once
#include <functional>

#include "OrtExecution.h"
#include "UtauRenderer.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <vector>

namespace hachi::backend
{
// Variable-hop Mel stretch order.  Mirrors the two compositions an editor can
// choose between: splice the per-segment stretched Mel first and pitch-shift
// afterwards (the HachiShifter variable-mel-hop default), or pitch-shift every
// source-time segment before joining them (Melodyne5's order: its frequency
// mask runs on each element before the Catmull-Rom time stretch and splice).
enum class NsfHifiganStretchOrder
{
    fixedHop,
    spliceThenShift,
    shiftThenSplice,
    hifiShifterMel,
    wcsNonLoop
};

struct NsfHifiganTimeMapPoint
{
    double targetSeconds = 0.0;
    double sourceSeconds = 0.0;
};

struct NsfHifiganRenderResult
{
    juce::AudioBuffer<float> buffer;
    juce::String error;
    juce::File modelFile;
    juce::String activeInference { "cpu" };
    bool usedModel = false;
    bool usedHarmonicNoise = false;
    int protectedUnvoicedFrames = 0;
    bool usedWcsParameters = false;
    bool reusedWcsSourceAnalysis = false;
    // Diagnostics for native incremental synthesis (not persisted in projects).
    int inferredChunks = 0;
    int reusedChunks = 0;
};

// Per-edge amplitude guard applied after a neural decode.  Each edge fade is an
// equal-power sine ramp that suppresses the isolated discontinuity at a clip
// boundary whose source phase is unrelated to the next clip.  Connected phrase
// seams (which are decoded inside the same pass) and mixer-crossfaded seams do
// not need it, so the caller can zero the corresponding edge to avoid the
// small 3 ms level dip it would otherwise bake into every splice point.
struct NsfHifiganEdgeGuard
{
    float startSeconds = 0.003f;
    float endSeconds = 0.003f;
    float neighborStartF0 = 0.0f;
    float neighborEndF0 = 0.0f;
};

class NsfHifiganRenderer final
{
public:
    // Call after all render services have stopped, before releasing the runtime.
    static void shutdown();
    [[nodiscard]] static bool modelAvailable(const juce::File& configuredModelDirectory);
    static NsfHifiganRenderResult render(
        const juce::AudioBuffer<float>& source,
        double sampleRate,
        int targetSamples,
        double framePeriodMs,
        const std::vector<float>& targetMidi,
        const std::vector<float>& formantSemitones,
        const std::vector<NsfHifiganTimeMapPoint>& timeMap,
        const juce::File& configuredModelDirectory,
        const OrtExecutionConfig& execution,
        NsfHifiganStretchOrder stretchOrder,
        bool normalizeVolume = false,
        const NsfHifiganEdgeGuard& edgeGuard = NsfHifiganEdgeGuard{},
        const std::function<bool()>& cancelled = {},
        bool protectNativeNoise = false,
        bool smoothPitchTransitions = true,
        // Native chunk reuse is independent of the optional H/N protection.
        bool incrementalNativeRender = false,
        // Required only by the non-U WCSNDM parameter stretch option.
        const std::vector<float>& sourceMidi = {});
};

// UTAU voicebank synthesis on the one NSF-HiFiGAN renderer (no separate
// renderer): these are the native building blocks that let the NSF-HiFiGAN
// path preserve UTAU timing while applying HiFisampler FLAG DSP -- OTO timing, consonant
// velocity, preutterance/overlap, vibrato/pitch as the model's F0 input, and
// overlap crossfade mixing -- in the spirit of hifisampler but without the
// classic time-domain resampler or an external executable.  Split out as pure
// functions so the timing/F0/mix maths can be checked without the ONNX model.

// One note's OTO timing (STP already folded into offset/end), in seconds.
struct NsfUtauSampleTiming
{
    double offsetSeconds = 0.0;
    double endSeconds = 0.0;
    double consonantSeconds = 0.0;
    double preutteranceSeconds = 0.0;
    double overlapSeconds = 0.0;
    double fileSeconds = 0.0;
    bool hasRegions = false;
    std::array<double, 4> regionSeconds {};
    juce::String mouClasses;
};

// The plan for feeding one note to NSF-HiFiGAN: the source region and a pure
// target->source time warp (pitch is the F0 input, never a resample).
struct NsfUtauNotePlan
{
    bool valid = false;
    double sourceStartSeconds = 0.0;
    double sourceEndSeconds = 0.0;
    double soundStartOffsetSeconds = 0.0;
    double outputSeconds = 0.0;
    std::vector<NsfHifiganTimeMapPoint> timeMap;
    bool usesRegions = false;
    int regionCount = 0;
    std::array<double, 4> outputRegionSeconds {};
    juce::String regionClasses;
};

[[nodiscard]] NsfUtauNotePlan buildNsfUtauNotePlan(
    const NsfUtauSampleTiming& timing, double noteStartSeconds,
    double noteDurationSeconds, double consonantVelocityScale, double tailSeconds,
    const std::array<double, 3>* manualFractions = nullptr,
    bool firstTwoRegionsOnly = false, double soundingOutputSeconds = 0.0);

// A pitch handle on a note, cents from its MIDI pitch (vibrato folded in).
struct NsfUtauPitchPoint
{
    double timeSeconds = 0.0;
    float cents = 0.0f;
};

[[nodiscard]] std::vector<float> buildNsfUtauTargetMidi(
    float midiNote, const std::vector<NsfUtauPitchPoint>& pitchCurve,
    double framePeriodMs, double outputSeconds, double soundStartOffsetSeconds,
    const std::function<float(double)>& timelinePitchCents = {});

// One synthesised note laid into the phrase (audio at soundStartSeconds,
// crossfading its head over overlapSeconds; a rest holds its place).
struct NsfUtauMixNote
{
    juce::AudioBuffer<float> audio;
    double soundStartSeconds = 0.0;
    double overlapSeconds = 0.0;
    bool rest = false;
    double standardEnvelopeSeconds = 0;
    int cvvcCrossfadeStyle = -1; // -1 preserves legacy NSF behavior, 0 linear, 1 smooth
};

[[nodiscard]] juce::AudioBuffer<float> mixNsfUtauNotes(
    const std::vector<NsfUtauMixNote>& notes, double totalSeconds, double sampleRate,
    int channels = 1, const std::function<bool()>& cancelled = {},
    const std::function<void(std::size_t, const UtauMixEnvelope&)>& envelopeReady = {});

// Synthesise one voicebank note through NSF-HiFiGAN (empty audio + reason when
// the model is unavailable, so the caller warns rather than failing silently).
struct NsfUtauSynthResult
{
    juce::AudioBuffer<float> audio;
    double sampleRate = 0.0;
    bool usedModel = false;
    juce::String error;
};

// Apply only saved native tension/breath, in native units, to one already
// decoded voicebank note. Gain and FLAG DSP belong to their existing stages.
// Exactly neutral controls are an immediate byte-preserving no-op.
void applyNsfUtauNativeExpression(juce::AudioBuffer<float>& audio, double sampleRate,
    double framePeriodMs, const std::vector<float>& targetMidi, const UtauNoteRenderSpec& note);

[[nodiscard]] NsfUtauSynthResult synthesizeNsfUtauNote(
    const juce::File& sampleFile, const NsfUtauNotePlan& plan,
    float midiNote, const std::vector<NsfUtauPitchPoint>& pitchCurve,
    const juce::File& modelDirectory, const OrtExecutionConfig& execution,
    const std::function<float(double)>& timelinePitchCents = {},
    const UtauNoteRenderSpec* flagsNote = nullptr,
    const std::function<bool()>& cancelled = {});

// Render a whole UTAU voicebank phrase through the one NSF-HiFiGAN renderer:
// resolve each note's sample by alias (and prefix-mapped pitch bank), plan its
// OTO timing, synthesise it with the note's F0, and overlap-mix the phrase.
// The request type is the shared UtauRenderRequest so the editor/model build it
// the same way; synthesis never touches the classic resampler or an external
// executable.  Empty buffer + warning when the model is unavailable.
[[nodiscard]] UtauRenderResult renderNsfUtauPhrase(const UtauRenderRequest& request,
    const juce::File& modelDirectory, const OrtExecutionConfig& execution);
// Controls only automatic decoder F0 edge conditioning, after curve sampling.
void conditionNsfPitchTransitions(std::vector<float>& f0, bool enabled);
[[nodiscard]] bool runNsfPitchTransitionsSmoke(const juce::File& folder, const juce::File& modelDirectory);
[[nodiscard]] bool runNativeNoiseSmoke(const juce::File& folder, const juce::File& modelDirectory, const juce::File& recording);
[[nodiscard]] bool runWcsNonLoopSmoke(const juce::File& folder, const juce::File& modelDirectory, const juce::File& recording = {});
[[nodiscard]] bool runHiFiShifterMelSmoke(const juce::File& folder, const juce::File& modelDirectory);
[[nodiscard]] bool runHifisamplerSmoke(const juce::File& folder, const juce::File& modelDirectory);
[[nodiscard]] bool runHnsepCacheProbe(const juce::File& folder, const juce::File& modelDirectory,
                                    const juce::File& pcmFile, bool force);

}
