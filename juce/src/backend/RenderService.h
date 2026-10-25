#pragma once

#include "Mld5Renderer.h"
#include "UtauRenderer.h"
#include "OrtExecution.h"
#include "PlaybackRenderQueue.h"
#include <juce_events/juce_events.h>
#include <functional>
#include <memory>

namespace hachi::backend
{
enum class PitchRenderBackend
{
    mld5 = 0,
    mld3 = 1,
    nsfHifigan = 2,
    world = 3,
    vslib = 4,
    llsm2 = 5
};

struct Mld5FileRenderRequest
{
    juce::File sourceFile;
    double sourceOffsetSeconds = 0.0;
    double sourceDurationSeconds = 0.0;
    double targetDurationSeconds = 0.0;
    double framePeriodMs = 5.0;
    std::vector<float> sourceMidi;
    std::vector<float> targetMidi;
    std::vector<float> formantSemitones;
    std::vector<float> noteGain;
    std::vector<float> tension;
    std::vector<float> breath;
    // Zero disables Robust Pitch Curve.  A positive value selects it and also
    // identifies the owning note, preventing filtering across note boundaries.
    // External callers may use 1.0 for one continuous robust region.
    std::vector<float> robustPitchCurve;
    std::vector<TimeMapPoint> timeMap;
    juce::File hifiganModelDirectory;
    OrtExecutionConfig inference;
    PitchRenderBackend pitchBackend = PitchRenderBackend::llsm2;
    int stretchAlgorithm = 0;
    bool normalizeVolume = false;
    bool nsfSmoothPitchTransitions = true;
    bool nsfNoiseProtection = true;
    bool matchNsfSourceLevel = false;
    // Source F0 is measured rather than a hand-created absolute target.
    // An identity request can play PCM directly, without vocoder coloration.
    bool preserveUneditedSource = false;
    // Per-edge neural guard overrides for the nsf-hifigan backend.  A negative
    // value keeps the renderer default (3 ms both sides).  A connected seam
    // sets the matching edge to 0 so the mixer crossfade owns the hand-off
    // instead of a baked-in 3 ms level dip at every splice point.
    float neuralGuardStartSeconds = -1.0f;
    float neuralGuardEndSeconds = -1.0f;
    // Neighbor F0 at boundary edges for same-source pitch-split transitions.
    // When 0.0f the edge context uses normal reflection; a positive value
    // interpolates from/to the neighbor's actual F0 at that seam.
    float neighborEdgeF0Start = 0.0f;
    float neighborEdgeF0End = 0.0f;
    bool isGlideMerged = false;
    WavExportComponent exportComponent = WavExportComponent::full;
    // Transient per-render diagnostic switch, never a project preference.
    // True preserves current native incremental decoding; false uses legacy
    // 4096-frame inference when noise protection is also disabled.
    bool nsfIncrementalRender = true;
};

struct RenderedAudio
{
    juce::AudioBuffer<float> buffer;
    double sampleRate = 0.0;
    juce::String backend;
    juce::String warning;
};

// Shared with the mixer so preserving source PCM also preserves its edges.
[[nodiscard]] bool canPreserveNativeSource(const Mld5FileRenderRequest& request,
                                          double sampleRate, int sourceSamples,
                                          int targetSamples);

class RenderService final
{
public:
    using Completion = std::function<void(juce::AudioBuffer<float>)>;
    using FileCompletion = std::function<void(RenderedAudio)>;

    RenderService();
    ~RenderService();
    void renderMld5(Mld5RenderRequest request, Completion completion);
    void renderMld5File(Mld5FileRenderRequest request, FileCompletion completion,
                       RenderSchedule schedule = {});
    void renderUtau(UtauRenderRequest request, FileCompletion completion,
                    RenderSchedule schedule = {});
    // Native NSF-HiFiGAN voicebank synthesis of a whole UTAU phrase.  Same
    // request the classic UTAU path uses; the one NSF-HiFiGAN renderer does the
    // synthesis (renderNsfUtauPhrase), so a voicebank track on NSF-HiFiGAN is a
    // native render, not the classic resampler.
    void renderNsfUtau(UtauRenderRequest request, juce::File modelDirectory,
                       OrtExecutionConfig execution, FileCompletion completion,
                       RenderSchedule schedule = {});
    void cancelAll();
    [[nodiscard]] bool hasActiveJobs() { return queue.hasActiveJobs(); }
    void setPlaybackPosition(double seconds) noexcept { queue.setPosition(seconds); }
    void beginUpdate() { queue.beginUpdate(); }
    void endUpdate(const std::unordered_set<std::string>& keys) { queue.endUpdate(keys); }
    [[nodiscard]] double playbackPriorityPosition() const
    { return queue.positionClock()->load(std::memory_order_relaxed); }

private:
    class RenderJob;
    class FileRenderJob;
    class UtauRenderJob;
    class NsfUtauRenderJob;
    PlaybackRenderQueue queue;
};
}
