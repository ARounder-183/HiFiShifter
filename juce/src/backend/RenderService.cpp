#include "RenderService.h"
#include "NativeExpression.h"
#include "Mld3Renderer.h"
#include "NsfHifiganRenderer.h"
#include "Llsm2Renderer.h"
#include "WorldRenderer.h"
#include "DiffSingerRenderer.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <signalsmith-stretch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <mutex>
#include <utility>

namespace hachi::backend
{
bool canPreserveNativeSource(const Mld5FileRenderRequest& request,
                             double sampleRate, int sourceSamples, int targetSamples)
{
    if (!(sampleRate > 0.0) || !request.preserveUneditedSource
        || request.exportComponent != WavExportComponent::full || request.normalizeVolume)
        return false;
    const auto neutral = [](const auto& values, float expected)
    {
        return std::all_of(values.begin(), values.end(), [&](float value)
        { return std::isfinite(value) && std::abs(value - expected) <= 1.0e-4f; });
    };
    const auto pitchUnchanged = request.sourceMidi.size() == request.targetMidi.size()
        && std::equal(request.sourceMidi.begin(), request.sourceMidi.end(), request.targetMidi.begin(),
            [](float sourcePitch, float targetPitch)
            { return std::isfinite(sourcePitch) && std::isfinite(targetPitch)
                && std::abs(sourcePitch - targetPitch) <= 1.0e-4f; });
    // Independently rounded source offset and duration can differ by one
    // sample at EOF after a free crop. That is not a time-stretch edit.
    const auto timeUnchanged = std::abs(sourceSamples - targetSamples) <= 1
        && std::abs(request.sourceDurationSeconds - request.targetDurationSeconds) <= .5 / sampleRate
        && std::all_of(request.timeMap.begin(), request.timeMap.end(), [&](const auto& point)
            { return std::isfinite(point.sourceSeconds) && std::isfinite(point.targetSeconds)
                && std::abs(point.sourceSeconds - point.targetSeconds) <= .5 / sampleRate; });
    return timeUnchanged && pitchUnchanged && neutral(request.formantSemitones, 0)
        && neutral(request.tension, 0) && neutral(request.robustPitchCurve, 0);
}

namespace
{
std::timed_mutex llsmRenderMutex;

std::optional<std::pair<float, float>> pitchAt(const std::vector<float>& sourceMidi,
                                                const std::vector<float>& targetMidi,
                                                double position)
{
    if (sourceMidi.empty() || targetMidi.empty() || !std::isfinite(position) || position < 0.0)
        return std::nullopt;
    const auto left = static_cast<std::size_t>(std::floor(position));
    if (left >= sourceMidi.size() || left >= targetMidi.size()) return std::nullopt;
    const auto right = std::min({ left + 1, sourceMidi.size() - 1, targetMidi.size() - 1 });
    const auto amount = static_cast<float>(position - static_cast<double>(left));
    const auto source = sourceMidi[left] + (sourceMidi[right] - sourceMidi[left]) * amount;
    const auto target = targetMidi[left] + (targetMidi[right] - targetMidi[left]) * amount;
    if (!(std::isfinite(source) && std::isfinite(target) && source > 0.0f && target > 0.0f))
        return std::nullopt;
    return std::pair { source, target };
}

float valueAt(const std::vector<float>& curve, double position, float fallback)
{
    if (curve.empty() || !std::isfinite(position) || position < 0.0) return fallback;
    const auto left = static_cast<std::size_t>(std::floor(position));
    if (left >= curve.size()) return fallback;
    const auto right = std::min(left + 1, curve.size() - 1);
    const auto amount = static_cast<float>(position - static_cast<double>(left));
    const auto a = curve[left];
    const auto b = curve[right];
    return std::isfinite(a) && std::isfinite(b) ? a + (b - a) * amount : fallback;
}

float median(std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    auto result = *middle;
    if ((values.size() & 1u) == 0u)
    {
        const auto lower = std::max_element(values.begin(), middle);
        result = (result + *lower) * 0.5f;
    }
    return result;
}

double activeRms(const juce::AudioBuffer<float>& audio)
{
    auto peak = 0.0f;
    for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        for (int sample = 0; sample < audio.getNumSamples(); ++sample)
            peak = std::max(peak, std::abs(audio.getSample(channel, sample)));
    if (peak <= 1.0e-7f) return 0.0;
    const auto gate = std::max(1.0e-5f, peak * 0.01f);
    auto power = 0.0;
    auto count = std::size_t{};
    for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        for (int sample = 0; sample < audio.getNumSamples(); ++sample)
        {
            const auto value = audio.getSample(channel, sample);
            if (!std::isfinite(value) || std::abs(value) < gate) continue;
            power += static_cast<double>(value) * value;
            ++count;
        }
    return count > 0 ? std::sqrt(power / static_cast<double>(count)) : 0.0;
}

void matchActiveRms(juce::AudioBuffer<float>& rendered,
                    const juce::AudioBuffer<float>& source,
                    juce::AudioBuffer<float>* noise = nullptr)
{
    const auto sourceRms = activeRms(source);
    const auto renderedRms = activeRms(rendered);
    if (!(sourceRms > 1.0e-8 && renderedRms > 1.0e-8)) return;
    auto gain = static_cast<float>(std::clamp(sourceRms / renderedRms, 0.25, 4.0));
    auto peak = 0.0f;
    for (int channel = 0; channel < rendered.getNumChannels(); ++channel)
        for (int sample = 0; sample < rendered.getNumSamples(); ++sample)
            peak = std::max(peak, std::abs(rendered.getSample(channel, sample)));
    if (peak * gain > 0.98f) gain = 0.98f / peak;
    rendered.applyGain(gain);
    if (noise != nullptr) noise->applyGain(gain);
}

void raiseActiveRmsFloor(juce::AudioBuffer<float>& rendered,
                         const juce::AudioBuffer<float>& source)
{
    const auto sourceRms = activeRms(source);
    const auto renderedRms = activeRms(rendered);
    if (!(sourceRms > 1.0e-8 && renderedRms > 1.0e-8) || renderedRms >= sourceRms) return;
    auto gain = static_cast<float>(std::clamp(sourceRms / renderedRms, 1.0, 4.0));
    auto peak = 0.0f;
    for (int channel = 0; channel < rendered.getNumChannels(); ++channel)
        for (int sample = 0; sample < rendered.getNumSamples(); ++sample)
            peak = std::max(peak, std::abs(rendered.getSample(channel, sample)));
    if (peak * gain > 0.98f) gain = 0.98f / peak;
    if (gain > 1.0f) rendered.applyGain(gain);
}

// Reconstructed Robust Pitch Curve stage.  The binary handler changes a flag
// on the detection audio source (boolean at source+0x1ac), before note edits
// are rendered.  Robustify source F0, then reapply the original target-source
// edit exactly.  Filtering the edit delta instead would leave an octave error
// present in both curves untouched and would alter saved tuning decisions.
void applyRobustPitchCurve(std::vector<float>& sourceMidi,
                           std::vector<float>& targetMidi,
                           const std::vector<float>& enabled)
{
    const auto count = std::min({ sourceMidi.size(), targetMidi.size(), enabled.size() });
    if (count < 3) return;
    std::vector<float> originalDelta(count, 0.0f);
    std::vector<float> filteredSource(sourceMidi.begin(), sourceMidi.begin()
                                      + static_cast<std::ptrdiff_t>(count));
    for (std::size_t index = 0; index < count; ++index)
        if (sourceMidi[index] > 0.0f && targetMidi[index] > 0.0f)
            originalDelta[index] = targetMidi[index] - sourceMidi[index];

    const auto sameRegion = [&](std::size_t left, std::size_t right)
    {
        return enabled[left] >= 0.5f && enabled[right] >= 0.5f
            && std::abs(enabled[left] - enabled[right]) < 0.25f
            && sourceMidi[left] > 0.0f && sourceMidi[right] > 0.0f
            && targetMidi[left] > 0.0f && targetMidi[right] > 0.0f;
    };

    constexpr std::ptrdiff_t radius = 4; // 45 ms at the native 5 ms frame grid
    for (std::size_t index = 0; index < count; ++index)
    {
        if (enabled[index] < 0.5f || sourceMidi[index] <= 0.0f || targetMidi[index] <= 0.0f)
            continue;
        std::vector<float> neighbours;
        const auto first = std::max<std::ptrdiff_t>(0,
            static_cast<std::ptrdiff_t>(index) - radius);
        const auto last = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(count) - 1,
            static_cast<std::ptrdiff_t>(index) + radius);
        for (auto cursor = first; cursor <= last; ++cursor)
            if (sameRegion(index, static_cast<std::size_t>(cursor)))
                neighbours.push_back(sourceMidi[static_cast<std::size_t>(cursor)]);
        if (neighbours.size() < 3) continue;
        auto centreValues = neighbours;
        const auto centre = median(centreValues);
        for (auto& value : neighbours) value = std::abs(value - centre);
        const auto deviation = median(neighbours);
        const auto residual = sourceMidi[index] - centre;
        // Harmonic tracking mistakes usually land near an integer octave.
        // Fold only a convincing octave candidate; preserve ordinary vibrato,
        // bends and portamento instead of median-filtering the entire note.
        const auto octave = std::round(residual / 12.0f);
        const auto folded = residual - octave * 12.0f;
        if (std::abs(octave) >= 1.0f && std::abs(folded) <= std::max(0.45f, deviation * 4.0f))
            filteredSource[index] = centre + folded;
        else
        {
            const auto limit = std::max(1.35f, deviation * 6.0f);
            if (std::abs(residual) > limit)
                filteredSource[index] = centre + juce::jlimit(-limit, limit, residual);
        }
    }

    // Repair a non-harmonic one-frame spike only when both neighbours agree.
    // A global slope limit would also reshape legitimate glissando/vibrato.
    for (std::size_t index = 1; index + 1 < count; ++index)
        if (sameRegion(index, index - 1) && sameRegion(index, index + 1)
            && std::abs(filteredSource[index - 1] - filteredSource[index + 1]) < 0.7f)
        {
            const auto expected = (filteredSource[index - 1] + filteredSource[index + 1]) * 0.5f;
            if (std::abs(filteredSource[index] - expected) > 1.35f)
                filteredSource[index] = expected;
        }
    for (std::size_t index = 0; index < count; ++index)
        if (enabled[index] >= 0.5f && sourceMidi[index] > 0.0f && targetMidi[index] > 0.0f)
        {
            sourceMidi[index] = filteredSource[index];
            targetMidi[index] = filteredSource[index] + originalDelta[index];
        }
}

void applyExpressionAndTension(juce::AudioBuffer<float>& audio, double sampleRate,
                               double framePeriodMs,
                               const std::vector<float>& targetMidi,
                               const std::vector<float>& noteGain,
                               const std::vector<float>& tension,
                               const std::vector<float>& breath,
                               juce::AudioBuffer<float>* noise = nullptr)
{
    applyNativeExpression(audio, sampleRate, framePeriodMs, targetMidi,
                          noteGain, tension, breath, noise);
}

juce::AudioBuffer<float> renderFormantPreserved(const juce::AudioBuffer<float>& source,
                                                 int targetSamples,
                                                 double sampleRate,
                                                 double framePeriodMs,
                                                 const std::vector<float>& sourceMidi,
                                                 const std::vector<float>& targetMidi,
                                                 const std::vector<float>& formantSemitones,
                                                 const std::vector<float>& noteGain,
                                                 const std::vector<float>& tension,
                                                 const std::vector<float>& breath,
                                                 const std::vector<TimeMapPoint>& timeMap,
                                                 PitchRenderBackend pitchBackend,
                                                 int stretchAlgorithm)
{
    const auto sourceSamples = source.getNumSamples();
    const auto channels = source.getNumChannels();
    if (sourceSamples <= 0 || channels <= 0 || targetSamples <= 0 || sampleRate <= 0.0) return {};

    auto hasPitchEdit = false;
    const auto curveSize = std::min(sourceMidi.size(), targetMidi.size());
    for (std::size_t index = 0; index < curveSize; ++index)
        if (sourceMidi[index] > 0.0f && targetMidi[index] > 0.0f
            && std::abs(targetMidi[index] - sourceMidi[index]) > 1.0e-4f)
        {
            hasPitchEdit = true;
            break;
        }
    auto hasTimeWarp = sourceSamples != targetSamples;
    for (const auto& point : timeMap)
        if (targetSamples > 0 && sourceSamples > 0
            && std::abs(point.sourceSeconds * sampleRate / sourceSamples
                        - point.targetSeconds * sampleRate / targetSamples) > 1.0e-6)
        {
            hasTimeWarp = true;
            break;
        }
    auto hasFormantEdit = false;
    auto hasLevelEdit = false;
    for (const auto value : formantSemitones)
        hasFormantEdit = hasFormantEdit || std::abs(value) > 1.0e-4f;
    for (const auto value : noteGain)
        hasLevelEdit = hasLevelEdit || std::abs(value - 1.0f) > 1.0e-4f;
    for (const auto value : tension)
        hasLevelEdit = hasLevelEdit || std::abs(value) > 1.0e-4f;
    for (const auto value : breath)
        hasLevelEdit = hasLevelEdit || value > 1.0e-4f;
    if (!hasTimeWarp && !hasPitchEdit && !hasFormantEdit && !hasLevelEdit) return source;

    // Gain and breath are amplitude/spectral expression controls.  Sending an
    // otherwise untouched note through an identity phase vocoder created the
    // audible doubled/echoed voice reported on the native rewrite.  Keep that
    // path sample-exact and apply expression below without a phase transform.
    const auto needsSpectralRender = hasTimeWarp || hasPitchEdit || hasFormantEdit;
    juce::AudioBuffer<float> result;
    if (!needsSpectralRender)
    {
        result = source;
        applyExpressionAndTension(result, sampleRate, framePeriodMs, targetMidi,
                                  noteGain, tension, breath);
        return result;
    }

    // Signalsmith's phase-coherent stretcher is used as the native model-free
    // component stage.  Formant compensation is explicitly enabled: pitch and
    // duration move independently while the vocal-tract envelope remains at
    // the source frequencies.  A fixed seed makes cache renders repeatable.
    signalsmith::stretch::SignalsmithStretch<float> stretch(0x48414348L);
    // Keep both mld5 and the model-free nsf-hifigan fallback on the same proven
    // single-pass, low-latency phase clock.  The former presetDefault route had
    // a much longer analysis window and audibly pre/post-rang short vocal
    // elements, which was perceived as an echo.  mld5-specific treatment is
    // deliberately applied only once after this common render below.
    const auto stableVocalClock = pitchBackend == PitchRenderBackend::mld5
        || pitchBackend == PitchRenderBackend::nsfHifigan;
    // The stretch selector changes the actual analysis clock, not only the
    // process-call chunk size.  This makes project/MPD algorithm overrides
    // audible and keeps each model-free route deterministic.
    if (stretchAlgorithm == 1) // variable Mel-hop: dense target-time updates
        stretch.configure(channels,
            std::max(512, static_cast<int>(std::llround(sampleRate * 0.046))),
            std::max(64, static_cast<int>(std::llround(sampleRate * 0.005))), false);
    else if (stretchAlgorithm == 2) // loop: long periodic window for stable vowels
        stretch.configure(channels,
            std::max(1'024, static_cast<int>(std::llround(sampleRate * 0.140))),
            std::max(128, static_cast<int>(std::llround(sampleRate * 0.020))), false);
    else if (stretchAlgorithm == 3) // SoundTouch-style shorter sequence/overlap clock
        stretch.configure(channels,
            std::max(512, static_cast<int>(std::llround(sampleRate * 0.055))),
            std::max(96, static_cast<int>(std::llround(sampleRate * 0.012))), false);
    else if (pitchBackend == PitchRenderBackend::mld3)
        // Melodyne 3's editor exposes pitch/formant ratios and explicit note
        // transition adaptation.  Its older renderer uses a coarser periodic
        // clock than the later component engine.
        stretch.configure(channels,
            std::max(1'024, static_cast<int>(std::llround(sampleRate * 0.072))),
            std::max(96, static_cast<int>(std::llround(sampleRate * 0.0075))), false);
    else if (pitchBackend == PitchRenderBackend::llsm2)
        // Used only if libllsm2 rejects an exceptional input.  Keep its
        // fallback route distinct from every other selectable algorithm.
        stretch.configure(channels,
            std::max(1'024, static_cast<int>(std::llround(sampleRate * 0.096))),
            std::max(128, static_cast<int>(std::llround(sampleRate * 0.010))), false);
    else if (stableVocalClock)
        stretch.presetCheaper(channels, static_cast<float>(sampleRate), false);
    else
        stretch.presetDefault(channels, static_cast<float>(sampleRate), false);
    const auto initialFormant = valueAt(formantSemitones, 0.0, 0.0f);
    if (pitchBackend == PitchRenderBackend::mld3)
        stretch.setFormantFactor(std::exp2(initialFormant / 12.0f), true);
    else
        stretch.setFormantSemitones(initialFormant, true);
    stretch.setFormantBase(200.0f / static_cast<float>(sampleRate));

    const auto inputLatency = stretch.inputLatency();
    const auto outputLatency = stretch.outputLatency();
    struct StretchChunk { int sourceStart; int sourceFrames; int targetStart; int targetFrames; };
    std::vector<StretchChunk> chunks;
    auto previousSource = 0;
    auto previousTarget = 0;
    for (std::size_t index = 1; index < timeMap.size(); ++index)
    {
        const auto nextSource = juce::jlimit(previousSource, sourceSamples,
            static_cast<int>(std::llround(timeMap[index].sourceSeconds * sampleRate)));
        const auto nextTarget = juce::jlimit(previousTarget, targetSamples,
            static_cast<int>(std::llround(timeMap[index].targetSeconds * sampleRate)));
        if (nextSource > previousSource && nextTarget > previousTarget)
            chunks.push_back({ previousSource, nextSource - previousSource,
                               previousTarget, nextTarget - previousTarget });
        previousSource = nextSource;
        previousTarget = nextTarget;
    }
    if (previousSource < sourceSamples && previousTarget < targetSamples)
        chunks.push_back({ previousSource, sourceSamples - previousSource,
                           previousTarget, targetSamples - previousTarget });
    if (chunks.empty())
        chunks.push_back({ 0, sourceSamples, 0, targetSamples });
    else
    {
        auto& last = chunks.back();
        last.sourceFrames += sourceSamples - (last.sourceStart + last.sourceFrames);
        last.targetFrames += targetSamples - (last.targetStart + last.targetFrames);
    }
    std::vector<std::vector<float>> paddedInput(static_cast<std::size_t>(channels));
    std::vector<std::vector<float>> allOutput(static_cast<std::size_t>(channels));
    for (int channel = 0; channel < channels; ++channel)
    {
        auto& input = paddedInput[static_cast<std::size_t>(channel)];
        input.resize(static_cast<std::size_t>(sourceSamples + inputLatency), 0.0f);
        std::copy_n(source.getReadPointer(channel), sourceSamples, input.begin());
        allOutput[static_cast<std::size_t>(channel)].reserve(
            static_cast<std::size_t>(targetSamples + outputLatency));
    }

    // Signalsmith reads inputLatency samples ahead of its processing clock.
    // Seed that look-ahead once, then process a stream shifted by exactly that
    // amount and discard only outputLatency.  Including both latencies in the
    // stretch ratio (the previous implementation) made short notes approach
    // 1x and displaced attacks/F0 relative to the requested duration.
    if (const auto initialPitch = pitchAt(sourceMidi, targetMidi, 0.0))
    {
        const auto semitones = juce::jlimit(-24.0f, 24.0f,
            initialPitch->second - initialPitch->first);
        if (pitchBackend == PitchRenderBackend::mld3)
            // Melodyne 3 exposes this parameter as an explicit multiplicative
            // _setPitchRatio rather than the later component-cent interface.
            stretch.setTransposeFactor(std::exp2(semitones / 12.0f),
                                       8'000.0f / static_cast<float>(sampleRate));
        else
            stretch.setTransposeSemitones(semitones,
                                          8'000.0f / static_cast<float>(sampleRate));
        const auto sourceHz = 440.0f * std::pow(2.0f, (initialPitch->first - 69.0f) / 12.0f);
        stretch.setFormantBase(juce::jlimit(45.0f, 1'200.0f, sourceHz)
                               / static_cast<float>(sampleRate));
    }
    // Prime Signalsmith once with the analysis look-ahead, then advance the
    // input stream by exactly that amount.  This mirrors the proven JS/Rust
    // wrapper.  Feeding sample zero again after priming creates two phase
    // histories for the same vowel (heard as a short echo), while cropping an
    // additional input-latency interval moves the waveform away from its F0.
    if (inputLatency > 0)
    {
        std::vector<const float*> seekPointers(static_cast<std::size_t>(channels));
        for (int channel = 0; channel < channels; ++channel)
            seekPointers[static_cast<std::size_t>(channel)] =
                paddedInput[static_cast<std::size_t>(channel)].data();
        const auto& firstChunk = chunks.front();
        const auto playbackRate = static_cast<double>(firstChunk.sourceFrames)
            / std::max(1, firstChunk.targetFrames);
        stretch.seek(seekPointers.data(), inputLatency, playbackRate);
    }
    // These are separate render clocks, rather than four UI aliases for the
    // same block scheduler.  Variable-mel follows edits more densely; the loop
    // path favours stable repeated vowels; SoundTouch-style operation updates
    // in longer grains.  Melodyne Hybrid stays at the proven vocal setting.
    const auto blockSize = stretchAlgorithm == 1 ? 64
        : stretchAlgorithm == 2 ? 256
        : stretchAlgorithm == 3 ? 512 : 128;
    auto outputProduced = 0;
    const auto framePeriod = std::max(0.1, framePeriodMs);
    for (const auto& chunk : chunks)
    {
      auto chunkInput = 0;
      auto chunkOutput = 0;
      while (chunkInput < chunk.sourceFrames && chunkOutput < chunk.targetFrames)
      {
        const auto blockInput = std::min(blockSize, chunk.sourceFrames - chunkInput);
        const auto expectedNextOutput = static_cast<int>(std::llround(
            static_cast<double>(chunkInput + blockInput) * chunk.targetFrames / chunk.sourceFrames));
        const auto blockOutput = std::min(chunk.targetFrames - chunkOutput,
            std::max(1, expectedNextOutput - chunkOutput));
        const auto audibleCentre = juce::jlimit(0.0, static_cast<double>(targetSamples),
            static_cast<double>(outputProduced + blockOutput / 2 - outputLatency));
        const auto curvePosition = audibleCentre / sampleRate * 1000.0 / framePeriod;
        const auto pitch = pitchAt(sourceMidi, targetMidi, curvePosition);
        if (pitch)
        {
            const auto semitones = juce::jlimit(-24.0f, 24.0f, pitch->second - pitch->first);
            if (pitchBackend == PitchRenderBackend::mld3)
                stretch.setTransposeFactor(std::exp2(semitones / 12.0f),
                                           8'000.0f / static_cast<float>(sampleRate));
            else
                stretch.setTransposeSemitones(semitones,
                                              8'000.0f / static_cast<float>(sampleRate));
            const auto sourceHz = 440.0f * std::pow(2.0f, (pitch->first - 69.0f) / 12.0f);
            stretch.setFormantBase(juce::jlimit(45.0f, 1'200.0f, sourceHz)
                                   / static_cast<float>(sampleRate));
        }
        else
        {
            // Unvoiced, breath and sibilant regions keep their original spectrum.
            stretch.setTransposeSemitones(0.0f);
        }
        const auto formant = valueAt(formantSemitones, curvePosition, 0.0f);
        if (pitchBackend == PitchRenderBackend::mld3)
            // The M3 binary keeps _setFormantRatio independent from pitch.
            // Compensate the pitch move first, then apply this independent
            // ratio; disabling compensation would reintroduce child/elder
            // vocal-tract shifts even when the stored ratio is neutral.
            stretch.setFormantFactor(std::exp2(formant / 12.0f), true);
        else
            stretch.setFormantSemitones(formant, true);

        std::vector<const float*> inputPointers(static_cast<std::size_t>(channels));
        std::vector<std::vector<float>> blockBuffers(static_cast<std::size_t>(channels));
        std::vector<float*> outputPointers(static_cast<std::size_t>(channels));
        for (int channel = 0; channel < channels; ++channel)
        {
            inputPointers[static_cast<std::size_t>(channel)] =
                paddedInput[static_cast<std::size_t>(channel)].data()
                    + inputLatency + chunk.sourceStart + chunkInput;
            auto& block = blockBuffers[static_cast<std::size_t>(channel)];
            block.resize(static_cast<std::size_t>(blockOutput), 0.0f);
            outputPointers[static_cast<std::size_t>(channel)] = block.data();
        }
        stretch.process(inputPointers.data(), blockInput, outputPointers.data(), blockOutput);
        for (int channel = 0; channel < channels; ++channel)
        {
            auto& output = allOutput[static_cast<std::size_t>(channel)];
            const auto& block = blockBuffers[static_cast<std::size_t>(channel)];
            output.insert(output.end(), block.begin(), block.end());
        }
        chunkInput += blockInput;
        chunkOutput += blockOutput;
        outputProduced += blockOutput;
      }
    }

    if (outputLatency > 0)
    {
        std::vector<std::vector<float>> flushBuffers(static_cast<std::size_t>(channels));
        std::vector<float*> flushPointers(static_cast<std::size_t>(channels));
        for (int channel = 0; channel < channels; ++channel)
        {
            flushBuffers[static_cast<std::size_t>(channel)].resize(
                static_cast<std::size_t>(outputLatency), 0.0f);
            flushPointers[static_cast<std::size_t>(channel)] =
                flushBuffers[static_cast<std::size_t>(channel)].data();
        }
        stretch.flush(flushPointers.data(), outputLatency);
        for (int channel = 0; channel < channels; ++channel)
        {
            auto& output = allOutput[static_cast<std::size_t>(channel)];
            const auto& flush = flushBuffers[static_cast<std::size_t>(channel)];
            output.insert(output.end(), flush.begin(), flush.end());
        }
    }

    if (needsSpectralRender)
    {
        result.setSize(channels, targetSamples);
        result.clear();
        for (int channel = 0; channel < channels; ++channel)
        {
            auto* destination = result.getWritePointer(channel);
            const auto& output = allOutput[static_cast<std::size_t>(channel)];
            for (int sample = 0; sample < targetSamples; ++sample)
            {
                const auto sourceIndex = static_cast<std::size_t>(outputLatency + sample);
                destination[sample] = sourceIndex < output.size() && std::isfinite(output[sourceIndex])
                    ? output[sourceIndex] : 0.0f;
            }
        }
    }

    // The component stage is energy preserving.  Bound the correction so a
    // quiet consonant or a clipped source does not create a gain jump.
    double sourcePower = 0.0;
    double outputPower = 0.0;
    for (int channel = 0; channel < channels; ++channel)
    {
        for (int sample = 0; sample < sourceSamples; ++sample)
        {
            const auto value = source.getSample(channel, sample);
            sourcePower += static_cast<double>(value) * value;
        }
        for (int sample = 0; sample < targetSamples; ++sample)
        {
            const auto value = result.getSample(channel, sample);
            outputPower += static_cast<double>(value) * value;
        }
    }
    const auto sourceRms = std::sqrt(sourcePower / std::max(1, channels * sourceSamples));
    const auto outputRms = std::sqrt(outputPower / std::max(1, channels * targetSamples));
    if (needsSpectralRender && sourceRms > 1.0e-7 && outputRms > 1.0e-7)
        result.applyGain(static_cast<float>(juce::jlimit(0.55, 1.8, sourceRms / outputRms)));

    applyExpressionAndTension(result, sampleRate, framePeriodMs, targetMidi,
                              noteGain, tension, breath);

    // Component clips are cut at analysis element boundaries, which are not
    // guaranteed zero crossings.  Melodyne applies a compact amplitude/phase
    // hand-off there even when no user fade is visible.  Keep this de-click
    // shorter than a consonant so it removes the impulse without smearing it.
    const auto edgeSamples = std::min(targetSamples / 2,
        std::max(1, static_cast<int>(std::llround(sampleRate * 0.0025))));
    for (int channel = 0; channel < channels; ++channel)
    {
        auto* samples = result.getWritePointer(channel);
        for (int index = 0; index < edgeSamples; ++index)
        {
            const auto phase = static_cast<float>(index + 1)
                / static_cast<float>(edgeSamples + 1);
            const auto gain = std::sin(juce::MathConstants<float>::halfPi * phase);
            samples[index] *= gain;
            samples[targetSamples - 1 - index] *= gain;
        }
    }
    return result;
}
}

class RenderService::RenderJob final : public juce::ThreadPoolJob
{
public:
    RenderJob(Mld5RenderRequest requestToUse, Completion completionToUse)
        : ThreadPoolJob("mld5-render"), request(std::move(requestToUse)), completion(std::move(completionToUse))
    {
        if (request.input != nullptr)
        {
            ownedInput = *request.input;
            request.input = &ownedInput;
        }
    }

    JobStatus runJob() override
    {
        if (shouldExit()) return jobHasFinished;
        auto output = renderer.render(request);
        if (shouldExit()) return jobHasFinished;
        juce::MessageManager::callAsync([callback = std::move(completion), result = std::move(output)]() mutable
        {
            if (callback) callback(std::move(result));
        });
        return jobHasFinished;
    }

private:
    juce::AudioBuffer<float> ownedInput;
    Mld5RenderRequest request;
    Completion completion;
    Mld5Renderer renderer;
};

class RenderService::FileRenderJob final : public juce::ThreadPoolJob
{
public:
    FileRenderJob(Mld5FileRenderRequest requestToUse, FileCompletion completionToUse)
        : ThreadPoolJob("mld5-file-render"), request(std::move(requestToUse)),
          completion(std::move(completionToUse))
    {
    }

    JobStatus runJob() override
    {
        if (shouldExit()) return jobHasFinished;
        juce::AudioFormatManager formats;
        const auto separate = request.exportComponent != WavExportComponent::full;
        if (separate && request.pitchBackend != PitchRenderBackend::world
            && request.pitchBackend != PitchRenderBackend::llsm2)
        {
            RenderedAudio failure;
            failure.warning = "Selected backend cannot export voice components";
            if (completion) completion(std::move(failure));
            return jobHasFinished;
        }
        // Preserve the local MLD5 renderer for existing projects. This is an
        // independent experimental renderer, never the official Melodyne provider.
        if (request.pitchBackend == PitchRenderBackend::mld3
            || request.pitchBackend == PitchRenderBackend::vslib)
        {
            RenderedAudio failure;
            failure.warning = "Selected backend is disabled or has no native implementation";
            if (completion) completion(std::move(failure));
            return jobHasFinished;
        }
        formats.registerBasicFormats();
        auto reader = std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(request.sourceFile));
        if (reader == nullptr || reader->sampleRate <= 0.0)
        {
            if (completion) completion({});
            return jobHasFinished;
        }
        const auto sourceStart = juce::jlimit<juce::int64>(0, reader->lengthInSamples,
            static_cast<juce::int64>(std::llround(request.sourceOffsetSeconds * reader->sampleRate)));
        const auto requestedSourceSamples = static_cast<juce::int64>(std::llround(
            std::max(0.0, request.sourceDurationSeconds) * reader->sampleRate));
        const auto sourceSamples64 = std::max<juce::int64>(1,
            std::min(requestedSourceSamples, reader->lengthInSamples - sourceStart));
        const auto sourceSamples = static_cast<int>(std::min<juce::int64>(sourceSamples64,
                                                                          std::numeric_limits<int>::max()));
        const auto channels = juce::jlimit(1, 2, static_cast<int>(reader->numChannels));
        juce::AudioBuffer<float> source(channels, sourceSamples);
        source.clear();
        reader->read(&source, 0, sourceSamples, sourceStart, true, channels > 1);
        if (shouldExit()) return jobHasFinished;

        const auto targetSamples = std::max(1, static_cast<int>(std::llround(
            std::max(0.0, request.targetDurationSeconds) * reader->sampleRate)));
        const auto neutral = [](const auto& values, float expected)
        {return std::all_of(values.begin(),values.end(),[&](float value)
            {return std::isfinite(value)&&std::abs(value-expected)<=1.0e-4f;});};
        if (canPreserveNativeSource(request, reader->sampleRate, sourceSamples, targetSamples))
        {
            // Keep fractional source offset aligned with the timeline mixer.
            // Rounding only the source side otherwise shifts a free crop by
            // up to half a sample even when its time map is the identity.
            const auto exactStart = std::max(0.0, request.sourceOffsetSeconds * reader->sampleRate);
            const auto floorStart = static_cast<juce::int64>(std::floor(exactStart));
            const auto fraction = static_cast<float>(exactStart - std::floor(exactStart));
            if (fraction > 1.e-7f)
            {
                juce::AudioBuffer<float> aligned(channels, targetSamples + 1);
                aligned.clear(); reader->read(&aligned,0,targetSamples+1,floorStart,true,channels>1);
                const auto available = static_cast<int>(std::max<juce::int64>(1,
                    std::min<juce::int64>(targetSamples+1,reader->lengthInSamples-floorStart)));
                source.setSize(channels,targetSamples);
                for(int ch=0;ch<channels;++ch)for(int i=0;i<targetSamples;++i)
                {
                    const auto a=aligned.getSample(ch,std::min(i,available-1));
                    const auto b=aligned.getSample(ch,std::min(i+1,available-1));
                    source.setSample(ch,i,a+(b-a)*fraction);
                }
            }
            if (source.getNumSamples() != targetSamples)
            {
                const auto last = source.getNumSamples() - 1;
                std::vector<float> edge(static_cast<std::size_t>(channels));
                for (int ch=0;ch<channels;++ch) edge[static_cast<std::size_t>(ch)]=source.getSample(ch,last);
                source.setSize(channels,targetSamples,true,true);
                for(int ch=0;ch<channels;++ch)for(int i=last+1;i<targetSamples;++i)
                    source.setSample(ch,i,edge[static_cast<std::size_t>(ch)]);
            }
            // Detection and display do not edit audio. Gain/breath edits alone
            // can be applied to PCM without rebuilding its spectrum or phase.
            if (!neutral(request.noteGain,1) || !neutral(request.breath,0))
                applyExpressionAndTension(source,reader->sampleRate,request.framePeriodMs,
                    request.targetMidi,request.noteGain,{},request.breath);
            if (shouldExit()) return jobHasFinished;
            if (completion) completion({std::move(source),reader->sampleRate,"native-source-preserved"});
            return jobHasFinished;
        }
        // Robust pitch curve is a native analysis stabilizer, not a UTAU flag.
        // NSF has no source-MIDI input tensor, so apply the same corrected
        // target curve before its F0 conversion instead of silently dropping
        // the feature on the neural backend.
        if (request.pitchBackend == PitchRenderBackend::mld5
            || request.pitchBackend == PitchRenderBackend::nsfHifigan)
            applyRobustPitchCurve(request.sourceMidi, request.targetMidi,
                                  request.robustPitchCurve);
        juce::AudioBuffer<float> rendered, noise;
        auto* noiseOutput = separate ? &noise : nullptr;
        auto usedNsfModel = false;
        auto usedLlsm2 = false;
        juce::String nsfInference;
        if (request.pitchBackend == PitchRenderBackend::world)
        {
            std::vector<WorldTimeMapPoint> worldTimeMap;
            worldTimeMap.reserve(request.timeMap.size());
            for (const auto& point : request.timeMap)
                worldTimeMap.push_back({ point.targetSeconds, point.sourceSeconds });
            rendered = WorldRenderer::render(source, targetSamples, reader->sampleRate,
                request.framePeriodMs, request.sourceMidi, request.targetMidi,
                request.formantSemitones, worldTimeMap, noiseOutput);
            // A malformed or exceptionally constrained source still remains
            // playable through the established model-free route.
            if (rendered.getNumSamples() != targetSamples)
            {
                RenderedAudio failure;
                failure.warning = "WORLD render failed; backend substitution is prohibited";
                if (completion) completion(std::move(failure));
                return jobHasFinished;
            }
            else
                applyExpressionAndTension(rendered, reader->sampleRate, request.framePeriodMs,
                                          request.targetMidi, request.noteGain, request.tension,
                                          request.breath, noiseOutput);
        }
        else if (request.pitchBackend == PitchRenderBackend::mld5)
        {
            // Melodyne 5 algorithm path: independent MULSS spectral component
            // renderer (bandlimited harmonic remap + formant compensation +
            // granular hop), not the previous Signalsmith + finish-wrapper.
            Mld5RenderRequest mld5Request;
            mld5Request.input = &source;
            mld5Request.sampleRate = reader->sampleRate;
            mld5Request.framePeriodMs = request.framePeriodMs;
            mld5Request.sourceMidi = request.sourceMidi;
            mld5Request.targetMidi = request.targetMidi;
            mld5Request.formantSemitones = request.formantSemitones;
            mld5Request.noteGain = request.noteGain;
            for (const auto& point : request.timeMap)
                mld5Request.timeMap.push_back({ point.targetSeconds, point.sourceSeconds });
            mld5Request.targetSamples = targetSamples;
            Mld5Renderer mld5;
            rendered = mld5.render(mld5Request);
            applyExpressionAndTension(rendered, reader->sampleRate, request.framePeriodMs,
                request.targetMidi, {}, request.tension, request.breath);
        }
        else if (request.pitchBackend == PitchRenderBackend::mld3)
        {
            // The independent Melodyne algorithms are intentionally disabled
            // until their quality is validated. Never silently route mld3 to a
            // different backend.
            RenderedAudio disabled;
            disabled.warning = "mld3 backend is disabled";
            if (completion) completion(std::move(disabled));
            return jobHasFinished;
        }
        else if (request.pitchBackend == PitchRenderBackend::llsm2)
        {
            // Direct LLSM2 layer-0/layer-1 analysis and synthesis.  The vocal
            // tract envelope, glottal source phase, harmonic component and
            // noise component stay separate while time and F0 are remapped.
            std::unique_lock llsmLock(llsmRenderMutex, std::defer_lock);
            while (!llsmLock.try_lock_for(std::chrono::milliseconds(50)))
                if (shouldExit()) return jobHasFinished;
            if (shouldExit()) return jobHasFinished;
            try
            {
                rendered = Llsm2Renderer::render(source, targetSamples, reader->sampleRate,
                    request.framePeriodMs, request.sourceMidi, request.targetMidi,
                    request.formantSemitones, request.tension, request.timeMap, noiseOutput);
            }
            catch (const std::bad_alloc&)
            {
                rendered.setSize(0, 0);
            }
            llsmLock.unlock();
            if (rendered.getNumSamples() == targetSamples)
            {
                usedLlsm2 = true;
                applyExpressionAndTension(rendered, reader->sampleRate,
                    request.framePeriodMs, request.targetMidi, request.noteGain,
                    {}, request.breath, noiseOutput);
            }
            else
                try
                {
                    RenderedAudio failure;
                    failure.warning = "LLSM2 render failed; backend substitution is prohibited";
                    if (completion) completion(std::move(failure));
                    return jobHasFinished;
                }
                catch (const std::bad_alloc&)
                {
                    rendered.setSize(0, 0);
                }
        }
        else if (request.pitchBackend == PitchRenderBackend::nsfHifigan)
        {
            if (!NsfHifiganRenderer::modelAvailable(request.hifiganModelDirectory))
            {
                RenderedAudio disabled;
                disabled.warning = "NSF-HiFiGAN is selected but its model pack is unavailable";
                if (completion) completion(std::move(disabled));
                return jobHasFinished;
            }
            // WCSNDM non-looping stretch interpolates LLSM parameters before
            // physical Mel reconstruction. The other neural routes extract from
            // original PCM.  This avoids a preliminary time-domain stretch
            // changing the vocal tract before the vocoder sees it.  Standard
            // mode maps fixed-hop Mel frames; the NSF-exclusive modes extract
            // per-segment variable-hop Mel and join it before decoding once.
            // splice-first (variableMelHop) joins then shifts the formants,
            // shift-first (nsfShiftThenSplice) applies the formant curve to
            // every source-time segment before the join (Melodyne5's order).
            const auto stretchOrder = request.stretchAlgorithm == 6
                ? NsfHifiganStretchOrder::wcsNonLoop
                : request.stretchAlgorithm == 5
                ? NsfHifiganStretchOrder::hifiShifterMel
                : request.stretchAlgorithm == 4
                ? NsfHifiganStretchOrder::shiftThenSplice
                : request.stretchAlgorithm == 1
                    ? NsfHifiganStretchOrder::spliceThenShift
                    : NsfHifiganStretchOrder::fixedHop;
            std::vector<NsfHifiganTimeMapPoint> neuralTimeMap;
            neuralTimeMap.reserve(request.timeMap.size());
            for (const auto& point : request.timeMap)
                neuralTimeMap.push_back({ point.targetSeconds, point.sourceSeconds });
            NsfHifiganEdgeGuard edgeGuard;
            if (request.neuralGuardStartSeconds >= 0.0f)
                edgeGuard.startSeconds = request.neuralGuardStartSeconds;
            if (request.neuralGuardEndSeconds >= 0.0f)
                edgeGuard.endSeconds = request.neuralGuardEndSeconds;
            edgeGuard.neighborStartF0 = request.neighborEdgeF0Start;
            edgeGuard.neighborEndF0 = request.neighborEdgeF0End;
            auto neural = NsfHifiganRenderer::render(source, reader->sampleRate,
                targetSamples, request.framePeriodMs, request.targetMidi,
                request.formantSemitones, neuralTimeMap, request.hifiganModelDirectory,
                request.inference, stretchOrder, request.normalizeVolume, edgeGuard,
                [this] { return shouldExit(); }, request.nsfNoiseProtection, request.nsfSmoothPitchTransitions, request.nsfIncrementalRender, request.sourceMidi);
            if (shouldExit()) return jobHasFinished;
            if (neural.usedModel && neural.buffer.getNumSamples() == targetSamples)
            {
                rendered = std::move(neural.buffer);
                usedNsfModel = true;
                nsfInference = neural.activeInference;
                if (neural.usedWcsParameters) nsfInference += "+wcs-nonloop";
                if (request.nsfNoiseProtection)
                    nsfInference += neural.usedHarmonicNoise ? "+hnsep" : "+source-uv";
                if (request.matchNsfSourceLevel
                    && (request.stretchAlgorithm == 1 || request.stretchAlgorithm == 4))
                    raiseActiveRmsFloor(rendered, source);
                applyExpressionAndTension(rendered, reader->sampleRate,
                    request.framePeriodMs, request.targetMidi, request.noteGain,
                    request.tension, request.breath);
            }
            else
            {
                RenderedAudio disabled;
                disabled.warning = neural.error.isNotEmpty() ? neural.error
                    : "NSF-HiFiGAN render failed";
                if (completion) completion(std::move(disabled));
                return jobHasFinished;
            }
        }
        else if (request.pitchBackend == PitchRenderBackend::vslib)
        {
            // vslib is not available to the Linux build. Do not silently
            // substitute another backend there. Windows keeps its native
            // implementation path.
#if !JUCE_WINDOWS
            RenderedAudio disabled;
            disabled.warning = "vslib backend is unavailable on this platform";
            if (completion) completion(std::move(disabled));
            return jobHasFinished;
#else
            rendered = renderFormantPreserved(source, targetSamples, reader->sampleRate,
                request.framePeriodMs, request.sourceMidi, request.targetMidi,
                request.formantSemitones, request.noteGain, request.tension, request.breath,
                request.timeMap, request.pitchBackend, request.stretchAlgorithm);
#endif
        }
        else
        {
            RenderedAudio disabled;
            disabled.warning = "requested pitch backend is disabled";
            if (completion) completion(std::move(disabled));
            return jobHasFinished;
        }
        if (request.normalizeVolume && !usedNsfModel)
            matchActiveRms(rendered, source, noiseOutput);
        auto backend = request.pitchBackend == PitchRenderBackend::mld5
            ? juce::String("mld5-single-pass-stable-clock")
            : request.pitchBackend == PitchRenderBackend::mld3
                ? juce::String("mld3-period-transition-clock")
            : request.pitchBackend == PitchRenderBackend::nsfHifigan
                ? (usedNsfModel ? juce::String("nsf-hifigan-onnx")
                                : juce::String("nsf-hifigan-fallback"))
            : request.pitchBackend == PitchRenderBackend::llsm2
                ? (usedLlsm2 ? juce::String("llsm2-direct")
                             : juce::String("llsm2-fallback"))
            : request.pitchBackend == PitchRenderBackend::world ? juce::String("WORLD")
            : juce::String("vslib");
        if (usedNsfModel && nsfInference.isNotEmpty()) backend << "[" << nsfInference << "]";
        if (request.normalizeVolume) backend << "+normalized";
        backend += request.stretchAlgorithm == 5 ? "+hifishifter-mel"
            : request.stretchAlgorithm == 1 ? "+variable-mel-hop-splice-first"
            : request.stretchAlgorithm == 4 ? "+nsf-shift-then-splice"
            : request.stretchAlgorithm == 2 ? "+loop"
            : request.stretchAlgorithm == 3 ? "+soundtouch"
            : "+melodyne-hybrid";
        if (separate)
        {
            if (noise.getNumSamples() != rendered.getNumSamples()
                || noise.getNumChannels() != rendered.getNumChannels())
            {
                RenderedAudio failure;
                failure.warning = "Selected backend returned no separated component";
                if (completion) completion(std::move(failure));
                return jobHasFinished;
            }
            if (request.exportComponent == WavExportComponent::breath)
                rendered = std::move(noise);
            else
                for (int channel = 0; channel < rendered.getNumChannels(); ++channel)
                    rendered.addFrom(channel, 0, noise, channel, 0, rendered.getNumSamples(), -1.0f);
            backend += request.exportComponent == WavExportComponent::breath ? "+breath" : "+non-breath";
        }
        RenderedAudio result { std::move(rendered), reader->sampleRate, backend };
        if (shouldExit()) return jobHasFinished;
        // AudioEngine publishes the completed buffer through release/acquire
        // atomics, so this completion is thread-safe and does not need the GUI
        // message queue.  Calling it here also lets headless MCP/CLI rendering
        // finish while their stdin loop occupies the JUCE message thread.
        if (completion) completion(std::move(result));
        return jobHasFinished;
    }

private:
    Mld5FileRenderRequest request;
    FileCompletion completion;
};

class RenderService::UtauRenderJob final : public juce::ThreadPoolJob
{
public:
    UtauRenderJob(UtauRenderRequest requestToUse, FileCompletion completionToUse)
        : ThreadPoolJob("utau-phrase-render"), request(std::move(requestToUse)),
          completion(std::move(completionToUse))
    {
    }

    JobStatus runJob() override
    {
        if (shouldExit()) return jobHasFinished;
        const auto cancelled = request.cancelled;
        request.cancelled = [this, cancelled] { return shouldExit() || (cancelled && cancelled()); };
        auto rendered = UtauRenderer::render(request);
        if (shouldExit()) return jobHasFinished;
        if (rendered.warning.isNotEmpty())
            juce::Logger::writeToLog("UTAU: " + rendered.warning);
        if (completion)
            completion({ std::move(rendered.buffer), rendered.sampleRate,
                         std::move(rendered.backend), std::move(rendered.warning) });
        return jobHasFinished;
    }

private:
    UtauRenderRequest request;
    FileCompletion completion;
};

class RenderService::NsfUtauRenderJob final : public juce::ThreadPoolJob
{
public:
    NsfUtauRenderJob(UtauRenderRequest requestToUse, juce::File modelDirectoryToUse,
                     OrtExecutionConfig executionToUse, FileCompletion completionToUse)
        : ThreadPoolJob("nsf-utau-phrase-render"), request(std::move(requestToUse)),
          modelDirectory(std::move(modelDirectoryToUse)),
          execution(std::move(executionToUse)),
          completion(std::move(completionToUse))
    {
    }

    JobStatus runJob() override
    {
        if (shouldExit()) return jobHasFinished;
        const auto cancelled = request.cancelled;
        request.cancelled = [this, cancelled] { return shouldExit() || (cancelled && cancelled()); };
        auto rendered = renderNsfUtauPhrase(request, modelDirectory, execution);
        if (shouldExit()) return jobHasFinished;
        if (rendered.warning.isNotEmpty())
            juce::Logger::writeToLog("NSF-UTAU: " + rendered.warning);
        if (completion)
            completion({ std::move(rendered.buffer), rendered.sampleRate,
                         std::move(rendered.backend), std::move(rendered.warning) });
        return jobHasFinished;
    }

private:
    UtauRenderRequest request;
    juce::File modelDirectory;
    OrtExecutionConfig execution;
    FileCompletion completion;
};

RenderService::RenderService()
    : queue(std::clamp(juce::SystemStats::getNumCpus() - 1, 1, 4))
{
}

RenderService::~RenderService()
{
    cancelAll();
}

void RenderService::renderMld5(Mld5RenderRequest request, Completion completion)
{
    queue.add(std::make_unique<RenderJob>(std::move(request), std::move(completion)));
}

void RenderService::renderMld5File(Mld5FileRenderRequest request, FileCompletion completion,
                                 RenderSchedule schedule)
{
    const auto lane = request.pitchBackend == PitchRenderBackend::llsm2 ? RenderLane::llsm
        : request.pitchBackend == PitchRenderBackend::nsfHifigan ? RenderLane::neural : RenderLane::parallel;
    queue.add(std::make_unique<FileRenderJob>(std::move(request), std::move(completion)),
              std::move(schedule), lane);
}

void RenderService::renderUtau(UtauRenderRequest request, FileCompletion completion,
                              RenderSchedule schedule)
{
    const auto family = resolveRequestedVoicebankFamily(request);
    const auto ds = family.family == RequestedVoicebankFamily::diffSinger;
    if (std::isfinite(schedule.startSeconds))
        request.priorityPosition = [clock = queue.positionClock()]
        { return clock->load(std::memory_order_relaxed); };
    const auto lane = ds ? RenderLane::diffSinger
        : request.resamplerExecutable.existsAsFile() ? RenderLane::classicUtau : RenderLane::parallel;
    queue.add(std::make_unique<UtauRenderJob>(std::move(request), std::move(completion)),
              std::move(schedule), lane);
}

void RenderService::renderNsfUtau(UtauRenderRequest request, juce::File modelDirectory,
                                  OrtExecutionConfig execution, FileCompletion completion,
                                  RenderSchedule schedule)
{
    queue.add(std::make_unique<NsfUtauRenderJob>(std::move(request), std::move(modelDirectory),
                                     std::move(execution), std::move(completion)),
              std::move(schedule), RenderLane::neural);
}

void RenderService::cancelAll()
{
    queue.cancelAll();
}
}
