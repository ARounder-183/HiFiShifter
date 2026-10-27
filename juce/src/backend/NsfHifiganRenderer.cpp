#include "FcpeAnalyzer.h"
#include "SourceVoicing.h"
#include "NsfHifiganRenderer.h"
#include "NativeExpression.h"
#include "NsfRenderChunkCache.h"
#include "WcsNonLoopFeatures.h"
#include "ChineseCvvcPhonemizer.h"
#include "AmplitudeEnvelopeCurve.h"
#include "AdvancedEnvelope.h"
#include "HifisamplerFlags.h"
#include <juce_cryptography/juce_cryptography.h>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
#include <onnxruntime_cxx_api.h>
#endif

namespace hachi::backend
{
void conditionNsfPitchTransitions(std::vector<float>& f0, bool enabled)
{
    if (!enabled) return;
    // A hard voiced->unvoiced f0 step (e.g. 240 Hz -> 0 in one frame) makes
    // the NSF source filter switch its excitation abruptly and pops at the
    // vowel/consonant boundary.  Taper the frame that touches the edge so
    // the drop/rise is a short glide instead of an impulse.
    for (std::size_t frame = 1; frame + 1 < f0.size(); ++frame)
    {
        if (f0[frame] > 0.0f && f0[frame + 1] <= 0.0f && f0[frame - 1] > 0.0f)
            f0[frame] *= 0.55f;
        else if (f0[frame] <= 0.0f && f0[frame + 1] > 0.0f && f0[frame - 1] <= 0.0f)
            f0[frame] = f0[frame + 1] * 0.55f;
    }
    // A hard voiced->voiced f0 step (e.g. a Melodyne hard-disconnect split:
    // AudioEngine only glides connected notes) jumps the NSF excitation by
    // a musical interval in one hop, which the decoder turns into a click.
    // Glide the single transition frame to the geometric midpoint so the
    // interval becomes a ~2 frame portamento without smearing a real
    // glissando (neighbours are read from a snapshot to avoid cascading).
    const auto f0Snapshot = f0;
    for (std::size_t frame = 1; frame + 1 < f0.size(); ++frame)
    {
        const auto a = f0Snapshot[frame - 1];
        const auto b = f0Snapshot[frame];
        const auto c = f0Snapshot[frame + 1];
        if (!(a > 0.0f && b > 0.0f && c > 0.0f)) continue;
        const auto step = std::abs(12.0 * std::log2(c / a));
        if (step <= 1.5) continue;
        // A real glissando has the middle frame mid-slope (b roughly the
        // geometric mean of a and c), so it is flat against neither
        // neighbour.  Only a hard step sits flat against one side.
        const auto leftGap = std::abs(12.0 * std::log2(b / a));
        const auto rightGap = std::abs(12.0 * std::log2(c / b));
        if (leftGap < 0.5 || rightGap < 0.5)
            f0[frame] = std::sqrt(a * c);
    }
}

#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
namespace
{
// Each render owns its cancellation callback.  Nested helper calls (including
// HN-SEP) use the same token without cancelling another engine's session.
thread_local const std::function<bool()>* currentCancellation = nullptr;
struct NsfCancellationScope
{
    explicit NsfCancellationScope(const std::function<bool()>& callback)
        : previous(currentCancellation) { currentCancellation = &callback; }
    ~NsfCancellationScope() { currentCancellation = previous; }
    const std::function<bool()>* previous;
};
void checkNsfCancellation()
{
    if (currentCancellation && *currentCancellation && (*currentCancellation)())
        throw std::runtime_error("NSF-HiFiGAN render cancelled");
}
std::unique_lock<std::mutex> cancellableNsfLock(std::mutex& mutex)
{
    std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
    while (!lock.try_lock())
    {
        checkNsfCancellation();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    checkNsfCancellation();
    return lock;
}
// ORT Run is synchronous; signal its own RunOptions from a short-lived watcher
// so closing the editor interrupts an active decode rather than waiting for it.
class CancellableNsfRun final
{
public:
    CancellableNsfRun()
    {
        checkNsfCancellation();
        if (!currentCancellation || !*currentCancellation) return;
        const auto cancelled = *currentCancellation;
        watcher = std::thread([this, cancelled]
        {
            std::unique_lock lock(mutex);
            while (!changed.wait_for(lock, std::chrono::milliseconds(10), [this] { return finished; }))
            {
                lock.unlock();
                try
                {
                    if (cancelled()) { options.SetTerminate(); return; }
                }
                catch (...) { try { options.SetTerminate(); } catch (...) {} return; }
                lock.lock();
            }
        });
    }
    ~CancellableNsfRun()
    {
        { const std::scoped_lock lock(mutex); finished = true; }
        changed.notify_all();
        if (watcher.joinable()) watcher.join();
    }
    Ort::RunOptions options;
private:
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    std::thread watcher;
};

struct Config
{
    int sampleRate = 44'100;
    int melBands = 128;
    int hop = 512;
    int fftSize = 2'048;
    int windowSize = 2'048;
    float minimumHz = 40.0f;
    float maximumHz = 16'000.0f;
};

struct ModelFiles
{
    juce::File onnx;
    juce::File config;
};

float curveAt(const std::vector<float>& curve, double position)
{
    if (curve.empty() || !std::isfinite(position) || position < 0.0) return 0.0f;
    const auto left = static_cast<std::size_t>(std::floor(position));
    if (left >= curve.size()) return 0.0f;
    const auto right = std::min(left + 1, curve.size() - 1);
    const auto amount = static_cast<float>(position - static_cast<double>(left));
    const auto a = curve[left];
    const auto b = curve[right];
    if (!(std::isfinite(a) && std::isfinite(b))) return 0.0f;
    const auto aVoiced = a > 0.0f;
    const auto bVoiced = b > 0.0f;
    // Zero is a voicing boundary, not MIDI 0.  Linear interpolation across
    // that boundary creates a brief false sub-bass F0 at note endings.
    if (aVoiced != bVoiced) return amount < 0.5f ? a : b;
    return aVoiced ? a + (b - a) * amount : 0.0f;
}

float scalarCurveAt(const std::vector<float>& curve, double position, float fallback = 0.0f)
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

float midiToHz(float midi)
{
    return std::isfinite(midi) && midi > 0.0f
        ? 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f) : 0.0f;
}

std::optional<ModelFiles> modelFiles(const juce::File& configured)
{
    const auto validate = [](juce::File directory) -> std::optional<ModelFiles>
    {
        // Settings may point to a model file with its original release name.
        const auto selectedOnnx = directory.existsAsFile() && directory.hasFileExtension("onnx")
            ? directory : juce::File{};
        if (directory.existsAsFile()) directory = directory.getParentDirectory();
        const auto onnx = selectedOnnx != juce::File{} ? selectedOnnx
            : directory.getChildFile("pc_nsf_hifigan.onnx");
        const auto config = directory.getChildFile("config.json");
        if (onnx.existsAsFile() && config.existsAsFile()) return ModelFiles { onnx, config };
        return std::nullopt;
    };
    if (configured != juce::File{})
        if (auto files = validate(configured)) return files;
    if (const auto value = juce::SystemStats::getEnvironmentVariable(
            "HACHISHIFTER_NSF_HIFIGAN_MODEL_DIR", {}); value.isNotEmpty())
        if (auto files = validate(juce::File(value))) return files;
    if (const auto value = juce::SystemStats::getEnvironmentVariable(
            "HACHISHIFTER_NSF_HIFIGAN_ONNX", {}); value.isNotEmpty())
    {
        const juce::File onnx(value);
        auto config = onnx.getSiblingFile("config.json");
        const auto configuredConfig = juce::SystemStats::getEnvironmentVariable(
            "HACHISHIFTER_NSF_HIFIGAN_CONFIG", {});
        if (configuredConfig.isNotEmpty()) config = juce::File(configuredConfig);
        if (onnx.existsAsFile() && config.existsAsFile()) return ModelFiles { onnx, config };
    }
    if (auto files = validate(juce::File::getSpecialLocation(
            juce::File::currentExecutableFile).getParentDirectory()
                .getChildFile("models").getChildFile("nsf_hifigan")))
        return files;
    return std::nullopt;
}

std::optional<Config> readConfig(const juce::File& file, juce::String& error)
{
    const auto parsed = juce::JSON::parse(file.loadFileAsString());
    const auto* object = parsed.getDynamicObject();
    if (object == nullptr)
    {
        error = "NSF-HiFiGAN config.json is not a JSON object";
        return std::nullopt;
    }
    Config result;
    const auto integer = [&](const char* name, int fallback)
    {
        const auto value = object->getProperty(name);
        return value.isInt() || value.isInt64() || value.isDouble()
            ? static_cast<int>(value) : fallback;
    };
    const auto number = [&](const char* name, float fallback)
    {
        const auto value = object->getProperty(name);
        return value.isInt() || value.isInt64() || value.isDouble()
            ? static_cast<float>(value) : fallback;
    };
    result.sampleRate = integer("sampling_rate", result.sampleRate);
    result.melBands = integer("num_mels", result.melBands);
    result.hop = integer("hop_size", result.hop);
    result.fftSize = integer("n_fft", result.fftSize);
    result.windowSize = integer("win_size", result.windowSize);
    result.minimumHz = number("fmin", result.minimumHz);
    result.maximumHz = number("fmax", result.maximumHz);
    const auto powerOfTwo = result.fftSize > 0
        && (result.fftSize & (result.fftSize - 1)) == 0;
    if (result.sampleRate < 8'000 || result.melBands <= 0 || result.hop <= 0
        || !powerOfTwo || result.windowSize <= 0 || result.windowSize > result.fftSize
        || result.minimumHz < 0.0f || result.maximumHz <= result.minimumHz)
    {
        error = "NSF-HiFiGAN config.json contains invalid audio parameters";
        return std::nullopt;
    }
    result.maximumHz = std::min(result.maximumHz, result.sampleRate * 0.5f);
    return result;
}

std::vector<float> resample(const float* input, int samples, int inputRate, int outputRate)
{
    if (input == nullptr || samples <= 0 || inputRate <= 0 || outputRate <= 0) return {};
    if (inputRate == outputRate) return { input, input + samples };
    const auto ratio = static_cast<double>(outputRate) / inputRate;
    const auto outputSamples = std::max(1, static_cast<int>(std::llround(samples * ratio)));
    std::vector<float> output(static_cast<std::size_t>(outputSamples));
    // Linear resampling noticeably tilts a vocal spectral envelope whenever a
    // model pack uses a rate different from the project.  That error is then
    // interpreted by the neural source/filter decoder as a smaller vocal
    // tract (the reported child-like colour).  A compact band-limited Lanczos
    // conversion keeps the Mel envelope stable without another DSP library.
    constexpr int radius = 12;
    const auto cutoff = std::min(1.0, ratio);
    const auto sinc = [](double x)
    {
        if (std::abs(x) < 1.0e-10) return 1.0;
        const auto angle = juce::MathConstants<double>::pi * x;
        return std::sin(angle) / angle;
    };
    for (int index = 0; index < outputSamples; ++index)
    {
        if ((index & 255) == 0) checkNsfCancellation();
        const auto position = static_cast<double>(index) / ratio;
        const auto centre = static_cast<int>(std::floor(position));
        auto value = 0.0;
        auto weightSum = 0.0;
        for (int tap = centre - radius + 1; tap <= centre + radius; ++tap)
        {
            const auto distance = position - static_cast<double>(tap);
            if (std::abs(distance) >= radius) continue;
            const auto weight = cutoff * sinc(distance * cutoff)
                * sinc(distance / static_cast<double>(radius));
            const auto sourceIndex = juce::jlimit(0, samples - 1, tap);
            value += static_cast<double>(input[sourceIndex]) * weight;
            weightSum += weight;
        }
        output[static_cast<std::size_t>(index)] = static_cast<float>(
            std::abs(weightSum) > 1.0e-12 ? value / weightSum : 0.0);
    }
    return output;
}

std::size_t reflectIndex(std::ptrdiff_t index, std::size_t size)
{
    if (size <= 1) return 0;
    const auto period = static_cast<std::ptrdiff_t>(2 * (size - 1));
    auto reflected = index % period;
    if (reflected < 0) reflected += period;
    return reflected < static_cast<std::ptrdiff_t>(size)
        ? static_cast<std::size_t>(reflected)
        : static_cast<std::size_t>(period - reflected);
}

float hzToMel(float hz)
{
    constexpr auto spacing = 200.0f / 3.0f;
    constexpr auto logStartHz = 1'000.0f;
    constexpr auto logStartMel = logStartHz / spacing;
    const auto logStep = std::log(6.4f) / 27.0f;
    return hz >= logStartHz ? logStartMel + std::log(hz / logStartHz) / logStep
                            : hz / spacing;
}

float melToHz(float mel)
{
    constexpr auto spacing = 200.0f / 3.0f;
    constexpr auto logStartHz = 1'000.0f;
    constexpr auto logStartMel = logStartHz / spacing;
    const auto logStep = std::log(6.4f) / 27.0f;
    return mel >= logStartMel ? logStartHz * std::exp(logStep * (mel - logStartMel))
                              : spacing * mel;
}

std::vector<float> filterbank(const Config& config)
{
    const auto frequencies = config.fftSize / 2 + 1;
    const auto melMinimum = hzToMel(config.minimumHz);
    const auto melMaximum = hzToMel(config.maximumHz);
    std::vector<float> hzPoints(static_cast<std::size_t>(config.melBands + 2));
    for (int index = 0; index < config.melBands + 2; ++index)
        hzPoints[static_cast<std::size_t>(index)] = melToHz(melMinimum
            + (melMaximum - melMinimum) * static_cast<float>(index)
                / static_cast<float>(config.melBands + 1));
    std::vector<float> weights(static_cast<std::size_t>(config.melBands * frequencies));
    for (int band = 0; band < config.melBands; ++band)
    {
        const auto left = hzPoints[static_cast<std::size_t>(band)];
        const auto centre = hzPoints[static_cast<std::size_t>(band + 1)];
        const auto right = hzPoints[static_cast<std::size_t>(band + 2)];
        const auto normalization = 2.0f / std::max(1.0e-6f, right - left);
        for (int bin = 0; bin < frequencies; ++bin)
        {
            const auto frequency = static_cast<float>(bin * config.sampleRate) / config.fftSize;
            weights[static_cast<std::size_t>(band * frequencies + bin)] = std::max(0.0f,
                std::min((frequency - left) / std::max(1.0e-6f, centre - left),
                         (right - frequency) / std::max(1.0e-6f, right - centre)))
                * normalization;
        }
    }
    return weights;
}

struct MelData
{
    std::vector<float> values;
    std::size_t frames = 0;
};

// buildVariableHopSplicedMel applies the shift-first order before returning.
void shiftMelFormants(MelData& mel, const Config& config,
                      const std::vector<float>& formantSemitones,
                      double framePeriodMs, std::size_t frameOffset = 0);

MelData buildMelWithHop(const std::vector<float>& waveform, const Config& config,
                        int analysisHop)
{
    if (waveform.empty()) return {};
    analysisHop = std::max(1, analysisHop);
    auto leftPad = static_cast<std::size_t>(std::max(0,
        (config.windowSize - analysisHop) / 2));
    auto rightPad = static_cast<std::size_t>(std::max(0,
        (config.windowSize - analysisHop + 1) / 2));
    // A consonant/Attack segment can be shorter than one model hop.  The
    // training-style padding then still needs to provide a complete FFT
    // window; returning a synthetic silent Mel frame here caused a hard
    // spectral step at the consonant/vowel boundary.
    const auto paddedSamples = leftPad + waveform.size() + rightPad;
    if (paddedSamples < static_cast<std::size_t>(config.windowSize))
    {
        const auto missing = static_cast<std::size_t>(config.windowSize) - paddedSamples;
        leftPad += missing / 2;
        rightPad += missing - missing / 2;
    }
    std::vector<float> padded(leftPad + waveform.size() + rightPad);
    for (std::size_t index = 0; index < padded.size(); ++index)
        padded[index] = waveform[reflectIndex(static_cast<std::ptrdiff_t>(index)
            - static_cast<std::ptrdiff_t>(leftPad), waveform.size())];
    if (padded.size() < static_cast<std::size_t>(config.windowSize))
        return { std::vector<float>(static_cast<std::size_t>(config.melBands),
                                    std::log(1.0e-9f)), 1 };
    const auto frames = 1 + (padded.size() - static_cast<std::size_t>(config.windowSize))
        / static_cast<std::size_t>(analysisHop);
    const auto frequencies = config.fftSize / 2 + 1;
    const auto weights = filterbank(config);
    std::vector<float> window(static_cast<std::size_t>(config.windowSize));
    for (int index = 0; index < config.windowSize; ++index)
        window[static_cast<std::size_t>(index)] = 0.5f - 0.5f * std::cos(
            juce::MathConstants<float>::twoPi * index
                / static_cast<float>(std::max(1, config.windowSize - 1)));
    auto order = 0;
    for (auto size = config.fftSize; size > 1; size >>= 1) ++order;
    juce::dsp::FFT fft(order);
    std::vector<std::complex<float>> input(static_cast<std::size_t>(config.fftSize));
    std::vector<std::complex<float>> spectrum(static_cast<std::size_t>(config.fftSize));
    std::vector<float> magnitude(static_cast<std::size_t>(frequencies));
    std::vector<float> mel(static_cast<std::size_t>(config.melBands) * frames);
    for (std::size_t frame = 0; frame < frames; ++frame)
    {
        checkNsfCancellation();
        std::fill(input.begin(), input.end(), std::complex<float>{});
        for (int index = 0; index < config.windowSize; ++index)
            input[static_cast<std::size_t>(index)] = {
                padded[frame * static_cast<std::size_t>(analysisHop)
                    + static_cast<std::size_t>(index)] * window[static_cast<std::size_t>(index)], 0.0f
            };
        fft.perform(input.data(), spectrum.data(), false);
        for (int bin = 0; bin < frequencies; ++bin)
            magnitude[static_cast<std::size_t>(bin)] = std::abs(
                spectrum[static_cast<std::size_t>(bin)]);
        for (int band = 0; band < config.melBands; ++band)
        {
            auto sum = 0.0f;
            for (int bin = 0; bin < frequencies; ++bin)
                sum += weights[static_cast<std::size_t>(band * frequencies + bin)]
                    * magnitude[static_cast<std::size_t>(bin)];
            // ONNX expects [batch, mel, time], so each band owns one contiguous row.
            mel[static_cast<std::size_t>(band) * frames + frame]
                = std::log(std::max(1.0e-9f, sum));
        }
    }
    return { std::move(mel), frames };
}

// Windowed Mel frame centred on an absolute source sample index.  Consecutive
// target frames may therefore use different source hops without segment-local
// padding or an additional time interpolation pass.
std::vector<float> melFrameAt(const std::vector<float>& waveform,
                              const Config& config, double centerSample,
                              const std::vector<float>& filterWeights,
                              const std::vector<float>& window)
{
    const auto size = static_cast<std::ptrdiff_t>(waveform.size());
    if (size <= 0 || config.windowSize <= 0 || config.fftSize <= 0)
        return {};
    auto order = 0;
    for (auto fft = config.fftSize; fft > 1; fft >>= 1) ++order;
    juce::dsp::FFT fft(order);
    std::vector<std::complex<float>> input(static_cast<std::size_t>(config.fftSize));
    std::vector<std::complex<float>> spectrum(static_cast<std::size_t>(config.fftSize));
    const auto windowStart = static_cast<std::ptrdiff_t>(std::llround(centerSample))
        - static_cast<std::ptrdiff_t>(config.windowSize) / 2;
    for (int index = 0; index < config.windowSize; ++index)
    {
        auto sourceIndex = windowStart + index;
        // Periodic reflection of the whole waveform, mirroring reflectIndex.
        if (size > 1)
        {
            const auto period = 2 * (size - 1);
            sourceIndex %= period;
            if (sourceIndex < 0) sourceIndex += period;
            if (sourceIndex >= size) sourceIndex = period - sourceIndex;
        }
        else sourceIndex = 0;
        input[static_cast<std::size_t>(index)] = {
            waveform[static_cast<std::size_t>(sourceIndex)]
                * window[static_cast<std::size_t>(index)], 0.0f };
    }
    fft.perform(input.data(), spectrum.data(), false);
    const auto frequencies = config.fftSize / 2 + 1;
    std::vector<float> mel(static_cast<std::size_t>(config.melBands));
    for (int band = 0; band < config.melBands; ++band)
    {
        auto sum = 0.0f;
        for (int bin = 0; bin < frequencies; ++bin)
            sum += filterWeights[static_cast<std::size_t>(band * frequencies + bin)]
                * std::abs(spectrum[static_cast<std::size_t>(bin)]);
        mel[static_cast<std::size_t>(band)] = std::log(std::max(1.0e-9f, sum));
    }
    return mel;
}

MelData interpolateMelTime(const MelData& source, int melBands,
                           std::size_t targetFrames)
{
    if (source.frames == 0 || targetFrames == 0 || melBands <= 0) return {};
    if (source.frames == targetFrames) return source;
    MelData result;
    result.frames = targetFrames;
    result.values.resize(static_cast<std::size_t>(melBands) * targetFrames);
    const auto scale = targetFrames <= 1 ? 0.0
        : static_cast<double>(source.frames - 1) / static_cast<double>(targetFrames - 1);
    for (int band = 0; band < melBands; ++band)
        for (std::size_t frame = 0; frame < targetFrames; ++frame)
        {
            const auto sourcePosition = static_cast<double>(frame) * scale;
            const auto left = std::min(source.frames - 1,
                static_cast<std::size_t>(std::floor(sourcePosition)));
            const auto right = std::min(source.frames - 1, left + 1);
            const auto amount = static_cast<float>(sourcePosition - std::floor(sourcePosition));
            const auto row = static_cast<std::size_t>(band) * source.frames;
            result.values[static_cast<std::size_t>(band) * targetFrames + frame]
                = source.values[row + left]
                + (source.values[row + right] - source.values[row + left]) * amount;
        }
    return result;
}

MelData spliceMelToTimeMap(const MelData& source, int melBands,
                           std::size_t targetFrames, double targetHopSeconds,
                           double sourceHopSeconds,
                           const std::vector<NsfHifiganTimeMapPoint>& timeMap,
                           double targetFrameOffset = 0.0,
                           double sourceFrameOffset = 0.0)
{
    if (timeMap.size() < 2 || source.frames == 0 || targetFrames == 0
        || targetHopSeconds <= 0.0 || sourceHopSeconds <= 0.0)
        return interpolateMelTime(source, melBands, targetFrames);
    MelData result;
    result.frames = targetFrames;
    result.values.resize(static_cast<std::size_t>(melBands) * targetFrames);
    std::size_t segment = 0;
    for (std::size_t frame = 0; frame < targetFrames; ++frame)
    {
        checkNsfCancellation();
        const auto targetSeconds = (static_cast<double>(frame) + targetFrameOffset)
            * targetHopSeconds;
        while (segment + 1 < timeMap.size()
               && targetSeconds >= timeMap[segment + 1].targetSeconds)
            ++segment;
        const auto rightIndex = std::min(segment + 1, timeMap.size() - 1);
        const auto& leftPoint = timeMap[std::min(segment, timeMap.size() - 1)];
        const auto& rightPoint = timeMap[rightIndex];
        const auto duration = rightPoint.targetSeconds - leftPoint.targetSeconds;
        const auto amount = duration > 1.0e-9
            ? juce::jlimit(0.0, 1.0,
                (targetSeconds - leftPoint.targetSeconds) / duration)
            : 0.0;
        const auto sourceSeconds = leftPoint.sourceSeconds
            + (rightPoint.sourceSeconds - leftPoint.sourceSeconds) * amount;
        const auto sourcePosition = juce::jlimit(0.0,
            static_cast<double>(source.frames - 1),
            sourceSeconds / sourceHopSeconds - sourceFrameOffset);
        const auto sourceLeft = static_cast<std::size_t>(std::floor(sourcePosition));
        const auto sourceRight = std::min(source.frames - 1, sourceLeft + 1);
        const auto sourceAmount = static_cast<float>(sourcePosition - std::floor(sourcePosition));
        for (int band = 0; band < melBands; ++band)
        {
            const auto row = static_cast<std::size_t>(band) * source.frames;
            result.values[static_cast<std::size_t>(band) * targetFrames + frame]
                = source.values[row + sourceLeft]
                + (source.values[row + sourceRight] - source.values[row + sourceLeft])
                    * sourceAmount;
        }
    }
    return result;
}

// HiFiShifter main 6432687: round(T_in / playback_rate), then align the
// first/last Mel frames and linearly interpolate. Ordinary two-anchor clips
// exactly follow that stretch rule. Interior anchors extend it to this editor's
// nonuniform note timing; duplicate-time anchors retain source discontinuities.
MelData stretchHiFiShifterMel(const MelData& source, int bands,
                             double sourceSeconds, double targetSeconds,
                             const std::vector<NsfHifiganTimeMapPoint>& timeMap)
{
    if (source.frames == 0 || sourceSeconds <= 0 || targetSeconds <= 0) return {};
    const auto frames = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::llround(static_cast<double>(source.frames) * targetSeconds / sourceSeconds)));
    if (source.frames <= 1 || frames <= 1 || timeMap.size() < 2)
        return interpolateMelTime(source, bands, frames);
    return spliceMelToTimeMap(source, bands, frames,
        targetSeconds / static_cast<double>(frames - 1),
        sourceSeconds / static_cast<double>(source.frames - 1), timeMap);
}

// Match upstream's two-hop tail ramp before padding/truncating to the exact
// timeline duration; frame-count rounding can leave more than one hop at high
// stretch ratios. Do not treat that expected shortfall as an inference failure.
void alignHiFiShifterTail(std::vector<float>& output, std::size_t target, std::size_t ramp)
{
    if (output.size() == target) return;
    const auto end = std::min(output.size(), target);
    const auto start = end > ramp ? end - ramp : 0;
    const auto count = end - start;
    if (count >= 2)
        for (auto i = start; i < end; ++i)
            output[i] *= static_cast<float>(end - i) / static_cast<float>(count);
    output.resize(target, 0.0f);
}

void normalizeMelToReference(MelData& mel, const MelData& reference, int melBands)
{
    if (mel.frames == 0 || mel.frames != reference.frames || melBands <= 0) return;
    std::vector<float> correction(mel.frames, 0.0f);
    for (std::size_t frame = 0; frame < mel.frames; ++frame)
    {
        checkNsfCancellation();
        auto current = 0.0;
        auto target = 0.0;
        for (int band = 0; band < melBands; ++band)
        {
            const auto offset = static_cast<std::size_t>(band) * mel.frames + frame;
            current += std::exp(static_cast<double>(mel.values[offset]));
            target += std::exp(static_cast<double>(reference.values[offset]));
        }
        if (current > 1.0e-10 && target > 1.0e-10)
        {
            const auto maximumCorrection = 2.0 * std::log(10.0) / 20.0;
            correction[frame] = static_cast<float>(std::clamp(
                std::log(target / current), -maximumCorrection, maximumCorrection));
        }
    }
    const auto raw = correction;
    constexpr std::ptrdiff_t radius = 2;
    for (std::size_t frame = 0; frame < mel.frames; ++frame)
    {
        checkNsfCancellation();
        auto sum = 0.0f;
        auto count = 0;
        const auto centre = static_cast<std::ptrdiff_t>(frame);
        for (auto offset = -radius; offset <= radius; ++offset)
        {
            const auto index = centre + offset;
            if (index < 0 || index >= static_cast<std::ptrdiff_t>(mel.frames)) continue;
            sum += raw[static_cast<std::size_t>(index)];
            ++count;
        }
        correction[frame] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
    for (int band = 0; band < melBands; ++band)
        for (std::size_t frame = 0; frame < mel.frames; ++frame)
            mel.values[static_cast<std::size_t>(band) * mel.frames + frame]
                += correction[frame];
}

MelData buildVariableHopSplicedMel(const std::vector<float>& waveform,
                                   const Config& config,
                                   std::size_t targetFrames,
                                   const std::vector<NsfHifiganTimeMapPoint>& timeMap,
                                   bool shiftBeforeSplice = false,
                                   const std::vector<float>* formantSemitones = nullptr,
                                   double framePeriodMs = 5.0)
{
    if (targetFrames == 0 || waveform.empty()) return {};
    const auto targetHopSeconds = static_cast<double>(config.hop) / config.sampleRate;
    if (timeMap.size() < 2)
        return interpolateMelTime(buildMelWithHop(waveform, config, config.hop),
                                  config.melBands, targetFrames);
    MelData result;
    result.frames = targetFrames;
    result.values.assign(static_cast<std::size_t>(config.melBands) * targetFrames,
                         std::log(1.0e-9f));

    // Analyse one non-uniformly spaced source window for each target Mel frame.
    // Melodyne time maps can be much denser than the model hop (for example,
    // 173 anchors for about 30 target frames).  Quantising every anchor pair to
    // targetBegin/targetEnd made five or six source segments overwrite the same
    // target frame, leaving hard spectral jumps that the NSF decoder amplified
    // into full-scale bursts.  Sampling the complete map once per target frame
    // preserves every timing edit without creating overlapping Mel segments.
    std::vector<float> window(static_cast<std::size_t>(config.windowSize));
    for (int index = 0; index < config.windowSize; ++index)
        window[static_cast<std::size_t>(index)] = 0.5f - 0.5f * std::cos(
            juce::MathConstants<float>::twoPi * index
                / static_cast<float>(std::max(1, config.windowSize - 1)));
    const auto filterWeights = filterbank(config);
    std::vector<double> seamSeconds;
    for (std::size_t index = 1; index < timeMap.size(); ++index)
        if (std::abs(timeMap[index].targetSeconds - timeMap[index - 1].targetSeconds) <= 1.0e-7
            && std::abs(timeMap[index].sourceSeconds - timeMap[index - 1].sourceSeconds)
                > targetHopSeconds * 0.5)
            seamSeconds.push_back(timeMap[index].targetSeconds);
    std::size_t segment = 0;
    for (std::size_t frame = 0; frame < targetFrames; ++frame)
    {
        checkNsfCancellation();
        const auto targetSeconds = (static_cast<double>(frame) + 0.5)
            * targetHopSeconds;
        while (segment + 1 < timeMap.size()
               && targetSeconds >= timeMap[segment + 1].targetSeconds)
            ++segment;
        const auto rightIndex = std::min(segment + 1, timeMap.size() - 1);
        const auto& leftPoint = timeMap[std::min(segment, timeMap.size() - 1)];
        const auto& rightPoint = timeMap[rightIndex];
        const auto duration = rightPoint.targetSeconds - leftPoint.targetSeconds;
        const auto amount = duration > 1.0e-9
            ? juce::jlimit(0.0, 1.0,
                (targetSeconds - leftPoint.targetSeconds) / duration)
            : 0.0;
        const auto sourceSeconds = leftPoint.sourceSeconds
            + (rightPoint.sourceSeconds - leftPoint.sourceSeconds) * amount;
        const auto centerSample = juce::jlimit(0.0,
            static_cast<double>(waveform.size() - 1),
            sourceSeconds * config.sampleRate);
        const auto frameMel = melFrameAt(waveform, config, centerSample,
                                         filterWeights, window);
        if (frameMel.size() != static_cast<std::size_t>(config.melBands)) continue;
        for (int band = 0; band < config.melBands; ++band)
            result.values[static_cast<std::size_t>(band) * targetFrames + frame]
                = frameMel[static_cast<std::size_t>(band)];
    }

    // Shift-first applies its per-frame frequency mapping before the explicit
    // source discontinuity is blended below.  Splice-first is shifted by the
    // caller after this function returns.
    if (shiftBeforeSplice && formantSemitones != nullptr
        && !formantSemitones->empty())
        shiftMelFormants(result, config, *formantSemitones, framePeriodMs);

    // A duplicate target anchor represents a real element splice: source time
    // jumps at that instant instead of interpolating through the gap/overlap.
    // Blend only the two Mel frames on either side so the decoder does not turn
    // the valid discontinuity into a burst followed by limiter gain reduction.
    constexpr std::size_t seamRadius = 2;
    for (const auto seam : seamSeconds)
    {
        const auto firstAfter = static_cast<std::size_t>(std::clamp<juce::int64>(
            static_cast<juce::int64>(std::floor(seam / targetHopSeconds + 0.5)),
            0, static_cast<juce::int64>(targetFrames - 1)));
        const auto begin = firstAfter > seamRadius ? firstAfter - seamRadius : 0;
        const auto end = std::min(targetFrames - 1, firstAfter + seamRadius - 1);
        if (end <= begin) continue;
        for (int band = 0; band < config.melBands; ++band)
        {
            const auto row = static_cast<std::size_t>(band) * targetFrames;
            const auto left = result.values[row + begin];
            const auto right = result.values[row + end];
            const auto leftAmplitude = std::exp(static_cast<double>(left));
            const auto rightAmplitude = std::exp(static_cast<double>(right));
            for (auto frame = begin + 1; frame < end; ++frame)
            {
                const auto amount = static_cast<float>(frame - begin)
                    / static_cast<float>(end - begin);
                const auto smooth = amount * amount * (3.0f - 2.0f * amount);
                result.values[row + frame] = static_cast<float>(std::log(std::max(1.0e-9,
                    leftAmplitude + (rightAmplitude - leftAmplitude) * smooth)));
            }
        }
    }
    return result;
}

std::vector<float> stableMonoInput(const juce::AudioBuffer<float>& source)
{
    const auto channels = source.getNumChannels();
    const auto samples = source.getNumSamples();
    std::vector<float> mono(static_cast<std::size_t>(std::max(0, samples)));
    if (channels <= 0 || samples <= 0) return mono;
    if (channels == 1)
    {
        std::copy_n(source.getReadPointer(0), samples, mono.begin());
        return mono;
    }

    const auto* left = source.getReadPointer(0);
    const auto* right = source.getReadPointer(1);
    auto leftPower = 0.0;
    auto rightPower = 0.0;
    auto crossPower = 0.0;
    for (int sample = 0; sample < samples; ++sample)
    {
        leftPower += static_cast<double>(left[sample]) * left[sample];
        rightPower += static_cast<double>(right[sample]) * right[sample];
        crossPower += static_cast<double>(left[sample]) * right[sample];
    }
    const auto denominator = std::sqrt(leftPower * rightPower);
    const auto correlation = denominator > 1.0e-12 ? crossPower / denominator : 1.0;
    if (correlation >= 0.35)
        for (int sample = 0; sample < samples; ++sample)
            mono[static_cast<std::size_t>(sample)] = 0.5f * (left[sample] + right[sample]);
    else
    {
        // Phase-widened stereo vocals can lose their low/mid spectral envelope
        // when averaged, which the decoder presents as a smaller/child-like
        // vocal tract.  Use the more energetic intact channel in that case.
        const auto* selected = leftPower >= rightPower ? left : right;
        std::copy_n(selected, samples, mono.begin());
    }
    return mono;
}

void addModelEdgeContext(MelData& mel, std::vector<float>& f0, int melBands,
                         std::size_t contextFrames,
                         const NsfHifiganEdgeGuard& edgeGuard = {})
{
    if (mel.frames == 0 || f0.size() != mel.frames || melBands <= 0
        || contextFrames == 0) return;
    const auto originalFrames = mel.frames;
    MelData padded;
    padded.frames = originalFrames + contextFrames * 2;
    padded.values.resize(static_cast<std::size_t>(melBands) * padded.frames);
    std::vector<float> paddedF0(padded.frames);
    for (std::size_t frame = 0; frame < padded.frames; ++frame)
    {
        checkNsfCancellation();
        const auto relative = static_cast<std::ptrdiff_t>(frame)
            - static_cast<std::ptrdiff_t>(contextFrames);
        const auto sourceFrame = reflectIndex(relative, originalFrames);
        paddedF0[frame] = f0[sourceFrame];
        for (int band = 0; band < melBands; ++band)
            padded.values[static_cast<std::size_t>(band) * padded.frames + frame]
                = mel.values[static_cast<std::size_t>(band) * originalFrames + sourceFrame];
    }
    // Interpolate F0 at edges when neighbor F0 is known (same-source
    // pitch-split).  The model then sees a smooth pitch glide across the
    // boundary instead of a hard reflected step.
    if (edgeGuard.neighborStartF0 > 0.0f && f0.front() > 0.0f)
        for (std::size_t frame = 0; frame < contextFrames; ++frame)
        {
            const auto amount = static_cast<float>(frame + 1)
                / static_cast<float>(contextFrames + 1);
            paddedF0[frame] = edgeGuard.neighborStartF0
                + (f0.front() - edgeGuard.neighborStartF0) * amount;
        }
    if (edgeGuard.neighborEndF0 > 0.0f && f0.back() > 0.0f)
    {
        const auto offset = originalFrames + contextFrames;
        for (std::size_t frame = 0; frame < contextFrames; ++frame)
        {
            const auto amount = static_cast<float>(frame + 1)
                / static_cast<float>(contextFrames + 1);
            paddedF0[offset + frame] = f0.back()
                + (edgeGuard.neighborEndF0 - f0.back()) * amount;
        }
    }
    mel = std::move(padded);
    f0 = std::move(paddedF0);
}

void shiftMelFormants(MelData& mel, const Config& config,
                      const std::vector<float>& formantSemitones,
                      double framePeriodMs, std::size_t frameOffset)
{
    if (mel.frames == 0 || formantSemitones.empty()) return;
    const auto melMinimum = hzToMel(std::max(0.0f, config.minimumHz));
    const auto melMaximum = hzToMel(std::max(config.minimumHz + 1.0f, config.maximumHz));
    const auto melRange = std::max(1.0e-9f, melMaximum - melMinimum);
    const auto bands = static_cast<float>(config.melBands);
    const auto silence = std::log(1.0e-9f);
    std::vector<float> centres(static_cast<std::size_t>(config.melBands));
    std::vector<float> column(static_cast<std::size_t>(config.melBands));
    for (int band = 0; band < config.melBands; ++band)
        centres[static_cast<std::size_t>(band)] = melToHz(melMinimum
            + (static_cast<float>(band) + 1.0f) * melRange / (bands + 1.0f));
    const auto hopSeconds = static_cast<double>(config.hop) / config.sampleRate;
    const auto curvePeriod = std::max(0.1, framePeriodMs);
    for (std::size_t frame = 0; frame < mel.frames; ++frame)
    {
        checkNsfCancellation();
        const auto shift = scalarCurveAt(formantSemitones,
            static_cast<double>(frameOffset + frame) * hopSeconds * 1000.0 / curvePeriod);
        if (!std::isfinite(shift) || std::abs(shift) < 5.0e-4f) continue;
        const auto ratio = std::exp2(shift / 12.0f);
        if (!std::isfinite(ratio) || ratio <= 0.0f) continue;
        for (int band = 0; band < config.melBands; ++band)
            column[static_cast<std::size_t>(band)] = mel.values[
                static_cast<std::size_t>(band) * mel.frames + frame];
        for (int band = 0; band < config.melBands; ++band)
        {
            const auto sourceHz = centres[static_cast<std::size_t>(band)] / ratio;
            const auto sourceBin = (hzToMel(std::max(0.0f, sourceHz)) - melMinimum)
                / melRange * (bands + 1.0f) - 1.0f;
            const auto left = static_cast<int>(std::floor(sourceBin));
            auto value = silence;
            if (left >= 0 && left < config.melBands)
            {
                if (left == config.melBands - 1)
                    value = column[static_cast<std::size_t>(left)];
                else
                {
                    const auto amount = juce::jlimit(0.0f, 1.0f,
                        sourceBin - static_cast<float>(left));
                    value = column[static_cast<std::size_t>(left)]
                        + (column[static_cast<std::size_t>(left + 1)]
                           - column[static_cast<std::size_t>(left)]) * amount;
                }
            }
            mel.values[static_cast<std::size_t>(band) * mel.frames + frame] = value;
        }
    }
}

void applyNeuralEdgeGuard(float* samples, int count, double sampleRate,
                          double guardStartSeconds, double guardEndSeconds)
{
    if (samples == nullptr || count <= 0 || sampleRate <= 0.0) return;
    // Clip boundaries are not guaranteed to coincide with a decoder source
    // phase.  An equal-power guard suppresses that isolated discontinuity
    // without smearing attacks or creating an audible long crossfade.  Each
    // edge can be disabled: connected phrase seams are decoded continuously in
    // the stretch-splice-then-pitch order, and per-element seams are covered by
    // a longer mixer crossfade, so a baked-in 3 ms dip there only adds noise.
    const auto fadeIn = guardStartSeconds > 0.0
        ? std::min(count / 2, std::max(1,
            static_cast<int>(std::llround(sampleRate * guardStartSeconds)))) : 0;
    const auto fadeOut = guardEndSeconds > 0.0
        ? std::min(count / 2, std::max(1,
            static_cast<int>(std::llround(sampleRate * guardEndSeconds)))) : 0;
    for (int index = 0; index < fadeIn; ++index)
    {
        const auto phase = (static_cast<double>(index) + 1.0)
            / (static_cast<double>(fadeIn) + 1.0);
        const auto gainIn = static_cast<float>(std::sin(
            juce::MathConstants<double>::halfPi * phase));
        samples[index] *= gainIn;
    }
    for (int index = 0; index < fadeOut; ++index)
    {
        const auto phase = (static_cast<double>(index) + 1.0)
            / (static_cast<double>(fadeOut) + 1.0);
        const auto gainOut = static_cast<float>(std::sin(
            juce::MathConstants<double>::halfPi * phase));
        samples[count - 1 - index] *= gainOut;
    }
}

void conditionNeuralBoundary(float* samples, int count, double sampleRate,
                             double guardStartSeconds, double guardEndSeconds)
{
    if (samples == nullptr || count <= 0 || sampleRate <= 0.0) return;
    // Neural segments can carry a small DC step even when their Mel/F0 inputs
    // are continuous.  Remove it before the clip enters the project mixer.
    const auto pole = static_cast<float>(std::exp(
        -juce::MathConstants<double>::twoPi * 22.0 / sampleRate));
    auto previousInput = samples[0];
    auto previousOutput = 0.0f;
    for (int index = 0; index < count; ++index)
    {
        const auto input = samples[index];
        const auto output = input - previousInput + pole * previousOutput;
        samples[index] = std::isfinite(output) ? output : 0.0f;
        previousInput = input;
        previousOutput = samples[index];
    }
    // Repair isolated decoder impulses while leaving genuine multi-sample
    // consonant attacks untouched.  An NSF pop is a 1-3 sample broadband
    // spike: its value deviates far from the local mean while the immediate
    // neighbours stay smooth (a real attack ramps, it does not jump-and-hold).
    // A relative threshold makes the repair catch mid-level pops too, not only
    // near-full-scale single-sample clicks.
    for (int index = 2; index + 2 < count; ++index)
    {
        const auto localMean = 0.25f * (samples[index - 2] + samples[index - 1]
                                      + samples[index + 1] + samples[index + 2]);
        const auto localRms = std::sqrt(std::max(1.0e-9f,
            0.25f * (samples[index - 2] * samples[index - 2]
                   + samples[index - 1] * samples[index - 1]
                   + samples[index + 1] * samples[index + 1]
                   + samples[index + 2] * samples[index + 2])));
        const auto deviation = std::abs(samples[index] - localMean);
        const auto surroundingSmooth =
               std::abs(samples[index + 1] - samples[index - 1]) < 0.18f
            && std::abs(samples[index + 2] - samples[index - 2]) < 0.32f;
        if (deviation > std::max(0.16f, 2.5f * localRms) && surroundingSmooth)
            samples[index] = localMean;
    }
    applyNeuralEdgeGuard(samples,count,sampleRate,guardStartSeconds,guardEndSeconds);
}

#if JUCE_WINDOWS
using OrtPath = std::wstring;
OrtPath ortPath(const juce::File& file) { return file.getFullPathName().toWideCharPointer(); }
#else
using OrtPath = std::string;
OrtPath ortPath(const juce::File& file) { return file.getFullPathName().toStdString(); }
#endif

Ort::Env& environment()
{
    static Ort::Env value(ORT_LOGGING_LEVEL_WARNING, "hachishifter-nsf-hifigan");
    return value;
}

struct SessionEntry
{
    std::shared_ptr<Ort::Session> model;
    std::shared_ptr<std::mutex> runMutex;
    std::shared_ptr<NsfRenderChunkCache> nativeChunks = std::make_shared<NsfRenderChunkCache>();
    juce::String activeInference;
};

struct NsfSessionCache
{
    std::mutex mutex;
    std::unordered_map<std::string, SessionEntry> sessions;
};
NsfSessionCache& nsfSessionCache()
{
    // Construct the environment first so sessions are destroyed before it.
    (void)environment();
    static NsfSessionCache value;
    return value;
}
SessionEntry session(const juce::File& model, const OrtExecutionConfig& execution,
                     const juce::MemoryBlock* modelSnapshot = nullptr)
{
    auto& cache = nsfSessionCache();
    const auto lock = cancellableNsfLock(cache.mutex);
    auto& sessions = cache.sessions;
    const auto key = (model.getFullPathName() + "#"
        + juce::String(model.getLastModificationTime().toMilliseconds()) + "#"
        + juce::String(model.getSize()) + "#"
        + (modelSnapshot ? juce::SHA256(*modelSnapshot).toHexString() : juce::String{}) + "#"
        + inferenceBackendName(execution.requested) + "#"
        + juce::String(execution.deviceIndex) + "#"
        + juce::String(execution.intraOpThreads)).toStdString();
    if (const auto found = sessions.find(key); found != sessions.end())
        return found->second;
    SessionEntry value;
    auto options = makeOrtSessionOptions(execution, value.activeInference);
    value.model = modelSnapshot
        ? std::make_shared<Ort::Session>(environment(), modelSnapshot->getData(), modelSnapshot->getSize(), options)
        : std::make_shared<Ort::Session>(environment(), ortPath(model).c_str(), options);
    checkNsfCancellation();
    if (value.activeInference.startsWith("directml"))
        value.runMutex = std::make_shared<std::mutex>();
    sessions[key] = value;
    return value;
}

std::vector<float> infer(Ort::Session& model, const Config& config,
                         const MelData& mel, const std::vector<float>& f0,
                         std::mutex* runMutex, NsfRenderChunkCache* nativeChunks = nullptr,
                         NsfHifiganRenderResult* diagnostics = nullptr)
{
    Ort::AllocatorWithDefaultOptions allocator;
    if (model.GetInputCount() != 2 || model.GetOutputCount() < 1
        || mel.frames == 0 || f0.size() != mel.frames
        || mel.values.size() != static_cast<std::size_t>(config.melBands) * mel.frames)
        return {};
    const auto firstName = model.GetInputNameAllocated(0, allocator);
    const std::size_t melIndex = juce::String(firstName.get()).equalsIgnoreCase("f0") ? 1 : 0;
    const auto f0Index = 1 - melIndex;
    const auto melName = model.GetInputNameAllocated(melIndex, allocator);
    const auto f0Name = model.GetInputNameAllocated(f0Index, allocator);
    const auto outputName = model.GetOutputNameAllocated(0, allocator);
    const auto melType = model.GetInputTypeInfo(melIndex);
    const auto f0Type = model.GetInputTypeInfo(f0Index);
    const auto melInfo = melType.GetTensorTypeAndShapeInfo();
    const auto f0Info = f0Type.GetTensorTypeAndShapeInfo();
    const auto dimensions = melInfo.GetShape();
    const auto f0Dimensions = f0Info.GetShape();
    if (melInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || f0Info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || dimensions.size() != 3 || f0Dimensions.size() != 2
        || dimensions[0] > 1 || f0Dimensions[0] > 1)
        throw std::runtime_error("Unsupported NSF-HiFiGAN input types or dimensions");
    // OpenVPI's public ONNX model takes [batch, frames, mel bands]. Earlier
    // custom exports take [batch, mel bands, frames]. Keep the internal Mel
    // representation band-major and transpose only when the model needs it.
    const auto framesFirst = dimensions[2] == config.melBands;
    if (!framesFirst && dimensions[1] != config.melBands)
        throw std::runtime_error("NSF-HiFiGAN model Mel dimension does not match config.json");
    const char* inputNames[] { melName.get(), f0Name.get() };
    const char* outputNames[] { outputName.get() };
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Native edits typically affect seconds, not a 47-second tensor. Keep a
    // stable grid with the existing context/crossfade; UTAU keeps its old path.
    const std::size_t coreFrames = nativeChunks != nullptr ? 512 : 4'096;
    constexpr std::size_t contextFrames = 32;
    constexpr std::size_t overlapFrames = 16;
    const auto stepFrames = coreFrames - overlapFrames;
    std::vector<float> output(mel.frames * static_cast<std::size_t>(config.hop));
    std::vector<float> weight(output.size());
    for (std::size_t coreStart = 0; coreStart < mel.frames; coreStart += stepFrames)
    {
        checkNsfCancellation();
        const auto coreEnd = std::min(mel.frames, coreStart + coreFrames);
        const auto inputStart = coreStart > contextFrames ? coreStart - contextFrames : 0;
        const auto inputEnd = std::min(mel.frames, coreEnd + contextFrames);
        const auto inputFrames = inputEnd - inputStart;
        std::vector<float> melChunk(static_cast<std::size_t>(config.melBands) * inputFrames);
        for (int band = 0; band < config.melBands; ++band)
        {
            const auto sourceOffset = static_cast<std::size_t>(band) * mel.frames + inputStart;
            if (framesFirst)
                for (std::size_t frame = 0; frame < inputFrames; ++frame)
                    melChunk[frame * static_cast<std::size_t>(config.melBands)
                        + static_cast<std::size_t>(band)] = mel.values[sourceOffset + frame];
            else
                std::copy_n(mel.values.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                    inputFrames, melChunk.begin() + static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(band) * inputFrames));
        }
        std::vector<float> f0Chunk(f0.begin() + static_cast<std::ptrdiff_t>(inputStart),
                                   f0.begin() + static_cast<std::ptrdiff_t>(inputEnd));
        const auto crop = (coreStart - inputStart) * static_cast<std::size_t>(config.hop);
        const auto wanted = (coreEnd - coreStart) * static_cast<std::size_t>(config.hop);
        std::string chunkKey;
        NsfRenderChunkCache::Samples decoded;
        if (nativeChunks != nullptr)
        {
            chunkKey = (juce::String(config.hop) + ":" + juce::String(config.melBands)
                + ":" + juce::String(framesFirst ? 1 : 0) + ":" + juce::String(inputFrames)
                + ":" + juce::String(crop) + ":" + juce::String(wanted) + ":"
                + juce::SHA256(melChunk.data(), melChunk.size() * sizeof(float)).toHexString()
                + ":" + juce::SHA256(f0Chunk.data(), f0Chunk.size() * sizeof(float)).toHexString()).toStdString();
            decoded = nativeChunks->get(chunkKey);
        }
        if (decoded)
        {
            if (diagnostics != nullptr) ++diagnostics->reusedChunks;
        }
        else
        {
            const std::array<int64_t, 3> melShape {
                1, framesFirst ? static_cast<int64_t>(inputFrames) : config.melBands,
                framesFirst ? config.melBands : static_cast<int64_t>(inputFrames)
            };
            const std::array<int64_t, 2> f0Shape { 1, static_cast<int64_t>(inputFrames) };
            auto melTensor = Ort::Value::CreateTensor<float>(memory, melChunk.data(), melChunk.size(),
                                                              melShape.data(), melShape.size());
            auto f0Tensor = Ort::Value::CreateTensor<float>(memory, f0Chunk.data(), f0Chunk.size(),
                                                             f0Shape.data(), f0Shape.size());
            std::array<Ort::Value, 2> inputs { std::move(melTensor), std::move(f0Tensor) };
            std::unique_lock<std::mutex> runLock;
            if (runMutex != nullptr) runLock = cancellableNsfLock(*runMutex);
            CancellableNsfRun run;
            auto rendered = model.Run(run.options, inputNames, inputs.data(),
                                      inputs.size(), outputNames, 1);
            checkNsfCancellation();
            if (rendered.empty()) return {};
            const auto info = rendered[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return {};
            const auto count = info.GetElementCount();
            const auto* values = rendered[0].GetTensorData<float>();
            const auto available = count > crop ? std::min(wanted, count - crop) : 0;
            // A short tensor would otherwise leave a zero-filled hole while still
            // claiming that the neural backend succeeded.  Treat an incompatible
            // model/output shape as a render failure so the established fallback
            // remains audible for the complete clip.
            if (available != wanted) return {};
            decoded = std::make_shared<const std::vector<float>>(values + crop, values + crop + wanted);
            checkNsfCancellation();
            if (nativeChunks != nullptr) nativeChunks->put(chunkKey, decoded);
            if (diagnostics != nullptr) ++diagnostics->inferredChunks;
        }
        checkNsfCancellation();
        const auto overlapSamples = overlapFrames * static_cast<std::size_t>(config.hop);
        const auto outputOffset = coreStart * static_cast<std::size_t>(config.hop);
        for (std::size_t sample = 0; sample < wanted; ++sample)
        {
            auto blend = 1.0f;
            if (coreStart > 0 && sample < overlapSamples)
            {
                const auto phase = (static_cast<double>(sample) + 0.5)
                    / static_cast<double>(overlapSamples);
                blend *= static_cast<float>(std::sin(
                    juce::MathConstants<double>::halfPi * phase));
            }
            if (coreEnd < mel.frames && wanted - sample <= overlapSamples)
            {
                const auto phase = (static_cast<double>(wanted - sample) - 0.5)
                    / static_cast<double>(overlapSamples);
                blend *= static_cast<float>(std::sin(
                    juce::MathConstants<double>::halfPi * phase));
            }
            const auto destination = outputOffset + sample;
            output[destination] += (*decoded)[sample] * blend;
            weight[destination] += blend;
        }
        if (coreEnd == mel.frames) break;
    }
    for (std::size_t sample = 0; sample < output.size(); ++sample)
        if (weight[sample] > 1.0e-8f) output[sample] /= weight[sample];
    return output;
}
#include "HifisamplerDsp.inc"
}
#endif

void NsfHifiganRenderer::shutdown()
{
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    clearWcsNonLoopFeatureCache();
    auto& cache = nsfSessionCache();
    const std::scoped_lock lock(cache.mutex);
    cache.sessions.clear();
#endif
}

bool NsfHifiganRenderer::modelAvailable(const juce::File& configuredModelDirectory)
{
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    return modelFiles(configuredModelDirectory).has_value();
#else
    juce::ignoreUnused(configuredModelDirectory);
    return false;
#endif
}

NsfHifiganRenderResult NsfHifiganRenderer::render(
    const juce::AudioBuffer<float>& source, double sampleRate, int targetSamples,
    double framePeriodMs, const std::vector<float>& targetMidi,
    const std::vector<float>& formantSemitones,
    const std::vector<NsfHifiganTimeMapPoint>& timeMap,
    const juce::File& configuredModelDirectory,
    const OrtExecutionConfig& execution, NsfHifiganStretchOrder stretchOrder,
    bool normalizeVolume, const NsfHifiganEdgeGuard& edgeGuard,
    const std::function<bool()>& cancelled, bool protectNativeNoise, bool smoothPitchTransitions,
    bool incrementalNativeRender, const std::vector<float>& sourceMidi)
{
    NsfHifiganRenderResult result;
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    const NsfCancellationScope cancellation(cancelled);
    try
    {
        checkNsfCancellation();
        const auto files = modelFiles(configuredModelDirectory);
        if (!files)
        {
            result.error = "NSF-HiFiGAN model pack not found";
            return result;
        }
        result.modelFile = files->onnx;
        auto config = readConfig(files->config, result.error);
        if (!config) return result;
        auto ortSession = session(files->onnx, execution);
        result.activeInference = ortSession.activeInference;
        if (ortSession.model == nullptr)
        {
            result.error = "NSF-HiFiGAN ONNX session could not be created";
            return result;
        }
        const auto channels = source.getNumChannels();
        const auto sourceSamples = source.getNumSamples();
        if (channels <= 0 || sourceSamples <= 0 || targetSamples <= 0 || sampleRate <= 0.0)
        {
            result.error = "NSF-HiFiGAN received empty audio";
            return result;
        }
        result.buffer.setSize(channels, targetSamples);
        result.buffer.clear();
        // The public NSF model is mono.  Use a correlation-aware fold-down so
        // phase-widened stereo material keeps its vocal-tract envelope.
        auto mono = stableMonoInput(source);
        auto modelInput = resample(mono.data(), sourceSamples,
            static_cast<int>(std::llround(sampleRate)), config->sampleRate);
        const auto voicing = protectNativeNoise
            ? SourceVoicing::analyse(modelInput, config->sampleRate, checkNsfCancellation) : SourceVoicing{};
        std::vector<float> sourceNoise;
        if (protectNativeNoise && config->sampleRate == 44100 && hifiHnsepFile(*files).existsAsFile())
        {
            auto separated = hifiSeparate(modelInput, *files, execution, false);
            sourceNoise.resize(modelInput.size());
            for (std::size_t i=0; i<modelInput.size(); ++i)
                sourceNoise[i] = modelInput[i] - separated.harmonic[i];
            modelInput = std::move(separated.harmonic);
            result.usedHarmonicNoise = true;
        }
        // Keep noise on the original source clock, including free trim and
        // unequal per-note stretch. Duplicate-time anchors are source jumps.
        const auto sourceAt = [&](double seconds) {
            if (timeMap.size() < 2)
                return seconds * sourceSamples / targetSamples;
            auto right = std::upper_bound(timeMap.begin(),timeMap.end(),seconds,
                [](double t,const NsfHifiganTimeMapPoint& p) { return t < p.targetSeconds; });
            if (right == timeMap.begin()) return right->sourceSeconds;
            if (right == timeMap.end()) return timeMap.back().sourceSeconds;
            const auto& left = *(right-1);
            const auto span = right->targetSeconds-left.targetSeconds;
            const auto phase = span>1.e-12 ? juce::jlimit(0.0,1.0,(seconds-left.targetSeconds)/span) : 0.0;
            return left.sourceSeconds+(right->sourceSeconds-left.sourceSeconds)*phase;
        };
        const auto targetModelSamples = std::max<std::size_t>(1,
            static_cast<std::size_t>(std::llround(static_cast<double>(targetSamples)
                * config->sampleRate / sampleRate)));
        const auto targetFrames = std::max<std::size_t>(1,
            (targetModelSamples + static_cast<std::size_t>(config->hop) - 1)
                / static_cast<std::size_t>(config->hop));
        const auto hifiShifter = stretchOrder == NsfHifiganStretchOrder::hifiShifterMel;
        const auto wcsNonLoop = stretchOrder == NsfHifiganStretchOrder::wcsNonLoop;
        std::vector<float> wcsF0;
        MelData wcsMel;
        if (wcsNonLoop)
        {
            const auto hopSeconds = double(config->hop) / config->sampleRate;
            wcsF0.resize(targetFrames);
            for (std::size_t i = 0; i < targetFrames; ++i)
                wcsF0[i] = midiToHz(curveAt(targetMidi, i * hopSeconds * 1000. / std::max(.1, framePeriodMs)));
            const auto requested = wcsF0;
            conditionNsfPitchTransitions(wcsF0, smoothPitchTransitions);
            for (std::size_t i = 0; i < targetFrames; ++i)
                if (requested[i] <= 0 || (protectNativeNoise && voicing.at(sourceAt(i * hopSeconds)) >= .999f))
                    wcsF0[i] = 0;
            const auto sourceReference = buildMelWithHop(modelInput, *config, config->hop);
            WcsFeatureConfig featureConfig { config->sampleRate, config->hop, config->fftSize,
                config->windowSize, config->melBands, filterbank(*config) };
            auto features = buildWcsNonLoopMel(modelInput, featureConfig,
                double(targetSamples) / sampleRate, framePeriodMs, sourceMidi, wcsF0,
                timeMap, sourceReference.values, sourceReference.frames, checkNsfCancellation);
            wcsMel = { std::move(features.values), targetFrames };
            result.usedWcsParameters = true;
            result.reusedWcsSourceAnalysis = features.reusedSourceAnalysis;
        }
        const auto variableHop = stretchOrder == NsfHifiganStretchOrder::spliceThenShift
            || stretchOrder == NsfHifiganStretchOrder::shiftThenSplice;
        auto referenceMel = normalizeVolume
            ? spliceMelToTimeMap(buildMelWithHop(modelInput, *config, config->hop),
                config->melBands, targetFrames,
                static_cast<double>(config->hop) / config->sampleRate,
                static_cast<double>(config->hop) / config->sampleRate, timeMap,
                variableHop ? 0.5 : 0.0, variableHop ? 0.5 : 0.0)
            : MelData{};
        auto mel = wcsNonLoop ? std::move(wcsMel) : hifiShifter
            ? stretchHiFiShifterMel(buildMelWithHop(modelInput, *config, config->hop),
                config->melBands, static_cast<double>(sourceSamples) / sampleRate,
                static_cast<double>(targetSamples) / sampleRate, timeMap)
            : stretchOrder == NsfHifiganStretchOrder::spliceThenShift
            ? buildVariableHopSplicedMel(modelInput, *config, targetFrames, timeMap)
            : stretchOrder == NsfHifiganStretchOrder::shiftThenSplice
                ? buildVariableHopSplicedMel(modelInput, *config, targetFrames, timeMap,
                    true, &formantSemitones, framePeriodMs)
                : spliceMelToTimeMap(buildMelWithHop(modelInput, *config, config->hop),
                    config->melBands, targetFrames,
                    static_cast<double>(config->hop) / config->sampleRate,
                    static_cast<double>(config->hop) / config->sampleRate, timeMap);
        if (mel.frames == 0)
        {
            result.error = "NSF-HiFiGAN mel extraction returned no frames";
            result.buffer.setSize(0, 0);
            return result;
        }
        if (hifiShifter && referenceMel.frames != 0)
            referenceMel = mel;
        // Splice-first and fixed-hop orders shift the whole joined Mel once,
        // after the segments are blended; the shift-first order already applied
        // the formant curve inside buildVariableHopSplicedMel per segment.
        if (stretchOrder != NsfHifiganStretchOrder::shiftThenSplice)
            shiftMelFormants(mel, *config, formantSemitones, framePeriodMs);
        if (referenceMel.frames != 0)
            shiftMelFormants(referenceMel, *config, formantSemitones, framePeriodMs);
        if (normalizeVolume)
        {
            normalizeMelToReference(mel, referenceMel, config->melBands);
        }
        std::vector<float> f0 = wcsNonLoop ? std::move(wcsF0) : std::vector<float>(mel.frames);
        const auto hopSeconds = static_cast<double>(config->hop) / config->sampleRate;
        const auto framePeriod = std::max(0.1, framePeriodMs);
        for (std::size_t frame = 0; !wcsNonLoop && frame < mel.frames; ++frame)
        {
            const auto seconds = static_cast<double>(frame) * hopSeconds;
            f0[frame] = midiToHz(curveAt(targetMidi,
                seconds * 1000.0 / framePeriod));
        }
        const auto requestedF0 = f0;
        if (!wcsNonLoop) conditionNsfPitchTransitions(f0, smoothPitchTransitions);
        if (protectNativeNoise)
            for (std::size_t frame=0;frame<f0.size();++frame)
                if (requestedF0[frame] <= 0 || voicing.at(sourceAt(frame*hopSeconds)) >= .999f)
                {
                    f0[frame]=0; // Never invent a periodic source for sh/s/f/h.
                    ++result.protectedUnvoicedFrames;
                }
        constexpr std::size_t edgeContextFrames = 16;
        auto contextGuard=edgeGuard;
        if (!smoothPitchTransitions)
            contextGuard.neighborStartF0=contextGuard.neighborEndF0=0;
        addModelEdgeContext(mel, f0, config->melBands, edgeContextFrames, contextGuard);
        auto modelOutput = infer(*ortSession.model, *config, mel, f0,
                                 ortSession.runMutex.get(),
                                 (protectNativeNoise || incrementalNativeRender) ? ortSession.nativeChunks.get() : nullptr, &result);
        if (modelOutput.empty())
        {
            result.error = "NSF-HiFiGAN ONNX returned empty audio";
            result.buffer.setSize(0, 0);
            return result;
        }
        const auto contextSamples = edgeContextFrames * static_cast<std::size_t>(config->hop);
        const auto unpaddedSamples = (mel.frames - 2 * edgeContextFrames)
            * static_cast<std::size_t>(config->hop);
        if (modelOutput.size() < contextSamples + unpaddedSamples)
        {
            result.error = "NSF-HiFiGAN ONNX returned short context audio";
            result.buffer.setSize(0, 0);
            return result;
        }
        std::vector<float> croppedModelOutput(unpaddedSamples);
        std::copy_n(modelOutput.begin() + static_cast<std::ptrdiff_t>(contextSamples),
                    unpaddedSamples, croppedModelOutput.begin());
        auto output = resample(croppedModelOutput.data(),
                                      static_cast<int>(croppedModelOutput.size()),
                                      config->sampleRate,
                                      static_cast<int>(std::llround(sampleRate)));
        const auto wanted = static_cast<std::size_t>(targetSamples);
        if (hifiShifter)
            alignHiFiShifterTail(output, wanted, 2 * static_cast<std::size_t>(std::ceil(
                static_cast<double>(config->hop) * sampleRate / config->sampleRate)));
        const auto maximumNaturalShortfall = static_cast<std::size_t>(std::ceil(
            static_cast<double>(config->hop) * sampleRate / config->sampleRate)) + 2;
        if (output.size() + maximumNaturalShortfall < wanted)
        {
            result.error = "NSF-HiFiGAN ONNX returned short audio";
            result.buffer.setSize(0, 0);
            return result;
        }
        const auto sampleAt = [](const float* values, std::size_t count, double pos) {
            if (count==0) return 0.f;
            pos=juce::jlimit(0.0,double(count-1),pos);
            const auto i=std::size_t(pos), j=std::min(i+1,count-1);
            return values[i]+(values[j]-values[i])*float(pos-i);
        };
        for (int channel = 0; channel < channels; ++channel)
        {
            auto* destination = result.buffer.getWritePointer(channel);
            const auto copied = std::min(output.size(), wanted);
            std::copy_n(output.begin(), copied, destination);
            if (copied < wanted)
                std::fill(destination + static_cast<std::ptrdiff_t>(copied),
                          destination + static_cast<std::ptrdiff_t>(wanted), 0.0f);
            // DC/decoder impulse repair belongs to synthesized harmonics.
            // Applying it after the mix would also filter natural frication.
            if (protectNativeNoise)
                conditionNeuralBoundary(destination,targetSamples,sampleRate,0,0);
            if (protectNativeNoise)
                for (int sample=0;sample<targetSamples;++sample)
                {
                    if ((sample & 1023)==0) checkNsfCancellation();
                    const auto sourceSeconds=sourceAt(double(sample)/sampleRate);
                    if (!sourceNoise.empty())
                        destination[sample]+=sampleAt(sourceNoise.data(),sourceNoise.size(),sourceSeconds*config->sampleRate);
                    // For confidently unvoiced material, retain the original
                    // consonant itself (not its neural reconstruction), with
                    // a 10 ms mask transition into the pitched harmonic path.
                    const auto dry=voicing.at(sourceSeconds);
                    if(dry>0)
                        destination[sample] = destination[sample]*(1-dry)
                            + sampleAt(source.getReadPointer(channel),std::size_t(sourceSamples),sourceSeconds*sampleRate)*dry;
                }
            if (protectNativeNoise)
                applyNeuralEdgeGuard(destination,targetSamples,sampleRate,
                    edgeGuard.startSeconds,edgeGuard.endSeconds);
            else
                conditionNeuralBoundary(destination, targetSamples, sampleRate,
                    edgeGuard.startSeconds, edgeGuard.endSeconds);
        }
        checkNsfCancellation();
        result.usedModel = true;
        return result;
    }
    catch (const Ort::Exception& exception)
    {
        result.error = "NSF-HiFiGAN ONNX inference failed: " + juce::String(exception.what());
    }
    catch (const std::exception& exception)
    {
        result.error = "NSF-HiFiGAN render failed: " + juce::String(exception.what());
    }
#else
    juce::ignoreUnused(source, sampleRate, targetSamples, framePeriodMs, targetMidi,
                       formantSemitones, timeMap, configuredModelDirectory, execution,
                       stretchOrder, normalizeVolume, edgeGuard, cancelled, protectNativeNoise, smoothPitchTransitions,
                       incrementalNativeRender, sourceMidi);
    result.error = "NSF-HiFiGAN ONNX runtime is not included";
#endif
    if (!result.usedModel) result.buffer.setSize(0, 0);
    return result;
}

// ---------------------------------------------------------------------------
// UTAU voicebank synthesis on the one NSF-HiFiGAN renderer.
//
// These are the native building blocks that let the single NSF-HiFiGAN
// renderer reproduce the UTAU backend's non-flag behaviour -- OTO timing,
// consonant velocity, preutterance/overlap, vibrato/pitch as the model's F0
// input, and overlap crossfade mixing -- without a separate renderer, the
// classic time-domain resampler, or an external executable.  The timing/F0/mix
// maths are pure so they can be checked without the ONNX model present.

NsfUtauNotePlan buildNsfUtauNotePlan(const NsfUtauSampleTiming& timing,
                                     double noteStartSeconds,
                                     double noteDurationSeconds,
                                     double consonantVelocityScale,
                                     double tailSeconds,
                                     const std::array<double, 3>* manualFractions,
                                     bool firstTwoRegionsOnly, double soundingOutputSeconds)
{
    NsfUtauNotePlan plan;
    const auto regionStart = std::max(0.0, timing.offsetSeconds);
    auto regionEnd = std::max(regionStart + 0.001,
        timing.fileSeconds > 0.0 ? std::min(timing.endSeconds, timing.fileSeconds)
                                 : timing.endSeconds);
    if (regionEnd <= regionStart) return plan;

    // The preutterance cannot reach further back than the note's own start, or
    // the lead-in would begin before time zero.
    const auto preutterance = std::min(std::max(0.0, timing.preutteranceSeconds),
                                       std::max(0.0, noteStartSeconds));
    // The consonant is measured from the region start; velocity stretches only
    // this lead segment, reaching the vowel sooner or later.
    const auto consonant = std::max(0.0, std::min(timing.consonantSeconds,
                                                  regionEnd - regionStart));
    const auto scale = consonantVelocityScale > 1.0e-6 ? consonantVelocityScale : 1.0;
    const auto consonantSourceEnd = std::min(regionEnd, regionStart + consonant);

    const auto tail = std::max(0.0, tailSeconds);
    const auto outputSeconds = soundingOutputSeconds > 0.0
        ? std::max(0.001, soundingOutputSeconds)
        : std::max(0.01, preutterance + noteDurationSeconds + tail);

    if (timing.hasRegions)
    {
        auto sourceRegions = firstTwoRegionsOnly
            ? UtauRenderer::firstTwoRegions(timing.regionSeconds) : timing.regionSeconds;
        const auto annotated = timing.mouClasses.length() >= 2 && timing.mouClasses.length() <= 4;
        const auto count = firstTwoRegionsOnly ? 2 : annotated ? timing.mouClasses.length() : 4;
        // Clip cumulative source boundaries, including an STP entry clipped at the file edge.
        auto sourceTotal = 0.0;
        for (int i = 0; i < count; ++i)
        {
            auto& value = sourceRegions[static_cast<std::size_t>(i)];
            if (!std::isfinite(value) || value < 0.0) value = 0.0;
            value = std::min(value, std::max(0.0, regionEnd - regionStart - sourceTotal));
            sourceTotal += value;
        }
        if (sourceTotal > 1.0e-9)
        {
            if (firstTwoRegionsOnly) regionEnd = regionStart + sourceTotal;
            else sourceRegions[static_cast<std::size_t>(count - 1)] +=
                std::max(0.0, regionEnd - regionStart - sourceTotal);
            const auto velocity = static_cast<int>(std::lround(100.0 * (1.0 - std::log2(scale))));
            const juce::String spellingClasses("CV");
            const auto classes = firstTwoRegionsOnly ? &spellingClasses
                : annotated ? &timing.mouClasses : nullptr;
            const auto split = UtauRenderer::regionSplit(sourceRegions, outputSeconds,
                velocity, preutterance, manualFractions, classes);
            if (split.valid)
            {
                plan.regionClasses = classes != nullptr ? *classes : juce::String("CVVV");
                // Equal target coordinates are intentional: a zero-length output region
                // skips its source. The Mel mapper takes the last anchor at that time.
                // Keeping both endpoints preserves the preceding region at the beat.
                plan.timeMap.push_back({ 0.0, regionStart });
                auto target = 0.0, source = regionStart;
                for (int i = 0; i < count; ++i)
                {
                    const auto index = static_cast<std::size_t>(i);
                    target += split.seconds[index];
                    source += sourceRegions[index];
                    plan.timeMap.push_back({ target, source });
                }
                plan.timeMap.back() = { outputSeconds, regionEnd };
                plan.valid = true;
                plan.sourceStartSeconds = regionStart;
                plan.sourceEndSeconds = regionEnd;
                plan.soundStartOffsetSeconds = -preutterance;
                plan.outputSeconds = outputSeconds;
                plan.usesRegions = true;
                plan.regionCount = count;
                plan.outputRegionSeconds = split.seconds;
                return plan;
            }
        }
    }

    const auto consonantOutput = consonant * scale;
    std::vector<NsfHifiganTimeMapPoint> map;
    map.push_back({ 0.0, regionStart });
    if (consonantOutput > 1.0e-4 && consonantSourceEnd > regionStart + 1.0e-6)
        map.push_back({ consonantOutput, consonantSourceEnd });
    // The vowel traverses its whole source once over the remaining target span.
    // Mel frames are interpolated through this map at the decoder's fixed hop.
    map.push_back({ outputSeconds, regionEnd });

    std::vector<NsfHifiganTimeMapPoint> cleaned;
    for (const auto& point : map)
    {
        if (!cleaned.empty()
            && point.targetSeconds <= cleaned.back().targetSeconds + 1.0e-7)
        {
            cleaned.back().sourceSeconds = std::max(cleaned.back().sourceSeconds,
                                                    point.sourceSeconds);
            continue;
        }
        cleaned.push_back(point);
    }
    if (cleaned.size() < 2) return plan;

    plan.valid = true;
    plan.sourceStartSeconds = regionStart;
    plan.sourceEndSeconds = regionEnd;
    plan.soundStartOffsetSeconds = -preutterance;
    plan.outputSeconds = outputSeconds;
    plan.timeMap = std::move(cleaned);
    return plan;
}

std::vector<float> buildNsfUtauTargetMidi(float midiNote,
                                          const std::vector<NsfUtauPitchPoint>& pitchCurve,
                                          double framePeriodMs, double outputSeconds,
                                          double soundStartOffsetSeconds,
                                          const std::function<float(double)>& timelinePitchCents)
{
    const auto framePeriod = std::max(0.1, framePeriodMs) / 1000.0;
    const auto frames = std::max(1, static_cast<int>(std::ceil(
        std::max(0.001, outputSeconds) / framePeriod)) + 1);
    std::vector<float> target(static_cast<std::size_t>(frames), midiNote);
    if (pitchCurve.empty() && !timelinePitchCents) return target;

    const auto centsAt = [&](double localTime)
    {
        if (timelinePitchCents) return timelinePitchCents(localTime);
        if (localTime <= pitchCurve.front().timeSeconds) return pitchCurve.front().cents;
        if (localTime >= pitchCurve.back().timeSeconds) return pitchCurve.back().cents;
        for (std::size_t i = 1; i < pitchCurve.size(); ++i)
        {
            if (localTime > pitchCurve[i].timeSeconds) continue;
            const auto& a = pitchCurve[i - 1];
            const auto& b = pitchCurve[i];
            const auto span = b.timeSeconds - a.timeSeconds;
            const auto amount = span > 1.0e-9
                ? static_cast<float>((localTime - a.timeSeconds) / span) : 0.0f;
            return a.cents + (b.cents - a.cents) * amount;
        }
        return pitchCurve.back().cents;
    };
    for (int frame = 0; frame < frames; ++frame)
    {
        const auto localTime = static_cast<double>(frame) * framePeriod
            + soundStartOffsetSeconds;
        target[static_cast<std::size_t>(frame)] = midiNote + centsAt(localTime) / 100.0f;
    }
    return target;
}

juce::AudioBuffer<float> mixNsfUtauNotes(const std::vector<NsfUtauMixNote>& notes,
                                         double totalSeconds, double sampleRate,
                                         int channels, const std::function<bool()>& cancelled,
                                         const std::function<void(std::size_t, const UtauMixEnvelope&)>& envelopeReady)
{
    const auto chans = juce::jlimit(1, 2, channels);
    const auto totalSamples = std::max(1, static_cast<int>(std::ceil(
        std::max(0.001, totalSeconds) * sampleRate)));
    juce::AudioBuffer<float> mix(chans, totalSamples);
    mix.clear();
    if (sampleRate <= 0.0) return mix;

    std::vector<std::size_t> order(notes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b)
    {
        return notes[a].soundStartSeconds < notes[b].soundStartSeconds;
    });

    // Plan all windows before applying gain. This lets a standard CVVC edge
    // be replaced by its actual crossing, without pre-baking a second fade.
    std::vector<UtauMixEnvelope> envelopes(notes.size());
    for (std::size_t oi=0;oi<order.size();++oi)
    {
        if(cancelled&&cancelled())return {};
        const auto& note=notes[order[oi]];
        if(note.rest||note.audio.getNumSamples()<=0)continue;
        const auto start=static_cast<int>(std::lround(note.soundStartSeconds*sampleRate));
        const auto fade=std::max(0,static_cast<int>(std::lround(note.overlapSeconds*sampleRate)));
        const auto gap=std::min(0,static_cast<int>(std::lround(note.overlapSeconds*sampleRate)));
        const auto length=gap<0?std::max(1,static_cast<int>(std::lround(.003*sampleRate))):fade;
        const auto outgoingStart=start+gap-(gap<0?length:0);
        UtauMixRamp incoming {0,std::max(1,fade),true,true,.5,fade};
        UtauMixRamp outgoing {0,std::max(1,length),false,gap==0,length>0?.5:1.0,-1,gap==0&&fade>0};
        if(note.cvvcCrossfadeStyle>=0)
        {
            incoming.equalPower=outgoing.equalPower=false;
            incoming.phaseOffset=0;outgoing.phaseOffset=length>0?0:1;
            incoming.smoothCurve=note.cvvcCrossfadeStyle==1;
            outgoing.smoothCurve=outgoing.crossfade&&note.cvvcCrossfadeStyle==1;
        }
        bool preceding=false;
        for(std::size_t previous=0;previous<oi;++previous)
        {
            auto& earlier=envelopes[order[previous]];
            if(!earlier.available())continue;
            const auto relative=outgoingStart-static_cast<int>(std::lround(earlier.startSeconds*sampleRate));
            if(relative>=earlier.samples)continue;
            preceding=true;
            auto ramp=outgoing;ramp.start=relative;earlier.ramps.push_back(ramp);
        }
        incoming.crossfade=preceding&&fade>0;
        envelopes[order[oi]]={sampleRate,start/sampleRate,note.audio.getNumSamples(),{incoming}};
    }
    for(std::size_t i=0;i<notes.size();++i)
    {
        auto& envelope=envelopes[i];if(!envelope.available())continue;
        const auto& note=notes[i];
        if(note.standardEnvelopeSeconds>0)useStandardUtauEnvelope(envelope,note.standardEnvelopeSeconds);
        const auto start=static_cast<int>(std::lround(envelope.startSeconds*sampleRate));
        for(int sample=0;sample<note.audio.getNumSamples();++sample)
        {
            if((sample&1023)==0&&cancelled&&cancelled())return {};
            const auto target=start+sample;if(target<0||target>=totalSamples)continue;
            const auto gain=envelope.gainAtSample(sample);
            for(int channel=0;channel<chans;++channel)
                mix.addSample(channel,target,note.audio.getSample(std::min(channel,note.audio.getNumChannels()-1),sample)*gain);
        }
    }
    if (envelopeReady)
        for (std::size_t i = 0; i < envelopes.size(); ++i)
            if (envelopes[i].available()) envelopeReady(i, envelopes[i]);
    auto peak = 0.0f;
    for (int channel = 0; channel < mix.getNumChannels(); ++channel)
        peak = std::max(peak, mix.getMagnitude(channel, 0, mix.getNumSamples()));
    if (peak > 0.98f) mix.applyGain(0.98f / peak);
    return mix;
}

void applyNsfUtauNativeExpression(juce::AudioBuffer<float>& audio, double sampleRate,
    double framePeriodMs, const std::vector<float>& targetMidi, const UtauNoteRenderSpec& note)
{
    const auto tension = std::isfinite(note.nativeTension) ? note.nativeTension : 0.0f;
    const auto breath = std::isfinite(note.nativeBreath) ? juce::jmax(0.0f, note.nativeBreath) : 0.0f;
    if ((std::abs(tension) <= 1.0e-4f && breath == 0.0f)
        || audio.getNumSamples() <= 0 || !std::isfinite(sampleRate) || sampleRate <= 0.0) return;
    const auto frames = std::max(targetMidi.size(), static_cast<std::size_t>(std::ceil(
        audio.getNumSamples() / sampleRate * 1000.0 / std::max(0.1, framePeriodMs))) + 1);
    // Gain is deliberately empty: note.gain and the editing amplitude envelope
    // are applied once by the existing voicebank phrase mixer afterwards.
    applyNativeExpression(audio, sampleRate, framePeriodMs, targetMidi, {},
        std::vector<float>(frames, tension), std::vector<float>(frames, breath));
}

NsfUtauSynthResult synthesizeNsfUtauNote(const juce::File& sampleFile,
                                         const NsfUtauNotePlan& plan,
                                         float midiNote,
                                         const std::vector<NsfUtauPitchPoint>& pitchCurve,
                                         const juce::File& modelDirectory,
                                         const OrtExecutionConfig& execution,
                                         const std::function<float(double)>& timelinePitchCents,
                                         const UtauNoteRenderSpec* flagsNote,
                                         const std::function<bool()>& cancelled)
{
    NsfUtauSynthResult result;
    if (cancelled && cancelled()) { result.error = "NSF-HiFiGAN render cancelled"; return result; }
    if (!plan.valid)
    {
        result.error = "invalid note plan";
        return result;
    }
    if (!NsfHifiganRenderer::modelAvailable(modelDirectory))
    {
        result.error = "NSF-HiFiGAN model pack unavailable";
        return result;
    }

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    auto reader = std::unique_ptr<juce::AudioFormatReader>(
        formats.createReaderFor(sampleFile));
    if (reader == nullptr || reader->sampleRate <= 0.0)
    {
        result.error = "could not read " + sampleFile.getFileName();
        return result;
    }
    const auto sourceRate = reader->sampleRate;
#if !defined(HACHI_HAS_ONNX_ANALYSIS) || !HACHI_HAS_ONNX_ANALYSIS
    const auto regionStart = static_cast<juce::int64>(std::llround(
        plan.sourceStartSeconds * sourceRate));
    const auto regionEnd = static_cast<juce::int64>(std::llround(
        plan.sourceEndSeconds * sourceRate));
    const auto regionSamples = static_cast<int>(std::max<juce::int64>(1,
        regionEnd - regionStart));
#endif
    const auto channels = juce::jlimit(1, 2, static_cast<int>(reader->numChannels));
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    // Analyse the complete recording. OTO remains an output/time-map boundary;
    // cutting before STFT/HNSEP discards the real consonant and tail context.
    if (reader->lengthInSamples <= 0 || reader->lengthInSamples > std::numeric_limits<int>::max()) {
        result.error = "voice recording is empty or too large"; return result;
    }
    const auto sourceSamples = static_cast<int>(reader->lengthInSamples);
    juce::AudioBuffer<float> source(channels, sourceSamples);
    source.clear();
    reader->read(&source, 0, sourceSamples, 0, true, channels > 1);
    const auto& timeMap = plan.timeMap;
#else
    juce::AudioBuffer<float> source(channels, regionSamples);
    source.clear();
    reader->read(&source, 0, regionSamples, regionStart, true, channels > 1);

    std::vector<NsfHifiganTimeMapPoint> timeMap;
    timeMap.reserve(plan.timeMap.size());
    for (const auto& point : plan.timeMap)
        timeMap.push_back({ point.targetSeconds,
                            point.sourceSeconds - plan.sourceStartSeconds });
#endif

    constexpr auto framePeriodMs = 5.0;
    const auto targetMidi = buildNsfUtauTargetMidi(midiNote, pitchCurve, framePeriodMs,
        plan.outputSeconds, plan.soundStartOffsetSeconds, timelinePitchCents);
#if !defined(HACHI_HAS_ONNX_ANALYSIS) || !HACHI_HAS_ONNX_ANALYSIS
    const std::vector<float> formant(targetMidi.size(), flagsNote != nullptr ? flagsNote->nativeFormantSemitones : 0.0f);
#endif
    const auto targetSamples = std::max(1, static_cast<int>(std::lround(
        plan.outputSeconds * sourceRate)));

    NsfHifiganRenderResult neural;
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    UtauNoteRenderSpec defaults;
    neural = renderHifisamplerNote(source, sourceRate, targetSamples, framePeriodMs,
        targetMidi, timeMap, plan, flagsNote != nullptr ? *flagsNote : defaults, modelDirectory, execution, cancelled);
#else
    neural = NsfHifiganRenderer::render(source, sourceRate, targetSamples,
        framePeriodMs, targetMidi, formant, timeMap, modelDirectory, execution,
        NsfHifiganStretchOrder::fixedHop, false, {}, cancelled);
#endif
    if (!neural.usedModel || neural.buffer.getNumSamples() != targetSamples)
    {
        result.error = neural.error.isNotEmpty() ? neural.error
                                                  : "NSF-HiFiGAN render failed";
        return result;
    }
    if (flagsNote != nullptr)
        applyNsfUtauNativeExpression(neural.buffer, sourceRate, framePeriodMs, targetMidi, *flagsNote);
    result.audio = std::move(neural.buffer);
    result.sampleRate = sourceRate;
    result.usedModel = true;
    return result;
}

UtauRenderResult renderNsfUtauPhrase(const UtauRenderRequest& request,
                                     const juce::File& modelDirectory,
                                     const OrtExecutionConfig& execution)
{
    const auto family = resolveRequestedVoicebankFamily(request, true);
    if (family.warning.isNotEmpty() || request.requestedFamily == RequestedVoicebankFamily::diffSinger)
    {
        UtauRenderResult failure;
        failure.warning = family.warning.isNotEmpty() ? family.warning
            : juce::String("DiffSinger track cannot use the UTAU NSF voicebank renderer; no backend fallback was attempted");
        return failure;
    }
    if (request.chineseCvvc && !request.fourRegion)
        return renderChineseCvvc(request, [&](const auto& expanded) { return renderNsfUtauPhrase(expanded, modelDirectory, execution); });
    UtauRenderResult result;
    result.backend = "hifisampler-native-voicebank";

    // 2^(1 - velocity/100): 100 -> 1.0, larger reaches the vowel sooner.
    const auto velocityScaleOf = [](int velocity)
    {
        return std::pow(2.0, 1.0 - juce::jlimit(0, 200, velocity) / 100.0);
    };
    auto missing = 0;
    auto synthesised = 0;
    double phraseRate = 0.0;
    std::vector<NsfUtauMixNote> mixNotes;
    mixNotes.reserve(request.notes.size());
    std::vector<std::size_t> mixOwners;
    mixOwners.reserve(request.notes.size());
    std::vector<std::size_t> silentContext;
    std::vector<std::size_t> pianoNotes;
    auto missingPiano = 0;
    std::vector<UtauRenderer::ResolvedSample> samples;
    samples.reserve(request.notes.size());
    for (const auto& note : request.notes)
    {
        if (request.cancelled && request.cancelled()) return result;
        samples.push_back(UtauRenderer::resolveVoiceSample(
            request.voicebankDirectory, note.alias, note.midiNote,
            note.consonantVelocity, request.fourRegion, request.consonantClasses,
            note.stpSeconds, note.preutteranceOverrideEnabled, note.preutteranceSeconds,
            note.overlapOverrideEnabled, note.overlapSeconds, &note.oto, note.aliasIsResolved));
    }
    for (std::size_t noteIndex = 0; noteIndex < request.notes.size(); ++noteIndex)
    {
        const auto& note = request.notes[noteIndex];
        if (request.cancelled && request.cancelled()) return result;
        if (isRestLyric(note.alias) || (request.silenceEmptyLyrics && note.alias.trim().isEmpty()))
        {
            mixNotes.push_back({ {}, note.startSeconds, 0.0, true });
            mixOwners.push_back(noteIndex);
            continue;
        }
        const auto& resolved = samples[noteIndex];
        if (note.alias.trim().isEmpty() || !resolved.found)
        {
            if (!request.pianoPreviewEnabled || (note.preserveEnvelopeTiming && note.gain <= 0))
            {
                mixNotes.push_back({ {}, note.startSeconds, 0.0, true });
                mixOwners.push_back(noteIndex);
                continue;
            }
            // Generate at the final phrase rate, sharing the classic piano voice.
            if (note.alias.trim().isNotEmpty()) ++missingPiano;
            pianoNotes.push_back(mixNotes.size());
            mixNotes.push_back({ {}, note.startSeconds, 0.004, false });
            mixOwners.push_back(noteIndex);
            continue;
        }
        if (note.preserveEnvelopeTiming && note.gain <= 0)
        {
            silentContext.push_back(mixNotes.size());
            mixNotes.push_back({ {}, note.startSeconds - std::min(std::max(0.0, resolved.preutteranceSeconds),
                std::max(0.0, note.startSeconds)), resolved.overlapSeconds, false,
                             0.0,
                             request.cvvcPhonemes ? (request.cvvcCurveCrossfade?1:0) : -1 });
            mixOwners.push_back(noteIndex);
            continue;
        }

        if (!NsfHifiganRenderer::modelAvailable(modelDirectory))
        {
            result.warning = "NSF-HiFiGAN model pack unavailable";
            return result;
        }
        NsfUtauSampleTiming timing;
        timing.offsetSeconds = resolved.offsetSeconds;
        timing.endSeconds = resolved.endSeconds;
        timing.consonantSeconds = resolved.consonantSeconds;
        timing.preutteranceSeconds = resolved.preutteranceSeconds;
        timing.overlapSeconds = resolved.overlapSeconds;
        timing.fileSeconds = resolved.fileSeconds;
        timing.hasRegions = resolved.hasRegions;
        timing.regionSeconds = resolved.regionSeconds;
        timing.mouClasses = resolved.mouClasses;
        auto soundEnd = note.startSeconds + note.durationSeconds;
        if (noteIndex + 1 < samples.size() && samples[noteIndex + 1].found)
        {
            const auto& next = request.notes[noteIndex + 1];
            const auto& nextSample = samples[noteIndex + 1];
            const auto nextStart = next.startSeconds - std::min(
                std::max(0.0, nextSample.preutteranceSeconds), std::max(0.0, next.startSeconds));
            if (UtauRenderer::crossfadesInto(nextStart, soundEnd))
                soundEnd = UtauRenderer::crossfadeEnd(nextStart, nextSample.overlapSeconds,
                    next.startSeconds + next.durationSeconds);
        }
        const auto pre = std::min(std::max(0.0, timing.preutteranceSeconds),
                                   std::max(0.0, note.startSeconds));
        const auto soundingLength = std::max(0.001, soundEnd - note.startSeconds + pre);
        const auto plan = buildNsfUtauNotePlan(timing, note.startSeconds,
            note.durationSeconds, velocityScaleOf(note.consonantVelocity),
            std::max(0.0, soundEnd - note.startSeconds - note.durationSeconds),
            note.jieSplitSet ? &note.jieSplit : nullptr,
            UtauRenderer::readsOnlyFirstTwoRegions(request.fourRegion, request.consonantClasses,
                note.durationSeconds), timing.hasRegions ? soundingLength : 0.0);
        if (!plan.valid) { ++missing; continue; }

        std::vector<NsfUtauPitchPoint> pitch;
        pitch.reserve(note.pitchCurve.size());
        for (const auto& point : note.pitchCurve)
            pitch.push_back({ point.timeSeconds, point.cents });

        // The model resynthesises the recording at the target F0 directly, so
        // the target MIDI is simply the note's pitch (the recording's own pitch
        // is what the mel carries; no source-vs-target ratio is applied here).
        auto synth = synthesizeNsfUtauNote(resolved.file, plan, note.midiNote,
                                           pitch, modelDirectory, execution, note.timelinePitchCents, &note, request.cancelled);
        if (request.cancelled && request.cancelled()) return result;
        if (!synth.usedModel || synth.audio.getNumSamples() <= 0)
        {
            if (result.warning.isEmpty() && synth.error.isNotEmpty())
                result.warning = synth.error;
            ++missing;
            continue;
        }
        // Resizing leaves the editing envelope's release at its original time.
        // Fit it to the actual OTO/neighbor overlap span before applying gain,
        // exactly as the traditional UTAU mixer does (including notePiece).
        const auto fittedEnvelope = note.preserveEnvelopeTiming ? note.amplitudeEnvelope : UtauRenderer::fitAmplitudeEnvelope(note.amplitudeEnvelope,
            -plan.soundStartOffsetSeconds, plan.soundStartOffsetSeconds + plan.outputSeconds);
        const auto baseGainAt = [&fittedEnvelope](double time) {
            const auto& points = fittedEnvelope;
            if (points.empty()) return 1.0f;
            if (time <= points.front().timeSeconds) return envelopeGainFromDb(points.front().gainDb);
            if (time >= points.back().timeSeconds) return envelopeGainFromDb(points.back().gainDb);
            for (std::size_t j = 1; j < points.size(); ++j) {
                if (time > points[j].timeSeconds) continue;
                const auto& left = points[j - 1]; const auto& right = points[j];
                const auto u = static_cast<float>(juce::jlimit(0.0, 1.0,
                    (time - left.timeSeconds) / std::max(1.0e-9, right.timeSeconds - left.timeSeconds)));
                return envelopeGainFromDb(envelopeDbBetween(left.gainDb, right.gainDb, u, left.linearToNext,left.bezier));
            }
            return 1.0f;
        };
        auto headEnd = std::min(note.durationSeconds, resolved.consonantSeconds);
        auto tailStart = headEnd;
        const auto envelopeEnd = plan.soundStartOffsetSeconds + plan.outputSeconds;
        if (plan.usesRegions) {
            headEnd = plan.soundStartOffsetSeconds + plan.outputRegionSeconds[0];
            tailStart = envelopeEnd - plan.outputRegionSeconds[static_cast<std::size_t>(plan.regionCount - 1)];
        }
        const auto envelopeAt = [&](double time) {
            return advancedEnvelopeGain(note.tailFadeMode, time, tailStart, envelopeEnd,
                plan.soundStartOffsetSeconds, headEnd, note.tailFadeSettings, baseGainAt);
        };
        const auto noteGain = juce::jlimit(0.0f, 4.0f, note.gain);
        if (request.notePhonemes && plan.usesRegions)
        {
            std::vector<UtauPhonemeSpan> spans;
            auto start = plan.soundStartOffsetSeconds;
            static const char* names[] { "onset", "glide", "nucleus", "coda" };
            for (int i = 0; i < plan.regionCount; ++i)
            {
                const auto end = start + plan.outputRegionSeconds[static_cast<std::size_t>(i)];
                spans.push_back({ note.alias, timing.mouClasses.isEmpty() ? names[i]
                    : timing.mouClasses.substring(i, i + 1), start, end });
                start = end;
            }
            request.notePhonemes(noteIndex, spans);
        }
        if (request.cancelled && request.cancelled()) return result;
        if (request.notePiece && !request.noteMixEnvelope)
            request.notePiece(noteIndex, synth.audio, synth.sampleRate, -plan.soundStartOffsetSeconds,
                [&](double time) { return envelopeAt(time) * noteGain; },
                [noteGain](double) { return noteGain; });
        for (int i = 0; i < synth.audio.getNumSamples(); ++i) {
            if ((i & 1023) == 0 && request.cancelled && request.cancelled()) return result;
            const auto gain = (note.cvvcStandardEnvelope ? envelopeGainFromDb(fittedEnvelope[1].gainDb)
                : envelopeAt(plan.soundStartOffsetSeconds + i / synth.sampleRate)) * noteGain;
            for (int c = 0; c < synth.audio.getNumChannels(); ++c)
                synth.audio.setSample(c, i, synth.audio.getSample(c, i) * gain);
        }
        phraseRate = synth.sampleRate;
        mixNotes.push_back({ std::move(synth.audio),
                             note.startSeconds + plan.soundStartOffsetSeconds,
                             resolved.overlapSeconds, false,
                             note.cvvcStandardEnvelope ? fittedEnvelope.back().timeSeconds-fittedEnvelope.front().timeSeconds : 0.0,
                             request.cvvcPhonemes ? (request.cvvcCurveCrossfade?1:0) : -1 });
        mixOwners.push_back(noteIndex);
        ++synthesised;
    }

    if (phraseRate <= 0.0) phraseRate = 44100.0;
    for (const auto index : pianoNotes)
    {
        const auto owner = mixOwners[index];
        const auto& note = request.notes[owner];
        auto& piece = mixNotes[index];
        piece.audio = UtauRenderer::pianoPreview(note, phraseRate);
        const auto envelope = note.preserveEnvelopeTiming ? note.amplitudeEnvelope
            : UtauRenderer::fitAmplitudeEnvelope(note.amplitudeEnvelope, 0.0, note.durationSeconds);
        for (int i = 0; i < piece.audio.getNumSamples(); ++i)
            piece.audio.setSample(0, i, piece.audio.getSample(0, i)
                * UtauRenderer::amplitudeGain(envelope, i / phraseRate));
        if (request.notePiece && !request.noteMixEnvelope)
            request.notePiece(owner, piece.audio, phraseRate, 0.0,
                [](double) { return 1.0f; }, [](double) { return 1.0f; });
    }
    for (const auto index : silentContext)
    {
        // A zero-valued crossfade still attenuates the selected preceding note.
        // No model inference, and no song-length silent sample allocations.
        auto& context = mixNotes[index];
        context.audio.setSize(1, std::max(1, static_cast<int>(std::ceil(std::max(0.0, context.overlapSeconds) * phraseRate))));
        context.audio.clear();
    }
    result.sampleRate = phraseRate;
    const auto publishEnvelope = [&](std::size_t index, const UtauMixEnvelope& absolute)
    {
        if (!request.noteMixEnvelope) return;
        const auto owner = mixOwners[index];
        const auto& piece = mixNotes[index];
        auto local = absolute;
        local.startSeconds -= request.notes[owner].startSeconds;
        request.noteMixEnvelope(owner, local);
        // CVVC previews use the same automatic gain as the final mix. Manual
        // amplitude is already in piece.audio; never multiply it a second time.
        if (request.notePiece && !(request.notes[owner].preserveEnvelopeTiming && request.notes[owner].gain <= 0))
            request.notePiece(owner, piece.audio, phraseRate, -local.startSeconds,
                [&](double time) { return local.gainAt(time); },
                [&](double time) { return local.gainAt(time); });
    };
    result.buffer = mixNsfUtauNotes(mixNotes,
        std::max(0.001, request.targetDurationSeconds), phraseRate, 2, request.cancelled,
        request.noteMixEnvelope ? std::function<void(std::size_t,const UtauMixEnvelope&)>(publishEnvelope) : nullptr);
    if (missing > 0)
        result.warning = (result.warning.isEmpty() ? juce::String()
                                                     : result.warning + "; ")
            + juce::String(missing) + " note(s) not synthesised";
    if (!pianoNotes.empty()) result.backend += "+piano-preview";
    if (missingPiano > 0)
        result.warning += (result.warning.isEmpty() ? "" : "; ")
            + juce::String(missingPiano) + " alias(es) were not found, played as piano";
    juce::ignoreUnused(synthesised);
    return result;
}
#include "HifisamplerSmoke.inc"
#include "HiFiShifterMelSmoke.inc"
#include "WcsNonLoopSmoke.inc"
#include "NativeNoiseSmoke.inc"
#include "NsfPitchTransitionsSmoke.inc"

}
