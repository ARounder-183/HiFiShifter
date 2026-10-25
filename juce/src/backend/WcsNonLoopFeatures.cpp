#include "WcsNonLoopFeatures.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>

extern "C"
{
#include <llsm.h>
}

namespace hachi::backend
{
namespace
{
using Container = std::unique_ptr<llsm_container, decltype(&llsm_delete_container)>;
using Chunk = std::unique_ptr<llsm_chunk, decltype(&llsm_delete_chunk)>;
using Options = std::unique_ptr<llsm_aoptions, decltype(&llsm_delete_aoptions)>;
constexpr std::size_t maxCacheBytes = 64 * 1024 * 1024;
constexpr int maxAnalysisFrames = 24000;

struct SourceFeatures
{
    Container configuration { nullptr, llsm_delete_container };
    std::vector<Container> frames;
    std::vector<float> sourceF0;
    float melOffset = 0;
    std::size_t bytes = 0;
};
struct Cache
{
    std::mutex mutex;
    std::list<std::pair<juce::String, std::shared_ptr<const SourceFeatures>>> entries;
    std::size_t bytes = 0;
};
Cache& featureCache() { static Cache cache; return cache; }

float pitchHzAt(const std::vector<float>& midi, double position)
{
    if (midi.empty()) return 0;
    position = juce::jlimit(0.0, double(midi.size() - 1), position);
    const auto left = std::size_t(position), right = std::min(left + 1, midi.size() - 1);
    const auto phase = float(position - left);
    // Interpolation across zero F0 must not create a spurious pitched source.
    const auto pitch = midi[left] > 0 && midi[right] > 0
        ? midi[left] + (midi[right] - midi[left]) * phase
        : midi[phase < .5f ? left : right];
    return std::isfinite(pitch) && pitch > 0 ? 440.f * std::exp2((pitch - 69.f) / 12.f) : 0.f;
}

float* array(llsm_container* frame, int key)
{
    return static_cast<float*>(llsm_container_get(frame, key));
}
void interpolateArray(llsm_container* a, llsm_container* b, int key, float phase)
{
    auto* x = array(a, key); auto* y = array(b, key);
    if (!x || !y) return;
    for (int i = 0, n = std::min(llsm_fparray_length(x), llsm_fparray_length(y)); i < n; ++i)
        x[i] += (y[i] - x[i]) * phase;
}

// Layer-1 VT and noise PSD are log-domain parameters. Interpolating these
// before reconstructing the harmonics differs from interpolating log-Mel bins.
Container mappedFrame(const SourceFeatures& source, double position, float targetF0)
{
    position = juce::jlimit(0.0, double(source.frames.size() - 1), position);
    const auto left = std::size_t(position), right = std::min(left + 1, source.frames.size() - 1);
    const auto phase = float(position - left);
    auto* a = source.frames[left].get(); auto* b = source.frames[right].get();
    const auto fa = *array(a, LLSM_FRAME_F0), fb = *array(b, LLSM_FRAME_F0);
    const auto sameVoicing = (fa > 0) == (fb > 0);
    Container frame(llsm_copy_container(!sameVoicing && phase >= .5f ? b : a), llsm_delete_container);
    if (!frame) throw std::runtime_error("WCSNDM parameter copy failed");
    auto* f0 = array(frame.get(), LLSM_FRAME_F0);
    if (sameVoicing)
    {
        *f0 = fa + (fb - fa) * phase;
        auto* rd = array(frame.get(), LLSM_FRAME_RD);
        *rd += (*array(b, LLSM_FRAME_RD) - *rd) * phase;
        interpolateArray(frame.get(), b, LLSM_FRAME_VTMAGN, phase);
        interpolateArray(frame.get(), b, LLSM_FRAME_PSDRES, phase);
        auto* nm = static_cast<llsm_nmframe*>(llsm_container_get(frame.get(), LLSM_FRAME_NM));
        auto* next = static_cast<llsm_nmframe*>(llsm_container_get(b, LLSM_FRAME_NM));
        for (int i = 0; i < std::min(nm->npsd, next->npsd); ++i)
            nm->psd[i] += (next->psd[i] - nm->psd[i]) * phase;
    }
    const auto oldF0 = *f0;
    *f0 = oldF0 > 0 && targetF0 > 0 ? targetF0 : 0.f;
    if (*f0 > 0)
    {
        auto* vt = array(frame.get(), LLSM_FRAME_VTMAGN);
        const auto compensation = 20.f * std::log10(*f0 / oldF0);
        for (int i = 0; i < llsm_fparray_length(vt); ++i) vt[i] -= compensation;
        // Power spectra do not need source phase. Resize to the new harmonic
        // count so a downward pitch change can reconstruct upper harmonics.
        const auto nyquist = *array(source.configuration.get(), LLSM_CONF_FNYQ);
        const auto harmonics = std::min(2048, std::max(1, int(nyquist / *f0)));
        llsm_container_attach(frame.get(), LLSM_FRAME_VSPHSE, llsm_create_fparray(harmonics),
            llsm_delete_fparray, llsm_copy_fparray);
        llsm_frame_tolayer0(frame.get(), source.configuration.get());
    }
    return frame;
}

std::vector<float> physicalMel(llsm_container* frame, const WcsFeatureConfig& config)
{
    const auto bins = config.fftSize / 2 + 1;
    const auto f0 = *array(frame, LLSM_FRAME_F0);
    auto* hm = static_cast<llsm_hmframe*>(llsm_container_get(frame, LLSM_FRAME_HM));
    auto* nm = static_cast<llsm_nmframe*>(llsm_container_get(frame, LLSM_FRAME_NM));
    auto* residual = array(frame, LLSM_FRAME_PSDRES);
    const auto residualCount = residual ? llsm_fparray_length(residual) : 0;
    std::vector<double> power(std::size_t(bins), 0.0);
    const auto df = double(config.sampleRate) / config.fftSize;
    if (nm && nm->psd)
        for (int i = 0; i < bins; ++i)
        {
            const auto position = double(i) * (nm->npsd - 1) / (bins - 1);
            const auto left = int(position), right = std::min(left + 1, nm->npsd - 1);
            const auto dbAt = [&](int j) {
                return nm->psd[j] + (j < residualCount ? residual[j] - float(.375 / 2.3025851 * 10.) : 0.f);
            };
            const auto db = dbAt(left) + (dbAt(right) - dbAt(left)) * (position - left);
            // WCSNDM HF export: PSD + residual bias correction; power = PSD*df.
            power[std::size_t(i)] = std::pow(10., juce::jlimit(-160., 80., db) / 10.) * df;
        }
    if (f0 > 0 && hm)
    {
        // Hann STFT peak gain, with WCSNDM's adaptive three-bin spreading.
        const auto gain = double(config.windowSize - 1) / 4.;
        const auto side = .25 * juce::jlimit(0., 1., (f0 / df - 4.) / 2.);
        const double spread[3] { side, 1. - 2. * side, side };
        for (int h = 0; h < hm->nhar; ++h)
        {
            const auto bin = int(std::llround(f0 * (h + 1) / df));
            if (bin >= bins) break;
            for (int j = 0; j < 3; ++j)
            {
                const auto b = bin + j - 1;
                if (b >= 0 && b < bins)
                {
                    const auto magnitude = hm->ampl[h] * gain * spread[j];
                    power[std::size_t(b)] += magnitude * magnitude;
                }
            }
        }
    }
    std::vector<float> magnitude(std::size_t(bins), 0.f), mel(std::size_t(config.melBands), 0.f);
    for (int i = 0; i < bins; ++i) magnitude[std::size_t(i)] = float(std::sqrt(std::max(1.e-20, power[std::size_t(i)])));
    for (int band = 0; band < config.melBands; ++band)
    {
        double sum = 0;
        for (int i = 0; i < bins; ++i) sum += magnitude[std::size_t(i)] * config.melWeights[std::size_t(band * bins + i)];
        mel[std::size_t(band)] = float(std::log(std::max(1.e-9, sum)));
    }
    return mel;
}

std::shared_ptr<const SourceFeatures> analyseSource(
    const std::vector<float>& waveform, const WcsFeatureConfig& config,
    const std::vector<float>& f0, const std::vector<float>& referenceMel,
    std::size_t referenceFrames, const std::function<void()>& check)
{
    auto source = std::make_shared<SourceFeatures>();
    source->sourceF0 = f0;
    source->frames.reserve(f0.size());
    Options options(llsm_create_aoptions(), llsm_delete_aoptions);
    options->thop = float(double(config.hop) / config.sampleRate);
    options->f0_refine = 0;
    options->hm_method = LLSM_AOPTION_HMCZT;
    options->maxnhar = 2048;
    options->npsd = config.fftSize / 2 + 1;
    // Analyse bounded blocks with real source context. Cancellation can take
    // effect between blocks rather than waiting for an entire long recording.
    constexpr int coreFrames = 256, contextFrames = 16;
    for (int first = 0; first < int(f0.size()); first += coreFrames)
    {
        check();
        const auto last = std::min(int(f0.size()), first + coreFrames);
        const auto begin = std::max(0, first - contextFrames);
        const auto end = std::min(int(f0.size()), last + contextFrames);
        const auto sampleBegin = std::min(int(waveform.size()) - 1, begin * config.hop);
        const auto sampleEnd = std::min(int(waveform.size()), end * config.hop);
        Chunk chunk(llsm_analyze(options.get(), const_cast<float*>(waveform.data()) + sampleBegin,
            sampleEnd - sampleBegin, float(config.sampleRate), const_cast<float*>(f0.data()) + begin,
            end - begin, nullptr), llsm_delete_chunk);
        check();
        if (!chunk) throw std::runtime_error("WCSNDM source parameter analysis failed");
        llsm_chunk_tolayer1(chunk.get(), config.fftSize);
        check();
        if (!source->configuration) source->configuration.reset(llsm_copy_container(chunk->conf));
        for (int i = first; i < last; ++i)
        {
            if (!llsm_frame_checklayer1(chunk->frames[i - begin])) throw std::runtime_error("WCSNDM invalid source parameters");
            source->frames.emplace_back(llsm_copy_container(chunk->frames[i - begin]), llsm_delete_container);
            if (!source->frames.back()) throw std::runtime_error("WCSNDM source parameter copy failed");
        }
    }
    // Match the ORIGINAL source Mel level once, instead of aligning each
    // stretched output to a fixed loudness. Silence and expression stay intact.
    double physicalSum = 0, referenceSum = 0, allPhysical = 0, allReference = 0;
    std::size_t count = 0, allCount = 0;
    for (std::size_t i = 0; i < std::min(referenceFrames, source->frames.size()); ++i)
    {
        check();
        auto frame = mappedFrame(*source, double(i), source->sourceF0[i]);
        const auto mel = physicalMel(frame.get(), config);
        for (int band = 0; band < config.melBands; ++band)
        {
            const auto reference = referenceMel[std::size_t(band) * referenceFrames + i];
            allPhysical += mel[std::size_t(band)]; allReference += reference; ++allCount;
            if (reference < -14.f) continue;
            physicalSum += mel[std::size_t(band)]; referenceSum += reference; ++count;
        }
    }
    source->melOffset = count ? float((referenceSum - physicalSum) / count)
        : allCount ? float((allReference - allPhysical) / allCount) : 0.f;
    // Conservative upper estimate includes HM, phase, VT, PSD, residual,
    // channel envelopes and container overhead (not just the public vectors).
    for (const auto& frame : source->frames)
    {
        auto* nm = static_cast<llsm_nmframe*>(llsm_container_get(frame.get(), LLSM_FRAME_NM));
        auto* hm = static_cast<llsm_hmframe*>(llsm_container_get(frame.get(), LLSM_FRAME_HM));
        source->bytes += 4096 + std::size_t(nm->npsd) * 2 * sizeof(float);
        if (hm) source->bytes += std::size_t(hm->nhar) * 2 * sizeof(float);
        for (const auto field : { LLSM_FRAME_VTMAGN, LLSM_FRAME_VSPHSE })
            if (auto* values = array(frame.get(), field))
                source->bytes += std::size_t(llsm_fparray_length(values)) * sizeof(float);
        for (int c = 0; c < nm->nchannel; ++c)
            if (nm->eenv[c]) source->bytes += std::size_t(nm->eenv[c]->nhar) * 2 * sizeof(float);
    }
    return source;
}
}

double wcsNonLoopSourceTime(const std::vector<NsfHifiganTimeMapPoint>& map,
                           double time, double sourceDuration, double targetDuration)
{
    if (map.size() < 2) return juce::jlimit(0., sourceDuration, time * sourceDuration / targetDuration);
    const auto right = std::upper_bound(map.begin(), map.end(), time,
        [](double t, const auto& p) { return t < p.targetSeconds; });
    if (right == map.begin()) return right->sourceSeconds;
    if (right == map.end()) return map.back().sourceSeconds;
    const auto& left = *(right - 1);
    const auto amount = juce::jlimit(0., 1., (time - left.targetSeconds) / (right->targetSeconds - left.targetSeconds));
    return left.sourceSeconds + (right->sourceSeconds - left.sourceSeconds) * amount;
}

double wcsNonLoopTargetTime(const std::vector<NsfHifiganTimeMapPoint>& map,
                           double time, double sourceDuration, double targetDuration)
{
    if (map.size() < 2) return juce::jlimit(0., targetDuration, time * targetDuration / sourceDuration);
    const auto right = std::upper_bound(map.begin(), map.end(), time,
        [](double t, const auto& p) { return t < p.sourceSeconds; });
    if (right == map.begin()) return right->targetSeconds;
    if (right == map.end()) return map.back().targetSeconds;
    const auto& left = *(right - 1);
    const auto amount = juce::jlimit(0., 1., (time - left.sourceSeconds) / (right->sourceSeconds - left.sourceSeconds));
    return left.targetSeconds + (right->targetSeconds - left.targetSeconds) * amount;
}

WcsNonLoopMel buildWcsNonLoopMel(
    const std::vector<float>& waveform, const WcsFeatureConfig& config,
    double targetDuration, double curveFramePeriodMs,
    const std::vector<float>& sourceMidi, std::vector<float>& targetF0,
    const std::vector<NsfHifiganTimeMapPoint>& timeMap,
    const std::vector<float>& referenceMel, std::size_t referenceFrames,
    const std::function<void()>& check)
{
    check();
    if (waveform.size() < 8 || sourceMidi.empty() || targetF0.empty()
        || !std::isfinite(targetDuration) || targetDuration <= 0
        || !std::isfinite(curveFramePeriodMs) || curveFramePeriodMs <= 0 || config.sampleRate <= 0
        || config.hop <= 0 || config.fftSize < 64 || config.melBands < 1
        || config.melWeights.size() != std::size_t(config.melBands * (config.fftSize / 2 + 1))
        || referenceMel.size() != referenceFrames * std::size_t(config.melBands))
        throw std::runtime_error("WCSNDM non-loop stretch received invalid source features");
    const auto sourceDuration = double(waveform.size()) / config.sampleRate;
    double previousTarget = -1, previousSource = -1;
    for (const auto& point : timeMap)
    {
        if (!std::isfinite(point.targetSeconds) || !std::isfinite(point.sourceSeconds)
            || point.targetSeconds < 0 || point.sourceSeconds < 0
            || point.targetSeconds < previousTarget || point.sourceSeconds < previousSource
            || point.targetSeconds > targetDuration + .0001 || point.sourceSeconds > sourceDuration + .0001)
            throw std::runtime_error("WCSNDM non-loop stretch requires a forward source time map");
        previousTarget = point.targetSeconds; previousSource = point.sourceSeconds;
    }
    const auto hop = double(config.hop) / config.sampleRate;
    const auto frames = std::size_t(std::ceil(sourceDuration / hop)) + 1;
    if (frames > maxAnalysisFrames || targetF0.size() > maxAnalysisFrames)
        throw std::runtime_error("WCSNDM non-loop stretch exceeds parameter memory limit");
    std::vector<float> f0(frames);
    for (std::size_t i = 0; i < frames; ++i)
    {
        const auto time = wcsNonLoopTargetTime(timeMap, std::min(sourceDuration, i * hop), sourceDuration, targetDuration);
        f0[i] = pitchHzAt(sourceMidi, time * 1000. / curveFramePeriodMs);
        if (f0[i] > 0 && (f0[i] < 40 || f0[i] > std::min(8000., config.sampleRate * .45)))
            throw std::runtime_error("WCSNDM source pitch is outside the analysis range");
    }
    juce::MemoryOutputStream keyBytes;
    keyBytes.writeInt(config.sampleRate); keyBytes.writeInt(config.hop); keyBytes.writeInt(config.fftSize);
    keyBytes.writeInt(config.windowSize); keyBytes.writeInt(config.melBands);
    keyBytes.write(waveform.data(), waveform.size() * sizeof(float));
    keyBytes.write(f0.data(), f0.size() * sizeof(float));
    keyBytes.write(config.melWeights.data(), config.melWeights.size() * sizeof(float));
    const auto key = juce::SHA256(keyBytes.getData(), keyBytes.getDataSize()).toHexString();
    WcsNonLoopMel result;
    std::shared_ptr<const SourceFeatures> source;
    auto& cache = featureCache();
    {
        const std::scoped_lock lock(cache.mutex);
        for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it)
            if (it->first == key)
            {
                source = it->second; cache.entries.splice(cache.entries.begin(), cache.entries, it);
                result.reusedSourceAnalysis = true; break;
            }
    }
    if (!source)
    {
        source = analyseSource(waveform, config, f0, referenceMel, referenceFrames, check);
        check();
        const std::scoped_lock lock(cache.mutex);
        if (source->bytes <= maxCacheBytes)
        {
            while (!cache.entries.empty() && (cache.bytes + source->bytes > maxCacheBytes || cache.entries.size() >= 4))
            {
                cache.bytes -= cache.entries.back().second->bytes; cache.entries.pop_back();
            }
            cache.entries.emplace_front(key, source); cache.bytes += source->bytes;
        }
    }
    result.values.resize(targetF0.size() * std::size_t(config.melBands));
    for (std::size_t i = 0; i < targetF0.size(); ++i)
    {
        check();
        const auto time = wcsNonLoopSourceTime(timeMap, i * hop, sourceDuration, targetDuration);
        auto frame = mappedFrame(*source, time / hop, targetF0[i]);
        targetF0[i] = *array(frame.get(), LLSM_FRAME_F0);
        const auto mel = physicalMel(frame.get(), config);
        for (int band = 0; band < config.melBands; ++band)
            result.values[std::size_t(band) * targetF0.size() + i] = juce::jlimit(-20.723267f, 5.f, mel[std::size_t(band)] + source->melOffset);
    }
    return result;
}

void clearWcsNonLoopFeatureCache()
{
    auto& cache = featureCache();
    const std::scoped_lock lock(cache.mutex);
    cache.entries.clear(); cache.bytes = 0;
}
}
