// Adapted from openhachimi/hifisampler, Apache-2.0.
// Included inside NsfHifiganRenderer.cpp's ONNX anonymous namespace.
// See third_party/hifisampler/NOTICE.txt for the pinned source and adaptations.

struct HifiSpectrum
{
    int fftSize = 0, hop = 0;
    std::size_t frames = 0;
    std::vector<std::complex<float>> values; // frequency-major
};

HifiSpectrum hifiStft(const std::vector<float>& wave, int size, int hop, bool reflect = false)
{
    if (wave.empty()) return {};
    auto order = 0;
    for (auto n = size; n > 1; n >>= 1) ++order;
    if ((1 << order) != size) throw std::runtime_error("HN-SEP FFT size must be a power of two");
    juce::dsp::FFT fft(order);
    HifiSpectrum spec { size, hop, 1 + wave.size() / static_cast<std::size_t>(hop), {} };
    const auto bins = size / 2 + 1;
    spec.values.resize(static_cast<std::size_t>(bins) * spec.frames);
    std::vector<std::complex<float>> input(static_cast<std::size_t>(size)), output(input.size());
    for (std::size_t frame = 0; frame < spec.frames; ++frame)
    {
        checkNsfCancellation();
        for (int j = 0; j < size; ++j) {
            const auto pos = static_cast<std::ptrdiff_t>(frame * hop) + j - size / 2;
            const auto sample = pos >= 0 && pos < static_cast<std::ptrdiff_t>(wave.size())
                ? wave[static_cast<std::size_t>(pos)]
                : reflect ? wave[reflectIndex(pos, wave.size())] : 0.0f;
            // torch.hann_window is periodic (the generic editor Mel used a
            // symmetric window). Hifisampler uses the periodic variant.
            input[static_cast<std::size_t>(j)] = { sample * (0.5f - 0.5f * std::cos(
                juce::MathConstants<float>::twoPi * j / size)), 0.0f };
        }
        fft.perform(input.data(), output.data(), false);
        for (int bin = 0; bin < bins; ++bin)
            spec.values[static_cast<std::size_t>(bin) * spec.frames + frame] = output[static_cast<std::size_t>(bin)];
    }
    return spec;
}

std::vector<float> hifiIstft(const HifiSpectrum& spec, std::size_t length)
{
    if (spec.frames == 0) return {};
    auto order = 0; for (auto n = spec.fftSize; n > 1; n >>= 1) ++order;
    juce::dsp::FFT fft(order);
    const auto size = spec.fftSize;
    const auto fullSize = (spec.frames - 1) * static_cast<std::size_t>(spec.hop) + size;
    std::vector<float> wave(fullSize, 0), weights(fullSize, 0);
    std::vector<std::complex<float>> input(static_cast<std::size_t>(size)), output(input.size());
    for (std::size_t frame = 0; frame < spec.frames; ++frame) {
        checkNsfCancellation();
        for (int bin = 0; bin <= size / 2; ++bin)
            input[static_cast<std::size_t>(bin)] = spec.values[static_cast<std::size_t>(bin) * spec.frames + frame];
        for (int bin = size / 2 + 1; bin < size; ++bin)
            input[static_cast<std::size_t>(bin)] = std::conj(input[static_cast<std::size_t>(size-bin)]);
        fft.perform(input.data(), output.data(), true);
        for (int j = 0; j < size; ++j) {
            const auto window = 0.5f - 0.5f * std::cos(juce::MathConstants<float>::twoPi * j / size);
            const auto at = frame * spec.hop + static_cast<std::size_t>(j);
            wave[at] += output[static_cast<std::size_t>(j)].real() * window;
            weights[at] += window * window;
        }
    }
    std::vector<float> result(length, 0);
    for (std::size_t i = 0; i < length && i + size/2 < wave.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
        const auto at = i + size/2;
        result[i] = weights[at] > 1.0e-10f ? wave[at] / weights[at] : 0.0f;
    }
    return result;
}

#include "HnsepDiskCache.inc"

struct HifiSeparation { std::vector<float> harmonic; bool cacheHit = false; };

juce::File hifiHnsepFile(const ModelFiles& files)
{
    const auto overridePath = juce::SystemStats::getEnvironmentVariable("HACHISHIFTER_HNSEP_ONNX", {});
    if (overridePath.isNotEmpty()) return juce::File(overridePath);
    auto path = files.onnx.getSiblingFile("hnsep").getChildFile("model.onnx");
    if (path.existsAsFile()) return path;
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory()
        .getChildFile("models/nsf_hifigan/hnsep/model.onnx");
}

HifiSeparation hifiSeparate(const std::vector<float>& wave, const ModelFiles& files,
    const OrtExecutionConfig& execution, bool force)
{
    const auto path = hifiHnsepFile(files);
    if (!path.existsAsFile()) throw std::runtime_error("HiFisampler Hb/Hv/Ht requires HN-SEP: " + path.getFullPathName().toStdString());
    juce::MemoryBlock modelBytes, configBytes;
    if (!path.loadFileAsData(modelBytes) || !path.getSiblingFile("config.json").loadFileAsData(configBytes))
        throw std::runtime_error("Could not snapshot HN-SEP model/configuration");
    const auto config = juce::JSON::parse(juce::String::createStringFromData(configBytes.getData(),
        static_cast<int>(configBytes.getSize())));
    if (!config.isObject()) throw std::runtime_error("Invalid HN-SEP configuration JSON");
    const auto size = config.hasProperty("n_fft") ? static_cast<int>(config["n_fft"]) : 2048;
    const auto hop = config.hasProperty("hop_length") ? static_cast<int>(config["hop_length"]) : 512;
    const auto rate = config.hasProperty("sr") ? static_cast<int>(config["sr"]) : 44100;
    if (rate != 44100 || size < 128 || size > 8192 || hop <= 0 || hop > size / 2)
        throw std::runtime_error("Invalid HN-SEP STFT configuration");
    if (wave.empty() || !std::all_of(wave.begin(), wave.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Invalid HN-SEP input PCM");
    const auto key = hnCache::identity(wave, modelBytes, configBytes, rate, size, hop);
    static std::mutex cacheMutex;
    static std::map<juce::String, std::vector<float>> cache;
    hnCache::Lock diskLock(key);
    {
        std::scoped_lock lock(cacheMutex);
        if (!force) if (const auto found = cache.find(key); found != cache.end()) return { found->second, true };
    }
    if (!force && diskLock.held) {
        auto cached = hnCache::load(key, wave.size());
        if (!cached.empty()) return { std::move(cached), true };
    }
    // Same length rounding and zero padding as HnsepModel.predict_fromaudio.
    const auto segment = static_cast<std::size_t>(32 * hop);
    const auto t1 = wave.size() + hop;
    const auto pad = segment * ((t1 - 1) / segment + 1) - t1;
    const auto left = (pad / 2 / hop) * hop;
    std::vector<float> padded(wave.size() + pad, 0);
    std::copy(wave.begin(), wave.end(), padded.begin() + static_cast<std::ptrdiff_t>(left));
    auto spec = hifiStft(padded, size, hop);
    const auto count = spec.values.size();
    std::vector<float> tensor(count * 2);
    for (std::size_t i = 0; i < count; ++i) { tensor[i] = spec.values[i].real(); tensor[count+i] = spec.values[i].imag(); }
    const std::array<int64_t, 4> shape { 1, 2, size/2+1, static_cast<int64_t>(spec.frames) };
    auto entry = session(path, execution, &modelBytes);
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, tensor.data(), tensor.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    const auto inputName = entry.model->GetInputNameAllocated(0, allocator);
    const auto outputName = entry.model->GetOutputNameAllocated(0, allocator);
    const char* inputs[] { inputName.get() }; const char* outputs[] { outputName.get() };
    std::unique_lock<std::mutex> inferenceLock;
    if (entry.runMutex) inferenceLock = cancellableNsfLock(*entry.runMutex);
    CancellableNsfRun run;
    auto result = entry.model->Run(run.options, inputs, &input, 1, outputs, 1);
    checkNsfCancellation();
    if (!result[0].IsTensor() || result[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || result[0].GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>(shape.begin(), shape.end()))
        throw std::runtime_error("Unexpected HN-SEP mask dimensions/type");
    const auto* mask = result[0].GetTensorData<float>();
    for (std::size_t i = 0; i < count; ++i) spec.values[i] *= std::complex<float>(mask[i], mask[count+i]);
    auto restored = hifiIstft(spec, padded.size());
    std::vector<float> harmonic(restored.begin() + static_cast<std::ptrdiff_t>(left),
        restored.begin() + static_cast<std::ptrdiff_t>(left + wave.size()));
    if (!std::all_of(harmonic.begin(), harmonic.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Non-finite HN-SEP output");
    checkNsfCancellation();
    if (diskLock.held) hnCache::save(key, harmonic);
    {
        std::scoped_lock lock(cacheMutex);
        // Bound audio memory; no cache sidecars are written into user voicebanks.
        std::size_t bytes = harmonic.size()*sizeof(float);
        for (const auto& item : cache) bytes += item.second.size()*sizeof(float);
        if (bytes > 32*1024*1024 || cache.size() >= 16) cache.clear();
        if (harmonic.size()*sizeof(float) <= 32*1024*1024) cache[key] = harmonic;
    }
    return { std::move(harmonic), false };
}

float hifiPeak(const std::vector<float>& wave)
{
    float peak = 0; for (const auto v : wave) peak = std::max(peak, std::abs(v)); return peak;
}

std::vector<float> hifiTension(const std::vector<float>& harmonic, float tension, const Config& config)
{
    if (tension == 0 || hifiPeak(harmonic) < 1.0e-10f) return harmonic;
    auto padded = harmonic;
    padded.resize(((padded.size()+config.hop-1)/config.hop)*config.hop, 0);
    auto spec = hifiStft(padded, config.fftSize, config.hop, true);
    const auto b = -tension / 50.0f;
    const auto x0 = (config.fftSize/2+1) * 1500.0f / (config.sampleRate/2.0f);
    for (int bin = 0; bin <= config.fftSize/2; ++bin) {
        const auto gain = std::exp(juce::jlimit(-2.0f, 2.0f, b - b*bin/x0));
        for (std::size_t frame = 0; frame < spec.frames; ++frame)
            spec.values[static_cast<std::size_t>(bin)*spec.frames+frame] *= gain;
    }
    auto filtered = hifiIstft(spec, padded.size());
    const auto peak = hifiPeak(filtered);
    const auto gain = peak > 1.0e-10f ? hifiPeak(harmonic)/peak * (1.0f + juce::jlimit(0.0f, .33f, b/-15.0f)) : 0;
    filtered.resize(harmonic.size()); for (auto& v : filtered) v *= gain;
    return filtered;
}

// Bluestein keeps the exact arbitrary FFT length used by the upstream g flag;
// rounding to a power of two would introduce discrete formant jumps.
class HifiFft
{
public:
    explicit HifiFft(int length) : n(length)
    {
        int order = 0;
        if ((n & (n-1)) == 0) {
            while ((1 << order) < n) ++order;
            size = n; fft = std::make_unique<juce::dsp::FFT>(order);
            direct = true; input.resize(static_cast<std::size_t>(size)); output.resize(input.size()); return;
        }
        while ((1 << order) < 2*n-1) ++order;
        size = 1 << order; fft = std::make_unique<juce::dsp::FFT>(order);
        chirp.resize(static_cast<std::size_t>(n)); kernel.assign(static_cast<std::size_t>(size), {});
        for (int i = 0; i < n; ++i) {
            const auto angle = juce::MathConstants<double>::pi * static_cast<double>(i) * i / n;
            chirp[static_cast<std::size_t>(i)] = std::complex<float>(static_cast<float>(std::cos(angle)), static_cast<float>(-std::sin(angle)));
            kernel[static_cast<std::size_t>(i)] = std::conj(chirp[static_cast<std::size_t>(i)]);
            if (i != 0) kernel[static_cast<std::size_t>(size-i)] = kernel[static_cast<std::size_t>(i)];
        }
        auto temporary = kernel; fft->perform(temporary.data(), kernel.data(), false);
        input.resize(static_cast<std::size_t>(size)); output.resize(input.size());
    }
    void magnitude(const std::vector<float>& window, std::vector<float>& result)
    {
        if (direct) {
            for (int i = 0; i < n; ++i) input[static_cast<std::size_t>(i)] = {window[static_cast<std::size_t>(i)],0};
            fft->perform(input.data(), output.data(), false);
            result.resize(static_cast<std::size_t>(n/2+1));
            for (int i = 0; i <= n/2; ++i) result[static_cast<std::size_t>(i)] = std::abs(output[static_cast<std::size_t>(i)]);
            return;
        }
        std::fill(input.begin(), input.end(), std::complex<float>{});
        for (int i = 0; i < n; ++i) input[static_cast<std::size_t>(i)] = window[static_cast<std::size_t>(i)]*chirp[static_cast<std::size_t>(i)];
        fft->perform(input.data(), output.data(), false);
        for (int i = 0; i < size; ++i) input[static_cast<std::size_t>(i)] = output[static_cast<std::size_t>(i)]*kernel[static_cast<std::size_t>(i)];
        fft->perform(input.data(), output.data(), true);
        result.resize(static_cast<std::size_t>(n/2+1));
        for (int i = 0; i <= n/2; ++i) result[static_cast<std::size_t>(i)] = std::abs(output[static_cast<std::size_t>(i)]*chirp[static_cast<std::size_t>(i)]);
    }
private:
    int n = 0, size = 0; bool direct = false; std::unique_ptr<juce::dsp::FFT> fft;
    std::vector<std::complex<float>> chirp, kernel, input, output;
};

struct HifiFeatures { MelData mel; float scale = 1; };
HifiFeatures hifiFeatures(const std::vector<float>& source, const std::vector<float>& harmonic,
    const HifisamplerFlags& flags, const Config& config)
{
    auto wave = source;
    const auto breath = flags.get("Hb")/100, voice = flags.get("Hv")/100;
    const auto tension = flags.get("Ht");
    if (breath != voice || tension != 0) {
        if (harmonic.size() != source.size()) throw std::runtime_error("Missing HN-SEP harmonic waveform");
        auto voiced = harmonic; for (auto& v : voiced) v *= voice;
        voiced = hifiTension(voiced, tension, config);
        for (std::size_t i = 0; i < wave.size(); ++i) wave[i] = breath*(source[i]-harmonic[i])+voiced[i];
    }
    else for (auto& v : wave) v *= breath;
    HifiFeatures result;
    const auto peak = hifiPeak(wave);
    result.scale = peak >= .5f ? .5f / peak : 1.0f;
    for (auto& v : wave) v *= result.scale;
    constexpr int analysisHop = 128;
    const auto factor = std::exp2(flags.get("g") / 1200.0f);
    const auto fftSize = static_cast<int>(std::nearbyint(config.fftSize * factor));
    const auto windowSize = static_cast<int>(std::nearbyint(config.windowSize * factor));
    const auto pad = std::max(0, (windowSize-analysisHop)/2);
    const auto rightPad = std::max(0, (windowSize-analysisHop+1)/2);
    const auto paddedLength = std::max(static_cast<std::size_t>(windowSize), wave.size()+pad+rightPad);
    result.mel.frames = 1+(paddedLength-windowSize)/analysisHop;
    result.mel.values.resize(static_cast<std::size_t>(config.melBands)*result.mel.frames);
    const auto weights = filterbank(config);
    const auto bins = config.fftSize/2+1;
    HifiFft fft(fftSize);
    std::vector<float> window(static_cast<std::size_t>(fftSize), 0), magnitude;
    for (std::size_t frame = 0; frame < result.mel.frames; ++frame) {
        checkNsfCancellation();
        std::fill(window.begin(), window.end(), 0);
        for (int j = 0; j < windowSize; ++j)
            window[static_cast<std::size_t>(j+(fftSize-windowSize)/2)] = wave[reflectIndex(
                static_cast<std::ptrdiff_t>(frame*analysisHop)+j-pad, wave.size())]
                * (0.5f - 0.5f*std::cos(juce::MathConstants<float>::twoPi*j/windowSize));
        fft.magnitude(window, magnitude);
        for (int band = 0; band < config.melBands; ++band) {
            float value = 0;
            for (int bin = 0; bin < bins && bin < static_cast<int>(magnitude.size()); ++bin)
                value += weights[static_cast<std::size_t>(band*bins+bin)]*magnitude[static_cast<std::size_t>(bin)]
                    * config.windowSize / windowSize;
            result.mel.values[static_cast<std::size_t>(band)*result.mel.frames+frame] = std::log(std::max(1.0e-9f, value));
        }
    }
    return result;
}

struct HifiBiquad
{
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
    double process(double input) {
        const auto output = b0*input+z1; z1 = b1*input-a1*output+z2; z2 = b2*input-a2*output;
        return output;
    }
    static HifiBiquad highpass(double rate, double cutoff, double q) {
        const auto w = juce::MathConstants<double>::twoPi*cutoff/rate, c = std::cos(w), a = std::sin(w)/(2*q);
        return { (1+c)/(2*(1+a)), -(1+c)/(1+a), (1+c)/(2*(1+a)), -2*c/(1+a), (1-a)/(1+a) };
    }
};

void hifiGrowl(std::vector<float>& wave, double rate, const std::vector<HifisamplerFlags>& flags, double hopSeconds)
{
    if (wave.empty() || std::none_of(flags.begin(), flags.end(), [](const auto& f) { return f.get("HG") != 0; })) return;
    auto hp1 = HifiBiquad::highpass(rate, std::max(400.0, rate*.005), .541196100146197);
    auto hp2 = HifiBiquad::highpass(rate, std::max(400.0, rate*.005), 1.306562964876377);
    std::vector<double> band(wave.size()), ratios(wave.size()), modulated(wave.size());
    double sum = 0;
    for (std::size_t i = 0; i < wave.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
        band[i] = hp2.process(hp1.process(wave[i]));
        const auto frame = std::min(flags.size()-1, static_cast<std::size_t>(i/rate/hopSeconds));
        const auto lfo = std::fmod(juce::MathConstants<double>::twoPi*80*(i/rate), juce::MathConstants<double>::twoPi)
            < juce::MathConstants<double>::pi ? 1.0 : -1.0;
        ratios[i] = std::exp2(lfo*flags[frame].get("HG")/1200.0); sum += ratios[i];
    }
    const auto mean = sum / wave.size(); sum = 0;
    auto driftFilter = HifiBiquad::highpass(rate, 20, std::sqrt(.5));
    double oldPower = 0, newPower = 0;
    for (std::size_t i = 0; i < wave.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
        sum += ratios[i];
        auto drift = sum-ratios[0]-i*mean;
        if (wave.size() > 100) drift = driftFilter.process(drift);
        const auto pos = juce::jlimit(0.0, static_cast<double>(wave.size()-1), static_cast<double>(i)+drift);
        const auto left = static_cast<std::size_t>(pos), right = std::min(left+1, wave.size()-1);
        modulated[i] = band[left]+(pos-left)*(band[right]-band[left]);
        oldPower += band[i]*band[i]; newPower += modulated[i]*modulated[i];
    }
    const auto gain = newPower > 1.0e-20 ? std::sqrt(oldPower/newPower) : 1.0;
    for (std::size_t i = 0; i < wave.size(); ++i) wave[i] = wave[i]-band[i]+static_cast<float>(modulated[i]*gain);
}

// ITU-R BS.1770 K weighting and two-stage -70 LUFS / relative -10 LU gates.
double hifiLoudness(const std::vector<float>& wave, double rate)
{
    if (wave.empty()) return -std::numeric_limits<double>::infinity();
    // Same default K-weighting filters as pyloudnorm.Meter (not DeMan mode).
    const auto w = juce::MathConstants<double>::twoPi*1500/rate, c = std::cos(w);
    const auto a = std::pow(10.0, 4.0/40), alpha = std::sin(w)/(2*std::sqrt(.5));
    const auto beta = 2*std::sqrt(a)*alpha, a0 = a+1-(a-1)*c+beta;
    HifiBiquad shelf { a*(a+1+(a-1)*c+beta)/a0, -2*a*(a-1+(a+1)*c)/a0,
        a*(a+1+(a-1)*c-beta)/a0, 2*(a-1-(a+1)*c)/a0, (a+1-(a-1)*c-beta)/a0 };
    // pyloudnorm's default Meter uses the RBJ high-pass at 38 Hz, Q=.5.
    auto highpass = HifiBiquad::highpass(rate, 38.0, .5);
    std::vector<double> power(wave.size()+1, 0);
    for (std::size_t i = 0; i < wave.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
        const auto value = highpass.process(shelf.process(wave[i])); power[i+1] = power[i]+value*value;
    }
    const auto block = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(rate*.4)));
    const auto step = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(rate*.1)));
    std::vector<double> energies;
    for (std::size_t i = 0; i+block <= wave.size(); i += step) {
        if ((i & 1023) == 0) checkNsfCancellation();
        const auto e = (power[i+block]-power[i])/block;
        if (e > 0 && -.691+10*std::log10(e) >= -70) energies.push_back(e);
    }
    if (energies.empty()) return -std::numeric_limits<double>::infinity();
    double sum = 0; for (const auto e : energies) sum += e;
    const auto relative = -.691+10*std::log10(sum/energies.size())-10;
    sum = 0; std::size_t count = 0;
    for (const auto e : energies) if (-.691+10*std::log10(e) > relative) { sum += e; ++count; }
    return count > 0 ? -.691+10*std::log10(sum/count) : -std::numeric_limits<double>::infinity();
}

void hifiNormalize(std::vector<float>& wave, double rate, const std::vector<HifisamplerFlags>& flags,
    double hopSeconds, bool trimSilence)
{
    if (wave.empty() || std::none_of(flags.begin(), flags.end(), [](const auto& f) { return f.normalize && f.get("P") != 0; })) return;
    std::size_t start = 0, end = wave.size();
    if (trimSilence) {
        const auto window = std::max<std::size_t>(1, static_cast<std::size_t>(rate*.02));
        const auto hop = std::max<std::size_t>(1, static_cast<std::size_t>(rate*.01));
        bool found = false;
        for (std::size_t at = 0; at+window < wave.size(); at += hop) {
            checkNsfCancellation();
            double power = 0; for (std::size_t i = at; i < at+window; ++i) power += wave[i]*wave[i];
            if (power/window > std::pow(10.0, -52.0/10)) {
                if (!found) start = at;
                found = true; end = std::min(wave.size(), at+hop+window+static_cast<std::size_t>(rate*.1));
            }
        }
        if (!found) return;
    }
    std::vector<float> measure(wave.begin()+static_cast<std::ptrdiff_t>(start), wave.begin()+static_cast<std::ptrdiff_t>(end));
    const auto originalSize = measure.size();
    if (measure.size() < static_cast<std::size_t>(rate*.4)) {
        measure.resize(static_cast<std::size_t>(std::ceil(rate*.4)));
        for (std::size_t i = originalSize; i < measure.size(); ++i) measure[i] = measure[reflectIndex(static_cast<std::ptrdiff_t>(i), originalSize)];
    }
    const auto measured = hifiLoudness(measure, rate);
    if (!std::isfinite(measured)) return;
    for (std::size_t i = 0; i < wave.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
        const auto frame = std::min(flags.size()-1, static_cast<std::size_t>(i/rate/hopSeconds));
        wave[i] *= static_cast<float>(std::pow(10.0, (-16-measured)*(flags[frame].normalize ? flags[frame].get("P") : 0)/2000));
    }
    // Preserve the editor's OTO head/tail envelope ownership. Upstream's
    // normalization routine adds an extra 200 ms fade; that would splice two
    // tails again, so here silence trimming affects measurement only.
}

double hifiSourceTime(const std::vector<NsfHifiganTimeMapPoint>& timeMap,
    const NsfUtauNotePlan& plan, double time, int sampleRate, bool loop)
{
    if (timeMap.size() < 2) return time;
    std::size_t segment = 0;
    while (segment+1 < timeMap.size() && time >= timeMap[segment+1].targetSeconds) ++segment;
    const auto& left = timeMap[segment];
    const auto& right = timeMap[std::min(segment+1, timeMap.size()-1)];
    const auto length = right.sourceSeconds-left.sourceSeconds;
    const auto targetLength = right.targetSeconds-left.targetSeconds;
    // Interpolate source progress over the whole target span, including long notes.
    // Never replace this map with periodic frame indexing unless He is explicit.
    auto sourceTime = left.sourceSeconds+length*juce::jlimit(0.0, 1.0,
        (time-left.targetSeconds)/std::max(1.0e-9, targetLength));
    const auto regionClass = plan.regionClasses.substring(static_cast<int>(segment), static_cast<int>(segment+1));
    const auto fixedConsonant = plan.usesRegions && (regionClass == "C" || regionClass == "S");
    if (loop && !fixedConsonant && length > 1.0e-6 && targetLength > length && (plan.usesRegions || segment > 0)) {
        const auto frames = std::max<std::size_t>(1, static_cast<std::size_t>(std::round(length*sampleRate/128)));
        const auto index = (time-left.targetSeconds)*sampleRate/128;
        const auto a = static_cast<std::ptrdiff_t>(std::floor(index)), b = a+1;
        const auto u = index-a;
        sourceTime = left.sourceSeconds+((1-u)*reflectIndex(a, frames)+u*reflectIndex(b, frames))*128/sampleRate;
    }
    return sourceTime;
}

// Decoder context comes from the recording outside the OTO cut. Continue the
// adjacent stretch slope outside the output span, without looping that context.
double hifiContextSourceTime(const std::vector<NsfHifiganTimeMapPoint>& map,
    const NsfUtauNotePlan& plan, double time, int sampleRate, bool loop)
{
    if (map.size() < 2) return time;
    if (time >= map.front().targetSeconds && time < map.back().targetSeconds)
        return hifiSourceTime(map, plan, time, sampleRate, loop);
    std::size_t left = 0, right = 1;
    if (time < map.front().targetSeconds) {
        while (right+1 < map.size() && map[right].targetSeconds <= map[left].targetSeconds) ++right;
        // A skipped opening region belongs to the cut, not the audible onset.
        while (left+1 < right && map[left+1].targetSeconds == map[left].targetSeconds) ++left;
    } else {
        right = map.size()-1; left = right-1;
        while (left > 0 && map[left].targetSeconds >= map[right].targetSeconds) --left;
        while (right > left+1 && map[right-1].targetSeconds == map[right].targetSeconds) --right;
    }
    const auto span = map[right].targetSeconds-map[left].targetSeconds;
    if (span <= 1.0e-9) return map[left].sourceSeconds;
    return map[left].sourceSeconds+(time-map[left].targetSeconds)
        *(map[right].sourceSeconds-map[left].sourceSeconds)/span;
}

float hifiSampleMel(const MelData& mel, int band, double sourceTime, int sampleRate)
{
    // Analysis frames are centred at (frame + 0.5) * 128 samples.
    const auto pos = juce::jlimit(0.0, static_cast<double>(mel.frames-1), sourceTime*sampleRate/128-.5);
    const auto a = static_cast<std::size_t>(pos), b = std::min(a+1, mel.frames-1);
    const auto offset = static_cast<std::size_t>(band)*mel.frames;
    return mel.values[offset+a]+static_cast<float>(pos-a)*(mel.values[offset+b]-mel.values[offset+a]);
}

NsfHifiganRenderResult renderHifisamplerNote(const juce::AudioBuffer<float>& source, double sampleRate,
    int targetSamples, double framePeriodMs, const std::vector<float>& midi,
    const std::vector<NsfHifiganTimeMapPoint>& timeMap, const NsfUtauNotePlan& plan,
    const UtauNoteRenderSpec& note, const juce::File& directory, const OrtExecutionConfig& execution,
    const std::function<bool()>& cancelled = {})
{
    NsfHifiganRenderResult result;
    const NsfCancellationScope cancellation(cancelled);
    try {
        checkNsfCancellation();
        const auto files = modelFiles(directory);
        if (!files) throw std::runtime_error("HiFisampler PC-NSF-HiFiGAN model pack unavailable");
        auto config = readConfig(files->config, result.error); if (!config) return result;
        const auto options = juce::JSON::parse(files->onnx.getSiblingFile("hifisampler.json").loadFileAsString());
        // Voicebank notes stretch each OTO span once across its target duration.
        // Older packs ship loop_mode=true from the upstream standalone sampler;
        // that must not override the editor's OTO time map. Looping is opt-in He.
        const auto waveNorm = !options.hasProperty("wave_norm") || static_cast<bool>(options["wave_norm"]);
        const auto trim = !options.hasProperty("trim_silence") || static_cast<bool>(options["trim_silence"]);
        auto input = stableMonoInput(source);
        input = resample(input.data(), static_cast<int>(input.size()), static_cast<int>(std::llround(sampleRate)), config->sampleRate);
        // Promote before multiplying: a 44.1 kHz note already overflows int
        // at about 1.1 seconds, which would decode a tiny span then pad silence.
        const auto samples = static_cast<std::size_t>(std::max(1.0, std::round(static_cast<double>(targetSamples)*config->sampleRate/sampleRate)));
        const auto frames = (samples+config->hop-1)/config->hop;
        const auto hopSeconds = static_cast<double>(config->hop)/config->sampleRate;
        std::vector<HifisamplerFlags> flags; flags.reserve(frames);
        bool separate = false, force = false;
        for (std::size_t i = 0; i < frames; ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
            flags.push_back(hifiFlagsAt(note, plan.outputRegionSeconds, plan.regionCount,
                i*hopSeconds, plan.soundStartOffsetSeconds));
            const auto& f = flags.back(); separate |= f.get("Hb") != f.get("Hv") || f.get("Ht") != 0; force |= f.force;
        }
        auto harmonic = separate ? hifiSeparate(input, *files, execution, force).harmonic : std::vector<float>{};
        std::map<juce::String, HifiFeatures> features;
        static std::mutex featuresMutex;
        static std::map<juce::String, HifiFeatures> featuresCache;
        const auto featurePrefix = juce::SHA256(input.data(), input.size()*sizeof(float)).toHexString()
            + (harmonic.empty() ? juce::String{} : juce::SHA256(harmonic.data(), harmonic.size()*sizeof(float)).toHexString())
            + juce::SHA256(files->config).toHexString();
        constexpr std::size_t context = 16;
        const auto modelFrames = frames+2*context;
        MelData mel; mel.frames = modelFrames; mel.values.resize(static_cast<std::size_t>(config->melBands)*modelFrames);
        std::vector<float> f0(modelFrames), pitch(frames), scales(frames, 1);
        for (std::size_t modelFrame = 0; modelFrame < modelFrames; ++modelFrame) {
        checkNsfCancellation();
            const auto localFrame = static_cast<std::ptrdiff_t>(modelFrame)-static_cast<std::ptrdiff_t>(context);
            const auto i = static_cast<std::size_t>(juce::jlimit<std::ptrdiff_t>(0, frames-1, localFrame));
            auto f = flags[i];
            juce::String key;
            for (const auto* name : { "Hb", "Hv", "Ht", "g" }) {
                // Upstream text flags are integer-valued. Curves are sampled
                // to that same precision; no engine-specific units change.
                f.set(name, std::round(f.get(name))); key += juce::String(name)+juce::String(f.get(name))+"_";
            }
            auto found = features.find(key);
            if (found == features.end()) {
                const auto globalKey = featurePrefix+key;
                HifiFeatures value;
                {
                    std::scoped_lock lock(featuresMutex);
                    if (!force) if (const auto cached = featuresCache.find(globalKey); cached != featuresCache.end()) value = cached->second;
                }
                if (value.mel.frames == 0) {
                    value = hifiFeatures(input, harmonic, f, *config);
                    std::scoped_lock lock(featuresMutex);
                    std::size_t bytes = value.mel.values.size()*sizeof(float);
                    for (const auto& item : featuresCache) bytes += item.second.mel.values.size()*sizeof(float);
                    if (bytes > 64*1024*1024 || featuresCache.size() >= 128) featuresCache.clear();
                    featuresCache[globalKey] = value;
                }
                if (features.size() >= 16) features.erase(features.begin());
                found = features.emplace(key, std::move(value)).first;
            }
            const auto& feature = found->second;
            const auto time = (localFrame+.5)*hopSeconds;
            const auto sourceTime = hifiContextSourceTime(timeMap, plan, time, config->sampleRate, f.loop);
            for (int band = 0; band < config->melBands; ++band) {
                mel.values[static_cast<std::size_t>(band)*modelFrames+modelFrame]
                    = hifiSampleMel(feature.mel, band, sourceTime, config->sampleRate);
            }
            // Pitch and flag curves retain the upstream integer-hop grid. The
            // half-hop correction applies to Mel sampling, not the note timeline.
            const auto framePitch = curveAt(midi, i*hopSeconds*1000/framePeriodMs)+flags[i].get("t")/100;
            f0[modelFrame] = midiToHz(framePitch);
            if (localFrame >= 0 && localFrame < static_cast<std::ptrdiff_t>(frames)) {
                scales[i] = feature.scale; pitch[i] = framePitch;
            }
        }
        // Native formant is in semitones, independently composed after the
        // existing FLAG g/Ht feature transforms and target-time Mel warp. Never
        // transpose F0 or substitute native values into FLAG-specific units.
        // Cover the full decoder context; skip neutral to preserve old bytes.
        if (std::isfinite(note.nativeFormantSemitones) && std::abs(note.nativeFormantSemitones) >= 5.0e-4f)
        {
            const std::vector<float> nativeFormants(modelFrames, note.nativeFormantSemitones);
            shiftMelFormants(mel, *config, nativeFormants, hopSeconds * 1000.0);
        }
        auto entry = session(files->onnx, execution); result.activeInference = entry.activeInference;
        auto decoded = infer(*entry.model, *config, mel, f0, entry.runMutex.get());
        const auto crop = context*config->hop;
        if (decoded.size() < crop+samples) throw std::runtime_error("HiFisampler decoder returned short audio");
        std::vector<float> output(decoded.begin()+static_cast<std::ptrdiff_t>(crop), decoded.begin()+static_cast<std::ptrdiff_t>(crop+samples));
        std::vector<float> amplitude(frames, 1);
        for (std::size_t i = 0; i < frames; ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
            const auto derivative = frames <= 1 ? 0 : i == 0 ? (pitch[1]-pitch[0])/hopSeconds
                : i+1 == frames ? (pitch[i]-pitch[i-1])/hopSeconds : (pitch[i+1]-pitch[i-1])/(2*hopSeconds);
            amplitude[i] = static_cast<float>(std::pow(5.0, 1.0e-4*flags[i].get("A")*derivative))/scales[i];
        }
        for (std::size_t i = 0; i < output.size(); ++i) {
        if ((i & 1023) == 0) checkNsfCancellation();
            const auto at = static_cast<double>(i)/config->hop;
            output[i] *= scalarCurveAt(amplitude, std::min(at, static_cast<double>(frames-1)), amplitude.back());
            const auto& f = flags[std::min(flags.size()-1, i/static_cast<std::size_t>(config->hop))];
            if (f.get("Hb") == 0 && f.get("Hv") == 0) output[i] = 0;
        }
        hifiGrowl(output, config->sampleRate, flags, hopSeconds);
        if (waveNorm) hifiNormalize(output, config->sampleRate, flags, hopSeconds, trim);
        const auto peak = hifiPeak(output); if (peak > 1) for (auto& v : output) v /= peak;
        output = resample(output.data(), static_cast<int>(output.size()), config->sampleRate, static_cast<int>(std::llround(sampleRate)));
        output.resize(static_cast<std::size_t>(targetSamples), 0);
        result.buffer.setSize(source.getNumChannels(), targetSamples);
        for (int c = 0; c < result.buffer.getNumChannels(); ++c) {
            std::copy(output.begin(), output.end(), result.buffer.getWritePointer(c));
            conditionNeuralBoundary(result.buffer.getWritePointer(c), targetSamples, sampleRate, .003f, .003f);
        }
        checkNsfCancellation();
        result.modelFile = files->onnx; result.usedModel = true;
    }
    catch (const std::exception& e) { result.error = "HiFisampler: " + juce::String(e.what()); }
    if (!result.usedModel) result.buffer.setSize(0, 0);
    return result;
}
