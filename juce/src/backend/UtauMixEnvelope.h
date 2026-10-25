#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace hachi::backend
{
// The automatic mixer gain, separate from the user's editable envelope.
// Times belong to the callback note; CVVC translates only startSeconds when
// folding a generated phoneme back into its score note. Never persisted.
struct UtauMixRamp
{
    int start = 0, length = 1;
    bool rising = true, equalPower = false;
    double phaseOffset = 0;
    int activeSamples = -1; // finite NSF attenuation window; -1 holds the end
    bool crossfade = false;
    bool smoothCurve = false;

    [[nodiscard]] float gainAt(int sample) const
    {
        const auto local = sample - start;
        if (activeSamples >= 0 && (local < 0 || local >= activeSamples)) return 1.f;
        if (local < 0) return rising ? 0.f : 1.f;
        if (local >= length) return rising ? 1.f : 0.f;
        const auto u = std::clamp((local + phaseOffset) / std::max(1, length), 0.0, 1.0);
        if (smoothCurve) { const auto v=u*u*(3-2*u); return static_cast<float>(rising?v:1-v); }
        if (!equalPower) return static_cast<float>(rising ? u : 1.0 - u);
        constexpr double halfPi = 1.57079632679489661923;
        return static_cast<float>(rising ? std::sin(u * halfPi) : std::cos(u * halfPi));
    }
};

struct UtauMixEnvelope
{
    double sampleRate = 0, startSeconds = 0;
    int samples = 0;
    std::vector<UtauMixRamp> ramps;
    bool replacesStandardEnvelope = false;

    [[nodiscard]] bool available() const { return sampleRate > 0 && samples > 0; }
    [[nodiscard]] double endSeconds() const { return startSeconds + (sampleRate > 0 ? samples / sampleRate : 0); }
    [[nodiscard]] float gainAtSample(int sample) const
    {
        if (sample < 0 || sample >= samples) return 0.f;
        float gain = 1;
        for (const auto& ramp : ramps) gain *= ramp.gainAt(sample);
        return gain;
    }
    [[nodiscard]] float gainAt(double seconds) const
    {
        return available() ? gainAtSample(static_cast<int>(std::llround((seconds-startSeconds)*sampleRate))) : 1.f;
    }
};

// Standard CVVC edges and automatic transitions are alternatives, not two
// envelopes multiplied together. Keep a single ramp at each sounding edge.
inline void useStandardUtauEnvelope(UtauMixEnvelope& envelope, double seconds)
{
    envelope.replacesStandardEnvelope=true;
    auto attack=.005, release=.035;
    if(seconds<=attack+release+.005)attack=release=seconds/3;
    for(auto& ramp:envelope.ramps)if(ramp.rising&&!ramp.crossfade)
        ramp={0,std::max(1,static_cast<int>(std::lround(attack*envelope.sampleRate))),true};
    if(std::none_of(envelope.ramps.begin(),envelope.ramps.end(),[](const auto& r){return !r.rising;}))
        envelope.ramps.push_back({static_cast<int>(std::lround((seconds-release)*envelope.sampleRate)),
            std::max(1,static_cast<int>(std::lround(release*envelope.sampleRate))),false});
}
}
