#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace hachi::backend
{
namespace nativeExpressionDetail
{
inline float valueAt(const std::vector<float>& curve, double position, float fallback)
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
}

// Native note units: gain 1 is neutral, tension [-1,1], breath [0,1].
// Shared by native-file rendering and native NSF voicebank notes. No FLAG-unit
// translation, pitch change, synthesis or resampling is performed here.
// This is the original RenderService expression implementation, kept verbatim
// so extracting it does not alter the native-file DSP or its loudness matching.
inline void applyNativeExpression(juce::AudioBuffer<float>& audio, double sampleRate,
                                   double framePeriodMs,
                                   const std::vector<float>& targetMidi,
                                   const std::vector<float>& noteGain,
                                   const std::vector<float>& tension,
                                   const std::vector<float>& breath,
                                   juce::AudioBuffer<float>* noise = nullptr)
{
    using nativeExpressionDetail::valueAt;
    if (audio.getNumSamples() <= 0 || sampleRate <= 0.0) return;
    const auto framePeriod = std::max(0.1, framePeriodMs);
    const auto hasTension = std::any_of(tension.begin(), tension.end(),
        [](float value) { return std::abs(value) > 1.0e-4f; });

    // Tension is applied separately from gain/breath so its spectral tilt can
    // preserve loudness. Combining both in one loop made +100% tension nearly
    // double the RMS even though the user only requested a timbre change.
    if (hasTension)
        for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        {
            auto tiltLow = audio.getSample(channel, 0);
            auto noiseLow = noise != nullptr ? noise->getSample(channel, 0) : 0.0f;
            auto smoothedTension = 0.0f;
            double inputPower = 0.0;
            double outputPower = 0.0;
            auto inputPeak = 0.0f;
            auto outputPeak = 0.0f;
            for (int sample = 0; sample < audio.getNumSamples(); ++sample)
            {
                const auto curvePosition = static_cast<double>(sample) / sampleRate
                    * 1000.0 / framePeriod;
                const auto desiredTension = juce::jlimit(-1.0f, 1.0f,
                    valueAt(tension, curvePosition, 0.0f));
                const auto smoothing = static_cast<float>(1.0
                    - std::exp(-1.0 / (sampleRate * 0.008)));
                smoothedTension += (desiredTension - smoothedTension) * smoothing;

                const auto midi = valueAt(targetMidi, curvePosition, 60.0f);
                const auto f0 = midi > 0.0f
                    ? 440.0 * std::pow(2.0, (static_cast<double>(midi) - 69.0) / 12.0)
                    : 220.0;
                const auto splitHz = juce::jlimit(260.0, 2'400.0, f0 * 2.0);
                const auto coefficient = static_cast<float>(1.0
                    - std::exp(-juce::MathConstants<double>::twoPi * splitHz / sampleRate));
                const auto current = audio.getSample(channel, sample);
                tiltLow += coefficient * (current - tiltLow);
                const auto tiltHigh = current - tiltLow;
                const auto highGain = std::pow(10.0f, smoothedTension * 8.0f / 20.0f);
                const auto lowGain = std::pow(10.0f, -smoothedTension * 1.5f / 20.0f);
                const auto tilted = tiltLow * lowGain + tiltHigh * highGain;
                audio.setSample(channel, sample, tilted);
                if (noise != nullptr)
                {
                    const auto n = noise->getSample(channel, sample);
                    noiseLow += coefficient * (n - noiseLow);
                    noise->setSample(channel, sample, noiseLow * lowGain + (n - noiseLow) * highGain);
                }
                inputPower += static_cast<double>(current) * current;
                outputPower += static_cast<double>(tilted) * tilted;
                inputPeak = std::max(inputPeak, std::abs(current));
                outputPeak = std::max(outputPeak, std::abs(tilted));
            }

            const auto inputRms = std::sqrt(inputPower / audio.getNumSamples());
            const auto outputRms = std::sqrt(outputPower / audio.getNumSamples());
            auto correction = inputRms > 1.0e-8 && outputRms > 1.0e-8
                ? static_cast<float>(inputRms / outputRms) : 1.0f;
            correction = juce::jlimit(0.4f, 2.5f, correction);
            if (inputPeak > 1.0e-6f && outputPeak > 1.0e-6f)
                correction = std::min(correction, inputPeak * 1.6f / outputPeak);
            audio.applyGain(channel, 0, audio.getNumSamples(), correction);
            if (noise != nullptr) noise->applyGain(channel, 0, noise->getNumSamples(), correction);
        }

    for (int channel = 0; channel < audio.getNumChannels(); ++channel)
    {
        auto airLow = audio.getSample(channel, 0);
        auto noiseLow = noise != nullptr ? noise->getSample(channel, 0) : 0.0f;
        const auto airCoefficient = static_cast<float>(1.0
            - std::exp(-juce::MathConstants<double>::twoPi * 3500.0 / sampleRate));
        for (int sample = 0; sample < audio.getNumSamples(); ++sample)
        {
            const auto curvePosition = static_cast<double>(sample) / sampleRate
                * 1000.0 / framePeriod;
            const auto gain = juce::jlimit(0.0f, 4.0f,
                valueAt(noteGain, curvePosition, 1.0f));
            const auto air = juce::jlimit(0.0f, 1.0f,
                valueAt(breath, curvePosition, 0.0f));
            const auto current = audio.getSample(channel, sample);
            airLow += airCoefficient * (current - airLow);
            const auto airHigh = current - airLow;
            audio.setSample(channel, sample, (current + airHigh * air * 0.22f) * gain);
            if (noise != nullptr)
            {
                const auto n = noise->getSample(channel, sample);
                noiseLow += airCoefficient * (n - noiseLow);
                noise->setSample(channel, sample, (n + (n - noiseLow) * air * 0.22f) * gain);
            }
        }
    }
}
}
