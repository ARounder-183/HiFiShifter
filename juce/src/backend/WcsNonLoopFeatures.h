#pragma once

#include "NsfHifiganRenderer.h"

namespace hachi::backend
{
// WCSNDM-style non-looping parameter clock, independent of OTO/phoneme timing.
// Reconstruct a physical magnitude spectrum from interpolated LLSM layer-1
// parameters, then feed the editor's existing NSF model. No PCM stretch pass.
struct WcsFeatureConfig
{
    int sampleRate = 44100;
    int hop = 512;
    int fftSize = 2048;
    int windowSize = 2048;
    int melBands = 128;
    std::vector<float> melWeights;
};

struct WcsNonLoopMel
{
    std::vector<float> values; // [mel band, target frame]
    bool reusedSourceAnalysis = false;
};

double wcsNonLoopSourceTime(const std::vector<NsfHifiganTimeMapPoint>& map,
                           double time, double sourceDuration, double targetDuration);
double wcsNonLoopTargetTime(const std::vector<NsfHifiganTimeMapPoint>& map,
                           double time, double sourceDuration, double targetDuration);

WcsNonLoopMel buildWcsNonLoopMel(
    const std::vector<float>& waveform, const WcsFeatureConfig& config,
    double targetDuration, double curveFramePeriodMs,
    const std::vector<float>& sourceMidi, std::vector<float>& targetF0,
    const std::vector<NsfHifiganTimeMapPoint>& timeMap,
    const std::vector<float>& sourceReferenceMel, std::size_t referenceFrames,
    const std::function<void()>& checkCancellation);

void clearWcsNonLoopFeatureCache();
}
