// 锁定 VST3 SDK 的原生布局 oracle，用于发现 Rust 音频 ABI 的偏移或对齐错误。
#include <cstddef>
#include <cstdio>
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"

using namespace Steinberg::Vst;

#define TYPE_LAYOUT(T) std::printf("\"" #T ".size\":%zu,\"" #T ".align\":%zu,", sizeof(T), alignof(T))
#define FIELD_LAYOUT(T, F) std::printf("\"" #T "." #F "\":%zu,", offsetof(T, F))

// 输出真实头文件计算的 JSON，不依赖 Rust 常量或代码生成结果。
int main() {
    std::printf("{");
    TYPE_LAYOUT(ProcessSetup);
    FIELD_LAYOUT(ProcessSetup, processMode);
    FIELD_LAYOUT(ProcessSetup, symbolicSampleSize);
    FIELD_LAYOUT(ProcessSetup, maxSamplesPerBlock);
    FIELD_LAYOUT(ProcessSetup, sampleRate);
    TYPE_LAYOUT(AudioBusBuffers);
    FIELD_LAYOUT(AudioBusBuffers, numChannels);
    FIELD_LAYOUT(AudioBusBuffers, silenceFlags);
    FIELD_LAYOUT(AudioBusBuffers, channelBuffers32);
    FIELD_LAYOUT(AudioBusBuffers, channelBuffers64);
    TYPE_LAYOUT(ProcessData);
    FIELD_LAYOUT(ProcessData, processMode);
    FIELD_LAYOUT(ProcessData, symbolicSampleSize);
    FIELD_LAYOUT(ProcessData, numSamples);
    FIELD_LAYOUT(ProcessData, numInputs);
    FIELD_LAYOUT(ProcessData, numOutputs);
    FIELD_LAYOUT(ProcessData, inputs);
    FIELD_LAYOUT(ProcessData, outputs);
    FIELD_LAYOUT(ProcessData, inputParameterChanges);
    FIELD_LAYOUT(ProcessData, outputParameterChanges);
    FIELD_LAYOUT(ProcessData, inputEvents);
    FIELD_LAYOUT(ProcessData, outputEvents);
    FIELD_LAYOUT(ProcessData, processContext);
    TYPE_LAYOUT(ProcessContext);
    FIELD_LAYOUT(ProcessContext, state);
    FIELD_LAYOUT(ProcessContext, sampleRate);
    FIELD_LAYOUT(ProcessContext, projectTimeSamples);
    FIELD_LAYOUT(ProcessContext, systemTime);
    FIELD_LAYOUT(ProcessContext, continousTimeSamples);
    FIELD_LAYOUT(ProcessContext, projectTimeMusic);
    FIELD_LAYOUT(ProcessContext, barPositionMusic);
    FIELD_LAYOUT(ProcessContext, cycleStartMusic);
    FIELD_LAYOUT(ProcessContext, cycleEndMusic);
    FIELD_LAYOUT(ProcessContext, tempo);
    FIELD_LAYOUT(ProcessContext, timeSigNumerator);
    FIELD_LAYOUT(ProcessContext, timeSigDenominator);
    FIELD_LAYOUT(ProcessContext, chord);
    FIELD_LAYOUT(ProcessContext, smpteOffsetSubframes);
    FIELD_LAYOUT(ProcessContext, frameRate);
    FIELD_LAYOUT(ProcessContext, samplesToNextClock);
    TYPE_LAYOUT(Chord);
    FIELD_LAYOUT(Chord, keyNote);
    FIELD_LAYOUT(Chord, rootNote);
    FIELD_LAYOUT(Chord, chordMask);
    TYPE_LAYOUT(FrameRate);
    FIELD_LAYOUT(FrameRate, framesPerSecond);
    FIELD_LAYOUT(FrameRate, flags);
    std::printf("\"end\":0}\n");
    return 0;
}
