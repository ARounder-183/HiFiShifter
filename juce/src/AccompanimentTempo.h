#pragma once
#include "ProjectModel.h"
#include <atomic>

namespace hachi
{
// A captured audio region, never a mutable ProjectModel. No tempo/undo state
// is written by this task. The task owns and cancels its Python process.
class AccompanimentTempoTask final : public juce::Thread
{
public:
    explicit AccompanimentTempoTask(ClipData region, juce::File runtime = {});
    ~AccompanimentTempoTask() override;
    void cancel() { cancelled.store(true); }
    bool finished() const { return done.load(std::memory_order_acquire); }
    int stage() const { return phase.load(); }
    juce::var result() const { return finished() ? outcome : juce::var(); }
    static bool canAnalyse(const ClipData&);
    static juce::File runtimeConfig();
private:
    void run() override;
    juce::var analyse();
    ClipData clip;
    juce::File config;
    std::atomic<bool> cancelled{false}, done{false};
    std::atomic<int> phase{0};
    juce::var outcome;
};
}
