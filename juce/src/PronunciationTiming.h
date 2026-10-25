#pragma once
#include <juce_core/juce_core.h>
#include <vector>

namespace hachi
{
struct SampleRegionSetting;
struct NativeSegment;
struct SourceTimePoint;

// Identity of a complete pronunciation, never the C/V role of an internal zone.
enum class PronunciationKind { normal, transition, breath };
[[nodiscard]] juce::String pronunciationKindName(PronunciationKind);
[[nodiscard]] PronunciationKind parsePronunciationKind(const juce::String&);

// Start/end/alignment/C-end positions are ABSOLUTE seconds in the unchanged
// recording. overlapSeconds alone remains the signed OTO offset from start;
// its source marker is start + overlapSeconds (negative/long values are valid).
// SampleRegionSetting keeps the classic consonant duration as its wire format;
// this is the one shared interpretation used by native and OTO editor views.
struct PronunciationTiming
{
    double sourceStartSeconds = 0.0;
    double sourceEndSeconds = 0.0;
    double alignmentSeconds = 0.0;
    double consonantEndSeconds = 0.0;
    double overlapSeconds = 0.0;
    bool simplifiedConsonant = false;
};

[[nodiscard]] PronunciationTiming pronunciationTimingFor(const SampleRegionSetting&);
[[nodiscard]] double protectedPronunciationSourceEnd(const PronunciationTiming&);
[[nodiscard]] double consonantAfterAlignmentSeconds(const PronunciationTiming&);
bool applyPronunciationTiming(SampleRegionSetting&, PronunciationTiming, juce::String& error);
bool editPronunciationAlignment(SampleRegionSetting&, double absoluteSourceSeconds, juce::String& error);
bool editPronunciationConsonant(SampleRegionSetting&, double absoluteSourceSeconds, juce::String& error);
// The relative position is encoded by the existing C/P values, not a second
// redundant persisted numeric authority. Pass -1.0 when toggling to retain
// the current nonnegative gap; a value >= 0 explicitly edits that gap.
bool setSimplifiedConsonant(SampleRegionSetting&, bool enabled,
                            double afterAlignmentSeconds, juce::String& error);

// Local-to-pronunciation segments. C and everything before P are fixed;
// existing internal vowel subdivisions/weights are retained beyond that zone.
[[nodiscard]] std::vector<NativeSegment> pronunciationSegmentsFor(const SampleRegionSetting&);
// Target-local / source-absolute map. Refuses impossible contractions rather
// than stretching protected consonants or silently throwing source audio away.
bool makePronunciationStretchMap(const SampleRegionSetting&, double targetDurationSeconds,
                                 std::vector<SourceTimePoint>& result, juce::String& error);
}
