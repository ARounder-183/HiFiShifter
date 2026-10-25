#include "PronunciationTiming.h"
#include "SampleSettings.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace hachi
{
juce::String pronunciationKindName(PronunciationKind kind)
{
    switch (kind)
    {
        case PronunciationKind::transition: return "transition";
        case PronunciationKind::breath: return "breath";
        default: return "normal";
    }
}

PronunciationKind parsePronunciationKind(const juce::String& value)
{
    const auto name = value.trim().toLowerCase();
    if (name == "transition" || name == "t") return PronunciationKind::transition;
    if (name == "breath" || name == "b") return PronunciationKind::breath;
    return PronunciationKind::normal;
}

PronunciationTiming pronunciationTimingFor(const SampleRegionSetting& row)
{
    PronunciationTiming timing;
    timing.sourceStartSeconds = std::isfinite(row.regionStartSeconds) ? std::max(0.0, row.regionStartSeconds) : 0.0;
    timing.sourceEndSeconds = std::isfinite(row.regionEndSeconds) && row.regionEndSeconds > timing.sourceStartSeconds
        ? row.regionEndSeconds : timing.sourceStartSeconds + .001;
    timing.alignmentSeconds = juce::jlimit(timing.sourceStartSeconds, timing.sourceEndSeconds,
        std::isfinite(row.alignmentSeconds) ? row.alignmentSeconds : timing.sourceStartSeconds);
    timing.consonantEndSeconds = juce::jlimit(timing.sourceStartSeconds, timing.sourceEndSeconds,
        timing.sourceStartSeconds + (std::isfinite(row.fixedDurationSeconds) ? row.fixedDurationSeconds : 0.0));
    timing.overlapSeconds = std::isfinite(row.overlapSeconds) ? row.overlapSeconds : 0.0;
    timing.simplifiedConsonant = row.simplifiedConsonant;
    if (timing.simplifiedConsonant) timing.consonantEndSeconds = std::max(timing.alignmentSeconds, timing.consonantEndSeconds);
    return timing;
}

double protectedPronunciationSourceEnd(const PronunciationTiming& timing)
{
    return std::max(timing.alignmentSeconds, timing.consonantEndSeconds);
}

double consonantAfterAlignmentSeconds(const PronunciationTiming& timing)
{
    return std::max(0.0, timing.consonantEndSeconds - timing.alignmentSeconds);
}

namespace
{
constexpr double timingEpsilon = 1.e-9;

bool validSourceZones(const SampleRegionSetting& row, juce::String& error)
{
    if (!row.vowelSplitsExplicit) return true; // raw analysis epochs are not authoring zones
    std::vector<std::pair<double, double>> zones;
    for (const auto& segment : row.segments)
    {
        if (!std::isfinite(segment.sourceStartSeconds) || !std::isfinite(segment.sourceEndSeconds)
            || segment.sourceEndSeconds <= segment.sourceStartSeconds)
        { error = "Invalid native source zone"; return false; }
        zones.emplace_back(segment.sourceStartSeconds, segment.sourceEndSeconds);
    }
    std::sort(zones.begin(), zones.end());
    for (std::size_t i = 1; i < zones.size(); ++i)
        if (zones[i].first < zones[i - 1].second - timingEpsilon)
        { error = "Native source zones overlap"; return false; }
    return true;
}

std::vector<NativeSegment> canonicalAbsoluteSegments(const SampleRegionSetting& row, const PronunciationTiming& timing)
{
    const auto start = timing.sourceStartSeconds, end = timing.sourceEndSeconds;
    std::vector<double> cuts{start, end, timing.consonantEndSeconds, protectedPronunciationSourceEnd(timing)};
    const auto authoredZone = [&](const NativeSegment& segment)
    {
        if (!row.vowelSplitsExplicit || segment.role == NativeSegmentRole::consonant) return false;
        return segment.role != NativeSegmentRole::vowel
            || (segment.alias.trim().isNotEmpty() && segment.alias.trim() != "-");
    };
    for (const auto& segment : row.segments)
        if (authoredZone(segment) && std::isfinite(segment.sourceStartSeconds) && std::isfinite(segment.sourceEndSeconds)
            && segment.sourceEndSeconds > segment.sourceStartSeconds + timingEpsilon)
        {
            // Named vowel subdivisions do not split the canonical C prefix.
            for (const auto boundary : {segment.sourceStartSeconds, segment.sourceEndSeconds})
                if (segment.role != NativeSegmentRole::vowel || boundary > timing.consonantEndSeconds + timingEpsilon)
                    cuts.push_back(juce::jlimit(start, end, boundary));
        }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double a, double b) { return std::abs(a - b) <= timingEpsilon; }), cuts.end());
    struct Piece { NativeSegment segment; int owner = -1; };
    std::vector<Piece> pieces;
    for (std::size_t i = 1; i < cuts.size(); ++i)
    {
        if (cuts[i] - cuts[i - 1] <= timingEpsilon) continue;
        const auto middle = (cuts[i] + cuts[i - 1]) * .5;
        int owner = -1;
        if (row.vowelSplitsExplicit)
        {
            for (std::size_t j = 0; j < row.segments.size(); ++j)
                if (authoredZone(row.segments[j]) && row.segments[j].sourceStartSeconds <= middle && row.segments[j].sourceEndSeconds >= middle)
                { owner = static_cast<int>(j); break; }
        }
        // A default pronunciation is ONE C and ONE logical V, regardless of
        // how many pitch-analysis epochs happen to exist in its raw metadata.
        // Reusing one prior ID is harmless; preserving all prior cuts is not.
        if (owner < 0 && middle < timing.consonantEndSeconds)
        {
            for (std::size_t j = 0; j < row.segments.size(); ++j)
                if (row.segments[j].role == NativeSegmentRole::consonant
                    && row.segments[j].sourceStartSeconds < timing.consonantEndSeconds
                    && row.segments[j].sourceEndSeconds > start)
                { owner = static_cast<int>(j); break; }
        }
        else if (owner < 0 && !row.vowelSplitsExplicit)
        {
            for (std::size_t j = row.segments.size(); j > 0; --j)
                if (row.segments[j - 1].role == NativeSegmentRole::vowel
                    && row.segments[j - 1].sourceEndSeconds > timing.consonantEndSeconds
                    && row.segments[j - 1].sourceStartSeconds < end)
                { owner = static_cast<int>(j - 1); break; }
        }
        else if (owner < 0)
        {
            // A stale C tag outside the canonical prefix does not add a cut,
            // but an already-required interval may retain its old stable ID.
            for (std::size_t j = 0; j < row.segments.size(); ++j)
                if (row.segments[j].role == NativeSegmentRole::consonant
                    && row.segments[j].sourceStartSeconds <= middle && row.segments[j].sourceEndSeconds >= middle)
                { owner = static_cast<int>(j); break; }
        }
        NativeSegment segment;
        if (owner >= 0) segment = row.segments[static_cast<std::size_t>(owner)];
        else
        {
            segment.alias = row.name.trim().isEmpty() ? juce::String("-") : row.name.trim();
            segment.provenance = row.provenance.isEmpty() ? juce::String("estimated") : row.provenance;
            segment.confidence = row.confidence;
        }
        if (!row.vowelSplitsExplicit)
        {
            segment.alias = row.name.trim().isEmpty() ? juce::String("-") : row.name.trim();
            segment.stretchWeight = 1.0;
        }
        // Whole-pronunciation identity is not an internal zone role. Only an
        // explicit authoring snapshot may retain advanced non-C/V source zones.
        const bool otherZone = row.vowelSplitsExplicit && owner >= 0
            && segment.role != NativeSegmentRole::consonant && segment.role != NativeSegmentRole::vowel;
        if (!otherZone) segment.role = middle < timing.consonantEndSeconds ? NativeSegmentRole::consonant : NativeSegmentRole::vowel;
        segment.sourceStartSeconds = cuts[i - 1]; segment.sourceEndSeconds = cuts[i];
        segment.alignmentSeconds = timing.alignmentSeconds;
        segment.overlapSeconds = timing.overlapSeconds; // signed offset, not an absolute marker
        segment.stretchable = segment.role == NativeSegmentRole::vowel && middle >= protectedPronunciationSourceEnd(timing);
        if (!pieces.empty() && pieces.back().owner == owner && pieces.back().segment.role == segment.role
            && pieces.back().segment.stretchable == segment.stretchable
            && std::abs(pieces.back().segment.sourceEndSeconds - segment.sourceStartSeconds) <= timingEpsilon)
            pieces.back().segment.sourceEndSeconds = segment.sourceEndSeconds;
        else pieces.push_back({std::move(segment), owner});
    }
    // Keep each original ID on its best surviving semantic piece, preferring
    // the elastic remainder of a split vowel. Newly introduced cuts get IDs
    // using the existing segment_N convention without colliding with old IDs.
    std::set<juce::String> reserved, used;
    for (const auto& segment : row.segments) if (segment.id.isNotEmpty()) reserved.insert(segment.id);
    for (auto& piece : pieces) piece.segment.id.clear();
    for (std::size_t j = 0; j < row.segments.size(); ++j)
    {
        const auto& original = row.segments[j];
        if (original.id.isEmpty() || used.contains(original.id)) continue;
        int best = -1, score = -1;
        for (std::size_t i = 0; i < pieces.size(); ++i)
            if (pieces[i].owner == static_cast<int>(j))
            {
                const int candidate = (pieces[i].segment.role == original.role ? 2 : 0)
                    + (original.role == NativeSegmentRole::vowel && pieces[i].segment.stretchable ? 1 : 0);
                if (candidate > score) { score = candidate; best = static_cast<int>(i); }
            }
        if (best >= 0) { pieces[static_cast<std::size_t>(best)].segment.id = original.id; used.insert(original.id); }
    }
    int number = 1;
    std::vector<NativeSegment> result;
    for (auto& piece : pieces)
    {
        if (piece.segment.id.isEmpty())
        {
            juce::String id;
            do { id = "segment_" + juce::String(number++); } while (reserved.contains(id) || used.contains(id));
            piece.segment.id = id; used.insert(id);
        }
        result.push_back(std::move(piece.segment));
    }
    return result;
}
}

std::vector<NativeSegment> pronunciationSegmentsFor(const SampleRegionSetting& row)
{
    const auto timing = pronunciationTimingFor(row);
    auto segments = canonicalAbsoluteSegments(row, timing);
    for (auto& segment : segments)
    {
        segment.sourceStartSeconds -= timing.sourceStartSeconds;
        segment.sourceEndSeconds -= timing.sourceStartSeconds;
        segment.alignmentSeconds -= timing.sourceStartSeconds;
    }
    return segments;
}

bool applyPronunciationTiming(SampleRegionSetting& row, PronunciationTiming timing, juce::String& error)
{
    error.clear();
    if (!std::isfinite(timing.sourceStartSeconds) || !std::isfinite(timing.sourceEndSeconds)
        || !std::isfinite(timing.alignmentSeconds) || !std::isfinite(timing.consonantEndSeconds)
        || !std::isfinite(timing.overlapSeconds) || timing.sourceStartSeconds < 0
        || timing.sourceEndSeconds - timing.sourceStartSeconds <= timingEpsilon)
    { error = "Invalid pronunciation source timing"; return false; }
    if (!validSourceZones(row, error)) return false;
    timing.alignmentSeconds = juce::jlimit(timing.sourceStartSeconds, timing.sourceEndSeconds, timing.alignmentSeconds);
    timing.consonantEndSeconds = juce::jlimit(timing.sourceStartSeconds, timing.sourceEndSeconds, timing.consonantEndSeconds);
    if (timing.simplifiedConsonant) timing.consonantEndSeconds = std::max(timing.alignmentSeconds, timing.consonantEndSeconds);
    auto next = row;
    next.regionStartSeconds = timing.sourceStartSeconds; next.regionEndSeconds = timing.sourceEndSeconds;
    next.alignmentSeconds = timing.alignmentSeconds;
    next.fixedDurationSeconds = timing.consonantEndSeconds - timing.sourceStartSeconds;
    next.overlapSeconds = timing.overlapSeconds;
    next.simplifiedConsonant = timing.simplifiedConsonant;
    next.segments = canonicalAbsoluteSegments(next, timing);
    row = std::move(next);
    return true;
}

bool editPronunciationAlignment(SampleRegionSetting& row, double absoluteSourceSeconds, juce::String& error)
{
    if (!std::isfinite(absoluteSourceSeconds)) { error = "Invalid alignment marker"; return false; }
    auto timing = pronunciationTimingFor(row);
    const auto delta = consonantAfterAlignmentSeconds(timing);
    timing.alignmentSeconds = juce::jlimit(timing.sourceStartSeconds, timing.sourceEndSeconds, absoluteSourceSeconds);
    if (timing.simplifiedConsonant) timing.consonantEndSeconds = std::min(timing.sourceEndSeconds, timing.alignmentSeconds + delta);
    return applyPronunciationTiming(row, timing, error);
}

bool editPronunciationConsonant(SampleRegionSetting& row, double absoluteSourceSeconds, juce::String& error)
{
    if (!std::isfinite(absoluteSourceSeconds)) { error = "Invalid consonant marker"; return false; }
    auto timing = pronunciationTimingFor(row);
    timing.consonantEndSeconds = juce::jlimit(timing.simplifiedConsonant ? timing.alignmentSeconds : timing.sourceStartSeconds,
                                             timing.sourceEndSeconds, absoluteSourceSeconds);
    return applyPronunciationTiming(row, timing, error);
}

bool setSimplifiedConsonant(SampleRegionSetting& row, bool enabled, double afterAlignmentSeconds, juce::String& error)
{
    auto timing = pronunciationTimingFor(row);
    // -1 captures the current derived gap; nonnegative values are an explicit
    // delta edit, including on first enable. No second numeric authority exists.
    if (!std::isfinite(afterAlignmentSeconds) || (afterAlignmentSeconds < 0 && afterAlignmentSeconds != -1.0))
    { error = "Consonant-after-alignment duration must be nonnegative or -1 to keep the current gap"; return false; }
    const auto delta = afterAlignmentSeconds == -1.0
        ? consonantAfterAlignmentSeconds(timing) : afterAlignmentSeconds;
    timing.simplifiedConsonant = enabled;
    if (enabled) timing.consonantEndSeconds = std::min(timing.sourceEndSeconds, timing.alignmentSeconds + delta);
    return applyPronunciationTiming(row, timing, error);
}

bool makePronunciationStretchMap(const SampleRegionSetting& row, double targetDurationSeconds,
                                 std::vector<SourceTimePoint>& result, juce::String& error)
{
    error.clear();
    if (!std::isfinite(targetDurationSeconds) || targetDurationSeconds <= 0)
    { error = "Target duration must be positive"; return false; }
    if (!std::isfinite(row.regionStartSeconds) || !std::isfinite(row.regionEndSeconds)
        || row.regionStartSeconds < 0 || row.regionEndSeconds <= row.regionStartSeconds + timingEpsilon
        || !std::isfinite(row.alignmentSeconds) || !std::isfinite(row.fixedDurationSeconds) || !std::isfinite(row.overlapSeconds))
    { error = "Invalid pronunciation source timing"; return false; }
    if (!validSourceZones(row, error)) return false;
    const auto timing = pronunciationTimingFor(row);
    const auto segments = pronunciationSegmentsFor(row);
    double fixed = 0, weightSum = 0;
    std::vector<double> lengths(segments.size()), weights(segments.size());
    std::vector<std::size_t> elastic;
    for (std::size_t i = 0; i < segments.size(); ++i)
    {
        const auto& segment = segments[i];
        lengths[i] = segment.sourceEndSeconds - segment.sourceStartSeconds;
        if (segment.role == NativeSegmentRole::vowel && segment.stretchable)
        {
            if (!std::isfinite(segment.stretchWeight) || segment.stretchWeight <= 0)
            { error = "Elastic vowel weight must be positive"; return false; }
            weights[i] = lengths[i] * segment.stretchWeight;
            weightSum += weights[i]; elastic.push_back(i);
        }
        else fixed += lengths[i];
    }
    constexpr double vowelMinimum = 1.e-6;
    const auto sourceDuration = timing.sourceEndSeconds - timing.sourceStartSeconds;
    if (elastic.empty())
    {
        if (std::abs(targetDurationSeconds - sourceDuration) > 1.e-7)
        { error = "This pronunciation has no elastic vowel source"; return false; }
    }
    else
    {
        auto budget = targetDurationSeconds - fixed;
        if (budget < vowelMinimum * elastic.size() - 1.e-12 || weightSum <= 0 || !std::isfinite(weightSum))
        { error = "Target is shorter than the protected source and minimum vowels"; return false; }
        auto remaining = elastic;
        // Weighted source-duration allocation with positive vowel floors. Fixed
        // source zones retain their exact lengths, including early V before P.
        while (!remaining.empty())
        {
            bool floored = false;
            for (auto it = remaining.begin(); it != remaining.end();)
            {
                const auto i = *it;
                if (budget * weights[i] / weightSum < vowelMinimum)
                {
                    lengths[i] = vowelMinimum; budget -= vowelMinimum; weightSum -= weights[i];
                    it = remaining.erase(it); floored = true;
                }
                else ++it;
            }
            if (!floored)
            {
                for (const auto i : remaining) lengths[i] = budget * weights[i] / weightSum;
                break;
            }
        }
    }
    std::vector<SourceTimePoint> mapped{{0.0, timing.sourceStartSeconds}};
    double target = 0;
    for (std::size_t i = 0; i < segments.size(); ++i)
    {
        target += lengths[i];
        mapped.push_back({target, timing.sourceStartSeconds + segments[i].sourceEndSeconds});
    }
    if (!mapped.empty()) mapped.back().targetSeconds = targetDurationSeconds;
    result = std::move(mapped);
    return true;
}
}
