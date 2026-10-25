#pragma once
#include <juce_core/juce_core.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace hachi::backend
{
// Internal CVVC dividers, keyed by the phoneme to their right (CV=2, release=3).
// Positions are relative to the score note, in units of its duration. A CV
// divider may be negative because the preceding VC starts before the note.
using CvvcBoundaryPositions = std::map<int, double>;

inline double clampCvvcBoundary(double position, double from, double to)
{
    const auto margin = std::min(0.001, std::max(0.0, to-from)/4.0);
    return std::clamp(position, from+margin, to-margin);
}

// Shared by the planner and the consonant-head preview. Caller supplies
// phonemes in their existing order, with a fixed first onset and group end.
inline void applyCvvcBoundaries(std::vector<double>& starts, const std::vector<int>& parts,
                               double duration, double end, const CvvcBoundaryPositions& edits)
{
    if(edits.empty() || starts.empty() || starts.size()!=parts.size() || end<=starts.front())return;
    const auto margin=std::min(0.001,(end-starts.front())/(2.0*starts.size()));
    for(size_t i=1;i<starts.size();++i)
    {
        const auto found=edits.find(parts[i]);
        const auto wanted=found==edits.end()?starts[i]:found->second*duration;
        starts[i]=std::clamp(wanted,starts[i-1]+margin,end-(starts.size()-i)*margin);
    }
}

inline juce::var cvvcBoundariesToVar(const CvvcBoundaryPositions& positions)
{
    juce::Array<juce::var> result;
    for (const auto& [part, position] : positions)
        result.add(juce::var(juce::Array<juce::var>{part, position}));
    return result;
}

inline CvvcBoundaryPositions cvvcBoundariesFromVar(const juce::var& value)
{
    CvvcBoundaryPositions result;
    if (const auto* list=value.getArray()) for (const auto& item:*list)
        if (const auto* row=item.getArray(); row && row->size()==2)
        {
            const auto part=static_cast<int>((*row)[0]);
            const auto position=static_cast<double>((*row)[1]);
            if ((part==2 || part==3) && std::isfinite(position)) result[part]=position;
        }
    return result;
}
}
