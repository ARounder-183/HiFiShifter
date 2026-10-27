#pragma once
#include "ProjectModel.h"

namespace hachi
{
struct JoinedVibrato
{
    std::vector<juce::String> notes;
    double start = 0.0, duration = 0.0;
};

// The shared clock also identifies groups saved by update 246. Independently
// created vibratos have no reference clock and never acquire a group here.
inline std::vector<JoinedVibrato> joinedVibratos(const TrackData& track)
{
    struct Entry { const NoteData* note; double start, end; };
    std::vector<Entry> entries;
    for (const auto& clip : track.clips)
        for (const auto& note : clip.notes)
            entries.push_back({&note, clip.startSeconds+note.startSeconds,
                clip.startSeconds+note.startSeconds+note.durationSeconds});
    std::stable_sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.start<b.start;});
    std::vector<JoinedVibrato> groups;
    for (size_t first=0;first<entries.size();)
    {
        const auto& head=entries[first];const auto duration=head.note->vibratoReferenceDurationSeconds;
        const auto origin=head.start-head.note->vibratoTimeOffsetSeconds;
        auto last=first;
        while (duration>0 && !head.note->nativeUnpitched && last+1<entries.size())
        {
            const auto& next=entries[last+1];const auto seam=entries[last].end;
            if (next.note->nativeUnpitched || std::abs(next.start-seam)>1.e-7
                || std::abs(next.note->midiNote-head.note->midiNote)>1.e-4
                || std::abs(next.note->vibratoReferenceDurationSeconds-duration)>1.e-7
                || std::abs(next.start-next.note->vibratoTimeOffsetSeconds-origin)>1.e-7) break;
            if (std::any_of(entries.begin(),entries.end(),[&](const auto& other){return other.note!=entries[last].note
                &&other.note!=next.note&&other.start<seam-1.e-7&&other.end>seam+1.e-7;})) break;
            ++last;
        }
        if (last>first)
        {
            JoinedVibrato group;group.start=origin;group.duration=duration;
            for(auto i=first;i<=last;++i)group.notes.push_back(entries[i].note->id);
            groups.push_back(std::move(group));
        }
        first=last+1;
    }
    return groups;
}

inline std::vector<juce::String> joinedVibratoTargets(const TrackData& track,
    const std::vector<juce::String>& selected)
{
    auto targets=selected;
    for(const auto& group:joinedVibratos(track))
        if(std::any_of(group.notes.begin(),group.notes.end(),[&](const auto& id){return std::find(selected.begin(),selected.end(),id)!=selected.end();}))
            for(const auto& id:group.notes)
                if(std::find(targets.begin(),targets.end(),id)==targets.end())targets.push_back(id);
    return targets;
}

inline void copyVibratoParameters(NoteData& target,const NoteData& source)
{
    target.vibratoEnabled=source.vibratoEnabled;target.vibratoRealLine=source.vibratoRealLine;
    target.vibratoLengthPercent=source.vibratoLengthPercent;target.vibratoCycleMs=source.vibratoCycleMs;
    target.vibratoDepthCents=source.vibratoDepthCents;target.vibratoFadeInPercent=source.vibratoFadeInPercent;
    target.vibratoFadeOutPercent=source.vibratoFadeOutPercent;target.vibratoPhasePercent=source.vibratoPhasePercent;
    target.vibratoOffsetPercent=source.vibratoOffsetPercent;target.vibratoEndPercent=source.vibratoEndPercent;
}
}
