#pragma once
#include "NativeAudioDisconnect.h"

namespace hachi
{
// Cover the audio clock, rather than guessing whether a gap is a consonant,
// background noise or silence. The row pitch is only a display position.
inline std::size_t appendNativeUnpitchedRegions(ClipData& clip)
{
    if (!clip.sourceFile.existsAsFile() || !clip.parts.empty()) return 0;
    const auto first = clip.audioStartSeconds;
    const auto last = std::min(clip.durationSeconds, first + nativeAudioPreviewClip(clip).durationSeconds);
    if (last <= first) return 0;
    const auto clock = nativeClipClock(clip);
    std::vector<const NoteData*> order;
    for (const auto& note : clip.notes) order.push_back(&note);
    std::stable_sort(order.begin(), order.end(), [](auto a, auto b) { return a->startSeconds < b->startSeconds; });
    std::vector<NoteData> gaps;
    const auto emit = [&](double begin, double end, float displayPitch)
    {
        if (end - begin < 0.005) return;
        NoteData note;
        note.id = "unpitched_" + juce::Uuid().toString();
        note.label = ""; // Localised type label is drawn independently of lyrics.
        note.nativeUnpitched = true;
        note.nativeIndependentPitch = true;
        note.utauAutoPitchTransition = false;
        note.startSeconds = begin;
        note.durationSeconds = end - begin;
        note.consonantSeconds = 0;
        note.midiNote = note.sourceMidiCenter = displayPitch;
        note.sourcePitchMeasured = true;
        note.contour = {{0, 0, 0, false}, {note.durationSeconds, 0, 0, false}};
        note.nativeProvenance = "uncovered-audio";
        bindNativeNoteSource(note, clip, &clock);
        gaps.push_back(std::move(note));
    };
    auto cursor = first;
    auto pitch = order.empty() ? 60.0f : order.front()->midiNote;
    for (const auto* note : order)
    {
        const auto begin = juce::jlimit(first, last, note->startSeconds);
        const auto end = juce::jlimit(first, last, note->startSeconds + note->durationSeconds);
        emit(cursor, begin, pitch);
        cursor = std::max(cursor, end);
        pitch = note->midiNote;
    }
    emit(cursor, last, pitch);
    const auto count = gaps.size();
    if (count) clip.sourceTimeMap = clock;
    for (auto& note : gaps) clip.notes.push_back(std::move(note));
    std::stable_sort(clip.notes.begin(), clip.notes.end(), [](const auto& a, const auto& b) { return a.startSeconds < b.startSeconds; });
    return count;
}

// Selected spans minus audio still owned by unselected notes. Merge the
// result so a shared interval is closed only once by ripple deletion.
inline std::vector<juce::Range<double>> nativeNoteRemovalRanges(
    const ClipData& clip, const std::vector<juce::String>& ids)
{
    const auto selected = [&](const auto& id) { return std::find(ids.begin(),ids.end(),id)!=ids.end(); };
    std::vector<juce::Range<double>> cuts;
    for(const auto& note:clip.notes) if(selected(note.id))
    {
        const auto a=std::max(0.0,note.startSeconds), b=std::min(clip.durationSeconds,note.startSeconds+note.durationSeconds);
        if(b>a)cuts.emplace_back(a,b);
    }
    for(const auto& note:clip.notes) if(!selected(note.id))
    {
        std::vector<juce::Range<double>> remaining;
        const juce::Range<double> protectedRange(note.startSeconds,note.startSeconds+note.durationSeconds);
        for(const auto& cut:cuts)
        {
            if(!cut.intersects(protectedRange)){remaining.push_back(cut);continue;}
            if(cut.getStart()<protectedRange.getStart())remaining.emplace_back(cut.getStart(),protectedRange.getStart());
            if(cut.getEnd()>protectedRange.getEnd())remaining.emplace_back(protectedRange.getEnd(),cut.getEnd());
        }
        cuts=std::move(remaining);
    }
    std::stable_sort(cuts.begin(),cuts.end(),[](auto a,auto b){return a.getStart()<b.getStart();});
    std::vector<juce::Range<double>> merged;
    for(const auto& cut:cuts)
    {
        if(!merged.empty()&&cut.getStart()<=merged.back().getEnd())merged.back().setEnd(std::max(merged.back().getEnd(),cut.getEnd()));
        else merged.push_back(cut);
    }
    return merged;
}

// Delete actual playback spans. Keep the source file intact and reuse the
// source clock/trim machinery. Ordinary delete leaves the timeline gap.
inline std::vector<ClipData> removeNativeNoteAudio(
    const ClipData& original, const std::vector<juce::String>& ids, bool ripple = false)
{
    const auto selected = [&](const auto& id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    if (std::none_of(original.notes.begin(), original.notes.end(), [&](const auto& n) { return selected(n.id); }))
        return {original};
    std::vector<ClipData> result;
    for (auto source : expandedClipParts(original))
    {
        const auto cuts = nativeNoteRemovalRanges(source, ids);
        if (cuts.empty()) { result.push_back(std::move(source)); continue; }
        rememberNativeTrimSource(source);
        const auto absoluteStart = source.startSeconds;
        source.startSeconds = 0;
        source.sourceTimeMap = nativeClipClock(source);
        const auto emit = [&](double begin, double end)
        {
            if (end - begin < 1.e-7) return;
            auto parts = slicedClipParts({source}, begin, end, false, true);
            if (parts.empty()) return;
            auto part = std::move(parts.front());
            part.startSeconds = absoluteStart + begin;
            for (auto note : source.notes)
                if (!selected(note.id) && note.startSeconds >= begin - 1.e-7
                    && note.startSeconds + note.durationSeconds <= end + 1.e-7)
                {
                    bindNativeNoteSource(note, source, &source.sourceTimeMap);
                    note.startSeconds -= begin;
                    note.clipPartId.clear();
                    part.notes.push_back(std::move(note));
                }
            result.push_back(std::move(part));
        };
        auto cursor = 0.0;
        for (const auto& cut : cuts)
        {
            emit(cursor, cut.getStart());
            cursor = std::max(cursor, cut.getEnd());
        }
        emit(cursor, source.durationSeconds);
    }
    if(ripple)
    {
        const auto cuts=nativeNoteRemovalRanges(original,ids);
        for(auto& part:result)
        {
            const auto local=part.startSeconds-original.startSeconds;
            double shift=0;
            for(const auto& cut:cuts)shift+=std::max(0.0,std::min(local,cut.getEnd())-cut.getStart());
            part.startSeconds-=shift;
        }
    }
    return result;
}
}
