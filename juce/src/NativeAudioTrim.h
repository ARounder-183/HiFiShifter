#pragma once
#include "NativeAudioLink.h"
#include "NativeTrimSource.h"

namespace hachi
{
// Move a word boundary on an unchanged recording clock. Author controls stay
// at their original absolute times; newly exposed pitch comes from the source.
inline void repartitionNativeTrimNote(NoteData& note,ClipData& source,double start,double end)
{
    const auto before=note;
    const auto added=before.startSeconds-start;
    note.startSeconds=start;note.durationSeconds=end-start;
    const auto shift=[&](auto& points){for(auto& p:points)p.timeSeconds+=added;};
    shift(note.pitchControlPoints);shift(note.amplitudeEnvelope);
    shift(note.diffSingerPitchReference);shift(note.diffSingerPitchOffset);
    for(auto& curve:note.utauFlagCurves)shift(curve.points);
    for(auto& t:note.sibilantMarkers)t+=added;
    if(note.vibratoEnabled||note.vibratoReferenceDurationSeconds>0)
    {
        if(note.vibratoReferenceDurationSeconds<=0)note.vibratoReferenceDurationSeconds=before.durationSeconds;
        note.vibratoTimeOffsetSeconds-=added;
    }
    if(start>=before.startSeconds&&end<=before.startSeconds+before.durationSeconds)
        note.nativeEnvelope=backend::slicedNativeEnvelope(before.nativeEnvelope,
            -added/before.durationSeconds,(end-before.startSeconds)/before.durationSeconds);
    else if(note.nativeEnvelope.mode!=0)
    {
        auto& s=note.nativeEnvelope.shape;
        s.startFraction=(added+s.startFraction*before.durationSeconds)/note.durationSeconds;
        s.endFraction=(added+s.endFraction*before.durationSeconds)/note.durationSeconds;
    }
    extendNativeTrimContour(note,before,source,added);
    if(!note.contour.empty())
    {
        const auto at=[&](double t)
        {
            const auto right=std::lower_bound(note.contour.begin(),note.contour.end(),t,[](const auto& p,double time){return p.timeSeconds<time;});
            if(right==note.contour.end())return note.contour.back();
            if(right==note.contour.begin()||std::abs(right->timeSeconds-t)<1.e-9)return *right;
            auto p=*std::prev(right);const auto u=float((t-p.timeSeconds)/(right->timeSeconds-p.timeSeconds));
            p.relativeCents+=(right->relativeCents-p.relativeCents)*u;p.withoutVibratoCents+=(right->withoutVibratoCents-p.withoutVibratoCents)*u;
            p.manualTargetCents+=(right->manualTargetCents-p.manualTargetCents)*u;
            p.hasManualTarget &= right->hasManualTarget;p.voiced &= right->voiced;return p;
        };
        auto a=at(0),b=at(note.durationSeconds);a.timeSeconds=0;b.timeSeconds=note.durationSeconds;
        std::erase_if(note.contour,[&](const auto& p){return p.timeSeconds<=0||p.timeSeconds>=note.durationSeconds;});
        note.contour.insert(note.contour.begin(),a);note.contour.push_back(b);
    }
    note.consonantSeconds=juce::jlimit(0.0,note.durationSeconds,before.consonantSeconds+added);
    bindNativeNoteSource(note,source);
}

// Crop or expose source audio without changing the retained source clock.
// Shared by the drag preview and the committed edit; the source file is untouched.
inline std::optional<NativeNoteMovePlan> planNativeNoteTrim(
    const ClipData& original, const juce::String& id, double requested, bool left,
    bool allowOverlap = false)
{
    if (!std::isfinite(requested)) return {};
    auto sources = expandedClipParts(original);
    auto owner = std::find_if(sources.begin(), sources.end(), [&](const auto& source) {
        return std::any_of(source.notes.begin(), source.notes.end(), [&](const auto& n) { return n.id == id; });
    });
    if (owner == sources.end() || !owner->sourceFile.existsAsFile()) return {};
    rememberNativeTrimSource(*owner);
    const auto note = std::find_if(owner->notes.begin(), owner->notes.end(), [&](const auto& n) { return n.id == id; });
    // Two words of one continuous source share a boundary, not two crop edges.
    // Work inside the existing source child so grouping/gain layers and the
    // recording clock remain intact, including a child in a merged region.
    const auto boundary=left?note->startSeconds:note->startSeconds+note->durationSeconds;
    NoteData* adjacent=nullptr;bool ambiguous=false;
    for(auto& other:owner->notes)if(other.id!=id
        &&std::abs((left?other.startSeconds+other.durationSeconds:other.startSeconds)-boundary)<1.e-7)
    {if(adjacent)ambiguous=true;adjacent=&other;}
    if(adjacent&&!ambiguous)
    {
        auto* first=left?adjacent:&*note;auto* second=left?&*note:adjacent;
        const auto begin=first->startSeconds,end=second->startSeconds+second->durationSeconds;
        for(const auto& other:owner->notes)if(other.id!=first->id&&other.id!=second->id
            &&std::min(end,other.startSeconds+other.durationSeconds)>std::max(begin,other.startSeconds)+1.e-7)ambiguous=true;
        if(!ambiguous)
        {
            const auto minimum=std::min(1.e-6,(end-begin)*.25);
            const auto next=juce::jlimit(begin+minimum,end-minimum,boundary+requested);
            const auto delta=next-boundary;if(std::abs(delta)<1.e-9)return NativeNoteMovePlan{original,0,true};
            TrackData envelopeTrack;envelopeTrack.pitchAlgorithm=PitchAlgorithm::world;envelopeTrack.clips={*owner};
            const auto envelopes=nativeSharedEnvelopes(envelopeTrack);
            repartitionNativeTrimNote(*first,*owner,begin,next);
            repartitionNativeTrimNote(*second,*owner,next,end);
            for(auto* value:{first,second})
                if(const auto shared=envelopes.find(value->id);shared!=envelopes.end())
                    value->amplitudeEnvelope=unscaledAmplitudeEnvelope(sliceNativeSharedEnvelope(*shared->second.points,
                        owner->startSeconds+value->startSeconds,owner->startSeconds+value->startSeconds+value->durationSeconds),
                        value->amplitudeEnvelopeBasePercent);
            auto result=original;rememberNativeTrimSources(result);
            const auto offset=owner->startSeconds-original.startSeconds;
            for(auto& target:result.notes)for(const auto* value:{first,second})if(target.id==value->id)
            {const auto partId=target.clipPartId;target=*value;target.startSeconds+=offset;target.clipPartId=partId;}
            return NativeNoteMovePlan{std::move(result),delta,true};
        }
    }
    if ((left && requested < 0) || (!left && requested > 0))
    {
        juce::AudioFormatManager formats;formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(owner->sourceFile));
        if(!reader || reader->sampleRate<=0)return {};
        const auto fileEnd=reader->lengthInSamples/reader->sampleRate;
        const auto oldStart=note->startSeconds,oldEnd=oldStart+note->durationSeconds;
        auto low=std::max(-owner->startSeconds,nativeClockExtrapolated(owner->nativeTrimClock,0,true));
        auto high=nativeClockExtrapolated(owner->nativeTrimClock,fileEnd,true);
        if(!allowOverlap)
            for(const auto& source:sources)for(const auto& other:source.notes)
                if(other.id!=id)
                {
                    const auto a=source.startSeconds+other.startSeconds-owner->startSeconds;
                    const auto b=a+other.durationSeconds;
                    if(b<=oldStart+1.e-7)low=std::max(low,b);
                    if(a>=oldEnd-1.e-7)high=std::min(high,a);
                }
        const auto delta=left?std::max(requested,std::min(0.0,low-oldStart))
            :std::min(requested,std::max(0.0,high-oldEnd));
        if(std::abs(delta)<1.e-9)return NativeNoteMovePlan{original,0};
        const auto before=*note;
        const auto start=left?oldStart+delta:oldStart,end=left?oldEnd:oldEnd+delta;
        const auto begin=std::min(0.0,start),finish=std::max(owner->durationSeconds,end);
        const auto audioStart=std::min(owner->audioStartSeconds,start);
        const auto audioEnd=std::max(owner->audioStartSeconds+owner->audioLength(),end);
        const auto sourceStart=juce::jlimit(0.0,fileEnd,nativeClockExtrapolated(owner->nativeTrimClock,audioStart));
        const auto sourceEnd=juce::jlimit(sourceStart,fileEnd,nativeClockExtrapolated(owner->nativeTrimClock,audioEnd));
        owner->sourceTimeMap.clear();owner->sourceTimeMap.push_back({audioStart-begin,0});
        for(const auto& p:owner->nativeTrimClock)
            if(p.targetSeconds>audioStart+1.e-8&&p.targetSeconds<audioEnd-1.e-8)
                owner->sourceTimeMap.push_back({p.targetSeconds-begin,p.sourceSeconds-sourceStart});
        owner->sourceTimeMap.push_back({audioEnd-begin,sourceEnd-sourceStart});
        owner->sourceOffsetSeconds=sourceStart;owner->sourceDurationSeconds=sourceEnd-sourceStart;
        owner->startSeconds+=begin;owner->durationSeconds=finish-begin;
        owner->audioStartSeconds=audioStart-begin;owner->audioDurationSeconds=audioEnd-audioStart;
        shiftClipGainEnvelopes(*owner,-begin);
        for(auto& p:owner->nativeTrimClock)p.targetSeconds-=begin;
        for(auto& n:owner->notes)n.startSeconds-=begin;
        const auto added=oldStart-start;
        note->startSeconds=start-begin;note->durationSeconds=end-start;
        const auto shift=[&](auto& points){for(auto& p:points)p.timeSeconds+=added;};
        shift(note->pitchControlPoints);shift(note->amplitudeEnvelope);
        shift(note->diffSingerPitchReference);shift(note->diffSingerPitchOffset);
        for(auto& curve:note->utauFlagCurves)shift(curve.points);
        for(auto& t:note->sibilantMarkers)t+=added;
        if(note->vibratoEnabled||note->vibratoReferenceDurationSeconds>0)
        {if(note->vibratoReferenceDurationSeconds<=0)note->vibratoReferenceDurationSeconds=before.durationSeconds;note->vibratoTimeOffsetSeconds-=added;}
        // Keep the existing envelope in place, hold its edge in exposed audio.
        if(note->nativeEnvelope.mode!=0)
        {
            auto& s=note->nativeEnvelope.shape;
            s.startFraction=(added+s.startFraction*before.durationSeconds)/note->durationSeconds;
            s.endFraction=(added+s.endFraction*before.durationSeconds)/note->durationSeconds;
        }
        extendNativeTrimContour(*note,before,*owner,added);
        note->utauAutoPitchTransition=false;
        if(left)note->connectedToPrevious=false;else note->connectedToNext=false;
        bindNativeNoteSource(*note,*owner);
        auto result=sources.size()==1?std::move(sources.front()):assembledLinkedAudio(sources);
        result.id=original.id;result.nativeAudioLinked=!result.parts.empty()&&original.nativeAudioLinked;
        result.showNoteHints=original.showNoteHints;result.showNormalDisplay=original.showNormalDisplay;
        return NativeNoteMovePlan{std::move(result),delta};
    }
    const auto room = std::max(0.0, note->durationSeconds - 1.e-6);
    const auto delta = left ? juce::jlimit(0.0, room, requested) : juce::jlimit(-room, 0.0, requested);
    if (std::abs(delta) < 1.e-9) return NativeNoteMovePlan{original, 0};
    const auto oldStart = note->startSeconds, oldEnd = oldStart + note->durationSeconds;
    auto cutBegin = left ? oldStart : oldEnd + delta;
    auto cutEnd = left ? oldStart + delta : oldEnd;
    const auto first = std::min_element(owner->notes.begin(), owner->notes.end(),
        [](const auto& a, const auto& b) { return a.startSeconds < b.startSeconds; });
    const auto last = std::max_element(owner->notes.begin(), owner->notes.end(),
        [](const auto& a, const auto& b) { return a.startSeconds + a.durationSeconds < b.startSeconds + b.durationSeconds; });
    if (left && first->id == id) cutBegin = 0;
    if (!left && last->id == id) cutEnd = owner->durationSeconds;
    for (const auto& other : owner->notes)
        if (other.id != id && std::min(cutEnd, other.startSeconds + other.durationSeconds)
            > std::max(cutBegin, other.startSeconds) + 1.e-9) return {};

    std::vector<ClipData> retained;
    for (auto source : sources)
    {
        if (source.id != owner->id) { retained.push_back(std::move(source)); continue; }
        const auto absoluteStart = source.startSeconds;
        source.startSeconds = 0;
        source.sourceTimeMap = nativeClipClock(source);
        const auto emit = [&](double begin, double end)
        {
            if (end - begin < 1.e-9) return;
            auto pieces = slicedClipParts({source}, begin, end, false, true);
            if (pieces.empty()) return;
            auto part = std::move(pieces.front());
            part.startSeconds = absoluteStart + begin;
            for (auto value : source.notes)
            {
                const auto a = std::max(begin, value.startSeconds);
                const auto b = std::min(end, value.startSeconds + value.durationSeconds);
                if (b - a < 1.e-9) continue;
                const auto offset = a - value.startSeconds;
                if (value.id == id)
                {
                    // Keep out-of-range support points so existing Bezier,
                    // pitch and gain interpolation is unchanged at the cut.
                    const auto shift = [&](auto& points) { for (auto& p : points) p.timeSeconds -= offset; };
                    if (!value.contour.empty())
                    {
                        const auto at = [&](double time)
                        {
                            const auto next=std::lower_bound(value.contour.begin(),value.contour.end(),time,
                                [](const auto& p,double t){return p.timeSeconds<t;});
                            if(next==value.contour.end())return value.contour.back();
                            if(next==value.contour.begin()||std::abs(next->timeSeconds-time)<1.e-9)return *next;
                            const auto& prev=*(next-1);auto p=prev;
                            const auto u=static_cast<float>((time-prev.timeSeconds)/(next->timeSeconds-prev.timeSeconds));
                            p.relativeCents+=(next->relativeCents-prev.relativeCents)*u;
                            p.withoutVibratoCents+=(next->withoutVibratoCents-prev.withoutVibratoCents)*u;
                            p.manualTargetCents+=(next->manualTargetCents-prev.manualTargetCents)*u;
                            p.hasManualTarget=prev.hasManualTarget&&next->hasManualTarget;
                            p.voiced=prev.voiced&&next->voiced;return p;
                        };
                        auto firstPoint=at(offset),lastPoint=at(offset+b-a);
                        std::erase_if(value.contour,[&](const auto& p){return p.timeSeconds<=offset||p.timeSeconds>=offset+b-a;});
                        shift(value.contour);firstPoint.timeSeconds=0;lastPoint.timeSeconds=b-a;
                        value.contour.insert(value.contour.begin(),firstPoint);value.contour.push_back(lastPoint);
                    }
                    shift(value.pitchControlPoints); shift(value.amplitudeEnvelope);
                    shift(value.diffSingerPitchReference); shift(value.diffSingerPitchOffset);
                    for (auto& curve : value.utauFlagCurves) shift(curve.points);
                    for (auto& marker : value.sibilantMarkers) marker -= offset;
                    if (value.vibratoEnabled || value.vibratoReferenceDurationSeconds > 0)
                    {
                        if (value.vibratoReferenceDurationSeconds <= 0)
                            value.vibratoReferenceDurationSeconds = value.durationSeconds;
                        value.vibratoTimeOffsetSeconds += offset;
                    }
                    value.nativeEnvelope=backend::slicedNativeEnvelope(value.nativeEnvelope,offset/value.durationSeconds,(offset+b-a)/value.durationSeconds);
                    value.startSeconds = a; value.durationSeconds = b - a;
                    value.consonantSeconds = juce::jlimit(0.0, value.durationSeconds, value.consonantSeconds - offset);
                    value.utauAutoPitchTransition = false;
                    if (left) value.connectedToPrevious = false; else value.connectedToNext = false;
                    bindNativeNoteSource(value, source, &source.sourceTimeMap);
                }
                if (std::abs(b - cutBegin) < 1.e-8) value.connectedToNext = false;
                if (std::abs(a - cutEnd) < 1.e-8) value.connectedToPrevious = false;
                value.startSeconds -= begin; value.clipPartId.clear();
                part.notes.push_back(std::move(value));
            }
            retained.push_back(std::move(part));
        };
        emit(0, cutBegin);
        emit(cutEnd, source.durationSeconds);
    }
    if (retained.empty()) return {};
    auto result = retained.size() == 1 ? std::move(retained.front()) : assembledLinkedAudio(retained);
    result.id = original.id;
    result.nativeAudioLinked = !result.parts.empty() && (original.nativeAudioLinked || original.parts.empty());
    result.showNoteHints = original.showNoteHints;
    result.showNormalDisplay = original.showNormalDisplay;
    return NativeNoteMovePlan{std::move(result), delta};
}
}
