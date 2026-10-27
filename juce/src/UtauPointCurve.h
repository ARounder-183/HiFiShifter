#pragma once
#include "ProjectModel.h"
#include "backend/UtauRenderer.h"
#include <map>
#include <memory>

namespace hachi
{
// Immutable authored geometry: anchors first, then the connecting line.
struct UtauPointCurve
{
    std::vector<PitchCurveEditPoint> points;
    std::vector<std::vector<PitchCurveEditPoint>> segments;
    float midiAt(double time) const
    {
        const auto right=std::upper_bound(points.begin(),points.end(),time,[](double t,const auto& p){return t<p.timeSeconds;});
        if(right==points.begin())return points.front().targetMidi;
        if(right==points.end())return points.back().targetMidi;
        return evaluatePitchCurve(segments[static_cast<size_t>(right-points.begin()-1)],time);
    }
    void prepare()
    {
        std::stable_sort(points.begin(),points.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
        // Existing monotone interpolation only needs the two adjacent secants.
        for(size_t i=0;i+1<points.size();++i)
            segments.emplace_back(points.begin()+(i?i-1:0),points.begin()+std::min(points.size(),i+3));
    }
};

inline std::map<juce::String,std::shared_ptr<const UtauPointCurve>> utauPointCurves(const TrackData& track)
{
    struct Entry {const NoteData* note;double start;};
    std::vector<Entry> entries;
    for(const auto& c:track.clips)for(const auto& n:c.notes)entries.push_back({&n,c.startSeconds+n.startSeconds});
    std::stable_sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.start<b.start;});
    const auto adjacent=[](const Entry& a,const Entry& b)
    {return a.note->utauAutoPitchTransition&&b.note->utauAutoPitchTransition
        &&!a.note->nativeUnpitched&&!b.note->nativeUnpitched
        &&!backend::isRestLyric(a.note->label)&&!backend::isRestLyric(b.note->label)
        &&std::abs(a.start+a.note->durationSeconds-b.start)<.002;};
    const auto inset=[](double a,double b){return std::max(.002,std::min({.020,std::max(.01,a)*.2,std::max(.01,b)*.2}));};
    std::map<juce::String,std::shared_ptr<const UtauPointCurve>> result;
    for(size_t first=0;first<entries.size();)
    {
        auto last=first+1;while(last<entries.size()&&adjacent(entries[last-1],entries[last]))++last;
        const auto authored=std::any_of(entries.begin()+first,entries.begin()+last,[](const auto& e){return !e.note->pitchControlPoints.empty();});
        if(last-first>1&&authored)
        {
            auto line=std::make_shared<UtauPointCurve>();
            for(auto i=first;i<last;++i)
            {
                const auto& e=entries[i];auto points=ownPitchPoints(*e.note);
                if(e.note->pitchControlPoints.empty()&&points.size()>=2)
                {
                    if(i>first)
                    {
                        const auto from=inset(entries[i-1].note->durationSeconds,e.note->durationSeconds);
                        const auto begin=points.front().timeSeconds;
                        std::erase_if(points,[&](const auto& p){return p.timeSeconds>begin&&p.timeSeconds<=from;});
                        points.front().timeSeconds=from;
                    }
                    if(i+1<last)
                    {
                        const auto to=e.note->durationSeconds-inset(e.note->durationSeconds,entries[i+1].note->durationSeconds);
                        const auto end=points.back().timeSeconds;
                        std::erase_if(points,[&](const auto& p){return p.timeSeconds>=to&&p.timeSeconds<end;});
                        points.back().timeSeconds=to;
                    }
                    points.front().shape=PitchCurveShape::natural;
                }
                for(auto p:points){p.timeSeconds+=e.start;line->points.push_back(p);}
                result[e.note->id]=line;
            }
            line->prepare();
        }
        first=last;
    }
    return result;
}
}
