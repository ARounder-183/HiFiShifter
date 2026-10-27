#pragma once
#include "UtauRenderer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

namespace hachi::backend
{
// An OTO head or crossfade tail can sound over another score note.
// The rendered note is not necessarily the owner of pitch at that instant.
// Keep one immutable host-projected timeline per touching, non-rest run; both
// sides of every phoneme crossfade query it, including OTO heads and tails.
struct UtauPitchTimeline
{
    struct Entry
    {
        double start;
        float midi;
        std::vector<UtauPitchPoint> points;
        std::function<float(double)> evaluate;
    };
    std::vector<Entry> entries;

    float midiAt(double absolute) const
    {
        auto it=std::upper_bound(entries.begin(),entries.end(),absolute,
            [](double t,const Entry& e){return t<e.start;});
        if(it!=entries.begin())--it;
        const auto local=absolute-it->start;
        if(it->evaluate)return it->midi+it->evaluate(local)/100.f;
        if(it->points.empty())return it->midi;
        const auto right=std::lower_bound(it->points.begin(),it->points.end(),local,
            [](const auto& p,double t){return p.timeSeconds<t;});
        if(right==it->points.begin())return it->midi+right->cents/100.f;
        if(right==it->points.end())return it->midi+it->points.back().cents/100.f;
        const auto& left=*std::prev(right);
        const auto u=static_cast<float>((local-left.timeSeconds)/(right->timeSeconds-left.timeSeconds));
        return it->midi+(left.cents+(right->cents-left.cents)*u)/100.f;
    }

    void bind(UtauNoteRenderSpec& note,double absoluteStart,
              const std::shared_ptr<const UtauPitchTimeline>& self) const
    {
        note.timelinePitchCents=[self,absoluteStart,midi=note.midiNote](double t)
        {return (self->midiAt(absoluteStart+t)-midi)*100.f;};
    }

    void apply(UtauNoteRenderSpec& piece,double lead,
               const std::shared_ptr<const UtauPitchTimeline>& self) const
    {
        bind(piece,piece.startSeconds,self);
        // Samples describe the piece's audible head/body. Renderers use the
        // evaluator for any extra crossfade tail, without clamping to a lyric.
        const auto from=piece.startSeconds-lead,to=piece.startSeconds+piece.durationSeconds;
        std::vector<double> times{from,to};
        auto it=std::upper_bound(entries.begin(),entries.end(),from,
            [](double t,const Entry& e){return t<e.start;});
        if(it!=entries.begin())--it;
        for(;it!=entries.end()&&it->start<=to;++it)
        {
            const auto end=std::next(it)==entries.end()?to:std::min(to,std::next(it)->start);
            const auto begin=it==entries.begin()?from:std::max(from,it->start);
            if(it->start>from)
            {times.push_back(std::nextafter(it->start,-std::numeric_limits<double>::infinity()));times.push_back(it->start);}
            auto p=std::lower_bound(it->points.begin(),it->points.end(),begin-it->start,
                [](const auto& point,double t){return point.timeSeconds<t;});
            for(;p!=it->points.end()&&it->start+p->timeSeconds<end;++p)times.push_back(it->start+p->timeSeconds);
            if(it->evaluate)for(auto t=begin;t<end;t+=.005)times.push_back(t);
        }
        std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
        piece.pitchCurve.clear();piece.pitchCurve.reserve(times.size());
        for(const auto t:times)piece.pitchCurve.push_back({t-piece.startSeconds,(midiAt(t)-piece.midiNote)*100.f});
    }
};

inline std::vector<std::shared_ptr<const UtauPitchTimeline>> makeUtauPitchTimelines(
    const std::vector<UtauNoteRenderSpec>& notes)
{
    std::vector<std::size_t> order(notes.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return notes[a].startSeconds<notes[b].startSeconds;});
    const auto adjacent=[&](std::size_t a,std::size_t b)
    {
        return !isRestLyric(notes[a].alias)&&!isRestLyric(notes[b].alias)
            &&notes[a].alias.trim().isNotEmpty()&&notes[b].alias.trim().isNotEmpty()
            &&std::abs(notes[a].startSeconds+notes[a].durationSeconds-notes[b].startSeconds)<.002;
    };
    std::vector<std::shared_ptr<const UtauPitchTimeline>> result(notes.size());
    for(std::size_t first=0;first<order.size();)
    {
        auto last=first+1;while(last<order.size()&&adjacent(order[last-1],order[last]))++last;
        auto timeline=std::make_shared<UtauPitchTimeline>();timeline->entries.reserve(last-first);
        for(auto i=first;i<last;++i)
        {
            const auto& n=notes[order[i]];
            timeline->entries.push_back({n.startSeconds,n.midiNote,n.pitchCurve,n.timelinePitchCents});
            result[order[i]]=timeline;
        }
        first=last;
    }
    return result;
}
}
