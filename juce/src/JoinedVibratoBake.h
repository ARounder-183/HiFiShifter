#pragma once
#include "JoinedVibrato.h"

namespace hachi
{
// Fit ordinary editable cubics to the EXISTING evaluator. This is a conversion,
// never another runtime pitch source. Validate with the same cubic evaluator
// used by the editor and renderer, subdividing until below 0.005 cents.
inline bool fitBakedVibrato(const std::function<float(double)>& value,
    std::vector<double> breaks, std::vector<PitchCurveEditPoint>& result)
{
    std::sort(breaks.begin(),breaks.end());
    breaks.erase(std::unique(breaks.begin(),breaks.end(),[](double a,double b){return std::abs(a-b)<1.e-10;}),breaks.end());
    if(breaks.size()<2)return false;
    result={{breaks.front(),value(breaks.front()),PitchCurveShape::linear}};
    std::function<bool(double,double,int)> append=[&](double a,double b,int depth)
    {
        if(result.size()>100000)return false;
        auto left=result.back();left.timeSeconds=a;
        PitchCurveEditPoint right{b,value(b),PitchCurveShape::linear};
        const double y0=left.targetMidi,y3=right.targetMidi,dy=y3-y0;
        const double r1=27*value(a+(b-a)/3)-8*y0-y3;
        const double r2=27*value(a+2*(b-a)/3)-y0-8*y3;
        bool representable=true;
        if(std::abs(dy)>1.e-8)
        {
            right.shape=PitchCurveShape::customBezier;right.highPrecisionBezier=true;
            right.bezierX1=1.f/3;right.bezierX2=2.f/3;
            right.bezierY1=float(((2*r1-r2)/18-y0)/dy);
            right.bezierY2=float(((2*r2-r1)/18-y0)/dy);
            representable=std::isfinite(right.bezierY1)&&std::isfinite(right.bezierY2)
                &&right.bezierY1>=-2&&right.bezierY1<=3&&right.bezierY2>=-2&&right.bezierY2<=3;
        }
        double error=0;
        const std::vector<PitchCurveEditPoint> segment{left,right};
        for(int i=1;i<16;++i)
        {
            const auto t=a+(b-a)*i/16;
            error=std::max(error,double(std::abs(evaluatePitchCurve(segment,t)-value(t))));
        }
        if((!representable||error>5.e-5)&&b-a>1.e-8&&depth<24)
        {
            const auto middle=(a+b)*.5;
            return append(a,middle,depth+1)&&append(middle,b,depth+1);
        }
        if(!representable)right.shape=PitchCurveShape::linear;
        result.push_back(right);return true;
    };
    for(size_t i=1;i<breaks.size();++i)if(!append(breaks[i-1],breaks[i],0))return false;
    return true;
}

inline bool bakeJoinedVibrato(TrackData& track,std::vector<NativeConnection>& connections,
    const JoinedVibrato& group)
{
    const auto before=track;
    const auto lines=sharedPitchLines(before,connections);
    struct Member { const NoteData* note; double start,end; std::vector<PitchCurveEditPoint> own; };
    std::vector<Member> members;
    for(const auto& clip:before.clips)for(const auto& note:clip.notes)
        if(std::find(group.notes.begin(),group.notes.end(),note.id)!=group.notes.end())
            members.push_back({&note,clip.startSeconds+note.startSeconds,clip.startSeconds+note.startSeconds+note.durationSeconds,ownPitchPoints(note)});
    std::sort(members.begin(),members.end(),[](const auto& a,const auto& b){return a.start<b.start;});
    if(members.empty()||std::none_of(members.begin(),members.end(),[](const auto& m){return m.note->vibratoEnabled;}))return false;
    const auto swingAt=[&](double t)
    {
        for(const auto& m:members)if(t>=m.start-1.e-9&&t<=m.end+1.e-9)
            return float(vibratoCentsAt(*m.note,std::clamp(t-m.start,0.,m.note->durationSeconds))/100.);
        return 0.f;
    };
    std::map<juce::String,std::vector<PitchCurveEditPoint>> baked;
    std::optional<PitchCurveEditPoint> seam;
    for(const auto& m:members)
    {
        const auto* shared=lines.memberFor(m.note->id);
        const auto base=[&](double t){return shared&&shared->renderSharedCurve
            ?shared->line->midiAt(t):evaluatePitchCurve(m.own,t-m.start);};
        const auto value=[&](double t){return base(t)+float(vibratoCentsAt(*m.note,t-m.start)/100.);};
        std::vector<double> breaks{m.start,m.end};
        const auto step=std::clamp(m.note->vibratoCycleMs/8000.,.0005,.02);
        for(double t=m.start+step;t<m.end;t+=step)breaks.push_back(t);
        if(shared&&shared->renderSharedCurve)
        {const auto corners=shared->line->cornersBetween(m.start,m.end);breaks.insert(breaks.end(),corners.begin(),corners.end());}
        else for(const auto& p:m.own)if(p.timeSeconds>0&&p.timeSeconds<m.note->durationSeconds)breaks.push_back(m.start+p.timeSeconds);
        const auto duration=m.note->vibratoReferenceDurationSeconds>0?m.note->vibratoReferenceDurationSeconds:m.note->durationSeconds;
        const auto end=duration*std::clamp(m.note->vibratoEndPercent,0.,100.)/100.;
        const auto start=std::max(0.,end-duration*std::clamp(m.note->vibratoLengthPercent,0.,100.)/100.);
        for(double t:{start,end,start+(end-start)*std::clamp(m.note->vibratoFadeInPercent,0.,100.)/100.,
            end-(end-start)*std::clamp(m.note->vibratoFadeOutPercent,0.,100.)/100.})
        {t+=m.start-m.note->vibratoTimeOffsetSeconds;if(t>m.start&&t<m.end)breaks.push_back(t);}
        auto& points=baked[m.note->id];if(!fitBakedVibrato(value,std::move(breaks),points))return false;
        // The right member wins duplicate seam times in the shared point line;
        // retain the incoming cubic of the left member at that shared endpoint.
        if(seam&&std::abs(seam->timeSeconds-points.front().timeSeconds)<1.e-7
            &&std::abs(seam->targetMidi-points.front().targetMidi)<1.e-4)points.front()=*seam;
        seam=points.back();
        for(auto& p:points)p.timeSeconds-=m.start;
    }
    // Explicit connection curves override the body curve. Fold the same swing
    // into those windows as well, so baking does not discard their contribution.
    for(const auto& resolved:resolvedConnectionPitchCurves(before,connections))
    {
        if(resolved.issue.isNotEmpty()||resolved.points.size()<2)continue;
        const auto from=resolved.points.front().timeSeconds,to=resolved.points.back().timeSeconds;
        if(to<=members.front().start||from>=members.back().end)continue;
        std::vector<double> breaks{from,to};
        for(const auto& p:resolved.points)breaks.push_back(p.timeSeconds);
        for(double t=from+.005;t<to;t+=.005)breaks.push_back(t);
        std::vector<PitchCurveEditPoint> points;
        if(!fitBakedVibrato([&](double t){return evaluatePitchCurve(resolved.points,t)+swingAt(t);},breaks,points))return false;
        if(points.size()>2048)return false;
        for(auto& connection:connections)if(connection.id==resolved.connectionId)
        {
            connection.pitchCurve=std::move(points);connection.pitchCurveEndpointRelative=false;
            break;
        }
    }
    for(auto& clip:track.clips)for(auto& note:clip.notes)
        if(const auto found=baked.find(note.id);found!=baked.end())
        {
            note.pitchControlPoints=found->second;
            for(auto& p:note.contour){p.hasManualTarget=true;p.manualTargetCents=(evaluatePitchCurve(note.pitchControlPoints,p.timeSeconds)-note.midiNote)*100;}
            note.nativePitchHandlesPlaced=trackShowsAllNativeRegions(track)||note.nativePitchHandlesPlaced;
            note.vibratoEnabled=false;note.vibratoRealLine=false;
            note.vibratoReferenceDurationSeconds=0;note.vibratoTimeOffsetSeconds=0;
        }
    return true;
}
}
