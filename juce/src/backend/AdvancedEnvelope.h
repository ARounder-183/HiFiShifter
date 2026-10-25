#pragma once
#include "AmplitudeEnvelopeCurve.h"
#include "TailFadeSettings.h"
#include <algorithm>
#include <vector>
namespace hachi::backend
{
// Mode 1 is straight in amplitude; mode 2 supports legacy smoothstep and Bezier.
template<class Settings> double mixedEnvelopeGain(double progress,const Settings& s)
{
    if(progress<=0)return s.startGain;if(progress>=1)return s.endGain;
    double time=0,gain=s.startGain;bool curved=s.firstSegmentCurved;auto bezier=s.firstBezier;
    for(size_t i=0;i<=s.knots.size();++i)
    {
        const auto nextTime=i<s.knots.size()?s.knots[i].time:1.0;
        const auto nextGain=i<s.knots.size()?s.knots[i].gain:s.endGain;
        if(progress<=nextTime)
        {
            auto u=(progress-time)/(nextTime-time);if(curved)u=tailFadeBezierProgress(u,bezier);
            return gain+(nextGain-gain)*u;
        }
        time=nextTime;gain=nextGain;curved=s.knots[i].curvedToNext;bezier=s.knots[i].bezier;
    }
    return s.endGain;
}
inline float tailFadeGain(int mode,double seconds,double start,double end,const TailFadeSettings& settings = {})
{
    if(mode<1||mode>2||end-start<=1.e-9||!settings.valid())return 1.0f;
    const auto length=end-start;
    const auto first=start+length*settings.startFraction,last=start+length*settings.endFraction;
    if(seconds<=first)return static_cast<float>(settings.startGain);
    if(seconds>=last)return static_cast<float>(settings.endGain);
    const auto progress=(seconds-first)/(last-first);
    if(mode==2&&settings.mixed)return static_cast<float>(mixedEnvelopeGain(progress,settings));
    const auto u=mode==2?std::pow(progress,settings.curvePower):progress;
    const auto shaped=mode==2?(settings.customCurve?tailFadeBezierProgress(progress,settings):u*u*(3.0-2.0*u)):u;
    return static_cast<float>(settings.startGain+(settings.endGain-settings.startGain)*shaped);
}
// Replace the original attack up to its chosen end, anchored at the original
// end level. Afterwards the end multiplier follows the original body envelope.
template<class GainAt> float headEnvelopeGain(double seconds,double start,double end,
                                              const HeadEnvelopeSettings& s,GainAt baseGainAt)
{
    if(s.mode==0||end-start<=1.e-9||!s.valid())return baseGainAt(seconds);
    const auto first=start+(end-start)*s.startFraction,last=start+(end-start)*s.endFraction;
    const auto progress=std::clamp((seconds-first)/(last-first),0.0,1.0);
    if(s.mode==2&&s.mixed)return baseGainAt(std::max(seconds,last))*static_cast<float>(mixedEnvelopeGain(progress,s));
    const auto u=s.mode==2?std::pow(progress,s.curvePower):progress;
    const auto shaped=s.mode==2?(s.customCurve?tailFadeBezierProgress(progress,s):u*u*(3-2*u)):u;
    return baseGainAt(std::max(seconds,last))*static_cast<float>(s.startGain+(s.endGain-s.startGain)*shaped);
}
// Replace the old ending from the chosen fade start. Sampling the original
// envelope at every point would multiply two tails and bend a linear fade.
// The level at the start anchors the entire new ending; the original stays stored.
template<class GainAt> float tailFadeEnvelopeGain(int mode,double seconds,double start,double end,
                                                const TailFadeSettings& settings,GainAt baseGainAt)
{
    if(mode<1||mode>2||end-start<=1.e-9||!settings.valid())return baseGainAt(seconds);
    const auto first=start+(end-start)*settings.startFraction;
    return baseGainAt(std::min(seconds,first))*tailFadeGain(mode,seconds,start,end,settings);
}
template<class GainAt> float advancedEnvelopeGain(int tailMode,double seconds,double tailStart,double tailEnd,
                                                 double headStart,double headEnd,const TailFadeSettings& settings,GainAt baseGainAt)
{
    return tailFadeEnvelopeGain(tailMode,seconds,tailStart,tailEnd,settings,[&](double t){
        return headEnvelopeGain(t,headStart,headEnd,settings.head,baseGainAt);});
}
// Only the picture is sampled. Editing handles and the stored envelope stay sparse.
template<class Point> std::vector<Point> tailFadePicture(const std::vector<Point>& base,int mode,double start,double end,const TailFadeSettings& settings = {},double headStart=0,double headEnd=0)
{
    if(base.size()<2||!settings.valid())return base;
    if((mode==0||end-start<=1.e-9)&&(settings.head.mode==0||headEnd-headStart<=1.e-9))return base;
    const auto tailStart=start,tailEnd=end;
    start=tailStart+(tailEnd-tailStart)*settings.startFraction;
    end=tailStart+(tailEnd-tailStart)*settings.endFraction;
    std::vector<double> times;for(const auto& p:base)times.push_back(p.timeSeconds);
    times.push_back(start);times.push_back(end);
    for(int i=1;i<96;++i)times.push_back(start+(end-start)*i/96.0);
    for(size_t i=1;i<base.size();++i)
    {
        const auto a=std::max(start,base[i-1].timeSeconds),b=std::min(end,base[i].timeSeconds);
        if(b>a)for(int j=1;j<24;++j)times.push_back(a+(b-a)*j/24.0);
    }
    const auto headFirst=headStart+(headEnd-headStart)*settings.head.startFraction;
    const auto headLast=headStart+(headEnd-headStart)*settings.head.endFraction;
    const auto mixedTimes=[&](const auto& s,double first,double last)
    {
        double previous=0;
        for(size_t i=0;i<=s.knots.size();++i)
        {
            const auto next=i<s.knots.size()?s.knots[i].time:1.0;
            for(int j=0;j<=32;++j)times.push_back(first+(last-first)*(previous+(next-previous)*j/32.0));
            previous=next;
        }
    };
    if(mode==2&&settings.mixed)mixedTimes(settings,start,end);
    if(settings.head.mode==2&&settings.head.mixed)mixedTimes(settings.head,headFirst,headLast);
    if(settings.head.mode!=0&&headEnd-headStart>1.e-9)
        for(int i=0;i<=96;++i)times.push_back(headFirst+(headLast-headFirst)*i/96.0);
    std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end(),[](double a,double b){return std::abs(a-b)<1.e-9;}),times.end());
    const auto baseGainAt=[&](double t)
    {
        const auto after=std::upper_bound(base.begin(),base.end(),t,[](double at,const Point& p){return at<p.timeSeconds;});
        float db=base.front().gainDb;
        if(after==base.end())db=base.back().gainDb;
        else if(after!=base.begin())
        {const auto& left=*std::prev(after);const auto& right=*after;db=envelopeDbBetween(left.gainDb,right.gainDb,static_cast<float>((t-left.timeSeconds)/std::max(1.e-9,right.timeSeconds-left.timeSeconds)),left.linearToNext);}
        return envelopeGainFromDb(db);
    };
    std::vector<Point> result;result.reserve(times.size());
    for(auto t:times)
    {
        const auto gain=advancedEnvelopeGain(mode,t,tailStart,tailEnd,headStart,headEnd,settings,baseGainAt);
        const auto after=std::upper_bound(base.begin(),base.end(),t,[](double at,const Point& p){return at<p.timeSeconds;});
        const auto editedTail=mode!=0&&tailEnd>tailStart&&t>=start;
        const auto editedHead=settings.head.mode!=0&&headEnd>headStart&&t<headLast;
        const auto linear=!editedTail&&!editedHead&&after!=base.begin()&&after!=base.end()?std::prev(after)->linearToNext:true;
        result.push_back({t,envelopeDbFromGain(gain),linear});
    }
    return result;
}
}
