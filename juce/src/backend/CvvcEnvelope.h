#pragma once
#include <juce_core/juce_core.h>
#include "AmplitudeEnvelopeCurve.h"
#include "NativeEnvelope.h"
#include <algorithm>
#include <map>
#include <vector>

namespace hachi::backend
{
// Positions are fractions of one generated phoneme's sounding span, so an
// edit follows timing/pitch changes without becoming another score note.
struct CvvcEnvelopePoint
{
    double position = 0;
    float value = 0;
    bool linear = true;
    std::optional<EnvelopeBezier> bezier;
    bool operator==(const CvvcEnvelopePoint&) const = default;
};
using CvvcEnvelopeCurve = std::vector<CvvcEnvelopePoint>;
using CvvcEnvelopeEdits = std::map<int, std::map<juce::String, CvvcEnvelopeCurve>>;
// One independent standard envelope per generated phoneme. The values are
// linear amplitude (not a straight line in dB). Never store defaults as edits.
inline CvvcEnvelopeCurve standardCvvcEnvelope(double soundingSeconds)
{
    const auto duration=std::max(1.e-6,soundingSeconds);
    auto attack=.005/duration,release=1.0-.035/duration;
    if(release<=attack+.005/duration){attack=1.0/3;release=2.0/3;}
    return {{0,-60,true},{attack,0,true},{release,0,true},{1,-60,true}};
}
inline bool isStandardCvvcEnvelope(const CvvcEnvelopeCurve& points,double seconds)
{
    const auto standard=standardCvvcEnvelope(seconds);
    if(points.size()!=standard.size())return false;
    for(size_t i=0;i<points.size();++i)
        if(std::abs(points[i].position-standard[i].position)>1.e-6
            ||std::abs(points[i].value-standard[i].value)>.001f||!points[i].linear||points[i].bezier)return false;
    return true;
}
inline bool hasCvvcFlagCurve(const CvvcEnvelopeEdits& edits, const juce::String& key = {})
{
    for (const auto& [part, curves] : edits)
    {
        if(key.isEmpty()) {for(const auto& [flag,points]:curves)if(flag.isNotEmpty()&&!points.empty())return true;}
        else if(const auto it=curves.find(key);it!=curves.end()) {if(!it->second.empty())return true;}
        else if(key.startsWith("HIFI:"))if(const auto it=curves.find(key.substring(5));it!=curves.end()&&!it->second.empty())return true;
    }
    return false;
}
// Stable roles: onset VC, main CV/VCV, release. Empty key denotes loudness.
inline int cvvcEnvelopePart(const juce::String& kind)
{ return kind == "VC" ? 1 : kind == "release" ? 3 : (kind == "CV" || kind == "VCV") ? 2 : 0; }
inline float cvvcEnvelopeValue(const CvvcEnvelopeCurve& curve, double position, bool amplitude)
{
    if (curve.empty()) return 0;
    if (position <= curve.front().position) return curve.front().value;
    for (size_t i=1;i<curve.size();++i) if(position<=curve[i].position)
    {
        const auto& a=curve[i-1];const auto& b=curve[i];
        const auto u=static_cast<float>(juce::jlimit(0.0,1.0,(position-a.position)/std::max(1.e-9,b.position-a.position)));
        return amplitude ? envelopeDbBetween(a.value,b.value,u,a.linear,a.bezier) : a.value+(b.value-a.value)*u;
    }
    return curve.back().value;
}
inline juce::var cvvcEnvelopesToVar(const CvvcEnvelopeEdits& edits)
{
    juce::Array<juce::var> list;
    for (const auto& [part,curves]:edits) for(const auto& [key,points]:curves)
    {
        auto* entry=new juce::DynamicObject(); entry->setProperty("part",part);entry->setProperty("flag",key);
        juce::Array<juce::var> values;
        for(const auto& p:points)
        {
            juce::Array<juce::var> row{p.position,p.value,p.linear};
            if(p.bezier)row.add(envelopeBezierToVar(*p.bezier));
            values.add(juce::var(row));
        }
        entry->setProperty("points",juce::var(values));list.add(juce::var(entry));
    }
    return juce::var(list);
}
inline CvvcEnvelopeEdits cvvcEnvelopesFromVar(const juce::var& data)
{
    CvvcEnvelopeEdits edits;
    if (const auto* list=data.getArray()) for(const auto& entry:*list)
    {
        const int part=entry["part"];if(part<1||part>3)continue;
        const auto key=entry["flag"].toString();CvvcEnvelopeCurve points;
        if(const auto* values=entry["points"].getArray())for(const auto& value:*values)
            if(const auto* fields=value.getArray();fields&&fields->size()>=2)
            {
                const double t=fields->getReference(0);const float v=fields->getReference(1);
                if(!std::isfinite(t)||!std::isfinite(v))continue;
                points.push_back({juce::jlimit(0.0,1.0,t),
                    juce::jlimit(key.isEmpty()?-60.f:-100000.f,key.isEmpty()?12.f:100000.f,v),
                    fields->size()<3||static_cast<bool>(fields->getReference(2))});
                if(!points.empty()&&fields->size()>3)
                {EnvelopeBezier b;if(envelopeBezierFromVar(fields->getReference(3),b))points.back().bezier=b;}
            }
        std::stable_sort(points.begin(),points.end(),[](const auto& a,const auto& b){return a.position<b.position;});
        points.erase(std::unique(points.begin(),points.end(),[](const auto& a,const auto& b){return std::abs(a.position-b.position)<1.e-8;}),points.end());
        if(points.size()>=2 || (key.isNotEmpty() && entry["points"].isArray()
            && entry["points"].getArray()->isEmpty()))edits[part][key]=std::move(points);
    }
    return edits;
}

// The advanced panel and the lane edit the same sparse per-phoneme curve.
inline NativeEnvelopeSettings cvvcEnvelopeSettings(const CvvcEnvelopeCurve& points)
{
    NativeEnvelopeSettings out;out.mode=2;auto& s=out.shape;
    if(points.size()<2)return out;
    s.startFraction=points.front().position;s.endFraction=points.back().position;
    s.startGain=envelopeGainFromDb(points.front().value);s.endGain=envelopeGainFromDb(points.back().value);
    s.firstSegmentCurved=points.front().bezier.has_value();s.firstBezier=points.front().bezier.value_or(EnvelopeBezier{});
    for(size_t i=1;i+1<points.size();++i)s.knots.push_back({
        (points[i].position-s.startFraction)/(s.endFraction-s.startFraction),envelopeGainFromDb(points[i].value),
        points[i].bezier.has_value(),points[i].bezier.value_or(EnvelopeBezier{})});
    return out;
}
inline CvvcEnvelopeCurve cvvcEnvelopeFromSettings(NativeEnvelopeSettings settings)
{
    if(settings.mode==0)return {{0,0,true},{1,0,true}};
    const auto& s=settings.shape;CvvcEnvelopeCurve result;
    const auto add=[&](double t,double gain,std::optional<EnvelopeBezier> b={}){
        if(!result.empty()&&std::abs(result.back().position-t)<1.e-9)result.back()={t,envelopeDbFromGain(static_cast<float>(gain)),true,b};
        else result.push_back({t,envelopeDbFromGain(static_cast<float>(gain)),true,b});};
    add(0,s.startGain);
    // Legacy power curves are not cubic Beziers. Bound their approximation;
    // ordinary linear/Bezier edits below retain their exact sparse controls.
    if(settings.mode==2&&!s.mixed&&!s.customCurve&&s.curvePower!=1)
    {
        for(int i=0;i<=96;++i){const auto t=s.startFraction+(s.endFraction-s.startFraction)*i/96.0;add(t,tailFadeGain(2,t,0,1,s));}
        add(1,s.endGain);return result;
    }
    add(s.startFraction,s.startGain,settings.mode==2?(s.mixed?(s.firstSegmentCurved?std::optional{s.firstBezier}:std::nullopt)
        :std::optional{s.customCurve?EnvelopeBezier{s.control1Time,s.control1Progress,s.control2Time,s.control2Progress}:EnvelopeBezier{}}):std::nullopt);
    if(settings.mode==2&&s.mixed)for(const auto& k:s.knots)
        add(s.startFraction+(s.endFraction-s.startFraction)*k.time,k.gain,k.curvedToNext?std::optional{k.bezier}:std::nullopt);
    add(s.endFraction,s.endGain);add(1,s.endGain);return result;
}
}
