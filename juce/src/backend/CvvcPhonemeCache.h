#pragma once
#include "UtauRenderer.h"

namespace hachi::backend
{
struct CvvcPhonemeCache
{
    std::uint64_t layout = 0, context = 0;
    std::vector<UtauPhonemeSpan> spans;
};
inline juce::String cvvcPhonemeCacheText(const CvvcPhonemeCache& cache)
{
    auto root=std::make_unique<juce::DynamicObject>();
    root->setProperty("version",1);
    root->setProperty("layout",juce::String::toHexString(static_cast<juce::int64>(cache.layout)));
    root->setProperty("context",juce::String::toHexString(static_cast<juce::int64>(cache.context)));
    juce::Array<juce::var> rows;
    for(const auto& p:cache.spans)
    {
        auto row=std::make_unique<juce::DynamicObject>();
        row->setProperty("token",p.token);row->setProperty("kind",p.kind);
        row->setProperty("from",p.startSeconds);row->setProperty("to",p.endSeconds);
        row->setProperty("part",p.envelopePart);row->setProperty("lead",p.leadInSeconds);
        const auto& mix=p.mixEnvelope;
        if(mix.available())
        {
            row->setProperty("rate",mix.sampleRate);row->setProperty("mixStart",mix.startSeconds);row->setProperty("samples",mix.samples);
            row->setProperty("standardEnvelope",mix.replacesStandardEnvelope);
            juce::Array<juce::var> ramps;
            for(const auto& r:mix.ramps)ramps.add(juce::var(juce::Array<juce::var>{r.start,r.length,r.rising,r.equalPower,r.phaseOffset,r.activeSamples,r.crossfade,r.smoothCurve}));
            row->setProperty("ramps",ramps);
        }
        rows.add(row.release());
    }
    root->setProperty("spans",rows);
    return juce::JSON::toString(juce::var(root.release()),true,17);
}
inline CvvcPhonemeCache readCvvcPhonemeCache(const juce::String& text)
{
    CvvcPhonemeCache result;
    if(text.isEmpty()||text.length()>131072)return result;
    const auto root=juce::JSON::parse(text);
    const auto* rows=root["spans"].getArray();
    if((int)root["version"]!=1||!rows||rows->isEmpty()||rows->size()>64)return result;
    result.layout=static_cast<std::uint64_t>(root["layout"].toString().getHexValue64());
    result.context=static_cast<std::uint64_t>(root["context"].toString().getHexValue64());
    double previous=-std::numeric_limits<double>::infinity();
    for(const auto& row:*rows)
    {
        UtauPhonemeSpan p {row["token"].toString(),row["kind"].toString(),(double)row["from"],(double)row["to"],(int)row["part"],(double)row["lead"]};
        if(p.token.length()>512||!std::isfinite(p.startSeconds)||!std::isfinite(p.endSeconds)||!std::isfinite(p.leadInSeconds)
            ||p.endSeconds<=p.startSeconds||p.startSeconds<previous||p.leadInSeconds<0||p.envelopePart<1||p.envelopePart>3)return {};
        previous=p.startSeconds;
        auto& mix=p.mixEnvelope;mix.sampleRate=(double)row["rate"];mix.startSeconds=(double)row["mixStart"];mix.samples=(int)row["samples"];
        mix.replacesStandardEnvelope=(bool)row["standardEnvelope"];
        if(!std::isfinite(mix.sampleRate)||!std::isfinite(mix.startSeconds)||mix.sampleRate<0||mix.sampleRate>768000||mix.samples<0)return {};
        if(const auto* ramps=row["ramps"].getArray())
        {
            if(ramps->size()>128)return {};
            for(const auto& r:*ramps)
            {
                if(!r.isArray()||(r.size()!=7&&r.size()!=8))return {};
                UtauMixRamp ramp {(int)r[0],(int)r[1],(bool)r[2],(bool)r[3],(double)r[4],(int)r[5],(bool)r[6]};
                if(r.size()==8)ramp.smoothCurve=(bool)r[7];
                if(ramp.length<1||ramp.activeSamples< -1||!std::isfinite(ramp.phaseOffset))return {};
                mix.ramps.push_back(ramp);
            }
        }
        result.spans.push_back(std::move(p));
    }
    return result;
}
}
