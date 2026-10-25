#pragma once
#include "../backend/ChineseCvvcPhonemizer.h"
#include "CvvcLegacyEnvelopeSmoke.h"

namespace hachi
{
inline bool hifisamplerEnvelopeSmoke(const juce::File& output, const juce::File& projectFile,
    const juce::File& models, const juce::File& resampler)
{
    using namespace backend;
    output.createDirectory(); bool ok=true; juce::Array<juce::var> checks;
    const auto check=[&](const juce::String& name,bool passed)
    {
        ok &= passed; auto row=std::make_unique<juce::DynamicObject>();
        row->setProperty("name",name);row->setProperty("passed",passed);checks.add(row.release());
        std::cout<<name<<'='<<passed<<std::endl;
    };
    constexpr double rate=48000;
    const auto flat=[](double length,float value)
    { juce::AudioBuffer<float> audio(1,static_cast<int>(length*rate));audio.clear();for(int i=0;i<audio.getNumSamples();++i)audio.setSample(0,i,value);return audio; };
    const auto at=[](const auto& audio,double time){return audio.getSample(0,static_cast<int>(time*rate));};
    for(const auto overlap:{.05,0.,-.03})
    {
        std::vector<NsfUtauMixNote> notes{{flat(1,.2f),0,0,false},{flat(.25,.1f),.2,overlap,false}};
        std::vector<UtauMixEnvelope> envelopes(2);
        auto mix=mixNsfUtauNotes(notes,1,rate,1,{},[&](size_t i,const auto& e){envelopes[i]=e;});
        const auto name=juce::String(overlap,2);
        check("old_voice_stays_silent_after_crossfade_"+name,std::abs(at(mix,.3)-.1f)<1.e-6);
        check("old_voice_does_not_return_after_new_note_ends_"+name,std::abs(at(mix,.6))<1.e-6);
        check("display_gain_stays_zero_after_crossfade_"+name,envelopes[0].gainAt(.3)==0);
        auto without=mixNsfUtauNotes(notes,1,rate,1);double error=0,foldError=0;
        for(int i=0;i<mix.getNumSamples();++i)
        {
            error=std::max(error,static_cast<double>(std::abs(mix.getSample(0,i)-without.getSample(0,i))));
            const auto t=i/rate;
            const auto folded=.2f*envelopes[0].gainAt(t)+.1f*envelopes[1].gainAt(t);
            foldError=std::max(foldError,static_cast<double>(std::abs(mix.getSample(0,i)-folded)));
        }
        check("preview_metadata_matches_audio_"+name,error<1.e-6&&foldError<1.e-6);
        if(overlap<0)check("negative_overlap_keeps_silent_gap",std::abs(at(mix,.185))<1.e-6);
    }
    std::vector<NsfUtauMixNote> shortNext{{flat(1,.2f),0,0,false},{flat(.01,.1f),.2,.05,false}};
    const auto shortMix=mixNsfUtauNotes(shortNext,1,rate,1);
    check("short_next_note_does_not_resurrect_previous",std::abs(at(shortMix,.3))<1.e-6&&at(shortMix,.225)<.2f);
    shortNext.push_back({flat(.2,.15f),.4,.03,false});
    check("three_voice_overlap_does_not_accumulate_old_tails",std::abs(at(mixNsfUtauNotes(shortNext,1,rate,1),.48)-.15f)<1.e-6);
    check("cancellation_still_supported",mixNsfUtauNotes(shortNext,1,rate,1,[]{return true;}).getNumSamples()==0);
    for(int style:{0,1})
    {
        std::vector<NsfUtauMixNote> notes{{flat(.4,.2f),0,0,false,.4,style},{flat(.3,.1f),.2,.05,false,.3,style}};
        std::vector<UtauMixEnvelope> windows(2);
        const auto mix=mixNsfUtauNotes(notes,.5,rate,1,{},[&](size_t i,const auto& e){windows[i]=e;});
        bool exact=true;
        for(int s=0;s<static_cast<int>(.05*rate);++s)
        {
            const auto u=s/(.05*rate),v=style?u*u*(3-2*u):u;
            const auto expected=.2*(1-v)+.1*v;
            exact&=std::abs(mix.getSample(0,static_cast<int>(.2*rate)+s)-expected)<1.e-6;
        }
        check(style?"nsf_standard_curve_is_single_smooth_crossfade":"nsf_standard_linear_is_single_straight_crossfade",exact);
        check("nsf_standard_isolated_edges_use_5_35_ms_"+juce::String(style),
            std::abs(windows[0].gainAt(.0025)-.5)<1.e-6&&std::abs(windows[1].gainAt(.4825)-.5)<1.e-6);
    }

    if(projectFile.existsAsFile())
    {
        ProjectModel model;juce::String error;
        check("user_project_loads_read_only",model.load(projectFile,error));
        verifyLegacyCvvcEnvelopeLoad(projectFile,output,model,check);
        const auto data=model.snapshot();int clips=0,ramps=0;
        for(const auto& track:data.tracks)for(const auto& clip:track.clips)
        {
            if(!track.chineseCvvc||clip.notes.empty())continue;
            UtauRenderRequest request;request.chineseCvvc=true;request.voicebankDirectory=track.voicebankDirectory;
            request.bpm=data.bpm;request.targetDurationSeconds=clip.durationSeconds+2;
            request.notes=AudioEngine::utauRequestNotesForClip(data,clip.id);
            const auto planned=planChineseCvvc(request);
            check("real_project_all_six_phonemes_use_standard_linear_defaults",planned.request.notes.size()==6
                &&std::all_of(planned.request.notes.begin(),planned.request.notes.end(),[](const auto& n){
                    return n.cvvcStandardEnvelope&&n.amplitudeEnvelope.size()==4
                        &&std::all_of(n.amplitudeEnvelope.begin(),n.amplitudeEnvelope.end(),[](const auto& p){return p.linearToNext;});}));
            std::vector<std::vector<UtauPhonemeSpan>> spans(request.notes.size());
            request.notePhonemes=[&](size_t i,const auto& p){spans[i]=p;};
            auto hi=renderNsfUtauPhrase(request,models,{});
            check("real_project_hifisampler_renders",hi.buffer.getNumSamples()>0&&hi.warning.isEmpty());
            bool holds=true;
            for(const auto& parts:spans)for(const auto& p:parts)for(const auto& r:p.mixEnvelope.ramps)
                if(!r.rising&&r.crossfade)
                {++ramps;holds&=r.gainAt(r.start+r.length+1)==0;}
            check("real_cvvc_crossfades_hold_silence",holds);
            const auto save=[&](const UtauRenderResult& audio,const juce::String& name)
            {
                if(audio.buffer.getNumSamples()==0)return false;
                juce::WavAudioFormat format;auto stream=output.getChildFile(name+".wav").createOutputStream();
                if(!stream)return false;stream->setPosition(0);stream->truncate();
                std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),audio.sampleRate,audio.buffer.getNumChannels(),24,{},0));
                return writer&&writer->writeFromAudioSampleBuffer(audio.buffer,0,audio.buffer.getNumSamples());
            };
            check("real_project_hifisampler_wav_saved",save(hi,"hifisampler-"+juce::String(clips)));
            const auto hiSpans=spans;
            if(hi.buffer.getNumSamples()>0)verifyRealCvvcEnvelopeDisplay(model,track,clip,spans,output,"hifisampler",check);
            auto traditional=request;traditional.requireExternalResampler=true;traditional.resamplerExecutable=resampler;
            // Compare the wavtool-equivalent envelope independently of the
            // external HF daemon. K2 changes the reference timbre, not timing
            // or automatic mixing; these changes exist only in this fixture.
            for(auto& note:traditional.notes)note.flags="K2";
            const auto wcs=UtauRenderer::render(traditional);
            std::cout<<"wcsndm_k2_reference_warning="<<wcs.warning<<std::endl;
            check("real_project_wcsndm_k2_reference_renders",wcs.buffer.getNumSamples()>0&&wcs.backend.contains("utau-resampler"));
            check("real_project_wcsndm_k2_wav_saved",save(wcs,"wcsndm-k2-"+juce::String(clips)));
            if(wcs.buffer.getNumSamples()>0)verifyRealCvvcEnvelopeDisplay(model,track,clip,spans,output,"wcsndm",check);
            for(bool curved:{false,true})
            {
                auto direct=planned.request;direct.cvvcCurveCrossfade=curved;
                direct.requireExternalResampler=true;direct.resamplerExecutable=resampler;
                for(auto& n:direct.notes)n.flags="K2";
                std::map<size_t,UtauMixEnvelope> windows;bool single=true,shape=true;int samples=0;
                direct.noteMixEnvelope=[&](size_t i,const auto& e){windows[i]=e;};
                direct.notePiece=[&](size_t i,const auto&,double,double,const auto& gain,const auto&){
                    const auto& e=windows.at(i);single&=e.replacesStandardEnvelope;
                    for(const auto& r:e.ramps)for(int s=std::max(0,r.start);s<std::min(e.samples,r.start+r.length);++s)
                    {
                        ++samples;const auto t=e.startSeconds+s/e.sampleRate;
                        single&=std::abs(gain(t)-e.gainAtSample(s))<1.e-5;
                        const auto u=static_cast<double>(s-r.start)/r.length;
                        const auto v=curved&&r.crossfade?u*u*(3-2*u):u;
                        shape&=std::abs(r.gainAt(s)-(r.rising?v:1-v))<1.e-6;
                    }
                };
                const auto audio=UtauRenderer::render(direct);
                check(curved?"real_project_curve_mode_has_no_extra_edge_curve":"real_project_linear_mode_has_no_extra_edge_curve",
                    audio.buffer.getNumSamples()>0&&samples>100&&single&&shape);
                std::cout<<"real_project_checked_edge_samples_"<<curved<<'='<<samples<<std::endl;
            }
            int compared=0;bool sameWindows=true;
            const auto window=[](const UtauPhonemeSpan& piece)->std::optional<std::pair<double,double>>
            {
                for(const auto& ramp:piece.mixEnvelope.ramps)if(!ramp.rising&&ramp.crossfade)
                    return std::pair{piece.mixEnvelope.startSeconds+ramp.start/piece.mixEnvelope.sampleRate,
                        piece.mixEnvelope.startSeconds+(ramp.start+ramp.length)/piece.mixEnvelope.sampleRate};
                return {};
            };
            for(size_t i=0;i<std::min(hiSpans.size(),spans.size());++i)
                for(size_t j=0;j<std::min(hiSpans[i].size(),spans[i].size());++j)
                {
                    const auto a=window(hiSpans[i][j]),b=window(spans[i][j]);
                    if(!a||!b)continue;
                    ++compared;sameWindows&=std::abs(a->first-b->first)<.0001&&std::abs(a->second-b->second)<.0001;
                }
            check("real_project_automatic_windows_match_wcsndm",compared>0&&sameWindows);
            ++clips;
        }
        check("real_project_contains_cvvc_crossfades",clips>0&&ramps>0);
    }
    auto report=std::make_unique<juce::DynamicObject>();report->setProperty("passed",ok);report->setProperty("checks",checks);
    output.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report.release()),false));
    return ok;
}
}
