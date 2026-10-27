#pragma once
#include "../backend/ChineseCvvcPhonemizer.h"
#include "../backend/NsfHifiganRenderer.h"

namespace hachi
{
inline bool cvvcPitchProjectSmoke(const juce::File& output,const juce::File& input,
    const juce::File& resampler,const juce::String& flags)
{
    using namespace backend;
    output.createDirectory();bool ok=true;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<"="<<pass<<std::endl;};
    ProjectModel model;check("real_project_load",model.load(input,error));
    auto original=model.snapshot();
    const auto found=std::find_if(original.tracks.begin(),original.tracks.end(),[](const auto& t){return t.chineseCvvc&&!t.clips.empty();});
    if(found==original.tracks.end())return false;
    const auto trackIndex=static_cast<size_t>(found-original.tracks.begin());
    const auto clipId=found->clips.front().id;
    const auto cents=[](const UtauNoteRenderSpec& n,double t)
    {
        if(n.timelinePitchCents)return n.timelinePitchCents(t);
        if(n.pitchCurve.empty())return 0.f;
        const auto r=std::lower_bound(n.pitchCurve.begin(),n.pitchCurve.end(),t,[](const auto& p,double time){return p.timeSeconds<time;});
        if(r==n.pitchCurve.begin())return r->cents;if(r==n.pitchCurve.end())return n.pitchCurve.back().cents;
        const auto& l=*std::prev(r);return l.cents+(r->cents-l.cents)*static_cast<float>((t-l.timeSeconds)/(r->timeSeconds-l.timeSeconds));
    };
    const auto signature=[](const ChineseCvvcPlan& p)
    {
        juce::String s;
        for(size_t i=0;i<p.request.notes.size();++i){const auto& n=p.request.notes[i];s+=n.alias+"/"+juce::String(n.startSeconds,9)+"/"+juce::String(n.durationSeconds,9)+"/"+juce::String(n.midiNote)+"/"+juce::String(n.consonantVelocity)+";";}
        for(const auto& spans:p.spans)for(const auto& s0:spans)s+=juce::String(s0.leadInSeconds,9)+";";
        return s;
    };
    juce::String originalTiming,originalPackets,originalMix;std::vector<float> firstAudio;
    for(int pass=0;pass<4;++pass)
    {
        auto data=original;auto& track=data.tracks[trackIndex];
        // Change pitch only, including a note feeding a later VC and the VC's owner.
        if(pass==1||pass==2)for(auto& n:track.clips.front().notes)
            if(n.startSeconds>=70.24&&n.startSeconds<71.01)
            {
                n.pitchControlPoints={{0,n.midiNote+(pass==1?2.f:-1.f)},
                    {n.durationSeconds*.35,n.midiNote-1.f},{n.durationSeconds*.8,n.midiNote+1.f}};
                n.pitchSupportAuthored=true;
            }
        UtauRenderRequest request;request.chineseCvvc=true;request.voicebankDirectory=track.voicebankDirectory;
        request.bpm=data.bpm;request.cvvcCurveCrossfade=track.cvvcCurveCrossfade;
        request.notes=AudioEngine::utauRequestNotesForClip(data,clipId);
        std::erase_if(request.notes,[](const auto& n){return n.startSeconds<69||n.startSeconds>=71.25;});
        // The normal external-render worker uses sampled packets, not diagnostic callbacks.
        for(auto& n:request.notes){n.startSeconds-=68.5;n.flags=flags;n.timelinePitchCents={};}
        request.targetDurationSeconds=3;request.resamplerExecutable=resampler;request.requireExternalResampler=true;
        auto plan=planChineseCvvc(request);const auto tag="pass"+juce::String(pass)+"_";
        check(tag+"real_linked_phonemes",plan.request.notes.size()>request.notes.size());
        if(pass==0)originalTiming=signature(plan);
        check(tag+"pitch_edit_preserves_alias_timing_velocity",signature(plan)==originalTiming);
        const auto expected=[&](double t)->std::optional<float>
        {
            for(const auto& n:request.notes)if(t>=n.startSeconds&&t<n.startSeconds+n.durationSeconds&&!isRestLyric(n.alias)&&n.alias.isNotEmpty())
                return n.midiNote*100+cents(n,t-n.startSeconds);
            return {};
        };
        double maximum=0,wireMaximum=0,nsfMaximum=0;int comparisons=0;juce::String packets,detail;
        const juce::String alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");
        for(size_t i=0;i<plan.request.notes.size();++i)
        {
            const auto& n=plan.request.notes[i];const auto owner=plan.owners[i];
            const auto span=std::find_if(plan.spans[owner].begin(),plan.spans[owner].end(),[&](const auto& s){return std::abs(s.startSeconds-plan.offsets[i])<1.e-8;});
            if(span==plan.spans[owner].end())continue;
            double pieceError=0;
            for(double t=-span->leadInSeconds;t<n.durationSeconds+.05;t+=.001)
                if(const auto target=expected(n.startSeconds+t))
                {
                    pieceError=std::max(pieceError,std::abs(double(n.midiNote*100+cents(n,t)-*target)));++comparisons;
                }
            maximum=std::max(maximum,pieceError);
            const auto encoded=UtauRenderer::diagnosticPitchbend(n,request.bpm,span->leadInSeconds,span->leadInSeconds+n.durationSeconds+.05);
            packets+=encoded;
            std::vector<NsfUtauPitchPoint> nsfPoints;for(const auto& p:n.pitchCurve)nsfPoints.push_back({p.timeSeconds,p.cents});
            const auto targetFrames=buildNsfUtauTargetMidi(n.midiNote,nsfPoints,5,span->leadInSeconds+n.durationSeconds+.05,-span->leadInSeconds,n.timelinePitchCents);
            for(size_t j=0;j<targetFrames.size();++j)if(const auto target=expected(n.startSeconds+j*.005-span->leadInSeconds))
                nsfMaximum=std::max(nsfMaximum,std::abs(double(targetFrames[j]*100-*target)));
            for(int j=0;j+1<encoded.length();j+=2)
            {
                const auto t=(j/2)*60.0/(request.bpm*96)-span->leadInSeconds;
                const auto v=alphabet.indexOfChar(encoded[j])*64+alphabet.indexOfChar(encoded[j+1]);
                if(const auto target=expected(n.startSeconds+t))wireMaximum=std::max(wireMaximum,
                    std::abs(double(std::lround(n.midiNote)*100+(v>=2048?v-4096:v)-*target)));
            }
            detail+=n.alias+","+juce::String(n.startSeconds,6)+","+juce::String(pieceError,3)+"\n";
        }
        output.getChildFile(tag+"pitch_errors.csv").replaceWithText("alias,start,max_error_cents\n"+detail);
        std::cout<<tag<<"maximum_pitch_error_cents="<<maximum<<" wire_error_cents="<<wireMaximum<<" samples="<<comparisons<<std::endl;
        check(tag+"all_overlaps_read_score_timeline",comparisons>1000&&maximum<.02);
        check(tag+"resampler_pitch_packet_matches_timeline",wireMaximum<=.501);
        check(tag+"hifisampler_f0_frames_match_timeline",nsfMaximum<.02);
        if(pass==0)originalPackets=packets;
        check(tag+"pitch_packets_follow_edit_and_restore",pass==0||((pass==3)==(packets==originalPackets)));
        juce::String mix;
        plan.request.noteMixEnvelope=[&](size_t i,const UtauMixEnvelope& e)
        {
            mix+=juce::String(i)+":"+juce::String(e.startSeconds,9)+":"+juce::String(e.samples)+":"+juce::String(e.sampleRate)+";";
            for(const auto& r:e.ramps)mix+=juce::String(r.start)+","+juce::String(r.length)+","+juce::String(static_cast<int>(r.rising))+";";
        };
        auto rendered=UtauRenderer::render(plan.request);
        std::cout<<tag<<"backend="<<rendered.backend<<" warning="<<rendered.warning<<std::endl;
        check(tag+"external_audio_rendered",rendered.buffer.getNumSamples()>0&&rendered.backend=="utau-resampler+internal-wavtool"&&rendered.warning.isEmpty());
        if(pass==0)originalMix=mix;
        check(tag+"actual_mix_timing_unchanged",mix.isNotEmpty()&&mix==originalMix);
        auto stream=output.getChildFile(tag+"audio.wav").createOutputStream();
        if(stream){stream->setPosition(0);stream->truncate();juce::WavAudioFormat wav;
            std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),rendered.sampleRate,rendered.buffer.getNumChannels(),24,{},0));
            check(tag+"audio_written",writer&&writer->writeFromAudioSampleBuffer(rendered.buffer,0,rendered.buffer.getNumSamples()));}
        if(pass==0&&rendered.buffer.getNumSamples()>0)firstAudio.assign(rendered.buffer.getReadPointer(0),rendered.buffer.getReadPointer(0)+rendered.buffer.getNumSamples());
        if(pass==1&&rendered.buffer.getNumSamples()==static_cast<int>(firstAudio.size()))
        {double delta=0;for(int i=0;i<rendered.buffer.getNumSamples();++i)delta+=std::abs(firstAudio[i]-rendered.buffer.getSample(0,i));check(tag+"actual_audio_changes_with_pitch",delta>1);}
        if(pass==3)check(tag+"restored_pitch_restores_cached_audio",rendered.buffer.getNumSamples()==static_cast<int>(firstAudio.size())&&std::equal(firstAudio.begin(),firstAudio.end(),rendered.buffer.getReadPointer(0)));
    }
    for(const auto rest:{false,true})
    {
        UtauRenderRequest request;request.chineseCvvc=true;request.voicebankDirectory=found->voicebankDirectory;
        UtauNoteRenderSpec a;a.alias="peng";a.startSeconds=1;a.durationSeconds=.5;a.midiNote=59;a.pitchCurve={{0,0},{.5,100}};
        auto b=a;b.alias="you";b.startSeconds=rest?1.5:1.6;b.midiNote=57;b.pitchCurve={{0,0},{.5,-100}};
        request.notes={a};
        if(rest){auto r=a;r.alias="RR";r.startSeconds=1.5;r.durationSeconds=0;request.notes.push_back(r);}
        request.notes.push_back(b);const auto owner=request.notes.size()-1;
        const auto before=planChineseCvvc(request);
        for(auto& p:request.notes[0].pitchCurve)p.cents+=1200;
        const auto after=planChineseCvvc(request);bool isolated=true,checked=false;
        for(size_t i=0;i<before.owners.size();++i)if(before.owners[i]==owner)
            for(const auto t:{-.2,-.05,0.,.1,.5})
            {checked=true;isolated&=std::abs(cents(before.request.notes[i],t)-cents(after.request.notes[i],t))<.001;}
        check(rest?"explicit_rest_breaks_pitch_run":"gap_breaks_pitch_run",checked&&isolated);
    }
    output.getChildFile("checks.txt").replaceWithText(log);return ok;
}
}
