#pragma once
#include "../backend/NsfHifiganRenderer.h"

namespace hachi
{
inline bool utauOverlapPitchSmoke(const juce::File& output,const juce::File& resampler,const juce::String& flags)
{
    using namespace backend;
    output.createDirectory();bool ok=true;juce::String log;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<"="<<pass<<std::endl;};
    const auto bank=output.getChildFile("bank");bank.createDirectory();
    bank.getChildFile("oto.ini").replaceWithText("tone.wav=a,0,220,-900,180,220\ntone.wav=b,0,220,-900,180,220\n");
    bank.getChildFile("oto.jie.ini").replaceWithText(bank.getChildFile("oto.ini").loadFileAsString());
    bank.getChildFile("oto4.ini").replaceWithText("tone.wav=0,100,220,700\n");
    bank.getChildFile("otomou.ini").replaceWithText("tone.wav=CVVC,0,100,220,700\n");
    {
        juce::AudioBuffer<float> wave(1,44100);
        for(int s=0;s<wave.getNumSamples();++s)wave.setSample(0,s,.2f*std::sin(float(s*2*juce::MathConstants<double>::pi*220/44100)));
        auto stream=bank.getChildFile("tone.wav").createOutputStream();if(!stream)return false;stream->setPosition(0);stream->truncate();
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),44100,1,24,{},0));
        if(!writer||!writer->writeFromAudioSampleBuffer(wave,0,wave.getNumSamples()))return false;
    }
    UtauRenderer::invalidateVoicebankCache();
    const auto sample=[](const UtauNoteRenderSpec& n,double t)
    {
        if(n.timelinePitchCents)return n.timelinePitchCents(t);
        if(n.pitchCurve.empty())return 0.f;
        const auto r=std::lower_bound(n.pitchCurve.begin(),n.pitchCurve.end(),t,[](const auto& p,double time){return p.timeSeconds<time;});
        if(r==n.pitchCurve.begin())return r->cents;if(r==n.pitchCurve.end())return n.pitchCurve.back().cents;
        const auto& l=*std::prev(r);return l.cents+(r->cents-l.cents)*static_cast<float>((t-l.timeSeconds)/(r->timeSeconds-l.timeSeconds));
    };
    const juce::String alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        ProjectData original;original.bpm=120;TrackData track;track.id="track";track.trackType=TrackType::utau;
        track.utauMode=mode;track.pitchAlgorithm=PitchAlgorithm::utau;track.voicebankDirectory=bank;
        ClipData clip;clip.id="clip";clip.startSeconds=10;clip.durationSeconds=2.2;
        for(int i=0;i<3;++i)
        {
            NoteData n;n.id="n"+juce::String(i);n.label=i%2?"b":"a";n.startSeconds=.5+i*.5;n.durationSeconds=.5;
            n.midiNote=i%2?64.f:60.f;n.utauAutoPitchTransition=false;n.utauFlags=flags;
            n.pitchControlPoints={{0,60.f+2*i,PitchCurveShape::linear},{.25,61.f+2*i,PitchCurveShape::linear},{.5,62.f+2*i,PitchCurveShape::linear}};
            n.contour={{0,0,0,true},{.5,0,0,true}};clip.notes.push_back(n);
        }
        track.clips={clip};original.tracks={track};const auto modeTag="mode"+juce::String(static_cast<int>(mode))+"_";
        juce::String initialTiming,initialPackets;std::vector<float> initialAudio;
        for(int pass=0;pass<3;++pass)
        {
            auto data=original;if(pass==1)for(auto& p:data.tracks[0].clips[0].notes[0].pitchControlPoints)p.targetMidi+=1.5f;
            auto request=UtauRenderRequest{};request.voicebankDirectory=bank;request.resamplerExecutable=resampler;request.requireExternalResampler=true;
            request.fourRegion=mode!=UtauMode::classic;request.consonantClasses=mode==UtauMode::mou;request.bpm=120;request.targetDurationSeconds=2.2;
            request.notes=AudioEngine::utauRequestNotesForClip(data,"clip");for(auto& n:request.notes)n.flags=flags;
            const auto projected=AudioEngine::projectTrackNotePitch(data.tracks[0],{});
            const auto desired=[&](double t)->std::optional<float>
            {for(const auto& p:projected)if(t+10>=p.absoluteStartSeconds&&t+10<p.absoluteStartSeconds+p.durationSeconds)
                return p.midiNote*100+p.continuousLocalCents(t+10-p.absoluteStartSeconds);return {};};
            double maximum=0,wireMaximum=0,nsfMaximum=0;int count=0;juce::String packets;
            for(const auto& n:request.notes)
            {
                for(double t=-.18;t<n.durationSeconds+.04;t+=.001)if(const auto expected=desired(n.startSeconds+t))
                {maximum=std::max(maximum,std::abs(double(n.midiNote*100+sample(n,t)-*expected)));++count;}
                const auto wire=UtauRenderer::diagnosticPitchbend(n,120,.18,.74);packets+=wire;
                for(int i=0;i+1<wire.length();i+=2)if(const auto expected=desired(n.startSeconds+(i/2)*60./(120*96)-.18))
                {const auto v=alphabet.indexOfChar(wire[i])*64+alphabet.indexOfChar(wire[i+1]);wireMaximum=std::max(wireMaximum,std::abs(double(std::lround(n.midiNote)*100+(v>=2048?v-4096:v)-*expected)));}
                std::vector<NsfUtauPitchPoint> points;for(const auto& p:n.pitchCurve)points.push_back({p.timeSeconds,p.cents});
                const auto f0=buildNsfUtauTargetMidi(n.midiNote,points,5,.74,-.18,n.timelinePitchCents);
                for(size_t i=0;i<f0.size();++i)if(const auto expected=desired(n.startSeconds+i*.005-.18))nsfMaximum=std::max(nsfMaximum,std::abs(double(f0[i]*100-*expected)));
            }
            const auto tag=modeTag+"pass"+juce::String(pass)+"_";
            std::cout<<tag<<"pitch_error_cents="<<maximum<<" wire_error="<<wireMaximum<<" samples="<<count<<std::endl;
            check(tag+"overlaps_follow_timeline",count>1000&&maximum<.02);
            check(tag+"engine_packet_follows_timeline",wireMaximum<.51);
            check(tag+"hifisampler_frames_follow_timeline",nsfMaximum<.02);
            if(pass==0)initialPackets=packets;
            check(tag+"pitch_changes_and_restores",pass==0||((pass==2)==(initialPackets==packets)));
            auto selected=AudioEngine::diagnosticUtauRequestNotes(data,"clip",{"n1"});bool selection=selected.size()==1;
            if(selection)for(const auto t:{-.16,-.1,0.,.2,.53})selection&=std::abs(sample(selected.front(),t)-sample(request.notes[1],t))<.01;
            check(tag+"selection_keeps_neighbour_pitch",selection);
            juce::String timing;request.noteMixEnvelope=[&](size_t i,const UtauMixEnvelope& e)
            {timing+=juce::String(i)+":"+juce::String(e.startSeconds,9)+":"+juce::String(e.samples)+";";for(const auto& r:e.ramps)timing+=juce::String(r.start)+","+juce::String(r.length)+";";};
            const auto audio=UtauRenderer::render(request);
            std::cout<<tag<<"backend="<<audio.backend<<" warning="<<audio.warning<<std::endl;
            check(tag+"actual_external_render",audio.buffer.getNumSamples()>0&&audio.backend=="utau-resampler+internal-wavtool"&&audio.warning.isEmpty());
            if(pass==0)initialTiming=timing;
            check(tag+"actual_timing_is_unchanged",timing.isNotEmpty()&&timing==initialTiming);
            if(audio.buffer.getNumSamples()>0)
            {
                auto stream=output.getChildFile(tag+"audio.wav").createOutputStream();stream->setPosition(0);stream->truncate();
                juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),audio.sampleRate,audio.buffer.getNumChannels(),24,{},0));
                if(writer)writer->writeFromAudioSampleBuffer(audio.buffer,0,audio.buffer.getNumSamples());
                if(pass==0)initialAudio.assign(audio.buffer.getReadPointer(0),audio.buffer.getReadPointer(0)+audio.buffer.getNumSamples());
                else if(initialAudio.size()==static_cast<size_t>(audio.buffer.getNumSamples()))
                {const auto equal=std::equal(initialAudio.begin(),initialAudio.end(),audio.buffer.getReadPointer(0));check(tag+"pcm_changes_and_restores",equal==(pass==2));}
            }
            if(pass==1)
            {
                // Exercise the real background worker too: discarding the
                // continuous evaluator here used to undo a correct DTO.
                auto offline=data;offline.tracks[0].clips[0].startSeconds=0;
                offline.tracks[0].outputResampler=resampler;
                AudioEngine engine(false);engine.prepareToPlay(512,44100);
                engine.setUtauResamplerFile(resampler);engine.setUtauOutputDefaults(UtauOutputEngine::resampler,{});
                engine.selectEveryUtauNote(offline);engine.syncProject(offline,true);
                const auto deadline=juce::Time::getMillisecondCounterHiRes()+60000;
                while(engine.renderProgress()&&juce::Time::getMillisecondCounterHiRes()<deadline)juce::Thread::sleep(10);
                const auto rendered=!engine.renderProgress()&&engine.activeRenderWarnings().isEmpty();
                juce::String error;const auto file=output.getChildFile(tag+"worker.wav");
                check(tag+"background_worker_exports",rendered&&engine.exportWav(file,error,{},0,2.2,{44100,2,24}));
                juce::WavAudioFormat wav;auto reader=std::unique_ptr<juce::AudioFormatReader>(wav.createReaderFor(file.createInputStream().release(),true));
                bool equal=false;
                if(reader&&reader->lengthInSamples==audio.buffer.getNumSamples())
                {
                    juce::AudioBuffer<float> actual(2,int(reader->lengthInSamples));reader->read(&actual,0,actual.getNumSamples(),0,true,true);
                    double xy=0,xx=0,yy=0;for(int s=0;s<actual.getNumSamples();++s){const double x=actual.getSample(0,s),y=audio.buffer.getSample(0,s);xy+=x*y;xx+=x*x;yy+=y*y;}
                    const auto correlation=xy/std::sqrt(std::max(1.e-20,xx*yy));std::cout<<tag<<"worker_pcm_correlation="<<correlation<<std::endl;
                    equal=correlation>.99999;
                }
                check(tag+"worker_keeps_full_timeline_pitch",equal);
            }
        }
        auto narrow=original;auto& n=narrow.tracks[0].clips[0].notes[0];
        n.pitchControlPoints={{0,60,PitchCurveShape::linear},{.021,60,PitchCurveShape::linear},{.022,61,PitchCurveShape::linear},{.023,60,PitchCurveShape::linear},{.5,60,PitchCurveShape::linear}};
        const auto narrowRequest=AudioEngine::utauRequestNotesForClip(narrow,"clip");
        check(modeTag+"sub_5ms_pitch_corner_is_preserved",std::abs(sample(narrowRequest[0],.022)-100)<.1);
        ProjectModel model;model.resetDocument(narrow);I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1200,700);roll.setFocusedTrack("track");roll.setFocusedClip("clip");
        const auto shown=roll.diagnosticPitchLineAt("n0",10.522);
        check(modeTag+"display_recognizes_short_pitch_corner",shown&&std::abs(*shown-61)<.001);
        auto curved=original;auto& cn=curved.tracks[0].clips[0].notes[0];
        cn.pitchControlPoints={{-.04,59},{0,60},{.024,64,PitchCurveShape::customBezier},{.08,60,PitchCurveShape::smooth},{.5,62}};
        cn.pitchControlPoints[2].bezierX1=.1f;cn.pitchControlPoints[2].bezierY1=.95f;cn.pitchControlPoints[2].bezierX2=.9f;cn.pitchControlPoints[2].bezierY2=.05f;
        cn.vibratoEnabled=true;cn.vibratoLengthPercent=100;cn.vibratoDepthCents=35;
        const auto curvedRequest=AudioEngine::utauRequestNotesForClip(curved,"clip");double curveError=0;
        for(int i=0;i<900;++i){const auto t=-.04+i*.0001;const auto target=(evaluatePitchCurve(cn.pitchControlPoints,t)-cn.midiNote)*100+float(vibratoCentsAt(cn,juce::jlimit(0.,cn.durationSeconds,t)));
            curveError=std::max(curveError,std::abs(double(sample(curvedRequest[0],t)-target)));}
        std::cout<<modeTag<<"bezier_vibrato_error_cents="<<curveError<<std::endl;
        check(modeTag+"complete_bezier_and_single_vibrato",curveError<.15);
        for(const auto rest:{false,true})
        {
            auto isolated=original;auto& notes=isolated.tracks[0].clips[0].notes;
            if(rest){auto r=notes[1];r.id="rest";r.label="RR";r.durationSeconds=0;notes.insert(notes.begin()+1,r);}
            else for(size_t i=1;i<notes.size();++i)notes[i].startSeconds+=.1;
            const auto req=AudioEngine::utauRequestNotesForClip(isolated,"clip");const auto& next=req[rest?2:1];
            check(modeTag+(rest?"rest_isolates_pitch":"gap_isolates_pitch"),std::abs(next.midiNote+sample(next,-.1)/100.f-62)<.001);
        }
        auto separate=original;auto& tr=separate.tracks[0];auto second=tr.clips[0];second.id="clip2";second.startSeconds=11;second.notes.erase(second.notes.begin());
        for(auto& note:second.notes)note.startSeconds-=1;tr.clips[0].notes.resize(1);tr.clips.push_back(second);
        const auto other=AudioEngine::utauRequestNotesForClip(separate,"clip2");
        check(modeTag+"cross_clip_head_reads_previous_curve",other.size()==2&&std::abs(other[0].midiNote+sample(other[0],-.1)/100.f-61.6f)<.001);
    }
    output.getChildFile("checks.txt").replaceWithText(log);return ok;
}
}
