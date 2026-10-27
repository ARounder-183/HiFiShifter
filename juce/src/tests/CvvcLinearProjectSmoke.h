#pragma once
#include "CvvcLegacyEnvelopeSmoke.h"
#include "../backend/ChineseCvvcPhonemizer.h"
#include "UtauEnvelopeDefaultsSmoke.h"

namespace hachi
{
inline bool cvvcLinearProjectSmoke(const juce::File& output,const juce::File& input,
    const juce::File& resampler)
{
    using namespace backend;
    output.createDirectory();bool ok=true;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<"="<<pass<<std::endl;};
    ProjectModel model;check("real_project_load",model.load(input,error));
    auto data=model.snapshot();
    auto found=std::find_if(data.tracks.begin(),data.tracks.end(),[](const auto& t){return t.chineseCvvc&&!t.clips.empty();});
    if(found==data.tracks.end())return false;
    const auto trackId=found->id,clipId=found->clips.front().id;
    verifyUtauEnvelopeDefaults(data,*found,output,check);
    check("new_track_defaults_to_linear",!TrackData{}.cvvcCurveCrossfade);
    check("real_project_selected_linear",!found->cvvcCurveCrossfade);
    std::vector<std::pair<juce::String,juce::String>> lyrics;
    for(const auto& n:found->clips.front().notes)if(n.label.isNotEmpty()&&n.label!="-")lyrics.emplace_back(n.id,n.label);
    check("real_project_has_lyric_section",lyrics.size()>=20);
    I18n liveStrings;PianoRollComponent liveRoll(model,liveStrings);
    liveRoll.setSize(1400,780);liveRoll.setFocusedTrack(trackId);liveRoll.setFocusedClip(clipId);
    for(int pass=0;pass<5;++pass)
    {
        const auto tag="pass"+juce::String(pass)+"_";
        const bool curved=pass%2!=0;
        model.setTrackCvvcCurveCrossfade(trackId,curved);
        model.dispatchPendingMessages();liveRoll.diagnosticRefresh();
        bool noStale=true;
        for(const auto& [id,text]:lyrics)for(const auto& span:liveRoll.diagnosticPhonemeSpans(id))
            for(const auto& ramp:span.mixEnvelope.ramps)if(ramp.crossfade)noStale&=ramp.smoothCurve==curved;
        check(tag+"previous_style_metadata_not_reused",noStale);
        if(pass==1)
        {
            auto blanks=lyrics;for(auto& pair:blanks)pair.second="-";
            model.setNoteLabels(blanks);model.dispatchPendingMessages();
            I18n strings;PianoRollComponent roll(model,strings);
            roll.setSize(1400,780);roll.setFocusedTrack(trackId);roll.setFocusedClip(clipId);
            for(const auto& [id,text]:lyrics)
            {
                check(tag+"inline_lyric_opens_"+id,roll.diagnosticBeginAliasEdit(id));
                roll.diagnosticCommitAliasEdit(text);
                model.dispatchPendingMessages();roll.diagnosticRefresh();
            }
        }
        if(pass>=2)
        {
            const auto copy=output.getChildFile("roundtrip.hjpx");
            check(tag+"save_reopen",model.save(copy,error)&&model.load(copy,error));
        }
        data=model.snapshot();const auto& track=*std::find_if(data.tracks.begin(),data.tracks.end(),[&](const auto& t){return t.id==trackId;});
        auto clip=track.clips.front();
        std::erase_if(clip.notes,[](const auto& n){return n.label.isEmpty()||n.label=="-";});
        check(tag+"no_legacy_parent_template",std::all_of(clip.notes.begin(),clip.notes.end(),[](const auto& n){return n.amplitudeEnvelope.empty();}));
        UtauRenderRequest request;request.chineseCvvc=true;request.voicebankDirectory=track.voicebankDirectory;
        request.bpm=data.bpm;request.cvvcCurveCrossfade=track.cvvcCurveCrossfade;
        request.notes=AudioEngine::utauRequestNotesForClip(data,clipId);
        std::erase_if(request.notes,[](const auto& n){return n.alias.isEmpty()||n.alias=="-";});
        for(auto& n:request.notes){n.flags="K2";request.targetDurationSeconds=std::max(request.targetDurationSeconds,n.startSeconds+n.durationSeconds+.2);}
        request.resamplerExecutable=resampler;request.requireExternalResampler=true;
        auto plan=planChineseCvvc(request);
        check(tag+"every_phoneme_has_standard_linear_envelope",!plan.request.notes.empty()&&std::all_of(plan.request.notes.begin(),plan.request.notes.end(),[](const auto& n){return n.cvvcStandardEnvelope&&std::all_of(n.amplitudeEnvelope.begin(),n.amplitudeEnvelope.end(),[](const auto& p){return p.linearToNext;});}));
        std::map<size_t,UtauMixEnvelope> windows;int samples=0;bool single=true,straight=true;
        plan.request.noteMixEnvelope=[&](size_t i,const auto& e){windows[i]=e;};
        plan.request.notePiece=[&](size_t i,const auto&,double,double,const auto& gain,const auto&)
        {
            const auto& e=windows.at(i);single&=e.replacesStandardEnvelope;
            for(const auto& r:e.ramps)for(int s=std::max(0,r.start);s<std::min(e.samples,r.start+r.length);s+=13)
            {
                ++samples;const auto u=double(s-r.start)/r.length;
                const auto v=curved&&r.crossfade?u*u*(3-2*u):u;
                straight&=std::abs(r.gainAt(s)-(r.rising?v:1-v))<1.e-6;
                single&=std::abs(gain(e.startSeconds+s/e.sampleRate)-e.gainAtSample(s))<1.e-5;
            }
        };
        const auto audio=UtauRenderer::render(plan.request);
        check(tag+"real_audio_rendered",audio.buffer.getNumSamples()>0&&audio.backend.contains("utau-resampler"));
        check(tag+(curved?"single_curved_ramp_in_audio":"single_straight_ramp_in_audio"),samples>100&&single&&straight);
        std::cout<<tag<<"checked_samples="<<samples<<std::endl;
        // Feed the exact rendered mix envelopes back through the real lane painter.
        std::vector<size_t> cursor(plan.spans.size());
        for(size_t i=0;i<plan.owners.size();++i)
        {
            const auto owner=plan.owners[i];auto& span=plan.spans[owner][cursor[owner]++];
            if(windows.contains(i)){span.mixEnvelope=windows[i];span.mixEnvelope.startSeconds+=plan.offsets[i];}
        }
        auto metadata=std::make_shared<std::vector<UtauNoteWaveform>>();
        for(size_t i=0;i<clip.notes.size()&&i<plan.spans.size();++i)
        {
            const auto& n=clip.notes[i];UtauNoteWaveform w;w.noteId=n.id;w.phonemes=plan.spans[i];w.startSeconds=n.startSeconds;
            w.audioHash=AudioEngine::utauNoteAudioHash(n);w.renderHash=AudioEngine::utauNoteRenderHash(n);
            w.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(n);w.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(track);
            metadata->push_back(w);
        }
        liveRoll.setUtauNoteWaveforms(metadata);model.dispatchPendingMessages();liveRoll.diagnosticRefresh();
        int displayed=0;bool sameShape=true;
        for(const auto& [id,text]:lyrics)for(const auto& span:liveRoll.diagnosticPhonemeSpans(id))
            for(const auto& ramp:span.mixEnvelope.ramps)if(ramp.crossfade){++displayed;sameShape&=ramp.smoothCurve==curved&&!ramp.equalPower;}
        check(tag+"live_lane_uses_current_audio_shape",displayed>0&&sameShape);
        if(clip.notes.size()==plan.spans.size())
            verifyRealCvvcEnvelopeDisplay(model,track,clip,plan.spans,output,tag+"wcsndm",check);
    }
    output.getChildFile("report.txt").replaceWithText(log);return ok;
}
}
