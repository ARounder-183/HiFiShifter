#pragma once
#include "../MainComponent.h"
#include "../backend/McpServer.h"
#include "../backend/ChineseCvvcPhonemizer.h"
#include "../backend/NsfHifiganRenderer.h"
#include <iostream>

namespace hachi
{
inline bool MainComponent::diagnosticUtauSelectionScope(const juce::File& output)
{
    using namespace backend;
    stopTimer(); audio.stop(); bool ok=true; int checks=0;
    const auto check=[&](const juce::String& name,bool value)
    { ++checks;ok &= value;std::cout<<name<<"="<<value<<std::endl; };
    check("output",output.createDirectory().wasOk());
    pianoRoll.setReadsVoicebankInBackground(false);setSize(1400,1000);
    const auto pump=[&]
    {
       #if JUCE_WINDOWS
        MSG message{};
        for(int i=0;i<256&&PeekMessageW(&message,nullptr,0,0,PM_REMOVE);++i)
        {if(message.message==WM_QUIT){PostQuitMessage(static_cast<int>(message.wParam));break;}TranslateMessage(&message);DispatchMessageW(&message);}
       #endif
        project.dispatchPendingMessages();
    };
    const auto noteJson=[](const NoteData& n){return juce::JSON::toString(McpServer::noteJson(n),false);};
    for(int mode=0;mode<4;++mode)
    {
        ProjectData data;TrackData track;track.id="scope-track";track.trackType=TrackType::utau;
        track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
        track.utauMode=mode==1?UtauMode::jie:mode==2?UtauMode::mou:UtauMode::classic;
        track.chineseCvvc=mode==3;
        ClipData clip;clip.id="scope-clip";clip.durationSeconds=15;
        for(int i=0;i<5;++i)
        {NoteData n;n.id="n"+juce::String(i);n.startSeconds=i<3?i:10+(i-3)*.1;
         n.durationSeconds=.6;n.midiNote=60+i;n.label=i<3?"aa":"unknown-distant";clip.notes.push_back(n);}
        track.clips.push_back(clip);data.tracks.push_back(track);
        auto other=track;other.id="other-track";other.clips[0].id="other-clip";
        for(auto& n:other.clips[0].notes)n.id="other-"+n.id;
        data.tracks.push_back(other);
        project.resetDocument(data);project.dispatchPendingMessages();
        selectedTrackId=track.id;selectedClipId=clip.id;selectedNoteId="n0";
        pianoRoll.setFocusedTrack(track.id);refreshProjectControls();
        pianoRoll.setSelectedNoteIds({"n0","n2"});refreshSelectedNoteParameter();
        const auto before=project.snapshot();
        const auto untouched=[&]
        {
            const auto now=project.snapshot();
            for(int i:{1,3,4})if(noteJson(now.tracks[0].clips[0].notes[i])!=noteJson(before.tracks[0].clips[0].notes[i]))return false;
            for(int i=0;i<5;++i)if(noteJson(now.tracks[1].clips[0].notes[i])!=noteJson(before.tracks[1].clips[0].notes[i]))return false;
            return true;
        };
        noteConsonantVelocityEditor.setText("137",false);noteConsonantVelocityDirty=true;commitNoteConsonantVelocity();
        noteFlagsEditor.setText("g12",false);noteFlagsDirty=true;commitNoteFlags();
        flagCurveButton.setToggleState(true,juce::dontSendNotification);flagCurveButton.onClick();
        project.dispatchPendingMessages();
        const auto edited=project.snapshot();
        check(juce::String(mode)+"_velocity_flags_exact_noncontiguous_selection",untouched()
            &&edited.tracks[0].clips[0].notes[0].utauConsonantVelocity==137
            &&edited.tracks[0].clips[0].notes[2].utauFlags=="g12"
            &&edited.tracks[0].clips[0].notes[2].utauFlagCurveEnabled);
        pianoRoll.setSelectedNoteIds({"n0"});
        pianoRoll.diagnosticShowBatchLyrics("n0");
        auto* dialog=dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
        check("batch_dialog_opens",dialog!=nullptr);
        if(dialog){dialog->getTextEditor("lyrics")->setText("one two three");dialog->exitModalState(1);pump();}
        check(juce::String(mode)+"_single_selected_batch_does_not_fill_following_notes",untouched()
            &&project.snapshot().tracks[0].clips[0].notes[0].label=="one"
            &&project.snapshot().tracks[0].clips[0].notes[2].label=="aa");
        pianoRoll.setSelectedNoteIds({"n2","n0"});pianoRoll.diagnosticShowBatchLyrics("n0");
        dialog=dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
        if(dialog){dialog->getTextEditor("lyrics")->setText("first second excess");dialog->exitModalState(1);pump();}
        check(juce::String(mode)+"_batch_uses_time_order_and_skips_unselected",untouched()
            &&project.snapshot().tracks[0].clips[0].notes[0].label=="first"
            &&project.snapshot().tracks[0].clips[0].notes[2].label=="second");
        const auto stamp=project.contentFingerprint();pianoRoll.clearNoteSelection();
        check("empty_envelope_preset_does_not_edit",pianoRoll.applyEnvelopePreset(PianoRollComponent::envelopePresets().front())==0
            &&project.contentFingerprint()==stamp);
        if(mode!=3)
        {
            pianoRoll.setSelectedNoteIds({"n0","n2"});
            check("selected_envelope_preset",pianoRoll.applyEnvelopePreset(PianoRollComponent::envelopePresets().front())==2&&untouched());
            check("envelope_undo_keeps_other_notes",project.undo()&&project.contentFingerprint()==stamp);
        }
        std::vector<juce::String> scope;juce::String error;
        check("omitted_scope_uses_current",McpServer::resolveUtauRenderSelection(data,juce::JSON::parse("{}"),{"n0"},scope,error)&&scope==std::vector<juce::String>{"n0"});
        check("empty_scope_stays_empty",McpServer::resolveUtauRenderSelection(data,juce::JSON::parse(R"({"note_ids":[]})"),{"n0"},scope,error)&&scope.empty());
        check("headless_omitted_scope_is_empty",McpServer::resolveUtauRenderSelection(data,juce::JSON::parse("{}"),{},scope,error)&&scope.empty());
        check("invalid_scope_rejected",!McpServer::resolveUtauRenderSelection(data,juce::JSON::parse(R"({"note_ids":["deleted"]})"),{},scope,error));
        check("whole_scope_explicit",McpServer::resolveUtauRenderSelection(data,juce::JSON::parse(R"({"whole_project":true})"),{},scope,error)&&scope.size()==10);
        check("ambiguous_scope_rejected",!McpServer::resolveUtauRenderSelection(data,juce::JSON::parse(R"({"whole_project":true,"note_ids":[]})"),{},scope,error));
        const auto saved=output.getChildFile("mode-"+juce::String(mode)+".hjpx");ProjectModel reopened;
        check("selection_edits_roundtrip",project.save(saved,error)&&reopened.load(saved,error)
            &&noteJson(reopened.snapshot().tracks[0].clips[0].notes[1])==noteJson(before.tracks[0].clips[0].notes[1]));
    }
    {
        McpServer server(project,&audio);
        const auto empty=server.executeTool("utau_render_selection",juce::JSON::parse(R"({"note_ids":[]})"));
        check("headless_dispatch_empty_clears",!static_cast<bool>(empty["isError"])
            &&empty["content"][0]["text"].toString()=="note_ids=0");
        check("headless_dispatch_invalid_fails",static_cast<bool>(server.executeTool("utau_render_selection",
            juce::JSON::parse(R"({"note_ids":["deleted"]})"))["isError"]));
        startLiveMcp();pianoRoll.setSelectedNoteIds({"n0"});
        const auto call=[&](const char* args)
        {auto request=juce::JSON::parse(R"({"jsonrpc":"2.0","id":230,"method":"tools/call","params":{"name":"utau_render_selection"}})");
         auto arguments=juce::JSON::parse(args);
         arguments.getDynamicObject()->setProperty("expected_revision",static_cast<juce::int64>(project.revisionNumber()));
         request["params"].getDynamicObject()->setProperty("arguments",arguments);
         bool respond=false;return handleLiveMcp(request,respond)["result"];};
        check("live_dispatch_omitted_uses_one_GUI_note",!static_cast<bool>(call("{}")["isError"])&&activeUtauSelectionCount==1);
        check("live_dispatch_explicit_noncontiguous_IDs",!static_cast<bool>(call(R"({"note_ids":["n0","n2"]})")["isError"])&&activeUtauSelectionCount==2);
        check("live_dispatch_empty_clears",!static_cast<bool>(call(R"({"note_ids":[]})")["isError"])&&activeUtauSelectionCount==0);
        check("live_dispatch_invalid_does_not_expand",static_cast<bool>(call(R"({"note_ids":["deleted"]})")["isError"])&&activeUtauSelectionCount==0);
        stopLiveMcp();
    }
    const auto bank=output.getChildFile("bank");bank.createDirectory();
    {juce::AudioBuffer<float> pcm(1,44100);for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,.2f*std::sin(float(i)*.0313f));
     auto stream=bank.getChildFile("tone.wav").createOutputStream();juce::WavAudioFormat wav;
     std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),44100,1,16,{},0));
     check("wave_fixture",writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));}
    bank.getChildFile("oto.ini").replaceWithText("tone.wav=aa,0,100,-700,80,40\n");
    bank.getChildFile("presamp.ini").replaceWithText("[VOWEL]\na=A=aa=100\n[CONSONANT]\nb=ba=1\n");
    UtauRenderer::invalidateVoicebankCache();
    UtauRenderRequest request;request.requestedFamily=RequestedVoicebankFamily::utau;
    request.voicebankDirectory=bank;request.targetDurationSeconds=4;request.chineseCvvc=true;
    UtauNoteRenderSpec note;note.alias="aa";note.startSeconds=.2;note.durationSeconds=.5;note.midiNote=60;request.notes={note};
    const auto base=UtauRenderer::render(request);
    note.alias="unresolved-remote";note.startSeconds=3;note.gain=0;request.notes.push_back(note);
    note.startSeconds=3.1;request.notes.push_back(note);
    const auto local=UtauRenderer::render(request);
    const auto samePcm=[](const UtauRenderResult& a,const UtauRenderResult& b)
    {if(a.buffer.getNumChannels()!=b.buffer.getNumChannels()||a.buffer.getNumSamples()!=b.buffer.getNumSamples())return false;
     for(int c=0;c<a.buffer.getNumChannels();++c)for(int i=0;i<a.buffer.getNumSamples();++i)
        if(std::abs(a.buffer.getSample(c,i)-b.buffer.getSample(c,i))>1.e-6f)return false;return true;};
    check("cvvc_unselected_overlapping_bad_lyrics_do_not_warn_or_change_pcm",local.warning.isEmpty()&&samePcm(base,local)
        &&local.buffer.getMagnitude(0,local.buffer.getNumSamples())>.001f);
    request.notes.back().gain=1;
    check("cvvc_selected_bad_lyric_still_warns",UtauRenderer::render(request).warning.contains("Missing CV"));
    request.notes.back().gain=0;
    bank.getChildFile("dsconfig.yaml").replaceWithText("phonemes: phonemes.txt\n");
    check("mixed_bank_silent_unresolved_context_does_not_reject_selected_sample",UtauRenderer::render(request).warning.isEmpty());
    bank.getChildFile("dsconfig.yaml").deleteFile();
    // Both renderers can exercise actual PCM selection without a neural model:
    // a blank selected note uses the shared piano, unresolved context stays silent.
    request.notes[0].alias="";
    for(bool nsf:{false,true})
    {
        auto baseline=request;baseline.notes.resize(1);
        auto a=nsf?renderNsfUtauPhrase(baseline,{},{}):UtauRenderer::render(baseline);
        auto b=nsf?renderNsfUtauPhrase(request,{},{}):UtauRenderer::render(request);
        check(nsf?"hifisampler_silent_context_pcm_and_warning":"resampler_silent_context_pcm_and_warning",
            b.warning.isEmpty()&&samePcm(a,b)&&b.buffer.getNumSamples()>0);
    }
    std::cout<<"checks="<<checks<<" passed="<<ok<<std::endl;return ok;
}
}
