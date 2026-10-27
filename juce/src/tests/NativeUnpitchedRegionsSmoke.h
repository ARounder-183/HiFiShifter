#pragma once
#include "../NativeUnpitchedRegions.h"
#include "../NativeAudioClipboard.h"
#include "../Theme.h"
#include "../SampleSettings.h"
#include "../PianoRollComponent.h"
#include "../TimelineComponent.h"
#include "../AudioEngine.h"
#include "../backend/AnalysisService.h"
#include <iostream>

namespace hachi
{
inline bool runNativeUnpitchedRegionsSmoke(const juce::File& folder,
    const juce::File& recording = {}, const juce::File& models = {})
{
    folder.createDirectory(); bool ok = true; int checks = 0;
    const auto check = [&](const char* key, bool pass) { ok &= pass; ++checks; std::cout << key << '=' << pass << std::endl; };
    const auto near = [](double a, double b) { return std::abs(a-b) < 1.e-6; };
    const auto file = folder.getChildFile("voice-and-noise.wav");
    juce::AudioBuffer<float> pcm(1, 144000); juce::Random random(17);
    for (int i=0; i<pcm.getNumSamples(); ++i)
    {
        const auto t = i/48000.0;
        const auto voiced = (t>=.5 && t<1) || (t>=2 && t<2.5);
        pcm.setSample(0,i,voiced ? static_cast<float>(.2*std::sin(t*220*juce::MathConstants<double>::twoPi))
                                : .035f*(random.nextFloat()*2-1));
    }
    juce::WavAudioFormat wav; auto stream = file.createOutputStream(); if (!stream) return false;
    stream->setPosition(0); stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(),48000,1,24,{},0)); if (!writer) return false;
    stream.release(); check("fixture_written", writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples())); writer.reset();
    juce::MemoryBlock originalBytes; file.loadFileAsData(originalBytes);
    ClipData clip; clip.id="source"; clip.sourceFile=file; clip.startSeconds=1; clip.durationSeconds=clip.sourceDurationSeconds=3;
    for (int i=0; i<2; ++i)
    {
        NoteData n; n.id="voice"+juce::String(i); n.startSeconds=i==0?.5:2; n.durationSeconds=.5;
        n.consonantSeconds=0; n.midiNote=n.sourceMidiCenter=57; n.sourcePitchMeasured=true;
        n.contour={{0,0,0,true},{.5,0,0,true}}; clip.notes.push_back(n);
    }
    TrackData track; track.id="native"; track.pitchAlgorithm=PitchAlgorithm::world; track.normalizeVolume=false; track.clips={clip};
    ProjectData data; data.tracks={track}; ProjectModel model; model.replace(data);
    check("legacy_mark_three_gaps", model.markNativeUnpitchedRegions(clip.id)==3);
    const auto marked = model.snapshot();
    check("repeated_mark_is_noop", model.markNativeUnpitchedRegions(clip.id)==0);
    std::vector<juce::String> gaps; juce::String middle;
    for (const auto& n : marked.tracks[0].clips[0].notes) if (n.nativeUnpitched)
    {
        gaps.push_back(n.id); if (near(n.startSeconds,1)) middle=n.id;
    }
    check("all_gap_objects_unvoiced",gaps.size()==3 && std::all_of(marked.tracks[0].clips[0].notes.begin(),marked.tracks[0].clips[0].notes.end(),[](const auto& n)
        {return !n.nativeUnpitched || (n.pitchControlPoints.empty() && std::none_of(n.contour.begin(),n.contour.end(),[](const auto& p){return p.voiced;}));}));
    model.flattenNotePitch(gaps,true); check("flatten_does_not_invent_pitch",model.snapshot().tracks[0].clips[0].notes[2].pitchControlPoints.empty());
    check("pitch_drawing_rejected",!model.setNotePitchCurve(middle,{{0,72},{1,72}},true));
    const auto request=AudioEngine::diagnosticNativeRequest(marked.tracks[0].clips[0],track);
    check("renderer_receives_no_f0_in_noise_gap",std::all_of(request.targetMidi.begin()+240,request.targetMidi.begin()+360,[](float f){return f==0;}));
    auto refreshed=marked.tracks[0].clips[0]; backend::AnalysisService::applySourcePitch(refreshed,clip.notes);
    check("source_reanalysis_keeps_markers_unvoiced",std::all_of(refreshed.notes.begin(),refreshed.notes.end(),[](const auto& n)
        {return !n.nativeUnpitched || std::none_of(n.contour.begin(),n.contour.end(),[](const auto& p){return p.voiced;});}));
    const auto saved=folder.getChildFile("marked.hjpx"); juce::String error; ProjectModel reopened;
    check("project_saves_and_reopens",model.save(saved,error)&&reopened.load(saved,error));
    check("project_preserves_special_type",reopened.snapshot().tracks[0].clips[0].notes[2].nativeUnpitched);
    std::vector<SampleRegionSetting> rows;
    for(const auto& n:marked.tracks[0].clips[0].notes)
    {SampleRegionSetting row;row.name=n.label;row.regionStartSeconds=n.startSeconds;row.regionEndSeconds=n.startSeconds+n.durationSeconds;row.nativeUnpitched=n.nativeUnpitched;rows.push_back(row);}
    const auto sidecarSaved=SampleSettings::save(file,rows,error);
    const auto sidecarRows=SampleSettings::loadOrDerive(file,{});
    check("hjm_preserves_special_type",sidecarSaved
        && std::count_if(sidecarRows.begin(),sidecarRows.end(),[](const auto& r){return r.nativeUnpitched;})==3);
    // Fresh imports use the same complement, including files with no detected F0.
    auto fresh=data; fresh.tracks[0].clips[0].notes.clear(); ProjectModel imported; imported.replace(fresh);
    check("import_automatically_marks_gaps",imported.setClipAudioAnalysis(clip.id,clip.notes,fresh.tracks[0].clips[0])
        && imported.snapshot().tracks[0].clips[0].notes.size()==5);
    imported.replace(fresh);check("fully_unpitched_file_is_editable",imported.setClipAudioAnalysis(clip.id,{},fresh.tracks[0].clips[0])
        && imported.snapshot().tracks[0].clips[0].notes.size()==1 && imported.snapshot().tracks[0].clips[0].notes[0].nativeUnpitched);
    // The real keyboard delete path must remove PCM, not just an annotation.
    Palette::applyTheme("dark",juce::Colour(0xff846cca),juce::Colour(0xffcfc9ff),juce::Colour(0xffffbd78));
    I18n strings; PianoRollComponent roll(model,strings); roll.setFocusedTrack(track.id); roll.setFocusedClip(clip.id);
    roll.setPixelsPerSecond(200); roll.setRowHeight(28); roll.setSize(1000,560); roll.setTool(PianoRollComponent::Tool::note);
    roll.diagnosticRefresh(); roll.setSelectedNoteIds({middle});
    const auto screenshot=folder.getChildFile("unpitched-markers.png"); auto imageStream=screenshot.createOutputStream();
    if(imageStream){imageStream->setPosition(0);imageStream->truncate();juce::PNGImageFormat png;
        const auto at=roll.diagnosticHitBounds(2);const auto preview=juce::Rectangle<int>(0,static_cast<int>(at.getY())-72,1000,190);
        png.writeImageToStream(roll.createComponentSnapshot(preview, true, 1.0f),*imageStream);}
    roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey)); const auto removed=model.snapshot();
    check("delete_slices_playback_into_two_regions",removed.tracks[0].clips.size()==2);
    check("delete_keeps_neighbor_timing",removed.tracks[0].clips.size()==2
        && near(removed.tracks[0].clips[0].startSeconds,1) && near(removed.tracks[0].clips[0].durationSeconds,1)
        && near(removed.tracks[0].clips[1].startSeconds,3) && near(removed.tracks[0].clips[1].sourceOffsetSeconds,2));
    check("gap_cannot_be_readded_after_deletion",model.markNativeUnpitchedRegions(clip.id)==0);
    model.undo();check("undo_restores_source_audio",model.snapshot().tracks[0].clips.size()==1 && model.snapshot().tracks[0].clips[0].notes.size()==5);
    model.redo();check("redo_removes_gap_again",model.snapshot().tracks[0].clips.size()==2);
    const auto render=[&](const ProjectData& d,const char* name,juce::AudioBuffer<float>& output)
    {
        AudioEngine engine; engine.prepareToPlay(256,48000); engine.syncProject(d);
        for(int i=0;i<1000 && engine.renderProgress();++i)juce::Thread::sleep(10);
        WavExportOptions options;options.sampleRate=48000;options.channels=1;options.bitDepth=32;
        const auto dest=folder.getChildFile(name); if(!engine.hasCurrentRenderedAudio()||!engine.exportWav(dest,error,track.id,1,4,options))return false;
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> r(formats.createReaderFor(dest));if(!r)return false;
        output.setSize(1,static_cast<int>(r->lengthInSamples));return r->read(&output,0,output.getNumSamples(),0,true,false);
    };
    juce::AudioBuffer<float> before,after;
    check("exports_before_and_after_delete",render(marked,"before.wav",before)&&render(removed,"after.wav",after));
    if(before.getNumSamples()==144000 && after.getNumSamples()==144000)
    {
        check("original_noise_was_audible",before.getRMSLevel(0,57600,28800)>.01f);
        check("deleted_noise_is_silent",after.getRMSLevel(0,57600,28800)<1.e-7f);
        double difference=0;for(int i=6000;i<144000-6000;++i)if(i<42000||i>102000)
            difference=std::max(difference,static_cast<double>(std::abs(before.getSample(0,i)-after.getSample(0,i))));
        check("surrounding_source_pcm_unchanged",difference<1.e-5);
    }
    else check("export_length_preserved",false);
    // The ordinary source-backed clipboard and trim tool also work on gaps.
    model.replace(marked);const auto copied=copyNativeAudioNotes(marked,{middle});
    check("clipboard_keeps_noise_source_and_type",copied.clips.size()==1 && copied.clips[0].notes.size()==1
        && copied.clips[0].notes[0].nativeUnpitched && near(copied.clips[0].sourceOffsetSeconds,1));
    const auto pasted=model.insertNativeAudioClips(track.id,copied.clips,copied.connections,7);
    check("noise_fragment_pastes",!pasted.empty() && model.snapshot().tracks[0].clips.size()==2
        && model.snapshot().tracks[0].clips.back().notes[0].nativeUnpitched);
    model.replace(marked);check("noise_fragment_can_be_cropped",model.trimNativeNoteEdge(middle,-.25,false));
    bool unvoiced=true;for(const auto& c:model.snapshot().tracks[0].clips)for(const auto& n:c.notes)if(n.id==middle)
        unvoiced &= n.nativeUnpitched && std::none_of(n.contour.begin(),n.contour.end(),[](const auto& p){return p.voiced;});
    check("crop_preserves_unpitched_type",unvoiced);
    model.replace(marked);check("noise_loudness_envelope_editable",model.setNoteAmplitudeEnvelope(middle,{{0,-6},{1,-6}}));
    juce::AudioBuffer<float> reduced;check("edited_noise_envelope_exports",render(model.snapshot(),"quieter-noise.wav",reduced));
    check("noise_envelope_reduces_actual_pcm",reduced.getNumSamples()==144000 && before.getNumSamples()==144000
        && std::abs(reduced.getRMSLevel(0,57600,28800)/before.getRMSLevel(0,57600,28800)-.501187)<.01);
    // Nonlinear clocks and merged source parts must preserve source positions.
    auto warped=marked;warped.tracks[0].clips[0].sourceTimeMap={{0,0},{1,.7},{2,2.2},{3,3}};model.replace(warped);model.removeNotes({middle});
    auto cut=model.snapshot();check("stretched_gap_uses_actual_source_clock",cut.tracks[0].clips.size()==2
        && near(cut.tracks[0].clips[0].sourceDurationSeconds,.7) && near(cut.tracks[0].clips[1].sourceOffsetSeconds,2.2));
    auto merged=marked;auto child=merged.tracks[0].clips[0];child.id="child";child.startSeconds=0;child.notes.clear();merged.tracks[0].clips[0].parts={child};
    for(auto& n:merged.tracks[0].clips[0].notes)n.clipPartId="child";model.replace(merged);model.removeNotes({middle});
    check("merged_audio_gap_deletes_actual_span",model.snapshot().tracks[0].clips.size()==2);
    // Pitched notes must remove the same playback source as unpitched notes.
    // Exercise the keyboard entry point and an already-warm audio engine.
    model.resetDocument(marked);model.dispatchPendingMessages();roll.diagnosticRefresh();
    roll.setFocusedClip(clip.id);roll.setSelectedNoteIds({"voice0"});
    AudioEngine live;live.prepareToPlay(256,48000);live.syncProject(marked);
    for(int i=0;i<1000&&live.renderProgress();++i)juce::Thread::sleep(10);
    const auto sampleLive=[&](double seconds)
    {
        juce::AudioBuffer<float> block(2,512);block.clear();live.setPosition(seconds);live.play();
        juce::AudioSourceChannelInfo info(&block,0,512);live.getNextAudioBlock(info);live.stop();
        return block.getRMSLevel(0,0,512);
    };
    check("pitched_source_audible_before_keyboard_delete",sampleLive(1.75)>.01f);
    roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));const auto pitchedRemoved=model.snapshot();
    const auto audioAt=[](const ProjectData& d,double at)
    {for(const auto& t:d.tracks)for(const auto& c:t.clips)for(const auto& p:expandedClipParts(c))
        if(at>=p.startSeconds+p.audioStartSeconds&&at<p.startSeconds+p.audioStartSeconds+p.audioLength())return true;return false;};
    check("pitched_delete_removes_source_span_keeps_neighbors",!audioAt(pitchedRemoved,1.75)&&audioAt(pitchedRemoved,1.25)&&audioAt(pitchedRemoved,3.25));
    live.syncProject(pitchedRemoved);for(int i=0;i<1000&&live.renderProgress();++i)juce::Thread::sleep(10);
    check("warm_playback_cache_no_longer_sounds_deleted_note",sampleLive(1.75)<1.e-7f);
    juce::AudioBuffer<float> pitchedAfter;
    check("pitched_delete_export_silent_and_other_note_audible",render(pitchedRemoved,"pitched-deleted.wav",pitchedAfter)
        &&pitchedAfter.getNumSamples()==144000&&pitchedAfter.getRMSLevel(0,28800,14400)<1.e-7f
        &&pitchedAfter.getRMSLevel(0,100800,14400)>.01f);
    TimelineComponent timeline(model);timeline.setPixelsPerSecond(200);timeline.setSize(1000,220);
    timeline.diagnosticRefresh();(void)timeline.createComponentSnapshot(timeline.getLocalBounds());
    timeline.setSelectedClips({});const juce::Point<float> blank(timeline.pixelForSeconds(1.75),timeline.getRulerHeight()+45.f);
    const juce::MouseEvent click(juce::Desktop::getInstance().getMainMouseSource(),blank,juce::ModifierKeys::leftButtonModifier,
        0,0,0,0,0,&timeline,&timeline,juce::Time::getCurrentTime(),blank,juce::Time::getCurrentTime(),1,false);
    timeline.mouseDown(click);timeline.mouseUp(click);
    check("upper_timeline_deleted_span_has_no_clip_hit",timeline.selectedClipIds().empty());
    check("pitched_delete_one_undo_restores_audio",model.undo()&&!model.canUndo()&&audioAt(model.snapshot(),1.75));
    check("pitched_delete_redo_restores_silence",model.redo()&&!audioAt(model.snapshot(),1.75));
    ProjectModel deletedReload;
    check("pitched_delete_survives_save_reopen",model.save(folder.getChildFile("pitched-deleted.hjpx"),error)
        &&deletedReload.load(folder.getChildFile("pitched-deleted.hjpx"),error)&&!audioAt(deletedReload.snapshot(),1.75));
    auto pitchedWarp=marked;pitchedWarp.tracks[0].clips[0].sourceTimeMap={{0,0},{.5,.3},{1,1.2},{3,3}};
    model.resetDocument(pitchedWarp);model.removeNotes({"voice0"});
    auto warpCut=model.snapshot();bool correctOffset=false;
    for(const auto& c:warpCut.tracks[0].clips)if(near(c.startSeconds,2))correctOffset=near(c.sourceOffsetSeconds,1.2);
    check("pitched_delete_uses_stretched_source_clock",correctOffset&&!audioAt(warpCut,1.75));
    model.resetDocument(merged);model.removeNotes({"voice0"});
    check("pitched_delete_in_merged_source_removes_pcm_span",!audioAt(model.snapshot(),1.75)&&audioAt(model.snapshot(),3.25));
    auto overlapping=marked;auto keep=overlapping.tracks[0].clips[0].notes[1];keep.id="overlap-keep";
    keep.startSeconds=.75;keep.durationSeconds=.25;overlapping.tracks[0].clips[0].notes.push_back(keep);
    model.resetDocument(overlapping);model.removeNotes({"voice0"});
    check("shared_source_unselected_overlap_still_owns_audio",!audioAt(model.snapshot(),1.6)&&audioAt(model.snapshot(),1.85));
    model.resetDocument(marked);model.removeNotes({"voice0","voice1"});
    check("noncontiguous_pitched_delete_leaves_middle_audio",!audioAt(model.snapshot(),1.75)&&!audioAt(model.snapshot(),3.25)&&audioAt(model.snapshot(),2.5));
    model.resetDocument(marked);model.removeNotesRippling({"voice0"});const auto rippled=model.snapshot();
    bool neighborMoved=false,sourceMoved=false;
    for(const auto& c:rippled.tracks[0].clips)
    {
        for(const auto& n:c.notes)if(n.id=="voice1")neighborMoved=near(c.startSeconds+n.startSeconds,2.5);
        if(near(c.startSeconds,1.5))sourceMoved=near(c.sourceOffsetSeconds,1)&&near(c.durationSeconds,2);
    }
    check("ripple_delete_moves_source_and_notes_together",neighborMoved&&sourceMoved);
    check("ripple_delete_single_undo_restores_both",model.undo()&&!model.canUndo()&&model.snapshot().tracks[0].clips.size()==1);
    juce::AudioBuffer<float> rippleAfter;
    bool ripplePcm=render(rippled,"pitched-ripple-deleted.wav",rippleAfter)&&rippleAfter.getNumSamples()==120000;
    if(ripplePcm&&before.getNumSamples()==144000)
        for(int i=78000;i<90000;++i)ripplePcm &= std::abs(rippleAfter.getSample(0,i)-before.getSample(0,i+24000))<1.e-5f;
    check("ripple_delete_audio_matches_shifted_retained_source",ripplePcm);
    model.resetDocument(marked);model.removeNotesRippling({"voice0","voice1"});
    const auto noncontiguousRipple=model.snapshot();
    bool keptGap=false;for(const auto& c:noncontiguousRipple.tracks[0].clips)for(const auto& n:c.notes)
        if(n.id==middle)keptGap=near(n.durationSeconds,1)&&near(c.startSeconds+n.startSeconds,1.5);
    check("noncontiguous_ripple_preserves_unselected_middle",keptGap);
    std::vector<juce::String> all;for(const auto& n:marked.tracks[0].clips[0].notes)all.push_back(n.id);
    model.resetDocument(marked);model.removeNotes(all);check("delete_all_notes_removes_all_source_audio",model.snapshot().tracks[0].clips.empty());
    for(const auto type:{TrackType::utau,TrackType::diffSinger})
    {auto score=marked;score.tracks[0].trackType=type;score.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;
        model.resetDocument(score);model.removeNotes({"voice0"});const auto d=model.snapshot();
        check("voicebank_score_delete_does_not_split_regions",d.tracks[0].clips.size()==1&&d.tracks[0].clips[0].notes.size()==4);}
    auto u=data;u.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;model.replace(u);check("utau_has_no_gap_marking",model.markNativeUnpitchedRegions(clip.id)==0);
    juce::MemoryBlock currentBytes; file.loadFileAsData(currentBytes);
    check("original_audio_file_untouched",currentBytes==originalBytes);
    if (recording.existsAsFile())
    {
        backend::AnalysisConfig config;
        if (models.isDirectory()) { config.gameModelDirectory=models.getChildFile("game/medium"); config.fcpeModelPath=models.getChildFile("fcpe/fcpe.onnx"); }
        config.inference=backend::InferenceBackend::cpu;
        auto analysis=backend::AnalysisService::analyse(recording,config,error);
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(recording));
        if(!reader) return false;
        auto real=clip;real.id="real";real.sourceFile=recording;real.startSeconds=0;real.durationSeconds=real.sourceDurationSeconds=reader->lengthInSamples/reader->sampleRate;
        real.sourceTimeMap.clear();real.notes=analysis.notes;
        const auto count=appendNativeUnpitchedRegions(real);
        check("real_recording_has_editable_uncovered_regions",!analysis.notes.empty() && count>0);
        bool clock=true;double seconds=0;
        for(const auto& n:real.notes)if(n.nativeUnpitched)
        { seconds+=n.durationSeconds;clock &= near(n.nativeSourceStartSeconds,n.startSeconds)&&near(n.nativeSourceEndSeconds,n.startSeconds+n.durationSeconds); }
        check("real_recording_marker_source_clock_matches",clock);
        std::cout<<"real_backend="<<analysis.status.activeBackend<<" real_pitched_notes="<<analysis.notes.size()
                 <<" real_unpitched_regions="<<count<<" real_unpitched_seconds="<<seconds<<std::endl;
    }
    std::cout<<"checks="<<checks<<" passed="<<ok<<std::endl; return ok;
}
}
