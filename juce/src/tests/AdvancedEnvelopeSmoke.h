#pragma once
#include "../backend/AdvancedEnvelope.h"
namespace hachi
{
inline bool MainComponent::diagnosticAdvancedEnvelope(const juce::File& folder)
{
    folder.createDirectory();stopTimer();bool ok=true;
    const auto check=[&](const char* name,bool value){ok=ok&&value;std::cout<<name<<'='<<value<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<3.e-4;};
    const auto bank=folder.getChildFile("bank");bank.createDirectory();
    {
        juce::AudioBuffer<float> samples(1,44100);
        for(int i=0;i<44100;++i)samples.setSample(0,i,.15f*(float)std::sin(juce::MathConstants<double>::twoPi*261.625565*i/44100));
        auto stream=bank.getChildFile("a.wav").createOutputStream();stream->setPosition(0);stream->truncate();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),44100,1,16,{},0));
        check("voice_fixture_written",writer&&writer->writeFromAudioSampleBuffer(samples,0,44100));
    }
    bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,100,-1000,100,20\n");
    juce::StringArray warnings;auto rows=SampleSettings::loadVoicebankOto(bank,warnings,true,true);
    check("oto_fixture_loaded",rows.size()==1);if(rows.empty())return false;
    auto annotated=rows[0];annotated.hasJieOto=true;annotated.jieOnsetMs=100;annotated.jieGlideMs=250;annotated.jieNucleusMs=750;annotated.mouClasses="CVVV";juce::String error;
    check("oto_regions_written",SampleSettings::updateMouOtoEntry(rows[0],annotated,error));backend::UtauRenderer::invalidateVoicebankCache();
    ProjectData data;TrackData track;track.id="track";track.name=juce::String::fromUTF8("OTO 尾段淡出");track.pitchAlgorithm=PitchAlgorithm::utau;track.utauMode=UtauMode::mou;track.voicebankDirectory=bank;
    ClipData clip;clip.id="clip";clip.startSeconds=1;clip.durationSeconds=clip.sourceDurationSeconds=5;
    for(int count=2;count<=4;++count)
    {NoteData note;note.id="n"+juce::String(count);note.label="a";note.startSeconds=(count-2)*1.4;note.durationSeconds=1.2;note.midiNote=60;note.utauFlags="g-10B25";note.utauOto=SampleSettings::noteOtoFromEntry(annotated,true);note.utauOto.classes=count==2?"CV":count==3?"CVV":"CVVV";note.amplitudeEnvelope={{-.1,-60,true},{-.095,0,true},{1.165,0,true},{1.2,-60}};clip.notes.push_back(note);}
    track.clips.push_back(clip);data.tracks.push_back(track);
    ProjectModel model;model.resetDocument(data);PianoRollComponent roll(model,strings);roll.setReadsVoicebankInBackground(false);roll.setFocusedTrack("track");roll.setFocusedClip("clip");roll.setSelectedNoteIds({"n2","n3","n4"});roll.diagnosticRefresh();
    for(int count=2;count<=4;++count)
    {
        const auto span=roll.diagnosticTailFadeSpan("n"+juce::String(count));const auto edges=roll.diagnosticRegionEdges((size_t)(count-2));
        check("last_region_matches_displayed_oto_boundary",span&&near(span->startSeconds+1+(count-2)*1.4,edges[(size_t)(count-2)])&&near(span->endSeconds,1.2));
    }
    backend::UtauSampleTiming zeroTail;zeroTail.hasRegions=true;zeroTail.mouClasses="CVVV";zeroTail.regionSeconds={.1,.2,.7,0};
    check("zero_length_tail_is_not_invented",!backend::UtauRenderer::tailFadeSpan(zeroTail,-.1,1.2,1.2,100,true,true));
    backend::UtauSampleTiming classicTiming;classicTiming.consonantSeconds=.2;
    check("short_classic_note_without_vowel_tail_is_unchanged",!backend::UtauRenderer::tailFadeSpan(classicTiming,-.1,.05,.05,100,false,false));
    classicTiming.preutteranceSeconds=.1;const auto pinnedClassic=backend::UtauRenderer::tailFadeSpan(classicTiming,-.2,1.2,1.2,100,false,false);
    check("pinned_pre_preserves_postbeat_classic_boundary",pinnedClassic&&near(pinnedClassic->startSeconds,.1));
    classicTiming.consonantSeconds=.05;const auto insideLead=backend::UtauRenderer::tailFadeSpan(classicTiming,-.2,1.2,1.2,100,false,false);
    check("pinned_pre_retimes_boundary_inside_lead",insideLead&&near(insideLead->startSeconds,-.1));
    zeroTail.regionSeconds={.1,.2,.5,.2};zeroTail.mouClasses={};const auto spelling=backend::UtauRenderer::tailFadeSpan(zeroTail,-.1,.04,.0,100,true,false);
    check("jie_spelling_uses_its_last_of_two_regions",spelling&&near(spelling->startSeconds,0));
    const auto original=model.contentFingerprint();
    check("batch_linear_applies_to_all_selected_notes",roll.applyTailFade(1)==3);
    check("batch_fade_is_one_undo",model.undo()&&model.contentFingerprint()==original&&!model.canUndo());model.redo();roll.diagnosticRefresh();
    check("base_envelope_and_flags_are_preserved",model.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==4&&model.snapshot().tracks[0].clips[0].notes[0].utauFlags=="g-10B25");
    const auto span=roll.diagnosticTailFadeSpan("n4");check("linear_fade_has_expected_quarter_levels",near(backend::tailFadeGain(1,.25,0,1),.75)&&near(backend::tailFadeGain(1,.75,0,1),.25));
    const std::vector<AmplitudeEnvelopePoint> attack{{-.1,-60,false},{-.085,0,false},{1.2,0,false}};
    const auto attackPicture=backend::tailFadePicture(attack,2,.8,1.2);
    check("drawing_keeps_original_pre_tail_interpolation",near(PianoRollComponent::diagnosticAmplitudeDbAt(attackPicture,-.095),PianoRollComponent::diagnosticAmplitudeDbAt(attack,-.095)));
    const std::vector<AmplitudeEnvelopePoint> shaped{{0,-6,true},{.2,0,false},{1.2,-6,false}};const auto layered=backend::tailFadePicture(shaped,2,0,1.2);
    const auto expected=backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(shaped,0))*backend::tailFadeGain(2,.15,0,1.2);
    check("drawing_anchors_tail_at_original_start_level",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(layered,.15)),expected));
    check("smooth_fade_has_expected_curve",near(backend::tailFadeGain(2,.25,0,1),.84375)&&near(backend::tailFadeGain(2,.75,0,1),.15625));
    check("fade_is_unity_before_tail_and_silent_after_end",near(backend::tailFadeGain(1,-.1,0,1),1)&&near(backend::tailFadeGain(2,1.1,0,1),0));
    if(span)
    {const auto picture=roll.diagnosticTailFadePicture("n4");check("effective_picture_uses_sparse_base_without_baking_samples",picture.size()>4&&model.snapshot().tracks[0].clips[0].notes[2].amplitudeEnvelope.size()==4);check("effective_picture_reaches_silence",picture.back().gainDb==-60);}
    auto current=model.snapshot();const auto plain=current.tracks[0].clips[0].notes[2];auto faded=plain;faded.utauTailFadeMode=2;
    check("fade_invalidates_mix_but_keeps_raw_note_cache",AudioEngine::utauNoteRenderHash(plain)!=AudioEngine::utauNoteRenderHash(faded)&&AudioEngine::utauNoteAudioHash(plain)==AudioEngine::utauNoteAudioHash(faded));
    check("smooth_mode_applies",roll.applyTailFade(2)==3);
    const auto saved=folder.getChildFile("advanced-envelope.hjpx");check("save_with_advanced_envelope",model.save(saved,error));ProjectModel loaded;check("load_with_advanced_envelope",loaded.load(saved,error));
    check("all_modes_oto_and_original_envelope_roundtrip",loaded.snapshot().tracks[0].clips[0].notes[2].utauTailFadeMode==2&&loaded.snapshot().tracks[0].clips[0].notes[2].utauOto.classes=="CVVV"&&loaded.snapshot().tracks[0].clips[0].notes[2].amplitudeEnvelope.size()==4);
    juce::MemoryBlock bytes;saved.loadFileAsData(bytes);auto old=juce::ValueTree::readFromData(bytes.getData(),bytes.getSize());for(auto t:old)for(auto c:t)for(auto n:c)if(n.hasType("Note"))n.removeProperty("utauTailFadeMode",nullptr);juce::MemoryOutputStream oldBytes;old.writeToStream(oldBytes);const auto oldFile=folder.getChildFile("legacy.hjpx");oldFile.replaceWithData(oldBytes.getData(),oldBytes.getDataSize());ProjectModel legacy;legacy.load(oldFile,error);
    check("old_projects_default_fade_off",legacy.snapshot().tracks[0].clips[0].notes[0].utauTailFadeMode==0);
    model.setNotesUtauJieSplit({"n4"},.1,.4,.8);roll.diagnosticRefresh();const auto manually=roll.diagnosticTailFadeSpan("n4");
    check("manual_boundary_moves_fade_start",manually&&near(manually->startSeconds,-.1+1.3*.8));
    const auto manualGuides=roll.otoRegionGuidesFor("n4");
    check("envelope_guides_follow_manual_region_boundaries",manualGuides.size()==4&&manually&&near(manualGuides.back().startSeconds,manually->startSeconds));
    model.resizeNote("n4",2.8,2.0);roll.diagnosticRefresh();const auto resizedSpan=roll.diagnosticTailFadeSpan("n4");
    check("resize_recalculates_tail_fade_from_oto",resizedSpan&&near(resizedSpan->endSeconds,2.0)&&resizedSpan->startSeconds>manually->startSeconds);
    roll.applyTailFade(0);roll.diagnosticRefresh();check("turning_off_restores_original_envelope",model.snapshot().tracks[0].clips[0].notes[2].amplitudeEnvelope.size()==4&&model.snapshot().tracks[0].clips[0].notes[2].utauTailFadeMode==0);
    auto ds=data;ds.tracks[0].voicebankDirectory=folder.getChildFile("ds-bank");ds.tracks[0].voicebankDirectory.createDirectory();ds.tracks[0].voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("# DS identity");model.resetDocument(ds);check("ds_without_oto_is_not_enabled",!model.setNotesTailFade({"n4"},1));
    auto classic=data;classic.tracks[0].utauMode=UtauMode::classic;model.resetDocument(classic);check("classic_oto_two_region_is_supported",model.setNotesTailFade({"n4"},1));roll.diagnosticRefresh();const auto classicSpan=roll.diagnosticTailFadeSpan("n4");check("classic_last_region_follows_scaled_consonant",classicSpan&&near(classicSpan->startSeconds,0));
    auto source=data;source.tracks[0].pitchAlgorithm=PitchAlgorithm::world;model.resetDocument(source);check("non_utau_notes_are_not_enabled",!model.setNotesTailFade({"n4"},1));
    // Compare actual mixer gain for one isolated note through the public renderer.
    backend::UtauRenderRequest request;request.voicebankDirectory=bank;request.fourRegion=true;request.consonantClasses=true;request.targetDurationSeconds=2.4;
    backend::UtauNoteRenderSpec spec;spec.alias="a";spec.startSeconds=.5;spec.durationSeconds=1.2;spec.oto=clip.notes[2].utauOto;request.notes={spec};
    const auto timing=backend::UtauRenderer::sampleTiming(bank,"a",60,100,true,true,&spec.oto);const auto actualSpan=timing?backend::UtauRenderer::tailFadeSpan(*timing,-.1,1.2,1.2,100,true,true):std::nullopt;
    check("real_render_tail_resolves",actualSpan.has_value());
    if(actualSpan)
    {
        std::array<double,3> probes{actualSpan->startSeconds-.025,actualSpan->startSeconds+(1.2-actualSpan->startSeconds)*.25,actualSpan->startSeconds+(1.2-actualSpan->startSeconds)*.75};
        std::array<std::array<float,3>,3> gains{};std::array<juce::AudioBuffer<float>,3> buffers;
        for(int mode=0;mode<3;++mode)
        {request.notes[0].tailFadeMode=mode;request.notePiece=[&](size_t,const juce::AudioBuffer<float>&,double,double,const std::function<float(double)>& gain,const std::function<float(double)>&){for(size_t i=0;i<3;++i)gains[(size_t)mode][i]=gain(probes[i]);};const auto result=backend::UtauRenderer::render(request);buffers[(size_t)mode]=result.buffer;check("actual_renderer_produces_audio",result.buffer.getNumSamples()>0&&result.buffer.getMagnitude(0,result.buffer.getNumSamples())>.001f);}
        check("actual_renderer_preserves_pre_tail_level",near(gains[0][0],gains[1][0])&&near(gains[0][0],gains[2][0]));
        check("actual_renderer_linear_matches_curve",near(gains[1][1]/gains[0][1],.75)&&near(gains[1][2]/gains[0][2],.25));
        check("actual_renderer_smooth_matches_curve",near(gains[2][1]/gains[0][1],.84375)&&near(gains[2][2]/gains[0][2],.15625));
        for(int mode=0;mode<3;++mode){auto stream=folder.getChildFile(mode==0?"01-original.wav":mode==1?"02-tail-linear.wav":"03-tail-smooth.wav").createOutputStream();juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),44100,1,24,{},0));if(writer)writer->writeFromAudioSampleBuffer(buffers[(size_t)mode],0,buffers[(size_t)mode].getNumSamples());}
    }
    project.resetDocument(data);project.dispatchPendingMessages();stopTimer();focusClip("clip");pianoRoll.setReadsVoicebankInBackground(false);pianoRoll.diagnosticRefresh();pianoRoll.setSelectedNoteIds({"n2","n3","n4"});pianoRoll.applyTailFade(2);project.dispatchPendingMessages();stopTimer();pianoRoll.diagnosticRefresh();setSize(1280,1000);resized();setEnvelopeLane(EnvelopeLane::amplitude);pianoViewport.setViewPosition(0,(int)pianoRoll.diagnosticYForMidi(64));
    check("advanced_button_fits_toolbar",advancedEnvelopeButton.isVisible()&&advancedEnvelopeButton.getWidth()==90
        &&getLocalBounds().contains(advancedEnvelopeButton.getBounds())
        &&!advancedEnvelopeButton.getBounds().intersects(envelopePresetMenu.getBounds())
        &&!advancedEnvelopeButton.getBounds().intersects(autoCrossfadeButton.getBounds()));
    check("editor_display_options_reachable_in_view_menu",diagnosticHasEditorViewItems());
    auto shot=createComponentSnapshot(getLocalBounds(),true,1.0f);juce::PNGImageFormat png;auto stream=folder.getChildFile("advanced-envelope.png").createOutputStream();check("advanced_envelope_preview_written",stream&&png.writeImageToStream(shot,*stream));
    return ok;
}
}
