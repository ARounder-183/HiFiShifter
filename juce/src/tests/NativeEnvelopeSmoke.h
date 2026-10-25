#pragma once
#include "../AdvancedEnvelopePanel.h"
#include "../NativeAudioTrim.h"
#include "../NativeAudioOverlap.h"
namespace hachi
{
inline bool MainComponent::diagnosticNativeEnvelope(const juce::File& folder)
{
    folder.createDirectory();stopTimer();bool ok=true;juce::Array<juce::var> checks;
    const auto check=[&](const char* name,bool passed){ok&=passed;auto* o=new juce::DynamicObject();o->setProperty("name",name);o->setProperty("passed",passed);checks.add(juce::var(o));std::cout<<name<<'='<<passed<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<2.e-4;};
    juce::AudioBuffer<float> source(1,48000);for(int i=0;i<48000;++i)source.setSample(0,i,.2f*(float)std::sin(juce::MathConstants<double>::twoPi*220*i/48000));
    const auto file=folder.getChildFile("source.wav");{auto out=file.createOutputStream();if(out){out->setPosition(0);out->truncate();}juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(out.release(),48000,1,24,{},0));check("native_envelope_source_written",writer&&writer->writeFromAudioSampleBuffer(source,0,48000));}
    ClipData clip;clip.id="native-clip";clip.startSeconds=0;clip.durationSeconds=clip.sourceDurationSeconds=1;clip.sourceFile=file;
    NoteData note;note.id="native-note";note.label="audio";note.durationSeconds=1;note.sourceMidiCenter=note.midiNote=57;note.sourcePitchMeasured=true;note.utauAutoPitchTransition=false;
    for(int i=0;i<=200;++i)note.contour.push_back({i*.005,0,0,true});bindNativeNoteSource(note,clip);clip.notes={note};
    TrackData track;track.id="native-track";track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.compose=true;track.normalizeVolume=false;track.clips={clip};
    ProjectData data;data.tracks={track};ProjectModel model;model.resetDocument(data);
    auto settings=backend::NativeEnvelopeSettings{};check("native_envelope_default_unity",settings.mode==0&&near(backend::nativeEnvelopeGain(settings,0,1),1)&&near(backend::nativeEnvelopeGain(settings,1,1),1));
    const auto originalRequest=AudioEngine::diagnosticNativeRequest(clip,track);
    check("native_default_request_unity",std::all_of(originalRequest.noteGain.begin(),originalRequest.noteGain.end(),[](float g){return g==1;}));
    check("native_default_source_preserved",backend::canPreserveNativeSource(originalRequest,48000,48000,48000));
    settings.mode=2;settings.shape.startGain=.2;settings.shape.endGain=.3;settings.shape.knots={{.3,1,true},{.7,.4,false}};settings.shape.knots[0].bezier={.1,.6,.7,.95};
    const auto before=model.contentFingerprint();check("native_envelope_apply",model.setNotesNativeEnvelope({note.id},settings));
    check("native_envelope_undo",model.undo()&&model.contentFingerprint()==before);model.redo();
    auto edited=model.snapshot();const auto& editedClip=edited.tracks[0].clips[0];
    const auto request=AudioEngine::diagnosticNativeRequest(editedClip,edited.tracks[0]);
    check("native_gain_request_uses_envelope",near(request.noteGain[80],backend::nativeEnvelopeGain(settings,.4,1)));
    check("native_envelope_does_not_force_vocoder",backend::canPreserveNativeSource(request,48000,48000,48000));
    check("native_envelope_cache_changes",AudioEngine::utauNoteRenderHash(note)!=AudioEngine::utauNoteRenderHash(editedClip.notes[0]));
    juce::String error;check("native_envelope_project_saved",model.save(folder.getChildFile("native-envelope.hjpx"),error));ProjectModel loaded;
    check("native_envelope_project_reopens",loaded.load(folder.getChildFile("native-envelope.hjpx"),error)&&loaded.snapshot().tracks[0].clips[0].notes[0].nativeEnvelope==settings);
    auto invalid=settings;invalid.shape.knots[0].gain=3;const auto validBefore=model.contentFingerprint();check("native_invalid_parameters_rejected",!model.setNotesNativeEnvelope({note.id},invalid)&&model.contentFingerprint()==validBefore);
    const auto splitId=model.splitNote(note.id,.43);const auto split=model.snapshot().tracks[0].clips[0];bool sliced=splitId.isNotEmpty()&&split.notes.size()==2;
    for(const auto& n:split.notes)for(int i=0;i<=40;++i){const auto t=n.durationSeconds*i/40.;sliced&=n.nativeEnvelope.valid()&&near(backend::nativeEnvelopeGain(n.nativeEnvelope,t,n.durationSeconds),backend::nativeEnvelopeGain(settings,n.startSeconds+t,1));}
    check("native_split_keeps_cubic_shape",sliced);model.resetDocument(edited);
    check("native_trim_applies",model.trimNativeNoteEdge(note.id,.271,true));const auto trimmed=model.snapshot().tracks[0].clips[0];const auto tn=trimmed.notes[0];bool trimMatches=tn.nativeEnvelope.valid();
    for(int i=0;i<=40;++i){const auto t=tn.durationSeconds*i/40.;trimMatches&=near(backend::nativeEnvelopeGain(tn.nativeEnvelope,t,tn.durationSeconds),backend::nativeEnvelopeGain(settings,.271+t,1));}
    check("native_trim_keeps_cubic_shape",trimMatches);
    const auto cropped=cropNativeAudioHead(editedClip,.271);bool overlapMatches=cropped&&cropped->notes.size()==1;
    if(overlapMatches)for(int i=0;i<=40;++i){const auto& n=cropped->notes[0];const auto t=n.durationSeconds*i/40.;overlapMatches&=n.nativeEnvelope.valid()&&near(backend::nativeEnvelopeGain(n.nativeEnvelope,t,n.durationSeconds),backend::nativeEnvelopeGain(settings,.271+t,1));}
    check("native_overlap_crop_keeps_cubic_shape",overlapMatches);
    auto stretched=editedClip;stretched.durationSeconds=2;stretched.notes[0].durationSeconds=2;
    check("native_envelope_follows_stretch",near(backend::nativeEnvelopeGain(stretched.notes[0].nativeEnvelope,.8,2),backend::nativeEnvelopeGain(settings,.4,1)));
    project.resetDocument(edited);project.dispatchPendingMessages();diagnosticSelectTrack(track.id);pianoRoll.setFocusedClip(clip.id);pianoRoll.setSelectedNoteIds({note.id});pianoRoll.diagnosticRefresh();refreshProjectControls();setSize(1600,1000);
    check("native_envelope_button_visible",advancedEnvelopeButton.isVisible()&&advancedEnvelopeButton.getWidth()>0);
    const auto picture=pianoRoll.diagnosticTailFadePicture(note.id);
    check("native_editor_preview_matches_audio",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(picture,.4)),backend::nativeEnvelopeGain(settings,.4,1)));
    showAdvancedEnvelopeMenu();auto* window=dynamic_cast<juce::DialogWindow*>(juce::ModalComponentManager::getInstance()->getModalComponent(0));
    auto* panel=window?dynamic_cast<AdvancedEnvelopePanel*>(window->getContentComponent()):nullptr;check("native_editor_dialog_opens",panel!=nullptr);
    if(panel)
    {
        const auto prior=project.contentFingerprint();panel->diagnosticSegment(1,true);panel->diagnosticCurveControl(1,40);
        check("native_editor_is_draft",project.contentFingerprint()==prior);panel->setSize(740,660);check("native_editor_controls_fit",panel->diagnosticControlsFit());
        {auto out=folder.getChildFile("native-envelope-panel.png").createOutputStream();juce::PNGImageFormat png;check("native_editor_snapshot",out&&png.writeImageToStream(panel->createComponentSnapshot(panel->getLocalBounds(),true,1.5f),*out));}
        panel->diagnosticApply();check("native_editor_apply_changes_only_native_settings",near(project.snapshot().tracks[0].clips[0].notes[0].nativeEnvelope.shape.knots[0].bezier.control1Progress,.4)&&project.snapshot().tracks[0].clips[0].notes[0].utauTailFadeMode==0);
        window->exitModalState(0);
    }
    project.resetDocument(data);project.dispatchPendingMessages();diagnosticSelectTrack(track.id);pianoRoll.setFocusedClip(clip.id);pianoRoll.setSelectedNoteIds({note.id});refreshProjectControls();
    closeEnvelopeLanes();const auto unchanged=project.contentFingerprint();advancedEnvelopeButton.onClick();
    auto* detailedWindow=dynamic_cast<juce::DialogWindow*>(juce::ModalComponentManager::getInstance()->getModalComponent(0));
    check("native_advanced_button_opens_precise_shape_editor",detailedWindow
        &&dynamic_cast<AdvancedEnvelopePanel*>(detailedWindow->getContentComponent())!=nullptr
        &&advancedEnvelopeButton.getButtonText()==juce::String::fromUTF8("高级包络"));
    if(detailedWindow)detailedWindow->exitModalState(0);
    if(pianoRoll.onOpenNativeEnvelope)pianoRoll.onOpenNativeEnvelope();refreshProjectControls();
    check("native_loudness_context_opens_lane",openEnvelopeLane()==EnvelopeLane::amplitude&&!advancedEnvelopeButton.getToggleState());
    {auto out=folder.getChildFile("native-loudness-button.png").createOutputStream();if(out){out->setPosition(0);out->truncate();juce::PNGImageFormat().writeImageToStream(advancedEnvelopeButton.createComponentSnapshot(advancedEnvelopeButton.getLocalBounds(),true,3.f),*out);}}
    check("opening_loudness_does_not_edit_audio",project.contentFingerprint()==unchanged);
    setEnvelopeLane(EnvelopeLane::none);check("native_loudness_context_closes_lane",openEnvelopeLane()==EnvelopeLane::none);
    auto laneData=data;auto second=clip;second.id="second-clip";second.startSeconds=1.5;second.notes[0].id="second-note";second.durationSeconds=second.sourceDurationSeconds=second.notes[0].durationSeconds=.5;
    bindNativeNoteSource(second.notes[0],second);laneData.tracks[0].clips.push_back(second);
    ProjectModel laneModel;laneModel.resetDocument(laneData);I18n laneStrings;PianoRollComponent roll(laneModel,laneStrings);
    roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);roll.setSize(1200,600);roll.setPixelsPerSecond(400);roll.setTool(PianoRollComponent::Tool::amplitude);roll.diagnosticRefresh();
    const auto refresh=[&]{laneModel.dispatchPendingMessages();roll.diagnosticRefresh();};
    const auto event=[&](juce::Point<float> at,juce::Point<float> down,bool right=false){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(right?juce::ModifierKeys::rightButtonModifier:juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
    const auto y=roll.diagnosticAmplitudeLaneY(0);const juce::Point<float> middle(58+400*.5f,y);
    roll.mouseDoubleClick(event(middle,middle));refresh();
    check("native_lane_double_click_adds_point",laneModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==3);
    const auto half=roll.diagnosticAmplitudeLaneY(-6.0206f);const juce::Point<float> lowered(middle.x,half);
    roll.mouseDown(event(middle,middle));roll.mouseDrag(event(lowered,middle));roll.mouseUp(event(lowered,middle));refresh();
    const auto shaped=laneModel.snapshot().tracks[0].clips[0].notes[0];
    check("native_lane_drag_changes_loudness",shaped.amplitudeEnvelope.size()==3&&near(backend::envelopeGainFromDb(shaped.amplitudeEnvelope[1].gainDb),.5));
    check("native_lane_edits_base_not_advanced",shaped.nativeEnvelope.mode==0&&near(AudioEngine::diagnosticNativeRequest(laneModel.snapshot().tracks[0].clips[0],track).noteGain[100],.5));
    const auto laneEdited=laneModel.snapshot();const auto laneFingerprint=laneModel.contentFingerprint();check("native_lane_undo_redo",laneModel.undo()&&laneModel.redo()&&laneModel.contentFingerprint()==laneFingerprint);refresh();
    roll.mouseDown(event(lowered,lowered,true));refresh();check("native_lane_right_click_deletes_point",laneModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==2);
    const juce::Point<float> secondMiddle(58+400*1.75f,y);roll.mouseDoubleClick(event(secondMiddle,secondMiddle));refresh();
    check("native_lane_edits_other_visible_region",laneModel.snapshot().tracks[0].clips[1].notes[0].amplitudeEnvelope.size()==3);
    roll.setSelectedNoteIds({note.id,"second-note"});roll.mouseDoubleClick(event(middle,middle));refresh();
    check("native_lane_multiselect_maps_duration",laneModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==3&&laneModel.snapshot().tracks[0].clips[1].notes[0].amplitudeEnvelope.size()==3&&near(laneModel.snapshot().tracks[0].clips[1].notes[0].amplitudeEnvelope[1].timeSeconds,.25));
    laneModel.resetDocument(laneEdited);refresh();
    {auto out=folder.getChildFile("native-loudness-lane.png").createOutputStream();if(out){out->setPosition(0);out->truncate();}juce::PNGImageFormat png;check("native_loudness_lane_snapshot",out&&png.writeImageToStream(roll.createComponentSnapshot(roll.getLocalBounds()),*out));}
    auto rising=settings;rising.mode=1;rising.shape.mixed=false;rising.shape.startGain=0;rising.shape.endGain=1;
    AdvancedEnvelopePanel nativePreset({{note.id,"audio",{0,1},{{0,0,true},{1,0,true}}, {}}},1,1,rising.shape,false,folder.getChildFile("native-presets-"+juce::Uuid().toString()+".json"),true);
    check("native_linear_fade_in_allowed",nativePreset.values().valid()&&near(backend::tailFadeGain(1,.5,0,1,nativePreset.values()),.5));
    check("native_envelope_preset_saved",nativePreset.diagnosticSavePreset("Native fade in"));nativePreset.setValues(0,backend::NativeEnvelopeSettings{}.shape);nativePreset.diagnosticPreset(100);
    check("native_envelope_preset_recalled",nativePreset.mode()==1&&nativePreset.values()==rising.shape);
    auto utau=data;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;model.resetDocument(utau);check("utau_not_changed_by_native_action",!model.setNotesNativeEnvelope({note.id},settings));
    AudioEngine engine(false);engine.prepareToPlay(256,48000);WavExportOptions options;options.sampleRate=48000;options.channels=2;options.bitDepth=32;
    const auto render=[&](const ProjectData& d,const char* name){engine.syncProject(d);for(int i=0;i<1500&&engine.renderProgress();++i)juce::Thread::sleep(10);return engine.hasCurrentRenderedAudio()&&engine.exportWav(folder.getChildFile(name),error,track.id,0,1,options);};
    check("native_envelope_wav_exports",render(data,"original.wav")&&render(edited,"edited.wav"));
    juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> a(formats.createReaderFor(folder.getChildFile("original.wav"))),b(formats.createReaderFor(folder.getChildFile("edited.wav")));
    bool levels=a&&b;double energy=0,diff=0;if(levels){juce::AudioBuffer<float> x(2,48000),y(2,48000);a->read(&x,0,48000,0,true,true);b->read(&y,0,48000,0,true,true);
        for(int i=480;i<47520;++i){const auto expected=x.getSample(0,i)*backend::nativeEnvelopeGain(settings,i/48000.,1);const auto e=expected-y.getSample(0,i);diff+=e*e;energy+=expected*expected;}levels=energy>1&&diff/energy<2.e-4;}
    check("native_export_gain_matches_bezier",levels);engine.releaseResources();
    engine.prepareToPlay(256,48000);check("native_loudness_lane_wav_exports",render(laneEdited,"loudness.wav"));
    std::unique_ptr<juce::AudioFormatReader> loudness(formats.createReaderFor(folder.getChildFile("loudness.wav")));bool laneLevels=a&&loudness;energy=diff=0;
    if(laneLevels){juce::AudioBuffer<float> x(2,48000),yBuffer(2,48000);a->read(&x,0,48000,0,true,true);loudness->read(&yBuffer,0,48000,0,true,true);
        for(int i=480;i<47520;++i){const auto t=i/48000.;const auto expected=x.getSample(0,i)*(t<.5?1-t:t);const auto e=expected-yBuffer.getSample(0,i);diff+=e*e;energy+=expected*expected;}laneLevels=energy>1&&diff/energy<2.e-4;}
    check("native_loudness_lane_export_matches_points",laneLevels);engine.releaseResources();
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),false));return ok;
}
}
