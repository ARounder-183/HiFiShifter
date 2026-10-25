#pragma once
#include "../MainComponent.h"
#include "../OttoMaterialIO.h"
#include <iostream>
#if JUCE_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef small
#undef small
#endif
#endif
namespace hachi
{
inline bool MainComponent::diagnosticProjectRegistrationUi(const juce::File& output)
{
    const auto root=output.getChildFile("main-registration-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;juce::String log,error;audio.stop();stopTimer();
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto finish=[&]{cancelProjectMaterialRegistration();projectRegistrationFolderPickerForTest={};projectRegistrationRootWriterForTest={};const auto written=root.getChildFile("main-project-registration-ui.txt").replaceWithText(log);std::cout<<"main_registration_fixture="<<root.getFullPathName()<<std::endl;return ok&&written;};
#if !JUCE_WINDOWS
    check("windows_real_message_pump",false);return finish();
#else
    const auto pump=[&]
    {
        MSG message{};for(int i=0;i<128&&PeekMessageW(&message,nullptr,0,0,PM_REMOVE);++i){if(message.message==WM_QUIT){PostQuitMessage(static_cast<int>(message.wParam));return false;}TranslateMessage(&message);DispatchMessageW(&message);}
        project.dispatchPendingMessages();return true;
    };
    const auto wait=[&](const std::function<bool()>& condition,int timeout)
    {const auto deadline=juce::Time::getMillisecondCounterHiRes()+timeout;while(!condition()&&juce::Time::getMillisecondCounterHiRes()<deadline){if(!pump())return false;juce::Thread::sleep(5);}pump();return condition();};
    const auto prompt=[]()->juce::AlertWindow*
    {auto* window=dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());return window&&window->getComponentID()=="project-material-registration-confirm"?window:nullptr;};
    const auto closeWarnings=[&]{for(int i=0;i<8;++i){auto* modal=juce::Component::getCurrentlyModalComponent();if(!modal)break;modal->exitModalState(0);pump();}};
    const auto existing=root.getChildFile("existing-material");existing.createDirectory();const auto source=existing.getChildFile("original.wav");
    juce::AudioBuffer<float> wave(1,48000);for(int i=0;i<48000;++i)wave.setSample(0,i,.12f*std::sin(juce::MathConstants<double>::twoPi*261.6255653*i/48000));
    auto stream=source.createOutputStream();juce::WavAudioFormat format;std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),48000,1,24,{},0):nullptr);
    check("physical_source_fixture",writer&&writer->writeFromAudioSampleBuffer(wave,0,48000));writer.reset();
    SampleRegionSetting row;row.name="existing";row.materialAudioId="already-registered";row.materialRegionId="existing-row";row.regionEndSeconds=1;row.role=NativeSegmentRole::vowel;
    OttoMaterialIO::Document document;document.audio={{row.materialAudioId,source.getFileName(),{row}}};check("existing_registration_fixture",OttoMaterialIO::saveDocument(existing.getChildFile("otto.ini"),document,error));if(!ok)return finish();
    ProjectData seed;seed.name="registration-gui";TrackData track;track.id="native-registration";track.pitchAlgorithm=PitchAlgorithm::world;
    for(int i=0;i<2;++i)
    {
        ClipData c;c.id=i?"unregistered-owner":"registered-owner";c.startSeconds=i;c.durationSeconds=c.sourceDurationSeconds=1;c.sourceFile=source;c.sourceTimeMap={{0,0},{1,1}};
        NoteData n;n.id=i?"unregistered-note":"registered-note";n.label="KEEP";n.durationSeconds=1;n.midiNote=n.sourceMidiCenter=60;n.sourcePitchMeasured=true;n.nativeRole=NativeSegmentRole::vowel;n.nativeSourceStartSeconds=0;n.nativeSourceEndSeconds=1;
        n.pitchControlPoints={{0,60,PitchCurveShape::linear},{1,60,PitchCurveShape::linear}};for(int j=0;j<=200;++j)n.contour.push_back({j*.005,0,0,true});
        if(!i){n.materialAudioId=row.materialAudioId;n.materialRegionId=row.materialRegionId;}
        c.notes={n};track.clips.push_back(c);
    }
    seed.tracks={track};ProjectModel fixture;fixture.resetDocument(seed);const auto input=root.getChildFile("input.hjpx");check("existing_HJPX_saved_fixture",fixture.save(input,error));
    const auto inputHash=juce::SHA256(input).toHexString(),sourceHash=juce::SHA256(source).toHexString();
    const auto firstLoaded=loadProjectFile(input);const auto firstPrompt=firstLoaded&&wait([&]{return prompt()!=nullptr;},30000);
    const auto autoTrace="first_auto_load="+juce::String(firstLoaded?1:0)+" prompt="+juce::String(firstPrompt?1:0)+" registration_active="+juce::String(projectRegistrationOperation?1:0)+" main_showing="+juce::String(isShowing()?1:0)
        +" modal="+(juce::Component::getCurrentlyModalComponent()?juce::Component::getCurrentlyModalComponent()->getComponentID():juce::String("none"))+"\n"+projectRegistrationTrace;
    log+=autoTrace+"\n";std::cout<<autoTrace<<std::endl;root.getChildFile("first-auto-offer-trace.txt").replaceWithText(autoTrace);
    check("real_Main_load_offers_optional_registration",firstLoaded&&firstPrompt);
    const auto afterInitialPump=project.snapshot();
    check("constructor_picker_notifications_do_not_rewrite_loaded_WORLD_track",afterInitialPump.tracks.size()==1&&afterInitialPump.tracks[0].pitchAlgorithm==PitchAlgorithm::world);
    const auto music=ProjectModel::registrationMusicSourceFingerprint(project.snapshot());
    const auto loaded=project.contentFingerprint();const auto loadedRevision=project.revisionNumber();
    if(auto* p=prompt())p->exitModalState(0);
    check("decline_keeps_successfully_loaded_project_no_undo_no_copy",wait([&]{return !projectRegistrationOperation;},5000)&&project.contentFingerprint()==loaded&&project.revisionNumber()==loadedRevision);
    std::function<void(const juce::File&)> folderResult;int folderRequests=0;
    projectRegistrationFolderPickerForTest=[&](auto callback){++folderRequests;folderResult=std::move(callback);};
    menuItemSelected(863,0);check("manual_retry_uses_same_confirmation",wait([&]{return prompt()!=nullptr;},30000));if(auto* p=prompt())p->exitModalState(1);
    check("YES_reaches_native_folder_result_handler",wait([&]{return bool(folderResult);},5000)&&folderRequests==1);
    if(folderResult)folderResult({});
    check("folder_cancel_no_refs_or_music_mutation",wait([&]{return !projectRegistrationOperation;},5000)&&project.contentFingerprint()==loaded&&project.revisionNumber()==loadedRevision);folderResult={};
    const auto destination=root.getChildFile("destination");destination.createDirectory();
    menuItemSelected(863,0);wait([&]{return prompt()!=nullptr;},30000);if(auto* p=prompt())p->exitModalState(1);wait([&]{return bool(folderResult);},5000);
    if(folderResult)folderResult(destination);check("registration_phase_jobs_finish",wait([&]{return !projectRegistrationOperation;},120000));folderResult={};
    const auto adopted=project.snapshot();const auto& kept=adopted.tracks[0].clips[0];const auto& moved=adopted.tracks[0].clips[1];
    check("mixed_scope_existing_registered_owner_untouched",kept.sourceFile==source&&kept.notes[0].materialAudioId==row.materialAudioId&&kept.notes[0].materialRegionId==row.materialRegionId);
    check("selected_unregistered_owner_copied_and_bound_once",moved.sourceFile!=source&&moved.sourceFile.existsAsFile()&&moved.notes[0].materialAudioId.isNotEmpty()&&moved.notes[0].materialRegionId.isNotEmpty()&&juce::SHA256(moved.sourceFile).toHexString()==sourceHash);
    check("refs_only_adopt_one_undo_and_exact_music_source_clock",project.revisionNumber()==loadedRevision+1&&ProjectModel::registrationMusicSourceFingerprint(adopted)==music&&moved.notes[0].label=="KEEP");
    check("original_project_not_automatically_saved",juce::SHA256(input).toHexString()==inputHash);
    std::vector<SampleRegionSetting> actualRows;check("generated_variant_has_confirmed_dash_not_lexical_inference",OttoMaterialIO::loadAudio(moved.sourceFile.getParentDirectory().getChildFile("otto.ini"),moved.sourceFile,actualRows,error,nullptr,moved.notes[0].materialAudioId)&&actualRows.size()==1&&actualRows[0].name=="-");
    check("root_registered_only_after_model_commit",assetManager&&assetManager->materialRoots().contains(moved.sourceFile.getParentDirectory().getFullPathName()));
    const auto saved=root.getChildFile("user-saved.hjpx");check("explicit_user_save_persists_new_refs",saveProjectTo(saved));
    check("already_bound_reopen_never_reprompts_or_recopies",loadProjectFile(saved)&&wait([&]{return !projectRegistrationOperation;},30000)&&prompt()==nullptr&&folderRequests==2);
    // Stale editing is a real UI-time project change, not a forged ticket.
    closeWarnings();check("reload_original_for_stale_case",loadProjectFile(input)&&wait([&]{return prompt()!=nullptr;},30000));if(auto* p=prompt())p->exitModalState(1);wait([&]{return bool(folderResult);},5000);
    if(folderResult)folderResult(destination);project.setTrackVolume(track.id,.75f);
    check("stale_snapshot_rejects_without_overwriting_live_user_edit",wait([&]{return !projectRegistrationOperation;},120000)&&project.snapshot().tracks[0].volume==.75f&&project.snapshot().tracks[0].clips[1].sourceFile==source);closeWarnings();folderResult={};
    check("reload_original_for_preference_failure",loadProjectFile(input)&&wait([&]{return prompt()!=nullptr;},30000));if(auto* p=prompt())p->exitModalState(1);wait([&]{return bool(folderResult);},5000);
    projectRegistrationRootWriterForTest=[](const juce::File&,juce::String& error){error="Injected preference persistence failure";return false;};
    if(folderResult)folderResult(destination);check("adopted_folder_permanent_despite_preference_failure",wait([&]{return !projectRegistrationOperation;},120000)&&projectRegistrationRetryRoots.size()==1);
    const auto retained=project.snapshot().tracks[0].clips[1].sourceFile;check("preference_failure_never_deletes_adopted_audio_OTTO",retained.existsAsFile()&&retained.getParentDirectory().getChildFile("otto.ini").existsAsFile());closeWarnings();
    const auto retainedState=project.contentFingerprint();const auto retainedRevision=project.revisionNumber();projectRegistrationRootWriterForTest={};menuItemSelected(864,0);
    check("preference_retry_does_not_copy_or_readopt",projectRegistrationRetryRoots.empty()&&project.contentFingerprint()==retainedState&&project.revisionNumber()==retainedRevision&&retained.existsAsFile());
    check("original_audio_and_project_bytes_unchanged",juce::SHA256(source).toHexString()==sourceHash&&juce::SHA256(input).toHexString()==inputHash);
    return finish();
#endif
}
}
