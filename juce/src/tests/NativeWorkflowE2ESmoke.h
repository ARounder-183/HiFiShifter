#pragma once
#include "../MainComponent.h"
#include "../OtoWaveformEditorComponent.h"
#include "../OttoMaterialIO.h"
#include "../backend/AnalysisService.h"
#include "../backend/NsfHifiganRenderer.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#if JUCE_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// Legacy Win16/RPC macros must not leak into the other smoke headers that
// share Main.cpp (many correctly use a local lambda named `near`).
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
// Real Main/manager/editor/import/analyser/ProjectModel/AudioEngine workflow.
// Main's normal constructor may open the OS output driver. This test never
// auditions or starts playback; its two export engines explicitly open no device.
// The generated harmonic stereo fixture is NOT a real-project backend reference.
inline bool MainComponent::diagnosticNativeWorkflow(const juce::File& output,const juce::File& nsfModel)
{
    const auto root=output.getChildFile("native-workflow-"+juce::Uuid().toString());
    if(!root.createDirectory().wasOk())return false;
    bool ok=true;int total=0;juce::String log,error;juce::Array<juce::var> checks;
    std::vector<juce::Component::SafePointer<juce::DialogWindow>> openedWindows;
    const auto check=[&](const juce::String& name,bool pass,const juce::String& detail=juce::String{})
    {
        ++total;ok&=pass;
        log+=name+"="+(pass?"PASS":"FAIL")+(detail.isEmpty()?juce::String{}:" | "+detail)+"\n";
        auto* record=new juce::DynamicObject();record->setProperty("name",name);record->setProperty("passed",pass);
        record->setProperty("detail",detail);checks.add(juce::var(record));
        std::cout<<name<<"="<<(pass?"PASS":"FAIL")<<" "<<detail<<std::endl;
    };
    const auto closeWindow=[](juce::Component::SafePointer<juce::DialogWindow> window)
    {
        if(window==nullptr)return;
        window->setVisible(false);
        if(window!=nullptr&&window->isCurrentlyModal())window->exitModalState(0);
    };
    const auto finish=[&]
    {
        for(auto window:openedWindows)closeWindow(window);
        audio.stop();stopTimer();
        auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("total",total);
        report->setProperty("fixtureKind","generated stereo harmonic tone; NOT a real-project reference");
        report->setProperty("mainConstructorMayOpenDriver",true);report->setProperty("playbackStartedByTest",false);
        report->setProperty("checks",checks);
        const bool textWritten=root.getChildFile("verification.txt").replaceWithText(log);
        const bool jsonWritten=root.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
        std::cout<<"native_workflow_passed="<<(ok?1:0)<<"|checks="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;
        return ok&&textWritten&&jsonWritten;
    };
    check("message_thread_required",juce::MessageManager::getInstance()->isThisTheMessageThread());
    const auto isolated=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SETTINGS_DIR",{});
    check("isolated_settings_environment_required",isolated.isNotEmpty()&&isolated!="HachiShifterNext",isolated);
    check("main_preferences_available",preferences!=nullptr);
    check("actual_nsf_model_available",backend::NsfHifiganRenderer::modelAvailable(nsfModel),nsfModel.getFullPathName());
#if !JUCE_WINDOWS
    check("bounded_message_pump_supported",false,"This diagnostic uses the Windows test-only message pump, not JUCE modal loops.");
    return finish();
#else
    if(!ok)return finish();
    audio.stop();stopTimer();pianoRoll.setReadsVoicebankInBackground(false);
    const auto previousRoots=SampleSettings::materialRoots();
    struct RestoreRoots
    {
        juce::StringArray before;
        ~RestoreRoots(){SampleSettings::setMaterialRoots(before);}
    } restoreRoots{previousRoots};
    // The caller must launch a dedicated diagnostic Main under the isolated
    // settings directory. No user material folder is registered by this test.
    preferences->setValue("assets.materialFolders",juce::String{});
    preferences->setValue("ui.language",static_cast<int>(I18n::Language::enUS)+1);
    preferences->setValue("algorithm.hifiganPath",nsfModel.getFullPathName());
    preferences->setValue("algorithm.inference",2);preferences->setValue("algorithm.device",1);
    preferences->setValue("algorithm.gameModel","medium");preferences->setValue("ui.drawAction","pitch-curve");
    auto config=backend::AnalysisService::configFromProperties(preferences.get());
    auto status=backend::AnalysisService::status(config);
    if(!status.gameModelReady||!status.fcpeModelReady)
    {
        // Resolve model siblings supplied with the explicit NSF model bundle;
        // these are real model paths, never fabricated analysis or fake F0.
        for(const auto& candidate:{nsfModel.getParentDirectory(),nsfModel,nsfModel.getParentDirectory().getParentDirectory().getChildFile("models")})
        {
            if(!status.gameModelReady&&candidate.getChildFile("game").isDirectory())
                preferences->setValue("algorithm.gamePath",candidate.getChildFile("game").getFullPathName());
            if(!status.fcpeModelReady&&candidate.getChildFile("fcpe/fcpe.onnx").existsAsFile())
                preferences->setValue("algorithm.fcpePath",candidate.getChildFile("fcpe/fcpe.onnx").getFullPathName());
            config=backend::AnalysisService::configFromProperties(preferences.get());status=backend::AnalysisService::status(config);
        }
    }
    if(status.gameModelReady)preferences->setValue("algorithm.gamePath",status.gameModelDirectory.getFullPathName());
    if(status.fcpeModelReady)preferences->setValue("algorithm.fcpePath",status.fcpeModelPath.getFullPathName());
    check("real_game_fcpe_runtime_ready",status.gameModelReady&&status.fcpeModelReady&&status.onnxRuntimeReady,
        backend::AnalysisService::backendText(status)+" | "+status.message);
    if(!ok)return finish();
    preferences->saveIfNeeded();
    check("preferences_are_test_isolated",preferences->getFile().getFullPathName().contains(isolated),preferences->getFile().getFullPathName());
    if(!ok)return finish();
    applyPreferences();refreshTexts();stopTimer();setSize(1280,850);
    project.resetDocument(ProjectData{});project.dispatchPendingMessages();
    selectedTrackId=project.addTrack("Native workflow",true);
    project.setTrackPitchAlgorithm(selectedTrackId,PitchAlgorithm::nsfHifigan);
    project.setTrackNsfNoiseProtection(selectedTrackId,false);project.setTrackNormalizeVolume(selectedTrackId,false);
    project.setTrackVolume(selectedTrackId,.25f);
    const auto trackId=selectedTrackId;
    project.dispatchPendingMessages();refreshProjectControls();stopTimer();
    const auto nativeTrack=project.snapshot().tracks;
    check("native_nsf_source_mode_explicit",nativeTrack.size()==1&&nativeTrack[0].nativeNsfAudio
        &&nativeTrack[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan&&!trackUsesVoicebankSynthesis(nativeTrack[0])
        &&nativeTrack[0].voicebankDirectory==juce::File{});
    const auto bank=root.getChildFile("fixture-material");bank.createDirectory();
    const auto source=bank.getChildFile("harmonic-stereo.wav"),manifest=bank.getChildFile("otto.ini");
    {
        juce::AudioBuffer<float> pcm(2,96000);double phase=0;
        for(int i=0;i<pcm.getNumSamples();++i)
        {
            const auto time=i/48000.0;
            const auto hz=220.0*std::exp2(.12*std::sin(time*juce::MathConstants<double>::twoPi*5.0)/12.0);
            phase+=juce::MathConstants<double>::twoPi*hz/48000.0;
            const auto edge=std::min({1.0,time/.025,(2.0-time)/.025});
            pcm.setSample(0,i,static_cast<float>(edge*.22*(std::sin(phase)+.24*std::sin(phase*2)+.10*std::sin(phase*3))));
            pcm.setSample(1,i,static_cast<float>(edge*.18*(std::sin(phase+.09)+.20*std::sin(phase*2+.04)+.08*std::sin(phase*3))));
        }
        auto stream=source.createOutputStream();juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),48000,2,24,{},0):nullptr);
        check("generated_stereo_fixture_written",writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));
    }
    juce::MemoryBlock sourceBytes;check("original_source_bytes_recorded",source.loadFileAsData(sourceBytes));
    if(!ok)return finish();
    const auto pump=[&]()
    {
        MSG message{};int dispatched=0;
        while(dispatched++<256&&PeekMessageW(&message,nullptr,0,0,PM_REMOVE))
        {
            if(message.message==WM_QUIT){PostQuitMessage(static_cast<int>(message.wParam));return false;}
            TranslateMessage(&message);DispatchMessageW(&message);
        }
        project.dispatchPendingMessages();timerCallback();stopTimer();return true;
    };
    const auto wait=[&](const std::function<bool()>& ready,int timeoutMs)
    {
        const auto deadline=juce::Time::getMillisecondCounterHiRes()+timeoutMs;
        while(!ready()&&juce::Time::getMillisecondCounterHiRes()<deadline)
        {if(!pump())return false;juce::Thread::sleep(5);}
        if(!pump())return false;return ready();
    };
    setAssetManagerVisible(true);project.setAssetManagerVisible(true);resized();
    check("actual_main_material_manager_available",assetManager!=nullptr);
    if(!assetManager)return finish();
    int preflights=0,publications=0;
    struct RestoreCallbacks
    {
        juce::Component::SafePointer<AssetManagerComponent> manager;
        decltype(AssetManagerComponent::onMaterialPreflight) preflight;
        decltype(AssetManagerComponent::onMaterialSaved) saved;
        ~RestoreCallbacks(){if(manager!=nullptr){manager->onMaterialPreflight=std::move(preflight);manager->onMaterialSaved=std::move(saved);}}
    } restoreCallbacks{assetManager.get(),assetManager->onMaterialPreflight,assetManager->onMaterialSaved};
    check("main_central_publication_callbacks_wired",bool(restoreCallbacks.preflight)&&bool(restoreCallbacks.saved));
    assetManager->onMaterialPreflight=[&,actual=restoreCallbacks.preflight](const auto& file,const auto& rows,juce::String& reason)
    {++preflights;return actual&&actual(file,rows,reason);};
    assetManager->onMaterialSaved=[&,actual=restoreCallbacks.saved](const auto& file,const auto& rows)
    {++publications;if(actual)actual(file,rows);};
    const auto folderIndex=assetManager->diagnosticRegisterFolder(bank);
    check("import_folder_through_actual_registration",folderIndex>=0&&assetManager->diagnosticFolderCount()==1
        &&assetManager->diagnosticMembersOf(folderIndex).size()==1);
    OttoMaterialIO::Document empty;
    check("registration_creates_only_empty_native_manifest",OttoMaterialIO::loadDocument(manifest,empty,error)&&empty.audio.empty(),error);
    const auto findEditor=[&]() -> juce::DialogWindow*
    {
        auto& desktop=juce::Desktop::getInstance();
        for(int i=0;i<desktop.getNumComponents();++i)
            if(auto* window=dynamic_cast<juce::DialogWindow*>(desktop.getComponent(i));window&&window->isVisible())
                if(auto* editor=dynamic_cast<OtoWaveformEditorComponent*>(window->getContentComponent());
                    editor&&editor->originalEntry().audioFile==source)return window;
        return nullptr;
    };
    assetManager->editMaterial(source);pump();
    juce::Component::SafePointer<juce::DialogWindow> editorWindow(findEditor());
    if(editorWindow!=nullptr)openedWindows.push_back(editorWindow);
    auto* editor=editorWindow!=nullptr?dynamic_cast<OtoWaveformEditorComponent*>(editorWindow->getContentComponent()):nullptr;
    check("actual_precise_factory_opens_native_overview",editor&&editor->diagnosticMaterialOverview());
    check("precise_window_is_modeless",editorWindow!=nullptr&&!editorWindow->isCurrentlyModal()&&!isCurrentlyBlockedByAnotherModalComponent());
    if(!editor)return finish();
    assetManager->editMaterial(source);pump();
    check("precise_same_audio_reuses_existing_draft_window",findEditor()==editorWindow.getComponent());
    check("create_first_contiguous_pronunciation",editor->diagnosticCreateMaterialAlias(0,1,"ka",error),error);
    check("create_second_contiguous_pronunciation",editor->diagnosticCreateMaterialAlias(1,2,"a",error),error);
    auto authored=editor->diagnosticMaterialRows();
    check("native_overview_has_exact_two_ranges",authored.size()==2&&authored[0].regionStartSeconds==0
        &&authored[0].regionEndSeconds==1&&authored[1].regionStartSeconds==1&&authored[1].regionEndSeconds==2);
    if(authored.size()!=2)return finish();
    const auto firstRegion=authored[0].materialRegionId,secondRegion=authored[1].materialRegionId;
    const auto snapshot=[&](juce::Component& component,const juce::String& filename)
    {
        auto stream=root.getChildFile(filename+".png").createOutputStream();juce::PNGImageFormat format;
        const auto image=component.createComponentSnapshot(component.getLocalBounds());
        check(filename+"_snapshot",stream&&image.isValid()&&format.writeImageToStream(image,*stream));
    };
    const auto theme=[&](const juce::String& value)
    {preferences->setValue("ui.theme",value);applyPreferences();refreshTexts();stopTimer();pump();};
    for(const auto& colour:{juce::String("light"),juce::String("dark")})
    {
        theme(colour);
        for(const int width:{900,1000})
        {
            editor->setSize(width,780);snapshot(*editor,"oto-"+colour+"-"+juce::String(width)+"-overview");
            check("oto_detail_select_"+colour+juce::String(width),editor->selectMaterialPronunciation(firstRegion,true));
            check("multi_pronunciation_outer_bounds_locked",editor->diagnosticOuterRangeLocked());
            snapshot(*editor,"oto-"+colour+"-"+juce::String(width)+"-detail");
            check("oto_back_to_overview_"+colour+juce::String(width),editor->diagnosticMaterialBack());
        }
    }
    const auto priorPublication=publications;editor->diagnosticSave();pump();
    check("actual_save_runs_main_preflight_and_publication",publications==priorPublication+1&&preflights>=1);
    std::vector<SampleRegionSetting> persisted;
    check("primary_annotation_published_through_native_editor",OttoMaterialIO::loadAudio(manifest,source,persisted,error)
        &&persisted.size()==2&&persisted[0].materialRegionId==firstRegion&&persisted[1].materialRegionId==secondRegion,error);
    closeWindow(editorWindow);pump();editor=nullptr;
    assetManager->diagnosticToggleAudio(source);pump();
    const auto labels=assetManager->diagnosticPronunciationLabels(source);
    check("material_tree_exact_folder_audio_two_pronunciations",assetManager->diagnosticTreeRowCount()==4
        &&labels.size()==2&&labels[0].contains("ka")&&labels[1].startsWith("a ["));
    pianoViewport.setViewPosition(0,0);audio.setPosition(0);resized();
    const auto description=assetManager->audioDragDescription(source);
    const auto dropPoint=pianoViewport.getPosition()+juce::Point<int>(58,50);
    juce::DragAndDropTarget::SourceDetails drop(description,assetManager.get(),dropPoint);
    check("actual_manager_drag_description_uses_native_manifest",bool(description["nativeMaterial"])
        &&description["manifest"].toString()==manifest.getFullPathName()&&isInterestedInDragSource(drop));
    itemDropped(drop);pump();
    const auto placed=project.snapshot();
    check("actual_main_drop_places_two_native_pronunciations",placed.tracks.size()==1&&placed.tracks[0].clips.size()==1
        &&placed.tracks[0].clips[0].notes.size()==2&&placed.tracks[0].nativeNsfAudio
        &&placed.tracks[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan&&!trackUsesVoicebankSynthesis(placed.tracks[0]));
    check("native_analysis_was_scheduled",pendingNativeAnalyses>0||nativeAnalysisProgress==1.0);
    check("native_analysis_finishes_with_bounded_wait",wait([&]{return pendingNativeAnalyses==0;},120000),statusLabel.getText());
    auto measured=project.snapshot();
    bool measuredF0=measured.tracks.size()==1&&measured.tracks[0].clips.size()==1&&measured.tracks[0].clips[0].notes.size()==2;
    if(measuredF0)for(const auto& note:measured.tracks[0].clips[0].notes)
        measuredF0&=note.sourcePitchMeasured&&note.sourceMidiCenter>0&&note.contour.size()>50
            &&std::any_of(note.contour.begin(),note.contour.end(),[](const auto& point){return point.voiced;});
    check("real_analysis_populates_measured_source_f0_no_fake_contour",measuredF0,statusLabel.getText());
    if(!measuredF0)return finish();
    const auto clipId=measured.tracks[0].clips[0].id;
    const auto firstNote=measured.tracks[0].clips[0].notes[0].id;
    check("placed_notes_keep_material_identities",measured.tracks[0].clips[0].notes[0].materialRegionId==firstRegion
        &&measured.tracks[0].clips[0].notes[1].materialRegionId==secondRegion);
    focusClip(clipId);selectedNoteId=firstNote;pianoRoll.setSelectedNoteIds({firstNote});
    pianoRoll.setFocusedTrack(trackId);pianoRoll.setFocusedClip(clipId);pianoRoll.setPixelsPerSecond(200);
    pianoRoll.setShowPitchLine(true);pianoRoll.setShowOriginalPitchLine(true);pianoRoll.setShowNativeWaveforms(true);
    pianoRoll.setRowHeight(26);pianoRoll.setDrawAction(PianoRollComponent::DrawAction::pitchCurve);pianoRoll.setTool(PianoRollComponent::Tool::draw);
    pianoRoll.diagnosticRefresh();
    check("actual_native_roll_has_two_editable_notes",pianoRoll.diagnosticHitCount()==2);
    if(pianoRoll.diagnosticHitCount()!=2)return finish();
    const auto mouse=[&](juce::Point<float> at,juce::Point<float> down)
    {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&pianoRoll,&pianoRoll,
            juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);
    };
    const auto from=pianoRoll.diagnosticHitBounds(0).getCentre(),to=from.translated(48,-24);
    const auto curveRevision=project.revisionNumber();
    pianoRoll.mouseDown(mouse(from,from));pianoRoll.mouseDrag(mouse(from.translated(24,-12),from));
    pianoRoll.mouseDrag(mouse(to,from));pianoRoll.mouseUp(mouse(to,from));pump();pianoRoll.diagnosticRefresh();
    auto edited=project.snapshot();
    const auto hasTargets=[](const NoteData& note)
    {return std::any_of(note.contour.begin(),note.contour.end(),[](const auto& point){return point.hasManualTarget&&std::abs(point.manualTargetCents)>.01f;});};
    check("real_freehand_changes_target_pitch_not_note_count",edited.tracks[0].clips[0].notes.size()==2
        &&hasTargets(edited.tracks[0].clips[0].notes[0])&&project.revisionNumber()==curveRevision+1);
    pianoRoll.setSelectedNoteIds({firstNote});
    check("common_envelope_preset_applies_to_actual_native_note",pianoRoll.applyEnvelopePreset(.04,.12,-3.0f)>0);
    auto vibrato=project.snapshot().tracks[0].clips[0].notes[0];vibrato.vibratoDepthCents=22;vibrato.vibratoCycleMs=175;
    project.setNotesVibrato({firstNote},vibrato,true,true);pump();
    const auto automated=project.snapshot();const auto priorRequest=AudioEngine::diagnosticNativeRequest(automated.tracks[0].clips[0],automated.tracks[0]);
    check("shared_native_automation_is_real",automated.tracks[0].clips[0].notes[0].vibratoEnabled
        &&!automated.tracks[0].clips[0].notes[0].amplitudeEnvelope.empty()&&hasTargets(automated.tracks[0].clips[0].notes[0]));
    // Reopen through the actual manager and save through Main's same callbacks.
    assetManager->editMaterial(source);pump();editorWindow=findEditor();
    if(editorWindow!=nullptr)openedWindows.push_back(editorWindow);
    editor=editorWindow!=nullptr?dynamic_cast<OtoWaveformEditorComponent*>(editorWindow->getContentComponent()):nullptr;
    check("reopen_precise_editor_starts_native_overview",editor&&editor->diagnosticMaterialOverview()
        &&editorWindow!=nullptr&&!editorWindow->isCurrentlyModal());
    if(!editor)return finish();
    check("reopened_manifest_keeps_pronunciation_ids",editor->diagnosticMaterialRows().size()==2
        &&editor->diagnosticMaterialRows()[0].materialRegionId==firstRegion);
    check("rename_alias_in_existing_exact_range",editor->diagnosticCreateMaterialAlias(0,1,"ka-renamed",error)
        &&editor->diagnosticRemoveMaterialAlias(firstRegion,"ka",error),error);
    const auto renamePublications=publications;editor->diagnosticSave();pump();
    check("rename_publishes_through_same_main_route",publications==renamePublications+1&&preflights>=2);
    const auto renamed=project.snapshot();const auto renamedRequest=AudioEngine::diagnosticNativeRequest(renamed.tracks[0].clips[0],renamed.tracks[0]);
    check("publication_updates_placed_alias_instantly_by_stable_id",renamed.tracks[0].clips[0].notes[0].id==firstNote
        &&renamed.tracks[0].clips[0].notes[0].materialRegionId==firstRegion&&renamed.tracks[0].clips[0].notes[0].label=="ka-renamed");
    check("publication_preserves_target_and_gain_automation",priorRequest.targetMidi==renamedRequest.targetMidi
        &&priorRequest.noteGain==renamedRequest.noteGain&&renamed.tracks[0].clips[0].notes[0].vibratoEnabled
        &&hasTargets(renamed.tracks[0].clips[0].notes[0]));
    const auto sameMeasuredSources=[](const ClipData& before,const ClipData& after)
    {
        if(before.sourceFile!=after.sourceFile||before.sourceOffsetSeconds!=after.sourceOffsetSeconds
            ||before.sourceDurationSeconds!=after.sourceDurationSeconds||before.audioStartSeconds!=after.audioStartSeconds
            ||before.audioLength()!=after.audioLength()||before.sourceTimeMap.size()!=after.sourceTimeMap.size()
            ||before.notes.size()!=after.notes.size())return false;
        for(std::size_t i=0;i<before.sourceTimeMap.size();++i)
            if(before.sourceTimeMap[i].targetSeconds!=after.sourceTimeMap[i].targetSeconds
                ||before.sourceTimeMap[i].sourceSeconds!=after.sourceTimeMap[i].sourceSeconds)return false;
        for(const auto& note:before.notes)
        {
            const auto found=std::find_if(after.notes.begin(),after.notes.end(),[&](const auto& n){return n.id==note.id;});
            if(found==after.notes.end()||!note.sourcePitchMeasured||!found->sourcePitchMeasured
                ||note.materialRegionId!=found->materialRegionId||note.sourceMidiCenter!=found->sourceMidiCenter
                ||note.nativeSourceStartSeconds!=found->nativeSourceStartSeconds||note.nativeSourceEndSeconds!=found->nativeSourceEndSeconds
                ||note.contour.size()!=found->contour.size())return false;
            for(std::size_t i=0;i<note.contour.size();++i)
            {
                const auto& a=note.contour[i];const auto& b=found->contour[i];
                if(a.timeSeconds!=b.timeSeconds||a.relativeCents!=b.relativeCents
                    ||a.withoutVibratoCents!=b.withoutVibratoCents||a.voiced!=b.voiced)return false;
            }
        }
        return true;
    };
    check("alias_publication_preserves_measured_source_facts_and_clock",sameMeasuredSources(automated.tracks[0].clips[0],renamed.tracks[0].clips[0]));
    closeWindow(editorWindow);pump();editor=nullptr;
    const auto renamedLabels=assetManager->diagnosticPronunciationLabels(source);
    check("material_tree_refreshes_alias_without_duplicate_pronunciation",renamedLabels.size()==2&&renamedLabels[0].contains("ka-renamed"));
    check("no_legacy_annotation_outputs_created",!bank.getChildFile("oto.ini").exists()&&!bank.getChildFile("oto.jie.ini").exists()
        &&!bank.getChildFile("oto4.ini").exists()&&!bank.getChildFile("otomou.ini").exists()
        &&!SampleSettings::sidecarFor(source).exists()&&!juce::File(source.getFullPathName()+".hachi.csv").exists());
    const auto projectFile=root.getChildFile("native-workflow.hjpx");
    check("save_via_actual_main_project_entry",saveProjectTo(projectFile));pump();
    const auto saved=project.snapshot();
    struct RenderedPcm{juce::AudioBuffer<float> pcm;double rate=0;juce::String backend,warnings;bool success=false;};
    const auto exportActual=[&](const ProjectData& data,const juce::String& name)
    {
        RenderedPcm result;
        AudioEngine engine(false);engine.prepareToPlay(512,48000);engine.setHifiganModelDirectory(nsfModel);
        engine.setInferenceConfiguration(backend::InferenceBackend::cpu,-1);engine.syncProject(data);
        const auto deadline=juce::Time::getMillisecondCounterHiRes()+180000;
        while(engine.renderProgress()&&juce::Time::getMillisecondCounterHiRes()<deadline)
        {if(!pump())break;juce::Thread::sleep(10);}
        result.backend=engine.activeRenderBackends();result.warnings=engine.activeRenderWarnings();
        check(name+"_actual_nsf_onnx_backend_no_fallback",!engine.renderProgress()&&engine.hasCurrentRenderedAudio()
            &&result.backend.containsIgnoreCase("nsf-hifigan-onnx")&&!result.backend.containsIgnoreCase("fallback")
            &&!result.backend.containsIgnoreCase("world"),result.backend+" | "+result.warnings);
        check(name+"_render_has_no_warnings",result.warnings.isEmpty(),result.warnings);
        WavExportOptions options;options.sampleRate=48000;options.channels=2;options.bitDepth=32;
        const auto file=root.getChildFile(name+".wav");
        const auto& clip=data.tracks[0].clips[0];
        const bool exported=!engine.renderProgress()&&engine.hasCurrentRenderedAudio()
            &&engine.exportWav(file,error,trackId,clip.startSeconds,clip.startSeconds+clip.durationSeconds,options);
        check(name+"_actual_audio_engine_export",exported,error);
        juce::AudioFormatManager formats;formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(exported?formats.createReaderFor(file):nullptr);
        const bool metadata=reader&&reader->sampleRate==48000&&reader->numChannels==2
            &&std::abs(static_cast<double>(reader->lengthInSamples)-96000)<=2;
        check(name+"_pcm_length_rate_and_channels",metadata);
        if(metadata)
        {
            result.rate=reader->sampleRate;result.pcm.setSize(2,static_cast<int>(reader->lengthInSamples));
            result.success=reader->read(&result.pcm,0,result.pcm.getNumSamples(),0,true,true);
            bool finite=true;double energy=0;
            for(int ch=0;ch<2;++ch)for(int i=0;i<result.pcm.getNumSamples();++i)
            {const auto sample=result.pcm.getSample(ch,i);finite&=std::isfinite(sample);energy+=static_cast<double>(sample)*sample;}
            const auto rms=std::sqrt(energy/std::max(1,2*result.pcm.getNumSamples()));
            check(name+"_finite_audible_pcm",result.success&&finite&&rms>1.e-5,"rms="+juce::String(rms,9));
            result.success&=finite&&rms>1.e-5;
        }
        engine.releaseResources();return result;
    };
    auto before=exportActual(saved,"before-project-reopen");
    check("reopen_via_actual_main_project_entry",loadProjectFile(projectFile));pump();
    check("reopened_source_analysis_settles",wait([&]{return pendingNativeAnalyses==0;},120000));
    const auto reopened=project.snapshot();
    const bool reopenedShape=reopened.tracks.size()==1&&reopened.tracks[0].clips.size()==1&&reopened.tracks[0].clips[0].notes.size()==2;
    check("project_reopen_keeps_native_source_mode_and_material_ids",reopenedShape&&reopened.tracks[0].nativeNsfAudio
        &&reopened.tracks[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan&&!trackUsesVoicebankSynthesis(reopened.tracks[0])
        &&reopened.tracks[0].clips[0].notes[0].materialRegionId==firstRegion&&reopened.tracks[0].clips[0].notes[1].materialRegionId==secondRegion);
    if(!reopenedShape)return finish();
    const auto reopenedRequest=AudioEngine::diagnosticNativeRequest(reopened.tracks[0].clips[0],reopened.tracks[0]);
    check("project_reopen_preserves_measured_f0_and_target_automation",reopened.tracks[0].clips[0].notes[0].sourcePitchMeasured
        &&reopened.tracks[0].clips[0].notes[0].label=="ka-renamed"&&priorRequest.targetMidi==reopenedRequest.targetMidi
        &&priorRequest.noteGain==reopenedRequest.noteGain&&reopened.tracks[0].clips[0].notes[0].vibratoEnabled);
    auto after=exportActual(reopened,"after-project-reopen");
    bool comparable=before.success&&after.success&&before.rate==after.rate&&before.pcm.getNumSamples()==after.pcm.getNumSamples();
    double squaredError=0,referenceEnergy=0,cross=0,otherEnergy=0;float maxError=0;
    if(comparable)for(int ch=0;ch<2;++ch)for(int i=0;i<before.pcm.getNumSamples();++i)
    {
        const auto a=before.pcm.getSample(ch,i),b=after.pcm.getSample(ch,i);const auto difference=static_cast<double>(a)-b;
        squaredError+=difference*difference;referenceEnergy+=static_cast<double>(a)*a;
        otherEnergy+=static_cast<double>(b)*b;cross+=static_cast<double>(a)*b;maxError=std::max(maxError,std::abs(a-b));
    }
    const auto relativeRms=referenceEnergy>0?std::sqrt(squaredError/referenceEnergy):1.0;
    const auto correlation=referenceEnergy>0&&otherEnergy>0?cross/std::sqrt(referenceEnergy*otherEnergy):0.0;
    check("cold_engine_reopen_pcm_matches_same_configuration",comparable&&referenceEnergy>1.e-6&&maxError<=1.e-5f
        &&relativeRms<=1.e-5&&correlation>.99999,"max_abs="+juce::String(maxError,10)+" relative_rms="+juce::String(relativeRms,10)+" correlation="+juce::String(correlation,10));
    focusClip(clipId);selectedNoteId=firstNote;pianoRoll.setSelectedNoteIds({firstNote});pianoRoll.setTool(PianoRollComponent::Tool::draw);
    for(const auto& colour:{juce::String("light"),juce::String("dark")})
    {
        theme(colour);
        for(const int width:{1280,1000})
        {
            setSize(width,850);resized();pianoRoll.diagnosticRefresh();
            pianoViewport.setViewPosition(0,std::max(0,static_cast<int>(pianoRoll.diagnosticYForMidi(reopened.tracks[0].clips[0].notes[0].midiNote))-pianoViewport.getViewHeight()/2));
            snapshot(*this,"main-native-"+colour+"-"+juce::String(width));
        }
    }
    juce::MemoryBlock finalSource;check("original_stereo_source_bytes_never_rewritten",source.loadFileAsData(finalSource)&&finalSource==sourceBytes);
    check("test_never_started_audio_playback",!audio.isPlaying());
    SampleSettings::setMaterialRoots(previousRoots);
    return finish();
#endif
}
}
