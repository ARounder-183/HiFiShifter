#pragma once
#include "../MainComponent.h"
#include "../PronunciationCandidateBox.h"
#include "../MaterialPronunciationCatalog.h"
#include "../backend/AnalysisService.h"
#include "../backend/NsfHifiganRenderer.h"
#include "ForcedPortamentoAudioSmoke.h"
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
inline bool MainComponent::diagnosticMaterialSourceReplacementUi(const juce::File& output,const juce::File& nsfModel)
{
    const auto root=output.getChildFile("main-source-replacement-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto finish=[&]{dismissMaterialPronunciationInput();const auto written=root.getChildFile("main-source-replacement-ui.txt").replaceWithText(log);std::cout<<"main_source_replacement_fixture="<<root.getFullPathName()<<std::endl;return ok&&written;};
    check("actual_message_thread",juce::MessageManager::getInstance()->isThisTheMessageThread());
    const auto isolated=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SETTINGS_DIR",{});
    check("isolated_settings_required",preferences&&isolated.isNotEmpty()&&preferences->getFile().getFullPathName().contains(isolated));
    check("real_nsf_model_required",backend::NsfHifiganRenderer::modelAvailable(nsfModel));
#if !JUCE_WINDOWS
    check("windows_bounded_message_pump_required",false);return finish();
#else
    if(!ok)return finish();audio.stop();stopTimer();pianoRoll.setReadsVoicebankInBackground(false);
    preferences->setValue("assets.materialFolders",juce::String{});preferences->setValue("algorithm.hifiganPath",nsfModel.getFullPathName());
    preferences->setValue("algorithm.inference",2);preferences->setValue("algorithm.device",1);preferences->setValue("algorithm.gameModel","medium");
    auto config=backend::AnalysisService::configFromProperties(preferences.get());auto analysisStatus=backend::AnalysisService::status(config);
    for(const auto& candidate:{nsfModel.getParentDirectory(),nsfModel,nsfModel.getParentDirectory().getParentDirectory().getChildFile("models")})
    {
        if(!analysisStatus.gameModelReady&&candidate.getChildFile("game").isDirectory())preferences->setValue("algorithm.gamePath",candidate.getChildFile("game").getFullPathName());
        if(!analysisStatus.fcpeModelReady&&candidate.getChildFile("fcpe/fcpe.onnx").existsAsFile())preferences->setValue("algorithm.fcpePath",candidate.getChildFile("fcpe/fcpe.onnx").getFullPathName());
        config=backend::AnalysisService::configFromProperties(preferences.get());analysisStatus=backend::AnalysisService::status(config);
    }
    check("actual_GAME_FCPE_runtime_required_no_synthetic_analysis",analysisStatus.gameModelReady&&analysisStatus.fcpeModelReady&&analysisStatus.onnxRuntimeReady);
    log+="resolved_game_path="+analysisStatus.gameModelDirectory.getFullPathName()+"\nresolved_fcpe_path="+analysisStatus.fcpeModelPath.getFullPathName()+"\nconfigured_status="+backend::AnalysisService::backendText(analysisStatus)+"\n";
    if(!ok)return finish();applyPreferences();setSize(1320,850);stopTimer();
    struct ActualUiHost final:juce::DocumentWindow
    {
        ActualUiHost():juce::DocumentWindow("Main source replacement acceptance",juce::Colours::darkgrey,juce::DocumentWindow::allButtons){setUsingNativeTitleBar(true);}
        void closeButtonPressed() override{setVisible(false);}
    };
    std::unique_ptr<ActualUiHost> host;
    if(!isShowing()){host=std::make_unique<ActualUiHost>();host->setContentNonOwned(this,false);host->centreWithSize(1320,850);host->setVisible(true);host->toFront(true);}
    check("actual_main_has_showing_native_host",isShowing());
    const auto pump=[&]
    {
        MSG message{};int count=0;while(count++<64&&PeekMessageW(&message,nullptr,0,0,PM_REMOVE))
        {if(message.message==WM_QUIT){PostQuitMessage(static_cast<int>(message.wParam));return false;}TranslateMessage(&message);DispatchMessageW(&message);}
        project.dispatchPendingMessages();pianoRoll.diagnosticRefresh();return true;
    };
    const auto wait=[&](const std::function<bool()>& ready,int timeout)
    {const auto end=juce::Time::getMillisecondCounterHiRes()+timeout;while(!ready()&&juce::Time::getMillisecondCounterHiRes()<end){if(!pump())return false;juce::Thread::sleep(5);}pump();return ready();};
    const auto explicitB=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SOURCE_REPLACEMENT_B",{}),explicitC=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SOURCE_REPLACEMENT_C",{});
    juce::File referenceB=explicitB.isEmpty()?juce::File{}:juce::File(explicitB),referenceC=explicitC.isEmpty()?juce::File{}:juce::File(explicitC);
    if(referenceB==juce::File{}||referenceC==juce::File{})
    {
        ProjectModel reference;juce::String referenceError;
        const auto loaded=reference.load(juce::File("D:\\hjm\\hachishifter\\consistency\\teio.hjpx"),referenceError);
        check("existing_loader_reads_real_Japanese_reference_project",loaded);
        juce::StringArray paths;
        if(loaded)for(const auto& track:reference.snapshot().tracks)for(const auto& parent:track.clips)
        {
            if(parent.parts.empty()){if(parent.sourceFile.existsAsFile()&&parent.sourceFile.hasFileExtension("wav;flac;aif;aiff;ogg;mp3"))paths.addIfNotAlreadyThere(parent.sourceFile.getFullPathName());}
            else for(const auto& source:parent.parts)if(source.sourceFile.existsAsFile()&&source.sourceFile.hasFileExtension("wav;flac;aif;aiff;ogg;mp3"))paths.addIfNotAlreadyThere(source.sourceFile.getFullPathName());
        }
        paths.sort(false);if(referenceB==juce::File{}&&paths.size()>0)referenceB=juce::File(paths[0]);
        if(referenceC==juce::File{})for(const auto& path:paths)if(juce::File(path)!=referenceB){referenceC=juce::File(path);break;}
        log+="reference_project_error="+referenceError+"\n";
    }
    check("two_explicit_or_project_referenced_real_Japanese_recordings",referenceB.existsAsFile()&&referenceC.existsAsFile()&&referenceB!=referenceC);
    if(!ok)return finish();
    const auto referenceHashB=juce::SHA256(referenceB).toHexString(),referenceHashC=juce::SHA256(referenceC).toHexString();
    const auto bank=root.getChildFile("material");bank.createDirectory();const auto a=bank.getChildFile("old-C4.wav");
    const auto b=bank.getChildFile("chosen-B-real-vocal"+referenceB.getFileExtension()),c=bank.getChildFile("chosen-C-real-vocal"+referenceC.getFileExtension());
    log+="real_reference_B="+referenceB.getFullPathName()+"\nreal_reference_C="+referenceC.getFullPathName()+"\n";
    const auto wave=[](const juce::File& file,double midi,int seconds,bool voicedLike)
    {
        constexpr int rate=48000;juce::AudioBuffer<float> pcm(2,rate*seconds);const auto carrier=440*std::exp2((midi-69)/12.0);double phase=0;
        for(int i=0;i<pcm.getNumSamples();++i)
        {
            const auto time=i/double(rate);
            // Same validated physical signal as NativeWorkflowE2ESmoke: mild
            // vocal-like FM, three harmonics,25ms envelope, stereo phase offsets.
            // A stays stationary C4 so its seeded existing source facts are exact;
            // B/C require actual GAME+FCPE measurement and never get F0 hints.
            const auto hz=carrier*std::exp2((voicedLike?.12:0)*std::sin(time*juce::MathConstants<double>::twoPi*5.0)/12.0);
            phase+=juce::MathConstants<double>::twoPi*hz/rate;const auto edge=std::min({1.0,time/.025,(seconds-time)/.025});
            pcm.setSample(0,i,static_cast<float>(edge*.22*(std::sin(phase)+.24*std::sin(phase*2)+.10*std::sin(phase*3))));
            pcm.setSample(1,i,static_cast<float>(edge*.18*(std::sin(phase+.09)+.20*std::sin(phase*2+.04)+.08*std::sin(phase*3))));
        }
        auto stream=file.createOutputStream();juce::WavAudioFormat format;std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),rate,2,24,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    check("original_C4_and_independent_real_vocal_copies",wave(a,60,3,false)&&referenceB.copyFileTo(b)&&referenceC.copyFileTo(c)
        &&referenceHashB==juce::SHA256(b).toHexString()&&referenceHashC==juce::SHA256(c).toHexString());
    if(!ok)return finish();
    struct MeasuredRegion{double from=0,to=0,median=0;bool valid=false;};
    const auto voicedRegion=[](const std::vector<NoteData>& notes)
    {
        struct Frame{double t=0,midi=0;};std::vector<Frame> frames;
        for(const auto& n:notes)for(const auto& p:n.contour)if(p.voiced)
        {const auto midi=(n.sourceMidiCenter>=0?n.sourceMidiCenter:n.midiNote)+p.relativeCents/100.0;if(std::isfinite(midi))frames.push_back({n.startSeconds+p.timeSeconds,midi});}
        std::stable_sort(frames.begin(),frames.end(),[](const auto& x,const auto& y){return x.t<y.t;});
        MeasuredRegion best;std::size_t first=0;double longest=0;
        for(std::size_t i=1;i<=frames.size();++i)
        {
            if(i<frames.size()&&frames[i].t-frames[i-1].t<=.015)continue;
            const auto span=frames[i-1].t-frames[first].t;
            if(span>=.35&&span>longest)
            {
                const auto from=frames[first].t+.025,to=std::min(frames[i-1].t-.025,from+.8);std::vector<double> pitches;
                for(auto k=first;k<i;++k)if(frames[k].t>=from&&frames[k].t<=to)pitches.push_back(frames[k].midi);
                if(pitches.size()>=20){best={from,to,forcedPortamentoAudioSmokeDetail::median(pitches),true};longest=span;}
            }
            first=i;
        }
        return best;
    };
    MeasuredRegion measuredRegionB,measuredRegionC;
    for(const auto& source:{b,c})
    {
        juce::String probeError;const auto probe=backend::AnalysisService::analyse(source,config,probeError);
        const auto trace="real_vocal_preflight file="+source.getFileName()+" active="+probe.status.activeBackend+" warning="+probe.warning+" error="+probeError;
        log+=trace+"\n";std::cout<<trace<<std::endl;
        check(source==b?"B_real_vocal_actual_GAME_FCPE_preflight":"C_real_vocal_actual_GAME_FCPE_preflight",probe.status.activeBackend=="GAME+FCPE"&&!probe.notes.empty());
        juce::String detail="file="+source.getFullPathName()+" notes="+juce::String(static_cast<int>(probe.notes.size()))+"\n";
        std::vector<std::pair<double,double>> voicedFrames;int noteIndex=0,totalFrames=0;
        for(const auto& n:probe.notes)
        {
            int voiced=0;std::vector<double> pitches;
            for(const auto& p:n.contour)if(p.voiced)
            {const auto midi=(n.sourceMidiCenter>=0?n.sourceMidiCenter:n.midiNote)+p.relativeCents/100.0;if(std::isfinite(midi)){voicedFrames.emplace_back(n.startSeconds+p.timeSeconds,midi);pitches.push_back(midi);++voiced;}}
            totalFrames+=static_cast<int>(n.contour.size());
            detail+="note="+juce::String(noteIndex++)+" abs_start="+juce::String(n.startSeconds,7)+" duration="+juce::String(n.durationSeconds,7)
                +" local_first="+juce::String(n.contour.empty()?0:n.contour.front().timeSeconds,7)+" local_last="+juce::String(n.contour.empty()?0:n.contour.back().timeSeconds,7)
                +" contour="+juce::String(static_cast<int>(n.contour.size()))+" voiced="+juce::String(voiced)+" source_center="+juce::String(n.sourceMidiCenter,7);
            if(!pitches.empty()){const auto bounds=std::minmax_element(pitches.begin(),pitches.end());detail+=" min="+juce::String(*bounds.first,7)+" median="+juce::String(forcedPortamentoAudioSmokeDetail::median(pitches),7)+" max="+juce::String(*bounds.second,7);}
            detail+="\n";
        }
        std::sort(voicedFrames.begin(),voicedFrames.end());double longestRun=0,maxGap=0,runStart=voicedFrames.empty()?0:voicedFrames.front().first;
        for(std::size_t i=0;i<voicedFrames.size();++i)
        {if(i){const auto gap=voicedFrames[i].first-voicedFrames[i-1].first;maxGap=std::max(maxGap,gap);if(gap>.015)runStart=voicedFrames[i].first;}longestRun=std::max(longestRun,voicedFrames[i].first-runStart);}
        detail+="TOTAL contour="+juce::String(totalFrames)+" voiced="+juce::String(static_cast<int>(voicedFrames.size()))+" longest_contiguous_voiced="+juce::String(longestRun,7)+" max_voiced_gap="+juce::String(maxGap,7)+"\n";
        log+=detail;std::cout<<detail<<std::endl;root.getChildFile(source==b?"B-source-facts.txt":"C-source-facts.txt").replaceWithText(detail);
        const auto measured=voicedRegion(probe.notes);if(source==b)measuredRegionB=measured;else measuredRegionC=measured;
        check(source==b?"B_actual_voiced_interval_meets_declared_eligibility":"C_actual_voiced_interval_meets_declared_eligibility",measured.valid);
    }
    check("real_measured_voiced_regions_not_filename_pitch_hints",measuredRegionB.valid&&measuredRegionC.valid);
    log+="B_actual_source_median="+juce::String(measuredRegionB.median,7)+" window="+juce::String(measuredRegionB.from,7)+","+juce::String(measuredRegionB.to,7)+"\nC_actual_source_median="+juce::String(measuredRegionC.median,7)+" window="+juce::String(measuredRegionC.from,7)+","+juce::String(measuredRegionC.to,7)+"\n";
    check("real_C_source_has_measured_contrast_from_target_C4",measuredRegionC.valid&&std::abs(measuredRegionC.median-60)>=2);
    if(!ok)return finish();
    SampleRegionSetting row;row.name="same-primary";row.regionStartSeconds=row.alignmentSeconds=measuredRegionB.from;row.regionEndSeconds=measuredRegionB.to;row.role=NativeSegmentRole::vowel;row.provenance="user";
    const auto aliasB="source-b-"+juce::Uuid().toString().substring(0,8),aliasC="source-c-"+juce::Uuid().toString().substring(0,8);
    row.additionalAliases.add(aliasB);check("B_annotated_secondary_alias",SampleSettings::save(b,{row},error));row.additionalAliases.clear();row.additionalAliases.add(aliasC);
    row.regionStartSeconds=row.alignmentSeconds=measuredRegionC.from;row.regionEndSeconds=measuredRegionC.to;
    check("C_same_primary_different_physical_source",SampleSettings::save(c,{row},error));
    juce::MemoryBlock aBytes,bBytes,cBytes;check("source_bytes_captured",a.loadFileAsData(aBytes)&&b.loadFileAsData(bBytes)&&c.loadFileAsData(cBytes));
    setAssetManagerVisible(true);check("actual_material_registry",assetManager&&assetManager->diagnosticRegisterFolder(bank)>=0);if(!ok)return finish();
    ProjectData seed;TrackData t;t.id="native-source-ui";t.pitchAlgorithm=PitchAlgorithm::nsfHifigan;t.nativeNsfAudio=true;t.normalizeVolume=false;t.nsfNoiseProtection=false;t.nsfSmoothPitchTransitions=false;
    ClipData parent;parent.id="multi-owner";parent.sourceFile=a;parent.durationSeconds=parent.sourceDurationSeconds=3;parent.sourceTimeMap={{0,0},{3,3}};
    for(int i=0;i<3;++i)
    {
        NoteData n;n.id=i==0?"left":i==1?"target":"right";n.label="old";n.startSeconds=i;n.durationSeconds=1;n.midiNote=n.sourceMidiCenter=60;n.sourcePitchMeasured=true;
        n.nativeRole=NativeSegmentRole::vowel;n.nativeSourceStartSeconds=i;n.nativeSourceEndSeconds=i+1;n.consonantSeconds=0;
        n.pitchControlPoints={{0,60,PitchCurveShape::linear},{1,60,PitchCurveShape::linear}};n.gain=.67f;n.amplitudeEnvelope={{0,-2},{.5,-1},{1,-3}};
        if(i==1){n.vibratoEnabled=true;n.vibratoDepthCents=15;n.vibratoCycleMs=190;}
        for(int j=0;j<=200;++j)n.contour.push_back({j*.005,0,0,true});parent.notes.push_back(n);
    }
    t.clips={parent};seed.tracks={t};project.resetDocument(seed);project.dispatchPendingMessages();focusNote("left");audio.setPosition(.1);
    pianoRoll.setFocusedTrack(t.id);pianoRoll.setFocusedClip(parent.id);pianoRoll.setSelectedNoteIds({"left"});pianoRoll.diagnosticRefresh();
    const auto owner=[](const ProjectData& d,const juce::String& id)
    {for(const auto& track:d.tracks)for(const auto& p:track.clips)for(const auto& n:p.notes)if(n.id==id){if(p.parts.empty())return p;for(const auto& part:p.parts)if(part.id==n.clipPartId)return part;}return ClipData{};};
    const auto note=[](const ProjectData& d,const juce::String& id)
    {for(const auto& track:d.tracks)for(const auto& p:track.clips)for(const auto& n:p.notes)if(n.id==id)return n;return NoteData{};};
    const auto picker=[&]()->PronunciationCandidateBox*
    {
        std::function<PronunciationCandidateBox*(juce::Component*)> find=[&](juce::Component* component)->PronunciationCandidateBox*
        {if(auto* box=dynamic_cast<PronunciationCandidateBox*>(component))return box;if(component)for(int i=0;i<component->getNumChildComponents();++i)if(auto* result=find(component->getChildComponent(i)))return result;return nullptr;};
        return find(materialPronunciationPopup.getComponent());
    };
    const auto open=[&](const juce::String& id,const juce::String& query)
    {pianoRoll.onRequestMaterialSourceInput(id,query);return wait([&]{auto* p=picker();return p&&p->diagnosticCandidateCount()==1;},30000);};
    const auto choose=[&]
    {if(auto* p=picker()){p->diagnosticSelect(0);p->diagnosticCommit();return true;}return false;};
    const auto before=project.contentFingerprint();const auto revision=project.revisionNumber();
    check("replace_callback_opens_existing_actual_candidate_window",open("target",aliasB));
    check("opening_and_query_never_mutate_score",project.contentFingerprint()==before&&project.revisionNumber()==revision);
    dismissMaterialPronunciationInput();pump();check("cancel_no_mutation_or_undo",project.contentFingerprint()==before&&project.revisionNumber()==revision&&!project.canUndo());
    focusNote("target");pianoRoll.setSelectedNoteIds({"target"});pianoRoll.diagnosticRefresh();
    pianoViewport.setViewPosition(0,std::max(0,static_cast<int>(pianoRoll.diagnosticNoteY(1))-100));pump();
    const auto beforeInline=project.contentFingerprint();const auto inlineRevision=project.revisionNumber();
    const auto inlineOpened=pianoRoll.diagnosticBeginAliasEdit("target");check("actual_inline_editor_opens_on_exact_target",inlineOpened);
    if(inlineOpened)
    {
        check("inline_editor_has_real_keyboard_focus_before_picker",wait([&]{return pianoRoll.diagnosticAliasEditorHasFocus();},5000));
        pianoRoll.diagnosticTypeInAliasEditor(aliasB);
        juce::TextEditor* aliasEditor=nullptr;for(int i=0;i<pianoRoll.getNumChildComponents();++i)
            if(auto* candidate=dynamic_cast<juce::TextEditor*>(pianoRoll.getChildComponent(i));candidate&&candidate->isVisible())aliasEditor=candidate;
        const auto handled=aliasEditor&&aliasEditor->keyPressed(juce::KeyPress(juce::KeyPress::returnKey,juce::ModifierKeys::ctrlModifier,0));
        check("actual_inline_CtrlEnter_uses_same_source_picker",handled&&wait([&]{auto* p=picker();return p&&p->diagnosticCandidateCount()==1;},30000));
        check("inline_focus_loss_suspended_without_alias_commit",project.contentFingerprint()==beforeInline&&project.revisionNumber()==inlineRevision&&pianoRoll.diagnosticAliasEditorNoteId()=="target");
        dismissMaterialPronunciationInput();const auto restoredFocus=wait([&]{return pianoRoll.diagnosticAliasEditorHasFocus();},5000);
        const auto restoredText=pianoRoll.diagnosticAliasEditorText();const auto restoredId=pianoRoll.diagnosticAliasEditorNoteId();
        const auto restoreTrace="inline_resume focus="+juce::String(restoredFocus?1:0)+" id="+restoredId+" text="+restoredText+" expected_text="+aliasB
            +" rev="+juce::String(static_cast<juce::int64>(project.revisionNumber()))+" expected_rev="+juce::String(static_cast<juce::int64>(inlineRevision))
            +" fingerprint_equal="+juce::String(project.contentFingerprint()==beforeInline?1:0)+" main_showing="+juce::String(isShowing()?1:0)+" popup="+juce::String(materialPronunciationPopup?1:0);
        log+=restoreTrace+"\n";std::cout<<restoreTrace<<std::endl;
        check("inline_cancel_resumes_typed_text_without_model_edit",restoredFocus&&restoredText==aliasB&&project.contentFingerprint()==beforeInline&&project.revisionNumber()==inlineRevision);
        pianoRoll.diagnosticCancelAliasEdit();pump();
    }
    // Change the active selection away again: callback ID, not playhead or
    // whatever is selected while the modeless catalog runs, remains authority.
    focusNote("left");pianoRoll.setSelectedNoteIds({"left"});
    check("actual_picker_B_selection",open("target",aliasB)&&choose());
    check("source_choice_targets_captured_note_not_playhead_or_current_selection",wait([&]{return owner(project.snapshot(),"target").sourceFile==b;},30000)
        &&owner(project.snapshot(),"left").sourceFile==a&&owner(project.snapshot(),"right").sourceFile==a);
    const auto bBound=project.snapshot();const auto bTarget=note(bBound,"target");
    check("chosen_additional_alias_bound_once",bTarget.label==aliasB&&bTarget.materialBoundAlias==aliasB&&!bTarget.materialAliasFollowsPrimary);
    check("one_source_choice_one_music_revision",project.revisionNumber()==revision+1&&bTarget.midiNote==60&&bTarget.startSeconds==1&&bTarget.durationSeconds==1&&bTarget.gain==parent.notes[1].gain&&bTarget.vibratoEnabled&&bTarget.vibratoDepthCents==15&&bTarget.vibratoCycleMs==190
        &&bTarget.amplitudeEnvelope.size()>=parent.notes[1].amplitudeEnvelope.size());
    const bool sawPending=owner(bBound,"target").materialSourceAnalysisPending;
    check("unknown_source_is_really_render_gated",!sawPending||(!bTarget.sourcePitchMeasured&&bTarget.sourceMidiCenter<0&&owner(bBound,"target").muted));
    const auto pendingFile=root.getChildFile("pending-source.hachi");check("pending_project_saved",project.save(pendingFile,error));
    check("undo_before_async_analysis",project.undo()&&project.contentFingerprint()==before);
    check("late_B_analysis_does_not_restore_undone_binding",wait([&]{return pendingNativeAnalyses==0;},180000)&&owner(project.snapshot(),"target").sourceFile==a);
    check("redo_resumes_from_guarded_hash_cache",project.redo()&&wait([&]{const auto d=project.snapshot();return owner(d,"target").sourceFile==b&&!owner(d,"target").materialSourceAnalysisPending;},30000));
    const auto measuredB=project.snapshot();const auto actualB=note(measuredB,"target");
    log+="measured_B_source_midi="+juce::String(actualB.sourceMidiCenter,7)+"\n";
    check("actual_Japanese_B_source_facts_match_measured_reference_and_score_C4_kept",actualB.sourcePitchMeasured&&std::abs(actualB.sourceMidiCenter-measuredRegionB.median)<.25&&actualB.midiNote==60&&!owner(measuredB,"target").muted);
    const auto measuredFile=root.getChildFile("measured-source.hachi");check("measured_project_saved",project.save(measuredFile,error));
    const auto backendTrace="actual_completed_backend="+lastNativeAnalysisBackend+"\n"+lastNativeAnalysisTrace;
    log+=backendTrace+"\n";std::cout<<backendTrace<<std::endl;root.getChildFile("actual-analysis-worker.txt").replaceWithText(backendTrace);
    check("real_GAME_FCPE_completed_in_existing_Main_worker",lastNativeAnalysisBackend.containsIgnoreCase("GAME")&&lastNativeAnalysisBackend.containsIgnoreCase("FCPE"));
    check("actual_Main_reopen_pending_reschedules_or_reuses_cache",loadProjectFile(pendingFile)&&wait([&]{return !owner(project.snapshot(),"target").materialSourceAnalysisPending&&note(project.snapshot(),"target").sourcePitchMeasured;},180000));
    check("actual_Main_reopen_measured_keeps_binding_and_target",loadProjectFile(measuredFile)&&owner(project.snapshot(),"target").sourceFile==b&&note(project.snapshot(),"target").midiNote==60);
    // Launch another real B analysis, then choose C through the same popup.
    // Both workers may finish in either order; no callback may restore B.
    const auto pendingTicket=ProjectModel::MaterialSourceBindingTicket{"target","multi-owner",note(project.snapshot(),"target").clipPartId,b,juce::SHA256(b).toHexString()};
    scheduleAnalysis(b,{},pendingTicket,true);
    check("B_to_C_actual_picker_choice",open("target",aliasC)&&choose()&&wait([&]{return owner(project.snapshot(),"target").sourceFile==c;},30000));
    bool neverRestoredB=true;const auto deadline=juce::Time::getMillisecondCounterHiRes()+180000;
    while(pendingNativeAnalyses>0&&juce::Time::getMillisecondCounterHiRes()<deadline){pump();neverRestoredB&=owner(project.snapshot(),"target").sourceFile==c;juce::Thread::sleep(5);}pump();
    const auto measuredC=project.snapshot();
    check("late_real_B_callback_cannot_overwrite_C",pendingNativeAnalyses==0&&neverRestoredB&&owner(measuredC,"target").sourceFile==c&&note(measuredC,"target").materialBoundAlias==aliasC);
    check("C_receives_actual_measured_Japanese_source_facts_but_score_remains_C4",note(measuredC,"target").sourcePitchMeasured&&std::abs(note(measuredC,"target").sourceMidiCenter-measuredRegionC.median)<.25&&note(measuredC,"target").midiNote==60);
    // Render the actual captured Main result, not a synthetic source-F0 fixture.
    AudioEngine offline(false);offline.setHifiganModelDirectory(nsfModel);offline.setInferenceConfiguration(backend::InferenceBackend::cpu,-1);offline.prepareToPlay(256,48000);offline.syncProject(measuredC);
    const auto renderDeadline=juce::Time::getMillisecondCounterHiRes()+180000;
    while(offline.renderProgress().has_value()&&juce::Time::getMillisecondCounterHiRes()<renderDeadline){pump();juce::Thread::sleep(5);}
    WavExportOptions options;options.sampleRate=48000;options.channels=2;options.bitDepth=32;const auto rendered=root.getChildFile("actual-main-replaced-Japanese-C-to-C4.wav");
    const bool exported=!offline.renderProgress().has_value()&&offline.exportWav(rendered,error,t.id,1,2,options);
    check("actual_NSF_output_not_native_PCM_bypass",exported&&offline.activeRenderBackends().contains("nsf-hifigan-onnx")&&!offline.activeRenderBackends().containsIgnoreCase("fallback")&&offline.devices().getCurrentAudioDevice()==nullptr);
    if(exported)
    {
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(rendered));
        juce::AudioBuffer<float> waveOut;if(reader){waveOut.setSize(reader->numChannels,static_cast<int>(reader->lengthInSamples));reader->read(&waveOut,0,waveOut.getNumSamples(),0,true,true);}
        std::vector<double> measured;for(double at=.2;at<.8;at+=.05)if(reader){const auto p=forcedPortamentoAudioSmokeDetail::measurePitch(waveOut,reader->sampleRate,at);if(p.voiced)measured.push_back(p.midi);}
        const auto median=forcedPortamentoAudioSmokeDetail::median(measured);log+="actual_output_median_midi="+juce::String(median,7)+"\n";
        check("unconditioned_audio_F0_is_C4_not_measured_Japanese_source_pitch",measured.size()>=8&&std::abs(median-60)<.4&&std::abs(median-measuredRegionC.median)>1.5);
    }
    juce::MemoryBlock aa,bb,cc;check("all_original_source_bytes_unchanged",a.loadFileAsData(aa)&&b.loadFileAsData(bb)&&c.loadFileAsData(cc)&&aa==aBytes&&bb==bBytes&&cc==cBytes);
    check("both_external_Japanese_originals_unmodified",juce::SHA256(referenceB).toHexString()==referenceHashB&&juce::SHA256(referenceC).toHexString()==referenceHashC);
    return finish();
#endif
}
}
