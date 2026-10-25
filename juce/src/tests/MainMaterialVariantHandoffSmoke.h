#pragma once
#include "../MainComponent.h"
#include "../OttoMaterialIO.h"
#include "../OtoWaveformEditorComponent.h"
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
inline bool MainComponent::diagnosticMaterialVariantHandoffUi(const juce::File& output)
{
    const auto root=output.getChildFile("main-variant-handoff-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    stopTimer();audio.stop();bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto finish=[&]{const auto written=root.getChildFile("main-variant-handoff-ui.txt").replaceWithText(log);std::cout<<"variant_handoff_fixture="<<root.getFullPathName()<<std::endl;return ok&&written;};
#if !JUCE_WINDOWS
    check("windows_actual_message_pump",false);return finish();
#else
    const auto pump=[&]
    {
        MSG message{};for(int i=0;i<128&&PeekMessageW(&message,nullptr,0,0,PM_REMOVE);++i){if(message.message==WM_QUIT){PostQuitMessage(static_cast<int>(message.wParam));return false;}TranslateMessage(&message);DispatchMessageW(&message);}
        project.dispatchPendingMessages();return true;
    };
    const auto wait=[&](const std::function<bool()>& ready,int milliseconds)
    {const auto deadline=juce::Time::getMillisecondCounterHiRes()+milliseconds;while(!ready()&&juce::Time::getMillisecondCounterHiRes()<deadline){if(!pump())return false;juce::Thread::sleep(5);}pump();return ready();};
    const auto source=root.getChildFile("one-physical-audio.wav"),manifest=root.getChildFile("otto.ini");
    juce::AudioBuffer<float> pcm(1,48000);for(int i=0;i<48000;++i)pcm.setSample(0,i,.1f*std::sin(juce::MathConstants<double>::twoPi*220*i/48000.0));
    auto stream=source.createOutputStream();juce::WavAudioFormat format;std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),48000,1,24,{},0):nullptr);
    check("single_physical_source_fixture",writer&&writer->writeFromAudioSampleBuffer(pcm,0,48000));writer.reset();
    SampleRegionSetting a;a.materialAudioId="annotation-A";a.materialRegionId="same-row-id";a.name="A-only";a.regionStartSeconds=a.alignmentSeconds=.1;a.regionEndSeconds=.5;a.role=NativeSegmentRole::vowel;
    auto b=a;b.materialAudioId="annotation-B";b.name="B-only";b.regionStartSeconds=b.alignmentSeconds=.2;b.regionEndSeconds=.8;
    auto alternate=a;alternate.materialRegionId="overlapping-alternative";alternate.name="A-alternative";
    alternate.regionStartSeconds=alternate.alignmentSeconds=.05;alternate.regionEndSeconds=.7;
    OttoMaterialIO::Document document;document.audio={{a.materialAudioId,source.getFileName(),{a,alternate}},{b.materialAudioId,source.getFileName(),{b}}};
    check("two_variants_same_path_and_scoped_row_ids_saved",OttoMaterialIO::saveDocument(manifest,document,error));if(!ok)return finish();
    juce::MemoryBlock sourceBefore,manifestBefore;source.loadFileAsData(sourceBefore);manifest.loadFileAsData(manifestBefore);
    setAssetManagerVisible(true);check("actual_asset_manager_registered",assetManager&&assetManager->diagnosticRegisterFolder(root)>=0);if(!assetManager)return finish();
    check("typed_Main_selection_callback_bound",bool(assetManager->onLoadMaterialSelection));
    ProjectData seed;TrackData track;track.id="variant-host";track.pitchAlgorithm=PitchAlgorithm::world;seed.tracks={track};project.resetDocument(seed);project.dispatchPendingMessages();selectedTrackId=track.id;pianoRoll.setFocusedTrack(track.id);audio.setPosition(0);setSize(1320,850);resized();
    const auto beforeRevision=project.revisionNumber();
    assetManager->onLoadMaterialSelection({source,root,b.materialAudioId,{b.materialRegionId}});
    const auto placed=project.snapshot();
    check("typed_callback_inserts_exact_variant_not_same_path_sibling",placed.tracks.size()==1&&placed.tracks[0].clips.size()==1&&placed.tracks[0].clips[0].notes.size()==1
        &&placed.tracks[0].clips[0].notes[0].materialAudioId==b.materialAudioId&&placed.tracks[0].clips[0].notes[0].label==b.name
        &&placed.tracks[0].clips[0].notes[0].materialSourceStartSeconds==b.regionStartSeconds&&placed.tracks[0].clips[0].notes[0].materialSourceEndSeconds==b.regionEndSeconds);
    check("typed_callback_single_music_undo",project.revisionNumber()==beforeRevision+1&&project.canUndo());
    check("typed_analysis_finishes",wait([&]{return pendingNativeAnalyses==0;},180000));
    if(project.snapshot().tracks[0].clips.empty())return finish();const auto boundNote=project.snapshot().tracks[0].clips[0].notes[0].id;
    openSelectedMaterialEditor(true,boundNote);auto* quickWindow=noteOtoWindows.findVisible<OtoWaveformEditorComponent>();
    auto* quick=quickWindow?dynamic_cast<OtoWaveformEditorComponent*>(quickWindow->getContentComponent()):nullptr;
    const auto quickRows=quick?quick->diagnosticMaterialRows():std::vector<SampleRegionSetting>{};
    check("precise_quick_editor_receives_bound_variant_identity",quickRows.size()==1&&quickRows[0].materialAudioId==b.materialAudioId&&quickRows[0].name==b.name);
    if(quickWindow)quickWindow->setVisible(false);noteOtoWindows.removeClosed();
    pianoRoll.diagnosticRefresh();
    const auto annotationMenu=pianoRoll.diagnosticEnabledNoteMenuIds(boundNote);
    check("native_note_menu_offers_material_annotation",std::find(annotationMenu.begin(),annotationMenu.end(),34)!=annotationMenu.end()
        &&pianoRoll.diagnosticNoteMenuLabel(boundNote,34)==strings.text("Edit material annotations..."));
    pianoRoll.applyNoteMenuChoice(boundNote,34,0,{});pump();
    bool exactFull=false;
    for(int i=0;i<juce::Desktop::getInstance().getNumComponents();++i)
        if(auto* window=dynamic_cast<juce::DialogWindow*>(juce::Desktop::getInstance().getComponent(i));window&&window->isVisible())
            if(auto* editor=dynamic_cast<OtoWaveformEditorComponent*>(window->getContentComponent()))
            {const auto rows=editor->diagnosticMaterialRows();if(rows.size()==1&&rows[0].materialAudioId==b.materialAudioId&&rows[0].name==b.name){exactFull=editor->diagnosticMaterialOverview()&&editor->diagnosticMaterialRegionId()==b.materialRegionId;window->setVisible(false);}}
    check("full_manager_editor_receives_exact_registration",exactFull);
    project.resetDocument(seed);project.dispatchPendingMessages();selectedTrackId=track.id;pianoRoll.setFocusedTrack(track.id);resized();
    const auto payload=assetManager->pronunciationDragDescription(source,{a.materialRegionId},a.materialAudioId);
    check("actual_drag_payload_contains_variant",payload["audioVariantId"].toString()==a.materialAudioId&&bool(payload["nativeMaterial"]));
    const auto dropPoint=pianoViewport.getPosition()+juce::Point<int>(58,40);juce::DragAndDropTarget::SourceDetails drop(payload,assetManager.get(),dropPoint);
    check("Main_accepts_real_manager_variant_drag",isInterestedInDragSource(drop));itemDropped(drop);
    const auto dragged=project.snapshot();
    check("actual_drop_preserves_explicit_variant_and_shared_row_id",dragged.tracks[0].clips.size()==1&&dragged.tracks[0].clips[0].notes.size()==1
        &&dragged.tracks[0].clips[0].notes[0].materialAudioId==a.materialAudioId&&dragged.tracks[0].clips[0].notes[0].label==a.name);
    check("drag_analysis_finishes",wait([&]{return pendingNativeAnalyses==0;},180000));
    project.resetDocument(seed);project.dispatchPendingMessages();selectedTrackId=track.id;pianoRoll.setFocusedTrack(track.id);resized();
    const auto defaultPayload=assetManager->audioDragDescription(source,a.materialAudioId);
    juce::DragAndDropTarget::SourceDetails defaultDrop(defaultPayload,assetManager.get(),dropPoint);
    const auto target=audioDropTarget(dropPoint.x,dropPoint.y);
    itemDropped(defaultDrop);
    const auto defaultPlaced=project.snapshot();
    check("file_row_drag_chooses_first_list_entry_not_earliest_time",defaultPlaced.tracks[0].clips.size()==1
        &&defaultPlaced.tracks[0].clips[0].notes.size()==1
        &&defaultPlaced.tracks[0].clips[0].notes[0].materialRegionId==a.materialRegionId
        &&std::abs(defaultPlaced.tracks[0].clips[0].startSeconds-target.first)<1.e-8);
    check("default_drag_analysis_finishes",wait([&]{return pendingNativeAnalyses==0;},180000));
    project.resetDocument(seed);project.dispatchPendingMessages();selectedTrackId=track.id;pianoRoll.setFocusedTrack(track.id);resized();
    const auto alternatePayload=assetManager->pronunciationDragDescription(source,{alternate.materialRegionId},a.materialAudioId);
    juce::DragAndDropTarget::SourceDetails alternateDrop(alternatePayload,assetManager.get(),dropPoint);itemDropped(alternateDrop);
    const auto alternatePlaced=project.snapshot();
    check("explicit_second_pronunciation_stays_second",alternatePlaced.tracks[0].clips.size()==1
        &&alternatePlaced.tracks[0].clips[0].notes.size()==1
        &&alternatePlaced.tracks[0].clips[0].notes[0].materialRegionId==alternate.materialRegionId);
    check("alternate_drag_analysis_finishes",wait([&]{return pendingNativeAnalyses==0;},180000));
    const auto unchanged=project.contentFingerprint();const auto unchangedRevision=project.revisionNumber();
    const auto closeWarnings=[&]{for(int i=0;i<8;++i){auto* modal=juce::Component::getCurrentlyModalComponent();if(!modal)break;modal->exitModalState(0);pump();}};
    assetManager->onLoadMaterialSelection({source,root,"unknown-registration",{a.materialRegionId}});closeWarnings();
    check("unknown_explicit_variant_does_not_fall_back_or_mutate",project.contentFingerprint()==unchanged&&project.revisionNumber()==unchangedRevision);
    importAssetAudio(source,root,1,2,track.id,{},"unknown-registration");closeWarnings();
    check("generic_audio_branch_unknown_variant_never_falls_back",project.contentFingerprint()==unchanged&&project.revisionNumber()==unchangedRevision);
    importMaterialPronunciations(source,manifest,{"removed-row"},2,track.id,{},a.materialAudioId);closeWarnings();
    check("stale_explicit_region_never_selects_sibling",project.contentFingerprint()==unchanged&&project.revisionNumber()==unchangedRevision);
    juce::MemoryBlock sourceAfter,manifestAfter;source.loadFileAsData(sourceAfter);manifest.loadFileAsData(manifestAfter);
    check("handoff_does_not_rewrite_source_or_annotation_variants",sourceBefore==sourceAfter&&manifestBefore==manifestAfter);
    // External file drops must become editable project materials without a
    // folder-registration/copy step, including after undo and project reload.
    const auto looseRoot=output.getChildFile("loose-audio-"+juce::Uuid().toString());
    const auto loose=looseRoot.getChildFile("external.wav"), unrelated=looseRoot.getChildFile("unrelated.wav");
    check("loose_audio_fixture",looseRoot.createDirectory().wasOk()&&source.copyFileTo(loose)&&source.copyFileTo(unrelated));
    project.resetDocument(seed);project.dispatchPendingMessages();selectedTrackId=track.id;pianoRoll.setFocusedTrack(track.id);
    setAssetManagerVisible(false);project.setAssetManagerVisible(false);resized();
    const auto registeredBefore=preferences->getValue("assets.materialFolders");
    const auto externalPoint=pianoViewport.getPosition()+juce::Point<int>(58,40);
    filesDropped({loose.getFullPathName()},externalPoint.x,externalPoint.y);pump();
    check("external_drop_opens_material_manager",assetManagerVisible&&project.snapshot().assetManagerVisible);
    check("external_analysis_finishes",wait([&]{return pendingNativeAnalyses==0;},180000));
    const auto visibleCount=[&](const juce::String& name)
    {
        int count=0;const auto tree=assetManager->diagnosticTreeSnapshot();
        if(auto* rows=tree.getArray())for(const auto& row:*rows)
            if(int(row["depth"])==1&&row["label"].toString().contains(name))++count;
        return count;
    };
    check("dropped_audio_immediately_visible_once",visibleCount("external.wav")==1);
    check("unrelated_sibling_never_registered_or_scanned",visibleCount("unrelated.wav")==0
        &&assetManager->diagnosticMembersOf(0)==juce::StringArray{"external.wav"});
    const auto toggleProjectFolder=[&]
    {
        const juce::Point<float> at(16,90);const auto now=juce::Time::getCurrentTime();
        assetManager->mouseDown(juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
            juce::ModifierKeys::leftButtonModifier,juce::MouseInputSource::defaultPressure,0,0,0,0,
            assetManager.get(),assetManager.get(),now,at,now,1,false));
    };
    toggleProjectFolder();toggleProjectFolder();
    check("reexpand_project_folder_keeps_only_referenced_audio",visibleCount("external.wav")==1&&visibleCount("unrelated.wav")==0);
    check("import_does_not_write_otto_or_global_registry",!looseRoot.getChildFile("otto.ini").exists()
        &&preferences->getValue("assets.materialFolders")==registeredBefore);
    check("unchanged_project_skips_browser_rebuild",!assetManager->setProjectVoicebanks(project.snapshot()));
    assetManager->diagnosticSearchEditor().setText("unrelated");pump();
    check("project_material_search_excludes_unreferenced_siblings",visibleCount("unrelated.wav")==0);
    assetManager->diagnosticSearchEditor().onEscapeKey();pump();
    check("clearing_search_restores_project_audio",visibleCount("external.wav")==1);
    if(project.snapshot().tracks[0].clips.empty())return finish();
    project.removeClip(project.snapshot().tracks[0].clips[0].id);pump();
    check("delete_last_reference_removes_project_file_row",visibleCount("external.wav")==0);
    check("undo_reference_removal",project.undo());pump();
    check("undo_restores_project_file_row",visibleCount("external.wav")==1);
    check("redo_reference_removal",project.redo());pump();
    check("redo_removes_project_file_row",visibleCount("external.wav")==0);
    project.undo();pump();
    const auto savedProject=output.getChildFile("automatic-project-audio.hachi");
    check("save_automatic_project_view",project.save(savedProject,error));
    project.resetDocument(seed);pump();check("switch_project_removes_transient_audio",visibleCount("external.wav")==0);
    check("reopen_restores_project_audio",project.load(savedProject,error));pump();
    check("reopened_audio_visible_without_registration",visibleCount("external.wav")==1);
    pianoRoll.diagnosticRefresh();
    const auto looseNote=project.snapshot().tracks[0].clips[0].notes[0].id;
    selectedNoteId=boundNote; // Context action must use the clicked note, not stale host selection.
    pianoRoll.applyNoteMenuChoice(looseNote,34,0,{});pump();
    OtoWaveformEditorComponent* looseEditor=nullptr;juce::DialogWindow* looseWindow=nullptr;
    for(int i=0;i<juce::Desktop::getInstance().getNumComponents();++i)
        if(auto* window=dynamic_cast<juce::DialogWindow*>(juce::Desktop::getInstance().getComponent(i));window&&window->isVisible())
            if(auto* editor=dynamic_cast<OtoWaveformEditorComponent*>(window->getContentComponent());editor&&window->getName().contains("external.wav"))
            {looseEditor=editor;looseWindow=window;}
    check("unregistered_audio_opens_native_annotation_editor",looseEditor&&looseEditor->diagnosticMaterialOverview());
    if(looseEditor)
    {
        check("bare_audio_starts_without_invented_alias",looseEditor->diagnosticMaterialRows().empty());
        const auto bareSelection=looseEditor->diagnosticMaterialSourceSelection();
        check("bare_note_source_range_selected_without_publishing",bareSelection&&bareSelection->getLength()>0&&!looseEditor->diagnosticScopeDirty());
        check("native_alias_creation",looseEditor->diagnosticCreateMaterialAlias(.1,.3,"external-alias",error));
        check("middle_and_tail_aliases_created",looseEditor->diagnosticCreateMaterialAlias(.45,.6,"same-alias",error)
            &&looseEditor->diagnosticCreateMaterialAlias(.75,.9,"same-alias",error));
        looseEditor->diagnosticSave();pump();
        check("explicit_save_publishes_native_annotation",looseRoot.getChildFile("otto.ini").existsAsFile()&&!looseEditor->diagnosticScopeDirty());
        if(looseWindow)looseWindow->setVisible(false);
    }
    juce::MemoryBlock looseAfter;loose.loadFileAsData(looseAfter);
    check("native_annotation_preserves_original_pcm",looseAfter==sourceBefore);
    check("annotation_refresh_does_not_expose_siblings",visibleCount("unrelated.wav")==0);
    assetManager->diagnosticToggleAudio(loose);
    check("saved_alias_available_in_project_material_tree",assetManager->diagnosticPronunciationLabels(loose).joinIntoString("\n").contains("external-alias"));
    // Duplicate clip references and composite source parts share one file row.
    auto duplicate=project.snapshot();auto extra=duplicate.tracks[0].clips[0];extra.id="duplicate-source";extra.startSeconds+=3;
    duplicate.tracks[0].clips.push_back(extra);project.resetDocument(duplicate);pump();
    check("duplicate_source_has_one_browser_row",visibleCount("external.wav")==1);
    auto composite=seed;ClipData outer,inner,leaf;leaf.sourceFile=loose;inner.parts={leaf};outer.parts={inner,leaf};
    composite.tracks[0].clips={outer};assetManager->setProjectVoicebanks(composite);
    check("nested_composite_parts_share_project_source_row",visibleCount("external.wav")==1);
    assetManager->setProjectVoicebanks(project.snapshot());
    const auto managerImage=assetManager->createComponentSnapshot(assetManager->getLocalBounds());
    auto imageStream=output.getChildFile("automatic-project-audio.png").createOutputStream();
    check("project_material_view_capture",imageStream&&juce::PNGImageFormat().writeImageToStream(managerImage,*imageStream));
    // Unbound legacy notes: source clock, clip offset and part placement must
    // resolve the physical pronunciation, independently of labels/display time.
    auto timed=seed;ClipData part;part.id="source-part";part.sourceFile=loose;part.startSeconds=2;
    part.durationSeconds=part.audioDurationSeconds=10;part.sourceDurationSeconds=1;
    part.sourceTimeMap={{0,0},{10,1}};
    ClipData parent;parent.id="focus-parent";parent.startSeconds=30;parent.durationSeconds=12;parent.parts={part};
    NoteData middle;middle.id="focus-middle";middle.clipPartId=part.id;middle.startSeconds=6.5;middle.durationSeconds=1.5;middle.label="renamed";
    auto tail=middle;tail.id="focus-tail";tail.startSeconds=9.5;parent.notes={middle,tail};
    timed.tracks[0].clips={parent};project.resetDocument(timed);pump();pianoRoll.diagnosticRefresh();
    const auto beforeNavigation=project.contentFingerprint();
    juce::MemoryBlock annotationsBefore;looseRoot.getChildFile("otto.ini").loadFileAsData(annotationsBefore);
    const auto findFocusedEditor=[&]() -> OtoWaveformEditorComponent*
    {
        for(int i=0;i<juce::Desktop::getInstance().getNumComponents();++i)
            if(auto* window=dynamic_cast<juce::DialogWindow*>(juce::Desktop::getInstance().getComponent(i));window&&window->isVisible()&&window->getName().contains("external.wav"))
                if(auto* editor=dynamic_cast<OtoWaveformEditorComponent*>(window->getContentComponent()))return editor;
        return nullptr;
    };
    pianoRoll.applyNoteMenuChoice(middle.id,34,0,{});pump();auto* focused=findFocusedEditor();
    const auto isFocused=[&](double start,double end)
    {
        if(!focused||!focused->diagnosticMaterialOverview())return false;
        const auto range=focused->diagnosticMaterialSourceSelection();
        return range&&std::abs(range->getStart()-start)<1.e-6&&std::abs(range->getEnd()-end)<1.e-6;
    };
    check("unbound_stretched_merged_note_focuses_middle_source_region",isFocused(.45,.6));
    if(focused)
        check("overview_keeps_explicit_detail_entry",focused->selectMaterialPronunciation(focused->diagnosticMaterialRegionId(),true)
            &&!focused->diagnosticMaterialOverview());
    pianoRoll.applyNoteMenuChoice(tail.id,34,0,{});pump();
    check("reused_detail_window_returns_to_overview_and_retargets",focused==findFocusedEditor()&&isFocused(.75,.9));
    if(focused)
    {
        auto capture=output.getChildFile("note-source-region-focus.png").createOutputStream();
        check("focused_region_capture",capture&&juce::PNGImageFormat().writeImageToStream(focused->createComponentSnapshot(focused->getLocalBounds()),*capture));
        check("ambiguous_time_range_stays_unpublished_selection",focused->focusMaterialSourceRange({.1,.9})
            &&focused->diagnosticMaterialOverview()&&focused->diagnosticMaterialRegionId().isEmpty()&&!focused->diagnosticScopeDirty());
        check("single_mode_enable_from_note_editor",focused->selectMaterialPronunciation(focused->diagnosticMaterialRows().back().materialRegionId,false)
            &&focused->setSingleSyllableMaterial(true));
        pianoRoll.applyNoteMenuChoice(middle.id,34,0,{});pump();
        const auto singleRange=focused->diagnosticMaterialSourceSelection();
        check("note_menu_honours_single_mode_and_retargets",focused==findFocusedEditor()&&!focused->diagnosticMaterialOverview()
            &&singleRange&&std::abs(singleRange->getStart()-.45)<1.e-6);
        check("single_mode_can_restore_note_overview",focused->setSingleSyllableMaterial(false)&&focused->diagnosticMaterialOverview());
    }
    juce::MemoryBlock annotationsAfter;looseRoot.getChildFile("otto.ini").loadFileAsData(annotationsAfter);
    check("automatic_focus_changes_neither_project_nor_otto",beforeNavigation==project.contentFingerprint()&&annotationsBefore==annotationsAfter);
    const auto written=root.getChildFile("main-variant-handoff-ui.txt").replaceWithText(log);std::cout<<"variant_handoff_fixture="<<root.getFullPathName()<<std::endl;return ok&&written;
#endif
}
}
