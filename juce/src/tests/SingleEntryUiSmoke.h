#pragma once
#include "../MainComponent.h"
#include "../AssetManagerComponent.h"
#include "../OtoWaveformEditorComponent.h"
#include "../SettingsComponent.h"
#include "../Theme.h"
#include <iostream>
#include <array>
#include <set>
namespace hachi
{
inline bool MainComponent::diagnosticSingleEntryUi(const juce::File& output)
{
    output.createDirectory();stopTimer();bool ok=true;juce::String log;
    const auto check=[&](const juce::String& name,bool value){ok&=value;log+=name+"="+(value?"PASS":"FAIL")+"\n";};
    check("display_options_in_main_view_only",diagnosticHasEditorViewItems());
    check("no_root_zoom_slider_entry",zoomSlider.getParentComponent()==nullptr&&vZoomSlider.getParentComponent()==nullptr);
    int plusMinus=0,displayButtons=0,duplicateFileButtons=0;
    for(int i=0;i<getNumChildComponents();++i)
        if(auto* button=dynamic_cast<juce::Button*>(getChildComponent(i)))
        {
            plusMinus+=button->isVisible()&&(button->getButtonText()=="+"||button->getButtonText()=="-");
            displayButtons+=button->isVisible()&&button->getButtonText()==strings.text("native.display");
            const auto id=button->getComponentID();
            duplicateFileButtons+=button->isVisible()&&(id=="icon.open"||id=="icon.save"||id=="icon.audio"||id=="icon.melodyne");
        }
    check("no_root_plus_minus_duplicates",plusMinus==0);check("no_piano_display_dropdown",displayButtons==0);
    check("file_actions_only_in_file_menu",duplicateFileButtons==0);
    showSettings(0);auto* first=settingsWindows.findVisible<SettingsComponent>();
    showSettings(1);auto* second=settingsWindows.findVisible<SettingsComponent>();
    check("settings_uses_one_existing_window",first&&first==second&&settingsWindows.size()==1);
    if(first){auto image=output.getChildFile("settings-main-theme.png").createOutputStream();check("settings_theme_snapshot",image&&juce::PNGImageFormat().writeImageToStream(first->createComponentSnapshot(first->getLocalBounds()),*image));}
    if(first)first->setVisible(false);settingsWindows.removeClosed();
    const auto file=output.getChildFile("source.wav");
    juce::AudioBuffer<float> pcm(1,48000);pcm.clear();
    auto stream=file.createOutputStream();juce::WavAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),48000,1,16,{},0):nullptr);
    check("source_fixture",writer&&writer->writeFromAudioSampleBuffer(pcm,0,48000));writer.reset();
    ClipData clip;clip.id="single-entry-clip";clip.sourceFile=file;clip.durationSeconds=clip.sourceDurationSeconds=1;
    NoteData note;note.id="single-entry-note";note.label="source";note.durationSeconds=1;note.midiNote=note.sourceMidiCenter=60;
    clip.notes={note};TrackData track;track.id="single-entry-track";track.compose=true;
    track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.clips={clip};
    ProjectData data;data.tracks={track};project.resetDocument(data);project.dispatchPendingMessages();
    selectedTrackId=track.id;selectedClipId=clip.id;selectedNoteId=note.id;
    pianoRoll.setFocusedTrack(track.id);pianoRoll.setFocusedClip(clip.id);pianoRoll.setSelectedNoteIds({note.id});
    refreshProjectControls();setSize(1280,800);
    check("legacy_segment_selector_not_always_visible",!sampleRegionSelector.isVisible());
    project.setNoteLabel(note.id,"renamed");noteLabelCommitted(note.id);
    check("label_edit_never_switches_backend_or_binds_default",project.snapshot().tracks[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan
        &&project.snapshot().tracks[0].voicebankDirectory==juce::File{});
    wrenchButton.onClick();auto* quickWindow=noteOtoWindows.findVisible<OtoWaveformEditorComponent>();
    auto* quick=quickWindow?dynamic_cast<OtoWaveformEditorComponent*>(quickWindow->getContentComponent()):nullptr;
    check("wrench_reuses_original_oto_class",quick&&quick->diagnosticMaterialOverview());
    wrenchButton.onClick();check("quick_reopen_does_not_duplicate_window",quickWindow&&noteOtoWindows.findVisible<OtoWaveformEditorComponent>()==quickWindow);
    check("quick_open_does_not_publish_annotations",!file.getSiblingFile("otto.ini").existsAsFile());
    if(quickWindow)quickWindow->setVisible(false);noteOtoWindows.removeClosed();
    {
        SampleRegionSetting pronunciation;pronunciation.name="ka";pronunciation.regionEndSeconds=1;
        juce::String pickerError;
        check("picker_fixture_explicit_annotation",SampleSettings::save(file,{pronunciation},pickerError));
        setAssetManagerVisible(true);
        check("picker_fixture_registered_once",assetManager&&assetManager->diagnosticRegisterFolder(output)>=0);
        const auto beforePicker=project.contentFingerprint();const auto beforePickerRevision=project.revisionNumber();
        showMaterialPronunciationInput();auto* firstPopup=materialPronunciationPopup.getComponent();
        check("material_input_has_one_visible_owned_picker",firstPopup&&firstPopup->isVisible());
        juce::Component::SafePointer<juce::CallOutBox> oldPopup(firstPopup);
        showMaterialPronunciationInput();auto* replacement=materialPronunciationPopup.getComponent();
        check("repeated_material_input_replaces_not_stacks",replacement&&replacement->isVisible()
            &&replacement!=oldPopup.getComponent()&&(!oldPopup||!oldPopup->isVisible()));
        dismissMaterialPronunciationInput();
        check("cancel_material_input_closes_picker_without_edit",materialPronunciationPopup.getComponent()==nullptr
            &&project.revisionNumber()==beforePickerRevision&&project.contentFingerprint()==beforePicker);
    }
    if(preferences){preferences->setValue("ui.theme","light");applyPreferences();refreshTexts();}
    setSize(1280,800);resized();
    auto out=output.getChildFile("main-native-light.png").createOutputStream();
    check("main_native_screenshot",out&&juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()),*out));
    // Source-reviewed action matrix: these are different operations, not seven
    // aliases for the same action. Left-mode callbacks and actual UI bounds are
    // exercised below; gesture semantics have their dedicated model smoke tests.
    const juce::String actionMatrix=
        "tool | left | drag | double click | right\n"
        "Arrow | select/move tool | note move or edge resize | native P-to-grid alignment | note context\n"
        "Pencil | explicit pitch/notes action | freehand pitch OR explicit note-length draw | none | existing action/length menu\n"
        "Points | point tool | shared multi-point edit | add point | reset/delete/mode menu\n"
        "Line | line tool | two-endpoint pitch stroke | none | note context\n"
        "Wrench | precise native/OTO editor | none | modeless editor reused | none\n"
        "Trim | source trim tool | crop source edge | none | existing source context\n"
        "Source chain | source-boundary tool | no pitch bridge | split source note or join native seam | note context\n"
        "Force pitch text action | separate parent-owned pitch graph operation | separate graph handles | separate | separate\n";
    check("toolbar_action_ownership_matrix_written",output.getChildFile("toolbar-action-matrix.txt").replaceWithText(actionMatrix));
    const std::array<std::pair<juce::Button*,PianoRollComponent::Tool>,6> toolActions{{
        {&noteEditButton,PianoRollComponent::Tool::note},{&drawButton,PianoRollComponent::Tool::draw},
        {&pointButton,PianoRollComponent::Tool::points},{&lineButton,PianoRollComponent::Tool::line},
        {&trimButton,PianoRollComponent::Tool::trim},{&connectButton,PianoRollComponent::Tool::connect}}};
    const auto toolRevision=project.revisionNumber();
    for(const auto& action:toolActions)
    {
        const bool callable=action.first->isEnabled()&&bool(action.first->onClick);
        if(callable)action.first->onClick();
        check("actual_left_tool_"+action.first->getComponentID(),callable&&pianoRoll.currentTool()==action.second);
    }
    check("tool_mode_buttons_do_not_edit_music",project.revisionNumber()==toolRevision);
    check("pencil_and_points_keep_distinct_right_menus",bool(drawButton.onSecondaryClick)&&bool(pointButton.onSecondaryClick));
    const auto pointMenu=pitchPointModeMenu();check("points_right_menu_is_actual_shared_builder",pointMenu.getNumItems()>0);
    check("source_chain_is_distinct_from_force_pitch_action",static_cast<juce::Button*>(&connectButton)!=static_cast<juce::Button*>(&spliceButton)
        &&connectButton.getTooltip()==strings.text("Source-note boundary split/join, not forced pitch portamento"));
    setAssetManagerVisible(true);assetManagerWidth=320;refreshProjectControls();stopTimer();
    pianoRoll.setPitchScope(PianoRollComponent::PitchScope::focusedClip);
    const auto scopeRevision=project.revisionNumber();const auto scopeFingerprint=project.contentFingerprint();
    const auto scopeRenderHash=AudioEngine::utauNoteRenderHash(project.snapshot().tracks[0].clips[0].notes[0]);
    const auto scopeScheduled=audio.diagnosticScheduledNsfUtauRenders();
    applyEditorViewChoice(trackPitchScopeMenuItem-600);
    check("one_View_item_persists_track_pitch_scope",pianoRoll.pitchScope()==PianoRollComponent::PitchScope::currentTrack
        &&preferences&&preferences->getValue("ui.pitchScope")=="track");
    check("scope_View_toggle_keeps_music_and_render_authority",project.revisionNumber()==scopeRevision
        &&project.contentFingerprint()==scopeFingerprint&&AudioEngine::utauNoteRenderHash(project.snapshot().tracks[0].clips[0].notes[0])==scopeRenderHash
        &&audio.diagnosticScheduledNsfUtauRenders()==scopeScheduled);
    pianoRoll.setShowOverlap(false);
    juce::PopupMenu overlapView;appendEditorViewItems(overlapView);int overlapEntries=0;bool overlapUnticked=false;
    for(juce::PopupMenu::MenuItemIterator it(overlapView);it.next();)if(it.getItem().itemID==613)
    {++overlapEntries;overlapUnticked=!it.getItem().isTicked&&it.getItem().isEnabled;}
    check("one_overlap_checkbox_in_existing_View_menu",overlapEntries==1&&overlapUnticked);
    menuItemSelected(613,0);
    check("actual_View_overlap_dispatch_persists_enabled",pianoRoll.showsOverlap()&&preferences&&preferences->getBoolValue("ui.showOverlap",false));
    menuItemSelected(613,0);
    check("actual_View_overlap_dispatch_off_is_display_only",!pianoRoll.showsOverlap()&&preferences&&!preferences->getBoolValue("ui.showOverlap",true)
        &&project.revisionNumber()==scopeRevision&&project.contentFingerprint()==scopeFingerprint
        &&AudioEngine::utauNoteRenderHash(project.snapshot().tracks[0].clips[0].notes[0])==scopeRenderHash
        &&audio.diagnosticScheduledNsfUtauRenders()==scopeScheduled);
    std::array<juce::Button*,7> icons{{&noteEditButton,&drawButton,&pointButton,&lineButton,&wrenchButton,&trimButton,&connectButton}};
    for(const auto colour:{juce::String("light"),juce::String("dark")})
    {
        if(preferences){preferences->setValue("ui.theme",colour);applyPreferences();refreshTexts();}
        for(const int width:{1000,1280})
        {
            setSize(width,850);resized();const auto label=colour+"-"+juce::String(width);
            bool usable=true,overlap=false;std::set<juce::String> identities;
            const auto editorRight=assetManager?assetManager->getX():getWidth();
            for(auto* icon:icons)
            {
                usable&=icon->isVisible()&&icon->getWidth()>=24&&icon->getHeight()>=24
                    &&getLocalBounds().contains(icon->getBounds())&&icon->getRight()<=editorRight;
                identities.insert(icon->getComponentID());
                for(auto* other:icons)if(icon!=other)overlap|=icon->getBounds().intersects(other->getBounds());
                for(auto* other:{static_cast<juce::Component*>(&pitchLabel),static_cast<juce::Component*>(&pitchAlgorithm),
                    static_cast<juce::Component*>(&renderOrderLabel),static_cast<juce::Component*>(&renderOrder),
                    static_cast<juce::Component*>(&stretchLabel),static_cast<juce::Component*>(&stretchAlgorithm),
                    static_cast<juce::Component*>(&spliceButton)})
                    if(other->isVisible())overlap|=icon->getBounds().intersects(other->getBounds());
                log+=label+" "+icon->getComponentID()+" bounds="+icon->getBounds().toString()+"\n";
            }
            check("all_seven_unique_icons_usable_"+label,usable&&!overlap&&identities.size()==7);
            check("native_Attack_remains_existing_distinct_context_control_"+label,attackParamButton.isVisible()&&attackParamButton.getWidth()>=50);
            auto image=output.getChildFile("main-toolbar-"+label+".png").createOutputStream();
            check("actual_responsive_Main_snapshot_"+label,image&&juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()),*image));
        }
    }
    // Leave the prior export-scope diagnostic in its expected light theme.
    if(preferences){preferences->setValue("ui.theme","light");applyPreferences();refreshTexts();}
    exportMidi();
    auto* scopeDialog=dynamic_cast<juce::AlertWindow*>(juce::ModalComponentManager::getInstance()->getModalComponent(0));
    check("one_midi_export_flow_has_explicit_scope",scopeDialog&&scopeDialog->getComboBoxComponent("scope")
        &&scopeDialog->getComboBoxComponent("scope")->getSelectedId()==1);
    if(scopeDialog)
    {
        auto image=output.getChildFile("midi-scope-light.png").createOutputStream();
        check("midi_scope_real_ui_screenshot",image&&juce::PNGImageFormat().writeImageToStream(
            scopeDialog->createComponentSnapshot(scopeDialog->getLocalBounds()),*image));
        scopeDialog->exitModalState(0);
    }
    output.getChildFile("verification.txt").replaceWithText(log);
    std::cout<<log<<std::endl;return ok;
}
}
