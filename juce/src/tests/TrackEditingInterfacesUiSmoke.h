#pragma once
#include "../MainComponent.h"
#include "UtauSourceEditorSmoke.h"
#include <iostream>

namespace hachi
{
inline bool MainComponent::diagnosticTrackEditingInterfacesUi(const juce::File& output)
{
    if(output.createDirectory().failed())return false;stopTimer();audio.stop();bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto hadDefault = preferences->containsKey("algorithm.utauVoicebank");
    const auto oldDefault = preferences->getValue("algorithm.utauVoicebank");
    const juce::ScopeGuard restoreDefault([&]
    {
        if (hadDefault) preferences->setValue("algorithm.utauVoicebank", oldDefault);
        else preferences->removeValue("algorithm.utauVoicebank");
    });
    const auto legacyBank = output.getChildFile("legacy-default-bank");
    check("legacy_default_bank_fixture", legacyBank.createDirectory().wasOk());
    preferences->setValue("algorithm.utauVoicebank", legacyBank.getFullPathName());
    project.resetDocument(ProjectData{});project.dispatchPendingMessages();selectedTrackId.clear();selectedClipId.clear();selectedNoteId.clear();pianoRoll.setReadsVoicebankInBackground(false);
    setSize(1400,1000);const auto startRevision=project.revisionNumber();
    menuItemSelected(36,2);const auto hachiId=selectedTrackId;
    menuItemSelected(861,2);const auto utauId=selectedTrackId;
    menuItemSelected(862,2);const auto dsId=selectedTrackId;
    const auto created=project.snapshot();
    check("existing_track_menu_creates_three_distinct_persisted_types",created.tracks.size()==3&&hachiId!=utauId&&utauId!=dsId
        &&effectiveTrackType(created.tracks[0])==TrackType::hachi&&effectiveTrackType(created.tracks[1])==TrackType::utau&&effectiveTrackType(created.tracks[2])==TrackType::diffSinger);
    check("typed_creation_one_undo_each",project.revisionNumber()==startRevision+3&&project.canUndo());
    check("fresh_tracks_never_autobind_a_global_voicebank",std::all_of(created.tracks.begin(),created.tracks.end(),[](const auto& t){return t.voicebankDirectory==juce::File{};}));
    applyPreferences(); project.dispatchPendingMessages();
    const auto afterPreferences = project.snapshot();
    check("applying_preferences_never_binds_legacy_default", std::all_of(afterPreferences.tracks.begin(), afterPreferences.tracks.end(),
        [](const auto& track) { return track.voicebankDirectory == juce::File{}; }));
    const auto stable=project.contentFingerprint();const auto stableRevision=project.revisionNumber();
    const auto choose=[&](const juce::String& id){if(trackList.onTrackSelected)trackList.onTrackSelected(id);project.dispatchPendingMessages();refreshProjectControls();resized();};
    choose(hachiId);
    check("hachi_interface_has_native_source_and_expression_groups",displayedTrackType==TrackType::hachi&&trimButton.isVisible()&&connectButton.isVisible()
        &&pitchAlgorithm.isVisible()&&pitchParamButton.isVisible()&&driftParamButton.isVisible()&&breathParamButton.isVisible()&&!noteFlagsEditor.isVisible()&&!diffSingerTimingButton.isVisible());
    choose(utauId);
    check("utau_interface_has_oto_velocity_flags_not_native_source_tools",displayedTrackType==TrackType::utau&&wrenchButton.isVisible()&&noteAliasEditor.isVisible()
        &&noteConsonantVelocityEditor.isVisible()&&noteFlagsEditor.isVisible()&&flagCurveButton.isVisible()&&!trimButton.isVisible()&&!connectButton.isVisible()
        &&!pitchParamButton.isVisible()&&!diffSingerTimingButton.isVisible());
    choose(dsId);
    check("unconfigured_DS_still_has_actual_DS_editing_interface",displayedTrackType==TrackType::diffSinger&&diffSingerModelButton.isVisible()&&diffSingerTimingButton.isVisible()
        &&diffSingerPronunciationButton.isVisible()&&diffSingerPitchButton.isVisible()&&diffSingerParametersButton.isVisible()&&diffSingerSettingsButton.isVisible()
        &&diffSingerContextLabel.isVisible()&&noteAliasEditor.isVisible()&&!noteFlagsEditor.isVisible()&&!noteConsonantVelocityEditor.isVisible()
        &&pitchAlgorithm.isVisible()&&pitchAlgorithm.isEnabled()&&!stretchAlgorithm.isVisible()&&!trimButton.isVisible()&&!connectButton.isVisible()&&!wrenchButton.isVisible());
    check("DS_controls_reuse_existing_real_service_actions",bool(diffSingerModelButton.onClick)&&bool(diffSingerTimingButton.onClick)&&bool(diffSingerPronunciationButton.onClick)
        &&bool(diffSingerPitchButton.onClick)&&bool(diffSingerParametersButton.onClick)&&bool(diffSingerSettingsButton.onClick));
    choose(hachiId);choose(dsId);choose(utauId);choose(hachiId);
    check("interface_switch_is_view_only_no_score_engine_bank_type_mutation",project.contentFingerprint()==stable&&project.revisionNumber()==stableRevision);
    const auto current=project.snapshot();
    check("each_track_retains_own_type_after_repeated_selection",effectiveTrackType(current.tracks[0])==TrackType::hachi&&effectiveTrackType(current.tracks[1])==TrackType::utau&&effectiveTrackType(current.tracks[2])==TrackType::diffSinger);
    const auto screenshot=[&](const juce::String& id,const char* filename)
    {choose(id);auto out=output.getChildFile(filename).createOutputStream();if(out){out->setPosition(0);out->truncate();}return out&&juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()),*out);};
    check("actual_hachi_interface_screenshot",screenshot(hachiId,"interface-hachi.png"));
    check("actual_utau_interface_screenshot",screenshot(utauId,"interface-utau.png"));
    check("actual_DS_interface_screenshot",screenshot(dsId,"interface-diffsinger.png"));
    const auto onScreen=[&](const juce::Component& component){return !component.isVisible()||(component.getWidth()>10&&component.getHeight()>10&&getLocalBounds().contains(component.getBounds()));};
    setSize(980,1000);resized();
    check("DS_real_controls_have_nonempty_responsive_bounds",onScreen(diffSingerTimingButton)&&onScreen(diffSingerPronunciationButton)&&onScreen(diffSingerPitchButton)&&onScreen(diffSingerParametersButton)&&onScreen(diffSingerContextLabel));
    const auto saved=output.getChildFile("three-track-interfaces.hachi");ProjectModel loaded;
    check("per_track_type_roundtrip",project.save(saved,error)&&loaded.load(saved,error)&&loaded.snapshot().tracks.size()==3
        &&effectiveTrackType(loaded.snapshot().tracks[0])==TrackType::hachi&&effectiveTrackType(loaded.snapshot().tracks[1])==TrackType::utau&&effectiveTrackType(loaded.snapshot().tracks[2])==TrackType::diffSinger);
    const auto right=trackAreaMenu(std::nullopt);bool rightHachi=false,rightUtau=false,rightDS=false;
    juce::PopupMenu::MenuItemIterator items(right,true);
    while(items.next()){const auto& item=items.getItem();rightHachi|=item.itemID==1;rightUtau|=item.itemID==861;rightDS|=item.itemID==862;}
    check("empty_area_reuses_the_same_three_typed_actions",rightHachi&&rightUtau&&rightDS);
    for (auto mode : {UtauMode::classic, UtauMode::jie, UtauMode::mou})
    {
        choose(utauId);
        pitchAlgorithm.setSelectedId(utauModePickerItem(mode), juce::sendNotificationSync);
        project.dispatchPendingMessages(); refreshProjectControls(); resized();
        const auto unbound = project.snapshot();
        check("UTAU_mode_picker_keeps_voicebank_empty", unbound.tracks[1].utauMode == mode
            && unbound.tracks[1].voicebankDirectory == juce::File{});
        check("all_UTAU_modes_offer_real_voicebank_chooser", utauVoicebankButton.isVisible()
            && utauVoicebankButton.isEnabled() && bool(utauVoicebankButton.onClick));
        for (int width : {1400, 980, 640})
        {
            setSize(width, 1000); resized();
            check("voicebank_chooser_stays_right_of_connect_and_on_screen", onScreen(utauVoicebankButton)
                && utauVoicebankButton.getY() == spliceButton.getY()
                && utauVoicebankButton.getX() > spliceButton.getRight());
            check("voicebank_name_right_of_algorithm_only", onScreen(utauVoicebankName)
                && utauVoicebankName.getY()==pitchAlgorithm.getY()
                && utauVoicebankName.getX()>pitchAlgorithm.getRight()&&!trackEditingHeading.isVisible());
        }
        setSize(1400, 1000); resized();
        auto out = output.getChildFile("voicebank-" + utauModeKey(mode) + ".png").createOutputStream();
        check("voicebank_mode_screenshot", out && juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()), *out));
    }
    const auto bank = output.getChildFile("test-voicebank");
    check("voicebank_fixture_created", bank.createDirectory().wasOk());
    const auto previous = project.snapshot();
    const auto globalDefault = preferences->getValue("algorithm.utauVoicebank");
    ProjectModel blankReopened;
    check("unbound_track_save_reopen_stays_empty", project.save(output.getChildFile("unbound-track.hachi"), error)
        && blankReopened.load(output.getChildFile("unbound-track.hachi"), error)
        && blankReopened.snapshot().tracks[1].voicebankDirectory == juce::File{});
    check("restored_chooser_uses_existing_binding", selectVoicebank(utauId, bank));
    choose(utauId);
    const auto bound = project.snapshot();
    check("voicebank_binding_retains_mode_and_other_tracks", bound.tracks[1].voicebankDirectory == bank
        && bound.tracks[1].utauMode == UtauMode::mou && bound.tracks[1].trackType == TrackType::utau
        && bound.tracks[0].voicebankDirectory == previous.tracks[0].voicebankDirectory
        && bound.tracks[2].voicebankDirectory == previous.tracks[2].voicebankDirectory
        && preferences->getValue("algorithm.utauVoicebank") == globalDefault);
    check("chooser_tooltip_shows_bound_folder", utauVoicebankButton.getTooltip().contains(bank.getFullPathName()));
    check("bound_bank_name_and_path_visible", utauVoicebankName.isVisible()
        && utauVoicebankName.getText()==bank.getFileName() && utauVoicebankName.getTooltip().contains(bank.getFullPathName()));
    check("bound_voicebank_screenshot",screenshot(utauId,"voicebank-bound.png"));
    const auto namePoint=utauVoicebankName.getLocalBounds().toFloat().getCentre();
    const auto doubleNameClick=juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),namePoint,
        juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&utauVoicebankName,&utauVoicebankName,
        juce::Time::getCurrentTime(),namePoint,juce::Time::getCurrentTime(),2,false);
    mouseDoubleClick(doubleNameClick);
    check("name_doubleclick_opens_mode_specific_oto_library",voicebankWindows.findVisible<VoicebankSettingsComponent>(
        "bank|"+bank.getFullPathName()+"|"+juce::String(static_cast<int>(UtauMode::mou)))!=nullptr);
    voicebankWindows.clear();
    const auto boundRevision = project.revisionNumber();
    check("cancel_folder_choice_keeps_project", !selectVoicebank(utauId, {}) && project.revisionNumber() == boundRevision);
    check("voicebank_binding_undo_redo", project.undo()
        && project.snapshot().tracks[1].voicebankDirectory == previous.tracks[1].voicebankDirectory
        && project.redo() && project.snapshot().tracks[1].voicebankDirectory == bank);
    ProjectModel reopened;
    check("restored_chooser_binding_persists", project.save(output.getChildFile("voicebank-binding.hachi"), error)
        && reopened.load(output.getChildFile("voicebank-binding.hachi"), error)
        && reopened.snapshot().tracks[1].voicebankDirectory == bank);
    choose(hachiId); check("native_interface_keeps_UTAU_chooser_hidden", !utauVoicebankButton.isVisible()&&!utauVoicebankName.isVisible());
    choose(dsId); check("DS_keeps_own_model_chooser", !utauVoicebankButton.isVisible() && diffSingerModelButton.isVisible());
    verifyUtauSourceEditors(output,check);
    // Convert the existing musical track through the same menu action, not a replacement track.
    auto conversionFixture = project.snapshot();
    ClipData conversionClip; conversionClip.id="conversion-clip"; conversionClip.startSeconds=3; conversionClip.durationSeconds=2;
    NoteData conversionNote; conversionNote.id="conversion-note"; conversionNote.startSeconds=.25;
    conversionNote.durationSeconds=.8; conversionNote.midiNote=64; conversionNote.label=juce::String::fromUTF8("你好");
    conversionNote.pitchControlPoints={{.05,64.25f},{.65,65.5f}};
    conversionNote.utauFlags="g-10";
    conversionClip.notes.push_back(conversionNote); conversionFixture.tracks[1].clips.push_back(conversionClip);
    project.resetDocument(conversionFixture); project.dispatchPendingMessages(); choose(utauId);
    const auto musicPreserved=[&]()
    {
        const auto d=project.snapshot();const auto& t=d.tracks[1];const auto& c=t.clips.back();const auto& n=c.notes.front();
        return d.tracks.size()==3&&t.id==utauId&&c.id==conversionClip.id&&c.startSeconds==3&&c.durationSeconds==2
            &&n.id==conversionNote.id&&n.startSeconds==.25&&n.durationSeconds==.8&&n.midiNote==64&&n.label==conversionNote.label
            &&n.utauFlags==conversionNote.utauFlags&&n.pitchControlPoints.size()==2
            &&n.pitchControlPoints[0].timeSeconds==.05&&n.pitchControlPoints[0].targetMidi==64.25f
            &&n.pitchControlPoints[1].timeSeconds==.65&&n.pitchControlPoints[1].targetMidi==65.5f
            &&effectiveTrackType(d.tracks[0])==TrackType::hachi&&effectiveTrackType(d.tracks[2])==TrackType::diffSinger;
    };
    for(auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        project.setTrackUtauMode(utauId,mode);project.setTrackVoicebankDirectory(utauId,bank);project.dispatchPendingMessages();choose(utauId);
        check("unified_picker_has_all_algorithms_and_DS",pitchAlgorithm.getNumItems()==7
            &&pitchAlgorithm.indexOfItemId(10)>=0&&pitchAlgorithm.getSelectedId()==utauModePickerItem(mode));
        check("picker_replaces_heading_mode_menu_and_old_label",!pitchLabel.isVisible()
            &&!trackEditingHeading.isVisible()&&utauVoicebankName.getText()==bank.getFileName()
            &&pitchAlgorithm.getX()<utauVoicebankName.getX()&&utauVoicebankName.getX()<noteEditButton.getX()
            &&pitchAlgorithm.findColour(juce::ComboBox::backgroundColourId)==Palette::accent);
        const auto oldFingerprint=project.contentFingerprint(); const auto oldRevision=project.revisionNumber();
        pitchAlgorithm.setSelectedId(0,juce::sendNotificationSync);
        check("mode_menu_cancel_is_noop",project.contentFingerprint()==oldFingerprint&&project.revisionNumber()==oldRevision);
        pitchAlgorithm.setSelectedId(10,juce::sendNotificationSync);
        const auto converted=project.snapshot();
        check("all_UTAU_variants_convert_to_actual_DS",trackIsDiffSinger(converted.tracks[1])
            &&converted.tracks[1].pitchAlgorithm==PitchAlgorithm::utau&&converted.tracks[1].utauMode==UtauMode::mou
            &&converted.tracks[1].voicebankDirectory==juce::File{}&&musicPreserved());
        check("conversion_routes_audio_to_DS",AudioEngine::diagnosticUtauRequestedFamily(converted,conversionClip.id)
            ==backend::RequestedVoicebankFamily::diffSinger);
        check("conversion_refreshes_DS_editor",displayedTrackType==TrackType::diffSinger&&diffSingerModelButton.isVisible()
            &&diffSingerTimingButton.isVisible()&&!utauVoicebankButton.isVisible()&&!noteFlagsEditor.isVisible());
        check("conversion_single_undo_restores_full_track",project.revisionNumber()==oldRevision+1&&project.undo()
            &&project.contentFingerprint()==oldFingerprint);
        project.dispatchPendingMessages();
        check("conversion_undo_refreshes_UTAU_editor",displayedTrackType==TrackType::utau&&utauVoicebankButton.isVisible());
        check("conversion_redo",project.redo());project.dispatchPendingMessages();
        ProjectModel convertedReload;
        check("converted_track_type_and_music_save_reopen",project.save(output.getChildFile("converted-track.hachi"),error)
            &&convertedReload.load(output.getChildFile("converted-track.hachi"),error)
            &&trackIsDiffSinger(convertedReload.snapshot().tracks[1])
            &&convertedReload.snapshot().tracks[1].clips.back().notes.front().label==conversionNote.label);
        pitchAlgorithm.setSelectedId(7,juce::sendNotificationSync);
        check("DS_can_switch_back_to_UTAU",effectiveTrackType(project.snapshot().tracks[1])==TrackType::utau&&musicPreserved()
            &&utauVoicebankButton.isVisible()&&!diffSingerModelButton.isVisible());
    }
    const auto fakeDS=output.getChildFile("conversion-ds-model");fakeDS.createDirectory();fakeDS.getChildFile("dsconfig.yaml").replaceWithText("name: fixture\n");
    pitchAlgorithm.setSelectedId(10,juce::sendNotificationSync);project.setTrackVoicebankDirectory(utauId,fakeDS);project.dispatchPendingMessages();
    check("DS_model_name_reuses_header_position",utauVoicebankName.isVisible()&&!trackEditingHeading.isVisible()
        &&utauVoicebankName.getText()==fakeDS.getFileName()&&utauVoicebankName.getTooltip()==fakeDS.getFullPathName()
        &&utauVoicebankName.getX()>pitchAlgorithm.getRight());
    check("converted_DS_screenshot",screenshot(utauId,"converted-diffsinger.png"));
    const auto beforeEnvelope=project.contentFingerprint();
    check("DS_loudness_entry_and_presets_visible",volumeParamButton.isVisible()&&volumeParamButton.isEnabled()
        &&volumeParamButton.getButtonText()==juce::String::fromUTF8("响度包络")&&envelopePresetMenu.isVisible());
    volumeParamButton.onClick();resized();
    check("DS_loudness_opens_existing_amplitude_lane",envelopeLane==EnvelopeLane::amplitude
        &&pianoRoll.currentTool()==PianoRollComponent::Tool::amplitude&&volumeParamButton.getToggleState());
    volumeParamButton.onClick();resized();
    check("DS_loudness_closes_without_changing_score",envelopeLane==EnvelopeLane::none
        &&!volumeParamButton.getToggleState()&&project.contentFingerprint()==beforeEnvelope);
    // Exercise actual envelope mouse gestures without a model/OTO dependency.
    {
        ProjectModel envelopeModel;ProjectData d;TrackData t;t.id="ds-envelope";t.trackType=TrackType::diffSinger;
        t.pitchAlgorithm=PitchAlgorithm::utau;ClipData c;c.id="envelope-clip";c.durationSeconds=2;
        NoteData n;n.id="envelope-note";n.label="ni";n.startSeconds=.5;n.durationSeconds=1;n.midiNote=60;
        c.notes.push_back(n);t.clips.push_back(c);d.tracks.push_back(t);envelopeModel.resetDocument(d);
        PianoRollComponent roll(envelopeModel,strings);roll.setFocusedTrack(t.id);roll.setFocusedClip(c.id);
        roll.setSize(1200,600);roll.setPixelsPerSecond(400);roll.setTool(PianoRollComponent::Tool::amplitude);roll.diagnosticRefresh();
        const auto refresh=[&]{envelopeModel.dispatchPendingMessages();roll.diagnosticRefresh();};
        const auto event=[&](juce::Point<float> at,juce::Point<float> down,bool right=false){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(right?juce::ModifierKeys::rightButtonModifier:juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
        const auto flat=roll.diagnosticDrawnEnvelope(n.id);
        check("DS_default_is_unity_linear_no_UTAU_attack",flat.size()==2&&flat[0].gainDb==0&&flat[1].gainDb==0
            &&flat[0].linearToNext&&envelopeModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.empty());
        const juce::Point<float> middle(roll.diagnosticEdgeX(1.),roll.diagnosticAmplitudeLaneY(0));
        roll.mouseDoubleClick(event(middle,middle));refresh();
        check("DS_loudness_double_click_adds_point",envelopeModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==3);
        const juce::Point<float> lowered(middle.x,roll.diagnosticAmplitudeLaneY(-6.0206f));
        roll.mouseDown(event(middle,middle));roll.mouseDrag(event(lowered,middle));roll.mouseUp(event(lowered,middle));refresh();
        const auto edited=envelopeModel.snapshot().tracks[0].clips[0].notes[0];
        check("DS_loudness_drag_half_gain_preserves_music",edited.amplitudeEnvelope.size()==3
            &&std::abs(edited.amplitudeEnvelope[1].gainDb+6.0206f)<.02f&&edited.label==n.label
            &&edited.startSeconds==n.startSeconds&&edited.durationSeconds==n.durationSeconds&&edited.midiNote==n.midiNote);
        const auto fingerprint=envelopeModel.contentFingerprint();ProjectModel reloaded;
        check("DS_loudness_save_undo_redo",envelopeModel.save(output.getChildFile("ds-loudness.hachi"),error)
            &&reloaded.load(output.getChildFile("ds-loudness.hachi"),error)
            &&reloaded.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==3
            &&envelopeModel.undo()&&envelopeModel.redo()&&envelopeModel.contentFingerprint()==fingerprint);refresh();
        {auto stream=output.getChildFile("ds-loudness-lane.png").createOutputStream();
            check("DS_loudness_lane_screenshot",stream&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(roll.getLocalBounds()),*stream));}
        roll.mouseDown(event(lowered,lowered,true));refresh();
        check("DS_loudness_right_click_deletes_point",envelopeModel.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==2);
        roll.setSelectedNoteIds({n.id});
        check("DS_loudness_existing_presets_apply",roll.applyEnvelopePreset(PianoRollComponent::envelopePresets().front())==1);
    }
    pitchAlgorithm.setSelectedId(7,juce::sendNotificationSync);
    check("DS_model_not_reused_as_OTO_bank",project.snapshot().tracks[1].voicebankDirectory==juce::File{}&&musicPreserved());
    pitchAlgorithm.setSelectedId(3,juce::sendNotificationSync);
    check("picker_can_select_native_WORLD_mode",effectiveTrackType(project.snapshot().tracks[1])==TrackType::hachi&&trimButton.isVisible()&&musicPreserved());
    for(int algorithmId:{6,8,10,9,3})
    {
        const auto before=project.contentFingerprint();const auto revision=project.revisionNumber();
        pitchAlgorithm.setSelectedId(algorithmId,juce::sendNotificationSync);
        check("unified_algorithm_change_is_single_transaction",project.revisionNumber()==revision+1
            &&pitchAlgorithm.getSelectedId()==algorithmId&&musicPreserved());
        check("unified_algorithm_undo_restores_everything",project.undo()&&project.contentFingerprint()==before);
        project.dispatchPendingMessages();project.redo();project.dispatchPendingMessages();
    }
    for(int width:{1400,980,640})
    {
        setSize(width,1000);resized();
        check("unified_picker_visible_at_narrow_width",onScreen(pitchAlgorithm)&&onScreen(trackEditingHeading)
            &&pitchAlgorithm.getX()<trackEditingHeading.getX()&&!pitchLabel.isVisible());
    }
    setSize(1400,1000);resized();
    const auto finalRevision=project.revisionNumber();pitchAlgorithm.setSelectedId(3,juce::sendNotificationSync);project.setTrackEditingType("deleted-track",TrackType::diffSinger);
    check("same_mode_and_stale_target_are_noops",project.revisionNumber()==finalRevision);
    const auto written=output.getChildFile("track-editing-interfaces-ui.txt").replaceWithText(log);return ok&&written;
}
}
