#pragma once
#include "../MainComponent.h"
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
    check("hachi_interface_has_native_source_and_expression_groups",trackEditingHeading.getText().contains("HachiShifter")&&trimButton.isVisible()&&connectButton.isVisible()
        &&pitchAlgorithm.isVisible()&&pitchParamButton.isVisible()&&driftParamButton.isVisible()&&breathParamButton.isVisible()&&!noteFlagsEditor.isVisible()&&!diffSingerTimingButton.isVisible());
    choose(utauId);
    check("utau_interface_has_oto_velocity_flags_not_native_source_tools",trackEditingHeading.getText().contains("UTAU")&&wrenchButton.isVisible()&&noteAliasEditor.isVisible()
        &&noteConsonantVelocityEditor.isVisible()&&noteFlagsEditor.isVisible()&&flagCurveButton.isVisible()&&!trimButton.isVisible()&&!connectButton.isVisible()
        &&!pitchParamButton.isVisible()&&!diffSingerTimingButton.isVisible());
    choose(dsId);
    check("unconfigured_DS_still_has_actual_DS_editing_interface",trackEditingHeading.getText().contains("DiffSinger")&&diffSingerModelButton.isVisible()&&diffSingerTimingButton.isVisible()
        &&diffSingerPronunciationButton.isVisible()&&diffSingerPitchButton.isVisible()&&diffSingerParametersButton.isVisible()&&diffSingerSettingsButton.isVisible()
        &&diffSingerContextLabel.isVisible()&&noteAliasEditor.isVisible()&&!noteFlagsEditor.isVisible()&&!noteConsonantVelocityEditor.isVisible()
        &&!pitchAlgorithm.isVisible()&&!stretchAlgorithm.isVisible()&&!trimButton.isVisible()&&!connectButton.isVisible()&&!wrenchButton.isVisible());
    check("DS_controls_reuse_existing_real_service_actions",bool(diffSingerModelButton.onClick)&&bool(diffSingerTimingButton.onClick)&&bool(diffSingerPronunciationButton.onClick)
        &&bool(diffSingerPitchButton.onClick)&&bool(diffSingerParametersButton.onClick)&&bool(diffSingerSettingsButton.onClick));
    choose(hachiId);choose(dsId);choose(utauId);choose(hachiId);
    check("interface_switch_is_view_only_no_score_engine_bank_type_mutation",project.contentFingerprint()==stable&&project.revisionNumber()==stableRevision);
    const auto current=project.snapshot();
    check("each_track_retains_own_type_after_repeated_selection",effectiveTrackType(current.tracks[0])==TrackType::hachi&&effectiveTrackType(current.tracks[1])==TrackType::utau&&effectiveTrackType(current.tracks[2])==TrackType::diffSinger);
    const auto screenshot=[&](const juce::String& id,const char* filename)
    {choose(id);auto out=output.getChildFile(filename).createOutputStream();return out&&juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()),*out);};
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
            check("voicebank_name_right_of_chooser", onScreen(utauVoicebankName)
                && utauVoicebankName.getY()==utauVoicebankButton.getY()
                && utauVoicebankName.getX()>utauVoicebankButton.getRight());
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
        && utauVoicebankName.getText()==bank.getFileName() && utauVoicebankName.getTooltip()==bank.getFullPathName());
    check("bound_voicebank_screenshot",screenshot(utauId,"voicebank-bound.png"));
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
    const auto written=output.getChildFile("track-editing-interfaces-ui.txt").replaceWithText(log);return ok&&written;
}
}
