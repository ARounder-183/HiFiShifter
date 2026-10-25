#pragma once

namespace hachi
{
inline bool MainComponent::diagnosticNsfPicker()
{
    const auto configured = preferences != nullptr
        ? juce::File(preferences->getValue("algorithm.hifiganPath")) : juce::File{};
    if (!backend::NsfHifiganRenderer::modelAvailable(configured))
    {
        std::cout << "model_available=0\n";
        return false;
    }
    std::cout << "model_available=1\n";
    const auto testRoot = juce::SystemStats::getEnvironmentVariable("HACHI_TEST_SETTINGS_DIR", {});
    const auto folder = (testRoot.isNotEmpty() ? juce::File(testRoot)
        : juce::File::getSpecialLocation(juce::File::tempDirectory).getNonexistentChildFile("hachi-picker", ""))
        .getChildFile("mode-switch");
    const auto bank = folder.getChildFile("bank"); bank.createDirectory();
    {
        juce::AudioBuffer<float> samples(1, 4410); samples.clear();
        juce::WavAudioFormat format; auto stream = bank.getChildFile("a.wav").createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(), 44100, 1, 16, {}, 0));
        if (!writer || !writer->writeFromAudioSampleBuffer(samples, 0, samples.getNumSamples())) return false;
    }
    bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,10,-100,10,5\n");
    const auto previousBank = preferences->getValue("algorithm.utauVoicebank");
    const auto hadBank = preferences->containsKey("algorithm.utauVoicebank");
    const juce::ScopeGuard restoreBank([&] {
        if (hadBank) preferences->setValue("algorithm.utauVoicebank", previousBank);
        else preferences->removeValue("algorithm.utauVoicebank");
    });
    preferences->setValue("algorithm.utauVoicebank", bank.getFullPathName());
    const auto other = project.addTrack("Other track", true);
    project.setTrackPitchAlgorithm(other, PitchAlgorithm::world);
    const auto selected = project.addTrack("NSF picker", true);
    project.dispatchPendingMessages();
    diagnosticSelectTrack(selected);
    // This fixture checks retention of a deliberately selected bank, not an
    // automatic binding from the legacy default setting.
    if (!selectVoicebank(selected, bank)) return false;
    bool ok = true;
    juce::Array<juce::var> results;
    const auto check = [&](const juce::String& name, bool pass) {
        auto* row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("passed", pass);
        results.add(juce::var(row)); ok &= pass; std::cout << name << '=' << pass << '\n';
    };
    for (const auto id : { 3, 2, 6, 2, 8, 9, 7, 2 })
    {
        const auto started = juce::Time::getMillisecondCounterHiRes();
        pitchAlgorithm.setSelectedId(id, juce::sendNotificationSync);
        project.dispatchPendingMessages();
        refreshProjectControls();
        const auto expected = id == 2 ? PitchAlgorithm::nsfHifigan
            : id == 3 ? PitchAlgorithm::world : id >= 7 ? PitchAlgorithm::utau : PitchAlgorithm::llsm2;
        bool trackChanged = false, otherUnchanged = false;
        for (const auto& track : project.snapshot().tracks)
        {
            if (track.id == selected) trackChanged = track.pitchAlgorithm == expected
                && track.utauMode == (id == 8 ? UtauMode::jie : id == 9 ? UtauMode::mou : UtauMode::classic);
            if (track.id == other) otherUnchanged = track.pitchAlgorithm == PitchAlgorithm::world;
        }
        const auto passed = trackChanged && otherUnchanged && pitchAlgorithm.getSelectedId() == id;
        check("picker_" + juce::String(id) + "_persists_without_changing_other_track", passed);
        const auto elapsed = juce::Time::getMillisecondCounterHiRes() - started;
        check("picker_" + juce::String(id) + "_empty_track_responsive", elapsed < 1500.0);
        std::cout << "picker_" << id << "_milliseconds=" << elapsed << '\n';
    }
    const auto selectedState = [&]() {
        for (const auto& track : project.snapshot().tracks) if (track.id == selected) return track;
        return TrackData{};
    };
    check("native_mode_retains_bank_without_voicebank_synthesis", selectedState().voicebankDirectory == bank
        && selectedState().nativeNsfAudio && !trackUsesVoicebankSynthesis(selectedState())
        && trackShowsAllNativeRegions(selectedState()));
    juce::Array<juce::File> sidecars; bank.findChildFiles(sidecars, juce::File::findFiles, true, "*.hjm");
    check("empty_switch_does_not_convert_voicebank", sidecars.isEmpty());
    check("switch_undo", project.undo() && selectedState().pitchAlgorithm == PitchAlgorithm::utau);
    check("switch_redo", project.redo() && !trackUsesVoicebankSynthesis(selectedState()));
    juce::String error; ProjectModel reopened;
    const auto saved = folder.getChildFile("native-with-remembered-bank.hjpx");
    bool persisted = project.save(saved, error) && reopened.load(saved, error);
    for (const auto& track : reopened.snapshot().tracks) if (track.id == selected)
        persisted &= track.nativeNsfAudio && !trackUsesVoicebankSynthesis(track) && track.voicebankDirectory == bank;
    check("native_mode_save_reopen", persisted);
    auto legacy = selectedState(); legacy.nativeNsfAudio = false;
    check("legacy_nsf_voicebank_requests_still_supported", trackUsesVoicebankSynthesis(legacy)
        && effectiveUtauOutputEngine(legacy) == UtauOutputEngine::pcNsfHifigan);
    check("hifishifter_option_visible_only_nsf", stretchAlgorithm.indexOfItemId(6)>=0
        && stretchAlgorithmItemsFor(3)==std::vector<int>{1}
        && stretchAlgorithmItemsFor(7)==std::vector<int>{1});
    stretchAlgorithm.setSelectedId(6,juce::sendNotificationSync);
    project.dispatchPendingMessages(); refreshProjectControls();
    check("hifishifter_picker_updates_track", selectedState().stretchAlgorithm==StretchAlgorithm::hifiShifterMel
        && stretchAlgorithm.getSelectedId()==6);
    check("hifishifter_choice_undo",project.undo() && selectedState().stretchAlgorithm!=StretchAlgorithm::hifiShifterMel);
    check("hifishifter_choice_redo",project.redo() && selectedState().stretchAlgorithm==StretchAlgorithm::hifiShifterMel);
    ProjectModel stretchReopened;
    persisted=project.save(saved,error) && stretchReopened.load(saved,error);
    bool found=false;
    for(const auto& track:stretchReopened.snapshot().tracks) if(track.id==selected) {
        found=true; persisted &= track.stretchAlgorithm==StretchAlgorithm::hifiShifterMel;
    }
    check("hifishifter_choice_save_reopen",persisted && found);
    check("wcs_nonloop_option_visible_only_nsf", stretchAlgorithm.indexOfItemId(7)>=0
        && stretchAlgorithmItemsFor(3)==std::vector<int>{1} && stretchAlgorithmItemsFor(7)==std::vector<int>{1});
    stretchAlgorithm.setSelectedId(7,juce::sendNotificationSync);
    project.dispatchPendingMessages(); refreshProjectControls();
    check("wcs_nonloop_picker_updates_track", selectedState().stretchAlgorithm==StretchAlgorithm::wcsNonLoop
        && stretchAlgorithm.getSelectedId()==7);
    check("wcs_nonloop_choice_undo",project.undo() && selectedState().stretchAlgorithm==StretchAlgorithm::hifiShifterMel);
    check("wcs_nonloop_choice_redo",project.redo() && selectedState().stretchAlgorithm==StretchAlgorithm::wcsNonLoop);
    ProjectModel wcsReopened;
    persisted=project.save(saved,error) && wcsReopened.load(saved,error); found=false;
    for (const auto& track:wcsReopened.snapshot().tracks) if (track.id==selected) {
        found=true; persisted &= track.stretchAlgorithm==StretchAlgorithm::wcsNonLoop;
    }
    check("wcs_nonloop_choice_save_reopen",persisted && found);
    const auto shot=folder.getChildFile("wcs-nonloop-selected.png").createOutputStream();
    juce::PNGImageFormat wcsPng;
    check("wcs_nonloop_selected_screenshot",shot && wcsPng.writeImageToStream(createComponentSnapshot(getLocalBounds()),*shot));
    check("pitch_smoothing_checkbox_visible_in_native_nsf",nsfPitchTransitionsButton.isVisible()
        && nsfPitchTransitionsButton.getToggleState());
    const auto smoothRevision=project.revisionNumber();
    nsfPitchTransitionsButton.setToggleState(false,juce::dontSendNotification);nsfPitchTransitionsButton.onClick();
    project.dispatchPendingMessages();refreshProjectControls();
    check("pitch_smoothing_checkbox_updates_track",!selectedState().nsfSmoothPitchTransitions
        && !nsfPitchTransitionsButton.getToggleState() && project.revisionNumber()==smoothRevision+1);
    bool otherSmooth=true;for(const auto& t:project.snapshot().tracks)if(t.id==other)otherSmooth=t.nsfSmoothPitchTransitions;
    check("pitch_smoothing_changes_only_selected_track",otherSmooth);
    check("pitch_smoothing_undo",project.undo() && selectedState().nsfSmoothPitchTransitions);
    check("pitch_smoothing_redo",project.redo() && !selectedState().nsfSmoothPitchTransitions);
    ProjectModel smoothReopened;persisted=project.save(saved,error) && smoothReopened.load(saved,error);found=false;
    for(const auto& t:smoothReopened.snapshot().tracks)if(t.id==selected){found=true;persisted &= !t.nsfSmoothPitchTransitions;}
    check("pitch_smoothing_save_reopen",persisted && found);
    auto probeTrack=selectedState();ClipData probe;probe.id="smooth-probe";probe.sourceFile=bank.getChildFile("a.wav");
    probe.durationSeconds=probe.sourceDurationSeconds=.1;
    NoteData probeNote;probeNote.id="smooth-probe-note";probeNote.durationSeconds=.1;probeNote.midiNote=60;
    probeNote.contour={{0,0,0,true},{.1,0,0,true}};probe.notes.push_back(probeNote);probeTrack.clips={probe};
    ProjectData probeProject;probeProject.tracks={probeTrack};
    check("wcs_nonloop_single_and_phrase_requests", AudioEngine::diagnosticNativeRequest(probe,probeTrack).stretchAlgorithm==6
        && AudioEngine::mergedRequestFor({&probe},probeTrack,{},{}).stretchAlgorithm==6);
    auto previousStretch=probeTrack;previousStretch.stretchAlgorithm=StretchAlgorithm::hifiShifterMel;
    const auto wcsWaveHash=AudioEngine::nativeClipWaveformHash(probe,probeTrack);
    const auto wcsAudioKey=AudioEngine::diagnosticUtauRenderKey(probeProject,probe.id);
    auto oldProject=probeProject;oldProject.tracks[0]=previousStretch;
    check("wcs_nonloop_invalidates_audio_and_waveform",wcsWaveHash!=AudioEngine::nativeClipWaveformHash(probe,previousStretch)
        && wcsAudioKey!=AudioEngine::diagnosticUtauRenderKey(oldProject,probe.id));
    const auto offKey=AudioEngine::diagnosticUtauRenderKey(probeProject,probe.id);
    const auto offHash=AudioEngine::nativeClipWaveformHash(probe,probeTrack);
    const auto singleOff=AudioEngine::diagnosticNativeRequest(probe,probeTrack);
    const auto phraseOff=AudioEngine::mergedRequestFor({&probe},probeTrack,{},{});
    probeProject.tracks[0].nsfSmoothPitchTransitions=true;
    check("pitch_smoothing_invalidates_audio_and_waveform_cache",offKey!=AudioEngine::diagnosticUtauRenderKey(probeProject,probe.id)
        && offHash!=AudioEngine::nativeClipWaveformHash(probe,probeProject.tracks[0]));
    check("pitch_smoothing_single_and_phrase_requests",!singleOff.nsfSmoothPitchTransitions && !phraseOff.nsfSmoothPitchTransitions
        && AudioEngine::diagnosticNativeRequest(probe,probeProject.tracks[0]).nsfSmoothPitchTransitions);
    check("noise_protection_checkbox_visible_and_default_on", nsfNoiseProtectionButton.isVisible()
        && nsfNoiseProtectionButton.getToggleState() && selectedState().nsfNoiseProtection);
    const auto noiseRevision = project.revisionNumber();
    nsfNoiseProtectionButton.setToggleState(false, juce::dontSendNotification); nsfNoiseProtectionButton.onClick();
    project.dispatchPendingMessages(); refreshProjectControls();
    check("noise_protection_checkbox_updates_track", !selectedState().nsfNoiseProtection
        && !nsfNoiseProtectionButton.getToggleState() && project.revisionNumber() == noiseRevision + 1);
    bool otherProtected = false; for (const auto& t : project.snapshot().tracks) if (t.id == other) otherProtected = t.nsfNoiseProtection;
    check("noise_protection_changes_only_selected_track", otherProtected && !selectedState().nsfSmoothPitchTransitions);
    check("noise_protection_undo", project.undo() && selectedState().nsfNoiseProtection);
    check("noise_protection_redo", project.redo() && !selectedState().nsfNoiseProtection);
    ProjectModel noiseReopened; persisted = project.save(saved, error) && noiseReopened.load(saved, error); found = false;
    for (const auto& t : noiseReopened.snapshot().tracks) if (t.id == selected) { found = true; persisted &= !t.nsfNoiseProtection; }
    check("noise_protection_save_reopen", persisted && found);
    juce::MemoryBlock legacyBytes; saved.loadFileAsData(legacyBytes);
    auto legacyTree = juce::ValueTree::readFromData(legacyBytes.getData(), legacyBytes.getSize());
    for (auto trackTree : legacyTree) if (trackTree.hasType("Track")) trackTree.removeProperty("nsfNoiseProtection", nullptr);
    juce::MemoryOutputStream legacyStream; legacyTree.writeToStream(legacyStream);
    const auto legacyFile = folder.getChildFile("legacy-no-noise-option.hjpx");
    ProjectModel legacyReopened;
    persisted = legacyFile.replaceWithData(legacyStream.getData(), legacyStream.getDataSize()) && legacyReopened.load(legacyFile, error); found = false;
    for (const auto& t : legacyReopened.snapshot().tracks) if (t.id == selected) { found = true; persisted &= t.nsfNoiseProtection; }
    check("legacy_project_keeps_previous_noise_protection", persisted && found);
    probeTrack.nsfNoiseProtection = false; probeProject.tracks = {probeTrack};
    const auto noiseOffKey = AudioEngine::diagnosticUtauRenderKey(probeProject, probe.id);
    const auto noiseOffHash = AudioEngine::nativeClipWaveformHash(probe, probeTrack);
    const auto noiseSingle = AudioEngine::diagnosticNativeRequest(probe, probeTrack);
    const auto noisePhrase = AudioEngine::mergedRequestFor({&probe}, probeTrack, {}, {});
    probeProject.tracks[0].nsfNoiseProtection = true;
    check("noise_protection_invalidates_audio_and_waveform_cache", noiseOffKey != AudioEngine::diagnosticUtauRenderKey(probeProject, probe.id)
        && noiseOffHash != AudioEngine::nativeClipWaveformHash(probe, probeProject.tracks[0]));
    check("noise_protection_single_and_phrase_requests", !noiseSingle.nsfNoiseProtection && !noisePhrase.nsfNoiseProtection
        && AudioEngine::diagnosticNativeRequest(probe, probeProject.tracks[0]).nsfNoiseProtection);
    setSize(900, 760); resized();
    check("native_switches_fit_minimum_window", getLocalBounds().contains(nsfNoiseProtectionButton.getBounds())
        && nsfNoiseProtectionButton.getWidth() == 156 && !nsfNoiseProtectionButton.getBounds().intersects(nsfPitchTransitionsButton.getBounds()));
    {
        auto stream = folder.getChildFile("noise-option-minimum-width.png").createOutputStream();
        const auto bounds = advancedEnvelopeButton.getBounds().getUnion(nsfNoiseProtectionButton.getBounds()).expanded(4);
        juce::PNGImageFormat png;
        check("native_switches_screenshot_written", stream && png.writeImageToStream(createComponentSnapshot(bounds), *stream));
    }
    setSize(1280, 760); resized();
    pitchAlgorithm.setSelectedId(3,juce::sendNotificationSync);
    project.dispatchPendingMessages();refreshProjectControls();
    check("switch_backend_drops_nsf_only_stretch",stretchAlgorithm.indexOfItemId(7)<0 && stretchAlgorithm.indexOfItemId(6)<0
        && selectedState().stretchAlgorithm==StretchAlgorithm::melodyneHybrid);
    check("pitch_smoothing_checkbox_hidden_on_other_backends",!nsfPitchTransitionsButton.isVisible());
    check("noise_protection_hidden_on_other_backends", !nsfNoiseProtectionButton.isVisible());
    pitchAlgorithm.setSelectedId(7,juce::sendNotificationSync);project.dispatchPendingMessages();refreshProjectControls();
    check("pitch_smoothing_checkbox_hidden_in_utau",!nsfPitchTransitionsButton.isVisible());
    check("noise_protection_hidden_in_utau", !nsfNoiseProtectionButton.isVisible());
    pitchAlgorithm.setSelectedId(2,juce::sendNotificationSync);project.dispatchPendingMessages();refreshProjectControls();
    check("pitch_smoothing_restored_when_returning_nsf",nsfPitchTransitionsButton.isVisible() && !nsfPitchTransitionsButton.getToggleState());
    check("noise_protection_restored_when_returning_nsf", nsfNoiseProtectionButton.isVisible() && !nsfNoiseProtectionButton.getToggleState());
    auto* report = new juce::DynamicObject(); report->setProperty("passed", ok); report->setProperty("checks", results);
    folder.getChildFile("validation.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
    return ok;
}
}
