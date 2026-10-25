#pragma once
#include "../UtauOutputEnginePanel.h"
#include "../backend/HifisamplerFlags.h"

namespace hachi
{
inline bool MainComponent::diagnosticOutputEngine(const juce::File& folder,
    const juce::File& modelFolder, const juce::File& wavtool)
{
    folder.createDirectory();
    bool ok = true;
    juce::Array<juce::var> checks;
    const auto check = [&](const char* name, bool passed) {
        auto* value = new juce::DynamicObject(); value->setProperty("name", name); value->setProperty("passed", passed);
        checks.add(juce::var(value)); ok = ok && passed; std::cout << name << '=' << passed << '\n';
    };
    const auto bank = folder.getChildFile("bank"); bank.createDirectory();
    juce::AudioBuffer<float> source(1, 44100);
    for (int i = 0; i < source.getNumSamples(); ++i)
        source.setSample(0, i, .2f * std::sin(static_cast<float>(i * 2 * juce::MathConstants<double>::pi * 220 / 44100)));
    {
        juce::WavAudioFormat format; auto stream = bank.getChildFile("a.wav").createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(), 44100, 1, 16, {}, 0));
        if (!writer || !writer->writeFromAudioSampleBuffer(source, 0, source.getNumSamples())) return false;
    }
    bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,100,-1000,100,30\n");
    bank.getChildFile("oto.jie.ini").replaceWithText("a.wav=a,0,100,-1000,100,30\n");
    bank.getChildFile("oto4.ini").replaceWithText("a.wav=0,100,200,700\n");
    bank.getChildFile("otomou.ini").replaceWithText("a.wav=CVVC,0,100,200,700\n");
    NoteData note; note.id = "engine-note"; note.label = "a"; note.startSeconds = .3; note.durationSeconds = .7;
    ClipData clip; clip.id = "engine-clip"; clip.durationSeconds = 1.2; clip.notes = { note };
    TrackData track; track.id = "engine-track"; track.name = "Output engine";
    track.pitchAlgorithm = PitchAlgorithm::utau; track.utauMode = UtauMode::mou;
    track.voicebankDirectory = bank; track.clips = { clip };
    TrackData other = track; other.id = "other"; other.clips.clear();
    ProjectData data; data.tracks = { track, other };
    project.resetDocument(data); project.dispatchPendingMessages(); diagnosticSelectTrack(track.id);
    check("duplicate_nsf_jie_mou_algorithms_removed", pitchAlgorithm.indexOfItemId(10) < 0 && pitchAlgorithm.indexOfItemId(11) < 0);
    check("ordinary_mou_mode_visible", pitchAlgorithm.getSelectedId() == 9);
    auto menu = clipContextMenu(clip.id, .5);
    bool submenu = false, allChoices = false;
    for (juce::PopupMenu::MenuItemIterator i(menu); i.next();)
        if (i.getItem().subMenu != nullptr && i.getItem().text == strings.text("output.choose")) {
            submenu = i.getItem().isEnabled;
            allChoices = i.getItem().subMenu->getNumItems() == 3;
        }
    check("region_menu_has_three_output_engine_choices", submenu && allChoices);
    auto emptyMenu = trackAreaMenu(TimelineComponent::Anchor { track.id, .9 });
    bool emptySubmenu = false;
    for (juce::PopupMenu::MenuItemIterator i(emptyMenu); i.next();) emptySubmenu |= i.getItem().subMenu != nullptr;
    check("empty_track_lane_has_output_engine_menu", emptySubmenu);
    const auto oldMenu = trackList.onTrackMenu;
    bool headerRequested = false;
    trackList.onTrackMenu = [&](const auto& id, auto) { headerRequested = id == track.id; };
    trackList.diagnosticRightClick(50, 58);
    trackList.onTrackMenu = oldMenu;
    check("track_header_right_click_routes_without_toggling_mute", headerRequested && !project.snapshot().tracks[0].muted);
    outputEngineItemChosen(102, track.id); project.dispatchPendingMessages();
    auto changed = project.snapshot();
    {
        auto display=project.snapshot();display.tracks[0].clips[0].sourceFile=folder.getChildFile("Tokyo Teddy Bear.mid");
        display.tracks[0].clips[0].durationSeconds=8;
        ProjectModel preview;preview.resetDocument(display);TimelineComponent lane(preview);
        lane.outputEngineNameProvider=[&](const auto& t){return audio.outputEngineDisplayName(t);};
        lane.setPixelsPerSecond(140);lane.setRowHeight(96);
        const auto image=lane.createComponentSnapshot({0,0,1200,lane.getHeight()});
        juce::PNGImageFormat png;auto stream=folder.getChildFile("region-engine-label.png").createOutputStream();
        if(stream)png.writeImageToStream(image,*stream);
    }
    check("native_engine_selection_preserves_mou_and_other_track", changed.tracks[0].pitchAlgorithm == PitchAlgorithm::utau
        && changed.tracks[0].utauMode == UtauMode::mou && changed.tracks[0].outputEngine == UtauOutputEngine::pcNsfHifigan
        && changed.tracks[1].outputEngine == UtauOutputEngine::inherit);
    check("native_selection_is_undoable", project.undo() && project.snapshot().tracks[0].outputEngine == UtauOutputEngine::inherit);
    check("native_selection_is_redoable", project.redo() && project.snapshot().tracks[0].outputEngine == UtauOutputEngine::pcNsfHifigan);
    const auto nativeName=juce::String::fromUTF8("HiFisampler（PC-NSF-HiFiGAN）");
    audio.setUtauOutputDefaults(UtauOutputEngine::pcNsfHifigan,{});
    check("inherited_output_label_resolves_actual_hifisampler_default",audio.outputEngineDisplayName(data.tracks[0])==nativeName);
    audio.setUtauOutputDefaults(UtauOutputEngine::resampler,{});
    check("inherited_output_label_names_actual_bundled_resampler",audio.outputEngineDisplayName(data.tracks[0])
        ==audio.currentUtauResamplerFile().getFileNameWithoutExtension());
    check("explicit_hifisampler_overrides_resampler_header_default",timeline.clipHeaderText(changed.tracks[0],clip,1).contains(nativeName));
    auto customTrack=data.tracks[0];customTrack.outputEngine=UtauOutputEngine::resampler;
    customTrack.outputResampler=folder.getChildFile("custom-resampler.exe");
    check("per_track_resampler_name_shown_in_region_header",timeline.clipHeaderText(customTrack,clip,1).contains("custom-resampler"));
    auto backing=customTrack;backing.accompaniment=true;
    check("accompaniment_has_no_synthesis_engine_caption",audio.outputEngineDisplayName(backing).isEmpty());
    const auto classicKey = AudioEngine::diagnosticUtauRenderKey(data, clip.id);
    const auto nativeKey = AudioEngine::diagnosticUtauRenderKey(data, clip.id, UtauOutputEngine::pcNsfHifigan);
    check("global_default_changes_inherited_cache_key", classicKey != nativeKey);
    refreshDiffSingerFlagContext();
    check("hifisampler_ui_uses_own_curve_units", pianoRoll.flagLaneFlag() == "HIFI:g"
        && flagCurveKindFor("HIFI:g").maximum == 600 && flagCurveKindFor("g").maximum == 50
        && flagCurveKindFor("HIFI:Hb").defaultValue == 100);

    // Native numeric curves: absolute native values, legacy fallback and
    // preutterance use the same nominal-note time in every editing mode.
    for (const auto& kind : hifisamplerFlagCurveKinds()) {
        const auto name=juce::String(kind.flag).substring(5);
        backend::UtauNoteRenderSpec n;n.flags=name+juce::String(kind.defaultValue);n.flagCurve=true;
        n.flagSplit=true;n.regionFlags[1]=name+juce::String(kind.maximum);
        n.flagCurves={{name,{{-.1,kind.minimum},{1.1,kind.maximum}}},
            {kind.flag,{{-.1,kind.maximum},{1.1,kind.minimum}}}};
        const std::array<double,4> spans{.1,.2,.4,.4};
        const auto value=backend::hifiFlagsAt(n,spans,4,.6,-.1).get(name);
        check(("hifi_linear_"+name+"_overrides_region_and_legacy").toRawUTF8(),std::abs(value-(kind.minimum+kind.maximum)*.5f)<.001f);
        n.flagCurve=false;
        check(("hifi_linear_"+name+"_off_restores_text").toRawUTF8(),backend::hifiFlagsAt(n,spans,4,.6,-.1).get(name)==kind.defaultValue);
    }
    auto legacyData=data;legacyData.tracks.resize(1);auto& legacyNote=legacyData.tracks[0].clips[0].notes[0];
    legacyNote.utauFlagCurveEnabled=true;legacyNote.utauFlags="g120Hb50";
    legacyNote.utauFlagCurves={{"g",{{0,-40},{1,40}}},{"Mt",{{0,-20},{1,20}}}};
    ProjectModel legacyCurve;legacyCurve.replace(legacyData);
    const auto oldKey=AudioEngine::diagnosticUtauRenderKey(legacyCurve.snapshot(),clip.id,UtauOutputEngine::pcNsfHifigan);
    const auto nativeCurveRequest=[&] {const auto d=legacyCurve.snapshot();return AudioEngine::diagnosticUtauRequestNotes(d,clip.id).front();};
    const auto readNative=[&] {return backend::hifiFlagsAt(nativeCurveRequest(),{.1,.2,.4,.3},4,.5,0).get("g");};
    check("hifi_legacy_linear_curve_is_read",readNative()==0);
    legacyCurve.setNoteUtauFlagCurve(legacyNote.id,"HIFI:g",{{0,300},{1,300}});
    check("hifi_explicit_curve_has_priority",readNative()==300);
    check("hifi_reset_native_curve",legacyCurve.resetNotesUtauFlagCurve({legacyNote.id},"HIFI:g"));
    check("hifi_reset_restores_text_and_keeps_other_engine",readNative()==120
        &&flagCurvePointsFor(legacyCurve.snapshot().tracks[0].clips[0].notes[0],"g").size()==2
        &&flagCurvePointsFor(legacyCurve.snapshot().tracks[0].clips[0].notes[0],"Mt").size()==2
        &&flagCurvePointsFor(legacyCurve.snapshot().tracks[0].clips[0].notes[0],"HIFI:g").empty());
    check("hifi_reset_invalidates_phrase_cache",oldKey!=AudioEngine::diagnosticUtauRenderKey(legacyCurve.snapshot(),clip.id,UtauOutputEngine::pcNsfHifigan));
    legacyCurve.undo();check("hifi_reset_undo_restores_native_curve",readNative()==300);
    legacyCurve.redo();check("hifi_reset_redo_restores_fixed_flag",readNative()==120);
    juce::String curveError;ProjectModel reopened;
    check("hifi_reset_survives_save_reopen",legacyCurve.save(folder.getChildFile("cleared-native-curve.hjpx"),curveError)
        &&reopened.load(folder.getChildFile("cleared-native-curve.hjpx"),curveError)
        &&backend::hifiFlagsAt(AudioEngine::diagnosticUtauRequestNotes(reopened.snapshot(),clip.id).front(),{.1,.2,.4,.3},4,.5,0).get("g")==120);
    legacyCurve.replace(legacyData);
    check("hifi_reset_unprefixed_curve_directly",legacyCurve.resetNotesUtauFlagCurve({legacyNote.id},"HIFI:g")&&readNative()==120);
    legacyCurve.replace(legacyData);
    check("hifi_clear_native_lane_suppresses_legacy",legacyCurve.setNoteUtauFlagCurve(legacyNote.id,"HIFI:g",{})&&readNative()==120);
    check("hifi_clear_is_idempotent",!legacyCurve.setNoteUtauFlagCurve(legacyNote.id,"HIFI:g",{}));
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou}){
        legacyCurve.setTrackUtauMode(legacyData.tracks[0].id,mode);
        legacyCurve.setNoteUtauFlagCurve(legacyNote.id,"HIFI:t",{{-.1,-1200},{1.1,1200,PitchCurveShape::easeIn}});
        const auto n=nativeCurveRequest();
        const auto t=backend::hifiFlagsAt(n,{.1,.2,.4,.3},4,.5,0).get("t");
        check(("hifi_shaped_curve_"+utauModeKey(mode)).toRawUTF8(),n.flagCurve&&t<-100&&t>-1000);
    }

    check("explicit_engine_ignores_global_backend", effectiveUtauOutputEngine(changed.tracks[0], UtauOutputEngine::resampler) == UtauOutputEngine::pcNsfHifigan);
    const auto toolFile = folder.getChildFile("tool.exe"); toolFile.replaceWithText("serialization fixture");
    project.setTrackOutputEngine(track.id, UtauOutputEngine::resampler, toolFile, wavtool);
    auto withTools = project.snapshot();
    check("per_track_tool_paths_invalidate_cache", classicKey != AudioEngine::diagnosticUtauRenderKey(withTools, clip.id));
    juce::String error; const auto saved = folder.getChildFile("output-engine.hjpx"); ProjectModel loaded;
    check("engine_and_both_paths_roundtrip", project.save(saved, error) && loaded.load(saved, error)
        && loaded.snapshot().tracks[0].outputEngine == UtauOutputEngine::resampler
        && loaded.snapshot().tracks[0].outputResampler == toolFile && loaded.snapshot().tracks[0].outputWavtool == wavtool);
    ProjectModel legacy;
    const auto legacyFile = folder.getChildFile("legacy.hjpx");
    check("legacy_native_jie_mou_projects_migrate_without_losing_engine", legacy.load(legacyFile, error)
        && legacy.snapshot().tracks[0].pitchAlgorithm == PitchAlgorithm::utau
        && legacy.snapshot().tracks[0].outputEngine == UtauOutputEngine::pcNsfHifigan
        && legacy.snapshot().tracks[0].utauMode == UtauMode::mou
        && legacy.snapshot().tracks[0].clips[0].notes[0].utauJieSplitSet);
    outputEngineItemChosen(101, track.id);
    check("follow_settings_clears_overrides", project.snapshot().tracks[0].outputEngine == UtauOutputEngine::inherit
        && project.snapshot().tracks[0].outputResampler == juce::File{} && project.snapshot().tracks[0].outputWavtool == juce::File{});
    UtauOutputEnginePanel panel(strings, "Output engine", {}, {});
    panel.setLookAndFeel(&lookAndFeel);
    {
        juce::PNGImageFormat png; auto stream = folder.getChildFile("resampler-panel.png").createOutputStream();
        check("resampler_settings_panel_snapshot", stream && png.writeImageToStream(panel.createComponentSnapshot(panel.getLocalBounds()), *stream));
    }
    panel.setLookAndFeel(nullptr);
    auto native = data; native.tracks.resize(1); native.tracks[0].outputEngine = UtauOutputEngine::pcNsfHifigan;
    native.tracks[0].clips[0].notes[0].utauFlags = "g20";
    check("native_engine_accepts_hifisampler_flags_in_ordinary_mou", AudioEngine::renderCapabilityWarnings(native).isEmpty());
    native.tracks[0].clips[0].notes[0].utauFlags = "Mt20";
    check("native_engine_reports_wcs_only_flags", !AudioEngine::renderCapabilityWarnings(native).isEmpty());
    check("native_engine_disables_component_export", AudioEngine::componentExportIssue(native, toolFile).isNotEmpty());
    native.tracks[0].clips[0].notes[0].utauFlags.clear();
    AudioEngine engine; engine.setHifiganModelDirectory(modelFolder);
    engine.setUtauOutputDefaults(UtauOutputEngine::pcNsfHifigan, {});
    engine.selectEveryUtauNote(native); engine.syncProject(native);
    const auto wait = [&] { for (int spin = 0; spin < 2000 && engine.renderProgress(); ++spin) juce::Thread::sleep(20); };
    wait();
    check("ordinary_mou_routes_to_real_native_model", engine.hasCurrentRenderedAudio()
        && engine.activeRenderBackends().contains("hifisampler-native-voicebank") && engine.activeRenderWarnings().isEmpty());
    engine.refreshUtauWaveforms();
    check("native_backend_publishes_note_waveform_and_four_regions", engine.utauNoteWaveforms() && !engine.utauNoteWaveforms()->empty()
        && engine.utauNoteWaveforms()->front().phonemes.size() == 4);
    check("native_engine_exports_audio_from_ordinary_mou", engine.exportWav(folder.getChildFile("ordinary-mou-native.wav"), error));
    const auto renders=engine.diagnosticScheduledNsfUtauRenders();
    engine.syncProject(native); wait();
    check("ordinary_sync_reuses_native_phrase_cache",engine.diagnosticScheduledNsfUtauRenders()==renders);
    const auto wavKey=AudioEngine::diagnosticUtauRenderKey(native,clip.id);
    const auto sampleFile=bank.getChildFile("a.wav");
    source.setSize(1,66150); // A longer recording also changes OTO duration metadata.
    for(int j=0;j<source.getNumSamples();++j) source.setSample(0,j,.2f*std::sin(static_cast<float>(j*2*juce::MathConstants<double>::pi*330/44100)));
    sampleFile.deleteFile();
    {juce::WavAudioFormat format;auto stream=sampleFile.createOutputStream();
     std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,16,{},0));
     if(!writer || !writer->writeFromAudioSampleBuffer(source,0,source.getNumSamples())) return false;}
    check("replaced_wav_changes_outer_phrase_key",wavKey!=AudioEngine::diagnosticUtauRenderKey(native,clip.id));
    engine.syncProject(native); wait();
    check("replaced_wav_queues_new_native_phrase",engine.diagnosticScheduledNsfUtauRenders()==renders+1 && engine.hasCurrentRenderedAudio());
    const auto resolved=backend::UtauRenderer::resolveVoiceSample(bank,"a",60,100,true,true);
    check("replaced_wav_refreshes_voicebank_duration",std::abs(resolved.fileSeconds-1.5)<1.e-6);
    check("replaced_wav_exports_new_audio",engine.exportWav(folder.getChildFile("replaced-source.wav"),error)
        && juce::SHA256(folder.getChildFile("ordinary-mou-native.wav"))!=juce::SHA256(folder.getChildFile("replaced-source.wav")));
    native.tracks[0].clips[0].notes[0].utauFlags="G";
    engine.setUtauRenderNoteSelection({note.id}); engine.syncProject(native); wait();
    const auto forced=engine.diagnosticScheduledNsfUtauRenders();
    engine.syncProject(native); wait();
    check("g_does_not_loop_on_ui_sync",engine.diagnosticScheduledNsfUtauRenders()==forced);
    engine.setUtauRenderNoteSelection({note.id}); engine.syncProject(native); wait();
    check("g_new_selection_forces_phrase_regeneration",engine.diagnosticScheduledNsfUtauRenders()==forced+1 && engine.hasCurrentRenderedAudio());
    engine.selectEveryUtauNote(native); engine.syncProject(native); wait();
    check("g_export_selection_forces_phrase_regeneration",engine.diagnosticScheduledNsfUtauRenders()==forced+2 && engine.hasCurrentRenderedAudio());
    native.tracks[0].clips[0].notes[0].utauFlags="HG30";
    engine.syncProject(native); wait(); const auto growl=engine.diagnosticScheduledNsfUtauRenders();
    engine.setUtauRenderNoteSelection({note.id}); engine.syncProject(native); wait();
    check("hg_is_not_misread_as_force_g",engine.diagnosticScheduledNsfUtauRenders()==growl);
    native.tracks[0].clips[0].notes[0].utauFlags.clear();
    native.tracks[0].outputEngine = UtauOutputEngine::inherit;
    engine.syncProject(native); wait();
    check("inherited_global_native_engine_routes_identically", engine.hasCurrentRenderedAudio() && engine.activeRenderBackends().contains("hifisampler-native-voicebank"));
    native.tracks[0].outputEngine = UtauOutputEngine::resampler;
    // A labelled piano-free native fallback exercises the real classic assembly without loading a second neural daemon.
    engine.setUtauResamplerFile({});
    native.tracks[0].outputResampler = folder.getChildFile("missing.exe");
    backend::UtauRenderRequest request; request.voicebankDirectory = bank; request.targetDurationSeconds = 1.2;
    request.fourRegion = request.consonantClasses = true;
    backend::UtauNoteRenderSpec spec; spec.alias = "a"; spec.startSeconds = .3; spec.durationSeconds = .7;
    request.notes = { spec }; request.resamplerExecutable = native.tracks[0].outputResampler; request.requireExternalResampler = true;
    check("missing_explicit_resampler_does_not_silently_fallback", backend::UtauRenderer::render(request).buffer.getNumSamples() == 0);
    const auto bundled = AudioEngine::bundledUtauResampler(juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory());
    native.tracks[0].outputResampler = bundled;
    native.tracks[0].outputWavtool = wavtool;
    native.tracks[0].clips[0].notes[0].utauFlags = "V2";
    engine.syncProject(native); wait();
    check("explicit_resampler_overrides_global_native_backend_and_uses_selected_wavtool", engine.hasCurrentRenderedAudio()
        && engine.activeRenderBackends().contains("utau-resampler+external-wavtool") && engine.activeRenderWarnings().isEmpty());
    request.resamplerExecutable = juce::File{}; request.requireExternalResampler = false;
    const auto internal = backend::UtauRenderer::render(request);
    request.wavtoolExecutable = wavtool;
    const auto assembled = backend::UtauRenderer::render(request);
    std::cout << "wavtool_warning=" << assembled.warning << "|backend=" << assembled.backend
        << "|samples=" << assembled.buffer.getNumSamples() << "|rate=" << assembled.sampleRate << '\n';
    check("selected_wavtool_actually_runs_and_produces_audio", assembled.warning.isEmpty() && assembled.backend.contains("external-wavtool")
        && assembled.buffer.getNumSamples() > 0 && assembled.buffer.getMagnitude(0, assembled.buffer.getNumSamples()) > .001);
    check("selected_wavtool_preserves_timeline_duration", std::abs(internal.buffer.getNumSamples() / internal.sampleRate
        - assembled.buffer.getNumSamples() / assembled.sampleRate) < .005);
    const auto benchmarkBank=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_REAL_BANK",{});
    if(benchmarkBank.isNotEmpty()) {
        auto timingTrack=track; timingTrack.voicebankDirectory=juce::File(benchmarkBank); timingTrack.clips.clear();
        ProjectData timingProject;timingProject.tracks={timingTrack};AudioEngine timingEngine;
        const auto begin=juce::Time::getMillisecondCounterHiRes();
        for(int j=0;j<10;++j) timingEngine.syncProject(timingProject);
        auto* timing=new juce::DynamicObject();timing->setProperty("bank",benchmarkBank);
        timing->setProperty("average_sync_ms",(juce::Time::getMillisecondCounterHiRes()-begin)/10);
        folder.getChildFile("bank-metadata-timing.json").replaceWithText(juce::JSON::toString(juce::var(timing),true));
    }
    auto* report = new juce::DynamicObject(); report->setProperty("ok", ok); report->setProperty("checks", checks);
    folder.getChildFile("output-engine-validation.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
    return ok;
}
}
