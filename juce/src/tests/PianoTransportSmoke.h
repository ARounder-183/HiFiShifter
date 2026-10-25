#pragma once

namespace hachi
{
inline bool MainComponent::diagnosticPianoTransport(const juce::File& folder, const juce::File& sourceProject)
{
    folder.createDirectory(); stopTimer(); setSize(1280, 850);
    bool ok = true; int checks = 0;
    const auto check = [&](const juce::String& name, bool pass)
    { ++checks; ok &= pass; std::cout << name << '=' << pass << std::endl; };
    juce::String error;
    check("output_device_available", audio.ensureOutputDevice(error));
    if (!ok) { std::cout << error << std::endl; return false; }
    // Keep the real transport/device checks, but pull PCM ourselves so driver
    // callbacks cannot race the measurement or make this test play out loud.
    audio.devices().getCurrentAudioDevice()->stop();
    audio.prepareToPlay(512, 44100);
    const auto wait = [&]
    {
        const auto deadline = juce::Time::getMillisecondCounterHiRes() + 60000;
        while (audio.renderProgress() && juce::Time::getMillisecondCounterHiRes() < deadline)
            juce::Thread::sleep(10);
        timerCallback(); stopTimer();
        check("render_completed", !audio.renderProgress());
    };
    const auto read = [&](double at)
    {
        // Do not call audio.play(): the production button/timer must start it.
        audio.setPosition(at);
        juce::AudioBuffer<float> samples(2, 11025); samples.clear();
        for (int offset = 0; offset < samples.getNumSamples(); offset += 512)
        {
            juce::AudioSourceChannelInfo block(&samples, offset, std::min(512, samples.getNumSamples()-offset));
            audio.getNextAudioBlock(block);
        }
        return samples.getRMSLevel(0, 0, samples.getNumSamples());
    };
    const auto load = [&](const ProjectData& data, const juce::String& clip)
    {
        audio.stop(); playWhenRenderReady = false;
        project.resetDocument(data); project.dispatchPendingMessages(); stopTimer();
        focusClip(clip); pianoRoll.setSelectedNoteIds({}); updateUtauRenderSelection();
        syncAudio(project.snapshot());
    };
    const auto bank = folder.getChildFile("bank"); bank.createDirectory();
    {
        juce::AudioBuffer<float> samples(1, 44100); samples.clear();
        auto stream = bank.getChildFile("a.wav").createOutputStream();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),44100,1,16,{},0));
        check("bank_fixture", writer && writer->writeFromAudioSampleBuffer(samples,0,samples.getNumSamples()));
    }
    // Two aliases avoid the intentional single-sample instrument mapping,
    // which accepts every lyric and therefore cannot test a missing alias.
    bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,100,-800,50,20\na.wav=i,0,100,-800,50,20\n");
    int caseId = 0;
    for (const auto mode : {UtauMode::classic, UtauMode::jie, UtauMode::mou})
        for (const bool loaded : {false, true})
        {
            const auto prefix = juce::String(++caseId) + "-";
            ProjectData data; TrackData track;
            track.id = prefix+"utau"; track.trackType=TrackType::utau;
            track.pitchAlgorithm=PitchAlgorithm::utau; track.utauMode=mode; track.compose=true;
            track.outputEngine=UtauOutputEngine::resampler;
            if (loaded) track.voicebankDirectory=bank;
            ClipData clip; clip.id=prefix+"clip"; clip.durationSeconds=clip.sourceDurationSeconds=4;
            for (int i=0;i<4;++i)
            {
                NoteData note; note.id=prefix+juce::String(i); note.startSeconds=.2+i;
                note.durationSeconds=.6; note.midiNote=60;
                note.label=i==0?" \t ":i==1?"RR":i==2?"missing-sample":"a";
                clip.notes.push_back(note);
            }
            track.clips={clip}; data.tracks={track};
            load(data,clip.id);
            check(prefix+"fresh_no_history_audio", !audio.hasCurrentRenderedAudio());
            const auto fingerprint=project.contentFingerprint();
            audio.setPosition(.25); playButton.onClick(); wait();
            check(prefix+"first_play_started",audio.isPlaying()&&!playWhenRenderReady);
            check(prefix+"first_play_blank_piano",read(.3)>.003f);
            check(prefix+"rest_stays_silent",read(1.3)==0);
            check(prefix+"unselected_voiced_notes_not_synthesised",read(2.3)==0&&read(3.3)==0);
            check(prefix+"play_does_not_edit_or_select",fingerprint==project.contentFingerprint()&&pianoRoll.selectedNoteIds().empty());
            audio.stop();
            pianoRoll.setSelectedNoteIds({prefix+"2"}); updateUtauRenderSelection();
            playButton.onClick(); wait();
            check(prefix+"selected_missing_alias_piano",read(2.3)>.003f);
            check(prefix+"selection_excludes_other_blank",read(.3)==0);
            audio.stop();
            pianoRoll.setSelectedNoteIds({}); updateUtauRenderSelection(); audio.setPosition(.25);
            playButton.onClick(); wait();
            check(prefix+"history_plus_blank",read(.3)>.003f&&read(2.3)>.003f);
            audio.stop();
            check(prefix+"toggle_visible_default_on",pianoPreviewButton.isVisible()&&pianoPreviewButton.getToggleState());
            if(mode==UtauMode::classic)
                check(prefix+"toggle_right_of_phonemizer",pianoPreviewButton.getX()>=phonemizerButton.getRight()
                    &&pianoPreviewButton.getY()==phonemizerButton.getY());
            pianoPreviewButton.setToggleState(false,juce::dontSendNotification); pianoPreviewButton.onClick();
            project.dispatchPendingMessages(); stopTimer();
            check(prefix+"toggle_updates_model",!project.snapshot().tracks[0].utauPianoPreview);
            audio.setPosition(.25); playButton.onClick(); wait();
            check(prefix+"disabled_blank_and_missing_silent",read(.3)==0&&read(2.3)==0);
            audio.stop();
            pianoPreviewButton.setToggleState(true,juce::dontSendNotification); pianoPreviewButton.onClick();
            project.dispatchPendingMessages(); stopTimer();
            audio.setPosition(.25); playButton.onClick(); wait();
            check(prefix+"reenable_invalidates_silent_cache",read(.3)>.003f&&read(2.3)>.003f);
            audio.stop();
        }
    auto independent=project.snapshot();
    auto second=independent.tracks[0];second.id="independent";
    second.clips[0].id="independent-clip";
    for(auto& note:second.clips[0].notes)note.id="independent-"+note.id;
    independent.tracks.push_back(second);
    load(independent,independent.tracks[0].clips[0].id);
    pianoPreviewButton.setToggleState(false,juce::dontSendNotification);pianoPreviewButton.onClick();
    project.dispatchPendingMessages();stopTimer();
    check("toggle_is_per_track",!project.snapshot().tracks[0].utauPianoPreview&&project.snapshot().tracks[1].utauPianoPreview);
    project.undo();project.dispatchPendingMessages();stopTimer();
    check("toggle_undo",project.snapshot().tracks[0].utauPianoPreview&&pianoPreviewButton.getToggleState());
    project.redo();project.dispatchPendingMessages();stopTimer();
    check("toggle_redo",!project.snapshot().tracks[0].utauPianoPreview&&!pianoPreviewButton.getToggleState());
    const auto saved=folder.getChildFile("piano-settings.hjpx");
    check("toggle_saved",project.save(saved,error));
    ProjectModel reopened;check("toggle_reopened",reopened.load(saved,error)
        &&!reopened.snapshot().tracks[0].utauPianoPreview&&reopened.snapshot().tracks[1].utauPianoPreview);
    focusClip("independent-clip");refreshProjectControls();
    check("switch_track_reads_own_toggle",pianoPreviewButton.getToggleState());
    audio.setPosition(.25);playButton.onClick();wait();
    check("other_track_still_has_piano",read(.3)>.003f);audio.stop();
    project.setTrackPianoPreview("independent",false);project.dispatchPendingMessages();stopTimer();
    audio.setPosition(.25);playButton.onClick();wait();
    check("both_tracks_off_silent",read(.3)==0);audio.stop();
    project.setTrackPianoPreview(independent.tracks[0].id,true);project.dispatchPendingMessages();stopTimer();
    // A failed, explicitly DiffSinger region must stay failed, while another
    // track's piano remains playable through the same button and timer.
    auto mixed=project.snapshot();
    auto ds=mixed.tracks.front(); ds.id="ds"; ds.trackType=TrackType::diffSinger;
    ds.voicebankDirectory={}; ds.clips[0].id="ds-clip";
    ds.clips[0].notes.resize(1); ds.clips[0].notes[0].id="ds-note";
    mixed.tracks.push_back(ds); load(mixed,mixed.tracks.front().clips.front().id);
    audio.setUtauRenderNoteSelection({"ds-note"}); syncAudio(mixed); wait();
    check("diffsinger_error_retained",audio.activeRenderWarnings().contains("dsconfig.yaml"));
    audio.setPosition(.25); playButton.onClick(); wait();
    check("other_track_piano_plays_despite_ds_error",audio.isPlaying()&&read(.3)>.003f);
    check("diffsinger_never_becomes_piano",audio.activeRenderWarnings().contains("piano fallback is disabled"));
    audio.stop();
    if (sourceProject.existsAsFile())
    {
        const auto hash=juce::SHA256(sourceProject).toHexString();
        ProjectModel original;
        check("actual_project_loaded",original.load(sourceProject,error));
        auto data=original.snapshot();
        // The saved project may still contain the previous solo setting. Match
        // the user's latest screenshot only in this disposable in-memory copy.
        for(auto& track:data.tracks) track.solo=false;
        load(data,data.tracks.front().clips.front().id);
        const auto at=data.tracks.front().clips.front().startSeconds
            +data.tracks.front().clips.front().notes.front().startSeconds+.1;
        audio.setPosition(at); playButton.onClick(); wait();
        const auto rms=read(at);
        std::cout<<"actual_project_first_play_rms="<<rms<<std::endl;
        check("actual_project_first_play_piano",audio.isPlaying()&&rms>.003f);
        check("actual_project_file_untouched",hash==juce::SHA256(sourceProject).toHexString());
        audio.stop();
    }
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");
    return ok;
}
}
