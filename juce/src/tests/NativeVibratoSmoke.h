#pragma once
#include "../NativePitchIdentity.h"
#include "JoinedVibratoBakeSmoke.h"

namespace hachi
{
template<class Check>
void checkJoinedVibrato(const juce::File& folder, Check check)
{
    ProjectData data;TrackData track;track.id="joined";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;
    for(int i=0;i<4;++i)
    {
        ClipData c;c.id="c"+juce::String(i);c.startSeconds=3+i*.5;c.durationSeconds=c.sourceDurationSeconds=.5;
        NoteData n;n.id="n"+juce::String(i);n.durationSeconds=.5;n.midiNote=n.sourceMidiCenter=i<3?60.f:64.f;
        n.sourcePitchMeasured=true;n.label="a";n.contour={{0,0,0,true},{.5,0,0,true}};c.notes={n};track.clips.push_back(c);
    }
    data.tracks={track};ProjectModel model;model.resetDocument(data);
    NoteData parameters;parameters.vibratoLengthPercent=85;parameters.vibratoCycleMs=213;
    parameters.vibratoDepthCents=50;parameters.vibratoFadeInPercent=30;parameters.vibratoFadeOutPercent=25;
    parameters.vibratoPhasePercent=17;parameters.vibratoEndPercent=92;
    auto reference=parameters;reference.vibratoEnabled=true;reference.durationSeconds=1.5;
    const auto original=model.contentFingerprint();
    model.setNotesVibrato({"n2","n0","n1","n3"},parameters,true,true);
    const auto joined=model.snapshot();bool same=true,requests=true;
    for(int i=0;i<3;++i)
    {
        const auto& c=joined.tracks[0].clips[i];const auto& n=c.notes[0];
        same &= n.vibratoReferenceDurationSeconds==1.5&&n.vibratoTimeOffsetSeconds==i*.5;
        for(int f=0;f<=500;++f)same &= std::abs(vibratoCentsAt(n,f*.001)-vibratoCentsAt(reference,i*.5+f*.001))<1.e-7;
        const auto r=AudioEngine::diagnosticNativeRequest(c,joined.tracks[0]);
        for(size_t f=1;f+1<r.targetMidi.size();++f)
            requests &= std::abs(r.targetMidi[f]-60-vibratoCentsAt(reference,i*.5+f*r.framePeriodMs/1000.)/100)<.0002;
    }
    check("joined_vibrato_matches_one_complete_note_including_fades_and_phase",same);
    check("joined_vibrato_native_requests_follow_one_clock",requests);
    check("different_pitch_starts_independent_vibrato",joined.tracks[0].clips[3].notes[0].vibratoReferenceDurationSeconds==0);
    check("joined_vibrato_one_undo",model.undo()&&model.contentFingerprint()==original);
    check("joined_vibrato_redo",model.redo());
    juce::String error;ProjectModel reopened;
    check("joined_vibrato_save_reopen",model.save(folder.getChildFile("joined.hjpx"),error)&&reopened.load(folder.getChildFile("joined.hjpx"),error)
        &&std::abs(vibratoCentsAt(reopened.snapshot().tracks[0].clips[1].notes[0],.19)-vibratoCentsAt(reference,.69))<1.e-7);
    auto changedParameters=parameters;changedParameters.vibratoDepthCents=91;changedParameters.vibratoPhasePercent=63;
    model.setNotesVibrato({"n1"},changedParameters,true);
    const auto edited=model.snapshot();bool unified=true;
    for(int i=0;i<3;++i)unified &= edited.tracks[0].clips[i].notes[0].vibratoDepthCents==91&&edited.tracks[0].clips[i].notes[0].vibratoPhasePercent==63;
    check("one_member_parameter_edit_updates_entire_group_only",unified&&edited.tracks[0].clips[3].notes[0].vibratoDepthCents==50);
    model.setNotesVibratoRealLine({"n2"},false);const auto display=model.snapshot();
    check("one_member_display_toggle_updates_group",!display.tracks[0].clips[0].notes[0].vibratoRealLine&&!display.tracks[0].clips[1].notes[0].vibratoRealLine);
    model.setNotesVibrato({"n1"},changedParameters,false);const auto off=model.snapshot();
    check("one_member_disable_updates_entire_group_only",!off.tracks[0].clips[0].notes[0].vibratoEnabled&&!off.tracks[0].clips[2].notes[0].vibratoEnabled&&off.tracks[0].clips[3].notes[0].vibratoEnabled);
    auto legacy=joined;legacy.tracks[0].clips[1].notes[0].vibratoDepthCents=12;model.resetDocument(legacy);
    check("legacy_joined_parameters_unified_on_reopen",model.save(folder.getChildFile("legacy-joined.hjpx"),error)&&reopened.load(folder.getChildFile("legacy-joined.hjpx"),error)
        &&reopened.snapshot().tracks[0].clips[1].notes[0].vibratoDepthCents==50);
    model.resetDocument(data);model.setNotesVibrato({"n0","n2"},parameters,true);
    const auto isolated=model.snapshot();check("unselected_middle_note_breaks_vibrato_group",!isolated.tracks[0].clips[1].notes[0].vibratoEnabled
        &&isolated.tracks[0].clips[0].notes[0].vibratoReferenceDurationSeconds==0&&isolated.tracks[0].clips[2].notes[0].vibratoReferenceDurationSeconds==0);
    bool separate=true;
    for(double shift:{-.05,.01})
    {
        auto p=data;p.tracks[0].clips[1].startSeconds+=shift;model.resetDocument(p);model.setNotesVibrato({"n0","n1"},parameters,true);
        const auto d=model.snapshot();separate &= d.tracks[0].clips[0].notes[0].vibratoReferenceDurationSeconds==0&&d.tracks[0].clips[1].notes[0].vibratoReferenceDurationSeconds==0;
    }
    check("gaps_and_overlaps_do_not_join_vibrato",separate);
    auto multi=data;multi.tracks.push_back(track);multi.tracks[1].id="other";
    for(auto& c:multi.tracks[1].clips){c.id+="other";for(auto& n:c.notes)n.id+="other";}
    model.resetDocument(multi);model.setNotesVibrato({"n0","n1other"},parameters,true);
    const auto split=model.snapshot();check("different_tracks_do_not_join_vibrato",split.tracks[0].clips[0].notes[0].vibratoReferenceDurationSeconds==0&&split.tracks[1].clips[1].notes[0].vibratoReferenceDurationSeconds==0);
    bool modes=true;
    for(auto type:{TrackType::utau,TrackType::diffSinger})
    {
        auto p=data;p.tracks[0].trackType=type;p.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;model.resetDocument(p);
        model.setNotesVibrato({"n0","n1","n2"},parameters,true);const auto d=model.snapshot();
        for(int i=0;i<3;++i)for(int f=1;f<500;++f)modes &= std::abs(vibratoCentsAt(d.tracks[0].clips[i].notes[0],f*.001)-vibratoCentsAt(reference,i*.5+f*.001))<1.e-7;
    }
    check("utau_and_ds_use_same_continuous_vibrato_clock",modes);
}

inline void MainComponent::diagnosticNativeVibrato(const juce::File& folder,
                                                  std::function<void(bool)> finished)
{
    struct State
    {
        bool ok = true;
        juce::Array<juce::var> checks;
        ProjectData initial;
        void check(const char* name, bool passed)
        {
            auto* row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("passed", passed);
            checks.add(row); ok &= passed; std::cout << name << '=' << passed << std::endl;
        }
    };
    folder.createDirectory();
    auto state = std::make_shared<State>();
    checkJoinedVibrato(folder,[state](const char* name,bool pass){state->check(name,pass);});
    checkJoinedVibratoBake(folder,[state](const char* name,bool pass){state->check(name,pass);});
    const auto finish = [state, folder, finished]
    {
        auto* report = new juce::DynamicObject(); report->setProperty("passed", state->ok); report->setProperty("checks", state->checks);
        finished(folder.getChildFile("ui-report.json").replaceWithText(juce::JSON::toString(juce::var(report), true)) && state->ok);
    };
    ClipData clip; clip.id = "audio"; clip.durationSeconds = clip.sourceDurationSeconds = 3;
    for (int i = 0; i < 3; ++i)
    {
        NoteData note; note.id = "note-" + juce::String(i); note.label = "audio";
        note.startSeconds = i; note.durationSeconds = 1; note.midiNote = note.sourceMidiCenter = static_cast<float>(i < 2 ? 57 : 59);
        note.sourcePitchMeasured = true; note.utauAutoPitchTransition = false;
        note.contour = {{0, 0, 0, true}, {1, 0, 0, true}};
        note.connectedToPrevious = i > 0; note.connectedToNext = i < 2;
        clip.notes.push_back(note);
    }
    TrackData track; track.id = "native"; track.pitchAlgorithm = PitchAlgorithm::world; track.clips = {clip};
    state->initial.tracks = {track}; project.replace(state->initial); project.dispatchPendingMessages();
    diagnosticSelectTrack(track.id); focusClip(clip.id); pianoRoll.diagnosticRefresh();
    setSize(900, 760); resized();
    state->check("native_button_visible_without_selection_and_disabled", nativeVibratoButton.isVisible() && !nativeVibratoButton.isEnabled());
    state->check("minimum_width_has_room_for_vibrato_and_other_controls", nativeVibratoButton.getWidth() == 64
        && getLocalBounds().contains(nativeVibratoButton.getBounds())
        && !nativeVibratoButton.getBounds().intersects(advancedEnvelopeButton.getBounds())
        && !nativeVibratoButton.getBounds().intersects(nativeAudioOverlapButton.getBounds())
        && !nativeVibratoButton.getBounds().intersects(autoCrossfadeButton.getBounds()));
    state->check("editor_display_options_reachable_in_view_menu", diagnosticHasEditorViewItems());
    pianoRoll.setSelectedNoteIds({"note-0", "note-1"});
    state->check("bulk_native_selection_enables_button", nativeVibratoButton.isEnabled());
    nativeVibratoButton.onClick();
    auto* dialog = dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
    state->check("button_opens_vibrato_dialog", dialog != nullptr && dialog->getComponentID() == "vibrato-editor"
        && dialog->getName() == juce::String::fromUTF8("颤音"));
    if (dialog == nullptr) { finish(); return; }
    dialog->getTextEditor("length")->setText("80");
    dialog->getTextEditor("cycle")->setText("200");
    dialog->getTextEditor("depth")->setText("60");
    juce::ToggleButton* real = nullptr;
    for (auto* child : dialog->getChildren()) if (child->getComponentID() == "vibrato-real-line") real = dynamic_cast<juce::ToggleButton*>(child);
    state->check("native_actual_pitch_preview_defaults_on", real != nullptr && real->getToggleState());
    {
        auto stream = folder.getChildFile("vibrato-dialog.png").createOutputStream(); juce::PNGImageFormat png;
        if (stream) { stream->setPosition(0); stream->truncate(); }
        state->check("dialog_screenshot_written", stream && png.writeImageToStream(dialog->createComponentSnapshot(dialog->getLocalBounds()), *stream));
    }
    // Modal callbacks can run after an ordinary timer during application startup.
    // Wait for the actual dialog completion, then inspect after its edit callback.
    const auto afterClose = [state](juce::AlertWindow* window, int choice, std::function<void()> next)
    {
        juce::ModalComponentManager::getInstance()->attachCallback(window, juce::ModalCallbackFunction::create(
            [state, choice, next](int result)
            {
                state->check("modal_result_matches_action", result == choice);
                juce::Timer::callAfterDelay(20, next);
            }));
        window->exitModalState(choice);
    };
    afterClose(dialog, 1, [this, state, folder, finish, afterClose]
    {
        project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
        auto data = project.snapshot(); const auto& notes = data.tracks[0].clips[0].notes;
        std::cout << "after_apply_enabled=" << notes[0].vibratoEnabled << ',' << notes[1].vibratoEnabled
            << " cycle=" << notes[0].vibratoCycleMs << " depth=" << notes[1].vibratoDepthCents
            << " real=" << notes[0].vibratoRealLine << ',' << notes[1].vibratoRealLine << std::endl;
        state->check("dialog_applies_to_all_selected_only", notes[0].vibratoEnabled && notes[1].vibratoEnabled && !notes[2].vibratoEnabled
            && notes[0].vibratoRealLine && notes[1].vibratoRealLine && notes[1].vibratoDepthCents == 60 && notes[0].vibratoCycleMs == 200);
        state->check("dialog_joins_adjacent_equal_pitch_selection",notes[0].vibratoReferenceDurationSeconds==2
            &&notes[1].vibratoReferenceDurationSeconds==2&&notes[1].vibratoTimeOffsetSeconds==1
            &&std::abs(vibratoCentsAt(notes[0],1)-vibratoCentsAt(notes[1],0))<1.e-7);
        bool unchanged = true;
        for (int i = 0; i < 3; ++i) unchanged &= notes[i].midiNote == state->initial.tracks[0].clips[0].notes[i].midiNote
            && notes[i].pitchControlPoints.empty() && notes[i].contour[0].relativeCents == 0 && notes[i].startSeconds == i && notes[i].durationSeconds == 1;
        state->check("base_pitch_source_reference_and_timing_unchanged", unchanged);
        if (!notes[0].vibratoEnabled || !notes[1].vibratoEnabled) { finish(); return; }
        state->check("bulk_parameters_and_display_have_one_undo", project.undo()
            && !project.snapshot().tracks[0].clips[0].notes[0].vibratoEnabled && !project.snapshot().tracks[0].clips[0].notes[1].vibratoRealLine);
        state->check("bulk_vibrato_redo", project.redo() && project.snapshot().tracks[0].clips[0].notes[1].vibratoEnabled);
        juce::String error; ProjectModel reopened;
        bool saved = project.save(folder.getChildFile("vibrato.hjpx"), error) && reopened.load(folder.getChildFile("vibrato.hjpx"), error);
        if (saved) { const auto copy = reopened.snapshot(); const auto& n = copy.tracks[0].clips[0].notes;
            saved = n[0].vibratoEnabled && n[1].vibratoRealLine && n[1].vibratoCycleMs == 200 && !n[2].vibratoEnabled; }
        state->check("vibrato_save_reopen", saved);
        project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh(); pianoRoll.setTool(PianoRollComponent::Tool::note);
        pianoRoll.setPixelsPerSecond(280); pianoRoll.diagnosticRefresh();
        const auto view = pianoRoll.diagnosticHitBounds(0).getUnion(pianoRoll.diagnosticHitBounds(1)).expanded(32).getSmallestIntegerContainer();
        const auto actual = pianoRoll.createComponentSnapshot(view);
        project.setNotesVibratoRealLine({"note-0", "note-1"}, false); project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
        const auto reference = pianoRoll.createComponentSnapshot(view);
        int changedPixels = 0;
        for (int y = 0; y < actual.getHeight(); ++y) for (int x = 0; x < actual.getWidth(); ++x)
            changedPixels += actual.getPixelAt(x, y) != reference.getPixelAt(x, y);
        state->check("actual_vibrato_changes_drawn_pitch_curve", changedPixels > 30);
        project.undo(); project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
        {
            auto stream = folder.getChildFile("actual-vibrato-line.png").createOutputStream(); juce::PNGImageFormat png;
            if (stream) { stream->setPosition(0); stream->truncate(); }
            state->check("actual_pitch_screenshot_written", stream && png.writeImageToStream(actual, *stream));
        }
        const auto depth = pianoRoll.diagnosticVibratoHandle("note-0", "depth");
        state->check("native_vibrato_handles_available", depth.has_value());
        state->check("joined_vibrato_draws_one_control_set",pianoRoll.diagnosticVibratoControlSetCount()==1);
        bool sameHandles=true;for(const auto* name:{"length","end","fadeIn","fadeOut","depth","cycle","offset"})
            sameHandles &= pianoRoll.diagnosticVibratoHandle("note-0",name)==pianoRoll.diagnosticVibratoHandle("note-1",name);
        state->check("both_members_resolve_same_seven_controls",sameHandles);
        pianoRoll.setSelectedNoteIds({"note-1"});
        state->check("selecting_second_member_keeps_whole_control_set",pianoRoll.diagnosticVibratoControlSetCount()==1);
        if (depth)
        {
            const auto event = [this](juce::Point<float> at, juce::Point<float> from)
            { return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(), at, juce::ModifierKeys::leftButtonModifier,
                1, 0, 0, 0, 0, &pianoRoll, &pianoRoll, juce::Time::getCurrentTime(), from, juce::Time::getCurrentTime(), 1, false); };
            const auto to = *depth + juce::Point<float>(0, -6);
            pianoRoll.mouseDown(event(*depth, *depth)); pianoRoll.mouseDrag(event(to, *depth)); pianoRoll.mouseUp(event(to, *depth));
            state->check("native_depth_handle_edits_entire_joined_vibrato", project.snapshot().tracks[0].clips[0].notes[0].vibratoDepthCents > 60
                && project.snapshot().tracks[0].clips[0].notes[1].vibratoDepthCents == project.snapshot().tracks[0].clips[0].notes[0].vibratoDepthCents);
            project.undo(); project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
            const auto end=pianoRoll.diagnosticVibratoHandle("note-1","end");
            if(end)
            {
                const auto moved=*end+juce::Point<float>(-56,0);
                pianoRoll.mouseDown(event(*end,*end));pianoRoll.mouseDrag(event(moved,*end));pianoRoll.mouseUp(event(moved,*end));
                const auto changed=project.snapshot();const auto& a=changed.tracks[0].clips[0].notes[0];const auto& b=changed.tracks[0].clips[0].notes[1];
                state->check("end_drag_uses_full_duration_and_updates_both",std::abs(a.vibratoEndPercent-90)<.01
                    &&a.vibratoEndPercent==b.vibratoEndPercent&&std::abs(a.vibratoLengthPercent-70)<.01&&a.vibratoLengthPercent==b.vibratoLengthPercent);
                project.undo();project.dispatchPendingMessages();pianoRoll.diagnosticRefresh();
            }
        }
        pianoRoll.setSelectedNoteIds({"note-0", "note-1"}); nativeVibratoButton.onClick();
        auto* disable = dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
        if (!disable) { state->check("disable_dialog_opened", false); finish(); return; }
        afterClose(disable, 2, [this, state, folder, finish, afterClose]
        {
            project.dispatchPendingMessages(); pianoRoll.diagnosticRefresh();
            const auto off = project.snapshot(); const auto& n = off.tracks[0].clips[0].notes;
            state->check("disable_vibrato_preserves_base_and_parameters", !n[0].vibratoEnabled && !n[1].vibratoEnabled
                && n[0].midiNote == 57 && n[1].vibratoDepthCents == 60 && n[0].contour[0].relativeCents == 0);
            const auto revision = project.revisionNumber(); nativeVibratoButton.onClick();
            auto* cancel = dynamic_cast<juce::AlertWindow*>(juce::Component::getCurrentlyModalComponent());
            if (!cancel) { state->check("cancel_dialog_opened", false); finish(); return; }
            afterClose(cancel, 0, [this, state, folder, finish, revision]
            {
                state->check("cancel_makes_no_edit", project.revisionNumber() == revision);
                pianoRoll.clearNoteSelection(); state->check("clearing_selection_disables_button", !nativeVibratoButton.isEnabled());
                project.setTrackPitchAlgorithm("native", PitchAlgorithm::nsfHifigan); project.dispatchPendingMessages(); refreshProjectControls(); resized();
                state->check("nsf_mode_button_and_switches_fit_minimum_width", nativeVibratoButton.isVisible()
                    && nativeVibratoButton.getWidth() == 64 && getLocalBounds().contains(nativeVibratoButton.getBounds())
                    && getLocalBounds().contains(nsfNoiseProtectionButton.getBounds()) && nsfNoiseProtectionButton.getWidth() == 156
                    && !nativeVibratoButton.getBounds().intersects(advancedEnvelopeButton.getBounds())
                    && !nativeVibratoButton.getBounds().intersects(nativeAudioOverlapButton.getBounds())
                    && !nativeVibratoButton.getBounds().intersects(nsfPitchTransitionsButton.getBounds())
                    && !nativeVibratoButton.getBounds().intersects(nsfNoiseProtectionButton.getBounds()));
                state->check("nsf_display_options_reachable_in_view_menu", diagnosticHasEditorViewItems());
                auto stream = folder.getChildFile("native-toolbar.png").createOutputStream(); juce::PNGImageFormat png;
                if (stream) { stream->setPosition(0); stream->truncate(); }
                const auto bounds = advancedEnvelopeButton.getBounds().getUnion(nsfNoiseProtectionButton.getBounds()).expanded(4);
                state->check("toolbar_screenshot_written", stream && png.writeImageToStream(createComponentSnapshot(bounds), *stream));
                project.setTrackPitchAlgorithm("native", PitchAlgorithm::utau); project.dispatchPendingMessages(); refreshProjectControls(); resized();
                state->check("native_shortcut_hidden_in_utau", !nativeVibratoButton.isVisible());
                finish();
            });
        });
    });
}

inline bool runNativeVibratoRenderSmoke(const juce::File& folder, const juce::File& modelDirectory)
{
    using namespace backend;
    folder.createDirectory(); bool ok = true; juce::Array<juce::var> checks;
    const auto check = [&](const char* name, bool passed)
    {
        auto* row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("passed", passed);
        checks.add(row); ok &= passed; std::cout << name << '=' << passed << std::endl;
    };
    constexpr int rate = 44100;
    juce::AudioBuffer<float> samples(1, rate * 2); juce::Random random(139);
    for (int i = 0; i < samples.getNumSamples(); ++i)
    {
        const auto t = double(i) / rate;
        samples.setSample(0, i, t < .2 ? float((random.nextFloat() * 2 - 1) * .12)
            : float(.18 * std::sin(juce::MathConstants<double>::twoPi * 220 * t)
                + .07 * std::sin(juce::MathConstants<double>::twoPi * 440 * t)));
    }
    const auto source = folder.getChildFile("source.wav");
    {
        auto stream = source.createOutputStream(); if (stream) { stream->setPosition(0); stream->truncate(); }
        juce::WavAudioFormat format; std::unique_ptr<juce::AudioFormatWriter> writer(stream
            ? format.createWriterFor(stream.release(), rate, 1, 24, {}, 0) : nullptr);
        check("source_fixture_written", writer && writer->writeFromAudioSampleBuffer(samples, 0, samples.getNumSamples()));
    }
    ClipData clip; clip.id = "audio"; clip.sourceFile = source; clip.durationSeconds = clip.sourceDurationSeconds = 2;
    NoteData note; note.id = "vowel"; note.durationSeconds = 2; note.midiNote = note.sourceMidiCenter = 57;
    note.sourcePitchMeasured = true; note.utauAutoPitchTransition = false;
    for (int i = 0; i <= 400; ++i) note.contour.push_back({i * .005, 0, 0, i >= 40});
    clip.notes = {note}; TrackData track; track.id = "native"; track.pitchAlgorithm = PitchAlgorithm::nsfHifigan;
    track.nativeNsfAudio = true; track.nsfNoiseProtection = true; track.nsfSmoothPitchTransitions = false;
    track.stretchAlgorithm = StretchAlgorithm::hifiShifterMel; track.clips = {clip};
    const auto baseline = AudioEngine::diagnosticNativeRequest(clip, track);
    auto& vibrato = clip.notes[0]; vibrato.vibratoEnabled = true; vibrato.vibratoLengthPercent = 100;
    vibrato.vibratoCycleMs = 200; vibrato.vibratoDepthCents = 80; vibrato.vibratoFadeInPercent = vibrato.vibratoFadeOutPercent = 10;
    track.clips = {clip};
    const auto request = AudioEngine::diagnosticNativeRequest(clip, track);
    const auto phrase = AudioEngine::mergedRequestFor({&clip}, track, {}, {});
    double targetError = 0; bool unvoiced = true;
    for (std::size_t i = 0; i < request.targetMidi.size(); ++i)
    {
        if (i < 40) unvoiced &= request.targetMidi[i] == 0 && phrase.targetMidi[i] == 0;
        else targetError = std::max(targetError, std::abs(double(request.targetMidi[i]) - 57 - vibratoCentsAt(vibrato, i * .005) / 100));
    }
    check("vibrato_dense_target_matches_parameters", targetError < 1.e-5);
    check("single_and_phrase_agree", request.targetMidi == phrase.targetMidi && request.sourceMidi == baseline.sourceMidi);
    check("unvoiced_mask_unchanged", unvoiced);
    check("enabled_vibrato_disables_original_pcm_bypass", canPreserveNativeSource(baseline, rate, rate * 2, rate * 2)
        && !canPreserveNativeSource(request, rate, rate * 2, rate * 2) && !nativePitchIsUnedited(vibrato));
    const auto beforeHash = AudioEngine::nativeClipWaveformHash(track.clips[0], track);
    auto changed = track; changed.clips[0].notes[0].vibratoDepthCents += 10;
    check("vibrato_invalidates_waveform_cache", beforeHash != AudioEngine::nativeClipWaveformHash(changed.clips[0], changed));
    // Linked bodies keep a shared baseline, with a per-body swing added once.
    auto linked = track; linked.clips[0].notes.clear(); linked.clips[0].notes.push_back(vibrato);
    linked.clips[0].notes[0].durationSeconds = 1; linked.clips[0].notes[0].connectedToNext = true;
    auto second = vibrato; second.id = "second"; second.startSeconds = 1; second.durationSeconds = 1;
    second.connectedToPrevious = true; second.vibratoEnabled = false; linked.clips[0].notes.push_back(second);
    for (auto& n : linked.clips[0].notes) { n.contour = {{0,0,0,true},{1,0,0,true}}; n.pitchControlPoints = {{0,57},{1,57}}; }
    const auto linkedRequest = AudioEngine::diagnosticNativeRequest(linked.clips[0], linked);
    bool isolated = true;
    for (std::size_t i = 1; i < linkedRequest.targetMidi.size(); ++i)
    {
        const auto t = i * .005;
        const auto expected = 57 + (t < 1 ? vibratoCentsAt(linked.clips[0].notes[0], t) / 100 : 0);
        isolated &= std::abs(linkedRequest.targetMidi[i] - expected) < 1.e-5;
    }
    check("linked_shared_curve_adds_once_without_leaking_to_neighbor", isolated);
    RenderService service;
    const auto render = [&](Mld5FileRenderRequest req)
    {
        req.hifiganModelDirectory = modelDirectory; req.inference = {InferenceBackend::cpu, -1, 2};
        std::atomic<bool> done{false}; RenderedAudio audio;
        service.renderMld5File(req, [&](auto result) { audio = std::move(result); done.store(true, std::memory_order_release); });
        const auto until = juce::Time::getMillisecondCounterHiRes() + 45000;
        while (!done.load(std::memory_order_acquire) && juce::Time::getMillisecondCounterHiRes() < until) juce::Thread::sleep(2);
        service.cancelAll(); return audio;
    };
    const auto sound = render(request);
    check("real_nsf_vibrato_render_is_audible", sound.buffer.getNumSamples() == rate * 2
        && sound.buffer.getMagnitude(0, sound.buffer.getNumSamples()) > .02f && sound.backend.contains("nsf"));
    const auto estimateMidi = [&](double t)
    {
        if (sound.buffer.getNumSamples() != rate * 2) return 0.0;
        const auto* p = sound.buffer.getReadPointer(0); const int centre = int(t * rate), width = int(.04 * rate);
        const int low = int(rate / 260.0), high = int(rate / 180.0); std::vector<double> score(high + 2);
        int best = low;
        for (int lag = low - 1; lag <= high + 1; ++lag)
        {
            double sum = 0, left = 0, right = 0;
            for (int i = centre - width / 2; i < centre + width / 2; ++i)
            { sum += p[i] * p[i + lag]; left += p[i] * p[i]; right += p[i + lag] * p[i + lag]; }
            score[lag] = sum / std::sqrt(std::max(1.e-20, left * right));
            if (lag >= low && lag <= high && score[lag] > score[best]) best = lag;
        }
        const auto denominator = score[best - 1] - 2 * score[best] + score[best + 1];
        const auto fraction = std::abs(denominator) > 1.e-12 ? .5 * (score[best - 1] - score[best + 1]) / denominator : 0;
        return 69 + 12 * std::log2((rate / (best + fraction)) / 440.0);
    };
    double squared = 0, correlation = 0, measuredEnergy = 0; int count = 0;
    for (double t = .35; t < 1.7; t += .025)
    {
        const auto expected = vibratoCentsAt(vibrato, t) / 100, measured = estimateMidi(t) - 57;
        squared += (measured - expected) * (measured - expected); correlation += expected * measured;
        measuredEnergy += expected * expected; ++count;
    }
    const auto rmsCents = 100 * std::sqrt(squared / count); const auto response = correlation / measuredEnergy;
    std::cout << "rendered_vibrato_rms_cents=" << rmsCents << "\nrendered_vibrato_response=" << response << std::endl;
    check("actual_audio_f0_follows_vibrato", rmsCents < 35 && response > .55 && response < 1.4);
    const auto dry = render(baseline);
    check("turning_off_returns_source_pcm", dry.backend.contains("native-source-preserved") && dry.buffer.getNumSamples() == rate * 2);
    double clearError = 0;
    if (sound.buffer.getNumSamples() == rate * 2 && dry.buffer.getNumSamples() == rate * 2)
        for (int i = int(.04 * rate); i < int(.15 * rate); ++i) clearError = std::max(clearError, double(std::abs(sound.buffer.getSample(0, i) - dry.buffer.getSample(0, i))));
    else clearError = 1;
    check("clear_consonant_pcm_preserved_with_vibrato", clearError < 2.e-5);
    if (sound.buffer.getNumSamples() > 0)
    {
        auto out = folder.getChildFile("vibrato.wav").createOutputStream(); juce::WavAudioFormat wav;
        if (out) { out->setPosition(0); out->truncate(); }
        std::unique_ptr<juce::AudioFormatWriter> writer(out ? wav.createWriterFor(out.release(), sound.sampleRate, 1, 24, {}, 0) : nullptr);
        check("rendered_audio_written", writer && writer->writeFromAudioSampleBuffer(sound.buffer, 0, sound.buffer.getNumSamples()));
    }
    auto* report = new juce::DynamicObject(); report->setProperty("passed", ok); report->setProperty("checks", checks);
    report->setProperty("rendered_vibrato_rms_cents", rmsCents); report->setProperty("rendered_vibrato_response", response);
    report->setProperty("clear_pcm_max_error", clearError);
    return folder.getChildFile("render-report.json").replaceWithText(juce::JSON::toString(juce::var(report), true)) && ok;
}
}
