#pragma once
#include <iostream>

namespace hachi
{
bool MainComponent::diagnosticClipLength(const juce::File& output)
{
    if (output.createDirectory().failed()) return false;
    stopTimer(); audio.stop(); pianoRoll.setReadsVoicebankInBackground(false);
    bool ok = true; juce::String log;
    const auto check = [&](const juce::String& name, bool pass)
    { ok &= pass; log += name + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << pass << std::endl; };
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-8; };
    for (int mode = 0; mode < 4; ++mode)
    {
        ProjectData fixture; TrackData track; track.id = "length-track";
        track.trackType = mode == 1 ? TrackType::utau : mode == 2 ? TrackType::diffSinger : TrackType::hachi;
        track.compose = mode != 3; track.accompaniment = mode == 3;
        track.pitchAlgorithm = mode == 1 || mode == 2 ? PitchAlgorithm::utau : PitchAlgorithm::nsfHifigan;
        ClipData clip; clip.id = "length-clip"; clip.startSeconds = 3; clip.durationSeconds = 4;
        NoteData note; note.id = "length-note"; note.startSeconds = 1; note.durationSeconds = 2;
        note.midiNote = 60; note.pitchControlPoints = {{0,60}, {2,62}};
        clip.notes = {note}; clip.fadeInSeconds = .2; clip.fadeOutSeconds = .4;
        track.clips = {clip}; fixture.tracks = {track};
        project.resetDocument(fixture); project.dispatchPendingMessages();
        const auto first = [&] { return project.snapshot().tracks[0].clips[0]; };
        const auto tag = juce::String(mode) + "_";
        const auto revision = project.revisionNumber();
        auto menu = clipContextMenu(clip.id, 4);
        bool entry = false;
        bool outerOutput = false;
        for (juce::PopupMenu::MenuItemIterator i(menu); i.next();)
        {
            if (i.getItem().itemID == 7) entry = i.getItem().isEnabled && i.getItem().text == strings.text("clip.editLength");
            if (i.getItem().text == strings.text("output.choose")) outerOutput = i.getItem().subMenu != nullptr;
        }
        check(tag + "menu_available", entry);
        check(tag + "outer_output_menu_preserved", outerOutput);
        {
            auto dialog = makeClipLengthDialog(clip.id);
            check(tag + "dialog_current_seconds", dialog && near(dialog->getTextEditorContents("length").getDoubleValue(), 4));
            if (dialog)
            {
                auto* outputChoice = dialog->getComboBoxComponent("output");
                check(tag + "output_options_match_existing_menu", outputChoice && outputChoice->getNumItems() == 3
                    && outputChoice->getSelectedId() == 1 && outputChoice->isEnabled() == (mode == 1));
                if (outputChoice) outputChoice->setSelectedId(3, juce::sendNotificationSync);
                auto* editor = dialog->getTextEditor("length");
                editor->setText("0", false); editor->onTextChange();
                check(tag + "invalid_disables_apply", !dialog->getButton(0)->isEnabled());
                editor->setText("8.000000", false); editor->onTextChange();
                check(tag + "valid_enables_apply", dialog->getButton(0)->isEnabled());
                if (mode == 1)
                {
                    dialog->setSize(600, 360);
                    const auto image = dialog->createComponentSnapshot(dialog->getLocalBounds());
                    auto stream = output.getChildFile("clip-length-dialog.png").createOutputStream();
                    check("dialog_capture", stream && juce::PNGImageFormat().writeImageToStream(image, *stream));
                }
            }
        }
        check(tag + "open_cancel_no_edit", project.revisionNumber() == revision && near(first().durationSeconds, 4));
        for (const auto* invalid : {"", "0", "-1", "0.001", "nan", "inf", "1.2.3", "8oops"})
            check(tag + "reject_" + invalid, !applyClipLengthText(clip.id, invalid) && project.revisionNumber() == revision);
        check(tag + "extend_exactly", applyClipLengthText(clip.id, "8") && near(first().durationSeconds, 8) && near(first().startSeconds, 3));
        check(tag + "notes_pitch_fades_unchanged", near(first().notes[0].startSeconds, 1)
            && near(first().notes[0].durationSeconds, 2) && near(first().notes[0].pitchControlPoints.back().timeSeconds, 2)
            && near(first().notes[0].pitchControlPoints.back().targetMidi, 62) && near(first().fadeOutSeconds, .4));
        check(tag + "single_undo", project.revisionNumber() == revision + 1 && project.undo() && near(first().durationSeconds, 4));
        check(tag + "redo", project.redo() && near(first().durationSeconds, 8));
        const auto unchanged = project.revisionNumber();
        check(tag + "unchanged_no_undo", applyClipLengthText(clip.id, "8.000000") && project.revisionNumber() == unchanged);
        check(tag + "shorten_exactly", applyClipLengthText(clip.id, "2") && near(first().durationSeconds, 2)
            && near(first().notes[0].durationSeconds, 2) && near(first().notes[0].startSeconds, 1) && near(first().startSeconds, 3));
        juce::String error; ProjectModel loaded;
        const auto file = output.getChildFile(tag + "length.hjpx");
        check(tag + "save_reopen", project.save(file, error) && loaded.load(file, error)
            && near(loaded.snapshot().tracks[0].clips[0].durationSeconds, 2)
            && near(loaded.snapshot().tracks[0].clips[0].notes[0].startSeconds, 1)
            && near(loaded.snapshot().tracks[0].clips[0].notes[0].durationSeconds, 2));
        check(tag + "reextend_keeps_hidden_note_tail", loaded.setClipEditableLength(clip.id, 8)
            && near(loaded.snapshot().tracks[0].clips[0].notes[0].durationSeconds, 2));
        check(tag + "hide_whole_note_without_deleting", applyClipLengthText(clip.id, "0.5")
            && first().notes.size() == 1 && near(first().notes[0].startSeconds, 1)
            && applyClipLengthText(clip.id, "4") && near(first().notes[0].durationSeconds, 2));
        check(tag + "stale_id_rejected", !applyClipLengthText("deleted", "2") && !makeClipLengthDialog("deleted"));
        if (mode == 1)
        {
            project.resetDocument(fixture);
            auto edit = makeClipLengthDialog(clip.id);
            auto* draft = dynamic_cast<ClipEditDialog*>(edit.get());
            check("edit_dialog_has_shared_tool_configuration", draft != nullptr);
            if (draft)
            {
                const auto executable = output.getChildFile("fixture-resampler.exe");
                check("tool_path_fixture", executable.replaceWithText("UI fixture only; never executed"));
                draft->getComboBoxComponent("output")->setSelectedId(3, juce::sendNotificationSync);
                check("resampler_configuration_enabled", draft->configure.isEnabled());
                draft->resampler = executable;
                draft->getTextEditor("length")->setText("bad", false);
                const auto before = project.contentFingerprint();
                check("invalid_length_applies_neither_draft", !applyClipEditDialog(clip.id, *draft, error)
                    && project.contentFingerprint() == before);
                draft->getTextEditor("length")->setText("6", false);
                draft->wavtool = output.getChildFile("missing-tool.exe");
                check("invalid_tool_applies_neither_draft", !applyClipEditDialog(clip.id, *draft, error)
                    && project.contentFingerprint() == before);
                draft->wavtool = {};
                check("apply_length_and_resampler", applyClipEditDialog(clip.id, *draft, error)
                    && near(first().durationSeconds, 6) && near(first().notes[0].startSeconds, 1)
                    && near(first().notes[0].durationSeconds, 2)
                    && project.snapshot().tracks[0].outputEngine == UtauOutputEngine::resampler
                    && project.snapshot().tracks[0].outputResampler == executable);
                auto reflected = makeClipLengthDialog(clip.id);
                check("editor_reads_back_selected_engine", reflected && reflected->getComboBoxComponent("output")->getSelectedId() == 3);
                auto outside = clipContextMenu(clip.id, 4); bool resamplerTicked = false;
                for (juce::PopupMenu::MenuItemIterator i(outside, true); i.next();)
                    if (i.getItem().itemID == 103) resamplerTicked = i.getItem().isTicked;
                check("outer_menu_reflects_editor_change", resamplerTicked);
                const auto saved = output.getChildFile("edit-engine.hjpx"); ProjectModel reopened;
                check("editor_engine_persists_in_project", project.save(saved, error) && reopened.load(saved, error)
                    && reopened.snapshot().tracks[0].outputEngine == UtauOutputEngine::resampler
                    && reopened.snapshot().tracks[0].outputResampler == executable);
                reflected->getComboBoxComponent("output")->setSelectedId(2, juce::dontSendNotification);
                const bool modelReady = backend::NsfHifiganRenderer::modelAvailable(preferences != nullptr
                    ? juce::File(preferences->getValue("algorithm.hifiganPath")) : juce::File{});
                const auto beforeHifi = project.contentFingerprint();
                check("editor_hifisampler_uses_existing_model_guard", applyClipEditDialog(clip.id, *reflected, error) == modelReady
                    && (modelReady ? project.snapshot().tracks[0].outputEngine == UtauOutputEngine::pcNsfHifigan
                                   : project.contentFingerprint() == beforeHifi));
                check("outside_engine_choice_still_works", outputEngineItemChosen(101, track.id)
                    && project.snapshot().tracks[0].outputEngine == UtauOutputEngine::inherit);
                reflected = makeClipLengthDialog(clip.id);
                check("editor_reflects_outside_choice", reflected && reflected->getComboBoxComponent("output")->getSelectedId() == 1);
                check("deleted_clip_rejects_editor_apply", !applyClipEditDialog("deleted", *draft, error));
            }
        }
        if (mode == 0)
        {
            auto collision = fixture;
            collision.tracks[0].allowNativeAudioOverlap = false;
            collision.tracks[0].clips[0].sourceFile = output.getChildFile("source.wav");
            auto next = collision.tracks[0].clips[0]; next.id = "neighbor"; next.startSeconds = 8;
            collision.tracks[0].clips.push_back(next);
            project.resetDocument(collision);
            check("extend_edit_area_without_extending_audio", applyClipLengthText(clip.id, "8")
                && near(first().durationSeconds, 8) && near(first().audioLength(), 4)
                && near(project.snapshot().tracks[0].clips[1].startSeconds, 8));
            check("shorten_keeps_full_audio_window", applyClipLengthText(clip.id, "0.5")
                && near(first().audioLength(), 4) && near(first().notes[0].durationSeconds, 2));
            const auto retained = output.getChildFile("retained-audio.hjpx");
            check("save_retained_audio", project.save(retained, error));
            ProjectModel retainedModel;
            check("reload_retains_audio_window", retainedModel.load(retained, error)
                && near(retainedModel.snapshot().tracks[0].clips[0].audioLength(), 4));
            check("reextend_restores_audio_window", retainedModel.setClipEditableLength(clip.id, 4)
                && near(retainedModel.snapshot().tracks[0].clips[0].audioLength(), 4));
        }
    }
    return output.getChildFile("clip-length.txt").replaceWithText(log) && ok;
}
}
