#pragma once

namespace hachi
{
inline bool MainComponent::diagnosticVoicebankSelection(const juce::File& output,
                                                       const juce::File& bank)
{
    if (!bank.isDirectory() || backend::DiffSingerRenderer::isVoicebank(bank)) return false;
    output.createDirectory();
    bool ok = true;
    juce::StringArray report;
    const auto check = [&](const char* name, bool pass)
    {
        report.add(juce::String(name) + "=" + (pass ? "1" : "0"));
        ok &= pass;
    };
    const auto selected = project.addTrack("CVVC selection", true);
    const auto other = project.addTrack("Unchanged track", true);
    project.setTrackPitchAlgorithm(selected, PitchAlgorithm::utau);
    project.dispatchPendingMessages();
    diagnosticSelectTrack(selected);
    showAssetManager();
    const auto state = [&](const juce::String& id)
    {
        for (const auto& track : project.snapshot().tracks) if (track.id == id) return track;
        return TrackData{};
    };
    const auto original = state(selected).voicebankDirectory;
    const auto otherOriginal = state(other).voicebankDirectory;
    const auto defaultBefore = preferences->getValue("algorithm.utauVoicebank");
    const auto revisionBefore = project.revisionNumber();
    const auto waitsBefore = backend::UtauRenderer::diagnosticMessageThreadWaits();
    const auto buildsBefore = backend::UtauRenderer::diagnosticIndexBuilds();
    const auto started = juce::Time::getMillisecondCounterHiRes();
    check("selection_accepted", selectVoicebank(selected, bank));
    const auto selectionMs = juce::Time::getMillisecondCounterHiRes() - started;
    project.dispatchPendingMessages();
    const auto refreshMs = juce::Time::getMillisecondCounterHiRes() - started;
    report.add("selection_ms=" + juce::String(selectionMs, 3));
    report.add("selection_with_ui_refresh_ms=" + juce::String(refreshMs, 3));
    check("selection_responsive", selectionMs < 1500.0);
    check("ui_refresh_responsive", refreshMs < 1500.0);
    check("bank_bound", state(selected).voicebankDirectory == bank);
    check("other_track_unchanged", state(other).voicebankDirectory == otherOriginal);
    check("algorithm_preserved", state(selected).pitchAlgorithm == PitchAlgorithm::utau);
    check("global_default_unchanged", preferences->getValue("algorithm.utauVoicebank") == defaultBefore);
    check("one_model_edit", project.revisionNumber() == revisionBefore + 1);
    check("no_index_wait_on_message_thread", backend::UtauRenderer::diagnosticMessageThreadWaits() == waitsBefore);
    check("empty_track_needs_no_index", backend::UtauRenderer::diagnosticIndexBuilds() == buildsBefore);
    check("undo_bank_selection", project.undo() && state(selected).voicebankDirectory == original);
    check("redo_bank_selection", project.redo() && state(selected).voicebankDirectory == bank);
    const auto revisionAfter = project.revisionNumber();
    check("same_bank_no_edit", selectVoicebank(selected, bank) && project.revisionNumber() == revisionAfter);
    check("cancel_no_edit", !selectVoicebank(selected, {}) && project.revisionNumber() == revisionAfter);
    check("removed_track_no_edit", !selectVoicebank("missing-track", bank)
        && project.revisionNumber() == revisionAfter);
    check("other_track_stays_unbound", state(other).voicebankDirectory == otherOriginal);
    const auto text = report.joinIntoString("\n") + "\n";
    std::cout << text << std::flush;
    return output.getChildFile("selection.txt").replaceWithText(text) && ok;
}
}
