#pragma once
#include "../MainComponent.h"
#include "../backend/UtauRenderer.h"
#include <iostream>

namespace hachi
{
inline void MainComponent::diagnosticAssetAudio(const juce::File& output, const juce::File& source,
                                               std::function<void(bool)> completed)
{
    output.createDirectory(); stopTimer();
    struct State
    {
        bool ok = true;
        juce::StringArray report;
        void check(const char* name, bool value)
        { ok &= value; report.add(juce::String(name) + "=" + (value ? "1" : "0")); }
    };
    auto state = std::make_shared<State>();
    const auto bank = output.getChildFile("bank"); bank.getChildFile("C4").createDirectory();
    const auto wav = bank.getChildFile("C4/recording.wav");
    state->check("real_recording_copied", source.copyFileTo(wav));
    const auto oto = bank.getChildFile("C4/oto.ini");
    const juce::String otoText("recording.wav=alias,20,100,-200,80,20\n");
    oto.replaceWithText(otoText);
    const auto otoHash = juce::SHA256(oto).toHexString();
    const auto hash = juce::SHA256(wav).toHexString();
    juce::PropertiesFile::Options options;
    options.applicationName = "asset-audio-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    preferences = std::make_unique<juce::PropertiesFile>(output.getChildFile("settings.xml"), options);
    preferences->setValue("algorithm.inference", 2);
    preferences->setValue("assets.otoImportKind", 1);
    project.replace(ProjectData{});
    const auto native = project.addTrack("Native");
    project.setTrackPitchAlgorithm(native, PitchAlgorithm::world);
    selectedTrackId = native;
    audio.setPosition(1.25);
    showAssetManager();
    const auto builds = backend::UtauRenderer::diagnosticIndexBuilds();
    const auto index = assetManager->diagnosticRegisterFolder(bank);
    state->check("bank_available_in_native_mode", index >= 0
        && assetManager->diagnosticMembersOf(index).size() == 1);
    assetManager->setSize(320, 280);
    const auto click = [&](float x, float y)
    {
        const juce::Point<float> position(x, y);
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(), position,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), 0, 0, 0, 0, 0,
            assetManager.get(), assetManager.get(), juce::Time::getCurrentTime(), position,
            juce::Time::getCurrentTime(), 1, false);
    };
    assetManager->mouseDown(click(16, 90));
    assetManager->mouseDown(click(80, 130));
    state->check("nested_recording_selected", assetManager->diagnosticSelectedAudio() == wav);
    state->check("enter_imports_recording", assetManager->keyPressed(juce::KeyPress(juce::KeyPress::returnKey)));
    auto data = project.snapshot();
    state->check("native_track_and_algorithm_preserved", data.tracks.size() == 1
        && data.tracks[0].id == native && data.tracks[0].pitchAlgorithm == PitchAlgorithm::world
        && data.tracks[0].clips.size() == 1);
    const auto firstId = selectedClipId;
    state->check("loaded_at_playhead_and_focused", firstId.isNotEmpty() && selectedTrackId == native
        && !data.tracks[0].clips.empty() && std::abs(data.tracks[0].clips[0].startSeconds - 1.25) < .001);
    const auto utau = project.addTrack("UTAU");
    project.setTrackPitchAlgorithm(utau, PitchAlgorithm::utau);
    selectedTrackId = utau;
    assetManager->onLoadAudio(wav); // The same callback dispatched by the popup.
    data = project.snapshot();
    state->check("utau_selection_creates_native_track", data.tracks.size() == 3
        && data.tracks[1].clips.empty() && trackShowsAllNativeRegions(data.tracks.back())
        && data.tracks.back().clips.size() == 1 && selectedTrackId == data.tracks.back().id);
    state->check("ordinary_analysis_scheduled", pendingNativeAnalyses == 2);
    state->check("load_does_not_build_oto_index", backend::UtauRenderer::diagnosticIndexBuilds() == builds);
    const auto start = juce::Time::getMillisecondCounterHiRes();
    auto poll = std::make_shared<std::function<void()>>();
    const auto weak = std::weak_ptr<std::function<void()>>(poll);
    *poll = [this, state, output, wav, oto, otoHash, hash, start, completed, weak]
    {
        if (pendingNativeAnalyses > 0 && juce::Time::getMillisecondCounterHiRes() - start < 120000)
        {
            if (auto next = weak.lock()) juce::Timer::callAfterDelay(50, [next] { (*next)(); });
            return;
        }
        state->check("analysis_completed", pendingNativeAnalyses == 0);
        const auto current = project.snapshot();
        int clips = 0;
        bool analysed = true;
        for (const auto& track : current.tracks) for (const auto& clip : track.clips)
        {
            ++clips;
            analysed &= clip.sourceFile == wav && !clip.notes.empty()
                && std::any_of(clip.notes.begin(), clip.notes.end(), [](const auto& n)
                    { return !n.nativeUnpitched && !n.contour.empty(); });
        }
        state->check("recordings_have_native_pitch_analysis", clips == 2 && analysed);
        state->check("recording_and_oto_unchanged", juce::SHA256(wav).toHexString() == hash
            && juce::SHA256(oto).toHexString() == otoHash);
        state->check("ordinary_import_keeps_hjm_annotation", SampleSettings::sidecarFor(wav).existsAsFile());
        juce::String error;
        const auto file = output.getChildFile("native-bank.hjpx");
        ProjectModel reopened;
        state->check("save_and_reopen", project.save(file, error) && reopened.load(file, error)
            && reopened.snapshot().tracks.size() == 3
            && reopened.snapshot().tracks[0].clips[0].sourceFile == wav);
        const auto text = state->report.joinIntoString("\n") + "\n";
        std::cout << text << std::flush;
        output.getChildFile("verification.txt").replaceWithText(text);
        completed(state->ok);
    };
    juce::Timer::callAfterDelay(50, [poll] { (*poll)(); });
}
}
