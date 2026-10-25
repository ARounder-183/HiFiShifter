#pragma once
#include "../AssetManagerComponent.h"
#include "../AudioEngine.h"
#include <iostream>

namespace hachi
{
inline bool assetContextSmoke(const juce::File& folder)
{
    folder.createDirectory();
    bool ok = true; int checks = 0;
    const auto check = [&](const char* name, bool pass)
    { ++checks; ok &= pass; std::cout << name << "=" << pass << std::endl; };
    const auto bank = folder.getChildFile("bank"); bank.createDirectory();
    const auto file = bank.getChildFile("tone.wav"), other = bank.getChildFile("other.wav");
    {
        juce::AudioBuffer<float> buffer(1, 4410); buffer.clear();
        juce::WavAudioFormat wav;
        auto out = file.createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(out.release(), 44100, 1, 16, {}, 0));
        check("fixture_written", writer && writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples()));
    }
    file.copyFileTo(other);
    bank.getChildFile("oto.ini").replaceWithText("tone.wav=a,0,0,-90,0,0\nother.wav=i,0,0,-90,0,0\n");
    const auto hash = juce::SHA256(file).toHexString();
    const auto otoHash = juce::SHA256(bank.getChildFile("oto.ini")).toHexString();
    juce::PropertiesFile::Options options; options.storageFormat = juce::PropertiesFile::storeAsXML;
    const auto settings = folder.getChildFile("assets.settings");
    I18n strings;
    ProjectData project; TrackData track; track.id = "bank-track"; track.voicebankDirectory = bank;
    track.compose = true; track.pitchAlgorithm = PitchAlgorithm::utau; track.trackType = TrackType::utau;
    project.tracks.push_back(track);
    {
        juce::PropertiesFile props(settings, options);
        AssetManagerComponent manager(strings, props);
        juce::Component parent; parent.setBounds(100, 60, 1300, 800);
        parent.addAndMakeVisible(manager); manager.setBounds(900, 80, 300, 600);
        for (const auto scale : {1.0f, 1.5f, 2.0f})
        {
            parent.setTransform(juce::AffineTransform::scale(scale));
            for (const auto point : {juce::Point<int>{1700, 220}, juce::Point<int>{-600, 400}})
            {
                const auto popup = manager.contextMenuOptions(point);
                check("popup_at_exact_right_click", popup.getTargetScreenArea() == juce::Rectangle<int>(point.x, point.y, 1, 1));
                check("popup_retains_dpi_target", popup.getTargetComponent() == &parent);
            }
        }
        manager.setProjectVoicebanks(project);
        manager.diagnosticSearchEditor().setText(".wav", false);
        manager.diagnosticSearchEditor().onTextChange();
        check("selected_root_already_loaded", !manager.diagnosticSearchPending());
        check("recordings_visible", manager.diagnosticSearchLabels().size() == 2);
        manager.removeFromBrowser(file, false);
        check("one_recording_removed", manager.diagnosticSearchLabels().size() == 1 && manager.diagnosticSearchLabels()[0].contains("other.wav"));
        manager.setProjectVoicebanks(project);
        check("resync_keeps_recording_removed", manager.diagnosticSearchLabels().size() == 1);
        manager.removeFromBrowser(bank, true);
        check("project_bank_removed", manager.diagnosticTreeRowCount() == 0);
        manager.setProjectVoicebanks(project);
        check("resync_does_not_readd_bank", manager.diagnosticTreeRowCount() == 0);
        check("project_binding_unchanged", project.tracks[0].voicebankDirectory == bank);
        check("audio_and_oto_unchanged", juce::SHA256(file).toHexString() == hash && juce::SHA256(bank.getChildFile("oto.ini")).toHexString() == otoHash);
    }
    {
        juce::PropertiesFile props(settings, options);
        AssetManagerComponent manager(strings, props);
        manager.setProjectVoicebanks(project);
        check("removed_bank_stays_removed_after_reopen", manager.diagnosticTreeRowCount() == 0);
        check("explicit_reimport_succeeds", manager.diagnosticRegisterFolder(bank) >= 0);
        manager.diagnosticSearchEditor().setText(".wav", false);
        manager.diagnosticSearchEditor().onTextChange();
        check("explicit_reimport_restores_recordings", manager.diagnosticSearchLabels().size() == 2);
        manager.removeFromBrowser(other, false);
    }
    {
        juce::PropertiesFile props(settings, options);
        AssetManagerComponent manager(strings, props);
        manager.diagnosticSearchEditor().setText(".wav", false);
        manager.diagnosticSearchEditor().onTextChange();
        check("reopened_root_already_loaded", !manager.diagnosticSearchPending());
        check("removed_recording_stays_removed_after_reopen", manager.diagnosticSearchLabels().size() == 1 && manager.diagnosticSearchLabels()[0].contains("tone.wav"));
        manager.removeFromBrowser(bank, true);
        check("registered_folder_removed", manager.diagnosticTreeRowCount() == 0 && props.getValue("assets.materialFolders").isEmpty());
        check("disk_folder_preserved", bank.isDirectory() && file.existsAsFile() && other.existsAsFile());
    }
    folder.getChildFile("result.json").replaceWithText("{\"checks\":" + juce::String(checks) + ",\"ok\":" + (ok ? "true" : "false") + "}");
    return ok;
}

inline bool pianoProjectInspection(const juce::File& file, const juce::File& folder)
{
    folder.createDirectory();
    const auto originalHash = juce::SHA256(file).toHexString();
    ProjectModel model; juce::String error;
    if (!model.load(file, error)) { std::cout << error << std::endl; return false; }
    auto data = model.snapshot();
    const auto measure = [&](const ProjectData& project)
    {
        AudioEngine audio(false); audio.prepareToPlay(512, 44100);
        // Only request the initial note of each track; retain the real project
        // routing, note times, volume, solo, voicebank and engine settings.
        std::vector<juce::String> selected;
        for (const auto& track : project.tracks)
            for (const auto& clip : track.clips)
                if (!clip.notes.empty()) selected.push_back(clip.notes.front().id);
        audio.setUtauRenderNoteSelection(selected); audio.syncProject(project);
        const auto deadline = juce::Time::getMillisecondCounter() + 60000;
        while (audio.renderProgress() && juce::Time::getMillisecondCounter() < deadline) juce::Thread::sleep(10);
        audio.setPosition(16.8); audio.play();
        juce::AudioBuffer<float> samples(2, 22050); samples.clear();
        for (int p = 0; p < samples.getNumSamples(); p += 512)
        {
            juce::AudioSourceChannelInfo block(&samples, p, std::min(512, samples.getNumSamples() - p));
            audio.getNextAudioBlock(block);
        }
        audio.stop();
        std::cout << "warnings=" << audio.activeRenderWarnings() << std::endl;
        return samples.getRMSLevel(0, 0, samples.getNumSamples());
    };
    const auto originalRms = measure(data);
    for (auto& track : data.tracks) track.solo = false;
    const auto withoutSoloRms = measure(data);
    const auto untouched = originalHash == juce::SHA256(file).toHexString();
    std::cout << "saved_project_rms=" << originalRms << "|without_solo_rms=" << withoutSoloRms << "|original_untouched=" << untouched << std::endl;
    return originalRms == 0 && withoutSoloRms > .001f && untouched;
}
}
