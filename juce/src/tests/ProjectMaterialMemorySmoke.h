#pragma once
#include "../AssetManagerComponent.h"
#include "../ProjectModel.h"
#include <iostream>
namespace hachi
{
inline void runProjectMaterialMemorySmoke(const juce::File& output, std::function<void(bool)> completed)
{
    struct State
    {
        I18n strings; ProjectModel model;
        std::unique_ptr<juce::PropertiesFile> preferences;
        std::unique_ptr<AssetManagerComponent> browser;
        bool ok = true; int checks = 0; juce::String report;
        void check(const char* name, bool pass)
        { ++checks; ok &= pass; report += juce::String(name) + "=" + juce::String(pass ? 1 : 0) + "\n";
          std::cout << name << "=" << pass << std::endl; }
        void show(ProjectData data)
        { model.resetDocument(std::move(data)); browser->setProjectVoicebanks(model.snapshot()); }
    };
    auto s = std::make_shared<State>();
    const auto root = output.getChildFile("fixture-" + juce::Uuid().toString()); root.createDirectory();
    const auto a = root.getChildFile("A bank"), b = root.getChildFile("B bank"),
        manual = root.getChildFile("Manual A"), ds = root.getChildFile("DS"), native = root.getChildFile("Native");
    for (const auto& f : {a, b, manual, ds, native}) f.createDirectory();
    for (const auto& f : {a, b, native})
    { f.getChildFile("used.wav").replaceWithText(""); f.getChildFile("unused.wav").replaceWithText(""); }
    ds.getChildFile("dsconfig.yaml").replaceWithText("{}\n");
    ds.getChildFile("used.wav").replaceWithText("");
    juce::PropertiesFile::Options options; options.storageFormat = juce::PropertiesFile::storeAsXML;
    s->preferences = std::make_unique<juce::PropertiesFile>(root.getChildFile("global.xml"), options);
    // An old installation's registry must not silently be adopted by any document.
    s->preferences->setValue("assets.materialFolders", manual.getFullPathName());
    s->preferences->setValue("assets.hiddenFolders", a.getFullPathName().replaceCharacter('\\', '/').toLowerCase());
    s->preferences->saveIfNeeded();
    const auto globalHash = juce::SHA256(s->preferences->getFile()).toHexString();
    s->browser = std::make_unique<AssetManagerComponent>(s->strings, *s->preferences);
    s->browser->onBrowserStateChanged = [weak = std::weak_ptr<State>(s)](const auto& state, const auto& session)
    { if (auto p = weak.lock()) if (p->model.snapshot().materialBrowserSessionId == session) p->model.setMaterialBrowserState(state); };
    auto& browser = *s->browser;
    s->check("legacy_global_registry_not_adopted", browser.diagnosticFolderCount() == 0);
    ProjectData projectA; TrackData t; t.id = "a"; t.trackType = TrackType::utau;
    t.pitchAlgorithm = PitchAlgorithm::utau; t.voicebankDirectory = a;
    projectA.tracks.push_back(t);
    t.id = "jie"; t.utauMode = UtauMode::jie; projectA.tracks.push_back(t);
    t.id = "mou"; t.utauMode = UtauMode::mou; projectA.tracks.push_back(t);
    t.id = "ds"; t.trackType = TrackType::diffSinger; t.voicebankDirectory = ds;
    ClipData dsClip; dsClip.id = "ds-source"; dsClip.sourceFile = ds.getChildFile("used.wav"); t.clips = {dsClip};
    projectA.tracks.push_back(t);
    TrackData n; n.id = "native"; ClipData clip, part; clip.id = "outer"; part.id = "inner";
    part.sourceFile = native.getChildFile("used.wav"); clip.parts.push_back(part); n.clips.push_back(clip); projectA.tracks.push_back(n);
    s->show(projectA);
    s->check("all_utau_modes_share_one_bank", browser.diagnosticProjectVoicebanks() == juce::StringArray{a.getFullPathName()});
    s->check("native_nested_source_auto_included_ds_excluded", browser.materialRoots().size() == 2
        && browser.materialRoots().contains(native.getFullPathName()) && !browser.materialRoots().contains(ds.getFullPathName()));
    s->check("native_only_used_recording", browser.diagnosticMembersOf(0) == juce::StringArray{"used.wav"});
    s->check("auto_banks_never_migrated", !a.getChildFile("otto.ini").exists() && !b.getChildFile("otto.ini").exists());
    juce::String error;
    s->check("manual_register", browser.registerMaterialRoot(manual, error));
    browser.removeFromBrowser(a.getChildFile("unused.wav"), false);
    browser.removeFromBrowser(a, true);
    const auto savedA = s->model.snapshot();
    const auto fileA = root.getChildFile("A.hjpx"), fileB = root.getChildFile("B.hjpx");
    s->check("save_A", s->model.save(fileA, error));
    ProjectData projectB; t = TrackData{}; t.id = "b"; t.trackType = TrackType::utau;
    t.pitchAlgorithm = PitchAlgorithm::utau; t.voicebankDirectory = b; projectB.tracks = {t};
    s->show(projectB);
    s->check("B_never_contains_A_manual_or_auto", browser.materialRoots() == juce::StringArray{b.getFullPathName()});
    s->check("save_B", s->model.save(fileB, error));
    // Even a shared bank has independent removal memory.
    projectB.tracks.front().voicebankDirectory = a; s->show(projectB);
    s->check("A_removal_does_not_hide_shared_bank_in_B", browser.diagnosticTreeRowCount() == 1);
    browser.diagnosticSearchEditor().setText(".wav", false); browser.diagnosticSearchEditor().onTextChange();
    s->check("A_recording_removal_does_not_affect_B", browser.diagnosticSearchLabels().size() == 2);
    browser.removeFromBrowser(a.getChildFile("unused.wav"), false);
    const auto sharedFile = root.getChildFile("shared-bank.hjpx");
    s->check("save_individual_removal", s->model.save(sharedFile, error));
    s->show(ProjectData{});
    s->check("reload_individual_removal", s->model.load(sharedFile, error));
    browser.setProjectVoicebanks(s->model.snapshot());
    browser.diagnosticSearchEditor().setText(".wav", false); browser.diagnosticSearchEditor().onTextChange();
    s->check("individual_removal_survives_project_file_roundtrip", browser.diagnosticSearchLabels().size() == 1
        && browser.diagnosticSearchLabels()[0].startsWith("used.wav"));
    s->show(ProjectData{});
    s->check("new_project_empty_and_search_cleared", browser.diagnosticFolderCount() == 0
        && browser.diagnosticSearchEditor().getText().isEmpty() && browser.diagnosticSelectedAudio() == juce::File{});
    s->check("load_A", s->model.load(fileA, error)); browser.setProjectVoicebanks(s->model.snapshot());
    s->check("A_manual_restored", browser.materialRoots().contains(manual.getFullPathName()));
    s->check("A_hidden_bank_restored", browser.diagnosticTreeRowCount() == 3); // native folder + used audio + manual folder
    const auto session = s->model.snapshot().materialBrowserSessionId;
    s->model.setTrackMuted("a", true); s->model.undo(); s->model.redo();
    s->check("undo_redo_preserve_browser_memory", s->model.snapshot().materialBrowserState == savedA.materialBrowserState
        && s->model.snapshot().materialBrowserSessionId == session);
    auto replacement = s->model.snapshot(); replacement.materialBrowserState.clear(); s->model.replace(replacement);
    s->check("ordinary_replace_preserves_browser_memory", s->model.snapshot().materialBrowserState == savedA.materialBrowserState);
    s->check("load_B", s->model.load(fileB, error)); browser.setProjectVoicebanks(s->model.snapshot());
    s->check("reopened_B_isolated", browser.materialRoots() == juce::StringArray{b.getFullPathName()});
    for (int i = 0; i < 4; ++i)
    {
        s->model.load(fileA, error); browser.setProjectVoicebanks(s->model.snapshot());
        s->check("repeat_A_isolated", browser.materialRoots().size() == 3 && !browser.materialRoots().contains(b.getFullPathName()));
        s->model.load(fileB, error); browser.setProjectVoicebanks(s->model.snapshot());
        s->check("repeat_B_isolated", browser.materialRoots() == juce::StringArray{b.getFullPathName()});
    }
    s->check("global_preferences_untouched", juce::SHA256(s->preferences->getFile()).toHexString() == globalHash);
    // Leave a bank search in flight, then switch to a project using just one
    // recording in that same folder. Old results cannot expand its membership.
    ProjectData search; t.voicebankDirectory = a; search.tracks = {t}; t.voicebankDirectory = b; search.tracks.push_back(t);
    s->show(search); browser.diagnosticSearchEditor().setText(".wav", false); browser.diagnosticSearchEditor().onTextChange();
    s->check("old_search_started", browser.diagnosticSearchPending());
    ProjectData isolated; n.clips.clear(); clip.parts.clear(); clip.sourceFile = b.getChildFile("used.wav"); n.clips.push_back(clip); isolated.tracks = {n};
    s->show(isolated);
    juce::Timer::callAfterDelay(250, [s, b, output, completed]
    {
        s->check("late_search_cannot_import_unrelated_recordings", s->browser->diagnosticMembersOf(0)
            == juce::StringArray{"used.wav"});
        s->check("late_search_keeps_new_project_selection_empty", s->browser->diagnosticSelectedAudio() == juce::File{}
            && !s->browser->diagnosticSearchPending());
        output.getChildFile("result.txt").replaceWithText(s->report);
        output.getChildFile("result.json").replaceWithText("{\"checks\":" + juce::String(s->checks) + ",\"ok\":" + (s->ok ? "true" : "false") + "}");
        completed(s->ok);
    });
}
}
