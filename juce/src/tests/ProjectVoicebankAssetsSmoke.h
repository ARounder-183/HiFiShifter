#pragma once
#include "../AssetManagerComponent.h"
#include "../ProjectModel.h"
#include <iostream>

namespace hachi
{
inline bool runProjectVoicebankAssetsSmoke(const juce::File& root)
{
    root.createDirectory();
    juce::PropertiesFile::Options options;
    options.applicationName = "asset-project-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(root.getChildFile("assets.settings"), options);
    I18n strings;
    AssetManagerComponent manager(strings, properties);
    int failures = 0;
    const auto check = [&](const char* label, bool passed)
    {
        std::cout << label << "=" << (passed ? 1 : 0) << std::endl;
        if (!passed) ++failures;
    };
    const auto bankA = root.getChildFile("UTAU A");
    const auto bankB = root.getChildFile("UTAU B");
    const auto manual = root.getChildFile("Manual materials");
    const auto ds = root.getChildFile("DiffSinger");
    for (const auto& folder : {bankA, bankB, manual, ds}) folder.createDirectory();
    ds.getChildFile("dsconfig.yaml").replaceWithText("{}\n");
    // No audio needs decoding: the folder list must work even on empty banks.
    manager.diagnosticRegisterFolder(manual);
    const auto registry = properties.getValue("assets.materialFolders");
    ProjectData project;
    TrackData track;
    track.id = "bank-a";
    track.pitchAlgorithm = PitchAlgorithm::utau;
    track.voicebankDirectory = bankA;
    project.tracks = {track, track};
    project.tracks.back().id = "bank-a-2";
    check("shared_bank_once", manager.setProjectVoicebanks(project)
        && manager.diagnosticProjectVoicebanks().size() == 1 && manager.diagnosticFolderCount() == 2);
    check("no_global_import", properties.getValue("assets.materialFolders") == registry);
    project.tracks.front().muted = true;
    check("ordinary_edits_no_rescan", !manager.setProjectVoicebanks(project));
    project.tracks.erase(project.tracks.begin());
    check("remaining_user_keeps_bank", !manager.setProjectVoicebanks(project));
    track.id = "bank-b";
    track.voicebankDirectory = bankB;
    track.pitchAlgorithm = PitchAlgorithm::nsfHifigan;
    track.nativeNsfAudio = false;
    project.tracks.push_back(track);
    check("native_nsf_voicebank_included", manager.setProjectVoicebanks(project)
        && manager.diagnosticProjectVoicebanks().size() == 2);
    track.voicebankDirectory = manual;
    track.nativeNsfAudio = true;
    project.tracks.push_back(track);
    track.pitchAlgorithm = PitchAlgorithm::utau;
    track.accompaniment = true;
    project.tracks.push_back(track);
    track.accompaniment = false;
    track.voicebankDirectory = ds;
    track.utauMode = UtauMode::mou;
    project.tracks.push_back(track);
    check("exclude_native_accompaniment_diffsinger", !manager.setProjectVoicebanks(project));
    ProjectModel model;
    model.replace(project);
    juce::String error;
    const auto saved = model.save(root.getChildFile("project.hjpx"), error);
    ProjectModel reopened;
    const auto loaded = reopened.load(root.getChildFile("project.hjpx"), error);
    manager.setProjectVoicebanks(ProjectData{});
    check("new_project_clears_only_automatic", manager.diagnosticFolderCount() == 1
        && manager.diagnosticProjectVoicebanks().isEmpty());
    manager.setProjectVoicebanks(reopened.snapshot());
    check("reopen_restores_banks", saved && loaded && manager.diagnosticProjectVoicebanks().size() == 2);
    manager.setSize(720, 350);
    if (auto stream = root.getChildFile("project-assets.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(manager.createComponentSnapshot(manager.getLocalBounds()), *stream);
    // Explicit registration of an automatic bank promotes it to reusable material.
    manager.diagnosticRegisterFolder(bankA);
    check("registration_deduplicates", manager.diagnosticFolderCount() == 3);
    const auto promotedRegistry = properties.getValue("assets.materialFolders");
    check("automatic_bank_not_persisted_on_save", !promotedRegistry.contains(bankB.getFullPathName()));
    manager.setProjectVoicebanks(ProjectData{});
    check("manual_registration_survives", manager.diagnosticFolderCount() == 2);
    AssetManagerComponent nextSession(strings, properties);
    check("restart_does_not_leak_project_banks", nextSession.diagnosticFolderCount() == 2
        && nextSession.diagnosticProjectVoicebanks().isEmpty());
    project.tracks = {project.tracks.front()};
    project.tracks.front().voicebankDirectory = root.getChildFile("Missing bank");
    manager.setProjectVoicebanks(project);
    check("missing_reference_visible", manager.diagnosticProjectVoicebanks().size() == 1);
    check("sync_preserves_global_registry", properties.getValue("assets.materialFolders") == promotedRegistry);

    // Exercise actual row hit-testing at the narrow dock width, including scroll.
    const auto uiRoot = root.getChildFile("tree-ui");
    uiRoot.createDirectory();
    juce::PropertiesFile uiProperties(uiRoot.getChildFile("settings.xml"), options);
    AssetManagerComponent tree(strings, uiProperties);
    const auto uiBank = uiRoot.getChildFile("Voicebank");
    uiBank.getChildFile("C4").createDirectory();
    for (const auto* name : {"a.wav", "i.wav", "C4/u.wav"})
        uiBank.getChildFile(name).replaceWithText(""); // Enumeration only; no decoding.
    ProjectData uiProject;
    TrackData uiTrack;
    uiTrack.pitchAlgorithm = PitchAlgorithm::utau;
    uiTrack.voicebankDirectory = uiBank;
    uiProject.tracks = {uiTrack};
    tree.setProjectVoicebanks(uiProject);
    tree.setSize(320, 280);
    const auto click = [&](float x, float y)
    {
        const juce::Point<float> position(x, y);
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(), position,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), 0, 0, 0, 0, 0, &tree, &tree,
            juce::Time::getCurrentTime(), position, juce::Time::getCurrentTime(), 1, false);
    };
    check("tree_initially_collapsed", tree.diagnosticTreeRowCount() == 1);
    tree.mouseDown(click(16, 90));
    check("arrow_expands_nested_audio", tree.diagnosticTreeRowCount() == 4);
    tree.mouseDown(click(80, 130));
    check("expanded_audio_selectable", tree.diagnosticSelectedAudio() == uiBank.getChildFile("a.wav"));
    juce::File requested;
    int requests = 0;
    tree.onLoadAudio = [&](const juce::File& file) { requested = file; ++requests; };
    check("enter_loads_selected_original", tree.keyPressed(juce::KeyPress(juce::KeyPress::returnKey))
        && requested == uiBank.getChildFile("a.wav") && requests == 1);
    tree.mouseDown(click(90, 90));
    check("folder_does_not_load_audio", !tree.keyPressed(juce::KeyPress(juce::KeyPress::returnKey)) && requests == 1);
    tree.mouseDown(click(80, 130));
    check("ordinary_sync_preserves_expansion", !tree.setProjectVoicebanks(uiProject) && tree.diagnosticTreeRowCount() == 4);
    if (auto stream = root.getChildFile("expanded-assets.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(tree.createComponentSnapshot(tree.getLocalBounds()), *stream);
    tree.mouseDown(click(16, 90));
    check("arrow_collapses_audio", tree.diagnosticTreeRowCount() == 1);
    tree.mouseDoubleClick(click(90, 90));
    check("folder_doubleclick_expands", tree.diagnosticTreeRowCount() == 4);
    tree.mouseDown(click(16, 90));
    for (int i = 0; i < 60; ++i)
        uiBank.getChildFile("z" + juce::String(i) + ".wav").replaceWithText("");
    tree.mouseDown(click(16, 90));
    juce::MouseWheelDetails wheel{};
    wheel.deltaY = -100.0f;
    tree.mouseWheelMove(click(80, 130), wheel);
    tree.mouseDown(click(80, 260));
    check("scroll_reaches_last_audio", tree.diagnosticSelectedAudio() == uiBank.getChildFile("z59.wav"));
    if (auto stream = root.getChildFile("scrolled-assets.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(tree.createComponentSnapshot(tree.getLocalBounds()), *stream);
    tree.setProjectVoicebanks(ProjectData{});
    check("project_switch_removes_expanded_children", tree.diagnosticTreeRowCount() == 0);
    check("expansion_not_registered_globally", uiProperties.getValue("assets.materialFolders").isEmpty());
    // Bank identity, rather than selected row/index, owns the import format.
    juce::PropertiesFile bankProperties(uiRoot.getChildFile("bank-options.xml"), options);
    bankProperties.setValue("assets.otoImportKind", 4);
    AssetManagerComponent perBank(strings, bankProperties);
    perBank.diagnosticRegisterFolder(bankA);
    perBank.diagnosticRegisterFolder(bankB);
    check("legacy_oto_choice_initialises_banks", perBank.otoImportKind(bankA) == 4 && perBank.otoImportKind(bankB) == 4);
    perBank.setOtoImportKind(bankA, 3);
    perBank.setOtoImportKind(bankB, 2);
    check("bank_formats_are_independent", perBank.otoImportKind(bankA) == 3 && perBank.otoImportKind(bankB) == 2);
    const auto aFile = bankA.getChildFile("nested/a.wav"), bFile = bankB.getChildFile("b.wav");
    check("drag_uses_source_bank_not_selected_bank", static_cast<int>(perBank.audioDragDescription(aFile)["otoKind"]) == 3
        && static_cast<int>(perBank.audioDragDescription(bFile)["otoKind"]) == 2
        && perBank.rootForAudio(aFile) == bankA);
    ProjectData reordered; TrackData bankTrack; bankTrack.pitchAlgorithm = PitchAlgorithm::utau;
    bankTrack.voicebankDirectory = bankB; reordered.tracks = {bankTrack};
    perBank.setProjectVoicebanks(reordered);
    check("folder_reorder_keeps_oto_choices", perBank.otoImportKind(aFile) == 3 && perBank.otoImportKind(bFile) == 2);
    juce::PropertiesFile restartedProperties(uiRoot.getChildFile("bank-options.xml"), options);
    AssetManagerComponent restarted(strings, restartedProperties);
    check("bank_choices_survive_restart", restarted.otoImportKind(aFile) == 3 && restarted.otoImportKind(bFile) == 2);
    perBank.setSize(320, 240);
    if (auto stream = root.getChildFile("per-bank-oto.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(perBank.createComponentSnapshot(perBank.getLocalBounds()), *stream);
    perBank.setSize(220, 240);
    if (auto stream = root.getChildFile("per-bank-oto-narrow.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(perBank.createComponentSnapshot(perBank.getLocalBounds()), *stream);
    return failures == 0;
}

inline void runAssetSearchSmoke(const juce::File& root, std::function<void(bool)> completed)
{
    struct State
    {
        I18n strings;
        std::unique_ptr<juce::PropertiesFile> settings;
        std::unique_ptr<AssetManagerComponent> manager;
        bool ok = true;
        void check(const char* name, bool pass)
        { ok &= pass; std::cout << name << "=" << pass << std::endl; }
    };
    auto state = std::make_shared<State>();
    root.createDirectory();
    const auto bankA = root.getChildFile("Voice A"), bankB = root.getChildFile("Voice B");
    bankA.getChildFile("C4").createDirectory(); bankB.getChildFile("D4").createDirectory();
    const auto fileA = bankA.getChildFile("C4/gou.wav"), fileB = bankB.getChildFile("D4/gou.wav");
    fileA.replaceWithText(""); fileB.replaceWithText("");
    bankB.getChildFile("D4/gong.wav").replaceWithText("");
    bankB.getChildFile(juce::String::fromUTF8("D4/你好.flac")).replaceWithText("");
    bankB.getChildFile("D4/gou.txt").replaceWithText("");
    juce::PropertiesFile::Options options; options.applicationName="asset-search-smoke";
    state->settings=std::make_unique<juce::PropertiesFile>(root.getChildFile("settings.xml"), options);
    state->manager=std::make_unique<AssetManagerComponent>(state->strings,*state->settings);
    auto& manager=*state->manager;
    ProjectData project; TrackData track; track.pitchAlgorithm=PitchAlgorithm::utau;
    track.voicebankDirectory=bankA; project.tracks.push_back(track);
    track.voicebankDirectory=bankB; project.tracks.push_back(track);
    manager.setProjectVoicebanks(project); manager.setSize(320,280);
    manager.setOtoImportKind(bankA,2); manager.setOtoImportKind(bankB,3);
    const auto query=[&manager](const juce::String& text)
    { manager.diagnosticSearchEditor().setText(text,false); manager.diagnosticSearchEditor().onTextChange(); };
    query("go"); query("GOU");
    state->check("search_uses_background_for_unopened_bank",manager.diagnosticSearchPending());
    auto poll=std::make_shared<std::function<void()>>();
    const auto weak=std::weak_ptr<std::function<void()>>(poll);
    const auto start=juce::Time::getMillisecondCounterHiRes();
    *poll=[state,root,fileA,fileB,bankB,completed,weak,start]
    {
        auto& manager=*state->manager;
        if(manager.diagnosticSearchPending()&&juce::Time::getMillisecondCounterHiRes()-start<10000)
        { if(auto next=weak.lock())juce::Timer::callAfterDelay(20,[next]{(*next)();});return; }
        state->check("search_scan_completes",!manager.diagnosticSearchPending());
        const auto labels=manager.diagnosticSearchLabels();
        state->check("search_latest_query_case_insensitive_and_excludes_non_audio",labels.size()==2);
        state->check("search_labels_identify_both_voicebanks",labels.contains(juce::String::fromUTF8("gou.wav（Voice A）"))
            &&labels.contains(juce::String::fromUTF8("gou.wav（Voice B）")));
        const juce::Point<float> position(80,106); // Second compact search row.
        manager.mouseDown(juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),position,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&manager,&manager,
            juce::Time::getCurrentTime(),position,juce::Time::getCurrentTime(),1,false));
        state->check("search_result_selects_exact_source",manager.diagnosticSelectedAudio()==fileB);
        juce::File requested; manager.onLoadAudio=[&requested](const juce::File& f){requested=f;};
        state->check("search_result_enter_loads_original",manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey))&&requested==fileB);
        manager.onLoadAudio=nullptr;
        const auto drag=manager.audioDragDescription(manager.diagnosticSelectedAudio());
        state->check("search_result_drag_keeps_bank_auto_authority",drag["file"].toString()==fileB.getFullPathName()
            &&drag["root"].toString()==bankB.getFullPathName()&&static_cast<int>(drag["otoKind"])==1);
        if(auto stream=root.getChildFile("search-results.png").createOutputStream())
            juce::PNGImageFormat().writeImageToStream(manager.createComponentSnapshot(manager.getLocalBounds()),*stream);
        const auto query=[&manager](const juce::String& text)
        {manager.diagnosticSearchEditor().setText(text,false);manager.diagnosticSearchEditor().onTextChange();};
        query(juce::String::fromUTF8("你好"));
        state->check("search_chinese_and_flac",manager.diagnosticSearchLabels().size()==1
            &&manager.diagnosticSearchLabels()[0]==juce::String::fromUTF8("你好.flac（Voice B）"));
        query("missing");
        state->check("search_no_results_clears_old_selection",manager.diagnosticTreeRowCount()==0
            &&!manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey)));
        query("GOU"); manager.setProjectVoicebanks(ProjectData{});
        state->check("search_project_switch_removes_old_banks",manager.diagnosticTreeRowCount()==0);
        manager.diagnosticRegisterFolder(fileA.getParentDirectory().getParentDirectory());
        state->check("search_newly_registered_folder_appears",manager.diagnosticSearchLabels().size()==1);
        query("");
        state->check("search_clear_restores_collapsed_tree",manager.diagnosticTreeRowCount()==1);
        completed(state->ok);
    };
    juce::Timer::callAfterDelay(20,[poll]{(*poll)();});
}
}
