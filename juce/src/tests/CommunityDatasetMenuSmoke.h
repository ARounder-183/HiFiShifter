#pragma once
#include "../AssetManagerComponent.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
// Unregistered GUI-entry smoke. Tests the shared production menu action/dispatch,
// not the host wizard, parser, publication or end-to-end Main integration.
inline bool runCommunityDatasetMenuSmoke(const juce::File& output)
{
    const auto base = output.getChildFile("dataset-menu-" + juce::Uuid().toString());
    const auto root = base.getChildFile("dataset");
    if (root.createDirectory().failed()) return false;
    const auto audio = root.getChildFile("original.wav"), csv = root.getChildFile("transcriptions.csv");
    {
        juce::AudioBuffer<float> pcm(1, 4800); pcm.clear(); auto stream = audio.createOutputStream(); if (!stream) return false;
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        if (!writer || !writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false;
    }
    if (!csv.replaceWithText("deliberately invalid dataset input; menu must not parse this\n")) return false;
    const auto bytes = [](const juce::File& file)
    { juce::MemoryBlock data; file.loadFileAsData(data); return data; };
    juce::PropertiesFile::Options options; options.applicationName = "dataset-menu-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(base.getChildFile("preferences.xml"), options);
    I18n strings; AssetManagerComponent manager(strings, properties);
    bool ok = true; juce::String log, error;
    const auto check = [&](const char* label, bool passed)
    { ok &= passed; log += juce::String(label) + "=" + (passed ? "PASS" : "FAIL") + "\n"; std::cout << label << '=' << passed << std::endl; };
    check("production_root_registration", manager.registerMaterialRoot(root, error));
    const auto pcmBefore = bytes(audio), csvBefore = bytes(csv), manifestBefore = bytes(root.getChildFile("otto.ini"));
    const auto disabled = manager.diagnosticCommunityDatasetImportMenu();
    int disabledCount = 0; bool disabledItem = false;
    for (juce::PopupMenu::MenuItemIterator it(disabled); it.next();)
    { ++disabledCount; disabledItem = !it.getItem().isEnabled && it.getItem().text.contains("DiffSinger"); }
    check("exactly_one_shared_disabled_action_without_host", disabledCount == 1 && disabledItem
        && !manager.diagnosticRequestCommunityDatasetImport(root));
    int requests = 0; juce::File requested;
    manager.onRequestCommunityDatasetImport = [&](const juce::File& initialRoot) { ++requests; requested = initialRoot; };
    const auto enabled = manager.diagnosticCommunityDatasetImportMenu();
    int enabledCount = 0; bool enabledItem = false;
    for (juce::PopupMenu::MenuItemIterator it(enabled); it.next();)
    { ++enabledCount; enabledItem = it.getItem().isEnabled && it.getItem().text.contains("DiffSinger"); }
    check("exactly_one_shared_enabled_action_with_host", enabledCount == 1 && enabledItem);
    check("top_import_action_requests_chooser", manager.diagnosticRequestCommunityDatasetImport()
        && requests == 1 && requested == juce::File{});
    check("root_context_passes_exact_root", manager.diagnosticRequestCommunityDatasetImport(root)
        && requests == 2 && requested == root);
    manager.onRequestCommunityDatasetImport = {};
    check("unbound_callback_no_fake_success", !manager.diagnosticRequestCommunityDatasetImport(root) && requests == 2);
    check("menu_dispatch_no_parser_no_data_or_pcm_write", bytes(audio) == pcmBefore && bytes(csv) == csvBefore
        && bytes(root.getChildFile("otto.ini")) == manifestBefore && manager.diagnosticFolderCount() == 1
        && root.findChildFiles(juce::File::findFiles, false).size() == 3);
    return output.getChildFile("community-dataset-menu.txt").replaceWithText(log) && ok;
}
}
