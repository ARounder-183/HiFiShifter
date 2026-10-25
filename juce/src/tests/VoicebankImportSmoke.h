#pragma once
#include "../AssetManagerComponent.h"
#include "../backend/UtauRenderer.h"

namespace hachi
{
inline bool runVoicebankImportSmoke(const juce::File& output, const juce::File& bank)
{
    if (!bank.isDirectory()) return false;
    output.createDirectory();
    juce::PropertiesFile::Options options;
    options.applicationName = "voicebank-import-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(output.getChildFile("registry.settings"), options);
    I18n strings;
    AssetManagerComponent manager(strings, properties);
    const auto builds = backend::UtauRenderer::diagnosticIndexBuilds();
    const auto waits = backend::UtauRenderer::diagnosticMessageThreadWaits();
    const auto started = juce::Time::getMillisecondCounterHiRes();
    // Same authoritative folder registration called by the import chooser.
    const auto index = manager.diagnosticRegisterFolder(bank);
    const auto elapsed = juce::Time::getMillisecondCounterHiRes() - started;
    const auto members = manager.diagnosticMembersOf(index);
    const auto count = manager.diagnosticFolderCount();
    bool ok = true;
    juce::StringArray report;
    const auto check = [&](const char* label, bool value)
    { ok &= value; report.add(juce::String(label) + "=" + (value ? "1" : "0")); };
    report.add("import_ms=" + juce::String(elapsed, 3));
    report.add("audio_files=" + juce::String(members.size()));
    check("registered", index >= 0 && !members.isEmpty());
    check("responsive", elapsed < 1500);
    check("saved_registry", properties.getValue("assets.materialFolders").contains(bank.getFullPathName()));
    check("no_index_build", backend::UtauRenderer::diagnosticIndexBuilds() == builds);
    check("no_index_wait", backend::UtauRenderer::diagnosticMessageThreadWaits() == waits);
    check("repeat_import_no_duplicate", manager.diagnosticRegisterFolder(bank) == index && manager.diagnosticFolderCount() == count);
    check("invalid_directory_no_edit", manager.diagnosticRegisterFolder({}) < 0 && manager.diagnosticFolderCount() == count);
    AssetManagerComponent reopened(strings, properties);
    check("registration_survives_reopen", reopened.diagnosticFolderCount() == count && reopened.diagnosticMembersOf(index) == members);
    const auto text = report.joinIntoString("\n") + "\n";
    std::cout << text << std::flush;
    return output.getChildFile("import.txt").replaceWithText(text) && ok;
}
}
