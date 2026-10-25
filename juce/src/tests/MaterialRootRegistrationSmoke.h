#pragma once
#include "../AssetManagerComponent.h"
#include "../OttoMaterialIO.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
// Production API test; not registered until the parent adds a command entry.
inline bool runMaterialRootRegistrationSmoke(const juce::File& output)
{
    const auto base = output.getChildFile("root-registration-" + juce::Uuid().toString());
    const auto bank = base.getChildFile("bank");
    if (bank.createDirectory().failed()) return false;
    const auto manifest = bank.getChildFile("otto.ini"), audio = bank.getChildFile("original.wav");
    {
        juce::AudioBuffer<float> pcm(1, 4800); pcm.clear();
        auto stream = audio.createOutputStream(); if (!stream) return false;
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        if (!writer || !writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false;
    }
    OttoMaterialIO::Document doc;
    juce::String error;
    if (!OttoMaterialIO::saveDocument(manifest, doc, error)) return false;
    const auto bytes = [](const juce::File& file)
    { juce::MemoryBlock data; file.loadFileAsData(data); return data; };
    const auto sourceBytes = bytes(audio), sourceManifest = bytes(manifest);
    bool ok = true; juce::String report;
    const auto check = [&](const char* name, bool passed)
    { ok &= passed; report += juce::String(name) + "=" + (passed ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << passed << std::endl; };
    juce::PropertiesFile::Options options; options.applicationName = "root-register-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    I18n strings;
    juce::PropertiesFile preferences(base.getChildFile("preferences.xml"), options);
    AssetManagerComponent manager(strings, preferences);
    check("production_registration_success", manager.registerMaterialRoot(bank, error) && error.isEmpty());
    check("production_registration_idempotent", manager.registerMaterialRoot(bank, error)
        && manager.diagnosticFolderCount() == 1 && manager.materialRoots() == juce::StringArray{bank.getFullPathName()});
    {
        juce::PropertiesFile reopened(base.getChildFile("preferences.xml"), options);
        check("preferences_actually_persisted", reopened.getValue("assets.materialFolders").contains(bank.getFullPathName()));
    }
    check("no_audio_copy_or_annotation_edit", bytes(audio) == sourceBytes && bytes(manifest) == sourceManifest
        && bank.findChildFiles(juce::File::findFiles, false).size() == 2 && !base.getChildFile("Copy").exists());
    check("invalid_directory_rejects", !manager.registerMaterialRoot(base.getChildFile("missing"), error)
        && error.isNotEmpty() && manager.diagnosticFolderCount() == 1);
    const auto bad = base.getChildFile("invalid-manifest"); bad.createDirectory();
    bad.getChildFile("otto.ini").replaceWithText("[otto]\nversion=99\nunit=ms\n");
    const auto badBytes = bytes(bad.getChildFile("otto.ini"));
    check("invalid_manifest_rejects_without_mutation", !manager.registerMaterialRoot(bad, error)
        && error.isNotEmpty() && bytes(bad.getChildFile("otto.ini")) == badBytes && manager.diagnosticFolderCount() == 1);
    // A regular file as a parent makes persistence fail deterministically without
    // changing filesystem permissions, deleting user data, or relying on ACLs.
    const auto blocker = base.getChildFile("blocked-parent");
    check("preference_failure_fixture", blocker.replaceWithText("fixture: not a directory"));
    {
        juce::PropertiesFile blockedPreferences(blocker.getChildFile("preferences.xml"), options);
        AssetManagerComponent blockedManager(strings, blockedPreferences);
        check("reports_preferences_failure_after_registration", !blockedManager.registerMaterialRoot(bank, error)
            && error.containsIgnoreCase("preferences") && error.containsIgnoreCase("retained in memory"));
        check("adopted_root_retained_on_preferences_failure", blockedManager.materialRoots() == juce::StringArray{bank.getFullPathName()}
            && blockedManager.diagnosticFolderCount() == 1 && audio.existsAsFile() && manifest.existsAsFile());
        check("failed_preference_retry_never_duplicates", !blockedManager.registerMaterialRoot(bank, error)
            && blockedManager.diagnosticFolderCount() == 1);
    }
    check("failed_operations_preserve_source_bytes", bytes(audio) == sourceBytes && bytes(manifest) == sourceManifest);
    return output.getChildFile("material-root-registration.txt").replaceWithText(report) && ok;
}
}
