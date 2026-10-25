#pragma once
#include "../OttoMaterialIO.h"
#include "../OttoMaterialMigration.h"
#include "../MaterialPronunciationCatalog.h"
#include "../MaterialAnnotationSession.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <iostream>

namespace hachi
{
// Parent-owned command wiring; no runtime pass is implied by this source.
inline bool runMaterialVariantSmoke(const juce::File& output)
{
    using IO = OttoMaterialIO;
    using Catalog = MaterialPronunciationCatalog;
    const auto root = output.getChildFile("annotation-variants-" + juce::Uuid().toString());
    if (root.createDirectory().failed()) return false;
    const auto audio = root.getChildFile("one.wav"), manifest = root.getChildFile("otto.ini");
    bool ok = true;
    juce::String error, log;
    const auto check = [&](const char* name, bool passed)
    {
        ok &= passed;
        log += juce::String(name) + "=" + (passed ? "PASS" : "FAIL") + "\n";
        std::cout << name << '=' << passed << std::endl;
    };
    const auto bytes = [](const juce::File& file)
    { juce::MemoryBlock value; file.loadFileAsData(value); return value; };
    {
        juce::AudioBuffer<float> pcm(1, 48000); pcm.clear();
        juce::WavAudioFormat wav;
        auto stream = audio.createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(stream
            ? wav.createWriterFor(stream.release(), 48000, 1, 16, {}, 0) : nullptr);
        check("fixture", writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples()));
    }
    const auto sourceBytes = bytes(audio);
    const auto sourceHash = juce::SHA256(sourceBytes).toHexString();
    SampleRegionSetting a;
    a.name = "-"; a.materialAudioId = "take_A"; a.materialRegionId = "p01";
    a.regionStartSeconds = .1; a.regionEndSeconds = .6; a.alignmentSeconds = .2;
    a.fixedDurationSeconds = .15; a.overlapSeconds = -.02;
    a.melodyneData = true; a.melodynePitchCenterCents = 6123.25;
    IO::Document document;
    check("single_variant_registration", IO::replaceAudio(document, manifest, audio, {a}, error)
        && IO::saveDocument(manifest, document, error));
    std::vector<SampleRegionSetting> rows;
    juce::String token;
    check("old_single_read_unchanged", IO::loadAudio(manifest, audio, rows, error, &token)
        && rows.size() == 1 && rows[0].materialAudioId == "take_A" && rows[0].name == "-");
    auto b = a;
    b.materialAudioId = "take_B"; // Same alias AND same region ID are independent.
    b.regionStartSeconds = .25; b.regionEndSeconds = .9; b.alignmentSeconds = .3;
    b.melodynePitchCenterCents = 6500;
    check("same_path_distinct_variant_registration", IO::replaceAudio(document, manifest, audio, {b}, error)
        && document.audio.size() == 2 && IO::saveDocument(manifest, document, error));
    juce::String encoded;
    IO::Document decoded;
    check("v1_repeated_path_roundtrip", IO::encode(document, encoded, error)
        && IO::decode(encoded, decoded, error) && decoded.audio.size() == 2
        && decoded.audio[0].relativePath == decoded.audio[1].relativePath);
    auto duplicate = document;
    duplicate.audio.push_back(duplicate.audio[0]);
    check("duplicate_audio_id_rejected", !IO::validate(duplicate, error));
    auto duplicateRow = document;
    duplicateRow.audio[0].rows.push_back(a);
    check("duplicate_region_id_within_variant_rejected", !IO::validate(duplicateRow, error));
    auto foreignLink = document;
    foreignLink.audio[1].rows[0].materialRegionId = "other_region";
    foreignLink.audio[0].rows[0].connectedNextRegionId = "other_region";
    check("cross_variant_connection_rejected", !IO::validate(foreignLink, error));
    std::vector<IO::Audio> variants;
    check("explicit_list_all_variants", IO::listAudioVariants(manifest, audio, variants, error)
        && variants.size() == 2 && variants[0].id == "take_A" && variants[1].id == "take_B");
    rows = {a}; token = "unchanged";
    check("ambiguous_default_read_no_first_match", !IO::loadAudio(manifest, audio, rows, error, &token)
        && error.containsIgnoreCase("ambiguous") && rows.size() == 1
        && rows[0].materialAudioId == "take_A" && token == "unchanged");
    check("ambiguous_migration_read_no_fallback", !OttoMaterialMigration::readAudioAnnotations(
        audio, root, UtauMode::classic, rows, error) && error.containsIgnoreCase("ambiguous"));
    check("explicit_variant_B_read", OttoMaterialMigration::readAudioAnnotations(
        audio, root, UtauMode::classic, rows, error, "take_B")
        && rows.size() == 1 && rows[0].regionEndSeconds == .9 && rows[0].materialRegionId == "p01");
    check("explicit_missing_id_reject", !IO::loadAudio(manifest, audio, rows, error, nullptr, "absent"));
    check("settings_ambiguous_no_derived_row", SampleSettings::loadOrDerive(audio, ProjectData{}, &error).empty()
        && error.containsIgnoreCase("ambiguous"));
    const auto baseline = SampleSettings::loadOrDerive(audio, ProjectData{}, &error, false, "take_B");
    check("selected_local_source_baseline", error.isEmpty() && baseline.size() == 1
        && baseline[0].materialAudioId == "take_B" && baseline[0].melodynePitchCenterCents == 6500);
    const auto beforeFailure = bytes(manifest);
    auto noId = a; noId.materialAudioId.clear();
    check("path_only_save_reject", !IO::saveAudio(manifest, audio, {noId}, error));
    check("empty_path_only_save_reject", !IO::saveAudio(manifest, audio, {}, error));
    check("mixed_variant_rows_reject", !IO::saveAudio(manifest, audio, {a,b}, error));
    check("wrong_path_id_reject", !IO::saveAudio(manifest, root.getChildFile("different.wav"), {a}, error));
    check("failed_operations_manifest_exact", bytes(manifest) == beforeFailure);
    auto candidate = document;
    juce::String candidateBefore, candidateAfter;
    IO::encode(candidate, candidateBefore, error);
    check("failed_merge_transactional", !IO::replaceAudio(candidate, manifest, audio, {a,b}, error)
        && IO::encode(candidate, candidateAfter, error) && candidateBefore == candidateAfter);
    const auto catalog = Catalog::index({{root, UtauMode::classic}});
    check("catalog_both_same_alias_and_region", catalog.ok() && catalog.entries.size() == 2
        && catalog.entries[0].audioId != catalog.entries[1].audioId);
    bool allResolved = catalog.entries.size() == 2;
    for (const auto& entry : catalog.entries)
    {
        Catalog::Entry resolved;
        std::vector<SampleRegionSetting> full;
        allResolved &= Catalog::resolve(entry, resolved, error, &full) && full.size() == 1
            && full[0].materialAudioId == entry.audioId && resolved.audioId == entry.audioId
            && full[0].regionEndSeconds == (entry.audioId == "take_A" ? .6 : .9);
    }
    check("catalog_resolve_selected_full_rows_only", allResolved);
    IO::Document siblingBefore; siblingBefore.audio.push_back(document.audio[1]);
    juce::String siblingTextBefore, siblingTextAfter;
    IO::encode(siblingBefore, siblingTextBefore, error);
    check("load_variant_token", IO::loadAudio(manifest, audio, rows, error, &token, "take_A"));
    if (!rows.empty()) rows[0].name = "renamed";
    auto inserted = a;
    inserted.name = "new"; inserted.materialRegionId.clear();
    inserted.regionStartSeconds = .7; inserted.regionEndSeconds = .95;
    inserted.alignmentSeconds = .75;
    rows.insert(rows.begin(), inserted); // New order must not reassign existing identity.
    check("update_only_exact_variant", IO::saveAudio(manifest, audio, rows, error, token, "take_A"));
    IO::Document published;
    check("published_sibling_readback", IO::loadDocument(manifest, published, error) && published.audio.size() == 2);
    if (published.audio.size() == 2)
    {
        IO::Document siblingAfter; siblingAfter.audio.push_back(published.audio[1]);
        IO::encode(siblingAfter, siblingTextAfter, error);
        check("sibling_all_canonical_data_unchanged", siblingTextBefore == siblingTextAfter);
        const auto& updated = published.audio[0].rows;
        check("inserted_row_and_existing_identity_stable", updated.size() == 2
            && updated[0].materialRegionId != "p01" && updated[0].materialRegionId.isNotEmpty()
            && updated[1].materialRegionId == "p01" && updated[1].materialAudioId == "take_A"
            && updated[1].name == "renamed");
    }
    else check("sibling_all_canonical_data_unchanged", false);
    const auto afterUpdate = bytes(manifest);
    check("stale_variant_write_rejected", !IO::saveAudio(manifest, audio, {b}, error, token, "take_B"));
    check("stale_write_manifest_exact", bytes(manifest) == afterUpdate);
    check("stale_document_atomic_reject", !IO::saveDocument(manifest, document, error)
        && bytes(manifest) == afterUpdate);
    check("explicit_empty_variant_creation", IO::saveAudio(manifest, audio, {}, error, {}, "take_C"));
    check("explicit_empty_variant_read", IO::loadAudio(manifest, audio, rows, error, nullptr, "take_C") && rows.empty());
    check("source_pcm_bytes_and_hash_exact", bytes(audio) == sourceBytes
        && juce::SHA256(bytes(audio)).toHexString() == sourceHash);
    check("no_parallel_legacy_format", !SampleSettings::sidecarFor(audio).exists());
    if (!output.createDirectory().wasOk() || !output.getChildFile("material-variants.txt").replaceWithText(log)) return false;
    return ok;
}
}
