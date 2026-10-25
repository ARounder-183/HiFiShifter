#pragma once
#include "../OttoMaterialIO.h"
#include "../OttoMaterialMigration.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
inline bool runOttoMaterialSmoke(const juce::File& outputDirectory)
{
    const auto root = outputDirectory.getChildFile("otto-" + juce::Uuid().toString());
    if (!root.createDirectory()) return false;
    juce::Array<juce::var> checks;
    juce::String error;
    int failed = 0;
    const auto check = [&](const char* name, bool ok)
    {
        auto* record = new juce::DynamicObject();
        record->setProperty("check", name); record->setProperty("passed", ok);
        if (!ok && error.isNotEmpty()) record->setProperty("error", error);
        checks.add(juce::var(record));
        if (!ok) { ++failed; std::cerr << "FAIL " << name << std::endl; }
    };
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-8; };
    const auto tone = [](const juce::File& file)
    {
        file.getParentDirectory().createDirectory();
        juce::AudioBuffer<float> pcm(1, 48000); pcm.clear();
        juce::WavAudioFormat format;
        auto stream = file.createOutputStream();
        if (!stream) return false;
        std::unique_ptr<juce::AudioFormatWriter> writer(
            format.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        return writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples());
    };
    const auto a = root.getChildFile(juce::String::fromUTF8("takes/声 源=1].wav"));
    const auto b = root.getChildFile("other.wav");
    check("audio_fixtures", tone(a) && tone(b));
    const auto manifest = root.getChildFile("otto.ini");
    SampleSettings::setMaterialRoots({root.getFullPathName()});
    check("registered_root_covers_subdirectories", SampleSettings::annotationFor(a) == manifest);

    SampleRegionSetting first;
    first.name = "ka"; first.materialAudioId = "a01"; first.materialRegionId = "p01";
    first.regionStartSeconds = .1; first.regionEndSeconds = .6;
    first.fixedDurationSeconds = .1; first.alignmentSeconds = .18; first.overlapSeconds = .02;
    first.additionalAliases.add(juce::String::fromUTF8("k;a|= %音"));
    first.connectedNextRegionId = "p02";
    first.segments = {{"c", "k", NativeSegmentRole::consonant, .1, .2, "user", 1.f, .18, .02, false, 1.0},
                      {"v1", "a", NativeSegmentRole::vowel, .2, .35, "user", .8f, .2, 0, true, .7},
                      {"v2", "a", NativeSegmentRole::vowel, .35, .6, "user", .9f, .35, 0, true, 1.2}};
    first.melodyneData = true; first.melodynePitchCenterCents = 6123.5;
    first.melodyneOriginalPitchCenterCents = 6010;
    first.melodynePitchDrift = .75; first.melodynePitchModulation = .25;
    first.melodyneTransitionSeconds = .012; first.melodyneFormantCents = 55;
    first.melodyneAmplitude = .7; first.melodyneSibilantBalance = .3;
    first.melodyneAttackSeconds = .045; first.melodyneDecayElongation = .2;
    first.relativePitchCents = 13.25; first.role = NativeSegmentRole::consonant;
    first.provenance = "user"; first.confidence = .75f;
    first.amplitudeEnvelope = {{0, -12, true}, {.3, -3, false}};
    SampleRegionSetting second;
    second.name = "a"; second.materialAudioId = "a01"; second.materialRegionId = "p02";
    second.regionStartSeconds = .6; second.regionEndSeconds = 1;
    second.alignmentSeconds = .6; second.pronunciationKind = PronunciationKind::transition;
    second.role = NativeSegmentRole::transition;
    second.segments = {{"v", "a", NativeSegmentRole::vowel, .6, 1, "user", 1.f, .6, 0, true, 1}};
    check("save_multi_pronunciation", SampleSettings::save(a, {first, second}, error));
    std::vector<SampleRegionSetting> read;
    juce::String token;
    const auto loaded = OttoMaterialIO::loadAudio(manifest, a, read, error, &token);
    check("read_multi_pronunciation", loaded && read.size() == 2);
    if (loaded && read.size() == 2)
    {
        const auto& r = read.front();
        check("stable_audio_and_pronunciation_ids", r.materialAudioId == "a01" && r.materialRegionId == "p01");
        check("absolute_segment_coordinates", r.segments.size() == 3
            && near(r.segments[0].sourceStartSeconds, .1) && near(r.segments[2].sourceEndSeconds, .6));
        check("vowel_multisegment_metadata", r.segments.size() == 3
            && r.segments[1].id == "v1" && r.segments[1].alias == "a"
            && near(r.segments[1].stretchWeight, .7) && !r.segments[0].stretchable);
        check("special_alias_escaped_roundtrip", r.additionalAliases == first.additionalAliases);
        check("continuous_link_and_transition", r.connectedNextRegionId == "p02" && read[1].pronunciationKind == PronunciationKind::transition);
        check("expression_metadata_preserved", r.melodyneData && near(r.melodynePitchCenterCents, 6123.5)
            && near(r.melodyneFormantCents, 55) && near(r.melodynePitchModulation, .25)
            && near(r.melodyneAmplitude, .7) && near(r.relativePitchCents, 13.25)
            && near(r.melodyneDecayElongation, .2));
        check("envelope_interpolation_preserved", r.amplitudeEnvelope.size() == 2
            && r.amplitudeEnvelope[0].linearToNext && near(r.amplitudeEnvelope[1].gainDb, -3));
    }
    {
        OttoMaterialIO::Document kinds;
        auto typed = first; typed.pronunciationKind = PronunciationKind::breath; typed.simplifiedConsonant = true;
        kinds.audio.push_back({"a01","takes/kinds.wav",{typed,second}});
        juce::String encoded;
        OttoMaterialIO::Document decoded;
        check("kind_and_simplified_flags_are_compact",OttoMaterialIO::encode(kinds,encoded,error)
            &&encoded.contains("flags:breath,simple-consonant"));
        check("whole_kind_is_not_internal_cv_role",OttoMaterialIO::decode(encoded,decoded,error)
            &&decoded.audio[0].rows[0].pronunciationKind==PronunciationKind::breath
            &&decoded.audio[0].rows[0].simplifiedConsonant&&decoded.audio[0].rows[0].segments[0].role==NativeSegmentRole::consonant);
        check("contradictory_whole_kinds_rejected",!OttoMaterialIO::decode(
            encoded.replace("flags:breath,simple-consonant","flags:normal,breath"),decoded,error));
        typed.pronunciationKind=PronunciationKind::normal;typed.role=NativeSegmentRole::breath;typed.simplifiedConsonant=false;
        kinds.audio[0].rows[0]=typed;
        check("explicit_normal_preserves_legacy_internal_role",OttoMaterialIO::encode(kinds,encoded,error)
            &&encoded.contains("flags:normal")&&OttoMaterialIO::decode(encoded,decoded,error)
            &&decoded.audio[0].rows[0].pronunciationKind==PronunciationKind::normal
            &&decoded.audio[0].rows[0].role==NativeSegmentRole::breath);
        kinds.audio[0].rows[0].simplifiedConsonant=true;kinds.audio[0].rows[0].fixedDurationSeconds=.01;
        check("simplified_cannot_precede_preutterance",!OttoMaterialIO::validate(kinds,error));
    }
    check("one_manifest_no_new_csv", manifest.existsAsFile()
        && !SampleSettings::sidecarFor(a).existsAsFile() && !a.getSiblingFile("otto.ini").existsAsFile());

    SampleRegionSetting other;
    other.name = "sa"; other.regionEndSeconds = 1;
    check("save_sibling_audio", SampleSettings::save(b, {other}, error));
    auto simpleIsCompact = false;
    for (const auto& line : juce::StringArray::fromLines(manifest.loadFileAsString()))
        if (line.contains("=sa|")) simpleIsCompact = !line.contains("meta:") && line.length() < 100;
    check("ordinary_pronunciation_has_no_verbose_metadata", simpleIsCompact);
    auto sibling = SampleSettings::loadOrDerive(b, ProjectData{}, &error);
    auto kept = SampleSettings::loadOrDerive(a, ProjectData{}, &error);
    check("sibling_merge_preserves_existing", sibling.size() == 1 && kept.size() == 2 && kept[0].materialRegionId == "p01");
    const auto siblingId = sibling.empty() ? juce::String() : sibling[0].materialRegionId;
    const auto siblingAudioId = sibling.empty() ? juce::String() : sibling[0].materialAudioId;
    if (!sibling.empty())
    {
        sibling[0].name = "renamed";
        check("rename_save", SampleSettings::save(b, sibling, error));
        sibling = SampleSettings::loadOrDerive(b, ProjectData{}, &error);
        check("rename_keeps_identity", sibling.size() == 1 && sibling[0].materialRegionId == siblingId
            && sibling[0].materialAudioId == siblingAudioId && sibling[0].name == "renamed");
    }
    const auto beforeInvalid = manifest.loadFileAsString();
    auto invalid = kept;
    if (!invalid.empty()) invalid[0].connectedNextRegionId = invalid[0].materialRegionId;
    check("reject_self_link", !OttoMaterialIO::saveAudio(manifest, a, invalid, error));
    check("failed_write_preserves_file", manifest.loadFileAsString() == beforeInvalid);
    error.clear();
    check("explicit_stale_draft_rejected", !OttoMaterialIO::saveAudio(manifest, a, read, error, token));

    OttoMaterialIO::Document document;
    error.clear();
    check("decode_traversal_rejected", !OttoMaterialIO::decode(
        "[otto]\nversion=1\nunit=ms\n[a01:../escape.wav]\np01=a|0,100\n", document, error));
    check("future_version_rejected", !OttoMaterialIO::decode("[otto]\nversion=99\nunit=ms\n", document, error));
    check("unknown_fields_not_silently_dropped", !OttoMaterialIO::decode(
        "[otto]\nversion=1\nunit=ms\nunknown=must-not-be-lost\n", document, error));
    const auto copy = root.getChildFile("copied.wav");
    check("copy_audio", a.copyFileTo(copy));
    check("copy_only_selected_audio_metadata", SampleSettings::copyAnnotations(a, copy, error));
    const auto copied = SampleSettings::loadOrDerive(copy, ProjectData{}, &error);
    check("copy_has_new_audio_identity", copied.size() == 2 && !copied[0].materialAudioId.isEmpty()
        && copied[0].materialAudioId != "a01" && copied[0].connectedNextRegionId == "p02");

    const auto legacyRoot = root.getChildFile("legacy"); legacyRoot.createDirectory();
    const auto legacyAudio = legacyRoot.getChildFile("old.wav");
    check("legacy_audio_fixture", tone(legacyAudio));
    SampleSettings::sidecarFor(legacyAudio).replaceWithText(
        "name,region_start_sec,region_end_sec,note_alignment_sec,fixed_duration_sec,relative_pitch_cents\n"
        "old,0,1,0.08,0.1,0\n");
    const auto oldBytes = SampleSettings::sidecarFor(legacyAudio).loadFileAsString();
    const auto oldRows = SampleSettings::loadOrDerive(legacyAudio, ProjectData{}, &error, true);
    check("legacy_csv_read_supported", error.isEmpty() && oldRows.size() == 1 && oldRows[0].name == "old");
    check("otto_authority_requires_explicit_legacy_migration", SampleSettings::loadOrDerive(legacyAudio, ProjectData{}).empty()
        && !manifest.loadFileAsString().contains("old.wav"));
    check("explicit_save_migrates_to_root_manifest", SampleSettings::save(legacyAudio, oldRows, error));
    check("legacy_file_unchanged", SampleSettings::sidecarFor(legacyAudio).loadFileAsString() == oldBytes);
    check("migrated_read_is_native", SampleSettings::loadOrDerive(legacyAudio, ProjectData{}, &error).size() == 1
        && SampleSettings::hasNativeAnnotation(legacyAudio, &error));

    const auto repeatedAudio = root.getChildFile("repeated.wav"); tone(repeatedAudio);
    SampleRegionSetting repeatA, repeatB;
    repeatA.name = "a"; repeatA.regionEndSeconds = .4; repeatA.overlapSeconds = .8;
    repeatB.name = "a"; repeatB.regionStartSeconds = .4; repeatB.regionEndSeconds = 1;
    check("repeated_aliases_are_distinct_pronunciations", SampleSettings::save(repeatedAudio, {repeatA, repeatB}, error));
    const auto repeatedRows = SampleSettings::loadOrDerive(repeatedAudio, ProjectData{}, &error);
    check("repeated_alias_identity_and_long_overlap_preserved", repeatedRows.size() == 2
        && repeatedRows[0].materialRegionId != repeatedRows[1].materialRegionId
        && near(repeatedRows[0].overlapSeconds, .8));
    const auto emptyAudio = root.getChildFile("empty.wav"); tone(emptyAudio);
    check("empty_audio_section_is_authoritative", OttoMaterialIO::saveAudio(manifest, emptyAudio, {}, error)
        && SampleSettings::hasNativeAnnotation(emptyAudio, &error)
        && SampleSettings::loadOrDerive(emptyAudio, ProjectData{}, &error).empty());
    const auto corruptRoot = root.getChildFile("future"); corruptRoot.createDirectory();
    const auto corruptAudio = corruptRoot.getChildFile("source.wav"); tone(corruptAudio);
    corruptRoot.getChildFile("otto.ini").replaceWithText("[otto]\nversion=99\nunit=ms\n");
    SampleSettings::setMaterialRoots({corruptRoot.getFullPathName()});
    const auto corruptRows = SampleSettings::loadOrDerive(corruptAudio, ProjectData{}, &error);
    check("invalid_manifest_never_silently_derives", corruptRows.empty() && error.isNotEmpty());
    SampleSettings::setMaterialRoots({});

    const auto migrateRoot = root.getChildFile("migrate-folder"); migrateRoot.createDirectory();
    const auto migrateA = migrateRoot.getChildFile("sub/a.wav"); tone(migrateA);
    const auto migrateB = migrateRoot.getChildFile("sub/b.wav"); tone(migrateB);
    const auto migrateCsv = SampleSettings::sidecarFor(migrateA);
    migrateCsv.replaceWithText("name,region_start_sec,region_end_sec,note_alignment_sec,fixed_duration_sec,relative_pitch_cents\n"
        "ka,0.1,0.8,0.18,0.12,0\n");
    const auto legacyBeforeMigration = migrateCsv.loadFileAsString();
    OttoMaterialMigration::Report migration;
    check("folder_explicit_migration", OttoMaterialMigration::migrateFolder(
        migrateRoot, UtauMode::classic, false, migration, error));
    check("migration_reports_only_annotated_audio", migration.audioFiles == 1 && migration.pronunciations == 1);
    check("folder_migration_preserves_legacy_bytes", migrateCsv.loadFileAsString() == legacyBeforeMigration);
    check("folder_migration_idempotent", OttoMaterialMigration::migrateFolder(
        migrateRoot, UtauMode::classic, false, migration, error) && migration.audioFiles == 0);
    check("explicit_include_bare_adds_missing_audio_only", OttoMaterialMigration::migrateFolder(
        migrateRoot, UtauMode::classic, true, migration, error) && migration.audioFiles == 1);
    auto exportRows = SampleSettings::loadOrDerive(migrateA, ProjectData{}, &error);
    if (!exportRows.empty())
    {
        exportRows[0].additionalAliases.add("ka_alt");
        check("export_alias_fixture", SampleSettings::save(migrateA, exportRows, error));
    }
    const auto standardOto = migrateRoot.getChildFile("oto.ini");
    const juce::String untouched("; unrelated comment\r\nother.wav=other,1,2,3,4,5\r\n");
    standardOto.replaceWithText(untouched, false, false, "\r\n");
    OttoMaterialMigration::Report exported;
    check("aggregate_standard_oto_export", OttoMaterialMigration::exportFolder(
        migrateRoot, standardOto, false, exported, error));
    const auto otoText = standardOto.loadFileAsString();
    check("aggregate_export_preserves_other_rows_and_comments", otoText.contains("; unrelated comment")
        && otoText.contains("other.wav=other,1,2,3,4,5"));
    check("export_relative_subdirectory_and_aliases", otoText.contains("sub/a.wav=ka,")
        && otoText.contains("sub/a.wav=ka_alt,") && otoText.contains("sub/b.wav="));
    const auto nativeManifestBefore = migrateRoot.getChildFile("otto.ini").loadFileAsString();
    check("export_never_overwrites_native_manifest", !OttoMaterialMigration::exportFolder(
        migrateRoot, migrateRoot.getChildFile("otto.ini"), true, exported, error)
        && migrateRoot.getChildFile("otto.ini").loadFileAsString() == nativeManifestBefore);
    error.clear();
    std::vector<SampleRegionSetting> standardRows;
    check("standard_cutoff_roundtrip", SampleSettings::importOto(standardOto, migrateA, 1.0, standardRows, error)
        && standardRows.size() == 2 && near(standardRows[0].regionEndSeconds, .8)
        && near(standardRows[0].alignmentSeconds, .18));
    const auto malformedRoot = root.getChildFile("malformed-migration"); malformedRoot.createDirectory();
    const auto badAudio = malformedRoot.getChildFile("bad.wav"); tone(badAudio);
    SampleSettings::sidecarFor(badAudio).replaceWithText("name,region_start_sec,region_end_sec,note_alignment_sec,fixed_duration_sec,relative_pitch_cents\n"
        "bad,invalid,0.1,0,0,0\n");
    check("malformed_legacy_transaction_refused", !OttoMaterialMigration::migrateFolder(
        malformedRoot, UtauMode::classic, false, migration, error)
        && !malformedRoot.getChildFile("otto.ini").exists());
    SampleSettings::setMaterialRoots({});
    auto* report = new juce::DynamicObject();
    report->setProperty("passed", checks.size() - failed); report->setProperty("total", checks.size());
    report->setProperty("checks", juce::var(checks));
    report->setProperty("fixture", root.getFullPathName());
    root.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
    std::cout << "otto_passed=" << checks.size() - failed << "|total=" << checks.size()
              << "|fixture=" << root.getFullPathName() << std::endl;
    return failed == 0;
}
}
