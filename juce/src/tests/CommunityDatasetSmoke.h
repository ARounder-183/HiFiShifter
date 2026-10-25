#pragma once
#include "../CommunityDatasetIO.h"
#include "../MaterialPronunciationCatalog.h"
#include "../MaterialAnnotationSession.h"
#include "../AudioEngine.h"
#include "../NativeNoteTiming.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

namespace hachi
{
// Optional actual public fixture: the explicitly labelled SYNTHETIC SynthGT
// req-3943 subset described in validation/community-dataset-fixture/PROVENANCE.md.
// Unit-only invocation is NOT the public dataset/material/render acceptance.
inline bool runCommunityDatasetSmoke(const juce::File& output, const juce::File& publicFixture = {})
{
    using IO = CommunityDatasetIO;
    const auto base = output.getChildFile("community-dataset-" + juce::Uuid().toString());
    if (base.createDirectory().failed()) return false;
    bool ok = true; int count = 0; juce::String log, error;
    const auto check = [&](const char* name, bool pass)
    { ++count; ok &= pass; log += juce::String(name) + "=" + (pass ? "PASS" : "FAIL") + "\n"; };
    const auto finish = [&]()
    {
        std::cout << "community_dataset=" << ok << "|checks=" << count << "|public_fixture="
                  << (publicFixture != juce::File{}) << "|output=" << base.getFullPathName() << std::endl;
        if (!ok) std::cerr << log << std::endl;
        return base.getChildFile("verification.txt").replaceWithText(log) && ok;
    };
    const auto audioFixture = [](const juce::File& audio)
    {
        if (audio.getParentDirectory().createDirectory().failed()) return false;
        juce::AudioBuffer<float> pcm(1, 48000);
        for (int i = 0; i < pcm.getNumSamples(); ++i) pcm.setSample(0, i, .1f * static_cast<float>(std::sin(i * .03)));
        auto out = audio.createOutputStream(); juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out ? wav.createWriterFor(out.release(), 48000, 1, 16, {}, 0) : nullptr);
        return writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples());
    };
    const auto unit = base.getChildFile("unit"); unit.createDirectory();
    const auto first = unit.getChildFile("wavs/take,one.wav"), second = unit.getChildFile("wavs/second.wav");
    check("unit_audio", audioFixture(first) && audioFixture(second) && audioFixture(unit.getChildFile("unrelated.wav")));
    const auto csv = unit.getChildFile("transcriptions.csv"), dictionaryFile = unit.getChildFile("dictionary.txt");
    const juce::String header = "name,ph_seq,ph_dur,ph_num,note_seq,note_dur,note_glide,extra\r\n";
    const juce::String row1 = "\"take,one\",AP sh ir zh e,0.1 0.1 0.2 0.1 0.4,2 2 1,rest D#3 C4,0.2 0.3 0.4,none none none,\"quoted, \"\"field\"\"\nline\"\r\n";
    const juce::String row2 = "second,AP sh ir zh e,0.1 0.1 0.2 0.1 0.4,2 2 1,rest D#3 C4,0.2 0.3 0.4,none none none,plain\r\n";
    // File::replaceWithText defaults to CRLF rewriting, INCLUDING quoted CSV
    // payloads. Preserve this intentional mixed CRLF-record/LF-payload fixture.
    check("unit_csv_and_declared_dictionary", csv.replaceWithText(header + row1 + row2, false, false, nullptr)
        && dictionaryFile.replaceWithText("shi\tsh ir\nshi_alt\tsh ir\nzhe\tzh e\n"));
    const auto csvHash = juce::SHA256(csv).toHexString(), firstHash = juce::SHA256(first).toHexString();
    IO::Dataset dataset; IO::Dictionary dictionary; IO::Report report;
    check("strict_RFC4180_optional_metadata", IO::read(unit, dataset, report) && dataset.recordings.size() == 2
        && dataset.recordings.front().extraFields["extra"] == "quoted, \"field\"\nline"
        && dataset.recordings.front().wordPhoneCounts == std::vector<int>({2, 2, 1})
        && dataset.recordings.front().notes.size() == 3 && !report.warnings.isEmpty());
    IO::Dataset crlfPayload;
    check("quoted_CRLF_payload_preserved_not_normalized", csv.replaceWithText(header + row1.replace("\nline", "\r\nline") + row2, false, false, nullptr)
        && IO::read(unit, crlfPayload, report) && crlfPayload.recordings.size() == 2
        && crlfPayload.recordings.front().extraFields["extra"] == "quoted, \"field\"\r\nline");
    check("restore_mixed_newline_fixture_bytes", csv.replaceWithText(header + row1 + row2, false, false, nullptr)
        && juce::SHA256(csv).toHexString() == csvHash);
    check("declared_two_part_dictionary", IO::readTwoPartDictionary(dictionaryFile, dictionary, report) && dictionary.rules.size() == 3);
    if (dataset.recordings.size() != 2 || dictionary.rules.size() != 3) return finish();
    IO::Options options; IO::Plan plan;
    check("missing_P_C_confirmation_rejects", !IO::planImport(dataset, dictionary, {}, options, plan, report) && !plan.ready);
    options.confirmNucleusAsAlignmentAndConsonant = true;
    check("ambiguous_alias_no_default", !IO::planImport(dataset, dictionary, {}, options, plan, report));
    options.primaryAliases["sh ir"] = "shi";
    check("SP_AP_requires_explicit_policy", !IO::planImport(dataset, dictionary, {}, options, plan, report));
    options.specials = IO::Specials::skipWithReport;
    check("unique_dictionary_plan_no_writes", IO::planImport(dataset, dictionary, {}, options, plan, report)
        && plan.audio.size() == 2 && report.pronunciations == 4 && report.skippedSpecialPhones == 2
        && !plan.manifest.exists() && juce::SHA256(csv).toHexString() == csvHash && juce::SHA256(first).toHexString() == firstHash);
    if (!plan.ready) return finish();
    const auto plannedFirstId = plan.audio.front().rows.front().materialRegionId;
    check("same_alias_different_audio_identity_separate", plan.audio.front().rows.front().name == plan.audio.back().rows.front().name
        && plan.audio.front().rows.front().materialAudioId != plan.audio.back().rows.front().materialAudioId
        && plan.audio.front().rows.front().materialRegionId != plan.audio.back().rows.front().materialRegionId);
    const auto& shi = plan.audio.front().rows.front();
    check("ph_num_V_leading_is_NOT_CV_grouping", shi.name == "shi" && std::abs(shi.regionStartSeconds - .1) < 1.e-9
        && std::abs(shi.alignmentSeconds - .2) < 1.e-9 && std::abs(shi.regionEndSeconds - .4) < 1.e-9
        && shi.additionalAliases.contains("shi_alt") && shi.segments.size() == 2 && !shi.vowelSplitsExplicit);
    check("cancel_before_publication_is_read_only", !IO::publish(plan, report, [] { return true; }) && report.cancelled && !plan.manifest.exists());
    csv.appendText("\r\n");
    check("stale_CSV_prevents_publish", !IO::publish(plan, report) && !plan.manifest.exists());
    check("restore_exact_unit_CSV", csv.replaceWithText(header + row1 + row2, false, false, nullptr) && juce::SHA256(csv).toHexString() == csvHash);
    check("atomic_folder_publish", IO::publish(plan, report) && report.published && plan.manifest.existsAsFile());
    check("plan_cannot_double_publish", !IO::publish(plan, report));
    IO::Plan denied;
    check("existing_annotation_default_reject", !IO::planImport(dataset, dictionary, {}, options, denied, report));
    options.replacePreviouslyImportedAudio = true;
    check("explicit_reimport_stable_identity", IO::planImport(dataset, dictionary, {}, options, denied, report)
        && denied.audio.front().rows.front().materialRegionId == plannedFirstId);
    // Existing unrelated author rows are preserved by the existing codec merge.
    OttoMaterialIO::Document doc; check("load_published_document", OttoMaterialIO::loadDocument(plan.manifest, doc, error));
    SampleRegionSetting unrelated; unrelated.name = "manual"; unrelated.materialAudioId = "manual-a";
    unrelated.materialRegionId = "manual-p"; unrelated.regionStartSeconds = 0; unrelated.regionEndSeconds = .5;
    unrelated.alignmentSeconds = .1; unrelated.fixedDurationSeconds = .1; unrelated.provenance = "user";
    check("unrelated_manual_fixture", OttoMaterialIO::replaceAudio(doc, plan.manifest, unit.getChildFile("unrelated.wav"), {unrelated}, error, unrelated.materialAudioId)
        && OttoMaterialIO::saveDocument(plan.manifest, doc, error));
    auto sibling = unrelated; sibling.materialAudioId = "manual-sibling"; sibling.materialRegionId = "manual-sibling-p";
    sibling.name = "foreign-same-recording";
    check("foreign_same_recording_variant_fixture", OttoMaterialIO::replaceAudio(doc, plan.manifest, first, {sibling}, error, sibling.materialAudioId)
        && OttoMaterialIO::saveDocument(plan.manifest, doc, error));
    check("stale_manifest_guard_owned_by_codec", !IO::publish(denied, report));
    check("fresh_merge_preserves_unrelated", IO::planImport(dataset, dictionary, {}, options, denied, report)
        && denied.document.audio.size() == 4 && IO::publish(denied, report));
    std::vector<SampleRegionSetting> siblingRows;
    check("reimport_targets_only_dataset_registration", OttoMaterialIO::loadAudio(plan.manifest, first, siblingRows, error, nullptr, "manual-sibling")
        && siblingRows.size() == 1 && siblingRows.front().name == "foreign-same-recording"
        && siblingRows.front().materialRegionId == "manual-sibling-p");
    IO::Dataset reordered;
    csv.replaceWithText(header + row2 + row1, false, false, nullptr);
    check("reordered_CSV_read", IO::read(unit, reordered, report));
    IO::Plan reorderedPlan;
    bool stable = IO::planImport(reordered, dictionary, {}, options, reorderedPlan, report);
    if (stable)
    {
        const auto found = std::find_if(reorderedPlan.audio.begin(), reorderedPlan.audio.end(), [&](const auto& a) { return a.audio == first; });
        stable = found != reorderedPlan.audio.end() && found->rows.front().materialRegionId == plannedFirstId;
    }
    check("CSV_order_not_pronunciation_identity", stable);
    check("cancelled_reader_transactional", !IO::read(unit, reordered, report, [] { return true; }) && report.cancelled);
    const auto siblingRoot = base.getChildFile("new-sibling-import"); siblingRoot.getChildFile("wavs").createDirectory();
    check("isolated_variant_source_fixture", csv.copyFileTo(siblingRoot.getChildFile("transcriptions.csv"))
        && first.copyFileTo(siblingRoot.getChildFile("wavs/take,one.wav"))
        && second.copyFileTo(siblingRoot.getChildFile("wavs/second.wav")));
    OttoMaterialIO::Document onlyForeign; const auto siblingManifest = siblingRoot.getChildFile("otto.ini");
    check("foreign_only_registration_fixture", OttoMaterialIO::replaceAudio(onlyForeign, siblingManifest,
        siblingRoot.getChildFile("wavs/take,one.wav"), {sibling}, error, sibling.materialAudioId)
        && OttoMaterialIO::saveDocument(siblingManifest, onlyForeign, error));
    IO::Dataset siblingDataset; IO::Plan siblingPlan; auto siblingOptions = options;
    siblingOptions.replacePreviouslyImportedAudio = false;
    check("new_variant_requires_explicit_permission", IO::read(siblingRoot, siblingDataset, report)
        && !IO::planImport(siblingDataset, dictionary, {}, siblingOptions, siblingPlan, report)
        && report.errors.joinIntoString(" ").contains("separate dataset variant"));
    siblingOptions.allowSiblingAnnotationVariant = true;
    check("explicit_dataset_sibling_appends_without_PCM_copy", IO::planImport(siblingDataset, dictionary, {}, siblingOptions, siblingPlan, report)
        && siblingPlan.document.audio.size() == 3 && IO::publish(siblingPlan, report)
        && juce::SHA256(siblingRoot.getChildFile("wavs/take,one.wav")).toHexString() == firstHash);
    siblingRows.clear();
    check("appended_sibling_keeps_original_registration", OttoMaterialIO::loadAudio(siblingManifest,
        siblingRoot.getChildFile("wavs/take,one.wav"), siblingRows, error, nullptr, sibling.materialAudioId)
        && siblingRows.size() == 1 && siblingRows.front().name == "foreign-same-recording");
    auto ambiguous = dictionary;
    ambiguous.rules.push_back({"sh", {"sh"}, 0}); ambiguous.rules.push_back({"ir", {"ir"}, 0});
    check("multiple_segmentations_require_manual_spans", !IO::planImport(reordered, ambiguous, {}, options, denied, report)
        && report.errors.joinIntoString(" ").contains("No unique syllable segmentation"));
    const auto bad = base.getChildFile("bad"); bad.createDirectory();
    bad.getChildFile("transcriptions.csv").replaceWithText("name,ph_seq,ph_dur\n../escape,a,1\n");
    check("path_traversal_rejected", !IO::read(bad, reordered, report));
    bad.getChildFile("transcriptions.csv").replaceWithText("name,ph_seq,ph_dur\n\"unterminated,a,1\n");
    check("malformed_quote_rejected", !IO::read(bad, reordered, report));
    bad.getChildFile("dictionary.txt").replaceWithText("poly\tk a t\n");
    check("polysyllabic_dictionary_not_guessed", !IO::readTwoPartDictionary(bad.getChildFile("dictionary.txt"), ambiguous, report));
    check("original_unit_audio_unchanged", juce::SHA256(first).toHexString() == firstHash);

    if (publicFixture == juce::File{})
    {
        log += "PUBLIC_DATASET_E2E=NOT_RUN (unit tests only; no public fixture argument)\n";
        return finish();
    }
    log += "FIXTURE_BASIS=SYNTHETIC public SynthGT req-3943, CC BY-NC-SA 4.0, Silas Antonisen and Ivan Lopez-Espejo; NOT human singing or Tokyo/teio oracle\n";
    log += "CSV=https://huggingface.co/datasets/Silasimo/SynthGT/raw/main/splits/test/%E6%88%AF%E7%99%BD%E3%83%A1%E3%83%AA%E3%83%BC%20DiffSinger%20V2.0/transcriptions.csv\n";
    log += "AUDIO=https://huggingface.co/datasets/Silasimo/SynthGT/resolve/main/waveforms/req-3943.wav\n";
    log += "SYLLABLES=https://huggingface.co/datasets/Silasimo/SynthGT/raw/main/metadata/syllable.jsonl\n";
    log += "MAPPING=explicit upstream syllable spans; metadata dh_ax alias mapped to CSV dh ah; nucleus offsets manually confirmed, NOT implicit spelling normalization or ASR\n";
    const auto publicCsv = publicFixture.getChildFile("transcriptions.csv"), publicAudio = publicFixture.getChildFile("wavs/req-3943.wav");
    const auto externalCsvHash = juce::SHA256(publicCsv).toHexString(), externalAudioHash = juce::SHA256(publicAudio).toHexString();
    check("known_public_waveform_provenance_hash", externalAudioHash == "1db5d8dead9be7c664b3b949cebf8bb7aeb0e6c9b0ea997e051b8487d5fd5175");
    const auto imported = base.getChildFile("public-import"); imported.getChildFile("wavs").createDirectory();
    check("isolated_public_fixture_copy", publicCsv.copyFileTo(imported.getChildFile("transcriptions.csv"))
        && publicAudio.copyFileTo(imported.getChildFile("wavs/req-3943.wav")));
    IO::Dataset actual; IO::Dictionary manual; IO::Plan actualPlan;
    check("actual_community_CSV_read", IO::read(imported, actual, report) && actual.recordings.size() == 1
        && actual.recordings.front().name == "req-3943" && actual.recordings.front().phones.size() == 24);
    if (actual.recordings.size() != 1) { log += report.errors.joinIntoString("\n"); return finish(); }
    const std::vector<IO::Selection> chosen{
        {"req-3943", "y_uw", 1, 2, 1, {}}, {"req-3943", "s_p_ih_n", 3, 4, 2, {}},
        {"req-3943", "dh_ax", 7, 2, 1, {}}, {"req-3943", "r_iy_l", 9, 3, 1, {}},
        {"req-3943", "y_uw", 12, 2, 1, {}}, {"req-3943", "p_ah_l_s", 14, 4, 1, {}},
        {"req-3943", "dh_ax", 18, 2, 1, {}}, {"req-3943", "f_iy_l", 20, 3, 1, {}}
    };
    IO::Options actualOptions; actualOptions.confirmNucleusAsAlignmentAndConsonant = true;
    actualOptions.specials = IO::Specials::importDeclaredSPAP;
    check("actual_explicit_mapping_OTTO_publish", IO::planImport(actual, manual, chosen, actualOptions, actualPlan, report)
        && actualPlan.audio.size() == 1 && actualPlan.audio.front().rows.size() == 10 && IO::publish(actualPlan, report));
    if (!actualPlan.published) { log += report.errors.joinIntoString("\n"); return finish(); }
    const auto source = actual.recordings.front().audio;
    auto reopened = MaterialAnnotationSession::open(source, actualPlan.manifest, error);
    check("actual_shared_editor_session_reopen_and_save", reopened && reopened->rows().size() == 10 && reopened->save(error));
    if (!reopened) return finish();
    const auto catalog = MaterialPronunciationCatalog::index({{imported}});
    std::set<juce::String> ids; int repeatedYou = 0; bool resolvedAll = true;
    for (const auto& entry : catalog.entries)
    {
        ids.insert(entry.regionId); repeatedYou += entry.alias == "y_uw";
        MaterialPronunciationCatalog::Entry resolved;
        resolvedAll &= MaterialPronunciationCatalog::resolve(entry, resolved, error);
    }
    check("actual_catalog_repeated_aliases_stable_source_IDs", catalog.ok() && catalog.entries.size() == 10
        && ids.size() == 10 && repeatedYou == 2 && resolvedAll);
    ProjectData seed; TrackData track; track.id = "community-native"; track.name = "Synthetic public dataset import";
    track.pitchAlgorithm = PitchAlgorithm::nsfHifigan; track.nativeNsfAudio = true; track.compose = true;
    track.normalizeVolume = false; track.smoothOverlaps = false; track.nsfSmoothPitchTransitions = false;
    track.volume = 1; track.pan = 0; seed.tracks = {track};
    ProjectModel model; model.resetDocument(seed);
    std::vector<juce::String> selectedIds; for (const auto& row : reopened->rows()) selectedIds.push_back(row.materialRegionId);
    const auto clipId = model.addMaterialPronunciations(source, reopened->rows(), selectedIds, 0, track.id, error);
    const auto before = model.snapshot();
    check("actual_material_placement_single_transaction", clipId.isNotEmpty() && before.tracks.front().clips.size() == 1
        && before.tracks.front().clips.front().notes.size() == 10);
    if (clipId.isEmpty()) { log += error; return finish(); }
    bool unmeasured = true;
    for (const auto& note : before.tracks.front().clips.front().notes) unmeasured &= !note.sourcePitchMeasured;
    check("dataset_note_metadata_not_fake_measured_F0", unmeasured);
    const auto projectFile = base.getChildFile("community-import.hachi");
    ProjectModel loaded;
    check("actual_project_save_reopen", model.save(projectFile, error) && loaded.load(projectFile, error)
        && loaded.snapshot().tracks.front().clips.front().notes.size() == 10);
    const auto readPcm = [](const juce::File& file, juce::AudioBuffer<float>& pcm, double& rate)
    {
        juce::AudioFormatManager formats; formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if (!reader || reader->lengthInSamples <= 0 || reader->lengthInSamples > 10000000) return false;
        rate = reader->sampleRate; pcm.setSize(static_cast<int>(reader->numChannels), static_cast<int>(reader->lengthInSamples));
        return reader->read(&pcm, 0, pcm.getNumSamples(), 0, true, true);
    };
    juce::AudioBuffer<float> original; double sourceRate = 0;
    check("actual_public_audio_test_decode", readPcm(source, original, sourceRate));
    if (sourceRate <= 0) return finish();
    WavExportOptions wavOptions; wavOptions.sampleRate = static_cast<int>(sourceRate);
    wavOptions.channels = original.getNumChannels(); wavOptions.bitDepth = 32;
    const auto duration = actual.recordings.front().phones.back().end;
    const auto render = [&](const ProjectData& project, const juce::File& file, const char* stage)
    {
        AudioEngine engine(false); engine.prepareToPlay(512, sourceRate);
        engine.setHifiganModelDirectory(base.getChildFile("intentionally-missing-model"));
        engine.syncProject(project);
        const auto deadline = juce::Time::getMillisecondCounterHiRes() + 30000;
        while (engine.renderProgress() && juce::Time::getMillisecondCounterHiRes() < deadline) juce::Thread::sleep(5);
        log += juce::String(stage) + " backend=" + engine.activeRenderBackends() + " warnings=" + engine.activeRenderWarnings() + "\n";
        return !engine.renderProgress() && engine.hasCurrentRenderedAudio() && engine.hasPlayableRenderedAudio()
            && engine.activeRenderBackends() == "native-source-preserved" && engine.activeRenderWarnings().isEmpty()
            && engine.devices().getCurrentAudioDevice() == nullptr
            && engine.exportWav(file, error, track.id, 0, duration, wavOptions);
    };
    const auto firstRender = base.getChildFile("import-render.wav"), coldRender = base.getChildFile("reopened-render.wav");
    check("actual_public_dataset_engine_render", render(before, firstRender, "initial"));
    check("actual_public_dataset_cold_reopen_render", render(loaded.snapshot(), coldRender, "reopened"));
    juce::AudioBuffer<float> rendered, cold; double renderedRate = 0, coldRate = 0;
    bool exact = readPcm(firstRender, rendered, renderedRate) && readPcm(coldRender, cold, coldRate)
        && renderedRate == sourceRate && coldRate == sourceRate && rendered.getNumSamples() > 0
        && rendered.getNumSamples() == cold.getNumSamples() && rendered.getNumChannels() == cold.getNumChannels()
        && rendered.getNumChannels() == original.getNumChannels()
        && std::abs(rendered.getNumSamples() - static_cast<int>(std::llround(duration * sourceRate))) <= 1
        && rendered.getNumSamples() <= original.getNumSamples() + 1;
    const auto& placedClip = before.tracks.front().clips.front();
    const auto sourceClock = nativeClipClock(placedClip);
    // The importer has NOT changed source PCM. A centred MONO track uses the
    // existing equal-power pan law (AudioEngine::panGains), then mono WAV export
    // averages L/R. Therefore expected output is source * sqrt(.5), NOT raw
    // unity amplitude. These are declared track/export facts, never a fitted
    // gain/lag or a change to the production engine/acceptance threshold.
    const float monoLeft = std::sqrt(.5f * (1.0f - track.pan));
    const float monoRight = std::sqrt(.5f * (1.0f + track.pan));
    const bool neutralPlacement = placedClip.parts.empty() && placedClip.sourceFile == source
        && placedClip.sourceOffsetSeconds == 0 && placedClip.startSeconds == 0
        && placedClip.gain == 1 && track.volume == 1 && track.pan == 0
        && placedClip.fadeInSeconds == 0 && placedClip.fadeOutSeconds == 0
        && sourceClock.size() == 2;
    double peak = 0, sourceError = 0, sourceGridError = 0;
    bool sourceCoverage = neutralPlacement;
    if (exact && sourceCoverage)
        for (int sample = 0; sample < rendered.getNumSamples(); ++sample)
        {
            const auto targetSeconds = static_cast<double>(sample) / sourceRate - placedClip.startSeconds;
            const auto sourcePosition = (placedClip.sourceOffsetSeconds + nativeSourceTimeAt(sourceClock, targetSeconds)) * sourceRate;
            const auto sourceSample = static_cast<juce::int64>(std::llround(sourcePosition));
            sourceGridError = std::max(sourceGridError, std::abs(sourcePosition - static_cast<double>(sourceSample)));
            // ceil(export duration * rate) may add one exact zero at file end;
            // it is checked explicitly, not omitted from the PCM oracle.
            sourceCoverage &= sourceSample >= 0 && sourceSample <= original.getNumSamples();
            for (int channel = 0; channel < rendered.getNumChannels(); ++channel)
            {
                exact &= rendered.getSample(channel, sample) == cold.getSample(channel, sample);
                peak = std::max(peak, std::abs(static_cast<double>(rendered.getSample(channel, sample))));
                float expected = sourceSample >= 0 && sourceSample < original.getNumSamples()
                    ? original.getSample(channel, static_cast<int>(sourceSample)) : 0.0f;
                if (original.getNumChannels() == 1)
                    expected = .5f * (expected * monoLeft) + .5f * (expected * monoRight);
                sourceError = std::max(sourceError, std::abs(static_cast<double>(rendered.getSample(channel, sample) - expected)));
            }
        }
    check("actual_cold_reopen_PCM_exact_non_silent", exact && peak > .01);
    check("actual_authoritative_source_clock_coverage", sourceCoverage && sourceGridError < 1.e-5);
    check("actual_source_preserved_PCM_declared_pan_downmix", exact && sourceCoverage && sourceGridError < 1.e-5 && sourceError < 1.e-6);
    log += "peak=" + juce::String(peak, 9) + " source_peak_error_declared_mix=" + juce::String(sourceError, 12)
        + " declared_mono_pan_downmix=" + juce::String(.5f * monoLeft + .5f * monoRight, 9)
        + " source_grid_error_samples=" + juce::String(sourceGridError, 12)
        + " source_offset_seconds=" + juce::String(placedClip.sourceOffsetSeconds, 12) + "\n";
    check("external_public_fixture_unchanged", juce::SHA256(publicCsv).toHexString() == externalCsvHash
        && juce::SHA256(publicAudio).toHexString() == externalAudioHash
        && juce::SHA256(source).toHexString() == externalAudioHash);
    return finish();
}
}
