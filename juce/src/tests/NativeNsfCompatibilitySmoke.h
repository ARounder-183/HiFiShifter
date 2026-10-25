#pragma once
#include "../AudioEngine.h"
#include "../ProjectModel.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runNativeNsfCompatibilitySmoke(const juce::File& output)
{
    if (!output.createDirectory()) return false;
    bool ok = true; int checks = 0; juce::Array<juce::var> results;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass; ++checks; std::cout << name << '=' << pass << std::endl;
        auto row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("pass", pass);
        results.add(juce::var(row));
    };
    const auto readTree = [](const juce::File& file)
    {
        juce::MemoryBlock bytes;
        return file.loadFileAsData(bytes) ? juce::ValueTree::readFromData(bytes.getData(), bytes.getSize()) : juce::ValueTree{};
    };
    const auto writeTree = [](const juce::ValueTree& tree, const juce::File& file)
    {
        juce::MemoryOutputStream bytes; tree.writeToStream(bytes);
        return file.replaceWithData(bytes.getData(), bytes.getDataSize());
    };
    const auto trackTree = [](const juce::ValueTree& root)
    { return root.getChildWithName("Track"); };
    TrackData track; track.id = "nsf-compatibility"; track.name = "Native";
    track.compose = true; track.pitchAlgorithm = PitchAlgorithm::nsfHifigan; track.nativeNsfAudio = true;
    check("new_tracks_keep_noise_protection_enabled", track.nsfNoiseProtection);
    check("new_tracks_keep_pitch_conditioner_enabled", track.nsfSmoothPitchTransitions);
    for (int i = 0; i < 2; ++i)
    {
        ClipData clip; clip.id = "part-" + juce::String(i); clip.startSeconds = i * .1;
        clip.durationSeconds = clip.sourceDurationSeconds = .1; clip.sourceOffsetSeconds = i * .1;
        clip.sourceFile = output.getChildFile("source.wav");
        NoteData note; note.id = "note-" + juce::String(i); note.label = "a"; note.durationSeconds = .1;
        note.midiNote = note.sourceMidiCenter = 57; note.sourcePitchMeasured = true;
        note.contour = {{0, 0, 0, i == 0}, {.1, 0, 0, i == 0}};
        clip.notes.push_back(note); track.clips.push_back(clip);
    }
    ProjectData data; data.tracks = {track}; ProjectModel current; current.resetDocument(data);
    juce::String error; const auto newFile = output.getChildFile("new-explicit.hjpx");
    check("new_project_fixture_saves", current.save(newFile, error));
    const auto newTree = readTree(newFile); const auto newTrack = trackTree(newTree);
    check("new_project_persists_explicit_noise_setting", newTrack.isValid()
        && newTrack.hasProperty("nsfNoiseProtection") && static_cast<bool>(newTrack.getProperty("nsfNoiseProtection")));
    check("project_format_version_stays_19", static_cast<int>(newTree.getProperty("version")) == 19);
    auto legacyTree = newTree.createCopy(); auto legacyTrack = trackTree(legacyTree);
    legacyTrack.removeProperty("nsfNoiseProtection", nullptr);
    legacyTrack.removeProperty("nsfSmoothPitchTransitions", nullptr);
    const auto legacyFile = output.getChildFile("missing-options.hjpx");
    check("missing_option_legacy_fixture_written", writeTree(legacyTree, legacyFile));
    ProjectModel old;
    const bool oldLoaded = old.load(legacyFile, error) && old.snapshot().tracks.size() == 1;
    check("missing_option_legacy_fixture_loads", oldLoaded);
    if (oldLoaded)
    {
        const auto loaded = old.snapshot();
        check("missing_noise_flag_defaults_to_legacy_false", !loaded.tracks[0].nsfNoiseProtection);
        check("missing_smoothing_flag_retains_old_enabled_conditioner", loaded.tracks[0].nsfSmoothPitchTransitions);
        const auto oldRevision = old.revisionNumber(); const auto fingerprint = old.contentFingerprint();
        const auto migratedFile = output.getChildFile("legacy-resaved.hjpx");
        check("legacy_false_setting_resaves", old.save(migratedFile, error));
        const auto migrated = trackTree(readTree(migratedFile));
        check("legacy_resave_materializes_false_without_changing_version", migrated.hasProperty("nsfNoiseProtection")
            && !static_cast<bool>(migrated.getProperty("nsfNoiseProtection"))
            && static_cast<int>(readTree(migratedFile).getProperty("version")) == 19);
        ProjectModel reopened;
        check("legacy_false_survives_save_reload", reopened.load(migratedFile, error)
            && reopened.snapshot().tracks.size() == 1 && !reopened.snapshot().tracks[0].nsfNoiseProtection
            && reopened.snapshot().tracks[0].nsfSmoothPitchTransitions);
        check("compatibility_save_does_not_mutate_project_or_revision", old.contentFingerprint() == fingerprint && old.revisionNumber() == oldRevision);
    }
    auto explicitTree = legacyTree.createCopy();
    trackTree(explicitTree).setProperty("nsfNoiseProtection", true, nullptr);
    const auto enabledFile = output.getChildFile("explicit-noise-on.hjpx");
    ProjectModel enabled;
    check("explicit_noise_true_is_not_migrated_off", writeTree(explicitTree, enabledFile)
        && enabled.load(enabledFile, error) && enabled.snapshot().tracks.size() == 1 && enabled.snapshot().tracks[0].nsfNoiseProtection);
    trackTree(explicitTree).setProperty("nsfSmoothPitchTransitions", false, nullptr);
    const auto smoothingOffFile = output.getChildFile("explicit-smoothing-off.hjpx");
    ProjectModel disabledSmoothing;
    check("explicit_smoothing_false_remains_false", writeTree(explicitTree, smoothingOffFile)
        && disabledSmoothing.load(smoothingOffFile, error) && disabledSmoothing.snapshot().tracks.size() == 1
        && !disabledSmoothing.snapshot().tracks[0].nsfSmoothPitchTransitions);

    const auto merged = [](const TrackData& value, std::optional<bool> legacyCarrier = {})
    {
        std::vector<const ClipData*> group;
        for (const auto& clip : value.clips) group.push_back(&clip);
        return AudioEngine::mergedRequestFor(group, value, {}, {}, legacyCarrier);
    };
    constexpr std::size_t uvFrame = 25; // .125s, inside the second clip's UV span.
    const auto pitchAt = [](const auto& request, std::size_t frame)
    { return frame < request.targetMidi.size() ? request.targetMidi[frame] : -1.f; };
    const auto newRequest = merged(track);
    check("new_protected_merged_requests_keep_zero_uv_mask", pitchAt(newRequest, uvFrame) == 0 && newRequest.nsfNoiseProtection && newRequest.nsfIncrementalRender);
    auto legacy = track; legacy.nsfNoiseProtection = false;
    const auto legacyRequest = merged(legacy);
    check("unprotected_merged_requests_derive_old_carrier", std::abs(pitchAt(legacyRequest, uvFrame) - 57.f) < 1.e-6
        && legacyRequest.sourceMidi.size() > uvFrame && std::abs(legacyRequest.sourceMidi[uvFrame] - 57.f) < 1.e-6);
    check("unprotected_merged_requests_select_legacy_4096_decoder", !legacyRequest.nsfIncrementalRender && !legacyRequest.nsfNoiseProtection
        && legacyRequest.nsfSmoothPitchTransitions);
    const auto forcedMask = merged(legacy, false);
    check("explicit_false_diagnostic_override_still_disables_carrier", pitchAt(forcedMask, uvFrame) == 0);
    const auto forcedCarrier = merged(track, true);
    check("explicit_true_diagnostic_override_still_enables_carrier", std::abs(pitchAt(forcedCarrier, uvFrame) - 57.f) < 1.e-6);
    const auto explicitNew = merged(track, false);
    check("new_project_default_request_matches_previous_explicit_mask_policy", newRequest.targetMidi == explicitNew.targetMidi
        && newRequest.sourceMidi == explicitNew.sourceMidi && newRequest.noteGain == explicitNew.noteGain
        && newRequest.nsfIncrementalRender == explicitNew.nsfIncrementalRender);
    auto unpitched = legacy; unpitched.clips[1].notes[0].nativeUnpitched = true;
    const auto silent = merged(unpitched);
    check("explicit_native_unpitched_never_receives_legacy_carrier", pitchAt(silent, uvFrame) == 0
        && silent.sourceMidi.size() > uvFrame && silent.sourceMidi[uvFrame] == 0);
    const auto forcedSilent = merged(unpitched, true);
    check("even_forced_carrier_preserves_native_unpitched_zero", pitchAt(forcedSilent, uvFrame) == 0);
    const auto singleLegacy = AudioEngine::diagnosticNativeRequest(legacy.clips[1], legacy);
    check("legacy_single_clip_keeps_own_uv_mask_and_old_decoder", !singleLegacy.nsfIncrementalRender
        && std::all_of(singleLegacy.targetMidi.begin(), singleLegacy.targetMidi.end(), [](float value) { return value == 0; }));
    const auto singleNew = AudioEngine::diagnosticNativeRequest(track.clips[1], track);
    check("new_single_clip_keeps_current_incremental_decoder", singleNew.nsfIncrementalRender && singleNew.nsfNoiseProtection);
    auto report = new juce::DynamicObject(); report->setProperty("ok", ok); report->setProperty("checks", checks);
    report->setProperty("results", juce::var(results));
    const bool written = output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout << "native_nsf_compatibility_ok=" << (ok && written) << "; checks=" << checks << std::endl;
    return ok && written;
}
}
