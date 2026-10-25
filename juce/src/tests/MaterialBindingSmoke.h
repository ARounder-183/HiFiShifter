#pragma once
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "../NativeNoteTiming.h"
#include <iostream>

namespace hachi
{
// No AudioEngine, model pack, GUI component, or audio device is required.
inline bool runMaterialBindingSmoke(const juce::File& output)
{
    if (output.createDirectory().failed()) return false;
    bool ok = true;
    juce::String log;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass;
        log += juce::String(name) + "=" + (pass ? "PASS" : "FAIL") + "\n";
        std::cout << name << '=' << pass << std::endl;
    };
    const auto near = [](double a, double b) { return std::abs(a-b) < 1.e-7; };
    const auto source = output.getChildFile("binding-source.wav");
    juce::AudioBuffer<float> pcm(1, 144000); pcm.clear();
    juce::WavAudioFormat wav;
    auto stream = source.createOutputStream();
    if (!stream) return false;
    stream->setPosition(0); stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), 48000, 1, 16, {}, 0));
    if (!writer) return false;
    stream.release();
    check("wav_fixture", writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())); writer.reset();

    SampleRegionSetting a, b;
    a.name = "a"; a.regionStartSeconds = 0; a.regionEndSeconds = 1;
    a.materialAudioId = "audio-one"; a.materialRegionId = "region-a";
    a.role = NativeSegmentRole::vowel; a.provenance = "user";
    a.connectedNextRegionId = "region-b"; a.additionalAliases.add("alias-a");
    b = a; b.name = "b"; b.regionStartSeconds = 1; b.regionEndSeconds = 2;
    b.materialRegionId = "region-b"; b.connectedNextRegionId.clear(); b.alignmentSeconds = 1;
    const auto makeNote = [&](const SampleRegionSetting& row, const char* id)
    {
        NoteData n; n.id = id; n.label = row.name;
        n.materialAudioId = row.materialAudioId; n.materialRegionId = row.materialRegionId;
        n.materialSourceStartSeconds = n.nativeSourceStartSeconds = row.regionStartSeconds;
        n.materialSourceEndSeconds = n.nativeSourceEndSeconds = row.regionEndSeconds;
        n.materialAdditionalAliases = row.additionalAliases;
        n.materialConnectedNextRegionId = row.connectedNextRegionId;
        n.nativeRole = row.role; n.nativeProvenance = row.provenance;
        n.nativeSegments = SampleSettings::nativeSegmentsFor(row);
        n.startSeconds = row.regionStartSeconds; n.durationSeconds = 1;
        n.midiNote = 65; n.sourceMidiCenter = 60; n.gain = .37f;
        n.pitchControlPoints = {{0, 65}, {.45, 67}, {1, 64}};
        n.amplitudeEnvelope = {{0, -8}, {.3, -2}, {1, -9}};
        n.contour = {{0, 12, 2, true}, {1, 20, 3, true}};
        n.vibratoEnabled = true; n.vibratoDepthCents = 71;
        n.consonantSeconds = .09; n.attackSpeed = 1.3f;
        return n;
    };
    ClipData clip; clip.id = "placement-one"; clip.sourceFile = source;
    clip.startSeconds = 1; clip.durationSeconds = clip.sourceDurationSeconds = 2;
    clip.notes = {makeNote(a, "a-one"), makeNote(b, "b-one")};
    clip.sourceTimeMap = {{0, 0}, {.4, .2}, {1, 1}, {1.6, 1.7}, {2, 2}};
    auto repeat = clip; repeat.id = "placement-two"; repeat.startSeconds = 5;
    repeat.notes[0].id = "a-two"; repeat.notes[1].id = "b-two";
    auto local = clip; local.id = "local"; local.startSeconds = 9;
    local.notes[0].id = "local-a"; local.notes[1].id = "local-b";
    for (auto& n : local.notes) n.materialLocalOverride = true;
    TrackData track; track.id = "native"; track.pitchAlgorithm = PitchAlgorithm::world;
    track.clips = {clip, repeat, local};
    ProjectData fixture; fixture.tracks = {track};
    fixture.nativeConnections.push_back({"connection", "a-one", "b-one", "pitch-and-amplitude", 2, {}, {}});
    ProjectModel model; model.resetDocument(fixture);
    const auto revision = model.revisionNumber();
    auto result = model.applySourceSettingsWithReport(source, {b, a});
    check("reordered_noop_no_revision_or_undo", !result.rejected && !result.changed()
        && model.revisionNumber() == revision && !model.canUndo());
    a.name = "renamed-a"; a.additionalAliases.add("new-alias");
    b.name = "renamed-b"; b.pronunciationKind = PronunciationKind::transition;
    a.melodyneAmplitude = 3; a.relativePitchCents = 1200; a.amplitudeEnvelope = {{0, 0}, {1, 0}};
    result = model.applySourceSettingsWithReport(source, {b, a});
    auto data = model.snapshot();
    check("all_repeated_ids_not_row_order", !result.rejected && result.updatedNotes == 4
        && data.tracks[0].clips[0].notes[0].label == a.name
        && data.tracks[0].clips[1].notes[1].label == b.name);
    const auto preserved = [&](const NoteData& n)
    {
        return n.midiNote == 65 && n.gain == .37f && n.pitchControlPoints.size() == 3
            && n.pitchControlPoints[1].targetMidi == 67 && near(n.pitchControlPoints[1].timeSeconds, .45)
            && n.amplitudeEnvelope.size() == 3 && n.amplitudeEnvelope[1].gainDb == -2
            && n.contour.size() == 2 && n.contour[0].relativeCents == 12
            && n.vibratoEnabled && near(n.vibratoDepthCents, 71)
            && near(n.consonantSeconds, .09) && n.attackSpeed == 1.3f;
    };
    check("target_controls_preserved", preserved(data.tracks[0].clips[0].notes[0]));
    check("local_overrides_untouched", result.skippedLocalOverrides == 2
        && data.tracks[0].clips[2].notes[0].label == "a"
        && data.tracks[0].clips[2].sourceTimeMap[2].sourceSeconds == 1);
    model.resetDocument(data);
    a.regionEndSeconds = .8; b.regionStartSeconds = .8;
    const auto previewFingerprint = model.contentFingerprint();
    const auto previewRevision = model.revisionNumber();
    result = model.applySourceSettingsWithReport(source, {b, a}, false);
    check("publication_preflight_is_non_mutating", !result.rejected && result.changed()
        && model.contentFingerprint() == previewFingerprint && model.revisionNumber() == previewRevision && !model.canUndo());
    result = model.applySourceSettingsWithReport(source, {b, a});
    data = model.snapshot();
    const auto& warped = data.tracks[0].clips[0];
    check("coherent_shared_source_boundary", !result.rejected && result.updatedNotes == 4
        && near(nativeSourceTimeAt(nativeClipClock(warped), 1), .8)
        && near(nativeSourceTimeAt(nativeClipClock(warped), .4), .16)
        && near(nativeSourceTimeAt(nativeClipClock(warped), 1.6), 1.64)
        && warped.sourceTimeMap.size() >= clip.sourceTimeMap.size());
    check("timeline_connections_and_edits_preserved", near(warped.startSeconds, 1)
        && near(warped.durationSeconds, 2) && near(warped.notes[1].startSeconds, 1)
        && near(warped.notes[0].durationSeconds, 1) && preserved(warped.notes[0])
        && data.nativeConnections.size() == 1 && near(data.nativeConnections[0].boundarySeconds, 2));
    check("single_transaction_undo", model.undo() && !model.canUndo()
        && near(model.snapshot().tracks[0].clips[0].notes[0].nativeSourceEndSeconds, 1));
    const auto undoRevision = model.revisionNumber();
    auto oldA = a, oldB = b; oldA.regionEndSeconds = oldB.regionStartSeconds = 1;
    result = model.applySourceSettingsWithReport(source, {oldB, oldA});
    check("noop_keeps_redo", !result.changed() && model.canRedo() && model.revisionNumber() == undoRevision);
    check("redo_boundary", model.redo());

    juce::String error;
    const auto projectFile = output.getChildFile("binding-roundtrip.hachi");
    check("save", model.save(projectFile, error));
    ProjectModel loaded; check("load", loaded.load(projectFile, error));
    auto loadedData = loaded.snapshot();
    const auto& saved = loadedData.tracks[0].clips[0].notes[0];
    check("identity_and_metadata_roundtrip", saved.materialAudioId == a.materialAudioId
        && saved.materialRegionId == a.materialRegionId && near(saved.materialSourceEndSeconds, .8)
        && saved.materialAdditionalAliases == a.additionalAliases
        && saved.materialConnectedNextRegionId == a.connectedNextRegionId
        && loadedData.tracks[0].clips[2].notes[0].materialLocalOverride);
    result = loaded.applySourceSettingsWithReport(source, {b});
    loadedData = loaded.snapshot();
    check("deleted_references_preserve_notes_and_snapshot", !result.rejected && result.unresolvedNotes == 2
        && loadedData.tracks[0].clips[0].notes.size() == 2
        && loadedData.tracks[0].clips[0].notes[0].materialReferenceUnresolved
        && loadedData.tracks[0].clips[0].notes[0].label == a.name
        && near(loadedData.tracks[0].clips[0].notes[0].nativeSourceEndSeconds, .8));
    const auto unresolvedFile = output.getChildFile("unresolved-roundtrip.hachi");
    ProjectModel missingReload;
    check("unresolved_snapshot_roundtrip", loaded.save(unresolvedFile, error)
        && missingReload.load(unresolvedFile, error)
        && missingReload.snapshot().tracks[0].clips[0].notes[0].materialReferenceUnresolved
        && missingReload.snapshot().tracks[0].clips[0].notes[0].label == a.name);
    const auto missingRevision = loaded.revisionNumber();
    result = loaded.applySourceSettingsWithReport(source, {b});
    check("repeated_missing_is_noop", !result.changed() && loaded.revisionNumber() == missingRevision);
    result = loaded.applySourceSettingsWithReport(source, {});
    check("empty_complete_annotation_preserves_all_notes", !result.rejected
        && loaded.snapshot().tracks[0].clips[0].notes.size() == 2);
    result = loaded.applySourceSettingsWithReport(source, {a, b});
    check("restored_reference_resolves", !result.rejected
        && !loaded.snapshot().tracks[0].clips[0].notes[0].materialReferenceUnresolved);

    // A conflict in a later placement must roll back even earlier valid plans.
    auto conflict = data;
    conflict.tracks[0].clips[1].notes[0].materialLocalOverride = true;
    model.resetDocument(conflict);
    const auto beforeConflict = model.contentFingerprint();
    const auto conflictRevision = model.revisionNumber();
    auto badA = a, badB = b; badA.regionEndSeconds = badB.regionStartSeconds = .7;
    result = model.applySourceSettingsWithReport(source, {badA, badB});
    check("local_shared_boundary_conflict_is_atomic", result.rejected && !result.changed()
        && model.contentFingerprint() == beforeConflict && model.revisionNumber() == conflictRevision && !model.canUndo());
    model.resetDocument(data);
    const auto beforeOverlap = model.contentFingerprint();
    badA = a; badB = b; badA.regionEndSeconds = 1.1;
    result = model.applySourceSettingsWithReport(source, {badA, badB});
    check("overlapping_new_boundaries_rejected", result.rejected && model.contentFingerprint() == beforeOverlap);

    // Split references retain the full pronunciation baseline but must never
    // expand their cropped source slices on an alias-only refresh.
    model.resetDocument(data);
    const auto splitId = model.splitNote("a-one", .4);
    const auto splitBefore = model.snapshot().tracks[0].clips[0];
    auto aliasOnly = a; aliasOnly.name = "split-alias";
    result = model.applySourceSettingsWithReport(source, {b, aliasOnly});
    const auto splitAfter = model.snapshot().tracks[0].clips[0];
    check("split_alias_preserves_source_slices", splitId.isNotEmpty() && !result.rejected
        && splitAfter.notes.size() == splitBefore.notes.size()
        && near(splitAfter.notes[0].nativeSourceEndSeconds, splitBefore.notes[0].nativeSourceEndSeconds)
        && splitAfter.notes[0].materialRegionId == a.materialRegionId
        && near(splitAfter.notes[0].materialSourceEndSeconds, .8));
    const auto splitFingerprint = model.contentFingerprint();
    aliasOnly.regionEndSeconds = .7; badB = b; badB.regionStartSeconds = .7;
    result = model.applySourceSettingsWithReport(source, {badB, aliasOnly});
    check("cropped_boundary_rejected_without_mutation", result.rejected
        && model.contentFingerprint() == splitFingerprint);

    // Merged clips own source clocks in children, notes stay in the flat parent.
    auto mergedFixture = fixture;
    auto parent = clip; parent.id = "merged"; parent.sourceFile = juce::File{};
    auto child = clip; child.id = "child"; child.startSeconds = .5; child.notes.clear();
    parent.parts = {child}; parent.durationSeconds = 3;
    for (auto& n : parent.notes) { n.clipPartId = child.id; n.startSeconds += .5; }
    mergedFixture.tracks[0].clips = {parent};
    model.resetDocument(mergedFixture);
    result = model.applySourceSettingsWithReport(source, {b, a});
    const auto mergedAfter = model.snapshot().tracks[0].clips[0];
    check("merged_child_clock_updates_not_parent", !result.rejected && result.updatedNotes == 2
        && near(nativeSourceTimeAt(nativeClipClock(mergedAfter.parts[0]), 1), .8)
        && near(mergedAfter.notes[1].startSeconds, 1.5)
        && mergedAfter.sourceTimeMap[2].sourceSeconds == 1);

    // Identical labels and list order cannot disambiguate legacy notes.
    auto legacy = fixture; legacy.tracks[0].clips.resize(1);
    for (auto& n : legacy.tracks[0].clips[0].notes)
    { n.materialAudioId.clear(); n.materialRegionId.clear(); n.materialSourceStartSeconds = n.materialSourceEndSeconds = -1; }
    model.resetDocument(legacy);
    auto duplicate = oldA; duplicate.materialRegionId = "other-region";
    result = model.applySourceSettingsWithReport(source, {duplicate, oldB, oldA});
    auto legacyData = model.snapshot();
    check("legacy_ambiguity_no_index_or_alias_guess", !result.rejected && result.ambiguousLegacyNotes == 1
        && legacyData.tracks[0].clips[0].notes[0].materialRegionId.isEmpty()
        && legacyData.tracks[0].clips[0].notes[1].materialRegionId == oldB.materialRegionId);
    result = model.applySourceSettingsWithReport(source, {oldB, oldA});
    check("legacy_unique_edges_bind", !result.rejected && result.boundLegacyNotes == 1
        && model.snapshot().tracks[0].clips[0].notes[0].materialRegionId == oldA.materialRegionId);
    // Source filename scopes IDs, even when another manifest reuses both IDs.
    auto scoped = fixture; scoped.tracks[0].clips[1].sourceFile = output.getChildFile("different.wav");
    model.resetDocument(scoped);
    result = model.applySourceSettingsWithReport(source, {oldB, oldA});
    check("source_path_scopes_composite_identity", !result.rejected
        && model.snapshot().tracks[0].clips[1].notes[0].label == "a");

    // Import consumes persisted manifest IDs, not only handcrafted test notes.
    check("annotation_save", SampleSettings::save(source, {oldA, oldB}, error));
    const auto rows = SampleSettings::loadOrDerive(source, ProjectData{}, &error);
    ProjectModel imported;
    const auto importedId = imported.addAudioFile(source, 3);
    const auto importedData = imported.snapshot();
    check("import_binds_manifest_identity", error.isEmpty() && importedId.isNotEmpty() && rows.size() == 2
        && !importedData.tracks.empty() && !importedData.tracks[0].clips[0].notes.empty()
        && importedData.tracks[0].clips[0].notes[0].materialAudioId == rows[0].materialAudioId
        && importedData.tracks[0].clips[0].notes[0].materialRegionId == rows[0].materialRegionId);
    check("report_written", output.getChildFile("material-binding-smoke.txt").replaceWithText(log));
    return ok;
}
}
