#pragma once
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "../NativeNoteTiming.h"
#include <iostream>

namespace hachi
{
inline bool runMaterialInsertionSmoke(const juce::File& output)
{
    if (output.createDirectory().failed()) return false;
    bool ok = true; juce::String log, error;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass;
        log += juce::String(name) + "=" + (pass ? "PASS" : "FAIL") + "\n";
        std::cout << name << '=' << pass << std::endl;
    };
    const auto near = [](double a, double b) { return std::abs(a-b) < 1.e-7; };
    const auto audio = output.getChildFile("insertion-source.wav");
    juce::AudioBuffer<float> pcm(1, 96000);
    for (int i = 0; i < pcm.getNumSamples(); ++i)
        pcm.setSample(0, i, static_cast<float>(.2 * std::sin(i * juce::MathConstants<double>::twoPi * 220 / 48000)));
    juce::WavAudioFormat format; auto stream = audio.createOutputStream();
    if (!stream) return false;
    stream->setPosition(0); stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.get(), 48000, 1, 16, {}, 0));
    if (!writer) return false;
    stream.release(); check("wav_fixture", writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())); writer.reset();
    juce::MemoryBlock sourceBytes;
    check("source_bytes_captured", audio.loadFileAsData(sourceBytes));
    std::vector<SampleRegionSetting> rows(3);
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        auto& r = rows[i]; r.name = "same-alias"; r.materialAudioId = "audio-one";
        r.materialRegionId = "region-" + juce::String(static_cast<int>(i));
        r.regionStartSeconds = .2 + .4 * i; r.regionEndSeconds = .6 + .4 * i;
        r.alignmentSeconds = r.regionStartSeconds + .03; r.fixedDurationSeconds = .08;
        r.melodyneAmplitude = .75; r.relativePitchCents = 125;
        r.melodynePitchDrift = .6; r.melodynePitchModulation = .7;
        r.melodyneFormantCents = 150; r.melodyneSibilantBalance = .2;
        r.melodyneAttackSeconds = .04; r.role = NativeSegmentRole::vowel;
        r.provenance = "user"; r.confidence = .9f;
        r.additionalAliases.add("extra-" + juce::String(static_cast<int>(i)));
        r.amplitudeEnvelope = {{0, -7}, {.15, -1}, {.4, -5}};
        r.segments = {{"v", r.name, NativeSegmentRole::vowel, r.regionStartSeconds,
            r.regionEndSeconds, "user", .9f, r.alignmentSeconds, 0, true, 1}};
        if (i < 2) r.connectedNextRegionId = "region-" + juce::String(static_cast<int>(i + 1));
    }
    TrackData nativeTrack; nativeTrack.id = "native"; nativeTrack.pitchAlgorithm = PitchAlgorithm::world;
    ProjectData emptyNative; emptyNative.tracks = {nativeTrack};
    ProjectModel model; model.resetDocument(emptyNative);
    const auto revision = model.revisionNumber();
    const auto fullId = model.addMaterialPronunciations(audio, rows,
        {"region-2", "region-0", "region-1"}, 2, "native", error);
    auto data = model.snapshot();
    check("full_three_row_drag_one_revision", fullId.isNotEmpty() && error.isEmpty()
        && model.revisionNumber() == revision + 1 && data.tracks.size() == 1 && data.tracks[0].clips.size() == 1);
    if (fullId.isEmpty() || data.tracks.empty() || data.tracks[0].clips.empty()) return false;
    const auto full = data.tracks[0].clips[0];
    check("full_shared_clock_excludes_exterior_padding", full.parts.empty() && full.notes.size() == 3
        && near(full.startSeconds, 2) && near(full.sourceOffsetSeconds, .2)
        && near(full.sourceDurationSeconds, 1.2) && near(full.durationSeconds, 1.2)
        && near(nativeSourceTimeAt(nativeClipClock(full), .6), .6));
    check("same_aliases_resolve_only_by_ids", full.notes[0].materialRegionId == "region-0"
        && full.notes[1].materialRegionId == "region-1" && full.notes[2].materialRegionId == "region-2"
        && full.notes[0].label == full.notes[1].label && full.notes[0].id != full.notes[1].id);
    const auto& middle = full.notes[1];
    check("expression_and_source_segments_preserved", near(middle.gain, .75)
        && near(middle.midiNote, 61.25) && near(middle.formantSemitones, 1.5)
        && near(middle.attackSpeed, 2) && near(middle.consonantSeconds, .08)
        && middle.amplitudeEnvelope.size() == 3 && middle.amplitudeEnvelope[1].gainDb == -1
        && middle.nativeSegments.size() == 1 && near(middle.nativeSegments[0].sourceStartSeconds, 0)
        && near(middle.nativeSegments[0].sourceEndSeconds, .4)
        && middle.materialAdditionalAliases == rows[1].additionalAliases);
    check("neutral_stub_not_fake_measured_f0", !middle.sourcePitchMeasured && middle.sourceMidiCenter < 0
        && middle.contour.size() == 2 && middle.contour[0].relativeCents == 0);
    check("explicit_links_create_matching_connection_flags", data.nativeConnections.size() == 2
        && !full.notes[0].connectedToPrevious && full.notes[0].connectedToNext
        && full.notes[1].connectedToPrevious && full.notes[1].connectedToNext
        && full.notes[2].connectedToPrevious && !full.notes[2].connectedToNext
        && data.nativeConnections[0].leftNoteId == full.notes[0].id
        && data.nativeConnections[0].rightNoteId == full.notes[1].id
        && near(data.nativeConnections[0].boundarySeconds, 2.4));
    const auto savedFile = output.getChildFile("material-insertion.hachi");
    check("full_save", model.save(savedFile, error));
    ProjectModel reopened; check("full_load", reopened.load(savedFile, error));
    const auto reopenedData = reopened.snapshot();
    check("identity_and_link_roundtrip", reopenedData.tracks[0].clips[0].notes[1].materialRegionId == "region-1"
        && near(reopenedData.tracks[0].clips[0].notes[1].materialSourceStartSeconds, .6)
        && reopenedData.nativeConnections.size() == 2 && reopenedData.tracks[0].clips[0].notes[1].connectedToNext);
    check("full_single_step_undo", model.undo() && !model.canUndo()
        && model.snapshot().tracks[0].clips.empty() && model.snapshot().nativeConnections.empty());
    check("full_redo", model.redo() && model.snapshot().tracks[0].clips[0].notes.size() == 3);

    ProjectModel single; single.resetDocument(emptyNative);
    const auto middleId = single.addMaterialPronunciations(audio, rows, {"region-1"}, 4, "native", error);
    const auto singleData = single.snapshot();
    check("single_middle_only_source_slice", middleId.isNotEmpty() && singleData.tracks[0].clips[0].notes.size() == 1
        && near(singleData.tracks[0].clips[0].sourceOffsetSeconds, .6)
        && near(singleData.tracks[0].clips[0].sourceDurationSeconds, .4)
        && near(singleData.tracks[0].clips[0].notes[0].startSeconds, 0)
        && singleData.tracks[0].clips[0].notes[0].materialRegionId == "region-1"
        && singleData.nativeConnections.empty() && !singleData.tracks[0].clips[0].notes[0].connectedToNext);
    const auto repeatId = single.addMaterialPronunciations(audio, rows, {"region-1"}, 6, "native", error);
    auto repeated = single.snapshot();
    check("repeated_identity_distinct_note_ids", repeatId.isNotEmpty() && repeated.tracks[0].clips.size() == 2
        && repeated.tracks[0].clips[0].notes[0].materialRegionId == repeated.tracks[0].clips[1].notes[0].materialRegionId
        && repeated.tracks[0].clips[0].notes[0].id != repeated.tracks[0].clips[1].notes[0].id);

    ProjectModel adjacent; adjacent.resetDocument(emptyNative);
    const auto adjacentId = adjacent.addMaterialPronunciations(audio, rows, {"region-2", "region-1"}, 3, "native", error);
    const auto adjacentData = adjacent.snapshot();
    const auto& pair = adjacentData.tracks[0].clips[0];
    check("adjacent_two_common_source_clock", adjacentId.isNotEmpty() && pair.parts.empty() && pair.notes.size() == 2
        && near(pair.sourceOffsetSeconds, .6) && near(pair.sourceDurationSeconds, .8)
        && near(pair.notes[1].startSeconds, .4) && adjacentData.nativeConnections.size() == 1);
    auto unlinkedRows = rows;
    for (auto& r : unlinkedRows) r.connectedNextRegionId.clear();
    ProjectModel unlinked; unlinked.resetDocument(emptyNative);
    check("touching_is_not_implicit_connection", unlinked.addMaterialPronunciations(audio, unlinkedRows,
        {"region-0", "region-1"}, 1, "native", error).isNotEmpty()
        && unlinked.snapshot().nativeConnections.empty()
        && !unlinked.snapshot().tracks[0].clips[0].notes[0].connectedToNext);

    ProjectModel disjoint; disjoint.resetDocument(emptyNative);
    const auto disjointId = disjoint.addMaterialPronunciations(audio, rows, {"region-2", "region-0"}, 1, "native", error);
    auto disjointData = disjoint.snapshot();
    const auto& disjointClip = disjointData.tracks[0].clips[0];
    const auto sourceParts = expandedClipParts(disjointClip);
    check("unselected_intermediate_audio_excluded", disjointId.isNotEmpty() && sourceParts.size() == 2
        && disjointClip.parts.size() == 2 && disjointClip.notes.size() == 2
        && near(sourceParts[0].sourceOffsetSeconds, .2) && near(sourceParts[0].sourceDurationSeconds, .4)
        && near(sourceParts[1].sourceOffsetSeconds, 1) && near(sourceParts[1].sourceDurationSeconds, .4)
        && near(sourceParts[1].startSeconds - (sourceParts[0].startSeconds + sourceParts[0].audioLength()), .4)
        && disjointData.nativeConnections.empty() && !disjointClip.nativeAudioLinked);
    const auto disjointFile = output.getChildFile("disjoint-insertion.hachi");
    ProjectModel disjointReload;
    check("disjoint_parts_roundtrip", disjoint.save(disjointFile, error) && disjointReload.load(disjointFile, error)
        && disjointReload.snapshot().tracks[0].clips[0].parts.size() == 2
        && disjointReload.snapshot().tracks[0].clips[0].notes[1].clipPartId.isNotEmpty());

    TrackData utau; utau.id = "utau"; utau.pitchAlgorithm = PitchAlgorithm::utau; utau.utauGlobalFlags = "g12";
    ProjectData utauData; utauData.tracks = {utau}; ProjectModel routed; routed.resetDocument(utauData);
    const auto routedId = routed.addMaterialPronunciations(audio, rows, {"region-1"}, 1, "utau", error);
    const auto routedData = routed.snapshot();
    check("utau_target_gets_new_native_track_not_override", routedId.isNotEmpty() && routedData.tracks.size() == 2
        && routedData.tracks[0].pitchAlgorithm == PitchAlgorithm::utau && routedData.tracks[0].utauGlobalFlags == "g12"
        && routedData.tracks[0].clips.empty() && trackShowsAllNativeRegions(routedData.tracks[1]));
    TrackData backing = utau; backing.id = "backing"; backing.accompaniment = true; backing.compose = false;
    ProjectData backingData; backingData.tracks = {backing}; routed.resetDocument(backingData);
    check("accompaniment_target_kept_unchanged", routed.addMaterialPronunciations(audio, rows,
        {"region-1"}, 1, "backing", error).isNotEmpty() && routed.snapshot().tracks.size() == 2
        && routed.snapshot().tracks[0].clips.empty() && routed.snapshot().tracks[0].accompaniment
        && trackShowsAllNativeRegions(routed.snapshot().tracks[1]));

    const auto dsFolder = output.getChildFile("ds-fixture");
    check("ds_fixture", dsFolder.createDirectory().wasOk()
        && dsFolder.getChildFile("dsconfig.yaml").replaceWithText("name: insertion-test\n"));
    auto ds = utau; ds.id = "ds"; ds.utauMode = UtauMode::mou; ds.voicebankDirectory = dsFolder;
    ProjectData dsData; dsData.tracks = {ds}; routed.resetDocument(dsData);
    check("diffsinger_target_kept_unchanged", routed.addMaterialPronunciations(audio, rows,
        {"region-1"}, 1, "ds", error).isNotEmpty() && routed.snapshot().tracks.size() == 2
        && trackIsDiffSinger(routed.snapshot().tracks[0]) && routed.snapshot().tracks[0].clips.empty()
        && trackShowsAllNativeRegions(routed.snapshot().tracks[1]));
    auto pitchedRows = rows; pitchedRows[1].melodyneOriginalPitchCenterCents = 6000;
    pitchedRows[1].melodynePitchCenterCents = 6400;
    ProjectModel pitched; pitched.resetDocument(emptyNative);
    check("annotated_pitch_center_not_marked_sampled", pitched.addMaterialPronunciations(audio, pitchedRows,
        {"region-1"}, 1, "native", error).isNotEmpty()
        && near(pitched.snapshot().tracks[0].clips[0].notes[0].sourceMidiCenter, 60)
        && near(pitched.snapshot().tracks[0].clips[0].notes[0].midiNote, 64)
        && !pitched.snapshot().tracks[0].clips[0].notes[0].sourcePitchMeasured);
    auto gappedRows = unlinkedRows;
    for (std::size_t i = 1; i < gappedRows.size(); ++i)
    {
        auto& r = gappedRows[i]; r.regionStartSeconds += .05; r.regionEndSeconds += .05; r.alignmentSeconds += .05;
        for (auto& s : r.segments) { s.sourceStartSeconds += .05; s.sourceEndSeconds += .05; s.alignmentSeconds += .05; }
    }
    ProjectModel gapped; gapped.resetDocument(emptyNative);
    const auto gappedId = gapped.addMaterialPronunciations(audio, gappedRows,
        {"region-0", "region-1", "region-2"}, 1, "native", error);
    const auto gappedData = gapped.snapshot();
    check("unannotated_gap_is_silent_without_compressing_time", gappedId.isNotEmpty()
        && gappedData.tracks[0].clips[0].parts.size() == 2
        && near(gappedData.tracks[0].clips[0].notes[1].startSeconds, .45)
        && near(gappedData.tracks[0].clips[0].parts[1].startSeconds, .45)
        && near(gappedData.tracks[0].clips[0].parts[0].sourceDurationSeconds, .4)
        && gappedData.nativeConnections.empty());

    const auto beforeFailure = model.contentFingerprint(); const auto failureRevision = model.revisionNumber();
    check("unknown_id_no_mutation", model.addMaterialPronunciations(audio, rows,
        {"same-alias"}, 1, "native", error).isEmpty() && error.isNotEmpty()
        && model.contentFingerprint() == beforeFailure && model.revisionNumber() == failureRevision);
    check("duplicate_selection_no_mutation", model.addMaterialPronunciations(audio, rows,
        {"region-0", "region-0"}, 1, "native", error).isEmpty() && model.contentFingerprint() == beforeFailure);
    auto outside = unlinkedRows; outside[2].regionEndSeconds = 3; outside[2].segments.clear();
    check("outside_actual_wav_rejected", model.addMaterialPronunciations(audio, outside,
        {"region-2"}, 1, "native", error).isEmpty() && model.contentFingerprint() == beforeFailure);
    auto overlap = unlinkedRows; overlap[0].regionEndSeconds = .7; overlap[0].segments.clear();
    check("overlapping_selected_ranges_rejected", model.addMaterialPronunciations(audio, overlap,
        {"region-0", "region-1"}, 1, "native", error).isEmpty() && model.contentFingerprint() == beforeFailure);
    auto idless = rows; idless[0].materialRegionId.clear();
    check("unstable_identity_rejected", model.addMaterialPronunciations(audio, idless,
        {"region-1"}, 1, "native", error).isEmpty() && model.contentFingerprint() == beforeFailure);
    juce::MemoryBlock afterBytes;
    check("source_bytes_unchanged", audio.loadFileAsData(afterBytes) && sourceBytes == afterBytes);
    check("report_written", output.getChildFile("material-insertion-smoke.txt").replaceWithText(log));
    return ok;
}
}
