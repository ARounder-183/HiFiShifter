#pragma once
#include "../ProjectModel.h"
#include <iostream>
#include <map>
#include <set>

namespace hachi
{
// Pure ProjectModel regression: no waveform analysis, audio device or GUI.
inline bool runNativeConnectionGraphSmoke(const juce::File& output)
{
    if (!output.createDirectory()) return false;
    bool ok = true; int checks = 0; juce::Array<juce::var> results;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass; ++checks; std::cout << name << '=' << pass << std::endl;
        auto row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("pass", pass);
        results.add(juce::var(row));
    };
    const auto fixture = [&]
    {
        ProjectData data;
        for (int t = 0; t < 2; ++t)
        {
            TrackData track; track.id = t == 0 ? "primary" : "unrelated";
            track.compose = true; track.pitchAlgorithm = PitchAlgorithm::world;
            const int count = t == 0 ? 4 : 2;
            for (int i = 0; i < count; ++i)
            {
                ClipData clip; clip.id = track.id + "-clip-" + juce::String(i);
                clip.startSeconds = i; clip.durationSeconds = clip.sourceDurationSeconds = 1;
                clip.sourceOffsetSeconds = i; clip.sourceFile = output.getChildFile("source.wav");
                NoteData note; note.id = juce::String::charToString(static_cast<juce::juce_wchar>((t == 0 ? 'A' : 'X') + i));
                note.startSeconds = 0; note.durationSeconds = 1; note.midiNote = 60;
                if (t == 1)
                {
                    note.connectedToPrevious = i > 0; note.connectedToNext = i == 0;
                    clip.glideConnectedFromPrevious = i > 0; clip.glideConnectedToNext = i == 0;
                }
                clip.notes.push_back(note); track.clips.push_back(clip);
            }
            data.tracks.push_back(track);
        }
        data.nativeConnections.push_back({ "unrelated-link", "X", "Y", "pitch-and-amplitude", .5,
                                          {{0, 60}, {1, 61}}, {{0, -1}, {1, 0}} });
        return data;
    };
    const auto edge = [](const ProjectData& data, const juce::String& left, const juce::String& right) -> const NativeConnection*
    {
        for (const auto& connection : data.nativeConnections)
            if (connection.leftNoteId == left && connection.rightNoteId == right) return &connection;
        return nullptr;
    };
    const auto coherent = [](const ProjectData& data)
    {
        struct Entry { const NoteData* note; const ClipData* clip; std::size_t track; };
        std::map<juce::String, Entry> notes;
        std::map<juce::String, std::pair<bool, bool>> expected;
        std::map<const ClipData*, std::pair<bool, bool>> clipExpected;
        for (std::size_t t = 0; t < data.tracks.size(); ++t)
            for (const auto& clip : data.tracks[t].clips)
            {
                if (!clip.notes.empty()) clipExpected[&clip] = {false, false};
                for (const auto& note : clip.notes)
                {
                    if (!notes.emplace(note.id, Entry{&note, &clip, t}).second) return false;
                    expected[note.id] = {false, false};
                }
            }
        std::set<juce::String> outgoing, incoming, ids;
        for (const auto& connection : data.nativeConnections)
        {
            const auto a = notes.find(connection.leftNoteId), b = notes.find(connection.rightNoteId);
            if (a == notes.end() || b == notes.end() || a == b || a->second.track != b->second.track
                || !outgoing.insert(a->first).second || !incoming.insert(b->first).second
                || connection.id.isEmpty() || !ids.insert(connection.id).second) return false;
            expected[a->first].second = true; expected[b->first].first = true;
            if (a->second.clip != b->second.clip)
            { clipExpected[a->second.clip].second = true; clipExpected[b->second.clip].first = true; }
        }
        for (const auto& [id, entry] : notes)
            if (entry.note->connectedToPrevious != expected[id].first || entry.note->connectedToNext != expected[id].second) return false;
        for (const auto& [clip, flags] : clipExpected)
            if (clip->glideConnectedFromPrevious != flags.first || clip->glideConnectedToNext != flags.second) return false;
        return true;
    };
    const auto unrelatedIntact = [&](const ProjectData& data)
    {
        const auto* link = edge(data, "X", "Y");
        return link && link->id == "unrelated-link" && link->pitchCurve.size() == 2 && link->amplitudeCurve.size() == 2
            && data.tracks[1].clips[0].notes[0].connectedToNext && data.tracks[1].clips[1].notes[0].connectedToPrevious;
    };

    ProjectModel model; model.resetDocument(fixture());
    model.toggleNoteConnection("B"); auto first = model.snapshot();
    check("toggle_adds_ab_graph_and_flags", edge(first, "A", "B") && coherent(first));
    const auto firstId = edge(first, "A", "B") ? edge(first, "A", "B")->id : juce::String{};
    const auto firstFingerprint = model.contentFingerprint();
    model.toggleNoteConnection("C"); auto chain = model.snapshot();
    check("adding_bc_preserves_ab", edge(chain, "A", "B") && edge(chain, "B", "C") && coherent(chain));
    check("preserved_ab_keeps_identity", edge(chain, "A", "B") && edge(chain, "A", "B")->id == firstId);
    check("toggle_one_undo_restores_before_bc", model.undo() && model.contentFingerprint() == firstFingerprint && coherent(model.snapshot()));
    check("toggle_redo_restores_bc", model.redo() && edge(model.snapshot(), "B", "C") && coherent(model.snapshot()));
    model.toggleNoteConnection("D"); const auto fullChain = model.snapshot();
    check("three_edge_chain_is_coherent", edge(fullChain, "A", "B") && edge(fullChain, "B", "C") && edge(fullChain, "C", "D") && coherent(fullChain));
    model.toggleNoteConnection("C"); auto off = model.snapshot();
    check("toggle_off_bc_preserves_both_other_directions", edge(off, "A", "B") && !edge(off, "B", "C") && edge(off, "C", "D") && coherent(off));
    check("unrelated_track_survives_toggles", unrelatedIntact(off));
    check("undo_off_has_no_dangling_flags", model.undo() && coherent(model.snapshot()) && edge(model.snapshot(), "B", "C"));
    juce::String error; ProjectModel reopened;
    check("graph_save_load_preserves_chain", model.save(output.getChildFile("chain.hjpx"), error)
        && reopened.load(output.getChildFile("chain.hjpx"), error) && coherent(reopened.snapshot())
        && edge(reopened.snapshot(), "A", "B") && edge(reopened.snapshot(), "B", "C") && edge(reopened.snapshot(), "C", "D"));
    reopened.toggleNoteConnection("C");
    check("reopened_toggle_and_undo_are_coherent", coherent(reopened.snapshot()) && reopened.undo() && coherent(reopened.snapshot()));

    model.resetDocument(fullChain); const auto beforeBatch = model.contentFingerprint();
    const auto revision = model.revisionNumber(); model.setNotesConnection({"B", "C"}, true);
    check("batch_repeating_existing_pair_is_noop", model.revisionNumber() == revision && model.contentFingerprint() == beforeBatch);
    model.setNotesConnection({"B", "C"}, false); auto batchOff = model.snapshot();
    check("batch_off_matches_toggle_off", edge(batchOff, "A", "B") && !edge(batchOff, "B", "C") && edge(batchOff, "C", "D") && coherent(batchOff));
    check("batch_off_is_one_undo", model.undo() && model.contentFingerprint() == beforeBatch && !model.canUndo());
    model.setNotesConnection({"A", "C"}, true); auto replaced = model.snapshot();
    check("directional_replacement_removes_only_conflicts", edge(replaced, "A", "C") && edge(replaced, "C", "D")
        && !edge(replaced, "A", "B") && !edge(replaced, "B", "C") && coherent(replaced));
    check("directional_replacement_preserves_unrelated_track", unrelatedIntact(replaced));
    model.resetDocument(fixture()); model.setNotesConnection({"D", "B", "A", "C"}, true); auto batchChain = model.snapshot();
    check("batch_start_order_creates_complete_chain", edge(batchChain, "A", "B") && edge(batchChain, "B", "C") && edge(batchChain, "C", "D") && coherent(batchChain));
    check("batch_complete_chain_has_one_undo", model.undo() && coherent(model.snapshot()) && !edge(model.snapshot(), "A", "B") && !model.canUndo());

    auto legacy = fixture(); legacy.tracks[0].clips[0].notes[0].connectedToNext = true;
    legacy.tracks[0].clips[1].notes[0].connectedToPrevious = true;
    model.resetDocument(legacy); const auto legacyFingerprint = model.contentFingerprint();
    model.toggleNoteConnection("C"); auto migrated = model.snapshot();
    check("legacy_flag_ab_materialized_before_adding_bc", edge(migrated, "A", "B") && edge(migrated, "B", "C") && coherent(migrated));
    check("legacy_migration_and_edit_share_one_undo", model.undo() && model.contentFingerprint() == legacyFingerprint && !model.canUndo());
    legacy.tracks[0].clips[1].notes[0].connectedToNext = true;
    legacy.tracks[0].clips[2].notes[0].connectedToPrevious = true;
    model.resetDocument(legacy); model.toggleNoteConnection("C"); migrated = model.snapshot();
    check("legacy_complete_chain_toggle_off_preserves_ab", edge(migrated, "A", "B") && !edge(migrated, "B", "C") && coherent(migrated));
    model.resetDocument(legacy); model.setNotesConnection({"C", "D"}, true); migrated = model.snapshot();
    check("batch_migrates_full_legacy_chain_without_loss", edge(migrated, "A", "B") && edge(migrated, "B", "C") && edge(migrated, "C", "D") && coherent(migrated));
    check("migrated_graph_roundtrips", model.save(output.getChildFile("migrated.hjpx"), error)
        && reopened.load(output.getChildFile("migrated.hjpx"), error) && coherent(reopened.snapshot()) && edge(reopened.snapshot(), "B", "C"));
    auto clipLegacy = fixture(); clipLegacy.tracks[0].clips[0].glideConnectedToNext = true;
    clipLegacy.tracks[0].clips[1].glideConnectedFromPrevious = true;
    model.resetDocument(clipLegacy); model.toggleNoteConnection("C"); migrated = model.snapshot();
    check("legacy_clip_only_join_materialized", edge(migrated, "A", "B") && edge(migrated, "B", "C") && coherent(migrated));
    auto oneSided = fixture(); oneSided.tracks[0].clips[1].notes[0].connectedToPrevious = true;
    model.resetDocument(oneSided); model.toggleNoteConnection("C"); migrated = model.snapshot();
    check("one_sided_imported_note_join_is_not_lost", edge(migrated, "A", "B") && edge(migrated, "B", "C") && coherent(migrated));
    auto otherLegacy = fixture(); otherLegacy.nativeConnections.clear();
    model.resetDocument(otherLegacy); model.toggleNoteConnection("B"); const auto untouchedLegacy = model.snapshot();
    check("unrelated_legacy_track_flags_are_not_reprojected", untouchedLegacy.tracks[1].clips[0].notes[0].connectedToNext
        && untouchedLegacy.tracks[1].clips[1].notes[0].connectedToPrevious
        && untouchedLegacy.tracks[1].clips[0].glideConnectedToNext && untouchedLegacy.tracks[1].clips[1].glideConnectedFromPrevious
        && !edge(untouchedLegacy, "X", "Y"));

    model.resetDocument(fixture()); const auto untouched = model.contentFingerprint(); const auto initialRevision = model.revisionNumber();
    model.toggleNoteConnection("missing"); model.toggleNoteConnection("A");
    model.setNotesConnection({"A", "A"}, true); model.setNotesConnection({"A", "X"}, true);
    model.setNotesConnection({"A", "B", "missing"}, true);
    check("unknown_self_first_and_cross_track_requests_noop", model.contentFingerprint() == untouched && model.revisionNumber() == initialRevision && !model.canUndo());
    auto overlap = fixture(); overlap.tracks[0].clips[1].startSeconds = .5;
    model.resetDocument(overlap); const auto overlapFingerprint = model.contentFingerprint();
    model.toggleNoteConnection("B"); model.setNotesConnection({"A", "B"}, true);
    check("independent_overlapping_bodies_are_not_joined", model.contentFingerprint() == overlapFingerprint && !model.canUndo());
    auto lateOverlap = fixture(); lateOverlap.tracks[0].clips[2].startSeconds = 1.5;
    model.resetDocument(lateOverlap); const auto lateFingerprint = model.contentFingerprint();
    model.setNotesConnection({"A", "B", "C"}, true);
    check("invalid_later_batch_pair_does_not_partially_commit", model.contentFingerprint() == lateFingerprint && !model.canUndo());
    auto lead = fixture(); lead.tracks[0].clips[0].notes[0].durationSeconds = 0;
    lead.tracks[0].clips[1].startSeconds = 0;
    model.resetDocument(lead); model.toggleNoteConnection("B");
    check("zero_duration_same_time_leadin_remains_connectable", edge(model.snapshot(), "A", "B") && coherent(model.snapshot()));

    auto report = new juce::DynamicObject(); report->setProperty("ok", ok); report->setProperty("checks", checks);
    report->setProperty("results", juce::var(results));
    const bool written = output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout << "native_connection_graph_ok=" << (ok && written) << "; checks=" << checks << std::endl;
    return ok && written;
}
}
