#pragma once
#include "../ProjectModel.h"
#include "../backend/UtauRenderer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>

namespace hachi
{
inline bool runMidiExportScopeSmoke(const juce::File& output)
{
    if (!output.createDirectory()) return false;
    bool ok = true; int checks = 0; juce::Array<juce::var> results;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass; ++checks; std::cout << name << '=' << pass << std::endl;
        auto row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("pass", pass);
        results.add(juce::var(row));
    };
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-6; };
    struct NoteEvent { int track = 0, pitch = 0, velocity = 0; double tick = 0; };
    struct Decoded
    {
        bool valid = false; int type = -1, timeFormat = 0, tracks = 0;
        std::vector<NoteEvent> ons, offs;
        std::vector<std::tuple<int, juce::String, double>> lyrics;
        std::vector<std::pair<double, double>> tempos;
        std::vector<std::pair<int, int>> meters;
        std::vector<std::pair<int, juce::String>> names;
    };
    const auto decode = [](const juce::File& file)
    {
        Decoded decoded; auto stream = file.createInputStream(); juce::MidiFile midi;
        if (!stream || !midi.readFrom(*stream, false, &decoded.type)) return decoded;
        decoded.valid = true; decoded.timeFormat = midi.getTimeFormat(); decoded.tracks = midi.getNumTracks();
        for (int t = 0; t < midi.getNumTracks(); ++t)
            if (const auto* sequence = midi.getTrack(t))
                for (int i = 0; i < sequence->getNumEvents(); ++i)
                {
                    const auto& message = sequence->getEventPointer(i)->message;
                    if (message.isNoteOn()) decoded.ons.push_back({t, message.getNoteNumber(), message.getVelocity(), message.getTimeStamp()});
                    if (message.isNoteOff()) decoded.offs.push_back({t, message.getNoteNumber(), 0, message.getTimeStamp()});
                    if (message.isTextMetaEvent() && message.getMetaEventType() == 5)
                        decoded.lyrics.emplace_back(t, message.getTextFromTextMetaEvent(), message.getTimeStamp());
                    if (message.isTempoMetaEvent()) decoded.tempos.emplace_back(message.getTimeStamp(), message.getTempoSecondsPerQuarterNote());
                    if (message.isTimeSignatureMetaEvent())
                    { int numerator = 0, denominator = 0; message.getTimeSignatureInfo(numerator, denominator); decoded.meters.emplace_back(numerator, denominator); }
                    if (message.isTrackNameEvent()) decoded.names.emplace_back(t, message.getTextFromTextMetaEvent());
                }
        return decoded;
    };
    const auto hasNote = [&](const Decoded& midi, int pitch, double tick, int velocity)
    {
        return std::any_of(midi.ons.begin(), midi.ons.end(), [&](const auto& note)
            { return note.pitch == pitch && near(note.tick, tick) && note.velocity == velocity; });
    };
    const auto hasOff = [&](const Decoded& midi, int pitch, double tick)
    {
        return std::any_of(midi.offs.begin(), midi.offs.end(), [&](const auto& note)
            { return note.pitch == pitch && near(note.tick, tick); });
    };
    const auto hasLyric = [&](const Decoded& midi, const juce::String& lyric, double tick)
    {
        return std::any_of(midi.lyrics.begin(), midi.lyrics.end(), [&](const auto& event)
            { return std::get<1>(event) == lyric && near(std::get<2>(event), tick); });
    };
    const auto conductorValid = [&](const Decoded& midi)
    {
        return midi.valid && midi.type == 1 && midi.timeFormat == 480 && midi.tempos.size() == 3
            && near(midi.tempos[0].first, 0) && near(midi.tempos[0].second, .5)
            && near(midi.tempos[1].first, 960) && near(midi.tempos[1].second, .666667)
            && near(midi.tempos[2].first, 2400) && near(midi.tempos[2].second, .4)
            && midi.meters.size() == 1 && midi.meters[0] == std::pair{3, 4};
    };
    ProjectData data; data.name = "Scope score"; data.bpm = 120; data.numerator = 3; data.denominator = 4;
    data.beatOriginSeconds = .125; data.tempoChanges = {{2, 90}, {5, 150}};
    const auto makeNote = [&](const char* id, const juce::String& lyric, int midi, double quarter, double length, float gain)
    {
        NoteData note; note.id = id; note.label = lyric; note.midiNote = static_cast<float>(midi); note.gain = gain;
        note.startSeconds = data.secondsForQuarterPosition(quarter) - data.beatOriginSeconds;
        note.durationSeconds = data.secondsForQuarterPosition(quarter + length) - data.secondsForQuarterPosition(quarter);
        return note;
    };
    const auto chineseLyric = juce::String::fromUTF8("测试");
    ClipData singing; singing.id = "singing"; singing.startSeconds = data.beatOriginSeconds; singing.durationSeconds = 5;
    singing.notes = {makeNote("lead-note", chineseLyric, 60, 1.5, 1.5, .64f),
                     makeNote("literal-r", "R", 62, 3.5, .5, 2.0f),
                     makeNote("rest-upper", "RR", 120, 0, .25, 1),
                     makeNote("rest-trimmed", " rr ", 121, .5, .25, 1)};
    auto unpitched = makeNote("native-noise", "noise", 122, 6, .25, 1); unpitched.nativeUnpitched = true;
    singing.notes.push_back(unpitched);
    ClipData mutedClip; mutedClip.id = "muted-clip"; mutedClip.startSeconds = data.beatOriginSeconds;
    mutedClip.durationSeconds = 5; mutedClip.muted = true; mutedClip.notes = {makeNote("muted-note", "muted clip", 65, 4.5, .5, .8f)};
    TrackData lead; lead.id = "lead"; lead.name = "Lead"; lead.compose = true; lead.solo = true;
    lead.pitchAlgorithm = PitchAlgorithm::world; lead.clips = {singing, mutedClip};
    TrackData other; other.id = "other"; other.name = "No engine gate"; other.compose = false; other.accompaniment = true;
    ClipData otherClip; otherClip.id = "other-clip"; otherClip.startSeconds = data.beatOriginSeconds; otherClip.durationSeconds = 5;
    otherClip.notes = {makeNote("other-note", "harmony", 67, 5.5, .5, 0)}; other.clips = {otherClip};
    TrackData muted; muted.id = "muted"; muted.name = "Muted track"; muted.muted = true;
    ClipData mutedTrackClip; mutedTrackClip.id = "muted-track-clip"; mutedTrackClip.startSeconds = data.beatOriginSeconds;
    mutedTrackClip.durationSeconds = 5; mutedTrackClip.notes = {makeNote("muted-track-note", "muted track", 70, 6.5, .5, .9f)};
    muted.clips = {mutedTrackClip};
    TrackData empty; empty.id = "empty"; empty.name = "Empty";
    data.tracks = {lead, other, muted, empty};
    ProjectModel model; model.resetDocument(data);
    const auto frozen = model.snapshot(); const auto fingerprint = model.contentFingerprint(); const auto revision = model.revisionNumber();
    juce::String error;
    const auto allFile = output.getChildFile("all-default.mid"), selectedFile = output.getChildFile("selected.mid");
    check("legacy_three_argument_call_exports_default_whole_scope", ProjectModel::writeMidiFile(frozen, allFile, error));
    const auto all = decode(allFile);
    check("whole_score_conductor_retains_all_tempos_meter_and_ppq", conductorValid(all));
    check("default_scope_excludes_muted_tracks_and_clips", all.tracks == 3 && all.ons.size() == 3 && all.offs.size() == 3
        && hasNote(all, 60, 720, 64) && hasNote(all, 62, 1680, 127) && hasNote(all, 67, 2640, 1));
    check("reserved_rr_rests_and_native_unpitched_are_excluded", std::none_of(all.ons.begin(), all.ons.end(),
        [](const auto& note) { return note.pitch >= 120; }));
    check("literal_r_is_preserved_by_shared_rest_predicate", !backend::isRestLyric("R") && backend::isRestLyric(" rr ") && hasLyric(all, "R", 1680));
    check("solo_does_not_hide_other_score_tracks", hasNote(all, 67, 2640, 1));
    check("any_note_data_can_export_without_engine_or_compose_gates", std::any_of(all.names.begin(), all.names.end(),
        [](const auto& name) { return name.second == "No engine gate"; }));
    check("unicode_lyrics_and_gain_velocity_survive_decode", hasLyric(all, chineseLyric, 720) && hasNote(all, 60, 720, 64));
    check("notes_crossing_tempo_change_keep_musical_tick_lengths", hasOff(all, 60, 1440) && hasOff(all, 62, 1920));
    {
        auto stream = allFile.createInputStream(); juce::MidiFile timed;
        const bool read = stream && timed.readFrom(*stream, false);
        double on = -1, off = -1;
        if (read)
        {
            timed.convertTimestampTicksToSeconds();
            for (int t = 0; t < timed.getNumTracks(); ++t)
                if (const auto* sequence = timed.getTrack(t))
                    for (int i = 0; i < sequence->getNumEvents(); ++i)
                    {
                        const auto& message = sequence->getEventPointer(i)->message;
                        if (message.isNoteOn() && message.getNoteNumber() == 60) on = message.getTimeStamp();
                        if (message.isNoteOff() && message.getNoteNumber() == 60) off = message.getTimeStamp();
                    }
        }
        check("decoded_conductor_reconstructs_cross_tempo_note_seconds", read && near(on, .75) && near(off, 1.666667));
    }
    check("current_track_scope_writes_only_selected_track", ProjectModel::writeMidiFile(frozen, selectedFile, error, "lead"));
    const auto selected = decode(selectedFile);
    check("selected_scope_retains_full_conductor_and_lyrics", conductorValid(selected) && selected.tracks == 2
        && selected.ons.size() == 2 && hasLyric(selected, chineseLyric, 720) && hasLyric(selected, "R", 1680));
    const auto otherFile = output.getChildFile("other-selected.mid");
    check("explicit_other_track_ignores_audio_solo", ProjectModel::writeMidiFile(frozen, otherFile, error, "other")
        && decode(otherFile).ons.size() == 1 && hasNote(decode(otherFile), 67, 2640, 1));
    const auto allMutedFile = output.getChildFile("all-include-muted.mid");
    check("include_muted_scope_succeeds", ProjectModel::writeMidiFile(frozen, allMutedFile, error, {}, true));
    const auto withMuted = decode(allMutedFile);
    check("include_muted_restores_clip_and_track_notes_but_not_rests", conductorValid(withMuted) && withMuted.tracks == 4
        && withMuted.ons.size() == 5 && hasNote(withMuted, 65, 2160, 80) && hasNote(withMuted, 70, 3120, 90)
        && std::none_of(withMuted.ons.begin(), withMuted.ons.end(), [](const auto& note) { return note.pitch >= 120; }));
    const auto selectedMutedFile = output.getChildFile("selected-muted.mid");
    check("explicit_muted_track_can_be_included", ProjectModel::writeMidiFile(frozen, selectedMutedFile, error, "muted", true)
        && decode(selectedMutedFile).ons.size() == 1 && hasNote(decode(selectedMutedFile), 70, 3120, 90));

    juce::MemoryBlock original; check("existing_midi_bytes_captured", allFile.loadFileAsData(original));
    const auto unchanged = [&] { juce::MemoryBlock after; return allFile.loadFileAsData(after) && after == original; };
    check("unknown_scope_fails_without_replacing_existing_midi", !ProjectModel::writeMidiFile(frozen, allFile, error, "missing")
        && error.containsIgnoreCase("unknown") && unchanged());
    const auto absent = output.getChildFile("absent-parent/never.mid");
    check("unknown_scope_does_not_create_output_or_parent", !ProjectModel::writeMidiFile(frozen, absent, error, "missing")
        && !absent.exists() && !absent.getParentDirectory().exists());
    check("muted_selected_track_reports_zero_instead_of_other_tracks", !ProjectModel::writeMidiFile(frozen, allFile, error, "muted")
        && error.containsIgnoreCase("selected track") && unchanged());
    check("empty_selected_track_preserves_existing_destination", !ProjectModel::writeMidiFile(frozen, allFile, error, "empty") && unchanged());
    auto onlyMutedClip = frozen; onlyMutedClip.tracks[0].clips = {mutedClip};
    check("scope_with_only_muted_clips_does_not_fall_back_to_whole_project", !ProjectModel::writeMidiFile(onlyMutedClip, allFile, error, "lead") && unchanged());
    auto onlyRests = frozen; onlyRests.tracks[0].clips[0].notes = {singing.notes[2], singing.notes[3], unpitched};
    onlyRests.tracks[0].clips.resize(1);
    check("rest_only_selected_scope_is_zero_eligible", !ProjectModel::writeMidiFile(onlyRests, allFile, error, "lead", true) && unchanged());
    auto allMuted = frozen; for (auto& track : allMuted.tracks) track.muted = true;
    check("all_muted_default_whole_scope_is_atomic_failure", !ProjectModel::writeMidiFile(allMuted, allFile, error) && unchanged());
    check("export_does_not_mutate_any_project_fields_or_revision", model.contentFingerprint() == fingerprint && model.revisionNumber() == revision && !model.canUndo());

    ProjectModel switched; switched.resetDocument(frozen); const auto captured = switched.snapshot();
    auto another = frozen; another.name = "Another document"; another.tracks[0].clips[0].notes[0].label = "changed after dialog";
    switched.resetDocument(another); const auto switchedFingerprint = switched.contentFingerprint(); const auto switchedRevision = switched.revisionNumber();
    const auto frozenFile = output.getChildFile("frozen-dialog-snapshot.mid");
    check("captured_export_snapshot_survives_document_switch", ProjectModel::writeMidiFile(captured, frozenFile, error, "lead")
        && hasLyric(decode(frozenFile), chineseLyric, 720) && !hasLyric(decode(frozenFile), "changed after dialog", 720)
        && switched.contentFingerprint() == switchedFingerprint && switched.revisionNumber() == switchedRevision);
    auto report = new juce::DynamicObject(); report->setProperty("ok", ok); report->setProperty("checks", checks);
    report->setProperty("results", juce::var(results));
    const bool saved = output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout << "midi_export_scope_ok=" << (ok && saved) << "; checks=" << checks << std::endl;
    return ok && saved;
}
}
