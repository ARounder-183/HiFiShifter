#pragma once
#include "../MaterialAnnotationSession.h"
#include "../OtoWaveformEditorComponent.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
// Destination/callback-contract checks ONLY. Not evidence of a native/UTAU
// ProjectModel adapter, project reopen, renderer or musical-undo acceptance.
inline bool runLocalMaterialScopeSmoke(const juce::File& output)
{
    using Session = MaterialAnnotationSession;
    using Scope = Session::EditScope;
    const auto root = output.getChildFile("local-scope-" + juce::Uuid().toString());
    if (root.createDirectory().failed()) return false;
    bool ok = true; int checks = 0; juce::String log, error;
    const auto check = [&](const char* name, bool passed)
    { ++checks; ok &= passed; log += juce::String(name) + "=" + (passed ? "PASS" : "FAIL") + "\n"; };
    const auto finish = [&]()
    {
        std::cout << "local_scope_contract=" << ok << "|checks=" << checks << "|fixture=" << root.getFullPathName() << std::endl;
        if (!ok) std::cerr << log << std::endl;
        return root.getChildFile("verification.txt").replaceWithText(log) && ok;
    };
    const auto audio = root.getChildFile("source.wav"), manifest = root.getChildFile("otto.ini"), oldOto = root.getChildFile("oto.ini");
    {
        juce::AudioBuffer<float> pcm(1, 48000); pcm.clear();
        auto stream = audio.createOutputStream(); juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(stream ? wav.createWriterFor(stream.release(), 48000, 1, 16, {}, 0) : nullptr);
        check("audio_fixture", writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples()));
    }
    check("manifest_fixture", manifest.replaceWithText("[otto]\nversion=1\nunit=ms\n[a1:source.wav]\n"
        "p1=chang|0,600|100,80,20|C:100,V:600|next:p2\n"
        "p2=other|600,1000|50,40,0|C:650,V:1000\n"));
    check("legacy_fixture", oldOto.replaceWithText("source.wav=must-not-be-read,0,100,-600,80,20\n"));
    juce::MemoryBlock audioBytes, manifestBytes, oldBytes;
    check("baseline_bytes", audio.loadFileAsData(audioBytes) && manifest.loadFileAsData(manifestBytes) && oldOto.loadFileAsData(oldBytes));
    const auto unchanged = [&]()
    {
        juce::MemoryBlock a, m, o;
        return audio.loadFileAsData(a) && manifest.loadFileAsData(m) && oldOto.loadFileAsData(o)
            && a == audioBytes && m == manifestBytes && o == oldBytes;
    };
    auto session = Session::open(audio, manifest, error);
    check("source_default_clean", session && session->editScope() == Scope::source && !session->dirty() && session->rows().size() == 2);
    if (!session || session->rows().size() != 2) return finish();
    const auto sourceRow = session->rows().front();
    // This is an explicitly labelled callback-contract harness, NOT a substitute
    // NoteData/model or a new persisted private-row format.
    SampleRegionSetting applied;
    int applies = 0, restores = 0, sourcePreflights = 0;
    juce::uint64 liveRevision = 12;
    juce::String liveFingerprint = "fixture-source-byte-token";
    Session::NoteScopeBinding binding;
    binding.noteId = "actual-note-ticket"; binding.audioId = "a1"; binding.regionId = "p1";
    binding.sourceAudio = audio; binding.sourceFingerprint = liveFingerprint;
    binding.projectRevision = liveRevision; binding.localRow = sourceRow;
    binding.apply = [&](const auto& ticket, const SampleRegionSetting& row, juce::uint64& nextRevision, juce::String& problem)
    {
        if (ticket.noteId != binding.noteId || ticket.projectRevision != liveRevision
            || ticket.sourceAudio != audio || ticket.sourceFingerprint != liveFingerprint
            || row.materialAudioId != "a1" || row.materialRegionId != "p1")
        { problem = "stale callback ticket"; return false; }
        ++applies; applied = row; nextRevision = ++liveRevision; return true;
    };
    binding.restore = [&](const auto& ticket, SampleRegionSetting& restored, juce::uint64& nextRevision, juce::String& problem)
    {
        if (ticket.projectRevision != liveRevision || ticket.sourceFingerprint != liveFingerprint)
        { problem = "stale restore ticket"; return false; }
        ++restores; restored = sourceRow; nextRevision = ++liveRevision; return true;
    };
    session->validatePublication = [&](const auto&, auto&) { ++sourcePreflights; return true; };
    check("bind_and_enter_same_session_local", session->enableNoteScope(binding, error)
        && session->setEditScope(Scope::thisNote, error) && session->rows().size() == 1 && !session->dirty());
    if (session->editScope() != Scope::thisNote) return finish();
    const auto vowel = session->rows().front().segments.back().id;
    check("same_named_vowel_planner", session->splitVowel("p1", vowel, .35, "an", "ng", error)
        && session->rows().front().vowelSplitsExplicit && session->rows().front().segments.size() == 3);
    auto edited = session->rows().front();
    edited.simplifiedConsonant = true; edited.pronunciationKind = PronunciationKind::transition;
    check("same_C_P_and_kind_planner", session->setRow("p1", edited, error));
    check("same_local_draft_undo_redo", session->undo() && !session->rows().front().simplifiedConsonant
        && session->redo() && session->rows().front().simplifiedConsonant && unchanged());
    check("dirty_switch_rejects_transactionally", session->dirty() && !session->setEditScope(Scope::source, error)
        && session->editScope() == Scope::thisNote && unchanged());
    check("whole_row_topology_rejected", !session->createPronunciation(.7, .9, "new", error)
        && !session->erase({"p1"}, error) && session->rows().size() == 1);
    auto wrong = session->rows().front(); wrong.materialRegionId = "p2";
    check("other_identity_rejected", !session->setRow("p1", wrong, error));
    check("local_save_full_row_no_source_path", session->save(error) && applies == 1 && sourcePreflights == 0
        && applied.vowelSplitsExplicit && applied.segments.size() == 3
        && applied.segments[1].alias == "an" && applied.segments[2].alias == "ng"
        && applied.simplifiedConsonant && applied.pronunciationKind == PronunciationKind::transition
        && applied.connectedNextRegionId == "p2" && unchanged() && !session->dirty());
    check("source_switch_reloads_not_private_snapshot", session->setEditScope(Scope::source, error)
        && session->rows().size() == 2 && !session->rows().front().vowelSplitsExplicit && unchanged());
    check("local_reentry_keeps_applied_snapshot", session->setEditScope(Scope::thisNote, error)
        && session->rows().front().vowelSplitsExplicit && session->rows().front().segments.size() == 3);
    ++liveRevision; // external project transaction invalidates the captured ticket
    check("stale_revision_rejects_even_noop_apply", !session->save(error) && applies == 1 && unchanged());
    --liveRevision;
    liveFingerprint = "external-source-change";
    check("stale_source_fingerprint_rejects", !session->save(error) && applies == 1 && unchanged());
    liveFingerprint = binding.sourceFingerprint;
    check("restore_project_callback_only", session->restoreLocal(error) && restores == 1
        && !session->rows().front().vowelSplitsExplicit && sourcePreflights == 0 && unchanged());
    edited = session->rows().front(); edited.overlapSeconds = -.03;
    check("local_unpublished_edit", session->setRow("p1", edited, error) && session->dirty());
    check("explicit_discard_does_not_apply", session->discardAndSetEditScope(Scope::source, error)
        && applies == 1 && session->rows().size() == 2 && !session->dirty() && unchanged());

    I18n strings; strings.setLanguage(I18n::Language::enUS);
    int sourcePublications = 0;
    auto editor = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings,
        [&](const auto&) { ++sourcePublications; }, true, &error);
    check("one_component_default_native_overview", editor && editor->diagnosticMaterialOverview());
    if (!editor) return finish();
    binding.projectRevision = liveRevision; binding.localRow = sourceRow;
    check("same_component_enables_note_destination", editor->enableNoteScope(binding, Scope::thisNote, error)
        && !editor->diagnosticMaterialOverview() && editor->editScope() == Scope::thisNote
        && editor->diagnosticSaveCaption() == "Apply to this note");
    check("local_has_no_source_overview_backdoor", !editor->diagnosticMaterialBack());
    editor->diagnosticPronunciationKind(3);
    editor->diagnosticSave();
    check("component_save_never_calls_source_publisher", sourcePublications == 0 && applies == 2
        && applied.pronunciationKind == PronunciationKind::breath && unchanged());
    check("component_source_return_is_same_editor", editor->setEditScope(Scope::source, error)
        && editor->diagnosticMaterialOverview() && sourcePublications == 0 && unchanged());
    editor->diagnosticSave();
    check("source_save_still_uses_original_publisher", sourcePublications == 1 && applies == 2);
    juce::MemoryBlock finalAudio, finalOld;
    check("all_source_audio_and_old_OTO_unchanged", audio.loadFileAsData(finalAudio) && oldOto.loadFileAsData(finalOld)
        && finalAudio == audioBytes && finalOld == oldBytes);
    return finish();
}
}
