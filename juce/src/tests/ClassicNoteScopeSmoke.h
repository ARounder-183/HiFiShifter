#pragma once
#include "../OtoWaveformEditorComponent.h"
#include "../OttoMaterialMigration.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
// Unregistered compiled GUI/model test of the actual classic factory/session.
// Synthetic timing fixtures are not acoustic or full Main-integration acceptance.
inline bool runClassicNoteScopeSmoke(const juce::File& output)
{
    using Editor = OtoWaveformEditorComponent;
    using Session = MaterialAnnotationSession;
    const auto base = output.getChildFile("classic-scope-" + juce::Uuid().toString());
    const auto root = base.getChildFile("bank"); if (root.createDirectory().failed()) return false;
    const auto audio = root.getChildFile("voice.wav"), oto = root.getChildFile("oto.ini"), manifest = root.getChildFile("otto.ini");
    {
        juce::AudioBuffer<float> pcm(1, 96000); pcm.clear();
        auto stream = audio.createOutputStream(); if (!stream) return false; juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        if (!writer || !writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false;
    }
    if (!oto.replaceWithText("voice.wav=ka,0,100,-900,80,20\n")) return false;
    const auto bytes = [](const juce::File& file)
    { juce::MemoryBlock value; file.loadFileAsData(value); return value; };
    const auto sourceBytes = bytes(audio), otoBytes = bytes(oto);
    ProjectData data; TrackData track; ClipData clip; NoteData note;
    track.id = "utau"; track.compose = true; track.pitchAlgorithm = PitchAlgorithm::utau;
    track.utauMode = UtauMode::classic; track.voicebankDirectory = root;
    clip.id = "clip"; clip.durationSeconds = 1;
    note.id = "note"; note.label = "ka"; note.midiNote = 64; note.durationSeconds = .5;
    note.gain = .7f; note.formantSemitones = 1.25f; note.materialLocalOverride = false;
    clip.notes = {note}; track.clips = {clip}; data.tracks = {track};
    ProjectModel model; model.resetDocument(data); I18n strings;
    bool ok = true; juce::String error, report;
    const auto check = [&](const char* name, bool passed)
    { ok &= passed; report += juce::String(name) + "=" + (passed ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << passed << std::endl; };
    const auto near = [](double a, double b) { return std::abs(a-b) < 1.e-6; };
    auto revision = model.revisionNumber();
    auto sample = backend::UtauRenderer::resolveVoiceSample(root, "ka", 64, 100, false, false);
    check("canonical_source_ticket_available", sample.found && sample.sourceOto && sample.sourceOto->lineIndex == 0);
    const auto beforeOpen = model.contentFingerprint();
    auto editor = Editor::forClassicNote(model, "note", strings, sample, revision, error, Editor::EditScope::source);
    check("source_open_no_musical_mutation", editor && model.contentFingerprint() == beforeOpen && model.revisionNumber() == revision);
    check("same_session_switches_to_scalar_local", editor && editor->setEditScope(Editor::EditScope::thisNote, error)
        && editor->diagnosticClassicScalarScope() && !editor->diagnosticNativeAuthoringEnabled());
    if (editor)
    {
        editor->diagnosticTypeParameter(2, "-800");
        editor->diagnosticTypeParameter(1, "120");
        editor->diagnosticTypeParameter(3, "90");
        editor->diagnosticTypeParameter(4, "-10");
        editor->diagnosticSave();
    }
    const auto applied = model.snapshot().tracks[0].clips[0].notes[0];
    check("scalar_C_P_overlap_saved_on_actual_note", applied.utauOto.enabled
        && near(applied.utauOto.consonantMs,120) && near(applied.utauOto.preutteranceMs,90)
        && near(applied.utauOto.overlapMs,-10) && near(applied.utauOto.cutoffMs,-800));
    check("scalar_not_fake_native_binding_or_local_override", applied.materialAudioId.isEmpty() && applied.materialRegionId.isEmpty()
        && !applied.materialLocalOverride && applied.midiNote == note.midiNote && applied.gain == note.gain
        && applied.formantSemitones == note.formantSemitones);
    check("classic_restore_is_engine_inheritance", editor && editor->restoreLocal(error)
        && !model.snapshot().tracks[0].clips[0].notes[0].utauOto.enabled
        && !model.snapshot().tracks[0].clips[0].notes[0].materialLocalOverride);
    check("return_source_preserves_source_view", editor && editor->setEditScope(Editor::EditScope::source, error)
        && !editor->diagnosticClassicScalarScope() && editor->diagnosticMaterialRows().size() == 1);
    const auto revisionAfterRestore = model.revisionNumber();
    auto legacy = Editor::forNote(model, "note", error);
    check("old_factory_delegates_shared_session", legacy && legacy->diagnosticClassicScalarScope());
    if (legacy)
    {
        legacy->diagnosticTypeParameter(2, "-500");
        legacy->diagnosticTypeParameter(0, "1200");
        legacy->diagnosticTypeParameter(3, "200");
        legacy->diagnosticSave();
    }
    const auto shifted = model.snapshot().tracks[0].clips[0].notes[0].utauOto;
    check("legacy_negative_cutoff_offset_sequence_preserved", shifted.enabled && near(shifted.offsetMs,1200)
        && near(shifted.cutoffMs,-500) && near(shifted.preutteranceMs,200)
        && near(shifted.consonantMs,100) && near(shifted.overlapMs,20) && model.revisionNumber() > revisionAfterRestore);
    auto missing = sample; missing.sourceOto.reset();
    check("missing_ticket_no_alias_relookup", !Editor::forClassicNote(model,"note",strings,missing,model.revisionNumber(),error));
    check("stale_resolver_revision_rejects", !Editor::forClassicNote(model,"note",strings,sample,revision,error));
    const auto stableProject = model.contentFingerprint();
    check("source_PCM_OTO_unchanged_no_readtime_migration", bytes(audio) == sourceBytes && bytes(oto) == otoBytes && !manifest.exists());

    // Actual Session capability guard, independent of disabled widgets.
    auto session = Session::open(audio, manifest, error);
    if (session && !session->rows().empty())
    {
        Session::NoteScopeBinding binding;
        binding.noteId = "guard"; binding.audioId = session->rows()[0].materialAudioId;
        binding.regionId = session->rows()[0].materialRegionId; binding.sourceAudio = audio;
        binding.sourceFingerprint = "guard-test-only"; binding.localRow = session->rows()[0];
        binding.capability = Session::NoteScopeCapability::classicScalarOto;
        binding.apply = [](const auto&,const auto&,juce::uint64&,juce::String&){return true;};
        check("session_scalar_guard_enabled", session->enableNoteScope(binding,error)
            && session->setEditScope(Session::EditScope::thisNote,error));
        const auto original = session->rows()[0]; auto invalid = original;
        invalid.pronunciationKind = PronunciationKind::breath;
        check("scalar_kind_rejected_before_commit", !session->setRow(binding.regionId,invalid,error));
        invalid = original; invalid.simplifiedConsonant = true;
        check("scalar_simple_consonant_rejected", !session->setRow(binding.regionId,invalid,error));
        invalid = original; invalid.vowelSplitsExplicit = true;
        check("scalar_named_weighted_V_rejected", !session->setRow(binding.regionId,invalid,error));
        invalid = original; invalid.name = "not_scalar_storage";
        check("scalar_alias_metadata_not_pretend_stored", !session->setRow(binding.regionId,invalid,error));
        invalid = original; invalid.regionStartSeconds = 1.8; invalid.regionEndSeconds = 2.3;
        invalid.alignmentSeconds = 1.88; invalid.segments.clear();
        check("source_EOF_limit_rejection_is_atomic", !session->setRow(binding.regionId,invalid,error)
            && session->rows()[0].regionStartSeconds == original.regionStartSeconds
            && session->rows()[0].regionEndSeconds == original.regionEndSeconds
            && model.contentFingerprint() == stableProject);
    }
    else check("session_scalar_guard_enabled", false);
    check("capability_rejections_do_not_mutate_project", model.contentFingerprint() == stableProject);
    // Change only the OTO bytes: an immutable old ticket must fail before an
    // editor opens, rather than rediscovering another entry by label or path.
    oto.appendText("# external source revision\n");
    check("stale_OTO_ticket_rejects", !Editor::forClassicNote(model,"note",strings,sample,model.revisionNumber(),error));
    check("stale_ticket_does_not_mutate_music_or_PCM", model.contentFingerprint() == stableProject && bytes(audio) == sourceBytes);
    return output.getChildFile("classic-note-scope.txt").replaceWithText(report) && ok;
}
}
