#pragma once
#include "../ProjectMaterialRegistration.h"
#include <juce_cryptography/juce_cryptography.h>
#include <cstring>
#include <iostream>

namespace hachi
{
// Needs the parent-owned PM declarations and inclusion of Model.inc. Explicit
// output prefix must already exist; every run creates a NEW uniquely-owned child.
inline bool runProjectMaterialRegistrationModelSmoke(const juce::File& outputPrefix)
{
    using R = ProjectMaterialRegistration;
    if (outputPrefix == juce::File{} || !outputPrefix.isDirectory()) return false;
    const auto output = outputPrefix.getChildFile("registration-model-" + juce::Uuid().toString());
    if (output.exists() || output.createDirectory().failed()) return false;
    bool ok = true; juce::String log, error;
    const auto check = [&](const char* key, bool pass)
    { ok &= pass; log += juce::String(key) + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << key << '=' << pass << std::endl; };
    const auto source = output.getChildFile("original.wav");
    juce::AudioBuffer<float> pcm(1, 96000);
    for (int i = 0; i < pcm.getNumSamples(); ++i) pcm.setSample(0, i, .13f * std::sin(float(i) * .031f));
    juce::WavAudioFormat wav; auto stream = source.createOutputStream(); if (!stream) return false;
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), 48000, 1, 16, {}, 0));
    if (!writer) return false; stream.release();
    if (!writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false; writer.reset();
    const auto sourceSha = juce::SHA256(source).toHexString();
    NoteData note; note.id = "a"; note.label = "unchanged-project-label";
    note.durationSeconds = 1; note.nativeSourceStartSeconds = .1; note.nativeSourceEndSeconds = 1.1;
    note.midiNote = 64; note.sourceMidiCenter = 60; note.gain = .43f; note.sourcePitchMeasured = true;
    note.consonantSeconds = .07; note.attackSpeed = 1.7f; note.formantSemitones = 1.2f;
    note.pitchControlPoints = {{-.03, 63}, {.4, 66}, {1.03, 62}};
    note.pitchSupportAuthored = true; note.nativePitchHandlesPlaced = true;
    note.amplitudeEnvelope = {{0, -7}, {.4, -2}, {1, -9}};
    note.contour = {{0, 13, 5, true}, {.5, 17, 3, true}, {1, 2, 0, false}};
    note.vibratoEnabled = true; note.vibratoDepthCents = 53;
    note.nativeSegments = {{"legacy-epoch", "-", NativeSegmentRole::vowel, 0, 1}};
    ClipData clip; clip.id = "first"; clip.sourceFile = source; clip.durationSeconds = 1.5; clip.sourceDurationSeconds = 1.5;
    clip.sourceTimeMap = {{0, 0}, {.4, .2}, {1.5, 1.5}};
    clip.nativeTrimClock = {{0, 0}, {.4, .2}, {1.5, 1.5}};
    clip.nativeSourcePitch = std::make_shared<const std::vector<NativeSourcePitchPoint>>(
        std::vector<NativeSourcePitchPoint>{{0, 60, 59, true}, {1.7, 65, 63, true}});
    clip.nativeSourcePitchComplete = true; clip.notes = {note};
    auto repeat = clip; repeat.id = "second"; repeat.startSeconds = 3; repeat.notes[0].id = "b";
    repeat.notes[0].midiNote = 72; repeat.notes[0].gain = .81f;
    TrackData track; track.id = "native"; track.pitchAlgorithm = PitchAlgorithm::world;
    track.nsfNoiseProtection = true; track.nsfSmoothPitchTransitions = false; track.clips = {clip, repeat};
    ProjectData fixture; fixture.name = "registration-model"; fixture.tracks = {track};
    ProjectModel model; model.resetDocument(fixture);
    const auto baseline = model.snapshot(); const auto revision = model.revisionNumber();
    const auto fingerprint = ProjectModel::registrationMusicSourceFingerprint(baseline);
    R::RowFacts row; row.key = "explicit-carried-pronunciation";
    row.annotation.regionStartSeconds = .1; row.annotation.regionEndSeconds = 1.2;
    row.annotation.alignmentSeconds = .25; row.annotation.fixedDurationSeconds = .3;
    row.members = {{"native", "first", {}, "a"}, {"native", "second", {}, "b"}};
    R::VariantFacts variant; variant.recording = source; variant.variantKey = "authored-source";
    variant.explicitPronunciationEvidence = true; variant.rows = {row};
    const auto plan = R::prepare(baseline, revision, output, "registered", {variant},
        ProjectModel::registrationMusicSourceFingerprint);
    check("plan_uses_authoritative_full_fingerprint", plan.ready && plan.musicSourceFingerprint == fingerprint);
    auto lease = R::stage(plan, error);
    check("model_fixture_stage", bool(lease));
    if (!lease) { log += error; output.getChildFile("verification.txt").replaceWithText(log); return false; }
    check("model_fixture_publication", lease->publish(error));
    auto badOwners = plan.owners; badOwners[0].sourceSha256 = juce::String::repeatedString("0", 64);
    check("bad_SHA_no_mutation_revision_undo", !model.adoptRegisteredMaterialBindings(revision, fingerprint, badOwners, plan.bindings, error)
        && model.revisionNumber() == revision && !model.canUndo() && model.snapshot().tracks[0].clips[0].sourceFile == source);
    auto badBindings = plan.bindings; badBindings[0].regionId = "missing-region";
    check("missing_published_identity_no_mutation", !model.adoptRegisteredMaterialBindings(revision, fingerprint, plan.owners, badBindings, error)
        && model.revisionNumber() == revision && !model.canUndo());
    auto incomplete = plan.bindings; incomplete.pop_back();
    check("partial_note_coverage_refused", !model.adoptRegisteredMaterialBindings(revision, fingerprint, plan.owners, incomplete, error)
        && model.revisionNumber() == revision && !model.canUndo());
    check("stale_revision_refused", !model.adoptRegisteredMaterialBindings(revision + 1, fingerprint, plan.owners, plan.bindings, error)
        && model.revisionNumber() == revision && !model.canUndo());
    const bool adoptionCommitted = model.adoptRegisteredMaterialBindings(revision, fingerprint, plan.owners, plan.bindings, error);
    if (adoptionCommitted) lease->retainAfterAdoption(); // BEFORE any fallible post-adoption work
    check("adoption_one_transaction", adoptionCommitted && model.revisionNumber() == revision + 1 && model.canUndo());
    if (!adoptionCommitted) { log += error; output.getChildFile("verification.txt").replaceWithText(log); return false; }
    const auto retainedFolder = lease->currentFolder();
    // Simulate a post-commit manager preference failure and even a damaged
    // provenance marker. Neither may re-arm destructor deletion of adopted PCM.
    retainedFolder.getChildFile(".hachi-registration-owner").appendText("simulated-preference-failure");
    lease.reset();
    check("post_adoption_destructor_pref_failure_keeps_media", retainedFolder.isDirectory()
        && retainedFolder.getChildFile(plan.recordings[0].copiedRelativePath).existsAsFile());
    const auto adopted = model.snapshot();
    check("all_music_source_clock_and_arrays_exact", ProjectModel::registrationMusicSourceFingerprint(adopted) == fingerprint
        && adopted.tracks[0].clips[0].nativeSourcePitch == baseline.tracks[0].clips[0].nativeSourcePitch
        && adopted.tracks[0].clips[1].notes[0].midiNote == 72 && adopted.tracks[0].clips[1].notes[0].gain == .81f
        && adopted.tracks[0].clips[0].notes[0].nativeSegments[0].id == "legacy-epoch"
        && adopted.tracks[0].clips[0].notes[0].label == note.label);
    check("copied_paths_and_variant_bindings_only", adopted.tracks[0].clips[0].sourceFile != source
        && adopted.tracks[0].clips[0].sourceFile == adopted.tracks[0].clips[1].sourceFile
        && adopted.tracks[0].clips[0].notes[0].materialAudioId == plan.bindings[0].audioId
        && adopted.tracks[0].clips[0].notes[0].materialLocalOverride);
    model.undo();
    check("single_undo_restores_original_paths_and_ids", !model.canUndo() && model.snapshot().tracks[0].clips[0].sourceFile == source
        && model.snapshot().tracks[0].clips[0].notes[0].materialAudioId.isEmpty()
        && ProjectModel::registrationMusicSourceFingerprint(model.snapshot()) == fingerprint);
    model.redo();
    check("redo_restores_reference_only_adoption", model.snapshot().tracks[0].clips[0].sourceFile == adopted.tracks[0].clips[0].sourceFile
        && ProjectModel::registrationMusicSourceFingerprint(model.snapshot()) == fingerprint);
    const auto saved = output.getChildFile("registered-copy.hjpx");
    check("save_new_project_copy", model.save(saved, error));
    ProjectModel cold; const bool loaded = cold.load(saved, error);
    check("cold_reopen_keeps_full_fingerprint", loaded && error.isEmpty()
        && ProjectModel::registrationMusicSourceFingerprint(cold.snapshot()) == fingerprint
        && cold.snapshot().tracks[0].clips[0].sourceFile == adopted.tracks[0].clips[0].sourceFile);
    check("original_recording_not_written", juce::SHA256(source).toHexString() == sourceSha);
    check("committed_sources_remain_available_for_redo_and_reopen", retainedFolder.isDirectory());
    if (error.isNotEmpty()) log += "last_error=" + error + "\n";
    output.getChildFile("verification.txt").replaceWithText(log);
    return ok;
}

// Mandatory real-project hook. Host must freeze/log engine/model/settings and
// render with its REAL engine. No default renderer, substitute or reanalysis.
// Explicit facts are mandatory; this hook never guesses teio pronunciation cuts.
using RegistrationRealRender = std::function<bool(const ProjectData&, const juce::File&, juce::String&)>;
inline bool compareRegistrationPcm(const juce::File& baseline, const juce::File& candidate, juce::String& error)
{
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> a(formats.createReaderFor(baseline)), b(formats.createReaderFor(candidate));
    if (!a || !b || a->sampleRate != b->sampleRate || a->numChannels != b->numChannels
        || a->numChannels == 0 || a->numChannels > 2
        || a->lengthInSamples <= 0 || a->lengthInSamples != b->lengthInSamples)
    { error = "Real registration render audio headers/length differ"; return false; }
    juce::AudioBuffer<float> x(int(a->numChannels), 8192), y(int(b->numChannels), 8192); float peak = 0;
    for (juce::int64 offset = 0; offset < a->lengthInSamples; offset += 8192)
    {
        const int count = int(std::min<juce::int64>(8192, a->lengthInSamples - offset));
        if (!a->read(&x, 0, count, offset, true, true) || !b->read(&y, 0, count, offset, true, true))
        { error = "Cannot decode real registration PCM"; return false; }
        for (int channel = 0; channel < x.getNumChannels(); ++channel)
        {
            peak = std::max(peak, x.getMagnitude(channel, 0, count));
            if (std::memcmp(x.getReadPointer(channel), y.getReadPointer(channel), std::size_t(count) * sizeof(float)) != 0)
            { error = "Real registration render PCM is not EXACT"; return false; }
        }
    }
    if (!(peak > 1.e-6f)) { error = "Silent output is not real-project registration acceptance"; return false; }
    return true;
}
inline bool runProjectMaterialRegistrationRealProjectGate(const juce::File& originalProject,
    const juce::File& outputPrefix, const std::vector<ProjectMaterialRegistration::VariantFacts>& explicitFacts,
    const RegistrationRealRender& render, juce::String& error)
{
    using R = ProjectMaterialRegistration;
    error.clear();
    if (!render || !originalProject.existsAsFile() || !outputPrefix.isDirectory())
    { error = "Real renderer, original project and existing output prefix are required"; return false; }
    const auto originalSha = juce::SHA256(originalProject).toHexString();
    const auto output = outputPrefix.getChildFile("registration-real-" + juce::Uuid().toString());
    if (output.exists() || output.createDirectory().failed()) { error = "Cannot create isolated real registration gate output"; return false; }
    ProjectModel model;
    if (!model.load(originalProject, error) || error.isNotEmpty()) return false;
    const auto before = model.snapshot(); const auto revision = model.revisionNumber();
    const auto fingerprint = ProjectModel::registrationMusicSourceFingerprint(before);
    const auto plan = R::prepare(before, revision, output, "registered-material", explicitFacts,
        ProjectModel::registrationMusicSourceFingerprint);
    if (!plan.ready) { error = plan.diagnostics.joinIntoString("\n"); return false; }
    const auto beforeWav = output.getChildFile("before.wav");
    if (!render(before, beforeWav, error)) return false;
    auto lease = R::stage(plan, error); if (!lease || !lease->publish(error)) return false;
    if (!model.adoptRegisteredMaterialBindings(revision, fingerprint, plan.owners, plan.bindings, error)) return false;
    lease->retainAfterAdoption(); // adopted references must survive later render/prefs/test failure
    const auto after = model.snapshot();
    if (ProjectModel::registrationMusicSourceFingerprint(after) != fingerprint) { error = "Real adoption fingerprint changed"; return false; }
    const auto afterWav = output.getChildFile("after-adoption.wav");
    if (!render(after, afterWav, error) || !compareRegistrationPcm(beforeWav, afterWav, error)) return false;
    const auto saved = output.getChildFile("registered-copy.hjpx");
    if (!model.save(saved, error)) return false;
    ProjectModel cold; if (!cold.load(saved, error) || error.isNotEmpty()) return false;
    if (ProjectModel::registrationMusicSourceFingerprint(cold.snapshot()) != fingerprint) { error = "Real cold-reopen fingerprint changed"; return false; }
    const auto coldWav = output.getChildFile("cold-reopen.wav");
    if (!render(cold.snapshot(), coldWav, error) || !compareRegistrationPcm(beforeWav, coldWav, error)) return false;
    if (juce::SHA256(originalProject).toHexString() != originalSha) { error = "Original project bytes changed during acceptance"; return false; }
    for (const auto& recording : plan.recordings)
        if (juce::SHA256(recording.original).toHexString() != recording.sha256)
        { error = "Original recording bytes changed during acceptance"; return false; }
    return lease->release(error);
}
}
