#pragma once
#include "../ProjectMaterialRegistrationFacts.h"
#include "../NativeNoteTiming.h"
#include <juce_cryptography/juce_cryptography.h>
#include <iostream>

namespace hachi
{
inline bool runProjectMaterialRegistrationFactsSmoke(const juce::File& outputPrefix)
{
    using F = ProjectMaterialRegistrationFacts;
    using R = ProjectMaterialRegistration;
    if (!outputPrefix.isDirectory()) return false;
    const auto output = outputPrefix.getChildFile("registration-facts-" + juce::Uuid().toString());
    if (output.exists() || output.createDirectory().failed()) return false;
    bool ok = true; juce::String log, error;
    const auto check = [&](const char* name, bool pass)
    { ok &= pass; log += juce::String(name) + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << pass << std::endl; };
    const auto source = output.getChildFile("original.wav");
    juce::AudioBuffer<float> pcm(1, 96000); pcm.clear();
    juce::WavAudioFormat wav; auto stream = source.createOutputStream(); if (!stream) return false;
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), 48000, 1, 16, {}, 0));
    if (!writer) return false; stream.release();
    if (!writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false; writer.reset();
    const auto sha = juce::SHA256(source).toHexString();
    NoteData n; n.id = "a"; n.label = "not-a-new-pronunciation-name"; n.durationSeconds = .8;
    n.nativeSourceStartSeconds = .15; n.nativeSourceEndSeconds = .9;
    n.materialSourceStartSeconds = n.materialSourceEndSeconds = -1;
    n.utauPreutteranceOverrideEnabled = true; n.utauPreutteranceSeconds = .04; n.consonantSeconds = .05;
    n.midiNote = 64; n.gain = .4f;
    n.nativeSegments = {{"legacy0", "-", NativeSegmentRole::vowel, 0, .2},
                        {"legacy1", "-", NativeSegmentRole::vowel, .2, .75}};
    for (int i = 0; i < 1000; ++i) n.contour.push_back({i * .0008, float(i % 13), 0, true});
    ClipData a; a.id = "owner-a"; a.sourceFile = source; a.sourceOffsetSeconds = .1;
    a.durationSeconds = a.sourceDurationSeconds = 1; a.sourceTimeMap = {{0, 0}, {.3, .1}, {1, 1}}; a.notes = {n};
    auto b = a; b.id = "owner-b"; b.startSeconds = 3; b.notes[0].id = "b"; b.notes[0].midiNote = 73; b.notes[0].gain = .8f;
    TrackData track; track.id = "track"; track.pitchAlgorithm = PitchAlgorithm::world; track.clips = {a, b};
    ProjectData project; project.tracks = {track};
    const auto beforeFp = ProjectModel::registrationMusicSourceFingerprint(project);
    const auto unconfirmed = F::derive(project);
    check("default_policy_does_not_promote_notes", !unconfirmed.ready && unconfirmed.variants.empty() && !unconfirmed.diagnostics.isEmpty());
    const auto facts = F::derive(project, F::Policy::ConfirmedCarriedPlacements);
    check("confirmation_creates_owner_variants_not_F0_rows", facts.ready && facts.variants.size() == 2
        && facts.variants[0].rows.size() == 1 && facts.variants[1].rows.size() == 1
        && facts.variants[0].variantKey != facts.variants[1].variantKey);
    const auto plan = R::prepare(project, 4, output, "registered", facts.variants, ProjectModel::registrationMusicSourceFingerprint);
    check("one_physical_copy_two_independent_variant_IDs", plan.ready && plan.recordings.size() == 1 && plan.document.audio.size() == 2
        && plan.document.audio[0].id != plan.document.audio[1].id
        && plan.document.audio[0].relativePath == plan.document.audio[1].relativePath);
    if (!plan.ready) { log += plan.diagnostics.joinIntoString("\n"); output.getChildFile("verification.txt").replaceWithText(log); return false; }
    for (const auto& audio : plan.document.audio) for (const auto& row : audio.rows)
        check("source_rows_unnamed_no_tuning_or_inferred_manual_V", row.name == "-" && !row.vowelSplitsExplicit
            && row.relativePitchCents == 0 && row.melodyneAmplitude == 1 && row.amplitudeEnvelope.empty());
    ProjectData candidate;
    check("reference_only_candidate", R::makeCandidate(project, plan, ProjectModel::registrationMusicSourceFingerprint, candidate, error));
    if (candidate.tracks.empty()) return false;
    const auto beforeP = nativeNoteSourceTiming(n, a, nativeClipClock(a));
    const auto& afterOwner = candidate.tracks[0].clips[0];
    const auto afterP = nativeNoteSourceTiming(afterOwner.notes[0], afterOwner, nativeClipClock(afterOwner));
    check("missing_local_baseline_stays_missing_P_exact", afterOwner.notes[0].materialSourceStartSeconds == -1
        && afterOwner.notes[0].materialSourceEndSeconds == -1 && beforeP.alignmentSeconds == afterP.alignmentSeconds
        && beforeP.fixedDurationSeconds == afterP.fixedDurationSeconds
        && beforeP.alignmentSeconds == .15 + .04);
    check("all_legacy_local_segments_pitch_and_source_maps_retained", ProjectModel::registrationMusicSourceFingerprint(candidate) == beforeFp
        && afterOwner.notes[0].nativeSegments.size() == 2 && afterOwner.notes[0].nativeSegments[0].id == "legacy0"
        && afterOwner.notes[0].contour.size() == 1000 && candidate.tracks[0].clips[1].notes[0].midiNote == 73);
    auto wrongBaseline = project; wrongBaseline.tracks[0].clips[0].notes[0].materialSourceStartSeconds = beforeP.regionStartSeconds;
    check("fingerprint_detects_P_origin_shift", ProjectModel::registrationMusicSourceFingerprint(wrongBaseline) != beforeFp);
    auto manual = project;
    for (auto& clip : manual.tracks[0].clips)
    {
        auto& note = clip.notes[0]; note.vowelSplitsExplicit = true;
        note.nativeSegments[0].alias = "an"; note.nativeSegments[0].stretchWeight = 2;
        note.nativeSegments[1].alias = "ng"; note.nativeSegments[1].stretchWeight = 3;
    }
    const auto manualFacts = F::derive(manual, F::Policy::ConfirmedCarriedPlacements);
    const auto manualPlan = R::prepare(manual, 4, output, "manual", manualFacts.variants, ProjectModel::registrationMusicSourceFingerprint);
    bool named = manualFacts.ready && manualPlan.ready;
    if (named)
    {
        bool an = false, ng = false;
        for (const auto& s : manualPlan.document.audio[0].rows[0].segments)
        { an |= s.alias == "an" && s.stretchWeight == 2; ng |= s.alias == "ng" && s.stretchWeight == 3; }
        named = an && ng;
    }
    check("existing_explicit_named_weighted_V_only", named);
    auto reversed = project; reversed.tracks[0].clips[0].sourceTimeMap[1].sourceSeconds = 2;
    check("reversed_clock_unresolved_no_reconstruction", !F::derive(reversed, F::Policy::ConfirmedCarriedPlacements).ready);
    auto mixed = project;
    auto& registered = mixed.tracks[0].clips[0];
    registered.sourceFile = output.getChildFile("unselected-existing-library-missing.wav");
    registered.materialSourceAnalysisPending = true; // unselected state is not validated/hashed as a source
    registered.notes[0].materialAudioId = "already-registered-audio";
    registered.notes[0].materialRegionId = "already-registered-region";
    registered.notes[0].materialSourceStartSeconds = .33;
    const R::OwnerScope onlySecond = std::vector<R::Locator>{{"track", "owner-b", {}, {}}};
    const auto selectedFacts = F::derive(mixed, F::Policy::ConfirmedCarriedPlacements, onlySecond);
    const auto selectedPlan = R::prepare(mixed, 4, output, "selected", selectedFacts.variants,
        ProjectModel::registrationMusicSourceFingerprint, {}, onlySecond);
    ProjectData selectedCandidate;
    check("selected_unregistered_owner_only_full_snapshot_retained", selectedFacts.ready && selectedFacts.variants.size() == 1
        && selectedPlan.ready && selectedPlan.owners.size() == 1 && selectedPlan.bindings.size() == 1 && selectedPlan.recordings.size() == 1
        && R::makeCandidate(mixed, selectedPlan, ProjectModel::registrationMusicSourceFingerprint, selectedCandidate, error)
        && selectedCandidate.tracks[0].clips[0].sourceFile == registered.sourceFile
        && selectedCandidate.tracks[0].clips[0].notes[0].materialAudioId == "already-registered-audio"
        && selectedCandidate.tracks[0].clips[0].notes[0].materialRegionId == "already-registered-region"
        && selectedCandidate.tracks[0].clips[0].notes[0].materialSourceStartSeconds == .33
        && selectedCandidate.tracks[0].clips[1].sourceFile != source
        && ProjectModel::registrationMusicSourceFingerprint(selectedCandidate) == ProjectModel::registrationMusicSourceFingerprint(mixed));
    const R::OwnerScope explicitlyEmpty = std::vector<R::Locator>{};
    const auto emptyFacts = F::derive(mixed, F::Policy::ConfirmedCarriedPlacements, explicitlyEmpty);
    bool fingerprintCalled = false;
    const auto emptyPlan = R::prepare(mixed, 4, output, "empty-selection", selectedFacts.variants,
        [&](const ProjectData& p) { fingerprintCalled = true; return ProjectModel::registrationMusicSourceFingerprint(p); }, {}, explicitlyEmpty);
    check("present_empty_scope_is_no_selection_not_all", !emptyFacts.ready && emptyFacts.variants.empty() && !emptyPlan.ready
        && emptyPlan.owners.empty() && emptyPlan.recordings.empty() && !fingerprintCalled
        && !output.getChildFile("empty-selection").exists());
    const R::OwnerScope missing = std::vector<R::Locator>{{"track", "not-an-owner", {}, {}}};
    check("missing_scoped_owner_refused", !F::derive(mixed, F::Policy::ConfirmedCarriedPlacements, missing).ready);
    check("adapter_never_mutates_input_or_original", ProjectModel::registrationMusicSourceFingerprint(project) == beforeFp
        && juce::SHA256(source).toHexString() == sha && !plan.publishedFolder.exists());
    output.getChildFile("verification.txt").replaceWithText(log); return ok;
}
}
