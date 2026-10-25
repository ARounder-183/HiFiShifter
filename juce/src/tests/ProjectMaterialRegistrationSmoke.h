#pragma once
#include "../ProjectMaterialRegistration.h"
#include <juce_cryptography/juce_cryptography.h>
#include <iostream>

namespace hachi
{
// Explicit output PREFIX required. Creates a NEW unique child; never removes or
// overwrites the supplied prefix or any user recording/project. No device/analysis.
inline bool runProjectMaterialRegistrationSmoke(const juce::File& outputPrefix)
{
    using R = ProjectMaterialRegistration;
    if (outputPrefix == juce::File{} || !outputPrefix.isDirectory()) return false;
    const auto output = outputPrefix.getChildFile("registration-" + juce::Uuid().toString());
    if (output.exists() || output.createDirectory().failed()) return false;
    bool ok = true; juce::String log;
    const auto check = [&](const char* name, bool pass)
    { ok &= pass; log += juce::String(name) + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << pass << std::endl; };
    const auto originals = output.getChildFile("originals"); originals.createDirectory();
    const auto destination = output.getChildFile("destination"); destination.createDirectory();
    const auto source = originals.getChildFile("voice.WAV");
    juce::AudioBuffer<float> pcm(1, 48000);
    for (int i = 0; i < pcm.getNumSamples(); ++i) pcm.setSample(0, i, float(.1 * std::sin(i * .031)));
    juce::WavAudioFormat wav; auto stream = source.createOutputStream();
    if (!stream) return false;
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(), 48000, 1, 16, {}, 0));
    if (!writer) return false;
    stream.release(); if (!writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples())) return false; writer.reset();
    const auto originalHash = juce::SHA256(source).toHexString();
    NoteData a; a.id = "note-a"; a.label = "keep-label"; a.durationSeconds = .4;
    a.nativeSourceStartSeconds = .1; a.nativeSourceEndSeconds = .5;
    a.midiNote = 65; a.sourceMidiCenter = 60; a.gain = .37f;
    a.consonantSeconds = .027; a.attackSpeed = 1.3f;
    a.pitchControlPoints = {{0, 65}, {.2, 67}, {.4, 64}};
    a.contour = {{0, 12, 2, true}, {.4, 20, 3, true}};
    a.amplitudeEnvelope = {{0, -8}, {.2, -2}, {.4, -9}};
    a.vibratoEnabled = true; a.vibratoDepthCents = 71;
    a.nativeSegments = {{"old-analysis", "-", NativeSegmentRole::vowel, 0, .4}};
    ClipData clip; clip.id = "clip-a"; clip.sourceFile = source;
    clip.durationSeconds = clip.sourceDurationSeconds = 1;
    clip.sourceTimeMap = {{0, 0}, {.25, .1}, {1, 1}};
    clip.nativeTrimClock = {{-.1, -.2}, {0, 0}, {1, 1}};
    clip.nativeSourcePitch = std::make_shared<const std::vector<NativeSourcePitchPoint>>(
        std::vector<NativeSourcePitchPoint>{{-.1, 60, 59, true}, {1.2, 65, 64, true}});
    clip.nativeSourcePitchComplete = true;
    clip.notes = {a};
    auto repeat = clip; repeat.id = "clip-b"; repeat.startSeconds = 3; repeat.notes[0].id = "note-b";
    repeat.notes[0].midiNote = 72; repeat.notes[0].gain = .81f;
    TrackData track; track.id = "native"; track.pitchAlgorithm = PitchAlgorithm::world; track.clips = {clip, repeat};
    ProjectData project; project.tracks = {track};
    R::RowFacts row; row.key = "carried-syllable";
    row.annotation.regionStartSeconds = .1; row.annotation.regionEndSeconds = .8;
    row.annotation.alignmentSeconds = .22; row.annotation.fixedDurationSeconds = .25;
    // Deliberately include old analysis cuts; source authoring projection may
    // coalesce them, but the candidate's nativeSegments MUST NOT change.
    row.annotation.segments = {{"epoch1", "-", NativeSegmentRole::vowel, .1, .4},
                               {"epoch2", "-", NativeSegmentRole::vowel, .4, .8}};
    row.annotation.relativePitchCents = 700; row.annotation.melodyneAmplitude = .37;
    row.annotation.amplitudeEnvelope = a.amplitudeEnvelope;
    row.members = {{"native", "clip-a", {}, "note-a"}, {"native", "clip-b", {}, "note-b"}};
    R::VariantFacts facts; facts.recording = source; facts.variantKey = "carried-v1";
    facts.explicitPronunciationEvidence = true; facts.rows = {row};
    // Fixture-only fingerprint, NOT a production replacement for PM's complete
    // exact serializer. Include the fixture's musical authority, exclude refs.
    const R::Fingerprint fingerprint = [](const ProjectData& p)
    {
        juce::MemoryOutputStream bytes;
        for (const auto& t : p.tracks)
        {
            bytes.writeString(t.id); bytes.writeInt(int(t.pitchAlgorithm));
            for (const auto& c : t.clips)
            {
                bytes.writeString(c.id); bytes.writeDouble(c.startSeconds); bytes.writeDouble(c.durationSeconds);
                bytes.writeDouble(c.sourceOffsetSeconds); bytes.writeDouble(c.sourceDurationSeconds);
                for (const auto& q : c.sourceTimeMap) { bytes.writeDouble(q.targetSeconds); bytes.writeDouble(q.sourceSeconds); }
                for (const auto& q : c.nativeTrimClock) { bytes.writeDouble(q.targetSeconds); bytes.writeDouble(q.sourceSeconds); }
                bytes.writeBool(c.nativeSourcePitchComplete);
                if (c.nativeSourcePitch) for (const auto& q : *c.nativeSourcePitch)
                { bytes.writeDouble(q.sourceSeconds); bytes.writeFloat(q.midi); bytes.writeFloat(q.withoutVibratoMidi); bytes.writeBool(q.voiced); }
                for (const auto& n : c.notes)
                {
                    bytes.writeString(n.id); bytes.writeString(n.label); bytes.writeFloat(n.midiNote); bytes.writeFloat(n.gain);
                    bytes.writeDouble(n.consonantSeconds); bytes.writeFloat(n.attackSpeed); bytes.writeBool(n.vibratoEnabled);
                    bytes.writeDouble(n.vibratoDepthCents); bytes.writeDouble(n.nativeSourceStartSeconds); bytes.writeDouble(n.nativeSourceEndSeconds);
                    for (const auto& q : n.pitchControlPoints) { bytes.writeDouble(q.timeSeconds); bytes.writeFloat(q.targetMidi); }
                    for (const auto& q : n.contour) { bytes.writeDouble(q.timeSeconds); bytes.writeFloat(q.relativeCents); }
                    for (const auto& q : n.amplitudeEnvelope) { bytes.writeDouble(q.timeSeconds); bytes.writeFloat(q.gainDb); }
                    for (const auto& s : n.nativeSegments) { bytes.writeString(s.id); bytes.writeString(s.alias); bytes.writeDouble(s.sourceStartSeconds); bytes.writeDouble(s.sourceEndSeconds); }
                }
            }
        }
        return juce::SHA256(bytes.getData(), bytes.getDataSize()).toHexString();
    };
    auto plan = R::prepare(project, 17, destination, "registered", {facts}, fingerprint);
    check("explicit_plan_ready_one_physical_copy", plan.ready && plan.recordings.size() == 1 && plan.bindings.size() == 2 && plan.owners.size() == 2);
    if (!plan.ready) { log += plan.diagnostics.joinIntoString("\n"); output.getChildFile("verification.txt").replaceWithText(log); return false; }
    check("prepare_writes_nothing", !plan.publishedFolder.exists() && destination.getNumberOfChildFiles(juce::File::findFilesAndDirectories) == 0);
    const auto& publishedRow = plan.document.audio[0].rows[0];
    check("literal_dash_and_no_global_project_tuning", publishedRow.name == "-" && publishedRow.relativePitchCents == 0
        && publishedRow.melodyneAmplitude == 1 && publishedRow.amplitudeEnvelope.empty() && !publishedRow.vowelSplitsExplicit);
    check("P_not_C_and_explicit_variant_identity", publishedRow.alignmentSeconds == .22 && publishedRow.fixedDurationSeconds == .25
        && plan.variants.size() == 1 && plan.variants[0].variantKey == facts.variantKey);
    ProjectData candidate; juce::String error;
    check("candidate_music_clock_fingerprint_exact", R::makeCandidate(project, plan, fingerprint, candidate, error)
        && fingerprint(candidate) == fingerprint(project) && candidate.tracks[0].clips[0].sourceFile != source
        && candidate.tracks[0].clips[0].notes[0].nativeSegments[0].id == "old-analysis"
        && candidate.tracks[0].clips[1].notes[0].midiNote == 72 && candidate.tracks[0].clips[1].notes[0].gain == .81f
        && candidate.tracks[0].clips[0].notes[0].materialLocalOverride
        && candidate.tracks[0].clips[0].nativeSourcePitch == project.tracks[0].clips[0].nativeSourcePitch);
    auto changed = project; changed.tracks[0].clips[0].notes[0].gain = .99f;
    auto untouched = project;
    check("stale_candidate_refused_output_unchanged", !R::makeCandidate(changed, plan, fingerprint, untouched, error)
        && untouched.tracks[0].clips[0].sourceFile == source);
    auto part = clip; part.id = "part-1"; part.notes.clear();
    ClipData merged; merged.id = "merged"; merged.durationSeconds = 1; merged.parts = {part};
    auto partNote = a; partNote.clipPartId = part.id; merged.notes = {partNote};
    auto partProject = project; partProject.tracks[0].clips = {merged};
    auto partFacts = facts; partFacts.rows[0].members = {{"native", "merged", "part-1", "note-a"}};
    const auto partPlan = R::prepare(partProject, 17, destination, "part", {partFacts}, fingerprint);
    ProjectData partCandidate;
    check("source_part_owner_reference_only", partPlan.ready && R::makeCandidate(partProject, partPlan, fingerprint, partCandidate, error)
        && partCandidate.tracks[0].clips[0].parts[0].sourceFile != source
        && partCandidate.tracks[0].clips[0].parts[0].nativeSourcePitch == part.nativeSourcePitch
        && partCandidate.tracks[0].clips[0].notes[0].clipPartId == "part-1");
    auto unresolved = facts; unresolved.explicitPronunciationEvidence = false;
    check("pitch_epochs_not_pronunciations", !R::prepare(project, 17, destination, "unknown", {unresolved}, fingerprint).ready);
    auto duplicate = facts; duplicate.variantKey = "independent-v2"; duplicate.rows[0].members.clear();
    const auto variants = R::prepare(project, 17, destination, "variants", {facts, duplicate}, fingerprint);
    check("independent_variants_over_one_copy", variants.ready && variants.recordings.size() == 1 && variants.document.audio.size() == 2
        && variants.document.audio[0].id != variants.document.audio[1].id
        && variants.document.audio[0].relativePath == variants.document.audio[1].relativePath);
    const auto stageWithReport = [&](const R::Plan& prepared)
    {
        auto staged = R::stage(prepared, error);
        if (!staged)
        {
            const auto diagnostic = "stage_error[" + prepared.publishedFolder.getFileName() + "]=" + error;
            log += diagnostic + "\n"; std::cout << diagnostic << std::endl;
        }
        return staged;
    };
    auto variantLease = stageWithReport(variants);
    check("multiple_variants_stage_one_physical_audio", variantLease != nullptr);
    if (variantLease)
    {
        const auto manifest = variantLease->currentFolder().getChildFile("otto.ini");
        const auto copied = variantLease->currentFolder().getChildFile(variants.recordings[0].copiedRelativePath);
        std::vector<SampleRegionSetting> selected;
        check("variant_read_requires_stable_audio_ID", !OttoMaterialIO::loadAudio(manifest, copied, selected, error));
        bool explicitReads = true;
        for (const auto& audio : variants.document.audio)
            explicitReads &= OttoMaterialIO::loadAudio(manifest, copied, selected, error, nullptr, audio.id)
                && selected.size() == 1 && selected[0].materialAudioId == audio.id;
        check("both_independent_variants_read_by_ID", explicitReads);
        std::vector<OttoMaterialIO::Audio> listed;
        check("variant_inventory_preserves_both", OttoMaterialIO::listAudioVariants(manifest, copied, listed, error) && listed.size() == 2);
        check("variant_lease_cleanup", variantLease->rollback(error));
    }
    auto unnamed = facts; unnamed.rows[0].annotation.vowelSplitsExplicit = true;
    check("reserved_internal_vowel_dash_still_refused", !R::prepare(project, 17, destination, "unnamed", {unnamed}, fingerprint).ready);
    auto named = facts; named.rows[0].annotation.vowelSplitsExplicit = true;
    named.rows[0].annotation.segments[0].alias = "an"; named.rows[0].annotation.segments[0].stretchWeight = 2;
    named.rows[0].annotation.segments[1].alias = "ng"; named.rows[0].annotation.segments[1].stretchWeight = 3;
    const auto namedPlan = R::prepare(project, 17, destination, "named", {named}, fingerprint);
    bool namesRetained = namedPlan.ready;
    if (namesRetained)
    {
        bool an = false, ng = false;
        for (const auto& segment : namedPlan.document.audio[0].rows[0].segments)
        { an |= segment.alias == "an" && segment.stretchWeight == 2; ng |= segment.alias == "ng" && segment.stretchWeight == 3; }
        namesRetained = an && ng && namedPlan.document.audio[0].rows[0].name == "-";
    }
    check("explicit_named_weighted_V_retained", namesRetained);
    auto pending = project; pending.tracks[0].clips[0].nativeSourcePitchPending = true;
    check("pending_acoustic_source_refused_not_reanalysed", !R::prepare(pending, 17, destination, "pending", {facts}, fingerprint).ready);
    auto cancelledPlan = R::prepare(project, 17, destination, "cancelled", {facts}, fingerprint, [] { return true; });
    check("prepare_cancel_no_publication", !cancelledPlan.ready && !cancelledPlan.publishedFolder.exists());
    check("stage_cancel_no_outputs", !R::stage(plan, error, [] { return true; })
        && destination.getNumberOfChildFiles(juce::File::findFilesAndDirectories) == 0);
    auto lease = stageWithReport(plan);
    check("stage_shared_codec_and_original_bytes", lease && !plan.publishedFolder.exists()
        && juce::SHA256(source).toHexString() == originalHash);
    if (lease)
    {
        const auto staging = lease->currentFolder();
        OttoMaterialIO::Document readback;
        check("staged_document_readback", OttoMaterialIO::loadDocument(staging.getChildFile("otto.ini"), readback, error)
            && readback.audio.size() == 1 && readback.audio[0].rows[0].name == "-");
        check("staged_copy_exact_sha_and_extension", juce::SHA256(staging.getChildFile(plan.recordings[0].copiedRelativePath)).toHexString() == originalHash
            && plan.recordings[0].copiedRelativePath.endsWith(".WAV"));
#if JUCE_WINDOWS
        check("atomic_publish", lease->publish(error) && plan.publishedFolder.isDirectory() && !staging.exists());
        check("owned_published_compensating_rollback", lease->rollback(error) && !plan.publishedFolder.exists());
#else
        check("unsupported_platform_publication_explicit", !lease->publish(error));
        check("owned_staged_rollback", lease->rollback(error));
#endif
    }
    const auto sentinel = destination.getChildFile("registered"); sentinel.createDirectory();
    sentinel.getChildFile("keep.txt").replaceWithText("do not delete");
    check("existing_destination_never_removed", !R::stage(plan, error)
        && sentinel.getChildFile("keep.txt").loadFileAsString() == "do not delete");
    // Different physical files with identical bytes are NOT collapsed.
    const auto elsewhere = originals.getChildFile("elsewhere"); elsewhere.createDirectory();
    const auto source2 = elsewhere.getChildFile("voice.WAV"); source.copyFileTo(source2);
    auto two = project; two.tracks[0].clips[1].sourceFile = source2;
    auto facts1 = facts; facts1.rows[0].members.resize(1);
    auto facts2 = facts; facts2.recording = source2; facts2.rows[0].members.erase(facts2.rows[0].members.begin());
    auto twoPlan = R::prepare(two, 17, destination, "two", {facts1, facts2}, fingerprint);
    check("equal_SHA_distinct_paths_and_collision_names", twoPlan.ready && twoPlan.recordings.size() == 2
        && twoPlan.recordings[0].sha256 == twoPlan.recordings[1].sha256
        && twoPlan.recordings[0].copiedRelativePath != twoPlan.recordings[1].copiedRelativePath
        && twoPlan.recordings[1].copiedRelativePath.endsWith(".WAV"));
    auto tampered = twoPlan; tampered.recordings[0].copiedRelativePath = "../escape.WAV";
    check("tampered_plan_path_escape_refused", !R::stage(tampered, error) && !output.getChildFile("escape.WAV").exists());
    if (twoPlan.ready)
    {
        auto interrupted = stageWithReport(twoPlan);
        if (interrupted)
        {
            const auto owned = interrupted->currentFolder();
            check("cancel_before_publication", !interrupted->publish(error, [] { return true; }) && !twoPlan.publishedFolder.exists());
            interrupted.reset(); check("destructor_owned_failure_isolation", !owned.exists());
        }
        else check("second_stage", false);
    }
    const auto racePlan = R::prepare(project, 17, destination, "race", {facts}, fingerprint);
    auto raceLease = stageWithReport(racePlan);
    if (raceLease)
    {
        racePlan.publishedFolder.createDirectory(); racePlan.publishedFolder.getChildFile("sentinel.txt").replaceWithText("existing");
        check("publication_race_does_not_replace_existing_folder", !raceLease->publish(error)
            && racePlan.publishedFolder.getChildFile("sentinel.txt").loadFileAsString() == "existing");
        check("rollback_never_removes_racing_destination", raceLease->rollback(error)
            && racePlan.publishedFolder.getChildFile("sentinel.txt").loadFileAsString() == "existing");
    }
    else check("publication_race_stage", false);
    const auto corruptPlan = R::prepare(project, 17, destination, "corrupt", {facts}, fingerprint);
    auto corruptLease = stageWithReport(corruptPlan);
    if (corruptLease)
    {
        corruptLease->currentFolder().getChildFile(corruptPlan.recordings[0].copiedRelativePath).appendText("tampered");
        check("changed_copy_refuses_publication", !corruptLease->publish(error) && !corruptPlan.publishedFolder.exists());
        check("failed_copy_owned_cleanup", corruptLease->rollback(error));
    }
    else check("copy_tamper_stage", false);
    check("original_still_exact", juce::SHA256(source).toHexString() == originalHash);
    output.getChildFile("verification.txt").replaceWithText(log);
    return ok;
}
}
