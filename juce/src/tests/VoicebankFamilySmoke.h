#pragma once
#include "../AudioEngine.h"
#include "../backend/UtauRenderer.h"
#include "../backend/NsfHifiganRenderer.h"
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool voicebankFamilySmoke(const juce::File& output)
{
    using namespace backend;
    using Family = RequestedVoicebankFamily;
    bool ok = true; int checks = 0; juce::Array<juce::var> failures;
    const auto check = [&](const juce::String& name, bool pass)
    {
        ++checks; ok &= pass; if (!pass) failures.add(name);
        std::cout << "voicebank_family_" << name << "=" << pass << std::endl;
    };
    check("output_created", output.createDirectory().wasOk());
    const auto root = output.getChildFile("fixtures-" + juce::Uuid().toString());
    const auto plain = root.getChildFile("plain-utau"), mixed = root.getChildFile("mixed"), dsOnly = root.getChildFile("ds-only");
    check("fixtures_created", plain.createDirectory().wasOk() && mixed.createDirectory().wasOk() && dsOnly.createDirectory().wasOk());
    const auto writeBank = [&](const juce::File& bank)
    {
        juce::AudioBuffer<float> audio(1, 4410);
        for (int i = 0; i < audio.getNumSamples(); ++i)
            audio.setSample(0, i, static_cast<float>(.1 * std::sin(juce::MathConstants<double>::twoPi * 440 * i / 44100.0)));
        juce::WavAudioFormat wav;
        auto stream = bank.getChildFile("a.wav").createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(stream ? wav.createWriterFor(stream.release(),44100,1,16,{},0) : nullptr);
        const auto written = writer && writer->writeFromAudioSampleBuffer(audio, 0, audio.getNumSamples());
        writer.reset();
        return written && bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,0,-100,0,0\n");
    };
    check("physical_utau_banks_written", writeBank(plain) && writeBank(mixed));
    check("ds_identity_fixtures_written", mixed.getChildFile("dsconfig.yaml").replaceWithText("name: mixed-format-fixture\n")
        && dsOnly.getChildFile("dsconfig.yaml").replaceWithText("name: deliberately-incomplete-no-inference-fixture\n"));
    juce::MemoryBlock sourceBefore, sourceAfter; plain.getChildFile("a.wav").loadFileAsData(sourceBefore);
    UtauRenderRequest request; request.voicebankDirectory = plain; request.targetDurationSeconds = .12;
    UtauNoteRenderSpec note; note.alias = "a"; note.durationSeconds = .1; note.midiNote = 69;
    request.notes = {note};
    const auto legacyPlain = resolveRequestedVoicebankFamily(request, true);
    check("legacy_plain_route", legacyPlain.family == Family::utau && legacyPlain.warning.isEmpty() && legacyPlain.cacheDiscriminator.isEmpty());
    auto typed = request; typed.requestedFamily = Family::utau;
    const auto matching = resolveRequestedVoicebankFamily(typed, true);
    check("matching_utau_keeps_historic_cache_identity", matching.family == Family::utau && matching.warning.isEmpty() && matching.cacheDiscriminator.isEmpty());
    const auto oldAudio = UtauRenderer::render(request);
    const auto typedAudio = UtauRenderer::render(typed);
    const auto samePcm = [](const UtauRenderResult& a, const UtauRenderResult& b)
    {
        if (a.sampleRate != b.sampleRate || a.buffer.getNumChannels() != b.buffer.getNumChannels()
            || a.buffer.getNumSamples() != b.buffer.getNumSamples()) return false;
        for (int c = 0; c < a.buffer.getNumChannels(); ++c)
            for (int s = 0; s < a.buffer.getNumSamples(); ++s)
                if (a.buffer.getSample(c, s) != b.buffer.getSample(c, s)) return false;
        return true;
    };
    check("matching_utau_actual_pcm_unchanged", oldAudio.buffer.getNumSamples() > 0 && typedAudio.buffer.getNumSamples() > 0
        && oldAudio.backend == typedAudio.backend && oldAudio.warning == typedAudio.warning && samePcm(oldAudio, typedAudio));
    typed.voicebankDirectory = mixed;
    const auto mixedRoute = resolveRequestedVoicebankFamily(typed, true);
    check("explicit_utau_mixed_bank_never_DS", mixedRoute.family == Family::utau && mixedRoute.warning.isEmpty() && mixedRoute.cacheDiscriminator.isNotEmpty());
    const auto mixedAudio = UtauRenderer::render(typed);
    check("mixed_valid_oto_actual_utau_audio", mixedAudio.buffer.getNumSamples() > 0 && mixedAudio.warning.isEmpty()
        && !mixedAudio.backend.containsIgnoreCase("diffsinger") && !mixedAudio.backend.containsIgnoreCase("piano") && samePcm(oldAudio, mixedAudio));
    auto copied = typed;
    check("request_copy_retains_explicit_family", copied.requestedFamily == Family::utau);
    juce::Array<juce::var> aliasDiagnostics;
    const auto traceAlias = [&](const juce::String& scenario, const juce::String& inputLyric,
                                const UtauRenderRequest& req, const UtauRenderResult& result)
    {
        const auto& n = req.notes[0];
        const auto resolved = UtauRenderer::resolveVoiceSample(req.voicebankDirectory, n.alias, n.midiNote,
            n.consonantVelocity, req.fourRegion, req.consonantClasses, n.stpSeconds,
            n.preutteranceOverrideEnabled, n.preutteranceSeconds, n.overlapOverrideEnabled, n.overlapSeconds,
            &n.oto, n.aliasIsResolved);
        const auto timing = UtauRenderer::sampleTiming(req.voicebankDirectory, n.alias, n.midiNote,
            n.consonantVelocity, req.fourRegion, req.consonantClasses);
        auto* row = new juce::DynamicObject;
        row->setProperty("scenario", scenario); row->setProperty("inputLyric", inputLyric);
        row->setProperty("selectedAlias", n.alias); row->setProperty("aliasIsResolved", n.aliasIsResolved);
        row->setProperty("canonicalFound", resolved.found);
        row->setProperty("resolvedAlias", resolved.found && timing ? timing->resolvedAlias : juce::String());
        row->setProperty("resolvedSource", resolved.file.getFullPathName());
        row->setProperty("backend", result.backend); row->setProperty("warning", result.warning);
        row->setProperty("pcmSamples", result.buffer.getNumSamples()); row->setProperty("sampleRate", result.sampleRate);
        const juce::var data(row); aliasDiagnostics.add(data);
        std::cout << "voicebank_family_alias_diagnostic=" << juce::JSON::toString(data, true) << std::endl;
    };
    copied.notes[0].alias = "missing";
    const auto minimalBank = UtauRenderer::render(copied);
    traceAlias("legitimate-single-sample-remap", "missing", copied, minimalBank);
    check("single_sample_canonical_remap_is_not_rejected", minimalBank.buffer.getNumSamples() > 0
        && minimalBank.warning.isEmpty() && !minimalBank.backend.containsIgnoreCase("piano") && samePcm(oldAudio, minimalBank));
    // Two entries disable the canonical one-sample shorthand. This now really
    // is an unresolved source request, rather than a valid minimal-bank alias.
    const auto multiAlias = root.getChildFile("mixed-two-aliases");
    check("two_alias_fixture_written", multiAlias.createDirectory().wasOk() && writeBank(multiAlias)
        && multiAlias.getChildFile("oto.ini").replaceWithText("a.wav=a,0,0,-100,0,0\na.wav=b,0,0,-100,0,0\n")
        && multiAlias.getChildFile("dsconfig.yaml").replaceWithText("name: mixed-two-aliases\n"));
    copied.voicebankDirectory = multiAlias;
    const auto missingMixed = UtauRenderer::render(copied);
    traceAlias("truly-unresolved-two-alias-bank", "missing", copied, missingMixed);
    check("mixed_unresolved_alias_no_piano_fallback", missingMixed.buffer.getNumSamples() == 0
        && missingMixed.warning.contains("cannot resolve alias") && missingMixed.backend.isEmpty());
    auto boundSource = typed; boundSource.notes[0].aliasIsResolved = true;
    const auto independentLyric = juce::String("edited score lyric, source binding remains a");
    const auto boundAudio = UtauRenderer::render(boundSource);
    traceAlias("explicit-source-binding-independent-of-lyric", independentLyric, boundSource, boundAudio);
    check("valid_bound_source_survives_independent_lyric_edit", boundAudio.buffer.getNumSamples() > 0
        && boundAudio.warning.isEmpty() && samePcm(oldAudio, boundAudio));
    boundSource.notes[0].alias = "missing-bound-source";
    const auto missingBound = UtauRenderer::render(boundSource);
    traceAlias("truly-missing-explicit-source-binding", independentLyric, boundSource, missingBound);
    check("missing_bound_source_never_uses_single_sample_shortcut", missingBound.buffer.getNumSamples() == 0
        && missingBound.warning.contains("cannot resolve alias") && missingBound.backend.isEmpty());
    auto legacyMixed = typed; legacyMixed.requestedFamily = Family::legacyAuto;
    const auto historicalDs = resolveRequestedVoicebankFamily(legacyMixed);
    check("legacy_auto_DS_detection_preserved_without_inference", historicalDs.family == Family::diffSinger && historicalDs.warning.isEmpty() && historicalDs.cacheDiscriminator.isEmpty());
    auto typedDs = legacyMixed; typedDs.requestedFamily = Family::diffSinger;
    const auto matchingDs = resolveRequestedVoicebankFamily(typedDs);
    check("matching_DS_cache_identity_unchanged", matchingDs.family == Family::diffSinger && matchingDs.warning.isEmpty() && matchingDs.cacheDiscriminator.isEmpty());
    typedDs.voicebankDirectory = plain; typedDs.chineseCvvc = true;
    const auto blockedDsIdentity = resolveRequestedVoicebankFamily(typedDs);
    check("blocked_old_DS_to_UTAU_fallback_has_distinct_cache_identity", blockedDsIdentity.family == Family::diffSinger
        && blockedDsIdentity.warning.isNotEmpty() && blockedDsIdentity.cacheDiscriminator.isNotEmpty());
    int callbacks = 0;
    typedDs.notePiece = [&](size_t, const auto&, double, double, const auto&, const auto&) { ++callbacks; };
    const auto wrongDs = UtauRenderer::render(typedDs);
    check("explicit_DS_with_UTAU_bank_no_classic_or_piano", wrongDs.buffer.getNumSamples() == 0 && wrongDs.backend.isEmpty()
        && wrongDs.warning.contains("DiffSinger track requires") && callbacks == 0);
    typedDs.voicebankDirectory = root.getChildFile("nonexistent");
    const auto noDs = UtauRenderer::render(typedDs);
    check("explicit_DS_missing_bank_fails_before_CVVC", noDs.buffer.getNumSamples() == 0 && noDs.warning.contains("DiffSinger track requires") && callbacks == 0);
    typedDs.voicebankDirectory = dsOnly;
    const auto noNsfDs = renderNsfUtauPhrase(typedDs, {}, {});
    check("DS_cannot_enter_NSF_before_model_check", noNsfDs.buffer.getNumSamples() == 0 && noNsfDs.warning.contains("DiffSinger track cannot use"));
    auto incompatible = typed; incompatible.voicebankDirectory = dsOnly; incompatible.notes[0].alias = "";
    const auto builds = UtauRenderer::diagnosticIndexBuilds();
    const auto scheduling = resolveRequestedVoicebankFamily(incompatible);
    check("scheduler_selection_does_not_load_bank_index", scheduling.family == Family::utau && UtauRenderer::diagnosticIndexBuilds() == builds);
    const auto noSamples = UtauRenderer::render(incompatible);
    check("DS_only_UTAU_fails_before_empty_lyric_piano", noSamples.buffer.getNumSamples() == 0
        && noSamples.warning.contains("DiffSinger-only bank") && noSamples.backend.isEmpty());
    const auto noSamplesNsf = renderNsfUtauPhrase(incompatible, {}, {});
    check("DS_only_NSF_fails_before_model_or_fallback", noSamplesNsf.buffer.getNumSamples() == 0 && noSamplesNsf.warning.contains("DiffSinger-only bank"));
    auto legacyNoBank = request; legacyNoBank.voicebankDirectory = root.getChildFile("no-bank"); legacyNoBank.notes[0].alias = "";
    const auto piano = UtauRenderer::render(legacyNoBank);
    check("legacy_auto_empty_bank_piano_unchanged", piano.buffer.getNumSamples() > 0 && piano.backend.containsIgnoreCase("piano"));

    TrackData track; track.id="family-track"; track.pitchAlgorithm=PitchAlgorithm::utau; track.trackType=TrackType::utau; track.voicebankDirectory=plain;
    ClipData clip; clip.id="family-clip"; clip.durationSeconds=.12;
    NoteData scoreNote; scoreNote.id="family-note"; scoreNote.label="a"; scoreNote.midiNote=69; scoreNote.durationSeconds=.1;
    clip.notes={scoreNote}; track.clips={clip}; ProjectData project; project.tracks={track};
    const auto key = AudioEngine::diagnosticUtauRenderKey(project, clip.id);
    check("AE_actual_builder_carries_UTAU", AudioEngine::diagnosticUtauRequestedFamily(project, clip.id) == Family::utau);
    project.tracks[0].trackType.reset();
    check("matching_legacy_UTAU_cache_bytes_unchanged", AudioEngine::diagnosticUtauRenderKey(project,clip.id)==key);
    project.tracks[0].trackType=TrackType::diffSinger; project.tracks[0].utauMode=UtauMode::mou; project.tracks[0].voicebankDirectory=mixed;
    check("AE_actual_builder_carries_DS", AudioEngine::diagnosticUtauRequestedFamily(project,clip.id)==Family::diffSinger);
    const auto dsKey=AudioEngine::diagnosticUtauRenderKey(project,clip.id);
    project.tracks[0].trackType.reset();
    check("matching_legacy_DS_cache_bytes_unchanged", AudioEngine::diagnosticUtauRenderKey(project,clip.id)==dsKey);
    project.tracks[0].trackType=TrackType::hachi; project.tracks[0].pitchAlgorithm=PitchAlgorithm::nsfHifigan; project.tracks[0].nativeNsfAudio=true;
    const auto hachiKey=AudioEngine::diagnosticUtauRenderKey(project,clip.id);
    project.tracks[0].trackType.reset();
    check("native_file_family_cache_unchanged", AudioEngine::diagnosticUtauRenderKey(project,clip.id)==hachiKey);
    project.tracks[0].trackType=TrackType::diffSinger; project.tracks[0].pitchAlgorithm=PitchAlgorithm::utau; project.tracks[0].voicebankDirectory=plain;
    const auto fingerprintBefore=project.tracks[0].clips[0].notes[0].label;
    check("family_checks_do_not_mutate_score", AudioEngine::diagnosticUtauRequestedFamily(project,clip.id)==Family::diffSinger
        && project.tracks[0].trackType==TrackType::diffSinger && project.tracks[0].clips[0].notes[0].label==fingerprintBefore);
    plain.getChildFile("a.wav").loadFileAsData(sourceAfter);
    check("source_PCM_not_modified", sourceBefore==sourceAfter);
    auto* report=new juce::DynamicObject;
    report->setProperty("ok",ok); report->setProperty("checks",checks); report->setProperty("failures",failures);
    report->setProperty("aliasDiagnostics", aliasDiagnostics);
    report->setProperty("scope","typed-family guards, real generated UTAU PCM; no actual DS model inference or real-project acceptance implied");
    const auto written=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok && written;
}
}
