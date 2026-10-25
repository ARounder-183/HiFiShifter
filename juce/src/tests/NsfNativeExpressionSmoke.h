#pragma once
#include "../AudioEngine.h"
#include "../backend/NativeExpression.h"
#include "../backend/NsfHifiganRenderer.h"
#include <cmath>
#include <cstring>
#include <iostream>

namespace hachi
{
inline bool runNsfNativeExpressionSmoke(const juce::File& output,
    const juce::File& modelDirectory = {}, const juce::File& recording = {})
{
    if (!output.createDirectory()) return false;
    bool ok = true; int checks = 0; juce::Array<juce::var> results;
    auto report = new juce::DynamicObject();
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass; ++checks; std::cout << name << '=' << pass << std::endl;
        auto row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("pass", pass);
        results.add(juce::var(row));
    };
    const auto identical = [](const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b)
    {
        if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples()) return false;
        for (int c = 0; c < a.getNumChannels(); ++c)
            if (std::memcmp(a.getReadPointer(c), b.getReadPointer(c), a.getNumSamples() * sizeof(float)) != 0) return false;
        return true;
    };
    const auto finite = [](const juce::AudioBuffer<float>& audio)
    {
        if (audio.getNumSamples() <= 0 || audio.getNumChannels() <= 0) return false;
        for (int c = 0; c < audio.getNumChannels(); ++c)
            for (int i = 0; i < audio.getNumSamples(); ++i)
                if (!std::isfinite(audio.getSample(c, i))) return false;
        return true;
    };
    const auto difference = [](const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b)
    {
        if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples()) return -1.0;
        double power = 0; std::size_t samples = 0;
        for (int c = 0; c < a.getNumChannels(); ++c)
            for (int i = 0; i < a.getNumSamples(); ++i)
            { const double d = a.getSample(c, i) - b.getSample(c, i); power += d * d; ++samples; }
        return samples > 0 ? std::sqrt(power / samples) : 0.0;
    };
    constexpr double rate = 48000, period = 5;
    constexpr int samples = 24000;
    juce::AudioBuffer<float> source(2, samples);
    for (int c = 0; c < source.getNumChannels(); ++c)
        for (int i = 0; i < samples; ++i)
        {
            const auto time = i / rate;
            const auto wave = .2 * std::sin(juce::MathConstants<double>::twoPi * 220 * time)
                + .055 * std::sin(juce::MathConstants<double>::twoPi * 660 * time)
                + .035 * std::sin(juce::MathConstants<double>::twoPi * 3300 * time);
            source.setSample(c, i, static_cast<float>(wave * (c == 0 ? 1.0 : .7)));
        }
    const auto frames = static_cast<std::size_t>(std::ceil(samples / rate * 1000 / period)) + 1;
    const std::vector<float> midi(frames, 57.0f);
    backend::UtauNoteRenderSpec defaults;
    check("native_note_spec_neutrals_are_zero", defaults.nativeFormantSemitones == 0 && defaults.nativeTension == 0 && defaults.nativeBreath == 0);
    auto neutral = source;
    backend::applyNsfUtauNativeExpression(neutral, rate, period, midi, defaults);
    check("neutral_native_voicebank_expression_is_byte_identical", identical(source, neutral));
    auto sharedNeutral = source;
    backend::applyNativeExpression(sharedNeutral, rate, period, midi, {}, std::vector<float>(frames, 0), std::vector<float>(frames, 0));
    check("extracted_native_file_core_neutral_is_byte_identical", identical(source, sharedNeutral));
    auto emptyCurves = source;
    backend::applyNativeExpression(emptyCurves, rate, period, midi, {}, {}, {});
    check("empty_native_expression_curves_are_byte_identical", identical(source, emptyCurves));
    auto silent = source; silent.clear(); const auto silentBefore = silent;
    backend::applyNsfUtauNativeExpression(silent, rate, period, midi, defaults);
    check("silent_neutral_pcm_is_unchanged", identical(silentBefore, silent));

    backend::UtauNoteRenderSpec expression; expression.nativeTension = .7f; expression.nativeBreath = .45f;
    expression.gain = .125f; expression.flags = "g25Ht15t100";
    auto fromVoicebank = source, fromNativeFile = source;
    backend::applyNsfUtauNativeExpression(fromVoicebank, rate, period, midi, expression);
    backend::applyNativeExpression(fromNativeFile, rate, period, midi, {},
        std::vector<float>(frames, expression.nativeTension), std::vector<float>(frames, expression.nativeBreath));
    check("voicebank_expression_matches_native_file_core_exactly", identical(fromVoicebank, fromNativeFile));
    check("nonneutral_native_expression_changes_finite_pcm", finite(fromVoicebank) && difference(source, fromVoicebank) > 1.e-5);
    auto withoutGainOrFlags = expression; withoutGainOrFlags.gain = 1; withoutGainOrFlags.flags.clear();
    auto noDoubleGain = source;
    backend::applyNsfUtauNativeExpression(noDoubleGain, rate, period, midi, withoutGainOrFlags);
    check("native_expression_does_not_reapply_gain_or_flag_dsp", identical(fromVoicebank, noDoubleGain));
    auto negativeTension = defaults; negativeTension.nativeTension = -.6f;
    auto negative = source, negativeExpected = source;
    backend::applyNsfUtauNativeExpression(negative, rate, period, midi, negativeTension);
    backend::applyNativeExpression(negativeExpected, rate, period, midi, {}, std::vector<float>(frames, -.6f), std::vector<float>(frames, 0));
    check("negative_tension_uses_same_native_units", identical(negative, negativeExpected) && difference(source, negative) > 1.e-5);
    auto airNote = defaults; airNote.nativeBreath = .8f; auto air = source;
    backend::applyNsfUtauNativeExpression(air, rate, period, midi, airNote);
    check("native_breath_is_effective_without_any_flag", finite(air) && difference(source, air) > 1.e-5);
    auto tenseSilence = silentBefore;
    backend::applyNsfUtauNativeExpression(tenseSilence, rate, period, midi, expression);
    check("expression_does_not_invent_sound_from_silence", identical(tenseSilence, silentBefore));

    ProjectData project; TrackData track; track.id = "nsf-expression"; track.compose = true;
    track.pitchAlgorithm = PitchAlgorithm::nsfHifigan; track.nativeNsfAudio = false;
    track.voicebankDirectory = output; track.outputEngine = UtauOutputEngine::pcNsfHifigan;
    track.utauGlobalFlags = "Ht5";
    ClipData clip; clip.id = "expression-clip"; clip.durationSeconds = .5;
    NoteData note; note.id = "native-controls"; note.label = "a"; note.durationSeconds = .5;
    note.midiNote = 57; note.gain = .8f; note.formantSemitones = 3.5f; note.tension = .37f; note.breath = .62f;
    note.utauFlags = "g15t10"; clip.notes = {note}; track.clips = {clip}; project.tracks = {track};
    const auto mapped = AudioEngine::diagnosticUtauRequestNotes(project, clip.id);
    check("make_utau_request_maps_native_controls_without_unit_conversion", mapped.size() == 1
        && mapped[0].nativeFormantSemitones == note.formantSemitones && mapped[0].nativeTension == note.tension
        && mapped[0].nativeBreath == note.breath);
    check("native_control_mapping_does_not_rewrite_flags_gain_or_pitch", mapped.size() == 1
        && mapped[0].flags == note.utauFlags + track.utauGlobalFlags && mapped[0].gain == note.gain && mapped[0].midiNote == note.midiNote);
    const auto key = AudioEngine::diagnosticUtauRenderKey(project, clip.id, UtauOutputEngine::pcNsfHifigan);
    auto changed = project; changed.tracks[0].clips[0].notes[0].formantSemitones += 1;
    check("formant_invalidates_whole_clip_cache", key != AudioEngine::diagnosticUtauRenderKey(changed, clip.id, UtauOutputEngine::pcNsfHifigan));
    changed = project; changed.tracks[0].clips[0].notes[0].tension += .1f;
    check("tension_invalidates_whole_clip_cache", key != AudioEngine::diagnosticUtauRenderKey(changed, clip.id, UtauOutputEngine::pcNsfHifigan));
    changed = project; changed.tracks[0].clips[0].notes[0].breath += .1f;
    check("breath_invalidates_whole_clip_cache", key != AudioEngine::diagnosticUtauRenderKey(changed, clip.id, UtauOutputEngine::pcNsfHifigan));

    report->setProperty("modelDirectory", modelDirectory.getFullPathName());
    report->setProperty("modelTestRequested", modelDirectory != juce::File{});
    if (modelDirectory != juce::File{})
    {
        const bool available = backend::NsfHifiganRenderer::modelAvailable(modelDirectory);
        check("requested_real_nsf_model_is_available", available);
        if (available)
        {
            auto input = recording;
            bool sourceReady = input.existsAsFile();
            if (input == juce::File{})
            {
                input = output.getChildFile("expression-source.wav");
                auto stream = input.createOutputStream();
                if (stream) { stream->setPosition(0); stream->truncate(); }
                juce::WavAudioFormat wav;
                std::unique_ptr<juce::AudioFormatWriter> writer(stream ? wav.createWriterFor(stream.get(), rate, 1, 24, {}, 0) : nullptr);
                if (writer) { stream.release(); sourceReady = writer->writeFromAudioSampleBuffer(source, 0, samples); }
            }
            check("real_model_source_fixture_readable", sourceReady);
            juce::AudioFormatManager formats; formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(sourceReady ? formats.createReaderFor(input) : nullptr);
            if (reader)
            {
                backend::NsfUtauSampleTiming timing;
                timing.fileSeconds = reader->lengthInSamples / reader->sampleRate;
                timing.endSeconds = timing.fileSeconds; timing.consonantSeconds = std::min(.06, timing.fileSeconds * .2);
                timing.preutteranceSeconds = .02;
                const auto plan = backend::buildNsfUtauNotePlan(timing, .2, .25, 1, .025);
                const std::vector<backend::NsfUtauPitchPoint> pitch{{0, 0}, {.25, 0}};
                backend::OrtExecutionConfig execution;
                auto baseline = backend::synthesizeNsfUtauNote(input, plan, 57, pitch, modelDirectory, execution);
                check("real_model_default_note_renders", baseline.usedModel && finite(baseline.audio));
                report->setProperty("realModelError", baseline.error);
                if (baseline.usedModel)
                {
                    const auto target = backend::buildNsfUtauTargetMidi(57, pitch, period, plan.outputSeconds, plan.soundStartOffsetSeconds);
                    auto sameDecode = baseline.audio;
                    backend::applyNsfUtauNativeExpression(sameDecode, baseline.sampleRate, period, target, defaults);
                    check("real_decoded_pcm_neutral_native_stage_is_byte_identical", identical(baseline.audio, sameDecode));
                    auto explicitZero = backend::synthesizeNsfUtauNote(input, plan, 57, pitch, modelDirectory, execution, {}, &defaults);
                    check("real_model_explicit_zero_preserves_render_shape", explicitZero.usedModel && finite(explicitZero.audio)
                        && explicitZero.audio.getNumSamples() == baseline.audio.getNumSamples() && explicitZero.sampleRate == baseline.sampleRate);
                    // A vocoder may use a fresh excitation phase on each call.
                    // Report cross-inference equality honestly; do not weaken a
                    // byte test or attribute random phase differences to controls.
                    report->setProperty("separateNeuralRunsZeroPcmIdentical", identical(baseline.audio, explicitZero.audio));
                    report->setProperty("separateNeuralRunsZeroRmsDifference", difference(baseline.audio, explicitZero.audio));
                    auto modelExpression = baseline.audio, modelExpected = baseline.audio;
                    backend::applyNsfUtauNativeExpression(modelExpression, baseline.sampleRate, period, target, expression);
                    const auto count = std::max(target.size(), static_cast<std::size_t>(std::ceil(
                        baseline.audio.getNumSamples() / baseline.sampleRate * 1000 / period)) + 1);
                    backend::applyNativeExpression(modelExpected, baseline.sampleRate, period, target, {},
                        std::vector<float>(count, expression.nativeTension), std::vector<float>(count, expression.nativeBreath));
                    check("real_decoded_expression_matches_native_file_core_bytes", identical(modelExpression, modelExpected)
                        && finite(modelExpression) && difference(baseline.audio, modelExpression) > 1.e-7);
                    auto formantNote = defaults; formantNote.nativeFormantSemitones = 5;
                    auto shifted = backend::synthesizeNsfUtauNote(input, plan, 57, pitch, modelDirectory, execution, {}, &formantNote);
                    check("real_model_native_formant_renders_finite_pcm", shifted.usedModel && finite(shifted.audio)
                        && shifted.audio.getNumSamples() == baseline.audio.getNumSamples());
                    report->setProperty("nativeFormantRmsDifference", difference(baseline.audio, shifted.audio));
                    // This is a live end-to-end output check, not a promise of
                    // phase-identical inference or a perceptual formant metric.
                    check("real_model_nonzero_formant_has_nonidentical_output", shifted.usedModel && !identical(baseline.audio, shifted.audio));
                    auto combinedNote = expression; combinedNote.nativeFormantSemitones = 3; combinedNote.flags = "g10";
                    auto combined = backend::synthesizeNsfUtauNote(input, plan, 57, pitch, modelDirectory, execution, {}, &combinedNote);
                    check("native_controls_compose_with_flags_in_real_model", combined.usedModel && finite(combined.audio)
                        && combined.audio.getNumSamples() == baseline.audio.getNumSamples());
                }
            }
            else check("source_fixture_audio_reader_created", false);
        }
    }
    report->setProperty("ok", ok); report->setProperty("checks", checks); report->setProperty("results", juce::var(results));
    const bool saved = output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout << "nsf_native_expression_ok=" << (ok && saved) << "; checks=" << checks << std::endl;
    return ok && saved;
}
}
