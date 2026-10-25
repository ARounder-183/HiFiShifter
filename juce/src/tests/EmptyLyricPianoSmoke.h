#pragma once
#include "../AudioEngine.h"
#include "../backend/NsfHifiganRenderer.h"
#include <iostream>

namespace hachi
{
inline bool emptyLyricPianoSmoke(const juce::File& folder)
{
    using namespace backend;
    bool ok = true;
    int checks = 0;
    const auto check = [&](const juce::String& name, bool pass)
    { ++checks; ok &= pass; std::cout << name << "=" << pass << std::endl; };
    check("output_directory", folder.createDirectory().wasOk());
    const auto bank = folder.getChildFile("bank");
    bank.createDirectory();
    constexpr double rate = 44100;
    juce::AudioBuffer<float> tone(1, 88200);
    for (int i = 0; i < tone.getNumSamples(); ++i)
        tone.setSample(0, i, 0.2f * static_cast<float>(std::sin(i * juce::MathConstants<double>::twoPi * 220 / rate)));
    {
        auto stream = bank.getChildFile("tone.wav").createOutputStream();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), rate, 1, 16, {}, 0));
        check("fixture_audio", writer && writer->writeFromAudioSampleBuffer(tone, 0, tone.getNumSamples()));
    }
    bank.getChildFile("oto.ini").replaceWithText("tone.wav=aa,0,100,1500,80,40\ntone.wav=ii,500,100,1000,80,40\n");
    bank.getChildFile("presamp.ini").replaceWithText("[VOWEL]\na=A=aa=100\ni=I=ii=100\n[CONSONANT]\nb=ba=1\n");
    UtauRenderer::invalidateVoicebankCache();
    const auto rms = [](const juce::AudioBuffer<float>& buffer, double sr, double start)
    {
        const auto first = static_cast<int>(start * sr), count = static_cast<int>(0.25 * sr);
        return buffer.getNumChannels() > 0 && first + count <= buffer.getNumSamples()
            ? buffer.getRMSLevel(0, first, count) : -1.0f;
    };
    UtauRenderRequest request;
    request.requestedFamily = RequestedVoicebankFamily::utau;
    request.voicebankDirectory = bank;
    request.targetDurationSeconds = 3;
    for (const auto& alias : {juce::String(" \t "), juce::String("RR"), juce::String("unknown-lyric")})
    {
        UtauNoteRenderSpec note;
        note.alias = alias; note.startSeconds = 0.2 + request.notes.size();
        note.durationSeconds = 0.5; note.midiNote = 64;
        request.notes.push_back(note);
    }
    for (int mode = 0; mode < 3; ++mode)
    {
        request.fourRegion = mode != 0; request.consonantClasses = mode == 2;
        for (const bool nsf : {false, true})
            for (const bool exporting : {false, true})
            {
                request.silenceEmptyLyrics = exporting;
                const auto result = nsf ? renderNsfUtauPhrase(request, {}, {}) : UtauRenderer::render(request);
                const auto key = juce::String(mode) + (nsf ? "_nsf" : "_classic") + (exporting ? "_export" : "_preview");
                const auto blank = rms(result.buffer, result.sampleRate, 0.3);
                check(key + "_blank", exporting ? blank == 0 : blank > 0.005f);
                check(key + "_rest", rms(result.buffer, result.sampleRate, 1.3) == 0);
                check(key + "_missing", rms(result.buffer, result.sampleRate, 2.3) > 0.005f);
                check(key + "_diagnostic", result.warning.contains("1 alias(es) were not found"));
            }
    }
    request.fourRegion = request.consonantClasses = false;
    request.chineseCvvc = true;
    for (const bool nsf : {false, true})
      for (const bool exporting : {false, true})
    {
        request.silenceEmptyLyrics = exporting;
        const auto result = nsf ? renderNsfUtauPhrase(request, {}, {}) : UtauRenderer::render(request);
        const auto blank = rms(result.buffer, result.sampleRate, 0.3);
        check(exporting ? "cvvc_export_blank" : "cvvc_preview_blank", exporting ? blank == 0 : blank > .005f);
        check("cvvc_rest", rms(result.buffer, result.sampleRate, 1.3) == 0);
        check("cvvc_missing_lyric_piano", rms(result.buffer, result.sampleRate, 2.3) > .005f);
        check("cvvc_missing_lyric_diagnosed", result.warning.contains("Missing CV"));
    }

    request.pianoPreviewEnabled = false;
    request.silenceEmptyLyrics = false;
    for (const bool nsf : {false, true})
    {
        const auto result = nsf ? renderNsfUtauPhrase(request, {}, {}) : UtauRenderer::render(request);
        check("disabled_piano_all_fallbacks_silent", result.buffer.getNumSamples()==0
            || result.buffer.getMagnitude(0,result.buffer.getNumSamples())==0);
    }
    request.pianoPreviewEnabled = true;
    // Exercise the production playback mixer, export preparation and WAV writer
    // in the same engine, including switching back after an export.
    for (const bool loadedBank : {true, false})
    {
        ProjectData project;
        TrackData track;
        track.id = "utau"; track.trackType = TrackType::utau;
        track.pitchAlgorithm = PitchAlgorithm::utau; track.compose = true;
        track.outputEngine = UtauOutputEngine::resampler;
        if (loadedBank) track.voicebankDirectory = bank;
        ClipData clip; clip.id = "phrase"; clip.durationSeconds = 3;
        for (int i = 0; i < 3; ++i)
        {
            NoteData note; note.id = "note-" + juce::String(i);
            note.startSeconds = .2 + i; note.durationSeconds = .5; note.midiNote = 64;
            note.label = i == 0 ? " \t " : i == 1 || !loadedBank ? "RR" : "aa";
            clip.notes.push_back(note);
        }
        track.clips.push_back(clip); project.tracks.push_back(track);
        AudioEngine audio(false);
        audio.prepareToPlay(512, rate);
        audio.selectEveryUtauNote(project);
        const auto wait = [&]
        {
            const auto deadline = juce::Time::getMillisecondCounter() + 30000;
            while (audio.renderProgress() && juce::Time::getMillisecondCounter() < deadline)
                juce::Thread::sleep(10);
            check("render_completed", !audio.renderProgress() && audio.activeRenderWarnings().isEmpty());
        };
        for (int cycle = 0; cycle < 2; ++cycle)
        {
            project.tracks[0].utauPianoPreview = true;
            audio.syncProject(project);
            wait();
            audio.setPosition(0); audio.play();
            juce::AudioBuffer<float> preview(2, 132300); preview.clear();
            for (int offset = 0; offset < preview.getNumSamples(); offset += 512)
            {
                juce::AudioSourceChannelInfo info(&preview, offset, std::min(512, preview.getNumSamples() - offset));
                audio.getNextAudioBlock(info);
            }
            audio.stop();
            check("playback_blank", rms(preview, rate, .3) > .003f);
            check("playback_rest", rms(preview, rate, 1.3) == 0);
            juce::String error;
            check("preview_cannot_leak_to_export", !audio.exportWav(folder.getChildFile("unsafe.wav"), error));
            audio.syncProject(project, true);
            wait();
            const auto target = folder.getChildFile(juce::String(loadedBank ? "loaded" : "empty") + juce::String(cycle) + ".wav");
            check("wav_written", audio.exportWav(target, error, {}, 0, 3, {44100, 2, 32}));
            juce::WavAudioFormat wav;
            std::unique_ptr<juce::AudioFormatReader> reader(wav.createReaderFor(target.createInputStream().release(), true));
            juce::AudioBuffer<float> exported;
            if (reader)
            {
                exported.setSize(2, static_cast<int>(reader->lengthInSamples));
                reader->read(&exported, 0, exported.getNumSamples(), 0, true, true);
            }
            check("export_blank_silent", rms(exported, rate, .3) == 0);
            check("export_rest_silent", rms(exported, rate, 1.3) == 0);
            if (loadedBank)
            {
                check("valid_lyric_still_sings", rms(exported, rate, 2.3) > .003f);
                check("valid_lyric_unchanged", std::abs(rms(exported, rate, 2.3) - rms(preview, rate, 2.3)) < 1.e-5f);
            }
            project.tracks[0].utauPianoPreview = false;
            audio.syncProject(project); wait();
            audio.setPosition(0); audio.play();
            juce::AudioBuffer<float> disabled(2,132300); disabled.clear();
            for(int offset=0;offset<disabled.getNumSamples();offset+=512)
            {
                juce::AudioSourceChannelInfo info(&disabled,offset,std::min(512,disabled.getNumSamples()-offset));
                audio.getNextAudioBlock(info);
            }
            audio.stop();
            check("toggle_off_blank_silent",rms(disabled,rate,.3)==0);
            if(loadedBank)check("toggle_off_preserves_valid_voice",std::abs(rms(disabled,rate,2.3)-rms(preview,rate,2.3))<1.e-5f);
        }
    }
    folder.getChildFile("result.json").replaceWithText("{\"checks\":" + juce::String(checks) + ",\"ok\":" + (ok ? "true" : "false") + "}");
    return ok;
}
}
