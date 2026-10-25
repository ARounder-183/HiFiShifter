#pragma once
#include "../MainComponent.h"
#include "../NativeSourceTimeMap.h"
#include <iostream>

namespace hachi
{
inline void MainComponent::diagnosticOtoAudio(const juce::File& output, std::function<void(bool)> completed)
{
    output.createDirectory(); stopTimer(); setSize(1500, 900);
    struct State
    {
        bool ok = true;
        juce::StringArray report;
        void check(const juce::String& name, bool passed)
        { ok &= passed; report.add(name + "=" + (passed ? "1" : "0")); }
    };
    auto state = std::make_shared<State>();
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-7; };
    const auto bank = output.getChildFile("bank");
    const auto nested = bank.getChildFile("nested"); nested.createDirectory();
    const auto wav = nested.getChildFile("sample.wav");
    {
        juce::AudioBuffer<float> tone(1, 96000);
        for (int i = 0; i < tone.getNumSamples(); ++i)
        {
            const auto t = i / 48000.0;
            tone.setSample(0, i, .2f * static_cast<float>(std::sin(juce::MathConstants<double>::twoPi * (220 * t + 9 * t * t))));
        }
        auto stream = wav.createOutputStream();
        if (stream) { stream->setPosition(0); stream->truncate(); }
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(), 48000, 1, 24, {}, 0));
        state->check("fixture_wav", writer && writer->writeFromAudioSampleBuffer(tone, 0, tone.getNumSamples()));
    }
    const auto classic = nested.getChildFile("oto.ini");
    const auto jie = nested.getChildFile("oto.jie.ini");
    const auto four = nested.getChildFile("oto4.ini");
    const auto mou = nested.getChildFile("otomou.ini");
    classic.replaceWithText("sample.wav=classic,100,200,-1200,80,30\n");
    jie.replaceWithText("sample.wav=jie,150,200,-1200,80,30\n");
    four.replaceWithText("sample.wav=150,200,350,900\nsample.wav=100,50,250,700\n");
    mou.replaceWithText("sample.wav=CVS,100,250,950,1100\n");
    bank.getChildFile("oto.ini").replaceWithText("nested/sample.wav=parent,0,100,-500,20,10\n");
    std::vector<std::pair<juce::File, juce::String>> hashes;
    for (const auto& file : {wav, classic, jie, four, mou})
        hashes.push_back({file, juce::SHA256(file).toHexString()});
    const auto parsed = [&](UtauMode mode)
    {
        juce::StringArray warnings;
        return SampleSettings::loadAudioOto(wav, bank, mode, warnings);
    };
    const auto normal = parsed(UtauMode::classic), jieRows = parsed(UtauMode::jie), mouRows = parsed(UtauMode::mou);
    state->check("nearest_folder_and_classic_selected", normal.size() == 1 && normal[0].alias == "classic");
    state->check("independent_jie_timing_selected", jieRows.size() == 1 && jieRows[0].alias == "jie" && near(jieRows[0].offsetMs, 150));
    state->check("mou_classes_read", mouRows.size() == 1 && mouRows[0].mouClasses == "CVS");
    mou.replaceWithText("sample.wav=CV,100,250\n");
    const auto compact = parsed(UtauMode::mou);
    state->check("compact_two_region_mou_file", compact.size() == 1 && compact[0].mouClasses == "CV"
        && near(compact[0].jieOnsetMs, 250));
    mou.replaceWithText("sample.wav=CVS,100,250,950,1100\n");
    if (normal.empty() || jieRows.empty() || mouRows.empty()) { completed(false); return; }
    const auto c = SampleSettings::nativeAudioFromOto(normal[0], UtauMode::classic, 2);
    const auto j = SampleSettings::nativeAudioFromOto(jieRows[0], UtauMode::jie, 2);
    const auto m = SampleSettings::nativeAudioFromOto(mouRows[0], UtauMode::mou, 2);
    state->check("classic_fixed_boundary_not_preutterance_or_oto4", c && c->notes.size() == 2
        && near(c->sourceOffsetSeconds, .1) && near(c->durationSeconds, 1.2) && near(c->notes[0].durationSeconds, .2));
    state->check("jie_four_authored_regions", j && j->notes.size() == 4
        && near(j->sourceOffsetSeconds, .15) && near(j->notes[2].startSeconds, .35)
        && near(j->notes[3].startSeconds, .9));
    state->check("mou_three_regions_and_silence", m && m->notes.size() == 3
        && near(m->notes[1].startSeconds, .25) && near(m->notes[2].startSeconds, .95)
        && m->notes[2].nativeUnpitched && m->notes[0].nativeRole == NativeSegmentRole::consonant);
    auto altered = normal[0]; altered.cutoffMs = 500;
    auto mapped = SampleSettings::nativeAudioFromOto(altered, UtauMode::classic, 2);
    state->check("positive_cutoff_from_file_end", mapped && near(mapped->durationSeconds, 1.4));
    altered = mouRows[0]; altered.mouClasses = "CV";
    mapped = SampleSettings::nativeAudioFromOto(altered, UtauMode::mou, 2);
    state->check("mou_two_regions", mapped && mapped->notes.size() == 2 && near(mapped->notes.back().durationSeconds, .95));
    altered.mouClasses = "CVVC";
    mapped = SampleSettings::nativeAudioFromOto(altered, UtauMode::mou, 2);
    state->check("mou_four_regions", mapped && mapped->notes.size() == 4 && near(mapped->notes.back().durationSeconds, .1));
    altered = normal[0]; altered.hasJieOto = false; altered.mouClasses.clear();
    state->check("missing_selected_format_is_not_fallback", !SampleSettings::nativeAudioFromOto(altered, UtauMode::jie, 2)
        && !SampleSettings::nativeAudioFromOto(altered, UtauMode::mou, 2));
    altered.offsetMs = 3000;
    state->check("out_of_audio_trim_rejected", !SampleSettings::nativeAudioFromOto(altered, UtauMode::classic, 2));
    altered = jieRows[0]; altered.jieGlideMs = altered.jieOnsetMs;
    mapped = SampleSettings::nativeAudioFromOto(altered, UtauMode::jie, 2);
    state->check("zero_width_region_not_created", mapped && mapped->notes.size() == 3);
    // Model insertion owns one undo step, and all source time remains relative to the physical recording.
    ProjectModel model;
    const auto track = model.addTrack("Native"); model.setTrackPitchAlgorithm(track, PitchAlgorithm::world);
    const auto id = model.addAudioFile(wav, 2, 3, track, &jieRows[0], UtauMode::jie);
    const auto imported = model.snapshot().tracks[0].clips[0];
    state->check("model_keeps_trim_and_unique_editable_notes", id.isNotEmpty() && imported.notes.size() == 4
        && imported.notes[0].id != imported.notes[1].id && near(imported.startSeconds, 3)
        && near(imported.notes[1].nativeSourceStartSeconds, .35));
    const auto timeMap = nativeSourceTimeMap(imported);
    state->check("initial_source_mapping_has_no_stretch", near(nativeSourceTimeAt(timeMap, .7), .7));
    state->check("one_undo_removes_import", model.undo() && model.snapshot().tracks[0].clips.empty());
    state->check("redo_restores_split", model.redo() && model.snapshot().tracks[0].clips[0].notes.size() == 4);
    // Drive the actual dock selector and the internal drag target, then await real analysis.
    juce::PropertiesFile::Options options; options.applicationName = "oto-audio-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    preferences = std::make_unique<juce::PropertiesFile>(output.getChildFile("settings.xml"), options);
    preferences->setValue("algorithm.inference", 2); preferences->setValue("assets.otoImportKind", 2);
    project.replace(ProjectData{});
    selectedTrackId = project.addTrack("Native"); project.setTrackPitchAlgorithm(selectedTrackId, PitchAlgorithm::world);
    showAssetManager(); assetManager->diagnosticRegisterFolder(bank);
    std::vector<ClipData> beforeAnalysis;
    for (int kind = 2; kind <= 4; ++kind)
    {
        assetManager->setOtoImportKind(bank, kind);
        const auto description = assetManager->audioDragDescription(wav);
        const auto position = pianoViewport.getPosition() + juce::Point<int>(80, 40);
        juce::DragAndDropTarget::SourceDetails drop(description, assetManager.get(), position);
        state->check("drag_metadata_" + juce::String(kind), static_cast<int>(description["otoKind"]) == kind
            && description["root"].toString() == bank.getFullPathName() && isInterestedInDragSource(drop));
        itemDropped(drop);
        const auto data = project.snapshot();
        state->check("native_drop_" + juce::String(kind), data.tracks.size() == 1
            && data.tracks[0].clips.size() == static_cast<std::size_t>(kind - 1));
        if (!data.tracks[0].clips.empty()) beforeAnalysis.push_back(data.tracks[0].clips.back());
    }
    AssetManagerComponent reopenedManager(strings, *preferences);
    state->check("bank_oto_choice_persists", reopenedManager.otoImportKind(wav) == 4);
    if (auto stream = output.getChildFile("oto-selector.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(assetManager->createComponentSnapshot(assetManager->getLocalBounds()), *stream);
    const auto start = juce::Time::getMillisecondCounterHiRes();
    auto poll = std::make_shared<std::function<void()>>();
    const auto weak = std::weak_ptr<std::function<void()>>(poll);
    *poll = [this, state, output, beforeAnalysis, hashes, wav, start, completed, weak, near]
    {
        if (pendingNativeAnalyses > 0 && juce::Time::getMillisecondCounterHiRes() - start < 120000)
        { if (auto next = weak.lock()) juce::Timer::callAfterDelay(50, [next] { (*next)(); }); return; }
        state->check("analysis_completed", pendingNativeAnalyses == 0);
        const auto data = project.snapshot();
        bool same = beforeAnalysis.size() == 3, measured = same;
        for (const auto& before : beforeAnalysis)
        {
            const ClipData* after = nullptr;
            for (const auto& track : data.tracks) for (const auto& clip : track.clips) if (clip.id == before.id) after = &clip;
            if (!after || after->notes.size() != before.notes.size()) { same = measured = false; continue; }
            same &= near(after->sourceOffsetSeconds, before.sourceOffsetSeconds) && near(after->durationSeconds, before.durationSeconds);
            for (std::size_t i = 0; i < before.notes.size(); ++i)
            {
                same &= near(after->notes[i].startSeconds, before.notes[i].startSeconds)
                    && near(after->notes[i].durationSeconds, before.notes[i].durationSeconds);
                measured &= after->notes[i].sourcePitchMeasured && !after->notes[i].contour.empty();
            }
        }
        state->check("analysis_preserves_oto_crop_and_all_boundaries", same);
        state->check("all_split_notes_receive_source_pitch", measured);
        bool untouched = !SampleSettings::sidecarFor(wav).existsAsFile();
        for (const auto& [file, hash] : hashes) untouched &= juce::SHA256(file).toHexString() == hash;
        state->check("audio_and_all_oto_unchanged_no_hjm_conversion", untouched);
        juce::String error; ProjectModel restored;
        state->check("project_saves_and_reopens_native_splits", project.save(output.getChildFile("oto-native.hjpx"), error)
            && restored.load(output.getChildFile("oto-native.hjpx"), error)
            && restored.snapshot().tracks[0].clips.size() == 3
            && restored.snapshot().tracks[0].clips[1].notes.size() == 4);
        if (data.tracks.size() == 1 && data.tracks[0].clips.size() == 3)
        {
            auto renderData = data;
            auto clip = renderData.tracks[0].clips[1]; clip.startSeconds = .5;
            renderData.tracks[0].clips = {clip}; renderData.tracks[0].normalizeVolume = false;
            AudioEngine checkEngine; checkEngine.prepareToPlay(256, 48000); checkEngine.syncProject(renderData);
            for (int i = 0; i < 1000 && checkEngine.renderProgress(); ++i) juce::Thread::sleep(10);
            WavExportOptions options; options.sampleRate = 48000; options.channels = 2; options.bitDepth = 32;
            const auto rendered = output.getChildFile("trimmed-audio.wav");
            const auto exported = checkEngine.exportWav(rendered, error, renderData.tracks[0].id, 0, 2, options);
            state->check("trimmed_native_audio_renders", exported);
            juce::AudioFormatManager formats; formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(rendered));
            std::unique_ptr<juce::AudioFormatReader> source(formats.createReaderFor(wav));
            if (reader && source)
            {
                juce::AudioBuffer<float> actual(2, 96000), original(1, 96000);
                reader->read(&actual, 0, 96000, 0, true, true); source->read(&original, 0, 96000, 0, true, false);
                state->check("audio_outside_oto_window_is_silent", actual.getRMSLevel(0, 0, 23000) < 1.e-5f
                    && actual.getRMSLevel(0, 83000, 13000) < 1.e-5f);
                double xy = 0, xx = 0, yy = 0;
                for (int i = 26000; i < 79000; ++i)
                {
                    const auto x = original.getSample(0, i - 24000 + 7200), y = actual.getSample(0, i);
                    xy += x * y; xx += x * x; yy += y * y;
                }
                state->check("render_uses_oto_offset_without_restretch", xx > 1 && yy > 1
                    && xy / std::sqrt(xx * yy) > .999);
            }
            else state->check("rendered_audio_readable", false);
            checkEngine.releaseResources();
        }
        const auto text = state->report.joinIntoString("\n") + "\n";
        output.getChildFile("verification.txt").replaceWithText(text); std::cout << text << std::flush;
        completed(state->ok);
    };
    juce::Timer::callAfterDelay(50, [poll] { (*poll)(); });
}
}
