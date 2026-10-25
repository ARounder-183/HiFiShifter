#pragma once
#include "../AccompanimentTempoPanel.h"
#include "../backend/McpServer.h"
#include <iostream>

namespace hachi
{
inline void MainComponent::diagnosticAccompanimentTempo(const juce::File& folder,
    const juce::File& audioFile, std::function<void(bool)> completed)
{
    struct State
    {
        bool ok = true;
        int stage = 0, pulses = 0;
        double start = juce::Time::getMillisecondCounterHiRes();
        juce::Array<juce::var> checks;
        juce::var mainResult, mergedResult;
        ClipData clip;
        std::uint64_t revision = 0;
        juce::int64 fingerprint = 0;
        std::unique_ptr<AccompanimentTempoTask> task;
        void check(const char* name, bool passed)
        {
            auto* row = new juce::DynamicObject(); row->setProperty("name", name); row->setProperty("passed", passed);
            checks.add(row); ok &= passed; std::cout << name << '=' << passed << std::endl;
        }
    };
    folder.createDirectory(); stopTimer();
    auto state = std::make_shared<State>();
    const auto writeWav = [&](const juce::File& file, const juce::AudioBuffer<float>& samples, int rate)
    {
        auto stream = file.createOutputStream(); if (!stream) return false;
        stream->setPosition(0); stream->truncate(); juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), rate, samples.getNumChannels(), 32, {}, 0));
        return writer && writer->writeFromAudioSampleBuffer(samples, 0, samples.getNumSamples());
    };
    const auto source = audioFile.existsAsFile() ? audioFile : folder.getChildFile("120-then-80.wav");
    if (!audioFile.existsAsFile())
    {
        constexpr int rate = 22050, duration = 60;
        juce::AudioBuffer<float> samples(1, rate * duration); samples.clear();
        juce::Random random(420);
        for (int section = 0; section < 2; ++section)
        {
            const double period = section == 0 ? .5 : .75;
            for (int beat = 0; beat * period < 30; ++beat)
                for (int i = 0; i < rate / 4; ++i)
                {
                    const auto index = static_cast<int>((30 * section + beat * period) * rate) + i;
                    if (index >= samples.getNumSamples()) break;
                    const auto t = static_cast<double>(i) / rate;
                    const auto kick = .65 * std::sin(juce::MathConstants<double>::twoPi * (55 * t + 1.5 * (1 - std::exp(-25 * t)))) * std::exp(-22 * t);
                    const auto snare = beat % 2 ? .28 * (random.nextFloat() * 2 - 1) * std::exp(-30 * t) : 0;
                    samples.addSample(0, index, static_cast<float>(kick + snare));
                }
        }
        state->check("rhythmic_fixture_written", writeWav(source, samples, rate));
    }
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(source));
    if (!reader) { state->check("source_decodes", false); completed(false); return; }
    const auto duration = reader->lengthInSamples / reader->sampleRate;
    ClipData clip; clip.id = "tempo-backing"; clip.sourceFile = source;
    clip.startSeconds = 7.25; clip.sourceOffsetSeconds = audioFile.existsAsFile() ? std::min(30.0, duration / 4) : 0;
    clip.durationSeconds = clip.sourceDurationSeconds = std::min(30.0, duration - clip.sourceOffsetSeconds);
    clip.gain = .75f; state->clip = clip;
    TrackData backing; backing.id = "backing"; backing.accompaniment = true; backing.compose = false; backing.clips = {clip};
    TrackData normal; normal.id = "normal"; normal.compose = false; normal.clips = {clip}; normal.clips.front().id = "normal-audio";
    ProjectData data; data.bpm = 137; data.tracks = {backing, normal};
    project.replace(data); project.dispatchPendingMessages(); stopTimer(); setSize(1280, 760);
    const auto hasItem = [&](const juce::String& id, bool enabled)
    {
        auto menu = clipContextMenu(id, 10);
        for (juce::PopupMenu::MenuItemIterator it(menu); it.next();)
            if (it.getItem().itemID == recogniseTempoMenuItem) return it.getItem().isEnabled == enabled;
        return false;
    };
    state->check("backing_context_has_enabled_detect_tempo", hasItem(clip.id, true));
    state->check("ordinary_audio_has_no_detect_tempo", !hasItem("normal-audio", true) && !hasItem("normal-audio", false));
    const auto beforeInvalid = tempoWindows.size();
    clipContextMenuItemChosen(recogniseTempoMenuItem, "normal-audio", 10);
    state->check("ordinary_audio_cannot_launch_tempo", tempoWindows.size() == beforeInvalid);
    state->revision = project.revisionNumber(); state->fingerprint = project.contentFingerprint();
    folder.getChildFile("before.json").replaceWithText(juce::JSON::toString(backend::McpServer(project, nullptr).snapshotJson(), true));
    clipContextMenuItemChosen(recogniseTempoMenuItem, clip.id, 10);
    auto* window = tempoWindows.findVisible<AccompanimentTempoPanel>();
    state->check("menu_launches_actual_modeless_result_window", window && !window->isCurrentlyModal());
    auto* originalWindow = window;
    clipContextMenuItemChosen(recogniseTempoMenuItem, clip.id, 10);
    state->check("repeat_click_reuses_task", tempoWindows.findVisible<AccompanimentTempoPanel>() == originalWindow && tempoWindows.size() == 1);
    if (!window) { completed(false); return; }
    auto poll = std::make_shared<std::function<void()>>();
    *poll = [this, folder, completed, state, weak = std::weak_ptr<std::function<void()>>(poll)]
    {
        ++state->pulses;
        const auto finish = [&]
        {
            folder.getChildFile("after.json").replaceWithText(juce::JSON::toString(backend::McpServer(project, nullptr).snapshotJson(), true));
            state->check("project_revision_unchanged", project.revisionNumber() == state->revision);
            state->check("project_content_and_audio_settings_unchanged", project.contentFingerprint() == state->fingerprint);
            state->check("project_tempo_stays_137", project.snapshot().bpm == 137);
            tempoWindows.clear(); state->task.reset();
            auto* report = new juce::DynamicObject(); report->setProperty("passed", state->ok);
            report->setProperty("checks", state->checks); report->setProperty("model_result", state->mainResult);
            report->setProperty("merged_result", state->mergedResult);
            const bool saved = folder.getChildFile("validation.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
            completed(state->ok && saved);
        };
        if (juce::Time::getMillisecondCounterHiRes() - state->start > 180000)
        { state->check("analysis_completed_without_timeout", false); finish(); return; }
        if (state->stage == 0)
        {
            auto* window = tempoWindows.findVisible<AccompanimentTempoPanel>();
            auto* panel = window ? dynamic_cast<AccompanimentTempoPanel*>(window->getContentComponent()) : nullptr;
            if (!panel) { state->check("window_survives_analysis", false); finish(); return; }
            if (panel->finished())
            {
                panel->refreshResult(); state->mainResult = panel->result();
                state->check("existing_beat_this_model_returns_bpm", (bool)state->mainResult["ok"] && (double)state->mainResult["bpm"] > 0);
                state->check("only_visible_clip_duration_analysed", std::abs((double)state->mainResult["duration"] - state->clip.durationSeconds) < .001);
                state->check("dialog_displays_bpm", panel->displayedValue().contains("BPM") && !panel->displayedValue().contains("tempo."));
                state->check("message_thread_keeps_processing", state->pulses > 2);
                if (state->clip.sourceFile.getFileName() == "120-then-80.wav")
                {
                    const double bpm = state->mainResult["bpm"];
                    state->check("trim_excludes_later_80_bpm_section", std::min({std::abs(bpm - 120), std::abs(bpm - 60), std::abs(bpm - 240)}) < 2);
                }
                const auto image = folder.getChildFile("tempo-result.png"); image.deleteFile();
                if (auto stream = image.createOutputStream()) juce::PNGImageFormat().writeImageToStream(panel->createComponentSnapshot(panel->getLocalBounds()), *stream);
                tempoWindows.clear();
                auto merged = state->clip; auto first = merged, second = merged;
                first.id = "first"; first.startSeconds = 0; first.durationSeconds = first.sourceDurationSeconds = merged.durationSeconds / 2;
                second.id = "second"; second.startSeconds = first.durationSeconds; second.sourceOffsetSeconds += first.durationSeconds;
                second.durationSeconds = second.sourceDurationSeconds = merged.durationSeconds / 2;
                merged.sourceFile = juce::File{}; merged.parts = {first, second};
                state->check("merged_audio_is_supported", AccompanimentTempoTask::canAnalyse(merged));
                state->task = std::make_unique<AccompanimentTempoTask>(merged); state->task->startThread(); state->stage = 1;
            }
        }
        else if (state->stage == 1 && state->task->finished())
        {
            state->mergedResult = state->task->result();
            state->check("merged_audio_model_returns_bpm", (bool)state->mergedResult["ok"]);
            state->check("splitting_same_source_keeps_estimate", std::abs((double)state->mergedResult["bpm"] - (double)state->mainResult["bpm"]) < 1);
            state->task = std::make_unique<AccompanimentTempoTask>(state->clip, folder.getChildFile("absent/runtime.json"));
            state->task->startThread(); state->stage = 2;
        }
        else if (state->stage == 2 && state->task->finished())
        {
            state->check("missing_runtime_is_explicit_error", !(bool)state->task->result()["ok"] && state->task->result()["code"].toString() == "missing_runtime");
            auto* config = new juce::DynamicObject();
            config->setProperty("python", juce::JSON::parse(AccompanimentTempoTask::runtimeConfig())["python"]);
            const auto runtime = folder.getChildFile("missing-model/runtime.json"); runtime.getParentDirectory().createDirectory();
            runtime.replaceWithText(juce::JSON::toString(juce::var(config), true)); runtime.getSiblingFile("tempo_worker.py").replaceWithText("# no model\n");
            state->task = std::make_unique<AccompanimentTempoTask>(state->clip, runtime); state->task->startThread(); state->stage = 3;
        }
        else if (state->stage == 3 && state->task->finished())
        {
            state->check("missing_model_does_not_download_or_fabricate_bpm", state->task->result()["code"].toString() == "missing_model");
            state->task = std::make_unique<AccompanimentTempoTask>(state->clip); state->task->startThread(); state->stage = 4;
        }
        else if (state->stage == 4 && state->task->stage() == 1)
        {
            state->task->cancel(); state->stage = 5;
        }
        else if (state->stage == 5 && state->task->finished())
        {
            state->check("model_task_cancels_without_result", state->task->result()["code"].toString() == "cancelled");
            state->task.reset();
            AccompanimentTempoTask cancelled(state->clip); cancelled.cancel(); cancelled.startThread();
            const auto started = juce::Time::getMillisecondCounterHiRes();
            while (!cancelled.finished()) juce::Thread::sleep(2);
            state->check("pre_cancelled_task_skips_model", cancelled.result()["code"].toString() == "cancelled");
            state->check("cancelled_task_join_is_prompt", juce::Time::getMillisecondCounterHiRes() - started < 1500);
            finish(); return;
        }
        if (auto next = weak.lock()) juce::Timer::callAfterDelay(40, [next] { (*next)(); });
    };
    juce::Timer::callAfterDelay(40, [poll] { (*poll)(); });
}
}
