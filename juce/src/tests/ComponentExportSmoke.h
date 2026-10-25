#pragma once
#include "../AudioEngine.h"
#include "../backend/Llsm2Renderer.h"
#include "../backend/WorldRenderer.h"

namespace hachi
{
inline void componentExportChecks(const juce::File& folder, AudioEngine& audio,
                                  const std::function<void(juce::String, bool)>& check)
{
    const auto fixture = folder.getChildFile("voice-components-source.wav");
    juce::AudioBuffer<float> source(1, 32000);
    juce::Random random(341);
    for (int i = 0; i < source.getNumSamples(); ++i)
    {
        const auto phase = juce::MathConstants<double>::twoPi * 220.0 * i / 32000;
        source.setSample(0, i, static_cast<float>(0.18 * std::sin(phase) + 0.05 * std::sin(2 * phase)
            + 0.02 * (2.0 * random.nextFloat() - 1.0)));
    }
    {
        auto stream=fixture.createOutputStream(); stream->setPosition(0); stream->truncate();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),32000,1,24,{},0));
        check("component_fixture",writer && writer->writeFromAudioSampleBuffer(source,0,source.getNumSamples()));
    }
    ProjectModel model; (void) model.addAudioFile(fixture,1.0);
    auto project=model.snapshot(); auto& track=project.tracks.front(); auto& clip=track.clips.front();
    track.name="Component fixture"; track.compose=true; track.pitchAlgorithm=PitchAlgorithm::world;
    NoteData note; note.id="stem-note"; note.label="a"; note.midiNote=60; note.startSeconds=0; note.durationSeconds=1;
    clip.notes={note};
    const auto enginePath=juce::SystemStats::getEnvironmentVariable("HACHI_COMPONENT_TEST_ENGINE",{});
    const juce::File engine(enginePath);
    check("world_component_capability",AudioEngine::componentExportIssue(project,engine).isEmpty());
    track.pitchAlgorithm=PitchAlgorithm::llsm2;
    check("llsm_component_capability",AudioEngine::componentExportIssue(project,engine).isEmpty());
    track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;
    check("neural_component_disabled",AudioEngine::componentExportIssue(project,engine).isNotEmpty());
    track.pitchAlgorithm=PitchAlgorithm::world;
    auto mixed=project; auto accompaniment=track; accompaniment.id="accompaniment";accompaniment.compose=false;
    mixed.tracks.push_back(accompaniment);
    check("mixed_all_tracks_disabled",AudioEngine::componentExportIssue(mixed,engine).isNotEmpty());
    check("single_supported_track_ignores_other_tracks",AudioEngine::componentExportIssue(mixed,engine,track.id).isEmpty());
    mixed.tracks.back().muted=true;
    check("muted_unsupported_track_ignored",AudioEngine::componentExportIssue(mixed,engine).isEmpty());
    mixed.tracks.back().muted=false; mixed.tracks.front().solo=true;
    check("solo_excludes_unsupported_track",AudioEngine::componentExportIssue(mixed,engine).isEmpty());
    auto raw=project; raw.tracks.front().compose=false;
    audio.syncProject(raw);
    juce::String error;
    auto protectedFile=folder.getChildFile("unsupported-component.wav");protectedFile.replaceWithText("preserve");
    check("cannot_export_full_cache_as_breath",!audio.exportWav(protectedFile,error,{},0,0,
        {32000,1,32,WavExportComponent::breath}) && protectedFile.loadFileAsString()=="preserve");
    audio.syncProject(raw,true,WavExportComponent::breath);
    check("unsupported_component_fails_closed",!audio.exportWav(protectedFile,error,{},0,0,
        {32000,1,32,WavExportComponent::breath}) && protectedFile.loadFileAsString()=="preserve");

    backend::RenderService service;
    const auto render=[&](backend::Mld5FileRenderRequest request)
    {
        // Shared state survives a timeout; no worker can write into dead stack storage.
        struct Result { juce::WaitableEvent done; backend::RenderedAudio audio; };
        auto state=std::make_shared<Result>();
        service.renderMld5File(std::move(request),[state](auto rendered)
        { state->audio=std::move(rendered);state->done.signal(); });
        if (!state->done.wait(45000)) { check("component_render_completed",false);return backend::RenderedAudio{}; }
        return std::move(state->audio);
    };
    for (auto backend : {backend::PitchRenderBackend::world,backend::PitchRenderBackend::llsm2})
    {
        backend::Mld5FileRenderRequest request;
        request.sourceFile=fixture;request.sourceDurationSeconds=1;request.targetDurationSeconds=1;
        request.sourceMidi.assign(201,57);request.targetMidi.assign(201,60);
        request.pitchBackend=backend;request.noteGain.assign(201,0.7f);request.tension.assign(201,0.35f);
        request.breath.assign(201,0.6f);request.normalizeVolume=true;
        const auto name=backend==backend::PitchRenderBackend::world ? juce::String("world") : juce::String("llsm");
        auto full=render(request);
        request.exportComponent=WavExportComponent::breath;auto noise=render(request);
        request.exportComponent=WavExportComponent::nonBreath;auto harmonic=render(request);
        const auto sizes=full.buffer.getNumSamples()==32000 && noise.buffer.getNumSamples()==32000
            && harmonic.buffer.getNumSamples()==32000;
        check(name+"_three_outputs_succeeded",sizes && full.warning.isEmpty() && noise.warning.isEmpty() && harmonic.warning.isEmpty());
        if (!sizes) continue;
        const auto nr=noise.buffer.getRMSLevel(0,3200,25600), hr=harmonic.buffer.getRMSLevel(0,3200,25600);
        check(name+"_components_nonzero_and_distinct",nr>1e-5f && hr>nr*1.2f);
        bool finite=true;double residual=0;
        for (int i=0;i<32000;++i)
        {
            auto delta=full.buffer.getSample(0,i)-noise.buffer.getSample(0,i)-harmonic.buffer.getSample(0,i);
            finite=finite && std::isfinite(delta);residual+=delta*delta;
        }
        check(name+"_components_finite",finite);
        // WORLD restarts its noise RNG per synthesis. LLSM uses a shared random
        // sequence, so independent requests need not sum sample-for-sample.
        if(backend==backend::PitchRenderBackend::world)
            check("world_components_sum_to_full_after_expression_and_normalization",std::sqrt(residual/32000)<1e-6);
    }
    // The unedited WORLD bypass must still analyse when a component is requested.
    std::vector<float> pitch(201,57);juce::AudioBuffer<float> noise;
    auto analysed=backend::WorldRenderer::render(source,32000,32000,5,pitch,pitch,{}, {}, &noise);
    check("unchanged_world_material_still_separates",analysed.getNumSamples()==32000 && noise.getNumSamples()==32000
        && noise.getRMSLevel(0,3200,25600)<analysed.getRMSLevel(0,3200,25600)*0.8f);

    // End-to-end offline mix, cache separation, formats, range and restoration.
    const auto waitRender=[&]
    {
        auto timeout=juce::Time::getMillisecondCounterHiRes()+45000;
        while (audio.renderProgress() && juce::Time::getMillisecondCounterHiRes()<timeout) juce::Thread::sleep(10);
        return !audio.renderProgress() && audio.activeRenderWarnings().isEmpty();
    };
    track.pitchAlgorithm=PitchAlgorithm::world;track.pan=1.0f;
    audio.syncProject(project);check("normal_voice_prepared",waitRender());
    auto fullFile=folder.getChildFile("native-full-before.wav");
    check("normal_voice_export",audio.exportWav(fullFile,error,track.id,0,0,{44100,2,24}));
    for(auto component : {WavExportComponent::breath,WavExportComponent::nonBreath})
    {
        audio.syncProject(project,true,component);check("separated_voice_prepared",waitRender());
        const auto name=component==WavExportComponent::breath ? juce::String("breath") : juce::String("non-breath");
        audio.clearAuditionFile();audio.play();
        juce::AudioBuffer<float> liveBlock(2,256);liveBlock.clear();
        juce::AudioSourceChannelInfo info(&liveBlock,0,256);audio.getNextAudioBlock(info);audio.stop();
        check(name+"_never_leaks_into_live_playback",liveBlock.getMagnitude(0,256)==0.0f);
        auto file=folder.getChildFile(name+"-engine.wav");
        check(name+"_engine_range_export",audio.exportWav(file,error,track.id,.1,.9,{48000,2,32,component}));
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatReader> reader(wav.createReaderFor(file.createInputStream().release(),true));
        check(name+"_engine_range_format",reader && reader->numChannels==2 && reader->usesFloatingPointData
            && reader->sampleRate==48000 && reader->lengthInSamples==38400);
        if(reader)
        {
            juce::AudioBuffer<float> samples(2,38400);reader->read(&samples,0,38400,0,true,true);
            check(name+"_engine_pan_and_audio",samples.getMagnitude(0,0,38400)<1e-6f && samples.getMagnitude(1,0,38400)>1e-5f);
        }
    }
    audio.syncProject(project);check("normal_voice_restored",waitRender());
    auto afterFile=folder.getChildFile("native-full-after.wav");
    juce::MemoryBlock beforeBytes,afterBytes;
    check("normal_voice_unchanged_after_stems",audio.exportWav(afterFile,error,track.id,0,0,{44100,2,24})
        && fullFile.loadFileAsData(beforeBytes) && afterFile.loadFileAsData(afterBytes) && beforeBytes==afterBytes);

    if (engine.existsAsFile())
    {
        for(const auto* flags : {"K2","K2Mb100","K2HF2","K2Mx1","L2g-12","K2HN1","HNK2","HN0K2","HN2K2","HN-1K2","HN10K2"})
            check(juce::String("wcs_enabled_")+flags,backend::UtauRenderer::supportsComponentExport(engine,flags));
        for(const auto* flags : {"","K1","V2","M1","HF2K2","K2u","K2Mm0","K2Mm100","K1K2","HN1","HN1K2","HNHN1K2","HN0HN1K2","HN2HN1K2"})
            check(juce::String("wcs_disabled_")+flags,!backend::UtauRenderer::supportsComponentExport(engine,flags));
        const auto bank=folder.getChildFile("component-bank");bank.createDirectory();fixture.copyFileTo(bank.getChildFile("a.wav"));
        bank.getChildFile("oto.ini").replaceWithText("a.wav=a,0,50,-950,0,0\n");
        backend::UtauRenderRequest request;request.voicebankDirectory=bank;request.resamplerExecutable=engine;
        request.targetDurationSeconds=1;backend::UtauNoteRenderSpec n;n.alias="a";n.flags="K2P0Mb20";
        n.durationSeconds=.8;n.startSeconds=.1;n.midiNote=57;request.notes.push_back(n);
        for(auto component : {WavExportComponent::breath,WavExportComponent::nonBreath})
        {
            request.exportComponent=component;auto result=backend::UtauRenderer::render(request);
            check(component==WavExportComponent::breath ? "wcs_real_breath" : "wcs_real_non_breath",
                result.warning.isEmpty() && result.buffer.getNumSamples()>0 && result.buffer.getMagnitude(0,0,result.buffer.getNumSamples())>1e-5f);
        }
        request.notes.front().flags="K1";
        auto rejected=backend::UtauRenderer::render(request);
        check("wcs_runtime_rejects_wrong_kernel",rejected.buffer.getNumSamples()==0 && rejected.warning.isNotEmpty());
        request.notes.front().flags="K2";
        request.voicebankDirectory=folder.getChildFile("empty-component-bank");request.voicebankDirectory.createDirectory();
        rejected=backend::UtauRenderer::render(request);
        check("wcs_runtime_rejects_piano_fallback",rejected.buffer.getNumSamples()==0 && rejected.warning.isNotEmpty());
    }
    else check("configured_wcs_engine_available",false);
}
}
