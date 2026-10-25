#pragma once
#include "../AudioEngine.h"
#include "../MaterialAnnotationSession.h"
#include "../NativeNoteTiming.h"
#include "../NativePitchIdentity.h"
#include <cstring>
#include <iostream>

namespace hachi
{
// Real offline engine/export proof, not a request-only or clock-only check.
// AudioEngine(false) must not initialise a device (default GUI behavior remains
// unchanged). No model, audio device, user window, or source analysis is used.
inline bool runMaterialTopologyAudioSmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;
    bool ok=true;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass)
    {ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto finish=[&]
    {
        const auto written=output.getChildFile("material-topology-audio-smoke.txt").replaceWithText(log);
        std::cout<<"report_written="<<written<<std::endl;return ok&&written;
    };
    constexpr int rate=48000,sourceFrames=rate*2,secondStart=rate*3,exportFrames=rate*5;
    const auto bank=output.getChildFile("bank");if(bank.createDirectory().failed())return false;
    const auto source=bank.getChildFile("stereo-source.wav"),manifest=bank.getChildFile("otto.ini");
    {
        juce::AudioBuffer<float> tone(2,sourceFrames);
        for(int i=0;i<sourceFrames;++i)
        {
            const auto t=static_cast<double>(i)/rate;
            tone.setSample(0,i,static_cast<float>(.19*std::sin(juce::MathConstants<double>::twoPi*217*t)
                +.035*std::sin(juce::MathConstants<double>::twoPi*431*t)));
            tone.setSample(1,i,static_cast<float>(.17*std::cos(juce::MathConstants<double>::twoPi*331*t)
                +.025*std::sin(juce::MathConstants<double>::twoPi*613*t)));
        }
        auto stream=source.createOutputStream();if(!stream){check("source_stream",false);return finish();}
        stream->setPosition(0);stream->truncate();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(),rate,2,24,{},0));
        if(writer)stream.release();
        const auto written=writer&&writer->writeFromAudioSampleBuffer(tone,0,sourceFrames);
        check("two_second_stereo_fixture_written",written);if(!written)return finish();
    }
    juce::MemoryBlock sourceBytes;check("source_bytes_captured",source.loadFileAsData(sourceBytes));
    SampleRegionSetting a,b;a.name="r1";a.materialAudioId="audio-stereo";a.materialRegionId="r1";
    a.regionStartSeconds=0;a.regionEndSeconds=1;a.alignmentSeconds=0;a.role=NativeSegmentRole::vowel;
    a.provenance="user";a.confidence=1;a.connectedNextRegionId="r2";
    b=a;b.name="r2";b.materialRegionId="r2";b.regionStartSeconds=1;b.regionEndSeconds=2;b.alignmentSeconds=1;b.connectedNextRegionId.clear();
    OttoMaterialIO::Document document;
    if(manifest.existsAsFile()&&!OttoMaterialIO::loadDocument(manifest,document,error))
    {check("existing_fixture_manifest_read",false);log+=error+"\n";return finish();}
    const auto published=OttoMaterialIO::replaceAudio(document,manifest,source,{a,b},error)
        &&OttoMaterialIO::saveDocument(manifest,document,error);
    check("real_otto_fixture_published",published);if(!published){log+=error+"\n";return finish();}
    auto session=MaterialAnnotationSession::open(source,manifest,error);
    check("real_annotation_session_opened",session&&session->rows().size()==2);
    if(!session){log+=error+"\n";return finish();}
    std::vector<juce::String> allIds;for(const auto& r:session->rows())allIds.push_back(r.materialRegionId);
    ProjectData seed;TrackData track;track.id="native-stereo";track.name="Native stereo PCM";
    track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.compose=true;
    track.normalizeVolume=false;track.smoothOverlaps=false;track.nsfSmoothPitchTransitions=false;
    track.volume=1;track.pan=0;seed.tracks={track};ProjectModel model;model.resetDocument(seed);
    const auto first=model.addMaterialPronunciations(source,session->rows(),allIds,0,track.id,error);
    const auto repeat=model.addMaterialPronunciations(source,session->rows(),allIds,3,track.id,error);
    const auto baseline=model.snapshot();
    check("two_real_material_placements_inserted",first.isNotEmpty()&&repeat.isNotEmpty()
        &&baseline.tracks[0].clips.size()==2&&baseline.tracks[0].clips[0].notes.size()==2);
    if(first.isEmpty()||repeat.isEmpty()){log+=error+"\n";return finish();}
    const auto allNeutral=[](const ProjectData& data)
    {
        for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)
            if(!nativePendingPitchIsNeutral(n)||n.sourcePitchMeasured||!n.pitchControlPoints.empty())return false;
        return true;
    };
    check("baseline_unmeasured_neutral_no_fake_f0",allNeutral(baseline));
    const auto verifyRequests=[&](const ProjectData& data,const juce::String& stage)
    {
        bool identity=true;
        for(const auto& t:data.tracks)for(const auto& c:t.clips)
        {
            const auto request=AudioEngine::diagnosticNativeRequest(c,t);
            identity&=request.preserveUneditedSource&&backend::canPreserveNativeSource(request,rate,sourceFrames,sourceFrames);
        }
        check(stage+"_actual_request_allows_source_preservation",identity);
    };
    verifyRequests(baseline,"baseline");
    WavExportOptions options;options.sampleRate=rate;options.channels=2;options.bitDepth=32;
    const auto render=[&](AudioEngine& engine,const ProjectData& data,const juce::File& file,const juce::String& stage)
    {
        check(stage+"_no_audio_device_before_sync",engine.devices().getCurrentAudioDevice()==nullptr);
        engine.syncProject(data);
        const auto deadline=juce::Time::getMillisecondCounterHiRes()+30000;
        while(engine.renderProgress()&&juce::Time::getMillisecondCounterHiRes()<deadline)juce::Thread::sleep(5);
        const bool ready=!engine.renderProgress()&&engine.hasCurrentRenderedAudio()&&engine.hasPlayableRenderedAudio();
        const auto backend=engine.activeRenderBackends(),warnings=engine.activeRenderWarnings();
        log+=stage+" backend="+backend+" warnings="+warnings+"\n";
        check(stage+"_bounded_render_complete",ready);
        check(stage+"_real_backend_is_native_source_preserved",backend=="native-source-preserved");
        check(stage+"_no_backend_warning_or_fallback",warnings.isEmpty()&&!backend.containsIgnoreCase("fallback"));
        if(!ready||backend!="native-source-preserved"||warnings.isNotEmpty())return false;
        const auto written=engine.exportWav(file,error,track.id,0,5,options);
        check(stage+"_real_float32_stereo_export",written);if(!written)log+=error+"\n";
        check(stage+"_no_audio_device_after_export",engine.devices().getCurrentAudioDevice()==nullptr);
        return written;
    };
    AudioEngine engine(false);engine.prepareToPlay(512,rate);
    // If preservation fails, this deliberately missing model must NOT produce a
    // plausible neural/fallback result that could accidentally satisfy the test.
    const auto missingModel=output.getChildFile("missing-neural-model-"+juce::Uuid().toString());
    check("no_model_resources_provided",!missingModel.exists());engine.setHifiganModelDirectory(missingModel);
    const auto beforeFile=output.getChildFile("before-split.wav"),afterFile=output.getChildFile("after-split.wav");
    if(!render(engine,baseline,beforeFile,"baseline"))return finish();
    check("session_splits_first_pronunciation_into_three",session->split({.25,.75},error)&&session->rows().size()==4);
    if(session->rows().size()!=4){log+=error+"\n";return finish();}
    const auto fingerprint=model.contentFingerprint();const auto revision=model.revisionNumber();
    auto preview=model.applySourceSettingsWithReport(source,session->rows(),false);
    check("topology_preflight_without_project_mutation",!preview.rejected&&preview.changed()
        &&model.contentFingerprint()==fingerprint&&model.revisionNumber()==revision);
    if(preview.rejected){log+=preview.diagnostics.joinIntoString("\n")+"\n";return finish();}
    int validationCalls=0;
    session->validatePublication=[&](const std::vector<SampleRegionSetting>& rows,juce::String& e)
    {
        ++validationCalls;const auto r=model.applySourceSettingsWithReport(source,rows,false);
        if(r.rejected)e=r.diagnostics.joinIntoString("\n");return !r.rejected;
    };
    const auto materialSaved=session->save(error);
    check("session_publishes_split_only_after_project_preflight",materialSaved&&validationCalls==1);
    if(!materialSaved){log+=error+"\n";return finish();}
    auto applied=model.applySourceSettingsWithReport(source,session->rows());const auto expanded=model.snapshot();
    check("split_updates_both_placements_in_one_transaction",!applied.rejected&&applied.changed()
        &&model.revisionNumber()==revision+1&&expanded.tracks[0].clips[0].notes.size()==4&&expanded.tracks[0].clips[1].notes.size()==4);
    if(applied.rejected){log+=applied.diagnostics.joinIntoString("\n")+"\n";return finish();}
    check("expanded_members_retain_pcm_neutral_predicate",allNeutral(expanded));
    check("explicit_chains_and_migrated_neighbors_preserved",expanded.nativeConnections.size()==6
        &&expanded.tracks[0].clips[0].notes[0].connectedToNext&&expanded.tracks[0].clips[0].notes[1].connectedToPrevious
        &&expanded.tracks[0].clips[0].notes[2].connectedToNext&&expanded.tracks[0].clips[0].notes[3].connectedToPrevious);
    bool sameSourceClock=true;
    for(std::size_t i=0;i<baseline.tracks[0].clips.size();++i)
    {
        const auto& old=baseline.tracks[0].clips[i];const auto& now=expanded.tracks[0].clips[i];
        sameSourceClock&=old.startSeconds==now.startSeconds&&old.durationSeconds==now.durationSeconds
            &&old.sourceOffsetSeconds==now.sourceOffsetSeconds&&old.sourceDurationSeconds==now.sourceDurationSeconds
            &&old.sourceTimeMap.size()==now.sourceTimeMap.size();
        if(old.sourceTimeMap.size()==now.sourceTimeMap.size())for(std::size_t j=0;j<old.sourceTimeMap.size();++j)
            sameSourceClock&=old.sourceTimeMap[j].sourceSeconds==now.sourceTimeMap[j].sourceSeconds&&old.sourceTimeMap[j].targetSeconds==now.sourceTimeMap[j].targetSeconds;
    }
    check("timeline_source_pcm_coverage_and_clock_unchanged",sameSourceClock);
    verifyRequests(expanded,"expanded");
    if(!render(engine,expanded,afterFile,"expanded"))return finish();
    struct Decoded {juce::AudioBuffer<float> pcm;double rate=0;unsigned int channels=0;int bits=0;bool floating=false,valid=false;};
    const auto decode=[](const juce::File& file)
    {
        Decoded d;auto input=file.createInputStream();if(!input)return d;
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatReader> reader(wav.createReaderFor(input.release(),true));
        if(!reader||reader->lengthInSamples<=0||reader->lengthInSamples>1000000)return d;
        d.rate=reader->sampleRate;d.channels=reader->numChannels;d.bits=reader->bitsPerSample;d.floating=reader->usesFloatingPointData;
        d.pcm.setSize(static_cast<int>(reader->numChannels),static_cast<int>(reader->lengthInSamples));
        d.valid=reader->read(&d.pcm,0,d.pcm.getNumSamples(),0,true,true);return d;
    };
    const auto original=decode(source),before=decode(beforeFile),after=decode(afterFile);
    const auto exportHeader=[&](const Decoded& d){return d.valid&&d.rate==rate&&d.channels==2&&d.bits==32&&d.floating&&d.pcm.getNumSamples()==exportFrames;};
    check("exact_export_channel_rate_float_and_frame_counts",exportHeader(before)&&exportHeader(after));
    if(!exportHeader(before)||!exportHeader(after)||!original.valid)return finish();
    const auto samePcm=[&](const Decoded& a,const Decoded& b,const juce::String& stage)
    {
        bool exact=exportHeader(a)&&exportHeader(b);double maximum=0;
        if(exact)for(int ch=0;ch<2;++ch)
        {
            exact&=std::memcmp(a.pcm.getReadPointer(ch),b.pcm.getReadPointer(ch),static_cast<std::size_t>(exportFrames)*sizeof(float))==0;
            for(int i=0;i<exportFrames;++i)maximum=std::max(maximum,std::abs(static_cast<double>(a.pcm.getSample(ch,i))-b.pcm.getSample(ch,i)));
        }
        log+=stage+" maxDecodedPcmDifference="+juce::String(maximum,16)+"\n";check(stage,exact);
    };
    samePcm(before,after,"before_after_decoded_pcm_bytes_identical");
    bool sourceExact=original.channels==2&&original.rate==rate&&original.pcm.getNumSamples()==sourceFrames;
    if(sourceExact)for(int ch=0;ch<2;++ch)
    {
        sourceExact&=std::memcmp(before.pcm.getReadPointer(ch),original.pcm.getReadPointer(ch),static_cast<std::size_t>(sourceFrames)*sizeof(float))==0;
        sourceExact&=std::memcmp(before.pcm.getReadPointer(ch,secondStart),original.pcm.getReadPointer(ch),static_cast<std::size_t>(sourceFrames)*sizeof(float))==0;
        sourceExact&=before.pcm.getMagnitude(ch,sourceFrames,rate)==0;
    }
    check("both_placements_are_exact_decoded_source_pcm_with_silent_gap",sourceExact);
    check("non_silent_independent_stereo_channels",before.pcm.getRMSLevel(0,0,sourceFrames)>.05f&&before.pcm.getRMSLevel(1,0,sourceFrames)>.05f
        &&std::memcmp(before.pcm.getReadPointer(0),before.pcm.getReadPointer(1),static_cast<std::size_t>(sourceFrames)*sizeof(float))!=0);
    ProjectModel reopened;const auto projectFile=output.getChildFile("expanded-topology-audio.hachi");
    const auto saved=model.save(projectFile,error)&&reopened.load(projectFile,error);
    check("expanded_project_saved_and_reopened",saved);if(!saved){log+=error+"\n";return finish();}
    check("reopened_members_still_unmeasured_neutral",allNeutral(reopened.snapshot()));
    const auto reopenedFile=output.getChildFile("after-reopen.wav");
    {
        AudioEngine fresh(false);fresh.prepareToPlay(512,rate);fresh.setHifiganModelDirectory(missingModel);
        if(!render(fresh,reopened.snapshot(),reopenedFile,"reopened"))return finish();
        fresh.releaseResources();
    }
    const auto redecoded=decode(reopenedFile);check("reopened_export_exact_header",exportHeader(redecoded));
    samePcm(before,redecoded,"save_reopen_export_decoded_pcm_bytes_identical");
    juce::MemoryBlock finalBytes;check("source_wav_file_bytes_unchanged",source.loadFileAsData(finalBytes)&&sourceBytes==finalBytes);
    engine.releaseResources();return finish();
}
}
