#pragma once
#include "../AudioEngine.h"
#include "../ProjectModel.h"
#include "../backend/NsfHifiganRenderer.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace hachi
{
namespace forcedPortamentoAudioSmokeDetail
{
struct PitchMeasurement { double time=0, midi=0, confidence=0; bool voiced=false; };

inline PitchMeasurement measurePitch(const juce::AudioBuffer<float>& audio, double rate, double time)
{
    // Unconditioned normalized autocorrelation: no requested F0 is supplied to
    // the estimator. Range MIDI50..78 covers both plateaus and the whole glide.
    PitchMeasurement result; result.time=time;
    const auto size=static_cast<int>(std::lround(.035*rate));
    const auto first=static_cast<int>(std::lround(time*rate))-size/2;
    if(first<0||first+size>audio.getNumSamples()||audio.getNumChannels()<1)return result;
    std::vector<double> wave(static_cast<std::size_t>(size));double mean=0,power=0;
    for(int i=0;i<size;++i)
    {
        double value=0;for(int c=0;c<audio.getNumChannels();++c)value+=audio.getSample(c,first+i);
        wave[static_cast<std::size_t>(i)]=value/audio.getNumChannels();mean+=wave[static_cast<std::size_t>(i)];
    }
    mean/=size;
    for(auto& value:wave){value-=mean;power+=value*value;}
    if(power/size<1.e-8)return result;
    const auto hz=[](double midi){return 440.0*std::exp2((midi-69)/12);};
    const auto low=std::max(2,static_cast<int>(std::floor(rate/hz(78))));
    const auto high=std::min(size/2,static_cast<int>(std::ceil(rate/hz(50))));
    std::vector<double> correlation(static_cast<std::size_t>(high+2),-1);
    double maximum=-1;
    for(int lag=low-1;lag<=high+1;++lag)
    {
        double xy=0,xx=0,yy=0;
        for(int i=0;i+lag<size;++i)
        {const auto a=wave[static_cast<std::size_t>(i)],b=wave[static_cast<std::size_t>(i+lag)];xy+=a*b;xx+=a*a;yy+=b*b;}
        const auto value=xy/std::sqrt(std::max(1.e-24,xx*yy));correlation[static_cast<std::size_t>(lag)]=value;
        if(lag>=low&&lag<=high)maximum=std::max(maximum,value);
    }
    int best=-1;
    // Prefer the first convincing local period, not an octave-down multiple.
    for(int lag=low;lag<=high;++lag)
        if(correlation[static_cast<std::size_t>(lag)]>=std::max(.60,maximum*.90)
            &&correlation[static_cast<std::size_t>(lag)]>=correlation[static_cast<std::size_t>(lag-1)]
            &&correlation[static_cast<std::size_t>(lag)]>correlation[static_cast<std::size_t>(lag+1)]){best=lag;break;}
    if(best<0)return result;
    const auto a=correlation[static_cast<std::size_t>(best-1)],b=correlation[static_cast<std::size_t>(best)],c=correlation[static_cast<std::size_t>(best+1)];
    const auto denominator=a-2*b+c;
    const auto fraction=std::abs(denominator)>1.e-12?juce::jlimit(-.5,.5,.5*(a-c)/denominator):0;
    result.confidence=b;result.midi=69+12*std::log2(rate/(best+fraction)/440);
    result.voiced=std::isfinite(result.midi)&&b>=.60;
    return result;
}
inline double median(std::vector<double> values)
{
    if(values.empty())return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
inline double pcmDifference(const juce::AudioBuffer<float>& a,const juce::AudioBuffer<float>& b)
{
    if(a.getNumChannels()!=b.getNumChannels()||a.getNumSamples()!=b.getNumSamples())return std::numeric_limits<double>::infinity();
    double maximum=0;for(int c=0;c<a.getNumChannels();++c)for(int i=0;i<a.getNumSamples();++i)
    {
        if(!std::isfinite(a.getSample(c,i))||!std::isfinite(b.getSample(c,i)))return std::numeric_limits<double>::infinity();
        maximum=std::max(maximum,double(std::abs(a.getSample(c,i)-b.getSample(c,i))));
    }
    return maximum;
}
inline bool finiteAudible(const juce::AudioBuffer<float>& audio,int expectedSamples=96000)
{
    if(audio.getNumChannels()!=2||audio.getNumSamples()!=expectedSamples)return false;
    double power=0;for(int c=0;c<2;++c)for(int i=0;i<audio.getNumSamples();++i)
    {const auto value=audio.getSample(c,i);if(!std::isfinite(value))return false;power+=double(value)*value;}
    const auto rms=std::sqrt(power/(2*audio.getNumSamples()));return rms>.005&&rms<.8;
}
struct AcousticMetrics
{
    std::vector<PitchMeasurement> points;
    double left=0,right=0,leftCoverage=0,rightCoverage=0,bridgeCoverage=0;
    double targetRmsCents=std::numeric_limits<double>::infinity(),latency=0,seamJump=0;
    int intermediate=0;
};
inline AcousticMetrics acousticMetrics(const juce::AudioBuffer<float>& audio,double rate,
                                      const ResolvedConnectionPitchCurve* expected=nullptr)
{
    AcousticMetrics result;
    for(int i=0;i<9;++i){result.points.push_back(measurePitch(audio,rate,.3+i*.05));result.points.push_back(measurePitch(audio,rate,1.3+i*.05));}
    for(int i=0;i<=40;++i)result.points.push_back(measurePitch(audio,rate,.9+i*.005));
    std::sort(result.points.begin(),result.points.end(),[](const auto& a,const auto& b){return a.time<b.time;});
    std::vector<double> left,right,before,after;int bridgeTotal=0,bridgeVoiced=0;
    for(const auto& p:result.points)
    {
        if(p.voiced&&p.time>=.299&&p.time<=.701)left.push_back(p.midi);
        if(p.voiced&&p.time>=1.299&&p.time<=1.701)right.push_back(p.midi);
        if(p.time>=.95-1.e-8&&p.time<=1.05+1.e-8&&std::abs(p.time-1)>=.010-1.e-8)
        {
            ++bridgeTotal;if(p.voiced){++bridgeVoiced;if(p.midi>60.65&&p.midi<66.35)++result.intermediate;}
        }
        if(p.voiced&&p.time>=.979&&p.time<=.991)before.push_back(p.midi);
        if(p.voiced&&p.time>=1.009&&p.time<=1.021)after.push_back(p.midi);
    }
    result.left=median(left);result.right=median(right);result.leftCoverage=left.size()/9.0;result.rightCoverage=right.size()/9.0;
    result.bridgeCoverage=bridgeTotal?double(bridgeVoiced)/bridgeTotal:0;
    result.seamJump=std::abs(median(after)-median(before));
    if(expected)
        for(int offset=-15;offset<=15;++offset)
        {
            const auto latency=offset*.001;double squared=0;int count=0;
            for(const auto& p:result.points)
                if(p.voiced&&p.time>=expected->fromSeconds-1.e-8&&p.time<=expected->toSeconds+1.e-8&&std::abs(p.time-1)>=.010-1.e-8)
                {const auto wanted=evaluatePitchCurve(expected->points,p.time-latency);const auto cents=(p.midi-wanted)*100;squared+=cents*cents;++count;}
            if(count>0&&std::sqrt(squared/count)<result.targetRmsCents){result.targetRmsCents=std::sqrt(squared/count);result.latency=latency;}
        }
    return result;
}
inline juce::var metricsJson(const AcousticMetrics& metrics)
{
    const auto numeric=[](double value){return std::isfinite(value)?juce::var(value):juce::var();};
    auto object=new juce::DynamicObject();object->setProperty("leftMidi",numeric(metrics.left));object->setProperty("rightMidi",numeric(metrics.right));
    object->setProperty("leftVoicedCoverage",metrics.leftCoverage);object->setProperty("rightVoicedCoverage",metrics.rightCoverage);
    object->setProperty("bridgeVoicedCoverage",metrics.bridgeCoverage);object->setProperty("intermediateSamples",metrics.intermediate);
    object->setProperty("bridgeTargetRmsCents",std::isfinite(metrics.targetRmsCents)?metrics.targetRmsCents:-1);
    object->setProperty("bestFixedLatencySeconds",metrics.latency);object->setProperty("nearSeamMidiJump",numeric(metrics.seamJump));
    juce::Array<juce::var> samples;for(const auto& p:metrics.points){auto item=new juce::DynamicObject();item->setProperty("seconds",p.time);
        item->setProperty("midi",p.midi);item->setProperty("confidence",p.confidence);item->setProperty("voiced",p.voiced);samples.add(juce::var(item));}
    object->setProperty("samples",juce::var(samples));return juce::var(object);
}
}

inline bool runForcedPortamentoAudioSmoke(const juce::File& output,const juce::File& nsfModel)
{
    using namespace forcedPortamentoAudioSmokeDetail;
    if(!output.createDirectory())return false;
    bool ok=true;int checks=0;juce::Array<juce::var> results;auto report=new juce::DynamicObject();juce::var ownedReport(report);
    const auto check=[&](const char* name,bool pass){ok&=pass;++checks;std::cout<<name<<'='<<pass<<std::endl;
        auto item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("pass",pass);results.add(juce::var(item));};
    // Fixed before any inference: failures are reported, never used to loosen
    // these gates. This is an acoustic test, not merely a request-array test.
    auto limits=new juce::DynamicObject();limits->setProperty("plateauCenterToleranceCents",35);limits->setProperty("plateauVoicedCoverage",.90);
    limits->setProperty("bridgeVoicedCoverage",.70);limits->setProperty("bridgeTargetRmsToleranceCents",100);
    limits->setProperty("maxFixedLatencySeconds",.015);limits->setProperty("seamExclusionSeconds",.010);
    limits->setProperty("minimumIntermediateSamples",6);limits->setProperty("minimumMoreIntermediateThanOff",3);
    limits->setProperty("minimumSeamJumpReductionSemitones",1.5);limits->setProperty("coldReloadPcmMaxDifference",1.e-5);
    limits->setProperty("autocorrelationWindowSeconds",.035);limits->setProperty("autocorrelationMidiRange","50..78");
    report->setProperty("predeclaredLimits",juce::var(limits));report->setProperty("modelDirectory",nsfModel.getFullPathName());
    const auto finish=[&]{report->setProperty("ok",ok);report->setProperty("checks",checks);report->setProperty("results",juce::var(results));
        const bool saved=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(ownedReport));
        std::cout<<"forced_portamento_audio_ok="<<(ok&&saved)<<"; checks="<<checks<<std::endl;return ok&&saved;};
    check("real_nsf_model_available_no_fallback",backend::NsfHifiganRenderer::modelAvailable(nsfModel));if(!ok)return finish();
    constexpr double rate=48000;constexpr int sourceSamples=48000;
    const auto sourceA=output.getChildFile("voiced-220.wav"),sourceB=output.getChildFile("voiced-440.wav");
    const auto writeSource=[&](const juce::File& file,double hz)
    {
        juce::AudioBuffer<float> wave(1,sourceSamples);
        for(int i=0;i<sourceSamples;++i){const auto phase=juce::MathConstants<double>::twoPi*hz*i/rate;
            wave.setSample(0,i,static_cast<float>(.17*std::sin(phase)+.06*std::sin(2*phase)+.025*std::sin(3*phase)+.01*std::sin(5*phase)));}
        auto stream=file.createOutputStream();if(!stream)return false;stream->setPosition(0);stream->truncate();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.get(),rate,1,24,{},0));if(!writer)return false;
        stream.release();return writer->writeFromAudioSampleBuffer(wave,0,sourceSamples);
    };
    check("two_different_voiced_source_fixtures_written",writeSource(sourceA,220)&&writeSource(sourceB,440));if(!ok)return finish();
    const auto hashA=juce::SHA256(sourceA).toHexString(),hashB=juce::SHA256(sourceB).toHexString();
    check("source_files_are_physically_distinct",sourceA!=sourceB&&hashA!=hashB);
    const auto makeClip=[&](const char* id,const char* noteId,const juce::File& file,double start,float sourceMidi,float targetMidi)
    {
        ClipData clip;clip.id=id;clip.startSeconds=start;clip.durationSeconds=clip.sourceDurationSeconds=1;clip.sourceFile=file;clip.sourceTimeMap={{0,0},{1,1}};
        NoteData note;note.id=noteId;note.label="voiced";note.durationSeconds=1;note.midiNote=targetMidi;note.sourceMidiCenter=sourceMidi;note.sourcePitchMeasured=true;
        note.nativeSourceStartSeconds=0;note.nativeSourceEndSeconds=1;note.nativeRole=NativeSegmentRole::vowel;note.utauAutoPitchTransition=false;
        note.consonantSeconds=0;note.gain=1;for(int i=0;i<=200;++i)note.contour.push_back({i*.005,0,0,true,0,false});clip.notes={note};return clip;
    };
    TrackData main;main.id="forced-main";main.name="Forced cross-source";main.compose=true;main.nativeNsfAudio=true;main.pitchAlgorithm=PitchAlgorithm::nsfHifigan;
    main.renderOrder=RenderOrder::processThenSplice;main.stretchAlgorithm=StretchAlgorithm::variableMelHop;
    main.normalizeVolume=false;main.nsfNoiseProtection=false;main.nsfSmoothPitchTransitions=false;
    main.clips={makeClip("left-source","left",sourceA,0,57,60),makeClip("right-source","right",sourceB,1,69,67)};
    auto unrelated=main;unrelated.id="unrelated";unrelated.name="Unrelated";unrelated.clips={makeClip("unrelated-source","third",sourceB,0,69,72)};
    ProjectData data;data.bpm=120;data.tracks={main,unrelated};ProjectModel model;model.resetDocument(data);const auto off=model.snapshot();
    const auto musicHash=[](ProjectData project){project.nativeConnections.clear();ProjectModel snapshot;snapshot.resetDocument(std::move(project));return snapshot.contentFingerprint();};
    const auto baselineMusic=musicHash(off),baselineFingerprint=model.contentFingerprint();
    struct Render{juce::AudioBuffer<float> main,unrelated;double rate=0;juce::String backend,warning,error;bool done=false,noDevice=false;};
    const auto render=[&](const ProjectData& project,const char* tag)
    {
        Render result;AudioEngine audio(false);audio.setHifiganModelDirectory(nsfModel);audio.setInferenceConfiguration(backend::InferenceBackend::cpu,-1);
        audio.prepareToPlay(256,48000);audio.syncProject(project);const auto deadline=juce::Time::getMillisecondCounterHiRes()+180000;
        while(audio.renderProgress().has_value()&&juce::Time::getMillisecondCounterHiRes()<deadline)juce::Thread::sleep(5);
        if(audio.renderProgress().has_value()){result.error="Render deadline exceeded";return result;}
        result.backend=audio.activeRenderBackends();result.warning=audio.activeRenderWarnings();
        WavExportOptions options;options.sampleRate=48000;options.channels=2;options.bitDepth=32;
        const auto mainFile=output.getChildFile(juce::String(tag)+".wav"),otherFile=output.getChildFile(juce::String(tag)+"-unrelated.wav");
        if(!audio.exportWav(mainFile,result.error,"forced-main",0,2,options)||!audio.exportWav(otherFile,result.error,"unrelated",0,1,options))return result;
        juce::AudioFormatManager formats;formats.registerBasicFormats();
        const auto read=[&](const juce::File& file,juce::AudioBuffer<float>& wave,double* rateOut)
        {
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));if(!reader||reader->lengthInSamples>10000000)return false;
            wave.setSize(static_cast<int>(reader->numChannels),static_cast<int>(reader->lengthInSamples));
            if(rateOut)*rateOut=reader->sampleRate;return reader->read(&wave,0,wave.getNumSamples(),0,true,true);
        };
        result.done=read(mainFile,result.main,&result.rate)&&read(otherFile,result.unrelated,nullptr);
        result.noDevice=!audio.isPlaying()&&audio.devices().getCurrentAudioDevice()==nullptr;
        return result;
    };
    const auto checkRender=[&](const char* key,const Render& rendered)
    {
        auto object=new juce::DynamicObject();object->setProperty("backend",rendered.backend);object->setProperty("warning",rendered.warning);object->setProperty("error",rendered.error);
        report->setProperty(key,juce::var(object));check(key,rendered.done&&rendered.rate==48000&&finiteAudible(rendered.main)&&finiteAudible(rendered.unrelated,48000)&&rendered.noDevice
            &&rendered.backend.contains("nsf-hifigan-onnx")&&rendered.backend.containsIgnoreCase("cpu")&&!rendered.backend.containsIgnoreCase("fallback"));
    };
    const auto baseline=render(off,"off");checkRender("off_real_cpu_nsf_audio",baseline);if(!baseline.done)return finish();
    juce::String error;check("force_adds_graph_only_one_transaction",model.setNotesPortamento({"left","right"},true,error,.1));
    const auto on=model.snapshot();const auto initiallyForcedFingerprint=model.contentFingerprint();
    check("force_has_one_undo_back_to_original_score",model.undo()&&model.contentFingerprint()==baselineFingerprint);
    check("force_redo_restores_exact_graph",model.redo()&&model.contentFingerprint()==initiallyForcedFingerprint);
    check("force_preserves_all_music_source_clock_and_amplitude_data",musicHash(on)==baselineMusic);
    check("force_is_pitch_only_not_amplitude_splice",on.nativeConnections.size()==1&&on.nativeConnections[0].type=="pitch-portamento"
        &&on.nativeConnections[0].amplitudeCurve.empty()&&!on.tracks[0].clips[0].notes[0].utauSplice&&!on.tracks[0].clips[1].notes[0].utauSplice
        &&!on.tracks[0].clips[0].notes[0].connectedToNext&&!on.tracks[0].clips[1].notes[0].connectedToPrevious);
    const auto resolved=resolvedConnectionPitchCurves(on.tracks[0],on.nativeConnections);
    check("forced_curve_resolves_on_live_cross_file_endpoints",resolved.size()==1&&resolved[0].active&&resolved[0].points.size()>=2
        &&std::abs(resolved[0].fromSeconds-.95)<1.e-6&&std::abs(resolved[0].toSeconds-1.05)<1.e-6);
    if(resolved.empty()||!resolved[0].active)return finish();
    const auto shared=sharedPitchLines(on.tracks[0],on.nativeConnections);
    check("graph_aware_shared_line_exists_for_both_files",shared.memberFor("left")&&shared.memberFor("right"));
    const auto requestOff=AudioEngine::diagnosticNativeTargetMidi(off,0,0),requestOn=AudioEngine::diagnosticNativeTargetMidi(on,0,0);
    check("manual_force_changes_actual_graph_aware_request",requestOff!=requestOn);
    check("unrelated_track_request_is_unchanged",AudioEngine::diagnosticNativeTargetMidi(off,1,0)==AudioEngine::diagnosticNativeTargetMidi(on,1,0));
    const auto forced=render(on,"on");checkRender("on_real_cpu_nsf_audio",forced);if(!forced.done)return finish();
    const auto offMetrics=acousticMetrics(baseline.main,baseline.rate),onMetrics=acousticMetrics(forced.main,forced.rate,&resolved[0]);
    report->setProperty("offAcoustics",metricsJson(offMetrics));report->setProperty("onAcoustics",metricsJson(onMetrics));
    check("off_actual_F0_centers_match_60_and_67",std::abs(offMetrics.left-60)<=.35&&std::abs(offMetrics.right-67)<=.35
        &&offMetrics.leftCoverage>=.9&&offMetrics.rightCoverage>=.9);
    check("on_actual_F0_centers_match_60_and_67",std::abs(onMetrics.left-60)<=.35&&std::abs(onMetrics.right-67)<=.35
        &&onMetrics.leftCoverage>=.9&&onMetrics.rightCoverage>=.9);
    check("actual_F0_tracks_resolved_graph_curve_not_only_request",onMetrics.bridgeCoverage>=.7&&onMetrics.targetRmsCents<=100);
    check("actual_on_glide_has_more_intermediate_F0_than_off_step",onMetrics.intermediate>=6&&onMetrics.intermediate>=offMetrics.intermediate+3);
    check("actual_on_glide_is_more_continuous_than_off_step",std::isfinite(onMetrics.seamJump)&&std::isfinite(offMetrics.seamJump)
        &&onMetrics.seamJump+1.5<=offMetrics.seamJump);
    check("unrelated_actual_audio_is_unchanged_by_force",pcmDifference(baseline.unrelated,forced.unrelated)<=1.e-5);
    const auto forcedFingerprint=model.contentFingerprint();const auto forcedFile=output.getChildFile("forced.hjpx");ProjectModel reopened;
    check("forced_project_saves_and_reopens",model.save(forcedFile,error)&&reopened.load(forcedFile,error));
    const auto cold=render(reopened.snapshot(),"on-cold-reopened");checkRender("cold_on_real_cpu_nsf_audio",cold);
    const auto coldDifference=pcmDifference(forced.main,cold.main);report->setProperty("coldOnPcmMaximumDifference",std::isfinite(coldDifference)?juce::var(coldDifference):juce::var());
    check("application_cache_cold_reload_preserves_forced_PCM",cold.done&&coldDifference<=1.e-5);
    auto points=resolved[0].points;const auto midpoint=(resolved[0].fromSeconds+resolved[0].toSeconds)*.5;
    const auto baseMidi=evaluatePitchCurve(points,midpoint);
    points.insert(points.begin()+1,{midpoint,baseMidi+1.2f,PitchCurveShape::smooth});
    check("interior_graph_pitch_edit_commits",model.setConnectionPitchCurve(resolved[0].connectionId,points,error));
    const auto edited=model.snapshot();check("interior_graph_edit_does_not_touch_music_or_unrelated_source",musicHash(edited)==baselineMusic);
    const auto editedResolved=resolvedConnectionPitchCurves(edited.tracks[0],edited.nativeConnections);
    const auto editedAudio=render(edited,"edited");checkRender("edited_real_cpu_nsf_audio",editedAudio);
    check("unrelated_actual_audio_is_unchanged_by_interior_graph_edit",editedAudio.done&&pcmDifference(baseline.unrelated,editedAudio.unrelated)<=1.e-5);
    if(!editedResolved.empty()&&editedResolved[0].active&&editedAudio.done)
    {
        const auto editedMetrics=acousticMetrics(editedAudio.main,editedAudio.rate,&editedResolved[0]);report->setProperty("editedAcoustics",metricsJson(editedMetrics));
        check("edited_graph_curve_is_audible_and_actual_F0_tracks_it",editedMetrics.bridgeCoverage>=.7&&editedMetrics.targetRmsCents<=100
            &&pcmDifference(forced.main,editedAudio.main)>1.e-5);
    }
    else check("edited_graph_curve_is_audible_and_actual_F0_tracks_it",false);
    const auto editedFile=output.getChildFile("edited.hjpx");ProjectModel editedReopened;
    check("edited_graph_project_saves_and_reopens",model.save(editedFile,error)&&editedReopened.load(editedFile,error));
    const auto editedCold=render(editedReopened.snapshot(),"edited-cold-reopened");checkRender("cold_edited_real_cpu_nsf_audio",editedCold);
    const auto editedDifference=pcmDifference(editedAudio.main,editedCold.main);report->setProperty("coldEditedPcmMaximumDifference",std::isfinite(editedDifference)?juce::var(editedDifference):juce::var());
    check("application_cache_cold_reload_preserves_edited_PCM",editedCold.done&&editedDifference<=1.e-5);
    check("one_undo_restores_original_forced_curve",model.undo()&&model.contentFingerprint()==forcedFingerprint);
    check("remove_force_returns_original_score_state",model.setNotesPortamento({"left","right"},false,error,.1)
        &&model.snapshot().nativeConnections.empty()&&model.contentFingerprint()==baselineFingerprint);
    const auto removed=render(model.snapshot(),"removed");checkRender("removed_force_real_cpu_nsf_audio",removed);
    const auto removedDifference=pcmDifference(baseline.main,removed.main);report->setProperty("removedForcePcmMaximumDifference",std::isfinite(removedDifference)?juce::var(removedDifference):juce::var());
    check("removing_force_restores_baseline_actual_PCM",removed.done&&removedDifference<=1.e-5);
    check("source_PCM_files_are_byte_identical_after_all_force_edits",juce::SHA256(sourceA).toHexString()==hashA&&juce::SHA256(sourceB).toHexString()==hashB);
    report->setProperty("source220Sha256",hashA);report->setProperty("source440Sha256",hashB);
    return finish();
}
}
