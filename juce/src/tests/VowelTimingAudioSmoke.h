#pragma once
#include "../AudioEngine.h"
#include "../MaterialAnnotationSession.h"
#include "../NativeNoteTiming.h"
#include "../PronunciationTiming.h"
#include "../backend/NsfHifiganRenderer.h"
#include "../backend/SourceVoicing.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace hachi
{
// Actual offline neural rendering, not just timing/request tests. The recording
// and its F0 are synthetic ground truth, NOT a real-recording analysis or an
// original-project reference. No audio device or audition is opened.
inline bool runVowelTimingAudioSmoke(const juce::File& output,const juce::File& nsfModel)
{
    const auto root=output.getChildFile("vowel-timing-audio-"+juce::Uuid().toString());
    if(root.createDirectory().failed())return false;
    bool ok=true;int total=0;juce::String log,error;juce::Array<juce::var> results,sourceMaskEvidence;
    // Declared before rendering. These are acceptance limits, not adaptive
    // thresholds: a failure is reported rather than increasing the tolerance.
    constexpr int rate=48000;
    constexpr double peakTolerance=.001,minimumBurstRms=.005,minimumCorrelation=.98,maximumRelativeRmsDifference=.05;
    constexpr double reloadMaximumError=1.e-5,reloadRelativeRms=1.e-5;
    // One physical definition fixed before classifier/NN probes, never tuned to
    // a passing seed: continuous frication alone receives 25% of original C energy.
    constexpr double fricationBedEnergyFraction=.25;
    double fixtureBedAmplitude=0,fixtureBurstAmplitude=0;
    const auto check=[&](const juce::String& name,bool pass,const juce::String& detail=juce::String{})
    {
        ++total;ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+(detail.isEmpty()?juce::String{}:" | "+detail)+"\n";
        auto* record=new juce::DynamicObject();record->setProperty("name",name);record->setProperty("passed",pass);record->setProperty("detail",detail);results.add(juce::var(record));
        std::cout<<name<<"="<<(pass?"PASS":"FAIL")<<" "<<detail<<std::endl;
    };
    const auto finish=[&]
    {
        auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",total);
        report->setProperty("sourceBasis","generated stereo 220Hz vowels and continuous high-frequency aperiodic C bed plus Gaussian bursts, matched to original total C energy; known synthetic F0, not real-recording analysis");
        report->setProperty("fixtureRevision","continuous-HF-frication-25percent-energy-v4");
        report->setProperty("fixtureCorrection","Build13b periodic cosine C was incorrectly labelled UV; v2/v3 had weak-tail and partial-window classification failures. V4 assigns a predeclared 25% of ORIGINAL total C energy to continuous frication, retaining the same balanced random carrier/seed and solving Gaussian amplitude including the overlap cross-term. Original Gaussian times/sigma, TOTAL C energy, vowel PCM and ALL acceptance gates retained; burst peak is lower, not gain-boosted.");
        report->setProperty("fricationBedStandaloneEnergyFraction",fricationBedEnergyFraction);
        report->setProperty("fricationBedAmplitude",fixtureBedAmplitude);
        report->setProperty("gaussianBurstAmplitude",fixtureBurstAmplitude);
        report->setProperty("sourcePreflight","Actual C++ SourceVoicing on decoded source mono at 48000Hz, before ONNX; model inference and its normal input conversion remain unchanged.");
        report->setProperty("audioDevicesOpenedByTest",false);report->setProperty("peakTimingToleranceSeconds",peakTolerance);
        report->setProperty("minimumBurstRms",minimumBurstRms);report->setProperty("minimumProtectedWindowCorrelation",minimumCorrelation);
        report->setProperty("maximumProtectedWindowRelativeRmsDifference",maximumRelativeRmsDifference);
        report->setProperty("reloadMaximumAbsolutePcmError",reloadMaximumError);report->setProperty("reloadMaximumRelativeRmsError",reloadRelativeRms);
        report->setProperty("sourceCppDryMaskFrames",sourceMaskEvidence);
        report->setProperty("results",results);
        const bool written=root.getChildFile("verification.txt").replaceWithText(log)
            &&root.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
        std::cout<<"vowel_timing_audio_passed="<<(ok?1:0)<<"|checks="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;return ok&&written;
    };
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-7;};
    check("actual_neural_model_available",backend::NsfHifiganRenderer::modelAvailable(nsfModel),nsfModel.getFullPathName());
    if(!ok)return finish();
    const auto source=root.getChildFile("two-pronunciations-stereo.wav"),manifest=root.getChildFile("otto.ini");
    {
        juce::AudioBuffer<float> pcm(2,rate*2);
        juce::Random carrier(0xC501CE);
        std::array<std::vector<float>,2> signs;
        for(auto& channel:signs)
        {
            channel.resize(rate);
            for(auto& value:channel)value=carrier.nextBool()?1.0f:-1.0f;
        }
        const auto oldBurstAmplitude=.55*std::sqrt((1.0+.18*.18)/2.0);
        double gaussianSum=0,gaussianSquareSum=0;int consonantSamples=0;
        for(int i=0;i<rate/5;++i)
        {
            const auto local=static_cast<double>(i)/rate;double gaussian=0;
            for(const double centre:{.06,.14}){const auto x=local-centre;gaussian+=std::exp(-.5*x*x/(.0035*.0035));}
            gaussianSum+=gaussian;gaussianSquareSum+=gaussian*gaussian;++consonantSamples;
        }
        // Solve A^2*sum(g^2)+2*A*b*sum(g)+N*b^2 = old C energy.
        // The compensated envelope peak A+b is LOWER than the old peak.
        const auto oldCenergy=oldBurstAmplitude*oldBurstAmplitude*gaussianSquareSum;
        const auto bedAmplitude=std::sqrt(fricationBedEnergyFraction*oldCenergy/consonantSamples);
        const auto bedLinear=bedAmplitude*gaussianSum;
        const auto burstAmplitude=(-bedLinear+std::sqrt(bedLinear*bedLinear+gaussianSquareSum
            *(oldCenergy-consonantSamples*bedAmplitude*bedAmplitude)))/gaussianSquareSum;
        fixtureBedAmplitude=bedAmplitude;fixtureBurstAmplitude=burstAmplitude;
        check("physical_C_bed_receives_predeclared_quarter_energy",
            std::abs(consonantSamples*bedAmplitude*bedAmplitude/oldCenergy-fricationBedEnergyFraction)<1.e-12,
            "standalone_bed_fraction="+juce::String(consonantSamples*bedAmplitude*bedAmplitude/oldCenergy,12));
        check("continuous_C_bed_compensated_peak_not_increased",burstAmplitude>0&&burstAmplitude+bedAmplitude<=oldBurstAmplitude,
            "bed="+juce::String(bedAmplitude,9)+" burst="+juce::String(burstAmplitude,12)
            +" compensated_peak="+juce::String(burstAmplitude+bedAmplitude,12)+" prior_peak="+juce::String(oldBurstAmplitude,12));
        for(int i=0;i<pcm.getNumSamples();++i)
        {
            const auto t=static_cast<double>(i)/rate,local=t-std::floor(t);
            for(int ch=0;ch<2;++ch)
            {
                double value=0;
                if(local<.2)
                {
                    // Balanced random polarity pairs [s,-s] on a 16k grid,
                    // each held for 3 source samples at48k: aperiodic HF colour,
                    // same seed, constant carrier energy, no seed/gate search.
                    const auto frame16k=(i%rate)/3;
                    const auto polarity=(frame16k%2)==0?1.0:-1.0;
                    const auto sign=polarity*signs[static_cast<std::size_t>(ch)][static_cast<std::size_t>(frame16k/2)];
                    double gaussian=0;
                    for(const double centre:{.06,.14})
                    {
                        const auto x=local-centre;
                        gaussian+=std::exp(-.5*x*x/(.0035*.0035));
                    }
                    value=(bedAmplitude+burstAmplitude*gaussian)*sign;
                }
                else
                {
                    const auto fade=std::min({1.0,(local-.2)/.012,(1-local)/.012});
                    value=fade*(.20*std::sin(juce::MathConstants<double>::twoPi*220*t+.07*ch)
                        +.055*std::sin(juce::MathConstants<double>::twoPi*440*t)
                        +.025*std::sin(juce::MathConstants<double>::twoPi*660*t));
                }
                pcm.setSample(ch,i,static_cast<float>(value));
            }
        }
        auto out=source.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?wav.createWriterFor(out.release(),rate,2,24,{},0):nullptr);
        check("synthetic_stereo_fixture_written",writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));
    }
    juce::MemoryBlock originalBytes;check("source_bytes_captured",source.loadFileAsData(originalBytes));if(!ok)return finish();
    auto session=MaterialAnnotationSession::open(source,manifest,error);
    check("actual_material_session",session&&session->rows().empty(),error);if(!session)return finish();
    check("actual_first_pronunciation",session->createPronunciation(0,1,"first",error),error);
    if(session->rows().size()!=1)return finish();const auto firstRegion=session->rows()[0].materialRegionId;
    check("actual_adjacent_second_pronunciation",session->createAdjacentPronunciation(firstRegion,false,2,"second",error),error);
    if(session->rows().size()!=2)return finish();
    const auto authoredRows=session->rows();
    for(auto row:authoredRows)
    {
        auto timing=pronunciationTimingFor(row);timing.alignmentSeconds=row.regionStartSeconds+.10;
        timing.consonantEndSeconds=row.regionStartSeconds+.20;timing.overlapSeconds=0;
        check("canonical_C_P_"+row.name,applyPronunciationTiming(row,timing,error)&&session->setRow(row.materialRegionId,row,error),error);
    }
    check("actual_otto_annotation_saved",session->save(error),error);
    auto annotated=MaterialAnnotationSession::open(source,manifest,error);
    check("otto_roundtrip_C_P_and_shared_audio",annotated&&annotated->rows().size()==2
        &&annotated->rows()[0].materialAudioId==annotated->rows()[1].materialAudioId
        &&near(annotated->rows()[0].fixedDurationSeconds,.2)&&near(annotated->rows()[0].alignmentSeconds,.1)
        &&near(annotated->rows()[1].fixedDurationSeconds,.2)&&near(annotated->rows()[1].alignmentSeconds,1.1),error);
    if(!annotated||annotated->rows().size()!=2)return finish();
    ProjectData emptyScore;emptyScore.bpm=120;emptyScore.gridDivision="1/16";
    ProjectModel model;model.resetDocument(emptyScore);const auto trackId=model.addTrack("Neural protected C timing",true);
    model.setTrackPitchAlgorithm(trackId,PitchAlgorithm::nsfHifigan);model.setTrackNsfNoiseProtection(trackId,true);
    model.setTrackNsfSmoothPitchTransitions(trackId,false);model.setTrackNormalizeVolume(trackId,false);
    model.setTrackSmoothOverlaps(trackId,false);model.setTrackVolume(trackId,.35f);
    std::vector<juce::String> regionIds;for(const auto& row:annotated->rows())regionIds.push_back(row.materialRegionId);
    const auto clipId=model.addMaterialPronunciations(source,annotated->rows(),regionIds,0,trackId,error);
    check("actual_material_placement",clipId.isNotEmpty()&&model.snapshot().tracks.size()==1
        &&model.snapshot().tracks[0].clips.size()==1&&model.snapshot().tracks[0].clips[0].notes.size()==2,error);
    if(clipId.isEmpty()||model.snapshot().tracks[0].clips.empty())return finish();
    const auto imported=model.snapshot().tracks[0].clips[0];
    NoteData knownF0;knownF0.id="synthetic-ground-truth";knownF0.startSeconds=0;knownF0.durationSeconds=2;
    knownF0.sourceMidiCenter=knownF0.midiNote=57;knownF0.sourcePitchMeasured=true;
    for(int i=0;i<=400;++i)
    {
        const auto t=i*.005,local=t-std::floor(t);
        knownF0.contour.push_back({t,0,0,local>=.2&&t<2,0,false});
    }
    check("explicit_known_synthetic_F0_applied_via_project",model.setClipAudioAnalysis(clipId,{knownF0},imported));
    auto data=model.snapshot();if(data.tracks[0].clips[0].notes.size()!=2)return finish();
    const auto firstNote=data.tracks[0].clips[0].notes[0].id,secondNote=data.tracks[0].clips[0].notes[1].id;
    // An actual target transposition forces ONNX inference even in the baseline;
    // this test must not quietly pass through native-source-preserved audio.
    model.setNotesMidi({firstNote,secondNote},60);
    const auto baseline=model.snapshot();const auto baselineFingerprint=model.contentFingerprint();
    check("native_neural_mode_and_ground_truth_provenance",baseline.tracks[0].nativeNsfAudio
        &&baseline.tracks[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan&&baseline.tracks[0].nsfNoiseProtection
        &&!trackUsesVoicebankSynthesis(baseline.tracks[0])&&baseline.tracks[0].clips[0].sourceFile==source
        &&baseline.tracks[0].clips[0].notes[0].sourcePitchMeasured&&baseline.tracks[0].clips[0].notes[1].sourcePitchMeasured);
    const auto absoluteTarget=[](const ProjectData& project,double absoluteSource)
    {const auto& c=project.tracks[0].clips[0];return c.startSeconds+nativeTargetTimeAt(nativeClipClock(c),absoluteSource-c.sourceOffsetSeconds);};
    check("baseline_source_clock",near(absoluteTarget(baseline,0),0)&&near(absoluteTarget(baseline,.2),.2)
        &&near(absoluteTarget(baseline,1),1)&&near(absoluteTarget(baseline,2),2));
    struct Decoded{juce::AudioBuffer<float> pcm;double sampleRate=0;bool valid=false;};
    const auto render=[&](const ProjectData& project,const juce::String& name,double endSeconds)
    {
        Decoded result;AudioEngine engine(false);engine.prepareToPlay(512,rate);
        check(name+"_no_device_before",engine.devices().getCurrentAudioDevice()==nullptr);
        engine.setHifiganModelDirectory(nsfModel);engine.setInferenceConfiguration(backend::InferenceBackend::cpu,-1);engine.syncProject(project);
        const auto deadline=juce::Time::getMillisecondCounterHiRes()+180000;
        while(engine.renderProgress()&&juce::Time::getMillisecondCounterHiRes()<deadline)juce::Thread::sleep(10);
        const auto actualBackend=engine.activeRenderBackends(),warnings=engine.activeRenderWarnings();
        const bool ready=!engine.renderProgress()&&engine.hasCurrentRenderedAudio();
        check(name+"_actual_NSF_ONNX_no_fallback",ready&&actualBackend.containsIgnoreCase("nsf-hifigan-onnx")
            &&!actualBackend.containsIgnoreCase("fallback")&&!actualBackend.containsIgnoreCase("world")
            &&actualBackend!="native-source-preserved",actualBackend);
        check(name+"_no_render_warnings",warnings.isEmpty(),warnings);
        WavExportOptions options;options.sampleRate=rate;options.channels=2;options.bitDepth=32;
        const auto file=root.getChildFile(name+".wav");
        const bool exported=ready&&engine.exportWav(file,error,trackId,0,endSeconds,options);
        check(name+"_actual_export",exported,error);
        juce::WavAudioFormat wav;auto input=exported?file.createInputStream():nullptr;
        std::unique_ptr<juce::AudioFormatReader> reader(input?wav.createReaderFor(input.release(),true):nullptr);
        const bool header=reader&&reader->sampleRate==rate&&reader->numChannels==2&&reader->bitsPerSample==32
            &&reader->usesFloatingPointData&&reader->lengthInSamples==static_cast<juce::int64>(std::llround(endSeconds*rate));
        check(name+"_exact_float_stereo_rate_length",header);
        if(header)
        {
            result.sampleRate=reader->sampleRate;result.pcm.setSize(2,static_cast<int>(reader->lengthInSamples));
            result.valid=reader->read(&result.pcm,0,result.pcm.getNumSamples(),0,true,true);
            bool finite=true;double energy=0;
            for(int ch=0;ch<2;++ch)for(int i=0;i<result.pcm.getNumSamples();++i)
            {const auto value=result.pcm.getSample(ch,i);finite&=std::isfinite(value);energy+=static_cast<double>(value)*value;}
            const auto rms=std::sqrt(energy/std::max(1,2*result.pcm.getNumSamples()));
            check(name+"_finite_audible_pcm",result.valid&&finite&&rms>1.e-4,"rms="+juce::String(rms,9));result.valid&=finite&&rms>1.e-4;
        }
        check(name+"_no_device_after",engine.devices().getCurrentAudioDevice()==nullptr);engine.releaseResources();return result;
    };
    struct Peak{double time=0,rms=0;bool valid=false;};
    const auto peak=[&](const Decoded& audio,double expected)
    {
        Peak result;if(!audio.valid)return result;
        const auto first=std::max(24,static_cast<int>(std::llround((expected-.018)*rate)));
        const auto last=std::min(audio.pcm.getNumSamples()-25,static_cast<int>(std::llround((expected+.018)*rate)));
        double maximum=-1;int best=0;
        // Fixed 49-sample (~1ms) stereo RMS envelope suppresses carrier phase.
        for(int centre=first;centre<=last;++centre)
        {
            double sum=0;for(int ch=0;ch<2;++ch)for(int j=centre-24;j<=centre+24;++j)
            {const auto value=audio.pcm.getSample(ch,j);sum+=static_cast<double>(value)*value;}
            if(sum>maximum){maximum=sum;best=centre;}
        }
        if(maximum>=0){result.time=static_cast<double>(best)/rate;result.rms=std::sqrt(maximum/98);result.valid=true;}return result;
    };
    constexpr std::array<double,4> sourcePeaks{.06,.14,1.06,1.14};
    // Validate the PHYSICAL source contract before interpreting any neural
    // result. A manually supplied UV contour does not make a periodic carrier
    // clear unvoiced evidence for production's independent source detector.
    Decoded physicalSource;
    {
        juce::WavAudioFormat wav;auto input=source.createInputStream();
        std::unique_ptr<juce::AudioFormatReader> reader(input?wav.createReaderFor(input.release(),true):nullptr);
        if(reader&&reader->sampleRate==rate&&reader->numChannels==2&&reader->lengthInSamples==rate*2)
        {
            physicalSource.sampleRate=reader->sampleRate;physicalSource.pcm.setSize(2,rate*2);
            physicalSource.valid=reader->read(&physicalSource.pcm,0,rate*2,0,true,true);
        }
    }
    check("physical_source_preflight_decoded",physicalSource.valid);if(!physicalSource.valid)return finish();
    bool physicalValid=true;
    for(std::size_t i=0;i<sourcePeaks.size();++i)
    {
        const auto measured=peak(physicalSource,sourcePeaks[i]);
        const bool valid=measured.valid&&measured.rms>=minimumBurstRms&&std::abs(measured.time-sourcePeaks[i])<=peakTolerance;
        physicalValid&=valid;
        check("physical_source_C_peak_"+juce::String(static_cast<int>(i)),valid,"actual="+juce::String(measured.time,9)
            +" expected="+juce::String(sourcePeaks[i],9)+" rms="+juce::String(measured.rms,9));
    }
    std::vector<float> sourceMono(static_cast<std::size_t>(rate*2));
    for(int i=0;i<rate*2;++i)sourceMono[static_cast<std::size_t>(i)]=.5f*(physicalSource.pcm.getSample(0,i)+physicalSource.pcm.getSample(1,i));
    const auto physicalVoicing=backend::SourceVoicing::analyse(sourceMono,rate);
    for(std::size_t i=0;i<physicalVoicing.unvoiced.size();++i)
    {
        auto* frame=new juce::DynamicObject();frame->setProperty("sourceSeconds",i*backend::SourceVoicing::period);
        frame->setProperty("dryMask",physicalVoicing.unvoiced[i]);sourceMaskEvidence.add(juce::var(frame));
    }
    for(std::size_t i=0;i<sourcePeaks.size();++i)
    {
        const auto dry=physicalVoicing.at(sourcePeaks[i]);const bool valid=dry>=.999f;physicalValid&=valid;
        check("physical_source_clear_UV_at_burst_"+juce::String(static_cast<int>(i)),valid,"actual_cpp_dry_mask="+juce::String(dry,9));
    }
    for(int part=0;part<2;++part)
    {
        float minimum=1;
        for(int i=0;i<150;++i)minimum=std::min(minimum,physicalVoicing.at(part+.025+i*.001));
        const bool valid=minimum>=.999f;physicalValid&=valid;
        check("physical_source_clear_UV_protected_window_"+juce::String(part),valid,"minimum_actual_cpp_dry_mask="+juce::String(minimum,9));
    }
    double expectedCenergy=0,actualCenergy=0;
    for(int i=0;i<rate*2;++i)
    {
        const auto local=static_cast<double>(i%rate)/rate;if(local>=.2)continue;
        double gaussian=0;for(const double centre:{.06,.14}){const auto x=local-centre;gaussian+=std::exp(-.5*x*x/(.0035*.0035));}
        expectedCenergy+=2*.55*.55*((1+.18*.18)/2)*gaussian*gaussian;
        for(int ch=0;ch<2;++ch){const auto value=physicalSource.pcm.getSample(ch,i);actualCenergy+=static_cast<double>(value)*value;}
    }
    const auto energyRatio=expectedCenergy>0?actualCenergy/expectedCenergy:0;
    const bool matchedEnergy=std::abs(energyRatio-1)<=1.e-5;physicalValid&=matchedEnergy;
    check("physical_C_carrier_matches_original_mean_square_energy",matchedEnergy,"actual_to_original_reference_energy="+juce::String(energyRatio,12));
    if(!physicalValid)return finish(); // invalid fixture is not a feature pass
    const auto before=render(baseline,"baseline-neural",2);if(!before.valid)return finish();
    std::array<Peak,4> beforePeaks;
    for(std::size_t i=0;i<sourcePeaks.size();++i)
    {
        beforePeaks[i]=peak(before,sourcePeaks[i]);
        check("baseline_C_peak_"+juce::String(static_cast<int>(i)),beforePeaks[i].valid&&beforePeaks[i].rms>=minimumBurstRms
            &&std::abs(beforePeaks[i].time-sourcePeaks[i])<=peakTolerance,
            "actual="+juce::String(beforePeaks[i].time,9)+" expected="+juce::String(sourcePeaks[i],9)+" rms="+juce::String(beforePeaks[i].rms,9));
    }
    const auto protectedWindow=[&](const Decoded& actual,double targetShift,const juce::String& stage,int pronunciation)
    {
        const int first=static_cast<int>(std::llround((pronunciation+.025)*rate));
        const int count=static_cast<int>(std::llround(.15*rate));
        const int delta=static_cast<int>(std::llround(targetShift*rate));
        bool valid=before.valid&&actual.valid&&first>=0&&first+count<=before.pcm.getNumSamples()
            &&first+delta>=0&&first+delta+count<=actual.pcm.getNumSamples();
        double a2=0,b2=0,cross=0,difference=0;
        if(valid)for(int ch=0;ch<2;++ch)for(int i=0;i<count;++i)
        {
            const auto a=before.pcm.getSample(ch,first+i),b=actual.pcm.getSample(ch,first+delta+i);
            a2+=static_cast<double>(a)*a;b2+=static_cast<double>(b)*b;cross+=static_cast<double>(a)*b;
            const auto d=static_cast<double>(a)-b;difference+=d*d;
        }
        const auto correlation=a2>0&&b2>0?cross/std::sqrt(a2*b2):0;
        const auto relative=a2>0?std::sqrt(difference/a2):1.0;
        const auto rms=count>0?std::sqrt(b2/(2*count)):0;
        check(stage+"_protected_C_window_"+juce::String(pronunciation),valid&&rms>=minimumBurstRms
            &&correlation>=minimumCorrelation&&relative<=maximumRelativeRmsDifference,
            "correlation="+juce::String(correlation,10)+" relative_rms_error="+juce::String(relative,10)+" rms="+juce::String(rms,10));
    };
    check("actual_PM_right_edge_stretch",model.resizeNativeNoteEdge(firstNote,1,false));
    const auto stretched=model.snapshot();
    check("first_V_only_expands_point8_to_1point8",near(stretched.tracks[0].clips[0].notes[0].durationSeconds,2)
        &&near(absoluteTarget(stretched,.2)-absoluteTarget(stretched,0),.2)
        &&near(absoluteTarget(stretched,1)-absoluteTarget(stretched,.2),1.8));
    check("normal_second_pronunciation_shifts_rigidly_plus1",near(stretched.tracks[0].clips[0].notes[1].startSeconds,2)
        &&near(stretched.tracks[0].clips[0].notes[1].durationSeconds,1)
        &&near(absoluteTarget(stretched,1.2)-absoluteTarget(stretched,1),.2)
        &&near(absoluteTarget(stretched,2),3));
    const auto stretchedFingerprint=model.contentFingerprint();
    check("vowel_stretch_is_one_undo_and_redo",model.undo()&&model.contentFingerprint()==baselineFingerprint
        &&model.redo()&&model.contentFingerprint()==stretchedFingerprint);
    const auto longer=render(stretched,"stretched-neural",3);
    std::array<Peak,4> longerPeaks;
    for(std::size_t i=0;i<sourcePeaks.size();++i)
    {
        const auto shift=i<2?0.0:1.0,expected=sourcePeaks[i]+shift;
        longerPeaks[i]=peak(longer,expected);
        check("stretched_C_peak_"+juce::String(static_cast<int>(i)),longerPeaks[i].valid&&longerPeaks[i].rms>=minimumBurstRms
            &&std::abs(longerPeaks[i].time-expected)<=peakTolerance
            &&std::abs((longerPeaks[i].time-beforePeaks[i].time)-shift)<=peakTolerance,
            "actual="+juce::String(longerPeaks[i].time,9)+" expected="+juce::String(expected,9)+" measured_shift="+juce::String(longerPeaks[i].time-beforePeaks[i].time,9));
    }
    for(int part=0;part<2;++part)
        check("stretched_C_burst_distance_not_expanded_"+juce::String(part),longerPeaks[part*2].valid&&longerPeaks[part*2+1].valid
            &&std::abs((longerPeaks[part*2+1].time-longerPeaks[part*2].time)-.08)<=peakTolerance);
    protectedWindow(longer,0,"stretched",0);protectedWindow(longer,1,"stretched",1);
    const auto savedFile=root.getChildFile("stretched-native.hjpx");ProjectModel reopened;
    const bool saved=model.save(savedFile,error)&&reopened.load(savedFile,error);
    check("stretched_actual_project_save_reopen",saved,error);if(!saved)return finish();
    const auto reloaded=reopened.snapshot();
    check("reloaded_project_keeps_measured_fixture_source_facts",reloaded.tracks.size()==1&&reloaded.tracks[0].clips.size()==1
        &&reloaded.tracks[0].clips[0].notes.size()==2&&reloaded.tracks[0].clips[0].sourceFile==source
        &&reloaded.tracks[0].clips[0].notes[0].sourcePitchMeasured&&reloaded.tracks[0].clips[0].notes[1].sourcePitchMeasured);
    if(reloaded.tracks.empty()||reloaded.tracks[0].clips.empty())return finish();
    const auto replay=render(reloaded,"stretched-reopened-neural",3);
    bool same=longer.valid&&replay.valid&&longer.pcm.getNumSamples()==replay.pcm.getNumSamples();
    double maxError=0,diffEnergy=0,referenceEnergy=0;
    if(same)for(int ch=0;ch<2;++ch)for(int i=0;i<longer.pcm.getNumSamples();++i)
    {
        const auto a=longer.pcm.getSample(ch,i),b=replay.pcm.getSample(ch,i);const auto d=static_cast<double>(a)-b;
        maxError=std::max(maxError,std::abs(d));diffEnergy+=d*d;referenceEnergy+=static_cast<double>(a)*a;
    }
    const auto relative=referenceEnergy>0?std::sqrt(diffEnergy/referenceEnergy):1;
    check("cold_reopen_actual_PCM_matches",same&&referenceEnergy>1.e-6&&maxError<=reloadMaximumError&&relative<=reloadRelativeRms,
        "max_abs="+juce::String(maxError,12)+" relative_rms="+juce::String(relative,12));
    // Isolated alignment branch starts from the exact actual baseline project.
    model.resetDocument(baseline);const auto alignFingerprint=model.contentFingerprint();
    check("actual_preutterance_grid_alignment",model.alignNotePreutteranceToGrid(firstNote,.25,error),error);
    const auto aligned=model.snapshot();const auto& selected=aligned.tracks[0].clips[0].notes[0];
    check("P_aligns_point10_to_point125_with_whole_pronunciation_translation",near(aligned.tracks[0].clips[0].startSeconds
        +nativeNotePreutteranceTarget(selected,aligned.tracks[0].clips[0]),.125)
        &&near(aligned.tracks[0].clips[0].startSeconds+selected.startSeconds,.025)
        &&near(selected.durationSeconds,1)&&selected.midiNote==baseline.tracks[0].clips[0].notes[0].midiNote
        &&selected.sourceMidiCenter==baseline.tracks[0].clips[0].notes[0].sourceMidiCenter);
    const auto& oldSelected=baseline.tracks[0].clips[0].notes[0];
    bool sameScore=selected.id==oldSelected.id&&selected.label==oldSelected.label
        &&selected.pronunciationKind==oldSelected.pronunciationKind&&selected.durationSeconds==oldSelected.durationSeconds
        &&selected.midiNote==oldSelected.midiNote&&selected.sourceMidiCenter==oldSelected.sourceMidiCenter
        &&selected.sourcePitchMeasured==oldSelected.sourcePitchMeasured&&selected.consonantSeconds==oldSelected.consonantSeconds
        &&selected.contour.size()==oldSelected.contour.size()&&selected.pitchControlPoints.size()==oldSelected.pitchControlPoints.size()
        &&selected.amplitudeEnvelope.size()==oldSelected.amplitudeEnvelope.size();
    if(sameScore)
    {
        for(std::size_t i=0;i<selected.contour.size();++i)
        {const auto& a=oldSelected.contour[i];const auto& b=selected.contour[i];sameScore&=a.timeSeconds==b.timeSeconds
            &&a.relativeCents==b.relativeCents&&a.withoutVibratoCents==b.withoutVibratoCents&&a.voiced==b.voiced
            &&a.hasManualTarget==b.hasManualTarget&&a.manualTargetCents==b.manualTargetCents;}
        for(std::size_t i=0;i<selected.pitchControlPoints.size();++i)
            sameScore&=selected.pitchControlPoints[i].timeSeconds==oldSelected.pitchControlPoints[i].timeSeconds
                &&selected.pitchControlPoints[i].targetMidi==oldSelected.pitchControlPoints[i].targetMidi;
        for(std::size_t i=0;i<selected.amplitudeEnvelope.size();++i)
            sameScore&=selected.amplitudeEnvelope[i].timeSeconds==oldSelected.amplitudeEnvelope[i].timeSeconds
                &&selected.amplitudeEnvelope[i].gainDb==oldSelected.amplitudeEnvelope[i].gainDb;
    }
    check("selected_whole_pronunciation_local_score_automation_unchanged",sameScore);
    check("following_V_absorbs_minus_point025_C_fixed",near(absoluteTarget(aligned,2)-absoluteTarget(aligned,1.2),.775)
        &&near(absoluteTarget(aligned,1.2)-absoluteTarget(aligned,1),.2)
        &&near(absoluteTarget(aligned,2),2));
    const auto alignedFingerprint=model.contentFingerprint();
    check("alignment_one_undo_and_redo",model.undo()&&model.contentFingerprint()==alignFingerprint&&!model.canUndo()
        &&model.redo()&&model.contentFingerprint()==alignedFingerprint);
    const auto shifted=render(aligned,"aligned-neural",2);
    std::array<Peak,4> shiftedPeaks;
    for(std::size_t i=0;i<sourcePeaks.size();++i)
    {
        const auto expected=absoluteTarget(aligned,sourcePeaks[i]);shiftedPeaks[i]=peak(shifted,expected);
        check("aligned_C_peak_"+juce::String(static_cast<int>(i)),shiftedPeaks[i].valid&&shiftedPeaks[i].rms>=minimumBurstRms
            &&std::abs(shiftedPeaks[i].time-expected)<=peakTolerance
            &&std::abs((shiftedPeaks[i].time-beforePeaks[i].time)-(expected-sourcePeaks[i]))<=peakTolerance
            &&(i>=2||std::abs((shiftedPeaks[i].time-beforePeaks[i].time)-.025)<=peakTolerance),
            "actual="+juce::String(shiftedPeaks[i].time,9)+" expected_clock="+juce::String(expected,9)+" shift="+juce::String(shiftedPeaks[i].time-beforePeaks[i].time,9));
    }
    for(int part=0;part<2;++part)
    {
        check("aligned_C_burst_distance_fixed_"+juce::String(part),shiftedPeaks[part*2].valid&&shiftedPeaks[part*2+1].valid
            &&std::abs((shiftedPeaks[part*2+1].time-shiftedPeaks[part*2].time)-.08)<=peakTolerance);
        protectedWindow(shifted,absoluteTarget(aligned,part+.06)-(part+.06),"aligned",part);
    }
    juce::MemoryBlock finalBytes;check("original_stereo_WAV_bytes_unchanged",source.loadFileAsData(finalBytes)&&finalBytes==originalBytes);
    check("no_legacy_sidecar_or_OTO_outputs",!SampleSettings::sidecarFor(source).exists()&&!root.getChildFile("oto.ini").exists());
    return finish();
}
}
