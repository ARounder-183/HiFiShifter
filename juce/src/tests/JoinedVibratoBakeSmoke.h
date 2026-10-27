#pragma once
namespace hachi
{
template<class Check>
void checkJoinedVibratoBake(const juce::File& folder,Check check)
{
    ProjectData data;TrackData track;track.id="bake";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;
    double start=0;const std::array<double,3> durations{.63,.07,.8};
    for(int i=0;i<3;++i)
    {
        ClipData c;c.id="bake-c"+juce::String(i);c.startSeconds=start;c.durationSeconds=c.sourceDurationSeconds=durations[i];
        NoteData n;n.id="bake-n"+juce::String(i);n.durationSeconds=durations[i];n.midiNote=n.sourceMidiCenter=60;n.sourcePitchMeasured=true;
        n.label="a";n.utauAutoPitchTransition=false;n.nativePitchHandlesPlaced=true;
        n.pitchControlPoints={{0,60+float(start*.3)},{durations[i],60+float((start+durations[i])*.3),PitchCurveShape::linear}};
        for(double t=0;t<n.durationSeconds;t+=.005)n.contour.push_back({t,0,0,true});n.contour.push_back({n.durationSeconds,0,0,true});
        c.notes={n};track.clips.push_back(c);start+=durations[i];
    }
    auto neighbor=track.clips.back();neighbor.id="outside-clip";neighbor.startSeconds=2;neighbor.notes[0].id="outside-note";track.clips.push_back(neighbor);
    data.tracks={track};ProjectModel model;model.resetDocument(data);
    NoteData params;params.vibratoLengthPercent=83;params.vibratoEndPercent=94;params.vibratoCycleMs=173;
    params.vibratoDepthCents=76;params.vibratoFadeInPercent=27;params.vibratoFadeOutPercent=31;params.vibratoPhasePercent=23;params.vibratoOffsetPercent=12;
    model.setNotesVibrato({"bake-n0","bake-n1","bake-n2"},params,true,true);
    const auto before=model.snapshot();const auto fingerprint=model.contentFingerprint();
    check("bake_from_short_middle_member_converts_group",model.bakeNoteVibratoIntoPitch("bake-n1"));
    const auto after=model.snapshot();bool all=true;
    for(int i=0;i<3;++i){const auto& n=after.tracks[0].clips[i].notes[0];all &= !n.vibratoEnabled&&!n.vibratoRealLine&&n.vibratoReferenceDurationSeconds==0&&n.pitchControlPoints.size()>2;}
    check("baked_group_has_only_editable_points_no_second_vibrato_layer",all);
    const auto compare=[&](const ProjectData& original,const ProjectData& baked)
    {
        const auto a=sharedPitchLines(original.tracks[0],original.nativeConnections),b=sharedPitchLines(baked.tracks[0],baked.nativeConnections);
        double lineError=0,requestError=0;I18n strings;ProjectModel displayModel;displayModel.resetDocument(baked);PianoRollComponent roll(displayModel,strings);
        roll.setFocusedTrack("bake");roll.setTool(PianoRollComponent::Tool::points);double displayError=0;
        for(int i=0;i<3;++i)
        {
            const auto& c=original.tracks[0].clips[i];const auto& n=c.notes[0];
            for(int f=0;f<=3000;++f)
            {
                const auto t=n.durationSeconds*f/3000.;const auto absolute=c.startSeconds+t;
                const auto expected=a.memberFor(n.id)->line->midiAt(absolute)+float(vibratoCentsAt(n,t)/100.);
                lineError=std::max(lineError,double(std::abs(b.memberFor(n.id)->line->midiAt(absolute)-expected)));
                const auto shown=roll.diagnosticPitchLineAt(n.id,absolute);
                displayError=std::max(displayError,shown?double(std::abs(*shown-expected)):1000.);
            }
            const auto oldRequest=original.nativeConnections.empty()?AudioEngine::diagnosticNativeRequest(c,original.tracks[0])
                :AudioEngine::mergedRequestFor({&c},original.tracks[0],{},{},{},false,&a);
            const auto newRequest=baked.nativeConnections.empty()?AudioEngine::diagnosticNativeRequest(baked.tracks[0].clips[i],baked.tracks[0])
                :AudioEngine::mergedRequestFor({&baked.tracks[0].clips[i]},baked.tracks[0],{},{},{},false,&b);
            if(oldRequest.targetMidi.size()!=newRequest.targetMidi.size())requestError=1000;
            else for(size_t f=0;f<oldRequest.targetMidi.size();++f)requestError=std::max(requestError,double(std::abs(oldRequest.targetMidi[f]-newRequest.targetMidi[f])));
        }
        std::cout<<"bake_line_error_cents="<<lineError*100<<" display="<<displayError*100<<" request="<<requestError*100<<std::endl;
        return std::array<double,3>{lineError,displayError,requestError};
    };
    const auto errors=compare(before,after);
    check("joined_bake_line_and_display_within_point01_cent",errors[0]<.0001&&errors[1]<.0001);
    check("joined_bake_native_requests_within_point01_cent",errors[2]<.0001);
    check("joined_bake_leaves_unrelated_note_unchanged",after.tracks[0].clips[3].notes[0].pitchControlPoints.size()==before.tracks[0].clips[3].notes[0].pitchControlPoints.size()
        &&!after.tracks[0].clips[3].notes[0].vibratoEnabled&&after.tracks[0].clips[3].notes[0].contour[0].hasManualTarget==before.tracks[0].clips[3].notes[0].contour[0].hasManualTarget);
    check("joined_bake_single_undo_restores_entire_group",model.undo()&&model.contentFingerprint()==fingerprint);
    check("joined_bake_redo",model.redo());juce::String error;ProjectModel reopened;
    check("joined_bake_save_reopen",model.save(folder.getChildFile("joined-baked.hjpx"),error)&&reopened.load(folder.getChildFile("joined-baked.hjpx"),error));
    const auto reopenedErrors=compare(before,reopened.snapshot());check("joined_bake_reopen_retains_pitch",reopenedErrors[0]<.0001&&reopenedErrors[2]<.0001);
    auto graph=before;NativeConnection connection;connection.id="bake-graph";connection.leftNoteId="bake-n0";connection.rightNoteId="bake-n1";
    connection.pitchCurve={{.57,60.171f},{.64,60.3f,PitchCurveShape::smooth},{.68,60.204f,PitchCurveShape::smooth}};graph.nativeConnections={connection};
    model.resetDocument(graph);check("joined_bake_handles_explicit_connection",model.bakeNoteVibratoIntoPitch("bake-n0"));
    const auto graphErrors=compare(graph,model.snapshot());check("joined_bake_connection_pitch_retained",graphErrors[0]<.0001&&graphErrors[1]<.0001&&graphErrors[2]<.0001);
    // Render both versions with a deterministic supplied-F0 backend, rather than
    // only checking the stored parameters or requiring an external voicebank.
    auto audioBefore=before;audioBefore.tracks[0].clips.resize(3);audioBefore.tracks[0].pitchAlgorithm=PitchAlgorithm::mld5;
    audioBefore.tracks[0].normalizeVolume=false;
    for(auto& c:audioBefore.tracks[0].clips)
    {
        c.sourceFile=folder.getChildFile(c.id+".wav");juce::AudioBuffer<float> pcm(1,int(std::round(c.durationSeconds*48000)));
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,float(.2*std::sin(i*2*juce::MathConstants<double>::pi*261.625565/48000)));
        auto stream=c.sourceFile.createOutputStream();if(!stream){check("bake_audio_fixture",false);return;}stream->setPosition(0);stream->truncate();
        juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),48000,1,24,{},0));
        if(!writer||!writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples())){check("bake_audio_fixture",false);return;}writer.reset();bindNativeNoteSource(c.notes[0],c);
    }
    model.resetDocument(audioBefore);model.bakeNoteVibratoIntoPitch("bake-n2");const auto audioAfter=model.snapshot();
    const auto render=[&](const ProjectData& p,const char* name)
    {
        AudioEngine engine(false);engine.prepareToPlay(256,48000);engine.syncProject(p);
        for(int i=0;i<3000&&engine.renderProgress();++i)juce::Thread::sleep(10);
        WavExportOptions options;options.sampleRate=48000;options.channels=1;options.bitDepth=32;
        const auto file=folder.getChildFile(name);check(name,engine.hasCurrentRenderedAudio()&&engine.exportWav(file,error,"bake",0,1.5,options));
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        juce::AudioBuffer<float> pcm(1,72000);pcm.clear();if(reader)reader->read(&pcm,0,72000,0,true,false);return pcm;
    };
    const auto pcmBefore=render(audioBefore,"joined-before.wav"),pcmAfter=render(audioAfter,"joined-after.wav");double energy=0,difference=0;
    for(int i=0;i<72000;++i){energy+=std::pow(pcmBefore.getSample(0,i),2);difference+=std::pow(pcmBefore.getSample(0,i)-pcmAfter.getSample(0,i),2);}
    const auto relative=std::sqrt(difference/std::max(1.e-20,energy));std::cout<<"joined_bake_pcm_relative_error="<<relative<<std::endl;
    check("joined_bake_audio_is_audible_and_matches",energy/72000>1.e-5&&relative<.002);
}
}
