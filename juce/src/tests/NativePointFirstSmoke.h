#pragma once
namespace hachi
{
inline bool runNativePointFirstSmoke(const juce::File& folder, const juce::File& input)
{
    folder.createDirectory();bool ok=true;int checks=0;
    const auto check=[&](const juce::String& name,bool pass){ok &= pass;++checks;std::cout<<name<<'='<<pass<<std::endl;};
    const auto verify=[&](const ProjectData& data,const juce::String& tag){
        ProjectModel model;model.resetDocument(data);I18n strings;PianoRollComponent roll(model,strings);
        roll.setSize(1500,800);roll.setPixelsPerSecond(500);roll.setTool(PianoRollComponent::Tool::points);
        roll.setShowNativeWaveforms(false);roll.setShowOriginalPitchLine(false);
        auto expanded=data;expandProjectClipParts(expanded);
        double pointError=0,displayError=0,renderError=0;int count=0;juce::String detail;
        for(const auto& track:expanded.tracks)if(trackShowsAllNativeRegions(track))
        {
            const auto lines=sharedPitchLines(track,data.nativeConnections);roll.setFocusedTrack(track.id);
            for(const auto& clip:track.clips)
            {
                roll.setFocusedClip(clip.id);
                const auto request=AudioEngine::diagnosticNativeRequest(clip,track);
                for(const auto& note:clip.notes)
                {
                    const auto* member=lines.memberFor(note.id);
                    if(!member||!member->renderSharedCurve||member->connectionPitchOnly)continue;
                    const auto start=clip.startSeconds+note.startSeconds;
                    for(const auto& p:roll.diagnosticOfferedPitchAnchors(note.id))
                    {
                        const auto error=std::abs(member->line->midiAt(start+p.timeSeconds)-p.targetMidi);
                        pointError=std::max(pointError,double(error));++count;
                        const auto shown=roll.diagnosticPitchLineAt(note.id,start+p.timeSeconds);
                        displayError=std::max(displayError,shown?double(std::abs(*shown-p.targetMidi)):10000.);
                        if(error>.0002)detail+=note.id+","+juce::String(start+p.timeSeconds,9)+","+juce::String(error,6)+"\n";
                    }
                    for(size_t f=0;f<request.targetMidi.size();++f)
                    {
                        const auto t=std::min(request.targetDurationSeconds,f*request.framePeriodMs/1000.);
                        if(t<note.startSeconds+1.e-7||t>note.startSeconds+note.durationSeconds-1.e-7||request.targetMidi[f]<=0)continue;
                        // Explicit connection graphs have their own later overlay.
                        if(!member->connectionPitchWindows.empty())continue;
                        const auto expected=member->line->midiAt(clip.startSeconds+t)+float(vibratoCentsAt(note,t-note.startSeconds)/100.);
                        renderError=std::max(renderError,double(std::abs(request.targetMidi[f]-expected)));
                    }
                }
            }
        }
        folder.getChildFile(tag+"-errors.csv").replaceWithText("note,time,midi_error\n"+detail);
        std::cout<<tag<<" points="<<count<<" point_error="<<pointError<<" display_error="<<displayError<<" render_error="<<renderError<<std::endl;
        check(tag+"_points_define_line",count>0&&pointError<.0002);
        check(tag+"_display_reads_same_line",displayError<.0002);
        check(tag+"_synthesis_reads_same_line",renderError<.0002);
        if(tag=="cross-recording")
        {
            roll.setFocusedClip(data.tracks[0].clips[0].id);
            const auto y=int(roll.diagnosticYForMidi(70))-20;
            auto image=roll.createComponentSnapshot({450,y,1000,360});
            if(auto stream=folder.getChildFile("point-line.png").createOutputStream())juce::PNGImageFormat{}.writeImageToStream(image,*stream);
        }
    };
    ProjectData data;TrackData track;track.id="native";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;
    for(int i=0;i<3;++i)
    {
        ClipData c;c.id="clip"+juce::String(i);c.startSeconds=1+i*.5;c.durationSeconds=c.sourceDurationSeconds=.5;c.showNormalDisplay=true;
        NoteData n;n.id="note"+juce::String(i);n.durationSeconds=.5;n.midiNote=n.sourceMidiCenter=69-i*3;
        n.nativeIndependentPitch=n.nativePitchHandlesPlaced=true;n.sourcePitchMeasured=true;
        n.pitchControlPoints={{i==1?-.10:.05,n.midiNote},{.24,n.midiNote-1}};
        for(int f=0;f<=100;++f)n.contour.push_back({f*.005,0,0,true,0,true});
        c.notes={n};track.clips.push_back(c);
    }
    data.tracks={track};verify(data,"cross-recording");
    for(const auto algorithm:{PitchAlgorithm::world,PitchAlgorithm::nsfHifigan,PitchAlgorithm::llsm2})
    {auto other=data;other.tracks[0].pitchAlgorithm=algorithm;verify(other,"backend-"+juce::String(int(algorithm)));}
    auto crossed=data;crossed.tracks[0].clips[0].notes[0].pitchControlPoints.back().timeSeconds=.48;
    verify(crossed,"interleaved-points");
    // Reproduce the saved project's 0.3395 ms split with the following note's
    // control moved before that split. It must not create two simultaneous F0s.
    for(double gap:{.0003395,.00175588,.002})
    {
        auto tiny=data;tiny.tracks[0].clips[1].startSeconds+=gap;tiny.tracks[0].clips[2].startSeconds+=gap;
        tiny.tracks[0].clips[0].notes[0].pitchControlPoints.back().targetMidi=55;
        tiny.tracks[0].clips[1].notes[0].pitchControlPoints.front().timeSeconds=-.1;
        const auto lines=sharedPitchLines(tiny.tracks[0]);
        const auto* a=lines.memberFor("note0");const auto* b=lines.memberFor("note1");
        check("sub_two_ms_seam_has_one_control_point_stream",a&&b&&a->line==b->line);
        check("crossed_seam_control_has_one_pitch_on_both_sides",a&&b
            &&std::abs(a->line->midiAt(1.4+gap)-66)<.0002&&std::abs(b->line->midiAt(1.4+gap)-66)<.0002);
        verify(tiny,"near-seam-"+juce::String(gap,7));
    }
    auto legacy=data;
    for(auto& clip:legacy.tracks[0].clips)
    {
        auto& note=clip.notes[0];note.nativePitchHandlesPlaced=false;
        note.midiNote+=1;
        auto& points=note.pitchControlPoints;points.push_back({.15,note.midiNote});points.push_back({.24,note.midiNote+1});
    }
    verify(legacy,"legacy-unordered-duplicates");
    const auto originalHash=AudioEngine::nativeClipWaveformHash(data.tracks[0].clips[0],data.tracks[0]);
    auto changed=data;changed.tracks[0].clips[1].notes[0].pitchControlPoints.front().targetMidi+=2;
    check("neighbour_point_invalidates_affected_audio_cache",originalHash!=AudioEngine::nativeClipWaveformHash(changed.tracks[0].clips[0],changed.tracks[0]));
    auto separated=data;separated.tracks[0].clips[1].startSeconds+=.1;separated.tracks[0].clips[2].startSeconds+=.1;
    const auto separatedLines=sharedPitchLines(separated.tracks[0]);
    check("real_gap_keeps_independent_curves",!separatedLines.memberFor("note0")||!separatedLines.memberFor("note1")
        ||separatedLines.memberFor("note0")->line!=separatedLines.memberFor("note1")->line);
    if(input.existsAsFile())
    {
        const auto hash=juce::SHA256(input).toHexString();ProjectModel actual;juce::String error;
        check("actual_project_load",actual.load(input,error));verify(actual.snapshot(),"actual-project");
        auto expanded=actual.snapshot();expandProjectClipParts(expanded);int nearSeams=0;bool continuous=true;
        for(const auto& t:expanded.tracks)if(trackShowsAllNativeRegions(t))
        {
            struct Span{juce::String id;double start,end;};std::vector<Span> spans;
            for(const auto& c:t.clips)for(const auto& n:c.notes)if(!n.nativeUnpitched)spans.push_back({n.id,c.startSeconds+n.startSeconds,c.startSeconds+n.startSeconds+n.durationSeconds});
            std::sort(spans.begin(),spans.end(),[](const auto& a,const auto& b){return a.start<b.start;});
            const auto lines=sharedPitchLines(t);
            for(size_t i=1;i<spans.size();++i)
            {
                const auto& a=spans[i-1];const auto& b=spans[i];const auto gap=b.start-a.end;
                if(gap<=1.e-7||gap>.002)continue;
                if(std::any_of(spans.begin(),spans.end(),[&](const auto& n){return n.id!=a.id&&n.id!=b.id&&n.start<a.end-1.e-7&&n.end>a.end+1.e-7;}))continue;
                const auto* x=lines.memberFor(a.id);const auto* y=lines.memberFor(b.id);
                if(!x||!y||!x->renderSharedCurve||!y->renderSharedCurve)continue;
                ++nearSeams;continuous &= x->line==y->line;
                std::cout<<"actual_near_seam="<<a.end<<" gap_ms="<<gap*1000<<" one_line="<<(x->line==y->line)<<std::endl;
            }
        }
        check("actual_sub_two_ms_seams_do_not_split_authored_line",continuous);
        std::cout<<"actual_near_seams_checked="<<nearSeams<<std::endl;
        check("actual_project_not_modified",juce::SHA256(input).toHexString()==hash);
    }
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");return ok;
}
}
