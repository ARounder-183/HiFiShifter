#pragma once
namespace hachi
{
inline bool pointFirstPitchSmoke(const juce::File& output,const juce::File& input)
{
    output.createDirectory();bool ok=true;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<"="<<pass<<std::endl;};
    const auto verify=[&](const ProjectData& data,const juce::String& tag)
    {
        ProjectModel model;model.resetDocument(data);I18n strings;PianoRollComponent roll(model,strings);
        roll.setSize(1400,700);roll.setPixelsPerSecond(600);roll.setTool(PianoRollComponent::Tool::points);
        double maximum=0,displayMaximum=0,requestMaximum=0;int count=0;juce::String detail;
        for(const auto& track:data.tracks)
        {
            if(effectiveTrackType(track)!=TrackType::utau)continue;
            roll.setFocusedTrack(track.id);
            const auto plan=AudioEngine::projectTrackNotePitch(track,data.nativeConnections);
            for(const auto& clip:track.clips)
            {
                roll.setFocusedClip(clip.id);
                const auto request=AudioEngine::utauRequestNotesForClip(data,clip.id);
                size_t noteIndex=0;
                for(const auto& n:clip.notes)
                {
                    const auto index=noteIndex++;
                    const auto p=std::find_if(plan.begin(),plan.end(),[&](const auto& x){return x.noteId==n.id;});
                    if(p==plan.end()||!p->continuousLocalCents)continue;
                    for(const auto& a:roll.diagnosticOfferedPitchAnchors(n.id))
                    {
                        if(n.pitchControlPoints.empty())continue;
                        const auto vibrato=float(vibratoCentsAt(n,juce::jlimit(0.,n.durationSeconds,a.timeSeconds)));
                        const auto actual=p->midiNote*100+p->continuousLocalCents(a.timeSeconds)-vibrato;
                        const auto delta=std::abs(double(actual-a.targetMidi*100));maximum=std::max(maximum,delta);++count;
                        if(index<request.size()&&request[index].timelinePitchCents)
                            requestMaximum=std::max(requestMaximum,std::abs(double(request[index].midiNote*100+request[index].timelinePitchCents(a.timeSeconds)-vibrato-a.targetMidi*100)));
                        if(const auto shown=roll.diagnosticPitchLineAt(n.id,clip.startSeconds+n.startSeconds+a.timeSeconds))
                            displayMaximum=std::max(displayMaximum,std::abs(double(*shown*100-vibrato-a.targetMidi*100)));
                        else displayMaximum=10000;
                        if(delta>.02)detail+=n.label+","+n.id+","+juce::String(a.timeSeconds,9)+","+juce::String(delta,6)+"\n";
                    }
                }
            }
        }
        output.getChildFile(tag+"-mismatches.csv").replaceWithText("lyric,id,local_time,error_cents\n"+detail);
        std::cout<<tag<<"_points="<<count<<" render_error_cents="<<maximum<<" display_error_cents="<<displayMaximum<<std::endl;
        check(tag+"_curve_passes_through_points",count>0&&maximum<.02&&displayMaximum<.02);
        check(tag+"_notes_read_line_at_points",requestMaximum<.15);
        if(tag=="synthetic")
        {
            const auto top=int(roll.diagnosticYForMidi(62))-25;
            const auto bottom=int(roll.diagnosticYForMidi(53))+25;
            auto image=roll.createComponentSnapshot({260,top,900,bottom-top});auto stream=output.getChildFile("points.png").createOutputStream();
            if(stream){stream->setPosition(0);stream->truncate();juce::PNGImageFormat{}.writeImageToStream(image,*stream);}
        }
    };
    ProjectData data;TrackData track;track.id="track";track.trackType=TrackType::utau;track.pitchAlgorithm=PitchAlgorithm::utau;
    ClipData clip;clip.id="clip";clip.durationSeconds=3;
    NoteData a;a.id="a";a.label="sheng";a.startSeconds=.5;a.durationSeconds=.5;a.midiNote=60;a.utauAutoPitchTransition=true;
    a.pitchControlPoints={{.08,60},{.48,60}};
    NoteData b=a;b.id="b";b.label="ge";b.startSeconds=1;b.midiNote=55;b.pitchControlPoints={{.12,55},{.4,55}};
    clip.notes={a,b};track.clips={clip};data.tracks={track};
    verify(data,"synthetic");
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        data.tracks[0].utauMode=mode;
        verify(data,"mode"+juce::String(int(mode)));
        auto moved=data;moved.tracks[0].clips[0].notes[0].pitchControlPoints.back()={.49,61.5f};
        moved.tracks[0].clips[0].notes[1].pitchControlPoints.front()={-.03,58.f};
        verify(moved,"crossed"+juce::String(int(mode)));
    }
    ProjectModel real;check("load_user_project",real.load(input,error));if(error.isEmpty())verify(real.snapshot(),"real_project");
    output.getChildFile("checks.txt").replaceWithText(log);return ok;
}
}
