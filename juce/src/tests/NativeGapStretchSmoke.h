#pragma once
#include "../NativeAudioOverlap.h"
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include <iostream>

namespace hachi
{
inline bool runNativeGapStretchSmoke(const juce::File& folder,const juce::File& savedProject={})
{
    folder.createDirectory();bool ok=true;int checks=0;
    const auto check=[&](const char* name,bool pass){ok&=pass;++checks;std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-6;};
    const auto file=folder.getChildFile("source.wav");juce::WavAudioFormat wav;
    auto stream=file.createOutputStream();if(!stream)return false;stream->setPosition(0);stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),48000,1,24,{},0));if(!writer)return false;
    juce::AudioBuffer<float> pcm(1,48000);
    for(int i=0;i<48000;++i)pcm.setSample(0,i,float(.2*std::sin(i*220*juce::MathConstants<double>::twoPi/48000)));
    writer->writeFromAudioSampleBuffer(pcm,0,48000);writer.reset();const auto hash=juce::SHA256(file).toHexString();
    const auto fixture=[&](double gap,bool linked)
    {
        ProjectData d;TrackData t;t.id="native";t.trackType=TrackType::hachi;t.pitchAlgorithm=PitchAlgorithm::world;t.normalizeVolume=false;
        std::vector<ClipData> parts;
        for(int i=0;i<2;++i)
        {
            ClipData c;c.id="source-"+juce::String(i);c.sourceFile=file;c.startSeconds=1+i*(.4+gap);
            c.durationSeconds=c.sourceDurationSeconds=.4;c.sourceOffsetSeconds=i*.5;
            NoteData n;n.id="n"+juce::String(i);n.durationSeconds=.4;n.midiNote=n.sourceMidiCenter=57;
            n.sourcePitchMeasured=true;n.utauAutoPitchTransition=false;n.contour={{0,0,0,true},{.4,0,0,true}};
            bindNativeNoteSource(n,c);c.notes={n};parts.push_back(c);
        }
        t.clips=linked?std::vector<ClipData>{assembledLinkedAudio(parts)}:parts;d.tracks={t};return d;
    };
    const auto span=[](const ProjectData& d,const juce::String& id)
    {for(const auto& t:d.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==id)
        return juce::Range<double>(c.startSeconds+n.startSeconds,c.startSeconds+n.startSeconds+n.durationSeconds);return juce::Range<double>{};};
    ProjectModel model;I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1400,650);
    const auto drag=[&](const ProjectData& d,bool left,double travel)
    {
        model.resetDocument(d);model.dispatchPendingMessages();roll.setFocusedTrack("native");roll.setFocusedClip(d.tracks[0].clips[0].id);
        roll.setTool(PianoRollComponent::Tool::note);roll.setNativeStretchMode(true);roll.setPixelsPerSecond(800);roll.diagnosticRefresh();
        (void)roll.createComponentSnapshot(roll.getLocalBounds());
        const auto b=roll.diagnosticHitBounds(left?1:0);
        const juce::Point<float> down(left?b.getX()+2:b.getRight()-2,b.getCentreY());const auto up=down.translated(float(travel*800),0);
        const auto event=[&](auto pos){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),pos,
            juce::ModifierKeys::leftButtonModifier,0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,pos!=down);};
        roll.mouseDown(event(down));roll.mouseDrag(event(up));roll.mouseUp(event(up));return model.snapshot();
    };
    for(const auto linked:{false,true})for(const auto gap:{.2,.01,.004})for(const auto left:{false,true})
    {
        const auto original=fixture(gap,linked);const auto next=drag(original,left,(left?-1:1)*(gap+.07));
        const auto a=span(next,"n0"),b=span(next,"n1");
        check("drag_fills_empty_gap_to_exact_neighbor",near(a.getEnd(),b.getStart()));
        check("drag_keeps_opposite_endpoint_and_unselected_note",left
            ?near(a.getStart(),1)&&near(a.getEnd(),1.4)&&near(b.getEnd(),1.8+gap)
            :near(a.getStart(),1)&&near(b.getStart(),1.4+gap)&&near(b.getEnd(),1.8+gap));
        const auto sources=expandedClipParts(next.tracks[0].clips[0]);
        check("source_audio_clock_reaches_the_same_note_boundary",left||near(sources.front().startSeconds+sources.front().audioStartSeconds+sources.front().audioLength(),a.getEnd()));
        check("one_undo_restores_gap",model.undo()&&!model.canUndo()&&near(span(model.snapshot(),"n1").getStart()-span(model.snapshot(),"n0").getEnd(),gap));
    }
    auto linked=fixture(.01,true);model.resetDocument(linked);check("model_closes_linked_gap",model.resizeNativeNoteEdge("n0",.01,false));
    const auto closed=model.snapshot();const auto parts=expandedClipParts(closed.tracks[0].clips[0]);
    check("closed_proxy_gap_does_not_warp_next_source",parts.size()==2&&near(parts[1].startSeconds,1.41)&&near(parts[1].audioLength(),.4)
        &&near(nativeTargetTimeAt(nativeClipClock(parts[1]),.2),.2));
    juce::String error;ProjectModel reopened;
    check("closed_gap_survives_save_reopen",model.save(folder.getChildFile("closed.hjpx"),error)&&reopened.load(folder.getChildFile("closed.hjpx"),error)
        &&near(span(reopened.snapshot(),"n0").getEnd(),span(reopened.snapshot(),"n1").getStart()));
    // Delete a middle note, then stretch its neighbour into the actual hole.
    auto deleted=fixture(.01,true);auto& c=deleted.tracks[0].clips[0];auto n=c.notes[0];n.id="split";n.durationSeconds=.2;
    c.notes[0].durationSeconds=.2;c.notes[0].startSeconds=.2;c.notes.insert(c.notes.begin(),n);
    model.resetDocument(deleted);model.removeNotes({"n0"});check("delete_gap_can_be_filled",model.resizeNativeNoteEdge("split",.2,false)
        &&near(span(model.snapshot(),"split").getEnd(),1.4));
    AudioEngine engine(false);engine.prepareToPlay(256,48000);engine.syncProject(closed);
    for(int i=0;i<1000&&engine.renderProgress();++i)juce::Thread::sleep(10);
    WavExportOptions options;options.sampleRate=48000;options.channels=1;options.bitDepth=32;
    const auto output=folder.getChildFile("filled.wav");bool audio=engine.exportWav(output,error,"native",1,1.81,options);
    juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(output));
    if(reader){juce::AudioBuffer<float> tail(1,240);reader->read(&tail,0,240,19320,true,false);audio &=tail.getRMSLevel(0,0,240)>.005f;}else audio=false;
    check("formerly_empty_gap_now_contains_stretched_audio",audio);check("source_file_unchanged",hash==juce::SHA256(file).toHexString());
    if(savedProject.existsAsFile())
    {
        const auto savedHash=juce::SHA256(savedProject).toHexString();ProjectModel real;check("actual_project_loaded",real.load(savedProject,error));
        const auto data=real.snapshot();int gaps=0;
        for(const auto& t:data.tracks)if(trackShowsAllNativeRegions(t))for(const auto& c:t.clips)if(c.nativeAudioLinked)
        {
            auto notes=c.notes;std::sort(notes.begin(),notes.end(),[](const auto& a,const auto& b){return a.startSeconds<b.startSeconds;});
            for(std::size_t i=1;i<notes.size();++i)
            {
                const auto gap=notes[i].startSeconds-notes[i-1].startSeconds-notes[i-1].durationSeconds;
                if(gap<1.e-6||gap>.1||notes[i].clipPartId==notes[i-1].clipPartId)continue;
                ++gaps;real.resetDocument(data);const auto done=real.resizeNativeNoteEdge(notes[i-1].id,gap,false);
                check("actual_project_linked_gap_fills_exactly",done&&near(span(real.snapshot(),notes[i-1].id).getEnd(),c.startSeconds+notes[i].startSeconds));
                std::cout<<"actual_gap_seconds="<<gap<<std::endl;
            }
        }
        check("actual_project_gap_reproduced",gaps>0);check("actual_project_file_untouched",savedHash==juce::SHA256(savedProject).toHexString());
    }
    std::cout<<"checks="<<checks<<" passed="<<ok<<std::endl;return ok;
}
}
