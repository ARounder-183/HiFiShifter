#pragma once
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runConnectionPitchUiSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("connection-pitch-ui-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;int total=0;juce::String log,error;
    const auto check=[&](const juce::String& name,bool value)
    {++total;ok&=value;log+=name+"="+(value?"PASS":"FAIL")+(value||error.isEmpty()?juce::String{}:" | "+error)+"\n";};
    const auto finish=[&]{const bool written=root.getChildFile("verification.txt").replaceWithText(log);std::cout<<"connection_pitch_ui_passed="<<(ok?1:0)<<"|checks="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;if(!ok)std::cerr<<log<<std::endl;return ok&&written;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-5;};
    const auto makeAudio=[](const juce::File& file,float midi)
    {
        juce::AudioBuffer<float> pcm(1,48000);const auto hz=440*std::exp2((midi-69)/12);
        for(int i=0;i<48000;++i)pcm.setSample(0,i,.12f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*hz*i/48000)));
        auto out=file.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?wav.createWriterFor(out.release(),48000,1,16,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,48000);
    };
    const auto leftFile=root.getChildFile("source-57.wav"),rightFile=root.getChildFile("source-64.wav");
    check("own_different_physical_sources",makeAudio(leftFile,57)&&makeAudio(rightFile,64));
    juce::MemoryBlock leftBytes,rightBytes;check("original_source_bytes",leftFile.loadFileAsData(leftBytes)&&rightFile.loadFileAsData(rightBytes));if(!ok)return finish();
    const auto note=[](const juce::String& id,float midi)
    {
        NoteData n;n.id=id;n.label=id;n.durationSeconds=1;n.midiNote=n.sourceMidiCenter=midi;n.sourcePitchMeasured=true;
        n.nativePitchHandlesPlaced=true;n.utauAutoPitchTransition=false;n.pitchControlPoints={{0,midi,PitchCurveShape::linear},{1,midi,PitchCurveShape::linear}};
        n.amplitudeEnvelope={{0,-3},{1,-1}};for(int i=0;i<=200;++i)n.contour.push_back({i*.005,0,0,true});return n;
    };
    ClipData a;a.id="left-clip";a.sourceFile=leftFile;a.durationSeconds=a.sourceDurationSeconds=1;a.notes={note("left",57)};
    auto b=a;b.id="right-clip";b.sourceFile=rightFile;b.startSeconds=1;b.notes={note("right",64)};
    TrackData track;track.id="track";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.clips={a,b};
    ProjectData seed;seed.tracks={track};ProjectModel model;model.resetDocument(seed);
    check("actual_model_pitch_only_force",model.setNotesPortamento({"left","right"},true,error,.4));
    auto forced=model.snapshot();auto graphs=resolvedConnectionPitchCurves(forced.tracks[0],forced.nativeConnections);
    check("one_resolved_57_to_64_graph",graphs.size()==1&&graphs[0].active&&near(graphs[0].points.front().targetMidi,57)&&near(graphs[0].points.back().targetMidi,64));
    if(graphs.size()!=1||!graphs[0].active)return finish();const auto id=graphs[0].connectionId;
    const auto originalNoteKeys=std::array<std::uint64_t,2>{AudioEngine::utauNoteRenderHash(forced.tracks[0].clips[0].notes[0]),AudioEngine::utauNoteRenderHash(forced.tracks[0].clips[1].notes[0])};
    const auto sourceFacts=[&](const ProjectData& data)
    {return data.tracks.size()==1&&data.tracks[0].clips.size()==2&&data.tracks[0].clips[0].sourceFile==leftFile&&data.tracks[0].clips[1].sourceFile==rightFile
        &&data.tracks[0].clips[0].startSeconds==0&&data.tracks[0].clips[1].startSeconds==1&&data.tracks[0].clips[0].durationSeconds==1&&data.tracks[0].clips[1].durationSeconds==1
        &&AudioEngine::utauNoteRenderHash(data.tracks[0].clips[0].notes[0])==originalNoteKeys[0]&&AudioEngine::utauNoteRenderHash(data.tracks[0].clips[1].notes[0])==originalNoteKeys[1];};
    I18n strings;strings.setLanguage(I18n::Language::enUS);PianoRollComponent roll(model,strings);roll.setReadsVoicebankInBackground(false);
    roll.setBounds(0,0,1000,1900);roll.setPixelsPerSecond(350);roll.setRowHeight(26);roll.setFocusedTrack(track.id);roll.setFocusedClip(a.id);roll.setShowPitchLine(true);
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();roll.setTool(PianoRollComponent::Tool::points);
    const auto mouse=[&](juce::Point<float> at,juce::Point<float> down,int clicks=1)
    {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),clicks,at!=down);};
    const auto drawSnapshot=[&](const juce::String& name)
    {auto out=root.getChildFile(name+".png").createOutputStream();const juce::Rectangle<int> box(0,static_cast<int>(roll.diagnosticYForMidi(71)-40),950,620);check(name,out&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(box),*out));};
    for(const auto scope:{PianoRollComponent::PitchScope::focusedClip,PianoRollComponent::PitchScope::currentTrack})
    {
        model.resetDocument(forced);refresh();roll.setPitchScope(scope);roll.setTool(PianoRollComponent::Tool::points);const juce::String tag=scope==PianoRollComponent::PitchScope::currentTrack?"track_":"clip_";
        check(tag+"graph_DTO_visible",roll.diagnosticConnectionPitchCurves().size()==1);
        const auto& initial=graphs[0];const auto time=initial.fromSeconds+(initial.toSeconds-initial.fromSeconds)*.37;
        const auto midi=evaluatePitchCurve(initial.points,time);const juce::Point<float> added(roll.diagnosticEdgeX(time),roll.diagnosticYForMidi(midi));
        const auto addRevision=model.revisionNumber();const auto addFingerprint=model.contentFingerprint();
        roll.mouseDoubleClick(mouse(added,added,2));refresh();auto current=roll.diagnosticConnectionPitchCurves();
        check(tag+"actual_doubleclick_adds_graph_point_not_note",current.size()==1&&current[0].points.size()==initial.points.size()+1&&model.revisionNumber()==addRevision+1&&sourceFacts(model.snapshot()));
        if(current.empty())continue;
        const auto pointIt=std::min_element(current[0].points.begin(),current[0].points.end(),[&](const auto& p,const auto& q){return std::abs(p.timeSeconds-time)<std::abs(q.timeSeconds-time);});
        const int index=static_cast<int>(pointIt-current[0].points.begin());
        auto at=roll.diagnosticConnectionPitchPoint(id,index);check(tag+"actual_graph_handle_hit_position",at.has_value());
        if(at)
        {
            const auto beforePoint=current[0].points[static_cast<std::size_t>(index)];const auto revision=model.revisionNumber();
            const auto to=at->translated(8,-13);roll.mouseDown(mouse(*at,*at));roll.mouseDrag(mouse(to,*at));
            const auto preview=roll.diagnosticConnectionPitchCurves();
            check(tag+"drag_preview_not_model_write",model.revisionNumber()==revision&&preview.size()==1&&preview[0].points[index].targetMidi>beforePoint.targetMidi);
            roll.mouseUp(mouse(to,*at));refresh();current=roll.diagnosticConnectionPitchCurves();
            check(tag+"actual_inner_drag_XY_one_undo",model.revisionNumber()==revision+1&&current.size()==1
                &&current[0].points[index].targetMidi>beforePoint.targetMidi&&current[0].points[index].timeSeconds>beforePoint.timeSeconds&&sourceFacts(model.snapshot()));
            check(tag+"actual_shared_shape_action",roll.diagnosticConnectionPitchShape(id,index,PitchCurveShape::easeOut));refresh();
            current=roll.diagnosticConnectionPitchCurves();check(tag+"shape_persisted_in_graph_DTO",current[0].points[index].shape==PitchCurveShape::easeOut&&sourceFacts(model.snapshot()));
        }
        at=roll.diagnosticConnectionPitchPoint(id,0);const auto beforeEnd=roll.diagnosticConnectionPitchCurves()[0].points.front();const auto endpointRevision=model.revisionNumber();
        if(at){const auto to=at->translated(8,-60);roll.mouseDown(mouse(*at,*at));roll.mouseDrag(mouse(to,*at));roll.mouseUp(mouse(to,*at));refresh();}
        current=roll.diagnosticConnectionPitchCurves();check(tag+"outer_time_edit_MIDI_live_attached",current.size()==1&&current[0].points.front().timeSeconds>beforeEnd.timeSeconds
            &&near(current[0].points.front().targetMidi,57)&&model.revisionNumber()==endpointRevision+1&&sourceFacts(model.snapshot()));
        const auto deleteRevision=model.revisionNumber();roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));refresh();
        check(tag+"endpoint_delete_guard_no_note_deletion",model.revisionNumber()==deleteRevision&&sourceFacts(model.snapshot()));
        current=roll.diagnosticConnectionPitchCurves();at=roll.diagnosticConnectionPitchPoint(id,index);
        if(at){roll.mouseDown(mouse(*at,*at));roll.mouseUp(mouse(*at,*at));}
        const auto oldCount=current[0].points.size();const auto removalRevision=model.revisionNumber();roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));refresh();
        check(tag+"actual_interior_Delete_only_graph",roll.diagnosticConnectionPitchCurves()[0].points.size()==oldCount-1&&model.revisionNumber()==removalRevision+1&&sourceFacts(model.snapshot()));
        roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));refresh();check(tag+"Delete_repeat_never_deletes_physical_note",sourceFacts(model.snapshot()));
        drawSnapshot(tag+"actual-graph-handles-two-sources");
        const auto projectFile=root.getChildFile(tag+"graph.hjpx");check(tag+"actual_project_save",model.save(projectFile,error));ProjectModel reloaded;
        check(tag+"actual_project_reopen",reloaded.load(projectFile,error));const auto reopened=reloaded.snapshot();
        auto resolved=resolvedConnectionPitchCurves(reopened.tracks[0],reopened.nativeConnections);
        check(tag+"reopen_graph_and_source_facts",resolved.size()==1&&resolved[0].active&&sourceFacts(reopened));
        if(!resolved.empty())
        {
            double maxError=0,worstTime=0;float worstActual=0,worstExpected=0;std::size_t worstClip=0;
            for(std::size_t ci=0;ci<reopened.tracks[0].clips.size();++ci)
            {
                const auto& clip=reopened.tracks[0].clips[ci];
                const auto request=AudioEngine::diagnosticNativeRequest(clip,reopened.tracks[0]);
                // The clip-only request intentionally has no graph context.
                const auto target=AudioEngine::diagnosticNativeTargetMidi(reopened,0,static_cast<int>(ci));
                for(std::size_t frame=0;frame<target.size();++frame)
                {const auto t=clip.startSeconds+frame*request.framePeriodMs/1000;if(t>resolved[0].fromSeconds+.001&&t<resolved[0].toSeconds-.001)
                    {const auto expected=evaluatePitchCurve(resolved[0].points,t);const auto difference=static_cast<double>(std::abs(target[frame]-expected));
                     if(difference>maxError){maxError=difference;worstTime=t;worstActual=target[frame];worstExpected=expected;worstClip=ci;}}}
            }
            log+=tag+"graph_target max_error="+juce::String(maxError,12)+" clip="+juce::String(static_cast<int>(worstClip))+" abs="+juce::String(worstTime,9)
                +" actual="+juce::String(worstActual,9)+" expected="+juce::String(worstExpected,9)+"\n";
            check(tag+"actual_NativeTargetMidi_uses_same_graph_DTO",maxError<=1.e-4);
        }
        // One actual mixed whole-track brush must update graph authority AND
        // note-owned pieces in one undo, never overwrite hidden base windows.
        model.resetDocument(forced);refresh();roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);roll.setTool(PianoRollComponent::Tool::line);
        const auto beforeMixed=model.contentFingerprint();const auto mixedRevision=model.revisionNumber();
        const juce::Point<float> from(roll.diagnosticEdgeX(.7),roll.diagnosticYForMidi(59)),to(roll.diagnosticEdgeX(1.3),roll.diagnosticYForMidi(63));
        roll.mouseDown(mouse(from,from));roll.mouseDrag(mouse(to,from));const auto pending=roll.diagnosticConnectionPitchCurves();
        check(tag+"mixed_preview_keeps_model_revision",model.revisionNumber()==mixedRevision&&!pending.empty());roll.mouseUp(mouse(to,from));refresh();
        check(tag+"actual_mixed_brush_is_one_model_revision",model.revisionNumber()==mixedRevision+1&&model.snapshot().nativeConnections.size()==1);
        check(tag+"mixed_one_undo_restores_graph_and_notes",model.undo()&&model.contentFingerprint()==beforeMixed&&!model.canUndo());refresh();
        (void)addFingerprint;
    }
    juce::MemoryBlock finalLeft,finalRight;check("original_WAV_files_unchanged",leftFile.loadFileAsData(finalLeft)&&rightFile.loadFileAsData(finalRight)&&finalLeft==leftBytes&&finalRight==rightBytes);
    return finish();
}
}
