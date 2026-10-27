#pragma once
#include "../PianoRollComponent.h"
#include <iostream>
namespace hachi
{
inline bool runNativeInitialPitchSmoke(const juce::File& output)
{
    output.createDirectory();bool ok=true;int checks=0;
    const auto check=[&](const char* name,bool pass){++checks;ok &= pass;std::cout<<name<<'='<<pass<<std::endl;};
    ProjectData data;TrackData track;track.id="native";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;
    ClipData clip;clip.id="audio";clip.startSeconds=.037;clip.durationSeconds=clip.sourceDurationSeconds=2;
    clip.sourceFile=output.getChildFile("recording.wav");
    NoteData note;note.id="n";note.durationSeconds=1;note.midiNote=note.sourceMidiCenter=60.23f;note.sourcePitchMeasured=true;
    note.drift=note.modulation=1;note.utauAutoPitchTransition=false;
    note.contour={{0,-12,-12,true},{.25,32,32,true},{.7,-17,-17,true},{1,3,3,true}};
    clip.notes={note};auto other=note;other.id="other";other.startSeconds=1;clip.notes.push_back(other);
    track.clips={clip};data.tracks={track};ProjectModel model;I18n strings;PianoRollComponent roll(model,strings);
    const auto install=[&](ProjectData d){model.resetDocument(std::move(d));model.dispatchPendingMessages();roll.diagnosticRefresh();
        roll.setFocusedTrack("native");roll.setFocusedClip("audio");roll.setTool(PianoRollComponent::Tool::note);
        roll.setPixelsPerSecond(300);roll.setRowHeight(24);roll.setSize(1000,roll.getHeight());(void)roll.createComponentSnapshot(roll.getLocalBounds());};
    const auto click=[&]{const auto at=roll.diagnosticHitBounds(0).getCentre();
        const auto event=[&](int clicks){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
            juce::ModifierKeys::leftButtonModifier,0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),clicks,false);};
        roll.mouseDown(event(1));roll.mouseUp(event(1));roll.mouseDown(event(2));roll.mouseUp(event(2));roll.mouseDoubleClick(event(2));};
    install(data);const auto original=model.contentFingerprint();const auto revision=model.revisionNumber();click();
    const auto initialized=model.snapshot();const auto& n=initialized.tracks[0].clips[0].notes[0];
    check("doubleclick_initializes_source_pitch_to_semitone",n.nativeIndependentPitch&&n.midiNote==60&&n.pitchControlPoints.size()==2
        &&std::all_of(n.contour.begin(),n.contour.end(),[](const auto& p){return p.hasManualTarget&&p.manualTargetCents==0;}));
    check("initialization_never_moves_audio_or_notes",initialized.tracks[0].clips[0].startSeconds==.037&&n.startSeconds==0&&n.durationSeconds==1);
    check("other_note_keeps_original_pitch",initialized.tracks[0].clips[0].notes[1].pitchControlPoints.empty()
        &&initialized.tracks[0].clips[0].notes[1].midiNote==other.midiNote);
    check("source_measurements_preserved",n.sourceMidiCenter==note.sourceMidiCenter&&n.contour[1].relativeCents==32);
    check("one_undo_only",model.revisionNumber()==revision+1&&model.undo()&&model.contentFingerprint()==original&&!model.canUndo());
    check("redo_restores_initialization",model.redo()&&model.snapshot().tracks[0].clips[0].notes[0].nativeIndependentPitch);
    roll.diagnosticRefresh();const auto once=model.contentFingerprint();const auto onceRevision=model.revisionNumber();click();
    check("second_doubleclick_does_not_overwrite",model.contentFingerprint()==once&&model.revisionNumber()==onceRevision);
    juce::String error;ProjectModel reopened;
    check("save_reopen_still_protects_initialized_pitch",model.save(output.getChildFile("initialized.hjpx"),error)
        &&reopened.load(output.getChildFile("initialized.hjpx"),error)&&!reopened.initializeNativeNotePitch("n"));
    const std::vector<std::function<void(NoteData&)>> edits={
        [](auto& v){v.midiNote+=1;},[](auto& v){v.contour[1].hasManualTarget=true;v.contour[1].manualTargetCents=32;},
        [](auto& v){v.pitchControlPoints={{0,v.midiNote},{1,v.midiNote}};},[](auto& v){v.drift=.5f;},
        [](auto& v){v.modulation=.5f;},[](auto& v){v.vibratoEnabled=true;},[](auto& v){v.robustPitchCurve=true;},
        [](auto& v){v.nativePitchHandlesPlaced=true;},[](auto& v){v.connectedToNext=true;},
        [](auto& v){v.sourcePitchMeasured=false;},[](auto& v){v.nativeUnpitched=true;},
        [](auto& v){v.pitchSupportAuthored=true;}};
    for(const auto& edit:edits){auto changed=data;edit(changed.tracks[0].clips[0].notes[0]);install(changed);
        const auto before=model.contentFingerprint();const auto rev=model.revisionNumber();click();
        check("authored_or_unknown_pitch_rejects_doubleclick_without_moving",model.contentFingerprint()==before&&model.revisionNumber()==rev);}
    auto connected=data;NativeConnection connection;connection.leftNoteId="n";connection.rightNoteId="other";connected.nativeConnections.push_back(connection);
    install(connected);check("explicit_pitch_connection_is_protected",!model.initializeNativeNotePitch("n"));
    for(const auto type:{TrackType::utau,TrackType::diffSinger}){auto u=data;u.tracks[0].trackType=type;model.resetDocument(u);
        check("voicebank_modes_never_native_initialize",!model.initializeNativeNotePitch("n"));}
    output.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");return ok;
}
}
