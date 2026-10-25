#pragma once
#include "../PianoRollComponent.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_graphics/juce_graphics.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
// Actual component gestures and ProjectModel commits; no application window,
// playback device, synthesis backend, or parallel note/pitch implementation.
inline bool runUnifiedDrawActionSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("unified-draw-"+juce::Uuid().toString());
    if(!root.createDirectory().wasOk())return false;
    int total=0,failed=0;juce::String log;
    const auto check=[&](const juce::String& name,bool pass)
    {++total;if(!pass)++failed;log+=name+"="+(pass?"PASS":"FAIL")+"\n";};
    const auto finish=[&]
    {
        const bool written=root.getChildFile("verification.txt").replaceWithText(log);
        std::cout<<"unified_draw_passed="<<total-failed<<"|total="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;
        if(failed)std::cerr<<log<<std::endl;
        return failed==0&&written;
    };
    const auto audio=root.getChildFile("source.wav");
    {
        juce::AudioBuffer<float> pcm(1,96000);
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,.12f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*220*i/48000.0)));
        juce::WavAudioFormat format;auto stream=audio.createOutputStream();
        std::unique_ptr<juce::AudioFormatWriter> writer(stream?format.createWriterFor(stream.release(),48000,1,16,{},0):nullptr);
        const bool ok=writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
        check("own_source_fixture",ok);if(!ok)return finish();
    }
    juce::MemoryBlock originalAudio;check("source_bytes_recorded",audio.loadFileAsData(originalAudio));
    const auto bank=root.getChildFile("voicebank");bank.createDirectory();
    I18n strings;strings.setLanguage(I18n::Language::enUS);
    const auto event=[](juce::Component& component,juce::Point<float> at,juce::Point<float> down)
    {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&component,&component,
            juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);
    };
    const auto gesture=[&](PianoRollComponent& roll,juce::Point<float> from,juce::Point<float> to)
    {
        roll.mouseDown(event(roll,from,from));
        roll.mouseDrag(event(roll,from.translated((to.x-from.x)*.5f,(to.y-from.y)*.5f),from));
        roll.mouseDrag(event(roll,to,from));roll.mouseUp(event(roll,to,from));roll.diagnosticRefresh();
    };
    const auto countNotes=[](const ProjectData& data)
    {std::size_t count=0;for(const auto& track:data.tracks)for(const auto& clip:track.clips)count+=clip.notes.size();return count;};
    const auto sameCurve=[](const std::vector<PitchCurveEditPoint>& a,const std::vector<PitchCurveEditPoint>& b)
    {
        if(a.size()!=b.size())return false;
        for(std::size_t i=0;i<a.size();++i)if(a[i].timeSeconds!=b[i].timeSeconds||a[i].targetMidi!=b[i].targetMidi)return false;
        return true;
    };
    const auto sourceFacts=[](const ClipData& before,const ClipData& after)
    {
        if(before.sourceFile!=after.sourceFile||before.sourceOffsetSeconds!=after.sourceOffsetSeconds
            ||before.sourceDurationSeconds!=after.sourceDurationSeconds||before.audioStartSeconds!=after.audioStartSeconds
            ||before.audioLength()!=after.audioLength())return false;
        const auto found=std::find_if(after.notes.begin(),after.notes.end(),[&](const auto& note){return note.id==before.notes[0].id;});
        if(found==after.notes.end())return false;
        const auto& a=before.notes[0];const auto& b=*found;
        if(a.startSeconds!=b.startSeconds||a.durationSeconds!=b.durationSeconds||a.sourceMidiCenter!=b.sourceMidiCenter
            ||a.sourcePitchMeasured!=b.sourcePitchMeasured||a.contour.size()!=b.contour.size())return false;
        for(std::size_t i=0;i<a.contour.size();++i)
            if(a.contour[i].timeSeconds!=b.contour[i].timeSeconds||a.contour[i].relativeCents!=b.contour[i].relativeCents
                ||a.contour[i].withoutVibratoCents!=b.contour[i].withoutVibratoCents||a.contour[i].voiced!=b.contour[i].voiced)return false;
        return true;
    };
    for(int engine=0;engine<3;++engine)
    {
        const auto name=engine==0?juce::String("utau"):engine==1?juce::String("nsf-voicebank"):juce::String("native");
        ProjectData data;TrackData track;track.id="track";track.compose=true;track.accompaniment=false;
        track.pitchAlgorithm=engine==0?PitchAlgorithm::utau:engine==1?PitchAlgorithm::nsfHifigan:PitchAlgorithm::world;
        if(engine<2)track.voicebankDirectory=bank;
        ClipData clip;clip.id="clip";clip.sourceFile=audio;clip.startSeconds=0;
        clip.durationSeconds=clip.sourceDurationSeconds=2;
        NoteData note;note.id="original";note.label="a";note.startSeconds=.25;note.durationSeconds=.6;
        note.midiNote=note.sourceMidiCenter=60;note.sourcePitchMeasured=true;note.utauAutoPitchTransition=false;
        note.pitchControlPoints={{0,60},{.6,60}};
        for(int i=0;i<=120;++i)note.contour.push_back({i*.005,0,0,true,0,false});
        clip.notes={note};track.clips={clip};data.tracks={track};
        ProjectModel model;model.replace(data);
        PianoRollComponent roll(model,strings);roll.setReadsVoicebankInBackground(false);
        roll.setBounds(0,0,1050,1900);roll.setPixelsPerSecond(220);roll.setRowHeight(22);
        roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);roll.setShowPitchLine(true);roll.diagnosticRefresh();
        check(name+"_engine_fixture",trackUsesVoicebankSynthesis(model.snapshot().tracks[0])==(engine<2));
        check(name+"_default_action_is_pitch",roll.drawAction()==PianoRollComponent::DrawAction::pitchCurve);
        const auto choiceRevision=model.revisionNumber();const auto choiceContent=model.contentFingerprint();
        roll.setDrawAction(PianoRollComponent::DrawAction::notes);roll.setDrawAction(PianoRollComponent::DrawAction::pitchCurve);
        roll.setTool(PianoRollComponent::Tool::draw);
        check(name+"_choosing_action_is_view_only",model.revisionNumber()==choiceRevision&&model.contentFingerprint()==choiceContent);
        check(name+"_existing_note_hit",roll.diagnosticHitCount()==1);
        if(roll.diagnosticHitCount()!=1)continue;
        const auto beforeCurve=model.snapshot();const auto curveFingerprint=model.contentFingerprint();
        const auto curveRevision=model.revisionNumber();
        const auto from=roll.diagnosticHitBounds(0).getCentre();
        gesture(roll,from,from.translated(28,-14));
        const auto drawn=model.snapshot();
        const auto& drawnNote=drawn.tracks[0].clips[0].notes[0];
        const bool editedTargets=std::any_of(drawnNote.contour.begin(),drawnNote.contour.end(),[](const auto& point)
            {return point.hasManualTarget&&std::abs(point.manualTargetCents)>.01f;});
        check(name+"_default_gesture_draws_actual_pitch_targets",countNotes(drawn)==1&&editedTargets
            &&!sameCurve(drawnNote.pitchControlPoints,beforeCurve.tracks[0].clips[0].notes[0].pitchControlPoints));
        check(name+"_curve_is_one_project_edit",model.revisionNumber()==curveRevision+1);
        check(name+"_curve_preserves_source_facts",sourceFacts(beforeCurve.tracks[0].clips[0],drawn.tracks[0].clips[0]));
        const juce::Rectangle<int> pictureArea(0,static_cast<int>(roll.diagnosticYForMidi(70)-35),1000,420);
        {
            auto stream=root.getChildFile(name+"-pitch-action.png").createOutputStream();juce::PNGImageFormat png;
            check(name+"_actual_component_snapshot",stream&&png.writeImageToStream(roll.createComponentSnapshot(pictureArea),*stream));
        }
        check(name+"_curve_undo_restores_previous_project",model.undo()&&model.contentFingerprint()==curveFingerprint);roll.diagnosticRefresh();
        const auto emptyRevision=model.revisionNumber();
        const juce::Point<float> empty(roll.diagnosticEdgeX(1.2),roll.diagnosticYForMidi(65));
        gesture(roll,empty,empty.translated(24,-8));
        check(name+"_pitch_action_in_empty_space_never_creates_note",countNotes(model.snapshot())==1&&model.revisionNumber()==emptyRevision);
        const auto explicitChoiceRevision=model.revisionNumber();roll.setDrawAction(PianoRollComponent::DrawAction::notes);
        check(name+"_explicit_notes_choice_does_not_edit_project",model.revisionNumber()==explicitChoiceRevision);
        const auto beforeNotes=model.snapshot();const auto notesFingerprint=model.contentFingerprint();const auto notesRevision=model.revisionNumber();
        gesture(roll,empty,empty.translated(30,0));
        const auto added=model.snapshot();
        check(name+"_explicit_notes_creates_in_same_clip",countNotes(added)==2&&added.tracks[0].clips.size()==1&&added.tracks[0].clips[0].notes.size()==2);
        check(name+"_note_creation_uses_one_existing_model_commit",model.revisionNumber()==notesRevision+1);
        check(name+"_note_creation_keeps_original_source",sourceFacts(beforeNotes.tracks[0].clips[0],added.tracks[0].clips[0]));
        check(name+"_new_note_in_recording_bounds",added.tracks[0].clips[0].notes.size()==2
            &&added.tracks[0].clips[0].notes[1].startSeconds>=0
            &&added.tracks[0].clips[0].notes[1].startSeconds+added.tracks[0].clips[0].notes[1].durationSeconds<=2);
        check(name+"_note_creation_undo_restores_project",model.undo()&&model.contentFingerprint()==notesFingerprint);roll.diagnosticRefresh();
        const auto boundedRevision=model.revisionNumber();
        const juce::Point<float> beyond(roll.diagnosticEdgeX(2.5),roll.diagnosticYForMidi(65));
        gesture(roll,beyond,beyond.translated(20,0));
        check(name+"_notes_action_does_not_author_beyond_source",countNotes(model.snapshot())==1&&model.revisionNumber()==boundedRevision);
        // Public setFocusedTrack({}) intentionally clears an unrelated focused
        // clip as well. With no explicit target, drawing must not pick a random
        // visible track. The old test accidentally assumed that clip survived.
        roll.setFocusedTrack({});roll.diagnosticRefresh();
        const auto unfocusedRevision=model.revisionNumber();
        gesture(roll,empty,empty.translated(20,0));
        check(name+"_clearing_track_selection_requires_explicit_target",countNotes(model.snapshot())==1
            &&model.revisionNumber()==unfocusedRevision);
        roll.setSourceEditMode(true);const auto sourceEditCount=countNotes(model.snapshot());
        gesture(roll,empty,empty.translated(24,0));
        check(name+"_source_editor_never_creates_arbitrary_notes",countNotes(model.snapshot())==sourceEditCount);
        roll.setSourceEditMode(false);roll.setFocusedTrack(track.id);
        auto windowed=data;windowed.tracks[0].clips[0].audioStartSeconds=.2;
        windowed.tracks[0].clips[0].audioDurationSeconds=1.0;
        model.replace(windowed);roll.diagnosticRefresh();
        const auto holeRevision=model.revisionNumber();
        const juce::Point<float> beforeSource(roll.diagnosticEdgeX(.05),roll.diagnosticYForMidi(65));
        const juce::Point<float> afterSource(roll.diagnosticEdgeX(1.5),roll.diagnosticYForMidi(65));
        gesture(roll,beforeSource,beforeSource.translated(20,0));gesture(roll,afterSource,afterSource.translated(20,0));
        check(name+"_notes_action_rejects_uncovered_source_holes",countNotes(model.snapshot())==1&&model.revisionNumber()==holeRevision);
        auto emptyTrack=data;emptyTrack.tracks[0].clips.clear();model.replace(emptyTrack);roll.diagnosticRefresh();
        roll.setFocusedClip({});roll.setDrawAction(PianoRollComponent::DrawAction::pitchCurve);
        const auto emptyTrackRevision=model.revisionNumber();gesture(roll,empty,empty.translated(20,0));
        check(name+"_default_pitch_never_seeds_empty_composition",model.snapshot().tracks[0].clips.empty()&&model.revisionNumber()==emptyTrackRevision);
        auto backing=data;backing.tracks[0].accompaniment=true;model.replace(backing);roll.diagnosticRefresh();
        roll.setFocusedClip(clip.id);roll.setDrawAction(PianoRollComponent::DrawAction::notes);
        const auto backingRevision=model.revisionNumber();gesture(roll,empty,empty.translated(20,0));
        check(name+"_accompaniment_ineligible_for_note_action",countNotes(model.snapshot())==1&&model.revisionNumber()==backingRevision);
    }
    juce::MemoryBlock afterAudio;check("source_wav_exact_bytes_unchanged",audio.loadFileAsData(afterAudio)&&afterAudio==originalAudio);
    return finish();
}
}
