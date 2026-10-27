#pragma once
#include "../DiffSingerPitchHandles.h"
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include <iostream>
namespace hachi
{
inline bool runDiffSingerPitchHandlesSmoke(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;juce::Array<juce::var> checks;
    const auto check=[&](const char* name,bool passed) {
        auto* c=new juce::DynamicObject();c->setProperty("name",name);c->setProperty("passed",passed);
        checks.add(c);ok=ok&&passed;std::cout<<name<<'='<<passed<<std::endl;
    };
    const auto visible=[](const auto& points) {
        return std::count_if(points.begin(),points.end(),[](const auto& p){return !p.diffSingerRestoreSupport;});
    };
    const auto exact=[](const auto& a,const auto& b,bool metadata=false) {
        if(a.size()!=b.size())return false;
        for(size_t i=0;i<a.size();++i)if(a[i].timeSeconds!=b[i].timeSeconds||a[i].targetMidi!=b[i].targetMidi
            ||a[i].shape!=b[i].shape||a[i].bezierX1!=b[i].bezierX1||a[i].bezierX2!=b[i].bezierX2
            ||a[i].bezierY1!=b[i].bezierY1||a[i].bezierY2!=b[i].bezierY2
            ||(metadata&&a[i].diffSingerRestoreSupport!=b[i].diffSingerRestoreSupport))return false;
        return true;
    };
    const auto error=[](const auto& a,const auto& b,double start,double end) {
        float result=0;for(auto t=start;t<=end;t+=.001)result=std::max(result,std::abs(evaluatePitchCurve(a,t)-evaluatePitchCurve(b,t)));
        return result;
    };
    ProjectModel model;ProjectData data;TrackData track;track.id="ds";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
    track.voicebankDirectory=folder.getChildFile("bank");track.voicebankDirectory.createDirectory();
    track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");
    ClipData clip;clip.id="clip";clip.startSeconds=3;clip.durationSeconds=5;
    NoteData note;note.id="n0";note.label="ni";note.startSeconds=1;note.durationSeconds=1.2;note.midiNote=60;
    clip.notes.push_back(note);note.id="n1";note.label="hao";note.startSeconds=2.6;clip.notes.push_back(note);
    track.clips.push_back(clip);data.tracks.push_back(track);model.replace(data);
    std::vector<PitchCurveEditPoint> prediction;
    for(int i=-24;i<=240;++i) {
        const double t=i*.005;
        const auto midi=59.8+.35*std::sin(t*18.85)-3*std::exp(-std::pow((t+.1)/.07,2))-4*std::exp(-std::pow((t-1.1)/.11,2));
        prediction.push_back({t,static_cast<float>(midi)});
    }
    auto second=prediction;for(auto& p:second)p.targetMidi+=3;
    const auto n=[&](int index=0){return model.snapshot().tracks[0].clips[0].notes[static_cast<size_t>(index)];};
    check("generation_applied",model.applyDiffSingerPitch(model.revisionNumber(),{{"n0",prediction},{"n1",second}}));
    const auto generated=model.snapshot();
    check("full_prediction_and_reference_preserved",exact(n().pitchControlPoints,prediction)&&exact(n().diffSingerPitchReference,prediction));
    check("generation_has_2_to_16_key_handles",visible(n().pitchControlPoints)>=2&&visible(n().pitchControlPoints)<=16);
    check("negative_onset_and_tail_handles_kept",!n().pitchControlPoints.front().diffSingerRestoreSupport&&!n().pitchControlPoints.back().diffSingerRestoreSupport);
    auto raw=generated;for(auto& c:raw.tracks[0].clips)for(auto& x:c.notes)for(auto& p:x.pitchControlPoints)p.diffSingerRestoreSupport=false;
    check("handle_compaction_does_not_change_audio_key",AudioEngine::diagnosticUtauRenderKey(raw,"clip")==AudioEngine::diagnosticUtauRenderKey(generated,"clip"));
    I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1800,1100);roll.setPixelsPerSecond(450);roll.setRowHeight(26);
    roll.setFocusedTrack("ds");roll.setFocusedClip("clip");roll.setTool(PianoRollComponent::Tool::points);roll.setShowPitchLine(true);roll.setSelectedNoteIds({"n0"});
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();
    check("paint_hit_test_use_compact_handles",roll.diagnosticOfferedPitchAnchors("n0").size()==static_cast<size_t>(visible(n().pitchControlPoints)));
    const auto shot=[&](const char* name) {
        const auto area=juce::Rectangle<int>((int)roll.diagnosticEdgeX(3.8),(int)roll.diagnosticYForMidi(65),720,340);
        if(auto out=folder.getChildFile(name).createOutputStream()) {
            out->setPosition(0);out->truncate();juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(area),*out);
        }
    };
    shot("compact-generated-pitch.png");
    model.replace(raw);refresh();const auto legacyRevision=model.revisionNumber();
    const auto denseBefore=n().pitchControlPoints;
    check("legacy_dense_project_also_displays_sparse_handles",roll.diagnosticOfferedPitchAnchors("n0").size()<=16&&visible(n().pitchControlPoints)>200);
    check("opening_legacy_project_does_not_rewrite_pitch",model.revisionNumber()==legacyRevision&&exact(n().pitchControlPoints,denseBefore,true));
    // A previously moved frame remains accessible instead of disappearing.
    auto editedLegacy=raw;editedLegacy.tracks[0].clips[0].notes[0].pitchControlPoints[110].targetMidi+=2;
    model.replace(editedLegacy);refresh();bool keptManual=false;
    for(const auto& p:roll.diagnosticOfferedPitchAnchors("n0"))if(p.timeSeconds==prediction[110].timeSeconds&&p.targetMidi==prediction[110].targetMidi+2)keptManual=true;
    check("legacy_manual_point_never_hidden",keptManual);
    model.replace(generated);refresh();auto handles=roll.diagnosticOfferedPitchAnchors("n0");
    size_t selected=1;double largest=0;
    for(size_t i=1;i+1<handles.size();++i) {
        const auto room=std::min(handles[i].timeSeconds-handles[i-1].timeSeconds,handles[i+1].timeSeconds-handles[i].timeSeconds);
        if(room>largest){largest=room;selected=i;}
    }
    check("key_points_have_editable_spacing",largest>.015&&handles.size()>=3);
    const auto active=handles[selected];const auto left=handles[selected-1].timeSeconds,right=handles[selected+1].timeSeconds;
    const auto at=juce::Point<float>(roll.diagnosticEdgeX(4+active.timeSeconds),roll.diagnosticYForMidi(active.targetMidi));
    const auto event=[&](juce::Point<float> position) {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),position,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,
            juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,false);
    };
    const auto beforeClick=model.revisionNumber();roll.mouseDown(event(at));roll.mouseUp(event(at));refresh();
    check("click_without_drag_keeps_pitch_and_revision",model.revisionNumber()==beforeClick&&exact(n().pitchControlPoints,generated.tracks[0].clips[0].notes[0].pitchControlPoints,true));
    const auto targetTime=active.timeSeconds+std::min(.025,largest*.3);
    const auto target=juce::Point<float>(roll.diagnosticEdgeX(4+targetTime),roll.diagnosticYForMidi(active.targetMidi+1.5f));
    roll.mouseDown(event(at));roll.mouseDrag(event(target));roll.mouseUp(event(target));refresh();
    const auto changed=n().pitchControlPoints;bool moved=false;
    for(const auto& p:changed)if(!p.diffSingerRestoreSupport&&std::abs(p.timeSeconds-targetTime)<.001&&std::abs(p.targetMidi-active.targetMidi-1.5f)<.001)moved=true;
    check("drag_moves_between_visible_neighbours_not_frames",moved);
    check("drag_changes_a_local_segment",error(changed,prediction,active.timeSeconds-.01,active.timeSeconds+.01)>.3f);
    check("drag_preserves_left_outside_curve",error(changed,prediction,-.12,left)<.002f);
    check("drag_preserves_right_outside_curve",error(changed,prediction,right,1.2)<.002f);
    check("drag_preserves_original_reference_and_other_note",exact(n().diffSingerPitchReference,prediction,true)&&exact(n(1).pitchControlPoints,generated.tracks[0].clips[0].notes[1].pitchControlPoints,true));
    check("drag_keeps_sparse_handles",roll.diagnosticOfferedPitchAnchors("n0").size()<=16);
    const auto curveAtTarget=roll.diagnosticPitchLineAt("n0",4+targetTime);
    check("display_and_playback_curve_agree",curveAtTarget&&std::abs(*curveAtTarget-evaluatePitchCurve(changed,targetTime))<.002f);
    model.undo();refresh();check("undo_restores_generated_frames_and_handles",exact(n().pitchControlPoints,generated.tracks[0].clips[0].notes[0].pitchControlPoints,true));
    model.redo();refresh();check("redo_restores_edit",exact(n().pitchControlPoints,changed,true));
    roll.setDiffSingerPitchRestoreMode(true);roll.mouseDown(event(target));roll.mouseUp(event(target));refresh();
    check("sparse_point_restore_matches_reference",error(n().pitchControlPoints,prediction,left+.011,right-.011)<.002f
        &&roll.diagnosticOfferedPitchAnchors("n0").size()+1==handles.size());
    roll.setDiffSingerPitchRestoreMode(false);check("restored_handle_stays_deleted",roll.diagnosticOfferedPitchAnchors("n0").size()+1==handles.size());
    juce::String failure;ProjectModel loaded;
    check("save_load_preserves_sparse_and_deleted_handles",model.save(folder.getChildFile("sparse-pitch.hjpx"),failure)
        &&loaded.load(folder.getChildFile("sparse-pitch.hjpx"),failure)
        &&exact(loaded.snapshot().tracks[0].clips[0].notes[0].pitchControlPoints,n().pitchControlPoints,true));
    auto retake=prediction;for(auto& p:retake)p.targetMidi+=.2f;
    check("regeneration_keeps_sparse_handles",model.applyDiffSingerPitch(model.revisionNumber(),{{"n0",retake}})&&visible(n().pitchControlPoints)<=16
        &&exact(n().diffSingerPitchReference,retake)&&exact(n(1).pitchControlPoints,generated.tracks[0].clips[0].notes[1].pitchControlPoints,true));
    refresh();shot("compact-regenerated-pitch.png");
    auto classic=raw;model.replace(classic);model.setTrackEditingType("ds",TrackType::utau);refresh();
    check("explicit_conversion_changes_synthesis_family",AudioEngine::diagnosticUtauRequestedFamily(model.snapshot(),"clip")
        ==backend::RequestedVoicebankFamily::utau);
    check("ordinary_UTAU_explicit_points_unchanged",roll.diagnosticOfferedPitchAnchors("n0").size()==prediction.size());
    auto shortCurve=prediction;for(auto& p:shortCurve)p.timeSeconds=(p.timeSeconds+.12)*.02;
    const auto shortResult=compactDiffSingerPitchHandles(shortCurve,shortCurve,.03);
    check("short_note_has_only_two_handles",visible(shortResult)==2&&exact(shortResult,shortCurve));
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);
    report->setProperty("prediction_frames",static_cast<int>(prediction.size()));report->setProperty("visible_handles",static_cast<int>(handles.size()));
    folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));return ok;
}
}
