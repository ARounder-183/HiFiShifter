#pragma once
#include "../ProjectModel.h"
#include <iostream>
#include <limits>

namespace hachi
{
inline bool runGraphPitchEditBatchSmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<=1.e-4;};
    ProjectData seed;seed.name="mixed-curve-atomicity";TrackData track;track.id="primary";track.pitchAlgorithm=PitchAlgorithm::world;
    for(int i=0;i<3;++i)
    {
        ClipData clip;clip.id="clip-"+juce::String(i);clip.startSeconds=i;clip.durationSeconds=clip.sourceDurationSeconds=1;
        clip.sourceFile=output.getChildFile("distinct-source-"+juce::String(i)+".wav");clip.sourceTimeMap={{0,0},{.4,.3},{1,1}};
        NoteData note;note.id=juce::String::charToString(static_cast<juce::juce_wchar>('A'+i));note.label="voice";
        note.midiNote=note.sourceMidiCenter=i==0?57.f:i==1?64.f:69.f;note.durationSeconds=1;note.sourcePitchMeasured=true;note.nativePitchHandlesPlaced=true;
        note.pitchControlPoints={{0,note.midiNote,PitchCurveShape::linear},{1,note.midiNote,PitchCurveShape::linear}};
        for(int j=0;j<=200;++j)note.contour.push_back({j*.005,0,0,true});
        note.gain=.73f;note.amplitudeEnvelope={{0,-2},{1,-3}};clip.notes={note};track.clips.push_back(clip);
    }
    seed.tracks={track};auto foreign=track;foreign.id="foreign";foreign.clips.resize(1);foreign.clips[0].id="foreign-clip";foreign.clips[0].notes[0].id="X";foreign.clips[0].startSeconds=10;seed.tracks.push_back(foreign);
    ProjectModel model;model.resetDocument(seed);
    check("fixture_has_real_canonical_bridge",model.setNotesPortamento({"A","B"},true,error));
    const auto base=model.snapshot();if(base.nativeConnections.empty())return false;const auto id=base.nativeConnections.front().id;
    const auto getCurve=[&](const ProjectData& d)
    {for(const auto& c:resolvedConnectionPitchCurves(d.tracks[0],d.nativeConnections))if(c.connectionId==id)return c;return ResolvedConnectionPitchCurve{};};
    const auto oldCurve=getCurve(base);
    const auto stageScene=[&](const ProjectData& original,const ProjectModel::PointCurvePlanPreview& preview)
    {
        auto result=original;for(const auto& t:preview.tracks)for(auto& originalTrack:result.tracks)if(originalTrack.id==t.id)originalTrack=t;
        result.nativeConnections=preview.connections;return result;
    };
    const auto fp=[](const ProjectData& d){ProjectModel m;m.resetDocument(d);return m.contentFingerprint();};
    const auto preservedFacts=[&](const ProjectData& a,const ProjectData& b)
    {
        for(std::size_t t=0;t<a.tracks.size();++t)for(std::size_t c=0;c<a.tracks[t].clips.size();++c)
        {
            const auto& x=a.tracks[t].clips[c];const auto& y=b.tracks[t].clips[c];
            if(x.sourceFile!=y.sourceFile||x.sourceOffsetSeconds!=y.sourceOffsetSeconds||x.sourceDurationSeconds!=y.sourceDurationSeconds
                ||x.startSeconds!=y.startSeconds||x.durationSeconds!=y.durationSeconds||x.muted!=y.muted||x.gain!=y.gain
                ||x.materialSourceFingerprint!=y.materialSourceFingerprint||x.materialSourceAnalysisPending!=y.materialSourceAnalysisPending
                ||x.sourceTimeMap.size()!=y.sourceTimeMap.size())return false;
            for(std::size_t i=0;i<x.sourceTimeMap.size();++i)if(x.sourceTimeMap[i].sourceSeconds!=y.sourceTimeMap[i].sourceSeconds||x.sourceTimeMap[i].targetSeconds!=y.sourceTimeMap[i].targetSeconds)return false;
            const auto& n=x.notes[0];const auto& m=y.notes[0];
            if(n.id!=m.id||n.midiNote!=m.midiNote||n.sourceMidiCenter!=m.sourceMidiCenter||n.sourcePitchMeasured!=m.sourcePitchMeasured||n.gain!=m.gain
                ||n.materialAudioId!=m.materialAudioId||n.materialRegionId!=m.materialRegionId||n.materialBoundAlias!=m.materialBoundAlias
                ||n.pronunciationKind!=m.pronunciationKind||n.simplifiedConsonant!=m.simplifiedConsonant||n.vowelSplitsExplicit!=m.vowelSplitsExplicit
                ||n.connectedToNext!=m.connectedToNext||n.connectedToPrevious!=m.connectedToPrevious)return false;
            for(const auto& p:n.contour)
            {
                const auto found=std::find_if(m.contour.begin(),m.contour.end(),[&](const auto& q){return q.timeSeconds==p.timeSeconds;});
                if(found==m.contour.end()||found->relativeCents!=p.relativeCents||found->withoutVibratoCents!=p.withoutVibratoCents||found->voiced!=p.voiced)return false;
            }
        }
        return true;
    };
    std::vector<PitchCurveEditPoint> composed;
    check("absolute_graph_composer_real_curve_not_fake_note",composeConnectionPitchStroke(base.tracks[0],base.nativeConnections,id,{{.985,61},{1.015,62}},composed,error));
    double outside=0;for(double t=.95;t<=1.05;t+=.0005)if(t<.985||t>1.015)
        outside=std::max(outside,std::abs(static_cast<double>(evaluatePitchCurve(oldCurve.points,t))-evaluatePitchCurve(composed,t)));
    log+="absolute_graph_outside_max_midi_error="+juce::String(outside,12)+"\n";
    check("graph_composer_preserves_exact_outside_window",outside<=1.e-5&&near(evaluatePitchCurve(composed,1),61.5)
        &&composed.front().timeSeconds==oldCurve.fromSeconds&&composed.back().timeSeconds==oldCurve.toSeconds);
    check("graph_composer_offscope_is_harmless_noop",!composeConnectionPitchStroke(base.tracks[0],base.nativeConnections,id,{{3,65},{4,66}},composed,error)&&error.isEmpty()&&composed.empty());
    ProjectModel::PointCurveEdit noteA;noteA.kind=ProjectModel::PointCurveEdit::pitch;noteA.noteId="A";noteA.pitchPoints={{0,60,PitchCurveShape::linear},{1,60,PitchCurveShape::linear}};
    auto noteB=noteA;noteB.noteId="B";noteB.pitchPoints={{0,67,PitchCurveShape::linear},{1,67,PitchCurveShape::linear}};
    ProjectModel::PointCurveEdit stroke;stroke.kind=ProjectModel::PointCurveEdit::connectionPitchStroke;stroke.connectionId=id;stroke.pitchPoints={{.98,64,PitchCurveShape::linear},{1.02,64.2f,PitchCurveShape::linear}};
    // Deliberately put the GRAPH command first: the planner must still propose
    // all note edits before deriving its new live endpoint chord.
    const std::vector<ProjectModel::PointCurveEdit> edits{stroke,noteA,noteB};model.resetDocument(base);
    const auto before=model.contentFingerprint();const auto revision=model.revisionNumber();ProjectModel::PointCurvePlanPreview preview;
    check("mixed_preview_is_pure_and_only_copies_affected_real_track",model.previewPointCurves(edits,revision,preview,error)&&preview.changed&&preview.tracks.size()==1
        &&preview.tracks[0].id=="primary"&&model.contentFingerprint()==before&&model.revisionNumber()==revision&&!model.canUndo());
    const auto proposed=stageScene(base,preview);const auto proposedCurve=getCurve(proposed);
    check("graph_stroke_uses_proposed_note_endpoint_scores",proposedCurve.active&&near(proposedCurve.points.front().targetMidi,60)&&near(proposedCurve.points.back().targetMidi,67)
        &&near(evaluatePitchCurve(proposedCurve.points,1),64.1));
    check("mixed_notes_and_bridge_commit_one_transaction",model.setPointCurves(edits,revision,&error)&&model.revisionNumber()==revision+1&&model.contentFingerprint()==fp(proposed));
    const auto committed=model.snapshot();const auto committedFingerprint=model.contentFingerprint();
    check("mixed_preview_exactly_equals_commit",committedFingerprint==fp(proposed)&&near(evaluatePitchCurve(getCurve(committed).points,1),64.1));
    check("mixed_batch_source_facts_aliases_pending_and_physics_untouched",preservedFacts(base,committed)
        &&committed.tracks[1].clips[0].notes[0].pitchControlPoints[0].targetMidi==base.tracks[1].clips[0].notes[0].pitchControlPoints[0].targetMidi);
    check("mixed_batch_undo_restores_both_note_and_graph_in_one_step",model.undo()&&!model.canUndo()&&model.contentFingerprint()==before);
    check("mixed_batch_redo_restores_exact_proposal",model.redo()&&model.contentFingerprint()==committedFingerprint);
    const auto noOpRevision=model.revisionNumber();
    check("repeated_mixed_proposal_is_noop",!model.setPointCurves(edits,noOpRevision,&error)&&error.isEmpty()&&model.revisionNumber()==noOpRevision&&model.contentFingerprint()==committedFingerprint);
    model.resetDocument(base);const auto rejectFingerprint=model.contentFingerprint();const auto rejectRevision=model.revisionNumber();
    ProjectModel::PointCurveEdit invalid;invalid.kind=ProjectModel::PointCurveEdit::connectionPitch;invalid.connectionId=id;invalid.pitchPoints=oldCurve.points;
    invalid.pitchPoints.back().timeSeconds=2.2;invalid.reanchorConnectionEndpoints=true;
    check("late_invalid_graph_geometry_rolls_back_valid_note_edit",!model.setPointCurves({noteA,invalid},rejectRevision,&error)&&error.isNotEmpty()
        &&model.contentFingerprint()==rejectFingerprint&&model.revisionNumber()==rejectRevision&&!model.canUndo());
    ProjectModel::PointCurvePlanPreview invalidPreview;
    check("invalid_preview_never_exposes_partial_note_state",!model.previewPointCurves({noteA,invalid},rejectRevision,invalidPreview,error)
        &&invalidPreview.tracks.empty()&&invalidPreview.connections.empty()&&!invalidPreview.changed&&model.contentFingerprint()==rejectFingerprint);
    auto nan=stroke;nan.pitchPoints[1].targetMidi=std::numeric_limits<float>::quiet_NaN();
    check("late_nan_graph_payload_rejects_entire_batch",!model.setPointCurves({noteA,nan},rejectRevision,&error)&&model.contentFingerprint()==rejectFingerprint&&model.revisionNumber()==rejectRevision&&!model.canUndo());
    auto missing=stroke;missing.connectionId="missing";
    check("unknown_graph_id_rejects_entire_batch",!model.setPointCurves({noteA,missing},rejectRevision,&error)&&model.contentFingerprint()==rejectFingerprint&&!model.canUndo());
    check("stale_mixed_revision_rejected_before_any_note_write",!model.setPointCurves(edits,rejectRevision+1,&error)&&model.contentFingerprint()==rejectFingerprint&&!model.canUndo());
    // Whole-curve handle edits remain strict by default. Automatic endpoint
    // following is an explicit command option, not silently ignored bad MIDI.
    auto drag=invalid;drag.pitchPoints=oldCurve.points;drag.pitchPoints.insert(drag.pitchPoints.begin()+1,{1,64,PitchCurveShape::smooth});drag.reanchorConnectionEndpoints=false;
    check("strict_mixed_handle_edit_rejects_stale_outer_pitch",!model.setPointCurves({noteA,noteB,drag},rejectRevision,&error)&&model.contentFingerprint()==rejectFingerprint&&!model.canUndo());
    drag.reanchorConnectionEndpoints=true;
    check("explicit_mixed_drag_reanchors_outer_pitch_after_note_proposals",model.setPointCurves({drag,noteA,noteB},rejectRevision,&error));
    const auto dragged=model.snapshot();const auto draggedCurve=getCurve(dragged);
    check("mixed_drag_keeps_interior_absolute_pitch_and_new_live_ends",near(draggedCurve.points.front().targetMidi,60)&&near(draggedCurve.points.back().targetMidi,67)&&near(evaluatePitchCurve(draggedCurve.points,1),64));
    check("mixed_drag_is_single_undo",model.undo()&&!model.canUndo()&&model.contentFingerprint()==rejectFingerprint);
    auto outsideStroke=stroke;outsideStroke.pitchPoints={{3,65},{4,66}};const auto offscopeRevision=model.revisionNumber();
    check("offscope_graph_stroke_creates_no_undo_or_revision",!model.setPointCurves({outsideStroke},offscopeRevision,&error)&&error.isEmpty()&&model.contentFingerprint()==rejectFingerprint&&model.revisionNumber()==offscopeRevision&&!model.canUndo());
    check("preview_source_facts_stay_identical",preservedFacts(base,proposed));
    model.resetDocument(base);const std::vector<PitchCurveEditPoint> extended={{-.1,57,PitchCurveShape::linear},{0,57,PitchCurveShape::linear},{1,60,PitchCurveShape::linear},{1.2,60,PitchCurveShape::linear}};
    check("imported_or_local_geometry_alone_never_authors_support",model.setNotePitchCurve("A",extended,true)&&!model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    const auto unauthored=model.snapshot();model.resetDocument(unauthored);const auto supportBefore=model.contentFingerprint();const auto supportRevision=model.revisionNumber();
    ProjectModel::PointCurveEdit author;author.noteId="A";author.pitchPoints=extended;author.authorPitchSupport=true;
    ProjectModel::PointCurvePlanPreview supportPreview;
    check("support_intent_preview_pure_and_uses_same_setter",model.previewPointCurves({author},supportRevision,supportPreview,error)&&supportPreview.changed
        &&supportPreview.tracks[0].clips[0].notes[0].pitchSupportAuthored&&!model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored&&model.contentFingerprint()==supportBefore);
    check("explicit_support_intent_batch_is_one_undo",model.setPointCurves({author},supportRevision,&error)&&model.revisionNumber()==supportRevision+1&&model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    check("support_intent_undo_restores_legacy_default",model.undo()&&!model.canUndo()&&model.contentFingerprint()==supportBefore&&!model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    check("support_intent_redo",model.redo()&&model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    const auto intentFile=output.getChildFile("authored-pitch-support.hachi");ProjectModel intentReload;
    check("support_intent_optional_serde",model.save(intentFile,error)&&intentReload.load(intentFile,error)&&intentReload.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored
        &&!intentReload.snapshot().tracks[0].clips[1].notes[0].pitchSupportAuthored);
    auto local=extended;local[1].targetMidi+=.1f;
    check("later_local_edit_preserves_prior_explicit_support",model.setNotePitchCurve("A",local,true)&&model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    const auto copiedIntent=model.snapshot();check("ordinary_note_copy_preserves_musical_support_intent",NoteData(copiedIntent.tracks[0].clips[0].notes[0]).pitchSupportAuthored);
    check("explicit_restore_source_clears_authored_support",model.restoreNativeSourcePitch({"A"})&&!model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
    const auto written=output.getChildFile("graph-pitch-edit-batch-smoke.txt").replaceWithText(log);std::cout<<"report_written="<<written<<std::endl;return ok&&written;
}
}
