#pragma once
#include "../ProjectModel.h"
#include "../NativeNoteTiming.h"
#include <iostream>

namespace hachi
{
// Pure score/resolver contract. Actual render + UI wiring is a separate phase;
// nonzero MIDI here never claims to create voicing in a rest or breath source.
inline bool runForcedPortamentoSmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<.0002;};
    const auto fixture=[&]
    {
        ProjectData d;TrackData t;t.id="voices";t.pitchAlgorithm=PitchAlgorithm::world;
        for(int i=0;i<3;++i)
        {
            const auto midi=i==0?57.f:i==1?64.f:69.f;ClipData c;c.id="clip-"+juce::String(i);c.startSeconds=i;c.durationSeconds=c.sourceDurationSeconds=1;
            c.sourceFile=output.getChildFile("different-source-"+juce::String(i)+".wav");c.sourceTimeMap={{0,0},{.3,.2},{1,1}};
            NoteData n;n.id=juce::String::charToString(static_cast<juce::juce_wchar>('A'+i));n.label="voice";n.durationSeconds=1;
            n.midiNote=n.sourceMidiCenter=midi;n.sourcePitchMeasured=true;n.contour={{0,0,0,true},{.5,0,0,true},{1,0,0,true}};
            n.pitchControlPoints={{0,midi,PitchCurveShape::linear},{1,midi,PitchCurveShape::linear}};n.gain=.77f;n.amplitudeEnvelope={{0,-4},{.4,-2},{1,-5}};
            c.notes={n};t.clips.push_back(c);
        }
        d.tracks={t};auto unrelated=t;unrelated.id="unrelated";unrelated.clips.resize(1);unrelated.clips[0].id="other";
        unrelated.clips[0].startSeconds=10;unrelated.clips[0].notes[0].id="X";d.tracks.push_back(unrelated);return d;
    };
    const auto edge=[](const ProjectData& d,const char* a,const char* b)->NativeConnection
    {for(const auto& c:d.nativeConnections)if(c.leftNoteId==a&&c.rightNoteId==b)return c;return {};};
    const auto resolved=[](const ProjectData& d,const juce::String& id)->ResolvedConnectionPitchCurve
    {for(const auto& c:resolvedConnectionPitchCurves(d.tracks[0],d.nativeConnections))if(c.connectionId==id)return c;return {};};
    const auto baseAt=[](const TrackData& t,const char* id,double time)
    {
        const auto lines=sharedPitchLines(t);if(const auto* n=lines.memberFor(id))return n->line->midiAt(time);
        for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==id)return evaluatePitchCurve(ownPitchPoints(n),time-c.startSeconds-n.startSeconds);return 60.f;
    };
    const auto pitchAt=[&](const ProjectData& d,const char* id,double time)
    {const auto lines=sharedPitchLines(d.tracks[0],d.nativeConnections);const auto* n=lines.memberFor(id);return n?n->line->midiAt(time):baseAt(d.tracks[0],id,time);};
    const auto sourceAndMusicSame=[](const ProjectData& a,const ProjectData& b)
    {
        if(a.tracks.size()!=b.tracks.size())return false;
        for(std::size_t t=0;t<a.tracks.size();++t)
        {
            if(a.tracks[t].clips.size()!=b.tracks[t].clips.size())return false;
            for(std::size_t i=0;i<a.tracks[t].clips.size();++i)
            {
                const auto& x=a.tracks[t].clips[i];const auto& y=b.tracks[t].clips[i];
                if(x.sourceFile!=y.sourceFile||x.startSeconds!=y.startSeconds||x.durationSeconds!=y.durationSeconds||x.sourceOffsetSeconds!=y.sourceOffsetSeconds
                    ||x.sourceDurationSeconds!=y.sourceDurationSeconds||x.sourceTimeMap.size()!=y.sourceTimeMap.size()||x.fadeInSeconds!=y.fadeInSeconds||x.fadeOutSeconds!=y.fadeOutSeconds
                    ||x.glideConnectedFromPrevious!=y.glideConnectedFromPrevious||x.glideConnectedToNext!=y.glideConnectedToNext||x.notes.size()!=y.notes.size())return false;
                for(std::size_t p=0;p<x.sourceTimeMap.size();++p)if(x.sourceTimeMap[p].targetSeconds!=y.sourceTimeMap[p].targetSeconds||x.sourceTimeMap[p].sourceSeconds!=y.sourceTimeMap[p].sourceSeconds)return false;
                for(std::size_t j=0;j<x.notes.size();++j)
                {
                    const auto& n=x.notes[j];const auto& m=y.notes[j];
                    if(n.id!=m.id||n.midiNote!=m.midiNote||n.gain!=m.gain||n.utauSplice!=m.utauSplice||n.connectedToPrevious!=m.connectedToPrevious||n.connectedToNext!=m.connectedToNext
                        ||n.pitchControlPoints.size()!=m.pitchControlPoints.size()||n.amplitudeEnvelope.size()!=m.amplitudeEnvelope.size()||n.contour.size()!=m.contour.size())return false;
                    for(std::size_t p=0;p<n.pitchControlPoints.size();++p)if(n.pitchControlPoints[p].timeSeconds!=m.pitchControlPoints[p].timeSeconds||n.pitchControlPoints[p].targetMidi!=m.pitchControlPoints[p].targetMidi)return false;
                    for(std::size_t p=0;p<n.contour.size();++p)if(n.contour[p].relativeCents!=m.contour[p].relativeCents||n.contour[p].voiced!=m.contour[p].voiced)return false;
                }
            }
        }
        return true;
    };
    const auto initial=fixture();ProjectModel model;model.resetDocument(initial);const auto initialFingerprint=model.contentFingerprint();const auto revision=model.revisionNumber();
    check("different_recordings_force_connect",model.setNotesPortamento({"A","B"},true,error));auto ab=model.snapshot();const auto connection=edge(ab,"A","B");
    const auto curve=resolved(ab,connection.id);
    check("one_graph_authority_real_smooth_curve",connection.id.isNotEmpty()&&connection.type=="pitch-portamento"&&connection.pitchCurveEndpointRelative&&connection.pitchCurve.size()==2
        &&curve.active&&near(curve.fromSeconds,.95)&&near(curve.toSeconds,1.05)&&near(curve.points.front().targetMidi,57)&&near(curve.points.back().targetMidi,64));
    check("force_is_one_undo_not_source_or_amplitude_join",model.revisionNumber()==revision+1&&sourceAndMusicSame(initial,ab));
    bool monotonic=true;float previous=56;
    for(int i=0;i<=100;++i){const auto t=.95+i*.001;const auto value=pitchAt(ab,t<1?"A":"B",t);monotonic&=value>=previous&&value>=57-.0001&&value<=64+.0001;previous=value;}
    check("57_to_64_bridge_is_monotonic_and_not_linear_chord",monotonic&&near(pitchAt(ab,"A",.975),58.09375)&&near(pitchAt(ab,"B",1),60.5)&&near(pitchAt(ab,"B",1.025),62.90625));
    bool outside=true;for(int i=-100;i<=2100;++i){const auto t=i*.001;if(t<.95||t>=1.05)outside&=pitchAt(ab,"A",t)==baseAt(initial.tracks[0],"A",t)&&pitchAt(ab,"B",t)==baseAt(initial.tracks[0],"B",t);}
    check("outside_window_returns_exact_old_per_note_base",outside);
    const auto lines=sharedPitchLines(ab.tracks[0],ab.nativeConnections);
    check("raw_native_only_overrides_request_inside_graph_window",lines.memberFor("A")&&lines.memberFor("A")->connectionPitchOnly&&lines.memberFor("B")->connectionPitchOnly
        &&lines.memberFor("A")->connectionPitchWindows.size()==1&&!lines.memberFor("A")->owns(1)&&lines.memberFor("A")->owns(0));
    const auto digestA=connectionPitchDigest(ab.tracks[0],ab.nativeConnections,ab.tracks[0].clips[0]);
    check("digest_scoped_to_actual_incident_clips",digestA.isNotEmpty()&&connectionPitchDigest(ab.tracks[0],ab.nativeConnections,ab.tracks[0].clips[2]).isEmpty()
        &&connectionPitchDigest(ab.tracks[1],ab.nativeConnections,ab.tracks[1].clips[0]).isEmpty());
    check("force_undo_restores_every_source_and_score_field",model.undo()&&model.contentFingerprint()==initialFingerprint&&!model.canUndo());check("force_redo",model.redo());
    const auto beforeNoop=model.contentFingerprint();const auto noopRevision=model.revisionNumber();
    check("repeat_force_and_identical_curve_are_noop",!model.setNotesPortamento({"A","B"},true,error)&&!model.setConnectionPitchCurve(connection.id,curve.points,error)
        &&model.contentFingerprint()==beforeNoop&&model.revisionNumber()==noopRevision);
    check("adding_BC_preserves_AB",model.setNotesPortamento({"B","C"},true,error)&&edge(model.snapshot(),"A","B").id==connection.id&&edge(model.snapshot(),"B","C").id.isNotEmpty());
    auto chain=model.snapshot();const auto chainLines=sharedPitchLines(chain.tracks[0],chain.nativeConnections);
    check("middle_has_two_incident_scopes_no_bankwide_borrow",chainLines.memberFor("B")->connectionPitchOnly&&chainLines.memberFor("B")->connectionPitchWindows.size()==2
        &&near(pitchAt(chain,"A",2),57)&&near(pitchAt(chain,"B",1.5),64)&&sourceAndMusicSame(initial,chain));
    check("unrelated_clip_digest_unchanged_adding_BC",connectionPitchDigest(chain.tracks[0],chain.nativeConnections,chain.tracks[0].clips[0])==digestA);
    auto impossible=curve.points;impossible.back().timeSeconds=1.99;impossible.back().targetMidi=64;const auto conflictFingerprint=model.contentFingerprint();
    check("overlapping_incident_bridges_reject_atomically",!model.setConnectionPitchCurve(connection.id,impossible,error)&&error.contains("overlap")&&model.contentFingerprint()==conflictFingerprint);
    model.resetDocument(ab);auto edited=curve.points;edited.insert(edited.begin()+1,{.995,61.2f,PitchCurveShape::customBezier,.2f,.1f,.8f,.9f});
    check("interior_pitch_and_curvature_edit",model.setConnectionPitchCurve(connection.id,edited,error)&&near(pitchAt(model.snapshot(),"A",.995),61.2));
    const auto editedFingerprint=model.contentFingerprint();auto first=resolved(model.snapshot(),connection.id).points.front();first.timeSeconds=.9;first.targetMidi=57;
    check("outer_endpoint_time_edit",model.setConnectionPitchPoint(connection.id,0,first,error)&&near(resolved(model.snapshot(),connection.id).fromSeconds,.9));
    const auto configured=model.snapshot();const auto configuredCurve=resolved(configured,connection.id);
    model.moveClips({"clip-0","clip-1","clip-2"},.4,0);
    const auto moved=model.snapshot();const auto movedCurve=resolved(moved,connection.id);
    check("curve_retargets_live_note_moves",movedCurve.active&&near(movedCurve.fromSeconds,configuredCurve.fromSeconds+.4)
        &&near(movedCurve.points[1].timeSeconds,configuredCurve.points[1].timeSeconds+.4)&&near(movedCurve.points[1].targetMidi,configuredCurve.points[1].targetMidi));
    model.transposeNote("B",3);const auto transposed=model.snapshot();const auto transposedCurve=resolved(transposed,connection.id);
    const auto u=transposed.nativeConnections.front().pitchCurve[1].timeSeconds;
    check("curve_retargets_transposition_with_relative_residual",transposedCurve.active&&near(transposedCurve.points.back().targetMidi,67)
        &&near(transposedCurve.points[1].targetMidi,movedCurve.points[1].targetMidi+3*u)
        &&connectionPitchDigest(transposed.tracks[0],transposed.nativeConnections,transposed.tracks[0].clips[0])!=connectionPitchDigest(moved.tracks[0],moved.nativeConnections,moved.tracks[0].clips[0]));
    juce::String saveError;ProjectModel loaded;const auto file=output.getChildFile("forced-portamento.hachi");
    check("relative_bridge_save_load",model.save(file,saveError)&&loaded.load(file,saveError)&&resolved(loaded.snapshot(),connection.id).active
        &&near(resolved(loaded.snapshot(),connection.id).points[1].targetMidi,transposedCurve.points[1].targetMidi));
    model.resetDocument(configured);model.moveClip("clip-2",2.5);model.moveClip("clip-1",1.2);const auto gapped=model.snapshot();
    check("move_creating_gap_disables_not_deletes_curve",!resolved(gapped,connection.id).active&&edge(gapped,"A","B").pitchCurve.size()==3);
    model.moveClip("clip-1",1);check("restoring_adjacency_restores_edited_curve",resolved(model.snapshot(),connection.id).active);
    model.resetDocument(configured);const auto graphEditRevision=model.revisionNumber();ProjectModel::PointCurveEdit batch;batch.kind=ProjectModel::PointCurveEdit::connectionPitch;batch.connectionId=connection.id;
    batch.pitchPoints=configuredCurve.points;batch.pitchPoints[1].targetMidi+=.25f;
    check("existing_point_batch_edits_graph_only",model.setPointCurves({batch},graphEditRevision)&&near(resolved(model.snapshot(),connection.id).points[1].targetMidi,configuredCurve.points[1].targetMidi+.25)
        &&sourceAndMusicSame(initial,model.snapshot()));
    check("graph_point_batch_single_undo",model.undo()&&!model.canUndo()&&model.contentFingerprint()==[&]{ProjectModel m;m.resetDocument(configured);return m.contentFingerprint();}());
    model.resetDocument(configured);check("disconnect_removes_effect_not_user_pitch",model.setNotesPortamento({"A","B"},false,error)&&edge(model.snapshot(),"A","B").id.isEmpty()
        &&pitchAt(model.snapshot(),"A",.98)==baseAt(initial.tracks[0],"A",.98)&&sourceAndMusicSame(initial,model.snapshot()));
    auto physical=initial;physical.nativeConnections={{"old-source-link","A","B","pitch-and-amplitude",1,{},{{0,-2}}}};
    physical.tracks[0].clips[0].notes[0].connectedToNext=true;physical.tracks[0].clips[1].notes[0].connectedToPrevious=true;
    physical.tracks[0].clips[0].glideConnectedToNext=true;physical.tracks[0].clips[1].glideConnectedFromPrevious=true;
    model.resetDocument(physical);check("upgrade_existing_edge_keeps_original_source_intent",model.setNotesPortamento({"A","B"},true,error)&&edge(model.snapshot(),"A","B").id=="old-source-link"&&sourceAndMusicSame(physical,model.snapshot()));
    check("pitch_disconnect_retains_independent_physical_edge",model.setNotesPortamento({"A","B"},false,error)&&edge(model.snapshot(),"A","B").id=="old-source-link"
        &&edge(model.snapshot(),"A","B").pitchCurve.empty()&&edge(model.snapshot(),"A","B").amplitudeCurve.size()==1&&sourceAndMusicSame(physical,model.snapshot()));
    const auto rejects=[&](ProjectData d,std::vector<juce::String> ids,const char* name)
    {model.resetDocument(std::move(d));const auto f=model.contentFingerprint();const auto r=model.revisionNumber();check(name,!model.setNotesPortamento(ids,true,error)&&error.isNotEmpty()&&model.contentFingerprint()==f&&model.revisionNumber()==r&&!model.canUndo());};
    rejects(initial,{"A","C"},"intervening_note_rejected");rejects(initial,{"A","X"},"cross_track_rejected");rejects(initial,{"A","missing"},"missing_id_rejected");
    auto gap=initial;gap.tracks[0].clips[1].startSeconds=1.1;rejects(gap,{"A","B"},"gap_never_creates_audio");
    auto overlap=initial;overlap.tracks[0].clips[1].startSeconds=.9;rejects(overlap,{"A","B"},"overlapping_endpoints_rejected");
    auto breath=initial;breath.tracks[0].clips[1].notes[0].pronunciationKind=PronunciationKind::breath;rejects(breath,{"A","B"},"breath_never_forced_voiced");
    auto unpitched=initial;unpitched.tracks[0].clips[1].notes[0].nativeUnpitched=true;rejects(unpitched,{"A","B"},"unpitched_never_forced_voiced");
    auto rest=initial;rest.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;rest.tracks[0].clips[1].notes[0].label="RR";rejects(rest,{"A","B"},"utau_rest_never_forced_voiced");
    rest.tracks[0].clips[1].notes[0].label="R";model.resetDocument(rest);
    check("single_R_remains_a_normal_lyric_per_canonical_classifier",model.setNotesPortamento({"A","B"},true,error));
    auto utau=initial;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;model.resetDocument(utau);
    check("same_model_resolver_supports_utau_score",model.setNotesPortamento({"A","B"},true,error)&&near(pitchAt(model.snapshot(),"B",1),60.5)&&sourceAndMusicSame(utau,model.snapshot()));
    auto pAnchored=initial;pAnchored.tracks[0].clips[1].notes[0].utauPreutteranceOverrideEnabled=true;pAnchored.tracks[0].clips[1].notes[0].utauPreutteranceSeconds=.1;
    model.resetDocument(pAnchored);check("default_right_anchor_uses_shared_physical_P",model.setNotesPortamento({"A","B"},true,error)
        &&near(resolved(model.snapshot(),edge(model.snapshot(),"A","B").id).anchorSeconds,1+nativeNotePreutteranceTarget(pAnchored.tracks[0].clips[1].notes[0],pAnchored.tracks[0].clips[1])));
    // Pure exact stroke composition, including a cut INSIDE a shaped segment.
    NoteData sparse;sparse.id="stroke";sparse.durationSeconds=2;sparse.midiNote=sparse.sourceMidiCenter=60;sparse.sourcePitchMeasured=true;
    sparse.pitchControlPoints={{0,60},{.4,63,PitchCurveShape::customBezier,.2f,.1f,.8f,.9f},{1.2,61,PitchCurveShape::smooth},{2,65,PitchCurveShape::natural}};
    std::vector<PitchCurveEditPoint> out;bool store=false;const std::vector<PitchCurveEditPoint> stroke={{.6,55,PitchCurveShape::linear},{.9,57,PitchCurveShape::linear},{1.1,56,PitchCurveShape::linear}};
    check("compose_sparse_stroke",composeNativePitchStroke(sparse,stroke,out,store,error)&&store);
    double outsideError=0;for(int i=0;i<=2000;++i){const auto t=i*.001;if(t<.6||t>1.1)outsideError=std::max(outsideError,std::abs(static_cast<double>(evaluatePitchCurve(sparse.pitchControlPoints,t))-evaluatePitchCurve(out,t)));}
    log+="compose_outside_max_midi_error="+juce::String(outsideError,12)+"\n";
    check("compose_keeps_exact_cubic_outside_and_guarded_boundary",outsideError<.0002&&near(evaluatePitchCurve(out,.9),57)
        &&near(evaluatePitchCurve(out,.6),evaluatePitchCurve(sparse.pitchControlPoints,.6))&&near(evaluatePitchCurve(out,1.1),evaluatePitchCurve(sparse.pitchControlPoints,1.1)));
    NoteData precision;precision.durationSeconds=1.4;precision.midiNote=60;
    precision.pitchControlPoints={{-.15,58,PitchCurveShape::linear},{0,59.5f,PitchCurveShape::easeIn},{.35,61.5f,PitchCurveShape::customBezier},
        {.8,62,PitchCurveShape::easeOut},{1.1,63,PitchCurveShape::smooth},{1.55,61.5f,PitchCurveShape::natural}};
    precision.pitchControlPoints[2].bezierX1=.22f;precision.pitchControlPoints[2].bezierY1=.12f;precision.pitchControlPoints[2].bezierX2=.78f;precision.pitchControlPoints[2].bezierY2=.88f;
    std::vector<PitchCurveEditPoint> exact;
    const bool precise=composeNativePitchStroke(precision,{{.25,65,PitchCurveShape::linear},{1.4,66,PitchCurveShape::linear}},exact,store,error);
    double exactError=0;
    if(precise)for(double t=-.15;t<=1.55;t+=.0025)if(t<.25-1.e-8||t>1.4+1.e-8)
        exactError=std::max(exactError,std::abs(static_cast<double>(evaluatePitchCurve(precision.pitchControlPoints,t))-evaluatePitchCurve(exact,t)));
    log+="three_ulp_regression_max_midi_error="+juce::String(exactError,12)+"\n";
    check("cropped_cubic_double_evaluation_meets_fixed_1e_minus5",precise&&exactError<=1.e-5
        &&std::any_of(exact.begin(),exact.end(),[](const auto& p){return p.highPrecisionBezier;})
        &&std::none_of(precision.pitchControlPoints.begin(),precision.pitchControlPoints.end(),[](const auto& p){return p.highPrecisionBezier;}));
    ProjectData precisionData;TrackData precisionTrack;precisionTrack.id="precision";precisionTrack.pitchAlgorithm=PitchAlgorithm::world;
    ClipData precisionClip;precisionClip.id="precision-clip";precisionClip.durationSeconds=precisionClip.sourceDurationSeconds=1.4;precision.id="precision-note";precision.pitchControlPoints=exact;
    precisionClip.notes={precision};precisionTrack.clips={precisionClip};precisionData.tracks={precisionTrack};ProjectModel precisionModel,precisionReload;precisionModel.resetDocument(precisionData);
    const auto precisionFile=output.getChildFile("exact-cubic-precision.hachi");
    const auto precisionSaved=precisionModel.save(precisionFile,error)&&precisionReload.load(precisionFile,error);
    const auto precisionLoaded=precisionReload.snapshot();
    check("new_cropped_precision_flag_roundtrips_without_affecting_defaults",precisionSaved
        &&std::any_of(precisionLoaded.tracks[0].clips[0].notes[0].pitchControlPoints.begin(),precisionLoaded.tracks[0].clips[0].notes[0].pitchControlPoints.end(),[](const auto& p){return p.highPrecisionBezier;}));
    NoteData seamA;seamA.durationSeconds=1;seamA.pitchControlPoints={{0,57},{1,57}};auto seamB=seamA;seamB.pitchControlPoints={{0,64},{1,64}};
    std::vector<PitchCurveEditPoint> paintedA,paintedB;
    const auto paintA=composeNativePitchStroke(seamA,{{0,65},{1,66}},paintedA,store,error,true,false);
    const auto paintB=composeNativePitchStroke(seamB,{{0,66},{1,67}},paintedB,store,error,false,true);
    check("artificial_per_note_edges_do_not_reset_continuous_paint",paintA&&paintB&&evaluatePitchCurve(paintedA,1)==66&&evaluatePitchCurve(paintedB,0)==66);
    NoteData unionNote;unionNote.durationSeconds=1;unionNote.pitchControlPoints={{-.2,55},{0,57},{1,60},{1.3,58}};
    std::vector<PitchCurveEditPoint> unionPaint;
    check("authored_union_domain_accepts_real_negative_and_tail_support",composeNativePitchStroke(unionNote,{{-.2,65},{1.3,67}},unionPaint,store,error,false,false)
        &&unionPaint.front().timeSeconds==-.2&&unionPaint.back().timeSeconds==1.3&&unionPaint.front().targetMidi==65&&unionPaint.back().targetMidi==67);
    check("free_edge_cannot_discard_real_outside_global_support",!composeNativePitchStroke(unionNote,{{0,65},{1,66}},unionPaint,store,error,false,false));
    const std::vector<PitchCurveEditPoint> straight={{.2,65,PitchCurveShape::linear},{1.8,67,PitchCurveShape::linear}};
    std::vector<PitchCurveEditPoint> guardedStraight;const auto straightOk=composeNativePitchStroke(sparse,straight,guardedStraight,store,error);
    double straightError=0;for(double t=.202;t<1.799;t+=.001)straightError=std::max(straightError,std::abs(static_cast<double>(evaluatePitchCurve(straight,t))-evaluatePitchCurve(guardedStraight,t)));
    log+="guarded_straight_sparse_max_midi_error="+juce::String(straightError,12)+"\n";
    check("true_guards_do_not_shift_new_sparse_straight_line_slope",straightOk&&straightError<=1.e-5);
    auto derived=sparse;derived.pitchControlPoints.clear();derived.contour.clear();for(int i=0;i<=20;++i)derived.contour.push_back({i/10.,static_cast<float>(i*2),static_cast<float>(i),true});
    derived.contour.insert(derived.contour.begin()+2,{.1337,12.37f,3.14f,false});
    const auto derivedStraight=composeNativePitchStroke(derived,straight,guardedStraight,store,error);double derivedStraightError=0;
    for(double t=.202;t<1.799;t+=.001)derivedStraightError=std::max(derivedStraightError,std::abs(static_cast<double>(evaluatePitchCurve(straight,t))-evaluatePitchCurve(guardedStraight,t)));
    check("true_guards_do_not_shift_new_derived_straight_line_slope",derivedStraight&&!store&&derivedStraightError<=1.e-5);
    check("compose_derived_returns_only_local_window",composeNativePitchStroke(derived,stroke,out,store,error)&&!store&&out.front().timeSeconds==.6&&out.back().timeSeconds==1.1);
    ProjectData strokeData;TrackData strokeTrack;strokeTrack.id="strokes";strokeTrack.pitchAlgorithm=PitchAlgorithm::world;ClipData strokeClip;strokeClip.id="stroke-clip";strokeClip.durationSeconds=strokeClip.sourceDurationSeconds=2;strokeClip.notes={derived};strokeTrack.clips={strokeClip};strokeData.tracks={strokeTrack};model.resetDocument(strokeData);
    ProjectModel::PointCurveEdit edit;edit.noteId="stroke";edit.pitchPoints=out;edit.storePitchControlPoints=store;
    check("derived_batch_uses_non_promoting_setter",model.setPointCurves({edit},model.revisionNumber())&&model.snapshot().tracks[0].clips[0].notes[0].pitchControlPoints.empty());
    const auto afterStroke=model.snapshot().tracks[0].clips[0].notes[0];bool facts=true;
    for(const auto& p:derived.contour)
    {
        const auto found=std::find_if(afterStroke.contour.begin(),afterStroke.contour.end(),[&](const auto& q){return q.timeSeconds==p.timeSeconds;});
        facts&=found!=afterStroke.contour.end();if(found==afterStroke.contour.end())continue;
        facts&=found->relativeCents==p.relativeCents&&found->withoutVibratoCents==p.withoutVibratoCents&&found->voiced==p.voiced;
        if(p.timeSeconds<.6||p.timeSeconds>1.1)facts&=found->hasManualTarget==p.hasManualTarget&&found->manualTargetCents==p.manualTargetCents;
    }
    check("local_derived_edit_preserves_every_old_source_tuple_outside_flags",facts);
    const auto noStrokeFingerprint=model.contentFingerprint();const auto noStrokeRevision=model.revisionNumber();
    check("zero_and_offscope_strokes_do_not_create_undo",!composeNativePitchStroke(derived,{{.5,60},{.5,61}},out,store,error)
        &&!composeNativePitchStroke(derived,{{-.1,60},{.2,61}},out,store,error)&&model.contentFingerprint()==noStrokeFingerprint&&model.revisionNumber()==noStrokeRevision);
    const auto written=output.getChildFile("forced-portamento-smoke.txt").replaceWithText(log);std::cout<<"report_written="<<written<<std::endl;return ok&&written;
}
}
