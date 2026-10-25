#pragma once
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runTrackPitchScopeSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("track-pitch-scope-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;int count=0;juce::String log;
    constexpr double outsideMidiTolerance=1.e-5; // exact cubic representation, float-rounding allowance only
    const auto check=[&](const juce::String& name,bool pass,const juce::String& detail=juce::String{})
    {++count;ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+(detail.isEmpty()?juce::String{}:" | "+detail)+"\n";};
    const auto finish=[&]{const bool written=root.getChildFile("verification.txt").replaceWithText(log);std::cout<<"track_pitch_scope_passed="<<(ok?1:0)<<"|checks="<<count<<"|fixture="<<root.getFullPathName()<<std::endl;if(!ok)std::cerr<<log<<std::endl;return ok&&written;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-6;};
    const auto makeAudio=[](const juce::File& file,double midi)
    {
        juce::AudioBuffer<float> pcm(1,67200);const auto hz=440*std::exp2((midi-69)/12);
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,.1f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*hz*i/48000.0)));
        auto out=file.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?wav.createWriterFor(out.release(),48000,1,16,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    const auto sourceA=root.getChildFile("source-a.wav"),sourceB=root.getChildFile("source-b.wav");
    check("own_distinct_source_files",makeAudio(sourceA,60)&&makeAudio(sourceB,64));
    juce::MemoryBlock bytesA,bytesB;check("source_bytes_captured",sourceA.loadFileAsData(bytesA)&&sourceB.loadFileAsData(bytesB));if(!ok)return finish();
    const auto sparse=[](float centre)
    {
        std::vector<PitchCurveEditPoint> points{{-.15,centre-2,PitchCurveShape::linear},{0,centre-.5f,PitchCurveShape::easeIn},
            {.35,centre+1.5f,PitchCurveShape::customBezier},{.8,centre+2,PitchCurveShape::easeOut},
            {1.1,centre+3,PitchCurveShape::smooth},{1.55,centre+1.5f,PitchCurveShape::natural}};
        points[2].bezierX1=.22f;points[2].bezierY1=.12f;points[2].bezierX2=.78f;points[2].bezierY2=.88f;return points;
    };
    const auto makeNote=[&](const juce::String& id,float midi,bool dense)
    {
        NoteData note;note.id=id;note.label=id;note.durationSeconds=1.4;note.midiNote=note.sourceMidiCenter=midi;
        note.sourcePitchMeasured=true;note.nativePitchHandlesPlaced=true;note.utauAutoPitchTransition=false;
        note.pronunciationKind=id=="c"?PronunciationKind::breath:PronunciationKind::normal;
        note.amplitudeEnvelope={{0,-3},{1.4,-2}};
        if(!dense)note.pitchControlPoints=sparse(midi);
        for(int i=0;i<=280;++i)note.contour.push_back({i*.005,0,0,!(i>=20&&i<=30),dense?static_cast<float>(25*std::sin(i*.03)):0,dense});
        return note;
    };
    const auto facts=[&](const ProjectData& before,const ProjectData& after)
    {
        if(before.tracks.size()!=after.tracks.size())return false;
        for(std::size_t t=0;t<before.tracks.size();++t)
        {
            if(before.tracks[t].pitchAlgorithm!=after.tracks[t].pitchAlgorithm||before.tracks[t].clips.size()!=after.tracks[t].clips.size())return false;
            for(std::size_t c=0;c<before.tracks[t].clips.size();++c)
            {
                const auto& a=before.tracks[t].clips[c];const auto& b=after.tracks[t].clips[c];
                if(a.id!=b.id||a.sourceFile!=b.sourceFile||a.startSeconds!=b.startSeconds||a.durationSeconds!=b.durationSeconds
                    ||a.sourceOffsetSeconds!=b.sourceOffsetSeconds||a.sourceDurationSeconds!=b.sourceDurationSeconds
                    ||a.audioStartSeconds!=b.audioStartSeconds||a.sourceTimeMap.size()!=b.sourceTimeMap.size()||a.notes.size()!=b.notes.size())return false;
                for(std::size_t i=0;i<a.sourceTimeMap.size();++i)if(a.sourceTimeMap[i].sourceSeconds!=b.sourceTimeMap[i].sourceSeconds||a.sourceTimeMap[i].targetSeconds!=b.sourceTimeMap[i].targetSeconds)return false;
                for(std::size_t n=0;n<a.notes.size();++n)
                {
                    const auto& x=a.notes[n];const auto& y=b.notes[n];
                    if(x.id!=y.id||x.label!=y.label||x.startSeconds!=y.startSeconds||x.durationSeconds!=y.durationSeconds
                        ||x.midiNote!=y.midiNote||x.sourceMidiCenter!=y.sourceMidiCenter||x.sourcePitchMeasured!=y.sourcePitchMeasured
                        ||x.nativeSourceStartSeconds!=y.nativeSourceStartSeconds||x.nativeSourceEndSeconds!=y.nativeSourceEndSeconds
                        ||x.materialAudioId!=y.materialAudioId||x.materialRegionId!=y.materialRegionId||x.pronunciationKind!=y.pronunciationKind
                        ||x.simplifiedConsonant!=y.simplifiedConsonant||x.vowelSplitsExplicit!=y.vowelSplitsExplicit
                        ||x.connectedToNext!=y.connectedToNext||x.connectedToPrevious!=y.connectedToPrevious||x.amplitudeEnvelope.size()!=y.amplitudeEnvelope.size())return false;
                    for(std::size_t i=0;i<x.amplitudeEnvelope.size();++i)if(x.amplitudeEnvelope[i].timeSeconds!=y.amplitudeEnvelope[i].timeSeconds||x.amplitudeEnvelope[i].gainDb!=y.amplitudeEnvelope[i].gainDb)return false;
                    for(const auto& p:x.contour)
                    {
                        const auto found=std::find_if(y.contour.begin(),y.contour.end(),[&](const auto& q){return q.timeSeconds==p.timeSeconds;});
                        if(found==y.contour.end()||found->relativeCents!=p.relativeCents||found->withoutVibratoCents!=p.withoutVibratoCents||found->voiced!=p.voiced)return false;
                    }
                }
            }
        }
        return before.nativeConnections.size()==after.nativeConnections.size();
    };
    const auto keys=[](const ProjectData& data)
    {std::vector<std::uint64_t> values;for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)values.push_back(AudioEngine::utauNoteRenderHash(n));return values;};
    I18n strings;strings.setLanguage(I18n::Language::enUS);
    for(int engine=0;engine<2;++engine)
    {
        const juce::String prefix=engine==0?"native_":"utau_";
        ProjectData seed;TrackData track;track.id="current";track.compose=true;track.pitchAlgorithm=engine==0?PitchAlgorithm::world:PitchAlgorithm::utau;
        const auto clip=[&](const juce::String& id,double start,const juce::File& source,float midi,bool dense)
        {ClipData c;c.id=id;c.startSeconds=start;c.durationSeconds=c.sourceDurationSeconds=1.4;c.sourceFile=source;c.notes={makeNote(id,midi,dense)};return c;};
        track.clips={clip("a",0,sourceA,60,false),clip("b",2,sourceB,64,true),clip("c",4,{},67,false)};
        auto other=track;other.id="foreign";other.clips={clip("foreign-note",.5,sourceA,60,false)};seed.tracks={track,other};
        ProjectModel model;model.resetDocument(seed);PianoRollComponent roll(model,strings);roll.setReadsVoicebankInBackground(false);
        roll.setBounds(0,0,1500,1900);roll.setPixelsPerSecond(220);roll.setRowHeight(26);roll.setFocusedTrack("current");roll.setFocusedClip("a");roll.setShowPitchLine(true);
        const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();
        const auto event=[&](juce::Point<float> at,juce::Point<float> down,bool ctrl=false)
        {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier|(ctrl?juce::ModifierKeys::ctrlModifier:0)),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
        const auto point=[&](double t,float midi){return juce::Point<float>(roll.diagnosticEdgeX(t),roll.diagnosticYForMidi(midi));};
        const auto stroke=[&](double first,double last,float a,float b)
        {
            const auto from=point(first,a),to=point(last,b);roll.mouseDown(event(from,from));
            roll.mouseDrag(event(point(first+(last-first)*.4,a+(b-a)*.4f),from));roll.mouseDrag(event(to,from));roll.mouseUp(event(to,from));refresh();
        };
        check(prefix+"default_focused_clip_scope",roll.pitchScope()==PianoRollComponent::PitchScope::focusedClip);
        const auto revision=model.revisionNumber();const auto fingerprint=model.contentFingerprint();const auto renderKeys=keys(model.snapshot());
        roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);roll.setPitchScope(PianoRollComponent::PitchScope::focusedClip);roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);
        check(prefix+"scope_toggle_view_only_keeps_render_keys",model.revisionNumber()==revision&&model.contentFingerprint()==fingerprint&&keys(model.snapshot())==renderKeys);
        check(prefix+"all_current_track_clips_visible_not_foreign",roll.diagnosticHitCount()==3);
        roll.setFocusedTrack({});roll.setFocusedClip("a");refresh();
        check(prefix+"current_track_owner_fallback",roll.diagnosticHitCount()==3);roll.setFocusedTrack("current");
        roll.setTool(PianoRollComponent::Tool::draw);roll.setDrawAction(PianoRollComponent::DrawAction::pitchCurve);
        const auto before=model.snapshot();const auto beforeFingerprint=model.contentFingerprint();const auto beforeRevision=model.revisionNumber();
        stroke(.25,2.8,65,67);const auto drawn=model.snapshot();
        check(prefix+"one_freehand_changes_both_file_curves_in_one_revision",model.revisionNumber()==beforeRevision+1
            &&keys(drawn)[0]!=keys(before)[0]&&keys(drawn)[1]!=keys(before)[1]&&keys(drawn)[2]==keys(before)[2]&&keys(drawn)[3]==keys(before)[3]);
        check(prefix+"freehand_preserves_source_clocks_and_nonpitch_facts",facts(before,drawn));
        double outsideError=0,outsideAt=0;float outsideOld=0,outsideNew=0;
        const auto& oldSparse=before.tracks[0].clips[0].notes[0].pitchControlPoints;const auto& newSparse=drawn.tracks[0].clips[0].notes[0].pitchControlPoints;
        if(newSparse.empty())outsideError=1;
        // Existing authored A tail1.4..1.55 lies INSIDE global .25..2.8;
        // it is intentionally painted under the approved sparse union domain.
        else for(double t=-.15;t<=1.55;t+=.0025)if(t<.25-1.e-8)
        {
            const auto oldValue=evaluatePitchCurve(oldSparse,t),newValue=evaluatePitchCurve(newSparse,t);
            const auto difference=static_cast<double>(std::abs(oldValue-newValue));
            if(difference>outsideError){outsideError=difference;outsideAt=t;outsideOld=oldValue;outsideNew=newValue;}
        }
        check(prefix+"sparse_Bezier_outside_exact",!newSparse.empty()&&outsideError<=outsideMidiTolerance,
            "max_midi_error="+juce::String(outsideError,12)+" at="+juce::String(outsideAt,12)
                +" old="+juce::String(outsideOld,12)+" new="+juce::String(outsideNew,12));
        if(outsideError>outsideMidiTolerance)
        {
            const auto describe=[&](const char* name,const std::vector<PitchCurveEditPoint>& points)
            {
                for(const auto& p:points)log+=prefix+name+" t="+juce::String(p.timeSeconds,12)+" midi="+juce::String(p.targetMidi,12)
                    +" shape="+juce::String(static_cast<int>(p.shape))+" x1="+juce::String(p.bezierX1,12)+" y1="+juce::String(p.bezierY1,12)
                    +" x2="+juce::String(p.bezierX2,12)+" y2="+juce::String(p.bezierY2,12)+"\n";
            };
            describe("outside_original",oldSparse);describe("outside_composed",newSparse);
        }
        const auto& oldDense=before.tracks[0].clips[1].notes[0];const auto& newDense=drawn.tracks[0].clips[1].notes[0];
        bool denseOutside=newDense.pitchControlPoints.empty();
        for(const auto& p:oldDense.contour)if(p.timeSeconds>.8+1.e-8)
        {const auto found=std::find_if(newDense.contour.begin(),newDense.contour.end(),[&](const auto& q){return q.timeSeconds==p.timeSeconds;});denseOutside&=found!=newDense.contour.end()&&found->hasManualTarget==p.hasManualTarget&&found->manualTargetCents==p.manualTargetCents;}
        check(prefix+"derived_dense_outside_flags_values_unchanged_no_sparse_promotion",denseOutside);
        check(prefix+"one_undo_restores_all_files",model.undo()&&model.contentFingerprint()==beforeFingerprint&&!model.canUndo());refresh();
        const auto noNotesBefore=model.contentFingerprint();stroke(1.55,1.85,65,66);
        check(prefix+"gap_only_stroke_never_creates_notes_or_bridge",model.contentFingerprint()==noNotesBefore&&model.snapshot().nativeConnections.empty());
        // Independent genuine outside test retains BOTH prefix and suffix
        // coverage after extending global paint to existing sparse tail domains.
        model.resetDocument(seed);refresh();roll.setTool(PianoRollComponent::Tool::line);
        stroke(.25,1.05,65,66);const auto isolated=model.snapshot();
        const auto& isolatedCurve=isolated.tracks[0].clips[0].notes[0].pitchControlPoints;
        check(prefix+"in_body_stroke_retains_legacy_support_intent_false",
            !seed.tracks[0].clips[0].notes[0].pitchSupportAuthored&&!isolated.tracks[0].clips[0].notes[0].pitchSupportAuthored);
        double isolatedError=isolatedCurve.empty()?1.0:0.0;
        if(!isolatedCurve.empty())for(double t=-.15;t<=1.55;t+=.0025)if(t<.25-1.e-8||t>1.05+1.e-8)
            isolatedError=std::max(isolatedError,static_cast<double>(std::abs(evaluatePitchCurve(seed.tracks[0].clips[0].notes[0].pitchControlPoints,t)-evaluatePitchCurve(isolatedCurve,t))));
        check(prefix+"isolated_paint_both_Bezier_sides_outside_exact",!isolatedCurve.empty()&&isolatedError<=outsideMidiTolerance,
            "max_midi_error="+juce::String(isolatedError,12));
        auto alreadyAuthored=seed;alreadyAuthored.tracks[0].clips[0].notes[0].pitchSupportAuthored=true;
        model.resetDocument(alreadyAuthored);refresh();stroke(.25,1.05,65,66);
        check(prefix+"in_body_stroke_preserves_prior_explicit_support_intent",model.snapshot().tracks[0].clips[0].notes[0].pitchSupportAuthored);
        model.resetDocument(seed);refresh();roll.setTool(PianoRollComponent::Tool::line);
        const auto lineBefore=model.snapshot();const auto lineFingerprint=model.contentFingerprint();const auto lineRevision=model.revisionNumber();
        stroke(.3,4.75,66,69);const auto lineAfter=model.snapshot();
        check(prefix+"one_line_edits_audio_and_source_less_score_not_wrong_track",model.revisionNumber()==lineRevision+1&&keys(lineAfter)[0]!=keys(lineBefore)[0]
            &&keys(lineAfter)[1]!=keys(lineBefore)[1]&&keys(lineAfter)[2]!=keys(lineBefore)[2]&&keys(lineAfter)[3]==keys(lineBefore)[3]);
        check(prefix+"line_preserves_sources_and_no_implicit_connections",facts(lineBefore,lineAfter)&&lineAfter.nativeConnections.empty());
        {
            auto painted=root.getChildFile(prefix+"actual-global-line-across-files.png").createOutputStream();
            const juce::Rectangle<int> box(0,static_cast<int>(roll.diagnosticYForMidi(73)-40),1400,560);
            check(prefix+"actual_post_line_snapshot",painted&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(box),*painted));
        }
        check(prefix+"line_batch_one_undo",model.undo()&&model.contentFingerprint()==lineFingerprint&&!model.canUndo());refresh();
        // Contiguous DIFFERENT-file seams are inside the overall paint range,
        // not untouched outside edges. Inspect actual renderer request targets,
        // not only the controls or a line visually connecting two restored Ys.
        for(int sparseRight=0;sparseRight<2;++sparseRight)
        {
            auto contiguous=seed;contiguous.tracks[0].clips.resize(2);contiguous.tracks[0].clips[1].startSeconds=1.4;
            if(engine==0){contiguous.tracks[0].pitchAlgorithm=PitchAlgorithm::nsfHifigan;contiguous.tracks[0].nativeNsfAudio=true;}
            if(sparseRight)contiguous.tracks[0].clips[1].notes[0].pitchControlPoints=sparse(64);
            model.resetDocument(contiguous);refresh();roll.setTool(PianoRollComponent::Tool::line);
            const auto seamRevision=model.revisionNumber();const auto seamFingerprint=model.contentFingerprint();
            stroke(.25,2.6,65,67);const auto seam=model.snapshot();
            const auto tag=prefix+(sparseRight?juce::String("sparse_seam_"):juce::String("dense_seam_"));
            check(tag+"one_atomic_stroke_no_physical_or_graph_change",model.revisionNumber()==seamRevision+1&&facts(contiguous,seam)&&seam.nativeConnections.empty());
            check(tag+"actual_authored_tail_stroke_records_explicit_intent",seam.tracks[0].clips[0].notes[0].pitchSupportAuthored);
            check(tag+"other_note_intent_matches_actual_prebody_paint",seam.tracks[0].clips[1].notes[0].pitchSupportAuthored==static_cast<bool>(sparseRight));
            const std::vector<PitchCurveEditPoint> requested{{.25,65,PitchCurveShape::linear},{2.6,67,PitchCurveShape::linear}};
            constexpr double seamTolerance=1.e-4; // MIDI float interpolation, not a 1ms legacy-pitch excursion
            bool visibleContinuous=true;double visibleError=0;
            for(const auto noteId:{juce::String("a"),juce::String("b")})for(const double t:{1.399,1.4,1.401})
            {
                if(noteId=="b"&&!sparseRight&&t<1.4)continue; // derived target has no authored pre-body domain
                const auto shown=roll.diagnosticPitchLineAt(noteId,t);
                const auto expected=evaluatePitchCurve(requested,t);
                if(!shown)visibleContinuous=false;else visibleError=std::max(visibleError,static_cast<double>(std::abs(*shown-expected)));
                log+=tag+"paint "+noteId+" abs="+juce::String(t,9)+" actual="+(shown?juce::String(*shown,9):juce::String("missing"))+" requested="+juce::String(expected,9)+"\n";
            }
            check(tag+"actual_drawn_curve_has_no_restored_old_pitch",visibleContinuous&&visibleError<=seamTolerance,"max_midi_error="+juce::String(visibleError,12));
            if(engine!=0)
            {
                const auto projected=AudioEngine::projectTrackNotePitch(seam.tracks[0],seam.nativeConnections);
                for(const auto& plan:projected)
                {
                    const auto ci=plan.noteId=="a"?0:1;
                    const auto& note=seam.tracks[0].clips[ci].notes[0];
                    const auto own=ownPitchPoints(note);
                    log+=tag+"host "+plan.noteId+" support=["+juce::String(plan.supportStartSeconds,9)+","+juce::String(plan.supportEndSeconds,9)
                        +"] packetLast="+(plan.sampledLocalCents.empty()?juce::String("empty"):juce::String(plan.sampledLocalCents.back().timeSeconds,9))+"\n";
                    for(const double absolute:{1.399,1.4,1.401,1.405})
                    {
                        const auto local=absolute-plan.absoluteStartSeconds;
                        const auto shown=roll.diagnosticPitchLineAt(plan.noteId,absolute);
                        log+=tag+"host "+plan.noteId+" abs="+juce::String(absolute,9)+" local="+juce::String(local,9)
                            +" own="+juce::String(evaluatePitchCurve(own,local),9)
                            +" projected="+(plan.continuousLocalCents?juce::String(plan.midiNote+plan.continuousLocalCents(local)/100.f,9):juce::String("missing"))
                            +" piano="+(shown?juce::String(*shown,9):juce::String("missing"))+"\n";
                    }
                }
            }
            if(engine==0)
            {
                bool requestContinuous=true;double requestError=0;
                for(std::size_t ci=0;ci<2;++ci)
                {
                    const auto& c=seam.tracks[0].clips[ci];const auto req=AudioEngine::diagnosticNativeRequest(c,seam.tracks[0]);
                    const std::array<double,3> localTimes=ci==0?std::array<double,3>{1.39,1.395,1.4}:std::array<double,3>{0,.005,.01};
                    for(const auto t:localTimes)
                    {
                        const auto index=static_cast<std::size_t>(std::llround(t*1000/req.framePeriodMs));
                        const auto absolute=c.startSeconds+t;const auto expected=evaluatePitchCurve(requested,absolute);
                        if(index>=req.targetMidi.size()){requestContinuous=false;continue;}
                        const auto actual=req.targetMidi[index];requestError=std::max(requestError,static_cast<double>(std::abs(actual-expected)));
                        log+=tag+"NativeTargetMidi clip="+c.id+" frame="+juce::String(static_cast<int>(index))+" abs="+juce::String(absolute,9)
                            +" actual="+juce::String(actual,9)+" requested="+juce::String(expected,9)+"\n";
                    }
                }
                check(tag+"actual_NativeTargetMidi_seam_matches_user_line",requestContinuous&&requestError<=seamTolerance,"max_midi_error="+juce::String(requestError,12));
            }
            {
                auto painted=root.getChildFile(tag+"actual-global-line.png").createOutputStream();
                const juce::Rectangle<int> box(0,static_cast<int>(roll.diagnosticYForMidi(73)-40),800,560);
                check(tag+"actual_contiguous_line_snapshot",painted&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(box),*painted));
            }
            check(tag+"one_undo_restores_all_files",model.undo()&&model.contentFingerprint()==seamFingerprint&&!model.canUndo());refresh();
        }

        // Existing point selection uses actual Ctrl clicks across two different
        // source groups; its existing batch setter must remain one undo.
        auto pointData=seed;pointData.tracks[0].clips[1].notes[0].pitchControlPoints=sparse(64);
        model.resetDocument(pointData);refresh();roll.setTool(PianoRollComponent::Tool::points);
        const auto pa=roll.diagnosticPitchAnchors("a"),pb=roll.diagnosticPitchAnchors("b");
        if(pa.size()>2&&pb.size()>2)
        {
            const auto a=point(pa[2].timeSeconds,pa[2].targetMidi),b=point(2+pb[2].timeSeconds,pb[2].targetMidi);
            roll.mouseDown(event(a,a));roll.mouseUp(event(a,a));roll.mouseDown(event(b,b,true));roll.mouseUp(event(b,b,true));
            check(prefix+"actual_Ctrl_selects_points_across_files",roll.selectedNoteIds().size()==2);
            const auto groupBefore=model.contentFingerprint();const auto groupRevision=model.revisionNumber();const auto to=a.translated(11,-13);
            roll.mouseDown(event(a,a));roll.mouseDrag(event(to,a));roll.mouseUp(event(to,a));refresh();const auto moved=model.snapshot();
            const auto& ma=moved.tracks[0].clips[0].notes[0].pitchControlPoints;const auto& mb=moved.tracks[0].clips[1].notes[0].pitchControlPoints;
            check(prefix+"point_group_across_sources_moves_both_one_revision",model.revisionNumber()==groupRevision+1&&ma.size()==pa.size()&&mb.size()==pb.size()
                &&ma[2].targetMidi>pa[2].targetMidi&&near(ma[2].targetMidi-pa[2].targetMidi,mb[2].targetMidi-pb[2].targetMidi)
                &&near(ma[2].timeSeconds-pa[2].timeSeconds,mb[2].timeSeconds-pb[2].timeSeconds));
            check(prefix+"point_group_one_undo",model.undo()&&model.contentFingerprint()==groupBefore&&!model.canUndo());refresh();
        }
        else check(prefix+"actual_point_group_fixture_available",false);
        model.resetDocument(seed);refresh();roll.setTool(PianoRollComponent::Tool::draw);
        const auto cancelFingerprint=model.contentFingerprint();const auto from=point(.25,65),to=point(2.8,67);
        roll.mouseDown(event(from,from));roll.mouseDrag(event(to,from));roll.setPitchScope(PianoRollComponent::PitchScope::focusedClip);roll.mouseUp(event(to,from));refresh();
        check(prefix+"scope_switch_cancels_pending_stroke_without_commit",model.contentFingerprint()==cancelFingerprint);
        roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);roll.setSourceEditMode(true);refresh();
        check(prefix+"source_editor_remains_one_recording",roll.diagnosticHitCount()==1);roll.setSourceEditMode(false);refresh();
        roll.setPitchScope(PianoRollComponent::PitchScope::currentTrack);roll.setTool(PianoRollComponent::Tool::points);
        auto image=root.getChildFile(prefix+"whole-track-curves.png").createOutputStream();const juce::Rectangle<int> box(0,static_cast<int>(roll.diagnosticYForMidi(72)-40),1400,500);
        check(prefix+"actual_track_scope_component_snapshot",image&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(box),*image));
    }
    juce::MemoryBlock finalA,finalB;check("original_source_files_unchanged",sourceA.loadFileAsData(finalA)&&sourceB.loadFileAsData(finalB)&&finalA==bytesA&&finalB==bytesB);
    return finish();
}
}
