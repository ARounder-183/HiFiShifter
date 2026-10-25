#pragma once
#include "../NativeNoteTiming.h"
#include "../PianoRollComponent.h"
#include <iostream>

namespace hachi
{
inline bool runVowelStretchAlignmentSmoke(const juce::File& output)
{
    if (!output.createDirectory()) return false;
    bool ok=true;int checks=0;juce::Array<juce::var> results;
    const auto check=[&](const char* name,bool pass){ok&=pass;++checks;std::cout<<name<<'='<<pass<<std::endl;
        auto item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("pass",pass);results.add(juce::var(item));};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-6;};
    const auto makeNote=[](const juce::String& id,double start,double source)
    {
        NoteData note;note.id=id;note.label="a";note.startSeconds=start;note.durationSeconds=1;
        note.midiNote=60.4f;note.sourceMidiCenter=60;note.sourcePitchMeasured=true;note.consonantSeconds=.2;
        note.utauPreutteranceOverrideEnabled=true;note.utauPreutteranceSeconds=.12;
        note.materialSourceStartSeconds=note.nativeSourceStartSeconds=source;note.materialSourceEndSeconds=note.nativeSourceEndSeconds=source+1;
        note.pitchControlPoints={{-.05,59.2f},{0,60.f},{.12,61.25f},{.5,60.75f},{1.1,62.f}};
        note.contour={{0,-20,-10,true,37,true},{.5,25,20,true,41,true},{1,0,0,true,12,true}};
        note.amplitudeEnvelope={{0,-2},{1,-1}};note.vibratoReferenceDurationSeconds=1;note.vibratoTimeOffsetSeconds=.03;
        note.nativeSegments={{"c","a",NativeSegmentRole::consonant,0,.2,"user",1,.12,0,false,1},
            {"v","a",NativeSegmentRole::vowel,.2,1,"user",1,.12,0,true,1}};
        return note;
    };
    ClipData base;base.id="physical";base.startSeconds=1.03;base.durationSeconds=base.sourceDurationSeconds=1;
    base.sourceOffsetSeconds=4;base.sourceFile=output.getChildFile("physical-source.wav");base.sourceTimeMap={{0,0},{.2,.2},{1,1}};
    base.notes={makeNote("n",0,4)};
    const auto valuesKept=[](const NoteData& a,const NoteData& b)
    {
        if(a.midiNote!=b.midiNote||a.sourceMidiCenter!=b.sourceMidiCenter||a.pitchControlPoints.size()!=b.pitchControlPoints.size()
            ||a.contour.size()!=b.contour.size()||a.pronunciationKind!=b.pronunciationKind)return false;
        for(std::size_t i=0;i<a.pitchControlPoints.size();++i)if(a.pitchControlPoints[i].targetMidi!=b.pitchControlPoints[i].targetMidi)return false;
        for(std::size_t i=0;i<a.contour.size();++i)if(a.contour[i].relativeCents!=b.contour[i].relativeCents
            ||a.contour[i].manualTargetCents!=b.contour[i].manualTargetCents||a.contour[i].voiced!=b.contour[i].voiced)return false;
        return true;
    };
    const auto localDataKept=[&](const NoteData& a,const NoteData& b)
    {
        if(!valuesKept(a,b)||a.durationSeconds!=b.durationSeconds||a.consonantSeconds!=b.consonantSeconds
            ||a.vibratoReferenceDurationSeconds!=b.vibratoReferenceDurationSeconds||a.vibratoTimeOffsetSeconds!=b.vibratoTimeOffsetSeconds
            ||a.amplitudeEnvelope.size()!=b.amplitudeEnvelope.size())return false;
        for(std::size_t i=0;i<a.pitchControlPoints.size();++i)if(a.pitchControlPoints[i].timeSeconds!=b.pitchControlPoints[i].timeSeconds)return false;
        for(std::size_t i=0;i<a.contour.size();++i)if(a.contour[i].timeSeconds!=b.contour[i].timeSeconds)return false;
        for(std::size_t i=0;i<a.amplitudeEnvelope.size();++i)if(a.amplitudeEnvelope[i].timeSeconds!=b.amplitudeEnvelope[i].timeSeconds
            ||a.amplitudeEnvelope[i].gainDb!=b.amplitudeEnvelope[i].gainDb)return false;
        return true;
    };
    const auto targetAt=[](const ClipData& clip,double localSource){return nativeTargetTimeAt(nativeClipClock(clip),localSource);};
    const auto right=planNativeNoteMove(base,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("right_stretch_1_to_2_keeps_C_point20_and_P_point12",right&&near(right->clip.notes[0].durationSeconds,2)
        &&near(targetAt(right->clip,.2),.2)&&near(targetAt(right->clip,.12),.12)&&near(right->clip.notes[0].consonantSeconds,.2));
    check("right_stretch_preserves_source_and_pitch_values",right&&near(right->clip.notes[0].nativeSourceStartSeconds,4)
        &&near(right->clip.notes[0].nativeSourceEndSeconds,5)&&valuesKept(base.notes[0],right->clip.notes[0]));
    const auto left=planNativeNoteMove(base,{"n"},.1,0,NativeNoteTimeEdit::leftEdge);
    check("left_stretch_moves_leading_protected_block_rigidly",left&&near(left->clip.notes[0].startSeconds,.1)
        &&near(left->clip.notes[0].startSeconds+left->clip.notes[0].durationSeconds,1)
        &&near(targetAt(left->clip,.2)-targetAt(left->clip,0),.2)
        &&near(nativeNotePreutteranceTarget(left->clip.notes[0],left->clip),.22));
    const auto shrink=planNativeNoteMove(base,{"n"},-.95,0,NativeNoteTimeEdit::rightEdge);
    check("impossible_contraction_clamps_without_compressing_C",shrink&&shrink->delta>-.81
        &&near(targetAt(shrink->clip,.2),.2)&&shrink->clip.notes[0].durationSeconds>.2);
    auto early=base;early.notes[0].consonantSeconds=.1;early.notes[0].utauPreutteranceSeconds=.3;
    early.notes[0].nativeSegments={{"c","a",NativeSegmentRole::consonant,0,.1,"user",1,.3,0,false,1},
        {"v","a",NativeSegmentRole::vowel,.1,1,"user",1,.3,0,true,1}};
    const auto earlyPlan=planNativeNoteMove(early,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("early_V_before_P_is_protected_even_when_C_earlier",earlyPlan&&near(targetAt(earlyPlan->clip,.1),.1)&&near(targetAt(earlyPlan->clip,.3),.3));
    auto custom=base;custom.sourceTimeMap={{0,0},{.1,.05},{.3,.2},{.5,.4},{1,1}};custom.notes[0].consonantSeconds=.3;
    const auto oldP=targetAt(custom,.12);const auto customPlan=planNativeNoteMove(custom,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("custom_old_C_clock_duration_is_preserved_not_raw_reset",customPlan&&near(targetAt(customPlan->clip,.2),.3)
        &&near(targetAt(customPlan->clip,.12),oldP)&&near(targetAt(customPlan->clip,.05),.1));
    check("custom_old_source_anchors_survive_composed_warp",customPlan&&near(nativeSourceTimeAt(nativeClipClock(customPlan->clip),.5+(.5-.3)/.7),.4));
    auto zones=base;zones.notes[0].vowelSplitsExplicit=true;zones.notes[0].nativeSegments={
        {"c","a",NativeSegmentRole::consonant,0,.2,"user",1,.12,0,false,1},
        {"v1","a",NativeSegmentRole::vowel,.2,.4,"user",1,.12,0,true,1},
        {"noise","x",NativeSegmentRole::noise,.4,.5,"user",1,.12,0,false,1},
        {"v2","a",NativeSegmentRole::vowel,.5,.8,"user",1,.12,0,true,1},
        {"end","x",NativeSegmentRole::ending,.8,1,"user",1,.12,0,false,1}};
    const auto zonePlan=planNativeNoteMove(zones,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("internal_noise_and_ending_keep_heard_durations",zonePlan
        &&near(targetAt(zonePlan->clip,.5)-targetAt(zonePlan->clip,.4),.1)
        &&near(targetAt(zonePlan->clip,1)-targetAt(zonePlan->clip,.8),.2));
    auto weighted=base;weighted.notes[0].vowelSplitsExplicit=true;weighted.notes[0].nativeSegments={
        {"c","a",NativeSegmentRole::consonant,0,.2,"user",1,.12,0,false,1},
        {"v1","a",NativeSegmentRole::vowel,.2,.5,"user",1,.12,0,true,1},
        {"v2","a",NativeSegmentRole::vowel,.5,.8,"user",1,.12,0,true,3},
        {"end","x",NativeSegmentRole::ending,.8,1,"user",1,.12,0,false,1}};
    const auto weights=planNativeNoteMove(weighted,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("weighted_V_progress_controls_only_elastic_displacement",weights&&near(targetAt(weights->clip,.5),.75)
        &&near(targetAt(weights->clip,.8),1.8)&&near(targetAt(weights->clip,1),2));
    auto analysisEpochs=base;analysisEpochs.notes[0].nativeSegments.clear();
    for(int i=0;i<1000;++i)
    {
        NativeSegment segment;segment.id="epoch_"+juce::String(i);segment.alias="automatic";
        segment.role=i<200?NativeSegmentRole::consonant:NativeSegmentRole::vowel;
        segment.sourceStartSeconds=i/1000.0;segment.sourceEndSeconds=(i+1)/1000.0;segment.alignmentSeconds=.12;
        analysisEpochs.notes[0].nativeSegments.push_back(segment);
    }
    const auto epochRow=nativeNoteSourceTiming(analysisEpochs.notes[0],analysisEpochs,nativeClipClock(analysisEpochs));
    check("runtime_analysis_epochs_do_not_infer_authored_vowel_splits",!epochRow.vowelSplitsExplicit
        &&pronunciationSegmentsFor(epochRow).size()==2&&analysisEpochs.notes[0].nativeSegments.size()==1000);
    const auto epochPlan=planNativeNoteMove(analysisEpochs,{"n"},1,0,NativeNoteTimeEdit::rightEdge);
    check("runtime_default_epoch_coalescing_still_preserves_C_stretch_policy",epochPlan&&near(targetAt(epochPlan->clip,.2),.2)
        &&near(epochPlan->clip.notes[0].durationSeconds,2));
    const auto authoredRow=nativeNoteSourceTiming(weighted.notes[0],weighted,nativeClipClock(weighted));
    check("runtime_authored_split_intent_is_copied_not_guessed",authoredRow.vowelSplitsExplicit
        &&pronunciationSegmentsFor(authoredRow).size()==4);
    auto three=base;three.durationSeconds=three.sourceDurationSeconds=3;three.sourceTimeMap={{0,0},{3,3}};
    three.notes={makeNote("x",0,4),makeNote("a",1,5),makeNote("b",2,6)};
    const auto alignPlan=planNativeNoteMove(three,{"b"},.1,0,NativeNoteTimeEdit::move);
    check("align_previous_V_absorbs_delta_without_rippling_prior_chain",alignPlan
        &&near(alignPlan->clip.notes[0].startSeconds,0)&&near(alignPlan->clip.notes[0].durationSeconds,1)
        &&near(alignPlan->clip.notes[1].startSeconds,1)&&near(alignPlan->clip.notes[1].durationSeconds,1.1)
        &&near(targetAt(alignPlan->clip,1.2)-targetAt(alignPlan->clip,1),.2)
        &&near(alignPlan->clip.notes[2].startSeconds,2.1)&&near(alignPlan->clip.notes[2].durationSeconds,1));
    const auto ordinaryLeft=planNativeNoteMove(three,{"b"},.1,0,NativeNoteTimeEdit::leftEdge);
    check("ordinary_left_drag_still_ripples_touching_previous_notes_rigidly",ordinaryLeft
        &&near(ordinaryLeft->clip.notes[0].durationSeconds,1)&&near(ordinaryLeft->clip.notes[1].durationSeconds,1)
        &&near(ordinaryLeft->clip.startSeconds+ordinaryLeft->clip.notes[0].startSeconds,three.startSeconds+.1));

    auto linked=base;linked.sourceFile=juce::File{};linked.durationSeconds=linked.sourceDurationSeconds=2;
    linked.sourceOffsetSeconds=0;linked.sourceTimeMap={{0,0},{2,2}};linked.nativeAudioLinked=true;linked.parts.clear();linked.notes.clear();
    for(int i=0;i<2;++i)
    {
        auto part=base;part.id="source-"+juce::String(i);part.startSeconds=i;part.sourceOffsetSeconds=10+i*10;
        part.sourceFile=output.getChildFile("source-"+juce::String(i)+".wav");part.notes.clear();linked.parts.push_back(part);
        auto note=makeNote("owner-"+juce::String(i),i,part.sourceOffsetSeconds);note.clipPartId=part.id;linked.notes.push_back(note);
    }
    const auto linkedPlan=planNativeNoteMove(linked,{"owner-1"},1,0,NativeNoteTimeEdit::rightEdge);
    check("linked_multiple_sources_project_physical_C_before_proxy_clear",linkedPlan&&linkedPlan->clip.parts.size()==2
        &&near(linkedPlan->clip.parts[1].sourceOffsetSeconds,20)&&near(targetAt(linkedPlan->clip.parts[1],.2),.2)
        &&near(linkedPlan->clip.parts[1].durationSeconds,2)&&near(linkedPlan->clip.notes[1].nativeSourceStartSeconds,20));
    auto independent=linked;independent.nativeAudioLinked=false;
    const auto independentPlan=planNativeNoteMove(independent,{"owner-1"},1,0,NativeNoteTimeEdit::rightEdge);
    check("independent_multiple_source_children_reuse_protected_planner",independentPlan&&independentPlan->clip.parts.size()==2
        &&near(targetAt(independentPlan->clip.parts[0],.2),.2)&&near(targetAt(independentPlan->clip.parts[1],.2),.2));

    ProjectData data;data.bpm=120;data.gridDivision="1/16";TrackData track;track.id="timing";track.compose=true;
    track.pitchAlgorithm=PitchAlgorithm::world;track.clips={base};data.tracks={track};
    ProjectModel model;model.resetDocument(data);const auto fingerprint=model.contentFingerprint();juce::String error;
    check("physical_align_action_succeeds",model.alignNotePreutteranceToGrid("n",.25,error));
    auto aligned=model.snapshot();const auto& alignedClip=aligned.tracks[0].clips[0];
    check("physical_preutterance_hits_exact_grid_not_nominal_onset",near(alignedClip.startSeconds+nativeNotePreutteranceTarget(alignedClip.notes[0],alignedClip),1.125)
        &&near(alignedClip.notes[0].durationSeconds,base.notes[0].durationSeconds)
        &&localDataKept(base.notes[0],alignedClip.notes[0]));
    check("alignment_is_exactly_one_undo",model.undo()&&model.contentFingerprint()==fingerprint&&!model.canUndo());
    check("alignment_redo_preserves_pitch_data",model.redo()&&valuesKept(base.notes[0],model.snapshot().tracks[0].clips[0].notes[0]));
    const auto saved=output.getChildFile("aligned.hjpx");ProjectModel reopened;
    check("aligned_source_clock_and_pitch_survive_reopen",model.save(saved,error)&&reopened.load(saved,error)
        &&valuesKept(base.notes[0],reopened.snapshot().tracks[0].clips[0].notes[0])
        &&near(reopened.snapshot().tracks[0].clips[0].startSeconds+nativeNotePreutteranceTarget(reopened.snapshot().tracks[0].clips[0].notes[0],reopened.snapshot().tracks[0].clips[0]),1.125));
    auto tempo=data;tempo.tempoChanges={{2,90}};tempo.tracks[0].clips[0].startSeconds=.99;model.resetDocument(tempo);
    check("alignment_uses_full_tempo_map",model.alignNotePreutteranceToGrid("n",.25,error));
    aligned=model.snapshot();
    check("tempo_changed_pre_target_is_exact_musical_grid",near(aligned.tracks[0].clips[0].startSeconds
        +nativeNotePreutteranceTarget(aligned.tracks[0].clips[0].notes[0],aligned.tracks[0].clips[0]),tempo.secondsForQuarterPosition(2.25)));
    auto nearest=data;nearest.tracks[0].clips[0].startSeconds=1.12;
    nearest.tracks[0].clips[0].durationSeconds=nearest.tracks[0].clips[0].sourceDurationSeconds=1.205;
    nearest.tracks[0].clips[0].sourceTimeMap={{0,0},{1.205,1.205}};
    auto tinyNext=makeNote("tiny-next",1,5);tinyNext.durationSeconds=.205;tinyNext.nativeSourceEndSeconds=5.205;
    nearest.tracks[0].clips[0].notes.push_back(tinyNext);
    model.resetDocument(nearest);check("nearest_infeasible_grid_searches_next_feasible_grid",model.alignNotePreutteranceToGrid("n",.25,error));
    aligned=model.snapshot();check("alignment_never_returns_clamped_off_grid",near(aligned.tracks[0].clips[0].startSeconds
        +nativeNotePreutteranceTarget(aligned.tracks[0].clips[0].notes[0],aligned.tracks[0].clips[0]),1.125));
    auto fixed=data;fixed.tracks[0].clips[0].durationSeconds=fixed.tracks[0].clips[0].sourceDurationSeconds=2;
    fixed.tracks[0].clips[0].sourceTimeMap={{0,0},{2,2}};
    auto fixedNeighbour=makeNote("fixed-neighbour",1,5);fixedNeighbour.consonantSeconds=1;
    fixed.tracks[0].clips[0].notes.push_back(fixedNeighbour);
    model.resetDocument(fixed);const auto fixedBefore=model.contentFingerprint();
    check("fixed_neighbour_impossible_alignment_rejects_without_undo",!model.alignNotePreutteranceToGrid("n",.25,error)
        &&error.isNotEmpty()&&model.contentFingerprint()==fixedBefore&&!model.canUndo());
    for(const auto engine:{PitchAlgorithm::world,PitchAlgorithm::nsfHifigan})
    {
        auto same=data;same.tracks[0].pitchAlgorithm=engine;same.tracks[0].clips[0].notes[0].pronunciationKind=PronunciationKind::breath;
        model.resetDocument(same);const bool done=model.alignNotePreutteranceToGrid("n",.25,error);const auto result=model.snapshot();
        check("same_timing_action_preserves_engine_and_breath_identity",done&&result.tracks[0].pitchAlgorithm==engine
            &&result.tracks[0].clips[0].notes[0].pronunciationKind==PronunciationKind::breath
            &&valuesKept(same.tracks[0].clips[0].notes[0],result.tracks[0].clips[0].notes[0]));
    }
    for(const auto engine:{PitchAlgorithm::world,PitchAlgorithm::utau})
    {
        auto score=data;score.tracks[0].pitchAlgorithm=engine;auto& clip=score.tracks[0].clips[0];clip.sourceFile=juce::File{};clip.parts.clear();
        clip.startSeconds=.2;clip.durationSeconds=2;clip.notes[0].startSeconds=.13;clip.notes[0].nativeSourceStartSeconds=-1;
        model.resetDocument(score);const auto before=clip.notes[0];const bool done=model.alignNotePreutteranceToGrid("n",.25,error);const auto result=model.snapshot();
        check("source_less_score_uses_nominal_onset_without_changing_engine_or_curves",done&&result.tracks[0].pitchAlgorithm==engine
            &&near(result.tracks[0].clips[0].startSeconds+result.tracks[0].clips[0].notes[0].startSeconds,.375)
            &&localDataKept(before,result.tracks[0].clips[0].notes[0]));
    }
    model.resetDocument(data);I18n strings;PianoRollComponent roll(model,strings);roll.setBounds(0,0,1100,750);
    roll.setPixelsPerSecond(240);roll.setFocusedTrack(track.id);roll.setFocusedClip(base.id);roll.setTool(PianoRollComponent::Tool::note);roll.diagnosticRefresh();
    const auto hit=roll.diagnosticHitBounds(0);const auto at=juce::Point<float>(hit.getCentreX(),hit.getCentreY());
    const auto event=juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),1,0,0,0,0,
        &roll,&roll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),2,false);
    roll.mouseDoubleClick(event);const auto clicked=model.snapshot();
    check("actual_note_tool_double_click_aligns_without_flatten_or_semitone_snap",near(clicked.tracks[0].clips[0].startSeconds
        +nativeNotePreutteranceTarget(clicked.tracks[0].clips[0].notes[0],clicked.tracks[0].clips[0]),1.125)
        &&localDataKept(base.notes[0],clicked.tracks[0].clips[0].notes[0])&&!clicked.tracks[0].clips[0].notes[0].nativeIndependentPitch);
    check("actual_double_click_adds_only_one_undo",model.undo()&&model.contentFingerprint()==fingerprint&&!model.canUndo());
    // Native OTTO notes retain their timing action; all UTAU lyrics remain
    // editable, regardless of bank, mode, renderer, selection or alias text.
    auto nativeBlank=data;nativeBlank.tracks[0].trackType=TrackType::hachi;
    nativeBlank.tracks[0].clips[0].notes[0].label.clear();
    model.resetDocument(nativeBlank);roll.diagnosticRefresh();roll.mouseDoubleClick(event);
    check("native_otto_empty_note_does_not_open_lyric_editor",!roll.diagnosticAliasEditorOpen());
    roll.setReadsVoicebankInBackground(false);
    const auto bank=output.getChildFile("empty-lyric-bank");bank.createDirectory();
    for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
        for(const bool bound:{false,true})
        for(const auto engine:{PitchAlgorithm::utau,PitchAlgorithm::nsfHifigan})
        for(const auto lyric:{juce::String{},juce::String("   "),juce::String::fromUTF8("你好"),
            juce::String("a R"),juce::String("-"),juce::String("R"),juce::String("missing-alias")})
        {
            ProjectData score;TrackData t;t.id="lyrics-track";t.trackType=TrackType::utau;
            t.pitchAlgorithm=engine;t.utauMode=mode;t.compose=true;
            if(bound)t.voicebankDirectory=bank;
            ClipData c;c.id="lyrics-clip";c.durationSeconds=3;
            NoteData n;n.id="empty-note";n.startSeconds=.13;n.durationSeconds=.8;n.midiNote=60;
            n.label=lyric;c.notes={n};
            n.id="other-note";n.startSeconds=1.5;n.label="keep";c.notes.push_back(n);t.clips={c};score.tracks={t};
            model.resetDocument(score);roll.setFocusedTrack(t.id);roll.setFocusedClip(c.id);roll.diagnosticRefresh();
            roll.setSelectedNoteIds(bound?std::vector<juce::String>{"empty-note","other-note"}:std::vector<juce::String>{});
            const auto before=model.contentFingerprint();
            const auto position=roll.diagnosticHitBounds(0).getCentre();
            const auto click=[&](int count){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),position,
                juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),1,0,0,0,0,&roll,&roll,
                juce::Time::getCurrentTime(),position,juce::Time::getCurrentTime(),count,false);};
            roll.mouseDown(click(1));roll.mouseUp(click(1));
            roll.mouseDown(click(2));roll.mouseUp(click(2));roll.mouseDoubleClick(click(2));
            check("utau_any_lyric_doubleclick_opens_without_alignment",roll.diagnosticAliasEditorOpen()
                &&roll.diagnosticAliasEditorNoteId()=="empty-note"&&roll.diagnosticAliasEditorText()==lyric
                &&roll.diagnosticAliasEditorAllSelected()&&model.contentFingerprint()==before);
            roll.diagnosticTypeInAliasEditor("discard");roll.diagnosticCancelAliasEdit();
            check("utau_any_lyric_cancel_keeps_score",!roll.diagnosticAliasEditorOpen()&&model.contentFingerprint()==before);
            roll.mouseDoubleClick(click(2));roll.diagnosticCommitAliasEdit(juce::String::fromUTF8("修改后"));
            const auto result=model.snapshot();
            check("utau_any_lyric_commit_targets_only_clicked_note",!roll.diagnosticAliasEditorOpen()
                &&result.tracks[0].clips[0].notes[0].label==juce::String::fromUTF8("修改后")
                &&result.tracks[0].clips[0].notes[1].label=="keep"
                &&near(result.tracks[0].clips[0].notes[0].startSeconds,.13)
                &&near(result.tracks[0].clips[0].notes[0].durationSeconds,.8));
        }
    auto report=new juce::DynamicObject();report->setProperty("ok",ok);report->setProperty("checks",checks);report->setProperty("results",juce::var(results));
    const bool written=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout<<"vowel_stretch_alignment_ok="<<(ok&&written)<<"; checks="<<checks<<std::endl;return ok&&written;
}
}
