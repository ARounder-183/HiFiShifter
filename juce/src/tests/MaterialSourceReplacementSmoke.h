#pragma once
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "../NativeNoteTiming.h"
#include "../NativeTrimSource.h"
#include "../AudioEngine.h"
#include "../ProjectFileIO.h"
#include <iostream>
#include <functional>

namespace hachi
{
// Model + real native request contract only. No devices, GUI, model inference,
// or claims about actual resynthesized PCM (covered by a later adapter test).
inline bool runMaterialSourceReplacementSmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;bool ok=true;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    const auto done=[&]{const auto saved=output.getChildFile("material-source-replacement-smoke.txt").replaceWithText(log);std::cout<<"report_written="<<saved<<std::endl;return ok&&saved;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-6;};
    const auto aFile=output.getChildFile("source-A-C4.wav"),bFile=output.getChildFile("source-B-A3.wav"),cFile=output.getChildFile("source-C-A4.wav");
    const auto wav=[&](const juce::File& file,double midi,int seconds)
    {
        constexpr int rate=48000;juce::AudioBuffer<float> pcm(1,rate*seconds);const auto hz=440*std::pow(2.,(midi-69)/12);
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,static_cast<float>(.2*std::sin(juce::MathConstants<double>::twoPi*hz*i/rate)));
        auto stream=file.createOutputStream();if(!stream)return false;stream->setPosition(0);stream->truncate();juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.get(),rate,1,24,{},0));if(!writer)return false;stream.release();return writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    check("fixtures_written",wav(aFile,60,4)&&wav(bFile,57,2)&&wav(cFile,69,2));
    juce::MemoryBlock aBytes,bBytes,cBytes;check("fixture_bytes_captured",aFile.loadFileAsData(aBytes)&&bFile.loadFileAsData(bBytes)&&cFile.loadFileAsData(cBytes));
    const auto sourceFacts=[](float midi,double duration)
    {
        NoteData n;n.id="analysis";n.durationSeconds=duration;n.midiNote=n.sourceMidiCenter=midi;n.sourcePitchMeasured=true;
        for(int i=0;i<=static_cast<int>(std::llround(duration*200));++i)n.contour.push_back({i*.005,0,0,true});return std::vector<NoteData>{n};
    };
    SampleRegionSetting b;b.name="same";b.materialAudioId="B";b.materialRegionId="pron";b.regionStartSeconds=.25;b.regionEndSeconds=1.75;b.alignmentSeconds=.25;
    b.role=NativeSegmentRole::vowel;b.provenance="user";b.additionalAliases={"alt","other"};
    auto c=b;c.materialAudioId="C";
    const auto makeNote=[&](const char* id,const char* region,double start,float midi)
    {
        NoteData n;n.id=id;n.label=region;n.materialAudioId="A";n.materialRegionId=region;n.materialBoundAlias=region;
        n.startSeconds=start;n.durationSeconds=1;n.midiNote=midi;n.sourceMidiCenter=60;n.sourcePitchMeasured=true;
        n.nativeSourceStartSeconds=n.materialSourceStartSeconds=start;n.nativeSourceEndSeconds=n.materialSourceEndSeconds=start+1;
        n.nativeRole=NativeSegmentRole::vowel;n.nativeProvenance="user";n.contour={{0,0,0,true},{.5,0,0,true},{1,0,0,true}};
        n.pitchControlPoints={{0,midi},{.4,midi+1.2f,PitchCurveShape::customBezier,.12f,.8f,.86f,.3f},{1,midi+.4f,PitchCurveShape::natural}};
        n.amplitudeEnvelope={{0,-8,true},{.45,-1,true},{1,-5}};n.gain=.63f;n.breath=.12f;n.formantSemitones=.7f;n.tension=.2f;
        n.vibratoEnabled=true;n.vibratoDepthCents=53;n.vibratoCycleMs=170;n.vibratoReferenceDurationSeconds=1;n.vibratoTimeOffsetSeconds=.05;
        n.consonantSeconds=.08;n.attackSpeed=1.35f;n.nativeEnvelope.mode=1;return n;
    };
    ClipData clip;clip.id="parent";clip.sourceFile=aFile;clip.durationSeconds=clip.sourceDurationSeconds=3.5;clip.startSeconds=.5;
    clip.sourceTimeMap={{0,0},{1,1},{1.5,1.3},{2,2},{2.5,2.5},{3.5,3.5}};
    clip.gain=.7f;clip.gainEnvelope={{0,-2},{3.5,-5}};clip.nativeSourcePitch=nativeSourcePitchReference(sourceFacts(60,4),[](double t){return t;});clip.nativeSourcePitchComplete=true;
    clip.notes={makeNote("left","a-left",0,60),makeNote("middle","a-middle",1,64),makeNote("right","a-right",2.5,67)};
    TrackData track;track.id="native";track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;track.normalizeVolume=false;track.clips={clip};
    ProjectData fixture;fixture.tracks={track};fixture.nativeConnections={{"L-M","left","middle","pitch-and-amplitude",1.5,{},{}},{"M-R","middle","right","pitch-and-amplitude",2.5,{},{}}};
    fixture.tracks[0].clips[0].notes[0].connectedToNext=true;fixture.tracks[0].clips[0].notes[1].connectedToPrevious=true;
    fixture.tracks[0].clips[0].notes[1].connectedToNext=true;fixture.tracks[0].clips[0].notes[2].connectedToPrevious=true;
    const auto noteOf=[](const ProjectData& data,const juce::String& id){for(const auto& t:data.tracks)for(const auto& p:t.clips)for(const auto& n:p.notes)if(n.id==id)return n;return NoteData{};};
    const auto ownerOf=[&](const ProjectData& data,const juce::String& id){const auto n=noteOf(data,id);for(const auto& t:data.tracks)for(const auto& p:t.clips)for(const auto& v:p.notes)if(v.id==id){if(p.parts.empty())return p;for(const auto& s:p.parts)if(s.id==n.clipPartId)return s;}return ClipData{};};
    const auto compare=[&](const ProjectData& before,const ProjectData& after,const char* name)
    {
        const auto oldLines=sharedPitchLines(before.tracks[0]),newLines=sharedPitchLines(after.tracks[0]);const auto oldEnv=nativeSharedEnvelopes(before.tracks[0]),newEnv=nativeSharedEnvelopes(after.tracks[0]);
        bool stable=true;double pitchError=0,dbError=0,vibError=0;
        for(const auto& oldParent:before.tracks[0].clips)for(const auto& old:oldParent.notes)
        {
            const auto n=noteOf(after,old.id);stable&=n.id==old.id&&n.startSeconds==old.startSeconds&&n.durationSeconds==old.durationSeconds&&n.midiNote==old.midiNote
                &&n.gain==old.gain&&n.nativeEnvelope==old.nativeEnvelope&&n.formantSemitones==old.formantSemitones&&n.breath==old.breath&&n.tension==old.tension&&n.attackSpeed==old.attackSpeed&&n.consonantSeconds==old.consonantSeconds;
            const auto abs=oldParent.startSeconds+old.startSeconds;
            for(int i=0;i<1000;++i)
            {
                const auto t=i*.001+.0000001;const auto* x=oldLines.memberFor(old.id);const auto* y=newLines.memberFor(n.id);
                const auto a=x?x->line->midiAt(abs+t):evaluatePitchCurve(ownPitchPoints(old),t);const auto d=y?y->line->midiAt(abs+t):evaluatePitchCurve(ownPitchPoints(n),t);
                pitchError=std::max(pitchError,std::abs(static_cast<double>(a)-d));
                const auto e=oldEnv.find(old.id),f=newEnv.find(n.id);
                const auto g=e==oldEnv.end()?nativeEnvelopeDbAt(scaledAmplitudeEnvelope(old.amplitudeEnvelope,old.amplitudeEnvelopeBasePercent),t):nativeEnvelopeDbAt(*e->second.points,abs+t);
                const auto h=f==newEnv.end()?nativeEnvelopeDbAt(scaledAmplitudeEnvelope(n.amplitudeEnvelope,n.amplitudeEnvelopeBasePercent),t):nativeEnvelopeDbAt(*f->second.points,abs+t);
                dbError=std::max(dbError,std::abs(static_cast<double>(g)-h));vibError=std::max(vibError,std::abs(vibratoCentsAt(old,t)-vibratoCentsAt(n,t)));
            }
        }
        log+=juce::String(name)+" pitch="+juce::String(pitchError,12)+" db="+juce::String(dbError,12)+" vibrato="+juce::String(vibError,12)+"\n";
        check(name,stable&&pitchError<.0002&&dbError<.00005&&vibError<1.e-8);
    };
    ProjectModel model;model.resetDocument(fixture);const auto before=model.contentFingerprint();const auto revision=model.revisionNumber();
    auto preview=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","alt",error,false);
    check("pure_source_preflight",!preview.rejected&&preview.changed&&model.contentFingerprint()==before&&model.revisionNumber()==revision&&!model.canUndo());
    auto result=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","alt",error);auto bound=model.snapshot();
    check("middle_only_atomic_source_assignment",!result.rejected&&result.changed&&result.pendingAnalysis&&model.revisionNumber()==revision+1&&bound.tracks[0].clips[0].parts.size()==3);
    if(result.rejected){log+=error+"\n";return done();}
    const auto selected=noteOf(bound,"middle"),left=noteOf(bound,"left"),right=noteOf(bound,"right");
    const auto newOwner=ownerOf(bound,"middle"),oldLeft=ownerOf(bound,"left"),oldRight=ownerOf(bound,"right");
    check("alias_and_identity_not_ambiguous_filename_lookup",selected.label=="alt"&&selected.materialBoundAlias=="alt"&&!selected.materialAliasFollowsPrimary&&selected.materialAudioId=="B"&&newOwner.sourceFile==bFile);
    check("pending_new_source_has_no_old_acoustic_facts",selected.sourceMidiCenter<0&&!selected.sourcePitchMeasured&&!newOwner.nativeSourcePitch&&newOwner.materialSourceAnalysisPending&&newOwner.muted&&!newOwner.materialSourceResumeMuted
        &&std::all_of(selected.contour.begin(),selected.contour.end(),[](const auto& p){return !p.voiced&&p.relativeCents==0&&p.withoutVibratoCents==0&&p.hasManualTarget;}));
    check("other_refs_and_exact_source_gap_preserved",oldLeft.sourceFile==aFile&&oldRight.sourceFile==aFile&&left.materialRegionId=="a-left"&&right.materialRegionId=="a-right"
        &&near(oldRight.startSeconds,2)&&near(oldRight.sourceOffsetSeconds,2)&&near(oldRight.durationSeconds,1.5));
    check("selected_normalized_warp_preserved",near(newOwner.sourceOffsetSeconds,.25)&&near(newOwner.sourceDurationSeconds,1.5)&&near(nativeSourceTimeAt(nativeClipClock(newOwner),.5),.45));
    bool gainsOnce=bound.tracks[0].clips[0].gain==clip.gain&&bound.tracks[0].clips[0].gainEnvelope.size()==clip.gainEnvelope.size();
    for(const auto& p:bound.tracks[0].clips[0].parts)gainsOnce&=p.gain==1&&p.gainEnvelope.empty();
    for(const auto& p:expandedClipParts(bound.tracks[0].clips[0]))gainsOnce&=p.gain==clip.gain&&p.inheritedGainEnvelopes.size()==1;
    check("parent_gain_layers_applied_once",gainsOnce);
    check("engine_and_musical_connection_graph_unchanged",bound.tracks[0].pitchAlgorithm==track.pitchAlgorithm&&bound.nativeConnections.size()==2&&bound.nativeConnections[0].leftNoteId=="left"&&bound.nativeConnections[0].rightNoteId=="middle");
    compare(fixture,bound,"every_1ms_all_note_score_expression_preserved_pending");
    check("legacy_unguarded_analysis_cannot_complete_new_binding",!model.setNativeTrimSourceAnalysis(bFile,sourceFacts(57,2))
        &&!noteOf(model.snapshot(),"middle").sourcePitchMeasured&&ownerOf(model.snapshot(),"middle").muted);
    check("source_action_single_undo",model.undo()&&!model.canUndo()&&model.contentFingerprint()==before);
    check("source_action_redo_restores_pending_binding",model.redo()&&model.pendingMaterialSourceAnalyses().size()==1);
    const auto pendingFile=output.getChildFile("source-pending.hachi");ProjectModel pendingReload;
    check("pending_binding_roundtrip",model.save(pendingFile,error)&&pendingReload.load(pendingFile,error)&&pendingReload.pendingMaterialSourceAnalyses().size()==1&&ownerOf(pendingReload.snapshot(),"middle").muted);
    const auto loadTicket=pendingReload.pendingMaterialSourceAnalyses();
    check("reopen_cache_miss_requests_reschedule",!pendingReload.completeMaterialSourceAnalysis(loadTicket.front(),{},error)&&error.isNotEmpty());
    check("reopened_pending_gets_guarded_facts",pendingReload.completeMaterialSourceAnalysis(loadTicket.front(),sourceFacts(57,2),error));
    compare(fixture,pendingReload.snapshot(),"every_1ms_roundtrip_source_attach_preserves_score");
    // Current edits made while waiting must win over all old worker snapshots.
    model.transposeNote("middle",2);model.setNoteGain("middle",.29f);const auto editedPending=model.snapshot();const auto analysisRevision=model.revisionNumber();
    check("actual_B_facts_attached_without_second_music_undo",model.completeMaterialSourceAnalysis(result.ticket,sourceFacts(57,2),error)&&model.revisionNumber()==analysisRevision);
    auto measured=model.snapshot();check("new_source_fact_is_A3_target_is_not_retuned_to_A3",near(noteOf(measured,"middle").sourceMidiCenter,57)&&noteOf(measured,"middle").sourcePitchMeasured&&near(noteOf(measured,"middle").midiNote,66)&&!ownerOf(measured,"middle").muted&&!ownerOf(measured,"middle").materialSourceAnalysisPending);
    compare(editedPending,measured,"every_1ms_current_user_edits_survive_analysis");
    const auto measuredFile=output.getChildFile("source-measured.hachi");ProjectModel measuredReload;
    check("measured_binding_roundtrip",model.save(measuredFile,error)&&measuredReload.load(measuredFile,error)&&near(noteOf(measuredReload.snapshot(),"middle").sourceMidiCenter,57)&&measuredReload.pendingMaterialSourceAnalyses().empty());
    model.resetDocument(measured);const auto noopFingerprint=model.contentFingerprint();const auto noopRevision=model.revisionNumber();
    auto noop=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","alt",error);
    check("same_binding_alias_is_noop",!noop.rejected&&!noop.changed&&model.contentFingerprint()==noopFingerprint&&model.revisionNumber()==noopRevision&&!model.canUndo());
    auto toC=model.replaceNoteMaterialSource("middle",cFile,{c},"C","pron","same",error);const auto cPending=model.snapshot();const auto cFingerprint=model.contentFingerprint();
    check("duplicate_alias_can_choose_different_recording",!toC.rejected&&ownerOf(cPending,"middle").sourceFile==cFile&&noteOf(cPending,"middle").materialAudioId=="C");
    check("stale_B_callback_never_restores_B_or_old_score",!model.completeMaterialSourceAnalysis(result.ticket,sourceFacts(57,2),error)&&model.contentFingerprint()==cFingerprint&&ownerOf(model.snapshot(),"middle").sourceFile==cFile);
    check("C_gets_only_actual_C_facts",model.completeMaterialSourceAnalysis(toC.ticket,sourceFacts(69,2),error)&&near(noteOf(model.snapshot(),"middle").sourceMidiCenter,69));
    compare(cPending,model.snapshot(),"every_1ms_B_to_C_current_score_unchanged");
    // Cache stale B facts across undo: redo must be able to finish, not remain
    // permanently muted because an old path+mtime 'attempted' set suppressed it.
    model.resetDocument(fixture);auto cachedB=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","same",error);
    check("undo_before_source_analysis",model.undo()&&ownerOf(model.snapshot(),"middle").sourceFile==aFile);
    check("late_undone_B_only_warms_cache",!model.completeMaterialSourceAnalysis(cachedB.ticket,sourceFacts(57,2),error)&&ownerOf(model.snapshot(),"middle").sourceFile==aFile);
    check("redo_can_reuse_immutable_B_analysis",model.redo()&&model.completeMaterialSourceAnalysis(cachedB.ticket,{},error)&&near(noteOf(model.snapshot(),"middle").sourceMidiCenter,57)&&!ownerOf(model.snapshot(),"middle").muted);
    // Additional aliases are not primary names and custom project labels are
    // not source selections. Neither is silently reset by a later publication.
    model.resetDocument(measured);auto renamed=b;renamed.name="renamed-primary";
    auto aliasUpdate=model.applySourceSettingsWithReport(bFile,{renamed});
    check("valid_selected_additional_alias_survives_primary_rename",!aliasUpdate.rejected&&noteOf(model.snapshot(),"middle").label=="alt"&&noteOf(model.snapshot(),"middle").materialBoundAlias=="alt");
    renamed.additionalAliases.removeString("alt");aliasUpdate=model.applySourceSettingsWithReport(bFile,{renamed});
    check("removed_selected_alias_keeps_snapshot_with_diagnostic",!aliasUpdate.rejected&&aliasUpdate.diagnostics.joinIntoString(" ").contains("alias")&&noteOf(model.snapshot(),"middle").label=="alt"&&!noteOf(model.snapshot(),"middle").materialReferenceUnresolved);
    model.setNoteLabel("middle","project-custom");const auto labelSnapshot=model.snapshot();renamed.name="renamed-again";model.applySourceSettings(bFile,{renamed});
    check("free_bound_display_label_is_not_clobbered",noteOf(model.snapshot(),"middle").label=="project-custom"&&!noteOf(model.snapshot(),"middle").materialLabelFollowsAlias&&noteOf(model.snapshot(),"middle").nativeRole==noteOf(labelSnapshot,"middle").nativeRole);
    model.resetDocument(measured);auto primary=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","same",error);auto p=b;p.name="new-primary";
    model.applySourceSettings(bFile,{p});check("explicit_primary_selection_follows_primary_rename",primary.changed&&noteOf(model.snapshot(),"middle").label=="new-primary"&&noteOf(model.snapshot(),"middle").materialBoundAlias=="new-primary");
    // A source-less composed C4 is a score, NOT an unmeasured acoustic C4.
    ProjectData blank;TrackData blankTrack;blankTrack.id="score";blankTrack.pitchAlgorithm=PitchAlgorithm::world;
    ClipData empty;empty.id="empty";empty.durationSeconds=empty.sourceDurationSeconds=.5;NoteData score;score.id="score-C4";score.midiNote=score.sourceMidiCenter=60;score.durationSeconds=.5;score.contour={{0,0,0,true},{.5,0,0,true}};empty.notes={score};blankTrack.clips={empty};blank.tracks={blankTrack};
    auto half=b;half.regionStartSeconds=half.alignmentSeconds=0;half.regionEndSeconds=.5;
    model.resetDocument(blank);auto midiBinding=model.replaceNoteMaterialSource("score-C4",bFile,{half},"B","pron","same",error);
    check("source_blank_C4_does_not_adopt_new_source_pitch",!midiBinding.rejected&&model.completeMaterialSourceAnalysis(midiBinding.ticket,sourceFacts(57,2),error)&&near(noteOf(model.snapshot(),"score-C4").midiNote,60)&&near(noteOf(model.snapshot(),"score-C4").sourceMidiCenter,57));
    const auto c4=model.snapshot();const auto playable=expandedClipParts(c4.tracks[0].clips[0]);const auto request=AudioEngine::diagnosticNativeRequest(playable.front(),c4.tracks[0]);
    check("real_native_request_is_A3_to_C4_not_fake_pcm_bypass",request.sourceFile==bFile&&!request.sourceMidi.empty()&&!request.targetMidi.empty()&&near(request.sourceMidi[10],57)&&near(request.targetMidi[10],60)&&!backend::canPreserveNativeSource(request,48000,24000,24000));
    model.removeNote("score-C4");check("deleting_assigned_note_does_not_leave_playing_source_child",model.snapshot().tracks[0].clips.empty()||model.snapshot().tracks[0].clips[0].parts.empty());
    const auto changedFile=output.getChildFile("changed-during-analysis.wav");
    check("fingerprint_fixture_written",wav(changedFile,57,2));model.resetDocument(fixture);
    const auto oldFileTicket=model.replaceNoteMaterialSource("middle",changedFile,{b},"B","pron","same",error);
    const auto beforeExternalChange=model.contentFingerprint();check("external_file_change_fixture",wav(changedFile,69,2));
    check("file_fingerprint_blocks_stale_acoustic_facts",!oldFileTicket.rejected&&!model.completeMaterialSourceAnalysis(oldFileTicket.ticket,sourceFacts(57,2),error)
        &&error.contains("changed")&&model.contentFingerprint()==beforeExternalChange&&ownerOf(model.snapshot(),"middle").muted);
    const auto rejects=[&](ProjectData d,const juce::String& id,const juce::String& region,const juce::String& name){model.resetDocument(std::move(d));const auto f=model.contentFingerprint();const auto r=model.revisionNumber();auto x=model.replaceNoteMaterialSource(id,bFile,{b},"B",region,"same",error);check(name,x.rejected&&!x.changed&&model.contentFingerprint()==f&&model.revisionNumber()==r&&!model.canUndo());};
    rejects(fixture,"missing","pron","missing_note_rejected_atomically");rejects(fixture,"middle","missing","missing_region_rejected_atomically");
    auto local=fixture;local.tracks[0].clips[0].notes[1].materialLocalOverride=true;rejects(local,"middle","pron","private_material_override_requires_explicit_clear");
    auto ambiguous=fixture;ambiguous.tracks[0].clips[0].notes[0].id="middle";rejects(ambiguous,"middle","pron","ambiguous_note_id_rejected");
    auto utau=fixture;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;rejects(utau,"middle","pron","unsupported_voicebank_engine_not_silently_changed");
    for(auto algorithm:{PitchAlgorithm::world,PitchAlgorithm::llsm2,PitchAlgorithm::nsfHifigan})
    {auto supported=fixture;supported.tracks[0].pitchAlgorithm=algorithm;model.resetDocument(supported);auto x=model.replaceNoteMaterialSource("middle",bFile,{b},"B","pron","same",error);check("native_physical_capability_"+juce::String(static_cast<int>(algorithm)),!x.rejected&&model.snapshot().tracks[0].pitchAlgorithm==algorithm);}
    ProjectModel inserted;ProjectData insertion;insertion.tracks={blankTrack};insertion.tracks[0].clips.clear();inserted.resetDocument(insertion);
    const auto aliasClip=inserted.addMaterialPronunciations(bFile,{b},{"pron"},0,"score",error,"alt");
    check("insertion_can_select_additional_alias_not_only_primary",aliasClip.isNotEmpty()&&inserted.snapshot().tracks[0].clips[0].notes[0].label=="alt");
    auto kindRow=b;kindRow.pronunciationKind=PronunciationKind::transition;kindRow.simplifiedConsonant=true;kindRow.role=NativeSegmentRole::consonant;
    ProjectModel kinds;kinds.resetDocument(insertion);
    const auto kindClip=kinds.addMaterialPronunciations(bFile,{kindRow},{"pron"},0,"score",error,"alt");
    check("whole_pronunciation_kind_not_internal_cv_role",kindClip.isNotEmpty()&&kinds.snapshot().tracks[0].clips[0].notes[0].pronunciationKind==PronunciationKind::transition
        &&kinds.snapshot().tracks[0].clips[0].notes[0].nativeRole==NativeSegmentRole::consonant&&kinds.snapshot().tracks[0].clips[0].notes[0].simplifiedConsonant);
    kindRow.pronunciationKind=PronunciationKind::breath;kindRow.simplifiedConsonant=false;kinds.applySourceSettings(bFile,{kindRow});
    const auto kindFile=output.getChildFile("pronunciation-kind.hachi");ProjectModel kindReload;
    check("kind_and_simplified_flag_propagate_and_roundtrip",kinds.save(kindFile,error)&&kindReload.load(kindFile,error)
        &&kindReload.snapshot().tracks[0].clips[0].notes[0].pronunciationKind==PronunciationKind::breath&&!kindReload.snapshot().tracks[0].clips[0].notes[0].simplifiedConsonant
        &&kindReload.snapshot().tracks[0].clips[0].notes[0].nativeRole==NativeSegmentRole::consonant);
    check("simplified_consonant_default_off",!NoteData{}.simplifiedConsonant);
    // Old files have no whole-pronunciation kind. Recover only old transition/
    // breath intent, without merging, moving or relabeling internal C/V notes.
    ProjectModel legacyBase;legacyBase.resetDocument(fixture);const auto legacyBaseFile=output.getChildFile("legacy-kind-base.hachi");
    juce::MemoryBlock legacyBytes;
    if(legacyBase.save(legacyBaseFile,error)&&legacyBaseFile.loadFileAsData(legacyBytes))
    {
        struct LegacyCase{bool transition;NativeSegmentRole role;PronunciationKind expected;};
        const std::vector<LegacyCase> cases={{true,NativeSegmentRole::consonant,PronunciationKind::transition},{false,NativeSegmentRole::breath,PronunciationKind::breath},
            {false,NativeSegmentRole::transition,PronunciationKind::transition},{false,NativeSegmentRole::consonant,PronunciationKind::normal},{false,NativeSegmentRole::vowel,PronunciationKind::normal}};
        for(std::size_t i=0;i<cases.size();++i)
        {
            auto tree=projectio::validatedTree(legacyBytes).createCopy();
            std::function<void(juce::ValueTree)> migrate=[&](juce::ValueTree node)
            {
                if(node.hasType("Note")&&node.getProperty("id").toString()=="middle")
                {
                    node.removeProperty("pronunciationKind",nullptr);node.removeProperty("simplifiedConsonant",nullptr);
                    node.setProperty("materialTransitionPronunciation",cases[i].transition,nullptr);node.setProperty("nativeRole",nativeSegmentRoleName(cases[i].role),nullptr);
                }
                for(auto child:node)migrate(child);
            };
            migrate(tree);juce::MemoryOutputStream encoded;tree.writeToStream(encoded);
            const auto file=output.getChildFile("legacy-kind-"+juce::String(static_cast<int>(i))+".hachi");ProjectModel old;
            const auto loaded=projectio::save(file,encoded.getMemoryBlock(),error,false)&&old.load(file,error);
            const auto restored=loaded?noteOf(old.snapshot(),"middle"):NoteData{};
            check("legacy_kind_fallback_"+juce::String(static_cast<int>(i)),loaded&&restored.pronunciationKind==cases[i].expected&&restored.nativeRole==cases[i].role
                &&!restored.simplifiedConsonant&&restored.startSeconds==1&&restored.durationSeconds==1&&near(restored.consonantSeconds,.08)&&old.snapshot().tracks[0].clips[0].notes.size()==3);
        }
    }
    else check("legacy_kind_fixture_written",false);
    juce::MemoryBlock aa,bb,cc;check("all_recording_bytes_unchanged",aFile.loadFileAsData(aa)&&bFile.loadFileAsData(bb)&&cFile.loadFileAsData(cc)&&aa==aBytes&&bb==bBytes&&cc==cBytes);
    return done();
}
}
