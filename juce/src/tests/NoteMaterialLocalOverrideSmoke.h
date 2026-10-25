#pragma once
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "../OttoMaterialIO.h"
#include "../NativeNoteTiming.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runNoteMaterialLocalOverrideSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("note-local-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;int count=0;juce::Array<juce::var> checks;juce::String error;
    const auto check=[&](const char* name,bool pass){ok&=pass;++count;std::cout<<name<<'='<<pass<<std::endl;
        auto item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("passed",pass);checks.add(juce::var(item));};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-7;};
    const auto finish=[&]{auto report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",count);report->setProperty("results",juce::var(checks));
        report->setProperty("basis","Generated synthetic source. Actual source-byte/read-clock checks, not a neural acoustic-render acceptance claim.");
        const auto saved=root.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
        std::cout<<"note_material_local_override_ok="<<(ok&&saved)<<"|checks="<<count<<"|fixture="<<root.getFullPathName()<<std::endl;return ok&&saved;};
    const auto audio=root.getChildFile("source.wav");constexpr int rate=48000;
    {
        juce::AudioBuffer<float> pcm(1,rate*2);for(int i=0;i<pcm.getNumSamples();++i)
            pcm.setSample(0,i,static_cast<float>(.2*std::sin(juce::MathConstants<double>::twoPi*220*i/rate)+.035*std::sin(juce::MathConstants<double>::twoPi*440*i/rate)));
        auto stream=audio.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(stream?wav.createWriterFor(stream.release(),rate,1,24,{},0):nullptr);
        check("synthetic_source_written",writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples()));
    }
    if(!ok)return finish();
    std::vector<SampleRegionSetting> rows;
    for(int i=0;i<2;++i){SampleRegionSetting row;row.name=i?"b":"a";row.regionStartSeconds=i;row.regionEndSeconds=i+1;
        row.alignmentSeconds=i+.12;row.fixedDurationSeconds=.2;row.overlapSeconds=-.03;row.provenance="user";row.confidence=.9f;rows.push_back(row);}
    check("source_annotation_fixture_saved",SampleSettings::save(audio,rows,error));
    rows=SampleSettings::loadOrDerive(audio,ProjectData{},&error,false);
    check("source_stable_audio_region_identities_resolved",rows.size()==2&&rows[0].materialAudioId.isNotEmpty()&&rows[0].materialRegionId.isNotEmpty());
    if(rows.size()!=2)return finish();
    juce::MemoryBlock sourceBytes,manifestBytes;audio.loadFileAsData(sourceBytes);const auto manifest=SampleSettings::annotationFor(audio);manifest.loadFileAsData(manifestBytes);
    juce::AudioBuffer<float> decoded(1,rate*2);
    {juce::WavAudioFormat wav;auto in=audio.createInputStream();std::unique_ptr<juce::AudioFormatReader> reader(in?wav.createReaderFor(in.release(),true):nullptr);
        check("physical_source_PCM_decoded",reader&&reader->read(&decoded,0,rate*2,0,true,true));}
    const auto makeNote=[&](int i)
    {
        const auto& row=rows[static_cast<std::size_t>(i)];NoteData n;n.id=i?"B":"A";n.label=i?"custom music B":"custom music A";
        n.materialLabelFollowsAlias=false;n.materialAliasFollowsPrimary=true;n.materialBoundAlias=row.name;
        n.startSeconds=i;n.durationSeconds=1;n.midiNote=64+i;n.sourceMidiCenter=57;n.sourcePitchMeasured=false;n.consonantSeconds=.2;
        n.nativeRole=row.role;n.nativeProvenance=row.provenance;n.nativeConfidence=row.confidence;
        n.materialAudioId=row.materialAudioId;n.materialRegionId=row.materialRegionId;n.materialSourceStartSeconds=n.nativeSourceStartSeconds=i;
        n.materialSourceEndSeconds=n.nativeSourceEndSeconds=i+1;n.nativeSegments=SampleSettings::nativeSegmentsFor(row);
        n.utauPreutteranceOverrideEnabled=true;n.utauPreutteranceSeconds=row.alignmentSeconds-row.regionStartSeconds;
        n.utauOverlapOverrideEnabled=true;n.utauOverlapSeconds=row.overlapSeconds;
        n.contour={{0,0,0,true},{1,0,0,true}};n.pitchControlPoints={{0,n.midiNote},{.8,n.midiNote+.35f}};n.pitchSupportAuthored=true;
        n.amplitudeEnvelope={{0,-3},{.7,-1}};n.gain=.65f;n.formantSemitones=1.25f;n.breath=.2f;n.drift=.8f;n.modulation=.7f;
        n.vibratoEnabled=true;n.vibratoDepthCents=23;n.vibratoPhasePercent=27;n.vibratoTimeOffsetSeconds=.03;n.vibratoReferenceDurationSeconds=1.1;return n;
    };
    TrackData track;track.id="native";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::nsfHifigan;track.nativeNsfAudio=true;
    ClipData clip;clip.id="owner";clip.sourceFile=audio;clip.durationSeconds=clip.sourceDurationSeconds=2;clip.sourceTimeMap={{0,0},{2,2}};clip.notes={makeNote(0),makeNote(1)};track.clips={clip};
    ProjectData initial;initial.tracks={track};ProjectModel model;model.resetDocument(initial);
    const auto noteHash=[](const NoteData& note){ProjectData p;TrackData t;t.id="t";t.compose=true;t.pitchAlgorithm=PitchAlgorithm::world;ClipData c;c.id="c";c.durationSeconds=2;c.notes={note};t.clips={c};p.tracks={t};ProjectModel m;m.resetDocument(p);return m.contentFingerprint();};
    const auto musicSame=[&](const NoteData& before,const NoteData& after)
    {
        auto stripped=after;
        stripped.nativeRole=before.nativeRole;stripped.nativeProvenance=before.nativeProvenance;stripped.nativeConfidence=before.nativeConfidence;
        stripped.nativeSourceStartSeconds=before.nativeSourceStartSeconds;stripped.nativeSourceEndSeconds=before.nativeSourceEndSeconds;stripped.nativeSegments=before.nativeSegments;
        stripped.materialBoundAlias=before.materialBoundAlias;stripped.materialSourceStartSeconds=before.materialSourceStartSeconds;stripped.materialSourceEndSeconds=before.materialSourceEndSeconds;
        stripped.materialLocalOverride=before.materialLocalOverride;stripped.materialReferenceUnresolved=before.materialReferenceUnresolved;
        stripped.materialAdditionalAliases=before.materialAdditionalAliases;stripped.materialConnectedNextRegionId=before.materialConnectedNextRegionId;
        stripped.pronunciationKind=before.pronunciationKind;stripped.simplifiedConsonant=before.simplifiedConsonant;stripped.vowelSplitsExplicit=before.vowelSplitsExplicit;
        stripped.nativeUnpitched=before.nativeUnpitched;stripped.consonantSeconds=before.consonantSeconds;
        stripped.utauPreutteranceOverrideEnabled=before.utauPreutteranceOverrideEnabled;stripped.utauPreutteranceSeconds=before.utauPreutteranceSeconds;
        stripped.utauOverlapOverrideEnabled=before.utauOverlapOverrideEnabled;stripped.utauOverlapSeconds=before.utauOverlapSeconds;
        return noteHash(before)==noteHash(stripped);
    };
    const auto weighted=[](SampleRegionSetting row)
    {
        row.name+=" local";row.pronunciationKind=PronunciationKind::transition;row.simplifiedConsonant=true;row.vowelSplitsExplicit=true;row.segments.clear();
        NativeSegment c;c.id="C";c.alias="C";c.role=NativeSegmentRole::consonant;c.sourceStartSeconds=row.regionStartSeconds;c.sourceEndSeconds=row.regionStartSeconds+.2;
        c.alignmentSeconds=row.alignmentSeconds;c.overlapSeconds=row.overlapSeconds;c.stretchable=false;
        auto a=c;a.id="V-an";a.alias="an";a.role=NativeSegmentRole::vowel;a.sourceStartSeconds=row.regionStartSeconds+.2;a.sourceEndSeconds=row.regionStartSeconds+.6;a.stretchable=true;
        auto b=a;b.id="V-ng";b.alias="ng";b.sourceStartSeconds=row.regionStartSeconds+.6;b.sourceEndSeconds=row.regionEndSeconds;b.stretchWeight=3;row.segments={c,a,b};return row;
    };
    ProjectModel::NoteMaterialLocalBinding ticket;SampleRegionSetting local;
    const auto before=model.contentFingerprint();const auto beforeRevision=model.revisionNumber();
    check("getter_resolves_exact_native_binding_without_mutation",model.getNoteMaterialLocalBinding("A",ticket,local,error)&&ticket.bindingResolved&&ticket.nativeFullRow
        &&ticket.sourceFile==audio&&ticket.audioId==rows[0].materialAudioId&&ticket.regionId==rows[0].materialRegionId
        &&model.contentFingerprint()==before&&model.revisionNumber()==beforeRevision&&!model.canUndo());
    auto edit=weighted(local);std::uint64_t newRevision=0;
    check("native_full_row_apply_commits_one_project_transaction",model.applyNoteMaterialLocalRow(ticket,edit,newRevision,error)&&newRevision==model.revisionNumber()&&newRevision>beforeRevision);
    const auto applied=model.snapshot();const auto& a=applied.tracks[0].clips[0].notes[0];
    check("full_annotation_kind_simple_signed_overlap_named_V_weights_persist",a.materialLocalOverride&&a.pronunciationKind==PronunciationKind::transition&&a.simplifiedConsonant&&a.vowelSplitsExplicit
        &&near(a.utauOverlapSeconds,-.03)&&a.nativeSegments.size()==3&&a.nativeSegments[1].id=="V-an"&&a.nativeSegments[2].alias=="ng"&&near(a.nativeSegments[2].stretchWeight,3));
    check("all_musical_controls_alias_intent_and_author_support_preserved",musicSame(initial.tracks[0].clips[0].notes[0],a));
    check("unselected_note_is_byte_equivalent_and_engine_unchanged",noteHash(applied.tracks[0].clips[0].notes[1])==noteHash(initial.tracks[0].clips[0].notes[1])&&applied.tracks[0].pitchAlgorithm==track.pitchAlgorithm);
    const auto clock=nativeClipClock(applied.tracks[0].clips[0]);
    check("manual_V_ratios_change_actual_source_clock_only_in_selected_window",near(nativeTargetTimeAt(clock,.6),.4)&&near(nativeSourceTimeAt(clock,1.5),1.5));
    bool fixed=true;
    for(int i=25;i<195;++i){const auto t=i*.001;const auto s=nativeSourceTimeAt(clock,t)+applied.tracks[0].clips[0].sourceOffsetSeconds;
        fixed&=near(s,t)&&decoded.getSample(0,static_cast<int>(std::lround(s*rate)))==decoded.getSample(0,static_cast<int>(std::lround(t*rate)));}
    check("protected_C_actual_decoded_PCM_read_positions_are_identical",fixed);
    juce::MemoryBlock sourceAfter,manifestAfter;audio.loadFileAsData(sourceAfter);manifest.loadFileAsData(manifestAfter);
    check("local_apply_writes_neither_source_PCM_nor_manifest",sourceAfter==sourceBytes&&manifestAfter==manifestBytes);
    const auto appliedFingerprint=model.contentFingerprint();
    check("one_undo_and_redo_restore_exact_annotation_transaction",model.undo()&&model.contentFingerprint()==before&&!model.canUndo()&&model.redo()&&model.contentFingerprint()==appliedFingerprint);
    const auto projectFile=root.getChildFile("local.hjpx");ProjectModel reopened;
    check("local_full_metadata_survives_project_reopen",model.save(projectFile,error)&&reopened.load(projectFile,error)
        &&reopened.snapshot().tracks[0].clips[0].notes[0].materialLocalOverride&&reopened.snapshot().tracks[0].clips[0].notes[0].nativeSegments.size()==3
        &&musicSame(initial.tracks[0].clips[0].notes[0],reopened.snapshot().tracks[0].clips[0].notes[0]));
    auto latest=rows;latest[0].name="a latest";latest[0].pronunciationKind=PronunciationKind::breath;latest[0].alignmentSeconds=.15;latest[0].fixedDurationSeconds=.25;
    check("latest_source_baseline_fixture_published",SampleSettings::save(audio,latest,error));
    const auto skipped=model.applySourceSettingsWithReport(audio,latest);
    check("normal_source_update_still_skips_local_override",!skipped.rejected&&skipped.skippedLocalOverrides==1&&model.snapshot().tracks[0].clips[0].notes[0].materialLocalOverride);
    ProjectModel::NoteMaterialLocalBinding restoreTicket;SampleRegionSetting current,restored;
    check("local_snapshot_getter_roundtrips_real_annotation",model.getNoteMaterialLocalBinding("A",restoreTicket,current,error)&&current.vowelSplitsExplicit&&current.segments.size()==3&&near(current.fixedDurationSeconds,.2));
    manifestAfter.reset();manifest.loadFileAsData(manifestAfter);const auto restoreBefore=model.contentFingerprint();
    check("restore_reads_latest_source_row_not_open_draft",model.restoreNoteMaterialLocalRow(restoreTicket,restored,newRevision,error)
        &&restored.pronunciationKind==PronunciationKind::breath&&near(restored.alignmentSeconds,.15)&&near(restored.fixedDurationSeconds,.25));
    const auto restoredState=model.snapshot();
    check("restore_clears_only_override_and_keeps_music_and_other_note",!restoredState.tracks[0].clips[0].notes[0].materialLocalOverride
        &&musicSame(initial.tracks[0].clips[0].notes[0],restoredState.tracks[0].clips[0].notes[0])
        &&noteHash(restoredState.tracks[0].clips[0].notes[1])==noteHash(initial.tracks[0].clips[0].notes[1]));
    juce::MemoryBlock manifestRestored;manifest.loadFileAsData(manifestRestored);
    const auto restoreAfter=model.contentFingerprint();const bool restoreUndo=model.undo();
    const auto undoneRestore=model.contentFingerprint();const bool restoreRedo=model.redo();
    check("restore_is_project_only_one_undo",manifestRestored==manifestAfter&&restoreUndo&&undoneRestore==restoreBefore
        &&restoreRedo&&model.contentFingerprint()==restoreAfter);
    std::cout<<"restore_manifest_before_bytes="<<manifestAfter.getSize()<<"|after_bytes="<<manifestRestored.getSize()
        <<"|undo="<<restoreUndo<<"|redo="<<restoreRedo<<std::endl;
    const auto reject=[&](const char* label,const ProjectModel::NoteMaterialLocalBinding& binding,const SampleRegionSetting& row)
    {const auto fp=model.contentFingerprint();const auto rev=model.revisionNumber();std::uint64_t out=999;
        check(label,!model.applyNoteMaterialLocalRow(binding,row,out,error)&&model.contentFingerprint()==fp&&model.revisionNumber()==rev);};
    model.resetDocument(initial);model.getNoteMaterialLocalBinding("A",ticket,local,error);model.setNotesMidi({"A"},70);
    reject("stale_revision_rejects_without_changing_new_music",ticket,local);
    model.resetDocument(initial);model.getNoteMaterialLocalBinding("A",ticket,local,error);
    auto wrong=ticket;wrong.regionId="other";reject("wrong_stable_region_rejects_atomically",wrong,local);
    auto musical=local;musical.melodyneAmplitude+=.1;reject("annotation_cannot_silently_reset_gain_or_music",ticket,musical);
    auto conflict=local;conflict.regionEndSeconds=1.05;reject("shared_neighbour_boundary_conflict_rejects_atomically",ticket,conflict);
    audio.appendText("external mutation");reject("source_SHA_change_rejects_atomically",ticket,local);audio.replaceWithData(sourceBytes.getData(),sourceBytes.getSize());
    auto cropped=initial;cropped.tracks[0].clips[0].notes[0].startSeconds=.2;cropped.tracks[0].clips[0].notes[0].durationSeconds=.6;
    model.resetDocument(cropped);model.getNoteMaterialLocalBinding("A",ticket,local,error);
    check("cropped_owner_capability_is_explicit_not_source_replacement",ticket.bindingResolved&&!ticket.nativeFullRow&&ticket.diagnostic.isNotEmpty());
    ticket.nativeFullRow=true;reject("forged_crop_capability_still_rejects_in_transaction",ticket,local);
    auto shortData=initial;auto& sc=shortData.tracks[0].clips[0];sc.notes.resize(1);sc.durationSeconds=.3;sc.sourceDurationSeconds=1;sc.sourceTimeMap={{0,0},{.2,.2},{.3,1}};sc.notes[0].durationSeconds=.3;
    model.resetDocument(shortData);model.getNoteMaterialLocalBinding("A",ticket,local,error);local.fixedDurationSeconds=.4;
    reject("impossible_protected_C_budget_has_no_mutation",ticket,local);
    auto custom=initial;custom.tracks[0].clips[0].sourceTimeMap={{0,0},{.3,.2},{1,1},{2,2}};custom.tracks[0].clips[0].notes[0].consonantSeconds=.3;
    model.resetDocument(custom);model.getNoteMaterialLocalBinding("A",ticket,local,error);reject("nonunit_existing_C_cannot_be_stretched_by_local_V_edit",ticket,weighted(local));
    local.pronunciationKind=PronunciationKind::transition;
    check("metadata_only_local_edit_preserves_existing_custom_clock",model.applyNoteMaterialLocalRow(ticket,local,newRevision,error)
        &&near(nativeTargetTimeAt(nativeClipClock(model.snapshot().tracks[0].clips[0]),.2),.3));
    auto duplicate=initial;duplicate.tracks[0].clips[0].notes[1].id="A";model.resetDocument(duplicate);
    check("ambiguous_note_ID_is_never_nearest_matched",!model.getNoteMaterialLocalBinding("A",ticket,local,error));
    model.resetDocument(applied);model.getNoteMaterialLocalBinding("A",ticket,local,error);SampleSettings::save(audio,{rows[1]},error);
    const auto missingBefore=model.contentFingerprint();restored.name="untouched";
    check("deleted_latest_source_restore_retains_local_data_and_output",!model.restoreNoteMaterialLocalRow(ticket,restored,newRevision,error)
        &&model.contentFingerprint()==missingBefore&&restored.name=="untouched"&&!model.canUndo());
    // Two independently selectable annotations of the SAME physical WAV.
    // Region IDs deliberately match across variants: audio identity is required.
    check("variant_A_source_fixture_restored",OttoMaterialIO::saveAudio(manifest,audio,rows,error,{},rows[0].materialAudioId));
    const auto variantBId=juce::Uuid().toString().removeCharacters("-");auto variantB=rows;
    for(auto& row:variantB)
    {
        row.materialAudioId=variantBId;row.name+=" variant B";row.pronunciationKind=PronunciationKind::breath;
        row.alignmentSeconds=row.regionStartSeconds+.08;row.fixedDurationSeconds=.15;
        check("variant_B_source_timing_normalized",applyPronunciationTiming(row,pronunciationTimingFor(row),error));
    }
    check("same_PCM_second_annotation_variant_published",OttoMaterialIO::saveAudio(manifest,audio,variantB,error,{},variantBId));
    SampleRegionSetting sentinel;sentinel.name="untouched";std::vector<SampleRegionSetting> ambiguous{sentinel};
    check("path_only_read_rejects_multiple_variants_without_choosing_first",!OttoMaterialIO::loadAudio(manifest,audio,ambiguous,error)
        &&ambiguous.size()==1&&ambiguous[0].name=="untouched");
    model.resetDocument(applied);model.getNoteMaterialLocalBinding("A",ticket,local,error);
    check("restore_selects_bound_variant_A_despite_shared_physical_path",model.restoreNoteMaterialLocalRow(ticket,restored,newRevision,error)
        &&restored.materialAudioId==rows[0].materialAudioId&&restored.materialRegionId==rows[0].materialRegionId
        &&near(restored.alignmentSeconds,.12)&&near(restored.fixedDurationSeconds,.2)&&restored.pronunciationKind==PronunciationKind::normal);
    auto boundB=applied;for(auto& note:boundB.tracks[0].clips[0].notes){note.materialAudioId=variantBId;note.materialLocalOverride=true;}
    model.resetDocument(boundB);model.getNoteMaterialLocalBinding("A",ticket,local,error);
    check("restore_selects_variant_B_same_region_ID_not_variant_A",model.restoreNoteMaterialLocalRow(ticket,restored,newRevision,error)
        &&restored.materialAudioId==variantBId&&restored.materialRegionId==rows[0].materialRegionId
        &&near(restored.alignmentSeconds,.08)&&near(restored.fixedDurationSeconds,.15)&&restored.pronunciationKind==PronunciationKind::breath
        &&musicSame(boundB.tracks[0].clips[0].notes[0],model.snapshot().tracks[0].clips[0].notes[0]));
    auto bank=initial;bank.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;bank.tracks[0].nativeNsfAudio=false;bank.tracks[0].utauMode=UtauMode::classic;
    bank.tracks[0].clips[0].notes[0].materialAudioId.clear();bank.tracks[0].clips[0].notes[0].materialRegionId.clear();model.resetDocument(bank);
    check("unbound_classic_bank_returns_capability_diagnostic_not_alias_guess",model.getNoteMaterialLocalBinding("A",ticket,local,error)
        &&ticket.utauScalarTiming&&!ticket.bindingResolved&&!ticket.nativeFullRow&&ticket.diagnostic.isNotEmpty());
    SampleRegionSetting scalar;scalar.regionStartSeconds=.1;scalar.regionEndSeconds=1.1;scalar.alignmentSeconds=.22;scalar.fixedDurationSeconds=.2;scalar.overlapSeconds=-.03;
    backend::UtauOtoOverride oto;
    check("classic_scalar_adapter_reuses_OTO_conversion",ProjectModel::makeClassicNoteMaterialOto(scalar,oto,error)&&oto.enabled
        &&near(oto.offsetMs,100)&&near(oto.consonantMs,200)&&near(oto.preutteranceMs,120)&&near(oto.overlapMs,-30)&&near(oto.cutoffMs,-1000));
    model.setNoteUtauOto("A",oto);
    check("classic_own_OTO_remains_independent_of_material_local_flag",model.snapshot().tracks[0].clips[0].notes[0].utauOto.enabled&&!model.snapshot().tracks[0].clips[0].notes[0].materialLocalOverride);
    scalar.vowelSplitsExplicit=true;const auto oldOffset=oto.offsetMs;
    check("classic_named_weighted_V_is_rejected_without_silent_loss",!ProjectModel::makeClassicNoteMaterialOto(scalar,oto,error)&&oto.offsetMs==oldOffset);
    scalar.vowelSplitsExplicit=false;scalar.pronunciationKind=PronunciationKind::breath;
    check("classic_unsupported_kind_is_rejected_not_pretend_stored",!ProjectModel::makeClassicNoteMaterialOto(scalar,oto,error));
    // Session validates immutable resolved source/entry provenance; the model
    // owns the same-lock expected-revision/identity check and scalar mutation.
    model.resetDocument(bank);const auto classicBefore=model.snapshot();const auto classicBeforeFp=model.contentFingerprint();
    const auto classicRev=model.revisionNumber();auto checkedOto=oto;checkedOto.overlapMs=-42;
    const auto classicSourceSha=juce::SHA256(audio).toHexString(),classicManifestSha=juce::SHA256(manifest).toHexString();
    check("checked_classic_OTO_commits_exact_revision_one_undo",model.setNoteUtauOtoChecked("A",classicRev,checkedOto,newRevision,error)
        &&newRevision==classicRev+1&&model.snapshot().tracks[0].clips[0].notes[0].utauOto==checkedOto);
    const auto classicApplied=model.snapshot();auto musicOnly=classicApplied.tracks[0].clips[0].notes[0];musicOnly.utauOto=classicBefore.tracks[0].clips[0].notes[0].utauOto;
    check("checked_classic_changes_no_music_clock_or_native_binding_flag",noteHash(musicOnly)==noteHash(classicBefore.tracks[0].clips[0].notes[0])
        &&!classicApplied.tracks[0].clips[0].notes[0].materialLocalOverride
        &&near(nativeSourceTimeAt(nativeClipClock(classicApplied.tracks[0].clips[0]),.75),.75)
        &&noteHash(classicApplied.tracks[0].clips[0].notes[1])==noteHash(classicBefore.tracks[0].clips[0].notes[1]));
    const auto classicAppliedFp=model.contentFingerprint();const auto noOpRev=model.revisionNumber();
    check("checked_classic_noop_honestly_retains_revision",model.setNoteUtauOtoChecked("A",noOpRev,checkedOto,newRevision,error)
        &&newRevision==noOpRev&&model.revisionNumber()==noOpRev&&model.contentFingerprint()==classicAppliedFp);
    check("checked_classic_one_undo_redo_preserves_exact_snapshots",model.undo()&&model.contentFingerprint()==classicBeforeFp&&!model.canUndo()
        &&model.redo()&&model.contentFingerprint()==classicAppliedFp);
    check("checked_classic_restore_uses_same_atomic_guard",model.setNoteUtauOtoChecked("A",model.revisionNumber(),backend::UtauOtoOverride{},newRevision,error)
        &&model.contentFingerprint()==classicBeforeFp&&model.undo()&&model.contentFingerprint()==classicAppliedFp&&model.redo());
    const auto rejectChecked=[&](const char* label,const juce::String& id,std::uint64_t expected,const backend::UtauOtoOverride& value)
    {
        const auto fp=model.contentFingerprint();const auto rev=model.revisionNumber();std::uint64_t resulting=999;
        check(label,!model.setNoteUtauOtoChecked(id,expected,value,resulting,error)&&error.isNotEmpty()
            &&model.contentFingerprint()==fp&&model.revisionNumber()==rev);
    };
    rejectChecked("checked_classic_stale_revision_rejects_without_mutation","A",model.revisionNumber()-1,checkedOto);
    rejectChecked("checked_classic_missing_note_rejects_without_mutation","missing",model.revisionNumber(),checkedOto);
    rejectChecked("checked_classic_empty_note_ID_rejects_without_mutation",{},model.revisionNumber(),checkedOto);
    auto duplicatedBank=bank;duplicatedBank.tracks[0].clips[0].notes[1].id="A";model.resetDocument(duplicatedBank);
    rejectChecked("checked_classic_duplicate_note_ID_rejects_without_mutation","A",model.revisionNumber(),checkedOto);
    for(const auto family:{TrackType::hachi,TrackType::diffSinger})
    {
        auto unsupported=bank;unsupported.tracks[0].trackType=family;
        unsupported.tracks[0].pitchAlgorithm=family==TrackType::hachi?PitchAlgorithm::nsfHifigan:PitchAlgorithm::utau;
        unsupported.tracks[0].nativeNsfAudio=family==TrackType::hachi;unsupported.tracks[0].utauMode=family==TrackType::diffSinger?UtauMode::mou:UtauMode::classic;
        model.resetDocument(unsupported);
        rejectChecked(family==TrackType::hachi?"checked_classic_rejects_native_family":"checked_classic_rejects_DS_family","A",model.revisionNumber(),checkedOto);
    }
    auto regional=bank;regional.tracks[0].utauMode=UtauMode::jie;model.resetDocument(regional);
    rejectChecked("checked_classic_rejects_regional_track","A",model.revisionNumber(),checkedOto);
    auto alreadyLocal=bank;alreadyLocal.tracks[0].clips[0].notes[0].materialLocalOverride=true;model.resetDocument(alreadyLocal);
    check("checked_classic_preserves_existing_true_material_override_too",model.setNoteUtauOtoChecked("A",model.revisionNumber(),checkedOto,newRevision,error)
        &&model.snapshot().tracks[0].clips[0].notes[0].materialLocalOverride);
    model.resetDocument(bank);auto regionalOto=checkedOto;regionalOto.hasRegions=true;
    rejectChecked("checked_classic_rejects_unrenderable_regional_payload","A",model.revisionNumber(),regionalOto);
    check("checked_classic_never_writes_audio_or_source_OTO_manifest",juce::SHA256(audio).toHexString()==classicSourceSha&&juce::SHA256(manifest).toHexString()==classicManifestSha);
    sourceAfter.reset();audio.loadFileAsData(sourceAfter);check("all_model_annotation_actions_preserve_original_source_bytes",sourceAfter==sourceBytes);
    return finish();
}
}
