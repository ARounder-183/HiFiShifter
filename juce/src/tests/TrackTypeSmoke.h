#pragma once
#include "../ProjectModel.h"
#include "../backend/UstImporter.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runTrackTypeSmoke(const juce::File& output)
{
    if(output.createDirectory().failed())return false;bool ok=true;int count=0;juce::String error;juce::StringArray warnings;juce::Array<juce::var> checks;
    const auto check=[&](const char* name,bool pass){ok&=pass;++count;std::cout<<name<<'='<<pass<<std::endl;auto row=new juce::DynamicObject();row->setProperty("name",name);row->setProperty("passed",pass);checks.add(juce::var(row));};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-7;};
    const auto readTree=[](const juce::File& file){juce::MemoryBlock data;return file.loadFileAsData(data)?juce::ValueTree::readFromData(data.getData(),data.getSize()):juce::ValueTree{};};
    const auto writeTree=[](const juce::ValueTree& tree,const juce::File& file){juce::MemoryOutputStream data;tree.writeToStream(data);return file.replaceWithData(data.getData(),data.getDataSize());};
    ProjectModel model;
    const auto h=model.addTypedTrack();const auto u=model.addTypedTrack(TrackType::utau,"U");const auto d=model.addTypedTrack(TrackType::diffSinger,"DS");
    auto data=model.snapshot();
    check("three_explicit_first_class_creation_types",data.tracks.size()==3&&data.tracks[0].id==h&&data.tracks[0].trackType==TrackType::hachi
        &&data.tracks[1].id==u&&data.tracks[1].trackType==TrackType::utau&&data.tracks[2].id==d&&data.tracks[2].trackType==TrackType::diffSinger);
    check("Hachi_creation_has_native_not_UTAU_renderer",data.tracks[0].pitchAlgorithm!=PitchAlgorithm::utau&&trackShowsAllNativeRegions(data.tracks[0]));
    check("native_default_is_NSF_and_splice_before_synthesis",data.tracks[0].pitchAlgorithm==PitchAlgorithm::nsfHifigan&&data.tracks[0].nativeNsfAudio&&data.tracks[0].renderOrder==RenderOrder::stretchSpliceThenPitch&&defaultPitchAlgorithm(output.getChildFile("missing-model"))==PitchAlgorithm::nsfHifigan);
    check("UTAU_creation_is_classic_bank_interface",data.tracks[1].pitchAlgorithm==PitchAlgorithm::utau&&data.tracks[1].utauMode==UtauMode::classic&&trackUsesVoicebankSynthesis(data.tracks[1])&&!trackIsDiffSinger(data.tracks[1]));
    check("unconfigured_DS_remains_explicit_DS_interface",data.tracks[2].voicebankDirectory==juce::File{}&&trackIsDiffSinger(data.tracks[2])&&trackUsesVoicebankSynthesis(data.tracks[2]));
    const auto beforeDefault=model.snapshot().tracks.size();const auto defaultTrack=model.addTrack("default");data=model.snapshot();
    check("ordinary_new_track_defaults_Hachi_not_previous_DS",data.tracks.size()==beforeDefault+1&&data.tracks.back().id==defaultTrack&&effectiveTrackType(data.tracks.back())==TrackType::hachi&&data.tracks.back().pitchAlgorithm!=PitchAlgorithm::utau);
    check("new_track_is_single_undo_redo",model.undo()&&model.snapshot().tracks.size()==beforeDefault&&model.redo()&&model.snapshot().tracks.back().id==defaultTrack);
    const auto accompaniment=model.addTrack("backing",false,true,true);data=model.snapshot();
    check("accompaniment_is_preserved_independently",data.tracks.back().id==accompaniment&&data.tracks.back().accompaniment&&!data.tracks.back().compose&&!data.tracks.back().referenceOnly&&!trackUsesVoicebankSynthesis(data.tracks.back()));
    const auto bank=output.getChildFile("bank");bank.createDirectory();const auto config=bank.getChildFile("dsconfig.yaml");
    check("synthetic_bank_config_fixture_created",config.replaceWithText("phonemes: []\n"));
    const auto uBefore=model.snapshot().tracks[1];model.setTrackVoicebankDirectory(u,bank);model.setTrackUtauMode(u,UtauMode::mou);data=model.snapshot();
    check("bank_file_does_not_convert_explicit_UTAU_to_DS",effectiveTrackType(data.tracks[1])==TrackType::utau&&!trackIsDiffSinger(data.tracks[1])&&data.tracks[1].pitchAlgorithm==uBefore.pitchAlgorithm);
    const auto saved=output.getChildFile("typed.hjpx");ProjectModel reopened;
    check("explicit_types_serialize_and_reopen",model.save(saved,error)&&reopened.load(saved,error)&&reopened.snapshot().tracks[0].trackType==TrackType::hachi
        &&reopened.snapshot().tracks[1].trackType==TrackType::utau&&reopened.snapshot().tracks[2].trackType==TrackType::diffSinger);
    auto invalid=readTree(saved);invalid.getChildWithName("Track").setProperty("trackType","unsupported-future-family",nullptr);const auto invalidFile=output.getChildFile("invalid-type.hjpx");
    const auto previous=reopened.contentFingerprint();check("unknown_explicit_type_is_rejected_atomically",writeTree(invalid,invalidFile)&&!reopened.load(invalidFile,error)&&reopened.contentFingerprint()==previous);
    // One ordinary native-file track and one old UTAU track. Delete only the
    // newly introduced property to exercise the real legacy loader.
    TrackData native;native.id="native";native.name="native";native.pitchAlgorithm=PitchAlgorithm::nsfHifigan;native.nativeNsfAudio=true;
    native.nsfNoiseProtection=false;native.nsfSmoothPitchTransitions=true;native.trackType=TrackType::hachi;native.voicebankDirectory=bank;
    ClipData clip;clip.id="material";clip.sourceFile=output.getChildFile("same-source.wav");clip.durationSeconds=clip.sourceDurationSeconds=1;clip.sourceTimeMap={{0,0},{.2,.2},{1,1}};
    NoteData note;note.id="native-note";note.label="a";note.durationSeconds=1;note.midiNote=64;note.sourceMidiCenter=57;note.pitchControlPoints={{-.01,63},{.4,65}};
    note.amplitudeEnvelope={{0,-3},{.8,-1}};note.gain=.7f;note.vibratoEnabled=true;note.vibratoPhasePercent=31;note.pitchSupportAuthored=true;clip.notes={note};native.clips={clip};
    {juce::AudioBuffer<float> pcm(1,8000);for(int i=0;i<8000;++i)pcm.setSample(0,i,static_cast<float>(.1*std::sin(juce::MathConstants<double>::twoPi*220*i/8000)));
        auto stream=clip.sourceFile.createOutputStream();juce::WavAudioFormat wav;std::unique_ptr<juce::AudioFormatWriter> writer(stream?wav.createWriterFor(stream.release(),8000,1,24,{},0):nullptr);
        check("physical_native_source_fixture_written",writer&&writer->writeFromAudioSampleBuffer(pcm,0,8000));}
    const auto sourceHash=juce::SHA256(clip.sourceFile).toHexString();
    auto utau=native;utau.id="utau-old";utau.pitchAlgorithm=PitchAlgorithm::utau;utau.nativeNsfAudio=false;utau.utauMode=UtauMode::jie;utau.voicebankDirectory=juce::File{};utau.trackType=TrackType::utau;
    utau.utauGlobalFlags="HF2Mf100";utau.clips[0].id="score";utau.clips[0].notes[0].id="utau-note";
    ProjectData legacy;legacy.tracks={native,utau};model.resetDocument(legacy);const auto legacyFile=output.getChildFile("legacy-missing-type.hjpx");model.save(legacyFile,error);
    auto tree=readTree(legacyFile);for(auto child:tree)if(child.hasType("Track"))child.removeProperty("trackType",nullptr);
    check("legacy_fixture_missing_only_new_type_field",writeTree(tree,legacyFile));
    ProjectModel loaded;check("missing_type_legacy_load_infers_native_and_UTAU_once",loaded.load(legacyFile,error)&&loaded.snapshot().tracks[0].trackType==TrackType::hachi&&loaded.snapshot().tracks[1].trackType==TrackType::utau);
    const auto migrated=loaded.snapshot();
    check("legacy_native_audio_is_never_reinterpreted_as_UTAU",migrated.tracks[0].pitchAlgorithm==native.pitchAlgorithm&&migrated.tracks[0].nativeNsfAudio
        &&!trackUsesVoicebankSynthesis(migrated.tracks[0])&&!migrated.tracks[0].nsfNoiseProtection&&migrated.tracks[0].nsfSmoothPitchTransitions);
    const auto& n=migrated.tracks[0].clips[0].notes[0];
    check("legacy_inference_does_not_reset_music_or_source_clock",n.id==note.id&&n.label==note.label&&n.midiNote==note.midiNote&&n.gain==note.gain&&n.pitchSupportAuthored
        &&n.pitchControlPoints.size()==note.pitchControlPoints.size()&&n.pitchControlPoints[0].timeSeconds==note.pitchControlPoints[0].timeSeconds
        &&n.pitchControlPoints[1].targetMidi==note.pitchControlPoints[1].targetMidi&&n.amplitudeEnvelope.size()==note.amplitudeEnvelope.size()
        &&n.vibratoPhasePercent==note.vibratoPhasePercent&&migrated.tracks[0].clips[0].sourceTimeMap.size()==clip.sourceTimeMap.size());
    check("legacy_UTAU_algorithm_mode_and_flags_are_unchanged",migrated.tracks[1].pitchAlgorithm==PitchAlgorithm::utau&&migrated.tracks[1].utauMode==UtauMode::jie&&migrated.tracks[1].utauGlobalFlags=="HF2Mf100");
    auto ds=utau;ds.id="old-ds";ds.utauMode=UtauMode::mou;ds.voicebankDirectory=bank;ds.trackType=TrackType::diffSinger;ds.clips[0].id="ds-score";ds.clips[0].notes[0].id="ds-note";
    legacy.tracks={ds};model.resetDocument(legacy);const auto dsFile=output.getChildFile("legacy-ds.hjpx");model.save(dsFile,error);tree=readTree(dsFile);tree.getChildWithName("Track").removeProperty("trackType",nullptr);writeTree(tree,dsFile);
    ProjectModel oldDs;check("legacy_DS_is_adopted_as_explicit_DS",oldDs.load(dsFile,error)&&oldDs.snapshot().tracks[0].trackType==TrackType::diffSinger);
    const auto fsAlternative=output.getChildFile("bank-without-config");fsAlternative.createDirectory();oldDs.setTrackVoicebankDirectory("old-ds",fsAlternative);
    check("configured_file_changes_cannot_toggle_adopted_DS_identity",trackIsDiffSinger(oldDs.snapshot().tracks[0])&&effectiveTrackType(oldDs.snapshot().tracks[0])==TrackType::diffSinger);
    const auto ust=output.getChildFile("typed.ust");
    ust.replaceWithText("[#VERSION]\nUST Version1.2\n[#SETTING]\nTempo=120\nProjectName=Typed song\nMode2=True\n[#0000]\nLength=480\nNoteNum=60\nLyric=a\nFlags=HF2\nIntensity=80\nPBS=-10;0\nPBW=50\nPBY=20\n[#TRACKEND]\n");
    const auto ustHash=juce::SHA256(ust).toHexString();
    for(const auto type:{TrackType::hachi,TrackType::utau,TrackType::diffSinger})
    {
        ProjectModel song;juce::String id;warnings.clear();const auto imported=song.addUstFile(ust,error,warnings,ProjectModel::UstImportMode::addTrack,&id,0,type);
        const auto score=song.snapshot();check((juce::String("UST_target_")+trackTypeName(type)).toRawUTF8(),imported&&score.tracks.size()==1&&score.tracks[0].trackType==type
            &&score.tracks[0].clips[0].notes.size()==1&&near(score.tracks[0].clips[0].notes[0].durationSeconds,.5)&&score.tracks[0].clips[0].notes[0].label=="a"
            &&near(score.tracks[0].clips[0].notes[0].gain,.8)&&score.tracks[0].clips[0].notes[0].utauFlags=="HF2"&&score.tracks[0].clips[0].notes[0].pitchControlPoints.size()==2);
        const auto scoreFile=output.getChildFile("ust-"+trackTypeName(type)+".hjpx");ProjectModel cold;
        check((juce::String("UST_target_reopen_")+trackTypeName(type)).toRawUTF8(),song.save(scoreFile,error)&&cold.load(scoreFile,error)&&cold.snapshot().tracks[0].trackType==type);
        check((juce::String("UST_single_undo_")+trackTypeName(type)).toRawUTF8(),song.undo()&&song.snapshot().tracks.empty()&&song.redo());
    }
    ProjectModel defaultImport;check("UST_default_destination_is_Hachi",defaultImport.addUstFile(ust,error,warnings)&&defaultImport.snapshot().tracks[0].trackType==TrackType::hachi);
    const auto ustx=output.getChildFile("typed.ustx");
    ustx.replaceWithText(R"USTX(ustx_version: '0.6'
resolution: 480
name: Typed multipart song
time_signatures: [{bar_position: 0, beat_per_bar: 3, beat_unit: 4}]
tempos: [{position: 0, bpm: 120}, {position: 960, bpm: 60}]
tracks: [{track_name: Lead}, {track_name: Harmony}]
voice_parts:
- track_no: 0
  position: 480
  duration: 1200
  notes:
  - {position: 0, duration: 480, tone: 60, lyric: a}
  - {position: 960, duration: 240, tone: 62, lyric: b}
- track_no: 1
  position: 1920
  duration: 480
  notes:
  - {position: 0, duration: 480, tone: 67, lyric: c}
)USTX");
    // OpenUtau's actual format is fixed at480. The old synthetic960 fixture
    // was rejected before conversion; every tick coordinate above was halved,
    // preserving the EXACT same absolute quarters, tempo map and assertions.
    const auto parsedUstx=backend::UstImporter::read(ustx,error,warnings);
    std::cout<<"USTX_parse_error="<<error<<"|parsed="<<bool(parsedUstx)<<std::endl;
    if(parsedUstx)
    {
        std::cout<<"USTX_source_bpm="<<parsedUstx->tempo<<"|meter="<<parsedUstx->numerator<<'/'<<parsedUstx->denominator<<std::endl;
        for(const auto& tempo:parsedUstx->tempoChanges)std::cout<<"USTX_tempo_Q="<<tempo.quarterPosition<<"|bpm="<<tempo.bpm<<std::endl;
        for(const auto& note:parsedUstx->notes)std::cout<<"USTX_note_Q="<<note.positionQuarterNotes.value_or(-1)<<"|duration_Q="<<note.durationQuarterNotes.value_or(-1)<<std::endl;
    }
    for(const auto type:{TrackType::hachi,TrackType::utau,TrackType::diffSinger})
    {
        ProjectModel song;error.clear();const auto imported=song.addUstFile(ustx,error,warnings,ProjectModel::UstImportMode::addTrack,nullptr,0,type);const auto score=song.snapshot();
        std::cout<<"USTX_import_type="<<trackTypeName(type)<<"|ok="<<imported<<"|error="<<error<<"|tracks="<<score.tracks.size()<<std::endl;
        check((juce::String("USTX_shared_converter_target_")+trackTypeName(type)).toRawUTF8(),imported&&score.tracks.size()==2&&score.tracks[0].trackType==type&&score.tracks[1].trackType==type
            &&score.numerator==3&&score.denominator==4&&score.tracks[0].clips[0].notes.size()==2&&score.tracks[1].clips[0].notes.size()==1
            &&near(score.tracks[0].clips[0].notes[0].startSeconds,.5)&&near(score.tracks[0].clips[0].notes[0].durationSeconds,.5)
            &&near(score.tracks[0].clips[0].notes[1].startSeconds,2)&&near(score.tracks[0].clips[0].notes[1].durationSeconds,.5)
            &&near(score.tracks[1].clips[0].notes[0].startSeconds,3)&&near(score.tracks[1].clips[0].notes[0].durationSeconds,1));
        const auto path=output.getChildFile("ustx-"+trackTypeName(type)+".hjpx");ProjectModel cold;
        check((juce::String("USTX_source_raw_and_type_reopen_")+trackTypeName(type)).toRawUTF8(),imported&&song.save(path,error)&&cold.load(path,error)&&cold.snapshot().tracks.size()==2
            &&cold.snapshot().tracks[1].trackType==type&&cold.snapshot().tracks[0].ustSourceDocument==ustx.loadFileAsString());
        check((juce::String("USTX_multi_track_one_undo_")+trackTypeName(type)).toRawUTF8(),imported&&song.undo()&&song.snapshot().tracks.empty()&&song.redo());
    }
    const auto beforeReplace=defaultImport.contentFingerprint();
    check("replace_score_import_is_one_undo_not_history_destruction",defaultImport.addUstFile(ustx,error,warnings,ProjectModel::UstImportMode::replaceProject)
        &&defaultImport.snapshot().tracks.size()==2&&defaultImport.undo()&&defaultImport.contentFingerprint()==beforeReplace);
    check("typed_creation_and_import_never_rewrite_source_media",juce::SHA256(clip.sourceFile).toHexString()==sourceHash&&juce::SHA256(ust).toHexString()==ustHash);
    auto report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",count);report->setProperty("results",juce::var(checks));
    const auto written=output.getChildFile("verification.json").replaceWithText(juce::JSON::toString(juce::var(report)));
    std::cout<<"track_types_ok="<<(ok&&written)<<"|checks="<<count<<std::endl;return ok&&written;
}
}
