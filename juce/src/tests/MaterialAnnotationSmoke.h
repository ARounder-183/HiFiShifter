#pragma once
#include "../MaterialAnnotationSession.h"
#include "../NativeSharedEnvelope.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>
namespace hachi
{
inline bool runMaterialAnnotationSmoke(const juce::File& output)
{
    const auto root = output.getChildFile("annotation-"+juce::Uuid().toString()); root.createDirectory();
    const auto audio = root.getChildFile("source.wav");
    juce::AudioBuffer<float> pcm(1,48000); pcm.clear();
    juce::WavAudioFormat format;
    auto stream = audio.createOutputStream(); if (!stream) return false;
    std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),48000,1,16,{},0));
    if (!writer || !writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples())) return false;
    writer.reset();
    juce::MemoryBlock sourceBefore; audio.loadFileAsData(sourceBefore);
    int total=0, failed=0; juce::String log,error;
    const auto check=[&](const char* name,bool yes)
    { ++total; if(!yes)++failed; log+=juce::String(name)+"="+(yes?"PASS":"FAIL")+" "+(yes?juce::String():error)+"\n"; };
    const auto manifest=root.getChildFile("otto.ini");
    auto session=MaterialAnnotationSession::open(audio,manifest,error);
    check("open_bare_audio_empty_draft",session && session->rows().empty());
    if(!session){std::cerr<<error<<std::endl;return false;}
    check("opening_does_not_publish",!manifest.exists());
    const bool created=session->createPronunciation(0,1,"source",error);
    check("explicit_full_range_pronunciation",created && session->rows().size()==1
        && std::abs(session->rows()[0].regionEndSeconds-1)<1.e-9);
    if(!created || session->rows().size()!=1)return false;
    const auto originalId=session->rows()[0].materialRegionId;
    check("exact_range_aliases_share_one_stable_pronunciation",session->createPronunciation(0,1,"source_alt",error)
        &&session->createPronunciation(0,1,"source_other",error)&&session->rows().size()==1
        &&session->rows()[0].materialRegionId==originalId
        &&session->rows()[0].additionalAliases==juce::StringArray{"source_alt","source_other"});
    check("exact_range_duplicate_alias_is_idempotent",session->createPronunciation(0,1,"source_alt",error)
        &&session->rows().size()==1&&session->rows()[0].materialRegionId==originalId
        &&session->rows()[0].additionalAliases.size()==2);
    check("cancel_one_alias_preserves_other_alias_and_identity",session->removeAlias(originalId,"source_alt",error)
        &&session->rows().size()==1&&session->rows()[0].materialRegionId==originalId
        &&session->rows()[0].additionalAliases==juce::StringArray{"source_other"});
    check("cancel_remaining_extra_alias_does_not_publish",session->removeAlias(originalId,"source_other",error)
        &&session->rows().size()==1&&session->rows()[0].materialRegionId==originalId
        &&session->rows()[0].name=="source"&&session->rows()[0].additionalAliases.isEmpty()&&!manifest.exists());
    auto row=session->rows()[0];
    row.amplitudeEnvelope={{0,-12,true},{1,0,false}};
    check("author_envelope",session->setRow(row.materialRegionId,row,error));
    check("batch_non_destructive_split",session->split({.25,.7},error)&&session->rows().size()==3);
    if(session->rows().size()==3)
    {
        const auto& a=session->rows()[0];const auto& b=session->rows()[1];const auto& c=session->rows()[2];
        check("split_stable_left_identity",a.materialRegionId==row.materialRegionId);
        check("split_preserves_continuous_chain",a.connectedNextRegionId==b.materialRegionId&&b.connectedNextRegionId==c.materialRegionId);
        bool same=true;
        for(const auto& part:session->rows()) for(int i=0;i<=20;++i)
        {
            const auto t=(part.regionEndSeconds-part.regionStartSeconds)*i/20.0;
            same &= std::abs(nativeEnvelopeDbAt(part.amplitudeEnvelope,t)
                -nativeEnvelopeDbAt(row.amplitudeEnvelope,part.regionStartSeconds+t))<1.e-4;
        }
        check("split_envelope_gain_unchanged",same);
    }
    check("batch_alias_assignment",session->rename({"ka","a","i"},error));
    check("batch_alias_one_undo",session->undo()&&session->rows().size()==3&&session->rows()[0].name==row.name);
    check("batch_alias_redo",session->redo()&&session->rows()[2].name=="i");
    const auto leftId=session->rows()[0].materialRegionId;
    check("move_connected_shared_boundary",session->moveBoundary(leftId,.3,error)
        &&std::abs(session->rows()[0].regionEndSeconds-.3)<1.e-9&&std::abs(session->rows()[1].regionStartSeconds-.3)<1.e-9);
    check("invalid_boundary_rolls_back",!session->moveBoundary(leftId,.9,error)&&std::abs(session->rows()[0].regionEndSeconds-.3)<1.e-9);
    check("save_native_manifest",session->save(error));
    check("save_never_creates_csv",!SampleSettings::sidecarFor(audio).exists());
    const auto id=session->rows()[1].materialRegionId;
    auto reopened=MaterialAnnotationSession::open(audio,manifest,error);
    check("reopen_preserves_pronunciation_id",reopened&&reopened->rows()[1].materialRegionId==id);
    check("multiple_editor_conflict_fixture",session->rename({"ga","a","i"},error)&&session->save(error));
    if(reopened)
    {
        check("stale_editor_rejects_publication",reopened->rename({"old","a","i"},error)&&!reopened->save(error));
    }
    auto current=MaterialAnnotationSession::open(audio,manifest,error);
    if(!current)return false;
    const auto aId=current->rows()[0].materialRegionId,bId=current->rows()[1].materialRegionId;
    check("merge_adjacent_preserves_source_and_outer_link",current->merge({aId,bId},error)
        &&current->rows().size()==2&&current->rows()[0].materialRegionId==aId
        &&current->rows()[0].connectedNextRegionId==current->rows()[1].materialRegionId);
    check("merge_is_undoable",current->undo()&&current->rows().size()==3);
    const auto removed=current->rows()[1].materialRegionId;
    check("erase_severs_link_no_dangling_ids",current->erase({removed},error)&&current->rows().size()==2
        &&current->rows()[0].connectedNextRegionId.isEmpty());
    current->undo();
    current->validatePublication=[](const auto&,juce::String& e){e="Fixture blocks unsafe project remapping";return false;};
    const auto bytes=manifest.loadFileAsString();
    check("project_preflight_before_disk_publication",!current->save(error)&&manifest.loadFileAsString()==bytes);
    juce::MemoryBlock sourceAfter;audio.loadFileAsData(sourceAfter);
    check("pcm_bytes_unchanged",sourceBefore==sourceAfter);
    root.getChildFile("verification.txt").replaceWithText(log);
    std::cout<<"annotation_passed="<<total-failed<<"|total="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;
    if(failed)std::cerr<<log<<std::endl;
    return failed==0;
}
}
