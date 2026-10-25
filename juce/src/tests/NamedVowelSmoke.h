#pragma once
#include "../MaterialAnnotationSession.h"
#include "../OtoWaveformEditorComponent.h"
#include "../OttoMaterialIO.h"
#include "../NativeNoteTiming.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace hachi
{
inline bool runNamedVowelSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("named-vowels-"+juce::Uuid().toString());if(root.createDirectory().failed())return false;
    bool ok=true;int total=0;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass)
    {++total;ok&=pass;log+=name+"="+(pass?"PASS":"FAIL")+(pass||error.isEmpty()?juce::String{}:" | "+error)+"\n";};
    const auto finish=[&]
    {const bool written=root.getChildFile("verification.txt").replaceWithText(log);std::cout<<"named_vowel_passed="<<(ok?1:0)<<"|checks="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;if(!ok)std::cerr<<log<<std::endl;return ok&&written;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-8;};
    const auto vowels=[](const SampleRegionSetting& row)
    {std::vector<NativeSegment> values;for(const auto& s:row.segments)if(s.role==NativeSegmentRole::vowel&&s.stretchable)values.push_back(s);return values;};
    const auto source=root.getChildFile("302-second-source.wav"),manifest=root.getChildFile("otto.ini");
    {
        // Long-source UI regression without a large RAM allocation or any
        // analysis/model injection. PCM is generated once and never rewritten.
        juce::AudioBuffer<float> second(1,8000);for(int i=0;i<8000;++i)second.setSample(0,i,.04f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*220*i/8000.0)));
        auto out=source.createOutputStream();juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?wav.createWriterFor(out.release(),8000,1,16,{},0):nullptr);
        bool written=bool(writer);for(int i=0;written&&i<302;++i)written=writer->writeFromAudioSampleBuffer(second,0,8000);
        check("own_302_second_source",written);
    }
    juce::MemoryBlock sourceBytes;check("source_bytes_captured",source.loadFileAsData(sourceBytes));if(!ok)return finish();
    SampleRegionSetting legacy;legacy.name="chang";legacy.materialAudioId="audio-long";legacy.materialRegionId="pron-long";
    legacy.regionStartSeconds=0;legacy.regionEndSeconds=302;legacy.fixedDurationSeconds=.2;legacy.alignmentSeconds=.1;
    legacy.overlapSeconds=-.025;legacy.pronunciationKind=PronunciationKind::transition;legacy.simplifiedConsonant=true;
    legacy.relativePitchCents=37.125;legacy.amplitudeEnvelope={{0,-6},{302,-1}};legacy.vowelSplitsExplicit=false;
    legacy.segments.push_back({"legacy-c","chang",NativeSegmentRole::consonant,0,.2,"analysis",1,.1,-.025,false,1});
    for(int i=0;i<200;++i)
        legacy.segments.push_back({"analysis-v-"+juce::String(i),"chang",NativeSegmentRole::vowel,
            .2+301.8*i/200.0,.2+301.8*(i+1)/200.0,"analysis",1,.1,-.025,true,1});
    check("raw_legacy_200_vowel_fixture",OttoMaterialIO::saveAudio(manifest,source,{legacy},error));
    std::vector<SampleRegionSetting> raw;check("raw_file_really_contains_200_vowels",OttoMaterialIO::loadAudio(manifest,source,raw,error)
        &&raw.size()==1&&vowels(raw[0]).size()==200&&!raw[0].vowelSplitsExplicit);
    const auto originalManifest=manifest.loadFileAsString();
    auto session=MaterialAnnotationSession::open(source,manifest,error);
    check("open_semantically_normalizes_actual_draft_to_one_V",session&&session->rows().size()==1&&vowels(session->rows()[0]).size()==1
        &&std::count_if(session->rows()[0].segments.begin(),session->rows()[0].segments.end(),[](const auto& s){return s.role==NativeSegmentRole::consonant;})==1
        &&!session->rows()[0].vowelSplitsExplicit);
    check("normalization_is_observable_and_read_only",session&&!session->normalizationSummary().isEmpty()
        &&manifest.loadFileAsString()==originalManifest&&!session->canUndo());
    if(!session||session->rows().empty()||vowels(session->rows()[0]).size()!=1)return finish();
    const auto region=session->rows()[0].materialRegionId,defaultV=vowels(session->rows()[0])[0].id;
    const auto preserved=[](const SampleRegionSetting& row)
    {return row.name=="chang"&&row.regionStartSeconds==0&&row.regionEndSeconds==302
        &&row.fixedDurationSeconds==.2&&row.alignmentSeconds==.1&&row.overlapSeconds==-.025
        &&row.pronunciationKind==PronunciationKind::transition&&row.simplifiedConsonant
        &&row.relativePitchCents==37.125&&row.amplitudeEnvelope.size()==2
        &&row.amplitudeEnvelope[0].gainDb==-6&&row.amplitudeEnvelope[1].timeSeconds==302;};
    check("normalization_preserves_whole_identity_C_P_and_expressive_facts",preserved(session->rows()[0]));
    check("empty_left_name_rejects_without_undo",!session->splitVowel(region,defaultV,151.1,"","ng",error)&&!session->canUndo());
    check("empty_right_name_rejects_without_undo",!session->splitVowel(region,defaultV,151.1,"an"," ",error)&&!session->canUndo());
    check("unassigned_placeholder_is_not_a_manual_vowel_name",!session->splitVowel(region,defaultV,151.1,"an","-",error)&&!session->canUndo());
    check("duplicate_endpoint_cut_rejects_without_undo",!session->splitVowel(region,defaultV,.2,"an","ng",error)
        &&!session->splitVowel(region,defaultV,302,"an","ng",error)&&!session->canUndo());
    check("unknown_segment_id_rejects_without_undo",!session->splitVowel(region,"missing",151.1,"an","ng",error)&&!session->canUndo());
    check("all_rejected_edits_leave_raw_file_untouched",manifest.loadFileAsString()==originalManifest&&vowels(session->rows()[0]).size()==1);
    I18n strings;strings.setLanguage(I18n::Language::enUS);int publications=0;
    auto editor=OtoWaveformEditorComponent::forMaterial(source,manifest,strings,[&](const auto&){++publications;},false,&error);
    check("actual_factory_default_one_V",editor&&editor->diagnosticMaterialRows().size()==1&&vowels(editor->diagnosticMaterialRows()[0]).size()==1);
    if(!editor)return finish();editor->diagnosticSetAnalysisLayers(false,false);editor->setSize(1000,780);
    check("actual_detail_selection",editor->selectMaterialPronunciation(region,true));
    const auto defaultItems=editor->diagnosticMaterialVowelItems();
    check("actual_selector_one_ordinal_not_raw_C_index",defaultItems.size()==1&&defaultItems[0].startsWith("1: chang"));
    {
        auto out=root.getChildFile("actual-default-single-vowel-no-analysis-cuts.png").createOutputStream();
        check("default_single_V_actual_component_snapshot",out&&juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*out));
    }
    editor->diagnosticBeginMaterialVowelSplit();
    check("split_prompt_alone_makes_no_actual_cut",editor->diagnosticMaterialVowelItems().size()==1
        &&manifest.loadFileAsString()==originalManifest&&!editor->diagnosticMaterialRows()[0].vowelSplitsExplicit);
    check("prompt_empty_names_reject_actual_split",!editor->diagnosticConfirmMaterialVowelSplit("","ng")
        &&editor->diagnosticMaterialVowelItems().size()==1);
    editor->diagnosticCancelMaterialVowelSplit();
    check("prompt_cancel_preserves_source_and_draft",editor->diagnosticMaterialVowelItems().size()==1&&manifest.loadFileAsString()==originalManifest);
    check("explicit_actual_named_split",editor->diagnosticSplitMaterialVowel("an","ng"));
    auto namedItems=editor->diagnosticMaterialVowelItems();
    check("actual_named_selector_ordinal_mapping",namedItems.size()==2&&namedItems[0].startsWith("1: an")&&namedItems[1].startsWith("2: ng"));
    log+="actual_combo_items="+namedItems.joinIntoString(" | ")+"\n";
    check("edit_first_actual_stretch_weight",editor->diagnosticEditMaterialVowel("an","2"));
    editor->diagnosticSelectMaterialVowel(2);check("edit_second_actual_stretch_weight",editor->diagnosticEditMaterialVowel("ng","3"));
    const auto snapshot=[&](const juce::String& name)
    {auto out=root.getChildFile(name+".png").createOutputStream();check(name,out&&juce::PNGImageFormat().writeImageToStream(editor->createComponentSnapshot(editor->getLocalBounds()),*out));};
    snapshot("actual-named-vowel-selector-ng");editor->diagnosticSelectMaterialVowel(1);snapshot("actual-named-vowel-selector-an");
    check("named_split_not_published_before_explicit_save",manifest.loadFileAsString()==originalManifest&&publications==0);
    check("commit_named_detail",editor->diagnosticMaterialBack());
    const auto staged=editor->diagnosticMaterialRows();
    check("primary_chang_remains_whole_alias",staged.size()==1&&preserved(staged[0])&&staged[0].vowelSplitsExplicit
        &&vowels(staged[0]).size()==2&&vowels(staged[0])[0].alias=="an"&&vowels(staged[0])[1].alias=="ng");
    editor->diagnosticSave();
    auto named=MaterialAnnotationSession::open(source,manifest,error);
    check("named_flag_weights_and_names_survive_otto_reopen",publications==1&&named&&named->rows().size()==1
        &&named->rows()[0].vowelSplitsExplicit&&vowels(named->rows()[0]).size()==2
        &&vowels(named->rows()[0])[0].alias=="an"&&vowels(named->rows()[0])[1].alias=="ng"
        &&near(vowels(named->rows()[0])[0].stretchWeight,2)&&near(vowels(named->rows()[0])[1].stretchWeight,3)
        &&preserved(named->rows()[0])&&manifest.loadFileAsString().contains("vowel-splits"));
    if(!named||named->rows().empty()||vowels(named->rows()[0]).size()!=2)return finish();
    const auto namedManifest=manifest.loadFileAsString();
    check("raw_observation_fingerprint_still_protects_normalized_draft",!session->save(error)&&manifest.loadFileAsString()==namedManifest);
    auto currentVowels=vowels(named->rows()[0]);const auto firstV=currentVowels[0].id,secondV=currentVowels[1].id;
    check("zero_and_negative_weight_reject_without_undo",!named->editVowel(region,firstV,"an",0,error)
        &&!named->editVowel(region,firstV,"an",-1,error)&&!named->canUndo());
    check("nonfinite_weight_reject_without_undo",!named->editVowel(region,firstV,"an",std::numeric_limits<double>::infinity(),error)&&!named->canUndo());
    auto duplicateId=named->rows()[0];duplicateId.segments.back().id=firstV;
    check("duplicate_segment_id_rejects_without_undo",!named->setRow(region,duplicateId,error)&&!named->canUndo());
    std::vector<SourceTimePoint> map;
    check("shared_weighted_stretch_map",makePronunciationStretchMap(named->rows()[0],604,map,error));
    if(!map.empty())
    {
        const auto at=[&](double sourceTime){return nativeTargetTimeAt(map,sourceTime);};
        const auto firstSpan=at(currentVowels[0].sourceEndSeconds)-at(currentVowels[0].sourceStartSeconds);
        const auto secondSpan=at(currentVowels[1].sourceEndSeconds)-at(currentVowels[1].sourceStartSeconds);
        check("weights_control_V_ratio_without_changing_fixed_C",near(at(.2),.2)&&secondSpan>0&&std::abs(firstSpan/secondSpan-2.0/3)<1.e-7);
    }
    const auto cut=(currentVowels[0].sourceStartSeconds+currentVowels[0].sourceEndSeconds)*.5;
    check("repeated_phoneme_text_is_allowed_with_distinct_ids",named->splitVowel(region,firstV,cut,"a","a",error)
        &&vowels(named->rows()[0]).size()==3&&vowels(named->rows()[0])[0].alias=="a"&&vowels(named->rows()[0])[1].alias=="a"
        &&vowels(named->rows()[0])[0].id!=vowels(named->rows()[0])[1].id);
    const auto repeatedIds=vowels(named->rows()[0]);
    check("manual_split_is_one_atomic_undo",named->undo()&&vowels(named->rows()[0]).size()==2
        &&vowels(named->rows()[0])[0].id==firstV&&vowels(named->rows()[0])[1].id==secondV&&!named->canUndo());
    check("redo_keeps_manual_segment_ids",named->redo()&&vowels(named->rows()[0])[1].id==repeatedIds[1].id);
    for(int i=0;i<12;++i)
    {
        currentVowels=vowels(named->rows()[0]);const auto& last=currentVowels.back();
        check("unlimited_manual_vowel_"+juce::String(i),named->splitVowel(region,last.id,
            (last.sourceStartSeconds+last.sourceEndSeconds)*.5,"left"+juce::String(i),"right"+juce::String(i),error));
    }
    check("more_than_legacy_four_or_eight_vowels_is_explicit_intent",vowels(named->rows()[0]).size()==15
        &&named->rows()[0].vowelSplitsExplicit&&preserved(named->rows()[0]));
    auto merge=MaterialAnnotationSession::open(source,manifest,error);
    check("merge_last_manual_pair_returns_default_one_V",merge&&merge->mergeVowels(region,firstV,error)
        &&vowels(merge->rows()[0]).size()==1&&!merge->rows()[0].vowelSplitsExplicit&&vowels(merge->rows()[0])[0].alias=="chang");
    check("merge_one_undo_restores_named_pair",merge&&merge->undo()&&merge->rows()[0].vowelSplitsExplicit
        &&vowels(merge->rows()[0]).size()==2&&vowels(merge->rows()[0])[0].alias=="an");
    check("explicit_many_named_vowels_save",named->save(error));
    auto many=MaterialAnnotationSession::open(source,manifest,error);
    check("explicit_named_cuts_not_coalesced_on_reopen",many&&many->rows().size()==1&&many->rows()[0].vowelSplitsExplicit&&vowels(many->rows()[0]).size()==15
        &&preserved(many->rows()[0])&&vowels(many->rows()[0])[0].alias=="a"&&vowels(many->rows()[0])[1].alias=="a"
        &&vowels(many->rows()[0])[0].id!=vowels(many->rows()[0])[1].id);
    juce::MemoryBlock finalBytes;check("unchanged_original_302_second_PCM",source.loadFileAsData(finalBytes)&&finalBytes==sourceBytes);
    return finish();
}
}
