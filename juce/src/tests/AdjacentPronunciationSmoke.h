#pragma once
#include "../MaterialAnnotationSession.h"
#include "../OtoWaveformEditorComponent.h"
#include "../PronunciationTiming.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace hachi
{
inline bool runAdjacentPronunciationSmoke(const juce::File& output)
{
    const auto root=output.getChildFile("adjacent-"+juce::Uuid().toString());if(!root.createDirectory().wasOk())return false;
    int total=0,failed=0;juce::String log,error;
    const auto check=[&](const juce::String& name,bool pass)
    {++total;if(!pass)++failed;log+=name+"="+(pass?"PASS":"FAIL")+(pass||error.isEmpty()?juce::String{}:" | "+error)+"\n";};
    const auto finish=[&]
    {const bool written=root.getChildFile("verification.txt").replaceWithText(log);std::cout<<"adjacent_passed="<<total-failed<<"|total="<<total<<"|fixture="<<root.getFullPathName()<<std::endl;if(failed)std::cerr<<log<<std::endl;return failed==0&&written;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-8;};
    const auto makeAudio=[](const juce::File& file)
    {
        file.getParentDirectory().createDirectory();juce::AudioBuffer<float> pcm(1,192000);
        for(int i=0;i<pcm.getNumSamples();++i)pcm.setSample(0,i,.1f*static_cast<float>(std::sin(juce::MathConstants<double>::twoPi*220*i/48000.0)));
        auto out=file.createOutputStream();juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(out?format.createWriterFor(out.release(),48000,1,16,{},0):nullptr);
        return writer&&writer->writeFromAudioSampleBuffer(pcm,0,pcm.getNumSamples());
    };
    const auto source=root.getChildFile("session/source.wav"),manifest=source.getSiblingFile("otto.ini");
    check("session_fixture",makeAudio(source));juce::MemoryBlock sourceBefore;check("source_snapshot",source.loadFileAsData(sourceBefore));
    auto session=MaterialAnnotationSession::open(source,manifest,error);check("bare_session_empty",session&&session->rows().empty());if(!session)return finish();
    check("first_anchor",session->createPronunciation(1,2,"anchor",error));
    double firstStart=0,firstEnd=0;
    if(session->rows().size()==1)
    {
        const auto firstAnchor=session->rows()[0].materialRegionId;
        check("bare_left_free_edge_clips_recording_zero",session->previewAdjacentRange(firstAnchor,true,-4,firstStart,firstEnd,error)
            &&near(firstStart,0)&&near(firstEnd,1));
        check("bare_right_free_edge_clips_recording_duration",session->previewAdjacentRange(firstAnchor,false,8,firstStart,firstEnd,error)
            &&near(firstStart,2)&&near(firstEnd,4));
    }
    check("anchor_and_outside_neighbors",session->createPronunciation(0,.5,"upstream",error)&&session->createPronunciation(3,4,"downstream",error));
    if(session->rows().size()!=3)return finish();
    const auto anchor=session->rows()[1].materialRegionId,upstream=session->rows()[0].materialRegionId,downstream=session->rows()[2].materialRegionId;
    const auto audioId=session->rows()[1].materialAudioId;
    double start=0,end=0;
    check("left_preview_clips_to_nearest_occupied_edge",session->previewAdjacentRange(anchor,true,-4,start,end,error)&&near(start,.5)&&near(end,1));
    check("left_adjacent_atomic_insert",session->createAdjacentPronunciation(anchor,true,-4,"left",error)&&session->rows().size()==4);
    auto rows=session->rows();
    check("left_exact_join_links_and_shared_audio",rows.size()==4&&near(rows[1].regionStartSeconds,.5)&&near(rows[1].regionEndSeconds,1)
        &&rows[0].materialRegionId==upstream&&rows[0].connectedNextRegionId==rows[1].materialRegionId
        &&rows[1].connectedNextRegionId==anchor&&rows[1].materialAudioId==audioId);
    check("adjacent_insert_is_one_undo",session->undo()&&session->rows().size()==3&&session->rows()[0].connectedNextRegionId.isEmpty());
    check("adjacent_insert_redo_keeps_identity",session->redo()&&session->rows().size()==4&&session->rows()[1].materialRegionId==rows[1].materialRegionId);
    check("right_preview_clips_to_nearest_occupied_edge",session->previewAdjacentRange(anchor,false,8,start,end,error)&&near(start,2)&&near(end,3));
    check("right_adjacent_atomic_insert",session->createAdjacentPronunciation(anchor,false,8,"right",error)&&session->rows().size()==5);
    rows=session->rows();
    check("right_exact_join_and_downstream_chain",rows.size()==5&&rows[2].materialRegionId==anchor
        &&rows[2].connectedNextRegionId==rows[3].materialRegionId&&rows[3].connectedNextRegionId==downstream
        &&near(rows[3].regionStartSeconds,2)&&near(rows[3].regionEndSeconds,3));
    bool chain=true;for(std::size_t i=0;i<rows.size();++i)
    {chain&=rows[i].materialAudioId==audioId;if(i+1<rows.size())chain&=rows[i].connectedNextRegionId==rows[i+1].materialRegionId&&near(rows[i].regionEndSeconds,rows[i+1].regionStartSeconds);}
    check("one_wav_audio_id_monotone_gapless_chain",chain);
    check("blocked_left_join_rejected",!session->createAdjacentPronunciation(anchor,true,0,"blocked",error)&&session->rows().size()==5);
    check("blocked_right_join_rejected",!session->createAdjacentPronunciation(anchor,false,4,"blocked",error)&&session->rows().size()==5);
    check("missing_anchor_rejected",!session->createAdjacentPronunciation("missing",true,.2,"bad",error));
    check("exact_range_alias_not_duplicate_pronunciation",session->createPronunciation(1,2,"alias",error)&&session->rows().size()==5
        &&session->rows()[2].materialRegionId==anchor&&session->rows()[2].additionalAliases.contains("alias"));
    check("cancel_exact_range_alias_keeps_identity",session->removeAlias(anchor,"alias",error)&&session->rows()[2].materialRegionId==anchor
        &&session->rows()[2].additionalAliases.isEmpty());
    auto row=session->rows()[2];row.amplitudeEnvelope={{0,-3},{1,0}};row.relativePitchCents=37;
    const auto initialSegments=row.segments;
    for(int kind=0;kind<3;++kind)
    {
        row.pronunciationKind=static_cast<PronunciationKind>(kind);
        check("whole_kind_"+juce::String(kind)+"_is_metadata_only",session->setRow(anchor,row,error)
            &&session->rows()[2].pronunciationKind==static_cast<PronunciationKind>(kind)
            &&session->rows()[2].relativePitchCents==37&&session->rows()[2].amplitudeEnvelope.size()==2
            &&session->rows()[2].segments.size()==initialSegments.size()&&session->rows()[2].segments[0].role==NativeSegmentRole::vowel);
    }
    check("simplified_default_off",!row.simplifiedConsonant);
    row.overlapSeconds=-.04;
    check("canonical_independent_markers",editPronunciationConsonant(row,1.3,error)&&editPronunciationAlignment(row,1.1,error)
        &&near(pronunciationTimingFor(row).consonantEndSeconds,1.3));
    check("enable_simplified_captures_existing_gap",setSimplifiedConsonant(row,true,-1,error)
        &&editPronunciationAlignment(row,1.4,error)&&near(pronunciationTimingFor(row).consonantEndSeconds,1.6));
    check("explicit_gap_uses_same_canonical_timing",setSimplifiedConsonant(row,true,.075,error)
        &&near(pronunciationTimingFor(row).consonantEndSeconds,1.475));
    check("save_session_kind_and_simplified",session->setRow(anchor,row,error)&&session->save(error));
    auto reopened=MaterialAnnotationSession::open(source,manifest,error);
    check("reopen_preserves_chain_kind_simplified",reopened&&reopened->rows().size()==5
        &&reopened->rows()[2].materialRegionId==anchor&&reopened->rows()[2].pronunciationKind==PronunciationKind::breath
        &&reopened->rows()[2].simplifiedConsonant&&near(reopened->rows()[2].alignmentSeconds,1.4)
        &&near(pronunciationTimingFor(reopened->rows()[2]).consonantEndSeconds,1.475)&&near(reopened->rows()[2].overlapSeconds,-.04));
    check("new_source_revision_fixture",session->createPronunciation(1,2,"new-alias",error)&&session->save(error));
    const auto publishedBytes=manifest.loadFileAsString();
    check("stale_fingerprint_still_rejects_publication",reopened&&reopened->createPronunciation(1,2,"stale",error)
        &&!reopened->save(error)&&manifest.loadFileAsString()==publishedBytes);
    // Actual factory and overview mouse range gestures, never a parallel model.
    const auto uiSource=root.getChildFile("editor/source.wav"),uiManifest=uiSource.getSiblingFile("otto.ini");
    check("editor_fixture",makeAudio(uiSource));juce::MemoryBlock uiBefore;uiSource.loadFileAsData(uiBefore);
    I18n strings;strings.setLanguage(I18n::Language::enUS);int publications=0;
    auto editor=OtoWaveformEditorComponent::forMaterial(uiSource,uiManifest,strings,[&](const auto&){++publications;},false,&error);
    check("actual_factory_native_overview",editor&&editor->diagnosticMaterialOverview());if(!editor)return finish();
    int adjacentControls=0;bool adjacentDefaultOff=true;
    const auto inspectAdjacent=[&](auto&& self,juce::Component& component)->void
    {
        for(int i=0;i<component.getNumChildComponents();++i)
        {
            auto* child=component.getChildComponent(i);
            if(auto* tick=dynamic_cast<juce::ToggleButton*>(child);tick&&tick->getButtonText()=="Adjacent pronunciation")
            {++adjacentControls;adjacentDefaultOff&=!tick->getToggleState();}
            self(self,*child);
        }
    };
    inspectAdjacent(inspectAdjacent,*editor);
    check("one_overview_adjacency_option_default_off",adjacentControls==1&&adjacentDefaultOff);
    check("actual_factory_anchor",editor->diagnosticCreateMaterialAlias(1,2,"anchor",error));
    auto uiRows=editor->diagnosticMaterialRows();if(uiRows.size()!=1)return finish();const auto uiAnchor=uiRows[0].materialRegionId;
    editor->setSize(1000,780);editor->diagnosticMaterialAdjacent(true);
    const auto mouse=[](juce::Component& c,juce::Point<float> at,juce::Point<float> down)
    {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&c,&c,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
    const auto rangeGesture=[&](double from,double to,const juce::String& alias)
    {
        auto* wave=editor->diagnosticMaterialOverviewWaveform();if(!wave)return false;
        const auto point=[&](double time){return juce::Point<float>(10.f+static_cast<float>(time/4)*(wave->getWidth()-20.f),60.f);};
        const auto down=point(from),up=point(to);wave->mouseDown(mouse(*wave,down,down));wave->mouseDrag(mouse(*wave,up,down));wave->mouseUp(mouse(*wave,up,down));
        editor->diagnosticMaterialAliasText(alias);editor->diagnosticCreateSelectedMaterialAlias();return true;
    };
    check("factory_select_adjacent_anchor",editor->selectMaterialPronunciation(uiAnchor,false));
    check("actual_left_mouse_gesture",rangeGesture(1.2,.25,"left"));
    uiRows=editor->diagnosticMaterialRows();
    check("actual_left_gesture_uses_exact_anchor_start",uiRows.size()==2&&near(uiRows[0].regionStartSeconds,.25)
        &&near(uiRows[0].regionEndSeconds,1)&&uiRows[0].connectedNextRegionId==uiAnchor&&uiRows[1].materialRegionId==uiAnchor);
    check("factory_reselect_anchor",editor->selectMaterialPronunciation(uiAnchor,false));
    {
        auto* wave=editor->diagnosticMaterialOverviewWaveform();
        if(wave)
        {
            const auto point=[&](double time){return juce::Point<float>(10.f+static_cast<float>(time/4)*(wave->getWidth()-20.f),60.f);};
            const auto down=point(1.8),right=point(2.5),crossed=point(.1);
            wave->mouseDown(mouse(*wave,down,down));wave->mouseDrag(mouse(*wave,right,down));
            wave->mouseDrag(mouse(*wave,crossed,down));wave->mouseUp(mouse(*wave,crossed,down));
            editor->diagnosticMaterialAliasText("must-not-flip");editor->diagnosticCreateSelectedMaterialAlias();
        }
        check("adjacent_gesture_side_is_monotonic_and_zero_rejects",editor->diagnosticMaterialRows().size()==2);
    }
    check("factory_reselect_after_rejected_crossing",editor->selectMaterialPronunciation(uiAnchor,false));
    check("actual_right_mouse_gesture",rangeGesture(1.8,3,"right"));
    uiRows=editor->diagnosticMaterialRows();
    check("actual_right_gesture_uses_exact_anchor_end",uiRows.size()==3&&uiRows[1].materialRegionId==uiAnchor
        &&near(uiRows[2].regionStartSeconds,2)&&near(uiRows[2].regionEndSeconds,3)&&uiRows[1].connectedNextRegionId==uiRows[2].materialRegionId);
    check("actual_detail_selection",editor->selectMaterialPronunciation(uiAnchor,true));
    check("actual_multi_row_outer_lock",editor->diagnosticOuterRangeLocked());
    bool simplifiedDefaultOff=false;
    for(int i=0;i<editor->getNumChildComponents();++i)
        if(auto* tick=dynamic_cast<juce::ToggleButton*>(editor->getChildComponent(i));tick&&tick->getButtonText()=="Simplified consonant")
            simplifiedDefaultOff=tick->isVisible()&&!tick->getToggleState();
    check("detail_simplified_control_default_off",simplifiedDefaultOff);
    editor->diagnosticTypeParameter(0,"500");editor->diagnosticTypeParameter(2,"500");editor->diagnosticDragOffsetTo(500);
    check("actual_outer_typing_drag_unchanged",near(editor->diagnosticParameterText(0).getDoubleValue(),1000)
        &&near(editor->diagnosticParameterText(2).getDoubleValue(),2000));
    editor->diagnosticTypeParameter(1,"300");editor->diagnosticTypeParameter(3,"100");
    editor->diagnosticSimplifiedConsonant(true);editor->diagnosticTypeParameter(3,"250");
    check("typed_P_follows_derived_gap",near(editor->diagnosticParameterText(1).getDoubleValue(),450));
    editor->diagnosticDragMaterialAlignmentTo(1600);
    check("actual_P_drag_follows_derived_gap",near(editor->diagnosticParameterText(1).getDoubleValue(),800)
        &&near(editor->diagnosticParameterText(3).getDoubleValue(),600));
    editor->diagnosticConsonantGap("50");
    check("gap_input_changes_C_not_P",near(editor->diagnosticParameterText(1).getDoubleValue(),650)
        &&near(editor->diagnosticParameterText(3).getDoubleValue(),600));
    editor->diagnosticDragMaterialConsonantTo(1800);editor->diagnosticDragMaterialAlignmentTo(1500);
    check("actual_C_drag_updates_derived_gap",near(editor->diagnosticParameterText(1).getDoubleValue(),700));
    editor->diagnosticSimplifiedConsonant(false);editor->diagnosticDragMaterialAlignmentTo(1400);
    check("disabled_simplified_keeps_C_independent",near(editor->diagnosticParameterText(1).getDoubleValue(),700));
    editor->diagnosticSimplifiedConsonant(true);editor->diagnosticConsonantGap("80");editor->diagnosticPronunciationKind(3);
    int kindChoices=0;
    for(int i=0;i<editor->getNumChildComponents();++i)
        if(auto* combo=dynamic_cast<juce::ComboBox*>(editor->getChildComponent(i));combo&&combo->getNumItems()==3
            &&combo->getItemText(0)=="Normal syllable"&&combo->getItemText(2)=="Breath")++kindChoices;
    check("one_three_kind_dropdown_no_internal_role_identity",kindChoices==1);
    editor->diagnosticMaterialMode(2);
    check("classic_view_reads_same_C_P",near(editor->diagnosticParameterText(1).getDoubleValue(),480)
        &&near(editor->diagnosticParameterText(3).getDoubleValue(),400));
    editor->diagnosticMaterialMode(1);
    check("native_view_reads_same_C_P",near(editor->diagnosticParameterText(1).getDoubleValue(),480)
        &&near(editor->diagnosticParameterText(3).getDoubleValue(),400));
    const auto image=editor->createComponentSnapshot(editor->getLocalBounds());auto png=root.getChildFile("native-detail-three-kind-simplified.png").createOutputStream();
    check("actual_detail_snapshot",png&&juce::PNGImageFormat().writeImageToStream(image,*png));
    check("detail_return_commits_shared_draft",editor->diagnosticMaterialBack());editor->diagnosticSave();
    auto reloaded=MaterialAnnotationSession::open(uiSource,uiManifest,error);
    check("actual_factory_save_roundtrip",publications==1&&reloaded&&reloaded->rows().size()==3
        &&reloaded->rows()[1].materialRegionId==uiAnchor&&reloaded->rows()[1].pronunciationKind==PronunciationKind::breath
        &&reloaded->rows()[1].simplifiedConsonant&&near(reloaded->rows()[1].alignmentSeconds,1.4)
        &&near(pronunciationTimingFor(reloaded->rows()[1]).consonantEndSeconds,1.48));
    juce::MemoryBlock sourceAfter,uiAfter;
    check("all_source_pcm_bytes_unchanged",source.loadFileAsData(sourceAfter)&&sourceBefore==sourceAfter
        &&uiSource.loadFileAsData(uiAfter)&&uiBefore==uiAfter);
    check("no_legacy_pcm_copy_or_csv",!SampleSettings::sidecarFor(source).exists()&&!SampleSettings::sidecarFor(uiSource).exists());
    return finish();
}
}
