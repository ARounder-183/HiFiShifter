#pragma once
#include "../AdvancedEnvelopePanel.h"
namespace hachi
{
inline bool MainComponent::diagnosticAdvancedEnvelopePanel(const juce::File& folder)
{
    folder.createDirectory();bool ok=diagnosticAdvancedEnvelope(folder.getChildFile("baseline"));stopTimer();
    const auto check=[&](const char* name,bool v){ok=ok&&v;std::cout<<name<<'='<<v<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<4.e-4;};
    // The same graph edits OTO head/tail, native and CVVC drafts.
    for(int context=0;context<4;++context)
    {
        backend::TailFadeSettings v;v.mixed=true;v.knots={{.25,.8,false},{.55,.5,false},{.8,.2,false}};
        backend::copyEnvelopeShape(v,v.head);v.head.mode=2;
        AdvancedEnvelopePanel panel({{"selection","Point selection",{0,1},{{0,0,true},{1,0,true}}, {},backend::UtauTailFadeSpan{0,1}}},
            1,2,v,false,{},context>=2,context==3);
        if(context==1)panel.diagnosticSection(true);
        auto& graph=panel.graph;int applies=0;backend::TailFadeSettings applied;
        panel.onApply=[&](int,const auto& value,bool){++applies;applied=value;return juce::String();};
        const auto event=[&](juce::Point<float> pos,juce::Point<float> down,bool ctrl=false){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),pos,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier|(ctrl?juce::ModifierKeys::ctrlModifier:0)),
            0,0,0,0,0,&graph,&graph,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,pos!=down);};
        const auto click=[&](juce::Point<float> pos){graph.mouseDown(event(pos,pos));graph.mouseUp(event(pos,pos));};
        const auto del=[&]{return graph.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));};
        const auto resetSelection=[&]{panel.setValues(2,v);graph.fit.onClick();};
        const auto before=panel.values();click(graph.knotPoint(2));
        check("advanced_point_release_preserves_highlight_and_draft",graph.selectedKnot==2&&graph.draggedKnot==-1&&panel.values()==before&&applies==0);
        if(context==3)if(auto out=folder.getChildFile("point-selected-after-release.png").createOutputStream())
        {out->setPosition(0);out->truncate();juce::PNGImageFormat().writeImageToStream(panel.createComponentSnapshot(panel.getLocalBounds()),*out);}
        check("advanced_delete_after_release_removes_only_selected_knot",del()&&panel.editingValues().knots.size()==2
            &&near(panel.editingValues().knots[0].time,.25)&&near(panel.editingValues().knots[1].time,.8)&&applies==0);
        const auto deleted=panel.values();del();del();check("advanced_repeat_delete_does_not_remove_more_points",panel.values()==deleted);
        check("advanced_delete_preserves_other_section",context==1?panel.values().knots==v.knots:panel.values().head==v.head);
        panel.diagnosticApply();check("advanced_apply_receives_deleted_curve",applies==1&&applied==deleted);
        resetSelection();auto from=graph.knotPoint(2),to=from+juce::Point<float>(8,4);
        graph.mouseDown(event(from,from));graph.mouseDrag(event(to,from));graph.mouseUp(event(to,from));
        check("advanced_drag_release_keeps_selection",graph.selectedKnot==2&&!near(panel.editingValues().knots[1].time,.55));
        del();check("advanced_drag_then_delete",panel.editingValues().knots.size()==2);
        resetSelection();click(graph.knotPoint(1));click(graph.knotPoint(3));del();
        check("advanced_new_click_changes_point_selection",panel.editingValues().knots.size()==2&&near(panel.editingValues().knots.back().time,.55));
        resetSelection();click(graph.knotPoint(2));click(graph.plot().getRelativePoint(.1f,.75f));del();
        check("advanced_blank_click_clears_point",panel.editingValues().knots.size()==3);
        resetSelection();click(graph.knotPoint(2));graph.focusLost(juce::Component::focusChangedDirectly);del();
        check("advanced_focus_loss_clears_point",panel.editingValues().knots.size()==3);
        resetSelection();click(graph.knotPoint(2));panel.setValues(2,v);del();
        check("advanced_replacing_draft_invalidates_selection",panel.editingValues().knots.size()==3);
        resetSelection();click(graph.knotPoint(2));panel.diagnosticSelectShape(1);panel.diagnosticSelectShape(3);del();
        check("advanced_shape_change_clears_point",panel.editingValues().knots.size()==3);
        resetSelection();click(graph.knotPoint(0));del();click(graph.knotPoint(4));del();
        check("advanced_endpoints_retained",panel.editingValues().knots.size()==3);
        resetSelection();from=graph.knotPoint(2);graph.mouseDown(event(from,from));del();graph.mouseUp(event(from,from));
        check("advanced_delete_during_drag_not_resurrected",panel.editingValues().knots.size()==2);
        resetSelection();graph.setZoom(2,2);click(graph.knotPoint(2));
        check("advanced_zoomed_backspace",graph.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey))&&panel.editingValues().knots.size()==2);
        resetSelection();click(graph.knotPoint(2));panel.diagnosticSelectPreview(0);del();
        check("advanced_preview_change_clears_point",panel.editingValues().knots.size()==3);
        const auto ctrlClick=[&](int index){const auto p=graph.knotPoint(index);graph.mouseDown(event(p,p,true));graph.mouseUp(event(p,p,true));};
        resetSelection();click(graph.knotPoint(1));ctrlClick(2);
        check("advanced_ctrl_multiselect",graph.selectedKnots.size()==2);
        ctrlClick(2);check("advanced_ctrl_toggle",graph.selectedKnots.size()==1);ctrlClick(2);
        const auto old=panel.editingValues();from=graph.knotPoint(1);to=from+juce::Point<float>(12,-10);
        graph.mouseDown(event(from,from));graph.mouseDrag(event(to,from));graph.mouseUp(event(to,from));
        const auto translated=panel.editingValues();
        check("advanced_group_shared_time_delta",translated.knots[0].time>old.knots[0].time
            &&near(translated.knots[0].time-old.knots[0].time,translated.knots[1].time-old.knots[1].time));
        check("advanced_group_shared_gain_delta",translated.knots[0].gain>old.knots[0].gain
            &&near(translated.knots[0].gain-old.knots[0].gain,translated.knots[1].gain-old.knots[1].gain));
        check("advanced_group_unselected_unchanged",translated.knots[2]==old.knots[2]);
        graph.keyPressed(juce::KeyPress('R',juce::ModifierKeys::ctrlModifier,0));
        check("advanced_group_reset_defaults",near(panel.editingValues().knots[0].gain,1)&&near(panel.editingValues().knots[1].gain,1)
            &&near(panel.editingValues().knots[0].time,translated.knots[0].time));
        resetSelection();click(graph.knotPoint(1));ctrlClick(2);del();
        check("advanced_group_delete",panel.editingValues().knots.size()==1&&near(panel.editingValues().knots[0].time,.8));
        resetSelection();click(graph.knotPoint(1));ctrlClick(2);from=graph.knotPoint(1);to=from+juce::Point<float>(1000,-1000);
        graph.mouseDown(event(from,from));graph.mouseDrag(event(to,from));graph.mouseUp(event(to,from));
        const auto limited=panel.editingValues();
        check("advanced_group_boundary_preserves_spacing",near(limited.knots[1].time-limited.knots[0].time,.3)&&limited.knots[1].time<limited.knots[2].time);
        resetSelection();click(graph.knotPoint(0));ctrlClick(1);ctrlClick(4);del();
        check("advanced_group_preserves_endpoints",panel.editingValues().knots.size()==2);

    }
    const backend::TailFadeSettings settings{.2,.8,.8,.15,2.0};
    check("settings_validate",settings.valid());
    auto invalid=settings;invalid.endFraction=.1;check("reversed_range_rejected",!invalid.valid());invalid=settings;invalid.startGain=std::numeric_limits<double>::infinity();check("nonfinite_level_rejected",!invalid.valid());invalid=settings;invalid.endGain=1.2;check("rising_gain_rejected",!invalid.valid());
    check("gain_before_range_and_after_end",near(backend::tailFadeGain(2,-1,0,1,settings),.8)&&near(backend::tailFadeGain(2,1,0,1,settings),.15));
    check("linear_stays_straight_regardless_of_curve_power",near(backend::tailFadeGain(1,.5,0,1,settings),.8-(.8-.15)*.5));
    check("custom_s_curve_power_matches_value",near(backend::tailFadeGain(2,.5,0,1,settings),.8-(.8-.15)*.15625));
    check("custom_range_follows_tail_length",near(backend::tailFadeGain(1,3,2,4,settings),backend::tailFadeGain(1,.5,0,1,settings)));
    check("disabled_effect_is_unity",near(backend::tailFadeGain(0,.5,0,1,settings),1));
    backend::TailFadeSettings bezier;bezier.customCurve=true;
    bool legacyMatch=true,monotone=true;float previous=1;
    for(int i=0;i<=1000;++i){const auto t=i/1000.0;const auto gain=backend::tailFadeGain(2,t,0,1,bezier);
        legacyMatch=legacyMatch&&near(gain,backend::tailFadeGain(2,t,0,1));monotone=monotone&&gain<=previous+1.e-6f;previous=gain;}
    check("bezier_s_matches_legacy_and_monotone",legacyMatch&&monotone);
    bezier.control1Time=0;bezier.control2Time=0;bezier.control1Progress=.2;bezier.control2Progress=.8;
    check("bezier_vertical_tangents_finite",std::isfinite(backend::tailFadeGain(2,.00001,0,1,bezier)));
    auto badBezier=bezier;badBezier.control1Time=.9;check("bezier_reversed_controls_rejected",!badBezier.valid());
    bezier.control1Time=.15;bezier.control2Time=.8;bezier.control1Progress=.7;bezier.control2Progress=.95;
    check("bezier_time_is_inverted",backend::tailFadeGain(2,.5,0,1,bezier)<.3);
    AdvancedEnvelopePanel customPanel({{"custom","Bezier",{0,1},{{0,0,true},{1,0,false}}, {}}},1,2,bezier,false,folder.getChildFile("presets-"+juce::Uuid().toString()+".json"));
    check("shape_menu_has_line_curve_and_mixed",customPanel.diagnosticShapeCount()==3);
    customPanel.setValues(2,settings);check("unified_curve_keeps_legacy_s_exact",customPanel.values()==settings&&customPanel.mode()==2);
    customPanel.diagnosticCurveControl(1,20);check("legacy_s_handles_become_editable_curve",customPanel.values().customCurve&&near(customPanel.values().control1Progress,.2)&&near(customPanel.values().curvePower,1));
    customPanel.diagnosticSelectShape(1);customPanel.diagnosticSelectShape(2);check("curve_menu_opens_handle_editor",customPanel.mode()==2&&customPanel.values().customCurve);
    customPanel.setValues(2,bezier);
    check("custom_curve_initial_values",customPanel.values().customCurve&&near(customPanel.values().control1Time,.15));
    customPanel.diagnosticDragCurveHandle(1,5,8);check("drag_control_changes_both_coordinates",customPanel.values().control1Time>.15&&customPanel.values().control1Progress>.7);
    customPanel.diagnosticCurveControl(0,99);check("control_time_clamps_to_order",customPanel.values().valid()&&near(customPanel.values().control1Time,customPanel.values().control2Time));
    customPanel.diagnosticPreset(3);check("fast_first_preset",backend::tailFadeGain(2,.5,0,1,customPanel.values())<.3);
    customPanel.diagnosticPreset(4);check("slow_first_preset",backend::tailFadeGain(2,.5,0,1,customPanel.values())>.7);
    customPanel.diagnosticPreset(5);check("uniform_bezier_preset",near(backend::tailFadeGain(2,.3,0,1,customPanel.values()),.7));
    customPanel.diagnosticPreset(2);customPanel.setSize(740,660);check("custom_controls_fit_minimum_size",customPanel.diagnosticControlsFit());
    {auto out=folder.getChildFile("custom-bezier-controls.png").createOutputStream();juce::PNGImageFormat png;check("custom_controls_snapshot",out&&png.writeImageToStream(customPanel.createComponentSnapshot(customPanel.getLocalBounds(),true,1.5f),*out));}
    // View transforms must not change the draft; real mouse drags use the inverse mapping.
    for(const bool phoneme:{false,true})
    {
        backend::TailFadeSettings v;v.mixed=true;v.allowRise=true;v.knots={{.5,.6,false}};
        AdvancedEnvelopePanel zoomPanel({{"zoom","Zoom",{0,1},{{0,0,true},{1,0,true}}, {}}},1,2,v,false,{},true,phoneme);
        auto& graph=zoomPanel.graph;const auto original=zoomPanel.values();int applies=0;
        zoomPanel.onApply=[&](int,const auto&,bool){++applies;return juce::String();};
        {
            const auto at=graph.plot().getRelativePoint(.27f,.63f);
            const auto wheel=[&](int flags,float dx,float dy,bool reversed=false){
                const juce::MouseEvent e(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(flags),
                    0,0,0,0,0,&graph,&graph,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),0,false);
                juce::MouseWheelDetails d{};d.deltaX=dx;d.deltaY=dy;d.isReversed=reversed;graph.mouseWheelMove(e,d);
            };
            graph.setZoom(4,4);
            const auto time=graph.timeAt(at.x,*zoomPanel.current());const auto gain=graph.gainAtY(at.y);
            wheel(juce::ModifierKeys::ctrlModifier,0,.2f);
            check("wheel_ctrl_zooms_only_gain_at_cursor",graph.zoomY>4&&near(graph.zoomX,4)&&near(graph.gainAtY(at.y),gain)&&near(graph.timeAt(at.x,*zoomPanel.current()),time));
            wheel(juce::ModifierKeys::ctrlModifier,0,.2f,true);
            check("wheel_reversed_zoom_restores_scale",near(graph.zoomY,4));
            wheel(juce::ModifierKeys::ctrlModifier|juce::ModifierKeys::shiftModifier,0,.2f);
            check("wheel_ctrl_shift_zooms_only_time_at_cursor",graph.zoomX>4&&near(graph.zoomY,4)&&near(graph.timeAt(at.x,*zoomPanel.current()),time));
            const auto x=graph.viewX,y=graph.viewY;
            wheel(0,0,.2f);check("wheel_plain_scrolls_vertically",graph.viewY<y&&near(graph.viewX,x));
            const auto afterY=graph.viewY;wheel(juce::ModifierKeys::shiftModifier,0,.2f);
            check("wheel_shift_scrolls_horizontally",graph.viewX<x&&near(graph.viewY,afterY));
            const auto afterX=graph.viewX;wheel(0,-.1f,0);
            check("wheel_horizontal_trackpad_scrolls_time",graph.viewX>afterX&&near(graph.viewY,afterY));
            graph.setZoom(32,16);const auto maxX=graph.viewX,maxY=graph.viewY;
            wheel(juce::ModifierKeys::ctrlModifier,0,1);wheel(juce::ModifierKeys::ctrlModifier|juce::ModifierKeys::shiftModifier,0,1);
            check("wheel_zoom_upper_limits_do_not_drift",near(graph.zoomX,32)&&near(graph.zoomY,16)&&near(graph.viewX,maxX)&&near(graph.viewY,maxY));
            graph.fit.onClick();wheel(juce::ModifierKeys::ctrlModifier,0,-1);wheel(juce::ModifierKeys::ctrlModifier|juce::ModifierKeys::shiftModifier,0,-1);
            check("wheel_minimum_zoom_and_document_unchanged",near(graph.zoomX,1)&&near(graph.zoomY,1)&&near(graph.viewX,0)&&near(graph.viewY,0)&&zoomPanel.values()==original&&applies==0);
        }
        graph.xPlus.onClick();graph.xPlus.onClick();graph.yPlus.onClick();
        check("zoom_axes_independent_and_view_only",near(graph.zoomX,4)&&near(graph.zoomY,2)&&zoomPanel.values()==original&&applies==0);
        graph.horizontal.setCurrentRangeStart(.3,juce::sendNotificationSync);
        graph.vertical.setCurrentRangeStart(.2,juce::sendNotificationSync);
        check("zoom_scrollbars_move_visible_window",near(graph.viewX,.3)&&near(graph.viewY,.2));
        const auto from=graph.knotPoint(1);
        const auto to=juce::Point<float>{graph.x(.55,*zoomPanel.current()),graph.y(.66f)};
        const auto event=[&](juce::Point<float> pos){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),pos,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&graph,&graph,juce::Time::getCurrentTime(),from,juce::Time::getCurrentTime(),1,pos!=from);};
        graph.mouseDown(event(from));graph.mouseDrag(event(to));graph.mouseUp(event(to));
        check("zoomed_knot_drag_matches_time_and_gain",near(zoomPanel.values().knots[0].time,.55)&&near(zoomPanel.values().knots[0].gain,.66)&&applies==0);
        const auto edited=zoomPanel.values();graph.fit.onClick();
        check("fit_restores_both_axes_without_resetting_envelope",near(graph.zoomX,1)&&near(graph.zoomY,1)&&near(graph.viewX,0)&&near(graph.viewY,0)&&zoomPanel.values()==edited);
        graph.setZoom(32,16);graph.horizontal.setCurrentRangeStart(1,juce::sendNotificationSync);graph.vertical.setCurrentRangeStart(1,juce::sendNotificationSync);
        check("zoom_pan_clamps_to_note_end_and_zero_gain",near(graph.visibleLast(*zoomPanel.current()),1)&&near(graph.gainAtY(graph.plot().getBottom()),0));
        graph.setZoom(4,2);graph.horizontal.setCurrentRangeStart(.38,juce::sendNotificationSync);graph.vertical.setCurrentRangeStart(.2,juce::sendNotificationSync);
        auto output=folder.getChildFile(phoneme?"zoomed-phoneme.png":"zoomed-native.png").createOutputStream();juce::PNGImageFormat png;
        check("zoom_panel_snapshot",output&&png.writeImageToStream(zoomPanel.createComponentSnapshot(zoomPanel.getLocalBounds(),true,1.0f),*output));
    }
    customPanel.setValues(2,bezier);customPanel.graph.fit.onClick();customPanel.diagnosticDragCurveHandle(1,4,3);const auto unzoomedEdit=customPanel.values();
    customPanel.setValues(2,bezier);customPanel.graph.setZoom(4,2);customPanel.diagnosticDragCurveHandle(1,16,6);
    check("zoomed_bezier_drag_matches_unzoomed_gesture",near(customPanel.values().control1Time,unzoomedEdit.control1Time)&&near(customPanel.values().control1Progress,unzoomedEdit.control1Progress));
    customPanel.graph.fit.onClick();
    customPanel.setValues(1,settings);customPanel.graph.setZoom(4,1);
    customPanel.graph.horizontal.setCurrentRangeStart(.1,juce::sendNotificationSync);
    {
        auto& graph=customPanel.graph;const auto start=juce::Point<float>{graph.x(.2,*customPanel.current()),graph.plot().getY()};
        const auto target=juce::Point<float>{graph.x(.25,*customPanel.current()),start.y};
        const auto event=[&](juce::Point<float> pos){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),pos,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&graph,&graph,juce::Time::getCurrentTime(),start,juce::Time::getCurrentTime(),1,pos!=start);};
        graph.mouseDown(event(start));graph.mouseDrag(event(target));graph.mouseUp(event(target));
        check("zoomed_boundary_drag_uses_visible_time_range",near(customPanel.values().startFraction,.25)&&near(customPanel.values().endFraction,.8));
        graph.setZoom(32,16);graph.horizontal.setCurrentRangeStart(.2,juce::sendNotificationSync);
        graph.horizontal.moveScrollbarInSteps(1,juce::sendNotificationSync);
        check("high_zoom_scroll_step_is_fraction_of_visible_span",near(graph.viewX,.2+.1/32));
        graph.fit.onClick();
    }
    const auto presetDraft=settings;auto customDraft=presetDraft;customDraft.customCurve=true;customDraft.control1Time=.15;customDraft.control1Progress=.7;customDraft.control2Time=.8;customDraft.control2Progress=.95;
    customPanel.setValues(2,customDraft);int presetApplies=0;customPanel.onApply=[&](int,const auto&,bool){++presetApplies;return juce::String();};
    check("save_named_preset",customPanel.diagnosticSavePreset(juce::String::fromUTF8("轻柔尾音")));
    check("duplicate_and_empty_names_rejected",!customPanel.diagnosticSavePreset(juce::String::fromUTF8("轻柔尾音"))&&!customPanel.diagnosticSavePreset("   ")&&customPanel.diagnosticPresetCount()==1);
    customPanel.setValues(1,{});customPanel.diagnosticPreset(100);
    check("preset_restores_full_draft_without_applying",customPanel.mode()==2&&customPanel.values()==customDraft&&presetApplies==0);
    customPanel.diagnosticApply();check("preset_apply_is_explicit",presetApplies==1);
    customPanel.diagnosticDeletePreset();check("delete_preset_keeps_current_draft",customPanel.diagnosticPresetCount()==0&&customPanel.values()==customDraft);
    const auto libraryFile=folder.getChildFile("library-"+juce::Uuid().toString()+".json");TailFadePresetStore library(libraryFile);juce::String libraryError;
    TailFadePreset savedPreset;savedPreset.name=juce::String::fromUTF8("副歌淡出");savedPreset.shape=3;savedPreset.enabled=false;savedPreset.settings=customDraft;
    check("library_save",library.add(savedPreset,libraryError));std::vector<TailFadePreset> reopened;
    check("library_survives_new_instance",TailFadePresetStore(libraryFile).load(reopened,libraryError)&&reopened.size()==1&&reopened[0].settings==customDraft&&reopened[0].shape==3&&!reopened[0].enabled);
    AdvancedEnvelopePanel reopenedPanel({},1,1,{},false,libraryFile);reopenedPanel.diagnosticPreset(100);
    check("reopened_panel_restores_disabled_custom_shape",reopenedPanel.mode()==0&&reopenedPanel.values()==customDraft);
    savedPreset.name="Linear";savedPreset.shape=1;savedPreset.enabled=true;savedPreset.settings=settings;
    TailFadePreset legacyPreset;legacyPreset.name="Legacy S";legacyPreset.shape=2;legacyPreset.settings=settings;
    const auto legacyFile=folder.getChildFile("legacy-presets-"+juce::Uuid().toString()+".json");
    check("legacy_s_preset_fixture",TailFadePresetStore(legacyFile).add(legacyPreset,libraryError));
    AdvancedEnvelopePanel legacyPanel({},1,1,{},false,legacyFile);legacyPanel.diagnosticPreset(100);
    check("legacy_s_preset_maps_to_unified_curve",legacyPanel.diagnosticShapeCount()==3&&legacyPanel.mode()==2&&legacyPanel.values()==settings);
    check("library_second_writer_preserves_first_preset",TailFadePresetStore(libraryFile).add(savedPreset,libraryError)&&library.load(reopened,libraryError)&&reopened.size()==2);
    check("library_delete_preserves_other_preset",library.remove(savedPreset.id,libraryError)&&library.load(reopened,libraryError)&&reopened.size()==1);
    libraryFile.replaceWithText("broken json");const auto broken=libraryFile.loadFileAsString();savedPreset.name="new";
    check("corrupt_library_is_not_overwritten",!library.add(savedPreset,libraryError)&&libraryFile.loadFileAsString()==broken);
    customPanel.diagnosticSavePreset(juce::String::fromUTF8("轻柔尾音"));customPanel.setSize(740,660);
    {auto out=folder.getChildFile("custom-preset-library.png").createOutputStream();juce::PNGImageFormat png;check("preset_library_snapshot",out&&png.writeImageToStream(customPanel.createComponentSnapshot(customPanel.getLocalBounds(),true,1.5f),*out));}
    const auto data=project.snapshot();ProjectModel model;model.resetDocument(data);const auto original=model.contentFingerprint();
    check("detailed_multi_note_apply",model.setNotesTailFade({"n2","n4"},1,settings));auto changed=model.snapshot();
    check("only_target_notes_receive_parameters",changed.tracks[0].clips[0].notes[0].utauTailFade==settings&&changed.tracks[0].clips[0].notes[2].utauTailFade==settings&&changed.tracks[0].clips[0].notes[1].utauTailFade==backend::TailFadeSettings{});
    check("detailed_apply_preserves_flags_and_sparse_envelope",changed.tracks[0].clips[0].notes[0].utauFlags==data.tracks[0].clips[0].notes[0].utauFlags&&changed.tracks[0].clips[0].notes[0].amplitudeEnvelope.size()==4);
    check("all_parameters_undo_together",model.undo()&&model.contentFingerprint()==original&&!model.canUndo());model.redo();
    const auto before=model.contentFingerprint();check("invalid_settings_do_not_mutate",!model.setNotesTailFade({"n2"},1,invalid)&&before==model.contentFingerprint());
    juce::String error;const auto file=folder.getChildFile("detailed-tail.hjpx");check("detailed_project_saved",model.save(file,error));ProjectModel loaded;check("detailed_project_loaded",loaded.load(file,error));
    check("all_parameters_roundtrip",loaded.snapshot().tracks[0].clips[0].notes[0].utauTailFade==settings&&loaded.snapshot().tracks[0].clips[0].notes[0].utauTailFadeMode==1);
    model.setNotesTailFade({"n2"},0);check("disable_keeps_settings_for_reenable",model.snapshot().tracks[0].clips[0].notes[0].utauTailFade==settings);model.setNotesTailFade({"n2"},2);check("shape_switch_keeps_parameters",model.snapshot().tracks[0].clips[0].notes[0].utauTailFade==settings);
    const auto beforeBezier=model.contentFingerprint();model.setNotesTailFade({"n2"},2,bezier);
    check("bezier_saved",model.save(folder.getChildFile("bezier.hjpx"),error));ProjectModel bezierLoaded;
    check("bezier_roundtrip",bezierLoaded.load(folder.getChildFile("bezier.hjpx"),error)&&bezierLoaded.snapshot().tracks[0].clips[0].notes[0].utauTailFade==bezier);
    check("bezier_undo",model.undo()&&model.contentFingerprint()==beforeBezier);
    const auto plain=data.tracks[0].clips[0].notes[0];auto edited=plain;edited.utauTailFade=settings;
    check("detailed_settings_invalidate_mix_not_raw_cache",AudioEngine::utauNoteRenderHash(plain)!=AudioEngine::utauNoteRenderHash(edited)&&AudioEngine::utauNoteAudioHash(plain)==AudioEngine::utauNoteAudioHash(edited));
    edited=plain;edited.utauTailFade=bezier;
    check("bezier_invalidate_mix_only",AudioEngine::utauNoteRenderHash(plain)!=AudioEngine::utauNoteRenderHash(edited)&&AudioEngine::utauNoteAudioHash(plain)==AudioEngine::utauNoteAudioHash(edited));
    const std::vector<AmplitudeEnvelopePoint> curve{{0,-3,false},{.1,0,true},{.7,-3,false},{1.2,-12,false}};
    const auto picture=backend::tailFadePicture(curve,2,.4,1.2,settings);
    const auto expected=[&](double t){return backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(curve,std::min(t,.56)))*backend::tailFadeGain(2,t,.4,1.2,settings);};
    check("preview_matches_actual_level_during_fade",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(picture,.85)),expected(.85)));
    check("preview_holds_single_tail_end_level",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(picture,1.15)),expected(1.15)));
    check("silent_base_stays_silent",near(backend::tailFadeGain(2,.5,0,1,settings)*backend::envelopeGainFromDb(-60),0));
    // An original steep fall used to bend the final part of a supposedly straight ending.
    const std::vector<AmplitudeEnvelopePoint> steep{{0,0,true},{.8,0,false},{1,-60,false}};
    const auto straight=backend::tailFadePicture(steep,1,.4,1.0);
    const auto smooth=backend::tailFadePicture(steep,2,.4,1.0);
    bool lineOk=true,curveOk=true;
    for(double t=.4;t<1;t+=.01)
    {
        const auto u=(t-.4)/.6;
        lineOk=lineOk&&near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(straight,t)),1-u);
        // Stored display points clamp below -60 dB to silence, unlike the float mixer.
        curveOk=curveOk&&std::abs(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(smooth,t))-(1-u*u*(3-2*u)))<.0011;
        curveOk=curveOk&&near(backend::tailFadeEnvelopeGain(2,t,.4,1.0,{},[](double at){return at<=.8?1.0f:0.0f;}),1-u*u*(3-2*u));
    }
    check("linear_ending_does_not_bend_at_original_fall",lineOk);
    check("smooth_ending_is_one_curve_through_original_fall",curveOk);
    check("original_envelope_unchanged_when_disabled",backend::tailFadePicture(steep,0,.4,1).size()==steep.size());
    for(int mode=1;mode<=2;++mode)
    {
        AdvancedEnvelopePanel sample({{"test","single ending",{.4,1.0},steep,{}}},1,mode,{},false);
        auto out=folder.getChildFile(mode==1?"single-straight-ending.png":"single-curved-ending.png").createOutputStream();
        juce::PNGImageFormat png;check("single_ending_snapshot",out&&png.writeImageToStream(sample.createComponentSnapshot(sample.getLocalBounds(),true,1.0f),*out));
    }
    // Exercise the real toolbar dialog, including the draft/Apply boundary and revision guard.
    showAdvancedEnvelopeMenu();auto* window=dynamic_cast<juce::DialogWindow*>(juce::ModalComponentManager::getInstance()->getModalComponent(0));
    auto* panel=window?dynamic_cast<AdvancedEnvelopePanel*>(window->getContentComponent()):nullptr;
    check("toolbar_opens_full_tail_fade_dialog",panel!=nullptr);
    if(panel)
    {
        for(int index=0;index<3;++index)
        {
            panel->diagnosticSelectPreview(index);
            const auto regions=panel->diagnosticPreviewRegions();
            const auto edges=pianoRoll.diagnosticRegionEdges((size_t)index);
            const auto absolute=1.0+index*1.4;
            bool aligned=regions.size()==(size_t)(index+2);
            for(size_t i=0;i+1<regions.size();++i)aligned=aligned&&near(regions[i].endSeconds+absolute,edges[i]);
            check("preview_uses_actual_region_count_and_boundaries",aligned);
            check("preview_regions_include_consonant_lead_in",!regions.empty()&&near(regions.front().startSeconds,-.1));
        }
        panel->diagnosticSelectPreview(2);panel->setValues(0,{});
        check("disabled_fade_keeps_four_region_guides",panel->diagnosticPreviewRegions().size()==4);
        {
            juce::PNGImageFormat png;auto out=folder.getChildFile("tail-fade-disabled-four-regions.png").createOutputStream();
            check("disabled_fade_four_regions_snapshot",out&&png.writeImageToStream(panel->createComponentSnapshot(panel->getLocalBounds(),true,1.0f),*out));
        }
        const auto prior=project.contentFingerprint();panel->setValues(2,settings);panel->diagnosticSetControl(4,1.5);
        check("controls_are_draft_until_apply",project.contentFingerprint()==prior);
        panel->diagnosticSetControl(0,95);check("range_controls_keep_order",panel->values().valid()&&panel->values().startFraction<panel->values().endFraction);
        panel->setValues(2,settings);panel->diagnosticSetControl(3,180);check("end_level_cannot_exceed_start",panel->values().valid()&&panel->values().endGain==panel->values().startGain);
        panel->setValues(2,settings);panel->setSize(820,680);check("default_dialog_controls_fit",panel->diagnosticControlsFit());
        juce::PNGImageFormat png;auto out=folder.getChildFile("tail-fade-dialog.png").createOutputStream();check("full_dialog_preview_written",out&&png.writeImageToStream(panel->createComponentSnapshot(panel->getLocalBounds(),true,1.0f),*out));
        panel->setSize(740,660);check("minimum_dialog_controls_fit",panel->diagnosticControlsFit());auto small=folder.getChildFile("tail-fade-dialog-small.png").createOutputStream();check("minimum_dialog_preview_written",small&&png.writeImageToStream(panel->createComponentSnapshot(panel->getLocalBounds(),true,1.0f),*small));
        panel->diagnosticApply();check("dialog_apply_updates_all_selected",project.snapshot().tracks[0].clips[0].notes[2].utauTailFade==settings&&project.snapshot().tracks[0].clips[0].notes[0].utauTailFade==settings);
        panel->setValues(1,{});panel->diagnosticApply();check("repeat_dialog_apply_uses_new_revision",project.snapshot().tracks[0].clips[0].notes[2].utauTailFade==backend::TailFadeSettings{});
        project.setNoteGain("n2",.7f);const auto newer=project.contentFingerprint();panel->setValues(2,settings);panel->diagnosticApply();check("stale_dialog_does_not_overwrite_external_edit",project.contentFingerprint()==newer);window->exitModalState(0);
    }
    // Head envelope follows the actual first OTO region and can run with the tail off.
    for(int n=2;n<=4;++n)
    {
        const auto head=pianoRoll.headEnvelopeSpanFor("n"+juce::String(n));const auto regions=pianoRoll.otoRegionGuidesFor("n"+juce::String(n));
        check("head_span_matches_first_of_2_3_4_regions",head&&!regions.empty()&&near(head->startSeconds,regions.front().startSeconds)&&near(head->endSeconds,regions.front().endSeconds));
    }
    backend::TailFadeSettings both=bezier;both.head.mode=2;both.head.customCurve=true;both.head.startGain=.1;both.head.endGain=.9;
    both.head.control1Time=.2;both.head.control1Progress=.4;both.head.control2Time=.7;both.head.control2Progress=.8;
    auto headLinear=both.head;headLinear.mode=1;headLinear.startGain=0;headLinear.endGain=1;
    check("head_linear_replaces_original_attack",near(backend::headEnvelopeGain(-.05,-.1,0,headLinear,[](double t){return t<-.01?.01f:1.0f;}),.5));
    check("head_leaves_body_at_end_multiplier",near(backend::headEnvelopeGain(.5,-.1,0,both.head,[](double){return .8f;}),.72));
    backend::UtauSampleTiming classicHead;classicHead.preutteranceSeconds=.1;classicHead.consonantSeconds=.08;
    const auto pinnedHead=backend::UtauRenderer::headEnvelopeSpan(classicHead,-.2,1,1,100,false,false);
    check("classic_head_follows_pinned_preutterance",pinnedHead&&near(pinnedHead->startSeconds,-.2)&&near(pinnedHead->endSeconds,-.04));
    backend::TailFadeSettings headOnly;headOnly.head=headLinear;
    const std::vector<AmplitudeEnvelopePoint> headBase{{-.1,-60,true},{-.01,0,true},{1,0,true}};
    const auto headPicture=backend::tailFadePicture(headBase,0,.7,1,headOnly,-.1,0);
    check("head_only_piano_preview_matches_audio",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(headPicture,-.05)),.5));
    check("head_zero_span_keeps_original",near(backend::headEnvelopeGain(0,0,0,both.head,[](double){return .6f;}),.6));
    auto badHead=both;badHead.head.endFraction=badHead.head.startFraction;check("head_invalid_range_rejected",!badHead.valid());
    auto headNote=plain;headNote.utauTailFadeMode=0;headNote.utauTailFade=both;
    check("head_only_invalidates_mix_not_raw",AudioEngine::utauNoteRenderHash(headNote)!=AudioEngine::utauNoteRenderHash(plain)&&AudioEngine::utauNoteAudioHash(headNote)==AudioEngine::utauNoteAudioHash(plain));
    ProjectModel headModel;headModel.resetDocument(data);const auto beforeHead=headModel.contentFingerprint();
    check("head_tail_batch_apply",headModel.setNotesTailFade({"n2","n4"},2,both));
    check("head_tail_single_undo",headModel.undo()&&headModel.contentFingerprint()==beforeHead);headModel.redo();
    check("head_tail_save",headModel.save(folder.getChildFile("head-tail.hjpx"),error));ProjectModel headLoaded;
    check("head_tail_roundtrip",headLoaded.load(folder.getChildFile("head-tail.hjpx"),error)&&headLoaded.snapshot().tracks[0].clips[0].notes[0].utauTailFade==both);
    const auto headLibrary=folder.getChildFile("head-library-"+juce::Uuid().toString()+".json");
    AdvancedEnvelopePanel headPanel({{"head","Head + tail",{.7,1.2},{{-.1,-60,true},{-.08,0,true},{1.2,0,false}}, {},backend::UtauTailFadeSpan{-.1,0}}},1,2,both,false,headLibrary);
    headPanel.diagnosticSection(true);check("switch_to_head_keeps_tail",headPanel.values()==both&&headPanel.mode()==2);
    headPanel.diagnosticSetControl(2,20);headPanel.diagnosticSetControl(3,110);const auto editedBoth=headPanel.values();
    check("head_allows_rising_gain",editedBoth.valid()&&near(editedBoth.head.startGain,.2)&&near(editedBoth.head.endGain,1.1)&&editedBoth.startGain==both.startGain);
    headPanel.diagnosticSection(false);check("switch_to_tail_keeps_head",headPanel.values()==editedBoth);headPanel.diagnosticSection(true);
    check("head_tail_preset_save",headPanel.diagnosticSavePreset("Both ends"));AdvancedEnvelopePanel recalled({},1,0,{},false,headLibrary);recalled.diagnosticPreset(100);
    check("head_tail_preset_recall",recalled.mode()==2&&recalled.values()==editedBoth);
    headPanel.setSize(740,660);check("head_panel_controls_fit",headPanel.diagnosticControlsFit());
    {auto out=folder.getChildFile("head-envelope-panel.png").createOutputStream();juce::PNGImageFormat png;check("head_preview_written",out&&png.writeImageToStream(headPanel.createComponentSnapshot(headPanel.getLocalBounds(),true,1.5f),*out));}
    // Version-1 presets must still load with the head disabled.
    const auto oldLibrary=folder.getChildFile("old-library-"+juce::Uuid().toString()+".json");auto oldRoot=juce::JSON::parse(headLibrary.loadFileAsString());
    oldRoot.getDynamicObject()->setProperty("version",1);oldLibrary.replaceWithText(juce::JSON::toString(oldRoot));std::vector<TailFadePreset> oldRows;
    check("old_presets_default_head_off",TailFadePresetStore(oldLibrary).load(oldRows,error)&&!oldRows.empty()&&oldRows[0].settings.head.mode==0);
    // Test the common native/external mixer callback, not just the mathematical helper.
    backend::UtauRenderRequest request;request.voicebankDirectory=data.tracks[0].voicebankDirectory;request.fourRegion=true;request.consonantClasses=true;request.targetDurationSeconds=2.4;
    backend::UtauNoteRenderSpec spec;spec.alias="a";spec.startSeconds=.5;spec.durationSeconds=1.2;spec.oto=data.tracks[0].clips[0].notes[2].utauOto;spec.amplitudeEnvelope={{0,0,true},{.6,0,false},{1.2,-60,false}};request.notes={spec};
    const auto timing=backend::UtauRenderer::sampleTiming(request.voicebankDirectory,"a",60,100,true,true,&spec.oto);
    const auto span=timing?backend::UtauRenderer::tailFadeSpan(*timing,-.1,1.2,1.2,100,true,true):std::nullopt;check("custom_render_tail_resolves",span.has_value());
    if(span)
    {
        const auto headSpan=backend::UtauRenderer::headEnvelopeSpan(*timing,-.1,1.2,1.2,100,true,true);
        check("actual_render_head_resolves",headSpan.has_value());
        const std::array<double,4> times{headSpan?(headSpan->startSeconds+headSpan->endSeconds)*.5:-.05,span->startSeconds-.02,span->startSeconds+(span->endSeconds-span->startSeconds)*.5,span->startSeconds+(span->endSeconds-span->startSeconds)*.9};
        std::array<float,4> reference{},actual{};
        for(int pass=0;pass<4;++pass)
        {
            request.notes[0].tailFadeMode=pass==0||pass==3?0:2;request.notes[0].tailFadeSettings=pass>=2?both:settings;
            request.notePiece=[&](size_t,const juce::AudioBuffer<float>&,double,double,const std::function<float(double)>& gain,const std::function<float(double)>&){for(size_t i=0;i<times.size();++i)(pass==0?reference:actual)[i]=gain(times[i]);};
            const auto rendered=backend::UtauRenderer::render(request);check("custom_actual_renderer_produces_audio",rendered.buffer.getNumSamples()>0);
            if(pass>0)for(size_t i=0;i<times.size();++i)check(pass==3?"head_only_actual_mixer_matches_preview":pass==2?"bezier_actual_mixer_matches_preview":"smooth_actual_mixer_matches_preview",near(actual[i],backend::advancedEnvelopeGain(request.notes[0].tailFadeMode,times[i],span->startSeconds,span->endSeconds,headSpan?headSpan->startSeconds:0,headSpan?headSpan->endSeconds:0,request.notes[0].tailFadeSettings,[&](double t){
                return backend::envelopeGainFromDb(t<=.6?0.0f:backend::envelopeDbBetween(0,-60,(float)((t-.6)/.6),false));})));
        }
        for(size_t i=0;i<times.size();++i)check("custom_actual_mixer_matches_preview",near(actual[i],backend::advancedEnvelopeGain(0,times[i],span->startSeconds,span->endSeconds,headSpan?headSpan->startSeconds:0,headSpan?headSpan->endSeconds:0,both,[&](double t){
            const auto db=t<=.6?0.0f:backend::envelopeDbBetween(0,-60,(float)((t-.6)/.6),false);
            return backend::envelopeGainFromDb(db);
        })));
    }

    backend::TailFadeSettings mixedSettings;mixedSettings.mixed=true;
    mixedSettings.knots={{.25,.8,true},{.75,.2,false}};
    mixedSettings.head.mode=2;mixedSettings.head.mixed=true;
    mixedSettings.head.knots={{.4,1.3,false},{.8,.7,true}};
    check("mixed_settings_valid",mixedSettings.valid());
    check("mixed_line_segment_exact",near(backend::tailFadeGain(2,.125,0,1,mixedSettings),.9));
    check("mixed_curve_segment_exact",near(backend::tailFadeGain(2,.375,0,1,mixedSettings),.70625));
    check("mixed_last_line_exact",near(backend::tailFadeGain(2,.875,0,1,mixedSettings),.1));
    check("mixed_knot_continuity",near(backend::tailFadeGain(2,.25-1.e-7,0,1,mixedSettings),backend::tailFadeGain(2,.25+1.e-7,0,1,mixedSettings)));
    check("mixed_scales_with_oto_span",near(backend::tailFadeGain(2,2.75,2,4,mixedSettings),.70625));
    check("mixed_head_can_rise_and_fall",near(backend::headEnvelopeGain(.4,0,1,mixedSettings.head,[](double){return 1.f;}),1.3));
    auto badMixed=mixedSettings;badMixed.knots[1].time=.1;check("mixed_unsorted_points_rejected",!badMixed.valid());
    badMixed=mixedSettings;badMixed.knots[0].gain=std::numeric_limits<double>::quiet_NaN();check("mixed_nonfinite_points_rejected",!badMixed.valid());
    const auto mixedLibrary=folder.getChildFile("mixed-library-"+juce::Uuid().toString()+".json");
    AdvancedEnvelopePanel mixedPanel({{"mixed","Mixed",{.4,1.2},{{-.1,0,true},{1.2,0,true}}, {},backend::UtauTailFadeSpan{-.1,.3}}},1,2,mixedSettings,false,mixedLibrary);
    check("mixed_panel_initial_state",mixedPanel.values()==mixedSettings&&mixedPanel.mode()==2);
    mixedPanel.diagnosticAddKnot(.5,1.4);check("mixed_add_point",mixedPanel.values().knots.size()==3&&near(mixedPanel.values().knots[1].gain,1.4));
    mixedPanel.diagnosticMoveKnot(2,.9,.6);check("mixed_drag_point_cannot_cross_neighbor",mixedPanel.values().knots[1].time<mixedPanel.values().knots[2].time&&mixedPanel.values().valid());
    mixedPanel.diagnosticSegment(0,true);check("mixed_independent_segment_type",mixedPanel.values().firstSegmentCurved&&mixedPanel.values().knots.back().curvedToNext==false);
    mixedPanel.diagnosticDeleteKnot(2);check("mixed_delete_interior_point",mixedPanel.values().knots.size()==2);
    mixedPanel.diagnosticDeleteKnot(0);check("mixed_endpoints_cannot_be_deleted",mixedPanel.values().knots.size()==2);
    const auto tailDraft=mixedPanel.values();mixedPanel.diagnosticSection(true);
    check("mixed_switch_head_preserves_tail",mixedPanel.values()==tailDraft);
    mixedPanel.diagnosticAddKnot(.6,.9);mixedPanel.diagnosticSegment(1,true);const auto mixedBoth=mixedPanel.values();
    check("mixed_head_edits_independent",mixedBoth.knots==tailDraft.knots&&mixedBoth.head.knots.size()==3);
    check("mixed_preset_save",mixedPanel.diagnosticSavePreset("Mixed head and tail"));
    AdvancedEnvelopePanel mixedRecall({},1,0,{},false,mixedLibrary);mixedRecall.diagnosticPreset(100);
    check("mixed_preset_roundtrip",mixedRecall.values()==mixedBoth&&mixedRecall.mode()==2);
    mixedPanel.diagnosticSection(false);mixedPanel.setSize(740,660);check("mixed_controls_fit",mixedPanel.diagnosticControlsFit());
    {auto out=folder.getChildFile("mixed-envelope-panel.png").createOutputStream();juce::PNGImageFormat png;check("mixed_snapshot",out&&png.writeImageToStream(mixedPanel.createComponentSnapshot(mixedPanel.getLocalBounds(),true,1.5f),*out));}
    ProjectModel mixedModel;mixedModel.resetDocument(data);const auto mixedBefore=mixedModel.contentFingerprint();
    check("mixed_apply_batch",mixedModel.setNotesTailFade({"n2","n4"},2,mixedBoth));
    check("mixed_undo",mixedModel.undo()&&mixedModel.contentFingerprint()==mixedBefore);mixedModel.redo();
    check("mixed_save",mixedModel.save(folder.getChildFile("mixed.hjpx"),error));ProjectModel mixedLoaded;
    check("mixed_project_roundtrip",mixedLoaded.load(folder.getChildFile("mixed.hjpx"),error)&&mixedLoaded.snapshot().tracks[0].clips[0].notes[0].utauTailFade==mixedBoth);
    edited=plain;edited.utauTailFade=mixedBoth;
    check("mixed_invalidates_mix_not_raw_cache",AudioEngine::utauNoteRenderHash(plain)!=AudioEngine::utauNoteRenderHash(edited)&&AudioEngine::utauNoteAudioHash(plain)==AudioEngine::utauNoteAudioHash(edited));
    auto changedSegment=edited;changedSegment.utauTailFade.firstSegmentCurved=!changedSegment.utauTailFade.firstSegmentCurved;
    check("mixed_segment_change_invalidates_cache",AudioEngine::utauNoteRenderHash(edited)!=AudioEngine::utauNoteRenderHash(changedSegment));
    auto tailOnly=mixedSettings;tailOnly.head.mode=0;
    const auto mixedPicture=backend::tailFadePicture(std::vector<AmplitudeEnvelopePoint>{{0,0,true},{1,0,true}},2,0,1,tailOnly);
    check("mixed_picture_matches_gain",near(backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(mixedPicture,.375)),.70625));
    if(span)
    {
        request.notes[0].tailFadeMode=2;request.notes[0].tailFadeSettings=tailOnly;
        bool callback=false,matched=true;
        request.notePiece=[&](size_t,const juce::AudioBuffer<float>&,double,double,const std::function<float(double)>& gain,const std::function<float(double)>&){
            callback=true;
            for(const auto u:{.125,.25,.375,.75,.875})
            {
                const auto t=span->startSeconds+(span->endSeconds-span->startSeconds)*u;
                const auto expected=backend::advancedEnvelopeGain(2,t,span->startSeconds,span->endSeconds,0,0,tailOnly,[](double at){return backend::envelopeGainFromDb(at<=.6?0.f:backend::envelopeDbBetween(0,-60,(float)((at-.6)/.6),false));});
                matched&=near(gain(t),expected);
            }
        };
        const auto rendered=backend::UtauRenderer::render(request);
        check("mixed_actual_mixer_matches_preview",callback&&matched&&rendered.buffer.getNumSamples()>0);
    }


    // Each mixed segment owns its cubic handles; old S curves use the same defaults.
    auto segmented=mixedSettings;segmented.firstSegmentCurved=true;
    segmented.firstBezier={.2,.4,.8,.9};segmented.knots[0].bezier={.1,.6,.2,.9};
    segmented.head.firstSegmentCurved=true;segmented.head.firstBezier={.15,.2,.7,.95};
    segmented.head.knots[0].bezier={.2,.35,.75,.9};segmented.head.knots[0].curvedToNext=true;
    check("segment_bezier_settings_valid",segmented.valid());
    // Parametric t=.5 gives x=.2375, y=.6875 for this cubic.
    check("segment_bezier_inverts_local_time",near(backend::tailFadeGain(2,.25+.5*.2375,0,1,segmented),.8-.6*.6875));
    check("segment_bezier_preserves_neighbor_line",near(backend::tailFadeGain(2,.875,0,1,segmented),.1));
    auto invalidBezier=segmented;invalidBezier.knots[0].bezier.control1Time=.9;
    check("segment_bezier_reversed_handles_rejected",!invalidBezier.valid());
    invalidBezier=segmented;invalidBezier.firstBezier.control1Progress=std::numeric_limits<double>::quiet_NaN();
    check("segment_bezier_nonfinite_handles_rejected",!invalidBezier.valid());
    auto oldMixed=backend::mixedEnvelopeToVar(mixedSettings);oldMixed.getDynamicObject()->removeProperty("firstBezier");
    for(auto& p:*oldMixed.getDynamicObject()->getProperty("knots").getArray())p.getDynamicObject()->removeProperty("bezier");
    backend::TailFadeSettings legacyMixed;
    check("legacy_mixed_s_controls_default",backend::mixedEnvelopeFromVar(oldMixed,legacyMixed)&&legacyMixed.firstBezier==backend::EnvelopeBezier{}
        &&near(backend::tailFadeGain(2,.375,0,1,legacyMixed),.70625));
    const auto segmentLibrary=folder.getChildFile("segment-bezier-library-"+juce::Uuid().toString()+".json");
    AdvancedEnvelopePanel segmentPanel({{"segment","Bezier segments",{.4,1.2},{{-.1,0,true},{1.2,0,true}}, {},backend::UtauTailFadeSpan{-.1,.3}}},1,2,segmented,false,segmentLibrary);
    segmentPanel.diagnosticSegment(1,true);segmentPanel.diagnosticCurveControl(1,50);
    check("segment_parameter_edits_only_selected_segment",near(segmentPanel.values().knots[0].bezier.control1Progress,.5)
        &&segmentPanel.values().firstBezier==segmented.firstBezier&&segmentPanel.values().head==segmented.head);
    const auto priorHandle=segmentPanel.values().knots[0].bezier;
    segmentPanel.diagnosticDragCurveHandle(1,3,3);
    check("segment_handle_drag_changes_time_and_progress",segmentPanel.values().knots[0].bezier.control1Time>priorHandle.control1Time
        &&segmentPanel.values().knots[0].bezier.control1Progress>priorHandle.control1Progress);
    const auto kept=segmentPanel.values();segmentPanel.diagnosticSegment(1,false);
    check("segment_line_keeps_handles",segmentPanel.values().knots[0].bezier==kept.knots[0].bezier);
    segmentPanel.diagnosticSegment(1,true);check("segment_curve_restores_handles",segmentPanel.values()==kept);
    segmentPanel.diagnosticSection(true);segmentPanel.diagnosticSegment(0,true);segmentPanel.diagnosticCurveControl(0,10);
    check("segment_head_edits_do_not_change_tail",segmentPanel.values().knots==kept.knots&&segmentPanel.values().firstBezier==kept.firstBezier
        &&near(segmentPanel.values().head.firstBezier.control1Time,.1));
    const auto allSegments=segmentPanel.values();
    check("segment_bezier_preset_save",segmentPanel.diagnosticSavePreset("Independent Bezier segments"));
    AdvancedEnvelopePanel segmentRecall({},1,0,{},false,segmentLibrary);segmentRecall.diagnosticPreset(100);
    check("segment_bezier_preset_roundtrip",segmentRecall.values()==allSegments);
    mixedModel.setNotesTailFade({"n2","n4"},2,allSegments);const auto beforeHandle=mixedModel.contentFingerprint();
    auto updatedHandle=allSegments;updatedHandle.firstBezier.control1Progress=.3;mixedModel.setNotesTailFade({"n2"},2,updatedHandle);
    check("segment_bezier_undo",mixedModel.undo()&&mixedModel.contentFingerprint()==beforeHandle);
    check("segment_bezier_project_save",mixedModel.save(folder.getChildFile("segment-bezier.hjpx"),error));ProjectModel segmentLoaded;
    check("segment_bezier_project_roundtrip",segmentLoaded.load(folder.getChildFile("segment-bezier.hjpx"),error)
        &&segmentLoaded.snapshot().tracks[0].clips[0].notes[0].utauTailFade==allSegments);
    edited.utauTailFade=allSegments;changedSegment=edited;changedSegment.utauTailFade.knots[0].bezier.control1Progress=.1;
    check("segment_bezier_invalidates_mix_only",AudioEngine::utauNoteRenderHash(edited)!=AudioEngine::utauNoteRenderHash(changedSegment)
        &&AudioEngine::utauNoteAudioHash(edited)==AudioEngine::utauNoteAudioHash(changedSegment));
    segmentPanel.diagnosticSection(false);segmentPanel.diagnosticSegment(1,true);segmentPanel.setSize(740,660);
    check("segment_bezier_minimum_layout_fits",segmentPanel.diagnosticControlsFit());
    {auto out=folder.getChildFile("segment-bezier-panel.png").createOutputStream();juce::PNGImageFormat png;check("segment_bezier_snapshot",out&&png.writeImageToStream(segmentPanel.createComponentSnapshot(segmentPanel.getLocalBounds(),true,1.5f),*out));}
    if(span)
    {
        request.notes[0].tailFadeMode=2;request.notes[0].tailFadeSettings=segmented;request.notes[0].tailFadeSettings.head.mode=0;
        bool called=false,matched=true;
        request.notePiece=[&](size_t,const juce::AudioBuffer<float>&,double,double,const std::function<float(double)>& gain,const std::function<float(double)>&){
            called=true;const auto base=backend::envelopeGainFromDb(span->startSeconds<=.6?0.f:backend::envelopeDbBetween(0,-60,(float)((span->startSeconds-.6)/.6),false));
            matched=near(gain(span->startSeconds+(span->endSeconds-span->startSeconds)*(.25+.5*.2375)),base*(.8-.6*.6875));
        };
        const auto rendered=backend::UtauRenderer::render(request);
        check("segment_bezier_actual_mixer_matches_cubic",called&&matched&&rendered.buffer.getNumSamples()>0);
    }

    return ok;
}
}
