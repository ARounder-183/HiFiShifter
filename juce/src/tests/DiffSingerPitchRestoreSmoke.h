#pragma once
#include "../MainComponent.h"
#include "../DiffSingerPitchRestore.h"
#include <iostream>

namespace hachi
{
inline bool MainComponent::diagnosticPitchRestore(const juce::File& folder)
{
    folder.createDirectory(); bool ok=true; juce::Array<juce::var> checks;
    const auto check=[&](const char* name,bool passed) {
        auto* item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("passed",passed);
        checks.add(item);ok=ok&&passed;std::cout<<name<<'='<<passed<<std::endl;
    };
    const auto equal=[](const auto& a,const auto& b) {
        if(a.size()!=b.size())return false;
        for(size_t i=0;i<a.size();++i)
            if(std::abs(a[i].timeSeconds-b[i].timeSeconds)>1e-8||std::abs(a[i].targetMidi-b[i].targetMidi)>1e-5
                ||a[i].shape!=b[i].shape||a[i].bezierX1!=b[i].bezierX1||a[i].bezierX2!=b[i].bezierX2
                ||a[i].bezierY1!=b[i].bezierY1||a[i].bezierY2!=b[i].bezierY2
                ||a[i].diffSingerRestoreSupport!=b[i].diffSingerRestoreSupport)return false;
        return true;
    };
    const auto maxError=[](const auto& a,const auto& b,double from,double to) {
        float error=0;for(double t=from;t<=to;t+=.0005)
            error=std::max(error,std::abs(evaluatePitchCurve(a,t)-evaluatePitchCurve(b,t)));
        return error;
    };
    const auto find=[](const ProjectData& d,const juce::String& id) {
        for(const auto& t:d.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==id)return n;
        return NoteData{};
    };
    ProjectModel model; ProjectData data; TrackData track; track.id="ds";
    track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
    track.voicebankDirectory=folder.getChildFile("fixture-bank");track.voicebankDirectory.createDirectory();
    track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");
    ClipData clip;clip.id="clip";clip.startSeconds=3;clip.durationSeconds=5;
    const std::vector<PitchCurveEditPoint> reference{{-.12,58},{0,60},{.2,60.5f},{.4,60.8f},{.6,60.1f},{.8,59.5f},{1.2,60}};
    const std::vector<PitchCurveEditPoint> edited{{-.12,58},{0,59},{.2,63},{.4,65},{.6,64},{.8,62},{1,61},{1.2,60}};
    NoteData n;n.id="n0";n.label="ni";n.startSeconds=1;n.durationSeconds=1.2;n.midiNote=60;
    n.diffSingerPitchReference=reference;n.pitchControlPoints=edited;clip.notes.push_back(n);
    n.id="n1";n.label="hao";n.startSeconds=3;clip.notes.push_back(n);
    track.clips.push_back(clip);data.tracks.push_back(track);model.replace(data);
    const auto note=[&](const juce::String& id="n0"){return find(model.snapshot(),id);};
    I18n testStrings;PianoRollComponent roll(model,testStrings);roll.setSize(1800,1100);
    roll.setPixelsPerSecond(320);roll.setRowHeight(25);roll.setFocusedTrack("ds");roll.setFocusedClip("clip");
    roll.setTool(PianoRollComponent::Tool::points);roll.setShowPitchLine(true);roll.setSelectedNoteIds({"n0"});
    const auto refresh=[&] {model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();
    check("DS_reference_enables_mode",roll.diffSingerPitchRestoreAvailable());
    roll.setDiffSingerPitchRestoreMode(true);check("restore_mode_enabled",roll.diffSingerPitchRestoreMode());
    const auto menus=roll.diagnosticEnabledAnchorMenuIds("n0",3);
    check("anchor_context_menu_also_offers_restore",std::find(menus.begin(),menus.end(),10)!=menus.end());
    const auto event=[&](juce::Point<float> pos,int clicks=1) {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),pos,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,
            juce::Time::getCurrentTime(),pos,juce::Time::getCurrentTime(),clicks,false);
    };
    const auto at=juce::Point<float>(roll.diagnosticEdgeX(4.4),roll.diagnosticYForMidi(65));
    const auto shot=[&](const char* file) {
        const auto area=juce::Rectangle<int>((int)roll.diagnosticEdgeX(3.7),(int)roll.diagnosticYForMidi(68),670,310);
        if(auto stream=folder.getChildFile(file).createOutputStream()) {
            stream->setPosition(0);stream->truncate();juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(area),*stream);
        }
    };
    shot("before-restore.png");
    const auto beforePath=roll.diagnosticDiffSingerPitchReference("n0").toString();
    const auto beforeHandles=roll.diagnosticOfferedPitchAnchors("n0").size();
    const auto beforeRevision=model.revisionNumber();const auto beforeKey=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip");
    roll.mouseDown(event(at));roll.mouseUp(event(at));refresh();
    const auto restored=note().pitchControlPoints;
    check("click_restores_pitch_exactly_at_point",std::abs(evaluatePitchCurve(restored,.4)-evaluatePitchCurve(reference,.4))<.0001f);
    check("clicked_handle_disappears",roll.diagnosticOfferedPitchAnchors("n0").size()+1==beforeHandles);
    check("local_interval_matches_prediction",maxError(restored,reference,.211,.589)<.002f);
    check("left_untouched_curve_preserved",maxError(restored,edited,-.12,.2)<.002f);
    check("right_untouched_curve_preserved",maxError(restored,edited,.6,1.2)<.002f);
    check("neighbour_edits_preserved",std::abs(evaluatePitchCurve(restored,.2)-63)<.0001f&&std::abs(evaluatePitchCurve(restored,.6)-64)<.0001f);
    check("reference_data_and_dashed_path_unchanged",equal(note().diffSingerPitchReference,reference)
        &&roll.diagnosticDiffSingerPitchReference("n0").toString()==beforePath);
    check("other_note_unchanged",equal(note("n1").pitchControlPoints,edited));
    check("one_undo_step",model.revisionNumber()==beforeRevision+1);
    check("restored_pitch_changes_audio_key",AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip")!=beforeKey);
    const auto display=roll.diagnosticPitchLineAt("n0",4.4);
    check("display_uses_restored_pitch",display&&std::abs(*display-evaluatePitchCurve(reference,.4))<.002f);
    shot("after-restore.png");
    model.undo();refresh();check("undo_recovers_point_and_curve",equal(note().pitchControlPoints,edited)&&roll.diagnosticOfferedPitchAnchors("n0").size()==beforeHandles);
    model.redo();refresh();check("redo_recovers_restoration",equal(note().pitchControlPoints,restored));
    juce::String error;ProjectModel loaded;
    check("save_load_restored_curve_and_handle_state",model.save(folder.getChildFile("restore.hjpx"),error)
        &&loaded.load(folder.getChildFile("restore.hjpx"),error)&&equal(find(loaded.snapshot(),"n0").pitchControlPoints,restored)
        &&equal(find(loaded.snapshot(),"n0").diffSingerPitchReference,reference));
    auto withoutMetadata=model.snapshot();for(auto& t:withoutMetadata.tracks)for(auto& c:t.clips)for(auto& x:c.notes)
        for(auto& p:x.pitchControlPoints)p.diffSingerRestoreSupport=false;
    check("handle_metadata_not_in_audio_key",AudioEngine::diagnosticUtauRenderKey(withoutMetadata,"clip")==AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip"));
    const auto unchangedRevision=model.revisionNumber();
    roll.mouseDown(event(at));roll.mouseUp(event(at));roll.mouseDoubleClick(event(at,2));refresh();
    check("blank_or_double_click_does_not_add_point",model.revisionNumber()==unchangedRevision);
    check("stale_revision_rejected",!model.restoreDiffSingerPitchPoint(unchangedRevision-1,"n0",restored,0));
    auto nextIndex=-1;for(size_t i=0;i<restored.size();++i)if(std::abs(restored[i].timeSeconds-.6)<1e-8)nextIndex=(int)i;
    check("adjacent_point_can_restore",model.restoreDiffSingerPitchPoint(model.revisionNumber(),"n0",restored,nextIndex));refresh();
    check("adjacent_restore_preserves_previous_deleted_point",roll.diagnosticOfferedPitchAnchors("n0").size()+2==beforeHandles
        &&maxError(note().pitchControlPoints,reference,.211,.789)<.002f);
    for(int iteration=0;iteration<10;++iteration) {
        const auto points=note().pitchControlPoints;
        auto it=std::find_if(points.begin(),points.end(),[](const auto& p){return !p.diffSingerRestoreSupport;});
        if(it==points.end())break;
        model.restoreDiffSingerPitchPoint(model.revisionNumber(),"n0",points,(int)(it-points.begin()));
    }
    refresh();check("restore_all_including_endpoints_and_negative_onset",roll.diagnosticOfferedPitchAnchors("n0").empty()
        &&maxError(note().pitchControlPoints,reference,-.12,1.2)<.002f);
    const auto single=restoredDiffSingerPitchPoint(n,{{.5,66}},0);
    check("single_point_restores_full_reference",!single.empty()&&maxError(single,reference,-.12,1.2)<.002f);
    check("escape_keeps_restored_samples_hidden",roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey))&&!roll.diffSingerPitchRestoreMode()
        &&roll.diagnosticOfferedPitchAnchors("n0").empty());
    model.replace(data);model.setNotePitchCurve("n0",restored,true);refresh();
    const auto restoredAt=juce::Point<float>(roll.diagnosticEdgeX(4.4),roll.diagnosticYForMidi(evaluatePitchCurve(reference,.4)));
    const auto draggedTo=juce::Point<float>(restoredAt.x,restoredAt.y-40);
    const auto handlesBeforeAdd=roll.diagnosticOfferedPitchAnchors("n0").size();
    roll.mouseDoubleClick(event(restoredAt,2));refresh();
    check("normal_double_click_adds_one_handle_in_restored_region",roll.diagnosticOfferedPitchAnchors("n0").size()==handlesBeforeAdd+1
        &&maxError(note().pitchControlPoints,restored,-.12,1.2)<.002f);
    roll.mouseDown(event(restoredAt));roll.mouseDrag(event(draggedTo));roll.mouseUp(event(draggedTo));refresh();
    bool userPoint=false;for(const auto& p:note().pitchControlPoints)
        if(std::abs(p.timeSeconds-.4)<1e-7&&!p.diffSingerRestoreSupport)userPoint=true;
    check("normal_drag_reactivates_restored_sample",userPoint&&!equal(note().pitchControlPoints,restored));
    roll.setDiffSingerPitchRestoreMode(true);roll.setSourceEditMode(true);
    check("source_mode_exits_restore",!roll.diffSingerPitchRestoreMode());roll.setSourceEditMode(false);
    roll.setDiffSingerPitchRestoreMode(true);roll.setTool(PianoRollComponent::Tool::draw);
    check("other_tool_exits_restore",!roll.diffSingerPitchRestoreMode());roll.setTool(PianoRollComponent::Tool::points);
    roll.setDiffSingerPitchRestoreMode(true);roll.setFocusedTrack("another");
    check("track_switch_exits_restore",!roll.diffSingerPitchRestoreMode());roll.setFocusedTrack("ds");
    roll.setDiffSingerPitchRestoreMode(true);roll.setShowPitchLine(false);
    check("hiding_pitch_exits_restore",!roll.diffSingerPitchRestoreMode());roll.setShowPitchLine(true);
    auto noReference=data;noReference.tracks[0].clips[0].notes[0].diffSingerPitchReference.clear();
    noReference.tracks[0].clips[0].notes[1].diffSingerPitchReference.clear();
    roll.setDiffSingerPitchRestoreMode(true);model.replace(noReference);refresh();
    check("missing_reference_disabled_and_rejected",!roll.diffSingerPitchRestoreAvailable()&&!roll.diffSingerPitchRestoreMode()
        &&!model.restoreDiffSingerPitchPoint(model.revisionNumber(),"n0",edited,3));
    auto classic=data;classic.tracks[0].voicebankDirectory=juce::File{};model.replace(classic);refresh();roll.setDiffSingerPitchRestoreMode(true);
    check("ordinary_UTAU_disabled_and_rejected",!roll.diffSingerPitchRestoreAvailable()&&!roll.diffSingerPitchRestoreMode()
        &&!model.restoreDiffSingerPitchPoint(model.revisionNumber(),"n0",edited,3));

    // A typical edit: neighbours already follow DS, only the clicked point moved.
    auto oneEdit=data;oneEdit.tracks[0].clips[0].notes[0].pitchControlPoints=reference;
    oneEdit.tracks[0].clips[0].notes[0].pitchControlPoints[3].targetMidi=65;
    model.replace(oneEdit);refresh();roll.setDiffSingerPitchRestoreMode(true);
    shot("single-edit-before.png");roll.mouseDown(event(at));roll.mouseUp(event(at));refresh();shot("single-edit-after.png");
    check("single_edit_returns_local_curve_to_prediction",maxError(note().pitchControlPoints,reference,.2,.6)<.002f);

    // Exercise the actual toolbar menu and its secondary-click handler.
    const auto previousView=viewOptions;project.replace(data);selectedTrackId="ds";selectedClipId="clip";
    pianoRoll.setFocusedTrack("ds");pianoRoll.setFocusedClip("clip");pianoRoll.diagnosticRefresh();
    const auto menuItem=[&](bool enabled,bool ticked) {
        auto menu=pitchPointModeMenu();for(juce::PopupMenu::MenuItemIterator it(menu);it.next();)
            if(it.getItem().itemID==2)return it.getItem().isEnabled==enabled&&it.getItem().isTicked==ticked;
        return false;
    };
    check("toolbar_menu_DS_enabled",menuItem(true,false));
    const auto toolbarRevision=project.revisionNumber();const auto wasToggled=pointButton.getToggleState();
    const auto right=juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),{5,5},
        juce::ModifierKeys(juce::ModifierKeys::rightButtonModifier),0,0,0,0,0,&pointButton,&pointButton,
        juce::Time::getCurrentTime(),{5,5},juce::Time::getCurrentTime(),1,false);
    pointButton.mouseDown(right);static_cast<juce::Component*>(&pointButton)->mouseUp(right);
    check("toolbar_right_click_does_not_edit_or_toggle",project.revisionNumber()==toolbarRevision&&pointButton.getToggleState()==wasToggled);
    juce::PopupMenu::dismissAllActiveMenus();
    viewOptions.pitchLine=false;applyViewOptions();selectPitchPointMode(2);
    check("menu_choice_enables_points_restore_and_pitch_visibility",menuItem(true,true)&&viewOptions.pitchLine
        &&pianoRoll.currentTool()==PianoRollComponent::Tool::points&&pointButton.getToggleState());
    check("main_escape_exits_restore",keyPressed(juce::KeyPress(juce::KeyPress::escapeKey))&&menuItem(true,false));
    selectPitchPointMode(2);selectPitchPointMode(1);check("normal_menu_returns_to_editing",menuItem(true,false));
    project.replace(classic);pianoRoll.diagnosticRefresh();selectPitchPointMode(2);
    check("toolbar_classic_disabled",menuItem(false,false)&&!pianoRoll.diffSingerPitchRestoreMode());
    project.replace(noReference);pianoRoll.diagnosticRefresh();check("toolbar_no_reference_disabled",menuItem(false,false));
    viewOptions=previousView;applyViewOptions();if(preferences!=nullptr)storeViewOptions(*preferences,viewOptions);
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);
    folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));return ok;
}
}
