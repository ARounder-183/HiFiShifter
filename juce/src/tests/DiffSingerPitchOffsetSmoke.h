#pragma once
#include "../MainComponent.h"
#include "../backend/DiffSingerRenderer.h"
#include <iostream>
namespace hachi
{
inline bool MainComponent::diagnosticPitchOffset(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;juce::Array<juce::var> checks;
    const auto check=[&](const char* name,bool passed){auto* c=new juce::DynamicObject();c->setProperty("name",name);c->setProperty("passed",passed);checks.add(c);ok=ok&&passed;std::cout<<name<<'='<<passed<<std::endl;};
    const auto same=[](const auto& a,const auto& b){if(a.size()!=b.size())return false;for(size_t i=0;i<a.size();++i)if(a[i].timeSeconds!=b[i].timeSeconds||a[i].targetMidi!=b[i].targetMidi)return false;return true;};
    ProjectData data;TrackData track;track.id="ds";track.pitchAlgorithm=PitchAlgorithm::utau;track.compose=true;track.utauMode=UtauMode::mou;
    track.voicebankDirectory=folder.getChildFile("fixture-bank");track.voicebankDirectory.createDirectory();
    track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");
    ClipData clip;clip.id="clip";clip.durationSeconds=3.5;
    const std::vector<PitchCurveEditPoint> base{{-.15,59,PitchCurveShape::linear},{0,60,PitchCurveShape::linear},{.25,61,PitchCurveShape::linear},{.5,59.5f,PitchCurveShape::linear},{.75,60.5f,PitchCurveShape::linear},{1,60,PitchCurveShape::linear}};
    NoteData n;n.id="n0";n.label="ni";n.startSeconds=.5;n.durationSeconds=1;n.midiNote=60;n.utauAutoPitchTransition=false;n.pitchControlPoints=base;n.diffSingerPitchReference=base;clip.notes.push_back(n);
    n.id="n1";n.label="hao";n.startSeconds=1.5;clip.notes.push_back(n);track.clips.push_back(clip);data.tracks.push_back(track);
    ProjectModel model;model.replace(data);
    const auto find=[](const ProjectData& d,const juce::String& id){for(const auto& t:d.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==id)return n;return NoteData{};};
    const auto note=[&](const juce::String& id="n0"){return find(model.snapshot(),id);};
    I18n lang;PianoRollComponent roll(model,lang);roll.setSize(1600,1200);roll.setPixelsPerSecond(420);roll.setRowHeight(30);roll.setFocusedTrack("ds");roll.setFocusedClip("clip");roll.setShowPitchLine(true);roll.setTool(PianoRollComponent::Tool::points);roll.setSelectedNoteIds({"n0"});
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();
    const auto reference=roll.diagnosticDiffSingerPitchReference("n0").toString();
    const auto initialRequest=AudioEngine::diagnosticUtauRequestNotes(model.snapshot(),"clip");
    const auto initialKey=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip");
    const auto actual=[&](const ProjectData& d){return AudioEngine::diagnosticUtauRequestNotes(d,"clip");};
    const auto event=[&](double time,float midi,int clicks=1,bool ctrl=false){const auto pos=juce::Point<float>(roll.diagnosticEdgeX(time),roll.diagnosticYForMidi(midi));return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),pos,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier|(ctrl?juce::ModifierKeys::ctrlModifier:0)),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),pos,juce::Time::getCurrentTime(),clicks,false);};
    const auto lineAt=[&](double time){return roll.diagnosticPitchLineAt("n0",.5+time).value_or(-999);};
    const auto lineBefore=lineAt(.3);
    check("DS_offset_mode_available",roll.diffSingerPitchOffsetAvailable());roll.setDiffSingerPitchOffsetMode(true);
    check("DS_offset_mode_enabled",roll.diffSingerPitchOffsetMode()&&!roll.diffSingerPitchRestoreMode());
    const auto rev=model.revisionNumber();roll.mouseDown(event(.9,60));roll.mouseUp(event(.9,60));refresh();
    check("default_line_click_does_not_edit",model.revisionNumber()==rev&&note().diffSingerPitchOffset.empty());
    roll.mouseDown(event(.9,60));roll.mouseDrag(event(.9,58));
    check("live_preview_moves_actual_curve",std::abs(lineAt(.3)-lineBefore+2)<1e-4);
    check("preview_does_not_commit",note().diffSingerPitchOffset.empty());roll.mouseUp(event(.9,58));refresh();
    check("whole_line_drag_minus_two",std::abs(diffSingerPitchOffsetAt(note(),.3)+2)<1e-5);
    check("base_and_reference_untouched",same(note().pitchControlPoints,base)&&same(note().diffSingerPitchReference,base));
    check("dashed_reference_path_unchanged",roll.diagnosticDiffSingerPitchReference("n0").toString()==reference);
    auto shifted=actual(model.snapshot());float error=0;
    for(double t=-.15;t<1;t+=.01)error=std::max(error,std::abs(shifted[0].timelinePitchCents(t)-initialRequest[0].timelinePitchCents(t)+200));
    check("render_pitch_adds_minus_200_cents_without_flattening",error<.001f);
    backend::UtauRenderRequest renderedRequest;renderedRequest.notes=shifted;renderedRequest.voicebankDirectory=track.voicebankDirectory;
    const auto encoded=backend::DiffSingerRenderer::requestJson(renderedRequest,"render");
    float jsonError=0;for(const auto& p:*encoded["notes"][0]["pitch"].getArray())
        jsonError=std::max(jsonError,std::abs(static_cast<float>(p[1])-(60+initialRequest[0].timelinePitchCents(static_cast<double>(p[0]))/100-2)));
    check("DS_runner_receives_shifted_pitch",jsonError<.0001f);
    check("other_note_audio_unchanged",std::abs(shifted[1].timelinePitchCents(.3)-initialRequest[1].timelinePitchCents(.3))<.001f);
    check("offset_invalidates_audio_cache",AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip")!=initialKey);
    model.undo();refresh();check("undo_removes_offset",note().diffSingerPitchOffset.empty()&&std::abs(lineAt(.3)-lineBefore)<1e-5);
    model.redo();refresh();check("redo_restores_offset",std::abs(diffSingerPitchOffsetAt(note(),.3)+2)<1e-5);
    model.setDiffSingerPitchOffset("n0",{});refresh();
    roll.mouseDown(event(1.0,60));roll.mouseUp(event(1.0,60));
    roll.mouseDown(event(1.0,60,2));roll.mouseUp(event(1.0,60,2));roll.mouseDoubleClick(event(1.0,60,2));refresh();
    check("double_click_adds_zero_offset_point",note().diffSingerPitchOffset.size()==3);
    check("adding_zero_point_keeps_base_and_reference",same(note().pitchControlPoints,base)&&same(note().diffSingerPitchReference,base));
    const auto addRevision=model.revisionNumber();
    roll.mouseDown(event(1.0,60,2));roll.mouseUp(event(1.0,60,2));roll.mouseDoubleClick(event(1.0,60,2));refresh();
    check("double_click_existing_point_does_not_duplicate",note().diffSingerPitchOffset.size()==3&&model.revisionNumber()==addRevision);
    model.undo();refresh();check("undo_added_zero_point",note().diffSingerPitchOffset.empty());
    model.redo();refresh();check("redo_added_zero_point",note().diffSingerPitchOffset.size()==3);
    juce::String addedError;ProjectModel addedReload;
    check("zero_point_survives_save_reopen",model.save(folder.getChildFile("added-zero-point.hjpx"),addedError)
        && addedReload.load(folder.getChildFile("added-zero-point.hjpx"),addedError)
        && find(addedReload.snapshot(),"n0").diffSingerPitchOffset.size()==3);
    roll.mouseDown(event(1.0,60));roll.mouseDrag(event(1.0,61));roll.mouseUp(event(1.0,61));refresh();
    check("offset_point_drag",std::abs(diffSingerPitchOffsetAt(note(),.5)-1)<1e-5);
    model.setDiffSingerPitchOffset("n0",{{0,-2,PitchCurveShape::linear},{1,2,PitchCurveShape::linear}});refresh();
    shifted=actual(model.snapshot());error=0;
    for(double t=0;t<=1;t+=.01)error=std::max(error,std::abs(shifted[0].timelinePitchCents(t)-initialRequest[0].timelinePitchCents(t)-static_cast<float>((-2+4*t)*100)));
    check("sloped_offset_preserves_prediction_detail",error<.002f);
    const auto slopedBefore=note().diffSingerPitchOffset;
    roll.mouseDown(event(.75,59,2));roll.mouseUp(event(.75,59,2));roll.mouseDoubleClick(event(.75,59,2));refresh();
    check("add_point_on_sloped_line",note().diffSingerPitchOffset.size()==3&&std::abs(diffSingerPitchOffsetAt(note(),.25)+1)<1e-5);
    float addedCurveError=0;for(double t=0;t<=1;t+=.01)addedCurveError=std::max(addedCurveError,
        std::abs(diffSingerPitchOffsetAt(note(),t)-evaluatePitchCurve(slopedBefore,t)));
    check("add_point_does_not_reshape_slope",addedCurveError<1e-5);
    model.undo();refresh();
    const auto slope=note().diffSingerPitchOffset;
    roll.mouseDown(event(1.0,60));roll.mouseDrag(event(1.0,63));roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));roll.mouseUp(event(1.0,63));refresh();
    check("escape_cancels_gesture",same(note().diffSingerPitchOffset,slope)&&roll.diffSingerPitchOffsetMode());
    roll.mouseDown(event(.65,59,1,true));roll.mouseDrag(event(.8,61,1,true));roll.mouseDrag(event(1.1,62,1,true));roll.mouseUp(event(1.1,62,1,true));refresh();
    check("ctrl_freehand_sets_offset",std::abs(diffSingerPitchOffsetAt(note(),.3)-1)<.001&&std::abs(diffSingerPitchOffsetAt(note(),.6)-2)<.001);
    check("freehand_preserves_outside",std::abs(diffSingerPitchOffsetAt(note(),.05)-(-1.8f))<.001&&std::abs(diffSingerPitchOffsetAt(note(),.9)-1.6f)<.001);
    model.undo();refresh();check("freehand_one_undo",same(note().diffSingerPitchOffset,slope));
    juce::String errorText;const auto saved=folder.getChildFile("offset.hjpx");check("save_offset_project",model.save(saved,errorText));ProjectModel loaded;
    check("reopen_offset_project",loaded.load(saved,errorText));check("negative_and_sloped_offset_persist",same(find(loaded.snapshot(),"n0").diffSingerPitchOffset,slope));
    check("saved_reference_unchanged",same(find(loaded.snapshot(),"n0").diffSingerPitchReference,base));
    auto prediction=base;for(auto& p:prediction)p.targetMidi+=.4f;
    check("regenerate_prediction",model.applyDiffSingerPitch(model.revisionNumber(),{{"n0",prediction}}));refresh();
    check("regeneration_keeps_offset_layer",same(note().diffSingerPitchOffset,slope)&&same(note().diffSingerPitchReference,prediction));
    const auto regen=actual(model.snapshot());model.applyDiffSingerPitch(model.revisionNumber(),{{"n0",prediction}});
    check("repeat_prediction_does_not_accumulate_offset",std::abs(actual(model.snapshot())[0].timelinePitchCents(.3)-regen[0].timelinePitchCents(.3))<1e-5);
    model.transposeNotes({"n0"},2);check("transpose_keeps_relative_offset",same(note().diffSingerPitchOffset,slope));model.undo();
    model.resizeNote("n0",.5,2);check("resize_scales_offset_time",std::abs(note().diffSingerPitchOffset.back().timeSeconds-2)<1e-6);model.undo();
    const auto split=model.splitNote("n0",.5);check("split_preserves_offset_value",split.isNotEmpty()&&std::abs(diffSingerPitchOffsetAt(note(),.5))<1e-5&&std::abs(diffSingerPitchOffsetAt(note(split),0))<1e-5);model.undo();
    const auto duplicate=model.duplicateNotes({"n0"},"clip",3);check("duplicate_preserves_offset",duplicate.size()==1&&same(note(duplicate.front()).diffSingerPitchOffset,slope));model.undo();
    refresh();roll.setDiffSingerPitchOffsetMode(false);
    const auto anchors=roll.diagnosticPitchAnchors("n0");const auto p=anchors[2];
    const auto audible=p.targetMidi+diffSingerPitchOffsetAt(note(),p.timeSeconds);
    roll.mouseDown(event(.5+p.timeSeconds,audible));roll.mouseDrag(event(.5+p.timeSeconds,audible+1));roll.mouseUp(event(.5+p.timeSeconds,audible+1));refresh();
    check("ordinary_point_edit_does_not_bake_offset",std::abs(evaluatePitchCurve(note().pitchControlPoints,p.timeSeconds)-p.targetMidi-1)<.01&&same(note().diffSingerPitchOffset,slope));model.undo();refresh();
    roll.setDiffSingerPitchOffsetMode(true);roll.setDiffSingerPitchRestoreMode(true);check("offset_and_restore_mutually_exclusive",roll.diffSingerPitchRestoreMode()&&!roll.diffSingerPitchOffsetMode());roll.setDiffSingerPitchOffsetMode(true);
    const auto area=juce::Rectangle<int>((int)roll.diagnosticEdgeX(.25),(int)roll.diagnosticYForMidi(64),1120,310);
    if(auto stream=folder.getChildFile("pitch-offset.png").createOutputStream()){stream->setPosition(0);stream->truncate();juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(area),*stream);}
    roll.setTool(PianoRollComponent::Tool::note);check("changing_tool_exits_offset_mode",!roll.diffSingerPitchOffsetMode());
    auto ordinary=model.snapshot();ordinary.tracks[0].voicebankDirectory=juce::File{};ProjectModel ordinaryModel;ordinaryModel.replace(ordinary);
    check("ordinary_UTAU_rejects_offset_edit",!ordinaryModel.setDiffSingerPitchOffset("n0",{{0,2}}));
    auto ordinaryWithout=ordinary;for(auto& c:ordinaryWithout.tracks[0].clips)for(auto& n:c.notes)n.diffSingerPitchOffset.clear();
    check("ordinary_UTAU_ignores_stored_offset",std::abs(actual(ordinary)[0].timelinePitchCents(.3)-actual(ordinaryWithout)[0].timelinePitchCents(.3))<1e-5);
    project.replace(data);project.dispatchPendingMessages();diagnosticSelectTrack("ds");pianoRoll.setFocusedClip("clip");pianoRoll.diagnosticRefresh();
    const auto menu=pitchPointModeMenu();bool offered=false;for(juce::PopupMenu::MenuItemIterator it(menu,true);it.next();)if(it.getItem().itemID==4)offered=it.getItem().isEnabled;
    check("toolbar_context_menu_offers_offset",offered);selectPitchPointMode(4);check("toolbar_menu_enters_offset_mode",pianoRoll.diffSingerPitchOffsetMode());selectPitchPointMode(1);check("normal_menu_exits_offset_mode",!pianoRoll.diffSingerPitchOffsetMode());
    const auto realBankPath=juce::SystemStats::getEnvironmentVariable("HACHI_TEST_DS_BANK",{});
    if (realBankPath.isNotEmpty())
    {
        auto realData=data;realData.tracks[0].voicebankDirectory=juce::File(realBankPath);
        ProjectModel realModel;realModel.replace(realData);
        check("real_bank_fixture",trackIsDiffSinger(realData.tracks[0]));
        check("real_baseline_project",realModel.save(folder.getChildFile("real-baseline.hjpx"),errorText));
        realModel.setDiffSingerPitchOffset("n0",{{0,-2,PitchCurveShape::linear},{1,-2,PitchCurveShape::linear}});
        realModel.setDiffSingerPitchOffset("n1",{{0,-2,PitchCurveShape::linear},{1,-2,PitchCurveShape::linear}});
        check("real_offset_project",realModel.save(folder.getChildFile("real-offset.hjpx"),errorText));
    }
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));return ok;
}
}
