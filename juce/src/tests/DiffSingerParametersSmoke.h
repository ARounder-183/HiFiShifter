#pragma once
#include "../DiffSingerParameterCurves.h"
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include "../backend/DiffSingerRenderer.h"
#include <iostream>
namespace hachi {
inline bool runDiffSingerParametersSmoke(const juce::File& output, const juce::File& realBank = {})
{
    output.createDirectory();
    const auto bank=output.getChildFile("fixture-bank");bank.createDirectory();
    bank.getChildFile("dsconfig.yaml").replaceWithText("acoustic: test.onnx\n");
    bool ok=true;juce::Array<juce::var> checks;
    auto check=[&](const char* name,bool passed){ok=ok&&passed;std::cout<<name<<'='<<passed<<std::endl;
        auto* v=new juce::DynamicObject();v->setProperty("name",name);v->setProperty("passed",passed);checks.add(juce::var(v));};
    ProjectData data;TrackData t;t.id="ds";t.compose=true;t.pitchAlgorithm=PitchAlgorithm::utau;t.voicebankDirectory=bank;
    ClipData c;c.id="clip";c.durationSeconds=3;
    NoteData n;n.id="n";n.label="ni";n.startSeconds=.5;n.durationSeconds=1;
    c.notes.push_back(n);n.id="other";n.startSeconds=1.5;c.notes.push_back(n);t.clips.push_back(c);data.tracks.push_back(t);
    ProjectModel model;model.replace(data);
    auto current=[&]{return model.snapshot().tracks[0].clips[0].notes[0];};
    FlagCurve ref{"DS:REF:BREC",{}};
    for(int i=0;i<=120;++i)ref.points.push_back({-.2+i*.01,(float)(-50+5*std::sin(i*.1))});
    const auto before=model.revisionNumber();
    check("accept_prediction",model.applyDiffSingerParameters(before,{{"n",{ref}}}));
    check("stale_prediction_rejected",!model.applyDiffSingerParameters(before,{{"other",{ref}}}));
    check("only_selected_note",model.snapshot().tracks[0].clips[0].notes[1].utauFlagCurves.empty());
    check("exact_reference_as_base",flagCurvePointsFor(current(),"DS:ABS:BREC").size()==121);
    check("flag_off_still_synthesizes_base",sampleDiffSingerFlagCurves(current()).size()==1);
    check("sparse_handles_dense_synthesis",diffSingerParameterHandles(current(),"DS:ABS:BREC").size()<=24);
    I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1100,700);roll.diagnosticRefresh();
    roll.setDiffSingerFlagContext(true,juce::JSON::parse(R"([{"key":"DS:DYN","supported":true},{"key":"DS:BREC","supported":true},{"key":"DS:ENE","supported":false}])"));
    roll.setFlagLaneFlag("DS:ABS:BREC");
    check("actual_lane_without_flag_enable",roll.flagCurveActiveFor("n")&&roll.flagLaneFlag()=="DS:ABS:BREC");
    roll.setFlagLaneFlag("DS:ABS:ENE");check("unsupported_actual_blocked",roll.flagLaneFlag()=="DS:ABS:BREC");
    const auto keyBefore=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip");
    auto handles=diffSingerParameterHandles(current(),"DS:ABS:BREC");
    const auto untouched=handles[1].timeSeconds;
    handles[0].value+=4;
    check("edit_actual",model.setNoteUtauFlagCurve("n","DS:ABS:BREC",handles));
    check("reference_unchanged",flagCurveValueAt(flagCurvePointsFor(current(),"DS:REF:BREC"),-.2)==ref.points[0].value);
    bool outside=true;
    for(double x=untouched+.001;x<1;x+=.001)outside=outside&&std::abs(flagCurveValueAt(flagCurvePointsFor(current(),"DS:ABS:BREC"),x)-flagCurveValueAt(ref.points,x))<.001f;
    check("outside_edited_segment_exact",outside);
    check("cache_invalidated",keyBefore!=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),"clip"));
    model.undo();check("undo_actual",flagCurveValueAt(flagCurvePointsFor(current(),"DS:ABS:BREC"),-.2)==ref.points[0].value);
    model.redo();check("redo_actual",flagCurveValueAt(flagCurvePointsFor(current(),"DS:ABS:BREC"),-.2)==ref.points[0].value+4);
    model.setNotesUtauFlagCurveEnabled({"n"},true);model.setNoteUtauFlagCurve("n","DS:BREC",{{0,50}});
    backend::UtauRenderRequest request;request.voicebankDirectory=bank;
    request.notes=AudioEngine::diagnosticUtauRequestNotes(model.snapshot(),"clip");
    auto json=backend::DiffSingerRenderer::requestJson(request,"render");
    check("separate_actual_and_offset_request",json["notes"][0]["parameters"]["BREC"].size()>2&&json["notes"][0]["expressions"]["DS:BREC"].size()==1);
    juce::String error;check("save",model.save(output.getChildFile("parameters.hjpx"),error));
    ProjectModel loaded;check("reload",loaded.load(output.getChildFile("parameters.hjpx"),error));
    const auto saved=loaded.snapshot().tracks[0].clips[0].notes[0];
    check("persist_exact_reference_and_base",flagCurvePointsFor(saved,"DS:REF:BREC").size()==ref.points.size()
        &&flagCurveValueAt(flagCurvePointsFor(saved,"DS:ABS:BREC"),-.2)==ref.points[0].value+4
        &&diffSingerParameterHandles(saved,"DS:ABS:BREC").size()==handles.size());
    roll.diagnosticRefresh();roll.setTool(PianoRollComponent::Tool::flagCurve);
    roll.setFlagLaneFlag("DS:ABS:BREC");
    {juce::FileOutputStream stream(output.getChildFile("actual-lane.png"));juce::PNGImageFormat png;
        check("render_actual_lane",png.writeImageToStream(roll.createComponentSnapshot(roll.getLocalBounds()),stream));}
    // Render both layers and an in-progress offset gesture for visual verification.
    const auto screenshot=[&](const char* name) {
        auto stream=output.getChildFile(name).createOutputStream();stream->setPosition(0);stream->truncate();
        return juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(roll.flagLaneBounds().toNearestInt()),*stream);
    };
    model.setNoteUtauFlagCurve("n","DS:BREC",{{0,-60},{.5,60},{1,-30}});roll.diagnosticRefresh();
    check("actual_with_offset_reference",screenshot("actual-with-offset.png"));
    roll.setFlagLaneFlag("DS:BREC");
    check("offset_with_actual_reference",screenshot("offset-with-actual.png"));
    const auto dragRevision=model.revisionNumber();
    const auto down=roll.diagnosticFlagHandleCentre("n",1), moved=down.withY(roll.flagLaneY(90));
    const auto gesture=[&](juce::Point<float> at){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
        juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,false);};
    roll.mouseDown(gesture(down));roll.mouseDrag(gesture(moved));
    check("live_offset_preview_before_commit",model.revisionNumber()==dragRevision&&screenshot("offset-live-preview.png"));
    roll.mouseUp(gesture(moved));model.undo();model.undo();roll.diagnosticRefresh();roll.setFlagLaneFlag("DS:ABS:BREC");
    model.setNotesUtauFlagCurveEnabled({"n"},false);roll.diagnosticRefresh();
    const auto strokeBefore=model.revisionNumber();
    roll.setFlagEditMode(PianoRollComponent::FlagEditMode::continuous);
    auto mouse=[&](int action,double time,float value) {
        const auto at=juce::Point<float>(roll.diagnosticEdgeX(time),roll.flagLaneY(value));
        const auto event=juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,
            juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,false);
        if(action==0)roll.mouseDown(event);else if(action==1)roll.mouseDrag(event);else roll.mouseUp(event);
    };
    mouse(0,.8,-35);mouse(1,1.1,-25);
    check("stroke_preview_no_write",model.revisionNumber()==strokeBefore);
    mouse(2,1.1,-25);
    check("continuous_actual_without_flag_enable",model.revisionNumber()==strokeBefore+1
        &&std::abs(flagCurveValueAt(flagCurvePointsFor(current(),"DS:ABS:BREC"),.45)+30)<.05f);
    model.undo();model.setNotesUtauFlagCurveEnabled({"n"},true);
    model.resetNotesUtauFlagCurves({"n"});check("reset_flags_preserves_actual",!flagCurvePointsFor(current(),"DS:ABS:BREC").empty()&&flagCurvePointsFor(current(),"DS:BREC").empty());
    model.setNotesUtauFlagCurveEnabled({"n"},false);model.resetNotesUtauFlagCurve({"n"},"DS:ABS:BREC");
    check("restore_original_while_flag_off",flagCurveValueAt(flagCurvePointsFor(current(),"DS:ABS:BREC"),-.2)==ref.points[0].value);
    check("reference_readonly",!model.setNoteUtauFlagCurve("n","DS:REF:BREC",{{0,-10}}));
    const auto right=model.splitNote("n",.5);
    check("split_created",right.isNotEmpty());
    const auto splitSnapshot=model.snapshot();
    for(const auto& nn:splitSnapshot.tracks[0].clips[0].notes) if(nn.id==right)
        check("split_preserves_actual",std::abs(flagCurveValueAt(flagCurvePointsFor(nn,"DS:ABS:BREC"),0)-flagCurveValueAt(ref.points,.5))<.001);
    model.undo();model.resizeNote("n",.5,2);
    check("resize_keeps_parameter_time",std::abs(flagCurvePointsFor(current(),"DS:REF:BREC").back().timeSeconds-2)<.001);
    model.applyDiffSingerParameters(model.revisionNumber(),{{"n",{}}});
    check("release_frozen_prediction",sampleDiffSingerFlagCurves(current()).empty());
    ProjectModel defaults;auto plain=data;plain.tracks[0].voicebankDirectory=juce::File{};defaults.replace(plain);
    defaults.setTrackVoicebankDirectory("ds",bank);
    check("enter_DS_enables_existing_notes",defaults.snapshot().tracks[0].clips[0].notes[0].utauFlagCurveEnabled);
    const auto added=defaults.addNote("clip",2.6,.3,60);
    const auto findAdded=[&](const ProjectData& d){for(const auto& note:d.tracks[0].clips[0].notes)if(note.id==added)return note;return NoteData{};};
    check("new_DS_note_enabled",added.isNotEmpty()&&findAdded(defaults.snapshot()).utauFlagCurveEnabled);
    defaults.setNotesUtauFlagCurveEnabled({added},false);
    check("manual_disable",!findAdded(defaults.snapshot()).utauFlagCurveEnabled);
    const auto disabledFile=output.getChildFile("offset-disabled.hjpx");
    check("save_disabled_offset",defaults.save(disabledFile,error));ProjectModel disabled;
    check("explicit_off_survives_reopen",disabled.load(disabledFile,error)&&!findAdded(disabled.snapshot()).utauFlagCurveEnabled);
    auto stream=disabledFile.createInputStream();auto legacy=juce::ValueTree::readFromStream(*stream);stream.reset();
    const auto removeMarker=[](auto&& self,juce::ValueTree tree)->void {
        tree.removeProperty("diffSingerOffsetEnabled",nullptr);for(auto child:tree)self(self,child);
    };
    removeMarker(removeMarker,legacy);const auto legacyFile=output.getChildFile("legacy-offset.hjpx");
    {auto dest=legacyFile.createOutputStream();dest->setPosition(0);dest->truncate();legacy.writeToStream(*dest);}
    ProjectModel migrated;check("legacy_DS_default_enabled",migrated.load(legacyFile,error)&&findAdded(migrated.snapshot()).utauFlagCurveEnabled);
    defaults.undo();check("undo_disable_restores_enabled",findAdded(defaults.snapshot()).utauFlagCurveEnabled);
    defaults.redo();check("redo_disable_restores_off",!findAdded(defaults.snapshot()).utauFlagCurveEnabled);
    ProjectModel classic;classic.replace(plain);const auto classicId=classic.addNote("clip",2.6,.3,60);
    bool classicOff=false;const auto classicData=classic.snapshot();for(const auto& note:classicData.tracks[0].clips[0].notes)if(note.id==classicId)classicOff=!note.utauFlagCurveEnabled;
    check("ordinary_UTAU_default_unchanged",classicOff);
    // Automatic predictions persist independently of frozen/manual actual values.
    ProjectModel automatic; automatic.replace(data);
    std::vector<FlagCurve> remembered;
    for (const auto& kind:diffSingerParameterKinds()) {
        FlagCurve curve{"DS:AUTO:"+juce::String(kind.flag).substring(7),{}};
        for (int i=0;i<80;++i) curve.points.push_back({-.15+i*.014,
            juce::String(kind.flag).endsWith("TENC") ? static_cast<float>(.3+std::sin(i*.1)) : static_cast<float>(-50+3*std::sin(i*.1))});
        remembered.push_back(std::move(curve));
    }
    const auto renderBefore=AudioEngine::diagnosticUtauRenderKey(automatic.snapshot(),"clip");
    const auto revisionBefore=automatic.revisionNumber();
    check("remember_all_automatic_channels",automatic.rememberDiffSingerParameters({{"n",remembered}}));
    check("automatic_prediction_marks_unsaved",automatic.revisionNumber()>revisionBefore);
    check("automatic_snapshot_does_not_rerender",renderBefore==AudioEngine::diagnosticUtauRenderKey(automatic.snapshot(),"clip"));
    check("automatic_snapshot_does_not_freeze",sampleDiffSingerFlagCurves(automatic.snapshot().tracks[0].clips[0].notes[0]).empty());
    check("duplicate_automatic_result_no_change",!automatic.rememberDiffSingerParameters({{"n",remembered}}));
    const auto cacheFile=output.getChildFile("automatic-parameters.hjpx");
    check("save_automatic_parameters",automatic.save(cacheFile,error));ProjectModel reopened;
    check("reopen_automatic_parameters",reopened.load(cacheFile,error));
    const auto reopenedNote=reopened.snapshot().tracks[0].clips[0].notes[0];
    for (const auto& curve:remembered) {
        const auto code=curve.flag.substring(8);const auto stored=flagCurvePointsFor(reopenedNote,"DS:ABS:"+code);
        check(("exact_automatic_roundtrip_"+code).toRawUTF8(),stored.size()==curve.points.size()
            && std::equal(stored.begin(),stored.end(),curve.points.begin(),[](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds && a.value==b.value;}));
    }
    auto editAuto=diffSingerParameterHandles(reopenedNote,"DS:ABS:BREC"); editAuto[0].value+=4;
    check("edit_reopened_automatic_base",reopened.setDiffSingerParameterCurve("n","DS:ABS:BREC",editAuto));
    const auto editedBase=flagCurvePointsFor(reopened.snapshot().tracks[0].clips[0].notes[0],"DS:ABS:BREC");
    auto newer=remembered; for(auto& p:newer[1].points)p.value+=2;
    reopened.rememberDiffSingerParameters({{"n",newer}});
    check("automatic_result_preserves_manual_base",flagCurvePointsFor(reopened.snapshot().tracks[0].clips[0].notes[0],"DS:ABS:BREC")[0].value==editedBase[0].value);
    if (realBank.isDirectory()) {
        auto real=data;real.tracks[0].voicebankDirectory=realBank;
        backend::DiffSingerOptions options;options.backend=2;options.preview=1;backend::DiffSingerRenderer::configure(options);
        AudioEngine engine;engine.selectEveryUtauNote(real);engine.syncProject(real);
        for (int i=0;i<4800 && !engine.hasCurrentRenderedAudio();++i) {
            if(i>40 && !engine.renderProgress())break;juce::Thread::sleep(25);
        }
        check("real_automatic_render",engine.hasCurrentRenderedAudio());engine.refreshUtauWaveforms();
        std::map<juce::String,std::vector<FlagCurve>> collected;
        const auto waves=engine.utauNoteWaveforms();
        if(waves)for(const auto& w:*waves)if(!w.diffSingerParameters.empty())collected[w.noteId]=w.diffSingerParameters;
        check("real_render_returns_all_note_predictions",collected.size()==2 && collected["n"].size()>=3);
        ProjectModel actual;actual.replace(real);check("real_capture_into_project",actual.rememberDiffSingerParameters(collected));
        check("real_save_automatic",actual.save(output.getChildFile("real-automatic.hjpx"),error));
        ProjectModel reload;check("real_reopen_automatic",reload.load(output.getChildFile("real-automatic.hjpx"),error));
        const auto saved=reload.snapshot().tracks[0].clips[0].notes[0];
        check("real_reopened_prediction_visible",flagCurvePointsFor(saved,"DS:ABS:BREC").size()>20 && flagCurvePointsFor(saved,"DS:REF:TENC").size()>20);
        check("real_reopen_retains_auto_inference",sampleDiffSingerFlagCurves(saved).empty());
    }
    auto* report=new juce::DynamicObject();report->setProperty("ok",ok);report->setProperty("checks",checks);
    output.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok;
}
}
