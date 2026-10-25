#pragma once
#include "../ProjectModel.h"
#include "../PianoRollComponent.h"
#include "../AudioEngine.h"
#include "../backend/DiffSingerRenderer.h"
#include <iostream>

namespace hachi
{
inline bool runDiffSingerExpressionsSmoke()
{
    const auto folder = juce::File::getSpecialLocation(juce::File::tempDirectory)
        .getNonexistentChildFile("hachi-ds-expressions", {}, false);
    folder.createDirectory();
    struct Cleanup { juce::File file; ~Cleanup() { file.deleteRecursively(); } } cleanup { folder };
    folder.getChildFile("dsconfig.yaml").replaceWithText("acoustic: test.onnx\n");
    bool ok = true;
    auto check = [&](const char* name, bool value)
    { ok = ok && value; std::cout << name << '=' << value << std::endl; };
    ProjectModel model;
    ProjectData data;
    TrackData track; track.id="ds"; track.pitchAlgorithm=PitchAlgorithm::utau;
    track.voicebankDirectory=folder;
    ClipData clip; clip.id="clip"; clip.durationSeconds=2;
    NoteData note; note.id="note"; note.durationSeconds=1; note.label="a";
    clip.notes.push_back(note); track.clips.push_back(clip); data.tracks.push_back(track);
    auto plain=track; plain.id="plain"; plain.voicebankDirectory=juce::File{};
    plain.clips.front().id="plain-clip"; plain.clips.front().notes.front().id="plain-note";
    data.tracks.push_back(plain); model.replace(data);
    model.setNotesUtauFlagCurveEnabled({"note","plain-note"},true);
    check("ordinary_U_DF_eligible",model.snapshot().tracks[0].clips[0].notes[0].utauFlagCurveEnabled);
    check("ordinary_UTAU_WCSNDM_eligible",model.snapshot().tracks[1].clips[0].notes[0].utauFlagCurveEnabled);
    check("DF_namespace_rejected_in_WCSNDM",!model.setNoteUtauFlagCurve("plain-note","DS:DYN",{{0,-120}}));
    model.setNoteUtauFlagCurve("note","DS:DYN",{{0,-120},{1,0}});
    FlagCurvePoint end{1,100}; end.shape=PitchCurveShape::smooth;
    model.setNoteUtauFlagCurve("note","DS:VELC",{{0,200},end});
    const auto before=model.revisionNumber();
    model.setNoteUtauFlagCurve("note","Mb",{{0,100}});
    check("resampler_flag_rejected_for_DF",model.revisionNumber()==before);
    auto current=[&] { return model.snapshot().tracks[0].clips[0].notes[0]; };
    check("multiple_curves",current().utauFlagCurves.size()==2);
    auto samples=sampleDiffSingerFlagCurves(current());
    check("bezier_sampled",samples.size()==2 && samples[1].second.size()>100);
    backend::UtauRenderRequest request; request.voicebankDirectory=folder;
    request.notes=AudioEngine::diagnosticUtauRequestNotes(model.snapshot(),"clip");
    const auto json=backend::DiffSingerRenderer::requestJson(request,"render");
    check("render_request_multiple_curves",json["notes"][0]["expressions"]["DS:DYN"].size()==2
        && json["notes"][0]["expressions"]["DS:VELC"].size()>100);
    model.resetNotesUtauFlagCurve({"note"},"DS:DYN");
    check("selective_reset",current().utauFlagCurves.size()==1 && current().utauFlagCurves[0].flag=="DS:VELC");
    model.undo(); check("undo",current().utauFlagCurves.size()==2);
    model.redo(); check("redo",current().utauFlagCurves.size()==1); model.undo();
    juce::String error;
    check("save",model.save(folder.getChildFile("curves.hjpx"),error));
    ProjectModel loaded; check("load",loaded.load(folder.getChildFile("curves.hjpx"),error));
    const auto restored=loaded.snapshot().tracks[0].clips[0].notes[0];
    check("persist_curves_and_shape",restored.utauFlagCurves.size()==2
        && flagCurvePointsFor(restored,"DS:VELC").back().shape==PitchCurveShape::smooth);
    I18n strings;
    PianoRollComponent roll(model,strings); roll.diagnosticRefresh();
    auto caps=juce::JSON::parse(R"([{"key":"DS:DYN","supported":true},{"key":"DS:VELC","supported":true},{"key":"DS:PEXP","supported":true},{"key":"DS:BREC","supported":false}])");
    roll.setDiffSingerFlagContext(true,caps);
    check("DF_lane_default",roll.flagLaneFlag()=="DS:DYN" && roll.flagCurveActiveFor("note"));
    roll.setFlagLaneFlag("DS:PEXP");
    check("neutral_100",roll.flagLaneCurveFor(restored).front().value==100);
    roll.setFlagLaneFlag("DS:BREC");
    check("unsupported_disabled",roll.flagLaneFlag()=="DS:PEXP");
    roll.setDiffSingerFlagContext(false,{});
    check("backend_lane_isolation",roll.flagLaneFlag()=="g" && !roll.flagCurveActiveFor("note"));
    model.setNotesUtauFlagCurveEnabled({"note"},false);
    check("disable_preserves_edits",current().utauFlagCurves.size()==2 && sampleDiffSingerFlagCurves(current()).empty());
    return ok;
}
}
