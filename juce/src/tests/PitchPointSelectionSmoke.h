#pragma once
#include "../PianoRollComponent.h"
#include <iostream>

namespace hachi
{
inline bool runPitchPointSelectionSmoke(const juce::File& folder)
{
    folder.createDirectory(); bool ok = true; juce::Array<juce::var> checks;
    const auto check = [&](const juce::String& name, bool passed) {
        auto* item = new juce::DynamicObject(); item->setProperty("name", name);
        item->setProperty("passed", passed); checks.add(item); ok &= passed;
        std::cout << name << '=' << passed << std::endl;
    };
    for (const auto mode : { 0, 1, 2 })
    {
        const juce::String prefix = mode == 0 ? "native_" : mode == 1 ? "utau_" : "ds_";
        ProjectData data; TrackData track; track.id = "t"; track.compose = true;
        track.pitchAlgorithm = mode == 0 ? PitchAlgorithm::world : PitchAlgorithm::utau;
        if (mode == 2)
        {
            track.voicebankDirectory = folder.getChildFile("fixture-bank");
            track.voicebankDirectory.createDirectory();
            track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");
        }
        NoteData n; n.id = "n"; n.label = "a"; n.startSeconds = .5; n.durationSeconds = 2;
        n.midiNote = n.sourceMidiCenter = 60; n.sourcePitchMeasured = true; n.nativePitchHandlesPlaced = true;
        n.contour = {{0,0,0,true},{2,0,0,true}};
        n.pitchControlPoints = {{0,60},{.4,63},{.8,65},{1.2,62},{2,60}};
        if (mode == 2) n.diffSingerPitchReference = n.pitchControlPoints;
        ClipData clip; clip.id = "c"; clip.durationSeconds = clip.sourceDurationSeconds = 3; clip.notes = {n};
        track.clips = {clip}; data.tracks = {track};
        ProjectModel model; model.resetDocument(data); I18n strings;
        PianoRollComponent roll(model, strings); roll.setSize(1400,1000); roll.setPixelsPerSecond(300);
        roll.setRowHeight(25); roll.setFocusedTrack("t"); roll.setFocusedClip("c");
        roll.setTool(PianoRollComponent::Tool::points); roll.setShowPitchLine(true);
        const auto refresh = [&] {model.dispatchPendingMessages(); roll.diagnosticRefresh();};
        const auto reset = [&] {model.resetDocument(data); refresh(); roll.setSelectedNoteIds({"n"});
            roll.setTool(PianoRollComponent::Tool::points);};
        const auto points = [&] {return model.snapshot().tracks[0].clips[0].notes[0].pitchControlPoints;};
        const auto notes = [&] {return model.snapshot().tracks[0].clips[0].notes.size();};
        const auto at = [&](double t, float midi) {return juce::Point<float>(roll.diagnosticEdgeX(.5+t),roll.diagnosticYForMidi(midi));};
        const auto event = [&](juce::Point<float> p, juce::Point<float> down) {
            return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),p,
                juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&roll,&roll,
                juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,p!=down);
        };
        const auto click = [&](juce::Point<float> p) {roll.mouseDown(event(p,p));roll.mouseUp(event(p,p));};
        const auto del = [&] {return roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));};
        reset(); const auto original = model.contentFingerprint();
        click(at(.8,65));
        check(prefix+"click_release_does_not_edit",model.contentFingerprint()==original&&!roll.diagnosticDraggingAnchor());
        if (mode == 1)
        {
            auto image = roll.createComponentSnapshot(juce::Rectangle<int>(190,(int)roll.diagnosticYForMidi(68),650,300));
            if (auto stream=folder.getChildFile("selected-after-release.png").createOutputStream())
                juce::PNGImageFormat().writeImageToStream(image,*stream);
        }
        check(prefix+"delete_after_release_keeps_note",del()&&notes()==1&&points().size()==4);
        const auto afterDelete=points();
        check(prefix+"deletes_clicked_point",std::none_of(afterDelete.begin(),afterDelete.end(),[](const auto& p){return p.timeSeconds==.8;}));
        const auto deleted = model.contentFingerprint(); del(); del();
        check(prefix+"repeat_delete_does_not_delete_note",notes()==1&&model.contentFingerprint()==deleted);
        check(prefix+"undo_restores_curve",model.undo()&&model.contentFingerprint()==original); refresh();
        check(prefix+"redo_restores_deletion",model.redo()&&model.contentFingerprint()==deleted); refresh();
        juce::String error; ProjectModel loaded;
        check(prefix+"save_reopen",model.save(folder.getChildFile(prefix+"deleted.hjpx"),error)
            &&loaded.load(folder.getChildFile(prefix+"deleted.hjpx"),error)
            &&loaded.snapshot().tracks[0].clips[0].notes[0].pitchControlPoints.size()==4);
        reset(); click(at(.4,63)); click(at(1.2,62)); del();
        const auto switched=points();
        check(prefix+"second_click_selects_new_point",switched.size()==4&&switched[1].timeSeconds==.4&&switched[2].timeSeconds==.8);
        reset(); const auto from=at(.8,65), to=at(.9,66);
        roll.mouseDown(event(from,from));roll.mouseDrag(event(to,from));roll.mouseUp(event(to,from));
        check(prefix+"drag_commits",points().size()==5&&std::abs(points()[2].targetMidi-66)<.01);
        // No model-message pump: Delete immediately after releasing must use the committed curve.
        roll.deleteSelectedNotes();
        check(prefix+"delete_immediately_after_drag",notes()==1&&points().size()==4&&points()[2].timeSeconds==1.2);
        reset(); click(at(.8,65)); click(at(1.5,72)); del();
        check(prefix+"blank_click_clears_point",notes()==1&&points().size()==5);
        reset(); click(at(.8,65)); roll.focusLost(juce::Component::focusChangedDirectly); del();
        check(prefix+"focus_elsewhere_restores_note_delete",notes()==0);
        reset(); click(at(.8,65)); roll.setTool(PianoRollComponent::Tool::note); del();
        check(prefix+"tool_change_restores_note_delete",notes()==0);
        reset(); click(at(.8,65)); roll.selectAllNotes(); del();
        check(prefix+"select_all_restores_note_delete",notes()==0);
        reset(); click(at(.8,65)); model.setNotePitchCurve("n",{{0,60},{.2,61},{2,60}},true);refresh();del();
        check(prefix+"stale_index_cannot_delete_another_point_or_note",notes()==1&&points().size()==3);
        reset(); click(at(0,60));del();
        check(prefix+"endpoint_preserves_context_menu_rule",notes()==1&&points().size()==5);
        reset(); roll.mouseDown(event(from,from)); del(); roll.mouseUp(event(from,from)); refresh();
        check(prefix+"delete_before_release_not_resurrected",notes()==1&&points().size()==4);
        reset(); click(at(.8,65));
        check(prefix+"backspace_matches_delete",roll.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey))&&notes()==1&&points().size()==4);
    }
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);
    folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok;
}
}
