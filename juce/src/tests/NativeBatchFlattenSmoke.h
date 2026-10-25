#pragma once
#include "../NativeSharedEnvelope.h"
namespace hachi
{
inline bool runNativeBatchFlattenSmoke()
{
    bool ok = true;
    const auto check = [&](const char* name, bool pass)
    { ok &= pass; std::cout << name << '=' << pass << std::endl; };
    I18n strings;
    ProjectData data;
    TrackData track; track.id = "native-batch"; track.pitchAlgorithm = PitchAlgorithm::nsfHifigan;
    for (int region = 0; region < 2; ++region)
    {
        ClipData clip; clip.id = "region" + juce::String(region);
        clip.startSeconds = region == 0 ? 1.0 : 4.0;
        clip.durationSeconds = clip.sourceDurationSeconds = 1.5;
        const auto count = region == 0 ? 3 : 2;
        for (int index = 0; index < count; ++index)
        {
            NoteData note; note.id = clip.id + "-" + juce::String(index);
            note.label = "a"; note.startSeconds = region == 0 ? index * .5 : index * .8;
            note.durationSeconds = .5;
            note.midiNote = 60.31f + static_cast<float>(region * 3 + index) * 1.22f;
            note.sourceMidiCenter = note.midiNote - .2f;
            note.nativeSourceStartSeconds = index * .5;
            note.nativeSourceEndSeconds = index * .5 + .5;
            note.contour = {{0, -90, -80, true, -30, true},
                            {.25, 80, 60, true, 40, true},
                            {.5, -20, -10, false, 20, true}};
            note.pitchControlPoints = {{0, note.midiNote - 1},
                                       {.25, note.midiNote}, {.5, note.midiNote + 1}};
            clip.notes.push_back(note);
        }
        track.clips.push_back(clip);
    }
    data.tracks.push_back(track);
    const auto unchangedSource = [](const NoteData& a, const NoteData& b)
    {
        if (a.startSeconds != b.startSeconds || a.durationSeconds != b.durationSeconds
            || a.sourceMidiCenter != b.sourceMidiCenter
            || a.nativeSourceStartSeconds != b.nativeSourceStartSeconds
            || a.nativeSourceEndSeconds != b.nativeSourceEndSeconds
            || a.contour.size() != b.contour.size()) return false;
        for (std::size_t i = 0; i < a.contour.size(); ++i)
            if (a.contour[i].timeSeconds != b.contour[i].timeSeconds
                || a.contour[i].relativeCents != b.contour[i].relativeCents
                || a.contour[i].withoutVibratoCents != b.contour[i].withoutVibratoCents
                || a.contour[i].voiced != b.contour[i].voiced) return false;
        return true;
    };
    for (const auto tool : {PianoRollComponent::Tool::note, PianoRollComponent::Tool::points})
    {
        ProjectModel model; model.replace(data);
        PianoRollComponent roll(model, strings);
        roll.setBounds(0, 0, 1400, 900); roll.setPixelsPerSecond(160);
        roll.setFocusedTrack(track.id); roll.setFocusedClip(track.clips[0].id);
        roll.setTool(tool); roll.diagnosticRefresh();
        const auto event = [&](juce::Point<float> at, juce::Point<float> down, int clicks = 1)
        {
            return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(), at,
                juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), 1.0f, 0, 0, 0, 0,
                &roll, &roll, juce::Time::getCurrentTime(), down,
                juce::Time::getCurrentTime(), clicks, at != down);
        };
        auto selection = roll.diagnosticHitBounds(0);
        for (int i = 1; i < 4; ++i) selection = selection.getUnion(roll.diagnosticHitBounds(i));
        selection = selection.expanded(10, 45);
        const auto down = selection.getTopLeft(), end = selection.getBottomRight();
        roll.mouseDown(event(down, down)); roll.mouseDrag(event(end, down)); roll.mouseUp(event(end, down));
        check("native_batch_marquee_across_regions", roll.selectedNoteIds().size() == 4);
        const auto before = model.snapshot(); const auto fingerprint = model.contentFingerprint();
        const auto revision = model.revisionNumber();
        if (tool == PianoRollComponent::Tool::points)
        {
            // Double-click inside a native note body, between its existing
            // anchors: this used to flatten the entire selected batch.
            const auto bounds = roll.diagnosticHitBounds(1);
            const auto at = juce::Point<float>(bounds.getX()+bounds.getWidth()*.25f, bounds.getCentreY());
            const auto doubleClick = [&](juce::Point<float> p) {
                roll.mouseDown(event(p,p)); roll.mouseUp(event(p,p));
                roll.mouseDown(event(p,p,2)); roll.mouseDoubleClick(event(p,p,2));
                roll.mouseUp(event(p,p,2));
                model.dispatchPendingMessages(); roll.diagnosticRefresh();
            };
            doubleClick(at);
            const auto after = model.snapshot();
            const auto& original = before.tracks[0].clips[0].notes[1];
            const auto& edited = after.tracks[0].clips[0].notes[1];
            check("point_double_click_inserts_native_anchor", edited.pitchControlPoints.size()==4);
            check("point_double_click_never_snaps_or_flattens", edited.midiNote==original.midiNote
                && !edited.nativeIndependentPitch && edited.sourceMidiCenter==original.sourceMidiCenter
                && edited.startSeconds==original.startSeconds && edited.durationSeconds==original.durationSeconds
                && edited.nativeSourceStartSeconds==original.nativeSourceStartSeconds
                && edited.nativeSourceEndSeconds==original.nativeSourceEndSeconds
                && edited.pitchControlPoints.front().targetMidi==original.pitchControlPoints.front().targetMidi
                && edited.pitchControlPoints.back().targetMidi==original.pitchControlPoints.back().targetMidi);
            bool othersKept=true;
            for(int r=0;r<2;++r) for(int n=0;n<(r==0?3:2);++n)
                if(r!=0 || n!=1) {
                    const auto& x=before.tracks[0].clips[r].notes[n];
                    const auto& y=after.tracks[0].clips[r].notes[n];
                    othersKept &= x.midiNote==y.midiNote && x.pitchControlPoints.size()==y.pitchControlPoints.size()
                        && x.contour[1].manualTargetCents==y.contour[1].manualTargetCents;
                }
            check("point_double_click_leaves_other_selected_notes",othersKept);
            const auto addedFingerprint=model.contentFingerprint();
            check("point_insertion_single_undo",model.revisionNumber()==revision+1
                && model.undo() && model.contentFingerprint()==fingerprint);
            check("point_insertion_redo",model.redo() && model.contentFingerprint()==addedFingerprint);
            model.dispatchPendingMessages(); roll.diagnosticRefresh();
            doubleClick(roll.diagnosticHitBounds(1).getCentre());
            check("point_existing_anchor_double_click_no_flatten",model.contentFingerprint()==addedFingerprint);
            continue;
        }
        // Flattening now has its explicit context action. Default double-click
        // is preutterance alignment and must never flatten or quantize pitch.
        roll.applyNoteMenuChoice(before.tracks[0].clips[0].notes[1].id,18,0,{});
        const auto after = model.snapshot();
        model.dispatchPendingMessages(); roll.diagnosticRefresh();
        bool shownFlat = true, renderedFlat = true;
        for (int region = 0; region < 2; ++region)
        {
            const auto& clip = after.tracks[0].clips[region];
            const auto request = AudioEngine::diagnosticNativeRequest(clip, after.tracks[0]);
            for (int index = 0; index < (region == 0 ? 3 : 1); ++index)
            {
                const auto& note = clip.notes[index];
                for (double u : {.01, .25, .5, .75, .99})
                {
                    const auto t = note.startSeconds + note.durationSeconds * u;
                    const auto shown = roll.diagnosticPitchLineAt(note.id, clip.startSeconds + t);
                    shownFlat &= shown && std::abs(*shown - note.midiNote) < 1.e-4f;
                    const auto midi = request.targetMidi[static_cast<std::size_t>(std::lround(t / .005))];
                    renderedFlat &= midi == 0.f || std::abs(midi - note.midiNote) < 1.e-4f;
                }
            }
        }
        check("native_batch_display_independently_flat", shownFlat);
        check("native_batch_render_independently_flat", renderedFlat);
        const auto gainGroups = nativeSharedEnvelopes(after.tracks[0]);
        check("native_batch_keeps_shared_loudness", gainGroups.contains("region0-0")
            && gainGroups.contains("region0-1")
            && gainGroups.at("region0-0").points == gainGroups.at("region0-1").points);
        bool allFlat = true, allSourceKept = true;
        for (int region = 0; region < 2; ++region)
            for (int index = 0; index < (region == 0 ? 3 : 1); ++index)
            {
                const auto& original = before.tracks[0].clips[region].notes[index];
                const auto& note = after.tracks[0].clips[region].notes[index];
                allFlat &= note.nativeIndependentPitch && note.midiNote == original.midiNote
                    && note.pitchControlPoints.size() == 2 && note.drift == 0 && note.modulation == 0
                    && std::all_of(note.pitchControlPoints.begin(), note.pitchControlPoints.end(),
                        [&](const auto& p) { return p.targetMidi == note.midiNote; })
                    && std::all_of(note.contour.begin(), note.contour.end(),
                        [](const auto& p) { return p.hasManualTarget && p.manualTargetCents == 0; });
                allSourceKept &= unchangedSource(original, note);
            }
        check("native_batch_context_flatten_preserves_pitch_centers", allFlat);
        check("native_batch_retains_selection", roll.selectedNoteIds().size() == 4);
        check("native_batch_keeps_source_and_timing", allSourceKept);
        const auto& untouched = after.tracks[0].clips[1].notes[1];
        check("native_batch_unselected_unchanged", untouched.midiNote == track.clips[1].notes[1].midiNote
            && untouched.pitchControlPoints.size() == 3 && untouched.contour[1].manualTargetCents == 40);
        const auto flattenedFingerprint = model.contentFingerprint();
        check("native_batch_one_undo_restores_all", model.revisionNumber() == revision + 1
            && model.undo() && model.contentFingerprint() == fingerprint);
        check("native_batch_redo_restores_all", model.redo() && model.contentFingerprint() == flattenedFingerprint);
        model.undo(); model.dispatchPendingMessages(); roll.diagnosticRefresh();
        // A different, unselected note should still be a single-note action.
        const auto outside = roll.diagnosticHitBounds(4).getCentre();
        roll.mouseDown(event(outside, outside)); roll.mouseUp(event(outside, outside));
        roll.applyNoteMenuChoice(before.tracks[0].clips[1].notes[1].id,18,0,{});
        const auto single = model.snapshot();
        check("native_batch_unselected_click_targets_only_it", roll.selectedNoteIds().size() == 1
            && single.tracks[0].clips[1].notes[1].pitchControlPoints.size() == 2
            && single.tracks[0].clips[0].notes[0].pitchControlPoints.size() == 3
            && single.tracks[0].clips[1].notes[0].pitchControlPoints.size() == 3);
    }
    return ok;
}
}
