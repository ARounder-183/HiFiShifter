#pragma once

namespace hachi
{
inline bool MainComponent::diagnosticClipSplit(const juce::File& folder)
{
    folder.createDirectory(); stopTimer(); setSize(1280, 800);
    bool ok = true;
    const auto check = [&](const char* name, bool passed)
    { ok = ok && passed; std::cout << name << '=' << passed << std::endl; };
    const auto close = [](double a, double b) { return std::abs(a - b) < 1.0e-7; };
    const auto source = folder.getChildFile("split-source.wav");
    juce::AudioBuffer<float> samples(2, 288000);
    for (int i = 0; i < samples.getNumSamples(); ++i)
        for (int c = 0; c < 2; ++c)
        {
            const auto time = i / 48000.0;
            samples.setSample(c, i, static_cast<float>(0.2 * std::sin(
                juce::MathConstants<double>::twoPi * ((227 + c * 159) * time + 17 * time * time))));
        }
    {
        auto stream = source.createOutputStream();
        if (!stream) return false;
        stream->setPosition(0); stream->truncate(); juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), 48000, 2, 32, {}, 0));
        check("source_written", writer && writer->writeFromAudioSampleBuffer(samples, 0, samples.getNumSamples()));
    }
    const auto exportAudio = [&](const ProjectData& data, const juce::String& name)
    {
        audio.syncProject(data);
        const auto file = folder.getChildFile(name + ".wav");
        juce::String error;
        check("export_succeeded", audio.exportWav(file, error, "backing", 0, 3.5, {48000, 2, 32}));
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatReader> reader(wav.createReaderFor(file.createInputStream().release(), true));
        juce::AudioBuffer<float> result;
        if (reader)
        {
            result.setSize(2, static_cast<int>(reader->lengthInSamples));
            reader->read(&result, 0, result.getNumSamples(), 0, true, true);
        }
        return result;
    };
    ProjectData preview;
    for (const bool accompaniment : {true, false})
    {
        ProjectData data;
        TrackData track; track.id = "backing"; track.name = accompaniment ? "Backing" : "Raw audio";
        track.compose = false; track.accompaniment = accompaniment; track.volume = 0.8f; track.pan = -0.2f;
        ClipData clip; clip.id = "source"; clip.sourceFile = source; clip.startSeconds = 0.5;
        clip.durationSeconds = 3.0; clip.sourceDurationSeconds = 4.5; clip.sourceOffsetSeconds = 0.25;
        clip.gain = 0.7f; clip.fadeInSeconds = 0.1; clip.fadeOutSeconds = 0.15;
        track.clips.push_back(clip); data.tracks.push_back(track);
        project.replace(data); project.dispatchPendingMessages(); stopTimer(); resized();
        const auto beforeAudio = exportAudio(data, accompaniment ? "backing-before" : "raw-before");
        const auto revision = project.revisionNumber();
        if (accompaniment)
        {
            timeline.setPixelsPerSecond(140.0f);
            timeline.setPlayheadSeconds(1.75);
            (void) timeline.createComponentSnapshot(timeline.getLocalBounds());
            juce::String clicked; double at = -1;
            const auto originalMenu = timeline.onClipMenu;
            timeline.onClipMenu = [&](const juce::String& id, double seconds, juce::Point<int>)
            { clicked = id; at = seconds; };
            const auto click = [&](float x, float y)
            {
                const juce::Point<float> p(x, y);
                const juce::MouseEvent event(juce::Desktop::getInstance().getMainMouseSource(), p,
                    juce::ModifierKeys(juce::ModifierKeys::rightButtonModifier), 0, 0, 0, 0, 0,
                    &timeline, &timeline, juce::Time::getCurrentTime(), p, juce::Time::getCurrentTime(), 1, false);
                timeline.mouseDown(event); timeline.mouseDrag(event); timeline.mouseUp(event);
            };
            click(76, static_cast<float>(timeline.getRulerHeight() + 24));
            check("right_click_mute_badge_opens_menu_without_muting", clicked == "source"
                && !project.snapshot().tracks[0].clips[0].muted && project.revisionNumber() == revision);
            click(280, static_cast<float>(timeline.getRulerHeight() + 54));
            check("right_click_targets_red_playhead_not_pointer_without_dragging", clicked == "source" && close(at, 1.75)
                && project.revisionNumber() == revision);
            timeline.onClipMenu = originalMenu;
            auto menu = clipContextMenu(clicked, at); bool enabled = false;
            for (juce::PopupMenu::MenuItemIterator it(menu); it.next();)
                if (it.getItem().itemID == 1) enabled = it.getItem().isEnabled && it.getItem().text == strings.text("clip.split");
            check("split_menu_enabled_inside_clip", enabled);
            clipContextMenuItemChosen(0, clicked, at);
            check("dismiss_menu_keeps_project", project.revisionNumber() == revision);
            clipContextMenuItemChosen(1, clicked, at);
            check("split_keeps_original_piece_selected", selectedClipId == clicked);
        }
        else (void) project.splitClip("source", 1.75);
        const auto after = project.snapshot();
        check("two_pieces_on_same_track", after.tracks.size() == 1 && after.tracks[0].clips.size() == 2);
        if (after.tracks[0].clips.size() != 2) return false;
        const auto& a = after.tracks[0].clips[0]; const auto& b = after.tracks[0].clips[1];
        const auto expectedSourceCut = accompaniment ? 1.25 : 1.875;
        check("split_preserves_timeline_and_source_position", a.id == "source" && b.id != a.id
            && close(a.startSeconds, .5) && close(a.durationSeconds, 1.25) && close(b.startSeconds, 1.75)
            && close(b.durationSeconds, 1.75) && close(b.sourceOffsetSeconds, .25 + expectedSourceCut)
            && close(a.sourceDurationSeconds, expectedSourceCut));
        check("split_preserves_gain_source_and_outer_fades", a.sourceFile == source && b.sourceFile == source
            && a.gain == .7f && b.gain == .7f && close(a.fadeInSeconds, .1) && a.fadeOutSeconds == 0
            && b.fadeInSeconds == 0 && close(b.fadeOutSeconds, .15));
        check("single_undo_revision", project.revisionNumber() == revision + 1);
        const auto afterAudio = exportAudio(after, accompaniment ? "backing-after" : "raw-after");
        float maxDifference = 0;
        const auto sameLength = beforeAudio.getNumSamples() > 0 && beforeAudio.getNumSamples() == afterAudio.getNumSamples();
        if (sameLength)
            for (int c = 0; c < 2; ++c) for (int i = 0; i < beforeAudio.getNumSamples(); ++i)
                maxDifference = std::max(maxDifference, std::abs(beforeAudio.getSample(c, i) - afterAudio.getSample(c, i)));
        std::cout << "split_audio_max_difference=" << maxDifference << std::endl;
        check("split_preserves_audio_including_seam", sameLength && maxDifference < 1.0e-6f);
        check("can_split_again_between_samples", project.splitClip(b.id, 2.31417).isNotEmpty());
        const auto twiceAudio = exportAudio(project.snapshot(), accompaniment ? "backing-twice" : "raw-twice");
        float twiceDifference = 0;
        const auto twiceLength = twiceAudio.getNumSamples() == beforeAudio.getNumSamples() && beforeAudio.getNumSamples() > 0;
        if (twiceLength)
            for (int c = 0; c < 2; ++c) for (int i = 0; i < beforeAudio.getNumSamples(); ++i)
                twiceDifference = std::max(twiceDifference, std::abs(beforeAudio.getSample(c, i) - twiceAudio.getSample(c, i)));
        check("fractional_cut_has_no_missing_or_doubled_sample", twiceLength && twiceDifference < 1.0e-6f);
        check("undo_second_split_independently", project.undo() && project.snapshot().tracks[0].clips.size() == 2);
        check("undo_restores_single_clip", project.undo() && project.snapshot().tracks[0].clips.size() == 1
            && close(project.snapshot().tracks[0].clips[0].durationSeconds, 3));
        check("redo_restores_both_pieces", project.redo() && project.snapshot().tracks[0].clips.size() == 2
            && project.snapshot().tracks[0].clips[1].id == b.id);
        const auto invalidRevision = project.revisionNumber();
        check("reject_edges_nan_missing_clip", project.splitClip(a.id, .5).isEmpty()
            && project.splitClip(b.id, 3.5).isEmpty() && project.splitClip("missing", 2).isEmpty()
            && project.splitClip(b.id, std::numeric_limits<double>::quiet_NaN()).isEmpty()
            && project.revisionNumber() == invalidRevision);
        auto menu = clipContextMenu(a.id, .5); bool disabled = false;
        for (juce::PopupMenu::MenuItemIterator it(menu); it.next();) if (it.getItem().itemID == 1) disabled = !it.getItem().isEnabled;
        check("menu_disabled_at_clip_edge", disabled);
        juce::String error; const auto saved = folder.getChildFile(accompaniment ? "backing.hjpx" : "raw.hjpx");
        ProjectModel reopened;
        check("save_reopen_keeps_split", project.save(saved, error) && reopened.load(saved, error)
            && reopened.snapshot().tracks[0].clips.size() == 2
            && close(reopened.snapshot().tracks[0].clips[1].sourceOffsetSeconds, b.sourceOffsetSeconds));
        if (accompaniment) preview = after;
    }
    ProjectData score; TrackData track; track.id = "score";
    ClipData clip; clip.id = "score-clip"; clip.startSeconds = 1; clip.durationSeconds = 4;
    clip.sourceFile = source; clip.sourceOffsetSeconds = .3; clip.sourceDurationSeconds = 3;
    clip.sourceTimeMap = {{0,0}, {1,.5}, {2,2}, {4,3}}; clip.muted = true;
    NoteData early; early.id = "early"; early.startSeconds = .2; early.durationSeconds = .5;
    NoteData middle; middle.id = "middle"; middle.startSeconds = 1; middle.durationSeconds = 2;
    middle.utauFlags = "Mb73g-20"; middle.utauFlagCurveEnabled = true;
    middle.utauFlagCurves = {{"g", {{0,-20}, {2,20}}}, {"DS:REF:BREC", {{0,-60}, {2,-20}}}};
    middle.pitchControlPoints = {{0,60}, {2,64}}; middle.diffSingerPitchReference = {{0,59}, {2,63}};
    middle.diffSingerPitchOffset = {{0,-1}, {2,1}};
    middle.diffSingerTiming = R"({"phonemes":[{"phone":"a","start":-0.1,"end":2}]})";
    NoteData late; late.id = "late"; late.startSeconds = 3; late.durationSeconds = .5; late.utauFlags = "Mb42";
    clip.notes = {early, middle, late}; track.clips.push_back(clip); score.tracks.push_back(track);
    project.replace(score); const auto before = project.revisionNumber();
    const auto rightId = project.splitClip(clip.id, 3);
    const auto divided = project.snapshot();
    check("note_clip_split_succeeded", rightId.isNotEmpty() && divided.tracks[0].clips.size() == 2);
    if (rightId.isEmpty()) return false;
    const auto& left = divided.tracks[0].clips[0]; const auto& right = divided.tracks[0].clips[1];
    check("click_inside_note_snaps_to_nearest_gap", close(left.durationSeconds,1) && close(right.startSeconds,2));
    check("nonlinear_source_map_rebased", close(left.sourceDurationSeconds,.5) && close(right.sourceOffsetSeconds,.8)
        && close(right.sourceDurationSeconds,2.5) && close(right.sourceTimeMap.front().sourceSeconds,0)
        && close(right.sourceTimeMap.back().targetSeconds,3));
    check("muted_state_preserved", left.muted && right.muted);
    check("notes_partitioned_without_cutting", left.notes.size()==1 && right.notes.size()==2
        && left.notes[0].id==early.id && right.notes[0].id==middle.id && right.notes[1].id==late.id
        && close(right.notes[0].durationSeconds,2) && close(right.notes[0].startSeconds,0)
        && close(right.notes[1].startSeconds,2));
    if(left.notes.size()!=1 || right.notes.size()!=2)return false;
    check("flags_and_pitch_remain_note_local",right.notes[0].utauFlags==middle.utauFlags
        &&close(flagCurveValueAt(flagCurvePointsFor(right.notes[0],"g"),.25),-15)
        &&close(right.notes[0].pitchControlPoints.front().targetMidi,60)
        &&close(right.notes[0].diffSingerPitchReference.front().targetMidi,59)
        &&close(flagCurveValueAt(flagCurvePointsFor(right.notes[0],"DS:REF:BREC"),0),-60));
    // Compare the complete saved note content after converting local starts
    // back to project time. Clip ownership is allowed to change; note data is not.
    const auto noteBytes=[&](const ProjectData& input)
    {
        ProjectData canonical;TrackData owner;owner.id="notes";ClipData notes;notes.id="notes";notes.durationSeconds=60;
        for(const auto& t:input.tracks)for(const auto& c:t.clips)for(auto n:c.notes)
        {n.startSeconds=std::round((c.startSeconds+n.startSeconds)*1.0e9)/1.0e9;n.clipPartId.clear();notes.notes.push_back(std::move(n));}
        std::sort(notes.notes.begin(),notes.notes.end(),[](const auto& a,const auto& b){return a.id<b.id;});
        owner.clips={notes};canonical.tracks={owner};ProjectModel serialiser;serialiser.replace(canonical);
        juce::String error;juce::MemoryBlock bytes;const auto file=folder.getChildFile("note-content-check.hjpx");
        if(serialiser.save(file,error))file.loadFileAsData(bytes);return bytes;
    };
    const auto originalBytes=noteBytes(score);
    check("all_note_fields_unchanged",originalBytes.getSize()>0&&originalBytes==noteBytes(divided));
    check("note_gap_split_one_undo",project.revisionNumber()==before+1&&project.undo()
        &&project.snapshot().tracks[0].clips.size()==1&&originalBytes==noteBytes(project.snapshot()));
    check("note_gap_split_redo",project.redo()&&project.snapshot().tracks[0].clips.size()==2
        &&originalBytes==noteBytes(project.snapshot()));
    juce::String noteError;ProjectModel loaded;const auto noteFile=folder.getChildFile("note-gap.hjpx");
    const auto savedNotes=project.save(noteFile,noteError)&&loaded.load(noteFile,noteError);
    const auto loadedBytes=noteBytes(loaded.snapshot());
    if(savedNotes&&originalBytes!=loadedBytes)
    {
        folder.getChildFile("notes-before.xml").replaceWithText(juce::ValueTree::readFromData(originalBytes.getData(),originalBytes.getSize()).toXmlString());
        folder.getChildFile("notes-reopened.xml").replaceWithText(juce::ValueTree::readFromData(loadedBytes.getData(),loadedBytes.getSize()).toXmlString());
    }
    check("note_gap_save_reopen",savedNotes
        &&originalBytes==noteBytes(loaded.snapshot()));
    const auto gapCase=[&](const char* name,std::vector<std::pair<double,double>> spans,double clicked,
                           std::optional<double> expected, bool compose=true, bool backing=false)
    {
        auto fixture=score;auto& owner=fixture.tracks[0];owner.compose=compose;owner.accompaniment=backing;
        auto& c=owner.clips[0];c.notes.clear();c.durationSeconds=8;c.sourceTimeMap.clear();c.sourceFile=juce::File{};c.sourceDurationSeconds=8;
        for(std::size_t i=0;i<spans.size();++i)
        {auto n=middle;n.id="test-"+juce::String(static_cast<int>(i));n.startSeconds=spans[i].first;n.durationSeconds=spans[i].second;c.notes.push_back(n);}
        project.replace(fixture);const auto revision=project.revisionNumber();const auto bytes=noteBytes(fixture);
        const auto allowed=project.canSplitClip(c.id,c.startSeconds+clicked);
        const auto id=project.splitClip(c.id,c.startSeconds+clicked);const auto after=project.snapshot();
        if(!expected){check(name,!allowed&&id.isEmpty()&&project.revisionNumber()==revision&&after.tracks[0].clips.size()==1);return;}
        check(name,allowed&&id.isNotEmpty()&&after.tracks[0].clips.size()==2
            &&close(after.tracks[0].clips[1].startSeconds,c.startSeconds+*expected));
        if(compose&&!backing)check("snapped_split_keeps_every_note_complete",bytes.getSize()>0&&bytes==noteBytes(after));
    };
    gapCase("gap_click_keeps_exact_time",{{.5,1},{3,1}},2.125,2.125);
    gapCase("inside_first_note_uses_following_gap",{{.5,1},{3,1}},.8,1.5);
    gapCase("inside_last_note_uses_preceding_gap",{{.5,1},{3,1}},3.8,3);
    gapCase("touching_notes_use_common_boundary",{{.5,1},{1.5,1}},2,1.5);
    gapCase("decimal_touching_notes_are_not_an_overlap",{{.1,.2},{.3,.2}},.2,.3);
    gapCase("nearest_previous_gap",{{.5,.5},{2,2},{5,.5}},2.25,2);
    gapCase("nearest_following_gap",{{.5,.5},{2,2},{5,.5}},3.75,4);
    gapCase("tie_prefers_earlier_gap",{{.5,.5},{2,2},{5,.5}},3,2);
    gapCase("unsorted_notes_find_same_gap",{{5,.5},{2,2},{.5,.5}},3.75,4);
    gapCase("overlap_chain_skips_unsafe_boundary",{{.5,3},{1,1},{3,1},{5,.5}},2,4);
    gapCase("nested_note_does_not_hide_long_note_end",{{.5,3},{1,.5},{4,.5}},2,3.5);
    gapCase("all_overlapping_disables_split",{{.5,4},{1,1},{2,1}},2,{});
    gapCase("one_note_disables_split",{{.5,4}},2,{});
    auto disabledMenu=clipContextMenu("score-clip",3);bool disabledSplit=false;
    for(juce::PopupMenu::MenuItemIterator it(disabledMenu);it.next();)if(it.getItem().itemID==1)disabledSplit=!it.getItem().isEnabled;
    check("single_note_menu_is_disabled",disabledSplit);
    gapCase("empty_tuning_clip_splits_at_click",{},2.345,2.345);
    gapCase("audio_track_keeps_exact_split",{{.5,4}},2.345,2.345,false);
    gapCase("accompaniment_keeps_exact_split",{{.5,4}},2.345,2.345,false,true);
    // The real right-click action must use the same snapping as model callers.
    project.replace(score);project.dispatchPendingMessages();stopTimer();
    clipContextMenuItemChosen(1,"score-clip",3.7);project.dispatchPendingMessages();stopTimer();
    check("menu_action_uses_note_gap",project.snapshot().tracks[0].clips.size()==2
        &&close(project.snapshot().tracks[0].clips[1].startSeconds,4)
        &&selectedClipId=="score-clip"&&originalBytes==noteBytes(project.snapshot()));
    // A merged region uses its visible parent notes, with each source still aligned.
    project.replace(score);const auto part=project.splitClip("score-clip",3.7);
    check("merge_after_note_gap_split",project.mergeClips({"score-clip",part})=="score-clip");
    const auto mergedBytes=noteBytes(project.snapshot());const auto redivided=project.splitClip("score-clip",2.1);
    check("merged_note_region_snaps_without_cutting",redivided.isNotEmpty()
        &&close(project.snapshot().tracks[0].clips[1].startSeconds,2)
        &&mergedBytes==noteBytes(project.snapshot()));
    project.replace(preview); project.dispatchPendingMessages(); stopTimer();
    timelineHorizontalZoom = 260; timeline.setPixelsPerSecond(260); timelineViewport.setViewPosition(0, 0);
    focusClip(preview.tracks[0].clips[0].id); resized();
    if (auto stream = folder.getChildFile("clip-split.png").createOutputStream())
        check("preview_written", juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()), *stream));
    return ok;
}
}
