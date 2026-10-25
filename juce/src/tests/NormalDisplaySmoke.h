#pragma once
#include "../FlagCurveDrawing.h"
namespace hachi
{
inline bool MainComponent::diagnosticNormalDisplay(const juce::File& folder)
{
    folder.createDirectory(); stopTimer(); setSize(1280,800); bool ok=true;
    const auto check=[&](const char* name,bool pass){ok=ok&&pass;std::cout<<name<<'='<<pass<<std::endl;};
    const auto event=[](juce::Component& c,juce::Point<float> at,juce::Point<float> down)
    {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
        juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),0,0,0,0,0,&c,&c,
        juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
    ProjectData data; TrackData track;track.id="voice";track.pitchAlgorithm=PitchAlgorithm::utau;
    const auto make=[](const juce::String& id,double start,int index)
    {ClipData c;c.id=id;c.startSeconds=start;c.durationSeconds=c.sourceDurationSeconds=2;
        for(int i=0;i<2;++i){NoteData n;n.id=id+juce::String(i);n.label="a";n.startSeconds=.25+i;n.durationSeconds=.5;
            n.midiNote=60.0f+static_cast<float>(index+i)*2;n.utauFlagCurveEnabled=true;n.utauAutoPitchTransition=false;
            n.pitchControlPoints={{0,n.midiNote},{.5,n.midiNote+.5f}};n.diffSingerPitchReference=n.pitchControlPoints;c.notes.push_back(n);}return c;};
    track.clips={make("left",1,0),make("right",3,2),make("hidden",6,4)};
    auto empty=make("empty",9,0);empty.notes.clear();track.clips.push_back(empty);
    TrackData other=track;other.id="other";other.clips={make("other",1,6)};
    TrackData backing;backing.id="backing";backing.accompaniment=true;backing.compose=false;backing.clips={make("backing",1,8)};
    data.tracks={track,other,backing};project.replace(data);project.dispatchPendingMessages();stopTimer();focusClip("left");
    const auto refresh=[&]{project.dispatchPendingMessages();stopTimer();pianoRoll.diagnosticRefresh();};
    const auto menu=[&](const juce::String& id,bool on)
    {int value=-1;auto m=clipContextMenu(id,2);for(juce::PopupMenu::MenuItemIterator it(m);it.next();)
        if(it.getItem().itemID==4){if(it.getItem().text!=strings.text(on?"clip.stopNormalDisplay":"clip.startNormalDisplay"))return -2;value=it.getItem().isEnabled?1:0;}return value;};
    check("normal_display_starts_disabled",pianoRoll.diagnosticHitCount()==2&&menu("right",false)==1);
    check("empty_tuning_region_can_enable",menu("empty",false)==1);
    check("backing_region_cannot_enable",menu("backing",false)==0);
    const auto hash=AudioEngine::diagnosticUtauRenderKey(project.snapshot(),"right");const auto rev=project.revisionNumber();
    clipContextMenuItemChosen(4,"right",4);refresh();
    check("menu_enables_only_clicked_region",project.snapshot().tracks[0].clips[1].showNormalDisplay
        &&!project.snapshot().tracks[0].clips[0].showNormalDisplay&&pianoRoll.diagnosticHitCount()==4&&menu("right",true)==1);
    check("display_change_keeps_render_cache",!hash.empty()&&hash==AudioEngine::diagnosticUtauRenderKey(project.snapshot(),"right"));
    check("toggle_is_one_undo",project.revisionNumber()==rev+1&&project.undo()&&!project.snapshot().tracks[0].clips[1].showNormalDisplay);
    check("redo_restores_toggle",project.redo()&&project.snapshot().tracks[0].clips[1].showNormalDisplay);refresh();
    const auto noOp=project.revisionNumber();project.setClipNormalDisplay("right",true);project.setClipNormalDisplay("backing",true);project.setClipNormalDisplay("missing",true);
    check("invalid_and_unchanged_toggles_noop",project.revisionNumber()==noOp);
    project.setClipNormalDisplay("other",true);refresh();
    check("other_tracks_not_made_editable",pianoRoll.diagnosticHitCount()==4);
    pianoRoll.setPixelsPerSecond(120);pianoRoll.setRowHeight(22);pianoRoll.setTool(PianoRollComponent::Tool::note);
    const auto rightAt=pianoRoll.diagnosticHitBounds(2).getCentre();
    pianoRoll.mouseDown(event(pianoRoll,rightAt,rightAt));pianoRoll.mouseUp(event(pianoRoll,rightAt,rightAt));
    check("click_extra_note_keeps_active_region",selectedClipId=="left"&&selectedNoteId=="right0"&&pianoRoll.diagnosticHitCount()==4);
    const auto raised=rightAt.translated(0,-22);
    pianoRoll.mouseDown(event(pianoRoll,rightAt,rightAt));pianoRoll.mouseDrag(event(pianoRoll,raised,rightAt));pianoRoll.mouseUp(event(pianoRoll,raised,rightAt));refresh();
    check("extra_note_can_be_dragged",std::abs(project.snapshot().tracks[0].clips[1].notes[0].midiNote-65.0f)<.01f
        &&project.snapshot().tracks[0].clips[0].notes[0].midiNote==60);
    (void)project.undo();refresh();
    pianoRoll.selectAllNotes();check("select_all_includes_enabled_region",pianoRoll.selectedNoteIds().size()==4&&selectedClipId=="left");
    check("delete_all_visible_notes",pianoRoll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey))
        &&project.snapshot().tracks[0].clips[0].notes.empty()&&project.snapshot().tracks[0].clips[1].notes.empty()
        &&project.snapshot().tracks[0].clips[2].notes.size()==2&&project.snapshot().tracks[1].clips[0].notes.size()==2);
    (void)project.undo();refresh();focusClip("left");
    pianoRoll.setTool(PianoRollComponent::Tool::points);const auto selectedAt=pianoRoll.diagnosticHitBounds(2).getCentre();
    pianoRoll.mouseDown(event(pianoRoll,selectedAt,selectedAt));pianoRoll.mouseUp(event(pianoRoll,selectedAt,selectedAt));
    clipContextMenuItemChosen(4,"right",4);refresh();
    check("disable_hides_and_clears_extra_selection",pianoRoll.diagnosticHitCount()==2&&pianoRoll.selectedNoteIds().empty()&&selectedClipId=="left"&&menu("right",false)==1);
    project.setClipNormalDisplay("right",true);project.setClipNoteHints("right",true);refresh();
    const auto enabled=project.snapshot();
    pianoRoll.setTool(PianoRollComponent::Tool::draw);
    pianoRoll.setDrawAction(PianoRollComponent::DrawAction::notes);
    const juce::Point<float> drawAt(pianoRoll.diagnosticEdgeX(3.9),pianoRoll.diagnosticYForMidi(70));
    pianoRoll.mouseDown(event(pianoRoll,drawAt,drawAt));pianoRoll.mouseUp(event(pianoRoll,drawAt,drawAt));refresh();
    check("drawing_in_extra_region_uses_its_ownership",project.snapshot().tracks[0].clips[1].notes.size()==3
        &&project.snapshot().tracks[0].clips[0].notes.size()==2&&selectedClipId=="left");
    (void)project.undo();refresh();

    const auto equal=[](const juce::Image& a,const juce::Image& b)
    {if(a.getBounds()!=b.getBounds())return false;for(int y=0;y<a.getHeight();++y)for(int x=0;x<a.getWidth();++x)
        if(a.getPixelAt(x,y)!=b.getPixelAt(x,y))return false;return true;};
    for(int mode=0;mode<5;++mode)
    {
        auto source=enabled;source.tracks.resize(1);source.tracks[0].clips.resize(2);auto& t=source.tracks[0];
        if(mode==0)t.pitchAlgorithm=PitchAlgorithm::world;
        else t.utauMode=mode==1?UtauMode::classic:mode==2?UtauMode::jie:UtauMode::mou;
        if(mode==4){t.voicebankDirectory=folder.getChildFile("fixture-bank");t.voicebankDirectory.createDirectory();
            t.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");}
        ProjectModel model;model.replace(source);PianoRollComponent roll(model,strings);
        roll.setPixelsPerSecond(120);roll.setRowHeight(22);roll.setFocusedTrack("voice");roll.setFocusedClip("left");
        roll.setShowPitchLine(true);roll.setShowUtauWaveforms(true);roll.setReadsVoicebankInBackground(false);
        auto waves=std::make_shared<std::vector<UtauNoteWaveform>>();
        for(const auto& c:t.clips)for(const auto& n:c.notes){UtauNoteWaveform w;w.noteId=n.id;w.audioHash=AudioEngine::utauNoteAudioHash(n);
            w.durationSeconds=.5;w.unshapedMaxima.assign(60,.5f);w.unshapedMinima.assign(60,-.5f);waves->push_back(w);}roll.setUtauNoteWaveforms(waves);
        check("all_engines_normal_region_edit_hits",roll.diagnosticHitCount()==4);
        const juce::Rectangle<int> area(58,static_cast<int>(roll.diagnosticYForMidi(72)),670,350);
        for(const auto tool:{PianoRollComponent::Tool::note,PianoRollComponent::Tool::points,PianoRollComponent::Tool::amplitude,PianoRollComponent::Tool::flagCurve})
        {
            roll.setTool(tool);const auto pinned=roll.createComponentSnapshot(area);
            roll.setFocusedClip({});const auto overview=roll.createComponentSnapshot(area);
            check("normal_display_matches_full_editable_paint",equal(pinned,overview));roll.setFocusedClip("left");
        }
        roll.setTool(PianoRollComponent::Tool::points);
        if(mode==4)check("ds_references_available_in_both_regions",!roll.diagnosticDiffSingerPitchReference("left0").isEmpty()
            &&!roll.diagnosticDiffSingerPitchReference("right0").isEmpty());
        if(mode==1)
        {
            roll.setTool(PianoRollComponent::Tool::flagCurve);roll.setFlagLaneFlag("g");roll.setFlagEditMode(PianoRollComponent::FlagEditMode::continuous);
            const juce::Point<float> from(roll.diagnosticEdgeX(1.5),roll.flagLaneY(10)),to(roll.diagnosticEdgeX(4.5),roll.flagLaneY(30));
            roll.mouseDown(event(roll,from,from));roll.mouseDrag(event(roll,to,from));roll.mouseUp(event(roll,to,from));
            check("continuous_flag_edits_both_regions",!flagCurvePointsFor(model.snapshot().tracks[0].clips[0].notes[0],"g").empty()
                &&!flagCurvePointsFor(model.snapshot().tracks[0].clips[1].notes[0],"g").empty());
        }
        if(mode==1||mode==4){roll.setTool(PianoRollComponent::Tool::points);auto out=folder.getChildFile(mode==1?"UTAU-normal.png":"DS-normal.png").createOutputStream();
            check("normal_display_preview_written",out&&juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(area),*out));}
        roll.setSourceEditMode(true);check("source_edit_remains_single_region",roll.diagnosticHitCount()==2);
        roll.setSourceEditMode(false);roll.setFocusedClip("right");model.setClipNormalDisplay("right",false);model.dispatchPendingMessages();roll.diagnosticRefresh();
        check("active_region_remains_normal_when_disabled",roll.diagnosticHitCount()==(mode==0?4:2));
    }
    ProjectModel model;model.replace(enabled);PianoRollComponent roll(model,strings);roll.setFocusedClip("left");
    model.setClipNormalDisplay("hidden",true);model.dispatchPendingMessages();roll.diagnosticRefresh();
    check("multiple_regions_can_be_kept_editable",roll.diagnosticHitCount()==6);
    roll.setFocusedClip("other");check("track_switch_does_not_leak_normal_regions",roll.diagnosticHitCount()==2);
    roll.setFocusedClip("left");model.setClipNormalDisplay("right",false);model.dispatchPendingMessages();roll.diagnosticRefresh();
    check("hint_and_normal_display_are_independent",model.snapshot().tracks[0].clips[1].showNoteHints&&roll.diagnosticHitCount()==4);

    juce::String error;const auto file=folder.getChildFile("normal-display.hjpx");ProjectModel saved;
    check("normal_display_save_reopen",project.save(file,error)&&saved.load(file,error)&&saved.snapshot().tracks[0].clips[1].showNormalDisplay
        &&!saved.snapshot().tracks[0].clips[0].showNormalDisplay);
    const auto duplicate=saved.duplicateClip("right",12);check("duplicate_inherits_normal_display",duplicate.isNotEmpty()&&saved.snapshot().tracks[0].clips.back().showNormalDisplay);
    const auto split=saved.splitClip("right",4);const auto find=[&](const juce::String& id){for(const auto& tr:saved.snapshot().tracks)for(const auto& c:tr.clips)if(c.id==id)return c.showNormalDisplay;return false;};
    check("split_inherits_normal_display",split.isNotEmpty()&&find("right")&&find(split));
    saved.setClipNormalDisplay("right",false);const auto merged=saved.mergeClips({"right",split});
    check("merge_preserves_enabled_normal_display",merged=="right"&&find(merged));
    saved.setClipNormalDisplay(merged,false);const auto again=saved.splitClip(merged,4);
    check("merged_off_state_survives_split",again.isNotEmpty()&&!find(merged)&&!find(again));
    juce::MemoryBlock bytes;file.loadFileAsData(bytes);auto legacy=juce::ValueTree::readFromData(bytes.getData(),bytes.getSize());
    for(auto tr:legacy)for(auto c:tr)if(c.hasType("Clip"))c.removeProperty("showNormalDisplay",nullptr);
    const auto oldFile=folder.getChildFile("legacy.hjpx");if(auto out=oldFile.createOutputStream()){out->setPosition(0);out->truncate();legacy.writeToStream(*out);}
    ProjectModel old;check("old_projects_default_normal_display_off",old.load(oldFile,error)&&!old.snapshot().tracks[0].clips[1].showNormalDisplay);
    return ok;
}
}
