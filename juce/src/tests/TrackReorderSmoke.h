#pragma once
namespace hachi
{
inline bool MainComponent::diagnosticTrackReorder(const juce::File& folder)
{
    folder.createDirectory();stopTimer();setSize(1400,900);
    bool ok=true;int checks=0;
    const auto check=[&](const char* key,bool value){++checks;ok&=value;std::cout<<key<<'='<<value<<std::endl;};
    ProjectData original;
    for(int i=0;i<8;++i)
    {
        TrackData track;track.id="t"+juce::String(i);track.name=track.id;
        track.trackType=i==2?TrackType::diffSinger:TrackType::utau;
        track.pitchAlgorithm=PitchAlgorithm::utau;track.compose=true;
        track.volume=.5f+.05f*i;track.pan=.03f*i;track.utauPianoPreview=i%2==0;
        track.utauGlobalFlags="g"+juce::String(i);
        ClipData clip;clip.id="c"+juce::String(i);clip.startSeconds=i*.1;clip.durationSeconds=2;
        NoteData note;note.id="n"+juce::String(i);note.startSeconds=.2;note.durationSeconds=.5;note.midiNote=60+i;
        clip.notes={note};track.clips={clip};original.tracks.push_back(track);
    }
    project.resetDocument(original);project.dispatchPendingMessages();stopTimer();
    const auto fingerprint=[](ProjectData data)
    {
        std::sort(data.tracks.begin(),data.tracks.end(),[](const auto& a,const auto& b){return a.id<b.id;});
        ProjectModel model;model.resetDocument(std::move(data));return model.contentFingerprint();
    };
    const auto baseline=fingerprint(project.snapshot());
    const auto order=[&]{juce::String result;for(const auto& t:project.snapshot().tracks)result+=t.id+",";return result;};
    const auto flush=[&]{project.dispatchPendingMessages();stopTimer();};
    const auto event=[&](juce::Point<float> point,juce::Point<float> down,int buttons=juce::ModifierKeys::leftButtonModifier)
    {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(buttons),
            0,0,0,0,0,&trackList,&trackList,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,point!=down);
    };
    const auto drag=[&](juce::Point<float> from,juce::Point<float> to)
    {trackList.mouseDown(event(from,from));trackList.mouseDrag(event(to,from));trackList.mouseUp(event(to,from,0));flush();};
    const auto row=timeline.getRowHeight();const auto top=timeline.getRulerHeight();
    const auto title=[&](int i){return juce::Point<float>(65.0f,static_cast<float>(top+i*row+15));};
    const auto gap=[&](int i){return juce::Point<float>(65.0f,static_cast<float>(top+i*row));};
    focusClip("c5");
    const auto revision=project.revisionNumber();
    trackList.mouseDown(event(title(0),title(0)));
    trackList.mouseDrag(event(gap(3),title(0)));
    check("drag_preview_does_not_edit",project.revisionNumber()==revision&&order().startsWith("t0,t1,t2"));
    {juce::PNGImageFormat png;auto out=folder.getChildFile("drag-preview.png").createOutputStream();png.writeImageToStream(createComponentSnapshot(getLocalBounds()),*out);}
    trackList.mouseUp(event(gap(3),title(0),0));flush();
    check("drag_down_reorders",order()=="t1,t2,t0,t3,t4,t5,t6,t7,");
    check("one_undo_transaction",project.revisionNumber()==revision+1);
    check("dragged_track_stays_selected",selectedTrackId=="t0"&&selectedClipId.isEmpty());
    check("timeline_order_matches",timeline.trackIdForPixel(top+row/2)=="t1"&&timeline.trackIdForPixel(top+2*row+row/2)=="t0");
    check("all_track_content_preserved",fingerprint(project.snapshot())==baseline);
    project.undo();flush();check("undo_restores_order",order()=="t0,t1,t2,t3,t4,t5,t6,t7,");
    project.redo();flush();check("redo_restores_order",order()=="t1,t2,t0,t3,t4,t5,t6,t7,");
    drag(title(2),gap(0));check("drag_up",order()=="t0,t1,t2,t3,t4,t5,t6,t7,");
    drag(title(0),gap(8));check("drag_to_bottom",order()=="t1,t2,t3,t4,t5,t6,t7,t0,");
    drag(title(7),gap(0));check("drag_to_top",order()=="t0,t1,t2,t3,t4,t5,t6,t7,");
    const auto noEdit=project.revisionNumber();
    drag(title(0),title(0).translated(1,2));
    drag(title(0),gap(1));
    check("click_and_same_position_no_undo",project.revisionNumber()==noEdit);
    trackList.mouseDown(event(title(0),title(0)));trackList.mouseDrag(event(gap(3),title(0)));
    check("escape_handled",trackList.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey)));
    trackList.mouseUp(event(gap(3),title(0),0));flush();
    check("escape_cancels",project.revisionNumber()==noEdit);
    drag(title(0),{-20.0f,gap(3).y});check("outside_panel_cancels",project.revisionNumber()==noEdit);
    check("invalid_model_target_rejected",!project.moveTrackBefore("missing","t0")&&!project.moveTrackBefore("t0","missing"));
    // Buttons and sliders still own their original gestures.
    const auto toggle=juce::Point<float>(45,static_cast<float>(top+42));
    drag(toggle,gap(4));check("mute_button_never_reorders",order()=="t0,t1,t2,t3,t4,t5,t6,t7,"&&project.snapshot().tracks[0].muted);
    const auto slider=juce::Point<float>(180,static_cast<float>(top+42));
    drag(slider,{210,static_cast<float>(top+4*row)});
    check("volume_drag_never_reorders",order()=="t0,t1,t2,t3,t4,t5,t6,t7,");
    // Holding at an edge scrolls without more mouse movement.
    trackViewport.setViewPosition(0,0);
    const auto edge=juce::Point<float>(65,static_cast<float>(trackViewport.getHeight()-2));
    trackList.mouseDown(event(title(0),title(0)));trackList.mouseDrag(event(edge,title(0)));
    for(int i=0;i<8;++i)trackList.diagnosticReorderScrollTick();
    check("edge_auto_scroll",trackViewport.getViewPositionY()>0);
    trackList.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));
    trackViewport.setViewPosition(0,0);
    // Non-default row zoom uses the same insertion rule.
    trackList.setRowHeight(140);
    drag({65,39},{65,24+3*140});
    check("zoomed_rows_reorder",order().startsWith("t1,t2,t0,"));
    juce::String error;const auto saved=folder.getChildFile("reordered.hjpx");
    check("save_order",project.save(saved,error));
    ProjectModel reopened;check("reopen_order",reopened.load(saved,error)&&reopened.snapshot().tracks[2].id=="t0");
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");
    return ok;
}
}
