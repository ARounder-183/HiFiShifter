#pragma once
namespace hachi
{
inline bool runNativeSparsePitchSmoke(const juce::File& folder)
{
    folder.createDirectory(); bool ok=true; int checks=0;
    const auto check=[&](const char* name,bool pass){ok &= pass;++checks;std::cout<<name<<'='<<pass<<std::endl;};
    ProjectData data;TrackData track;track.id="native";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;
    std::vector<juce::String> ids;
    for(int i=0;i<6;++i)
    {
        ClipData c;c.id="clip"+juce::String(i);c.startSeconds=1+i*.5;c.durationSeconds=c.sourceDurationSeconds=.5;
        NoteData n;n.id="note"+juce::String(i);n.durationSeconds=.5;n.midiNote=n.sourceMidiCenter=i<3?60.f:64.f;n.sourcePitchMeasured=true;
        for(int f=0;f<=100;++f)n.contour.push_back({f*.005,5.f,5.f,true});
        c.notes={n};track.clips.push_back(c);ids.push_back(n.id);
    }
    data.tracks={track};ProjectModel model;I18n strings;PianoRollComponent roll(model,strings);
    roll.setSize(1600,1700);roll.setPixelsPerSecond(300);roll.setRowHeight(24);roll.setFocusedTrack("native");roll.setTool(PianoRollComponent::Tool::points);
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();roll.setFocusedTrack("native");};
    const auto install=[&](ProjectData p){model.resetDocument(std::move(p));refresh();};
    const auto offeredCount=[&]{size_t count=0;for(const auto& id:ids)count+=roll.diagnosticOfferedPitchAnchors(id).size();return count;};
    install(data);const auto original=model.contentFingerprint();const auto revision=model.revisionNumber();
    model.flattenNotePitch(ids,true);refresh();const auto flat=model.snapshot();
    size_t stored=0;for(const auto& c:flat.tracks[0].clips)stored+=c.notes[0].pitchControlPoints.size();
    check("six_notes_two_pitch_runs_store_four_points",stored==4&&offeredCount()==4);
    check("interior_notes_do_not_regenerate_points",roll.diagnosticOfferedPitchAnchors("note1").empty()&&roll.diagnosticOfferedPitchAnchors("note4").empty());
    const auto lines=sharedPitchLines(flat.tracks[0]);bool pitch=true,requests=true;
    for(const auto& c:flat.tracks[0].clips)
    {
        const auto& n=c.notes[0];const auto* member=lines.memberFor(n.id);pitch &= member!=nullptr;
        if(member)for(double t:{.01,.25,.49})pitch &= std::abs(member->line->midiAt(c.startSeconds+t)-n.midiNote)<.0001;
        const auto r=AudioEngine::diagnosticNativeRequest(c,flat.tracks[0]);
        for(size_t i=1;i+1<r.targetMidi.size();++i)requests &= std::abs(r.targetMidi[i]-n.midiNote)<.0001;
    }
    check("same_pitch_runs_remain_complete_straight_lines",pitch);check("interior_notes_synthesize_shared_line",requests);
    check("initialization_is_one_undo",model.revisionNumber()==revision+1&&model.undo()&&model.contentFingerprint()==original);
    check("redo_restores_sparse_runs",model.redo());refresh();
    juce::String error;ProjectModel reopened;
    check("save_reopen_preserves_sparse_run",model.save(folder.getChildFile("sparse.hjpx"),error)&&reopened.load(folder.getChildFile("sparse.hjpx"),error));
    install(reopened.snapshot());check("reopened_has_only_four_handles",offeredCount()==4);
    auto picture=roll.createComponentSnapshot({250,int(roll.diagnosticYForMidi(66))-25,1100,300});
    if(auto stream=folder.getChildFile("initialized.png").createOutputStream())juce::PNGImageFormat{}.writeImageToStream(picture,*stream);
    const auto mouse=[&](juce::Point<float> at,juce::Point<float> down){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,
        juce::ModifierKeys::leftButtonModifier,0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
    const auto drag=[&](const char* id,int index,double absolute,float midi){
        refresh();const auto anchors=roll.diagnosticOfferedPitchAnchors(id);if(index>=int(anchors.size()))return false;
        const auto d=model.snapshot();double start=0;for(const auto& c:d.tracks[0].clips)for(const auto& n:c.notes)if(n.id==id)start=c.startSeconds+n.startSeconds;
        const auto a=anchors[(size_t)index];const juce::Point<float> down(roll.diagnosticEdgeX(start+a.timeSeconds),roll.diagnosticYForMidi(a.targetMidi));
        const juce::Point<float> up(roll.diagnosticEdgeX(absolute),roll.diagnosticYForMidi(midi));
        roll.mouseDown(mouse(down,down));roll.mouseDrag(mouse(up,down));roll.mouseUp(mouse(up,down));refresh();
        const auto changed=model.snapshot();for(const auto& c:changed.tracks[0].clips)for(const auto& n:c.notes)if(n.id==id)
            for(const auto& p:n.pitchControlPoints)if(std::abs(c.startSeconds+n.startSeconds+p.timeSeconds-absolute)<.002&&std::abs(p.targetMidi-midi)<.02)return true;
        return false;
    };
    check("single_sparse_endpoint_can_cross_other_note_points",drag("note0",0,3.3,67));
    check("sparse_endpoint_can_move_back_again",drag("note0",0,.7,61));
    // A dense set in one note must also reorder, rather than pinning the middle point.
    install(data);model.setNotePitchCurve("note0",{{0,60},{.1,61},{.2,62},{.3,63}},true);refresh();
    check("interior_point_crosses_sibling_points",drag("note0",1,1.42,65));
    check("crossed_point_remains_draggable",drag("note0",3,1.05,59));
    auto partial=data;install(partial);model.flattenNotePitch({"note0","note2","note3","note4","note5"},true);refresh();
    check("unselected_note_keeps_its_source_curve",model.snapshot().tracks[0].clips[1].notes[0].pitchControlPoints.empty()
        &&!model.snapshot().tracks[0].clips[1].notes[0].nativeSparsePitchControls);
    check("unselected_gap_does_not_collapse_selected_runs",model.snapshot().tracks[0].clips[0].notes[0].pitchControlPoints.size()==2
        &&model.snapshot().tracks[0].clips[2].notes[0].pitchControlPoints.size()==2);
    auto gap=data;gap.tracks[0].clips[2].startSeconds+=.1;install(gap);model.flattenNotePitch(ids,true);refresh();
    check("real_gap_splits_equal_pitch_run",model.snapshot().tracks[0].clips[2].notes[0].pitchControlPoints.size()==2);
    install(flat);model.setNotePitchCurve("note1",{{.1,65},{.4,65}},false);refresh();
    const auto drawn=model.snapshot();const auto drawnLines=sharedPitchLines(drawn.tracks[0]);
    const auto* drawnMember=drawnLines.memberFor("note1");
    check("freehand_after_sparse_initialization_is_audible",!drawn.tracks[0].clips[1].notes[0].nativeSparsePitchControls
        &&drawnMember&&std::abs(drawnMember->line->midiAt(1.75)-65)<.001);
    install(flat);check("restore_source_clears_sparse_marker",model.restoreNativeSourcePitch(ids));refresh();
    const auto restored=model.snapshot();check("restored_source_has_no_sparse_members",std::all_of(restored.tracks[0].clips.begin(),restored.tracks[0].clips.end(),[](const auto& c){return !c.notes[0].nativeSparsePitchControls;}));
    // Right-click curve types follow the shared point order, not local index.
    auto menuData=flat;menuData.tracks[0].clips.resize(2);
    menuData.tracks[0].clips[0].notes[0].pitchControlPoints={{0,60}};
    menuData.tracks[0].clips[1].notes[0].pitchControlPoints={{.5,72}};
    install(menuData);
    const auto has=[&](const char* id,int choice){const auto items=roll.diagnosticEnabledAnchorMenuIds(id,0);return std::find(items.begin(),items.end(),choice)!=items.end();};
    bool menuOk=true;for(int choice=1;choice<=6;++choice)menuOk &= has("note1",choice)&&!has("note0",choice);
    check("shared_endpoint_menu_offers_all_six_shapes",menuOk);
    check("global_first_endpoint_rejects_shape_edit",!roll.setPitchAnchorShape("note0",0,PitchCurveShape::easeIn));
    bool shapesOk=true,displayOk=true,renderOk=true,undoOk=true;
    for(const auto shape:{PitchCurveShape::natural,PitchCurveShape::linear,PitchCurveShape::smooth,
        PitchCurveShape::easeIn,PitchCurveShape::easeOut,PitchCurveShape::customBezier})
    {
        const auto before=model.contentFingerprint();
        shapesOk &= roll.setPitchAnchorShape("note1",0,shape,std::array<float,4>{.25f,.1f,.75f,.9f});refresh();
        const auto changed=model.snapshot();const auto& endpoint=changed.tracks[0].clips[1].notes[0].pitchControlPoints[0];
        shapesOk &= endpoint.shape==shape && changed.tracks[0].clips[0].notes[0].pitchControlPoints[0].shape==PitchCurveShape::natural;
        auto end=endpoint;end.timeSeconds=2;const std::vector<PitchCurveEditPoint> expected={{1,60},end};
        for(const auto& c:changed.tracks[0].clips)
        {
            const auto request=AudioEngine::diagnosticNativeRequest(c,changed.tracks[0]);
            for(size_t f=1;f+1<request.targetMidi.size();++f)
            {
                const auto t=c.startSeconds+f*request.framePeriodMs/1000.;
                const auto value=evaluatePitchCurve(expected,t);
                const auto shown=roll.diagnosticPitchLineAt(c.notes[0].id,t);
                displayOk &= shown&&std::abs(*shown-value)<.0002;
                renderOk &= std::abs(request.targetMidi[f]-value)<.0002;
            }
        }
        undoOk &= model.undo()&&model.contentFingerprint()==before;refresh();
    }
    check("all_shapes_apply_to_cross_note_incoming_segment",shapesOk);
    check("all_shapes_display_matches_interpolation",displayOk);
    check("all_shapes_render_matches_interpolation",renderOk);
    check("shape_edits_are_undoable",undoOk);
    check("custom_bezier_apply",roll.setPitchAnchorShape("note1",0,PitchCurveShape::customBezier,std::array<float,4>{.2f,.3f,.7f,.8f}));refresh();
    check("custom_bezier_save_reopen",model.save(folder.getChildFile("shape.hjpx"),error)&&reopened.load(folder.getChildFile("shape.hjpx"),error));
    install(reopened.snapshot());const auto persisted=model.snapshot().tracks[0].clips[1].notes[0].pitchControlPoints[0];
    check("custom_bezier_values_persist",persisted.shape==PitchCurveShape::customBezier&&std::abs(persisted.bezierY1-.3f)<1.e-6&&has("note1",6));
    auto isolated=menuData;isolated.tracks[0].clips[1].startSeconds+=.1;install(isolated);
    check("real_gap_keeps_first_point_without_incoming",!has("note1",2));
    install(menuData);check("move_first_point_past_neighbor",drag("note0",0,2.1,65));
    check("menu_follows_global_order_after_crossing",has("note0",2)&&!has("note1",2));
    // Render real PCM: a middle note has no controls of its own, but must hear
    // changes to the two endpoints of its entire same-pitch run.
    auto audio=flat;audio.tracks[0].normalizeVolume=false;audio.tracks[0].compose=true;
    // Use the model-free backend with supplied F0; WORLD's independent voicing
    // detector can classify a synthetic sine fixture as unvoiced.
    audio.tracks[0].pitchAlgorithm=PitchAlgorithm::mld5;
    for(auto& c:audio.tracks[0].clips)
    {
        c.sourceFile=folder.getChildFile(c.id+".wav");juce::WavAudioFormat wav;auto stream=c.sourceFile.createOutputStream();
        if(!stream)return false;stream->setPosition(0);stream->truncate();
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),48000,1,24,{},0));if(!writer)return false;
        juce::AudioBuffer<float> pcm(1,24000);const auto hz=440*std::pow(2.,(c.notes[0].sourceMidiCenter-69)/12.);
        for(int i=0;i<24000;++i){const auto phase=i*2*juce::MathConstants<double>::pi*hz/48000;
            pcm.setSample(0,i,float(.25*std::sin(phase)+.07*std::sin(phase*2)));}
        if(!writer->writeFromAudioSampleBuffer(pcm,0,24000))return false;writer.reset();bindNativeNoteSource(c.notes[0],c);
    }
    const auto render=[&](const ProjectData& p,const char* name){
        AudioEngine engine(false);engine.prepareToPlay(256,48000);engine.syncProject(p);
        for(int i=0;i<3000&&engine.renderProgress();++i)juce::Thread::sleep(10);
        std::cout<<name<<" backend="<<engine.activeRenderBackends()<<" warning="<<engine.activeRenderWarnings()<<std::endl;
        WavExportOptions options;options.sampleRate=48000;options.channels=1;options.bitDepth=32;
        check(name,engine.hasCurrentRenderedAudio()&&engine.exportWav(folder.getChildFile(name),error,"native",1,4,options));
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(folder.getChildFile(name)));
        juce::AudioBuffer<float> pcm(1,144000);pcm.clear();if(reader)reader->read(&pcm,0,144000,0,true,false);return pcm;
    };
    const auto baseline=render(audio,"sparse-audio.wav");
    for(auto& p:audio.tracks[0].clips[0].notes[0].pitchControlPoints)p.targetMidi=72;
    for(auto& p:audio.tracks[0].clips[2].notes[0].pitchControlPoints)p.targetMidi=72;
    const auto shifted=render(audio,"moved-endpoints-audio.wav");bool audible=true;
    for(int i=0;i<6;++i)audible &= baseline.getRMSLevel(0,i*24000+6000,12000)>.001f&&shifted.getRMSLevel(0,i*24000+6000,12000)>.001f;
    check("all_six_notes_render_audible_pcm",audible);
    double delta=0;for(int i=30000;i<42000;++i)delta+=std::pow(baseline.getSample(0,i)-shifted.getSample(0,i),2);
    check("endpoint_edit_changes_middle_note_audio",std::sqrt(delta/12000)>.005);
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");return ok;
}
}
