#pragma once
namespace hachi
{
inline bool MainComponent::diagnosticNativeLoudness(const juce::File& folder)
{
    folder.createDirectory(); stopTimer(); setSize(1600,900);
    bool ok=true; int checks=0;
    const auto check=[&](const char* name,bool pass){ok &= pass;++checks;std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-5;};
    const auto file=folder.getChildFile("source.wav");juce::WavAudioFormat wav;auto stream=file.createOutputStream();
    if(!stream)return false;stream->setPosition(0);stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),48000,1,24,{},0));if(!writer)return false;
    juce::AudioBuffer<float> pcm(1,96000);
    for(int i=0;i<96000;++i)pcm.setSample(0,i,float(.35*std::sin(i*2*juce::MathConstants<double>::pi*220/48000)));
    check("fixture_written",writer->writeFromAudioSampleBuffer(pcm,0,96000));writer.reset();const auto sourceHash=juce::SHA256(file).toHexString();
    ProjectData data;TrackData track;track.id="native";track.trackType=TrackType::hachi;track.pitchAlgorithm=PitchAlgorithm::world;track.normalizeVolume=false;
    ClipData clip;clip.id="audio";clip.sourceFile=file;clip.startSeconds=1;clip.durationSeconds=clip.sourceDurationSeconds=2;
    for(int i=0;i<2;++i)
    {
        NoteData n;n.id="note-"+juce::String(i);n.startSeconds=i;n.durationSeconds=1;n.midiNote=n.sourceMidiCenter=57+i*3;
        n.sourcePitchMeasured=true;n.utauAutoPitchTransition=false;
        for(int j=0;j<=200;++j)n.contour.push_back({j*.005,0,0,true});
        bindNativeNoteSource(n,clip);clip.notes.push_back(n);
    }
    track.clips={clip};data.tracks={track};
    const auto refresh=[&]{project.dispatchPendingMessages();stopTimer();pianoRoll.diagnosticRefresh();};
    const auto install=[&](const ProjectData& d){project.resetDocument(d);refresh();diagnosticSelectTrack("native");refreshProjectControls();resized();
        pianoRoll.setFocusedTrack("native");pianoRoll.setFocusedClip("audio");pianoRoll.setPixelsPerSecond(300);pianoRoll.setRowHeight(24);
        pianoRoll.setShowNativeWaveforms(true);pianoRoll.setShowWaveforms(false);refresh();};
    const auto current=[&]{return project.snapshot().tracks[0].clips[0];};
    const auto mouse=[&](juce::Point<float> p,juce::Point<float> down){return juce::MouseEvent(
        juce::Desktop::getInstance().getMainMouseSource(),p,juce::ModifierKeys::leftButtonModifier,0,0,0,0,0,
        &pianoRoll,&pianoRoll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,p!=down);};
    const auto capture=[&](const char* name){
        const auto bounds=pianoRoll.diagnosticHitBounds(0);
        juce::Image image(juce::Image::RGB,950,220,true,juce::SoftwareImageType{});
        {juce::Graphics g(image);g.fillAll(Palette::background);g.setOrigin(0,110-int(bounds.getCentreY()));pianoRoll.paint(g);}
        if(auto out=folder.getChildFile(name).createOutputStream()){out->setPosition(0);out->truncate();juce::PNGImageFormat().writeImageToStream(image,*out);}
        // Warm waveform pixels near its centre; exclude note edges and the gain label.
        int height=0;const int x=int(bounds.getCentreX());
        for(int y=75;y<145;++y){const auto c=image.getPixelAt(x,y);if(c.getRed()>c.getBlue()+35&&c.getRed()>100)++height;}
        return height;
    };
    install(data);
    check("loudness_replaces_stretch_next_to_trim",nativeLoudnessButton.isVisible()&&nativeLoudnessButton.isEnabled()
        &&findChildWithID("tool.nativeLoudness")==&nativeLoudnessButton&&findChildWithID("tool.clipStretch")==nullptr
        &&nativeLoudnessButton.getX()>=trimButton.getRight());
    nativeLoudnessButton.onClick();check("toolbar_activates_only_loudness",nativeLoudnessButton.getToggleState()&&pianoRoll.isNativeLoudnessMode()&&!timeline.clipStretchMode());
    for(int i=0;i<50;++i){juce::Thread::sleep(10);(void)pianoRoll.createComponentSnapshot(pianoRoll.getLocalBounds());}
    pianoRoll.setSelectedNoteIds({"note-0"});
    auto down=pianoRoll.diagnosticHitBounds(0).getCentre();auto up=down.translated(0,-40);
    pianoRoll.mouseDown(mouse(down,down));const auto small=capture("before.png");const auto rev=project.revisionNumber();
    pianoRoll.mouseDrag(mouse(up,down));const auto large=capture("preview-up.png");
    check("live_preview_doubles_original_waveform_height",small>4&&large>small*1.6&&large<small*2.5);
    check("preview_does_not_mutate_model",near(current().notes[0].gain,1)&&project.revisionNumber()==rev);
    pianoRoll.mouseUp(mouse(up,down));refresh();const auto gain=juce::Decibels::decibelsToGain(6.f);
    check("up_40_pixels_is_plus_6_db",near(current().notes[0].gain,gain));
    check("unselected_note_unchanged",near(current().notes[1].gain,1));
    const auto louder=project.snapshot();
    check("timing_pitch_source_clock_unchanged",near(current().notes[0].startSeconds,0)&&near(current().notes[0].durationSeconds,1)
        &&near(current().notes[0].midiNote,57)&&near(current().durationSeconds,2)&&current().notes[0].contour.size()==201);
    check("one_undo_per_gesture",project.revisionNumber()==rev+1&&project.undo()&&near(current().notes[0].gain,1));refresh();
    check("redo_restores_loudness",project.redo()&&near(current().notes[0].gain,gain));refresh();
    juce::String error;ProjectModel reopened;
    check("save_reopen_keeps_per_note_gain",project.save(folder.getChildFile("loudness.hjpx"),error)
        &&reopened.load(folder.getChildFile("loudness.hjpx"),error)&&near(reopened.snapshot().tracks[0].clips[0].notes[0].gain,gain));
    const auto drag=[&](float dy,bool cancel=false){const auto start=pianoRoll.diagnosticHitBounds(0).getCentre();const auto end=start.translated(0,dy);
        pianoRoll.mouseDown(mouse(start,start));pianoRoll.mouseDrag(mouse(end,start));
        if(cancel)(void)pianoRoll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));
        pianoRoll.mouseUp(mouse(end,start));refresh();};
    drag(40);check("drag_down_returns_to_original",near(current().notes[0].gain,1));
    const auto fp=project.contentFingerprint();drag(-40,true);check("escape_cancels_preview",project.contentFingerprint()==fp);
    auto revision=project.revisionNumber();drag(0);check("click_without_motion_has_no_undo",project.revisionNumber()==revision);
    drag(-1000);check("upper_gain_limit_is_finite",near(current().notes[0].gain,4));
    drag(1000);check("down_to_silence",near(current().notes[0].gain,0));drag(-40);check("silent_note_can_be_raised_again",current().notes[0].gain>0);
    down=pianoRoll.diagnosticHitBounds(0).getCentre();up=down.translated(0,-40);revision=project.revisionNumber();
    pianoRoll.mouseDown(mouse(down,down));pianoRoll.mouseDrag(mouse(up,down));trimButton.onClick();pianoRoll.mouseUp(mouse(up,down));refresh();
    check("switching_tool_cancels_pending_gain",!pianoRoll.isNativeLoudnessMode()&&project.revisionNumber()==revision);
    nativeLoudnessButton.onClick();nativeLoudnessButton.onClick();check("second_click_exits",!pianoRoll.isNativeLoudnessMode());
    auto batch=data;
    auto& batchNotes=batch.tracks[0].clips[0].notes;
    batchNotes[0].gain=.5f;batchNotes[1].durationSeconds=.5;
    auto third=batchNotes[1];third.id="note-2";third.startSeconds=1.5;batchNotes.push_back(third);
    install(batch);nativeLoudnessButton.onClick();
    const auto firstBox=pianoRoll.diagnosticHitBounds(0),secondBox=pianoRoll.diagnosticHitBounds(1);
    const juce::Point<float> boxStart(firstBox.getX()-12,std::min(firstBox.getY(),secondBox.getY())-18);
    const juce::Point<float> boxEnd(secondBox.getRight()-4,std::max(firstBox.getBottom(),secondBox.getBottom())+12);
    pianoRoll.mouseDown(mouse(boxStart,boxStart));pianoRoll.mouseDrag(mouse(boxEnd,boxStart));pianoRoll.mouseUp(mouse(boxEnd,boxStart));
    const auto selected=pianoRoll.selectedNoteIds();
    check("loudness_tool_marquee_selects_two_notes",selected.size()==2
        &&std::find(selected.begin(),selected.end(),"note-0")!=selected.end()&&std::find(selected.begin(),selected.end(),"note-1")!=selected.end());
    down=firstBox.getCentre();up=down.translated(0,-40);revision=project.revisionNumber();
    pianoRoll.mouseDown(mouse(down,down));pianoRoll.mouseDrag(mouse(up,down));
    (void)capture("batch-preview.png");
    check("dragging_selected_note_keeps_multi_selection",pianoRoll.selectedNoteIds().size()==2);
    check("batch_preview_defers_model_mutation",project.revisionNumber()==revision&&near(current().notes[0].gain,.5));
    pianoRoll.mouseUp(mouse(up,down));refresh();const auto batchLouder=project.snapshot();
    check("batch_applies_same_db_preserving_balance",near(current().notes[0].gain,.5*gain)&&near(current().notes[1].gain,gain)
        &&near(current().notes[1].gain/current().notes[0].gain,2));
    check("batch_does_not_touch_unselected_note",near(current().notes[2].gain,1));
    check("batch_one_undo_restores_both",project.revisionNumber()==revision+1&&project.undo()
        &&near(current().notes[0].gain,.5)&&near(current().notes[1].gain,1));refresh();
    check("batch_redo_restores_both",project.redo()&&near(current().notes[0].gain,.5*gain)&&near(current().notes[1].gain,gain));refresh();
    check("batch_save_reopen",project.save(folder.getChildFile("batch.hjpx"),error)&&reopened.load(folder.getChildFile("batch.hjpx"),error)
        &&near(reopened.snapshot().tracks[0].clips[0].notes[0].gain,.5*gain)&&near(reopened.snapshot().tracks[0].clips[0].notes[1].gain,gain));
    const auto batchFingerprint=project.contentFingerprint();drag(-40,true);
    check("escape_cancels_entire_batch",project.contentFingerprint()==batchFingerprint);
    drag(40);check("batch_down_reverses_up",near(current().notes[0].gain,.5)&&near(current().notes[1].gain,1));
    const auto thirdDown=pianoRoll.diagnosticHitBounds(2).getCentre(),thirdUp=thirdDown.translated(0,-40);
    pianoRoll.mouseDown(mouse(thirdDown,thirdDown));pianoRoll.mouseDrag(mouse(thirdUp,thirdDown));pianoRoll.mouseUp(mouse(thirdUp,thirdDown));refresh();
    check("clicking_unselected_note_returns_to_single",pianoRoll.selectedNoteIds().size()==1&&near(current().notes[0].gain,.5)
        &&near(current().notes[1].gain,1)&&near(current().notes[2].gain,gain));
    const auto stable=project.contentFingerprint();
    check("stale_batch_rejected_atomically",!project.setNoteGains({{"note-0",1},{"note-1",2}},project.revisionNumber()-1)&&project.contentFingerprint()==stable);
    check("missing_target_rejects_entire_batch",!project.setNoteGains({{"note-0",1},{"missing",2}})&&project.contentFingerprint()==stable);
    revision=project.revisionNumber();project.setNoteGains({{"note-0",.5},{"note-1",1}});
    check("unchanged_batch_has_no_undo",project.revisionNumber()==revision);
    for(const auto type:{TrackType::utau,TrackType::diffSinger})
    {auto other=data;other.tracks[0].trackType=type;other.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;install(other);
     check("voicebank_modes_hide_and_disable_loudness",!nativeLoudnessButton.isVisible()&&!nativeLoudnessButton.isEnabled()&&!pianoRoll.isNativeLoudnessMode());}
    const auto render=[&](const ProjectData& d,const char* name){
        AudioEngine engine(false);engine.prepareToPlay(256,48000);engine.syncProject(d);
        for(int i=0;i<2000&&engine.renderProgress();++i)juce::Thread::sleep(10);
        WavExportOptions options;options.sampleRate=48000;options.channels=2;options.bitDepth=32;const auto output=folder.getChildFile(name);
        check(name,engine.hasCurrentRenderedAudio()&&engine.exportWav(output,error,"native",1,3,options));
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(output));
        std::array<float,2> rms{};if(reader)for(int i=0;i<2;++i){juce::AudioBuffer<float> buffer(1,24000);reader->read(&buffer,0,24000,i*48000+12000,true,false);rms[i]=buffer.getRMSLevel(0,0,24000);}return rms;
    };
    const auto before=render(data,"original.wav"),after=render(louder,"louder.wav");
    check("actual_audio_gain_matches_display",before[0]>.001f&&std::abs(after[0]/before[0]-gain)<.04f);
    check("adjacent_audio_unchanged",before[1]>.001f&&std::abs(after[1]/before[1]-1)<.02f);
    render(batch,"batch-original.wav");render(batchLouder,"batch-louder.wav");
    const auto rmsAt=[&](const char* name,double seconds){
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(folder.getChildFile(name)));
        if(!reader)return 0.f;juce::AudioBuffer<float> b(1,4800);reader->read(&b,0,4800,juce::int64(seconds*48000),true,false);return b.getRMSLevel(0,0,4800);};
    for(const auto t:{.3,1.2,1.7})
    {const auto a=rmsAt("batch-original.wav",t),b=rmsAt("batch-louder.wav",t);
     check(t<1.5?"batch_selected_audio_matches_gain":"batch_unselected_audio_unchanged",a>.001f&&std::abs(b/a-(t<1.5?gain:1.f))<.04f);}
    check("source_file_untouched",juce::SHA256(file).toHexString()==sourceHash);
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");
    return ok;
}
}
