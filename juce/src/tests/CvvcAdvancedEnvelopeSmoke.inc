// Included in diagnosticChineseCvvc: exercise the actual selection and toolbar.
{
    stopTimer();focusClip(clip.id);pianoRoll.setReadsVoicebankInBackground(false);
    auto state=project.snapshot();auto req=request;req.notes=AudioEngine::utauRequestNotesForClip(state,clip.id);
    std::vector<std::vector<UtauPhonemeSpan>> layouts(req.notes.size());
    req.notePhonemes=[&](size_t i,const auto& p){layouts[i]=p;};UtauRenderer::render(req);
    auto metadata=std::make_shared<std::vector<UtauNoteWaveform>>();
    for(size_t i=0;i<clip.notes.size();++i){const auto& n=state.tracks[0].clips[0].notes[i];UtauNoteWaveform w;
        w.noteId=n.id;w.startSeconds=n.startSeconds;w.phonemes=layouts[i];
        w.audioHash=AudioEngine::utauNoteAudioHash(n);w.renderHash=AudioEngine::utauNoteRenderHash(n);
        w.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(n);w.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(state.tracks[0]);
        for(const auto& p:w.phonemes)w.leadInSeconds=std::max(w.leadInSeconds,p.leadInSeconds-p.startSeconds);metadata->push_back(w);}
    pianoRoll.setUtauNoteWaveforms(metadata);pianoRoll.setPixelsPerSecond(1000);setEnvelopeLane(EnvelopeLane::amplitude);
    project.dispatchPendingMessages();stopTimer();pianoRoll.diagnosticRefresh();
    const auto clickTab=[&](int part){
        pianoRoll.setSelectedNoteIds({});pianoRoll.diagnosticRefresh();const auto at=pianoRoll.diagnosticCvvcPhonemeTab("b",part).getCentre();
        const juce::MouseEvent event(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&pianoRoll,&pianoRoll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,false);
        pianoRoll.mouseDown(event);pianoRoll.mouseUp(event);
    };
    for(int part:{1,2,3})
    {
        clickTab(part);const auto before=project.contentFingerprint();const auto beforeNote=project.snapshot().tracks[0].clips[0].notes[1];
        advancedEnvelopeButton.onClick();
        auto* window=dynamic_cast<juce::DialogWindow*>(juce::ModalComponentManager::getInstance()->getModalComponent(0));
        auto* panel=window?dynamic_cast<AdvancedEnvelopePanel*>(window->getContentComponent()):nullptr;
        check("phoneme_tab_advanced_button_opens_scoped_panel",panel&&panel->diagnosticPhonemeMode()&&pianoRoll.selectedCvvcEnvelope()->part==part);
        if(!panel)continue;
        panel->diagnosticApply();check("opening_and_unchanged_apply_do_not_rewrite_envelope",project.contentFingerprint()==before);
        TailFadeSettings shape;shape.startGain=.8;shape.endGain=.2;shape.customCurve=true;
        shape.control1Time=.15;shape.control1Progress=.05;shape.control2Time=.7;shape.control2Progress=.95;shape.allowRise=true;
        panel->setValues(2,shape);check("advanced_phoneme_edit_is_draft_until_apply",project.contentFingerprint()==before);
        panel->diagnosticApply();const auto after=project.snapshot().tracks[0].clips[0].notes[1];
        bool only=after.amplitudeEnvelope.size()==beforeNote.amplitudeEnvelope.size()&&std::equal(after.amplitudeEnvelope.begin(),after.amplitudeEnvelope.end(),beforeNote.amplitudeEnvelope.begin(),[](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds&&a.gainDb==b.gainDb&&a.linearToNext==b.linearToNext;})&&after.utauTailFade==beforeNote.utauTailFade;
        for(int other:{1,2,3})if(other!=part)only&=after.cvvcEnvelopes.contains(other)==beforeNote.cvvcEnvelopes.contains(other);
        check("advanced_apply_changes_only_selected_phoneme",only&&after.cvvcEnvelopes.at(part).at("").size()==2
            &&after.cvvcEnvelopes.at(part).at("")[0].bezier.has_value());
        const auto curve=after.cvvcEnvelopes.at(part).at("");bool exact=true;
        for(int i=0;i<=1000;++i)exact&=std::abs(envelopeGainFromDb(cvvcEnvelopeValue(curve,i/1000.,true))-tailFadeGain(2,i/1000.,0,1,shape))<1.e-5;
        check("sparse_phoneme_bezier_matches_advanced_preview",exact);
        check("phoneme_bezier_invalidates_mix_only",AudioEngine::utauNoteRenderHash(after)!=AudioEngine::utauNoteRenderHash(beforeNote)
            &&AudioEngine::utauNoteAudioHash(after)==AudioEngine::utauNoteAudioHash(beforeNote));
        ProjectModel loaded;const auto path=folder.getChildFile("advanced-part-"+juce::String(part)+".hjpx");
        check("phoneme_advanced_shape_save_reopen",project.save(path,error)&&loaded.load(path,error)
            &&loaded.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes==after.cvvcEnvelopes);
        auto snapshot=panel->createComponentSnapshot(panel->getLocalBounds());juce::PNGImageFormat png;
        auto image=folder.getChildFile("advanced-part-"+juce::String(part)+".png").createOutputStream();if(image)png.writeImageToStream(snapshot,*image);
        window->exitModalState(0);
        check("phoneme_advanced_single_undo",project.undo()&&project.contentFingerprint()==before);
        project.dispatchPendingMessages();stopTimer();pianoRoll.diagnosticRefresh();
    }
    // Renderer transport must preserve the controls, not flatten or discard them.
    auto direct=planChineseCvvc(req).request;direct.notes={direct.notes.back()};direct.notePhonemes={};
    auto& n=direct.notes[0];n.cvvcStandardEnvelope=false;
    const auto first=n.amplitudeEnvelope.front().timeSeconds,last=n.amplitudeEnvelope.back().timeSeconds;
    const EnvelopeBezier bezier{.15,.05,.7,.95};n.amplitudeEnvelope={{first,0,true,bezier},{last,-20,true}};
    UtauMixEnvelope window;bool matched=true;int probes=0;
    direct.noteMixEnvelope=[&](size_t,const auto& e){window=e;};
    direct.notePiece=[&](size_t,const auto&,double,double,const auto& gain,const auto&){
        for(int i=0;i<window.samples;++i){const auto t=window.startSeconds+i/window.sampleRate;
            const auto u=std::clamp((t-first)/(last-first),0.,1.);
            const auto value=(1-.9*tailFadeBezierProgress(u,bezier))*window.gainAt(t);
            matched&=std::abs(gain(t)-value)<1.e-5;++probes;}};
    const auto actual=UtauRenderer::render(direct);
    check("phoneme_advanced_curve_reaches_traditional_mixer",actual.buffer.getNumSamples()>0&&probes>0&&matched);
    if(modelFolder.isDirectory())
    {
        direct.notePiece={};auto dry=direct;dry.notes[0].amplitudeEnvelope={{first,0,true},{last,0,true}};
        const auto a=renderNsfUtauPhrase(dry,modelFolder,{}),b=renderNsfUtauPhrase(direct,modelFolder,{});
        double difference=0;for(int i=0;i<std::min(a.buffer.getNumSamples(),b.buffer.getNumSamples());++i)
            difference+=std::abs(a.buffer.getSample(0,i)-b.buffer.getSample(0,i));
        check("phoneme_advanced_curve_changes_hifisampler_pcm",a.buffer.getNumSamples()>0&&b.buffer.getNumSamples()>0&&difference>.001);
    }
}
