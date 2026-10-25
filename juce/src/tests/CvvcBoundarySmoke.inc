// Included in diagnosticChineseCvvc. Exercise real mouse routing, not a mock setter.
{
    auto initial=project.snapshot();
    for(auto& n:initial.tracks[0].clips[0].notes){n.cvvcBoundaries.clear();n.cvvcEnvelopes.clear();n.amplitudeEnvelope.clear();}
    // The existing right-click action resets authored CVVC dividers, even
    // while derived layouts are absent. Other timing/envelope settings survive.
    {
        auto state=initial;auto& notes=state.tracks[0].clips[0].notes;
        notes[0].cvvcBoundaries={{3,.7}};notes[1].cvvcBoundaries={{2,-.05},{3,.65}};
        notes[1].utauConsonantVelocity=130;
        notes[1].cvvcEnvelopes[3][""]={{0,-12,true},{1,-6,true}};
        ProjectModel model;model.resetDocument(state);I18n labels;PianoRollComponent roll(model,labels);
        roll.setSize(1200,750);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
        check("cvvc_reset_menu_name_without_layout",roll.diagnosticNoteMenuLabel("b",12)==juce::String::fromUTF8("强制重置"));
        const auto fingerprint=model.contentFingerprint();const auto beforeRevision=model.revisionNumber();
        roll.setSelectedNoteIds({"b"});roll.applyNoteMenuChoice("b",12,0,{});
        const auto resetState=model.snapshot();const auto& resetNote=resetState.tracks[0].clips[0].notes[1];
        check("cvvc_reset_clears_all_selected_note_dividers",resetNote.cvvcBoundaries.empty()
            &&resetNote.cvvcPhonemeCache.isEmpty()&&!resetState.tracks[0].clips[0].notes[0].cvvcBoundaries.empty());
        check("cvvc_reset_preserves_envelopes_velocity_and_score",resetNote.cvvcEnvelopes==notes[1].cvvcEnvelopes
            &&resetNote.utauConsonantVelocity==130&&near(resetNote.startSeconds,notes[1].startSeconds)
            &&near(resetNote.durationSeconds,notes[1].durationSeconds));
        auto automatic=state;automatic.tracks[0].clips[0].notes[1].cvvcBoundaries.clear();
        auto expected=request;expected.notes=AudioEngine::utauRequestNotesForClip(automatic,clip.id);
        auto actual=request;actual.notes=AudioEngine::utauRequestNotesForClip(resetState,clip.id);
        const auto e=planChineseCvvc(expected),a=planChineseCvvc(actual);bool equal=e.spans[1].size()==a.spans[1].size();
        for(size_t i=0;equal&&i<e.spans[1].size();++i)equal=near(e.spans[1][i].startSeconds,a.spans[1][i].startSeconds)
            &&near(e.spans[1][i].endSeconds,a.spans[1][i].endSeconds);
        check("cvvc_reset_restores_automatic_planner_positions",equal);
        check("cvvc_reset_invalidates_layout_and_audio",AudioEngine::utauNotePhonemeHash(resetNote)!=AudioEngine::utauNotePhonemeHash(notes[1])
            &&AudioEngine::utauNoteAudioHash(resetNote)!=AudioEngine::utauNoteAudioHash(notes[1]));
        ProjectModel loaded;const auto path=folder.getChildFile("boundary-reset.hjpx");
        check("cvvc_reset_survives_save_reload",model.save(path,error)&&loaded.load(path,error)
            &&loaded.snapshot().tracks[0].clips[0].notes[1].cvvcBoundaries.empty());
        check("cvvc_reset_single_undo_restores_all_dividers",model.undo()&&model.contentFingerprint()==fingerprint);
        model.dispatchPendingMessages();roll.diagnosticRefresh();roll.setSelectedNoteIds({"a","b"});
        roll.applyNoteMenuChoice("b",12,0,{});
        check("cvvc_reset_multi_selection",model.snapshot().tracks[0].clips[0].notes[0].cvvcBoundaries.empty()
            &&model.snapshot().tracks[0].clips[0].notes[1].cvvcBoundaries.empty());
        const auto unchangedRevision=model.revisionNumber();roll.applyNoteMenuChoice("b",12,0,{});
        check("cvvc_reset_repeated_is_noop",model.revisionNumber()==unchangedRevision);
        check("cvvc_reset_batch_one_undo",model.undo()&&model.contentFingerprint()==fingerprint);
        check("cvvc_reset_rejects_stale_revision",!model.resetCvvcBoundaries({"b"},beforeRevision));
        // A dormant CVVC setting must not change the old ordinary/four-region action.
        for(const auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
        {
            auto ordinary=state;ordinary.tracks[0].utauMode=mode;ordinary.tracks[0].chineseCvvc=mode!=UtauMode::classic;
            ordinary.tracks[0].clips[0].notes[1].utauPreutteranceOverrideEnabled=true;
            ordinary.tracks[0].clips[0].notes[1].utauPreutteranceSeconds=.2;
            model.resetDocument(ordinary);roll.diagnosticRefresh();roll.setSelectedNoteIds({"b"});
            check("ordinary_utau_keeps_consonant_reset_name",roll.diagnosticNoteMenuLabel("b",12)==juce::String::fromUTF8("辅音强制重置"));
            roll.applyNoteMenuChoice("b",12,0,{});const auto result=model.snapshot().tracks[0].clips[0].notes[1];
            check("ordinary_utau_keeps_consonant_reset_behavior",!result.utauPreutteranceOverrideEnabled
                &&result.cvvcBoundaries==notes[1].cvvcBoundaries);
        }
    }
    for(const auto part:{2,3}) for(const auto delta:{-.035,.025,-10.,10.})
    {
        ProjectModel model;model.resetDocument(initial);I18n labels;PianoRollComponent roll(model,labels);
        juce::Viewport viewport;viewport.setSize(1500,650);viewport.setViewedComponent(&roll,false);
        roll.setSize(1600,900);roll.setPixelsPerSecond(1000);
        roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
        roll.setTool(part==3?PianoRollComponent::Tool::amplitude:PianoRollComponent::Tool::note);
        const auto requestFor=[&]{auto r=request;r.notes=AudioEngine::utauRequestNotesForClip(model.snapshot(),clip.id);
            r.notePhonemes={};r.notePiece={};return r;};
        const auto publish=[&](const auto& layout){
            const auto state=model.snapshot();const auto& t=state.tracks[0];
            std::vector<std::pair<juce::String,juce::String>> caches;
            for(size_t i=0;i<layout.spans.size();++i){const auto& n=t.clips[0].notes[i];
                caches.emplace_back(n.id,cvvcPhonemeCacheText({AudioEngine::utauNotePhonemeHash(n),
                    AudioEngine::utauPhonemeMixContextHash(t),layout.spans[i]}));}
            model.rememberCvvcPhonemes(caches,model.revisionNumber());roll.diagnosticRefresh();};
        const auto baseline=planChineseCvvc(requestFor());publish(baseline);
        viewport.setViewPosition(0,static_cast<int>(roll.diagnosticYForMidi(60)-180));
        const auto old=baseline.spans[1];const auto index=part==2?1u:2u;
        const auto boundary=roll.diagnosticCvvcBoundary("b",part);
        check("cvvc_boundary_has_drag_target",!boundary.isEmpty());if(boundary.isEmpty())continue;
        const auto at=boundary.getCentre();
        const auto event=[&](juce::Point<float> point){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),point,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,point!=at);};
        const auto fingerprint=model.contentFingerprint();const auto revision=model.revisionNumber();
        roll.mouseDown(event(at));roll.mouseUp(event(at));
        check("cvvc_boundary_click_does_not_edit",model.contentFingerprint()==fingerprint&&model.revisionNumber()==revision);
        roll.mouseDown(event(at));const auto dragged=at.translated(static_cast<float>(delta*1000),0);
        roll.mouseDrag(event(dragged));const auto preview=roll.diagnosticPhonemeSpans("b");
        const auto expected=clampCvvcBoundary(old[index].startSeconds+delta,old[index-1].startSeconds,old[index].endSeconds);
        check("cvvc_boundary_drag_is_draft",model.contentFingerprint()==fingerprint&&near(preview[index].startSeconds,expected));
        roll.mouseUp(event(dragged));model.dispatchPendingMessages();roll.diagnosticRefresh();
        const auto next=planChineseCvvc(requestFor());const auto now=next.spans[1];
        const auto after=model.snapshot().tracks[0].clips[0].notes[1];const auto before=initial.tracks[0].clips[0].notes[1];
        check("cvvc_boundary_release_matches_preview",near(now[index].startSeconds,expected)
            &&near(roll.diagnosticPhonemeSpans("b")[index].startSeconds,expected));
        check("cvvc_boundary_pair_exchanges_duration",near(now[index-1].startSeconds,old[index-1].startSeconds)
            &&near(now[index].endSeconds,old[index].endSeconds)&&near(now[index-1].endSeconds,now[index].startSeconds)
            &&now[index-1].endSeconds>now[index-1].startSeconds&&now[index].endSeconds>now[index].startSeconds);
        check("cvvc_boundary_preserves_score_and_velocity",near(after.startSeconds,before.startSeconds)
            &&near(after.durationSeconds,before.durationSeconds)&&after.utauConsonantVelocity==before.utauConsonantVelocity
            &&model.snapshot().tracks[0].clips[0].notes.size()==2&&after.cvvcEnvelopes==before.cvvcEnvelopes);
        check("cvvc_boundary_invalidates_all_relevant_caches",AudioEngine::utauNotePhonemeHash(after)!=AudioEngine::utauNotePhonemeHash(before)
            &&AudioEngine::utauNoteAudioHash(after)!=AudioEngine::utauNoteAudioHash(before)
            &&AudioEngine::utauPhonemeMixContextHash(model.snapshot().tracks[0])!=AudioEngine::utauPhonemeMixContextHash(initial.tracks[0]));
        publish(next);check("cvvc_boundary_background_layout_matches_preview",near(roll.diagnosticPhonemeSpans("b")[index].startSeconds,expected));
        const auto path=folder.getChildFile("boundary-roundtrip.hjpx");ProjectModel loaded;
        check("cvvc_boundary_save_reopen",model.save(path,error)&&loaded.load(path,error)
            &&loaded.snapshot().tracks[0].clips[0].notes[1].cvvcBoundaries==after.cvvcBoundaries);
        check("cvvc_boundary_rejects_stale_commit",!model.setCvvcBoundary("b",part,.5,revision));
        check("cvvc_boundary_one_undo_and_redo",model.undo()&&model.contentFingerprint()==fingerprint&&model.redo()
            &&model.snapshot().tracks[0].clips[0].notes[1].cvvcBoundaries==after.cvvcBoundaries);
        model.dispatchPendingMessages();publish(next);
        // Escape must restore the committed line without recording a second operation.
        const auto newAt=roll.diagnosticCvvcBoundary("b",part).getCentre();
        const auto stable=model.contentFingerprint();roll.mouseDown(event(newAt));roll.mouseDrag(event(newAt.translated(20,0)));
        roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));
        check("cvvc_boundary_escape_cancels",model.contentFingerprint()==stable
            &&near(roll.diagnosticPhonemeSpans("b")[index].startSeconds,expected));
        if(part==3&&delta==-.035)
        {
            auto rendered=requestFor();std::vector<UtauPhonemeSpan> actual;
            rendered.notePhonemes=[&](size_t i,const auto& spans){if(i==1)actual=spans;};
            const auto audio=UtauRenderer::render(rendered);
            check("cvvc_boundary_traditional_render_uses_edit",audio.buffer.getNumSamples()>0&&actual.size()==3&&near(actual[2].startSeconds,expected));
            if(modelFolder.isDirectory())
            {
                actual.clear();const auto nsf=renderNsfUtauPhrase(rendered,modelFolder,{});
                check("cvvc_boundary_hifi_render_uses_edit",nsf.buffer.getNumSamples()>0&&actual.size()==3&&near(actual[2].startSeconds,expected));
            }
            // Capture a compact region containing both the upper note and lower lane.
            auto image=roll.createComponentSnapshot(viewport.getViewArea());
            juce::PNGImageFormat png;auto stream=folder.getChildFile("boundary-release-drag.png").createOutputStream();
            if(stream)png.writeImageToStream(image,*stream);
        }
    }
    auto shifted=request;shifted.notes[1].cvvcBoundaries={{2,-.05},{3,.65}};
    auto layout=planChineseCvvc(shifted);
    check("cvvc_two_dividers_keep_both_positions",near(layout.spans[1][1].startSeconds,-.025)
        &&near(layout.spans[1][2].startSeconds,.325)&&near(layout.spans[1].front().startSeconds,-.08)
        &&near(layout.spans[1].back().endSeconds,.5));
    // Existing head-velocity drag must evaluate the same clamped dividers.
    for(const auto target:{-68,260})
    {
        auto state=initial;state.tracks[0].clips[0].notes[1].cvvcBoundaries={{2,-.12},{3,.65}};
        ProjectModel model;model.resetDocument(state);I18n labels;PianoRollComponent roll(model,labels);
        roll.setSize(1600,900);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
        auto r=request;r.notes=AudioEngine::utauRequestNotesForClip(state,clip.id);
        const auto p=planChineseCvvc(r);std::vector<std::pair<juce::String,juce::String>> caches;
        for(size_t i=0;i<p.spans.size();++i){const auto& n=state.tracks[0].clips[0].notes[i];
            caches.emplace_back(n.id,cvvcPhonemeCacheText({AudioEngine::utauNotePhonemeHash(n),
                AudioEngine::utauPhonemeMixContextHash(state.tracks[0]),p.spans[i]}));}
        model.rememberCvvcPhonemes(caches,model.revisionNumber());roll.diagnosticRefresh();
        auto wanted=r;wanted.notes[1].consonantVelocity=target;double lead=0;
        const auto wantedPlan=planChineseCvvc(wanted);
        for(const auto& span:wantedPlan.spans[1])lead=std::max(lead,span.leadInSeconds-span.startSeconds);
        const auto grabbed=roll.diagnosticGrabConsonantHandle(1);roll.diagnosticDragConsonantTo(lead);
        const auto preview=roll.diagnosticConsonantPreview();roll.diagnosticReleaseDrag();
        r.notes=AudioEngine::utauRequestNotesForClip(model.snapshot(),clip.id);double actual=0;
        const auto actualPlan=planChineseCvvc(r);
        for(const auto& span:actualPlan.spans[1])actual=std::max(actual,span.leadInSeconds-span.startSeconds);
        check("manual_cvvc_divider_and_head_velocity_share_timing",grabbed&&near(lead,preview)&&near(actual,preview));
    }
    // Optional real-project probe uses a read-only source and edits only a model copy.
    const juce::File realProject(juce::SystemStats::getEnvironmentVariable("HACHI_TEST_CVVC_PROJECT",{}));
    if(realProject.existsAsFile())
    {
        ProjectModel real;const auto loaded=real.load(realProject,error);int checked=0;bool exact=loaded;
        if(loaded)for(const auto& t:real.snapshot().tracks)if(t.chineseCvvc&&t.utauMode==UtauMode::classic)
            for(const auto& c:t.clips)
            {
                UtauRenderRequest original;original.chineseCvvc=true;original.voicebankDirectory=t.voicebankDirectory;
                original.notes=AudioEngine::utauRequestNotesForClip(real.snapshot(),c.id);
                const auto before=planChineseCvvc(original);
                for(size_t n=0;n<before.spans.size();++n)for(size_t i=1;i<before.spans[n].size();++i)
                {
                    const auto& a=before.spans[n][i-1];const auto& b=before.spans[n][i];
                    auto edited=original;const auto desired=clampCvvcBoundary(b.startSeconds-.02,a.startSeconds,b.endSeconds);
                    edited.notes[n].cvvcBoundaries[b.envelopePart]=desired/edited.notes[n].durationSeconds;
                    const auto after=planChineseCvvc(edited);const auto& spans=after.spans[n];
                    exact&=spans.size()==before.spans[n].size()&&near(spans[i].startSeconds,desired)
                        &&near(spans[i-1].endSeconds,desired)&&near(spans[i-1].startSeconds,a.startSeconds)&&near(spans[i].endSeconds,b.endSeconds);
                    ++checked;
                }
            }
        check("real_project_all_cvvc_internal_dividers_preserve_pair_span",exact&&checked>=3);
    }
    shifted.notes[1].alias="[ba]";
    check("cvvc_manual_alias_ignores_saved_dividers",planChineseCvvc(shifted).spans[1].size()==1);
    shifted=request;shifted.notes[1].cvvcBoundaries={{2,1.e9},{3,-1.e9}};
    layout=planChineseCvvc(shifted);
    check("cvvc_stale_dividers_cannot_invert_phonemes",std::all_of(layout.request.notes.begin(),layout.request.notes.end(),[](const auto& n){return n.durationSeconds>0;})
        &&near(layout.spans[1].back().endSeconds,.5));
}
