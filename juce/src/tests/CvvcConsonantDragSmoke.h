#pragma once
namespace hachi
{
template<class Check>
void verifyCvvcConsonantDrag(const ProjectData& initial, Check&& check)
{
    const auto leadFor = [](const auto& spans)
    {
        double lead = 0;
        for (const auto& p : spans) lead = std::max(lead, p.leadInSeconds - p.startSeconds);
        return lead;
    };
    for (const int target : {-300, -68, 0, 100, 260, 500})
    {
        ProjectModel model; model.resetDocument(initial);
        I18n labels; PianoRollComponent roll(model, labels);
        roll.setSize(1200, 750); roll.setPixelsPerSecond(700);
        const auto& track = initial.tracks[0]; const auto& clip = track.clips[0];
        roll.setFocusedTrack(track.id); roll.setFocusedClip(clip.id);
        const auto requestFor = [&]
        {
            backend::UtauRenderRequest r; r.chineseCvvc = true;
            r.voicebankDirectory = track.voicebankDirectory; r.bpm = initial.bpm;
            r.targetDurationSeconds = clip.durationSeconds;
            r.notes = AudioEngine::utauRequestNotesForClip(model.snapshot(), clip.id);
            return r;
        };
        const auto publish = [&](const auto& plan)
        {
            const auto data = model.snapshot(); const auto& current = data.tracks[0];
            std::vector<std::pair<juce::String,juce::String>> caches;
            for (size_t i = 0; i < plan.spans.size(); ++i)
            {
                const auto& n = current.clips[0].notes[i];
                caches.emplace_back(n.id, backend::cvvcPhonemeCacheText({AudioEngine::utauNotePhonemeHash(n),
                    AudioEngine::utauPhonemeMixContextHash(current), plan.spans[i]}));
            }
            model.rememberCvvcPhonemes(caches, model.revisionNumber()); roll.diagnosticRefresh();
        };
        publish(backend::planChineseCvvc(requestFor()));
        auto desired = requestFor(); desired.notes[1].consonantVelocity = target;
        const auto wantedPlan = backend::planChineseCvvc(desired);
        const auto wanted = leadFor(wantedPlan.spans[1]);
        const auto grabbed = roll.diagnosticGrabConsonantHandle(1);
        roll.diagnosticDragConsonantTo(wanted);
        const auto preview = roll.diagnosticConsonantPreview();
        const auto velocity = roll.diagnosticConsonantVelocity();
        check(("cvvc_head_preview_matches_planner_"+juce::String(target)).toRawUTF8(),
            grabbed && std::abs(preview-wanted)<1.e-6);
        roll.diagnosticReleaseDrag(); model.dispatchPendingMessages(); roll.diagnosticRefresh();
        const auto committed = requestFor(); const auto plan = backend::planChineseCvvc(committed);
        check(("cvvc_head_release_matches_preview_"+juce::String(target)).toRawUTF8(),
            committed.notes[1].consonantVelocity == velocity && std::abs(leadFor(plan.spans[1])-preview)<1.e-6);
        check(("cvvc_head_waiting_layout_keeps_position_"+juce::String(target)).toRawUTF8(),
            std::abs(clip.startSeconds+clip.notes[1].startSeconds-roll.diagnosticSoundingSpan("1").first-preview)<1.e-6);
        publish(plan);
        check(("cvvc_head_completed_layout_keeps_position_"+juce::String(target)).toRawUTF8(),
            std::abs(clip.startSeconds+clip.notes[1].startSeconds-roll.diagnosticSoundingSpan("1").first-preview)<1.e-6);
        if (target == -68)
        {
            auto rendered = committed; double actualLead = -1;
            rendered.notePiece = [&](size_t i,const auto&,double,double lead,const auto&,const auto&)
            { if (i==1) actualLead=lead; };
            const auto audio = backend::UtauRenderer::render(rendered);
            check("cvvc_negative_velocity_actual_audio_matches_drag",audio.buffer.getNumSamples()>0
                &&std::abs(actualLead-preview)<2.0/44100);
            model.undo(); model.dispatchPendingMessages(); roll.diagnosticRefresh();
            check("cvvc_head_drag_single_undo_restores_velocity",requestFor().notes[1].consonantVelocity==100);
        }
    }
    // OTO pre beyond the fixed consonant region must not scale its vowel part.
    const auto timing = backend::UtauSampleTiming{.15,.1,.02,false,{}, {},.5,{}};
    check("cvvc_preutterance_keeps_unscaled_region",std::abs(backend::scaledUtauPreutterance(timing,0)-.25)<1.e-9);
    check("cvvc_preutterance_caps_to_source_audio",std::abs(backend::scaledUtauPreutterance(timing,-300)-.5)<1.e-9);
    for (const auto mode : {UtauMode::classic, UtauMode::jie, UtauMode::mou})
        for (const int velocity : {-68, 260})
        {
            auto data=initial; auto& track=data.tracks[0];
            track.chineseCvvc=false; track.utauMode=mode;
            ProjectModel model; model.resetDocument(data); I18n labels; PianoRollComponent roll(model,labels);
            roll.setSize(1200,750); roll.setFocusedTrack(track.id); roll.diagnosticRefresh();
            const auto& note=track.clips[0].notes[1];
            const auto timing=backend::UtauRenderer::sampleTiming(track.voicebankDirectory,note.label,note.midiNote,
                velocity,utauModeUsesRegions(mode),mode==UtauMode::mou);
            const auto grabbed=roll.diagnosticGrabConsonantHandle(1);
            roll.diagnosticDragConsonantTo(timing ? timing->preutteranceSeconds : 0);
            const auto preview=roll.diagnosticConsonantPreview();
            roll.diagnosticReleaseDrag();model.dispatchPendingMessages();roll.diagnosticRefresh();
            check(("ordinary_utau_head_preview_and_commit_mode_"+juce::String(static_cast<int>(mode))+"_"+juce::String(velocity)).toRawUTF8(),
                timing&&grabbed&&std::abs(preview-timing->preutteranceSeconds)<1.e-6
                &&std::abs(note.startSeconds+track.clips[0].startSeconds-roll.diagnosticSoundingSpan(note.id).first-preview)<1.e-6
                &&model.snapshot().tracks[0].clips[0].notes[1].utauConsonantVelocity==velocity);
        }
}
}
