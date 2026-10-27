#pragma once
#include "../NativePitchIdentity.h"
namespace hachi
{
inline bool runNativeSourcePitchRestoreSmoke(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;int checks=0;
    const auto check=[&](const char* name,bool pass){ok &= pass;++checks;std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<1.e-5;};
    ClipData clip;clip.id="clip";clip.startSeconds=2.35;clip.durationSeconds=1.6;
    clip.sourceOffsetSeconds=.1;clip.sourceDurationSeconds=.8;clip.gain=.65;
    clip.sourceTimeMap={{0,0},{.4,.1},{1.2,.7},{1.6,.8}};
    for(int i=0;i<3;++i)
    {
        NoteData n;n.id="n"+juce::String(i);n.label="source";n.startSeconds=i==0?0:i==1?.4:1.2;
        n.durationSeconds=i==1?.8:.4;n.midiNote=63;n.sourceMidiCenter=57;n.sourcePitchMeasured=true;
        n.nativeSourceStartSeconds=i==0?0:i==1?.1:.7;n.nativeSourceEndSeconds=i==0?.1:i==1?.7:.8;
        n.nativeSegments={{"v","source",NativeSegmentRole::vowel,0,n.durationSeconds}};
        n.gain=.8f;n.amplitudeEnvelope={{0,-3},{n.durationSeconds/2,0},{n.durationSeconds,-2}};
        n.formantSemitones=1.5f;n.breath=.1f;n.tension=.2f;
        n.drift=.2f;n.modulation=0;n.vibratoEnabled=true;n.vibratoRealLine=true;n.robustPitchCurve=true;
        n.pitchControlPoints={{0,65},{n.durationSeconds,60}};
        for(int k=0;k<=static_cast<int>(std::round(n.durationSeconds/.005));++k)
        {
            const auto t=k*.005;const auto cents=static_cast<float>(60*std::sin((n.startSeconds+t)*12));
            PitchPoint point{t,cents,cents*.6f,!(t>.15&&t<.2)};
            point.hasManualTarget=true;point.manualTargetCents=200-100*static_cast<float>(t);
            n.contour.push_back(point);
        }
        clip.notes.push_back(n);
    }
    TrackData track;track.id="native";track.pitchAlgorithm=PitchAlgorithm::world;track.clips={clip};
    ProjectData edited;edited.tracks={track};ProjectModel model;model.replace(edited);
    model.setNotesConnection({"n0","n1","n2"},true);edited=model.snapshot();
    const auto beforeHash=AudioEngine::nativeClipWaveformHash(edited.tracks[0].clips[0],edited.tracks[0]);
    I18n strings;PianoRollComponent roll(model,strings);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
    roll.setTool(PianoRollComponent::Tool::note);roll.setPixelsPerSecond(240);roll.setRowHeight(24);roll.setSize(1300,roll.getHeight());
    roll.diagnosticRefresh();
    const auto has=[](const auto& ids,int id){return std::find(ids.begin(),ids.end(),id)!=ids.end();};
    check("native_context_offers_restore",has(roll.diagnosticNoteMenuIds("n0"),28)
        &&has(roll.diagnosticEnabledNoteMenuIds("n0"),28));
    const auto event=[](juce::Component& c,juce::Point<float> at,juce::Point<float> down,int mods)
    {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(mods),0,0,0,0,0,
        &c,&c,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,at!=down);};
    const auto first=roll.diagnosticHitBounds(0),second=roll.diagnosticHitBounds(1);
    const juce::Point<float> from(first.getX()-12,first.getY()-18),to(second.getRight()-.5f,second.getBottom()+18);
    roll.mouseDown(event(roll,from,from,juce::ModifierKeys::rightButtonModifier));
    roll.mouseDrag(event(roll,to,from,juce::ModifierKeys::rightButtonModifier));
    roll.mouseUp(event(roll,to,from,juce::ModifierKeys::rightButtonModifier));
    juce::PopupMenu::dismissAllActiveMenus();
    check("right_marquee_selects_two_fragments",roll.diagnosticSelectedCount()==2);
    roll.applyNoteMenuChoice("n0",28,0,{});model.dispatchPendingMessages();roll.diagnosticRefresh();
    const auto restored=model.snapshot();const auto& result=restored.tracks[0].clips[0];
    const auto sourceSame=[&](const auto& a,const auto& b){
        if(a.contour.size()!=b.contour.size())return false;
        for(std::size_t i=0;i<a.contour.size();++i)
            if(!near(a.contour[i].timeSeconds,b.contour[i].timeSeconds)
                ||!near(a.contour[i].relativeCents,b.contour[i].relativeCents)
                ||!near(a.contour[i].withoutVibratoCents,b.contour[i].withoutVibratoCents)
                ||a.contour[i].voiced!=b.contour[i].voiced)return false;
        return true;};
    const auto timingSame=[&](const ClipData& a,const ClipData& b){
        if(!near(a.startSeconds,b.startSeconds)||!near(a.durationSeconds,b.durationSeconds)
            ||!near(a.sourceOffsetSeconds,b.sourceOffsetSeconds)||!near(a.sourceDurationSeconds,b.sourceDurationSeconds)
            ||a.sourceTimeMap.size()!=b.sourceTimeMap.size()||a.notes.size()!=b.notes.size())return false;
        for(std::size_t i=0;i<a.sourceTimeMap.size();++i)
            if(!near(a.sourceTimeMap[i].targetSeconds,b.sourceTimeMap[i].targetSeconds)
                ||!near(a.sourceTimeMap[i].sourceSeconds,b.sourceTimeMap[i].sourceSeconds))return false;
        for(std::size_t i=0;i<a.notes.size();++i)
            if(!near(a.notes[i].startSeconds,b.notes[i].startSeconds)||!near(a.notes[i].durationSeconds,b.notes[i].durationSeconds)
                ||!near(a.notes[i].nativeSourceStartSeconds,b.notes[i].nativeSourceStartSeconds)
                ||!near(a.notes[i].nativeSourceEndSeconds,b.notes[i].nativeSourceEndSeconds))return false;
        return true;};
    check("restore_keeps_timeline_and_nonlinear_source_clock",timingSame(result,clip));
    bool sourceRetained=true,originalPitch=true,nonPitch=true;
    for(int i=0;i<2;++i){const auto& n=result.notes[i];const auto& old=clip.notes[i];
        sourceRetained &= sourceSame(n,old);
        originalPitch &= nativePitchIsUnedited(n)&&n.pitchControlPoints.empty()&&!n.robustPitchCurve
            &&!n.connectedToPrevious&&!n.connectedToNext&&!n.utauAutoPitchTransition;
        nonPitch &= near(n.gain,old.gain)&&near(n.formantSemitones,old.formantSemitones)
            &&near(n.breath,old.breath)&&near(n.tension,old.tension)&&n.label==old.label
            &&n.amplitudeEnvelope.size()==old.amplitudeEnvelope.size()&&n.nativeSegments.size()==old.nativeSegments.size();
        for(std::size_t j=0;j<n.amplitudeEnvelope.size();++j)
            nonPitch &= near(n.amplitudeEnvelope[j].timeSeconds,old.amplitudeEnvelope[j].timeSeconds)
                &&near(n.amplitudeEnvelope[j].gainDb,old.amplitudeEnvelope[j].gainDb);
    }
    check("dense_source_f0_and_unvoiced_mask_retained",sourceRetained);
    check("batch_restore_removes_all_pitch_edits",originalPitch);
    check("gain_envelope_segments_and_timbre_preserved",nonPitch&&near(result.gain,clip.gain));
    check("unselected_note_pitch_edits_untouched",near(result.notes[2].midiNote,clip.notes[2].midiNote)
        &&result.notes[2].vibratoEnabled&&result.notes[2].pitchControlPoints.size()==2
        &&result.notes[2].contour[4].hasManualTarget);
    check("synthetic_pitch_connections_are_unlinked",restored.nativeConnections.empty()&&!result.notes[2].connectedToPrevious);
    const auto target=AudioEngine::diagnosticNativeTargetMidi(restored,0,0);
    // Compare on the renderer's actual 5 ms grid against an unedited source
    // baseline with the identical warp, including its voiced/unvoiced boundary
    // sampling.  Point times minus region offsets can straddle those boundaries
    // by floating-point rounding, so analytic point labels are not a grid oracle.
    auto sourceBaseline=edited;
    for(auto& n:sourceBaseline.tracks[0].clips[0].notes)
    {
        n.midiNote=n.sourceMidiCenter;n.modulation=n.drift=1;
        n.vibratoEnabled=false;n.pitchControlPoints.clear();n.robustPitchCurve=false;
        n.connectedToPrevious=n.connectedToNext=false;n.utauAutoPitchTransition=false;
        for(auto& point:n.contour){point.hasManualTarget=false;point.manualTargetCents=0;}
    }
    const auto sourceTargets=AudioEngine::diagnosticNativeTargetMidi(sourceBaseline,0,0);
    bool rendererMatches=target.size()==sourceTargets.size();
    for(std::size_t frame=0;rendererMatches&&frame<target.size()&&frame*.005<1.2;++frame)
        rendererMatches=near(target[frame],sourceTargets[frame]);
    check("actual_renderer_uses_original_pitch_at_every_stretched_frame",rendererMatches);
    check("render_and_waveform_cache_invalidated",beforeHash!=AudioEngine::nativeClipWaveformHash(result,restored.tracks[0]));
    const auto revision=model.revisionNumber();roll.applyNoteMenuChoice("n0",28,0,{});
    check("repeat_restore_is_noop",model.revisionNumber()==revision);
    juce::String error;const auto projectFile=folder.getChildFile("restored-pitch.hjpx");
    check("restored_project_saved",model.save(projectFile,error));ProjectModel reopened;
    check("restored_project_reopens",reopened.load(projectFile,error));const auto loaded=reopened.snapshot();
    check("saved_restore_preserves_pitch_and_warp",nativePitchIsUnedited(loaded.tracks[0].clips[0].notes[0])
        &&nativePitchIsUnedited(loaded.tracks[0].clips[0].notes[1])&&timingSame(loaded.tracks[0].clips[0],result));
    model.undo();model.dispatchPendingMessages();roll.diagnosticRefresh();const auto undone=model.snapshot();
    check("one_undo_restores_batch_pitch_and_connections",beforeHash==AudioEngine::nativeClipWaveformHash(undone.tracks[0].clips[0],undone.tracks[0])
        &&undone.nativeConnections.size()==edited.nativeConnections.size());
    model.replace(edited);model.dispatchPendingMessages();roll.diagnosticRefresh();
    const auto at=roll.diagnosticHitBounds(2).getCentre();
    roll.mouseDown(event(roll,at,at,juce::ModifierKeys::leftButtonModifier));
    roll.mouseUp(event(roll,at,at,juce::ModifierKeys::leftButtonModifier));
    roll.applyNoteMenuChoice("n2",28,0,{});model.dispatchPendingMessages();roll.diagnosticRefresh();
    check("single_selected_fragment_restore",nativePitchIsUnedited(model.snapshot().tracks[0].clips[0].notes[2])
        &&model.snapshot().tracks[0].clips[0].notes[0].contour[2].hasManualTarget);
    model.transposeNotes({"n2"},2);check("restored_pitch_can_be_edited_again",!nativePitchIsUnedited(model.snapshot().tracks[0].clips[0].notes[2]));
    roll.flattenPitchLine("n2");model.dispatchPendingMessages();roll.diagnosticRefresh();roll.restoreOriginalPitch("n2");
    check("double_click_flatten_can_be_restored",nativePitchIsUnedited(model.snapshot().tracks[0].clips[0].notes[2]));
    auto unavailable=edited;unavailable.tracks[0].clips[0].notes[0].sourceMidiCenter=-1;
    model.replace(unavailable);model.dispatchPendingMessages();roll.diagnosticRefresh();
    const auto unknownAt=roll.diagnosticHitBounds(0).getCentre();
    roll.mouseDown(event(roll,unknownAt,unknownAt,juce::ModifierKeys::leftButtonModifier));
    roll.mouseUp(event(roll,unknownAt,unknownAt,juce::ModifierKeys::leftButtonModifier));
    check("unknown_source_pitch_is_disabled",!has(roll.diagnosticEnabledNoteMenuIds("n0"),28));
    check("unknown_source_pitch_not_fabricated",!model.restoreNativeSourcePitch({"n0"}));
    auto utau=edited;utau.tracks[0].trackType=TrackType::utau;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;model.replace(utau);
    model.dispatchPendingMessages();roll.diagnosticRefresh();const auto beforeUtau=model.revisionNumber();
    check("utau_menu_excludes_native_restore",!has(roll.diagnosticNoteMenuIds("n0"),28));
    roll.applyNoteMenuChoice("n0",28,0,{});
    check("utau_restore_refused",model.revisionNumber()==beforeUtau&&!model.restoreNativeSourcePitch({"n0"}));
    std::cout<<"native_source_pitch_restore_ok="<<ok<<"; checks="<<checks<<std::endl;return ok;
}
}
