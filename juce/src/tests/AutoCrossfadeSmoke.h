#pragma once
#include "../NativeNoteTiming.h"
namespace hachi
{
inline bool runAutoCrossfadeSmoke(const juce::File& folder)
{
    folder.createDirectory(); bool ok=true;
    const auto check=[&](const char* key,bool pass){ok&=pass;std::cout<<key<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<2.e-4;};
    ClipData a; a.id="a";a.startSeconds=1;a.durationSeconds=a.sourceDurationSeconds=1;
    ClipData b=a;b.id="b";b.startSeconds=1.75;
    NoteData n;n.id="n1";n.durationSeconds=1;n.midiNote=n.sourceMidiCenter=60;n.sourcePitchMeasured=true;
    n.contour={{0,0,0,true},{1,0,0,true}};a.notes={n};n.id="n2";b.notes={n};
    TrackData track;track.id="track";track.pitchAlgorithm=PitchAlgorithm::world;track.allowNativeAudioOverlap=true;track.clips={a,b};
    ProjectData data;data.tracks={track};ProjectModel model;model.resetDocument(data);
    I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1300,750);roll.setPixelsPerSecond(300);
    roll.setFocusedTrack(track.id);roll.setSelectedNoteIds({"n1","n2"});roll.diagnosticRefresh();
    const auto original=model.contentFingerprint();
    check("ui_action_finds_native_overlap",roll.crossfadeSelectedNotes()==1);
    auto edited=model.snapshot();
    const auto gain=[](const auto& note,double t){return backend::envelopeGainFromDb(nativeEnvelopeDbAt(note.amplitudeEnvelope,t));};
    check("linear_complementary_gains",near(gain(edited.tracks[0].clips[0].notes[0],.875),.5)
        &&near(gain(edited.tracks[0].clips[1].notes[0],.125),.5));
    check("timing_and_pitch_unchanged",edited.tracks[0].clips[0].startSeconds==1 &&edited.tracks[0].clips[1].startSeconds==1.75
        &&edited.tracks[0].clips[0].notes[0].durationSeconds==1 &&edited.tracks[0].clips[1].notes[0].midiNote==60);
    bool renders=true;
    for(auto algorithm:{PitchAlgorithm::world,PitchAlgorithm::nsfHifigan,PitchAlgorithm::mld5,PitchAlgorithm::llsm2,PitchAlgorithm::vocalShifter})
    {
        auto tr=edited.tracks[0];tr.pitchAlgorithm=algorithm;
        const auto left=AudioEngine::diagnosticNativeRequest(tr.clips[0],tr);
        const auto right=AudioEngine::diagnosticNativeRequest(tr.clips[1],tr);
        renders &= near(left.noteGain[175],.5)&&near(right.noteGain[25],.5);
    }
    check("all_native_request_backends_receive_fades",renders);
    const auto once=model.contentFingerprint();
    check("repeat_action_is_idempotent",roll.crossfadeSelectedNotes()==1&&model.contentFingerprint()==once);
    check("single_undo_restores_both_notes",model.undo()&&model.contentFingerprint()==original);
    check("single_redo_restores_both_notes",model.redo()&&model.contentFingerprint()==once);
    juce::String error;ProjectModel reopened;
    check("saved_fades_reopen",model.save(folder.getChildFile("crossfade.hjpx"),error)&&reopened.load(folder.getChildFile("crossfade.hjpx"),error)
        &&near(gain(reopened.snapshot().tracks[0].clips[0].notes[0],.875),.5));
    auto image=roll.createComponentSnapshot(roll.getLocalBounds());juce::PNGImageFormat png;
    if(auto out=folder.getChildFile("crossfade.png").createOutputStream())png.writeImageToStream(image,*out);
    auto spans=std::vector<ProjectModel::CrossfadeSpan>{{"n1",0,1,2,{{1,0,true},{2,0,true}}},
        {"n2",0,1.75,2.75,{{1.75,0,true},{2.75,0,true}}}};
    check("stale_resolution_rejected",model.crossfadeNotes(spans,model.revisionNumber()+1)==0);
    auto separated=data;separated.tracks[0].clips[1].startSeconds=2;
    model.resetDocument(separated);roll.diagnosticRefresh();roll.setSelectedNoteIds({"n1","n2"});
    const auto noOverlap=model.contentFingerprint();
    check("touching_notes_not_moved_or_faded",roll.crossfadeSelectedNotes()==0&&model.contentFingerprint()==noOverlap);
    auto splitTrack=data;splitTrack.tracks.push_back(track);splitTrack.tracks[1].id="other";
    splitTrack.tracks[0].clips={a};splitTrack.tracks[1].clips={b};model.resetDocument(splitTrack);
    check("cross_track_rejected",model.crossfadeNotes(spans,model.revisionNumber())==0);
    // The same persisted envelope is consumed by every UTAU request variant.
    for(auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        auto utau=data;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;utau.tracks[0].utauMode=mode;
        model.resetDocument(utau);check("utau_action",model.crossfadeNotes(spans,model.revisionNumber())==1);
        const auto notes=AudioEngine::utauRequestNotesForClip(model.snapshot(),"a");
        check("utau_request_has_authored_fade",!notes.empty()&&notes[0].amplitudeEnvelope.size()==3
            &&notes[0].amplitudeEnvelope.back().gainDb==-60&&notes[0].amplitudeEnvelope[1].linearToNext);
    }
    // Curved fades use the same persisted automation in every synthesis mode.
    model.resetDocument(data);roll.diagnosticRefresh();roll.setSelectedNoteIds({"n1","n2"});
    check("curved_ui_action",roll.crossfadeSelectedNotes(true)==1);
    const auto curved=model.snapshot();const auto curvedHash=model.contentFingerprint();
    double sumError=0,shapeError=0;
    for(int i=0;i<=1000;++i)
    {
        const auto u=i/1000.;const auto expected=u*u*(3-2*u);
        const auto left=gain(curved.tracks[0].clips[0].notes[0],.75+.25*u);
        const auto right=gain(curved.tracks[0].clips[1].notes[0],.25*u);
        sumError=std::max(sumError,static_cast<double>(std::abs(left+right-1)));
        shapeError=std::max(shapeError,std::abs(right-expected));
    }
    std::cout<<"curve_max_sum_error="<<sumError<<" curve_max_shape_error="<<shapeError<<std::endl;
    // The shared envelope convention treats <= -59.9 dB (~0.00101 gain)
    // as silence. Include that existing floor, not only sampling error.
    check("curve_matches_cvvc_smoothstep_and_is_complementary",sumError<.0011&&shapeError<.0011);
    bool nativeCurves=true;
    for(auto algorithm:{PitchAlgorithm::world,PitchAlgorithm::nsfHifigan,PitchAlgorithm::mld5,PitchAlgorithm::llsm2,PitchAlgorithm::vocalShifter})
    {
        auto tr=curved.tracks[0];tr.pitchAlgorithm=algorithm;
        const auto left=AudioEngine::diagnosticNativeRequest(tr.clips[0],tr);
        const auto right=AudioEngine::diagnosticNativeRequest(tr.clips[1],tr);
        nativeCurves&=std::abs(left.noteGain[162]-.854848)<.0011&&std::abs(right.noteGain[12]-.145152)<.0011;
    }
    check("all_native_request_backends_receive_curves",nativeCurves);
    check("curve_is_different_from_linear",near(gain(curved.tracks[0].clips[1].notes[0],.0625),.15625));
    check("curved_repeat_is_idempotent",roll.crossfadeSelectedNotes(true)==1&&model.contentFingerprint()==curvedHash);
    check("curve_undo_redo",model.undo()&&model.contentFingerprint()==original&&model.redo()&&model.contentFingerprint()==curvedHash);
    check("curved_save_reopen",model.save(folder.getChildFile("curved.hjpx"),error)&&reopened.load(folder.getChildFile("curved.hjpx"),error)
        &&near(gain(reopened.snapshot().tracks[0].clips[1].notes[0],.0625),.15625));
    check("switch_back_removes_curve_points",roll.crossfadeSelectedNotes(false)==1
        &&model.snapshot().tracks[0].clips[1].notes[0].amplitudeEnvelope.size()==3);
    bool curveRequests=true;
    for(auto mode:{UtauMode::classic,UtauMode::jie,UtauMode::mou})
    {
        auto utau=data;utau.tracks[0].pitchAlgorithm=PitchAlgorithm::utau;utau.tracks[0].utauMode=mode;
        model.resetDocument(utau);curveRequests&=model.crossfadeNotes(spans,model.revisionNumber(),true)==1;
        const auto notes=AudioEngine::utauRequestNotesForClip(model.snapshot(),"b");
        curveRequests&=!notes.empty()&&notes[0].amplitudeEnvelope.size()==34;
    }
    check("all_utau_modes_receive_curve_points",curveRequests);
    auto parts=spans;parts[0].part=2;parts[1].part=3;model.resetDocument(data);
    check("cvvc_curve_action",model.crossfadeNotes(parts,model.revisionNumber(),true)==1);
    auto cvvc=model.snapshot();
    const auto& part=cvvc.tracks[0].clips[1].notes[0].cvvcEnvelopes.at(3).at(juce::String());
    check("cvvc_curve_same_gain",near(backend::envelopeGainFromDb(backend::cvvcEnvelopeValue(part,.0625,true)),.15625));
    const auto partHash=model.contentFingerprint();
    check("cvvc_curve_save_reopen",model.save(folder.getChildFile("cvvc-curved.hjpx"),error)&&reopened.load(folder.getChildFile("cvvc-curved.hjpx"),error)
        &&reopened.contentFingerprint()==partHash);
    {
        MainComponent component;
        const auto chosen=[](const juce::PopupMenu& menu){int id=0;for(juce::PopupMenu::MenuItemIterator it(menu);it.next();)if(it.getItem().isTicked)id=it.getItem().itemID;return id;};
        const auto before=chosen(component.autoCrossfadeModeMenu());
        component.selectAutoCrossfadeMode(2);
        check("menu_curve_checked",chosen(component.autoCrossfadeModeMenu())==2);
        component.selectAutoCrossfadeMode(0);
        check("cancel_menu_keeps_mode",chosen(component.autoCrossfadeModeMenu())==2);
        {MainComponent reopenedUi;check("menu_preference_reopens",chosen(reopenedUi.autoCrossfadeModeMenu())==2);}
        component.selectAutoCrossfadeMode(1);
        check("menu_line_checked",chosen(component.autoCrossfadeModeMenu())==1);
        component.selectAutoCrossfadeMode(before);
    }
    return ok;
}
}
