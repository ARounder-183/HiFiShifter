#pragma once
#include "../MainComponent.h"
#include <iostream>
namespace hachi {
inline bool MainComponent::diagnosticDiffSingerHistoryPlayback(const juce::File& bank,const juce::File& output)
{
    output.createDirectory();stopTimer();audio.stop();bool ok=true;juce::String log,error;
    const auto check=[&](const char* name,bool pass){ok&=pass;log+=juce::String(name)+"="+(pass?"PASS":"FAIL")+"\n";std::cout<<name<<'='<<pass<<std::endl;};
    ProjectData data;TrackData track;track.id="ds-history";track.trackType=TrackType::diffSinger;
    track.pitchAlgorithm=PitchAlgorithm::utau;track.utauMode=UtauMode::mou;track.voicebankDirectory=bank;
    ClipData clip;clip.id="ds-history-clip";clip.durationSeconds=6;
    for(int i=0;i<4;++i){NoteData n;n.id="note-"+juce::String(i);n.label=i%2?"wang":"zhang";
        n.startSeconds=i==0?1.:i==1?2.5:i==2?4.:4.3;n.durationSeconds=.7;n.midiNote=60+(float)i;
        clip.notes.push_back(n);}
    track.clips.push_back(clip);data.tracks.push_back(track);project.resetDocument(data);project.dispatchPendingMessages();
    selectedTrackId=track.id;selectedClipId=clip.id;pianoRoll.setFocusedTrack(track.id);pianoRoll.setFocusedClip(clip.id);
    pianoRoll.diagnosticRefresh();refreshProjectControls();
    const auto wait=[&]{for(int i=0;i<2400&&audio.renderProgress();++i)juce::Thread::sleep(25);return !audio.renderProgress();};
    const auto select=[&](int i){pianoRoll.setSelectedNoteIds({"note-"+juce::String(i)});updateUtauRenderSelection();return wait();};
    const auto publish=[&]{audio.refreshUtauWaveforms();pianoRoll.setUtauNoteWaveforms(audio.utauNoteWaveforms());};
    const auto visible=[&](int i){const auto id="note-"+juce::String(i);
        const auto picture=pianoRoll.diagnosticDrawnNoteWaveform(id);
        return picture && !picture->picture.empty() && !pianoRoll.diagnosticPhonemeSpans(id).empty();};
    // A visited but failed selection must not be mistaken for rendered audio.
    check("overlap_selection_finishes_with_warning",select(2)&&audio.activeRenderWarnings().contains(juce::String::fromUTF8("单声部")));
    check("first_valid_selection_renders",select(0)&&audio.hasCurrentRenderedAudio()&&audio.activeRenderWarnings().isEmpty());
    check("first_selection_capture",audio.exportWav(output.getChildFile("first.wav"),error));
    publish();check("first_DS_wave_and_regions_visible",visible(0));
    const auto firstPicture=pianoRoll.diagnosticDrawnNoteWaveform("note-0");
    const auto firstRegions=pianoRoll.diagnosticPhonemeSpans("note-0");
    pianoRoll.setSelectedNoteIds({"note-1"});updateUtauRenderSelection();publish();
    check("pending_other_selection_keeps_completed_DS_display",visible(0));
    check("second_valid_selection_renders",wait()&&audio.hasCurrentRenderedAudio()&&audio.activeRenderWarnings().isEmpty());
    publish();check("both_completed_DS_waveforms_and_regions_visible",visible(0)&&visible(1));
    const auto retainedPicture=pianoRoll.diagnosticDrawnNoteWaveform("note-0");
    const auto retainedRegions=pianoRoll.diagnosticPhonemeSpans("note-0");
    bool sameRegions=firstRegions.size()==retainedRegions.size();
    for(size_t i=0;i<std::min(firstRegions.size(),retainedRegions.size());++i)
        sameRegions &= firstRegions[i].token==retainedRegions[i].token
            &&firstRegions[i].startSeconds==retainedRegions[i].startSeconds&&firstRegions[i].endSeconds==retainedRegions[i].endSeconds;
    check("previous_DS_display_is_unchanged",firstPicture&&retainedPicture
        &&firstPicture->picture==retainedPicture->picture&&sameRegions);
    rememberDiffSingerParameters();project.dispatchPendingMessages();publish();
    check("remembered_predictions_keep_completed_DS_display",visible(0)&&visible(1));
    check("second_selection_capture",audio.exportWav(output.getChildFile("second.wav"),error));
    check("failed_new_scope_does_not_remove_completed_DS_display",select(2));publish();
    check("failed_scope_retains_both_completed_regions",visible(0)&&visible(1)&&!visible(2));
    // Revisit an already completed scope: history must not double its amplitude.
    check("revisited_selection_renders",select(0)&&audio.hasCurrentRenderedAudio());
    publish();check("revisit_keeps_other_notes_and_unique_waveforms",visible(0)&&visible(1)
        &&audio.utauNoteWaveforms()->size()==2);
    pianoRoll.setSelectedNoteIds({});updateUtauRenderSelection();audio.setPosition(0);
    publish();check("clearing_selection_keeps_DS_display",visible(0)&&visible(1));
    check("space_key_handled",keyPressed(juce::KeyPress(juce::KeyPress::spaceKey)));
    check("space_reuses_completed_DS_without_new_inference",audio.isPlaying()&&!playWhenRenderReady
        &&audio.hasCurrentRenderedAudio()&&!audio.renderProgress()&&audio.activeRenderWarnings().isEmpty());
    check("space_key_stops",keyPressed(juce::KeyPress(juce::KeyPress::spaceKey))&&!audio.isPlaying());
    check("history_capture",audio.exportWav(output.getChildFile("history.wav"),error));
    publish();check("history_playback_keeps_DS_display",visible(0)&&visible(1));
    setSize(1400,900);pianoRoll.setPixelsPerSecond(180);pianoRoll.diagnosticRefresh();
    pianoRoll.setShowUtauWaveforms(true);pianoRoll.setShowPhonemeInfo(true);
    if(auto image=output.getChildFile("DS-retained-waveforms.png").createOutputStream())
        juce::PNGImageFormat().writeImageToStream(createComponentSnapshot(getLocalBounds()),*image);
    juce::AudioFormatManager formats;formats.registerBasicFormats();
    const auto read=[&](const char* name){juce::AudioBuffer<float> samples;
        if(auto reader=std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(output.getChildFile(name))))
        {samples.setSize(2,(int)reader->lengthInSamples);reader->read(&samples,0,samples.getNumSamples(),0,true,true);}return samples;};
    const auto first=read("first.wav"),second=read("second.wav"),history=read("history.wav");
    bool matches=history.getNumSamples()>0&&history.getNumSamples()==first.getNumSamples()&&history.getNumSamples()==second.getNumSamples();
    double residual=0,energy=0;
    if(matches)for(int ch=0;ch<2;++ch)for(int s=0;s<history.getNumSamples();++s)
    {const auto expected=first.getSample(ch,s)+second.getSample(ch,s),delta=history.getSample(ch,s)-expected;energy+=expected*expected;residual+=delta*delta;}
    check("history_PCM_contains_both_scopes_once",matches&&energy>1e-5&&residual/energy<1e-5);
    for(int i=0;i<3;++i){audio.setPosition(.5);keyPressed(juce::KeyPress(juce::KeyPress::spaceKey));
        check("repeat_space_keeps_cached_audio_and_red_line",audio.isPlaying()&&!audio.renderProgress()&&audio.position()>=.5&&audio.position()<.7);
        audio.stop();}
    project.setNoteLabel("note-0","RR");project.dispatchPendingMessages();
    check("edited_score_does_not_replay_stale_history",!audio.hasCurrentRenderedAudio());
    publish();check("score_edit_invalidates_stale_DS_display",!visible(0)&&!visible(1));
    // Explicit selection still requests fresh synthesis and obeys DS overlap rules.
    check("new_selection_still_validates_overlap",select(2)&&audio.activeRenderWarnings().contains(juce::String::fromUTF8("单声部")));
    output.getChildFile("report.txt").replaceWithText(log);return ok;
}
}
