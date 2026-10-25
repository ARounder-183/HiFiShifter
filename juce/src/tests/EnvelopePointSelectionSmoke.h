#pragma once
#include "../PianoRollComponent.h"
#include <iostream>

namespace hachi
{
inline bool runEnvelopePointSelectionSmoke(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;juce::Array<juce::var> checks;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;std::cout<<name<<'='<<pass<<std::endl;
        auto* item=new juce::DynamicObject();item->setProperty("name",name);item->setProperty("passed",pass);checks.add(item);};
    for(int mode=0;mode<5;++mode)
    {
        const bool amp=mode==0||mode==1||mode==3,cvvc=mode>=3;
        const juce::String prefix=mode==0?"native_amp_":mode==1?"utau_amp_":mode==2?"flag_":mode==3?"cvvc_amp_":"cvvc_flag_";
        ProjectData data;TrackData track;track.id="t";track.compose=true;track.chineseCvvc=cvvc;
        track.pitchAlgorithm=mode==0?PitchAlgorithm::world:PitchAlgorithm::utau;
        if(cvvc){track.voicebankDirectory=folder.getChildFile("fixture-cvvc");track.voicebankDirectory.createDirectory();}
        NoteData n;n.id="n";n.label="a";n.startSeconds=.5;n.durationSeconds=2;n.midiNote=n.sourceMidiCenter=60;
        n.sourcePitchMeasured=true;n.contour={{0,0,0,true},{2,0,0,true}};n.utauFlagCurveEnabled=true;
        n.amplitudeEnvelope={{0,-60,true},{.4,-6,true},{.8,0,true},{1.2,-3,true},{2,-60,true}};
        n.utauFlagCurves={{"g",{{0,0},{.4,10},{.8,20},{1.2,30},{2,0}}},{"Mt",{{0,5},{2,5}}}};
        if(cvvc)for(int part:{1,2,3})
        {
            n.cvvcEnvelopes[part][""]={{0,-60,true},{.2,-6,true},{.5,0,true},{.8,-3,true},{1,-60,true}};
            n.cvvcEnvelopes[part]["g"]={{0,0,true},{.2,10,true},{.5,20,true},{.8,30,true},{1,0,true}};
        }
        ClipData clip;clip.id="c";clip.durationSeconds=clip.sourceDurationSeconds=3;clip.notes={n};track.clips={clip};data.tracks={track};
        ProjectModel model;model.resetDocument(data);I18n strings;PianoRollComponent roll(model,strings);
        roll.setSize(1400,900);roll.setPixelsPerSecond(350);roll.setFocusedTrack("t");roll.setFocusedClip("c");
        const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};
        const auto reset=[&]{model.resetDocument(data);refresh();roll.setSelectedNoteIds({"n"});
            roll.setTool(amp?PianoRollComponent::Tool::amplitude:PianoRollComponent::Tool::flagCurve);
            roll.setFlagLaneFlag("g");roll.setFlagEditMode(PianoRollComponent::FlagEditMode::perNote);
            if(cvvc){auto waves=std::make_shared<std::vector<UtauNoteWaveform>>();UtauNoteWaveform wave;
                wave.noteId="n";wave.startSeconds=.5;wave.audioHash=AudioEngine::utauNoteAudioHash(n);
                wave.renderHash=AudioEngine::utauNoteRenderHash(n);wave.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(n);
                wave.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(track);
                wave.phonemes={{"VC","vc",0,.4,1},{"CV","cv",.4,1.5,2},{"R","release",1.5,2,3}};
                waves->push_back(wave);roll.setUtauNoteWaveforms(waves);refresh();}};
        const auto stored=[&]{auto snapshot=model.snapshot();return snapshot.tracks[0].clips[0].notes.empty()?NoteData{}:snapshot.tracks[0].clips[0].notes[0];};
        const auto count=[&]{const auto note=stored();if(note.id.isEmpty())return size_t(0);
            if(cvvc)return note.cvvcEnvelopes.at(2).at(amp?juce::String():juce::String("g")).size();
            return amp?note.amplitudeEnvelope.size():flagCurvePointsFor(note,"g").size();};
        const auto at=[&](int index){
            if(cvvc){const auto h=roll.diagnosticCvvcEnvelopeHandles("n",2,amp);return index<static_cast<int>(h.size())?h[(size_t)index]:juce::Point<float>(-100,-100);}
            const double times[]={0,.4,.8,1.2,2};const float values[]={-60,-6,0,-3,-60},flags[]={0,10,20,30,0};
            return juce::Point<float>(roll.diagnosticEdgeX(.5+times[index]),amp?roll.diagnosticAmplitudeLaneY(values[index]):roll.flagLaneY(flags[index]));};
        const auto event=[&](juce::Point<float> p,juce::Point<float> down,bool right=false){return juce::MouseEvent(
            juce::Desktop::getInstance().getMainMouseSource(),p,juce::ModifierKeys(right?juce::ModifierKeys::rightButtonModifier:juce::ModifierKeys::leftButtonModifier),
            0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,p!=down);};
        const auto click=[&](juce::Point<float> p){roll.mouseDown(event(p,p));roll.mouseUp(event(p,p));};
        const auto del=[&]{return roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));};
        reset();
        if(cvvc){const auto offered=roll.diagnosticCvvcEnvelopeHandles("n",2,amp).size()==5;
            check(prefix+"fixture_exposes_phoneme_handles",offered);if(!offered)return false;}
        const auto original=model.contentFingerprint();const auto revision=model.revisionNumber();click(at(2));
        check(prefix+"select_release_does_not_edit",model.contentFingerprint()==original&&model.revisionNumber()==revision);
        if(auto stream=folder.getChildFile(prefix+"selected.png").createOutputStream())
            juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(juce::Rectangle<int>(100,(int)at(2).y-110,1000,235)),*stream);
        check(prefix+"delete_after_release",del()&&stored().id=="n"&&count()==4);
        if(cvvc){const auto note=stored();check(prefix+"other_phonemes_and_lane_unchanged",note.cvvcEnvelopes.at(1).at(amp?juce::String():juce::String("g")).size()==5
            &&note.cvvcEnvelopes.at(3).at(amp?juce::String():juce::String("g")).size()==5
            &&note.cvvcEnvelopes.at(2).at(amp?juce::String("g"):juce::String()).size()==5);}
        else check(prefix+"other_lane_unchanged",amp?flagCurvePointsFor(stored(),"g").size()==5:stored().amplitudeEnvelope.size()==5);
        const auto deleted=model.contentFingerprint();del();del();check(prefix+"repeat_delete_safe",count()==4&&model.contentFingerprint()==deleted);
        check(prefix+"undo",model.undo()&&model.contentFingerprint()==original);refresh();
        check(prefix+"redo",model.redo()&&model.contentFingerprint()==deleted);refresh();
        juce::String error;ProjectModel reopened;check(prefix+"save_reopen",model.save(folder.getChildFile(prefix+"edit.hjpx"),error)
            &&reopened.load(folder.getChildFile(prefix+"edit.hjpx"),error)&&reopened.snapshot().tracks[0].clips[0].notes.size()==1);
        reset();click(at(1));click(at(3));del();check(prefix+"second_point_takes_selection",count()==4);
        reset();auto from=at(2),to=juce::Point<float>(from.x+15,from.y-15);
        roll.mouseDown(event(from,from));roll.mouseDrag(event(to,from));roll.mouseUp(event(to,from));
        check(prefix+"drag_commits",model.contentFingerprint()!=original&&count()==5);
        del();check(prefix+"delete_immediately_after_drag",count()==4);
        reset();click(at(2));roll.focusLost(juce::Component::focusChangedDirectly);del();check(prefix+"focus_change_clears_selection",stored().id.isEmpty());
        reset();click(at(2));roll.setTool(PianoRollComponent::Tool::note);del();check(prefix+"tool_change_clears_selection",stored().id.isEmpty());
        reset();click(at(2));roll.selectAllNotes();del();check(prefix+"select_all_restores_note_delete",stored().id.isEmpty());
        reset();from=at(2);roll.mouseDown(event(from,from));del();roll.mouseUp(event(from,from));refresh();
        check(prefix+"delete_during_drag_not_resurrected",count()==4);
        reset();click(at(2));check(prefix+"backspace",roll.keyPressed(juce::KeyPress(juce::KeyPress::backspaceKey))&&count()==4);
        reset();click(at(0));del();check(prefix+"endpoint_matches_existing_rules",count()==(mode==2?4:5));
        if(!amp){reset();click(at(2));roll.setFlagLaneFlag("Mt");del();check(prefix+"flag_switch_clears_selection",stored().id.isEmpty());}
    }
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",checks);
    folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));return ok;
}
}
