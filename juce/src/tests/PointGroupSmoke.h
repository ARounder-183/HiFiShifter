#pragma once
#include "../PianoRollComponent.h"
#include "../DiffSingerParameterCurves.h"
namespace hachi
{
inline bool runPointGroupSmoke(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;
    const auto check=[&](const juce::String& name,bool pass){ok&=pass;std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<2.e-4;};
    for(int mode=0;mode<10;++mode)
    {
        const bool pitch=mode<3,amp=mode==3||mode==4||mode==6,cvvc=mode==6||mode==7,native=mode==0||mode==3;
        const juce::String prefix="group"+juce::String(mode)+"_";
        ProjectData data;TrackData track;track.id="t";track.compose=true;track.chineseCvvc=cvvc;
        track.pitchAlgorithm=native?PitchAlgorithm::world:PitchAlgorithm::utau;
        if(cvvc||mode==2||mode==9){track.voicebankDirectory=folder.getChildFile("bank"+juce::String(mode));track.voicebankDirectory.createDirectory();}
        if(mode==2||mode==9)track.voicebankDirectory.getChildFile("dsconfig.yaml").replaceWithText("acoustic: fixture.onnx\n");
        const juce::String key=mode==9?"DS:ABS:TENC":mode==8?"HIFI:g":"g";
        NoteData n;n.id="a";n.label="a";n.startSeconds=.5;n.durationSeconds=2;n.midiNote=n.sourceMidiCenter=60;
        n.sourcePitchMeasured=true;n.nativePitchHandlesPlaced=true;n.contour={{0,0,0,true},{2,0,0,true}};
        n.pitchControlPoints={{0,60},{.4,63},{.8,65},{1.2,62},{2,60}};
        if(mode==2)n.diffSingerPitchReference=n.pitchControlPoints;
        n.amplitudeEnvelope={{0,-60,true},{.4,-6,true},{.8,-3,true},{1.2,-4,true},{2,-60,true}};
        n.utauFlagCurveEnabled=true;n.utauFlagCurves={{key,{{0,0},{.4,10},{.8,20},{1.2,30},{2,0}}}};
        if(mode==9)n.utauFlagCurves={{key,{{0,0},{.4,1},{.8,2},{1.2,3},{2,0}}},{"DS:PTS:TENC",{{0,0},{.4,1},{.8,2},{1.2,3},{2,0}}},{"DS:REF:TENC",{{0,.25},{2,.25}}}};
        if(cvvc)for(int part:{1,2,3})
        {n.cvvcEnvelopes[part][""]={{0,-60,true},{.2,-6,true},{.5,-3,true},{.8,-4,true},{1,-60,true}};
         n.cvvcEnvelopes[part][key]={{0,0,true},{.2,10,true},{.5,20,true},{.8,30,true},{1,0,true}};}
        auto other=n;other.id="b";other.startSeconds=3;
        ClipData clip;clip.id="c";clip.durationSeconds=clip.sourceDurationSeconds=6;clip.notes={n,other};track.clips={clip};data.tracks={track};
        ProjectModel model;model.resetDocument(data);I18n strings;PianoRollComponent roll(model,strings);
        roll.setSize(1700,1000);roll.setPixelsPerSecond(270);roll.setRowHeight(25);roll.setFocusedTrack("t");roll.setFocusedClip("c");
        const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};
        const auto reset=[&]{model.resetDocument(data);refresh();roll.setSelectedNoteIds({"a"});
            roll.setTool(pitch?PianoRollComponent::Tool::points:amp?PianoRollComponent::Tool::amplitude:PianoRollComponent::Tool::flagCurve);
            roll.setShowPitchLine(true);roll.setHifisamplerFlagContext(mode==8);
            if(mode==9){auto* c=new juce::DynamicObject();c->setProperty("key","DS:TENC");c->setProperty("supported",true);roll.setDiffSingerFlagContext(true,juce::Array<juce::var>{juce::var(c)});}
            roll.setFlagLaneFlag(key);roll.setFlagEditMode(PianoRollComponent::FlagEditMode::perNote);
            if(cvvc){auto waves=std::make_shared<std::vector<UtauNoteWaveform>>();for(const auto& note:clip.notes){UtauNoteWaveform w;
                w.noteId=note.id;w.startSeconds=note.startSeconds;w.audioHash=AudioEngine::utauNoteAudioHash(note);w.renderHash=AudioEngine::utauNoteRenderHash(note);
                w.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(note);w.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(track);
                w.phonemes={{"VC","vc",0,.4,1},{"CV","cv",.4,1.5,2},{"R","release",1.5,2,3}};waves->push_back(w);}roll.setUtauNoteWaveforms(waves);refresh();}};
        const auto at=[&](const juce::String& id,int index,int part=2){
            const auto* note=roll.findNote(id);const auto start=note->startSeconds;
            if(cvvc){const auto hs=roll.diagnosticCvvcEnvelopeHandles(id,part,amp);return hs.size()>size_t(index)?hs[size_t(index)]:juce::Point<float>(-100,-100);}
            if(pitch){const auto p=roll.pitchAnchorsFor(*note).at(size_t(index));return juce::Point<float>(roll.timeToX(start+p.timeSeconds),roll.diagnosticYForMidi(p.targetMidi));}
            if(amp){const auto ps=roll.amplitudeEnvelopeFor(*note,start);const auto target=note->amplitudeEnvelope.at(size_t(index)).timeSeconds;
                const auto found=std::min_element(ps.begin(),ps.end(),[&](auto& a,auto& b){return std::abs(a.timeSeconds-target)<std::abs(b.timeSeconds-target);});
                const auto p=*found;return juce::Point<float>(roll.timeToX(start+p.timeSeconds),roll.amplitudeLaneY(p.gainDb));}
            const auto p=roll.flagLaneCurveFor(*note).at(size_t(index));return juce::Point<float>(roll.timeToX(start+p.timeSeconds),roll.flagLaneY(p.value));};
        const auto event=[&](juce::Point<float> pos,juce::Point<float> down,bool ctrl=false){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),pos,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier|(ctrl?juce::ModifierKeys::ctrlModifier:0)),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,pos!=down);};
        const auto click=[&](juce::Point<float> p,bool ctrl=false){roll.mouseDown(event(p,p,ctrl));roll.mouseUp(event(p,p,ctrl));};
        const auto select=[&]{click(at("a",1));click(at("a",2),true);click(at("b",2,cvvc?3:2),true);};
        const auto stored=[&](int which,int part=2){std::vector<std::pair<double,double>> result;const auto snapshot=model.snapshot();const auto& note=snapshot.tracks[0].clips[0].notes[size_t(which)];
            if(cvvc)for(const auto& p:note.cvvcEnvelopes.at(part).at(amp?juce::String():key))result.emplace_back(p.position,amp?backend::envelopeGainFromDb(p.value):p.value);
            else if(pitch)for(const auto& p:note.pitchControlPoints)result.emplace_back(p.timeSeconds,p.targetMidi);
            else if(amp)for(const auto& p:note.amplitudeEnvelope)result.emplace_back(p.timeSeconds,backend::envelopeGainFromDb(p.gainDb));
            else for(const auto& p:mode==9?diffSingerParameterHandles(note,key):flagCurvePointsFor(note,key))result.emplace_back(p.timeSeconds,p.value);return result;};
        const auto del=[&]{return roll.keyPressed(juce::KeyPress(juce::KeyPress::deleteKey));};
        reset();const auto original=model.contentFingerprint();select();
        check(prefix+"ctrl_select_across_notes_and_phonemes",roll.selectedPoints.size()==3&&model.contentFingerprint()==original);
        click(at("a",2),true);check(prefix+"ctrl_toggle",roll.selectedPoints.size()==2);click(at("a",2),true);
        const auto beforeA=stored(0),beforeB=stored(1,cvvc?3:2);const auto from=at("a",1),to=from+juce::Point<float>(12,-12);
        roll.mouseDown(event(from,from));roll.mouseDrag(event(to,from));
        check(prefix+"preview_is_uncommitted",model.contentFingerprint()==original&&roll.pointGroupGesture&&roll.pointGroupGesture->changed);
        roll.mouseUp(event(to,from));refresh();const auto afterA=stored(0),afterB=stored(1,cvvc?3:2);
        const auto dt=afterA[1].first-beforeA[1].first,dv=afterA[1].second-beforeA[1].second;
        check(prefix+"common_time_delta",dt>0&&near(dt,afterA[2].first-beforeA[2].first)&&near(dt*(cvvc?1.1:1), (afterB[2].first-beforeB[2].first)*(cvvc?.5:1)));
        check(prefix+"common_display_value_delta",dv>0&&near(dv,afterA[2].second-beforeA[2].second)&&near(dv,afterB[2].second-beforeB[2].second));
        check(prefix+"unselected_points_unchanged",near(afterA[3].first,beforeA[3].first)&&near(afterA[3].second,beforeA[3].second)&&near(afterB[1].first,beforeB[1].first)&&near(afterB[1].second,beforeB[1].second));
        const auto moved=model.contentFingerprint();check(prefix+"one_undo_restores_all",model.undo()&&model.contentFingerprint()==original);refresh();
        check(prefix+"redo_restores_group",model.redo()&&model.contentFingerprint()==moved);refresh();
        reset();select();roll.keyPressed(juce::KeyPress('R',juce::ModifierKeys::ctrlModifier,0));
        const auto defaultA=stored(0),defaultB=stored(1,cvvc?3:2);const double value=pitch?60:amp?1:mode==9?.25:flagCurveKindFor(key).defaultValue;
        check(prefix+"reset_defaults_keeps_times",near(defaultA[1].second,value)&&near(defaultA[2].second,value)&&near(defaultB[2].second,value)&&near(defaultA[1].first,beforeA[1].first)&&near(defaultB[2].first,beforeB[2].first));
        check(prefix+"reset_one_undo",model.undo()&&model.contentFingerprint()==original);refresh();
        reset();select();del();refresh();check(prefix+"batch_delete",stored(0).size()==3&&stored(1,cvvc?3:2).size()==4);
        const auto deleted=model.contentFingerprint();del();del();check(prefix+"repeat_delete_preserves_notes",model.contentFingerprint()==deleted&&model.snapshot().tracks[0].clips[0].notes.size()==2);
        juce::String error;ProjectModel loaded;const auto file=folder.getChildFile(prefix+"deleted.hjpx");
        check(prefix+"save_reopen",model.save(file,error)&&loaded.load(file,error)&&loaded.contentFingerprint()==model.contentFingerprint());
        check(prefix+"delete_one_undo",model.undo()&&model.contentFingerprint()==original);refresh();
        reset();select();const auto held=at("a",1);roll.mouseDown(event(held,held));del();roll.mouseUp(event(held,held));
        check(prefix+"delete_during_drag_no_resurrection",stored(0).size()==3&&stored(1,cvvc?3:2).size()==4);
        reset();select();model.setNotePitchCurve("a",{{0,60},{2,60}},true);const auto stale=model.contentFingerprint();del();
        check(prefix+"stale_selection_safe",model.contentFingerprint()==stale);
        reset();select();click(at("a",3));del();check(prefix+"plain_click_replaces_group",stored(0).size()==4&&stored(1,cvvc?3:2).size()==5);
        reset();select();const auto far=at("a",1);roll.mouseDown(event(far,far));roll.mouseDrag(event(far+juce::Point<float>(400,-400),far));roll.mouseUp(event(far+juce::Point<float>(400,-400),far));
        const auto limitedA=stored(0),limitedB=stored(1,cvvc?3:2);
        if(mode==0)
        {
            const auto delta=400.0/270.0;
            check(prefix+"native_group_crosses_unselected_points",near(limitedA[2].first,beforeA[1].first+delta)
                &&near(limitedA[4].first,beforeA[2].first+delta)&&near(limitedB[4].first,beforeB[2].first+delta)
                &&near(limitedA[1].first,beforeA[3].first)&&near(limitedA[3].first,beforeA[4].first));
            refresh();roll.keyPressed(juce::KeyPress('R',juce::ModifierKeys::ctrlModifier,0));
            const auto resetA=stored(0),resetB=stored(1);
            check(prefix+"crossing_preserves_selected_point_identity",near(resetA[2].second,60)&&near(resetA[4].second,60)
                &&near(resetB[4].second,60)&&near(resetA[1].second,62)&&near(resetB[2].second,62));
        }
        else check(prefix+"group_stops_at_first_boundary",near(limitedA[1].first-beforeA[1].first,limitedA[2].first-beforeA[2].first)&&limitedA[2].first<limitedA[3].first
            &&near((limitedA[1].first-beforeA[1].first)*(cvvc?1.1:1),(limitedB[2].first-beforeB[2].first)*(cvvc?.5:1)));
        reset();select();const auto anchor=at("a",1),target=anchor+juce::Point<float>(8,-8);
        roll.mouseDown(event(anchor,anchor));roll.mouseDrag(event(target,anchor));roll.mouseUp(event(target,anchor));refresh();del();
        check(prefix+"drag_then_group_delete",stored(0).size()==3&&stored(1,cvvc?3:2).size()==4);
    }
    return ok;
}
}
