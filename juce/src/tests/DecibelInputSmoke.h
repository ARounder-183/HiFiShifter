#pragma once
#include "../DecibelGainDialog.h"
#include "../PanValueDialog.h"
namespace hachi
{
inline bool MainComponent::diagnosticDecibelInput(const juce::File& folder)
{
    folder.createDirectory();stopTimer();setSize(1280,850);
    bool ok=true;int checks=0;
    const auto check=[&](const char* key,bool pass){++checks;ok&=pass;std::cout<<key<<'='<<pass<<std::endl;};
    const auto near=[](float a,float b){return std::abs(a-b)<1.e-7f;};
    for(const auto* text:{"","abc","NaN","inf","1.2.3","3oops","-121","1000"})
        check("invalid_rejected",!DecibelGainDialog::parse(text,2));
    check("decimal_exact",near(*DecibelGainDialog::parse(" -6.125 dB ",2),std::pow(10.0f,-6.125f/20)));
    check("silence_explicit",DecibelGainDialog::parse("-inf",2)==0&&DecibelGainDialog::parse(juce::String::fromUTF8("-∞"),2)==0);
    check("minus60_is_not_silence",near(*DecibelGainDialog::parse("-60",2),.001f));
    check("different_track_clip_limits",!DecibelGainDialog::parse("9",2)&&DecibelGainDialog::parse("9",4).has_value());
    ProjectData data;
    for(int i=0;i<2;++i)
    {
        TrackData track;track.id="t"+juce::String(i);track.accompaniment=true;track.compose=false;
        ClipData clip;clip.id="c"+juce::String(i);clip.startSeconds=.5;clip.durationSeconds=3;track.clips={clip};data.tracks.push_back(track);
    }
    project.resetDocument(data);project.dispatchPendingMessages();stopTimer();
    timeline.setPixelsPerSecond(140);timeline.setRowHeight(96);trackList.setRowHeight(96);
    (void)timeline.createComponentSnapshot(timeline.getLocalBounds());
    const auto event=[](juce::Component& component,juce::Point<float> p,juce::Point<float> down)
    {return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),p,juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier),
        0,0,0,0,0,&component,&component,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,p!=down);};
    juce::String trackTarget,clipTarget;
    const auto oldTrack=trackList.onTrackGainRequested,oldClip=timeline.onClipGainRequested;
    trackList.onTrackGainRequested=[&](const auto& id){trackTarget=id;};
    timeline.onClipGainRequested=[&](const auto& id){clipTarget=id;};
    auto point=juce::Point<float>(195,24+61+8);
    const auto revision=project.revisionNumber();
    trackList.mouseDown(event(trackList,point,point));trackList.mouseUp(event(trackList,point,point));
    check("track_readout_single_click",trackTarget=="t0"&&project.revisionNumber()==revision);
    // At 140 px/sec, the region begins at x=70; numeric gain is after its
    // ascending bars and knob. Use the painted bottom readout, not the body.
    point={125,105};
    timeline.mouseDown(event(timeline,point,point));timeline.mouseUp(event(timeline,point,point));
    check("clip_readout_single_click",clipTarget=="c0"&&project.revisionNumber()==revision);
    clipTarget.clear();
    timeline.mouseDown(event(timeline,point,point));timeline.mouseDrag(event(timeline,point.translated(0,-20),point));
    timeline.mouseUp(event(timeline,point.translated(0,-20),point));
    check("numeric_drag_preserved_without_dialog",clipTarget.isEmpty()&&project.snapshot().tracks[0].clips[0].gain>1);
    trackList.onTrackGainRequested=oldTrack;timeline.onClipGainRequested=oldClip;
    DecibelGainDialog dialog(strings,"dB",1,2,this);
    auto* editor=dialog.getTextEditor("gain");editor->setText("abc",false);editor->onTextChange();
    check("invalid_disables_apply",!dialog.getButton(0)->isEnabled());
    editor->setText("-6.125",false);editor->onTextChange();
    check("valid_enables_apply",dialog.getButton(0)->isEnabled()&&dialog.value().has_value());
    const auto before=project.revisionNumber();
    project.setTrackVolume(trackTarget,*dialog.value());project.dispatchPendingMessages();stopTimer();
    check("exact_track_only",near(project.snapshot().tracks[0].volume,std::pow(10.0f,-6.125f/20))&&project.snapshot().tracks[1].volume==1);
    check("one_undo",project.revisionNumber()==before+1&&project.undo()&&project.snapshot().tracks[0].volume==1);
    check("redo",project.redo()&&near(project.snapshot().tracks[0].volume,*dialog.value()));
    const auto quiet=*DecibelGainDialog::parse("-100.125",4);
    project.setClipGain("c0",quiet);project.setClipGain("c0",*DecibelGainDialog::parse("-100.126",4));
    check("quiet_fractional_db_not_ignored",project.snapshot().tracks[0].clips[0].gain!=quiet);
    DecibelGainDialog unchanged(strings,"dB",quiet,4,this);check("unchanged_input_lossless",unchanged.value()==quiet);
    // Pan shares the numeric-readout interaction but uses a linear bipolar range.
    juce::String panTarget;
    const auto oldPan=trackList.onTrackPanRequested;
    trackList.onTrackPanRequested=[&](const auto& id){panTarget=id;};
    point={95,24+96+85};
    const auto panRevision=project.revisionNumber();
    trackList.mouseDown(event(trackList,point,point));trackList.mouseUp(event(trackList,point,point));
    check("pan_second_track_click_no_mutation",panTarget=="t1"&&project.revisionNumber()==panRevision);
    trackList.onTrackPanRequested=oldPan;
    PanValueDialog panDialog(strings,strings.text("track.pan"),0,this);
    auto* panEditor=panDialog.getTextEditor("pan");
    for(const auto* input:{"", "NaN", "inf", "-1.01", "1.01", "0.5oops"})
    {
        panEditor->setText(input,false);panEditor->onTextChange();
        check("pan_invalid_disables_apply",!panDialog.value()&&!panDialog.getButton(0)->isEnabled());
    }
    panEditor->setText("-0.125",false);panEditor->onTextChange();
    check("pan_decimal_enabled",panDialog.value()==-.125f&&panDialog.getButton(0)->isEnabled());
    project.setTrackPan(panTarget,*panDialog.value());
    check("pan_only_target_track",project.snapshot().tracks[0].pan==0&&project.snapshot().tracks[1].pan==-.125f);
    check("pan_undo_redo",project.undo()&&project.snapshot().tracks[1].pan==0
        &&project.redo()&&project.snapshot().tracks[1].pan==-.125f);
    check("pan_endpoints",PanValueDialog::parse("-1")==-1&&PanValueDialog::parse("+1")==1&&PanValueDialog::parse("0")==0);
    PanValueDialog panUnchanged(strings,"Pan",.123456789f,this);
    check("pan_unchanged_lossless",panUnchanged.value()==.123456789f);
    project.dispatchPendingMessages();stopTimer();
    // The original rail still drags without opening the numeric editor.
    panTarget.clear();trackList.onTrackPanRequested=[&](const auto& id){panTarget=id;};
    point={180,24+85};const auto panEnd=juce::Point<float>(trackList.getWidth()-40.0f,24+85.0f);
    trackList.mouseDown(event(trackList,point,point));trackList.mouseDrag(event(trackList,panEnd,point));
    trackList.mouseUp(event(trackList,panEnd,point));
    check("pan_slider_preserved",panTarget.isEmpty()&&project.snapshot().tracks[0].pan>.9f
        &&project.snapshot().tracks[1].pan==-.125f);
    trackList.onTrackPanRequested=oldPan;
    auto panStream=folder.getChildFile("pan-dialog.png").createOutputStream();
    juce::PNGImageFormat().writeImageToStream(panDialog.createComponentSnapshot(panDialog.getLocalBounds()),*panStream);
    juce::String error;const auto file=folder.getChildFile("gain.hjpx");ProjectModel reopened;
    check("saved_reopened",project.save(file,error)&&reopened.load(file,error)
        &&reopened.snapshot().tracks[0].volume==project.snapshot().tracks[0].volume
        &&reopened.snapshot().tracks[0].clips[0].gain==project.snapshot().tracks[0].clips[0].gain
        &&reopened.snapshot().tracks[0].pan==project.snapshot().tracks[0].pan
        &&reopened.snapshot().tracks[1].pan==-.125f);
    auto stream=folder.getChildFile("dialog.png").createOutputStream();
    juce::PNGImageFormat().writeImageToStream(dialog.createComponentSnapshot(dialog.getLocalBounds()),*stream);
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(checks)+",\"ok\":"+(ok?"true":"false")+"}");
    return ok;
}
}
