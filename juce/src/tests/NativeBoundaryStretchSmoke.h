#pragma once
#include "../NativeAudioLink.h"
#include "../NativeNoteTiming.h"
namespace hachi
{
inline bool runNativeBoundaryStretchSmoke(const juce::File& folder)
{
    folder.createDirectory();bool ok=true;int count=0;
    const auto check=[&](const char* name,bool pass){ok&=pass;++count;std::cout<<name<<'='<<pass<<std::endl;};
    const auto near=[](double a,double b){return std::abs(a-b)<2.e-6;};
    const auto file=folder.getChildFile("source.wav");juce::WavAudioFormat wav;auto stream=file.createOutputStream();
    if(!stream)return false;stream->setPosition(0);stream->truncate();
    std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(),48000,1,24,{},0));if(!writer)return false;
    juce::AudioBuffer<float> pcm(1,76800);
    for(int i=0;i<76800;++i)pcm.setSample(0,i,float(.1*(1+i/19200)*std::sin(i*220*juce::MathConstants<double>::twoPi/48000)));
    writer->writeFromAudioSampleBuffer(pcm,0,76800);writer.reset();const auto hash=juce::SHA256(file).toHexString();
    const auto fixture=[&](int mode){
        ProjectData d;TrackData t;t.id="native";t.trackType=TrackType::hachi;t.pitchAlgorithm=PitchAlgorithm::mld5;
        ClipData whole;whole.id="whole";whole.sourceFile=file;whole.startSeconds=1;whole.durationSeconds=whole.sourceDurationSeconds=1.6;
        std::vector<ClipData> parts;
        for(int i=0;i<4;++i){auto c=whole;c.id="part"+juce::String(i);c.startSeconds=1+i*.4;c.sourceOffsetSeconds=i*.4;c.durationSeconds=c.sourceDurationSeconds=.4;
            NoteData n;n.id="n"+juce::String(i);n.durationSeconds=.4;n.midiNote=n.sourceMidiCenter=57;n.sourcePitchMeasured=true;n.utauAutoPitchTransition=false;
            for(int f=0;f<=80;++f)n.contour.push_back({f*.005,0,0,true});n.pitchControlPoints={{0,57},{.2,57},{.4,57}};
            bindNativeNoteSource(n,c);c.notes={n};parts.push_back(c);n.startSeconds=i*.4;whole.notes.push_back(n);}
        t.clips=mode==0?std::vector<ClipData>{whole}:mode==1?std::vector<ClipData>{assembledLinkedAudio(parts)}:parts;
        d.tracks={t};return d;
    };
    const auto span=[](const ProjectData& d,const juce::String& id){for(const auto& t:d.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)
        if(n.id==id)return juce::Range<double>(c.startSeconds+n.startSeconds,c.startSeconds+n.startSeconds+n.durationSeconds);return juce::Range<double>{};};
    ProjectModel model;I18n strings;PianoRollComponent roll(model,strings);roll.setSize(1700,1000);roll.setPixelsPerSecond(500);roll.setRowHeight(25);
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};
    const auto install=[&](const ProjectData& d){model.resetDocument(d);refresh();roll.setFocusedTrack("native");roll.setFocusedClip(d.tracks[0].clips[0].id);roll.setTool(PianoRollComponent::Tool::note);refresh();};
    const auto event=[&](juce::Point<float> p,juce::Point<float> down){return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),p,
        juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier|juce::ModifierKeys::altModifier),0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),down,juce::Time::getCurrentTime(),1,p!=down);};
    for(int mode=0;mode<3;++mode)for(bool fromRight:{false,true})for(double delta:{-.2,.2})
    {
        const auto original=fixture(mode);install(original);
        const auto a=roll.diagnosticHitBounds(1),b=roll.diagnosticHitBounds(2);
        const juce::Point<float> boxStart(a.getX()+3,a.getY()-15),boxEnd(b.getRight()-3,b.getBottom()+15);
        roll.mouseDown(event(boxStart,boxStart));roll.mouseDrag(event(boxEnd,boxStart));roll.mouseUp(event(boxEnd,boxStart));
        check("marquee_selects_only_adjacent_pair",roll.selectedNoteIds().size()==2);
        const auto before=model.contentFingerprint();const auto revision=model.revisionNumber();
        const juce::Point<float> down(fromRight?b.getX()+1:a.getRight()-1,fromRight?b.getCentreY():a.getCentreY());const auto up=down.translated(float(delta*500),0);
        roll.mouseDown(event(down,down));roll.mouseDrag(event(up,down));
        auto image=roll.createComponentSnapshot({550,int(a.getY())-20,800,90});
        if(mode==0&&!fromRight&&delta>0)if(auto out=folder.getChildFile("shared-boundary-preview.png").createOutputStream())juce::PNGImageFormat{}.writeImageToStream(image,*out);
        check("preview_does_not_commit",model.contentFingerprint()==before);
        roll.mouseUp(event(up,down));refresh();const auto changed=model.snapshot();
        const auto x=span(changed,"n1"),y=span(changed,"n2");
        check("drag_changes_common_boundary_both_directions",near(x.getEnd(),1.8+delta)&&near(y.getStart(),1.8+delta));
        check("pair_outer_edges_and_total_length_fixed",near(x.getStart(),1.4)&&near(y.getEnd(),2.2)&&near(x.getLength()+y.getLength(),.8));
        check("unselected_neighbours_keep_positions",near(span(changed,"n0").getStart(),1)&&near(span(changed,"n0").getEnd(),1.4)
            &&near(span(changed,"n3").getStart(),2.2)&&near(span(changed,"n3").getEnd(),2.6));
        bool clocks=true,points=true;
        for(const auto& parent:changed.tracks[0].clips)for(const auto& c:expandedClipParts(parent))for(const auto& n:c.notes)if(n.id=="n1"||n.id=="n2")
        {
            const auto clock=nativeClipClock(c);const auto sourceStart=c.sourceOffsetSeconds+nativeSourceTimeAt(clock,n.startSeconds);
            const auto sourceEnd=c.sourceOffsetSeconds+nativeSourceTimeAt(clock,n.startSeconds+n.durationSeconds);
            clocks &= near(sourceStart,n.id=="n1"?.4:.8)&&near(sourceEnd,n.id=="n1"?.8:1.2);
            points &= n.pitchControlPoints.size()==3&&near(n.pitchControlPoints[1].timeSeconds,n.durationSeconds*.5);
        }
        check("each_note_stretches_its_complete_original_audio",clocks);check("pitch_points_follow_stretched_clock",points);
        check("selection_survives_drag",roll.selectedNoteIds().size()==2);
        check("one_undo_restores_both_notes",model.revisionNumber()==revision+1&&model.undo()&&model.contentFingerprint()==before);
        check("redo_restores_boundary",model.redo()&&near(span(model.snapshot(),"n1").getEnd(),1.8+delta));
        juce::String error;ProjectModel reopened;check("save_reopen_preserves_pair",model.save(folder.getChildFile("pair.hjpx"),error)&&reopened.load(folder.getChildFile("pair.hjpx"),error)
            &&near(span(reopened.snapshot(),"n1").getEnd(),1.8+delta)&&near(span(reopened.snapshot(),"n2").getEnd(),2.2));
    }
    const auto original=fixture(0);install(original);roll.setSelectedNoteIds({"n1","n2"});
    const auto b=roll.diagnosticHitBounds(1);const juce::Point<float> down(b.getRight()-1,b.getCentreY()),up=down.translated(100,0);
    const auto before=model.contentFingerprint();roll.mouseDown(event(down,down));roll.mouseDrag(event(up,down));roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));roll.mouseUp(event(up,down));
    check("escape_cancels_entire_pair",before==model.contentFingerprint());
    check("stale_gesture_rejected",!model.stretchNativeNoteBoundary("n1","n2",.2,model.revisionNumber()-1)&&before==model.contentFingerprint());
    check("nonadjacent_selection_rejected",!model.stretchNativeNoteBoundary("n0","n2",.2,model.revisionNumber()));
    auto overlap=original;auto n=overlap.tracks[0].clips[0].notes[1];n.id="overlap";n.startSeconds+=.1;overlap.tracks[0].clips[0].notes.push_back(n);install(overlap);
    check("unselected_overlapping_voice_protected",!model.stretchNativeNoteBoundary("n1","n2",.2,model.revisionNumber()));
    for(double delta:{-10.,10.}){install(original);model.stretchNativeNoteBoundary("n1","n2",delta,model.revisionNumber());const auto d=model.snapshot();
        check("extreme_drag_keeps_both_notes_positive_and_total_fixed",span(d,"n1").getLength()>=.01-1.e-7&&span(d,"n2").getLength()>=.01-1.e-7
            &&near(span(d,"n1").getStart(),1.4)&&near(span(d,"n2").getEnd(),2.2));}
    install(original);model.stretchNativeNoteBoundary("n1","n2",.2,model.revisionNumber());const auto stretched=model.snapshot();
    const auto render=[&](const ProjectData& d,const char* name){AudioEngine engine(false);engine.prepareToPlay(256,48000);engine.syncProject(d);
        for(int i=0;i<2000&&engine.renderProgress();++i)juce::Thread::sleep(10);
        WavExportOptions options;options.sampleRate=48000;options.channels=1;options.bitDepth=32;juce::String error;
        const auto output=folder.getChildFile(name);check(name,engine.hasCurrentRenderedAudio()&&engine.exportWav(output,error,"native",1,2.6,options));
        juce::AudioFormatManager formats;formats.registerBasicFormats();std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(output));
        juce::AudioBuffer<float> samples(1,76800);samples.clear();if(reader)reader->read(&samples,0,76800,0,true,false);return samples;};
    const auto audioBefore=render(original,"before.wav"),audioAfter=render(stretched,"after.wav");
    const auto rms=[](const auto& audio,double t){return audio.getRMSLevel(0,int(t*48000),2400);};
    check("actual_audio_boundary_follows_new_seam",rms(audioBefore,.86)>.05&&std::abs(rms(audioAfter,.86)/rms(audioBefore,.86)-2.f/3.f)<.13);
    check("other_audio_and_pair_tail_still_sound",std::abs(rms(audioAfter,.2)-rms(audioBefore,.2))<.01&&rms(audioAfter,1.1)>.08&&rms(audioAfter,1.4)>.1);
    check("original_audio_file_unchanged",hash==juce::SHA256(file).toHexString());
    folder.getChildFile("result.json").replaceWithText("{\"checks\":"+juce::String(count)+",\"ok\":"+(ok?"true":"false")+"}");return ok;
}
}
