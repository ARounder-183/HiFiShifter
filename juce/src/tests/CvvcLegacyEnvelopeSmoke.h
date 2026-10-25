#pragma once
#include "../ProjectFileIO.h"

namespace hachi
{
template<class Check>
void verifyLegacyCvvcEnvelopeLoad(const juce::File& original, const juce::File& output,
    ProjectModel& model, Check&& check)
{
    juce::MemoryBlock bytes;original.loadFileAsData(bytes);
    auto tree=projectio::validatedTree(bytes);
    // The user may already have saved the migrated project. Keep legacy
    // compatibility probes independent of that evolving real-world input.
    if(tree.hasProperty("cvvcEnvelopeVersion"))
    {
        tree=tree.createCopy();tree.removeProperty("cvvcEnvelopeVersion",nullptr);
        int index=0;
        for(auto track:tree)for(auto clip:track)for(auto note:clip)if(note.hasType("Note"))
        {
            for(int i=note.getNumChildren()-1;i>=0;--i)if(note.getChild(i).hasType("AmplitudeEnvelopePoint"))note.removeChild(i,nullptr);
            const auto duration=static_cast<double>(note.getProperty("durationSeconds"));
            const auto attack=index++%2?.005:.015;
            for(const auto [t,g]:std::array<std::pair<double,double>,4>{{{-.08,-60},{-.08+attack,0},{duration-.035,0},{duration,-60}}})
            {juce::ValueTree p("AmplitudeEnvelopePoint");p.setProperty("timeSeconds",t,nullptr);p.setProperty("gainDb",g,nullptr);note.appendChild(p,nullptr);}
        }
    }
    check("legacy_fixture_has_no_envelope_version",tree.isValid()&&!tree.hasProperty("cvvcEnvelopeVersion"));
    const auto data=model.snapshot();int notes=0;bool defaults=true;
    for(const auto& track:data.tracks)if(track.chineseCvvc)for(const auto& clip:track.clips)for(const auto& n:clip.notes)
    {++notes;defaults&=n.amplitudeEnvelope.empty();}
    check("real_saved_legacy_defaults_become_independent_standard_phonemes",notes==3&&defaults);
    juce::String error;ProjectModel reopened;
    const auto saved=output.getChildFile("migrated-copy.hjpx");
    check("migrated_copy_save_and_reopen",model.save(saved,error)&&reopened.load(saved,error)
        &&model.contentFingerprint()==reopened.contentFingerprint());
    juce::MemoryBlock savedBytes;saved.loadFileAsData(savedBytes);
    check("saved_copy_marks_envelope_semantics",static_cast<int>(projectio::validatedTree(savedBytes).getProperty("cvvcEnvelopeVersion"))==1);
    const auto variant=[&](const char* name, auto edit, bool expectMigration)
    {
        auto copy=tree.createCopy();auto track=copy.getChild(0);auto note=track.getChild(0).getChildWithName("Note");
        edit(copy,track,note);
        juce::MemoryOutputStream encoded;copy.writeToStream(encoded);
        const auto file=output.getChildFile(juce::String(name)+".hjpx");
        ProjectModel test;const auto loaded=file.replaceWithData(encoded.getData(),encoded.getDataSize())&&test.load(file,error);
        check(name,loaded&&!test.snapshot().tracks.empty()
            &&test.snapshot().tracks[0].clips[0].notes[0].amplitudeEnvelope.empty()==expectMigration);
    };
    variant("modern_explicit_soft_curve_is_preserved",[](auto root,auto,auto){root.setProperty("cvvcEnvelopeVersion",1,nullptr);},false);
    variant("non_cvvc_legacy_curve_is_preserved",[](auto,auto track,auto){track.setProperty("utauPhonemizer","manual",nullptr);},false);
    variant("custom_legacy_level_is_preserved",[](auto,auto,auto note){
        int i=0;for(auto p:note)if(p.hasType("AmplitudeEnvelopePoint")&&i++==1)p.setProperty("gainDb",-3,nullptr);
    },false);
    variant("custom_legacy_timing_is_preserved",[](auto,auto,auto note){
        int i=0;for(auto p:note)if(p.hasType("AmplitudeEnvelopePoint")&&i++==1)
            p.setProperty("timeSeconds",static_cast<double>(p.getProperty("timeSeconds"))+.009,nullptr);
    },false);
    variant("explicit_legacy_linear_curve_is_preserved",[](auto,auto,auto note){
        for(auto p:note)if(p.hasType("AmplitudeEnvelopePoint"))p.setProperty("linearToNext",true,nullptr);
    },false);
    variant("advanced_legacy_head_is_preserved",[](auto,auto,auto note){note.setProperty("utauHeadEnvelopeMode",1,nullptr);},false);
}

template<class Check>
void verifyRealCvvcEnvelopeDisplay(ProjectModel& model,const TrackData& track,const ClipData& clip,
    const std::vector<std::vector<backend::UtauPhonemeSpan>>& spans,
    const juce::File& output,const juce::String& engine,Check&& check)
{
    using namespace backend;
    I18n labels;PianoRollComponent roll(model,labels);juce::Viewport viewport;
    viewport.setSize(1400,780);viewport.setViewedComponent(&roll,false);
    roll.setSize(1400,780);roll.setPixelsPerSecond(5000);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
    auto metadata=std::make_shared<std::vector<UtauNoteWaveform>>();
    for(size_t i=0;i<clip.notes.size();++i)
    {
        const auto& n=clip.notes[i];UtauNoteWaveform w;w.noteId=n.id;w.phonemes=spans[i];w.startSeconds=n.startSeconds;
        w.audioHash=AudioEngine::utauNoteAudioHash(n);w.renderHash=AudioEngine::utauNoteRenderHash(n);
        w.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(n);w.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(track);
        for(const auto& p:w.phonemes)w.leadInSeconds=std::max(w.leadInSeconds,p.leadInSeconds-p.startSeconds);
        metadata->push_back(w);
    }
    roll.setUtauNoteWaveforms(metadata);roll.setTool(PianoRollComponent::Tool::amplitude);roll.setShowPhonemeInfo(true);
    model.dispatchPendingMessages();roll.diagnosticRefresh();
    for(const bool tail:{false,true})
    {
        const auto owner=tail?clip.notes.size()-1:0;const auto& n=clip.notes[owner];
        const auto& piece=tail?spans[owner].back():spans[owner].front();const auto& env=piece.mixEnvelope;
        const auto ramp=std::find_if(env.ramps.begin(),env.ramps.end(),[&](const auto& r){return r.rising!=tail&&!r.crossfade;});
        const auto name=engine+(tail?"_phrase_tail":"_phrase_head");
        check(name+"_has_single_standard_edge",env.replacesStandardEnvelope&&ramp!=env.ramps.end());
        if(ramp==env.ramps.end())continue;
        const auto edge=n.startSeconds+clip.startSeconds+env.startSeconds+ramp->start/env.sampleRate;
        viewport.setViewPosition(std::max(0,static_cast<int>(roll.diagnosticEdgeX(edge))-120),
            static_cast<int>(roll.diagnosticYForMidi(n.midiNote)-170));
        const auto img=roll.createComponentSnapshot(viewport.getViewArea());
        juce::PNGImageFormat png;auto file=output.getChildFile(name+".png").createOutputStream();if(file)png.writeImageToStream(img,*file);
        bool straight=true;int pixels=0;
        for(double u:{.1,.25,.5,.75,.9})
        {
            const auto time=edge+u*ramp->length/env.sampleRate;const auto gain=static_cast<float>(tail?1-u:u);
            const auto x=static_cast<int>(roll.diagnosticEdgeX(time))-viewport.getViewPositionX();
            const auto y=static_cast<int>(roll.diagnosticAmplitudeLaneY(envelopeDbFromGain(gain)))-viewport.getViewPositionY();
            bool colour=false;
            for(int dy=-3;dy<=3;++dy)for(int dx=-3;dx<=3;++dx)if(img.getBounds().contains(x+dx,y+dy))
            {const auto c=img.getPixelAt(x+dx,y+dy);colour|=tail?(c.getBlue()>120&&c.getRed()>90&&c.getBlue()>c.getGreen()*1.05f)
                :(c.getRed()>160&&c.getGreen()>90&&c.getBlue()<c.getGreen()*.9f);}
            straight&=colour;pixels+=colour;
        }
        std::cout<<name<<"_straight_pixels="<<pixels<<std::endl;
        check(name+"_paints_straight_amplitude_ramp",straight);
    }
}
}
