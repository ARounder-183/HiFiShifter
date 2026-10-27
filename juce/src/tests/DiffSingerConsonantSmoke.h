#pragma once
#include "../AudioEngine.h"
#include "../PianoRollComponent.h"
#include "../backend/DiffSingerRenderer.h"
#include <iostream>

namespace hachi
{
inline bool runDiffSingerConsonantSmoke(const juce::File& input, const juce::File& folder)
{
    folder.createDirectory(); bool ok=true; juce::String error;
    const auto check=[&](const char* name,bool pass){ok &= pass;std::cout<<name<<'='<<pass<<std::endl;};
    ProjectModel model; if(!model.load(input,error)) {std::cout<<error<<std::endl;return false;}
    const auto original=model.snapshot();
    if(original.tracks.empty()) return false;
    const auto renderWait=[](AudioEngine& engine,const ProjectData& data)
    {
        engine.selectEveryUtauNote(data);engine.syncProject(data);
        for(int i=0;i<2000 && !engine.hasCurrentRenderedAudio();++i)
        {if(i>40 && !engine.renderProgress()) break;juce::Thread::sleep(25);}
        engine.refreshUtauWaveforms();return engine.hasCurrentRenderedAudio();
    };
    AudioEngine engine;
    check("user_project_render",renderWait(engine,original));
    check("user_project_export",engine.exportWav(folder.getChildFile("user-project-fixed.wav"),error));
    auto data=original;data.tracks.resize(1);auto& track=data.tracks[0];
    track.utauMode=UtauMode::mou;track.muted=false;track.volume=1;track.clips.clear();
    ClipData clip;clip.id="test-clip";clip.durationSeconds=4;
    for(int i=0;i<2;++i)
    {
        NoteData n;n.id=juce::String(i);n.label=i?"wang":"zhang";
        n.startSeconds=1.+i;n.durationSeconds=1.;n.midiNote=60+(float)i*2;n.utauAutoPitchTransition=false;
        n.amplitudeEnvelope={{0,-60},{.015,0},{.965,0},{1,-60}};
        clip.notes.push_back(n);
    }
    track.clips.push_back(clip);model.replace(data);
    backend::UtauRenderRequest request;request.voicebankDirectory=track.voicebankDirectory;request.targetDurationSeconds=4;
    request.notes=AudioEngine::diagnosticUtauRequestNotes(data,clip.id);
    std::vector<std::vector<backend::UtauPhonemeSpan>> spans(2);
    struct Piece{juce::AudioBuffer<float> raw;double lead=0;std::function<float(double)> gain;};
    std::vector<Piece> pieces(2);
    request.notePhonemes=[&](std::size_t i,const auto& p){spans[i]=p;};
    request.notePiece=[&](std::size_t i,const auto& raw,double,double lead,const auto& gain,const auto&)
    {pieces[i].raw=raw;pieces[i].lead=lead;pieces[i].gain=gain;};
    const auto full=backend::DiffSingerRenderer::render(request);
    check("real_zhang_wang_render",full.warning.isEmpty() && full.buffer.getNumSamples()>0);
    check("zhang_has_zh_before_beat",spans[0].size()==2 && spans[0][0].token=="zh/zh"
        && spans[0][0].startSeconds<-.02 && std::abs(spans[0][1].startSeconds)<.012);
    check("wang_has_w_before_beat",spans[1].size()==2 && spans[1][0].token=="zh/w"
        && spans[1][0].startSeconds<-.02 && std::abs(spans[1][1].startSeconds)<.012);
    if(spans[0].empty() || spans[1].empty()) return false;
    check("different_consonants_not_padding_limit",pieces[0].lead>.02 && pieces[0].lead<.45);
    const auto first=std::max(0,(int)std::llround((2+spans[1][0].startSeconds)*full.sampleRate));
    const auto last=(int)std::llround(2*full.sampleRate);
    const auto energy=full.buffer.getMagnitude(0,first,last-first);
    check("audible_consonant_before_vowel",energy>.002f);
    request.notes[0].gain=0;
    const auto mutedPrevious=backend::DiffSingerRenderer::render(request);
    float difference=0;
    for(int s=first;s<last;++s) difference=std::max(difference,std::abs(full.buffer.getSample(0,s)-mutedPrevious.buffer.getSample(0,s)));
    check("previous_note_mute_does_not_erase_next_consonant",difference<1e-7f);
    const auto local=(int)std::llround((pieces[1].lead-.025)*full.sampleRate);
    check("raw_waveform_and_single_envelope",local>0 && local<pieces[1].raw.getNumSamples()
        && std::abs(pieces[1].raw.getSample(0,local)*pieces[1].gain(-.025)
            -mutedPrevious.buffer.getSample(0,last-(int)std::llround(.025*full.sampleRate)))<.002f);
    check("fixture_render",renderWait(engine,model.snapshot()));
    check("fixture_export",engine.exportWav(folder.getChildFile("zhang-wang-fixed.wav"),error));
    check("fixture_save",model.save(folder.getChildFile("zhang-wang-test.hjpx"),error));
    I18n strings;PianoRollComponent roll(model,strings);roll.setFocusedTrack(track.id);
    roll.diagnosticRefresh();roll.setPixelsPerSecond(260);roll.setRowHeight(30);
    roll.setShowUtauWaveforms(true);roll.setShowNoteRange(true);roll.setUtauNoteWaveforms(engine.utauNoteWaveforms());
    const auto range=roll.diagnosticSoundingSpan("1");const auto wave=roll.diagnosticWaveformSpan("1");
    check("roll_consonant_span",range.first<1.98 && std::abs(range.first-(2+spans[1][0].startSeconds))<.001);
    check("waveform_matches_phoneme_span",wave && std::abs(wave->first-range.first)<.001);
    const auto envelope=roll.diagnosticDrawnEnvelope("1");
    check("envelope_starts_at_consonant",envelope.size()==4
        && std::abs(envelope.front().timeSeconds-spans[1][0].startSeconds)<.001
        && std::abs(envelope[1].timeSeconds-envelope[0].timeSeconds-.015)<1e-7);
    if(auto stream=folder.getChildFile("main-roll-consonants.png").createOutputStream())
    {
        const auto area=roll.diagnosticHitBounds(0).getUnion(roll.diagnosticHitBounds(1))
            .expanded(90,70).getSmallestIntegerContainer().getIntersection(roll.getLocalBounds());
        juce::PNGImageFormat().writeImageToStream(roll.createComponentSnapshot(area),*stream);
    }
    // Selecting just wang must retain zhang as duration-model context.
    const auto selected=AudioEngine::diagnosticUtauRequestNotes(data,clip.id,{"1"});
    check("selection_keeps_context",selected.size()==2 && selected[0].gain==0 && selected[1].gain==1);
    // A clip moved to a later timeline position can sound before its own start.
    auto moved=data;moved.tracks[0].clips[0].startSeconds=3;
    for(auto& n:moved.tracks[0].clips[0].notes) n.startSeconds-=1;
    check("clip_lead_in_render",renderWait(engine,moved));
    engine.refreshUtauWaveforms();const auto waves=engine.utauNoteWaveforms();
    check("clip_lead_in_preserved",waves && !waves->empty() && waves->front().leadInSeconds>.02);
    check("clip_lead_in_export",engine.exportWav(folder.getChildFile("clip-start-consonant.wav"),error));
    // Loudness is post-synthesis gain: editing it must not regenerate a
    // different performance, change phoneme timing, or bake the gain twice.
    auto neutral=data;
    for(auto& n:neutral.tracks[0].clips[0].notes)n.amplitudeEnvelope.clear();
    model.resetDocument(neutral);
    request.notes=AudioEngine::diagnosticUtauRequestNotes(model.snapshot(),clip.id);
    const auto dry=backend::DiffSingerRenderer::render(request);
    const auto drySpans=spans;
    check("DS_neutral_loudness_render",dry.warning.isEmpty()&&dry.buffer.getNumSamples()>0);
    check("DS_neutral_export",renderWait(engine,model.snapshot())&&engine.exportWav(folder.getChildFile("ds-envelope-unity.wav"),error));
    const auto dryKey=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),clip.id);
    std::vector<std::pair<juce::String,std::vector<AmplitudeEnvelopePoint>>> halves;
    for(const auto& n:neutral.tracks[0].clips[0].notes)halves.push_back({n.id,{{0,-6.0206f,true},{1,-6.0206f,true}}});
    check("DS_half_gain_committed",model.setNotesAmplitudeEnvelopes(halves));
    check("DS_envelope_invalidates_playback_cache",dryKey!=AudioEngine::diagnosticUtauRenderKey(model.snapshot(),clip.id));
    check("DS_half_gain_export",renderWait(engine,model.snapshot())&&engine.exportWav(folder.getChildFile("ds-envelope-half.wav"),error));
    juce::AudioFormatManager formats;formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> dryReader(formats.createReaderFor(folder.getChildFile("ds-envelope-unity.wav")));
    std::unique_ptr<juce::AudioFormatReader> halfReader(formats.createReaderFor(folder.getChildFile("ds-envelope-half.wav")));
    bool scaled=dryReader&&halfReader&&dryReader->lengthInSamples==halfReader->lengthInSamples&&dryReader->sampleRate==halfReader->sampleRate;
    if(scaled)
    {
        const auto count=(int)dryReader->lengthInSamples;juce::AudioBuffer<float> a(2,count),b(2,count);
        dryReader->read(&a,0,count,0,true,true);halfReader->read(&b,0,count,0,true,true);
        double energy=0,residual=0;
        for(int ch=0;ch<2;++ch)for(int s=0;s<count;++s)
        {const auto expected=.5*a.getSample(ch,s),delta=b.getSample(ch,s)-expected;energy+=expected*expected;residual+=delta*delta;}
        scaled=energy>1e-5&&residual/energy<1e-6;
    }
    check("DS_export_has_exactly_one_half_gain",scaled);
    check("DS_envelope_undo_restores_render_key",model.undo()&&AudioEngine::diagnosticUtauRenderKey(model.snapshot(),clip.id)==dryKey);
    check("DS_undo_export",renderWait(engine,model.snapshot())&&engine.exportWav(folder.getChildFile("ds-envelope-undo.wav"),error));
    juce::MemoryBlock dryBytes,undoBytes;folder.getChildFile("ds-envelope-unity.wav").loadFileAsData(dryBytes);
    folder.getChildFile("ds-envelope-undo.wav").loadFileAsData(undoBytes);
    check("DS_undo_restores_identical_audio",dryBytes==undoBytes);
    // Verify the visible curve against the real rendered PCM, including the
    // consonant before the MIDI beat, while switching straight/curved/straight.
    for(bool linear:{true,false,true})
    {
        model.setNotesAmplitudeEnvelopes({{"0",{{0,-60,linear},{.25,0,linear},{.7,0,linear},{1,-60,true}}},
                                         {"1",{{0,-60,linear},{.25,0,linear},{.7,0,linear},{1,-60,true}}}});
        request.notes=AudioEngine::diagnosticUtauRequestNotes(model.snapshot(),clip.id);
        const auto shaped=backend::DiffSingerRenderer::render(request);
        bool matches=shaped.warning.isEmpty()&&shaped.buffer.getNumSamples()==dry.buffer.getNumSamples();
        check("DS_shaped_playback_ready",renderWait(engine,model.snapshot()));
        engine.refreshUtauWaveforms();roll.diagnosticRefresh();roll.setTool(PianoRollComponent::Tool::amplitude);
        roll.setUtauNoteWaveforms(engine.utauNoteWaveforms());
        double residual=0,energy=0;
        for(std::size_t i=0;i<spans.size();++i)
        {
            matches=matches&&spans[i].size()==drySpans[i].size();
            if(spans[i].empty()) {matches=false;continue;}
            for(std::size_t j=0;j<spans[i].size()&&j<drySpans[i].size();++j)
                matches=matches&&spans[i][j].startSeconds==drySpans[i][j].startSeconds&&spans[i][j].endSeconds==drySpans[i][j].endSeconds;
            const auto drawn=roll.diagnosticDrawnEnvelope(juce::String((int)i));
            const auto onset=request.notes[i].startSeconds+spans[i].front().startSeconds;
            const auto end=request.notes[i].startSeconds+spans[i].back().endSeconds;
            for(int s=(int)std::ceil(onset*dry.sampleRate)+2;s<std::min((int)(end*dry.sampleRate)-2,shaped.buffer.getNumSamples());++s)
            {
                const auto gain=backend::envelopeGainFromDb(PianoRollComponent::diagnosticAmplitudeDbAt(drawn,s/dry.sampleRate-request.notes[i].startSeconds));
                const auto expected=dry.buffer.getSample(0,s)*gain,delta=shaped.buffer.getSample(0,s)-expected;
                energy+=expected*expected;residual+=delta*delta;
            }
        }
        check(linear?"DS_linear_picture_matches_single_audio_gain":"DS_curved_picture_matches_single_audio_gain",matches&&energy>1e-5&&residual/energy<1e-6);
    }
    std::cout<<"wang_lead_ms="<<pieces[1].lead*1000<<"|consonant_peak="<<energy<<std::endl;
    return ok;
}
}
