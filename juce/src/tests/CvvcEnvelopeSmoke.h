#pragma once
#include "../backend/CvvcPhonemeCache.h"
#include "CvvcConsonantDragSmoke.h"
namespace hachi
{
template<class Check>
void verifyCvvcEnvelopes(const juce::File& folder,const backend::UtauRenderRequest& original,
    const juce::File& modelFolder,Check&& check)
{
    using namespace backend;
    const auto plan=planChineseCvvc(original);
    auto defaults=original;
    for(auto& n:defaults.notes){n.amplitudeEnvelope.clear();n.cvvcEnvelopes.clear();}
    const auto defaultPlan=planChineseCvvc(defaults);
    bool standardParts=defaultPlan.request.notes.size()==4;
    for(const auto& n:defaultPlan.request.notes)
    {
        const auto& p=n.amplitudeEnvelope;
        standardParts &= p.size()==4;
        if(p.size()!=4)continue;
        standardParts &= p[0].gainDb==-60&&p[1].gainDb==0&&p[2].gainDb==0&&p[3].gainDb==-60
            &&p[0].linearToNext&&p[1].linearToNext&&p[2].linearToNext
            &&std::abs(p[1].timeSeconds-p[0].timeSeconds-.005)<1.e-7
            &&std::abs(p[3].timeSeconds-p[2].timeSeconds-.035)<1.e-7
            &&std::abs(p.back().timeSeconds-n.durationSeconds)<1.e-7;
    }
    check("unedited_cv_vc_release_each_have_standard_linear_trapezoid",standardParts);
    for(bool curved:{false,true})
    {
        auto request=defaultPlan.request;request.cvvcCurveCrossfade=curved;
        std::map<size_t,UtauMixEnvelope> windows;
        request.noteMixEnvelope=[&](size_t i,const auto& e){windows[i]=e;};
        bool single=true,shape=true;int crossings=0;
        request.notePiece=[&](size_t i,const auto&,double,double,const auto& gain,const auto&)
        {
            const auto& e=windows.at(i);single&=e.replacesStandardEnvelope;
            for(const auto& r:e.ramps)if(r.crossfade)
            {
                ++crossings;
                for(double u:{.01,.02,.1,.25,.5,.75,.9,.98,.99})
                {
                    const auto sample=r.start+static_cast<int>(u*r.length);
                    const auto t=e.startSeconds+sample/e.sampleRate;
                    single&=std::abs(gain(t)-e.gainAt(t))<1.e-5;
                    const auto v=static_cast<double>(sample-r.start)/r.length;
                    const auto expected=curved?v*v*(3-2*v):v;
                    shape&=std::abs(r.gainAt(sample)-(r.rising?expected:1-expected))<1.e-6;
                }
            }
        };
        const auto rendered=UtauRenderer::render(request);
        check(curved?"standard_curve_has_one_smooth_fade_without_extra_kink":"standard_linear_has_one_straight_fade_without_extra_kink",
            rendered.buffer.getNumSamples()>0&&crossings>0&&single&&shape);
    }
    const auto preset=PianoRollComponent::envelopePresetPoints(PianoRollComponent::envelopePresets().front(),0,.2);
    const auto canonical=standardCvvcEnvelope(.2);bool samePreset=preset.size()==canonical.size();
    for(size_t i=0;i<std::min(preset.size(),canonical.size());++i)
        samePreset&=std::abs(preset[i].timeSeconds-canonical[i].position*.2)<1.e-8
            &&preset[i].gainDb==canonical[i].value&&preset[i].linearToNext==canonical[i].linear;
    check("standard_button_and_new_phonemes_share_exact_rule",samePreset);
    const auto tiny=standardCvvcEnvelope(.012);
    check("short_default_keeps_ordered_four_points",tiny.size()==4&&tiny[1].position>0
        &&tiny[1].position<tiny[2].position&&tiny[2].position<1);
    const auto standard=standardCvvcEnvelope(.2);
    check("standard_release_is_linear_in_amplitude",std::abs(envelopeGainFromDb(
        cvvcEnvelopeValue(standard,(standard[2].position+1)*.5,true))-.5f)<1.e-6);
    auto manual=defaults;manual.notes[1].cvvcEnvelopes[3][""]={{0,-12,false},{.6,-3,false},{1,-24,false}};
    const auto manualPlan=planChineseCvvc(manual);
    check("manual_release_override_preserved",manualPlan.request.notes.back().amplitudeEnvelope.size()==3
        &&manualPlan.request.notes.back().amplitudeEnvelope.front().gainDb==-12
        &&!manualPlan.request.notes.back().amplitudeEnvelope.front().linearToNext);
    auto quiet=defaults;quiet.notes[1].cvvcDefaultEnvelopeGain=.5f;
    const auto quietPlan=planChineseCvvc(quiet);
    check("default_base_gain_applied_once",std::abs(envelopeGainFromDb(quietPlan.request.notes.back().amplitudeEnvelope[1].gainDb)-.5f)<1.e-6);
    using Spans=std::vector<std::vector<UtauPhonemeSpan>>;
    Spans renderedSpans, hiSpans;
    const auto withPreview=[&](UtauRenderRequest request,auto render,const char* name,Spans& spans)
    {
        spans.resize(request.notes.size());
        struct Piece {size_t index;double start,rate;juce::AudioBuffer<float> audio;};
        std::vector<Piece> preview;
        request.notePhonemes=[&](size_t i,const auto& p){spans[i]=p;};
        request.notePiece=[&](size_t i,const auto& audio,double rate,double lead,const auto& gain,const auto&)
        {
            Piece p{i,request.notes[i].startSeconds-lead,rate,audio};
            for(int s=0;s<audio.getNumSamples();++s)for(int c=0;c<audio.getNumChannels();++c)
                p.audio.setSample(c,s,audio.getSample(c,s)*gain(s/rate-lead));
            preview.push_back(std::move(p));
        };
        auto result=render(request);
        juce::AudioBuffer<float> sum(result.buffer.getNumChannels(),result.buffer.getNumSamples());sum.clear();
        bool complete=preview.size()==request.notes.size();
        for(const auto& p:preview)
        {
            complete&=p.rate==result.sampleRate;
            const auto start=static_cast<int>(std::lround(p.start*result.sampleRate));
            const auto from=std::max(0,-start),to=std::min(p.audio.getNumSamples(),sum.getNumSamples()-start);
            for(int c=0;c<sum.getNumChannels();++c)if(to>from)
                sum.addFrom(c,start+from,p.audio,std::min(c,p.audio.getNumChannels()-1),from,to-from);
        }
        auto peak=0.f;for(int c=0;c<sum.getNumChannels();++c)peak=std::max(peak,sum.getMagnitude(c,0,sum.getNumSamples()));
        if(peak>.98f)sum.applyGain(.98f/peak);
        float difference=0;
        for(int c=0;c<sum.getNumChannels();++c)for(int s=0;s<sum.getNumSamples();++s)
            difference=std::max(difference,std::abs(sum.getSample(c,s)-result.buffer.getSample(c,s)));
        check(name,complete&&difference<.0001f);
        return result;
    };
    auto changed=original;
    changed.notes[1].cvvcEnvelopes[1][""]={{0,-60,true},{1,-60,true}};
    changed.notes[1].cvvcEnvelopes[2]["HIFI:g"]={{0,12,true},{1,24,true}};
    auto edited=planChineseCvvc(changed);
    check("part_roles_and_leads",plan.spans[1].size()==3&&plan.spans[1][0].envelopePart==1
        &&plan.spans[1][1].envelopePart==2&&plan.spans[1][2].envelopePart==3
        &&std::abs(plan.spans[1][0].leadInSeconds-.04)<1.e-6);
    check("vc_loudness_does_not_change_cv_or_release",edited.request.notes[1].amplitudeEnvelope.front().gainDb==-60
        &&edited.request.notes[1].amplitudeEnvelope.back().gainDb==-60
        &&edited.request.notes[2].amplitudeEnvelope.front().gainDb==plan.request.notes[2].amplitudeEnvelope.front().gainDb
        &&edited.request.notes[3].amplitudeEnvelope.back().gainDb==plan.request.notes[3].amplitudeEnvelope.back().gainDb);
    check("cv_flag_not_shared_with_vc",edited.request.notes[2].flagCurves[0].second.front().second==12
        &&edited.request.notes[1].flagCurves[0].second==plan.request.notes[1].flagCurves[0].second
        &&edited.request.notes[3].flagCurves[0].second==plan.request.notes[3].flagCurves[0].second);
    changed.notes[1].durationSeconds=1;
    auto stretched=planChineseCvvc(changed);
    const auto& cv=stretched.request.notes[2];
    check("normalized_piece_curve_follows_duration",std::abs(cv.flagCurves[0].second.back().first-cv.durationSeconds)<1.e-6);
    auto silentRelease=original;silentRelease.notes[1].cvvcEnvelopes[3][""]={{0,-60,true},{1,-60,true}};
    const auto plainAudio=withPreview(original,[](const auto& r){return UtauRenderer::render(r);},
        "traditional_cvvc_preview_matches_final_crossfaded_mix",renderedSpans);
    const auto mutedAudio=UtauRenderer::render(silentRelease);
    check("renderer_publishes_crossfade_metadata_for_all_parts",renderedSpans.size()==2
        &&renderedSpans[1].size()==3&&std::all_of(renderedSpans[1].begin(),renderedSpans[1].end(),[](const auto& s){return s.mixEnvelope.available();}));
    if(!renderedSpans[0].empty()&&renderedSpans[1].size()==3)
    {
        const auto& outgoing=renderedSpans[0][0].mixEnvelope;
        const auto& incoming=renderedSpans[1][0].mixEnvelope;
        bool aligned=true;
        for(double u:{.25,.5,.75})
        {
            const auto at=original.notes[1].startSeconds+incoming.startSeconds+u*.01;
            aligned&=std::abs(incoming.gainAt(at-original.notes[1].startSeconds)-u)<.003
                &&std::abs(outgoing.gainAt(at-original.notes[0].startSeconds)-(1-u))<.003;
        }
        check("linear_cv_vc_crossing_uses_actual_oto_overlap",aligned);
    }
    auto flat=plan.request;for(auto& n:flat.notes){n.amplitudeEnvelope={{-2,0,true},{3,0,true}};n.splice=true;}
    std::map<size_t,UtauMixEnvelope> flatMix;
    flat.noteMixEnvelope=[&](size_t i,const auto& e){flatMix[i]=e;};
    bool exact=true;size_t observed=0;
    flat.notePiece=[&](size_t i,const auto&,double,double,const auto& gain,const auto&)
    {++observed;const auto& e=flatMix.at(i);for(const auto& ramp:e.ramps)for(double u:{.25,.5,.75})
        {const auto t=e.startSeconds+(ramp.start+u*ramp.length)/e.sampleRate;exact&=std::abs(gain(t)-e.gainAt(t))<1.e-6;}};
    UtauRenderer::render(flat);
    check("equal_power_display_gain_is_actual_traditional_mixer_gain",exact&&observed==flat.notes.size());
    const auto level=[](const auto& audio,double from,double seconds){
        const auto at=static_cast<int>(from*audio.sampleRate),length=static_cast<int>(seconds*audio.sampleRate);
        return audio.buffer.getNumSamples()>=at+length?audio.buffer.getMagnitude(0,at,length):0.f;};
    check("release_override_changes_actual_audio",level(plainAudio,1.26,.025)>.001f&&level(mutedAudio,1.26,.025)<1.e-6f);
    check("release_override_preserves_previous_body",std::abs(level(plainAudio,1.05,.04)-level(mutedAudio,1.05,.04))<1.e-6f);
    if(modelFolder.isDirectory())
    {
        const auto hiPlain=withPreview(original,[&](const auto& r){return renderNsfUtauPhrase(r,modelFolder,{});},
            "hifisampler_cvvc_preview_matches_final_crossfaded_mix",hiSpans);
        const auto hiMuted=renderNsfUtauPhrase(silentRelease,modelFolder,{});
        check("hifisampler_applies_release_envelope",level(hiPlain,1.26,.025)>.0001f&&level(hiMuted,1.26,.025)<1.e-6f);
        check("hifisampler_preserves_other_parts",std::abs(level(hiPlain,1.05,.04)-level(hiMuted,1.05,.04))<1.e-6f);
        check("hifisampler_metadata_uses_selected_linear_default",hiSpans[1].size()==3
            &&!hiSpans[1][0].mixEnvelope.ramps.front().equalPower&&!hiSpans[1][0].mixEnvelope.ramps.front().smoothCurve);
        auto curvedRequest=defaults;curvedRequest.cvvcCurveCrossfade=true;
        Spans curvedSpans;
        withPreview(curvedRequest,[&](const auto& r){return renderNsfUtauPhrase(r,modelFolder,{});},
            "hifisampler_smooth_standard_preview_matches_pcm",curvedSpans);
        check("hifisampler_model_render_receives_smooth_choice",curvedSpans[1].size()==3
            &&curvedSpans[1][0].mixEnvelope.ramps.front().smoothCurve&&curvedSpans[1][0].mixEnvelope.replacesStandardEnvelope);
        auto noFollowing=original;
        noFollowing.notes[1].cvvcEnvelopes[2][""]={{0,-60,true},{1,-60,true}};
        noFollowing.notes[1].cvvcEnvelopes[3][""]={{0,-60,true},{1,-60,true}};
        const auto onlyOutgoing=renderNsfUtauPhrase(noFollowing,modelFolder,{});
        check("hifisampler_vc_does_not_return_after_cv_crossfade",level(onlyOutgoing,.75,.015)<1.e-6f);
        auto longOverlap=original;longOverlap.chineseCvvc=false;
        for(auto& n:longOverlap.notes){n.amplitudeEnvelope={{-1,0,true},{2,0,true}};n.gain=.2f;}
        longOverlap.notes[1].overlapOverrideEnabled=true;longOverlap.notes[1].overlapSeconds=.15;
        longOverlap.notes[1].gain=0;
        const auto extendedTail=renderNsfUtauPhrase(longOverlap,modelFolder,{});
        check("hifisampler_overlap_past_beat_has_audio_to_fade",level(extendedTail,.82,.02)>.00001f
            &&level(extendedTail,.88,.02)<1.e-6f);
        auto external=original;external.requireExternalResampler=true;
        external.resamplerExecutable=modelFolder.getParentDirectory().getParentDirectory().getChildFile("engines/WCSNDM-0.0803.exe");
        for(auto& n:external.notes)n.flags="K2";
        const auto exPlain=UtauRenderer::render(external);external.notes[1].cvvcEnvelopes=silentRelease.notes[1].cvvcEnvelopes;
        const auto exMuted=UtauRenderer::render(external);
        check("external_resampler_applies_release_envelope",exPlain.backend.contains("utau-resampler")&&!exPlain.backend.contains("fallback")
            &&level(exPlain,1.26,.025)>.0001f&&level(exMuted,1.26,.025)<1.e-6f);
        check("external_resampler_preserves_other_parts",std::abs(level(exPlain,1.05,.04)-level(exMuted,1.05,.04))<1.e-6f);
    }

    ProjectModel model;ProjectData data;TrackData track;
    track.id="parts";track.compose=true;track.pitchAlgorithm=PitchAlgorithm::utau;
    track.utauMode=UtauMode::classic;track.chineseCvvc=true;track.voicebankDirectory=original.voicebankDirectory;
    track.outputEngine=UtauOutputEngine::resampler;
    ClipData clip;clip.id="parts-clip";clip.durationSeconds=1.6;
    for(size_t i=0;i<original.notes.size();++i)
    {
        const auto& source=original.notes[i];NoteData n;n.id=juce::String(static_cast<int>(i));n.label=source.alias;
        n.startSeconds=source.startSeconds;n.durationSeconds=source.durationSeconds;n.utauFlagCurveEnabled=true;
        for(const auto& p:source.amplitudeEnvelope)n.amplitudeEnvelope.push_back({p.timeSeconds,p.gainDb,p.linearToNext});
        clip.notes.push_back(n);
    }
    track.clips={clip};data.tracks={track};model.resetDocument(data);
    verifyCvvcConsonantDrag(data, check);
    I18n labels;PianoRollComponent roll(model,labels);
    juce::Viewport viewport;viewport.setSize(1200,750);viewport.setViewedComponent(&roll,false);
    roll.setSize(1200,750);roll.setPixelsPerSecond(700);roll.setFocusedTrack(track.id);roll.setFocusedClip(clip.id);
    const auto metadata=std::make_shared<std::vector<UtauNoteWaveform>>();
    for(size_t i=0;i<clip.notes.size();++i)
    {
        UtauNoteWaveform wave;const auto& n=clip.notes[i];wave.noteId=n.id;wave.phonemes=renderedSpans[i];
        wave.audioHash=AudioEngine::utauNoteAudioHash(n);wave.renderHash=AudioEngine::utauNoteRenderHash(n);
        wave.phonemeLayoutHash=AudioEngine::utauNotePhonemeHash(n);wave.startSeconds=n.startSeconds;
        wave.phonemeMixContextHash=AudioEngine::utauPhonemeMixContextHash(track);
        for(const auto& span:wave.phonemes)wave.leadInSeconds=std::max(wave.leadInSeconds,span.leadInSeconds-span.startSeconds);
        metadata->push_back(wave);
    }
    const auto beforeMetadata=model.contentFingerprint();const auto metadataRevision=model.revisionNumber();
    roll.setUtauNoteWaveforms(metadata);roll.setTool(PianoRollComponent::Tool::amplitude);
    check("phoneme_cache_does_not_dirty_or_advance_revision",model.contentFingerprint()==beforeMetadata&&model.revisionNumber()==metadataRevision);
    check("rendered_phonemes_cached_for_project",!model.snapshot().tracks[0].clips[0].notes[1].cvvcPhonemeCache.isEmpty());
    juce::String cacheError;ProjectModel cachedModel;
    check("phoneme_cache_saved_and_loaded",model.save(folder.getChildFile("phoneme-cache.hjpx"),cacheError)
        &&cachedModel.load(folder.getChildFile("phoneme-cache.hjpx"),cacheError));
    {
        PianoRollComponent cachedRoll(cachedModel,labels);cachedRoll.setSize(1200,750);cachedRoll.setFocusedTrack(track.id);cachedRoll.diagnosticRefresh();
        check("reopened_phonemes_available_without_audio",cachedRoll.diagnosticPhonemeSpans("1").size()==3
            &&cachedRoll.diagnosticPhonemeSpans("1")[0].mixEnvelope.available());
        cachedRoll.setSelectedNoteIds({"0"});cachedRoll.setShowPhonemeInfo(false);
        check("phoneme_info_off_shows_selection_only",cachedRoll.phonemeInformationVisible("0")&&!cachedRoll.phonemeInformationVisible("1"));
        cachedRoll.setShowPhonemeInfo(true);cachedRoll.setSelectedNoteIds({});
        check("phoneme_info_on_survives_deselection",cachedRoll.phonemeInformationVisible("0")&&cachedRoll.phonemeInformationVisible("1"));
        cachedModel.setNoteLabel("0","ba");cachedRoll.diagnosticRefresh();
        check("neighbor_lyric_change_invalidates_saved_layout",cachedRoll.diagnosticPhonemeSpans("1").empty());
    }
    const auto partial=std::make_shared<std::vector<UtauNoteWaveform>>();partial->push_back(metadata->back());
    roll.setUtauNoteWaveforms(partial);
    check("partial_render_retains_other_phonemes",!roll.diagnosticPhonemeSpans("0").empty());
    roll.setUtauNoteWaveforms({});
    check("clearing_waveforms_retains_saved_phonemes",roll.diagnosticPhonemeSpans("1").size()==3);
    roll.setUtauNoteWaveforms(metadata);
    const auto cache=backend::readCvvcPhonemeCache(model.snapshot().tracks[0].clips[0].notes[1].cvvcPhonemeCache);
    check("phoneme_cache_rejects_malformed_input",backend::readCvvcPhonemeCache("{bad}").spans.empty()
        &&backend::readCvvcPhonemeCache("{\"version\":99,\"spans\":[]}").spans.empty());
    check("phoneme_cache_hash_roundtrip",cache.context==AudioEngine::utauPhonemeMixContextHash(track)
        &&cache.layout==AudioEngine::utauNotePhonemeHash(clip.notes[1]));
    const auto refresh=[&]{model.dispatchPendingMessages();roll.diagnosticRefresh();};refresh();
    viewport.setViewPosition(0,static_cast<int>(roll.diagnosticYForMidi(60)-180));
    // Wait for the existing background OTO index read, never read the bank on the UI thread.
    for(int i=0;i<100&&roll.diagnosticCvvcEnvelopeHandles("1",1,true).empty();++i)
    {juce::Thread::sleep(10);refresh();}
    const auto eventFor=[&](juce::Point<float> at,int mods=juce::ModifierKeys::leftButtonModifier){
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(),at,juce::ModifierKeys(mods),
            0,0,0,0,0,&roll,&roll,juce::Time::getCurrentTime(),at,juce::Time::getCurrentTime(),1,false);};
    auto handles=roll.diagnosticCvvcEnvelopeHandles("1",1,true);
    check("lane_exposes_three_independent_piece_handles",handles.size()>=2
        &&roll.diagnosticCvvcEnvelopeHandles("1",2,true).size()>=2&&roll.diagnosticCvvcEnvelopeHandles("1",3,true).size()>=2);
    if(handles.size()<2)return;
    const auto screenshot=[&](const char* name){juce::PNGImageFormat png;auto out=folder.getChildFile(name).createOutputStream();
        const auto image=roll.createComponentSnapshot(viewport.getViewArea());if(out)png.writeImageToStream(image,*out);return image;};
    const auto viewBefore=viewport.getViewPosition();
    roll.setPixelsPerSecond(4000);refresh();viewport.setViewPosition(static_cast<int>(roll.diagnosticEdgeX(.60)),viewBefore.y);
    auto image=screenshot("cvvc-crossfade-detail.png");
    const auto& vcEnvelope=renderedSpans[1][0].mixEnvelope;
    const auto local=vcEnvelope.startSeconds+.005;
    CvvcEnvelopeCurve inherited;for(const auto& p:plan.request.notes[1].amplitudeEnvelope)inherited.push_back({p.timeSeconds,p.gainDb,p.linearToNext});
    const auto raw=cvvcEnvelopeValue(inherited,local-plan.offsets[1],true);
    const auto expected=juce::Point<int>(static_cast<int>(roll.diagnosticEdgeX(original.notes[1].startSeconds+local))-viewport.getViewPositionX(),
        static_cast<int>(roll.diagnosticAmplitudeLaneY(envelopeDbFromGain(envelopeGainFromDb(raw)*vcEnvelope.gainAt(local))))-viewport.getViewPositionY());
    bool cyan=false;
    for(int y=expected.y-3;y<=expected.y+3;++y)for(int x=expected.x-3;x<=expected.x+3;++x)
        if(image.getBounds().contains(x,y)){const auto c=image.getPixelAt(x,y);cyan|=c.getGreen()>100&&c.getGreen()>c.getRed()*1.25f&&c.getBlue()>c.getRed()*1.15f;}
    check("loudness_lane_paints_crossfaded_curve_at_rendered_gain",cyan);
    if(!hiSpans.empty())
    {
        for(size_t i=0;i<metadata->size();++i)(*metadata)[i].phonemes=hiSpans[i];
        roll.setUtauNoteWaveforms(std::make_shared<std::vector<UtauNoteWaveform>>(*metadata));refresh();screenshot("cvvc-hifisampler-crossfade-detail.png");
        for(size_t i=0;i<metadata->size();++i)(*metadata)[i].phonemes=renderedSpans[i];
        roll.setUtauNoteWaveforms(metadata);
    }
    roll.setPixelsPerSecond(700);refresh();viewport.setViewPosition(viewBefore);handles=roll.diagnosticCvvcEnvelopeHandles("1",1,true);
    auto alteredTrack=track;alteredTrack.clips[0].notes[0].utauOverlapSeconds=.1;alteredTrack.clips[0].notes[0].utauOverlapOverrideEnabled=true;
    check("neighbor_timing_invalidates_crossfade_metadata",AudioEngine::utauPhonemeMixContextHash(track)!=AudioEngine::utauPhonemeMixContextHash(alteredTrack));
    alteredTrack=track;alteredTrack.clips[0].notes[0].amplitudeEnvelope={{0,-6,true},{.5,0,true}};
    check("manual_envelope_keeps_valid_crossfade_metadata",AudioEngine::utauPhonemeMixContextHash(track)==AudioEngine::utauPhonemeMixContextHash(alteredTrack));
    const auto before=model.snapshot();const auto rev=model.revisionNumber();
    const auto point=handles.front();
    roll.mouseDown(eventFor(point));roll.mouseDrag(eventFor({point.x,roll.diagnosticAmplitudeLaneY(-6)}));
    check("piece_drag_has_no_intermediate_project_write",model.revisionNumber()==rev);
    roll.mouseUp(eventFor(point));refresh();
    auto note=model.snapshot().tracks[0].clips[0].notes[1];
    check("vc_drag_commits_only_vc",model.revisionNumber()==rev+1&&note.cvvcEnvelopes.size()==1
        &&note.cvvcEnvelopes.contains(1)&&note.cvvcEnvelopes.at(1).at("").front().value< -5.9f);
    check("piece_edit_preserves_score_identity",model.snapshot().tracks[0].clips[0].notes.size()==2
        &&note.label=="ba"&&note.startSeconds==.8&&note.durationSeconds==.5);
    check("edit_keeps_piece_handles_while_render_pending",roll.diagnosticCvvcEnvelopeHandles("1",3,true).size()>=2);
    check("piece_edit_undo_redo",model.undo()&&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.empty()
        &&model.redo()&&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes==note.cvvcEnvelopes);refresh();
    const auto cvHandles=roll.diagnosticCvvcEnvelopeHandles("1",2,true);
    roll.mouseDown(eventFor(roll.diagnosticCvvcPhonemeTab("1",2).getCentre()));
    const juce::Point<float> mid((cvHandles.front().x+cvHandles.back().x)*.5f,roll.diagnosticAmplitudeLaneY(0));
    roll.mouseDoubleClick(eventFor(mid));refresh();
    auto added=roll.diagnosticCvvcEnvelopeHandles("1",2,true);
    check("double_click_adds_only_to_cv",added.size()>cvHandles.size()
        &&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.at(1)==note.cvvcEnvelopes.at(1));
    const auto inserted=std::min_element(added.begin(),added.end(),[&](auto a,auto b){return std::abs(a.x-mid.x)<std::abs(b.x-mid.x);});
    roll.mouseDown(eventFor(*inserted,juce::ModifierKeys::rightButtonModifier));refresh();
    check("right_click_removes_internal_piece_point",roll.diagnosticCvvcEnvelopeHandles("1",2,true).size()==cvHandles.size());
    handles=roll.diagnosticCvvcEnvelopeHandles("1",1,true);
    roll.mouseDown(eventFor(roll.diagnosticCvvcPhonemeTab("1",1).getCentre()));
    const auto cancelRev=model.revisionNumber();roll.mouseDown(eventFor(handles.front()));roll.mouseDrag(eventFor({handles.front().x,handles.front().y+10}));
    roll.keyPressed(juce::KeyPress(juce::KeyPress::escapeKey));roll.mouseUp(eventFor(handles.front()));
    check("piece_drag_escape_cancels",model.revisionNumber()==cancelRev);
    roll.setFlagLaneFlag("g");roll.setTool(PianoRollComponent::Tool::flagCurve);refresh();
    handles=roll.diagnosticCvvcEnvelopeHandles("1",3,false);
    roll.mouseDown(eventFor(roll.diagnosticCvvcPhonemeTab("1",3).getCentre()));
    if(!handles.empty())
    {roll.mouseDown(eventFor(handles.back()));roll.mouseDrag(eventFor({handles.back().x,roll.flagLaneY(20)}));roll.mouseUp(eventFor(handles.back()));refresh();}
    note=model.snapshot().tracks[0].clips[0].notes[1];
    check("release_flag_drag_is_independent",note.cvvcEnvelopes.contains(3)&&note.cvvcEnvelopes.at(3).contains("g")
        &&note.cvvcEnvelopes.at(3).at("g").back().value>19.9f&&!note.cvvcEnvelopes.at(1).contains("g"));
    auto sent=AudioEngine::utauRequestNotesForClip(model.snapshot(),clip.id);
    check("render_request_receives_per_part_data",sent.size()==2&&sent[1].cvvcEnvelopes==note.cvvcEnvelopes);
    const auto oldNote=before.tracks[0].clips[0].notes[1];
    check("part_edits_invalidate_render_not_layout",AudioEngine::utauNoteRenderHash(note)!=AudioEngine::utauNoteRenderHash(oldNote)
        &&AudioEngine::utauNotePhonemeHash(note)==AudioEngine::utauNotePhonemeHash(oldNote));
    juce::String error;ProjectModel loaded;
    const auto savedFile=folder.getChildFile("cvvc-envelopes.hjpx");
    check("part_curves_project_roundtrip",model.save(savedFile,error)&&loaded.load(savedFile,error)
        &&loaded.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes==note.cvvcEnvelopes);
    const auto copies=loaded.duplicateNotes({"1"},clip.id,2.0);
    const auto copied=loaded.snapshot();bool kept=false;
    for(const auto& n:copied.tracks[0].clips[0].notes)if(!copies.empty()&&n.id==copies.front())kept=n.cvvcEnvelopes==note.cvvcEnvelopes;
    check("duplicating_note_preserves_piece_curves",copies.size()==1&&kept);
    check("stale_piece_edit_rejected",!model.setCvvcEnvelope("1",3,"",{{0,0,true},{1,0,true}},cancelRev));
    for(bool amplitude:{true,false})
    {
        roll.setTool(amplitude?PianoRollComponent::Tool::amplitude:PianoRollComponent::Tool::flagCurve);refresh();
        juce::PNGImageFormat png;auto out=folder.getChildFile(amplitude?"cvvc-amplitude-pieces.png":"cvvc-flag-pieces.png").createOutputStream();
        if(out)png.writeImageToStream(roll.createComponentSnapshot(viewport.getViewArea()),*out);
    }
    check("hifi_reset_retains_other_engine_piece_curve",model.resetNotesUtauFlagCurve({"1"},"HIFI:g")
        &&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.at(3).contains("g")
        &&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.at(3).at("HIFI:g").empty());
    const auto suppressed=model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes;
    check("empty_native_override_roundtrip",cvvcEnvelopesFromVar(cvvcEnvelopesToVar(suppressed))==suppressed);
    check("reset_flag_clears_only_flag_overrides",model.resetNotesUtauFlagCurve({"1"},"g")
        &&!hasCvvcFlagCurve(model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes)
        &&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.at(1).contains(""));
    refresh();roll.setFlagLaneFlag("g");roll.setTool(PianoRollComponent::Tool::flagCurve);
    roll.setFlagEditMode(PianoRollComponent::FlagEditMode::continuous);refresh();
    const auto strokeBefore=model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes;
    handles=roll.diagnosticCvvcEnvelopeHandles("1",2,false);
    roll.mouseDown(eventFor(roll.diagnosticCvvcPhonemeTab("1",2).getCentre()));
    const auto from=handles.front().x+(handles.back().x-handles.front().x)*.3f;
    const auto to=handles.front().x+(handles.back().x-handles.front().x)*.6f;
    roll.mouseDown(eventFor({from,roll.flagLaneY(10)}));roll.mouseDrag(eventFor({to,roll.flagLaneY(30)}));roll.mouseUp(eventFor({to,roll.flagLaneY(30)}));refresh();
    const auto strokeAfter=model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes;
    check("continuous_flag_is_scoped_to_chosen_piece",strokeAfter.at(2).contains("g")
        &&strokeAfter.at(1)==strokeBefore.at(1)&&strokeAfter.at(3)==strokeBefore.at(3));
    check("whole_note_envelope_replaces_piece_loudness",model.setNoteAmplitudeEnvelope("1",{{0,0,true},{.5,0,true}})
        &&!model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.at(1).contains(""));
    auto clean=data;for(auto& n:clean.tracks[0].clips[0].notes)n.amplitudeEnvelope.clear();
    model.resetDocument(clean);refresh();
    auto defaultMetadata=std::make_shared<std::vector<UtauNoteWaveform>>(*metadata);
    auto defaultPreview=defaults;
    defaultPreview.notePhonemes=[&](size_t i,const auto& spans){if(i<defaultMetadata->size())(*defaultMetadata)[i].phonemes=spans;};
    UtauRenderer::render(defaultPreview);
    for(size_t i=0;i<defaultMetadata->size();++i)
    {
        const auto& n=clean.tracks[0].clips[0].notes[i];auto& wave=(*defaultMetadata)[i];
        wave.renderHash=AudioEngine::utauNoteRenderHash(n);wave.audioHash=AudioEngine::utauNoteAudioHash(n);
    }
    roll.setUtauNoteWaveforms(defaultMetadata);roll.setTool(PianoRollComponent::Tool::amplitude);refresh();
    bool standardHandles=true;
    for(int part:{1,2,3})
    {
        const auto h=roll.diagnosticCvvcEnvelopeHandles("1",part,true);standardHandles&=h.size()==4;
        if(h.size()==4)standardHandles&=std::abs(h[1].y-roll.diagnosticAmplitudeLaneY(0))<.01
            &&std::abs(h[2].y-h[1].y)<.01&&std::abs(h[0].y-roll.diagnosticAmplitudeLaneY(-60))<.01;
    }
    check("default_lane_all_parts_show_independent_trapezoids",standardHandles);
    check("viewing_defaults_does_not_write_project_curves",model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes.empty()
        &&model.snapshot().tracks[0].clips[0].notes[1].amplitudeEnvelope.empty());
    roll.setPixelsPerSecond(3000);refresh();viewport.setViewPosition(static_cast<int>(roll.diagnosticEdgeX(.96)),viewBefore.y);
    screenshot("cvvc-default-release-trapezoid.png");
    auto volume=clean;volume.tracks[0].clips[0].notes[1].amplitudeEnvelopeBasePercent=50;
    auto volumeRequest=defaults;volumeRequest.notes=AudioEngine::utauRequestNotesForClip(volume,clip.id);
    const auto volumePlan=planChineseCvvc(volumeRequest);
    check("project_base_volume_keeps_default_semantics",volumeRequest.notes[1].amplitudeEnvelope.empty()
        &&std::abs(envelopeGainFromDb(volumePlan.request.notes.back().amplitudeEnvelope[1].gainDb)-.5f)<1.e-6);
    roll.setSelectedNoteIds({"0","1"});
    const auto beforeCrossfade=model.contentFingerprint();
    check("auto_crossfade_two_cvvc_score_notes",roll.crossfadeSelectedNotes()>0);
    const auto crossfaded=model.snapshot();
    check("auto_crossfade_writes_independent_cvvc_parts",!crossfaded.tracks[0].clips[0].notes[0].cvvcEnvelopes.empty()
        &&crossfaded.tracks[0].clips[0].notes[1].cvvcEnvelopes.contains(1)
        &&crossfaded.tracks[0].clips[0].notes[1].amplitudeEnvelope.empty());
    auto crossRequest=defaults;crossRequest.notes=AudioEngine::utauRequestNotesForClip(crossfaded,clip.id);
    const auto crossPlan=planChineseCvvc(crossRequest);
    check("auto_crossfade_reaches_cvvc_render_plan",crossPlan.request.notes[1].amplitudeEnvelope.size()>=2);
    const auto afterCrossfade=model.contentFingerprint();
    check("auto_crossfade_cvvc_repeat_is_idempotent",roll.crossfadeSelectedNotes()>0&&model.contentFingerprint()==afterCrossfade);
    const auto crossAudio=UtauRenderer::render(crossRequest);
    const auto defaultAudio=UtauRenderer::render(defaults);
    double difference=0;
    for(int i=0;i<std::min(crossAudio.buffer.getNumSamples(),defaultAudio.buffer.getNumSamples());++i)
        difference+=std::abs(crossAudio.buffer.getSample(0,i)-defaultAudio.buffer.getSample(0,i));
    check("auto_crossfade_changes_rendered_pcm",difference>1.e-5);
    check("auto_crossfade_cvvc_single_undo",model.undo()&&model.contentFingerprint()==beforeCrossfade);
    refresh();roll.setPixelsPerSecond(700);roll.setTool(PianoRollComponent::Tool::amplitude);refresh();
    viewport.setViewPosition(viewBefore);
    roll.setSelectedNoteIds({});roll.setShowPhonemeInfo(false);
    const auto tab=roll.diagnosticCvvcPhonemeTab("1",1);
    const auto selectionRevision=model.revisionNumber();
    check("cvvc_tabs_available_without_parent_selection_or_info_option",!tab.isEmpty()
        &&!roll.diagnosticCvvcPhonemeTab("1",2).isEmpty()&&!roll.diagnosticCvvcPhonemeTab("1",3).isEmpty());
    roll.mouseDown(eventFor(tab.getCentre()));roll.mouseUp(eventFor(tab.getCentre()));
    check("tab_selects_phoneme_without_project_write",roll.hasSelectedCvvcPhoneme()
        &&roll.selectedNoteIds()==std::vector<juce::String>{"1"}&&model.revisionNumber()==selectionRevision);
    screenshot("cvvc-selected-phoneme-tab.png");
    const auto presetBefore=model.snapshot().tracks[0].clips[0].notes[1];
    check("selected_phoneme_preset_applies",roll.applyEnvelopePreset(.01,.02,-6)==1);refresh();
    const auto presetAfter=model.snapshot().tracks[0].clips[0].notes[1];
    check("phoneme_preset_does_not_write_parent_or_other_parts",presetAfter.amplitudeEnvelope.size()==presetBefore.amplitudeEnvelope.size()
        &&std::equal(presetAfter.amplitudeEnvelope.begin(),presetAfter.amplitudeEnvelope.end(),presetBefore.amplitudeEnvelope.begin(),
            [](const auto& a,const auto& b){return a.timeSeconds==b.timeSeconds&&a.gainDb==b.gainDb&&a.linearToNext==b.linearToNext;})
        &&presetAfter.cvvcEnvelopes.contains(1)&&!presetAfter.cvvcEnvelopes.contains(2)&&!presetAfter.cvvcEnvelopes.contains(3));
    check("phoneme_preset_single_undo",model.undo()&&model.snapshot().tracks[0].clips[0].notes[1].cvvcEnvelopes==presetBefore.cvvcEnvelopes);refresh();
    const auto blockedBefore=model.contentFingerprint();
    const auto other=roll.diagnosticCvvcEnvelopeHandles("1",3,true);
    if(!other.empty()){roll.mouseDown(eventFor(other.back()));roll.mouseDrag(eventFor({other.back().x,other.back().y-20}));roll.mouseUp(eventFor(other.back()));}
    check("focused_phoneme_does_not_edit_hidden_other_handles",model.contentFingerprint()==blockedBefore);
    roll.setTool(PianoRollComponent::Tool::flagCurve);refresh();
    check("phoneme_selection_survives_envelope_lane_switch",roll.hasSelectedCvvcPhoneme());
    roll.setSelectedNoteIds({"1"});
    check("whole_note_selection_exits_phoneme_focus",!roll.hasSelectedCvvcPhoneme());
    const auto beforeStandard=model.contentFingerprint();
    check("whole_cvvc_standard_preset_applies_per_phoneme",roll.applyEnvelopePreset(PianoRollComponent::envelopePresets().front())==1);refresh();
    const auto allStandard=model.snapshot().tracks[0].clips[0].notes[1];
    check("whole_cvvc_standard_writes_three_independent_parts",allStandard.cvvcEnvelopes.contains(1)
        &&allStandard.cvvcEnvelopes.contains(2)&&allStandard.cvvcEnvelopes.contains(3)&&allStandard.amplitudeEnvelope.empty());
    check("whole_cvvc_preset_is_single_undo",model.undo()&&model.contentFingerprint()==beforeStandard);
}
}
