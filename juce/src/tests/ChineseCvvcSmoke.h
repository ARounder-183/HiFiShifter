#pragma once
#include "../backend/ChineseCvvcPhonemizer.h"
#include "CvvcEnvelopeSmoke.h"

namespace hachi
{
inline bool MainComponent::diagnosticChineseCvvc(const juce::File& folder, const juce::File& modelFolder)
{
    using namespace backend;
    folder.createDirectory();
    bool ok = true;
    juce::Array<juce::var> checks;
    const auto check = [&](const char* name, bool pass)
    {
        ok &= pass;
        auto* value = new juce::DynamicObject(); value->setProperty("name", name); value->setProperty("passed", pass);
        checks.add(juce::var(value)); std::cout << name << '=' << pass << std::endl;
    };
    const auto near = [](double a, double b) { return std::abs(a-b) < 1.e-6; };
    const auto bank = folder.getChildFile("bank"); bank.createDirectory();
    const juce::String rules = "[VOWEL]\na=A=a,ba,pa=100\ni=I=bi=100\n[CONSONANT]\nb=ba,bi=1\np=pa=1\n[REPLACE]\nbah=ba\n";
    bank.getChildFile("presamp.ini").replaceWithText(rules);
    const juce::String oto = "tone.wav=a,0,100,-900,80,20\ntone.wav=ba,0,100,-900,80,20\n"
        "tone.wav=bi,0,100,-900,80,20\ntone.wav=pa,0,100,-900,80,20\n"
        "tone.wav=a b,0,100,-900,40,10\ntone.wav=a R,0,100,-900,20,10\ntone.wav=i R,0,100,-900,20,10\n";
    bank.getChildFile("oto.ini").replaceWithText(oto);
    {
        juce::AudioBuffer<float> wave(1,44100);
        for (int i=0;i<wave.getNumSamples();++i) wave.setSample(0,i,.15f*std::sin(static_cast<float>(i*2*juce::MathConstants<double>::pi*261.6256/44100)));
        auto out=bank.getChildFile("tone.wav").createOutputStream();
        if (!out) return false; out->setPosition(0); out->truncate();
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(out.release(),44100,1,16,{},0));
        if (!writer || !writer->writeFromAudioSampleBuffer(wave,0,wave.getNumSamples())) return false;
    }
    UtauRenderer::invalidateVoicebankCache();
    UtauRenderRequest request; request.chineseCvvc=true; request.voicebankDirectory=bank;
    request.targetDurationSeconds=1.6; request.bpm=120;
    UtauNoteRenderSpec a; a.alias="a"; a.startSeconds=.3; a.durationSeconds=.5; a.midiNote=60;
    UtauNoteRenderSpec b=a; b.alias="ba"; b.startSeconds=.8;
    b.timelinePitchCents=[](double t){return static_cast<float>(t*100);};
    b.amplitudeEnvelope={{-.08,-60,true},{-.05,0,true},{.48,0,true},{.5,-60,true}};
    b.flagCurves={{"HIFI:g",{{0,0},{.5,20}}}};
    request.notes={a,b};
    const auto aliases=[](const ChineseCvvcPlan& plan){juce::StringArray names;for(const auto& note:plan.request.notes)names.add(note.alias);return names.joinIntoString("|");};
    auto plan=planChineseCvvc(request);
    check("cv_vc_cv_release",aliases(plan)=="a|a b|ba|a R");
    if(plan.request.notes.size()!=4)return false;
    check("vc_position_from_next_cv_oto",near(plan.request.notes[1].startSeconds,.72));
    check("previous_body_stops_at_vc",near(plan.request.notes[0].durationSeconds,.42));
    check("release_length_is_min_one_sixth_or_60_ticks",near(plan.request.notes[3].durationSeconds,.0625));
    check("ownership_and_negative_phoneme_span",plan.owners==std::vector<std::size_t>{0,1,1,1}&&near(plan.spans[1][0].startSeconds,-.08));
    check("parent_envelope_starts_before_vc",plan.request.notes[1].preserveEnvelopeTiming&&plan.request.notes[1].amplitudeEnvelope.front().timeSeconds<0);
    check("flag_curve_keeps_parent_clock",near(plan.request.notes[1].flagCurves[0].second[0].first,.08));
    check("pitch_curve_keeps_parent_clock",near(plan.request.notes[1].timelinePitchCents(0),-8));
    check("score_and_lyrics_not_modified",request.notes.size()==2&&request.notes[1].alias=="ba"&&near(request.notes[0].durationSeconds,.5));
    auto changed=request; changed.notes[1].consonantVelocity=200;
    check("velocity_changes_vc_duration",near(planChineseCvvc(changed).spans[1][0].startSeconds,-.04));
    changed=request; changed.notes[0].durationSeconds=.015; changed.notes[1].startSeconds=.315;
    auto shortPlan=planChineseCvvc(changed);
    check("short_note_keeps_positive_durations",near(shortPlan.spans[1][0].startSeconds,-.01)&&std::all_of(shortPlan.request.notes.begin(),shortPlan.request.notes.end(),[](const auto& n){return n.durationSeconds>0;}));
    changed=request; changed.notes[1].alias="bah";
    check("presamp_replace_exact_match",aliases(planChineseCvvc(changed))==aliases(plan));
    changed=request; changed.notes[1].alias=juce::String::fromUTF8("八");
    check("single_hanzi_to_pinyin",aliases(planChineseCvvc(changed))==aliases(plan));
    changed=request; changed.notes[1].alias="pa";
    check("missing_vc_falls_back_to_cv",aliases(planChineseCvvc(changed))=="a|pa|a R");
    changed=request; changed.notes[1].alias="absent";
    auto absent=planChineseCvvc(changed);
    check("missing_cv_keeps_failed_lyric_for_piano",aliases(absent).contains("absent")&&!aliases(absent).contains("RR")&&absent.warning.contains("Missing CV"));
    changed=request; changed.notes[1].startSeconds=1;
    check("gap_breaks_vc_context",!aliases(planChineseCvvc(changed)).contains("a b"));
    changed=request; changed.notes[0].alias="RR";
    check("rest_breaks_vc_context",!aliases(planChineseCvvc(changed)).contains("a b"));
    changed=request; changed.notes[1].alias="R";
    check("explicit_release_uses_previous_vowel",aliases(planChineseCvvc(changed))=="a|a R");
    changed=request; changed.notes[1].alias="[a b]";
    check("explicit_alias_bypasses_phonemizer",aliases(planChineseCvvc(changed))=="a|a b");
    bank.getChildFile("oto.ini").replaceWithText(oto+"tone.wav=a ba,0,100,-900,80,20\n");
    UtauRenderer::invalidateVoicebankCache();
    check("full_context_alias_preferred",aliases(planChineseCvvc(request))=="a|a ba|a R");
    bank.getChildFile("oto.ini").replaceWithText("tone.wav=a,0,100,-900,80,20\n");
    UtauRenderer::invalidateVoicebankCache();
    check("strict_lookup_rejects_single_sample_fallback",!UtauRenderer::mappedAliasTiming(bank,"a b",60));
    check("manual_single_sample_fallback_preserved",UtauRenderer::resolveVoiceSample(bank,"a b",60).found);
    bank.getChildFile("oto.ini").replaceWithText(oto
        +"tone.wav=low_a b,0,100,-900,40,10\ntone.wav=high_ba,0,100,-900,80,20\n"
         "tone.wav=low_low_a b,0,100,-900,170,10\n");
    bank.getChildFile("prefix.map").replaceWithText("C4\tlow_\t\nD4\thigh_\t\n");
    UtauRenderer::invalidateVoicebankCache();
    changed=request; changed.notes[1].midiNote=62;
    const auto mappedPlan=planChineseCvvc(changed);
    check("vc_maps_previous_pitch_cv_maps_current_pitch",aliases(mappedPlan)=="a|low_a b|high_ba|a R");
    check("vc_pitch_cents_adjust_to_previous_key",near(mappedPlan.request.notes[1].timelinePitchCents(0),192));
    const auto exact=UtauRenderer::resolveVoiceSample(bank,"low_a b",60,100,false,false,0,false,0,false,0,nullptr,true);
    check("mapped_alias_is_not_prefixed_twice",exact.found&&near(exact.preutteranceSeconds,.04));
    // Restore fixture without deleting unrelated paths.
    bank.getChildFile("prefix.map").replaceWithText("");bank.getChildFile("oto.ini").replaceWithText(oto);
    UtauRenderer::invalidateVoicebankCache();
    const auto noConfig=folder.getChildFile("no-config");noConfig.createDirectory();
    changed=request;changed.voicebankDirectory=noConfig;
    auto fallback=planChineseCvvc(changed);
    check("missing_config_retains_manual_request_and_warns",fallback.request.notes.size()==2&&fallback.warning.contains("presamp.ini"));
    juce::String error;juce::MemoryBlock gbk;
    check("gbk_fixture_encoded",LegacyTextCodec::encode(juce::String::fromUTF8(";中文配置\n")+rules,936,false,gbk,error));
    bank.getChildFile("presamp.ini").replaceWithData(gbk.getData(),gbk.getSize());
    check("legacy_presamp_encoding",aliases(planChineseCvvc(request))=="a|a b|ba|a R");
    bank.getChildFile("presamp.ini").replaceWithText(rules);

    const auto variantRules = rules + "[VOWEL]\ne0=e0=nue,lue=100\n[CONSONANT]\nny=nve=0\nly=lve=0\n";
    const auto variantOto = oto + "tone.wav=nve,0,100,-900,80,20\ntone.wav=lve,0,100,-900,80,20\n"
        "tone.wav=a ny,0,100,-900,40,10\ntone.wav=a ly,0,100,-900,40,10\n"
        "tone.wav=e0 R,0,100,-900,20,10\ntone.wav=e0 b,0,100,-900,40,10\n";
    bank.getChildFile("presamp.ini").replaceWithText(variantRules);
    bank.getChildFile("oto.ini").replaceWithText(variantOto);
    UtauRenderer::invalidateVoicebankCache();
    changed=request;changed.notes[1].alias="nue";
    check("nue_uses_recorded_nve_and_vowel_tail",aliases(planChineseCvvc(changed))=="a|a ny|nve|e0 R");
    changed.notes[1].alias=juce::String::fromUTF8("nüe");
    check("umlaut_spelling_uses_bank_alias",aliases(planChineseCvvc(changed))=="a|a ny|nve|e0 R");
    changed.notes[1].alias="lue";
    check("lue_uses_recorded_lve_and_vowel_tail",aliases(planChineseCvvc(changed))=="a|a ly|lve|e0 R");
    changed=request;changed.notes[0].alias="nue";
    check("resolved_spelling_keeps_next_vc_context",aliases(planChineseCvvc(changed))=="nve|e0 b|ba|a R");
    changed=request;changed.notes[1].alias="[nue]";
    check("explicit_alias_not_respelled",aliases(planChineseCvvc(changed))=="a|nue");
    bank.getChildFile("presamp.ini").replaceWithText(variantRules+"[REPLACE]\nnue=absent\n");
    changed.notes[1].alias="nue";
    check("explicit_replace_not_overridden",planChineseCvvc(changed).warning.contains("Missing CV: absent"));
    bank.getChildFile("presamp.ini").replaceWithText(variantRules);
    bank.getChildFile("oto.ini").replaceWithText(variantOto+"tone.wav=a nue,0,100,-900,80,20\n");
    UtauRenderer::invalidateVoicebankCache();
    check("exact_contextual_alias_not_respelled",aliases(planChineseCvvc(changed))=="a|a nue|e0 R");
    const std::map<juce::String,juce::String> conflicting{{"nve","e0"},{"nue","other"}};
    check("conflicting_vowel_groups_not_inferred",ChineseCvvcRules::lookup(conflicting,juce::String::fromUTF8("nüe"))==juce::String::fromUTF8("nüe"));
    check("explicit_vowel_group_wins",ChineseCvvcRules::lookup(conflicting,"nue")=="other");

    // Public presamp format variants must drive actual OTO selection, not
    // merely parse successfully or fall back to the old space-separated form.
    const auto customOto = oto
        + "tone.wav=start_a,0,100,-900,80,20\ntone.wav=a+b,0,100,-900,40,10\n"
          "tone.wav=bA,0,100,-900,110,20\ntone.wav=a_tail,0,100,-900,20,10\n"
          "tone.wav=a_ba,0,100,-900,80,20\ntone.wav=ab,0,100,-900,40,10\n"
          "tone.wav=a  b,0,100,-900,40,10\ntone.wav=A_end,0,100,-900,20,10\n";
    bank.getChildFile("oto.ini").replaceWithText(customOto);
    UtauRenderer::invalidateVoicebankCache();
    const juce::String formats = "[ALIAS]\nBEGINING_CV=start_%CV%\nVCV=%v%%VCVPAD%%CV%\n"
        "VC=%v%%vcpad%%c%\nCV=missing_%CV%,%c%%V%,%CV%\n"
        "VCPAD=+\nVCVPAD=_\nENDING1=%v%_tail\n[PRIORITY]\nba\n";
    bank.getChildFile("presamp.ini").replaceWithText(rules+formats);
    check("alias_templates_candidates_and_priority",aliases(planChineseCvvc(request))=="start_a|a+b|bA|a_tail");
    const auto upperAlias=UtauRenderer::mappedAliasTiming(bank,"bA",60);
    const auto lowerAlias=UtauRenderer::mappedAliasTiming(bank,"ba",60);
    const auto upperSample=UtauRenderer::resolveVoiceSample(bank,"bA",60,100,false,false,0,false,0,false,0,nullptr,true);
    check("phoneme_case_preserved_through_render_lookup",upperAlias&&lowerAlias
        &&near(upperAlias->preutteranceSeconds,.11)&&near(lowerAlias->preutteranceSeconds,.08)
        &&upperSample.found&&near(upperSample.preutteranceSeconds,.11));
    const auto noPriority = formats.upToFirstOccurrenceOf("[PRIORITY]",false,false);
    bank.getChildFile("presamp.ini").replaceWithText(rules+noPriority);
    check("custom_vcv_separator",aliases(planChineseCvvc(request))=="start_a|a_ba|a_tail");
    bank.getChildFile("presamp.ini").replaceWithText(rules+formats+"[VCPAD]\n \n[ENDTYPE]\n%v% R\n[ENDTYPE1]\n%V%_end\n");
    check("legacy_pad_and_ending_precedence",aliases(planChineseCvvc(request))=="start_a|a b|bA|A_end");
    changed=request;changed.notes[1].alias="R";
    check("explicit_release_uses_configured_format",aliases(planChineseCvvc(changed))=="start_a|A_end");
    bank.getChildFile("presamp.ini").replaceWithText(rules+"[VCPAD]\n+\n[ENDTYPE]\n%v%_tail\n"+formats+"[ALIAS]\nVCPAD=_\n");
    check("legacy_sections_win_independent_of_order",aliases(planChineseCvvc(request))=="start_a|a+b|bA|a_tail");
    bank.getChildFile("presamp.ini").replaceWithText(rules+formats+"[ALIAS]\nVCPAD=\n");
    check("empty_pad_has_no_separator",aliases(planChineseCvvc(request))=="start_a|ab|bA|a_tail");
    bank.getChildFile("presamp.ini").replaceWithText(rules+formats+"[ALIAS]\nVCPAD=  \n");
    check("padding_spaces_are_preserved",aliases(planChineseCvvc(request))=="start_a|a  b|bA|a_tail");
    changed=request;changed.notes[1].alias="[ba]";
    check("explicit_alias_bypasses_templates",aliases(planChineseCvvc(changed))=="start_a|ba");
    bank.getChildFile("presamp.ini").replaceWithText(rules+formats+"[ALIAS]\nCV=%unknown%\n");
    check("unknown_placeholder_does_not_invent_alias",planChineseCvvc(request).warning.contains("Missing CV: ba"));
    bank.getChildFile("oto.ini").replaceWithText(oto);
    bank.getChildFile("presamp.ini").replaceWithText(rules);
    UtauRenderer::invalidateVoicebankCache();

    std::vector<std::size_t> callbacks;std::vector<std::vector<UtauPhonemeSpan>> spans(2);
    verifyCvvcEnvelopes(folder, request, modelFolder, check);
    auto renderedRequest=request;
    renderedRequest.notePhonemes=[&](std::size_t i,const auto& p){if(i<spans.size())spans[i]=p;};
    renderedRequest.notePiece=[&](std::size_t i,const auto& audio,double rate,double lead,const auto&,const auto&)
    {callbacks.push_back(i);check(i==0?"first_waveform_valid":"second_waveform_valid",audio.getNumSamples()>0&&rate>0&&lead>=0);};
    const auto rendered=UtauRenderer::render(renderedRequest);
    check("classic_render_succeeds",rendered.buffer.getNumSamples()>0&&rendered.buffer.getMagnitude(0,rendered.buffer.getNumSamples())>.01f);
    check("generated_waveforms_fold_into_two_score_notes",callbacks==std::vector<std::size_t>{0,1}&&spans[1].size()==3);
    check("classic_render_uses_cvvc_adapter",rendered.backend.contains("zh-cvvc"));
    auto muted=request;muted.notes[0].gain=0;
    const auto mutedPlan=planChineseCvvc(muted);
    check("muted_context_still_generates_same_aliases",aliases(mutedPlan)==aliases(plan)&&mutedPlan.request.notes[0].gain==0&&mutedPlan.request.notes[1].gain==1);
    std::vector<std::size_t> selectedPieces;
    muted.notePiece=[&](std::size_t i,const auto&,double,double,const auto&,const auto&){selectedPieces.push_back(i);};
    const auto selectedRender=UtauRenderer::render(muted);
    check("classic_context_is_not_synthesized_or_published",selectedPieces==std::vector<std::size_t>{1}
        &&selectedRender.buffer.getMagnitude(0,static_cast<int>(.6*selectedRender.sampleRate))==0);
    if(modelFolder.isDirectory())
    {
        const auto native=renderNsfUtauPhrase(request,modelFolder,{});
        check("hifisampler_actual_render",native.backend.contains("zh-cvvc")&&native.warning.isEmpty()&&native.buffer.getNumSamples()>0&&native.buffer.getMagnitude(0,native.buffer.getNumSamples())>.001f);
        selectedPieces.clear();const auto selectedNative=renderNsfUtauPhrase(muted,modelFolder,{});
        check("hifisampler_context_is_not_synthesized_or_published",selectedPieces==std::vector<std::size_t>{1}
            &&selectedNative.buffer.getNumSamples()>0&&selectedNative.buffer.getMagnitude(0,static_cast<int>(.6*selectedNative.sampleRate))==0);
        auto external=request;
        external.resamplerExecutable=modelFolder.getParentDirectory().getParentDirectory().getChildFile("engines/WCSNDM-0.0803.exe");
        external.requireExternalResampler=true;
        for(auto& note:external.notes)note.flags="K2";
        const auto resampled=UtauRenderer::render(external);
        check("external_resampler_actual_render",resampled.backend.contains("utau-resampler")&&!resampled.backend.contains("fallback")
            &&resampled.warning.isEmpty()&&resampled.buffer.getNumSamples()>0&&resampled.buffer.getMagnitude(0,resampled.buffer.getNumSamples())>.001f);
    }

    TrackData track;track.id="cvvc-track";track.pitchAlgorithm=PitchAlgorithm::utau;track.utauMode=UtauMode::classic;track.voicebankDirectory=bank;track.compose=true;
    ClipData clip;clip.id="cvvc-clip";clip.durationSeconds=1.6;
    NoteData na;na.id="a";na.label="a";na.startSeconds=.3;na.durationSeconds=.5;
    NoteData nb=na;nb.id="b";nb.label="ba";nb.startSeconds=.8;
    clip.notes={na,nb};track.clips={clip};ProjectData data;data.tracks={track};
    project.resetDocument(data);project.dispatchPendingMessages();
    diagnosticSelectTrack(track.id);
    setSize(1280,800);refreshProjectControls();
    check("phonemizer_button_below_stretch",phonemizerButton.isVisible()
        && phonemizerButton.getX()==stretchAlgorithm.getX()
        && phonemizerButton.getY()>stretchAlgorithm.getBottom()
        && !phonemizerButton.getBounds().intersects(autoCrossfadeButton.getBounds()));
    check("editor_display_options_reachable_in_view_menu",diagnosticHasEditorViewItems());
    setSize(900,600);
    check("phonemizer_button_minimum_window",phonemizerButton.isVisible()
        && !phonemizerButton.getBounds().isEmpty()
        && getLocalBounds().contains(phonemizerButton.getBounds())
        && !phonemizerButton.getBounds().intersects(autoCrossfadeButton.getBounds()));
    check("crossfade_button_minimum_window",autoCrossfadeButton.isVisible()
        && getLocalBounds().contains(autoCrossfadeButton.getBounds())
        && !autoCrossfadeButton.getBounds().intersects(advancedEnvelopeButton.getBounds())
        && autoCrossfadeButton.getRight()<phonemizerButton.getX());
    check("minimum_window_preserves_view_menu_display_options",diagnosticHasEditorViewItems());
    setSize(1280,800);
    auto menu=clipContextMenu(clip.id,.9);bool choices=false;
    for(juce::PopupMenu::MenuItemIterator it(menu);it.next();)
        if(it.getItem().text==strings.text("phonemizer.choose"))choices=it.getItem().isEnabled&&it.getItem().subMenu&&it.getItem().subMenu->getNumItems()==3;
    check("standard_utau_menu_offers_two_phonemizers",choices);
    check("manual_aliases_default",!project.snapshot().tracks[0].chineseCvvc);
    outputEngineItemChosen(105,track.id);project.dispatchPendingMessages();
    check("menu_enables_chinese_cvvc",project.snapshot().tracks[0].chineseCvvc);
    check("toolbar_reflects_phonemizer",phonemizerButton.getButtonText().contains(strings.text("phonemizer.zhCvvc")));
    {juce::PNGImageFormat png;auto stream=folder.getChildFile("phonemizer-toolbar.png").createOutputStream();
     if(stream)png.writeImageToStream(createComponentSnapshot(getLocalBounds()),*stream);}
    check("setting_undo_redo",project.undo()&&!project.snapshot().tracks[0].chineseCvvc&&project.redo()&&project.snapshot().tracks[0].chineseCvvc);
    const auto file=folder.getChildFile("cvvc.hjpx");
    ProjectModel reopened;
    check("setting_persists_in_project",project.save(file,error)&&reopened.load(file,error)&&reopened.snapshot().tracks[0].chineseCvvc);
    const auto linearState=project.snapshot();
    check("cvvc_default_crossfade_is_linear",!linearState.tracks[0].cvvcCurveCrossfade);
    outputEngineItemChosen(107,track.id);
    check("crossfade_menu_selects_curve",project.snapshot().tracks[0].cvvcCurveCrossfade);
    check("crossfade_choice_invalidates_audio_and_mix",AudioEngine::diagnosticUtauRenderKey(linearState,clip.id,UtauOutputEngine::resampler,{},{})
        !=AudioEngine::diagnosticUtauRenderKey(project.snapshot(),clip.id,UtauOutputEngine::resampler,{},{})
        &&AudioEngine::utauPhonemeMixContextHash(linearState.tracks[0])!=AudioEngine::utauPhonemeMixContextHash(project.snapshot().tracks[0]));
    check("crossfade_choice_undo_redo",project.undo()&&!project.snapshot().tracks[0].cvvcCurveCrossfade
        &&project.redo()&&project.snapshot().tracks[0].cvvcCurveCrossfade);
    check("crossfade_choice_project_roundtrip",project.save(file,error)&&reopened.load(file,error)&&reopened.snapshot().tracks[0].cvvcCurveCrossfade);
    outputEngineItemChosen(106,track.id);
    auto saved=project.snapshot();
    const auto selected=AudioEngine::diagnosticUtauRequestNotes(saved,clip.id,{"b"});
    check("selection_preserves_muted_neighbors",selected.size()==2&&selected[0].gain==0&&selected[1].gain>0);
    const auto key=AudioEngine::diagnosticUtauRenderKey(saved,clip.id,UtauOutputEngine::resampler,{},{});
    saved.tracks[0].chineseCvvc=false;
    check("phonemizer_in_render_cache_key",key!=AudioEngine::diagnosticUtauRenderKey(saved,clip.id,UtauOutputEngine::resampler,{},{}));
    saved=project.snapshot();bank.getChildFile("presamp.ini").appendText(";cache edit\n");
    check("presamp_changes_invalidate_render_cache",key!=AudioEngine::diagnosticUtauRenderKey(saved,clip.id,UtauOutputEngine::resampler,{},{}));
    #include "CvvcAdvancedEnvelopeSmoke.inc"
    #include "CvvcBoundarySmoke.inc"
    project.setTrackUtauMode(track.id,UtauMode::jie);project.dispatchPendingMessages();
    menu=clipContextMenu(clip.id,.9);bool disabled=false;
    for(juce::PopupMenu::MenuItemIterator it(menu);it.next();)
        if(it.getItem().text==strings.text("phonemizer.choose"))disabled=!it.getItem().isEnabled;
    check("regional_modes_do_not_enable_standard_phonemizer",disabled);
    outputEngineItemChosen(104,track.id);
    check("regional_mode_cannot_change_saved_phonemizer",project.snapshot().tracks[0].chineseCvvc);
    check("regional_mode_hides_toolbar_phonemizer",!phonemizerButton.isVisible());
    project.setTrackUtauMode(track.id,UtauMode::classic);
    project.setTrackPitchAlgorithm(track.id,PitchAlgorithm::world);project.dispatchPendingMessages();
    refreshProjectControls();
    check("native_mode_hides_toolbar_phonemizer",!phonemizerButton.isVisible());
    auto* report=new juce::DynamicObject();report->setProperty("passed",ok);report->setProperty("checks",juce::var(checks));
    folder.getChildFile("report.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok;
}
}
