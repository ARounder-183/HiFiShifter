// Cross-process cache probe: exact float32 PCM permits C++/Python interchange checks.
bool runHnsepCacheProbe(const juce::File& folder, const juce::File& directory,
                       const juce::File& pcmFile, bool force)
{
    folder.createDirectory();
    auto* report = new juce::DynamicObject();
    bool ok = false;
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    try {
        const auto files = modelFiles(directory);
        juce::MemoryBlock input;
        if (!files || !pcmFile.loadFileAsData(input) || input.isEmpty() || input.getSize() % 4 != 0)
            throw std::runtime_error("Invalid probe model/PCM");
        std::vector<float> wave(input.getSize()/4);
        std::memcpy(wave.data(), input.getData(), input.getSize());
        auto separated = hifiSeparate(wave, *files, {}, force);
        ok = separated.harmonic.size() == wave.size()
            && folder.getChildFile("harmonic.f32").replaceWithData(separated.harmonic.data(), separated.harmonic.size()*4);
        report->setProperty("cache_hit", separated.cacheHit);
    } catch (const std::exception& e) { report->setProperty("error", e.what()); }
#endif
    report->setProperty("ok", ok);
    folder.getChildFile("cache-probe.json").replaceWithText(juce::JSON::toString(juce::var(report), true));
    return ok;
}

// Real ONNX + numerical DSP integration checks, invoked by --smoke-hifisampler.
bool runHifisamplerSmoke(const juce::File& folder, const juce::File& directory)
{
    folder.createDirectory(); bool ok = true;
    juce::Array<juce::var> checks;
    const auto check = [&](const char* name, bool passed) {
        auto* v = new juce::DynamicObject(); v->setProperty("name", name); v->setProperty("passed", passed);
        checks.add(juce::var(v)); ok &= passed;
    };
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    try {
        const auto flags = HifisamplerFlags::parse("/g-900g600Hb600Hv190Ht-150HG120P-50t1800A-140GHe");
        check("ranges_and_first_occurrence", flags.get("g") == -600 && flags.get("Hb") == 500
            && flags.get("Hv") == 150 && flags.get("Ht") == -100 && flags.get("HG") == 100
            && flags.get("P") == 0 && flags.get("t") == 1200 && flags.get("A") == -100);
        check("bare_switches", flags.force && flags.loop);
        const auto plain = HifisamplerFlags::parse("");
        check("empty_flags_defaults", plain.get("g") == 0 && plain.get("Hb") == 100 && plain.get("Hv") == 100
            && plain.get("Ht") == 0 && !plain.force && !plain.normalize && !plain.loop);
        check("bare_p_uses_100", HifisamplerFlags::parse("P").normalize && HifisamplerFlags::parse("P").get("P") == 100);
        check("wcs_flags_do_not_alias_hifi", HifisamplerFlags::parse("Mt50RG20HF2V2").unsupported.size() == 4
            && !HifisamplerFlags::parse("Mt50RG20HF2V2").force);
        const auto files = modelFiles(directory); check("vocoder_available", files.has_value());
        if (!files) throw std::runtime_error("missing vocoder");
        juce::String configError; const auto config = readConfig(files->config, configError);
        if (!config) throw std::runtime_error("invalid vocoder config");
        std::vector<float> wave(44100);
        juce::Random random(707);
        for (std::size_t i = 0; i < wave.size(); ++i) {
            const auto time = i/44100.0;
            wave[i] = static_cast<float>((.15+.035*std::sin(time*9)) * std::sin(time*juce::MathConstants<double>::twoPi*220)
                + .04*std::sin(time*juce::MathConstants<double>::twoPi*660) + .025*(random.nextFloat()-.5));
        }
        const auto write = [&](const char* name, const std::vector<float>& data) {
            folder.getChildFile(name).replaceWithData(data.data(), data.size()*sizeof(float));
        };
        write("source.f32", wave);
        const auto restored = hifiIstft(hifiStft(wave, 2048, 512), wave.size());
        float error = 0; for (std::size_t i = 0; i < wave.size(); ++i) error = std::max(error, std::abs(restored[i]-wave[i]));
        check("stft_roundtrip", error < 1.0e-5f);
        auto original = hifiFeatures(wave, {}, plain, *config);
        write("mel-g0.f32", original.mel.values);
        auto shifted = hifiFeatures(wave, {}, HifisamplerFlags::parse("g275"), *config);
        write("mel-g275.f32", shifted.mel.values);
        check("g_changes_mel_without_changing_time", shifted.mel.frames == original.mel.frames && shifted.mel.values != original.mel.values);
        auto tension = hifiTension(wave, 60, *config); write("tension60.f32", tension);
        check("tension_keeps_length", tension.size() == wave.size() && tension != wave);
        auto growled = wave;
        hifiGrowl(growled, 44100, { HifisamplerFlags::parse("HG65") }, 1);
        write("growl65.f32", growled);
        check("growl_keeps_length", growled.size() == wave.size() && growled != wave);
        auto normalized = wave;
        hifiNormalize(normalized, 44100, { HifisamplerFlags::parse("P100") }, 1, false);
        write("normalized100.f32", normalized);
        check("p_targets_minus16_lufs", std::abs(hifiLoudness(normalized, 44100)+16) < .02);
        auto p0 = wave; hifiNormalize(p0, 44100, { HifisamplerFlags::parse("P0") }, 1, false);
        check("p_zero_is_identity", p0 == wave);
        auto sep = hifiSeparate(wave, *files, {}, true); write("harmonic.f32", sep.harmonic);
        check("hnsep_real_model_output", sep.harmonic.size() == wave.size() && hifiPeak(sep.harmonic) > .01f && !sep.cacheHit);
        auto cached = hifiSeparate(wave, *files, {}, false);
        check("hnsep_cache_reused", cached.cacheHit && cached.harmonic == sep.harmonic);
        check("g_switch_forces_hnsep_regeneration", !hifiSeparate(wave, *files, {}, true).cacheHit);
        const auto voice = hifiFeatures(wave, sep.harmonic, HifisamplerFlags::parse("Hb0"), *config);
        const auto breath = hifiFeatures(wave, sep.harmonic, HifisamplerFlags::parse("Hv0"), *config);
        write("mel-voice.f32", voice.mel.values); write("mel-breath.f32", breath.mel.values);
        check("hb_hv_adjust_independent_components", voice.mel.values != original.mel.values
            && breath.mel.values != original.mel.values && voice.mel.values != breath.mel.values);
        UtauNoteRenderSpec spec; spec.flags = "g25Hb100"; spec.flagSplit = true; spec.regionFlags[1] = "g-150Hb25";
        const std::array<double,4> lengths { .1,.2,.4,.3 };
        check("region_overrides_note_and_inherits_remaining", hifiFlagsAt(spec,lengths,4,.2,0).get("g") == -150
            && hifiFlagsAt(spec,lengths,4,.8,0).get("g") == 25);
        spec.flagCurve = true; spec.flagCurves = {{ "HIFI:g", {{-.1,-100},{.9,100}} }};
        check("curve_overrides_region_in_note_time", std::abs(hifiFlagsAt(spec,lengths,4,.5,-.1).get("g")) < .001f);
        NsfUtauSampleTiming timing { 0,1,.1,.1,.03,1,true,{.1,.2,.4,.3},"CVVC" };
        const auto plan = buildNsfUtauNotePlan(timing,.3,1.5,1,0);
        std::vector<NsfHifiganTimeMapPoint> map;
        for (const auto& p : plan.timeMap) map.push_back({p.targetSeconds,p.sourceSeconds-plan.sourceStartSeconds});
        check("oto_regions_preserved", plan.valid && plan.usesRegions && plan.regionCount == 4 && plan.regionClasses == "CVVC");
        check("loop_does_not_repeat_consonant", hifiSourceTime(map,plan,.05,44100,true) == hifiSourceTime(map,plan,.05,44100,false));
        const auto probe = (map[1].targetSeconds+map[2].targetSeconds)*.5;
        check("he_changes_stretched_region_mapping", std::abs(hifiSourceTime(map,plan,probe,44100,true)-hifiSourceTime(map,plan,probe,44100,false)) > .001);
        // Check full source coverage and monotone progress, not just output length:
        // the previous globally enabled reflection loop passed duration checks.
        const auto oncePerSpan = [&](const NsfUtauNotePlan& candidate) {
            if (!candidate.valid || candidate.timeMap.size() < 2) return false;
            const auto& anchors = candidate.timeMap;
            auto previous = anchors.front().sourceSeconds;
            for (int i = 0; i <= 2048; ++i) {
                const auto position = candidate.outputSeconds*i/2048;
                const auto mapped = hifiSourceTime(anchors,candidate,position,44100,plain.loop);
                if (mapped+1.0e-8 < previous || mapped < candidate.sourceStartSeconds-1.0e-8
                    || mapped > candidate.sourceEndSeconds+1.0e-8) return false;
                previous = mapped;
            }
            if (std::abs(previous-candidate.sourceEndSeconds) > 1.0e-8) return false;
            for (std::size_t i = 1; i < anchors.size(); ++i) {
                const auto& a = anchors[i-1]; const auto& b = anchors[i];
                if (b.targetSeconds-a.targetSeconds < 1.0e-8) continue;
                for (const auto fraction : { .01,.25,.5,.75,.99 }) {
                    const auto mapped = hifiSourceTime(anchors,candidate,
                        a.targetSeconds+(b.targetSeconds-a.targetSeconds)*fraction,44100,plain.loop);
                    if (std::abs(mapped-(a.sourceSeconds+(b.sourceSeconds-a.sourceSeconds)*fraction)) > 1.0e-8)
                        return false;
                }
            }
            return true;
        };
        auto classic = timing; classic.hasRegions = false;
        check("classic_long_note_stretches_source_once",oncePerSpan(buildNsfUtauNotePlan(classic,.3,4,1,0)));
        check("classic_compressed_vowel_keeps_full_source",oncePerSpan(buildNsfUtauNotePlan(classic,.3,.2,1,0)));
        for (const auto count : {2,3,4}) {
            auto regions = timing;
            regions.mouClasses = count == 2 ? "CV" : count == 3 ? "CVS" : "CVVC";
            regions.regionSeconds = count == 2 ? std::array<double,4>{.1,.9,0,0}
                : count == 3 ? std::array<double,4>{.1,.5,.4,0} : timing.regionSeconds;
            const auto name = "oto_"+juce::String(count)+"_regions_stretch_each_source_once";
            check(name.toRawUTF8(),oncePerSpan(buildNsfUtauNotePlan(regions,.3,4,1,0)));
        }
        const std::array<double,3> manual { .15,.35,.8 };
        check("manual_regions_keep_linear_stretch",oncePerSpan(buildNsfUtauNotePlan(timing,.3,4,1,0,&manual)));
        const auto velocityPlan = buildNsfUtauNotePlan(timing,.3,4,.5,0);
        check("consonant_velocity_and_vowels_stay_independent",oncePerSpan(velocityPlan)
            && std::abs(velocityPlan.outputRegionSeconds[0]-.1) < 1.0e-8
            && std::abs(velocityPlan.outputRegionSeconds[3]-.15) < 1.0e-8);
        NsfUtauNotePlan skipped; skipped.valid=true; skipped.usesRegions=true; skipped.regionCount=4;
        skipped.regionClasses="CVVC"; skipped.sourceEndSeconds=1; skipped.outputSeconds=3;
        skipped.timeMap={{0,0},{.1,.1},{.1,.3},{2.9,.7},{3,1}};
        check("zero_target_region_skips_source_without_loop",oncePerSpan(skipped)
            && std::abs(hifiSourceTime(skipped.timeMap,skipped,.1,44100,false)-.3) < 1.0e-8);
        auto silentSpan = skipped; silentSpan.timeMap={{0,0},{.1,.1},{1,.1},{2.9,.7},{3,1}};
        check("zero_source_region_holds_boundary_without_repeating_neighbors",oncePerSpan(silentSpan));
        auto sClassPlan = plan; sClassPlan.regionClasses="SVVC";
        const std::vector<NsfHifiganTimeMapPoint> stretchedOnset {{0,0},{.3,.1},{.6,.3},{1,.7},{1.3,1}};
        check("he_does_not_loop_s_onset_when_lead_exceeds_source",hifiSourceTime(stretchedOnset,sClassPlan,.25,44100,true)
            == hifiSourceTime(stretchedOnset,sClassPlan,.25,44100,false));
        auto vClassPlan=sClassPlan;vClassPlan.regionClasses="VVVC";
        check("he_still_loops_explicitly_stretched_voiced_onset",std::abs(hifiSourceTime(stretchedOnset,vClassPlan,.25,44100,true)
            -hifiSourceTime(stretchedOnset,vClassPlan,.25,44100,false))>.001);
        juce::AudioBuffer<float> source(1,44100); std::copy(wave.begin(),wave.end(),source.getWritePointer(0));
        const auto sampleFile = folder.getChildFile("a.wav");
        { juce::WavAudioFormat format; auto stream=sampleFile.createOutputStream();
          std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,24,{},0));
          if (!writer || !writer->writeFromAudioSampleBuffer(source,0,44100)) throw std::runtime_error("could not write test recording"); }
        const auto render = [&](const char* text, const char* name) {
            UtauNoteRenderSpec n; n.flags=text;
            const auto audio = synthesizeNsfUtauNote(sampleFile, plan, 60, {{0,-10},{.3,35},{.8,-20},{1.5,0}}, directory, {}, {}, &n);
            check(name, audio.usedModel && audio.audio.getNumSamples() == static_cast<int>(std::lround(plan.outputSeconds*44100)));
            if (!audio.usedModel) throw std::runtime_error(audio.error.toStdString());
            const auto outputFile=folder.getChildFile(juce::String(name)+".wav");
            juce::WavAudioFormat format; auto stream=outputFile.createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,24,{},0));
            if (writer) writer->writeFromAudioSampleBuffer(audio.audio,0,audio.audio.getNumSamples());
            return audio;
        };
        // A nonzero OTO cut must retain source frames on both sides. Check the
        // actual sampling helper against an independent full-recording Mel grid.
        NsfUtauNotePlan contextPlan; contextPlan.valid=true; contextPlan.sourceStartSeconds=.2;
        contextPlan.sourceEndSeconds=.8; contextPlan.outputSeconds=.9;
        contextPlan.timeMap={{0,.2},{.9,.8}};
        const auto targetHop=static_cast<double>(config->hop)/config->sampleRate;
        const auto contextFrames=static_cast<int>(std::ceil(.9/targetHop))+32;
        std::vector<float> contextMel(static_cast<std::size_t>(config->melBands)*contextFrames);
        std::vector<double> contextTimes;
        for(int j=0;j<contextFrames;++j) {
            const auto time=(j-16+.5)*targetHop;
            const auto at=hifiContextSourceTime(contextPlan.timeMap,contextPlan,time,config->sampleRate,false);
            contextTimes.push_back(at);
            for(int band=0;band<config->melBands;++band)
                contextMel[static_cast<std::size_t>(band)*contextFrames+j]=hifiSampleMel(original.mel,band,at,config->sampleRate);
        }
        folder.getChildFile("context-mel.f32").replaceWithData(contextMel.data(),contextMel.size()*sizeof(float));
        folder.getChildFile("context-source-times.f64").replaceWithData(contextTimes.data(),contextTimes.size()*sizeof(double));
        check("oto_context_reads_real_recording_before_and_after_cut",contextTimes.front()<.2 && contextTimes.back()>.8);
        check("mel_output_uses_half_hop_centres",std::abs(contextTimes[16]-(.2+.5*targetHop*.6/.9))<1.e-12);
        check("context_preserves_zero_duration_region_skip",std::abs(hifiContextSourceTime({{0,.1},{0,.2},{.9,.8}},contextPlan,-.01,44100,false)-(.2-.01*.6/.9))<1.e-12);
        check("context_never_reflects_explicit_he_tail",std::abs(hifiContextSourceTime(contextPlan.timeMap,contextPlan,.95,44100,true)-(.8+.05*.6/.9))<1.e-12);
        const auto contextAudio=synthesizeNsfUtauNote(sampleFile,contextPlan,60,{},directory,{}, {},nullptr);
        check("nonzero_oto_cut_keeps_exact_output_length",contextAudio.usedModel && contextAudio.audio.getNumSamples()==39690);
        juce::AudioBuffer<float> changedContext(source);
        changedContext.clear(0,0,static_cast<int>(.2*44100));
        changedContext.clear(0,static_cast<int>(.8*44100),static_cast<int>(.2*44100));
        const auto changedFile=folder.getChildFile("changed-context.wav");
        {juce::WavAudioFormat format;auto stream=changedFile.createOutputStream();
         std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,24,{},0));
         if(!writer || !writer->writeFromAudioSampleBuffer(changedContext,0,44100)) throw std::runtime_error("context fixture write failed");}
        const auto changedAudio=synthesizeNsfUtauNote(changedFile,contextPlan,60,{},directory,{}, {},nullptr);
        double contextDifference=0;
        if(contextAudio.usedModel && changedAudio.usedModel)
            for(int j=0;j<contextAudio.audio.getNumSamples();++j)
                contextDifference+=std::abs(contextAudio.audio.getSample(0,j)-changedAudio.audio.getSample(0,j));
        check("audio_outside_oto_cut_reaches_actual_decoder",changedAudio.usedModel && contextDifference>1.e-3);
        const auto plainDecoded = render("", "empty_flags_real_decode");
        check("long_decode_has_audio_at_middle_and_tail",plainDecoded.audio.getRMSLevel(0,22050,4410) > .001f
            && plainDecoded.audio.getRMSLevel(0,57330,4410) > .001f);
        render("g275t1200A40HG65P100Hb25Hv125Ht60HeG", "all_ten_flags_real_decode");
        const auto mute = render("Hb0Hv0P100", "both_components_muted_real_decode");
        check("both_components_zero_are_silent", mute.audio.getMagnitude(0,0,mute.audio.getNumSamples()) == 0);
        // A rising-amplitude recording must rise once across a four-second vowel.
        // Constant tones cannot distinguish a loop from real stretching.
        juce::AudioBuffer<float> ramp(1,44100);
        for (int i=0;i<44100;++i) {
            const auto time=i/44100.0;
            ramp.setSample(0,i,static_cast<float>((.015+.3*time)*
                (std::sin(time*juce::MathConstants<double>::twoPi*220)
                 +.25*std::sin(time*juce::MathConstants<double>::twoPi*660))));
        }
        const auto rampFile=folder.getChildFile("rising-source.wav");
        { juce::WavAudioFormat format; auto stream=rampFile.createOutputStream();
          std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,24,{},0));
          if (!writer || !writer->writeFromAudioSampleBuffer(ramp,0,44100)) throw std::runtime_error("could not write rising recording"); }
        const auto rampPlan=buildNsfUtauNotePlan(classic,.3,4,1,0);
        UtauNoteRenderSpec stretchNote;
        const auto stretched=synthesizeNsfUtauNote(rampFile,rampPlan,57,{},directory,{}, {},&stretchNote);
        if (!stretched.usedModel) throw std::runtime_error(stretched.error.toStdString());
        auto loopNote=stretchNote; loopNote.flags="He";
        const auto looped=synthesizeNsfUtauNote(rampFile,rampPlan,57,{},directory,{}, {},&loopNote);
        if (!looped.usedModel) throw std::runtime_error(looped.error.toStdString());
        const auto rmsAt=[&](const auto& audio,double time) {
            return audio.audio.getRMSLevel(0,static_cast<int>(time*44100),static_cast<int>(.2*44100));
        };
        juce::Array<juce::var> stretchRms, loopRms;
        float previousRms=0; bool rising=true;
        for (const auto time : {.35,1.4,2.5,3.5}) {
            const auto rms=rmsAt(stretched,time); stretchRms.add(rms); loopRms.add(rmsAt(looped,time));
            rising &= rms > previousRms; previousRms=rms;
        }
        check("real_long_note_keeps_one_rising_source_envelope",rising && rmsAt(stretched,3.5) > 2*rmsAt(stretched,.35));
        check("four_second_decode_is_audible_through_tail",rmsAt(stretched,2.5) > .001f && rmsAt(stretched,3.5) > .001f);
        check("only_explicit_he_loops_real_long_note",rmsAt(stretched,3.5) > 1.3f*rmsAt(looped,3.5));
        for (const auto item : {std::make_pair("rising-stretched.wav",&stretched),std::make_pair("rising-he-loop.wav",&looped)}) {
            juce::WavAudioFormat format; auto stream=folder.getChildFile(item.first).createOutputStream();
            std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.release(),44100,1,24,{},0));
            if (writer) writer->writeFromAudioSampleBuffer(item.second->audio,0,item.second->audio.getNumSamples());
        }
        auto* stretchInfo=new juce::DynamicObject(); stretchInfo->setProperty("rms_times_seconds",juce::var(juce::Array<juce::var>{.35,1.4,2.5,3.5}));
        stretchInfo->setProperty("stretched_rms",juce::var(stretchRms)); stretchInfo->setProperty("explicit_he_rms",juce::var(loopRms));
        folder.getChildFile("stretch-audio-info.json").replaceWithText(juce::JSON::toString(juce::var(stretchInfo),true));
        auto* info = new juce::DynamicObject(); info->setProperty("frames",static_cast<int>(original.mel.frames));
        info->setProperty("mel_bands",config->melBands); info->setProperty("input_lufs",hifiLoudness(wave,44100));
        info->setProperty("normalized_lufs",hifiLoudness(normalized,44100)); info->setProperty("stft_error",error);
        folder.getChildFile("dsp-info.json").replaceWithText(juce::JSON::toString(juce::var(info),true));
    }
    catch (const std::exception& e) { check(e.what(),false); }
#else
    check("ONNX_runtime_missing", false);
#endif
    auto* report=new juce::DynamicObject(); report->setProperty("ok",ok); report->setProperty("checks",juce::var(checks));
    folder.getChildFile("hifisampler-validation.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok;
}
