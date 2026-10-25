// The selected option must actually use source parameters, not a renamed Mel
// clock. Verify monotonic mapping, real model output, cache/cancellation and UV.
bool runWcsNonLoopSmoke(const juce::File& folder, const juce::File& directory, const juce::File& recording)
{
    folder.createDirectory();
    bool ok = true;
    juce::Array<juce::var> checks;
    const auto check = [&](const char* name, bool passed) {
        auto* row = new juce::DynamicObject(); row->setProperty("name", name);
        row->setProperty("passed", passed); checks.add(juce::var(row)); ok &= passed;
        std::cout << name << '=' << passed << '\n';
    };
#if defined(HACHI_HAS_ONNX_ANALYSIS) && HACHI_HAS_ONNX_ANALYSIS
    try
    {
        check("uniform_extension_does_not_fix_attack", std::abs(wcsNonLoopSourceTime({}, .4, 1, 4) - .1) < 1.e-9);
        check("uniform_compression", std::abs(wcsNonLoopSourceTime({}, .2, 1, .5) - .4) < 1.e-9);
        bool forward = true; double previous = -1;
        for (int i = 0; i <= 1000; ++i)
        {
            const auto time = wcsNonLoopSourceTime({}, i * .004, 1, 4);
            forward &= time >= previous && time <= 1; previous = time;
        }
        check("extension_never_loops_or_reverses", forward && previous == 1);
        const std::vector<NsfHifiganTimeMapPoint> unequal {{0,0},{1,.2},{4,1}};
        check("piecewise_stretch_preserved", std::abs(wcsNonLoopSourceTime(unequal, 2.5, 1, 4) - .6) < 1.e-9);
        check("source_pitch_inverse_uses_same_clock", std::abs(wcsNonLoopTargetTime(unequal, .6, 1, 4) - 2.5) < 1.e-9);
        const std::vector<NsfHifiganTimeMapPoint> jump {{0,0},{1,.2},{1,.7},{2,1}};
        check("deleted_source_jump_is_right_continuous", wcsNonLoopSourceTime(jump, .999, 1, 2) < .2
            && std::abs(wcsNonLoopSourceTime(jump, 1, 1, 2) - .7) < 1.e-9);
        check("source_jump_inverse_is_finite", std::abs(wcsNonLoopTargetTime(jump, .4, 1, 2) - 1) < 1.e-9);

        constexpr int rate = 44100;
        juce::AudioBuffer<float> source(1, rate);
        for (int i = 0; i < rate; ++i)
        {
            const auto time = double(i) / rate;
            const auto phase = juce::MathConstants<double>::twoPi * 220. * time;
            source.setSample(0, i, float((.04 + .16 * time) * (std::sin(phase) + .2 * std::sin(phase * 2))));
        }
        const auto render = [&](double duration, const std::vector<NsfHifiganTimeMapPoint>& map,
                                bool protect, const std::vector<float>& sourcePitch, const std::vector<float>& targetPitch,
                                const std::function<bool()>& cancel = {}) {
            return NsfHifiganRenderer::render(source, rate, int(std::llround(duration * rate)), 5,
                targetPitch, {}, map, directory, {InferenceBackend::cpu,-1,2},
                NsfHifiganStretchOrder::wcsNonLoop, false, {}, cancel, protect, false, true, sourcePitch);
        };
        const auto valid = [](const NsfHifiganRenderResult& result, int samples) {
            if (!result.usedModel || !result.usedWcsParameters || result.buffer.getNumSamples() != samples) return false;
            for (int i = 0; i < samples; ++i) if (!std::isfinite(result.buffer.getSample(0, i))) return false;
            return result.buffer.getRMSLevel(0, 0, samples) > .001f;
        };
        const auto save = [&](const juce::String& name, const juce::AudioBuffer<float>& buffer, double sampleRate) {
            auto stream = folder.getChildFile(name).createOutputStream();
            juce::WavAudioFormat format;
            std::unique_ptr<juce::AudioFormatWriter> writer(stream ? format.createWriterFor(stream.release(), sampleRate, 1, 24, {}, 0) : nullptr);
            return writer && writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
        };
        clearWcsNonLoopFeatureCache();
        std::vector<float> src(801,57), dst(801,64);
        auto extended = render(4, {{0,0},{4,1}}, false, src, dst);
        check("real_model_fourfold_extension", valid(extended, rate * 4));
        check("first_render_analyses_source", extended.usedWcsParameters && !extended.reusedWcsSourceAnalysis);
        if (extended.error.isNotEmpty()) std::cout << "render_error=" << extended.error << '\n';
        if (extended.buffer.getNumSamples() == rate * 4)
        {
            const auto early = extended.buffer.getRMSLevel(0, rate / 2, rate / 2);
            const auto late = extended.buffer.getRMSLevel(0, rate * 3, rate / 2);
            check("extension_retains_forward_source_envelope", late > early * 1.3f);
            check("extended_tail_not_silent", extended.buffer.getRMSLevel(0, rate * 7 / 2, rate / 3) > .005f);
            check("extended_audio_written", save("wcs-fourfold.wav", extended.buffer, rate));
            int bestLag = 0; double bestCorrelation = -2;
            const auto* samples = extended.buffer.getReadPointer(0) + rate * 2;
            for (int lag = rate / 420; lag <= rate / 280; ++lag)
            {
                double xy = 0, xx = 0, yy = 0;
                for (int i = 0; i < 8192; ++i) { xy += samples[i]*samples[i+lag]; xx += samples[i]*samples[i]; yy += samples[i+lag]*samples[i+lag]; }
                const auto correlation = xy / std::sqrt(std::max(1.e-20,xx*yy));
                if (correlation > bestCorrelation) { bestCorrelation = correlation; bestLag = lag; }
            }
            check("target_pitch_not_applied_twice", bestLag > 0 && std::abs(double(rate)/bestLag-midiToHz(64))<6.0 && bestCorrelation>.65);
        }
        dst.assign(801, 69);
        auto changed = render(4, {{0,0},{4,1}}, false, src, dst);
        check("pitch_edit_reuses_source_parameters", valid(changed, rate * 4) && changed.reusedWcsSourceAnalysis);
        auto repeated = render(4, {{0,0},{4,1}}, false, src, dst);
        check("unchanged_render_reuses_decoder_chunks", valid(repeated, rate * 4) && repeated.reusedChunks > 0 && repeated.inferredChunks == 0);
        auto compressed = render(.25, {{0,0},{.25,1}}, false, std::vector<float>(51,57), std::vector<float>(51,64));
        check("real_model_fourfold_compression", valid(compressed, rate / 4));
        auto uneven = render(2, {{0,0},{.7,.15},{2,1}}, false, std::vector<float>(401,57), std::vector<float>(401,64));
        check("real_model_piecewise_extension", valid(uneven, rate * 2));
        auto cut = render(2, jump, false, std::vector<float>(401,57), std::vector<float>(401,64));
        check("real_model_deleted_source_jump", valid(cut, rate * 2));
        auto hn = render(2, {{0,0},{2,1}}, true, std::vector<float>(401,57), std::vector<float>(401,64));
        check("optional_harmonic_noise_protection_compatible", valid(hn, rate * 2) && hn.usedHarmonicNoise);
        auto old = NsfHifiganRenderer::render(source, rate, rate * 4, 5, std::vector<float>(801,64), {}, {{0,0},{4,1}},
            directory, {InferenceBackend::cpu,-1,2}, NsfHifiganStretchOrder::hifiShifterMel, false, {}, {}, false, false, true);
        double difference = 0;
        if (old.usedModel && old.buffer.getNumSamples() == extended.buffer.getNumSamples())
            for (int i = 0; i < old.buffer.getNumSamples(); ++i)
                difference += std::abs(old.buffer.getSample(0,i) - extended.buffer.getSample(0,i));
        check("wcs_parameters_distinct_from_mel_interpolation", old.usedModel && !old.usedWcsParameters && difference > 1);

        // UV region inside a note remains unpitched, including with protection
        // switched off: classification is not a licence to synthesize fake F0.
        auto sourcePitch = std::vector<float>(401,57), targetPitch = std::vector<float>(401,64);
        for (int i = 140; i < 200; ++i) sourcePitch[std::size_t(i)] = targetPitch[std::size_t(i)] = 0;
        auto uv = render(2, {{0,0},{2,1}}, false, sourcePitch, targetPitch);
        check("unpitched_source_regions_supported", valid(uv, rate * 2));
        WcsFeatureConfig c {rate, 512, 2048, 2048, 128, filterbank(Config{})};
        const auto wave = stableMonoInput(source);
        const auto reference = buildMelWithHop(wave, Config{}, 512);
        std::vector<float> f0(173,330);
        for (std::size_t i = 0; i < f0.size(); ++i)
            if (i * 512. / rate >= .7 && i * 512. / rate < 1.) f0[i] = 0;
        const auto features = buildWcsNonLoopMel(wave,c,2,5,sourcePitch,f0,{{0,0},{2,1}},reference.values,reference.frames,[]{});
        check("zero_f0_not_recreated_by_parameter_interpolation", f0[70] == 0 && features.values.size() == f0.size() * 128);
        auto cancelled = render(2, {{0,0},{2,1}}, false, sourcePitch, targetPitch, []{ return true; });
        check("cancelled_render_does_not_publish_audio", !cancelled.usedModel && cancelled.buffer.getNumSamples() == 0 && cancelled.error.containsIgnoreCase("cancel"));
        int polls = 0; bool cancelledDuringAnalysis = false;
        clearWcsNonLoopFeatureCache();
        try
        {
            std::vector<float> testF0(173,330);
            buildWcsNonLoopMel(wave,c,2,5,sourcePitch,testF0,{{0,0},{2,1}},reference.values,reference.frames,[&] {
                if (++polls == 3) throw std::runtime_error("cancel at analysis block");
            });
        }
        catch (const std::exception&) { cancelledDuringAnalysis = polls == 3; }
        check("source_analysis_obeys_cancellation", cancelledDuringAnalysis);
        auto bad = render(2, {{0,0},{1,.8},{2,.2}}, false, sourcePitch, targetPitch);
        check("reverse_map_rejected_without_silent_fallback", !bad.usedModel && bad.error.contains("forward"));
        auto noPitch = render(2, {{0,0},{2,1}}, false, {}, targetPitch);
        check("missing_analysis_rejected_without_substitution", !noPitch.usedModel && !noPitch.error.isEmpty());

        const auto fullSource = source;
        source.setSize(1,1200,true);
        auto shortExtended = render(1200.*8/rate,{{0,0},{1200.*8/rate,1200./rate}},false,
            std::vector<float>(50,57),std::vector<float>(50,64));
        check("short_source_eightfold_extension", valid(shortExtended,9600));
        check("short_source_extended_tail_is_present", shortExtended.usedModel && shortExtended.buffer.getRMSLevel(0,8000,1000)>.001f);
        source.setSize(1,rate); source.clear();
        auto silence = render(2,{{0,0},{2,1}},false,std::vector<float>(401,0),std::vector<float>(401,64));
        check("silence_stays_silent_with_authored_pitch",silence.usedModel && silence.usedWcsParameters
            && silence.buffer.getRMSLevel(0,0,silence.buffer.getNumSamples())<.0001f);
        source=fullSource;

        if (recording.existsAsFile())
        {
            juce::AudioFormatManager formats; formats.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(recording));
            check("recording_readable", reader != nullptr);
            if (reader)
            {
                juce::AudioBuffer<float> input(int(reader->numChannels),int(reader->lengthInSamples));
                reader->read(&input,0,input.getNumSamples(),0,true,true);
                juce::Array<juce::File> models;
                directory.getParentDirectory().getChildFile("fcpe").findChildFiles(models,juce::File::findFiles,true,"*.onnx");
                juce::String error;
                const auto analysed = FcpeAnalyzer::analyse(recording,models.isEmpty() ? juce::File{} : models[0],
                    {InferenceBackend::cpu,-1,2},error);
                check("recording_source_pitch_analysis", !analysed.empty() && error.isEmpty());
                if (analysed.empty()) throw std::runtime_error(error.toStdString());
                const auto duration = double(input.getNumSamples()) / reader->sampleRate;
                const auto outputDuration = duration * 1.5;
                std::vector<float> original(std::size_t(std::ceil(outputDuration / .005)) + 1), target(original.size());
                std::size_t cursor = 0;
                for (std::size_t i = 0; i < original.size(); ++i)
                {
                    const auto time = i * .005 / 1.5;
                    while (cursor + 1 < analysed.size() && analysed[cursor + 1].timeSeconds <= time) ++cursor;
                    original[i] = analysed[cursor].voiced ? analysed[cursor].midi : 0;
                    target[i] = original[i] > 0 ? original[i] + 2.f : 0.f;
                }
                const auto started = juce::Time::getMillisecondCounterHiRes();
                auto real = NsfHifiganRenderer::render(input,reader->sampleRate,int(std::llround(outputDuration * reader->sampleRate)),5,
                    target,{},{{0,0},{outputDuration,duration}},directory,{InferenceBackend::cpu,-1,2},
                    NsfHifiganStretchOrder::wcsNonLoop,false,{}, {},true,false,true,original);
                std::cout << "recording_render_ms=" << juce::Time::getMillisecondCounterHiRes() - started << '\n';
                check("real_recording_stretched_and_pitch_shifted", valid(real,int(std::llround(outputDuration * reader->sampleRate))));
                if (real.error.isNotEmpty()) std::cout << "recording_error=" << real.error << '\n';
                check("real_recording_audio_written", real.usedModel && save("recording-wcs-1.5x.wav",real.buffer,reader->sampleRate));
                auto again = NsfHifiganRenderer::render(input,reader->sampleRate,int(std::llround(outputDuration * reader->sampleRate)),5,
                    target,{},{{0,0},{outputDuration,duration}},directory,{InferenceBackend::cpu,-1,2},
                    NsfHifiganStretchOrder::wcsNonLoop,false,{}, {},true,false,true,original);
                check("medium_recording_reuses_source_analysis", again.usedModel && again.reusedWcsSourceAnalysis);
            }
        }
    }
    catch (const std::exception& error) { std::cout << "exception=" << error.what() << '\n'; check("exception_free", false); }
#else
    juce::ignoreUnused(directory, recording); check("onnx_enabled", false);
#endif
    auto* report = new juce::DynamicObject(); report->setProperty("passed",ok); report->setProperty("checks",checks);
    folder.getChildFile("validation.json").replaceWithText(juce::JSON::toString(juce::var(report),true));
    return ok;
}
