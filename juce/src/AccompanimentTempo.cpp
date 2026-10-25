#include "AccompanimentTempo.h"
#include "ClipParts.h"
#include <cmath>

namespace hachi
{
namespace
{
juce::var failure(const char* code, const juce::String& detail = {})
{
    auto* value = new juce::DynamicObject();
    value->setProperty("ok", false); value->setProperty("code", code);
    value->setProperty("error", detail); return value;
}
}
AccompanimentTempoTask::AccompanimentTempoTask(ClipData region, juce::File runtime)
    : Thread("Accompaniment tempo"), clip(std::move(region)),
      config(runtime == juce::File() ? runtimeConfig() : runtime) {}
AccompanimentTempoTask::~AccompanimentTempoTask()
{
    cancel(); stopThread(-1); // run() checks cancellation while decoding/waiting.
}
juce::File AccompanimentTempoTask::runtimeConfig()
{
    const auto override = juce::SystemStats::getEnvironmentVariable("HACHI_HAMOOD_RUNTIME", {});
    return override.isNotEmpty() ? juce::File(override)
        : juce::File::getSpecialLocation(juce::File::currentExecutableFile)
            .getSiblingFile("engines").getChildFile("hamood/runtime.json");
}
bool AccompanimentTempoTask::canAnalyse(const ClipData& c)
{
    if (!std::isfinite(c.durationSeconds) || c.durationSeconds <= 0) return false;
    const auto regions = clipSourceRegions(c);
    return !regions.empty() && std::all_of(regions.begin(), regions.end(), [](const auto* part)
    { return part->sourceFile.existsAsFile() && std::isfinite(part->audioLength()) && part->audioLength() > 0; });
}
void AccompanimentTempoTask::run()
{
    try { outcome = analyse(); }
    catch (const std::exception& e) { outcome = failure("worker_error", juce::String::fromUTF8(e.what())); }
    catch (...) { outcome = failure("worker_error"); }
    if (cancelled.load()) outcome = failure("cancelled");
    done.store(true, std::memory_order_release);
}
juce::var AccompanimentTempoTask::analyse()
{
    if (cancelled.load()) return failure("cancelled");
    if (!canAnalyse(clip)) return failure("invalid_audio");
    const auto runtime = juce::JSON::parse(config);
    const auto path = runtime["python"].toString();
    const auto python = path.isNotEmpty() ? config.getParentDirectory().getChildFile(path) : juce::File();
    const auto script = config.getSiblingFile("tempo_worker.py");
    if (!python.existsAsFile() || !script.existsAsFile()) return failure("missing_runtime", config.getFullPathName());
    if (!config.getParentDirectory().getChildFile("models/beat-this-small0.ckpt").existsAsFile())
        return failure("missing_model");
    // Every file here belongs to this task. The directory is random, private
    // to the analysis, and removed on success, error and cancellation.
    struct Scratch
    {
        juce::File folder = juce::File::getSpecialLocation(juce::File::tempDirectory)
            .getChildFile("HachiShifter-tempo-" + juce::Uuid().toString());
        ~Scratch() { folder.deleteRecursively(); }
    } scratch;
    if (scratch.folder.createDirectory().failed()) return failure("worker_error", "Cannot create analysis temporary directory");
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    juce::Array<juce::var> segments;
    int number = 0;
    for (const auto* part : clipSourceRegions(clip))
    {
        if (cancelled.load()) return failure("cancelled");
        const auto start = (clip.parts.empty() ? 0.0 : part->startSeconds) + part->audioStartSeconds;
        if (!std::isfinite(start) || !std::isfinite(part->sourceOffsetSeconds) || part->sourceOffsetSeconds < 0)
            return failure("invalid_audio");
        const auto begin = std::max(0.0, start), end = std::min(clip.durationSeconds, start + part->audioLength());
        if (end <= begin) continue;
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(part->sourceFile));
        if (!reader || reader->sampleRate <= 0 || reader->numChannels == 0) return failure("invalid_audio", part->sourceFile.getFileName());
        const auto sourceBegin = part->sourceOffsetSeconds + begin - start;
        const auto first = static_cast<juce::int64>(std::llround(sourceBegin * reader->sampleRate));
        const auto count = std::min(reader->lengthInSamples - first,
            static_cast<juce::int64>(std::llround((end - begin) * reader->sampleRate)));
        if (first < 0 || count <= 0) continue;
        const auto decoded = scratch.folder.getChildFile("part-" + juce::String(number++) + ".wav");
        auto stream = decoded.createOutputStream(); juce::WavAudioFormat wav;
        if (!stream) return failure("worker_error", "Cannot write analysis audio");
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), reader->sampleRate, 1, 32, {}, 0));
        if (!writer) return failure("worker_error", "Cannot write analysis audio");
        constexpr int blockSize = 32768;
        juce::AudioBuffer<float> block(static_cast<int>(reader->numChannels), blockSize), mono(1, blockSize);
        for (juce::int64 offset = 0; offset < count; offset += blockSize)
        {
            if (cancelled.load()) return failure("cancelled");
            const auto n = static_cast<int>(std::min<juce::int64>(blockSize, count - offset));
            if (!reader->read(&block, 0, n, first + offset, true, true)) return failure("invalid_audio");
            mono.clear();
            for (int c = 0; c < block.getNumChannels(); ++c)
                mono.addFrom(0, 0, block, c, 0, n, 1.0f / block.getNumChannels());
            if (!writer->writeFromAudioSampleBuffer(mono, 0, n)) return failure("worker_error", "Cannot decode analysis audio");
        }
        writer.reset();
        auto* segment = new juce::DynamicObject();
        segment->setProperty("file", decoded.getFullPathName()); segment->setProperty("start", begin);
        segments.add(segment);
    }
    if (segments.isEmpty()) return failure("too_short");
    auto* request = new juce::DynamicObject(); request->setProperty("segments", segments);
    request->setProperty("duration", clip.durationSeconds);
    const auto input = scratch.folder.getChildFile("request.json"), output = scratch.folder.getChildFile("result.json");
    if (!input.replaceWithText(juce::JSON::toString(juce::var(request), true))) return failure("worker_error", "Cannot write analysis request");
    if (cancelled.load()) return failure("cancelled");
    phase.store(1);
    juce::ChildProcess process;
    if (!process.start(juce::StringArray{python.getFullPathName(), "-I", "-B", script.getFullPathName(),
            "--request", input.getFullPathName(), "--output", output.getFullPathName()}, 0))
        return failure("worker_error", "Could not start Beat This! worker");
    const auto started = juce::Time::getMillisecondCounterHiRes();
    while (process.isRunning())
    {
        if (cancelled.load() || juce::Time::getMillisecondCounterHiRes() - started > 1800000)
        {
            process.kill(); process.waitForProcessToFinish(2000);
            return failure(cancelled.load() ? "cancelled" : "timeout");
        }
        juce::Thread::sleep(40);
    }
    if (cancelled.load()) return failure("cancelled");
    const auto result = juce::JSON::parse(output);
    if (!result.isObject()) return failure("worker_error", "Beat This! worker returned no result");
    if (!(bool)result["ok"]) return result;
    const double bpm = result["bpm"];
    if (!std::isfinite(bpm) || bpm <= 0 || (int)result["beat_count"] < 4) return failure("no_beats");
    return result;
}
}
