#include "LegacyTextCodec.h"
#include "UtauRenderer.h"
#include "ChineseCvvcPhonemizer.h"
#include "PlaybackRenderPriority.h"
#include "DiffSingerRenderer.h"
#include "../SampleSettings.h"
#include "../OttoMaterialIO.h"
#include "AmplitudeEnvelopeCurve.h"
#include "AdvancedEnvelope.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <future>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <unordered_map>

#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include <windows.h>
#endif

namespace hachi::backend
{
namespace
{
#if JUCE_WINDOWS
// Model warm-up remains asynchronous, but the daemon belongs to the editor
// which started it. Windows also closes these job handles on abnormal exit,
// preventing an orphaned pythonw from retaining the model indefinitely.
struct OwnedHfDaemon
{
    HANDLE process = nullptr;
    HANDLE job = nullptr;
    int port = 0;
    ~OwnedHfDaemon()
    {
        if (job != nullptr) CloseHandle(job); // KILL_ON_JOB_CLOSE, including descendants.
        if (process != nullptr)
        {
            WaitForSingleObject(process, 2000);
            CloseHandle(process);
        }
    }
};

struct HfDaemonLifetime
{
    std::mutex mutex;
    bool closing = false;
    std::vector<std::unique_ptr<OwnedHfDaemon>> processes;
    ~HfDaemonLifetime() { shutdown(); }
    void shutdown()
    {
        std::vector<std::unique_ptr<OwnedHfDaemon>> owned;
        {
            const std::scoped_lock lock(mutex);
            closing = true;
            owned.swap(processes);
        }
        // Wait only for processes we created; no port owner/name-based kill.
        owned.clear();
    }
};

HfDaemonLifetime& hfDaemonLifetime()
{
    static HfDaemonLifetime state;
    return state;
}
#endif

std::atomic<std::uint64_t>& voicebankCacheRevision()
{
    static std::atomic<std::uint64_t> revision { 0 };
    return revision;
}

constexpr double mixSampleRate = 44'100.0;

struct VoiceSample
{
    juce::File file;
    juce::String alias;
    double offset = 0.0;
    double end = 0.0;
    // The whole recording, so a shifted entry can be kept inside it.
    double fileSeconds = 0.0;
    double preutterance = 0.0;
    double consonant = 0.0;
    double overlap = 0.0;
    float sourceMidi = 60.0f;
    bool hasRegions = false;
    std::array<double, 4> regionSeconds {};
    // HJM is the native annotation after import.  Keep every segment here;
    // the legacy four-region array above is only a compatibility projection
    // for resamplers that still accept that protocol.
    std::vector<NativeSegment> nativeSegments;
    // One letter per region from 谋•OTO; empty unless this voicebank has been
    // annotated and the track is in 谋•UTAU mode.
    juce::String mouClasses;
    std::shared_ptr<const hachi::VoicebankOtoEntry> sourceOto;
};

void projectNativeSegmentsToLegacyRegions(VoiceSample& sample)
{
    if (sample.nativeSegments.empty()) return;
    const auto span = std::max(1.0e-6, sample.end - sample.offset);
    std::vector<double> boundaries { 0.0 };
    for (const auto& segment : sample.nativeSegments)
        boundaries.push_back(juce::jlimit(0.0, span,
            segment.sourceEndSeconds - segment.sourceStartSeconds));
    boundaries.push_back(span);
    std::stable_sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end(),
        [](double left, double right) { return std::abs(left - right) < 1.0e-6; }),
        boundaries.end());
    if (boundaries.size() < 2) return;
    sample.hasRegions = boundaries.size() <= 5;
    if (!sample.hasRegions) return;
    sample.regionSeconds.fill(0.0);
    for (std::size_t index = 1; index < boundaries.size() && index <= 4; ++index)
        sample.regionSeconds[index - 1] = boundaries[index] - boundaries[index - 1];
}

struct RenderedNote
{
    juce::AudioBuffer<float> audio;
    double preutterance = 0.0;
    double overlap = 0.0;
    bool found = false;
    bool external = false;
    bool piano = false;
    bool rest = false;
    // A lyric the voicebank has no sample for: sung as the piano, and still
    // reported as missing.
    bool missing = false;
    std::optional<UtauSampleTiming> timing;
};

struct ExternalResamplerAttempt
{
    juce::AudioBuffer<float> audio;
    juce::String error;
    bool succeeded = false;
};

struct VoicebankIndex
{
    bool nativeAuthority = false;
    std::vector<VoiceSample> samples;
    std::map<juce::String, std::pair<juce::String, juce::String>> prefixMap;
    // The first sample for each alias and for each file name, keyed the way
    // equalsIgnoreCase compares -- character by character in upper case -- so
    // a lookup finds exactly what scanning the samples in order found.  Asked
    // once per note on every layout of the roll, the scan was most of the time
    // an edit took.
    std::unordered_map<std::string, std::size_t> byAlias;
    // Phonemizer output can distinguish phonemes such as a/A or v/V.
    // Keep the legacy folded index for manual-lyric compatibility.
    std::unordered_map<std::string, std::size_t> byExactAlias;
    std::unordered_map<std::string, std::size_t> byFileName;
};

std::string foldedKey(const juce::String& text)
{
    return text.toUpperCase().toStdString();
}

juce::String decodeText(const juce::File& file)
{
    juce::String error;const auto doc=LegacyTextCodec::read(file,error);return doc?doc->text:juce::String();
}

juce::String midiName(float midi)
{
    static constexpr const char* names[] {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    const auto value = juce::jlimit(0, 127, static_cast<int>(std::lround(midi)));
    return juce::String(names[value % 12]) + juce::String(value / 12 - 1);
}

}  // namespace

bool isRestLyric(const juce::String& lyric)
{
    return lyric.trim().equalsIgnoreCase("RR");
}

namespace
{
// The consonant velocity the head of a sample answers to.  Four things are
// about that head and no other part of it: the lead-in, the consonant length
// the roll draws, the velocity the resampler is handed for its fixed range,
// and the local stretcher's own mapping of that range.  All four are region
// 0's, so in 谋 they follow region 0's class -- the velocity stretches a
// consonant, and a first region the entry calls a vowel is not one.  Marked
// regions further in still answer to it; that happens in regionSplit, which
// reads each region's own class.
//
// A two-region entry whose last region is a consonant answers nowhere at all:
// it is not a syllable with a consonant at its front but the join between two
// of them, and the velocity is the front consonant's.
//
// mouClasses is empty outside 谋•UTAU, so no other mode can reach any of this.
constexpr int neutralConsonantVelocity = 100;

int headConsonantVelocity(const VoiceSample& sample, int velocity)
{
    const auto count = sample.mouClasses.length();
    if (count < 2 || count > 4) return velocity;          // no annotation
    // S counts with C here, as it does everywhere else the classes are read.
    return sample.mouClasses[0] == 'V' ? neutralConsonantVelocity : velocity;
}

double adjustedPreutterance(const VoiceSample& sample, int velocity)
{
    // Velocity maps every source position inside the fixed-consonant region
    // by the same scale.  Preutterance is a point on that source timeline,
    // not a duration from which the whole consonant reduction can be
    // subtracted.  The latter can incorrectly collapse a valid lead-in to 0.
    return scaledUtauPreutterance(sample.preutterance, sample.consonant,
        sample.end - sample.offset, velocity);
}

float pitchCentsAt(const UtauNoteRenderSpec& note, double localSeconds)
{
    if (note.timelinePitchCents) return note.timelinePitchCents(localSeconds);
    if (note.pitchCurve.empty()) return 0.0f;
    const auto right = std::lower_bound(note.pitchCurve.begin(), note.pitchCurve.end(), localSeconds,
        [](const UtauPitchPoint& point, double time) { return point.timeSeconds < time; });
    if (right == note.pitchCurve.begin()) return right->cents;
    if (right == note.pitchCurve.end()) return note.pitchCurve.back().cents;
    const auto& left = *std::prev(right);
    const auto amount = right->timeSeconds > left.timeSeconds
        ? static_cast<float>(juce::jlimit(0.0, 1.0,
            (localSeconds - left.timeSeconds) / (right->timeSeconds - left.timeSeconds)))
        : 0.0f;
    return left.cents + (right->cents - left.cents) * amount;
}

juce::String encodePitchbend(const UtauNoteRenderSpec& note, double bpm,
                             double preutteranceSeconds, double outputSeconds)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto safeBpm = juce::jlimit(20.0, 400.0, bpm);
    const auto interval = 60.0 / (safeBpm * 96.0);
    const auto count = std::max(1, static_cast<int>(std::ceil(outputSeconds / interval)));
    const auto roundedMidi = static_cast<float>(std::lround(note.midiNote));
    const auto baseOffsetCents = (note.midiNote - roundedMidi) * 100.0f;
    juce::String encoded;
    encoded.preallocateBytes(static_cast<std::size_t>(count * 2));
    for (int index = 0; index < count; ++index)
    {
        const auto outputTime = static_cast<double>(index) * interval;
        const auto curveStart = note.pitchCurve.empty()
            ? 0.0 : std::min(0.0, note.pitchCurve.front().timeSeconds);
        const auto curveEnd = note.pitchCurve.empty()
            ? std::max(0.0, note.durationSeconds)
            : std::max(note.durationSeconds, note.pitchCurve.back().timeSeconds);
        auto localTime = note.timelinePitchCents
            ? outputTime - preutteranceSeconds
            : juce::jlimit(curveStart, curveEnd, outputTime - preutteranceSeconds);
        // retimeLeadIn later maps the natural head onto the requested head.
        // Read pitch at that final timeline position, not the pre-stretch time.
        if (note.timelinePitchCents && note.preutteranceOverrideEnabled
            && localTime < 0.0 && preutteranceSeconds > 1.0e-9)
            localTime *= std::min(std::max(0.0, note.preutteranceSeconds),
                                 std::max(0.0, note.startSeconds)) / preutteranceSeconds;
        auto value = juce::jlimit(-2048, 2047, static_cast<int>(std::lround(
            baseOffsetCents + pitchCentsAt(note, localTime))));
        if (value < 0) value += 4096;
        encoded += juce::String::charToString(alphabet[(value >> 6) & 63]);
        encoded += juce::String::charToString(alphabet[value & 63]);
    }
    return encoded.isNotEmpty() ? encoded : juce::String("AA");
}

std::optional<float> midiFromText(const juce::String& text)
{
    const auto upper = text.toUpperCase();
    for (int index = 0; index < upper.length(); ++index)
    {
        const auto letter = upper[index];
        auto semitone = letter == 'C' ? 0 : letter == 'D' ? 2 : letter == 'E' ? 4
            : letter == 'F' ? 5 : letter == 'G' ? 7 : letter == 'A' ? 9
            : letter == 'B' ? 11 : -100;
        if (semitone < 0) continue;
        auto cursor = index + 1;
        if (cursor < text.length() && text[cursor] == '#')
        {
            ++semitone;
            ++cursor;
        }
        else if (cursor < text.length() && text[cursor] == 'b')
        {
            --semitone;
            ++cursor;
        }
        const auto numberStart = cursor;
        if (cursor < text.length() && text[cursor] == '-') ++cursor;
        while (cursor < text.length() && juce::CharacterFunctions::isDigit(text[cursor]))
            ++cursor;
        if (cursor == numberStart || (cursor == numberStart + 1 && text[numberStart] == '-'))
            continue;
        const auto octave = text.substring(numberStart, cursor).getIntValue();
        const auto midi = (octave + 1) * 12 + semitone;
        if (midi >= 0 && midi <= 127) return static_cast<float>(midi);
    }
    return std::nullopt;
}

std::map<juce::String, std::pair<juce::String, juce::String>> loadPrefixMap(
    const juce::File& root)
{
    std::map<juce::String, std::pair<juce::String, juce::String>> result;
    juce::Array<juce::File> files;
    root.findChildFiles(files, juce::File::findFiles, true, "prefix.map");
    for (const auto& file : files)
        for (const auto& raw : juce::StringArray::fromLines(decodeText(file)))
        {
            const auto fields = juce::StringArray::fromTokens(raw, "\t", "");
            if (fields.size() < 3) continue;
            // The prefix and the suffix are taken as written.  A leading space
            // is what separates them from the lyric in most Chinese CVVC banks
            // -- "a" at C4 is the alias "a M", not "aM" -- so trimming them
            // resolved nothing through the map at any pitch, and every note
            // fell through to whatever sample happened to share the lyric's
            // filename.  The line ending is already off: these come from
            // fromLines.
            result[fields[0].trim().toLowerCase()] = { fields[1], fields[2] };
        }
    return result;
}

void inferSourceMidi(VoiceSample& sample)
{
    if (const auto aliasPitch = midiFromText(sample.alias))
        sample.sourceMidi = *aliasPitch;
    else if (const auto filePitch = midiFromText(sample.file.getFileNameWithoutExtension()))
        sample.sourceMidi = *filePitch;
    else if (const auto directoryPitch = midiFromText(sample.file.getParentDirectory().getFileName()))
        sample.sourceMidi = *directoryPitch;
}

// An oto row's numbers laid onto a sample: where the entry starts and stops in
// the recording, its lead-in, consonant and overlap, and its regions.  One
// conversion for the voicebank's rows and for a note's own, so a note given an
// exact copy of its entry renders exactly as the entry does.
void applyOtoNumbers(VoiceSample& sample, double duration, double offsetMs,
                     double consonantMs, double cutoffMs, double preutteranceMs,
                     double overlapMs, bool hasRegions, double onsetMs,
                     double glideMs, double nucleusMs, const juce::String& classes)
{
    sample.fileSeconds = duration;
    sample.offset = juce::jlimit(0.0, duration, offsetMs / 1000.0);
    const auto requestedEnd = cutoffMs < 0.0
        ? sample.offset - cutoffMs / 1000.0
        : duration - cutoffMs / 1000.0;
    sample.end = juce::jlimit(sample.offset + 0.001,
        std::max(sample.offset + 0.001, duration), requestedEnd);
    sample.preutterance = juce::jlimit(0.0, sample.end - sample.offset,
                                       preutteranceMs / 1000.0);
    sample.consonant = juce::jlimit(0.0, sample.end - sample.offset,
                                   consonantMs / 1000.0);
    sample.overlap = overlapMs / 1000.0;
    sample.hasRegions = false;
    sample.regionSeconds = {};
    sample.mouClasses = {};
    if (hasRegions)
    {
        const auto span = sample.end - sample.offset;
        const auto b1 = juce::jlimit(0.0, span, onsetMs / 1000.0);
        const auto b2 = juce::jlimit(b1, span, glideMs / 1000.0);
        const auto b3 = juce::jlimit(b2, span, nucleusMs / 1000.0);
        // Two or three regions carry fewer boundaries; the last one runs to
        // the end of the sample and the rest of the array stays empty.
        const auto count = classes.isEmpty() ? 4 : classes.length();
        if (count == 2)      sample.regionSeconds = { b1, span - b1, 0.0, 0.0 };
        else if (count == 3) sample.regionSeconds = { b1, b2 - b1, span - b2, 0.0 };
        else                 sample.regionSeconds = { b1, b2 - b1, b3 - b2, span - b3 };
        sample.hasRegions = true;
        sample.mouClasses = classes;
    }
}

// The entry a note's lyric resolved to, with the note's own numbers in place
// of the row's.  The recording, the alias and the pitch it was recorded at
// are still the entry's: a note's own oto says where in that recording to
// read, not which recording.
//
// Read the way the track's mode reads a voicebank row.  The note may have been
// edited in another mode, but a row loaded for UTAU has no regions and one
// loaded for 界 has no classes, so neither does this.
VoiceSample withNoteOto(const VoiceSample& entry, const UtauOtoOverride& oto,
                        bool fourRegion, bool consonantClasses)
{
    auto sample = entry;
    applyOtoNumbers(sample, entry.fileSeconds, oto.offsetMs, oto.consonantMs,
                    oto.cutoffMs, oto.preutteranceMs, oto.overlapMs,
                    oto.hasRegions && fourRegion, oto.onsetMs, oto.glideMs,
                    oto.nucleusMs, consonantClasses ? oto.classes : juce::String());
    return sample;
}

std::vector<VoiceSample> loadHjmVoicebank(const juce::File& root,
                                          juce::AudioFormatManager& formats)
{
    std::vector<VoiceSample> result;
    if (!root.isDirectory()) return result;
    std::vector<std::pair<juce::File, std::vector<SampleRegionSetting>>> recordings;
    const auto manifest = root.getChildFile("otto.ini");
    if (manifest.exists())
    {
        OttoMaterialIO::Document document; juce::String error;
        if (!OttoMaterialIO::loadDocument(manifest, document, error)) return result;
        recordings.reserve(document.audio.size());
        for (auto& audio : document.audio)
            recordings.emplace_back(root.getChildFile(audio.relativePath), std::move(audio.rows));
    }
    else
    {
        juce::Array<juce::File> files;
        root.findChildFiles(files, juce::File::findFiles, true, "*"); files.sort();
        for (const auto& file : files)
            if (file.hasFileExtension("wav;flac;aif;aiff;mp3;ogg") && SampleSettings::hasNativeAnnotation(file))
                recordings.emplace_back(file, SampleSettings::loadOrDerive(file, ProjectData{}));
    }
    for (const auto& [file, rows] : recordings)
    {
        auto reader = std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(file));
        if (reader == nullptr || reader->sampleRate <= 0.0) continue;
        const auto duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        for (const auto& row : rows)
        {
            VoiceSample sample;
            sample.file = file;
            sample.alias = row.name.trim().isNotEmpty()
                ? row.name.trim() : file.getFileNameWithoutExtension();
            sample.fileSeconds = duration;
            sample.offset = juce::jlimit(0.0, duration, row.regionStartSeconds);
            sample.end = juce::jlimit(sample.offset + 0.001,
                std::max(sample.offset + 0.001, duration), row.regionEndSeconds);
            sample.preutterance = juce::jlimit(0.0, sample.end - sample.offset,
                row.alignmentSeconds - row.regionStartSeconds);
            sample.consonant = juce::jlimit(0.0, sample.end - sample.offset,
                row.fixedDurationSeconds);
            sample.overlap = row.overlapSeconds;
            sample.nativeSegments = SampleSettings::nativeSegmentsFor(row);
            projectNativeSegmentsToLegacyRegions(sample);
            if (row.melodyneOriginalPitchCenterCents > 0.0)
                sample.sourceMidi = static_cast<float>(
                    row.melodyneOriginalPitchCenterCents / 100.0);
            else if (row.melodynePitchCenterCents > 0.0)
                sample.sourceMidi = static_cast<float>(
                    row.melodynePitchCenterCents / 100.0);
            else
                inferSourceMidi(sample);
            for (const auto& alias : row.additionalAliases)
            {
                auto alternate = sample; alternate.alias = alias;
                result.push_back(std::move(alternate));
            }
            result.push_back(std::move(sample));
        }
    }
    return result;
}

std::vector<VoiceSample> loadVoicebank(const juce::File& root, bool fourRegion,
                                      bool consonantClasses)
{
    std::vector<VoiceSample> result;
    if (!root.isDirectory()) return result;

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    // An explicit native folder manifest is authoritative over its legacy OTO,
    // including an intentionally empty manifest. Never revive deleted aliases.
    if (root.getChildFile("otto.ini").exists()) return loadHjmVoicebank(root, formats);
    // UTAU voicebanks are OTO-authoritative. HJM is a separate native-material
    // fallback for banks that do not have OTO at all; a partial HJM conversion
    // must never hide OTO rows or make an entire mixed bank switch authority.
    juce::StringArray warnings;
    const auto otoEntries = SampleSettings::loadVoicebankOto(root, warnings, fourRegion,
                                                             consonantClasses);
    for (const auto& entry : otoEntries)
    {
        auto reader = std::unique_ptr<juce::AudioFormatReader>(
            formats.createReaderFor(entry.audioFile));
        if (reader == nullptr || reader->sampleRate <= 0.0) continue;
        const auto duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        VoiceSample sample;
        sample.file = entry.audioFile;
        if (!fourRegion && !consonantClasses && entry.lineIndex >= 0 && entry.sourceFingerprint.isNotEmpty())
            sample.sourceOto = std::make_shared<const hachi::VoicebankOtoEntry>(entry);
        sample.alias = entry.alias.trim().isNotEmpty()
            ? entry.alias.trim() : entry.audioFile.getFileNameWithoutExtension();
        applyOtoNumbers(sample, duration, entry.offsetMs, entry.consonantMs,
                        entry.cutoffMs, entry.preutteranceMs, entry.overlapMs,
                        entry.hasJieOto, entry.jieOnsetMs, entry.jieGlideMs,
                        entry.jieNucleusMs, entry.mouClasses);
        inferSourceMidi(sample);
        result.push_back(std::move(sample));
    }
    // OTO remains authoritative for rows it defines. HJM-only recordings may
    // still be appended as migration fallbacks, but never replace an OTO row.
    if (!result.empty())
    {
        if (auto native = loadHjmVoicebank(root, formats); !native.empty())
        {
            const auto sameEntry = [&result](const VoiceSample& candidate)
            {
                return std::any_of(result.begin(), result.end(), [&](const VoiceSample& existing)
                {
                    return existing.file.getFullPathName().equalsIgnoreCase(
                               candidate.file.getFullPathName())
                        && existing.alias.equalsIgnoreCase(candidate.alias);
                });
            };
            for (auto& candidate : native)
                if (!sameEntry(candidate)) result.push_back(std::move(candidate));
        }
        return result;
    }

    // A folder without oto.ini remains usable as a one-sample/minimal bank by
    // deriving regions from the existing HJM sidecars or the whole file.
    juce::Array<juce::File> files;
    root.findChildFiles(files, juce::File::findFiles, true, "*");
    for (const auto& file : files)
    {
        if (!file.hasFileExtension("wav;flac;aif;aiff;mp3;ogg")) continue;
        auto reader = std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(file));
        if (reader == nullptr || reader->sampleRate <= 0.0) continue;
        const auto duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        juce::String annotationError;
        auto rows = SampleSettings::loadOrDerive(file, ProjectData{}, &annotationError);
        const auto hasSidecar = SampleSettings::hasNativeAnnotation(file, &annotationError);
        if (annotationError.isNotEmpty()) continue;
        if (!hasSidecar)
        {
            rows.clear();
            SampleRegionSetting row;
            row.name = file.getFileNameWithoutExtension();
            row.regionEndSeconds = duration;
            rows.push_back(std::move(row));
        }
        for (const auto& row : rows)
        {
            VoiceSample sample;
            sample.file = file;
            sample.alias = row.name.trim().isNotEmpty()
                ? row.name.trim() : file.getFileNameWithoutExtension();
            sample.fileSeconds = duration;
            sample.offset = juce::jlimit(0.0, duration, row.regionStartSeconds);
            sample.end = juce::jlimit(sample.offset + 0.001, std::max(sample.offset + 0.001, duration),
                                      row.regionEndSeconds);
            sample.preutterance = juce::jlimit(0.0, sample.end - sample.offset,
                                               row.alignmentSeconds - row.regionStartSeconds);
            sample.consonant = juce::jlimit(0.0, sample.end - sample.offset,
                                            row.fixedDurationSeconds);
            sample.overlap = row.overlapSeconds;
            if (row.melodyneOriginalPitchCenterCents > 0.0)
                sample.sourceMidi = static_cast<float>(row.melodyneOriginalPitchCenterCents / 100.0);
            else if (row.melodynePitchCenterCents > 0.0)
                sample.sourceMidi = static_cast<float>(row.melodynePitchCenterCents / 100.0);
            else inferSourceMidi(sample);
            result.push_back(std::move(sample));
        }
    }
    return result;
}

using IndexPointer = std::shared_ptr<const VoicebankIndex>;

// A file an index was read from, and when it was last written (-1: absent).
struct FileStamp
{
    juce::File file;
    juce::int64 modified = -1;
    juce::int64 size = 0;
};

juce::int64 modifiedStamp(const juce::File& file)
{
    return file.existsAsFile() ? file.getLastModificationTime().toMilliseconds() : -1;
}

// What an index depends on: every oto the bank has, the 界 and 谋 files beside
// each one whether or not they exist yet, and the prefix maps.
//
// Not the folder's own modification time, which is what the cache used to be
// keyed on.  The engine writes its analysis cache beside the samples, so
// every sample rendered for the first time touched the folder, and the next
// edit read the whole bank again -- on the message thread, for seconds.
std::vector<FileStamp> voicebankStamps(const juce::File& root)
{
    std::vector<FileStamp> stamps;
    const auto stamp = [&stamps](const juce::File& file)
    {
        stamps.push_back({ file, modifiedStamp(file), file.getSize() });
    };
    // Native material annotations share the same index invalidation boundary.
    // Include the absent root path so first publication is observable too.
    stamp(root.getChildFile("otto.ini"));
    juce::Array<juce::File> nativeManifests;
    root.findChildFiles(nativeManifests, juce::File::findFiles, true, "otto.ini");
    for (const auto& file : nativeManifests) if (file != root.getChildFile("otto.ini")) stamp(file);
    juce::Array<juce::File> otoFiles;
    root.findChildFiles(otoFiles, juce::File::findFiles, true, "oto.ini");
    for (const auto& oto : otoFiles)
    {
        stamp(oto);
        stamp(LegacyTextCodec::directoryConfig(oto.getParentDirectory()));
        stamp(SampleSettings::jieClassicOtoFileFor(oto));
        stamp(SampleSettings::jieOtoFileFor(oto));
        stamp(SampleSettings::mouOtoFileFor(oto));
    }
    // A bank read from its sidecars has no oto yet; one appearing is a change.
    if (otoFiles.isEmpty()) {stamp(root.getChildFile("oto.ini"));stamp(LegacyTextCodec::directoryConfig(root));}
    juce::Array<juce::File> maps;
    root.findChildFiles(maps, juce::File::findFiles, true, "prefix.map");
    for (const auto& map : maps) {stamp(map);stamp(LegacyTextCodec::directoryConfig(map.getParentDirectory()));}
    // WAV headers determine duration and positive/negative cutoff geometry.
    // Stamp before reading so replacing a recording also refreshes that index.
    juce::Array<juce::File> recordings;
    root.findChildFiles(recordings, juce::File::findFiles, true, "*.wav");
    for (const auto& file : recordings) stamp(file);
    return stamps;
}

struct IndexCache
{
    struct Entry
    {
        IndexPointer index;
        std::shared_future<IndexPointer> pending;
        std::vector<std::function<void()>> whenReady;
        std::vector<FileStamp> stamps;
        bool rootExisted = false;
        double checkedAt = 0.0;
        std::uint64_t lastUsed = 0;
    };
    std::mutex mutex;
    std::map<std::string, Entry> entries;
    std::uint64_t clock = 0;
    std::atomic<int> builds { 0 };
    std::atomic<int> messageThreadWaits { 0 };
};

IndexCache& indexCache()
{
    static IndexCache cache;
    return cache;
}

// Background readings of a bank for voicebankIndexReady.  Made after the
// cache, so it is destroyed before it.
juce::ThreadPool& indexReadingPool()
{
    indexCache();
    static juce::ThreadPool pool(2);
    return pool;
}

// Files written from outside this application are noticed within a second:
// checking them on every lookup would be a handful of disk queries per note.
// Writes made here count in the key, and are seen at once.
constexpr double voicebankRecheckMs = 1000.0;

std::string indexKey(const juce::File& root, bool fourRegion, bool consonantClasses)
{
    return (root.getFullPathName() + "|"
            + (consonantClasses ? "mou|" : fourRegion ? "jie|" : "classic|")
            + juce::String(static_cast<juce::int64>(
                  voicebankCacheRevision().load(std::memory_order_relaxed))) + "|"
            + juce::String(static_cast<juce::int64>(
                  SampleSettings::voicebankFilesRevision()))).toStdString();
}

// Lock held.  Whether a cached index still describes the files on disk.
bool entryCurrent(IndexCache::Entry& entry, const juce::File& root)
{
    const auto now = juce::Time::getMillisecondCounterHiRes();
    if (now - entry.checkedAt < voicebankRecheckMs) return true;
    if (root.isDirectory() != entry.rootExisted) return false;
    for (const auto& stamp : entry.stamps)
        if (modifiedStamp(stamp.file) != stamp.modified || stamp.file.getSize() != stamp.size) return false;
    entry.checkedAt = now;
    return true;
}

// Lock held.  Keeps the four most recently used banks that are not being read.
void trimIndexCache(IndexCache& cache)
{
    constexpr std::size_t kept = 4;
    for (;;)
    {
        std::size_t idle = 0;
        auto oldest = cache.entries.end();
        for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it)
        {
            if (it->second.pending.valid() || !it->second.whenReady.empty()) continue;
            ++idle;
            if (oldest == cache.entries.end() || it->second.lastUsed < oldest->second.lastUsed)
                oldest = it;
        }
        if (idle <= kept || oldest == cache.entries.end()) return;
        cache.entries.erase(oldest);
    }
}

// Lock held.  The index when it is ready and current.  Otherwise pending is
// the reading under way -- one is started, and promise set, when there is
// none, and the caller then does the reading.
IndexPointer lookupIndex(IndexCache& cache, const std::string& key, const juce::File& root,
                         std::shared_future<IndexPointer>& pending,
                         std::shared_ptr<std::promise<IndexPointer>>& promise)
{
    auto& entry = cache.entries[key];
    entry.lastUsed = ++cache.clock;
    if (entry.index != nullptr)
    {
        if (entryCurrent(entry, root)) return entry.index;
        entry.index.reset();
    }
    if (!entry.pending.valid())
    {
        promise = std::make_shared<std::promise<IndexPointer>>();
        entry.pending = promise->get_future().share();
    }
    pending = entry.pending;
    return nullptr;
}

// Reads the bank and files the result.  Every caller waiting on the reading
// is answered, including with a failure; nothing is left waiting forever.
void readVoicebankIndex(const std::string& key, const juce::File& root, bool fourRegion,
                        bool consonantClasses, std::promise<IndexPointer>& promise)
{
    auto& cache = indexCache();
    try
    {
        // Stamped before reading, so a file written while it is read is newer
        // than what was stamped and the index is read again.
        const auto rootExisted = root.isDirectory();
        auto stamps = voicebankStamps(root);
        auto loaded = std::make_shared<VoicebankIndex>();
        loaded->nativeAuthority = root.getChildFile("otto.ini").exists();
        loaded->samples = loadVoicebank(root, fourRegion, consonantClasses);
        loaded->prefixMap = loadPrefixMap(root);
        for (std::size_t index = 0; index < loaded->samples.size(); ++index)
        {
            const auto& sample = loaded->samples[index];
            loaded->byAlias.emplace(foldedKey(sample.alias), index);
            loaded->byExactAlias.emplace(sample.alias.toStdString(), index);
            loaded->byFileName.emplace(
                foldedKey(sample.file.getFileNameWithoutExtension()), index);
        }
        cache.builds.fetch_add(1, std::memory_order_relaxed);
        std::vector<std::function<void()>> answered;
        {
            const std::scoped_lock lock(cache.mutex);
            auto& entry = cache.entries[key];
            entry.index = loaded;
            entry.stamps = std::move(stamps);
            entry.rootExisted = rootExisted;
            entry.checkedAt = juce::Time::getMillisecondCounterHiRes();
            entry.pending = {};
            answered.swap(entry.whenReady);
            trimIndexCache(cache);
        }
        promise.set_value(loaded);
        for (auto& callback : answered)
            juce::MessageManager::callAsync(std::move(callback));
    }
    catch (...)
    {
        // Those waiting are told as well, so they can ask again rather than
        // wait on a reading that is never coming.
        std::vector<std::function<void()>> answered;
        {
            const std::scoped_lock lock(cache.mutex);
            auto& entry = cache.entries[key];
            entry.pending = {};
            answered.swap(entry.whenReady);
        }
        promise.set_exception(std::current_exception());
        for (auto& callback : answered)
            juce::MessageManager::callAsync(std::move(callback));
    }
}

IndexPointer loadVoicebankIndex(const juce::File& root, bool fourRegion, bool consonantClasses)
{
    auto& cache = indexCache();
    const auto key = indexKey(root, fourRegion, consonantClasses);
    std::shared_future<IndexPointer> pending;
    std::shared_ptr<std::promise<IndexPointer>> promise;
    {
        const std::scoped_lock lock(cache.mutex);
        if (auto index = lookupIndex(cache, key, root, pending, promise)) return index;
    }
    if (juce::MessageManager::existsAndIsCurrentThread())
        cache.messageThreadWaits.fetch_add(1, std::memory_order_relaxed);
    // Two renders of one bank used to read it twice at once; the second now
    // waits for the first.
    if (promise != nullptr)
        readVoicebankIndex(key, root, fourRegion, consonantClasses, *promise);
    return pending.get();
}

// The same entry, moved bodily along the recording.
//
// Every boundary an oto entry holds -- overlap, preutterance, consonant, and
// the four region lengths -- is a length measured from the offset, so moving
// the offset carries all of them with it and the span between offset and
// cutoff is unchanged.  That is what an STP does: it decides which audio the
// entry describes, not where the note sits in the piece.
//
// Applied per note rather than at load time: one entry is shared by every
// note that sings that alias, and an STP belongs to one of them.
VoiceSample shiftedBy(const VoiceSample& sample, double seconds)
{
    if (!std::isfinite(seconds) || std::abs(seconds) < 1.0e-9) return sample;
    auto shifted = sample;
    const auto span = std::max(0.0, sample.end - sample.offset);
    // Not past either edge of the recording: there is no audio out there, and
    // a resampler handed an offset beyond the file returns silence.
    const auto furthest = std::max(0.0, sample.fileSeconds - span);
    shifted.offset = juce::jlimit(0.0, furthest, sample.offset + seconds);
    shifted.end = shifted.offset + span;
    return shifted;
}

// The rule, as a scan.  Kept as the reference the tables are checked against.
const VoiceSample* resolveSampleByScan(
    const std::vector<VoiceSample>& samples,
    const std::map<juce::String, std::pair<juce::String, juce::String>>& prefixMap,
    const juce::String& requestedAlias, float midi, bool strictAliases = false)
{
    if (samples.empty()) return nullptr;
    auto alias = requestedAlias.trim();
    // Imported MIDI notes do not necessarily carry lyrics.  Treating an empty
    // lyric as "a" makes a marquee appear to play unrelated cached fragments.
    // Only an explicitly assigned alias is allowed to resolve a voice sample.
    if (alias.isEmpty()) return nullptr;
    juce::StringArray candidates;
    if (const auto found = prefixMap.find(midiName(midi).toLowerCase()); found != prefixMap.end())
        candidates.add(found->second.first + alias + found->second.second);
    candidates.add(alias);
    for (const auto& candidate : candidates)
        for (const auto& sample : samples)
            if (sample.alias.equalsIgnoreCase(candidate)) return &sample;
    if (strictAliases) return nullptr;
    // A single recorded tone is deliberately useful as a complete minimal
    // voicebank: any lyric and MIDI pitch map to that one region.
    if (samples.size() == 1) return &samples.front();
    for (const auto& sample : samples)
        if (sample.file.getFileNameWithoutExtension().equalsIgnoreCase(alias)) return &sample;
    return nullptr;
}

// The same rule through the index's tables: the prefix-map spelling first,
// then the lyric as it is, then -- for a bank of one sample -- that sample,
// and last a sample whose file is named like the lyric.
const VoiceSample* indexedAlias(const VoicebankIndex& voicebank, const juce::String& alias,
                               bool preferExact)
{
    if (preferExact)
        if (const auto found = voicebank.byExactAlias.find(alias.toStdString()); found != voicebank.byExactAlias.end())
            return &voicebank.samples[found->second];
    const auto found = voicebank.byAlias.find(foldedKey(alias));
    return found != voicebank.byAlias.end() ? &voicebank.samples[found->second] : nullptr;
}

const VoiceSample* resolveSample(const VoicebankIndex& voicebank,
                                 const juce::String& requestedAlias, float midi, bool exactAlias = false)
{
    const auto& samples = voicebank.samples;
    if (samples.empty()) return nullptr;
    const auto alias = requestedAlias.trim();
    if (alias.isEmpty()) return nullptr;
    const auto byAlias = [&voicebank, exactAlias](const juce::String& candidate) -> const VoiceSample*
    {
        return indexedAlias(voicebank, candidate, exactAlias);
    };
    if (exactAlias) return byAlias(alias);
    if (const auto found = voicebank.prefixMap.find(midiName(midi).toLowerCase());
        found != voicebank.prefixMap.end())
        if (const auto* sample = byAlias(found->second.first + alias + found->second.second))
            return sample;
    if (const auto* sample = byAlias(alias)) return sample;
    if (voicebank.nativeAuthority) return nullptr;
    if (samples.size() == 1) return &samples.front();
    const auto file = voicebank.byFileName.find(foldedKey(alias));
    return file != voicebank.byFileName.end() ? &samples[file->second] : nullptr;
}

bool readAudio(const juce::File& file, juce::AudioBuffer<float>& buffer, double& rate)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    auto reader = std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(file));
    if (reader == nullptr || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0)
        return false;
    const auto count = static_cast<int>(std::min<juce::int64>(
        reader->lengthInSamples, std::numeric_limits<int>::max()));
    buffer.setSize(juce::jlimit(1, 2, static_cast<int>(reader->numChannels)), count);
    buffer.clear();
    reader->read(&buffer, 0, count, 0, true, buffer.getNumChannels() > 1);
    rate = reader->sampleRate;
    return true;
}

ExternalResamplerAttempt runExternalResampler(
    const UtauRenderRequest& request, const VoiceSample& sample,
    const UtauNoteRenderSpec& note, double effectivePreutterance,
    double outputSeconds)
{
    ExternalResamplerAttempt attempt;
    if (request.cancelled && request.cancelled())
    {
        attempt.error = "resampler cancelled";
        return attempt;
    }
    if (!request.resamplerExecutable.existsAsFile())
    {
        attempt.error = "resampler executable was not found";
        return attempt;
    }
    auto output = juce::File::createTempFile("hachi-utau.wav");
    output.deleteFile();
    const auto offsetMs = sample.offset * 1000.0;
    const auto consonantMs = sample.consonant * 1000.0;
    const auto cutoffMs = -(sample.end - sample.offset) * 1000.0;
    juce::StringArray args;
    args.add(request.resamplerExecutable.getFullPathName());
    args.add(sample.file.getFullPathName());
    args.add(output.getFullPathName());
    args.add(midiName(note.midiNote));
    args.add(juce::String(headConsonantVelocity(sample, note.consonantVelocity)));
    // UTAU's resampler protocol is positional.  JUCE's Windows command-line
    // builder drops an empty String, shifting every argument after Flags one
    // place to the left.  g0 is the ecosystem-wide neutral spelling and keeps
    // the position stable for the default resampler, moresampler and WCSNDM.
    args.add(note.flags.isEmpty() ? juce::String("g0") : note.flags);
    args.add(juce::String(offsetMs, 3));
    args.add(juce::String(outputSeconds * 1000.0, 3));
    args.add(juce::String(consonantMs, 3));
    args.add(juce::String(cutoffMs, 3));
    args.add(juce::String(juce::jlimit(0, 200, static_cast<int>(std::lround(note.gain * 100.0f)))));
    args.add(juce::String(note.modulationPercent, 6));
    const auto noteBpm = note.bpm > 0.0 ? note.bpm : request.bpm;
    args.add("!" + juce::String(juce::jlimit(20.0, 400.0, noteBpm), 2));
    args.add(encodePitchbend(note, noteBpm, effectivePreutterance, outputSeconds));
    // Optional 14th argument: the four-region split for this note.  Only
    // WCSNDM reads it; classic resamplers take argv[1..13] by index and
    // ignore the extra.  Lengths are sent in ms and the engine normalises
    // them to the note, so these are effectively proportions.
    // Sent whenever there are four regions, not only when someone has moved
    // a boundary by hand.  Left to plan for itself the engine puts the onset
    // at the oto4 mark, while these lengths put it at the lead-in, and the two
    // are not the same number: the first drag of any boundary switched the
    // note from one to the other, so touching the third region changed how the
    // consonant sounded.  One path, and it cannot.
    auto regionArgument = juce::String();
    if (request.fourRegion && sample.hasRegions)
    {
        const auto split = UtauRenderer::regionSplit(sample.regionSeconds,
            outputSeconds, note.consonantVelocity, effectivePreutterance,
            note.jieSplitSet ? &note.jieSplit : nullptr,
            request.consonantClasses && sample.mouClasses.isNotEmpty()
                ? &sample.mouClasses : nullptr);
        if (split.valid)
        {
            // Left of the bar: the class string and the source boundaries it
            // describes.  Only sent in 谋•UTAU, and only for an annotated
            // entry -- left empty, as it always was, the engine reads the
            // boundaries from the file itself, which is what UTAU and 界•UTAU
            // still do to the byte.
            auto left = juce::String();
            if (request.consonantClasses && sample.mouClasses.isNotEmpty())
            {
                const auto b1 = sample.regionSeconds[0];
                const auto b2 = b1 + sample.regionSeconds[1];
                const auto b3 = b2 + sample.regionSeconds[2];
                left = sample.mouClasses
                    + "," + juce::String(b1 * 1000.0, 3)
                    + "," + juce::String(b2 * 1000.0, 3)
                    + "," + juce::String(b3 * 1000.0, 3);
            }
            // As many lengths as the entry has regions.  The engine reads
            // the count from the class string, so sending four for a
            // three-region entry would leave it waiting for a fourth.
            const auto count = request.consonantClasses
                    && sample.mouClasses.isNotEmpty()
                ? sample.mouClasses.length() : 4;
            juce::StringArray lengths;
            for (int index = 0; index < count; ++index)
                lengths.add(juce::String(split.seconds[
                    static_cast<std::size_t>(index)] * 1000.0, 3));
            regionArgument = left + "|" + lengths.joinIntoString(",");
        }
    }
    // Per-region flags travel in the 15th argument, so the 14th has to be
    // present even when this note has no region lengths of its own.  A bare
    // separator parses as neither boundaries nor lengths, and JUCE would drop
    // an empty string and shift everything after it.
    auto flagArgument = juce::String();
    if (request.fourRegion && sample.hasRegions && note.flagSplit)
    {
        auto anySet = false;
        for (const auto& text : note.regionFlags) anySet = anySet || text.isNotEmpty();
        if (anySet)
            flagArgument = note.regionFlags[0] + "|" + note.regionFlags[1] + "|"
                + note.regionFlags[2] + "|" + note.regionFlags[3];
    }
    // Per-frame flag curves travel in the 16th argument, in milliseconds from
    // the start of the rendered segment.  A note's own points are written
    // against its nominal start, so each moves forward by the preutterance.
    auto curveArgument = juce::String();
    if (UtauRenderer::sendsFlagCurves(note.flagCurve))
    {
        juce::StringArray curves;
        for (const auto& [flag, points] : note.flagCurves)
        {
            if (flag.startsWith("DS:") || flag.startsWith("HIFI:")) continue;
            if (points.empty()) continue;
            juce::StringArray written;
            for (const auto& [timeSeconds, value] : points)
                written.add(juce::String((timeSeconds + effectivePreutterance) * 1000.0, 3)
                            + "," + juce::String(value, 3));
            curves.add(flag + ":" + written.joinIntoString(";"));
        }
        curveArgument = curves.joinIntoString("|");
    }
    // The three extra arguments are positional, so an earlier one has to be
    // there for a later one to land.  JUCE drops an empty string and shifts
    // everything after it, so the placeholders are written out: "|" parses as
    // neither boundaries nor lengths, and "|||" as four empty region flags,
    // which the engine reads as no split at all.
    const auto needFifteenth = flagArgument.isNotEmpty() || curveArgument.isNotEmpty();
    if (regionArgument.isNotEmpty() || needFifteenth)
        args.add(regionArgument.isNotEmpty() ? regionArgument : juce::String("|"));
    if (needFifteenth)
        args.add(flagArgument.isNotEmpty() ? flagArgument : juce::String("|||"));
    if (curveArgument.isNotEmpty()) args.add(curveArgument);
    // A per-track resampler may differ from the one warmed in application
    // preferences. Start its HF service here as well so the engine never
    // needs to create an unowned daemon of its own.
    (void) UtauRenderer::startHfDaemonIfNeeded(request.resamplerExecutable);
    if (request.cancelled && request.cancelled())
    {
        output.deleteFile();
        attempt.error = "resampler cancelled";
        return attempt;
    }
    juce::ChildProcess process;
    // Capture the engine's own diagnostics.  WCSNDM explains its failures on
    // stderr ("extension not installed", "daemon failed to start"), and
    // without this the host could only report an exit code.
    if (!process.start(args, juce::ChildProcess::wantStdOut
                                 | juce::ChildProcess::wantStdErr))
    {
        output.deleteFile();
        attempt.error = "could not start " + request.resamplerExecutable.getFileName();
        return attempt;
    }
    // First use may analyse a sample and create a sidecar cache.  A neural
    // backend goes further: the first note of a session waits for the
    // vocoder daemon to load its model, which alone takes half a minute, so
    // 30 seconds killed exactly the note that needed the most patience.
    const auto processStarted = juce::Time::getMillisecondCounterHiRes();
    auto cancelled = false;
    auto timedOut = false;
    while (process.isRunning())
    {
        cancelled = request.cancelled && request.cancelled();
        timedOut = juce::Time::getMillisecondCounterHiRes() - processStarted >= 120'000.0;
        if (cancelled || timedOut) break;
        if (process.waitForProcessToFinish(50)) break;
    }
    cancelled = cancelled || (request.cancelled && request.cancelled());
    if (cancelled || timedOut)
    {
        if (process.isRunning()) process.kill();
        process.waitForProcessToFinish(2000);
        output.deleteFile();
        attempt.error = request.resamplerExecutable.getFileName()
            + (cancelled ? " cancelled" : " timed out after 120 seconds");
        return attempt;
    }
    // Safe to drain only now that the child is gone: the pipe cannot refill.
    const auto diagnostics = process.readAllProcessOutput().trim();
    juce::AudioBuffer<float> rendered;
    double rate = 0.0;
    if (process.getExitCode() != 0)
    {
        const auto exitCode = process.getExitCode();
        output.deleteFile();
        attempt.error = request.resamplerExecutable.getFileName()
            + " exited with code " + juce::String(exitCode);
        // The last line is where an engine states why it gave up.
        const auto lines = juce::StringArray::fromLines(diagnostics);
        for (auto index = lines.size() - 1; index >= 0; --index)
            if (lines[index].trim().isNotEmpty())
            { attempt.error += ": " + lines[index].trim(); break; }
        return attempt;
    }
    if (!readAudio(output, rendered, rate))
    {
        output.deleteFile();
        attempt.error = request.resamplerExecutable.getFileName()
            + " did not produce a readable WAV";
        return attempt;
    }
    output.deleteFile();
    // The mixer handles arbitrary source rates, so retain the rate by
    // resampling into its fixed clock here.
    const auto targetSamples = std::max(1, static_cast<int>(std::lround(
        rendered.getNumSamples() * mixSampleRate / rate)));
    juce::AudioBuffer<float> converted(rendered.getNumChannels(), targetSamples);
    for (int channel = 0; channel < converted.getNumChannels(); ++channel)
        for (int index = 0; index < targetSamples; ++index)
        {
            const auto position = static_cast<double>(index) * rate / mixSampleRate;
            const auto left = juce::jlimit(0, rendered.getNumSamples() - 1,
                                           static_cast<int>(std::floor(position)));
            const auto right = std::min(rendered.getNumSamples() - 1, left + 1);
            const auto amount = static_cast<float>(position - std::floor(position));
            const auto* source = rendered.getReadPointer(channel);
            converted.setSample(channel, index,
                source[left] + (source[right] - source[left]) * amount);
        }
    attempt.audio = std::move(converted);
    attempt.succeeded = true;
    return attempt;
}

juce::AudioBuffer<float> renderNative(const VoiceSample& sample,
                                      const UtauNoteRenderSpec& note,
                                      double outputSeconds)
{
    juce::AudioBuffer<float> source;
    double sourceRate = 0.0;
    if (!readAudio(sample.file, source, sourceRate)) return {};
    const auto outputSamples = std::max(1, static_cast<int>(std::lround(outputSeconds * mixSampleRate)));
    juce::AudioBuffer<float> result(source.getNumChannels(), outputSamples);
    result.clear();
    const auto sourceBegin = sample.offset * sourceRate;
    const auto sourceEnd = std::max(sourceBegin + 1.0, sample.end * sourceRate);
    const auto anchorSource = juce::jlimit(sourceBegin, sourceEnd,
        (sample.offset + sample.preutterance) * sourceRate);
    const auto sustainBegin = juce::jlimit(anchorSource, sourceEnd,
        (sample.offset + std::max(sample.preutterance, sample.consonant)) * sourceRate);
    const auto pitchRatio = std::pow(2.0,
        static_cast<double>(note.midiNote - sample.sourceMidi) / 12.0);
    // Match the standard UTAU velocity convention used by WCSNDM:
    // conOut = consonant * 2^(1 - velocity / 100).  A value of 100 is
    // neutral, while a larger value reaches the vowel sooner.
    const auto velocityScale = consonantVelocityScale(
        headConsonantVelocity(sample, note.consonantVelocity));
    const auto rawConsonantOutputSamples =
        sample.consonant * velocityScale * mixSampleRate;
    const auto consonantOutputSamples = rawConsonantOutputSamples >= outputSamples
        ? outputSamples
        : std::max(0, static_cast<int>(std::lround(rawConsonantOutputSamples)));
    const auto consonantLogicalSamples = sample.consonant * mixSampleRate;
    const auto preSamples = sample.preutterance * mixSampleRate;
    for (int index = 0; index < outputSamples; ++index)
    {
        const auto logicalIndex = consonantOutputSamples > 0 && index < consonantOutputSamples
            ? static_cast<double>(index) / velocityScale
            : consonantLogicalSamples
                + static_cast<double>(std::max(0, index - consonantOutputSamples));
        double position;
        if (preSamples > 0.0 && logicalIndex < preSamples)
            position = sourceBegin + (anchorSource - sourceBegin)
                * logicalIndex / preSamples;
        else
        {
            position = anchorSource + std::max(0.0, logicalIndex - preSamples)
                * sourceRate / mixSampleRate * pitchRatio;
            if (position >= sourceEnd && sourceEnd - sustainBegin > 2.0)
                position = sustainBegin + std::fmod(position - sustainBegin,
                                                     sourceEnd - sustainBegin);
            position = juce::jlimit(sourceBegin, sourceEnd - 1.0, position);
        }
        const auto left = juce::jlimit(0, source.getNumSamples() - 1,
                                       static_cast<int>(std::floor(position)));
        const auto right = std::min(source.getNumSamples() - 1, left + 1);
        const auto amount = static_cast<float>(position - std::floor(position));
        for (int channel = 0; channel < result.getNumChannels(); ++channel)
        {
            const auto* input = source.getReadPointer(channel);
            result.setSample(channel, index,
                (input[left] + (input[right] - input[left]) * amount) * note.gain);
        }
    }
    return result;
}

juce::AudioBuffer<float> renderPianoPreview(const UtauNoteRenderSpec& note, double sampleRate = mixSampleRate)
{
    // A lightweight deterministic piano-like preview for MIDI notes that do
    // not yet have a lyric/voicebank alias.  It follows the note's MIDI pitch
    // and duration and is deliberately kept out of the UTAU sample cache.
    const auto duration = std::max(0.04, note.durationSeconds);
    const auto outputSeconds = duration + 0.18;
    const auto samples = std::max(1, static_cast<int>(std::ceil(
        outputSeconds * sampleRate)));
    juce::AudioBuffer<float> result(1, samples);
    const auto fundamental = 440.0 * std::pow(2.0,
        (static_cast<double>(note.midiNote) - 69.0) / 12.0);
    static constexpr double harmonicGain[] { 1.0, 0.42, 0.20, 0.105, 0.055 };
    for (int index = 0; index < samples; ++index)
    {
        const auto time = static_cast<double>(index) / sampleRate;
        const auto attack = 1.0 - std::exp(-time / 0.0035);
        const auto decay = std::exp(-3.2 * time / std::max(0.16, duration));
        const auto release = time <= duration ? 1.0
            : std::exp(-24.0 * (time - duration));
        auto value = 0.0;
        for (std::size_t harmonic = 0; harmonic < std::size(harmonicGain); ++harmonic)
        {
            const auto partial = static_cast<double>(harmonic + 1);
            // A tiny inharmonic stretch makes the preview read as a struck
            // string instead of an organ while retaining the requested pitch.
            const auto stretched = partial * (1.0 + 0.00045 * partial * partial);
            value += harmonicGain[harmonic] * std::sin(
                juce::MathConstants<double>::twoPi * fundamental * stretched * time);
        }
        result.setSample(0, index, static_cast<float>(
            0.16 * juce::jlimit(0.0f, 2.0f, note.gain)
            * attack * decay * release * value));
    }
    return result;
}

void retimeLeadIn(juce::AudioBuffer<float>& audio,
                  double naturalPreutterance, double targetPreutterance)
{
    if (audio.getNumSamples() <= 0
        || std::abs(naturalPreutterance - targetPreutterance)
            < 0.5 / mixSampleRate)
        return;

    const auto sourceHead = juce::jlimit(0, audio.getNumSamples(),
        static_cast<int>(std::lround(std::max(0.0, naturalPreutterance)
                                     * mixSampleRate)));
    const auto targetHead = std::max(0, static_cast<int>(std::lround(
        std::max(0.0, targetPreutterance) * mixSampleRate)));
    const auto tailSamples = audio.getNumSamples() - sourceHead;
    juce::AudioBuffer<float> retimed(audio.getNumChannels(),
                                     std::max(1, targetHead + tailSamples));
    retimed.clear();
    for (int channel = 0; channel < retimed.getNumChannels(); ++channel)
    {
        const auto* source = audio.getReadPointer(channel);
        auto* target = retimed.getWritePointer(channel);
        for (int index = 0; index < retimed.getNumSamples(); ++index)
        {
            double sourcePosition;
            if (index < targetHead)
            {
                sourcePosition = targetHead > 1 && sourceHead > 1
                    ? static_cast<double>(index) * static_cast<double>(sourceHead - 1)
                        / static_cast<double>(targetHead - 1)
                    : 0.0;
            }
            else
            {
                sourcePosition = static_cast<double>(sourceHead + index - targetHead);
            }
            sourcePosition = juce::jlimit(0.0,
                static_cast<double>(audio.getNumSamples() - 1), sourcePosition);
            const auto left = static_cast<int>(std::floor(sourcePosition));
            const auto right = std::min(audio.getNumSamples() - 1, left + 1);
            const auto amount = static_cast<float>(sourcePosition - left);
            target[index] = source[left] + (source[right] - source[left]) * amount;
        }
    }
    audio = std::move(retimed);
}

float amplitudeGainAt(const std::vector<UtauAmplitudePoint>& points, double localSeconds)
{
    if (points.empty()) return 1.0f;
    const auto right = std::upper_bound(points.begin(), points.end(), localSeconds,
        [](double value, const UtauAmplitudePoint& point)
        {
            return value < point.timeSeconds;
        });
    auto gainDb = points.front().gainDb;
    if (right == points.end())
        gainDb = points.back().gainDb;
    else if (right != points.begin())
    {
        const auto& left = *std::prev(right);
        const auto span = right->timeSeconds - left.timeSeconds;
        const auto amount = span > 1.0e-9
            ? static_cast<float>(juce::jlimit(0.0, 1.0,
                (localSeconds - left.timeSeconds) / span)) : 0.0f;
        gainDb = envelopeDbBetween(left.gainDb, right->gainDb, amount, left.linearToNext,left.bezier);
    }
    return gainDb <= -59.9f ? 0.0f : std::pow(10.0f, gainDb / 20.0f);
}

std::vector<UtauAmplitudePoint> envelopeForNoteAsItIs(
    const std::vector<UtauAmplitudePoint>& drawn, double preutteranceSeconds,
    double soundingEndSeconds)
{
    // An envelope belongs to the note as it stood when it was drawn, and the
    // note moves afterwards: a slower consonant reaches further back for it,
    // and stretching it gives it a longer body.  Past its last point an
    // envelope holds that gain, so one that ends early silences the rest.
    //
    // Both ends therefore follow, and each takes its ramp with it, so the
    // shape that was chosen survives the change instead of being stretched
    // out of it.  This is the rule the roll draws by, so what is heard is
    // what is shown.
    //
    // The far end is where the note stops sounding, which is not always its
    // own end: where the next note's overlap is longer than its lead-in the
    // tail carries on past the beat.  Pinned to the note's end instead, the
    // envelope's closing gain -- silence -- was held across the whole of that
    // tail, so however far the note was rendered and however carefully the
    // mixer faded it, nothing was heard past the bar line.  That is what an
    // overlap looked like doing nothing on any note anyone had sung, whatever
    // the preutterance was.
    if (drawn.size() < 2) return drawn;
    auto points = drawn;
    const auto first = std::min(0.0, -preutteranceSeconds);
    const auto last = std::max(first + 0.02, soundingEndSeconds);
    const auto startShift = first - points.front().timeSeconds;
    const auto endShift = last - points.back().timeSeconds;
    if (points.size() >= 4)
    {
        points[1].timeSeconds += startShift;
        points[points.size() - 2].timeSeconds += endShift;
    }
    points.front().timeSeconds = first;
    points.back().timeSeconds = last;
    // Shorter than the ramps it was given: nothing may overtake what comes
    // before it, and nothing may pass the end.
    auto earliest = first;
    for (auto& point : points)
    {
        point.timeSeconds = juce::jlimit(earliest, last, point.timeSeconds);
        earliest = point.timeSeconds;
    }
    points.back().timeSeconds = last;
    return points;
}

// Everything the mix multiplies one sample of a note by: its envelope, the
// fade in across its overlap, and the fade out -- into the next note, or off
// the end of the piece when nothing follows.  Written once, because the
// waveform drawn for a note is shaped by it too, and two copies of a rule are
// how a picture and a sound come to disagree.
struct NoteMixGain
{
    const std::vector<UtauAmplitudePoint>* envelope = nullptr;
    int destinationStart = 0;
    double preutteranceSeconds = 0.0;
    UtauMixEnvelope automatic;
    int tailFadeMode = 0;
    TailFadeSettings tailFadeSettings;
    std::optional<UtauTailFadeSpan> tailFade,headEnvelope;

    // withEnvelope false leaves the envelope out and keeps the fades: the
    // piece as the line in the envelope lane acts on it.
    [[nodiscard]] float at(int index, bool withEnvelope = true) const
    {
        if (index < 0 || index >= automatic.samples) return 0.0f;
        const auto localSeconds = static_cast<double>(index) / mixSampleRate
            - preutteranceSeconds;
        const auto baseGainAt=[&](double t){return envelope!=nullptr?amplitudeGainAt(*envelope,t):1.0f;};
        auto gain=withEnvelope?(automatic.replacesStandardEnvelope&&envelope&&envelope->size()==4
            ?envelopeGainFromDb((*envelope)[1].gainDb):baseGainAt(localSeconds)):1.0f;
        if(withEnvelope&&(tailFade||headEnvelope))gain=advancedEnvelopeGain(tailFadeMode,localSeconds,
            tailFade?tailFade->startSeconds:0,tailFade?tailFade->endSeconds:0,
            headEnvelope?headEnvelope->startSeconds:0,headEnvelope?headEnvelope->endSeconds:0,tailFadeSettings,baseGainAt);
        return gain * automatic.gainAtSample(index);
    }
};

NoteMixGain noteMixGain(int samples, int destinationStart, double preutteranceSeconds,
                        const std::vector<UtauAmplitudePoint>& amplitudeEnvelope,
                        double fadeInSeconds, bool equalPowerFadeIn = false,
                        std::optional<int> sequenceFadeOutStart = std::nullopt,
                        double sequenceFadeOutSeconds = 0.0, bool equalPowerFadeOut = false)
{
    NoteMixGain gain;
    gain.envelope = &amplitudeEnvelope;
    gain.destinationStart = destinationStart;
    gain.preutteranceSeconds = preutteranceSeconds;
    const auto fadeIn = std::max(1, static_cast<int>(std::lround(
        std::max(0.003, fadeInSeconds) * mixSampleRate)));
    const auto fadeOut = std::max(1, static_cast<int>(std::lround(
        std::max(0.003, sequenceFadeOutSeconds) * mixSampleRate)));
    const auto naturalFadeOut = std::max(1, static_cast<int>(std::lround(0.012 * mixSampleRate)));
    gain.automatic = {mixSampleRate, -preutteranceSeconds, samples,
        {{0, fadeIn, true, equalPowerFadeIn}}};
    if (sequenceFadeOutStart)
        gain.automatic.ramps.push_back({*sequenceFadeOutStart-destinationStart, fadeOut, false, equalPowerFadeOut,0,-1,true});
    else
        gain.automatic.ramps.push_back({samples-1-naturalFadeOut, naturalFadeOut, false, false});
    return gain;
}

void mixNote(juce::AudioBuffer<float>& mix, const juce::AudioBuffer<float>& note,
             const NoteMixGain& gain)
{
    for (int index = 0; index < note.getNumSamples(); ++index)
    {
        const auto destination = gain.destinationStart + index;
        if (destination < 0 || destination >= mix.getNumSamples()) continue;
        const auto level = gain.at(index);
        for (int channel = 0; channel < mix.getNumChannels(); ++channel)
        {
            const auto sourceChannel = std::min(channel, note.getNumChannels() - 1);
            mix.addSample(channel, destination, note.getSample(sourceChannel, index) * level);
        }
    }
}

// How far past its own end each note has to go on sounding.
//
// The next note begins sounding a preutterance before its beat and fades in
// across its overlap; the note before it fades out over that same stretch,
// which ends overlap-minus-preutterance after the beat.  Where the overlap is
// the longer of the two that moment is past this note's end -- the tail of one
// syllable carries into the next, which is what an overlap is for.
//
// A note rendered only as far as its own end has nothing left to fade there.
// The fade was always scheduled correctly and simply ran off the end of the
// buffer, so raising the overlap past the twenty milliseconds every note
// carried did nothing at all.  This is the length UTAU asks its resampler
// for: the note, plus however much of the next note's overlap reaches back.
//
// Not past the next note's own end: beyond that the note after it is in
// charge of the seam, and an overlap typed larger than the note it belongs to
// would otherwise leave the previous syllable droning under the whole phrase.
std::vector<double> crossfadeTails(const UtauRenderRequest& request,
                                   const VoicebankIndex& voicebank)
{
    std::vector<double> tails(request.notes.size(), 0.0);
    std::vector<std::size_t> order(request.notes.size());
    std::iota(order.begin(), order.end(), std::size_t { 0 });
    std::stable_sort(order.begin(), order.end(), [&](auto left, auto right)
    {
        return request.notes[left].startSeconds < request.notes[right].startSeconds;
    });
    for (std::size_t position = 0; position + 1 < order.size(); ++position)
    {
        const auto index = order[position];
        const auto& note = request.notes[index];
        const auto& next = request.notes[order[position + 1]];
        if (isRestLyric(next.alias) || next.alias.trim().isEmpty()) continue;
        const auto* resolved = resolveSample(voicebank, next.alias, next.midiNote, next.aliasIsResolved);
        if (resolved == nullptr) continue;
        // The next note's own oto, when it has one, is where its lead-in and
        // overlap come from.
        std::optional<VoiceSample> own;
        if (next.oto.enabled)
            own = withNoteOto(*resolved, next.oto, request.fourRegion,
                              request.consonantClasses);
        const auto* sample = own ? &*own : resolved;
        // The same two numbers the renderer and the mixer work from.  An STP
        // moves the entry along the recording and leaves both untouched, so
        // the unshifted entry answers for them.
        const auto natural = std::min(
            adjustedPreutterance(*sample,
                headConsonantVelocity(*sample, next.consonantVelocity)),
            std::max(0.0, next.startSeconds));
        const auto nextPreutterance = next.preutteranceOverrideEnabled
            ? std::min(std::max(0.0, next.preutteranceSeconds),
                       std::max(0.0, next.startSeconds))
            : natural;
        const auto nextOverlap = next.overlapOverrideEnabled
            ? next.overlapSeconds : sample->overlap;
        const auto soundStart = next.startSeconds - nextPreutterance;
        const auto noteEnd = note.startSeconds + note.durationSeconds;
        // Exactly the case the mixer crossfades: where the next note starts
        // sounding after this one has finished there is no crossing at all.
        // Splicing is not an exception here -- it changes the shape of the
        // seam, not where it is, so the same tail must be rendered under it.
        // The old splice short-circuit left nothing to fade across, which is
        // the hollow/click heard at a spliced join.
        if (!UtauRenderer::crossfadesInto(soundStart, noteEnd)) continue;
        tails[index] = std::max(0.0,
            UtauRenderer::crossfadeEnd(soundStart, nextOverlap,
                next.startSeconds + next.durationSeconds) - noteEnd);
    }
    return tails;
}

std::string renderedNoteKey(const UtauRenderRequest& request, const VoiceSample& sample,
                            const UtauNoteRenderSpec& note, double effectivePreutterance,
                            double outputSeconds, double naturalPreutterance,
                            double naturalOutputSeconds)
{
    juce::MemoryOutputStream stream;
    stream.writeDouble(note.modulationPercent);
    stream.writeInt(15); // Timeline PIT and native expression parameters.
    stream.writeString(encodePitchbend(note, note.bpm > 0.0 ? note.bpm : request.bpm,
                                      naturalPreutterance, naturalOutputSeconds));
    stream.writeBool(request.fourRegion);
    stream.writeBool(request.consonantClasses);
    stream.writeBool(sample.hasRegions);
    for (const auto value : sample.regionSeconds) stream.writeDouble(value);
    stream.writeBool(note.jieSplitSet);
    for (const auto value : note.jieSplit) stream.writeDouble(value);
    stream.writeBool(note.flagCurve);
    for (const auto& [flag, points] : note.flagCurves)
    {
        const auto raw = flag.toUTF8();
        stream.write(raw.getAddress(), raw.sizeInBytes());
        for (const auto& [timeSeconds, value] : points)
        {
            stream.writeDouble(timeSeconds);
            stream.writeDouble(value);
        }
    }
    stream.writeBool(note.flagSplit);
    for (const auto& text : note.regionFlags)
    {
        const auto raw = text.toUTF8();
        stream.write(raw.getAddress(), raw.sizeInBytes());
    }
    const auto samplePath = sample.file.getFullPathName().toUTF8();
    stream.write(samplePath.getAddress(), samplePath.sizeInBytes());
    stream.writeInt64(sample.file.getLastModificationTime().toMilliseconds());
    stream.writeInt64(sample.file.getSize());
    stream.writeDouble(sample.offset);
    stream.writeDouble(sample.end);
    stream.writeDouble(sample.preutterance);
    stream.writeDouble(sample.consonant);
    stream.writeDouble(sample.overlap);
    stream.writeFloat(sample.sourceMidi);
    const auto resamplerPath = request.resamplerExecutable.getFullPathName().toUTF8();
    stream.write(resamplerPath.getAddress(), resamplerPath.sizeInBytes());
    stream.writeInt64(request.resamplerExecutable.getLastModificationTime().toMilliseconds());
    stream.writeInt64(request.resamplerExecutable.getSize());
    const auto flags = note.flags.toUTF8();
    stream.write(flags.getAddress(), flags.sizeInBytes());
    stream.writeFloat(note.midiNote);
    stream.writeFloat(note.gain);
    stream.writeFloat(note.nativeFormantSemitones);
    stream.writeFloat(note.nativeTension);
    stream.writeFloat(note.nativeBreath);
    stream.writeInt(note.consonantVelocity);
    stream.writeBool(note.preutteranceOverrideEnabled);
    stream.writeDouble(note.preutteranceSeconds);
    stream.writeBool(note.overlapOverrideEnabled);
    stream.writeDouble(note.overlapSeconds);
    stream.writeDouble(note.durationSeconds);
    stream.writeDouble(effectivePreutterance);
    stream.writeDouble(outputSeconds);
    stream.writeDouble(note.bpm > 0.0 ? note.bpm : request.bpm);
    stream.writeInt64(static_cast<juce::int64>(note.pitchCurve.size()));
    for (const auto& point : note.pitchCurve)
    {
        stream.writeDouble(point.timeSeconds);
        stream.writeFloat(point.cents);
    }
    return std::string(static_cast<const char*>(stream.getData()), stream.getDataSize());
}

struct RenderedNoteCache
{
    std::mutex mutex;
    std::unordered_map<std::string, RenderedNote> entries;
    std::deque<std::string> insertionOrder;
    std::size_t bytes = 0;
};

RenderedNoteCache& renderedNoteCache()
{
    static RenderedNoteCache cache;
    return cache;
}

bool restoreRenderedNote(const std::string& key, RenderedNote& destination)
{
    auto& cache = renderedNoteCache();
    const std::scoped_lock lock(cache.mutex);
    const auto found = cache.entries.find(key);
    if (found == cache.entries.end()) return false;
    destination = found->second;
    return true;
}

void storeRenderedNote(const std::string& key, const RenderedNote& rendered)
{
    constexpr std::size_t maximumBytes = 128u * 1024u * 1024u;
    auto& cache = renderedNoteCache();
    const std::scoped_lock lock(cache.mutex);
    if (cache.entries.contains(key)) return;
    const auto bytes = static_cast<std::size_t>(rendered.audio.getNumChannels())
        * static_cast<std::size_t>(rendered.audio.getNumSamples()) * sizeof(float);
    while (!cache.insertionOrder.empty() && cache.bytes + bytes > maximumBytes)
    {
        const auto oldest = cache.insertionOrder.front();
        cache.insertionOrder.pop_front();
        if (const auto found = cache.entries.find(oldest); found != cache.entries.end())
        {
            cache.bytes -= static_cast<std::size_t>(found->second.audio.getNumChannels())
                * static_cast<std::size_t>(found->second.audio.getNumSamples()) * sizeof(float);
            cache.entries.erase(found);
        }
    }
    cache.entries.emplace(key, rendered);
    cache.insertionOrder.push_back(key);
    cache.bytes += bytes;
}
}

juce::String UtauRenderer::diagnosticPitchbend(const UtauNoteRenderSpec& note,
    double bpm, double preutterance, double outputSeconds)
{
    return encodePitchbend(note, bpm, preutterance, outputSeconds);
}

void UtauRenderer::invalidateVoicebankCache()
{
    voicebankCacheRevision().fetch_add(1, std::memory_order_relaxed);
}

juce::String UtauRenderer::hfDaemonInterpreter(const juce::File& engineDirectory)
{
    juce::String interpreter = "pythonw";
    const auto config = engineDirectory.getChildFile("hf_backend").getChildFile("python.txt");
    if (!config.existsAsFile()) return interpreter;
    juce::MemoryBlock bytes;
    config.loadFileAsData(bytes);
    const auto* data = static_cast<const char*>(bytes.getData());
    const auto size = static_cast<int>(bytes.getSize());
    // The engine reads it with fopen, in the local code page; a UTF-8 file is
    // taken as UTF-8.
    juce::String text;
    if (juce::CharPointer_UTF8::isValidString(data, size))
        text = juce::String::fromUTF8(data, size);
#if JUCE_WINDOWS
    else if (size > 0)
    {
        const auto wide = MultiByteToWideChar(CP_ACP, 0, data, size, nullptr, 0);
        std::wstring buffer(static_cast<std::size_t>(std::max(0, wide)), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_ACP, 0, data, size, buffer.data(), wide);
        text = juce::String(buffer.c_str());
    }
#endif
    // First line; whitespace off its end and spaces and tabs off its start.
    auto line = text.upToFirstOccurrenceOf("\n", false, false);
    while (line.isNotEmpty() && static_cast<juce::juce_wchar>(line.getLastCharacter()) <= ' ')
        line = line.dropLastCharacters(1);
    while (line.startsWithChar(' ') || line.startsWithChar('\t'))
        line = line.substring(1);
    interpreter = line.isNotEmpty() ? line : juce::String("pythonw");
    // As the engine does: anything that is neither a drive path nor a UNC path
    // is taken from the engine's folder -- including a bare "pythonw" written
    // in the file, which the engine resolves the same way.
    const auto drive = interpreter.length() >= 2 && interpreter[1] == ':';
    const auto unc = interpreter.startsWith("\\\\");
    if (!drive && !unc)
        interpreter = engineDirectory.getFullPathName() + "\\" + interpreter;
    return interpreter;
}

bool UtauRenderer::startHfDaemonIfNeeded(const juce::File& resamplerExecutable, int port)
{
    const auto backend = resamplerExecutable.getParentDirectory().getChildFile("hf_backend");
    const auto script = backend.getChildFile("hf_daemon.py");
    if (!resamplerExecutable.existsAsFile() || !script.existsAsFile()) return false;
#if JUCE_WINDOWS
    auto& lifetime = hfDaemonLifetime();
    const std::scoped_lock lock(lifetime.mutex);
    if (lifetime.closing) return false;
    std::erase_if(lifetime.processes, [](const auto& owned)
    { return WaitForSingleObject(owned->process, 0) == WAIT_OBJECT_0; });
    // A warming process has not necessarily bound its socket yet. Retaining
    // it also avoids duplicate launches in that startup window.
    if (std::any_of(lifetime.processes.begin(), lifetime.processes.end(),
        [port](const auto& owned) { return owned->port == port; })) return false;
#endif
    {
        // A connection that says nothing is safe: the daemon answers it with
        // ERR and goes on serving.
        juce::StreamingSocket probe;
        if (probe.connect("127.0.0.1", port, 300)) return false;
    }
#if JUCE_WINDOWS
    const auto interpreter = hfDaemonInterpreter(resamplerExecutable.getParentDirectory());
    const auto commandText = "\"" + interpreter + "\" \"" + script.getFullPathName() + "\"";
    std::wstring commandLine(commandText.toWideCharPointer());
    auto owned = std::make_unique<OwnedHfDaemon>();
    owned->port = port;
    owned->job = CreateJobObjectW(nullptr, nullptr);
    if (owned->job == nullptr) return false;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(owned->job, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits))) return false;
    STARTUPINFOW startup {};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process {};
    // No inherited console or pipe handles. Bind the suspended child to our
    // job before it can load the model or spawn any further processes.
    const auto started = CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, FALSE,
        DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED, nullptr,
        backend.getFullPathName().toWideCharPointer(), &startup, &process);
    if (!started) return false;
    owned->process = process.hProcess;
    const auto assigned = AssignProcessToJobObject(owned->job, process.hProcess);
    const auto resumed = assigned && ResumeThread(process.hThread) != static_cast<DWORD>(-1);
    CloseHandle(process.hThread);
    if (!resumed)
    {
        // Only the suspended process just created here; never an existing
        // server. Do not leave a daemon alive when ownership cannot be set.
        TerminateProcess(process.hProcess, 1);
        return false;
    }
    lifetime.processes.push_back(std::move(owned));
    return true;
#else
    return false;
#endif
}

void UtauRenderer::shutdownHfDaemon()
{
#if JUCE_WINDOWS
    hfDaemonLifetime().shutdown();
#endif
}

void UtauRenderer::prewarmHfDaemon(const juce::File& resamplerExecutable)
{
    juce::Thread::launch([resamplerExecutable]
    {
        (void) startHfDaemonIfNeeded(resamplerExecutable);
    });
}

bool UtauRenderer::voicebankIndexReady(const juce::File& voicebankDirectory, bool fourRegion,
                                       bool consonantClasses, std::function<void()> whenReady)
{
    auto& cache = indexCache();
    const auto key = indexKey(voicebankDirectory, fourRegion, consonantClasses);
    std::shared_future<IndexPointer> pending;
    std::shared_ptr<std::promise<IndexPointer>> promise;
    {
        const std::scoped_lock lock(cache.mutex);
        if (lookupIndex(cache, key, voicebankDirectory, pending, promise) != nullptr)
            return true;
        if (whenReady) cache.entries[key].whenReady.push_back(std::move(whenReady));
    }
    if (promise != nullptr)
        indexReadingPool().addJob([key, voicebankDirectory, fourRegion, consonantClasses, promise]
        {
            readVoicebankIndex(key, voicebankDirectory, fourRegion, consonantClasses, *promise);
        });
    return false;
}

int UtauRenderer::diagnosticIndexBuilds()
{
    return indexCache().builds.load(std::memory_order_relaxed);
}

int UtauRenderer::diagnosticMessageThreadWaits()
{
    return indexCache().messageThreadWaits.load(std::memory_order_relaxed);
}

void UtauRenderer::diagnosticRecheckVoicebankFiles()
{
    auto& cache = indexCache();
    const std::scoped_lock lock(cache.mutex);
    for (auto& [key, entry] : cache.entries) entry.checkedAt = 0.0;
}

void UtauRenderer::recheckVoicebankFiles(const juce::File& root)
{
    auto& cache = indexCache();
    const std::scoped_lock lock(cache.mutex);
    const auto prefix = (root.getFullPathName()+"|").toStdString();
    for (auto& [key, entry] : cache.entries)
        if (key.starts_with(prefix)) {
            // The caller has detected a changed file set, including new WAVs
            // that did not exist when this index's stamps were captured.
            entry.index.reset(); entry.checkedAt = 0.0;
        }
}

juce::String UtauRenderer::diagnosticResolve(const juce::File& voicebankDirectory,
                                             const juce::String& alias, float midiNote,
                                             bool fourRegion, bool consonantClasses,
                                             bool byScan)
{
    const auto voicebank = loadVoicebankIndex(voicebankDirectory, fourRegion, consonantClasses);
    const auto* sample = byScan
        ? resolveSampleByScan(voicebank->samples, voicebank->prefixMap, alias, midiNote, voicebank->nativeAuthority)
        : resolveSample(*voicebank, alias, midiNote);
    if (sample == nullptr) return "none";
    return sample->file.getFullPathName() + "|" + sample->alias + "|"
        + juce::String(sample->offset, 6);
}

std::optional<UtauTailFadeSpan> UtauRenderer::tailFadeSpan(
    const UtauSampleTiming& timing,double soundStart,double soundEnd,double nominalDuration,
    int velocity,bool fourRegion,bool mou,const std::array<double,3>* manual)
{
    if(soundEnd-soundStart<=1.e-9)return std::nullopt;
    if(!fourRegion||!timing.hasRegions)
    {
        // Classic OTO has a fixed consonant followed by its stretching vowel.
        const auto naturalLead=std::max(0.0,timing.preutteranceSeconds);
        const auto consonant=std::max(0.0,timing.consonantSeconds);
        // A pinned Pre retimes the lead-in, keeping the part after the beat in place.
        const auto local=consonant<=naturalLead&&naturalLead>1.e-9
            ? soundStart+consonant*(-soundStart)/naturalLead : consonant-naturalLead;
        const auto start=juce::jlimit(soundStart,soundEnd,local);
        if(soundEnd-start<=1.e-6)return std::nullopt;return UtauTailFadeSpan{start,soundEnd};
    }
    const bool spelling=readsOnlyFirstTwoRegions(fourRegion,mou,nominalDuration);
    const int count=spelling?2:mou&&timing.mouClasses.length()>=2&&timing.mouClasses.length()<=4?timing.mouClasses.length():4;
    const auto source=spelling?firstTwoRegions(timing.regionSeconds):timing.regionSeconds;
    const auto split=regionSplit(source,soundEnd-soundStart,velocity,-soundStart,manual,
                                mou&&timing.mouClasses.isNotEmpty()?&timing.mouClasses:nullptr);
    if(!split.valid)return std::nullopt;
    auto start=soundStart;for(int i=0;i<count-1;++i)start+=split.seconds[(size_t)i];
    if(soundEnd-start<=1.e-6)return std::nullopt;return UtauTailFadeSpan{start,soundEnd};
}

std::optional<UtauTailFadeSpan> UtauRenderer::headEnvelopeSpan(
    const UtauSampleTiming& timing,double soundStart,double soundEnd,double nominalDuration,
    int velocity,bool fourRegion,bool mou,const std::array<double,3>* manual)
{
    if(soundEnd-soundStart<=1.e-9)return std::nullopt;
    auto end=soundEnd;
    if(!fourRegion||!timing.hasRegions)
    {
        const auto lead=std::max(0.0,timing.preutteranceSeconds),consonant=std::max(0.0,timing.consonantSeconds);
        end=juce::jlimit(soundStart,soundEnd,consonant<=lead&&lead>1.e-9?soundStart+consonant*(-soundStart)/lead:consonant-lead);
    }
    else
    {
        const auto source=readsOnlyFirstTwoRegions(fourRegion,mou,nominalDuration)?firstTwoRegions(timing.regionSeconds):timing.regionSeconds;
        const auto split=regionSplit(source,soundEnd-soundStart,velocity,-soundStart,manual,mou&&timing.mouClasses.isNotEmpty()?&timing.mouClasses:nullptr);
        if(!split.valid)return std::nullopt;end=soundStart+split.seconds[0];
    }
    if(end-soundStart<=1.e-6)return std::nullopt;return UtauTailFadeSpan{soundStart,end};
}

std::vector<UtauAmplitudePoint> UtauRenderer::fitAmplitudeEnvelope(
    const std::vector<UtauAmplitudePoint>& drawn, double preutteranceSeconds,
    double soundingEndSeconds)
{
    return envelopeForNoteAsItIs(drawn, preutteranceSeconds, soundingEndSeconds);
}

UtauRegionSplit UtauRenderer::regionSplit(const std::array<double, 4>& sourceSeconds,
                                          double outputSeconds, int consonantVelocity,
                                          double leadInSeconds,
                                          const std::array<double, 3>* manualFractions,
                                          const juce::String* classes)
{
    UtauRegionSplit split;
    auto sourceTotal = 0.0;
    for (const auto value : sourceSeconds) sourceTotal += std::max(0.0, value);
    if (sourceTotal <= 1.0e-9 || outputSeconds <= 1.0e-9) return split;

    // The lead-in is how much of the consonant is sung before the beat, and
    // the onset region is what fills it: the two are one thing seen from two
    // sides, not two independent numbers.  So the onset ends where the note
    // starts, and its length is the lead-in.
    //
    // The oto4 mark says where the consonant ends in the *source*; the engine
    // reads that itself and maps it onto the length sent here.  In the entries
    // of a bank that were marked by hand the two agree to about a millisecond,
    // which is what they are meant to be -- the same boundary, written down
    // twice.  Taking the onset's output length from the mark instead put the
    // boundary wherever the mark happened to fall, which for an unmarked entry
    // is the oto's fixed range, most of the way into the vowel.
    //
    // It does not follow the note's length, so stretching a note still leaves
    // the consonant alone; that was never what the lead-in depended on.
    const auto velocityScale = consonantVelocityScale(consonantVelocity);
    const auto onset = juce::jlimit(0.0, outputSeconds, leadInSeconds);

    // How many regions this entry actually has.  The ones past it do not
    // exist: they take no length, and nothing is allowed to spill into them.
    const auto annotated = classes != nullptr && classes->length() >= 2
        && classes->length() <= 4;
    const auto regions = static_cast<std::size_t>(annotated ? classes->length() : 4);

    if (manualFractions != nullptr)
    {
        // Hand-placed boundaries divide what is left after the onset, keeping
        // the proportions they were given.  They are stored as fractions of
        // the whole note, so they are read against where the onset really
        // ends rather than where the first of them says it does.  Only the
        // boundaries this entry has are read: with three regions there are
        // two, and the last one runs to the end whatever the third fraction
        // was left saying.
        split.seconds[0] = onset;
        const auto available = std::max(0.0, outputSeconds - onset);
        const auto pinnedFraction = juce::jlimit(0.0, 1.0, onset / outputSeconds);
        const auto rest = 1.0 - pinnedFraction;
        if (rest > 1.0e-9)
        {
            auto from = pinnedFraction;
            for (std::size_t index = 1; index < regions; ++index)
            {
                const auto to = index + 1 < regions
                    ? juce::jlimit(from, 1.0, (*manualFractions)[index]) : 1.0;
                split.seconds[index] = available * (to - from) / rest;
                from = to;
            }
        }
        else
            split.seconds[std::min<std::size_t>(2, regions - 1)] = available;
        split.valid = true;
        return split;
    }

    // Bounded water-filling, mirroring wcs_region_plan() in the engine's
    // wcs_regions.h, which stays the reference implementation.  Reproducing it
    // here is what lets the piano roll draw exactly what will be rendered.
    // Equal weights and no limits on the three vowel regions: filling shares
    // the surplus by w * natural, so equal weights share it by source length
    // and every region ends up scaled by the same ratio.  Their proportions
    // are therefore whatever the oto says, and stretching the note stretches
    // the vowel uniformly -- the classic two-region timeline, written as four
    // regions.  The onset takes no share and is pinned to its natural rate,
    // as the consonant is there.
    //
    // 谋-UTAU may say that some other region is a consonant too -- a Chinese
    // coda -n / -ng is one, sitting in region 3.  Every region marked 'C' is
    // then treated the way the onset always has been: no share of the note's
    // length, pinned to its natural rate, answering only to the consonant
    // velocity.  With no annotation the arrays below are what they always
    // were, so UTAU and 界-UTAU are untouched.
    const auto classOf = [&classes](std::size_t index)
    {
        // Beyond the count this entry declares there is no region at all;
        // calling it a vowel keeps it out of the pinning and its length is
        // zero anyway.
        const auto count = classes == nullptr ? 0 : classes->length();
        if (count < 2 || count > 4 || static_cast<int>(index) >= count) return 'V';
        const auto c = (*classes)[static_cast<int>(index)];
        return c == 'C' || c == 'S' ? 'C' : 'V';
    };
    // Region 0 is the onset, held to the lead-in -- unless the entry says
    // otherwise.  A VC sample's first region is a vowel tail, and pinning it
    // would leave the note with nothing that can stretch at all.  With no
    // class string region 0 is pinned as it always was.
    const auto pinned = [&](std::size_t index)
    {
        return annotated ? classOf(index) == 'C' : index == 0;
    };
    // Region 0 ends where the note starts, whatever it is.  The lead-in is how
    // much of the sample is sung before the beat and the first boundary is
    // that same instant seen from the other side, so it is not the class's to
    // move -- a first region called a vowel is still a vowel, and still ends
    // there.  The class goes on deciding how it is synthesised and whether the
    // consonant velocity reaches it.  With no annotation this is what pinned()
    // already said, so UTAU and 界-UTAU keep every number they had.
    const auto holdsLength = [&](std::size_t index)
    {
        return index == 0 || pinned(index);
    };
    // The consonant velocity is the consonant's, and in 谋 the entry says which
    // regions those are.  Unannotated it reaches regions 0 and 1, as it always
    // has -- the onset and the glide of a 界 entry.
    // Every region a consonant: the velocity is the front one's, and there is
    // no vowel here for the rest to be measured against, so only region 0
    // answers to it.  Region 0's own length is the lead-in, which the caller
    // has already worked out from the same class -- so what this decides is
    // that the regions behind it stop moving with it.
    auto allConsonants = annotated;
    for (std::size_t index = 0; index < regions; ++index)
        allConsonants = allConsonants && classOf(index) == 'C';
    const auto velocityReaches = [&](std::size_t index)
    {
        if (!annotated) return index < 2;
        return allConsonants ? index == 0 : pinned(index);
    };

    std::array<double, 4> weights {}, minimums {}, maximums {};
    for (std::size_t index = 0; index < 4; ++index)
    {
        weights[index] = holdsLength(index) ? 0.00 : 1.00;
        minimums[index] = holdsLength(index) ? 1.00 : 0.00;
        maximums[index] = holdsLength(index) ? 1.00 : 0.00;
    }

    std::array<double, 4> natural {}, low {}, high {}, out {};
    std::array<bool, 4> free {};
    for (std::size_t index = 0; index < 4; ++index)
    {
        // The onset is the lead-in, whole, so that it ends on the note
        // start.  Its natural length is the lead-in, not what the oto says:
        // the two are one boundary written twice.
        natural[index] = index >= regions ? 0.0
            : index == 0 ? onset
            : std::max(0.0, sourceSeconds[index])
                  * (velocityReaches(index) ? velocityScale : 1.0);
        low[index] = natural[index] * minimums[index];
        high[index] = maximums[index] > 0.0 ? natural[index] * maximums[index] : 1.0e12;
        if (high[index] < low[index]) high[index] = low[index];
        out[index] = natural[index];
        free[index] = index < regions && natural[index] > 1.0e-9;
    }

    for (auto iteration = 0; iteration < 16; ++iteration)
    {
        auto remaining = outputSeconds;
        for (const auto value : out) remaining -= value;
        if (std::abs(remaining) < 1.0e-9) break;
        auto weightSum = 0.0;
        for (std::size_t index = 0; index < 4; ++index)
            if (free[index]) weightSum += weights[index] * natural[index];
        if (weightSum <= 1.0e-12) break;
        auto changed = false;
        for (std::size_t index = 0; index < 4; ++index)
        {
            if (!free[index]) continue;
            auto value = out[index] + remaining * weights[index] * natural[index] / weightSum;
            if (value < low[index]) { value = low[index]; free[index] = false; changed = true; }
            else if (value > high[index]) { value = high[index]; free[index] = false; changed = true; }
            out[index] = value;
        }
        if (!changed) break;
    }

    auto remaining = outputSeconds;
    for (const auto value : out) remaining -= value;
    if (std::abs(remaining) > 1.0e-9)
    {
        // Whatever is left over goes to a vowel, never to a consonant: giving
        // it to region 2 regardless would stretch a coda marked 'C' by the
        // back door, which is the whole thing this is meant to stop.
        auto sink = regions > 2 ? std::size_t{2} : std::size_t{1};
        if (annotated && holdsLength(sink))
        {
            sink = 4;
            // From region 1: region 0 ends at the note start and has nothing
            // to give.
            for (std::size_t index = 1; index < regions; ++index)
                if (!holdsLength(index) && natural[index] > 1.0e-9) sink = index;
        }
        if (sink < 4 && natural[sink] > 1.0e-9)
            out[sink] = std::max(0.0, out[sink] + remaining);
    }
    auto total = 0.0;
    for (const auto value : out) total += value;
    if (total > 1.0e-9 && std::abs(total - outputSeconds) > 1.0e-9)
    {
        // Scaling everything alike would undo the pinning, so the consonants
        // keep their length and the vowels absorb the correction between
        // them.  With no annotation only region 0 is pinned, which is what
        // the engine's own normalisation does, and the result is the same
        // numbers as before.
        auto held = 0.0, elastic = 0.0;
        for (std::size_t index = 0; index < regions; ++index)
            (holdsLength(index) ? held : elastic) += out[index];
        const auto room = outputSeconds - held;
        if (elastic > 1.0e-9 && room > 0.0)
        {
            const auto scale = room / elastic;
            for (std::size_t index = 0; index < regions; ++index)
                if (!holdsLength(index)) out[index] *= scale;
        }
        else
        {
            // Nothing may stretch, and the note still has to be its own
            // length.  Region 0 is the one thing that cannot move -- it ends
            // where the note starts -- so the regions after it take the
            // correction between them.
            const auto rest = total - out[0];
            const auto after = std::max(0.0, outputSeconds - out[0]);
            if (rest > 1.0e-9)
                for (std::size_t index = 1; index < regions; ++index)
                    out[index] *= after / rest;
            else if (regions > 1)
                out[1] = after;
        }
    }

    split.seconds = out;
    split.valid = true;
    return split;
}

double UtauRenderer::crossfadeEnd(double soundStart, double overlap,
                                  double nextNoteEnd)
{
    return std::min(soundStart + overlap, nextNoteEnd);
}

bool UtauRenderer::crossfadesInto(double soundStart, double noteEnd)
{
    return soundStart <= noteEnd + 1.0e-6;
}

bool UtauRenderer::sendsFlagCurves(bool noteFlagCurve)
{
    return noteFlagCurve;
}

bool UtauRenderer::readsOnlyFirstTwoRegions(bool fourRegion, bool consonantClasses,
                                            double durationSeconds)
{
    return fourRegion && !consonantClasses && durationSeconds <= 1.0e-12;
}

std::array<double, 4> UtauRenderer::firstTwoRegions(
    const std::array<double, 4>& regionSeconds)
{
    return { regionSeconds[0], regionSeconds[1], 0.0, 0.0 };
}

std::optional<UtauSampleTiming> UtauRenderer::sampleTiming(
    const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
    int consonantVelocity, bool fourRegion, bool consonantClasses)
{
    return sampleTiming(voicebankDirectory, alias, midiNote, consonantVelocity,
                        fourRegion, consonantClasses, nullptr);
}

std::optional<UtauSampleTiming> UtauRenderer::sampleTiming(
    const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
    int consonantVelocity, bool fourRegion, bool consonantClasses,
    const UtauOtoOverride* noteOto)
{
    // No look at the disk here: the cache knows whether the folder is still
    // there, and asking again for every note was part of what an edit cost.
    if (alias.trim().isEmpty() || isRestLyric(alias)) return std::nullopt;
    const auto voicebank = loadVoicebankIndex(voicebankDirectory, fourRegion,
                                             consonantClasses);
    if (const auto* resolved = resolveSample(*voicebank, alias, midiNote))
    {
        // Read on every repaint for every note, so a note without its own oto
        // is answered from the shared entry and nothing is copied for it.
        std::optional<VoiceSample> own;
        if (noteOto != nullptr && noteOto->enabled)
            own = withNoteOto(*resolved, *noteOto, fourRegion, consonantClasses);
        const auto* sample = own ? &*own : resolved;
        const auto velocity = headConsonantVelocity(*sample, consonantVelocity);
        const auto scale = consonantVelocityScale(velocity);
        return UtauSampleTiming { adjustedPreutterance(*sample, velocity),
                                  sample->consonant * scale, sample->overlap,
                                  sample->hasRegions, sample->regionSeconds,
                                  sample->mouClasses,
                                  std::max(0.0, sample->end - sample->offset),
                                  sample->alias };
    }
    return std::nullopt;
}

UtauRenderer::ResolvedSample UtauRenderer::resolveVoiceSample(
    const juce::File& voicebankDirectory, const juce::String& alias, float midiNote,
    int consonantVelocity, bool fourRegion, bool consonantClasses, double stpSeconds,
    bool preutteranceOverrideEnabled, double preutteranceSecondsOverride,
    bool overlapOverrideEnabled, double overlapSecondsOverride,
    const UtauOtoOverride* noteOto, bool aliasIsResolved)
{
    ResolvedSample resolved;
    if (!voicebankDirectory.isDirectory() || alias.trim().isEmpty()
        || isRestLyric(alias))
        return resolved;
    const auto voicebank = loadVoicebankIndex(voicebankDirectory, fourRegion,
                                              consonantClasses);
    const auto* found = ::hachi::backend::resolveSample(
        *voicebank, alias, midiNote, aliasIsResolved);
    if (found == nullptr) return resolved;
    resolved.sourceOto = found->sourceOto;
    resolved.resolvedAlias = found->alias;
    // STP moves the whole entry inside the recording, exactly as the render
    // path does, so the region read here matches what would be sung.
    const auto entry = noteOto != nullptr && noteOto->enabled
        ? withNoteOto(*found, *noteOto, fourRegion, consonantClasses) : *found;
    const auto sample = shiftedBy(entry, stpSeconds);
    const auto velocity = headConsonantVelocity(sample, consonantVelocity);
    resolved.found = true;
    resolved.file = sample.file;
    resolved.offsetSeconds = sample.offset;
    resolved.endSeconds = sample.end;
    resolved.fileSeconds = sample.fileSeconds;
    resolved.preutteranceSeconds = preutteranceOverrideEnabled
        ? std::max(0.0, preutteranceSecondsOverride)
        : adjustedPreutterance(sample, velocity);
    resolved.consonantSeconds = sample.consonant;
    resolved.overlapSeconds = overlapOverrideEnabled ? overlapSecondsOverride
                                                     : sample.overlap;
    resolved.sourceMidi = sample.sourceMidi;
    resolved.hasRegions = fourRegion && sample.hasRegions;
    resolved.regionSeconds = resolved.hasRegions ? sample.regionSeconds : std::array<double, 4>{};
    resolved.mouClasses = consonantClasses ? sample.mouClasses : juce::String();
    return resolved;
}

bool UtauRenderer::supportsComponentExport(const juce::File& engine,
                                               const juce::String& flags)
{
    if (!engine.existsAsFile() || !engine.getFileNameWithoutExtension().startsWithIgnoreCase("WCSNDM"))
        return false;
    // Match WCSNDM's first-wins tokenizer, including the shared kernel slot.
    juce::String kernel;
    double kernelValue = 0.0;
    bool direct = false, modelMix = false, seenMix = false;
    const juce::StringArray pairs { "bh", "SK", "LD", "PB", "Rd", "Ab", "NA", "RG",
        "DN", "JZ", "Jt", "BX", "TM", "QX", "CR", "HF", "HN", "OD", "FC", "FQ" };
    for (int i = 0; i < flags.length();)
    {
        if (!juce::CharacterFunctions::isLetter(flags[i])) { ++i; continue; }
        auto key = flags.substring(i, i + 1); ++i;
        if (i < flags.length() && juce::CharacterFunctions::isLetter(flags[i])
            && (key == "M" || pairs.contains(key + flags.substring(i, i + 1))))
            { key += flags.substring(i, i + 1); ++i; }
        auto end = i;
        if (end < flags.length() && (flags[end] == '+' || flags[end] == '-')) ++end;
        while (end < flags.length() && juce::CharacterFunctions::isDigit(flags[end])) ++end;
        const auto hasNumber = end > i;
        const auto value = flags.substring(i, end).getDoubleValue();
        i = end;
        if (kernel.isEmpty() && (key == "K" || key == "L" || key == "V" || key == "M" || key == "HF" || (key == "HN" && hasNumber && value == 1.0)))
        { kernel = key; kernelValue = value; }
        if (key == "u") direct = true;
        if (key == "Mm" && !seenMix) { seenMix = true; modelMix = true; }
    }
    return !direct && !modelMix && (kernel == "K" || kernel == "L")
        && static_cast<int>(kernelValue + 0.5) == 2;
}

std::optional<UtauSampleTiming> UtauRenderer::mappedAliasTiming(
    const juce::File& directory, const juce::String& alias, float midiNote)
{
    if (alias.trim().isEmpty()) return std::nullopt;
    const auto bank = loadVoicebankIndex(directory, false, false);
    juce::StringArray candidates;
    if (const auto it = bank->prefixMap.find(midiName(midiNote).toLowerCase()); it != bank->prefixMap.end())
        candidates.add(it->second.first + alias.trim() + it->second.second);
    candidates.add(alias.trim());
    for (const auto& candidate : candidates)
        if (const auto* found = indexedAlias(*bank, candidate, true))
        {
            const auto& sample = *found;
            return UtauSampleTiming { sample.preutterance, sample.consonant, sample.overlap,
                sample.hasRegions, sample.regionSeconds, sample.mouClasses,
                std::max(0.0, sample.end - sample.offset), sample.alias };
        }
    return std::nullopt;
}

VoicebankFamilyResolution resolveRequestedVoicebankFamily(const UtauRenderRequest& request,
                                                           bool validateUtauSamples)
{
    VoicebankFamilyResolution resolution;
    const bool dsBank = DiffSingerRenderer::isVoicebank(request.voicebankDirectory);
    const auto legacy = dsBank ? RequestedVoicebankFamily::diffSinger : RequestedVoicebankFamily::utau;
    resolution.family = request.requestedFamily == RequestedVoicebankFamily::legacyAuto
        ? legacy : request.requestedFamily;
    if (request.requestedFamily != RequestedVoicebankFamily::legacyAuto && resolution.family != legacy)
        resolution.cacheDiscriminator = resolution.family == RequestedVoicebankFamily::diffSinger
            ? "explicit-diffsinger-family-v1" : "explicit-utau-family-v1";
    if (request.requestedFamily == RequestedVoicebankFamily::diffSinger && !dsBank)
    {
        resolution.warning = "DiffSinger track requires a compatible voicebank containing dsconfig.yaml; UTAU/piano fallback is disabled";
        return resolution;
    }
    if (validateUtauSamples && request.requestedFamily == RequestedVoicebankFamily::utau && dsBank)
    {
        // A mixed-format bank is valid when the chosen UTAU path can actually
        // resolve samples. Reuse the renderer's cached index, never scan from
        // scheduling/UI/cache-key code and never switch family to make it work.
        const auto bank = loadVoicebankIndex(request.voicebankDirectory, request.fourRegion, request.consonantClasses);
        if (bank->samples.empty())
            resolution.warning = "UTAU track selected a DiffSinger-only bank with no readable UTAU samples; DiffSinger/piano fallback is disabled";
        else if (!request.chineseCvvc || request.cvvcPhonemes || request.fourRegion)
            for (const auto& note : request.notes)
            {
                if (isRestLyric(note.alias) || note.alias.trim().isEmpty()) continue;
                if (resolveSample(*bank, note.alias, note.midiNote, note.aliasIsResolved) == nullptr)
                {
                    resolution.warning = "UTAU track cannot resolve alias '" + note.alias
                        + "' in the selected mixed/DiffSinger bank; DiffSinger/piano fallback is disabled";
                    break;
                }
            }
    }
    return resolution;
}

juce::AudioBuffer<float> UtauRenderer::pianoPreview(const UtauNoteRenderSpec& note, double sampleRate)
{
    return renderPianoPreview(note, sampleRate);
}

float UtauRenderer::amplitudeGain(const std::vector<UtauAmplitudePoint>& envelope, double localSeconds)
{
    return amplitudeGainAt(envelope, localSeconds);
}

UtauRenderResult UtauRenderer::render(const UtauRenderRequest& request)
{
    const auto family = resolveRequestedVoicebankFamily(request, true);
    if (family.warning.isNotEmpty())
    {
        UtauRenderResult failure; failure.warning = family.warning; return failure;
    }
    const bool diffSinger = family.family == RequestedVoicebankFamily::diffSinger;
    if (request.chineseCvvc && !request.fourRegion && !diffSinger)
        return renderChineseCvvc(request, [](const auto& expanded) { return UtauRenderer::render(expanded); });
    if (request.exportComponent != WavExportComponent::full)
    {
        UtauRenderResult failure;
        failure.warning = "Selected UTAU engine/kernel cannot export separated voice components";
        if (diffSinger) return failure;
        auto separated = request;
        separated.exportComponent = WavExportComponent::full;
        const auto flag = request.exportComponent == WavExportComponent::breath ? "Mx2" : "Mx1";
        for (auto& note : separated.notes)
        {
            if (isRestLyric(note.alias) || (request.silenceEmptyLyrics && note.alias.trim().isEmpty())) continue;
            if (note.alias.trim().isEmpty() || !supportsComponentExport(request.resamplerExecutable, note.flags))
                return failure;
            note.flags = juce::String(flag) + note.flags;
            // Region flags replace the frame flags; give every region the same
            // output selector while retaining the user's other settings.
            for (auto& region : note.regionFlags)
                if (region.isNotEmpty()) region = juce::String(flag) + region;
        }
        auto result = render(separated);
        if (result.warning.isNotEmpty() || result.backend.contains("fallback")
            || result.backend.contains("piano") || !result.backend.contains("utau-resampler"))
        {
            failure.warning = "Separated export requires successful external synthesis: " + result.warning;
            return failure;
        }
        result.backend += request.exportComponent == WavExportComponent::breath ? "+breath" : "+non-breath";
        return result;
    }
    if (diffSinger)
        return DiffSingerRenderer::render(request);
    UtauRenderResult result;
    if (request.requireExternalResampler && !request.resamplerExecutable.existsAsFile()) {
        result.warning = "Selected resampler does not exist: " + request.resamplerExecutable.getFullPathName();
        return result;
    }
    if (request.wavtoolExecutable != juce::File{} && !request.wavtoolExecutable.existsAsFile()) {
        result.warning = "Selected wavtool does not exist: " + request.wavtoolExecutable.getFullPathName();
        return result;
    }
    result.sampleRate = mixSampleRate;
    if (request.progress) request.progress(0.01);
    const auto voicebank = loadVoicebankIndex(request.voicebankDirectory,
                                              request.fourRegion,
                                              request.consonantClasses);
    const auto& samples = voicebank->samples;
    if (samples.empty())
    {
        if (std::any_of(request.notes.begin(), request.notes.end(), [](const auto& note)
            { return note.alias.trim().isNotEmpty() && !isRestLyric(note.alias); }))
            result.warning = "UTAU voicebank contains no readable audio";
        if (request.voicebankDirectory.getChildFile("otto.ini").exists())
        {
            result.warning = "Native OTTO material contains no readable pronunciations; check its annotations";
            return result;
        }
        // Unlabelled MIDI notes still have the built-in piano preview, so an
        // unavailable voicebank must not abort the whole mixed request.
    }
    if (request.progress) request.progress(0.05);
    const auto& prefixMap = voicebank->prefixMap;
    const auto totalSamples = std::max(1, static_cast<int>(std::ceil(
        std::max(0.001, request.targetDurationSeconds) * mixSampleRate)));
    result.buffer.setSize(2, totalSamples);
    result.buffer.clear();
    std::vector<RenderedNote> renderedNotes(request.notes.size());
    // How much of each note's tail the note after it reaches back into.
    const auto crossfadeTailSeconds = crossfadeTails(request, *voicebank);
    std::vector<std::pair<double, double>> prioritySpans;
    prioritySpans.reserve(request.notes.size());
    for (const auto& note : request.notes)
        prioritySpans.emplace_back(request.timelineStartSeconds + note.startSeconds,
            request.timelineStartSeconds + note.startSeconds + note.durationSeconds);
    PlaybackNoteOrder nextNote(std::move(prioritySpans));
    std::atomic<std::size_t> completedNotes { 0 };
    std::atomic<bool> externalResamplerHealthy {
        request.resamplerExecutable.existsAsFile()
    };
    juce::String externalFailure;
    const auto renderOne = [&]
    {
        for (;;)
        {
            if (request.cancelled && request.cancelled()) break;
            const auto next = nextNote.take(request.priorityPosition
                ? std::optional<double>{request.priorityPosition()} : std::nullopt);
            if (!next) break;
            const auto index = *next;
            const auto& note = request.notes[index];
            auto& destination = renderedNotes[index];
            if (isRestLyric(note.alias))
            {
                // Silent by request.  Found, so it is not reported as an alias
                // the voicebank is missing, and with no audio nothing is mixed
                // -- the note holds its stretch of the phrase open.
                destination.found = true;
                destination.rest = true;
                destination.preutterance = 0.0;
                destination.overlap = 0.0;
            }
            else if (note.alias.trim().isEmpty())
            {
                destination.found = true;
                destination.preutterance = 0.0;
                if (request.pianoPreviewEnabled && !request.silenceEmptyLyrics)
                {
                    // A loaded bank does not change unlabelled-note audition.
                    destination.piano = true;
                    destination.overlap = 0.004;
                    destination.audio = renderPianoPreview(note);
                }
                else
                {
                    // Offline export keeps the unlabelled note's time silent.
                    destination.rest = true;
                    destination.overlap = 0.0;
                }
            }
            else if (const auto* found = resolveSample(
                         *voicebank, note.alias, note.midiNote, note.aliasIsResolved))
            {
                destination.found = true;
                // The note's own STP first, and then nothing below knows the
                // difference: the entry it works from is already the shifted
                // one, so the arguments, the native path and the cache key all
                // follow it without a second rule.
                // This note's own oto first, when it has one, then its STP:
                // the STP moves whatever entry the note is sung from.
                auto sample = shiftedBy(
                    note.oto.enabled ? withNoteOto(*found, note.oto, request.fourRegion,
                                                   request.consonantClasses)
                                     : *found,
                    note.stpSeconds);
                // 界: a 拼字 note reads the onset and the glide and nothing
                // after them.  The recording is cut where the second region
                // ends, so neither the engine nor the native path has anything
                // past it to read.  The engine gives a region with no source no
                // time, so cut there the nucleus and coda take none of the note
                // without being told separately.  Left whole, all four were
                // squeezed into what a note of no length sounds, and whatever
                // lay behind the glide came out at the end of it.
                if (sample.hasRegions && UtauRenderer::readsOnlyFirstTwoRegions(
                        request.fourRegion, request.consonantClasses,
                        note.durationSeconds))
                {
                    const auto kept = std::max(0.001,
                        sample.regionSeconds[0] + sample.regionSeconds[1]);
                    sample.end = std::min(sample.end, sample.offset + kept);
                    sample.preutterance = std::min(sample.preutterance,
                                                   sample.end - sample.offset);
                    sample.consonant = std::min(sample.consonant,
                                                sample.end - sample.offset);
                }
                // Velocity is applied first.  A per-note Pre value then
                // stretches/compresses that already velocity-adjusted lead-in
                // without changing the following vowel/tail duration.
                const auto naturalPreutterance = std::min(
                    adjustedPreutterance(sample,
                        headConsonantVelocity(sample, note.consonantVelocity)),
                    std::max(0.0, note.startSeconds));
                destination.preutterance = note.preutteranceOverrideEnabled
                    ? std::min(std::max(0.0, note.preutteranceSeconds),
                               std::max(0.0, note.startSeconds))
                    : naturalPreutterance;
                destination.overlap = note.overlapOverrideEnabled
                    ? note.overlapSeconds : sample.overlap;
                // Long enough to still be sounding where the next note's
                // overlap fades it out, or there is nothing there to fade.
                const auto tail = crossfadeTailSeconds[index];
                const auto naturalOutputSeconds = std::max(0.03,
                    naturalPreutterance + note.durationSeconds + 0.02 + tail);
                const auto finalOutputSeconds = std::max(0.03,
                    destination.preutterance + note.durationSeconds + 0.02 + tail);
                destination.timing=UtauSampleTiming{naturalPreutterance,
                    sample.consonant*consonantVelocityScale(headConsonantVelocity(sample,note.consonantVelocity)),
                    destination.overlap,sample.hasRegions,sample.regionSeconds,sample.mouClasses};
                if (note.preserveEnvelopeTiming && note.gain <= 0)
                {
                    // Retain the neighbour's OTO seam without synthesising unselected context.
                    destination.audio.setSize(1, 1);
                    destination.audio.clear();
                    continue;
                }
                const auto cacheKey = renderedNoteKey(request, sample, note,
                    destination.preutterance, finalOutputSeconds,
                    naturalPreutterance, naturalOutputSeconds);
                if (restoreRenderedNote(cacheKey, destination))
                {
                }
                else
                {
                    auto external = externalResamplerHealthy.load(std::memory_order_acquire)
                        ? runExternalResampler(request, sample, note,
                            naturalPreutterance, naturalOutputSeconds)
                        : ExternalResamplerAttempt {};
                    // Cancelling an external request is not an engine error;
                    // do not begin a potentially long native fallback while
                    // the application's render queue is waiting to stop.
                    if (request.cancelled && request.cancelled()) break;
                    if (external.succeeded)
                    {
                        destination.audio = std::move(external.audio);
                        retimeLeadIn(destination.audio, naturalPreutterance,
                                     destination.preutterance);
                        destination.external = true;
                    }
                    else
                    {
                        // Once an executable rejects or times out on a request,
                        // do not repeat that wait for every remaining note in the
                        // same selection.  The native resampler remains audible.
                        externalResamplerHealthy.store(false, std::memory_order_release);
                        if (externalFailure.isEmpty())
                            externalFailure = external.error.isNotEmpty()
                                ? external.error : juce::String("external resampler was disabled after an earlier failure");
                        destination.audio = renderNative(sample, note,
                                                         naturalOutputSeconds);
                        retimeLeadIn(destination.audio, naturalPreutterance,
                                     destination.preutterance);
                    }
                    // Never cache an internal fallback under an external-engine
                    // key.  A transient launch/error must be retried next time.
                    if (destination.external || !request.resamplerExecutable.existsAsFile())
                        storeRenderedNote(cacheKey, destination);
                }
            }
            else
            {
                // A lyric the voicebank has no sample for still sounds: the
                // piano preview at the note's pitch, so the melody is heard
                // while the lyric is put right.  It is still reported as
                // missing below -- that report is what tells a typo from a
                // note left without a lyric on purpose.
                destination.found = true;
                destination.missing = true;
                destination.piano = request.pianoPreviewEnabled;
                destination.preutterance = 0.0;
                destination.overlap = request.pianoPreviewEnabled ? 0.004 : 0.0;
                destination.rest = !request.pianoPreviewEnabled;
                if (request.pianoPreviewEnabled) destination.audio = renderPianoPreview(note);
            }
            const auto done = completedNotes.fetch_add(1, std::memory_order_relaxed) + 1;
            if (request.progress && !request.notes.empty())
                request.progress(0.05 + 0.90 * static_cast<double>(done)
                    / static_cast<double>(request.notes.size()));
        }
    };
    // Classic UTAU resamplers are commonly not safe to invoke concurrently.
    // Keep their calls sequential; the model-free native route can still use
    // several workers.
    const auto workerCount = request.resamplerExecutable.existsAsFile()
        ? std::size_t { 1 }
        : std::max<std::size_t>(1,
            std::min<std::size_t>({ 4, request.notes.size(),
                static_cast<std::size_t>(std::max(1, juce::SystemStats::getNumCpus())) }));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker)
        workers.emplace_back(renderOne);
    for (auto& worker : workers) worker.join();

    if (request.cancelled && request.cancelled()) return {};

    auto externalCount = 0;
    auto nativeCount = 0;
    auto pianoCount = 0;
    auto missingCount = 0;
    std::vector<std::size_t> mixOrder(request.notes.size());
    std::iota(mixOrder.begin(), mixOrder.end(), std::size_t { 0 });
    std::stable_sort(mixOrder.begin(), mixOrder.end(), [&](auto left, auto right)
    {
        return request.notes[left].startSeconds < request.notes[right].startSeconds;
    });
    for (std::size_t orderIndex = 0; orderIndex < mixOrder.size(); ++orderIndex)
    {
        const auto index = mixOrder[orderIndex];
        const auto& note = request.notes[index];
        auto& rendered = renderedNotes[index];
        if (!rendered.found)
        {
            ++missingCount;
            continue;
        }
        if (rendered.rest || (note.preserveEnvelopeTiming && note.gain <= 0)) continue;
        if (rendered.missing) ++missingCount;
        if (rendered.piano) ++pianoCount;
        else if (rendered.external) ++externalCount;
        else ++nativeCount;
        if (rendered.audio.getNumSamples() <= 0) continue;
        const auto start = static_cast<int>(std::lround(
            (note.startSeconds - rendered.preutterance) * mixSampleRate));
        std::optional<int> sequenceFadeOutStart;
        auto sequenceFadeOutSeconds = 0.0;
        auto equalPowerFadeOut = false;
        auto crossfadeVisible = false;
        if (orderIndex + 1 < mixOrder.size())
        {
            const auto nextIndex = mixOrder[orderIndex + 1];
            const auto& followingNote = request.notes[nextIndex];
            const auto& nextRendered = renderedNotes[nextIndex];
            if (nextRendered.found && nextRendered.audio.getNumSamples() > 0)
            {
                const auto nextSoundStart = followingNote.startSeconds
                    - nextRendered.preutterance;
                const auto currentNominalEnd = note.startSeconds + note.durationSeconds;
                const auto nextNominalEnd = followingNote.startSeconds
                    + followingNote.durationSeconds;
                if (UtauRenderer::crossfadesInto(nextSoundStart,
                                                 currentNominalEnd))
                {
                    if (nextRendered.overlap >= 0.0)
                    {
                        crossfadeVisible = nextRendered.overlap > 0.0;
                        sequenceFadeOutStart = static_cast<int>(std::lround(
                            nextSoundStart * mixSampleRate));
                        sequenceFadeOutSeconds = UtauRenderer::crossfadeEnd(
                            nextSoundStart, nextRendered.overlap, nextNominalEnd)
                            - nextSoundStart;
                        // Splicing chooses the crossfade curve and nothing
                        // else: a quarter-sine pair holds the level across the
                        // seam.  It must not also cap the fade span at this
                        // note's own end -- that cut off the part of the
                        // overlap reaching past the beat, which is the very
                        // seam the splice exists to smooth.
                        equalPowerFadeOut = followingNote.splice;
                    }
                    else
                    {
                        // A negative overlap is a literal silent interval:
                        // finish the previous note before this note's actual
                        // sounding start by abs(overlap), with a short declick.
                        constexpr auto declickSeconds = 0.003;
                        const auto previousFadeEnd = nextSoundStart
                            + nextRendered.overlap;
                        sequenceFadeOutStart = static_cast<int>(std::lround(
                            (previousFadeEnd - declickSeconds) * mixSampleRate));
                        sequenceFadeOutSeconds = declickSeconds;
                    }
                }
            }
        }
        const auto envelope = note.preserveEnvelopeTiming ? note.amplitudeEnvelope : fitAmplitudeEnvelope(note.amplitudeEnvelope,
            rendered.preutterance, note.durationSeconds + crossfadeTailSeconds[index]);
        auto gain = noteMixGain(rendered.audio.getNumSamples(), start,
                                      rendered.preutterance, envelope,
                                      rendered.overlap,
                                      note.splice && rendered.overlap > 0.0,
                                      sequenceFadeOutStart, sequenceFadeOutSeconds,
                                      equalPowerFadeOut);
        gain.automatic.ramps.back().crossfade = crossfadeVisible;
        if(request.cvvcPhonemes)
        {
            bool preceding=false;
            if(orderIndex>0)
            {
                const auto previous=mixOrder[orderIndex-1];
                preceding=renderedNotes[previous].found&&!renderedNotes[previous].rest
                    &&UtauRenderer::crossfadesInto(note.startSeconds-rendered.preutterance,
                        request.notes[previous].startSeconds+request.notes[previous].durationSeconds);
            }
            gain.automatic.ramps.front().crossfade=preceding&&rendered.overlap>0;
            for(auto& ramp:gain.automatic.ramps)if(ramp.crossfade)
            {ramp.equalPower=false;ramp.smoothCurve=request.cvvcCurveCrossfade;}
            if(note.cvvcStandardEnvelope)
            {
                if(!sequenceFadeOutStart)gain.automatic.ramps.pop_back();
                useStandardUtauEnvelope(gain.automatic,envelope.back().timeSeconds-envelope.front().timeSeconds);
            }
        }
        if((note.tailFadeMode!=0||note.tailFadeSettings.head.mode!=0)&&rendered.timing)
        {
            auto end=note.durationSeconds+crossfadeTailSeconds[index];
            if(sequenceFadeOutStart)end=std::max(-rendered.preutterance+.001,
                (*sequenceFadeOutStart/mixSampleRate+sequenceFadeOutSeconds)-note.startSeconds);
            gain.tailFade=tailFadeSpan(*rendered.timing,-rendered.preutterance,end,
                note.durationSeconds,note.consonantVelocity,request.fourRegion,request.consonantClasses,
                note.jieSplitSet?&note.jieSplit:nullptr);
            gain.headEnvelope=headEnvelopeSpan(*rendered.timing,-rendered.preutterance,end,
                note.durationSeconds,note.consonantVelocity,request.fourRegion,request.consonantClasses,
                note.jieSplitSet?&note.jieSplit:nullptr);
            gain.tailFadeMode=note.tailFadeMode;
            gain.tailFadeSettings=note.tailFadeSettings;
        }
        if (request.noteMixEnvelope) request.noteMixEnvelope(index, gain.automatic);
        // Handed over before it is mixed, which is the only moment it is
        // this note's audio and nothing else's, with the very gain the mix
        // is about to apply, so that what is drawn is shaped as it is heard.
        if (request.notePiece && rendered.audio.getNumSamples() > 0)
        {
            const auto sampleAt = [preutterance = rendered.preutterance](double localSeconds)
            {
                return static_cast<int>(std::llround(
                    (localSeconds + preutterance) * mixSampleRate));
            };
            request.notePiece(index, rendered.audio, mixSampleRate, rendered.preutterance,
                              [&gain, sampleAt](double localSeconds)
                              {
                                  return gain.at(sampleAt(localSeconds));
                              },
                              [&gain, &envelope, sampleAt](double localSeconds)
                              {
                                  // Inside the envelope's span its shape is
                                  // left out -- that is what the lane's line is
                                  // there to change.  Outside it the envelope
                                  // holds its end value, which no point can
                                  // move, so that part stays: past its last
                                  // point a piece is silence however it is
                                  // drawn, and was showing as sound.
                                  const auto inside = envelope.size() < 2
                                      || (localSeconds >= envelope.front().timeSeconds
                                          && localSeconds <= envelope.back().timeSeconds);
                                  return gain.at(sampleAt(localSeconds), !inside);
                              });
        }
        mixNote(result.buffer, rendered.audio, gain);
    }
    if (request.progress) request.progress(0.98);
    auto peak = 0.0f;
    for (int channel = 0; channel < result.buffer.getNumChannels(); ++channel)
        peak = std::max(peak, result.buffer.getMagnitude(channel, 0,
                                                         result.buffer.getNumSamples()));
    if (peak > 0.98f) result.buffer.applyGain(0.98f / peak);
    if (externalCount > 0 && nativeCount == 0)
        result.backend = "utau-resampler+internal-wavtool";
    else if (externalCount > 0)
        result.backend = "utau-resampler+native-fallback+internal-wavtool";
    else if (request.resamplerExecutable.existsAsFile())
        result.backend = "utau-native-fallback+internal-wavtool";
    else
        result.backend = "utau-native-resampler+internal-wavtool";
    if (pianoCount > 0)
        result.backend += "+piano-preview";
    if (nativeCount > 0 && request.resamplerExecutable.existsAsFile())
    {
        result.warning = juce::String(nativeCount) + " note(s) used native resampler fallback";
        if (externalFailure.isNotEmpty()) result.warning += ": " + externalFailure;
    }
    if (missingCount > 0)
    {
        if (result.warning.isNotEmpty()) result.warning += "; ";
        result.warning += juce::String(missingCount)
            + " alias(es) were not found, played as piano";
    }
    if (request.requireExternalResampler && nativeCount > 0) {
        result.buffer.setSize(0, 0);
        result.warning = "Selected resampler failed; no fallback used: " + externalFailure;
        return result;
    }
    if (request.wavtoolExecutable != juce::File{} && result.buffer.getNumSamples() > 0) {
        juce::TemporaryFile input(".wav"), output(".wav");
        const auto whd = juce::File(output.getFile().getFullPathName() + ".whd");
        const auto dat = juce::File(output.getFile().getFullPathName() + ".dat");
        struct Sidecars { juce::File whd, dat; ~Sidecars() { whd.deleteFile(); dat.deleteFile(); } } cleanup { whd, dat };
        const auto fail = [&](const juce::String& error) {
            result.warning = "wavtool: " + error;
            result.buffer.setSize(0, 0);
        };
        {
            // The original wavtool assumes the canonical 44-byte, mono PCM16 header.
            // JUCE writes a JUNK/RF64 reservation chunk which that tool mistakes for fmt.
            auto stream = input.getFile().createOutputStream();
            if (!stream) { fail("could not write assembly input"); return result; }
            const auto samples = result.buffer.getNumSamples();
            const auto rate = static_cast<int>(std::lround(result.sampleRate));
            stream->write("RIFF", 4); stream->writeInt(36 + samples * 2); stream->write("WAVEfmt ", 8);
            stream->writeInt(16); stream->writeShort(1); stream->writeShort(1);
            stream->writeInt(rate); stream->writeInt(rate * 2); stream->writeShort(2); stream->writeShort(16);
            stream->write("data", 4); stream->writeInt(samples * 2);
            for (int i = 0; i < samples; ++i) {
                float value = 0;
                for (int c = 0; c < result.buffer.getNumChannels(); ++c) value += result.buffer.getSample(c, i);
                value /= static_cast<float>(result.buffer.getNumChannels());
                stream->writeShort(static_cast<short>(std::lround(juce::jlimit(-1.0f, 1.0f, value) * 32767.0f)));
            }
            stream->flush();
            if (stream->getStatus().failed()) { fail("could not write assembly input"); return result; }
        }
        const auto seconds = result.buffer.getNumSamples() / result.sampleRate;
        const auto bpm = std::max(1.0, request.bpm);
        juce::StringArray args { request.wavtoolExecutable.getFullPathName(), output.getFile().getFullPathName(),
            input.getFile().getFullPathName(), "0", juce::String(seconds * bpm * 8.0, 8) + "@" + juce::String(bpm, 8),
            "0", "0", "0", "100", "100", "100", "100", "0", "0", "0", "100" };
        juce::ChildProcess process;
        if (!process.start(args, juce::ChildProcess::wantStdOut | juce::ChildProcess::wantStdErr)) {
            fail("could not start selected tool"); return result;
        }
        const auto deadline = juce::Time::getMillisecondCounterHiRes() + 120000.0;
        while (process.isRunning()) {
            if ((request.cancelled && request.cancelled()) || juce::Time::getMillisecondCounterHiRes() >= deadline) {
                process.kill(); fail("cancelled or timed out"); return result;
            }
            juce::Thread::sleep(20);
        }
        const auto diagnostics = process.readAllProcessOutput().trim();
        if (process.getExitCode() != 0) { fail("exit " + juce::String(process.getExitCode()) + " " + diagnostics); return result; }
        // UTAU wavtool produces header/data sidecars; other tools may write WAV directly.
        if (!output.getFile().existsAsFile() && whd.existsAsFile() && dat.existsAsFile()) {
            auto combined = output.getFile().createOutputStream();
            auto header = whd.createInputStream(); auto data = dat.createInputStream();
            if (combined && header && data) {
                combined->writeFromInputStream(*header, -1); combined->writeFromInputStream(*data, -1);
            }
        }
        juce::AudioBuffer<float> assembled;
        double rate = 0;
        if (!readAudio(output.getFile(), assembled, rate)) { fail("tool did not produce a readable WAV: " + diagnostics); return result; }
        result.buffer = std::move(assembled);
        result.sampleRate = rate;
        result.backend = result.backend.replace("internal-wavtool", "external-wavtool:" + request.wavtoolExecutable.getFileName());
    }
    if (request.progress) request.progress(1.0);
    return result;
}
}
