#include "CommunityDatasetIO.h"
#include "PronunciationTiming.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>

namespace hachi
{
namespace
{
using IO = CommunityDatasetIO;
using juce::File;
using juce::String;
using juce::StringArray;
bool fail(IO::Report& report, const String& error) { report.errors.add(error); return false; }
bool stop(IO::Report& report, const IO::Cancel& cancel)
{ if (cancel && cancel()) { report.cancelled = true; return true; } return false; }
String pathKey(const File& file)
{
    auto value = file.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    value = value.toLowerCase();
#endif
    return value;
}
String digest(const String& value)
{ return juce::SHA256(value.toRawUTF8(), value.getNumBytesAsUTF8()).toHexString(); }
StringArray words(const String& text)
{ auto result = StringArray::fromTokens(text, " \t\r\n", ""); result.removeEmptyStrings(); return result; }
bool special(const String& token) { return token == "SP" || token == "AP"; }
bool number(const String& text, double& value)
{
    const auto trimmed = text.trim();
    if (trimmed.isEmpty()) return false;
    auto* begin = trimmed.toRawUTF8(); char* end = nullptr;
    value = std::strtod(begin, &end);
    return end != begin && *end == 0 && std::isfinite(value) && value >= 0;
}
bool relativeName(const String& value)
{
    if (value.isEmpty() || value != value.trim() || value.containsAnyOf(":\\\r\n")
        || value.startsWithChar('/') || value.endsWithChar('/') || value.contains("//") || File::isAbsolutePath(value)) return false;
    const auto parts = StringArray::fromTokens(value, "/", "");
    for (const auto& part : parts) if (part.isEmpty() || part == "." || part == "..") return false;
    return true;
}
bool utf8(const File& file, String& text, String& fingerprint, IO::Report& report)
{
    if (!file.existsAsFile() || file.getSize() > 64 * 1024 * 1024)
        return fail(report, "Missing or oversized UTF-8 metadata: " + file.getFullPathName());
    juce::MemoryBlock bytes;
    if (!file.loadFileAsData(bytes)) return fail(report, "Cannot read metadata: " + file.getFullPathName());
    const auto* data = static_cast<const char*>(bytes.getData());
    const auto size = static_cast<int>(bytes.getSize());
    for (int i = 0; i < size; ++i) if (data[i] == 0) return fail(report, "NUL in metadata: " + file.getFullPathName());
    if (size > 0 && !juce::CharPointer_UTF8::isValidString(data, size)) return fail(report, "Invalid UTF-8 metadata: " + file.getFullPathName());
    fingerprint = juce::SHA256(bytes).toHexString();
    const int bom = size >= 3 && static_cast<unsigned char>(data[0]) == 0xef
        && static_cast<unsigned char>(data[1]) == 0xbb && static_cast<unsigned char>(data[2]) == 0xbf ? 3 : 0;
    text = size > bom ? String::fromUTF8(data + bom, size - bom) : String();
    return true;
}
// Strict comma RFC4180, including embedded newlines and doubled quotes. There
// is no reuse of classic OTO's permissive, line-based CSV parser here.
bool csv(const String& text, std::vector<StringArray>& records, IO::Report& report, const IO::Cancel& cancel)
{
    StringArray row; String field;
    bool quoted = false, closed = false, started = false;
    const auto flushField = [&]() { row.add(field); field.clear(); closed = started = false; };
    const auto flushRow = [&]()
    {
        flushField();
        if (!(row.size() == 1 && row[0].isEmpty())) records.push_back(row);
        row.clear();
    };
    for (int i = 0; i < text.length(); ++i)
    {
        if ((i & 4095) == 0 && stop(report, cancel)) return false;
        const auto ch = text[i];
        if (quoted)
        {
            if (ch == '"')
            {
                if (i + 1 < text.length() && text[i + 1] == '"') { field += '"'; ++i; }
                else { quoted = false; closed = true; }
            }
            else field += ch;
            continue;
        }
        if (ch == ',') { flushField(); continue; }
        if (ch == '\n' || ch == '\r')
        {
            flushRow();
            if (ch == '\r' && i + 1 < text.length() && text[i + 1] == '\n') ++i;
            continue;
        }
        if (closed) return fail(report, "Unexpected text after closing CSV quote");
        if (ch == '"')
        {
            if (started || field.isNotEmpty()) return fail(report, "CSV quote must begin a field");
            quoted = true; started = true;
        }
        else { field += ch; started = true; }
    }
    if (quoted) return fail(report, "Unterminated CSV quoted field");
    if (started || closed || !row.isEmpty()) flushRow();
    return true;
}
bool currentInputs(const IO::Dataset& dataset, const IO::Dictionary& dictionary,
                   IO::Report& report, const IO::Cancel& cancel)
{
    if (stop(report, cancel)) return false;
    if (!dataset.csv.existsAsFile() || juce::SHA256(dataset.csv).toHexString() != dataset.csvFingerprint)
        return fail(report, "Dataset CSV changed; reload before importing");
    if (dictionary.file != File{} && (!dictionary.file.existsAsFile()
        || juce::SHA256(dictionary.file).toHexString() != dictionary.fingerprint))
        return fail(report, "Dictionary changed; reload before importing");
    for (const auto& recording : dataset.recordings)
    {
        if (stop(report, cancel)) return false;
        if (!recording.audio.existsAsFile() || recording.audioFingerprint.isEmpty()
            || juce::SHA256(recording.audio).toHexString() != recording.audioFingerprint)
            return fail(report, "Dataset recording changed: " + recording.audio.getFullPathName());
    }
    return true;
}
struct Span
{
    int first = 0, count = 0, nucleus = 0;
    String alias; StringArray additional;
    bool silence = false, breath = false;
};
String sequence(const IO::Recording& recording, int first, int count)
{
    StringArray values;
    for (int i = 0; i < count; ++i) values.add(recording.phones[static_cast<std::size_t>(first + i)].symbol);
    return values.joinIntoString(" ");
}
}

bool IO::read(const File& root, Dataset& output, Report& report, Cancel cancel)
{
    report = {};
    if (!root.isDirectory()) return fail(report, "Dataset root is not a directory: " + root.getFullPathName());
    Dataset dataset; dataset.root = root; dataset.csv = root.getChildFile("transcriptions.csv");
    String text;
    if (!utf8(dataset.csv, text, dataset.csvFingerprint, report)) return false;
    std::vector<StringArray> records;
    if (!csv(text, records, report, cancel)) return false;
    if (records.empty()) return fail(report, "Empty transcriptions.csv");
    std::map<String, int> columns;
    for (int i = 0; i < records.front().size(); ++i)
        if (records.front()[i].isEmpty() || !columns.emplace(records.front()[i], i).second)
            return fail(report, "Empty or duplicate CSV header");
    for (const auto* required : {"name", "ph_seq", "ph_dur"})
        if (!columns.count(required)) return fail(report, "Missing community CSV column: " + String(required)
            + "; original DiffSinger text and DS-only fallback are not imported");
    std::set<String> names, audioPaths;
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    for (std::size_t r = 1; r < records.size(); ++r)
    {
        if (stop(report, cancel)) return false;
        const auto& fields = records[r];
        if (fields.size() != records.front().size()) return fail(report, "CSV column count mismatch at record " + String(static_cast<int>(r + 1)));
        const auto field = [&](const char* column)
        { const auto found = columns.find(column); return found == columns.end() ? String() : fields[found->second]; };
        Recording recording; recording.name = field("name");
        if (!relativeName(recording.name) || !names.insert(recording.name).second)
            return fail(report, "Unsafe or duplicate recording name: " + recording.name);
        const auto wav = root.getChildFile("wavs/" + recording.name + ".wav");
        const auto flac = root.getChildFile("wavs/" + recording.name + ".flac");
        recording.audio = wav.existsAsFile() ? wav : flac;
        if (!recording.audio.existsAsFile() || !recording.audio.isAChildOf(root)
            || !audioPaths.insert(pathKey(recording.audio)).second)
            return fail(report, "Missing or duplicate WAV/FLAC for recording: " + recording.name);
        if (wav.existsAsFile() && flac.existsAsFile()) report.warnings.add("WAV takes precedence over FLAC: " + recording.name);
        const auto phonemes = words(field("ph_seq")), durations = words(field("ph_dur"));
        if (phonemes.isEmpty() || phonemes.size() != durations.size()) return fail(report, "ph_seq/ph_dur length mismatch: " + recording.name);
        double position = 0;
        for (int i = 0; i < phonemes.size(); ++i)
        {
            double duration = 0;
            if (!number(durations[i], duration) || !std::isfinite(position + duration))
                return fail(report, "Invalid phoneme duration: " + recording.name);
            recording.phones.push_back({phonemes[i], position, position + duration}); position += duration;
        }
        if (position <= 0) return fail(report, "Zero-length recording annotation: " + recording.name);
        const auto counts = words(field("ph_num"));
        juce::int64 countTotal = 0;
        for (const auto& token : counts)
        {
            double count = 0;
            if (!number(token, count) || count < 1 || count > phonemes.size() || count != std::floor(count))
                return fail(report, "Invalid ph_num word count: " + recording.name);
            countTotal += static_cast<int>(count); recording.wordPhoneCounts.push_back(static_cast<int>(count));
        }
        if (!counts.isEmpty() && countTotal != phonemes.size()) return fail(report, "ph_num does not cover ph_seq: " + recording.name);
        const auto notes = words(field("note_seq")), noteDurations = words(field("note_dur")), glides = words(field("note_glide"));
        if (notes.size() != noteDurations.size() || (!glides.isEmpty() && glides.size() != notes.size()))
            return fail(report, "note_seq/note_dur/note_glide length mismatch: " + recording.name);
        double notePosition = 0;
        for (int i = 0; i < notes.size(); ++i)
        {
            double duration = 0;
            if (!number(noteDurations[i], duration) || !std::isfinite(notePosition + duration))
                return fail(report, "Invalid note duration: " + recording.name);
            recording.notes.push_back({notes[i], glides.isEmpty() ? String() : glides[i], notePosition, notePosition + duration});
            notePosition += duration;
        }
        if (!notes.isEmpty()) report.warnings.add("Score notes retained as dataset metadata, NOT target pitch or measured F0: " + recording.name);
        for (const auto& [name, index] : columns)
            if (name != "name" && name != "ph_seq" && name != "ph_dur" && name != "ph_num"
                && name != "note_seq" && name != "note_dur" && name != "note_glide")
                recording.extraFields.emplace(name, fields[index]);
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(recording.audio));
        if (!reader || !std::isfinite(reader->sampleRate) || reader->sampleRate <= 0 || reader->lengthInSamples <= 0 || reader->numChannels == 0)
            return fail(report, "Unreadable recording header: " + recording.audio.getFullPathName());
        recording.physicalDuration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        if (position > recording.physicalDuration + 1.e-9)
            return fail(report, "Phoneme timing exceeds physical audio end (no clamping): " + recording.name);
        if (recording.physicalDuration - position > 1.0 / reader->sampleRate)
            report.warnings.add("Unannotated trailing audio stays outside imported pronunciation ranges: " + recording.name);
        if (!notes.isEmpty() && std::abs(notePosition - position) > 1.0 / reader->sampleRate)
            report.warnings.add("Note and phoneme totals differ; score metadata is retained without retiming audio: " + recording.name);
        recording.audioFingerprint = juce::SHA256(recording.audio).toHexString();
        dataset.recordings.push_back(std::move(recording));
    }
    if (dataset.recordings.empty()) return fail(report, "Dataset has no recording rows");
    report.recordings = static_cast<int>(dataset.recordings.size());
    output = std::move(dataset);
    return true;
}

bool IO::readTwoPartDictionary(const File& file, Dictionary& output, Report& report, Cancel cancel)
{
    report = {}; Dictionary dictionary; dictionary.file = file;
    String text;
    if (!utf8(file, text, dictionary.fingerprint, report)) return false;
    std::set<String> aliases, vowels, consonants;
    vowels.insert("SP"); vowels.insert("AP");
    for (const auto& line : StringArray::fromLines(text))
    {
        if (stop(report, cancel)) return false;
        if (line.trim().isEmpty()) continue;
        const auto tab = line.indexOfChar('\t');
        if (tab <= 0 || line.indexOfChar(tab + 1, '\t') >= 0) return fail(report, "Dictionary requires alias<TAB>phones");
        Rule rule; rule.alias = line.substring(0, tab); rule.phones = words(line.substring(tab + 1));
        if (rule.alias != rule.alias.trim() || !aliases.insert(rule.alias).second || rule.phones.isEmpty() || rule.phones.size() > 2)
            return fail(report, "Not an unambiguous declared two-part dictionary: " + rule.alias);
        rule.nucleusOffset = rule.phones.size() - 1;
        vowels.insert(rule.phones[rule.nucleusOffset]);
        if (rule.nucleusOffset == 1) consonants.insert(rule.phones[0]);
        dictionary.rules.push_back(std::move(rule));
    }
    for (const auto& token : consonants) if (vowels.count(token)) return fail(report, "Conflicting declared dictionary C/V inventory: " + token);
    if (dictionary.rules.empty()) return fail(report, "Empty declared dictionary");
    output = std::move(dictionary);
    return true;
}

bool IO::planImport(const Dataset& dataset, const Dictionary& dictionary, const std::vector<Selection>& selections,
                    const Options& options, Plan& output, Report& report, Cancel cancel)
{
    report = {};
    if (!options.confirmNucleusAsAlignmentAndConsonant)
        return fail(report, "Explicit confirmation of P=C=selected nucleus onset is required; CSV has no author P/C markers");
    if (!currentInputs(dataset, dictionary, report, cancel)) return false;
    Plan plan; plan.dataset = dataset; plan.dictionary = dictionary; plan.manifest = dataset.root.getChildFile("otto.ini");
    if (plan.manifest.exists())
    {
        String error;
        if (!OttoMaterialIO::loadDocument(plan.manifest, plan.document, error)) return fail(report, error);
    }
    std::set<String> selectedNames;
    for (const auto& selected : selections) selectedNames.insert(selected.recordingName);
    std::set<String> recordingNames;
    for (const auto& recording : dataset.recordings) recordingNames.insert(recording.name);
    for (const auto& name : selectedNames) if (!recordingNames.count(name)) return fail(report, "Selection refers to unknown dataset recording: " + name);
    for (const auto& recording : dataset.recordings)
    {
        if (stop(report, cancel)) return false;
        std::vector<Span> spans;
        if (selectedNames.count(recording.name))
        {
            for (const auto& selected : selections) if (selected.recordingName == recording.name)
            {
                if (selected.firstPhone < 0 || selected.phoneCount <= 0 || selected.firstPhone > static_cast<int>(recording.phones.size()) - selected.phoneCount
                    || selected.nucleusOffset < 0 || selected.nucleusOffset >= selected.phoneCount || selected.alias.trim().isEmpty())
                    return fail(report, "Invalid explicit syllable selection: " + recording.name);
                spans.push_back({selected.firstPhone, selected.phoneCount, selected.nucleusOffset, selected.alias, selected.additionalAliases});
            }
        }
        else
        {
            if (dictionary.rules.empty()) return fail(report, "Explicit syllable spans or a declared dictionary required: " + recording.name);
            // Dynamic programming counts possible exact dictionary segmentations,
            // capped at two. Ambiguity is reported, never alphabetically guessed.
            const int size = static_cast<int>(recording.phones.size());
            std::vector<int> ways(static_cast<std::size_t>(size + 1), 0), next(static_cast<std::size_t>(size), -1);
            std::vector<std::vector<const Rule*>> candidates(static_cast<std::size_t>(size));
            ways.back() = 1;
            for (int i = size - 1; i >= 0; --i)
            {
                if (stop(report, cancel)) return false;
                if (special(recording.phones[static_cast<std::size_t>(i)].symbol))
                { ways[static_cast<std::size_t>(i)] = ways[static_cast<std::size_t>(i + 1)]; next[static_cast<std::size_t>(i)] = i + 1; continue; }
                std::map<String, std::vector<const Rule*>> matches;
                for (const auto& rule : dictionary.rules)
                {
                    const int count = rule.phones.size();
                    if (count <= 0 || i + count > size || rule.nucleusOffset < 0 || rule.nucleusOffset >= count) continue;
                    bool match = true;
                    for (int p = 0; p < count; ++p)
                        if (special(recording.phones[static_cast<std::size_t>(i + p)].symbol)
                            || rule.phones[p] != recording.phones[static_cast<std::size_t>(i + p)].symbol) { match = false; break; }
                    if (match) matches[rule.phones.joinIntoString(" ") + "|" + String(rule.nucleusOffset)].push_back(&rule);
                }
                for (const auto& [key, rules] : matches)
                {
                    juce::ignoreUnused(key);
                    const int end = i + rules.front()->phones.size();
                    if (!ways[static_cast<std::size_t>(end)]) continue;
                    ways[static_cast<std::size_t>(i)] = std::min(2, ways[static_cast<std::size_t>(i)] + ways[static_cast<std::size_t>(end)]);
                    next[static_cast<std::size_t>(i)] = end; candidates[static_cast<std::size_t>(i)] = rules;
                }
            }
            if (ways.front() != 1) return fail(report, "No unique syllable segmentation; explicit phone spans/aliases required: " + recording.name);
            for (int i = 0; i < size;)
            {
                if (special(recording.phones[static_cast<std::size_t>(i)].symbol)) { ++i; continue; }
                const auto& rules = candidates[static_cast<std::size_t>(i)];
                String alias = rules.front()->alias;
                if (rules.size() > 1)
                {
                    const auto chosen = options.primaryAliases.find(rules.front()->phones.joinIntoString(" "));
                    if (chosen == options.primaryAliases.end()) return fail(report, "Multiple dictionary aliases need an explicit primary choice: " + recording.name + " / " + rules.front()->phones.joinIntoString(" "));
                    alias = chosen->second;
                    if (std::none_of(rules.begin(), rules.end(), [&](const auto* rule) { return rule->alias == alias; }))
                        return fail(report, "Chosen alias is not declared by the dictionary: " + alias);
                }
                Span span; span.first = i; span.count = next[static_cast<std::size_t>(i)] - i;
                span.nucleus = rules.front()->nucleusOffset; span.alias = alias;
                for (const auto* rule : rules) if (rule->alias != alias) span.additional.add(rule->alias);
                spans.push_back(std::move(span)); i = next[static_cast<std::size_t>(i)];
            }
        }
        std::vector<bool> covered(recording.phones.size(), false);
        for (const auto& span : spans)
            for (int i = span.first; i < span.first + span.count; ++i)
            {
                if (covered[static_cast<std::size_t>(i)] || special(recording.phones[static_cast<std::size_t>(i)].symbol))
                    return fail(report, "Explicit speech spans overlap or include SP/AP; special-phone policy is separate: " + recording.name);
                covered[static_cast<std::size_t>(i)] = true;
            }
        for (std::size_t i = 0; i < recording.phones.size(); ++i)
        {
            const auto& phone = recording.phones[i];
            if (!special(phone.symbol))
            { if (!covered[i]) return fail(report, "Explicit syllable selections leave uncovered speech: " + recording.name); continue; }
            if (options.specials == Specials::reject) return fail(report, "Explicit SP/AP import/skip policy required: " + recording.name);
            if (options.specials == Specials::skipWithReport || phone.end == phone.start)
            { ++report.skippedSpecialPhones; continue; }
            Span span; span.first = static_cast<int>(i); span.count = 1; span.alias = phone.symbol;
            span.silence = phone.symbol == "SP"; span.breath = phone.symbol == "AP"; spans.push_back(std::move(span));
        }
        std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        PlannedAudio planned; planned.audio = recording.audio;
        const auto audioId = "a_dsc_" + digest(pathKey(dataset.root) + "|" + pathKey(recording.audio));
        planned.audioId = audioId;
        for (const auto& span : spans)
        {
            const auto& first = recording.phones[static_cast<std::size_t>(span.first)];
            const auto& last = recording.phones[static_cast<std::size_t>(span.first + span.count - 1)];
            const auto nucleus = recording.phones[static_cast<std::size_t>(span.first + span.nucleus)].start;
            if (last.end <= first.start) return fail(report, "Selected pronunciation has no positive source interval: " + recording.name);
            SampleRegionSetting row; row.name = span.alias; row.additionalAliases = span.additional;
            row.materialAudioId = audioId;
            row.materialRegionId = "p_dsc_" + digest(pathKey(dataset.root) + "|" + pathKey(recording.audio)
                + "|" + recording.name + "|" + String(span.first) + "|" + String(span.count) + "|"
                + sequence(recording, span.first, span.count) + "|" + String(first.start, 17) + "|" + String(last.end, 17));
            row.provenance = "openvpi-community"; row.confidence = 1;
            row.pronunciationKind = span.breath ? PronunciationKind::breath : PronunciationKind::normal;
            row.nativeUnpitched = span.silence;
            PronunciationTiming timing{first.start, last.end, nucleus, nucleus, 0, false};
            String error;
            if (!applyPronunciationTiming(row, timing, error)) return fail(report, error + ": " + recording.name);
            // Shared factory has already produced one C/one logical V; training
            // phone boundaries do NOT become explicit named authoring V splits.
            row.vowelSplitsExplicit = false;
            if (!row.segments.empty()) row.role = row.segments.front().role;
            planned.rows.push_back(std::move(row)); ++report.pronunciations;
        }
        const auto relative = recording.audio.getRelativePathFrom(dataset.root).replaceCharacter('\\', '/');
        bool targetExists = false, hasSibling = false;
        for (const auto& existing : plan.document.audio)
            if (pathKey(dataset.root.getChildFile(existing.relativePath)) == pathKey(recording.audio))
            {
                if (existing.id != audioId) { hasSibling = true; continue; }
                targetExists = true;
                if (!options.replacePreviouslyImportedAudio
                    || std::any_of(existing.rows.begin(), existing.rows.end(), [](const auto& row)
                    { return row.provenance != "openvpi-community" || !row.materialRegionId.startsWith("p_dsc_"); }))
                    return fail(report, "Existing target registration requires explicit replacement of dataset-owned rows only: " + relative);
                report.warnings.add("Explicitly replacing prior dataset annotations; changed phone spans/timing create new IDs: " + recording.name);
            }
        if (hasSibling && !targetExists && !options.allowSiblingAnnotationVariant)
            return fail(report, "Recording already has another annotation registration; explicitly allow a separate dataset variant: " + relative);
        if (hasSibling) report.warnings.add("Other same-recording annotation registrations retained unchanged: " + relative);
        String error;
        if (!OttoMaterialIO::replaceAudio(plan.document, plan.manifest, recording.audio, planned.rows, error, audioId)) return fail(report, error);
        plan.audio.push_back(std::move(planned)); ++report.recordings;
    }
    if (report.skippedSpecialPhones) report.warnings.add("Explicitly skipped SP/AP phones: " + String(report.skippedSpecialPhones));
    report.warnings.add("P=C uses explicitly confirmed nucleus onset. Dataset phoneme/note/word/extra facts remain in CSV/reader metadata, not target pitch; no manual vowel subdivisions inferred.");
    report.warnings.add("Publishing root OTTO makes it authoritative for the entire dataset root; old OTO files stay unchanged and are not fallback.");
    plan.ready = true; output = std::move(plan);
    return true;
}

bool IO::publish(Plan& plan, Report& report, Cancel cancel)
{
    report = {};
    if (!plan.ready || plan.published || plan.manifest != plan.dataset.root.getChildFile("otto.ini"))
        return fail(report, "Import plan is absent, already published or has a different governing manifest");
    if (!currentInputs(plan.dataset, plan.dictionary, report, cancel)) return false;
    if (stop(report, cancel)) return false;
    String error;
    // saveDocument retains and rechecks the original observed manifest token
    // under its existing process lock, then validates/rereads/atomically commits.
    if (!OttoMaterialIO::saveDocument(plan.manifest, plan.document, error)) return fail(report, error);
    plan.published = report.published = true;
    report.recordings = static_cast<int>(plan.audio.size());
    for (const auto& audio : plan.audio) report.pronunciations += static_cast<int>(audio.rows.size());
    return true;
}
}
