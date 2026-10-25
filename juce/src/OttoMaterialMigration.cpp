#include "OttoMaterialMigration.h"
#include "OttoMaterialIO.h"
#include "backend/LegacyTextCodec.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>

namespace hachi
{
namespace
{
using juce::File;
using juce::String;
using juce::StringArray;

String pathKey(const File& file)
{
    auto path = file.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    path = path.toLowerCase();
#endif
    return path;
}

bool fail(String& error, const String& message, const File& file)
{
    error = message + ": " + file.getFullPathName();
    return false;
}

bool durationOf(juce::AudioFormatManager& formats, const File& audio,
                double& duration, String& error)
{
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(audio));
    if (!reader || !std::isfinite(reader->sampleRate) || reader->sampleRate <= 0
        || reader->lengthInSamples <= 0 || reader->numChannels == 0)
        return fail(error, "Unreadable or empty audio", audio);
    duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
    return std::isfinite(duration) && duration > 0
        ? true : fail(error, "Invalid audio duration", audio);
}

bool durationRanges(const std::vector<SampleRegionSetting>& rows, double duration,
                    const File& audio, String& error)
{
    for (const auto& row : rows)
        if (!std::isfinite(row.regionStartSeconds) || !std::isfinite(row.regionEndSeconds)
            || row.regionStartSeconds < 0 || row.regionEndSeconds <= row.regionStartSeconds
            || row.regionEndSeconds > duration)
            return fail(error, "Pronunciation range is outside the recording (" + row.name + ")", audio);
    return true; // Remaining model invariants are checked by the native codec.
}

bool strictNumber(const String& text, double* value = nullptr)
{
    const auto trimmed = text.trim();
    if (trimmed.isEmpty()) return false;
    const auto* begin = trimmed.toRawUTF8();
    char* end = nullptr;
    const auto number = std::strtod(begin, &end);
    if (end == begin || *end != 0 || !std::isfinite(number)) return false;
    if (value) *value = number;
    return true;
}

// Match the legacy loader's CSV escaping, but reject unterminated quoting.
bool csvFields(const String& line, StringArray& fields)
{
    String field;
    bool quoted = false;
    for (int i = 0; i < line.length(); ++i)
    {
        const auto ch = line[i];
        if (ch == '"')
        {
            if (quoted && i + 1 < line.length() && line[i + 1] == '"')
            { field += '"'; ++i; }
            else quoted = !quoted;
        }
        else if (ch == ',' && !quoted) { fields.add(field); field.clear(); }
        else field += ch;
    }
    fields.add(field);
    return !quoted;
}

// loadOrDerive is intentionally permissive for interactive legacy editing.
// Migration may not let it skip malformed rows, substitute derived rows, or
// silently clamp envelope gain. Validate the on-disk legacy representation first.
bool preflightLegacy(const File& sidecar, int& count, String& error)
{
    auto source = backend::LegacyTextCodec::read(sidecar, error, 65001, false);
    if (!source) return false;
    count = 0;
    for (const auto& raw : StringArray::fromLines(source->text))
    {
        const auto line = raw.trim();
        if (line.isEmpty() || line.startsWithIgnoreCase("name,")) continue;
        StringArray fields;
        if (!csvFields(line, fields) || fields.size() < 6 || fields.size() > 25)
            return fail(error, "Unsupported or malformed legacy CSV row", sidecar);
        for (int i = 1; i < fields.size(); ++i)
            if (i != 19 && i != 20 && i != 22 && i != 23
                && fields[i].trim().isNotEmpty() && !strictNumber(fields[i]))
                return fail(error, "Invalid legacy numeric field", sidecar);
        double start = 0, end = 0;
        if (!strictNumber(fields[1], &start) || !strictNumber(fields[2], &end)
            || start < 0 || end <= start)
            return fail(error, "Invalid legacy source range", sidecar);
        for (const int column : {22, 23})
        {
            if (column >= fields.size() || fields[column].isEmpty()) continue;
            for (const auto& encoded : StringArray::fromTokens(fields[column], ";", ""))
            {
                const auto parts = StringArray::fromTokens(encoded, "|", "");
                if ((column == 22 && parts.size() != 11)
                    || (column == 23 && (parts.size() < 2 || parts.size() > 3)))
                    return fail(error, "Unsupported legacy segment/envelope record", sidecar);
                for (int i = 0; i < parts.size(); ++i)
                    if ((column == 23 || (i >= 3 && i != 5)) && !strictNumber(parts[i]))
                        return fail(error, "Invalid legacy segment/envelope number", sidecar);
                if (column == 23)
                {
                    const auto gain = parts[1].getDoubleValue();
                    if (gain < -60 || gain > 12)
                        return fail(error, "Legacy envelope would be clamped; migration refused", sidecar);
                }
            }
        }
        ++count;
    }
    if (count == 0) return fail(error, "Empty legacy annotation cannot be safely derived during migration", sidecar);
    return true;
}

bool rowFromOto(const VoicebankOtoEntry& entry, UtauMode mode, double duration,
                const OttoMaterialMigration::OtoSource& source,
                SampleRegionSetting& row, String& error)
{
    // Do not let the interactive OTO parser's permissive getDoubleValue turn
    // malformed authored text into apparently valid zero timing.
    if (entry.sourceFingerprint.isNotEmpty() && source.fingerprint != entry.sourceFingerprint)
        return fail(error, "OTO source changed during conversion", entry.otoFile);
    const auto& lines = source.lines;
    if (!juce::isPositiveAndBelow(entry.lineIndex, lines.size()))
        return fail(error, "OTO source row no longer exists", entry.otoFile);
    const auto equals = lines[entry.lineIndex].indexOfChar('=');
    StringArray fields;
    if (equals <= 0 || !csvFields(lines[entry.lineIndex].substring(equals + 1), fields)
        || fields.size() < 6)
        return fail(error, "Malformed OTO source row", entry.otoFile);
    for (int i = 1; i < 6; ++i)
        if (!strictNumber(fields[i])) return fail(error, "Invalid OTO numeric text", entry.otoFile);
    const double values[]{entry.offsetMs, entry.consonantMs, entry.cutoffMs,
                          entry.preutteranceMs, entry.overlapMs};
    for (const auto value : values)
        if (!std::isfinite(value)) return fail(error, "Nonfinite OTO timing", entry.audioFile);
    row.name = SampleSettings::entryLookupName(entry);
    row.regionStartSeconds = entry.offsetMs / 1000.0;
    row.regionEndSeconds = entry.cutoffMs < 0
        ? row.regionStartSeconds - entry.cutoffMs / 1000.0 : duration - entry.cutoffMs / 1000.0;
    row.fixedDurationSeconds = entry.consonantMs / 1000.0;
    row.alignmentSeconds = row.regionStartSeconds + entry.preutteranceMs / 1000.0;
    row.overlapSeconds = entry.overlapMs / 1000.0;
    row.provenance = mode == UtauMode::classic ? "oto" : mode == UtauMode::jie ? "oto4" : "otomou";
    row.confidence = 1;
    if (row.regionStartSeconds < 0 || row.regionEndSeconds > duration
        || row.regionEndSeconds <= row.regionStartSeconds
        || row.fixedDurationSeconds < 0
        || row.fixedDurationSeconds > row.regionEndSeconds - row.regionStartSeconds
        || row.alignmentSeconds < row.regionStartSeconds || row.alignmentSeconds > row.regionEndSeconds)
        return fail(error, "Invalid OTO timing/range (no clamping performed)", entry.audioFile);
    if (mode != UtauMode::classic && !entry.hasJieOto)
        return fail(error, "Selected OTO mode has no explicit region boundaries", entry.audioFile);
    const auto count = mode == UtauMode::classic ? 2 : mode == UtauMode::jie ? 4 : entry.mouClasses.length();
    if (count < 2 || count > 4) return fail(error, "Invalid Mou region count", entry.audioFile);
    const double boundaries[]{entry.jieOnsetMs, entry.jieGlideMs, entry.jieNucleusMs};
    double previous = row.regionStartSeconds;
    for (int i = 0; i < count; ++i)
    {
        const auto end = i == count - 1 ? row.regionEndSeconds : row.regionStartSeconds
            + (mode == UtauMode::classic ? entry.consonantMs : boundaries[i]) / 1000.0;
        if (!std::isfinite(end) || end < previous || end > row.regionEndSeconds)
            return fail(error, "Invalid ordered OTO boundary", entry.audioFile);
        NativeSegmentRole role;
        if (mode == UtauMode::mou)
        {
            const auto ch = entry.mouClasses[i];
            if (ch != 'C' && ch != 'V' && ch != 'S')
                return fail(error, "Unknown Mou region class", entry.audioFile);
            role = ch == 'C' ? NativeSegmentRole::consonant
                : ch == 'S' ? NativeSegmentRole::silence : NativeSegmentRole::vowel;
        }
        else if (mode == UtauMode::jie)
        {
            const NativeSegmentRole roles[]{NativeSegmentRole::consonant, NativeSegmentRole::transition,
                                            NativeSegmentRole::vowel, NativeSegmentRole::ending};
            role = roles[i];
        }
        else role = i == 0 ? NativeSegmentRole::consonant : NativeSegmentRole::vowel;
        // Zero-length authored partitions carry no audio; never expand them.
        if (end > previous)
            row.segments.push_back({"segment_" + String(i + 1), row.name, role, previous, end,
                row.provenance, 1, row.alignmentSeconds, row.overlapSeconds,
                role != NativeSegmentRole::consonant, 1});
        previous = end;
    }
    if (!row.segments.empty()) row.role = row.segments.front().role;
    return true; // No inferred links, even for adjacent UTAU aliases.
}

bool hasOtoForAudio(const File& audio, const File& root)
{
    for (auto folder = audio.getParentDirectory();; folder = folder.getParentDirectory())
    {
        for (const auto* name : {"oto.ini", "oto.jie.ini", "oto4.ini", "otomou.ini"})
            if (folder.getChildFile(name).exists()) return true;
        if (folder == root || folder.getParentDirectory() == folder) return false;
    }
}

bool readOtoRows(const File& audio, const File& root, UtauMode mode, double duration,
                 std::vector<SampleRegionSetting>& rows, StringArray& warnings, String& error)
{
    const auto entries = SampleSettings::loadAudioOto(audio, root, mode, warnings);
    if (!warnings.isEmpty())
        return fail(error, "OTO source reported errors: " + warnings.joinIntoString("; "), audio);
    std::map<String, OttoMaterialMigration::OtoSource> sources;
    for (const auto& entry : entries)
    {
        auto [it, inserted] = sources.try_emplace(pathKey(entry.otoFile));
        if (inserted && !OttoMaterialMigration::readOtoSource(entry, mode, it->second, error)) return false;
        SampleRegionSetting row;
        if (!OttoMaterialMigration::convertOtoEntry(entry, mode, duration, it->second, row, error)) return false;
        rows.push_back(std::move(row));
    }
    return true;
}

void countRows(OttoMaterialMigration::Report& report, const std::vector<SampleRegionSetting>& rows)
{
    ++report.audioFiles;
    report.pronunciations += static_cast<int>(rows.size());
    for (const auto& row : rows) report.aliases += 1 + row.additionalAliases.size();
}

struct TextLine { String text, newline; };
std::vector<TextLine> exactLines(const String& text)
{
    std::vector<TextLine> result;
    int start = 0;
    for (int i = 0; i < text.length(); ++i)
    {
        if (text[i] != '\n' && text[i] != '\r') continue;
        const auto end = i;
        String newline = String::charToString(text[i]);
        if (text[i] == '\r' && i + 1 < text.length() && text[i + 1] == '\n') { newline += '\n'; ++i; }
        result.push_back({text.substring(start, end), newline});
        start = i + 1;
    }
    if (start < text.length()) result.push_back({text.substring(start), {}});
    return result;
}

bool unsafeOtoField(const String& value)
{
    return value.isEmpty() || value != value.trim() || value.containsAnyOf("=,\r\n\"") ;
}
String milliseconds(double seconds) { return String(seconds * 1000.0, 9); }
}

bool OttoMaterialMigration::readOtoSource(const VoicebankOtoEntry& entry, UtauMode mode,
    OtoSource& result, String& error)
{
    auto sourceFile = entry.otoFile;
    if (mode == UtauMode::jie && !sourceFile.existsAsFile())
        sourceFile = entry.otoFile.getSiblingFile("oto.ini");
    const auto source = backend::LegacyTextCodec::read(sourceFile, error);
    if (!source) return false;
    if (entry.sourceFingerprint.isNotEmpty() && source->fingerprint != entry.sourceFingerprint)
        return fail(error, "OTO source changed during conversion", sourceFile);
    result = {StringArray::fromLines(source->text), source->fingerprint};
    return true;
}

bool OttoMaterialMigration::convertOtoEntry(const VoicebankOtoEntry& entry, UtauMode mode,
    double duration, const OtoSource& source, SampleRegionSetting& result, String& error)
{
    SampleRegionSetting row;
    if (!rowFromOto(entry, mode, duration, source, row, error)) return false;
    // Keep the historical classic/Jie/Mou IDs byte-for-byte unchanged.
    const auto digest = [](const String& text)
    { return juce::SHA256(text.toRawUTF8(), text.getNumBytesAsUTF8()).toHexString(); };
    row.materialAudioId = "a" + digest(pathKey(entry.audioFile));
    row.materialRegionId = "p" + digest(pathKey(entry.audioFile) + "|" + pathKey(entry.otoFile)
        + "|" + String(entry.lineIndex) + "|" + entry.alias + "|" + String(entry.offsetMs, 12)
        + "|" + entry.sourceFingerprint);
    result = std::move(row);
    return true;
}

bool OttoMaterialMigration::readAudioAnnotations(const File& audio, const File& root,
    UtauMode mode, std::vector<SampleRegionSetting>& rows, String& error, const String& audioId)
{
    error.clear();
    if (!root.isDirectory() || !audio.existsAsFile() || !audio.isAChildOf(root))
        return fail(error, "Recording must be a file inside the material root", audio);
    if (mode != UtauMode::classic && mode != UtauMode::jie && mode != UtauMode::mou)
        return fail(error, "Unsupported OTO mode", root);
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    double duration = 0;
    if (!durationOf(formats, audio, duration, error)) return false;
    std::vector<SampleRegionSetting> candidate;
    const auto manifest = root.getChildFile("otto.ini");
    if (manifest.exists())
    {
        OttoMaterialIO::Document document;
        if (!OttoMaterialIO::loadDocument(manifest, document, error)) return false;
        const OttoMaterialIO::Audio* selected = nullptr;
        for (const auto& recording : document.audio)
        {
            const bool matchesPath = pathKey(root.getChildFile(recording.relativePath)) == pathKey(audio);
            if (audioId.isNotEmpty() && recording.id == audioId && !matchesPath)
                return fail(error, "Audio identity belongs to another recording: " + audioId, audio);
            if (!matchesPath || (audioId.isNotEmpty() && recording.id != audioId)) continue;
            if (selected) return fail(error, "Ambiguous audio annotation variants; select audioId", audio);
            selected = &recording;
        }
        if (selected) candidate = selected->rows;
        else if (audioId.isNotEmpty()) return fail(error, "Audio annotation identity not found: " + audioId, audio);
        // A missing section is intentionally empty, never permission to use OTO.
    }
    else
    {
        StringArray warnings;
        if (!readOtoRows(audio, root, mode, duration, candidate, warnings, error)) return false;
    }
    if (audioId.isNotEmpty())
    {
        if (candidate.empty() && !manifest.exists())
            return fail(error, "Audio annotation identity not found: " + audioId, audio);
        for (const auto& row : candidate)
            if (row.materialAudioId != audioId)
                return fail(error, "Audio annotation identity not found: " + audioId, audio);
    }
    if (!durationRanges(candidate, duration, audio, error)) return false;
    rows = std::move(candidate);
    return true;
}

bool OttoMaterialMigration::ensureNativeFolder(const File& root, String& error)
{
    error.clear();
    if (!root.isDirectory()) return fail(error, "Material root is not a directory", root);
    const auto manifest = root.getChildFile("otto.ini");
    OttoMaterialIO::Document document;
    if (manifest.exists()) return OttoMaterialIO::loadDocument(manifest, document, error);
    // Filename-only early-exit discovery: never inventory/decode audio or seed
    // fabricated aliases while registering an existing material folder.
    for (const auto& entry : juce::RangedDirectoryIterator(root, true,
             "oto.ini;oto.jie.ini;oto4.ini;otomou.ini", File::findFiles))
    {
        juce::ignoreUnused(entry);
        return true;
    }
    return OttoMaterialIO::saveDocument(manifest, document, error);
}

bool OttoMaterialMigration::migrateFolder(const File& root, UtauMode mode,
                                         bool includeUnannotated, Report& report, String& error)
{
    report = {}; error.clear();
    if (!root.isDirectory()) return fail(error, "Material root is not a directory", root);
    if (mode != UtauMode::classic && mode != UtauMode::jie && mode != UtauMode::mou)
        return fail(error, "Unsupported OTO mode", root);
    const auto manifest = root.getChildFile("otto.ini");
    OttoMaterialIO::Document document;
    if (manifest.exists() && !OttoMaterialIO::loadDocument(manifest, document, error)) return false;
    std::set<String> existing;
    for (const auto& audio : document.audio) existing.insert(pathKey(root.getChildFile(audio.relativePath)));
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    auto files = root.findChildFiles(File::findFiles, true, formats.getWildcardForAllFormats());
    std::sort(files.begin(), files.end(), [](const File& a, const File& b)
    { return a.getFullPathName().compare(b.getFullPathName()) < 0; });
    Report pending;
    for (const auto& audio : files)
    {
        if (existing.count(pathKey(audio)) != 0) continue; // Preserve existing sections verbatim in memory.
        const auto nearest = OttoMaterialIO::findManifest(audio);
        if (nearest != File{} && pathKey(nearest) != pathKey(manifest) && nearest.isAChildOf(root))
            return fail(error, "Nested native manifest requires an explicit separate migration", nearest);
        double duration = 0;
        if (!durationOf(formats, audio, duration, error)) return false;
        std::vector<SampleRegionSetting> rows;
        auto legacy = SampleSettings::sidecarFor(audio);
        if (!legacy.exists()) legacy = File(audio.getFullPathName() + ".hachi.csv");
        // Explicit migration alone accepts CSV, and only when no OTO source
        // governs the recording. Old sidecars must never override authored OTO.
        if (legacy.exists() && !hasOtoForAudio(audio, root))
        {
            const auto governing = SampleSettings::annotationFor(audio);
            if (pathKey(governing) != pathKey(manifest) && governing.exists())
            {
                // An unrelated ancestor manifest is not authority over every
                // descendant audio. Refuse only actual competing annotations
                // (or a malformed manifest), not an absent audio section.
                const auto competing = OttoMaterialIO::containsAudio(governing, audio, error);
                if (error.isNotEmpty()) return false;
                if (competing)
                    return fail(error, "Migration cannot consolidate legacy data through a different native manifest; resolve manifest ambiguity first", governing);
            }
            int expected = 0;
            if (!preflightLegacy(legacy, expected, error)) return false;
            rows = SampleSettings::loadOrDerive(audio, ProjectData{}, &error, true);
            if (error.isNotEmpty()) return false;
            if (static_cast<int>(rows.size()) != expected)
                return fail(error, "Legacy loader omitted or substituted annotations", legacy);
        }
        else
        {
            StringArray warnings;
            const auto read = readOtoRows(audio, root, mode, duration, rows, warnings, error);
            report.warnings.addArray(warnings);
            if (!read) return false;
            if (rows.empty() && includeUnannotated)
            {
                SampleRegionSetting row;
                row.name = audio.getFileNameWithoutExtension();
                row.regionStartSeconds = row.alignmentSeconds = 0;
                row.regionEndSeconds = duration;
                rows.push_back(std::move(row));
            }
        }
        if (rows.empty()) continue;
        if (!durationRanges(rows, duration, audio, error)
            || !OttoMaterialIO::replaceAudio(document, manifest, audio, rows, error)) return false;
        countRows(pending, rows);
        existing.insert(pathKey(audio));
    }
    if (pending.audioFiles == 0) return true;
    if (!OttoMaterialIO::saveDocument(manifest, document, error)) return false;
    pending.warnings = report.warnings; report = std::move(pending);
    return true;
}

bool OttoMaterialMigration::exportFolder(const File& root, const File& target,
                                        bool replaceExisting, Report& report, String& error)
{
    report = {}; error.clear();
    if (!root.isDirectory()) return fail(error, "Material root is not a directory", root);
    const auto manifest = root.getChildFile("otto.ini");
    if (pathKey(manifest) == pathKey(target)) return fail(error, "Cannot overwrite native manifest", target);
    if (!target.getParentDirectory().isDirectory() || (target.exists() && !target.existsAsFile()))
        return fail(error, "OTO destination must be a file in an existing directory", target);
    OttoMaterialIO::Document document;
    if (!OttoMaterialIO::loadDocument(manifest, document, error)) return false;
    backend::LegacyTextDocument original;
    if (target.existsAsFile())
    {
        // Full replacement still fingerprints the original bytes, but need not decode them.
        original.existed = true;
        if (replaceExisting)
        {
            if (!target.loadFileAsData(original.bytes)) return fail(error, "Cannot read target", target);
        }
        else
        {
            const auto loaded = backend::LegacyTextCodec::read(target, error);
            if (!loaded) return false;
            original = *loaded;
        }
    }
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    std::map<String, StringArray> byRecording;
    // filename + alias conflicts are rejected even when their lossy timing happens
    // to match: two native pronunciation identities cannot become one OTO identity.
    std::set<std::pair<String, String>> identities;
    Report pending;
    for (const auto& source : document.audio)
    {
        const auto audio = root.getChildFile(source.relativePath);
        const auto key = pathKey(audio);
        if (key == pathKey(target)) return fail(error, "Cannot overwrite annotated source audio", target);
        double duration = 0;
        if (!durationOf(formats, audio, duration, error)
            || !durationRanges(source.rows, duration, audio, error)) return false;
        if (source.rows.empty()) continue;
        if (!audio.hasFileExtension("wav")) return fail(error, "Standard OTO export requires WAV recordings (no transcoding)", audio);
        const auto relative = audio.getRelativePathFrom(target.getParentDirectory()).replaceCharacter('\\', '/');
        if (File::isAbsolutePath(relative) || relative.containsChar(':') || unsafeOtoField(relative)
            || relative.startsWithChar(';') || relative.startsWithChar('#'))
            return fail(error, "Recording cannot be represented by a relative OTO filename", audio);
        auto& lines = byRecording[key];
        for (const auto& row : source.rows)
        {
            if (row.connectedNextRegionId.isNotEmpty())
                report.warnings.addIfNotAlreadyThere("Standard OTO output discards explicit pronunciation connections.");
            int vowels = 0;
            for (const auto& segment : row.segments) if (segment.role == NativeSegmentRole::vowel) ++vowels;
            if (vowels > 1)
                report.warnings.addIfNotAlreadyThere("Standard OTO output discards multiple-vowel segmentation.");
            if (row.pronunciationKind == PronunciationKind::transition || std::any_of(row.segments.begin(), row.segments.end(),
                [](const NativeSegment& s) { return s.role == NativeSegmentRole::transition; }))
                report.warnings.addIfNotAlreadyThere("Standard OTO output discards native transition information.");
            if (row.pronunciationKind == PronunciationKind::breath)
                report.warnings.addIfNotAlreadyThere("Standard OTO output cannot retain the explicit breath-pronunciation identity.");
            if (row.simplifiedConsonant)
                report.warnings.addIfNotAlreadyThere("Standard OTO output retains current C/P values but not their simplified-consonant linkage.");
            if (!row.segments.empty())
                report.warnings.addIfNotAlreadyThere("Standard OTO output discards native segment boundaries, roles and stretch policy.");
            if (row.melodyneData || row.relativePitchCents != 0 || !row.amplitudeEnvelope.empty() || row.nativeUnpitched)
                report.warnings.addIfNotAlreadyThere("Standard OTO output discards native expressive pitch, envelope and unpitched metadata.");
            auto aliases = row.additionalAliases; aliases.insert(0, row.name);
            for (const auto& alias : aliases)
            {
                if (unsafeOtoField(alias)) return fail(error, "Alias cannot be represented losslessly in standard OTO", audio);
                if (!identities.insert({key, alias}).second)
                    return fail(error, "Conflicting native pronunciations share the same recording and alias: " + alias, audio);
                lines.add(relative + "=" + alias + "," + milliseconds(row.regionStartSeconds)
                    + "," + milliseconds(row.fixedDurationSeconds) + "," + milliseconds(duration - row.regionEndSeconds)
                    + "," + milliseconds(row.alignmentSeconds - row.regionStartSeconds)
                    + "," + milliseconds(row.overlapSeconds));
            }
        }
        countRows(pending, source.rows);
    }
    String text;
    if (replaceExisting || !original.existed)
    {
        original.codePage = 65001; original.bom = false;
        text = "#Charset:UTF-8\n";
        for (const auto& recording : byRecording)
            for (const auto& line : recording.second) text += line + "\n";
    }
    else
    {
        const auto lines = exactLines(original.text);
        String newline = "\n";
        for (const auto& line : lines) if (line.newline.isNotEmpty()) { newline = line.newline; break; }
        std::set<String> emitted;
        for (const auto& line : lines)
        {
            const auto trimmed = line.text.trimStart();
            const auto eq = line.text.indexOfChar('=');
            String key;
            if (!trimmed.startsWithChar(';') && !trimmed.startsWithChar('#') && eq > 0)
            {
                const auto filename = line.text.substring(0, eq).trim().replaceCharacter('\\', '/');
                if (!File::isAbsolutePath(filename) && !filename.containsChar(':'))
                    key = pathKey(target.getParentDirectory().getChildFile(filename));
            }
            const auto found = byRecording.find(key);
            if (found == byRecording.end()) text += line.text + line.newline;
            else if (emitted.insert(key).second)
                for (const auto& generated : found->second) text += generated + newline;
        }
        for (const auto& recording : byRecording)
            if (emitted.insert(recording.first).second)
            {
                if (text.isNotEmpty() && !text.endsWithChar('\n') && !text.endsWithChar('\r')) text += newline;
                for (const auto& generated : recording.second) text += generated + newline;
            }
    }
    // Avoid publication from an obsolete native snapshot as well as from a stale
    // destination. LegacyTextCodec rechecks target bytes before/after temp write.
    juce::MemoryBlock current;
    if (!manifest.loadFileAsData(current) || current != document.fingerprint)
        return fail(error, "Native manifest changed during export; reload and retry", manifest);
    if (!backend::LegacyTextCodec::write(target, text, original, error)) return false;
    pending.warnings = report.warnings; report = std::move(pending);
    return true;
}
}
