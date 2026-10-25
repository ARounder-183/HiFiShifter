#include "MaterialPronunciationCatalog.h"
#include "OttoMaterialIO.h"
#include "OttoMaterialMigration.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>

namespace hachi
{
namespace
{
using Catalog = MaterialPronunciationCatalog;
using juce::File;
using juce::String;
String key(const File& file)
{
    auto value = file.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    value = value.toLowerCase();
#endif
    return value;
}
bool inside(const File& file, const File& folder) { return file.isAChildOf(folder); }
bool hasAlias(const SampleRegionSetting& row, const String& alias)
{ return row.name == alias || row.additionalAliases.contains(alias); }
Catalog::Format formatFor(UtauMode mode)
{
    return mode == UtauMode::classic ? Catalog::Format::classicOto
        : mode == UtauMode::jie ? Catalog::Format::jieOto : Catalog::Format::mouOto;
}
void append(std::vector<Catalog::Entry>& entries, const File& audio, const Catalog::Root& root,
            const SampleRegionSetting& row, Catalog::Format format, const String& fingerprint)
{
    Catalog::Entry entry;
    entry.audio = audio; entry.root = root.folder;
    entry.manifest = root.folder.getChildFile("otto.ini");
    entry.audioId = row.materialAudioId; entry.regionId = row.materialRegionId;
    entry.row = row; entry.format = format; entry.otoMode = root.otoMode;
    entry.sourceFingerprint = fingerprint;
    entry.alias = row.name;
    if (entry.alias.isNotEmpty()) entries.push_back(entry);
    for (const auto& alias : row.additionalAliases)
        if (alias.isNotEmpty() && alias != row.name)
        { entry.alias = alias; entries.push_back(entry); }
}
}

Catalog::Result Catalog::index(const std::vector<Root>& requested, Cancel cancel)
{
    Result result;
    std::map<String, Root> unique;
    std::set<String> rejected;
    for (const auto& root : requested)
    {
        if (root.folder == File{} || !root.folder.isDirectory())
        { result.errors.add("Material root is not a directory: " + root.folder.getFullPathName()); continue; }
        if (root.otoMode != UtauMode::classic && root.otoMode != UtauMode::jie && root.otoMode != UtauMode::mou)
        {
            result.errors.add("Unsupported OTO mode: " + root.folder.getFullPathName());
            rejected.insert(key(root.folder));
            unique.emplace(key(root.folder), root); // ownership still blocks parent fallback
            continue;
        }
        auto [it, added] = unique.emplace(key(root.folder), root);
        if (!added && it->second.otoMode != root.otoMode)
        {
            result.errors.add("Duplicate material root has conflicting OTO modes: " + root.folder.getFullPathName());
            rejected.insert(key(root.folder));
        }
    }
    const auto cancelled = [&]()
    {
        if (cancel && cancel()) { result.cancelled = true; result.entries.clear(); return true; }
        return false;
    };
    const auto owns = [&](const File& audio, const Root& root)
    {
        if (!inside(audio, root.folder)) return false;
        for (const auto& [path, other] : unique)
            if (path.length() > key(root.folder).length() && inside(other.folder, root.folder)
                && inside(audio, other.folder)) return false;
        return true;
    };
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    // Local header durations are shared by aliases. No reader->read() calls.
    std::map<String, double> durations;
    for (const auto& [rootKey, root] : unique)
    {
        if (rejected.count(rootKey)) continue;
        if (cancelled()) return result;
        std::vector<Entry> rootEntries;
        const auto manifest = root.folder.getChildFile("otto.ini");
        if (manifest.exists())
        {
            OttoMaterialIO::Document document;
            String error;
            if (!OttoMaterialIO::loadDocument(manifest, document, error))
            { result.errors.add(error); continue; } // Invalid OTTO NEVER permits OTO fallback.
            for (const auto& recording : document.audio)
            {
                if (cancelled()) return result;
                const auto audio = root.folder.getChildFile(recording.relativePath);
                if (!owns(audio, root)) continue;
                if (!audio.existsAsFile())
                { result.warnings.add("Missing annotated recording: " + audio.getFullPathName()); continue; }
                for (const auto& row : recording.rows)
                    append(rootEntries, audio, root, row, Format::otto, {});
            }
        }
        else
        {
            // Per-file parser avoids appendUnlistedAudio and any fabricated
            // filename pronunciation. Each governing OTO is parsed once/root.
            std::map<String, File> files;
            for (const auto& item : juce::RangedDirectoryIterator(root.folder, true,
                     root.otoMode == UtauMode::jie ? "oto.ini;oto.jie.ini" : "oto.ini", File::findFiles))
            {
                if (cancelled()) return result;
                const auto classic = item.getFile().getSiblingFile("oto.ini");
                if (owns(classic, root)) files.emplace(key(classic), classic);
            }
            std::map<String, std::vector<VoicebankOtoEntry>> byAudio;
            std::map<String, OttoMaterialMigration::OtoSource> sources;
            bool failed = false;
            for (const auto& [path, file] : files)
            {
                juce::ignoreUnused(path);
                if (cancelled()) return result;
                juce::StringArray warnings;
                auto entries = SampleSettings::loadVoicebankOto(root.folder, warnings,
                    root.otoMode == UtauMode::jie, root.otoMode == UtauMode::mou, file);
                result.warnings.addArray(warnings);
                if (!warnings.isEmpty())
                {
                    result.errors.add("OTO source reported errors: " + file.getFullPathName()
                                      + ": " + warnings.joinIntoString("; "));
                    failed = true; break;
                }
                for (const auto& entry : entries)
                {
                    if (entry.lineIndex < 0 || !owns(entry.audioFile, root)) continue;
                    // Only ancestors can govern, identical to loadAudioOto.
                    if (!inside(entry.audioFile, entry.otoFile.getParentDirectory())) continue;
                    byAudio[key(entry.audioFile)].push_back(entry);
                }
            }
            if (failed) continue;
            for (const auto& [audioKey, entries] : byAudio)
            {
                if (cancelled()) return result;
                const auto audio = entries.front().audioFile;
                if (!audio.existsAsFile())
                { result.warnings.add("Missing OTO recording: " + audio.getFullPathName()); continue; }
                int nearest = 0;
                for (const auto& entry : entries)
                    nearest = std::max(nearest, key(entry.otoFile.getParentDirectory()).length());
                for (const auto& entry : entries)
                {
                    if (key(entry.otoFile.getParentDirectory()).length() != nearest) continue;
                    if (cancelled()) return result;
                    String error;
                    auto [source, added] = sources.try_emplace(key(entry.otoFile));
                    if (added && !OttoMaterialMigration::readOtoSource(entry, root.otoMode, source->second, error))
                    { result.errors.add(error); failed = true; break; }
                    // Use the converter's exact operation order: an equivalent
                    // expression can round below end and spuriously fail bounds.
                    double duration = entry.offsetMs / 1000.0 - entry.cutoffMs / 1000.0;
                    if (entry.cutoffMs >= 0)
                    {
                        auto found = durations.find(audioKey);
                        if (found == durations.end())
                        {
                            std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(audio));
                            if (!reader || !std::isfinite(reader->sampleRate) || reader->sampleRate <= 0
                                || reader->lengthInSamples <= 0 || reader->numChannels == 0)
                            { result.errors.add("Unreadable or empty recording: " + audio.getFullPathName()); failed = true; break; }
                            found = durations.emplace(audioKey, static_cast<double>(reader->lengthInSamples) / reader->sampleRate).first;
                        }
                        duration = found->second;
                    }
                    SampleRegionSetting row;
                    if (!OttoMaterialMigration::convertOtoEntry(entry, root.otoMode, duration, source->second, row, error))
                    { result.errors.add(error); failed = true; break; }
                    // An empty classic alias means filename fallback in UTAU;
                    // this catalog intentionally never advertises that inference.
                    if (entry.alias.isNotEmpty())
                        append(rootEntries, audio, root, row, formatFor(root.otoMode), entry.sourceFingerprint);
                }
                if (failed) break;
            }
            if (failed) continue; // No partial root result after a conversion error.
        }
        result.entries.insert(result.entries.end(), rootEntries.begin(), rootEntries.end());
    }
    std::sort(result.entries.begin(), result.entries.end(), [](const Entry& a, const Entry& b)
    {
        if (key(a.root) != key(b.root)) return key(a.root) < key(b.root);
        if (key(a.audio) != key(b.audio)) return key(a.audio) < key(b.audio);
        if (a.audioId != b.audioId) return a.audioId < b.audioId;
        if (a.regionId != b.regionId) return a.regionId < b.regionId;
        return a.alias < b.alias;
    });
    return result;
}

bool Catalog::resolve(const Entry& selected, Entry& current, String& error,
                      std::vector<SampleRegionSetting>* fullRows)
{
    error.clear();
    const bool native = selected.manifest.exists();
    if (selected.manifest != selected.root.getChildFile("otto.ini")
        || (native != (selected.format == Format::otto)))
    { error = "Governing annotation format changed; rebuild pronunciation catalog"; return false; }
    std::vector<SampleRegionSetting> rows;
    if (!OttoMaterialMigration::readAudioAnnotations(selected.audio, selected.root,
                                                   selected.otoMode, rows, error, selected.audioId)) return false;
    for (const auto& row : rows)
        if (row.materialAudioId == selected.audioId && row.materialRegionId == selected.regionId
            && hasAlias(row, selected.alias))
        {
            auto resolved = selected;
            resolved.row = row;
            current = std::move(resolved);
            if (fullRows) *fullRows = std::move(rows);
            return true;
        }
    error = "Pronunciation source changed or alias/identity disappeared; rebuild pronunciation catalog";
    return false;
}
}
