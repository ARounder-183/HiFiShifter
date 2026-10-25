#pragma once
#include "SampleSettings.h"

namespace hachi
{
// Pure annotation reads plus explicit, synchronous folder transactions.
class OttoMaterialMigration final
{
public:
    struct Report
    {
        // Successfully published additions (migration) or exported records.
        int audioFiles = 0, pronunciations = 0, aliases = 0;
        juce::StringArray warnings;
    };
    // Read only: root OTTO is authoritative even when this recording has no
    // section. Otherwise read nearest OTO; absent annotations yield empty rows.
    // Legacy CSV is reserved for explicit migration, never automatic precedence.
    // Empty audioId rejects ambiguous native paths; explicit IDs never fall back.
    static bool readAudioAnnotations(const juce::File& audio, const juce::File& root,
                                    UtauMode otoMode, std::vector<SampleRegionSetting>& rows,
                                    juce::String& error, const juce::String& audioId = {});
    // Immutable per-operation source snapshot: reuse across entries from one OTO.
    struct OtoSource
    {
        juce::StringArray lines;
        juce::String fingerprint;
    };
    static bool readOtoSource(const VoicebankOtoEntry&, UtauMode, OtoSource&, juce::String& error);
    // The same strict conversion and unchanged stable-ID algorithm used by
    // readAudioAnnotations. duration is physical end, or (negative cutoff only)
    // the authored absolute end while indexing; selection revalidates physical end.
    static bool convertOtoEntry(const VoicebankOtoEntry&, UtauMode, double duration,
                               const OtoSource&, SampleRegionSetting&, juce::String& error);
    // Explicit registration only: validate existing OTTO, preserve OTO banks,
    // otherwise atomically create an empty manifest without opening recordings.
    static bool ensureNativeFolder(const juce::File& root, juce::String& error);
    static bool migrateFolder(const juce::File& root, UtauMode otoMode,
                              bool includeUnannotated, Report&, juce::String& error);
    // New/replaced output is UTF-8 without BOM, LF, with #Charset:UTF-8.
    // Merge preserves the original codec/BOM and each untouched line/newline.
    static bool exportFolder(const juce::File& root, const juce::File& targetOto,
                             bool replaceExisting, Report&, juce::String& error);
};
}
