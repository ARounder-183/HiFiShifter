#pragma once
#include "SampleSettings.h"
#include <functional>

namespace hachi
{
// Read-only, operation-local indexing. Call on a caller-owned worker, never paint
// or the audio callback. Does not register folders, create manifests or decode PCM.
class MaterialPronunciationCatalog final
{
public:
    struct Root { juce::File folder; UtauMode otoMode = UtauMode::classic; };
    enum class Format { otto, classicOto, jieOto, mouOto };
    struct Entry
    {
        juce::File audio, root, manifest; // root/otto.ini, possibly absent for OTO
        juce::String audioId, regionId, alias;
        SampleRegionSetting row; // untouched authoritative row, including secondary aliases
        Format format = Format::otto;
        UtauMode otoMode = UtauMode::classic;
        juce::String sourceFingerprint; // classic source-byte fingerprint
    };
    struct Result
    {
        std::vector<Entry> entries;
        juce::StringArray warnings, errors;
        bool cancelled = false;
        bool ok() const { return !cancelled && errors.isEmpty(); }
    };
    using Cancel = std::function<bool()>;
    // Same normalized root is visited once; conflicting modes reject that root
    // with an error. Deepest registered root owns a nested recording.
    // Order is normalized root/audio path, variant ID, row ID, alias (case sensitive).
    // Existing OTTO governs the whole root, even absent/empty audio sections.
    static Result index(const std::vector<Root>& roots, Cancel cancel = {});
    // Required immediately before insertion/replacement, on the caller worker.
    // Reload via readAudioAnnotations, validating physical bounds and source IDs.
    // A stale classic fingerprint/row ID rejects; never substitutes by row index
    // or alias. On failure output is unchanged. Does not publish annotations.
    // Optional fullRows receives ALL authoritative rows of the selected audioId
    // variant ONLY (never its same-physical-path siblings), for existing model
    // insertion/replacement, not a newly constructed single-alias business path.
    static bool resolve(const Entry& selected, Entry& current, juce::String& error,
                        std::vector<SampleRegionSetting>* fullRows = nullptr);
};
}
