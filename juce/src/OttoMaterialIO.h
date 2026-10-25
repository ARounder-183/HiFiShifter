#pragma once
#include "SampleSettings.h"

namespace hachi
{
// Strict folder-level native material codec; never reads or writes classic OTO.
class OttoMaterialIO final
{
public:
    struct Audio
    {
        // Annotation variant identity, NOT physical file/source identity.
        juce::String id, relativePath;
        std::vector<SampleRegionSetting> rows;
    };
    struct Document
    {
        std::vector<Audio> audio;
        // Exact byte fingerprint from loadDocument; do not modify these fields.
        juce::MemoryBlock fingerprint;
        bool existed = false;
    };
    static constexpr double connectionToleranceSeconds = 0.001;
    static juce::File findManifest(const juce::File& audio);
    // Empty audioId selects only an unambiguous path; multiple variants reject.
    // Keep the optional SHA-256 token with each editable draft and pass it to
    // saveAudio. Without it only a bounded last-observation cache is checked.
    static bool loadAudio(const juce::File& manifest, const juce::File& audio,
                          std::vector<SampleRegionSetting>& rows, juce::String& error,
                          juce::String* loadedFingerprint = nullptr,
                          const juce::String& audioId = {});
    static bool saveAudio(const juce::File& manifest, const juce::File& audio,
                          const std::vector<SampleRegionSetting>& rows, juce::String& error,
                          const juce::String& expectedFingerprint = {},
                          const juce::String& audioId = {});
    // Existence only: true for one or more variants, including empty sections.
    static bool containsAudio(const juce::File& manifest, const juce::File& audio,
                              juce::String& error);
    // All annotation variants for one physical path, in document order.
    static bool listAudioVariants(const juce::File& manifest, const juce::File& audio,
                                 std::vector<Audio>& variants, juce::String& error);
    // Pure merge for batch transactions; validates the entire candidate and
    // leaves document untouched on failure. Does not read or write any file.
    // Explicit audioId (or unanimous nonempty row IDs) selects exactly that
    // variant, creating a sibling if new. ID-less ambiguous paths reject.
    static bool replaceAudio(Document& document, const juce::File& manifest,
                             const juce::File& audio, const std::vector<SampleRegionSetting>& rows,
                             juce::String& error, const juce::String& audioId = {});
    // Helpers are transactional: failed decode/load does not change output.
    // saveDocument requires the unchanged fingerprint from loadDocument (or a
    // default Document for a new file), rechecks under the named process lock,
    // and updates the fingerprint only after successful atomic publication.
    static bool decode(const juce::String& text, Document& document, juce::String& error);
    static bool encode(const Document& document, juce::String& text, juce::String& error);
    static bool validate(const Document& document, juce::String& error);
    static bool loadDocument(const juce::File& manifest, Document& document, juce::String& error);
    static bool saveDocument(const juce::File& manifest, Document& document, juce::String& error);
};
}
