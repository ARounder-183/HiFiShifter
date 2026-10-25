#pragma once
#include "OttoMaterialIO.h"
#include <functional>
#include <map>

namespace hachi
{
// openvpi COMMUNITY transcriptions.csv adapter, not original DiffSinger's
// transcriptions.txt and not an ASR/forced-aligner. All timing comes from input.
class CommunityDatasetIO final
{
public:
    using Cancel = std::function<bool()>;
    struct Report
    {
        juce::StringArray errors, warnings;
        bool cancelled = false, published = false;
        int recordings = 0, pronunciations = 0, skippedSpecialPhones = 0;
        bool ok() const { return !cancelled && errors.isEmpty(); }
    };
    struct Phone { juce::String symbol; double start = 0, end = 0; };
    struct ScoreNote { juce::String token, glide; double start = 0, end = 0; };
    struct Recording
    {
        juce::String name;
        juce::File audio;
        std::vector<Phone> phones;
        // openvpi's V-leading WORD division, NEVER a C+V syllable division.
        std::vector<int> wordPhoneCounts;
        std::vector<ScoreNote> notes;
        std::map<juce::String, juce::String> extraFields;
        double physicalDuration = 0;
        juce::String audioFingerprint;
    };
    struct Dataset
    {
        juce::File root, csv;
        juce::String csvFingerprint;
        std::vector<Recording> recordings;
    };
    struct Rule
    {
        juce::String alias;
        juce::StringArray phones;
        int nucleusOffset = 0; // explicit dictionary/import contract, not acoustic inference
    };
    struct Dictionary
    {
        juce::File file;
        juce::String fingerprint;
        std::vector<Rule> rules;
    };
    // For polysyllabic/ambiguous dictionaries or CSVs without syllable text,
    // caller explicitly chooses spans/aliases. Every non-SP/AP phone must be
    // covered once; omitted/gapped/overlapping speech is rejected, not invented.
    struct Selection
    {
        juce::String recordingName, alias;
        int firstPhone = 0, phoneCount = 0, nucleusOffset = 0;
        juce::StringArray additionalAliases;
    };
    enum class Specials { reject, skipWithReport, importDeclaredSPAP };
    struct Options
    {
        // Required explicit consent: dataset has no independent native P/C.
        // The selected nucleus onset is mapped to BOTH markers, through the
        // shared PronunciationTiming authority. Never an implicit default.
        bool confirmNucleusAsAlignmentAndConsonant = false;
        Specials specials = Specials::reject;
        bool replacePreviouslyImportedAudio = false;
        // Separate annotation registration for the SAME physical recording.
        // Explicit opt-in only; never overwrites a foreign same-path variant.
        bool allowSiblingAnnotationVariant = false;
        // Exact phone-sequence key (space joined) -> selected dictionary alias.
        // Required when a dictionary declares multiple aliases for one sequence.
        std::map<juce::String, juce::String> primaryAliases;
    };
    struct PlannedAudio
    {
        juce::File audio;
        juce::String audioId; // explicit annotation-registration ID, including empty-row variants
        std::vector<SampleRegionSetting> rows;
    };
    struct Plan
    {
        Dataset dataset;
        Dictionary dictionary;
        juce::File manifest;
        OttoMaterialIO::Document document; // original observed write fingerprint retained
        std::vector<PlannedAudio> audio;
        bool ready = false, published = false;
    };
    // Synchronous worker APIs, operation-local state, no folder registration.
    // Read is transactional; WAV > FLAC as in upstream. Headers + streaming
    // content hashes only; no PCM decoding/analysis or Python runtime.
    static bool read(const juce::File& root, Dataset&, Report&, Cancel = {});
    // Explicit documented V / C-V dictionary mode. Rejects >2-phone rules and
    // conflicting inventories. It does NOT infer polysyllabic syllables.
    static bool readTwoPartDictionary(const juce::File&, Dictionary&, Report&, Cancel = {});
    static bool planImport(const Dataset&, const Dictionary&, const std::vector<Selection>&,
                           const Options&, Plan&, Report&, Cancel = {});
    // One existing conflict-protected OttoMaterialIO atomic folder transaction.
    // Rechecks CSV/dictionary/audio bytes before publishing; no cancellation
    // after commit starts, no file-copy/audio rewrite, no automatic registration.
    static bool publish(Plan&, Report&, Cancel = {});
};
}
