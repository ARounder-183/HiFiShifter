#pragma once
#include "OttoMaterialIO.h"
#include "ProjectMaterialRegistrationTypes.h"
#include <functional>
#include <memory>
#include <optional>

namespace hachi
{
// Explicit, non-analysing registration planner. Never mutates a model, registers
// a manager root, saves HJPX, or reapplies source settings. All IO is caller-worker IO.
class ProjectMaterialRegistration final
{
public:
    using Cancel = std::function<bool()>;
    // Host authority: excludes ONLY reference/path/binding metadata changed by
    // makeCandidate; includes every musical field, raw clock, segment and array.
    using Fingerprint = std::function<juce::String(const ProjectData&)>;
    using Locator = ProjectMaterialRegistrationLocator;
    // nullopt = all eligible native source owners; present EMPTY = no selection.
    // Entries identify physical owners (noteId MUST be empty), not note slices.
    using OwnerScope = std::optional<std::vector<Locator>>;
    // One pure selection authority shared by prepare and the confirmed adapter.
    // Resolves/validates selected owner IDs only; never hashes or decodes audio.
    static bool resolveOwnerScope(const ProjectData&, const OwnerScope&,
        std::vector<Locator>& selected, juce::String& error);
    struct RowFacts
    {
        juce::String key, nextRowKey;
        // Absolute recording coordinates. ONLY explicit carried pronunciation
        // facts; never a pitch event/analysis epoch masquerading as a row.
        SampleRegionSetting annotation;
        std::vector<Locator> members;
    };
    struct VariantFacts
    {
        juce::File recording;
        juce::String variantKey;
        bool explicitPronunciationEvidence = false;
        std::vector<RowFacts> rows;
    };
    struct Recording
    {
        juce::String key, sha256, copiedRelativePath;
        juce::int64 bytes = 0;
        juce::File original;
        std::vector<juce::File> aliases;
    };
    using OwnerBinding = ProjectMaterialRegistrationOwnerBinding;
    using Binding = ProjectMaterialRegistrationNoteBinding;
    struct Variant
    {
        // Distinct actual usage keys can share audioId only when their entire
        // published annotation sets (including explicit links) are identical.
        juce::String recordingKey, variantKey, audioId;
        std::vector<std::pair<juce::String, juce::String>> rowIds; // caller row key -> stable region ID
    };
    struct Plan
    {
        std::uint64_t expectedRevision = 0;
        juce::String musicSourceFingerprint;
        juce::File destinationParent, publishedFolder;
        std::vector<Recording> recordings;
        std::vector<Variant> variants;
        std::vector<OwnerBinding> owners;
        std::vector<Binding> bindings;
        OttoMaterialIO::Document document;
        juce::StringArray diagnostics;
        // False for unresolved evidence or failed shared-codec validation.
        // Independent audio IDs may share one copied physical recording path.
        bool ready = false;
    };
    // Read-only: hashes source bytes and inspects audio headers, writes NOTHING.
    // Failed/ambiguous plans retain preview diagnostics, never approximate cuts.
    static Plan prepare(const ProjectData&, std::uint64_t expectedRevision,
        const juce::File& destinationParent, const juce::String& newFolderName,
        const std::vector<VariantFacts>&, const Fingerprint&, Cancel = {},
        const OwnerScope& selectedOwners = std::nullopt);
    // Pure candidate: only sourceFile and explicit stable-binding metadata may
    // change. Existing source time maps, notes, nativeSegments and arrays stay exact.
    static bool makeCandidate(const ProjectData&, const Plan&, const Fingerprint&,
        ProjectData& candidate, juce::String& error);

    class Lease final
    {
    public:
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        const Plan& plan() const { return prepared; }
        const juce::File& currentFolder() const { return ownedFolder; }
        bool published() const { return isPublished; }
        // No replacement of an existing destination. Rechecks copied bytes and
        // original hashes. Parent must recheck revision BEFORE this operation.
        bool publish(juce::String& error, Cancel = {});
        // Compensating rollback ONLY, not a crash-atomic multi-resource commit.
        // Cleanup refuses an absent/mismatched ownership marker or reparse tree.
        bool rollback(juce::String& error);
        // Call IMMEDIATELY after successful ProjectModel adoption, BEFORE any
        // fallible manager preference write. Always disarms destructor rollback;
        // no IO/validation can fail or re-arm it. Keep marker as provenance.
        // Preference failure is a warning/retry, never deletion of adopted media.
        void retainAfterAdoption() noexcept { permanentlyRetained = true; released = true; }
        // Backward-compatible published-lease spelling; no filesystem validation
        // after adoption. Prefer retainAfterAdoption() at the exact commit boundary.
        bool release(juce::String& error);
    private:
        friend class ProjectMaterialRegistration;
        Lease() = default;
        Plan prepared;
        juce::File ownedFolder;
        juce::String ownershipToken;
        bool isPublished = false, released = false, permanentlyRetained = false;
        bool verify(juce::String& error, Cancel) const;
    };
    // Creates a unique child beneath destinationParent and writes via the shared
    // OttoMaterialIO::saveDocument only. No project/library side effects.
    static std::unique_ptr<Lease> stage(const Plan&, juce::String& error, Cancel = {});
};
}
