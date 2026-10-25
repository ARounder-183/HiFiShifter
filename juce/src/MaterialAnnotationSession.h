#pragma once
#include "OttoMaterialIO.h"
#include <memory>
#include <functional>

namespace hachi
{
// One draft model for both the material manager and in-project quick editor.
// All coordinates are absolute source-audio seconds. Editing never cuts PCM.
class MaterialAnnotationSession final
{
public:
    static std::unique_ptr<MaterialAnnotationSession> open(const juce::File& audio,
        const juce::File& manifest, juce::String& error,
        const juce::String& selectedAudioId = {});
    enum class EditScope { source, thisNote };
    enum class NoteScopeCapability { nativeFullRow, classicScalarOto };
    struct NoteScopeBinding
    {
        juce::String noteId, audioId, regionId, sourceFingerprint;
        juce::File sourceAudio;
        juce::uint64 projectRevision = 0;
        SampleRegionSetting localRow; // actual model facts, absolute source seconds
        NoteScopeCapability capability = NoteScopeCapability::nativeFullRow;
        // Host/model atomically checks ID/revision/source fingerprint/stable IDs.
        // Success is one project-only undo transaction, never a source publish.
        std::function<bool(const NoteScopeBinding&, const SampleRegionSetting&,
                           juce::uint64& resultingRevision, juce::String&)> apply;
        std::function<bool(const NoteScopeBinding&, SampleRegionSetting& restored,
                           juce::uint64& resultingRevision, juce::String&)> restore;
        // Read-only host getter, used only after a clean source publication.
        std::function<bool(const NoteScopeBinding& current, NoteScopeBinding& refreshed,
                           juce::String& error)> refresh;
    };
    bool enableNoteScope(NoteScopeBinding, juce::String& error);
    bool refreshNoteScopeBinding(juce::String& error);
    bool setEditScope(EditScope, juce::String& error);
    // Only after caller explicitly confirms discarding unpublished edits.
    bool discardAndSetEditScope(EditScope, juce::String& error);
    bool restoreLocal(juce::String& error);
    EditScope editScope() const { return scope; }
    bool hasNoteScope() const { return static_cast<bool>(noteBinding.apply); }
    bool isClassicNoteScope() const
    { return scope == EditScope::thisNote && noteBinding.capability == NoteScopeCapability::classicScalarOto; }
    bool canRestoreLocal() const { return static_cast<bool>(noteBinding.restore); }
    const NoteScopeBinding& noteScopeBinding() const { return noteBinding; }
    bool dirty() const;
    const std::vector<SampleRegionSetting>& rows() const { return draft; }
    double durationSeconds() const { return duration; }
    const juce::String& normalizationSummary() const { return normalizationMessage; }
    bool createPronunciation(double sourceStart, double sourceEnd, const juce::String& alias, juce::String& error);
    // Preview and creation share the same nearest-free-window planner. The
    // anchor join is exact; only the free outer edge can be constrained.
    bool previewAdjacentRange(const juce::String& anchorId, bool before, double freeEdge,
                              double& sourceStart, double& sourceEnd, juce::String& error) const;
    bool createAdjacentPronunciation(const juce::String& anchorId, bool before, double freeEdge,
                                     const juce::String& alias, juce::String& error);
    bool removeAlias(const juce::String& regionId, const juce::String& alias, juce::String& error);
    bool split(const std::vector<double>& sourceTimes, juce::String& error);
    // Manual internal vowels: explicit names are mandatory, never inferred from
    // analyser cuts. Row planners are shared with the editor's unpublished draft.
    bool splitVowel(const juce::String& regionId, const juce::String& segmentId, double sourceCut,
                    const juce::String& leftName, const juce::String& rightName, juce::String& error);
    bool editVowel(const juce::String& regionId, const juce::String& segmentId,
                   const juce::String& name, double stretchWeight, juce::String& error);
    bool mergeVowels(const juce::String& regionId, const juce::String& leftSegmentId, juce::String& error);
    static bool planSplitVowel(SampleRegionSetting&, const juce::String& segmentId, double sourceCut,
                              const juce::String& leftName, const juce::String& rightName, juce::String& error);
    static bool planEditVowel(SampleRegionSetting&, const juce::String& segmentId,
                             const juce::String& name, double stretchWeight, juce::String& error);
    static bool planMergeVowels(SampleRegionSetting&, const juce::String& leftSegmentId, juce::String& error);
    bool moveBoundary(const juce::String& leftRegionId, double sourceSeconds, juce::String& error);
    bool rename(const juce::StringArray& orderedAliases, juce::String& error);
    bool setRow(const juce::String& regionId, const SampleRegionSetting&, juce::String& error);
    bool merge(const std::vector<juce::String>& regionIds, juce::String& error);
    bool erase(const std::vector<juce::String>& regionIds, juce::String& error);
    bool undo(); bool redo();
    bool canUndo() const { return !undoStates.empty(); }
    bool canRedo() const { return !redoStates.empty(); }
    bool save(juce::String& error);
    const juce::File& sourceAudio() const { return audioFile; }
    const juce::File& manifestFile() const { return manifest; }
    const juce::String& revisionToken() const { return fingerprint; }
    // Called before publication, e.g. to reject an unsafe project remapping.
    std::function<bool(const std::vector<SampleRegionSetting>&, juce::String&)> validatePublication;
private:
    bool insertPronunciation(double start, double end, const juce::String& alias,
                             const juce::String& adjacentAnchor, bool before, juce::String& error);
    bool commit(std::vector<SampleRegionSetting>, juce::String& error);
    bool validate(const std::vector<SampleRegionSetting>&, juce::String& error) const;
    bool encodeDraft(const std::vector<SampleRegionSetting>&, juce::String& text, juce::String& error) const;
    bool switchScope(EditScope, bool discard, juce::String& error);
    EditScope scope = EditScope::source;
    NoteScopeBinding noteBinding;
    std::vector<SampleRegionSetting> cleanDraft;
    juce::File audioFile, manifest;
    juce::String audioId, fingerprint, normalizationMessage, noteScopeRefreshError;
    bool unpublishedEmptySource = false;
    double duration = 0;
    std::vector<SampleRegionSetting> draft;
    std::vector<std::vector<SampleRegionSetting>> undoStates, redoStates;
};
}
