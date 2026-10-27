#pragma once

#include "I18n.h"
#include "SampleSettings.h"
#include "MaterialAnnotationSession.h"
#include <optional>
#include "ModelessWindows.h"
#include <juce_gui_extra/juce_gui_extra.h>
#include <memory>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace hachi
{
struct ProjectData;
class OtoWaveformEditorComponent;
// One folder-based material browser. Root OTTO is authoritative; otherwise
// existing OTO remains read-only input until an explicit native annotation save.
// Bare registered folders receive an empty OTTO manifest, never fake aliases.
// Folder actions live in one menu; precise editing reuses the old OTO editor.
class AssetManagerComponent final : public juce::Component,
                                    public juce::FileDragAndDropTarget,
                                    private juce::ScrollBar::Listener
{
public:
    AssetManagerComponent(I18n& stringsToUse, juce::PropertiesFile& propertiesToUse);
    // Project references are views of the original banks, not global imports.
    // Returns false when the folder set is unchanged (no member rescan).
    bool setProjectVoicebanks(const ProjectData& project);
    std::function<void(const juce::String& state, const juce::String& session)> onBrowserStateChanged;
    // Production registration shares the folder action's authority/cache writer.
    // Registration/removal memory is published to the owning project.
    bool registerMaterialRoot(const juce::File& folder, juce::String& error);
    // Refresh visible pronunciation names after any shared editor publishes.
    void refreshMaterialAnnotations(const juce::File& source);
    // Load the original recording through the host's ordinary audio importer.
    struct MaterialSelection
    {
        juce::File audio, root;
        juce::String audioVariantId;
        juce::StringArray regionIds;
    };
    std::function<void(const MaterialSelection&)> onLoadMaterialSelection;
    std::function<void(const juce::File&)> onLoadAudio;
    std::function<void(const juce::File&, const juce::StringArray&)> onLoadPronunciations;
    std::function<void(const juce::File&)> onUseMaterialFolder;
    // Host owns the existing async dataset wizard/publication service.
    // Empty root requests a chooser; a root context supplies that exact folder.
    std::function<void(const juce::File& initialRoot)> onRequestCommunityDatasetImport;
    std::function<void(OtoWaveformEditorComponent&)> onConfigureMaterialEditor;
    // Synchronous preflight and completion for the shared material editor.
    std::function<bool(const juce::File&, const std::vector<SampleRegionSetting>&, juce::String&)> onMaterialPreflight;
    std::function<void(const juce::File&, const std::vector<SampleRegionSetting>&)> onMaterialSaved;
    [[nodiscard]] int otoImportKind(const juce::File& audioOrFolder = {}) const;
    // Compatibility seam only: format selection is automatic, never persisted.
    void setOtoImportKind(const juce::File& folder, int kind);
    [[nodiscard]] juce::File rootForAudio(const juce::File& file) const;
    [[nodiscard]] juce::var audioDragDescription(const juce::File& file,
                                                       const juce::String& audioVariantId = {}) const;
    [[nodiscard]] juce::var pronunciationDragDescription(const juce::File& file,
                                                       const juce::StringArray& ids,
                                                       const juce::String& audioVariantId = {}) const;
    [[nodiscard]] juce::StringArray materialRoots() const;
    // Access aliases (e.g. Ctrl+G) delegate to this same precise editor.
    void editMaterial(const juce::File& member, const juce::String& regionId = {},
                      const juce::String& audioVariantId = {},
                      std::optional<MaterialAnnotationSession::NoteScopeBinding> noteScope = {},
                      MaterialAnnotationSession::EditScope initialScope = MaterialAnnotationSession::EditScope::source,
                      std::optional<juce::Range<double>> sourceRange = {})
    { openMemberEditor(member, regionId, audioVariantId, std::move(noteScope), initialScope, sourceRange); }
    // External applications retain the original/copy policy; internal drops use OTO metadata.
    [[nodiscard]] juce::File dragFileFor(const juce::File& member);
    bool keyPressed(const juce::KeyPress& key) override;
    void paint(juce::Graphics& g) override;
    void resized() override;
    void lookAndFeelChanged() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails&) override;
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    // Test seams: the model without a window, so a check can build a folder,
    // add audio and assemble lyrics without a mouse.
    [[nodiscard]] int diagnosticFolderCount() const
    {
        return static_cast<int>(folders.size());
    }
    [[nodiscard]] juce::StringArray diagnosticMembersOf(int folderIndex) const;
    [[nodiscard]] juce::StringArray diagnosticProjectVoicebanks() const;
    [[nodiscard]] int diagnosticTreeRowCount() const { return static_cast<int>(treeRows.size()); }
    [[nodiscard]] juce::File diagnosticSelectedAudio() const;
    juce::TextEditor& diagnosticSearchEditor() { return searchEditor; }
    [[nodiscard]] bool diagnosticSearchPending() const { return searchScanPending; }
    [[nodiscard]] juce::StringArray diagnosticSearchLabels() const;
    [[nodiscard]] juce::PopupMenu::Options contextMenuOptions(juce::Point<int> screenPosition);
    void removeFromBrowser(const juce::File& path, bool isFolder);
    void diagnosticToggleAudio(const juce::File& audio, const juce::String& audioVariantId = {});
    bool diagnosticSelectPronunciation(const juce::File& audio, const juce::String& regionId,
                                       bool control = false, bool shift = false,
                                       const juce::String& audioVariantId = {});
    [[nodiscard]] juce::StringArray diagnosticSelectedRegionIds() const { return selectedRegionIds; }
    [[nodiscard]] juce::StringArray diagnosticPronunciationLabels(const juce::File& audio,
                                                                      const juce::String& audioVariantId = {});
    [[nodiscard]] juce::StringArray diagnosticRegistrationIds(const juce::File& audio);
    bool diagnosticSelectRegistration(const juce::File&, const juce::String& audioVariantId = {});
    [[nodiscard]] juce::var diagnosticTreeSnapshot() const;
    [[nodiscard]] juce::String diagnosticSelectedVariantId() const { return selectedAudioVariantId; }
    [[nodiscard]] int diagnosticManifestLoadCount() const { return manifestLoadCount; }
    [[nodiscard]] bool diagnosticAnnotationReadPending() const
    {
        for (const auto& entry : manifestCaches) if (!entry.second.pendingAudio.empty()) return true;
        return false;
    }
    [[nodiscard]] int diagnosticRegisterFolder(const juce::File& folder);
    // Exact same menu item/dispatch as both production folder menus.
    [[nodiscard]] juce::PopupMenu diagnosticCommunityDatasetImportMenu() const;
    bool diagnosticRequestCommunityDatasetImport(const juce::File& initialRoot = {})
    { return requestCommunityDatasetImport(initialRoot); }
    // Copy audio into a folder without inventing annotations, as a drop would.
    // Returns how many files were taken in.
    int diagnosticAddAudioToFolder(int folderIndex, const juce::StringArray& paths);
    // Lyrics -> pinyin -> match the source folder -> a new ordered folder.
    // Returns the created folder path, or empty with the reason in error.
    [[nodiscard]] juce::File diagnosticAssembleLyrics(int sourceFolderIndex,
        const juce::String& lyrics, const juce::String& newFolderName,
        int& matched, int& missing, juce::String& error);
    // Test seam: export the registered folder at this index as one oto.ini.
    int diagnosticExportFolderAsOto(int folderIndex, const juce::File& otoFile,
                                    juce::String& error);
    // Test seam: apply native timing to a member and report success.
    static bool diagnosticWriteNativeTiming(const juce::File& member,
        double offsetMs, double consonantMs, double cutoffMs,
        double preutterMs, double overlapMs, juce::String& error);

private:
    struct MaterialFolder
    {
        juce::File directory;
        juce::String name;
        bool registered = true;
        bool projectVoicebank = false;
        bool expanded = false;
        std::vector<juce::File> children;
        bool childrenLoaded = false;
        bool projectAudio = false;
    };
    // member -1: folder, -2: missing/empty folder; region -1: audio,
    // -2: no-pronunciation/error hint, >=0: cached pronunciation (ID authoritative).
    struct TreeRow { int folder; int member; int top; int height; int region = -1; juce::String regionId, audioVariantId; };
    struct Registration { juce::String audioId; std::vector<SampleRegionSetting> rows; };
    struct ManifestCache
    {
        bool loaded = false, exists = false;
        juce::int64 modified = 0, size = -1;
        juce::String error;
        std::map<juce::String, std::vector<Registration>> recordings;
        std::set<juce::String> pendingAudio;
        std::map<juce::String, juce::String> audioErrors, audioRevisions;
        std::map<juce::String, std::uint64_t> audioRequests;
    };
    void ensureOtoRows(const juce::File& audio);
    static constexpr int communityDatasetImportAction = 20;
    void addCommunityDatasetImportAction(juce::PopupMenu&) const;
    bool requestCommunityDatasetImport(const juce::File& initialRoot);
    void showFolderActions();
    const ManifestCache& ensureManifest(const juce::File& root) const;
    const std::vector<SampleRegionSetting>* cachedRows(const juce::File& audio,
                                                       const juce::String& audioVariantId = {}) const;
    bool resolveVariantId(const juce::File&, const juce::String& requested, juce::String& resolved) const;
    bool dispatchSelection(const juce::File&, const juce::String&, const juce::StringArray&);
    void toggleAudio(int folderIndex, int memberIndex, const juce::String& audioVariantId = {});
    bool selectPronunciation(const juce::File& audio, const juce::String& regionId,
                             bool control, bool shift, bool preserveGroup = false,
                             const juce::String& audioVariantId = {});
    void prunePronunciationSelection();
    void rebuildTreeRows();
    void ensureSearchFiles();
    juce::String audioRowLabel(const TreeRow& row) const;
    void updateScrollRange();
    void scrollBarMoved(juce::ScrollBar*, double newRangeStart) override;
    void toggleFolder(int index);
    int treeRowAt(int y) const;

    void load();
    bool updateProjectVoicebanks(const ProjectData&);
    bool updateProjectAudio(const ProjectData&, bool force);
    std::vector<juce::File> projectAudioSources;
    bool projectAudioDirty = false;
    void save();
    int registerFolder(const juce::File& folder, const juce::String& displayName = {});
    void createFolderDialog();
    juce::String registrationError;
    void importAudioFolderDialog();
    void addAudioToSelectedFolderDialog();
    void assembleLyricsDialog();
    void chooseOtoExport(const juce::File& folder);
    void migrateFolderDialog(const juce::File& folder);
    void removeSelectedFolder();
    // Export native OTTO as standard OTO for interoperability, merging by
    // recording path while preserving the original native manifest.
    // Returns rows written, or -1 with the reason in error.
    int exportFolderAsOto(const juce::File& folder, const juce::File& otoFile,
                          juce::String& error);
    // Copy/register audio only; explicit annotation edits create aliases.
    // Returns the landed file, reporting any registration error.
    juce::File takeAudioIntoFolder(const juce::File& source, const juce::File& folder,
                                   juce::String& error);
    [[nodiscard]] static std::vector<juce::File> membersOf(const juce::File& folder);
    void rebuildMemberList(bool refresh = true);
    // The single precise editor; native Save always publishes folder OTTO.
    void openMemberEditor(const juce::File& member, const juce::String& regionId = {},
                          const juce::String& audioVariantId = {},
                          std::optional<MaterialAnnotationSession::NoteScopeBinding> noteScope = {},
                          MaterialAnnotationSession::EditScope initialScope = MaterialAnnotationSession::EditScope::source,
                          std::optional<juce::Range<double>> sourceRange = {});
    bool loadAudio(const juce::File& member);
    // Legacy diagnostic compatibility only; no visible editor calls this writer.
    // Production publication goes through the shared session validator/save.
    static bool writeNativeTiming(const juce::File& member,
                                  const VoicebankOtoEntry& edited, juce::String& error);
    // First drag of a member uses the original; a repeat drag hands over a
    // copy, so an external editor that rewrites the file (Melodyne) never
    // touches the registered original.
    [[nodiscard]] juce::File copyRoot() const;

    // Normalised pinyin key of a filename stem for lyric matching:汉字 -> pinyin,
    // then digits and tone marks dropped, lower cased -- so ba / ba2 / bà align.
    [[nodiscard]] static juce::String pinyinMatchKey(const juce::String& stem);

    I18n& strings;
    juce::PropertiesFile& properties;
    juce::String browserState, browserSession;
    std::uint64_t browserGeneration = 0;
    juce::TextButton folderActionsButton;
    std::vector<MaterialFolder> folders;
    std::vector<juce::File> members;
    std::set<juce::String> draggedOnce;
    std::set<juce::String> expandedAudio;
    std::set<juce::String> hiddenFolders, hiddenAudio;
    mutable std::map<juce::String, ManifestCache> manifestCaches;
    mutable int manifestLoadCount = 0;
    std::uint64_t nextAnnotationRequest = 0;
    juce::File pronunciationSelectionAudio;
    juce::String selectedAudioVariantId;
    juce::StringArray selectedRegionIds;
    juce::String selectionAnchor;
    bool pronunciationSelectionActive = false;
    int selectedFolder = -1;
    int selectedMember = -1;
    juce::File dragStartFile;
    juce::String dragStartVariantId;
    juce::TextEditor searchEditor;
    juce::TextButton clearSearchButton;
    juce::String searchQuery;
    bool searchScanPending = false;
    std::vector<TreeRow> treeRows;
    juce::ScrollBar treeScroll { true };
    int scrollOffset = 0;
    std::unique_ptr<juce::FileChooser> chooser;
    std::unique_ptr<juce::AlertWindow> dialog;
    ModelessWindows editorWindows;
    static constexpr int toolbarHeight = 70;
    static constexpr int rowHeight = 40;
    static constexpr int audioRowHeight = 24;
    juce::ThreadPool searchPool { 1 };
};
}
