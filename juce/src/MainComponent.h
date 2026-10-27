#pragma once

#include "AudioEngine.h"
#include "ProjectRecovery.h"
#include "I18n.h"
#include "PianoRollComponent.h"
#include "SettingsComponent.h"
#include "SampleSettings.h"
#include "VoicebankSettingsComponent.h"
#include "AssetManagerComponent.h"
#include "Theme.h"
#include "ZoomScrollBar.h"
#include "TimelineComponent.h"
#include "TrackListComponent.h"
#include "backend/MelodyneImporter.h"
#include "backend/MelodyneProvider.h"
#include "backend/AnalysisService.h"
#include <juce_gui_extra/juce_gui_extra.h>
#include <set>
#include <map>
#include <optional>
#include <atomic>

namespace hachi
{
class OtoWaveformEditorComponent;
struct MaterialPronunciationCandidate;

class EditorViewport final : public juce::Viewport
{
public:
    std::function<void(bool vertical, bool leading)> onZoomBegin;
    std::function<void(bool vertical, double scale)> onZoomDrag;
    void enableZoomHandles() { zoomHandles = true; recreateScrollbars(); }
    std::function<bool(const juce::MouseEvent&, const juce::MouseWheelDetails&)> onWheel;
    void mouseWheelMove(const juce::MouseEvent& event,
                        const juce::MouseWheelDetails& wheel) override
    {
        if (onWheel && onWheel(event, wheel)) return;
        juce::Viewport::mouseWheelMove(event, wheel);
    }
private:
    bool zoomHandles = false;
    juce::ScrollBar* createScrollBarComponent(bool vertical) override
    {
        if (!zoomHandles) return juce::Viewport::createScrollBarComponent(vertical);
        auto* bar = new ZoomScrollBar(vertical);
        bar->onResizeBegin = [this, vertical](bool leading)
        { if (onZoomBegin) onZoomBegin(vertical, leading); };
        bar->onResize = [this, vertical](double scale)
        { if (onZoomDrag) onZoomDrag(vertical, scale); };
        return bar;
    }
};

// A button that opens a menu, and says so with the same chevron the mode
// switcher shows.  The text is drawn clear of the arrow's column rather than
// centred over the whole button, so a longer name cannot run into it.
class DropdownButton final : public juce::TextButton
{
public:
    void paintButton(juce::Graphics& g, bool highlighted, bool down) override;
    // Where the text goes, given the whole button.  Pure and static so a check
    // can ask whether a label fits without painting anything.
    [[nodiscard]] static juce::Rectangle<int> textAreaFor(juce::Rectangle<int> bounds);
};

// A tool button that answers the right button too, so a tool can carry its
// own settings without taking a second slot in a row that has none to spare.
class ToolButton final : public juce::TextButton
{
public:
    std::function<void(juce::Point<int>)> onSecondaryClick;
    void mouseDown(const juce::MouseEvent& event) override
    {
        if (event.mods.isPopupMenu() && onSecondaryClick)
        {
            onSecondaryClick(event.getScreenPosition());
            return;
        }
        juce::TextButton::mouseDown(event);
    }
};

class MainComponent final : public juce::Component,
                            public juce::FileDragAndDropTarget,
                            public juce::DragAndDropContainer,
                            public juce::DragAndDropTarget,
                            private juce::ChangeListener,
                            private juce::ComponentListener,
                            private juce::Timer,
                            private juce::MenuBarModel
{
public:
    bool diagnosticHamood(const juce::File& directory);
    void applyRenderingPreference();
    void startLiveMcp();
    void stopLiveMcp();
    void pollLiveMcp();
    bool mcpExportInProgress() const;
    void liveMcpDocumentChanged();
    void showDiffSingerError(const juce::String& message);
    juce::var handleLiveMcp(const juce::var&, bool& respond);
    juce::var liveMcpStatus() const;
    juce::var liveMcpNotes(const juce::var& args, bool selectionOnly) const;
    void finishMcpDiffSinger(bool ok, const juce::String& message = {});
    bool reportMcpDiffSingerError(const juce::String& message);
    struct McpState;
    std::shared_ptr<McpState> mcp;


    // Where a paste lands when nothing says otherwise.  Carried to another
    // track it goes to the moment it was taken from, which is the point of
    // carrying it; dropped back on its own track that would be on top of
    // itself, so there it goes to the playhead, as it always did.  Pure, so
    // the rule can be checked without a window.
    //
    // The pointer comes before the selection because after a copy the
    // selection is what was copied: pasting there put the copy exactly on top
    // of its own original, and off a UTAU track notes are placed rather than
    // pushed along, so nothing appeared to happen at all.
    // Where a copied piece of material lands.
    //
    // The lane under the pointer when there is one, which is the only way to
    // name another track: clicking an empty lane reports no clip and does not
    // say whose lane it was, so before this a paste could only ever go to the
    // track already in hand.  Failing that, the track in hand and the
    // playhead, and if the playhead is not past the original, immediately
    // after it -- so a plain copy-paste still stacks copies end to end.
    struct ClipPasteTarget { juce::String trackId; double seconds = 0.0; };
    [[nodiscard]] static ClipPasteTarget clipPasteTargetFor(
        const juce::String& pointerTrackId, std::optional<double> pointerSeconds,
        const juce::String& trackInHand, double playheadSeconds,
        double sourceStartSeconds, double sourceDurationSeconds);
    // Which clip on a track a phrase goes into: the one covering the moment
    // asked for, else the track's first.  Notes are held relative to their
    // clip, so which clip is chosen decides what their times mean.  Pure, and
    // public so a check can follow the whole chain without a window.
    // The clip a pasted phrase goes into, and what to do when there is
    // none.  A melodic track starts with no clips at all and every
    // note-making path needs one, so pasting onto a track just made did
    // nothing, silently -- the first paste makes the clip, as the first
    // stroke of the draw tool does, and from the same place: the start of
    // the timeline, so the bars before it stay reachable.  One bar is only a
    // seed; a clip with no recording behind it stretches to fit.
    struct PasteClipPlan
    {
        juce::String clipId;      // empty when one has to be made
        bool makeOne = false;     // false and empty means there is nowhere
        // From the start of the timeline, so the bars before the pasted
        // phrase stay reachable.  Carried here rather than written at the
        // call, so it is part of the rule a check can read.
        double startSeconds = 0.0;
        double seedSeconds = 0.0;
    };
    [[nodiscard]] static PasteClipPlan pasteClipPlanFor(const ProjectData& data,
                                                        const juce::String& trackId,
                                                        double atSeconds);
    [[nodiscard]] static juce::String pasteTargetClipIn(const ProjectData& data,
                                                        const juce::String& trackId,
                                                        double atSeconds);
    // Whether importing a UST has to ask which way: with a song already open
    // the choice between opening the new one and laying it beside the old one
    // is the user's, and with nothing open there is no choice to make.
    [[nodiscard]] static bool ustImportNeedsChoice(const ProjectData& data)
    {
        return !data.tracks.empty();
    }
    [[nodiscard]] static double pasteTargetSeconds(
        bool sameTrack, double playheadSeconds, double originSeconds,
        std::optional<double> selectionStartSeconds,
        std::optional<double> askedFor,
        std::optional<double> pointerSeconds);
    // Where a panel should scroll to while playing, or nothing to leave it be.
    // A run that fits the window is shown whole and then held still; anything
    // longer is paged after, as before.  Pure, so it can be checked directly.
    // Where the roll should look after a request to leave source-edit mode.
    // Leaving it really does move the view back to what the timeline is
    // showing; a request that changes nothing must not move it at all, since
    // both the loudness lane and the point tool ask for it on every click.
    // Seconds, not pixels: the two views run at different scales.  Pure.
    // Which stretch clocks a pitch algorithm actually reaches, as picker item
    // ids in display order.  Only two backends read the choice: the neural
    // decoder, which takes it as a splice order, and vslib, which *is* the
    // Signalsmith stretcher this configures.  mld5, mld3, WORLD and llsm2
    // stretch from the time map inside their own renderers and never receive
    // the value -- it is dropped at the dispatch.  Empty means the picker
    // would do nothing and is hidden instead of sitting there inert.
    // What the wheel does in the editors.  The wheel scrolls and the
    // modifiers zoom, which is the way round every other editor of this kind
    // works.  Pure, and public so a check reads the same mapping the handler
    // dispatches on.
    using WheelAction = EditorWheelAction;
    [[nodiscard]] static WheelAction wheelActionFor(const juce::ModifierKeys& modifiers);

    // The two envelope lanes at the bottom of the roll.  Only one can be open:
    // they occupy the same strip and drag the same way, so with both believing
    // they were open a drag went to whichever the roll's tool happened to be
    // while the buttons said something else.
    enum class EnvelopeLane { none, amplitude, flagCurve };
    // Clicking the button for a lane opens it, or closes it if it was already
    // the open one.  Pure, and public so a check reads the same rule the
    // buttons act on.
    [[nodiscard]] static EnvelopeLane nextEnvelopeLane(EnvelopeLane open,
                                                       EnvelopeLane clicked);
    void setEnvelopeLane(EnvelopeLane lane);
    void closeEnvelopeLanes();
    [[nodiscard]] EnvelopeLane openEnvelopeLane() const { return envelopeLane; }

    [[nodiscard]] static std::vector<int> stretchAlgorithmItemsFor(int pitchAlgorithmItemId);

    // What Delete removes.  The piano roll deletes notes and knows nothing of
    // clips, so an imported clip -- which arrives with no notes at all until
    // its analysis finishes -- could not be removed by keyboard: the key was
    // simply not handled, and the material stayed with its waveform showing.
    //
    // Notes first, since a selection of them is the more specific thing to
    // have asked for.  The same order Ctrl+C already uses.
    enum class DeleteTarget { nothing, notes, clip };
    [[nodiscard]] static DeleteTarget deleteTargetFor(bool notesSelected, bool clipSelected);
    void deleteSelectedClip();

    // What the roll draws over the notes, behind one dropdown.  Both switches
    // are looked at while tuning but flipped rarely, so a pair of permanent
    // buttons in the tool row cost more room than they earned.
    struct ViewOptions
    {
        bool noteRange = true;   // the orange box around each note's real extent
        bool envelope = false;   // the note's own amplitude shape, drawn on it
        // The peaks of what was actually synthesised, per note.  Off unless
        // asked for: they exist only for notes that have been rendered and
        // not touched since, so most of the time there is nothing to draw.
        bool utauWaveform = false;
        bool nativeRenderedWaveform = false;
        bool nativeWaveform = true; // warped source preview, available before synthesis
        // The line the note is sung along.  On by default: without it the
        // roll shows where notes are but not what they do.
        bool pitchLine = true;
        bool originalPitchLine = true; // unedited source pitch, drawn as a dashed reference
        bool consonantPitchDashed = false; // optional source-aperiodicity evidence on the target line
        bool lyrics = true;
        bool phonemeInfo = false;
    };
    // The settings keys are the ones the two buttons already used, so a
    // project opened after this change still shows what it showed before.
    // Pure, and public so a check reads the same names the application writes.
    [[nodiscard]] static ViewOptions viewOptionsFrom(const juce::PropertySet& properties);
    static void storeViewOptions(juce::PropertySet& properties, const ViewOptions& options);
    // Menu ids are positions in one list, so the menu that is shown and the
    // answer to "what was clicked" cannot drift apart.
    [[nodiscard]] static ViewOptions afterViewMenuChoice(ViewOptions options, int chosen, bool utauEditorActive = true);
    // Semantic track capabilities, shared by menu presentation and invocation.
    // Boolean overloads retain the legacy Hachi/UTAU test/API mapping.
    [[nodiscard]] static ViewOptions afterViewMenuChoice(ViewOptions options, int chosen, TrackType type);
    [[nodiscard]] static bool viewMenuItemEnabled(int chosen, TrackType type);
    [[nodiscard]] static bool viewMenuItemEnabled(int chosen, bool utauEditorActive);
    [[nodiscard]] TrackType selectedEditorTrackType() const;
    bool diagnosticEngineViewCapabilities(const juce::File& output);
    bool diagnosticClipLength(const juce::File& output);
    static constexpr int trackPitchScopeMenuItem = 612; // 610/611 belong to track creation.



    // The menu the space around the tracks offers, and the one place a track
    // is made from a menu -- the Track menu's own items come through here too.
    // What the draw tool offers on its own button: the step a drawn note
    // grows by, which is otherwise invisible and unreachable.
    juce::PopupMenu pitchPointModeMenu() const;
    void selectPitchPointMode(int chosen);
    void showPitchPointModeMenu(juce::Point<int> screenPosition);
    juce::PopupMenu flagEditModeMenu() const;
    juce::PopupMenu autoCrossfadeModeMenu() const;
    void selectAutoCrossfadeMode(int chosen);
    void showFlagEditModeMenu(juce::Point<int> screenPosition);
    void selectFlagEditMode(int chosen);
    void showDrawSettingsMenu(juce::Point<int> screenPosition);
    void setDrawLengthDivision(int division);
    void showTrackAreaMenu(juce::Point<int> screenPosition, std::optional<TimelineComponent::Anchor> anchor = {});
    // Its items as they stand, and what choosing one does.
    [[nodiscard]] juce::PopupMenu trackAreaMenu(std::optional<TimelineComponent::Anchor> anchor = {});
    void trackAreaMenuItemChosen(int chosen, std::optional<TimelineComponent::Anchor> anchor = {});
    void addTrackFromMenu(TrackType type);
    void addTrackFromMenu(bool compose); // compatibility entry; delegates typed Hachi or existing audio action
    void refreshTrackEditingInterface(const TrackData* track);
    void refreshTrackPitchItems(const TrackData* track);
    juce::String trackInterfaceOwner;
    TrackType displayedTrackType = TrackType::hachi;
    bool projectedTrackEditable = false;
    bool updatingTrackInterface = false;
    void addReferenceTrackFromMenu();
    void addAccompanimentTrackFromMenu();
    bool selectedTrackIsAccompaniment() const;
    // 导入 MIDI 轨道: a file, then -- when it has more than one track with
    // notes on it -- which of them, then that one as a new track. An explicit
    // target instead adds a region there, always confirming the MIDI part.
    void importMidiTrack(const juce::String& targetTrackId = {}, double startSeconds = 0.0);
    void importMidiTrackFrom(const juce::File& file, const juce::String& targetTrackId = {}, double startSeconds = 0.0);
    void addMidiTrackFrom(const juce::File& file, int trackIndex, const juce::String& targetTrackId = {}, double startSeconds = 0.0);
    // The one way the engine is told about the project.  It carries the track
    // being worked on with it, because a material track's audibility depends
    // on that and the engine decides it while syncing -- sending one without
    // the other would leave the two disagreeing.
    void syncAudio(const ProjectData& data);
    void rememberDiffSingerParameters();
    std::shared_ptr<const std::vector<UtauNoteWaveform>> rememberedParameterWaveforms;
    juce::String auditionTrackAtLastSync;
    void deleteSelectedTrack();

    // Which track the selection moves to once the one at this index is gone:
    // the one that takes its place, or the one before it when it was last,
    // and none at all when it was the only one.  Left where it was, the
    // selection names a track that no longer exists and every item that acts
    // on "the selected track" stays enabled and does nothing.  Pure, and
    // public so a check reads the same rule the deletion uses.
    [[nodiscard]] static juce::String selectionAfterRemoving(
        const std::vector<TrackData>& tracks, const juce::String& removedId);

    [[nodiscard]] static std::optional<double> viewSecondsLeavingSourceEdit(
        bool wasEnabled, bool nowEnabled, double timelineSeconds);
    [[nodiscard]] static std::optional<int> followViewPosition(
        int viewLeft, int viewWidth, int playheadX, int leftMargin,
        const std::optional<juce::Range<int>>& run);
    // The tracks an export writes, in project order, with the file each goes
    // to.  A track that would not sound -- muted, or unsoloed while something
    // else is soloed -- is left out rather than written as silence, and so is
    // one with no clips.  Naming: the chosen file itself for a single track,
    // "<stem> - <track>.wav" beside it when every track is being written.
    // Pure, so the rule can be checked without a window.
    // Where the export chooser opens.  The folder last exported to, if it is
    // still there; Documents the first time, or if that folder has since been
    // moved or removed.  Pure, so the rule can be checked without a window.
    [[nodiscard]] static juce::File exportStartFile(const juce::File& remembered,
                                                    const juce::String& suggestedName);
    struct ExportTarget { juce::String trackId, trackName; juce::File file; };
    [[nodiscard]] static std::vector<ExportTarget> exportTargets(
        const ProjectData& project, const juce::File& destination,
        const juce::String& onlyTrackId, const juce::String& untitledName);
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void openExternalFile(const juce::File& file);
    void requestClose(std::function<void()> approved);
    void startProjectRecovery();
    bool keyPressed(const juce::KeyPress& key) override;
    bool keyStateChanged(bool isKeyDown) override;
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    bool isInterestedInDragSource(const juce::DragAndDropTarget::SourceDetails&) override;
    void itemDropped(const juce::DragAndDropTarget::SourceDetails&) override;
    bool shouldDropFilesWhenDraggedExternally(const juce::DragAndDropTarget::SourceDetails&,
                                             juce::StringArray&, bool&) override;
    // Which tool is in hand, and what a layout pass does to it.  A rule in
    // resized() used to take the point tool away again, so no pure function
    // could have caught it: a check has to press the button on a real window
    // and then let the window lay itself out, as every selection does.
    void diagnosticPressTool(PianoRollComponent::Tool wanted);
    // Edit > 汉字转拼音.
    static constexpr int hanziToPinyinMenuItem = 52;
    // The edit menu as it is built right now: whether it offers an item, and
    // if it does, whether the item is enabled.
    [[nodiscard]] std::optional<bool> diagnosticEditMenuItemEnabled(int itemId);
    void diagnosticChooseMenuItem(int itemId);
    void diagnosticSelectTrack(const juce::String& trackId);
    // The track area's 导入 MIDI 轨道.
    static constexpr int importMidiTrackMenuItem = 4;
    static constexpr int accompanimentTrackMenuItem = 610;
    static constexpr int emptyTuningClipMenuItem = 611;
    struct MenuItemState
    {
        int id = 0;
        juce::String text;
        bool enabled = false;
    };
    [[nodiscard]] std::vector<MenuItemState> diagnosticTrackAreaMenu();
    [[nodiscard]] juce::String diagnosticSelectedTrack() const { return selectedTrackId; }
    void diagnosticSelectNotes(const std::vector<juce::String>& ids) { pianoRoll.setSelectedNoteIds(ids); }
    [[nodiscard]] juce::String diagnosticStatusText() const { return statusLabel.getText(); }
    [[nodiscard]] ProjectModel& diagnosticProject() { return project; }
    void diagnosticAssetAudio(const juce::File& output, const juce::File& source,
                              std::function<void(bool)> completed);
    void diagnosticOtoAudio(const juce::File& output, std::function<void(bool)> completed);
    [[nodiscard]] PianoRollComponent& diagnosticPianoRoll() { return pianoRoll; }
    [[nodiscard]] TimelineComponent& diagnosticTimeline() { return timeline; }
    [[nodiscard]] PianoRollComponent::Tool diagnosticTool() const;
    void diagnosticRefreshControls();
    [[nodiscard]] juce::ToggleButton& diagnosticNativeAudioOverlapButton() { return nativeAudioOverlapButton; }
    bool diagnosticDiffSingerExpressionLane();
    void diagnosticGenerateDiffSingerParameters() { generateDiffSingerParameters(); }
    void diagnosticShowDiffSingerParameterLane() { showDiffSingerParameterLane(); }
    void diagnosticLoadDiffSingerBank(const juce::String& trackId, const juce::File& directory);
    bool diagnosticPitchPersistence(const juce::File& original, const juce::File& output, bool reopenOnly);
    bool diagnosticPitchRestore(const juce::File& output);
    bool diagnosticPitchOffset(const juce::File& output);
    bool diagnosticAccompaniment(const juce::File& output);
    void diagnosticAccompanimentTempo(const juce::File& output, const juce::File& audioFile,
                                      std::function<void(bool)> completed);
    void diagnosticWavExport(const juce::File& output, std::function<void(bool)> done);
    void diagnosticParameterRetake(const juce::File& output, const juce::File& bank, std::function<void(bool)> done);
    bool diagnosticContinuousFlag(const juce::File& output);
    bool diagnosticClassicFlagLane(const juce::File& output);
    // The real preset picker, its entries, and neighboring toolbar bounds.
    struct EnvelopePresetLayout
    {
        juce::StringArray items;
        juce::Rectangle<int> picker;
        juce::Rectangle<int> viewMenu;
        int rightControlsStart = 0;
    };
    [[nodiscard]] EnvelopePresetLayout diagnosticEnvelopePresetLayout(juce::Image* preview = nullptr);
    // The texts refreshed again, as closing the settings does.
    void diagnosticRefreshTexts();
    bool diagnosticHasEditorViewItems();
    bool diagnosticSingleEntryUi(const juce::File& output);
    bool diagnosticTrackEditingInterfacesUi(const juce::File& output);
    bool diagnosticUtauSelectionScope(const juce::File& output);
    bool diagnosticDiffSingerHistoryPlayback(const juce::File& bank, const juce::File& output);
    bool diagnosticNativeWorkflow(const juce::File& output, const juce::File& nsfModel);
    bool diagnosticMaterialSourceReplacementUi(const juce::File& output, const juce::File& nsfModel);
    bool diagnosticMaterialVariantHandoffUi(const juce::File& output);
    bool diagnosticProjectRegistrationUi(const juce::File& output);
    [[nodiscard]] bool diagnosticRenderOrderPicker();
    [[nodiscard]] bool diagnosticIntegratedLayout(const juce::File& directory);
    [[nodiscard]] bool diagnosticScrollZoom(const juce::File& directory);
    [[nodiscard]] bool diagnosticClipSplit(const juce::File& directory);
    [[nodiscard]] bool diagnosticClipTrim(const juce::File& directory);
    [[nodiscard]] bool diagnosticTimelineSelection(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeLoudness(const juce::File& directory);
    [[nodiscard]] bool diagnosticClipMerge(const juce::File& directory);
    [[nodiscard]] bool diagnosticClipGainKnob(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeRenderedWaveform(const juce::File& directory, const juce::File& modelDirectory = {});
    [[nodiscard]] bool diagnosticNativeWaveformPreview(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeNoteMove(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeNoteCopyPaste(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeSourcePitch(const juce::File& directory,
                                                   const juce::File& models);
    [[nodiscard]] bool diagnosticTimelineNotes(const juce::File& directory);
    [[nodiscard]] bool diagnosticTrackGainEnvelope(const juce::File& directory);
    [[nodiscard]] bool diagnosticAdvancedEnvelope(const juce::File& directory);
    [[nodiscard]] bool diagnosticModelessOto(const juce::File& directory);
    [[nodiscard]] bool diagnosticPianoTransport(const juce::File& directory, const juce::File& sourceProject = {});
    [[nodiscard]] bool diagnosticTrackReorder(const juce::File& directory);
    [[nodiscard]] bool diagnosticDecibelInput(const juce::File& directory);
    [[nodiscard]] bool diagnosticOtoContinuity(const juce::File& directory,const juce::File& profileBank = {});
    [[nodiscard]] bool diagnosticAdvancedEnvelopePanel(const juce::File& directory);
    [[nodiscard]] bool diagnosticNativeEnvelope(const juce::File& directory);
    [[nodiscard]] bool diagnosticEmptyTuningClip(const juce::File& directory);
    [[nodiscard]] bool diagnosticClipEditorScope(const juce::File& directory);
    [[nodiscard]] bool diagnosticNoteHints(const juce::File& directory);
    [[nodiscard]] bool diagnosticCrossRegionEditing(const juce::File& directory);
    [[nodiscard]] bool diagnosticHamoodPersistence(const juce::File& directory);
    [[nodiscard]] bool diagnosticProjectSafety(const juce::File& directory);
    [[nodiscard]] bool diagnosticNormalDisplay(const juce::File& directory);
    [[nodiscard]] bool diagnosticIndependentZoom();
    [[nodiscard]] bool diagnosticNsfPicker();
    [[nodiscard]] bool diagnosticVoicebankSelection(const juce::File& output, const juce::File& bank);
    void diagnosticNativeVibrato(const juce::File& folder, std::function<void(bool)> finished);
    [[nodiscard]] bool diagnosticOutputEngine(const juce::File& folder, const juce::File& modelFolder, const juce::File& wavtool);
    [[nodiscard]] bool diagnosticChineseCvvc(const juce::File& folder, const juce::File& modelFolder);

private:
    [[nodiscard]] juce::PopupMenu clipContextMenu(const juce::String& clipId, double seconds);
    void clipContextMenuItemChosen(int result, const juce::String& clipId, double seconds);
    std::unique_ptr<juce::AlertWindow> makeClipLengthDialog(const juce::String& clipId);
    bool applyClipLengthText(const juce::String& clipId, const juce::String& text);
    bool applyClipEditDialog(const juce::String& clipId, const juce::AlertWindow&, juce::String& error);
    void showClipLengthDialog(const juce::String& clipId);
    void showAccompanimentTempo(const juce::String& clipId);
    static constexpr int recogniseTempoMenuItem = 6;
    void addOutputEngineMenu(juce::PopupMenu& menu, const juce::String& trackId);
    juce::PopupMenu phonemizerMenu(const TrackData& track);
    void showPhonemizerMenu();
    bool outputEngineItemChosen(int result, const juce::String& trackId);
    void showOutputEngineMenu(const juce::String& trackId, juce::Point<int> at);
    void showTrackResamplerSettings(const juce::String& trackId);
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void timerCallback() override;
    void updateNoteDanceKeys();
    bool noteDanceF8WasDown = false;
    struct ScrollZoomAnchor
    {
        double zoom = 1.0, anchor = 0.0, pixelOffset = 0.0;
        bool leading = false;
    } scrollZoomAnchor, timelineScrollZoomAnchor;
    juce::StringArray getMenuBarNames() override;
    juce::PopupMenu getMenuForIndex(int topLevelMenuIndex,
                                    const juce::String& menuName) override;
    void menuItemSelected(int menuItemID, int topLevelMenuIndex) override;
    void refreshTexts();
    void refreshProjectControls();
    void refreshSelectedNoteParameter();
    void applySelectedNoteParameter();
    [[nodiscard]] bool selectedTrackIsUtau() const;
    void adjustHorizontalZoom(double factor, bool upperPane = false);
    void adjustVerticalZoom(double factor, bool upperPane = false);
    double timelineHorizontalZoom = 140.0;
    double timelineVerticalZoom = 1.0;
    void refreshStretchAlgorithmItems(int preferredId = 0);
    void chooseUtauVoicebank();
    bool selectVoicebank(const juce::String& trackId, const juce::File& directory);
    void showVoicebankSettings();
    // Crossfades the shared boundaries of the selected adjacent notes.
    void spliceSelectedNotes();
    void refreshSpliceButton();
    // Open the oto / four-region editor straight on the entry a note uses,
    // without going through the voicebank list.  Empty noteId uses the
    // current selection.
    void showRegionEditorForNote(const juce::String& noteId = {});
    // 单独OTO编辑: the same editor, working on one note's own copy of its entry.
    void showNoteOtoEditorForNote(const juce::String& noteId);
    // 播放原音 in either OTO window: this window's output device, with the
    // song stopped before the recording starts.
    void attachOtoPlayback(OtoWaveformEditorComponent& editor);
    void stopOtoPreviews();
    void startPreparedPlayback();
    void refreshAfterVoicebankChange();
    std::uint64_t seenVoicebankFilesRevision = SampleSettings::voicebankFilesRevision();
    ModelessWindows voicebankWindows, noteOtoWindows, tempoWindows, settingsWindows;
    std::vector<juce::Component::SafePointer<OtoWaveformEditorComponent>> otoPreviews;
    void noteLabelCommitted(const juce::String& noteId);
    void commitNoteAlias();
    void commitNoteFlags();
    void commitNoteConsonantVelocity();
    void setToolButton(juce::Button& selected);
    void togglePlayback();
    void armSelectionPlayback();
    void setTracksCollapsed(bool collapsed);
    void refreshCollapseIcon();
    void newProject();
    void openProject();
    void replaceImportedProject(ProjectData replacement);
    bool loadProjectFile(const juce::File& file, bool recovering = false, bool offerRegistration = true);
    void showRecoveryDialog(bool automatic = false);
    void discardProjectRecovery();
    bool restoreProjectRecovery(const ProjectRecovery::Entry& entry);
    void saveProject(std::function<void(bool)> completion = {});
    void saveProjectAs(std::function<void(bool)> completion = {});
    bool saveProjectTo(const juce::File& file);
    void performWithUnsavedCheck(std::function<void()> action);
    void addRecentProject(const juce::File& file);
    void restoreRecentProjects();
    void exportMixdown();
    void requestTrackExport(const juce::String& trackId);
    void chooseExportDestination(const juce::String& trackId, WavExportOptions options,
                                 const std::vector<juce::String>& scopeNoteIds = {},
                                 juce::Range<double> range = {});
    void exportLastRender();
    void rememberExportDirectory(const juce::File& directory);
    juce::File lastExportDirectory;
    void showWavExportOptions(std::function<void(WavExportOptions)> confirmed,
                              const juce::String& trackId = {},
                              const std::vector<juce::String>& scopeNoteIds = {},
                              juce::Range<double> range = {});
    void startExport(std::vector<ExportTarget> targets,
                     const std::vector<juce::String>& scopeNoteIds,
                     juce::Range<double> range, WavExportOptions options);
    void finishExport();
    ProjectData projectForExport(const ProjectData& data, const std::vector<ExportTarget>& targets,
                                 juce::Range<double> range) const;
    // The marquee that was last set going.  Kept from the moment play starts,
    // so pausing partway through does not shorten it.
    std::vector<juce::String> lastRenderedNoteIds;
    juce::Range<double> lastRenderedSpan;
    void importAudio();
    juce::String addAnalysedAudioFile(const juce::File& file, double durationSeconds,
                              double startSeconds = 0.0,
                              const juce::String& targetTrackId = {},
                              const VoicebankOtoEntry* oto = nullptr,
                              UtauMode otoMode = UtauMode::classic);
    std::pair<double, juce::String> audioDropTarget(int x, int y) const;
    void importAssetAudio(const juce::File& file, const juce::File& root, int otoKind,
                          double startSeconds, const juce::String& trackId,
                          juce::Point<int> menuScreenPosition, const juce::String& audioVariantId = {});
    enum class MaterialInputAction { insert, replace };
    struct MaterialInputSession;
    void showMaterialPronunciationInput(MaterialInputAction action = MaterialInputAction::insert,
        const juce::String& targetNoteId = {}, const juce::String& initialQuery = {});
    void presentMaterialPronunciationInput(const std::shared_ptr<MaterialInputSession>&);
    void chooseMaterialPronunciation(const std::shared_ptr<MaterialInputSession>&, const MaterialPronunciationCandidate&);
    void finishMaterialPronunciationInput(bool accepted, const juce::String& error = {});
    void dismissMaterialPronunciationInput();
    void componentVisibilityChanged(juce::Component&) override;
    void componentBeingDeleted(juce::Component&) override;
    juce::Component::SafePointer<juce::CallOutBox> materialPronunciationPopup;
    juce::Component* materialPronunciationObserved = nullptr;
    std::shared_ptr<MaterialInputSession> materialInputSession;
    std::uint64_t materialInputSerial = 0;
    std::set<juce::String> materialAliasResumes;
    void openSelectedMaterialEditor(bool quick, const juce::String& noteId = {});
    void openMaterialEditor(const juce::File& source, const juce::File& manifest, bool quick,
                            const juce::String& pronunciationId = {}, const juce::String& audioVariantId = {},
                            std::optional<juce::Range<double>> sourceRange = {});
    bool preflightMaterialPublication(const juce::File&, const std::vector<SampleRegionSetting>&, juce::String&);
    void materialPublicationCompleted(const juce::File&, const std::vector<SampleRegionSetting>&);
    void importMaterialPronunciations(const juce::File& file, const juce::File& manifest,
        const juce::StringArray& regionIds, double startSeconds, const juce::String& trackId,
        const juce::String& chosenAlias = {}, const juce::String& audioVariantId = {});
    struct NativeAnalysisTask;
    void scheduleAnalysis(const juce::File& file, const juce::String& clipId,
        std::optional<ProjectModel::MaterialSourceBindingTicket> sourceTicket = std::nullopt, bool explicitRetry = false);
    bool originalSourceAnalysisPending(const juce::String& parentClipId, const juce::File& source) const;
    void scheduleNativeTrimSourceAnalysis(const ProjectData& data);
    void importMidi();
    void exportMidi();
    void exportUst();
    // A UTAU project file, which arrives as a plain UTAU track.
    void importUst();
    void loadUstFile(const juce::File& file);
    void importUstFile(const juce::File& file, ProjectModel::UstImportMode mode, int encoding = 0,
                       TrackType destinationType = TrackType::hachi);
    void importMelodyne();
    void showSettings(int page = 0);
    void applyPreferences();
    void applyUiScale();
    void loadSampleSettings();
    void refreshSampleEditors();
    void commitSampleEditors();
    void saveSampleSettings();
    void importOto();
    void exportOto();
    void showAssetManager();
    void setAssetManagerVisible(bool visible);
    void showClipGainDialog();
    void showTrackGainDialog(const juce::String& trackId);
    void showTrackPanDialog(const juce::String& trackId);
    void showRenameTrackDialog();
    void showHamood();
    void showTransposeNotesDialog();
    void showSetNotesPitchDialog();
    void copySelectedNotes(bool cut);
    // Nothing said: back at the moment it was copied from.  A time given:
    // there instead.
    void pasteCopiedNotes(std::optional<double> atSeconds = {});
    // Which clip a paste at this moment lands in, on the track in front.
    [[nodiscard]] juce::String pasteTargetTrack() const;
    [[nodiscard]] juce::String pasteTargetClipUnused(const juce::String& trackId,
                                               double atSeconds) const;
    void pasteCopiedNotesInto(const juce::String& clipId, double atSeconds,
                              const std::vector<juce::String>& replacing);
    void copySelectedClip();
    void pasteCopiedClip();
    void duplicateSelectedClip();
    void confirmDestructive(const juce::String& title, std::function<void()> action);
    void loadMelodyneFile(const juce::File& file);
    void presentMelodyneComposeSelection(backend::MelodyneImportResult imported);
    struct ProjectRegistrationOperation;
    std::shared_ptr<ProjectRegistrationOperation> projectRegistrationOperation;
    std::unique_ptr<juce::FileChooser> projectRegistrationChooser;
    std::vector<juce::File> projectRegistrationRetryRoots;
    juce::String projectRegistrationTrace;
    // Diagnostic host seams drive the SAME folder-result/prefs handlers; never
    // bypass the confirmation, planner, owned lease or model commit.
    std::function<void(std::function<void(const juce::File&)>)> projectRegistrationFolderPickerForTest;
    std::function<bool(const juce::File&,juce::String&)> projectRegistrationRootWriterForTest;
    void cancelProjectMaterialRegistration();
    void offerMaterialFolderForImport(const ProjectData& imported);
    void chooseProjectMaterialRegistrationDestination(const std::shared_ptr<ProjectRegistrationOperation>&);
    void stageProjectMaterialRegistration(const std::shared_ptr<ProjectRegistrationOperation>&, const juce::File&);
    void publishProjectMaterialRegistration(const std::shared_ptr<ProjectRegistrationOperation>&);
    bool registerRetainedMaterialRoot(const juce::File&, juce::String& error);
    void retryProjectMaterialRootRegistration();
    void focusClip(const juce::String& clipId);
    void focusNote(const juce::String& noteId);
    void updateUtauRenderSelection();
    void setSourceEditMode(bool enabled);
    void showError(const juce::String& message);

    HachiLookAndFeel lookAndFeel;
    I18n strings;
    ProjectModel project;
    AudioEngine audio;
    juce::TooltipWindow tooltipWindow;

    juce::MenuBarComponent menuBar;
    juce::Label bpmCaption;
    juce::Label bpmEditor;
    juce::Label beatsCaption;
    juce::Label beatsEditor;
    juce::Label denominatorLabel;
    juce::Label gridCaption;
    juce::ComboBox gridSelector;
    juce::Label stretchCaption;
    juce::ComboBox stretchSelector;
    juce::Label scaleCaption;
    juce::ComboBox scaleSelector;
    juce::TextButton collapseTracksButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::TextButton noteEditButton;
    juce::TextButton wrenchButton;
    ToolButton drawButton;
    juce::TextButton lineButton;
    ToolButton pointButton;
    juce::TextButton trimButton;
    juce::TextButton nativeLoudnessButton;
    juce::TextButton connectButton;
    juce::TextButton pitchParamButton;
    juce::TextButton driftParamButton;
    juce::TextButton attackParamButton;
    juce::TextButton breathParamButton;
    juce::TextButton tensionParamButton;
    juce::TextButton formantParamButton;
    ToolButton volumeParamButton;
    // Per-frame flags for the selected notes, and the lane that draws the
    // curve.  The lane is only reachable once the notes are switched over:
    // with flags held as a single number there is no curve to show.
    ToolButton flagCurveButton, flagEnvelopeButton;
    juce::TextButton advancedEnvelopeButton;
    void showAdvancedEnvelopeMenu();
    DropdownButton phonemizerButton;
    juce::ToggleButton pianoPreviewButton;
    juce::ToggleButton nativeAudioOverlapButton;
    juce::ToggleButton nsfPitchTransitionsButton;
    juce::ToggleButton nsfNoiseProtectionButton;
    juce::TextButton nativeVibratoButton;
    ViewOptions viewOptions;
    void appendEditorViewItems(juce::PopupMenu& menu);
    void refreshPitchScopeTitle();
    void applyEditorViewChoice(int chosen);
    void applyViewOptions();
    juce::ComboBox envelopePresetMenu;
    juce::ToggleButton robustPitchCurveButton;
    juce::ComboBox pitchAlgorithm;
    juce::ComboBox stretchAlgorithm;
    // Only the neural decoder can render a phrase in one pass, so this picker
    // is shown only when that decoder is the one selected.
    juce::ComboBox renderOrder;
    juce::Label pitchLabel;
    juce::Label stretchLabel;
    juce::Label renderOrderLabel;
    juce::Label statusLabel;
    juce::Label selectionSummaryLabel;
    void refreshSelectionSummary();
    std::vector<juce::String> selectionSummaryIds;
    std::uint64_t selectionSummaryRevision=std::numeric_limits<std::uint64_t>::max();
    juce::Label sourceEditHint;
    juce::ComboBox sampleRegionSelector;
    juce::TextEditor sampleAliasEditor;
    juce::TextEditor sampleStartEditor;
    juce::TextEditor sampleEndEditor;
    juce::TextEditor sampleAlignmentEditor;
    juce::TextEditor sampleFixedEditor;
    juce::Label sampleAliasLabel, sampleStartLabel, sampleEndLabel,
                sampleAlignmentLabel, sampleFixedLabel;
    juce::TextButton sampleSaveButton, otoImportButton, otoExportButton;
    juce::Label noteAliasLabel, noteConsonantVelocityLabel, noteFlagsLabel;
    juce::TextButton spliceButton, utauVoicebankButton;
    juce::Label utauVoicebankName;
    ToolButton autoCrossfadeButton;
    bool autoCrossfadeCurve = false;
    juce::TextEditor noteAliasEditor, noteConsonantVelocityEditor, noteFlagsEditor;
    bool noteConsonantVelocityMixed = false;
    bool noteConsonantVelocityDirty = false;
    bool noteFlagsMixed = false;
    bool noteFlagsDirty = false;
    juce::Label parameterTitle;
    juce::Label trackEditingHeading, diffSingerContextLabel;
    juce::TextButton diffSingerModelButton, diffSingerTimingButton, diffSingerPronunciationButton,
                     diffSingerPitchButton, diffSingerParametersButton, diffSingerSettingsButton;
    juce::Label smoothCaption;
    juce::Slider smoothSlider;
    juce::Slider zoomSlider;
    juce::Slider vZoomSlider;
    bool showWaveforms = true;
    double progress = 0.0;
    juce::ProgressBar progressBar;
    bool importInProgress = false;
    bool closingUI = false;
    // An export waits for every note in the song to render before it writes,
    // and the wait is done by the timer rather than by blocking the message
    // thread: a whole song of UTAU notes can take minutes.
    std::vector<ExportTarget> pendingExport;
    juce::Range<double> pendingExportRange;
    WavExportOptions pendingExportOptions;
    bool exportWaitingForRender = false;
    bool showingRenderProgress = false;
    bool playWhenRenderReady = false;
    int activeUtauSelectionCount = 0;
    int pendingNativeAnalyses = 0;
    std::map<juce::String, std::shared_ptr<NativeAnalysisTask>> nativeAnalysisTasks;
    std::map<juce::String, int> nativeAnalysisFailures;
    std::set<juce::String> nativeAnalysisRetryScheduled;
    double nativeAnalysisProgress = 0.0;
    juce::String nativeAnalysisName;
    juce::String lastNativeAnalysisBackend; // actual completed service, including cache-warming stale jobs
    juce::String lastNativeAnalysisTrace; // resolved backend/model paths + actual worker warnings/errors

    TrackListComponent trackList;
    TimelineComponent timeline;
    PianoRollComponent pianoRoll;
    EditorViewport trackViewport;
    EditorViewport timelineViewport;
    EditorViewport pianoViewport;
    juce::Component panelSplitter;
    // The material manager, docked on the right of the main window so it can be
    // operated while editing.  Hidden until its menu item is chosen; its width
    // is drag-resizable via the edge on its left and remembered.
    std::unique_ptr<AssetManagerComponent> assetManager;
    std::unique_ptr<juce::ResizableEdgeComponent> assetManagerResizer;
    // Records the width the drag handle settles on and re-lays the window, so
    // the panel keeps whatever width the user drags it to.
    struct AssetManagerConstrainer final : juce::ComponentBoundsConstrainer
    {
        std::function<void(int)> onWidth;
        void checkBounds(juce::Rectangle<int>& bounds,
                         const juce::Rectangle<int>& previous,
                         const juce::Rectangle<int>& limits,
                         bool isStretchingTop, bool isStretchingLeft,
                         bool isStretchingBottom, bool isStretchingRight) override
        {
            juce::ComponentBoundsConstrainer::checkBounds(bounds, previous, limits,
                isStretchingTop, isStretchingLeft, isStretchingBottom, isStretchingRight);
            if (onWidth) onWidth(bounds.getWidth());
        }
    };
    AssetManagerConstrainer assetManagerConstrainer;
    bool assetManagerVisible = false;
    int assetManagerWidth = 320;
    std::unique_ptr<juce::FileChooser> chooser;
    std::unique_ptr<juce::PropertiesFile> preferences;
    int lastTimelineX = 0;
    int lastPianoX = 0;
    // Arrangement folded away, leaving the whole window to the tuning editor.
    bool tracksCollapsed = false;
    // What this run of playback covers, when it was started for a selection.
    std::optional<juce::Range<double>> playbackRun;
    int lastTimelineY = 0;
    int lastTrackY = 0;
    bool syncingScroll = false;
    bool pianoInitialScrollSet = false;
    bool sourceEditActive = false;
    juce::File sampleSettingsFile;
    std::vector<SampleRegionSetting> sampleSettingsRows;
    int activeSampleSetting = 0;
    juce::String selectedClipId;
    juce::String selectedTrackId;
    juce::String selectedNoteId;
    std::unordered_set<std::string> accumulatedUtauNoteIds;
    juce::String copiedClipId;
    std::vector<NoteData> copiedNotes;
    std::vector<ClipData> copiedNativeClips;
    std::vector<NativeConnection> copiedNativeConnections;
    // Where the copied block sat on the timeline.  Pasting without saying
    // where puts it back at the same moment, on whichever track is in front.
    double copiedOriginSeconds = 0.0;
    // The track it was copied from.  Pasting back onto that same track means
    // something different from carrying it across to another one.
    juce::String copiedTrackId;
    std::unique_ptr<ProjectRecovery> projectRecovery;
    juce::String lastRecoveryError;
    juce::File currentProjectFile;
    juce::StringArray recentProjectPaths;
    std::uint64_t savedProjectRevision = 0;
    enum class ParameterMode { pitchSmooth, pitchDrift, attackSpeed, breath, tension, formant, volume };
    ParameterMode parameterMode = ParameterMode::pitchSmooth;
    bool updatingSmoothSlider = false;
    bool updatingRobustPitchCurve = false;
    bool smoothSliderDragging = false;
    // Both UTAU selector items -- plain (7) and four-region (8) -- must light
    // up every UTAU affordance.  Testing the id against 7 alone silently
    // hides the voicebank bar, its settings button, the point tool and the
    // amplitude envelope from the four-region mode.
    bool isUtauAlgorithmSelected() const;
    void loadDiffSingerVoicebank(const juce::String& trackId, const juce::File& directory);
    void generateDiffSingerParameters(bool wholeTrack = false, const juce::String& noteId = {});
    void showDiffSingerParameterLane();
    void generateDiffSingerPitch(bool wholeTrack = false, const juce::String& noteId = {});
    void showDiffSingerPronunciationEditor(const juce::String& noteId = {});
    void showDiffSingerPhonemeEditor(const juce::String& trackId = {}, const juce::String& noteId = {});
    void showDiffSingerSettings(const TrackData& track);
    bool refreshDiffSingerFlagContext();
    std::map<juce::String, juce::var> diffSingerCapabilities;
    juce::String diffSingerParameterLanePendingBank;
    std::set<juce::String> diffSingerCapabilityPending;
    std::shared_ptr<std::atomic<bool>> diffSingerCapabilityCancel = std::make_shared<std::atomic<bool>>(false);
    bool diffSingerBusy = false;
    double diffSingerStatusUntil = 0.0;
    std::shared_ptr<std::atomic<bool>> diffSingerCancel = std::make_shared<std::atomic<bool>>(false);
    // Kept in step with envelopeLane, which is what opens and closes it.
    bool utauAmplitudeEnvelopeActive = false;
    EnvelopeLane envelopeLane = EnvelopeLane::none;
    bool draggingPanelSplitter = false;
    int panelSplitterDragScreenY = 0;
    float panelSplitterDragRatio = 0.60f;
    float panelSplitRatio = 0.60f;
};
}
