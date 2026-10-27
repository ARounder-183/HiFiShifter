#pragma once

#include "SampleSettings.h"
#include "MaterialAnnotationSession.h"
#include "Theme.h"
#include "I18n.h"
#include "backend/OtoAudioAnalysis.h"
#include "backend/UtauRenderer.h"
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <array>
#include <functional>
#include <memory>
#include <optional>

namespace hachi
{
class MaterialAnnotationSession;
class MaterialAnnotationEditor;
class OtoWaveformEditorComponent final : public juce::Component,
                                         private juce::ScrollBar::Listener,
                                         private juce::Timer
{
public:
    // jieMode adds the three inner boundaries that turn a classic two-region
    // oto entry into the four-region (Jie) layout.  Its classic timing fields
    // are stored in oto.jie.ini and its extra boundaries in oto4.ini; the
    // voicebank's original oto.ini is never modified in Jie mode.
    // mouMode is jieMode plus the per-region classes: the same four
    // boundaries, written to otomou.ini with a class string in front.
    OtoWaveformEditorComponent(VoicebankOtoEntry entry, bool jieMode,
                               bool mouMode,
                               std::function<void()> savedCallback);
    ~OtoWaveformEditorComponent() override;
    // Source OTO editing is mode-specific, and opens one resolved phoneme
    // directly. It must not migrate the bank into native OTTO annotations.
    [[nodiscard]] static std::unique_ptr<OtoWaveformEditorComponent> forVoicebankSample(
        const juce::File& root, UtauMode mode, const backend::UtauRenderer::ResolvedSample& sample,
        std::function<void()> onSaved, juce::String& error);
    // The single material editor used by the manager and quick wrench. All
    // display modes share one native draft and save only to otto.ini.
    [[nodiscard]] static std::unique_ptr<OtoWaveformEditorComponent> forMaterial(
        const juce::File& audio, const juce::File& manifest, I18n& strings,
        std::function<void(const std::vector<SampleRegionSetting>&)> onSaved,
        bool quick = false, juce::String* error = nullptr,
        const juce::String& selectedAudioId = {});
    using EditScope = MaterialAnnotationSession::EditScope;
    using NoteScopeBinding = MaterialAnnotationSession::NoteScopeBinding;
    // Optional project-only destination on this SAME material editor/session.
    // Caller supplies a real model adapter; no native/UTAU persistence is guessed.
    bool enableNoteScope(NoteScopeBinding, EditScope initial, juce::String& error);
    bool refreshNoteScopeBinding(juce::String& error);
    juce::String diagnosticNoteContextId() const
    { return materialSession && materialSession->hasNoteScope() ? materialSession->noteScopeBinding().noteId : juce::String{}; }
    bool setEditScope(EditScope, juce::String& error);
    bool discardAndSetEditScope(EditScope, juce::String& error);
    bool restoreLocal(juce::String& error);
    EditScope editScope() const;
    bool diagnosticScopeDirty() const { return materialSession && materialSession->dirty(); }
    void setPublicationValidator(std::function<bool(
        const std::vector<SampleRegionSetting>&, juce::String&)> validator);
    bool selectMaterialPronunciation(const juce::String& regionId, bool detail = false);
    bool focusMaterialSourceRange(juce::Range<double> sourceRange, bool detail = true);
    void bindMaterialViewPreferences(juce::PropertiesFile&);
    bool setSingleSyllableMaterial(bool enabled);
    bool diagnosticSingleSyllableMaterial() const { return singleSyllableMaterial; }
    juce::String diagnosticMaterialRegionId() const { return materialRegionId; }
    std::optional<juce::Range<double>> diagnosticMaterialSourceSelection() const;
    juce::Range<double> diagnosticMaterialVisibleRange() const
    { const auto start = waveform.visibleStartMilliseconds() / 1000.0;
      return {start, start + waveform.visibleLengthMilliseconds() / 1000.0}; }
    [[nodiscard]] bool diagnosticMaterialOverview() const { return materialSession && !materialDetail; }
    [[nodiscard]] bool diagnosticOuterRangeLocked() const;
    bool diagnosticMaterialBack();
    [[nodiscard]] std::vector<SampleRegionSetting> diagnosticMaterialRows() const;
    void diagnosticMaterialMode(int id) { materialMode.setSelectedId(id, juce::sendNotificationSync); }
    void diagnosticDragMaterialBoundary(int index, double seconds) { waveform.diagnosticDragNativeBoundary(index, seconds); }
    bool diagnosticCreateMaterialAlias(double start, double end, const juce::String& alias, juce::String& error);
    bool diagnosticRemoveMaterialAlias(const juce::String& id, const juce::String& alias, juce::String& error);
    void diagnosticMaterialAdjacent(bool enabled);
    juce::Component* diagnosticMaterialOverviewWaveform();
    void diagnosticMaterialAliasText(const juce::String& text);
    void diagnosticCreateSelectedMaterialAlias();
    void diagnosticPronunciationKind(int id) { materialPronunciationKindChoice.setSelectedId(id, juce::sendNotificationSync); }
    void diagnosticSimplifiedConsonant(bool enabled);
    void diagnosticConsonantGap(const juce::String& milliseconds);
    void diagnosticDragMaterialAlignmentTo(double milliseconds) { waveform.diagnosticDragAlignmentTo(milliseconds); }
    void diagnosticDragMaterialConsonantTo(double milliseconds) { waveform.diagnosticDragConsonantTo(milliseconds); }
    void diagnosticBeginMaterialVowelSplit();
    bool diagnosticConfirmMaterialVowelSplit(const juce::String& leftName, const juce::String& rightName);
    void diagnosticCancelMaterialVowelSplit();
    bool diagnosticSplitMaterialVowel(const juce::String& leftName, const juce::String& rightName);
    void diagnosticSelectMaterialVowel(int ordinal) { materialSegmentChoice.setSelectedId(ordinal, juce::sendNotificationSync); }
    bool diagnosticEditMaterialVowel(const juce::String& name, const juce::String& weight);
    juce::StringArray diagnosticMaterialVowelItems() const;

    // 单独OTO编辑 for one note: the editor on that note's entry -- its own copy
    // when it already has one, so a second edit carries on from the first --
    // with Save handing the result to the note instead of to any file.  Empty,
    // with the reason in error, when the note cannot have one.  The window and
    // the checks both open it through here.
    [[nodiscard]] static std::unique_ptr<OtoWaveformEditorComponent> forNote(
        ProjectModel& project, const juce::String& noteId, juce::String& error);
    // Exact canonical renderer ticket only; no alias/file re-lookup. Local
    // timing uses the checked model commit and never persists native binding IDs.
    [[nodiscard]] static std::unique_ptr<OtoWaveformEditorComponent> forClassicNote(
        ProjectModel&, const juce::String& noteId, I18n&,
        const backend::UtauRenderer::ResolvedSample&, std::uint64_t resolvedAtRevision, juce::String& error,
        EditScope initial = EditScope::thisNote, const juce::String& selectedAudioId = {},
        const juce::String& selectedRegionId = {},
        std::function<void(const std::vector<SampleRegionSetting>&)> onSourceSaved = {});
    bool diagnosticClassicScalarScope() const
    { return materialSession && materialSession->isClassicNoteScope(); }
    bool diagnosticNativeAuthoringEnabled() const
    { return materialPronunciationKindChoice.isEnabled() && materialSplitSegment.isEnabled(); }
    [[nodiscard]] const VoicebankOtoEntry& originalEntry() const { return original; }
    // 单独OTO编辑: the edited entry goes to one note instead of into any oto
    // file.  Saving then changes nothing on disk.
    void saveToNoteInstead(std::function<void(const VoicebankOtoEntry&)> handoff);

    // 播放原音: what the play button sounds through.  The editor owns no audio
    // device; whoever opens the window hands one over, and says what has to
    // happen before a sound is made -- the song stopped, an output opened.
    struct PlaybackHost
    {
        // The device to play through, or nothing once there is none any more.
        std::function<juce::AudioDeviceManager*()> devices;
        // Asked before every start; false and nothing plays.
        std::function<bool()> beforeStart;
    };
    // Without a host the recording still plays, into nothing: that is how the
    // checks pull it and listen to what a device would have been sent.
    void stopPlaybackPreview() { stopPlayback(); }
    void setPlaybackHost(PlaybackHost host) { playbackHost = std::move(host); }

    // Test seams: press Save the way the button does, and read its caption.
    void diagnosticSave() { save(); }
    [[nodiscard]] juce::String diagnosticSaveCaption() const
    {
        return saveButton.getButtonText();
    }
    // Test seams for 播放原音: press the button the way a click does, read
    // what it says and where the playhead is, and pull the sound a device
    // would be sent -- up to maxSeconds at outputRate, in blocks, the playhead
    // moved after each block the way the window's timer moves it.  Appended to
    // out; returns how many samples came.
    bool diagnosticHasPlaybackHost() const { return bool(playbackHost.devices)&&bool(playbackHost.beforeStart); }
    void diagnosticPressPlay() { togglePlayback(); }
    [[nodiscard]] juce::String diagnosticPlayCaption() const
    {
        return playButton.getButtonText();
    }
    [[nodiscard]] bool diagnosticPreviewPlaying() const { return preview != nullptr; }
    [[nodiscard]] bool diagnosticPreviewAttached() const { return previewDevices != nullptr; }
    [[nodiscard]] std::optional<double> diagnosticPlayheadMs() const
    {
        return waveform.playheadMilliseconds();
    }
    int diagnosticPullPreview(juce::AudioBuffer<float>& out, double outputRate,
                              double maxSeconds);
    // Native-material save: when set, Save hands the edited entry to this
    // instead of writing oto.ini, so the same waveform editor persists a
    // from-scratch material as its native HJM sidecar.  Editing is one native
    // operation; OTO and HJM are just the two stores it can write to.  Return
    // true on success, or false with the reason in error.
    void setSaveOverride(std::function<bool(const VoicebankOtoEntry&, juce::String&)> handler)
    {
        saveOverride = std::move(handler);
    }

    void paint(juce::Graphics& g) override;
    void resized() override;
    void lookAndFeelChanged() override;
    // Read-only views for the offline layout check in --smoke-oto-editor.
    juce::String diagnosticParameterText(int index) const
    {
        return juce::isPositiveAndBelow(index, 8)
            ? parameterEditors[static_cast<std::size_t>(index)].getText() : juce::String{};
    }
    // Test seam: dialogs are unreachable without a mouse, so the count
    // is switched and read back the way a click would.
    void diagnosticSetRegionCount(int count) { setRegionCount(count); }
    [[nodiscard]] int diagnosticRegionCount() const { return regionCount(); }
    [[nodiscard]] juce::String diagnosticClasses() const { return edited.mouClasses; }
    void diagnosticTickConsonant(int index, bool consonant)
    {
        setRegionIsConsonant(index, consonant);
    }
    bool diagnosticConsonantTicked(int index) const
    {
        return juce::isPositiveAndBelow(index, 4)
            && consonantButtons[static_cast<std::size_t>(index)].getToggleState();
    }
    bool diagnosticConsonantEnabled(int index) const
    {
        return juce::isPositiveAndBelow(index, 4)
            && consonantButtons[static_cast<std::size_t>(index)].isEnabled();
    }
    juce::String diagnosticRegionName(int index) const;
    // Visible is not the same as present: a component with no parent still
    // reports the flag setVisible left on it, and paints nowhere.
    bool diagnosticParameterAttached(int index) const
    {
        return juce::isPositiveAndBelow(index, 8)
            && parameterEditors[static_cast<std::size_t>(index)]
                   .getParentComponent() == this
            && parameterLabels[static_cast<std::size_t>(index)]
                   .getParentComponent() == this;
    }
    juce::String diagnosticParameterLabel(int index) const
    {
        return juce::isPositiveAndBelow(index, 8)
            ? parameterLabels[static_cast<std::size_t>(index)].getText()
            : juce::String{};
    }
    bool diagnosticParameterShown(int index) const
    {
        return juce::isPositiveAndBelow(index, 8)
            && parameterEditors[static_cast<std::size_t>(index)].isVisible();
    }
    void diagnosticDragOffsetTo(double ms)
    {
        waveform.diagnosticDragOffsetTo(ms);
        refreshEditors();
    }
    void diagnosticDragBoundary(int index, double ms)
    {
        waveform.diagnosticDragBoundary(index, ms);
        refreshEditors();
    }
    juce::String diagnosticHandleAtX(float x) const
    {
        return waveform.diagnosticHandleAtX(x);
    }
    float diagnosticXForBoundary(int index) const
    {
        return waveform.diagnosticXForBoundary(index);
    }
    double diagnosticBoundaryMs(int index) const
    {
        return index == 0 ? edited.jieOnsetMs
             : index == 1 ? edited.jieGlideMs : edited.jieNucleusMs;
    }
    double diagnosticEntrySpanMs() const { return waveform.entrySpanMilliseconds(); }
    // Type into one parameter box and commit it, the way leaving the field does.
    void diagnosticTypeParameter(int index, const juce::String& text)
    {
        if (!juce::isPositiveAndBelow(index, 8)) return;
        parameterEditors[static_cast<std::size_t>(index)].setText(text, false);
        commitEditors();
    }
    // Whether every zoom caption fits the button it is drawn in, and by how
     // much.  JUCE indents button text from both edges before it fits it, so a
     // button can be wide enough for the glyphs and still show an ellipsis.
    bool diagnosticZoomCaptionsFit(juce::String* report = nullptr) const;
    void diagnosticZoomOut() { waveform.nudgeHorizontalZoom(1.0 / 1.4); }
    double diagnosticTotalMs() const { return waveform.totalMilliseconds(); }
    double diagnosticVisibleMs() const { return waveform.visibleLengthMilliseconds(); }
    juce::Rectangle<int> diagnosticScrollBounds() const { return waveformScroll.getBounds(); }
    bool diagnosticScrollEnabled() const { return waveformScroll.isEnabled(); }
    bool diagnosticAnalysisReady() { waveform.pollAnalysis(); return waveform.analysisReady(); }
    std::shared_ptr<const backend::OtoAudioAnalysisData> diagnosticAnalysis() const { return waveform.analysisData(); }
    void diagnosticSetAnalysisLayers(bool spectrum, bool f0)
    {
        spectrumButton.setToggleState(spectrum, juce::dontSendNotification);
        sourceF0Button.setToggleState(f0, juce::dontSendNotification);
        waveform.setAnalysisLayers(spectrum, f0);
    }
    juce::Rectangle<int> diagnosticSpectrumBounds() const { return waveform.analysisBounds(); }
    juce::Rectangle<int> diagnosticWaveformBounds() const { return waveform.waveBounds(); }
    float diagnosticXForTime(double ms) const { return waveform.timeX(ms); }
    void diagnosticZoomIn() { waveform.nudgeHorizontalZoom(1.4); }
    void diagnosticScrollTo(double ms) { waveform.setVisibleStartMilliseconds(ms); }
    juce::Component& diagnosticWaveformComponent() { return waveform; }
    juce::String diagnosticPitchReadout() const { return pitchReadoutHz.getText(); }
    juce::String diagnosticPitchTimeReadout() const { return pitchReadoutTime.getText(); }


private:
    class WaveformView final : public juce::Component, public juce::SettableTooltipClient,
                               private juce::Timer
    {
    public:
        WaveformView(VoicebankOtoEntry& entry, bool jieMode, bool mouMode,
                     std::function<void()> changedCallback);
        ~WaveformView() override;
        void pollAnalysis();
        bool analysisReady() const { return analysis && analysis->complete; }
        std::shared_ptr<const backend::OtoAudioAnalysisData> analysisData() const { return analysis; }
        void setAnalysisLayers(bool spectrum, bool f0);
        std::function<void(std::optional<backend::FcpeFrame>)> onPitchSelected;
        juce::Rectangle<int> analysisBounds() const;
        juce::Rectangle<int> waveBounds() const;
        float timeX(double ms) const { return xForMilliseconds(ms); }


        void paint(juce::Graphics& g) override;
        void mouseMove(const juce::MouseEvent& event) override;
        void mouseExit(const juce::MouseEvent&) override;
        void mouseDown(const juce::MouseEvent& event) override;
        void mouseDrag(const juce::MouseEvent& event) override;
        void mouseUp(const juce::MouseEvent&) override;
        double durationMilliseconds() const { return durationMs; }
        // Zoom is applied as a visible time window and an amplitude scale.
        // Nudging by a factor keeps the current centre, so repeatedly
        // pressing + closes in on whatever is on screen.
        void nudgeHorizontalZoom(double factor);
        void nudgeVerticalZoom(double factor);
        void frameEntry();          // fit the view to this oto entry
        void showWholeFile();       // the opening view: nothing off screen
        // The visible window, for the scroll bar underneath to mirror.
        double totalMilliseconds() const { return durationMs; }
        double visibleStartMilliseconds() const { return viewStartMs(); }
        double visibleLengthMilliseconds() const { return visibleSpanMs(); }
        void setVisibleStartMilliseconds(double startMs);
        std::function<void()> onViewChanged;
        // 播放原音: where the sound being played has got to in the file, drawn
        // over everything else; empty while nothing plays.
        void setPlayheadMilliseconds(std::optional<double> milliseconds)
        {
            playheadMs = milliseconds;
            repaint();
        }
        [[nodiscard]] std::optional<double> playheadMilliseconds() const { return playheadMs; }
        void diagnosticDragOffsetTo(double ms) { applyHandle(Handle::offset, ms); }
        void diagnosticDragAlignmentTo(double ms) { applyHandle(Handle::preutterance, ms); }
        void diagnosticDragConsonantTo(double ms) { applyHandle(Handle::consonant, ms); }
        // Drag one region boundary, 0 for the first, to an absolute time.
        void diagnosticDragBoundary(int index, double ms)
        {
            applyHandle(index == 0 ? Handle::jieOnset
                        : index == 1 ? Handle::jieGlide : Handle::jieNucleus, ms);
        }
        // What a press at this x would take hold of.
        juce::String diagnosticHandleAtX(float x) const;
        float diagnosticXForBoundary(int index) const
        {
            const std::array<double, 3> bounds { entry.jieOnsetMs, entry.jieGlideMs,
                                                 entry.jieNucleusMs };
            return xForMilliseconds(entry.offsetMs
                + bounds[static_cast<std::size_t>(juce::jlimit(0, 2, index))]);
        }
        // Length of the region this oto entry covers, offset to cutoff.
        double entrySpanMilliseconds() const { return endMilliseconds() - entry.offsetMs; }

    private:
        enum class Handle { none, offset, consonant, cutoff, preutterance, overlap,
                            jieOnset, jieGlide, jieNucleus };
        void loadWaveform();
        void timerCallback() override { pollAnalysis(); }
        void paintAnalysis(juce::Graphics& g);
        void paintSourceF0(juce::Graphics& g, juce::Rectangle<int> area);
        std::shared_ptr<backend::OtoAudioAnalysisRequest> analysisRequest;
        std::shared_ptr<const backend::OtoAudioAnalysisData> analysis;
        // Presentation-only recolouring of the immutable shared analysis image.
        juce::Image lightSpectrogramSource, lightSpectrogram;
        bool spectrumVisible = true, sourceF0Visible = true;
        std::optional<backend::FcpeFrame> selectedPitch;
        const backend::FcpeFrame* pitchFrameNear(juce::Point<float> position) const;

        juce::Rectangle<int> plotBounds() const;
        float xForMilliseconds(double milliseconds) const;
        double millisecondsForX(float x) const;
        double endMilliseconds() const;
        // Moves one boundary to an absolute time in the file.  Split out
        // of the drag so the offline editor check can exercise it.
        void applyHandle(Handle handle, double value);
        // How many boundaries this entry has: one fewer than its regions.
        int boundaryCount() const { return juce::jlimit(2, 4, regions) - 1; }
        // What bounds one boundary from above -- the next boundary that
        // exists, or the end of the entry for the last one.  The boundaries
        // past the count are not walls: they are values the file is keeping
        // for a different split, and a two-region entry's one boundary has to
        // be able to reach the end.
        double boundaryCeiling(int index) const;
        // Ordered, whatever the count.  A boundary the count does not reach
        // follows the last visible one rather than being crossed by it.
        void orderBoundaries();
        Handle handleNear(const juce::Point<float>& position) const;
        double millisecondsForHandle(Handle handle) const;
        struct HandleBadge
        {
            Handle handle;
            juce::String text;
            juce::Colour colour;
            juce::Rectangle<float> bounds;
        };
        std::vector<HandleBadge> handleBadges() const;
        void updateCursor(const juce::Point<float>& position);

        VoicebankOtoEntry& entry;
        bool jie;
        // 谋 names the regions by number rather than by the part of a Chinese
        // syllable they usually are: any of them can be a consonant here, so
        // 韵尾 would be saying something the entry may well contradict.
        bool mou;
    public:
        // How many regions this entry has, and what each of them is.  Four
        // and empty unless a 谋 class string says otherwise: the view then
        // draws that many bands and one boundary fewer, rather than three
        // lines two of which nothing can move.
        int regions = 4;
        juce::String classes;
        bool lockOuterRange = false;
        bool preferOuterRangeDrag = false;
        bool nativeDetail = false;
        std::vector<NativeSegment> nativeSegments;
        void setTimingMode(bool withRegions, bool withClasses) { jie = withRegions; mou = withClasses; repaint(); }
        void diagnosticDragNativeBoundary(int index, double seconds);
    private:
        int nativeBoundaryDragging = -1;
        int nativeBoundaryNear(juce::Point<float>) const;
    public:
    private:
        double viewZoom = 1.0;      // 1 = whole file visible
        double viewCentreMs = 0.0;
        double amplitudeZoom = 1.0;
        double panStartCentreMs = 0.0;
        bool panning = false;
        double visibleSpanMs() const;
        double viewStartMs() const;
        void clampView();
        std::function<void()> changed;
        juce::AudioFormatManager formats;
        std::vector<std::pair<float, float>> waveformPeaks;
        double durationMs = 0.0;
        Handle dragging = Handle::none;
        double dragGrabOffsetMs = 0.0;
        std::optional<double> playheadMs;
    };

    // Which parameters this mode shows, in the order they are laid out.  Jie
    // mode drops the classic consonant: the end of the onset region is the
    // same boundary, so showing both put two draggable lines on one position.
    std::vector<std::size_t> visibleParameters() const
    {
        if (!jie) return { 0, 1, 2, 3, 4 };
        // One boundary editor fewer per region fewer: three regions have two
        // inner boundaries, two regions have one.
        std::vector<std::size_t> shown { 0, 2, 3, 4 };
        for (int index = 0; index + 1 < regionCount(); ++index)
            shown.push_back(static_cast<std::size_t>(5 + index));
        return shown;
    }
    // The count lives in the class string; without one it is four, which is
    // what an oto4 entry has always had.
    [[nodiscard]] int regionCount() const;
    // Switch this entry to a different count: the letters that survive are
    // kept, and the boundaries that no longer exist collapse onto the end.
    void setRegionCount(int count);
    // Mark one region a consonant, or stop.  Unmarking writes a vowel unless
    // the region is a silence, which the tick boxes have no way to say and so
    // leave alone -- the class field is where an S is typed.
    void setRegionIsConsonant(int index, bool consonant);
    [[nodiscard]] bool regionIsConsonant(int index) const;
    // Push the count and the classes into the view and the editor row.
    void syncRegionCount();
    // Which parameter boxes are on screen, from visibleParameters().
    void applyParameterVisibility();
    void refreshEditors();
    void commitEditors();
    void save();
    void closeWindow();
    static juce::String formatNumber(double value);
    // 播放原音: start the whole recording from the top, or stop it.
    void togglePlayback();
    void startPlayback();
    void stopPlayback();
    // While playing: the playhead follows the sound, and the end of the
    // recording puts the button back.
    void timerCallback() override;

    // Modeless dialogs are not children of MainComponent and cannot inherit
    // its local LookAndFeel. Keep a complete theme alive for this editor tree.
    HachiLookAndFeel editorLookAndFeel;
    bool jie;
    bool mou;
    std::shared_ptr<MaterialAnnotationSession> materialSession;
    std::unique_ptr<I18n> ownedMaterialStrings; // legacy forNote factory lifetime
    std::function<bool(const VoicebankOtoEntry&, SampleRegionSetting&, juce::String&)> classicScalarProjection;
    std::unique_ptr<MaterialAnnotationEditor> materialOverview;
    I18n* materialStrings = nullptr;
    std::function<void(const std::vector<SampleRegionSetting>&)> materialSaved;
    juce::ComboBox materialMode;
    juce::Label materialModeLabel;
    juce::TextButton materialBackButton;
    juce::ToggleButton materialSingleSyllable;
    bool singleSyllableMaterial = false;
    juce::PropertiesFile* materialViewPreferences = nullptr;
    juce::String materialViewPreferenceKey;
    juce::ComboBox materialScope;
    juce::TextButton materialRestoreLocal;
    void refreshScopeView();
    void requestEditScope(EditScope);
    void requestRestoreLocal();
    juce::ComboBox materialSegmentChoice, materialPronunciationKindChoice;
    juce::Label materialSegmentLabel, materialGapLabel, materialKindLabel;
    juce::ToggleButton materialSimplifiedConsonant;
    juce::TextEditor materialConsonantGap;
    juce::TextButton materialSplitSegment, materialMergeSegment;
    juce::Label materialVowelNameLabel, materialVowelWeightLabel, materialSplitLeftLabel, materialSplitRightLabel;
    juce::TextEditor materialVowelName, materialVowelWeight, materialSplitLeftName, materialSplitRightName;
    juce::TextButton materialSplitConfirm, materialSplitCancel;
    std::vector<juce::String> materialVowelIds;
    juce::String materialVowelEditId, materialPendingSplitId, materialVowelWeightBaseline;
    double materialPendingSplitCut = 0;
    bool materialVowelSplitPending = false, refreshingMaterialVowel = false;
    int selectedMaterialVowelIndex() const;
    bool commitMaterialVowelFields();
    void beginMaterialVowelSplit();
    bool confirmMaterialVowelSplit();
    void cancelMaterialVowelSplit();
    SampleRegionSetting materialDetailDraft;
    VoicebankOtoEntry materialTimingEntry;
    bool syncMaterialTiming(juce::String& error);
    void refreshMaterialTimingControls();
    void commitMaterialConsonantGap();
    void refreshNativeSegmentControls();
    bool materialDetail = false;
    bool loadingMaterial = false;
    juce::String materialRegionId, materialDetailError;
    VoicebankOtoEntry detailBaseline;
    std::vector<NativeSegment> detailSegmentsBaseline;
    void refreshMaterialVisibility();
    void loadMaterialDetail(const SampleRegionSetting&);
    bool commitMaterialDetail(juce::String& error);
    void saveMaterial();
    void materialSourceSaved(const std::vector<SampleRegionSetting>& rows);
    juce::String materialText(const char* zh, const char* en) const;
    VoicebankOtoEntry original;
    VoicebankOtoEntry edited;
    std::function<void()> onSaved;
    // Set for 单独OTO编辑, where the entry belongs to one note.
    std::function<void(const VoicebankOtoEntry&)> noteHandoff;
    std::function<bool(const VoicebankOtoEntry&, juce::String&)> saveOverride;
    juce::Label fileLabel;
    juce::Label helpLabel;
    juce::Label pitchReadoutTitle, pitchReadoutHz, pitchReadoutTime;
    juce::Rectangle<int> pitchReadoutBounds;
    // The class string, shown only in 谋 mode: one letter per region, and its
    // length is the region count.
    juce::Label classesLabel;
    juce::TextEditor classesEditor;
    // Two / three / four regions, down the right of the waveform.  One of
    // them is always on, and picking one rewrites the class string.
    std::array<juce::TextButton, 3> countButtons;
    // One tick per region, ticked where that region is a consonant.  Any
    // number of them may be ticked, none included.
    juce::Label consonantLabel;
    std::array<juce::ToggleButton, 4> consonantButtons;
    // Five classic oto values, then the three Jie boundaries.  The last
    // three are only created and laid out when jie is true.
    std::array<juce::Label, 8> parameterLabels;
    std::array<juce::TextEditor, 8> parameterEditors;
    WaveformView waveform;
    juce::TextButton saveButton;
    juce::TextButton cancelButton;
    // Small +/- pair per axis, down the right-hand side of the waveform.
    juce::TextButton hZoomInButton, hZoomOutButton;
    juce::TextButton vZoomInButton, vZoomOutButton;
    juce::TextButton zoomResetButton;
    // Zooming in can leave the entry off screen; this is how it is reached
    // without having to drag the waveform itself.
    juce::ScrollBar waveformScroll { false };
    void scrollBarMoved(juce::ScrollBar* bar, double newRangeStart) override;
    void refreshScrollBar();
    bool refreshingEditors = false;
    // 播放原音: the whole recording the entry is cut from, never anything an
    // engine made of it.  It is read into memory when play is pressed and
    // handed to the host's device as a callback of its own, beside the song's
    // rather than instead of it: the song's position, and the view that
    // follows it, stay where they were.
    class SourcePreview;
    PlaybackHost playbackHost;
    std::unique_ptr<SourcePreview> preview;
    // Made only when there is a device to hand it to.  Held by pointer as well
    // because a player carries 256 channel pointers, and every editor the
    // checks build by value sits in the one stack frame they all share.
    std::unique_ptr<juce::AudioSourcePlayer> previewPlayer;
    juce::AudioDeviceManager* previewDevices = nullptr;
    juce::TextButton playButton;
    juce::ToggleButton spectrumButton, sourceF0Button;
};
}
