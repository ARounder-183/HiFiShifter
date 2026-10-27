#pragma once

#include "ProjectModel.h"
#include "RenderedWaveformPeaks.h"
#include <juce_audio_utils/juce_audio_utils.h>
#include <functional>
#include <memory>
#include <unordered_map>

namespace hachi
{
class TimelineComponent final : public juce::Component,
                                public juce::SettableTooltipClient,
                                private juce::ChangeListener,
                                private juce::Timer
{
public:
    explicit TimelineComponent(ProjectModel& modelToUse);
    ~TimelineComponent() override;

    void setShowNativeRenderedWaveforms(bool show);
    void setAudioOverlapLabel(juce::String label) { nativeOverlapLabel=std::move(label); }
    void setNativeFocusedNote(juce::String id) { if(nativeFocusedNote!=id){nativeFocusedNote=std::move(id);repaint();} }
    [[nodiscard]] juce::String diagnosticNativeFocusedNote() const { return nativeFocusedNote; }
    void setNativeClipWaveforms(std::shared_ptr<const std::vector<NativeRenderedWaveform>> waveforms);
    void paint(juce::Graphics& g) override;
    std::function<juce::String(const TrackData&)> outputEngineNameProvider;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    void mouseDoubleClick(const juce::MouseEvent& event) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    bool keyPressed(const juce::KeyPress& key) override;
    void setSelectedClips(std::vector<juce::String> ids, bool notify = true);
    [[nodiscard]] bool isClipSelected(const juce::String& id) const;
    [[nodiscard]] const std::vector<juce::String>& selectedClipIds() const { return selectedClips; }
    [[nodiscard]] const std::vector<juce::String>& selectedTrackIds() const { return selectedTracks; }
    void setPixelsPerSecond(float value);
    void setRowHeight(float value);
    void setClipStretchMode(bool enabled) { stretchClipEdges = enabled; }
    void setClipStretchHint(juce::String hint) { clipStretchHint = std::move(hint); }
    [[nodiscard]] bool clipStretchMode() const { return stretchClipEdges; }
    void setGainKnobHint(juce::String hint) { gainKnobHint = std::move(hint); }
    void setGainEnvelopeTexts(juce::String hint, juce::StringArray menu)
    { envelopeHint = std::move(hint); envelopeMenuTexts = std::move(menu); }
    void setPlayheadSeconds(double seconds);
    [[nodiscard]] int pixelForSeconds(double seconds) const;
    [[nodiscard]] double secondsForPixel(int pixel) const;
    [[nodiscard]] juce::String trackIdForPixel(int pixel) const;
    // The lane and the moment under the pointer, or nothing when the pointer
    // is not over the lanes.  A copied piece of material lands here, which is
    // the only way to say "that track, there" -- clicking an empty lane
    // reports no clip and does not name the track it belongs to, so the
    // selection alone could never carry a paste onto another track.
    struct Anchor { juce::String trackId; double seconds = 0.0; };
    [[nodiscard]] std::optional<Anchor> pointerAnchor() const;
    // The lane geometry, so a check can aim at a lane without guessing it.
    [[nodiscard]] int getRulerHeight() const { return rulerHeight; }
    [[nodiscard]] int getRowHeight() const { return rowHeight; }
    [[nodiscard]] int diagnosticRulerHeight() const { return getRulerHeight(); }
    [[nodiscard]] int diagnosticRowHeight() const { return getRowHeight(); }
    // Takes the model's current state, as a change message would.
    void diagnosticRefresh() { snapshot = model.snapshot(); }
    [[nodiscard]] juce::String clipHeaderText(const TrackData&, const ClipData&, float gain) const;
    void diagnosticShowTempoDialog(double quarterPosition) { showTempoDialog(quarterPosition); }
    std::function<void(double)> onSeek;
    // A right-click on empty lane space, in screen coordinates.
    std::function<void(juce::Point<int>)> onEmptyAreaMenu;
    std::function<void(const juce::String&)> onClipSelected;
    std::function<void(const std::vector<juce::String>&)> onTracksSelected;
    std::function<void()> onDeleteSelected;
    std::function<void(const juce::String&)> onClipGainRequested;
    // Clip id, red playhead seconds captured at menu opening, popup screen position.
    std::function<void(const juce::String&, double, juce::Point<int>)> onClipMenu;

private:
    juce::String nativeFocusedNote;
    juce::String nativeOverlapLabel=juce::String::fromUTF8("重叠");
    void changeListenerCallback(juce::ChangeBroadcaster*) override;
    void timerCallback() override;
    void rebuild();
    void notifySelection();
    void beginMarquee(const juce::MouseEvent& event);
    void updateMarquee(juce::Point<float> position);
    [[nodiscard]] float timeToX(double seconds) const;
    [[nodiscard]] double gridQuarterNotes() const;
    [[nodiscard]] double snapToGrid(double seconds) const;
    void showTempoMenu(double quarterPosition, juce::Point<int> screenPosition);
    void showTempoDialog(double quarterPosition);
    struct EnvelopeHit { juce::String clipId; std::size_t trackIndex; int pointIndex = -1; };
    [[nodiscard]] juce::Rectangle<float> envelopeArea(std::size_t trackIndex) const;
    [[nodiscard]] const std::vector<TrackGainPoint>& displayedEnvelope(const ClipData& clip) const;
    [[nodiscard]] std::optional<EnvelopeHit> envelopeHit(juce::Point<float> position, bool includeLine) const;
    void paintClipEnvelope(juce::Graphics&, const ClipData&, bool trackMuted, std::size_t);
    void showEnvelopePointMenu(const EnvelopeHit&, juce::Point<int>);
    void editEnvelopePoint(const juce::String& clipId, double time, int action);
    void clearEnvelopeDrag();
    [[nodiscard]] const ClipData* findEnvelopeClip(const juce::String&) const;
    [[nodiscard]] double envelopeDisplayOrigin(const ClipData&) const;

    struct ClipHit
    {
        juce::String id;
        juce::Rectangle<float> bounds;
        double startSeconds = 0.0;
        double durationSeconds = 0.0;
        double fadeInSeconds = 0.0;
        double fadeOutSeconds = 0.0;
        bool muted = false;
        double audioStartSeconds = 0.0;
        double audioDurationSeconds = 0.0;
        float gain = 1.0f;
        bool nativeStretch = false;
    };

    [[nodiscard]] bool nativeStretchEdgeAt(juce::Point<float> position) const;
    ProjectModel& model;
    ProjectData snapshot;
    juce::AudioFormatManager formats;
    juce::AudioThumbnailCache thumbnailCache { 96 };
    void rebuildNativeWaveformHashes();
    bool showNativeRenderedWaveforms=false;
    std::shared_ptr<const std::vector<NativeRenderedWaveform>> nativeWaveforms;
    std::unordered_map<std::uint64_t,const NativeRenderedPeaks*> nativePeaksByHash;
    std::unordered_map<std::string,std::uint64_t> nativeClipHashes;
    std::unordered_map<std::string, std::unique_ptr<juce::AudioThumbnail>> thumbnails;
    std::vector<ClipHit> clipHits;
    float pixelsPerSecond = 140.0f;
    static constexpr int rulerHeight = 24;
    int rowHeight = 96;
    std::optional<Anchor> hoverAnchor;
    void rememberPointer(const juce::MouseEvent& event);
    double playheadSeconds = 0.0;
    juce::String selectedClip;
    std::vector<juce::String> selectedClips, selectedTracks;
    std::vector<juce::String> marqueeBaseClips, marqueeBaseTracks;
    bool marqueePending = false, marqueeActive = false, marqueeAdditive = false;
    juce::Point<float> marqueeAnchor;
    juce::Rectangle<float> marqueeBounds;
    double earliestDraggedStart = 0.0;
    int draggedClipTrack = 0, draggedTrackDelta = 0, firstDraggedTrack = 0, lastDraggedTrack = 0;
    juce::String draggedClip;
    double draggedClipStart = 0.0;
    double draggedClipDuration = 0.0;
    double draggedClipPreviewStart = 0.0;
    double draggedClipPreviewDuration = 0.0;
    double draggedClipFadeIn = 0.0;
    double draggedClipFadeOut = 0.0;
    double draggedClipPreviewFadeIn = 0.0;
    double draggedClipPreviewFadeOut = 0.0;
    double draggedAudioStart = 0.0, draggedAudioDuration = 0.0;
    float draggedClipPreviewGain = 1.0f, draggedClipGainDb = 0.0f, gainDragY = 0.0f;
    juce::String gainKnobHint;
    bool gainValueClick = false;
    bool stretchClipEdges = false, draggedClipStretch = false;
    juce::String clipStretchHint;
    juce::String envelopeHint, selectedEnvelopeClip, draggedEnvelopeClip;
    juce::StringArray envelopeMenuTexts;
    double selectedEnvelopeTime = -1.0, draggedEnvelopeStart = 0.0, draggedEnvelopeDuration = 0.0;
    std::vector<TrackGainPoint> envelopeBeforeDrag, envelopePreview;
    int draggedEnvelopePoint = -1;
    std::size_t draggedEnvelopeClipIndex = 0;
    juce::Point<float> envelopeDragPosition;
    float dragAnchorX = 0.0f;
    enum class DragMode { none, move, resizeLeft, resizeRight, fadeIn, fadeOut, gain }
        dragMode = DragMode::none;
};
}
