#pragma once

#include "I18n.h"
#include "ProjectModel.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>
#include <optional>

namespace hachi
{
// A one-line field wide enough for a track name is not wide enough for a
// string of flags, and a value you cannot see all of is one you cannot check.
// This one opens out when it is clicked into and folds back when it is done
// with; TextEditor announces losing focus but not gaining it.
class FocusExpandingEditor final : public juce::TextEditor
{
public:
    std::function<void()> onFocusGained;
    void focusGained(FocusChangeType cause) override
    {
        juce::TextEditor::focusGained(cause);
        if (onFocusGained) onFocusGained();
    }
};

class TrackListComponent final : public juce::Component,
                                 public juce::TooltipClient,
                                 private juce::ChangeListener,
                                 private juce::Timer
{
public:
    // The five painted switches on a track row.  They are drawn, not built
    // out of buttons, so nothing carries a name a tooltip could read: the
    // geometry below is the only thing that knows which is which.
    enum class TrackToggle { none, compose, muted, solo, smoothOverlaps, normalizeVolume };
    // Which switch a point in a row falls on.  Pure, public, and shared with
    // the click, so a tooltip can never describe a switch other than the one
    // the click would flip.
    [[nodiscard]] static TrackToggle toggleAt(int x, int localY);
    // What to say about it, as an I18n key, or nullptr for none.
    [[nodiscard]] static const char* tooltipKeyFor(TrackToggle toggle);
    // The text for a point in the panel.  Empty away from the switches: a
    // tooltip over the name or the faders would only be in the way.
    [[nodiscard]] juce::String tooltipAt(int x, int y) const;
    // What JUCE asks for.  It gives no position, so the panel remembers where
    // the pointer was told it is -- the same way the timeline does.
    [[nodiscard]] juce::String getTooltip() override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;
    TrackListComponent(ProjectModel& modelToUse, I18n& stringsToUse);
    ~TrackListComponent() override;

    void paint(juce::Graphics& g) override;
    std::function<juce::String(const TrackData&)> outputEngineNameProvider;
    void setHighlightedTracks(const std::vector<juce::String>& ids) { highlightedTracks = ids; repaint(); }
    // Re-apply the Palette-derived label colours so a theme switch does not
    // leave them on the previous theme's (unreadable) colour.
    void lookAndFeelChanged() override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;
    bool keyPressed(const juce::KeyPress& key) override;
    void resized() override;
    void setSelectedTrack(const juce::String& trackId);
    // Test seam: keyboard focus needs a window on the desktop, which an
    // offline harness has not got, so the opened-out state can be asked for
    // directly.  Everything downstream of it is the real code path.
    void diagnosticOpenFlagsField(bool open)
    {
        flagsFieldForcedOpen = open;
        resized();
    }
    // The real path: focus, as a click would give it.  Needs the panel to be
    // on the desktop, which is why the forced seam exists as well.
    void diagnosticFocusFlagsField() { globalFlagsEditor.grabKeyboardFocus(); }
    [[nodiscard]] bool diagnosticFlagsFieldHasFocus() const
    {
        return globalFlagsEditor.hasKeyboardFocus(true);
    }
    [[nodiscard]] juce::Rectangle<int> diagnosticFlagsFieldBounds() const
    {
        return globalFlagsEditor.getBounds();
    }
    [[nodiscard]] int diagnosticFlagsTextWidth() const
    {
        return globalFlagsEditor.getTextWidth();
    }
    [[nodiscard]] int diagnosticFlagsTextHeight() const
    {
        return globalFlagsEditor.getTextHeight();
    }
    void setRowHeight(float value);

    // Which track a click at this height belongs to, or -1 for none: the
    // ruler above the rows, and the empty space below the last one.  That
    // empty space is the area around the tracks, where a right-click offers
    // to make one.  Pure, and public so a check reads the same rule the
    // component does.
    [[nodiscard]] static int rowIndexAt(int y, int rulerTop, int rowSize, int trackCount);

    // Test seam: the popup-menu modifier arrives from the desktop, which an
    // offline harness has not got.  Everything after it is the real path.
    void diagnosticRightClick(int x, int y);
    void diagnosticReorderScrollTick() { timerCallback(); }

    std::function<float(const juce::String&)> peakProvider;
    std::function<void(const juce::String&)> onTrackSelected;
    std::function<void(const juce::String&, juce::Point<int>)> onTrackMenu;
    std::function<void(const juce::String&)> onTrackGainRequested;
    std::function<void(const juce::String&)> onTrackPanRequested;
    // A right-click on the space around the tracks, in screen coordinates.
    std::function<void(juce::Point<int>)> onEmptyAreaMenu;

private:
    void changeListenerCallback(juce::ChangeBroadcaster*) override;
    void refreshUtauEditors();
    void commitUtauEditors();
    void timerCallback() override;
    void updateReorderTarget();
    void cancelReorder();

    ProjectModel& model;
    I18n& strings;
    ProjectData snapshot;
    static constexpr int rulerHeight = 24;
    int rowHeight = 96;
    juce::String selectedTrack;
    std::vector<juce::String> highlightedTracks;
    juce::String volumeDragTrack;
    float volumeDragPreview = 1.0f;
    juce::String panDragTrack;
    float panDragPreview = 0.0f;
    int volumeDragRow = -1;
    juce::String reorderTrack;
    juce::Point<int> reorderScreenPoint;
    bool reordering = false;
    int reorderGap = -1;
    juce::Label consonantVelocityLabel;
    juce::TextEditor consonantVelocityEditor;
    juce::Label globalFlagsLabel;
    FocusExpandingEditor globalFlagsEditor;
    // Rows of a single line's height that the opened-out field may grow to.
    static constexpr int expandedFlagsRows = 4;
    bool flagsFieldForcedOpen = false;
    // Where the pointer last was, for getTooltip.
    std::optional<juce::Point<int>> pointer;
    bool updatingUtauEditors = false;
};
}
