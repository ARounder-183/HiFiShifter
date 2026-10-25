#include "TrackListComponent.h"
#include "Theme.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace hachi
{
namespace
{
juce::String algorithmLabel(const TrackData& track, const juce::String& outputName)
{
    const auto pitch = track.pitchAlgorithm == PitchAlgorithm::nsfHifigan
        ? juce::String("nsf-hifigan") + (utauModeUsesRegions(track.utauMode)
            ? " · " + utauModeLabel(track.utauMode) : juce::String())
        : track.pitchAlgorithm == PitchAlgorithm::world ? juce::String("WORLD")
        : track.pitchAlgorithm == PitchAlgorithm::vocalShifter ? juce::String("vslib")
        : track.pitchAlgorithm == PitchAlgorithm::mld3 ? juce::String("mld3 (disabled)")
        : track.pitchAlgorithm == PitchAlgorithm::llsm2 ? juce::String("llsm2")
        : track.pitchAlgorithm == PitchAlgorithm::utau
            ? utauModeLabel(track.utauMode)
        : juce::String("mld5 (disabled)");
    const auto stretch = track.stretchAlgorithm == StretchAlgorithm::variableMelHop
        ? juce::String("variable-mel-hop")
        : track.stretchAlgorithm == StretchAlgorithm::loop ? juce::String("loop")
        : track.stretchAlgorithm == StretchAlgorithm::soundTouch ? juce::String("soundtouch")
        : track.stretchAlgorithm == StretchAlgorithm::wcsNonLoop
            ? juce::String("wcsndm-nonloop")
        : track.stretchAlgorithm == StretchAlgorithm::hifiShifterMel
            ? juce::String("hifishifter-mel")
        : track.stretchAlgorithm == StretchAlgorithm::nsfShiftThenSplice
            ? juce::String("nsf-shift-then-splice")
        : juce::String("melodyne-hybrid");
    if (trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track))
        return pitch + (track.chineseCvvc && track.utauMode == UtauMode::classic ? " · ZH CVVC" : "")
            + (outputName.isEmpty() ? juce::String{} : " / " + outputName);
    return pitch + " / " + stretch;
}
}

TrackListComponent::TrackListComponent(ProjectModel& modelToUse, I18n& stringsToUse)
    : model(modelToUse), strings(stringsToUse), snapshot(model.snapshot())
{
    model.addChangeListener(this);
    consonantVelocityLabel.setText(juce::String::fromUTF8("辅音速度"),
                                   juce::dontSendNotification);
    consonantVelocityLabel.setFont(juce::FontOptions(9.0f));
    consonantVelocityLabel.setColour(juce::Label::textColourId, Palette::textMuted);
    consonantVelocityLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(consonantVelocityLabel);

    consonantVelocityEditor.setMultiLine(false);
    consonantVelocityEditor.setInputRestrictions(0, "-0123456789");
    consonantVelocityEditor.setJustification(juce::Justification::centred);
    consonantVelocityEditor.setTooltip(juce::String::fromUTF8(
        "UTAU 辅音速度：100 为原速；大于 100 更快，小于 100 更慢；支持负数且不限制上下限"));
    addAndMakeVisible(consonantVelocityEditor);

    globalFlagsLabel.setText(juce::String::fromUTF8("全局 Flags"),
                             juce::dontSendNotification);
    globalFlagsLabel.setFont(juce::FontOptions(9.0f));
    globalFlagsLabel.setColour(juce::Label::textColourId, Palette::textMuted);
    globalFlagsLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(globalFlagsLabel);

    globalFlagsEditor.setMultiLine(false);
    globalFlagsEditor.setTooltip(juce::String::fromUTF8(
        "应用到此轨道全部 UTAU 音符的 Flags；每个音符自己的 Flags 会追加在后面"));
    addAndMakeVisible(globalFlagsEditor);

    consonantVelocityEditor.onReturnKey = [this] { commitUtauEditors(); };
    consonantVelocityEditor.onFocusLost = [this] { commitUtauEditors(); };
    consonantVelocityEditor.onEscapeKey = [this] { refreshUtauEditors(); };
    globalFlagsEditor.onReturnKey = [this] { commitUtauEditors(); };
    globalFlagsEditor.onFocusLost = [this]
    {
        commitUtauEditors();
        // Fold it back.  Committing an unchanged value announces nothing, so
        // waiting for the model to say something would leave it open.
        resized();
    };
    globalFlagsEditor.onEscapeKey = [this] { refreshUtauEditors(); };
    globalFlagsEditor.onFocusGained = [this] { resized(); };

    refreshUtauEditors();
    setSize(280, rulerHeight + std::max(rowHeight, static_cast<int>(snapshot.tracks.size()) * rowHeight));
}

TrackListComponent::~TrackListComponent()
{
    model.removeChangeListener(this);
}

void TrackListComponent::setSelectedTrack(const juce::String& trackId)
{
    if (selectedTrack == trackId) return;
    selectedTrack = trackId;
    refreshUtauEditors();
    resized();
    repaint();
}

void TrackListComponent::setRowHeight(float value)
{
    rowHeight = juce::jlimit(40, 220, static_cast<int>(std::round(value)));
    setSize(280, rulerHeight + std::max(rowHeight, static_cast<int>(snapshot.tracks.size()) * rowHeight));
    refreshUtauEditors();
    resized();
    repaint();
}

void TrackListComponent::changeListenerCallback(juce::ChangeBroadcaster*)
{
    cancelReorder();
    snapshot = model.snapshot();
    const auto selectedStillExists = std::any_of(snapshot.tracks.begin(), snapshot.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrack; });
    if (!selectedStillExists) selectedTrack.clear();
    setSize(280, rulerHeight + std::max(rowHeight, static_cast<int>(snapshot.tracks.size()) * rowHeight));
    refreshUtauEditors();
    resized();
    repaint();
}

void TrackListComponent::refreshUtauEditors()
{
    const auto found = std::find_if(snapshot.tracks.begin(), snapshot.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrack; });
    const auto visible = found != snapshot.tracks.end()
        && !found->accompaniment && found->pitchAlgorithm == PitchAlgorithm::utau && rowHeight >= 80;
    consonantVelocityLabel.setVisible(visible);
    consonantVelocityEditor.setVisible(visible);
    globalFlagsLabel.setVisible(visible);
    globalFlagsEditor.setVisible(visible);
    if (!visible) return;

    const juce::ScopedValueSetter<bool> guard(updatingUtauEditors, true);
    consonantVelocityEditor.setText(juce::String(found->utauConsonantVelocity), false);
    globalFlagsEditor.setText(found->utauGlobalFlags, false);
}

void TrackListComponent::commitUtauEditors()
{
    if (updatingUtauEditors || selectedTrack.isEmpty()) return;
    const auto velocityText = consonantVelocityEditor.getText().trim();
    const auto velocity = velocityText.isEmpty() ? 100 : velocityText.getIntValue();
    model.setTrackUtauConsonantVelocity(selectedTrack, velocity);
    // Flags are one line however many the field is showing them on: pasting
    // something with a line break in it must not put one into the value.
    const auto lineBreaks = juce::String::charToString(juce::juce_wchar(13))
                          + juce::String::charToString(juce::juce_wchar(10));
    const auto oneLine = globalFlagsEditor.getText()
        .replaceCharacters(lineBreaks, "  ").trim();
    model.setTrackUtauGlobalFlags(selectedTrack, oneLine);
    if (oneLine != globalFlagsEditor.getText())
        globalFlagsEditor.setText(oneLine, false);
    consonantVelocityEditor.setText(juce::String(velocity), false);
}

void TrackListComponent::resized()
{
    const auto found = std::find_if(snapshot.tracks.begin(), snapshot.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrack; });
    if (found == snapshot.tracks.end()) return;
    const auto index = static_cast<int>(std::distance(snapshot.tracks.begin(), found));
    const auto y = rulerHeight + index * rowHeight + 59;
    consonantVelocityLabel.setBounds(8, y, 49, 19);
    consonantVelocityEditor.setBounds(58, y, 40, 19);
    globalFlagsLabel.setBounds(103, y, 51, 19);
    if (!globalFlagsEditor.hasKeyboardFocus(true) && !flagsFieldForcedOpen)
    {
        globalFlagsEditor.setMultiLine(false);
        globalFlagsEditor.setBounds(155, y, std::max(20, getWidth() - 187), 19);
        return;
    }
    // Being edited: take the whole width of the panel, and as many lines as
    // the value needs, so all of it can be read and corrected at once.  The
    // width has to be set before the height is asked for, or the answer is
    // the number of lines the old width would have wrapped to.
    const auto width = std::max(20, getWidth() - 16);
    globalFlagsEditor.setMultiLine(true, true);
    globalFlagsEditor.setBounds(8, y, width, 19);
    globalFlagsEditor.setBounds(8, y, width,
        juce::jlimit(19, 19 * expandedFlagsRows,
                     globalFlagsEditor.getTextHeight() + 8));
    globalFlagsEditor.toFront(false);
}

void TrackListComponent::lookAndFeelChanged()
{
    consonantVelocityLabel.setColour(juce::Label::textColourId, Palette::textMuted);
    globalFlagsLabel.setColour(juce::Label::textColourId, Palette::textMuted);
    repaint();
}

void TrackListComponent::paint(juce::Graphics& g)
{
    g.fillAll(Palette::panel);
    g.setColour(Palette::background);
    g.fillRect(0, 0, getWidth(), rulerHeight);
    g.setColour(Palette::border);
    g.drawHorizontalLine(rulerHeight - 1, 0.0f, static_cast<float>(getWidth()));
    if (snapshot.tracks.empty())
    {
        g.setColour(Palette::textMuted);
        g.drawFittedText(strings.text("status.noTracks"), getLocalBounds().reduced(14),
                         juce::Justification::centred, 3);
        return;
    }

    const auto light = Palette::background.getPerceivedBrightness() > 0.5f;
    const std::array<juce::Colour, 6> colours {
        Palette::accent, Palette::accentLight, Palette::noteFill,
        juce::Colour(0xff9b8bdd), juce::Colour(0xffdedcff), juce::Colour(0xffffd94f)
    };
    for (std::size_t index = 0; index < snapshot.tracks.size(); ++index)
    {
        const auto& track = snapshot.tracks[index];
        auto row = juce::Rectangle<int>(0, rulerHeight + static_cast<int>(index) * rowHeight,
                                        getWidth(), rowHeight);
        auto rowSurface = index % 2 == 0 ? Palette::panel : Palette::panelRaised.darker(0.1f);
        g.setColour(rowSurface);
        g.fillRect(row);
        if (highlightedTracks.empty() ? track.id == selectedTrack
            : std::find(highlightedTracks.begin(), highlightedTracks.end(), track.id) != highlightedTracks.end())
        {
            const auto selectedOverlay = (light ? Palette::selection : Palette::accent).withAlpha(0.14f);
            rowSurface = rowSurface.overlaidWith(selectedOverlay);
            g.setColour(selectedOverlay);
            g.fillRect(row);
            g.setColour(Palette::accent);
            g.fillRect(row.getX(), row.getY(), 3, row.getHeight());
        }
        g.setColour(Palette::grid);
        g.drawLine(0.0f, static_cast<float>(row.getBottom() - 1),
                   static_cast<float>(getWidth()), static_cast<float>(row.getBottom() - 1));

        const auto colour = light
            ? (Palette::contrastRatio(Palette::trackColour(index), rowSurface) >= 3.0
                ? Palette::trackColour(index) : Palette::noteEdge)
            : colours[index % colours.size()];
        g.setColour(colour);
        g.fillEllipse(9.0f, static_cast<float>(row.getY() + 12), 9.0f, 9.0f);
        g.setColour(light ? Palette::contrastingText(rowSurface, track.muted)
                          : track.muted ? Palette::textMuted : Palette::text);
        g.setFont(13.0f);
        const auto displayedName = track.name
            + (track.accompaniment ? "  [" + strings.text("track.accompaniment") + "]"
                : track.compose ? "  [" + algorithmLabel(track, outputEngineNameProvider ? outputEngineNameProvider(track) : juce::String{}) + "]" : juce::String());
        g.drawText(displayedName, row.getX() + 24, row.getY() + 5, row.getWidth() - 60, 23,
                   juce::Justification::centredLeft, true);

        const auto buttonY = static_cast<float>(row.getY() + 34);
        const auto drawToggle = [&](float x, const char* label, bool active, juce::Colour activeColour)
        {
            const juce::Rectangle<float> bounds(x, buttonY, 28.0f, 21.0f);
            const auto surface = active ? (light ? Palette::primaryButton : activeColour)
                                        : light ? Palette::button : Palette::panelRaised;
            g.setColour(surface);
            g.fillRoundedRectangle(bounds, 3.0f);
            g.setColour(light ? Palette::contrastingText(surface, !active)
                              : active ? activeColour.contrasting(0.85f) : Palette::textMuted);
            g.setFont(11.0f);
            g.drawText(label, bounds.toNearestInt(), juce::Justification::centred);
        };
        if (!track.accompaniment) drawToggle(10.0f, "C", track.compose, Palette::accentLight);
        drawToggle(41.0f, "M", track.muted, Palette::noteFill);
        drawToggle(72.0f, "S", track.solo, Palette::accent);
        drawToggle(103.0f, "A", track.smoothOverlaps, juce::Colour(0xff34b56f));
        if (!track.accompaniment) drawToggle(134.0f, "N", track.normalizeVolume, juce::Colour(0xffff9f2f));

        const auto slider = juce::Rectangle<float>(170.0f, buttonY + 6.0f,
                                                     std::max(20.0f, static_cast<float>(getWidth()) - 210.0f), 8.0f);
        g.setColour(Palette::background);
        g.fillRoundedRectangle(slider, 4.0f);
        const auto displayedVolume = track.id == volumeDragTrack ? volumeDragPreview : track.volume;
        const auto normalized = juce::jlimit(0.0f, 1.0f, displayedVolume * 0.5f);
        g.setColour(colour);
        g.fillRoundedRectangle(slider.withWidth(slider.getWidth() * normalized), 4.0f);
        g.setColour(light ? Palette::noteEdge : Palette::accentLight);
        g.fillEllipse(slider.getX() + slider.getWidth() * normalized - 4.0f,
                      slider.getCentreY() - 4.0f, 8.0f, 8.0f);
        const auto db = displayedVolume <= 0.0f ? juce::String("-inf")
            : juce::String(20.0f * std::log10(displayedVolume), 1);
        g.setColour(light ? Palette::contrastingText(rowSurface, true) : Palette::textMuted);
        g.setFont(10.0f);
        const auto showingUtauEditors = track.id == selectedTrack
            && !track.accompaniment && track.pitchAlgorithm == PitchAlgorithm::utau && rowHeight >= 80;
        if (!showingUtauEditors)
        {
            g.drawText(strings.text("track.volume") + "  " + db + " dB", 170, row.getY() + 61, getWidth() - 208, 19,
                       juce::Justification::centredLeft);
            g.drawText(strings.text(track.accompaniment ? "track.originalAudio" : track.compose ? "track.compose" : "track.audio"),
                       10, row.getY() + 61, 94, 19, juce::Justification::centredLeft);
        }

        const auto displayedPan = track.id == panDragTrack ? panDragPreview : track.pan;
        const juce::Rectangle<float> panRail(170.0f, static_cast<float>(row.getY() + 84),
                                              std::max(20.0f, static_cast<float>(getWidth()) - 210.0f), 5.0f);
        g.setColour(Palette::background);
        g.fillRoundedRectangle(panRail, 2.5f);
        g.setColour(Palette::grid.brighter(0.25f));
        g.drawVerticalLine(static_cast<int>(panRail.getCentreX()), panRail.getY() - 2.0f,
                           panRail.getBottom() + 2.0f);
        const auto panX = panRail.getX() + (displayedPan + 1.0f) * 0.5f * panRail.getWidth();
        g.setColour(colour.brighter(0.4f));
        g.fillEllipse(panX - 3.5f, panRail.getCentreY() - 3.5f, 7.0f, 7.0f);
        g.setColour(light ? Palette::contrastingText(rowSurface, true) : Palette::textMuted);
        g.setFont(9.0f);
        g.drawText("P " + juce::String(displayedPan, 2), 65, row.getY() + 77, 45, 16,
                   juce::Justification::centredRight);

        const auto peak = peakProvider ? std::max(0.0f, peakProvider(track.id)) : 0.0f;
        const auto peakDb = peak > 1.0e-6f ? 20.0f * std::log10(peak) : -60.0f;
        const auto meterAmount = juce::jlimit(0.0f, 1.0f, (peakDb + 48.0f) / 51.0f);
        const juce::Rectangle<float> meterRail(static_cast<float>(getWidth() - 28),
                                                static_cast<float>(row.getY()), 28.0f,
                                                static_cast<float>(rowHeight));
        g.setColour(Palette::panel);
        g.fillRect(meterRail);
        g.setColour(peak >= 1.0f ? juce::Colours::red : Palette::textMuted);
        g.setFont(8.0f);
        g.drawText(peak >= 1.0f ? juce::String("+") + juce::String(std::max(0.0f, peakDb), 1)
                               : juce::String(peakDb, 0),
                   meterRail.withHeight(15.0f).toNearestInt(), juce::Justification::centred);
        auto well = meterRail.withTrimmedTop(17.0f).reduced(6.0f, 0.0f);
        g.setColour(juce::Colour(0xff1d1d1d));
        g.fillRect(well);
        auto fill = well.withTrimmedTop(well.getHeight() * (1.0f - meterAmount));
        g.setColour(peak >= 1.0f ? juce::Colours::red
                    : peakDb >= -6.0f ? juce::Colour(0xffff9f2f)
                    : peakDb >= -18.0f ? Palette::noteFill : juce::Colour(0xff34b56f));
        g.fillRect(fill);
    }
    if (reordering && reorderGap >= 0)
    {
        const auto y = juce::jlimit(1, getHeight()-2, rulerHeight + reorderGap * rowHeight);
        g.setColour(Palette::accent);
        g.fillRect(0, y-1, getWidth(), 3);
        g.fillEllipse(1.0f, static_cast<float>(y-4), 8.0f, 8.0f);
    }
}

int TrackListComponent::rowIndexAt(int y, int rulerTop, int rowSize, int trackCount)
{
    if (y < rulerTop || rowSize <= 0) return -1;
    const auto index = (y - rulerTop) / rowSize;
    return index >= 0 && index < trackCount ? index : -1;
}

void TrackListComponent::diagnosticRightClick(int x, int y)
{
    const juce::Point<float> where(static_cast<float>(x), static_cast<float>(y));
    const juce::MouseEvent event(juce::Desktop::getInstance().getMainMouseSource(),
        where, juce::ModifierKeys(juce::ModifierKeys::rightButtonModifier),
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, this, this,
        juce::Time::getCurrentTime(), where, juce::Time::getCurrentTime(), 1, false);
    mouseDown(event);
}

TrackListComponent::TrackToggle TrackListComponent::toggleAt(int x, int localY)
{
    // The row the switches are painted in, and the 28-pixel box each one is
    // drawn into.  Kept beside nothing: paint, click and tooltip all read it.
    if (localY < 34 || localY >= 56) return TrackToggle::none;
    if (x >= 10 && x < 38) return TrackToggle::compose;
    if (x >= 41 && x < 69) return TrackToggle::muted;
    if (x >= 72 && x < 100) return TrackToggle::solo;
    if (x >= 103 && x < 131) return TrackToggle::smoothOverlaps;
    if (x >= 134 && x < 162) return TrackToggle::normalizeVolume;
    return TrackToggle::none;
}

const char* TrackListComponent::tooltipKeyFor(TrackToggle toggle)
{
    switch (toggle)
    {
        case TrackToggle::compose:         return "track.tip.compose";
        case TrackToggle::muted:           return "track.tip.mute";
        case TrackToggle::solo:            return "track.tip.solo";
        case TrackToggle::smoothOverlaps:  return "track.tip.smooth";
        case TrackToggle::normalizeVolume: return "track.tip.normalize";
        case TrackToggle::none:            break;
    }
    return nullptr;
}

juce::String TrackListComponent::tooltipAt(int x, int y) const
{
    const auto index = rowIndexAt(y, rulerHeight, rowHeight,
                                 static_cast<int>(snapshot.tracks.size()));
    if (index < 0) return {};
    const auto toggle = toggleAt(x, y - rulerHeight - index * rowHeight);
    if (snapshot.tracks[static_cast<std::size_t>(index)].accompaniment
        && (toggle == TrackToggle::compose || toggle == TrackToggle::normalizeVolume))
        return strings.text("track.accompanimentHelp");
    const auto* key = tooltipKeyFor(toggle);
    return key != nullptr ? strings.text(key) : juce::String();
}

juce::String TrackListComponent::getTooltip()
{
    return pointer ? tooltipAt(pointer->x, pointer->y) : juce::String();
}

void TrackListComponent::mouseMove(const juce::MouseEvent& event)
{
    pointer = event.getPosition();
    const auto row = rowIndexAt(event.y,rulerHeight,rowHeight,static_cast<int>(snapshot.tracks.size()));
    setMouseCursor(row >= 0 && event.y-rulerHeight-row*rowHeight < 32 && event.x < getWidth()-28
        ? juce::MouseCursor::DraggingHandCursor : juce::MouseCursor::NormalCursor);
}

void TrackListComponent::mouseExit(const juce::MouseEvent&)
{
    // Or the last switch hovered would keep answering after the pointer had
    // gone somewhere else entirely.
    pointer.reset();
}

void TrackListComponent::mouseDown(const juce::MouseEvent& event)
{
    cancelReorder();
    const auto index = rowIndexAt(event.y, rulerHeight, rowHeight,
                                 static_cast<int>(snapshot.tracks.size()));
    if (index < 0)
    {
        // No track under the pointer.  A right-click on that space is asking
        // about the list itself rather than about any one track in it.
        if (event.mods.isPopupMenu() && onEmptyAreaMenu)
            onEmptyAreaMenu(event.getScreenPosition());
        return;
    }
    const auto& track = snapshot.tracks[static_cast<std::size_t>(index)];
    // Hit-test the readout that was visible before selecting this row (UTAU
    // selection replaces this part of the row with its velocity/Flags fields).
    const auto wasShowingUtauEditors = track.id == selectedTrack && !track.accompaniment
        && track.pitchAlgorithm == PitchAlgorithm::utau && rowHeight >= 80;
    const auto gainReadoutHit = !wasShowingUtauEditors
        && juce::Rectangle<int>(170,rulerHeight+index*rowHeight+61,getWidth()-208,19).contains(event.getPosition());
    selectedTrack = track.id;
    if (onTrackSelected) onTrackSelected(selectedTrack);
    refreshUtauEditors();
    resized();
    repaint();
    if (event.mods.isPopupMenu())
    {
        if (onTrackMenu) onTrackMenu(track.id, event.getScreenPosition());
        return;
    }
    const auto localY = event.y - rulerHeight - index * rowHeight;
    if (event.mods.isLeftButtonDown() && juce::Rectangle<int>(65,77,45,16).contains(event.x,localY))
    {
        if (onTrackPanRequested) onTrackPanRequested(track.id);
        return;
    }
    if(event.mods.isLeftButtonDown() && gainReadoutHit)
    {
        if(onTrackGainRequested)onTrackGainRequested(track.id);
        return;
    }
    if (event.mods.isLeftButtonDown() && localY < 32 && event.x < getWidth()-28)
    {
        reorderTrack = track.id;
        setWantsKeyboardFocus(true);
        if (isShowing()) grabKeyboardFocus();
        return;
    }
    switch (toggleAt(event.x, localY))
    {
        case TrackToggle::compose:
            if (!track.accompaniment) model.setTrackCompose(track.id, !track.compose); return;
        case TrackToggle::muted:
            model.setTrackMuted(track.id, !track.muted); return;
        case TrackToggle::solo:
            model.setTrackSolo(track.id, !track.solo); return;
        case TrackToggle::smoothOverlaps:
            model.setTrackSmoothOverlaps(track.id, !track.smoothOverlaps); return;
        case TrackToggle::normalizeVolume:
            if (!track.accompaniment) model.setTrackNormalizeVolume(track.id, !track.normalizeVolume); return;
        case TrackToggle::none: break;
    }
    if (localY >= 34 && localY < 58 && event.x >= 166 && event.x < getWidth() - 32)
    {
        volumeDragTrack = track.id;
        volumeDragRow = index;
        volumeDragPreview = track.volume;
        mouseDrag(event);
    }
    else if (localY >= 77 && localY < 96 && event.x >= 166 && event.x < getWidth() - 32)
    {
        panDragTrack = track.id;
        panDragPreview = track.pan;
        mouseDrag(event);
    }
}

void TrackListComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (reorderTrack.isNotEmpty())
    {
        if (!reordering && event.getDistanceFromDragStart() < 5) return;
        reordering = true;
        reorderScreenPoint = event.getScreenPosition();
        updateReorderTarget();
        startTimerHz(30);
        return;
    }
    const auto width = std::max(20, getWidth() - 210);
    const auto normalized = juce::jlimit(0.0f, 1.0f,
        static_cast<float>(event.x - 170) / static_cast<float>(width));
    if (volumeDragTrack.isNotEmpty()) volumeDragPreview = normalized * 2.0f;
    else if (panDragTrack.isNotEmpty()) panDragPreview = normalized * 2.0f - 1.0f;
    else return;
    repaint();
}

void TrackListComponent::mouseUp(const juce::MouseEvent& event)
{
    if (reorderTrack.isNotEmpty())
    {
        reorderScreenPoint = event.getScreenPosition();
        if (reordering) updateReorderTarget();
        const auto source = reorderTrack;
        const auto gap = reorderGap;
        const auto before = gap >= 0 && gap < static_cast<int>(snapshot.tracks.size())
            ? snapshot.tracks[static_cast<std::size_t>(gap)].id : juce::String();
        const auto commit = reordering && gap >= 0;
        cancelReorder();
        if (commit) model.moveTrackBefore(source,before);
    }
    if (volumeDragTrack.isNotEmpty())
        model.setTrackVolume(volumeDragTrack, volumeDragPreview);
    if (panDragTrack.isNotEmpty())
        model.setTrackPan(panDragTrack, panDragPreview);
    volumeDragTrack.clear();
    panDragTrack.clear();
    volumeDragRow = -1;
}

void TrackListComponent::cancelReorder()
{
    stopTimer(); reorderTrack.clear(); reordering=false; reorderGap=-1; repaint();
}

void TrackListComponent::updateReorderTarget()
{
    const auto point=getLocalPoint(nullptr,reorderScreenPoint);
    reorderGap=point.x < 0 || point.x >= getWidth() ? -1
        : juce::jlimit(0,static_cast<int>(snapshot.tracks.size()),
            static_cast<int>(std::floor(static_cast<double>(point.y-rulerHeight)/rowHeight+0.5)));
    repaint();
}

void TrackListComponent::timerCallback()
{
    if (!reordering) { stopTimer(); return; }
    if (auto* viewport=findParentComponentOfClass<juce::Viewport>())
    {
        const auto point=viewport->getLocalPoint(nullptr,reorderScreenPoint);
        if (point.x >= 0 && point.x < viewport->getWidth())
            viewport->autoScroll(point.x,point.y,24,12);
    }
    updateReorderTarget();
}

bool TrackListComponent::keyPressed(const juce::KeyPress& key)
{
    if (key == juce::KeyPress::escapeKey && reorderTrack.isNotEmpty())
    { cancelReorder(); return true; }
    return false;
}
}
