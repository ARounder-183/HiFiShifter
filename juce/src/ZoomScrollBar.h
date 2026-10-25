#pragma once
#include "Theme.h"
#include <functional>

namespace hachi
{
// Shared editor wheel modifiers; panels and main viewports use the same mapping.
enum class EditorWheelAction { scrollVertically, scrollHorizontally, zoomVertically, zoomHorizontally };
inline EditorWheelAction editorWheelActionFor(const juce::ModifierKeys& modifiers)
{
    if(modifiers.isCommandDown())
        return modifiers.isShiftDown()?EditorWheelAction::zoomHorizontally:EditorWheelAction::zoomVertically;
    return modifiers.isShiftDown()?EditorWheelAction::scrollHorizontally:EditorWheelAction::scrollVertically;
}

// JUCE still owns normal scrolling. Only the two circular end grips resize the view.
class ZoomScrollBar final : public juce::ScrollBar
{
public:
    explicit ZoomScrollBar(bool vertical) : juce::ScrollBar(vertical)
    {
        setLookAndFeel(&appearance);
        setAutoHide(false);
    }
    ~ZoomScrollBar() override { setLookAndFeel(nullptr); }

    std::function<void(bool leading)> onResizeBegin;
    std::function<void(double scale)> onResize;

    juce::Rectangle<float> thumbBounds() const
    {
        const auto length = isVertical() ? getHeight() : getWidth();
        if (length < 32 + minimumThumb) return {};
        const auto total = getRangeLimit();
        const auto range = getCurrentRange();
        auto size = juce::roundToInt(total.getLength() > 0.0
            ? range.getLength() * length / total.getLength() : length);
        size = juce::jlimit(juce::jmin(minimumThumb, length - 1), length, size);
        const auto start = total.getLength() > range.getLength()
            ? juce::roundToInt((range.getStart() - total.getStart()) * (length - size)
                              / (total.getLength() - range.getLength())) : 0;
        return isVertical() ? juce::Rectangle<float>(0, static_cast<float>(start),
                                static_cast<float>(getWidth()), static_cast<float>(size))
                            : juce::Rectangle<float>(static_cast<float>(start), 0,
                                static_cast<float>(size), static_cast<float>(getHeight()));
    }

    juce::Point<float> gripCentre(bool leading) const
    {
        const auto b = thumbBounds();
        return isVertical() ? juce::Point<float>(b.getCentreX(), leading ? b.getY() + 8 : b.getBottom() - 8)
                            : juce::Point<float>(leading ? b.getX() + 8 : b.getRight() - 8, b.getCentreY());
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(Palette::base);
        auto b = thumbBounds();
        if (b.isEmpty()) return;
        const auto colour = Palette::scrollThumb.brighter(isMouseOverOrDragging() ? 0.2f : 0.0f);
        const auto body = isVertical() ? b.reduced((b.getWidth() - 6.0f) * 0.5f, 16.0f)
                                      : b.reduced(16.0f, (b.getHeight() - 6.0f) * 0.5f);
        g.setColour(colour);
        g.fillRoundedRectangle(body, 3.0f);
        for (const bool leading : { true, false })
        {
            const auto p = gripCentre(leading);
            g.setColour(activeEdge == (leading ? -1 : 1) ? Palette::accentLight : colour);
            g.fillEllipse(p.x - 5.0f, p.y - 5.0f, 10.0f, 10.0f);
        }
    }

    void mouseMove(const juce::MouseEvent& e) override { updateCursor(e.position); repaint(); }
    void mouseExit(const juce::MouseEvent&) override { repaint(); }
    void mouseDown(const juce::MouseEvent& e) override
    {
        activeEdge = e.mods.isLeftButtonDown() ? edgeAt(e.position) : 0;
        if (activeEdge != 0 && onResize)
        {
            dragOrigin = isVertical() ? e.position.y : e.position.x;
            const auto b = thumbBounds();
            dragLength = isVertical() ? b.getHeight() : b.getWidth();
            if (onResizeBegin) onResizeBegin(activeEdge < 0);
            updateCursor(e.position);
            repaint();
            return;
        }
        activeEdge = 0;
        juce::ScrollBar::mouseDown(e);
    }
    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (activeEdge != 0)
        {
            const auto delta = (isVertical() ? e.position.y : e.position.x) - dragOrigin;
            const auto span = juce::jmax(4.0f, dragLength + (activeEdge < 0 ? -delta : delta));
            if (onResize) onResize(dragLength / span);
            return;
        }
        juce::ScrollBar::mouseDrag(e);
    }
    void mouseUp(const juce::MouseEvent& e) override
    {
        activeEdge = 0;
        juce::ScrollBar::mouseUp(e);
        updateCursor(e.position);
    }

private:
    static constexpr int minimumThumb = 44;
    struct Appearance final : juce::LookAndFeel_V4
    {
        bool areScrollbarButtonsVisible() override { return false; }
        int getMinimumScrollbarThumbSize(juce::ScrollBar&) override { return minimumThumb; }
    } appearance;
    int activeEdge = 0;
    float dragOrigin = 0.0f, dragLength = 1.0f;
    int edgeAt(juce::Point<float> p) const
    {
        if (thumbBounds().isEmpty()) return 0;
        for (const bool leading : { true, false })
            if (p.getDistanceFrom(gripCentre(leading)) <= 8.0f) return leading ? -1 : 1;
        return 0;
    }
    void updateCursor(juce::Point<float> p)
    {
        setMouseCursor(activeEdge != 0 || edgeAt(p) != 0
            ? (isVertical() ? juce::MouseCursor::UpDownResizeCursor : juce::MouseCursor::LeftRightResizeCursor)
            : juce::MouseCursor::NormalCursor);
    }
};
}
