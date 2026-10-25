#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace hachi
{
struct Palette
{
    static juce::Colour background;
    static juce::Colour base;
    static juce::Colour panel;
    static juce::Colour panelRaised;
    static juce::Colour button;
    static juce::Colour buttonHover;
    static juce::Colour border;
    static juce::Colour graphBackground;
    static juce::Colour clipBackground;
    static juce::Colour grid;
    static juce::Colour beatGrid;
    static juce::Colour accent;
    static juce::Colour accentLight;
    static juce::Colour noteFill;
    static juce::Colour noteLight;
    static juce::Colour noteEdge;
    static juce::Colour pitchLine;
    static juce::Colour playhead;
    static juce::Colour text;
    static juce::Colour textMuted;
    static juce::Colour scrollThumb;
    static juce::Colour heading;
    static juce::Colour selection;
    static juce::Colour primaryButton;
    static juce::Colour warning;
    // Opaque WCAG sRGB contrast, compositing translucent colours over base.
    [[nodiscard]] static double contrastRatio(juce::Colour foreground, juce::Colour background);
    // Prefer semantic text (muted when requested), then a >=4.5:1 fallback.
    [[nodiscard]] static juce::Colour contrastingText(juce::Colour background, bool muted = false);
    [[nodiscard]] static juce::Colour trackColour(std::size_t index);
    static void applyTheme(const juce::String& theme, juce::Colour accentColour,
                           juce::Colour accentLightColour, juce::Colour noteColour);
};

class HachiLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    HachiLookAndFeel();
    void refreshColours();
    void drawAlertBox(juce::Graphics&, juce::AlertWindow&, const juce::Rectangle<int>&,
                      juce::TextLayout&) override;
    void drawButtonBackground(juce::Graphics&, juce::Button&, const juce::Colour&,
                              bool highlighted, bool down) override;
    void drawButtonText(juce::Graphics&, juce::TextButton&, bool highlighted, bool down) override;
    void drawToggleButton(juce::Graphics&, juce::ToggleButton&, bool highlighted, bool down) override;
    void drawLabel(juce::Graphics&, juce::Label&) override;
    void drawPopupMenuItem(juce::Graphics&, const juce::Rectangle<int>&, bool separator,
                           bool active, bool highlighted, bool ticked, bool submenu,
                           const juce::String&, const juce::String&, const juce::Drawable*,
                           const juce::Colour*) override;
    void drawComboBox(juce::Graphics&, int width, int height, bool down,
                      int, int, int, int, juce::ComboBox&) override;
    // The chevron a combo box wears, so a button that opens a menu can wear
    // the same one.  One definition: two hand-drawn arrows that are meant to
    // match will not stay matched.
    static void drawDropdownArrow(juce::Graphics&, juce::Rectangle<int> bounds);
    // How much room on the right the arrow needs, for anything laying out
    // text beside it.
    static constexpr int dropdownArrowWidth = 18;
};
}
