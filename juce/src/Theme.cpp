#include "Theme.h"
#include <cmath>

namespace hachi
{
juce::Colour Palette::background  { 0xff353535 };
juce::Colour Palette::base        { 0xff2d2d2d };
juce::Colour Palette::panel       { 0xff2a2a2a };
juce::Colour Palette::panelRaised { 0xff404040 };
juce::Colour Palette::button      { 0xff3d3d3d };
juce::Colour Palette::buttonHover { 0xff484848 };
juce::Colour Palette::border      { 0xff505050 };
juce::Colour Palette::graphBackground { 0xff232323 };
juce::Colour Palette::clipBackground  { 0xff343434 };
juce::Colour Palette::grid        { 0xff373737 };
juce::Colour Palette::beatGrid    { 0xff4a4a4a };
juce::Colour Palette::accent      { 0xff7f69ca };
juce::Colour Palette::accentLight { 0xffcbcbfa };
// Initial values retain the existing dark palette. applyTheme supplies the
// lavender light-note treatment; yellow is reserved for warnings/track accents.
juce::Colour Palette::noteFill    { 0xff269b9b };
juce::Colour Palette::noteLight   { 0xff8be0d5 };
juce::Colour Palette::noteEdge    { 0xff4fc3b5 };
juce::Colour Palette::pitchLine   { 0xfff4f4f4 };
juce::Colour Palette::playhead    { 0xfff05a5a };
juce::Colour Palette::text        { 0xffd0d0d0 };
juce::Colour Palette::textMuted   { 0xff909090 };
juce::Colour Palette::scrollThumb { 0xff555555 };
juce::Colour Palette::heading     { 0xffd0d0d0 };
juce::Colour Palette::selection   { 0xff7f69ca };
juce::Colour Palette::primaryButton { 0xff7f69ca };
juce::Colour Palette::warning     { 0xfff4c000 };

namespace
{
double luminance(juce::Colour colour)
{
    const auto linear = [](double value)
    {
        return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * linear(colour.getRed() / 255.0)
         + 0.7152 * linear(colour.getGreen() / 255.0)
         + 0.0722 * linear(colour.getBlue() / 255.0);
}

juce::Colour buttonSurface(const juce::Button& button, bool highlighted, bool down)
{
    if (!button.isEnabled()) return Palette::panel;
    auto colour = button.getToggleState() ? Palette::primaryButton : Palette::button;
    if (highlighted) colour = button.getToggleState() ? colour.brighter(0.12f) : Palette::buttonHover;
    return down ? colour.darker(0.15f) : colour;
}
}

double Palette::contrastRatio(juce::Colour foreground, juce::Colour backgroundColour)
{
    backgroundColour = base.withAlpha(1.0f).overlaidWith(backgroundColour);
    foreground = backgroundColour.overlaidWith(foreground);
    const auto a = luminance(foreground), b = luminance(backgroundColour);
    return (juce::jmax(a, b) + 0.05) / (juce::jmin(a, b) + 0.05);
}

juce::Colour Palette::contrastingText(juce::Colour backgroundColour, bool muted)
{
    // Do not use Colour::contrasting(amount): its grey interpolation does not
    // guarantee text contrast (off-white on the default purple is only 4.386).
    const juce::Colour candidates[] { muted ? textMuted : text, heading,
                                     juce::Colour(0xfffffffb), juce::Colours::black,
                                     juce::Colours::white };
    for (auto candidate : candidates)
        if (contrastRatio(candidate, backgroundColour) >= 4.5)
            return candidate.withAlpha(1.0f);
    return juce::Colours::black; // Black/white guarantee >=4.5 on opaque sRGB.
}

juce::Colour Palette::trackColour(std::size_t index)
{
    static const juce::Colour colours[] {
        juce::Colour(0xff7f69ca), juce::Colour(0xffcbcbfa), juce::Colour(0xfff4c000),
        juce::Colour(0xff9b8bdd), juce::Colour(0xffdedcff), juce::Colour(0xffffd94f)
    };
    return colours[index % 6];
}

void Palette::applyTheme(const juce::String& theme, juce::Colour accentColour,
                         juce::Colour accentLightColour, juce::Colour noteColour)
{
    const auto light = theme == "light";
    // Tinted surfaces stay light enough for muted #637097 text. Reserve the
    // stronger lavender for selection/notes, where deep text is required.
    background = light ? juce::Colour(0xfffffffb) : juce::Colour(0xff353535);
    base = light ? juce::Colour(0xfffffffb) : juce::Colour(0xff2d2d2d);
    panel = light ? juce::Colour(0xfff8f8fb) : juce::Colour(0xff2a2a2a);
    panelRaised = light ? juce::Colour(0xfff7f7fb) : juce::Colour(0xff404040);
    button = light ? juce::Colour(0xffcbcbfa) : juce::Colour(0xff3d3d3d);
    buttonHover = light ? juce::Colour(0xffdedefa) : juce::Colour(0xff484848);
    border = light ? juce::Colour(0xffb5bad2) : juce::Colour(0xff505050);
    graphBackground = light ? juce::Colour(0xfffffffb) : juce::Colour(0xff232323);
    clipBackground = light ? juce::Colour(0xfff7f7fb) : juce::Colour(0xff343434);
    grid = light ? juce::Colour(0xffe7e8f1) : juce::Colour(0xff373737);
    beatGrid = light ? juce::Colour(0xffb5bad2) : juce::Colour(0xff4a4a4a);
    text = light ? juce::Colour(0xff524984) : juce::Colour(0xffd0d0d0);
    textMuted = light ? juce::Colour(0xff637097) : juce::Colour(0xff909090);
    heading = light ? juce::Colour(0xff42168c) : text;
    scrollThumb = light ? juce::Colour(0xff637097) : juce::Colour(0xff555555);
    pitchLine = light ? juce::Colour(0xff42168c) : juce::Colour(0xfff4f4f4);
    playhead = light ? juce::Colour(0xff42168c) : juce::Colour(0xfff05a5a);
    accent = accentColour.withAlpha(1.0f);
    accentLight = accentLightColour.withAlpha(1.0f);
    selection = light ? accentLight : accent;
    // Existing custom accent preferences remain authoritative. Only the stock
    // light accent gets the deep primary-button treatment, not white on #7F69CA.
    primaryButton = light && accent == juce::Colour(0xff7f69ca) ? heading : accent;
    // Keep note colours tied to the shared editor style.  Older preference
    // files may contain the retired yellow note colour, so do not reapply it.
    juce::ignoreUnused(noteColour);
    noteFill = light ? juce::Colour(0xffcbcbfa) : juce::Colour(0xff269b9b);
    noteLight = light ? juce::Colour(0xff7f69ca) : juce::Colour(0xff8be0d5);
    noteEdge = light ? juce::Colour(0xff42168c) : juce::Colour(0xff4fc3b5);
}

HachiLookAndFeel::HachiLookAndFeel()
{
    setUsingNativeAlertWindows(false);
    refreshColours();
}

void HachiLookAndFeel::refreshColours()
{
    const auto selectionText = Palette::contrastRatio(Palette::heading, Palette::selection) >= 4.5
        ? Palette::heading : Palette::contrastingText(Palette::selection);
    setColour(juce::ResizableWindow::backgroundColourId, Palette::background);
    setColour(juce::Label::textColourId, Palette::text);
    setColour(juce::TextButton::textColourOffId, Palette::text);
    setColour(juce::TextButton::buttonColourId, Palette::button);
    setColour(juce::TextButton::buttonOnColourId, Palette::primaryButton);
    setColour(juce::TextButton::textColourOnId, Palette::contrastingText(Palette::primaryButton));
    setColour(juce::Label::textWhenEditingColourId, Palette::text);
    setColour(juce::Label::backgroundWhenEditingColourId, Palette::base);
    setColour(juce::ToggleButton::textColourId, Palette::text);
    setColour(juce::ToggleButton::tickColourId, Palette::heading);
    setColour(juce::ToggleButton::tickDisabledColourId, Palette::contrastingText(Palette::panel, true));
    setColour(juce::ComboBox::backgroundColourId, Palette::base);
    setColour(juce::ComboBox::textColourId, Palette::text);
    setColour(juce::ComboBox::outlineColourId, Palette::border);
    setColour(juce::ComboBox::arrowColourId, Palette::text);
    // Text editors were left on the JUCE default (white field, black text),
    // which reads as an out-of-theme box in either mode.  Tie them to the
    // Palette so they follow dark/light like everything else.
    setColour(juce::TextEditor::backgroundColourId, Palette::base);
    setColour(juce::TextEditor::textColourId, Palette::text);
    setColour(juce::TextEditor::highlightedTextColourId, selectionText);
    setColour(juce::TextEditor::highlightColourId, Palette::selection);
    setColour(juce::TextEditor::outlineColourId, Palette::border);
    setColour(juce::TextEditor::focusedOutlineColourId, Palette::accent);
    setColour(juce::CaretComponent::caretColourId, Palette::text);
    setColour(juce::PopupMenu::backgroundColourId, Palette::panelRaised);
    setColour(juce::PopupMenu::textColourId, Palette::text);
    setColour(juce::PopupMenu::headerTextColourId, Palette::heading);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, Palette::selection);
    setColour(juce::PopupMenu::highlightedTextColourId, selectionText);
    setColour(juce::TreeView::backgroundColourId, Palette::panel);
    setColour(juce::TreeView::linesColourId, Palette::border);
    setColour(juce::TreeView::selectedItemBackgroundColourId, Palette::selection);
    setColour(juce::TreeView::oddItemsColourId, Palette::panel);
    setColour(juce::TreeView::evenItemsColourId, Palette::base);
    setColour(juce::ListBox::backgroundColourId, Palette::base);
    setColour(juce::ListBox::outlineColourId, Palette::border);
    setColour(juce::ListBox::textColourId, Palette::text);
    setColour(juce::TableHeaderComponent::backgroundColourId, Palette::panelRaised);
    setColour(juce::TableHeaderComponent::textColourId, Palette::heading);
    setColour(juce::TableHeaderComponent::outlineColourId, Palette::border);
    setColour(juce::TableHeaderComponent::highlightColourId, Palette::selection);
    setColour(juce::GroupComponent::textColourId, Palette::heading);
    setColour(juce::GroupComponent::outlineColourId, Palette::border);
    setColour(juce::PropertyComponent::backgroundColourId, Palette::panel);
    setColour(juce::PropertyComponent::labelTextColourId, Palette::text);
    setColour(juce::HyperlinkButton::textColourId, Palette::heading);
    setColour(juce::Slider::backgroundColourId, Palette::base);
    setColour(juce::Slider::trackColourId, Palette::accent);
    setColour(juce::Slider::thumbColourId, Palette::heading);
    setColour(juce::Slider::textBoxBackgroundColourId, Palette::base);
    setColour(juce::Slider::textBoxTextColourId, Palette::text);
    setColour(juce::Slider::textBoxOutlineColourId, Palette::border);
    setColour(juce::Slider::textBoxHighlightColourId, Palette::selection);
    setColour(juce::TooltipWindow::backgroundColourId, Palette::panelRaised);
    setColour(juce::TooltipWindow::textColourId, Palette::text);
    setColour(juce::TooltipWindow::outlineColourId, Palette::border);
    setColour(juce::AlertWindow::backgroundColourId, Palette::panel);
    setColour(juce::AlertWindow::textColourId, Palette::text);
    setColour(juce::AlertWindow::outlineColourId, Palette::border);
    setColour(juce::ScrollBar::backgroundColourId, Palette::base);
    setColour(juce::ScrollBar::trackColourId, Palette::base);
    setColour(juce::ScrollBar::thumbColourId, Palette::scrollThumb);
}

void HachiLookAndFeel::drawAlertBox(juce::Graphics& g, juce::AlertWindow& alert,
                                   const juce::Rectangle<int>& textArea, juce::TextLayout& layout)
{
    // Preserve dark alerts and non-warning icon layouts. Light warnings reserve
    // yellow for the warning symbol, with a deep readable exclamation mark.
    if (Palette::background != juce::Colour(0xfffffffb)
        || alert.getAlertType() != juce::MessageBoxIconType::WarningIcon)
    {
        LookAndFeel_V4::drawAlertBox(g, alert, textArea, layout);
        return;
    }
    g.fillAll(alert.findColour(juce::AlertWindow::backgroundColourId));
    const juce::Rectangle<float> icon(12.0f, static_cast<float>(textArea.getY()), 56.0f, 50.0f);
    juce::Path triangle;
    triangle.addTriangle(icon.getCentreX(), icon.getY(), icon.getRight(), icon.getBottom(),
                         icon.getX(), icon.getBottom());
    g.setColour(Palette::warning);
    g.fillPath(triangle);
    g.setColour(Palette::heading);
    g.strokePath(triangle, juce::PathStrokeType(1.5f));
    g.setFont(30.0f);
    g.drawText("!", icon.withTrimmedTop(10.0f), juce::Justification::centred);
    layout.draw(g, textArea.withTrimmedLeft(80).toFloat());
    g.setColour(alert.findColour(juce::AlertWindow::outlineColourId));
    g.drawRect(alert.getLocalBounds());
}

void HachiLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                            const juce::Colour&, bool highlighted, bool down)
{
    g.setColour(buttonSurface(button, highlighted, down));
    const auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);
    g.fillRoundedRectangle(bounds, 4.0f);
    // Disabled controls retain opaque, readable labels; the quiet surface and
    // outline distinguish them instead of multiplying text alpha twice.
    if (!button.isEnabled() || button.hasKeyboardFocus(true))
    {
        g.setColour(button.isEnabled() ? Palette::accent : Palette::border);
        g.drawRoundedRectangle(bounds, 4.0f, button.isEnabled() ? 2.0f : 1.0f);
    }
}

void HachiLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button, bool highlighted, bool down)
{
    const auto surface = buttonSurface(button, highlighted, down);
    auto foreground = Palette::contrastingText(surface, !button.isEnabled());
    const auto colourId = button.getToggleState() ? juce::TextButton::textColourOnId : juce::TextButton::textColourOffId;
    if (button.isEnabled() && button.isColourSpecified(colourId)
        && Palette::contrastRatio(button.findColour(colourId), surface) >= 4.5)
        foreground = button.findColour(colourId).withAlpha(1.0f);
    const auto id = button.getComponentID();
    if (button.getButtonText() == "+" || button.getButtonText() == "-")
    {
        // Draw zoom symbols geometrically so narrow buttons never ellipsize.
        const auto bounds = button.getLocalBounds().toFloat();
        const auto centre = bounds.getCentre();
        const auto radius = juce::jmin(6.0f, juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.25f);
        g.setColour(foreground);
        g.drawLine(centre.x - radius, centre.y, centre.x + radius, centre.y, 1.8f);
        if (button.getButtonText() == "+")
            g.drawLine(centre.x, centre.y - radius, centre.x, centre.y + radius, 1.8f);
        return;
    }
    if (!id.startsWith("icon."))
    {
        g.setFont(getTextButtonFont(button, button.getHeight()));
        g.setColour(foreground);
        const auto indent = juce::jmin(4, button.getHeight() / 4);
        g.drawFittedText(button.getButtonText(), button.getLocalBounds().reduced(indent, 1),
                         juce::Justification::centred, 1);
        return;
    }

    auto bounds = button.getLocalBounds().toFloat().reduced(6.0f);
    // User-approved Tabler Icons v3.34.1 (MIT, Paweł Kuna), shared 24px grid.
    // See juce/assets/toolbar-icons/LICENSE.txt. Cache parsing; tint copies.
    static const auto toolIcons = []
    {
        std::array<std::unique_ptr<juce::Drawable>, 7> icons;
        const std::array<const char*, 7> shapes {{
            "<path d='M7.904 17.563a1.2 1.2 0 0 0 2.228 .308l2.09 -3.093l4.907 4.907a1.067 1.067 0 0 0 1.509 0l1.047 -1.047a1.067 1.067 0 0 0 0 -1.509l-4.907 -4.907l3.113 -2.09a1.2 1.2 0 0 0 -.309 -2.228l-13.582 -3.904l3.904 13.563z'/>",
            "<path d='M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4'/><path d='M13.5 6.5l4 4'/>",
            "<path d='M6 18m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0'/><path d='M18 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0'/><path d='M7.5 16.5l9 -9'/>",
            "<path d='M3 3m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v2a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z'/><path d='M17 17m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v2a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z'/><path d='M7 5l7 0'/><path d='M10 19l7 0'/><path d='M9 19m-1 0a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'/><path d='M15 5m-1 0a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'/><path d='M7 5.5a5 6.5 0 0 1 5 6.5a5 6.5 0 0 0 5 6.5'/>",
            "<path d='M7 10h3v-3l-3.5 -3.5a6 6 0 0 1 8 8l6 6a2 2 0 0 1 -3 3l-6 -6a6 6 0 0 1 -8 -8l3.5 3.5'/>",
            "<path d='M9 15l6 -6'/><path d='M11 6l.463 -.536a5 5 0 0 1 7.071 7.072l-.534 .464'/><path d='M13 18l-.397 .534a5.068 5.068 0 0 1 -7.127 0a4.972 4.972 0 0 1 0 -7.071l.524 -.463'/>",
            "<path d='M8 5v10a1 1 0 0 0 1 1h10'/><path d='M5 8h10a1 1 0 0 1 1 1v10'/>"
        }};
        for (std::size_t index = 0; index < icons.size(); ++index)
        {
            const auto svg = juce::String("<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'><g fill='none' stroke='#000000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>")
                + shapes[index] + "</g></svg>";
            if (const auto xml = juce::parseXML(svg))
                icons[index] = juce::Drawable::createFromSVG(*xml);
        }
        return icons;
    }();
    const std::array<const char*, 7> toolIds {{ "icon.pointer", "icon.draw",
        "icon.line", "icon.points", "icon.wrench", "icon.connect", "icon.trim" }};
    for (std::size_t index = 0; index < toolIds.size(); ++index)
        if (id == toolIds[index] && toolIcons[index])
        {
            auto icon = toolIcons[index]->createCopy();
            icon->replaceColour(juce::Colours::black, foreground);
            icon->drawWithin(g, button.getLocalBounds().toFloat().reduced(4.0f),
                juce::RectanglePlacement::centred, 1.0f);
            return;
        }
    g.setColour(foreground);
    juce::Path path;
    if (id == "icon.play")
        path.addTriangle(bounds.getX() + 2.0f, bounds.getY(), bounds.getRight(), bounds.getCentreY(),
                         bounds.getX() + 2.0f, bounds.getBottom());
    else if (id == "icon.pause")
    {
        const auto barWidth = juce::jmax(2.0f, bounds.getWidth() * 0.28f);
        path.addRoundedRectangle(bounds.getX() + 1.0f, bounds.getY(), barWidth, bounds.getHeight(), 1.0f);
        path.addRoundedRectangle(bounds.getRight() - barWidth - 1.0f, bounds.getY(),
                                 barWidth, bounds.getHeight(), 1.0f);
    }
    else if (id == "icon.stop")
        path.addRectangle(bounds.reduced(2.0f));
    else if (id == "icon.open")
    {
        path.addRoundedRectangle(bounds.withTrimmedTop(4.0f), 2.0f);
        path.addRectangle(bounds.getX() + 2.0f, bounds.getY() + 1.0f, bounds.getWidth() * 0.42f, 5.0f);
    }
    else if (id == "icon.save")
    {
        path.addRoundedRectangle(bounds, 1.5f);
        path.addRectangle(bounds.reduced(3.0f).withHeight(5.0f));
        path.addRectangle(bounds.reduced(4.0f).withTrimmedTop(9.0f));
    }
    else if (id == "icon.pointer")
    {
        // Cursor arrow with a straight top edge and angular tip.
        path.startNewSubPath(bounds.getX() + 1.5f, bounds.getY() + 1.5f);
        path.lineTo(bounds.getX() + 1.5f, bounds.getBottom() - 4.5f);
        path.lineTo(bounds.getX() + 6.0f, bounds.getBottom() - 9.0f);
        path.lineTo(bounds.getX() + 9.0f, bounds.getBottom() - 1.5f);
        path.lineTo(bounds.getX() + 11.5f, bounds.getBottom() - 4.0f);
        path.lineTo(bounds.getX() + 8.5f, bounds.getBottom() - 11.0f);
        path.lineTo(bounds.getRight() - 2.5f, bounds.getBottom() - 9.5f);
        path.closeSubPath();
    }
    else if (id == "icon.draw")
    {
        // Pencil: diagonal body, pointed tip, visible lead.
        path.startNewSubPath(bounds.getX() + 1.5f, bounds.getBottom() - 3.0f);
        path.lineTo(bounds.getX() + 2.5f, bounds.getBottom() - 6.0f);
        path.lineTo(bounds.getRight() - 2.5f, bounds.getY() + 2.0f);
        path.lineTo(bounds.getRight() - 4.5f, bounds.getY() + 0.5f);
        path.lineTo(bounds.getX() + 5.5f, bounds.getBottom() - 7.5f);
        path.lineTo(bounds.getX() + 3.0f, bounds.getBottom() - 5.0f);
        path.lineTo(bounds.getX() + 1.5f, bounds.getBottom() - 3.0f);
        path.closeSubPath();
    }
    else if (id == "icon.line")
    {
        // Diagonal line with small start/end caps like a straight-line tool.
        path.startNewSubPath(bounds.getX() + 1.0f, bounds.getBottom() - 1.0f);
        path.lineTo(bounds.getRight() - 1.0f, bounds.getY() + 1.0f);
    }
    else if (id == "icon.points")
    {
        path.startNewSubPath(bounds.getX() + 1.0f, bounds.getBottom() - 2.0f);
        path.lineTo(bounds.getCentreX(), bounds.getCentreY() + 1.0f);
        path.lineTo(bounds.getRight() - 1.0f, bounds.getY() + 2.0f);
        g.strokePath(path, juce::PathStrokeType(1.6f, juce::PathStrokeType::curved));
        g.fillEllipse(bounds.getX() - 1.0f, bounds.getBottom() - 4.0f, 4.0f, 4.0f);
        g.fillEllipse(bounds.getCentreX() - 2.0f, bounds.getCentreY() - 1.0f, 4.0f, 4.0f);
        g.fillEllipse(bounds.getRight() - 3.0f, bounds.getY(), 4.0f, 4.0f);
        return;
    }
    else if (id == "icon.wrench")
    {
        // Open-end wrench: C-shaped jaw and shaft with an angled end.
        path.startNewSubPath(bounds.getX() + 2.0f, bounds.getCentreY() - 3.5f);
        path.lineTo(bounds.getX() + 2.0f, bounds.getY() + 2.0f);
        path.lineTo(bounds.getX() + 6.5f, bounds.getY() + 2.0f);
        path.lineTo(bounds.getX() + 6.5f, bounds.getCentreY() - 4.0f);
        path.closeSubPath();
        path.startNewSubPath(bounds.getX() + 2.0f, bounds.getCentreY() + 3.5f);
        path.lineTo(bounds.getX() + 2.0f, bounds.getBottom() - 2.0f);
        path.lineTo(bounds.getX() + 6.5f, bounds.getBottom() - 2.0f);
        path.lineTo(bounds.getX() + 6.5f, bounds.getCentreY() + 4.0f);
        path.closeSubPath();
        path.startNewSubPath(bounds.getX() + 6.0f, bounds.getCentreY() + 4.0f);
        path.lineTo(bounds.getRight() - 1.0f, bounds.getCentreY() - 2.0f);
    }
    else if (id == "icon.collapse" || id == "icon.expand")
    {
        // A chevron pointing the way the arrangement will go.
        const auto rise = id == "icon.collapse" ? -3.0f : 3.0f;
        path.startNewSubPath(bounds.getX() + 1.0f, bounds.getCentreY() - rise);
        path.lineTo(bounds.getCentreX(), bounds.getCentreY() + rise);
        path.lineTo(bounds.getRight() - 1.0f, bounds.getCentreY() - rise);
    }
    else if (id == "icon.connect")
    {
        path.addEllipse(bounds.getX(), bounds.getCentreY() - 3.0f, 6.0f, 6.0f);
        path.addEllipse(bounds.getRight() - 6.0f, bounds.getCentreY() - 3.0f, 6.0f, 6.0f);
        path.startNewSubPath(bounds.getX() + 5.0f, bounds.getCentreY());
        path.lineTo(bounds.getRight() - 5.0f, bounds.getCentreY());
    }
    else
    {
        g.setFont(10.0f);
        g.drawText(id == "icon.audio" ? "A+" : "M+", button.getLocalBounds(), juce::Justification::centred);
        return;
    }
    if (id == "icon.line" || id == "icon.wrench" || id == "icon.connect"
        || id == "icon.collapse" || id == "icon.expand")
        g.strokePath(path, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved,
                                                juce::PathStrokeType::rounded));
    else
        g.fillPath(path);
}

void HachiLookAndFeel::drawLabel(juce::Graphics& g, juce::Label& label)
{
    const auto labelBackground = label.findColour(juce::Label::backgroundColourId);
    g.fillAll(labelBackground);
    auto background = Palette::panel;
    if (auto* combo = dynamic_cast<juce::ComboBox*>(label.getParentComponent()))
        background = combo->isEnabled() ? combo->findColour(juce::ComboBox::backgroundColourId) : Palette::panel;
    else if (auto* parent = label.getParentComponent())
        background = parent->findColour(juce::ResizableWindow::backgroundColourId, true);
    background = background.overlaidWith(labelBackground);
    // JUCE deliberately paints accessibility-only labels with transparent text.
    // Contrast repair must never make those duplicate visual text/layout again.
    if (!label.isBeingEdited() && !label.findColour(juce::Label::textColourId).isTransparent())
    {
        g.setFont(getLabelFont(label));
        auto foreground = label.findColour(juce::Label::textColourId);
        // ComboBox creates its child Label with an explicit copied text colour;
        // refreshColours alone cannot update that cached colour after a switch.
        if (!label.isEnabled() || Palette::contrastRatio(foreground, background) < 4.5)
            foreground = Palette::contrastingText(background, !label.isEnabled());
        g.setColour(foreground);
        const auto area = label.getBorderSize().subtractedFrom(label.getLocalBounds());
        g.drawFittedText(label.getText(), area, label.getJustificationType(),
                         juce::jmax(1, static_cast<int>(area.getHeight() / g.getCurrentFont().getHeight())),
                         label.getMinimumHorizontalScale());
    }
    g.setColour(label.findColour(juce::Label::outlineColourId));
    g.drawRect(label.getLocalBounds());
}

void HachiLookAndFeel::drawPopupMenuItem(juce::Graphics& g, const juce::Rectangle<int>& area,
    bool separator, bool active, bool highlighted, bool ticked, bool submenu,
    const juce::String& text, const juce::String& shortcut, const juce::Drawable* icon,
    const juce::Colour* textColour)
{
    // Disabled state must remain visibly grey. Accessibility contrast repair
    // for active labels can turn muted grey into the same bright white as an
    // enabled item; do not apply that repair to this inactive semantic state.
    const auto disabledText = Palette::textMuted.withAlpha(1.0f);
    // JUCE otherwise fades disabled menu entries to ~30% opacity. Keep all
    // layout/tick/submenu machinery, but use opaque muted text and no highlight.
    LookAndFeel_V4::drawPopupMenuItem(g, area, separator, true, active && highlighted,
        ticked, submenu, text, shortcut, icon, active ? textColour : &disabledText);
}

void HachiLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                       bool highlighted, bool down)
{
    const auto size = juce::jmin(18.0f, button.getHeight() * 0.75f);
    const juce::Rectangle<float> box(3.0f, (button.getHeight() - size) * 0.5f, size, size);
    const auto surface = !button.isEnabled() ? Palette::panel
                       : button.getToggleState() ? Palette::selection : Palette::base;
    g.setColour(surface);
    g.fillRoundedRectangle(box, 3.0f);
    g.setColour(button.isEnabled() ? (highlighted || down || button.hasKeyboardFocus(true)
                                        ? Palette::accent : Palette::heading) : Palette::border);
    g.drawRoundedRectangle(box, 3.0f, 1.4f);
    if (button.getToggleState())
    {
        g.setColour(Palette::contrastingText(surface, !button.isEnabled()));
        juce::Path tick;
        tick.startNewSubPath(box.getX() + size * 0.2f, box.getY() + size * 0.5f);
        tick.lineTo(box.getX() + size * 0.43f, box.getY() + size * 0.73f);
        tick.lineTo(box.getX() + size * 0.8f, box.getY() + size * 0.26f);
        g.strokePath(tick, juce::PathStrokeType(2.0f));
    }
    g.setColour(button.isEnabled() ? button.findColour(juce::ToggleButton::textColourId)
                                 : Palette::contrastingText(Palette::panel, true));
    g.setFont(juce::jmin(15.0f, button.getHeight() * 0.75f));
    g.drawFittedText(button.getButtonText(), button.getLocalBounds().withTrimmedLeft(static_cast<int>(size) + 9),
                     juce::Justification::centredLeft, 2);
}

void HachiLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool,
                                    int, int, int, int, juce::ComboBox& combo)
{
    auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
    g.setColour(combo.isEnabled() ? combo.findColour(juce::ComboBox::backgroundColourId) : Palette::panel);
    g.fillRoundedRectangle(bounds.reduced(1.0f), 4.0f);
    g.setColour(combo.hasKeyboardFocus(true) ? Palette::accent : Palette::border);
    g.drawRoundedRectangle(bounds.reduced(1.0f), 4.0f, combo.hasKeyboardFocus(true) ? 2.0f : 1.0f);
    drawDropdownArrow(g, { 0, 0, width, height });
}

void HachiLookAndFeel::drawDropdownArrow(juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour(Palette::contrastingText(Palette::base));
    const auto x = static_cast<float>(bounds.getRight() - 14);
    const auto y = static_cast<float>(bounds.getCentreY());
    juce::Path arrow;
    arrow.startNewSubPath(x - 4.0f, y - 2.0f);
    arrow.lineTo(x, y + 2.0f);
    arrow.lineTo(x + 4.0f, y - 2.0f);
    g.strokePath(arrow, juce::PathStrokeType(1.5f));
}
}
