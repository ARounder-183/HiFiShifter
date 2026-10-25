#pragma once
#include "I18n.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <cmath>
#include <cstdlib>
#include <optional>

namespace hachi
{
class PanValueDialog final : public juce::AlertWindow
{
public:
    static std::optional<float> parse(const juce::String& input)
    {
        const auto text = input.trim().toStdString();
        char* end = nullptr;
        const auto value = std::strtod(text.c_str(), &end);
        if (text.empty() || end != text.c_str() + text.size()
            || !std::isfinite(value) || value < -1.0 || value > 1.0) return {};
        return static_cast<float>(value);
    }

    PanValueDialog(I18n& strings, const juce::String& title, float pan, juce::Component* owner)
        : juce::AlertWindow(title, strings.text("pan.inputHint"), juce::MessageBoxIconType::NoIcon, owner), initial(pan)
    {
        addTextEditor("pan", juce::String(pan, 6), strings.text("track.pan"));
        originalText = getTextEditorContents("pan");
        addButton(strings.text("dialog.apply"), 1, juce::KeyPress(juce::KeyPress::returnKey));
        addButton(strings.text("dialog.cancel"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
        auto* editor = getTextEditor("pan");
        editor->onTextChange = [this] { getButton(0)->setEnabled(value().has_value()); };
        editor->selectAll();
    }
    std::optional<float> value() const
    {
        const auto text = getTextEditorContents("pan");
        return text == originalText ? std::optional<float>(initial) : parse(text);
    }
private:
    float initial;
    juce::String originalText;
};
}
