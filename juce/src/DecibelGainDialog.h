#pragma once
#include "I18n.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <cmath>
#include <cstdlib>
#include <optional>

namespace hachi
{
class DecibelGainDialog final : public juce::AlertWindow
{
public:
    static std::optional<float> parse(const juce::String& input, float maximumGain)
    {
        auto text=input.trim();
        if(text.endsWithIgnoreCase("dB"))text=text.dropLastCharacters(2).trim();
        if(text.equalsIgnoreCase("-inf")||text==juce::String::fromUTF8("-∞"))return 0.0f;
        const auto bytes=text.toStdString();char* end=nullptr;
        const auto db=std::strtod(bytes.c_str(),&end);
        if(bytes.empty()||end!=bytes.c_str()+bytes.size()||!std::isfinite(db)
            ||db < -120.0||db > 20.0*std::log10(maximumGain)+0.000001)return {};
        return std::min(maximumGain,static_cast<float>(std::pow(10.0,db/20.0)));
    }
    DecibelGainDialog(I18n& strings, const juce::String& title, float gain, float maximumGain, juce::Component* owner)
        : juce::AlertWindow(title,strings.text("gain.inputHint")+"  [-120, "
              +juce::String(20.0*std::log10(maximumGain),4)+"] dB",
              juce::MessageBoxIconType::NoIcon,owner), maximum(maximumGain), initial(gain)
    {
        addTextEditor("gain",gain>0?juce::String(20.0*std::log10(gain),6):"-inf",strings.text("clip.gainDb"));
        originalText=getTextEditorContents("gain");
        addButton(strings.text("dialog.apply"),1,juce::KeyPress(juce::KeyPress::returnKey));
        addButton(strings.text("dialog.cancel"),0,juce::KeyPress(juce::KeyPress::escapeKey));
        auto* editor=getTextEditor("gain");
        editor->onTextChange=[this]{getButton(0)->setEnabled(value().has_value());};
        editor->selectAll();
    }
    std::optional<float> value() const
    {
        const auto text=getTextEditorContents("gain");
        if(text==originalText)return initial; // Opening and applying is lossless.
        return parse(text,maximum);
    }
private:
    float maximum,initial;
    juce::String originalText;
};
}
