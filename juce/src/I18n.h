#pragma once

#include <juce_core/juce_core.h>

namespace hachi
{
class I18n
{
public:
    enum class Language { zhCN, zhTW, jaJP, koKR, enUS };

    I18n();
    void setLanguage(Language languageToUse);
    // Legacy UI literals only; never use for model data, IDs or user input.
    [[nodiscard]] static juce::String uiText(const char* sourceLiteral);
    [[nodiscard]] Language getLanguage() const { return language; }
    [[nodiscard]] juce::String text(const juce::String& key) const;
    [[nodiscard]] juce::String text(const char* utf8Key) const { return text(juce::String::fromUTF8(utf8Key)); }

private:
    Language language = Language::zhCN;
};
}

