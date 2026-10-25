#pragma once
#include "I18n.h"
#include "ProjectModel.h"
#include "Theme.h"

namespace hachi
{
class DiffSingerPronunciationEditor final : public juce::Component
{
public:
    using Readings = std::vector<std::pair<juce::String, juce::String>>;
    using Validate = std::function<void(const Readings&, const juce::String&, bool,
                                       std::function<void(juce::var)>)>;
    DiffSingerPronunciationEditor(const juce::var& notes, const juce::StringArray& shown,
                                 const juce::String& dictionary, Validate validate)
        : onValidate(std::move(validate))
    {
        addAndMakeVisible(help); addAndMakeVisible(view); addAndMakeVisible(dictionaryLabel);
        addAndMakeVisible(userDictionary); addAndMakeVisible(status);
        for (auto* b : {&preview, &apply, &cancel, &reset}) addAndMakeVisible(*b);
        help.setText(I18n::uiText("读音留空时自动转换；修改读音不改变歌词。中文可填拼音；英文可用 en/hello。\n音素覆盖示例：[zh/zh zh/a en/ng]。多音节词后续音符填 +2、+3；普通延音用 +。"), juce::dontSendNotification);
        help.setFont(juce::FontOptions(15));
        view.setViewedComponent(&rows, false);
        view.setScrollBarsShown(true, false);
        if (const auto* values = notes.getArray()) for (const auto& value : *values)
        {
            if (!shown.isEmpty() && !shown.contains(value["id"].toString())) continue;
            auto row = std::make_unique<Row>();
            row->id = value["id"].toString();
            row->lyric.setText(value["lyric"].toString(), juce::dontSendNotification);
            row->lyric.setTooltip(value["lyric"].toString());
            row->reading.setText(value["pronunciation"].toString(), false);
            row->reading.setTextToShowWhenEmpty(I18n::uiText("自动读音"), juce::Colours::grey);
            row->reading.onTextChange = [this] { markChanged(); };
            row->result.setFont(juce::FontOptions(14));
            rows.addAndMakeVisible(row->lyric); rows.addAndMakeVisible(row->reading); rows.addAndMakeVisible(row->result);
            entries.push_back(std::move(row));
        }
        dictionaryLabel.setText(I18n::uiText("轨道词典：一行“歌词 = 读音”，例如 en/read = [r eh d]。修改词典会清除本轨手动音素时长。"), juce::dontSendNotification);
        userDictionary.setMultiLine(true); userDictionary.setReturnKeyStartsNewLine(true);
        userDictionary.setText(dictionary, false);
        userDictionary.onTextChange = [this] { markChanged(); };
        preview.setButtonText(I18n::uiText("预览发音"));
        apply.setButtonText(I18n::uiText("检查并应用"));
        cancel.setButtonText(I18n::uiText("取消"));
        reset.setButtonText(I18n::uiText("清除当前读音覆盖"));
        preview.onClick = [this] { this->validate(false); };
        apply.onClick = [this] { this->validate(true); };
        cancel.onClick = [this] { close(); };
        reset.onClick = [this] { for (auto& row : entries) row->reading.clear(); markChanged(); };
        setSize(950, 610);
    }
    Readings readings() const
    {
        Readings values;
        for (const auto& row : entries) values.emplace_back(row->id, row->reading.getText().trim());
        return values;
    }
    void setReading(const juce::String& id, const juce::String& text)
    { for (auto& row : entries) if (row->id == id) row->reading.setText(text); }
    juce::String previewText(const juce::String& id) const
    { for (const auto& row : entries) if (row->id == id) return row->result.getText(); return {}; }
    void validate(bool commit)
    {
        if (busy || !onValidate) return;
        busy = true;
        preview.setEnabled(false); apply.setEnabled(false); reset.setEnabled(false);
        userDictionary.setEnabled(false);
        for (auto& row : entries) row->reading.setEnabled(false);
        status.setText(I18n::uiText("正在转换并检查发音…"), juce::dontSendNotification);
        const juce::Component::SafePointer<DiffSingerPronunciationEditor> safe(this);
        onValidate(readings(), userDictionary.getText(), commit, [safe, commit](juce::var result)
        {
            if (safe == nullptr) return;
            safe->busy = false; safe->preview.setEnabled(true); safe->apply.setEnabled(true);
            safe->reset.setEnabled(true); safe->userDictionary.setEnabled(true);
            for (auto& row : safe->entries) row->reading.setEnabled(true);
            if (!(bool) result["ok"])
            { safe->status.setText(result["error"].toString(), juce::dontSendNotification); return; }
            if (auto* values = result["pronunciations"].getArray()) for (const auto& value : *values)
                for (auto& row : safe->entries) if (row->id == value["id"].toString())
                {
                    juce::StringArray tokens;
                    if (auto* phones = value["phones"].getArray()) for (const auto& token : *phones) tokens.add(token.toString());
                    const auto text = tokens.joinIntoString("  ") + "   · " + value["source"].toString();
                    row->result.setText(text, juce::dontSendNotification); row->result.setTooltip(text);
                }
            safe->status.setText(I18n::uiText("发音检查通过；英文生词预测及跨语言音素组合请试听确认。"), juce::dontSendNotification);
            if (commit) safe->close();
        });
    }
    void paint(juce::Graphics& g) override { g.fillAll(Palette::panel); }
    void resized() override
    {
        auto area = getLocalBounds().reduced(16);
        help.setBounds(area.removeFromTop(65));
        auto buttons = area.removeFromBottom(36);
        cancel.setBounds(buttons.removeFromRight(90)); buttons.removeFromRight(8);
        apply.setBounds(buttons.removeFromRight(140)); buttons.removeFromRight(8);
        preview.setBounds(buttons.removeFromRight(115)); reset.setBounds(buttons.removeFromLeft(190));
        area.removeFromBottom(8); status.setBounds(area.removeFromBottom(48));
        userDictionary.setBounds(area.removeFromBottom(95));
        dictionaryLabel.setBounds(area.removeFromBottom(30)); area.removeFromBottom(8);
        view.setBounds(area);
        rows.setSize(std::max(500, area.getWidth()-18), std::max(area.getHeight(), (int)entries.size()*72));
        int y = 0;
        for (auto& row : entries)
        {
            row->lyric.setBounds(0, y, 145, 30);
            row->reading.setBounds(155, y, rows.getWidth()-160, 30);
            row->result.setBounds(155, y+32, rows.getWidth()-160, 36);
            y += 72;
        }
    }
private:
    struct Row { juce::String id; juce::Label lyric, result; juce::TextEditor reading; };
    std::vector<std::unique_ptr<Row>> entries;
    juce::Label help, dictionaryLabel, status;
    juce::Component rows;
    juce::Viewport view;
    juce::TextEditor userDictionary;
    juce::TextButton preview, apply, cancel, reset;
    juce::TooltipWindow tooltip {this, 450};
    Validate onValidate;
    bool busy = false;
    void markChanged()
    {
        for (auto& row : entries) row->result.setText({}, juce::dontSendNotification);
        status.setText(I18n::uiText("读音已修改，点击预览检查；应用会再次检查并清除失效的音素时长。"), juce::dontSendNotification);
    }
    void close() { if (auto* window = findParentComponentOfClass<juce::DialogWindow>()) window->exitModalState(0); }
};
}
