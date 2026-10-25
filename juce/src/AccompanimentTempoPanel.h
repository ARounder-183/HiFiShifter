#pragma once
#include "AccompanimentTempo.h"
#include "I18n.h"
#include "Theme.h"
#include <juce_gui_extra/juce_gui_extra.h>

namespace hachi
{
class AccompanimentTempoPanel final : public juce::Component, private juce::Timer
{
public:
    AccompanimentTempoPanel(const I18n& text, const ClipData& clip)
        : strings(text), task(clip)
    {
        caption.setText(clip.parts.empty() ? clip.sourceFile.getFileName() : strings.text("tempo.mergedAudio"), juce::dontSendNotification);
        caption.setFont(juce::FontOptions(16.0f)); caption.setJustificationType(juce::Justification::centredLeft);
        value.setFont(juce::FontOptions(30.0f)); value.setJustificationType(juce::Justification::centred);
        help.setMultiLine(true); help.setReadOnly(true); help.setCaretVisible(false);
        help.setScrollbarsShown(true); help.setColour(juce::TextEditor::backgroundColourId, Palette::panel);
        help.setText(strings.text("tempo.readOnly"));
        button.setButtonText(strings.text("dialog.cancel"));
        button.onClick = [this] { task.cancel(); close(); };
        for (auto* component : std::initializer_list<juce::Component*>{&caption, &value, &help, &button}) addAndMakeVisible(component);
        value.setText(strings.text("tempo.preparing"), juce::dontSendNotification);
        setSize(510, 280); task.startThread(); startTimer(80);
    }
    ~AccompanimentTempoPanel() override { stopTimer(); task.cancel(); }
    bool finished() const { return task.finished(); }
    juce::var result() const { return task.result(); }
    juce::String displayedValue() const { return value.getText(); }
    void refreshResult() { timerCallback(); }
    void resized() override
    {
        auto b = getLocalBounds().reduced(20);
        caption.setBounds(b.removeFromTop(28)); b.removeFromTop(12);
        value.setBounds(b.removeFromTop(54)); b.removeFromTop(12);
        button.setBounds(b.removeFromBottom(34).removeFromRight(105)); b.removeFromBottom(12);
        help.setBounds(b);
    }
    void paint(juce::Graphics& g) override { g.fillAll(Palette::panel); }
private:
    void close() { if (auto* window = findParentComponentOfClass<juce::DialogWindow>()) window->setVisible(false); }
    void timerCallback() override
    {
        if (!task.finished())
        {
            value.setText(strings.text(task.stage() == 0 ? "tempo.preparing" : "tempo.running"), juce::dontSendNotification);
            return;
        }
        stopTimer(); const auto result = task.result();
        if ((bool)result["ok"])
        {
            value.setText(juce::String((double)result["bpm"], 2) + " BPM", juce::dontSendNotification);
            auto info = strings.text("tempo.estimateHint");
            if ((bool)result["variable_tempo"]) info += "\n" + strings.text("tempo.variable");
            info += "\n" + strings.text("tempo.readOnly"); help.setText(info);
        }
        else
        {
            value.setText(strings.text("tempo.failed"), juce::dontSendNotification);
            const auto code = result["code"].toString();
            const auto key = code == "missing_model" ? "tempo.missingModel"
                : code == "missing_runtime" ? "tempo.missingRuntime"
                : code == "no_beats" || code == "too_short" ? "tempo.noBeats"
                : code == "cancelled" ? "tempo.cancelled" : "tempo.workerError";
            help.setText(strings.text(key) + (result["error"].toString().isNotEmpty() ? "\n" + result["error"].toString() : juce::String()));
        }
        button.setButtonText(strings.text("tempo.close"));
    }
    I18n strings;
    AccompanimentTempoTask task;
    juce::Label caption, value;
    juce::TextEditor help;
    juce::TextButton button;
};
}
