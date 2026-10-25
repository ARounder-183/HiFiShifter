#include "PronunciationCandidateBox.h"
#include "Theme.h"
#include <algorithm>

namespace hachi
{
juce::String PronunciationCandidateBox::tr(const char* zh, const char* en) const
{
    const auto language = strings.getLanguage();
    return juce::String::fromUTF8(language == I18n::Language::zhCN || language == I18n::Language::zhTW ? zh : en);
}

PronunciationCandidateBox::PronunciationCandidateBox(I18n& i18n, std::vector<Candidate> sources,
    std::function<void(const Candidate&)> selected, std::function<void()> dismissed)
    : strings(i18n), candidates(std::move(sources)), chosen(std::move(selected)), cancelled(std::move(dismissed))
{
    setWantsKeyboardFocus(true);
    for (auto* component : std::initializer_list<juce::Component*>{&query, &list, &help, &empty,
         &chooseButton, &previewButton, &cancelButton}) addAndMakeVisible(component);
    query.setTextToShowWhenEmpty(tr("输入发音别名；选择明确的录音来源", "Type an alias; choose its recording source"), Palette::textMuted);
    query.onTextChange = [this] { filter(); };
    // TextEditor receives text/IME handling first. Number keys remain ordinary
    // text here; Alt+1..9 selects while typing, or 1..9 when the list has focus.
    query.onReturnKey = [this] { choose(list.getSelectedRow()); };
    query.onEscapeKey = [this] { cancel(); };
    query.addKeyListener(this);
    list.setModel(this); list.setRowHeight(42); list.setMultipleSelectionEnabled(false);
    list.addKeyListener(this);
    empty.setText(tr("没有匹配的发音。请更改输入或取消。", "No matching pronunciation. Change the query or cancel."), juce::dontSendNotification);
    empty.setJustificationType(juce::Justification::centred);
    empty.setColour(juce::Label::textColourId, Palette::textMuted);
    empty.setInterceptsMouseClicks(false, false);
    help.setText(tr("↑↓ 选择 · Enter 确认 · Esc 取消 · 列表中 1–9 / 输入时 Alt+1–9",
        "↑↓ select · Enter accept · Esc cancel · 1–9 in list / Alt+1–9 while typing"), juce::dontSendNotification);
    help.setColour(juce::Label::textColourId, Palette::textMuted);
    chooseButton.setButtonText(tr("选择", "Choose"));
    previewButton.setButtonText(tr("试听", "Preview"));
    previewButton.setTooltip(tr("由宿主提供原音试听，不启动音频设备。", "Recording preview is provided by the host; no audio device is opened here."));
    cancelButton.setButtonText(strings.text("dialog.cancel"));
    chooseButton.onClick = [this] { choose(list.getSelectedRow()); };
    previewButton.onClick = [this] { preview(); };
    cancelButton.onClick = [this] { cancel(); };
    filter();
    setSize(480, 260);
}

PronunciationCandidateBox::~PronunciationCandidateBox()
{
    query.removeKeyListener(this); list.removeKeyListener(this); list.setModel(nullptr);
}

void PronunciationCandidateBox::setQuery(juce::String text)
{
    query.setText(text, false);
    filter();
}

void PronunciationCandidateBox::filter()
{
    const auto text = query.getText().trim();
    const auto rank = [&](const Candidate& candidate)
    {
        if (text.isEmpty()) return 3;
        if (candidate.alias.equalsIgnoreCase(text)) return 0;
        if (candidate.alias.startsWithIgnoreCase(text)) return 1;
        if (candidate.alias.containsIgnoreCase(text)) return 2;
        return -1;
    };
    visible.clear();
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (rank(candidates[i]) >= 0) visible.push_back(i);
    // Stable original order breaks equal-rank ties. Never collapse aliases or
    // pronunciation IDs: IDs are scoped by audio, not globally unique.
    std::stable_sort(visible.begin(), visible.end(), [&](std::size_t a, std::size_t b)
        { return rank(candidates[a]) < rank(candidates[b]); });
    list.updateContent();
    if (visible.empty()) list.deselectAllRows();
    else { list.selectRow(0); list.scrollToEnsureRowIsOnscreen(0); }
    empty.setVisible(visible.empty());
    chooseButton.setEnabled(!visible.empty()); previewButton.setEnabled(!visible.empty() && static_cast<bool>(onPreview));
    list.repaint();
}

int PronunciationCandidateBox::getNumRows() { return static_cast<int>(visible.size()); }

juce::String PronunciationCandidateBox::candidateLabel(const Candidate& candidate) const
{
    return candidate.alias + juce::String::fromUTF8(" — ") + candidate.audio.getFileName() + " ("
        + candidate.audio.getParentDirectory().getFullPathName() + ") ["
        + juce::String(candidate.startSeconds * 1000, 1) + ", "
        + juce::String(candidate.endSeconds * 1000, 1) + " ms]"
        + (candidate.pronunciationKind == PronunciationKind::transition ? " [T]" : candidate.pronunciationKind == PronunciationKind::breath ? " [B]" : "");
}

void PronunciationCandidateBox::paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected)
{
    if (!juce::isPositiveAndBelow(row, static_cast<int>(visible.size()))) return;
    const auto& candidate = candidates[visible[static_cast<std::size_t>(row)]];
    if (selected) g.fillAll(Palette::accent.withAlpha(0.23f));
    g.setColour(Palette::text); g.setFont(13.0f);
    const auto number = row < 9 ? juce::String(row + 1) + ". " : juce::String();
    const auto kind = candidate.pronunciationKind == PronunciationKind::transition ? " [T]"
        : candidate.pronunciationKind == PronunciationKind::breath ? " [B]" : "";
    g.drawText(number + candidate.alias + kind + "  ["
        + juce::String(candidate.startSeconds * 1000, 1) + ", " + juce::String(candidate.endSeconds * 1000, 1) + " ms]",
        6, 1, width - 12, 20, juce::Justification::centredLeft, true);
    g.setColour(Palette::textMuted); g.setFont(11.0f);
    g.drawText(candidate.audio.getFileName() + " (" + candidate.audio.getParentDirectory().getFullPathName()
        + ")  " + candidate.audioId + "/" + candidate.regionId,
        6, 21, width - 12, std::max(1, height - 22), juce::Justification::centredLeft, true);
}

juce::String PronunciationCandidateBox::getTooltipForRow(int row)
{
    if (!juce::isPositiveAndBelow(row, static_cast<int>(visible.size()))) return {};
    const auto& candidate = candidates[visible[static_cast<std::size_t>(row)]];
    return candidateLabel(candidate) + "\n" + candidate.audioId + "/" + candidate.regionId;
}

void PronunciationCandidateBox::listBoxItemDoubleClicked(int row, const juce::MouseEvent&) { choose(row); }

void PronunciationCandidateBox::choose(int row)
{
    if (finishing || !juce::isPositiveAndBelow(row, static_cast<int>(visible.size()))) return;
    finishing = true;
    const auto result = candidates[visible[static_cast<std::size_t>(row)]];
    const auto callback = chosen;
    // The host may dismiss the CallOutBox/delete us. No member access follows.
    if (callback) callback(result);
}

void PronunciationCandidateBox::cancel()
{
    if (finishing) return;
    finishing = true;
    const auto callback = cancelled;
    if (callback) callback();
}

void PronunciationCandidateBox::preview()
{
    const auto row = list.getSelectedRow();
    if (finishing || !juce::isPositiveAndBelow(row, static_cast<int>(visible.size()))) return;
    const auto result = candidates[visible[static_cast<std::size_t>(row)]];
    const auto callback = onPreview;
    if (callback) callback(result);
}

juce::String PronunciationCandidateBox::diagnosticSelectedId() const
{
    const auto row = list.getSelectedRow();
    return juce::isPositiveAndBelow(row, static_cast<int>(visible.size()))
        ? candidates[visible[static_cast<std::size_t>(row)]].regionId : juce::String();
}

bool PronunciationCandidateBox::keyPressed(const juce::KeyPress& key)
{
    if (key.getKeyCode() == juce::KeyPress::escapeKey) { cancel(); return true; }
    if (key.getKeyCode() == juce::KeyPress::returnKey) { choose(list.getSelectedRow()); return true; }
    if (key.getKeyCode() == juce::KeyPress::upKey || key.getKeyCode() == juce::KeyPress::downKey)
    {
        if (!visible.empty())
        {
            const int direction = key.getKeyCode() == juce::KeyPress::upKey ? -1 : 1;
            const int row = juce::jlimit(0, static_cast<int>(visible.size()) - 1,
                std::max(0, list.getSelectedRow()) + direction);
            list.selectRow(row); list.scrollToEnsureRowIsOnscreen(row);
        }
        return true;
    }
    const auto ch = key.getKeyCode();
    if (ch >= '1' && ch <= '9' && !key.getModifiers().isCommandDown())
    { choose(static_cast<int>(ch - '1')); return true; }
    return false;
}

bool PronunciationCandidateBox::keyPressed(const juce::KeyPress& key, juce::Component* origin)
{
    if (origin == &query)
    {
        const auto code = key.getKeyCode();
        if (code == juce::KeyPress::upKey || code == juce::KeyPress::downKey) return keyPressed(key);
        if (key.getModifiers().isAltDown() && code >= '1' && code <= '9')
            return keyPressed(key);
        return false; // TextEditor retains typing, IME, Enter and Esc handling.
    }
    return keyPressed(key);
}

void PronunciationCandidateBox::paint(juce::Graphics& g) { g.fillAll(Palette::panel); }

void PronunciationCandidateBox::resized()
{
    auto area = getLocalBounds().reduced(8);
    query.setBounds(area.removeFromTop(26)); area.removeFromTop(4);
    auto buttons = area.removeFromBottom(26);
    cancelButton.setBounds(buttons.removeFromRight(80)); buttons.removeFromRight(4);
    chooseButton.setBounds(buttons.removeFromRight(80)); buttons.removeFromRight(4);
    previewButton.setBounds(buttons.removeFromRight(80));
    help.setBounds(area.removeFromBottom(24));
    list.setBounds(area); empty.setBounds(area.reduced(8));
}
}
