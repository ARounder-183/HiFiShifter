#include "MaterialAnnotationEditor.h"
#include "MaterialAnnotationSession.h"
#include "Theme.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace hachi
{
namespace
{
char roleCode(NativeSegmentRole role)
{
    switch (role)
    {
        case NativeSegmentRole::consonant: return 'C';
        case NativeSegmentRole::vowel: return 'V';
        case NativeSegmentRole::transition: return 'T';
        case NativeSegmentRole::silence: return 'S';
        case NativeSegmentRole::breath: return 'B';
        case NativeSegmentRole::noise: return 'N';
        case NativeSegmentRole::ending: return 'E';
        default: return 'U';
    }
}
bool parseRole(const juce::String& value, NativeSegmentRole& role)
{
    const auto code = value.trim().toUpperCase();
    const char* codes = "UCVTSBNE";
    if (code.length() != 1) return false;
    for (int i = 0; codes[i] != 0; ++i)
        if (code[0] == codes[i])
        {
            const NativeSegmentRole roles[]{NativeSegmentRole::unknown, NativeSegmentRole::consonant,
                NativeSegmentRole::vowel, NativeSegmentRole::transition, NativeSegmentRole::silence,
                NativeSegmentRole::breath, NativeSegmentRole::noise, NativeSegmentRole::ending};
            role = roles[i]; return true;
        }
    return false;
}
bool parseNumber(const juce::String& text, double& value)
{
    const auto trimmed = text.trim();
    if (trimmed.isEmpty()) return false;
    const auto* begin = trimmed.toRawUTF8();
    char* end = nullptr;
    value = std::strtod(begin, &end);
    return end != begin && *end == 0 && std::isfinite(value);
}
}

class MaterialAnnotationEditor::Waveform final : public juce::Component
{
public:
    explicit Waveform(MaterialAnnotationEditor& editor) : owner(editor) {}
    void lookAndFeelChanged() override
    {
        // TextEditor stores text-run colours rather than consulting its current
        // LookAndFeel when drawing. Refresh the overview's existing runs too.
        const auto refreshEditor = [](juce::TextEditor& editor)
        {
            const auto foreground = Palette::contrastingText(Palette::base, !editor.isEnabled());
            editor.setColour(juce::TextEditor::backgroundColourId, Palette::base);
            editor.setColour(juce::TextEditor::textColourId, foreground);
            editor.applyColourToAllText(foreground, false);
        };
        for (auto* editor : { &owner.alias, &owner.extraAliases, &owner.cuts, &owner.batchAliases, &owner.segmentEditor })
            refreshEditor(*editor);
        for (auto& editor : owner.timings) refreshEditor(editor);
        owner.list.setColour(juce::ListBox::backgroundColourId, Palette::base);
        owner.list.setColour(juce::ListBox::textColourId, Palette::contrastingText(Palette::base));
        owner.repaint();
    }
    void paint(juce::Graphics& g) override
    {
        g.fillAll(Palette::graphBackground);
        const auto area = getLocalBounds().reduced(10, 20);
        if (!owner.session || owner.session->durationSeconds() <= 0) return;
        const auto duration = owner.session->durationSeconds();
        const auto x = [&](double seconds)
        { return static_cast<float>(area.getX() + seconds / duration * area.getWidth()); };
        for (const auto& row : owner.session->rows())
        {
            const auto colour = row.materialRegionId == owner.activeId ? Palette::accent : Palette::textMuted;
            if (row.materialRegionId == owner.activeId)
            {
                g.setColour(colour.withAlpha(0.12f));
                g.fillRect(juce::Rectangle<float>(x(row.regionStartSeconds), static_cast<float>(area.getY()),
                    x(row.regionEndSeconds) - x(row.regionStartSeconds), static_cast<float>(area.getHeight())));
            }
        }
        g.setColour(Palette::contrastingText(Palette::graphBackground, true));
        owner.thumbnail.drawChannels(g, area, 0, duration, 1.0f);
        for (const auto& row : owner.session->rows())
        {
            const bool active = row.materialRegionId == owner.activeId;
            // Pronunciation boundaries: full-height solid lines. Internal native
            // segmentation: short dashed lines, deliberately different visuals.
            g.setColour(active ? Palette::accent : Palette::textMuted);
            for (const auto time : {row.regionStartSeconds, row.regionEndSeconds})
                g.drawVerticalLine(juce::roundToInt(x(time)), 2.0f, static_cast<float>(getHeight() - 2));
            if (active)
            {
                g.setColour(Palette::contrastingText(Palette::graphBackground));
                g.drawText(row.name, juce::roundToInt(x(row.regionStartSeconds)) + 3, 1,
                           std::max(30, juce::roundToInt(x(row.regionEndSeconds) - x(row.regionStartSeconds)) - 6),
                           18, juce::Justification::centredLeft, true);
                g.setColour(Palette::textMuted);
                const float dash[]{3, 4};
                for (const auto& segment : row.segments)
                    if (segment.sourceEndSeconds < row.regionEndSeconds)
                        g.drawDashedLine(juce::Line<float>(x(segment.sourceEndSeconds), static_cast<float>(area.getY()),
                            x(segment.sourceEndSeconds), static_cast<float>(area.getBottom())), dash, 2, 1.0f);
                const double times[]{row.regionStartSeconds + row.fixedDurationSeconds,
                                     row.alignmentSeconds, row.regionStartSeconds + row.overlapSeconds};
                const bool light = Palette::background == juce::Colour(0xfffffffb);
                const juce::Colour colours[]{Palette::noteLight, light ? Palette::heading : Palette::accentLight,
                                              light ? Palette::textMuted : Palette::pitchLine};
                const char* labels[]{"F", "P", "O"};
                for (int i = 0; i < 3; ++i)
                {
                    if (times[i] < 0 || times[i] > duration) continue;
                    g.setColour(colours[i]);
                    g.drawVerticalLine(juce::roundToInt(x(times[i])), static_cast<float>(area.getY()), static_cast<float>(area.getBottom()));
                    g.setColour(Palette::contrastingText(Palette::graphBackground));
                    g.drawText(labels[i], juce::roundToInt(x(times[i])) + 2, area.getBottom(), 20, 18, juce::Justification::left);
                }
            }
        }
        if (owner.rangeSelected)
        {
            g.setColour(Palette::accent.withAlpha(0.25f));
            g.fillRect(juce::Rectangle<float>(x(owner.selectedRangeStart), static_cast<float>(area.getY()),
                x(owner.selectedRangeEnd) - x(owner.selectedRangeStart), static_cast<float>(area.getHeight())));
            g.setColour(Palette::heading);
            g.drawRect(juce::Rectangle<float>(x(owner.selectedRangeStart), static_cast<float>(area.getY()),
                x(owner.selectedRangeEnd) - x(owner.selectedRangeStart), static_cast<float>(area.getHeight())), 1.5f);
        }
        g.setColour(Palette::textMuted);
        g.drawText("0", 3, getHeight() - 18, 45, 18, juce::Justification::left);
        g.drawText(juce::String(duration * 1000, 1) + " ms", getWidth() - 125, getHeight() - 18,
                   120, 18, juce::Justification::right);
    }
    double timeAt(float point) const
    {
        if (!owner.session) return 0;
        const auto area = getLocalBounds().reduced(10, 20);
        return juce::jlimit(0.0, owner.session->durationSeconds(),
            (point - area.getX()) / std::max(1, area.getWidth()) * owner.session->durationSeconds());
    }
    void mouseDown(const juce::MouseEvent& event) override
    {
        if (!owner.commitFields()) return;
        anchor = timeAt(event.position.x); dragging = true; sideChosen = false; owner.rangeSelected = false;
        if (owner.adjacentPronunciation.getToggleState() && !owner.session->rows().empty() && !owner.activeRow())
        {
            dragging = false;
            owner.showError(owner.tr("请先选择相邻发音的锚点。", "Select an existing pronunciation anchor first."));
            return;
        }
        owner.selectedRangeAnchor.clear();
        if (owner.adjacentPronunciation.getToggleState() && owner.activeRow())
            owner.selectedRangeAnchor = owner.activeId;
        else owner.selectAt(anchor);
        repaint();
    }
    void mouseDrag(const juce::MouseEvent& event) override
    {
        if (!dragging || event.getDistanceFromDragStart() < 4) return;
        const auto end = timeAt(event.position.x);
        if (owner.selectedRangeAnchor.isNotEmpty())
        {
            if (!sideChosen)
            {
                const auto* row = owner.activeRow();
                if (!row) return;
                owner.selectedRangeBefore = anchor <= row->regionStartSeconds
                    || (anchor < row->regionEndSeconds && end < anchor);
                sideChosen = true; // The gesture can shrink to zero, never flip sides.
            }
            juce::String error;
            owner.rangeSelected = owner.session->previewAdjacentRange(owner.selectedRangeAnchor,
                owner.selectedRangeBefore, end, owner.selectedRangeStart, owner.selectedRangeEnd, error);
            if (!owner.rangeSelected) owner.showError(error);
        }
        else
        {
            owner.rangeSelected = true; owner.selectedRangeStart = std::min(anchor, end);
            owner.selectedRangeEnd = std::max(anchor, end);
        }
        repaint();
    }
    void mouseUp(const juce::MouseEvent&) override
    {
        dragging = false;
        if (!owner.rangeSelected) return;
        owner.refreshing = true; owner.list.deselectAllRows(); owner.refreshing = false;
        owner.activeId.clear(); owner.showFields(); owner.alias.setEnabled(true);
        owner.status.setText(owner.tr("输入别名，然后点击“选区创建别名”。", "Type an alias, then choose Alias selected range."), juce::dontSendNotification);
    }
    void mouseDoubleClick(const juce::MouseEvent& event) override
    {
        owner.selectAt(timeAt(event.position.x)); owner.openDetail(owner.activeId);
    }
private:
    double anchor = 0;
    bool dragging = false, sideChosen = false;
    MaterialAnnotationEditor& owner;
};

juce::String MaterialAnnotationEditor::tr(const char* zh, const char* en) const
{
    juce::ignoreUnused(zh);
    return strings.text(en);
}

MaterialAnnotationEditor::MaterialAnnotationEditor(std::shared_ptr<MaterialAnnotationSession> draft,
    I18n& i18n, std::function<void(const std::vector<SampleRegionSetting>&)> saved)
    : strings(i18n), onSaved(std::move(saved)), session(std::move(draft))
{
    const auto audio = session ? session->sourceAudio() : juce::File{};
    setWantsKeyboardFocus(true);
    formats.registerBasicFormats();
    thumbnail.addChangeListener(this);
    waveform = std::make_unique<Waveform>(*this);
    addAndMakeVisible(*waveform);
    list.setModel(this); list.setMultipleSelectionEnabled(true); list.setRowHeight(28);
    addAndMakeVisible(list);
    for (auto* component : std::initializer_list<juce::Component*>{&title, &status, &aliasLabel, &extraLabel,
        &cutLabel, &batchLabel, &segmentLabel, &pronunciationKindLabel, &alias, &extraAliases, &cuts, &batchAliases, &segmentEditor, &pronunciationKindChoice, &adjacentPronunciation,
        &applyButton, &splitButton, &renameButton, &mergeButton, &deleteButton, &undoButton, &redoButton, &saveButton,
        &createAliasButton, &cancelAliasButton, &detailButton})
        addAndMakeVisible(component);
    title.setText(audio.getFileName(), juce::dontSendNotification);
    aliasLabel.setText(tr("主别名", "Primary alias"), juce::dontSendNotification);
    extraLabel.setText(tr("附加别名（分号分隔）", "Additional aliases (semicolon)"), juce::dontSendNotification);
    cutLabel.setText(tr("切分位置 ms（逗号或空格）", "Split positions ms (comma/space)"), juce::dontSendNotification);
    batchLabel.setText(tr("依次命名所有发音（空白分隔）", "Name all pronunciations in order (whitespace)"), juce::dontSendNotification);
    const char* zh[]{"起点 ms", "终点 ms", "固定时长 ms", "先行时长 ms", "重叠 ms"};
    const char* en[]{"Start ms", "End ms", "Fixed ms", "Preutterance ms", "Overlap ms"};
    for (std::size_t i = 0; i < timings.size(); ++i)
    {
        addAndMakeVisible(timingLabels[i]); addAndMakeVisible(timings[i]);
        timingLabels[i].setText(tr(zh[i], en[i]), juce::dontSendNotification);
        timings[i].onTextChange = [this] { if (!refreshing) fieldsDirty = true; };
        timings[i].onReturnKey = [this] { if (commitFields()) refresh(selectedIds()); };
    }
    segmentLabel.setText(tr("内部区段（角色:绝对终点 ms，如 C:100,V:350,V:600）",
        "Segments (role:absolute end ms, e.g. C:100,V:350,V:600)"), juce::dontSendNotification);
    segmentEditor.setTooltip(tr("U未知 C辅音 V元音 T过渡 S静音 B呼吸 N噪音 E收尾。编辑后为连续区段，最后终点须等于发音终点；不编辑则完整保留原区段及间隙。",
        "U unknown, C consonant, V vowel, T transition, S silence, B breath, N noise, E ending. Edited form makes contiguous segments ending at pronunciation end. Untouched form preserves all metadata and gaps."));
    alias.onTextChange = extraAliases.onTextChange = segmentEditor.onTextChange = [this] { if (!refreshing) fieldsDirty = true; };
    pronunciationKindLabel.setText(tr("发音身份", "Pronunciation identity"), juce::dontSendNotification);
    pronunciationKindChoice.addItem(tr("普通音节", "Normal syllable"), 1);
    pronunciationKindChoice.addItem(tr("过渡发音", "Transition pronunciation"), 2);
    pronunciationKindChoice.addItem(tr("呼吸", "Breath"), 3);
    pronunciationKindChoice.setTooltip(tr("整个发音的类别，不是内部 C/V 区段。", "Pronunciation identity, not an internal C/V segment role."));
    pronunciationKindChoice.onChange = [this] { if (!refreshing) fieldsDirty = true; };
    adjacentPronunciation.setButtonText(tr("相邻发音模式", "Adjacent pronunciation"));
    adjacentPronunciation.setTooltip(tr("先选择已有发音，向左或右拖选紧邻范围；接点固定。", "Select an existing pronunciation, then drag a neighboring range; its join stays exact."));
    adjacentPronunciation.onClick = [this] { rangeSelected = false; selectedRangeAnchor.clear(); waveform->repaint(); };
    batchAliases.setMultiLine(true); batchAliases.setReturnKeyStartsNewLine(true);
    applyButton.setButtonText(tr("应用参数", "Apply fields"));
    splitButton.setButtonText(tr("切分", "Split"));
    renameButton.setButtonText(tr("依次命名", "Batch name"));
    mergeButton.setButtonText(tr("合并所选", "Merge selected"));
    deleteButton.setButtonText(tr("删除所选", "Delete selected"));
    undoButton.setButtonText(tr("撤销", "Undo")); redoButton.setButtonText(tr("重做", "Redo"));
    saveButton.setButtonText(tr("保存标注", "Save annotations"));
    applyButton.onClick = [this] { if (commitFields()) refresh(selectedIds()); };
    splitButton.onClick = [this] { splitAtText(); };
    renameButton.onClick = [this]
    {
        if (!session || !commitFields()) return;
        const auto names = juce::StringArray::fromTokens(batchAliases.getText(), " \t\r\n", "");
        auto aliases = names; aliases.removeEmptyStrings();
        juce::String error;
        if (!session->rename(aliases, error)) showError(error); else { refresh(); status.setText({}, juce::dontSendNotification); }
    };
    mergeButton.onClick = [this]
    {
        if (!session || !commitFields()) return;
        const auto ids = selectedIds(); juce::String error;
        if (!session->merge(ids, error)) showError(error); else refresh(ids);
    };
    deleteButton.onClick = [this]
    {
        if (!session || !commitFields()) return;
        const auto ids = selectedIds(); juce::String error;
        if (!session->erase(ids, error)) showError(error); else refresh();
    };
    undoButton.onClick = [this] { history(true); }; redoButton.onClick = [this] { history(false); };
    saveButton.onClick = [this] { save(); };
    createAliasButton.setButtonText(tr("选区创建别名", "Alias selected range"));
    cancelAliasButton.setButtonText(tr("取消当前别名", "Remove current alias"));
    cancelAliasButton.setTooltip(tr("取消未创建的范围，或移除主别名框中输入的现有别名；原音频保持不变。",
        "Cancel a pending range, or remove the existing alias typed in Primary alias. Source audio stays unchanged."));
    detailButton.setButtonText(tr("详细参数", "Detail parameters"));
    createAliasButton.onClick = [this] { createRangeAlias(); };
    cancelAliasButton.onClick = [this] { cancelAlias(); };
    detailButton.onClick = [this] { openDetail(activeId); };
    if (session)
    {
        thumbnail.setSource(new juce::FileInputSource(audio));
        refresh();
        status.setText(session->normalizationSummary().isNotEmpty() ? session->normalizationSummary()
            : tr("拖选范围创建别名；双击发音进入详细波形参数；取消别名不改原音频。",
                "Drag a source range to name it. Double-click a pronunciation for waveform detail. Removing aliases never changes audio."), juce::dontSendNotification);
        status.setTooltip(session->normalizationSummary());
    }
    else { showError(tr("无法打开标注草稿。", "Cannot open annotation draft.")); refresh(); }
    // This is only the overview layer. Actual timing/segment manipulation is
    // the owning OtoWaveformEditorComponent's existing detailed waveform UI.
    segmentLabel.setVisible(false); segmentEditor.setVisible(false);
    for (std::size_t i = 2; i < timings.size(); ++i)
    { timingLabels[i].setVisible(false); timings[i].setVisible(false); }
    waveform->lookAndFeelChanged();
    setSize(1000, 720);
}

MaterialAnnotationEditor::~MaterialAnnotationEditor()
{
    thumbnail.removeChangeListener(this);
    list.setModel(nullptr);
}

juce::Component& MaterialAnnotationEditor::overviewWaveform() { return *waveform; }

void MaterialAnnotationEditor::selectSourceRange(juce::Range<double> range)
{
    refreshing = true; list.deselectAllRows(); refreshing = false;
    activeId.clear(); selectedRangeAnchor.clear();
    selectedRangeStart = range.getStart(); selectedRangeEnd = range.getEnd(); rangeSelected = true;
    showFields(); alias.setEnabled(true);
    status.setText(tr("已定位音符的原录音范围，可输入别名创建标注。", "Located the note's source range; enter an alias to annotate it."), juce::dontSendNotification);
    waveform->repaint();
}

void MaterialAnnotationEditor::openDetail(const juce::String& id)
{
    const auto chosenId = id; // setRow may replace the row vector during commit.
    if (chosenId.isEmpty() || !commitFields()) return;
    const auto callback = onEditPronunciation;
    if (callback) callback(chosenId);
}

void MaterialAnnotationEditor::listBoxItemDoubleClicked(int row, const juce::MouseEvent&)
{
    if (session && juce::isPositiveAndBelow(row, getNumRows()))
        openDetail(session->rows()[static_cast<std::size_t>(row)].materialRegionId);
}

void MaterialAnnotationEditor::createRangeAlias()
{
    if (!session || !rangeSelected)
    { showError(tr("请先拖选原音频范围。", "First drag a range on the source waveform.")); return; }
    const auto name = alias.getText().trim();
    juce::String error;
    const bool created = selectedRangeAnchor.isNotEmpty()
        ? session->createAdjacentPronunciation(selectedRangeAnchor, selectedRangeBefore,
            selectedRangeBefore ? selectedRangeStart : selectedRangeEnd, name, error)
        : session->createPronunciation(selectedRangeStart, selectedRangeEnd, name, error);
    if (!created)
    { showError(error); return; }
    rangeSelected = false; selectedRangeAnchor.clear(); fieldsDirty = false;
    juce::String id;
    for (const auto& row : session->rows())
        if (row.name == name || row.additionalAliases.contains(name)) { id = row.materialRegionId; break; }
    refresh(id.isEmpty() ? std::vector<juce::String>{} : std::vector<juce::String>{id});
}

void MaterialAnnotationEditor::cancelAlias()
{
    if (rangeSelected)
    {
        rangeSelected = false; selectedRangeAnchor.clear(); fieldsDirty = false; activeId.clear(); showFields(); waveform->repaint();
        status.setText(tr("已取消尚未创建的选区别名。", "Uncreated range alias cancelled."), juce::dontSendNotification);
        return;
    }
    const auto* row = activeRow();
    if (!session || row == nullptr) return;
    juce::String error;
    if (!session->removeAlias(row->materialRegionId, alias.getText().trim(), error)) { showError(error); return; }
    fieldsDirty = false; refresh();
}

void MaterialAnnotationEditor::setPublicationValidator(std::function<bool(
    const std::vector<SampleRegionSetting>&, juce::String&)> validator)
{
    if (session) session->validatePublication = std::move(validator);
}

int MaterialAnnotationEditor::getNumRows() { return session ? static_cast<int>(session->rows().size()) : 0; }
void MaterialAnnotationEditor::paintListBoxItem(int index, juce::Graphics& g, int width, int height, bool selected)
{
    if (!session || !juce::isPositiveAndBelow(index, getNumRows())) return;
    const auto& row = session->rows()[static_cast<std::size_t>(index)];
    const auto background = selected ? Palette::selection : Palette::base;
    g.fillAll(background);
    g.setColour(Palette::contrastingText(background));
    g.drawText(juce::String(index + 1) + ". " + row.name + "  [" + juce::String(row.regionStartSeconds * 1000, 1)
        + " – " + juce::String(row.regionEndSeconds * 1000, 1) + "]", 6, 0, width - 12, height,
        juce::Justification::centredLeft, true);
}
std::vector<juce::String> MaterialAnnotationEditor::selectedIds() const
{
    std::vector<juce::String> ids;
    if (!session) return ids;
    const auto selected = list.getSelectedRows();
    for (int i = 0; i < selected.size(); ++i)
        if (juce::isPositiveAndBelow(selected[i], static_cast<int>(session->rows().size())))
            ids.push_back(session->rows()[static_cast<std::size_t>(selected[i])].materialRegionId);
    return ids;
}
const SampleRegionSetting* MaterialAnnotationEditor::activeRow() const
{
    if (session)
        for (const auto& row : session->rows()) if (row.materialRegionId == activeId) return &row;
    return nullptr;
}
void MaterialAnnotationEditor::selectedRowsChanged(int last)
{
    if (refreshing || !session) return;
    if (!commitFields())
    {
        refreshing = true;
        for (int i = 0; i < getNumRows(); ++i)
            if (session->rows()[static_cast<std::size_t>(i)].materialRegionId == activeId)
            { list.selectRow(i); break; }
        refreshing = false;
        return;
    }
    activeId = juce::isPositiveAndBelow(last, getNumRows())
        ? session->rows()[static_cast<std::size_t>(last)].materialRegionId : juce::String();
    showFields(); waveform->repaint();
}
void MaterialAnnotationEditor::showFields()
{
    refreshing = true;
    const auto* row = activeRow();
    alias.setText(row ? row->name : juce::String(), false);
    extraAliases.setText(row ? row->additionalAliases.joinIntoString(";") : juce::String(), false);
    const double numbers[]{row ? row->regionStartSeconds : 0, row ? row->regionEndSeconds : 0,
        row ? row->fixedDurationSeconds : 0, row ? row->alignmentSeconds - row->regionStartSeconds : 0,
        row ? row->overlapSeconds : 0};
    for (std::size_t i = 0; i < timings.size(); ++i)
    { timings[i].setText(row ? juce::String(numbers[i] * 1000, 6) : juce::String(), false); timings[i].setEnabled(row != nullptr); }
    segmentBaseline.clear();
    if (row)
    {
        for (const auto& segment : row->segments)
        {
            if (segmentBaseline.isNotEmpty()) segmentBaseline += ",";
            segmentBaseline += juce::String::charToString(roleCode(segment.role)) + ":"
                + juce::String(segment.sourceEndSeconds * 1000, 6);
        }
        // A display suggestion only. An untouched bare row remains unsegmented.
        if (row->segments.empty()) segmentBaseline = "V:" + juce::String(row->regionEndSeconds * 1000, 6);
    }
    segmentEditor.setText(segmentBaseline, false); segmentEditor.setEnabled(row != nullptr);
    alias.setEnabled(row != nullptr); extraAliases.setEnabled(row != nullptr);
    pronunciationKindChoice.setEnabled(row != nullptr); applyButton.setEnabled(row != nullptr);
    pronunciationKindChoice.setSelectedId(row ? static_cast<int>(row->pronunciationKind) + 1 : 1, juce::dontSendNotification);
    fieldsDirty = false; refreshing = false;
}
void MaterialAnnotationEditor::refresh(const std::vector<juce::String>& ids)
{
    refreshing = true;
    list.updateContent(); list.deselectAllRows();
    juce::SparseSet<int> selection;
    for (int i = 0; i < getNumRows(); ++i)
        if (std::find(ids.begin(), ids.end(), session->rows()[static_cast<std::size_t>(i)].materialRegionId) != ids.end())
            selection.addRange({i, i + 1});
    if (selection.size() == 0 && getNumRows() > 0) selection.addRange({0, 1});
    list.setSelectedRows(selection, juce::dontSendNotification);
    if (selection.size() > 0) list.scrollToEnsureRowIsOnscreen(selection[0]);
    activeId = selection.size() > 0 ? session->rows()[static_cast<std::size_t>(selection[0])].materialRegionId : juce::String();
    undoButton.setEnabled(session && session->canUndo()); redoButton.setEnabled(session && session->canRedo());
    for (auto* button : {&splitButton, &renameButton, &mergeButton, &deleteButton, &saveButton}) button->setEnabled(session != nullptr);
    refreshing = false; showFields(); list.repaint(); waveform->repaint();
}
bool MaterialAnnotationEditor::commitFields()
{
    if (!fieldsDirty) return true;
    const auto* current = activeRow();
    if (!current || !session) return false;
    auto row = *current;
    double values[5]{};
    for (std::size_t i = 0; i < timings.size(); ++i)
        if (!parseNumber(timings[i].getText(), values[i]))
        { showError(tr("请输入有限的毫秒数。", "Enter finite numeric milliseconds.")); return false; }
    row.name = alias.getText();
    if (extraAliases.getText() != current->additionalAliases.joinIntoString(";"))
    {
        row.additionalAliases = juce::StringArray::fromTokens(extraAliases.getText(), ";", "");
        row.additionalAliases.trim(); row.additionalAliases.removeEmptyStrings();
    }
    const double before[]{current->regionStartSeconds, current->regionEndSeconds,
        current->fixedDurationSeconds, current->alignmentSeconds - current->regionStartSeconds,
        current->overlapSeconds};
    bool changed[5]{};
    for (std::size_t i = 0; i < timings.size(); ++i)
        changed[i] = timings[i].getText() != juce::String(before[i] * 1000, 6);
    // Editing an alias must not quantise inherited native timing to the display
    // precision. Only timing fields the user actually changed are replaced.
    if (changed[0]) row.regionStartSeconds = values[0] / 1000;
    if (changed[1]) row.regionEndSeconds = values[1] / 1000;
    if (changed[2]) row.fixedDurationSeconds = values[2] / 1000;
    if (changed[0] || changed[3])
        row.alignmentSeconds = row.regionStartSeconds + (changed[3] ? values[3] / 1000 : before[3]);
    if (changed[4]) row.overlapSeconds = values[4] / 1000;
    row.pronunciationKind = static_cast<PronunciationKind>(juce::jlimit(1, 3, pronunciationKindChoice.getSelectedId()) - 1);
    if (segmentEditor.getText() != segmentBaseline)
    {
        const auto tokens = juce::StringArray::fromTokens(segmentEditor.getText(), ",", "");
        std::vector<NativeSegment> segments;
        std::vector<bool> used(current->segments.size(), false);
        double begin = row.regionStartSeconds;
        if (tokens.isEmpty())
        { showError(tr("请输入内部区段。", "Enter internal segments.")); return false; }
        for (int i = 0; i < tokens.size(); ++i)
        {
            const auto colon = tokens[i].indexOfChar(':');
            NativeSegmentRole role = NativeSegmentRole::unknown;
            double milliseconds = 0;
            if (colon < 1 || !parseRole(tokens[i].substring(0, colon), role)
                || !parseNumber(tokens[i].substring(colon + 1), milliseconds))
            { showError(tr("区段格式无效，请使用 C:100,V:350 等。", "Invalid segment syntax; use C:100,V:350.")); return false; }
            double end = milliseconds / 1000;
            int match = -1;
            if (i < static_cast<int>(current->segments.size()) && current->segments[static_cast<std::size_t>(i)].role == role
                && !used[static_cast<std::size_t>(i)]) match = i;
            if (match < 0)
                for (std::size_t j = 0; j < current->segments.size(); ++j)
                    if (!used[j] && current->segments[j].role == role
                        && std::abs(current->segments[j].sourceEndSeconds - end) <= 1.e-9)
                    { match = static_cast<int>(j); break; }
            NativeSegment segment;
            if (match >= 0)
            {
                segment = current->segments[static_cast<std::size_t>(match)];
                used[static_cast<std::size_t>(match)] = true;
                if (tokens[i].substring(colon + 1).trim() == juce::String(segment.sourceEndSeconds * 1000, 6))
                    end = segment.sourceEndSeconds; // Retain exact inherited timing bits.
            }
            else
            {
                segment.id = "s" + juce::Uuid().toString().removeCharacters("-");
                segment.alias = row.name; segment.provenance = "user"; segment.confidence = 1;
                segment.alignmentSeconds = row.alignmentSeconds; segment.overlapSeconds = row.overlapSeconds;
                segment.stretchable = role != NativeSegmentRole::consonant;
                segment.stretchWeight = 1;
            }
            if (i == tokens.size() - 1 && std::abs(end - row.regionEndSeconds) <= 1.e-9)
                end = row.regionEndSeconds;
            if (!std::isfinite(end) || end <= begin || end > row.regionEndSeconds
                || (i == tokens.size() - 1 && end != row.regionEndSeconds))
            { showError(tr("区段须为正长度、递增且位于发音内，最后终点须等于发音终点。",
                "Segment ends must increase within the pronunciation; the last must equal its end.")); return false; }
            segment.role = role; segment.sourceStartSeconds = begin; segment.sourceEndSeconds = end;
            segments.push_back(std::move(segment)); begin = end;
        }
        row.segments = std::move(segments);
        if (!row.segments.empty()) row.role = row.segments.front().role;
    }
    juce::String error;
    if (!session->setRow(activeId, row, error)) { showError(error); return false; }
    fieldsDirty = false;
    undoButton.setEnabled(session->canUndo()); redoButton.setEnabled(session->canRedo());
    list.repaint(); waveform->repaint();
    return true;
}
void MaterialAnnotationEditor::showError(const juce::String& error)
{
    status.setText(error, juce::dontSendNotification);
    status.setTooltip(error);
}
void MaterialAnnotationEditor::splitAtText()
{
    if (!session || !commitFields()) return;
    auto tokens = juce::StringArray::fromTokens(cuts.getText(), ", \t\r\n", ""); tokens.removeEmptyStrings();
    std::vector<double> positions;
    for (const auto& token : tokens)
    {
        double value = 0;
        if (!parseNumber(token, value)) { showError(tr("切分位置无效。", "Invalid split position.")); return; }
        positions.push_back(value / 1000);
    }
    if (positions.empty()) { showError(tr("请输入切分位置。", "Enter split positions.")); return; }
    juce::String error;
    if (!session->split(positions, error)) showError(error); else refresh();
}
void MaterialAnnotationEditor::splitAt(double seconds)
{
    if (!session || !commitFields()) return;
    juce::String error;
    if (!session->split({seconds}, error)) showError(error); else { refresh(); selectAt(seconds); }
}
void MaterialAnnotationEditor::selectAt(double seconds)
{
    if (!session || !commitFields()) return;
    for (int i = 0; i < getNumRows(); ++i)
    {
        const auto& row = session->rows()[static_cast<std::size_t>(i)];
        if (seconds >= row.regionStartSeconds && seconds < row.regionEndSeconds)
        { list.selectRow(i); list.scrollToEnsureRowIsOnscreen(i); return; }
    }
}
void MaterialAnnotationEditor::save()
{
    if (!session || !commitFields()) return;
    const auto ids = selectedIds(); juce::String error;
    if (!session->save(error)) { showError(error); return; }
    refresh(ids);
    status.setText(tr("标注已保存；原音频未修改。", "Annotations saved; source audio unchanged."), juce::dontSendNotification);
    // The host may close/delete this component in its completion callback.
    const auto callback = onSaved;
    const auto rows = session->rows();
    if (callback) callback(rows);
}
void MaterialAnnotationEditor::history(bool back)
{
    if (!session) return;
    // Discard un-applied form text before moving the authoritative draft history.
    fieldsDirty = false;
    const auto ids = selectedIds();
    if (back ? session->undo() : session->redo()) refresh(ids); else showFields();
}
bool MaterialAnnotationEditor::keyPressed(const juce::KeyPress& key)
{
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'Z')
    { history(!key.getModifiers().isShiftDown()); return true; }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'S') { save(); return true; }
    return false;
}
void MaterialAnnotationEditor::changeListenerCallback(juce::ChangeBroadcaster*) { waveform->repaint(); }
void MaterialAnnotationEditor::paint(juce::Graphics& g) { g.fillAll(Palette::panel); }
void MaterialAnnotationEditor::resized()
{
    auto area = getLocalBounds().reduced(12);
    auto titleRow = area.removeFromTop(26);
    adjacentPronunciation.setBounds(titleRow.removeFromRight(230));
    title.setBounds(titleRow);
    waveform->setBounds(area.removeFromTop(std::max(140, getHeight() / 3)));
    area.removeFromTop(8);
    auto buttons = area.removeFromBottom(32);
    for (auto* button : {&undoButton, &redoButton, &mergeButton, &deleteButton})
    { button->setBounds(buttons.removeFromLeft(130).reduced(3, 0)); }
    saveButton.setBounds(buttons.removeFromRight(150));
    status.setBounds(area.removeFromBottom(44));
    auto left = area.removeFromLeft(std::max(230, getWidth() / 3)); left.removeFromRight(10); list.setBounds(left);
    auto line = area.removeFromTop(44); aliasLabel.setBounds(line.removeFromTop(20)); alias.setBounds(line);
    line = area.removeFromTop(44); extraLabel.setBounds(line.removeFromTop(20)); extraAliases.setBounds(line);
    line = area.removeFromTop(28);
    createAliasButton.setBounds(line.removeFromLeft(160)); line.removeFromLeft(5);
    cancelAliasButton.setBounds(line.removeFromLeft(160)); line.removeFromLeft(5);
    detailButton.setBounds(line.removeFromRight(140));
    auto timing = area.removeFromTop(48);
    const auto cellWidth = timing.getWidth() / 2;
    for (std::size_t i = 0; i < 2; ++i)
    {
        auto cell = i == 1 ? timing : timing.removeFromLeft(cellWidth);
        cell.reduce(2, 0); timingLabels[i].setBounds(cell.removeFromTop(20)); timings[i].setBounds(cell.removeFromTop(24));
    }
    line = area.removeFromTop(28); pronunciationKindLabel.setBounds(line.removeFromLeft(150));
    pronunciationKindChoice.setBounds(line.removeFromLeft(240)); applyButton.setBounds(line.removeFromRight(130));
    line = area.removeFromTop(48); cutLabel.setBounds(line.removeFromTop(20));
    splitButton.setBounds(line.removeFromRight(100)); line.removeFromRight(6); cuts.setBounds(line);
    line = area.removeFromTop(68); batchLabel.setBounds(line.removeFromTop(20));
    renameButton.setBounds(line.removeFromRight(120).removeFromTop(28)); line.removeFromRight(6); batchAliases.setBounds(line);
}
}
