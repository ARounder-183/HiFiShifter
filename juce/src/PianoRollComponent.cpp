#include "NativeAudioTrim.h"
#include "NativeAudioFocus.h"
#include "NativeAudioLink.h"
#include "NativeAudioDisconnect.h"
#include "SourceWaveformPreview.h"
#include "NativePitchVoicingDisplay.h"
#include "NativeNoteTiming.h"
#include "DiffSingerParameterCurves.h"
#include "PianoRollComponent.h"
#include "backend/CvvcPhonemeCache.h"
#include "backend/ChineseCvvcPhonemizer.h"
#include "ClipParts.h"
#include "backend/DiffSingerTiming.h"
#include "DiffSingerPhonemeDisplay.h"
#include "DiffSingerPitchHandles.h"
#include "Theme.h"
#include "FlagCurveDrawing.h"
#include "backend/UtauRenderer.h"
#include "backend/AmplitudeEnvelopeCurve.h"
#include "backend/AdvancedEnvelope.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace hachi
{
namespace
{
// Keep the established dark rendering; light colours always come from Theme.
// These select presentation roles only, never note data or editor behaviour.
bool physicalOverlapSource(const ClipData& clip)
{
    return clip.sourceFile != juce::File{} && clip.sourceFile.hasFileExtension("wav;aif;aiff;flac;ogg;mp3")
        && std::isfinite(clip.startSeconds) && std::isfinite(clip.audioStartSeconds)
        && std::isfinite(clip.audioLength()) && clip.audioLength() > 1.e-9;
}
juce::Range<double> physicalOverlapWindow(const ClipData& clip)
{ const auto start = clip.startSeconds + clip.audioStartSeconds; return {start,start+clip.audioLength()}; }

bool lightRollPalette() { return Palette::background.getPerceivedBrightness() > 0.5f; }
juce::Colour rollRole(juce::Colour dark, juce::Colour light)
{
    return lightRollPalette() ? light : dark;
}
juce::Colour rollText(juce::Colour dark, juce::Colour actualBackground, bool muted = false)
{
    return lightRollPalette() ? Palette::contrastingText(actualBackground, muted) : dark;
}
juce::Colour rollInk(juce::Colour ink, juce::Colour actualBackground)
{
    if (!lightRollPalette() || Palette::contrastRatio(ink, actualBackground) >= 3.0) return ink;
    return Palette::contrastRatio(Palette::heading, actualBackground) >= 3.0
        ? Palette::heading : Palette::contrastingText(actualBackground);
}

// Two pitch points only have to stay in order.  A millisecond is finer than
// any frame the engine resamples at, so this never stands between the user and
// the shape they are drawing -- including a deliberate near-vertical jump.
constexpr double minimumAnchorSeparation = 0.001;

// The seven numbers of a vibrato, in the order UTAU writes them, so a preset
// copied from there reads the same here.
constexpr std::array<const char*, 7> vibratoFieldKeys {
    "length", "cycle", "depth", "fadein", "fadeout", "phase", "offset"
};

}

juce::StringArray PianoRollComponent::vibratoBuiltInPresets()
{
    // name, length %, cycle ms, depth cents, fade in %, fade out %, phase %,
    // height % -- the order UTAU writes them, so a preset copied from there
    // reads the same here.  The first three are the ones UTAU ships with.
    return { juce::String::fromUTF8("默认,65,180,35,20,20,0,0"),
             juce::String::fromUTF8("强,65,210,55,20,20,0,0"),
             juce::String::fromUTF8("弱,65,165,20,20,20,0,0"),
             juce::String::fromUTF8("自定义预设1,77,153,35,20,7,153,13.7") };
}

juce::StringArray PianoRollComponent::vibratoPresetValues(const juce::String& line)
{
    auto numbers = juce::StringArray::fromTokens(line, ",", "");
    if (numbers.size() < 2) return {};
    numbers.remove(0);   // the name
    for (auto& value : numbers) value = value.trim();
    return numbers;
}

juce::String PianoRollComponent::vibratoPresetsWith(const juce::String& saved,
                                                    const juce::String& name,
                                                    const juce::StringArray& values)
{
    auto lines = juce::StringArray::fromLines(saved);
    lines.removeEmptyStrings();
    for (auto index = lines.size(); --index >= 0;)
        if (lines[index].upToFirstOccurrenceOf(",", false, false).trim() == name)
            lines.remove(index);
    lines.add(name + "," + values.joinIntoString(","));
    return lines.joinIntoString("\n");
}

namespace
{
// Presets for the vibrato dialog: the three UTAU ships with, then whatever the
// user has saved.  An entry is its name and its numbers on one line, which is
// both what is stored and what the list shows -- so a preset can be read at a
// glance without opening it.
class VibratoPresetBar final : public juce::Component
{
public:
    VibratoPresetBar(juce::AlertWindow& owner, juce::String saved,
                     std::function<void(const juce::String&)> persist)
        : dialog(owner), stored(std::move(saved)), save(std::move(persist))
    {
        chooser.setEditableText(true);
        chooser.setTextWhenNothingSelected(
            I18n::uiText("选择预设，或在此输入名称后保存"));
        chooser.onChange = [this] { applyChosen(); };
        addAndMakeVisible(chooser);
        keep.setButtonText(I18n::uiText("存为预设"));
        keep.setTooltip(I18n::uiText("把上面的数值存成预设；名称取自左边的输入框"));
        keep.onClick = [this] { keepCurrent(); };
        addAndMakeVisible(keep);
        remove.setButtonText(I18n::uiText("删除"));
        remove.setTooltip(I18n::uiText("删除选中的自定义预设"));
        remove.onClick = [this] { removeChosen(); };
        addAndMakeVisible(remove);
        rebuild();
        setSize(460, 26);
    }

    void resized() override
    {
        auto area = getLocalBounds();
        remove.setBounds(area.removeFromRight(56));
        area.removeFromRight(4);
        keep.setBounds(area.removeFromRight(84));
        area.removeFromRight(6);
        chooser.setBounds(area);
    }

private:
    juce::StringArray customs() const
    {
        auto lines = juce::StringArray::fromLines(stored);
        lines.removeEmptyStrings();
        return lines;
    }

    void rebuild()
    {
        chooser.clear(juce::dontSendNotification);
        auto id = 1;
        for (const auto& entry : PianoRollComponent::vibratoBuiltInPresets())
            chooser.addItem(entry, id++);
        const auto saved = customs();
        if (!saved.isEmpty()) chooser.addSeparator();
        firstCustomId = id;
        for (const auto& entry : saved) chooser.addItem(entry, id++);
    }

    void applyChosen()
    {
        const auto index = chooser.getSelectedItemIndex();
        if (index < 0) return;
        const auto numbers = PianoRollComponent::vibratoPresetValues(
            chooser.getItemText(index));
        if (numbers.isEmpty()) return;
        for (std::size_t field = 0; field < vibratoFieldKeys.size(); ++field)
            if (field < static_cast<std::size_t>(numbers.size()))
                if (auto* editor = dialog.getTextEditor(vibratoFieldKeys[field]))
                    editor->setText(numbers[static_cast<int>(field)].trim(), false);
    }

    void keepCurrent()
    {
        auto name = chooser.getText().trim();
        // A name that is really a preset line, because the box was left showing
        // the one that was chosen: take the part before the first comma.
        if (name.contains(",")) name = name.upToFirstOccurrenceOf(",", false, false).trim();
        if (name.isEmpty())
            name = juce::String::fromUTF8("自定义 ") + juce::String(customs().size() + 1);
        juce::StringArray values;
        for (const auto* key : vibratoFieldKeys)
            values.add(dialog.getTextEditorContents(key).trim());
        stored = PianoRollComponent::vibratoPresetsWith(stored, name, values);
        if (save) save(stored);
        rebuild();
        chooser.setText(name, juce::dontSendNotification);
    }

    void removeChosen()
    {
        const auto index = chooser.getSelectedItemIndex();
        if (index < 0) return;
        const auto id = chooser.getSelectedId();
        if (id < firstCustomId) return;   // the built-in three stay
        auto lines = customs();
        const auto position = id - firstCustomId;
        if (position < 0 || position >= lines.size()) return;
        lines.remove(position);
        stored = lines.joinIntoString("\n");
        if (save) save(stored);
        rebuild();
    }

    juce::AlertWindow& dialog;
    juce::String stored;
    std::function<void(const juce::String&)> save;
    juce::ComboBox chooser;
    juce::TextButton keep, remove;
    int firstCustomId = 1;
};


double automaticPitchTransitionInset(double leftDuration, double rightDuration)
{
    // A 40 ms transition is short enough to read as an ordinary note change,
    // while very short notes receive a proportionally smaller interval so the
    // head and tail handles can never cross.
    return std::max(0.002, std::min({ 0.020,
        std::max(0.01, leftDuration) * 0.20,
        std::max(0.01, rightDuration) * 0.20 }));
}

std::vector<PitchCurveEditPoint> simplifyPitchAnchors(
    const std::vector<PitchCurveEditPoint>& input, float toleranceMidi)
{
    if (input.size() <= 2) return input;
    std::vector<bool> keep(input.size(), false);
    keep.front() = true;
    keep.back() = true;
    std::function<void(std::size_t, std::size_t)> simplify =
        [&](std::size_t first, std::size_t last)
    {
        if (last <= first + 1) return;
        const auto span = input[last].timeSeconds - input[first].timeSeconds;
        auto largestError = 0.0f;
        auto largestIndex = first;
        for (auto index = first + 1; index < last; ++index)
        {
            const auto amount = span > 1.0e-9
                ? static_cast<float>((input[index].timeSeconds - input[first].timeSeconds) / span)
                : 0.0f;
            const auto interpolated = input[first].targetMidi
                + (input[last].targetMidi - input[first].targetMidi) * amount;
            const auto error = std::abs(input[index].targetMidi - interpolated);
            if (error > largestError)
            {
                largestError = error;
                largestIndex = index;
            }
        }
        if (largestError <= toleranceMidi) return;
        keep[largestIndex] = true;
        simplify(first, largestIndex);
        simplify(largestIndex, last);
    };
    simplify(0, input.size() - 1);
    std::vector<PitchCurveEditPoint> result;
    for (std::size_t index = 0; index < input.size(); ++index)
        if (keep[index]) result.push_back(input[index]);
    return result;
}

class BezierParameterEditor final : public juce::Component
{
public:
    BezierParameterEditor(const I18n& strings,
                          std::array<float, 4> initialValues)
    {
        const std::array<juce::String, 4> names {
            strings.text("pitchCurve.bezierX1"), strings.text("pitchCurve.bezierY1"),
            strings.text("pitchCurve.bezierX2"), strings.text("pitchCurve.bezierY2")
        };
        for (std::size_t index = 0; index < sliders.size(); ++index)
        {
            addAndMakeVisible(labels[index]);
            labels[index].setText(names[index], juce::dontSendNotification);
            labels[index].setJustificationType(juce::Justification::centredRight);
            labels[index].setColour(juce::Label::textColourId, Palette::text);
            addAndMakeVisible(sliders[index]);
            sliders[index].setSliderStyle(juce::Slider::LinearHorizontal);
            sliders[index].setTextBoxStyle(juce::Slider::TextBoxRight, false, 72, 22);
            sliders[index].setNumDecimalPlacesToDisplay(2);
            sliders[index].setRange(index == 0 || index == 2 ? 0.0 : -2.0,
                                    index == 0 || index == 2 ? 1.0 : 3.0, 0.01);
            sliders[index].setValue(initialValues[index], juce::dontSendNotification);
            sliders[index].onValueChange = [this] { repaint(); };
        }
        setSize(470, 330);
    }

    [[nodiscard]] std::array<float, 4> values() const
    {
        return { static_cast<float>(sliders[0].getValue()),
                 static_cast<float>(sliders[1].getValue()),
                 static_cast<float>(sliders[2].getValue()),
                 static_cast<float>(sliders[3].getValue()) };
    }

    void paint(juce::Graphics& g) override
    {
        auto preview = getLocalBounds().removeFromTop(170).reduced(34, 14).toFloat();
        g.setColour(Palette::graphBackground);
        g.fillRoundedRectangle(preview, 5.0f);
        g.setColour(Palette::border);
        g.drawRoundedRectangle(preview, 5.0f, 1.0f);

        auto plot = preview.reduced(18.0f, 12.0f);
        const auto current = values();
        auto minimumY = std::min({ 0.0f, 1.0f, current[1], current[3] });
        auto maximumY = std::max({ 0.0f, 1.0f, current[1], current[3] });
        const auto padding = std::max(0.12f, (maximumY - minimumY) * 0.1f);
        minimumY -= padding;
        maximumY += padding;
        const auto ySpan = std::max(0.01f, maximumY - minimumY);
        const auto pointFor = [plot, minimumY, ySpan](float x, float y)
        {
            return juce::Point<float>(plot.getX() + x * plot.getWidth(),
                plot.getBottom() - ((y - minimumY) / ySpan) * plot.getHeight());
        };
        const auto start = pointFor(0.0f, 0.0f);
        const auto first = pointFor(current[0], current[1]);
        const auto second = pointFor(current[2], current[3]);
        const auto end = pointFor(1.0f, 1.0f);

        g.setColour(Palette::grid.withAlpha(0.8f));
        g.drawLine(start.x, start.y, end.x, end.y, 1.0f);
        g.setColour(Palette::textMuted.withAlpha(0.8f));
        g.drawLine(start.x, start.y, first.x, first.y, 1.0f);
        g.drawLine(second.x, second.y, end.x, end.y, 1.0f);

        juce::Path curve;
        curve.startNewSubPath(start);
        curve.cubicTo(first, second, end);
        g.setColour(Palette::noteLight);
        g.strokePath(curve, juce::PathStrokeType(2.5f));
        g.setColour(Palette::accentLight);
        g.fillEllipse(first.x - 4.0f, first.y - 4.0f, 8.0f, 8.0f);
        g.fillEllipse(second.x - 4.0f, second.y - 4.0f, 8.0f, 8.0f);
        g.setColour(Palette::text);
        g.fillEllipse(start.x - 3.0f, start.y - 3.0f, 6.0f, 6.0f);
        g.fillEllipse(end.x - 3.0f, end.y - 3.0f, 6.0f, 6.0f);
    }

    void resized() override
    {
        auto area = getLocalBounds();
        area.removeFromTop(178);
        for (std::size_t index = 0; index < sliders.size(); ++index)
        {
            auto row = area.removeFromTop(36);
            labels[index].setBounds(row.removeFromLeft(88));
            sliders[index].setBounds(row.reduced(4, 2));
        }
    }

private:
    std::array<juce::Label, 4> labels;
    std::array<juce::Slider, 4> sliders;
};
}

PianoRollComponent::PianoRollComponent(ProjectModel& modelToUse,
                                       const I18n& stringsToUse)
    : model(modelToUse), strings(stringsToUse)
{
    formats.registerBasicFormats();
    consonantVoicing = std::make_unique<NativePitchVoicingCache>();
    consonantVoicing->addChangeListener(this);
    model.addChangeListener(this);
    setWantsKeyboardFocus(true);
    juce::Desktop::getInstance().addGlobalMouseListener(&pointSelectionMouseListener);
    addChildComponent(inlineAliasEditor);
    inlineAliasEditor.setMultiLine(false);
    inlineAliasEditor.setReturnKeyStartsNewLine(false);
    inlineAliasEditor.setSelectAllWhenFocused(true);
    inlineAliasEditor.setJustification(juce::Justification::centredLeft);
    inlineAliasEditor.setFont(13.0f);
    inlineAliasEditor.setColour(juce::TextEditor::backgroundColourId,
                                Palette::panelRaised.withAlpha(0.98f));
    inlineAliasEditor.setColour(juce::TextEditor::textColourId, Palette::text);
    inlineAliasEditor.setColour(juce::TextEditor::outlineColourId, Palette::accentLight);
    inlineAliasEditor.setColour(juce::TextEditor::focusedOutlineColourId, Palette::noteLight);
    inlineAliasEditor.onTextChange = [this] { repaint(); };
    inlineAliasEditor.onReturnKey = [this] { finishInlineAliasEdit(true); };
    inlineAliasEditor.onEscapeKey = [this] { finishInlineAliasEdit(false); };
    inlineAliasEditor.onFocusLost = [this] { if(!inlineAliasSuspended) finishInlineAliasEdit(true); };
    inlineAliasEditor.requestSource = [this]
    {
        if(inlineAliasNoteId.isEmpty()||!onRequestMaterialSourceInput)return;
        const auto id=inlineAliasNoteId,text=inlineAliasEditor.getText();const auto callback=onRequestMaterialSourceInput;
        callback(id,text); // Host captures/suspends before stealing editor focus.
    };
    rebuildLayout();
}

PianoRollComponent::~PianoRollComponent()
{
    juce::Desktop::getInstance().removeGlobalMouseListener(&pointSelectionMouseListener);
    cvvcLayoutPool.removeAllJobs(true,5000);
    consonantVoicing->removeChangeListener(this);
    consonantVoicing.reset();
    for (auto& [_, thumbnail] : thumbnails)
        thumbnail->removeChangeListener(this);
    model.removeChangeListener(this);
}

void PianoRollComponent::setConsonantPitchDashed(bool show)
{
    if (consonantPitchDashed == show) return;
    consonantPitchDashed = show;
    requestConsonantVoicing();
    repaint();
}

void PianoRollComponent::requestConsonantVoicing()
{
    std::vector<juce::File> files;
    if (consonantPitchDashed)
        for (const auto& track : snapshot.tracks) if (trackShowsAllNativeRegions(track))
            for (const auto& clip : track.clips) if (clip.sourceFile != juce::File())
                files.push_back(clip.sourceFile);
    consonantVoicing->request(files);
}

bool PianoRollComponent::diagnosticConsonantAnalysisPending() const
{
    return consonantVoicing->pending();
}

void PianoRollComponent::setPixelsPerSecond(float value)
{
    // The toolbar slider runs to 8000 for millisecond-level work; clamping
    // here to 600 quietly threw away everything above it.
    pixelsPerSecond = juce::jlimit(40.0f, 8000.0f, value);
    rebuildLayout();
}

void PianoRollComponent::setRowHeight(float value)
{
    rowHeight = juce::jlimit(12.0f, 48.0f, value);
    updateCanvasSize();
    repaint();
}

void PianoRollComponent::setSourceEditMode(bool enabled)
{
    if (sourceEditMode != enabled)
    {
        closeNativeOverlapSelector();
        trackPitchStrokeActive = false; trackPitchStrokeTrack.clear(); cancelPointDrag(); clearPointSelection();
    }
    if (enabled) { setDiffSingerPitchRestoreMode(false); setDiffSingerPitchOffsetMode(false); }
    sourceEditMode = enabled;
    if (sourceEditMode && focusedClip.isEmpty())
        for (const auto& track : snapshot.tracks)
            if (!track.clips.empty())
            {
                focusedClip = track.clips.front().id;
                break;
            }
    pruneHiddenNoteSelection();
    rebuildUtauSoundSpans();
    updateCanvasSize();
    repaint();
}

void PianoRollComponent::cancelPendingEditingForTrackChange()
{
    // Track/interface activation is view state: discard, never accept, any
    // editor that would otherwise commit when its old clip loses focus.
    finishInlineAliasEdit(false);
    closeNativeOverlapSelector();
    cancelContinuousFlag();cancelDiffSingerPitchOffsetGesture();
    pointGroupGesture.reset();trackPitchStrokeActive=false;trackPitchStrokeTrack.clear();
    cancelPointDrag();clearPointSelection();
    cvvcBoundaryDrag.reset();pendingCvvcBoundary.reset();cvvcLaneDrag.reset();pendingCvvcHead.clear();
    drawClipId.clear();drawLengthSeconds=drawAvailableSeconds=0;
    graphDragId.clear();graphDragOriginal.clear();graphDragIndex=-1;
    noteDragPassedThreshold=false;pointPitchHasMoved=false;envelopePointHasMoved=false;
    draggedVibratoHandle=VibratoHandle::none;nativeTrimDrag=false;
    marqueeShowsContextMenu=false;marqueeScrollStep=0;stopTimer();
    selectedNote.clear();selectedNotes.clear();hoverSeconds.reset();
    previewMoveDeltaSeconds=0;draggedSharedKey=0;
    repaint();
}

void PianoRollComponent::setFocusedClip(const juce::String& clipId)
{
    if (focusedClip != clipId)
    {
        closeNativeOverlapSelector();
        clearPointSelection();
        cancelContinuousFlag();
        cancelDiffSingerPitchOffsetGesture();
        finishInlineAliasEdit(true);
    }
    focusedClip = clipId;
    // A clip selected on another track must become visible before pruning
    // the old selection. Merged source parts share their parent's view id.
    for (const auto& track : snapshot.tracks)
        if (std::any_of(track.clips.begin(), track.clips.end(),
            [&](const auto& clip) { return clip.id == clipId; }))
        {
            setFocusedTrack(track.id);
            break;
        }
    pruneHiddenNoteSelection();
    updateCanvasSize();
    repaint();
}

void PianoRollComponent::setFocusedTrack(const juce::String& trackId)
{
    if (focusedTrack != trackId) { closeNativeOverlapSelector(); clearPointSelection(); }
    if (focusedTrack != trackId) { cancelContinuousFlag(); setDiffSingerPitchRestoreMode(false); setDiffSingerPitchOffsetMode(false); }
    if (focusedTrack != trackId && focusedClip.isNotEmpty())
    {
        auto belongs = false;
        for (const auto& track : snapshot.tracks)
            if (track.id == trackId)
                for (const auto& clip : track.clips) belongs = belongs || clip.id == focusedClip;
        if (!belongs) focusedClip.clear();
    }
    focusedTrack = trackId;
    pruneHiddenNoteSelection();
    rebuildNoteHits();
    repaint();
}

const TrackData* PianoRollComponent::currentPitchTrack() const
{
    for (const auto& track : snapshot.tracks) if (track.id == focusedTrack) return &track;
    for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips)
        if (clip.id == focusedClip) return &track;
    return nullptr;
}

void PianoRollComponent::setPitchScope(PitchScope scope)
{
    if (pitchScopeMode == scope) return;
    pitchScopeMode = scope;
    closeNativeOverlapSelector();
    trackPitchStrokeActive = false; trackPitchStrokeTrack.clear();
    cancelPointDrag(); clearPointSelection();
    // Scope is view state. Prune hidden selection silently: notifying the host
    // would change UTAU render selection merely because a View item was toggled.
    std::unordered_set<std::string> visible;
    for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips)
        if (clipIsVisible(track, clip)) for (const auto& note : clip.notes) visible.insert(note.id.toStdString());
    std::erase_if(selectedNotes, [&](const auto& id) { return !visible.contains(id); });
    if (selectedNote.isNotEmpty() && !visible.contains(selectedNote.toStdString()))
        selectedNote = selectedNotes.empty() ? juce::String{} : juce::String::fromUTF8(selectedNotes.begin()->c_str());
    rebuildNoteHits(); repaint();
}

bool PianoRollComponent::clipIsVisible(const TrackData& track, const ClipData& clip) const
{
    if (sourceEditMode) return clip.id == focusedClip;
    if (pitchScopeMode == PitchScope::currentTrack)
    {
        const auto* current = currentPitchTrack();
        return current != nullptr && current->id == track.id && track.compose && !track.accompaniment;
    }
    return track.compose && (focusedTrack.isEmpty() || track.id == focusedTrack)
        && (trackShowsAllNativeRegions(track) || focusedClip.isEmpty() || clip.id == focusedClip
            || (clip.showNormalDisplay && !track.accompaniment && track.id == focusedTrack));
}

void PianoRollComponent::pruneHiddenNoteSelection()
{
    std::unordered_set<std::string> visible;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
                for (const auto& note : clip.notes) visible.insert(note.id.toStdString());
    const auto previous = selectedNote;
    const auto count = selectedNotes.size();
    std::erase_if(selectedNotes, [&](const auto& id) { return !visible.contains(id); });
    if (selectedNote.isNotEmpty() && !visible.contains(selectedNote.toStdString()))
        selectedNote = selectedNotes.empty() ? juce::String{}
            : juce::String::fromUTF8(selectedNotes.begin()->c_str());
    if ((previous != selectedNote || count != selectedNotes.size()) && onNoteSelected)
        onNoteSelected(selectedNote);
}

void PianoRollComponent::setShowNoteLabels(bool enabled)
{
    if (showNoteLabels == enabled) return;
    showNoteLabels = enabled;
    repaint();
}

std::optional<juce::Range<double>> PianoRollComponent::selectedNotesTimeSpan() const
{
    std::optional<juce::Range<double>> span;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (!selectedNotes.contains(note.id.toStdString())) continue;
                const auto start = clip.startSeconds + note.startSeconds;
                juce::Range<double> range(start, start + note.durationSeconds);
                // A note reaches back before its written start for its
                // consonant, and may stop sounding before its written end.
                // The selection has been heard when all of that has been.
                if (const auto found = utauSoundSpans.find(note.id.toStdString());
                    found != utauSoundSpans.end())
                    range = range.getUnionWith(
                        juce::Range<double>(found->second.first, found->second.second));
                span = span ? span->getUnionWith(range) : range;
            }
    return span;
}

void PianoRollComponent::ensureDefaultEnvelope(const juce::String& noteId)
{
    // CVVC defaults belong to the generated phonemes and their mix windows.
    // Never bake the manual whole-note soft attack when committing lyrics,
    // including before asynchronous phoneme/waveform metadata has arrived.
    // Without a lyric a note has no voicebank entry, so nothing knows where it
    // starts or stops sounding and it is left carrying no envelope at all.
    // That is not the same as the shape it is drawn with: an absent envelope
    // is no shaping whatever, which is why such a note came out unlike its
    // neighbours until the soft rise was applied to it by hand.  Giving it a
    // lyric writes exactly the shape already on screen -- read back from the
    // same place the drawing comes from, so the two cannot disagree.
    if (noteId.isEmpty()) return;
    // The span follows from the lyric that was just set, and the layout that
    // knows about it would otherwise only be rebuilt on a later message.
    rebuildLayout();
    if (cvvcTimingEnabled(noteId)) return;
    if (utauSoundSpans.find(noteId.toStdString()) == utauSoundSpans.end()) return;
    for (const auto& track : snapshot.tracks)
    {
        if (!trackUsesVoicebankSynthesis(track) || trackIsDiffSinger(track)) continue;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (note.id != noteId) continue;
                // Already shaped, by hand or by a preset: leave it be.
                if (!note.amplitudeEnvelope.empty()) return;
                if (note.label.trim().isEmpty()) return;
                const auto absoluteStart = clip.startSeconds + note.startSeconds;
                model.setNotesAmplitudeEnvelopes(
                    { { noteId, amplitudeEnvelopeFor(note, absoluteStart) } });
                return;
            }
    }
}

std::optional<backend::UtauTailFadeSpan> PianoRollComponent::tailFadeSpanFor(const juce::String& id) const
{
    for(const auto& track:snapshot.tracks)
        if(track.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(track))
        for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(note.id==id)
        {
            if(!voicebankReadFor(track))return std::nullopt;
            const auto velocity=note.utauConsonantVelocity!=inheritedUtauConsonantVelocity?note.utauConsonantVelocity:track.utauConsonantVelocity;
            const auto timing=backend::UtauRenderer::sampleTiming(track.voicebankDirectory,note.label,note.midiNote,velocity,utauModeUsesRegions(track.utauMode),track.utauMode==UtauMode::mou,note.ownOto());
            if(!timing)return std::nullopt;
            const auto absolute=clip.startSeconds+note.startSeconds;
            auto first=-std::min(note.utauPreutteranceOverrideEnabled?std::max(0.0,note.utauPreutteranceSeconds):timing->preutteranceSeconds,std::max(0.0,absolute));
            auto last=note.durationSeconds;
            if(const auto span=utauSoundSpans.find(id.toStdString());span!=utauSoundSpans.end())
            {first=span->second.first-absolute;last=span->second.second-absolute;}
            const std::array<double,3> manual{note.utauJieSplit1,note.utauJieSplit2,note.utauJieSplit3};
            auto adjusted=*timing;adjusted.preutteranceSeconds=std::min(adjusted.preutteranceSeconds,std::max(0.0,absolute));
            return backend::UtauRenderer::tailFadeSpan(adjusted,first,last,note.durationSeconds,velocity,utauModeUsesRegions(track.utauMode),track.utauMode==UtauMode::mou,note.utauJieSplitSet?&manual:nullptr);
        }
    return std::nullopt;
}
std::optional<backend::UtauTailFadeSpan> PianoRollComponent::headEnvelopeSpanFor(const juce::String& id) const
{
    for(const auto& track:snapshot.tracks)
        if(track.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(track))
        for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(note.id==id)
        {
            if(!voicebankReadFor(track))return std::nullopt;
            const auto velocity=note.utauConsonantVelocity!=inheritedUtauConsonantVelocity?note.utauConsonantVelocity:track.utauConsonantVelocity;
            const auto timing=backend::UtauRenderer::sampleTiming(track.voicebankDirectory,note.label,note.midiNote,velocity,utauModeUsesRegions(track.utauMode),track.utauMode==UtauMode::mou,note.ownOto());
            if(!timing)return std::nullopt;
            const auto absolute=clip.startSeconds+note.startSeconds;
            auto first=-std::min(note.utauPreutteranceOverrideEnabled?std::max(0.0,note.utauPreutteranceSeconds):timing->preutteranceSeconds,std::max(0.0,absolute));
            auto last=note.durationSeconds;
            if(const auto span=utauSoundSpans.find(id.toStdString());span!=utauSoundSpans.end())
            {first=span->second.first-absolute;last=span->second.second-absolute;}
            const std::array<double,3> manual{note.utauJieSplit1,note.utauJieSplit2,note.utauJieSplit3};
            auto adjusted=*timing;adjusted.preutteranceSeconds=std::min(adjusted.preutteranceSeconds,std::max(0.0,absolute));
            return backend::UtauRenderer::headEnvelopeSpan(adjusted,first,last,note.durationSeconds,velocity,utauModeUsesRegions(track.utauMode),track.utauMode==UtauMode::mou,note.utauJieSplitSet?&manual:nullptr);
        }
    return std::nullopt;
}
std::vector<OtoRegionGuide> PianoRollComponent::otoRegionGuidesFor(const juce::String& id) const
{
    for (const auto& track : snapshot.tracks)
    {
        if (!trackUsesVoicebankSynthesis(track) || !utauModeUsesRegions(track.utauMode)
            || trackIsDiffSinger(track)) continue;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (note.id != id || backend::isRestLyric(note.label)) continue;
                const auto span = utauSoundSpans.find(id.toStdString());
                if (span == utauSoundSpans.end()) return {};
                const auto length = span->second.second - span->second.first;
                int count = 4;
                const auto fractions = jieFractionsFor(track, note, length, std::nullopt, &count);
                if (!fractions) return {};
                const auto relativeStart = span->second.first - (clip.startSeconds + note.startSeconds);
                static const std::array<const char*, 4> names { "声母", "介音", "韵腹", "韵尾" };
                std::vector<OtoRegionGuide> result;
                auto previous = relativeStart;
                for (int i = 0; i < count; ++i)
                {
                    const auto end = i + 1 == count ? relativeStart + length
                        : relativeStart + length * (*fractions)[static_cast<size_t>(i)];
                    const auto label = track.utauMode == UtauMode::mou
                        ? I18n::uiText("第") + juce::String(i + 1) + I18n::uiText("区")
                        : juce::String::fromUTF8(names[static_cast<size_t>(i)]);
                    result.push_back({ previous, end, label });
                    previous = end;
                }
                return result;
            }
    }
    return {};
}

int PianoRollComponent::applyTailFade(int mode)
{
    std::vector<juce::String> ids;for(const auto& id:selectedNoteIds())if(mode==0||tailFadeSpanFor(id))ids.push_back(id);
    return model.setNotesTailFade(ids,mode)?(int)ids.size():0;
}
std::vector<AmplitudeEnvelopePoint> PianoRollComponent::effectiveAmplitudeEnvelope(const NoteData& note,double absoluteStart,const std::vector<AmplitudeEnvelopePoint>& base) const
{
    juce::ignoreUnused(absoluteStart);
    if(nativeNoteTimingEnabled(note.id)&&note.nativeEnvelope.mode!=0)
        return backend::nativeEnvelopePicture(base,note.nativeEnvelope,note.durationSeconds);
    if(note.utauTailFadeMode!=0||note.utauTailFade.head.mode!=0)
    {
        const auto tail=tailFadeSpanFor(note.id),head=headEnvelopeSpanFor(note.id);
        return backend::tailFadePicture(base,note.utauTailFadeMode,tail?tail->startSeconds:0,tail?tail->endSeconds:0,
            note.utauTailFade,head?head->startSeconds:0,head?head->endSeconds:0);
    }
    return base;
}
std::vector<AmplitudeEnvelopePoint> PianoRollComponent::tailFadeBaseEnvelope(const juce::String& id) const
{
    for(const auto& track:snapshot.tracks)for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(note.id==id)
        return amplitudeEnvelopeFor(note,clip.startSeconds+note.startSeconds);
    return {};
}
std::vector<AmplitudeEnvelopePoint> PianoRollComponent::diagnosticTailFadePicture(const juce::String& id) const
{
    for(const auto& track:snapshot.tracks)for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(note.id==id)
    {const auto start=clip.startSeconds+note.startSeconds;return effectiveAmplitudeEnvelope(note,start,amplitudeEnvelopeFor(note,start));}
    return {};
}

const std::vector<PianoRollComponent::EnvelopePreset>& PianoRollComponent::envelopePresets()
{
    using From = EnvelopePresetPoint::From;
    // A share of the note's own level, in the dB a point holds.
    const auto percent = [](double value)
    {
        return static_cast<float>(20.0 * std::log10(value / 100.0));
    };
    const auto text = [](const char* utf8) { return I18n::uiText(utf8); };
    // Onsets first, then endings. Standard shares the new-phoneme linear
    // amplitude trapezoid. Soft onset preserves its existing dB interpolation.
    static const std::vector<EnvelopePreset> presets {
        { text("标准"), text("起音 5 ms，释放 35 ms（UTAU 默认）"),
          { { From::soundStart, 0.005, 0.0f, true }, { From::soundEnd, 0.035, 0.0f, true } }, true },
        { text("柔起"), text("起音 15 ms，释放 35 ms"),
          { { From::soundStart, 0.015, 0.0f }, { From::soundEnd, 0.035, 0.0f } } },
        // A soft vowel onset, the way OpenUtau's attack works: the consonant
        // comes in at 60% and the level climbs to full 50 ms into the vowel,
        // however long the consonant in front of it is.
        { text("缓起"), text("辅音以 60% 进入，拍点后 50 ms 升到满音量；释放 35 ms（按音量线性）"),
          { { From::soundStart, 0.005, percent(60.0), true },
            { From::beat, 0.050, 0.0f, true },
            { From::soundEnd, 0.035, 0.0f, true } }, true },
        { text("渐弱"), text("起音 5 ms，平台末尾降到 80%，释放 35 ms（按音量线性）"),
          { { From::soundStart, 0.005, 0.0f, true },
            { From::soundEnd, 0.035, percent(80.0), true } }, true },
        // A phrase ending.  Tuners fade the last note of a phrase over 70 to
        // 100 ms rather than UTAU's 35: down to half over the last 150 ms, then
        // out over the final 60.
        { text("句尾"), text("起音 5 ms；最后 150 ms 先降到 50%，再用 60 ms 淡出（按音量线性）"),
          { { From::soundStart, 0.005, 0.0f, true },
            { From::soundEnd, 0.150, 0.0f, true },
            { From::soundEnd, 0.060, percent(50.0), true } }, true },
        // A short tail that does not click.  5 ms straight in dB fell to half
        // level in half a millisecond, which is a cut; 10 ms straight in
        // amplitude is still short to the ear.
        { text("短收"), text("起音 5 ms，释放 10 ms（按音量线性，不爆音）"),
          { { From::soundStart, 0.005, 0.0f, true },
            { From::soundEnd, 0.010, 0.0f, true } }, true }
    };
    return presets;
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::envelopePresetPoints(
    const EnvelopePreset& preset, double first, double last)
{
    if(preset.name==envelopePresets().front().name)
    {
        std::vector<AmplitudeEnvelopePoint> points;
        for(const auto& p:backend::standardCvvcEnvelope(last-first))
            points.push_back({first+p.position*(last-first),p.value,p.linear});
        return points;
    }
    using From = EnvelopePresetPoint::From;
    std::vector<AmplitudeEnvelopePoint> points;
    points.reserve(preset.points.size() + 2);
    points.push_back({ first, -60.0f, preset.linearRise });
    for (const auto& point : preset.points)
        points.push_back({ point.from == From::soundStart ? first + point.seconds
                               : point.from == From::beat ? point.seconds
                                                          : last - point.seconds,
                           point.gainDb, point.linearToNext });
    points.push_back({ last, -60.0f });
    // A note too short to hold the shape would have its points cross.  It
    // gets the first of them a third of the way in and the last a third from
    // the end instead -- the fallback the default shape uses too.
    auto fits = true;
    for (std::size_t index = 2; index + 1 < points.size(); ++index)
        if (points[index].timeSeconds <= points[index - 1].timeSeconds + 0.005)
            fits = false;
    if (fits || preset.points.empty()) return points;
    const auto span = last - first;
    return { { first, -60.0f, preset.linearRise },
             { first + span / 3.0, preset.points.front().gainDb,
               preset.points.front().linearToNext },
             { first + span * 2.0 / 3.0, preset.points.back().gainDb,
               preset.points.back().linearToNext },
             { last, -60.0f } };
}

int PianoRollComponent::applyEnvelopePreset(const EnvelopePreset& preset)
{
    if (hasSelectedCvvcPhoneme())
        for (const auto& piece : cvvcLanePieces(true))
            if (piece.noteId == cvvcSelectedNote && piece.part == cvvcSelectedPart)
            {
                backend::CvvcEnvelopeCurve curve;
                const auto length = piece.to - piece.from;
                for (const auto& p : envelopePresetPoints(preset, 0, length))
                    curve.push_back({p.timeSeconds / length, p.gainDb, p.linearToNext});
                return model.setCvvcEnvelope(piece.noteId, piece.part, {}, std::move(curve), model.revisionNumber()) ? 1 : 0;
            }
    const auto selected = selectedNoteIds();
    if (selected.empty()) return 0;
    if(std::all_of(selected.begin(),selected.end(),[&](const auto& id){return usesCvvcEnvelope(id);}))
    {
        std::vector<ProjectModel::CvvcEnvelopeEdit> parts;
        for(const auto& piece:cvvcLanePieces(true))
            if(std::find(selected.begin(),selected.end(),piece.noteId)!=selected.end())
            {
                backend::CvvcEnvelopeCurve curve;
                const auto length=piece.to-piece.from;
                for(const auto& p:envelopePresetPoints(preset,0,length))curve.push_back({p.timeSeconds/length,p.gainDb,p.linearToNext});
                parts.push_back({piece.noteId,piece.part,{},std::move(curve)});
            }
        return model.setCvvcEnvelopes(std::move(parts),model.revisionNumber())?static_cast<int>(selected.size()):0;
    }
    std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> edits;
    for (const auto& track : snapshot.tracks)
    {
        if (track.accompaniment) continue;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (std::find(selected.begin(), selected.end(), note.id) == selected.end())
                    continue;
                const auto absoluteStart = clip.startSeconds + note.startSeconds;
                auto first = 0.0;
                auto last = std::max(0.02, note.durationSeconds);
                if (const auto span = utauSoundSpans.find(note.id.toStdString());
                    span != utauSoundSpans.end())
                {
                    first = std::min(0.0, span->second.first - absoluteStart);
                    last = std::max(first + 0.02, span->second.second - absoluteStart);
                }
                edits.emplace_back(note.id, envelopePresetPoints(preset, first, last));
            }
    }
    const auto count = static_cast<int>(edits.size());
    if (count > 0) model.setNotesAmplitudeEnvelopes(std::move(edits));
    return count;
}

int PianoRollComponent::applyEnvelopePreset(double attackSeconds, double releaseSeconds,
                                            float plateauEndDb)
{
    EnvelopePreset preset;
    preset.points = { { EnvelopePresetPoint::From::soundStart, attackSeconds, 0.0f },
                      { EnvelopePresetPoint::From::soundEnd, releaseSeconds, plateauEndDb } };
    return applyEnvelopePreset(preset);
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::envelopeWithPointAt(
    std::vector<AmplitudeEnvelopePoint> envelope, double timeSeconds)
{
    // On the line where it already runs, and the stretch it lands in keeps its
    // shape on both sides of it: a point added to a ramp straight in amplitude
    // leaves both halves of that ramp straight in amplitude.
    const auto after = std::upper_bound(envelope.begin(), envelope.end(), timeSeconds,
        [](double value, const AmplitudeEnvelopePoint& point)
        {
            return value < point.timeSeconds;
        });
    const auto linear = after != envelope.begin() && after != envelope.end()
        && std::prev(after)->linearToNext;
    const AmplitudeEnvelopePoint added { timeSeconds, amplitudeDbAt(envelope, timeSeconds),
                                         linear, true };
    envelope.insert(after, added);
    return envelope;
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::diagnosticMapAmplitudeEnvelope(
    const std::vector<AmplitudeEnvelopePoint>& source, const juce::String& sourceNoteId,
    const juce::String& targetNoteId) const
{
    return mapAmplitudeEnvelopeToNote(source, sourceNoteId, targetNoteId);
}

void PianoRollComponent::setShowNoteRange(bool enabled)
{
    if (showNoteRange == enabled) return;
    showNoteRange = enabled;
    repaint();
}

void PianoRollComponent::setShowEnvelope(bool enabled)
{
    if (showEnvelope == enabled) return;
    showEnvelope = enabled;
    repaint();
}

void PianoRollComponent::setShowPitchLine(bool enabled)
{
    if (showPitchLine == enabled) return;
    clearPointSelection();
    if (!enabled) { setDiffSingerPitchRestoreMode(false); setDiffSingerPitchOffsetMode(false); }
    showPitchLine = enabled;
    repaint();
}

void PianoRollComponent::setShowWaveforms(bool enabled)
{
    if (showWaveforms == enabled) return;
    showWaveforms = enabled;
    repaint();
}

void PianoRollComponent::setSampleRegions(const std::vector<SampleRegionSetting>& regions,
                                          int activeRegion)
{
    sampleRegions = regions;
    activeSampleRegion = juce::jlimit(-1, static_cast<int>(sampleRegions.size()) - 1,
                                      activeRegion);
    repaint();
}

void PianoRollComponent::setPlayheadSeconds(double seconds)
{
    // Thirty times a second, playing or not.  What actually changes is a one
    // pixel line and the mark on the edge that says which way it lies when it
    // is off screen, so repainting the whole roll for it spent a full window
    // repaint on every tick.  The band covers where the line was as well as
    // where it now is: on a scrolling view the old one is blitted along with
    // everything else and has to be painted over.
    if (std::abs(seconds - playheadSeconds) < 1.0e-9) return;
    const auto before = timeToX(playheadSeconds);
    playheadSeconds = seconds;
    const auto after = timeToX(playheadSeconds);
    const auto left = static_cast<int>(std::floor(std::min(before, after))) - 3;
    const auto right = static_cast<int>(std::ceil(std::max(before, after))) + 3;
    playheadBand = { left, 0, std::max(1, right - left), getHeight() };
    repaint(playheadBand);
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
    {
        const auto x = viewport->getViewPositionX();
        const auto y = viewport->getViewPositionY();
        repaint(x + 50, y, 24, 40);
        repaint(x + viewport->getViewWidth() - 24, y, 24, 40);
    }
}

bool PianoRollComponent::diffSingerPitchRestoreAvailable() const
{
    for (const auto& track : snapshot.tracks)
        if ((focusedTrack.isEmpty() || track.id == focusedTrack) && trackIsDiffSinger(track))
            for (const auto& clip : track.clips) for (const auto& note : clip.notes)
                if (!note.diffSingerPitchReference.empty()) return true;
    return false;
}

void PianoRollComponent::setDiffSingerPitchRestoreMode(bool enabled)
{
    enabled = enabled && !sourceEditMode && tool == Tool::points && diffSingerPitchRestoreAvailable();
    if (enabled) setDiffSingerPitchOffsetMode(false);
    if (dsPitchRestoreMode == enabled) return;
    dsPitchRestoreMode = enabled;
    draggedNote.clear(); draggedPitchAnchor = -1; pitchStroke.clear(); dragMode = DragMode::none;
    if (onPitchPointModeChanged) onPitchPointModeChanged();
    repaint();
}

void PianoRollComponent::setNativeLoudnessMode(bool active)
{
    if (nativeLoudnessMode == active) return;
    if (dragMode == DragMode::nativeLoudness) { draggedNote.clear(); dragMode = DragMode::none; }
    nativeLoudnessMode = active;
    setMouseCursor(juce::MouseCursor::NormalCursor);
    repaint();
}

float PianoRollComponent::displayedNativeGain(const NoteData& note) const
{
    if (dragMode == DragMode::nativeLoudness)
        if (const auto it = loudnessStartGains.find(note.id); it != loudnessStartGains.end())
        {
            if (std::abs(loudnessDeltaDb) < 1.e-6f) return it->second;
            const auto db = juce::jlimit(-60.0f, juce::Decibels::gainToDecibels(4.0f),
                juce::Decibels::gainToDecibels(it->second, -60.0f) + loudnessDeltaDb);
            return db <= -60.0f ? 0.0f : juce::Decibels::decibelsToGain(db);
        }
    return note.gain;
}

void PianoRollComponent::setTool(Tool nextTool)
{
    if (nextTool != Tool::note) setNativeLoudnessMode(false);
    if (tool != nextTool) closeNativeOverlapSelector();
    if (tool != nextTool && trackPitchStrokeActive)
    {
        trackPitchStrokeActive = false; trackPitchStrokeTrack.clear(); pitchStroke.clear(); dragMode = DragMode::none;
    }
    if (tool != nextTool) clearPointSelection();
    if (tool != nextTool) { cancelContinuousFlag(); cvvcLaneDrag.reset();
        if(cvvcBoundaryDrag){cvvcBoundaryDrag.reset();refreshFromModel();} }
    if (nextTool != Tool::points) { setDiffSingerPitchRestoreMode(false); setDiffSingerPitchOffsetMode(false); }
    tool = nextTool;
    setMouseCursor(tool == Tool::draw || tool == Tool::line || tool == Tool::points
                       || tool == Tool::amplitude || tool == Tool::flagCurve
                       ? juce::MouseCursor::CrosshairCursor
                       : juce::MouseCursor::NormalCursor);
    repaint();
}

void PianoRollComponent::setDrawAction(DrawAction action)
{
    if (drawActionMode == action) return;
    drawActionMode = action;
    trackPitchStrokeActive = false; trackPitchStrokeTrack.clear();
    // A tool choice cancels only an in-flight drawing preview. It must not
    // publish an unfinished curve/note or create a project undo transaction.
    if (dragMode == DragMode::drawPitch || dragMode == DragMode::drawNewNote)
    {
        dragMode = DragMode::none;
        draggedNote.clear(); pitchStroke.clear();
        drawClipId.clear(); drawLengthSeconds = 0.0;
    }
    repaint();
}

std::pair<double,double> PianoRollComponent::trackPitchEditDomain(const NoteData& note)
{
    double first=0,last=note.durationSeconds;
    // Authored sparse controls may reach outside nominal notes. Derived source
    // support is not an authored extension and never grows the editable domain.
    for(const auto& point:note.pitchControlPoints)
    {
        if(!std::isfinite(point.timeSeconds))
            return {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};
        first=std::min(first,point.timeSeconds);last=std::max(last,point.timeSeconds);
    }
    return {first,last};
}

bool PianoRollComponent::beginTrackPitchStroke(const juce::MouseEvent& event)
{
    if (sourceEditMode || pitchScopeMode != PitchScope::currentTrack || event.position.x < 58) return false;
    refreshFromModel();
    const auto* track = currentPitchTrack();
    if (!track || !track->compose || track->accompaniment) return false;
    clearPointSelection();
    draggedNote.clear(); draggedPitchAnchor = -1;
    trackPitchStrokeActive = true; trackPitchStrokeTrack = track->id;
    trackPitchStrokeRevision = model.revisionNumber(); pitchEditAbsoluteStart = 0;
    const auto absolute = std::max(0.0, static_cast<double>(event.position.x - 58) / pixelsPerSecond);
    pitchStroke = {{absolute,juce::jlimit(0.f,127.f,pitchMidiFromY(event.position.y)),PitchCurveShape::linear}};
    dragMode = tool == Tool::line ? DragMode::linePitch : DragMode::drawPitch;
    repaint(); return true;
}

bool PianoRollComponent::planTrackPitchStroke(std::vector<ProjectModel::PointCurveEdit>& edits,juce::String& error)
{
    edits.clear();error.clear();if(!trackPitchStrokeActive||pitchStroke.size()<2)return false;
    auto stroke=pitchStroke;
    std::stable_sort(stroke.begin(),stroke.end(),[](const auto& a,const auto& b){return a.timeSeconds<b.timeSeconds;});
    std::vector<PitchCurveEditPoint> ordered;
    for(const auto& point:stroke)
    {if(!ordered.empty()&&std::abs(ordered.back().timeSeconds-point.timeSeconds)<1.e-9)ordered.back()=point;else ordered.push_back(point);}
    if(ordered.size()<2||ordered.back().timeSeconds-ordered.front().timeSeconds<=1.e-9)return false;
    const auto track=std::find_if(snapshot.tracks.begin(),snapshot.tracks.end(),[&](const auto& t){return t.id==trackPitchStrokeTrack;});
    if(track==snapshot.tracks.end()||!track->compose||track->accompaniment)return false;
    const auto& graphs=connectionCurvesFor(*track,false);
    for(const auto& clip:track->clips)for(const auto& note:clip.notes)
    {
        if(note.nativeUnpitched)continue;
        const auto start=clip.startSeconds+note.startSeconds;const auto domain=trackPitchEditDomain(note);
        if(!std::isfinite(domain.first)||!std::isfinite(domain.second)){error="Invalid authored pitch domain";return false;}
        const auto first=std::max(start+domain.first,ordered.front().timeSeconds);
        const auto last=std::min(start+domain.second,ordered.back().timeSeconds);if(last-first<=1.e-9)continue;
        std::vector<juce::Range<double>> pieces{{first,last}};
        for(const auto& graph:graphs)if(graph.active)
        {
            std::vector<juce::Range<double>> outside;
            for(const auto& piece:pieces)
            {
                if(graph.toSeconds<=piece.getStart()||graph.fromSeconds>=piece.getEnd()){outside.push_back(piece);continue;}
                if(graph.fromSeconds>piece.getStart()+1.e-9)outside.emplace_back(piece.getStart(),graph.fromSeconds);
                if(graph.toSeconds<piece.getEnd()-1.e-9)outside.emplace_back(graph.toSeconds,piece.getEnd());
            }
            pieces=std::move(outside);
        }
        auto staged=note;std::optional<ProjectModel::PointCurveEdit> sparseFinal;bool authoredSupportInStroke=false;
        for(const auto& piece:pieces)
        {
            const auto from=piece.getStart(),to=piece.getEnd();if(to-from<=1.e-9)continue;
            // Graph-window boundaries preserve hidden base controls. Only a
            // genuine outside-domain clip inside the overall stroke releases a
            // note-edge guard; graph endpoints are owned/reanchored by Model.
            const bool preserveLeft=!(std::abs(from-(start+domain.first))<1.e-9&&from>ordered.front().timeSeconds+1.e-9);
            const bool preserveRight=!(std::abs(to-(start+domain.second))<1.e-9&&to<ordered.back().timeSeconds-1.e-9);
            std::vector<PitchCurveEditPoint> local;
            if(!sliceNativePitchCurve(ordered,from,to,local)){error="Pitch stroke cannot be represented exactly";return false;}
            for(auto& p:local){p.timeSeconds+=preserveLeft?from-start:domain.first;p.targetMidi=juce::jlimit(0.f,127.f,p.targetMidi-displayedPitchOffset(note,p.timeSeconds));}
            if(!preserveLeft)local.front().timeSeconds=domain.first;if(!preserveRight)local.back().timeSeconds=domain.second;
            ProjectModel::PointCurveEdit edit;edit.kind=ProjectModel::PointCurveEdit::pitch;edit.noteId=note.id;
            // Intent comes from the actual painted interval, never from old
            // controls retained outside it by the exact composer.
            authoredSupportInStroke|=local.front().timeSeconds < -1.e-9 || local.back().timeSeconds > note.durationSeconds+1.e-9;
            edit.authorPitchSupport=authoredSupportInStroke;
            if(!composeNativePitchStroke(staged,local,edit.pitchPoints,edit.storePitchControlPoints,error,preserveLeft,preserveRight))return false;
            if(edit.storePitchControlPoints){staged.pitchControlPoints=edit.pitchPoints;sparseFinal=std::move(edit);}
            else edits.push_back(std::move(edit)); // disjoint dense windows retain all other facts
        }
        if(sparseFinal)edits.push_back(std::move(*sparseFinal));
    }
    for(const auto& graph:graphs)if(graph.active&&ordered.back().timeSeconds>graph.fromSeconds&&ordered.front().timeSeconds<graph.toSeconds)
    {
        ProjectModel::PointCurveEdit edit;edit.kind=ProjectModel::PointCurveEdit::connectionPitchStroke;
        edit.connectionId=graph.connectionId;edit.pitchPoints=ordered;edits.push_back(std::move(edit));
    }
    return !edits.empty();
}
bool PianoRollComponent::commitTrackPitchStroke()
{
    std::vector<ProjectModel::PointCurveEdit> edits;juce::String error;
    if(!planTrackPitchStroke(edits,error)){if(onEditRejected&&!error.isEmpty())onEditRejected(error);return false;}
    const bool committed=model.setPointCurves(std::move(edits),trackPitchStrokeRevision,&error);
    if(!committed&&onEditRejected&&!error.isEmpty())onEditRejected(error);
    if(committed)refreshFromModel();return committed;
}

const NoteData* PianoRollComponent::findNote(const juce::String& noteId, bool* isUtau) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId)
                {
                    if (isUtau != nullptr)
                        *isUtau = trackUsesVoicebankSynthesis(track);
                    return &note;
                }
    if (isUtau != nullptr) *isUtau = false;
    return nullptr;
}

std::vector<PianoRollComponent::PositionedUtauNote>
PianoRollComponent::positionedUtauNotesFor(const juce::String& noteId) const
{
    std::vector<PositionedUtauNote> result;
    for (const auto& track : snapshot.tracks)
    {
        if (!trackUsesVoicebankSynthesis(track)) continue;
        auto containsTarget = false;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                containsTarget = containsTarget || note.id == noteId;
        if (!containsTarget) continue;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                const auto start = clip.startSeconds + note.startSeconds;
                result.push_back({ note.id, start, start + note.durationSeconds,
                                   note.utauAutoPitchTransition });
            }
        break;
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& left, const auto& right)
    {
        if (std::abs(left.startSeconds - right.startSeconds) > 1.0e-9)
            return left.startSeconds < right.startSeconds;
        return left.endSeconds < right.endSeconds;
    });
    return result;
}

std::optional<PianoRollComponent::PositionedUtauNote>
PianoRollComponent::previousUtauNoteFor(const juce::String& noteId) const
{
    const auto notes = positionedUtauNotesFor(noteId);
    const auto found = std::find_if(notes.begin(), notes.end(),
        [&](const auto& note) { return note.id == noteId; });
    if (found == notes.end() || found == notes.begin()) return std::nullopt;
    return *std::prev(found);
}

std::optional<PianoRollComponent::PositionedUtauNote>
PianoRollComponent::nextUtauNoteFor(const juce::String& noteId) const
{
    const auto notes = positionedUtauNotesFor(noteId);
    const auto found = std::find_if(notes.begin(), notes.end(),
        [&](const auto& note) { return note.id == noteId; });
    if (found == notes.end() || std::next(found) == notes.end()) return std::nullopt;
    return *std::next(found);
}

bool PianoRollComponent::formsAdjacentPitchBoundary(const PositionedUtauNote& left,
                                                     const PositionedUtauNote& right)
{
    // The same rule the renderer bridges by: touching, and both taking the
    // editor's transition.  A UST's notes meet at the pitch their file gives.
    return std::abs(left.endSeconds - right.startSeconds) <= 0.002
        && left.automaticTransition && right.automaticTransition;
}

std::optional<PianoRollComponent::IncomingJoinGlide>
PianoRollComponent::incomingJoinGlideFor(const NoteData& note)
{
    // A native connection (Melodyne pitch join or a forced connection) means
    // the renderer glides this note's head out of the previous note's tail
    // pitch.  The displayed pitch line has to do the same, or a connected note
    // shows a step the ear never hears.  UTAU adjacency is handled separately
    // by the crossfade bridge, so this is for the connection-flag path only.
    if (!note.connectedToPrevious) return std::nullopt;
    const NoteData* previousNote = nullptr;
    auto previousEnd = -std::numeric_limits<double>::infinity();
    for (const auto& track : snapshot.tracks)
    {
        auto ownsNote = false;
        for (const auto& clip : track.clips)
            for (const auto& candidate : clip.notes)
                ownsNote = ownsNote || candidate.id == note.id;
        if (!ownsNote) continue;
        // The joined start is this note's absolute position on its own track.
        double joinedStart = 0.0;
        for (const auto& clip : track.clips)
            for (const auto& candidate : clip.notes)
                if (candidate.id == note.id)
                    joinedStart = clip.startSeconds + candidate.startSeconds;
        for (const auto& clip : track.clips)
            for (const auto& candidate : clip.notes)
            {
                if (candidate.id == note.id) continue;
                const auto end = clip.startSeconds + candidate.startSeconds
                    + candidate.durationSeconds;
                if (end <= joinedStart + 0.002 && end > previousEnd)
                {
                    previousEnd = end;
                    previousNote = &candidate;
                }
            }
        break;
    }
    if (previousNote == nullptr) return std::nullopt;
    // The previous note's tail pitch is the last thing its own displayed line
    // reaches, so the two meet exactly.
    const auto& previousAnchors = pitchAnchorsFor(*previousNote);
    const auto leadMidi = previousAnchors.empty()
        ? static_cast<double>(previousNote->midiNote)
        : static_cast<double>(previousAnchors.back().targetMidi);
    // The same join window the renderer uses (AudioEngine::makeRenderRequest).
    const auto joinSeconds = std::min(0.08,
        std::max(0.012, note.durationSeconds * 0.22));
    return IncomingJoinGlide { leadMidi, joinSeconds };
}

float PianoRollComponent::pitchAt(const std::vector<PitchCurveEditPoint>& anchors,
                                  double timeSeconds)
{
    return evaluatePitchCurve(anchors, timeSeconds);
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::diagnosticDrawnEnvelope(
    const juce::String& id) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == id)
                    return amplitudeEnvelopeFor(note,
                        clip.startSeconds + note.startSeconds);
    return {};
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::amplitudeEnvelopeFor(
    const NoteData& note, double absoluteStart) const
{
    if(pointGroupGesture)for(const auto& d:pointGroupGesture->curves)
        if(d.identity.kind==PointSelection::amplitude)
        {
            if(d.identity.noteId==note.id)return d.edit.amplitudePoints;
            const auto* a=nativeEnvelopeMember(d.identity.noteId);const auto* b=nativeEnvelopeMember(note.id);
            if(a&&b&&a->points==b->points)
            {auto points=d.edit.amplitudePoints;for(auto& p:points)p.timeSeconds+=a->noteStart-b->noteStart;return points;}
        }
    // Everything that reads a note's envelope comes through here, so the base
    // value is applied in this one place: the lane's line, its handles, the
    // waveform behind it and the note's own row all show what will be heard.
    // commitAmplitudeEnvelopeToSelection takes it back off before storing, so
    // dragging a point still moves the point rather than the base.
    const auto withBase = [&note](std::vector<AmplitudeEnvelopePoint> points)
    {
        return scaledAmplitudeEnvelope(points, note.amplitudeEnvelopeBasePercent);
    };
    if (diffSingerNoteIds.contains(note.id.toStdString()))
    {
        auto from=0.0, to=note.durationSeconds;
        if (const auto span=utauSoundSpans.find(note.id.toStdString());span!=utauSoundSpans.end())
        {from=span->second.first-absoluteStart;to=span->second.second-absoluteStart;}
        if (note.amplitudeEnvelope.empty()) return withBase({{from,0,true},{to,0,true}});
        return withBase(backend::diffSingerEnvelope(note.amplitudeEnvelope,from,to));
    }
    bool isUtau = false;
    (void) findNote(note.id, &isUtau);
    if (!isUtau)
    {
        if (const auto* member = nativeEnvelopeMember(note.id))
            return member->relativePoints();
        // Analysis only describes the recording. An absent native envelope
        // means unity, including both edges; displaying it must not add fades.
        if (note.amplitudeEnvelope.empty())
            return withBase({{0.0, 0.0f, true}, {note.durationSeconds, 0.0f, true}});
        return withBase(note.amplitudeEnvelope);
    }
    if (!note.amplitudeEnvelope.empty())
    {
        auto points = note.amplitudeEnvelope;
        // The two outer points mark where the note starts and stops sounding
        // rather than shaping anything, and that span moves on its own: a
        // slower consonant reaches further back for its lead-in, and giving
        // the next note one lets this one sound for longer.  The ends have to
        // follow, or the note would be silenced before it really ends.
        //
        // They move as whole ramps, though.  Pinning only the outermost point
        // stretched the fade it belongs to -- lengthen the lead-in and the
        // rise laid down by a preset grew with it, which is a change of shape
        // and not the lengthening that was asked for.  The opening and
        // closing ramps keep the slope they were given; the plateau between
        // them takes up the difference, and points placed inside stay put.
        if (const auto span = utauSoundSpans.find(note.id.toStdString());
            span != utauSoundSpans.end() && points.size() >= 2)
        {
            const auto first = std::min(0.0, span->second.first - absoluteStart);
            const auto last = std::max(first + 0.02,
                                       span->second.second - absoluteStart);
            const auto startShift = first - points.front().timeSeconds;
            const auto endShift = last - points.back().timeSeconds;
            // With three points the middle one closes the rise and opens the
            // fall at once, so it cannot follow both ends; it is left alone.
            if (points.size() >= 4)
            {
                points[1].timeSeconds += startShift;
                points[points.size() - 2].timeSeconds += endShift;
            }
            points.front().timeSeconds = first;
            points.back().timeSeconds = last;
            // A note can also become shorter than the ramps it was given.
            // Nothing may overtake what comes before it, and nothing may pass
            // the end.
            auto earliest = first;
            for (auto& point : points)
            {
                point.timeSeconds = juce::jlimit(earliest, last, point.timeSeconds);
                earliest = point.timeSeconds;
            }
            points.back().timeSeconds = last;
            // And a spliced boundary crosses here too.  This used to be done
            // only for notes carrying no envelope of their own, so once typing
            // a lyric began writing one, splicing stopped showing at all -- the
            // sound crossfaded while the drawing said otherwise.
            //
            // A crossfade is the one thing that does change a ramp's slope:
            // the rise has to reach the instant the note before stops
            // sounding, and the fall has to start where the next one begins,
            // whatever slope was stored.  Points placed inside the note are
            // left where they are.
            if (points.size() >= 4)
            {
                const auto crossing = spliceCrossingFor(
                    note, absoluteStart, points.front().timeSeconds,
                    points.back().timeSeconds);
                if (crossing.riseEnd) points[1].timeSeconds = *crossing.riseEnd;
                if (crossing.fallStart)
                    points[points.size() - 2].timeSeconds = *crossing.fallStart;
                auto earliest = points.front().timeSeconds;
                for (auto& point : points)
                {
                    point.timeSeconds = juce::jlimit(earliest, last, point.timeSeconds);
                    earliest = point.timeSeconds;
                }
            }
        }
        return withBase(points);
    }
    auto firstTime = 0.0;
    auto lastTime = std::max(0.02, note.durationSeconds);
    if (const auto span = utauSoundSpans.find(note.id.toStdString());
        span != utauSoundSpans.end())
    {
        firstTime = std::min(0.0, span->second.first - absoluteStart);
        // The sounding end, on whichever side of the nominal end it falls: a
        // note handing over to the next one stops early, and putting the
        // release at the nominal end would place the fall in a stretch that
        // never plays, leaving a flat line where the note should die away.
        lastTime = std::max(firstTime + 0.02, span->second.second - absoluteStart);
    }

    // UTAU-style default: silence at the real sample boundaries with a short
    // attack/release and a wide unity plateau.  A negative first point keeps
    // the preutterance visible, so adjacent notes may overlap naturally.
    const auto spanSeconds = std::max(0.02, lastTime - firstTime);
    // All UTAU modes start with the standard linear-amplitude envelope.
    // Soft onset is an explicit preset, never an implicit lyric side effect.
    constexpr auto attackSeconds = 0.005;
    constexpr auto releaseSeconds = 0.035;
    auto attackEnd = firstTime + attackSeconds;
    auto releaseStart = lastTime - releaseSeconds;
    // A spliced boundary crossfades the overlap -- the stretch both notes
    // sound through -- and nothing else.  It is bounded by where the later
    // note starts sounding and where the earlier one stops, and both sides
    // read those same two instants, so the rise and the fall are mirror
    // images and the two envelopes really cross.  Neither span moves: an
    // overlap the oto does not leave has nothing to fade across, and the
    // notes keep their ordinary ramps.
    {
        const auto crossing = spliceCrossingFor(note, absoluteStart,
                                                firstTime, lastTime);
        if (crossing.riseEnd) attackEnd = *crossing.riseEnd;
        if (crossing.fallStart) releaseStart = *crossing.fallStart;
    }
    if (releaseStart <= attackEnd + 0.005)
    {
        attackEnd = firstTime + spanSeconds / 3.0;
        releaseStart = firstTime + spanSeconds * 2.0 / 3.0;
    }
    return withBase({ { firstTime, -60.0f, true }, { attackEnd, 0.0f, true },
                      { releaseStart, 0.0f, true }, { lastTime, -60.0f, true } });
}

const NativeSharedEnvelopeMember* PianoRollComponent::nativeEnvelopeMember(const juce::String& id) const
{
    for (const auto& track : snapshot.tracks)
    {
        if (!trackShowsAllNativeRegions(track)) continue;
        auto found = nativeEnvelopeCache.find(track.id);
        if (found == nativeEnvelopeCache.end())
            found = nativeEnvelopeCache.emplace(track.id, nativeSharedEnvelopes(track)).first;
        if (const auto member = found->second.find(id); member != found->second.end()) return &member->second;
    }
    return nullptr;
}

float PianoRollComponent::amplitudeDbAt(
    const std::vector<AmplitudeEnvelopePoint>& points, double timeSeconds)
{
    if (points.empty()) return 0.0f;
    const auto right = std::upper_bound(points.begin(), points.end(), timeSeconds,
        [](double value, const AmplitudeEnvelopePoint& point)
        {
            return value < point.timeSeconds;
        });
    if (right == points.begin()) return right->gainDb;
    if (right == points.end()) return points.back().gainDb;
    const auto& left = *std::prev(right);
    const auto span = right->timeSeconds - left.timeSeconds;
    const auto amount = span > 1.0e-9
        ? static_cast<float>(juce::jlimit(0.0, 1.0,
            (timeSeconds - left.timeSeconds) / span)) : 0.0f;
    return backend::envelopeDbBetween(left.gainDb, right->gainDb, amount, left.linearToNext);
}

float PianoRollComponent::amplitudeY(float midi, float gainDb) const
{
    // Unity sits on the top edge of the note block and silence on its bottom,
    // so an ordinary envelope fills the block rather than hovering across its
    // middle.  The scale still runs to 200%, which simply puts anything above
    // unity in the block's height again above it -- louder reads as taller
    // than the note.
    const auto blockHeight = std::max(4.0f, rowHeight - 6.0f);
    const auto top = midiToY(midi) + 3.0f - blockHeight;
    const auto normalized = juce::jlimit(0.0f, 1.0f,
        amplitudePercentFromDb(gainDb) / 200.0f);
    return top + (1.0f - normalized) * blockHeight * 2.0f;
}

float PianoRollComponent::amplitudeDbFromY(float midi, float y) const
{
    // The exact inverse of amplitudeY, so a handle lands where it was dropped.
    const auto blockHeight = std::max(4.0f, rowHeight - 6.0f);
    const auto top = midiToY(midi) + 3.0f - blockHeight;
    const auto normalized = juce::jlimit(0.0f, 1.0f,
        1.0f - (y - top) / (blockHeight * 2.0f));
    return amplitudeDbFromPercent(normalized * 200.0f);
}

PianoRollComponent::SpliceCrossing PianoRollComponent::spliceCrossingFor(
    const NoteData& note, double absoluteStart,
    double firstTime, double lastTime) const
{
    SpliceCrossing crossing;
    const auto neighbours = utauNeighbours.find(note.id.toStdString());
    if (neighbours == utauNeighbours.end()) return crossing;
    if (note.utauSplice && !neighbours->second.previous.empty())
        if (const auto span = utauSoundSpans.find(neighbours->second.previous);
            span != utauSoundSpans.end())
        {
            const auto riseEnd = span->second.second - absoluteStart;
            if (riseEnd > firstTime + 0.003 && riseEnd < lastTime)
                crossing.riseEnd = riseEnd;
        }
    if (neighbours->second.nextSpliced)
        if (const auto span = utauSoundSpans.find(neighbours->second.next);
            span != utauSoundSpans.end())
        {
            const auto fallStart = span->second.first - absoluteStart;
            if (fallStart > firstTime && fallStart < lastTime - 0.003)
                crossing.fallStart = fallStart;
        }
    return crossing;
}

bool PianoRollComponent::keyboardLabelVisible(int midi, float rowHeight)
{
    if (midi % 12 == 0) return true;                 // C always marks the octave
    if (rowHeight >= 11.0f) return true;             // room for all twelve
    if (rowHeight >= 7.5f) return !juce::MidiMessage::isMidiNoteBlack(midi);
    return false;
}

juce::String PianoRollComponent::amplitudeZoomLabel(bool zoomIn)
{
    return juce::String::fromUTF8(zoomIn ? "+" : "−");
}

juce::Rectangle<float> PianoRollComponent::flagLaneBounds() const
{
    // The same strip the loudness envelope uses: one lane, one place, so the
    // two read alike and there is never a question of which is on top.
    return amplitudeLaneBounds();
}

namespace {
juce::String pairedDiffSingerFlag(const juce::String& key)
{
    for (const auto& kind : diffSingerParameterKinds())
    {
        const auto actual = juce::String(kind.flag), offset = "DS:" + actual.substring(7);
        if (key == actual) return offset;
        if (key == offset) return actual;
    }
    return {};
}
}

juce::Rectangle<float> PianoRollComponent::flagLanePlotBounds() const
{
    auto plot = amplitudeLanePlotBounds();
    if (diffSingerFlagContext && pairedDiffSingerFlag(laneFlag).isNotEmpty()) plot.removeFromRight(48);
    return plot;
}

std::pair<float, float> PianoRollComponent::flagLaneWindow() const
{
    const auto& kind = flagCurveKindFor(laneFlag);
    const auto full = std::max(1.0e-3f, kind.maximum - kind.minimum);
    const auto span = full / juce::jlimit(1.0f, 16.0f, laneZoom);
    // At 1x this pins the window to the whole range whatever the centre says,
    // so the unzoomed lane is exactly what it always was.
    const auto low = juce::jlimit(kind.minimum, kind.maximum - span,
                                  laneCentre - span * 0.5f);
    return { low, low + span };
}

float PianoRollComponent::flagLaneY(float value) const
{
    const auto plot = flagLanePlotBounds();
    const auto [low, high] = flagLaneWindow();
    const auto span = std::max(1.0e-3f, high - low);
    const auto normalized = juce::jlimit(0.0f, 1.0f, (value - low) / span);
    return plot.getBottom() - normalized * plot.getHeight();
}

float PianoRollComponent::flagValueFromLaneY(float y) const
{
    const auto plot = flagLanePlotBounds();
    const auto& kind = flagCurveKindFor(laneFlag);
    if (plot.getHeight() <= 1.0f) return kind.minimum;
    const auto [low, high] = flagLaneWindow();
    // Held to what the flag allows, not to what is on show: a handle dragged
    // above the window still reads as the value it is heading for, and the
    // window is moved to follow it rather than pinning it at the edge.
    const auto normalized = (plot.getBottom() - y) / plot.getHeight();
    return juce::jlimit(kind.minimum, kind.maximum,
                        low + normalized * (high - low));
}

void PianoRollComponent::centreFlagLaneOnCurves()
{
    // Where the curves actually sit, so zooming in lands on the work rather
    // than on the middle of a range whose other half is empty.
    auto total = 0.0;
    auto count = 0;
    for (const auto& hit : noteHits)
    {
        bool utau = false;
        const auto* note = findNote(hit.id, &utau);
        if (note == nullptr || !utau) continue;
        for (const auto& point : flagLaneCurveFor(*note))
        {
            total += point.value;
            ++count;
        }
    }
    const auto& kind = flagCurveKindFor(laneFlag);
    laneCentre = count > 0
        ? static_cast<float>(total / count)
        : kind.defaultValue;
}

void PianoRollComponent::keepFlagValueInView(float value)
{
    const auto [low, high] = flagLaneWindow();
    const auto span = high - low;
    const auto margin = span * 0.08f;
    if (value > high - margin) laneCentre = value + span * 0.5f - margin;
    else if (value < low + margin) laneCentre = value - span * 0.5f + margin;
    else return;
    const auto& kind = flagCurveKindFor(laneFlag);
    laneCentre = juce::jlimit(kind.minimum + span * 0.5f,
                              kind.maximum - span * 0.5f, laneCentre);
}

void PianoRollComponent::nudgeFlagLaneZoom(bool zoomIn)
{
    static constexpr std::array<float, 6> levels { 1.0f, 2.0f, 3.0f, 4.0f,
                                                   6.0f, 8.0f };
    auto level = 0;
    auto distance = std::numeric_limits<float>::max();
    for (int index = 0; index < static_cast<int>(levels.size()); ++index)
        if (const auto next = std::abs(laneZoom - levels[static_cast<std::size_t>(index)]);
            next < distance)
        {
            distance = next;
            level = index;
        }
    const auto wanted = levels[static_cast<std::size_t>(
        juce::jlimit(0, static_cast<int>(levels.size()) - 1,
                     level + (zoomIn ? 1 : -1)))];
    // Leaving the full view, the centre is whatever the range's middle is,
    // which is rarely where anything is drawn; take it to the curves instead.
    if (laneZoom <= 1.0f && wanted > 1.0f) centreFlagLaneOnCurves();
    laneZoom = wanted;
    repaint();
}

void PianoRollComponent::setHifisamplerFlagContext(bool enabled)
{
    if (hifisamplerFlagContext == enabled) return;
    cancelContinuousFlag();
    hifisamplerFlagContext = enabled;
    if (!diffSingerFlagContext) setFlagLaneFlag(enabled ? "HIFI:g" : "g");
    repaint();
}

void PianoRollComponent::setDiffSingerFlagContext(bool enabled, const juce::var& capabilities)
{
    if (!enabled) { setDiffSingerPitchRestoreMode(false); setDiffSingerPitchOffsetMode(false); }
    const auto changed = enabled != diffSingerFlagContext;
    const auto ready = !diffSingerFlagCapabilities.isArray() && capabilities.isArray();
    diffSingerFlagContext = enabled;
    diffSingerFlagCapabilities = capabilities;
    if (changed) dsPreferActual = enabled;
    if (changed || !flagLaneParameterSupported(laneFlag) || (enabled && ready && dsPreferActual))
    {
        const auto wantedActual = dsPreferActual;
        if (!enabled) setFlagLaneFlag(hifisamplerFlagContext ? "HIFI:g" : "g");
        else if (!setDiffSingerParameterLayer(wantedActual))
        {
            setFlagLaneFlag("DS:DYN");
            dsPreferActual = wantedActual;
        }
    }
    repaint();
}

bool PianoRollComponent::hasDiffSingerParameters() const
{
    if (!diffSingerFlagContext) return false;
    for (const auto& kind : diffSingerParameterKinds())
        if (flagLaneParameterSupported(kind.flag)) return true;
    return false;
}

bool PianoRollComponent::setDiffSingerParameterLayer(bool actual)
{
    if (!diffSingerFlagContext) return false;
    dsPreferActual = actual;
    if (!actual)
    {
        setFlagLaneFlag(laneFlag.startsWith("DS:ABS:") ? "DS:" + laneFlag.substring(7)
                        : (flagLaneParameterSupported(laneFlag) ? laneFlag : juce::String("DS:DYN")));
        return true;
    }
    const auto corresponding = laneFlag.startsWith("DS:ABS:") ? laneFlag : "DS:ABS:" + laneFlag.substring(3);
    for (const auto& candidate : {corresponding, dsLastActualFlag})
        for (const auto& kind : diffSingerParameterKinds())
            if (candidate == kind.flag && flagLaneParameterSupported(candidate))
            { setFlagLaneFlag(candidate); return true; }
    for (const auto& kind : diffSingerParameterKinds())
        if (flagLaneParameterSupported(kind.flag)) { setFlagLaneFlag(kind.flag); return true; }
    return false;
}

bool PianoRollComponent::flagLaneParameterSupported(const juce::String& flag) const
{
    if (!diffSingerFlagContext) {
        if (hifisamplerFlagContext) return flag.startsWith("HIFI:");
        return !flag.startsWith("DS:") && !flag.startsWith("HIFI:");
    }
    if (isDiffSingerParameter(flag))
        return flagLaneParameterSupported("DS:"+flag.fromLastOccurrenceOf(":", false, false));
    if (const auto* items = diffSingerFlagCapabilities.getArray())
        for (const auto& item : *items)
            if (item["key"].toString() == flag) return (bool) item["supported"];
    return flag == "DS:DYN";
}

void PianoRollComponent::cancelContinuousFlag()
{
    if (dragMode == DragMode::continuousFlag) dragMode = DragMode::none;
    continuousFlagTargets.clear();
    continuousFlagPreview.clear();
    continuousFlagInk.clear();
    repaint();
}

void PianoRollComponent::setFlagEditMode(FlagEditMode mode)
{
    if (flagMode == mode) return;
    clearPointSelection();
    cancelContinuousFlag();
    if (dragMode == DragMode::flagPoint)
    {
        dragMode = DragMode::none;
        draggedNote.clear();
        flagStroke.clear();
    }
    flagMode = mode;
    repaint();
}

void PianoRollComponent::beginContinuousFlag(juce::Point<float> position)
{
    cancelContinuousFlag();
    rebuildLayout();
    if (!flagLaneParameterSupported(laneFlag)) return;
    auto trackId = focusedTrack;
    for (const auto& hit : noteHits)
    {
        if (!flagCurveActiveFor(hit.id)) continue;
        const auto* track = trackForNote(hit.id);
        if (track == nullptr) continue;
        if (trackId.isEmpty()) trackId = track->id;
        if (track->id != trackId) continue;
        const auto* note = findNote(hit.id);
        const auto [from, to] = flagLaneSpanFor(*note);
        continuousFlagTargets.emplace(hit.id, ContinuousFlagTarget{
            hit.startSeconds + hit.clipStartSeconds, std::max(-5.0, from),
            std::min(60.0, to), laneFlag.startsWith("DS:ABS:") ? flagCurvePointsFor(*note,laneFlag) : flagLaneCurveFor(*note)});
    }
    if (continuousFlagTargets.empty()) return;
    continuousFlagRevision = model.revisionNumber();
    dragMode = DragMode::continuousFlag;
    const auto plot = flagLanePlotBounds();
    continuousFlagLast = {(juce::jlimit(plot.getX(), plot.getRight(), position.x)-58.0)/pixelsPerSecond,
                          flagValueFromLaneY(position.y), PitchCurveShape::linear};
    continuousFlagInk.push_back(continuousFlagLast);
    drawContinuousFlag(position);
}

void PianoRollComponent::drawContinuousFlag(juce::Point<float> position)
{
    if (model.revisionNumber() != continuousFlagRevision || !flagLaneParameterSupported(laneFlag))
    {
        cancelContinuousFlag();
        return;
    }
    const auto plot = flagLanePlotBounds();
    FlagCurvePoint point{(juce::jlimit(plot.getX(), plot.getRight(), position.x)-58.0)/pixelsPerSecond,
                         flagValueFromLaneY(position.y), PitchCurveShape::linear};
    const auto lo = std::min(point.timeSeconds, continuousFlagLast.timeSeconds);
    const auto hi = std::max(point.timeSeconds, continuousFlagLast.timeSeconds);
    std::erase_if(continuousFlagInk, [=](const auto& p) {
        return p.timeSeconds >= lo-1.0e-5 && p.timeSeconds <= hi+1.0e-5;
    });
    if (hi-lo > 1.0e-5) continuousFlagInk.push_back(continuousFlagLast);
    continuousFlagInk.push_back(point);
    std::sort(continuousFlagInk.begin(), continuousFlagInk.end(),
        [](const auto& a, const auto& b) { return a.timeSeconds < b.timeSeconds; });
    continuousFlagLast = point;
    continuousFlagPreview.clear();
    for (const auto& [id, target] : continuousFlagTargets)
    {
        const auto from = std::max(target.from, continuousFlagInk.front().timeSeconds-target.start);
        const auto to = std::min(target.to, continuousFlagInk.back().timeSeconds-target.start);
        if (to < from) continue;
        auto local = continuousFlagInk;
        for (auto& p : local) p.timeSeconds -= target.start;
        continuousFlagPreview[id] = spliceFlagDrawing(target.original, from, to, local);
    }
    repaint();
}

void PianoRollComponent::setFlagLaneFlag(const juce::String& flag)
{
    if (flag != laneFlag) cancelContinuousFlag();
    if (!flagLaneParameterSupported(flag)) return;
    bool known = flag.startsWith("DS:CLR:") || flag.startsWith("DS:ABS:");
    const auto& kinds = diffSingerFlagContext ? diffSingerFlagCurveKinds() : hifisamplerFlagContext ? hifisamplerFlagCurveKinds() : flagCurveKinds();
    for (const auto& kind : kinds) known = known || flag == kind.flag;
    if (!known) return;
    if (laneFlag != flag) clearPointSelection();
    laneFlag = flag;
    if (diffSingerFlagContext)
    {
        dsPreferActual = flag.startsWith("DS:ABS:");
        if (dsPreferActual) dsLastActualFlag = flag;
    }
    centreFlagLaneOnCurves();
    if (dragMode == DragMode::flagPoint) dragMode = DragMode::none;
    repaint();
}

const TrackData* PianoRollComponent::trackForNote(const juce::String& noteId) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId) return &track;
    return nullptr;
}

std::pair<double, double> PianoRollComponent::flagLaneSpanFor(
    const NoteData& note) const
{
    const auto whole = std::max(0.02, note.durationSeconds);

    // What the note sounds for, in its own time.  It begins a lead-in ahead of
    // the note -- that lead-in is the consonant -- and ends where the next
    // note's overlap has taken over.  A flag reaches all of it, so all of it
    // can be edited: measuring from the note's own start instead left the
    // consonant drawn but untouchable, since it lies at negative times.
    auto sounds = std::pair<double, double>(0.0, whole);
    for (const auto& hit : noteHits)
        if (hit.id == note.id)
        {
            const auto absoluteStart = hit.startSeconds + hit.clipStartSeconds;
            if (const auto span = utauSoundSpans.find(note.id.toStdString());
                span != utauSoundSpans.end())
                sounds = { span->second.first - absoluteStart,
                           span->second.second - absoluteStart };
            break;
        }
    if (!flagCurveKindFor(laneFlag).onsetOnly) return sounds;
    auto leadIn = -sounds.first;
    const auto* track = trackForNote(note.id);
    if (track == nullptr)
    {
        if (leadIn <= 1.0e-4 && note.utauPreutteranceOverrideEnabled)
            leadIn = note.utauPreutteranceSeconds;
        return { -std::max(0.02, std::min(leadIn, whole)), 0.0 };
    }
    juce::ignoreUnused(whole);
    const auto velocity = note.utauConsonantVelocity != inheritedUtauConsonantVelocity
        ? note.utauConsonantVelocity : track->utauConsonantVelocity;
    const auto timing = voicebankReadFor(*track)
        ? backend::UtauRenderer::sampleTiming(
              track->voicebankDirectory, note.label, note.midiNote, velocity,
              utauModeUsesRegions(track->utauMode),
              track->utauMode == UtauMode::mou, note.ownOto())
        : std::nullopt;
    if (leadIn <= 1.0e-4 && timing) leadIn = timing->preutteranceSeconds;
    if (leadIn <= 1.0e-4 && note.utauPreutteranceOverrideEnabled)
        leadIn = note.utauPreutteranceSeconds;
    leadIn = std::max(0.02, std::min(leadIn, whole));

    // Where the engine's con_out lands, which is the only thing b and bh can
    // reach past.  With a region plan -- four-region on and the sample carrying
    // oto4 rows, exactly what makes the renderer send the region lengths -- it
    // is the onset region's own boundary, and that ends at the note's start.
    // Without one the engine falls back to the oto consonant scaled by
    // velocity, measured from the start of the rendered stretch, which runs
    // past the note's start whenever that is longer than the lead-in: la- has
    // a 129 ms consonant against a 65 ms lead-in, and bh reaches 64 ms into it.
    auto onsetEnd = 0.0;
    if (timing && !(utauModeUsesRegions(track->utauMode) && timing->hasRegions))
        onsetEnd = juce::jlimit(0.0, whole,
            timing->consonantSeconds * std::pow(2.0, 1.0 - velocity / 100.0)
                - leadIn);
    return { -leadIn, onsetEnd };
}

std::vector<juce::String> PianoRollComponent::chosenNoteIds(
    const juce::String& noteId) const
{
    std::vector<juce::String> ids;
    ids.reserve(selectedNotes.size());
    for (const auto& id : selectedNotes)
        ids.push_back(juce::String::fromUTF8(id.c_str()));
    if (ids.empty() && noteId.isNotEmpty()) ids.push_back(noteId);
    return ids;
}

bool PianoRollComponent::ownOtoRestoreAvailable(
    const std::vector<juce::String>& noteIds) const
{
    for (const auto& id : noteIds)
        if (const auto* note = findNote(id); note != nullptr && note->utauOto.enabled)
            return true;
    return false;
}

bool PianoRollComponent::flagCurveActiveFor(const juce::String& noteId) const
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return false;
    if (laneFlag.startsWith("DS:ABS:")) {
        const auto* track=trackForNote(noteId);
        return track && trackIsDiffSinger(*track) && flagLaneParameterSupported(laneFlag)
            && !flagCurvePointsFor(*note,laneFlag).empty();
    }
    if (!note->utauFlagCurveEnabled) return false;
    const auto* track = trackForNote(noteId);
    return track != nullptr && trackTakesFlagCurves(*track)
        && trackIsDiffSinger(*track) == laneFlag.startsWith("DS:")
        && flagLaneParameterSupported(laneFlag);
}

bool PianoRollComponent::flagResetAvailable(
    const std::vector<juce::String>& noteIds) const
{
    for (const auto& id : noteIds)
        if (const auto* note = findNote(id); note != nullptr
            && flagCurveActiveFor(id) && (!note->utauFlagCurves.empty() || backend::hasCvvcFlagCurve(note->cvvcEnvelopes)))
            return true;
    return false;
}

std::vector<juce::String> PianoRollComponent::laneNoteIds() const
{
    std::vector<juce::String> ids;
    ids.reserve(noteHits.size());
    for (const auto& hit : noteHits)
    {
        bool utau = false;
        if (const auto* note = findNote(hit.id, &utau); note != nullptr && utau)
            ids.push_back(hit.id);
    }
    return ids;
}

bool PianoRollComponent::flagResetAvailableForLane() const
{
    for (const auto& id : laneNoteIds())
        if (flagResetAvailableForNote(id)) return true;
    return false;
}

void PianoRollComponent::showFlagLaneSwitchContextMenu(juce::Point<int> screenPosition)
{
    juce::PopupMenu menu;
    menu.addSectionHeader(juce::String::fromUTF8(laneFlag.startsWith("DS:ABS:") ? "DS 实参 · " : "flag 包络 · ") + laneFlag);
    menu.addItem(1, juce::String::fromUTF8(laneFlag.startsWith("DS:ABS:") ? "恢复该实参的原始预测（整轨）" : "重置该 flag（整轨）"),
                 flagResetAvailableForLane(), false);
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        { screenPosition.x, screenPosition.y, 1, 1 }),
        [safe](int chosen)
        {
            if (safe == nullptr || chosen != 1) return;
            // This flag across every note on show, and nothing else: the other
            // sixteen curves on each note stay where they are.
            safe->model.resetNotesUtauFlagCurve(safe->laneNoteIds(), safe->laneFlag);
        });
}

bool PianoRollComponent::flagResetAvailableForNote(const juce::String& noteId) const
{
    const auto* note = findNote(noteId);
    return note != nullptr && flagCurveActiveFor(noteId)
        && (!flagCurvePointsFor(*note, laneFlag).empty() || backend::hasCvvcFlagCurve(note->cvvcEnvelopes,laneFlag));
}

std::pair<double, double> PianoRollComponent::flagLaneDrawnSpan(
    const NoteData& note, double absoluteStart) const
{
    // The same stretch that can be edited, so what is drawn is what answers a
    // click and there is nowhere on the line that cannot be worked on.
    const auto [from, to] = flagLaneSpanFor(note);
    return { absoluteStart + from, absoluteStart + to };
}

juce::Point<float> PianoRollComponent::diagnosticFlagHandleCentre(
    const juce::String& noteId, int index) const
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return {};
    for (const auto& hit : noteHits)
        if (hit.id == noteId)
        {
            const auto start = hit.startSeconds + hit.clipStartSeconds;
            const auto points = flagLaneCurveFor(*note);
            if (index < 0 || index >= static_cast<int>(points.size())) return {};
            const auto plot = flagLanePlotBounds();
            const auto [drawFrom, drawTo] = flagLaneDrawnSpan(*note, start);
            const auto left = juce::jlimit(plot.getX(), plot.getRight(),
                                           timeToX(drawFrom));
            const auto right = juce::jlimit(plot.getX(), plot.getRight(),
                                            timeToX(drawTo));
            const auto& point = points[static_cast<std::size_t>(index)];
            return { juce::jlimit(left, right, timeToX(start + point.timeSeconds)),
                     flagLaneY(point.value) };
        }
    return {};
}

std::vector<FlagCurvePoint> PianoRollComponent::flagLaneCurveFor(
    const NoteData& note) const
{
    if(pointGroupGesture)for(const auto& d:pointGroupGesture->curves)
        if(d.identity.kind==PointSelection::flag&&d.identity.noteId==note.id&&d.identity.key==laneFlag)return d.edit.flagPoints;
    if (laneFlag.startsWith("DS:ABS:")) return diffSingerParameterHandles(note,laneFlag);
    if (!note.utauFlagCurveEnabled) return {};
    auto points = flagCurvePointsFor(note, laneFlag);
    if (!points.empty()) return points;
    // Flat at the flag's own resting value, spanning the note: something to
    // take hold of.  Zero for the ones that run either side of it, the low end
    // for those that do not, since that is where "no effect" sits for them.
    const auto& kind = flagCurveKindFor(laneFlag);
    const auto resting = kind.defaultValue;
    // One handle, at the end.  A curve of a single point reads as that value
    // for the whole note -- both the engine's fc_eval and flagCurveValueAt
    // hold flat either side of a lone point -- so dragging it is setting the
    // flag's level for the note, which is what is wanted before there is any
    // shape to draw.  A second point is a click away, and makes it a curve.
    const auto [from, to] = flagLaneSpanFor(note);
    juce::ignoreUnused(from);
    return { { to, resting } };
}

juce::Rectangle<float> PianoRollComponent::flagLaneZoomButtonBounds(bool zoomIn) const
{
    // Left of the flag switch, laid out like the loudness lane's own pair so
    // the two lanes are worked the same way.
    const auto lane = flagLaneBounds();
    return { lane.getRight() - 96.0f - static_cast<float>(flagLaneExtraControlWidth()) - (zoomIn ? 25.0f : 49.0f),
             lane.getY() + 4.0f, 21.0f, 21.0f };
}

juce::Rectangle<float> PianoRollComponent::flagLaneSwitchBounds() const
{
    // Top-right of the lane, beside the heading that names the flag.
    auto header = flagLaneBounds().removeFromTop(28.0f);
    return header.removeFromRight(96.0f).reduced(6.0f, 4.0f);
}

int PianoRollComponent::flagLaneExtraControlWidth() const
{
    return diffSingerFlagContext ? 104 + (laneFlag.startsWith("DS:ABS:") ? 0 : 66) : 0;
}

juce::Rectangle<float> PianoRollComponent::flagLaneLayerBounds(bool actual) const
{
    if (!diffSingerFlagContext) return {};
    const auto lane = flagLaneBounds();
    return {lane.getRight() - (actual ? 200.0f : 148.0f), lane.getY() + 4, 48, 20};
}

juce::Rectangle<float> PianoRollComponent::flagLaneOffsetEnableBounds() const
{
    if (!diffSingerFlagContext || laneFlag.startsWith("DS:ABS:")) return {};
    const auto lane = flagLaneBounds();
    return {lane.getRight() - 266, lane.getY() + 4, 60, 20};
}

std::vector<juce::String> PianoRollComponent::diffSingerOffsetTargets() const
{
    std::vector<juce::String> targets;
    for (const auto& id : selectedNoteIds())
        if (const auto* track = trackForNote(id); track && trackIsDiffSinger(*track)) targets.push_back(id);
    return targets;
}

bool PianoRollComponent::diffSingerOffsetsEnabled() const
{
    const auto ids = diffSingerOffsetTargets();
    return !ids.empty() && std::all_of(ids.begin(), ids.end(), [this](const auto& id) {
        const auto* note = findNote(id); return note && note->utauFlagCurveEnabled;
    });
}

juce::StringArray PianoRollComponent::unsupportedDiffSingerFlagCurves() const
{
    juce::StringArray keys;
    if (!diffSingerFlagContext || !diffSingerFlagCapabilities.isArray()) return keys;
    for (const auto& id : laneNoteIds())
        if (const auto* note = findNote(id))
            for (const auto& curve : note->utauFlagCurves)
                if (curve.flag.startsWith("DS:") && !curve.flag.startsWith("DS:REF:") && !curve.flag.startsWith("DS:PTS:") && !curve.points.empty()
                    && !flagLaneParameterSupported(curve.flag)) keys.addIfNotAlreadyThere(curve.flag);
    return keys;
}

void PianoRollComponent::showFlagLaneSwitchMenu()
{
    juce::PopupMenu menu;
    std::vector<juce::String> keys;
    auto add = [&](const juce::String& key, const juce::String& label, bool enabled)
    {
        keys.push_back(key);
        menu.addItem(static_cast<int>(keys.size()), label, enabled, laneFlag == key);
    };
    if (diffSingerFlagContext && diffSingerFlagCapabilities.isArray())
    {
        if (laneFlag.startsWith("DS:ABS:"))
        {
            menu.addSectionHeader(I18n::uiText("FLAG 实参（直接值）"));
            for (const auto& kind : diffSingerParameterKinds())
                add(kind.flag, juce::String::fromUTF8(kind.label), flagLaneParameterSupported(kind.flag));
        }
        else
        {
            menu.addSectionHeader(I18n::uiText("FLAG 偏移 / 其他控制"));
            for (const auto& item : *diffSingerFlagCapabilities.getArray())
            {
                const auto key = item["key"].toString();
                auto label = key.substring(3) + juce::String::fromUTF8(" · ") + item["label"].toString();
                const auto enabled = (bool) item["supported"];
                if (!enabled) label += " (" + item["reason"].toString() + ")";
                add(key, label, enabled);
            }
        }
    }
    else
    {
        const auto& kinds = diffSingerFlagContext ? diffSingerFlagCurveKinds() : hifisamplerFlagContext ? hifisamplerFlagCurveKinds() : flagCurveKinds();
        for (const auto& kind : kinds)
            add(kind.flag, (juce::String(kind.flag).startsWith("HIFI:") ? juce::String(kind.flag).substring(5) : juce::String(kind.flag)) + juce::String::fromUTF8(" · ")
                + juce::String::fromUTF8(kind.label), flagLaneParameterSupported(kind.flag));
        if (diffSingerFlagContext)
            menu.addSectionHeader(I18n::uiText("正在读取音源能力；音源库设置可重新读取"));
    }
    const auto unsupported = unsupportedDiffSingerFlagCurves();
    if (!unsupported.isEmpty())
    {
        menu.addSeparator();
        menu.addSectionHeader(I18n::uiText("当前显示轨道：清除不支持的已有曲线（可撤销）"));
        for (int index = 0; index < unsupported.size(); ++index)
            menu.addItem(10000 + index, I18n::uiText("清除 ") + unsupported[index].substring(3));
    }
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    const auto where = flagLaneSwitchBounds().toNearestInt() + getScreenPosition();
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(where),
        [safe, keys, unsupported](int chosen)
        {
            if (safe == nullptr || chosen <= 0) return;
            if (chosen >= 10000 && chosen - 10000 < unsupported.size())
            {
                safe->model.resetNotesUtauFlagCurve(safe->laneNoteIds(), unsupported[chosen - 10000]);
                return;
            }
            if (chosen > static_cast<int>(keys.size())) return;
            safe->setFlagLaneFlag(keys[static_cast<std::size_t>(chosen - 1)]);
        });
}

PianoRollComponent::FlagPointMenuState PianoRollComponent::flagPointMenuState(
    int pointCount, float value)
{
    return { pointCount > 1, std::abs(value - flagCurveDefault) > 1.0e-4f };
}

void PianoRollComponent::showFlagPointMenu(const juce::String& noteId, int index,
                                           juce::Point<int> screenPosition)
{
    const auto* note = findNote(noteId);
    if (note == nullptr || index < 0) return;
    const auto shown = flagLaneCurveFor(*note);
    if (index >= static_cast<int>(shown.size())) return;
    const auto value = shown[static_cast<std::size_t>(index)].value;
    const auto state = flagPointMenuState(
        static_cast<int>(flagLaneCurveFor(*note).size()),
        value - flagCurveKindFor(laneFlag).defaultValue);
    juce::PopupMenu menu;
    menu.addItem(1, I18n::uiText("删除标点"), state.canDelete, false);
    menu.addItem(2, laneFlag.startsWith("DS:ABS:") ? I18n::uiText("恢复此点的原始预测")
        : I18n::uiText("重置为默认值（")+juce::String(flagCurveKindFor(laneFlag).defaultValue,0)+juce::String::fromUTF8("）"),
        laneFlag.startsWith("DS:ABS:") || state.canReset, false);
    menu.addItem(3, I18n::uiText("输入 flag 值…"), true, false);
    menu.addSeparator();
    // Only the flag on show, and only this note: the lane's other sixteen
    // curves and every other note are left as they are.
    menu.addItem(4, I18n::uiText("重置该音的 ") + laneFlag,
                 flagResetAvailableForNote(noteId), false);
    // The shape of the segment arriving at this handle, so the first one has
    // no segment to shape.  Same names and same maths as the pitch line.
    menu.addSeparator();
    if (index == 0)
        menu.addItem(20, I18n::uiText("线条形状（首个标点没有前段）"),
                     false, false);
    else
    {
        const auto shape = flagLaneCurveFor(*note)[static_cast<std::size_t>(index)].shape;
        juce::PopupMenu shapes;
        shapes.addItem(21, I18n::uiText("直线"), true,
                       shape == PitchCurveShape::linear);
        shapes.addItem(22, I18n::uiText("平滑"), true,
                       shape == PitchCurveShape::smooth);
        shapes.addItem(23, I18n::uiText("缓入"), true,
                       shape == PitchCurveShape::easeIn);
        shapes.addItem(24, I18n::uiText("缓出"), true,
                       shape == PitchCurveShape::easeOut);
        shapes.addSeparator();
        shapes.addItem(25, I18n::uiText("自定义贝塞尔…"), true,
                       shape == PitchCurveShape::customBezier);
        menu.addSubMenu(I18n::uiText("线条形状"), shapes);
    }
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    // At the handle, like every other right-click menu here.  Aimed at the
    // component instead, it opened in the corner of the roll, which for a lane
    // at the bottom of a tall canvas is nowhere near the cursor.
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        { screenPosition.x, screenPosition.y, 1, 1 }),
        [safe, noteId, index](int chosen)
        {
            if (safe == nullptr || chosen <= 0) return;
            if (chosen == 3) { safe->showFlagPointValueDialog(noteId, index); return; }
            if (chosen == 4)
            {
                // Dropping the curve leaves the note showing the starting
                // handle again, which is what the initial state is.
                safe->model.setNoteUtauFlagCurve(noteId, safe->laneFlag, {});
                return;
            }
            if (chosen == 25) { safe->showFlagPointBezierDialog(noteId, index); return; }
            if (chosen == 1)
            {
                safe->deleteEnvelopePoint({noteId,index,safe->model.revisionNumber(),PointSelection::flag,0,safe->laneFlag});
                return;
            }
            const auto* current = safe->findNote(noteId);
            if (current == nullptr) return;
            auto points = safe->flagLaneCurveFor(*current);
            if (index >= static_cast<int>(points.size())) return;
            auto& point = points[static_cast<std::size_t>(index)];
            if (chosen == 2) point.value = safe->laneFlag.startsWith("DS:ABS:")
                ? flagCurveValueAt(flagCurvePointsFor(*current,"DS:REF:"+safe->laneFlag.substring(7)),point.timeSeconds)
                : flagCurveKindFor(safe->laneFlag).defaultValue;
            else if (chosen >= 21 && chosen <= 24)
                point.shape = chosen == 21 ? PitchCurveShape::linear
                    : chosen == 22 ? PitchCurveShape::smooth
                    : chosen == 23 ? PitchCurveShape::easeIn
                                   : PitchCurveShape::easeOut;
            else return;
            safe->model.setNoteUtauFlagCurve(noteId, safe->laneFlag, std::move(points));
        });
}

juce::String PianoRollComponent::flagPointValueText(const juce::String& noteId,
                                                    int index) const
{
    const auto* note = findNote(noteId);
    if (note == nullptr || index < 0) return {};
    // What the lane is showing, not what the note has stored.  A note with no
    // curve yet still shows the handle every flag starts from, and the menu
    // offers to type its value -- read from the stored curve there was nothing
    // to read and the box never opened at all.
    const auto points = flagLaneCurveFor(*note);
    if (index >= static_cast<int>(points.size())) return {};
    return juce::String(points[static_cast<std::size_t>(index)].value, 3);
}

juce::String PianoRollComponent::flagPointValueBlurb(const juce::String& noteId,
                                                     int index) const
{
    const auto* note = findNote(noteId);
    if (note == nullptr || index < 0) return {};
    const auto shown = flagLaneCurveFor(*note);
    if (index >= static_cast<int>(shown.size())) return {};
    const auto& point = shown[static_cast<std::size_t>(index)];
    // The flag the lane is on, not g.  This was written for g and said so
    // whatever was being edited: Mt reaches +/-100 and Mq starts at 0, so the
    // range printed was wrong for sixteen of the seventeen flags and the value
    // it showed looked wrong against it.
    const auto& kind = flagCurveKindFor(laneFlag);
    auto blurb = juce::String(kind.flag) + juce::String::fromUTF8("：")
        + juce::String::fromUTF8(kind.label)
        + I18n::uiText("，范围 ") + juce::String(kind.minimum, 0)
        + juce::String::fromUTF8(" … ") + juce::String(kind.maximum, 0)
        + juce::String::fromUTF8("。\n");
    if (laneFlag == "g")
        blurb += I18n::uiText("正值下移共振峰（听感偏成熟/厚），负值上移（偏年轻/薄）。\n");
    // Which handle this is.  Curves overlap wherever notes do, so a number on
    // its own does not say which of them was picked up.
    return blurb + I18n::uiText("当前：第 ") + juce::String(index + 1)
        + I18n::uiText(" 个标点（共 ") + juce::String(shown.size())
        + I18n::uiText(" 个），位于音符内 ")
        + juce::String(point.timeSeconds, 3)
        + I18n::uiText(" 秒，现值 ") + juce::String(point.value, 3)
        + juce::String::fromUTF8("。");
}

void PianoRollComponent::showFlagPointValueDialog(const juce::String& noteId, int index)
{
    const auto* note = findNote(noteId);
    if (note == nullptr || index < 0) return;
    const auto shown = flagLaneCurveFor(*note);
    if (index >= static_cast<int>(shown.size())) return;
    const auto& point = shown[static_cast<std::size_t>(index)];
    auto* dialog = new juce::AlertWindow(
        I18n::uiText("标点数值"),
        flagPointValueBlurb(noteId, index),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("value", juce::String(point.value, 3),
                          laneFlag + I18n::uiText(" 值："));
    if (auto* editor = dialog->getTextEditor("value"))
    {
        editor->setSelectAllWhenFocused(true);
        editor->setInputRestrictions(0, "-0123456789.");
    }
    dialog->addButton(I18n::uiText("应用"), 1);
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, noteId, index](int result)
            {
                if (safe != nullptr && result == 1)
                    if (const auto* current = safe->findNote(noteId);
                        current != nullptr
                        && index < static_cast<int>(
                               safe->flagLaneCurveFor(*current).size()))
                    {
                        // The shown curve again, so typing a value into the
                        // handle a fresh note starts with writes that handle
                        // rather than falling through to nothing.
                        auto points = safe->flagLaneCurveFor(*current);
                        // The model clamps to the engine's own range, so a
                        // typed 900 lands on 50 rather than being refused.
                        points[static_cast<std::size_t>(index)].value =
                            dialog->getTextEditorContents("value").getFloatValue();
                        safe->model.setNoteUtauFlagCurve(noteId, safe->laneFlag, std::move(points));
                    }
                delete dialog;
            }), false);
}

void PianoRollComponent::showFlagPointBezierDialog(const juce::String& noteId,
                                                   int index)
{
    const auto* note = findNote(noteId);
    if (note == nullptr || index <= 0
        || index >= static_cast<int>(flagLaneCurveFor(*note).size()))
        return;
    const auto point = flagLaneCurveFor(*note)[static_cast<std::size_t>(index)];
    auto* dialog = new juce::AlertWindow(strings.text("pitchCurve.bezierTitle"),
        I18n::uiText("归一化控制点。X 是段内的时间位置（0–1），"
                                "Y 是这一段走完的比例；Y 超出 0–1 会先冲过头再回来。"),
        juce::MessageBoxIconType::NoIcon);
    const std::array<std::pair<const char*, float>, 4> fields {{
        { "x1", point.bezierX1 }, { "y1", point.bezierY1 },
        { "x2", point.bezierX2 }, { "y2", point.bezierY2 } }};
    const std::array<const char*, 4> labels { "pitchCurve.bezierX1",
        "pitchCurve.bezierY1", "pitchCurve.bezierX2", "pitchCurve.bezierY2" };
    for (std::size_t at = 0; at < fields.size(); ++at)
    {
        dialog->addTextEditor(fields[at].first, juce::String(fields[at].second, 3),
                              strings.text(labels[at]));
        if (auto* editor = dialog->getTextEditor(fields[at].first))
        {
            editor->setSelectAllWhenFocused(true);
            editor->setInputRestrictions(0, "-0123456789.");
        }
    }
    dialog->addButton(I18n::uiText("应用"), 1);
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, noteId, index](int result)
            {
                if (safe != nullptr && result == 1)
                    if (const auto* current = safe->findNote(noteId);
                        current != nullptr
                        && index < static_cast<int>(
                               safe->flagLaneCurveFor(*current).size()))
                    {
                        auto points = safe->flagLaneCurveFor(*current);
                        auto& edited = points[static_cast<std::size_t>(index)];
                        edited.shape = PitchCurveShape::customBezier;
                        edited.bezierX1 = dialog->getTextEditorContents("x1").getFloatValue();
                        edited.bezierY1 = dialog->getTextEditorContents("y1").getFloatValue();
                        edited.bezierX2 = dialog->getTextEditorContents("x2").getFloatValue();
                        edited.bezierY2 = dialog->getTextEditorContents("y2").getFloatValue();
                        safe->model.setNoteUtauFlagCurve(noteId, safe->laneFlag, std::move(points));
                    }
                delete dialog;
            }), false);
}

juce::Path PianoRollComponent::diffSingerPitchReferencePath(const NoteData& note,
                                                               double absoluteStart) const
{
    juce::Path path;
    const auto& points = note.diffSingerPitchReference;
    if (points.empty()) return path;
    auto from = std::min(0.0, points.front().timeSeconds);
    auto to = std::max(note.durationSeconds, points.back().timeSeconds);
    if (const auto span = utauSoundSpans.find(note.id.toStdString()); span != utauSoundSpans.end())
    {
        from = std::min(from, span->second.first - absoluteStart);
        to = std::min(to, span->second.second - absoluteStart);
    }
    auto visible = getLocalBounds();
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>()) visible = viewport->getViewArea();
    from = std::max(from, (visible.getX()-58.0)/pixelsPerSecond - absoluteStart);
    to = std::min(to, (visible.getRight()-58.0)/pixelsPerSecond - absoluteStart);
    if (to <= from) return path;
    const auto steps = std::max(2, static_cast<int>(std::ceil((to-from)*pixelsPerSecond/2.0)));
    auto open = false;
    for (int step = 0; step <= steps; ++step)
    {
        const auto time = from + (to-from)*step/steps;
        const auto midi = evaluatePitchCurve(points, time);
        if (!std::isfinite(midi)) continue;
        const juce::Point<float> point(timeToX(absoluteStart+time), midiToY(midi)+rowHeight*0.5f);
        if (!open) { path.startNewSubPath(point); open = true; } else path.lineTo(point);
    }
    return path;
}

juce::Path PianoRollComponent::diagnosticDiffSingerPitchReference(const juce::String& noteId) const
{
    const auto* track = trackForNote(noteId);
    const auto* note = findNote(noteId);
    if (track == nullptr || note == nullptr || !trackIsDiffSinger(*track)
        || !showOriginalPitchLine || sourceEditMode || tool == Tool::amplitude) return {};
    for (const auto& hit : noteHits)
        if (hit.id == noteId) return diffSingerPitchReferencePath(*note, hit.clipStartSeconds+hit.startSeconds);
    return {};
}

void PianoRollComponent::drawFlagLane(juce::Graphics& g) const
{
    const auto lane = flagLaneBounds();
    const auto plot = flagLanePlotBounds();
    g.setColour(Palette::panel.withAlpha(0.97f));
    g.fillRoundedRectangle(lane, 5.0f);
    g.setColour(Palette::border.brighter(0.18f));
    g.drawRoundedRectangle(lane, 5.0f, 1.2f);

    const auto pairedFlag = diffSingerFlagContext ? pairedDiffSingerFlag(laneFlag) : juce::String{};
    const auto hasPair = pairedFlag.isNotEmpty() && flagLaneParameterSupported(pairedFlag);
    const auto actualLayer = laneFlag.startsWith("DS:ABS:");
    const auto cvvcContext = std::any_of(noteHits.begin(),noteHits.end(),[this](const auto& hit){return usesCvvcEnvelope(hit.id);});
    const auto referenceColour = actualLayer ? rollRole(juce::Colour(0xffbd9ae8), Palette::heading) : rollRole(juce::Colour(0xff82b9df), Palette::textMuted);
    const auto& kind = flagCurveKindFor(laneFlag);
    const auto [windowLow, windowHigh] = flagLaneWindow();
    // The heading reports the slice on show, not the flag's whole range, so
    // the numbers beside the grid are never a surprise.
    const auto zoomText = laneZoom > 1.0f
        ? juce::String::fromUTF8("  ×") + juce::String(laneZoom, laneZoom < 2.0f ? 1 : 0)
        : juce::String();
    g.setFont(11.0f);
    g.setColour(Palette::text);
    g.drawText(juce::String::fromUTF8(laneFlag.startsWith("DS:ABS:") ? "DS 实参 · " : (diffSingerFlagContext ? "DF FLAG 偏移/控制 · " : hifisamplerFlagContext ? "HiFisampler · " : "flag 包络 · "))
                   + (laneFlag.startsWith("DS:ABS:") ? laneFlag.substring(7) : (diffSingerFlagContext ? laneFlag.substring(3) : hifisamplerFlagContext ? laneFlag.substring(5) : laneFlag))
                   + juce::String::fromUTF8("（") + juce::String::fromUTF8(kind.label)
                   + I18n::uiText("）  纵向 ")
                   + juce::String(windowLow, 0) + " … " + juce::String(windowHigh, 0)
                   + zoomText
                   + (hasPair ? juce::String::fromUTF8(actualLayer
                       ? "  虚线：紫=偏移(右2) 蓝=原预测 绿=最终"
                       : "  虚线：蓝=实参(右2) 绿=最终(右2)") : juce::String())
                   + juce::String::fromUTF8(flagMode == FlagEditMode::continuous
                       ? (cvvcContext ? "   · 发音片段独立绘制" : "   · 跨音符连续绘制")
                       : (cvvcContext ? "   · 发音片段控制点" : "   · 逐音符控制点"))
                   + (kind.onsetOnly ? I18n::uiText("   仅声母区")
                                     : juce::String()),
        lane.toNearestInt().removeFromTop(28).reduced(8, 0).withTrimmedRight(158 + flagLaneExtraControlWidth()),
        juce::Justification::centredLeft, false);

    for (const auto zoomIn : { false, true })
    {
        const auto button = flagLaneZoomButtonBounds(zoomIn);
        g.setColour(Palette::panelRaised);
        g.fillRoundedRectangle(button, 3.0f);
        g.setColour(Palette::border.brighter(0.25f));
        g.drawRoundedRectangle(button, 3.0f, 1.0f);
        // Greyed at the ends, so the pair says how far it can still go.
        const auto atEnd = zoomIn ? laneZoom >= 8.0f : laneZoom <= 1.0f;
        g.setColour(atEnd ? Palette::textMuted : Palette::text);
        g.setFont(15.0f);
        g.drawText(amplitudeZoomLabel(zoomIn), button.toNearestInt(),
                   juce::Justification::centred, false);
    }

    // The switch, at the top right of the box.
    const auto switchBounds = flagLaneSwitchBounds();
    g.setColour(Palette::panelRaised);
    g.fillRoundedRectangle(switchBounds, 3.0f);
    g.setColour(Palette::border.brighter(0.25f));
    g.drawRoundedRectangle(switchBounds, 3.0f, 1.0f);
    g.setColour(Palette::text);
    g.setFont(11.0f);
    g.drawText((laneFlag.startsWith("DS:ABS:") ? laneFlag.substring(7) : (diffSingerFlagContext ? laneFlag.substring(3) : hifisamplerFlagContext ? laneFlag.substring(5) : laneFlag)) + juce::String::fromUTF8("  ▾"), switchBounds.toNearestInt(),
               juce::Justification::centred, false);

    if (diffSingerFlagContext)
    {
        const auto drawButton = [&](juce::Rectangle<float> bounds, const juce::String& label, bool active, bool enabled) {
            g.setColour(active ? rollRole(juce::Colour(0xff437d81), Palette::selection) : Palette::panel.brighter(0.08f));
            g.fillRoundedRectangle(bounds, 3);
            g.setColour(active ? rollRole(juce::Colour(0xff8fd4d2), Palette::heading) : Palette::border.brighter(0.25f));
            g.drawRoundedRectangle(bounds, 3, 1);
            g.setColour(rollText(enabled ? Palette::text : Palette::textMuted.withAlpha(0.5f),
                active ? rollRole(juce::Colour(0xff437d81), Palette::selection) : Palette::panel.brighter(0.08f), !enabled));
            g.drawText(label, bounds.toNearestInt(), juce::Justification::centred, false);
        };
        const auto actual = laneFlag.startsWith("DS:ABS:");
        drawButton(flagLaneLayerBounds(true), I18n::uiText("实参"), actual, hasDiffSingerParameters());
        drawButton(flagLaneLayerBounds(false), I18n::uiText("偏移"), !actual, true);
        if (!actual) drawButton(flagLaneOffsetEnableBounds(), juce::String::fromUTF8(diffSingerOffsetsEnabled() ? "已启用" : "启用"),
                               diffSingerOffsetsEnabled(), !diffSingerOffsetTargets().empty());
    }

    g.setFont(10.0f);
    const auto windowSpan = windowHigh - windowLow;
    for (auto step = 0; step <= 4; ++step)
    {
        const auto value = windowLow + windowSpan * static_cast<float>(step) / 4.0f;
        const auto y = flagLaneY(value);
        // The line at zero is the one worth finding, when the window holds it.
        g.setColour(std::abs(value) < windowSpan * 0.005f
                        ? rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(0.36f)
                        : Palette::grid.withAlpha(0.70f));
        g.drawHorizontalLine(static_cast<int>(y), plot.getX(), plot.getRight());
        g.setColour(Palette::textMuted);
        g.drawText(juce::String(value, 0),
                   juce::Rectangle<int>(static_cast<int>(plot.getRight()) + 2,
                                        static_cast<int>(y) - 7, 34, 14),
                   juce::Justification::centredLeft, false);
    }
    // Zero is worth a line of its own when the window has been taken off it.
    if (windowLow < 0.0f && windowHigh > 0.0f)
    {
        g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(0.36f));
        g.drawHorizontalLine(static_cast<int>(flagLaneY(0.0f)),
                             plot.getX(), plot.getRight());
    }

    auto drawnAny = false;
    // The other layer has its own units and scale; neither scale changes the data.
    if (hasPair)
    {
        const auto& pairedKind = flagCurveKindFor(pairedFlag);
        const auto pairedY = [&](float value) {
            const auto unit = juce::jlimit(0.0f, 1.0f, (value-pairedKind.minimum)/(pairedKind.maximum-pairedKind.minimum));
            return plot.getBottom()-unit*plot.getHeight();
        };
        g.setColour(referenceColour);g.setFont(10.0f);
        for (int step=0;step<=4;++step)
        {
            const auto value=pairedKind.minimum+(pairedKind.maximum-pairedKind.minimum)*static_cast<float>(step)/4;
            g.drawText(juce::String(value,0),juce::Rectangle<int>(static_cast<int>(plot.getRight())+48,
                static_cast<int>(pairedY(value))-7,48,14),juce::Justification::centredLeft,false);
        }
        g.setFont(9.0f);
        g.drawText(juce::String::fromUTF8(actualLayer ? "偏移" : "实参"),
            juce::Rectangle<int>(static_cast<int>(plot.getRight())+48,static_cast<int>(plot.getBottom())+1,48,11),
            juce::Justification::centredLeft,false);
        const auto liveCurve = [&](const NoteData& note,const juce::String& key) {
            if (key==laneFlag)
            {
                if (const auto preview=continuousFlagPreview.find(note.id);preview!=continuousFlagPreview.end()) return preview->second;
                if (dragMode==DragMode::flagPoint && note.id==draggedNote)
                    return actualLayer ? spliceDiffSingerParameterHandles(note,key,flagStroke) : flagStroke;
            }
            return flagCurvePointsFor(note,key);
        };
        for (const auto& hit : noteHits)
        {
            const auto* note=findNote(hit.id);const auto* track=trackForNote(hit.id);
            if (!note || !track || !trackIsDiffSinger(*track)) continue;
            const auto code=actualLayer ? laneFlag.substring(7) : laneFlag.substring(3);
            const auto base=liveCurve(*note,"DS:ABS:"+code);
            const auto offset=liveCurve(*note,"DS:"+code);
            const auto original=flagCurvePointsFor(*note,"DS:REF:"+code);
            const auto& baseKind=flagCurveKindFor("DS:ABS:"+code);
            const auto& offsetKind=flagCurveKindFor("DS:"+code);
            const auto start=hit.startSeconds+hit.clipStartSeconds;
            const auto [from,to]=flagLaneDrawnSpan(*note,start);
            const auto left=juce::jlimit(plot.getX(),plot.getRight(),timeToX(from));
            const auto right=juce::jlimit(plot.getX(),plot.getRight(),timeToX(to));
            if (right<=left+.5f) continue;
            const auto baseY=[&](float value){return actualLayer ? flagLaneY(value) : pairedY(value);};
            const auto makeLine=[&](auto valueAt,auto yAt) {
                juce::Path line;const auto steps=std::max(1,static_cast<int>(std::ceil((right-left)/1.5f)));
                for(int i=0;i<=steps;++i) {
                    const auto x=left+(right-left)*static_cast<float>(i)/static_cast<float>(steps);
                    const auto t=(x-58.0)/pixelsPerSecond-start;
                    const auto y=yAt(valueAt(t));
                    if(i==0)line.startNewSubPath(x,y);else line.lineTo(x,y);
                }
                return line;
            };
            const auto dashed=[&](const juce::Path& line,juce::Colour colour,bool fine=false) {
                const float pattern[]{fine ? 2.0f : 6.0f,fine ? 3.0f : 4.0f};juce::Path stroke;
                juce::PathStrokeType(1.35f).createDashedStroke(stroke,line,pattern,2);
                g.setColour(colour);g.fillPath(stroke);
            };
            const auto offsetAt=[&](double t){return offset.empty() ? offsetKind.defaultValue : flagCurveValueAt(offset,t);};
            if (actualLayer)
            {
                dashed(makeLine(offsetAt,pairedY),referenceColour.withAlpha(note->utauFlagCurveEnabled ? .8f : .35f));
                if (!original.empty()) dashed(makeLine([&](double t){return flagCurveValueAt(original,t);},baseY),rollRole(juce::Colour(0xff82b9df), Palette::textMuted).withAlpha(.7f));
            }
            else if (!base.empty())
                dashed(makeLine([&](double t){return flagCurveValueAt(base,t);},pairedY),referenceColour.withAlpha(.85f));
            if (!base.empty())
            {
                drawnAny=true;
                // Preview reads the active gesture, before its single undoable commit.
                dashed(makeLine([&](double t) {
                    auto value=flagCurveValueAt(base,t);
                    if (note->utauFlagCurveEnabled)
                    {
                        const auto delta=offsetAt(t);
                        value+=code=="TENC" ? delta/20.0f : (delta-offsetKind.defaultValue)*.12f;
                    }
                    return juce::jlimit(baseKind.minimum,baseKind.maximum,value);
                },baseY),rollRole(juce::Colour(0xff80d7ab), Palette::noteEdge).withAlpha(.85f),true);
            }
        }
    }

    for (const auto& hit : noteHits)
    {
        bool utau = false;
        const auto* note = findNote(hit.id, &utau);
        if (note == nullptr || !utau || !flagCurveActiveFor(note->id) || usesCvvcEnvelope(note->id)) continue;
        const auto absoluteStart = hit.startSeconds + hit.clipStartSeconds;
        const auto dragging = dragMode == DragMode::flagPoint && note->id == draggedNote;
        const auto preview = continuousFlagPreview.find(note->id);
        const auto handles = dragging ? flagStroke : flagLaneCurveFor(*note);
        const auto absolute = laneFlag.startsWith("DS:ABS:");
        const auto points = preview != continuousFlagPreview.end() ? preview->second
            : (absolute ? (dragging ? spliceDiffSingerParameterHandles(*note,laneFlag,flagStroke)
                                   : flagCurvePointsFor(*note,laneFlag)) : handles);
        if (points.empty()) continue;
        drawnAny = true;
        // A curve reaches exactly as far as the note it belongs to: the engine
        // reads it across that note's rendered stretch and nowhere else.  Held
        // flat before the first handle and after the last, within that stretch,
        // which is what the engine does with the ends.
        const auto [soundStart, soundEnd] = flagLaneDrawnSpan(*note, absoluteStart);
        if (kind.onsetOnly)
        {
            // The onset alone.  What lies past it is shaded, so it is plain
            // that the flat stretch there is out of this flag's hands rather
            // than something that was not drawn yet.
            auto shadeEnd = absoluteStart + note->durationSeconds;
            if (const auto span = utauSoundSpans.find(note->id.toStdString());
                span != utauSoundSpans.end())
                shadeEnd = span->second.second;
            const auto shadeFrom = juce::jlimit(plot.getX(), plot.getRight(),
                                                timeToX(soundEnd));
            const auto shadeTo = juce::jlimit(plot.getX(), plot.getRight(),
                                              timeToX(shadeEnd));
            if (shadeTo > shadeFrom + 0.5f)
            {
                g.setColour(juce::Colours::black.withAlpha(0.20f));
                g.fillRect(juce::Rectangle<float>::leftTopRightBottom(
                    shadeFrom, plot.getY(), shadeTo, plot.getBottom()));
            }
        }
        const auto left = juce::jlimit(plot.getX(), plot.getRight(), timeToX(soundStart));
        const auto right = juce::jlimit(plot.getX(), plot.getRight(), timeToX(soundEnd));
        if (right <= left + 0.5f) continue;
        juce::Path path;
        const auto xAt = [&](double timeSeconds)
        {
            return juce::jlimit(left, right, timeToX(absoluteStart + timeSeconds));
        };
        // Evaluate at the visible sounding boundaries. Clamping off-range
        // guard points onto the edge would draw false vertical spikes there.
        const auto localFrom = std::max(soundStart, (plot.getX()-58.0)/pixelsPerSecond) - absoluteStart;
        const auto localTo = std::min(soundEnd, (plot.getRight()-58.0)/pixelsPerSecond) - absoluteStart;
        path.startNewSubPath(left, flagLaneY(flagCurveValueAt(points, localFrom)));
        if (points.front().timeSeconds > localFrom && points.front().timeSeconds < localTo)
            path.lineTo(xAt(points.front().timeSeconds), flagLaneY(points.front().value));
        for (std::size_t index = 1; index < points.size(); ++index)
        {
            const auto from = std::max(localFrom, points[index-1].timeSeconds);
            const auto to = std::min(localTo, points[index].timeSeconds);
            if (to <= from) continue;
            if (points[index].shape != PitchCurveShape::linear)
            {
                const auto steps = juce::jlimit(4, 48, static_cast<int>((xAt(to)-xAt(from))/3.0f));
                for (auto step = 1; step < steps; ++step)
                {
                    const auto at = from + (to-from)*step/static_cast<double>(steps);
                    path.lineTo(xAt(at), flagLaneY(flagCurveValueAt(points, at)));
                }
            }
            path.lineTo(xAt(to), flagLaneY(flagCurveValueAt(points, to)));
        }
        path.lineTo(right, flagLaneY(flagCurveValueAt(points, localTo)));
        // Where two notes overlap their curves are two lines of one colour.
        // With a note selected its curve is drawn brighter and heavier and the
        // rest step back, so which one is being edited can be seen before the
        // click rather than after it.  With nothing selected they are all as
        // they were.
        const auto anySelected = flagMode == FlagEditMode::perNote && !selectedNotes.empty();
        const auto isSelected = selectedNotes.contains(note->id.toStdString());
        g.setColour(rollRole(juce::Colour(0xffe4b363), Palette::heading).withAlpha(
            !anySelected ? 0.92f : (isSelected ? 1.0f : 0.42f)));
        g.strokePath(path, juce::PathStrokeType(
            anySelected && isSelected ? 2.3f : 1.6f));
        if (flagMode == FlagEditMode::perNote)
        for (std::size_t index = 0; index < handles.size(); ++index)
        {
            const auto centre = juce::Point<float>(xAt(handles[index].timeSeconds),
                                                   flagLaneY(handles[index].value));
            if (!plot.expanded(6.0f).contains(centre)) continue;
            const auto held = (dragging && static_cast<int>(index) == draggedFlagPoint)
                || pointIsSelected(PointSelection::flag,note->id,static_cast<int>(index),0,laneFlag);
            g.setColour(held ? rollRole(juce::Colours::white, Palette::heading) : rollRole(juce::Colour(0xffe4b363), Palette::heading));
            g.fillEllipse(centre.x - 3.5f, centre.y - 3.5f, 7.0f, 7.0f);
        }
    }
    drawnAny |= drawCvvcEnvelopeLane(g, false) > 0;
    if (!drawnAny)
    {
        g.setColour(Palette::textMuted);
        g.setFont(12.0f);
        g.drawText(juce::String::fromUTF8(laneFlag.startsWith("DS:ABS:") ? "右键 FLAG → 局部重算实参；先生成当前音符的预测曲线"
            : (diffSingerFlagContext ? "选择音符，再点击右上方「启用」编辑偏移；「实参」可返回预测曲线"
                                    : "先选中音符并开启「线性flag」，这里才有曲线可画（普通 UTAU / 界 / 谋）")),
                   plot.toNearestInt(), juce::Justification::centred, false);
    }
}

juce::Rectangle<float> PianoRollComponent::amplitudeLaneBounds() const
{
    auto visible = getLocalBounds().toFloat();
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
        visible = viewport->getViewArea().toFloat();
    const auto height = juce::jlimit(180.0f, 320.0f, visible.getHeight() * 0.46f);
    return juce::Rectangle<float>(visible.getX() + 60.0f,
        visible.getBottom() - height - 4.0f,
        std::max(120.0f, visible.getWidth() - 64.0f), height);
}

juce::Rectangle<float> PianoRollComponent::amplitudeLanePlotBounds() const
{
    auto lane = amplitudeLaneBounds();
    lane.removeFromTop(30.0f);
    lane.removeFromLeft(6.0f);
    lane.removeFromRight(56.0f);
    lane.removeFromBottom(12.0f);
    return lane;
}

float PianoRollComponent::amplitudeLaneY(float gainDb) const
{
    const auto plot = amplitudeLanePlotBounds();
    const auto normalized = juce::jlimit(0.0f, 1.0f,
        amplitudePercentFromDb(gainDb) / amplitudeLaneMaxPercent);
    return plot.getBottom() - normalized * plot.getHeight();
}

float PianoRollComponent::amplitudeDbFromLaneY(float y) const
{
    const auto plot = amplitudeLanePlotBounds();
    const auto normalized = juce::jlimit(0.0f, 1.0f,
        (plot.getBottom() - y) / std::max(1.0f, plot.getHeight()));
    return amplitudeDbFromPercent(normalized * amplitudeLaneMaxPercent);
}

juce::Rectangle<float> PianoRollComponent::amplitudeZoomButtonBounds(bool zoomIn) const
{
    const auto lane = amplitudeLaneBounds();
    return { lane.getRight() - (zoomIn ? 25.0f : 49.0f),
             lane.getY() + 4.0f, 21.0f, 21.0f };
}

float PianoRollComponent::amplitudePercentFromDb(float gainDb)
{
    return gainDb <= -59.9f ? 0.0f : std::pow(10.0f, gainDb / 20.0f) * 100.0f;
}

float PianoRollComponent::amplitudeDbFromPercent(float percent)
{
    if (percent <= 0.05f) return -60.0f;
    return juce::jlimit(-60.0f, 12.0f,
        20.0f * std::log10(percent / 100.0f));
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::mapAmplitudeEnvelopeToNote(
    const std::vector<AmplitudeEnvelopePoint>& source,
    const juce::String& sourceNoteId, const juce::String& targetNoteId) const
{
    if (source.empty() || sourceNoteId == targetNoteId) return source;
    const auto* sourceMember = nativeEnvelopeMember(sourceNoteId);
    const auto* targetMember = nativeEnvelopeMember(targetNoteId);
    if (sourceMember && targetMember && sourceMember->points == targetMember->points)
    {
        auto mapped = source;
        for (auto& point : mapped) point.timeSeconds += sourceMember->noteStart - targetMember->noteStart;
        return mapped;
    }
    bool utau = false;
    const auto* targetNote = findNote(targetNoteId, &utau);
    const auto targetHit = std::find_if(noteHits.begin(), noteHits.end(),
        [&](const auto& hit) { return hit.id == targetNoteId; });
    if (targetNote == nullptr || (!utau && !nativeNoteTimingEnabled(targetNoteId)) || targetHit == noteHits.end()) return {};

    const auto targetStart = targetHit->startSeconds + targetHit->clipStartSeconds;
    const auto targetDefault = amplitudeEnvelopeFor(*targetNote, targetStart);
    if (targetDefault.size() < 2) return {};
    const auto sourceFirst = source.front().timeSeconds;
    const auto sourceSpan = std::max(1.0e-6,
        source.back().timeSeconds - sourceFirst);
    const auto targetFirst = targetDefault.front().timeSeconds;
    const auto targetSpan = std::max(1.0e-6,
        targetDefault.back().timeSeconds - targetFirst);

    std::vector<AmplitudeEnvelopePoint> mapped;
    mapped.reserve(source.size());
    for (const auto& point : source)
    {
        const auto fraction = juce::jlimit(0.0, 1.0,
            (point.timeSeconds - sourceFirst) / sourceSpan);
        mapped.push_back({ targetFirst + fraction * targetSpan, point.gainDb, point.linearToNext, point.nativeSeamAnchor });
    }
    return mapped;
}

void PianoRollComponent::commitAmplitudeEnvelopeToSelection(
    const juce::String& sourceNoteId,
    const std::vector<AmplitudeEnvelopePoint>& source)
{
    std::vector<juce::String> ids;
    ids.reserve(selectedNotes.size() + 1);
    for (const auto& id : selectedNotes)
        ids.push_back(juce::String::fromUTF8(id.c_str()));
    if (std::find(ids.begin(), ids.end(), sourceNoteId) == ids.end())
        ids.push_back(sourceNoteId);

    // Selecting any fragment edits its complete linked envelope. Other
    // selected groups still receive the shape once, across their whole span.
    const auto initialIds = ids;
    for (const auto& id : initialIds)
        if (const auto* member = nativeEnvelopeMember(id))
            for (const auto& [trackId, envelopes] : nativeEnvelopeCache)
                for (const auto& [otherId, other] : envelopes)
                    if (other.points == member->points && std::find(ids.begin(), ids.end(), otherId) == ids.end())
                        ids.push_back(otherId);

    std::vector<std::pair<juce::String, std::vector<AmplitudeEnvelopePoint>>> edits;
    edits.reserve(ids.size());
    for (const auto& id : ids)
    {
        bool utau = false;
        if (findNote(id, &utau) == nullptr || (!utau && !nativeNoteTimingEnabled(id))) continue;
        auto mapped = mapAmplitudeEnvelopeToNote(source, sourceNoteId, id);
        if (const auto* member = nativeEnvelopeMember(id))
        {
            for (auto& point : mapped) point.timeSeconds += member->noteStart;
            mapped = sliceNativeSharedEnvelope(mapped, member->noteStart, member->noteEnd);
        }
        // What was dragged is the shape with the base already in it; the note
        // stores the shape without it, or the base would be multiplied in twice.
        if (const auto* target = findNote(id); target != nullptr)
            mapped = unscaledAmplitudeEnvelope(mapped, target->amplitudeEnvelopeBasePercent);
        if (mapped.size() >= 2) edits.emplace_back(id, std::move(mapped));
    }
    model.setNotesAmplitudeEnvelopes(std::move(edits));
}

bool PianoRollComponent::pitchAnchorHasIncoming(const NoteData& note, int anchorIndex)
{
    const auto& anchors = pitchAnchorsFor(note);
    if (anchorIndex < 0 || anchorIndex >= static_cast<int>(anchors.size())) return false;
    const auto* track = trackOf(note.id);
    if (track && trackShowsAllNativeRegions(*track))
    {
        const auto& lines = sharedLinesFor(*track);
        if (const auto* member = lines.memberFor(note.id); member && member->nativeSharedCurve)
        {
            double absolute = anchors[static_cast<size_t>(anchorIndex)].timeSeconds;
            for (const auto& clip : track->clips)
                for (const auto& candidate : clip.notes)
                    if (candidate.id == note.id) absolute += clip.startSeconds + candidate.startSeconds;
            // Sparse endpoints and freely moved points may have their predecessor
            // in another note. Use the resolved line's canonical handle order.
            if (!member->owns(absolute)) return false;
            for (const auto& entry : lines.byNote)
                if (entry.second.line == member->line)
                    for (const auto time : entry.second.nativeHandleTimes)
                        if (time < absolute - 1.e-7) return true;
            return false;
        }
    }
    return anchorIndex > 0;
}

bool PianoRollComponent::setPitchAnchorShape(const juce::String& noteId, int anchorIndex,
    PitchCurveShape shape, std::optional<std::array<float, 4>> bezier)
{
    const auto* note = findNote(noteId);
    if (!note || !pitchAnchorHasIncoming(*note, anchorIndex)) return false;
    auto anchors = pitchAnchorsFor(*note);
    prepareSampledPitchPointEdit(anchors, anchorIndex);
    auto& endpoint = anchors[static_cast<size_t>(anchorIndex)];
    endpoint.diffSingerRestoreSupport = false;
    endpoint.shape = shape;
    if (bezier)
    {
        endpoint.bezierX1 = juce::jlimit(0.f, 1.f, (*bezier)[0]);
        endpoint.bezierY1 = juce::jlimit(-2.f, 3.f, (*bezier)[1]);
        endpoint.bezierX2 = juce::jlimit(0.f, 1.f, (*bezier)[2]);
        endpoint.bezierY2 = juce::jlimit(-2.f, 3.f, (*bezier)[3]);
    }
    return model.setNotePitchCurve(noteId, std::move(anchors), true);
}

juce::PopupMenu PianoRollComponent::buildPitchCurveShapeMenu(
    const juce::String& noteId, int anchorIndex)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return {};
    auto& anchors = pitchAnchorsFor(*note);
    const auto validAnchor = anchorIndex >= 0
        && anchorIndex < static_cast<int>(anchors.size());
    const auto hasIncoming = pitchAnchorHasIncoming(*note, anchorIndex);

    juce::PopupMenu menu;
    menu.addSectionHeader(strings.text("pitchCurve.incomingSegment"));
    if (!hasIncoming)
        menu.addItem(1, strings.text("pitchCurve.noPrevious"), false, false);
    else
    {
        const auto current = anchors[static_cast<std::size_t>(anchorIndex)].shape;
        menu.addItem(1, strings.text("pitchCurve.natural"), true,
                     current == PitchCurveShape::natural);
        menu.addItem(2, strings.text("pitchCurve.linear"), true,
                     current == PitchCurveShape::linear);
        menu.addItem(3, strings.text("pitchCurve.smooth"), true,
                     current == PitchCurveShape::smooth);
        menu.addItem(4, strings.text("pitchCurve.easeIn"), true,
                     current == PitchCurveShape::easeIn);
        menu.addItem(5, strings.text("pitchCurve.easeOut"), true,
                     current == PitchCurveShape::easeOut);
        menu.addSeparator();
        menu.addItem(6, strings.text("pitchCurve.customBezier"), true,
                     current == PitchCurveShape::customBezier);
    }
    // Snapping is about the point itself rather than the segment leading into
    // it, so it stays available on the first point, where there is no segment.
    const auto alreadyOnNote = validAnchor
        && std::abs(anchors[static_cast<std::size_t>(anchorIndex)].targetMidi
                    - note->midiNote) < 1.0e-3f;
    menu.addSeparator();
    menu.addItem(7, strings.text("pitchCurve.snapToNote"),
                 validAnchor && !alreadyOnNote, false);
    // Typing the pitch in, for when the row a point has to land on is known
    // as a number rather than as somewhere to drag to.  Like snapping, this
    // is about the point and not the segment, so the first point has it too.
    menu.addItem(9, strings.text("pitchCurve.setFrequency"), validAnchor, false);
    // Only an interior point can be removed: the line has to be handed to a
    // neighbour on each side, and the two endpoints are what tie the curve to
    // the start and end of the note.
    const auto interior = validAnchor && anchorIndex > 0
        && anchorIndex + 1 < static_cast<int>(anchors.size());
    menu.addItem(8, strings.text("pitchCurve.deletePoint"), interior, false);
    const auto* track = trackForNote(noteId);
    if (track != nullptr && trackIsDiffSinger(*track))
    {
        menu.addSeparator();
        menu.addItem(10, I18n::uiText("还原至初始音高线（DS）"),
            !note->diffSingerPitchReference.empty(), dsPitchRestoreMode);
        if (!note->diffSingerPitchReference.empty())
            menu.addItem(11, juce::String::fromUTF8(note->diffSingerPitchReferenceFromSavedPitch
                ? "参考来源：旧工程载入时音高" : "参考来源：DS 原始预测"), false);
    }
    return menu;
}

namespace
{
// The ids in a menu, in the order they are offered.
std::vector<int> menuIds(const juce::PopupMenu& menu, bool enabledOnly,
                         bool searchSubMenus = false)
{
    std::vector<int> ids;
    juce::PopupMenu::MenuItemIterator walk(menu, searchSubMenus);
    while (walk.next())
    {
        const auto& item = walk.getItem();
        if (item.itemID == 0) continue;
        if (enabledOnly && !item.isEnabled) continue;
        ids.push_back(item.itemID);
    }
    return ids;
}

// A menu as it is laid out, one row at a time: an item as its id, a submenu as
// its name with the ids inside it in brackets, a separator as "-".  Section
// headers are left out, since nothing is chosen from them.
std::vector<juce::String> menuLayout(const juce::PopupMenu& menu)
{
    std::vector<juce::String> rows;
    juce::PopupMenu::MenuItemIterator walk(menu);
    while (walk.next())
    {
        const auto& item = walk.getItem();
        if (item.isSectionHeader) continue;
        if (item.isSeparator)
        {
            rows.push_back("-");
            continue;
        }
        if (item.subMenu != nullptr)
        {
            juce::StringArray inside;
            for (const auto id : menuIds(*item.subMenu, false))
                inside.add(juce::String(id));
            rows.push_back(item.text + "[" + inside.joinIntoString(",") + "]");
            continue;
        }
        if (item.itemID != 0) rows.push_back(juce::String(item.itemID));
    }
    return rows;
}
}

bool PianoRollComponent::anchorMenuChoiceHandled(int choice)
{
    return choice >= 1 && choice <= 9;
}

std::vector<int> PianoRollComponent::diagnosticAnchorMenuIds(
    const juce::String& noteId, int anchorIndex)
{
    return menuIds(buildPitchCurveShapeMenu(noteId, anchorIndex), false);
}

std::vector<int> PianoRollComponent::diagnosticEnabledAnchorMenuIds(
    const juce::String& noteId, int anchorIndex)
{
    return menuIds(buildPitchCurveShapeMenu(noteId, anchorIndex), true);
}

void PianoRollComponent::showPitchCurveShapeMenu(const juce::String& noteId,
                                                  int anchorIndex,
                                                  juce::Point<int> screenPosition)
{
    auto menu = buildPitchCurveShapeMenu(noteId, anchorIndex);
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        { screenPosition.x, screenPosition.y, 1, 1 }),
        [safe, noteId, anchorIndex](int result) mutable
        {
            if (safe != nullptr && result == 10)
            { safe->setDiffSingerPitchRestoreMode(!safe->diffSingerPitchRestoreMode()); return; }
            if (safe == nullptr || !anchorMenuChoiceHandled(result)) return;
            if (result == 6)
            {
                safe->showCustomBezierDialog(noteId, anchorIndex);
                return;
            }
            if (result == 9)
            {
                safe->showAnchorFrequencyDialog(noteId, anchorIndex);
                return;
            }
            const auto* currentNote = safe->findNote(noteId);
            if (currentNote == nullptr) return;
            if (result == 8)
            {
                safe->deletePitchAnchor(noteId, anchorIndex);
                return;
            }
            auto anchorsCopy = safe->pitchAnchorsFor(*currentNote);
            prepareSampledPitchPointEdit(anchorsCopy, anchorIndex);
            if (anchorIndex < 0
                || anchorIndex >= static_cast<int>(anchorsCopy.size())) return;
            anchorsCopy[static_cast<size_t>(anchorIndex)].diffSingerRestoreSupport = false;
            if (result == 7)
            {
                anchorsCopy[static_cast<std::size_t>(anchorIndex)].targetMidi =
                    currentNote->midiNote;
                safe->model.setNotePitchCurve(noteId, std::move(anchorsCopy), true);
                return;
            }
            safe->setPitchAnchorShape(noteId, anchorIndex,
                result == 2 ? PitchCurveShape::linear
                : result == 3 ? PitchCurveShape::smooth
                : result == 4 ? PitchCurveShape::easeIn
                : result == 5 ? PitchCurveShape::easeOut
                              : PitchCurveShape::natural);
        });
}

void PianoRollComponent::flattenPitchLine(const juce::String& noteId, bool snapNativeToSemitone)
{
    // The model does the whole thing in one step: setNotePitchCurve alone was
    // not enough, because it leaves the measured curve as it found it and the
    // renderer shifts away from that.
    model.flattenNotePitch(chosenNoteIds(noteId), snapNativeToSemitone);
    // The cached anchors were derived from the old line.
    pitchAnchorCache.clear();connectionPitchCache.clear();protocolPitchCache.clear();clearPitchPlanPreview();
    sharedLineCache.clear(); nativeEnvelopeCache.clear();
    draggedSharedKey = 0;
    repaint();
}

void PianoRollComponent::restoreOriginalPitch(const juce::String& noteId)
{
    if (!nativeNoteTimingEnabled(noteId)) return;
    if (!model.restoreNativeSourcePitch(chosenNoteIds(noteId))) return;
    pitchAnchorCache.clear();connectionPitchCache.clear();protocolPitchCache.clear();clearPitchPlanPreview();
    sharedLineCache.clear(); nativeEnvelopeCache.clear();
    draggedSharedKey = 0;
    repaint();
}

void PianoRollComponent::showTransposeDialog(const juce::String& noteId)
{
    std::vector<juce::String> targets;
    for (const auto& id : selectedNotes) targets.push_back(juce::String::fromUTF8(id.c_str()));
    if (targets.empty()) targets.push_back(noteId);
    if (targets.empty() || targets.front().isEmpty()) return;

    auto* dialog = new juce::AlertWindow(
        I18n::uiText("音高调整"),
        I18n::uiText("按半音平移选中的音符（12 平均律）。\n"
            "只改音高，不改时间；音高标点跟着一起移动。\n"
            "两个框都填时按两者之差平移。"),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("up", "0",
                          I18n::uiText("上移（半音）："));
    dialog->addTextEditor("down", "0",
                          I18n::uiText("下移（半音）："));
    for (const auto* key : { "up", "down" })
        if (auto* editor = dialog->getTextEditor(key))
        {
            editor->setInputRestrictions(4, "0123456789");
            editor->setSelectAllWhenFocused(true);
        }
    dialog->addButton(I18n::uiText("应用"), 1);
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, targets](int result)
            {
                if (safe != nullptr && result == 1)
                {
                    const auto up = dialog->getTextEditorContents("up").getIntValue();
                    const auto down = dialog->getTextEditorContents("down").getIntValue();
                    safe->model.transposeNotes(targets, static_cast<float>(up - down));
                }
                delete dialog;
            }), false);
}

void PianoRollComponent::clearPointSelection()
{
    if(graphPointDragging){graphPointDragging=false;graphDragValid=false;clearPitchPlanPreview();}
    if(pointGroupGesture){pointGroupGesture.reset();cancelPointDrag();}
    selectedPoints.clear();
    if (!selectedPoint) return;
    selectedPoint.reset();
    repaint();
}

void PianoRollComponent::focusLost(FocusChangeType)
{
    clearPointSelection();
}

bool PianoRollComponent::pointIsSelected(PointSelection::Kind kind, const juce::String& noteId,
                                       int index, int part, const juce::String& key) const
{
    if(kind==PointSelection::amplitude)if(const auto* member=nativeEnvelopeMember(noteId))
        for(const auto& p:selectedPoints)if(p.kind==kind&&p.index==index&&p.revision==model.revisionNumber())
            if(const auto* other=nativeEnvelopeMember(p.noteId);other&&other->points==member->points)return true;
    if(std::any_of(selectedPoints.begin(),selectedPoints.end(),[&](const auto& p){return p.kind==kind&&p.noteId==noteId&&p.index==index&&p.part==part&&p.key==key&&p.revision==model.revisionNumber();}))return true;
    return selectedPoint && selectedPoint->kind == kind && selectedPoint->noteId == noteId
        && selectedPoint->index == index && selectedPoint->part == part && selectedPoint->key == key
        && selectedPoint->revision == model.revisionNumber();
}

bool PianoRollComponent::deleteEnvelopePoint(const PointSelection& point)
{
    if (point.index < 0 || point.revision != model.revisionNumber()) return false;
    refreshFromModel();
    const auto* note = findNote(point.noteId);
    if (!note) return false;
    const auto revision = model.revisionNumber();
    if (point.kind == PointSelection::cvvc)
    {
        for (auto piece : cvvcLanePieces(point.key.isEmpty()))
            if (piece.noteId == point.noteId && piece.part == point.part && piece.key == point.key)
            {
                if (point.index <= 0 || point.index + 1 >= static_cast<int>(piece.points.size())) return false;
                piece.points.erase(piece.points.begin() + point.index);
                model.setCvvcEnvelope(piece.noteId,piece.part,piece.key,std::move(piece.points),revision);
                break;
            }
    }
    else if (point.kind == PointSelection::amplitude)
    {
        const auto hit = std::find_if(noteHits.begin(),noteHits.end(),[&](const auto& h){return h.id == point.noteId;});
        if (hit == noteHits.end()) return false;
        auto points = amplitudeEnvelopeFor(*note,hit->startSeconds + hit->clipStartSeconds);
        if (point.index <= 0 || point.index + 1 >= static_cast<int>(points.size())) return false;
        points.erase(points.begin() + point.index);
        commitAmplitudeEnvelopeToSelection(point.noteId,points);
    }
    else if (point.kind == PointSelection::flag && point.key == laneFlag)
    {
        auto points = flagLaneCurveFor(*note);
        if (points.size() <= 1 || point.index >= static_cast<int>(points.size())) return false;
        points.erase(points.begin() + point.index);
        model.setNoteUtauFlagCurve(point.noteId,point.key,std::move(points));
    }
    const auto changed = revision != model.revisionNumber();
    if (changed) { refreshFromModel(); repaint(); }
    return changed;
}

bool PianoRollComponent::deletePitchAnchor(const juce::String& noteId, int index)
{
    refreshFromModel();
    const auto* note = findNote(noteId);
    if (note == nullptr) return false;
    auto anchors = pitchAnchorsFor(*note);
    // Keep the same endpoint rules as the point's context menu.
    if (index <= 0 || index + 1 >= static_cast<int>(anchors.size())) return false;
    prepareSampledPitchPointEdit(anchors, index);
    anchors.erase(anchors.begin() + index);
    // The next point's shape now describes the span from the previous point.
    if (!model.setNotePitchCurve(noteId, std::move(anchors), true)) return false;
    refreshFromModel();
    repaint();
    return true;
}

void PianoRollComponent::deleteSelectedNotes()
{
    if(!selectedPoints.empty()){editSelectedPoints(false);return;}
    if (selectedPoint)
    {
        const auto point = *selectedPoint;
        selectedPoint->index = -1;
        // Delete is an alternative to finishing the point drag; mouseUp must
        // not write the old draft back after the point has been removed.
        if (dragMode == DragMode::pointPitch || dragMode == DragMode::amplitudePoint || dragMode == DragMode::flagPoint)
        {
            draggedNote.clear(); draggedPitchAnchor = -1;
            draggedAmplitudePoint = draggedFlagPoint = -1;
            amplitudeStroke.clear(); flagStroke.clear();
            pitchStroke.clear(); dragMode = DragMode::none;
        }
        cvvcLaneDrag.reset();
        if (point.revision == model.revisionNumber())
        {
            if (point.kind == PointSelection::pitch) deletePitchAnchor(point.noteId, point.index);
            else deleteEnvelopePoint(point);
        }
        repaint();
        return;
    }
    std::vector<juce::String> removed;
    removed.reserve(selectedNotes.size());
    for (const auto& id : selectedNotes) removed.push_back(juce::String::fromUTF8(id.c_str()));
    if (removed.empty()) removed.push_back(selectedNote);
    if (removed.empty() || removed.front().isEmpty()) return;
    selectedNote.clear();
    selectedNotes.clear();
    model.removeNotes(removed);
    if (onNoteSelected) onNoteSelected({});
}

namespace
{
// The note menu, in the order it is shown.  6 and 7 are the two faces of one
// item, so both carry the same answer.
enum class NoteMenuScope { both, utauOnly, plainOnly };
struct NoteMenuEntry { int id; NoteMenuScope scope; };
using Scope = NoteMenuScope;
const std::array<NoteMenuEntry, 28> noteMenuEntries {{
    { 34, Scope::plainOnly },   // 编辑素材标注：打开对应录音的共用原生编辑器
    {  1, Scope::both      },   // 分割音符
    {  2, Scope::both      },   // 合并音符
    { 20, Scope::utauOnly  },   // 添加拼字音符    -- a lead-in and nothing else
    { 30, Scope::plainOnly },   // 粘连音频：共享编辑时钟
    { 29, Scope::plainOnly },   // 断开音频：独立源区间和时间映射
    { 24, Scope::both      },   // Connect native notes without merging
    {  3, Scope::utauOnly  },   // 时序…             -- from the voicebank entry
    {  4, Scope::utauOnly  },   // 区域编辑器…       -- refuses off UTAU anyway
    // 高级 opens here, one level down, and holds the next four.
    { 19, Scope::utauOnly  },   // 修改 STP…         -- moves that entry in the wav
    { 21, Scope::utauOnly  },   // 单独OTO编辑…      -- this note's own entry
    { 22, Scope::utauOnly  },   // 恢复为音源OTO     -- and back to the voicebank's
    {  9, Scope::utauOnly  },   // flag 拆分…        -- needs the four regions
    {  5, Scope::both      },   // 颤音… -- shared target-pitch layer in both render paths
    {  6, Scope::both      },   // 改为真实颤音线
    {  7, Scope::both      },   // 退回为标准颤音模式
    {  8, Scope::both      },   // 修改颤音为音高标点… -- bakes into the anchor line
    { 14, Scope::utauOnly  },   // 重置线性flag    -- greyed off 界/谋
    { 12, Scope::utauOnly  },   // 辅音强制重置      -- a pinned preutterance is UTAU's
    { 23, Scope::both      },   // 包络基础值…       -- the envelope is a UTAU-mode thing
    { 13, Scope::both      },   // 时序删除          -- removes notes and closes the gap
    { 15, Scope::utauOnly  },   // 添加空格
    { 16, Scope::utauOnly  },   // 添加指定长度空格…
    { 17, Scope::utauOnly  },   // 批量输入歌词…     -- a lyric names a sample
    // Where the anchor line is the pitch line and there is no vibrato or
    // voicebank to fall back on, laying it flat is the way back.  In UTAU the
    // same thing is reached through the anchors and the vibrato items.
    { 18, Scope::plainOnly },   // 初始化音高线
    { 28, Scope::plainOnly },   // 还原为原始素材音高
    { 11, Scope::both      },   // 音高调整…
    { 10, Scope::both      },   // 删除音符
}};
}

std::vector<int> PianoRollComponent::noteMenuItemsFor(bool utauTrack)
{
    std::vector<int> items;
    items.reserve(noteMenuEntries.size());
    for (const auto& entry : noteMenuEntries)
        if (entry.scope == NoteMenuScope::both
            || (utauTrack ? entry.scope == NoteMenuScope::utauOnly
                          : entry.scope == NoteMenuScope::plainOnly))
            items.push_back(entry.id);
    return items;
}

PianoRollComponent::NoteMenu PianoRollComponent::buildNoteMenu(
    const juce::String& noteId) const
{
    NoteMenu built;
    const auto* note = findNote(noteId);
    if (note == nullptr) return built;

    std::optional<backend::UtauSampleTiming> noteTiming;
    for (const auto& track : snapshot.tracks)
    {
        if (!trackUsesVoicebankSynthesis(track)) continue;
        auto belongsToTrack = false;
        for (const auto& clip : track.clips)
            if (std::any_of(clip.notes.begin(), clip.notes.end(),
                [&](const auto& candidate) { return candidate.id == noteId; }))
                belongsToTrack = true;
        if (!belongsToTrack) continue;
        const auto velocity = note->utauConsonantVelocity
                != inheritedUtauConsonantVelocity
            ? note->utauConsonantVelocity : track.utauConsonantVelocity;
        noteTiming = backend::UtauRenderer::sampleTiming(
            track.voicebankDirectory, note->label, note->midiNote, velocity,
            utauModeUsesRegions(track.utauMode),
            track.utauMode == UtauMode::mou, note->ownOto());
        if (noteTiming)
        {
            if (note->utauPreutteranceOverrideEnabled)
                noteTiming->preutteranceSeconds = note->utauPreutteranceSeconds;
            if (note->utauOverlapOverrideEnabled)
                noteTiming->overlapSeconds = note->utauOverlapSeconds;
        }
        break;
    }

    // Editing uses the project's smallest stretch unit, which follows tempo
    // and time signature rather than being a fixed musical note value.
    const auto quantum = noteEditQuantumSeconds();
    const auto duration = note->durationSeconds;
    const auto canSplit = duration + 1.0e-8 >= quantum * 2.0;

    auto split = duration * 0.5;
    if (canSplit)
    {
        // Ties round toward a longer first half.  The second half uses the
        // exact remainder, so the original note end can never drift.
        split = std::ceil((duration * 0.5) / quantum - 1.0e-9) * quantum;
        split = juce::jlimit(quantum, duration - quantum, split);
    }

    std::vector<juce::String> mergeIds;
    mergeIds.reserve(selectedNotes.size());
    for (const auto& id : selectedNotes)
        mergeIds.push_back(juce::String::fromUTF8(id.c_str()));
    const auto allInOneClip = [&]
    {
        if (mergeIds.size() < 2) return false;
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
            {
                auto found = std::size_t{};
                for (const auto& candidate : clip.notes)
                    if (std::find(mergeIds.begin(), mergeIds.end(), candidate.id)
                        != mergeIds.end())
                        ++found;
                if (found == mergeIds.size()) return true;
            }
        return false;
    }();

    const auto utauTrack = isUtauNote(noteId);
    bool dsTrack = false;
    for (const auto& track : snapshot.tracks) if (trackIsDiffSinger(track))
        for (const auto& clip : track.clips) for (const auto& item : clip.notes)
            if (item.id == noteId) dsTrack = true;
    const auto offered = noteMenuItemsFor(utauTrack);
    const auto wanted = [&offered, dsTrack](int id)
    {
        if (dsTrack && (id == 3 || id == 9 || id == 12 || id == 19 || id == 20 || id == 21 || id == 22 || id == 24)) return false;
        return std::find(offered.begin(), offered.end(), id) != offered.end();
    };

    juce::PopupMenu menu;
    menu.addSectionHeader(strings.text("Note"));
    menu.addItem(33,strings.text("Replace material..."),onRequestMaterialSourceInput!=nullptr);
    if (nativeNoteTimingEnabled(noteId))
        menu.addItem(34, strings.text("Edit material annotations..."), onOpenRegionEditor != nullptr);
    bool physicalOwner = false;
    for (const auto& track : snapshot.tracks) for (const auto& clip : track.clips)
        if (physicalOverlapSource(clip) && std::any_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.id==noteId;})) physicalOwner=true;
    if (physicalOwner)
        menu.addItem(materialSelectorNoteMenuItem, strings.text("Material selector"), !sourceEditMode && overlapPairFor(noteId,std::numeric_limits<double>::quiet_NaN()).has_value(),
            nativeOverlapSelector && nativeOverlapSelector->targetNoteId==noteId);
    if(nativeNoteTimingEnabled(noteId))menu.addItem(31,strings.text("Loudness envelope"),onOpenNativeEnvelope!=nullptr);
    if (wanted(1))
        menu.addItem(1, I18n::uiText("分割音符"), canSplit, false);
    if (wanted(2))
        menu.addItem(2, I18n::uiText("合并音符"), allInOneClip, false);
    if (wanted(24))
        menu.addItem(24, I18n::uiText("强制连接（保留音符数据）"),
                     selectedNotes.size() >= 2, false);
    if (wanted(30) && nativeNoteTimingEnabled(noteId))
        menu.addItem(30, strings.text("native.linkAudio"),
            nativeAudioLinkAvailable(model.snapshot(),chosenNoteIds(noteId)),false);
    if (wanted(29) && nativeNoteTimingEnabled(noteId))
        menu.addItem(29, strings.text("native.disconnectAudio"),
            nativeAudioDisconnectAvailable(model.snapshot(),chosenNoteIds(noteId)),false);
    const auto canEditTiming = noteTiming.has_value() && selectedNotes.size() == 1;
    // A lead-in reaches back before the beat, so a note that begins at the
    // very start of its clip has nowhere to put one.
    const auto roomInFront = note->startSeconds > 1.0e-9;
    if (wanted(20))
        menu.addItem(20, I18n::uiText("添加拼字音符"), roomInFront, false);
    if (wanted(3))
        menu.addItem(3, I18n::uiText("时序…"), canEditTiming, false);
    if (wanted(4))
        menu.addItem(4, juce::String::fromUTF8(dsTrack ? "DS 音素时长…  Ctrl+G" : "区域编辑器…  Ctrl+G"),
                     onOpenRegionEditor != nullptr && selectedNotes.size() == 1, false);
    if (dsTrack)
    {
        menu.addItem(25, I18n::uiText("DS 发音转换 / 读音覆盖…"), onDiffSingerPronunciation != nullptr);
        menu.addItem(26, I18n::uiText("DS 局部重生成 pitch"), onDiffSingerPitch != nullptr);
        menu.addItem(27, I18n::uiText("DS 局部重新生成 FLAG（预测实参）"), onDiffSingerParameters != nullptr);
    }

    // 高级: the tools that reach into the entry a note is sung from, or split
    // what it carries, one level down -- hovering 高级 opens them.  Their ids
    // are unchanged, so choosing one reaches applyNoteMenuChoice as before.
    juce::PopupMenu advanced;
    // Offered whether or not the lyric resolves to an entry: a note may be
    // carrying an STP from a lyric it used to have, and refusing to open
    // would be the only thing standing between the user and clearing it.
    if (wanted(19))
        advanced.addItem(19, I18n::uiText("修改 STP…"), true, false);
    // Its own copy of the entry, for this note alone: the oto files are not
    // touched.  Every UTAU mode, since every one of them reads an entry.
    if (wanted(21))
        advanced.addItem(21, I18n::uiText("单独OTO编辑…"),
                         onOpenNoteOtoEditor != nullptr && selectedNotes.size() == 1, false);
    // And back to the voicebank's entry, which only a note with an oto of its
    // own has to go back to.
    if (wanted(22))
        advanced.addItem(22, I18n::uiText("恢复为音源OTO"),
                         ownOtoRestoreAvailable(chosenNoteIds(noteId)), false);
    // Splitting flags only means anything where the note has four regions.
    auto fourRegionTrack = false;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& candidate : clip.notes)
                if (candidate.id == noteId)
                    fourRegionTrack = track.pitchAlgorithm == PitchAlgorithm::utau
                        && utauModeUsesRegions(track.utauMode);
    const auto* flagNote = findNote(noteId);
    if (wanted(9))
        advanced.addItem(9, I18n::uiText("flag 拆分…"), fourRegionTrack,
                         flagNote != nullptr && flagNote->utauFlagSplit);
    // Only when something is in it: off UTAU none of these is offered, and an
    // empty 高级 would open onto nothing.
    if (advanced.getNumItems() > 0)
        menu.addSubMenu(I18n::uiText("高级"), advanced);

    const auto* vibratoNote = findNote(noteId);
    const auto hasVibrato = vibratoNote != nullptr && vibratoNote->vibratoEnabled;
    if (wanted(5))
        menu.addItem(5, I18n::uiText("颤音…"), vibratoNote != nullptr,
                     hasVibrato);
    if (wanted(6) && wanted(7))
    {
        if (hasVibrato && vibratoNote->vibratoRealLine)
            menu.addItem(7, I18n::uiText("退回为标准颤音模式"), true, false);
        else
            menu.addItem(6, I18n::uiText("改为真实颤音线"), hasVibrato, false);
    }
    if (wanted(8))
        menu.addItem(8, I18n::uiText("修改颤音为音高标点…"), hasVibrato, false);
    if (wanted(14))
        menu.addItem(14, I18n::uiText("重置线性flag"),
                     flagResetAvailable(chosenNoteIds(noteId)), false);
    if (wanted(12))
        menu.addItem(12, juce::String::fromUTF8(cvvcTimingEnabled(noteId)?"强制重置":"辅音强制重置"), true, false);
    // The envelope's height as a whole.  It belongs beside the envelope lane
    // rather than in 高级: raising or lowering a note is an ordinary edit, and
    // the lane is where the result is seen.
    if (wanted(23))
        menu.addItem(23, I18n::uiText("包络基础值…"), true, false);
    if (wanted(13))
        menu.addItem(13, I18n::uiText("时序删除"), true, false);
    // A silence opened in front of the note, on UTAU tracks in every mode.
    if (wanted(15))
        menu.addItem(15, I18n::uiText("添加空格"), true, false);
    if (wanted(16))
        menu.addItem(16, I18n::uiText("添加指定长度空格…"), true, false);
    if (wanted(17))
        menu.addItem(17, I18n::uiText("批量输入歌词…"),
                     !batchLyricTargets(noteId).empty(), false);
    if (wanted(18))
        menu.addItem(18, I18n::uiText("初始化音高线"), true, false);
    if (wanted(28) && nativeNoteTimingEnabled(noteId))
    {
        const auto ids = chosenNoteIds(noteId);
        const auto available = std::any_of(ids.begin(), ids.end(), [this](const auto& id) {
            const auto* item = findNote(id);
            return nativeNoteTimingEnabled(id) && item != nullptr && std::isfinite(item->sourceMidiCenter)
                && item->sourceMidiCenter >= 0.0f && !item->contour.empty();
        });
        menu.addItem(28, strings.text("native.restoreOriginalPitch"), available, false);
    }
    if (wanted(11))
        menu.addItem(11, I18n::uiText("音高调整…"), true, false);
    menu.addSeparator();
    if (wanted(10))
        menu.addItem(10, I18n::uiText("删除音符  Delete"), true, false);

    built.menu = std::move(menu);
    built.splitSeconds = split;
    built.mergeIds = std::move(mergeIds);
    return built;
}

std::vector<int> PianoRollComponent::diagnosticNoteMenuIds(
    const juce::String& noteId) const
{
    return menuIds(buildNoteMenu(noteId).menu, false, true);
}

std::vector<int> PianoRollComponent::diagnosticEnabledNoteMenuIds(
    const juce::String& noteId) const
{
    return menuIds(buildNoteMenu(noteId).menu, true, true);
}

juce::String PianoRollComponent::diagnosticNoteMenuLabel(const juce::String& noteId,int itemId) const
{
    auto built=buildNoteMenu(noteId);
    for(juce::PopupMenu::MenuItemIterator it(built.menu);it.next();)
        if(it.getItem().itemID==itemId)return it.getItem().text;
    return {};
}

std::vector<juce::String> PianoRollComponent::diagnosticNoteMenuLayout(
    const juce::String& noteId) const
{
    return menuLayout(buildNoteMenu(noteId).menu);
}

void PianoRollComponent::showNoteContextMenu(const juce::String& noteId,
                                              juce::Point<int> screenPosition)
{
    auto built = buildNoteMenu(noteId);
    const auto local = getLocalPoint(nullptr,screenPosition);
    const auto contextSeconds = std::max(0.0,static_cast<double>(local.x-58)/pixelsPerSecond);
    const auto split = built.splitSeconds;
    auto mergeIds = std::move(built.mergeIds);
    juce::Component::SafePointer<PianoRollComponent> keep(this);
    built.menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        { screenPosition.x, screenPosition.y, 1, 1 }),
        [keep, noteId, split, contextSeconds, mergeIds = std::move(mergeIds)](int result)
        {
            if (keep == nullptr) return;
            keep->applyNoteMenuChoice(noteId, result, split, mergeIds, contextSeconds);
        });
}

// What choosing an item does.  Out of the popup's callback so a check can ask
// it directly: inside one, every item in this menu was reachable only through
// a modal window, and an entry added to the table but not here would appear,
// accept a click and do nothing at all.
void PianoRollComponent::applyNoteMenuChoice(
    const juce::String& noteId, int result, double split,
    const std::vector<juce::String>& mergeIds, double contextSeconds)
{
    if (result == 0) { closeNativeOverlapSelector(); return; }
    if (result == 34)
    {
        if (nativeNoteTimingEnabled(noteId) && onOpenRegionEditor) onOpenRegionEditor(noteId);
        return;
    }
    if (result == materialSelectorNoteMenuItem) { openNativeOverlapSelector(noteId,contextSeconds); return; }
    if (result == 33 && onRequestMaterialSourceInput)
    {
        const auto* note=findNote(noteId);if(!note)return;
        const auto query=inlineAliasNoteId==noteId?inlineAliasEditor.getText():note->label;
        const auto callback=onRequestMaterialSourceInput;callback(noteId,query);return;
    }
    if (result == 25 && onDiffSingerPronunciation) { onDiffSingerPronunciation(noteId); return; }
    if (result == 31 && nativeNoteTimingEnabled(noteId) && onOpenNativeEnvelope) { onOpenNativeEnvelope(); return; }
    if (result == 26 && onDiffSingerPitch) { onDiffSingerPitch(noteId); return; }
    if (result == 27 && onDiffSingerParameters) { onDiffSingerParameters(noteId); return; }
    if (result == 3)
    {
        showNoteTimingDialog(noteId);
        return;
    }
    if (result == 19)
    {
        showStpDialog(noteId);
        return;
    }
    if (result == 23)
    {
        showEnvelopeBaseDialog(noteId);
        return;
    }
    if (result == 30)
    {
        const auto ids=chosenNoteIds(noteId);
        if(model.linkNativeAudio(ids)){refreshFromModel();setSelectedNoteIds({noteId});}
        return;
    }
    if (result == 29)
    {
        const auto ids=chosenNoteIds(noteId);
        if (model.disconnectNativeAudio(ids))
        {
            refreshFromModel();
            // End the batch operation with one active fragment. Keeping the
            // marquee selection would make the next drag move the whole set.
            setSelectedNoteIds({noteId});
        }
        return;
    }
    if (result == 24)
    {
        model.setNotesConnection(chosenNoteIds(noteId), true);
        return;
    }
    if (result == 20)
    {
        const auto prefix = model.insertPrefixNote(noteId,
            effectiveUtauOverlapFor(noteId));
        if (prefix.isEmpty()) return;
        // Selected, because the next thing anyone does with it is give
        // it a lyric of its own.
        selectedNote = prefix;
        selectedNotes.clear();
        selectedNotes.insert(prefix.toStdString());
        if (onNoteSelected) onNoteSelected(prefix);
        repaint();
        return;
    }
    if (result == 4)
    {
        if (onOpenRegionEditor) onOpenRegionEditor(noteId);
        return;
    }
    if (result == 21)
    {
        if (onOpenNoteOtoEditor) onOpenNoteOtoEditor(noteId);
        return;
    }
    if (result == 22)
    {
        model.clearNotesUtauOto(chosenNoteIds(noteId));
        return;
    }
    if (result == 5)
    {
        showVibratoDialog(noteId);
        return;
    }
    if (result == 6 || result == 7)
    {
        model.setNotesVibratoRealLine(chosenNoteIds(noteId),
                                            result == 6);
        return;
    }
    if (result == 8)
    {
        confirmBakeVibrato(noteId);
        return;
    }
    if (result == 14)
    {
        model.resetNotesUtauFlagCurves(chosenNoteIds(noteId));
        return;
    }
    if (result == 17)
    {
        showBatchLyricDialog(noteId);
        return;
    }
    if (result == 12)
    {
        resetConsonant(noteId);
        return;
    }
    if (result == 13)
    {
        deleteSelectedNotesRippling(noteId);
        return;
    }
    if (result == 9)
    {
        showRegionFlagDialog(noteId);
        return;
    }
    if (result == 28) { restoreOriginalPitch(noteId); return; }
    if (result == 18)
    {
        flattenPitchLine(noteId);
        return;
    }
    if (result == 11)
    {
        showTransposeDialog(noteId);
        return;
    }
    if (result == 15)
    {
        model.insertGapBeforeNote(noteId,
            quarterBarSecondsAt(
                std::max(0.0, absoluteStartOf(noteId))));
        return;
    }
    if (result == 16)
    {
        showGapDialog(noteId);
        return;
    }
    if (result == 10)
    {
        // Right-clicking already put this note in the selection, so
        // this removes exactly what Delete would have.
        deleteSelectedNotes();
        return;
    }
    if (result != 1 && result != 2) return;
    const auto resultingId = result == 1
        ? model.splitNote(noteId, split)
        : model.mergeNotes(mergeIds);
    if (resultingId.isEmpty()) return;
    selectedNote = resultingId;
    selectedNotes.clear();
    selectedNotes.insert(resultingId.toStdString());
    if (onNoteSelected) onNoteSelected(resultingId);
    repaint();
}

std::vector<PianoRollComponent::RegionFlagField>
PianoRollComponent::regionFlagFieldsFor(const juce::String& noteId) const
{
    std::vector<RegionFlagField> fields;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (note.id != noteId) continue;
                const std::array<juce::String, 4> stored {
                    note.utauRegionFlags1, note.utauRegionFlags2,
                    note.utauRegionFlags3, note.utauRegionFlags4 };
                // 界 is always the four parts of a Chinese syllable.  谋 says
                // per entry how many regions there are and lets any of them be
                // a consonant, so the boxes follow the entry and are named by
                // position -- the same names the wave and the tick boxes use.
                const auto mou = track.utauMode == UtauMode::mou;
                auto count = 4;
                if (mou)
                {
                    const auto velocity =
                        note.utauConsonantVelocity != inheritedUtauConsonantVelocity
                        ? note.utauConsonantVelocity : track.utauConsonantVelocity;
                    const auto timing = backend::UtauRenderer::sampleTiming(
                        track.voicebankDirectory, note.label, note.midiNote,
                        velocity, true, true, note.ownOto());
                    if (timing)
                        count = SampleSettings::mouRegionCount(timing->mouClasses);
                }
                static const std::array<const char*, 4> jieNames {
                    "\xe5\xa3\xb0\xe6\xaf\x8d", "\xe4\xbb\x8b\xe9\x9f\xb3",
                    "\xe9\x9f\xb5\xe8\x85\xb9", "\xe9\x9f\xb5\xe5\xb0\xbe" };
                for (int index = 0; index < count; ++index)
                {
                    const auto name = mou
                        ? juce::String::fromUTF8("\xe7\xac\xac")
                              + juce::String(index + 1)
                              + juce::String::fromUTF8("\xe5\x8c\xba")
                        : juce::String::fromUTF8(
                              jieNames[static_cast<std::size_t>(index)]);
                    fields.push_back({ "r" + juce::String(index + 1),
                                       name + juce::String::fromUTF8("\xef\xbc\x9a"),
                                       stored[static_cast<std::size_t>(index)] });
                }
                return fields;
            }
    return fields;
}

juce::StringArray PianoRollComponent::diagnosticRegionFlagFields(
    const juce::String& noteId) const
{
    juce::StringArray shown;
    for (const auto& field : regionFlagFieldsFor(noteId))
        shown.add(field.label + field.value);
    return shown;
}

double PianoRollComponent::absoluteStartOf(const juce::String& noteId) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId) return clip.startSeconds + note.startSeconds;
    return -1.0;
}

std::optional<PianoRollComponent::GapInfo> PianoRollComponent::gapAt(
    juce::Point<float> position) const
{
    if (sourceEditMode) return std::nullopt;
    const auto seconds = static_cast<double>(position.x - 58.0f) / pixelsPerSecond;
    if (seconds < 0.0) return std::nullopt;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose || track.pitchAlgorithm != PitchAlgorithm::utau
            || (focusedTrack.isNotEmpty() && track.id != focusedTrack))
            continue;
        for (const auto& clip : track.clips)
        {
            if (!clipIsVisible(track, clip)) continue;
            // In order, and by where each note really ends: notes can overlap,
            // so the silence begins after the furthest end so far rather than
            // after the previous note's own.
            std::vector<const NoteData*> ordered;
            for (const auto& note : clip.notes) ordered.push_back(&note);
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto* left, const auto* right)
                      {
                          return left->startSeconds < right->startSeconds;
                      });
            auto reach = -1.0;
            const NoteData* previous = nullptr;
            for (const auto* note : ordered)
            {
                const auto start = clip.startSeconds + note->startSeconds;
                if (previous != nullptr && start > reach + 1.0e-6
                    && seconds >= reach && seconds < start)
                    return GapInfo { clip.id, previous->id, note->id, reach, start,
                                     previous->midiNote };
                const auto end = start + note->durationSeconds;
                if (end > reach) { reach = end; previous = note; }
            }
        }
    }
    return std::nullopt;
}

juce::String PianoRollComponent::insertNoteIntoGap(const GapInfo& gap)
{
    const auto length = quarterBarSecondsAt(gap.fromSeconds);
    // The silence opens in front of the note that follows, which carries it
    // and everything after later; the new note goes in the space that leaves,
    // starting where the silence did.
    model.insertGapBeforeNote(gap.nextId, length);
    return model.addNote(gap.clipId, gap.fromSeconds, length, gap.midi);
}

void PianoRollComponent::showGapContextMenu(const GapInfo& gap,
                                            juce::Point<int> screenPosition)
{
    juce::PopupMenu menu;
    menu.addItem(1, I18n::uiText("删除空格"), true, false);
    menu.addItem(2, I18n::uiText("插入四分之一小节音符"), true, false);
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        { screenPosition.x, screenPosition.y, 1, 1 }),
        [safe, gap](int result)
        {
            if (safe == nullptr || result == 0) return;
            if (result == 1)
                safe->model.closeGapBeforeNote(gap.nextId,
                                               gap.toSeconds - gap.fromSeconds);
            else if (result == 2)
            {
                const auto id = safe->insertNoteIntoGap(gap);
                if (id.isNotEmpty())
                {
                    safe->selectedNote = id;
                    safe->selectedNotes.clear();
                    safe->selectedNotes.insert(id.toStdString());
                    if (safe->onNoteSelected) safe->onNoteSelected(id);
                }
            }
        });
}

bool PianoRollComponent::isUtauNote(const juce::String& noteId) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId)
                    return trackUsesVoicebankSynthesis(track);
    return false;
}

double PianoRollComponent::quarterBarSecondsAt(double seconds) const
{
    const auto beat = 60.0 / juce::jlimit(20.0, 400.0, snapshot.tempoAtSeconds(seconds));
    const auto bar = beat * static_cast<double>(snapshot.numerator) * 4.0
        / static_cast<double>(std::max(1, snapshot.denominator));
    return std::max(1.0e-6, bar / 4.0);
}

double PianoRollComponent::gapUnitSecondsAt(double seconds) const
{
    // A 128th of a quarter note.  The tempo is quarter notes a minute, so a
    // beat here is one of those whatever a bar happens to hold.
    const auto beat = 60.0 / juce::jlimit(20.0, 400.0, snapshot.tempoAtSeconds(seconds));
    return std::max(1.0e-9, beat / 128.0);
}

juce::StringArray PianoRollComponent::splitBatchLyrics(const juce::String& text)
{
    juce::StringArray words;
    juce::String current;
    for (auto character = text.getCharPointer(); !character.isEmpty(); ++character)
    {
        const auto letter = *character;
        // Space, tab and newline, and the full-width space U+3000, which is
        // what an input method set to Chinese or Japanese types.  Reading only
        // the ASCII one would put a whole line onto the first note.
        const auto separates = letter == ' ' || letter == '\t' || letter == '\n'
            || letter == '\r' || letter == juce::juce_wchar(0x3000);
        if (separates)
        {
            if (current.isNotEmpty()) words.add(current);
            current.clear();
            continue;
        }
        current += juce::String::charToString(letter);
    }
    if (current.isNotEmpty()) words.add(current);
    return words;
}

std::vector<juce::String> PianoRollComponent::batchLyricTargets(
    const juce::String& noteId) const
{
    // Voicebank lyric editing always respects the explicit selection, even
    // when it contains just one note. Keep native material phrase input intact.
    const auto* owner = trackForNote(noteId);
    const bool selectionOnly = owner && trackUsesVoicebankSynthesis(*owner);
    if (selectionOnly || selectedNotes.size() > 1)
    {
        std::vector<const NoteData*> chosen;
        std::vector<std::pair<double, const NoteData*>> ordered;
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (selectedNotes.contains(note.id.toStdString())
                        || (selectedNotes.empty() && note.id == noteId))
                        ordered.emplace_back(clip.startSeconds + note.startSeconds, &note);
        std::stable_sort(ordered.begin(), ordered.end(),
                         [](const auto& left, const auto& right)
                         { return left.first < right.first; });
        std::vector<juce::String> ids;
        ids.reserve(ordered.size());
        for (const auto& [when, note] : ordered) ids.push_back(note->id);
        return ids;
    }
    // Otherwise this note and everything after it in the same clip, so one
    // click and one line of text fills a phrase.
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
        {
            std::vector<std::pair<double, juce::String>> ordered;
            auto found = false;
            for (const auto& note : clip.notes)
            {
                ordered.emplace_back(note.startSeconds, note.id);
                if (note.id == noteId) found = true;
            }
            if (!found) continue;
            std::stable_sort(ordered.begin(), ordered.end(),
                             [](const auto& left, const auto& right)
                             { return left.first < right.first; });
            std::vector<juce::String> ids;
            auto reached = false;
            for (const auto& [when, id] : ordered)
            {
                if (id == noteId) reached = true;
                if (reached) ids.push_back(id);
            }
            return ids;
        }
    return {};
}

void PianoRollComponent::showBatchLyricDialog(const juce::String& noteId)
{
    const auto targets = batchLyricTargets(noteId);
    if (targets.empty()) return;
    // Title and the box, nothing else: the instructions ran longer than the
    // thing they explained, and the editor is the only place to type anyway.
    auto* dialog = new juce::AlertWindow(
        I18n::uiText("批量输入歌词"), {},
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("lyrics", {}, I18n::uiText("歌词"));
    if (auto* editor = dialog->getTextEditor("lyrics"))
    {
        editor->setSelectAllWhenFocused(true);
        editor->setEscapeAndReturnKeysConsumed(false);
    }
    dialog->addButton(I18n::uiText("填入"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, targets](int result)
            {
                if (safe != nullptr && result == 1)
                {
                    const auto words = splitBatchLyrics(
                        dialog->getTextEditorContents("lyrics"));
                    std::vector<std::pair<juce::String, juce::String>> labels;
                    const auto count = std::min(static_cast<std::size_t>(words.size()),
                                                targets.size());
                    labels.reserve(count);
                    for (std::size_t index = 0; index < count; ++index)
                        labels.emplace_back(targets[index],
                                            words[static_cast<int>(index)]);
                    // Fewer words than notes leaves the rest alone; more words
                    // than notes drops the extras.  Neither is an error worth
                    // stopping for -- the person can see what happened.
                    safe->model.setNoteLabels(labels);
                }
                delete dialog;
            }), false);
}

void PianoRollComponent::showGapDialog(const juce::String& noteId)
{
    const auto at = absoluteStartOf(noteId);
    if (at < 0.0) return;
    const auto unit = gapUnitSecondsAt(at);
    // It opens at the one-click amount, so the dialog can also say what that
    // amount is in the units it counts in.
    const auto preset = juce::jmax(1, juce::roundToInt(quarterBarSecondsAt(at) / unit));
    auto* dialog = new juce::AlertWindow(
        I18n::uiText("添加指定长度空格"),
        I18n::uiText("在该音符前插入一段空白，该轨道之后的音符整体后移。\n"
                               "单位：1 = 四分音符的 1/128；四分之一小节 = ")
            + juce::String(preset) + juce::String::fromUTF8("。"),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("units", juce::String(preset),
                          I18n::uiText("长度（单位）"));
    if (auto* editor = dialog->getTextEditor("units"))
    {
        editor->setInputRestrictions(7, "0123456789");
        editor->setSelectAllWhenFocused(true);
    }
    dialog->addButton(I18n::uiText("添加"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, noteId, unit](int result)
            {
                if (safe != nullptr && result == 1)
                {
                    const auto units = juce::jmax(0,
                        dialog->getTextEditorContents("units").getIntValue());
                    if (units > 0)
                        safe->model.insertGapBeforeNote(noteId, units * unit);
                }
                delete dialog;
            }), false);
}

double PianoRollComponent::effectiveUtauOverlapFor(const juce::String& noteId) const
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return 0.0;
    if (note->utauOverlapOverrideEnabled) return note->utauOverlapSeconds;
    for (const auto& track : snapshot.tracks)
    {
        if (track.pitchAlgorithm != PitchAlgorithm::utau) continue;
        for (const auto& clip : track.clips)
            for (const auto& candidate : clip.notes)
            {
                if (candidate.id != noteId) continue;
                const auto velocity = note->utauConsonantVelocity
                        != inheritedUtauConsonantVelocity
                    ? note->utauConsonantVelocity : track.utauConsonantVelocity;
                if (const auto timing = backend::UtauRenderer::sampleTiming(
                        track.voicebankDirectory, note->label, note->midiNote,
                        velocity, utauModeUsesRegions(track.utauMode),
                        track.utauMode == UtauMode::mou, note->ownOto()))
                    return timing->overlapSeconds;
                return 0.0;
            }
    }
    return 0.0;
}

void PianoRollComponent::showStpDialog(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    const auto targets = chosenNoteIds(noteId);
    auto* dialog = new juce::AlertWindow(
        I18n::uiText("修改 STP"),
        I18n::uiText("把这个音符所用 oto 条目中的全部分段信息整体移动：offset、重叠、"
            "先行声音、辅音和 cutoff，以及四区域的划分，一起移动同样的距离。\n"
            "正数为后移（读取录音中更靠后的部分），负数为前移；单位是毫秒。\n"
            "音符在曲子里的位置不变，变的只是它读到录音的哪一段。"),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("stp", juce::String(note->utauStpSeconds * 1000.0, 3),
                          juce::String::fromUTF8("STP，ms："));
    if (auto* editor = dialog->getTextEditor("stp"))
    {
        editor->setInputRestrictions(0, "-0123456789.");
        editor->setSelectAllWhenFocused(true);
    }
    dialog->addButton(I18n::uiText("应用"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, targets](int result)
            {
                if (safe != nullptr && result == 1)
                    safe->model.setNotesUtauStp(targets,
                        dialog->getTextEditorContents("stp").getDoubleValue()
                            / 1000.0);
                delete dialog;
            }), false);
}

void PianoRollComponent::showEnvelopeBaseDialog(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    const auto targets = chosenNoteIds(noteId);
    auto* dialog = new juce::AlertWindow(
        I18n::uiText("包络基础值"),
        juce::String(),
        juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("base",
                          juce::String(note->amplitudeEnvelopeBasePercent, 1),
                          I18n::uiText("基础值，% （0–200）："));
    if (auto* editor = dialog->getTextEditor("base"))
    {
        editor->setInputRestrictions(0, "0123456789.");
        editor->setSelectAllWhenFocused(true);
    }
    dialog->addButton(I18n::uiText("应用"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, targets](int result)
            {
                if (safe != nullptr && result == 1)
                    safe->model.setNotesAmplitudeEnvelopeBase(targets,
                        static_cast<float>(
                            dialog->getTextEditorContents("base").getDoubleValue()));
                delete dialog;
            }), false);
}

void PianoRollComponent::showRegionFlagDialog(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    std::vector<juce::String> targets;
    for (const auto& id : selectedNotes) targets.push_back(juce::String::fromUTF8(id.c_str()));
    if (targets.empty()) targets.push_back(noteId);

    auto* dialog = new juce::AlertWindow(
        juce::String::fromUTF8("flag \u62c6\u5206"),
        juce::String::fromUTF8(
            "\u6bcf\u4e2a\u533a\u5355\u72ec\u7684 flag\uff1b\u7559\u7a7a\u5219\u8be5\u533a"
            "\u6cbf\u7528\u97f3\u7b26\u548c\u8f68\u9053\u7684 flag\u3002\n"
            "\u4ec5\u9010\u5e27\u97f3\u8272\u7c7b flag \u53ef\u5206\u533a\uff08g / Mt / Mb / "
            "Mr / Mo / QX / Mu / b / bh / Mf / MH / Mq \u7b49\uff09\uff1b\n"
            "\u5185\u6838\u9009\u62e9\uff08K1/K2/M1/HF\uff09\u3001u\u3001P/p\u3001SK\u3001"
            "JZ/BX/TM \u7b49\u6574\u6bb5\u6027\u8d28\u7684 flag \u4ecd\u4e3a\u5168\u5c40\u3002"),
        juce::MessageBoxIconType::NoIcon);
    const auto fields = regionFlagFieldsFor(noteId);
    for (const auto& field : fields)
    {
        dialog->addTextEditor(field.key, field.value, field.label);
        if (auto* editor = dialog->getTextEditor(field.key))
            editor->setSelectAllWhenFocused(true);
    }
    // What the regions this entry does not have are carrying.  A box that is
    // not on screen was not edited, so its value goes back in as it was rather
    // than being cleared by a dialog that never showed it.
    const std::array<juce::String, 4> stored {
        note->utauRegionFlags1, note->utauRegionFlags2,
        note->utauRegionFlags3, note->utauRegionFlags4 };
    const auto shown = static_cast<int>(fields.size());
    dialog->addButton(juce::String::fromUTF8("\u5e94\u7528"), 1);
    dialog->addButton(juce::String::fromUTF8("\u5173\u95ed\u62c6\u5206"), 2);
    dialog->addButton(juce::String::fromUTF8("\u53d6\u6d88"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, targets, stored, shown](int result)
            {
                if (safe != nullptr && result != 0)
                {
                    auto values = stored;
                    for (int index = 0; index < shown; ++index)
                        values[static_cast<std::size_t>(index)] =
                            dialog->getTextEditorContents(
                                "r" + juce::String(index + 1));
                    safe->model.setNotesRegionFlags(targets, result == 1,
                        values[0], values[1], values[2], values[3]);
                }
                delete dialog;
            }), false);
}

void PianoRollComponent::confirmBakeVibrato(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr || !note->vibratoEnabled) return;
    const auto groupNotice = joinedVibratoFor(noteId)
        ? I18n::uiText("这段跨音符颤音会整体转换，所有成员同时关闭颤音参数。\n\n") : juce::String();
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::QuestionIcon,
        juce::String::fromUTF8("\u4fee\u6539\u98a4\u97f3"),
        groupNotice + juce::String::fromUTF8(
            "\u98a4\u97f3\u5c06\u5199\u6210\u666e\u901a\u7684\u97f3\u9ad8\u6807\u70b9\uff0c"
            "\u53ef\u5728\u6807\u70b9\u6a21\u5f0f\u4e0b\u9010\u70b9\u4fee\u6539\uff0c"
            "\u540c\u65f6\u98a4\u97f3\u53c2\u6570\u5173\u95ed\u3002\n\n"
            "\u8fd9\u4e00\u6b65\u53ef\u4ee5 Ctrl+Z \u64a4\u9500\uff1b"
            "\u4f46\u4e00\u65e6\u5f00\u59cb\u7f16\u8f91\u8fd9\u6761\u66f2\u7ebf\uff0c"
            "\u5c31\u53ea\u80fd\u9760 Ctrl+Z \u9010\u6b65\u9000\u56de\uff0c"
            "\u65e0\u6cd5\u76f4\u63a5\u6062\u590d\u6210\u53c2\u6570\u5f0f\u98a4\u97f3\u3002"),
        juce::String::fromUTF8("\u4fee\u6539"),
        juce::String::fromUTF8("\u53d6\u6d88"),
        this,
        juce::ModalCallbackFunction::create([safe, noteId](int result)
        {
            if (safe == nullptr || result == 0) return;
            safe->model.bakeNoteVibratoIntoPitch(noteId);
        }));
}

void PianoRollComponent::deleteSelectedNotesRippling(const juce::String& noteId)
{
    // Ordinary delete leaves a hole where the notes were, which is right when
    // the timing around them is meant to stay put.  This one takes the time
    // with them: what followed moves back to meet what came before, note or
    // silence either way.  It is what undoes a phrase pasted in.
    std::vector<juce::String> targets;
    for (const auto& id : selectedNotes)
        targets.push_back(juce::String::fromUTF8(id.c_str()));
    if (targets.empty()) targets.push_back(noteId);
    model.removeNotesRippling(targets);
    clearNoteSelection();
}

void PianoRollComponent::resetConsonant(const juce::String& noteId)
{
    if(cvvcTimingEnabled(noteId))
    {
        cvvcBoundaryDrag.reset();pendingCvvcBoundary.reset();
        model.resetCvvcBoundaries(chosenNoteIds(noteId),model.revisionNumber());
        refreshFromModel();repaint();
        return;
    }
    // What holds a consonant away from where the voicebank puts it is a pinned
    // preutterance.  It can be left somewhere unusable -- a lead-in so long
    // there is no handle left to grab, most of all -- and then there is no way
    // back through the handle itself.  This is the way back, and it puts the
    // consonant's end on the note's own start.
    //
    // Two things it does not touch.  The consonant velocity is a setting,
    // chosen deliberately and read as such, and it moves where the consonant
    // begins rather than where it ends.  And the two movable lines dividing
    // the vowel are somebody's work on the vowel, nothing to do with the
    // consonant: this used to throw them away along with the pin, which is a
    // great deal more than was asked for.
    //
    // They cannot simply be left alone, though.  They are held as fractions of
    // the sounding span, and restoring the lead-in moves where that span
    // begins -- so the same fractions would land the lines somewhere else.
    // Where they stand now is noted here and restated on the next layout,
    // against the span the reset has just changed.
    std::vector<juce::String> targets;
    for (const auto& id : selectedNotes)
        targets.push_back(juce::String::fromUTF8(id.c_str()));
    if (targets.empty()) targets.push_back(noteId);

    const auto record = [this](const juce::String& id)
    {
        const auto* note = findNote(id);
        if (note == nullptr || !note->utauJieSplitSet) return;
        const auto span = utauSoundSpans.find(id.toStdString());
        if (span == utauSoundSpans.end()) return;
        const auto start = span->second.first;
        const auto length = span->second.second - start;
        if (length <= 1.0e-9) return;
        pendingSplitRestate.push_back({ id, start + note->utauJieSplit1 * length,
                                        start + note->utauJieSplit2 * length,
                                        start + note->utauJieSplit3 * length });
    };
    for (const auto& id : targets)
    {
        record(id);
        // The note before this one ends where this one's lead-in begins, so
        // its span moves too, and its lines with it.
        if (const auto found = utauNeighbours.find(id.toStdString());
            found != utauNeighbours.end() && !found->second.previous.empty())
            record(juce::String::fromUTF8(found->second.previous.c_str()));
    }

    for (const auto& id : targets)
        model.setNoteUtauTimingOverrides(id, false, 0.0, 0.0);
}

void PianoRollComponent::restateStandingSplits()
{
    // Spend the list first: writing to the model brings us back through the
    // layout, and a list still holding entries would go round again.
    const auto standing = std::exchange(pendingSplitRestate,
                                        std::vector<StandingSplit>{});
    for (const auto& entry : standing)
    {
        const auto span = utauSoundSpans.find(entry.id.toStdString());
        if (span == utauSoundSpans.end()) continue;
        const auto start = span->second.first;
        const auto length = span->second.second - start;
        if (length <= 1.0e-9) continue;
        const auto asFraction = [start, length](double absolute)
        {
            return juce::jlimit(0.0, 1.0, (absolute - start) / length);
        };
        model.setNotesUtauJieSplit({ entry.id }, asFraction(entry.first),
                                   asFraction(entry.second), asFraction(entry.third));
    }
}

void PianoRollComponent::showVibratoDialog(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    const auto control = vibratoControlNote(*note);
    note = &control;
    // Apply to the whole selection when several notes are selected, so a chorus
    // of notes can be given the same swing in one go.
    std::vector<juce::String> targets;
    for (const auto& id : selectedNotes) targets.push_back(juce::String::fromUTF8(id.c_str()));
    if (targets.empty()) targets.push_back(noteId);
    const auto native = std::all_of(targets.begin(), targets.end(),
        [this](const auto& id) { return nativeNoteTimingEnabled(id); });

    auto* dialog = new juce::AlertWindow(
        I18n::uiText("颤音"),
        native ? I18n::uiText("对选中的 ") + juce::String(static_cast<int>(targets.size()))
            + I18n::uiText(" 个音频片段添加颤音，叠加在原有音高线上。同音高且首尾相接的选中音节视为一个完整音符，长度、淡入淡出和相位连续计算；不会修改原始音高参考。")
            : I18n::uiText("以音高线为基准上下颤动；长度从颤音终止处往前算。同音高且首尾相接的选中音节共用一段完整颤音。"),
        juce::MessageBoxIconType::NoIcon);
    dialog->setComponentID("vibrato-editor");
    struct Field { const char* key; const char* label; double value; };
    // Same order as vibratoFieldKeys, which is the order a preset lists them.
    const std::array<Field, 7> fields {{
        { "length",  "长度 (% 于音符总长)：", note->vibratoLengthPercent },
        { "cycle",   "周期 (ms)：",                                    note->vibratoCycleMs },
        { "depth",   "深度 (cents)：",                                 note->vibratoDepthCents },
        { "fadein",  "淡入 (%)：",                                    note->vibratoFadeInPercent },
        { "fadeout", "淡出 (%)：",                                    note->vibratoFadeOutPercent },
        { "phase",   "相位 (%)：",                                    note->vibratoPhasePercent },
        { "offset",  "中心偏移 (% 于深度)：", note->vibratoOffsetPercent }
    }};
    for (const auto& field : fields)
    {
        dialog->addTextEditor(field.key, juce::String(field.value, 3),
                              juce::String::fromUTF8(field.label));
        if (auto* editor = dialog->getTextEditor(field.key))
        {
            editor->setSelectAllWhenFocused(true);
            editor->setInputRestrictions(0, "-0123456789.");
        }
    }
    // Not one of UTAU's seven, so not part of a preset either: where the
    // swing stops.  100 is the note's end, where UTAU always stops it.
    dialog->addTextEditor("end", juce::String(note->vibratoEndPercent, 3),
                          I18n::uiText("终止位置 (% 于音符，100 = 末尾)："));
    if (auto* editor = dialog->getTextEditor("end"))
    {
        editor->setSelectAllWhenFocused(true);
        editor->setInputRestrictions(0, "0123456789.");
    }
    auto* presets = new VibratoPresetBar(*dialog,
        onLoadVibratoPresets ? onLoadVibratoPresets() : juce::String(),
        [safe = juce::Component::SafePointer<PianoRollComponent>(this)]
        (const juce::String& text)
        {
            if (safe != nullptr && safe->onSaveVibratoPresets)
                safe->onSaveVibratoPresets(text);
        });
    dialog->addCustomComponent(presets);
    juce::ToggleButton* realLine = nullptr;
    if (native)
    {
        realLine = new juce::ToggleButton(I18n::uiText("在音高线上显示实际颤音"));
        realLine->setName({}); // AlertWindow labels named custom components separately.
        realLine->setComponentID("vibrato-real-line");
        realLine->setTooltip(I18n::uiText("普通编辑时显示叠加后的音高；标点模式保留基础曲线与颤音参考，便于分别编辑。"));
        realLine->setSize(390, 28);
        realLine->setToggleState(!note->vibratoEnabled || note->vibratoRealLine,
                                juce::dontSendNotification);
        dialog->addCustomComponent(realLine);
    }
    dialog->addButton(I18n::uiText("应用"), 1);
    dialog->addButton(I18n::uiText("关闭颤音"), 2);
    dialog->addButton(I18n::uiText("取消"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, presets, realLine, targets](int result)
            {
                if (safe != nullptr && result != 0)
                {
                    NoteData parameters;
                    const auto number = [dialog](const char* key, double fallback)
                    {
                        const auto text = dialog->getTextEditorContents(key).trim();
                        return text.isEmpty() ? fallback : text.getDoubleValue();
                    };
                    parameters.vibratoLengthPercent = number("length", 65.0);
                    parameters.vibratoCycleMs = number("cycle", 180.0);
                    parameters.vibratoDepthCents = number("depth", 35.0);
                    parameters.vibratoFadeInPercent = number("fadein", 20.0);
                    parameters.vibratoFadeOutPercent = number("fadeout", 20.0);
                    parameters.vibratoPhasePercent = number("phase", 0.0);
                    parameters.vibratoOffsetPercent = number("offset", 0.0);
                    parameters.vibratoEndPercent = number("end", 100.0);
                    safe->model.setNotesVibrato(targets, parameters, result == 1,
                        realLine ? std::optional<bool>(realLine->getToggleState()) : std::nullopt);
                }
                dialog->removeCustomComponent(0);
                delete presets;
                if (realLine != nullptr)
                {
                    dialog->removeCustomComponent(0);
                    delete realLine;
                }
                delete dialog;
            }), false);
}

void PianoRollComponent::showNoteTimingDialog(const juce::String& noteId)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    std::optional<backend::UtauSampleTiming> timing;
    for (const auto& track : snapshot.tracks)
    {
        if (track.pitchAlgorithm != PitchAlgorithm::utau) continue;
        const auto contains = std::any_of(track.clips.begin(), track.clips.end(),
            [&](const auto& clip)
            {
                return std::any_of(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& candidate) { return candidate.id == noteId; });
            });
        if (!contains) continue;
        const auto velocity = note->utauConsonantVelocity
                != inheritedUtauConsonantVelocity
            ? note->utauConsonantVelocity : track.utauConsonantVelocity;
        timing = backend::UtauRenderer::sampleTiming(
            track.voicebankDirectory, note->label, note->midiNote, velocity,
            utauModeUsesRegions(track.utauMode),
            track.utauMode == UtauMode::mou, note->ownOto());
        break;
    }
    // No entry for this lyric: the note cannot sound, but it may still be
    // carrying a pin from a lyric it used to have, and refusing to open was
    // the only thing standing between the user and taking that pin out.
    const auto aliasKnown = timing.has_value();
    if (!timing) timing = backend::UtauSampleTiming {};

    const auto preutteranceSeconds = note->utauPreutteranceOverrideEnabled
        ? note->utauPreutteranceSeconds : timing->preutteranceSeconds;
    const auto overlapSeconds = note->utauOverlapOverrideEnabled
        ? note->utauOverlapSeconds : timing->overlapSeconds;
    auto* dialog = new juce::AlertWindow(I18n::uiText("时序"),
        I18n::uiText("先行声音在辅音速度计算后应用；重叠为正时交叉衔接，为负时形成静音间隔。\n"
            "填回音源值后直接应用，音符重新跟随音源；手动改过的数值则固定不变。")
        + (aliasKnown ? juce::String()
                      : I18n::uiText("\n音源库中没有这个歌词，所以下方的音源值是空的；"
                            "「填回音源值」仍可用来清掉上一个歌词留下的固定值。")),
                                          juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("preMilliseconds",
        juce::String(preutteranceSeconds * 1000.0, 3),
        I18n::uiText("先行声音 (Pre)，ms："));
    dialog->addTextEditor("overlapMilliseconds",
        juce::String(overlapSeconds * 1000.0, 3),
        I18n::uiText("重叠 (Overlap)，ms："));
    if (auto* editor = dialog->getTextEditor("preMilliseconds"))
    {
        editor->setSelectAllWhenFocused(true);
        editor->setInputRestrictions(0, "0123456789.");
    }
    if (auto* editor = dialog->getTextEditor("overlapMilliseconds"))
    {
        editor->setSelectAllWhenFocused(true);
        editor->setInputRestrictions(0, "-0123456789.");
    }
    // Restoring is a custom component rather than a dialog button: every
    // AlertWindow button dismisses the window, so as a button it put the
    // voicebank values back and left before they could be seen or adjusted.
    auto* restoreButton = new juce::TextButton(
        I18n::uiText("填回音源值"));
    restoreButton->setSize(120, 24);
    // Whether the fields still hold exactly what the voicebank says.  Applying
    // in that state hands the note back to the voicebank instead of pinning
    // the numbers, so a later oto edit reaches it.
    auto restored = std::make_shared<bool>(false);
    const auto sourcePre = timing->preutteranceSeconds;
    const auto sourceOverlap = timing->overlapSeconds;
    restoreButton->onClick = [dialog, restored, sourcePre, sourceOverlap]
    {
        // Silent set: notifying would immediately clear the flag below.
        if (auto* editor = dialog->getTextEditor("preMilliseconds"))
            editor->setText(juce::String(sourcePre * 1000.0, 3), false);
        if (auto* editor = dialog->getTextEditor("overlapMilliseconds"))
            editor->setText(juce::String(sourceOverlap * 1000.0, 3), false);
        *restored = true;
    };
    for (const auto* key : { "preMilliseconds", "overlapMilliseconds" })
        if (auto* editor = dialog->getTextEditor(key))
            editor->onTextChange = [restored] { *restored = false; };
    dialog->addCustomComponent(restoreButton);
    dialog->addButton(I18n::uiText("应用"), 1);
    dialog->addButton(I18n::uiText("取消"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, restoreButton, restored, noteId](int result)
            {
                if (safe != nullptr && result == 1)
                {
                    const auto enabled = !*restored;
                    const auto preutteranceSeconds = enabled
                        ? std::max(0.0, dialog->getTextEditorContents(
                            "preMilliseconds").getDoubleValue() / 1000.0)
                        : 0.0;
                    const auto overlapSeconds = enabled
                        ? dialog->getTextEditorContents(
                            "overlapMilliseconds").getDoubleValue() / 1000.0
                        : 0.0;
                    safe->model.setNoteUtauTimingOverrides(
                        noteId, enabled, preutteranceSeconds, overlapSeconds);
                }
                dialog->removeCustomComponent(0);
                delete restoreButton;
                delete dialog;
            }), false);
}

void PianoRollComponent::showCustomBezierDialog(const juce::String& noteId,
                                                 int anchorIndex)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    const auto& anchors = pitchAnchorsFor(*note);
    if (!pitchAnchorHasIncoming(*note, anchorIndex)) return;
    const auto& point = anchors[static_cast<std::size_t>(anchorIndex)];
    auto* editor = new BezierParameterEditor(strings,
        { point.bezierX1, point.bezierY1, point.bezierX2, point.bezierY2 });
    auto* dialog = new juce::AlertWindow(strings.text("pitchCurve.bezierTitle"),
        strings.text("pitchCurve.bezierHelp"), juce::MessageBoxIconType::NoIcon);
    dialog->addCustomComponent(editor);
    dialog->addButton(strings.text("dialog.apply"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, editor, noteId, anchorIndex](int result) mutable
            {
                if (safe != nullptr && result == 1)
                {
                    safe->setPitchAnchorShape(noteId, anchorIndex,
                        PitchCurveShape::customBezier, editor->values());
                }
                dialog->removeCustomComponent(0);
                delete editor;
                delete dialog;
            }), false);
}

std::optional<float> PianoRollComponent::anchorMidiForFrequency(double hertz)
{
    if (!std::isfinite(hertz) || hertz <= 0.0) return {};
    const auto midi = 69.0 + 12.0 * std::log2(hertz / 440.0);
    if (!std::isfinite(midi) || midi < 0.0 || midi > 127.0) return {};
    return static_cast<float>(midi);
}

double PianoRollComponent::anchorFrequencyForMidi(float midi)
{
    return 440.0 * std::pow(2.0, (static_cast<double>(midi) - 69.0) / 12.0);
}

std::vector<PitchCurveEditPoint> PianoRollComponent::anchorsWithFrequency(
    std::vector<PitchCurveEditPoint> anchors, int anchorIndex, double hertz)
{
    if (anchorIndex < 0 || anchorIndex >= static_cast<int>(anchors.size())) return {};
    const auto midi = anchorMidiForFrequency(hertz);
    if (!midi) return {};
    prepareSampledPitchPointEdit(anchors, anchorIndex);
    // Only the pitch of the one point.  Its time, the shape of the segment
    // leading into it and its Bezier handles are all left as they were.
    anchors[static_cast<std::size_t>(anchorIndex)].diffSingerRestoreSupport = false;
    anchors[static_cast<std::size_t>(anchorIndex)].targetMidi = *midi;
    return anchors;
}

void PianoRollComponent::showAnchorFrequencyDialog(const juce::String& noteId,
                                                    int anchorIndex)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return;
    const auto& anchors = pitchAnchorsFor(*note);
    if (anchorIndex < 0 || anchorIndex >= static_cast<int>(anchors.size())) return;
    // Opened on the pitch the point already has, so it can be nudged by a few
    // hertz without having to be worked out first.
    const auto current = anchorFrequencyForMidi(
        anchors[static_cast<std::size_t>(anchorIndex)].targetMidi
            + displayedPitchOffset(*note,anchors[static_cast<std::size_t>(anchorIndex)].timeSeconds));
    auto* dialog = new juce::AlertWindow(strings.text("pitchCurve.frequencyTitle"),
        strings.text("pitchCurve.frequencyHelp"), juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("hertz", juce::String(current, 3),
                          strings.text("pitchCurve.frequencyField"));
    if (auto* editor = dialog->getTextEditor("hertz"))
    {
        editor->setInputRestrictions(0, "0123456789.");
        editor->setSelectAllWhenFocused(true);
    }
    dialog->addButton(strings.text("dialog.apply"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, noteId, anchorIndex](int result)
            {
                if (safe != nullptr && result == 1)
                    if (const auto* currentNote = safe->findNote(noteId))
                    {
                        const auto& anchors=safe->pitchAnchorsFor(*currentNote);
                        if (anchorIndex>=0 && anchorIndex<static_cast<int>(anchors.size()))
                        {
                            // Convert to the base layer before the helper removes hidden
                            // samples and remaps the handle index.
                            const auto offset=safe->displayedPitchOffset(*currentNote,
                                anchors[static_cast<size_t>(anchorIndex)].timeSeconds);
                            const auto baseHertz=dialog->getTextEditorContents("hertz").getDoubleValue()
                                * std::pow(2.0,-static_cast<double>(offset)/12.0);
                            auto moved=PianoRollComponent::anchorsWithFrequency(anchors,anchorIndex,baseHertz);
                            if (!moved.empty()) safe->model.setNotePitchCurve(noteId,std::move(moved),true);
                        }
                    }
                delete dialog;
            }), false);
}

const TrackData* PianoRollComponent::trackOf(const juce::String& noteId) const
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId) return &track;
    return nullptr;
}

const AudioEngine::ProjectedNotePitch* PianoRollComponent::protocolPitchFor(const TrackData& track,const juce::String& noteId) const
{
    // Native FILE and DiffSinger adapters retain their own policies. Protocol
    // notes share the host's one pure OWN/nominal/explicit-graph planner.
    if(!trackUsesVoicebankSynthesis(track)||trackIsDiffSinger(track))return nullptr;
    auto& cache=pitchPlanPreview?protocolPreviewCache:protocolPitchCache;
    auto found=cache.find(track.id);
    if(found==cache.end())found=cache.emplace(track.id,AudioEngine::projectTrackNotePitch(pitchPreviewTrack(track),
        pitchPlanPreview?pitchPlanPreview->connections:snapshot.nativeConnections)).first;
    for(const auto& note:found->second)if(note.noteId==noteId)return &note;return nullptr;
}

const TrackData& PianoRollComponent::pitchPreviewTrack(const TrackData& track) const
{
    if(pitchPlanPreview)for(const auto& proposed:pitchPlanPreview->tracks)if(proposed.id==track.id)return proposed;
    return track;
}
const NoteData& PianoRollComponent::pitchPreviewNote(const NoteData& note) const
{
    if(pitchPlanPreview)for(const auto& track:pitchPlanPreview->tracks)for(const auto& clip:track.clips)
        for(const auto& proposed:clip.notes)if(proposed.id==note.id)return proposed;
    return note;
}
void PianoRollComponent::clearPitchPlanPreview()
{
    pitchPlanPreview.reset();pitchPendingEdits.clear();pitchPreviewSharedCache.clear();connectionPreviewCache.clear();pitchPreviewAnchorCache.clear();protocolPreviewCache.clear();
}
bool PianoRollComponent::updatePitchPlanPreview(const std::vector<ProjectModel::PointCurveEdit>& edits,std::uint64_t revision)
{
    ProjectModel::PointCurvePlanPreview proposed;juce::String error;
    if(!model.previewPointCurves(edits,revision,proposed,error))
    {if(onEditRejected&&!error.isEmpty())onEditRejected(error);return false;}
    pitchPlanPreview=std::move(proposed);pitchPendingEdits=edits;
    pitchPreviewSharedCache.clear();connectionPreviewCache.clear();pitchPreviewAnchorCache.clear();protocolPreviewCache.clear();return true;
}
const std::vector<ResolvedConnectionPitchCurve>& PianoRollComponent::connectionCurvesFor(const TrackData& track,bool preview) const
{
    auto& cache=preview&&pitchPlanPreview?connectionPreviewCache:connectionPitchCache;
    if(const auto found=cache.find(track.id);found!=cache.end())return found->second;
    return cache.emplace(track.id,resolvedConnectionPitchCurves(preview?pitchPreviewTrack(track):track,
        preview&&pitchPlanPreview?pitchPlanPreview->connections:snapshot.nativeConnections)).first->second;
}
std::vector<ResolvedConnectionPitchCurve> PianoRollComponent::visibleConnectionCurves(bool preview) const
{
    std::vector<ResolvedConnectionPitchCurve> visible;if(sourceEditMode)return visible;
    for(const auto& track:snapshot.tracks)
    {
        if(focusedTrack.isNotEmpty()&&track.id!=focusedTrack)continue;
        for(const auto& graph:connectionCurvesFor(track,preview))
        {
            if(!graph.active)continue;bool incidentVisible=false;
            for(const auto& clip:track.clips)if(clipIsVisible(track,clip))for(const auto& note:clip.notes)
                incidentVisible|=note.id==graph.leftNoteId||note.id==graph.rightNoteId;
            if(incidentVisible)visible.push_back(graph);
        }
    }
    return visible;
}
std::vector<ResolvedConnectionPitchCurve> PianoRollComponent::diagnosticConnectionPitchCurves() const
{ return visibleConnectionCurves(); }
std::optional<juce::Point<float>> PianoRollComponent::diagnosticConnectionPitchPoint(const juce::String& id,int index) const
{
    for(const auto& curve:visibleConnectionCurves())if(curve.connectionId==id&&juce::isPositiveAndBelow(index,static_cast<int>(curve.points.size())))
    {const auto& p=curve.points[static_cast<std::size_t>(index)];return juce::Point<float>(timeToX(p.timeSeconds),midiToY(p.targetMidi)+rowHeight*.5f);}
    return {};
}
std::optional<std::pair<juce::String,int>> PianoRollComponent::connectionPointAt(juce::Point<float> position) const
{
    float best=81;std::optional<std::pair<juce::String,int>> result;
    for(const auto& curve:visibleConnectionCurves(false))for(std::size_t i=0;i<curve.points.size();++i)
    {
        const auto& p=curve.points[i];const juce::Point<float> at(timeToX(p.timeSeconds),midiToY(p.targetMidi)+rowHeight*.5f);
        const auto distance=at.getDistanceSquaredFrom(position);
        if(distance<best){best=distance;result=std::pair{curve.connectionId,static_cast<int>(i)};}
    }
    return result;
}
std::optional<juce::String> PianoRollComponent::connectionCurveAt(juce::Point<float> position) const
{
    const auto seconds=static_cast<double>(position.x-58)/pixelsPerSecond;float best=std::numeric_limits<float>::max();std::optional<juce::String> result;
    for(const auto& graph:visibleConnectionCurves(false))if(seconds>=graph.fromSeconds-1.e-9&&seconds<=graph.toSeconds+1.e-9)
    {
        const auto y=midiToY(evaluatePitchCurve(graph.points,seconds))+rowHeight*.5f;
        if(std::abs(y-position.y)<best){best=std::abs(y-position.y);result=graph.connectionId;}
    }
    return result;
}
void PianoRollComponent::clipNotePitchAgainstConnections(juce::Graphics& g,const TrackData& track) const
{
    for(const auto& curve:connectionCurvesFor(track))if(curve.active)
    {
        const auto left=timeToX(curve.fromSeconds),right=timeToX(curve.toSeconds);
        g.excludeClipRegion({static_cast<int>(std::floor(left)),0,std::max(1,static_cast<int>(std::ceil(right-left))),getHeight()});
    }
}
void PianoRollComponent::drawConnectionPitchCurves(juce::Graphics& g)
{
    if(!showPitchLine||sourceEditMode||tool==Tool::amplitude)return;
    for(const auto& curve:visibleConnectionCurves())
    {
        juce::Path line;
        for(std::size_t i=1;i<curve.points.size();++i)
        {
            const auto from=curve.points[i-1].timeSeconds,to=curve.points[i].timeSeconds;
            const auto steps=std::max(2,static_cast<int>(std::ceil((to-from)*pixelsPerSecond/2)));
            for(int j=0;j<=steps;++j)
            {
                if(i>1&&j==0)continue;const auto time=from+(to-from)*j/steps;
                const juce::Point<float> at(timeToX(time),midiToY(evaluatePitchCurve(curve.points,time))+rowHeight*.5f);
                if(i==1&&j==0)line.startNewSubPath(at);else line.lineTo(at);
            }
        }
        g.setColour(Palette::pitchLine);g.strokePath(line,juce::PathStrokeType(2.2f,juce::PathStrokeType::curved));
        if(tool!=Tool::points)continue;
        for(std::size_t i=0;i<curve.points.size();++i)
        {
            const auto& p=curve.points[i];const bool selected=pointIsSelected(PointSelection::connectionPitch,curve.connectionId,static_cast<int>(i));
            const auto radius=selected?5.f:3.8f;const juce::Point<float> at(timeToX(p.timeSeconds),midiToY(p.targetMidi)+rowHeight*.5f);
            g.setColour(selected?Palette::selection:Palette::panelRaised);g.fillEllipse(at.x-radius,at.y-radius,2*radius,2*radius);
            g.setColour(rollInk(Palette::noteEdge,Palette::graphBackground));g.drawEllipse(at.x-radius,at.y-radius,2*radius,2*radius,selected?2.f:1.4f);
        }
    }
}

juce::PopupMenu PianoRollComponent::connectionPitchPointMenu(const juce::String& id,int index) const
{
    juce::PopupMenu menu;
    for(const auto& curve:visibleConnectionCurves(false))if(curve.connectionId==id&&juce::isPositiveAndBelow(index,static_cast<int>(curve.points.size())))
    {
        const auto shapeIndex=std::max(1,index);const auto shape=curve.points[static_cast<std::size_t>(shapeIndex)].shape;
        menu.addSectionHeader(I18n::uiText("连接音高标点 / Connection pitch"));
        const PitchCurveShape shapes[]{PitchCurveShape::natural,PitchCurveShape::linear,PitchCurveShape::smooth,PitchCurveShape::easeIn,PitchCurveShape::easeOut,PitchCurveShape::customBezier};
        for(int i=0;i<6;++i)menu.addItem(i+1,pitchCurveShapeName(shapes[i]),true,shape==shapes[i]);
        menu.addSeparator();menu.addItem(7,I18n::uiText("删除内部标点 / Delete interior point"),index>0&&index+1<static_cast<int>(curve.points.size()));break;
    }
    return menu;
}
bool PianoRollComponent::applyConnectionPitchPointMenu(const juce::String& id,int index,int result)
{
    for(const auto& curve:visibleConnectionCurves(false))if(curve.connectionId==id&&juce::isPositiveAndBelow(index,static_cast<int>(curve.points.size())))
    {
        auto points=curve.points;
        if(result==7)
        {
            if(index==0||index+1==static_cast<int>(points.size()))return false;
            points.erase(points.begin()+index);
        }
        else if(result>=1&&result<=6)
        {
            const PitchCurveShape shapes[]{PitchCurveShape::natural,PitchCurveShape::linear,PitchCurveShape::smooth,PitchCurveShape::easeIn,PitchCurveShape::easeOut,PitchCurveShape::customBezier};
            points[static_cast<std::size_t>(std::max(1,index))].shape=shapes[result-1];
        }
        else return false;
        ProjectModel::PointCurveEdit edit;edit.kind=ProjectModel::PointCurveEdit::connectionPitch;edit.connectionId=id;
        edit.pitchPoints=std::move(points);edit.reanchorConnectionEndpoints=true;juce::String error;
        const bool changed=model.setPointCurves({edit},model.revisionNumber(),&error);
        if(!changed&&!error.isEmpty()&&onEditRejected)onEditRejected(error);
        if(changed)
        {
            clearPointSelection();refreshFromModel();
            selectedPoint=PointSelection{id,result==7?-1:index,model.revisionNumber(),PointSelection::connectionPitch};repaint();
        }
        return changed;
    }
    return false;
}
bool PianoRollComponent::diagnosticConnectionPitchShape(const juce::String& id,int index,PitchCurveShape shape)
{
    const PitchCurveShape shapes[]{PitchCurveShape::natural,PitchCurveShape::linear,PitchCurveShape::smooth,PitchCurveShape::easeIn,PitchCurveShape::easeOut,PitchCurveShape::customBezier};
    for(int i=0;i<6;++i)if(shapes[i]==shape)return applyConnectionPitchPointMenu(id,index,i+1);return false;
}
void PianoRollComponent::showConnectionPitchPointMenu(const juce::String& id,int index,juce::Point<int> at)
{
    auto menu=connectionPitchPointMenu(id,index);const auto revision=model.revisionNumber();
    juce::Component::SafePointer<PianoRollComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({at.x,at.y,1,1}),[safe,id,index,revision](int result)
    {if(safe&&result&&safe->model.revisionNumber()==revision)safe->applyConnectionPitchPointMenu(id,index,result);});
}
bool PianoRollComponent::connectionPitchMouseDown(const juce::MouseEvent& event)
{
    if(sourceEditMode||tool!=Tool::points||dsPitchOffsetMode||snapshot.nativeConnections.empty())return false;
    refreshFromModel();const auto hit=connectionPointAt(event.position);
    if(!hit){graphPointDragging=false;return connectionCurveAt(event.position).has_value();}
    for(const auto& curve:visibleConnectionCurves(false))if(curve.connectionId==hit->first)
    {
        selectedPoint=PointSelection{curve.connectionId,hit->second,model.revisionNumber(),PointSelection::connectionPitch};
        if(event.mods.isPopupMenu())
        {graphPointDragging=false;showConnectionPitchPointMenu(curve.connectionId,hit->second,event.getScreenPosition());return true;}
        graphPointDragging=true;graphDragValid=false;graphDragId=curve.connectionId;graphDragIndex=hit->second;
        graphDragDown=event.position;graphDragOriginal=curve.points;graphDragRevision=model.revisionNumber();
        draggedNote.clear();dragMode=DragMode::none;repaint();return true;
    }
    return false;
}
void PianoRollComponent::moveConnectionPitchPoint(const juce::MouseEvent& event)
{
    if(!graphPointDragging||!juce::isPositiveAndBelow(graphDragIndex,static_cast<int>(graphDragOriginal.size())))return;
    const auto delta=event.position-graphDragDown;auto points=graphDragOriginal;
    auto& point=points[static_cast<std::size_t>(graphDragIndex)];
    const auto first=graphDragIndex>0?points[static_cast<std::size_t>(graphDragIndex-1)].timeSeconds+minimumAnchorSeparation:0.0;
    const auto last=graphDragIndex+1<static_cast<int>(points.size())?points[static_cast<std::size_t>(graphDragIndex+1)].timeSeconds-minimumAnchorSeparation:std::numeric_limits<double>::max();
    point.timeSeconds=juce::jlimit(first,last,point.timeSeconds+delta.x/pixelsPerSecond);
    if(graphDragIndex>0&&graphDragIndex+1<static_cast<int>(points.size()))point.targetMidi=juce::jlimit(0.f,127.f,point.targetMidi-delta.y/rowHeight);
    ProjectModel::PointCurveEdit edit;edit.kind=ProjectModel::PointCurveEdit::connectionPitch;edit.connectionId=graphDragId;
    edit.pitchPoints=std::move(points);edit.reanchorConnectionEndpoints=true;
    graphDragValid=updatePitchPlanPreview({edit},graphDragRevision);repaint();
}
bool PianoRollComponent::connectionPitchDoubleClick(const juce::MouseEvent& event)
{
    if(sourceEditMode||tool!=Tool::points||dsPitchOffsetMode)return false;
    if(connectionPointAt(event.position))return true;
    const auto owner=connectionCurveAt(event.position);if(!owner)return false;
    for(const auto& curve:visibleConnectionCurves(false))if(curve.connectionId==*owner)
    {
        const auto time=static_cast<double>(event.position.x-58)/pixelsPerSecond;
        if(time<=curve.fromSeconds+minimumAnchorSeparation||time>=curve.toSeconds-minimumAnchorSeparation)return true;
        if(std::any_of(curve.points.begin(),curve.points.end(),[&](const auto& p){return std::abs(p.timeSeconds-time)<minimumAnchorSeparation;}))return true;
        auto points=curve.points;PitchCurveEditPoint added{time,juce::jlimit(0.f,127.f,pitchMidiFromY(event.position.y)),PitchCurveShape::natural};
        points.insert(std::upper_bound(points.begin(),points.end(),time,[](double t,const auto& p){return t<p.timeSeconds;}),added);
        ProjectModel::PointCurveEdit edit;edit.kind=ProjectModel::PointCurveEdit::connectionPitch;edit.connectionId=curve.connectionId;
        edit.pitchPoints=std::move(points);edit.reanchorConnectionEndpoints=true;juce::String error;
        if(!model.setPointCurves({edit},model.revisionNumber(),&error)&&onEditRejected)onEditRejected(error);
        graphPointDragging=false;clearPointSelection();refreshFromModel();repaint();return true;
    }
    return true;
}

const SharedPitchLines& PianoRollComponent::sharedLinesFor(const TrackData& track,
                                                           bool followDrag) const
{
    if(followDrag&&pitchPlanPreview)
    {
        if(const auto found=pitchPreviewSharedCache.find(track.id);found!=pitchPreviewSharedCache.end())return found->second;
        return pitchPreviewSharedCache.emplace(track.id,sharedPitchLines(pitchPreviewTrack(track),pitchPlanPreview->connections)).first->second;
    }
    if (followDrag && dragMode == DragMode::pointPitch && !pitchStroke.empty()
        && draggedNote.isNotEmpty())
    {
        auto holdsDragged = false;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                holdsDragged = holdsDragged || note.id == draggedNote;
        if (holdsDragged)
        {
            // Once per stroke, not once per note painted: the key is the
            // stroke itself.
            auto key = static_cast<std::uint64_t>(draggedNote.hashCode64());
            for (const auto& point : pitchStroke)
            {
                key = key * 1099511628211ull
                    ^ static_cast<std::uint64_t>(std::llround(point.timeSeconds * 1.0e7));
                key = key * 1099511628211ull
                    ^ static_cast<std::uint64_t>(std::llround(point.targetMidi * 1.0e4));
                key = key * 1099511628211ull ^ static_cast<std::uint64_t>(point.shape);
                for (const auto handle : { point.bezierX1, point.bezierY1,
                                            point.bezierX2, point.bezierY2 })
                    key = key * 1099511628211ull
                        ^ static_cast<std::uint64_t>(std::llround(handle * 1.0e7));
            }
            if (key != draggedSharedKey)
            {
                draggedSharedLines = sharedPitchLines(track, snapshot.nativeConnections, draggedNote, &pitchStroke);
                draggedSharedKey = key;
            }
            return draggedSharedLines;
        }
    }
    const auto found = sharedLineCache.find(track.id);
    if (found != sharedLineCache.end()) return found->second;
    return sharedLineCache.emplace(track.id, sharedPitchLines(track, snapshot.nativeConnections)).first->second;
}

const NativeNoiseRanges& PianoRollComponent::nativePitchVisibilityFor(
    const ClipData& clip, const NoteData& note) const
{
    const auto found = nativePitchVisibilityCache.find(note.id);
    if (found != nativePitchVisibilityCache.end()) return found->second;
    return nativePitchVisibilityCache.emplace(note.id, nativePitchDisplayRanges(clip, note)).first->second;
}

const NativeNoiseRanges& PianoRollComponent::nativePitchAuthoredFor(
    const ClipData& clip, const NoteData& note) const
{
    const auto found = nativePitchAuthoredCache.find(note.id);
    if (found != nativePitchAuthoredCache.end()) return found->second;
    return nativePitchAuthoredCache.emplace(note.id, nativeAuthoredPitchDisplayRanges(clip, note)).first->second;
}

std::optional<PianoRollComponent::TransitionBridge> PianoRollComponent::transitionBridge(
    const TrackData& track, const NoteData& note, double absoluteStart)
{
    if(trackUsesVoicebankSynthesis(track)&&!trackIsDiffSinger(track))return std::nullopt; // nominal seams are already in the shared host projector
    const PositionedUtauNote positioned {
        note.id, absoluteStart, absoluteStart + note.durationSeconds,
        note.utauAutoPitchTransition
    };
    const auto& anchors = draggedNote == note.id && dragMode == DragMode::pointPitch
            && !pitchStroke.empty()
        ? pitchStroke : pitchAnchorsFor(note);
    if (anchors.empty()) return std::nullopt;
    // Two notes sharing a line are joined by the line itself.
    if (const auto* member = sharedLinesFor(track).memberFor(note.id);
        member != nullptr && member->joinsNext)
        return std::nullopt;
    const auto next = nextUtauNoteFor(note.id);
    if (!next || !formsAdjacentPitchBoundary(positioned, *next)) return std::nullopt;
    const auto* nextNote = findNote(next->id);
    if (nextNote == nullptr) return std::nullopt;
    const auto& nextAnchors = draggedNote == nextNote->id
            && dragMode == DragMode::pointPitch && !pitchStroke.empty()
        ? pitchStroke : pitchAnchorsFor(*nextNote);
    if (nextAnchors.empty()) return std::nullopt;
    TransitionBridge bridge;
    bridge.startSeconds = absoluteStart + anchors.back().timeSeconds;
    bridge.endSeconds = next->startSeconds + nextAnchors.front().timeSeconds;
    bridge.startMidi = anchors.back().targetMidi;
    bridge.endMidi = nextAnchors.front().targetMidi;
    if (bridge.endSeconds <= bridge.startSeconds + 1.0e-6) return std::nullopt;
    return bridge;
}

std::optional<PianoRollComponent::TransitionBridge>
PianoRollComponent::diagnosticTransitionBridge(const juce::String& id)
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == id)
                    return transitionBridge(track, note, clip.startSeconds + note.startSeconds);
    return std::nullopt;
}

std::vector<double> PianoRollComponent::pitchLineBreaks(const TrackData& track,
                                                        const NoteData& note,
                                                        double absoluteStart)
{
    std::vector<double> breaks;
    if(const auto* projection=protocolPitchFor(track,note.id))
    {
        breaks.push_back(projection->supportStartSeconds-absoluteStart);
        for(const auto& point:projection->sampledLocalCents)breaks.push_back(point.timeSeconds);
        breaks.push_back(projection->supportEndSeconds-absoluteStart);
        std::sort(breaks.begin(),breaks.end());breaks.erase(std::unique(breaks.begin(),breaks.end()),breaks.end());return breaks;
    }
    const auto includeOffsets=[&]
    {
        if (!trackIsDiffSinger(track) || breaks.empty()) return;
        const auto from=breaks.front(),to=breaks.back();
        const auto& offsets=offsetNoteId==note.id ? offsetStroke : note.diffSingerPitchOffset;
        for (const auto& p : offsets) if (p.timeSeconds>from && p.timeSeconds<to) breaks.push_back(p.timeSeconds);
        std::sort(breaks.begin(),breaks.end());breaks.erase(std::unique(breaks.begin(),breaks.end()),breaks.end());
    };
    if (const auto* member = sharedLinesFor(track).memberFor(note.id))
    {
        // Its part of the line it shares: its own stretch and the turn after
        // it, through every corner of the line in between.  Nothing, when a
        // later note's bend has taken the whole of it over.
        if (member->drawTo <= member->drawFrom + 1.0e-9) return breaks;
        for (const auto corner : member->line->cornersBetween(member->drawFrom, member->drawTo))
            breaks.push_back(corner - absoluteStart);
        includeOffsets();return breaks;
    }
    const auto& anchors = draggedNote == note.id && dragMode == DragMode::pointPitch
            && !pitchStroke.empty()
        ? pitchStroke : pitchAnchorsFor(note);
    for (const auto& anchor : anchors) breaks.push_back(anchor.timeSeconds);
    includeOffsets();return breaks;
}

float PianoRollComponent::pitchLineMidiAt(const TrackData& track, const NoteData& note,
                                          double absoluteStart, double time,
                                          const std::vector<PitchCurveEditPoint>* nativeDensePoints)
{
    if(const auto* projection=protocolPitchFor(track,note.id);projection&&projection->continuousLocalCents)
        return projection->midiNote+projection->continuousLocalCents(time)/100.f+displayedPitchOffset(note,time);
    if (const auto* member = sharedLinesFor(track).memberFor(note.id))
        return member->line->midiAt(absoluteStart + time) + displayedPitchOffset(note,time);
    if (pointGroupGesture)
        for (const auto& draft : pointGroupGesture->curves)
            if (draft.identity.kind == PointSelection::pitch && draft.identity.noteId == note.id)
                return evaluatePitchCurve(draft.edit.pitchPoints, time) + displayedPitchOffset(note,time);
    if (draggedNote == note.id && dragMode == DragMode::pointPitch && !pitchStroke.empty())
        return evaluatePitchCurve(pitchStroke, time) + displayedPitchOffset(note,time);
    if (trackShowsAllNativeRegions(track) && note.pitchControlPoints.empty())
    {
        // Reduced automatic anchors are HANDLE scaffolding, not the audible
        // curve. Read the same linear dense authority as paint and the renderer.
        if (nativeDensePoints) return evaluatePitchCurve(*nativeDensePoints, time) + displayedPitchOffset(note,time);
        return evaluatePitchCurve(ownPitchPoints(pitchPreviewNote(note)), time) + displayedPitchOffset(note,time);
    }
    return pitchAt(pitchAnchorsFor(note), time) + displayedPitchOffset(note,time);
}

bool PianoRollComponent::pitchHandleOffered(const NoteData& note, double absoluteSeconds) const
{
    const auto* track = trackOf(note.id);
    if (track == nullptr) return true;
    if(trackUsesVoicebankSynthesis(*track)&&!trackIsDiffSinger(*track))
    {
        for(const auto& curve:connectionCurvesFor(*track,false))if(curve.active&&absoluteSeconds>=curve.fromSeconds-1.e-9&&absoluteSeconds<=curve.toSeconds+1.e-9)return false;
        return true; // OWN handles, never implicit old PBW ownership borrowing.
    }
    // As the lines stood before any drag in progress: dragging a point must
    // not make another appear or disappear.
    const auto* member = sharedLinesFor(*track, false).memberFor(note.id);
    return member == nullptr || member->owns(absoluteSeconds);
}

std::optional<double> PianoRollComponent::lastShownPitchPoint(const juce::String& noteId,
                                                              double absoluteStart)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return std::nullopt;
    std::optional<double> last;
    for (const auto& anchor : pitchAnchorsFor(*note))
    {
        const auto at = absoluteStart + anchor.timeSeconds;
        if (pitchHandleOffered(*note, at) && (!last || at > *last)) last = at;
    }
    return last;
}

std::optional<double> PianoRollComponent::firstShownPitchPoint(const juce::String& noteId,
                                                               double absoluteStart)
{
    const auto* note = findNote(noteId);
    if (note == nullptr) return std::nullopt;
    std::optional<double> first;
    for (const auto& anchor : pitchAnchorsFor(*note))
    {
        const auto at = absoluteStart + anchor.timeSeconds;
        if (pitchHandleOffered(*note, at) && (!first || at < *first)) first = at;
    }
    return first;
}

std::vector<PitchCurveEditPoint> PianoRollComponent::diagnosticOfferedPitchAnchors(
    const juce::String& id)
{
    std::vector<PitchCurveEditPoint> offered;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == id)
                {
                    const auto start = clip.startSeconds + note.startSeconds;
                    for (const auto& anchor : pitchAnchorsFor(note))
                        if ((!anchor.diffSingerRestoreSupport
                                && (!dsPitchRestoreMode || !note.diffSingerPitchReference.empty()))
                            && pitchHandleOffered(note, start + anchor.timeSeconds))
                            offered.push_back(anchor);
                    return offered;
                }
    return offered;
}

std::optional<float> PianoRollComponent::diagnosticPitchLineAt(const juce::String& id,
                                                               double absoluteSeconds)
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == id)
                {
                    const auto start = clip.startSeconds + note.startSeconds;
                    return pitchLineMidiAt(track, note, start, absoluteSeconds - start);
                }
    return std::nullopt;
}

std::optional<std::pair<double, double>> PianoRollComponent::diagnosticPitchLineSpan(
    const juce::String& id)
{
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == id)
                {
                    const auto start = clip.startSeconds + note.startSeconds;
                    const auto breaks = pitchLineBreaks(track, note, start);
                    if (breaks.empty()) return std::nullopt;
                    return std::make_pair(start + breaks.front(), start + breaks.back());
                }
    return std::nullopt;
}

std::vector<PitchCurveEditPoint>& PianoRollComponent::pitchAnchorsFor(const NoteData& note)
{
    if(pitchPlanPreview&&std::any_of(pitchPendingEdits.begin(),pitchPendingEdits.end(),[&](const auto& e){return e.kind==ProjectModel::PointCurveEdit::pitch&&e.noteId==note.id;}))
    {
        const auto key=note.id.toStdString();if(const auto found=pitchPreviewAnchorCache.find(key);found!=pitchPreviewAnchorCache.end())return found->second;
        return pitchPreviewAnchorCache.emplace(key,ownPitchPoints(pitchPreviewNote(note))).first->second;
    }

    if(pointGroupGesture)for(auto& d:pointGroupGesture->curves)
        if(d.identity.kind==PointSelection::pitch&&d.identity.noteId==note.id)return d.edit.pitchPoints;
    const auto key = note.id.toStdString();
    if (note.nativeUnpitched || (note.nativeSparsePitchControls && note.pitchControlPoints.empty())) return pitchAnchorCache[key] = {};
    if (const auto found = pitchAnchorCache.find(key); found != pitchAnchorCache.end())
        return found->second;

    if (const auto* nativeTrack = trackOf(note.id); nativeTrack && trackShowsAllNativeRegions(*nativeTrack))
        if (const auto* track = trackOf(note.id))
            if (const auto* member = sharedLinesFor(*track).memberFor(note.id);
                member && !member->nativeBoundaryAnchors.empty())
                return pitchAnchorCache.emplace(key, member->nativeBoundaryAnchors).first->second;

    std::vector<PitchCurveEditPoint> anchors;
    // Whether this curve is the automatic one derived from the contour or a
    // set of points the user placed.  Only the automatic one may be reshaped
    // below; an explicit curve is the user's and is left alone.
    const auto derived = note.pitchControlPoints.empty();
    if (!derived)
    {
        anchors = note.pitchControlPoints;
        // Older saved DS projects may still expose every prediction frame.
        // This changes handle metadata in the cache only; audio data stays exact.
        const auto* track = trackOf(note.id);
        if (track != nullptr && trackIsDiffSinger(*track))
            anchors = compactDiffSingerPitchHandles(std::move(anchors),
                note.diffSingerPitchReference, note.durationSeconds);
    }
    else
    {
        std::vector<PitchCurveEditPoint> dense;
        dense.reserve(note.contour.size() + 2);
        for (const auto& point : note.contour)
            if (point.voiced)
                dense.push_back({ juce::jlimit(0.0, note.durationSeconds, point.timeSeconds),
                                  juce::jlimit(0.0f, 127.0f,
                                      note.midiNote + renderedPitchCents(note, point) / 100.0f) });
        if (dense.empty())
            dense = { { 0.0, note.midiNote }, { note.durationSeconds, note.midiNote } };
        std::stable_sort(dense.begin(), dense.end(), [](const auto& left, const auto& right)
        {
            return left.timeSeconds < right.timeSeconds;
        });
        const auto startPitch = pitchAt(dense, 0.0);
        const auto endPitch = pitchAt(dense, note.durationSeconds);
        if (dense.front().timeSeconds > 1.0e-7) dense.insert(dense.begin(), { 0.0, startPitch });
        else dense.front() = { 0.0, startPitch };
        if (dense.back().timeSeconds < note.durationSeconds - 1.0e-7)
            dense.push_back({ note.durationSeconds, endPitch });
        else dense.back() = { note.durationSeconds, endPitch };

        // Short UTAU notes have very little horizontal room.  Limit their handle
        // count by duration (roughly one interior point per 45 ms), and never show
        // more than 16.  A tolerance search keeps the most significant bends.
        const auto maximumAnchors = static_cast<std::size_t>(juce::jlimit(2, 16,
            static_cast<int>(std::floor(note.durationSeconds / 0.045)) + 2));
        anchors = simplifyPitchAnchors(dense, 0.025f); // 2.5 cents initially.
        if (anchors.size() > maximumAnchors)
        {
            auto low = 0.025f;
            auto high = 0.25f;
            while (simplifyPitchAnchors(dense, high).size() > maximumAnchors && high < 64.0f)
                high *= 2.0f;
            for (int iteration = 0; iteration < 16; ++iteration)
            {
                const auto middle = (low + high) * 0.5f;
                auto candidate = simplifyPitchAnchors(dense, middle);
                if (candidate.size() > maximumAnchors) low = middle;
                else
                {
                    high = middle;
                    anchors = std::move(candidate);
                }
            }
        }
    }

    // Where two notes meet, the automatic curve pulls its endpoints inwards to
    // leave room for the transition between them.  This must not be re-applied
    // to a stored curve: the stored endpoints already carry that inset from
    // when they were first derived, and forcing their time again pinned them
    // in place -- the endpoint at a boundary could be dragged but sprang back
    // the moment the anchors were rebuilt.
    const auto positioned = positionedUtauNotesFor(note.id);
    const auto current = std::find_if(positioned.begin(), positioned.end(),
        [&](const auto& value) { return value.id == note.id; });
    const auto* noteTrack = trackOf(note.id);
    const auto* sharedMember = noteTrack != nullptr
        ? sharedLinesFor(*noteTrack).memberFor(note.id) : nullptr;
    if (derived && anchors.size() >= 2 && current != positioned.end()
        && !(noteTrack && trackShowsAllNativeRegions(*noteTrack)))
    {
        if (current != positioned.begin()
            && !(sharedMember != nullptr && sharedMember->joinsPrevious))
        {
            const auto& previous = *std::prev(current);
            if (formsAdjacentPitchBoundary(previous, *current))
            {
                const auto inset = automaticPitchTransitionInset(
                    previous.endSeconds - previous.startSeconds, note.durationSeconds);
                anchors.erase(std::remove_if(std::next(anchors.begin()), anchors.end(),
                    [&](const auto& point) { return point.timeSeconds <= inset; }), anchors.end());
                anchors.front().timeSeconds = inset;
            }
        }
        if (std::next(current) != positioned.end()
            && !(sharedMember != nullptr && sharedMember->joinsNext))
        {
            const auto& next = *std::next(current);
            if (formsAdjacentPitchBoundary(*current, next))
            {
                const auto inset = automaticPitchTransitionInset(note.durationSeconds,
                    next.endSeconds - next.startSeconds);
                const auto tailTime = note.durationSeconds - inset;
                anchors.erase(std::remove_if(anchors.begin(), std::prev(anchors.end()),
                    [&](const auto& point) { return point.timeSeconds >= tailTime; }),
                    std::prev(anchors.end()));
                anchors.back().timeSeconds = tailTime;
            }
        }
    }
    return pitchAnchorCache.emplace(key, std::move(anchors)).first->second;
}

void PianoRollComponent::diagnosticRefresh()
{
    refreshFromModel();
}

void PianoRollComponent::refreshFromModel()
{
    joinedVibratoCache.clear();
    if (nativeOverlapSelector && nativeOverlapSelector->revision!=model.revisionNumber()) closeNativeOverlapSelector();
    pitchAnchorCache.clear();connectionPitchCache.clear();protocolPitchCache.clear();clearPitchPlanPreview();
    sharedLineCache.clear(); nativeEnvelopeCache.clear();
    draggedSharedKey = 0;
    rebuildLayout();
}

std::vector<PianoRollComponent::DiagnosticNote>
PianoRollComponent::diagnosticNotes() const
{
    std::vector<DiagnosticNote> result;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                DiagnosticNote entry;
                entry.id = note.id;
                entry.label = note.label;
                entry.start = clip.startSeconds + note.startSeconds;
                entry.end = entry.start + note.durationSeconds;
                entry.utau = track.pitchAlgorithm == PitchAlgorithm::utau;
                entry.soundingStart = entry.start;
                entry.soundingEnd = entry.end;
                if (const auto span = utauSoundSpans.find(note.id.toStdString());
                    span != utauSoundSpans.end())
                {
                    entry.soundingStart = span->second.first;
                    entry.soundingEnd = span->second.second;
                }
                entry.envelope = amplitudeEnvelopeFor(note, entry.start);
                result.push_back(std::move(entry));
            }
    return result;
}

void PianoRollComponent::selectAllNotes()
{
    clearPointSelection();
    selectedNotes.clear();
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose && !sourceEditMode) continue;
        if (!sourceEditMode && focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto& clip : track.clips)
        {
            // Selection follows the same region scope as drawing and hit testing.
            if (!clipIsVisible(track, clip)) continue;
            for (const auto& note : clip.notes) selectedNotes.insert(note.id.toStdString());
        }
    }
    selectedNote = selectedNotes.empty() ? juce::String()
        : juce::String::fromUTF8(selectedNotes.begin()->c_str());
    if (onNoteSelected) onNoteSelected(selectedNote);
    repaint();
}

void PianoRollComponent::clearNoteSelection()
{
    clearPointSelection();
    selectedNote.clear();
    selectedNotes.clear();
    if (onNoteSelected) onNoteSelected({});
    repaint();
}

void PianoRollComponent::setSelectedNoteIds(const std::vector<juce::String>& noteIds)
{
    if (nativeOverlapSelector && (noteIds.empty() || noteIds.front()!=nativeOverlapSelector->targetNoteId)) closeNativeOverlapSelector();
    clearPointSelection();
    cvvcSelectedNote.clear(); cvvcSelectedPart = 0;
    selectedNotes.clear();
    for (const auto& id : noteIds)
        if (id.isNotEmpty()) selectedNotes.insert(id.toStdString());
    selectedNote = noteIds.empty() ? juce::String{} : noteIds.front();
    if (onNoteSelected) onNoteSelected(selectedNote);
    repaint();
}

std::vector<juce::String> PianoRollComponent::selectedNoteIds() const
{
    std::vector<juce::String> result;
    result.reserve(selectedNotes.size() + (selectedNote.isNotEmpty() ? 1u : 0u));
    for (const auto& id : selectedNotes)
        result.push_back(juce::String::fromUTF8(id.c_str()));
    if (result.empty() && selectedNote.isNotEmpty()) result.push_back(selectedNote);
    return result;
}

bool PianoRollComponent::canEditSelectedNativeVibrato() const
{
    const auto ids = selectedNoteIds();
    return !sourceEditMode && !ids.empty()
        && std::all_of(ids.begin(), ids.end(), [this](const auto& id)
            { return findNote(id) != nullptr && nativeNoteTimingEnabled(id); });
}

void PianoRollComponent::showSelectedNativeVibratoDialog()
{
    if (!canEditSelectedNativeVibrato()) return;
    const auto ids = selectedNoteIds();
    showVibratoDialog(selectedNote.isNotEmpty() ? selectedNote : ids.front());
}

double PianoRollComponent::gridQuarterNotes() const
{
    auto text = snapshot.gridDivision.trim().toLowerCase();
    auto dotted = text.endsWithChar('.');
    auto triplet = text.endsWithChar('t');
    if (dotted || triplet) text = text.dropLastCharacters(1);
    const auto slash = text.indexOfChar('/');
    const auto denominator = slash >= 0 ? text.substring(slash + 1).getIntValue() : 16;
    auto quarters = 4.0 / static_cast<double>(std::max(1, denominator));
    if (dotted) quarters *= 1.5;
    if (triplet) quarters *= 2.0 / 3.0;
    return std::max(1.0 / 256.0, quarters);
}

double PianoRollComponent::gridSecondsAt(double seconds) const
{
    const auto quarter = snapshot.quarterPositionForSeconds(seconds);
    return std::max(0.001,
        snapshot.secondsForQuarterPosition(quarter + gridQuarterNotes())
            - snapshot.secondsForQuarterPosition(quarter));
}

double PianoRollComponent::drawnLengthFor(double draggedSeconds,
                                          double unitSeconds,
                                          double availableSeconds)
{
    const auto unit = std::max(1.0e-6, unitSeconds);
    // The unit the pointer is standing in counts, so a press that never moves
    // still makes one -- floor of the distance, plus the one being pointed at.
    const auto units = std::max(1.0, std::floor(draggedSeconds / unit) + 1.0);
    return std::min(units * unit, availableSeconds);
}

std::vector<int> PianoRollComponent::drawLengthDivisions()
{
    std::vector<int> divisions;
    for (auto division = 128; division >= 4; division /= 2)
        divisions.push_back(division);
    return divisions;
}

void PianoRollComponent::setDrawLengthDivision(int division)
{
    const auto offered = drawLengthDivisions();
    if (std::find(offered.begin(), offered.end(), division) == offered.end()) return;
    drawLengthDivisionValue = division;
}

double PianoRollComponent::drawUnitSecondsAt(double seconds) const
{
    // quarterBarSecondsAt is a quarter of a bar, so a bar is four of them.
    // Said this way rather than with a bar length of its own, so the two
    // cannot disagree when the signature or the tempo changes.
    const auto bar = quarterBarSecondsAt(seconds) * 4.0;
    return std::max(1.0e-6, bar / static_cast<double>(std::max(1, drawLengthDivisionValue)));
}

bool PianoRollComponent::canDrawNotesOnFocusedTrack() const
{
    for (const auto& track : snapshot.tracks)
        if (track.id == focusedTrack)
            return track.compose && !track.accompaniment;
    // No track selected: fall back to whoever owns the focused clip.
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == focusedClip)
                return track.compose && !track.accompaniment;
    return false;
}

juce::String PianoRollComponent::clipForDrawingAt(double seconds)
{
    const auto pick = [&](const ProjectData& data) -> juce::String
    {
        const ClipData* containing = nullptr;
        const ClipData* focused = nullptr;
        const ClipData* growable = nullptr;
        for (const auto& track : data.tracks)
        {
            if (!track.compose) continue;
            const auto mine = focusedTrack.isEmpty() || track.id == focusedTrack;
            for (const auto& clip : track.clips)
            {
                const auto contains = seconds >= clip.startSeconds
                    && seconds <= clip.startSeconds + clip.durationSeconds;
                if (clip.id == focusedClip)
                {
                    focused = &clip;
                    if (sourceEditMode || contains) return clip.id;
                }
                if (!mine || !clipIsVisible(track, clip)) continue;
                if (contains) containing = &clip;
                // A clip with no recording behind it is a span of the
                // timeline and stretches to fit, so one that begins before
                // here can still take the note.  Reaching for the latest such
                // clip rather than making a new one: drawing past the end of
                // the seed clip used to add a second clip every time, and
                // then find neither of them.
                if (clip.sourceFile == juce::File() && clip.startSeconds <= seconds
                    && (growable == nullptr || clip.startSeconds > growable->startSeconds))
                    growable = &clip;
            }
        }
        if (containing != nullptr) return containing->id;
        if (focused != nullptr) return focused->id;
        if (growable != nullptr) return growable->id;
        return {};
    };
    if (const auto found = pick(snapshot); found.isNotEmpty()) return found;

    // Nothing to draw into.  A melodic track starts with no clips at all and
    // no other part of the interface makes one, so the first stroke on a new
    // track makes it -- from the start of the timeline, since a clip that
    // began under the pointer would leave the bars before it unreachable.
    // One bar is only a seed: a composed clip grows to fit what is drawn.
    auto empty = false;
    for (const auto& track : snapshot.tracks)
        if (track.id == focusedTrack && track.compose && track.clips.empty())
            empty = true;
    if (!empty) return {};

    const auto seed = std::max(0.05, quarterBarSecondsAt(0.0) * 4.0);
    if (model.addClip(focusedTrack, 0.0, seed).isEmpty()) return {};
    // Asked of the model, not of this view: the copy here only catches up on
    // a message, and the clip was made a moment ago.
    return pick(model.snapshot());
}

void PianoRollComponent::beginDrawingNote(const juce::MouseEvent& event)
{
    const auto rawSeconds = std::max(0.0,
        static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond);
    const auto cellStart = snapDownToGrid(rawSeconds);

    const auto clipId = clipForDrawingAt(cellStart);
    if (clipId.isEmpty()) return;
    // From the model, so a clip made a moment ago is visible here.
    const auto data = model.snapshot();
    const ClipData* destination = nullptr;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId) destination = &clip;
    if (destination == nullptr) return;
    // A source-backed clip is not an infinitely growable composition region.
    // Clicking before/after its actual source window must not silently snap a
    // new note into that recording or author a note in an uncovered audio hole.
    auto sourceWindowEnd = std::numeric_limits<double>::max();
    const auto sources = clipSourceRegions(*destination);
    if (std::any_of(sources.begin(), sources.end(), [](const auto* part) { return part->sourceFile != juce::File(); }))
    {
        sourceWindowEnd = -1.0;
        for (const auto* part : sources)
        {
            if (part->sourceFile == juce::File()) continue;
            const auto begin = part->startSeconds + part->audioStartSeconds
                + (destination->parts.empty() ? 0.0 : destination->startSeconds);
            const auto end = begin + part->audioLength();
            if (cellStart >= begin && cellStart < end) sourceWindowEnd = std::max(sourceWindowEnd, end);
        }
        if (sourceWindowEnd < 0.0) return;
    }

    std::vector<ProjectModel::NoteSpan> occupied;
    occupied.reserve(destination->notes.size());
    for (const auto& note : destination->notes)
        occupied.push_back({ note.startSeconds, note.durationSeconds });

    // The cell clicked, or -- if something is already sounding there -- the
    // moment that note ends.  The model repeats this search under its own lock
    // when the button comes up; this copy is only what the preview draws.
    const auto localStart = ProjectModel::firstFreeStartFrom(
        cellStart - destination->startSeconds, occupied);
    // A clip with no recording behind it stretches to fit, so the only limit
    // is the next note; one backed by audio ends where its audio does.
    auto available = destination->sourceFile == juce::File()
        ? std::numeric_limits<double>::max()
        : std::min(destination->durationSeconds,
            destination->audioStartSeconds + destination->audioLength()) - localStart;
    available = std::min(available, sourceWindowEnd - destination->startSeconds - localStart);
    for (const auto& span : occupied)
        if (span.startSeconds > localStart)
            available = std::min(available, span.startSeconds - localStart);
    // The model's own floor for a note.  Below it nothing can be made, so
    // nothing is started and no rectangle appears promising otherwise.
    if (available < 0.01) return;

    drawClipId = destination->id;
    drawStartSeconds = destination->startSeconds + localStart;
    drawAvailableSeconds = available;
    drawMidi = juce::jlimit(0.0f, 127.0f, std::round(yToMidi(event.position.y)));
    drawUnitSeconds = drawUnitSecondsAt(drawStartSeconds);
    drawLengthSeconds = drawnLengthFor(0.0, drawUnitSeconds, drawAvailableSeconds);
    dragMode = DragMode::drawNewNote;
    repaint();
}

double PianoRollComponent::snapDownToGrid(double seconds) const
{
    const auto step = gridQuarterNotes();
    const auto quarter = snapshot.quarterPositionForSeconds(seconds);
    return std::max(0.0, snapshot.secondsForQuarterPosition(
        std::floor(quarter / step) * step));
}

double PianoRollComponent::snapToGrid(double seconds) const
{
    const auto step = gridQuarterNotes();
    const auto quarter = snapshot.quarterPositionForSeconds(seconds);
    return std::max(0.0, snapshot.secondsForQuarterPosition(
        std::round(quarter / step) * step));
}

int PianoRollComponent::pixelForSeconds(double seconds) const
{
    return static_cast<int>(std::round(timeToX(seconds)));
}

double PianoRollComponent::secondsForPixel(int pixel) const
{
    return std::max(0.0, static_cast<double>(pixel - 58) / pixelsPerSecond);
}

float PianoRollComponent::timeToX(double seconds) const
{
    return 58.0f + static_cast<float>(seconds) * pixelsPerSecond;
}

float PianoRollComponent::midiToY(float midi) const
{
    return static_cast<float>(highestMidi) * rowHeight - midi * rowHeight;
}

float PianoRollComponent::yToMidi(float y) const
{
    return static_cast<float>(highestMidi) - y / rowHeight;
}

void PianoRollComponent::rebuildLayout()
{
    nativePitchVisibilityCache.clear(); nativePitchAuthoredCache.clear();
    snapshot = model.snapshot();
    expandProjectClipParts(snapshot, true);
    dsPitchOffsetNoteIds.clear();
    for (const auto& track : snapshot.tracks) if (trackIsDiffSinger(track))
        for (const auto& clip : track.clips) for (const auto& note : clip.notes) dsPitchOffsetNoteIds.insert(note.id);
    if (!offsetNoteId.isEmpty() && model.revisionNumber()!=offsetRevision) cancelDiffSingerPitchOffsetGesture();
    if (dsPitchOffsetMode && !diffSingerPitchOffsetAvailable()) setDiffSingerPitchOffsetMode(false);
    if (dsPitchRestoreMode && !diffSingerPitchRestoreAvailable())
        setDiffSingerPitchRestoreMode(false);
    rebuildUtauSoundSpans();
    requestCvvcLayouts();
    applyCvvcBoundaryPreview();
    if (!pendingSplitRestate.empty())
    {
        // Restate before anything is drawn from these spans, so the lines do
        // not flicker through one frame of their old fractions.
        restateStandingSplits();
        snapshot = model.snapshot();
        expandProjectClipParts(snapshot, true);
        rebuildUtauSoundSpans();
    }
    pruneHiddenNoteSelection();
    std::unordered_map<std::string, std::unique_ptr<juce::AudioThumbnail>> next;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
        {
            const auto key = clip.sourceFile.getFullPathName().toStdString();
            if (next.contains(key)) continue;
            if (const auto found = thumbnails.find(key); found != thumbnails.end())
            {
                next.emplace(key, std::move(found->second));
                continue;
            }
            auto thumbnail = std::make_unique<juce::AudioThumbnail>(nativeWaveformSamplesPerPeak, formats, thumbnailCache);
            thumbnail->addChangeListener(this);
            if (clip.sourceFile.existsAsFile())
                thumbnail->setSource(new juce::FileInputSource(clip.sourceFile));
            next.emplace(key, std::move(thumbnail));
        }
    thumbnails = std::move(next);
    requestConsonantVoicing();
    rebuildNativeWaveformHashes();
    updateCanvasSize();
    repaint();
}

void PianoRollComponent::rebuildUtauSoundSpans()
{
    struct DisplayNote
    {
        juce::String id;
        double start = 0.0;
        double end = 0.0;
        bool spliced = false;
        std::optional<backend::UtauSampleTiming> timing;
    };

    auto previousSpans = std::move(utauSoundSpans);
    utauSoundSpans.clear();
    utauNeighbours.clear();
    diffSingerPhonemes.clear(); diffSingerNoteIds.clear();
    if (sourceEditMode) return;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose || !trackUsesVoicebankSynthesis(track))
            continue;
        if (trackIsDiffSinger(track))
        {
            std::unordered_map<std::string,const UtauNoteWaveform*> rendered;
            if (utauWaveforms) for (const auto& wave:*utauWaveforms) rendered[wave.noteId.toStdString()]=&wave;
            for (const auto& clip:track.clips) for (const auto& note:clip.notes)
            {
                const auto id=note.id.toStdString(); diffSingerNoteIds.insert(id);
                const auto found=rendered.find(id);
                if (found==rendered.end() || found->second->audioHash!=AudioEngine::utauNoteAudioHash(note)
                    || found->second->phonemes.empty()) continue;
                const auto& wave=*found->second;
                const auto start=clip.startSeconds+note.startSeconds;
                utauSoundSpans[id]={start+wave.phonemes.front().startSeconds,start+wave.phonemes.back().endSeconds};
                diffSingerPhonemes[id]=wave.phonemes;
            }
            continue;
        }
        if (!track.voicebankDirectory.isDirectory()) continue;
        if(track.chineseCvvc&&track.utauMode==UtauMode::classic)
        {
            const auto context=AudioEngine::utauPhonemeMixContextHash(track);
            for(const auto& clip:track.clips)for(const auto& note:clip.notes)
            {
                const auto cache=backend::readCvvcPhonemeCache(note.cvvcPhonemeCache);
                if(cache.layout!=AudioEngine::utauNotePhonemeHash(note)||cache.context!=context||cache.spans.empty())continue;
                const auto start=clip.startSeconds+note.startSeconds;
                diffSingerPhonemes[note.id.toStdString()]=cache.spans;
                auto from=start;
                for(const auto& p:cache.spans)from=std::min(from,start+p.startSeconds-p.leadInSeconds);
                utauSoundSpans[note.id.toStdString()]={from,start+cache.spans.back().endSeconds};
            }
        }
        if (!voicebankReadFor(track))
        {
            // Asking sampleTiming now would read the bank here, on the
            // message thread, which is the wait this exists to avoid.
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (const auto kept = previousSpans.find(note.id.toStdString());
                        kept != previousSpans.end())
                        utauSoundSpans.insert(*kept);
            continue;
        }
        std::vector<DisplayNote> notes;
        // sampleTiming stats the voicebank folder, builds a cache key and
        // scans every oto entry by name, so asking it once per note made an
        // edit cost thousands of string comparisons.  Notes repeat their
        // lyric and their pitch, so ask once per distinct answer.
        std::map<std::tuple<juce::String, float, int>,
                 std::optional<backend::UtauSampleTiming>> timings;
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                const auto start = clip.startSeconds + note.startSeconds;
                DisplayNote display { note.id, start, start + note.durationSeconds,
                                      note.utauSplice };
                const auto consonantVelocity =
                    note.utauConsonantVelocity != inheritedUtauConsonantVelocity
                    ? note.utauConsonantVelocity : track.utauConsonantVelocity;
                const auto timingKey = std::make_tuple(note.label, note.midiNote,
                                                       consonantVelocity);
                // A note with its own oto is its own case: this cache is keyed
                // by what an entry is looked up by, which says nothing of it.
                const auto cached = note.utauOto.enabled ? timings.end()
                                                         : timings.find(timingKey);
                display.timing = note.utauOto.enabled
                    ? backend::UtauRenderer::sampleTiming(
                          track.voicebankDirectory, note.label, note.midiNote,
                          consonantVelocity, utauModeUsesRegions(track.utauMode),
                          track.utauMode == UtauMode::mou, note.ownOto())
                    : cached != timings.end()
                    ? cached->second
                    : timings.emplace(timingKey, backend::UtauRenderer::sampleTiming(
                          track.voicebankDirectory, note.label, note.midiNote,
                          consonantVelocity, utauModeUsesRegions(track.utauMode),
                          track.utauMode == UtauMode::mou)).first->second;
                if (display.timing)
                {
                    if (note.utauPreutteranceOverrideEnabled)
                        display.timing->preutteranceSeconds =
                            std::max(0.0, note.utauPreutteranceSeconds);
                    if (note.utauOverlapOverrideEnabled)
                        display.timing->overlapSeconds = note.utauOverlapSeconds;
                }
                notes.push_back(std::move(display));
            }
        std::stable_sort(notes.begin(), notes.end(), [](const auto& left, const auto& right)
        {
            if (std::abs(left.start - right.start) > 1.0e-9) return left.start < right.start;
            return left.end < right.end;
        });
        for (std::size_t index = 0; index < notes.size(); ++index)
        {
            const auto& note = notes[index];
            if (!note.timing) continue;
            const auto leadIn = std::min(note.timing->preutteranceSeconds,
                                         std::max(0.0, note.start));
            auto soundingEnd = note.end;
            if (index + 1 < notes.size())
            {
                const auto& next = notes[index + 1];
                // Read through the renderer's own rules rather than
                // spelled out again here: what is drawn and what is mixed have
                // to say the same thing about where a note stops, and a second
                // copy of a rule is how they came to disagree.
                const auto nextSoundStart = next.timing
                    ? next.start - std::min(next.timing->preutteranceSeconds,
                                            std::max(0.0, next.start))
                    : next.start;
                if (next.timing && backend::UtauRenderer::crossfadesInto(
                        nextSoundStart, note.end))
                {
                    // Splicing is not an exception here either: it changes
                    // the shape of the crossfade and not where it happens.
                    soundingEnd = backend::UtauRenderer::crossfadeEnd(
                        nextSoundStart, next.timing->overlapSeconds, next.end);
                }
            }
            const auto soundingStart = note.start - leadIn;
            soundingEnd = std::max(soundingStart + 0.001, soundingEnd);
            auto& neighbours = utauNeighbours[note.id.toStdString()];
            if (index > 0) neighbours.previous = notes[index - 1].id.toStdString();
            if (index + 1 < notes.size())
            {
                neighbours.next = notes[index + 1].id.toStdString();
                neighbours.nextSpliced = notes[index + 1].spliced;
            }
            utauSoundSpans[note.id.toStdString()] = { soundingStart, soundingEnd };
        }
        const auto mixContext = track.chineseCvvc ? AudioEngine::utauPhonemeMixContextHash(track) : 0;
        if (track.chineseCvvc && track.utauMode == UtauMode::classic && utauWaveforms)
            for (const auto& clip : track.clips) for (const auto& note : clip.notes)
                for (const auto& wave : *utauWaveforms)
                    if (wave.noteId == note.id && (wave.phonemeLayoutHash != 0
                            ? wave.phonemeLayoutHash == AudioEngine::utauNotePhonemeHash(note)
                            : wave.audioHash == AudioEngine::utauNoteAudioHash(note))
                        && !wave.phonemes.empty()
                        && (wave.phonemeMixContextHash==0 || wave.phonemeMixContextHash==mixContext))
                    {
                        // Shared read-only phoneme drawing/hover, without DiffSinger editing semantics.
                        diffSingerPhonemes[note.id.toStdString()] = wave.phonemes;
                        if (wave.phonemeMixContextHash != mixContext)
                            for (auto& span : diffSingerPhonemes[note.id.toStdString()]) span.mixEnvelope = {};
                        const auto start = clip.startSeconds + note.startSeconds;
                        utauSoundSpans[note.id.toStdString()] = {
                            start - wave.leadInSeconds, start + wave.phonemes.back().endSeconds };
                        break;
                    }
        // Both saved layouts and live layouts define the same sounding range;
        // a plain CV OTO lookup above must not overwrite a restored CVVC span.
        if(track.chineseCvvc&&track.utauMode==UtauMode::classic)
            for(const auto& clip:track.clips)for(const auto& note:clip.notes)
                if(const auto found=diffSingerPhonemes.find(note.id.toStdString());found!=diffSingerPhonemes.end()&&!found->second.empty())
                {
                    const auto start=clip.startSeconds+note.startSeconds;auto from=start;
                    for(const auto& p:found->second)from=std::min(from,start+p.startSeconds-p.leadInSeconds);
                    utauSoundSpans[note.id.toStdString()]={from,start+found->second.back().endSeconds};
                }
    }
}

void PianoRollComponent::requestCvvcLayouts()
{
    // Keep the committed head where the preview put it while the layout-only
    // background job catches up. Any subsequent edit invalidates this bridge.
    if (pendingCvvcHead.isNotEmpty() && pendingCvvcHeadRevision == model.revisionNumber()
        && !diffSingerPhonemes.contains(pendingCvvcHead.toStdString()))
        if (auto span = utauSoundSpans.find(pendingCvvcHead.toStdString()); span != utauSoundSpans.end())
            span->second.first = pendingCvvcHeadStart;
    // The desktop uses background indexing. Offline fixtures can inject their
    // own metadata without starting unrelated asynchronous work.
    if(!readsVoicebankInBackground)return;
    for(const auto& track:snapshot.tracks)
    {
        if(!track.chineseCvvc||track.utauMode!=UtauMode::classic||!trackUsesVoicebankSynthesis(track)
            ||trackIsDiffSinger(track)||!track.voicebankDirectory.isDirectory())continue;
        const auto context=AudioEngine::utauPhonemeMixContextHash(track);
        auto missing=false;
        for(const auto& clip:track.clips)for(const auto& note:clip.notes)
            if(!backend::isRestLyric(note.label)&&!diffSingerPhonemes.contains(note.id.toStdString()))missing=true;
        const auto key=context^static_cast<std::uint64_t>(track.id.hashCode64());
        if(!missing||!cvvcLayoutRequests.insert(key).second)continue;
        auto data=snapshot;data.tracks={track};
        const auto revision=model.revisionNumber();
        juce::Component::SafePointer<PianoRollComponent> safe(this);
        cvvcLayoutPool.addJob([data=std::move(data),context,key,revision,safe]()
        {
            std::vector<std::pair<juce::String,juce::String>> caches;
            const auto& track=data.tracks.front();
            for(const auto& clip:track.clips)
            {
                backend::UtauRenderRequest request;
                request.voicebankDirectory=track.voicebankDirectory;request.chineseCvvc=true;request.bpm=data.bpm;
                request.notes=AudioEngine::utauRequestNotesForClip(data,clip.id);
                const auto plan=backend::planChineseCvvc(request);
                for(std::size_t i=0;i<clip.notes.size()&&i<plan.spans.size();++i)
                    if(!plan.spans[i].empty())caches.emplace_back(clip.notes[i].id,backend::cvvcPhonemeCacheText(
                        {AudioEngine::utauNotePhonemeHash(clip.notes[i]),context,plan.spans[i]}));
            }
            juce::MessageManager::callAsync([safe,key,revision,caches=std::move(caches)]()
            {
                if(!safe)return;
                if(safe->model.revisionNumber()!=revision)
                {safe->cvvcLayoutRequests.erase(key);safe->refreshFromModel();return;}
                if(safe->model.rememberCvvcPhonemes(caches,revision))
                {safe->cvvcLayoutRequests.erase(key);safe->refreshFromModel();}
            });
        });
    }
}

bool PianoRollComponent::voicebankReadFor(const TrackData& track) const
{
    if (trackIsDiffSinger(track) || !readsVoicebankInBackground) return true;
    std::function<void()> whenReady;
    if (!awaitingVoicebank)
        whenReady = [safe = juce::Component::SafePointer<PianoRollComponent>(
                         const_cast<PianoRollComponent*>(this))]
        {
            if (safe == nullptr) return;
            safe->awaitingVoicebank = false;
            safe->rebuildLayout();
        };
    const auto ready = backend::UtauRenderer::voicebankIndexReady(
        track.voicebankDirectory, utauModeUsesRegions(track.utauMode),
        track.utauMode == UtauMode::mou, std::move(whenReady));
    if (!ready) awaitingVoicebank = true;
    return ready;
}

void PianoRollComponent::updateCanvasSize()
{
    auto duration = snapshot.durationSeconds();
    if (sourceEditMode)
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                if (clip.id == focusedClip)
                    if (const auto found = thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
                        found != thumbnails.end())
                        duration = std::max(duration, found->second->getTotalLength());
    // Bounded because the width is duration times zoom: a long project at the
    // finest zoom would otherwise ask for a component wider than the drawing
    // coordinates stay exact in.
    const auto width = juce::jlimit(470.0, 3.2e7,
                                    static_cast<double>(timeToX(duration)) + 400.0);
    setSize(static_cast<int>(width),
            (highestMidi - lowestMidi + 1) * static_cast<int>(rowHeight));
    // Hit regions depend on exactly what this function does -- the note list,
    // the sounding spans and the time and pitch scales -- so they are rebuilt
    // here rather than at the top of every paint.
    rebuildNoteHits();
}

void PianoRollComponent::setShowUtauWaveforms(bool enabled)
{
    if (showUtauWaveforms == enabled) return;
    showUtauWaveforms = enabled;
    repaint();
}

void PianoRollComponent::setShowNativeRenderedWaveforms(bool show)
{
    if(show==showNativeRenderedWaveforms)return;
    showNativeRenderedWaveforms=show;rebuildNativeWaveformHashes();repaint();
}
void PianoRollComponent::rebuildNativeWaveformHashes()
{
    nativeClipHashes.clear();if(!showNativeRenderedWaveforms)return;
    for(const auto& track:snapshot.tracks)
        if(track.compose&&!track.accompaniment&&!trackUsesVoicebankSynthesis(track))
            for(const auto& clip:track.clips)
                nativeClipHashes.emplace(&clip,AudioEngine::nativeClipWaveformHash(clip,track));
}

void PianoRollComponent::setNativeClipWaveforms(std::shared_ptr<const std::vector<NativeRenderedWaveform>> waveforms)
{
    if(nativeWaveforms==waveforms)return;
    nativeWaveforms=std::move(waveforms);nativePeaksByHash.clear();
    if(nativeWaveforms)for(const auto& waveform:*nativeWaveforms)
        if(waveform.peaks)nativePeaksByHash.try_emplace(waveform.audioHash,waveform.peaks.get());
    repaint();
}
bool PianoRollComponent::nativeNoteTimingEnabled(const juce::String& id) const
{
    if(sourceEditMode) return false;
    for(const auto& track:snapshot.tracks) if(trackShowsAllNativeRegions(track))
        for(const auto& clip:track.clips) for(const auto& note:clip.notes)
            if(note.id==id) return true;
    return false;
}

const NativeRenderedPeaks* PianoRollComponent::nativeRenderedPeaksFor(const ClipData& clip) const
{
    if(!showNativeRenderedWaveforms)return nullptr;
    const auto cached=nativeClipHashes.find(&clip);
    std::uint64_t hash=0;
    if(cached!=nativeClipHashes.end()) hash=cached->second;
    else if(dragMode==DragMode::moveNativeNote || dragMode==DragMode::resizeNativeLeft || dragMode==DragMode::resizeNativeRight)
    {
        // Preview copies have different addresses. Resolve the actual clock
        // hash, hiding stale rendered audio only for the edited source.
        for(const auto& track:snapshot.tracks) for(const auto& candidate:track.clips)
            if(&candidate==&clip) hash=AudioEngine::nativeClipWaveformHash(clip,track);
    }
    else return nullptr;
    const auto found=nativePeaksByHash.find(hash);
    return found==nativePeaksByHash.end()?nullptr:found->second;
}

void PianoRollComponent::setUtauNoteWaveforms(
    std::shared_ptr<const std::vector<UtauNoteWaveform>> waveforms)
{
    // Same pointer, same peaks: a repaint per timer tick would otherwise cost
    // one for every note in the song, forever.
    if (utauWaveforms == waveforms) return;
    utauWaveforms = std::move(waveforms);
    const auto revision=model.revisionNumber();
    snapshot=model.snapshot();expandProjectClipParts(snapshot,true);
    std::vector<std::pair<juce::String,juce::String>> caches;
    if(utauWaveforms)for(const auto& track:snapshot.tracks)
        if(track.chineseCvvc&&track.utauMode==UtauMode::classic&&!trackIsDiffSinger(track))
        {
            const auto context=AudioEngine::utauPhonemeMixContextHash(track);
            for(const auto& clip:track.clips)for(const auto& note:clip.notes)for(const auto& wave:*utauWaveforms)
                if(wave.noteId==note.id&&wave.phonemeLayoutHash==AudioEngine::utauNotePhonemeHash(note)
                    &&wave.phonemeMixContextHash==context&&!wave.phonemes.empty())
                    caches.emplace_back(note.id,backend::cvvcPhonemeCacheText({wave.phonemeLayoutHash,context,wave.phonemes}));
        }
    if(model.rememberCvvcPhonemes(caches,revision))
    {snapshot=model.snapshot();expandProjectClipParts(snapshot,true);}
    rebuildUtauSoundSpans(); rebuildNoteHits();
    repaint();
}

namespace
{
// The piece's own span, in the note's own row: it starts a lead-in before the
// note, because that is where the consonant is sung, and it lasts as long as
// the engine made it.  Drawn from the note's own start and length instead, a
// note's consonant landed in the row of the note before it.
std::pair<double, double> waveformSpan(double noteStartSeconds,
                                       const UtauNoteWaveform& waveform)
{
    return { noteStartSeconds - waveform.leadInSeconds, waveform.durationSeconds };
}
}

std::optional<std::pair<double, double>> PianoRollComponent::diagnosticWaveformSpan(
    const juce::String& noteId) const
{
    if (utauWaveforms == nullptr) return std::nullopt;
    for (const auto& waveform : *utauWaveforms)
    {
        if (waveform.noteId != noteId) continue;
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == noteId)
                        return waveformSpan(clip.startSeconds + note.startSeconds, waveform);
    }
    return std::nullopt;
}

std::vector<AmplitudeEnvelopePoint> PianoRollComponent::displayAmplitudeEnvelope(
    const NoteData& note, double absoluteStart) const
{
    auto envelope = amplitudeEnvelopeFor(note, absoluteStart);
    // Mid-drag the project still holds the old shape: what is on screen is the
    // stroke under the pointer, and that is what the audio has to follow.
    const auto* ownGroup = nativeEnvelopeMember(note.id);
    const auto* dragGroup = nativeEnvelopeMember(draggedNote);
    auto groupSelected = false;
    if (ownGroup)
        for (const auto& [trackId, envelopes] : nativeEnvelopeCache)
            for (const auto& [id, other] : envelopes)
                groupSelected |= other.points == ownGroup->points && (selectedNotes.contains(id.toStdString()) || id == selectedNote);
    const auto participates = groupSelected || (ownGroup && dragGroup && ownGroup->points == dragGroup->points)
        || note.id == draggedNote
        || selectedNotes.contains(note.id.toStdString())
        || (selectedNotes.empty() && note.id == selectedNote);
    if (dragMode == DragMode::amplitudePoint && !amplitudeStroke.empty() && participates)
        if (auto mapped = mapAmplitudeEnvelopeToNote(amplitudeStroke, draggedNote, note.id);
            mapped.size() >= 2)
            envelope = std::move(mapped);
    return effectiveAmplitudeEnvelope(note,absoluteStart,envelope);
}

bool PianoRollComponent::withinEnvelope(const UtauNoteWaveform& waveform,
                                        const std::vector<AmplitudeEnvelopePoint>& envelope,
                                        std::size_t bucket)
{
    if (envelope.size() < 2) return false;
    // Buckets are a millisecond each, so the middle of one is near enough the
    // moment to read the envelope at.
    const auto seconds = (static_cast<double>(bucket) + 0.5) / 1000.0
        - waveform.leadInSeconds;
    return seconds >= envelope.front().timeSeconds
        && seconds <= envelope.back().timeSeconds;
}

float PianoRollComponent::notePictureAt(const UtauNoteWaveform& waveform,
                                        const std::vector<AmplitudeEnvelopePoint>& envelope,
                                        std::size_t bucket, bool high)
{
    if (std::any_of(waveform.phonemes.begin(), waveform.phonemes.end(), [](const auto& p){return p.envelopePart>0;}))
        return high ? waveform.maxima[bucket] : waveform.minima[bucket];
    // The envelope on screen, not the one the piece was rendered with.  The
    // two agree in the song, but a render is only ever of the notes selected:
    // the last of them was rendered with no note after it, sang on to its
    // own end, and was drawn doing so -- past the point where, in the song,
    // the next note takes over and the envelope on screen ends.  The line is
    // the note as it will be heard in the song, so the picture follows it,
    // past its ends too, where it holds the gain it ends on.
    const auto value = high ? waveform.unshapedMaxima[bucket]
                            : waveform.unshapedMinima[bucket];
    const auto seconds = (static_cast<double>(bucket) + 0.5) / 1000.0
        - waveform.leadInSeconds;
    const auto db = amplitudeDbAt(envelope, seconds);
    return db <= -59.9f ? 0.0f : value * std::pow(10.0f, db / 20.0f);
}

float PianoRollComponent::noteGhostAt(const UtauNoteWaveform& waveform,
                                      const std::vector<AmplitudeEnvelopePoint>& envelope,
                                      std::size_t bucket)
{
    if (std::any_of(waveform.phonemes.begin(), waveform.phonemes.end(), [](const auto& p){return p.envelopePart>0;}))
        return std::max(waveform.maxima[bucket], -waveform.minima[bucket]);
    // Between the envelope's first and last points its shape is left out:
    // that is what the lane's line is there to change.  Past them the
    // envelope holds the gain it ends on, and no point in the lane can move
    // that, so it stays -- a note ending in silence is silence there, however
    // far the render of a selection let it run on.
    if (withinEnvelope(waveform, envelope, bucket))
        return std::max(waveform.unshapedMaxima[bucket], -waveform.unshapedMinima[bucket]);
    return std::max(notePictureAt(waveform, envelope, bucket, true),
                    -notePictureAt(waveform, envelope, bucket, false));
}

std::optional<PianoRollComponent::DrawnNoteWaveform>
PianoRollComponent::diagnosticDrawnNoteWaveform(const juce::String& noteId) const
{
    if (utauWaveforms == nullptr) return std::nullopt;
    for (const auto& waveform : *utauWaveforms)
    {
        if (waveform.noteId != noteId) continue;
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                {
                    if (note.id != noteId) continue;
                    if (waveform.audioHash != AudioEngine::utauNoteAudioHash(note)
                        || waveform.unshapedMaxima.empty())
                        return std::nullopt;
                    const auto envelope = displayAmplitudeEnvelope(
                        note, clip.startSeconds + note.startSeconds);
                    DrawnNoteWaveform drawn;
                    drawn.leadInSeconds = waveform.leadInSeconds;
                    for (std::size_t bucket = 0; bucket < waveform.unshapedMaxima.size(); ++bucket)
                    {
                        drawn.picture.push_back(std::max(
                            notePictureAt(waveform, envelope, bucket, true),
                            -notePictureAt(waveform, envelope, bucket, false)));
                        drawn.ghost.push_back(noteGhostAt(waveform, envelope, bucket));
                    }
                    return drawn;
                }
    }
    return std::nullopt;
}

int PianoRollComponent::drawAmplitudeLaneWaveforms(juce::Graphics& g)
{
    auto drawn = 0;
    const auto nativePlot = amplitudeLanePlotBounds();
    const auto nativeFloor = amplitudeLaneY(amplitudeDbFromPercent(0));
    const auto nativeScale = nativeFloor-amplitudeLaneY(amplitudeDbFromPercent(100));
    if(showNativeWaveforms)for(const auto& track:snapshot.tracks)
    {
        if(!trackShowsAllNativeRegions(track))continue;
        for(const auto& clip:track.clips)
        {
            if(!clipIsVisible(track,clip))continue;
            const auto found=thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
            if(found==thumbnails.end()||found->second->getTotalLength()<=0)continue;
            const auto audio=nativeAudioPreviewClip(clip);const auto map=nativeSourceTimeMap(audio);
            const auto peak=std::max(.001f,found->second->getApproximatePeak());
            for(const auto& note:clip.notes)
            {
                const auto start=clip.startSeconds+note.startSeconds;
                const auto left=std::max(nativePlot.getX(),timeToX(start));
                const auto right=std::min(nativePlot.getRight(),timeToX(start+note.durationSeconds));
                if(right<=left)continue;
                const auto envelope=displayAmplitudeEnvelope(note,start);
                const auto selected=selectedNotes.contains(note.id.toStdString())||selectedNote==note.id;
                for(auto x=left;x<right;x+=1.f)
                {
                    const auto local=(x-timeToX(start))/pixelsPerSecond;
                    const auto target=start+local-audio.startSeconds;
                    const auto a=std::max(0.0,audio.sourceOffsetSeconds+nativeSourceTimeAt(map,target));
                    const auto b=std::min(found->second->getTotalLength(),audio.sourceOffsetSeconds+nativeSourceTimeAt(map,target+1.0/pixelsPerSecond));
                    float low=0,high=0;
                    if(b>a)found->second->getApproximateMinMax(a,b,0,low,high);
                    const auto level=std::max(std::abs(low),std::abs(high))/peak;
                    const auto gain=backend::envelopeGainFromDb(amplitudeDbAt(envelope,local));
                    const auto ghost=nativeScale*level,shaped=ghost*gain;
                    g.setColour(Palette::noteLight.withAlpha(selected?.16f:.09f));g.fillRect(x,nativeFloor-ghost,1.f,ghost);
                    g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(selected?.46f:.26f));g.fillRect(x,nativeFloor-shaped,1.f,shaped);
                }
                ++drawn;
            }
        }
    }
    // What the envelope is being drawn over.  Without it the lane is a curve
    // against an empty grid: where the consonant ends, where the tail dies
    // away, whether a point sits on sound or on silence -- none of it is
    // visible, and those are the moments an envelope is placed by.
    //
    // Two layers, because the lane has two questions to answer.  The solid
    // one is the audio as the envelope now shapes it -- drag a point and it
    // follows, which is the whole reason to draw audio here.  Behind it, faint,
    // is the piece before the envelope: what the line is acting on, so a
    // stretch pulled to silence still shows the sound being silenced rather
    // than an empty lane that says nothing about why.
    if (!showUtauWaveforms || utauWaveforms == nullptr || utauWaveforms->empty()) return drawn;
    const auto plot = amplitudeLanePlotBounds();
    const auto baseline = amplitudeLaneY(amplitudeDbFromPercent(0.0f));
    const auto fullScale = amplitudeLaneY(amplitudeDbFromPercent(100.0f));
    if (plot.getWidth() <= 1.0f || baseline - fullScale <= 1.0f) return 0;

    std::unordered_map<std::string, const UtauNoteWaveform*> byId;
    byId.reserve(utauWaveforms->size());
    for (const auto& waveform : *utauWaveforms)
        byId.emplace(waveform.noteId.toStdString(), &waveform);

    for (const auto& hit : noteHits)
    {
        const auto found = byId.find(hit.id.toStdString());
        if (found == byId.end()) continue;
        const auto& waveform = *found->second;
        if (waveform.unshapedMaxima.empty() || waveform.durationSeconds <= 0.0) continue;
        bool utau = false;
        const auto* note = findNote(hit.id, &utau);
        if (note == nullptr || !utau) continue;
        // The note has to still be the note that was rendered, exactly as in
        // the roll above: an edit moves the hash and the audio stops being
        // drawn until the new audio arrives.
        // The audio hash, not the render hash: the lane draws the piece
        // before the envelope, and an envelope edit leaves that piece alone.
        if (waveform.audioHash != AudioEngine::utauNoteAudioHash(*note)
            || (usesCvvcEnvelope(note->id) && waveform.renderHash != AudioEngine::utauNoteRenderHash(*note))) continue;

        // Its own peak reaches the 100% line, so the envelope's 100% and the
        // sound's natural level are the same height and a point at half the
        // line reads as half the sound.
        auto peak = 0.0f;
        for (const auto value : waveform.unshapedMaxima) peak = std::max(peak, value);
        for (const auto value : waveform.unshapedMinima) peak = std::max(peak, -value);
        if (peak <= 1.0e-4f) continue;

        const auto span = waveformSpan(hit.startSeconds + hit.clipStartSeconds, waveform);
        const auto left = timeToX(span.first);
        const auto width = static_cast<float>(span.second) * pixelsPerSecond;
        if (width < 1.0f) continue;
        const auto buckets = static_cast<int>(waveform.unshapedMaxima.size());
        const auto columns = juce::jlimit(1, buckets, static_cast<int>(std::ceil(width)));
        const auto selected = selectedNotes.contains(hit.id.toStdString())
            || (selectedNotes.empty() && hit.id == selectedNote);
        // The envelope on screen, which during a drag is the one under the
        // pointer rather than the one the project still holds.
        const auto envelope = displayAmplitudeEnvelope(*note,
            hit.startSeconds + hit.clipStartSeconds);
        const auto ghostColour = Palette::noteLight.withAlpha(selected ? 0.16f : 0.09f);
        const auto shapedColour = Palette::noteLight.withAlpha(selected ? 0.46f : 0.26f);
        for (int column = 0; column < columns; ++column)
        {
            const auto from = buckets * column / columns;
            const auto to = std::max(from + 1, buckets * (column + 1) / columns);
            auto loudest = 0.0f;
            auto shaped = 0.0f;
            for (auto bucket = from; bucket < to && bucket < buckets; ++bucket)
            {
                const auto index = static_cast<std::size_t>(bucket);
                loudest = std::max(loudest, noteGhostAt(waveform, envelope, index));
                shaped = std::max({ shaped, notePictureAt(waveform, envelope, index, true),
                                    -notePictureAt(waveform, envelope, index, false) });
            }
            const auto x = left + width * static_cast<float>(column)
                                  / static_cast<float>(columns);
            if (x + 1.0f < plot.getX() || x > plot.getRight()) continue;
            // Level, not a wave: the lane's floor is silence and its axis runs
            // one way, so what belongs here is how loud it is at that moment.
            const auto columnWidth = std::max(1.0f, width / static_cast<float>(columns));
            const auto ghostHeight = (baseline - fullScale) * loudest / peak;
            const auto shapedHeight = (baseline - fullScale) * shaped / peak;
            g.setColour(ghostColour);
            g.fillRect(x, baseline - ghostHeight, columnWidth,
                       std::max(1.0f, ghostHeight));
            g.setColour(shapedColour);
            g.fillRect(x, baseline - shapedHeight, columnWidth,
                       std::max(1.0f, shapedHeight));
        }
        ++drawn;
    }
    return drawn;
}

void PianoRollComponent::drawUtauNoteWaveforms(juce::Graphics& g)
{
    if (utauWaveforms == nullptr || utauWaveforms->empty()) return;
    // Indexed once rather than searched per note: a song is thousands of
    // notes and this runs inside paint.
    std::unordered_map<std::string, const UtauNoteWaveform*> byId;
    byId.reserve(utauWaveforms->size());
    for (const auto& waveform : *utauWaveforms)
        byId.emplace(waveform.noteId.toStdString(), &waveform);

    for (const auto& track : snapshot.tracks)
    {
        if (track.pitchAlgorithm != PitchAlgorithm::utau) continue;
        if (focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
            for (const auto& note : clip.notes)
            {
                const auto found = byId.find(note.id.toStdString());
                if (found == byId.end()) continue;
                const auto& waveform = *found->second;
                // The note has to still be the note that was rendered.  Edit
                // it and the hash moves, and the waveform stops being drawn
                // until the new audio arrives to replace it.
                //
                // The audio hash and not the render hash: an envelope decides
                // how loud the audio is and nothing else about it, so the
                // piece already rendered is still the right piece after an
                // envelope edit, and only its shape has to be redrawn.
                if (waveform.audioHash != AudioEngine::utauNoteAudioHash(note)
                    || (usesCvvcEnvelope(note.id) && waveform.renderHash != AudioEngine::utauNoteRenderHash(note))) continue;
                if (waveform.unshapedMaxima.empty() || waveform.durationSeconds <= 0.0) continue;
                // Always the piece shaped by the envelope on screen -- which
                // during a drag is the stroke under the pointer, so the picture
                // follows the point being dragged.  Never the audio as this
                // render shaped it: a render is of the selected notes only, and
                // the last of them sang on with no note after it, drawn past
                // the end of the envelope it will have in the song.
                const auto envelope = displayAmplitudeEnvelope(
                    note, clip.startSeconds + note.startSeconds);
                const auto shapeAt = [&](std::size_t bucket, bool high)
                {
                    return notePictureAt(waveform, envelope, bucket, high);
                };

                const auto span = waveformSpan(clip.startSeconds + note.startSeconds,
                                               waveform);
                const auto left = timeToX(span.first);
                const auto width = static_cast<float>(span.second) * pixelsPerSecond;
                if (width < 1.0f) continue;
                const auto height = rowHeight * 6.0f;
                const auto centre = midiToY(note.midiNote) + rowHeight * 0.5f;
                g.setColour(Palette::textMuted.withAlpha(track.muted ? 0.12f : 0.34f));
                const auto buckets = static_cast<int>(waveform.unshapedMaxima.size());
                const auto columns = juce::jlimit(1, buckets,
                                                  static_cast<int>(std::ceil(width)));
                for (int column = 0; column < columns; ++column)
                {
                    // Every bucket the column covers, so zooming out keeps the
                    // peaks instead of sampling one bucket in twenty.
                    const auto from = buckets * column / columns;
                    const auto to = std::max(from + 1, buckets * (column + 1) / columns);
                    auto low = 0.0f;
                    auto high = 0.0f;
                    for (auto bucket = from; bucket < to && bucket < buckets; ++bucket)
                    {
                        const auto index = static_cast<std::size_t>(bucket);
                        low = std::min(low, shapeAt(index, false));
                        high = std::max(high, shapeAt(index, true));
                    }
                    const auto x = left + width * static_cast<float>(column)
                                          / static_cast<float>(columns);
                    const auto top = centre - high * height * 0.5f;
                    const auto bottom = centre - low * height * 0.5f;
                    g.fillRect(x, top, std::max(1.0f, width / static_cast<float>(columns)),
                               std::max(1.0f, bottom - top));
                }
            }
    }
}

std::optional<PianoRollComponent::NativeOverlapPair> PianoRollComponent::overlapPairFor(
    const juce::String& noteId,double time) const
{
    const TrackData* ownerTrack=nullptr;const ClipData* owner=nullptr;const NoteData* target=nullptr;
    for(const auto& track:snapshot.tracks)for(const auto& clip:track.clips)
        for(const auto& note:clip.notes)if(note.id==noteId){ownerTrack=&track;owner=&clip;target=&note;}
    if(sourceEditMode||!ownerTrack||!owner||!target||!physicalOverlapSource(*owner)||!clipIsVisible(*ownerTrack,*owner))return {};
    if(!std::isfinite(time))time=owner->startSeconds+target->startSeconds+target->durationSeconds*.5;
    std::optional<NativeOverlapPair> chosen;double best=std::numeric_limits<double>::infinity();
    for(const auto& other:ownerTrack->clips)
    {
        if(&other==owner||!physicalOverlapSource(other)||other.notes.empty())continue;
        const auto overlap=physicalOverlapWindow(*owner).getIntersectionWith(physicalOverlapWindow(other));
        if(overlap.getLength()<=1.e-7)continue;
        // Never choose an unrelated nearby source: EVERY candidate contains the
        // explicitly requested note's physical source. Time only disambiguates
        // that source's own overlaps, not a nearest-note/source fallback.
        const auto distance=time<overlap.getStart()?overlap.getStart()-time:time>overlap.getEnd()?time-overlap.getEnd():0.0;
        if(chosen&&distance>=best)continue;
        best=distance;const bool ownerFirst=physicalOverlapWindow(*owner).getStart()<=physicalOverlapWindow(other).getStart();
        chosen=NativeOverlapPair{ownerTrack,ownerFirst?owner:&other,ownerFirst?&other:owner,overlap.getStart(),overlap.getEnd()};
    }
    return chosen;
}
void PianoRollComponent::openNativeOverlapSelector(const juce::String& noteId,double time)
{
    closeNativeOverlapSelector();refreshFromModel();const auto pair=overlapPairFor(noteId,time);if(!pair)return;
    if(!std::isfinite(time))time=(pair->from+pair->to)*.5;
    nativeOverlapSelector=NativeOverlapSelectorState{noteId,pair->left->notes.front().id,pair->right->notes.front().id,time,model.revisionNumber()};
    if(isShowing())grabKeyboardFocus();repaint();
}
void PianoRollComponent::closeNativeOverlapSelector()
{
    nativeOverlapSelector.reset();nativeOverlapSourceHits.clear();nativeOverlapCardCount=0;nativeOverlapSelectorClose={};repaint();
}

void PianoRollComponent::drawNativeAudioOverlaps(juce::Graphics& g)
{
    nativeOverlapHints.clear();nativeOverlapSourceHits.clear();nativeOverlapLines.clear();nativeOverlapCardCount=0;nativeOverlapSelectorClose={};
    if(nativeOverlapSelector&&nativeOverlapSelector->revision!=model.revisionNumber())closeNativeOverlapSelector();
    if(sourceEditMode)return;
    auto visible=g.getClipBounds().toFloat();
    if(const auto* viewport=findParentComponentOfClass<juce::Viewport>())visible=viewport->getViewArea().toFloat();
    for(const auto& track:snapshot.tracks)
    {
        if(focusedTrack.isNotEmpty()&&track.id!=focusedTrack)continue;
        std::vector<const ClipData*> sources;for(const auto& clip:track.clips)if(physicalOverlapSource(clip))sources.push_back(&clip);
        std::stable_sort(sources.begin(),sources.end(),[](const auto* a,const auto* b){return a->startSeconds+a->audioStartSeconds<b->startSeconds+b->audioStartSeconds;});
        for(std::size_t i=0;i<sources.size();++i)for(std::size_t j=i+1;j<sources.size();++j)
        {
            const auto& a=*sources[i];const auto& b=*sources[j];
            if(b.startSeconds+b.audioStartSeconds>=a.startSeconds+a.audioStartSeconds+a.audioLength()-1.e-7)break;
            const auto begin=std::max(a.startSeconds+a.audioStartSeconds,b.startSeconds+b.audioStartSeconds);
            const auto end=std::min(a.startSeconds+a.audioStartSeconds+a.audioLength(),b.startSeconds+b.audioStartSeconds+b.audioLength());
            if(end-begin<=1.e-7)continue;const auto left=timeToX(begin),right=timeToX(end);
            const bool matching = nativeOverlapSelector && !a.notes.empty() && !b.notes.empty()
                && ((nativeOverlapSelector->leftSourceId==a.notes.front().id&&nativeOverlapSelector->rightSourceId==b.notes.front().id)
                    ||(nativeOverlapSelector->leftSourceId==b.notes.front().id&&nativeOverlapSelector->rightSourceId==a.notes.front().id));
            if(!matching&&(!clipIsVisible(track,a)&&!clipIsVisible(track,b)))continue;
            if(!matching&&(right<visible.getX()+58||left>visible.getRight()))continue;
            const auto amber=juce::Colour(0xffffbf69),blue=rollRole(juce::Colour(0xff70dcff), Palette::heading);
            // Optional overlap marks belong to the two actual note rows, never
            // the full-height timeline. The explicit picker is independent.
            if(showOverlap)for(const auto& an:a.notes)for(const auto& bn:b.notes)
            {
                const auto noteBegin=std::max({begin,a.startSeconds+an.startSeconds,b.startSeconds+bn.startSeconds});
                const auto noteEnd=std::min({end,a.startSeconds+an.startSeconds+an.durationSeconds,b.startSeconds+bn.startSeconds+bn.durationSeconds});
                if(noteEnd-noteBegin<=1.e-7)continue;
                const auto mark=[&](const ClipData& owner,const NoteData& note)
                {
                    if(!clipIsVisible(track,owner))return;
                    const juce::Rectangle<float> line(timeToX(noteBegin),midiToY(note.midiNote)+rowHeight*.5f-1.f,
                        timeToX(noteEnd)-timeToX(noteBegin),2.f);
                    g.setColour(juce::Colour(0xffffd54f));g.fillRect(line);
                    nativeOverlapLines.emplace_back(note.id,line);
                };
                mark(a,an);mark(b,bn);
            }
            const auto title=strings.text("native.overlap")+" "+juce::String((end-begin)*1000,1)+" ms · "+strings.text("native.overlapMix");
            const auto tip=title+"\n"+strings.text("native.overlapEarlier")+": "+a.sourceFile.getFileName()+"\n"+strings.text("native.overlapLater")+": "+b.sourceFile.getFileName()
                +"\n"+juce::String(begin,3)+" – "+juce::String(end,3)+" s";
            if(matching&&nativeOverlapCardCount==0)
            {
                const auto width=std::min(270.f,std::max(190.f,right-left));
                const juce::Rectangle<float> card(juce::jlimit(visible.getX()+60,std::max(visible.getX()+60,visible.getRight()-width-4),timeToX(nativeOverlapSelector->targetSeconds)),
                    visible.getY()+4,width,80);
                g.setColour(rollRole(juce::Colour(0xff182126), Palette::panelRaised).withAlpha(.95f));g.fillRoundedRectangle(card,5);g.setColour(amber);g.drawRoundedRectangle(card,5,1.2f);
                g.setColour(rollText(amber, Palette::graphBackground.overlaidWith(Palette::panelRaised.withAlpha(.95f))));
                nativeOverlapSelectorClose={card.getRight()-19,card.getY()+2,16,16};
                g.setFont(11.f);g.drawText(strings.text("Material selector"),
                    card.reduced(7,3).withHeight(15).withTrimmedRight(18),juce::Justification::centredLeft);
                g.drawText("x",nativeOverlapSelectorClose,juce::Justification::centred);
                const auto wave=[&](const ClipData& clip,bool later)
                {
                    const auto row=card.reduced(7,0).withY(card.getY()+20+(later?28:0)).withHeight(25);
                    const auto opacity=nativeOverlapFocusOpacity(track,clip,selectedNote,focusedClip);
                    const auto active=selectedNote.isNotEmpty()?std::any_of(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.id==selectedNote;})
                        :focusedClip.isNotEmpty()&&clip.id==focusedClip;
                    if(active)
                    {g.setColour(blue.withAlpha(.12f));g.fillRoundedRectangle(row.expanded(3,1),3);}
                    if(!clip.notes.empty())nativeOverlapSourceHits.push_back({row.expanded(3,1),clip.notes.front().id});
                    const auto choiceSurface=Palette::graphBackground.overlaidWith(Palette::panelRaised.withAlpha(.95f))
                        .overlaidWith(active ? blue.withAlpha(.12f) : juce::Colours::transparentBlack);
                    g.setColour(rollText(later?blue:amber, choiceSurface));g.setFont(10.f);
                    g.drawText(strings.text(later?"native.overlapLater":"native.overlapEarlier")+" · "+clip.sourceFile.getFileName(),row.withHeight(11),juce::Justification::centredLeft,true);
                    const auto plot=row.withY(row.getY()+12).withHeight(12);const auto found=thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
                    if(found==thumbnails.end()){g.drawText("…",plot,juce::Justification::centred);return;}
                    const NativeAudioFocusLayer focus(g,opacity); // dim waveform only, never selector actions
                    const auto clock=nativeClipClock(clip);juce::Path shape;
                    for(int k=0;k<static_cast<int>(plot.getWidth());++k)
                    {
                        const auto t=begin+(end-begin)*k/plot.getWidth(),next=begin+(end-begin)*(k+1)/plot.getWidth();
                        float low=0,high=0;const auto first=clip.sourceOffsetSeconds+nativeSourceTimeAt(clock,t-clip.startSeconds);
                        const auto last=clip.sourceOffsetSeconds+nativeSourceTimeAt(clock,next-clip.startSeconds);
                        if(last>first)found->second->getApproximateMinMax(first,last,0,low,high);
                        const auto h=std::min(1.f,std::max(std::abs(low),std::abs(high))*2.f)*plot.getHeight()*.5f;
                        shape.startNewSubPath(plot.getX()+k,plot.getCentreY()-h);shape.lineTo(plot.getX()+k,plot.getCentreY()+h);
                    }
                    g.strokePath(shape,juce::PathStrokeType(1.f));
                };
                wave(a,false);wave(b,true);nativeOverlapHints.push_back({card,tip});++nativeOverlapCardCount;
            }

        }
    }
}

juce::Rectangle<float> PianoRollComponent::nativeWaveformLabelBounds(
    const TrackData& track, const ClipData& clip, const NoteData& note,
    juce::Rectangle<float> bounds) const
{
    if (sourceEditMode || !trackShowsAllNativeRegions(track) || clip.audioLength() <= 1.0e-9)
        return bounds;
    const auto found = thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
    const auto sourceReady = found != thumbnails.end() && found->second->getTotalLength() > 0.0;
    if (showWaveforms ? !sourceReady
                     : !(nativeRenderedPeaksFor(clip) || (showNativeWaveforms && sourceReady)))
        return bounds;

    auto centreMidi = note.midiNote;
    auto height = rowHeight * 2.4f;
    if (showWaveforms)
    {
        double weighted = 0.0, duration = 0.0;
        for (const auto& member : clip.notes)
        {
            const auto weight = std::max(0.01, member.durationSeconds);
            weighted += member.midiNote * weight;
            duration += weight;
        }
        centreMidi = duration > 0.0 ? static_cast<float>(weighted / duration) : 60.0f;
        height = rowHeight * 10.0f;
    }
    // Reserve the waveform's full drawing height, including loud peaks.  This
    // also follows the preview's pitch/clock when notes are moved or stretched.
    const auto waveformTop = midiToY(centreMidi) + rowHeight * 0.5f - height * 0.5f;
    const auto labelHeight = juce::jlimit(15.0f, 20.0f, rowHeight * 0.65f);
    bounds.setY(waveformTop - 3.0f - labelHeight);
    bounds.setHeight(labelHeight);
    return bounds;
}

void PianoRollComponent::drawNativeNoteWaveforms(juce::Graphics& g)
{
    if (sourceEditMode || (showWaveforms && !nativeLoudnessMode)) return; // the overview already draws it
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose || track.accompaniment || trackUsesVoicebankSynthesis(track)) continue;
        if (focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto* source : nativeAudioFocusOrder(track,sourceEditMode?juce::String{}:selectedNote,sourceEditMode?juce::String{}:focusedClip))
        {
            const auto& clip=*source;
            if (!clipIsVisible(track, clip)) continue;
            const auto* rendered = nativeLoudnessMode ? nullptr : nativeRenderedPeaksFor(clip);
            if(!rendered && !showNativeWaveforms && !nativeLoudnessMode)continue;
            const NativeAudioFocusLayer focus(g,sourceEditMode?1.f:nativeOverlapFocusOpacity(track,clip,selectedNote,focusedClip));
            const auto found = thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
            if (!rendered && found == thumbnails.end()) continue;
            const auto audio = nativeAudioPreviewClip(clip);
            const auto map = nativeSourceTimeMap(audio);
            for (const auto& note : clip.notes)
            {
                if (note.durationSeconds <= 1.0e-9) continue;
                const auto height = rowHeight * 2.4f * (rendered ? 1.0f : displayedNativeGain(note));
                if (height <= 0.0f) continue;
                const auto centre = midiToY(note.midiNote) + rowHeight * 0.5f;
                const juce::Graphics::ScopedSaveState save(g);
                g.reduceClipRegion(juce::Rectangle<float>(timeToX(clip.startSeconds + note.startSeconds),
                    centre - height * 0.5f, static_cast<float>(note.durationSeconds) * pixelsPerSecond,
                    height).toNearestInt());
                g.setColour(Palette::textMuted.withAlpha(track.muted || clip.muted ? 0.12f : 0.34f));
                const juce::Rectangle<float> bounds(timeToX(audio.startSeconds),centre-height*.5f,
                    static_cast<float>(audio.durationSeconds)*pixelsPerSecond,height);
                const auto opacity=track.muted||clip.muted ? .32f : .94f;
                const auto selected=selectedNotes.contains(note.id.toStdString())||selectedNote==note.id;
                if(rendered)drawNativeRenderedWaveform(g,*rendered,audio.durationSeconds,bounds,opacity,selected);
                else drawNativeSourceWaveform(g,*found->second,audio,map,bounds,opacity,selected);
            }
        }
    }
}

void PianoRollComponent::drawClipWaveforms(juce::Graphics& g)
{
    if (nativeLoudnessMode && !sourceEditMode) return;
    juce::String focusedSource;
    if (sourceEditMode)
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                if (clip.id == focusedClip)
                    focusedSource = clip.sourceFile.getFullPathName();
    bool sourceWaveformDrawn = false;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose && !sourceEditMode) continue;
        if (!sourceEditMode && focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto* source : nativeAudioFocusOrder(track,sourceEditMode?juce::String{}:selectedNote,sourceEditMode?juce::String{}:focusedClip))
        {
            const auto& clip=*source;
            if (!clipIsVisible(track, clip))
                continue;
            const NativeAudioFocusLayer focus(g,sourceEditMode?1.f:nativeOverlapFocusOpacity(track,clip,selectedNote,focusedClip));
            const auto found = thumbnails.find(clip.sourceFile.getFullPathName().toStdString());
            if (found == thumbnails.end()) continue;
            float centreMidi = 60.0f;
            if (!clip.notes.empty())
            {
                double weighted = 0.0;
                double duration = 0.0;
                for (const auto& note : clip.notes)
                {
                    const auto weight = std::max(0.01, note.durationSeconds);
                    weighted += static_cast<double>(note.midiNote) * weight;
                    duration += weight;
                }
                centreMidi = static_cast<float>(weighted / std::max(0.01, duration));
            }
            if (sourceEditMode && sourceWaveformDrawn) continue;
            if (!sourceEditMode && clip.audioLength() <= 1.0e-9) continue;
            const auto start = sourceEditMode ? 0.0 : clip.startSeconds + clip.audioStartSeconds;
            const auto sourceLength = sourceEditMode ? found->second->getTotalLength()
                                                      : (clip.sourceDurationSeconds > 1.0e-9
                                                          ? clip.sourceDurationSeconds : clip.audioLength());
            const auto waveformHeight = rowHeight * 10.0f;
            const juce::Rectangle<float> bounds(timeToX(start),
                midiToY(centreMidi) + rowHeight * 0.5f - waveformHeight * 0.5f,
                std::max(4.0f, static_cast<float>(sourceEditMode ? sourceLength : clip.audioLength())
                                      * pixelsPerSecond), waveformHeight);
            g.setColour(Palette::textMuted.withAlpha(track.muted ? 0.12f : 0.34f));
            // Stereo material uses one full-height editing waveform instead of
            // JUCE's stacked per-channel lanes.  Audio playback/export remains
            // stereo; this selects channel 1 for display only.
            if (!sourceEditMode && !trackUsesVoicebankSynthesis(track) && !track.accompaniment)
            {
                const auto audio = nativeAudioPreviewClip(clip);
                if(const auto* rendered=nativeRenderedPeaksFor(clip))
                    drawNativeRenderedWaveform(g,*rendered,audio.durationSeconds,bounds,
                        track.muted||clip.muted?.32f:.78f);
                else drawNativeSourceWaveform(g, *found->second, audio, nativeSourceTimeMap(audio), bounds,
                    track.muted || clip.muted ? 0.32f : 0.78f);
                continue;
            }
            found->second->drawChannel(
                g, bounds.toNearestInt(),
                sourceEditMode ? 0.0 : clip.sourceOffsetSeconds,
                sourceEditMode ? sourceLength : clip.sourceOffsetSeconds + sourceLength,
                0, 1.0f);
            if (sourceEditMode) sourceWaveformDrawn = true;
        }
    }
}

void PianoRollComponent::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == consonantVoicing.get()) { repaint(); return; }
    if (source == &model)
    {
        // Rebuild point handles from the authoritative contour after undo,
        // freehand/line edits, note resizing, and project loading.  The
        // simplifier prevents these rebuilds from exposing dense 5 ms frames.
        if (dragMode == DragMode::continuousFlag && model.revisionNumber() != continuousFlagRevision)
            cancelContinuousFlag();
        pitchAnchorCache.clear();connectionPitchCache.clear();protocolPitchCache.clear();clearPitchPlanPreview();
        sharedLineCache.clear(); nativeEnvelopeCache.clear();
        draggedSharedKey = 0;
        rebuildLayout();
    }
    else
    {
        updateCanvasSize();
        repaint();
    }
}

void PianoRollComponent::diagnosticBeginResizeDrag(const juce::String& noteId,
                                                   double previewStart,
                                                   double previewDuration)
{
    draggedNote = noteId;
    dragMode = DragMode::resizeRight;
    previewStartSeconds = previewStart;
    previewDurationSeconds = previewDuration;
}

void PianoRollComponent::diagnosticBeginConsonantDrag(const juce::String& noteId,
                                                      double previewPreutterance)
{
    draggedNote = noteId;
    dragMode = DragMode::consonantLeadIn;
    previewConsonantPreutterance = previewPreutterance;
    consonantDragTravelled = true;
}

bool PianoRollComponent::diagnosticGrabConsonantHandle(std::size_t index)
{
    std::size_t seen = 0;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (seen++ != index) continue;
                const auto span = utauSoundSpans.find(note.id.toStdString());
                if (span == utauSoundSpans.end()) return false;
                const auto handle = consonantHandleAt({ timeToX(span->second.first),
                    midiToY(note.midiNote) + 4.0f });
                if (!handle) return false;
                draggedNote = handle->noteId;
                consonantNoteAbsoluteStart = handle->absoluteStart;
                consonantScaledPreutterance = handle->scaledPreutterance;
                consonantUnscaledPreutterance = handle->unscaledPreutterance;
                consonantMinimumPreutterance = handle->minimumPreutterance;
                consonantMaximumPreutterance = handle->maximumPreutterance;
                previewConsonantPreutterance = handle->currentPreutterance;
                previewConsonantVelocity = handle->currentVelocity;
                previewConsonantVelocityBeforeDrag = handle->currentVelocity;
                consonantSetsPin = handle->setsPin;
                consonantOverlapSeconds = handle->overlapSeconds;
                consonantCvvcLead = handle->leadForVelocity;
                consonantDragTravelled = false;
                dragMode = DragMode::consonantLeadIn;
                return true;
            }
    return false;
}

void PianoRollComponent::dragConsonantTo(double cursorSeconds)
{
    const auto requested = juce::jlimit(consonantMinimumPreutterance,
        consonantMaximumPreutterance,
        consonantNoteAbsoluteStart - cursorSeconds);
    if (consonantCvvcLead)
    {
        // Invert the complete sounding lead, including VC advance and OTO
        // caps. The callback only evaluates captured timings, with no I/O.
        int low = -102200, high = 102300;
        while (high - low > 1)
        {
            const auto mid = low + (high - low) / 2;
            if (consonantCvvcLead(mid) > requested) low = mid; else high = mid;
        }
        previewConsonantVelocity = std::abs(consonantCvvcLead(low) - requested)
            < std::abs(consonantCvvcLead(high) - requested) ? low : high;
        // Preserve the current value on a capped plateau.
        if (std::abs(consonantCvvcLead(previewConsonantVelocity) - requested)
            >= std::abs(consonantCvvcLead(previewConsonantVelocityBeforeDrag) - requested))
            previewConsonantVelocity = previewConsonantVelocityBeforeDrag;
        previewConsonantPreutterance = consonantCvvcLead(previewConsonantVelocity);
        return;
    }
    if (consonantSetsPin)
    {
        // Straight to the lead-in.  谋 writes what the drag asked for rather
        // than the velocity that would produce it, so there is nothing to
        // express it through and the handle follows the cursor exactly.  How
        // far it may go is the handle's business, not this one's: free back to
        // the note in front when the entry calls the first region a vowel, and
        // what the oto can express when it calls it a consonant.
        previewConsonantPreutterance = requested;
        return;
    }
    const auto maximumScale = std::max(1.0e-12,
        (consonantMaximumPreutterance - consonantUnscaledPreutterance)
            / std::max(1.0e-9, consonantScaledPreutterance));
    const auto scale = juce::jlimit(1.0e-12, maximumScale,
        (requested - consonantUnscaledPreutterance)
            / std::max(1.0e-9, consonantScaledPreutterance));
    previewConsonantVelocity = static_cast<int>(std::lround(
        100.0 * (1.0 - std::log2(scale))));
    const auto exactScale = std::pow(2.0,
        1.0 - static_cast<double>(previewConsonantVelocity) / 100.0);
    previewConsonantPreutterance = juce::jlimit(
        consonantMinimumPreutterance, consonantMaximumPreutterance,
        consonantUnscaledPreutterance
            + consonantScaledPreutterance * exactScale);
}

void PianoRollComponent::diagnosticBeginMoveDrag(const juce::String& noteId,
                                                 double deltaSeconds)
{
    draggedNote = noteId;
    bool isUtau=false;
    (void)findNote(noteId,&isUtau);
    dragMode = isUtau ? DragMode::moveUtauNote : DragMode::moveNativeNote;
    if(!isUtau) noteDragPassedThreshold=true;
    selectedNotes.clear();
    selectedNotes.insert(noteId.toStdString());
    previewMoveDeltaSeconds = deltaSeconds;
    if (const auto* note = findNote(noteId))
    {
        dragStartMidi = note->midiNote;
        previewMidi = note->midiNote;
    }
}

juce::Rectangle<int> PianoRollComponent::diagnosticDragRepaintArea(float mouseX) const
{
    const auto band = dragRepaintBand(mouseX);
    if (band.isEmpty()) return getLocalBounds();
    const auto left = static_cast<int>(std::floor(band.getStart())) - 8;
    const auto right = static_cast<int>(std::ceil(band.getEnd())) + 8;
    return { left, 0, std::max(1, right - left), getHeight() };
}

juce::Range<float> PianoRollComponent::dragRepaintBand(float mouseX) const
{
    // Region handles are collected while painting, so a partial paint would
    // lose the ones it skipped.  Wrench mode always repaints in full.
    if (sourceEditMode || dragMode == DragMode::moveNativeNote
        || dragMode == DragMode::resizeNativeLeft || dragMode == DragMode::resizeNativeRight) return {};
    if (dragMode == DragMode::marquee)
        return { std::min(marqueeStart.x, marqueeCurrent.x),
                 std::max(marqueeStart.x, marqueeCurrent.x) };
    juce::Range<float> involved;
    auto found = false;
    for (const auto& hit : noteHits)
    {
        if (hit.id != draggedNote
            && !(dragMode == DragMode::moveUtauNote
                 && selectedNotes.contains(hit.id.toStdString())))
            continue;
        const juce::Range<float> bounds(hit.bounds.getX(), hit.bounds.getRight());
        involved = found ? involved.getUnionWith(bounds) : bounds;
        found = true;
    }
    if (!found) return {};
    // The hits still hold where the drag started, and whatever is being
    // dragged follows the cursor without growing, so it cannot have reached
    // further from the cursor than its own width.  The extra margin covers
    // the pitch line handing over to the note next door.
    const auto reach = involved.getLength() + 96.0f;
    return involved.getUnionWith({ mouseX - reach, mouseX + reach });
}

void PianoRollComponent::repaintDrag(float mouseX)
{
    const auto band = dragRepaintBand(mouseX);
    if (band.isEmpty())
    {
        lastDragBand = {};
        repaint();
        return;
    }
    const auto united = lastDragBand.isEmpty() ? band : band.getUnionWith(lastDragBand);
    lastDragBand = band;
    const auto left = static_cast<int>(std::floor(united.getStart())) - 8;
    const auto right = static_cast<int>(std::ceil(united.getEnd())) + 8;
    // Full height deliberately: the viewport clips it to the rows on screen
    // anyway, and a pitch line may leave its note's own row entirely.
    repaint(left, 0, std::max(1, right - left), getHeight());
}

void PianoRollComponent::rebuildNoteHits()
{
    noteHits.clear();
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose && !sourceEditMode) continue;
        if (!sourceEditMode && focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto* source : nativeAudioFocusOrder(track,sourceEditMode?juce::String{}:selectedNote,sourceEditMode?juce::String{}:focusedClip))
        {
            const auto& clip=*source;
            if (!clipIsVisible(track, clip)) continue;
            const auto sourceScale = sourceEditMode && clip.audioLength() > 1.0e-9
                ? (clip.sourceDurationSeconds > 1.0e-9 ? clip.sourceDurationSeconds
                                                        : clip.audioLength())
                    / clip.audioLength()
                : 1.0;
            for (const auto& note : clip.notes)
            {
                const auto absoluteStart = sourceEditMode
                    ? clip.sourceOffsetSeconds + (note.startSeconds - clip.audioStartSeconds) * sourceScale
                    : clip.startSeconds + note.startSeconds;
                const auto x = timeToX(absoluteStart);
                const auto y = midiToY(note.midiNote);
                const auto width = std::max(5.0f,
                    static_cast<float>(note.durationSeconds * sourceScale) * pixelsPerSecond);
                const auto blockHeight = std::max(6.0f, rowHeight - 4.0f);
                auto hitBounds = juce::Rectangle<float>(x, y + 2.0f, width, blockHeight);
                // Kept before the sounding span takes over, and worked out
                // here where the source-edit scaling is still to hand.
                const auto nominalBounds = hitBounds;
                if (track.pitchAlgorithm == PitchAlgorithm::utau)
                    if (const auto span = utauSoundSpans.find(note.id.toStdString());
                        span != utauSoundSpans.end())
                    {
                        const auto soundingX = timeToX(span->second.first);
                        hitBounds.setX(soundingX);
                        hitBounds.setWidth(std::max(2.0f,
                            timeToX(span->second.second) - soundingX));
                    }
                noteHits.push_back({ note.id,
                    hitBounds, nominalBounds,
                    note.midiNote, note.startSeconds, note.durationSeconds,
                    clip.startSeconds });
            }
        }
    }
}

const PianoRollComponent::NoteHit* PianoRollComponent::resizableTailAt(
    juce::Point<float> position, double& maximumDuration) const
{
    maximumDuration = 1.0e12;
    constexpr auto boundaryTolerance = 0.001;
    // The inside of a native note's left edge belongs to that note, even
    // when the preceding note's right-edge hit zone overlaps the same seam.
    for (const auto& hit : noteHits)
        if (nativeNoteTimingEnabled(hit.id) && hit.bounds.contains(position)
            && position.x <= hit.bounds.getX() + 6.0f)
            return nullptr;
    for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
    {
        const auto absoluteStart = it->clipStartSeconds + it->startSeconds;
        const auto nominalBounds = juce::Rectangle<float>(
            timeToX(absoluteStart), midiToY(it->midi) + 2.0f,
            std::max(5.0f, static_cast<float>(it->durationSeconds) * pixelsPerSecond),
            std::max(6.0f, rowHeight - 4.0f));
        const auto tail = juce::Rectangle<float>(nominalBounds.getRight() - 7.0f,
            nominalBounds.getY(), 14.0f, nominalBounds.getHeight());
        if (!tail.contains(position)) continue;
        if(nativeNoteTimingEnabled(it->id)) return &*it;

        auto nextStart = 1.0e12;
        for (const auto& candidate : noteHits)
        {
            if (candidate.id == it->id) continue;
            const auto candidateStart = candidate.clipStartSeconds + candidate.startSeconds;
            // The following note is a maximum boundary, not a reason to hide
            // the handle.  An adjacent note therefore still permits shrinking,
            // and the shortened note can later be extended back to this start.
            // If notes already overlap, dragging immediately snaps the tail
            // into the legal non-overlapping range.
            if (candidateStart > absoluteStart + boundaryTolerance)
                nextStart = std::min(nextStart, candidateStart);
        }
        if (nextStart < 1.0e11)
        {
            maximumDuration = nextStart - absoluteStart;
            if (maximumDuration + boundaryTolerance < noteEditQuantumSeconds())
                return nullptr;
        }
        return &*it;
    }
    return nullptr;
}

const JoinedVibrato* PianoRollComponent::joinedVibratoFor(const juce::String& noteId) const
{
    const auto* track=trackOf(noteId);if(!track)return nullptr;
    auto found=joinedVibratoCache.find(track->id);
    if(found==joinedVibratoCache.end())found=joinedVibratoCache.emplace(track->id,joinedVibratos(*track)).first;
    for(const auto& group:found->second)
        if(std::find(group.notes.begin(),group.notes.end(),noteId)!=group.notes.end())return &group;
    return nullptr;
}

bool PianoRollComponent::ownsVibratoControls(const NoteData& note) const
{
    const auto* group=joinedVibratoFor(note.id);
    if(!group)return selectedNotes.contains(note.id.toStdString())||note.id==selectedNote;
    if(std::none_of(group->notes.begin(),group->notes.end(),[&](const auto& id){return selectedNotes.contains(id.toStdString())||id==selectedNote;}))return false;
    const auto* track=trackOf(note.id);if(!track)return false;
    // A focused member may be in a different clip from the first member.
    for(const auto& id:group->notes)
        for(const auto& clip:track->clips)if(clipIsVisible(*track,clip))
            for(const auto& member:clip.notes)if(member.id==id)return id==note.id;
    return false;
}

NoteData PianoRollComponent::vibratoControlNote(const NoteData& note) const
{
    auto control=effectiveVibrato(note);
    if(const auto* group=joinedVibratoFor(note.id))
    {
        control.durationSeconds=group->duration;
        control.vibratoReferenceDurationSeconds=0;
        control.vibratoTimeOffsetSeconds=0;
    }
    return control;
}

NoteData PianoRollComponent::effectiveVibrato(const NoteData& note) const
{
    const auto* group=joinedVibratoFor(note.id);
    auto preview=note;
    if(group)if(const auto* leader=findNote(group->notes.front()))copyVibratoParameters(preview,*leader);
    if (dragMode == DragMode::vibrato && (note.id == draggedNote
        ||(group&&std::find(group->notes.begin(),group->notes.end(),draggedNote)!=group->notes.end())))
    {
        copyVibratoParameters(preview,previewVibrato);
        preview.vibratoEnabled=true;
    }
    return preview;
}

PianoRollComponent::VibratoHandlePlace PianoRollComponent::vibratoHandlePlace(
    const NoteData& note, VibratoHandle which)
{
    const auto swing = vibratoSpanOf(note);
    const auto start = swing.start;
    const auto end = swing.end;
    const auto span = end - start;
    const auto cycle = std::max(0.01, note.vibratoCycleMs) / 1000.0;
    constexpr auto minimumHandleCents = 25.0;
    switch (which)
    {
        case VibratoHandle::length:
            return { start, 0.0, 0.0 };
        case VibratoHandle::end:
            return { end, 0.0, 0.0 };
        case VibratoHandle::offset:
            // Vertically it marks where the swing is centred, so dragging it
            // slides the whole vibrato up or down.
            return { 0.0, note.vibratoDepthCents * note.vibratoOffsetPercent / 100.0,
                     -12.0 };
        case VibratoHandle::fadeIn:
            return { start + span * juce::jlimit(0.0, 100.0,
                        note.vibratoFadeInPercent) / 100.0, 0.0, 0.0 };
        case VibratoHandle::fadeOut:
            return { end - span * juce::jlimit(0.0, 100.0,
                        note.vibratoFadeOutPercent) / 100.0, 0.0, 0.0 };
        case VibratoHandle::cycle:
            // On the first trough, and the depth handle on the first crest.
            // Both are pushed at least a quarter semitone off the centre line
            // even when the swing is shallower than that, because otherwise
            // they collapse onto the three handles that live on the line and
            // become impossible to tell apart, let alone grab.
            return { std::min(end, start + cycle * 0.75),
                     -std::max(note.vibratoDepthCents, minimumHandleCents), 0.0 };
        case VibratoHandle::depth:
            return { std::min(end, start + cycle * 0.25),
                     std::max(note.vibratoDepthCents, minimumHandleCents), 0.0 };
        case VibratoHandle::none: break;
    }
    return { start, 0.0, 0.0 };
}

juce::Point<float> PianoRollComponent::vibratoHandleCentre(const NoteData& shown,
                                                          VibratoHandle which,
                                                          double absoluteStart) const
{
    const auto place = vibratoHandlePlace(shown, which);
    const auto baseY = midiToY(shown.midiNote) + rowHeight * 0.5f;
    juce::Point<float> centre(
        timeToX(absoluteStart + place.timeSeconds) + static_cast<float>(place.pixelOffsetX),
        baseY - static_cast<float>(place.cents / 100.0) * rowHeight);
    // With little or no fade-out the fade-out handle sits right on the end:
    // two dots in one place, and only one of them could ever be taken hold
    // of.  The end steps just below the line there instead.
    if (which == VibratoHandle::end)
    {
        const auto fadeOut = vibratoHandlePlace(shown, VibratoHandle::fadeOut);
        const auto fadeOutX = timeToX(absoluteStart + fadeOut.timeSeconds);
        if (std::abs(fadeOutX - centre.x) < 9.0f) centre.y += 9.0f;
    }
    return centre;
}

int PianoRollComponent::diagnosticVibratoControlSetCount() const
{
    int count=0;
    for(const auto& track:snapshot.tracks)
        for(const auto& clip:track.clips)if(clipIsVisible(track,clip))
            for(const auto& note:clip.notes)if(note.vibratoEnabled&&ownsVibratoControls(note))++count;
    return count;
}

std::optional<juce::Point<float>> PianoRollComponent::diagnosticVibratoHandle(
    const juce::String& noteId, const juce::String& which) const
{
    const std::array<std::pair<const char*, VibratoHandle>, 7> names {{
        { "length", VibratoHandle::length }, { "end", VibratoHandle::end },
        { "fadeIn", VibratoHandle::fadeIn }, { "fadeOut", VibratoHandle::fadeOut },
        { "depth", VibratoHandle::depth }, { "cycle", VibratoHandle::cycle },
        { "offset", VibratoHandle::offset }
    }};
    auto handle = VibratoHandle::none;
    for (const auto& [name, value] : names)
        if (which == name) handle = value;
    if (handle == VibratoHandle::none) return std::nullopt;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == noteId && note.vibratoEnabled)
                {
                    const auto* group=joinedVibratoFor(note.id);
                    return vibratoHandleCentre(vibratoControlNote(note), handle,
                        group?group->start:clip.startSeconds+note.startSeconds);
                }
    return std::nullopt;
}

std::optional<PianoRollComponent::VibratoHandleInfo>
PianoRollComponent::vibratoHandleAt(juce::Point<float> position) const
{
    constexpr auto radius = 5.0f;
    if (sourceEditMode || tool == Tool::amplitude || tool == Tool::flagCurve)
        return std::nullopt;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose || (focusedTrack.isNotEmpty() && track.id != focusedTrack))
            continue;
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
            for (const auto& note : clip.notes)
            {
                if (!note.vibratoEnabled) continue;
                // Handles belong to the note being worked on; showing them for
                // every note at once would bury the roll in dots.
                if (!ownsVibratoControls(note)) continue;
                const auto shown = vibratoControlNote(note);
                const auto* group=joinedVibratoFor(note.id);
                const auto absoluteStart = group?group->start:clip.startSeconds + note.startSeconds;
                const std::array<VibratoHandle, 7> order {
                    VibratoHandle::offset, VibratoHandle::depth, VibratoHandle::cycle,
                    VibratoHandle::fadeIn, VibratoHandle::fadeOut, VibratoHandle::end,
                    VibratoHandle::length
                };
                for (const auto which : order)
                {
                    const auto centre = vibratoHandleCentre(shown, which, absoluteStart);
                    if (centre.getDistanceFrom(position) <= radius)
                        return VibratoHandleInfo { note.id, which, absoluteStart };
                }
            }
    }
    return std::nullopt;
}

float PianoRollComponent::diagnosticNoteY(std::size_t index) const
{
    std::size_t seen = 0;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (seen++ == index) return midiToY(note.midiNote) + rowHeight * 0.5f;
    return 0.0f;
}

std::array<double, 3> PianoRollComponent::diagnosticRegionEdges(std::size_t index) const
{
    std::array<double, 3> edges { -1.0, -1.0, -1.0 };
    std::size_t seen = 0;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (seen++ != index) continue;
                const auto span = utauSoundSpans.find(note.id.toStdString());
                if (span == utauSoundSpans.end()) return edges;
                const auto previewing = note.id == draggedNote
                    && dragMode == DragMode::consonantLeadIn;
                const auto resizing = note.id == draggedNote
                    && (dragMode == DragMode::resizeLeft
                        || dragMode == DragMode::resizeRight);
                const auto settledStart = clip.startSeconds + note.startSeconds;
                const auto absoluteStart = resizing
                    ? clip.startSeconds + previewStartSeconds : settledStart;
                const auto leadIn = settledStart - span->second.first;
                const auto settledEnd = settledStart + note.durationSeconds;
                const auto handover = span->second.second < settledEnd - 1.0e-9
                    ? span->second.second : std::numeric_limits<double>::max();
                const auto start = previewing
                    ? absoluteStart - previewConsonantPreutterance
                    : (resizing ? absoluteStart - leadIn : span->second.first);
                const auto finish = resizing
                    ? std::min(absoluteStart + previewDurationSeconds, handover)
                    : span->second.second;
                const auto length = finish - start;
                const auto fractions = jieFractionsFor(track, note, length,
                    previewing ? std::optional<double>(previewConsonantPreutterance)
                               : std::nullopt);
                if (!fractions) return edges;
                for (std::size_t boundary = 0; boundary < 3; ++boundary)
                    edges[boundary] = start + (*fractions)[boundary] * length;
                return edges;
            }
    return edges;
}

std::optional<std::array<double, 3>> PianoRollComponent::jieFractionsFor(
    const TrackData& track, const NoteData& note, double spanSeconds,
    std::optional<double> leadInOverride, int* regionsOut) const
{
    if (regionsOut != nullptr) *regionsOut = 4;
    if (trackIsDiffSinger(track) || !utauModeUsesRegions(track.utauMode) || spanSeconds <= 1.0e-9) return std::nullopt;
    const auto velocity = note.utauConsonantVelocity != inheritedUtauConsonantVelocity
        ? note.utauConsonantVelocity : track.utauConsonantVelocity;
    // 谋 annotates its entries, and the number of regions is one of the things
    // it says.  Read without that flag the roll never saw the annotation at
    // all and drew four boundaries into a note the entry splits in three.
    const auto mou = track.utauMode == UtauMode::mou;
    // Drawn on every paint: a bank still being read leaves the lines out for
    // now rather than holding the window until it is in.
    if (!voicebankReadFor(track)) return std::nullopt;
    const auto timing = backend::UtauRenderer::sampleTiming(
        track.voicebankDirectory, note.label, note.midiNote, velocity, true, mou,
        note.ownOto());
    // A 拼字 note in 界 has two, whatever its entry says: the renderer reads
    // nothing past the glide, and the lines drawn in it have to agree.
    const auto spelling = backend::UtauRenderer::readsOnlyFirstTwoRegions(true, mou,
        note.durationSeconds);
    const auto regions = spelling ? 2
        : timing && mou ? SampleSettings::mouRegionCount(timing->mouClasses) : 4;
    if (regionsOut != nullptr) *regionsOut = regions;
    // The lead-in this note really has.  A pinned one outranks the oto's, and
    // the block on screen is drawn against it -- read the oto's instead and
    // the first boundary lands wherever the two differ rather than on the note
    // start, which is the one place it is not allowed to leave.
    const auto settledLeadIn = timing
        ? (note.utauPreutteranceOverrideEnabled
               ? std::max(0.0, note.utauPreutteranceSeconds)
               : timing->preutteranceSeconds)
        : 0.0;
    if (note.utauJieSplitSet)
    {
        // The first line is not the split's to decide.  The onset ends where
        // the note starts, hand-placed boundaries or not, and that is exactly
        // what the renderer does with these fractions -- it reads the lead-in
        // and ignores the stored first one.  Drawing the stored value put the
        // consonant line somewhere the rendered consonant never was, and left
        // it there after a reset had moved the lead-in back.
        auto first = juce::jlimit(0.0, 1.0, note.utauJieSplit1);
        if (timing)
            first = juce::jlimit(0.0, 1.0, std::min(
                leadInOverride.value_or(settledLeadIn), spanSeconds) / spanSeconds);
        // Only the boundaries this entry has: the rest sit at the end of the
        // note, where nothing draws them and nothing can take hold of them.
        std::array<double, 3> hand { 1.0, 1.0, 1.0 };
        hand[0] = first;
        auto previous = first;
        const std::array<double, 3> stored { first, note.utauJieSplit2,
                                             note.utauJieSplit3 };
        for (int index = 1; index + 1 < regions; ++index)
        {
            previous = juce::jlimit(previous, 1.0,
                                    stored[static_cast<std::size_t>(index)]);
            hand[static_cast<std::size_t>(index)] = previous;
        }
        return hand;
    }
    if (!timing || !timing->hasRegions) return std::nullopt;
    // Same allocation the renderer will run, so the lines start where the
    // regions actually are rather than at some arbitrary default.
    // The sounding span begins a lead-in before the note, so that lead-in is
    // what the onset gets: it has to end where the note starts.  While one is
    // being dragged the block is already drawn against the lead-in the drag is
    // proposing, and the boundaries have to agree with the block they sit in
    // -- read from the settled timing instead, the consonant slid off the note
    // start for as long as the drag lasted, furthest when the drag was
    // lengthening it.
    const auto leadIn = std::min(leadInOverride.value_or(settledLeadIn), spanSeconds);
    // The same classes the renderer will use, so the lines on the roll are
    // where the regions really end up.
    const auto sourceRegions = spelling
        ? backend::UtauRenderer::firstTwoRegions(timing->regionSeconds)
        : timing->regionSeconds;
    const auto split = backend::UtauRenderer::regionSplit(
        sourceRegions, spanSeconds, velocity, leadIn, nullptr,
        mou && timing->mouClasses.isNotEmpty() ? &timing->mouClasses : nullptr);
    if (!split.valid) return std::nullopt;
    std::array<double, 3> fractions {};
    auto cumulative = 0.0;
    for (std::size_t index = 0; index < 3; ++index)
    {
        cumulative += split.seconds[index];
        fractions[index] = juce::jlimit(0.0, 1.0, cumulative / spanSeconds);
    }
    return fractions;
}

int PianoRollComponent::diagnosticRegionCount(std::size_t index) const
{
    std::size_t seen = 0;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (seen++ != index) continue;
                const auto span = utauSoundSpans.find(note.id.toStdString());
                if (span == utauSoundSpans.end()) return 4;
                auto regions = 4;
                jieFractionsFor(track, note, span->second.second - span->second.first,
                                std::nullopt, &regions);
                return regions;
            }
    return 4;
}

std::optional<PianoRollComponent::JieSplitHandleInfo>
PianoRollComponent::jieSplitHandleAt(juce::Point<float> position) const
{
    constexpr auto handleRadius = 5.0f;
    if (tool != Tool::note || sourceEditMode) return std::nullopt;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose || !trackUsesVoicebankSynthesis(track)
            || !utauModeUsesRegions(track.utauMode) || !track.voicebankDirectory.isDirectory()
            || (focusedTrack.isNotEmpty() && track.id != focusedTrack))
            continue;
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
            for (const auto& note : clip.notes)
            {
                const auto span = utauSoundSpans.find(note.id.toStdString());
                if (span == utauSoundSpans.end()) continue;
                const auto spanStart = span->second.first;
                const auto spanEnd = span->second.second;
                const auto spanSeconds = spanEnd - spanStart;
                auto regions = 4;
                const auto fractions = jieFractionsFor(track, note, spanSeconds,
                                                       std::nullopt, &regions);
                if (!fractions) continue;
                const auto top = midiToY(note.midiNote) + 2.0f;
                const auto height = std::max(6.0f, rowHeight - 4.0f);
                // The first line is where the oto says the consonant ends,
                // and it no longer moves with the note: dragging it would
                // only snap back.  Per note, the consonant's length is set by
                // the consonant velocity, and in the voicebank by the region
                // editor.
                for (int boundary = regions - 2; boundary >= 1; --boundary)
                {
                    const auto x = timeToX(spanStart
                        + (*fractions)[static_cast<std::size_t>(boundary)] * spanSeconds);
                    const juce::Rectangle<float> hit(x - handleRadius, top,
                                                     handleRadius * 2.0f, height);
                    if (!hit.contains(position)) continue;
                    return JieSplitHandleInfo { note.id, boundary, spanStart, spanEnd,
                                                *fractions, regions };
                }
            }
    }
    return std::nullopt;
}

std::optional<PianoRollComponent::ConsonantHandleInfo>
PianoRollComponent::consonantHandleAt(juce::Point<float> position) const
{
    constexpr auto handleRadius = 7.0f;
    std::optional<ConsonantHandleInfo> best;
    auto bestDistance = std::numeric_limits<float>::max();
    for (const auto& track : snapshot.tracks)
    {
        if (trackIsDiffSinger(track) || !track.compose || !trackUsesVoicebankSynthesis(track)
            || !track.voicebankDirectory.isDirectory()
            || (focusedTrack.isNotEmpty() && track.id != focusedTrack))
            continue;
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
            for (const auto& note : clip.notes)
            {
                const auto span = utauSoundSpans.find(note.id.toStdString());
                if (span == utauSoundSpans.end()) continue;
                const auto absoluteStart = clip.startSeconds + note.startSeconds;
                const auto handle = juce::Rectangle<float>(
                    timeToX(span->second.first) - handleRadius,
                    midiToY(note.midiNote) + 2.0f,
                    handleRadius * 2.0f, std::max(6.0f, rowHeight - 4.0f));
                if (!handle.contains(position)) continue;
                // Nearest wins where two handles overlap.  A 拼字 note sits on
                // the beat of the note it leads into and is listed first, so
                // taking the first handle that contained the pointer handed
                // out its handle every time, and the note behind it could not
                // be taken hold of at all.  On an exact tie the later note
                // wins: the one listed first is the one put in front of it.
                const auto distance = std::abs(position.x
                                               - timeToX(span->second.first));
                if (distance > bestDistance) continue;
                if (!voicebankReadFor(track)) continue;
                const auto currentVelocity =
                    note.utauConsonantVelocity != inheritedUtauConsonantVelocity
                    ? note.utauConsonantVelocity : track.utauConsonantVelocity;
                if (track.chineseCvvc && track.utauMode == UtauMode::classic)
                {
                    const auto pieces = diffSingerPhonemes.find(note.id.toStdString());
                    if (pieces == diffSingerPhonemes.end() || pieces->second.empty()) continue;
                    struct PieceTiming { backend::UtauSampleTiming timing; double offset; int part; };
                    std::vector<PieceTiming> timings;
                    std::optional<backend::UtauSampleTiming> cv;
                    std::optional<double> previousDuration;
                    auto vcMidi = note.midiNote;
                    for (const auto& previous : clip.notes)
                        if (previous.id != note.id && !backend::isRestLyric(previous.label)
                            && previous.label.isNotEmpty()
                            && std::abs(previous.startSeconds + previous.durationSeconds - note.startSeconds) < .002)
                        { previousDuration = previous.durationSeconds; vcMidi = previous.midiNote; }
                    for (const auto& piece : pieces->second)
                    {
                        auto timing = backend::UtauRenderer::mappedAliasTiming(track.voicebankDirectory,
                            piece.token, piece.kind == "VC" ? vcMidi : note.midiNote);
                        if (!timing) continue;
                        if (piece.kind == "CV") cv = timing;
                        if ((piece.startSeconds==0 || piece.kind=="CV" || piece.kind=="VCV" || piece.kind=="manual") && note.ownOto())
                            timing = backend::UtauRenderer::sampleTiming(track.voicebankDirectory,
                                piece.token, note.midiNote, 100, false, false, note.ownOto());
                        if (timing) timings.push_back({*timing, piece.startSeconds, piece.envelopePart});
                    }
                    if (timings.empty()) continue;
                    const auto beat = 60.0 / std::max(1.0, snapshot.tempoAtSeconds(absoluteStart));
                    const auto lead = [timings=std::move(timings), cv, previousDuration, beat,
                        start=note.startSeconds, duration=note.durationSeconds, end=pieces->second.back().endSeconds,
                        boundaries=note.cvvcBoundaries](int velocity)
                    {
                        double result = 0;
                        std::vector<double> starts;std::vector<int> parts;
                        for (const auto& piece : timings)
                        {
                            starts.push_back(piece.part==1 && cv
                                ? -backend::cvvcTransitionDuration(*cv,beat,previousDuration,velocity) : piece.offset);
                            parts.push_back(piece.part);
                        }
                        backend::applyCvvcBoundaries(starts,parts,duration,end,boundaries);
                        for(size_t i=0;i<timings.size();++i)
                            result=std::max(result,backend::scaledUtauPreutterance(timings[i].timing,velocity)-starts[i]);
                        return std::min(std::max(0.0, start), result);
                    };
                    const auto minimum = lead(102300), maximum = lead(-102200);
                    if (maximum - minimum <= 1.e-5) continue;
                    bestDistance = distance;
                    best = ConsonantHandleInfo {note.id, absoluteStart, 0, 0, minimum, maximum,
                        absoluteStart - span->second.first, currentVelocity, false, 0, lead};
                    continue;
                }
                const auto base = backend::UtauRenderer::sampleTiming(
                    track.voicebankDirectory, note.label, note.midiNote, 100,
                    utauModeUsesRegions(track.utauMode),
                    track.utauMode == UtauMode::mou, note.ownOto());
                const auto current = backend::UtauRenderer::sampleTiming(
                    track.voicebankDirectory, note.label, note.midiNote,
                    currentVelocity, utauModeUsesRegions(track.utauMode),
                    track.utauMode == UtauMode::mou, note.ownOto());
                const auto slowest = backend::UtauRenderer::sampleTiming(
                    track.voicebankDirectory, note.label, note.midiNote,
                    std::numeric_limits<int>::min() + 1,
                    utauModeUsesRegions(track.utauMode),
                    track.utauMode == UtauMode::mou, note.ownOto());
                if (!base || !current || !slowest) continue;
                // A first region 谋 does not call a consonant: the consonant
                // velocity does not reach it, so a drag converted into one
                // moved nothing at all and rewrote a setting the note carries
                // for nothing.  Such a note has its lead-in set directly.  One
                // 谋 calls a consonant is a consonant, and goes on speaking
                // through the velocity as it does in 界.
                const auto mou = track.utauMode == UtauMode::mou;
                const auto firstIsConsonant = base->mouClasses.isEmpty()
                    || base->mouClasses[0] != 'V';
                const auto freeLeadIn = mou && !firstIsConsonant;
                const auto scaledPart = std::min(base->preutteranceSeconds,
                                                  base->consonantSeconds);
                const auto unscaledPart = std::max(0.0,
                    base->preutteranceSeconds - base->consonantSeconds);
                if (!freeLeadIn && scaledPart <= 1.0e-6) continue;
                // Approaching the unscaled part corresponds to increasingly
                // large velocity values.  A tiny positive scale keeps log2
                // finite while removing legacy UTAU's artificial 200 ceiling.
                auto minimum = std::min(absoluteStart,
                    unscaledPart + scaledPart * 1.0e-12);
                // Negative velocities lengthen the consonant.  The latest
                // possible handle is limited by the sample's available audio
                // and by the beginning of the project, not by velocity zero.
                auto maximum = std::min(absoluteStart,
                    slowest->preutteranceSeconds);
                if (freeLeadIn)
                {
                    // Free between the note in front of it and its own start,
                    // and no further back than the entry has audio to give.
                    auto previousStart = 0.0;
                    for (const auto& other : clip.notes)
                    {
                        const auto start = clip.startSeconds + other.startSeconds;
                        if (start < absoluteStart - 1.0e-9)
                            previousStart = std::max(previousStart, start);
                    }
                    minimum = 0.0;
                    maximum = std::min({ absoluteStart, absoluteStart - previousStart,
                                         base->sampleSeconds });
                }
                if (maximum - minimum <= 1.0e-5) continue;
                const auto overlap = note.utauOverlapOverrideEnabled
                    ? note.utauOverlapSeconds : current->overlapSeconds;
                // A pinned lead-in is where the handle is now, not what the
                // oto would have given it.
                const auto settled = note.utauPreutteranceOverrideEnabled
                    ? std::max(0.0, note.utauPreutteranceSeconds)
                    : current->preutteranceSeconds;
                bestDistance = distance;
                best = ConsonantHandleInfo {
                    note.id, absoluteStart, scaledPart, unscaledPart,
                    minimum, maximum,
                    juce::jlimit(minimum, maximum, settled),
                    currentVelocity, freeLeadIn, overlap
                };
            }
    }
    return best;
}

int PianoRollComponent::noteEditDivision() const
{
    // 1/N of a beat.  The default 64 is a quarter of the 1/64-bar unit this
    // started out as, which in 4/4 is the 1/256-bar quantum editing has always
    // used -- the arithmetic below keeps that true in every time signature.
    return juce::jlimit(2, 128, snapshot.noteEditDivision);
}

double PianoRollComponent::noteEditQuantumSeconds() const
{
    const auto beatSeconds = 60.0 / juce::jlimit(20.0, 400.0,
                                                snapshot.tempoAtSeconds(playheadSeconds));
    const auto barSeconds = beatSeconds * static_cast<double>(snapshot.numerator)
        * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
    return std::max(0.001, barSeconds / (4.0 * static_cast<double>(noteEditDivision())));
}

double PianoRollComponent::nativeEdgeDragDelta(double originalSeconds, double travelSeconds, bool free) const
{
    if (free || std::abs(travelSeconds) < 1.0e-9) return travelSeconds;
    const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
        * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
    const auto quantum = barQuarters / (4.0 * static_cast<double>(noteEditDivision()));
    const auto originalQuarter = snapshot.quarterPositionForSeconds(originalSeconds);
    const auto travelQuarter = snapshot.quarterPositionForSeconds(originalSeconds + travelSeconds) - originalQuarter;
    return snapshot.secondsForQuarterPosition(originalQuarter + std::round(travelQuarter / quantum) * quantum)
        - originalSeconds;
}

void PianoRollComponent::beginInlineAliasEdit(const NoteHit& hit)
{
    auto currentAlias = juce::String{};
    auto bounds = hit.bounds;
    for (const auto& track : snapshot.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (note.id == hit.id)
                {
                    currentAlias = note.label;
                    bounds = nativeWaveformLabelBounds(track, clip, note, bounds);
                }
    const auto aboveWaveform = bounds.getY() != hit.bounds.getY();
    const auto labelBottom = bounds.getBottom();

    ++inlineAliasGeneration;inlineAliasSuspended=false;suspendedAliasNoteId.clear();
    inlineAliasNoteId = hit.id;
    inlineAliasEditor.setText(currentAlias, false);
    bounds.setWidth(std::max(84.0f, bounds.getWidth()));
    bounds.setHeight(std::max(24.0f, bounds.getHeight()));
    if (aboveWaveform) bounds.setY(labelBottom - bounds.getHeight());
    if (bounds.getRight() > static_cast<float>(getWidth()) - 2.0f)
        bounds.setX(std::max(60.0f, static_cast<float>(getWidth()) - bounds.getWidth() - 2.0f));
    inlineAliasEditor.setBounds(bounds.getSmallestIntegerContainer().expanded(1, 1));
    inlineAliasEditor.setVisible(true);
    inlineAliasEditor.toFront(false);
    inlineAliasEditor.grabKeyboardFocus();
    inlineAliasEditor.selectAll();
}

bool PianoRollComponent::diagnosticBeginAliasEdit(const juce::String& noteId)
{
    if(!findNote(noteId))return false;
    for(const auto& hit:noteHits)if(hit.id==noteId){beginInlineAliasEdit(hit);return inlineAliasNoteId==noteId&&inlineAliasEditor.isVisible();}
    return false;
}

bool PianoRollComponent::suspendInlineAliasInput(const juce::String& noteId)
{
    if(noteId.isEmpty()||inlineAliasNoteId!=noteId||!inlineAliasEditor.isVisible()||!findNote(noteId))return false;
    inlineAliasSuspended=true;suspendedAliasNoteId=noteId;suspendedAliasText=inlineAliasEditor.getText();
    suspendedAliasGeneration=inlineAliasGeneration;suspendedAliasRevision=model.revisionNumber();return true;
}
void PianoRollComponent::resumeInlineAliasInput(const juce::String& noteId,bool discard)
{
    if(!inlineAliasSuspended||suspendedAliasNoteId!=noteId)return;
    const bool same=inlineAliasNoteId==noteId&&inlineAliasGeneration==suspendedAliasGeneration&&inlineAliasEditor.isVisible();
    const bool valid=same&&model.revisionNumber()==suspendedAliasRevision&&findNote(noteId);
    const auto text=suspendedAliasText;inlineAliasSuspended=false;suspendedAliasNoteId.clear();
    if(!same)return;
    if(discard||!valid){finishInlineAliasEdit(false);return;}
    inlineAliasEditor.setText(text,false);if(inlineAliasEditor.isShowing())inlineAliasEditor.grabKeyboardFocus();repaint();
}

void PianoRollComponent::finishInlineAliasEdit(bool accept)
{
    if (inlineAliasNoteId.isEmpty() || (accept&&inlineAliasSuspended)) return;
    inlineAliasSuspended=false;suspendedAliasNoteId.clear();
    const auto noteId = inlineAliasNoteId;
    inlineAliasNoteId.clear();
    const auto alias = inlineAliasEditor.getText();
    inlineAliasEditor.setVisible(false);
    if (accept) commitInlineAlias(noteId, alias);
    repaint();
}

void PianoRollComponent::commitInlineAlias(const juce::String& noteId, juce::String alias)
{
    alias = alias.trim();
    if (alias.endsWithIgnoreCase(".wav"))
        alias = alias.dropLastCharacters(4).trim();
    model.setNoteLabel(noteId, alias);
    ensureDefaultEnvelope(noteId);
    if (onNoteAliasCommitted) onNoteAliasCommitted(noteId);
    if (onNoteSelected) onNoteSelected(noteId);
}

juce::String PianoRollComponent::adjacentNoteOnTrack(const juce::String& noteId,
                                                     bool forward) const
{
    struct Placed
    {
        double startSeconds = 0.0;
        juce::String id;
    };
    for (const auto& track : snapshot.tracks)
    {
        std::vector<Placed> order;
        auto owns = false;
        for (const auto& clip : track.clips)
            if (clipIsVisible(track, clip))
            for (const auto& note : clip.notes)
            {
                order.push_back({ clip.startSeconds + note.startSeconds, note.id });
                owns = owns || note.id == noteId;
            }
        if (!owns) continue;
        // Keyboard lyric navigation stays inside the displayed region.
        std::stable_sort(order.begin(), order.end(), [](const Placed& left, const Placed& right)
        {
            return left.startSeconds < right.startSeconds;
        });
        const auto at = std::find_if(order.begin(), order.end(),
            [&noteId](const Placed& placed) { return placed.id == noteId; });
        if (forward)
            return std::next(at) == order.end() ? juce::String() : std::next(at)->id;
        return at == order.begin() ? juce::String() : std::prev(at)->id;
    }
    return {};
}

void PianoRollComponent::revealInViewport(juce::Rectangle<float> area)
{
    auto* viewport = findParentComponentOfClass<juce::Viewport>();
    if (viewport == nullptr) return;
    const auto view = viewport->getViewArea();
    const auto box = area.getSmallestIntegerContainer();
    auto x = view.getX();
    auto y = view.getY();
    // The keyboard strip covers the left of whatever is in view, and only the
    // start of the box has to show -- that is where the typing goes.
    constexpr auto keyboardWidth = 58;
    constexpr auto boxWidth = 84;
    if (box.getX() < view.getX() + keyboardWidth || box.getX() + boxWidth > view.getRight())
        x = box.getX() - keyboardWidth - view.getWidth() / 4;
    if (box.getY() < view.getY() || box.getBottom() > view.getBottom())
        y = box.getCentreY() - view.getHeight() / 2;
    if (x != view.getX() || y != view.getY())
        viewport->setViewPosition(std::max(0, x), std::max(0, y));
}

void PianoRollComponent::advanceInlineAliasEdit(bool forward)
{
    if (inlineAliasNoteId.isEmpty()) return;
    const auto noteId = inlineAliasNoteId;
    const auto nextId = adjacentNoteOnTrack(noteId, forward);
    if (nextId.isEmpty())
    {
        // Past the end of the track there is nowhere to go on to, so Tab
        // ends it the way Return does.
        finishInlineAliasEdit(true);
        return;
    }
    // The box stays up the whole way through.  Hidden even for a moment it
    // would lose focus, and the editor reports that later, by message -- by
    // when it would be sitting on the next note, and would close itself there.
    inlineAliasNoteId.clear();
    commitInlineAlias(noteId, inlineAliasEditor.getText());
    const auto found = std::find_if(noteHits.begin(), noteHits.end(),
        [&nextId](const NoteHit& hit) { return hit.id == nextId; });
    if (found == noteHits.end())
    {
        inlineAliasEditor.setVisible(false);
        repaint();
        return;
    }
    // A copy: the window answers a selection by refocusing the track, which
    // rebuilds the hit list underneath any reference into it.
    const auto target = *found;
    selectedNote = nextId;
    selectedNotes.clear();
    selectedNotes.insert(nextId.toStdString());
    if (onNoteSelected) onNoteSelected(nextId);
    revealInViewport(target.bounds);
    beginInlineAliasEdit(target);
    repaint();
}

void PianoRollComponent::lookAndFeelChanged()
{
    // The inline alias editor caches Palette colours; re-apply them so a theme
    // switch does not leave it on the previous theme's (unreadable) colours.
    inlineAliasEditor.setColour(juce::TextEditor::backgroundColourId,
                                Palette::panelRaised.withAlpha(0.98f));
    inlineAliasEditor.setColour(juce::TextEditor::textColourId, Palette::text);
    inlineAliasEditor.setColour(juce::TextEditor::outlineColourId, Palette::accentLight);
    inlineAliasEditor.setColour(juce::TextEditor::focusedOutlineColourId, Palette::noteLight);
    repaint();
}

void PianoRollComponent::drawNoteHints(juce::Graphics& g) const
{
    // Source-file editing has a different time axis; project references do
    // not belong there. Hints are paint only and never enter the hit list.
    if (sourceEditMode) return;
    const auto visible = g.getClipBounds().toFloat();
    for (std::size_t index = 0; index < snapshot.tracks.size(); ++index)
    {
        const auto& track = snapshot.tracks[index];
        if (!track.compose || track.accompaniment) continue;
        const auto colour = Palette::trackColour(index);
        for (const auto& clip : track.clips)
        {
            if (!clip.showNoteHints || clipIsVisible(track, clip)) continue;
            for (const auto& note : clip.notes)
            {
                const auto start = clip.startSeconds + note.startSeconds;
                if (!std::isfinite(start) || !std::isfinite(note.durationSeconds)
                    || !std::isfinite(note.midiNote) || note.durationSeconds <= 0.0) continue;
                const juce::Rectangle<float> bounds(timeToX(start), midiToY(note.midiNote) + 2.0f,
                    std::max(2.0f, static_cast<float>(note.durationSeconds) * pixelsPerSecond),
                    std::max(6.0f, rowHeight - 4.0f));
                if (!bounds.intersects(visible)) continue;
                g.setColour(colour.withAlpha(0.20f));
                g.fillRoundedRectangle(bounds, 2.0f);
                g.setColour(colour.withAlpha(0.38f));
                g.drawRoundedRectangle(bounds.reduced(0.5f), 2.0f, 1.0f);
            }
        }
    }
}

void PianoRollComponent::paint(juce::Graphics& g)
{
    const auto pitchSelectionCurrent = selectedPoint && selectedPoint->kind == PointSelection::pitch
        && selectedPoint->revision == model.revisionNumber();
    // Preview the exact same source-clock warp that mouse-up commits. All
    // layers, including waveform and measured F0, see these temporary notes.
    struct RestoreSnapshot
    {
        ProjectData& target;
        std::optional<ProjectData> original;
        ~RestoreSnapshot() { if (original) std::swap(target,*original); }
    } restore {snapshot, {}};
    if (!sourceEditMode && (dragMode == DragMode::moveNativeNote
        || dragMode == DragMode::resizeNativeLeft || dragMode == DragMode::resizeNativeRight))
    {
        restore.original = snapshot;
        std::swap(snapshot,*restore.original); // retain the original cached clip addresses
        snapshot=model.snapshot(); // edit linked parents before expanding source children
        std::vector<juce::String> ids;
        const auto edit=dragMode==DragMode::resizeNativeLeft?NativeNoteTimeEdit::leftEdge
            :dragMode==DragMode::resizeNativeRight?NativeNoteTimeEdit::rightEdge:NativeNoteTimeEdit::move;
        if(edit==NativeNoteTimeEdit::move)
            for (const auto& id:selectedNotes) ids.push_back(juce::String::fromUTF8(id.c_str()));
        else ids.push_back(draggedNote);
        auto delta=edit==NativeNoteTimeEdit::leftEdge?previewStartSeconds-dragStartSeconds
            :edit==NativeNoteTimeEdit::rightEdge?previewDurationSeconds-dragDurationSeconds:previewMoveDeltaSeconds;
        const auto pitch=edit==NativeNoteTimeEdit::move?previewMidi-dragStartMidi:0.f;
        if (sharedBoundaryNotes.size()==2)
        {
            for(auto& track:snapshot.tracks)
                if(auto plan=planNativeBoundaryStretch(track,sharedBoundaryNotes[0],sharedBoundaryNotes[1],delta))
                    for(auto& item:plan->clips)for(auto& clip:track.clips)
                        if(clip.id==item.clip.id){clip=std::move(item.clip);break;}
        }
        else if (nativeTrimDrag)
        {
            for (auto& track : snapshot.tracks) if (trackShowsAllNativeRegions(track))
                for (auto& clip : track.clips)
                    if (auto plan = planNativeNoteTrim(clip, draggedNote,
                        constrainedNativeTrimDelta(track,draggedNote,delta,edit == NativeNoteTimeEdit::leftEdge),
                        edit == NativeNoteTimeEdit::leftEdge,track.allowNativeAudioOverlap))
                        clip = std::move(plan->clip);
        }
        else
        {
        for (const auto& track:snapshot.tracks) if(trackShowsAllNativeRegions(track))
            for(const auto& clip:track.clips) if(auto plan=planNativeNoteMove(clip,ids,delta,0,edit))
                delta=delta<0?std::max(delta,plan->delta):std::min(delta,plan->delta);
        for(const auto& track:snapshot.tracks)
        {const auto safe=constrainedNativeNoteDelta(track,ids,delta,edit);delta=delta<0?std::max(delta,safe):std::min(delta,safe);}
        for(auto& track:snapshot.tracks) if(trackShowsAllNativeRegions(track))
            for(auto& clip:track.clips) if(auto plan=planNativeNoteMove(clip,ids,delta,pitch,edit))
                clip=std::move(plan->clip);
        }
        expandProjectClipParts(snapshot,true);
        // The preview has a temporary source clock; never reuse its mask for
        // the restored snapshot or a later drag position.
        nativePitchVisibilityCache.clear(); nativePitchAuthoredCache.clear();
    }

    struct ClearPreviewPitchMask
    {
        std::map<juce::String, NativeNoiseRanges>& cache;
        std::map<juce::String, NativeNoiseRanges>& authored;
        bool preview;
        ~ClearPreviewPitchMask() { if (preview) { cache.clear(); authored.clear(); } }
    } clearPreviewPitchMask {nativePitchVisibilityCache, nativePitchAuthoredCache, restore.original.has_value()};

    // Only the exposed strip is worth drawing.  At ordinary zoom the canvas
    // runs twenty times the width of the window, so anything that walked the
    // whole piece -- notes, grid lines, bar marks -- spent almost all of its
    // time on pixels the clip then threw away.
    const auto visibleArea = g.getClipBounds().toFloat();

    struct Dancer { juce::Rectangle<float> bounds; double start; juce::String id; };
    std::vector<Dancer> dancers;

    for (int midi = lowestMidi; midi <= highestMidi; ++midi)
    {
        const auto y = midiToY(static_cast<float>(midi));
        const auto black = juce::MidiMessage::isMidiNoteBlack(midi);
        if (black)
        {
            g.setColour(juce::Colours::black.withAlpha(0.16f));
            g.fillRect(0.0f, y, static_cast<float>(getWidth()), rowHeight);
        }
        g.setColour(Palette::grid);
        g.drawHorizontalLine(static_cast<int>(y), 0.0f, static_cast<float>(getWidth()));
    }

    auto keyboardX = 0;
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
        keyboardX = viewport->getViewPositionX();
    const auto drawKeyboard = [&]
    {
        g.setColour(Palette::panelRaised);
        g.fillRect(keyboardX, 0, 58, getHeight());
        for (int midi = lowestMidi; midi <= highestMidi; ++midi)
        {
            const auto y = midiToY(static_cast<float>(midi));
            if (juce::MidiMessage::isMidiNoteBlack(midi))
            {
                g.setColour(juce::Colours::black.withAlpha(0.55f));
                g.fillRect(static_cast<float>(keyboardX), y, 38.0f, rowHeight);
            }
            if (keyboardLabelVisible(midi, rowHeight))
            {
                // A sharp goes on the black key itself, where there is width
                // for three characters; a natural goes on the pale strip to
                // the right of them.  C stays brighter than the rest so an
                // octave can still be found at a glance.
                const auto black = juce::MidiMessage::isMidiNoteBlack(midi);
                const auto keySurface = black
                    ? Palette::panelRaised.overlaidWith(juce::Colours::black.withAlpha(0.55f))
                    : Palette::panelRaised;
                g.setColour(rollText(midi % 12 == 0 ? Palette::text
                           : black ? Palette::textMuted.brighter(0.40f)
                                   : Palette::textMuted, keySurface, midi % 12 != 0));
                g.setFont(juce::jlimit(7.0f, 10.0f, rowHeight - 2.0f));
                const auto name = juce::MidiMessage::getMidiNoteName(midi, true, true, 4);
                if (black)
                    g.drawText(name, keyboardX + 3, static_cast<int>(y),
                               32, static_cast<int>(rowHeight),
                               juce::Justification::centredRight);
                else
                    g.drawText(name, keyboardX + 39, static_cast<int>(y),
                               18, static_cast<int>(rowHeight),
                               juce::Justification::centred);
            }
        }
    };

    const auto gridStep = gridQuarterNotes();
    const auto firstVisibleSeconds = std::max(0.0,
        static_cast<double>(visibleArea.getX() - 58.0f) / pixelsPerSecond);
    const auto firstQuarter = snapshot.quarterPositionForSeconds(firstVisibleSeconds);
    const auto firstTick = std::floor(firstQuarter / gridStep) - 1.0;
    const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
        * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
    for (double tick = firstTick;; tick += 1.0)
    {
        const auto quarter = tick * gridStep;
        const auto seconds = snapshot.secondsForQuarterPosition(quarter);
        const auto x = timeToX(seconds);
        if (x > visibleArea.getRight() || x > static_cast<float>(getWidth())) break;
        if (x < 58.0f) continue;
        const auto isBeat = std::abs(quarter - std::round(quarter)) < 1.0e-6;
        const auto isBar = std::abs(quarter / barQuarters
                                    - std::round(quarter / barQuarters)) < 1.0e-6;
        g.setColour(isBar ? Palette::beatGrid.brighter(0.35f)
                   : isBeat ? Palette::beatGrid.withAlpha(0.60f) : Palette::beatGrid.withAlpha(0.30f));
        g.drawVerticalLine(static_cast<int>(x), 0.0f, static_cast<float>(getHeight()));
    }

    for (const auto& change : snapshot.tempoChanges)
    {
        const auto x = timeToX(snapshot.secondsForQuarterPosition(change.quarterPosition));
        if (x < 58.0f || x > getWidth()) continue;
        g.setColour(rollRole(juce::Colour(0xffffa94d), Palette::heading).withAlpha(0.72f));
        g.drawVerticalLine(static_cast<int>(x), 0.0f, static_cast<float>(getHeight()));
    }

    drawNoteHints(g);
    if (showWaveforms) drawClipWaveforms(g);
    if (showNativeWaveforms || showNativeRenderedWaveforms || nativeLoudnessMode) drawNativeNoteWaveforms(g);
    if (nativeLoudnessMode)
        for (const auto& hit : noteHits)
            if (hit.id == selectedNote || selectedNotes.contains(hit.id.toStdString()))
                if (const auto* note = findNote(hit.id))
                {
                    const auto gain = displayedNativeGain(*note);
                    const auto db = juce::Decibels::gainToDecibels(gain, -60.0f);
                    const auto text = gain <= 0 ? juce::String("-inf dB") : (db > 0 ? "+" : "") + juce::String(db, 1) + " dB";
                    auto label = juce::Rectangle<float>(hit.bounds.getX(), hit.bounds.getY() - 23, 84, 21);
                    g.setColour(Palette::background.withAlpha(.92f)); g.fillRoundedRectangle(label, 3);
                    g.setColour(Palette::text); g.drawText(text, label, juce::Justification::centred);
                }
    if (showUtauWaveforms) drawUtauNoteWaveforms(g);

    // In wrench mode all timing edits are made against the untouched source.
    // The four handles mirror the main-branch HJM editor and intentionally sit
    // above the note hit targets so a boundary can always be grabbed.
    if (sourceEditMode)
    {
        for (int index = 0; index < static_cast<int>(sampleRegions.size()); ++index)
        {
            const auto& region = sampleRegions[static_cast<std::size_t>(index)];
            const auto selected = index == activeSampleRegion;
            const auto startX = timeToX(region.regionStartSeconds);
            const auto endX = timeToX(region.regionEndSeconds);
            const auto fixedX = timeToX(region.regionStartSeconds + region.fixedDurationSeconds);
            const auto alignmentX = timeToX(region.alignmentSeconds);
            const auto band = juce::Rectangle<float>(startX, 4.0f,
                std::max(1.0f, endX - startX), static_cast<float>(getHeight() - 8));
            g.setColour(Palette::accentLight.withAlpha(selected ? 0.105f : 0.035f));
            g.fillRect(band);

            const auto drawHandle = [&](float x, RegionHandle handle, juce::Colour colour,
                                        float thickness, bool dashed)
            {
                juce::Path path;
                path.startNewSubPath(x, 4.0f);
                path.lineTo(x, static_cast<float>(getHeight() - 4));
                g.setColour(colour.withAlpha(selected ? 0.94f : 0.48f));
                if (dashed)
                {
                    const float dashes[] { 5.0f, 4.0f };
                    juce::Path dashedPath;
                    juce::PathStrokeType(thickness).createDashedStroke(dashedPath, path, dashes, 2);
                    g.fillPath(dashedPath);
                }
                else g.strokePath(path, juce::PathStrokeType(thickness));
                regionHandleHits.push_back({ index, handle, x });
            };
            drawHandle(startX, RegionHandle::start, Palette::accentLight, selected ? 2.2f : 1.0f, false);
            if (region.fixedDurationSeconds > 0.0)
                drawHandle(fixedX, RegionHandle::fixedEnd, Palette::noteLight,
                           selected ? 2.0f : 1.0f, true);
            drawHandle(alignmentX, RegionHandle::alignment, Palette::noteFill,
                       selected ? 2.4f : 1.2f, false);
            drawHandle(endX, RegionHandle::end, Palette::accentLight,
                       selected ? 2.2f : 1.0f, false);

            if (selected)
            {
                const auto label = region.name.isEmpty() ? "region " + juce::String(index + 1)
                                                         : region.name;
                const auto labelBounds = juce::Rectangle<float>(startX + 3.0f, 6.0f,
                    std::max(40.0f, endX - startX - 6.0f), 18.0f);
                g.setColour(Palette::panelRaised.withAlpha(0.88f));
                g.fillRoundedRectangle(labelBounds, 3.0f);
                g.setColour(Palette::text);
                g.setFont(11.0f);
                g.drawFittedText(label, labelBounds.toNearestInt().reduced(4, 0),
                                 juce::Justification::centredLeft, 1);
            }
        }
    }

    const std::array<juce::Colour, 5> contourColours {
        Palette::accentLight, Palette::accent, Palette::noteFill,
        rollRole(juce::Colour(0xff45b8aa), Palette::noteEdge), rollRole(juce::Colour(0xffad7ad6), Palette::accent)
    };
    struct PositionedNote
    {
        juce::String id;
        int midiRow = 0;
        double start = 0.0;
        double end = 0.0;
    };
    std::vector<PositionedNote> visibleNotes;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose && !sourceEditMode) continue;
        if (!sourceEditMode && focusedTrack.isNotEmpty() && track.id != focusedTrack) continue;
        for (const auto& clip : track.clips)
        {
            if (!clipIsVisible(track, clip)) continue;
            const auto sourceScale = sourceEditMode && clip.audioLength() > 1.0e-9
                ? (clip.sourceDurationSeconds > 1.0e-9 ? clip.sourceDurationSeconds : clip.audioLength())
                    / clip.audioLength() : 1.0;
            for (const auto& note : clip.notes)
            {
                const auto start = sourceEditMode
                    ? clip.sourceOffsetSeconds + (note.startSeconds - clip.audioStartSeconds) * sourceScale
                    : clip.startSeconds + note.startSeconds;
                visibleNotes.push_back({ note.id, static_cast<int>(std::lround(note.midiNote)),
                                         start, start + note.durationSeconds * sourceScale });
            }
        }
    }
    std::stable_sort(visibleNotes.begin(), visibleNotes.end(), [](const auto& left, const auto& right)
    {
        if (left.midiRow != right.midiRow) return left.midiRow < right.midiRow;
        if (std::abs(left.start - right.start) > 1.0e-9) return left.start < right.start;
        return left.end < right.end;
    });
    // Note blocks are drawn at a uniform full-row height (z-order handles
    // overlap), so the previous compact-sub-channel lane bookkeeping is no
    // longer needed.
    std::size_t trackIndex = 0;
    for (const auto& track : snapshot.tracks)
    {
        if (!track.compose && !sourceEditMode) { ++trackIndex; continue; }
        if (!sourceEditMode && focusedTrack.isNotEmpty() && track.id != focusedTrack)
        {
            ++trackIndex;
            continue;
        }
        const auto originalColour = contourColours[trackIndex % contourColours.size()];
        std::map<juce::String, NativeNoiseRanges> noiseByNote;
        std::map<const SharedPitchLine*, NativeNoiseRanges> noiseByLine;
        std::map<const SharedPitchLine*, NativeNoiseRanges> voicedByLine;
        std::map<const SharedPitchLine*, NativeNoiseRanges> authoredByLine;
        std::map<const SharedPitchLine*, bool> spanningAuthoredLine;
        const auto authoredFor = [&](const ClipData& sourceClip, const NoteData& sourceNote)
        {
            if (!sourceNote.nativeUnpitched && sourceNote.id == draggedNote && pitchStroke.size() >= 2
                && dragMode == DragMode::pointPitch)
            {
                const auto start = sourceClip.startSeconds + sourceNote.startSeconds;
                return NativeNoiseRanges{{start + pitchStroke.front().timeSeconds,
                                          start + pitchStroke.back().timeSeconds}};
            }
            return nativePitchAuthoredFor(sourceClip, sourceNote);
        };
        const auto maskNativePitch = trackShowsAllNativeRegions(track);
        if (maskNativePitch && !sourceEditMode && showPitchLine && tool != Tool::amplitude)
        {
            const auto& shared = sharedLinesFor(track);
            for (const auto& sourceClip : track.clips)
                for (const auto& sourceNote : sourceClip.notes)
                    if (const auto* member = shared.memberFor(sourceNote.id))
                    {
                        const auto& ranges = nativePitchVisibilityFor(sourceClip, sourceNote);
                        auto& grouped = voicedByLine[member->line.get()];
                        grouped.insert(grouped.end(), ranges.begin(), ranges.end());
                        const auto authored = authoredFor(sourceClip, sourceNote);
                        auto& guides = authoredByLine[member->line.get()];
                        guides.insert(guides.end(), authored.begin(), authored.end());
                        spanningAuthoredLine[member->line.get()] |= sourceNote.pitchControlPoints.size() >= 2
                            || (sourceNote.id == draggedNote && dragMode == DragMode::pointPitch
                                && pitchStroke.size() >= 2);
                    }
            for (auto& [_, ranges] : voicedByLine) mergeNativeNoiseRanges(ranges);
            for (auto& [line, ranges] : authoredByLine)
            {
                mergeNativeNoiseRanges(ranges);
                // A shared authored curve connects its actual control points
                // across cuts and gaps. Automatic source curves have no guide.
                // Local freehand edits retain their exact ranges; two separate
                // strokes must not fill the untouched source gap between them.
                if (!ranges.empty() && spanningAuthoredLine[line])
                    ranges = {{ranges.front().first, ranges.back().second}};
            }
        }
        const auto markConsonants = consonantPitchDashed && !sourceEditMode
            && showPitchLine && tool != Tool::amplitude && trackShowsAllNativeRegions(track);
        if (markConsonants)
        {
            const auto& shared = sharedLinesFor(track);
            for (const auto& sourceClip : track.clips)
            {
                const auto sourceMask = consonantVoicing->rangesFor(sourceClip.sourceFile);
                NativeNoiseRanges mapped;
                if (sourceMask)
                {
                    NoteData span; span.durationSeconds = sourceClip.durationSeconds;
                    mapped = nativeNoiseDisplayRanges(sourceClip, span, *sourceMask);
                }
                for (const auto& sourceNote : sourceClip.notes)
                {
                    auto& ranges = noiseByNote[sourceNote.id];
                    if (sourceMask)
                    {
                        const auto from = sourceClip.startSeconds + sourceNote.startSeconds;
                        const auto to = from + sourceNote.durationSeconds;
                        for (const auto& span : mapped)
                            if (std::min(to, span.second) > std::max(from, span.first))
                                ranges.emplace_back(std::max(from, span.first), std::min(to, span.second));
                    }
                    else ranges = measuredUnvoicedDisplayRanges(sourceClip, sourceNote);
                    if (const auto* member = shared.memberFor(sourceNote.id))
                    {
                        auto& grouped = noiseByLine[member->line.get()];
                        grouped.insert(grouped.end(), ranges.begin(), ranges.end());
                    }
                }
            }
            for (auto& [_, ranges] : noiseByLine) mergeNativeNoiseRanges(ranges);
        }
        for (const auto* source : nativeAudioFocusOrder(track,sourceEditMode?juce::String{}:selectedNote,sourceEditMode?juce::String{}:focusedClip))
        {
            const auto& clip=*source;
            if (!clipIsVisible(track, clip))
                continue;
            const NativeAudioFocusLayer focus(g,sourceEditMode?1.f:nativeOverlapFocusOpacity(track,clip,selectedNote,focusedClip));
            const auto sourceScale = sourceEditMode && clip.audioLength() > 1.0e-9
                ? (clip.sourceDurationSeconds > 1.0e-9 ? clip.sourceDurationSeconds : clip.audioLength())
                    / clip.audioLength()
                : 1.0;
            struct WaveformLabel { juce::String text; juce::Rectangle<float> bounds; float alpha; };
            std::vector<WaveformLabel> waveformLabels;
            for (const auto& note : clip.notes)
            {
                const auto movingSelected = dragMode == DragMode::moveUtauNote
                    && selectedNotes.contains(note.id.toStdString());
                const auto previewingResize = note.id == draggedNote
                    && (dragMode == DragMode::resizeLeft
                        || dragMode == DragMode::resizeRight);
                const auto effectiveStart = previewingResize
                    ? previewStartSeconds
                    : note.startSeconds + (movingSelected
                        ? previewMoveDeltaSeconds : 0.0);
                const auto effectiveDuration = previewingResize
                    ? previewDurationSeconds : note.durationSeconds;
                const auto effectiveMidi = juce::jlimit(0.0f, 127.0f,
                    note.midiNote + (movingSelected
                        ? previewMidi - dragStartMidi : 0.0f));
                const auto absoluteStart = sourceEditMode
                    ? clip.sourceOffsetSeconds + (effectiveStart - clip.audioStartSeconds) * sourceScale
                    : clip.startSeconds + effectiveStart;
                const auto x = timeToX(absoluteStart);
                const auto y = midiToY(effectiveMidi);
                const auto width = std::max(5.0f, static_cast<float>(effectiveDuration * sourceScale)
                                                     * pixelsPerSecond);
                const auto midiRow = static_cast<int>(std::lround(note.midiNote));
                // Note blocks keep a uniform full-row height regardless of how
                // many notes overlap on the same MIDI row.  Melodyne's editor
                // draws overlapping notes at the same row height with z-order
                // rather than shrinking each into a compact sub-channel, so
                // the previous `laneHeight = (rowHeight - 4) / laneCount`
                // behaviour is removed: the "amount" of overlap must not
                // change the visible size of a note.
                (void) midiRow;
                const auto blockHeight = std::max(6.0f, rowHeight - 4.0f);
                const auto bounds = juce::Rectangle<float>(x, y + 2.0f, width, blockHeight);
                auto displayBounds = bounds;
                // The visual editor is shared by both workflows.  Melodyne
                // compatible notes use the same UTAU-style block, contour,
                // boundary and baseline drawing; only the data operations
                // below remain gated by the real track algorithm.
                const auto utauMode = true;
                const auto dataIsUtau = trackUsesVoicebankSynthesis(track);
                auto notePitchVisibility = maskNativePitch
                    ? nativePitchVisibilityFor(clip, note) : NativeNoiseRanges{};
                if (maskNativePitch && sourceEditMode)
                    for (auto& span : notePitchVisibility)
                    {
                        span.first = clip.sourceOffsetSeconds
                            + (span.first - clip.startSeconds - clip.audioStartSeconds) * sourceScale;
                        span.second = clip.sourceOffsetSeconds
                            + (span.second - clip.startSeconds - clip.audioStartSeconds) * sourceScale;
                    }
                const auto nativeRangeColour = note.nativeUnpitched
                    ? rollRole(juce::Colour(0xffb0b7c3), Palette::textMuted) : rollRole(juce::Colour(0xff55caff), Palette::heading);
                const auto nativeEnvelopeColour = rollRole(juce::Colour(0xff64ffa3), Palette::noteEdge);
                const auto noteHasSelection = selectedNotes.contains(note.id.toStdString()) || note.id == selectedNote;
                const auto previewingConsonant = note.id == draggedNote
                    && dragMode == DragMode::consonantLeadIn;
                auto displaySpanSeconds = 0.0;
                std::optional<double> displayLeadIn;
                if (dataIsUtau)
                {
                    if (const auto span = utauSoundSpans.find(note.id.toStdString());
                        span != utauSoundSpans.end())
                    {
                        // A resize in progress moves the note under the
                        // block, so the block has to move with it: left where
                        // the settled span put it, it stood still while the
                        // note grew and then jumped on release, which reads as
                        // the consonant stretching and shifting.  The lead-in
                        // is carried across unchanged -- resizing a note is
                        // not a statement about its consonant.
                        const auto settledStart = clip.startSeconds + note.startSeconds;
                        const auto leadIn = settledStart - span->second.first;
                        const auto settledEnd = settledStart + note.durationSeconds;
                        // Where the next note cuts this one short, if it does.
                        const auto handover = span->second.second
                            < settledEnd - 1.0e-9
                                ? span->second.second
                                : std::numeric_limits<double>::max();
                        const auto soundingStart = previewingConsonant
                            ? absoluteStart - previewConsonantPreutterance
                            : (previewingResize
                                ? absoluteStart - leadIn
                                : span->second.first + (movingSelected
                                    ? previewMoveDeltaSeconds : 0.0));
                        const auto soundingEnd = previewingResize
                            ? std::min(absoluteStart + effectiveDuration, handover)
                            : span->second.second
                                + (movingSelected ? previewMoveDeltaSeconds : 0.0);
                        displaySpanSeconds = soundingEnd - soundingStart;
                        if (previewingConsonant)
                            displayLeadIn = previewConsonantPreutterance;
                        const auto soundingX = timeToX(soundingStart);
                        const auto soundingWidth = std::max(2.0f,
                            timeToX(soundingEnd) - soundingX);
                        displayBounds = juce::Rectangle<float>(
                            soundingX, bounds.getY() + 1.0f,
                            soundingWidth, bounds.getHeight() - 2.0f);
                    }
                }
                else
                {
                    // Native HJM notes already carry their source-region
                    // timing.  Use the same sounding rectangle for imported
                    // Melodyne/MIDI material instead of waiting for an OTO
                    // lookup, so the native boundary is visible immediately.
                    displaySpanSeconds = effectiveDuration * sourceScale;
                    // The imported onset is already inside the recording.
                    // Alignment is its internal line, not additional lead-in
                    // audio outside the clip's selected source range.
                    displayLeadIn = note.consonantSeconds;
                }
                // The sounding block reaches furthest left, the nominal block
                // furthest right, and together they bound everything drawn
                // below.  Vertical extent is deliberately not tested: a pitch
                // line may leave the note's own row entirely.
                // Collect before horizontal culling so scrolling never changes F-8-F-Q order.
                if (noteDance.isEnabled() && !(dataIsUtau && backend::isRestLyric(note.label)))
                    dancers.push_back({ displayBounds, absoluteStart, note.id });
                const auto drawnExtent = displayBounds.getUnion(bounds);
                if (cullOffscreenNotes && !(note.vibratoEnabled && ownsVibratoControls(note))
                    && (drawnExtent.getRight() < visibleArea.getX() - 2.0f
                        || drawnExtent.getX() > visibleArea.getRight() + 2.0f))
                    continue;
                bool nativeSegmentLabelsDrawn = false;
                if (utauMode)
                {
                    // A rest holds its stretch of the phrase open and sounds
                    // nothing, so it has no regions to be shaded with and no
                    // sounding block to be drawn as.  It gets an empty one of
                    // its own, which is also what keeps it visible with the
                    // note-range outline turned off.
                    if (dataIsUtau && backend::isRestLyric(note.label))
                    {
                        g.setColour(Palette::panelRaised.withAlpha(0.55f));
                        g.fillRoundedRectangle(bounds, 4.0f);
                        g.setColour(Palette::textMuted.withAlpha(0.75f));
                        g.drawRoundedRectangle(bounds.reduced(0.5f), 4.0f, 1.0f);
                    }
                    if (showNoteRange)
                    {
                        if (!dataIsUtau)
                        {
                            g.setColour(rollRole(juce::Colours::black, Palette::panelRaised).withAlpha(.72f));
                            g.drawRoundedRectangle(displayBounds, 3.0f, noteHasSelection ? 4.2f : 3.6f);
                            g.setColour(nativeRangeColour);
                            g.drawRoundedRectangle(displayBounds, 3.0f, noteHasSelection ? 2.8f : 2.1f);
                        }
                        else
                        {
                            g.setColour(rollRole(juce::Colour(0xffffbd78), Palette::heading).withAlpha(0.82f));
                            g.drawRoundedRectangle(displayBounds, 3.0f, 1.5f);
                        }
                    }

                    if (const auto found=diffSingerPhonemes.find(note.id.toStdString());dataIsUtau && found!=diffSingerPhonemes.end()
                        && phonemeInformationVisible(note.id))
                        for (const auto& phone:found->second)
                        {
                            const auto left=timeToX(absoluteStart+phone.startSeconds);
                            const auto right=timeToX(absoluteStart+phone.endSeconds);
                            auto box=juce::Rectangle<float>(left,bounds.getY()+1,std::max(1.0f,right-left),bounds.getHeight()-2);
                            g.setColour((phone.kind=="C" || phone.kind=="VC")?rollRole(juce::Colour(0xff55a6be), Palette::accent).withAlpha(.24f)
                                :phone.kind=="V"?rollRole(juce::Colour(0xff7fc995), Palette::noteLight).withAlpha(.13f):Palette::textMuted.withAlpha(.10f));
                            g.fillRect(box);
                            g.setColour(rollRole(juce::Colour(0xff8ac5d6), Palette::heading).withAlpha(.65f));
                            g.drawVerticalLine((int)left,box.getY(),box.getBottom());
                            if ((showPhonemeInfo || (showLyrics && (phone.kind=="C" || phone.kind=="VC" || phone.kind=="release"))) && box.getWidth()>10)
                            {
                                g.setFont(11);g.setColour(Palette::text);
                                g.drawFittedText(diffSingerPhonemeLabel(phone.token),box.reduced(2).toNearestInt(),juce::Justification::centred,1,.65f);
                            }
                        }
                    // Four-region split: three boundaries inside the
                    // sounding block.  Draggable in note-edit mode; in point
                    // mode they are shown faintly and cannot be touched, as
                    // marks to place a pitch point against -- where the
                    // consonant gives way to the vowel is exactly where a
                    // pitch move usually wants to be, and having to leave the
                    // tool to see it was a poor way to find out.
                    const auto guideOnly = tool != Tool::note;
                    if (utauModeUsesRegions(track.utauMode)
                        && dataIsUtau
                        && (tool == Tool::note || tool == Tool::points || tool == Tool::amplitude))
                    {
                        const auto dragging = note.id == draggedNote
                            && dragMode == DragMode::jieSplit;
                        const auto spanSeconds = std::max(1.0e-9,
                            jieSpanEndSeconds - jieSpanStartSeconds);
                        (void) spanSeconds;
                        std::optional<std::array<double, 3>> fractions;
                        auto regions = 4;
                        if (dragging)
                        {
                            fractions = previewJieFractions;
                            regions = previewJieRegions;
                        }
                        else if (displaySpanSeconds > 1.0e-9)
                            fractions = jieFractionsFor(track, note,
                                displaySpanSeconds, displayLeadIn, &regions);
                        if (fractions)
                        {
                            const std::array<juce::Colour, 4> regionColours {
                                rollRole(juce::Colour(0xffff7043), Palette::heading), rollRole(juce::Colour(0xffffca28), Palette::accent),
                                rollRole(juce::Colour(0xff4fc3f7), Palette::noteLight), rollRole(juce::Colour(0xff81c784), Palette::noteEdge)
                            };
                            const auto left = displayBounds.getX();
                            const auto width = displayBounds.getWidth();
                            // As many bands as the entry has regions, and one
                            // boundary fewer -- a three-region entry drew a
                            // fourth band of no width and a line on the note's
                            // own end.
                            const auto bands = static_cast<std::size_t>(
                                juce::jlimit(2, 4, regions));
                            std::array<float, 5> edges { left, 0.0f, 0.0f, 0.0f,
                                                         left + width };
                            for (std::size_t index = 0; index + 1 < bands; ++index)
                                edges[index + 1] = left + width
                                    * static_cast<float>((*fractions)[index]);
                            edges[bands] = left + width;
                            for (std::size_t index = 0; index < bands; ++index)
                            {
                                const auto tint = std::max(0.0f,
                                    edges[index + 1] - edges[index]);
                                if (tint <= 0.5f) continue;
                                g.setColour(regionColours[index].withAlpha(
                                    (note.utauJieSplitSet ? 0.30f : 0.17f)
                                        * (guideOnly ? 0.5f : 1.0f)));
                                g.fillRect(juce::Rectangle<float>(edges[index],
                                    displayBounds.getY() + 1.0f, tint,
                                    displayBounds.getHeight() - 2.0f));
                            }
                            for (std::size_t index = 0; index + 1 < bands; ++index)
                            {
                                g.setColour(rollInk(regionColours[index + 1].withAlpha(
                                    guideOnly ? 0.42f : (dragging ? 1.0f : 0.9f)), Palette::graphBackground));
                                g.drawLine(edges[index + 1],
                                    displayBounds.getY() + 1.0f, edges[index + 1],
                                    displayBounds.getBottom() - 1.0f,
                                    guideOnly ? 1.0f : (dragging ? 2.4f : 1.6f));
                            }
                        }
                    }

                    // A single syllable may contain a same-name C/V pair from
                    // OTO. It is still one note: show its label once in the
                    // ordinary view. Source editing and meaningful subdivisions
                    // retain their detailed boundaries; no segment data changes.
                    const auto& segments = note.nativeSegments;
                    const auto sameAlias = [&](const NativeSegment& segment)
                    { return segment.alias.trim() == note.label.trim(); };
                    const auto plainSyllable = segments.size() == 1
                        ? sameAlias(segments.front())
                            && (segments.front().role == NativeSegmentRole::consonant
                                || segments.front().role == NativeSegmentRole::vowel)
                        : segments.size() == 2
                            && sameAlias(segments[0]) && sameAlias(segments[1])
                            && segments[0].role == NativeSegmentRole::consonant
                            && segments[1].role == NativeSegmentRole::vowel
                            && std::abs(segments[0].sourceEndSeconds
                                        - segments[1].sourceStartSeconds) < 1.0e-6;
                    if (!segments.empty() && (dataIsUtau || sourceEditMode || !plainSyllable))
                    {
                        const auto segmentXAt = [&](double sourceLocal)
                        {
                            const auto sourceAbsolute = note.nativeSourceStartSeconds >= 0.0
                                ? note.nativeSourceStartSeconds + sourceLocal
                                : clip.sourceOffsetSeconds + note.startSeconds - clip.audioStartSeconds + sourceLocal;
                            if (sourceEditMode) return timeToX(sourceAbsolute);
                            const auto source = sourceAbsolute - clip.sourceOffsetSeconds;
                            auto target = clip.sourceDurationSeconds > 1.0e-9
                                ? clip.audioStartSeconds + source * clip.audioLength() / clip.sourceDurationSeconds
                                : clip.audioStartSeconds + source;
                            const auto& map = clip.sourceTimeMap;
                            if (!map.empty())
                            {
                                target = map.back().targetSeconds;
                                if (source <= map.front().sourceSeconds) target = map.front().targetSeconds;
                                else for (std::size_t i = 1; i < map.size(); ++i)
                                    if (source <= map[i].sourceSeconds)
                                    {
                                        const auto span = map[i].sourceSeconds - map[i-1].sourceSeconds;
                                        const auto u = span > 1.0e-9 ? (source-map[i-1].sourceSeconds)/span : 0.0;
                                        target = map[i-1].targetSeconds + u*(map[i].targetSeconds-map[i-1].targetSeconds);
                                        break;
                                    }
                            }
                            return timeToX(clip.startSeconds + target);
                        };
                        const auto segmentColour = [](const NativeSegment& segment)
                        {
                            switch (segment.role)
                            {
                                case NativeSegmentRole::consonant:
                                    return Palette::noteLight;
                                case NativeSegmentRole::transition:
                                    return rollRole(juce::Colour(0xff79c9c3), Palette::accent);
                                case NativeSegmentRole::unknown:
                                    return Palette::textMuted;
                                case NativeSegmentRole::silence:
                                    return Palette::panelRaised;
                                case NativeSegmentRole::breath:
                                    return rollRole(juce::Colour(0xff9fc6e8), Palette::textMuted);
                                case NativeSegmentRole::noise:
                                    return rollRole(juce::Colour(0xffc2a7de), Palette::accent);
                                case NativeSegmentRole::ending:
                                    return rollRole(juce::Colour(0xff6ea6a2), Palette::heading);
                                case NativeSegmentRole::vowel:
                                    return segment.stretchable ? rollRole(juce::Colour(0xff81c784), Palette::noteEdge)
                                                               : rollRole(juce::Colour(0xff4fc3f7), Palette::noteLight);
                            }
                            return Palette::noteFill;
                        };
                        for (std::size_t index = 0; index < note.nativeSegments.size(); ++index)
                        {
                            const auto& segment = note.nativeSegments[index];
                            const auto segmentX = segmentXAt(segment.sourceStartSeconds);
                            const auto segmentRight = segmentXAt(segment.sourceEndSeconds);
                            const auto segmentWidth = std::max(0.0f,
                                segmentRight - segmentX);
                            if (segmentWidth <= 0.5f) continue;
                            const auto alpha = segment.role == NativeSegmentRole::unknown
                                ? 0.12f : segment.role == NativeSegmentRole::transition
                                    ? 0.25f : 0.20f;
                            if (dataIsUtau)
                            {
                                g.setColour(segmentColour(segment).withAlpha(alpha));
                                g.fillRect(juce::Rectangle<float>(segmentX,
                                    displayBounds.getY() + 1.0f, segmentWidth,
                                    displayBounds.getHeight() - 2.0f));
                            }
                            if (index > 0)
                            {
                                g.setColour(rollInk(segmentColour(segment).withAlpha(
                                    segment.role == NativeSegmentRole::unknown ? 0.42f : 0.86f), Palette::graphBackground));
                                g.drawVerticalLine(static_cast<int>(std::lround(segmentX)),
                                    displayBounds.getY() + 1.0f,
                                    displayBounds.getBottom() - 1.0f);
                            }
                            const auto alias = segment.alias.trim().isEmpty()
                                ? juce::String("-") : segment.alias.trim();
                            // Carried C/V snapshots keep their original aliases for
                            // sound fidelity. Do not let inherited '-' segment labels
                            // hide the newly named pronunciation on the piano roll.
                            const bool inheritedPlaceholder=!dataIsUtau&&(alias=="-"||alias=="_")
                                &&note.materialAudioId.isNotEmpty()&&note.materialRegionId.isNotEmpty()
                                &&note.label.isNotEmpty()&&note.label!="-"&&note.label!="_";
                            if (!inheritedPlaceholder && !note.nativeUnpitched && showLyrics && segmentWidth >= 22.0f && (note.nativeSegments.size() > 1
                                || alias == "-" || alias == "_"))
                            {
                                const auto alpha = segment.role == NativeSegmentRole::unknown ? 0.60f : 0.86f;
                                const auto originalBounds = juce::Rectangle<float>(segmentX,
                                    displayBounds.getY(), segmentWidth, displayBounds.getHeight());
                                const auto aliasBounds = nativeWaveformLabelBounds(track, clip, note, originalBounds);
                                nativeSegmentLabelsDrawn |= !dataIsUtau;
                                if (aliasBounds.getY() != originalBounds.getY())
                                    waveformLabels.push_back({alias, aliasBounds, alpha});
                                else
                                {
                                    g.setColour(rollText(Palette::text.withAlpha(alpha),
                                        Palette::graphBackground.overlaidWith(segmentColour(segment).withAlpha(dataIsUtau
                                            ? segment.role == NativeSegmentRole::unknown ? .12f
                                                : segment.role == NativeSegmentRole::transition ? .25f : .20f
                                            : 0.0f))));
                                    g.setFont(std::min(10.0f, std::max(7.0f, displayBounds.getHeight() - 3.0f)));
                                    g.drawFittedText(alias, originalBounds.toNearestInt().reduced(3, 0),
                                        juce::Justification::centredLeft, 1);
                                }
                            }
                        }
                        const auto alignment = note.nativeSegments.front().alignmentSeconds;
                        if (alignment > 1.0e-9)
                        {
                            const auto alignmentX = segmentXAt(alignment);
                            const float dash[] { 3.0f, 2.0f };
                            juce::Path line;
                            line.startNewSubPath(alignmentX, displayBounds.getY());
                            line.lineTo(alignmentX, displayBounds.getBottom());
                            juce::Path dashed;
                            juce::PathStrokeType(1.0f).createDashedStroke(dashed, line, dash, 2);
                            g.setColour(Palette::accentLight.withAlpha(0.92f));
                            g.fillPath(dashed);
                        }
                    }

                    // Draw the range baseline first so it cannot cover the
                    // closed envelope's bottom edge in native mode.
                    if (!dataIsUtau)
                    {
                        g.setColour(nativeRangeColour);
                        g.fillRoundedRectangle(juce::Rectangle<float>(bounds.getX(),
                            bounds.getBottom() - 2.0f, bounds.getWidth(), 3.5f), 1.75f);
                    }

                    // The body carries its own amplitude envelope: the top
                    // edge follows the gain, so a glance says where the note
                    // fades in, holds and falls away.  Drawn with exactly the
                    // geometry the amplitude tool uses for its curve, so the
                    // two are the same line seen through different tools
                    // rather than two pictures of one envelope; the tool draws
                    // its own, so the shape steps aside there.
                    // Read once: the shape is drawn from it, and the number
                    // below is placed clear of the shape, which rises above the
                    // note whenever the gain goes over unity.
                    std::vector<AmplitudeEnvelopePoint> shownEnvelope;
                    if (showEnvelope && tool != Tool::amplitude && !usesCvvcEnvelope(note.id))
                        shownEnvelope = draggedNote == note.id
                                && dragMode == DragMode::amplitudePoint
                                && !amplitudeStroke.empty()
                            ? amplitudeStroke : amplitudeEnvelopeFor(note, absoluteStart);
                    if (showEnvelope && tool != Tool::amplitude && !usesCvvcEnvelope(note.id))
                    {
                        shownEnvelope=effectiveAmplitudeEnvelope(note,absoluteStart,shownEnvelope);
                        const auto& envelope = shownEnvelope;
                        if (envelope.size() >= 2)
                        {
                            const auto floorY = amplitudeY(effectiveMidi, -200.0f);
                            juce::Path shape;
                            juce::Path ridge;
                            for (std::size_t index = 0; index < envelope.size(); ++index)
                            {
                                const auto point = juce::Point<float>(
                                    timeToX(absoluteStart + envelope[index].timeSeconds),
                                    amplitudeY(effectiveMidi, envelope[index].gainDb));
                                if (index == 0)
                                {
                                    shape.startNewSubPath(point.x, floorY);
                                    ridge.startNewSubPath(point);
                                }
                                else ridge.lineTo(point);
                                shape.lineTo(point);
                            }
                            shape.lineTo(timeToX(absoluteStart
                                + envelope.back().timeSeconds), floorY);
                            shape.closeSubPath();
                            const auto lit = selectedNotes.contains(note.id.toStdString())
                                || note.id == selectedNote;
                            // An envelope runs to the note's nominal end while
                            // the note may stop sounding earlier, where the one
                            // after it takes over.  Clipping to the sounding
                            // block keeps the shape off the silence past it.
                            juce::Graphics::ScopedSaveState clipped(g);
                            // Horizontally the sounding block, but open above
                            // it: a gain over unity belongs outside the note,
                            // and cutting it off would read as a flat ceiling.
                            auto envelopeClip = displayBounds.withTop(
                                displayBounds.getY() - displayBounds.getHeight());
                            if (!dataIsUtau) envelopeClip = envelopeClip.expanded(3.0f);
                            g.reduceClipRegion(envelopeClip.getSmallestIntegerContainer());
                            if (dataIsUtau)
                            {
                                g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(lit ? 0.30f : 0.16f));
                                g.fillPath(shape);
                                g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(lit ? 0.95f : 0.60f));
                                g.strokePath(ridge, juce::PathStrokeType(lit ? 1.8f : 1.2f));
                            }
                            else
                            {
                                g.setColour(rollRole(juce::Colours::black, Palette::panelRaised).withAlpha(.72f));
                                g.strokePath(shape, juce::PathStrokeType(lit ? 4.2f : 3.6f));
                                g.setColour(nativeEnvelopeColour);
                                g.strokePath(shape, juce::PathStrokeType(lit ? 2.6f : 2.0f));
                            }
                        }
                    }

                    // The envelope's base value, small, at the note's top left
                    // corner: part of the envelope display, so it comes and
                    // goes with it.  Only when it is not 100 -- a number on
                    // every note would say nothing and cover the roll.
                    if (showEnvelope
                        && std::abs(note.amplitudeEnvelopeBasePercent - 100.0f) > 0.05f)
                    {
                        const auto text = juce::String(
                            note.amplitudeEnvelopeBasePercent, 0) + "%";
                        const auto left = displayBounds.getX();
                        const auto right = left + 44.0f;
                        // A base value over 100 is exactly what lifts the
                        // shape above the note, so the number rides up with it
                        // instead of being written through the line.  Only the
                        // stretch the number covers counts: a peak further
                        // along the note would leave it floating over nothing.
                        auto ceiling = displayBounds.getY();
                        for (std::size_t index = 0; index + 1 < shownEnvelope.size(); ++index)
                        {
                            const auto x0 = timeToX(absoluteStart
                                + shownEnvelope[index].timeSeconds);
                            const auto x1 = timeToX(absoluteStart
                                + shownEnvelope[index + 1].timeSeconds);
                            if (std::max(x0, x1) < left || std::min(x0, x1) > right) continue;
                            const auto y0 = amplitudeY(effectiveMidi, shownEnvelope[index].gainDb);
                            const auto y1 = amplitudeY(effectiveMidi,
                                                       shownEnvelope[index + 1].gainDb);
                            // The shape is drawn as straight lines between
                            // these points, so the highest pixel over the
                            // number's own span is at one of its two ends.
                            const auto span = x1 - x0;
                            auto from = 0.0f, to = 1.0f;
                            if (std::abs(span) > 1.0e-3f)
                            {
                                from = juce::jlimit(0.0f, 1.0f, (left - x0) / span);
                                to = juce::jlimit(0.0f, 1.0f, (right - x0) / span);
                            }
                            ceiling = std::min({ ceiling, y0 + (y1 - y0) * from,
                                                 y0 + (y1 - y0) * to });
                        }
                        g.setFont(10.0f);
                        // The lane's own orange, so the number reads as
                        // belonging to the envelope rather than to the note.
                        g.setColour(rollText(rollRole(juce::Colour(0xffffa94d), Palette::heading).withAlpha(0.92f), Palette::graphBackground));
                        g.drawText(text, static_cast<int>(left),
                                   std::max(0, static_cast<int>(ceiling) - 11),
                                   44, 10, juce::Justification::centredLeft, false);
                    }

                    // Keep the MIDI note's nominal range visible independently
                    // from oto.ini preutterance, overlap and consonant timing.
                    // The hollow outline above is the actual sounding range;
                    // this heavier baseline is always exactly [note start, end].
                    const auto nominalBar = juce::Rectangle<float>(
                        bounds.getX(), bounds.getBottom() - 2.0f,
                        bounds.getWidth(), 3.5f);
                    if (dataIsUtau)
                    {
                        g.setColour(rollRole(juce::Colour(0xffe7a34b), Palette::heading));
                        g.fillRoundedRectangle(nominalBar, 1.75f);
                    }
                    if (selectedNotes.contains(note.id.toStdString())
                        && displayBounds.getX() < bounds.getX() - 0.5f)
                    {
                        g.setColour(rollRole(juce::Colour(0xffffd19a), Palette::heading));
                        g.fillRoundedRectangle(juce::Rectangle<float>(
                            displayBounds.getX() - 1.5f, displayBounds.getY(),
                            3.0f, displayBounds.getHeight()), 1.5f);
                    }
                    if (note.id == draggedNote
                        && dragMode == DragMode::consonantLeadIn)
                    {
                        g.setColour(Palette::text);
                        g.setFont(10.5f);
                        g.drawText(juce::String::fromUTF8(dataIsUtau ? "辅音速度 " : "先行 ")
                                + juce::String(previewConsonantVelocity),
                            static_cast<int>(displayBounds.getX() + 6.0f),
                            static_cast<int>(displayBounds.getY() - 19.0f),
                            104, 18, juce::Justification::centredLeft, false);
                    }
                }
                else
                {
                    g.setColour(Palette::noteFill.withAlpha(0.90f));
                    g.fillRoundedRectangle(bounds, 4.0f);
                    const auto consonantWidth = juce::jlimit(0.0f, width,
                        static_cast<float>(note.consonantSeconds * sourceScale) * pixelsPerSecond);
                    g.setColour(Palette::noteLight.withAlpha(0.38f));
                    g.fillRoundedRectangle(bounds.withWidth(consonantWidth), 4.0f);
                    g.setColour(Palette::noteLight.withAlpha(0.82f));
                    g.drawRoundedRectangle(bounds, 3.0f, 1.2f);
                }
                if (selectedNotes.contains(note.id.toStdString()))
                {
                    g.setColour(Palette::text.withAlpha(0.95f));
                    g.drawRoundedRectangle(displayBounds.reduced(1.0f), 3.0f, 1.8f);
                }

                auto displayLabel = note.id == inlineAliasNoteId
                    ? inlineAliasEditor.getText() : note.label;
                if (note.nativeUnpitched)
                {
                    // Keep the waveform readable, while the grey striped body
                    // clearly differs from an ordinary pitched note.
                    juce::Graphics::ScopedSaveState state(g);
                    g.reduceClipRegion(displayBounds.toNearestInt());
                    g.setColour(rollRole(juce::Colour(0xff525965), Palette::selection).withAlpha(.60f));
                    g.fillRect(displayBounds);
                    g.setColour(nativeRangeColour.withAlpha(.38f));
                    for (auto stripe = displayBounds.getX() - displayBounds.getHeight(); stripe < displayBounds.getRight(); stripe += 10.0f)
                        g.drawLine(stripe, displayBounds.getBottom(), stripe + displayBounds.getHeight(), displayBounds.getY(), 1.0f);
                    g.setColour(noteHasSelection ? rollRole(juce::Colour(0xffe7edf7), Palette::heading) : nativeRangeColour);
                    g.drawRoundedRectangle(displayBounds.reduced(.7f), 3.0f, noteHasSelection ? 2.8f : 1.8f);
                }
                auto labelBounds = displayBounds;
                if (const auto found=diffSingerPhonemes.find(note.id.toStdString());found!=diffSingerPhonemes.end())
                    for (const auto& phone:found->second)
                        if (phone.kind=="V")
                        {
                            labelBounds.setLeft(timeToX(absoluteStart+phone.startSeconds));
                            labelBounds.setRight(timeToX(absoluteStart+phone.endSeconds));
                            break;
                        }
                if (((showLyrics && !nativeSegmentLabelsDrawn && displayLabel.isNotEmpty()
                    && !(showPhonemeInfo&&dataIsUtau&&diffSingerPhonemes.contains(note.id.toStdString()))) || note.nativeUnpitched)
                    && labelBounds.getWidth() >= 18.0f)
                {
                    if (note.nativeUnpitched) displayLabel = strings.text("note.unpitched");
                    const auto aboveBounds = nativeWaveformLabelBounds(track, clip, note, labelBounds);
                    if (aboveBounds.getY() != labelBounds.getY())
                        waveformLabels.push_back({displayLabel, aboveBounds, 0.96f});
                    else
                    {
                        g.setColour(rollText(Palette::text.withAlpha(0.86f),
                            Palette::graphBackground.overlaidWith(note.nativeUnpitched ? Palette::selection.withAlpha(.60f) : Palette::noteFill.withAlpha(.30f))));
                        g.setFont(std::min(11.0f, std::max(8.0f, displayBounds.getHeight() - 2.0f)));
                        g.drawFittedText(displayLabel, labelBounds.toNearestInt().reduced(4, 0),
                                         juce::Justification::centredLeft, 1);
                    }
                }

                if (utauMode && tool == Tool::amplitude && !usesCvvcEnvelope(note.id))
                {
                    const auto effective = displayAmplitudeEnvelope(note, absoluteStart);
                    const auto noteSelected = selectedNotes.contains(note.id.toStdString())
                        || note.id == selectedNote;

                    // A faint 0 dB reference makes the vertical scale readable
                    // without adding a separate automation lane.
                    g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(0.18f));
                    g.drawHorizontalLine(static_cast<int>(amplitudeY(note.midiNote, 0.0f)),
                        displayBounds.getX(), displayBounds.getRight());

                    juce::Path envelopePath;
                    for (std::size_t index = 0; index < effective.size(); ++index)
                    {
                        const auto point = juce::Point<float>(
                            timeToX(absoluteStart + effective[index].timeSeconds),
                            amplitudeY(note.midiNote, effective[index].gainDb));
                        if (index == 0) envelopePath.startNewSubPath(point);
                        else envelopePath.lineTo(point);
                    }
                    if (!dataIsUtau && !effective.empty())
                    {
                        // Closure is display geometry only, not extra silent
                        // envelope points or an implicit fade on the recording.
                        const auto floorY = amplitudeY(note.midiNote, -200.0f);
                        envelopePath.lineTo(timeToX(absoluteStart + effective.back().timeSeconds), floorY);
                        envelopePath.lineTo(timeToX(absoluteStart + effective.front().timeSeconds), floorY);
                        envelopePath.closeSubPath();
                        g.setColour(rollRole(juce::Colours::black, Palette::panelRaised).withAlpha(.72f));
                        g.strokePath(envelopePath, juce::PathStrokeType(noteSelected ? 4.4f : 3.6f));
                    }
                    const auto envelopeColour = dataIsUtau ? rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge) : nativeEnvelopeColour;
                    g.setColour(dataIsUtau ? envelopeColour.withAlpha(noteSelected ? 0.98f : 0.62f) : envelopeColour);
                    g.strokePath(envelopePath, juce::PathStrokeType(dataIsUtau
                        ? (noteSelected ? 2.2f : 1.5f) : (noteSelected ? 2.6f : 2.0f)));
                }

                g.setColour(Palette::textMuted.withAlpha(0.55f));
                const float dash[] { 4.0f, 3.0f };
                juce::Path boundary;
                boundary.startNewSubPath(bounds.getX(), bounds.getY() - rowHeight * 2.0f);
                boundary.lineTo(bounds.getX(), bounds.getBottom() + rowHeight * 2.0f);
                juce::Path dashedBoundary;
                juce::PathStrokeType(1.0f).createDashedStroke(dashedBoundary, boundary, dash, 2);
                g.fillPath(dashedBoundary);

                // Sibilant ticks are temporarily hidden. Keep the underlying
                // analysis/import data intact for project compatibility.

                if (showOriginalPitchLine && !sourceEditMode && tool != Tool::amplitude
                    && diffSingerNoteIds.contains(note.id.toStdString())
                    && !note.diffSingerPitchReference.empty())
                {
                    const auto reference = diffSingerPitchReferencePath(note, absoluteStart);
                    juce::Path dashedReference;
                    const float referenceDash[] {5.0f, 4.0f};
                    juce::PathStrokeType(1.5f, juce::PathStrokeType::curved)
                        .createDashedStroke(dashedReference, reference, referenceDash, 2);
                    g.setColour(rollRole(juce::Colour(0xff83bde8), Palette::textMuted).withAlpha(0.72f));
                    g.fillPath(dashedReference);
                }

                juce::Path originalContour;
                juce::Path contour;
                bool open = false;
                bool originalOpen = false;
                // The consonant sits in the lead-in, before the note start,
                // and it is sung with the head of the pitch curve held: that
                // is what encodePitchbend does when the output time falls
                // ahead of the curve.  Drawing the line only from the note
                // start left that stretch blank and made the pitch look like
                // it jumped at the note boundary.  Start the line where the
                // note actually starts sounding instead.
                auto leadInLocal = 0.0;
                auto tailLocal = note.durationSeconds;
                if (dataIsUtau)
                {
                    if (const auto span = utauSoundSpans.find(note.id.toStdString());
                        span != utauSoundSpans.end())
                    {
                        leadInLocal = std::min(0.0,
                            span->second.first - absoluteStart);
                        // A note stops sounding where the next one starts,
                        // which is earlier than its nominal end whenever the
                        // next note reaches back for its own consonant.
                        tailLocal = juce::jlimit(0.0, note.durationSeconds,
                            span->second.second - absoluteStart);
                    }
                    // Between two adjacent notes the pitch is carried by the S
                    // transition, and the renderer writes that same bridge into
                    // both of them.  Each note therefore hands its line over at
                    // its outermost anchor; drawing on past that point put a
                    // second, differently placed line beside the bridge, which
                    // is what read as a fork after a pitch change.
                    //
                    // Where no neighbour abuts, the line simply starts at the
                    // note.  The consonant reaching back before it is sung at
                    // the held head pitch, but drawing that as a flat run left a
                    // stray horizontal stub hanging off the front of the curve,
                    // so it is not drawn.
                    const PositionedUtauNote here {
                        note.id, absoluteStart, absoluteStart + note.durationSeconds,
                        note.utauAutoPitchTransition
                    };
                    const auto& handOver = pitchAnchorsFor(note);
                    if (!handOver.empty())
                    {
                        if (const auto previous = previousUtauNoteFor(note.id);
                            previous && formsAdjacentPitchBoundary(*previous, here))
                            leadInLocal = handOver.front().timeSeconds;
                        if (const auto next = nextUtauNoteFor(note.id);
                            next && formsAdjacentPitchBoundary(here, *next))
                            tailLocal = std::min(tailLocal, handOver.back().timeSeconds);
                    }
                }
                const auto foldRealVibratoIntoContour = note.vibratoEnabled
                    && note.vibratoRealLine && tool != Tool::points
                    && note.durationSeconds > 1.0e-9;
                const auto foldedVibrato = foldRealVibratoIntoContour
                    ? effectiveVibrato(note) : note;
                const auto nativeJoin = dataIsUtau ? std::nullopt : incomingJoinGlideFor(note);
                for (const auto& point : note.contour)
                {
                    if (!point.voiced)
                    {
                        open = false;
                        originalOpen = false;
                        continue;
                    }
                    const auto px = x + static_cast<float>(point.timeSeconds * sourceScale) * pixelsPerSecond;
                    const auto sourceCenter = note.sourceMidiCenter >= 0.0f
                        ? note.sourceMidiCenter : note.midiNote;
                    const auto originalPitch = sourceCenter + point.relativeCents / 100.0f;
                    const auto vibratoCents = foldRealVibratoIntoContour
                        ? static_cast<float>(vibratoCentsAt(foldedVibrato,
                            point.timeSeconds)) : 0.0f;
                    auto pitch = effectiveMidi
                        + (renderedPitchCents(note, point) + vibratoCents) / 100.0f;
                    if (nativeJoin && nativeJoin->joinSeconds > 1.0e-9
                        && point.timeSeconds < nativeJoin->joinSeconds)
                    {
                        const auto u = static_cast<float>(juce::jlimit(0.0, 1.0,
                            point.timeSeconds / nativeJoin->joinSeconds));
                        const auto shaped = u * u * (3.0f - 2.0f * u);
                        pitch = static_cast<float>(nativeJoin->leadMidi)
                            + (pitch - static_cast<float>(nativeJoin->leadMidi)) * shaped;
                    }
                    const auto py = midiToY(pitch) + rowHeight * 0.5f;
                    const auto originalY = midiToY(originalPitch) + rowHeight * 0.5f;
                    if (!originalOpen) originalContour.startNewSubPath(px, originalY);
                    else originalContour.lineTo(px, originalY);
                    originalOpen = true;
                    if (point.timeSeconds < leadInLocal - 1.0e-9) continue;
                    if (point.timeSeconds > tailLocal + 1.0e-9)
                    {
                        if (open)
                        {
                            const auto tailX = x + static_cast<float>(tailLocal
                                * sourceScale) * pixelsPerSecond;
                            contour.lineTo(tailX, py);
                            open = false;
                        }
                        break;
                    }
                    if (!open) contour.startNewSubPath(px, py);
                    else contour.lineTo(px, py);
                    open = true;
                }
                // A note set to the real vibrato line folds the swing into the
                // anchor line above; the separate trace below is then skipped so
                // the swing is not drawn twice.
                const auto realVibratoLine = foldRealVibratoIntoContour;

                // Only while the point tool is in hand, and only as the
                // scaffold the dots sit on.  Not whenever points exist: what
                // the native renderer follows is the contour, which
                // setNotePitchCurve writes the placed curve into -- so outside
                // the tool the contour is the honest line to show.
                const auto* nativeSharedMember = !dataIsUtau ? sharedLinesFor(track).memberFor(note.id) : nullptr;
                auto authoredPitch = maskNativePitch ? authoredFor(clip, note) : NativeNoiseRanges{};
                if (maskNativePitch && sourceEditMode)
                    for (auto& span : authoredPitch)
                    {
                        span.first = clip.sourceOffsetSeconds
                            + (span.first - clip.startSeconds - clip.audioStartSeconds) * sourceScale;
                        span.second = clip.sourceOffsetSeconds
                            + (span.second - clip.startSeconds - clip.audioStartSeconds) * sourceScale;
                    }
                const auto pointsOwnTheLine = (nativeSharedMember != nullptr && nativeSharedMember->nativeSharedCurve)
                    || (!dataIsUtau && tool == Tool::points
                    && (!note.pitchControlPoints.empty()
                        || (draggedNote == note.id && dragMode == DragMode::pointPitch)));
                if (!dataIsUtau && showOriginalPitchLine && !sourceEditMode
                    && tool != Tool::amplitude)
                {
                    juce::Graphics::ScopedSaveState pitchClip(g);
                    if (maskNativePitch)
                        reduceNativePitchClip(g, notePitchVisibility, pixelsPerSecond, timeToX(0));
                    g.setColour(originalColour.withAlpha(0.86f));
                    juce::Path dashed;
                    juce::PathStrokeType(1.35f, juce::PathStrokeType::curved)
                        .createDashedStroke(dashed, originalContour, dash, 2);
                    g.fillPath(dashed);
                }
                if (!dataIsUtau && !pointsOwnTheLine && showPitchLine && tool != Tool::amplitude)
                {
                    juce::Graphics::ScopedSaveState pitchClip(g);
                    if(!sourceEditMode)clipNotePitchAgainstConnections(g,track);
                    if (maskNativePitch)
                        reduceNativePitchClip(g, notePitchVisibility, pixelsPerSecond, timeToX(0));
                    g.setColour(Palette::pitchLine);
                    if (markConsonants)
                    {
                        const auto& ranges = noiseByNote[note.id];
                        strokeNativePitchWithNoise(g, contour, ranges, pixelsPerSecond, timeToX(0));
                    }
                    else g.strokePath(contour,
                        juce::PathStrokeType(2.0f, juce::PathStrokeType::curved));
                }

                // Vibrato belongs to the note, not to one tool, so it stays
                // visible while notes are being edited rather than only under
                // the point tool.  The amplitude lane is the one exception: it
                // draws its own envelope over the same rows.
                //
                // It is traced against the note's nominal pitch rather than on
                // top of the pitch line it actually swings around.  Riding that
                // line would make the swing unreadable wherever the line moves;
                // against the flat nominal pitch its depth and rate can be read
                // at a glance.  What is rendered adds the two together.
                if (tool != Tool::amplitude && showPitchLine
                    && note.vibratoEnabled && note.durationSeconds > 1.0e-9)
                {
                    const auto noteHasFocus = selectedNotes.contains(note.id.toStdString())
                        || note.id == selectedNote;
                    const auto shown = effectiveVibrato(note);
                    const auto baseY = midiToY(note.midiNote) + rowHeight * 0.5f;
                    // With the swing already in the pitch line, a second copy
                    // of it against the nominal pitch is just clutter.  The
                    // handles below are still drawn, so the shape stays
                    // editable in either display mode.
                    const auto drawSeparateTrace = !realVibratoLine;
                    const auto widthPixels = std::max(2.0f,
                        static_cast<float>(note.durationSeconds) * pixelsPerSecond);
                    const auto steps = std::max(8,
                        static_cast<int>(std::ceil(widthPixels / 2.0f)));
                    juce::Path swing;
                    auto open = false;
                    for (int step = 0; step <= steps; ++step)
                    {
                        const auto time = note.durationSeconds * step / steps;
                        const auto cents = vibratoCentsAt(shown, time);
                        const auto swingX = x + static_cast<float>(time) * pixelsPerSecond;
                        const auto swingY = baseY
                            - static_cast<float>(cents / 100.0) * rowHeight;
                        if (!open) { swing.startNewSubPath(swingX, swingY); open = true; }
                        else swing.lineTo(swingX, swingY);
                    }
                    if (drawSeparateTrace)
                    {
                        juce::Graphics::ScopedSaveState pitchClip(g);
                        if (maskNativePitch)
                            reduceNativePitchClip(g, notePitchVisibility, pixelsPerSecond, timeToX(0));
                        // The line the swing is centred on, so the depth reads
                        // as a distance rather than an absolute position.
                        g.setColour(rollRole(juce::Colour(0xff7fd4ff), Palette::heading).withAlpha(0.28f));
                        g.drawHorizontalLine(static_cast<int>(baseY), x, x + widthPixels);
                        g.setColour(rollRole(juce::Colour(0xff7fd4ff), Palette::heading)
                            .withAlpha(noteHasFocus ? 0.95f : 0.62f));
                        g.strokePath(swing, juce::PathStrokeType(1.2f));
                    }

                    // Handles on the swing itself, only for the note being
                    // worked on.  Each one drags a single parameter, so the
                    // shape can be dialled in without opening the dialog.
                    if (ownsVibratoControls(note))
                    {
                        const auto control=vibratoControlNote(note);
                        const auto* group=joinedVibratoFor(note.id);
                        const auto controlStart=group?group->start:absoluteStart;
                        const std::array<std::pair<VibratoHandle, juce::Colour>, 7> handles {{
                            { VibratoHandle::offset,  rollRole(juce::Colour(0xffc9a0ff), Palette::accent) },
                            { VibratoHandle::length,  rollRole(juce::Colour(0xffffc857), Palette::heading) },
                            { VibratoHandle::end,     rollRole(juce::Colour(0xffffc857), Palette::heading) },
                            { VibratoHandle::fadeIn,  rollRole(juce::Colour(0xff9ae66e), Palette::noteEdge) },
                            { VibratoHandle::fadeOut, rollRole(juce::Colour(0xff9ae66e), Palette::noteEdge) },
                            { VibratoHandle::cycle,   rollRole(juce::Colour(0xffff8fa3), Palette::accent) },
                            { VibratoHandle::depth,   rollRole(juce::Colour(0xff7fd4ff), Palette::heading) }
                        }};
                        for (const auto& handle : handles)
                        {
                            const auto centre = vibratoHandleCentre(control, handle.first,
                                                                    controlStart);
                            const auto hx = centre.x;
                            const auto hy = centre.y;
                            const auto held = dragMode == DragMode::vibrato
                                && note.id == draggedNote
                                && draggedVibratoHandle == handle.first;
                            const auto size = held ? 8.0f : 6.0f;
                            g.setColour(handle.second.withAlpha(held ? 1.0f : 0.85f));
                            g.fillEllipse(hx - size * 0.5f, hy - size * 0.5f, size, size);
                            g.setColour(juce::Colours::black.withAlpha(0.55f));
                            g.drawEllipse(hx - size * 0.5f, hy - size * 0.5f, size, size, 1.0f);
                        }
                    }
                }

                // The S transition between two adjacent notes is part of the
                // pitch line, not a point-tool guide: the renderer writes the
                // same bridge into both notes.  Drawing it only under the point
                // tool left a gap between their lines everywhere else -- and
                // drawing it whatever the switch said left the joins behind
                // when the line was hidden, which is a pitch line arriving at
                // every note from nowhere.
                if (dataIsUtau && tool != Tool::amplitude && showPitchLine)
                {
                // Anchor handles and the shape menu belong to the point
                // tool; the bridge itself is drawn for every tool, just
                // below, because it is part of the pitch line.
                const PositionedUtauNote positionedNote {
                    note.id, absoluteStart, absoluteStart + note.durationSeconds,
                    note.utauAutoPitchTransition
                };
                if (const auto span = transitionBridge(track, note, absoluteStart))
                {
                    const auto steps = std::max(4, static_cast<int>(std::ceil(
                        std::max(0.001, span->endSeconds - span->startSeconds)
                            * pixelsPerSecond / 3.0)));
                    juce::Path bridge;
                    for (int step = 0; step <= steps; ++step)
                    {
                        const auto u = static_cast<float>(step) / steps;
                        const auto shaped = u * u * (3.0f - 2.0f * u);
                        const auto time = span->startSeconds
                            + (span->endSeconds - span->startSeconds) * u;
                        const auto midi = span->startMidi
                            + (span->endMidi - span->startMidi) * shaped;
                        const auto point = juce::Point<float>(timeToX(time),
                            midiToY(midi) + rowHeight * 0.5f);
                        if (step == 0) bridge.startNewSubPath(point);
                        else bridge.lineTo(point);
                    }
                    g.setColour(Palette::noteLight.withAlpha(0.88f));
                    g.strokePath(bridge, juce::PathStrokeType(2.0f,
                        juce::PathStrokeType::curved));
                }
                }

                if ((dataIsUtau || pointsOwnTheLine || !authoredPitch.empty()) && showPitchLine
                    && tool != Tool::amplitude)
                {
                    const juce::Graphics::ScopedSaveState graphOwnershipClip(g);
                    if(!sourceEditMode)clipNotePitchAgainstConnections(g,track);
                    const auto& anchors = draggedNote == note.id
                            && dragMode == DragMode::pointPitch && !pitchStroke.empty()
                        ? pitchStroke : pitchAnchorsFor(note);
                    // Outside the point tool a note set to the real vibrato line
                    // shows the swing folded into this same line, so both views
                    // still come from one place.
                    const auto foldVibrato = note.vibratoEnabled && note.vibratoRealLine
                        && tool != Tool::points && note.durationSeconds > 1.0e-9;
                    const auto shownVibrato = effectiveVibrato(note);
                    // A connected note glides its head out of the previous
                    // note's tail pitch; drawing that here keeps the pitch line
                    // continuous across the seam, matching the render.  UTAU
                    // adjacency draws its own crossfade bridge below instead.
                    const auto joinGlide = dataIsUtau ? std::nullopt
                                                      : incomingJoinGlideFor(note);
                    const auto glideAt = [&](double time, float natural)
                    {
                        if (!joinGlide || joinGlide->joinSeconds <= 1.0e-9
                            || time >= joinGlide->joinSeconds)
                            return natural;
                        const auto u = static_cast<float>(
                            juce::jlimit(0.0, 1.0, time / joinGlide->joinSeconds));
                        const auto shaped = u * u * (3.0f - 2.0f * u);
                        return static_cast<float>(joinGlide->leadMidi)
                            + (natural - static_cast<float>(joinGlide->leadMidi)) * shaped;
                    };
                    juce::Path controlLine;
                    auto controlLineOpen = false;
                    // Its own handles, or its part of a line it shares with its
                    // neighbours -- the corners either way, so the line passes
                    // through every point exactly.
                    const auto denseAuthored = maskNativePitch && nativeSharedMember == nullptr
                        && note.pitchControlPoints.empty() && !authoredPitch.empty();
                    const auto densePoints = denseAuthored ? ownPitchPoints(pitchPreviewNote(note))
                        : std::vector<PitchCurveEditPoint>{};
                    auto breaks = pitchLineBreaks(track, note, absoluteStart);
                    if (denseAuthored)
                    {
                        breaks.clear();
                        for (const auto& point : densePoints) breaks.push_back(point.timeSeconds);
                    }
                    juce::ignoreUnused(anchors);
                    for (std::size_t index = 1; index < breaks.size(); ++index)
                    {
                        const auto startTime = breaks[index - 1];
                        const auto endTime = breaks[index];
                        const auto pixelWidth = std::max(1.0f,
                            static_cast<float>(endTime - startTime) * pixelsPerSecond);
                        auto steps = std::max(2, static_cast<int>(std::ceil(pixelWidth / 3.0f)));
                        if (foldVibrato)
                            steps = std::max(steps,
                                static_cast<int>(std::ceil(pixelWidth / 2.0f)));
                        if (joinGlide && startTime < joinGlide->joinSeconds)
                            steps = std::max(steps,
                                static_cast<int>(std::ceil(pixelWidth / 2.0f)));
                        for (int step = 0; step <= steps; ++step)
                        {
                            if (controlLineOpen && step == 0) continue;
                            const auto amount = static_cast<double>(step) / steps;
                            const auto time = startTime + (endTime - startTime) * amount;
                            const auto curveX = x + static_cast<float>(time) * pixelsPerSecond;
                            auto midi = pitchLineMidiAt(track, note, absoluteStart, time,
                                denseAuthored ? &densePoints : nullptr);
                            if (foldVibrato)
                                midi += static_cast<float>(vibratoCentsAt(shownVibrato,
                                    juce::jlimit(0.0, note.durationSeconds, time)) / 100.0);
                            const auto curveY = midiToY(midi) + rowHeight * 0.5f;
                            if (!controlLineOpen)
                            {
                                controlLine.startNewSubPath(curveX, curveY);
                                controlLineOpen = true;
                            }
                            else controlLine.lineTo(curveX, curveY);
                        }
                    }
                    const auto noteSelected = selectedNotes.contains(note.id.toStdString())
                        || note.id == selectedNote;
                    {
                        juce::Graphics::ScopedSaveState pitchClip(g);
                        if (maskNativePitch)
                        {
                            const auto& visiblePitch = !sourceEditMode && nativeSharedMember != nullptr
                                ? voicedByLine[nativeSharedMember->line.get()] : notePitchVisibility;
                            reduceNativePitchClip(g, visiblePitch, pixelsPerSecond, timeToX(0));
                        }
                        g.setColour(Palette::pitchLine.withAlpha(noteSelected ? 1.0f : 0.72f));
                        if (dataIsUtau || pointsOwnTheLine)
                        {
                            if (markConsonants)
                            {
                                const auto& ranges = nativeSharedMember != nullptr
                                    ? noiseByLine[nativeSharedMember->line.get()] : noiseByNote[note.id];
                                strokeNativePitchWithNoise(g, controlLine, ranges, pixelsPerSecond, timeToX(0));
                            }
                            else g.strokePath(controlLine,
                                juce::PathStrokeType(2.0f, juce::PathStrokeType::curved));
                        }
                    }
                    if (maskNativePitch)
                    {
                        const auto& visiblePitch = !sourceEditMode && nativeSharedMember != nullptr
                            ? voicedByLine[nativeSharedMember->line.get()] : notePitchVisibility;
                        const auto& guides = !sourceEditMode && nativeSharedMember != nullptr
                            ? authoredByLine[nativeSharedMember->line.get()] : authoredPitch;
                        g.setColour(Palette::pitchLine.withAlpha(noteSelected ? .9f : .68f));
                        strokeNativeAuthoredPitchDots(g, controlLine, visiblePitch, guides,
                            pixelsPerSecond, timeToX(0));
                    }
                }

                if (tool == Tool::points && !dsPitchOffsetMode)
                {
                    const auto& anchors = draggedNote == note.id
                            && dragMode == DragMode::pointPitch && !pitchStroke.empty()
                        ? pitchStroke : pitchAnchorsFor(note);
                    const auto noteSelected = selectedNotes.contains(note.id.toStdString())
                        || note.id == selectedNote;
                    // Which are shown is settled before a drag begins and
                    // holds until it ends: a handle appearing or vanishing
                    // under a drag reads as the drag having made or taken it.
                    const auto* handleMember = sharedLinesFor(track, false).memberFor(note.id);
                    for (std::size_t index = 0; index < anchors.size(); ++index)
                    {
                        if (anchors[index].diffSingerRestoreSupport
                            || (dsPitchRestoreMode && note.diffSingerPitchReference.empty())) continue;
                        const auto active = (draggedNote == note.id
                            && static_cast<int>(index) == draggedPitchAnchor)
                            || pointIsSelected(PointSelection::pitch,note.id,static_cast<int>(index))
                            || (pitchSelectionCurrent && selectedPoint->noteId == note.id
                                && selectedPoint->index == static_cast<int>(index)
                                && dragMode != DragMode::pointPitch);
                        // Past the moment a later note's bend takes over, a
                        // point of this one decides nothing, and offering it
                        // is how a point came to be dragged for nothing.
                        if (handleMember != nullptr && !active
                            && !handleMember->owns(absoluteStart + anchors[index].timeSeconds))
                            continue;
                        const auto anchorX = x + static_cast<float>(anchors[index].timeSeconds)
                            * pixelsPerSecond;
                        const auto anchorY = midiToY(anchors[index].targetMidi + displayedPitchOffset(note, anchors[index].timeSeconds))
                            + rowHeight * 0.5f;
                        const auto radius = index == 0 || index + 1 == anchors.size() ? 4.8f : 4.0f;
                        g.setColour(active ? Palette::noteFill
                                           : Palette::panelRaised.withAlpha(noteSelected ? 1.0f : 0.82f));
                        g.fillEllipse(anchorX - radius, anchorY - radius, radius * 2.0f, radius * 2.0f);
                        g.setColour(Palette::noteLight.withAlpha(noteSelected ? 1.0f : 0.72f));
                        g.drawEllipse(anchorX - radius, anchorY - radius, radius * 2.0f, radius * 2.0f,
                                      active ? 2.0f : 1.4f);
                    }
                }

                if (note.connectedToPrevious)
                {
                    g.setColour(Palette::noteLight);
                    g.fillEllipse(bounds.getX() - 3.0f, bounds.getCentreY() - 3.0f, 6.0f, 6.0f);
                }
            }
            // Paint after this source's curves, within its overlap-focus layer.
            // A compact dark backing keeps labels readable over neighbouring ink.
            for (const auto& label : waveformLabels)
            {
                const juce::Font font(juce::jlimit(11.0f, 13.0f, rowHeight * 0.55f));
                auto badge = label.bounds;
                badge.setWidth(std::min(badge.getWidth(), font.getStringWidthFloat(label.text) + 8.0f));
                const auto badgeInk = rollRole(juce::Colour(0xff202629), Palette::panelRaised).withAlpha(0.94f);
                const auto badgeSurface = Palette::graphBackground.overlaidWith(badgeInk);
                g.setColour(badgeInk);
                g.fillRoundedRectangle(badge, 3.0f);
                g.setColour(rollText(Palette::text.withAlpha(label.alpha), badgeSurface));
                g.setFont(font);
                g.drawFittedText(label.text, badge.toNearestInt().reduced(4, 0),
                    juce::Justification::centredLeft, 1);
            }
        }
        ++trackIndex;
    }

    if (dsPitchOffsetMode && !sourceEditMode && showPitchLine) drawDiffSingerPitchOffsets(g);
    drawCvvcPhonemeTabs(g);
    if (!sourceEditMode && tool == Tool::flagCurve) drawFlagLane(g);
    if (!sourceEditMode && tool == Tool::amplitude)
    {
        const auto lane = amplitudeLaneBounds();
        const auto plot = amplitudeLanePlotBounds();
        g.setColour(Palette::panel.withAlpha(0.97f));
        g.fillRoundedRectangle(lane, 5.0f);
        g.setColour(Palette::border.brighter(0.18f));
        g.drawRoundedRectangle(lane, 5.0f, 1.2f);

        g.setFont(11.0f);
        g.setColour(Palette::text);
        g.drawText(I18n::uiText("响度包络（线性百分比）  纵向 0–")
                + juce::String(static_cast<int>(amplitudeLaneMaxPercent)) + "%",
            lane.toNearestInt().removeFromTop(28).reduced(8, 0).withTrimmedRight(104),
            juce::Justification::centredLeft, false);

        for (const auto zoomIn : { false, true })
        {
            const auto button = amplitudeZoomButtonBounds(zoomIn);
            g.setColour(Palette::panelRaised);
            g.fillRoundedRectangle(button, 3.0f);
            g.setColour(Palette::border.brighter(0.25f));
            g.drawRoundedRectangle(button, 3.0f, 1.0f);
            g.setColour(Palette::text);
            g.setFont(15.0f);
            g.drawText(amplitudeZoomLabel(zoomIn), button.toNearestInt(),
                       juce::Justification::centred, false);
        }

        g.setFont(10.0f);
        for (int guide = 0; guide <= 4; ++guide)
        {
            const auto percent = amplitudeLaneMaxPercent
                * static_cast<float>(guide) / 4.0f;
            const auto guideY = amplitudeLaneY(amplitudeDbFromPercent(percent));
            g.setColour(std::abs(percent - 100.0f) < 0.1f
                    ? rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(0.36f)
                    : Palette::grid.withAlpha(0.70f));
            g.drawHorizontalLine(static_cast<int>(guideY), plot.getX(), plot.getRight());
            g.setColour(Palette::textMuted);
            const auto label = percent <= 0.05f
                ? I18n::uiText("0 静音")
                : juce::String(static_cast<int>(std::lround(percent))) + "%";
            g.drawText(label, static_cast<int>(plot.getRight() + 4.0f),
                static_cast<int>(guideY - 8.0f), 49, 16,
                juce::Justification::centredRight, false);
        }

        if(std::any_of(noteHits.begin(),noteHits.end(),[&](const auto& hit){return usesCvvcEnvelope(hit.id);}))
        {
            const auto known=std::any_of(diffSingerPhonemes.begin(),diffSingerPhonemes.end(),[](const auto& entry){
                return std::any_of(entry.second.begin(),entry.second.end(),[](const auto& span){return span.mixEnvelope.available();});});
            g.setFont(11.f);g.setColour(Palette::text.withAlpha(.85f));
            auto legend=lane.toNearestInt().removeFromTop(28).withTrimmedLeft(300).withTrimmedRight(85);
            if(legend.getWidth()>80)g.drawFittedText(juce::String::fromUTF8(known
                ?"实线：实际淡化  ·  虚线／圆点：编辑值  ·  点击音素标签可独立编辑"
                :"手动包络 · 等待渲染交叉淡化"),legend,juce::Justification::centredLeft,1);
        }
        auto paintedEnvelopeCount = 0;
        juce::Graphics::ScopedSaveState clip(g);
        g.reduceClipRegion(plot.toNearestInt().expanded(7));
        // The audio first, so the envelope and its handles stay on top of it.
        drawAmplitudeLaneWaveforms(g);
        // These are guides, not handles: envelope editing never changes OTO timing.
        for (const auto& hit : noteHits)
        {
            if (!selectedNotes.contains(hit.id.toStdString())
                && !(selectedNotes.empty() && hit.id == selectedNote)) continue;
            const auto start = hit.startSeconds + hit.clipStartSeconds;
            paintOtoRegionGuides(g, plot, otoRegionGuidesFor(hit.id),
                [this, start](double seconds) { return timeToX(start + seconds); });
        }
        std::set<const void*> drawnEnvelopeGroups;
        for (const auto& hit : noteHits)
        {
            if (const auto* group = nativeEnvelopeMember(hit.id); group && !drawnEnvelopeGroups.insert(group->points.get()).second) continue;
            auto selected = selectedNotes.contains(hit.id.toStdString())
                || (selectedNotes.empty() && hit.id == selectedNote);
            bool utau = false;
            const auto* envelopeNote = findNote(hit.id, &utau);
            if (envelopeNote == nullptr || (!utau && !nativeNoteTimingEnabled(hit.id)) || usesCvvcEnvelope(hit.id)) continue;
            const auto absoluteStart = hit.startSeconds + hit.clipStartSeconds;
            auto envelope = amplitudeEnvelopeFor(*envelopeNote, absoluteStart);
            const auto* group = nativeEnvelopeMember(hit.id);
            const auto* draggingGroup = nativeEnvelopeMember(draggedNote);
            if (group)
                for (const auto& [trackId, envelopes] : nativeEnvelopeCache)
                    for (const auto& [id, other] : envelopes)
                        selected |= other.points == group->points && (selectedNotes.contains(id.toStdString()) || id == selectedNote);
            const auto participatesInEdit = hit.id == draggedNote || selected
                || (group && draggingGroup && group->points == draggingGroup->points);
            if (dragMode == DragMode::amplitudePoint && amplitudeDragUsesLane
                && !amplitudeStroke.empty() && participatesInEdit)
                envelope = mapAmplitudeEnvelopeToNote(amplitudeStroke, draggedNote, hit.id);
            juce::ignoreUnused(participatesInEdit);
            if (envelope.size() < 2) continue;

            ++paintedEnvelopeCount;
            const auto isMaster = hit.id == draggedNote || hit.id == selectedNote;
            const auto colour = rollInk(rollRole(juce::Colour(utau ? 0xffffa94d : 0xff64ffa3),
                utau ? Palette::heading : Palette::noteEdge).withAlpha(
                isMaster ? 1.0f : selected ? 0.76f : 0.32f), Palette::panel);
            juce::Path path;
            for (std::size_t index = 0; index < envelope.size(); ++index)
            {
                const auto point = juce::Point<float>(
                    timeToX(absoluteStart + envelope[index].timeSeconds),
                    amplitudeLaneY(envelope[index].gainDb));
                if (index == 0) path.startNewSubPath(point);
                else
                {
                    // DS curves use the same gain interpolation as its mixer.
                    // A straight chord between handles misrepresents a dB ramp
                    // in this linear-percent lane.
                    if (diffSingerNoteIds.contains(hit.id.toStdString()) && !envelope[index-1].linearToNext)
                    {
                        const auto& left=envelope[index-1];
                        const auto x0=timeToX(absoluteStart+left.timeSeconds);
                        for(float x=std::max(x0+1,plot.getX());x<std::min(point.x,plot.getRight());x+=2)
                        {
                            const auto mix=(x-x0)/std::max(.001f,point.x-x0);
                            path.lineTo(x,amplitudeLaneY(backend::envelopeDbBetween(left.gainDb,envelope[index].gainDb,mix,false)));
                        }
                    }
                    path.lineTo(point);
                }
            }
            g.setColour(colour);
            g.strokePath(path, juce::PathStrokeType(
                isMaster ? 2.8f : selected ? 2.1f : utau ? 1.35f : 2.0f));
            if(!utau&&envelopeNote->nativeEnvelope.mode!=0)
            {
                const auto result=effectiveAmplitudeEnvelope(*envelopeNote,absoluteStart,envelope);
                juce::Path combined;for(size_t i=0;i<result.size();++i)
                {const juce::Point<float> p(timeToX(absoluteStart+result[i].timeSeconds),amplitudeLaneY(result[i].gainDb));if(i==0)combined.startNewSubPath(p);else combined.lineTo(p);}
                const float dashes[]{5,3};juce::Path dashed;juce::PathStrokeType(2).createDashedStroke(dashed,combined,dashes,2);
                g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(selected?.95f:.5f));g.fillPath(dashed);
            }
            if(envelopeNote->utauTailFadeMode!=0||envelopeNote->utauTailFade.head.mode!=0)
                if(const auto tail=[&]{const auto last=envelopeNote->utauTailFadeMode!=0?tailFadeSpanFor(envelopeNote->id):std::nullopt;
                    return last?last:envelopeNote->utauTailFade.head.mode!=0?headEnvelopeSpanFor(envelopeNote->id):std::nullopt;}())
                {
                    const auto result=effectiveAmplitudeEnvelope(*envelopeNote,absoluteStart,envelope);
                    juce::Path faded;for(size_t i=0;i<result.size();++i)
                    {const auto p=juce::Point<float>(timeToX(absoluteStart+result[i].timeSeconds),amplitudeLaneY(result[i].gainDb));if(i==0)faded.startNewSubPath(p);else faded.lineTo(p);}
                    const float dashes[]{5.0f,3.0f};juce::Path dashed;juce::PathStrokeType(2.0f).createDashedStroke(dashed,faded,dashes,2);
                    g.setColour(rollRole(juce::Colour(0xff72d6aa), Palette::noteEdge).withAlpha(selected?.95f:.5f));g.fillPath(dashed);
                    const auto x=timeToX(absoluteStart+tail->startSeconds);
                    g.drawVerticalLine((int)x,plot.getY(),plot.getBottom());
                    const auto labelWidth=std::min(82,(int)(timeToX(absoluteStart+tail->endSeconds)-x)-8);
                    if(selected&&labelWidth>=40)g.drawText(juce::String::fromUTF8(envelopeNote->utauTailFadeMode!=0&&tailFadeSpanFor(envelopeNote->id)?"尾段淡出":"音头包络"),(int)x+5,(int)plot.getY()+22,labelWidth,16,juce::Justification::centredLeft);
                }
            for (std::size_t index = 0; index < envelope.size(); ++index)
            {
                const auto point = juce::Point<float>(
                    timeToX(absoluteStart + envelope[index].timeSeconds),
                    amplitudeLaneY(envelope[index].gainDb));
                const auto active = ((envelopeNote->id == draggedNote
                    || (group && draggingGroup && group->points == draggingGroup->points))
                    && amplitudeDragUsesLane
                    && static_cast<int>(index) == draggedAmplitudePoint)
                    || pointIsSelected(PointSelection::amplitude,envelopeNote->id,static_cast<int>(index));
                const auto radius = active ? 6.0f : selected ? 5.0f : 3.8f;
                g.setColour(active ? Palette::noteFill
                                   : Palette::panelRaised.withAlpha(selected ? 1.0f : 0.70f));
                g.fillEllipse(point.x - radius, point.y - radius,
                              radius * 2.0f, radius * 2.0f);
                g.setColour(colour);
                g.drawEllipse(point.x - radius, point.y - radius,
                              radius * 2.0f, radius * 2.0f, active ? 2.4f : 1.8f);
                if (active)
                {
                    const auto percent = amplitudePercentFromDb(envelope[index].gainDb);
                    const auto value = percent <= 0.05f
                        ? I18n::uiText("响度 0（静音）")
                        : juce::String(percent, 1) + "%  ("
                            + juce::String(envelope[index].gainDb, 1) + " dB)";
                    g.setColour(Palette::text);
                    g.drawText(value, static_cast<int>(point.x + 9.0f),
                        static_cast<int>(point.y - 20.0f), 142, 18,
                        juce::Justification::centredLeft, false);
                }
            }
        }
        paintedEnvelopeCount += drawCvvcEnvelopeLane(g, true);
        if (paintedEnvelopeCount == 0)
        {
            g.setColour(Palette::textMuted);
            g.drawFittedText(I18n::uiText("当前轨道没有可显示的响度包络"),
                plot.toNearestInt().reduced(12), juce::Justification::centred, 1);
        }
    }

    drawConnectionPitchCurves(g);
    drawNativeAudioOverlaps(g);

    if (draggedNote.isNotEmpty() && dragMode == DragMode::pitch)
    {
        g.setColour(Palette::noteLight.withAlpha(0.8f));
        g.drawHorizontalLine(static_cast<int>(midiToY(previewMidi) + rowHeight * 0.5f),
                             58.0f, static_cast<float>(getWidth()));
    }

    if (!pitchStroke.empty() && !pitchPlanPreview
        && (dragMode == DragMode::drawPitch || dragMode == DragMode::linePitch))
    {
        juce::Path preview;
        for (std::size_t index = 0; index < pitchStroke.size(); ++index)
        {
            const auto x = timeToX(pitchEditAbsoluteStart
                                   + pitchStroke[index].timeSeconds);
            const auto* previewNote=findNote(draggedNote);
            const auto y = midiToY(pitchStroke[index].targetMidi
                + (previewNote?displayedPitchOffset(*previewNote,pitchStroke[index].timeSeconds):0.0f)) + rowHeight * 0.5f;
            if (index == 0) preview.startNewSubPath(x, y);
            else preview.lineTo(x, y);
        }
        g.setColour(Palette::noteLight.withAlpha(0.96f));
        const auto* previewTrack = trackOf(draggedNote);
        const auto* previewNote = findNote(draggedNote);
        if(trackPitchStrokeActive)
        {
            juce::RectangleList<int> spans;
            for(const auto& track:snapshot.tracks)if(track.id==trackPitchStrokeTrack)
                for(const auto& clip:track.clips)for(const auto& note:clip.notes)if(!note.nativeUnpitched)
                {
                    const auto domain=trackPitchEditDomain(note);
                    if(!std::isfinite(domain.first)||!std::isfinite(domain.second))continue;
                    const auto left=timeToX(clip.startSeconds+note.startSeconds+domain.first);
                    const auto right=timeToX(clip.startSeconds+note.startSeconds+domain.second);
                    spans.add(juce::Rectangle<int>(static_cast<int>(std::floor(left)),0,std::max(1,static_cast<int>(std::ceil(right-left))),getHeight()));
                }
            const juce::Graphics::ScopedSaveState saved(g);g.reduceClipRegion(spans);
            g.strokePath(preview,juce::PathStrokeType(2.2f,juce::PathStrokeType::curved));
        }
        else if (previewTrack != nullptr && previewNote != nullptr && trackShowsAllNativeRegions(*previewTrack))
        {
            for (const auto& clip : previewTrack->clips)
                if (std::any_of(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return note.id == draggedNote; }))
                {
                    const auto& visible = nativePitchVisibilityFor(clip, *previewNote);
                    {
                        juce::Graphics::ScopedSaveState pitchClip(g);
                        reduceNativePitchClip(g, visible, pixelsPerSecond, timeToX(0));
                        if (consonantPitchDashed)
                        {
                            const auto sourceMask = consonantVoicing->rangesFor(clip.sourceFile);
                            const auto noise = sourceMask ? nativeNoiseDisplayRanges(clip, *previewNote, *sourceMask)
                                : measuredUnvoicedDisplayRanges(clip, *previewNote);
                            strokeNativePitchWithNoise(g, preview, noise, pixelsPerSecond, timeToX(0));
                        }
                        else g.strokePath(preview, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved));
                    }
                    const NativeNoiseRanges authored {{pitchEditAbsoluteStart + pitchStroke.front().timeSeconds,
                                                       pitchEditAbsoluteStart + pitchStroke.back().timeSeconds}};
                    strokeNativeAuthoredPitchDots(g, preview, visible, authored, pixelsPerSecond, timeToX(0));
                    break;
                }
        }
        else g.strokePath(preview, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved));
    }

    if (dragMode == DragMode::marquee)
    {
        const auto selection = juce::Rectangle<float>(marqueeStart, marqueeCurrent);
        g.setColour(Palette::accent.withAlpha(0.12f));
        g.fillRect(selection);
        g.setColour(Palette::accentLight.withAlpha(0.92f));
        g.drawRect(selection, 1.0f);
    }

    if (dragMode == DragMode::drawNewNote && drawLengthSeconds > 0.0)
    {
        // Outlined rather than filled, because it is not a note yet: nothing
        // is written until the button comes up.
        const juce::Rectangle<float> pending(
            timeToX(drawStartSeconds), midiToY(drawMidi),
            static_cast<float>(drawLengthSeconds) * pixelsPerSecond, rowHeight);
        g.setColour(Palette::noteFill.withAlpha(0.28f));
        g.fillRoundedRectangle(pending, 3.0f);
        g.setColour(Palette::noteLight.withAlpha(0.95f));
        g.drawRoundedRectangle(pending, 3.0f, 1.6f);
    }

    drawBarNumbers(g);

    g.setColour(juce::Colours::red.withAlpha(0.9f));
    g.drawVerticalLine(static_cast<int>(timeToX(playheadSeconds)), 0.0f, static_cast<float>(getHeight()));
    // Scrolled away from the playhead, there was nothing on screen to say so,
    // and a zoom -- which keeps the playhead centred -- then looked like the
    // view had jumped somewhere at random.  A mark on the edge it lies beyond
    // keeps it present without dragging the view back to it.
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
    {
        const auto left = static_cast<float>(viewport->getViewPositionX());
        const auto right = left + static_cast<float>(viewport->getViewWidth());
        const auto top = static_cast<float>(viewport->getViewPositionY());
        const auto playheadX = timeToX(playheadSeconds);
        const auto beyondLeft = playheadX < left + 58.0f;
        if (beyondLeft || playheadX > right)
        {
            const auto edge = beyondLeft ? left + 58.0f : right;
            const auto point = beyondLeft ? edge : edge - 9.0f;
            const auto back = beyondLeft ? edge + 9.0f : edge;
            juce::Path arrow;
            arrow.startNewSubPath(point, top + 21.0f);
            arrow.lineTo(back, top + 16.0f);
            arrow.lineTo(back, top + 26.0f);
            arrow.closeSubPath();
            g.setColour(juce::Colours::red.withAlpha(0.85f));
            g.fillPath(arrow);
        }
    }
    if (noteDance.isEnabled())
    {
        std::sort(dancers.begin(), dancers.end(), [](const auto& a, const auto& b)
        {
            if (a.start != b.start) return a.start < b.start;
            if (a.bounds.getY() != b.bounds.getY()) return a.bounds.getY() < b.bounds.getY();
            return a.id < b.id;
        });
        const auto height = juce::jlimit(36.0f, 72.0f, rowHeight * 2.7f);
        for (std::size_t i = 0; i < dancers.size(); ++i)
            noteDance.draw(g, i, dancers[i].bounds, height);
    }
    drawKeyboard();
}

void PianoRollComponent::drawBarNumbers(juce::Graphics& g)
{
    // The arrangement ruler is out of sight while tuning, so the roll has to
    // say which bar is under the cursor by itself.  Numbering matches the
    // ruler's: bar 1 is the start of the song.
    const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
        * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
    if (barQuarters <= 1.0e-6) return;
    // The roll is taller and wider than its viewport, so the marks are pinned
    // to the visible corner the way the keyboard strip is.  Anchored to the
    // component instead they would scroll off with the top octave.
    auto viewX = 0.0f;
    auto viewY = 0.0f;
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
    {
        viewX = static_cast<float>(viewport->getViewPositionX());
        viewY = static_cast<float>(viewport->getViewPositionY());
    }
    // Laying out the text is the expensive part, and it was being done for
    // every bar in the piece even though the clip kept only the visible ones.
    const auto visible = g.getClipBounds().toFloat();
    const auto leftEdge = std::max(viewX + 58.0f, visible.getX());
    const auto firstSeconds = std::max(0.0,
        static_cast<double>(leftEdge - 58.0f) / pixelsPerSecond);
    const auto firstBar = std::max(0, static_cast<int>(std::floor(
        snapshot.quarterPositionForSeconds(firstSeconds) / barQuarters)));
    for (auto bar = firstBar;; ++bar)
    {
        const auto x = timeToX(snapshot.secondsForQuarterPosition(bar * barQuarters));
        if (x > visible.getRight() || x > static_cast<float>(getWidth())) break;
        // Behind the keyboard strip the mark would only be half readable.
        if (x < leftEdge) continue;
        const auto label = juce::String(bar + 1);
        const auto chip = juce::Rectangle<float>(
            x + 2.0f, viewY + 2.0f,
            9.0f + 6.0f * static_cast<float>(label.length()), 14.0f);
        // Over the notes rather than under them: behind a note the number
        // would disappear exactly where the editing is happening.  The chip
        // keeps it legible without hiding much of the note.
        g.setColour(Palette::background.withAlpha(0.72f));
        g.fillRoundedRectangle(chip, 3.0f);
        g.setColour(Palette::textMuted);
        g.setFont(10.0f);
        g.drawText(label, chip, juce::Justification::centred);
    }
}

std::optional<double> PianoRollComponent::pasteAnchorSeconds() const
{
    if (!hoverSeconds) return std::nullopt;
    return snapToGrid(*hoverSeconds);
}

void PianoRollComponent::mouseMove(const juce::MouseEvent& event)
{
    if (nativeLoudnessMode && !sourceEditMode)
    {
        bool overNote = false;
        for (const auto& hit : noteHits)
            if (hit.bounds.contains(event.position) && nativeNoteTimingEnabled(hit.id)) { overNote = true; break; }
        setMouseCursor(overNote ? juce::MouseCursor::UpDownResizeCursor : juce::MouseCursor::NormalCursor);
        setTooltip(strings.text("tool.nativeLoudnessHint"));
        return;
    }
    juce::String phonemeTip;
    for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
    {
        if (!it->bounds.contains(event.position)) continue;
        if (const auto found = diffSingerPhonemes.find(it->id.toStdString()); found != diffSingerPhonemes.end())
            for (const auto& phone : found->second)
                if (event.position.x >= timeToX(it->clipStartSeconds + it->startSeconds + phone.startSeconds)
                    && event.position.x < timeToX(it->clipStartSeconds + it->startSeconds + phone.endSeconds))
                { phonemeTip = phone.token; break; }
        if (phonemeTip.isNotEmpty()) break;
    }
    if (!sourceEditMode && tool == Tool::flagCurve && diffSingerFlagContext)
    {
        if (flagLaneLayerBounds(true).contains(event.position)) phonemeTip = I18n::uiText("直接编辑预测实参；蓝色虚线保留原始预测");
        else if (flagLaneLayerBounds(false).contains(event.position)) phonemeTip = I18n::uiText("编辑叠加在实参上的 FLAG 偏移，及 DYN 等控制");
        else if (flagLaneOffsetEnableBounds().contains(event.position)) phonemeTip = I18n::uiText("为所选音符开启或关闭偏移；实参与已画曲线保留");
    }
    for(const auto& hint:nativeOverlapHints)if(hint.first.contains(event.position)){phonemeTip=hint.second;break;}
    for(const auto& tab:cvvcPhonemeTabs())if(tab.bounds.contains(event.position))
    {phonemeTip=tab.token+I18n::uiText(" · 点击单独编辑此音素的响度／FLAG 包络；点击音符主体返回整音符");break;}
    if (!event.mods.isAnyMouseButtonDown() && cvvcBoundaryAt(event.position))
    {
        setTooltip(I18n::uiText("拖动音素分界：调整相邻音素时长，总长度不变（Esc 取消）"));
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        return;
    }
    if (tool==Tool::amplitude || tool==Tool::flagCurve) setMouseCursor(juce::MouseCursor::NormalCursor);
    setTooltip(phonemeTip);
    // Before the early returns below: a paste wants to know where the pointer
    // is whatever tool is in hand.
    hoverSeconds = std::max(0.0,
        static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond);
    if (sourceEditMode || (tool != Tool::note && tool != Tool::trim) || event.mods.isAnyMouseButtonDown()) return;
    if (const auto vibratoHandle = tool == Tool::note ? vibratoHandleAt(event.position) : std::nullopt)
    {
        setMouseCursor(vibratoHandle->which == VibratoHandle::depth
                || vibratoHandle->which == VibratoHandle::offset
            ? juce::MouseCursor::UpDownResizeCursor
            : juce::MouseCursor::LeftRightResizeCursor);
        return;
    }
    if (tool == Tool::note && (jieSplitHandleAt(event.position) || consonantHandleAt(event.position)))
    {
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
        return;
    }
    for(const auto& hit:noteHits)
        if(nativeNoteTimingEnabled(hit.id) && hit.bounds.contains(event.position)
            && event.position.x<=hit.bounds.getX()+6.f)
        {setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);return;}
    auto maximumDuration = 0.0;
    setMouseCursor(resizableTailAt(event.position, maximumDuration) != nullptr
        ? juce::MouseCursor::LeftRightResizeCursor
        : juce::MouseCursor::NormalCursor);
}

void PianoRollComponent::mouseExit(const juce::MouseEvent&)
{
    setTooltip({});
    // Off the roll there is no pointer to paste at, and the last place it was
    // is not where anyone means.
    hoverSeconds.reset();
    if (!sourceEditMode && tool == Tool::note && dragMode == DragMode::none)
        setMouseCursor(juce::MouseCursor::NormalCursor);
}

void PianoRollComponent::mouseDownSingle(const juce::MouseEvent& event)
{
    clearPointSelection();
    if (cvvcBoundaryMouseDown(event)) return;
    if (cvvcPhonemeTabMouseDown(event)) return;
    if(nativeOverlapSelector && event.mods.isLeftButtonDown() && !event.mods.isPopupMenu())
    {
        if(nativeOverlapSelectorClose.contains(event.position)){closeNativeOverlapSelector();return;}
        for(const auto& hit:nativeOverlapSourceHits)if(hit.bounds.contains(event.position))
        {
            const auto id=hit.noteId;dragMode=DragMode::none;draggedNote.clear();
            closeNativeOverlapSelector();setSelectedNoteIds({id});rebuildNoteHits();return;
        }
    }
    // Clicking elsewhere never opens a picker. It dismisses the previous target
    // before the ordinary editor/context-menu gesture is processed.
    if(nativeOverlapSelector)closeNativeOverlapSelector();
    if (connectionPitchMouseDown(event)) return;
    if (dsPitchOffsetMode) { offsetMouseDown(event); return; }
    hoverSeconds = std::max(0.0,
        static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond);
    noteDragPassedThreshold = false;
    nativeTrimDrag = false;
    sharedBoundaryNotes.clear();
    grabKeyboardFocus();
    if (const auto* viewport = findParentComponentOfClass<juce::Viewport>())
        if (event.x < viewport->getViewPositionX() + 58)
            return;
    if (sourceEditMode)
    {
        const RegionHandleHit* closest = nullptr;
        auto closestDistance = 7.0f;
        // Reverse order gives the active/later region priority at shared edges.
        for (auto it = regionHandleHits.rbegin(); it != regionHandleHits.rend(); ++it)
        {
            const auto distance = std::abs(event.position.x - it->x);
            if (distance <= closestDistance)
            {
                closest = &*it;
                closestDistance = distance;
            }
        }
        if (closest != nullptr)
        {
            draggedSampleRegion = closest->region;
            draggedRegionHandle = closest->handle;
            activeSampleRegion = closest->region;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            if (onSampleRegionEdited)
                onSampleRegionEdited(activeSampleRegion,
                    sampleRegions[static_cast<std::size_t>(activeSampleRegion)], false);
            repaint();
            return;
        }
    }
    // Hit regions must reflect the current model even when the first click
    // arrives before a pending repaint (common immediately after MIDI import,
    // track changes, zooming, or scrolling).
    rebuildNoteHits();
    if (nativeLoudnessMode && !sourceEditMode && event.mods.isLeftButtonDown())
    {
        clearPointSelection();
        draggedNote.clear(); dragMode = DragMode::none;
        for (auto hit = noteHits.rbegin(); hit != noteHits.rend(); ++hit)
        {
            if (!hit->bounds.contains(event.position) || !nativeNoteTimingEnabled(hit->id)) continue;
            const auto* note = findNote(hit->id); if (!note) continue;
            if (!selectedNotes.contains(hit->id.toStdString()))
            {
                if (!event.mods.isShiftDown() && !event.mods.isCommandDown()) selectedNotes.clear();
                selectedNotes.insert(hit->id.toStdString());
            }
            selectedNote = draggedNote = hit->id;
            loudnessStartGains.clear(); loudnessDeltaDb = 0;
            // Visible native hit targets define the editable scope, including
            // selections across visible source regions, never hidden tracks.
            for (const auto& target : noteHits)
                if (selectedNotes.contains(target.id.toStdString()) && nativeNoteTimingEnabled(target.id))
                    if (const auto* selected = findNote(target.id)) loudnessStartGains[target.id] = selected->gain;
            dragStartY = event.position.y;
            loudnessRevision = model.revisionNumber(); dragMode = DragMode::nativeLoudness;
            if (onNoteSelected) onNoteSelected(selectedNote);
            setMouseCursor(juce::MouseCursor::UpDownResizeCursor); repaint(); return;
        }
        playheadBeforeGesture = playheadSeconds;
        marqueeAddsToSelection = event.mods.isShiftDown() || event.mods.isCommandDown();
        if (!marqueeAddsToSelection)
        { selectedNote.clear(); selectedNotes.clear(); if (onNoteSelected) onNoteSelected({}); }
        marqueeShowsContextMenu = false;
        marqueeStart = marqueeCurrent = event.position; dragMode = DragMode::marquee;
        repaint(); return;
    }
    if(!sourceEditMode&&tool==Tool::connect&&!event.mods.isPopupMenu())
        if(const auto seam=nativeJoinAt(event.position))
        {
            // Neither click of a double click may toggle the connection first.
            draggedNote.clear();dragMode=DragMode::none;selectedNote=seam->first;
            selectedNotes={seam->first.toStdString(),seam->second.toStdString()};
            if(onNoteSelected)onNoteSelected(selectedNote);repaint();return;
        }
    // Wherever an edit begins is where the focus now is, so the playhead
    // follows it there.  That keeps the red line worth looking at, and gives a
    // zoom something real to hold still instead of a marker left at the start
    // of the piece.  Not for a right click, which asks a question rather than
    // editing, and not for the lane's own zoom buttons.
    const auto onLaneZoomButton = !sourceEditMode
        && ((tool == Tool::amplitude
             && (amplitudeZoomButtonBounds(true).contains(event.position)
                 || amplitudeZoomButtonBounds(false).contains(event.position)))
            || (tool == Tool::flagCurve
                && (flagLaneZoomButtonBounds(true).contains(event.position)
                    || flagLaneZoomButtonBounds(false).contains(event.position)
                    || flagLaneSwitchBounds().contains(event.position)
                    || flagLaneLayerBounds(true).contains(event.position)
                    || flagLaneLayerBounds(false).contains(event.position)
                    || flagLaneOffsetEnableBounds().contains(event.position))));
    playheadBeforeGesture = playheadSeconds;
    if (onEditPosition != nullptr && !event.mods.isPopupMenu() && !onLaneZoomButton)
        onEditPosition(std::max(0.0,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond));
    if (cvvcEnvelopeMouseDown(event, false)) return;
    if (!sourceEditMode && (tool!=Tool::amplitude || !amplitudeLaneBounds().contains(event.position))
        && (tool!=Tool::flagCurve || !flagLaneBounds().contains(event.position)))
    { cvvcSelectedNote.clear(); cvvcSelectedPart = 0; }
    if (!sourceEditMode && tool == Tool::amplitude)
    {
        const auto lane = amplitudeLaneBounds();
        if (lane.contains(event.position))
        {
            if (amplitudeZoomButtonBounds(true).contains(event.position)
                || amplitudeZoomButtonBounds(false).contains(event.position))
            {
                constexpr std::array<float, 5> zoomLevels { 100.0f, 150.0f, 200.0f,
                                                            300.0f, 400.0f };
                auto level = 0;
                auto distance = std::numeric_limits<float>::max();
                for (int index = 0; index < static_cast<int>(zoomLevels.size()); ++index)
                    if (const auto next = std::abs(amplitudeLaneMaxPercent
                                                   - zoomLevels[static_cast<std::size_t>(index)]);
                        next < distance)
                    {
                        distance = next;
                        level = index;
                    }
                level += amplitudeZoomButtonBounds(true).contains(event.position) ? -1 : 1;
                amplitudeLaneMaxPercent = zoomLevels[static_cast<std::size_t>(
                    juce::jlimit(0, static_cast<int>(zoomLevels.size()) - 1, level))];
                repaint();
                return;
            }

            const NoteData* note = nullptr;
            const NoteHit* hit = nullptr;
            std::vector<AmplitudeEnvelopePoint> envelope;
            auto closestPoint = -1;
            auto closestDistance = 11.0f;
            for (const auto& candidateHit : noteHits)
            {
                bool utau = false;
                const auto* candidateNote = findNote(candidateHit.id, &utau);
                if (candidateNote == nullptr || (!utau && !nativeNoteTimingEnabled(candidateHit.id))) continue;
                const auto candidateStart = candidateHit.startSeconds
                    + candidateHit.clipStartSeconds;
                const auto candidateEnvelope = amplitudeEnvelopeFor(
                    *candidateNote, candidateStart);
                for (std::size_t index = 0; index < candidateEnvelope.size(); ++index)
                {
                    const auto point = juce::Point<float>(
                        timeToX(candidateStart + candidateEnvelope[index].timeSeconds),
                        amplitudeLaneY(candidateEnvelope[index].gainDb));
                    const auto pointDistance = point.getDistanceFrom(event.position);
                    if (pointDistance <= closestDistance)
                    {
                        closestDistance = pointDistance;
                        closestPoint = static_cast<int>(index);
                        note = candidateNote;
                        hit = &candidateHit;
                        envelope = candidateEnvelope;
                    }
                }
            }
            if (closestPoint < 0 || note == nullptr || hit == nullptr) return;
            const auto absoluteStart = hit->startSeconds + hit->clipStartSeconds;
            const auto wasSelected = selectedNotes.contains(note->id.toStdString())
                || (selectedNotes.empty() && selectedNote == note->id);
            if (!wasSelected)
            {
                selectedNotes.clear();
                selectedNotes.insert(note->id.toStdString());
            }
            selectedNote = note->id;
            if (onNoteSelected) onNoteSelected(selectedNote);
            // Right-click removes an interior automation point.  The two
            // endpoints are retained so the envelope remains well-defined.
            if (event.mods.isPopupMenu())
            {
                deleteEnvelopePoint({note->id,closestPoint,model.revisionNumber(),PointSelection::amplitude});
                repaint();
                return;
            }

            draggedNote = note->id;
            draggedAmplitudePoint = closestPoint;
            selectedPoint = PointSelection {note->id,closestPoint,model.revisionNumber(),PointSelection::amplitude};
            envelopePointHasMoved = false;
            amplitudeStroke = std::move(envelope);
            amplitudeEditAbsoluteStart = absoluteStart;
            amplitudeEditMidi = note->midiNote;
            amplitudeDragUsesLane = true;
            amplitudePointMinimumTime = amplitudeStroke.front().timeSeconds;
            amplitudePointMaximumTime = amplitudeStroke.back().timeSeconds;
            if (closestPoint > 0)
                amplitudePointMinimumTime = amplitudeStroke[
                    static_cast<std::size_t>(closestPoint - 1)].timeSeconds + 0.005;
            if (closestPoint + 1 < static_cast<int>(amplitudeStroke.size()))
                amplitudePointMaximumTime = amplitudeStroke[
                    static_cast<std::size_t>(closestPoint + 1)].timeSeconds - 0.005;
            if (closestPoint == 0)
            {
                amplitudePointMinimumTime = nativeEnvelopeMember(note->id) ? amplitudeStroke.front().timeSeconds
                    : std::min(0.0, amplitudeStroke.front().timeSeconds);
                if (const auto span = utauSoundSpans.find(note->id.toStdString());
                    span != utauSoundSpans.end())
                    amplitudePointMinimumTime = std::min(0.0,
                        span->second.first - absoluteStart);
            }
            if (closestPoint + 1 == static_cast<int>(amplitudeStroke.size()))
                amplitudePointMaximumTime = nativeEnvelopeMember(note->id) ? amplitudeStroke.back().timeSeconds : note->durationSeconds;
            dragMode = DragMode::amplitudePoint;
            repaint();
            return;
        }
    }
    if (!sourceEditMode && tool == Tool::flagCurve)
    {
        const auto plot = flagLanePlotBounds();
        if (diffSingerFlagContext)
        {
            for (const bool actual : {true, false}) if (flagLaneLayerBounds(actual).contains(event.position))
            {
                if (!event.mods.isPopupMenu() && (!actual || hasDiffSingerParameters()))
                {
                    if (actual && onDiffSingerParameterLaneRequested) onDiffSingerParameterLaneRequested();
                    else setDiffSingerParameterLayer(actual);
                }
                return;
            }
            if (flagLaneOffsetEnableBounds().contains(event.position))
            {
                if (!event.mods.isPopupMenu()) model.setNotesUtauFlagCurveEnabled(diffSingerOffsetTargets(), !diffSingerOffsetsEnabled());
                return;
            }
        }
        if (flagLaneSwitchBounds().contains(event.position))
        {
            if (event.mods.isPopupMenu())
                showFlagLaneSwitchContextMenu(event.getScreenPosition());
            else
                showFlagLaneSwitchMenu();
            return;
        }
        for (const auto zoomIn : { true, false })
            if (flagLaneZoomButtonBounds(zoomIn).contains(event.position))
            {
                nudgeFlagLaneZoom(zoomIn);
                return;
            }
        if (flagLaneBounds().contains(event.position))
        {
            const NoteData* note = nullptr;
            if (flagMode == FlagEditMode::continuous && !event.mods.isPopupMenu()
                && plot.contains(event.position))
            {
                beginContinuousFlag(event.position);
                return;
            }
            const NoteHit* hit = nullptr;
            auto closestPoint = -1;
            auto closestDistance = 11.0f;
            for (const auto& candidateHit : noteHits)
            {
                bool utau = false;
                const auto* candidate = findNote(candidateHit.id, &utau);
                if (candidate == nullptr || !utau || !flagCurveActiveFor(candidate->id))
                    continue;
                const auto start = candidateHit.startSeconds + candidateHit.clipStartSeconds;
                const auto candidatePoints = flagLaneCurveFor(*candidate);
                // Clamped to the drawn stretch, exactly as the paint clamps it.
                // Testing the unclamped time looked for the handle where none
                // was drawn: a note in a phrase stops sounding well before its
                // written end, so its last handle -- drawn at that edge -- was
                // hunted for far to the right of itself and could not be picked
                // up at all.
                const auto [drawFrom, drawTo] =
                    flagLaneDrawnSpan(*candidate, start);
                const auto handleLeft = juce::jlimit(plot.getX(), plot.getRight(),
                                                     timeToX(drawFrom));
                const auto handleRight = juce::jlimit(plot.getX(), plot.getRight(),
                                                      timeToX(drawTo));
                for (std::size_t index = 0; index < candidatePoints.size(); ++index)
                {
                    const auto centre = juce::Point<float>(
                        juce::jlimit(handleLeft, handleRight,
                            timeToX(start + candidatePoints[index].timeSeconds)),
                        flagLaneY(candidatePoints[index].value));
                    if (const auto distance = centre.getDistanceFrom(event.position);
                        distance <= closestDistance)
                    {
                        closestDistance = distance;
                        closestPoint = static_cast<int>(index);
                        note = candidate;
                        hit = &candidateHit;
                    }
                }
            }
            if (note != nullptr && hit != nullptr)
            {
                if (event.mods.isPopupMenu())
                {
                    showFlagPointMenu(note->id, closestPoint,
                                      event.getScreenPosition());
                    return;
                }
                draggedNote = note->id;
                draggedFlagPoint = closestPoint;
                if (!selectedNotes.contains(note->id.toStdString())) selectedNotes = {note->id.toStdString()};
                selectedNote = note->id;
                if (onNoteSelected) onNoteSelected(selectedNote);
                selectedPoint = PointSelection {note->id,closestPoint,model.revisionNumber(),PointSelection::flag,0,laneFlag};
                envelopePointHasMoved = false;
                flagStroke = flagLaneCurveFor(*note);
                flagEditAbsoluteStart = hit->startSeconds + hit->clipStartSeconds;
                const auto [spanFrom, spanTo] = flagLaneSpanFor(*note);
                flagPointMinimumTime = closestPoint > 0
                    ? flagStroke[static_cast<std::size_t>(closestPoint) - 1].timeSeconds + 0.005
                    : spanFrom;
                flagPointMaximumTime = closestPoint + 1 < static_cast<int>(flagStroke.size())
                    ? flagStroke[static_cast<std::size_t>(closestPoint) + 1].timeSeconds - 0.005
                    : spanTo;
                // On its own a point is the note's flag level, not a corner of
                // a shape: it reads as that value everywhere, so moving it
                // along the note would change nothing while walking it off the
                // end it sits on.  It moves up and down only.
                if (flagStroke.size() == 1)
                    flagPointMinimumTime = flagPointMaximumTime =
                        flagStroke.front().timeSeconds;
                // b and bh do nothing past the onset, so their handles do not
                // go there; the rest of the flags have the whole note.
                flagPointMinimumTime = std::max(flagPointMinimumTime, spanFrom);
                flagPointMaximumTime = std::min(flagPointMaximumTime, spanTo);
                dragMode = DragMode::flagPoint;
                repaint();
                return;
            }
            // Nothing under the cursor: put a handle on whichever curve-bearing
            // note this column belongs to.  Notes overlap wherever a lead-in
            // reaches back into the note before it, so a column often belongs
            // to two of them -- taking the first left the note behind always
            // winning, and the one in front could not be edited there at all.
            // The one whose curve passes nearest the click is the one that was
            // being aimed at, which is also the one drawn under the cursor.
            if (!event.mods.isPopupMenu() && plot.contains(event.position))
            {
                const auto seconds = static_cast<double>(event.position.x - 58.0f)
                    / pixelsPerSecond;
                const NoteData* chosen = nullptr;
                auto chosenStart = 0.0;
                auto chosenDistance = std::numeric_limits<float>::max();
                for (const auto& candidateHit : noteHits)
                {
                    bool utau = false;
                    const auto* candidate = findNote(candidateHit.id, &utau);
                    if (candidate == nullptr || !utau || !flagCurveActiveFor(candidate->id))
                        continue;
                    const auto start = candidateHit.startSeconds + candidateHit.clipStartSeconds;
                    const auto [spanFrom, spanTo] = flagLaneSpanFor(*candidate);
                    if (seconds < start + spanFrom - 0.001
                        || seconds > start + spanTo + 0.001)
                        continue;
                    const auto points = flagLaneCurveFor(*candidate);
                    if (points.empty()) continue;
                    // Read where the curve is drawn at that column: held flat
                    // past either end, exactly as the paint holds it.
                    const auto at = juce::jlimit(spanFrom, spanTo, seconds - start);
                    const auto y = flagLaneY(flagCurveValueAt(points, at));
                    if (const auto distance = std::abs(y - event.position.y);
                        distance < chosenDistance)
                    {
                        chosenDistance = distance;
                        chosen = candidate;
                        chosenStart = start;
                    }
                }
                if (chosen != nullptr)
                {
                    auto points = flagLaneCurveFor(*chosen);
                    points.push_back({ seconds - chosenStart,
                                       flagValueFromLaneY(event.position.y) });
                    model.setNoteUtauFlagCurve(chosen->id, laneFlag, std::move(points));
                    return;
                }
            }
            return;
        }
    }
    if (!sourceEditMode && tool == Tool::points)
    {
        juce::String closestNoteId;
        auto closestAnchor = -1;
        auto closestDistance = 9.0f;
        double closestAbsoluteStart = 0.0;
        // Every visible note exposes its anchors in point mode -- the native
        // renderer reads the points too, so they mean the same thing there.
        // Tested before note rectangles so endpoints outside the hollow UTAU
        // block remain directly draggable.
        for (const auto& hit : noteHits)
        {
            if (const auto* note = findNote(hit.id); note != nullptr)
            {
                auto& anchors = pitchAnchorsFor(*note);
                for (std::size_t index = 0; index < anchors.size(); ++index)
                {
                    if (anchors[index].diffSingerRestoreSupport
                        || (dsPitchRestoreMode && note->diffSingerPitchReference.empty())) continue;
                    // Only the handles that are drawn can be taken hold of.
                    if (!pitchHandleOffered(*note, hit.startSeconds + hit.clipStartSeconds
                                                       + anchors[index].timeSeconds))
                        continue;
                    const auto anchor = juce::Point<float>(
                        timeToX(hit.startSeconds + hit.clipStartSeconds
                                + anchors[index].timeSeconds),
                        midiToY(anchors[index].targetMidi + displayedPitchOffset(*note, anchors[index].timeSeconds)) + rowHeight * 0.5f);
                    const auto distance = anchor.getDistanceFrom(event.position);
                    if (distance <= closestDistance)
                    {
                        closestDistance = distance;
                        closestNoteId = note->id;
                        closestAnchor = static_cast<int>(index);
                        closestAbsoluteStart = hit.startSeconds + hit.clipStartSeconds;
                    }
                }
            }
        }
        if (closestNoteId.isNotEmpty())
        {
            if (const auto* note = findNote(closestNoteId))
            {
                selectedNote = closestNoteId;
                // A native double-click may start on a pitch handle. Keep the
                // current batch through its initial press as well as the body.
                if (!nativeNoteTimingEnabled(closestNoteId)
                    || !selectedNotes.contains(closestNoteId.toStdString()))
                {
                    selectedNotes.clear();
                    selectedNotes.insert(closestNoteId.toStdString());
                }
                if (onNoteSelected) onNoteSelected(closestNoteId);
                if (event.mods.isPopupMenu())
                {
                    draggedNote.clear();
                    draggedPitchAnchor = -1;
                    pitchStroke.clear();
                    dragMode = DragMode::none;
                    repaint();
                    showPitchCurveShapeMenu(closestNoteId, closestAnchor,
                                            event.getScreenPosition());
                    return;
                }
                if (dsPitchRestoreMode)
                {
                    if (event.getNumberOfClicks() < 2)
                    {
                        const auto points = pitchAnchorsFor(*note);
                        model.restoreDiffSingerPitchPoint(model.revisionNumber(), closestNoteId, points, closestAnchor);
                        pitchAnchorCache.clear();connectionPitchCache.clear();protocolPitchCache.clear();clearPitchPlanPreview(); sharedLineCache.clear(); nativeEnvelopeCache.clear(); draggedSharedKey = 0;
                        rebuildLayout();
                    }
                    return;
                }
                draggedNote = closestNoteId;
                draggedPitchAnchor = closestAnchor;
                pitchEditAbsoluteStart = closestAbsoluteStart;
                pitchStroke = pitchAnchorsFor(*note);
                selectedPoint = PointSelection {
                    closestNoteId, closestAnchor, model.revisionNumber() };
                repaint();
                pointPitchHasMoved = false;
                pointPitchPressPosition = event.position;
                pitchStroke[static_cast<size_t>(closestAnchor)].diffSingerRestoreSupport = false;
                // Every anchor can move horizontally, the endpoints included.
                // What differs is how far: an endpoint is bounded by its one
                // neighbour on the inside and by the neighbouring note on the
                // outside, where an interior anchor is bounded on both sides.
                pointDragCanMoveHorizontally = pitchStroke.size() > 1;
                pointDragMinimumTime = 0.0;
                pointDragMaximumTime = note->durationSeconds;
                pointDragOwnedUntil = std::numeric_limits<double>::infinity();

                const PositionedUtauNote positionedNote {
                    note->id, closestAbsoluteStart,
                    closestAbsoluteStart + note->durationSeconds,
                    note->utauAutoPitchTransition
                };
                if (closestAnchor == 0)
                {
                    auto minimumAbsolute = positionedNote.startSeconds;
                    if (const auto span = utauSoundSpans.find(note->id.toStdString());
                        span != utauSoundSpans.end())
                        minimumAbsolute = std::min(minimumAbsolute, span->second.first);
                    // A head reaches back as far as the last point the note
                    // before it shows, and no further: the two may stand on one
                    // vertical line, never cross.  Past it, that point would be
                    // overruled by this one and vanish from the note before.
                    // A head a file already put further back than that is
                    // left where it is, not pulled forward on being taken.
                    if (const auto previous = previousUtauNoteFor(note->id))
                        minimumAbsolute = std::min(
                            lastShownPitchPoint(previous->id, previous->startSeconds)
                                .value_or(previous->startSeconds),
                            positionedNote.startSeconds + pitchStroke.front().timeSeconds);
                    pointDragMinimumTime = minimumAbsolute - positionedNote.startSeconds;
                }
                if (closestAnchor + 1 == static_cast<int>(pitchStroke.size())
                    && closestAnchor > 0)
                {
                    // The mirror of the head: a tail runs forward as far as the
                    // first point of the note after it, through a gap or across
                    // the boundary alike.  Both ends are bounded by each other
                    // and by nothing else, so the pair can be brought as close
                    // together as the user wants without ever crossing.
                    auto maximumAbsolute = positionedNote.endSeconds;
                    // As far as the first point the note after it shows: onto
                    // the same vertical line at most.
                    if (const auto next = nextUtauNoteFor(note->id))
                        maximumAbsolute = std::max(
                            firstShownPitchPoint(next->id, next->startSeconds)
                                .value_or(next->startSeconds),
                            positionedNote.startSeconds + pitchStroke.back().timeSeconds);
                    pointDragMaximumTime = maximumAbsolute - positionedNote.startSeconds;
                }
                // On a shared line a point decides only the stretch its note
                // owns, which ends at the next note's first point: any point of
                // this note -- not only its last -- goes up to that vertical
                // line at most, and would vanish under the pointer past it.
                if (const auto* noteTrack = trackOf(note->id))
                    if (const auto* member = sharedLinesFor(*noteTrack).memberFor(note->id))
                        if (std::isfinite(member->takeover))
                            pointDragOwnedUntil = member->takeover - positionedNote.startSeconds;
                if (nativeNoteTimingEnabled(note->id))
                {
                    pointDragCanMoveHorizontally = true; // a sparse endpoint may be this note's only point
                    pointDragMinimumTime = -closestAbsoluteStart;
                    pointDragMaximumTime = std::numeric_limits<double>::max();
                    pointDragOwnedUntil = std::numeric_limits<double>::infinity();
                }
                dragMode = DragMode::pointPitch;
                repaint();
                return;
            }
        }
    }
    if (!sourceEditMode && tool == Tool::points && dsPitchRestoreMode && !event.mods.isPopupMenu()) return;
    if (!sourceEditMode && tool==Tool::note && !nativeLoudnessMode && selectedNotes.size()==2 && event.mods.isLeftButtonDown())
    {
        std::vector<const NoteHit*> pair;
        for(const auto& hit:noteHits)if(selectedNotes.contains(hit.id.toStdString())&&nativeNoteTimingEnabled(hit.id))pair.push_back(&hit);
        if(pair.size()==2)
        {
            if(pair[0]->clipStartSeconds+pair[0]->startSeconds>pair[1]->clipStartSeconds+pair[1]->startSeconds)std::swap(pair[0],pair[1]);
            const auto& a=*pair[0];const auto& b=*pair[1];const auto seam=a.clipStartSeconds+a.startSeconds+a.durationSeconds;
            const auto* track=trackOf(a.id);
            if(track&&track==trackOf(b.id)&&std::abs(event.position.x-timeToX(seam))<=6.f
                &&(a.bounds.expanded(6.f,0.f).contains(event.position)||b.bounds.expanded(6.f,0.f).contains(event.position))
                &&planNativeBoundaryStretch(*track,a.id,b.id,0))
            {
                selectedNote=a.id;draggedNote=a.id;sharedBoundaryNotes={a.id,b.id};sharedBoundaryRevision=model.revisionNumber();
                dragStartSeconds=previewStartSeconds=a.startSeconds;dragDurationSeconds=previewDurationSeconds=a.durationSeconds;
                dragClipStartSeconds=a.clipStartSeconds;dragStartMidi=previewMidi=a.midi;
                dragMode=DragMode::resizeNativeRight;clearPointSelection();
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);repaint();return;
            }
        }
    }
    if (!sourceEditMode && (tool == Tool::trim || (tool == Tool::note && nativeStretchMode)) && event.mods.isLeftButtonDown())
    {
        // Only edge handles edit audio in this tool; the body selects it.
        double maximum = 0;
        const auto* tail = resizableTailAt(event.position, maximum);
        const NoteHit* chosen = tail && nativeNoteTimingEnabled(tail->id) ? tail : nullptr;
        bool left = false;
        if (!chosen) for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
            if (nativeNoteTimingEnabled(it->id) && it->bounds.contains(event.position))
            { chosen = &*it; left = event.position.x <= it->bounds.getX() + 6.f; break; }
        if (chosen)
        {
            const auto hit = *chosen;
            selectedNote = hit.id; selectedNotes.clear(); selectedNotes.insert(hit.id.toStdString());
            if (onNoteSelected) onNoteSelected(hit.id);
            if (left || tail)
            {
                draggedNote = hit.id; nativeTrimDrag = tool == Tool::trim;
                dragStartSeconds = previewStartSeconds = hit.startSeconds;
                dragDurationSeconds = previewDurationSeconds = hit.durationSeconds;
                dragClipStartSeconds = hit.clipStartSeconds;
                dragStartMidi = previewMidi = hit.midi;
                dragMode = left ? DragMode::resizeNativeLeft : DragMode::resizeNativeRight;
                setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            }
            else { draggedNote.clear(); dragMode = DragMode::none; }
            repaint(); return;
        }
    }
    if (!sourceEditMode && tool == Tool::note && event.mods.isLeftButtonDown())
    {
        if (const auto vibratoHandle = vibratoHandleAt(event.position))
        {
            selectedNote = vibratoHandle->noteId;
            const auto id = selectedNote.toStdString();
            if (!selectedNotes.contains(id))
            {
                selectedNotes.clear();
                selectedNotes.insert(id);
            }
            if (onNoteSelected) onNoteSelected(selectedNote);
            draggedNote = vibratoHandle->noteId;
            draggedVibratoHandle = vibratoHandle->which;
            vibratoDragAbsoluteStart = vibratoHandle->absoluteStart;
            if (const auto* note = findNote(vibratoHandle->noteId))
                previewVibrato = vibratoControlNote(*note);
            dragMode = DragMode::vibrato;
            setMouseCursor(draggedVibratoHandle == VibratoHandle::depth
                    || draggedVibratoHandle == VibratoHandle::offset
                ? juce::MouseCursor::UpDownResizeCursor
                : juce::MouseCursor::LeftRightResizeCursor);
            repaint();
            return;
        }
        if (const auto jieHandle = jieSplitHandleAt(event.position))
        {
            selectedNote = jieHandle->noteId;
            const auto id = selectedNote.toStdString();
            if (!selectedNotes.contains(id))
            {
                selectedNotes.clear();
                selectedNotes.insert(id);
            }
            if (onNoteSelected) onNoteSelected(selectedNote);
            draggedNote = jieHandle->noteId;
            draggedJieBoundary = jieHandle->boundary;
            jieSpanStartSeconds = jieHandle->spanStart;
            jieSpanEndSeconds = jieHandle->spanEnd;
            previewJieFractions = jieHandle->fractions;
            previewJieRegions = jieHandle->regions;
            dragMode = DragMode::jieSplit;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            repaint();
            return;
        }
        if (const auto consonant = consonantHandleAt(event.position))
        {
            selectedNote = consonant->noteId;
            const auto id = selectedNote.toStdString();
            if (!selectedNotes.contains(id))
            {
                selectedNotes.clear();
                selectedNotes.insert(id);
            }
            if (onNoteSelected) onNoteSelected(selectedNote);
            draggedNote = consonant->noteId;
            consonantNoteAbsoluteStart = consonant->absoluteStart;
            consonantScaledPreutterance = consonant->scaledPreutterance;
            consonantUnscaledPreutterance = consonant->unscaledPreutterance;
            consonantMinimumPreutterance = consonant->minimumPreutterance;
            consonantMaximumPreutterance = consonant->maximumPreutterance;
            previewConsonantPreutterance = consonant->currentPreutterance;
            previewConsonantVelocity = consonant->currentVelocity;
            previewConsonantVelocityBeforeDrag = consonant->currentVelocity;
            consonantSetsPin = consonant->setsPin;
            consonantOverlapSeconds = consonant->overlapSeconds;
            consonantCvvcLead = consonant->leadForVelocity;
            consonantDragTravelled = false;
            dragMode = DragMode::consonantLeadIn;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            repaint();
            return;
        }
        auto maximumDuration = 0.0;
        if (const auto* tail = resizableTailAt(event.position, maximumDuration))
        {
            const auto hit=*tail;
            selectedNote = hit.id;
            const auto id = selectedNote.toStdString();
            if (!selectedNotes.contains(id))
            {
                selectedNotes.clear();
                selectedNotes.insert(id);
            }
            if (onNoteSelected) onNoteSelected(selectedNote);
            draggedNote = hit.id;
            dragStartMidi = hit.midi;
            previewMidi = dragStartMidi;
            dragStartY = event.position.y;
            dragStartSeconds = hit.startSeconds;
            dragDurationSeconds = hit.durationSeconds;
            dragClipStartSeconds = hit.clipStartSeconds;
            previewStartSeconds = dragStartSeconds;
            previewDurationSeconds = dragDurationSeconds;
            resizeMaximumDurationSeconds = maximumDuration;
            dragMode = nativeNoteTimingEnabled(hit.id)?DragMode::resizeNativeRight:DragMode::resizeRight;
            setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
            repaint();
            return;
        }
    }
    if (!sourceEditMode && tool == Tool::draw && !event.mods.isPopupMenu()
        && drawActionMode == DrawAction::notes)
    {
        // Explicit note drawing is the same action for every composition engine.
        // Clicking an occupied cell still uses the existing free-start search.
        if (canDrawNotesOnFocusedTrack()) beginDrawingNote(event);
        return;
    }
    if (!sourceEditMode && pitchScopeMode == PitchScope::currentTrack && !event.mods.isPopupMenu()
        && (tool == Tool::line || (tool == Tool::draw && drawActionMode == DrawAction::pitchCurve)))
    {
        beginTrackPitchStroke(event);
        return; // Empty current-track space is never permission to add notes.
    }
    for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
    {
        const auto hit=*it;
        // UTAU noteHits deliberately use the real sounding span, which can
        // extend before the nominal note and end early.  The note popup must
        // instead be restricted to the standard grid rectangle shown by the
        // orange duration line.
        const auto nominalBounds = juce::Rectangle<float>(
            timeToX(hit.startSeconds + hit.clipStartSeconds),
            midiToY(hit.midi) + 2.0f,
            std::max(5.0f, static_cast<float>(hit.durationSeconds) * pixelsPerSecond),
            std::max(6.0f, rowHeight - 4.0f));
        if (!sourceEditMode && event.mods.isPopupMenu()
            && nominalBounds.contains(event.position))
        {
            // Pitch-anchor popup handling above has priority in point mode.
            selectedNote = hit.id;
            const auto id = selectedNote.toStdString();
            if (!selectedNotes.contains(id))
            {
                selectedNotes.clear();
                selectedNotes.insert(id);
            }
            draggedNote.clear();
            draggedPitchAnchor = -1;
            draggedAmplitudePoint = -1;
            pitchStroke.clear();
            amplitudeStroke.clear();
            dragMode = DragMode::none;
            if (onNoteSelected) onNoteSelected(selectedNote);
            repaint();
            showNoteContextMenu(selectedNote, event.getScreenPosition());
            return;
        }
        if (hit.bounds.contains(event.position))
        {
            selectedNote = hit.id;
            const auto id = selectedNote.toStdString();
            if (event.mods.isCommandDown())
            {
                if (selectedNotes.contains(id)) selectedNotes.erase(id);
                else selectedNotes.insert(id);
                if (!selectedNotes.contains(id))
                    selectedNote = selectedNotes.empty() ? juce::String()
                        : juce::String::fromUTF8(selectedNotes.begin()->c_str());
            }
            else if (event.mods.isShiftDown())
            {
                selectedNotes.insert(id);
            }
            else
            {
                if (!selectedNotes.contains(id))
                {
                    selectedNotes.clear();
                    selectedNotes.insert(id);
                }
            }
            if (onNoteSelected) onNoteSelected(selectedNote);
            if (selectedNote.isEmpty()) { repaint(); return; }
            if (const auto* note = findNote(hit.id); note && note->nativeUnpitched
                && (tool == Tool::points || tool == Tool::draw || tool == Tool::line))
            { draggedNote.clear(); dragMode = DragMode::none; repaint(); return; }
            if (tool == Tool::trim) { draggedNote.clear(); dragMode = DragMode::none; repaint(); return; }
            if (tool == Tool::connect && !sourceEditMode)
            {
                model.toggleNoteConnection(selectedNote);
                return;
            }
            if (!sourceEditMode && tool == Tool::points)
            {
                bool utau = false;
                if (const auto* note = findNote(hit.id, &utau); note != nullptr && utau)
                    (void) pitchAnchorsFor(*note);
                draggedNote.clear();
                draggedPitchAnchor = -1;
                dragMode = DragMode::none;
                repaint();
                return;
            }
            if (!sourceEditMode && tool == Tool::amplitude)
            {
                draggedNote.clear();
                draggedAmplitudePoint = -1;
                amplitudeStroke.clear();
                dragMode = DragMode::none;
                repaint();
                return;
            }
            if (!sourceEditMode && tool == Tool::flagCurve)
            {
                // Selecting a note while the flag lane is open is ordinary --
                // that is how a curve is put on one -- but dragging it around
                // is not what the click was for.
                draggedNote.clear();
                draggedFlagPoint = -1;
                flagStroke.clear();
                dragMode = DragMode::none;
                repaint();
                return;
            }
            draggedNote = hit.id;
            if (!sourceEditMode && (tool == Tool::draw || tool == Tool::line))
            {
                dragMode = tool == Tool::draw ? DragMode::drawPitch : DragMode::linePitch;
                pitchEditAbsoluteStart = static_cast<double>(hit.bounds.getX() - 58.0f)
                    / pixelsPerSecond;
                const auto local = juce::jlimit(0.0, hit.durationSeconds,
                    static_cast<double>(event.position.x - hit.bounds.getX()) / pixelsPerSecond);
                pitchStroke = { { local,
                    juce::jlimit(0.0f, 127.0f, pitchMidiFromY(event.position.y)
                        - (findNote(hit.id) ? displayedPitchOffset(*findNote(hit.id),local) : 0.0f)) } };
                repaint();
                return;
            }
            dragStartMidi = hit.midi;
            previewMidi = dragStartMidi;
            dragStartY = event.position.y;
            finePitchDrag = event.mods.isAltDown();
            dragStartSeconds = hit.startSeconds;
            dragDurationSeconds = hit.durationSeconds;
            dragClipStartSeconds = hit.clipStartSeconds;
            previewStartSeconds = dragStartSeconds;
            previewDurationSeconds = dragDurationSeconds;
            previewMoveDeltaSeconds = 0.0;
            minimumMoveDeltaSeconds = -dragStartSeconds;
            minimumDragPitchDelta=-127.f;maximumDragPitchDelta=127.f;
            for(const auto& track:snapshot.tracks)for(const auto& clip:track.clips)if(clipIsVisible(track,clip))
                for(const auto& note:clip.notes)if(selectedNotes.contains(note.id.toStdString()))
                {minimumDragPitchDelta=std::max(minimumDragPitchDelta,-note.midiNote);maximumDragPitchDelta=std::min(maximumDragPitchDelta,127.f-note.midiNote);}
            auto draggedNoteIsUtau = false;
            (void) findNote(hit.id, &draggedNoteIsUtau);
            if (draggedNoteIsUtau && !sourceEditMode)
            {
                // Keep the complete editable selection. Every region keeps its
                // ownership and contributes its local left boundary.
                for (const auto& track : snapshot.tracks)
                    for (const auto& clip : track.clips)
                        if (clipIsVisible(track,clip))
                            for (const auto& note : clip.notes)
                                if (selectedNotes.contains(note.id.toStdString()))
                                    minimumMoveDeltaSeconds=std::max(minimumMoveDeltaSeconds,-note.startSeconds);
            }
            if (!sourceEditMode && event.position.x <= hit.bounds.getX() + 6.0f)
                dragMode = nativeNoteTimingEnabled(hit.id)?DragMode::resizeNativeLeft:DragMode::resizeLeft;
            else if (draggedNoteIsUtau && !sourceEditMode)
                dragMode = DragMode::moveUtauNote;
            else if (nativeNoteTimingEnabled(hit.id))
            {
                dragMode = DragMode::moveNativeNote;
                minimumMoveDeltaSeconds = -dragClipStartSeconds - dragStartSeconds;
                for(const auto& track:snapshot.tracks) for(const auto& clip:track.clips)
                    for(const auto& note:clip.notes) if(selectedNotes.contains(note.id.toStdString()))
                        minimumMoveDeltaSeconds=std::max(minimumMoveDeltaSeconds,-clip.startSeconds-note.startSeconds);
            }
            else
                dragMode = DragMode::pitch;
            repaint();
            return;
        }
    }
    // Nothing under the pointer, so this may be the silence between two
    // notes.  Before the marquee, which is what an empty right-click meant.
    if (event.mods.isPopupMenu() && !sourceEditMode)
        if (const auto gap = gapAt(event.position))
        {
            showGapContextMenu(*gap, event.getScreenPosition());
            return;
        }
    // Pitch tools never create notes as a hidden empty-cell side effect.
    // Note creation has one explicit action in the existing draw-tool menu.
    if (!sourceEditMode && (tool == Tool::draw || tool == Tool::line)) return;
    marqueeAddsToSelection = event.mods.isShiftDown() || event.mods.isCommandDown();
    if (!marqueeAddsToSelection)
    {
        selectedNote.clear();
        selectedNotes.clear();
        if (onNoteSelected) onNoteSelected({});
    }
    if (!sourceEditMode && (tool == Tool::note || tool == Tool::points
                            || tool == Tool::amplitude || tool == Tool::flagCurve))
    {
        marqueeShowsContextMenu = event.mods.isPopupMenu();
        marqueeStart = event.position;
        marqueeCurrent = event.position;
        dragMode = DragMode::marquee;
    }
    repaint();
    if (onSeek)
        onSeek(std::max(0.0, static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond));
}

std::optional<std::pair<juce::String,juce::String>> PianoRollComponent::nativeJoinAt(juce::Point<float> point) const
{
    std::optional<std::pair<juce::String,juce::String>> result;auto distance=7.f;
    for(const auto& left:noteHits)for(const auto& right:noteHits)
    {
        if(left.id==right.id)continue;
        const auto end=left.clipStartSeconds+left.startSeconds+left.durationSeconds;
        if(std::abs(end-right.clipStartSeconds-right.startSeconds)>1.e-7)continue;
        const auto dx=std::abs(point.x-timeToX(end));
        if(dx>distance||point.y<std::min(left.bounds.getY(),right.bounds.getY())-7
            ||point.y>std::max(left.bounds.getBottom(),right.bounds.getBottom())+7)continue;
        if(model.canJoinNativeNotes(left.id,right.id)){distance=dx;result=std::pair{left.id,right.id};}
    }
    return result;
}

void PianoRollComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    if (nativeLoudnessMode && !sourceEditMode) return;
    if(connectionPitchDoubleClick(event))return;

    if (tool == Tool::trim) return;
    if(!sourceEditMode&&tool==Tool::connect&&!event.mods.isPopupMenu())
    {
        rebuildNoteHits();
        if(const auto seam=nativeJoinAt(event.position))
        {
            const auto joined=model.joinNativeNotes(seam->first,seam->second);
            if(joined.isNotEmpty())
            {
                draggedNote.clear();dragMode=DragMode::none;selectedNote=joined;
                selectedNotes={joined.toStdString()};if(onNoteSelected)onNoteSelected(joined);
                rebuildNoteHits();repaint();
            }
            return;
        }
    }
    if(!sourceEditMode)for(const auto& hit:nativeOverlapSourceHits)if(hit.bounds.contains(event.position))return;
    if (dsPitchOffsetMode) { offsetDoubleClick(event); return; }
    if (tool == Tool::points && dsPitchRestoreMode) return;
    if (sourceEditMode || tool == Tool::draw || tool == Tool::line) return;
    rebuildNoteHits();
    std::optional<NoteHit> doubleClickNote;
    if (tool == Tool::note && event.mods.isLeftButtonDown() && !event.mods.isPopupMenu())
        for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
            if (it->bounds.contains(event.position))
            {
                doubleClickNote=*it;
                break;
            }
    const auto* owner=doubleClickNote ? trackOf(doubleClickNote->id) : nullptr;
    const bool enterUtauLyric=owner && (effectiveTrackType(*owner)==TrackType::utau || trackIsDiffSinger(*owner));
    // In note mode a UTAU / DS note body always edits its lyric, including rests
    // and unresolved aliases. Phoneme overlays must not consume that gesture.
    if (!enterUtauLyric)
    {
        if (cvvcEnvelopeMouseDown(event, true)) return;
        if (cvvcPhonemeTabMouseDown(event)) return;
    }
    if (doubleClickNote)
    {
        const auto id=doubleClickNote->id;
        draggedNote.clear();
        draggedPitchAnchor = -1;
        dragMode = DragMode::none;
        pitchStroke.clear();
        selectedNote = id;
        if (!selectedNotes.contains(id.toStdString()))
        {
            selectedNotes.clear();
            selectedNotes.insert(id.toStdString());
        }
        nativeTrimDrag = false;
        previewMoveDeltaSeconds = 0;
        pointGroupGesture.reset(); cvvcLaneDrag.reset(); cvvcBoundaryDrag.reset();
        pendingCvvcBoundary.reset(); pendingCvvcHead.clear();
        if (onNoteSelected) onNoteSelected(id);
        if (enterUtauLyric)
        {
            beginInlineAliasEdit(*doubleClickNote);
            repaint();
            return;
        }
        if (nativeNoteTimingEnabled(id)) model.initializeNativeNotePitch(id);
        refreshFromModel(); repaint();
        return;
    }
    if (tool == Tool::amplitude)
    {
        if (dragMode == DragMode::marquee) dragMode = DragMode::none;
        if (!amplitudeLaneBounds().contains(event.position)) return;
        const NoteData* note = nullptr;
        const NoteHit* hit = nullptr;
        std::vector<AmplitudeEnvelopePoint> envelope;
        auto local = 0.0;
        auto closestLine = 14.0f;
        for (const auto& candidateHit : noteHits)
        {
            bool utau = false;
            const auto* candidateNote = findNote(candidateHit.id, &utau);
            if (candidateNote == nullptr || (!utau && !nativeNoteTimingEnabled(candidateHit.id))) continue;
            const auto candidateStart = candidateHit.startSeconds
                + candidateHit.clipStartSeconds;
            auto candidateEnvelope = amplitudeEnvelopeFor(*candidateNote, candidateStart);
            const auto candidateLocal = static_cast<double>(event.position.x
                - timeToX(candidateStart)) / pixelsPerSecond;
            if (candidateEnvelope.empty()
                || candidateLocal < candidateEnvelope.front().timeSeconds
                || candidateLocal > candidateEnvelope.back().timeSeconds)
                continue;
            const auto distance = std::abs(event.position.y
                - amplitudeLaneY(amplitudeDbAt(candidateEnvelope, candidateLocal)));
            if (distance <= closestLine)
            {
                closestLine = distance;
                note = candidateNote;
                hit = &candidateHit;
                local = candidateLocal;
                envelope = std::move(candidateEnvelope);
            }
        }
        if (note == nullptr || hit == nullptr) return;
        for (const auto& point : envelope)
            if (std::abs(point.timeSeconds - local) * pixelsPerSecond < 9.0f)
                return;
        envelope = envelopeWithPointAt(std::move(envelope), local);
        const auto wasSelected = selectedNotes.contains(note->id.toStdString())
            || (selectedNotes.empty() && selectedNote == note->id);
        if (!wasSelected)
        {
            selectedNotes.clear();
            selectedNotes.insert(note->id.toStdString());
        }
        selectedNote = note->id;
        if (onNoteSelected) onNoteSelected(selectedNote);
        commitAmplitudeEnvelopeToSelection(note->id, envelope);
        repaint();
        return;
    }
    if (tool == Tool::points)
    {
        // A double-click in point mode begins with the same mouse-down used by
        // marquee selection.  Cancel that pending marquee before inserting a
        // control point so its mouse-up cannot replace the selected note.
        if (dragMode == DragMode::marquee)
            dragMode = DragMode::none;
        const NoteHit* closestHit = nullptr;
        const NoteData* closestNote = nullptr;
        auto closestLocal = 0.0;
        auto closestDistance = std::numeric_limits<float>::max();
        // Point mode adds a handle on the visible native or UTAU curve.
        // It must never invoke the note tool's pitch-flattening action.
        for (const auto& hit : noteHits)
        {
            bool utau = false;
            const auto* note = findNote(hit.id, &utau);
            if (note == nullptr || (!utau && !nativeNoteTimingEnabled(hit.id))) continue;
            const auto absoluteStart = hit.startSeconds + hit.clipStartSeconds;
            const auto local = static_cast<double>(event.position.x - timeToX(absoluteStart))
                / pixelsPerSecond;
            auto& anchors = pitchAnchorsFor(*note);
            // On a shared line the point goes to the note that owns the moment
            // clicked -- which over a note's tail may be the next note's bend.
            const auto* noteTrack = trackOf(note->id);
            const auto* member = noteTrack != nullptr
                ? sharedLinesFor(*noteTrack).memberFor(note->id) : nullptr;
            if (member != nullptr && !anchors.empty())
            {
                const auto from = std::isfinite(member->ownFrom)
                    ? member->ownFrom - absoluteStart
                    : std::min(0.0, anchors.front().timeSeconds);
                // Into the turn as well: a point there becomes where the
                // turn begins.
                const auto to = std::min(member->takeover - absoluteStart - minimumAnchorSeparation,
                                         member->drawTo - absoluteStart);
                if (local < from || local > to) continue;
            }
            else if (local < 0.0 || local > note->durationSeconds) continue;
            const auto lineY = midiToY((member != nullptr
                    ? member->line->midiAt(absoluteStart + local) : pitchAt(anchors, local))
                    + displayedPitchOffset(*note,local))
                + rowHeight * 0.5f;
            const auto distance = std::abs(event.position.y - lineY);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestHit = &hit;
                closestNote = note;
                closestLocal = local;
            }
        }
        if (closestHit != nullptr && closestNote != nullptr)
        {
            auto& anchors = pitchAnchorsFor(*closestNote);
            for (const auto& anchor : anchors)
                if (!anchor.diffSingerRestoreSupport
                    && std::abs(anchor.timeSeconds - closestLocal) * pixelsPerSecond < 9.0f)
                    return;
            // At the pitch the line is at there -- inside a turn, that is the
            // turn, not the note's own line it is leaving.
            const auto* closestTrack = trackOf(closestNote->id);
            const auto* closestMember = closestTrack != nullptr
                ? sharedLinesFor(*closestTrack).memberFor(closestNote->id) : nullptr;
            const auto targetPitch = closestMember != nullptr
                ? closestMember->line->midiAt(closestHit->startSeconds + closestHit->clipStartSeconds
                                              + closestLocal)
                : pitchAt(anchors, closestLocal);
            // Restored samples are curve data, not editable handles. A double
            // click explicitly creates a new handle, reusing a coincident sample.
            auto sample = std::find_if(anchors.begin(), anchors.end(), [&](const auto& p) {
                return p.diffSingerRestoreSupport
                    && std::abs(p.timeSeconds - closestLocal) < minimumAnchorSeparation;
            });
            if (sample != anchors.end()) sample->diffSingerRestoreSupport = false;
            else
            {
                auto right = std::lower_bound(anchors.begin(), anchors.end(), closestLocal,
                    [](const auto& p, double t) { return p.timeSeconds < t; });
                const auto inRestored = right != anchors.end() && right->diffSingerRestoreSupport;
                anchors.push_back({ closestLocal, targetPitch,
                    inRestored ? PitchCurveShape::linear : PitchCurveShape::natural });
            }
            std::stable_sort(anchors.begin(), anchors.end(), [](const auto& left, const auto& right)
            {
                return left.timeSeconds < right.timeSeconds;
            });
            selectedNote = closestNote->id;
            selectedNotes.clear();
            selectedNotes.insert(selectedNote.toStdString());
            if (onNoteSelected) onNoteSelected(selectedNote);
            model.setNotePitchCurve(closestNote->id, anchors, true,
                closestLocal < -1.e-9 || closestLocal > closestNote->durationSeconds+1.e-9);
            repaint();
            return;
        }
        return;
    }
    for (auto it = noteHits.rbegin(); it != noteHits.rend(); ++it)
    {
        if (!it->bounds.contains(event.position)) continue;
        if (tool == Tool::note) return; // ordinary double-click alignment was handled above; F2/typing still rename
        for (const auto& track : snapshot.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == it->id)
                    {
                        draggedNote.clear();
                        dragMode = DragMode::none;
                        const auto local = juce::jlimit(0.0, note.durationSeconds,
                            static_cast<double>(event.position.x - it->bounds.getX())
                                / pixelsPerSecond);
                        if (tool == Tool::connect)
                        {
                            const auto created = model.splitNote(note.id, local);
                            if (created.isNotEmpty())
                            {
                                selectedNote = created;
                                selectedNotes.clear();
                                selectedNotes.insert(created.toStdString());
                                if (onNoteSelected) onNoteSelected(created);
                            }
                            return;
                        }
                        const auto noteStart = clip.startSeconds + note.startSeconds;
                        const auto boundary = noteStart + note.consonantSeconds;
                        const auto boundaryX = timeToX(boundary);
                        if (std::abs(event.position.x - boundaryX) <= 7.0f)
                        {
                            const auto quantized = snapToGrid(boundary);
                            model.setNoteAttack(note.id,
                                juce::jlimit(0.0, note.durationSeconds, quantized - noteStart),
                                note.attackSpeed);
                        }
                        else
                            model.transposeNotes({ note.id }, std::round(note.midiNote) - note.midiNote);
                        return;
                    }
        return;
    }
}

void PianoRollComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (dragMode == DragMode::nativeLoudness)
    {
        loudnessDeltaDb = (dragStartY - event.position.y) * 0.15f;
        repaint(); return;
    }
    if(pointGroupGesture){movePointGroup(event);return;}
    if(graphPointDragging){moveConnectionPitchPoint(event);return;}
    if (cvvcBoundaryDrag) { cvvcBoundaryMouseDrag(event); return; }
    if (cvvcLaneDrag) { cvvcEnvelopeMouseDrag(event); return; }
    if (dsPitchOffsetMode) { offsetMouseDrag(event); return; }
    if (dragMode == DragMode::continuousFlag)
    {
        drawContinuousFlag(event.position);
        return;
    }
    if (dragMode == DragMode::drawNewNote)
    {
        const auto pointer = std::max(0.0,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond);
        drawLengthSeconds = drawnLengthFor(pointer - drawStartSeconds,
                                           drawUnitSeconds, drawAvailableSeconds);
        repaint();
        return;
    }
    if (draggedSampleRegion >= 0
        && draggedSampleRegion < static_cast<int>(sampleRegions.size()))
    {
        auto& region = sampleRegions[static_cast<std::size_t>(draggedSampleRegion)];
        const auto seconds = std::max(0.0,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond);
        constexpr double minimum = 0.001;
        switch (draggedRegionHandle)
        {
            case RegionHandle::start:
            {
                const auto next = std::min(region.regionEndSeconds - minimum, seconds);
                const auto fixedEnd = region.regionStartSeconds + region.fixedDurationSeconds;
                region.regionStartSeconds = next;
                region.fixedDurationSeconds = juce::jlimit(0.0,
                    region.regionEndSeconds - region.regionStartSeconds,
                    fixedEnd - region.regionStartSeconds);
                region.alignmentSeconds = juce::jlimit(region.regionStartSeconds,
                    region.regionEndSeconds, region.alignmentSeconds);
                break;
            }
            case RegionHandle::fixedEnd:
                region.fixedDurationSeconds = juce::jlimit(0.0,
                    region.regionEndSeconds - region.regionStartSeconds,
                    seconds - region.regionStartSeconds);
                break;
            case RegionHandle::alignment:
                region.alignmentSeconds = juce::jlimit(region.regionStartSeconds,
                    region.regionEndSeconds, seconds);
                break;
            case RegionHandle::end:
                region.regionEndSeconds = std::max(region.regionStartSeconds + minimum, seconds);
                region.fixedDurationSeconds = std::min(region.fixedDurationSeconds,
                    region.regionEndSeconds - region.regionStartSeconds);
                region.alignmentSeconds = std::min(region.alignmentSeconds, region.regionEndSeconds);
                break;
            case RegionHandle::none: break;
        }
        if (onSampleRegionEdited) onSampleRegionEdited(draggedSampleRegion, region, false);
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::marquee)
    {
        marqueeCurrent = event.position;
        updateMarqueeAutoScroll(marqueeCurrent);
        repaintDrag(event.position.x);
        return;
    }
    if (trackPitchStrokeActive)
    {
        const auto absolute=std::max(0.0,static_cast<double>(event.position.x-58)/pixelsPerSecond);
        PitchCurveEditPoint point{absolute,juce::jlimit(0.f,127.f,pitchMidiFromY(event.position.y)),PitchCurveShape::linear};
        if(dragMode==DragMode::linePitch)
        {if(pitchStroke.size()==1)pitchStroke.push_back(point);else pitchStroke.back()=point;}
        else if(pitchStroke.empty()||std::abs(pitchStroke.back().timeSeconds-point.timeSeconds)>=.001
            ||std::abs(pitchStroke.back().targetMidi-point.targetMidi)>=.02f)pitchStroke.push_back(point);
        std::vector<ProjectModel::PointCurveEdit> edits;juce::String error;
        if(planTrackPitchStroke(edits,error))updatePitchPlanPreview(edits,trackPitchStrokeRevision);else clearPitchPlanPreview();
        repaint();return;
    }
    if (draggedNote.isEmpty()) return;
    if (tool == Tool::connect) return;
    if (dragMode == DragMode::vibrato)
    {
        const auto* note = findNote(draggedNote);
        if (note == nullptr) return;
        const auto duration = std::max(1.0e-6, previewVibrato.durationSeconds);
        const auto local = juce::jlimit(0.0, duration,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond
                - vibratoDragAbsoluteStart);
        const auto swing = vibratoSpanOf(previewVibrato);
        const auto start = swing.start;
        const auto end = swing.end;
        const auto span = end - start;
        // Never shorter than a hundredth of the note, as the length never was.
        const auto shortest = duration * 0.01;
        switch (draggedVibratoHandle)
        {
            case VibratoHandle::length:
                // Dragging the start earlier makes the vibrato cover more of
                // the note; its end stays where it is.
                previewVibrato.vibratoLengthPercent = (end
                    - juce::jlimit(0.0, std::max(0.0, end - shortest), local)) / duration * 100.0;
                break;
            case VibratoHandle::end:
            {
                // Moving the end moves only the end: the start stays put, so
                // the length is whatever now lies between them.
                const auto moved = juce::jlimit(std::min(duration, start + shortest), duration, local);
                previewVibrato.vibratoEndPercent = moved / duration * 100.0;
                previewVibrato.vibratoLengthPercent = (moved - start) / duration * 100.0;
                break;
            }
            case VibratoHandle::fadeIn:
                if (span > 1.0e-9)
                    previewVibrato.vibratoFadeInPercent =
                        juce::jlimit(0.0, 100.0, (local - start) / span * 100.0);
                break;
            case VibratoHandle::fadeOut:
                if (span > 1.0e-9)
                    previewVibrato.vibratoFadeOutPercent =
                        juce::jlimit(0.0, 100.0, (end - local) / span * 100.0);
                break;
            case VibratoHandle::cycle:
                // The handle rides three quarters of the way through the first
                // cycle, so that is what the cursor distance measures.
                previewVibrato.vibratoCycleMs =
                    juce::jlimit(20.0, 2000.0, (local - start) / 0.75 * 1000.0);
                break;
            case VibratoHandle::depth:
            {
                const auto baseY = midiToY(note->midiNote) + rowHeight * 0.5f;
                const auto cents = (baseY - event.position.y) / rowHeight * 100.0f;
                previewVibrato.vibratoDepthCents =
                    juce::jlimit(1.0, 600.0, std::abs(static_cast<double>(cents)));
                break;
            }
            case VibratoHandle::offset:
            {
                // Held as a share of the depth, so the swing keeps its shape
                // and only its centre moves.
                const auto baseY = midiToY(note->midiNote) + rowHeight * 0.5f;
                const auto cents = (baseY - event.position.y) / rowHeight * 100.0f;
                const auto depth = std::max(1.0, previewVibrato.vibratoDepthCents);
                previewVibrato.vibratoOffsetPercent = juce::jlimit(-200.0, 200.0,
                    static_cast<double>(cents) / depth * 100.0);
                break;
            }
            case VibratoHandle::none: break;
        }
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::jieSplit)
    {
        const auto span = std::max(1.0e-9, jieSpanEndSeconds - jieSpanStartSeconds);
        const auto cursorSeconds = static_cast<double>(event.position.x - 58.0f)
            / pixelsPerSecond;
        const auto index = static_cast<std::size_t>(draggedJieBoundary);
        const auto lower = index == 0 ? 0.0 : previewJieFractions[index - 1];
        const auto upper = static_cast<int>(index) + 2 >= previewJieRegions
            ? 1.0 : previewJieFractions[index + 1];
        previewJieFractions[index] = juce::jlimit(lower, upper,
            (cursorSeconds - jieSpanStartSeconds) / span);
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::consonantLeadIn)
    {
        // The preview answers at once, as a handle should.  Only what a
        // release writes waits for the pointer to have gone somewhere.
        if (event.getDistanceFromDragStart() >= consonantClickSlopPixels)
            consonantDragTravelled = true;
        dragConsonantTo(std::max(0.0,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond));
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::amplitudePoint && draggedAmplitudePoint >= 0
        && draggedAmplitudePoint < static_cast<int>(amplitudeStroke.size()))
    {
        auto& point = amplitudeStroke[static_cast<std::size_t>(draggedAmplitudePoint)];
        const auto previous = point;
        point.gainDb = amplitudeDragUsesLane
            ? amplitudeDbFromLaneY(event.position.y)
            : amplitudeDbFromY(amplitudeEditMidi, event.position.y);
        const auto local = static_cast<double>(event.position.x - 58.0f)
            / pixelsPerSecond - amplitudeEditAbsoluteStart;
        point.timeSeconds = juce::jlimit(amplitudePointMinimumTime,
                                         amplitudePointMaximumTime, local);
        envelopePointHasMoved |= point.timeSeconds != previous.timeSeconds || point.gainDb != previous.gainDb;
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::flagPoint && draggedFlagPoint >= 0
        && draggedFlagPoint < static_cast<int>(flagStroke.size()))
    {
        auto& point = flagStroke[static_cast<std::size_t>(draggedFlagPoint)];
        const auto previous = point;
        point.value = flagValueFromLaneY(event.position.y);
        // Zoomed in, the reachable values are a slice of the flag's range;
        // rather than pinning the handle at the edge, the slice follows it.
        keepFlagValueInView(point.value);
        const auto local = static_cast<double>(event.position.x - 58.0f)
            / pixelsPerSecond - flagEditAbsoluteStart;
        point.timeSeconds = juce::jlimit(flagPointMinimumTime,
                                         flagPointMaximumTime, local);
        envelopePointHasMoved |= point.timeSeconds != previous.timeSeconds || point.value != previous.value;
        repaint();
        return;
    }
    if (dragMode == DragMode::pointPitch && draggedPitchAnchor >= 0
        && draggedPitchAnchor < static_cast<int>(pitchStroke.size()))
    {
        if (!pointPitchHasMoved)
        {
            if (event.position.getDistanceFrom(pointPitchPressPosition) < .1f) return;
            prepareSampledPitchPointEdit(pitchStroke, draggedPitchAnchor);
            pointPitchHasMoved = true;
        }
        auto& anchor = pitchStroke[static_cast<std::size_t>(draggedPitchAnchor)];

        const auto freeNative = nativeNoteTimingEnabled(draggedNote);
        if (freeNative)
            anchor.timeSeconds = std::max(-pitchEditAbsoluteStart,
                static_cast<double>(event.position.x-58.0f)/pixelsPerSecond-pitchEditAbsoluteStart);
        else if (pointDragCanMoveHorizontally)
        {
            const auto local = static_cast<double>(event.position.x - 58.0f)
                    / pixelsPerSecond - pitchEditAbsoluteStart;
            if (draggedPitchAnchor == 0 && pitchStroke.size() > 1)
            {
                // Order first: a range computed from neighbours can invert on a
                // degenerate note, and jlimit would be undefined there.
                const auto highest = pitchStroke[1].timeSeconds - minimumAnchorSeparation;
                anchor.timeSeconds = juce::jlimit(
                    std::min(pointDragMinimumTime, highest), highest, local);
            }
            else if (draggedPitchAnchor > 0
                     && draggedPitchAnchor + 1 < static_cast<int>(pitchStroke.size()))
                anchor.timeSeconds = juce::jlimit(
                    pitchStroke[static_cast<std::size_t>(draggedPitchAnchor - 1)].timeSeconds
                        + minimumAnchorSeparation,
                    pitchStroke[static_cast<std::size_t>(draggedPitchAnchor + 1)].timeSeconds
                        - minimumAnchorSeparation,
                    local);
            else if (draggedPitchAnchor > 0
                     && draggedPitchAnchor + 1 == static_cast<int>(pitchStroke.size()))
            {
                const auto lowest =
                    pitchStroke[static_cast<std::size_t>(draggedPitchAnchor - 1)].timeSeconds
                        + minimumAnchorSeparation;
                anchor.timeSeconds = juce::jlimit(
                    lowest, std::max(lowest, pointDragMaximumTime), local);
            }
            // Whichever point it is, not into the stretch a later note's bend
            // decides: it would stop counting and vanish from under the pointer.
            if (anchor.timeSeconds > pointDragOwnedUntil)
            {
                const auto lowest = draggedPitchAnchor > 0
                    ? pitchStroke[static_cast<std::size_t>(draggedPitchAnchor - 1)].timeSeconds
                        + minimumAnchorSeparation
                    : std::min(pointDragMinimumTime, pointDragOwnedUntil);
                anchor.timeSeconds = std::max(lowest, pointDragOwnedUntil);
            }
        }
        anchor.targetMidi = juce::jlimit(0.0f, 127.0f, pitchMidiFromY(event.position.y)
            - (findNote(draggedNote) ? displayedPitchOffset(*findNote(draggedNote),anchor.timeSeconds) : 0.0f));
        if (freeNative)
        {
            // Reorder around the moved point instead of clamping it against its neighbours.
            auto index = static_cast<size_t>(draggedPitchAnchor);
            while(index>0 && pitchStroke[index].timeSeconds<pitchStroke[index-1].timeSeconds)
            {std::swap(pitchStroke[index],pitchStroke[index-1]);--index;}
            while(index+1<pitchStroke.size() && pitchStroke[index].timeSeconds>pitchStroke[index+1].timeSeconds)
            {std::swap(pitchStroke[index],pitchStroke[index+1]);++index;}
            draggedPitchAnchor=static_cast<int>(index);
            if(selectedPoint)selectedPoint->index=draggedPitchAnchor;
        }
        repaintDrag(event.position.x);
        return;
    }
    if (dragMode == DragMode::drawPitch || dragMode == DragMode::linePitch)
    {
        const auto local = std::max(0.0,
            static_cast<double>(event.position.x - 58.0f) / pixelsPerSecond
                - pitchEditAbsoluteStart);
        PitchCurveEditPoint point { local,
            juce::jlimit(0.0f, 127.0f, pitchMidiFromY(event.position.y)-(findNote(draggedNote) ? displayedPitchOffset(*findNote(draggedNote),local) : 0.0f)) };
        if (dragMode == DragMode::linePitch)
        {
            if (pitchStroke.size() == 1) pitchStroke.push_back(point);
            else pitchStroke.back() = point;
        }
        else if (pitchStroke.empty()
                 || std::abs(pitchStroke.back().timeSeconds - point.timeSeconds) >= 0.001
                 || std::abs(pitchStroke.back().targetMidi - point.targetMidi) >= 0.02f)
            pitchStroke.push_back(point);
        repaintDrag(event.position.x);
        return;
    }
    if (sharedBoundaryNotes.size()==2)
    {
        if(model.revisionNumber()!=sharedBoundaryRevision){cancelPointDrag();repaint();return;}
        if(!noteDragPassedThreshold&&event.getDistanceFromDragStart()<6)return;
        noteDragPassedThreshold=true;
        const auto delta=nativeEdgeDragDelta(dragClipStartSeconds+dragStartSeconds+dragDurationSeconds,
            static_cast<double>(event.getDistanceFromDragStartX())/pixelsPerSecond,event.mods.isAltDown());
        const auto data=model.snapshot();
        for(const auto& track:data.tracks)
            if(auto plan=planNativeBoundaryStretch(track,sharedBoundaryNotes[0],sharedBoundaryNotes[1],delta))
                previewDurationSeconds=dragDurationSeconds+plan->delta;
        repaintDrag(event.position.x);return;
    }
    if (nativeTrimDrag)
    {
        auto delta = static_cast<double>(event.position.x - event.mouseDownPosition.x) / pixelsPerSecond;
        const auto left=dragMode==DragMode::resizeNativeLeft;
        const auto data=model.snapshot();
        for(const auto& track:data.tracks)if(trackShowsAllNativeRegions(track))
            for(const auto& clip:track.clips)
                if(auto plan=planNativeNoteTrim(clip,draggedNote,delta,left,track.allowNativeAudioOverlap))
                    delta=constrainedNativeTrimDelta(track,draggedNote,plan->delta,left);
        noteDragPassedThreshold = std::abs(delta) > 1.e-9;
        if (dragMode == DragMode::resizeNativeLeft)
        {
            previewStartSeconds = dragStartSeconds + delta;
            previewDurationSeconds = dragStartSeconds + dragDurationSeconds - previewStartSeconds;
        }
        else previewDurationSeconds = dragDurationSeconds + delta;
        repaintDrag(event.position.x); return;
    }
    // Moving and resizing a note wait for the pointer to travel first.
    // Placing a lyric means double clicking the note, and two presses with a
    // pixel of hand shake between them used to move it -- vertically as well,
    // since a press near a row boundary needs only one pixel to land on the
    // next pitch.  Point, envelope and boundary handles are deliberately not
    // gated: those are grabbed deliberately and want to answer at once.
    constexpr auto dragThresholdPixels = 6;
    if (dragMode == DragMode::pitch || dragMode == DragMode::moveUtauNote || dragMode == DragMode::moveNativeNote
        || dragMode == DragMode::resizeLeft || dragMode == DragMode::resizeRight
        || dragMode == DragMode::resizeNativeLeft || dragMode == DragMode::resizeNativeRight)
    {
        if (!noteDragPassedThreshold
            && event.getDistanceFromDragStart() < dragThresholdPixels)
            return;
        noteDragPassedThreshold = true;
    }
    if (dragMode == DragMode::pitch || dragMode == DragMode::moveUtauNote || dragMode == DragMode::moveNativeNote)
    {
        finePitchDrag = finePitchDrag || event.mods.isAltDown();
        previewMidi = finePitchDrag
            ? juce::jlimit(0.0f, 127.0f,
                dragStartMidi - (event.position.y - dragStartY) / rowHeight)
            : juce::jlimit(0.0f, 127.0f, std::round(yToMidi(event.position.y)));
        previewMidi=dragStartMidi+juce::jlimit(minimumDragPitchDelta,maximumDragPitchDelta,previewMidi-dragStartMidi);
        if (dragMode == DragMode::moveUtauNote || dragMode == DragMode::moveNativeNote)
        {
            const auto travelX = event.getDistanceFromDragStartX();
            auto delta = static_cast<double>(travelX) / pixelsPerSecond;
            // A drag straight up or down asks for a different pitch and
            // nothing else.  Snapping used to run regardless of whether the
            // pointer had moved sideways at all, so a note not sitting on the
            // grid -- and plenty do not -- was pulled onto it the moment its
            // pitch was touched, moving in time when nothing had asked it to.
            if (std::abs(travelX) < dragThresholdPixels)
                delta = 0.0;
            // Alt keeps the established fine/free-edit convention.  Normal
            // drags snap to the same unit note resizing uses.
            else if (!event.mods.isAltDown())
            {
                const auto originalAbsolute = dragClipStartSeconds + dragStartSeconds;
                const auto targetAbsolute = originalAbsolute + delta;
                const auto barQuarters = static_cast<double>(std::max(1, snapshot.numerator))
                    * 4.0 / static_cast<double>(std::max(1, snapshot.denominator));
                const auto quantum = std::max(1.0 / 1024.0,
                    barQuarters / (4.0 * static_cast<double>(noteEditDivision())));
                const auto targetQuarter = snapshot.quarterPositionForSeconds(targetAbsolute);
                const auto snapped = snapshot.secondsForQuarterPosition(
                    std::round(targetQuarter / quantum) * quantum);
                delta = snapped - originalAbsolute;
            }
            previewMoveDeltaSeconds = std::max(minimumMoveDeltaSeconds, delta);
        }
    }
    else if (dragMode == DragMode::resizeLeft || dragMode == DragMode::resizeNativeLeft)
    {
        const auto end = dragStartSeconds + dragDurationSeconds;
        auto next = dragStartSeconds
            + static_cast<double>(event.getDistanceFromDragStartX()) / pixelsPerSecond;
        if (dragMode == DragMode::resizeNativeLeft)
            next = dragStartSeconds + nativeEdgeDragDelta(dragClipStartSeconds + dragStartSeconds,
                static_cast<double>(event.getDistanceFromDragStartX()) / pixelsPerSecond, event.mods.isAltDown());
        else if (!event.mods.isAltDown())
        {
            const auto absolute = dragClipStartSeconds + next;
            next = snapToGrid(absolute) - dragClipStartSeconds;
        }
        const auto minimum=dragMode==DragMode::resizeNativeLeft?-dragClipStartSeconds:0.0;
        previewStartSeconds = juce::jlimit(minimum, end - 0.01, next);
        previewDurationSeconds = end - previewStartSeconds;
    }
    else if (dragMode == DragMode::resizeRight || dragMode == DragMode::resizeNativeRight)
    {
        const auto quantum = noteEditQuantumSeconds();
        const auto rawDuration = dragDurationSeconds
            + static_cast<double>(event.getDistanceFromDragStartX()) / pixelsPerSecond;
        const auto native=dragMode==DragMode::resizeNativeRight;
        const auto snappedDuration = native ? dragDurationSeconds + nativeEdgeDragDelta(
            dragClipStartSeconds + dragStartSeconds + dragDurationSeconds,
            rawDuration - dragDurationSeconds, event.mods.isAltDown())
            : std::round(rawDuration / quantum) * quantum;
        previewDurationSeconds = native?std::max(.01,snappedDuration):juce::jlimit(quantum,
            std::max(quantum, resizeMaximumDurationSeconds), snappedDuration);
    }
    repaintDrag(event.position.x);
}

void PianoRollComponent::updateMarqueeAutoScroll(juce::Point<float> position)
{
    marqueeScrollStep = 0;
    if (auto* viewport = findParentComponentOfClass<juce::Viewport>())
    {
        const auto view = viewport->getViewArea();
        // The keyboard strip sits over the left edge of the visible area, so
        // the band that triggers a scroll starts where the roll really begins.
        constexpr auto band = 28.0f;
        const auto left = static_cast<float>(view.getX()) + 58.0f + band;
        const auto right = static_cast<float>(view.getRight()) - band;
        // Steady to begin with, faster the further past the edge the pointer
        // goes, so a long sweep does not take all day and a small overshoot
        // does not bolt.
        const auto speed = [](float overshoot)
        {
            return juce::jlimit(6, 30, 6 + juce::roundToInt(overshoot / 4.0f));
        };
        if (position.x < left) marqueeScrollStep = -speed(left - position.x);
        else if (position.x > right) marqueeScrollStep = speed(position.x - right);
    }
    if (marqueeScrollStep != 0) { if (!isTimerRunning()) startTimerHz(30); }
    else stopTimer();
}

void PianoRollComponent::timerCallback()
{
    auto* viewport = findParentComponentOfClass<juce::Viewport>();
    if (dragMode != DragMode::marquee || marqueeScrollStep == 0 || viewport == nullptr)
    {
        stopTimer();
        return;
    }
    const auto before = viewport->getViewPositionX();
    const auto furthest = std::max(0, getWidth() - viewport->getViewWidth());
    viewport->setViewPosition(juce::jlimit(0, furthest, before + marqueeScrollStep),
                              viewport->getViewPositionY());
    if (viewport->getViewPositionX() == before)
    {
        // Already against the end of the roll: nothing further to reveal, and
        // repainting thirty times a second for it would be waste.
        stopTimer();
        return;
    }
    // The roll slid underneath a pointer that has not moved, so the corner of
    // the marquee follows it to its new place in the component.
    marqueeCurrent = getMouseXYRelative().toFloat();
    updateMarqueeAutoScroll(marqueeCurrent);
    repaintDrag(marqueeCurrent.x);
}

void PianoRollComponent::mouseUp(const juce::MouseEvent& event)
{
    if(graphPointDragging)
    {
        graphPointDragging=false;
        if(graphDragValid&&!pitchPendingEdits.empty())
        {
            juce::String error;
            if(model.setPointCurves(pitchPendingEdits,graphDragRevision,&error))
            {if(selectedPoint)selectedPoint->revision=model.revisionNumber();}
            else if(onEditRejected&&!error.isEmpty())onEditRejected(error);
        }
        graphDragValid=false;clearPitchPlanPreview();refreshFromModel();repaint();return;
    }

    if(pointGroupGesture)
    {
        auto gesture=std::move(*pointGroupGesture);pointGroupGesture.reset();cancelPointDrag();
        if(gesture.changed)commitPointGroup(std::move(gesture.curves),gesture.revision);
        repaint();return;
    }
    if (cvvcBoundaryDrag) { finishCvvcBoundaryDrag(); return; }
    if (cvvcLaneDrag) { finishCvvcEnvelopeDrag(); return; }
    if (dsPitchOffsetMode) { offsetMouseUp(); return; }
    if (dragMode == DragMode::continuousFlag) drawContinuousFlag(event.position);
    const auto openSelectionMenu = dragMode == DragMode::marquee && marqueeShowsContextMenu;
    finishDrag();
    if (openSelectionMenu && selectedNote.isNotEmpty() && nativeNoteTimingEnabled(selectedNote))
        showNoteContextMenu(selectedNote, event.getScreenPosition());
}

void PianoRollComponent::finishDrag()
{
    if (dragMode == DragMode::nativeLoudness)
    {
        std::map<juce::String, float> gains;
        for (const auto& [id, gain] : loudnessStartGains)
            if (const auto* note = findNote(id)) gains[id] = displayedNativeGain(*note);
        draggedNote.clear(); dragMode = DragMode::none; loudnessStartGains.clear();
        if (std::abs(loudnessDeltaDb) > 1.e-6f) model.setNoteGains(gains, loudnessRevision);
        refreshFromModel(); repaint(); return;
    }
    if (cvvcBoundaryDrag) { finishCvvcBoundaryDrag(); return; }
    if (cvvcLaneDrag) { finishCvvcEnvelopeDrag(); return; }
    lastDragBand = {};
    if (dragMode == DragMode::continuousFlag)
    {
        auto edits = std::move(continuousFlagPreview);
        const auto revision = continuousFlagRevision;
        cancelContinuousFlag();
        model.setNoteUtauFlagCurves(laneFlag, edits, revision);
        repaint();
        return;
    }
    if (trackPitchStrokeActive)
    {
        commitTrackPitchStroke();trackPitchStrokeActive=false;trackPitchStrokeTrack.clear();
        pitchStroke.clear();dragMode=DragMode::none;draggedNote.clear();repaint();return;
    }
    if (dragMode == DragMode::drawNewNote)
    {
        dragMode = DragMode::none;
        // The model searches for the free start again under its own lock: the
        // snapshot this preview was drawn from can be a note out of date.
        const auto id = model.addNoteFrom(drawClipId, drawStartSeconds,
                                          drawLengthSeconds, drawMidi);
        drawClipId.clear();
        drawLengthSeconds = 0.0;
        if (id.isNotEmpty())
        {
            selectedNote = id;
            selectedNotes.clear();
            selectedNotes.insert(id.toStdString());
            if (onNoteSelected) onNoteSelected(id);
        }
        repaint();
        return;
    }
    if (draggedSampleRegion >= 0
        && draggedSampleRegion < static_cast<int>(sampleRegions.size()))
    {
        if (onSampleRegionEdited)
            onSampleRegionEdited(draggedSampleRegion,
                sampleRegions[static_cast<std::size_t>(draggedSampleRegion)], true);
        draggedSampleRegion = -1;
        draggedRegionHandle = RegionHandle::none;
        setMouseCursor(juce::MouseCursor::NormalCursor);
        return;
    }
    if (dragMode == DragMode::marquee)
    {
        stopTimer();
        marqueeScrollStep = 0;
        const auto selection = juce::Rectangle<float>(marqueeStart, marqueeCurrent);
        // Drawing a box round some notes is choosing them, not choosing a
        // place, so the playhead goes back where it was.  Left at the corner
        // the box was started from, the next paste landed there -- just before
        // the note that had been selected, which is nobody's idea of where a
        // paste should go.  A press that never became a box is still someone
        // putting the playhead down, and keeps its new place.
        if (onEditPosition != nullptr && selection.getWidth() > 3.0f)
            onEditPosition(playheadBeforeGesture);
        if (!marqueeAddsToSelection) selectedNotes.clear();
        // By the block that is drawn, not by the sounding stretch.  A note's
        // lead-in lies over the note before it, so a band drawn wholly inside
        // one note's own block was picking up the next one as well -- and the
        // popup menu and the resize tail already go by the block for exactly
        // that reason.
        for (const auto& hit : noteHits)
            if (selection.intersects(hit.nominalBounds))
                selectedNotes.insert(hit.id.toStdString());
        selectedNote = selectedNotes.empty() ? juce::String()
            : juce::String::fromUTF8(selectedNotes.begin()->c_str());
        dragMode = DragMode::none;
        if (onNoteSelected) onNoteSelected(selectedNote);
        repaint();
        marqueeShowsContextMenu = false;
        return;
    }
    if (draggedNote.isEmpty()) return;
    const auto geometryDrag = dragMode == DragMode::resizeNativeLeft || dragMode == DragMode::resizeNativeRight
        || dragMode == DragMode::moveNativeNote || dragMode == DragMode::pitch
        || dragMode == DragMode::moveUtauNote
        || dragMode == DragMode::resizeLeft
        || dragMode == DragMode::resizeRight;
    if ((geometryDrag && !noteDragPassedThreshold)
        || (dragMode == DragMode::pointPitch && !pointPitchHasMoved))
    {
        // Left exactly as it was.  Committing a zero move still writes an undo
        // step and asks for a re-render, which is what made a double click
        // feel like an edit even when nothing had changed.
    }
    else if (dragMode == DragMode::drawPitch || dragMode == DragMode::linePitch
        || dragMode == DragMode::pointPitch)
    {
        bool committed = false;
        if (dragMode == DragMode::pointPitch)
        {
            bool authorSupport=false;
            if(const auto* note=findNote(draggedNote);note&&draggedPitchAnchor>=0&&draggedPitchAnchor<static_cast<int>(pitchStroke.size()))
            {
                const auto outside=[&](double time){return time < -1.e-9 || time > note->durationSeconds+1.e-9;};
                authorSupport=outside(pitchStroke[static_cast<std::size_t>(draggedPitchAnchor)].timeSeconds);
                const auto original=pitchAnchorsFor(*note);
                if(draggedPitchAnchor<static_cast<int>(original.size()))authorSupport|=outside(original[static_cast<std::size_t>(draggedPitchAnchor)].timeSeconds);
            }
            committed = model.setNotePitchCurve(draggedNote, std::move(pitchStroke), true, authorSupport);
        }
        else if (const auto* note = findNote(draggedNote); note && pitchStroke.size() >= 2)
        {
            std::vector<PitchCurveEditPoint> composed; bool store = false; juce::String error;
            if (composeNativePitchStroke(*note, pitchStroke, composed, store, error))
                committed = model.setNotePitchCurve(draggedNote, std::move(composed), store,
                    pitchStroke.front().timeSeconds < -1.e-9 || pitchStroke.back().timeSeconds > note->durationSeconds+1.e-9);
            else if (onEditRejected) onEditRejected(error);
        }
        if (committed && dragMode == DragMode::pointPitch && selectedPoint)
            selectedPoint = PointSelection {
                draggedNote, draggedPitchAnchor, model.revisionNumber() };
    }
    else if (dragMode == DragMode::amplitudePoint || dragMode == DragMode::flagPoint)
    {
        if (envelopePointHasMoved)
        {
            if (dragMode == DragMode::amplitudePoint) commitAmplitudeEnvelopeToSelection(draggedNote, amplitudeStroke);
            else model.setNoteUtauFlagCurve(draggedNote, laneFlag, flagStroke);
            if (selectedPoint) selectedPoint->revision = model.revisionNumber();
        }
    }
    else if (dragMode == DragMode::vibrato)
        model.setNotesVibrato({ draggedNote }, previewVibrato, true);
    else if (dragMode == DragMode::jieSplit)
        model.setNotesUtauJieSplit({ draggedNote }, previewJieFractions[0],
                                   previewJieFractions[1], previewJieFractions[2]);
    else if (dragMode == DragMode::consonantLeadIn)
    {
        if (!consonantDragTravelled)
        {
            // A click on the handle, which is where a seam is clicked.  It
            // leaves the note as it was: writing the preview back released
            // the pin, and a note behind a 拼字 note went back to its entry's
            // own timing each time anyone clicked between the two.
        }
        else if (consonantSetsPin)
            // 谋 pins the lead-in the drag asked for.  Backing into it through
            // the consonant velocity rewrites a setting the note carries --
            // and for a first region the entry calls a vowel the velocity does
            // not reach the lead-in at all, so the handle moved nothing.  The
            // overlap goes back in as it was: the two share one pin.
            model.setNoteUtauTimingOverrides(draggedNote, true,
                                             previewConsonantPreutterance,
                                             consonantOverlapSeconds);
        else
        {
            // A pinned preutterance outranks the velocity, so with one in
            // place this handle could be dragged the width of the screen and
            // the lead-in would not move.  Reaching for it is a request for
            // the lead-in the velocity gives, so the pin is released.
            if (const auto* dragged = findNote(draggedNote);
                dragged != nullptr && dragged->utauPreutteranceOverrideEnabled)
                model.setNoteUtauTimingOverrides(draggedNote, false, 0.0, 0.0);
            model.setNotesUtauConsonantVelocity({ draggedNote },
                                                previewConsonantVelocity);
            if (consonantCvvcLead)
            {
                pendingCvvcHead = draggedNote;
                pendingCvvcHeadRevision = model.revisionNumber();
                pendingCvvcHeadStart = consonantNoteAbsoluteStart - previewConsonantPreutterance;
            }
        }
    }
    else if (dragMode == DragMode::pitch
             || dragMode == DragMode::moveUtauNote || dragMode == DragMode::moveNativeNote)
    {
        std::vector<juce::String> ids;
        ids.reserve(selectedNotes.size());
        for (const auto& id : selectedNotes) ids.push_back(juce::String::fromUTF8(id.c_str()));
        if (ids.empty()) ids.push_back(draggedNote);
        if (dragMode == DragMode::moveUtauNote || dragMode == DragMode::moveNativeNote)
        {
            juce::String error;
            if (dragMode == DragMode::moveNativeNote)
            {
                if (std::abs(previewMoveDeltaSeconds)<1.e-9)
                {
                    if(std::abs(previewMidi-dragStartMidi)>1.e-6f)
                        model.transposeNotes(ids,previewMidi-dragStartMidi);
                }
                else model.moveNativeNotes(ids,previewMoveDeltaSeconds,previewMidi-dragStartMidi,&error);
            }
            else model.moveUtauNotes(ids, previewMoveDeltaSeconds, previewMidi-dragStartMidi, &error);
            if (error.isNotEmpty() && onEditRejected) onEditRejected(error);
        }
        else
            model.transposeNotes(ids, previewMidi - dragStartMidi);
    }
    else if (sharedBoundaryNotes.size()==2)
        model.stretchNativeNoteBoundary(sharedBoundaryNotes[0],sharedBoundaryNotes[1],
            previewDurationSeconds-dragDurationSeconds,sharedBoundaryRevision);
    else if (nativeTrimDrag)
        model.trimNativeNoteEdge(draggedNote,
            dragMode==DragMode::resizeNativeLeft?previewStartSeconds-dragStartSeconds
                :previewDurationSeconds-dragDurationSeconds,dragMode==DragMode::resizeNativeLeft);
    else if(dragMode==DragMode::resizeNativeLeft || dragMode==DragMode::resizeNativeRight)
        model.resizeNativeNoteEdge(draggedNote,
            dragMode==DragMode::resizeNativeLeft?previewStartSeconds-dragStartSeconds
                :previewDurationSeconds-dragDurationSeconds,dragMode==DragMode::resizeNativeLeft);
    else
        model.resizeNote(draggedNote, previewStartSeconds, previewDurationSeconds);
    draggedNote.clear();
    nativeTrimDrag = false;
    sharedBoundaryNotes.clear();
    finePitchDrag = false;
    draggedPitchAnchor = -1;
    draggedAmplitudePoint = -1;
    pitchStroke.clear();
    amplitudeStroke.clear();
    amplitudeDragUsesLane = false;
    previewMoveDeltaSeconds = 0.0;
    minimumMoveDeltaSeconds = 0.0;
    pointDragCanMoveHorizontally = false;
    pointDragMaximumTime = 0.0;
    pointDragMinimumTime = 0.0;
    pointDragOwnedUntil = std::numeric_limits<double>::infinity();
    dragMode = DragMode::none;
}

bool PianoRollComponent::keyPressed(const juce::KeyPress& key)
{
    if(key.isKeyCode(juce::KeyPress::escapeKey)&&!sharedBoundaryNotes.empty())
    {cancelPointDrag();refreshFromModel();repaint();return true;}
    if (key.isKeyCode(juce::KeyPress::escapeKey) && dragMode == DragMode::nativeLoudness)
    { draggedNote.clear(); dragMode = DragMode::none; repaint(); return true; }
    // Double-click now aligns preutterance; keep ordinary lyric editing
    // reachable without opening the material replacement picker.
    const auto typed=key.getTextCharacter();
    const bool typeAlias=tool==Tool::note&&!isNoteDanceEnabled()&&typed>=0x21&&typed!=0x7f
        &&!key.getModifiers().isCommandDown()&&!key.getModifiers().isCtrlDown()&&!key.getModifiers().isAltDown();
    if(inlineAliasNoteId.isEmpty()&&(key.getKeyCode()==juce::KeyPress::F2Key||typeAlias)&&selectedNote.isNotEmpty())
        for(const auto& hit:noteHits)if(hit.id==selectedNote)
        {
            beginInlineAliasEdit(hit);
            if(typeAlias)inlineAliasEditor.insertTextAtCaret(juce::String::charToString(typed));
            return true;
        }
    if(selectedPoint&&selectedPoint->kind==PointSelection::connectionPitch
        &&(key.getKeyCode()==juce::KeyPress::deleteKey||key.getKeyCode()==juce::KeyPress::backspaceKey))
    {
        if(!selectedPoints.empty())editSelectedPoints(false);
        else applyConnectionPitchPointMenu(selectedPoint->noteId,selectedPoint->index,7);
        return true;
    }

    if (nativeOverlapSelector && key.getKeyCode()==juce::KeyPress::escapeKey) { closeNativeOverlapSelector(); return true; }
    if(key.isKeyCode(juce::KeyPress::escapeKey)&&pointGroupGesture){pointGroupGesture.reset();cancelPointDrag();clearPointSelection();repaint();return true;}
    if(key.getModifiers().isCtrlDown()&&key.getKeyCode()=='R'&&(!selectedPoints.empty()||selectedPoint))
    {if(selectedPoints.empty()&&selectedPoint->index>=0)selectedPoints={*selectedPoint};editSelectedPoints(true);return true;}
    if (key.isKeyCode(juce::KeyPress::escapeKey) && cvvcBoundaryDrag)
    { cvvcBoundaryDrag.reset(); refreshFromModel(); repaint(); return true; }
    if (key.isKeyCode(juce::KeyPress::escapeKey) && cvvcLaneDrag)
    { cvvcLaneDrag.reset(); repaint(); return true; }
    if (key.isKeyCode(juce::KeyPress::escapeKey) && dsPitchOffsetMode)
    { if (!cancelDiffSingerPitchOffsetGesture()) setDiffSingerPitchOffsetMode(false); return true; }
    if (key.isKeyCode(juce::KeyPress::escapeKey) && dsPitchRestoreMode)
    { setDiffSingerPitchRestoreMode(false); return true; }
    if (key.isKeyCode(juce::KeyPress::escapeKey) && dragMode == DragMode::continuousFlag)
    {
        cancelContinuousFlag();
        return true;
    }
    // The lyric box has no use for Tab, so it comes up to here.  Left alone it
    // would go on to the window, which moves focus to the next control and
    // closes the box; taken here, it goes on to the next note instead.
    if (key.isKeyCode(juce::KeyPress::tabKey) && inlineAliasNoteId.isNotEmpty())
    {
        advanceInlineAliasEdit(!key.getModifiers().isShiftDown());
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'A')
    {
        selectAllNotes();
        return true;
    }
    if ((key.getKeyCode() == juce::KeyPress::deleteKey
         || key.getKeyCode() == juce::KeyPress::backspaceKey)
        && (selectedPoint || selectedNote.isNotEmpty() || !selectedNotes.empty()))
    {
        deleteSelectedNotes();
        return true;
    }
    return false;
}
}

#include "PianoRollDiffSingerOffset.h"
#include "PianoRollCvvcEnvelope.h"
#include "PianoRollPointGroup.h"
