#pragma once
#include "SampleSettings.h"
#include "I18n.h"
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <functional>
#include <memory>

namespace hachi
{
class MaterialAnnotationSession;
// Internal overview layer of OtoWaveformEditorComponent, never a separate
// editor entry point. Shares its owner's exact native draft and owns no device.
class MaterialAnnotationEditor final : public juce::Component,
                                        private juce::ListBoxModel,
                                        private juce::ChangeListener
{
public:
    ~MaterialAnnotationEditor() override;
    void setPublicationValidator(std::function<bool(
        const std::vector<SampleRegionSetting>&, juce::String&)> validator);
    void paint(juce::Graphics&) override;
    void resized() override;
    bool keyPressed(const juce::KeyPress&) override;
    [[nodiscard]] juce::String errorText() const { return status.getText(); }
private:
    friend class OtoWaveformEditorComponent;
    // Internal overview only. The owning OTO editor supplies the one shared draft.
    MaterialAnnotationEditor(std::shared_ptr<MaterialAnnotationSession>, I18n&,
        std::function<void(const std::vector<SampleRegionSetting>&)> onSaved);
    std::function<void(const juce::String&)> onEditPronunciation;
    juce::Component& overviewWaveform();
    void openDetail(const juce::String& id);
    void createRangeAlias();
    void cancelAlias();
    void listBoxItemDoubleClicked(int, const juce::MouseEvent&) override;
    class Waveform;
    int getNumRows() override;
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override;
    void selectedRowsChanged(int) override;
    void changeListenerCallback(juce::ChangeBroadcaster*) override;
    juce::String tr(const char* zh, const char* en) const;
    std::vector<juce::String> selectedIds() const;
    const SampleRegionSetting* activeRow() const;
    void refresh(const std::vector<juce::String>& select = {});
    void showFields();
    bool commitFields();
    void showError(const juce::String&);
    void splitAtText();
    void splitAt(double);
    void selectAt(double);
    void selectSourceRange(juce::Range<double>);
    void save();
    void history(bool undo);

    I18n& strings;
    std::function<void(const std::vector<SampleRegionSetting>&)> onSaved;
    std::shared_ptr<MaterialAnnotationSession> session;
    bool rangeSelected = false, selectedRangeBefore = false;
    juce::String selectedRangeAnchor;
    double selectedRangeStart = 0, selectedRangeEnd = 0;
    juce::AudioFormatManager formats;
    juce::AudioThumbnailCache thumbnailCache { 4 };
    juce::AudioThumbnail thumbnail { 512, formats, thumbnailCache };
    std::unique_ptr<Waveform> waveform;
    juce::ListBox list;
    juce::Label title, status, aliasLabel, extraLabel, cutLabel, batchLabel, segmentLabel, pronunciationKindLabel;
    juce::TextEditor alias, extraAliases, cuts, batchAliases, segmentEditor;
    juce::String segmentBaseline;
    std::array<juce::Label, 5> timingLabels;
    std::array<juce::TextEditor, 5> timings;
    juce::ComboBox pronunciationKindChoice;
    juce::ToggleButton adjacentPronunciation;
    juce::TextButton applyButton, splitButton, renameButton, mergeButton, deleteButton;
    juce::TextButton undoButton, redoButton, saveButton, createAliasButton, cancelAliasButton, detailButton;
    juce::String activeId;
    bool refreshing = false, fieldsDirty = false;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MaterialAnnotationEditor)
};
}
