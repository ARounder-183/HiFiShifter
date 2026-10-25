#pragma once
#include "I18n.h"
#include "PronunciationTiming.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>
#include <vector>

namespace hachi
{
struct MaterialPronunciationCandidate
{
    juce::File audio, manifest;
    juce::String audioId, regionId, alias;
    double startSeconds = 0, endSeconds = 0;
    PronunciationKind pronunciationKind = PronunciationKind::normal;
};

// A source-disambiguating native-material picker. Owns no audio device and
// never silently accepts a matching alias, even when only one result remains.
class PronunciationCandidateBox final : public juce::Component,
                                        private juce::ListBoxModel,
                                        private juce::KeyListener
{
public:
    using Candidate = MaterialPronunciationCandidate;
    PronunciationCandidateBox(I18n&, std::vector<Candidate>,
        std::function<void(const Candidate&)> onChosen, std::function<void()> onCancelled);
    ~PronunciationCandidateBox() override;
    void setQuery(juce::String);
    bool keyPressed(const juce::KeyPress&) override;
    void paint(juce::Graphics&) override;
    void resized() override;
    std::function<void(const Candidate&)> onPreview;
    juce::TextEditor& queryEditor() { return query; }
    juce::TextEditor& diagnosticQueryEditor() { return query; }
    int diagnosticCandidateCount() const { return static_cast<int>(visible.size()); }
    juce::String diagnosticSelectedId() const;
    void diagnosticCommit() { choose(list.getSelectedRow()); }
    void diagnosticSelect(int row) { list.selectRow(row); }
private:
    int getNumRows() override;
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override;
    void listBoxItemDoubleClicked(int, const juce::MouseEvent&) override;
    juce::String getTooltipForRow(int) override;
    bool keyPressed(const juce::KeyPress&, juce::Component*) override;
    void filter();
    void choose(int);
    void cancel();
    void preview();
    juce::String tr(const char* zh, const char* en) const;
    juce::String candidateLabel(const Candidate&) const;
    I18n& strings;
    std::vector<Candidate> candidates;
    std::vector<std::size_t> visible;
    std::function<void(const Candidate&)> chosen;
    std::function<void()> cancelled;
    juce::TextEditor query;
    juce::ListBox list;
    juce::Label help, empty;
    juce::TextButton chooseButton, previewButton, cancelButton;
    bool finishing = false;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PronunciationCandidateBox)
};
}
