#pragma once
#include "CommunityDatasetIO.h"
#include "I18n.h"
#include <juce_gui_extra/juce_gui_extra.h>
#include <memory>

namespace hachi
{
// Import-choice wizard, NOT a third annotation editor. Business parsing,
// grouping, timing and publication remain exclusively CommunityDatasetIO.
class CommunityDatasetImportDialog final : public juce::Component
{
public:
    struct Host
    {
        // REQUIRED caller-owned executor: jobs run off the message thread and
        // accepted jobs must execute once (even to observe cancellation), and
        // are drained before application/message-loop shutdown. The
        // dialog creates no detached thread, private global service or GUI loop.
        std::function<void(std::function<void()>)> runOnWorker;
        // Message-thread callbacks, all required, nonthrowing. publishing(true)
        // acquires the host's edit/revision gate BEFORE preflight; false releases
        // after actual completion, including if this widget has been destroyed.
        // Host callbacks must guard their OWN lifetime (e.g. SafePointer<Main>).
        std::function<void(bool)> publishing;
        // Full candidate document + explicit variant IDs: never propagate only
        // new-variant rows onto unrelated same-file project registrations.
        std::function<bool(const CommunityDatasetIO::Plan&, juce::String&)> preflight;
        // Actual successful disk commit only. Register via the existing manager,
        // refresh/invalidate and apply the existing guarded model transaction.
        // This wizard performs NO model mutation, registration or PCM copy.
        std::function<void(const CommunityDatasetIO::Plan&, const CommunityDatasetIO::Report&)> published;
    };
    struct Input
    {
        juce::File root, dictionary, mappingFile;
        juce::String mappingJson; // explicit import-choice sidecar text, not CSV metadata
        juce::String provenanceNotes; // user-supplied label, never inferred human/synthetic status
        // primaryAliases is entered in visible mappingJson/mappingFile only;
        // a nonempty hidden Options::primaryAliases map is rejected.
        CommunityDatasetIO::Options options;
    };
    CommunityDatasetImportDialog(I18n&, Host, const juce::File& initialRoot = {});
    ~CommunityDatasetImportDialog() override;
    void resized() override;
    void paint(juce::Graphics&) override;
    void cancel();
    bool busy() const;
    bool publicationInFlight() const;
    // Widget/compiled-window seams use exactly the same event handlers. Input
    // edits invalidate preview; edits during publication reject explicitly.
    bool setInput(const Input&, juce::String& error);
    void preview();
    bool publishConfirmed(juce::String& error); // caller has explicitly confirmed
    bool previewReady() const;
    juce::String reportText() const;
    const CommunityDatasetIO::Plan* previewPlan() const; // message-thread, invalidated by edits
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
