// DRAFT: intentionally UNREGISTERED until the parent adds the six declarations
// and entry/close/shutdown hooks described in the handoff. This translation unit
// is Main orchestration only: no new import parser, publisher or model action.
#include "MainComponent.h"
#include "CommunityDatasetImportDialog.h"
#include "OtoWaveformEditorComponent.h"
#include "backend/UtauRenderer.h"
#include <algorithm>
#include <atomic>
#include <map>
#include <set>

namespace hachi
{
struct MainComponent::CommunityDatasetImportState
{
    struct Job final : juce::ThreadPoolJob
    {
        std::function<void()> work;
        std::atomic<bool> completed{false};
        explicit Job(std::function<void()> fn) : juce::ThreadPoolJob("community-dataset-import"), work(std::move(fn)) {}
        JobStatus runJob() override
        {
            // Always run accepted closures once, even after cancellation. The
            // wizard observes its cancellation token and reports the outcome.
            auto fn = std::move(work);
            if (fn) fn();
            completed = true;
            return jobHasFinished;
        }
    };
    juce::ThreadPool worker{1}; // owned by Main's state, NOT a detached/global service
    std::vector<std::unique_ptr<Job>> jobs;
    juce::Component::SafePointer<CommunityDatasetImportDialog> wizard;
    bool stopping = false, publicationGate = false, mainWasEnabled = true;
    std::vector<std::pair<juce::Component::SafePointer<juce::Component>, bool>> gatedEditors;
    std::uint64_t expectedRevision = 0;
    juce::File expectedRoot, expectedManifest;
    juce::MemoryBlock expectedManifestFingerprint; // exact service-observed token, not a second writer
    bool expectedManifestExisted = false;
    std::set<juce::String> targetRegistrationIds;
    std::map<juce::String, juce::String> expectedSourceFingerprints;

    void enqueue(std::function<void()> fn)
    {
        jobs.erase(std::remove_if(jobs.begin(), jobs.end(), [&](const auto& job)
        { return job->completed.load() && !worker.contains(job.get()); }), jobs.end());
        auto job = std::make_unique<Job>(std::move(fn));
        worker.addJob(job.get(), false); // retain pointers until waitForJobToFinish returns
        jobs.push_back(std::move(job));
    }
    void drain()
    {
        // IO closures NEVER wait for the message thread. Do not remove queued
        // jobs without running them: that would strand the wizard's completion.
        for (const auto& job : jobs) worker.waitForJobToFinish(job.get(), -1);
        jobs.clear();
    }
    ~CommunityDatasetImportState() { drain(); }
};

namespace
{
juce::String communityPathKey(const juce::File& file)
{
    auto value = file.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    value = value.toLowerCase();
#endif
    return value;
}
}

bool MainComponent::communityDatasetImportBlocksProjectEdits() const
{ return communityDatasetImportState && communityDatasetImportState->publicationGate; }

// SHUTDOWN ONLY. Ordinary dialog Cancel must use wizard.cancel(), retaining this
// state until its actual completion releases the gate/registers any committed
// root. Main::requestClose must reject/defer close while publicationGate is true.
void MainComponent::cancelCommunityDatasetImport()
{
    const auto state = communityDatasetImportState;
    if (!state) return;
    state->stopping = true;
    if (state->wizard) state->wizard->cancel();
    communityDatasetImportWindows.clear(); // widget destruction signals cancellation
    state->drain(); // before project, preferences, executor or message loop teardown
    const bool wasGated = state->publicationGate;
    if (wasGated)
    {
        // Normal close is gated. Forced process shutdown cannot turn an already
        // atomic disk commit into a rollback: keep the root/PCM and never delete
        // them. No promise of global MCP/process atomicity is made here.
        juce::Logger::writeToLog("Community import shutdown during publication: any committed OTTO is retained at "
                                 + state->expectedRoot.getFullPathName());
        state->publicationGate = false;
    }
    if (!closingUI && wasGated)
    {
        setEnabled(state->mainWasEnabled);
        for (auto& entry : state->gatedEditors) if (entry.first) entry.first->setEnabled(entry.second);
    }
    state->gatedEditors.clear();
    communityDatasetImportState.reset();
}

void MainComponent::showCommunityDatasetImport(const juce::File& initialRoot)
{
    if (closingUI) return;
    if (communityDatasetImportBlocksProjectEdits())
    { statusLabel.setText("Dataset annotation publication is in progress; wait for its actual result.", juce::dontSendNotification); return; }
    if (importInProgress || exportWaitingForRender || projectRegistrationOperation)
    { showError("Finish the current project import/export/material-registration operation before importing a community dataset."); return; }
    constexpr const char* windowKey = "community-dataset-import";
    communityDatasetImportWindows.removeClosed();
    if (communityDatasetImportWindows.showExisting(windowKey)) return;
    if (!assetManager) setAssetManagerVisible(true); // existing manager creation route, BEFORE preflight
    if (!assetManager) { showError("The material manager is unavailable."); return; }
    if (!communityDatasetImportState) communityDatasetImportState = std::make_shared<CommunityDatasetImportState>();
    const auto state = communityDatasetImportState;
    juce::Component::SafePointer<MainComponent> safe(this);
    CommunityDatasetImportDialog::Host host;
    host.runOnWorker = [state](std::function<void()> work) { state->enqueue(std::move(work)); };
    host.publishing = [safe,state](bool acquiring)
    {
        if (!acquiring)
        {
            if (!state->publicationGate) return;
            state->publicationGate = false;
            if (safe && !safe->closingUI && safe->communityDatasetImportState == state)
            {
                safe->setEnabled(state->mainWasEnabled);
                for (auto& entry : state->gatedEditors) if (entry.first) entry.first->setEnabled(entry.second);
            }
            state->gatedEditors.clear();
            return;
        }
        if (!safe || safe->closingUI || state->stopping || safe->communityDatasetImportState != state || state->publicationGate) return;
        state->publicationGate = true;
        state->mainWasEnabled = safe->isEnabled();
        state->gatedEditors.clear();
        // These are Main's already registered quick/precise/local OTO editors,
        // not arbitrary top-level windows from another project/Main instance.
        for (const auto& editor : safe->otoPreviews)
            if (editor)
            {
                state->gatedEditors.push_back({juce::Component::SafePointer<juce::Component>(editor.getComponent()), editor->isEnabled()});
                editor->setEnabled(false);
            }
        safe->audio.stop();
        safe->setEnabled(false); // main widgets, keys and drag/drop are gated
        safe->statusLabel.setText("Publishing dataset annotations; normal project editing is temporarily gated.", juce::dontSendNotification);
    };
    host.preflight = [safe,state](const CommunityDatasetIO::Plan& plan, juce::String& error)
    {
        error.clear();
        const auto reject = [&](const juce::String& message) { error = message; return false; };
        if (!safe || safe->closingUI || state->stopping || safe->communityDatasetImportState != state || !state->publicationGate)
            return reject("The Main publication gate is unavailable.");
        if (safe->importInProgress || safe->exportWaitingForRender || safe->projectRegistrationOperation || safe->pendingNativeAnalyses > 0)
            return reject("Finish pending project imports, material registration, source analysis or export before publishing this dataset.");
        if (!safe->assetManager || !plan.ready || plan.published || plan.audio.empty() || plan.manifest != plan.dataset.root.getChildFile("otto.ini"))
            return reject("The current preview/manager/root publication target is unavailable; preview again.");
        std::set<juce::String> registrations, sourcePaths;
        for (const auto& audio : plan.audio)
        {
            if (audio.audioId.isEmpty() || !audio.audio.isAChildOf(plan.dataset.root) || !registrations.insert(audio.audioId).second)
                return reject("Dataset preview has ambiguous annotation registration identities.");
            sourcePaths.insert(communityPathKey(audio.audio));
            for (const auto& row : audio.rows) if (row.materialAudioId != audio.audioId)
                return reject("Preview rows disagree with their explicit annotation registration ID.");
        }
        const auto revision = safe->project.revisionNumber();
        const auto project = safe->project.snapshot(); // one authoritative preflight snapshot only
        if (safe->project.revisionNumber() != revision) return reject("The project changed during dataset preflight; retry for the current project.");
        for (const auto& track : project.tracks)
        {
            // A voicebank-backed project may resolve sources without carrying
            // native variant IDs. Do not claim it is independent of this root.
            if (trackUsesVoicebankSynthesis(track) && track.voicebankDirectory != juce::File{}
                && (communityPathKey(track.voicebankDirectory) == communityPathKey(plan.dataset.root)
                    || track.voicebankDirectory.isAChildOf(plan.dataset.root) || plan.dataset.root.isAChildOf(track.voicebankDirectory)))
                return reject("This dataset root is used by a current voicebank track. Close that project or choose an independent dataset/root; live source updates are not supported by this import action.");
            for (const auto& clip : track.clips)
            {
                for (const auto& note : clip.notes)
                    if (registrations.count(note.materialAudioId))
                        return reject("Dataset annotation registration is used by current note " + note.id
                            + ". Close that project or choose an independent dataset/root. This conservative import action does not rewrite live note annotations.");
                const auto uncertainOwner = [&](const ClipData& owner, const juce::String& partId)
                {
                    if (!sourcePaths.count(communityPathKey(owner.sourceFile))) return false;
                    return std::any_of(clip.notes.begin(), clip.notes.end(), [&](const auto& note)
                    { return note.clipPartId == partId && note.materialAudioId.isEmpty(); });
                };
                if (clip.parts.empty())
                { if (uncertainOwner(clip, {})) return reject("A current carried note uses this recording without a registration ID; its independence cannot be established. Close the project or select an independent dataset/root."); }
                else for (const auto& part : clip.parts)
                    if (uncertainOwner(part, part.id)) return reject("A current carried part uses this recording without a registration ID; close that project or select an independent dataset/root.");
            }
        }
        // This import is NEW/unreferenced variants only. Foreign, explicit
        // same-file registrations remain untouched; NEVER call physical-path
        // applySourceSettingsWithReport with only these newly imported rows.
        state->expectedRevision = revision;
        state->expectedRoot = plan.dataset.root; state->expectedManifest = plan.manifest;
        state->expectedManifestFingerprint = plan.document.fingerprint;
        state->expectedManifestExisted = plan.document.existed;
        state->targetRegistrationIds = std::move(registrations);
        state->expectedSourceFingerprints.clear();
        for (const auto& recording : plan.dataset.recordings)
            state->expectedSourceFingerprints.emplace(communityPathKey(recording.audio), recording.audioFingerprint);
        return true; // service still owns source hashes + cooperative manifest token checks
    };
    host.published = [safe,state](const CommunityDatasetIO::Plan& plan, const CommunityDatasetIO::Report& report)
    {
        if (!report.published || !plan.published) return;
        if (!safe || safe->closingUI || state->stopping || safe->communityDatasetImportState != state)
        {
            juce::Logger::writeToLog("Community dataset OTTO committed and retained; Main registration could not complete: " + plan.dataset.root.getFullPathName());
            return; // never delete a successfully committed source root
        }
        const bool revisionDrift = safe->project.revisionNumber() != state->expectedRevision;
        juce::String error;
        // Use the REAL manager authority directly, never a diagnostic root-writer
        // hook. Reuse the existing retained-root retry list/menu, not a parallel
        // preference writer. No project save/adoption/model transaction or media copy.
        const bool registered = safe->assetManager && safe->assetManager->registerMaterialRoot(plan.dataset.root, error);
        if (!safe->assetManager) error = "Material manager became unavailable; retry registration of the retained root";
        backend::UtauRenderer::invalidateVoicebankCache();
        if (!registered)
        {
            if (std::find(safe->projectRegistrationRetryRoots.begin(), safe->projectRegistrationRetryRoots.end(), plan.dataset.root) == safe->projectRegistrationRetryRoots.end())
                safe->projectRegistrationRetryRoots.push_back(plan.dataset.root);
            safe->showError("Dataset OTTO was committed and is retained at:\n" + plan.dataset.root.getFullPathName()
                + "\nMaterial-library registration/persistence failed: " + error
                + "\nUse File > Retry material folder registration. No source data was deleted or rolled back; current note annotations were not changed.");
        }
        else safe->statusLabel.setText("Community dataset annotations imported: " + juce::String(report.recordings)
            + " recordings, " + juce::String(report.pronunciations) + " pronunciations. Current notes unchanged.", juce::dontSendNotification);
        if (revisionDrift)
            safe->showError("Dataset OTTO was committed, but the project revision changed while worker publication was running. Current notes were NOT rewritten or rebound. This UI gate is not a global MCP/model transaction; inspect current references before further source editing.");
        safe->menuItemsChanged();
    };
    auto* dialog = new CommunityDatasetImportDialog(strings, std::move(host), initialRoot);
    state->wizard = dialog;
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "Import community DiffSinger dataset annotations";
    options.dialogBackgroundColour = Palette::panel; options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true; options.resizable = true; options.content.setOwned(dialog);
    if (auto* window = communityDatasetImportWindows.show(options, windowKey)) window->setResizeLimits(1000, 820, 1800, 1400);
}
}
