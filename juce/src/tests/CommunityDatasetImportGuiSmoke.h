#pragma once
#include "../CommunityDatasetImportDialog.h"
#include "../Theme.h"
#include <juce_cryptography/juce_cryptography.h>
#include <juce_graphics/juce_graphics.h>
#include <atomic>
#include <functional>
#include <iostream>

namespace hachi
{
// Actual visible wizard + caller-owned worker + real public CSV/WAV IO.
// Host gate/model callbacks below are explicitly MOCK CONTRACTS, not proof of
// real Main/AssetManager/project propagation. Service/render acceptance has its
// separate CommunityDatasetSmoke. Invoke asynchronously on the message thread.
inline void runCommunityDatasetImportGuiSmoke(const juce::File& output,
    const juce::File& publicFixture, std::function<void(bool)> finished)
{
    struct Run final : juce::Timer, std::enable_shared_from_this<Run>
    {
        struct Task final : juce::ThreadPoolJob
        {
            std::function<void()> work;
            explicit Task(std::function<void()> f) : juce::ThreadPoolJob("community-import-smoke-worker"), work(std::move(f)) {}
            JobStatus runJob() override { work(); return jobHasFinished; }
        };
        struct Window final : juce::DocumentWindow
        {
            explicit Window(juce::Component* content) : juce::DocumentWindow("Community dataset import wizard — GUI contract test", Palette::panel, juce::DocumentWindow::closeButton)
            {
                setUsingNativeTitleBar(true); setContentOwned(content, true); setResizable(true, true);
                centreWithSize(1040, 880); setVisible(true); toFront(true);
            }
            void closeButtonPressed() override { setVisible(false); }
        };
        std::shared_ptr<Run> self;
        juce::ThreadPool pool{1};
        std::unique_ptr<Window> window;
        I18n strings;
        juce::File base, fixture, actualRoot, ambiguousRoot, mapping;
        CommunityDatasetImportDialog::Input actualInput;
        std::function<void(bool)> done;
        std::function<void()> heldJob;
        juce::String log, mappingJson, sourceHash, csvHash;
        bool ok = true, gate = false, rejectPreflight = false, holdNextJob = false, deleteOnPublished = false;
        int stage = 0, checks = 0, workers = 0, preflights = 0, publications = 0, gateAcquires = 0, gateReleases = 0;
        double deadline = 0;
        CommunityDatasetImportDialog* ui() const { return window ? dynamic_cast<CommunityDatasetImportDialog*>(window->getContentComponent()) : nullptr; }
        void check(const char* name, bool value)
        { ++checks; ok &= value; log += juce::String(name) + "=" + (value ? "PASS" : "FAIL") + "\n"; }
        bool capture(const char* filename)
        {
            if (!window || !window->isShowing()) return false;
            auto output = base.getChildFile(filename).createOutputStream();
            juce::PNGImageFormat format;
            return output && format.writeImageToStream(window->createComponentSnapshot(window->getLocalBounds(), true, 1), *output);
        }
        void finish()
        {
            auto keep = shared_from_this(); stopTimer();
            if (ui()) ui()->cancel();
            window.reset();
            if (heldJob) { pool.addJob(new Task(std::move(heldJob)), true); heldJob = {}; }
            pool.removeAllJobs(true, 5000);
            log += "SCOPE=actual wizard/worker/public dataset IO with MOCK host publication gate and callbacks; Main registration/project propagation NOT tested here\n";
            const bool written = base.getChildFile("verification.txt").replaceWithText(log);
            std::cout << "community_import_gui=" << ok << "|checks=" << checks << "|fixture=" << base.getFullPathName() << std::endl;
            if (!ok) std::cerr << log << std::endl;
            const auto callback = std::move(done); self.reset();
            if (callback) callback(ok && written);
        }
        bool setup(const juce::File& output, const juce::File& source, std::function<void(bool)> callback)
        {
            base = output.getChildFile("community-import-gui-" + juce::Uuid().toString()); fixture = source; done = std::move(callback);
            if (base.createDirectory().failed()) return false;
            strings.setLanguage(I18n::Language::enUS);
            const auto originalAudio = fixture.getChildFile("wavs/req-3943.wav"), originalCsv = fixture.getChildFile("transcriptions.csv");
            sourceHash = juce::SHA256(originalAudio).toHexString(); csvHash = juce::SHA256(originalCsv).toHexString();
            check("public_SYNTHETIC_fixture_hash", sourceHash == "1db5d8dead9be7c664b3b949cebf8bb7aeb0e6c9b0ea997e051b8487d5fd5175");
            actualRoot = base.getChildFile("public-import"); actualRoot.getChildFile("wavs").createDirectory();
            ambiguousRoot = base.getChildFile("ambiguous"); ambiguousRoot.getChildFile("wavs").createDirectory();
            check("isolated_actual_public_data", originalAudio.copyFileTo(actualRoot.getChildFile("wavs/req-3943.wav"))
                && originalCsv.copyFileTo(actualRoot.getChildFile("transcriptions.csv"))
                && originalAudio.copyFileTo(ambiguousRoot.getChildFile("wavs/one.wav")));
            check("ambiguous_alias_fixture", ambiguousRoot.getChildFile("transcriptions.csv").replaceWithText("name,ph_seq,ph_dur\none,k a,0.1 0.4\n", false, false, nullptr)
                && ambiguousRoot.getChildFile("dictionary.txt").replaceWithText("ka\tk a\nka_alt\tk a\n", false, false, nullptr));
            mappingJson = R"({"version":1,"selections":[{"recording":"req-3943","alias":"y_uw","firstPhone":1,"phoneCount":2,"nucleusOffset":1},{"recording":"req-3943","alias":"s_p_ih_n","firstPhone":3,"phoneCount":4,"nucleusOffset":2},{"recording":"req-3943","alias":"dh_ax","firstPhone":7,"phoneCount":2,"nucleusOffset":1},{"recording":"req-3943","alias":"r_iy_l","firstPhone":9,"phoneCount":3,"nucleusOffset":1},{"recording":"req-3943","alias":"y_uw","firstPhone":12,"phoneCount":2,"nucleusOffset":1},{"recording":"req-3943","alias":"p_ah_l_s","firstPhone":14,"phoneCount":4,"nucleusOffset":1},{"recording":"req-3943","alias":"dh_ax","firstPhone":18,"phoneCount":2,"nucleusOffset":1},{"recording":"req-3943","alias":"f_iy_l","firstPhone":20,"phoneCount":3,"nucleusOffset":1}],"primaryAliases":{}})";
            mapping = base.getChildFile("explicit-import-choices.json");
            check("explicit_choice_sidecar_written", mapping.replaceWithText(mappingJson, false, false, nullptr));
            actualInput.root = actualRoot; actualInput.mappingFile = mapping;
            actualInput.options.confirmNucleusAsAlignmentAndConsonant = true;
            actualInput.options.specials = CommunityDatasetIO::Specials::importDeclaredSPAP;
            actualInput.provenanceNotes = "SYNTHETIC public SynthGT req-3943; CC BY-NC-SA 4.0; Antonisen & Lopez-Espejo. Metadata dh_ax explicitly maps CSV dh ah; no implicit spelling normalization, human-recording or Tokyo/teio claim.";
            log += actualInput.provenanceNotes + "\nhttps://huggingface.co/datasets/Silasimo/SynthGT/raw/main/README.md\nhttps://huggingface.co/datasets/Silasimo/SynthGT/raw/main/metadata/syllable.jsonl\n";
            auto weak = std::weak_ptr<Run>(shared_from_this());
            CommunityDatasetImportDialog::Host host;
            host.runOnWorker = [weak](std::function<void()> task)
            {
                if (auto run = weak.lock())
                {
                    ++run->workers;
                    if (run->holdNextJob) { run->holdNextJob = false; run->heldJob = std::move(task); }
                    else run->pool.addJob(new Task(std::move(task)), true);
                }
            };
            host.publishing = [weak](bool acquiring)
            {
                if (auto run = weak.lock())
                {
                    run->check("gate_callback_is_message_thread", juce::MessageManager::getInstance()->isThisTheMessageThread());
                    if (acquiring) { ++run->gateAcquires; run->check("no_nested_mock_publication_gate", !run->gate); run->gate = true; }
                    else { ++run->gateReleases; run->check("release_owned_mock_gate", run->gate); run->gate = false; }
                }
            };
            host.preflight = [weak](const CommunityDatasetIO::Plan& plan, juce::String& error)
            {
                if (auto run = weak.lock())
                {
                    ++run->preflights;
                    run->check("mock_preflight_has_gate_and_explicit_variant", run->gate && plan.audio.size() == 1
                        && !plan.audio.front().audioId.isEmpty() && plan.document.audio.size() == 1
                        && juce::MessageManager::getInstance()->isThisTheMessageThread());
                    if (run->rejectPreflight) { error = "mock revision gate rejects preflight"; return false; }
                    return true;
                }
                error = "mock host closed"; return false;
            };
            host.published = [weak](const CommunityDatasetIO::Plan& plan, const CommunityDatasetIO::Report& result)
            {
                if (auto run = weak.lock())
                {
                    ++run->publications;
                    run->check("actual_success_callback_under_mock_gate", run->gate && plan.published && result.published
                        && result.pronunciations == 10 && juce::MessageManager::getInstance()->isThisTheMessageThread());
                    run->check("actual_commit_window_capture", run->capture("committed-callback.png"));
                    if (run->deleteOnPublished) run->window.reset();
                }
            };
            window = std::make_unique<Window>(new CommunityDatasetImportDialog(strings, std::move(host), actualRoot));
            check("actual_wizard_window_showing", window->isShowing() && ui() && ui()->isShowing());
            ui()->preview();
            check("defaults_require_explicit_timing_and_SPAP", !ui()->busy() && !ui()->previewReady() && workers == 0 && !actualRoot.getChildFile("otto.ini").exists());
            CommunityDatasetImportDialog::Input ambiguous;
            ambiguous.root = ambiguousRoot; ambiguous.dictionary = ambiguousRoot.getChildFile("dictionary.txt");
            ambiguous.options.confirmNucleusAsAlignmentAndConsonant = true; ambiguous.options.specials = CommunityDatasetIO::Specials::skipWithReport;
            juce::String error;
            check("set_ambiguous_dictionary_input", ui()->setInput(ambiguous, error)); ui()->preview(); stage = 1;
            deadline = juce::Time::getMillisecondCounterHiRes() + 120000; startTimer(20);
            return true;
        }
        void timerCallback() override
        {
            auto keep = shared_from_this();
            if (juce::Time::getMillisecondCounterHiRes() > deadline)
            { check("bounded_GUI_worker_completion", false); if (ui()) log += ui()->reportText() + "\n"; finish(); return; }
            if (stage == 11)
            {
                if (publications != 1 || gate)
                {
                    if (ui() && !ui()->busy() && !gate)
                    { check("actual_publication_result", false); log += ui()->reportText() + "\n"; finish(); }
                    return;
                }
                check("widget_deleted_by_publication_callback_gate_released", !window && gateAcquires == gateReleases);
                check("actual_root_OTTO_written", actualRoot.getChildFile("otto.ini").existsAsFile());
                OttoMaterialIO::Document document; juce::String error;
                check("actual_disk_registration_and_rows", OttoMaterialIO::loadDocument(actualRoot.getChildFile("otto.ini"), document, error)
                    && document.audio.size() == 1 && document.audio.front().rows.size() == 10);
                check("external_CSV_WAV_unchanged", juce::SHA256(fixture.getChildFile("transcriptions.csv")).toHexString() == csvHash
                    && juce::SHA256(fixture.getChildFile("wavs/req-3943.wav")).toHexString() == sourceHash
                    && juce::SHA256(actualRoot.getChildFile("wavs/req-3943.wav")).toHexString() == sourceHash);
                finish(); return;
            }
            if (!ui()) { check("wizard_lifetime_before_final_commit", false); finish(); return; }
            if (ui()->busy()) return;
            juce::String error;
            switch (stage)
            {
                case 1:
                {
                    check("ambiguous_alias_has_no_default_or_write", !ui()->previewReady()
                        && ui()->reportText().contains("Multiple dictionary aliases") && ui()->reportText().contains("ka | ka_alt")
                        && !ambiguousRoot.getChildFile("otto.ini").exists());
                    CommunityDatasetImportDialog::Input input;
                    input.root = ambiguousRoot; input.dictionary = ambiguousRoot.getChildFile("dictionary.txt");
                    input.mappingJson = R"({"version":1,"primaryAliases":{"k a":"ka_alt"}})";
                    input.options.confirmNucleusAsAlignmentAndConsonant = true; input.options.specials = CommunityDatasetIO::Specials::skipWithReport;
                    check("visible_primary_alias_choice", ui()->setInput(input, error)); ui()->preview(); stage = 2; break;
                }
                case 2:
                {
                    const auto* plan = ui()->previewPlan();
                    check("explicit_alias_choice_is_preserved", plan && plan->audio.front().rows.front().name == "ka_alt"
                        && plan->audio.front().rows.front().additionalAliases.contains("ka"));
                    auto input = actualInput; input.mappingFile = juce::File{}; input.mappingJson = R"({"version":1,"invented_csv_field":true})";
                    ui()->setInput(input, error); ui()->preview(); stage = 3; break;
                }
                case 3:
                {
                    check("choice_sidecar_unknown_key_rejected", !ui()->previewReady() && ui()->reportText().contains("Unknown import-choice key"));
                    auto input = actualInput; input.mappingFile = juce::File{};
                    input.mappingJson = R"({"version":1,"selections":[{"recording":"req-3943","alias":"wrong","firstPhone":0,"phoneCount":0,"nucleusOffset":0}]})";
                    ui()->setInput(input, error); ui()->preview(); stage = 4; break;
                }
                case 4:
                    check("invalid_choice_cardinality_rejected", !ui()->previewReady() && ui()->reportText().contains("phoneCount"));
                    ui()->setInput(actualInput, error); ui()->preview(); stage = 5; break;
                case 5:
                {
                    check("actual_public_preview_ready_no_write", ui()->previewReady() && !actualRoot.getChildFile("otto.ini").exists());
                    check("actual_preview_window_capture", capture("preview-ready.png"));
                    auto changed = actualInput; changed.provenanceNotes += " [input changed]";
                    ui()->setInput(changed, error);
                    check("changed_input_invalidates_publish", !ui()->previewReady() && !ui()->publishConfirmed(error)
                        && !actualRoot.getChildFile("otto.ini").exists());
                    ui()->preview(); stage = 6; break;
                }
                case 6:
                    check("second_preview_ready", ui()->previewReady());
                    rejectPreflight = true;
                    check("host_preflight_rejection_releases_gate", !ui()->publishConfirmed(error) && !gate && gateAcquires == gateReleases);
                    rejectPreflight = false;
                    mapping.appendText("\n", false, false, nullptr);
                    check("mapping_stale_publication_started", ui()->publishConfirmed(error)); stage = 7; break;
                case 7:
                    check("stale_mapping_fingerprint_stops_commit", ui()->reportText().contains("JSON file changed")
                        && publications == 0 && !gate && gateAcquires == gateReleases && !actualRoot.getChildFile("otto.ini").exists());
                    mapping.replaceWithText(mappingJson, false, false, nullptr);
                    ui()->setInput(actualInput, error); ui()->preview(); stage = 8; break;
                case 8:
                    check("preview_before_cancel_ready", ui()->previewReady()); holdNextJob = true;
                    check("cancellable_publication_queued", ui()->publishConfirmed(error) && gate && static_cast<bool>(heldJob));
                    ui()->cancel();
                    if (heldJob) { pool.addJob(new Task(std::move(heldJob)), true); heldJob = {}; }
                    stage = 9; break;
                case 9:
                    check("cancel_before_commit_not_success", publications == 0 && !gate && gateAcquires == gateReleases
                        && !actualRoot.getChildFile("otto.ini").exists() && ui()->reportText().contains("Cancelled before publication"));
                    ui()->preview(); stage = 10; break;
                case 10:
                    check("final_actual_preview_ready", ui()->previewReady()); deleteOnPublished = true;
                    check("actual_publication_started", ui()->publishConfirmed(error)); stage = 11; break;
                default: check("unexpected_GUI_state", false); finish(); break;
            }
        }
    };
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread()) { if (finished) finished(false); return; }
    auto run = std::make_shared<Run>(); run->self = run;
    if (!run->setup(output, publicFixture, std::move(finished))) { run->check("GUI_fixture_setup", false); run->finish(); }
}
}
