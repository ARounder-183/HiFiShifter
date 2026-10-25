#pragma once
#include "../AssetManagerComponent.h"
#include "../MaterialAnnotationSession.h"
#include "../OtoWaveformEditorComponent.h"
#include "../OttoMaterialIO.h"
#include "../Theme.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>

namespace hachi
{
// Parent wires this compiled GUI command. Captures must be opened/inspected;
// their mere existence is not human or visual acceptance.
inline bool runVariantAssetUiSmoke(const juce::File& output)
{
    using IO = OttoMaterialIO;
    using Session = MaterialAnnotationSession;
    const auto base = output.getChildFile("variant-ui-" + juce::Uuid().toString());
    const auto bank = base.getChildFile("bank");
    if (bank.createDirectory().failed()) return false;
    const auto audio = bank.getChildFile("same.wav"), manifest = bank.getChildFile("otto.ini");
    const auto makeAudio = [](const juce::File& file)
    {
        juce::AudioBuffer<float> pcm(1, 48000); pcm.clear();
        auto stream = file.createOutputStream(); if (!stream) return false;
        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> writer(wav.createWriterFor(stream.release(), 48000, 1, 16, {}, 0));
        return writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples());
    };
    const auto bytes = [](const juce::File& file)
    { juce::MemoryBlock result; file.loadFileAsData(result); return result; };
    bool ok = true; juce::String error, report;
    const auto check = [&](const char* label, bool pass)
    { ok &= pass; report += juce::String(label) + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << label << '=' << pass << std::endl; };
    check("one_physical_wav_fixture", makeAudio(audio));
    const auto pcmBefore = bytes(audio);
    SampleRegionSetting a;
    a.materialAudioId = "annotation_A"; a.materialRegionId = "p1"; a.name = "-";
    a.regionStartSeconds = .1; a.regionEndSeconds = .4; a.alignmentSeconds = .2; a.fixedDurationSeconds = .15;
    a.connectedNextRegionId = "p2";
    auto a2 = a; a2.materialRegionId = "p2"; a2.regionStartSeconds = .4; a2.regionEndSeconds = .65;
    a2.alignmentSeconds = .45; a2.fixedDurationSeconds = .1; a2.connectedNextRegionId.clear();
    auto b = a; b.materialAudioId = "annotation_B"; b.regionStartSeconds = .25; b.regionEndSeconds = .9;
    b.alignmentSeconds = .3; b.connectedNextRegionId.clear();
    IO::Document document;
    check("two_registrations_one_path", IO::replaceAudio(document, manifest, audio, {a,a2}, error)
        && IO::replaceAudio(document, manifest, audio, {b}, error) && IO::saveDocument(manifest, document, error));
    const auto originalManifest = bytes(manifest);
    juce::PropertiesFile::Options options; options.applicationName = "variant-ui-smoke";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    juce::PropertiesFile properties(base.getChildFile("settings.xml"), options);
    I18n strings; HachiLookAndFeel look;
    AssetManagerComponent manager(strings, properties);
    manager.setLookAndFeel(&look); manager.setSize(440, 620);
    check("register_root", manager.diagnosticRegisterFolder(bank) >= 0);
    manager.diagnosticToggleAudio(audio, "annotation_A");
    manager.diagnosticToggleAudio(audio, "annotation_B");
    check("both_registration_ids_visible", manager.diagnosticRegistrationIds(audio)
        == juce::StringArray{"annotation_A", "annotation_B"});
    const auto tree = manager.diagnosticTreeSnapshot();
    int registrationNodes = 0, pronunciationNodes = 0; bool correctDepth = true, labelled = true;
    if (const auto* rows = tree.getArray()) for (const auto& row : *rows)
    {
        const int depth = static_cast<int>(row["depth"]);
        correctDepth &= depth >= 0 && depth <= 2;
        if (depth == 1)
        { ++registrationNodes; labelled &= row["label"].toString().contains(row["audioVariantId"].toString()); }
        if (depth == 2 && row["regionId"].toString().isNotEmpty()) ++pronunciationNodes;
    }
    check("exactly_three_levels_two_audio_nodes", correctDepth && registrationNodes == 2 && pronunciationNodes == 3);
    check("duplicate_filename_captions_include_stable_ids", labelled);
    check("ambiguous_path_diagnostic_no_first_match", !manager.diagnosticSelectPronunciation(audio, "p1")
        && manager.diagnosticPronunciationLabels(audio).isEmpty() && manager.audioDragDescription(audio).isVoid());
    check("unknown_explicit_id_reject", !manager.diagnosticSelectRegistration(audio, "missing")
        && manager.audioDragDescription(audio, "missing").isVoid());
    check("select_A_then_contiguous_rows", manager.diagnosticSelectPronunciation(audio, "p1", false, false, "annotation_A")
        && manager.diagnosticSelectPronunciation(audio, "p2", false, true, "annotation_A")
        && manager.diagnosticSelectedRegionIds() == juce::StringArray{"p1", "p2"});
    check("same_region_id_select_B_does_not_toggle_A", manager.diagnosticSelectPronunciation(audio, "p1", true, false, "annotation_B")
        && manager.diagnosticSelectedVariantId() == "annotation_B"
        && manager.diagnosticSelectedRegionIds() == juce::StringArray{"p1"});
    const auto payload = manager.pronunciationDragDescription(audio, {"p1"}, "annotation_B");
    check("drag_preserves_exact_variant_and_region", payload["audioVariantId"].toString() == "annotation_B"
        && payload["file"].toString() == audio.getFullPathName()
        && payload["regionIds"].getArray() && payload["regionIds"].getArray()->size() == 1);
    AssetManagerComponent::MaterialSelection received; int loaded = 0, legacy = 0;
    manager.onLoadMaterialSelection = [&](const auto& selection) { received = selection; ++loaded; };
    manager.onLoadAudio = [&](const auto&) { ++legacy; };
    manager.onLoadPronunciations = [&](const auto&, const auto&) { ++legacy; };
    check("enter_pronunciation_passes_selected_variant", manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey))
        && loaded == 1 && legacy == 0 && received.audioVariantId == "annotation_B"
        && received.regionIds == juce::StringArray{"p1"} && received.root == bank);
    check("enter_whole_registration_preserves_id", manager.diagnosticSelectRegistration(audio, "annotation_A")
        && manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey)) && loaded == 2
        && received.audioVariantId == "annotation_A" && received.regionIds.isEmpty());
    manager.onLoadMaterialSelection = {};
    check("multi_variant_never_legacy_callback", !manager.keyPressed(juce::KeyPress(juce::KeyPress::returnKey)) && legacy == 0);
    const auto capture = [&](juce::Component& component, const juce::String& name)
    {
        const auto image = component.createComponentSnapshot(component.getLocalBounds(), true, 1.0f);
        auto stream = output.getChildFile(name).createOutputStream(); juce::PNGImageFormat png;
        return stream && png.writeImageToStream(image, *stream);
    };
    check("tree_capture_written", capture(manager, "variant-registration-tree.png"));
    juce::StringArray editorIds;
    manager.onConfigureMaterialEditor = [&](OtoWaveformEditorComponent& editor)
    {
        const auto rows = editor.diagnosticMaterialRows();
        const auto id = rows.empty() ? juce::String{} : rows.front().materialAudioId;
        editorIds.add(id); editor.setLookAndFeel(&look); editor.setSize(1000, 740);
        check("precise_editor_capture_written", capture(editor, "variant-editor-" + id + ".png"));
        editor.setLookAndFeel(nullptr);
    };
    manager.editMaterial(audio, "p1", "annotation_A"); manager.editMaterial(audio, "p1", "annotation_B");
    check("precise_editor_windows_bound_to_distinct_ids", editorIds == juce::StringArray{"annotation_A", "annotation_B"});
    check("ui_browsing_never_publishes", bytes(manifest) == originalManifest);

    check("session_idless_ambiguous_reject", !Session::open(audio, manifest, error) && error.containsIgnoreCase("ambiguous"));
    auto sourceA = Session::open(audio, manifest, error, "annotation_A");
    auto sourceB = Session::open(audio, manifest, error, "annotation_B");
    check("session_explicit_A_B_baselines", sourceA && sourceB && sourceA->rows().size() == 2 && sourceB->rows().size() == 1
        && sourceB->rows()[0].regionEndSeconds == .9);
    check("session_missing_explicit_no_fallback", !Session::open(audio, manifest, error, "missing"));
    IO::Document beforeSave; IO::loadDocument(manifest, beforeSave, error);
    IO::Document siblingBefore; siblingBefore.audio = {beforeSave.audio[1]}; juce::String siblingText, siblingAfterText;
    IO::encode(siblingBefore, siblingText, error);
    if (sourceA)
    {
        auto changed = sourceA->rows()[0]; changed.name = "edited_A";
        check("session_source_save_exact_A", sourceA->setRow("p1", changed, error) && sourceA->save(error));
    }
    else check("session_source_save_exact_A", false);
    IO::Document afterSave; IO::loadDocument(manifest, afterSave, error);
    IO::Document siblingAfter; if (afterSave.audio.size() == 2) siblingAfter.audio = {afterSave.audio[1]};
    IO::encode(siblingAfter, siblingAfterText, error);
    check("session_A_save_preserves_B_all_data", siblingText == siblingAfterText);
    sourceB = Session::open(audio, manifest, error, "annotation_B");
    if (sourceB && !sourceB->rows().empty())
    {
        const auto sourceRow = sourceB->rows()[0]; auto local = sourceRow; local.name = "private_B";
        Session::NoteScopeBinding binding;
        binding.noteId = "note_B"; binding.audioId = "annotation_B"; binding.regionId = "p1";
        binding.sourceAudio = audio; binding.sourceFingerprint = sourceB->revisionToken(); binding.localRow = local;
        int applies = 0, restores = 0;
        binding.apply = [&](const auto& bound, const auto& row, juce::uint64& revision, juce::String&)
        { ++applies; ++revision; return bound.audioId == "annotation_B" && row.materialAudioId == "annotation_B"; };
        binding.restore = [&](const auto& bound, auto& row, juce::uint64& revision, juce::String&)
        { ++restores; ++revision; row = sourceRow; return bound.audioId == "annotation_B"; };
        const auto sourceBytes = bytes(manifest);
        check("local_scope_bind_selected_B", sourceB->enableNoteScope(binding, error)
            && sourceB->setEditScope(Session::EditScope::thisNote, error) && sourceB->rows()[0].name == "private_B");
        check("local_B_save_restore_no_source_write", sourceB->save(error) && sourceB->restoreLocal(error)
            && applies == 1 && restores == 1 && sourceB->rows()[0].regionEndSeconds == .9
            && bytes(manifest) == sourceBytes);
        check("return_source_exact_B_not_A", sourceB->setEditScope(Session::EditScope::source, error)
            && sourceB->rows().size() == 1 && sourceB->rows()[0].materialAudioId == "annotation_B"
            && sourceB->rows()[0].regionEndSeconds == .9);
        auto privateDraft = sourceB->rows()[0]; privateDraft.name = "discard_me";
        check("discard_reloads_same_variant", sourceB->setRow("p1", privateDraft, error)
            && sourceB->discardAndSetEditScope(Session::EditScope::source, error)
            && sourceB->rows()[0].materialAudioId == "annotation_B" && sourceB->rows()[0].name == "-");
    }
    else check("local_scope_bind_selected_B", false);

    const auto bare = base.getChildFile("bare.wav"), bareManifest = base.getChildFile("otto.ini");
    check("bare_fixture", makeAudio(bare));
    auto bareSession = Session::open(bare, bareManifest, error);
    check("bare_initial_empty", bareSession && bareSession->rows().empty() && !bareManifest.exists());
    if (bareSession)
    {
        check("bare_create_draft", bareSession->createPronunciation(.1, .5, "user_alias", error));
        const auto initialId = bareSession->rows().empty() ? juce::String{} : bareSession->rows()[0].materialAudioId;
        check("bare_discard_no_fallback_or_publication", bareSession->discardAndSetEditScope(Session::EditScope::source, error)
            && bareSession->rows().empty() && !bareManifest.exists());
        check("bare_minted_variant_id_survives_discard", bareSession->createPronunciation(.2, .6, "user_alias", error)
            && bareSession->rows()[0].materialAudioId == initialId);
    }
    check("original_physical_wav_bytes_exact", bytes(audio) == pcmBefore);
    manager.setLookAndFeel(nullptr);
    return output.getChildFile("variant-asset-ui.txt").replaceWithText(report) && ok;
}
}
