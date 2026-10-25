#pragma once
#include "../OtoWaveformEditorComponent.h"
#include "../MaterialAnnotationEditor.h"
#include "../OttoMaterialIO.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_graphics/juce_graphics.h>
#include <cmath>
#include <functional>
#include <iostream>

namespace hachi
{
// Offline component diagnostic only: no application window, audio device, or real
// user assets. Invoke from a diagnostic branch on JUCE's message thread.
inline bool runUnifiedMaterialEditorSmoke(const juce::File& output)
{
    int total = 0, failed = 0;
    juce::String log, error;
    const auto root = output.getChildFile("unified-material-" + juce::Uuid().toString());
    const auto check = [&](const juce::String& name, bool success)
    {
        ++total; if (!success) ++failed;
        log += name + "=" + (success ? "PASS" : "FAIL")
            + (success || error.isEmpty() ? juce::String{} : " | " + error) + "\n";
    };
    if (!root.createDirectory().wasOk()) return false;
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread())
    {
        root.getChildFile("verification.txt").replaceWithText("FAIL: requires JUCE message thread\n");
        return false;
    }
    const auto near = [](double a, double b) { return std::abs(a - b) < 1.e-8; };
    const auto finish = [&]
    {
        const bool written = root.getChildFile("verification.txt").replaceWithText(log);
        std::cout << "unified_material_passed=" << total - failed << "|total=" << total
                  << "|fixture=" << root.getFullPathName() << std::endl;
        if (failed) std::cerr << log << std::endl;
        return failed == 0 && written;
    };
    I18n strings; strings.setLanguage(I18n::Language::enUS);
    HachiLookAndFeel look;
    // Restore every mutable palette colour when the offline diagnostic exits.
    struct PaletteRestore
    {
        std::vector<std::pair<juce::Colour*, juce::Colour>> colours;
        PaletteRestore()
        {
            for (auto* p : {&Palette::background, &Palette::base, &Palette::panel, &Palette::panelRaised,
                &Palette::button, &Palette::buttonHover, &Palette::border, &Palette::graphBackground,
                &Palette::clipBackground, &Palette::grid, &Palette::beatGrid, &Palette::accent,
                &Palette::accentLight, &Palette::noteFill, &Palette::noteLight, &Palette::noteEdge,
                &Palette::pitchLine, &Palette::playhead, &Palette::text, &Palette::textMuted, &Palette::scrollThumb})
                colours.emplace_back(p, *p);
        }
        ~PaletteRestore() { for (const auto& p : colours) *p.first = p.second; }
    } restore;
    const auto audio = root.getChildFile("source.wav");
    const auto manifest = root.getChildFile("otto.ini");
    {
        juce::AudioBuffer<float> pcm(1, 96000);
        for (int i = 0; i < pcm.getNumSamples(); ++i)
            pcm.setSample(0, i, .15f * static_cast<float>(std::sin(juce::MathConstants<double>::twoPi * 220 * i / 48000.0)));
        auto stream = audio.createOutputStream();
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer(stream
            ? format.createWriterFor(stream.release(), 48000, 1, 16, {}, 0) : nullptr);
        const bool ok = writer && writer->writeFromAudioSampleBuffer(pcm, 0, pcm.getNumSamples());
        check("own_two_second_wav_fixture", ok);
        if (!ok) return finish();
    }
    juce::MemoryBlock sourceBefore; check("read_original_pcm", audio.loadFileAsData(sourceBefore));
    int callbacks = 0, validations = 0;
    std::vector<SampleRegionSetting> published;
    auto editor = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings,
        [&](const auto& rows) { ++callbacks; published = rows; }, false, &error);
    check("actual_precise_factory", editor != nullptr);
    if (!editor) return finish();
    editor->setLookAndFeel(&look);
    editor->setVisible(true);
    check("bare_audio_starts_in_native_overview", editor->diagnosticMaterialOverview()
        && editor->diagnosticMaterialRows().empty() && !manifest.exists());

    const auto overviewOf = [](OtoWaveformEditorComponent& owner) -> MaterialAnnotationEditor*
    {
        for (int i = 0; i < owner.getNumChildComponents(); ++i)
            if (auto* c = dynamic_cast<MaterialAnnotationEditor*>(owner.getChildComponent(i))) return c;
        return nullptr;
    };
    const auto button = [](juce::Component& owner, const juce::String& text) -> juce::Button*
    {
        for (int i = 0; i < owner.getNumChildComponents(); ++i)
            if (auto* b = dynamic_cast<juce::Button*>(owner.getChildComponent(i)))
                if (b->getButtonText() == text) return b;
        return nullptr;
    };
    const auto click = [&](juce::Component& owner, const juce::String& text)
    {
        auto* b = button(owner, text);
        if (!b || !b->isVisible() || !b->isEnabled() || !b->onClick) return false;
        b->onClick(); return true;
    };
    const auto mouse = [](juce::Component& view, juce::Point<float> at, juce::Point<float> down, int clicks = 1)
    {
        return juce::MouseEvent(juce::Desktop::getInstance().getMainMouseSource(), at,
            juce::ModifierKeys(juce::ModifierKeys::leftButtonModifier), 0, 0, 0, 0, 0,
            &view, &view, juce::Time::getCurrentTime(), down, juce::Time::getCurrentTime(), clicks, at != down);
    };
    const auto visibleCombos = [](juce::Component& owner)
    {
        int count = 0;
        for (int i = 0; i < owner.getNumChildComponents(); ++i)
            if (auto* c = dynamic_cast<juce::ComboBox*>(owner.getChildComponent(i)); c && c->isVisible()) ++count;
        return count;
    };
    bool nativeDefault = false;
    for (int i = 0; i < editor->getNumChildComponents(); ++i)
        if (auto* c = dynamic_cast<juce::ComboBox*>(editor->getChildComponent(i)); c && c->isVisible())
            nativeDefault = c->getSelectedId() == 1 && c->getNumItems() == 4;
    check("native_default_and_segment_selector_hidden_in_overview", nativeDefault && visibleCombos(*editor) == 1);
    // Check only application-owned controls; JUCE's internal text/viewport
    // children may intentionally extend outside their clipped parent.
    const auto layout = [&](OtoWaveformEditorComponent& owner, const juce::String& tag)
    {
        bool fits = true; int controls = 0;
        const auto inspect = [&](juce::Component& parent)
        {
            for (int i = 0; i < parent.getNumChildComponents(); ++i)
            {
                auto* c = parent.getChildComponent(i);
                if (!c->isVisible()) continue;
                if (!(dynamic_cast<juce::Button*>(c) || dynamic_cast<juce::ComboBox*>(c)
                    || dynamic_cast<juce::TextEditor*>(c))) continue;
                ++controls;
                if (c->getWidth() <= 0 || c->getHeight() <= 0 || !parent.getLocalBounds().contains(c->getBounds()))
                {
                    fits = false;
                    log += tag + " bad_control=" + c->getName() + " bounds=" + c->getBounds().toString() + "\n";
                }
            }
        };
        inspect(owner);
        if (auto* overview = overviewOf(owner); overview && overview->isVisible()) inspect(*overview);
        check(tag + "_visible_control_bounds", fits && controls >= 4);
    };
    const auto snapshot = [&](OtoWaveformEditorComponent& owner, const juce::String& tag)
    {
        const auto image = owner.createComponentSnapshot(owner.getLocalBounds(), true, 1.f);
        const auto file = root.getChildFile(tag + ".png");
        auto stream = file.createOutputStream(); juce::PNGImageFormat png;
        check(tag + "_snapshot", image.isValid() && image.getWidth() == owner.getWidth()
            && stream && png.writeImageToStream(image, *stream));
    };
    // A real pending mouse selection is cancelled through the actual overview
    // button; no draft rows or files should be created by a cancelled range.
    auto* overview = overviewOf(*editor);
    check("factory_embeds_shared_overview", overview != nullptr);
    if (overview)
    {
        juce::Component* wave = nullptr;
        for (int i = 0; i < overview->getNumChildComponents(); ++i)
        {
            auto* c = overview->getChildComponent(i);
            if (!dynamic_cast<juce::Label*>(c) && !dynamic_cast<juce::TextEditor*>(c)
                && !dynamic_cast<juce::Button*>(c) && !dynamic_cast<juce::ListBox*>(c)) { wave = c; break; }
        }
        check("overview_waveform_available", wave != nullptr);
        if (wave)
        {
            const juce::Point<float> from(60.f, 60.f), to(230.f, 60.f);
            wave->mouseDown(mouse(*wave, from, from)); wave->mouseDrag(mouse(*wave, to, from));
            wave->mouseUp(mouse(*wave, to, from));
            check("pending_range_cancel_uses_real_button", click(*overview, "Remove current alias")
                && editor->diagnosticMaterialRows().empty() && !manifest.exists() && callbacks == 0);
        }
    }
    check("create_first_pronunciation_via_factory", editor->diagnosticCreateMaterialAlias(.1, .9, "ka", error));
    check("create_second_pronunciation_via_factory", editor->diagnosticCreateMaterialAlias(1.1, 1.9, "ki", error));
    auto rows = editor->diagnosticMaterialRows();
    check("two_distinct_authored_ranges", rows.size() == 2 && near(rows[0].regionStartSeconds, .1)
        && near(rows[0].regionEndSeconds, .9) && near(rows[1].regionStartSeconds, 1.1));
    if (rows.size() != 2) return finish();
    const auto first = rows[0].materialRegionId, second = rows[1].materialRegionId;
    check("same_range_additional_alias", editor->diagnosticCreateMaterialAlias(.1, .9, "ga", error)
        && editor->diagnosticMaterialRows().size() == 2
        && editor->diagnosticMaterialRows()[0].additionalAliases.contains("ga"));
    check("remove_additional_alias", editor->diagnosticRemoveMaterialAlias(first, "ga", error)
        && editor->diagnosticMaterialRows()[0].additionalAliases.isEmpty());
    check("create_temporary_alias", editor->diagnosticCreateMaterialAlias(.92, 1.02, "temporary", error));
    rows = editor->diagnosticMaterialRows();
    juce::String temporary;
    for (const auto& row : rows) if (row.name == "temporary") temporary = row.materialRegionId;
    check("remove_only_alias_removes_pronunciation", temporary.isNotEmpty()
        && editor->diagnosticRemoveMaterialAlias(temporary, "temporary", error)
        && editor->diagnosticMaterialRows().size() == 2);
    check("invalid_creation_changes_nothing", !editor->diagnosticCreateMaterialAlias(.5, .4, "bad", error)
        && editor->diagnosticMaterialRows().size() == 2);
    error.clear();

    for (const auto theme : {juce::String("light"), juce::String("dark")})
    {
        Palette::applyTheme(theme, restore.colours[11].second, restore.colours[12].second, restore.colours[13].second);
        look.refreshColours();
        editor->sendLookAndFeelChange();
        for (const int width : {900, 1000})
        {
            editor->setSize(width, 780);
            const auto tag = "precise-" + theme + "-" + juce::String(width) + "-overview";
            layout(*editor, tag); snapshot(*editor, tag);
        }
    }
    check("select_overview_row_does_not_open_detail", editor->selectMaterialPronunciation(first, false)
        && editor->diagnosticMaterialOverview());
    // Invoke the actual overview waveform's double-click handler, not a second
    // editor or parallel model. The point is inside the first pronunciation.
    bool doubled = false;
    if (overview)
        for (int i = 0; i < overview->getNumChildComponents(); ++i)
        {
            auto* c = overview->getChildComponent(i);
            if (dynamic_cast<juce::Label*>(c) || dynamic_cast<juce::TextEditor*>(c)
                || dynamic_cast<juce::Button*>(c) || dynamic_cast<juce::ListBox*>(c)) continue;
            const juce::Point<float> at(10.f + (c->getWidth() - 20.f) * .25f, 60.f);
            c->mouseDoubleClick(mouse(*c, at, at, 2)); doubled = !editor->diagnosticMaterialOverview(); break;
        }
    check("doubleclick_opens_existing_oto_detail", doubled);
    check("select_detail_via_factory", editor->selectMaterialPronunciation(first, true));
    check("native_segment_selectors_appear_in_detail_only", visibleCombos(*editor) == 3);
    check("multiple_pronunciations_lock_outer_range", editor->diagnosticOuterRangeLocked());
    const auto offset = editor->diagnosticParameterText(0), cutoff = editor->diagnosticParameterText(2);
    editor->diagnosticTypeParameter(0, "300"); editor->diagnosticTypeParameter(2, "200");
    editor->diagnosticDragOffsetTo(350);
    // Actual mouse cutoff drag (disabled handles must not be hittable).
    {
        auto& view = editor->diagnosticWaveformComponent();
        const float y = static_cast<float>(editor->diagnosticWaveformBounds().getCentreY());
        const juce::Point<float> from(editor->diagnosticXForTime(900), y), to(editor->diagnosticXForTime(700), y);
        view.mouseDown(mouse(view, from, from)); view.mouseDrag(mouse(view, to, from)); view.mouseUp(mouse(view, to, from));
    }
    check("typed_and_drag_outer_parameters_unchanged", editor->diagnosticParameterText(0) == offset
        && editor->diagnosticParameterText(2) == cutoff);
    check("named_vowel_split_opens_without_automatic_cut", click(*editor, "Split named vowel")
        && editor->diagnosticMaterialVowelItems().size() == 1);
    check("split_internal_vowel_requires_manual_names", editor->diagnosticConfirmMaterialVowelSplit("an", "ng"));
    editor->diagnosticDragMaterialBoundary(0, .43);
    check("commit_internal_boundary_on_return", editor->diagnosticMaterialBack());
    rows = editor->diagnosticMaterialRows();
    check("two_internal_vowels_and_dragged_shared_boundary", rows.size() == 2 && rows[0].segments.size() == 2
        && rows[0].segments[0].role == NativeSegmentRole::vowel && rows[0].segments[1].role == NativeSegmentRole::vowel
        && rows[0].vowelSplitsExplicit && rows[0].segments[0].alias == "an" && rows[0].segments[1].alias == "ng"
        && near(rows[0].segments[0].sourceEndSeconds, .43) && near(rows[0].segments[1].sourceStartSeconds, .43)
        && near(rows[0].regionStartSeconds, .1) && near(rows[0].regionEndSeconds, .9));
    check("return_preserves_second_pronunciation", rows.size() == 2 && rows[1].materialRegionId == second
        && rows[1].name == "ki" && near(rows[1].regionEndSeconds, 1.9));

    // Encode the real rows to compare all persistent metadata, IDs, and segments
    // after view-only mode switches, rather than comparing a few handpicked fields.
    const auto encodeRows = [&](const std::vector<SampleRegionSetting>& value)
    {
        OttoMaterialIO::Document doc;
        doc.audio.push_back({value.empty() ? juce::String("audio") : value.front().materialAudioId, "source.wav", value});
        juce::String text, problem;
        if (!OttoMaterialIO::encode(doc, text, problem)) { error = problem; return juce::String{}; }
        return text;
    };
    const auto beforeModes = encodeRows(rows);
    check("native_draft_encodes_before_view_switches", beforeModes.isNotEmpty());
    for (const int mode : {2, 3, 4, 1})
    {
        check("mode_" + juce::String(mode) + "_detail_selection", editor->selectMaterialPronunciation(first, true));
        editor->diagnosticMaterialMode(mode);
        check("mode_" + juce::String(mode) + "_single_control_native_only",
            button(*editor, "Single syllable / phoneme")->isVisible() == (mode == 1));
        for (const int width : {900, 1000})
        {
            editor->setSize(width, 780);
            layout(*editor, "precise-mode-" + juce::String(mode) + "-" + juce::String(width));
        }
        check("mode_" + juce::String(mode) + "_return", editor->diagnosticMaterialBack());
        check("mode_" + juce::String(mode) + "_no_native_data_loss", encodeRows(editor->diagnosticMaterialRows()) == beforeModes);
    }
    editor->setPublicationValidator([&](const auto& value, juce::String& reason)
    { ++validations; reason = "Intentional smoke preflight refusal"; return value.empty(); });
    editor->diagnosticSave();
    check("publication_preflight_prevents_first_write", validations == 1 && callbacks == 0 && !manifest.exists());
    editor->setPublicationValidator([&](const auto& value, juce::String&)
    { ++validations; return value.size() == 2; });
    editor->diagnosticSave();
    check("default_save_is_otto_and_callback_has_real_rows", manifest.existsAsFile() && callbacks == 1
        && validations == 2 && encodeRows(published) == beforeModes);
    check("no_legacy_annotation_files_written", !root.getChildFile("oto.ini").exists()
        && !root.getChildFile("oto.jie.ini").exists() && !root.getChildFile("oto4.ini").exists()
        && !root.getChildFile("otomou.ini").exists() && !SampleSettings::sidecarFor(audio).exists());
    const auto savedBytes = manifest.loadFileAsString();
    check("saved_manifest_nonempty", savedBytes.isNotEmpty());
    // Quick-wrench and precise routes instantiate exactly the same factory class.
    int quickCallbacks = 0;
    auto quick = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings,
        [&](const auto&) { ++quickCallbacks; }, true, &error);
    check("actual_quick_factory_reopens_saved_rows", quick && quick->diagnosticMaterialOverview()
        && encodeRows(quick->diagnosticMaterialRows()) == beforeModes);
    if (quick)
    {
        quick->setLookAndFeel(&look); quick->setVisible(true);
        for (const auto theme : {juce::String("light"), juce::String("dark")})
        {
            Palette::applyTheme(theme, restore.colours[11].second, restore.colours[12].second, restore.colours[13].second);
            look.refreshColours();
            quick->sendLookAndFeelChange();
            for (const int width : {900, 1000})
            {
                quick->setSize(width, 780);
                const auto tag = "quick-" + theme + "-" + juce::String(width);
                layout(*quick, tag + "-overview"); snapshot(*quick, tag + "-overview");
                check(tag + "_detail_select", quick->selectMaterialPronunciation(first, true));
                layout(*quick, tag + "-detail"); snapshot(*quick, tag + "-detail");
                check(tag + "_back", quick->diagnosticMaterialBack());
                check(tag + "_no_view_loss", encodeRows(quick->diagnosticMaterialRows()) == beforeModes);
            }
        }
        check("quick_cancel_draft_created", quick->diagnosticCreateMaterialAlias(.94, 1.04, "cancelled", error));
        int quickPreflights = 0;
        quick->setPublicationValidator([&](const auto& value, juce::String& reason)
        { ++quickPreflights; reason = "Intentional existing-file preflight refusal"; return value.empty(); });
        quick->diagnosticSave();
        check("existing_manifest_preflight_preserves_exact_bytes", quickPreflights == 1
            && quickCallbacks == 0 && manifest.loadFileAsString() == savedBytes);
        quick.reset();
        check("closing_quick_draft_never_saves", quickCallbacks == 0 && manifest.loadFileAsString() == savedBytes);
    }
    check("precise_detail_for_light_dark_capture", editor->selectMaterialPronunciation(first, true));
    for (const auto theme : {juce::String("light"), juce::String("dark")})
    {
        Palette::applyTheme(theme, restore.colours[11].second, restore.colours[12].second, restore.colours[13].second);
        look.refreshColours();
        editor->sendLookAndFeelChange();
        for (const int width : {900, 1000})
        {
            editor->setSize(width, 780);
            snapshot(*editor, "precise-" + theme + "-" + juce::String(width) + "-detail");
        }
    }
    editor->diagnosticTypeParameter(4, "25");
    editor.reset();
    check("closing_precise_unsaved_detail_never_saves", callbacks == 1 && manifest.loadFileAsString() == savedBytes);
    auto reopened = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings, {}, false, &error);
    check("reopen_keeps_saved_only_not_cancelled_changes", reopened && encodeRows(reopened->diagnosticMaterialRows()) == beforeModes);
    // Per-recording view preference: actual checkbox in both views, cold disk
    // reload, source boundaries, existing siblings, and empty unsaved recordings.
    juce::PropertiesFile::Options viewOptions; viewOptions.applicationName = "single-syllable-smoke";
    viewOptions.storageFormat = juce::PropertiesFile::storeAsXML;
    const auto viewFile = root.getChildFile("view-preferences.xml");
    juce::PropertiesFile viewPreferences(viewFile, viewOptions);
    const auto toggleSingle = [&](OtoWaveformEditorComponent& owner, bool enabled)
    {
        auto* control = button(owner, "Single syllable / phoneme");
        if (!control || !control->isVisible() || !control->isEnabled() || !control->onClick) return false;
        control->setToggleState(enabled, juce::dontSendNotification); control->onClick();
        return control->getToggleState() == enabled && owner.diagnosticSingleSyllableMaterial() == enabled;
    };
    if (reopened)
    {
        reopened->setLookAndFeel(&look); reopened->setVisible(true);
        reopened->bindMaterialViewPreferences(viewPreferences);
        check("single_default_off_in_overview", !reopened->diagnosticSingleSyllableMaterial() && reopened->diagnosticMaterialOverview());
        check("single_select_existing_second", reopened->selectMaterialPronunciation(second, false));
        const auto untouchedRows = encodeRows(reopened->diagnosticMaterialRows());
        check("single_checkbox_opens_selected_detail", toggleSingle(*reopened, true)
            && !reopened->diagnosticMaterialOverview() && reopened->diagnosticMaterialRegionId() == second);
        check("single_mode_preserves_all_existing_rows", encodeRows(reopened->diagnosticMaterialRows()) == untouchedRows
            && manifest.loadFileAsString() == savedBytes);
        check("single_detail_unlocks_outer_bounds_and_hides_back", !reopened->diagnosticOuterRangeLocked()
            && !button(*reopened, "Back to pronunciation overview")->isVisible() && !reopened->diagnosticMaterialBack());
        for (const int width : {900, 1100})
        {
            reopened->setSize(width, 780);
            layout(*reopened, "single-detail-" + juce::String(width));
            snapshot(*reopened, "single-detail-" + juce::String(width));
        }
        juce::PropertiesFile reloadedPreferences(viewFile, viewOptions);
        auto remembered = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings, {}, false, &error);
        if (remembered) remembered->bindMaterialViewPreferences(reloadedPreferences);
        check("single_disk_preference_reopens_same_pronunciation", remembered && remembered->diagnosticSingleSyllableMaterial()
            && !remembered->diagnosticMaterialOverview() && remembered->diagnosticMaterialRegionId() == second);
        for (const int mode : {2, 3, 4, 1})
        {
            reopened->diagnosticMaterialMode(mode);
            check("single_enabled_mode_" + juce::String(mode) + "_visibility_and_preference",
                button(*reopened, "Single syllable / phoneme")->isVisible() == (mode == 1)
                && reopened->diagnosticSingleSyllableMaterial());
        }
        check("single_navigation_never_returns_to_overview", reopened->selectMaterialPronunciation(first, false)
            && !reopened->diagnosticMaterialOverview() && reopened->focusMaterialSourceRange({1.1, 1.9}, false)
            && !reopened->diagnosticMaterialOverview());
        reopened->selectMaterialPronunciation(second, true);
        check("single_invalid_pending_weight_is_rejected", !reopened->diagnosticEditMaterialVowel("ki", "invalid"));
        check("single_invalid_fields_keep_detail_and_checkbox", !toggleSingle(*reopened, false)
            && reopened->diagnosticSingleSyllableMaterial() && !reopened->diagnosticMaterialOverview());
        check("single_repair_pending_weight", reopened->diagnosticEditMaterialVowel("ki", "1"));
        const auto originalSecond = reopened->diagnosticMaterialSourceSelection();
        reopened->diagnosticDragOffsetTo((originalSecond->getStart() + .01) * 1000);
        check("single_checkbox_off_returns_overview", toggleSingle(*reopened, false) && reopened->diagnosticMaterialOverview());
        const auto moved = reopened->diagnosticMaterialSourceSelection();
        check("single_outer_edit_keeps_sibling_and_source_file", moved && near(moved->getStart(), originalSecond->getStart() + .01)
            && near(reopened->diagnosticMaterialRows().front().regionStartSeconds, .1) && manifest.loadFileAsString() == savedBytes);
        reopened->setSize(900, 780); snapshot(*reopened, "single-off-overview");
        juce::PropertiesFile offPreferences(viewFile, viewOptions);
        auto off = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings, {}, false, &error);
        if (off) off->bindMaterialViewPreferences(offPreferences);
        check("single_uncheck_persists", off && !off->diagnosticSingleSyllableMaterial() && off->diagnosticMaterialOverview());
        reopened.reset();
    }
    const auto bareAudio = root.getChildFile("single-bare.wav");
    check("single_bare_fixture", audio.copyFileTo(bareAudio));
    auto bare = OtoWaveformEditorComponent::forMaterial(bareAudio, manifest, strings, {}, false, &error);
    if (bare)
    {
        bare->bindMaterialViewPreferences(viewPreferences);
        check("single_preference_is_per_recording", !bare->diagnosticSingleSyllableMaterial() && bare->diagnosticMaterialRows().empty());
        check("single_bare_source_range", bare->focusMaterialSourceRange({.2, .8}, false));
        check("single_bare_checkbox_creates_only_one_unsaved_draft", toggleSingle(*bare, true)
            && bare->diagnosticMaterialRows().size() == 1 && !bare->diagnosticMaterialOverview()
            && near(bare->diagnosticMaterialRows()[0].regionStartSeconds, .2)
            && near(bare->diagnosticMaterialRows()[0].regionEndSeconds, .8) && manifest.loadFileAsString() == savedBytes);
        auto other = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings, {}, false, &error);
        if (other) other->bindMaterialViewPreferences(viewPreferences);
        check("single_enabled_recording_does_not_change_other_recording", other && other->diagnosticMaterialOverview()
            && !other->diagnosticSingleSyllableMaterial());
        check("single_placeholder_allows_return", toggleSingle(*bare, false) && bare->diagnosticMaterialOverview());
    }
    else check("single_bare_factory", false);
    strings.setLanguage(I18n::Language::zhCN);
    auto chinese = OtoWaveformEditorComponent::forMaterial(audio, manifest, strings, {}, false, &error);
    if (chinese)
    {
        chinese->setLookAndFeel(&look); chinese->setVisible(true); chinese->setSize(900, 780);
        auto* control = button(*chinese, juce::String::fromUTF8("单音节音素"));
        check("single_chinese_label_and_tooltip", control && control->isVisible()
            && control->getTooltip().contains(juce::String::fromUTF8("发音总览")));
        snapshot(*chinese, "single-chinese-overview");
        for (const int mode : {2, 3, 4, 1})
        {
            chinese->diagnosticMaterialMode(mode);
            check("chinese_overview_mode_" + juce::String(mode) + "_single_visibility", control && control->isVisible() == (mode == 1));
        }
        check("single_chinese_detail", chinese->setSingleSyllableMaterial(true) && control && control->isVisible());
        snapshot(*chinese, "single-chinese-detail");
    }
    else check("single_chinese_factory", false);
    strings.setLanguage(I18n::Language::enUS);
    // Exercise actual line hits, including the default coincident C/P/O at
    // source start, and persist both source limits through a fresh editor.
    const auto trimRoot = root.getChildFile("single-trim"); trimRoot.createDirectory();
    const auto trimAudio = trimRoot.getChildFile("trim.wav"), trimManifest = trimRoot.getChildFile("otto.ini");
    check("single_trim_audio_fixture", audio.copyFileTo(trimAudio));
    auto trim = OtoWaveformEditorComponent::forMaterial(trimAudio, trimManifest, strings, {}, false, &error);
    const auto dragLine = [&](OtoWaveformEditorComponent& owner, double fromMs, double toMs)
    {
        auto& wave = owner.diagnosticWaveformComponent();
        const float y = static_cast<float>(owner.diagnosticWaveformBounds().getY() + 24);
        const juce::Point<float> from(owner.diagnosticXForTime(fromMs), y), to(owner.diagnosticXForTime(toMs), y);
        wave.mouseDown(mouse(wave, from, from)); wave.mouseDrag(mouse(wave, to, from)); wave.mouseUp(mouse(wave, to, from));
    };
    const auto hasRange = [&](OtoWaveformEditorComponent& owner, double start, double end)
    {
        const auto range = owner.diagnosticMaterialSourceSelection();
        return range && std::abs(range->getStart() - start) < 1.e-6 && std::abs(range->getEnd() - end) < 1.e-6;
    };
    if (trim)
    {
        trim->bindMaterialViewPreferences(viewPreferences); trim->setSize(1000, 780);
        check("single_trim_enable", trim->setSingleSyllableMaterial(true));
        dragLine(*trim, 0, 120); dragLine(*trim, 2000, 1720);
        check("single_purple_line_hits_trim_both_ends", hasRange(*trim, .12, 1.72));
        dragLine(*trim, 120, 80); dragLine(*trim, 1720, 1850);
        check("single_purple_lines_extend_both_ends", hasRange(*trim, .08, 1.85));
        check("single_drag_is_unsaved_until_save", !trimManifest.exists());
        trim->diagnosticSave();
        check("single_trim_save_publishes", trimManifest.existsAsFile() && !trim->diagnosticScopeDirty());
        const auto trimBytes = trimManifest.loadFileAsString();
        trim.reset();
        juce::PropertiesFile coldPreferences(viewFile, viewOptions);
        trim = OtoWaveformEditorComponent::forMaterial(trimAudio, trimManifest, strings, {}, false, &error);
        if (trim)
        {
            trim->bindMaterialViewPreferences(coldPreferences); trim->setSize(1000, 780);
            check("single_trim_reopens_saved_limits_and_mode", trim->diagnosticSingleSyllableMaterial()
                && !trim->diagnosticMaterialOverview() && hasRange(*trim, .08, 1.85));
            check("single_trim_reopens_offset_and_cutoff_values",
                std::abs(trim->diagnosticParameterText(0).getDoubleValue() - 80) < .001
                && std::abs(trim->diagnosticParameterText(2).getDoubleValue() - 150) < .001);
            trim->setLookAndFeel(&look); snapshot(*trim, "single-saved-purple-limits");
            const auto id = trim->diagnosticMaterialRegionId();
            check("single_trim_disable_restores_multi_lock", trim->setSingleSyllableMaterial(false)
                && trim->selectMaterialPronunciation(id, true) && trim->diagnosticOuterRangeLocked());
            dragLine(*trim, 80, 140); dragLine(*trim, 1850, 1800);
            check("multi_mode_one_row_still_locks_purple_limits", hasRange(*trim, .08, 1.85));
            check("multi_mode_does_not_publish_drag", trimManifest.loadFileAsString() == trimBytes);
            trim->setSingleSyllableMaterial(true); dragLine(*trim, 80, 160); trim.reset();
            auto cancelled = OtoWaveformEditorComponent::forMaterial(trimAudio, trimManifest, strings, {}, false, &error);
            if (cancelled) cancelled->bindMaterialViewPreferences(coldPreferences);
            check("single_unsaved_drag_does_not_replace_saved_limits", cancelled && hasRange(*cancelled, .08, 1.85)
                && trimManifest.loadFileAsString() == trimBytes);
        }
        else check("single_trim_reopen_factory", false);
        juce::MemoryBlock trimPcm;
        check("single_trim_preserves_pcm", trimAudio.loadFileAsData(trimPcm) && trimPcm == sourceBefore);
    }
    else check("single_trim_factory", false);
    juce::MemoryBlock sourceAfter;
    check("source_wav_bytes_exactly_unchanged", audio.loadFileAsData(sourceAfter) && sourceAfter == sourceBefore);
    return finish();
}
}
