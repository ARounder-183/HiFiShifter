#pragma once
#include "../MainComponent.h"
#include "../Theme.h"
#include <iostream>

namespace hachi
{
// Included once by MainComponent.cpp: exercise the production menu, dispatch
// and look-and-feel without launching a transient native popup window.
bool MainComponent::diagnosticEngineViewCapabilities(const juce::File& output)
{
    if (output.createDirectory().failed()) return false;
    stopTimer(); audio.stop(); pianoRoll.setReadsVoicebankInBackground(false);
    bool ok = true; juce::String log;
    const auto check = [&](const juce::String& name, bool pass)
    { ok &= pass; log += name + "=" + (pass ? "PASS" : "FAIL") + "\n"; std::cout << name << '=' << pass << std::endl; };
    ProjectData fixture;
    for (const auto type : { TrackType::hachi, TrackType::utau, TrackType::diffSinger })
    {
        TrackData track; track.id = trackTypeName(type); track.name = track.id;
        track.trackType = type; track.compose = true;
        track.pitchAlgorithm = type == TrackType::hachi ? PitchAlgorithm::nsfHifigan : PitchAlgorithm::utau;
        fixture.tracks.push_back(track);
    }
    project.resetDocument(fixture); project.dispatchPendingMessages();
    selectedClipId.clear(); selectedNoteId.clear();
    const auto choose = [&](int index)
    { diagnosticSelectTrack(fixture.tracks[static_cast<std::size_t>(index)].id); };
    const auto enabled = [&](int id) -> std::optional<bool>
    {
        auto menu = getMenuForIndex(3, {});
        for (juce::PopupMenu::MenuItemIterator item(menu); item.next();)
            if (item.getItem().itemID == id) return item.getItem().isEnabled;
        return {};
    };
    const auto revision = project.revisionNumber(); const auto fingerprint = project.contentFingerprint();
    viewOptions.phonemeInfo = true;
    if (preferences) storeViewOptions(*preferences, viewOptions);
    for (int i = 0; i < 3; ++i)
    {
        choose(i);
        check("semantic_type_" + juce::String(i), selectedEditorTrackType() == *fixture.tracks[static_cast<std::size_t>(i)].trackType);
        check("actual_menu_phoneme_" + juce::String(i), enabled(609) == std::optional<bool>(i != 0));
        check("actual_menu_native_wave_" + juce::String(i), enabled(605) == std::optional<bool>(i == 0));
        check("actual_menu_consonant_dash_" + juce::String(i), enabled(608) == std::optional<bool>(i == 0));
        check("common_pitch_and_envelope_" + juce::String(i), enabled(602).value_or(false) && enabled(604).value_or(false) && enabled(607).value_or(false));
    }
    choose(0);
    // A stale UTAU picker cannot grant a semantic Hachi feature.
    pitchAlgorithm.addItem("stale UTAU", 7); pitchAlgorithm.setSelectedId(7, juce::dontSendNotification);
    check("stale_picker_cannot_enable_Hachi_spelling", !enabled(609).value_or(true));
    menuItemSelected(609, 3); applyEditorViewChoice(9);
    check("disabled_dispatch_preserves_preference", viewOptions.phonemeInfo && preferences && preferences->getBoolValue("ui.showPhonemeInfo", false));
    choose(1); menuItemSelected(609, 3);
    check("UTAU_dispatch_toggles_real_preference", !viewOptions.phonemeInfo && preferences && !preferences->getBoolValue("ui.showPhonemeInfo", true));
    choose(2); menuItemSelected(609, 3);
    check("DS_native_alternative_dispatch_and_retained_preference", viewOptions.phonemeInfo && preferences && preferences->getBoolValue("ui.showPhonemeInfo", false));
    viewOptions.nativeRenderedWaveform = true; viewOptions.consonantPitchDashed = true;
    menuItemSelected(605, 3); menuItemSelected(608, 3);
    check("DS_disabled_native_dispatch_is_noop", viewOptions.nativeRenderedWaveform && viewOptions.consonantPitchDashed);
    for (const auto type : { TrackType::hachi, TrackType::utau, TrackType::diffSinger })
    {
        ViewOptions initial; const auto after = afterViewMenuChoice(initial, 3, type);
        check("waveform_routing_" + trackTypeName(type), type == TrackType::hachi
            ? after.nativeWaveform != initial.nativeWaveform && after.utauWaveform == initial.utauWaveform
            : after.utauWaveform != initial.utauWaveform && after.nativeWaveform == initial.nativeWaveform);
    }
    choose(0); choose(1); choose(2); choose(0);
    check("view_selection_dispatch_never_edits_music", project.revisionNumber() == revision && project.contentFingerprint() == fingerprint);
    for (const auto mode : { UtauMode::classic, UtauMode::jie, UtauMode::mou })
    {
        auto variant = fixture; variant.tracks[1].utauMode = mode;
        project.resetDocument(variant); project.dispatchPendingMessages(); choose(1);
        check("all_UTAU_modes_keep_spelling_" + utauModeKey(mode), enabled(609).value_or(false));
    }
    auto nsf = fixture; nsf.tracks[1].pitchAlgorithm = PitchAlgorithm::nsfHifigan;
    nsf.tracks[1].outputEngine = UtauOutputEngine::pcNsfHifigan;
    project.resetDocument(nsf); project.dispatchPendingMessages(); choose(1);
    check("UTAU_NSF_output_retains_semantic_capability", enabled(609).value_or(false) && !enabled(605).value_or(true));
    project.resetDocument(fixture); project.dispatchPendingMessages();
    const auto oldTheme = preferences ? preferences->getValue("ui.theme", "dark") : juce::String("dark");
    for (const auto theme : { juce::String("dark"), juce::String("light") })
    {
        Palette::applyTheme(theme, juce::Colour(0xff7f69ca), juce::Colour(0xffcbcbfa), juce::Colour(0xff269b9b));
        HachiLookAndFeel look;
        juce::Image image(juce::Image::RGB, 1020, 380, true);
        { // Flush deferred native graphics before encoding the evidence PNG.
        juce::Graphics graphics(image);
        graphics.fillAll(Palette::panelRaised);
        for (int i = 0; i < 3; ++i)
        {
            choose(i); graphics.setColour(Palette::text); graphics.setFont(17.f);
            graphics.drawText(fixture.tracks[static_cast<std::size_t>(i)].name, i * 340 + 12, 5, 320, 30, juce::Justification::centredLeft);
            auto menu = getMenuForIndex(3, {}); int row = 0;
            for (juce::PopupMenu::MenuItemIterator item(menu); item.next();)
            {
                const auto& entry = item.getItem();
                if (entry.itemID < 601 || entry.itemID > 609) continue;
                look.drawPopupMenuItem(graphics, { i * 340, 40 + row++ * 36, 338, 36 }, false,
                    entry.isEnabled, false, entry.isTicked, false, entry.text, {}, nullptr, nullptr);
            }
        }
        }
        check("menu_capture_contains_painted_surface_" + theme, image.getPixelAt(0, 0) == Palette::panelRaised);
        auto stream = output.getChildFile("engine-view-" + theme + ".png").createOutputStream();
        check("actual_menu_render_" + theme, stream && juce::PNGImageFormat().writeImageToStream(image, *stream));
        juce::Image disabled(juce::Image::RGB, 338, 36, true), hovered(juce::Image::RGB, 338, 36, true);
        { juce::Graphics g(disabled); g.fillAll(Palette::panelRaised); look.drawPopupMenuItem(g, disabled.getBounds(), false, false, false, true, false, "Disabled spelling information", {}, nullptr, nullptr); }
        { juce::Graphics g(hovered); g.fillAll(Palette::panelRaised); look.drawPopupMenuItem(g, hovered.getBounds(), false, false, true, true, false, "Disabled spelling information", {}, nullptr, nullptr); }
        int mutedPixels = 0; bool identical = true;
        for (int y = 0; y < 36; ++y) for (int x = 0; x < 338; ++x)
        { mutedPixels += disabled.getPixelAt(x, y) == Palette::textMuted.withAlpha(1.f); identical &= disabled.getPixelAt(x, y) == hovered.getPixelAt(x, y); }
        check("disabled_uses_opaque_muted_not_white_" + theme, mutedPixels > 5 && Palette::textMuted != juce::Colours::white);
        check("disabled_never_highlights_" + theme, identical);
    }
    Palette::applyTheme(oldTheme, juce::Colour(0xff7f69ca), juce::Colour(0xffcbcbfa), juce::Colour(0xff269b9b));
    return output.getChildFile("engine-view-capabilities.txt").replaceWithText(log) && ok;
}
}
