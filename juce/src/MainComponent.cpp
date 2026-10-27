#include "NativeAudioClipboard.h"
#include "MainComponent.h"
#include "StartupLog.h"
#include "backend/NsfHifiganRenderer.h"
#include "backend/DiffSingerRenderer.h"
#include "OtoWaveformEditorComponent.h"
#include "OttoMaterialIO.h"
#include "OttoMaterialMigration.h"
#include "PronunciationCandidateBox.h"
#include "MaterialPronunciationCatalog.h"
#include "NativeNoteTiming.h"
#include <juce_cryptography/juce_cryptography.h>
#include "AdvancedEnvelopePanel.h"
#include "AccompanimentTempoPanel.h"
#include "UtauOutputEnginePanel.h"
#include "DecibelGainDialog.h"
#include "PanValueDialog.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <thread>
#include <set>

namespace hachi
{
namespace
{
juce::String utf8(const char* text) { return I18n::uiText(text); }
constexpr bool nsfPickerItem(int id) { return id == 2; }

std::optional<double> clipLengthSeconds(const juce::String& input)
{
    const auto text = input.trim();
    if (text.isEmpty() || !text.containsOnly("0123456789.")
        || text.retainCharacters(".").length() > 1
        || text.retainCharacters("0123456789").isEmpty()) return {};
    const auto value = text.getDoubleValue();
    if (!std::isfinite(value) || value < 0.01) return {};
    return value;
}

// Draft controls only. The existing output tool picker edits these paths;
// project setters run only when the enclosing Edit dialog is applied.
class ClipEditDialog final : public juce::AlertWindow
{
public:
    ClipEditDialog(I18n& i18n, const TrackData& track, const ClipData& clip, juce::Component* owner)
        : juce::AlertWindow(i18n.text("clip.editLength"), i18n.text("clip.lengthHint"),
                           juce::MessageBoxIconType::NoIcon, owner),
          strings(i18n), trackId(track.id), trackName(track.name),
          resampler(track.outputResampler), wavtool(track.outputWavtool)
    {
        addTextEditor("length", juce::String(clip.durationSeconds, 6), strings.text("clip.lengthSeconds"));
        addComboBox("output", {strings.text("output.inherit"), juce::String::fromUTF8("HiFisampler（PC-NSF-HiFiGAN）"),
            strings.text("output.resampler")}, strings.text("clip.outputEngine"));
        auto* choice = getComboBoxComponent("output");
        choice->setSelectedId(track.outputEngine == UtauOutputEngine::inherit ? 1 : track.outputEngine == UtauOutputEngine::pcNsfHifigan ? 2 : 3,
                              juce::dontSendNotification);
        canChooseOutput = trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track);
        choice->setEnabled(canChooseOutput);
        configure.setButtonText(strings.text("clip.configureResampler"));
        configure.setSize(300, 30); addCustomComponent(&configure);
        choice->onChange = [this] { configure.setEnabled(canChooseOutput && getComboBoxComponent("output")->getSelectedId() == 3); };
        choice->onChange();
        configure.onClick = [this]
        {
            auto panel = std::make_unique<UtauOutputEnginePanel>(strings, trackName, resampler, wavtool);
            juce::Component::SafePointer<ClipEditDialog> safe(this);
            panel->onApply = [safe](const juce::File& r, const juce::File& w)
            { if (safe) { safe->resampler = r; safe->wavtool = w; } };
            juce::DialogWindow::LaunchOptions options;
            options.dialogTitle = strings.text("output.resampler"); options.dialogBackgroundColour = Palette::panel;
            options.content.setOwned(panel.release()); options.componentToCentreAround = this;
            options.useNativeTitleBar = true; options.escapeKeyTriggersCloseButton = true; options.resizable = true;
            options.launchAsync();
        };
        addButton(strings.text("dialog.apply"), 1, juce::KeyPress(juce::KeyPress::returnKey));
        addButton(strings.text("dialog.cancel"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
        getTextEditor("length")->onTextChange = [this]
        { getButton(0)->setEnabled(clipLengthSeconds(getTextEditorContents("length")).has_value()); };
    }
    UtauOutputEngine selectedEngine() const
    {
        const auto id = getComboBoxComponent("output")->getSelectedId();
        return id == 2 ? UtauOutputEngine::pcNsfHifigan : id == 3 ? UtauOutputEngine::resampler : UtauOutputEngine::inherit;
    }
    I18n& strings;
    juce::String trackId, trackName;
    juce::File resampler, wavtool;
    bool canChooseOutput = false;
    juce::TextButton configure;
};

class EnvelopePresetMenuItem final : public juce::PopupMenu::CustomComponent
{
public:
    explicit EnvelopePresetMenuItem(const PianoRollComponent::EnvelopePreset& value) : preset(value)
    {
        setName(preset.name);
    }
    void getIdealSize(int& width, int& height) override { width=166; height=34; }
    void paint(juce::Graphics& g) override
    {
        const auto highlighted=isItemHighlighted();
        g.fillAll(findColour(highlighted ? juce::PopupMenu::highlightedBackgroundColourId
                                        : juce::PopupMenu::backgroundColourId));
        const juce::Rectangle<float> plot(10.0f,8.0f,66.0f,18.0f);
        using From=PianoRollComponent::EnvelopePresetPoint::From;
        struct Knot { double seconds; bool fromEnd; float gainDb; };
        std::vector<Knot> knots{{0.0,false,-60.0f}};
        for(const auto& point:preset.points)
            knots.push_back({point.from==From::beat ? 0.06+point.seconds : point.seconds,
                             point.from==From::soundEnd,point.gainDb});
        knots.push_back({0.0,true,-60.0f});
        // Retain the old thumbnail proportions: make short ramps legible.
        double rampSeconds=0.0;
        for(size_t i=1;i<knots.size();++i)
            if(knots[i].fromEnd==knots[i-1].fromEnd)
                rampSeconds+=std::abs(knots[i].seconds-knots[i-1].seconds);
        const auto scale=plot.getWidth()*0.62/std::max(1.0e-6,rampSeconds);
        juce::Path shape;
        for(size_t i=0;i<knots.size();++i)
        {
            const auto& knot=knots[i];
            const auto x=static_cast<float>(knot.fromEnd ? plot.getRight()-scale*knot.seconds
                                                       : plot.getX()+scale*knot.seconds);
            const auto level=knot.gainDb<=-59.9f ? 0.0f : std::pow(10.0f,knot.gainDb/20.0f);
            const auto y=plot.getBottom()-plot.getHeight()*level;
            if(i==0)shape.startNewSubPath(x,y);else shape.lineTo(x,y);
        }
        shape.closeSubPath();
        const auto tint=juce::Colour(0xff72d6aa);
        g.setColour(tint.withAlpha(highlighted ? 0.42f : 0.26f));g.fillPath(shape);
        g.setColour(tint);g.strokePath(shape,juce::PathStrokeType(1.5f));
        g.setColour(findColour(highlighted ? juce::PopupMenu::highlightedTextColourId
                                         : juce::PopupMenu::textColourId));
        g.setFont(getLookAndFeel().getPopupMenuFont());
        g.drawText(preset.name,86,0,getWidth()-96,getHeight(),juce::Justification::centredLeft);
    }
private:
    PianoRollComponent::EnvelopePreset preset;
};

class WavExportAdvanced final : public juce::Component
{
public:
    WavExportAdvanced(const I18n& strings, const juce::String& unavailable)
        : caption(strings.text("export.advanced")), supported(unavailable.isEmpty())
    {
        header.setComponentID("export-advanced");
        breath.setComponentID("export-breath");
        nonBreath.setComponentID("export-non-breath");
        breath.setButtonText(strings.text("export.breathOnly"));
        nonBreath.setButtonText(strings.text("export.nonBreathOnly"));
        breath.setEnabled(supported); nonBreath.setEnabled(supported);
        breath.setTooltip(unavailable); nonBreath.setTooltip(unavailable);
        help.setText(supported ? strings.text("export.componentHelp") : unavailable, juce::dontSendNotification);
        help.setFont(juce::FontOptions(13.0f));
        help.setJustificationType(juce::Justification::topLeft);
        addAndMakeVisible(header); addChildComponent(breath); addChildComponent(nonBreath); addChildComponent(help);
        breath.onClick = [this] { if (breath.getToggleState()) nonBreath.setToggleState(false, juce::dontSendNotification); };
        nonBreath.onClick = [this] { if (nonBreath.getToggleState()) breath.setToggleState(false, juce::dontSendNotification); };
        header.onClick = [this]
        {
            expanded = !expanded;
            breath.setVisible(expanded); nonBreath.setVisible(expanded); help.setVisible(expanded);
            updateCaption();
            if (onExpand) onExpand();
        };
        updateCaption(); setSize(430, 32);
    }
    WavExportComponent selected() const
    {
        if (!supported) return WavExportComponent::full;
        return breath.getToggleState() ? WavExportComponent::breath
            : nonBreath.getToggleState() ? WavExportComponent::nonBreath : WavExportComponent::full;
    }
    int preferredHeight() const { return expanded ? 164 : 32; }
    std::function<void()> onExpand;
    void resized() override
    {
        auto bounds = getLocalBounds();
        header.setBounds(bounds.removeFromTop(30)); bounds.removeFromTop(6);
        breath.setBounds(bounds.removeFromTop(29));
        nonBreath.setBounds(bounds.removeFromTop(29)); bounds.removeFromTop(6);
        help.setBounds(bounds);
    }
private:
    void updateCaption()
    {
        auto text = (expanded ? juce::String::fromUTF8("▾ ") : juce::String::fromUTF8("▸ ")) + caption;
        if (!expanded && selected() != WavExportComponent::full)
            text += " · " + (selected() == WavExportComponent::breath ? breath.getButtonText() : nonBreath.getButtonText());
        header.setButtonText(text);
    }
    juce::TextButton header;
    juce::ToggleButton breath, nonBreath;
    juce::Label help;
    juce::String caption;
    bool supported = false, expanded = false;
};

class ComposeTrackSelector final : public juce::Component
{
public:
    ComposeTrackSelector(const std::vector<TrackData>& tracks, const I18n& strings)
    {
        viewport.setViewedComponent(&content, false);
        viewport.setScrollBarsShown(true, false);
        viewport.setScrollBarThickness(9);
        addAndMakeVisible(viewport);
        for (const auto& track : tracks)
        {
            auto toggle = std::make_unique<juce::ToggleButton>();
            toggle->setButtonText(track.name + utf8("  ·  ")
                                  + strings.text(track.compose ? "track.compose" : "track.audio"));
            toggle->setToggleState(track.compose, juce::dontSendNotification);
            content.addAndMakeVisible(*toggle);
            toggles.push_back(std::move(toggle));
        }
        setSize(430, juce::jlimit(56, 320, static_cast<int>(toggles.size()) * rowHeight + 4));
    }

    bool isCompose(std::size_t index) const
    {
        return index < toggles.size() && toggles[index]->getToggleState();
    }

    void resized() override
    {
        viewport.setBounds(getLocalBounds());
        const auto contentHeight = std::max(getHeight(), static_cast<int>(toggles.size()) * rowHeight + 4);
        content.setSize(std::max(1, getWidth() - viewport.getScrollBarThickness()), contentHeight);
        for (std::size_t index = 0; index < toggles.size(); ++index)
            toggles[index]->setBounds(6, 2 + static_cast<int>(index) * rowHeight,
                                      content.getWidth() - 12, rowHeight);
    }

private:
    static constexpr int rowHeight = 28;
    juce::Viewport viewport;
    juce::Component content;
    std::vector<std::unique_ptr<juce::ToggleButton>> toggles;
};
}

juce::Rectangle<int> DropdownButton::textAreaFor(juce::Rectangle<int> bounds)
{
    return bounds.reduced(4, 0).withTrimmedRight(HachiLookAndFeel::dropdownArrowWidth);
}

void DropdownButton::paintButton(juce::Graphics& g, bool highlighted, bool down)
{
    getLookAndFeel().drawButtonBackground(g, *this,
        findColour(juce::TextButton::buttonColourId), highlighted, down);
    g.setColour(findColour(getToggleState() ? juce::TextButton::textColourOnId
                                            : juce::TextButton::textColourOffId)
                    .withAlpha(highlighted || down ? 1.0f : 0.86f));
    g.setFont(getLookAndFeel().getTextButtonFont(*this, getHeight()));
    g.drawText(getButtonText(), textAreaFor(getLocalBounds()),
               juce::Justification::centred, false);
    HachiLookAndFeel::drawDropdownArrow(g, getLocalBounds());
}

struct MainComponent::MaterialInputSession
{
    MaterialInputAction action=MaterialInputAction::insert;
    juce::String noteId,aliasNoteId,trackId,query,bindingKey;
    double atSeconds=0;std::uint64_t serial=0;bool aliasSuspended=false,ready=false,resolving=false;
    std::atomic_bool active{true};
    std::vector<MaterialPronunciationCatalog::Entry> entries;
    juce::Component::SafePointer<PronunciationCandidateBox> picker;
};
struct MainComponent::NativeAnalysisTask
{
    struct Recipient{juce::String clipId;ClipData importedClip;std::optional<ProjectModel::MaterialSourceBindingTicket> ticket;};
    juce::File file;juce::String key,fingerprint;std::vector<Recipient> recipients;
};
namespace
{
juce::String materialBindingKey(const ProjectData& data,const juce::String& id)
{
    juce::String result;int count=0;
    for(const auto& t:data.tracks)for(const auto& parent:t.clips)for(const auto& n:parent.notes)if(n.id==id)
    {
        const ClipData* owner=&parent;
        if(!parent.parts.empty())
        {
            owner=nullptr;for(const auto& p:parent.parts)if(p.id==n.clipPartId){if(owner)return {};owner=&p;}
            if(!owner)return {};
        }
        result=t.id+"|"+parent.id+"|"+owner->id+"|"+owner->sourceFile.getFullPathName()+"|"+owner->materialSourceFingerprint+"|"+n.materialAudioId+"|"+n.materialRegionId;
        if(n.materialAudioId.isNotEmpty())result+="|"+(n.materialBoundAlias.isEmpty()?n.label:n.materialBoundAlias);
        ++count;
    }
    return count==1?result:juce::String{};
}
}

MainComponent::MainComponent()
    : tooltipWindow(this, 450), menuBar(nullptr), progressBar(progress),
      trackList(project, strings), timeline(project), pianoRoll(project, strings)
{
    startupLog("MainComponent: members constructed");
    juce::PropertiesFile::Options options;
    options.applicationName = "HachiShifterNext";
    options.filenameSuffix = "settings";
    options.folderName = juce::SystemStats::getEnvironmentVariable(
        "HACHI_TEST_SETTINGS_DIR", "HachiShifterNext");
    options.osxLibrarySubFolder = "Application Support";
    options.storageFormat = juce::PropertiesFile::storeAsXML;
    preferences = std::make_unique<juce::PropertiesFile>(options);
    startupLog("MainComponent: preferences opened");
    restoreRecentProjects();
    savedProjectRevision = project.revisionNumber();
    audio.restoreDeviceState(*preferences);
    applyPreferences();
    startupLog("MainComponent: preferences applied");
    setLookAndFeel(&lookAndFeel);
    // Give application dialogs and import forms the same palette as the main UI.
    juce::LookAndFeel::setDefaultLookAndFeel(&lookAndFeel);
    setOpaque(true);
    setWantsKeyboardFocus(true);

    for (auto* button : std::initializer_list<juce::TextButton*>{
                          &playButton, &stopButton, &noteEditButton, &wrenchButton,
                          &lineButton, &pointButton, &trimButton, &nativeLoudnessButton, &connectButton, &pitchParamButton,
                          &driftParamButton, &attackParamButton,
                          &breathParamButton, &tensionParamButton, &formantParamButton,
                          &volumeParamButton,
                          &spliceButton, &utauVoicebankButton, &flagCurveButton, &flagEnvelopeButton })
        addAndMakeVisible(*button);
    addAndMakeVisible(phonemizerButton);
    phonemizerButton.onClick = [this] { showPhonemizerMenu(); };
    addAndMakeVisible(pianoPreviewButton);
    pianoPreviewButton.onClick = [this]
    { project.setTrackPianoPreview(selectedTrackId, pianoPreviewButton.getToggleState()); };
    utauVoicebankButton.setComponentID("utau-choose-voicebank");
    utauVoicebankButton.onClick = [this] { chooseUtauVoicebank(); };
    addAndMakeVisible(utauVoicebankName);
    utauVoicebankName.setComponentID("utau-voicebank-name");
    utauVoicebankName.setFont(juce::Font(14.0f,juce::Font::bold));
    utauVoicebankName.setColour(juce::Label::textColourId,Palette::accentLight);
    utauVoicebankName.setMinimumHorizontalScale(1.0f);
    utauVoicebankName.setJustificationType(juce::Justification::centredLeft);
    utauVoicebankName.addMouseListener(this, false);
    utauVoicebankName.setMouseCursor(juce::MouseCursor::PointingHandCursor);
    addAndMakeVisible(nativeAudioOverlapButton);
    addAndMakeVisible(nsfPitchTransitionsButton);
    addAndMakeVisible(nsfNoiseProtectionButton);
    addAndMakeVisible(nativeVibratoButton);
    addAndMakeVisible(advancedEnvelopeButton);
    addAndMakeVisible(drawButton);
    for (auto* component : { static_cast<juce::Component*>(&menuBar),
                             static_cast<juce::Component*>(&bpmCaption),
                             static_cast<juce::Component*>(&bpmEditor),
                             static_cast<juce::Component*>(&beatsCaption),
                             static_cast<juce::Component*>(&beatsEditor),
                             static_cast<juce::Component*>(&denominatorLabel),
                             static_cast<juce::Component*>(&gridCaption),
                             static_cast<juce::Component*>(&gridSelector),
                             static_cast<juce::Component*>(&stretchCaption),
                             static_cast<juce::Component*>(&stretchSelector),
                             static_cast<juce::Component*>(&scaleCaption),
                             static_cast<juce::Component*>(&scaleSelector),
                             static_cast<juce::Component*>(&pitchAlgorithm),
                             static_cast<juce::Component*>(&stretchAlgorithm),
                             static_cast<juce::Component*>(&renderOrder),
                             static_cast<juce::Component*>(&pitchLabel),
                             static_cast<juce::Component*>(&stretchLabel),
                             static_cast<juce::Component*>(&renderOrderLabel),
                             static_cast<juce::Component*>(&statusLabel),
                             static_cast<juce::Component*>(&sourceEditHint),
                             static_cast<juce::Component*>(&sampleRegionSelector),
                             static_cast<juce::Component*>(&sampleAliasEditor),
                             static_cast<juce::Component*>(&sampleStartEditor),
                             static_cast<juce::Component*>(&sampleEndEditor),
                             static_cast<juce::Component*>(&sampleAlignmentEditor),
                             static_cast<juce::Component*>(&sampleFixedEditor),
                             static_cast<juce::Component*>(&sampleAliasLabel),
                             static_cast<juce::Component*>(&sampleStartLabel),
                             static_cast<juce::Component*>(&sampleEndLabel),
                             static_cast<juce::Component*>(&sampleAlignmentLabel),
                             static_cast<juce::Component*>(&sampleFixedLabel),
                             static_cast<juce::Component*>(&sampleSaveButton),
                             static_cast<juce::Component*>(&otoImportButton),
                             static_cast<juce::Component*>(&otoExportButton),
                             static_cast<juce::Component*>(&noteAliasLabel),
                             static_cast<juce::Component*>(&noteAliasEditor),
                             static_cast<juce::Component*>(&noteConsonantVelocityLabel),
                             static_cast<juce::Component*>(&noteConsonantVelocityEditor),
                             static_cast<juce::Component*>(&noteFlagsLabel),
                             static_cast<juce::Component*>(&noteFlagsEditor),
                             static_cast<juce::Component*>(&parameterTitle),
                             static_cast<juce::Component*>(&smoothCaption),
                             static_cast<juce::Component*>(&smoothSlider),
                             static_cast<juce::Component*>(&robustPitchCurveButton),
                             static_cast<juce::Component*>(&progressBar),
                             static_cast<juce::Component*>(&collapseTracksButton),
                             static_cast<juce::Component*>(&trackViewport),
                             static_cast<juce::Component*>(&timelineViewport),
                             static_cast<juce::Component*>(&pianoViewport),
                             static_cast<juce::Component*>(&panelSplitter) })
        addAndMakeVisible(*component);

    panelSplitter.setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
    panelSplitter.addMouseListener(this, false);

    for (auto* editor : { &sampleAliasEditor, &sampleStartEditor, &sampleEndEditor,
                          &sampleAlignmentEditor, &sampleFixedEditor })
        editor->setSelectAllWhenFocused(true);
    for (auto* editor : { &sampleStartEditor, &sampleEndEditor,
                          &sampleAlignmentEditor, &sampleFixedEditor })
        editor->setInputRestrictions(14, "0123456789.-");
    sampleRegionSelector.onChange = [this]
    {
        commitSampleEditors();
        activeSampleSetting = std::max(0, sampleRegionSelector.getSelectedItemIndex());
        refreshSampleEditors();
    };
    const auto commit = [this] { commitSampleEditors(); };
    sampleAliasEditor.onFocusLost = commit;
    sampleStartEditor.onFocusLost = commit;
    sampleEndEditor.onFocusLost = commit;
    sampleAlignmentEditor.onFocusLost = commit;
    sampleFixedEditor.onFocusLost = commit;
    sampleSaveButton.onClick = [this] { saveSampleSettings(); };
    otoImportButton.onClick = [this] { importOto(); };
    otoExportButton.onClick = [this] { exportOto(); };
    noteAliasEditor.setSelectAllWhenFocused(true);
    noteAliasEditor.onTextChange = [this]
    {
        // Typing saves, like the Flags and consonant-velocity fields beside it.
        // Only the value is written here; switching the track to UTAU, binding a
        // default voicebank and refreshing the layout stay on focus loss and
        // Return so they do not run once per keystroke.
        if (selectedNoteId.isEmpty() || !noteAliasEditor.isEnabled()) return;
        project.setNoteLabel(selectedNoteId, noteAliasEditor.getText());
        pianoRoll.ensureDefaultEnvelope(selectedNoteId);
    };
    noteAliasEditor.onFocusLost = [this] { commitNoteAlias(); };
    noteAliasEditor.onReturnKey = [this] { commitNoteAlias(); };
    noteFlagsEditor.setSelectAllWhenFocused(true);
    noteFlagsEditor.onTextChange = [this]
    {
        noteFlagsDirty = true;
        commitNoteFlags();
    };
    noteFlagsEditor.onFocusLost = [this] { commitNoteFlags(); };
    noteFlagsEditor.onReturnKey = [this] { commitNoteFlags(); };
    noteConsonantVelocityEditor.setSelectAllWhenFocused(true);
    noteConsonantVelocityEditor.setInputRestrictions(0, "-0123456789");
    noteConsonantVelocityEditor.setJustification(juce::Justification::centred);
    noteConsonantVelocityEditor.onTextChange = [this]
    {
        noteConsonantVelocityDirty = true;
        commitNoteConsonantVelocity();
    };
    noteConsonantVelocityEditor.onFocusLost = [this]
    {
        commitNoteConsonantVelocity();
        refreshSelectedNoteParameter();
    };
    noteConsonantVelocityEditor.onReturnKey = [this] { commitNoteConsonantVelocity(); };
    pianoRoll.onSampleRegionEdited = [this](int index, const SampleRegionSetting& row, bool)
    {
        if (index < 0 || index >= static_cast<int>(sampleSettingsRows.size())) return;
        activeSampleSetting = index;
        sampleSettingsRows[static_cast<std::size_t>(index)] = row;
        sampleRegionSelector.setSelectedItemIndex(index, juce::dontSendNotification);
        refreshSampleEditors();
    };

    refreshCollapseIcon();
    playButton.setComponentID("icon.play");
    stopButton.setComponentID("icon.stop");
    noteEditButton.setComponentID("icon.pointer");
    drawButton.setComponentID("icon.draw");
    lineButton.setComponentID("icon.line");
    pointButton.setComponentID("icon.points");
    wrenchButton.setComponentID("icon.wrench");
    trimButton.setComponentID("icon.trim");
    nativeLoudnessButton.setComponentID("tool.nativeLoudness");
    connectButton.setComponentID("icon.connect");

    bpmEditor.setEditable(true, false, false);
    beatsEditor.setEditable(true, false, false);
    for (auto* editor : { &bpmEditor, &beatsEditor })
    {
        editor->setJustificationType(juce::Justification::centred);
        editor->setColour(juce::Label::backgroundColourId, Palette::background);
        editor->setColour(juce::Label::outlineColourId, Palette::grid);
    }
    bpmEditor.onTextChange = [this]
    {
        const auto value = bpmEditor.getText().getDoubleValue();
        if (value >= 20.0 && value <= 400.0)
        {
            const auto data = project.snapshot();
            project.setTempo(value, data.numerator, data.denominator);
        }
    };
    beatsEditor.onTextChange = [this]
    {
        const auto value = beatsEditor.getText().getIntValue();
        if (value >= 1 && value <= 32)
        {
            const auto data = project.snapshot();
            project.setTempo(data.bpm, value, data.denominator);
        }
    };

    for (const auto& value : { "1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64",
                               "1/4.", "1/8.", "1/16.", "1/4t", "1/8t", "1/16t" })
        gridSelector.addItem(value, gridSelector.getNumItems() + 1);
    for (const auto& value : { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" })
        scaleSelector.addItem(value, scaleSelector.getNumItems() + 1);
    // Item ids are the denominators themselves, so the id is the setting.
    for (const auto value : { 2, 4, 8, 16, 32, 64, 128 })
        stretchSelector.addItem("1/" + juce::String(value), value);
    gridSelector.onChange = [this] { project.setGridDivision(gridSelector.getText()); };
    stretchSelector.onChange = [this]
    {
        project.setNoteEditDivision(stretchSelector.getSelectedId());
    };
    scaleSelector.onChange = [this] { project.setBaseScale(scaleSelector.getText()); };

    trackViewport.setViewedComponent(&trackList, false);
    trackViewport.setScrollBarsShown(true, false);
    trackViewport.setScrollBarThickness(10);
    timelineViewport.setViewedComponent(&timeline, false);
    timelineViewport.enableZoomHandles();
    timelineViewport.setScrollBarsShown(true, true);
    timelineViewport.setScrollBarThickness(16);
    pianoViewport.setViewedComponent(&pianoRoll, false);
    pianoViewport.enableZoomHandles();
    pianoViewport.setScrollBarsShown(true, true);
    pianoViewport.setScrollBarThickness(16);
    // The wheel scrolls and the modifiers zoom, which is the way round every
    // other editor of this kind works:
    //
    //     wheel              scroll up and down
    //     shift + wheel      scroll back and forth
    //     ctrl + wheel       zoom vertically
    //     ctrl + shift       zoom horizontally
    //
    // Plain scrolling is left to the Viewport itself rather than reimplemented,
    // and the sideways one is handed to the horizontal scrollbar with the
    // wheel's delta moved onto the axis that bar reads -- so both directions
    // move by the same amount per notch instead of by two hand-picked ones.
    const auto wheelFor = [this](EditorViewport& viewport)
    {
        return [this, &viewport](const juce::MouseEvent& event,
                                 const juce::MouseWheelDetails& wheel) -> bool
        {
            auto delta = wheel.deltaY;
            if (wheel.isReversed) delta = -delta;
            const auto factor = std::exp(static_cast<double>(delta) * 1.35);
            switch (wheelActionFor(event.mods))
            {
                case WheelAction::zoomHorizontally:
                    adjustHorizontalZoom(factor, &viewport != &pianoViewport);
                    return true;
                case WheelAction::zoomVertically:
                    adjustVerticalZoom(factor, &viewport != &pianoViewport);
                    return true;
                case WheelAction::scrollHorizontally:
                {
                    if (&viewport == &trackViewport) return false;
                    auto sideways = wheel;
                    sideways.deltaX = wheel.deltaY;
                    sideways.deltaY = 0.0f;
                    viewport.getHorizontalScrollBar().mouseWheelMove(event, sideways);
                    return true;
                }
                case WheelAction::scrollVertically:
                    break;
            }
            // Unhandled, so the Viewport scrolls the way it always has.
            return false;
        };
    };
    timelineViewport.onWheel = wheelFor(timelineViewport);
    trackViewport.onWheel = wheelFor(trackViewport);
    pianoViewport.onWheel = wheelFor(pianoViewport);

    pitchAlgorithm.addItem("nsf-hifigan", 2);
    pitchAlgorithm.addItem("WORLD", 3);
    pitchAlgorithm.addItem("llsm2", 6);
    pitchAlgorithm.addItem("UTAU", 7);
    // The other UTAU modes.  Same algorithm as item 7 plus TrackData::utauMode,
    // so every UTAU affordance is inherited rather than reimplemented.
    pitchAlgorithm.addItem(utf8("界•UTAU"), 8);
    pitchAlgorithm.addItem(utf8("谋•UTAU"), 9);
    pitchAlgorithm.addItem("DiffSinger (DS)", 10);
    const auto defaultPitchId = 2; // Native default is NSF; readiness is diagnosed when used.
    pitchAlgorithm.setSelectedId(defaultPitchId,juce::dontSendNotification);
    refreshStretchAlgorithmItems(defaultPitchId);
    pitchAlgorithm.onChange = [this]
    {
        auto id = pitchAlgorithm.getSelectedId();
        if (nsfPickerItem(id))
        {
            const auto configured = preferences != nullptr
                ? juce::File(preferences->getValue("algorithm.hifiganPath"))
                : juce::File{};
            if (!backend::NsfHifiganRenderer::modelAvailable(configured))
            {
                showError(utf8("NSF-HiFiGAN 模型未找到，未更改当前算法。\n"
                               "请将 pc_nsf_hifigan.onnx 和 config.json 放入软件目录下的\n"
                               "models/nsf_hifigan 文件夹，或在算法设置中填写模型目录。"));
                refreshProjectControls();
                return;
            }
        }
        const auto mode = utauModeForPickerItem(id);
        if (sourceEditActive || diffSingerBusy || !projectedTrackEditable
            || (id != 2 && id != 3 && id != 6 && !mode && id != 10))
        { refreshProjectControls(); return; }
        const auto type = id == 10 ? TrackType::diffSinger : mode ? TrackType::utau : TrackType::hachi;
        const auto algorithm = id == 2 ? PitchAlgorithm::nsfHifigan : id == 3 ? PitchAlgorithm::world
            : id == 6 ? PitchAlgorithm::llsm2 : PitchAlgorithm::utau;
        project.setTrackEditingType(selectedTrackId, type, algorithm, mode);
        project.dispatchPendingMessages();
        refreshProjectControls();
        refreshSelectedNoteParameter();
        resized();
    };
    renderOrder.onChange = [this]
    {
        const auto order = renderOrder.getSelectedId() == 2
            ? RenderOrder::stretchSpliceThenPitch : RenderOrder::processThenSplice;
        const auto data = project.snapshot();
        const auto selectedTrack = std::find_if(data.tracks.begin(), data.tracks.end(),
            [this](const auto& track)
            {
                return track.id == selectedTrackId;
            });
        if (selectedTrack != data.tracks.end())
            project.setTrackRenderOrder(selectedTrack->id, order, renderOrder.getSelectedId() == 3);
        else
            project.setRenderOrder(order);
    };
    stretchAlgorithm.onChange = [this]
    {
        const auto id = stretchAlgorithm.getSelectedId();
        const auto algorithm = id == 2 ? StretchAlgorithm::variableMelHop
            : id == 3 ? StretchAlgorithm::loop
            : id == 4 ? StretchAlgorithm::soundTouch
            : id == 7 ? StretchAlgorithm::wcsNonLoop
            : id == 6 ? StretchAlgorithm::hifiShifterMel
            : id == 5 ? StretchAlgorithm::nsfShiftThenSplice
            : StretchAlgorithm::melodyneHybrid;
        const auto data = project.snapshot();
        const auto selectedTrack = std::find_if(data.tracks.begin(), data.tracks.end(),
            [this](const auto& track)
            {
                return track.id == selectedTrackId;
            });
        if (selectedTrack != data.tracks.end())
            project.setTrackStretchAlgorithm(selectedTrack->id, algorithm);
        else
            project.setStretchAlgorithm(algorithm);
    };

    smoothSlider.setRange(0.0, 100.0, 1.0);
    smoothSlider.setValue(0.0);
    smoothSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    smoothSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 18);
    smoothSlider.onValueChange = [this]
    {
        if (updatingSmoothSlider || smoothSliderDragging || selectedNoteId.isEmpty()
            || !smoothSlider.isEnabled()) return;
        applySelectedNoteParameter();
    };
    smoothSlider.onDragStart = [this] { smoothSliderDragging = true; };
    smoothSlider.onDragEnd = [this]
    {
        smoothSliderDragging = false;
        if (selectedNoteId.isNotEmpty() && smoothSlider.isEnabled())
            applySelectedNoteParameter();
    };
    robustPitchCurveButton.onClick = [this]
    {
        if (updatingRobustPitchCurve || selectedNoteId.isEmpty()
            || !robustPitchCurveButton.isEnabled()) return;
        project.setNoteRobustPitchCurve(selectedNoteId,
            robustPitchCurveButton.getToggleState());
    };

    // Far enough in that a millisecond is eight pixels wide, which is the
    // smallest gap two pitch points are allowed to keep -- so the view can now
    // resolve the finest edit the model permits, rather than stopping at a
    // little over half a pixel per millisecond.
    zoomSlider.setRange(40.0, 8000.0, 1.0);
    // Without a skew the first tenth of the slider would hold everything from
    // reading a whole phrase to ordinary note editing.
    zoomSlider.setSkewFactorFromMidPoint(280.0);
    zoomSlider.setValue(140.0);
    zoomSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    zoomSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    zoomSlider.onValueChange = [this]
    {
        const auto zoom = static_cast<float>(zoomSlider.getValue());
        // What the zoom should hold still, decided before the scale changes.
        // The red playhead is the editing focus and stays centred whenever it
        // is actually in view; when it is not, centring on it threw the view
        // across the piece for no reason the user could see, so what is on
        // screen is held instead.  This callback also covers the toolbar
        // slider, mouse-wheel zoom, menu commands and the +/- buttons.
        const auto viewWidth = pianoViewport.getViewWidth();
        const auto viewLeft = pianoViewport.getViewPositionX();
        const auto playheadPixel = pianoRoll.pixelForSeconds(audio.position());
        const auto playheadInView = playheadPixel >= viewLeft
            && playheadPixel < viewLeft + viewWidth;
        const auto anchorSeconds = playheadInView
            ? audio.position()
            : pianoRoll.secondsForPixel(viewLeft + viewWidth / 2);
        pianoRoll.setPixelsPerSecond(zoom);
        if (viewWidth > 0)
        {
            const auto nextViewX = std::max(0,
                pianoRoll.pixelForSeconds(anchorSeconds) - viewWidth / 2);
            pianoViewport.setViewPosition(nextViewX, pianoViewport.getViewPositionY());
            // A zoom changes the viewport origin, but is not a request to scroll the other pane.
            lastPianoX = pianoViewport.getViewPositionX();
            lastTimelineX = timelineViewport.getViewPositionX();
        }
    };

    vZoomSlider.setRange(0.5, 2.0, 0.05);
    vZoomSlider.setValue(1.0);
    vZoomSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    vZoomSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    vZoomSlider.onValueChange = [this]
    {
        const auto factor = static_cast<float>(vZoomSlider.getValue());
        const auto pianoRow = 22.0f * factor;
        pianoRoll.setRowHeight(pianoRow);
        if (preferences != nullptr)
            preferences->setValue("ui.vZoom", vZoomSlider.getValue());
    };
    timelineViewport.onZoomBegin = [this](bool vertical, bool leading)
    {
        auto& anchor = timelineScrollZoomAnchor;
        anchor.leading = leading;
        anchor.zoom = vertical ? timelineVerticalZoom : timelineHorizontalZoom;
        if (vertical)
        {
            const auto y = timelineViewport.getViewPositionY()
                + (leading ? timelineViewport.getViewHeight() : 0);
            anchor.anchor = juce::jmax(0.0, (y - timeline.getRulerHeight())
                / static_cast<double>(timeline.getRowHeight()));
            // Keep the fixed-height ruler in place when resizing from the origin.
            anchor.pixelOffset = anchor.anchor * timeline.getRowHeight()
                + timeline.getRulerHeight() - timelineViewport.getViewPositionY();
        }
        else
        {
            const auto x = timelineViewport.getViewPositionX()
                + (leading ? timelineViewport.getViewWidth() : 0);
            anchor.anchor = timeline.secondsForPixel(x);
        }
    };
    timelineViewport.onZoomDrag = [this](bool vertical, double scale)
    {
        const auto& anchor = timelineScrollZoomAnchor;
        if (vertical)
        {
            timelineVerticalZoom = juce::jlimit(0.5, 2.0, std::round(anchor.zoom * scale * 20.0) / 20.0);
            timeline.setRowHeight(static_cast<float>(96.0 * timelineVerticalZoom));
            trackList.setRowHeight(static_cast<float>(96.0 * timelineVerticalZoom));
            const auto y = juce::roundToInt(anchor.anchor * timeline.getRowHeight()
                + timeline.getRulerHeight() - (anchor.leading ? timelineViewport.getViewHeight() : anchor.pixelOffset));
            timelineViewport.setViewPosition(timelineViewport.getViewPositionX(), juce::jmax(0, y));
            trackViewport.setViewPosition(0, timelineViewport.getViewPositionY());
            lastTimelineY = timelineViewport.getViewPositionY();
            lastTrackY = trackViewport.getViewPositionY();
            if (preferences != nullptr) preferences->setValue("ui.timelineVZoom", timelineVerticalZoom);
        }
        else
        {
            timelineHorizontalZoom = juce::jlimit(40.0, 8000.0, std::round(anchor.zoom * scale));
            timeline.setPixelsPerSecond(static_cast<float>(timelineHorizontalZoom));
            const auto x = timeline.pixelForSeconds(anchor.anchor)
                - (anchor.leading ? timelineViewport.getViewWidth() : 0);
            timelineViewport.setViewPosition(juce::jmax(0, x), timelineViewport.getViewPositionY());
        }
        // A zoom-induced position change must not be treated as a shared scroll.
        lastTimelineX = timelineViewport.getViewPositionX();
        lastPianoX = pianoViewport.getViewPositionX();
    };
    pianoViewport.onZoomBegin = [this](bool vertical, bool leading)
    {
        scrollZoomAnchor.leading = leading;
        scrollZoomAnchor.zoom = vertical ? vZoomSlider.getValue() : zoomSlider.getValue();
        if (vertical)
        {
            const auto y = pianoViewport.getViewPositionY() + (leading ? pianoViewport.getViewHeight() : 0);
            scrollZoomAnchor.anchor = y / juce::jlimit(12.0, 48.0, 22.0 * scrollZoomAnchor.zoom);
        }
        else
        {
            const auto x = pianoViewport.getViewPositionX() + (leading ? pianoViewport.getViewWidth() : 0);
            scrollZoomAnchor.anchor = pianoRoll.secondsForPixel(x);
            // At the origin, the fixed keyboard gutter precedes time zero.
            scrollZoomAnchor.pixelOffset = pianoRoll.pixelForSeconds(scrollZoomAnchor.anchor)
                - pianoViewport.getViewPositionX();
        }
    };
    pianoViewport.onZoomDrag = [this](bool vertical, double scale)
    {
        if (vertical)
        {
            vZoomSlider.setValue(scrollZoomAnchor.zoom * scale, juce::sendNotificationSync);
            const auto row = juce::jlimit(12.0, 48.0, 22.0 * vZoomSlider.getValue());
            const auto y = juce::roundToInt(scrollZoomAnchor.anchor * row)
                - (scrollZoomAnchor.leading ? pianoViewport.getViewHeight() : 0);
            pianoViewport.setViewPosition(pianoViewport.getViewPositionX(), juce::jmax(0, y));
        }
        else
        {
            zoomSlider.setValue(scrollZoomAnchor.zoom * scale, juce::sendNotificationSync);
            const auto x = pianoRoll.pixelForSeconds(scrollZoomAnchor.anchor)
                - (scrollZoomAnchor.leading ? pianoViewport.getViewWidth()
                    : juce::roundToInt(scrollZoomAnchor.pixelOffset));
            pianoViewport.setViewPosition(juce::jmax(0, x), pianoViewport.getViewPositionY());
            lastPianoX = pianoViewport.getViewPositionX();
            lastTimelineX = timelineViewport.getViewPositionX();
        }
    };
    timeline.onSeek = [this](double seconds) { audio.setPosition(seconds); };
    // The same menu from either half of the track area: the headers on the
    // left and the lanes on the right are one surface to the person using it.
    const auto trackAreaMenu = [this](juce::Point<int> at) { showTrackAreaMenu(at); };
    timeline.onEmptyAreaMenu = [this](juce::Point<int> at) { showTrackAreaMenu(at,timeline.pointerAnchor()); };
    trackList.onEmptyAreaMenu = trackAreaMenu;
    timeline.onClipSelected = [this](const juce::String& clipId)
    {
        if (selectedClipId != clipId) focusClip(clipId);
        // A region click takes display focus from the last edited note. Keep
        // other regions visible, but don't let an old note dim the clicked one.
        const auto data=project.snapshot();
        for(const auto& track:data.tracks)if(trackShowsAllNativeRegions(track))
            if(std::any_of(track.clips.begin(),track.clips.end(),[&](const auto& c){return c.id==clipId;}))
            {pianoRoll.setSelectedNoteIds({});break;}
    };
    timeline.onTracksSelected = [this](const auto& ids) { trackList.setHighlightedTracks(ids); };
    timeline.onDeleteSelected = [this] { deleteSelectedClip(); };
    timeline.onClipMenu = [this](const juce::String& clipId, double seconds, juce::Point<int> at)
    {
        juce::Component::SafePointer<MainComponent> safe(this);
        clipContextMenu(clipId, seconds).showMenuAsync(
            juce::PopupMenu::Options().withTargetScreenArea(juce::Rectangle<int>(at.x, at.y, 1, 1)),
            [safe, clipId, seconds](int result)
            {
                if (safe != nullptr) safe->clipContextMenuItemChosen(result, clipId, seconds);
            });
    };
    timeline.onClipGainRequested = [this](const juce::String& clipId)
    {
        selectedClipId = clipId;
        showClipGainDialog();
    };
    pianoRoll.onSeek = [this](double seconds) { audio.setPosition(seconds); };
    pianoRoll.onLoadVibratoPresets = [this]
    {
        return preferences != nullptr
            ? preferences->getValue("ui.vibratoPresets") : juce::String();
    };
    pianoRoll.onSaveVibratoPresets = [this](const juce::String& text)
    {
        if (preferences != nullptr) preferences->setValue("ui.vibratoPresets", text);
    };
    pianoRoll.onEditPosition = [this](double seconds)
    {
        // Following the edit must never derail playback: while something is
        // playing the playhead belongs to the transport, not to the cursor.
        if (!audio.isPlaying()) audio.setPosition(seconds);
    };
    for(auto* c:{static_cast<juce::Component*>(&trackEditingHeading),static_cast<juce::Component*>(&diffSingerContextLabel),static_cast<juce::Component*>(&diffSingerModelButton),
        static_cast<juce::Component*>(&diffSingerTimingButton),static_cast<juce::Component*>(&diffSingerPronunciationButton),static_cast<juce::Component*>(&diffSingerPitchButton),
        static_cast<juce::Component*>(&diffSingerParametersButton),static_cast<juce::Component*>(&diffSingerSettingsButton)})addChildComponent(c);
    trackEditingHeading.setFont(juce::Font(14.0f,juce::Font::bold));trackEditingHeading.setColour(juce::Label::textColourId,Palette::accentLight);
    trackEditingHeading.setComponentID("track-editing-interface");diffSingerContextLabel.setComponentID("diffsinger-track-properties");
    trackEditingHeading.setInterceptsMouseClicks(false, false);
    pitchAlgorithm.setComponentID("track-editor-algorithm");
    diffSingerModelButton.setComponentID("diffsinger-choose-model");diffSingerTimingButton.setComponentID("diffsinger-phoneme-timing");
    diffSingerPronunciationButton.setComponentID("diffsinger-pronunciation");diffSingerPitchButton.setComponentID("diffsinger-generate-pitch");
    diffSingerParametersButton.setComponentID("diffsinger-variance-parameters");diffSingerSettingsButton.setComponentID("diffsinger-language-speaker");
    diffSingerModelButton.onClick=[this]{showAssetManager();};
    diffSingerTimingButton.onClick=[this]{showDiffSingerPhonemeEditor(selectedTrackId,selectedNoteId);};
    diffSingerPronunciationButton.onClick=[this]{showDiffSingerPronunciationEditor(selectedNoteId);};
    diffSingerPitchButton.onClick=[this]{generateDiffSingerPitch(false,selectedNoteId);};
    diffSingerParametersButton.onClick=[this]{showDiffSingerParameterLane();};
    diffSingerSettingsButton.onClick=[this]{for(const auto& t:project.snapshot().tracks)if(t.id==selectedTrackId){showDiffSingerSettings(t);break;}};
    addAndMakeVisible(selectionSummaryLabel);
    selectionSummaryLabel.setJustificationType(juce::Justification::centredRight);
    pianoRoll.onEditRejected=[this](const juce::String& message){statusLabel.setText(message,juce::dontSendNotification);};
    pianoRoll.onNoteSelected = [this](const juce::String& noteId)
    {
        focusNote(noteId);
        timeline.setNativeFocusedNote(noteId);
        updateUtauRenderSelection();
        // Splicing needs a boundary, so the button follows the selection.
        refreshSpliceButton();
    };
    pianoRoll.onOpenRegionEditor = [this](const juce::String& noteId)
    {
        showRegionEditorForNote(noteId);
    };
    pianoRoll.onDiffSingerPronunciation = [this](const juce::String& id) { showDiffSingerPronunciationEditor(id); };
    pianoRoll.onDiffSingerPitch = [this](const juce::String& id) { generateDiffSingerPitch(false, id); };
    pianoRoll.onDiffSingerParameters = [this](const juce::String& id) { generateDiffSingerParameters(false, id); };
    pianoRoll.onOpenNoteOtoEditor = [this](const juce::String& noteId)
    {
        showNoteOtoEditorForNote(noteId);
    };
    pianoRoll.onRequestMaterialSourceInput = [this](const juce::String& noteId,const juce::String& query)
    {showMaterialPronunciationInput(MaterialInputAction::replace,noteId,query);};
    pianoRoll.onNoteAliasCommitted = [this](const juce::String& noteId)
    {
        noteLabelCommitted(noteId);
    };
    trackList.onTrackMenu = [this](const juce::String& id, juce::Point<int> at) { showOutputEngineMenu(id, at); };
    timeline.outputEngineNameProvider = [this](const auto& track) { return audio.outputEngineDisplayName(track); };
    trackList.outputEngineNameProvider = timeline.outputEngineNameProvider;
    trackList.peakProvider = [this](const juce::String& trackId) { return audio.trackPeak(trackId); };
    trackList.onTrackGainRequested = [this](const juce::String& id) { showTrackGainDialog(id); };
    trackList.onTrackPanRequested = [this](const juce::String& id) { showTrackPanDialog(id); };
    trackList.onTrackSelected = [this](const juce::String& trackId)
    {
        timeline.setSelectedClips({});
        if (selectedTrackId != trackId) { selectedClipId.clear(); selectedNoteId.clear(); }
        selectedTrackId = trackId;
        pianoRoll.setFocusedTrack(trackId);
        refreshProjectControls();
    };

    collapseTracksButton.onClick = [this] { setTracksCollapsed(!tracksCollapsed); };
    playButton.onClick = [this] { togglePlayback(); };
    stopButton.onClick = [this]
    {
        playWhenRenderReady = false;
        audio.stop();
        audio.setPosition(0.0);
    };
    noteEditButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::note);
        setToolButton(noteEditButton);
    };
    wrenchButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        openSelectedMaterialEditor(true);
    };
    drawButton.onSecondaryClick = [this](juce::Point<int> at)
    { showDrawSettingsMenu(at); };
    drawButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::draw);
        setToolButton(drawButton);
    };
    lineButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::line);
        setToolButton(lineButton);
    };
    pointButton.onSecondaryClick = [this](juce::Point<int> at) { showPitchPointModeMenu(at); };
    pianoRoll.onPitchPointModeChanged = [this]
    {
        if (pianoRoll.diffSingerPitchOffsetMode())
        {
            const auto help=utf8("DS 绘制偏移：拖动线整体平移，双击加点，拖动点调整，Ctrl 拖动自由绘制；右键可增加标点、输入半音或归零，Esc 取消/退出");
            pointButton.setTooltip(help);statusLabel.setText(help,juce::dontSendNotification);return;
        }
        const auto restore = pianoRoll.diffSingerPitchRestoreMode();
        pointButton.setTooltip(restore ? utf8("DS 还原模式：点击标点恢复原始音高；Esc 退出")
            : strings.text("tool.points") + utf8("；右键可选择 DS 音高还原或绘制偏移"));
        statusLabel.setText(restore ? utf8("DS 还原模式：点击标点恢复初始音高线；Esc 退出")
                                   : utf8("普通标点编辑模式"), juce::dontSendNotification);
    };
    pointButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::points);
        setToolButton(pointButton);
    };
    trimButton.onClick = [this]
    {
        if (!trimButton.isEnabled()) return;
        const auto active = pianoRoll.currentTool() == PianoRollComponent::Tool::trim;
        closeEnvelopeLanes(); setSourceEditMode(false);
        pianoRoll.setTool(active ? PianoRollComponent::Tool::note : PianoRollComponent::Tool::trim);
        setToolButton(active ? noteEditButton : trimButton);
    };
    nativeLoudnessButton.onClick = [this]
    {
        if (!nativeLoudnessButton.isEnabled()) return;
        const auto active = nativeLoudnessButton.getToggleState();
        closeEnvelopeLanes(); setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::note);
        setToolButton(active ? noteEditButton : nativeLoudnessButton);
        if (!active) statusLabel.setText(strings.text("tool.nativeLoudnessHint"), juce::dontSendNotification);
    };
    connectButton.onClick = [this]
    {
        closeEnvelopeLanes();
        setSourceEditMode(false);
        pianoRoll.setTool(PianoRollComponent::Tool::connect);
        setToolButton(connectButton);
    };
    for (auto* button : std::initializer_list<juce::TextButton*>{ &pitchParamButton, &driftParamButton, &attackParamButton,
                          &breathParamButton, &tensionParamButton,
                          &formantParamButton, &volumeParamButton })
        button->onClick = [this, button]
        {
            if (button == &volumeParamButton && (isUtauAlgorithmSelected()
                || displayedTrackType == TrackType::diffSinger))
            {
                setEnvelopeLane(nextEnvelopeLane(envelopeLane, EnvelopeLane::amplitude));
                return;
            }
            // Any other parameter lane closes both envelope lanes with it.
            closeEnvelopeLanes();
            setToolButton(*button);
            parameterMode = button == &driftParamButton ? ParameterMode::pitchDrift
                : button == &attackParamButton ? ParameterMode::attackSpeed
                : button == &breathParamButton ? ParameterMode::breath
                : button == &tensionParamButton ? ParameterMode::tension
                : button == &formantParamButton ? ParameterMode::formant
                : button == &volumeParamButton ? ParameterMode::volume
                : ParameterMode::pitchSmooth;
            refreshSelectedNoteParameter();
        };
    noteEditButton.setClickingTogglesState(false);
    wrenchButton.setClickingTogglesState(false);

    project.addChangeListener(this);
    audio.addChangeListener(this);
    // Opening a project read its whole voicebank before the window could draw
    // again: 0.8 s for a thousand-sample bank, several on a cold disk.
    pianoRoll.setReadsVoicebankInBackground(true);
    // MenuBarComponent caches its labels when the model is attached. Attach
    // only after applyPreferences() restores the saved application language.
    menuBar.setModel(this);
    refreshTexts();
    refreshProjectControls();
    setSourceEditMode(false);
    setToolButton(pitchParamButton);
    refreshSelectedNoteParameter();
    setSize(1280, 760);
    startTimerHz(30);
}

MainComponent::~MainComponent()
{
    closingUI = true;
    cancelProjectMaterialRegistration();
    dismissMaterialPronunciationInput();
    tempoWindows.clear(); // Cancel and join the analysis we own before exit.
    settingsWindows.clear();voicebankWindows.clear();noteOtoWindows.clear();otoPreviews.clear();
    stopLiveMcp();
    diffSingerCancel->store(true);
    diffSingerCapabilityCancel->store(true);
    stopTimer();
    if (preferences != nullptr) audio.saveDeviceState(*preferences);
    panelSplitter.removeMouseListener(this);
    utauVoicebankName.removeMouseListener(this);
    audio.removeChangeListener(this);
    project.removeChangeListener(this);
    menuBar.setModel(nullptr);
    if (&juce::LookAndFeel::getDefaultLookAndFeel() == &lookAndFeel)
        juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
    setLookAndFeel(nullptr);
}

void MainComponent::applyRenderingPreference()
{
   #if JUCE_WINDOWS
    if (preferences == nullptr) return;
    if (auto* peer = getPeer())
    {
        const auto engines = peer->getAvailableRenderingEngines();
        const auto software = engines.indexOf("Software Renderer");
        const auto gpu = engines.indexOf("Direct2D");
        const auto forceSoftware = preferences->getBoolValue("ui.softwareRendering", false)
            || juce::SystemStats::getEnvironmentVariable("HACHI_SOFTWARE_RENDERING", {}) == "1";
        const auto target = !forceSoftware && gpu >= 0 ? gpu : software;
        if (target >= 0 && peer->getCurrentRenderingEngine() != target)
            peer->setCurrentRenderingEngine(target);
        const auto current = peer->getCurrentRenderingEngine();
        const auto renderer = current >= 0 && current < engines.size()
            ? engines[current] : juce::String("Unavailable");
        startupLog("UI renderer: " + renderer
                   + "; requested=" + (forceSoftware ? "software" : "gpu"));
        repaint();
    }
   #endif
}

void MainComponent::applyPreferences()
{
    if (preferences == nullptr) return;
    strings.setLanguage(static_cast<I18n::Language>(juce::jlimit(1, 5,
        preferences->getIntValue("ui.language", static_cast<int>(strings.getLanguage()) + 1)) - 1));
    const auto parseColour = [](juce::String value, juce::Colour fallback)
    {
        value = value.trim().removeCharacters("#");
        if (value.length() != 6 && value.length() != 8) return fallback;
        if (value.length() == 6) value = "ff" + value;
        return juce::Colour::fromString(value);
    };
    Palette::applyTheme(preferences->getValue("ui.theme", "dark"),
        parseColour(preferences->getValue("ui.accent", "7F69CA"), juce::Colour(0xff7f69ca)),
        parseColour(preferences->getValue("ui.accentLight", "CBCBFA"), juce::Colour(0xffcbcbfa)),
        parseColour(preferences->getValue("ui.noteColour", "F4C000"), juce::Colour(0xfff4c000)));
    audio.setHifiganModelDirectory(juce::File(
        preferences->getValue("algorithm.hifiganPath")));
    // A missing or empty setting resolves to the bundled engine inside
    // AudioEngine, which is also where the headless path picks it up.
    audio.setUtauResamplerFile(juce::File(
        preferences->getValue("algorithm.utauResampler").trim().unquoted()));
    audio.setUtauOutputDefaults(parseUtauOutputEngine(preferences->getValue("algorithm.utauOutputEngine")),
        juce::File(preferences->getValue("algorithm.utauWavtool").trim().unquoted()));
    timeline.repaint();
    trackList.repaint();
    // The HF daemon takes 10-20 s to load its model.  Started now, that wait
    // is over by the time anything is played rather than spent on the first
    // HF note.
    backend::UtauRenderer::prewarmHfDaemon(audio.currentUtauResamplerFile());
    const auto analysisConfig = backend::AnalysisService::configFromProperties(preferences.get());
    audio.setInferenceConfiguration(analysisConfig.inference, analysisConfig.deviceIndex);
    backend::DiffSingerRenderer::configure(backend::DiffSingerOptions::read(*preferences));
    pianoRoll.setFlagEditMode(preferences->getBoolValue("ui.continuousFlag", false)
        ? PianoRollComponent::FlagEditMode::continuous : PianoRollComponent::FlagEditMode::perNote);
    pianoRoll.setShowNoteLabels(preferences->getBoolValue("ui.showNoteLabels", false));
    showWaveforms = preferences->getBoolValue("ui.showWaveforms", false);
    pianoRoll.setShowWaveforms(showWaveforms);
    pianoRoll.setDrawLengthDivision(
        preferences->getIntValue("ui.drawLengthDivision", 64));
    pianoRoll.setDrawAction(preferences->getValue("ui.drawAction", "pitch-curve") == "notes"
        ? PianoRollComponent::DrawAction::notes : PianoRollComponent::DrawAction::pitchCurve);
    pianoRoll.setPitchScope(preferences->getValue("ui.pitchScope", "clip") == "track"
        ? PianoRollComponent::PitchScope::currentTrack : PianoRollComponent::PitchScope::focusedClip);
    refreshPitchScopeTitle();
    pianoRoll.setShowOverlap(preferences->getBoolValue("ui.showOverlap", false));
    viewOptions = viewOptionsFrom(*preferences);
    applyViewOptions();
    setTracksCollapsed(preferences->getBoolValue("ui.tracksCollapsed", false));
    const auto vZoom = juce::jlimit(0.5, 2.0, preferences->getDoubleValue("ui.vZoom", 1.0));
    vZoomSlider.setValue(vZoom, juce::dontSendNotification);
    pianoRoll.setRowHeight(22.0f * static_cast<float>(vZoom));
    timelineVerticalZoom = juce::jlimit(0.5, 2.0,
        preferences->getDoubleValue("ui.timelineVZoom", vZoom));
    if (!preferences->containsKey("ui.timelineVZoom"))
        preferences->setValue("ui.timelineVZoom", timelineVerticalZoom);
    const auto row = 96.0f * static_cast<float>(timelineVerticalZoom);
    timeline.setRowHeight(row);
    trackList.setRowHeight(row);
    // Applying a new model path must invalidate and immediately reschedule
    // already imported compose clips; waiting for a later edit made the
    // settings change appear ineffective.
    syncAudio(project.snapshot());
    lookAndFeel.refreshColours();
    for (auto* editor : { &bpmEditor, &beatsEditor })
    {
        editor->setColour(juce::Label::backgroundColourId, Palette::background);
        editor->setColour(juce::Label::outlineColourId, Palette::grid);
    }
    // A theme switch changes the shared Palette; components that cached a
    // Palette colour through setColour (labels, editors) must re-apply it or
    // they keep the previous theme's colour and, in light mode, draw a pale
    // dark-theme text colour that is unreadable.  Cascade the change so each
    // one's lookAndFeelChanged() re-applies from the current Palette.
    sendLookAndFeelChange();
    // Modeless material/settings windows are independent JUCE roots, not
    // descendants of MainComponent. Refresh their cached theme state too.
    auto& desktop = juce::Desktop::getInstance();
    for (int index = 0; index < desktop.getNumComponents(); ++index)
        if (auto* root = desktop.getComponent(index); root != nullptr && root != getTopLevelComponent())
        {
            if (auto* themed = dynamic_cast<HachiLookAndFeel*>(&root->getLookAndFeel())) themed->refreshColours();
            root->sendLookAndFeelChange();
        }
    applyUiScale();
    applyRenderingPreference();
}

void MainComponent::applyUiScale()
{
    static const float systemScale = juce::Desktop::getInstance().getGlobalScaleFactor();
    const auto uiScale = static_cast<float>(juce::jlimit(0.6, 2.0,
        preferences != nullptr ? preferences->getDoubleValue("ui.uiScale", 1.0) : 1.0));
    juce::Desktop::getInstance().setGlobalScaleFactor(systemScale * uiScale);
}

void MainComponent::refreshTexts()
{
    timeline.setGainKnobHint(strings.text("clip.gainKnobHint"));
    timeline.setGainEnvelopeTexts(strings.text("track.envelopeHint"),
        {strings.text("track.envelopeDelete"),strings.text("track.envelopeResetPoint"),strings.text("track.envelopeReset")});
    collapseTracksButton.setTooltip(utf8(
        "折叠上方的音轨与编排，把整个窗口留给调音；再次点击展开"));
    playButton.setButtonText({});
    playButton.setTooltip(strings.text("transport.play"));
    stopButton.setButtonText({});
    stopButton.setTooltip(strings.text("transport.stop"));
    noteEditButton.setButtonText({});
    noteEditButton.setTooltip(strings.text("tool.main"));
    wrenchButton.setButtonText({});
    wrenchButton.setTooltip(strings.text("tool.wrench"));
    drawButton.setButtonText({});
    drawButton.setTooltip(strings.text("tool.draw") + "\n" + strings.text(pianoRoll.drawAction() == PianoRollComponent::DrawAction::notes
        ? "draw.action.notes" : "draw.action.pitch"));
    lineButton.setButtonText({});
    lineButton.setTooltip(strings.text("tool.line"));
    pointButton.setButtonText({});
    pointButton.setTooltip(strings.text("tool.points") + utf8("；右键可选择 DS 音高还原或绘制偏移"));
    trimButton.setButtonText({});
    trimButton.setTooltip(strings.text("tool.trim"));
    nativeLoudnessButton.setButtonText(strings.text("tool.nativeLoudness"));
    nativeLoudnessButton.setTooltip(strings.text("tool.nativeLoudnessHint"));
    connectButton.setButtonText({});
    connectButton.setTooltip(strings.text("Source-note boundary split/join, not forced pitch portamento"));
    refreshPitchScopeTitle();
    smoothCaption.setText(strings.text("editor.smooth"), juce::dontSendNotification);
    pitchParamButton.setButtonText(strings.text("param.pitch"));
    driftParamButton.setButtonText(strings.text("param.drift"));
    attackParamButton.setButtonText(strings.text("param.attack"));
    breathParamButton.setButtonText(strings.text("param.breath"));
    tensionParamButton.setButtonText(strings.text("param.tension"));
    formantParamButton.setButtonText(strings.text("param.formant"));
    volumeParamButton.setButtonText(strings.text("param.volume"));
    volumeParamButton.setTooltip(utf8("编辑音量；高级形状使用高级包络按钮"));
    advancedEnvelopeButton.setButtonText(utf8("高级包络"));
    advancedEnvelopeButton.setTooltip(utf8("打开尾段淡出窗口：调整位置、起止响度和曲线，预览后应用或试听。"));
    advancedEnvelopeButton.onClick=[this]{showAdvancedEnvelopeMenu();};
    advancedEnvelopeButton.setComponentID("native-envelope-editor");
    pianoRoll.onOpenNativeEnvelope=[this]{setEnvelopeLane(EnvelopeLane::amplitude);};
    volumeParamButton.onSecondaryClick = {}; // Detailed envelope has one visible entry.
    robustPitchCurveButton.setButtonText(strings.text("param.robustPitchCurveShort"));
    robustPitchCurveButton.setTooltip(strings.text("param.robustPitchCurve"));
    bpmCaption.setText("BPM", juce::dontSendNotification);
    beatsCaption.setText(strings.text("beats.bar"), juce::dontSendNotification);
    denominatorLabel.setText("/ 4", juce::dontSendNotification);
    gridCaption.setText(strings.text("grid"), juce::dontSendNotification);
    stretchCaption.setText(strings.text("stretch.unit"), juce::dontSendNotification);
    stretchSelector.setTooltip(strings.text("stretch.unitHelp"));
    scaleCaption.setText(strings.text("base.scale"), juce::dontSendNotification);
    pitchLabel.setText(strings.text("algo.pitch"), juce::dontSendNotification);
    stretchLabel.setText(strings.text("algo.stretch"), juce::dontSendNotification);
    pitchAlgorithm.setTooltip(strings.text("algo.pitch"));
    stretchAlgorithm.setTooltip(strings.text("algo.stretch"));
    refreshStretchAlgorithmItems(stretchAlgorithm.getSelectedId());
    renderOrderLabel.setText(strings.text("algo.order"), juce::dontSendNotification);
    renderOrder.setTooltip(strings.text("algo.order"));
    {
        const auto previousOrder = renderOrder.getSelectedId();
        renderOrder.clear(juce::dontSendNotification);
        renderOrder.addItem(strings.text("algo.order.processThenSplice"), 1);
        renderOrder.addItem(strings.text("algo.order.stretchSpliceThenPitch"), 2);
        renderOrder.addItem(strings.text("algo.order.processMergeContiguous"), 3);
        renderOrder.setSelectedId(previousOrder > 0 ? previousOrder : 1,
                                  juce::dontSendNotification);
    }
    statusLabel.setText(strings.text("status.ready"), juce::dontSendNotification);
    sourceEditHint.setText(strings.text("edit.source"), juce::dontSendNotification);
    sampleAliasLabel.setText(strings.text("sample.alias"), juce::dontSendNotification);
    sampleStartLabel.setText(strings.text("sample.start"), juce::dontSendNotification);
    sampleEndLabel.setText(strings.text("sample.end"), juce::dontSendNotification);
    sampleAlignmentLabel.setText(strings.text("sample.alignment"), juce::dontSendNotification);
    sampleFixedLabel.setText(strings.text("sample.fixed"), juce::dontSendNotification);
    sampleSaveButton.setButtonText(strings.text("sample.save"));
    otoImportButton.setButtonText(strings.text("sample.importOto"));
    otoExportButton.setButtonText(strings.text("sample.exportOto"));
    // A dropdown rather than dialog settings: these are reached while tuning,
    // so they stay in the tool row where a click finds them, but folded into
    // one button because the row is the scarcest space in the window.
    nativeAudioOverlapButton.setButtonText(strings.text("native.allowOverlap"));
    nsfPitchTransitionsButton.setButtonText(strings.text("native.nsfPitchTransitions"));
    nsfPitchTransitionsButton.setTooltip(strings.text("native.nsfPitchTransitionsHelp"));
    nsfPitchTransitionsButton.onClick=[this]{project.setTrackNsfSmoothPitchTransitions(selectedTrackId,nsfPitchTransitionsButton.getToggleState());};
    nsfNoiseProtectionButton.setButtonText(strings.text("native.nsfNoiseProtection"));
    nsfNoiseProtectionButton.setTooltip(strings.text("native.nsfNoiseProtectionHelp"));
    nsfNoiseProtectionButton.onClick=[this]{project.setTrackNsfNoiseProtection(selectedTrackId,nsfNoiseProtectionButton.getToggleState());};
    nativeVibratoButton.setButtonText(strings.text("native.vibrato"));
    nativeVibratoButton.setTooltip(strings.text("native.vibratoHelp"));
    nativeVibratoButton.onClick = [this] { pianoRoll.showSelectedNativeVibratoDialog(); };
    timeline.setAudioOverlapLabel(strings.text("native.overlap"));
    nativeAudioOverlapButton.setTooltip(strings.text("native.overlapHelp"));
    nativeAudioOverlapButton.onClick=[this]{project.setNativeAudioOverlap(selectedTrackId,nativeAudioOverlapButton.getToggleState());};
    envelopePresetMenu.clear(juce::dontSendNotification);
    envelopePresetMenu.setTextWhenNothingSelected(utf8("包络预设"));
    envelopePresetMenu.setTooltip(utf8("选择预设，应用到所选音符或单独选中的音素"));
    const auto& presets = PianoRollComponent::envelopePresets();
    for(std::size_t i=0;i<presets.size();++i)
        envelopePresetMenu.getRootMenu()->addCustomItem(static_cast<int>(i)+1,
            std::make_unique<EnvelopePresetMenuItem>(presets[i]),nullptr,presets[i].name);
    envelopePresetMenu.onChange=[this]
    {
        const auto index=envelopePresetMenu.getSelectedId()-1;
        // This picker is an action, not a stored parameter. Clear it silently
        // so the same preset can be applied again after changing the selection.
        envelopePresetMenu.setSelectedId(0,juce::dontSendNotification);
        const auto& choices=PianoRollComponent::envelopePresets();
        if(index<0 || index>=static_cast<int>(choices.size()))return;
        const auto count=pianoRoll.applyEnvelopePreset(choices[static_cast<std::size_t>(index)]);
        if(count>0)statusLabel.setText(utf8("包络预设：")+juce::String(count)+utf8(" 个音符"),juce::dontSendNotification);
        else showError(utf8("先选中要套用预设的音符。"));
    };
    addAndMakeVisible(envelopePresetMenu);
    autoCrossfadeButton.setButtonText(utf8("自动交叉淡化"));
    addAndMakeVisible(autoCrossfadeButton);
    autoCrossfadeCurve = preferences->getBoolValue("ui.autoCrossfadeCurve", false);
    selectAutoCrossfadeMode(autoCrossfadeCurve ? 2 : 1);
    autoCrossfadeButton.onSecondaryClick = [this](juce::Point<int> at)
    {
        auto menu=autoCrossfadeModeMenu();menu.setLookAndFeel(&getLookAndFeel());
        menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({at.x,at.y,1,1}),
            [safe=juce::Component::SafePointer<MainComponent>(this)](int chosen)
            {if(safe!=nullptr)safe->selectAutoCrossfadeMode(chosen);});
    };
    autoCrossfadeButton.onClick = [this]
    {
        const auto count = pianoRoll.crossfadeSelectedNotes(autoCrossfadeCurve);
        if (count > 0) setEnvelopeLane(EnvelopeLane::amplitude);
        statusLabel.setText(count > 0 ? utf8("已生成交叉淡化，可在响度包络中编辑或撤销。")
            : utf8("请选择同轨的两个前后重叠音符；无重叠、完全包含或音源尚未解析时无法交叉淡化。"), juce::dontSendNotification);
    };
    spliceButton.setButtonText(utf8("强制连接"));
    spliceButton.setTooltip(utf8("把选中的音符连接为一条可渲染的原生边界；"
                                  "保留每个音符和素材数据，支持跨片段连接"));
    spliceButton.onClick = [this] { spliceSelectedNotes(); };
    spliceButton.setEnabled(false);
    pianoRoll.onDiffSingerParameterLaneRequested = [this] { showDiffSingerParameterLane(); };
    flagCurveButton.onSecondaryClick = flagEnvelopeButton.onSecondaryClick =
        [this](juce::Point<int> at) { showFlagEditModeMenu(at); };
    flagCurveButton.setButtonText(utf8("线性flag"));
    flagCurveButton.setTooltip(utf8("让选中音符的 g 随时间连续变化，改由 flag 包络里的曲线决定；"
                                     "Flags 里写的 g 在开启期间不再起作用，其它 flag 照常。"
                                     "支持普通 UTAU、界•UTAU 与谋•UTAU；需使用支持曲线的 WCSNDM 引擎"));
    flagCurveButton.setClickingTogglesState(true);
    flagCurveButton.setEnabled(false);
    flagCurveButton.onClick = [this]
    {
        if (pianoRoll.diffSingerFlagContext) { showDiffSingerParameterLane(); return; }
        const auto noteIds = pianoRoll.selectedNoteIds();
        if (noteIds.empty()) return;
        const auto enable = flagCurveButton.getToggleState();
        project.setNotesUtauFlagCurveEnabled(noteIds, enable);
        statusLabel.setText((enable ? utf8("线性 flag：") : utf8("取消线性 flag："))
            + juce::String(static_cast<int>(noteIds.size())) + utf8(" 个音符"),
            juce::dontSendNotification);
        // Switching a note back to a plain flag leaves the lane with nothing
        // to draw, so it steps back to the note tool rather than sitting empty.
        if (!enable && pianoRoll.currentTool() == PianoRollComponent::Tool::flagCurve)
        {
            pianoRoll.setTool(PianoRollComponent::Tool::note);
            setToolButton(noteEditButton);
        }
        refreshSelectedNoteParameter();
    };
    flagEnvelopeButton.setButtonText(utf8("flag 包络"));
    flagEnvelopeButton.setTooltip(utf8("在底部通道里画 g 的渐变曲线；先对该音符开启线性 flag"
                                        "（普通 UTAU、界•UTAU 与谋•UTAU 模式）"));
    flagEnvelopeButton.setEnabled(false);
    flagEnvelopeButton.onClick = [this]
    {
        if (pianoRoll.diffSingerFlagContext && envelopeLane != EnvelopeLane::flagCurve)
        { showDiffSingerParameterLane(); return; }
        setEnvelopeLane(nextEnvelopeLane(envelopeLane, EnvelopeLane::flagCurve));
    };
    noteAliasLabel.setText(utf8("音符发音/别名"), juce::dontSendNotification);
    noteConsonantVelocityLabel.setText(strings.text("editor.attackSpeed"), juce::dontSendNotification);
    const auto noteVelocityHelp = strings.text("native.velocityHelp");
    noteConsonantVelocityLabel.setTooltip(noteVelocityHelp);
    noteConsonantVelocityEditor.setTooltip(noteVelocityHelp);
    noteConsonantVelocityEditor.setTextToShowWhenEmpty({}, Palette::textMuted);
    noteFlagsLabel.setText("Flags", juce::dontSendNotification);
    noteFlagsEditor.setTextToShowWhenEmpty({}, Palette::textMuted);
    refreshSelectedNoteParameter();
}

void MainComponent::refreshTrackPitchItems(const TrackData*)
{
    const auto previous = pitchAlgorithm.getSelectedId();
    pitchAlgorithm.clear(juce::dontSendNotification);
    pitchAlgorithm.addItem("nsf-hifigan", 2);
    pitchAlgorithm.addItem("WORLD", 3);
    pitchAlgorithm.addItem("llsm2", 6);
    pitchAlgorithm.addSeparator();
    pitchAlgorithm.addItem("UTAU", 7);
    pitchAlgorithm.addItem(utf8("界•UTAU"), 8);
    pitchAlgorithm.addItem(utf8("谋•UTAU"), 9);
    pitchAlgorithm.addSeparator();
    pitchAlgorithm.addItem("DiffSinger (DS)", 10);
    pitchAlgorithm.setSelectedId(previous, juce::dontSendNotification);
    pitchAlgorithm.setTooltip(utf8("变调算法 / 编辑模式"));
    pitchAlgorithm.setColour(juce::ComboBox::backgroundColourId, Palette::accent);
    pitchAlgorithm.setColour(juce::ComboBox::textColourId, Palette::contrastingText(Palette::accent));
}

void MainComponent::refreshTrackEditingInterface(const TrackData* track)
{
    const auto id=track?track->id:juce::String{};const auto type=track?effectiveTrackType(*track):TrackType::hachi;
    if(trackInterfaceOwner!=id||displayedTrackType!=type)
    {
        // Selection restoration calls focusNote -> focusClip -> this refresh.
        // Publish the new owner before those synchronous callbacks can re-enter.
        trackInterfaceOwner=id;displayedTrackType=type;
        std::vector<juce::String> keep;
        if(track)for(const auto& selected:pianoRoll.selectedNoteIds())for(const auto& clip:track->clips)for(const auto& n:clip.notes)if(n.id==selected)keep.push_back(n.id);
        finishMaterialPronunciationInput(true); // discard UI query only, never a model action
        pianoRoll.cancelPendingEditingForTrackChange();
        sourceEditActive=false;pianoRoll.setSourceEditMode(false);audio.clearAuditionFile();
        closeEnvelopeLanes();pianoRoll.setShowEnvelope(false);
        noteFlagsDirty=false;noteConsonantVelocityDirty=false;
        if(track)
        {
            if(std::none_of(track->clips.begin(),track->clips.end(),[&](const auto& c){return c.id==selectedClipId;}))selectedClipId.clear();
            bool ownsNote=false;for(const auto& c:track->clips)for(const auto& n:c.notes)ownsNote|=n.id==selectedNoteId;
            if(!ownsNote)selectedNoteId.clear();
        }
        else{selectedClipId.clear();selectedNoteId.clear();}
        pianoRoll.setSelectedNoteIds(keep);
        menuItemsChanged(); // Track semantics, not a possibly stale algorithm picker.
    }
    projectedTrackEditable=track&&track->compose&&!track->accompaniment&&!track->referenceOnly;
    utauVoicebankButton.setButtonText(strings.text("utau.chooseVoicebank"));
    const auto bankDirectory = track ? track->voicebankDirectory : juce::File{};
    utauVoicebankName.setText(bankDirectory != juce::File{} ? bankDirectory.getFileName()
        : strings.text("utau.noVoicebankSelected"), juce::dontSendNotification);
    utauVoicebankName.setTooltip(bankDirectory != juce::File{} ? bankDirectory.getFullPathName()
        + (type == TrackType::utau ? "\n" + strings.text("utau.editVoicebankHint") : juce::String{})
        : strings.text("utau.noVoicebankSelected"));
    utauVoicebankName.setMouseCursor(type == TrackType::utau ? juce::MouseCursor::PointingHandCursor
        : juce::MouseCursor::NormalCursor);
    utauVoicebankButton.setTooltip(strings.text("utau.chooseVoicebankHint")
        + (track && track->voicebankDirectory != juce::File{}
            ? "\n" + track->voicebankDirectory.getFullPathName() : juce::String{}));
    trackEditingHeading.setText(track ? track->name : juce::String{}, juce::dontSendNotification);
    trackEditingHeading.setTooltip(track ? track->name : juce::String{});
    diffSingerContextLabel.setText(track?(strings.text("Language: "))+track->diffSingerLanguage+(strings.text("  Speaker: "))
        +(track->diffSingerSpeaker.isEmpty()?juce::String("default"):track->diffSingerSpeaker):juce::String{},juce::dontSendNotification);
    diffSingerModelButton.setButtonText(strings.text("Choose model"));diffSingerTimingButton.setButtonText(strings.text("Phoneme timing"));
    diffSingerPronunciationButton.setButtonText(strings.text("Pronunciation"));diffSingerPitchButton.setButtonText(strings.text("Generate pitch"));
    diffSingerParametersButton.setButtonText(strings.text("Variance parameters"));diffSingerSettingsButton.setButtonText(strings.text("Language / speaker"));
    const bool ds=projectedTrackEditable&&type==TrackType::diffSinger;
    for(auto* c:{static_cast<juce::Component*>(&diffSingerContextLabel),static_cast<juce::Component*>(&diffSingerModelButton),static_cast<juce::Component*>(&diffSingerTimingButton),
        static_cast<juce::Component*>(&diffSingerPronunciationButton),static_cast<juce::Component*>(&diffSingerPitchButton),static_cast<juce::Component*>(&diffSingerParametersButton),static_cast<juce::Component*>(&diffSingerSettingsButton)})c->setVisible(ds);
    refreshTrackPitchItems(track);
}

void MainComponent::refreshProjectControls()
{
    // Which track is in hand decides whether a material track is heard, and
    // the engine only re-reads that while syncing.
    if (auditionTrackAtLastSync != selectedTrackId)
    {
        auditionTrackAtLastSync = selectedTrackId;
        syncAudio(project.snapshot());
    }
    const auto data = project.snapshot();
    const auto overlapTrack=std::find_if(data.tracks.begin(),data.tracks.end(),[&](const auto& t){return t.id==selectedTrackId;});
    nativeAudioOverlapButton.setVisible(overlapTrack!=data.tracks.end()&&trackShowsAllNativeRegions(*overlapTrack)&&!sourceEditActive);
    nativeVibratoButton.setVisible(nativeAudioOverlapButton.isVisible());
    nsfPitchTransitionsButton.setVisible(overlapTrack!=data.tracks.end()
        && trackShowsAllNativeRegions(*overlapTrack) && overlapTrack->pitchAlgorithm==PitchAlgorithm::nsfHifigan
        && !sourceEditActive);
    nsfPitchTransitionsButton.setToggleState(overlapTrack!=data.tracks.end() && overlapTrack->nsfSmoothPitchTransitions,juce::dontSendNotification);
    nsfNoiseProtectionButton.setVisible(nsfPitchTransitionsButton.isVisible());
    nsfNoiseProtectionButton.setToggleState(overlapTrack!=data.tracks.end() && overlapTrack->nsfNoiseProtection,juce::dontSendNotification);
    trimButton.setEnabled(overlapTrack!=data.tracks.end()&&trackShowsAllNativeRegions(*overlapTrack)&&!sourceEditActive);
    nativeLoudnessButton.setEnabled(trimButton.isEnabled());
    timeline.setClipStretchMode(false);
    pianoRoll.setNativeLoudnessMode(nativeLoudnessButton.isEnabled() && nativeLoudnessButton.getToggleState());
    if (!trimButton.isEnabled() && (pianoRoll.currentTool() == PianoRollComponent::Tool::trim || nativeLoudnessButton.getToggleState()))
    {
        pianoRoll.setTool(PianoRollComponent::Tool::note); setToolButton(noteEditButton);
    }
    nativeAudioOverlapButton.setToggleState(overlapTrack!=data.tracks.end()&&overlapTrack->allowNativeAudioOverlap,juce::dontSendNotification);
    bpmEditor.setText(juce::String(data.bpm, std::abs(data.bpm - std::floor(data.bpm)) < 1.0e-9 ? 0 : 2),
                      juce::dontSendNotification);
    beatsEditor.setText(juce::String(data.numerator), juce::dontSendNotification);
    gridSelector.setText(data.gridDivision, juce::dontSendNotification);
    stretchSelector.setSelectedId(juce::jlimit(2, 128, data.noteEditDivision),
                                  juce::dontSendNotification);
    scaleSelector.setText(data.baseScale, juce::dontSendNotification);
    auto selected = selectedTrackId.isNotEmpty()
        ? std::find_if(data.tracks.begin(), data.tracks.end(),
            [this](const auto& track) { return track.id == selectedTrackId; })
        : data.tracks.end();
    if (selected == data.tracks.end())
        selected = std::find_if(data.tracks.begin(), data.tracks.end(),
                                [](const auto& track) { return track.compose; });
    refreshTrackEditingInterface(selected!=data.tracks.end()?&*selected:nullptr);
    if (selected != data.tracks.end())
    {
        const auto pitchId = selected->accompaniment ? 0 : effectiveTrackType(*selected)==TrackType::diffSinger ? 10 : selected->pitchAlgorithm == PitchAlgorithm::nsfHifigan ? (trackUsesVoicebankSynthesis(*selected) ? utauModePickerItem(selected->utauMode) : 2)
            : selected->pitchAlgorithm == PitchAlgorithm::world ? 3
            : selected->pitchAlgorithm == PitchAlgorithm::vocalShifter ? 4
            : selected->pitchAlgorithm == PitchAlgorithm::llsm2 ? 6
            : selected->pitchAlgorithm == PitchAlgorithm::utau
                ? utauModePickerItem(selected->utauMode) : 6;
        const auto stretchId = selected->stretchAlgorithm == StretchAlgorithm::variableMelHop ? 2
            : selected->stretchAlgorithm == StretchAlgorithm::loop ? 3
            : selected->stretchAlgorithm == StretchAlgorithm::soundTouch ? 4
            : selected->stretchAlgorithm == StretchAlgorithm::wcsNonLoop ? 7
            : selected->stretchAlgorithm == StretchAlgorithm::hifiShifterMel ? 6
            : selected->stretchAlgorithm == StretchAlgorithm::nsfShiftThenSplice ? 5 : 1;
        pitchAlgorithm.setSelectedId(pitchAlgorithm.indexOfItemId(pitchId) >= 0 ? pitchId : 0,
                                     juce::dontSendNotification);
        const auto utauItem = trackUsesVoicebankSynthesis(*selected);
        if (pitchAlgorithm.getSelectedId() == 0 && utauAmplitudeEnvelopeActive)
        {
            closeEnvelopeLanes();
            pianoRoll.setTool(PianoRollComponent::Tool::note);
            setToolButton(noteEditButton);
        }
        else if (utauItem && volumeParamButton.getToggleState()
                 && !utauAmplitudeEnvelopeActive)
        {
            pianoRoll.setTool(PianoRollComponent::Tool::note);
            setToolButton(noteEditButton);
        }
        refreshStretchAlgorithmItems(stretchId);
        renderOrder.setSelectedId(
            selected->renderOrder == RenderOrder::stretchSpliceThenPitch ? 2 : selected->mergeContiguousSource ? 3 : 1,
            juce::dontSendNotification);
    }
    const auto accompaniment = selected != data.tracks.end() && selected->accompaniment;
    phonemizerButton.setVisible(selected != data.tracks.end() && selected->id == selectedTrackId
        && effectiveTrackType(*selected)==TrackType::utau && trackUsesVoicebankSynthesis(*selected) && selected->utauMode == UtauMode::classic
        && !trackIsDiffSinger(*selected) && !sourceEditActive);
    phonemizerButton.setButtonText(strings.text("phonemizer.choose") + ": "
        + strings.text(selected != data.tracks.end() && selected->chineseCvvc
            ? "phonemizer.zhCvvc" : "phonemizer.manualShort"));
    phonemizerButton.setTooltip(strings.text("phonemizer.toolbarHelp"));
    pianoPreviewButton.setVisible(selected != data.tracks.end()
        && effectiveTrackType(*selected) == TrackType::utau
        && trackUsesVoicebankSynthesis(*selected) && !accompaniment && !sourceEditActive);
    pianoPreviewButton.setButtonText(strings.text("utau.pianoPreview"));
    pianoPreviewButton.setTooltip(strings.text("utau.pianoPreviewHelp"));
    pianoPreviewButton.setToggleState(selected != data.tracks.end() && selected->utauPianoPreview,
                                     juce::dontSendNotification);
    const auto dsInterface=selected!=data.tracks.end()&&effectiveTrackType(*selected)==TrackType::diffSinger;
    pitchAlgorithm.setEnabled(projectedTrackEditable && !sourceEditActive && !diffSingerBusy);
    stretchAlgorithm.setEnabled(!accompaniment&&!dsInterface);
    renderOrder.setEnabled(!accompaniment&&!dsInterface);
    pianoRoll.setEnabled(!accompaniment);
    wrenchButton.setEnabled(!accompaniment);
    if (accompaniment)
    {
        if (sourceEditActive) setSourceEditMode(false);
        pitchAlgorithm.setText(strings.text("track.originalAudio"), juce::dontSendNotification);
        stretchAlgorithm.setSelectedId(0, juce::dontSendNotification);
        stretchAlgorithm.setText(strings.text("track.noProcessing"), juce::dontSendNotification);
    }
    refreshSelectedNoteParameter();
    refreshSpliceButton();
    resized();
}

bool MainComponent::selectedTrackIsUtau() const
{
    // The picker and the track itself both: every UTAU mode shows in the
    // picker, and the track is what the conversion is going to change.
    if (!isUtauAlgorithmSelected()) return false;
    const auto data = project.snapshot();
    return std::any_of(data.tracks.begin(), data.tracks.end(), [this](const auto& track)
    {
        return track.id == selectedTrackId && trackUsesVoicebankSynthesis(track);
    });
}

std::optional<bool> MainComponent::diagnosticEditMenuItemEnabled(int itemId)
{
    auto menu = getMenuForIndex(1, {});
    for (juce::PopupMenu::MenuItemIterator item(menu); item.next();)
        if (item.getItem().itemID == itemId) return item.getItem().isEnabled;
    return std::nullopt;
}

void MainComponent::diagnosticChooseMenuItem(int itemId)
{
    menuItemSelected(itemId, 1);
}

void MainComponent::diagnosticSelectTrack(const juce::String& trackId)
{
    if (trackList.onTrackSelected) trackList.onTrackSelected(trackId);
}

bool MainComponent::isUtauAlgorithmSelected() const
{
    // Every UTAU mode, not just the two that existed first: the whole UTAU
    // toolbar hangs off this, so a mode left out of it comes up wearing the
    // generic parameter set instead.
    if (utauModeForPickerItem(pitchAlgorithm.getSelectedId()).has_value()) return true;
    if (!nsfPickerItem(pitchAlgorithm.getSelectedId())) return false;
    const auto data = project.snapshot();
    return std::any_of(data.tracks.begin(), data.tracks.end(), [this](const auto& track)
    { return track.id == selectedTrackId && trackUsesVoicebankSynthesis(track); });
}

void MainComponent::refreshSelectedNoteParameter()
{
    nativeVibratoButton.setEnabled(pianoRoll.canEditSelectedNativeVibrato());
    const auto diffSinger = refreshDiffSingerFlagContext();
    NoteData selected;
    auto found = false;
    const auto data = project.snapshot();
    const auto selectedIds = pianoRoll.selectedNoteIds();
    std::vector<NoteData> selectedNotes;
    std::vector<int> selectedNoteConsonantVelocities;
    // Whether every selected note is on a track that can carry a 线性flag.
    auto flagCurvesOnSelection = true;
    selectedNotes.reserve(selectedIds.size());
    selectedNoteConsonantVelocities.reserve(selectedIds.size());
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
            {
                if (std::find(selectedIds.begin(), selectedIds.end(), note.id)
                    != selectedIds.end())
                {
                    selectedNotes.push_back(note);
                    selectedNoteConsonantVelocities.push_back(
                        note.utauConsonantVelocity != inheritedUtauConsonantVelocity
                            ? note.utauConsonantVelocity
                            : track.utauConsonantVelocity);
                    flagCurvesOnSelection = flagCurvesOnSelection
                        && trackTakesFlagCurves(track);
                }
                if (note.id == selectedNoteId)
                {
                    selected = note;
                    found = true;
                }
            }
    if (!found && !selectedNotes.empty())
    {
        selected = selectedNotes.front();
        found = true;
    }
    updatingSmoothSlider = true;
    double value = 0.0;
    if (parameterMode == ParameterMode::pitchSmooth)
    {
        smoothCaption.setText(strings.text("editor.smooth"), juce::dontSendNotification);
        smoothSlider.setRange(0.0, 100.0, 1.0);
        smoothSlider.setTextValueSuffix("%");
        value = (1.0 - static_cast<double>(selected.modulation)) * 100.0;
    }
    else if (parameterMode == ParameterMode::pitchDrift)
    {
        smoothCaption.setText(strings.text("editor.drift"), juce::dontSendNotification);
        // Melodyne stores the remaining drift factor.  Present the familiar
        // correction amount: 100% removes drift, 0% preserves it, while a
        // negative value retains imported emphasis factors above 1.0.
        smoothSlider.setRange(-100.0, 100.0, 1.0);
        smoothSlider.setTextValueSuffix("%");
        value = (1.0 - static_cast<double>(selected.drift)) * 100.0;
    }
    else if (parameterMode == ParameterMode::attackSpeed)
    {
        smoothCaption.setText(strings.text("editor.attackSpeed"), juce::dontSendNotification);
        smoothSlider.setRange(5.0, 2000.0, 1.0);
        smoothSlider.setTextValueSuffix("%");
        value = static_cast<double>(selected.attackSpeed) * 100.0;
    }
    else if (parameterMode == ParameterMode::breath)
    {
        smoothCaption.setText(strings.text("param.breath"), juce::dontSendNotification);
        smoothSlider.setRange(0.0, 100.0, 1.0);
        smoothSlider.setTextValueSuffix("%");
        value = static_cast<double>(selected.breath) * 100.0;
    }
    else if (parameterMode == ParameterMode::tension)
    {
        smoothCaption.setText(strings.text("param.tension"), juce::dontSendNotification);
        smoothSlider.setRange(-100.0, 100.0, 1.0);
        smoothSlider.setTextValueSuffix("%");
        value = static_cast<double>(selected.tension) * 100.0;
    }
    else if (parameterMode == ParameterMode::formant)
    {
        smoothCaption.setText(strings.text("param.formant"), juce::dontSendNotification);
        smoothSlider.setRange(-12.0, 12.0, 0.1);
        smoothSlider.setTextValueSuffix(" st");
        value = selected.formantSemitones;
    }
    else
    {
        smoothCaption.setText(strings.text("param.volume"), juce::dontSendNotification);
        smoothSlider.setRange(-60.0, 12.0, 0.1);
        smoothSlider.setTextValueSuffix(" dB");
        value = selected.gain > 1.0e-6f ? 20.0 * std::log10(selected.gain) : -60.0;
    }
    smoothSlider.setValue(value, juce::dontSendNotification);
    updatingSmoothSlider = false;
    smoothSlider.setEnabled(found);
    const auto multipleNotes = selectedNotes.size() > 1;
    noteAliasEditor.setEnabled(found && !multipleNotes);
    noteConsonantVelocityEditor.setEnabled(!selectedNotes.empty() && !diffSinger);
    noteConsonantVelocityEditor.setTooltip(diffSinger ? utf8("DS 使用音素时长编辑（Ctrl+G）；速度参数请使用 VELC 曲线") : juce::String{});
    noteFlagsEditor.setEnabled(!selectedNotes.empty() && !diffSinger);
    noteFlagsEditor.setTooltip(diffSinger ? utf8("点击 FLAG 编辑实参，在曲线通道切换实参 / 偏移") : juce::String{});
    // The switch follows the notes, and the lane opens only for notes that
    // have a curve to show.
    const auto anyFlagCurve = std::any_of(selectedNotes.begin(), selectedNotes.end(),
        [](const auto& note) { return note.utauFlagCurveEnabled; });
    const auto allFlagCurve = !selectedNotes.empty()
        && std::all_of(selectedNotes.begin(), selectedNotes.end(),
            [](const auto& note) { return note.utauFlagCurveEnabled; });
    // Ordinary UTAU, 界 and 谋 all use the same switch and curve lane.
    flagCurveButton.setEnabled(diffSinger || (!selectedNotes.empty() && flagCurvesOnSelection));
    flagCurveButton.setToggleState(diffSinger ? envelopeLane == EnvelopeLane::flagCurve : allFlagCurve, juce::dontSendNotification);
    flagEnvelopeButton.setEnabled(diffSinger || (anyFlagCurve && flagCurvesOnSelection));
    if (!noteAliasEditor.hasKeyboardFocus(false))
        noteAliasEditor.setText(multipleNotes ? juce::String::fromUTF8("×")
            : found ? selected.label : juce::String{}, false);
    if (!noteFlagsEditor.hasKeyboardFocus(false))
    {
        noteFlagsMixed = false;
        juce::String displayedFlags;
        if (!selectedNotes.empty())
        {
            displayedFlags = selectedNotes.front().utauFlags;
            noteFlagsMixed = std::any_of(std::next(selectedNotes.begin()), selectedNotes.end(),
                [&displayedFlags](const auto& note)
                {
                    return note.utauFlags != displayedFlags;
                });
            if (noteFlagsMixed) displayedFlags = "-";
        }
        noteFlagsDirty = false;
        noteFlagsEditor.setText(displayedFlags, false);
    }
    if (!noteConsonantVelocityEditor.hasKeyboardFocus(false))
    {
        noteConsonantVelocityMixed = false;
        juce::String displayedVelocity;
        if (!selectedNoteConsonantVelocities.empty())
        {
            const auto velocity = selectedNoteConsonantVelocities.front();
            noteConsonantVelocityMixed = std::any_of(
                std::next(selectedNoteConsonantVelocities.begin()),
                selectedNoteConsonantVelocities.end(),
                [velocity](const auto value)
                {
                    return value != velocity;
                });
            if (noteConsonantVelocityMixed)
                displayedVelocity = "-";
            else
                displayedVelocity = juce::String(velocity);
        }
        noteConsonantVelocityDirty = false;
        noteConsonantVelocityEditor.setText(displayedVelocity, false);
    }
    updatingRobustPitchCurve = true;
    robustPitchCurveButton.setToggleState(found && selected.robustPitchCurve,
                                           juce::dontSendNotification);
    updatingRobustPitchCurve = false;
    const auto showRobust = pitchAlgorithm.getSelectedId() == 1;
    const auto visibilityChanged = robustPitchCurveButton.isVisible() != showRobust;
    robustPitchCurveButton.setVisible(showRobust);
    robustPitchCurveButton.setEnabled(found && showRobust);
    if (visibilityChanged) resized();
}

void MainComponent::applySelectedNoteParameter()
{
    const auto value = smoothSlider.getValue();
    if (parameterMode == ParameterMode::pitchSmooth)
        project.setNoteModulation(selectedNoteId,
            1.0f - static_cast<float>(value / 100.0));
    else if (parameterMode == ParameterMode::pitchDrift)
        project.setNoteDrift(selectedNoteId,
            1.0f - static_cast<float>(value / 100.0));
    else if (parameterMode == ParameterMode::attackSpeed)
        project.setNoteAttackSpeed(selectedNoteId,
            static_cast<float>(value / 100.0));
    else if (parameterMode == ParameterMode::breath)
        project.setNoteBreath(selectedNoteId, static_cast<float>(value / 100.0));
    else if (parameterMode == ParameterMode::tension)
        project.setNoteTension(selectedNoteId, static_cast<float>(value / 100.0));
    else if (parameterMode == ParameterMode::formant)
        project.setNoteFormant(selectedNoteId, static_cast<float>(value));
    else
        project.setNoteGain(selectedNoteId, value <= -59.9 ? 0.0f
            : static_cast<float>(std::pow(10.0, value / 20.0)));
}

void MainComponent::commitNoteAlias()
{
    if (selectedNoteId.isEmpty() || !noteAliasEditor.isEnabled()) return;
    project.setNoteLabel(selectedNoteId, noteAliasEditor.getText());
    noteLabelCommitted(selectedNoteId);
    // After the track has a voicebank, or the note would have no span to take
    // its shape from.
    pianoRoll.ensureDefaultEnvelope(selectedNoteId);
}

void MainComponent::commitNoteFlags()
{
    if (!noteFlagsEditor.isEnabled()) return;
    const auto noteIds = pianoRoll.selectedNoteIds();
    if (noteIds.empty()) return;
    // Merely focusing and leaving a mixed-value field must not replace every
    // note's Flags with the visual "-" marker.
    if (noteFlagsMixed && !noteFlagsDirty) return;
    project.setNotesUtauFlags(noteIds, noteFlagsEditor.getText());
    noteFlagsMixed = false;
    noteFlagsDirty = false;
}

void MainComponent::commitNoteConsonantVelocity()
{
    if (!noteConsonantVelocityEditor.isEnabled()) return;
    const auto noteIds = pianoRoll.selectedNoteIds();
    if (noteIds.empty()) return;
    // Do not treat the mixed-value marker as an edited value merely because
    // the editor received or lost keyboard focus.
    if (noteConsonantVelocityMixed && !noteConsonantVelocityDirty) return;
    const auto text = noteConsonantVelocityEditor.getText().trim();
    // A lone minus sign is an in-progress negative number (or the mixed-value
    // marker), never a request to overwrite the selected notes with zero.
    if (text == "-") return;
    const auto velocity = text.isEmpty()
        ? inheritedUtauConsonantVelocity : text.getIntValue();
    project.setNotesUtauConsonantVelocity(noteIds, velocity);
    if (velocity != inheritedUtauConsonantVelocity
        && text != juce::String(velocity))
        noteConsonantVelocityEditor.setText(juce::String(velocity), false);
    noteConsonantVelocityMixed = false;
    noteConsonantVelocityDirty = false;
}

void MainComponent::chooseUtauVoicebank()
{
    const auto data = project.snapshot();
    const auto selected = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrackId; });
    if (selected == data.tracks.end()) return;
    auto initial = selected->voicebankDirectory.isDirectory()
        ? selected->voicebankDirectory : juce::File{};
    if (!initial.isDirectory() && preferences != nullptr)
    {
        const juce::File defaultDirectory(
            preferences->getValue("algorithm.utauVoicebank"));
        if (defaultDirectory.isDirectory()) initial = defaultDirectory;
    }
    chooser = std::make_unique<juce::FileChooser>(utf8("选择 UTAU / DiffSinger 音源库目录"), initial);
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectDirectories,
        [safe = juce::Component::SafePointer<MainComponent>(this), trackId = selected->id](const juce::FileChooser& selectedChooser)
        {
            if (safe != nullptr) safe->selectVoicebank(trackId, selectedChooser.getResult());
        });
}

bool MainComponent::selectVoicebank(const juce::String& trackId, const juce::File& directory)
{
    if (!directory.isDirectory()) return false;
    const auto data = project.snapshot();
    if (std::none_of(data.tracks.begin(), data.tracks.end(), [&](const auto& track)
        { return track.id == trackId && !track.accompaniment; })) return false;
    if (backend::DiffSingerRenderer::isVoicebank(directory))
    {
        loadDiffSingerVoicebank(trackId, directory);
        return true;
    }
    // Selecting an existing bank only binds its path. importVoicebank is an
    // inventory operation: it opens every WAV and repeatedly decodes the OTO
    // just to count aliases. None of those rows populate the renderer's index.
    // The existing asynchronous index supplies timing when notes need it.
    project.setTrackVoicebankDirectory(trackId, directory);
    // A track binding is not an implicit global-default edit.
    statusLabel.setText(utf8("UTAU 音源库：") + directory.getFileName(),
                        juce::dontSendNotification);
    return true;
}

void MainComponent::refreshSpliceButton()
{
    // One note has no boundary to splice, so the action stays out of reach
    // until at least two are selected.
    spliceButton.setEnabled(pianoRoll.selectedNoteIds().size() >= 2);
}

void MainComponent::spliceSelectedNotes()
{
    // Historical private name; this ONE visible action now always means pitch
    // portamento, never a backend-dependent audio splice or material join.
    const auto selected = pianoRoll.selectedNoteIds();
    if (selected.size() < 2) return;
    const auto data = project.snapshot();
    struct Placed { juce::String id, trackId; double start = 0; };
    std::vector<Placed> notes;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (std::find(selected.begin(), selected.end(), note.id) != selected.end())
                    notes.push_back({note.id,track.id,clip.startSeconds+note.startSeconds});
    std::stable_sort(notes.begin(),notes.end(),[](const auto& a,const auto& b)
    {return a.trackId==b.trackId?a.start<b.start:a.trackId<b.trackId;});
    bool allConnected=notes.size()==selected.size();
    for(std::size_t i=1;i<notes.size();++i)
        allConnected &= notes[i-1].trackId==notes[i].trackId
            && std::any_of(data.nativeConnections.begin(),data.nativeConnections.end(),[&](const auto& edge)
                {return edge.leftNoteId==notes[i-1].id&&edge.rightNoteId==notes[i].id&&!edge.pitchCurve.empty();});
    juce::String error;
    if(!project.setNotesPortamento(selected,!allConnected,error))
    {
        if(error.isNotEmpty())showError(error);
        return;
    }
    statusLabel.setText(allConnected ? utf8("已断开所选音符的滑音连接；素材与包络保持不变。")
        : utf8("已建立滑音连接；可编辑连接音高曲线，素材与包络保持不变。"),juce::dontSendNotification);
}

void MainComponent::showVoicebankSettings()
{
    const auto data = project.snapshot();
    const auto selected = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrackId; });
    if (selected == data.tracks.end()
        || !trackUsesVoicebankSynthesis(*selected))
        return;
    if (!selected->voicebankDirectory.isDirectory())
    {
        showError(utf8("请先为当前轨道选择 UTAU 音源库。"));
        return;
    }

    if (backend::DiffSingerRenderer::isVoicebank(selected->voicebankDirectory))
    {
        showDiffSingerSettings(*selected);
        return;
    }
    const auto windowKey="bank|"+selected->voicebankDirectory.getFullPathName()+"|"+juce::String(static_cast<int>(selected->utauMode));
    if(voicebankWindows.showExisting(windowKey))return;
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = utf8("音源库设置 — ") + selected->voicebankDirectory.getFileName();
    options.dialogBackgroundColour = Palette::panel;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = true;
    // Open on the entry the selected note uses, so the alias does not have to
    // be hunted down in a large voicebank.
    juce::String noteAlias;
    if (selectedNoteId.isNotEmpty())
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (note.id == selectedNoteId) noteAlias = note.label;
    auto* panel = new VoicebankSettingsComponent(
        selected->voicebankDirectory, utauModeUsesRegions(selected->utauMode),
        selected->utauMode == UtauMode::mou, noteAlias);
    juce::Component::SafePointer<MainComponent> safe(this);
    panel->configureOtoEditor = [safe](OtoWaveformEditorComponent& editor)
    { if (safe != nullptr) safe->attachOtoPlayback(editor); };
    options.content.setOwned(panel);
    if (auto* window = voicebankWindows.show(options,windowKey))
        window->setResizeLimits(780, 440, 1800, 1200);
}


void MainComponent::showRegionEditorForNote(const juce::String& noteId)
{
    // Context/keyboard access resolves into the material manager's ONE precise
    // editor; it must not create a third independent material writer.
    openSelectedMaterialEditor(false, noteId);
}

void MainComponent::showNoteOtoEditorForNote(const juce::String& noteId)
{
    if(noteOtoWindows.showExisting(noteId))return;
    const auto data = project.snapshot();
    for (const auto& track : data.tracks) if (trackIsDiffSinger(track))
        for (const auto& clip : track.clips) for (const auto& note : clip.notes)
            if (note.id == noteId) { showDiffSingerPhonemeEditor(track.id, noteId); return; }
    juce::String error;
    auto editor = OtoWaveformEditorComponent::forNote(project, noteId, error);
    if (editor == nullptr)
    {
        if (error.isNotEmpty()) showError(error);
        return;
    }
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = utf8("单独OTO编辑（只作用于此音符） — ")
        + editor->originalEntry().sourceName;
    options.dialogBackgroundColour = Palette::panel;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = true;
    attachOtoPlayback(*editor);
    options.content.setOwned(editor.release());
    if (auto* window = noteOtoWindows.show(options,noteId))
        window->setResizeLimits(720, 400, 1800, 1100);
}

void MainComponent::attachOtoPlayback(OtoWaveformEditorComponent& editor)
{
    otoPreviews.erase(std::remove_if(otoPreviews.begin(),otoPreviews.end(),[](const auto& e){return e==nullptr;}),otoPreviews.end());
    if(std::none_of(otoPreviews.begin(),otoPreviews.end(),[&](const auto& e){return e.getComponent()==&editor;}))otoPreviews.emplace_back(&editor);
    juce::Component::SafePointer<MainComponent> safe(this);
    OtoWaveformEditorComponent::PlaybackHost host;
    host.devices = [safe]() -> juce::AudioDeviceManager*
    {
        return safe != nullptr ? &safe->audio.devices() : nullptr;
    };
    host.beforeStart = [safe]
    {
        if (safe == nullptr) return false;
        // Hearing the recording is not hearing the song: the song stops rather
        // than playing on underneath it.
        safe->stopOtoPreviews();
        safe->playWhenRenderReady = false;
        if (safe->audio.isPlaying()) safe->audio.stop();
        juce::String deviceError;
        if (!safe->audio.ensureOutputDevice(deviceError))
        {
            safe->showError(safe->strings.text("settings.noAudioDevice") + "\n" + deviceError);
            return false;
        }
        return true;
    };
    editor.setPlaybackHost(std::move(host));
}

void MainComponent::refreshAfterVoicebankChange()
{
    const auto revision=SampleSettings::voicebankFilesRevision();
    if(revision==seenVoicebankFilesRevision)return;
    seenVoicebankFilesRevision=revision;
    // OTO files are external to the document. Refresh views and render requests
    // on the next message turn without creating an edit or clearing selection.
    project.sendChangeMessage();
}

void MainComponent::startPreparedPlayback()
{
    stopOtoPreviews();
    audio.play();
}

void MainComponent::stopOtoPreviews()
{
    for(auto& editor:otoPreviews)if(editor!=nullptr)editor->stopPlaybackPreview();
    otoPreviews.erase(std::remove_if(otoPreviews.begin(),otoPreviews.end(),[](const auto& e){return e==nullptr;}),otoPreviews.end());
}

void MainComponent::noteLabelCommitted(const juce::String& noteId)
{
    const auto data = project.snapshot();
    const TrackData* owner = nullptr;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (std::any_of(clip.notes.begin(), clip.notes.end(),
                [&noteId](const auto& note) { return note.id == noteId; }))
            {
                owner = &track;
                break;
            }
    if (owner == nullptr) return;

    selectedTrackId = owner->id;
    trackList.setSelectedTrack(owner->id);
    pianoRoll.setFocusedTrack(owner->id);
    // Editing a label is never permission to change the renderer or silently
    // bind a global voicebank. Material assignment has one explicit manager entry.
    if (trackUsesVoicebankSynthesis(*owner) && !owner->voicebankDirectory.isDirectory())
        statusLabel.setText(utf8("请在素材管理器中选择素材文件夹并用于当前轨道。"),
                            juce::dontSendNotification);
    refreshProjectControls();
}

void MainComponent::showAdvancedEnvelopeMenu()
{
    if(const auto selected=pianoRoll.selectedCvvcEnvelope())
    {
        const auto initial=backend::cvvcEnvelopeSettings(selected->points);
        auto* panel=new AdvancedEnvelopePanel({{selected->noteId,selected->label,{0,selected->to-selected->from},
            {{0,0,true},{selected->to-selected->from,0,true}}, {},{}}},1,initial.mode,initial.shape,false,
            preferences->getFile().getSiblingFile("cvvc-envelope-presets.json"),false,true);
        auto revision=std::make_shared<std::uint64_t>(project.revisionNumber());
        auto applied=std::make_shared<backend::NativeEnvelopeSettings>(initial);
        juce::Component::SafePointer<MainComponent> safe(this);
        panel->onApply=[safe,revision,applied,target=*selected](int mode,const backend::TailFadeSettings& shape,bool audition)->juce::String
        {
            if(!safe)return utf8("编辑器已关闭。");
            if(safe->project.revisionNumber()!=*revision)return utf8("工程已发生其他修改。请重新打开窗口更新音素范围。");
            backend::NativeEnvelopeSettings next;next.mode=mode;next.shape=shape;next.shape.head={};
            if(next!=*applied)
            {
                const auto curve=backend::cvvcEnvelopeFromSettings(next);
                safe->project.setCvvcEnvelope(target.noteId,target.part,{},curve,*revision);
                *revision=safe->project.revisionNumber();*applied=next;
            }
            safe->setEnvelopeLane(EnvelopeLane::amplitude);safe->syncAudio(safe->project.snapshot());
            if(audition){safe->audio.stop();safe->audio.setPosition(std::max(0.0,target.from));safe->togglePlayback();}
            const auto message=utf8("已应用到音素 ")+target.label+utf8("；支持撤销。");
            safe->statusLabel.setText(message,juce::dontSendNotification);return message;
        };
        juce::DialogWindow::LaunchOptions options;options.dialogTitle=utf8("高级包络 — ")+selected->label;
        options.dialogBackgroundColour=Palette::panel;options.escapeKeyTriggersCloseButton=true;
        options.useNativeTitleBar=true;options.resizable=true;options.content.setOwned(panel);
        if(auto* window=options.launchAsync())window->setResizeLimits(740,660,1400,1000);
        return;
    }
    const auto selected=pianoRoll.selectedNoteIds();const auto data=project.snapshot();
    const auto native=std::any_of(data.tracks.begin(),data.tracks.end(),[&](const auto& t){return trackShowsAllNativeRegions(t)&&std::any_of(t.clips.begin(),t.clips.end(),[&](const auto& c){return std::any_of(c.notes.begin(),c.notes.end(),[&](const auto& n){return std::find(selected.begin(),selected.end(),n.id)!=selected.end();});});});
    std::vector<AdvancedEnvelopePanel::PreviewNote> previews;std::vector<juce::String> ids,validIds;
    int mode=0;backend::TailFadeSettings initial;bool first=true,mixed=false;double seek=std::numeric_limits<double>::max();
    for(const auto& track:data.tracks)for(const auto& clip:track.clips)for(const auto& note:clip.notes)
        if(std::find(selected.begin(),selected.end(),note.id)!=selected.end()&&(native?trackShowsAllNativeRegions(track):track.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(track)))
        {
            ids.push_back(note.id);
            const auto span=native?std::optional<backend::UtauTailFadeSpan>{{0,note.durationSeconds}}:pianoRoll.tailFadeSpanFor(note.id),head=native?std::nullopt:pianoRoll.headEnvelopeSpanFor(note.id);
            if(span||head)
            {
                const auto nextMode=native?note.nativeEnvelope.mode:note.utauTailFadeMode;
                const auto nextSettings=native?note.nativeEnvelope.shape:note.utauTailFade;
                if(first){mode=nextMode;initial=nextSettings;first=false;}
                else if(mode!=nextMode||initial!=nextSettings)mixed=true;
                previews.push_back({note.id,note.label,span.value_or(backend::UtauTailFadeSpan{}),pianoRoll.tailFadeBaseEnvelope(note.id),native?std::vector<OtoRegionGuide>{}:pianoRoll.otoRegionGuidesFor(note.id),head});validIds.push_back(note.id);
                seek=std::min(seek,clip.startSeconds+note.startSeconds+std::min(0.0,previews.back().base.empty()?0.0:previews.back().base.front().timeSeconds));
            }
        }
    if(previews.empty()){showError(utf8("请选择原生音频片段，或具有有效 OTO 区域的 UTAU 音符。"));return;}
    auto* panel=new AdvancedEnvelopePanel(std::move(previews),(int)selected.size(),mode,initial,mixed,preferences->getFile().getSiblingFile(native?"native-envelope-presets.json":"tail-fade-presets.json"),native);
    auto revision=std::make_shared<std::uint64_t>(project.revisionNumber());juce::Component::SafePointer<MainComponent> safe(this);
    panel->onApply=[safe,revision,ids,validIds,seek,native](int next,const backend::TailFadeSettings& settings,bool audition)->juce::String
    {
        if(!safe)return utf8("编辑器已关闭。");
        if(safe->project.revisionNumber()!=*revision)return utf8("工程已发生其他修改。请关闭并重新打开此窗口，以更新预览和选区。");
        const auto& targets=next==0&&settings.head.mode==0?ids:validIds;
        if(native){backend::NativeEnvelopeSettings edited;edited.mode=next;edited.shape=settings;edited.shape.head={};safe->project.setNotesNativeEnvelope(targets,edited);}
        else safe->project.setNotesTailFade(targets,next,settings);
        *revision=safe->project.revisionNumber();
        safe->setEnvelopeLane(EnvelopeLane::amplitude);safe->syncAudio(safe->project.snapshot());
        if(audition){safe->audio.stop();safe->audio.setPosition(std::max(0.0,seek));safe->togglePlayback();}
        const auto message=utf8("已应用到 ")+juce::String((int)targets.size())+utf8(" 个音符；支持撤销。");
        safe->statusLabel.setText(message,juce::dontSendNotification);return message;
    };
    juce::DialogWindow::LaunchOptions options;options.dialogTitle=utf8(native?"音频包络编辑":"高级包络 — OTO 音头与音尾");options.dialogBackgroundColour=Palette::panel;
    options.escapeKeyTriggersCloseButton=true;options.useNativeTitleBar=true;options.resizable=true;options.content.setOwned(panel);
    if(auto* window=options.launchAsync())window->setResizeLimits(740,660,1400,1000);
}

void MainComponent::closeEnvelopeLanes()
{
    // The state and both buttons, without touching the tool: the callers
    // are choosing a tool of their own and set it themselves.
    envelopeLane = EnvelopeLane::none;
    advancedEnvelopeButton.setToggleState(false,juce::dontSendNotification);
    utauAmplitudeEnvelopeActive = false;
    if (pianoRoll.diffSingerFlagContext) flagCurveButton.setToggleState(false, juce::dontSendNotification);
    volumeParamButton.setToggleState(false, juce::dontSendNotification);
    flagEnvelopeButton.setToggleState(false, juce::dontSendNotification);
}

MainComponent::EnvelopeLane MainComponent::nextEnvelopeLane(EnvelopeLane open,
                                                            EnvelopeLane clicked)
{
    if (clicked == EnvelopeLane::none) return EnvelopeLane::none;
    // The button for the lane that is already open closes it; any other
    // button opens its own, which closes whatever was open before.
    return open == clicked ? EnvelopeLane::none : clicked;
}

void MainComponent::setEnvelopeLane(EnvelopeLane lane)
{
    envelopeLane = lane;
    utauAmplitudeEnvelopeActive = lane == EnvelopeLane::amplitude;
    if (lane != EnvelopeLane::none) setSourceEditMode(false);
    switch (lane)
    {
        case EnvelopeLane::amplitude:
            pianoRoll.setTool(PianoRollComponent::Tool::amplitude);
            setToolButton(volumeParamButton);
            parameterMode = ParameterMode::volume;
            break;
        case EnvelopeLane::flagCurve:
            pianoRoll.setTool(PianoRollComponent::Tool::flagCurve);
            setToolButton(flagEnvelopeButton);
            break;
        case EnvelopeLane::none:
            pianoRoll.setTool(PianoRollComponent::Tool::note);
            setToolButton(noteEditButton);
            break;
    }
    // Both buttons, every time: the one that was not clicked is exactly the
    // one that used to be left showing a lane it no longer had.
    advancedEnvelopeButton.setToggleState(false,juce::dontSendNotification);
    volumeParamButton.setToggleState((isUtauAlgorithmSelected() || displayedTrackType == TrackType::diffSinger)
                                     && lane == EnvelopeLane::amplitude,
                                     juce::dontSendNotification);
    flagEnvelopeButton.setToggleState(lane == EnvelopeLane::flagCurve,
                                      juce::dontSendNotification);
    refreshSelectedNoteParameter();
}

MainComponent::WheelAction MainComponent::wheelActionFor(
    const juce::ModifierKeys& modifiers)
{
    return editorWheelActionFor(modifiers);
}

std::vector<int> MainComponent::stretchAlgorithmItemsFor(int pitchAlgorithmItemId)
{
    // NSF also offers WCSNDM-style forward LLSM parameter interpolation.
    // Loop/SoundTouch are not NSF algorithms.
    if (nsfPickerItem(pitchAlgorithmItemId)) return { 1, 2, 5, 6, 7 };
    // vslib is the Signalsmith stretcher itself, so all three of its analysis
    // clocks are real -- and the two NSF orders mean nothing to it.
    if (pitchAlgorithmItemId == 4) return { 1, 3, 4 };
    return { 1 }; // Backend-native time mapping; no NSF-only Mel choices.
}

void MainComponent::addTrackFromMenu(TrackType type)
{
    const auto id=project.addTypedTrack(type);
    if(id.isEmpty())return;
    pianoRoll.cancelPendingEditingForTrackChange();
    selectedClipId.clear();selectedNoteId.clear();selectedTrackId=id;
    trackList.setSelectedTrack(id);pianoRoll.setFocusedTrack(id);
    refreshProjectControls();menuItemsChanged();
}
void MainComponent::addTrackFromMenu(bool compose)
{
    if(compose){addTrackFromMenu(TrackType::hachi);return;}
    selectedTrackId=project.addTrack(strings.text("track.audio"),false);
    trackList.setSelectedTrack(selectedTrackId);pianoRoll.setFocusedTrack(selectedTrackId);refreshProjectControls();menuItemsChanged();
}

bool MainComponent::selectedTrackIsAccompaniment() const
{
    const auto data = project.snapshot();
    return std::any_of(data.tracks.begin(), data.tracks.end(), [this](const auto& track)
        { return track.id == selectedTrackId && track.accompaniment; });
}

void MainComponent::addAccompanimentTrackFromMenu()
{
    selectedTrackId = project.addTrack(strings.text("track.accompaniment"), false, false, true);
    trackList.setSelectedTrack(selectedTrackId);
    pianoRoll.setFocusedTrack(selectedTrackId);
    refreshProjectControls();
    menuItemsChanged();
}

void MainComponent::addReferenceTrackFromMenu()
{
    // An audio track: material is a recording to work against, not a part to
    // be composed.
    selectedTrackId = project.addTrack(strings.text("track.newReference"), false, true);
    trackList.setSelectedTrack(selectedTrackId);
    refreshProjectControls();
    menuItemsChanged();
}

void MainComponent::syncAudio(const ProjectData& data)
{
    audio.setAuditionTrack(selectedTrackId);
    if (exportWaitingForRender && pendingExportOptions.component != WavExportComponent::full)
    {
        // Only prepare the chosen targets. Unrelated DS/accompaniment tracks
        // must neither disable a single-track export nor enter its stem mix.
        const auto scoped = projectForExport(data, pendingExport, pendingExportRange);
        audio.syncProject(scoped, true, pendingExportOptions.component);
    }
    else audio.syncProject(data, exportWaitingForRender || mcpExportInProgress());
}

juce::String MainComponent::selectionAfterRemoving(const std::vector<TrackData>& tracks,
                                                   const juce::String& removedId)
{
    for (std::size_t index = 0; index < tracks.size(); ++index)
        if (tracks[index].id == removedId)
        {
            if (index + 1 < tracks.size()) return tracks[index + 1].id;
            if (index > 0) return tracks[index - 1].id;
            return {};
        }
    return removedId;
}

void MainComponent::deleteSelectedTrack()
{
    const auto trackId = selectedTrackId;
    if (trackId.isEmpty()) return;
    confirmDestructive(strings.text("track.delete"), [this, trackId]
    {
        const auto next = selectionAfterRemoving(project.snapshot().tracks, trackId);
        project.removeTrack(trackId);
        selectedTrackId = next;
        trackList.setSelectedTrack(next);
        pianoRoll.setFocusedTrack(next);
        refreshProjectControls();
        menuItemsChanged();
    });
}

namespace
{
// The dropdown's contents, in order.  A menu id is a position in this list,
// so what is offered and what a click means come from the same place and
// cannot drift apart when an entry is added or moved.
struct ViewMenuEntry
{
    const char* text;
    bool MainComponent::ViewOptions::* flag;
    bool voicebankOnly;
};
const std::array<ViewMenuEntry, 9> viewMenuEntries {{
    { "native.range", &MainComponent::ViewOptions::noteRange, false },
    { "native.envelope", &MainComponent::ViewOptions::envelope, false },
    // Named for what it draws rather than kept as the button's 波形显示: the
    // View menu already offers 显示波形 for the clip's own waveform, and two
    // near identical names for two different pictures is a trap.
    { "native.renderedWave", &MainComponent::ViewOptions::utauWaveform, false },
    { "native.pitchLine", &MainComponent::ViewOptions::pitchLine, false },
    { "native.actualWave", &MainComponent::ViewOptions::nativeRenderedWaveform, false },
    { "view.showLyrics", &MainComponent::ViewOptions::lyrics, false },
    { "native.originalPitchLine", &MainComponent::ViewOptions::originalPitchLine, false },
    { "native.consonantPitchDashed", &MainComponent::ViewOptions::consonantPitchDashed, false },
    { "view.phonemeInfo", &MainComponent::ViewOptions::phonemeInfo, true },
}};
}

MainComponent::ViewOptions MainComponent::viewOptionsFrom(const juce::PropertySet& properties)
{
    ViewOptions options;
    options.noteRange = properties.getBoolValue("ui.showNoteRange", true);
    options.envelope = properties.getBoolValue("ui.nativeEnvelope", true);
    options.utauWaveform = properties.getBoolValue("ui.showUtauWaveforms", false);
    options.nativeWaveform = properties.getBoolValue("ui.showNativeWaveforms", true);
    options.nativeRenderedWaveform = properties.getBoolValue("ui.showNativeRenderedWaveforms", false);
    options.pitchLine = properties.getBoolValue("ui.showPitchLine", true);
    options.originalPitchLine = properties.getBoolValue("ui.showOriginalPitchLine", true);
    options.consonantPitchDashed = properties.getBoolValue("ui.consonantPitchDashed", false);
    options.lyrics = properties.getBoolValue("ui.showLyrics", true);
    options.phonemeInfo = properties.getBoolValue("ui.showPhonemeInfo", false);
    return options;
}

void MainComponent::storeViewOptions(juce::PropertySet& properties,
                                     const ViewOptions& options)
{
    properties.setValue("ui.showNoteRange", options.noteRange);
    properties.setValue("ui.showEnvelope", options.envelope);
    properties.setValue("ui.nativeEnvelope", options.envelope);
    properties.setValue("ui.showUtauWaveforms", options.utauWaveform);
    properties.setValue("ui.showNativeWaveforms", options.nativeWaveform);
    properties.setValue("ui.showNativeRenderedWaveforms", options.nativeRenderedWaveform);
    properties.setValue("ui.showPitchLine", options.pitchLine);
    properties.setValue("ui.showOriginalPitchLine", options.originalPitchLine);
    properties.setValue("ui.consonantPitchDashed", options.consonantPitchDashed);
    properties.setValue("ui.showLyrics", options.lyrics);
    properties.setValue("ui.showPhonemeInfo", options.phonemeInfo);
}

bool MainComponent::viewMenuItemEnabled(int chosen, TrackType type)
{
    const auto index = static_cast<std::size_t>(chosen - 1);
    if (chosen <= 0 || index >= viewMenuEntries.size()) return false;
    if (chosen == 5 || chosen == 8) return type == TrackType::hachi;
    // DiffSinger has its own real phoneme overlay; Hachi source labels are
    // independent material annotations, not the voicebank spelling overlay.
    return type != TrackType::hachi || !viewMenuEntries[index].voicebankOnly;
}

bool MainComponent::viewMenuItemEnabled(int chosen, bool utauEditorActive)
{
    return viewMenuItemEnabled(chosen, utauEditorActive ? TrackType::utau : TrackType::hachi);
}

MainComponent::ViewOptions MainComponent::afterViewMenuChoice(ViewOptions options,
                                                              int chosen, TrackType type)
{
    const auto index = static_cast<std::size_t>(chosen - 1);
    if (viewMenuItemEnabled(chosen, type))
    {
        const auto flag = chosen == 3 && type == TrackType::hachi
            ? &ViewOptions::nativeWaveform : viewMenuEntries[index].flag;
        options.*flag = !(options.*flag);
    }
    return options;
}

MainComponent::ViewOptions MainComponent::afterViewMenuChoice(ViewOptions options,
                                                              int chosen, bool utauEditorActive)
{
    return afterViewMenuChoice(options, chosen, utauEditorActive ? TrackType::utau : TrackType::hachi);
}

TrackType MainComponent::selectedEditorTrackType() const
{
    const auto data = project.snapshot();
    const auto selected = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this](const auto& track) { return track.id == selectedTrackId; });
    return selected != data.tracks.end() ? effectiveTrackType(*selected) : TrackType::hachi;
}

void MainComponent::applyViewOptions()
{
    pianoRoll.setShowNoteRange(viewOptions.noteRange);
    pianoRoll.setShowEnvelope(viewOptions.envelope);
    pianoRoll.setShowUtauWaveforms(viewOptions.utauWaveform);
    pianoRoll.setShowNativeWaveforms(viewOptions.nativeWaveform);
    pianoRoll.setShowNativeRenderedWaveforms(viewOptions.nativeRenderedWaveform);
    timeline.setShowNativeRenderedWaveforms(viewOptions.nativeRenderedWaveform);
    pianoRoll.setShowPitchLine(viewOptions.pitchLine);
    pianoRoll.setShowOriginalPitchLine(viewOptions.originalPitchLine);
    pianoRoll.setConsonantPitchDashed(viewOptions.consonantPitchDashed);
    pianoRoll.setShowLyrics(viewOptions.lyrics);
    pianoRoll.setShowPhonemeInfo(viewOptions.phonemeInfo);
}

void MainComponent::refreshPitchScopeTitle()
{
    const bool track = !sourceEditActive && pianoRoll.pitchScope() == PianoRollComponent::PitchScope::currentTrack;
    parameterTitle.setText(track ? (strings.text("Track pitch"))
        : strings.text("editor.parameters"), juce::dontSendNotification);
    parameterTitle.setTooltip(track
        ? strings.text("Whole-track pitch editing across all materials and clips; gaps are not connected automatically.")
        : strings.text("Clip-focused pitch editing; switch to whole-track scope in View."));
}

void MainComponent::appendEditorViewItems(juce::PopupMenu& menu)
{
    const auto type = selectedEditorTrackType();
    const bool voicebank = type != TrackType::hachi;
    for (std::size_t index = 0; index < viewMenuEntries.size(); ++index)
    {
        const auto chosen = static_cast<int>(index) + 1;
        menu.addItem(600 + chosen,
            strings.text(chosen == 3 && !voicebank ? "native.sourceWave" : viewMenuEntries[index].text),
            viewMenuItemEnabled(chosen, type),
            chosen == 3 && !voicebank ? viewOptions.nativeWaveform : viewOptions.*(viewMenuEntries[index].flag));
    }
    menu.addSeparator();
    menu.addItem(trackPitchScopeMenuItem, strings.text("Whole current track pitch edit"), !sourceEditActive,
        pianoRoll.pitchScope() == PianoRollComponent::PitchScope::currentTrack);
    menu.addItem(613, strings.text("Show note overlap lines"), true, pianoRoll.showsOverlap());
}

void MainComponent::applyEditorViewChoice(int chosen)
{
    if(chosen==13) // Existing View menu item 613; display state only.
    {
        pianoRoll.setShowOverlap(!pianoRoll.showsOverlap());
        if(preferences)preferences->setValue("ui.showOverlap",pianoRoll.showsOverlap());
        menuItemsChanged();repaint();return;
    }
    if (chosen == trackPitchScopeMenuItem - 600)
    {
        if (sourceEditActive) return;
        const bool track = pianoRoll.pitchScope() != PianoRollComponent::PitchScope::currentTrack;
        pianoRoll.setPitchScope(track ? PianoRollComponent::PitchScope::currentTrack : PianoRollComponent::PitchScope::focusedClip);
        if (preferences) preferences->setValue("ui.pitchScope", track ? "track" : "clip");
        const auto selected = pianoRoll.selectedNoteIds();
        selectedNoteId = selected.empty() ? juce::String{} : selected.front();
        refreshPitchScopeTitle(); refreshSpliceButton(); menuItemsChanged(); repaint();
        return;
    }
    const auto type = selectedEditorTrackType();
    if (!viewMenuItemEnabled(chosen, type)) return;
    viewOptions = afterViewMenuChoice(viewOptions, chosen, type);
    applyViewOptions();
    if (preferences != nullptr) storeViewOptions(*preferences, viewOptions);
    menuItemsChanged();
}

void MainComponent::setDrawLengthDivision(int division)
{
    pianoRoll.setDrawLengthDivision(division);
    if (preferences != nullptr)
        preferences->setValue("ui.drawLengthDivision", pianoRoll.drawLengthDivision());
}

juce::PopupMenu MainComponent::pitchPointModeMenu() const
{
    juce::PopupMenu menu;
    menu.addSectionHeader(utf8("标点编辑模式"));
    menu.addItem(1, utf8("普通标点编辑"), true, !pianoRoll.diffSingerPitchRestoreMode() && !pianoRoll.diffSingerPitchOffsetMode());
    menu.addItem(2, utf8("还原至初始音高线（DS）"), pianoRoll.diffSingerPitchRestoreAvailable(),
        pianoRoll.diffSingerPitchRestoreMode());
    menu.addItem(4, utf8("绘制偏移（DS）"), pianoRoll.diffSingerPitchOffsetAvailable(), pianoRoll.diffSingerPitchOffsetMode());
    if (!pianoRoll.diffSingerPitchRestoreAvailable())
        menu.addItem(3, utf8("需在 DS 轨道先生成 pitch，保存原始虚线参考"), false);
    return menu;
}

void MainComponent::selectPitchPointMode(int chosen)
{
    if (chosen != 1 && chosen != 2 && chosen != 4) return;
    if (chosen == 4 && !pianoRoll.diffSingerPitchOffsetAvailable()) return;
    if (chosen == 2 && !pianoRoll.diffSingerPitchRestoreAvailable()) return;
    closeEnvelopeLanes(); setSourceEditMode(false);
    pianoRoll.setTool(PianoRollComponent::Tool::points);
    viewOptions.pitchLine = true;
    applyViewOptions();
    if (preferences != nullptr) storeViewOptions(*preferences, viewOptions);
    pianoRoll.setDiffSingerPitchRestoreMode(chosen == 2);
    pianoRoll.setDiffSingerPitchOffsetMode(chosen == 4);
    setToolButton(pointButton);
}

void MainComponent::showPitchPointModeMenu(juce::Point<int> screenPosition)
{
    auto menu = pitchPointModeMenu(); menu.setLookAndFeel(&getLookAndFeel());
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        {screenPosition.x, screenPosition.y, 1, 1}),
        [safe = juce::Component::SafePointer<MainComponent>(this)](int chosen)
        { if (safe != nullptr) safe->selectPitchPointMode(chosen); });
}

juce::PopupMenu MainComponent::flagEditModeMenu() const
{
    juce::PopupMenu menu;
    if (pianoRoll.diffSingerPitchOffsetAvailable()) {
        menu.addSectionHeader(utf8("DS 参数层"));
        menu.addItem(10,utf8("实参曲线（预测值 / 直接编辑）"));
        menu.addItem(11,utf8("FLAG 偏移 / 其他控制"));
        menu.addItem(12,utf8("局部重算实参（选中音符；未选则整轨）"),!diffSingerBusy);
        menu.addItem(14,utf8("重新生成整轨实参"),!diffSingerBusy);
        menu.addItem(13,utf8("清除实参，恢复自动预测（选中音符；未选则整轨）"),!diffSingerBusy);
        menu.addSeparator();
    }
    menu.addSectionHeader(utf8("FLAG 编辑模式"));
    menu.addItem(1, utf8("逐音符（控制点）"), true,
        pianoRoll.flagEditMode() == PianoRollComponent::FlagEditMode::perNote);
    menu.addItem(2, utf8("跨音符（连续绘制）"), true,
        pianoRoll.flagEditMode() == PianoRollComponent::FlagEditMode::continuous);
    menu.addSeparator();
    menu.addItem(3, utf8("连续绘制：当前轨道已开启 FLAG 的音符；Esc 取消本笔"), false);
    return menu;
}

void MainComponent::selectFlagEditMode(int chosen)
{
    if (chosen>=10 && (!pianoRoll.diffSingerPitchOffsetAvailable() || diffSingerBusy)) return;
    if (chosen==14) {generateDiffSingerParameters(true);return;}
    if (chosen==10) {showDiffSingerParameterLane(); return;}
    if (chosen==11) {pianoRoll.setDiffSingerParameterLayer(false);setEnvelopeLane(EnvelopeLane::flagCurve);return;}
    if (chosen==12) {generateDiffSingerParameters();return;}
    if (chosen==13) {
        const auto selected=pianoRoll.selectedNoteIds();
        std::map<juce::String,std::vector<FlagCurve>> empty;
        for(const auto& t:project.snapshot().tracks) if(t.id==selectedTrackId && trackIsDiffSinger(t))
            for(const auto& c:t.clips) for(const auto& n:c.notes)
                if(selected.empty() || std::find(selected.begin(),selected.end(),n.id)!=selected.end()) empty[n.id]={};
        project.applyDiffSingerParameters(project.revisionNumber(),empty); return;
    }
    if (chosen != 1 && chosen != 2 && chosen != 4) return;
    if (chosen == 4 && !pianoRoll.diffSingerPitchOffsetAvailable()) return;
    pianoRoll.setFlagEditMode(chosen == 2 ? PianoRollComponent::FlagEditMode::continuous
                                         : PianoRollComponent::FlagEditMode::perNote);
    if (preferences != nullptr)
    {
        preferences->setValue("ui.continuousFlag", chosen == 2);
        preferences->saveIfNeeded();
    }
    statusLabel.setText(utf8(chosen == 2
        ? "FLAG 连续模式：拖动绘制当前轨道已开启 FLAG 的音符；Esc 取消本笔"
        : "FLAG 逐音符模式：单击加点、拖动控制点、右键编辑数值"), juce::dontSendNotification);
}

juce::PopupMenu MainComponent::autoCrossfadeModeMenu() const
{
    juce::PopupMenu menu;
    menu.addItem(1,utf8("直线"),true,!autoCrossfadeCurve);
    menu.addItem(2,utf8("曲线"),true,autoCrossfadeCurve);
    return menu;
}

void MainComponent::selectAutoCrossfadeMode(int chosen)
{
    if(chosen!=1&&chosen!=2)return;
    autoCrossfadeCurve=chosen==2;
    autoCrossfadeButton.setTooltip(utf8(autoCrossfadeCurve
        ? "自动交叉淡化：曲线。右键选择直线／曲线；左键对同轨两个音符的已有重叠区生成淡化，不移动音符。"
        : "自动交叉淡化：直线。右键选择直线／曲线；左键对同轨两个音符的已有重叠区生成淡化，不移动音符。"));
    if(preferences && preferences->getBoolValue("ui.autoCrossfadeCurve",false)!=autoCrossfadeCurve)
    {
        preferences->setValue("ui.autoCrossfadeCurve",autoCrossfadeCurve);
        preferences->saveIfNeeded();
    }
}

void MainComponent::showFlagEditModeMenu(juce::Point<int> screenPosition)
{
    auto menu = flagEditModeMenu();
    menu.setLookAndFeel(&getLookAndFeel());
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
        {screenPosition.x, screenPosition.y, 1, 1}),
        [safe = juce::Component::SafePointer<MainComponent>(this)](int chosen)
        { if (safe != nullptr) safe->selectFlagEditMode(chosen); });
}

void MainComponent::showDrawSettingsMenu(juce::Point<int> screenPosition)
{
    juce::PopupMenu units;
    const auto offered = PianoRollComponent::drawLengthDivisions();
    const auto current = pianoRoll.drawLengthDivision();
    for (std::size_t index = 0; index < offered.size(); ++index)
        units.addItem(static_cast<int>(index) + 1,
                      "1/" + juce::String(offered[index]) + utf8(" 小节"),
                      true, offered[index] == current);

    juce::PopupMenu menu;
    menu.setLookAndFeel(&getLookAndFeel());
    const auto notes = pianoRoll.drawAction() == PianoRollComponent::DrawAction::notes;
    menu.addItem(200, strings.text("draw.action.pitch"), true, !notes);
    menu.addItem(201, strings.text("draw.action.notes"), true, notes);
    if (notes)
    {
        menu.addSeparator();
        menu.addSubMenu(strings.text("draw.noteLength"), units);
    }
    menu.showMenuAsync(
        juce::PopupMenu::Options().withTargetScreenArea(
            juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
        [safe = juce::Component::SafePointer<MainComponent>(this), offered](int chosen)
        {
            if (safe == nullptr) return;
            if (chosen == 200 || chosen == 201)
            {
                safe->pianoRoll.setDrawAction(chosen == 201 ? PianoRollComponent::DrawAction::notes
                    : PianoRollComponent::DrawAction::pitchCurve);
                if (safe->preferences) safe->preferences->setValue("ui.drawAction", chosen == 201 ? "notes" : "pitch-curve");
                safe->refreshTexts();
                return;
            }
            const auto index = static_cast<std::size_t>(chosen - 1);
            if (chosen > 0 && index < offered.size()) safe->setDrawLengthDivision(offered[index]);
        });
}

MainComponent::DeleteTarget MainComponent::deleteTargetFor(bool notesSelected,
                                                           bool clipSelected)
{
    if (notesSelected) return DeleteTarget::notes;
    if (clipSelected) return DeleteTarget::clip;
    return DeleteTarget::nothing;
}

void MainComponent::deleteSelectedClip()
{
    auto ids = timeline.selectedClipIds();
    if (ids.empty() && selectedClipId.isNotEmpty()) ids.push_back(selectedClipId);
    if (ids.empty()) return;
    const auto label = strings.text("clip.delete") + (ids.size() > 1 ? " (" + juce::String(static_cast<int>(ids.size())) + ")" : juce::String{});
    confirmDestructive(label, [this, ids] { project.removeClips(ids); });
}

void MainComponent::addOutputEngineMenu(juce::PopupMenu& menu, const juce::String& trackId)
{
    for (const auto& track : project.snapshot().tracks)
        if (track.id == trackId)
        {
            juce::PopupMenu engines;
            engines.addItem(101, strings.text("output.inherit"), true, track.outputEngine == UtauOutputEngine::inherit);
            engines.addItem(102, juce::String::fromUTF8("HiFisampler（PC-NSF-HiFiGAN）"), true, track.outputEngine == UtauOutputEngine::pcNsfHifigan);
            engines.addItem(103, strings.text("output.resampler"), true, track.outputEngine == UtauOutputEngine::resampler);
            menu.addSubMenu(strings.text("output.choose"), engines,
                trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track));
            menu.addSubMenu(strings.text("phonemizer.choose"), phonemizerMenu(track),
                trackUsesVoicebankSynthesis(track) && track.utauMode == UtauMode::classic && !trackIsDiffSinger(track));
            return;
        }
}

juce::PopupMenu MainComponent::phonemizerMenu(const TrackData& track)
{
    juce::PopupMenu menu;
    menu.addItem(104, strings.text("phonemizer.manual"), true, !track.chineseCvvc);
    menu.addItem(105, strings.text("phonemizer.zhCvvc"), true, track.chineseCvvc);
    juce::PopupMenu crossfade;
    crossfade.addItem(106,utf8("直线"),true,!track.cvvcCurveCrossfade);
    crossfade.addItem(107,utf8("曲线"),true,track.cvvcCurveCrossfade);
    menu.addSubMenu(utf8("默认交叉方式"),crossfade,track.chineseCvvc);
    return menu;
}

void MainComponent::showPhonemizerMenu()
{
    const auto data = project.snapshot();
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(),
        [this](const auto& t) { return t.id == selectedTrackId; });
    if (track == data.tracks.end() || !trackUsesVoicebankSynthesis(*track)
        || track->utauMode != UtauMode::classic || trackIsDiffSinger(*track)) return;
    auto menu = phonemizerMenu(*track);
    menu.setLookAndFeel(&getLookAndFeel());
    juce::Component::SafePointer<MainComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&phonemizerButton),
        [safe, trackId = track->id](int item)
        { if (safe != nullptr) safe->outputEngineItemChosen(item, trackId); });
}

bool MainComponent::outputEngineItemChosen(int result, const juce::String& trackId)
{
    if (result < 101 || result > 107) return false;
    const auto data = project.snapshot();
    const auto track = std::find_if(data.tracks.begin(), data.tracks.end(), [&](const auto& t) { return t.id == trackId; });
    if (track == data.tracks.end() || !trackUsesVoicebankSynthesis(*track) || trackIsDiffSinger(*track)) return true;
    if(result==106||result==107)
    {
        if(track->chineseCvvc&&track->utauMode==UtauMode::classic)project.setTrackCvvcCurveCrossfade(trackId,result==107);
        refreshProjectControls();return true;
    }
    if (result == 104 || result == 105)
    {
        if (track->utauMode != UtauMode::classic) return true;
        if (result == 105 && !track->voicebankDirectory.getChildFile("presamp.ini").existsAsFile())
        {
            showError(strings.text("phonemizer.missingPresamp"));
            return true;
        }
        project.setTrackChineseCvvc(trackId, result == 105);
        refreshProjectControls();
        return true;
    }
    if (result == 103) { showTrackResamplerSettings(trackId); return true; }
    if (result == 102 && !backend::NsfHifiganRenderer::modelAvailable(preferences != nullptr
        ? juce::File(preferences->getValue("algorithm.hifiganPath")) : juce::File{}))
    {
        showError(utf8("HiFisampler（PC-NSF-HiFiGAN）模型未找到，请在设置中指定模型目录。"));
        return true;
    }
    // Retain alternate tool choices when temporarily switching to native NSF.
    project.setTrackOutputEngine(trackId, result == 101 ? UtauOutputEngine::inherit : UtauOutputEngine::pcNsfHifigan,
        result == 101 ? juce::File{} : track->outputResampler, result == 101 ? juce::File{} : track->outputWavtool);
    refreshProjectControls();
    return true;
}

void MainComponent::showOutputEngineMenu(const juce::String& trackId, juce::Point<int> at)
{
    juce::PopupMenu menu;
    menu.setLookAndFeel(&getLookAndFeel());
    const auto insertSeconds = audio.position();
    for (const auto& track : project.snapshot().tracks)
        if (track.id == trackId)
        {
            menu.addItem(108, strings.text("track.importMidiHere"),
                track.compose && !track.accompaniment && !track.referenceOnly);
            menu.addSeparator();
            break;
        }
    addOutputEngineMenu(menu, trackId);
    juce::Component::SafePointer<MainComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea({ at.x, at.y, 1, 1 }),
        [safe, trackId, insertSeconds](int item)
        {
            if (safe == nullptr) return;
            if (item == 108) safe->importMidiTrack(trackId, insertSeconds);
            else safe->outputEngineItemChosen(item, trackId);
        });
}

void MainComponent::showTrackResamplerSettings(const juce::String& trackId)
{
    for (const auto& track : project.snapshot().tracks)
        if (track.id == trackId)
        {
            auto panel = std::make_unique<UtauOutputEnginePanel>(strings, track.name,
                track.outputResampler, track.outputWavtool);
            juce::Component::SafePointer<MainComponent> safe(this);
            panel->onApply = [safe, trackId](const auto& resampler, const auto& wavtool)
            {
                if (safe != nullptr) {
                    safe->project.setTrackOutputEngine(trackId, UtauOutputEngine::resampler, resampler, wavtool);
                    safe->refreshProjectControls();
                }
            };
            juce::DialogWindow::LaunchOptions options;
            options.dialogTitle = strings.text("output.resampler");
            options.dialogBackgroundColour = Palette::panel;
            options.content.setOwned(panel.release());
            options.componentToCentreAround = this;
            options.useNativeTitleBar = true;
            options.escapeKeyTriggersCloseButton = true;
            options.resizable = true;
            options.launchAsync();
            return;
        }
}

juce::PopupMenu MainComponent::clipContextMenu(const juce::String& clipId, double seconds)
{
    juce::PopupMenu menu;
    menu.setLookAndFeel(&getLookAndFeel());
    menu.addItem(1, strings.text("clip.split"), project.canSplitClip(clipId, seconds));
    menu.addItem(2, strings.text("clip.merge"), timeline.isClipSelected(clipId)
        && project.canMergeClips(timeline.selectedClipIds()));
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                menu.addItem(7, strings.text("clip.editLength"));
                menu.addItem(8, strings.text("clip.muteToggle"), true, clip.muted);
                menu.addSeparator();
                if (track.accompaniment)
                    menu.addItem(recogniseTempoMenuItem, strings.text("tempo.recognise"), AccompanimentTempoTask::canAnalyse(clip));
                menu.addItem(3, strings.text(clip.showNoteHints ? "clip.stopNoteHints" : "clip.startNoteHints"),
                    clip.showNoteHints || (track.compose && !track.accompaniment && !clip.notes.empty()));
                if (trackShowsAllNativeRegions(track))
                {
                    const auto sources = clipSourceRegions(clip);
                    menu.addItem(5, strings.text("clip.markUnpitched"),
                        std::any_of(sources.begin(), sources.end(),
                            [](const auto* part) { return part->sourceFile.existsAsFile(); }));
                }
                addOutputEngineMenu(menu, track.id);
                if (!trackShowsAllNativeRegions(track))
                    menu.addItem(4, strings.text(clip.showNormalDisplay ? "clip.stopNormalDisplay" : "clip.startNormalDisplay"),
                        clip.showNormalDisplay || (track.compose && !track.accompaniment));
            }
    return menu;
}

void MainComponent::clipContextMenuItemChosen(int result, const juce::String& clipId, double seconds)
{
    if (result == 7) { showClipLengthDialog(clipId); return; }
    if (result == recogniseTempoMenuItem) { showAccompanimentTempo(clipId); return; }
    if (result == 5)
    {
        project.markNativeUnpitchedRegions(clipId);
        focusClip(clipId);
        return;
    }
    if (result >= 101 && result <= 103)
    {
        for (const auto& track : project.snapshot().tracks)
            for (const auto& clip : track.clips)
                if (clip.id == clipId) { outputEngineItemChosen(result, track.id); return; }
        return;
    }
    if (result == 3 || result == 4 || result == 8)
    {
        const auto data = project.snapshot();
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                if (clip.id == clipId)
                {
                    if (result == 3) project.setClipNoteHints(clipId, !clip.showNoteHints);
                    else if (result == 8) project.setClipMuted(clipId, !clip.muted);
                    else project.setClipNormalDisplay(clipId, !clip.showNormalDisplay);
                    return;
                }
        return;
    }
    if (result == 2 && timeline.isClipSelected(clipId))
    {
        const auto merged = project.mergeClips(timeline.selectedClipIds());
        if (merged.isNotEmpty())
        {
            timeline.setSelectedClips({merged});
            focusClip(merged);
        }
        return;
    }
    if (result != 1) return;
    const auto right = project.splitClip(clipId, seconds);
    // The original (left) piece keeps its id and the timeline highlight.
    if (right.isNotEmpty()) focusClip(clipId);
}

std::unique_ptr<juce::AlertWindow> MainComponent::makeClipLengthDialog(const juce::String& clipId)
{
    for (const auto& track : project.snapshot().tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                return std::make_unique<ClipEditDialog>(strings, track, clip, this);
            }
    return {};
}

bool MainComponent::applyClipLengthText(const juce::String& clipId, const juce::String& text)
{
    const auto requested = clipLengthSeconds(text);
    if (!requested) return false;
    for (const auto& track : project.snapshot().tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                if (text.trim() == juce::String(clip.durationSeconds, 6)) return true;
                return project.setClipEditableLength(clipId, *requested);
            }
    return false;
}

void MainComponent::showClipLengthDialog(const juce::String& clipId)
{
    auto owned = makeClipLengthDialog(clipId);
    if (!owned) return;
    auto* dialog = owned.release();
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create([safe, dialog, clipId](int result)
    {
        juce::String error;
        const bool applied = safe == nullptr || result != 1 || safe->applyClipEditDialog(clipId, *dialog, error);
        delete dialog;
        if (safe != nullptr && !applied) safe->showError(error);
    }), false);
}

bool MainComponent::applyClipEditDialog(const juce::String& clipId, const juce::AlertWindow& window, juce::String& error)
{
    const auto* dialog = dynamic_cast<const ClipEditDialog*>(&window);
    error = strings.text("clip.lengthInvalid");
    if (!dialog || !clipLengthSeconds(dialog->getTextEditorContents("length"))) return false;
    for (const auto& track : project.snapshot().tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId && track.id == dialog->trackId)
            {
                const bool canChoose = dialog->canChooseOutput && trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track);
                const auto engine = dialog->selectedEngine();
                const auto resampler = engine == UtauOutputEngine::inherit ? juce::File{} : dialog->resampler;
                const auto wavtool = engine == UtauOutputEngine::inherit ? juce::File{} : dialog->wavtool;
                const bool outputChanged = canChoose && (engine != track.outputEngine
                    || (engine != UtauOutputEngine::inherit && (resampler != track.outputResampler || wavtool != track.outputWavtool)));
                // Validate both drafts before changing either the range or the engine.
                if (outputChanged && engine == UtauOutputEngine::pcNsfHifigan
                    && !backend::NsfHifiganRenderer::modelAvailable(preferences != nullptr
                        ? juce::File(preferences->getValue("algorithm.hifiganPath")) : juce::File{}))
                { error = utf8("HiFisampler（PC-NSF-HiFiGAN）模型未找到，请在设置中指定模型目录。"); return false; }
                if (outputChanged && engine == UtauOutputEngine::resampler
                    && ((resampler != juce::File{} && !resampler.existsAsFile()) || (wavtool != juce::File{} && !wavtool.existsAsFile())))
                { error = strings.text("output.invalidTool"); return false; }
                if (!applyClipLengthText(clipId, dialog->getTextEditorContents("length"))) return false;
                if (outputChanged) project.setTrackOutputEngine(track.id, engine, resampler, wavtool);
                refreshProjectControls(); error.clear(); return true;
            }
    return false;
}

void MainComponent::showAccompanimentTempo(const juce::String& clipId)
{
    // One model task at a time in this window. A second click raises its
    // existing result/progress window rather than starting duplicate work.
    if (tempoWindows.showExisting("accompaniment-tempo")) return;
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        if (track.accompaniment)
            for (const auto& clip : track.clips)
                if (clip.id == clipId && AccompanimentTempoTask::canAnalyse(clip))
                {
                    juce::DialogWindow::LaunchOptions options;
                    options.dialogTitle = strings.text("tempo.recognise");
                    options.dialogBackgroundColour = Palette::panel;
                    options.content.setOwned(new AccompanimentTempoPanel(strings, clip));
                    options.componentToCentreAround = this;
                    options.useNativeTitleBar = true;
                    options.escapeKeyTriggersCloseButton = true;
                    options.resizable = false;
                    tempoWindows.show(options, "accompaniment-tempo");
                    return;
                }
}

juce::PopupMenu MainComponent::trackAreaMenu(std::optional<TimelineComponent::Anchor> anchor)
{
    juce::PopupMenu menu;
    menu.setLookAndFeel(&getLookAndFeel());
    menu.addItem(1,strings.text("New HachiShifter track"));
    menu.addItem(861,strings.text("New UTAU track"));
    menu.addItem(862,strings.text("New DiffSinger track"));
    // Somewhere to keep a reference vocal or a backing take: audible while it
    // is the track in hand, silent whenever anything else is playing.
    menu.addItem(3, strings.text("track.newReference"));
    menu.addItem(accompanimentTrackMenuItem, strings.text("track.newAccompaniment"));
    menu.addItem(importMidiTrackMenuItem, strings.text("track.importMidi"));
    if (anchor)
    {
        menu.addSeparator();
        addOutputEngineMenu(menu, anchor->trackId);
        menu.addItem(emptyTuningClipMenuItem,strings.text("clip.addEmptyTuning"),
            project.canAddEmptyTuningClip(anchor->trackId,anchor->seconds));
        menu.addSeparator();
    }
    // Named for the selected track rather than for the click, because the
    // click landed on the space between tracks and points at none of them --
    // and a right-click there deliberately leaves the selection alone.
    menu.addItem(2, strings.text("track.delete"), selectedTrackId.isNotEmpty());
    return menu;
}

void MainComponent::trackAreaMenuItemChosen(int chosen, std::optional<TimelineComponent::Anchor> anchor)
{
    if (anchor && outputEngineItemChosen(chosen, anchor->trackId)) return;
    if (chosen == emptyTuningClipMenuItem)
    {
        if (!anchor) return;
        const auto clip = project.addEmptyTuningClip(anchor->trackId,anchor->seconds);
        if (clip.isEmpty()) return;
        // Refresh the lane snapshot before selecting the newly created id.
        project.dispatchPendingMessages();
        if (sourceEditActive) setSourceEditMode(false);
        pianoRoll.clearNoteSelection();selectedNoteId.clear();
        focusClip(clip);
        pianoViewport.setViewPosition(std::max(0,pianoRoll.pixelForSeconds(anchor->seconds)-58),pianoViewport.getViewPositionY());
        return;
    }
    // A melodic track, which is what the app makes by default and what
    // ProjectModel::addTrack assumes; the Track menu still offers the audio
    // kind explicitly for the times that is not what is wanted.
    if (chosen == 1) addTrackFromMenu(TrackType::hachi);
    else if(chosen==861)addTrackFromMenu(TrackType::utau);
    else if(chosen==862)addTrackFromMenu(TrackType::diffSinger);
    else if (chosen == 3) addReferenceTrackFromMenu();
    else if (chosen == accompanimentTrackMenuItem) addAccompanimentTrackFromMenu();
    else if (chosen == importMidiTrackMenuItem) importMidiTrack();
    else if (chosen == 2) deleteSelectedTrack();
}

void MainComponent::showTrackAreaMenu(juce::Point<int> screenPosition, std::optional<TimelineComponent::Anchor> anchor)
{
    juce::Component::SafePointer<MainComponent> safe(this);
    trackAreaMenu(anchor).showMenuAsync(
        juce::PopupMenu::Options().withTargetScreenArea(
            juce::Rectangle<int>(screenPosition.x, screenPosition.y, 1, 1)),
        [safe,anchor](int chosen) { if (safe != nullptr) safe->trackAreaMenuItemChosen(chosen,anchor); });
}

std::vector<MainComponent::MenuItemState> MainComponent::diagnosticTrackAreaMenu()
{
    std::vector<MenuItemState> items;
    auto menu = trackAreaMenu();
    for (juce::PopupMenu::MenuItemIterator item(menu); item.next();)
        if (item.getItem().itemID != 0)
            items.push_back({ item.getItem().itemID, item.getItem().text, item.getItem().isEnabled });
    return items;
}

void MainComponent::importMidiTrack(const juce::String& targetTrackId, double startSeconds)
{
    chooser = std::make_unique<juce::FileChooser>(strings.text("track.importMidi"), juce::File{},
                                                  "*.mid;*.midi");
    juce::Component::SafePointer<MainComponent> safe(this);
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles,
        [safe, targetTrackId, startSeconds](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (safe != nullptr && file != juce::File{}) safe->importMidiTrackFrom(file, targetTrackId, startSeconds);
        });
}

void MainComponent::importMidiTrackFrom(const juce::File& file, const juce::String& targetTrackId, double startSeconds)
{
    juce::String error;
    const auto choices = ProjectModel::midiTrackChoices(file, error);
    if (choices.empty())
    {
        showError(strings.text("error.midi") + "\n" + error);
        return;
    }
    // One track with notes -- a type 0 file, or a type 1 file whose other
    // track only holds the tempo -- leaves nothing to ask.
    if (choices.size() == 1 && targetTrackId.isEmpty())
    {
        addMidiTrackFrom(file, choices.front().index);
        return;
    }
    // By the name the track gives itself; one without a name by its place in
    // the file, which counts the tempo track too, as a sequencer shows it.
    juce::StringArray labels;
    for (const auto& choice : choices)
        labels.add((choice.name.isNotEmpty()
                        ? choice.name
                        : strings.text("dialog.midiTrackLabel") + " " + juce::String(choice.index + 1))
                   + "  (" + juce::String(choice.noteCount) + " "
                   + strings.text("dialog.midiTrackNotes") + ")");
    juce::String destinationLabel;
    if (targetTrackId.isNotEmpty())
        for (const auto& track : project.snapshot().tracks)
            if (track.id == targetTrackId)
                destinationLabel = "\n" + track.name + "  @ " + juce::String(startSeconds, 3) + " s";
    auto* dialog = new juce::AlertWindow(strings.text("dialog.midiTrackTitle"),
        file.getFileName() + "\n" + strings.text(targetTrackId.isEmpty()
            ? "dialog.midiTrackMessage" : "dialog.midiTrackHereMessage") + destinationLabel,
        juce::MessageBoxIconType::QuestionIcon);
    dialog->addComboBox("track", labels, strings.text("dialog.midiTrackLabel"));
    if (auto* box = dialog->getComboBoxComponent("track")) box->setSelectedItemIndex(0);
    dialog->addButton(strings.text("dialog.import"), 1,
                      juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, file, choices, targetTrackId, startSeconds](int result)
            {
                const auto* box = dialog->getComboBoxComponent("track");
                const auto picked = box != nullptr ? box->getSelectedItemIndex() : -1;
                if (safe != nullptr && result == 1 && picked >= 0
                    && picked < static_cast<int>(choices.size()))
                    safe->addMidiTrackFrom(file, choices[static_cast<std::size_t>(picked)].index, targetTrackId, startSeconds);
                delete dialog;
            }), false);
}

void MainComponent::addMidiTrackFrom(const juce::File& file, int trackIndex, const juce::String& targetTrackId, double startSeconds)
{
    juce::String error;
    const auto trackId = project.addMidiTrack(file, trackIndex, error, targetTrackId, startSeconds);
    if (trackId.isEmpty())
    {
        showError(strings.text("error.midi") + "\n" + error);
        return;
    }
    // Show what was just imported, as a UST import does.
    selectedTrackId = trackId;
    trackList.setSelectedTrack(trackId);
    selectedNoteId.clear();
    pianoRoll.clearNoteSelection();
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        if (track.id == trackId)
        {
            if (!track.clips.empty()) focusClip(targetTrackId.isEmpty() ? track.clips.front().id : track.clips.back().id);
            statusLabel.setText(strings.text("status.midiTrackImported") + track.name,
                                juce::dontSendNotification);
            break;
        }
    refreshProjectControls();
    menuItemsChanged();
}

void MainComponent::refreshStretchAlgorithmItems(int preferredId)
{
    const auto previous = preferredId > 0 ? preferredId : stretchAlgorithm.getSelectedId();
    const auto items = stretchAlgorithmItemsFor(pitchAlgorithm.getSelectedId());
    stretchAlgorithm.clear(juce::dontSendNotification);
    for (const auto item : items)
        stretchAlgorithm.addItem(strings.text(
            item == 2 ? "algo.stretch.nsfVariableMel"
            : item == 3 ? "algo.stretch.loop"
            : item == 4 ? "algo.stretch.soundTouch"
            : item == 7 ? "algo.stretch.wcsNonLoop"
            : item == 6 ? "algo.stretch.hifiShifterMel"
            : item == 5 ? "algo.stretch.nsfShiftThenSplice"
            : "algo.stretch.melodyneHybrid"), item);
    // Keep the current choice only if it survived into the new list.  It often
    // does not: switching backend can take away the very clock that was
    // selected, and the caller resets the track to Melodyne Hybrid when the
    // selection moves.  Melodyne Hybrid is item 1 everywhere, so a picker that
    // is about to be hidden still reports a sane, harmless id.
    const auto kept = previous > 0 && stretchAlgorithm.indexOfItemId(previous) >= 0;
    stretchAlgorithm.setSelectedId(kept ? previous : 1, juce::dontSendNotification);
    stretchAlgorithm.setTooltip(stretchAlgorithm.getSelectedId()==7
        ? strings.text("algo.stretch.wcsNonLoop") + "\n" + strings.text("algo.stretch.wcsNonLoopHelp")
        : strings.text("algo.stretch"));
}

void MainComponent::diagnosticPressTool(PianoRollComponent::Tool wanted)
{
    // Through the button's own handler, so the check follows the path a click
    // follows and not a shortcut around it.
    juce::Button* button = &noteEditButton;
    if (wanted == PianoRollComponent::Tool::points) button = &pointButton;
    else if (wanted == PianoRollComponent::Tool::draw) button = &drawButton;
    else if (wanted == PianoRollComponent::Tool::line) button = &lineButton;
    else if (wanted == PianoRollComponent::Tool::trim) button = &trimButton;
    if (button->onClick) button->onClick();
}

PianoRollComponent::Tool MainComponent::diagnosticTool() const
{
    return pianoRoll.currentTool();
}

void MainComponent::diagnosticRefreshControls()
{
    refreshProjectControls();
}

MainComponent::EnvelopePresetLayout MainComponent::diagnosticEnvelopePresetLayout(juce::Image* preview)
{
    // UTAU in the picker, which is what the row keys its UTAU tools off, and
    // the row laid out again.
    pitchAlgorithm.setSelectedId(7, juce::dontSendNotification);
    resized();
    EnvelopePresetLayout layout;
    if(envelopePresetMenu.isVisible())layout.picker=envelopePresetMenu.getBounds();
    for(int i=0;i<envelopePresetMenu.getNumItems();++i)layout.items.add(envelopePresetMenu.getItemText(i));
    if(preview!=nullptr)
    {
        *preview=juce::Image(juce::Image::RGB,332,68*envelopePresetMenu.getNumItems(),true);
        juce::Graphics g(*preview);
        int y=0;
        for(juce::PopupMenu::MenuItemIterator it(*envelopePresetMenu.getRootMenu());it.next();)
            if(auto* item=it.getItem().customComponent.get())
            {
                int width=0,height=0;item->getIdealSize(width,height);
                item->setLookAndFeel(&getLookAndFeel());item->setSize(width,height);
                g.drawImageAt(item->createComponentSnapshot(item->getLocalBounds(),true,2.0f),0,y);
                item->setLookAndFeel(nullptr);y+=height*2;
            }
    }
    layout.viewMenu = {}; // Display controls now live only in the main View menu.
    layout.rightControlsStart = pianoViewport.getRight() - 4;
    return layout;
}

bool MainComponent::diagnosticHasEditorViewItems()
{
    auto menu = getMenuForIndex(3, strings.text("menu.view"));
    std::set<int> found;
    for (juce::PopupMenu::MenuItemIterator it(menu); it.next();)
        if (it.getItem().itemID >= 601 && it.getItem().itemID <= 609) found.insert(it.getItem().itemID);
    return found.size() == 9;
}

void MainComponent::diagnosticRefreshTexts()
{
    refreshTexts();
}

bool MainComponent::diagnosticIntegratedLayout(const juce::File& directory)
{
    directory.createDirectory();
    const auto id = project.addTrack("Integrated UTAU", true);
    project.setTrackPitchAlgorithm(id, PitchAlgorithm::utau);
    const auto clip = project.addClip(id, 0.0, 4.0);
    for (int i = 0; i < 6; ++i)
    {
        const auto note = project.addNote(clip, 0.5 + i * 0.5, 0.5, 60.0f + (i % 3) * 2.0f);
        project.setNoteLabel(note, "a");
    }
    project.dispatchPendingMessages();
    pianoRoll.diagnosticRefresh();
    diagnosticSelectTrack(id);
    if (!assetManagerVisible) showAssetManager();
    auto ok = true;
    for (const auto& theme : { juce::String("dark"), juce::String("light") })
    {
        preferences->setValue("ui.theme", theme);
        applyPreferences();
        refreshTexts();
        for (const auto width : { 1280, 1600 })
        {
            setSize(width, 800);
            diagnosticSelectTrack(id);
            resized();
            const auto edge = assetManager->getX();
            const auto right=envelopePresetMenu.getRight();
            ok=ok && envelopePresetMenu.isVisible() && envelopePresetMenu.getWidth()==110
                && envelopePresetMenu.getX()>=flagEnvelopeButton.getRight() && right<=edge;
            ok = ok && diagnosticHasEditorViewItems()
                && flagCurveButton.getRight() <= edge && flagCurveButton.getWidth() > 0;
            auto file = directory.getChildFile("integrated-" + theme + "-" + juce::String(width) + ".png");
            file.deleteFile();
            if (auto stream = file.createOutputStream())
            {
                juce::PNGImageFormat png;
                ok = png.writeImageToStream(createComponentSnapshot(getLocalBounds()), *stream) && ok;
            }
            else ok = false;
        }
    }
    return ok;
}

bool MainComponent::diagnosticRenderOrderPicker()
{
    selectedTrackId = project.addTrack("NSF order UI test", true);
    project.setTrackPitchAlgorithm(selectedTrackId, PitchAlgorithm::nsfHifigan);
    setSize(1280, 760);
    refreshProjectControls();
    const auto visible = renderOrder.isVisible() && renderOrderLabel.isVisible()
        && renderOrder.getWidth() > 0 && renderOrder.getNumItems() == 3;
    auto works = visible;
    for (const auto id : { 1, 2, 3 })
    {
        renderOrder.setSelectedId(id, juce::dontSendNotification);
        if (renderOrder.onChange) renderOrder.onChange();
        const auto data = project.snapshot();
        const auto expected = id == 2 ? RenderOrder::stretchSpliceThenPitch : RenderOrder::processThenSplice;
        works = works && data.tracks.back().renderOrder == expected
            && data.tracks.back().mergeContiguousSource == (id == 3);
        refreshProjectControls();
        works = works && renderOrder.getSelectedId() == id;
    }
    return works;
}

void MainComponent::setToolButton(juce::Button& selected)
{
    const std::array<juce::Button*, 8> editTools {
        &noteEditButton, &wrenchButton, &drawButton, &lineButton, &pointButton, &trimButton, &nativeLoudnessButton, &connectButton
    };
    timeline.setClipStretchMode(false);
    pianoRoll.setNativeStretchMode(false);
    pianoRoll.setNativeLoudnessMode(&selected == &nativeLoudnessButton);
    for (auto* button : editTools)
        button->setToggleState(button == &selected, juce::dontSendNotification);
    const std::array<juce::Button*, 7> parameterTools {
        &pitchParamButton, &driftParamButton, &attackParamButton,
        &breathParamButton, &tensionParamButton,
        &formantParamButton, &volumeParamButton
    };
    for (auto* button : parameterTools)
        if (&selected == button || &selected == &pitchParamButton || &selected == &driftParamButton
            || &selected == &attackParamButton
            || &selected == &breathParamButton
            || &selected == &tensionParamButton || &selected == &formantParamButton
            || &selected == &volumeParamButton)
            button->setToggleState(button == &selected, juce::dontSendNotification);
}

std::optional<int> MainComponent::followViewPosition(
    int viewLeft, int viewWidth, int playheadX, int leftMargin,
    const std::optional<juce::Range<int>>& run)
{
    if (viewWidth <= 0) return {};
    const auto rightMargin = 56;
    // Repeating a phrase means hearing it again, and again after that.  If it
    // fits the window there is no reason to move at all once it is in view:
    // paging away at the end left the phrase behind and had to be dragged
    // back to before it could be played a second time.
    if (run && run->getLength() > 0
        && run->getLength() <= viewWidth - leftMargin - rightMargin)
    {
        if (run->getStart() >= viewLeft + leftMargin
            && run->getEnd() <= viewLeft + viewWidth - rightMargin)
            return {};
        return std::max(0, run->getStart() - leftMargin - 24);
    }
    if (playheadX < viewLeft + leftMargin
        || playheadX > viewLeft + viewWidth - rightMargin)
        return std::max(0, playheadX - viewWidth / 4);
    return {};
}

void MainComponent::refreshCollapseIcon()
{
    // The chevron points the way the arrangement is about to go.  Only this
    // knows which one that is: applyPreferences settles the state before the
    // constructor has finished naming icons, so a second place naming one
    // would put the wrong chevron back.
    collapseTracksButton.setComponentID(tracksCollapsed ? "icon.expand"
                                                        : "icon.collapse");
    collapseTracksButton.repaint();
}

void MainComponent::setTracksCollapsed(bool collapsed)
{
    tracksCollapsed = collapsed;
    refreshCollapseIcon();
    if (preferences != nullptr)
        preferences->setValue("ui.tracksCollapsed", collapsed);
    resized();
}

void MainComponent::armSelectionPlayback()
{
    // Playing a selection should finish with it.  Left to run, the transport
    // carried on through the rest of the piece and the view followed it, which
    // is not what asking to hear the selected part means.
    audio.setPlayUntil(0.0);
    playbackRun.reset();
    if (sourceEditActive) return;
    const auto span = pianoRoll.selectedNotesTimeSpan();
    // Starting at or past the end -- a marquee drawn right to left leaves the
    // playhead there -- would stop before a sound was made.
    if (span && span->getEnd() > audio.position() + 0.01)
    {
        // This is the marquee "export the last render" means: noted as play
        // begins, so stopping halfway through leaves it whole.  Pressing space
        // with nothing selected plays the render history instead and never
        // reaches here, which is why that does not count as one.
        lastRenderedNoteIds = pianoRoll.selectedNoteIds();
        lastRenderedSpan = *span;
        audio.setPlayUntil(span->getEnd());
        // What the view should try to keep in front of you: the selection, and
        // wherever the playhead was set if that is earlier still.
        playbackRun = juce::Range<double>(
            std::min(audio.position(), span->getStart()), span->getEnd());
    }
}

void MainComponent::togglePlayback()
{
    if (audio.isPlaying())
    {
        playWhenRenderReady = false;
        audio.stop();
        return;
    }
    stopOtoPreviews();
    if (!sourceEditActive && pianoRoll.selectedNoteIds().empty())
    {
        const auto data = project.snapshot();
        if (audio.selectAllRenderedUtauNotes(data)) syncAudio(data);
    }
    juce::String deviceError;
    if (!audio.ensureOutputDevice(deviceError))
    {
        playWhenRenderReady = false;
        showError(strings.text("settings.noAudioDevice") + "\n" + deviceError);
        return;
    }
    // Wait for the current audio near the playhead, never a previous selection's
    // fallback. Unrelated pending regions can continue rendering during play.
    armSelectionPlayback();
    if (!sourceEditActive && audio.playbackNeedsRender())
    {
        playWhenRenderReady = true;
        return;
    }
    playWhenRenderReady = false;
    // Respect the transport position chosen by the user.  The marquee handler
    // already moves it to the selection once; subsequent Space presses and
    // explicit red-line seeks must never be overwritten by cached-audio bounds.
    armSelectionPlayback();
    startPreparedPlayback();
}

void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(Palette::background);
    g.setColour(Palette::panel);
    g.fillRect(0, 0, getWidth(), 61);
    g.setColour(Palette::border);
    g.drawHorizontalLine(60, 0.0f, static_cast<float>(getWidth()));
    const auto splitterBounds = panelSplitter.getBounds();
    g.setColour(Palette::background);
    g.fillRect(splitterBounds);
    g.setColour(Palette::border);
    g.drawHorizontalLine(splitterBounds.getY(), static_cast<float>(splitterBounds.getX()),
                         static_cast<float>(splitterBounds.getRight()));
    g.drawHorizontalLine(splitterBounds.getBottom() - 1, static_cast<float>(splitterBounds.getX()),
                         static_cast<float>(splitterBounds.getRight()));
}

void MainComponent::updateNoteDanceKeys()
{
    const auto f8Down = juce::KeyPress::isKeyCurrentlyDown(juce::KeyPress::F8Key);
    if (hasKeyboardFocus(true) && !isCurrentlyBlockedByAnotherModalComponent())
    {
        if (juce::KeyPress::isKeyCurrentlyDown('F') && juce::KeyPress::isKeyCurrentlyDown('Q'))
            pianoRoll.setNoteDanceEnabled(false);
        else if (f8Down && !noteDanceF8WasDown)
            pianoRoll.setNoteDanceEnabled(true);
    }
    noteDanceF8WasDown = f8Down;
}

bool MainComponent::keyStateChanged(bool)
{
    updateNoteDanceKeys();
    return false;
}

bool MainComponent::keyPressed(const juce::KeyPress& key)
{
    if (key.isKeyCode(juce::KeyPress::escapeKey) && pianoRoll.diffSingerPitchOffsetMode())
    { if (!pianoRoll.cancelDiffSingerPitchOffsetGesture()) pianoRoll.setDiffSingerPitchOffsetMode(false); return true; }
    if (key.isKeyCode(juce::KeyPress::escapeKey) && pianoRoll.diffSingerPitchRestoreMode())
    { pianoRoll.setDiffSingerPitchRestoreMode(false); return true; }
    if (key.getKeyCode() == juce::KeyPress::F8Key)
    {
        if (!noteDanceF8WasDown) pianoRoll.setNoteDanceEnabled(true);
        noteDanceF8WasDown = true;
        return true;
    }
    if ((key.getKeyCode() == 'F' || key.getKeyCode() == 'Q')
        && pianoRoll.isNoteDanceEnabled()
        && juce::KeyPress::isKeyCurrentlyDown('F') && juce::KeyPress::isKeyCurrentlyDown('Q'))
    {
        pianoRoll.setNoteDanceEnabled(false);
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'Z')
    {
        if (key.getModifiers().isShiftDown()) project.redo();
        else project.undo();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'Y')
    {
        project.redo();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == juce::KeyPress::returnKey)
    {
        showMaterialPronunciationInput();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'G')
    {
        showRegionEditorForNote();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'A')
    {
        pianoRoll.selectAllNotes();
        return true;
    }
    if (key.getKeyCode() == juce::KeyPress::deleteKey
        || key.getKeyCode() == juce::KeyPress::backspaceKey)
    {
        // Only reached when the roll did not take it, which it does whenever
        // notes are selected -- so in practice this is the clip case.
        switch (deleteTargetFor(!pianoRoll.selectedNoteIds().empty(),
                                selectedClipId.isNotEmpty()))
        {
            case DeleteTarget::notes:   pianoRoll.deleteSelectedNotes(); return true;
            case DeleteTarget::clip:    deleteSelectedClip(); return true;
            case DeleteTarget::nothing: return false;
        }
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'C')
    {
        if (!pianoRoll.selectedNoteIds().empty()) copySelectedNotes(false);
        else if (selectedClipId.isNotEmpty()) copySelectedClip();
        else return false;
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'X'
        && !pianoRoll.selectedNoteIds().empty())
    {
        copySelectedNotes(true);
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'V')
    {
        // Plain: the moment it was copied from when it is going to another
        // track, the playhead when it is going back onto its own -- there the
        // original moment would be on top of itself.  With shift: the playhead
        // either way, which is where clicking in the roll has already put it.
        if (!copiedNotes.empty())
            pasteCopiedNotes(key.getModifiers().isShiftDown()
                ? std::optional<double>(audio.position()) : std::nullopt);
        else if (copiedClipId.isNotEmpty()) pasteCopiedClip();
        else return false;
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'D'
        && selectedClipId.isNotEmpty())
    {
        duplicateSelectedClip();
        return true;
    }
    if (key == juce::KeyPress::spaceKey
        && (preferences == nullptr
            || preferences->getBoolValue("operation.spacePlayback", true)))
    {
        togglePlayback();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'N')
    {
        newProject();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'O')
    {
        openProject();
        return true;
    }
    if (key.getModifiers().isCommandDown() && key.getKeyCode() == 'S')
    {
        if (key.getModifiers().isShiftDown()) saveProjectAs();
        else saveProject();
        return true;
    }
    if (key.getKeyCode() == juce::KeyPress::homeKey)
    {
        audio.stop();
        audio.setPosition(0.0);
        return true;
    }
    return false;
}

bool MainComponent::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (const auto& path : files)
        if (juce::File(path).hasFileExtension("wav;flac;aif;aiff;mp3;ogg;hjpx;hspx;mpd;mid;midi;ust"))
            return true;
    return false;
}

void MainComponent::openExternalFile(const juce::File& file)
{
    if (file.hasFileExtension("mpd"))
        loadMelodyneFile(file);
    else if (file.hasFileExtension("hjpx;hspx"))
        performWithUnsavedCheck([this, file] { loadProjectFile(file); });
    else if (file.hasFileExtension("ust;ustx")) loadUstFile(file);
    else if (file.hasFileExtension("mid;midi"))
    {
        juce::String error;
        if (!project.addMidiFile(file, error))
            showError(strings.text("error.midi") + "\n" + error);
    }
    else if (const auto duration = audio.probeDuration(file))
        addAnalysedAudioFile(file, *duration);
}

std::pair<double, juce::String> MainComponent::audioDropTarget(int x, int y) const
{
    auto dropSeconds = audio.position();
    juce::String targetTrackId;
    if (timelineViewport.getBounds().contains(x, y))
    {
        dropSeconds = timeline.secondsForPixel(x - timelineViewport.getX()
                                               + timelineViewport.getViewPositionX());
        targetTrackId = timeline.trackIdForPixel(y - timelineViewport.getY()
                                                 + timelineViewport.getViewPositionY());
    }
    else if (pianoViewport.getBounds().contains(x, y))
    {
        dropSeconds = pianoRoll.secondsForPixel(x - pianoViewport.getX()
                                                + pianoViewport.getViewPositionX());
        targetTrackId = selectedTrackId;
    }
    return {std::max(0.0, dropSeconds), targetTrackId};
}

bool MainComponent::isInterestedInDragSource(const juce::DragAndDropTarget::SourceDetails& details)
{
    return assetManager != nullptr && details.sourceComponent.get() == assetManager.get()
        && details.description["type"].toString() == "hachi-material-audio";
}

void MainComponent::itemDropped(const juce::DragAndDropTarget::SourceDetails& details)
{
    if (!isInterestedInDragSource(details)) return;
    const auto point = details.localPosition;
    if (!timelineViewport.getBounds().contains(point) && !pianoViewport.getBounds().contains(point)) return;
    const auto [seconds, track] = audioDropTarget(point.x, point.y);
    if (static_cast<bool>(details.description["nativeMaterial"]))
    {
        juce::StringArray ids;
        const auto selectedIds = details.description["regionIds"];
        if (const auto* selected = selectedIds.getArray())
            for (const auto& id : *selected) ids.add(id.toString());
        importMaterialPronunciations(juce::File(details.description["file"].toString()),
            juce::File(details.description["manifest"].toString()), ids, seconds, track,{},details.description["audioVariantId"].toString());
        return;
    }
    importAssetAudio(juce::File(details.description["file"].toString()),
        juce::File(details.description["root"].toString()), static_cast<int>(details.description["otoKind"]),
        seconds, track, localPointToGlobal(point),details.description["audioVariantId"].toString());
}

bool MainComponent::shouldDropFilesWhenDraggedExternally(
    const juce::DragAndDropTarget::SourceDetails& details, juce::StringArray& files, bool& canMove)
{
    if (!isInterestedInDragSource(details)) return false;
    const juce::File file(details.description["file"].toString());
    if (!file.existsAsFile()) return false;
    files.add(assetManager->dragFileFor(file).getFullPathName());
    canMove = false;
    return true;
}

void MainComponent::filesDropped(const juce::StringArray& files, int x, int y)
{
    const auto [dropSeconds, targetTrackId] = audioDropTarget(x, y);
    auto nextDropSeconds = dropSeconds;
    for (const auto& path : files)
    {
        const juce::File file(path);
        if (file.hasFileExtension("wav;flac;aif;aiff;mp3;ogg"))
        {
            if (const auto duration = audio.probeDuration(file))
            {
                addAnalysedAudioFile(file, *duration, nextDropSeconds, targetTrackId);
                if (targetTrackId.isNotEmpty()) nextDropSeconds += *duration;
            }
        }
        else
            openExternalFile(file);
    }
}

void MainComponent::resized()
{
    auto area = getLocalBounds(); menuBar.setBounds(area.removeFromTop(27));
    auto toolbar = area.removeFromTop(34).reduced(5, 3);
    const auto take = [&toolbar](juce::Component& c, int width)
    { c.setBounds(toolbar.removeFromLeft(width)); toolbar.removeFromLeft(3); };
    take(bpmCaption,30); take(bpmEditor,50); take(beatsCaption,58); take(beatsEditor,32);
    take(denominatorLabel,27); take(gridCaption,28); take(gridSelector,67); take(stretchCaption,66);
    take(stretchSelector,70); take(scaleCaption,52); take(scaleSelector,58); toolbar.removeFromLeft(5);
    take(stopButton,28); take(playButton,28); toolbar.removeFromLeft(10); take(collapseTracksButton,28);
    progressBar.setBounds(toolbar.removeFromRight(100).reduced(4,5));
    auto footer = area.removeFromBottom(24);
    selectionSummaryLabel.setBounds(footer.removeFromRight(250).reduced(8,0)); statusLabel.setBounds(footer.reduced(8,0));
    if (assetManager != nullptr && assetManagerVisible)
    {
        const auto width = juce::jlimit(220,std::max(220,area.getWidth()-320),assetManagerWidth);
        auto panel = area.removeFromRight(width); assetManager->setBounds(panel);
        if (assetManagerResizer) assetManagerResizer->setBounds(panel.getX()-3,panel.getY(),6,panel.getHeight());
    }
    const auto utau = projectedTrackEditable&&displayedTrackType==TrackType::utau;
    const auto ds = projectedTrackEditable&&displayedTrackType==TrackType::diffSinger;
    const auto hachi = !utau&&!ds;
    if (utau || ds) volumeParamButton.setToggleState(envelopeLane==EnvelopeLane::amplitude,juce::dontSendNotification);
    volumeParamButton.setButtonText(utau || ds ? utf8("响度包络") : strings.text("param.volume"));
    volumeParamButton.setTooltip(ds
        ? utf8("编辑 DS 合成后的音量包络，默认平直、100% 原音量；拖动标点调整，双击曲线加点，右键内部标点删除。与能量参数分开，不改变音高或音素时长；试听和导出均生效。")
        : utau ? utf8("显示全部 UTAU 音符的振幅包络；拖动标点编辑，双击曲线加点，右键内部标点删除") : strings.text("param.volume"));
    refreshPitchScopeTitle();
    // Atomic label/control groups wrap as a unit. Tools are laid out FIRST,
    // never squeezed by right-hand algorithm pickers into one-pixel icons.
    struct Item { juce::Component* component; int width; };
    using Group = std::vector<Item>; using Groups = std::vector<Group>;
    const auto available = std::max(1,area.getWidth()-8);
    const auto add = [&](Groups& groups,std::initializer_list<Item> items)
    {
        Group group(items);
        for(auto& item:group)
        {
            int textWidth=0;
            if(auto* label=dynamic_cast<juce::Label*>(item.component))
                textWidth=juce::roundToInt(label->getFont().getStringWidthFloat(label->getText()))+8;
            else if(auto* button=dynamic_cast<juce::TextButton*>(item.component);button&&!button->getButtonText().isEmpty())
                textWidth=juce::roundToInt(button->getLookAndFeel().getTextButtonFont(*button,30).getStringWidthFloat(button->getButtonText()))+20;
            else if(auto* combo=dynamic_cast<juce::ComboBox*>(item.component))
                textWidth=juce::roundToInt(combo->getLookAndFeel().getComboBoxFont(*combo).getStringWidthFloat(combo->getText()))+32;
            if (item.component == &utauVoicebankName) textWidth=std::min(260,textWidth);
            if (item.component == &trackEditingHeading) textWidth=std::min(164,textWidth);
            item.width=std::min(available,std::max(item.width,textWidth));
        }
        groups.push_back(std::move(group));
    };
    const auto widthOf = [](const Group& group)
    { int width=0;for(const auto& item:group)width+=item.width;return width+3*std::max(0,static_cast<int>(group.size())-1); };
    const auto rowsFor = [&](const Groups& groups)
    {
        int rows=groups.empty()?0:1,used=0;
        for(const auto& group:groups){const auto width=widthOf(group);if(used>0&&used+width>available){++rows;used=0;}used+=width+6;}
        return rows;
    };
    const auto place = [&](const Groups& groups,juce::Rectangle<int> bounds)
    {
        const auto left=bounds.getX()+4,right=bounds.getRight()-4;int x=left,y=bounds.getY()+3;
        for(const auto& group:groups)
        {
            if(x>left&&x+widthOf(group)>right){x=left;y+=36;}
            for(const auto& item:group){item.component->setBounds(x,y,item.width,30);x+=item.width+3;}
            x+=3;
        }
    };
    Groups tools,expression,sample,mode;
    pointButton.setVisible(true);spliceButton.setVisible(true);
    trackEditingHeading.setVisible(hachi);parameterTitle.setVisible(false);
    pitchAlgorithm.setVisible(true);pitchLabel.setVisible(false);
    add(tools,{{&pitchAlgorithm,156}});
    utauVoicebankName.setVisible(utau||ds);
    if(utau||ds)add(tools,{{&utauVoicebankName,164}});
    else add(tools,{{&trackEditingHeading,std::min(available,164)}});
    add(tools,{{&noteEditButton,27},{&drawButton,27},{&pointButton,27},{&lineButton,27}});
    wrenchButton.setVisible(!ds);trimButton.setVisible(hachi);nativeLoudnessButton.setVisible(hachi);connectButton.setVisible(hachi);
    if(!ds)add(tools,{{&wrenchButton,27}});if(hachi)add(tools,{{&trimButton,27},{&nativeLoudnessButton,48},{&connectButton,27}});
    const auto showOrder=hachi&&nsfPickerItem(pitchAlgorithm.getSelectedId());
    renderOrder.setVisible(showOrder);renderOrderLabel.setVisible(showOrder);
    if(showOrder)add(tools,{{&renderOrderLabel,38},{&renderOrder,108}});
    stretchAlgorithm.setVisible(!ds);stretchLabel.setVisible(!ds);
    const auto stretchWidth=available>=900?168:124;
    if(!ds)add(tools,{{&stretchLabel,52},{&stretchAlgorithm,stretchWidth}});
    utauVoicebankButton.setVisible(utau);
    utauVoicebankButton.setEnabled(utau && !sourceEditActive);
    if(utau)add(tools,{{&spliceButton,100},{&utauVoicebankButton,120}});
    else add(tools,{{&spliceButton,100}});
    if(ds)
    {
        add(tools,{{&diffSingerModelButton,110},{&diffSingerSettingsButton,150}});
        add(expression,{{&diffSingerContextLabel,std::min(available,380)}});
        add(expression,{{&diffSingerTimingButton,120},{&diffSingerPronunciationButton,120},{&diffSingerPitchButton,120},{&diffSingerParametersButton,150}});
    }
    // Each existing expression control appears once. The old 537px empty
    // spacer and right-side reservations could consume the entire narrow row.
    for(auto* c:{static_cast<juce::Component*>(&smoothCaption),static_cast<juce::Component*>(&smoothSlider),
        static_cast<juce::Component*>(&pitchParamButton),static_cast<juce::Component*>(&driftParamButton),
        static_cast<juce::Component*>(&attackParamButton),static_cast<juce::Component*>(&breathParamButton),
        static_cast<juce::Component*>(&tensionParamButton),static_cast<juce::Component*>(&formantParamButton)})c->setVisible(hachi);
    volumeParamButton.setVisible(true);flagEnvelopeButton.setVisible(utau);
    if(utau)add(expression,{{&volumeParamButton,78},{&flagEnvelopeButton,78}});
    if(ds)add(expression,{{&volumeParamButton,78}});
    envelopePresetMenu.setVisible(!selectedTrackIsAccompaniment());
    if(envelopePresetMenu.isVisible())add(expression,{{&envelopePresetMenu,110}});
    advancedEnvelopeButton.setVisible(!ds&&(utau||nativeAudioOverlapButton.isVisible()));
    advancedEnvelopeButton.setButtonText(utf8("高级包络"));advancedEnvelopeButton.setToggleState(false,juce::dontSendNotification);
    advancedEnvelopeButton.setTooltip(utf8("编辑当前音符的高级响度包络"));
    if(advancedEnvelopeButton.isVisible())add(expression,{{&advancedEnvelopeButton,90}});
    if(nativeVibratoButton.isVisible())add(expression,{{&nativeVibratoButton,64}});
    if(nativeAudioOverlapButton.isVisible())add(expression,{{&nativeAudioOverlapButton,108}});
    if(nsfPitchTransitionsButton.isVisible())add(expression,{{&nsfPitchTransitionsButton,128}});
    if(nsfNoiseProtectionButton.isVisible())add(expression,{{&nsfNoiseProtectionButton,156}});
    autoCrossfadeButton.setVisible(!ds);if(!ds)add(expression,{{&autoCrossfadeButton,94}});
    if(phonemizerButton.isVisible())
        add(expression,{{&phonemizerButton,stretchWidth},{&pianoPreviewButton,110}});
    else if(pianoPreviewButton.isVisible())add(expression,{{&pianoPreviewButton,110}});
    for(auto* c:{static_cast<juce::Component*>(&sampleRegionSelector),static_cast<juce::Component*>(&sampleAliasEditor),
        static_cast<juce::Component*>(&sampleStartEditor),static_cast<juce::Component*>(&sampleEndEditor),
        static_cast<juce::Component*>(&sampleAlignmentEditor),static_cast<juce::Component*>(&sampleFixedEditor),
        static_cast<juce::Component*>(&sampleAliasLabel),static_cast<juce::Component*>(&sampleStartLabel),
        static_cast<juce::Component*>(&sampleEndLabel),static_cast<juce::Component*>(&sampleAlignmentLabel),
        static_cast<juce::Component*>(&sampleFixedLabel),static_cast<juce::Component*>(&sampleSaveButton),
        static_cast<juce::Component*>(&otoImportButton),static_cast<juce::Component*>(&otoExportButton)})c->setVisible(sourceEditActive);
    if(sourceEditActive)
    {
        add(sample,{{&sourceEditHint,std::min(220,available)}});add(sample,{{&sampleRegionSelector,104}});
        add(sample,{{&sampleAliasLabel,34},{&sampleAliasEditor,100}});add(sample,{{&sampleStartLabel,34},{&sampleStartEditor,64}});
        add(sample,{{&sampleEndLabel,30},{&sampleEndEditor,64}});add(sample,{{&sampleAlignmentLabel,44},{&sampleAlignmentEditor,64}});
        add(sample,{{&sampleFixedLabel,44},{&sampleFixedEditor,64}});add(sample,{{&sampleSaveButton,64}});
        add(sample,{{&otoImportButton,82}});add(sample,{{&otoExportButton,82}});
    }
    else sourceEditHint.setBounds({});
    noteAliasLabel.setVisible(utau||ds);noteAliasEditor.setVisible(utau||ds);
    noteAliasLabel.setText(ds?(strings.text("Lyrics")):(strings.text("OTO alias")),juce::dontSendNotification);
    for(auto* c:{static_cast<juce::Component*>(&noteFlagsLabel),static_cast<juce::Component*>(&noteFlagsEditor),static_cast<juce::Component*>(&flagCurveButton)})c->setVisible(utau);
    noteConsonantVelocityLabel.setVisible(!ds);noteConsonantVelocityEditor.setVisible(!ds);
    robustPitchCurveButton.setVisible(hachi&&pitchAlgorithm.getSelectedId()==1);
    if(utau||ds)
    {
        add(mode,{{&noteAliasLabel,64},{&noteAliasEditor,100}});
        if(utau)
        {
            add(mode,{{&noteConsonantVelocityLabel,72},{&noteConsonantVelocityEditor,52}});
            add(mode,{{&noteFlagsLabel,42},{&noteFlagsEditor,std::max(90,std::min(220,available-144))},{&flagCurveButton,92}});
        }
    }
    else
    {
        add(mode,{{&smoothCaption,42},{&smoothSlider,118}});add(mode,{{&pitchParamButton,64},{&driftParamButton,64},{&attackParamButton,64}});
        add(mode,{{&noteConsonantVelocityLabel,72},{&noteConsonantVelocityEditor,52}});
        add(mode,{{&breathParamButton,64},{&tensionParamButton,64},{&formantParamButton,72},{&volumeParamButton,64}});
        if(robustPitchCurveButton.isVisible())add(mode,{{&robustPitchCurveButton,104}});
    }
    const auto headerHeight=36*(rowsFor(tools)+rowsFor(expression)+rowsFor(sample)+rowsFor(mode));
    const auto splitAvailable=std::max(1,area.getHeight()-8-headerHeight);
    const auto upperHeight=tracksCollapsed?0:juce::jlimit(150,std::max(150,splitAvailable-120),
        static_cast<int>(std::round(static_cast<float>(splitAvailable)*panelSplitRatio)));
    for(auto* c:{static_cast<juce::Component*>(&trackViewport),static_cast<juce::Component*>(&timelineViewport),static_cast<juce::Component*>(&panelSplitter)})c->setVisible(!tracksCollapsed);
    if(!tracksCollapsed)
    {
        auto upper=area.removeFromTop(upperHeight);trackViewport.setBounds(upper.removeFromLeft(280));timelineViewport.setBounds(upper);
        panelSplitter.setBounds(area.removeFromTop(8));
    }
    place(tools,area.removeFromTop(36*rowsFor(tools)));place(expression,area.removeFromTop(36*rowsFor(expression)));
    place(sample,area.removeFromTop(36*rowsFor(sample)));place(mode,area.removeFromTop(36*rowsFor(mode)));
    pianoViewport.setBounds(area);
    if(!pianoInitialScrollSet&&pianoViewport.getHeight()>0)
    {pianoViewport.setViewPosition(0,std::max(0,(pianoRoll.getHeight()-pianoViewport.getHeight())/2));pianoInitialScrollSet=true;}
}

void MainComponent::adjustHorizontalZoom(double factor, bool upperPane)
{
    if (upperPane)
    {
        const auto next = juce::jlimit(40.0, 8000.0, std::round(timelineHorizontalZoom * factor));
        if (next == timelineHorizontalZoom) return;
        const auto left = timelineViewport.getViewPositionX();
        const auto width = timelineViewport.getViewWidth();
        const auto playhead = timeline.pixelForSeconds(audio.position());
        const auto anchor = playhead >= left && playhead < left + width
            ? audio.position() : timeline.secondsForPixel(left + width / 2);
        timelineHorizontalZoom = next;
        timeline.setPixelsPerSecond(static_cast<float>(next));
        timelineViewport.setViewPosition(juce::jmax(0, timeline.pixelForSeconds(anchor) - width / 2),
                                         timelineViewport.getViewPositionY());
        lastTimelineX = timelineViewport.getViewPositionX();
        lastPianoX = pianoViewport.getViewPositionX();
        return;
    }
    const auto nextZoom = juce::jlimit(zoomSlider.getMinimum(), zoomSlider.getMaximum(),
                                      zoomSlider.getValue() * factor);
    if (std::abs(nextZoom - zoomSlider.getValue()) < 0.001) return;
    zoomSlider.setValue(nextZoom, juce::sendNotificationSync);
}

void MainComponent::adjustVerticalZoom(double factor, bool upperPane)
{
    if (upperPane)
    {
        const auto next = juce::jlimit(0.5, 2.0, std::round(timelineVerticalZoom * factor * 20.0) / 20.0);
        if (next == timelineVerticalZoom) return;
        const auto viewHeight = timelineViewport.getViewHeight();
        const auto ruler = timeline.getRulerHeight();
        const auto anchor = (timelineViewport.getViewPositionY() + viewHeight / 2.0 - ruler)
            / static_cast<double>(timeline.getRowHeight());
        timelineVerticalZoom = next;
        timeline.setRowHeight(static_cast<float>(96.0 * next));
        trackList.setRowHeight(static_cast<float>(96.0 * next));
        timelineViewport.setViewPosition(timelineViewport.getViewPositionX(), juce::jmax(0,
            juce::roundToInt(anchor * timeline.getRowHeight() + ruler - viewHeight / 2.0)));
        trackViewport.setViewPosition(0, timelineViewport.getViewPositionY());
        lastTimelineY = timelineViewport.getViewPositionY();
        lastTrackY = trackViewport.getViewPositionY();
        lastTimelineX = timelineViewport.getViewPositionX();
        lastPianoX = pianoViewport.getViewPositionX();
        if (preferences != nullptr) preferences->setValue("ui.timelineVZoom", next);
        return;
    }
    const auto oldZoom = vZoomSlider.getValue();
    const auto nextZoom = juce::jlimit(vZoomSlider.getMinimum(), vZoomSlider.getMaximum(),
                                      oldZoom * factor);
    if (std::abs(nextZoom - oldZoom) < 0.001) return;

    const auto viewHeight = pianoViewport.getViewHeight();
    const auto oldCentreY = pianoViewport.getViewPositionY() + viewHeight / 2.0;
    vZoomSlider.setValue(nextZoom, juce::sendNotificationSync);
    const auto nextViewY = std::max(0, static_cast<int>(std::round(
        oldCentreY * juce::jlimit(12.0, 48.0, 22.0 * vZoomSlider.getValue())
            / juce::jlimit(12.0, 48.0, 22.0 * oldZoom) - viewHeight / 2.0)));
    pianoViewport.setViewPosition(pianoViewport.getViewPositionX(), nextViewY);
}

void MainComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    if (event.eventComponent == &utauVoicebankName && event.mods.isLeftButtonDown()
        && utauVoicebankName.isVisible() && displayedTrackType == TrackType::utau && !sourceEditActive)
        showVoicebankSettings();
}

void MainComponent::mouseDown(const juce::MouseEvent& event)
{
    if (event.eventComponent != &panelSplitter) return;
    draggingPanelSplitter = true;
    panelSplitterDragScreenY = event.getScreenY();
    panelSplitterDragRatio = panelSplitRatio;
}

void MainComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (!draggingPanelSplitter || event.eventComponent != &panelSplitter) return;
    const auto splitAvailable = std::max(1, getHeight() - 27 - 34 - 24 - 8 - 36);
    panelSplitRatio = juce::jlimit(0.15f, 0.85f,
        panelSplitterDragRatio + static_cast<float>(event.getScreenY() - panelSplitterDragScreenY)
            / static_cast<float>(splitAvailable));
    resized();
    repaint();
}

void MainComponent::mouseUp(const juce::MouseEvent&)
{
    draggingPanelSplitter = false;
}

void MainComponent::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &project)
    {
        const auto data = project.snapshot();
        setAssetManagerVisible(data.assetManagerVisible);
        if (assetManager != nullptr)
            assetManager->setProjectVoicebanks(data);
        const auto trackExists = std::any_of(data.tracks.begin(), data.tracks.end(),
            [this](const auto& track) { return track.id == selectedTrackId; });
        if (!trackExists) selectedTrackId.clear();
        if (selectedTrackId.isEmpty())
        {
            const auto firstCompose = std::find_if(data.tracks.begin(), data.tracks.end(),
                [](const auto& track) { return track.compose; });
            if (firstCompose != data.tracks.end()) selectedTrackId = firstCompose->id;
        }
        // A region retains its identity when transferred to another lane.
        // Rebind the lower editor to its new owner, including after undo/redo.
        for (const auto& track : data.tracks)
            if (std::any_of(track.clips.begin(), track.clips.end(),
                [this](const auto& clip) { return clip.id == selectedClipId; })) {
                selectedTrackId = track.id; break;
            }
        syncAudio(data);
        scheduleNativeTrimSourceAnalysis(data);
        trackList.setSelectedTrack(selectedTrackId);
        pianoRoll.setFocusedTrack(selectedTrackId);
        auto clipExists = false;
        for (const auto& track : data.tracks)
            clipExists = clipExists || std::any_of(track.clips.begin(), track.clips.end(),
                [this](const auto& clip) { return clip.id == selectedClipId; });
        if (!clipExists)
        {
            selectedClipId.clear();
            pianoRoll.setFocusedClip({});
        }
        auto noteExists = false;
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                noteExists = noteExists || std::any_of(clip.notes.begin(), clip.notes.end(),
                    [this](const auto& note) { return note.id == selectedNoteId; });
        if (!noteExists) selectedNoteId.clear();
        refreshProjectControls();
        refreshSelectedNoteParameter();
        menuItemsChanged();
    }
}

void MainComponent::refreshSelectionSummary()
{
    const auto ids=pianoRoll.selectedNoteIds();
    const auto revision=project.revisionNumber();
    if(ids==selectionSummaryIds && revision==selectionSummaryRevision)return;
    selectionSummaryIds=ids;selectionSummaryRevision=revision;
    if(ids.empty()){selectionSummaryLabel.setText({},juce::dontSendNotification);return;}
    int notes=0,regions=0;
    for(const auto& track:project.snapshot().tracks)for(const auto& clip:track.clips)
    {
        int count=0;for(const auto& note:clip.notes)
            if(std::find(ids.begin(),ids.end(),note.id)!=ids.end())++count;
        notes+=count;if(count)++regions;
    }
    selectionSummaryLabel.setText(notes ? utf8("选中 ")+juce::String(notes)+utf8(" 音符 · ")+juce::String(regions)+utf8(" 区域") : juce::String{},juce::dontSendNotification);
}

void MainComponent::timerCallback()
{
    refreshAfterVoicebankChange();
    pollLiveMcp();
    refreshSelectionSummary();
    updateNoteDanceKeys();
    pianoRoll.advanceNoteDance();
    static juce::String previousModalState;
    auto* modal = juce::Component::getCurrentlyModalComponent();
    const auto modalState = modal == nullptr ? juce::String("none")
        : modal->getName() + "; visible=" + juce::String(modal->isShowing() ? 1 : 0)
            + "; bounds=" + modal->getScreenBounds().toString();
    if (modalState != previousModalState)
    {
        startupLog("UI modal: " + modalState);
        previousModalState = modalState;
    }
    const auto expectedPlayIcon = audio.isPlaying() ? juce::String("icon.pause")
                                                     : juce::String("icon.play");
    if (playButton.getComponentID() != expectedPlayIcon)
    {
        playButton.setComponentID(expectedPlayIcon);
        playButton.setTooltip(strings.text(audio.isPlaying() ? "transport.pause" : "transport.play"));
        playButton.repaint();
    }
    pianoRoll.setPlayheadSeconds(audio.position());
    timeline.setPlayheadSeconds(audio.position());
    // Which UTAU notes have audio to show.  The roll compares each against the
    // note still under it, so a render finishing here is what makes a waveform
    // appear and an edit is what makes it go.
    rememberDiffSingerParameters();
    pianoRoll.setUtauNoteWaveforms(audio.utauNoteWaveforms());
    audio.refreshNativeWaveforms();
    const auto nativeWaveforms=audio.nativeClipWaveforms();
    pianoRoll.setNativeClipWaveforms(nativeWaveforms);
    timeline.setNativeClipWaveforms(nativeWaveforms);
    if (projectRecovery)
        if (const auto result=projectRecovery->tick(project,currentProjectFile,
            project.revisionNumber()!=savedProjectRevision,juce::Time::getMillisecondCounterHiRes()))
        {
            if (result->isNotEmpty() && *result!=lastRecoveryError)
                showError(strings.text("recovery.failed")+"\n"+*result);
            lastRecoveryError=*result;
        }
    trackList.repaint();
    // An export asked for the whole song to be rendered and is waiting for it.
    // Writing the files is quick -- it is only mixing what is already
    // rendered -- so it can happen here, where the render has just finished.
    if (exportWaitingForRender && !audio.renderProgress())
    {
        finishExport();
        return;
    }
    const auto keepDiffSingerStatus = diffSingerBusy
        || juce::Time::getMillisecondCounterHiRes() < diffSingerStatusUntil;
    if (!importInProgress)
    {
        if (const auto render = audio.renderProgress())
        {
            showingRenderProgress = true;
            progress = *render;
            const auto percent = juce::String(static_cast<int>(
                std::round(*render * 100.0))) + "%";
            if (!keepDiffSingerStatus) statusLabel.setText(activeUtauSelectionCount > 0
                    ? "UTAU selection: " + juce::String(activeUtauSelectionCount)
                        + " notes  |  Rendering " + percent
                    : strings.text("status.rendering") + "  " + percent,
                juce::dontSendNotification);
        }
        else
        {
            if (showingRenderProgress)
            {
                showingRenderProgress = false;
                progress = 0.0;
            }
            if (pendingNativeAnalyses > 0)
            {
                if (!keepDiffSingerStatus)
                    statusLabel.setText(strings.text("status.analyzing") + "  "
                        + nativeAnalysisName + "  "
                        + juce::String(static_cast<int>(std::round(nativeAnalysisProgress * 100.0)))
                        + "%", juce::dontSendNotification);
            }
            else
            {
                const auto backend = audio.activeRenderBackends();
                const auto renderWarning = audio.activeRenderWarnings();
                // Edits the chosen backend cannot honour: editing never hides a
                // feature, so this is the one place the difference is stated.
                const auto capability = AudioEngine::renderCapabilityWarnings(
                    project.snapshot(), audio.currentUtauOutputEngine()).joinIntoString("；");
                if (!keepDiffSingerStatus) statusLabel.setText((audio.isPlaying() ? strings.text("transport.play")
                                                        : strings.text("status.ready"))
                                        + "  " + juce::String(audio.position(), 2) + " s"
                                        + (backend.isNotEmpty() ? utf8("  ·  ") + backend : juce::String())
                                        + (renderWarning.isNotEmpty()
                                            ? utf8("  ·  UTAU: ") + renderWarning : juce::String())
                                        + (capability.isNotEmpty()
                                            ? utf8("  ·  ⚠ ") + capability : juce::String()),
                                    juce::dontSendNotification);
            }
        }
        if (playWhenRenderReady && !audio.playbackNeedsRender())
        {
            playWhenRenderReady = false;
            juce::String deviceError;
            if (!audio.hasCurrentRenderedAudio() && !audio.renderProgress())
            {
                statusLabel.setText(utf8("UTAU 当前选区没有生成可播放音频"),
                                    juce::dontSendNotification);
            }
            else if (audio.ensureOutputDevice(deviceError))
            {
                armSelectionPlayback();
                startPreparedPlayback();
            }
            else showError(strings.text("settings.noAudioDevice") + "\n" + deviceError);
        }
    }

    if (!syncingScroll)
    {
        // While playing, one decision, made for the roll, with the ruler put
        // on the same instant afterwards.
        //
        // Asking each panel separately looked tidier and was not: they are
        // different widths, so a run that fits one need not fit the other,
        // and then one held still while the other paged.  The sync below --
        // which cannot tell a scroll the user made from one the follow just
        // made -- carried the paging panel's position across and undid the
        // one that had held.  Next tick it held again.  That was the flicker,
        // and a long note brought it on because that is when the two panels
        // disagree about fitting.
        const auto playing = audio.isPlaying();
        if (playing)
        {
            const auto run = playbackRun
                ? std::optional<juce::Range<int>>(juce::Range<int>(
                      pianoRoll.pixelForSeconds(playbackRun->getStart()),
                      pianoRoll.pixelForSeconds(playbackRun->getEnd())))
                : std::nullopt;
            if (const auto next = followViewPosition(
                    pianoViewport.getViewPositionX(), pianoViewport.getViewWidth(),
                    pianoRoll.pixelForSeconds(audio.position()), 58, run))
            {
                pianoViewport.setViewPosition(*next, pianoViewport.getViewPositionY());
                if (!sourceEditActive)
                    timelineViewport.setViewPosition(
                        std::max(0, timeline.pixelForSeconds(
                            pianoRoll.secondsForPixel(*next))),
                        timelineViewport.getViewPositionY());
            }
        }
        const auto timelineX = timelineViewport.getViewPositionX();
        const auto pianoX = pianoViewport.getViewPositionX();
        syncingScroll = true;
        // Scrolling one moves the other to the same instant, not to the same
        // pixel: they measure from different origins and may be at different
        // scales, so a raw pixel is not a place they both understand.  Only
        // for a scroll the user made: while playing the follow owns both, and
        // this cannot tell the two apart.
        if (!playing && !sourceEditActive && timelineX != lastTimelineX)
            pianoViewport.setViewPosition(
                std::max(0, pianoRoll.pixelForSeconds(
                    timeline.secondsForPixel(timelineX))),
                pianoViewport.getViewPositionY());
        else if (!playing && !sourceEditActive && pianoX != lastPianoX)
            timelineViewport.setViewPosition(
                std::max(0, timeline.pixelForSeconds(
                    pianoRoll.secondsForPixel(pianoX))),
                timelineViewport.getViewPositionY());
        lastTimelineX = timelineViewport.getViewPositionX();
        lastPianoX = pianoViewport.getViewPositionX();
        const auto timelineY = timelineViewport.getViewPositionY();
        const auto trackY = trackViewport.getViewPositionY();
        if (timelineY != lastTimelineY)
            trackViewport.setViewPosition(0, timelineY);
        else if (trackY != lastTrackY)
            timelineViewport.setViewPosition(timelineViewport.getViewPositionX(), trackY);
        lastTimelineY = timelineViewport.getViewPositionY();
        lastTrackY = trackViewport.getViewPositionY();
        syncingScroll = false;
    }
}

std::optional<double> MainComponent::viewSecondsLeavingSourceEdit(
    bool wasEnabled, bool nowEnabled, double timelineSeconds)
{
    if (nowEnabled || !wasEnabled) return {};
    return std::max(0.0, timelineSeconds);
}

void MainComponent::setSourceEditMode(bool enabled)
{
    playWhenRenderReady = false;
    const auto wasEnabled = sourceEditActive;
    sourceEditActive = enabled;
    if (enabled && selectedClipId.isEmpty())
    {
        const auto data = project.snapshot();
        for (const auto& track : data.tracks)
            if (!track.clips.empty())
            {
                selectedClipId = track.clips.front().id;
                break;
            }
    }
    pianoRoll.setSourceEditMode(enabled);
    pianoRoll.setFocusedClip(selectedClipId);
    if (enabled)
    {
        const auto data = project.snapshot();
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                if (clip.id == selectedClipId)
                {
                    audio.setAuditionFile(clip.sourceFile);
                    pianoViewport.setViewPosition(std::max(0, pianoRoll.pixelForSeconds(clip.sourceOffsetSeconds)
                                                              - pianoViewport.getViewWidth() / 4),
                                                  pianoViewport.getViewPositionY());
                }
        loadSampleSettings();
    }
    else
    {
        audio.clearAuditionFile();
        // Coming back from source edit, the roll goes to whatever moment the
        // timeline is showing.  This used to copy the timeline's pixel column
        // straight across, and the two run at different pixels per second, so
        // it landed somewhere else -- and it ran even when source edit had not
        // been on, which both the loudness lane and the point tool ask for on
        // every click.  That is the view sliding for no reason.
        if (const auto seconds = viewSecondsLeavingSourceEdit(
                wasEnabled, enabled,
                timeline.secondsForPixel(timelineViewport.getViewPositionX())))
            pianoViewport.setViewPosition(
                std::max(0, pianoRoll.pixelForSeconds(*seconds)),
                pianoViewport.getViewPositionY());
    }
    noteEditButton.setToggleState(!enabled, juce::dontSendNotification);
    wrenchButton.setToggleState(enabled, juce::dontSendNotification);
    sourceEditHint.setVisible(enabled);
    resized();
}

void MainComponent::focusClip(const juce::String& clipId)
{
    if (clipId.isNotEmpty() && !timeline.isClipSelected(clipId))
        timeline.setSelectedClips({clipId}, false);
    selectedClipId = clipId;
    pianoRoll.setFocusedClip(clipId);
    const auto data = project.snapshot();
    const ClipData* selectedClip = nullptr;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == clipId)
            {
                selectedClip = &clip;
                selectedTrackId = track.id;
                trackList.setSelectedTrack(track.id);
                pianoRoll.setFocusedTrack(track.id);
                refreshProjectControls();
                break;
            }

    if (!sourceEditActive) return;
    if (selectedClip == nullptr)
    {
        audio.clearAuditionFile();
        return;
    }
    audio.setAuditionFile(selectedClip->sourceFile);
    pianoViewport.setViewPosition(std::max(0,
        pianoRoll.pixelForSeconds(selectedClip->sourceOffsetSeconds)
            - pianoViewport.getViewWidth() / 4), pianoViewport.getViewPositionY());
    loadSampleSettings();
}

void MainComponent::focusNote(const juce::String& noteId)
{
    selectedNoteId = noteId;
    if (noteId.isNotEmpty())
    {
        const auto data = project.snapshot();
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                if (std::any_of(clip.notes.begin(), clip.notes.end(),
                    [&noteId](const auto& note) { return note.id == noteId; }))
                {
                    // Editing an extra visible region must not replace the active
                    // region and hide its notes or prune a cross-region selection.
                    const auto keepRegion = !sourceEditActive && clip.showNormalDisplay
                        && track.compose && !track.accompaniment && track.id == selectedTrackId
                        && std::any_of(track.clips.begin(), track.clips.end(),
                            [&](const auto& current) { return current.id == selectedClipId; });
                    if (!keepRegion) focusClip(clip.id);
                    refreshSelectedNoteParameter();
                    return;
                }
    }
    refreshSelectedNoteParameter();
}

void MainComponent::updateUtauRenderSelection()
{
    const auto noteIds = pianoRoll.selectedNoteIds();
    activeUtauSelectionCount = static_cast<int>(noteIds.size());
    audio.setUtauRenderNoteSelection(noteIds);
    // mouseDown clears the visual selection before a new marquee is finished.
    // Do not rebuild an all-history phrase at that transient point; Space with
    // no selection explicitly requests history playback in togglePlayback().
    if (noteIds.empty()) return;
    const auto data = project.snapshot();
    if (audio.isPlaying()) { syncAudio(data); return; }

    std::unordered_set<std::string> validNoteIds;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                validNoteIds.insert(note.id.toStdString());
    accumulatedUtauNoteIds.clear();
    for (const auto& id : noteIds)
        if (validNoteIds.contains(id.toStdString()))
            accumulatedUtauNoteIds.insert(id.toStdString());
    std::optional<double> firstSelectedSeconds;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (accumulatedUtauNoteIds.contains(note.id.toStdString()))
                {
                    const auto start = clip.startSeconds + note.startSeconds;
                    firstSelectedSeconds = firstSelectedSeconds
                        ? std::min(*firstSelectedSeconds, start) : start;
                }
    if (firstSelectedSeconds)
        audio.setPosition(std::max(0.0, *firstSelectedSeconds - 0.05));
    // Establish the audition start before releasing the new render batch.
    syncAudio(data);
}

void MainComponent::loadSampleSettings()
{
    sampleSettingsFile = juce::File{};
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == selectedClipId)
                sampleSettingsFile = clip.sourceFile;
    sampleSettingsRows = sampleSettingsFile.existsAsFile()
        ? SampleSettings::loadOrDerive(sampleSettingsFile, data)
        : std::vector<SampleRegionSetting>{};
    activeSampleSetting = 0;
    sampleRegionSelector.clear(juce::dontSendNotification);
    for (std::size_t index = 0; index < sampleSettingsRows.size(); ++index)
        sampleRegionSelector.addItem(juce::String(index + 1) + utf8(" · ")
                                     + sampleSettingsRows[index].name,
                                     static_cast<int>(index + 1));
    if (!sampleSettingsRows.empty())
        sampleRegionSelector.setSelectedId(1, juce::dontSendNotification);
    pianoRoll.setSampleRegions(sampleSettingsRows, activeSampleSetting);
    refreshSampleEditors();
}

void MainComponent::refreshSampleEditors()
{
    const auto enabled = activeSampleSetting >= 0
        && activeSampleSetting < static_cast<int>(sampleSettingsRows.size());
    for (auto* editor : { &sampleAliasEditor, &sampleStartEditor, &sampleEndEditor,
                          &sampleAlignmentEditor, &sampleFixedEditor }) editor->setEnabled(enabled);
    if (!enabled)
    {
        for (auto* editor : { &sampleAliasEditor, &sampleStartEditor, &sampleEndEditor,
                              &sampleAlignmentEditor, &sampleFixedEditor }) editor->clear();
        return;
    }
    const auto& row = sampleSettingsRows[static_cast<std::size_t>(activeSampleSetting)];
    sampleAliasEditor.setText(row.name, false);
    sampleStartEditor.setText(juce::String(row.regionStartSeconds, 4), false);
    sampleEndEditor.setText(juce::String(row.regionEndSeconds, 4), false);
    sampleAlignmentEditor.setText(juce::String(row.alignmentSeconds, 4), false);
    sampleFixedEditor.setText(juce::String(row.fixedDurationSeconds, 4), false);
    pianoRoll.setSampleRegions(sampleSettingsRows, activeSampleSetting);
}

void MainComponent::commitSampleEditors()
{
    if (activeSampleSetting < 0
        || activeSampleSetting >= static_cast<int>(sampleSettingsRows.size())) return;
    auto& row = sampleSettingsRows[static_cast<std::size_t>(activeSampleSetting)];
    row.name = sampleAliasEditor.getText().trim();
    row.regionStartSeconds = std::max(0.0, sampleStartEditor.getText().getDoubleValue());
    row.regionEndSeconds = std::max(row.regionStartSeconds + 0.001,
                                    sampleEndEditor.getText().getDoubleValue());
    row.alignmentSeconds = juce::jlimit(row.regionStartSeconds, row.regionEndSeconds,
                                        sampleAlignmentEditor.getText().getDoubleValue());
    row.fixedDurationSeconds = juce::jlimit(0.0, row.regionEndSeconds - row.regionStartSeconds,
                                            sampleFixedEditor.getText().getDoubleValue());
    sampleRegionSelector.changeItemText(activeSampleSetting + 1,
        juce::String(activeSampleSetting + 1) + " · " + row.name);
    pianoRoll.setSampleRegions(sampleSettingsRows, activeSampleSetting);
}

void MainComponent::saveSampleSettings()
{
    // Legacy keyboard/test handoff: never publish through the retired inline
    // single-row form. Quick edits share the same guarded multi-row session.
    openSelectedMaterialEditor(true);
}

void MainComponent::importOto()
{
    chooser = std::make_unique<juce::FileChooser>(strings.text("sample.importOto"),
                                                   sampleSettingsFile.getParentDirectory(),
                                                   "oto.ini;*.ini");
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (file == juce::File{}) return;
            juce::String error;
            const auto duration = audio.probeDuration(sampleSettingsFile).value_or(0.0);
            if (!SampleSettings::importOto(file, sampleSettingsFile, duration,
                                           sampleSettingsRows, error))
            {
                showError(error);
                return;
            }
            activeSampleSetting = 0;
            sampleRegionSelector.clear(juce::dontSendNotification);
            for (std::size_t index = 0; index < sampleSettingsRows.size(); ++index)
                sampleRegionSelector.addItem(juce::String(index + 1) + utf8(" · ")
                    + sampleSettingsRows[index].name, static_cast<int>(index + 1));
            sampleRegionSelector.setSelectedId(1, juce::dontSendNotification);
            pianoRoll.setSampleRegions(sampleSettingsRows, activeSampleSetting);
            refreshSampleEditors();
        });
}

void MainComponent::exportOto()
{
    commitSampleEditors();
    if (!sampleSettingsFile.existsAsFile() || sampleSettingsRows.empty()) return;
    chooser = std::make_unique<juce::FileChooser>(strings.text("sample.exportOto"),
        sampleSettingsFile.getParentDirectory().getChildFile("oto.ini"), "*.ini");
    chooser->launchAsync(juce::FileBrowserComponent::saveMode
                             | juce::FileBrowserComponent::warnAboutOverwriting,
        [this](const juce::FileChooser& selected)
        {
            auto file = selected.getResult();
            if (file == juce::File{}) return;
            if (!file.hasFileExtension("ini")) file = file.withFileExtension("ini");
            const auto duration = audio.probeDuration(sampleSettingsFile).value_or(0.0);
            juce::String error;
            if (!SampleSettings::exportOto(file, sampleSettingsFile, sampleSettingsRows,
                                           duration, error)) showError(error);
        });
}

juce::StringArray MainComponent::getMenuBarNames()
{
    return { strings.text("menu.file"), strings.text("menu.edit"), strings.text("menu.track"),
             strings.text("menu.view"), strings.text("menu.help") };
}

juce::PopupMenu MainComponent::getMenuForIndex(int index, const juce::String&)
{
    juce::PopupMenu menu;
    if (index == 0)
    {
        menu.addItem(1, strings.text("file.new"));
        menu.addItem(2, strings.text("file.open"));
        menu.addItem(3, strings.text("file.save"));
        menu.addItem(11, strings.text("file.saveAs"));
        menu.addItem(63, strings.text("file.recovery"));
        juce::PopupMenu recent;
        for (int recentIndex = 0; recentIndex < recentProjectPaths.size(); ++recentIndex)
        {
            const juce::File file(recentProjectPaths[recentIndex]);
            recent.addItem(1'000 + recentIndex,
                file.getFileNameWithoutExtension() + "  —  "
                    + file.getParentDirectory().getFullPathName(),
                file.existsAsFile());
        }
        if (recentProjectPaths.isEmpty())
            recent.addItem(999, strings.text("file.recentEmpty"), false);
        menu.addSubMenu(strings.text("file.recent"), recent);
        menu.addItem(8, strings.text("file.export"));
        menu.addItem(12, strings.text("file.exportLastRender"),
                     !lastRenderedNoteIds.empty());
        menu.addItem(14, strings.text("file.exportMidi"));
        menu.addItem(64, utf8("导出当前轨道 UST…"), selectedTrackId.isNotEmpty());
        menu.addSeparator();
        menu.addItem(4, strings.text("file.audio"));
        menu.addItem(5, strings.text("file.melodyne"));
        menu.addItem(6, strings.text("file.midi"));
        menu.addItem(13, strings.text("file.ust"));
        menu.addSeparator();
        menu.addItem(9, strings.text("file.settings"));
        menu.addItem(10, strings.text("file.assets"));
        menu.addItem(850, strings.text("material.pronunciationInput"));
        menu.addItem(863,strings.text("Register project source materials..."),!projectRegistrationOperation);
        menu.addItem(864,strings.text("Retry material folder registration"),!projectRegistrationRetryRoots.empty());
        menu.addSeparator();
        menu.addItem(7, strings.text("file.exit"));
    }
    else if (index == 1)
    {
        menu.addItem(20, strings.text("edit.undo"), project.canUndo());
        menu.addItem(21, strings.text("edit.redo"), project.canRedo());
        menu.addSeparator();
        menu.addItem(22, strings.text("edit.selectAll"));
        menu.addItem(26, strings.text("edit.deselect"));
        menu.addSeparator();
        const auto hasNotes = !pianoRoll.selectedNoteIds().empty();
        menu.addItem(16, strings.text("edit.copyNotes"), hasNotes);
        menu.addItem(17, strings.text("edit.cutNotes"), hasNotes);
        menu.addItem(51, strings.text("edit.pasteNotesAtOrigin"), !copiedNotes.empty());
        menu.addItem(18, strings.text("edit.pasteNotes"), !copiedNotes.empty());
        menu.addSeparator();
        menu.addItem(58, utf8("HAMOOD 自动和声…"), !diffSingerBusy);
        menu.addItem(27, strings.text("edit.transposeCents"), hasNotes);
        menu.addItem(28, strings.text("edit.setPitch"), hasNotes);
        menu.addItem(29, strings.text("edit.averagePitch"), hasNotes);
        menu.addItem(19, strings.text("edit.quantizePitch"), hasNotes);
        menu.addSeparator();
        // Only in the UTAU modes: pinyin is what a UTAU voicebank's aliases
        // are spelt in, and the lyric of any other kind of note is a label.
        menu.addItem(hanziToPinyinMenuItem, strings.text("edit.hanziToPinyin"),
                     selectedTrackId.isNotEmpty());
        const auto dsData = project.snapshot();
        const auto dsTrack = std::find_if(dsData.tracks.begin(), dsData.tracks.end(),
            [this](const auto& t) { return t.id == selectedTrackId; });
        const auto canGenerate = dsTrack != dsData.tracks.end() && trackIsDiffSinger(*dsTrack);
        if (canGenerate)
        {
            menu.addSectionHeader(utf8("谋•UTAU / DiffSinger"));
            const auto selected = !pianoRoll.selectedNoteIds().empty();
            menu.addItem(53, utf8(selected ? "DS 重生成选中音符 pitch（保留其他音高）" : "DiffSinger 生成整轨 pitch"), !diffSingerBusy);
            if (selected) menu.addItem(57, utf8("DS 重新生成整轨 pitch"), !diffSingerBusy);
            menu.addItem(56, utf8("DS 发音转换 / 读音覆盖…"), !diffSingerBusy);
            menu.addItem(55, utf8("DS 音素时长…"), !diffSingerBusy);
        }
        menu.addItem(54, utf8("取消 DiffSinger 任务"), diffSingerBusy);
        menu.addSeparator();
        menu.addItem(23, strings.text("edit.copyClip"), selectedClipId.isNotEmpty());
        menu.addItem(24, strings.text("edit.pasteClip"), copiedClipId.isNotEmpty());
        menu.addItem(25, strings.text("edit.duplicateClip"), selectedClipId.isNotEmpty());
    }
    else if (index == 2)
    {
        menu.addItem(36,strings.text("New HachiShifter track"));
        menu.addItem(861,strings.text("New UTAU track"));
        menu.addItem(862,strings.text("New DiffSinger track"));
        menu.addItem(37, strings.text("track.addAudio"));
        menu.addItem(accompanimentTrackMenuItem, strings.text("track.newAccompaniment"));
        menu.addSeparator();
        menu.addItem(38, strings.text("track.rename"), selectedTrackId.isNotEmpty());
        menu.addItem(31, strings.text("track.toggleCompose"), selectedTrackId.isNotEmpty() && !selectedTrackIsAccompaniment());
        menu.addSeparator();
        auto selectedClipMuted = false;
        if (selectedClipId.isNotEmpty())
        {
            const auto data = project.snapshot();
            for (const auto& track : data.tracks)
                for (const auto& clip : track.clips)
                    if (clip.id == selectedClipId) selectedClipMuted = clip.muted;
        }
        menu.addItem(34, strings.text(selectedClipMuted ? "clip.unmute" : "clip.mute"),
                     selectedClipId.isNotEmpty());
        menu.addItem(35, strings.text("clip.gain"), selectedClipId.isNotEmpty());
        menu.addSeparator();
        menu.addItem(32, strings.text("track.delete"), selectedTrackId.isNotEmpty());
        menu.addItem(33, strings.text("clip.delete"), selectedClipId.isNotEmpty());
    }
    else if (index == 3)
    {
        menu.addItem(40, strings.text("view.zoomIn"));
        menu.addItem(41, strings.text("view.zoomOut"));
        menu.addItem(42, strings.text("view.zoomFit"));
        menu.addItem(44, strings.text("view.vZoomIn"));
        menu.addItem(45, strings.text("view.vZoomOut"));
        menu.addSeparator();
        menu.addItem(43, strings.text("view.showWaveforms"), true, showWaveforms);
        menu.addSeparator();
        appendEditorViewItems(menu);
    }
    else
        menu.addItem(50, strings.text("help.about"));
    return menu;
}

void MainComponent::menuItemSelected(int id, int)
{
    if ((id >= 601 && id <= 609) || id == trackPitchScopeMenuItem || id == 613) { applyEditorViewChoice(id - 600); return; }
    if (id == 1) newProject();
    else if (id == 2) openProject();
    else if (id == 3) saveProject();
    else if (id == 11) saveProjectAs();
    else if (id == 63) showRecoveryDialog();
    else if (id == 8) exportMixdown();
    else if (id == 12) exportLastRender();
    else if (id == 14) exportMidi();
    else if (id == 64) exportUst();
    else if (id == 4 || id == 30) importAudio();
    else if (id == 5) importMelodyne();
    else if (id == 6) importMidi();
    else if (id == 13) importUst();
    else if (id == 9) showSettings();
    else if (id == 10) showAssetManager();
    else if (id == 850) showMaterialPronunciationInput();
    else if(id==863)offerMaterialFolderForImport(project.snapshot());
    else if(id==864)retryProjectMaterialRootRegistration();
    else if (id == 7) requestClose([] { juce::JUCEApplication::getInstance()->quit(); });
    else if (id >= 1'000 && id < 1'000 + recentProjectPaths.size())
    {
        const juce::File file(recentProjectPaths[id - 1'000]);
        performWithUnsavedCheck([this, file] { loadProjectFile(file); });
    }
    else if (id == 20) project.undo();
    else if (id == 21) project.redo();
    else if (id == 22) pianoRoll.selectAllNotes();
    else if (id == 26) pianoRoll.clearNoteSelection();
    else if (id == 16) copySelectedNotes(false);
    else if (id == 17) copySelectedNotes(true);
    else if (id == 18) pasteCopiedNotes();
    else if (id == 58) showHamood();
    else if (id == 27) showTransposeNotesDialog();
    else if (id == 28) showSetNotesPitchDialog();
    else if (id == 29) project.averageNotesMidi(pianoRoll.selectedNoteIds());
    else if (id == 19) project.quantizeNotesMidi(pianoRoll.selectedNoteIds());
    else if (id == hanziToPinyinMenuItem && selectedTrackId.isNotEmpty())
        project.convertTrackLyricsToPinyin(selectedTrackId);
    else if (id == 51) pasteCopiedNotes(copiedOriginSeconds);
    else if (id == 53) generateDiffSingerPitch();
    else if (id == 57) generateDiffSingerPitch(true);
    else if (id == 56) showDiffSingerPronunciationEditor();
    else if (id == 55) showDiffSingerPhonemeEditor();
    else if (id == 54) diffSingerCancel->store(true);
    else if (id == 23) copySelectedClip();
    else if (id == 24) pasteCopiedClip();
    else if (id == 25) duplicateSelectedClip();
    else if(id==36)addTrackFromMenu(TrackType::hachi);
    else if(id==861)addTrackFromMenu(TrackType::utau);
    else if(id==862)addTrackFromMenu(TrackType::diffSinger);
    else if(id==37)addTrackFromMenu(false);
    else if (id == accompanimentTrackMenuItem) addAccompanimentTrackFromMenu();
    else if (id == 38) showRenameTrackDialog();
    else if (id == 31)
    {
        const auto data = project.snapshot();
        const auto found = std::find_if(data.tracks.begin(), data.tracks.end(),
            [this](const auto& track) { return track.id == selectedTrackId; });
        if (found != data.tracks.end()) project.setTrackCompose(found->id, !found->compose);
    }
    else if (id == 32) deleteSelectedTrack();
    else if (id == 33) deleteSelectedClip();
    else if (id == 34)
    {
        const auto data = project.snapshot();
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                if (clip.id == selectedClipId)
                {
                    project.setClipMuted(clip.id, !clip.muted);
                    return;
                }
    }
    else if (id == 35) showClipGainDialog();
    else if (id == 40) zoomSlider.setValue(zoomSlider.getValue() * 1.25);
    else if (id == 41) zoomSlider.setValue(zoomSlider.getValue() / 1.25);
    else if (id == 42)
    {
        const auto available = std::max(200, timelineViewport.getWidth());
        zoomSlider.setValue(static_cast<double>(available) / std::max(1.0, project.snapshot().durationSeconds()));
    }
    else if (id == 43)
    {
        showWaveforms = !showWaveforms;
        pianoRoll.setShowWaveforms(showWaveforms);
        if (preferences != nullptr)
            preferences->setValue("ui.showWaveforms", showWaveforms);
        menuItemsChanged();
    }
    else if (id == 44) vZoomSlider.setValue(vZoomSlider.getValue() * 1.2);
    else if (id == 45) vZoomSlider.setValue(vZoomSlider.getValue() / 1.2);
    else if (id == 50)
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
            strings.text("app.title"), strings.text("help.aboutText"));
}

void MainComponent::showSettings(int page)
{
    if (preferences == nullptr) return;
    if (auto* existing = settingsWindows.findVisible<SettingsComponent>())
    {
        static_cast<SettingsComponent*>(existing->getContentComponent())->selectPage(page);
        existing->toFront(true);
        return;
    }
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = strings.text("settings.title");
    options.dialogBackgroundColour = Palette::panel;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = true;
    juce::Component::SafePointer<MainComponent> safe(this);
    options.content.setOwned(new SettingsComponent(strings, audio.devices(), *preferences,
        [safe]
        {
            if (safe == nullptr || safe->closingUI) return;
            if (safe->preferences != nullptr)
                safe->audio.saveDeviceState(*safe->preferences);
            safe->applyPreferences();
            safe->refreshTexts();
            safe->menuItemsChanged();
            safe->repaint();
            safe->trackList.repaint();
            safe->timeline.repaint();
            safe->pianoRoll.repaint();
        }));
    static_cast<SettingsComponent*>(options.content.get())->selectPage(page);
    settingsWindows.show(options,"settings");
}

void MainComponent::showAssetManager()
{
    setAssetManagerVisible(!assetManagerVisible);
    project.setAssetManagerVisible(assetManagerVisible);
}

void MainComponent::setAssetManagerVisible(bool visible)
{
    if (preferences == nullptr || assetManagerVisible == visible) return;
    // Docked in the main window, not a separate window: it stays operable
    // alongside the piano roll, and toggles off when its menu item is chosen
    // again.  Created lazily the first time it is shown.
    if (assetManager == nullptr)
    {
        assetManager = std::make_unique<AssetManagerComponent>(strings, *preferences);
        assetManager->onBrowserStateChanged = [this](const juce::String& state, const juce::String& session)
        {
            if (project.snapshot().materialBrowserSessionId == session)
                project.setMaterialBrowserState(state);
        };
        assetManager->onMaterialPreflight = [this](const juce::File& source, const auto& rows, juce::String& error)
        { return preflightMaterialPublication(source, rows, error); };
        assetManager->onMaterialSaved = [this](const juce::File& source, const auto& rows)
        { materialPublicationCompleted(source, rows); };
        assetManager->onConfigureMaterialEditor = [this](OtoWaveformEditorComponent& editor)
        { attachOtoPlayback(editor); };
        assetManager->onUseMaterialFolder = [this](const juce::File& folder)
        { selectVoicebank(selectedTrackId, folder); };
        assetManager->onLoadMaterialSelection = [this](const AssetManagerComponent::MaterialSelection& selection)
        {
            if(selection.audioVariantId.isNotEmpty()||!selection.regionIds.isEmpty())
                importMaterialPronunciations(selection.audio,selection.root.getChildFile("otto.ini"),selection.regionIds,audio.position(),selectedTrackId,{},selection.audioVariantId);
            else importAssetAudio(selection.audio,selection.root,assetManager->otoImportKind(selection.audio),audio.position(),selectedTrackId,juce::Desktop::getMousePosition());
        };
        assetManager->onLoadPronunciations = [this](const juce::File& file, const juce::StringArray& ids)
        {
            importMaterialPronunciations(file, assetManager->rootForAudio(file).getChildFile("otto.ini"),
                ids, audio.position(), selectedTrackId);
        };
        assetManager->onLoadAudio = [this](const juce::File& file)
        {
            importAssetAudio(file, assetManager->rootForAudio(file), assetManager->otoImportKind(file),
                             audio.position(), selectedTrackId, juce::Desktop::getMousePosition());
        };
        addChildComponent(assetManager.get());
        assetManagerWidth = juce::jlimit(220, 640,
            preferences->getIntValue("ui.assetManagerWidth", 320));
        // A drag handle on the panel's left edge resizes its width.  The
        // constrainer bounds it; resized() re-lays everything from the width.
        assetManagerConstrainer.setMinimumWidth(220);
        assetManagerConstrainer.setMaximumWidth(640);
        assetManagerConstrainer.onWidth = [this](int width)
        {
            assetManagerWidth = juce::jlimit(220, 640, width);
            if (preferences != nullptr)
                preferences->setValue("ui.assetManagerWidth", assetManagerWidth);
            resized();
        };
        assetManagerResizer = std::make_unique<juce::ResizableEdgeComponent>(
            assetManager.get(), &assetManagerConstrainer,
            juce::ResizableEdgeComponent::leftEdge);
        addChildComponent(assetManagerResizer.get());
    }
    assetManagerVisible = visible;
    if (assetManagerVisible)
        assetManager->setProjectVoicebanks(project.snapshot());
    assetManager->setVisible(assetManagerVisible);
    assetManagerResizer->setVisible(assetManagerVisible);
    if (assetManagerVisible)
    {
        assetManager->toFront(false);
        assetManagerResizer->toFront(false);
        statusLabel.setText(strings.text("asset.browseHint"), juce::dontSendNotification);
    }
    resized();
}

void MainComponent::showClipGainDialog()
{
    if (selectedClipId.isEmpty()) return;
    auto gain = 1.0f;
    auto found = false;
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == selectedClipId)
            {
                gain = clip.gain;
                found = true;
            }
    if (!found) return;

    auto* dialog = new DecibelGainDialog(strings,strings.text("clip.gain"),gain,4.0f,this);
    const auto clipId = selectedClipId;
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create([safe, dialog, clipId](int result)
        {
            if (safe != nullptr && result == 1)
            {
                if(const auto value=dialog->value())safe->project.setClipGain(clipId,*value);
            }
            delete dialog;
        }), false);
}

void MainComponent::showTrackGainDialog(const juce::String& trackId)
{
    const auto data=project.snapshot();
    const auto found=std::find_if(data.tracks.begin(),data.tracks.end(),[&](const auto& t){return t.id==trackId;});
    if(found==data.tracks.end())return;
    const auto initial=found->volume;
    auto* dialog=new DecibelGainDialog(strings,strings.text("track.volume")+" — "+found->name,initial,2.0f,this);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,trackId,initial](int result)
    {
        if(safe!=nullptr && result==1)
            if(const auto value=dialog->value();value && *value!=initial)
                safe->project.setTrackVolume(trackId,*value);
        delete dialog;
    }),false);
}

void MainComponent::showTrackPanDialog(const juce::String& trackId)
{
    const auto data = project.snapshot();
    const auto found = std::find_if(data.tracks.begin(), data.tracks.end(), [&](const auto& t) { return t.id == trackId; });
    if (found == data.tracks.end()) return;
    const auto initial = found->pan;
    auto* dialog = new PanValueDialog(strings, strings.text("track.pan") + " — " + found->name, initial, this);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create([safe, dialog, trackId, initial](int result)
    {
        if (safe != nullptr && result == 1)
            if (const auto value = dialog->value(); value && *value != initial)
                safe->project.setTrackPan(trackId, *value);
        delete dialog;
    }), false);
}

void MainComponent::showRenameTrackDialog()
{
    if (selectedTrackId.isEmpty()) return;
    auto currentName = juce::String{};
    for (const auto& track : project.snapshot().tracks)
        if (track.id == selectedTrackId)
        {
            currentName = track.name;
            break;
        }
    if (currentName.isEmpty()) return;
    auto* dialog = new juce::AlertWindow(strings.text("track.rename"), juce::String{},
                                          juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("name", currentName, strings.text("track.name"));
    dialog->addButton(strings.text("dialog.apply"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    const auto trackId = selectedTrackId;
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create([safe, dialog, trackId](int result)
        {
            if (safe != nullptr && result == 1)
                safe->project.setTrackName(trackId,
                    dialog->getTextEditorContents("name"));
            delete dialog;
        }), false);
}

void MainComponent::showTransposeNotesDialog()
{
    const auto noteIds = pianoRoll.selectedNoteIds();
    if (noteIds.empty()) return;
    auto* dialog = new juce::AlertWindow(strings.text("edit.transposeCents"), juce::String{},
                                          juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("cents", "0", strings.text("edit.cents"));
    dialog->addButton(strings.text("dialog.apply"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create([safe, dialog, noteIds](int result)
        {
            if (safe != nullptr && result == 1)
            {
                const auto cents = juce::jlimit(-4'800.0, 4'800.0,
                    dialog->getTextEditorContents("cents").getDoubleValue());
                safe->project.transposeNotes(noteIds, static_cast<float>(cents / 100.0));
            }
            delete dialog;
        }), false);
}

void MainComponent::showSetNotesPitchDialog()
{
    const auto noteIds = pianoRoll.selectedNoteIds();
    if (noteIds.empty()) return;
    auto initial = 60.0f;
    auto found = false;
    for (const auto& track : project.snapshot().tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (std::find(noteIds.begin(), noteIds.end(), note.id) != noteIds.end())
                {
                    initial = note.midiNote;
                    found = true;
                    break;
                }
    if (!found) return;
    auto* dialog = new juce::AlertWindow(strings.text("edit.setPitch"), juce::String{},
                                          juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("midi", juce::String(initial, 2), strings.text("edit.midiNote"));
    dialog->addButton(strings.text("dialog.apply"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create([safe, dialog, noteIds](int result)
        {
            if (safe != nullptr && result == 1)
                safe->project.setNotesMidi(noteIds, static_cast<float>(
                    dialog->getTextEditorContents("midi").getDoubleValue()));
            delete dialog;
        }), false);
}

void MainComponent::copySelectedNotes(bool cut)
{
    const auto ids = pianoRoll.selectedNoteIds();
    if (ids.empty()) return;
    const auto data = project.snapshot();
    std::vector<std::pair<double, NoteData>> notes;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (std::find(ids.begin(), ids.end(), note.id) != ids.end())
                    notes.emplace_back(clip.startSeconds + note.startSeconds, note);
    if (notes.empty()) return;
    std::stable_sort(notes.begin(), notes.end(), [](const auto& left, const auto& right)
    {
        return left.first < right.first;
    });
    const auto origin = notes.front().first;
    copiedOriginSeconds = origin;
    copiedTrackId.clear();
    for (const auto& track : project.snapshot().tracks)
        for (const auto& clip : track.clips)
            for (const auto& note : clip.notes)
                if (std::find(ids.begin(), ids.end(), note.id) != ids.end())
                    copiedTrackId = track.id;
    copiedClipId.clear();
    copiedNotes.clear(); copiedNativeClips.clear(); copiedNativeConnections.clear();
    copiedNotes.reserve(notes.size());
    for (auto& [absolute, note] : notes)
    {
        note.startSeconds = absolute - origin;
        copiedNotes.push_back(std::move(note));
    }
    auto native = copyNativeAudioNotes(data, ids);
    if (!native.clips.empty())
    {
        copiedOriginSeconds = native.originSeconds;
        copiedNativeClips = std::move(native.clips);
        copiedNativeConnections = std::move(native.connections);
    }
    if (cut)
    {
        project.removeNotes(ids);
        pianoRoll.clearNoteSelection();
    }
    statusLabel.setText(strings.text(cut ? "status.notesCut" : "status.notesCopied"),
                        juce::dontSendNotification);
    menuItemsChanged();
}

double MainComponent::pasteTargetSeconds(bool sameTrack, double playheadSeconds,
                                         double originSeconds,
                                         std::optional<double> selectionStartSeconds,
                                         std::optional<double> askedFor,
                                         std::optional<double> pointerSeconds)
{
    if (askedFor) return std::max(0.0, *askedFor);
    // Wherever the pointer is, on either track: pointing at a place is the
    // plainest way of naming one, and it is what a paste is aimed at.
    if (pointerSeconds) return std::max(0.0, *pointerSeconds);
    if (!sameTrack) return std::max(0.0, originSeconds);
    // On its own track a paste goes in front of what is selected, which is
    // what having selected it means; with nothing selected, the playhead.
    return std::max(0.0, selectionStartSeconds.value_or(playheadSeconds));
}

juce::String MainComponent::pasteTargetTrack() const
{
    // The track in front, since that is the one being pasted onto.
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        if (track.id == selectedTrackId) return track.id;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == selectedClipId) return track.id;
    return data.tracks.empty() ? juce::String() : data.tracks.front().id;
}

MainComponent::PasteClipPlan MainComponent::pasteClipPlanFor(
    const ProjectData& data, const juce::String& trackId, double atSeconds)
{
    PasteClipPlan plan;
    plan.clipId = pasteTargetClipIn(data, trackId, atSeconds);
    if (plan.clipId.isNotEmpty()) return plan;

    // Only a melodic track can be given one: an audio track has nothing for
    // notes to sound through.
    for (const auto& track : data.tracks)
        if (track.id == trackId && track.compose && track.clips.empty())
        {
            const auto beat = 60.0 / juce::jlimit(20.0, 400.0, data.bpm);
            const auto bar = beat * static_cast<double>(data.numerator) * 4.0
                / static_cast<double>(std::max(1, data.denominator));
            plan.makeOne = true;
            plan.startSeconds = 0.0;
            plan.seedSeconds = std::max(0.05, bar);
        }
    return plan;
}

juce::String MainComponent::pasteTargetClipIn(const ProjectData& data,
                                              const juce::String& trackId,
                                              double atSeconds)
{
    for (const auto& track : data.tracks)
    {
        if (track.id != trackId || track.clips.empty()) continue;
        for (const auto& clip : track.clips)
            if (atSeconds >= clip.startSeconds - 1.0e-9
                && atSeconds < clip.startSeconds + clip.durationSeconds)
                return clip.id;
        return track.clips.front().id;
    }
    return {};
}

void MainComponent::pasteCopiedNotesInto(const juce::String& clipId, double atSeconds,
                                         const std::vector<juce::String>& replacing)
{
    // Replaced notes go first, then the pasted ones land at exactly the times
    // they were given.  Nothing else in the clip is touched: overwriting is
    // meant to swap a stretch out, not to renumber the piece around it.
    if (!replacing.empty()) project.removeNotes(replacing);
    const auto inserted = project.insertNotes(clipId, copiedNotes, atSeconds);
    if (inserted.empty())
    {
        // Silence here read as a broken paste.  It happens where the clip is a
        // recording and the target is past the end of it: there is no audio
        // out there for a note to sound through.
        statusLabel.setText(utf8("这里放不下：录音素材到此为止，音符只能落在它的范围内。"),
                            juce::dontSendNotification);
        return;
    }
    pianoRoll.setSelectedNoteIds(inserted);
    focusNote(inserted.front());
    statusLabel.setText(strings.text("status.notesPasted"),
                        juce::dontSendNotification);
    menuItemsChanged();
}

void MainComponent::pasteCopiedNotes(std::optional<double> atSeconds)
{
    if (copiedNotes.empty()) return;
    const auto data = project.snapshot();
    const auto trackId = pasteTargetTrack();
    const auto sameTrack = trackId.isNotEmpty() && trackId == copiedTrackId;
    // Where the selection begins, if anything is selected.
    std::optional<double> selectionStart;
    {
        const auto ids = pianoRoll.selectedNoteIds();
        for (const auto& track : data.tracks)
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes)
                    if (std::find(ids.begin(), ids.end(), note.id) != ids.end())
                    {
                        const auto start = clip.startSeconds + note.startSeconds;
                        selectionStart = selectionStart
                            ? std::min(*selectionStart, start) : start;
                    }
    }
    const auto target = pasteTargetSeconds(sameTrack, audio.position(),
                                           copiedOriginSeconds, selectionStart,
                                           atSeconds, pianoRoll.pasteAnchorSeconds());
    if (!copiedNativeClips.empty())
    {
        const auto inserted = project.insertNativeAudioClips(trackId, copiedNativeClips, copiedNativeConnections, target);
        if (inserted.empty())
        {
            statusLabel.setText(utf8("原生音频片段需粘贴到非 UTAU 旋律轨道，且原素材文件须可读取。"),juce::dontSendNotification);
            return;
        }
        // The model broadcasts changes asynchronously. Bring the roll up to
        // date before focus/selection pruning can discard the newly pasted IDs.
        pianoRoll.refreshFromModel();
        const auto updated = project.snapshot();
        for (const auto& track : updated.tracks)
            for (const auto& clip : track.clips)
                if (std::any_of(clip.notes.begin(), clip.notes.end(),
                    [&](const auto& note) { return note.id == inserted.front(); }))
                    focusClip(clip.id);
        pianoRoll.setSelectedNoteIds(inserted);
        statusLabel.setText(strings.text("status.notesPasted"),juce::dontSendNotification);
        menuItemsChanged();return;
    }
    const auto plan = pasteClipPlanFor(data, trackId, target);
    auto clipId = plan.clipId;
    if (plan.makeOne)
        clipId = project.addClip(trackId, plan.startSeconds, plan.seedSeconds);
    if (clipId.isEmpty())
    {
        // Nowhere to put them: an audio track has nothing for notes to sound
        // through, and going quiet read as a broken paste.
        statusLabel.setText(utf8("这条轨道放不下音符：请选择一条旋律轨道。"),
                            juce::dontSendNotification);
        return;
    }

    // On its own track a paste goes in, not over: what it lands on and
    // everything after it move along by the length of what was pasted, and the
    // piece gets that much longer.  Inserting already knows how to do that;
    // clearing the way first is what stopped it, since there was then nothing
    // left for it to push.  Carrying a phrase to another track is the case
    // where replacing is meant, and that still asks first.
    if (sameTrack)
    {
        pasteCopiedNotesInto(clipId, target, {});
        return;
    }

    // How far the pasted block reaches, so what it lands on can be found.
    auto extent = 0.0;
    for (const auto& note : copiedNotes)
        extent = std::max(extent, note.startSeconds + note.durationSeconds);
    const auto replacing = project.notesOverlapping(clipId, target, target + extent);
    if (replacing.empty())
    {
        pasteCopiedNotesInto(clipId, target, {});
        return;
    }
    juce::Component::SafePointer<MainComponent> safe(this);
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::QuestionIcon,
        utf8("粘贴音符"),
        utf8("这一段和轨道上已有的 ") + juce::String(static_cast<int>(replacing.size()))
            + utf8(" 个音符重叠。\n\n继续将删除这些音符并粘贴覆盖；"
                   "其余音符的位置不受影响。"),
        utf8("覆盖粘贴"), utf8("取消"), this,
        juce::ModalCallbackFunction::create(
            [safe, clipId, target, replacing](int result)
            {
                if (safe == nullptr || result == 0) return;
                safe->pasteCopiedNotesInto(clipId, target, replacing);
            }));
}

void MainComponent::copySelectedClip()
{
    if (selectedClipId.isEmpty()) return;
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == selectedClipId)
            {
                copiedNotes.clear(); copiedNativeClips.clear(); copiedNativeConnections.clear();
                copiedClipId = clip.id;
                selectedTrackId = track.id;
                statusLabel.setText(strings.text("status.clipCopied"),
                                    juce::dontSendNotification);
                menuItemsChanged();
                return;
            }
}

MainComponent::ClipPasteTarget MainComponent::clipPasteTargetFor(
    const juce::String& pointerTrackId, std::optional<double> pointerSeconds,
    const juce::String& trackInHand, double playheadSeconds,
    double sourceStartSeconds, double sourceDurationSeconds)
{
    if (pointerTrackId.isNotEmpty() && pointerSeconds)
        return { pointerTrackId, std::max(0.0, *pointerSeconds) };
    auto seconds = std::max(0.0, playheadSeconds);
    // Not past the original yet: land right after it, so pressing paste again
    // and again lays copies end to end instead of on top of one another.
    if (seconds <= sourceStartSeconds + 1.0e-6)
        seconds = sourceStartSeconds + sourceDurationSeconds;
    return { trackInHand, std::max(0.0, seconds) };
}

void MainComponent::pasteCopiedClip()
{
    if (copiedClipId.isEmpty()) return;
    const auto data = project.snapshot();
    auto sourceStart = 0.0;
    auto sourceDuration = 0.0;
    for (const auto& track : data.tracks)
        for (const auto& clip : track.clips)
            if (clip.id == copiedClipId)
            {
                sourceStart = clip.startSeconds;
                sourceDuration = clip.durationSeconds;
            }
    const auto anchor = timeline.pointerAnchor();
    const auto target = clipPasteTargetFor(
        anchor ? anchor->trackId : juce::String{},
        anchor ? std::optional<double>(anchor->seconds) : std::nullopt,
        selectedTrackId, audio.position(), sourceStart, sourceDuration);
    const auto inserted = project.duplicateClip(copiedClipId, target.seconds,
                                                target.trackId);
    if (inserted.isNotEmpty())
    {
        focusClip(inserted);
        statusLabel.setText(strings.text("status.clipPasted"),
                            juce::dontSendNotification);
    }
    else
    {
        copiedClipId.clear();
        menuItemsChanged();
    }
}

void MainComponent::duplicateSelectedClip()
{
    if (selectedClipId.isEmpty()) return;
    const auto inserted = project.duplicateClip(selectedClipId, -1.0,
                                                selectedTrackId);
    if (inserted.isNotEmpty())
    {
        copiedClipId = selectedClipId;
        focusClip(inserted);
        statusLabel.setText(strings.text("status.clipPasted"),
                            juce::dontSendNotification);
        menuItemsChanged();
    }
}

void MainComponent::confirmDestructive(const juce::String& title,
                                       std::function<void()> action)
{
    if (preferences == nullptr
        || !preferences->getBoolValue("operation.confirmDestructive", true))
    {
        if (action) action();
        return;
    }
    auto* dialog = new juce::AlertWindow(title, strings.text("dialog.destructiveMessage"),
                                          juce::MessageBoxIconType::WarningIcon);
    dialog->addButton(strings.text("dialog.delete"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, action = std::move(action)](int result) mutable
            {
                if (safe != nullptr && result == 1 && action) action();
                delete dialog;
            }), false);
}

void MainComponent::newProject()
{
    performWithUnsavedCheck([this]
    {
        audio.stop();
        audio.setPosition(0.0);
        backend::DiffSingerRenderer::openProjectCache();
        cancelProjectMaterialRegistration();
        project.clear();
        setAssetManagerVisible(false);
        discardProjectRecovery();
        liveMcpDocumentChanged();
        currentProjectFile = juce::File{};
        savedProjectRevision = project.revisionNumber();
        selectedTrackId.clear();
        selectedClipId.clear();
        selectedNoteId.clear();
        copiedClipId.clear();
        copiedNotes.clear(); copiedNativeClips.clear(); copiedNativeConnections.clear();
        pianoRoll.clearNoteSelection();
        pianoRoll.setFocusedClip({});
        statusLabel.setText(strings.text("status.ready"), juce::dontSendNotification);
    });
}

void MainComponent::openProject()
{
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.open"), juce::File{}, "*.hjpx;*.hspx");
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (file == juce::File{}) return;
            performWithUnsavedCheck([this, file] { loadProjectFile(file); });
        });
}

void MainComponent::replaceImportedProject(ProjectData replacement)
{
    cancelProjectMaterialRegistration();
    audio.stop();
    audio.setPosition(0.0);
    audio.setUtauRenderNoteSelection({});
    activeUtauSelectionCount=0;
    project.resetDocument(std::move(replacement));
    setAssetManagerVisible(project.snapshot().assetManagerVisible);
    discardProjectRecovery();
    liveMcpDocumentChanged();
    backend::DiffSingerRenderer::openProjectCache();
    currentProjectFile=juce::File{};
    savedProjectRevision=project.revisionNumber()-1;
    selectedTrackId.clear(); selectedClipId.clear(); selectedNoteId.clear();
    copiedClipId.clear(); copiedNotes.clear(); copiedNativeClips.clear(); copiedNativeConnections.clear();
    pianoRoll.clearNoteSelection();
    pianoRoll.setFocusedTrack({}); pianoRoll.setFocusedClip({});
}

bool MainComponent::loadProjectFile(const juce::File& file, bool recovering,bool offerRegistration)
{
    cancelProjectMaterialRegistration();
    juce::String error;
    // A project replacement invalidates every selection and any phrase render
    // derived from the previous document.  Stop transport before parsing and
    // clear those UI references immediately after a successful replacement,
    // before the asynchronous ChangeBroadcaster callbacks rebuild the views.
    audio.stop();
    if (!project.load(file, error))
    {
        showError(error);
        return false;
    }
    discardProjectRecovery();
    backend::DiffSingerRenderer::openProjectCache(recovering ? juce::File{} : file);
    liveMcpDocumentChanged();
    audio.setPosition(0.0);
    audio.setUtauRenderNoteSelection({});
    activeUtauSelectionCount = 0;
    selectedTrackId.clear();
    selectedClipId.clear();
    selectedNoteId.clear();
    pianoRoll.clearNoteSelection();
    pianoRoll.setFocusedTrack({});
    pianoRoll.setFocusedClip({});
    currentProjectFile = recovering ? juce::File{} : file;
    savedProjectRevision = project.revisionNumber() - (recovering ? 1 : 0);
    if (!recovering) addRecentProject(file);
    auto openedMessage = recovering ? strings.text("recovery.restored")
        : strings.text("status.projectOpened") + "  " + file.getFileName();
    const auto loadedData = project.snapshot();
    setAssetManagerVisible(loadedData.assetManagerVisible);
    auto hasSavedPitchReference = false;
    for (const auto& track : loadedData.tracks) for (const auto& clip : track.clips) for (const auto& note : clip.notes)
        hasSavedPitchReference = hasSavedPitchReference || note.diffSingerPitchReferenceFromSavedPitch;
    if (hasSavedPitchReference)
        openedMessage += utf8("；旧工程的虚线参考来自载入时音高，随工程保存");
    statusLabel.setText(openedMessage, juce::dontSendNotification);
    if (error.isNotEmpty())
        showError(strings.text("warning.missingMedia") + "\n" + error);
    if(offerRegistration&&!recovering&&file.hasFileExtension("hjpx;hspx"))offerMaterialFolderForImport(loadedData);
    return true;
}

void MainComponent::saveProject(std::function<void(bool)> completion)
{
    if (currentProjectFile != juce::File{})
    {
        const auto saved = saveProjectTo(currentProjectFile);
        if (completion) completion(saved);
        return;
    }
    saveProjectAs(std::move(completion));
}

void MainComponent::saveProjectAs(std::function<void(bool)> completion)
{
    const auto initial = currentProjectFile != juce::File{} ? currentProjectFile
        : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
            .getChildFile(project.snapshot().name + ".hjpx");
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.saveAs"),
                                                   initial,
                                                   "*.hjpx");
    chooser->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting,
        [this, completion = std::move(completion)](const juce::FileChooser& selected) mutable
        {
            auto file = selected.getResult();
            if (file == juce::File{})
            {
                if (completion) completion(false);
                return;
            }
            if (!file.hasFileExtension("hjpx")) file = file.withFileExtension("hjpx");
            const auto saved = saveProjectTo(file);
            if (completion) completion(saved);
        });
}

bool MainComponent::saveProjectTo(const juce::File& file)
{
    rememberDiffSingerParameters();
    pianoRoll.setUtauNoteWaveforms(audio.utauNoteWaveforms());
    juce::String error;
    if (!project.save(file, error))
    {
        showError(error);
        return false;
    }
    const auto cacheSaved=backend::DiffSingerRenderer::saveProjectCache(file);
    discardProjectRecovery();
    currentProjectFile = file;
    savedProjectRevision = project.revisionNumber();
    addRecentProject(file);
    statusLabel.setText(strings.text("status.projectSaved") + "  " + file.getFileName()
        + (cacheSaved ? juce::String{} : utf8("；DS 缓存未能迁移，工程已保存")),
                        juce::dontSendNotification);
    return true;
}

void MainComponent::performWithUnsavedCheck(std::function<void()> action)
{
    rememberDiffSingerParameters();
    if (project.revisionNumber() == savedProjectRevision)
    {
        if (action) action();
        return;
    }
    // Keep the unsaved-document prompt owned by the main window.  A bare
    // AlertWindow can create an unowned native peer on Windows; with the GDI
    // peer that peer may end up behind the editor while still remaining modal,
    // which makes every editor control appear dead.
    auto* dialog = new juce::AlertWindow(strings.text("dialog.unsavedTitle"),
        strings.text("dialog.unsavedMessage"), juce::MessageBoxIconType::WarningIcon, this);
    dialog->addButton(strings.text("dialog.save"), 1);
    dialog->addButton(strings.text("dialog.discard"), 2);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                      juce::KeyPress(juce::KeyPress::escapeKey));
    dialog->centreAroundComponent(getTopLevelComponent(), dialog->getWidth(), dialog->getHeight());
    dialog->setAlwaysOnTop(true);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create(
            [safe, dialog, action = std::move(action)](int result) mutable
            {
                if (safe != nullptr && result == 1)
                    safe->saveProject([safe, action = std::move(action)](bool saved) mutable
                    {
                        if (safe != nullptr && saved && action) action();
                    });
                else if (safe != nullptr && result == 2 && action)
                    action();
                delete dialog;
            }), false);
}

void MainComponent::requestClose(std::function<void()> approved)
{
    performWithUnsavedCheck([this, approved=std::move(approved)]
    {
        discardProjectRecovery();
        if (approved) approved();
    });
}


void MainComponent::discardProjectRecovery()
{
    if (projectRecovery) projectRecovery->discard();
    lastRecoveryError.clear();
}

void MainComponent::startProjectRecovery()
{
    if (projectRecovery) return;
    const auto base = preferences ? preferences->getFile().getParentDirectory()
        : juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile("HachiShifterNext");
    projectRecovery=std::make_unique<ProjectRecovery>(base.getChildFile("Recovery"));
    juce::Component::SafePointer<MainComponent> safe(this);
    juce::MessageManager::callAsync([safe] { if (safe) safe->showRecoveryDialog(true); });
}

bool MainComponent::restoreProjectRecovery(const ProjectRecovery::Entry& entry)
{
    if (!projectRecovery || entry.file.getParentDirectory()!=projectRecovery->directory()) return false;
    auto claim=ProjectRecovery::tryClaim(entry.file);
    if (!claim) { showError(strings.text("recovery.busy")); return false; }
    if (!loadProjectFile(entry.file,true)) return false;
    projectRecovery->adopt(entry.file,std::move(claim));
    return true;
}

void MainComponent::showRecoveryDialog(bool automatic)
{
    if (!projectRecovery) { startProjectRecovery(); return; }
    const auto entries=projectRecovery->entries();
    if (entries.empty())
    {
        if (!automatic) juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
            strings.text("file.recovery"),strings.text("recovery.none"),{},this);
        return;
    }
    juce::StringArray choices;
    for (const auto& entry:entries)
        choices.add(entry.name+" — "+juce::Time::fromISO8601(entry.savedAt).formatted("%Y-%m-%d %H:%M:%S")
            +(entry.original==juce::File{} ? juce::String{} : " — "+entry.original.getFileName()));
    auto* dialog=new juce::AlertWindow(strings.text("recovery.title"),strings.text("recovery.message"),
        juce::MessageBoxIconType::QuestionIcon,this);
    dialog->addComboBox("recovery",choices);
    dialog->getComboBoxComponent("recovery")->setSelectedItemIndex(0);
    dialog->addButton(strings.text("recovery.restore"),1);
    dialog->addButton(strings.text("recovery.later"),0,juce::KeyPress(juce::KeyPress::escapeKey));
    dialog->addButton(strings.text("recovery.openFolder"),2);
    dialog->centreAroundComponent(getTopLevelComponent(),dialog->getWidth(),dialog->getHeight());
    dialog->setAlwaysOnTop(true);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,entries](int result)
    {
        const auto index=dialog->getComboBoxComponent("recovery")->getSelectedItemIndex();
        delete dialog;
        if (!safe) return;
        if (result==2) { safe->projectRecovery->directory().revealToUser(); return; }
        if (result!=1 || index<0 || index>=static_cast<int>(entries.size())) return;
        const auto entry=entries[static_cast<std::size_t>(index)];
        safe->performWithUnsavedCheck([safe,entry] { if (safe) safe->restoreProjectRecovery(entry); });
    }),false);
}

void MainComponent::rememberExportDirectory(const juce::File& directory)
{
    if (!directory.isDirectory() || directory == lastExportDirectory) return;
    lastExportDirectory = directory;
    if (preferences == nullptr) return;
    preferences->setValue("files.lastExportDirectory", directory.getFullPathName());
    preferences->saveIfNeeded();
}

void MainComponent::restoreRecentProjects()
{
    recentProjectPaths.clear();
    if (preferences == nullptr) return;
    // Read here rather than in applyPreferences(): this runs once, at startup,
    // which is when the folder an export should open in is wanted.  A folder
    // that has since been moved or removed is dropped, and the chooser falls
    // back to Documents.
    const juce::File exported(preferences->getValue("files.lastExportDirectory"));
    if (exported.isDirectory()) lastExportDirectory = exported;
    juce::StringArray stored;
    stored.addLines(preferences->getValue("files.recentProjects"));
    for (const auto& path : stored)
    {
        const juce::File file(path);
        if (file.existsAsFile() && !recentProjectPaths.contains(file.getFullPathName()))
            recentProjectPaths.add(file.getFullPathName());
        if (recentProjectPaths.size() >= 8) break;
    }
}

void MainComponent::addRecentProject(const juce::File& file)
{
    if (file == juce::File{}) return;
    const auto path = file.getFullPathName();
    recentProjectPaths.removeString(path, true);
    recentProjectPaths.insert(0, path);
    while (recentProjectPaths.size() > 8) recentProjectPaths.remove(8);
    if (preferences != nullptr)
    {
        preferences->setValue("files.recentProjects", recentProjectPaths.joinIntoString("\n"));
        preferences->saveIfNeeded();
    }
    menuItemsChanged();
}

juce::File MainComponent::exportStartFile(const juce::File& remembered,
                                          const juce::String& suggestedName)
{
    const auto folder = remembered.isDirectory()
        ? remembered
        : juce::File::getSpecialLocation(juce::File::userDocumentsDirectory);
    return folder.getChildFile(juce::File::createLegalFileName(suggestedName) + ".wav");
}

std::vector<MainComponent::ExportTarget> MainComponent::exportTargets(
    const ProjectData& project, const juce::File& destination,
    const juce::String& onlyTrackId, const juce::String& untitledName)
{
    std::vector<ExportTarget> targets;
    const auto anySolo = std::any_of(project.tracks.begin(), project.tracks.end(),
                                     [](const auto& track) { return track.solo; });
    const auto stem = destination.getFileNameWithoutExtension();
    const auto folder = destination.getParentDirectory();
    for (const auto& track : project.tracks)
    {
        if (onlyTrackId.isNotEmpty() && track.id != onlyTrackId) continue;
        if (track.muted || (anySolo && !track.solo)) continue;
        if (track.clips.empty()) continue;
        auto name = track.name.trim();
        if (name.isEmpty()) name = untitledName;
        // One track was asked for by name, so it goes to the file that was
        // named.  A whole song needs one file each, and they are told apart by
        // the track they came from.
        const auto file = onlyTrackId.isNotEmpty()
            ? destination
            : folder.getChildFile(juce::File::createLegalFileName(
                  stem + " - " + name) + ".wav");
        targets.push_back({ track.id, name, file });
    }
    return targets;
}

void MainComponent::exportMixdown()
{
    const auto data = project.snapshot();
    juce::PopupMenu menu;
    menu.addItem(1, strings.text("export.allTracks"), !data.tracks.empty());
    juce::PopupMenu tracks;
    auto id = 2;
    for (const auto& track : data.tracks)
    {
        auto name = track.name.trim();
        if (name.isEmpty()) name = strings.text("export.untitledTrack");
        tracks.addItem(id++, name, !track.clips.empty());
    }
    menu.addSubMenu(strings.text("export.oneTrack"), tracks, !data.tracks.empty());
    // Where the File menu item was just clicked, rather than the corner of the
    // window.  The menu bar has closed by the time this runs, so the pointer is
    // the only record of where the choice was made; from a keyboard shortcut it
    // may be anywhere, and then the menu bar itself is the sensible anchor.
    const auto pointer = juce::Desktop::getInstance().getMainMouseSource()
                             .getScreenPosition().roundToInt();
    const auto overWindow = getScreenBounds().contains(pointer);
    const auto anchor = overWindow
        ? juce::Rectangle<int>(pointer.x, pointer.y, 1, 1)
        : menuBar.getScreenBounds().removeFromBottom(1);
    juce::Component::SafePointer<MainComponent> safe(this);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(anchor),
        [safe, data](int chosen)
        {
            if (safe == nullptr || chosen <= 0) return;
            if (chosen == 1) { safe->requestTrackExport({}); return; }
            const auto index = static_cast<std::size_t>(chosen - 2);
            if (index >= data.tracks.size()) return;
            safe->requestTrackExport(data.tracks[index].id);
        });
}

void MainComponent::requestTrackExport(const juce::String& trackId)
{
    juce::Component::SafePointer<MainComponent> safe(this);
    showWavExportOptions([safe, trackId](WavExportOptions options)
    {
        if (safe != nullptr) safe->chooseExportDestination(trackId, options);
    }, trackId);
}

void MainComponent::chooseExportDestination(const juce::String& trackId, WavExportOptions options,
                                          const std::vector<juce::String>& scopeNoteIds,
                                          juce::Range<double> range)
{
    if (exportWaitingForRender) return;
    const bool lastRender = !scopeNoteIds.empty();
    const auto data = project.snapshot();
    auto suggested = data.name;
    if (lastRender)
        suggested += " - " + strings.text("export.lastRenderSuffix");
    else if (trackId.isNotEmpty())
        for (const auto& track : data.tracks)
            if (track.id == trackId && track.name.trim().isNotEmpty())
                suggested = data.name + " - " + track.name.trim();
    chooser = std::make_unique<juce::FileChooser>(strings.text(lastRender ? "file.exportLastRender" : "file.export"),
        exportStartFile(lastExportDirectory, suggested), "*.wav");
    juce::Component::SafePointer<MainComponent> safe(this);
    chooser->launchAsync(juce::FileBrowserComponent::saveMode
                             | juce::FileBrowserComponent::warnAboutOverwriting,
        [safe, trackId, options, scopeNoteIds, range, lastRender](const juce::FileChooser& selected)
        {
            if (safe == nullptr) return;
            auto file = selected.getResult();
            if (file == juce::File{}) return;
            if (!file.hasFileExtension("wav")) file = file.withFileExtension("wav");
            // Remember where it went, not whether it worked: coming back to the
            // same folder is what saves the walk down from the drive root, and
            // a failed write is the likeliest reason to come straight back.
            safe->rememberExportDirectory(file.getParentDirectory());
            if (lastRender)
            {
                safe->startExport({ ExportTarget { {}, {}, file } }, scopeNoteIds, range, options);
                return;
            }
            auto targets = exportTargets(safe->project.snapshot(), file, trackId,
                                         safe->strings.text("export.untitledTrack"));
            if (targets.empty())
            {
                safe->showError(safe->strings.text(trackId.isNotEmpty() ? "error.exportSilent"
                                                            : "error.exportEmpty"));
                return;
            }
            safe->startExport(std::move(targets), {}, {}, options);
        });
}

void MainComponent::exportLastRender()
{
    if (lastRenderedNoteIds.empty() || lastRenderedSpan.getLength() <= 1.0e-6)
    {
        showError(strings.text("error.exportNoRender"));
        return;
    }
    juce::Component::SafePointer<MainComponent> safe(this);
    showWavExportOptions([safe, scopeNoteIds = lastRenderedNoteIds, range = lastRenderedSpan](WavExportOptions options)
    {
        if (safe != nullptr) safe->chooseExportDestination({}, options, scopeNoteIds, range);
    }, {}, lastRenderedNoteIds, lastRenderedSpan);
}

void MainComponent::showWavExportOptions(std::function<void(WavExportOptions)> confirmed,
                                       const juce::String& trackId,
                                       const std::vector<juce::String>& scopeNoteIds,
                                       juce::Range<double> range)
{
    if (exportWaitingForRender) return;
    WavExportOptions options;
    if (preferences != nullptr)
    {
        options.sampleRate = preferences->getIntValue("export.wav.sampleRate", 0);
        options.channels = preferences->getIntValue("export.wav.channels", 2);
        options.bitDepth = preferences->getIntValue("export.wav.bitDepth", 24);
        if (!options.isValid()) options = {};
    }
    auto* dialog = new juce::AlertWindow(strings.text("export.wavSettings"),
        strings.text("export.wavHelp"), juce::MessageBoxIconType::NoIcon, this);
    dialog->setComponentID("wav-export-settings");
    dialog->addComboBox("channels", { strings.text("export.mono"), strings.text("export.stereo") }, strings.text("export.channels"));
    dialog->getComboBoxComponent("channels")->setSelectedId(options.channels, juce::dontSendNotification);
    dialog->addComboBox("depth", { "16 bit PCM", "24 bit PCM", strings.text("export.float32") }, strings.text("export.bitDepth"));
    dialog->getComboBoxComponent("depth")->setSelectedId(options.bitDepth == 16 ? 1 : options.bitDepth == 24 ? 2 : 3, juce::dontSendNotification);
    dialog->addComboBox("rate", {}, strings.text("export.sampleRate"));
    auto* rates = dialog->getComboBoxComponent("rate");
    rates->addItem(strings.text("export.followDevice"), 1);
    for (const auto rate : { 8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000 })
        rates->addItem(juce::String(rate) + " Hz", rate);
    rates->setSelectedId(options.sampleRate > 0 ? options.sampleRate : 1, juce::dontSendNotification);
    if (rates->getSelectedId() == 0) rates->setSelectedId(1, juce::dontSendNotification);
    auto advanced = std::make_shared<WavExportAdvanced>(strings,
        AudioEngine::componentExportIssue(project.snapshot(), audio.currentUtauResamplerFile(),
            trackId, scopeNoteIds, range, selectedTrackId, audio.currentUtauOutputEngine()));
    dialog->addCustomComponent(advanced.get());
    advanced->onExpand = [dialog, panel = advanced.get()]
    {
        const auto centre = dialog->getBounds().getCentre();
        dialog->removeCustomComponent(0);
        panel->setSize(430, panel->preferredHeight());
        dialog->addCustomComponent(panel);
        dialog->setCentrePosition(centre);
    };
    dialog->addButton(strings.text("export.chooseDestination"), 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(strings.text("dialog.cancel"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
    dialog->centreAroundComponent(getTopLevelComponent(), dialog->getWidth(), dialog->getHeight());
    dialog->setAlwaysOnTop(true);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create(
        [safe, dialog, advanced, confirmed = std::move(confirmed)](int result) mutable
        {
            WavExportOptions selected;
            selected.component = advanced->selected();
            selected.channels = dialog->getComboBoxComponent("channels")->getSelectedId();
            const auto depth = dialog->getComboBoxComponent("depth")->getSelectedId();
            selected.bitDepth = depth == 1 ? 16 : depth == 2 ? 24 : 32;
            const auto rate = dialog->getComboBoxComponent("rate")->getSelectedId();
            selected.sampleRate = rate == 1 ? 0 : rate;
            delete dialog;
            if (safe == nullptr || result != 1 || !selected.isValid()) return;
            if (safe->preferences != nullptr)
            {
                safe->preferences->setValue("export.wav.sampleRate", selected.sampleRate);
                safe->preferences->setValue("export.wav.channels", selected.channels);
                safe->preferences->setValue("export.wav.bitDepth", selected.bitDepth);
                safe->preferences->saveIfNeeded();
            }
            // Resolve the device rate now, so pre-rendering cannot change this export's settings.
            if (selected.sampleRate == 0)
                selected.sampleRate = juce::jlimit(8000, 192000, static_cast<int>(std::lround(safe->audio.playbackSampleRate())));
            if (confirmed) confirmed(selected);
        }), false);
}

ProjectData MainComponent::projectForExport(const ProjectData& data,
    const std::vector<ExportTarget>& targets, juce::Range<double> range) const
{
    auto scoped = data;
    const auto anySolo = std::any_of(data.tracks.begin(), data.tracks.end(), [](const auto& t) { return t.solo; });
    const auto all = std::any_of(targets.begin(), targets.end(), [](const auto& t) { return t.trackId.isEmpty(); });
    std::erase_if(scoped.tracks, [&](const auto& t)
    {
        return !AudioEngine::trackIsAudible(t.muted, t.solo, anySolo, t.referenceOnly, t.id == selectedTrackId)
        || (!all && std::none_of(targets.begin(), targets.end(), [&](const auto& target) { return target.trackId == t.id; }));
    });
    if (range.getLength() > 0)
        for (auto& track : scoped.tracks)
        std::erase_if(track.clips, [&](const auto& clip)
        { return clip.startSeconds >= range.getEnd()
            || clip.startSeconds + clip.durationSeconds <= range.getStart(); });
    return scoped;
}

void MainComponent::startExport(std::vector<ExportTarget> targets,
                               const std::vector<juce::String>& scopeNoteIds,
                               juce::Range<double> range, WavExportOptions options)
{
    if (targets.empty() || exportWaitingForRender || !options.isValid()) return;
    const auto data = project.snapshot();
    if (options.component != WavExportComponent::full)
    {
        const auto issue = AudioEngine::componentExportIssue(projectForExport(data, targets, range),
            audio.currentUtauResamplerFile(), {}, scopeNoteIds, range, selectedTrackId, audio.currentUtauOutputEngine());
        if (issue.isNotEmpty()) { showError(issue); return; }
    }
    pendingExport = std::move(targets);
    pendingExportRange = range;
    pendingExportOptions = options;
    audio.stop();
    if (scopeNoteIds.empty())
    {
        // Everything in the song has to be rendered, not just whatever phrase
        // was last selected: UTAU rendering is selection-driven, so an export
        // that did not ask for the whole song would write out the last marquee
        // and silence everywhere else.
        if (const auto inScope = audio.selectEveryUtauNote(data); inScope > 0)
            activeUtauSelectionCount = inScope;
    }
    else
    {
        // The last render is exactly one marquee's worth, so the scope goes
        // back to that marquee and nothing else renders into the file.
        audio.setUtauRenderNoteSelection(scopeNoteIds);
        activeUtauSelectionCount = static_cast<int>(scopeNoteIds.size());
    }
    exportWaitingForRender = true;
    syncAudio(data);
    statusLabel.setText(strings.text("status.exportRendering"), juce::dontSendNotification);
    repaint();
}

void MainComponent::finishExport()
{
    exportWaitingForRender = false;
    const auto targets = std::exchange(pendingExport, {});
    const auto range = std::exchange(pendingExportRange, juce::Range<double>());
    statusLabel.setText(strings.text("status.exporting"), juce::dontSendNotification);
    repaint();
    auto written = 0;
    for (const auto& target : targets)
    {
        juce::String error;
        if (!audio.exportWav(target.file, error, target.trackId,
                             range.getStart(), range.getEnd(), pendingExportOptions))
        {
            showError(strings.text("error.export") + "\n" + target.trackName
                      + "\n" + error);
            break;
        }
        ++written;
    }
    statusLabel.setText(written > 0
            ? strings.text("status.exportDone") + " " + juce::String(written) + " "
                  + strings.text("status.exportFiles")
            : strings.text("status.ready"),
        juce::dontSendNotification);
    pendingExportOptions.component = WavExportComponent::full;
    // The audition scope was widened to the whole song to render it; hand it
    // back to whatever is selected in the roll.
    updateUtauRenderSelection();
    syncAudio(project.snapshot());
}

void MainComponent::importAudio()
{
    const auto importTrack = selectedTrackIsAccompaniment() ? selectedTrackId : juce::String{};
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.audio"), juce::File{},
                                                   "*.wav;*.flac;*.aif;*.aiff;*.mp3;*.ogg");
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles
                             | juce::FileBrowserComponent::canSelectMultipleItems,
        [this, importTrack](const juce::FileChooser& selected)
        {
            const auto files = selected.getResults();
            auto startSeconds = audio.position();
            for (const auto& file : files)
                if (const auto duration = audio.probeDuration(file))
                {
                    addAnalysedAudioFile(file, *duration, startSeconds, importTrack);
                    if (importTrack.isNotEmpty()) startSeconds += *duration;
                }
                else
                    showError(strings.text("error.audio") + "\n" + file.getFullPathName());
        });
}

bool MainComponent::preflightMaterialPublication(const juce::File& source,
    const std::vector<SampleRegionSetting>& rows, juce::String& error)
{
    const auto report = project.applySourceSettingsWithReport(source, rows, false);
    if (report.rejected) { error = report.diagnostics.joinIntoString("\n"); return false; }
    return true;
}

void MainComponent::materialPublicationCompleted(const juce::File& source,
    const std::vector<SampleRegionSetting>& rows)
{
    backend::UtauRenderer::invalidateVoicebankCache();
    const auto report = project.applySourceSettingsWithReport(source, rows);
    if (report.rejected || report.unresolvedNotes > 0) showError(report.diagnostics.joinIntoString("\n"));
    if (sourceEditActive && sampleSettingsFile == source) loadSampleSettings();
}

void MainComponent::openMaterialEditor(const juce::File& source, const juce::File& manifest,
    bool quick, const juce::String& pronunciationId,const juce::String& audioVariantId,
    std::optional<juce::Range<double>> sourceRange)
{
    // Resolve explicit IDs before opening/reusing any editor. Never fall back
    // to a sibling registration just because its physical WAV is the same.
    if(audioVariantId.isNotEmpty())
    {
        std::vector<SampleRegionSetting> rows;juce::String error;
        if(!OttoMaterialMigration::readAudioAnnotations(source,manifest.getParentDirectory(),UtauMode::classic,rows,error,audioVariantId))
        {showError(error);return;}
        if(pronunciationId.isNotEmpty()&&std::none_of(rows.begin(),rows.end(),[&](const auto& row){return row.materialRegionId==pronunciationId;}))
        {showError("The requested pronunciation no longer belongs to this annotation variant.");return;}
    }
    if (!quick)
    {
        if (!assetManagerVisible) setAssetManagerVisible(true);
        if (assetManager != nullptr) assetManager->editMaterial(source, pronunciationId,audioVariantId,
            {}, MaterialAnnotationSession::EditScope::source, sourceRange);
        return;
    }
    const auto key = "material-quick|" + source.getFullPathName()+(audioVariantId.isEmpty()?juce::String{}:"|"+audioVariantId);
    if (noteOtoWindows.showExisting(key)) return;
    juce::String error;
    auto editor = OtoWaveformEditorComponent::forMaterial(source, manifest, strings,
        [safe = juce::Component::SafePointer<MainComponent>(this), source](const auto& rows)
        {
            if (safe != nullptr) safe->materialPublicationCompleted(source, rows);
            if (safe != nullptr && safe->assetManager) safe->assetManager->refreshMaterialAnnotations(source);
        }, true, &error,audioVariantId);
    if (!editor) { showError(error); return; }
    editor->setPublicationValidator([safe = juce::Component::SafePointer<MainComponent>(this), source](const auto& rows, juce::String& problem)
    {
        if (safe == nullptr) { problem = "Project window was closed"; return false; }
        return safe->preflightMaterialPublication(source, rows, problem);
    });
    if (pronunciationId.isNotEmpty()) editor->selectMaterialPronunciation(pronunciationId, false);
    editor->bindMaterialViewPreferences(*preferences);
    attachOtoPlayback(*editor);
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = utf8("快速素材标注 — ") + source.getFileName();
    options.dialogBackgroundColour = Palette::panel; options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true; options.resizable = true; options.content.setOwned(editor.release());
    if (auto* window = noteOtoWindows.show(options,key)) window->setResizeLimits(900,700,1800,1200);
}

void MainComponent::openSelectedMaterialEditor(bool quick, const juce::String& requestedNote)
{
    const auto wanted = requestedNote.isNotEmpty() ? requestedNote : selectedNoteId;
    const auto data = project.snapshot();
    for (const auto& track : data.tracks) for (const auto& clip : track.clips)
    {
        const auto found = std::find_if(clip.notes.begin(),clip.notes.end(),[&](const auto& n){return n.id==wanted;});
        if (found == clip.notes.end() && !(wanted.isEmpty() && clip.id==selectedClipId)) continue;
        if (trackIsDiffSinger(track))
        { showError(utf8("神经音源没有可标注的源录音；请在素材管理器选择音频。")); return; }
        if (!trackUsesVoicebankSynthesis(track))
        {
            const ClipData* owner = &clip;
            if (!clip.parts.empty())
            {
                owner = nullptr;
                if (found != clip.notes.end())
                    for (const auto& part : clip.parts) if (part.id==found->clipPartId) owner=&part;
                if (!owner && clip.parts.size()==1) owner=&clip.parts.front();
            }
            if (!owner || !owner->sourceFile.existsAsFile())
            { showError(utf8("请选择一个有明确源录音的音符。")); return; }
            std::optional<juce::Range<double>> sourceRange;
            if (found != clip.notes.end())
            {
                auto localNote = *found;
                if (owner != &clip) localNote.startSeconds -= owner->startSeconds;
                const auto timing = nativeNoteSourceTiming(localNote, *owner, nativeClipClock(*owner));
                sourceRange = juce::Range<double>(timing.regionStartSeconds, timing.regionEndSeconds);
            }
            openMaterialEditor(owner->sourceFile, SampleSettings::annotationFor(owner->sourceFile), quick,
                found != clip.notes.end() ? found->materialRegionId : juce::String(),
                found != clip.notes.end() ? found->materialAudioId : juce::String(), sourceRange);
            return;
        }
        if (found == clip.notes.end()) { showError(utf8("请选择要编辑素材的音符。")); return; }
        const auto note = *found;
        const auto root = track.voicebankDirectory;
        const auto mode = track.utauMode;
        const auto regions = utauModeUsesRegions(track.utauMode), classes = track.utauMode==UtauMode::mou;
        // Resolve the actual recording in the existing bank service off the UI
        // thread. Never substitute the alphabetically nearest alias.
        std::thread([safe = juce::Component::SafePointer<MainComponent>(this), root, note, regions, classes, mode]
        {
            const auto sample = backend::UtauRenderer::resolveVoiceSample(root,note.label,note.midiNote,100,regions,classes);
            juce::MessageManager::callAsync([safe,root,sample,mode]
            {
                if (safe == nullptr) return;
                if (!sample.found) { safe->showError(utf8("没有找到此发音的源录音。")); return; }
                const auto key = "utau-source|" + root.getFullPathName() + "|" + juce::String(static_cast<int>(mode))
                    + "|" + sample.file.getFullPathName() + "|" + sample.resolvedAlias;
                if (safe->noteOtoWindows.showExisting(key)) return;
                juce::String error;
                auto editor = OtoWaveformEditorComponent::forVoicebankSample(root, mode, sample,
                    [safe] { if (safe) safe->refreshAfterVoicebankChange(); }, error);
                if (!editor) { safe->showError(error); return; }
                safe->attachOtoPlayback(*editor);
                juce::DialogWindow::LaunchOptions options;
                options.dialogTitle = (mode == UtauMode::mou ? utf8("谋•OTO 分区编辑器 — ")
                    : mode == UtauMode::jie ? utf8("界•OTO 四区编辑器 — ") : utf8("OTO 时序编辑器 — "))
                    + sample.file.getFileName() + " · " + sample.resolvedAlias;
                options.dialogBackgroundColour = Palette::panel;
                options.escapeKeyTriggersCloseButton = true; options.useNativeTitleBar = true; options.resizable = true;
                options.content.setOwned(editor.release());
                if (auto* window = safe->noteOtoWindows.show(options,key)) window->setResizeLimits(720,400,1800,1100);
            });
        }).detach();
        return;
    }
    showError(utf8("请先选择需要标注的音频或音符。"));
}

void MainComponent::finishMaterialPronunciationInput(bool accepted,const juce::String& error)
{
    const auto session=std::exchange(materialInputSession,{});
    if(session){session->active=false;if(session->aliasSuspended)materialAliasResumes.insert(session->aliasNoteId);}
    if(materialPronunciationObserved){materialPronunciationObserved->removeComponentListener(this);materialPronunciationObserved=nullptr;}
    if(auto* popup=materialPronunciationPopup.getComponent()){materialPronunciationPopup=nullptr;popup->setVisible(false);popup->dismiss();}
    materialPronunciationPopup=nullptr;
    const auto serial=materialInputSerial;juce::Component::SafePointer<MainComponent> safe(this);
    const auto resume=[safe,serial,accepted]
    {
        if(!safe||safe->closingUI||safe->materialInputSerial!=serial||safe->materialInputSession)return;
        auto ids=std::exchange(safe->materialAliasResumes,{});
        for(const auto& id:ids)safe->pianoRoll.resumeInlineAliasInput(id,accepted);
    };
    if(error.isNotEmpty()&&!closingUI)
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,strings.text("app.title"),error,{},this,
            juce::ModalCallbackFunction::create([resume](int){resume();}));
    else if(!closingUI)juce::MessageManager::callAsync(resume); // after callout's modal-dismiss message
}
void MainComponent::dismissMaterialPronunciationInput(){finishMaterialPronunciationInput(false);}
void MainComponent::componentVisibilityChanged(juce::Component& component)
{if(&component==materialPronunciationObserved&&!component.isVisible())finishMaterialPronunciationInput(false);}
void MainComponent::componentBeingDeleted(juce::Component& component)
{if(&component==materialPronunciationObserved){materialPronunciationObserved=nullptr;materialPronunciationPopup=nullptr;finishMaterialPronunciationInput(false);}}

void MainComponent::showMaterialPronunciationInput(MaterialInputAction action,const juce::String& noteId,const juce::String& initialQuery)
{
    if(closingUI)return;dismissMaterialPronunciationInput();
    auto session=std::make_shared<MaterialInputSession>();session->action=action;session->noteId=noteId;session->query=initialQuery;
    session->serial=++materialInputSerial;session->atSeconds=audio.position();session->trackId=selectedTrackId;
    session->aliasNoteId=action==MaterialInputAction::replace?noteId:selectedNoteId;
    session->aliasSuspended=pianoRoll.suspendInlineAliasInput(session->aliasNoteId);materialInputSession=session;
    const auto data=project.snapshot();
    if(action==MaterialInputAction::replace)
    {
        session->bindingKey=materialBindingKey(data,noteId);
        const TrackData* owner=nullptr;
        for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)if(n.id==noteId)owner=&t;
        if(!owner||session->bindingKey.isEmpty()){finishMaterialPronunciationInput(false,"The requested note/source identity is missing or ambiguous.");return;}
        if(!trackShowsAllNativeRegions(*owner)||owner->pitchAlgorithm==PitchAlgorithm::utau)
        {finishMaterialPronunciationInput(false,"This renderer has no explicit physical-recording adapter. Its engine and voicebank were left unchanged.");return;}
        session->trackId=owner->id;
    }
    std::vector<MaterialPronunciationCatalog::Root> roots;
    for(const auto& t:data.tracks)if(t.voicebankDirectory.isDirectory())roots.push_back({t.voicebankDirectory,t.utauMode});
    juce::StringArray registered;
    if(assetManager)registered=assetManager->materialRoots();
    else if(preferences)for(const auto& line:juce::StringArray::fromLines(preferences->getValue("assets.materialFolders")))
    {const auto tab=line.indexOfChar('\t');const auto path=(tab>=0?line.substring(0,tab):line).trim();if(path.isNotEmpty())registered.addIfNotAlreadyThere(path);}
    for(const auto& path:registered)
    {
        const juce::File root(path);
        if(std::none_of(roots.begin(),roots.end(),[&](const auto& r){return r.folder==root;}))roots.push_back({root,UtauMode::classic});
    }
    presentMaterialPronunciationInput(session); // same owned picker immediately, index on worker
    juce::Component::SafePointer<MainComponent> safe(this);
    std::thread([safe,session,roots=std::move(roots)]
    {
        auto result=MaterialPronunciationCatalog::index(roots,[session]{return !session->active.load();});
        juce::MessageManager::callAsync([safe,session,result=std::move(result)]()mutable
        {
            if(!safe||safe->closingUI||!session->active||safe->materialInputSession!=session)return;
            if(result.cancelled)return;
            if(!result.errors.isEmpty()){safe->finishMaterialPronunciationInput(false,result.errors.joinIntoString("\n"));return;}
            if(result.entries.empty()){safe->finishMaterialPronunciationInput(false,"No usable pronunciation was found. Register a material folder or a supported voicebank first.");return;}
            session->entries=std::move(result.entries);session->ready=true;safe->presentMaterialPronunciationInput(session);
        });
    }).detach();
}

void MainComponent::presentMaterialPronunciationInput(const std::shared_ptr<MaterialInputSession>& session)
{
    if(closingUI||materialInputSession!=session||!session->active)return;
    if(auto* old=materialPronunciationPopup.getComponent())
    {
        if(auto* previous=session->picker.getComponent())session->query=previous->queryEditor().getText();
        old->removeComponentListener(this);materialPronunciationObserved=nullptr;materialPronunciationPopup=nullptr;old->setVisible(false);old->dismiss();
    }
    std::vector<MaterialPronunciationCandidate> candidates;
    for(const auto& e:session->entries)candidates.push_back({e.audio,e.manifest,e.audioId,e.regionId,e.alias,e.row.regionStartSeconds,e.row.regionEndSeconds,e.row.pronunciationKind});
    auto box=std::make_shared<juce::Component::SafePointer<juce::CallOutBox>>();juce::Component::SafePointer<MainComponent> safe(this);
    auto picker=std::make_unique<PronunciationCandidateBox>(strings,std::move(candidates),
        [safe,session,box](const auto& candidate)
        {if(safe&&!safe->closingUI&&session->active&&safe->materialInputSession==session&&box->getComponent()==safe->materialPronunciationPopup.getComponent())safe->chooseMaterialPronunciation(session,candidate);},
        [safe,session,box]
        {if(safe&&safe->materialInputSession==session&&box->getComponent()==safe->materialPronunciationPopup.getComponent())safe->finishMaterialPronunciationInput(false);});
    picker->setQuery(session->query);
    picker->setName(session->action==MaterialInputAction::replace?"Replace note material source":"Insert material pronunciation");
    const auto anchor=pianoViewport.localAreaToGlobal(pianoViewport.getLocalBounds().removeFromTop(24));
    juce::Component::SafePointer<PronunciationCandidateBox> pickerSafe(picker.get());session->picker=pickerSafe;
    auto& callout=juce::CallOutBox::launchAsynchronously(std::move(picker),anchor,nullptr);*box=&callout;
    materialPronunciationPopup=&callout;materialPronunciationObserved=&callout;callout.addComponentListener(this);
    juce::MessageManager::callAsync([pickerSafe]{if(pickerSafe&&pickerSafe->isShowing())pickerSafe->queryEditor().grabKeyboardFocus();});
}

void MainComponent::chooseMaterialPronunciation(const std::shared_ptr<MaterialInputSession>& session,const MaterialPronunciationCandidate& candidate)
{
    if(!session->ready||session->resolving||materialInputSession!=session||!session->active)return;
    const auto match=[&](const auto& e){return e.audio==candidate.audio&&e.audioId==candidate.audioId&&e.regionId==candidate.regionId&&e.alias==candidate.alias;};
    if(std::count_if(session->entries.begin(),session->entries.end(),match)!=1){finishMaterialPronunciationInput(false,"The candidate source identity is ambiguous.");return;}
    const auto selected=*std::find_if(session->entries.begin(),session->entries.end(),match);session->resolving=true;
    juce::Component::SafePointer<MainComponent> safe(this);
    std::thread([safe,session,selected]
    {
        MaterialPronunciationCatalog::Entry current;std::vector<SampleRegionSetting> rows;juce::String error;
        const bool valid=MaterialPronunciationCatalog::resolve(selected,current,error,&rows);
        juce::MessageManager::callAsync([safe,session,current=std::move(current),rows=std::move(rows),error,valid]()mutable
        {
            if(!safe||safe->closingUI||!session->active||safe->materialInputSession!=session)return;
            if(!valid){safe->finishMaterialPronunciationInput(false,error);return;}
            juce::String failure;
            if(session->action==MaterialInputAction::insert)
            {
                const auto clip=safe->project.addMaterialPronunciations(current.audio,rows,{current.regionId},session->atSeconds,session->trackId,failure,current.alias);
                if(clip.isEmpty()){safe->finishMaterialPronunciationInput(false,failure);return;}
                safe->scheduleAnalysis(current.audio,clip);safe->focusClip(clip);safe->finishMaterialPronunciationInput(true);return;
            }
            const auto data=safe->project.snapshot();
            if(materialBindingKey(data,session->noteId)!=session->bindingKey)
            {safe->finishMaterialPronunciationInput(false,"The requested note source changed while choosing. Reopen the picker for that note.");return;}
            for(const auto& t:data.tracks)for(const auto& parent:t.clips)for(const auto& n:parent.notes)if(n.id==session->noteId)
            {
                const ClipData* owner=&parent;for(const auto& p:parent.parts)if(p.id==n.clipPartId)owner=&p;
                if(parent.notes.size()>1&&safe->originalSourceAnalysisPending(parent.id,owner->sourceFile))
                {safe->finishMaterialPronunciationInput(false,"The original recording is still being measured. Wait for that analysis before replacing one source in this group.");return;}
            }
            const auto report=safe->project.replaceNoteMaterialSource(session->noteId,current.audio,rows,current.audioId,current.regionId,current.alias,failure);
            if(report.rejected){safe->finishMaterialPronunciationInput(false,failure);return;}
            if(report.pendingAnalysis)safe->scheduleAnalysis(current.audio,{},report.ticket,true);
            safe->focusNote(session->noteId);safe->finishMaterialPronunciationInput(true);
        });
    }).detach();
}

void MainComponent::importMaterialPronunciations(const juce::File& file, const juce::File& manifest,
    const juce::StringArray& regionIds, double startSeconds, const juce::String& trackId,
    const juce::String& chosenAlias,const juce::String& audioVariantId)
{
    std::vector<SampleRegionSetting> rows;
    juce::String error;
    if (manifest == juce::File{} || !OttoMaterialMigration::readAudioAnnotations(file,
            manifest.getParentDirectory(), UtauMode::classic, rows, error,audioVariantId))
    { showError(error.isEmpty() ? "Material annotations cannot be read" : error); return; }
    if(audioVariantId.isNotEmpty()&&std::any_of(rows.begin(),rows.end(),[&](const auto& row){return row.materialAudioId!=audioVariantId;}))
    {showError("The requested annotation variant no longer matches this recording.");return;}
    std::vector<juce::String> selected;
    if (regionIds.isEmpty())
    {
        // A recording row is the default pronunciation, not a request to layer
        // every OTO alternative. Keep the annotation/list order, even collapsed.
        if (!rows.empty()) selected.push_back(rows.front().materialRegionId);
    }
    else for (const auto& id : regionIds) selected.push_back(id);
    if (selected.empty()) { showError("This recording has no annotated pronunciations"); return; }
    const auto clipId = project.addMaterialPronunciations(file, rows, selected, startSeconds, trackId, error, chosenAlias);
    if (clipId.isEmpty()) { showError(error); return; }
    scheduleAnalysis(file, clipId);
    focusClip(clipId);
}

void MainComponent::importAssetAudio(const juce::File& file, const juce::File& root, int otoKind,
                                    double startSeconds, const juce::String& trackId,
                                    juce::Point<int> menuScreenPosition,const juce::String& audioVariantId)
{
    juce::ignoreUnused(otoKind, menuScreenPosition); // No GUI read/not-read policy remains.
    std::vector<SampleRegionSetting> rows; juce::String error;
    if (!OttoMaterialMigration::readAudioAnnotations(file,root,UtauMode::classic,rows,error,audioVariantId))
    { showError(error); return; }
    if (!rows.empty()||audioVariantId.isNotEmpty())
    {
        importMaterialPronunciations(file,root.getChildFile("otto.ini"),{},startSeconds,trackId,{},audioVariantId);
        return;
    }
    const auto duration = audio.probeDuration(file);
    if (!duration) { showError(strings.text("error.audio") + "\n" + file.getFullPathName()); return; }
    juce::String targetTrack;
    for (const auto& track : project.snapshot().tracks)
        if (track.id == trackId && trackShowsAllNativeRegions(track)) targetTrack = track.id;
    focusClip(addAnalysedAudioFile(file,*duration,startSeconds,targetTrack));
}

juce::String MainComponent::addAnalysedAudioFile(const juce::File& file, double durationSeconds,
                                         double startSeconds,
                                         const juce::String& targetTrackId,
                                         const VoicebankOtoEntry* oto, UtauMode otoMode)
{
    const auto clipId = project.addAudioFile(file, durationSeconds, startSeconds, targetTrackId, oto, otoMode);
    if (clipId.isEmpty()) return {};
    setAssetManagerVisible(true);
    project.setAssetManagerVisible(true);
    const auto data = project.snapshot();
    if (assetManager) assetManager->setProjectVoicebanks(data);
    for (const auto& track : data.tracks)
        if (track.accompaniment && std::any_of(track.clips.begin(), track.clips.end(),
            [&](const auto& clip) { return clip.id == clipId; }))
        {
            focusClip(clipId);
            return clipId;
        }
    scheduleAnalysis(file, clipId);
    return clipId;
}

bool MainComponent::originalSourceAnalysisPending(const juce::String& clipId,const juce::File& file) const
{
    for(const auto& [key,task]:nativeAnalysisTasks)if(task->file==file)
        for(const auto& recipient:task->recipients)if(recipient.clipId==clipId&&!recipient.ticket)return true;
    return false;
}
void MainComponent::scheduleNativeTrimSourceAnalysis(const ProjectData& data)
{
    if(closingUI)return;
    if(materialInputSession&&materialInputSession->action==MaterialInputAction::replace
        &&materialBindingKey(data,materialInputSession->noteId)!=materialInputSession->bindingKey)
        finishMaterialPronunciationInput(false);
    std::map<juce::String,ProjectModel::MaterialSourceBindingTicket> pending;
    std::set<juce::String> live;
    for(const auto& ticket:project.pendingMaterialSourceAnalyses())
    {
        const auto key=ticket.sourceFile.getFullPathName()+":"+ticket.sourceFingerprint;live.insert(key);pending.emplace(key,ticket);
    }
    for(const auto& [key,ticket]:pending)
    {
        if(!ticket.sourceFile.existsAsFile()||juce::SHA256(ticket.sourceFile).toHexString()!=ticket.sourceFingerprint)continue;
        juce::String cacheError;
        if(project.completeMaterialSourceAnalysis(ticket,{},cacheError))continue;
        scheduleAnalysis(ticket.sourceFile,{},ticket);
    }
    std::map<juce::String,juce::File> files;
    for(const auto& track:data.tracks)if(trackShowsAllNativeRegions(track))
        for(const auto& clip:track.clips)for(const auto* source:clipSourceRegions(clip))
            if(source->nativeSourcePitchPending&&!source->materialSourceAnalysisPending&&source->sourceFile.existsAsFile())
                files[source->sourceFile.getFullPathName()]=source->sourceFile;
    for(const auto& [path,file]:files)
    {
        const auto key=path+":"+juce::SHA256(file).toHexString();live.insert(key);scheduleAnalysis(file,{});
    }
    // Undo removes the pending episode. A later redo may retry or use the model
    // hash cache; there is never a permanent path+mtime attempted blacklist.
    std::erase_if(nativeAnalysisFailures,[&](const auto& p){return !live.count(p.first)&&!nativeAnalysisTasks.count(p.first);});
}

void MainComponent::scheduleAnalysis(const juce::File& file,const juce::String& clipId,
    std::optional<ProjectModel::MaterialSourceBindingTicket> ticket,bool explicitRetry)
{
    if(closingUI||!file.existsAsFile())return;
    const auto fingerprint=juce::SHA256(file).toHexString();
    if(ticket&&fingerprint!=ticket->sourceFingerprint)return;
    const auto key=file.getFullPathName()+":"+fingerprint;
    NativeAnalysisTask::Recipient recipient;recipient.clipId=clipId;recipient.ticket=ticket;
    if(clipId.isNotEmpty())
    {
        bool found=false;const auto data=project.snapshot();
        for(const auto& t:data.tracks)for(const auto& c:t.clips)if(c.id==clipId)
        {if(t.accompaniment)return;recipient.importedClip=c;found=true;}
        if(!found)return;
    }
    if(explicitRetry){nativeAnalysisFailures.erase(key);nativeAnalysisRetryScheduled.erase(key);}
    if(const auto existing=nativeAnalysisTasks.find(key);existing!=nativeAnalysisTasks.end())
    {
        auto& targets=existing->second->recipients;
        const bool duplicate=std::any_of(targets.begin(),targets.end(),[&](const auto& r)
        {return r.clipId==clipId&&bool(r.ticket)==bool(ticket)&&(!ticket||r.ticket->sourcePartId==ticket->sourcePartId);});
        if(!duplicate)targets.push_back(std::move(recipient));return;
    }
    if(nativeAnalysisRetryScheduled.count(key)||nativeAnalysisFailures[key]>=3)return;
    auto task=std::make_shared<NativeAnalysisTask>();task->file=file;task->key=key;task->fingerprint=fingerprint;task->recipients.push_back(std::move(recipient));nativeAnalysisTasks[key]=task;
    juce::Component::SafePointer<MainComponent> safe(this);const auto config=backend::AnalysisService::configFromProperties(preferences.get());
    ++pendingNativeAnalyses;nativeAnalysisProgress=0;nativeAnalysisName=file.getFileName();
    statusLabel.setText(strings.text("status.analyzing")+"  "+file.getFileName(),juce::dontSendNotification);
    std::thread([safe,task,config]
    {
        juce::String error;backend::AnalysisResult result;
        if(!task->file.existsAsFile()||juce::SHA256(task->file).toHexString()!=task->fingerprint)error="Source changed before acoustic analysis started";
        else result=backend::AnalysisService::analyse(task->file,config,error,[safe,name=task->file.getFileName()](double value)
        {
            juce::MessageManager::callAsync([safe,name,value]{if(safe&&!safe->closingUI&&!safe->importInProgress){safe->nativeAnalysisName=name;safe->nativeAnalysisProgress=value;}});
        });
        if(!task->file.existsAsFile()||juce::SHA256(task->file).toHexString()!=task->fingerprint){error="Source changed during acoustic analysis";result.notes.clear();}
        juce::MessageManager::callAsync([safe,task,result=std::move(result),error]()mutable
        {
            if(!safe||safe->closingUI)return;
            const auto active=safe->nativeAnalysisTasks.find(task->key);
            if(active==safe->nativeAnalysisTasks.end()||active->second!=task)return;
            safe->nativeAnalysisTasks.erase(active);safe->pendingNativeAnalyses=std::max(0,safe->pendingNativeAnalyses-1);
            bool applied=false;juce::String attachError;
            const bool unchanged=task->file.existsAsFile()&&juce::SHA256(task->file).toHexString()==task->fingerprint;
            safe->lastNativeAnalysisTrace="file="+task->file.getFullPathName()+"\nsha="+task->fingerprint
                +"\nactive="+result.status.activeBackend+"\nrequested="+result.status.requestedBackend
                +"\ngame="+result.status.gameModelDirectory.getFullPathName()+"\nfcpe="+result.status.fcpeModelPath.getFullPathName()
                +"\ninference="+result.status.activeInference+"\nwarning="+result.warning+"\nerror="+error
                +"\nnotes="+juce::String(static_cast<int>(result.notes.size()))+"\nsource_unchanged="+juce::String(unchanged?1:0);
            if(!result.notes.empty()&&unchanged)
            {
                safe->lastNativeAnalysisBackend=backend::AnalysisService::backendText(result.status);
                for(const auto& r:task->recipients)
                {
                    if(r.ticket)applied|=safe->project.completeMaterialSourceAnalysis(*r.ticket,result.notes,attachError);
                    else if(r.clipId.isNotEmpty())applied|=safe->project.setClipAudioAnalysis(r.clipId,result.notes,r.importedClip);
                }
                // Same full-file facts also serve current trims/new bindings,
                // guarded by content hash. Never restore an old target snapshot.
                applied|=safe->project.setNativeTrimSourceAnalysis(task->file,result.notes,task->fingerprint);
                safe->nativeAnalysisFailures.erase(task->key);
            }
            else
            {
                const auto attempt=++safe->nativeAnalysisFailures[task->key];
                if(attempt<3&&safe->nativeAnalysisRetryScheduled.insert(task->key).second)
                    juce::Timer::callAfterDelay(2000*attempt,[safe,key=task->key]
                    {if(safe&&!safe->closingUI){safe->nativeAnalysisRetryScheduled.erase(key);safe->scheduleNativeTrimSourceAnalysis(safe->project.snapshot());}});
            }
            if(applied){safe->nativeAnalysisProgress=1;safe->statusLabel.setText(safe->strings.text("status.analysisComplete")+utf8(" · ")+backend::AnalysisService::backendText(result.status),juce::dontSendNotification);}
            else if(error.isNotEmpty()||attachError.isNotEmpty())safe->statusLabel.setText(safe->strings.text("status.analysisSkipped"),juce::dontSendNotification);
        });
    }).detach();
}

void MainComponent::importMelodyne()
{
    if (!backend::MelodyneProvider::nativeImportAvailable()
        && !backend::MelodyneProvider::experimentalSelfImportEnabled())
    {
        showError(backend::MelodyneProvider::statusText()
            + "\n当前 Melodyne 导入路径尚未启用。请使用受支持的 Melodyne 宿主接口。 ");
        return;
    }
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.melodyne"), juce::File{}, "*.mpd");
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (file == juce::File{}) return;
            loadMelodyneFile(file);
        });
}

void MainComponent::exportUst()
{
    const auto trackId=selectedTrackId;const auto data=project.snapshot();juce::String name;
    for(const auto& track:data.tracks)if(track.id==trackId&&track.compose)name=track.name;
    if(name.isEmpty()){showError(utf8("请选择一个音符轨道导出 UST。"));return;}
    auto* dialog=new juce::AlertWindow(utf8("导出当前轨道 UST"),name+utf8("\n保持原编码时，未修改的导入字段按原文写回。"),juce::MessageBoxIconType::NoIcon);
    dialog->addComboBox("encoding",{utf8("保持原编码（新轨道为 UTF-8）"),"UTF-8","Shift-JIS","GBK","Big5","GB18030"},utf8("文件编码"));
    dialog->addButton(utf8("选择保存位置"),1);dialog->addButton(strings.text("dialog.cancel"),0);
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,trackId,name](int result)
    {
        const int pages[]{0,65001,932,936,950,54936};const auto cp=pages[juce::jlimit(0,5,dialog->getComboBoxComponent("encoding")->getSelectedItemIndex())];delete dialog;
        if(result!=1||safe==nullptr)return;
        safe->chooser=std::make_unique<juce::FileChooser>(utf8("导出 UST"),exportStartFile(safe->lastExportDirectory,name).withFileExtension("ust"),"*.ust");
        safe->chooser->launchAsync(juce::FileBrowserComponent::saveMode|juce::FileBrowserComponent::warnAboutOverwriting,[safe,trackId,cp](const juce::FileChooser& selected)
        {
            if(safe==nullptr)return;auto file=selected.getResult();if(file==juce::File{})return;file=file.withFileExtension("ust");
            juce::String error;juce::StringArray warnings;if(!safe->project.exportUst(file,trackId,error,warnings,cp)){safe->showError(error);return;}
            safe->rememberExportDirectory(file.getParentDirectory());safe->statusLabel.setText(utf8("已导出 UST：")+file.getFileName(),juce::dontSendNotification);
            if(!warnings.isEmpty())juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,utf8("UST 导出说明"),warnings.joinIntoString("\n"));
        });
    }),false);
}

void MainComponent::exportMidi()
{
    // Freeze the score and selected identity before either asynchronous dialog.
    // Changing documents while choosing a filename cannot export another song.
    const auto data = std::make_shared<const ProjectData>(project.snapshot());
    juce::String currentTrack, currentName;
    for (const auto& track : data->tracks)
        if (track.id == selectedTrackId) { currentTrack = track.id; currentName = track.name; break; }
    const auto text = [this](const char*, const char* en) { return strings.text(en); };
    auto* dialog = new juce::AlertWindow(strings.text("file.exportMidi"),
        currentTrack.isEmpty() ? text("未选择轨道，默认导出整个工程。", "No current track; export defaults to the whole project.")
                               : currentName,
        juce::MessageBoxIconType::NoIcon);
    auto dialogLook = std::make_shared<HachiLookAndFeel>();
    dialog->setLookAndFeel(dialogLook.get());
    dialog->addComboBox("scope", { text("当前轨道", "Current track"), text("整个工程", "Whole project") },
                        text("导出范围", "Export scope"));
    auto* scope = dialog->getComboBoxComponent("scope");
    scope->setItemEnabled(1, currentTrack.isNotEmpty());
    scope->setSelectedId(currentTrack.isEmpty() ? 2 : 1, juce::dontSendNotification);
    // AlertWindow does not own custom components. The modal callback holds this
    // checkbox until the dialog has been deleted, including cancellation.
    auto includeMuted = std::make_shared<juce::ToggleButton>(text("包含静音轨道和片段", "Include muted tracks and clips"));
    includeMuted->setName({}); // AlertWindow otherwise repeats the checkbox text as an extra label.
    includeMuted->setSize(330, 25);
    dialog->addCustomComponent(includeMuted.get());
    dialog->addButton(text("选择保存位置", "Choose save location"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create(
        [safe, dialog, dialogLook, includeMuted, data, currentTrack, currentName](int result)
        {
            const bool currentOnly = dialog->getComboBoxComponent("scope")->getSelectedId() == 1;
            const bool muted = includeMuted->getToggleState();
            delete dialog;
            if (result != 1 || safe == nullptr) return;
            const auto trackId = currentOnly ? currentTrack : juce::String{};
            if (currentOnly && trackId.isEmpty()) return;
            const auto name = currentOnly && currentName.isNotEmpty() ? currentName : data->name;
            safe->chooser = std::make_unique<juce::FileChooser>(safe->strings.text("file.exportMidi"),
                exportStartFile(safe->lastExportDirectory, name), "*.mid");
            safe->chooser->launchAsync(juce::FileBrowserComponent::saveMode
                                          | juce::FileBrowserComponent::warnAboutOverwriting,
                [safe, data, trackId, muted](const juce::FileChooser& selected)
                {
                    if (safe == nullptr) return;
                    auto file = selected.getResult();
                    if (file == juce::File{}) return;
                    if (!file.hasFileExtension("mid;midi")) file = file.withFileExtension("mid");
                    juce::String error;
                    if (!ProjectModel::writeMidiFile(*data, file, error, trackId, muted))
                    {
                        safe->showError(safe->strings.text("error.exportMidi") + "\n" + error);
                        return;
                    }
                    safe->rememberExportDirectory(file.getParentDirectory());
                    safe->statusLabel.setText(safe->strings.text("status.midiExported") + "  " + file.getFileName(),
                                              juce::dontSendNotification);
                });
        }), false);
}

void MainComponent::importMidi()
{
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.midi"), juce::File{}, "*.mid;*.midi");
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (file == juce::File{}) return;
            juce::String error;
            if (!project.addMidiFile(file, error))
                showError(strings.text("error.midi") + "\n" + error);
        });
}

void MainComponent::importUst()
{
    chooser = std::make_unique<juce::FileChooser>(strings.text("file.ust"),
                                                  juce::File{}, "*.ust;*.ustx");
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& selected)
        {
            const auto file = selected.getResult();
            if (file == juce::File{}) return;
            loadUstFile(file);
        });
}

void MainComponent::loadUstFile(const juce::File& file)
{
    const bool choice=ustImportNeedsChoice(project.snapshot());
    auto* dialog = new juce::AlertWindow(strings.text("dialog.ustImportTitle"),
        choice?strings.text("dialog.ustImportMessage"):utf8("选择 UST 读取编码；乱码时可指定原文件编码后重新导入。"),juce::MessageBoxIconType::QuestionIcon);
    dialog->addComboBox("trackType",{"HachiShifter","UTAU","DiffSinger"},utf8("导入目标编辑界面"));
    dialog->getComboBoxComponent("trackType")->setSelectedId(1,juce::dontSendNotification);
    dialog->addComboBox("encoding",{utf8("自动识别"),"UTF-8","Shift-JIS","GBK","Big5","GB18030"},utf8("UST 编码（USTX 固定 UTF-8）"));
    dialog->getComboBoxComponent("encoding")->setEnabled(!file.hasFileExtension("ustx"));
    dialog->addButton(choice?strings.text("dialog.ustReplace"):utf8("导入"),1);
    if(choice)dialog->addButton(strings.text("dialog.ustAddTrack"),2);
    dialog->addButton(strings.text("dialog.cancel"),0,juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,juce::ModalCallbackFunction::create([safe,dialog,file,choice](int result)
    {
        const int pages[]{0,65001,932,936,950,54936};const int cp=file.hasFileExtension("ustx")?0:pages[juce::jlimit(0,5,dialog->getComboBoxComponent("encoding")->getSelectedItemIndex())];
        const TrackType destinations[]{TrackType::hachi,TrackType::utau,TrackType::diffSinger};
        const auto destination=destinations[juce::jlimit(0,2,dialog->getComboBoxComponent("trackType")->getSelectedItemIndex())];
        if(safe!=nullptr&&result==1)
        {
            const auto apply=[safe,file,cp,destination]{if(safe!=nullptr)safe->importUstFile(file,ProjectModel::UstImportMode::replaceProject,cp,destination);};
            if(choice)safe->performWithUnsavedCheck(apply);else apply();
        }
        else if(safe!=nullptr&&result==2)safe->importUstFile(file,ProjectModel::UstImportMode::addTrack,cp,destination);
        delete dialog;
    }),false);
}

void MainComponent::importUstFile(const juce::File& file,
                                  ProjectModel::UstImportMode mode, int encoding, TrackType destinationType)
{
    const auto replacing = mode == ProjectModel::UstImportMode::replaceProject;
    // Everything the old project's audio refers to is about to go; stop before
    // it does rather than let a render finish into a project that is gone.
    if (replacing) audio.stop();
    juce::String error;
    juce::StringArray warnings;
    juce::String importedTrack;
    if (!project.addUstFile(file, error, warnings, mode, &importedTrack, encoding, destinationType))
    {
        showError(strings.text("error.ust") + "\n" + error);
        return;
    }
    if (replacing)
    {
        audio.setPosition(0.0);
        audio.setUtauRenderNoteSelection({});
        activeUtauSelectionCount = 0;
        copiedClipId.clear();
        copiedNotes.clear(); copiedNativeClips.clear(); copiedNativeConnections.clear();
        discardProjectRecovery();
        liveMcpDocumentChanged();
        // The open .hjpx is not this song: saving now must ask where to put
        // it, or the previous project would be overwritten by this one.
        currentProjectFile = juce::File{};
    }
    selectedNoteId.clear();
    pianoRoll.clearNoteSelection();
    // Show what was just imported.  Keeping the old focus is how a second
    // import came to look like nothing had happened.
    const auto data = project.snapshot();
    for (const auto& track : data.tracks)
        if (track.id == importedTrack && !track.clips.empty())
        {
            focusClip(track.clips.front().id);
            break;
        }
    refreshProjectControls();
    menuItemsChanged();
    // A UST names a voicebank this application cannot resolve, and carries an
    // amplitude envelope it does not read.  Both are worth saying once, in the
    // status line, rather than leaving them to be discovered as a wrong sound.
    statusLabel.setText(warnings.isEmpty()
        ? strings.text("status.ustLoaded")
        : strings.text("status.ustLoaded") + "  |  " + warnings.joinIntoString("  |  "),
        juce::dontSendNotification);
}

void MainComponent::loadMelodyneFile(const juce::File& file)
{
    cancelProjectMaterialRegistration();
    startupLog("Import: MPD started " + file.getFullPathName());
    importInProgress = true;
    progress = 0.01;
    statusLabel.setText(strings.text("status.loading"), juce::dontSendNotification);
    juce::Component::SafePointer<MainComponent> safe(this);
    const auto recursiveMediaSearch = preferences == nullptr
        || preferences->getBoolValue("import.recursiveMedia", true);
    const auto preserveProjectEdits = preferences == nullptr
        || preferences->getBoolValue("import.preserveEdits", true);
    const auto reanalyseSourcePitch = preferences != nullptr
        && preferences->getIntValue("import.melodynePitchSource", 1) == 2;
    const auto analysisConfig = backend::AnalysisService::configFromProperties(preferences.get());
    juce::Thread::launch([safe, file, recursiveMediaSearch, preserveProjectEdits,
                          reanalyseSourcePitch, analysisConfig]
    {
        juce::String error;
        backend::MelodyneImportOptions options;
        options.recursiveMediaSearch = recursiveMediaSearch;
        options.preserveProjectEdits = preserveProjectEdits;
        auto imported = backend::MelodyneImporter::importProject(file, error,
            [safe, reanalyseSourcePitch](double value, const juce::String& stage)
            {
                const auto scaled = reanalyseSourcePitch ? value * 0.75 : value;
                juce::MessageManager::callAsync([safe, scaled, stage]
                {
                    if (safe == nullptr) return;
                    safe->progress = scaled;
                    safe->statusLabel.setText(safe->strings.text("status.loading") + "  "
                                                + safe->strings.text(juce::String("mpd.stage.") + stage),
                                              juce::dontSendNotification);
                });
            }, options);
        auto pitchReanalysed = false;
        backend::AnalysisStatus analysisStatus;
        if (imported && reanalyseSourcePitch)
        {
            juce::String pitchError;
            pitchReanalysed = backend::AnalysisService::reanalyseProjectSourcePitch(
                imported->project, analysisConfig, pitchError, [safe](double value)
                {
                    juce::MessageManager::callAsync([safe, value]
                    {
                        if (safe == nullptr) return;
                        safe->progress = 0.75 + value * 0.25;
                        safe->statusLabel.setText(
                            safe->strings.text("status.analyzing") + "  "
                                + safe->strings.text("mpd.stage.reanalyse_pitch"),
                            juce::dontSendNotification);
                    });
                }, &analysisStatus);
        }
        juce::MessageManager::callAsync([safe, imported = std::move(imported), error,
                                         reanalyseSourcePitch, pitchReanalysed,
                                         analysisStatus]() mutable
        {
            if (safe == nullptr) return;
            safe->importInProgress = false;
            startupLog("Import: parser returned; success=" + juce::String(imported.has_value() ? 1 : 0));
            safe->progress = 0.0;
            if (!imported)
            {
                safe->showError(safe->strings.text("error.mpd") + "\n" + error);
                return;
            }
            if (reanalyseSourcePitch)
            {
                safe->statusLabel.setText(safe->strings.text(
                    pitchReanalysed ? "status.analysisComplete" : "status.analysisSkipped"),
                    juce::dontSendNotification);
                safe->statusLabel.setText(safe->statusLabel.getText() + utf8(" · ")
                    + backend::AnalysisService::backendText(analysisStatus),
                    juce::dontSendNotification);
            }
            safe->presentMelodyneComposeSelection(std::move(*imported));
        });
    });
}

void MainComponent::presentMelodyneComposeSelection(backend::MelodyneImportResult imported)
{
    startupLog("Import: converting HJM");
    juce::StringArray annotationWarnings;
    SampleSettings::convertMelodyneProject(imported.project, annotationWarnings);
    startupLog("Import: HJM complete; warnings=" + juce::String(annotationWarnings.size()));
    for (const auto& warning : annotationWarnings)
        DBG("Could not convert Melodyne annotation: " + warning);

    const auto defaultAlgorithmId = defaultPitchAlgorithm(preferences != nullptr
        ? juce::File(preferences->getValue("algorithm.hifiganPath")) : juce::File{})
        == PitchAlgorithm::nsfHifigan ? 2 : 6;
    const auto algorithmId = preferences != nullptr
        ? preferences->getIntValue("import.algorithm", defaultAlgorithmId) : defaultAlgorithmId;
    const auto importedPitch = algorithmId == 2 ? PitchAlgorithm::nsfHifigan
        : algorithmId == 3 ? PitchAlgorithm::world
        : algorithmId == 4 ? PitchAlgorithm::vocalShifter
        : algorithmId == 1 ? PitchAlgorithm::mld5
        : algorithmId == 5 ? PitchAlgorithm::mld3 : PitchAlgorithm::llsm2;
    const auto stretchAlgorithmId = preferences != nullptr
        ? preferences->getIntValue("import.stretchAlgorithm", 1) : 1;
    auto importedStretch = stretchAlgorithmId == 2 ? StretchAlgorithm::variableMelHop
        : stretchAlgorithmId == 3 ? StretchAlgorithm::loop
        : stretchAlgorithmId == 4 ? StretchAlgorithm::soundTouch
        : stretchAlgorithmId == 7 ? StretchAlgorithm::wcsNonLoop
        : stretchAlgorithmId == 6 ? StretchAlgorithm::hifiShifterMel
        : stretchAlgorithmId == 5 ? StretchAlgorithm::nsfShiftThenSplice
        : StretchAlgorithm::melodyneHybrid;
    // The two NSF variable-mel-hop orders are the NSF-HiFiGAN-specific
    // duration paths.  Keep an imported project immediately renderable when
    // another pitch backend is selected in Settings, matching the toolbar's
    // available choices.
    if (importedPitch != PitchAlgorithm::nsfHifigan
        && (importedStretch == StretchAlgorithm::variableMelHop
            || importedStretch == StretchAlgorithm::nsfShiftThenSplice
            || importedStretch == StretchAlgorithm::hifiShifterMel
            || importedStretch == StretchAlgorithm::wcsNonLoop))
        importedStretch = StretchAlgorithm::melodyneHybrid;
    for (auto& track : imported.project.tracks)
    {
        track.pitchAlgorithm = importedPitch;
        track.nativeNsfAudio = importedPitch == PitchAlgorithm::nsfHifigan;
        track.trackType = TrackType::hachi;
        track.stretchAlgorithm = importedStretch;
        track.renderOrder = RenderOrder::stretchSpliceThenPitch;
    }

    const auto storedComposeMode = preferences != nullptr
        ? preferences->getIntValue("import.melodyneCompose", 1) : 1;
    const auto composeMode = storedComposeMode >= 1 && storedComposeMode <= 4
        ? storedComposeMode : 1;
    startupLog("Import: track choice mode=" + juce::String(composeMode)
        + "; stored=" + juce::String(storedComposeMode));
    if (composeMode != 1)
    {
        for (auto& track : imported.project.tracks)
        {
            if (composeMode == 3) track.compose = true;
            else if (composeMode == 4) track.compose = false;
            // Mode 2 retains the melodic classification stored by Melodyne.
        }
        performWithUnsavedCheck([this, imported=std::move(imported)]() mutable
        {
            const auto importedForFolders = imported.project;
            replaceImportedProject(std::move(imported.project));
            refreshProjectControls();
            menuItemsChanged();
            repaint();
            if (!imported.missingFiles.isEmpty())
                showError(strings.text("warning.missingMedia") + "\n"
                          + imported.missingFiles.joinIntoString("\n"));
            offerMaterialFolderForImport(importedForFolders);
        });
        return;
    }
    auto state = std::make_shared<backend::MelodyneImportResult>(std::move(imported));
    auto* selector = new ComposeTrackSelector(state->project.tracks, strings);
    auto* dialog = new juce::AlertWindow(strings.text("mpd.compose.title"),
                                          strings.text("mpd.compose.description"),
                                          juce::MessageBoxIconType::QuestionIcon, this);
    dialog->addCustomComponent(selector);
    dialog->addButton(strings.text("dialog.import"), 1);
    dialog->addButton(strings.text("dialog.cancel"), 0,
                       juce::KeyPress(juce::KeyPress::escapeKey));
    dialog->centreAroundComponent(getTopLevelComponent(), dialog->getWidth(), dialog->getHeight());
    startupLog("Import: showing track selector");
    juce::Component::SafePointer<MainComponent> safe(this);
    dialog->enterModalState(true,
        juce::ModalCallbackFunction::create([safe, dialog, selector, state](int result)
        {
            startupLog("Import: track selector result=" + juce::String(result));
            if (safe != nullptr && result == 1)
            {
                for (std::size_t index = 0; index < state->project.tracks.size(); ++index)
                    state->project.tracks[index].compose = selector->isCompose(index);
                safe->performWithUnsavedCheck([safe,state]
                {
                    if (!safe) return;
                    const auto importedForFolders = state->project;
                    safe->replaceImportedProject(std::move(state->project));
                    safe->refreshProjectControls();
                    safe->menuItemsChanged();
                    safe->repaint();
                    safe->offerMaterialFolderForImport(importedForFolders);
                });
            }
            dialog->removeCustomComponent(0);
            delete selector;
            delete dialog;
            if (safe != nullptr && result == 1 && !state->missingFiles.isEmpty())
                safe->showError(safe->strings.text("warning.missingMedia") + "\n"
                    + state->missingFiles.joinIntoString("\n"));
        }), false);
}

void MainComponent::showError(const juce::String& message)
{
    startupLog("UI warning: " + message);
    juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                                            strings.text("app.title"), message, {}, this);
}
}

#include "tests/EngineViewCapabilitiesSmoke.h"
#include "tests/ClipLengthSmoke.h"
