#include "AssetManagerComponent.h"
#include "OtoWaveformEditorComponent.h"
#include "OttoMaterialMigration.h"
#include "OttoMaterialIO.h"
#include "Pinyin.h"
#include "ProjectModel.h"
#include "Theme.h"
#include "backend/UtauRenderer.h"
#include <algorithm>
#include <juce_audio_formats/juce_audio_formats.h>

namespace hachi
{
namespace
{
constexpr const char* audioExtensions = "wav;flac;aif;aiff;mp3;ogg";
juce::String materialPathKey(const juce::File& file)
{
    auto key = file.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    key = key.toLowerCase();
#endif
    return key;
}

double probeDurationSeconds(const juce::File& file)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    auto reader = std::unique_ptr<juce::AudioFormatReader>(formats.createReaderFor(file));
    if (reader == nullptr || reader->sampleRate <= 0.0) return 0.0;
    return static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
}
}

AssetManagerComponent::AssetManagerComponent(I18n& stringsToUse,
                                             juce::PropertiesFile& propertiesToUse)
    : strings(stringsToUse), properties(propertiesToUse)
{
    addAndMakeVisible(folderActionsButton);
    folderActionsButton.setButtonText(strings.text("Material folders ▾"));
    folderActionsButton.onClick = [this] { showFolderActions(); };

    addAndMakeVisible(searchEditor);
    searchEditor.setTextToShowWhenEmpty(strings.text("asset.search"), Palette::textMuted);
    searchEditor.onTextChange = [this]
    {
        searchQuery = searchEditor.getText().trim();
        selectedMember = -1;
        scrollOffset = 0;
        clearSearchButton.setEnabled(searchEditor.getText().isNotEmpty());
        rebuildTreeRows();
    };
    searchEditor.onEscapeKey = [this] { searchEditor.setText({}); };
    addAndMakeVisible(clearSearchButton);
    clearSearchButton.setButtonText(juce::String::fromUTF8("×"));
    clearSearchButton.setTooltip(strings.text("asset.clearSearch"));
    clearSearchButton.setEnabled(false);
    clearSearchButton.onClick = [this] { searchEditor.setText({}); };

    addAndMakeVisible(treeScroll);
    treeScroll.addListener(this);

    load();
    setSize(720, 540);
}

// ---------------------------------------------------------------------------
// Persistence

void AssetManagerComponent::load()
{
    folders.clear();
    hiddenFolders.clear(); hiddenAudio.clear();
    for (const auto& path : juce::StringArray::fromLines(properties.getValue("assets.hiddenFolders")))
        if (path.isNotEmpty()) hiddenFolders.insert(path);
    for (const auto& path : juce::StringArray::fromLines(properties.getValue("assets.hiddenAudio")))
        if (path.isNotEmpty()) hiddenAudio.insert(path);
    // New folder-based registry.  A path per line, an optional display name
    // after a tab so a renamed folder keeps its label.
    const auto stored = juce::StringArray::fromLines(
        properties.getValue("assets.materialFolders"));
    for (const auto& line : stored)
    {
        if (line.trim().isEmpty()) continue;
        const auto tab = line.indexOfChar('\t');
        const auto path = tab >= 0 ? line.substring(0, tab) : line;
        const auto name = tab >= 0 ? line.substring(tab + 1) : juce::String();
        const juce::File directory(path.trim());
        if (directory.isDirectory())
            folders.push_back({ directory,
                name.trim().isNotEmpty() ? name.trim() : directory.getFileName() });
    }

    // Migrate the old flat file list: each registered file's parent folder
    // becomes a material folder, so nothing already registered is lost.
    const auto legacy = juce::StringArray::fromLines(properties.getValue("assets.files"));
    for (const auto& path : legacy)
    {
        const juce::File file(path.trim());
        if (file.existsAsFile())
        {
            const auto parent = file.getParentDirectory();
            const auto known = std::any_of(folders.begin(), folders.end(),
                [&](const auto& folder)
                {
                    return folder.directory.getFullPathName()
                        == parent.getFullPathName();
                });
            if (!known && parent.isDirectory())
                folders.push_back({ parent, parent.getFileName() });
        }
    }
    if (!legacy.isEmpty())
    {
        properties.removeValue("assets.files");
        save();
    }

    selectedFolder = folders.empty() ? -1 : 0;
    rebuildMemberList();
}

bool AssetManagerComponent::save()
{
    juce::StringArray lines;
    for (const auto& folder : folders)
        if (folder.registered)
            lines.add(folder.directory.getFullPathName() + "\t" + folder.name);
    properties.setValue("assets.materialFolders", lines.joinIntoString("\n"));
    juce::StringArray removedFolders, removedAudio;
    for (const auto& path : hiddenFolders) removedFolders.add(path);
    for (const auto& path : hiddenAudio) removedAudio.add(path);
    properties.setValue("assets.hiddenFolders", removedFolders.joinIntoString("\n"));
    properties.setValue("assets.hiddenAudio", removedAudio.joinIntoString("\n"));
    return properties.saveIfNeeded();
}

bool AssetManagerComponent::registerMaterialRoot(const juce::File& folder, juce::String& error)
{
    error.clear();
    const auto index = registerFolder(folder);
    if (index < 0 || registrationError.isNotEmpty())
    {
        error = registrationError;
        return false;
    }
    return true;
}

int AssetManagerComponent::registerFolder(const juce::File& folder,
                                          const juce::String& displayName)
{
    projectAudioDirty = true;
    registrationError.clear();
    if (!OttoMaterialMigration::ensureNativeFolder(folder, registrationError)) return -1;
    // Explicit re-import restores items removed from this browser root.
    const auto rootKey = materialPathKey(folder);
    hiddenFolders.erase(rootKey);
    std::erase_if(hiddenAudio, [&](const auto& path) { return path.startsWith(rootKey + "/"); });
    // Explicit registration also refreshes a newly published existing root,
    // even if the external writer retained the same size/timestamp.
    manifestCaches.erase(materialPathKey(folder));
    for (std::size_t index = 0; index < folders.size(); ++index)
        if (folders[index].directory == folder)
        {
            folders[index].registered = true;
            if (!save()) registrationError = "Material root retained in memory, but preferences could not be saved; retry registration: "
                + properties.getFile().getFullPathName();
            selectedFolder = static_cast<int>(index);
            rebuildMemberList();
            return selectedFolder;
        }
    folders.push_back({ folder,
        displayName.isNotEmpty() ? displayName : folder.getFileName() });
    if (!save()) registrationError = "Material root retained in memory, but preferences could not be saved; retry registration: "
        + properties.getFile().getFullPathName();
    selectedFolder = static_cast<int>(folders.size()) - 1;
    rebuildMemberList();
    return selectedFolder;
}

// ---------------------------------------------------------------------------
// Members and auto-detection

bool AssetManagerComponent::setProjectVoicebanks(const ProjectData& project)
{
    const auto banksChanged = updateProjectVoicebanks(project);
    return updateProjectAudio(project, banksChanged) || banksChanged;
}

bool AssetManagerComponent::updateProjectAudio(const ProjectData& project, bool force)
{
    std::vector<juce::File> sources;
    const auto collect = [&](const juce::File& file)
    {
        if (file != juce::File{} && file.hasFileExtension(audioExtensions)
            && std::find(sources.begin(), sources.end(), file) == sources.end()) sources.push_back(file);
    };
    const auto collectClip = [&](const auto& self, const ClipData& clip) -> void
    {
        collect(clip.sourceFile);
        for (const auto& part : clip.parts) self(self, part);
    };
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips) collectClip(collectClip, clip);
    std::sort(sources.begin(), sources.end(), [](const auto& a, const auto& b)
        { return materialPathKey(a) < materialPathKey(b); });
    if (!force && !projectAudioDirty && sources == projectAudioSources) return false;
    projectAudioDirty = false;

    const auto previous = juce::isPositiveAndBelow(selectedFolder, static_cast<int>(folders.size()))
        ? folders[static_cast<std::size_t>(selectedFolder)].directory : juce::File{};
    for (auto& folder : folders)
    {
        folder.projectAudio = false;
        if (!folder.registered && !folder.projectVoicebank) folder.children.clear();
    }
    for (const auto& source : sources)
    {
        // Reuse an explicitly registered authority; otherwise use the existing
        // native manifest, or the source parent. Never register/scan that parent.
        juce::File root;
        for (const auto& folder : folders)
            if ((folder.registered || folder.projectVoicebank) && source.isAChildOf(folder.directory)
                && (root == juce::File{} || folder.directory.isAChildOf(root))) root = folder.directory;
        if (root == juce::File{})
        {
            const auto manifest = OttoMaterialIO::findManifest(source);
            root = manifest == juce::File{} ? source.getParentDirectory() : manifest.getParentDirectory();
        }
        auto found = std::find_if(folders.begin(), folders.end(), [&](const auto& f) { return f.directory == root; });
        if (found == folders.end())
        {
            MaterialFolder folder; folder.directory = root; folder.name = root.getFileName();
            folder.registered = false; folder.childrenLoaded = true;
            folders.push_back(std::move(folder)); found = std::prev(folders.end());
        }
        found->projectAudio = true;
        const bool added = std::find(projectAudioSources.begin(), projectAudioSources.end(), source) == projectAudioSources.end();
        if (added) found->expanded = true;
        if (!found->childrenLoaded)
        {
            found->children = membersOf(root); // Only an already registered bank.
            found->childrenLoaded = true;
        }
        if (std::find(found->children.begin(), found->children.end(), source) == found->children.end())
            found->children.push_back(source);
    }
    folders.erase(std::remove_if(folders.begin(), folders.end(), [](const auto& f)
        { return !f.registered && !f.projectVoicebank && !f.projectAudio; }), folders.end());
    std::stable_partition(folders.begin(), folders.end(), [](const auto& f) { return f.projectAudio; });
    selectedFolder = folders.empty() ? -1 : 0;
    for (int i = 0; i < static_cast<int>(folders.size()); ++i)
        if (folders[static_cast<std::size_t>(i)].directory == previous) selectedFolder = i;
    projectAudioSources = std::move(sources);
    rebuildMemberList(false);
    return true;
}

bool AssetManagerComponent::updateProjectVoicebanks(const ProjectData& project)
{
    std::vector<juce::File> banks;
    for (const auto& track : project.tracks)
        if (trackUsesVoicebankSynthesis(track) && !trackIsDiffSinger(track)
            && track.voicebankDirectory != juce::File{}
            && std::find(banks.begin(), banks.end(), track.voicebankDirectory) == banks.end())
            banks.push_back(track.voicebankDirectory);

    // Do not copy expanded banks (potentially thousands of files) on each edit.
    if (static_cast<std::size_t>(std::count_if(folders.begin(), folders.end(),
            [](const auto& folder) { return folder.projectVoicebank; })) == banks.size()
        && std::all_of(banks.begin(), banks.end(), [&](const auto& bank)
            { return std::any_of(folders.begin(), folders.end(), [&](const auto& folder)
                { return folder.projectVoicebank && folder.directory == bank; }); }))
        return false;

    auto updated = folders;
    for (auto& folder : updated)
        folder.projectVoicebank = std::find(banks.begin(), banks.end(), folder.directory) != banks.end();
    updated.erase(std::remove_if(updated.begin(), updated.end(), [](const auto& folder)
        { return !folder.registered && !folder.projectVoicebank && !folder.projectAudio; }), updated.end());
    for (const auto& bank : banks)
        if (std::none_of(updated.begin(), updated.end(), [&](const auto& folder)
                { return folder.directory == bank; }))
            updated.push_back({ bank, bank.getFileName(), false, true });
    std::stable_partition(updated.begin(), updated.end(), [](const auto& folder)
        { return folder.projectVoicebank; });
    if (updated.size() == folders.size()
        && std::equal(updated.begin(), updated.end(), folders.begin(), [](const auto& a, const auto& b)
            { return a.directory == b.directory && a.projectVoicebank == b.projectVoicebank; }))
        return false;

    const auto previous = selectedFolder >= 0 && selectedFolder < static_cast<int>(folders.size())
        ? folders[static_cast<std::size_t>(selectedFolder)].directory : juce::File{};
    folders = std::move(updated);
    selectedFolder = folders.empty() ? -1 : 0;
    for (int i = 0; i < static_cast<int>(folders.size()); ++i)
        if (folders[static_cast<std::size_t>(i)].directory == previous)
            selectedFolder = i;
    const auto sameSelection = selectedFolder >= 0
        && folders[static_cast<std::size_t>(selectedFolder)].directory == previous;
    if (!sameSelection)
        rebuildMemberList();
    else
    {
        rebuildTreeRows();
    }
    return true;
}

juce::StringArray AssetManagerComponent::diagnosticProjectVoicebanks() const
{
    juce::StringArray result;
    for (const auto& folder : folders)
        if (folder.projectVoicebank) result.add(folder.directory.getFullPathName());
    return result;
}

std::vector<juce::File> AssetManagerComponent::membersOf(const juce::File& folder)
{
    std::vector<juce::File> result;
    if (!folder.isDirectory()) return result;
    juce::Array<juce::File> found;
    folder.findChildFiles(found, juce::File::findFiles, true, "*");
    for (const auto& file : found)
        if (file.hasFileExtension(audioExtensions))
            result.push_back(file);
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right)
    {
        return left.getFullPathName().compareNatural(right.getFullPathName()) < 0;
    });
    return result;
}

void AssetManagerComponent::rebuildMemberList(bool refresh)
{
    const auto previousAudio = diagnosticSelectedAudio();
    members.clear();
    selectedMember = -1;
    if (selectedFolder >= 0 && selectedFolder < static_cast<int>(folders.size()))
    {
        auto& folder = folders[static_cast<std::size_t>(selectedFolder)];
        if ((refresh || !folder.childrenLoaded) && (!folder.projectAudio || folder.registered || folder.projectVoicebank))
            folder.children = membersOf(folder.directory);
        folder.childrenLoaded = true;
        members = folder.children;
        const auto previous = std::find(members.begin(), members.end(), previousAudio);
        if (previous != members.end()) selectedMember = static_cast<int>(previous - members.begin());
    }
    rebuildTreeRows();
}

juce::File AssetManagerComponent::takeAudioIntoFolder(const juce::File& source,
                                                      const juce::File& folder,
                                                      juce::String& error)
{
    if (!source.existsAsFile() || !source.hasFileExtension(audioExtensions))
    {
        error = source.getFileName() + ": not an audio file";
        return {};
    }
    if (!folder.isDirectory() && !folder.createDirectory())
    {
        error = "Could not create " + folder.getFullPathName();
        return {};
    }
    auto destination = folder.getChildFile(source.getFileName());
    // Register-in-place when the audio is already inside the folder; otherwise
    // copy it in so the material folder is self-contained and reusable.
    if (source.getParentDirectory().getFullPathName() != folder.getFullPathName())
    {
        destination = folder.getNonexistentChildFile(
            source.getFileNameWithoutExtension(), source.getFileExtension(), false);
        if (!source.copyFileTo(destination))
        {
            error = "Could not copy " + source.getFileName();
            return {};
        }
    }
    // Taking in audio never invents a pronunciation or writes a sidecar.
    // Explicit range/alias creation in the shared editor supplies annotations.
    if (!OttoMaterialMigration::ensureNativeFolder(folder, error)) return destination;
    // Preserve existing native annotations when copying an authored recording;
    // importing bare PCM still creates no guessed pronunciation. Do not publish
    // a partial OTTO that would preempt an existing classic OTO destination.
    if (source != destination && folder.getChildFile("otto.ini").existsAsFile())
        if (!SampleSettings::copyAnnotations(source, destination, error)) return destination;
    return destination;
}

// ---------------------------------------------------------------------------
// Toolbar actions

void AssetManagerComponent::addCommunityDatasetImportAction(juce::PopupMenu& menu) const
{
    menu.addItem(communityDatasetImportAction,
        strings.getLanguage() == I18n::Language::zhCN
            ? juce::String::fromUTF8("导入社区 DiffSinger 数据集…") : "Import community DiffSinger dataset...",
        onRequestCommunityDatasetImport != nullptr);
}

bool AssetManagerComponent::requestCommunityDatasetImport(const juce::File& initialRoot)
{
    const auto callback = onRequestCommunityDatasetImport;
    if (!callback) return false;
    callback(initialRoot); // Host may close this manager; do not access it afterwards.
    return true;
}

juce::PopupMenu AssetManagerComponent::diagnosticCommunityDatasetImportMenu() const
{
    juce::PopupMenu menu;
    addCommunityDatasetImportAction(menu);
    return menu;
}

void AssetManagerComponent::showFolderActions()
{
    const auto selected = juce::isPositiveAndBelow(selectedFolder, static_cast<int>(folders.size()));
    juce::PopupMenu menu;
    menu.addItem(1, strings.text("asset.importFolder"));
    // Community dataset UI is postponed; keep the host seam dormant.
    menu.addItem(2, strings.text("asset.newFolder"));
    menu.addSeparator();
    menu.addItem(3, strings.text("asset.addAudio"), selected);
    menu.addItem(4, strings.text("asset.assemble"), selected);
    menu.addItem(5, strings.text("asset.remove"), selected);
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&folderActionsButton),
        [safe = juce::Component::SafePointer<AssetManagerComponent>(this)](int result)
        {
            if (safe == nullptr) return;
            if (result == communityDatasetImportAction) safe->requestCommunityDatasetImport({});
            else if (result == 1) safe->importAudioFolderDialog();
            else if (result == 2) safe->createFolderDialog();
            else if (result == 3) safe->addAudioToSelectedFolderDialog();
            else if (result == 4) safe->assembleLyricsDialog();
            else if (result == 5) safe->removeSelectedFolder();
        });
}

void AssetManagerComponent::createFolderDialog()
{
    chooser = std::make_unique<juce::FileChooser>(
        strings.text("asset.newFolder"), juce::File{});
    chooser->launchAsync(juce::FileBrowserComponent::saveMode
                             | juce::FileBrowserComponent::canSelectDirectories,
        [this](const juce::FileChooser& chosen)
        {
            auto folder = chosen.getResult();
            if (folder == juce::File{}) return;
            if (!folder.isDirectory() && !folder.createDirectory())
            {
                juce::AlertWindow::showMessageBoxAsync(
                    juce::MessageBoxIconType::WarningIcon,
                    strings.text("asset.newFolder"),
                    "Could not create " + folder.getFullPathName());
                return;
            }
            if (registerFolder(folder) < 0)
                juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                    strings.text("asset.newFolder"), registrationError);
        });
}

void AssetManagerComponent::importAudioFolderDialog()
{
    chooser = std::make_unique<juce::FileChooser>(
        strings.text("asset.importFolder"), juce::File{});
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectDirectories,
        [this](const juce::FileChooser& chosen)
        {
            const auto directory = chosen.getResult();
            if (!directory.isDirectory()) return;
            const auto index = registerFolder(directory);
            if (index < 0)
            {
                juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                    strings.text("asset.importFolder"), registrationError);
                return;
            }
            if (!folders[static_cast<std::size_t>(index)].expanded) toggleFolder(index);
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
                strings.text("asset.importFolder"), strings.text("asset.utauDone")
                    .replace("{files}", juce::String(members.size())));
        });
}

void AssetManagerComponent::addAudioToSelectedFolderDialog()
{
    if (selectedFolder < 0 || selectedFolder >= static_cast<int>(folders.size()))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
            strings.text("asset.addAudio"), strings.text("asset.pickFolderFirst"));
        return;
    }
    const auto folder = folders[static_cast<std::size_t>(selectedFolder)].directory;
    chooser = std::make_unique<juce::FileChooser>(strings.text("asset.addAudio"),
        juce::File{}, juce::String("*.") + juce::String(audioExtensions).replace(";", ";*."));
    chooser->launchAsync(juce::FileBrowserComponent::openMode
                             | juce::FileBrowserComponent::canSelectFiles
                             | juce::FileBrowserComponent::canSelectMultipleItems,
        [this, folder](const juce::FileChooser& chosen)
        {
            juce::StringArray failures;
            for (const auto& file : chosen.getResults())
            {
                juce::String error;
                takeAudioIntoFolder(file, folder, error);
                if (error.isNotEmpty()) failures.add(error);
            }
            rebuildMemberList();
            if (!failures.isEmpty())
                juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                    strings.text("asset.addAudio"), failures.joinIntoString("\n"));
        });
}

// ---------------------------------------------------------------------------
// 活字印刷: lyrics -> pinyin -> match a source folder -> a new ordered folder

juce::String AssetManagerComponent::pinyinMatchKey(const juce::String& stem)
{
    // The filename may already be pinyin, or a Chinese character, or carry a
    // tone digit / diacritic.  Convert any Chinese first, then strip tone
    // digits and lower case so ba / ba2 / bà all reduce to the same key.
    auto key = lyricInPinyin(stem).toLowerCase();
    juce::String cleaned;
    for (auto character : key)
        if (!juce::CharacterFunctions::isDigit(static_cast<juce::juce_wchar>(character)))
            cleaned += juce::String::charToString(character);
    return cleaned.trim();
}

juce::File AssetManagerComponent::diagnosticAssembleLyrics(int sourceFolderIndex,
    const juce::String& lyrics, const juce::String& newFolderName,
    int& matched, int& missing, juce::String& error)
{
    matched = 0;
    missing = 0;
    if (sourceFolderIndex < 0 || sourceFolderIndex >= static_cast<int>(folders.size()))
    {
        error = "no source folder";
        return {};
    }
    const auto source = folders[static_cast<std::size_t>(sourceFolderIndex)].directory;
    const auto sourceMembers = membersOf(source);
    if (sourceMembers.empty())
    {
        error = "source folder has no audio";
        return {};
    }

    // Build a lookup from normalised pinyin key to the first member that has it.
    // Chinese characters are matched directly too, so a bank named in 汉字 works.
    std::map<juce::String, juce::File> byKey;
    for (const auto& member : sourceMembers)
    {
        const auto stem = member.getFileNameWithoutExtension();
        const auto key = pinyinMatchKey(stem);
        if (key.isNotEmpty()) byKey.emplace(key, member);
    }

    const auto name = newFolderName.trim().isNotEmpty()
        ? newFolderName.trim() : juce::String("lyrics");
    const auto destination = copyRoot().getChildFile("lyrics-assemblies")
        .getChildFile(name);
    if (!destination.deleteRecursively() || !destination.createDirectory())
    {
        error = "could not create " + destination.getFullPathName();
        return {};
    }

    auto ordinal = 0;
    for (auto character : lyrics)
    {
        const auto wide = static_cast<juce::juce_wchar>(character);
        if (juce::CharacterFunctions::isWhitespace(wide)) continue;
        const auto single = juce::String::charToString(wide);
        const auto key = pinyinMatchKey(single);
        if (key.isEmpty()) continue;
        ++ordinal;
        const auto found = byKey.find(key);
        if (found == byKey.end()) { ++missing; continue; }
        const auto ordinalText = juce::String(ordinal).paddedLeft('0', 3);
        const auto target = destination.getChildFile(
            ordinalText + "_" + single + "_" + key + found->second.getFileExtension());
        if (found->second.copyFileTo(target))
        {
            // Copy one recording's rows into the new folder manifest, never
            // copy an entire bank's otto.ini alongside an unrelated audio file.
            if (!SampleSettings::copyAnnotations(found->second, target, error))
                return {};
            ++matched;
        }
        else ++missing;
    }
    if (matched == 0)
    {
        error = "no lyric matched the source folder";
        destination.deleteRecursively();
        return {};
    }
    return destination;
}

void AssetManagerComponent::assembleLyricsDialog()
{
    if (selectedFolder < 0 || selectedFolder >= static_cast<int>(folders.size()))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
            strings.text("asset.assemble"), strings.text("asset.pickFolderFirst"));
        return;
    }
    const auto sourceIndex = selectedFolder;
    dialog = std::make_unique<juce::AlertWindow>(strings.text("asset.assemble"),
        strings.text("asset.assembleHint"), juce::MessageBoxIconType::NoIcon);
    dialog->addTextEditor("name", "", strings.text("asset.assembleName"));
    dialog->addTextEditor("lyrics", "", strings.text("asset.assembleLyrics"), true);
    if (auto* editor = dialog->getTextEditor("lyrics"))
    {
        editor->setMultiLine(true);
        editor->setReturnKeyStartsNewLine(true);
        editor->setSize(360, 120);
    }
    dialog->addButton(strings.text("dialog.apply"), 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton(strings.text("dialog.cancel"), 0, juce::KeyPress(juce::KeyPress::escapeKey));
    juce::Component::SafePointer<AssetManagerComponent> safe(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create(
        [safe, sourceIndex](int result)
        {
            if (safe == nullptr) return;
            auto owned = std::move(safe->dialog);
            if (result != 1 || owned == nullptr) return;
            const auto name = owned->getTextEditorContents("name");
            const auto lyrics = owned->getTextEditorContents("lyrics");
            int matched = 0, missing = 0;
            juce::String error;
            const auto created = safe->diagnosticAssembleLyrics(sourceIndex, lyrics,
                name, matched, missing, error);
            if (created == juce::File{})
            {
                juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                    safe->strings.text("asset.assemble"), error);
                return;
            }
            safe->registerFolder(created);
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
                safe->strings.text("asset.assemble"),
                safe->strings.text("asset.assembleDone")
                    .replace("{matched}", juce::String(matched))
                    .replace("{missing}", juce::String(missing)));
        }), false);
}

int AssetManagerComponent::exportFolderAsOto(const juce::File& folder,
                                             const juce::File& otoFile,
                                             juce::String& error)
{
    OttoMaterialMigration::Report report;
    if (!OttoMaterialMigration::exportFolder(folder, otoFile, false, report, error)) return -1;
    return report.aliases;
}

int AssetManagerComponent::diagnosticExportFolderAsOto(int folderIndex,
    const juce::File& otoFile, juce::String& error)
{
    if (folderIndex < 0 || folderIndex >= static_cast<int>(folders.size()))
    {
        error = "no such folder";
        return -1;
    }
    return exportFolderAsOto(folders[static_cast<std::size_t>(folderIndex)].directory,
                             otoFile, error);
}

void AssetManagerComponent::migrateFolderDialog(const juce::File& folder)
{
    const auto zh = strings.getLanguage() == I18n::Language::zhCN || strings.getLanguage() == I18n::Language::zhTW;
    const auto mode = otoImportKind(folder) == 3 ? UtauMode::jie : otoImportKind(folder) == 4 ? UtauMode::mou : UtauMode::classic;
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::QuestionIcon, "OTTO",
        strings.text("Migrate this folder to one otto.ini. Bare audio becomes one pronunciation; existing OTTO entries are kept. Original audio and legacy annotations remain unchanged."),
        strings.text("dialog.apply"), strings.text("dialog.cancel"), this,
        juce::ModalCallbackFunction::create([safe = juce::Component::SafePointer<AssetManagerComponent>(this), folder, mode](int result)
        {
            if (safe == nullptr || result != 1) return;
            OttoMaterialMigration::Report report; juce::String error;
            if (!OttoMaterialMigration::migrateFolder(folder, mode, true, report, error))
            {
                juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "OTTO", error); return;
            }
            backend::UtauRenderer::invalidateVoicebankCache();
            safe->rebuildMemberList();
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon, "OTTO",
                safe->strings.text("Audio: ") + juce::String(report.audioFiles) + safe->strings.text(" / Pronunciations: ") + juce::String(report.pronunciations));
        }));
}

void AssetManagerComponent::chooseOtoExport(const juce::File& folder)
{
    if (!folder.getChildFile("otto.ini").existsAsFile())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon, "OTTO",
            strings.text("Migrate this folder to native OTTO annotations first (folder context menu)."));
        return;
    }
    const auto zh = strings.getLanguage() == I18n::Language::zhCN || strings.getLanguage() == I18n::Language::zhTW;
    juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::QuestionIcon, strings.text("asset.exportOto"),
        strings.text("Standard UTAU OTO loses native connections, multi-vowel segmentation, transition flags and expression data. otto.ini is unchanged; existing targets merge by recording path. Continue?"),
        strings.text("dialog.apply"), strings.text("dialog.cancel"), this,
        juce::ModalCallbackFunction::create([safe = juce::Component::SafePointer<AssetManagerComponent>(this), folder](int result)
        {
            if (safe == nullptr || result != 1) return;
            safe->chooser = std::make_unique<juce::FileChooser>(safe->strings.text("asset.exportOto"),
                folder.getChildFile("oto.ini"), "*.ini");
            safe->chooser->launchAsync(juce::FileBrowserComponent::saveMode,
                [safe, folder](const juce::FileChooser& chosen)
                {
                    if (safe == nullptr) return;
                    auto file = chosen.getResult();
                    if (file == juce::File{}) return;
                    if (!file.hasFileExtension("ini")) file = file.withFileExtension("ini");
                    OttoMaterialMigration::Report report; juce::String error;
                    if (!OttoMaterialMigration::exportFolder(folder, file, false, report, error))
                        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
                            safe->strings.text("asset.exportOto"), error);
                    else
                        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::InfoIcon,
                            safe->strings.text("asset.exportOto"),
                            safe->strings.text("asset.exportOtoDone").replace("{rows}", juce::String(report.aliases))
                            + "\n" + report.warnings.joinIntoString("\n"));
                });
        }));
}

void AssetManagerComponent::removeSelectedFolder()
{
    if (selectedFolder < 0 || selectedFolder >= static_cast<int>(folders.size())) return;
    removeFromBrowser(folders[static_cast<std::size_t>(selectedFolder)].directory, true);
}

void AssetManagerComponent::removeFromBrowser(const juce::File& path, bool isFolder)
{
    // Browser visibility is independent of synthesis/project references. Never
    // delete recordings/annotations or detach an existing track's voicebank.
    if (path == juce::File{}) return;
    if (isFolder)
    {
        hiddenFolders.insert(materialPathKey(path));
        for (auto& folder : folders)
            if (folder.directory == path) folder.registered = false;
        std::erase_if(folders, [&](const auto& folder)
            { return folder.directory == path && !folder.projectVoicebank && !folder.projectAudio; });
    }
    else hiddenAudio.insert(materialPathKey(path));
    selectedFolder = -1; selectedMember = -1;
    pronunciationSelectionAudio = juce::File{}; selectedAudioVariantId.clear(); selectedRegionIds.clear();
    selectionAnchor.clear(); pronunciationSelectionActive = false;
    save();
    rebuildMemberList(false);
}

// ---------------------------------------------------------------------------
// Drag out, with copy-on-reuse protection

juce::File AssetManagerComponent::copyRoot() const
{
    return properties.getFile().getParentDirectory().getChildFile("Copy");
}

juce::File AssetManagerComponent::dragFileFor(const juce::File& member)
{
    const auto key = member.getFullPathName();
    if (draggedOnce.find(key) == draggedOnce.end())
    {
        // First drag: the original, and remembered so the next is a copy.
        draggedOnce.insert(key);
        return member;
    }
    // A repeat drag hands over a copy in the user data area, so an external
    // editor that rewrites the file never touches the registered original.
    const auto root = copyRoot();
    if (!root.isDirectory() && !root.createDirectory()) return member;
    const auto copy = root.getNonexistentChildFile(
        member.getFileNameWithoutExtension(), member.getFileExtension(), false);
    if (!member.copyFileTo(copy)) return member;
    juce::String annotationError;
    if (!SampleSettings::copyAnnotations(member, copy, annotationError))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            "Material copy failed", annotationError);
        return member;
    }
    return copy;
}

// ---------------------------------------------------------------------------
// Drag in

bool AssetManagerComponent::isInterestedInFileDrag(const juce::StringArray& files)
{
    return std::any_of(files.begin(), files.end(), [](const auto& path)
    {
        const juce::File file(path);
        return file.isDirectory() || file.hasFileExtension(audioExtensions);
    });
}

void AssetManagerComponent::filesDropped(const juce::StringArray& files, int, int)
{
    juce::StringArray audioPaths;
    for (const auto& path : files)
    {
        const juce::File file(path);
        if (file.isDirectory()) registerFolder(file);
        else if (file.hasFileExtension(audioExtensions)) audioPaths.add(path);
    }
    // Bare audio drops need a folder to live in.  With one selected they join
    // it (and are detected); with none, the parent directory is registered.
    if (!audioPaths.isEmpty())
    {
        if (selectedFolder >= 0 && selectedFolder < static_cast<int>(folders.size()))
        {
            const auto folder = folders[static_cast<std::size_t>(selectedFolder)].directory;
            for (const auto& path : audioPaths)
            {
                juce::String error;
                takeAudioIntoFolder(juce::File(path), folder, error);
            }
        }
        else
        {
            for (const auto& path : audioPaths)
                registerFolder(juce::File(path).getParentDirectory());
        }
    }
    rebuildMemberList();
}

// ---------------------------------------------------------------------------
// Layout and interaction

const AssetManagerComponent::ManifestCache& AssetManagerComponent::ensureManifest(const juce::File& root) const
{
    auto& cache = manifestCaches[materialPathKey(root)];
    const auto file = root.getChildFile("otto.ini");
    const auto exists = file.exists();
    const auto modified = exists ? file.getLastModificationTime().toMilliseconds() : 0;
    const auto size = exists ? file.getSize() : -1;
    if (cache.loaded && cache.exists == exists && cache.modified == modified && cache.size == size) return cache;
    cache = {};
    cache.loaded = true; cache.exists = exists; cache.modified = modified; cache.size = size;
    if (exists)
    {
        OttoMaterialIO::Document document;
        ++manifestLoadCount;
        if (OttoMaterialIO::loadDocument(file, document, cache.error))
            for (auto& audio : document.audio)
                cache.recordings[materialPathKey(root.getChildFile(audio.relativePath))].push_back({audio.id, std::move(audio.rows)});
    }
    return cache;
}

void AssetManagerComponent::ensureOtoRows(const juce::File& audio)
{
    const auto root = rootForAudio(audio);
    if (ensureManifest(root).exists) return; // Root OTTO suppresses every OTO fallback.
    const auto rootKey = materialPathKey(root), key = materialPathKey(audio);
    auto& cache = manifestCaches[rootKey];
    juce::String revision;
    for (auto folder = audio.getParentDirectory();; folder = folder.getParentDirectory())
    {
        const auto oto = folder.getChildFile("oto.ini");
        revision += materialPathKey(oto) + ":" + juce::String(oto.getLastModificationTime().toMilliseconds())
            + ":" + juce::String(oto.existsAsFile() ? oto.getSize() : -1) + "|";
        if (folder == root || folder.getParentDirectory() == folder) break;
    }
    if (cache.audioRevisions[key] == revision
        && (cache.recordings.count(key) || cache.audioErrors.count(key) || cache.pendingAudio.count(key))) return;
    cache.audioRevisions[key] = revision;
    cache.recordings.erase(key); cache.audioErrors.erase(key); cache.pendingAudio.insert(key);
    const auto request = ++nextAnnotationRequest;
    cache.audioRequests[key] = request;
    searchPool.addJob([safe = juce::Component::SafePointer<AssetManagerComponent>(this), audio, root, rootKey, key, request]
    {
        std::vector<SampleRegionSetting> rows; juce::String error;
        OttoMaterialMigration::readAudioAnnotations(audio, root, UtauMode::classic, rows, error);
        juce::MessageManager::callAsync([safe, root, rootKey, key, request, rows = std::move(rows), error]() mutable
        {
            if (safe == nullptr) return;
            if (root.getChildFile("otto.ini").exists())
            { safe->ensureManifest(root); safe->rebuildTreeRows(); return; }
            const auto found = safe->manifestCaches.find(rootKey);
            if (found == safe->manifestCaches.end() || found->second.audioRequests[key] != request) return;
            if (std::none_of(safe->folders.begin(), safe->folders.end(),
                    [&](const auto& folder) { return folder.directory == root; }))
            { safe->manifestCaches.erase(found); return; }
            auto& cache = found->second;
            cache.pendingAudio.erase(key);
            if (error.isNotEmpty()) cache.audioErrors[key] = error;
            else
            {
                const auto id = rows.empty() ? juce::String{} : rows.front().materialAudioId;
                cache.recordings[key] = {{id, std::move(rows)}};
            }
            safe->rebuildTreeRows();
        });
    });
}

bool AssetManagerComponent::resolveVariantId(const juce::File& audio, const juce::String& requested,
                                             juce::String& resolved) const
{
    const auto& cache = ensureManifest(rootForAudio(audio));
    if (cache.error.isNotEmpty()) return false;
    const auto found = cache.recordings.find(materialPathKey(audio));
    if (found == cache.recordings.end() || found->second.empty())
    { if (requested.isNotEmpty()) return false; resolved.clear(); return true; }
    if (requested.isNotEmpty())
    {
        for (const auto& variant : found->second) if (variant.audioId == requested)
        { resolved = requested; return true; }
        return false;
    }
    if (found->second.size() != 1) return false;
    resolved = found->second.front().audioId; return true;
}

const std::vector<SampleRegionSetting>* AssetManagerComponent::cachedRows(
    const juce::File& audio, const juce::String& audioVariantId) const
{
    juce::String id;
    if (!resolveVariantId(audio, audioVariantId, id)) return nullptr;
    const auto folder = manifestCaches.find(materialPathKey(rootForAudio(audio)));
    if (folder == manifestCaches.end()) return nullptr;
    const auto found = folder->second.recordings.find(materialPathKey(audio));
    if (found == folder->second.recordings.end()) return nullptr;
    for (const auto& variant : found->second) if (variant.audioId == id) return &variant.rows;
    return nullptr;
}

void AssetManagerComponent::prunePronunciationSelection()
{
    if (pronunciationSelectionAudio == juce::File{}) return;
    const auto cache = manifestCaches.find(materialPathKey(rootForAudio(pronunciationSelectionAudio)));
    if (cache == manifestCaches.end() || cache->second.error.isNotEmpty()
        || cache->second.pendingAudio.count(materialPathKey(pronunciationSelectionAudio))
        || cache->second.audioErrors.count(materialPathKey(pronunciationSelectionAudio))) return; // Unknown is not deletion.
    const auto* rows = cachedRows(pronunciationSelectionAudio, selectedAudioVariantId);
    if (rows == nullptr)
    {
        selectedRegionIds.clear(); selectionAnchor.clear(); return;
    }
    for (int i = selectedRegionIds.size(); --i >= 0;)
        if (std::none_of(rows->begin(), rows->end(), [&](const auto& row) { return row.materialRegionId == selectedRegionIds[i]; }))
            selectedRegionIds.remove(i);
    if (std::none_of(rows->begin(), rows->end(), [&](const auto& row) { return row.materialRegionId == selectionAnchor; }))
        selectionAnchor.clear();
}

void AssetManagerComponent::toggleAudio(int folderIndex, int memberIndex, const juce::String& audioVariantId)
{
    if (!juce::isPositiveAndBelow(folderIndex, static_cast<int>(folders.size()))) return;
    auto& folder = folders[static_cast<std::size_t>(folderIndex)];
    if (!juce::isPositiveAndBelow(memberIndex, static_cast<int>(folder.children.size()))) return;
    const auto& audio = folder.children[static_cast<std::size_t>(memberIndex)];
    juce::String id;
    if (!resolveVariantId(audio, audioVariantId, id)) return;
    const auto key = materialPathKey(audio) + "\n" + id;
    if (expandedAudio.erase(key) == 0)
    {
        ensureManifest(folder.directory);
        expandedAudio.insert(key);
    }
    rebuildTreeRows();
}

bool AssetManagerComponent::selectPronunciation(const juce::File& audio, const juce::String& id,
                                               bool control, bool shift, bool preserveGroup, const juce::String& audioVariantId)
{
    ensureManifest(rootForAudio(audio));
    juce::String variantId;
    if (!resolveVariantId(audio, audioVariantId, variantId)) return false;
    const auto* rows = cachedRows(audio, variantId);
    if (rows == nullptr) return false;
    const auto current = std::find_if(rows->begin(), rows->end(), [&](const auto& row) { return row.materialRegionId == id; });
    if (current == rows->end()) return false;
    if (pronunciationSelectionAudio != audio || selectedAudioVariantId != variantId)
    { pronunciationSelectionAudio = audio; selectedAudioVariantId = variantId; selectedRegionIds.clear(); selectionAnchor.clear(); }
    if (shift && selectionAnchor.isNotEmpty())
    {
        const auto anchor = std::find_if(rows->begin(), rows->end(), [&](const auto& row) { return row.materialRegionId == selectionAnchor; });
        if (!control) selectedRegionIds.clear();
        const auto a = anchor == rows->end() ? current - rows->begin() : anchor - rows->begin();
        const auto b = current - rows->begin();
        for (auto i = std::min(a, b); i <= std::max(a, b); ++i)
            selectedRegionIds.addIfNotAlreadyThere((*rows)[static_cast<std::size_t>(i)].materialRegionId);
    }
    else if (control)
    {
        if (selectedRegionIds.contains(id)) selectedRegionIds.removeString(id); else selectedRegionIds.add(id);
        selectionAnchor = id;
    }
    else
    {
        if (!(preserveGroup && selectedRegionIds.contains(id))) { selectedRegionIds.clear(); selectedRegionIds.add(id); }
        selectionAnchor = id;
    }
    pronunciationSelectionActive = true;
    // Payload order follows authored pronunciation order, not click order.
    juce::StringArray ordered;
    for (const auto& row : *rows) if (selectedRegionIds.contains(row.materialRegionId)) ordered.add(row.materialRegionId);
    selectedRegionIds = ordered;
    repaint(); return true;
}

void AssetManagerComponent::rebuildTreeRows()
{
    ensureSearchFiles();
    juce::StringArray roots;
    for (const auto& folder : folders) roots.add(folder.directory.getFullPathName());
    SampleSettings::setMaterialRoots(roots);
    treeRows.clear(); dragStartFile = juce::File{}; dragStartVariantId.clear();
    int top = 0;
    const auto append = [&](int folder, int member, int region = -1,
                            const juce::String& regionId = {}, const juce::String& audioId = {})
    {
        const auto height = member >= 0 ? audioRowHeight : rowHeight;
        treeRows.push_back({folder, member, top, height, region, regionId, audioId}); top += height;
    };
    for (int index = 0; index < static_cast<int>(folders.size()); ++index)
    {
        const auto& folder = folders[static_cast<std::size_t>(index)];
        if (hiddenFolders.count(materialPathKey(folder.directory))) continue;
        const bool searching = searchQuery.isNotEmpty();
        if (!searching)
        {
            append(index, -1);
            if (!folder.expanded) continue;
            if (folder.children.empty()) append(index, -2);
        }
        // One manifest decode/revision supplies every registration, even while
        // its pronunciation children are collapsed. Never inventory/decode PCM.
        const auto& annotations = ensureManifest(folder.directory);
        for (int member = 0; member < static_cast<int>(folder.children.size()); ++member)
        {
            const auto& file = folder.children[static_cast<std::size_t>(member)];
            const auto key = materialPathKey(file);
            if (hiddenAudio.count(key)) continue;
            if (std::any_of(expandedAudio.begin(), expandedAudio.end(),
                    [&](const auto& expanded) { return expanded.startsWith(key + "\n"); })) ensureOtoRows(file);
            const auto found = annotations.recordings.find(key);
            const std::vector<Registration> empty = {{ {}, {} }};
            const auto& registrations = found == annotations.recordings.end() ? empty : found->second;
            if (registrations.size() == 1 && expandedAudio.erase(key + "\n") != 0)
                expandedAudio.insert(key + "\n" + registrations.front().audioId);
            for (const auto& registration : registrations)
            {
                const auto& id = registration.audioId;
                const auto& rows = registration.rows;
                const bool expanded = expandedAudio.count(key + "\n" + id) != 0;
                const bool filenameMatch = !searching || file.getFileName().containsIgnoreCase(searchQuery);
                const auto aliasMatch = [&](const SampleRegionSetting& row)
                {
                    if (row.name.containsIgnoreCase(searchQuery)) return true;
                    for (const auto& alias : row.additionalAliases) if (alias.containsIgnoreCase(searchQuery)) return true;
                    return false;
                };
                const bool matchesAlias = searching && std::any_of(rows.begin(), rows.end(), aliasMatch);
                if (searching && !filenameMatch && !matchesAlias) continue;
                append(index, member, -1, {}, id);
                if (expanded || matchesAlias)
                    for (int region = 0; region < static_cast<int>(rows.size()); ++region)
                        if (!searching || filenameMatch || aliasMatch(rows[static_cast<std::size_t>(region)]))
                            append(index, member, region, rows[static_cast<std::size_t>(region)].materialRegionId, id);
                if (expanded && rows.empty()) append(index, member, -2, {}, id);
            }
        }
    }
    prunePronunciationSelection(); updateScrollRange(); repaint();
}

void AssetManagerComponent::ensureSearchFiles()
{
    if (searchQuery.isEmpty() || searchScanPending) return;
    std::vector<juce::File> pending;
    for (const auto& folder : folders)
        if (!folder.childrenLoaded) pending.push_back(folder.directory);
    if (pending.empty()) return;
    searchScanPending = true;
    searchPool.addJob([pending = std::move(pending), safe = juce::Component::SafePointer<AssetManagerComponent>(this)]
    {
        std::vector<std::pair<juce::File, std::vector<juce::File>>> results;
        for (const auto& directory : pending) results.emplace_back(directory, membersOf(directory));
        juce::MessageManager::callAsync([safe, results = std::move(results)]() mutable
        {
            if (safe == nullptr) return;
            for (auto& [directory, files] : results)
                for (auto& folder : safe->folders)
                    if (folder.directory == directory && !folder.childrenLoaded)
                    {
                        folder.children = std::move(files);
                        folder.childrenLoaded = true;
                        break;
                    }
            safe->searchScanPending = false;
            safe->rebuildTreeRows();
        });
    });
}

juce::String AssetManagerComponent::audioRowLabel(const TreeRow& row) const
{
    const auto& folder = folders[static_cast<std::size_t>(row.folder)];
    const auto& file = folder.children[static_cast<std::size_t>(row.member)];
    if (row.region == -2)
    {
        const auto cache = manifestCaches.find(materialPathKey(folder.directory));
        if (cache != manifestCaches.end())
        {
            if (cache->second.error.isNotEmpty()) return "OTTO: " + cache->second.error;
            const auto key = materialPathKey(file);
            if (cache->second.pendingAudio.count(key)) return strings.text("Reading OTO annotations...");
            const auto problem = cache->second.audioErrors.find(key);
            if (problem != cache->second.audioErrors.end()) return "OTO: " + problem->second;
        }
        return strings.text("No pronunciations; right-click to annotate");
    }
    if (row.region >= 0)
    {
        const auto cache = manifestCaches.find(materialPathKey(folder.directory));
        if (cache != manifestCaches.end())
        {
            const auto recording = cache->second.recordings.find(materialPathKey(file));
            if (recording != cache->second.recordings.end())
                for (const auto& registration : recording->second)
                    for (const auto& pronunciation : registration.rows)
                    if (registration.audioId == row.audioVariantId && pronunciation.materialRegionId == row.regionId)
                    {
                        auto label = pronunciation.name;
                        if (!pronunciation.additionalAliases.isEmpty()) label += ";" + pronunciation.additionalAliases.joinIntoString(";");
                        label += "  [" + juce::String(pronunciation.regionStartSeconds * 1000, 1) + ", "
                            + juce::String(pronunciation.regionEndSeconds * 1000, 1) + " ms]";
                        if (pronunciation.pronunciationKind == PronunciationKind::transition) label += "  [T]";
                        if (pronunciation.pronunciationKind == PronunciationKind::breath) label += "  [B]";
                        return label;
                    }
        }
        return row.regionId;
    }
    auto label = searchQuery.isEmpty() ? file.getRelativePathFrom(folder.directory)
        : file.getFileName() + juce::String::fromUTF8("（") + folder.name + juce::String::fromUTF8("）");
    const auto cache = manifestCaches.find(materialPathKey(folder.directory));
    if (cache != manifestCaches.end())
    {
        const auto registrations = cache->second.recordings.find(materialPathKey(file));
        if (registrations != cache->second.recordings.end() && registrations->second.size() > 1)
            label += " [" + row.audioVariantId + "]";
    }
    return label;
}

juce::StringArray AssetManagerComponent::diagnosticSearchLabels() const
{
    juce::StringArray labels;
    for (const auto& row : treeRows) if (row.member >= 0) labels.add(audioRowLabel(row));
    return labels;
}

void AssetManagerComponent::diagnosticToggleAudio(const juce::File& audio, const juce::String& audioVariantId)
{
    for (int f = 0; f < static_cast<int>(folders.size()); ++f)
    {
        auto& folder = folders[static_cast<std::size_t>(f)];
        if (!audio.isAChildOf(folder.directory)) continue;
        if (!folder.childrenLoaded) { folder.children = membersOf(folder.directory); folder.childrenLoaded = true; }
        const auto found = std::find(folder.children.begin(), folder.children.end(), audio);
        if (found == folder.children.end()) continue;
        folder.expanded = true;
        toggleAudio(f, static_cast<int>(found - folder.children.begin()), audioVariantId); return;
    }
}

bool AssetManagerComponent::diagnosticSelectPronunciation(const juce::File& audio, const juce::String& id,
                                                         bool control, bool shift, const juce::String& audioVariantId)
{
    for (int f = 0; f < static_cast<int>(folders.size()); ++f)
    {
        auto& folder = folders[static_cast<std::size_t>(f)];
        if (!audio.isAChildOf(folder.directory)) continue;
        if (!folder.childrenLoaded) { folder.children = membersOf(folder.directory); folder.childrenLoaded = true; }
        const auto member = std::find(folder.children.begin(), folder.children.end(), audio);
        if (member == folder.children.end()) continue;
        selectedFolder = f; members = folder.children; selectedMember = static_cast<int>(member - folder.children.begin());
        return selectPronunciation(audio, id, control, shift, false, audioVariantId);
    }
    return false;
}

bool AssetManagerComponent::diagnosticSelectRegistration(const juce::File& audio, const juce::String& requested)
{
    juce::String id;
    if (!resolveVariantId(audio, requested, id)) return false;
    for (int f = 0; f < static_cast<int>(folders.size()); ++f)
    {
        auto& folder = folders[static_cast<std::size_t>(f)];
        if (!audio.isAChildOf(folder.directory)) continue;
        if (!folder.childrenLoaded) { folder.children = membersOf(folder.directory); folder.childrenLoaded = true; }
        const auto member = std::find(folder.children.begin(), folder.children.end(), audio);
        if (member == folder.children.end()) continue;
        selectedFolder = f; members = folder.children; selectedMember = static_cast<int>(member - folder.children.begin());
        pronunciationSelectionAudio = audio; selectedAudioVariantId = id;
        selectedRegionIds.clear(); selectionAnchor.clear(); pronunciationSelectionActive = false; repaint(); return true;
    }
    return false;
}

juce::var AssetManagerComponent::diagnosticTreeSnapshot() const
{
    juce::Array<juce::var> result;
    for (const auto& row : treeRows)
    {
        auto* item = new juce::DynamicObject();
        item->setProperty("depth", row.member < 0 ? 0 : row.region == -1 ? 1 : 2);
        item->setProperty("audioVariantId", row.audioVariantId); item->setProperty("regionId", row.regionId);
        item->setProperty("label", row.member < 0 ? folders[static_cast<std::size_t>(row.folder)].name : audioRowLabel(row));
        result.add(juce::var(item));
    }
    return juce::var(result);
}

juce::StringArray AssetManagerComponent::diagnosticRegistrationIds(const juce::File& audio)
{
    const auto& cache = ensureManifest(rootForAudio(audio));
    juce::StringArray ids;
    const auto found = cache.recordings.find(materialPathKey(audio));
    if (found != cache.recordings.end()) for (const auto& registration : found->second) ids.add(registration.audioId);
    return ids;
}

juce::StringArray AssetManagerComponent::diagnosticPronunciationLabels(const juce::File& audio,
                                                                     const juce::String& audioVariantId)
{
    ensureManifest(rootForAudio(audio));
    juce::StringArray labels;
    const auto* rows = cachedRows(audio, audioVariantId);
    if (rows) for (const auto& row : *rows)
        labels.add(row.name + (row.additionalAliases.isEmpty() ? juce::String() : ";" + row.additionalAliases.joinIntoString(";"))
            + " [" + juce::String(row.regionStartSeconds * 1000, 1) + ", "
            + juce::String(row.regionEndSeconds * 1000, 1) + " ms]" + (row.pronunciationKind == PronunciationKind::transition ? " [T]" : row.pronunciationKind == PronunciationKind::breath ? " [B]" : ""));
    return labels;
}

void AssetManagerComponent::updateScrollRange()
{
    const auto height = std::max(0, getHeight() - toolbarHeight);
    const auto contentHeight = treeRows.empty() ? 0 : treeRows.back().top + treeRows.back().height;
    scrollOffset = juce::jlimit(0, std::max(0, contentHeight - height), scrollOffset);
    treeScroll.setRangeLimits(0.0, static_cast<double>(std::max(height, contentHeight)), juce::dontSendNotification);
    treeScroll.setCurrentRange(scrollOffset, height, juce::dontSendNotification);
    treeScroll.setSingleStepSize(audioRowHeight);
    treeScroll.setVisible(contentHeight > height);
    treeScroll.setBounds(getWidth() - 12, toolbarHeight, 12, height);
}

void AssetManagerComponent::scrollBarMoved(juce::ScrollBar*, double newRangeStart)
{
    scrollOffset = juce::roundToInt(newRangeStart);
    repaint();
}

void AssetManagerComponent::mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel)
{
    scrollOffset -= juce::roundToInt(wheel.deltaY * audioRowHeight * 5.0f);
    updateScrollRange();
    repaint();
}

int AssetManagerComponent::treeRowAt(int y) const
{
    if (y < toolbarHeight || y >= getHeight()) return -1;
    const auto contentY = y - toolbarHeight + scrollOffset;
    const auto found = std::lower_bound(treeRows.begin(), treeRows.end(), contentY,
        [](const auto& row, int position) { return row.top + row.height <= position; });
    return found == treeRows.end() ? -1 : static_cast<int>(found - treeRows.begin());
}

void AssetManagerComponent::toggleFolder(int index)
{
    auto& folder = folders[static_cast<std::size_t>(index)];
    folder.expanded = !folder.expanded;
    if (folder.expanded)
    {
        if (!folder.projectAudio || folder.registered || folder.projectVoicebank)
            folder.children = membersOf(folder.directory);
        folder.childrenLoaded = true;
        if (index == selectedFolder)
        {
            const auto previous = diagnosticSelectedAudio();
            members = folder.children;
            const auto found = std::find(members.begin(), members.end(), previous);
            selectedMember = found == members.end() ? -1 : static_cast<int>(found - members.begin());
        }
    }
    rebuildTreeRows();
}

juce::File AssetManagerComponent::diagnosticSelectedAudio() const
{
    return selectedMember >= 0 && selectedMember < static_cast<int>(members.size())
        ? members[static_cast<std::size_t>(selectedMember)] : juce::File{};
}

juce::PopupMenu::Options AssetManagerComponent::contextMenuOptions(juce::Point<int> screenPosition)
{
    // The Point/Point Rectangle constructor means opposite corners, not size.
    return juce::PopupMenu::Options().withTargetComponent(getTopLevelComponent())
        .withTargetScreenArea({screenPosition.x, screenPosition.y, 1, 1});
}

void AssetManagerComponent::mouseDown(const juce::MouseEvent& event)
{
    // Capture before selection can rebuild/reposition the scrolled tree. Use the
    // native top-level window for DPI, not the viewport's moving content bounds.
    const auto popupOptions = contextMenuOptions(event.getScreenPosition());
    dragStartFile = juce::File{};
    const auto index = treeRowAt(event.y);
    if (index < 0) return;
    const auto row = treeRows[static_cast<std::size_t>(index)];
    if (row.member == -1 && event.x < 30 && !event.mods.isPopupMenu())
    {
        // The second down of a double-click must not undo the first toggle.
        if (event.getNumberOfClicks() == 1) toggleFolder(row.folder);
        return;
    }
    if (row.member >= 0 && row.region == -1 && event.x >= 30 && event.x < 50 && !event.mods.isPopupMenu())
    {
        if (event.getNumberOfClicks() == 1) toggleAudio(row.folder, row.member, row.audioVariantId);
        return;
    }
    if (row.region == -2 && !event.mods.isPopupMenu()) return;
    const auto file = row.member >= 0
        ? folders[static_cast<std::size_t>(row.folder)].children[static_cast<std::size_t>(row.member)] : juce::File{};
    if (selectedFolder != row.folder)
    {
        selectedFolder = row.folder;
        rebuildMemberList(false);
    }
    const auto member = std::find(members.begin(), members.end(), file);
    selectedMember = member != members.end() ? static_cast<int>(member - members.begin()) : -1;
    if (row.region >= 0)
        selectPronunciation(file, row.regionId, event.mods.isCtrlDown() || event.mods.isCommandDown(),
                            event.mods.isShiftDown(), true, row.audioVariantId);
    else
    {
        pronunciationSelectionAudio = file; selectedAudioVariantId = row.audioVariantId; selectedRegionIds.clear(); selectionAnchor.clear(); pronunciationSelectionActive = false;
    }
    dragStartFile = file; dragStartVariantId = row.audioVariantId;
    setWantsKeyboardFocus(true);
    if (isShowing()) grabKeyboardFocus();
    repaint();
    if (event.mods.isPopupMenu())
    {
        dragStartFile = juce::File{};
        if (row.member == -1)
        {
            const auto folder = folders[static_cast<std::size_t>(row.folder)].directory;
            juce::PopupMenu menu;
            menu.addItem(1, strings.text("Migrate to native OTTO annotations..."));
            menu.addItem(2, strings.text("asset.exportOto"));
            // Community dataset UI is postponed; no public root action.
            menu.addItem(3, strings.text("Use for current track"), onUseMaterialFolder != nullptr);
            menu.addSeparator();
            menu.addItem(4, strings.text("asset.remove"));
            menu.showMenuAsync(popupOptions,
                [safe = juce::Component::SafePointer<AssetManagerComponent>(this), folder](int result)
                {
                    if (safe == nullptr) return;
                    if (result == communityDatasetImportAction) safe->requestCommunityDatasetImport(folder);
                    else if (result == 1) safe->migrateFolderDialog(folder);
                    else if (result == 2) safe->chooseOtoExport(folder);
                    else if (result == 3 && safe->onUseMaterialFolder) safe->onUseMaterialFolder(folder);
                    else if (result == 4) safe->removeFromBrowser(folder, true);
                });
            return;
        }
        if (!file.existsAsFile()) return;
        juce::PopupMenu menu;
        menu.addItem(1, strings.text("asset.loadAudio"), onLoadMaterialSelection != nullptr || onLoadPronunciations != nullptr || onLoadAudio != nullptr);
        menu.addItem(2, strings.text("Edit material annotations..."));
        menu.addSeparator();
        menu.addItem(3, strings.text("asset.remove"));
        menu.showMenuAsync(popupOptions,
            [safe = juce::Component::SafePointer<AssetManagerComponent>(this), file,
             ids = row.region >= 0 ? selectedRegionIds : juce::StringArray{}, regionId = row.regionId, audioId = row.audioVariantId](int result)
            {
                if (safe == nullptr) return;
                if (result == 1)
                {
                    safe->dispatchSelection(file, audioId, ids);
                }
                else if (result == 2) safe->openMemberEditor(file, regionId, audioId);
                else if (result == 3) safe->removeFromBrowser(file, false);
            });
    }
}

bool AssetManagerComponent::dispatchSelection(const juce::File& member, const juce::String& requested,
                                              const juce::StringArray& ids)
{
    if (!member.existsAsFile()) return false;
    juce::String audioId;
    if (!resolveVariantId(member, requested, audioId)) return false;
    const auto* rows = cachedRows(member, audioId);
    for (const auto& id : ids)
        if (!rows || std::none_of(rows->begin(), rows->end(),
            [&](const auto& row) { return row.materialRegionId == id; })) return false;
    if (onLoadMaterialSelection)
    { onLoadMaterialSelection({member, rootForAudio(member), audioId, ids}); return true; }
    const auto& cache = ensureManifest(rootForAudio(member));
    const auto found = cache.recordings.find(materialPathKey(member));
    if (found != cache.recordings.end() && found->second.size() > 1) return false;
    // Legacy host callbacks cannot identify siblings. Use ONLY unambiguous paths.
    if ((!ids.isEmpty() || found != cache.recordings.end()) && onLoadPronunciations)
    { onLoadPronunciations(member, ids); return true; }
    if (!ids.isEmpty() || !onLoadAudio) return false;
    onLoadAudio(member); return true;
}

bool AssetManagerComponent::loadAudio(const juce::File& member)
{
    if (pronunciationSelectionAudio == member && pronunciationSelectionActive && selectedRegionIds.isEmpty()) return false;
    return dispatchSelection(member, selectedAudioVariantId,
        pronunciationSelectionAudio == member && pronunciationSelectionActive ? selectedRegionIds : juce::StringArray{});
}

bool AssetManagerComponent::keyPressed(const juce::KeyPress& key)
{
    if (key == juce::KeyPress(juce::KeyPress::returnKey)
        && selectedMember >= 0 && selectedMember < static_cast<int>(members.size()))
        return loadAudio(members[static_cast<std::size_t>(selectedMember)]);
    return false;
}

void AssetManagerComponent::mouseDrag(const juce::MouseEvent& event)
{
    if (dragStartFile == juce::File{} || event.getDistanceFromDragStart() < 6) return;
    const auto member = dragStartFile;
    const auto audioId = dragStartVariantId;
    dragStartFile = juce::File{}; dragStartVariantId.clear();
    if (pronunciationSelectionActive && pronunciationSelectionAudio == member && selectedRegionIds.isEmpty()) return;
    if (auto* container = juce::DragAndDropContainer::findParentDragContainerFor(this))
    {
        // An empty drag image makes JUCE snapshot this entire manager. Only
        // the recording is being dragged, so give it a compact filename badge.
        constexpr int width = 220, height = 26;
        juce::Image preview(juce::Image::ARGB, width * 2, height * 2, true);
        juce::Graphics g(preview);
        g.addTransform(juce::AffineTransform::scale(2.0f));
        const auto bounds = juce::Rectangle<float>(0, 0, width, height).reduced(.5f);
        g.setColour(Palette::panelRaised.withAlpha(.95f));
        g.fillRoundedRectangle(bounds, 4.0f);
        g.setColour(Palette::accentLight);
        g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
        g.setColour(Palette::text);
        g.setFont(13.0f);
        g.drawText(member.getFileName(), 8, 0, width - 16, height,
                   juce::Justification::centredLeft, true);
        const juce::Point<int> offset(-12, -height / 2);
        const auto ids = pronunciationSelectionAudio == member ? selectedRegionIds : juce::StringArray{};
        const auto description = pronunciationDragDescription(member, ids, audioId);
        if (description.isVoid()) return;
        container->startDragging(description, this,
            juce::ScaledImage(preview, 2.0), false, &offset, &event.source);
        return;
    }
    const juce::StringArray files { dragFileFor(member).getFullPathName() };
    juce::DragAndDropContainer::performExternalDragDropOfFiles(files, false, this);
}

juce::File AssetManagerComponent::rootForAudio(const juce::File& file) const
{
    if (selectedFolder >= 0 && selectedFolder < static_cast<int>(folders.size())
        && file.isAChildOf(folders[static_cast<std::size_t>(selectedFolder)].directory))
        return folders[static_cast<std::size_t>(selectedFolder)].directory;
    juce::File root;
    for (const auto& folder : folders)
        if (file.isAChildOf(folder.directory)
            && (root == juce::File{} || folder.directory.isAChildOf(root))) root = folder.directory;
    return root == juce::File{} ? file.getParentDirectory() : root;
}

int AssetManagerComponent::otoImportKind(const juce::File& audioOrFolder) const
{
    auto root = audioOrFolder;
    if (root == juce::File{} && juce::isPositiveAndBelow(selectedFolder, static_cast<int>(folders.size())))
        root = folders[static_cast<std::size_t>(selectedFolder)].directory;
    else if (!root.isDirectory()) root = rootForAudio(root);
    if (root.getChildFile("otto.ini").exists()) return 1;
    for (auto folder = audioOrFolder.isDirectory() || audioOrFolder == juce::File{}
             ? root : audioOrFolder.getParentDirectory();; folder = folder.getParentDirectory())
    {
        if (folder.getChildFile("oto.ini").existsAsFile()) return 2;
        if (folder == root || folder.getParentDirectory() == folder) break;
    }
    return 1;
}

void AssetManagerComponent::setOtoImportKind(const juce::File& folder, int kind)
{
    juce::ignoreUnused(folder, kind); // Automatic authority cannot be overridden.
}

juce::StringArray AssetManagerComponent::materialRoots() const
{
    juce::StringArray roots;
    for (const auto& folder : folders) roots.addIfNotAlreadyThere(folder.directory.getFullPathName());
    return roots;
}

juce::var AssetManagerComponent::audioDragDescription(const juce::File& file, const juce::String& audioVariantId) const
{
    return pronunciationDragDescription(file, {}, audioVariantId); // Recording row: first pronunciation, resolved on import.
}

juce::var AssetManagerComponent::pronunciationDragDescription(const juce::File& file,
                                                            const juce::StringArray& ids, const juce::String& audioVariantId) const
{
    juce::String audioId;
    if (!resolveVariantId(file, audioVariantId, audioId)) return {};
    const auto* rows = cachedRows(file, audioId);
    for (const auto& id : ids)
        if (!rows || std::none_of(rows->begin(), rows->end(),
            [&](const auto& row) { return row.materialRegionId == id; })) return {};
    const auto root = rootForAudio(file);
    const auto& native = ensureManifest(root);
    const bool hasNative = native.exists || otoImportKind(file) == 2
        || native.recordings.count(materialPathKey(file)) != 0 || native.error.isNotEmpty();
    auto* request = new juce::DynamicObject();
    request->setProperty("type", "hachi-material-audio");
    request->setProperty("file", file.getFullPathName());
    request->setProperty("root", root.getFullPathName());
    request->setProperty("otoKind", otoImportKind(file));
    request->setProperty("nativeMaterial", hasNative);
    if (hasNative) request->setProperty("manifest", root.getChildFile("otto.ini").getFullPathName());
    juce::Array<juce::var> regions;
    for (const auto& id : ids) regions.add(id);
    request->setProperty("regionIds", juce::var(regions));
    request->setProperty("audioVariantId", audioId);
    return juce::var(request);
}

void AssetManagerComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    const auto index = treeRowAt(event.y);
    if (index < 0) return;
    const auto row = treeRows[static_cast<std::size_t>(index)];
    if (row.member == -1)
    {
        if (event.x >= 30) toggleFolder(row.folder);
    }
    else if (row.member >= 0 && row.region == -1)
    {
        if (event.x < 30 || event.x >= 50) toggleAudio(row.folder, row.member, row.audioVariantId);
    }
    else if (row.member >= 0 && row.region >= 0)
        openMemberEditor(folders[static_cast<std::size_t>(row.folder)].children[static_cast<std::size_t>(row.member)], row.regionId, row.audioVariantId);
}

void AssetManagerComponent::refreshMaterialAnnotations(const juce::File& source)
{
    manifestCaches.erase(materialPathKey(rootForAudio(source)));
    rebuildMemberList();
}

void AssetManagerComponent::openMemberEditor(const juce::File& member, const juce::String& regionId,
                                              const juce::String& audioVariantId,
                                              std::optional<MaterialAnnotationSession::NoteScopeBinding> noteScope,
                                              MaterialAnnotationSession::EditScope initialScope,
                                              std::optional<juce::Range<double>> sourceRange)
{
    if (!member.existsAsFile()) return;
    const auto requestedAudio = noteScope && audioVariantId.isEmpty() ? noteScope->audioId : audioVariantId;
    const auto requestedRegion = noteScope && regionId.isEmpty() ? noteScope->regionId : regionId;
    if (noteScope && (noteScope->noteId.isEmpty() || noteScope->sourceAudio != member
        || noteScope->audioId != requestedAudio || noteScope->regionId != requestedRegion))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            strings.text("asset.editTitle"), "The requested note context does not match this source/variant/pronunciation");
        return;
    }
    juce::String audioId;
    const auto& authority = ensureManifest(rootForAudio(member));
    const bool explicitClassic = !authority.exists && authority.error.isEmpty() && requestedAudio.isNotEmpty();
    if (explicitClassic) audioId = requestedAudio;
    if (!explicitClassic && !resolveVariantId(member, requestedAudio, audioId))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            strings.text("asset.editTitle"), "Missing or ambiguous annotation variant; select a registration/audioId");
        return;
    }
    editorWindows.removeClosed();
    const auto noteContextId = noteScope ? noteScope->noteId : juce::String{};
    const auto windowKey = "material-precise|" + member.getFullPathName() + "|" + audioId
        + (noteScope ? "|note|" + noteContextId : juce::String{});
    if (auto* existing = editorWindows.findVisible<OtoWaveformEditorComponent>(windowKey))
    {
        // An existing note context is never rebound or navigated implicitly:
        // preserve its dirty draft and captured stale-revision checks.
        if (!noteScope)
            if (auto* content = dynamic_cast<OtoWaveformEditorComponent*>(existing->getContentComponent()))
            {
                content->bindMaterialViewPreferences(properties);
                // A source-range handoff from a note opens the pronunciation overview.
                if (requestedRegion.isNotEmpty()) content->selectMaterialPronunciation(requestedRegion, !sourceRange.has_value());
                else if (sourceRange) content->focusMaterialSourceRange(*sourceRange, false);
            }
        existing->toFront(true);
        return;
    }
    const auto root = rootForAudio(member);
    const auto manifest = root.getChildFile("otto.ini");
    juce::String openError;
    auto editor = OtoWaveformEditorComponent::forMaterial(member, manifest, strings,
        [safe = juce::Component::SafePointer<AssetManagerComponent>(this), member](const auto& rows)
        {
            if (safe == nullptr) return;
            backend::UtauRenderer::invalidateVoicebankCache();
            if (safe->onMaterialSaved) safe->onMaterialSaved(member, rows);
            if (safe != nullptr) safe->refreshMaterialAnnotations(member);
        }, false, &openError, audioId);
    if (!editor)
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            strings.text("asset.editTitle"), openError);
        return;
    }
    if (requestedRegion.isNotEmpty()) editor->selectMaterialPronunciation(requestedRegion, noteScope.has_value() || !sourceRange.has_value());
    else if (!noteScope && sourceRange) editor->focusMaterialSourceRange(*sourceRange, false);
    if (noteScope && !editor->enableNoteScope(std::move(*noteScope), initialScope, openError))
    {
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon,
            strings.text("asset.editTitle"), openError);
        return;
    }
    if (onConfigureMaterialEditor) onConfigureMaterialEditor(*editor);
    if (!noteScope) editor->bindMaterialViewPreferences(properties);
    editor->setPublicationValidator(
        [safe = juce::Component::SafePointer<AssetManagerComponent>(this), member](const auto& rows, juce::String& error)
        {
            if (safe == nullptr) { error = "Material manager was closed"; return false; }
            return !safe->onMaterialPreflight || safe->onMaterialPreflight(member, rows, error);
        });
    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = strings.text("asset.editTitle") + " — " + member.getFileName()
        + (audioId.isEmpty() ? juce::String{} : " [" + audioId + "]")
        + (noteContextId.isEmpty() ? juce::String{} : " — " + noteContextId);
    options.dialogBackgroundColour = Palette::panel;
    options.escapeKeyTriggersCloseButton = true; options.useNativeTitleBar = true;
    options.resizable = true; options.content.setOwned(editor.release());
    if (auto* window = editorWindows.show(options, windowKey)) window->setResizeLimits(900, 700, 1800, 1200);
}

bool AssetManagerComponent::writeNativeTiming(const juce::File& member,
    const VoicebankOtoEntry& edited, juce::String& error)
{
    const auto duration = probeDurationSeconds(member);
    auto stored = SampleSettings::loadOrDerive(member, ProjectData{}, &error);
    if (error.isNotEmpty()) return false;
    if (stored.empty()) stored.push_back({});
    auto& target = stored.front();
    const auto oldStart = target.regionStartSeconds;
    const auto oldEnd = target.regionEndSeconds;
    target.name = edited.alias.isNotEmpty() ? edited.alias
                                            : member.getFileNameWithoutExtension();
    target.regionStartSeconds = edited.offsetMs / 1000.0;
    target.fixedDurationSeconds = edited.consonantMs / 1000.0;
    target.alignmentSeconds = target.regionStartSeconds + edited.preutteranceMs / 1000.0;
    target.overlapSeconds = edited.overlapMs / 1000.0;
    // Same cutoff convention as importOto: negative is a length from offset,
    // positive trims that amount from the physical file end.
    target.regionEndSeconds = edited.cutoffMs < 0.0
        ? target.regionStartSeconds + (-edited.cutoffMs) / 1000.0
        : duration > 0.0 ? duration - edited.cutoffMs / 1000.0
                         : target.regionStartSeconds + 0.5;
    target.regionEndSeconds = std::max(target.regionStartSeconds + 0.001,
                                       target.regionEndSeconds);
    // Retain internal segmentation while moving its absolute source clock.
    // The old implementation changed the row but left its segment coordinates
    // outside the new interval, which a strict native manifest must reject.
    const auto oldLength = oldEnd - oldStart;
    const auto newLength = target.regionEndSeconds - target.regionStartSeconds;
    if (oldLength > 1.0e-9)
        for (auto& segment : target.segments)
        {
            const auto map = [&](double time)
            {
                return target.regionStartSeconds + (time - oldStart) * newLength / oldLength;
            };
            segment.sourceStartSeconds = map(segment.sourceStartSeconds);
            segment.sourceEndSeconds = map(segment.sourceEndSeconds);
            segment.alignmentSeconds = map(segment.alignmentSeconds);
        }
    target.provenance = "native";
    return SampleSettings::save(member, stored, error);
}

bool AssetManagerComponent::diagnosticWriteNativeTiming(const juce::File& member,
    double offsetMs, double consonantMs, double cutoffMs,
    double preutterMs, double overlapMs, juce::String& error)
{
    VoicebankOtoEntry edited;
    edited.audioFile = member;
    edited.alias = member.getFileNameWithoutExtension();
    edited.offsetMs = offsetMs;
    edited.consonantMs = consonantMs;
    edited.cutoffMs = cutoffMs;
    edited.preutteranceMs = preutterMs;
    edited.overlapMs = overlapMs;
    return writeNativeTiming(member, edited, error);
}

void AssetManagerComponent::lookAndFeelChanged()
{
    searchEditor.setTextToShowWhenEmpty(strings.text("asset.search"), Palette::textMuted);
    searchEditor.setColour(juce::TextEditor::backgroundColourId, Palette::base);
    const auto text = Palette::contrastingText(Palette::base);
    searchEditor.setColour(juce::TextEditor::textColourId, text);
    searchEditor.applyColourToAllText(text, false);
    repaint();
}

void AssetManagerComponent::paint(juce::Graphics& g)
{
    g.fillAll(Palette::panel);
    g.setColour(Palette::border);
    g.drawHorizontalLine(toolbarHeight - 1, 0.0f, static_cast<float>(getWidth()));
    const auto pane = getLocalBounds().withTrimmedTop(toolbarHeight)
        .withTrimmedRight(treeScroll.isVisible() ? 12 : 0);
    g.reduceClipRegion(pane);
    if (folders.empty() || (searchQuery.isNotEmpty() && treeRows.empty()))
    {
        g.setFont(13.0f);
        g.setColour(Palette::textMuted);
        g.drawFittedText(strings.text(searchQuery.isNotEmpty()
            ? (searchScanPending ? "asset.searching" : "asset.noSearchResults") : "asset.empty"),
            pane.reduced(16), juce::Justification::centred, 4);
        return;
    }
    const auto selectedAudio = diagnosticSelectedAudio();
    const auto first = treeRowAt(toolbarHeight);
    if (first < 0) return;
    for (int index = first; index < static_cast<int>(treeRows.size()); ++index)
    {
        const auto& entry = treeRows[static_cast<std::size_t>(index)];
        if (entry.top >= scrollOffset + pane.getHeight()) break;
        const auto& folder = folders[static_cast<std::size_t>(entry.folder)];
        auto row = juce::Rectangle<int>(0, toolbarHeight + entry.top - scrollOffset,
                                        pane.getWidth(), entry.height);
        const auto header = entry.member == -1;
        const auto file = entry.member >= 0 ? folder.children[static_cast<std::size_t>(entry.member)] : juce::File{};
        const auto selected = header ? entry.folder == selectedFolder
            : entry.region >= 0 ? file == pronunciationSelectionAudio && entry.audioVariantId == selectedAudioVariantId && selectedRegionIds.contains(entry.regionId)
            : entry.region == -1 && file != juce::File{} && file == selectedAudio && entry.audioVariantId == selectedAudioVariantId
                && (pronunciationSelectionAudio != file || !pronunciationSelectionActive);
        const auto rowFill = selected ? Palette::accent.withAlpha(0.22f)
            : header ? Palette::panelRaised : Palette::panel;
        const auto rowBackground = Palette::panel.overlaidWith(rowFill);
        const auto light = Palette::base.getPerceivedBrightness() > 0.5f;
        const auto mutedInk = light ? Palette::contrastingText(rowBackground, true) : Palette::textMuted;
        const auto disclosureInk = light ? Palette::noteEdge : Palette::accentLight;
        g.setColour(rowFill);
        g.fillRect(row);
        if (header)
        {
            g.setColour(disclosureInk);
            juce::Path arrow;
            const auto y = static_cast<float>(row.getCentreY());
            if (folder.expanded) arrow.addTriangle(10.0f, y - 3.0f, 22.0f, y - 3.0f, 16.0f, y + 4.0f);
            else arrow.addTriangle(13.0f, y - 6.0f, 13.0f, y + 6.0f, 20.0f, y);
            g.fillPath(arrow);
            row.removeFromLeft(30);
            auto title = row.reduced(4, 2).removeFromTop(20);
            if (folder.projectVoicebank || folder.projectAudio)
            {
                g.setFont(10.0f);
                g.drawText(folder.projectVoicebank ? strings.text("asset.projectVoicebank")
                    : strings.getLanguage() == I18n::Language::zhCN ? juce::String::fromUTF8("工程素材") : "Project audio", title.removeFromRight(76),
                           juce::Justification::centredRight, true);
            }
            g.setFont(13.0f);
            g.setColour(Palette::text);
            g.drawText(folder.name, title, juce::Justification::centredLeft, true);
            g.setFont(10.0f);
            g.setColour(mutedInk);
            g.drawText(folder.directory.getFullPathName(), row.reduced(4, 2).removeFromBottom(16),
                       juce::Justification::centredLeft, true);
        }
        else if (entry.member >= 0)
        {
            if (entry.region == -1)
            {
                g.setColour(disclosureInk);
                juce::Path arrow;
                const auto y = static_cast<float>(row.getCentreY());
                if (expandedAudio.count(materialPathKey(file) + "\n" + entry.audioVariantId) != 0)
                    arrow.addTriangle(34.0f, y - 3.0f, 46.0f, y - 3.0f, 40.0f, y + 4.0f);
                else arrow.addTriangle(37.0f, y - 5.0f, 37.0f, y + 5.0f, 44.0f, y);
                g.fillPath(arrow);
            }
            row.removeFromLeft(entry.region == -1 ? 50 : 66);
            g.setFont(entry.region == -2 ? 11.0f : 13.0f);
            g.setColour(entry.region == -2 ? mutedInk : Palette::text);
            g.drawText(audioRowLabel(entry), row.reduced(4, 1),
                       juce::Justification::centredLeft, true);
        }
        else
        {
            row.removeFromLeft(34);
            g.setFont(11.0f);
            g.setColour(Palette::textMuted);
            g.drawFittedText(strings.text(folder.directory.isDirectory() ? "asset.folderEmpty" : "asset.folderMissing"),
                            row.reduced(4), juce::Justification::centredLeft, 2);
        }
    }
}

void AssetManagerComponent::resized()
{
    updateScrollRange();
    auto toolbar = getLocalBounds().removeFromTop(38).reduced(5, 5);
    auto search = juce::Rectangle<int>(5, 40, std::max(0, getWidth() - 10), 24);
    clearSearchButton.setBounds(search.removeFromRight(26));
    search.removeFromRight(4);
    searchEditor.setBounds(search);
    folderActionsButton.setBounds(toolbar);

}

// ---------------------------------------------------------------------------
// Test seams

juce::StringArray AssetManagerComponent::diagnosticMembersOf(int folderIndex) const
{
    juce::StringArray names;
    if (folderIndex < 0 || folderIndex >= static_cast<int>(folders.size())) return names;
    const auto& folder = folders[static_cast<std::size_t>(folderIndex)];
    for (const auto& member : folder.projectAudio && !folder.registered && !folder.projectVoicebank
            ? folder.children : membersOf(folder.directory))
        names.add(member.getFileName());
    return names;
}

int AssetManagerComponent::diagnosticRegisterFolder(const juce::File& folder)
{
    return registerFolder(folder);
}

int AssetManagerComponent::diagnosticAddAudioToFolder(int folderIndex,
                                                      const juce::StringArray& paths)
{
    if (folderIndex < 0 || folderIndex >= static_cast<int>(folders.size())) return 0;
    const auto folder = folders[static_cast<std::size_t>(folderIndex)].directory;
    auto added = 0;
    for (const auto& path : paths)
    {
        juce::String error;
        const auto landed = takeAudioIntoFolder(juce::File(path), folder, error);
        if (landed != juce::File{}) ++added;
    }
    rebuildMemberList();
    return added;
}
}
