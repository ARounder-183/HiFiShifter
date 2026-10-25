#include "CommunityDatasetImportDialog.h"
#include "Theme.h"
#include <juce_cryptography/juce_cryptography.h>
#include <atomic>
#include <cmath>
#include <limits>
#include <set>

namespace hachi
{
namespace
{
using IO = CommunityDatasetIO;
using juce::String;
using juce::File;
using juce::var;
struct PreparedImport
{
    IO::Plan plan;
    File mappingFile;
    String mappingFingerprint, summary, inputFingerprint;
    juce::uint64 generation = 0;
};
struct Operation
{
    std::atomic<bool> cancelled{false};
    juce::uint64 generation = 0;
    bool publication = false;
    CommunityDatasetImportDialog::Host host;
};
bool reject(String& error, const String& text) { error = text; return false; }
String inputFingerprint(const CommunityDatasetImportDialog::Input& input)
{
    // Guard widget values as well as change-message generation: TextEditor
    // notifications may still be queued when Publish is requested.
    juce::MemoryOutputStream bytes;
    bytes.writeString(input.root.getFullPathName()); bytes.writeString(input.dictionary.getFullPathName());
    bytes.writeString(input.mappingFile.getFullPathName()); bytes.writeString(input.mappingJson); bytes.writeString(input.provenanceNotes);
    bytes.writeBool(input.options.confirmNucleusAsAlignmentAndConsonant); bytes.writeInt(static_cast<int>(input.options.specials));
    bytes.writeBool(input.options.replacePreviouslyImportedAudio); bytes.writeBool(input.options.allowSiblingAnnotationVariant);
    return juce::SHA256(bytes.getData(), bytes.getDataSize()).toHexString();
}
bool objectKeys(const var& value, std::initializer_list<const char*> allowed, String& error)
{
    const auto* object = value.getDynamicObject();
    if (!object) return reject(error, "Import choices must be a JSON object");
    for (const auto& property : object->getProperties())
    {
        bool known = false;
        for (const auto* name : allowed) if (property.name.toString() == name) { known = true; break; }
        if (!known) return reject(error, "Unknown import-choice key: " + property.name.toString());
    }
    return true;
}
bool integer(const var& value, int minimum, int& result, String& error, const char* label)
{
    if (!value.isInt() && !value.isInt64() && !value.isDouble()) return reject(error, "Expected integer: " + String(label));
    const double number = static_cast<double>(value);
    if (!std::isfinite(number) || number < minimum || number > std::numeric_limits<int>::max() || number != std::floor(number))
        return reject(error, "Invalid integer range: " + String(label));
    result = static_cast<int>(number); return true;
}
bool namedString(const var& value, String& result, String& error, const char* label)
{
    if (!value.isString()) return reject(error, "Expected string: " + String(label));
    result = value.toString();
    if (result.trim().isEmpty() || result != result.trim()) return reject(error, "Empty or padded import choice: " + String(label));
    return true;
}
// This is the wizard's explicitly documented IMPORT-CHOICE SIDECAR, not a
// purported native DiffSinger CSV field, dictionary or annotation format.
bool parseChoices(const String& text, std::vector<IO::Selection>& selections,
                  IO::Options& options, String& error)
{
    if (text.trim().isEmpty()) return true;
    var root;
    const auto parsed = juce::JSON::parse(text, root);
    if (parsed.failed()) return reject(error, "Invalid import-choice JSON: " + parsed.getErrorMessage());
    if (!objectKeys(root, {"version", "selections", "primaryAliases"}, error)) return false;
    int version = 0;
    if (!integer(root.getProperty("version", {}), 1, version, error, "version") || version != 1)
        return reject(error, "Import-choice sidecar requires version 1");
    const auto entries = root.getProperty("selections", {});
    if (!entries.isVoid())
    {
        const auto* array = entries.getArray();
        if (!array) return reject(error, "selections must be an array");
        for (const auto& value : *array)
        {
            if (!objectKeys(value, {"recording", "alias", "firstPhone", "phoneCount", "nucleusOffset", "additionalAliases"}, error)) return false;
            IO::Selection selection;
            if (!namedString(value.getProperty("recording", {}), selection.recordingName, error, "recording")
                || !namedString(value.getProperty("alias", {}), selection.alias, error, "alias")
                || !integer(value.getProperty("firstPhone", {}), 0, selection.firstPhone, error, "firstPhone")
                || !integer(value.getProperty("phoneCount", {}), 1, selection.phoneCount, error, "phoneCount")
                || !integer(value.getProperty("nucleusOffset", {}), 0, selection.nucleusOffset, error, "nucleusOffset")) return false;
            if (selection.nucleusOffset >= selection.phoneCount) return reject(error, "nucleusOffset must lie inside the selected phone span");
            const auto aliases = value.getProperty("additionalAliases", {});
            if (!aliases.isVoid())
            {
                const auto* values = aliases.getArray();
                if (!values) return reject(error, "additionalAliases must be an array");
                std::set<String> seen{selection.alias};
                for (const auto& alias : *values)
                {
                    String name;
                    if (!namedString(alias, name, error, "additional alias") || !seen.insert(name).second)
                        return reject(error, "Additional aliases must be unique nonempty strings");
                    selection.additionalAliases.add(name);
                }
            }
            selections.push_back(std::move(selection));
        }
    }
    const auto aliases = root.getProperty("primaryAliases", {});
    if (!aliases.isVoid())
    {
        const auto* object = aliases.getDynamicObject();
        if (!object) return reject(error, "primaryAliases must be an object mapping exact phone sequences to chosen aliases");
        for (const auto& property : object->getProperties())
        {
            String chosen;
            const auto sequence = property.name.toString();
            if (sequence.trim().isEmpty() || sequence != sequence.trim()
                || !namedString(property.value, chosen, error, "primary alias"))
                return reject(error, "Invalid primaryAliases choice");
            options.primaryAliases[sequence] = chosen;
        }
    }
    return true;
}
bool loadChoices(const CommunityDatasetImportDialog::Input& input, PreparedImport& prepared,
                 std::vector<IO::Selection>& selections, IO::Options& options, String& error)
{
    String text = input.mappingJson;
    if (input.mappingFile != File{})
    {
        if (text.trim().isNotEmpty()) return reject(error, "Choose either a mapping file or inline choices, not both");
        if (!input.mappingFile.existsAsFile() || input.mappingFile.getSize() > 64 * 1024 * 1024)
            return reject(error, "Missing or oversized import-choice JSON file");
        juce::MemoryBlock bytes;
        if (!input.mappingFile.loadFileAsData(bytes)) return reject(error, "Cannot read import-choice JSON file");
        const auto* data = static_cast<const char*>(bytes.getData()); const auto length = static_cast<int>(bytes.getSize());
        for (int i = 0; i < length; ++i) if (data[i] == 0) return reject(error, "Import-choice file must be UTF-8 without NUL");
        if (length && !juce::CharPointer_UTF8::isValidString(data, length)) return reject(error, "Import-choice file is not valid UTF-8");
        const int bom = length >= 3 && static_cast<unsigned char>(data[0]) == 0xef
            && static_cast<unsigned char>(data[1]) == 0xbb && static_cast<unsigned char>(data[2]) == 0xbf ? 3 : 0;
        text = length > bom ? String::fromUTF8(data + bom, length - bom) : String();
        prepared.mappingFile = input.mappingFile; prepared.mappingFingerprint = juce::SHA256(bytes).toHexString();
    }
    return parseChoices(text, selections, options, error);
}
void appendBounded(String& output, const String& value)
{
    constexpr int limit = 512 * 1024;
    if (output.length() < limit)
    {
        output += value;
        if (output.length() >= limit) output += "\n[Display truncated; full parsed data is still validated, not silently skipped.]\n";
    }
}
String messages(const IO::Report& report)
{
    String result;
    for (const auto& value : report.errors) appendBounded(result, "ERROR: " + value + "\n");
    for (const auto& value : report.warnings) appendBounded(result, "WARNING: " + value + "\n");
    if (report.cancelled && !report.published) result += "Cancelled before publication; no OTTO commit was performed.\n";
    return result;
}
String dictionaryChoices(const IO::Dictionary& dictionary)
{
    std::map<String, juce::StringArray> choices;
    for (const auto& rule : dictionary.rules) choices[rule.phones.joinIntoString(" ")].add(rule.alias);
    String result;
    for (const auto& [sequence, aliases] : choices)
        if (aliases.size() > 1) appendBounded(result, "Alias choice (NO default): " + sequence + " => " + aliases.joinIntoString(" | ") + "\n");
    return result;
}
String datasetFacts(const IO::Dataset& dataset)
{
    String result = "Dataset root: " + dataset.root.getFullPathName() + "\nCSV SHA256: " + dataset.csvFingerprint + "\n";
    std::size_t shown = 0;
    for (const auto& recording : dataset.recordings)
    {
        if (shown++ == 100) { result += "[Additional recordings omitted from display only.]\n"; break; }
        appendBounded(result, "Recording: " + recording.name + "\n  " + recording.audio.getFullPathName()
            + "\n  SHA256: " + recording.audioFingerprint + "\n  phone indices (zero-based): ");
        for (std::size_t i = 0; i < recording.phones.size() && i < 250; ++i)
        {
            const auto& phone = recording.phones[i];
            appendBounded(result, String(static_cast<int>(i)) + ":" + phone.symbol + "["
                + String(phone.start, 6) + "," + String(phone.end, 6) + "] ");
        }
        if (recording.phones.size() > 250) appendBounded(result, "[Remaining phone facts omitted from display only.]");
        appendBounded(result, "\n");
    }
    return result;
}
}

struct CommunityDatasetImportDialog::Impl
{
    CommunityDatasetImportDialog& owner;
    I18n& strings;
    Host host;
    HachiLookAndFeel look;
    juce::Label title, explanation, rootLabel, dictionaryLabel, mappingLabel, provenanceLabel, choicesLabel;
    juce::TextEditor root, dictionary, mappingFile, provenance, choices, report;
    juce::TextButton rootBrowse, dictionaryBrowse, mappingBrowse, previewButton, publishButton, cancelButton;
    juce::ToggleButton timing, replace, sibling;
    juce::ComboBox specials;
    std::unique_ptr<juce::FileChooser> chooser;
    std::shared_ptr<Operation> operation;
    std::shared_ptr<PreparedImport> prepared;
    juce::uint64 generation = 0;
    bool loading = false, working = false, committing = false;

    String tr(const char* zh, const char* en) const
    { return strings.getLanguage() == I18n::Language::zhCN ? String::fromUTF8(zh) : String(en); }
    bool hostReady() const { return host.runOnWorker && host.publishing && host.preflight && host.published; }
    Impl(CommunityDatasetImportDialog& o, I18n& s, Host h, const File& initialRoot) : owner(o), strings(s), host(std::move(h))
    {
        owner.setLookAndFeel(&look);
        title.setText(tr("导入社区 DiffSinger 数据集标注", "Import community DiffSinger dataset annotations"), juce::dontSendNotification);
        explanation.setText(tr("只读取数据集已有时间标注，不运行语音识别。预览不写文件；导入将原子写入根目录 otto.ini。",
            "Uses existing dataset timing; no speech recognition. Preview writes nothing; Import atomically publishes root otto.ini."), juce::dontSendNotification);
        explanation.setJustificationType(juce::Justification::centredLeft);
        rootLabel.setText(tr("数据集根目录", "Dataset root"), juce::dontSendNotification);
        dictionaryLabel.setText(tr("V / C-V 字典（可选）", "V / C-V dictionary (optional)"), juce::dontSendNotification);
        mappingLabel.setText(tr("导入选择 JSON（可选）", "Import-choice JSON (optional)"), juce::dontSendNotification);
        provenanceLabel.setText(tr("来源／许可说明", "Provenance / license notes"), juce::dontSendNotification);
        choicesLabel.setText(tr("显式选择（非 CSV 字段；无文件时可在此输入）", "Explicit import choices (NOT CSV fields; inline input when no file is selected)"), juce::dontSendNotification);
        choices.setMultiLine(true); choices.setReturnKeyStartsNewLine(true); choices.setScrollbarsShown(true);
        choices.setTextToShowWhenEmpty("{\"version\":1,\"selections\":[{\"recording\":\"...\",\"alias\":\"...\",\"firstPhone\":0,\"phoneCount\":2,\"nucleusOffset\":1}],\"primaryAliases\":{\"k a\":\"ka\"}}", Palette::textMuted);
        report.setMultiLine(true); report.setReturnKeyStartsNewLine(true); report.setReadOnly(true); report.setScrollbarsShown(true);
        timing.setButtonText(tr("我确认：将选定元音起点同时作为 P 与 C（CSV 本身没有独立 P/C 标点）", "I confirm: selected nucleus onset becomes BOTH P and C (CSV has no independent P/C markers)"));
        replace.setButtonText(tr("显式替换本导入器先前生成的登记（不替换其他作者标注）", "Explicitly replace this importer's prior registration, never other authored variants"));
        sibling.setButtonText(tr("允许同一录音新增独立标注登记（不复制音频、不覆盖原登记）", "Allow a separate same-recording annotation registration (no PCM copy or overwrite)"));
        specials.addItem(tr("请选择 SP/AP 处理方式…", "Choose SP/AP policy..."), 1);
        specials.addItem(tr("跳过 SP/AP，并明确报告", "Skip SP/AP with explicit report"), 2);
        specials.addItem(tr("按数据集声明导入 SP/AP", "Import declared SP/AP"), 3);
        specials.setSelectedId(1, juce::dontSendNotification);
        rootBrowse.setButtonText(tr("选择…", "Browse...")); dictionaryBrowse.setButtonText(rootBrowse.getButtonText()); mappingBrowse.setButtonText(rootBrowse.getButtonText());
        previewButton.setButtonText(tr("只读预览", "Read-only preview")); publishButton.setButtonText(tr("确认导入到 OTTO…", "Import into OTTO...")); cancelButton.setButtonText(tr("取消任务", "Cancel work"));
        for (auto* c : std::initializer_list<juce::Component*>{&title,&explanation,&rootLabel,&dictionaryLabel,&mappingLabel,&provenanceLabel,&choicesLabel,
                &root,&dictionary,&mappingFile,&provenance,&choices,&report,&rootBrowse,&dictionaryBrowse,&mappingBrowse,&previewButton,&publishButton,&cancelButton,&timing,&replace,&sibling,&specials}) owner.addAndMakeVisible(c);
        for (auto* field : {&root,&dictionary,&mappingFile,&provenance,&choices}) field->onTextChange = [this] { changed(); };
        timing.onClick = [this] { changed(); }; replace.onClick = [this] { changed(); }; sibling.onClick = [this] { changed(); };
        specials.onChange = [this] { changed(); };
        rootBrowse.onClick = [this] { browse(0); }; dictionaryBrowse.onClick = [this] { browse(1); }; mappingBrowse.onClick = [this] { browse(2); };
        previewButton.onClick = [this] { preview(); }; publishButton.onClick = [this] { confirmPublish(); }; cancelButton.onClick = [this] { cancel(); };
        root.setText(initialRoot == File{} ? String() : initialRoot.getFullPathName(), false);
        report.setText(hostReady() ? tr("选择根目录、字典或显式分组，确认导入语义后预览。不会默认猜测别名或音节。", "Choose a root and dictionary or explicit spans, confirm import semantics, then preview. No alias or syllable is guessed.")
                                  : "Host integration missing: worker, publication gate, preflight and completion callbacks are required.", false);
        buttons();
    }
    ~Impl()
    {
        if (operation) operation->cancelled = true;
        chooser.reset();
        owner.setLookAndFeel(nullptr);
        // A commit may already have happened. Its shared completion owns Host,
        // releases the publication gate and reports success even without widgets.
    }
    void buttons()
    {
        previewButton.setEnabled(hostReady() && !working);
        publishButton.setEnabled(hostReady() && !working && prepared && prepared->plan.ready && prepared->generation == generation);
        cancelButton.setEnabled(working);
        for (auto* c : std::initializer_list<juce::Component*>{&root,&dictionary,&mappingFile,&provenance,&choices,&rootBrowse,&dictionaryBrowse,&mappingBrowse,&timing,&replace,&sibling,&specials}) c->setEnabled(!committing);
        choices.setEnabled(!committing && mappingFile.getText().trim().isEmpty());
    }
    void changed()
    {
        if (loading || committing) return;
        ++generation; prepared.reset();
        if (operation) operation->cancelled = true;
        report.setText(tr("输入已改变；旧预览失效，请重新预览。", "Inputs changed; the previous preview is invalid. Preview again."), false);
        buttons();
    }
    Input input() const
    {
        Input value;
        const auto file = [](const String& text) { return text.trim().isEmpty() ? File{} : File(text.trim()); };
        value.root = file(root.getText()); value.dictionary = file(dictionary.getText()); value.mappingFile = file(mappingFile.getText());
        value.mappingJson = choices.getText(); value.provenanceNotes = provenance.getText();
        value.options.confirmNucleusAsAlignmentAndConsonant = timing.getToggleState();
        value.options.specials = specials.getSelectedId() == 2 ? IO::Specials::skipWithReport
            : specials.getSelectedId() == 3 ? IO::Specials::importDeclaredSPAP : IO::Specials::reject;
        value.options.replacePreviouslyImportedAudio = replace.getToggleState(); value.options.allowSiblingAnnotationVariant = sibling.getToggleState();
        return value;
    }
    void browse(int kind)
    {
        auto* field = kind == 0 ? &root : kind == 1 ? &dictionary : &mappingFile;
        chooser = std::make_unique<juce::FileChooser>(kind == 0 ? rootLabel.getText() : kind == 1 ? dictionaryLabel.getText() : mappingLabel.getText(),
            field->getText().isEmpty() ? File{} : File(field->getText()), kind == 2 ? "*.json" : String());
        juce::Component::SafePointer<CommunityDatasetImportDialog> safe(&owner);
        chooser->launchAsync(juce::FileBrowserComponent::openMode | (kind == 0 ? juce::FileBrowserComponent::canSelectDirectories : juce::FileBrowserComponent::canSelectFiles),
            [safe,kind](const juce::FileChooser& chosen)
            {
                if (!safe || chosen.getResult() == File{}) return;
                auto& ui = *safe->impl;
                if (kind == 2) ui.choices.setText({}, false);
                (kind == 0 ? ui.root : kind == 1 ? ui.dictionary : ui.mappingFile).setText(chosen.getResult().getFullPathName(), true);
            });
    }
    void cancel()
    {
        if (!operation) return;
        operation->cancelled = true;
        report.setText(committing ? tr("正在请求取消；若原子提交已开始，将如实报告提交结果。", "Cancellation requested; an atomic commit already underway will still report its actual result.")
                                  : tr("正在取消只读预览…", "Cancelling read-only preview..."), false);
    }
    void preview()
    {
        if (working) return;
        prepared.reset();
        if (!hostReady()) { report.setText("Required Host integration is missing.", false); buttons(); return; }
        const auto selected = input();
        if (!selected.options.confirmNucleusAsAlignmentAndConsonant || specials.getSelectedId() == 1)
        { report.setText(tr("必须显式确认 P=C 解释并选择 SP/AP 策略；未写入任何文件。", "Explicit P=C confirmation and SP/AP policy are required; no file was written."), false); buttons(); return; }
        working = true; committing = false;
        auto op = std::make_shared<Operation>(); op->host = host; op->generation = generation; operation = op;
        report.setText(tr("工作线程正在读取／验证数据集；此步骤不会写入 OTTO。", "Worker is reading and validating the dataset; this step does not write OTTO."), false); buttons();
        juce::Component::SafePointer<CommunityDatasetImportDialog> safe(&owner);
        op->host.runOnWorker([op, selected, safe]
        {
            auto candidate = std::make_shared<PreparedImport>(); candidate->generation = op->generation;
            candidate->inputFingerprint = inputFingerprint(selected);
            IO::Report outcome; IO::Dataset dataset; IO::Dictionary dictionary;
            std::vector<IO::Selection> selections; auto options = selected.options; String error, details, priorWarnings;
            const auto cancel = [op] { return op->cancelled.load(); };
            const auto* manager = juce::MessageManager::getInstanceWithoutCreating();
            bool valid = !manager || !manager->isThisTheMessageThread();
            if (!valid) outcome.errors.add("Host executor ran import work on the message thread");
            if (valid)
            {
                valid = IO::read(selected.root, dataset, outcome, cancel);
                priorWarnings = messages(outcome);
                if (valid) details = datasetFacts(dataset);
            }
            if (valid && selected.dictionary != File{})
            {
                valid = IO::readTwoPartDictionary(selected.dictionary, dictionary, outcome, cancel);
                priorWarnings += messages(outcome); details += dictionaryChoices(dictionary);
            }
            if (valid && !loadChoices(selected, *candidate, selections, options, error))
            { valid = false; outcome.errors.add(error); }
            if (valid)
            {
                valid = IO::planImport(dataset, dictionary, selections, options, candidate->plan, outcome, cancel);
                if (valid)
                {
                    appendBounded(details, "\nREADY (not published): " + String(outcome.recordings) + " recordings, " + String(outcome.pronunciations) + " pronunciations\n");
                    for (const auto& audio : candidate->plan.audio)
                    {
                        appendBounded(details, audio.audio.getFullPathName() + "\n  registration=" + audio.audioId + "\n");
                        for (const auto& row : audio.rows) appendBounded(details, "  " + row.materialRegionId + " " + row.name + " [" + String(row.regionStartSeconds,6) + "," + String(row.regionEndSeconds,6) + "]\n");
                    }
                }
            }
            candidate->summary = "Provenance notes (supplied by caller): " + selected.provenanceNotes + "\n"
                + "Deterministic dataset-timing import; no ASR/pitch analysis. No human-recording or legacy-oracle claim is inferred.\n"
                + priorWarnings + details + "\n" + messages(outcome);
            juce::MessageManager::callAsync([op,candidate,safe,valid]
            {
                if (!safe || safe->impl->operation != op) return;
                auto& ui = *safe->impl; ui.working = false; ui.operation.reset();
                if (op->cancelled || ui.generation != op->generation)
                    ui.report.setText("Preview cancelled or input changed; no publication occurred. Preview again.", false);
                else
                {
                    ui.report.setText(candidate->summary, false);
                    if (valid) ui.prepared = candidate;
                }
                ui.buttons();
            });
        });
    }
    bool publishConfirmed(String& error)
    {
        error.clear();
        if (working || !prepared || !prepared->plan.ready || prepared->generation != generation
            || prepared->inputFingerprint != inputFingerprint(input()))
            return reject(error, "A current successful read-only preview is required before Import");
        if (!hostReady()) return reject(error, "Required Host integration is missing");
        const auto candidate = prepared;
        auto op = std::make_shared<Operation>(); op->host = host; op->generation = generation; op->publication = true;
        // Host owns a real revision/edit gate. A bool preflight alone is NOT
        // presented as a race-free model/source publication transaction.
        const auto callbacks = host;
        juce::Component::SafePointer<CommunityDatasetImportDialog> alive(&owner);
        callbacks.publishing(true);
        if (!alive) { callbacks.publishing(false); return reject(error, "Dialog closed before publication preflight"); }
        if (!callbacks.preflight(candidate->plan, error)) { callbacks.publishing(false); return false; }
        if (!alive) { callbacks.publishing(false); return reject(error, "Dialog closed during publication preflight"); }
        if (working || prepared != candidate || generation != op->generation || candidate->inputFingerprint != inputFingerprint(input()))
        { callbacks.publishing(false); return reject(error, "Inputs changed during preflight; preview again"); }
        working = committing = true; operation = op;
        report.setText("Publishing guarded root OTTO transaction on the worker...", false); buttons();
        juce::Component::SafePointer<CommunityDatasetImportDialog> safe(&owner);
        op->host.runOnWorker([candidate,op,safe]
        {
            IO::Report result;
            const auto* manager = juce::MessageManager::getInstanceWithoutCreating();
            if (manager && manager->isThisTheMessageThread()) result.errors.add("Host executor ran publication on the message thread");
            else if (candidate->mappingFile != File{} && (!candidate->mappingFile.existsAsFile()
                || juce::SHA256(candidate->mappingFile).toHexString() != candidate->mappingFingerprint))
                result.errors.add("Import-choice JSON file changed; preview it again before publication");
            else IO::publish(candidate->plan, result, [op] { return op->cancelled.load(); });
            juce::MessageManager::callAsync([candidate,op,safe,result]
            {
                // These callbacks intentionally survive widget deletion. Host
                // guards its own lifetime and owns executor shutdown/draining.
                if (result.published) op->host.published(candidate->plan, result);
                op->host.publishing(false);
                if (!safe || safe->impl->operation != op) return;
                auto& ui = *safe->impl; ui.working = ui.committing = false; ui.operation.reset(); ui.prepared.reset();
                ui.report.setText(candidate->summary + "\n" + (result.published
                    ? String("PUBLISHED: root OTTO committed; Host completion notified. Source PCM/CSV/dictionary not rewritten.\n")
                    : String("NOT PUBLISHED.\n")) + messages(result), false);
                ui.buttons();
            });
        });
        return true;
    }
    void confirmPublish()
    {
        if (!prepared || working) return;
        const auto expected = prepared; const auto epoch = generation;
        juce::Component::SafePointer<CommunityDatasetImportDialog> safe(&owner);
        juce::AlertWindow::showOkCancelBox(juce::MessageBoxIconType::WarningIcon,
            tr("确认数据集导入", "Confirm dataset import"), prepared->plan.manifest.getFullPathName() + "\n\n"
            + tr("将只按预览中的独立登记写入根 OTTO。原音频与 CSV 不变。OTTO 将管辖整个根目录。是否继续？",
                 "Write only the previewed annotation registrations into root OTTO? Audio and CSV remain unchanged. OTTO becomes authoritative for the whole root."),
            tr("导入", "Import"), tr("取消", "Cancel"), &owner,
            juce::ModalCallbackFunction::create([safe,expected,epoch](int accepted)
            {
                if (!safe || !accepted) return;
                auto& ui = *safe->impl; String error;
                if (ui.prepared != expected || ui.generation != epoch) error = "Inputs changed while confirmation was open; preview again";
                else if (ui.publishConfirmed(error)) return;
                if (safe) safe->impl->report.setText(error, false);
            }));
    }
};

CommunityDatasetImportDialog::CommunityDatasetImportDialog(I18n& strings, Host host, const File& root)
    : impl(std::make_unique<Impl>(*this, strings, std::move(host), root))
{ setSize(1040, 880); }
CommunityDatasetImportDialog::~CommunityDatasetImportDialog() = default;
void CommunityDatasetImportDialog::paint(juce::Graphics& graphics) { graphics.fillAll(Palette::panel); }
void CommunityDatasetImportDialog::resized()
{
    if (!impl) return;
    auto& ui = *impl; auto area = getLocalBounds().reduced(14);
    ui.title.setBounds(area.removeFromTop(28)); ui.explanation.setBounds(area.removeFromTop(42)); area.removeFromTop(6);
    const auto pathRow = [&](juce::Label& label, juce::TextEditor& editor, juce::TextButton* browse)
    {
        auto line = area.removeFromTop(32); label.setBounds(line.removeFromLeft(210));
        if (browse) browse->setBounds(line.removeFromRight(94).reduced(3,2));
        editor.setBounds(line.reduced(2,2));
    };
    pathRow(ui.rootLabel,ui.root,&ui.rootBrowse); pathRow(ui.dictionaryLabel,ui.dictionary,&ui.dictionaryBrowse);
    pathRow(ui.mappingLabel,ui.mappingFile,&ui.mappingBrowse); pathRow(ui.provenanceLabel,ui.provenance,nullptr);
    ui.choicesLabel.setBounds(area.removeFromTop(24)); ui.choices.setBounds(area.removeFromTop(140)); area.removeFromTop(4);
    ui.timing.setBounds(area.removeFromTop(30)); ui.specials.setBounds(area.removeFromTop(30).removeFromLeft(530).reduced(0,2));
    ui.replace.setBounds(area.removeFromTop(28)); ui.sibling.setBounds(area.removeFromTop(28)); area.removeFromTop(6);
    auto commands = area.removeFromBottom(34); ui.previewButton.setBounds(commands.removeFromLeft(210).reduced(2));
    ui.publishButton.setBounds(commands.removeFromLeft(260).reduced(2)); ui.cancelButton.setBounds(commands.removeFromRight(160).reduced(2));
    area.removeFromBottom(6); ui.report.setBounds(area);
}
void CommunityDatasetImportDialog::cancel() { impl->cancel(); }
bool CommunityDatasetImportDialog::busy() const { return impl->working; }
bool CommunityDatasetImportDialog::publicationInFlight() const { return impl->committing; }
bool CommunityDatasetImportDialog::previewReady() const
{ return !impl->working && impl->prepared && impl->prepared->generation == impl->generation
    && impl->prepared->inputFingerprint == inputFingerprint(impl->input()); }
const IO::Plan* CommunityDatasetImportDialog::previewPlan() const { return previewReady() ? &impl->prepared->plan : nullptr; }
String CommunityDatasetImportDialog::reportText() const { return impl->report.getText(); }
void CommunityDatasetImportDialog::preview() { impl->preview(); }
bool CommunityDatasetImportDialog::publishConfirmed(String& error) { return impl->publishConfirmed(error); }
bool CommunityDatasetImportDialog::setInput(const Input& value, String& error)
{
    error.clear();
    if (impl->committing) return reject(error, "Inputs are gated during publication");
    if (!value.options.primaryAliases.empty())
        return reject(error, "Supply primaryAliases in the visible import-choice JSON, not a hidden Options map");
    auto& ui = *impl; ui.loading = true;
    const auto text = [](const File& file) { return file == File{} ? String() : file.getFullPathName(); };
    ui.root.setText(text(value.root), false); ui.dictionary.setText(text(value.dictionary), false);
    ui.mappingFile.setText(text(value.mappingFile), false); ui.choices.setText(value.mappingJson, false); ui.provenance.setText(value.provenanceNotes, false);
    ui.timing.setToggleState(value.options.confirmNucleusAsAlignmentAndConsonant, juce::dontSendNotification);
    ui.specials.setSelectedId(value.options.specials == IO::Specials::skipWithReport ? 2 : value.options.specials == IO::Specials::importDeclaredSPAP ? 3 : 1, juce::dontSendNotification);
    ui.replace.setToggleState(value.options.replacePreviouslyImportedAudio, juce::dontSendNotification);
    ui.sibling.setToggleState(value.options.allowSiblingAnnotationVariant, juce::dontSendNotification);
    ui.loading = false; ui.changed();
    return true;
}
}
