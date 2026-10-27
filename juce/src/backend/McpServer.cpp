#include "McpServer.h"
#include "LegacyTextCodec.h"
#include "MelodyneProvider.h"
#include "AudioFileReader.h"
#include "AnalysisService.h"
#include "DiffSingerRenderer.h"
#include <cmath>
#include <iostream>

namespace hachi::backend
{
namespace
{
juce::var object()
{
    return juce::var(new juce::DynamicObject());
}

void set(juce::var& value, const char* key, juce::var property)
{
    if (auto* dynamic = value.getDynamicObject()) dynamic->setProperty(key, std::move(property));
}

juce::var array(std::vector<juce::var> values)
{
    juce::Array<juce::var> result;
    result.ensureStorageAllocated(static_cast<int>(values.size()));
    for (auto& value : values) result.add(std::move(value));
    return juce::var(std::move(result));
}

// What a tool takes.  A client shows the model it drives only what the schema
// says, so a tool whose schema names no properties is one whose parameter
// names have to be guessed from a one-line description -- which is what all of
// these were: forty-eight tools sharing one empty object.  These sit beside the
// dispatcher that reads the arguments, and a parameter added there and not
// named here is invisible to everything that reads the schema.
struct Param
{
    const char* name;
    const char* type;                       // JSON Schema type
    const char* description;
    bool required = false;
    std::vector<const char*> choices {};    // the few words a value may be
    juce::var (*shape)() = nullptr;         // an array's items, or an object's own fields
};

juce::var typedSchema(std::vector<Param> params);

// A point of the pitch curve set_pitch_curve draws.
juce::var pitchPointShape()
{
    return typedSchema({
        { .name = "time_seconds", .type = "number",
          .description = "When, in seconds from the clip start", .required = true },
        { .name = "midi", .type = "number",
          .description = "The pitch sung there, as a MIDI note with fractions",
          .required = true },
    });
}

// One flag's curve across a note.
juce::var flagCurveShape()
{
    return typedSchema({
        { .name = "flag", .type = "string",
          .description = "Which flag the curve belongs to: g, Mt, Mb, Mo, Md, Ms, "
                         "bh and the rest of the timbre flags.  Default g." },
        { .name = "points", .type = "array",
          .description = "[[seconds, value], ...] measured from the note's start.  "
                         "Seconds may be negative, which reaches into the preutterance.  "
                         "A third entry names the shape of the segment arriving there: "
                         "linear, smooth, ease-in, ease-out or custom-bezier.  "
                         "An empty list drops this flag's curve and leaves the others." },
    });
}

// One region of a recording, as the otto.ini sidecar keeps them.
juce::var regionRowShape()
{
    return typedSchema({
        { .name = "name", .type = "string", .description = "What the region is called" },
        { .name = "region_start_seconds", .type = "number",
          .description = "Where the region starts in the recording", .required = true },
        { .name = "region_end_seconds", .type = "number",
          .description = "Where it ends", .required = true },
        { .name = "alignment_seconds", .type = "number",
          .description = "The moment inside it a note is lined up to" },
        { .name = "fixed_duration_seconds", .type = "number",
          .description = "The head of the region that is never stretched" },
        { .name = "relative_pitch_cents", .type = "number",
          .description = "Cents this region is moved by" },
        { .name = "melodyne_data", .type = "boolean",
          .description = "Whether the Melodyne fields below carry edits" },
        { .name = "melodyne_pitch_center_cents", .type = "number",
          .description = "The pitch the region is sung at, in cents" },
        { .name = "melodyne_original_pitch_center_cents", .type = "number",
          .description = "The pitch it was recorded at, in cents" },
        { .name = "melodyne_pitch_drift", .type = "number",
          .description = "How much of the recording's slow pitch movement is kept, 0 to 2" },
        { .name = "melodyne_pitch_modulation", .type = "number",
          .description = "How much of its vibrato is kept, 0 to 2" },
        { .name = "melodyne_transition_seconds", .type = "number",
          .description = "How long the slide into this region takes" },
        { .name = "melodyne_formant_cents", .type = "number",
          .description = "Formant shift, in cents" },
        { .name = "melodyne_amplitude", .type = "number",
          .description = "Level as a multiplier" },
        { .name = "melodyne_sibilant_balance", .type = "number",
          .description = "Sibilance against the rest of the sound" },
    });
}

// The five every tool that analyses audio takes: each overrides what the
// environment configures, and leaving them out keeps that configuration.
std::vector<Param> withAnalysis(std::vector<Param> params)
{
    params.push_back({ .name = "game_model_dir", .type = "string",
        .description = "Folder holding the GAME model, instead of the configured one" });
    params.push_back({ .name = "fcpe_model", .type = "string",
        .description = "FCPE model file, instead of the configured one" });
    params.push_back({ .name = "game_model", .type = "string",
        .description = "Which GAME model to run (bundled default: medium)", .choices = { "medium", "large", "small" } });
    params.push_back({ .name = "inference", .type = "string",
        .description = "What runs the models",
        .choices = { "automatic", "cpu", "directml", "cuda", "coreml" } });
    params.push_back({ .name = "device_index", .type = "integer",
        .description = "Which device that backend uses; -1 lets it choose" });
    return params;
}

juce::var typedSchema(std::vector<Param> params)
{
    auto schema = object();
    set(schema, "type", "object");
    auto properties = object();
    std::vector<juce::var> required;
    for (const auto& param : params)
    {
        const juce::String type(param.type);
        // An object's fields are the schema its shape returns; an array's are
        // the schema of one element.
        auto entry = param.shape != nullptr && type == "object" ? param.shape() : object();
        set(entry, "type", param.type);
        set(entry, "description", juce::String::fromUTF8(param.description));
        if (!param.choices.empty())
        {
            std::vector<juce::var> choices;
            for (const auto* choice : param.choices) choices.emplace_back(choice);
            set(entry, "enum", array(std::move(choices)));
        }
        if (param.shape != nullptr && type == "array") set(entry, "items", param.shape());
        set(properties, param.name, std::move(entry));
        if (param.required) required.emplace_back(param.name);
    }
    set(schema, "properties", std::move(properties));
    if (!required.empty()) set(schema, "required", array(std::move(required)));
    // Left open: several tools pass settings through to the analysis
    // configuration they share, and a caller that sends something not named
    // here is no worse off than it was when nothing was named at all.
    set(schema, "additionalProperties", true);
    return schema;
}

// Refusing without saying how to allow a folder reads as a broken tool.
juce::String outsideRootsMessage(const juce::File& path,
                                 const std::vector<juce::File>& roots)
{
    juce::StringArray named;
    for (const auto& root : roots) named.add(root.getFullPathName());
    named.removeDuplicates(true);
    return "Outside the folders this session may read: " + path.getFullPathName()
        + (named.isEmpty()
               ? juce::String("; none are allowed yet -- open a project or import "
                              "something first, or start the server with "
                              "--roots=FOLDER (or HACHISHIFTER_MCP_ROOTS)")
               : "; allowed: " + named.joinIntoString("; "));
}

juce::var makeTool(const char* name, const char* description,
                   std::vector<Param> params = {})
{
    auto tool = object();
    set(tool, "name", name);
    set(tool, "description", juce::String::fromUTF8(description));
    set(tool, "inputSchema", typedSchema(std::move(params)));
    return tool;
}

double number(const juce::var& args, const char* key, double fallback = 0.0)
{
    const auto value = args.getProperty(key, fallback);
    return value.isDouble() || value.isInt() || value.isInt64() ? static_cast<double>(value) : fallback;
}

juce::String string(const juce::var& args, const char* key)
{
    return args.getProperty(key, {}).toString();
}

std::vector<juce::String> strings(const juce::var& args, const char* key)
{
    std::vector<juce::String> result;
    const auto source = args.getProperty(key, {});
    if (const auto* values = source.getArray())
        for (const auto& value : *values)
            if (const auto text = value.toString(); text.isNotEmpty()) result.push_back(text);
    return result;
}

AnalysisConfig analysisConfig(const juce::var& args)
{
    auto config = AnalysisService::configFromEnvironment();
    if (const auto path = string(args, "game_model_dir"); path.isNotEmpty())
        config.gameModelDirectory = juce::File(path);
    if (const auto path = string(args, "fcpe_model"); path.isNotEmpty())
        config.fcpeModelPath = juce::File(path);
    const auto variant = string(args, "game_model").toLowerCase();
    if (variant.isNotEmpty()) config.gameModel = variant;
    const auto inference = string(args, "inference").toLowerCase();
    if (inference.isNotEmpty())
        config.inference = inference == "cpu" ? InferenceBackend::cpu
            : inference == "directml" ? InferenceBackend::directML
            : inference == "cuda" ? InferenceBackend::cuda
            : inference == "coreml" ? InferenceBackend::coreML
            : InferenceBackend::automatic;
    if (args.hasProperty("device_index"))
        config.deviceIndex = static_cast<int>(number(args, "device_index", -1.0));
    return config;
}

juce::String analysisSummary(const AnalysisStatus& status)
{
    return "requested=" + status.requestedBackend
        + "; active=" + AnalysisService::backendText(status)
        + "; game_variant=" + status.gameModel
        + "; game_ready=" + juce::String(status.gameModelReady ? 1 : 0)
        + "; game_path=" + status.gameModelDirectory.getFullPathName()
        + "; fcpe_ready=" + juce::String(status.fcpeModelReady ? 1 : 0)
        + "; fcpe_path=" + status.fcpeModelPath.getFullPathName()
        + "; onnx_runtime=" + juce::String(status.onnxRuntimeReady ? 1 : 0)
        + "; inference_requested=" + status.requestedInference
        + "; inference_active=" + status.activeInference
        + "; message=" + status.message;
}

juce::String pitchAlgorithmText(PitchAlgorithm value)
{
    if (value == PitchAlgorithm::nsfHifigan) return "nsf-hifigan";
    if (value == PitchAlgorithm::world) return "world";
    if (value == PitchAlgorithm::vocalShifter) return "vslib";
    if (value == PitchAlgorithm::mld3) return "mld3";
    if (value == PitchAlgorithm::llsm2) return "llsm2";
    if (value == PitchAlgorithm::utau) return "utau";
    return "mld5";
}

juce::String stretchAlgorithmText(StretchAlgorithm value)
{
    if (value == StretchAlgorithm::variableMelHop) return "variable-mel-hop";
    if (value == StretchAlgorithm::loop) return "loop";
    if (value == StretchAlgorithm::soundTouch) return "soundtouch";
    if (value == StretchAlgorithm::nsfShiftThenSplice) return "nsf-shift-then-splice";
    if (value == StretchAlgorithm::hifiShifterMel) return "hifishifter-mel";
    if (value == StretchAlgorithm::wcsNonLoop) return "wcsndm-nonloop";
    return "melodyne-hybrid";
}

juce::String renderOrderText(RenderOrder value)
{
    if (value == RenderOrder::stretchSpliceThenPitch) return "stretch-splice-then-pitch";
    return "process-then-splice";
}

juce::String summary(const ProjectData& project)
{
    std::size_t clips = 0;
    std::size_t notes = 0;
    for (const auto& track : project.tracks)
        for (const auto& clip : track.clips)
        {
            ++clips;
            notes += clip.notes.size();
        }
    return "project=" + project.name + "; bpm=" + juce::String(project.bpm)
        + "; tracks=" + juce::String(static_cast<juce::int64>(project.tracks.size()))
        + "; clips=" + juce::String(static_cast<juce::int64>(clips))
        + "; notes=" + juce::String(static_cast<juce::int64>(notes));
}

juce::var sampleRowsJson(const std::vector<SampleRegionSetting>& rows)
{
    std::vector<juce::var> values;
    values.reserve(rows.size());
    for (const auto& row : rows)
    {
        auto value = object();
        set(value, "name", row.name);
        set(value, "region_start_seconds", row.regionStartSeconds);
        set(value, "region_end_seconds", row.regionEndSeconds);
        set(value, "alignment_seconds", row.alignmentSeconds);
        set(value, "fixed_duration_seconds", row.fixedDurationSeconds);
        set(value, "relative_pitch_cents", row.relativePitchCents);
        set(value, "melodyne_data", row.melodyneData);
        set(value, "melodyne_pitch_center_cents", row.melodynePitchCenterCents);
        set(value, "melodyne_original_pitch_center_cents",
            row.melodyneOriginalPitchCenterCents);
        set(value, "melodyne_pitch_drift", row.melodynePitchDrift);
        set(value, "melodyne_pitch_modulation", row.melodynePitchModulation);
        set(value, "melodyne_transition_seconds", row.melodyneTransitionSeconds);
        set(value, "melodyne_formant_cents", row.melodyneFormantCents);
        set(value, "melodyne_amplitude", row.melodyneAmplitude);
        set(value, "melodyne_sibilant_balance", row.melodyneSibilantBalance);
        set(value, "melodyne_attack_seconds", row.melodyneAttackSeconds);
        set(value, "melodyne_decay_elongation", row.melodyneDecayElongation);
        set(value, "hjm_version", row.hjmVersion);
        set(value, "native_role", nativeSegmentRoleName(row.role)); // Legacy/internal zone role, not pronunciation identity.
        set(value, "pronunciation_kind", pronunciationKindName(row.pronunciationKind));
        set(value, "simplified_consonant", row.simplifiedConsonant);
        set(value, "vowel_splits_explicit", row.vowelSplitsExplicit);
        set(value, "material_audio_id", row.materialAudioId);
        set(value, "material_region_id", row.materialRegionId);
        set(value, "connected_next_region_id", row.connectedNextRegionId);
        std::vector<juce::var> aliases;
        for (const auto& alias : row.additionalAliases) aliases.emplace_back(alias);
        set(value, "additional_aliases", array(std::move(aliases)));
        set(value, "native_provenance", row.provenance);
        set(value, "native_confidence", row.confidence);
        std::vector<juce::var> segmentValues;
        for (const auto& segment : row.segments)
        {
            auto segmentValue = object();
            set(segmentValue, "id", segment.id);
            set(segmentValue, "alias", segment.alias);
            set(segmentValue, "role", nativeSegmentRoleName(segment.role));
            set(segmentValue, "source_start_seconds", segment.sourceStartSeconds);
            set(segmentValue, "source_end_seconds", segment.sourceEndSeconds);
            set(segmentValue, "provenance", segment.provenance);
            set(segmentValue, "confidence", segment.confidence);
            set(segmentValue, "alignment_seconds", segment.alignmentSeconds);
            set(segmentValue, "overlap_seconds", segment.overlapSeconds);
            set(segmentValue, "stretchable", segment.stretchable);
            set(segmentValue, "stretch_weight", segment.stretchWeight);
            segmentValues.push_back(std::move(segmentValue));
        }
        set(value, "native_segments", array(std::move(segmentValues)));
        values.push_back(std::move(value));
    }
    return array(std::move(values));
}

std::vector<SampleRegionSetting> sampleRowsFromJson(const juce::var& source)
{
    std::vector<SampleRegionSetting> rows;
    if (const auto* values = source.getArray())
        for (const auto& value : *values)
        {
            SampleRegionSetting row;
            row.name = string(value, "name");
            row.regionStartSeconds = number(value, "region_start_seconds");
            row.regionEndSeconds = number(value, "region_end_seconds", 0.5);
            row.alignmentSeconds = number(value, "alignment_seconds");
            row.fixedDurationSeconds = number(value, "fixed_duration_seconds");
            row.relativePitchCents = number(value, "relative_pitch_cents");
            row.melodyneData = static_cast<bool>(value.getProperty("melodyne_data", false));
            row.melodynePitchCenterCents = number(value, "melodyne_pitch_center_cents");
            row.melodyneOriginalPitchCenterCents = number(
                value, "melodyne_original_pitch_center_cents");
            row.melodynePitchDrift = number(value, "melodyne_pitch_drift", 1.0);
            row.melodynePitchModulation = number(value, "melodyne_pitch_modulation", 1.0);
            row.melodyneTransitionSeconds = number(value, "melodyne_transition_seconds");
            row.melodyneFormantCents = number(value, "melodyne_formant_cents");
            row.melodyneAmplitude = number(value, "melodyne_amplitude", 1.0);
            row.melodyneSibilantBalance = number(value, "melodyne_sibilant_balance");
            row.melodyneAttackSeconds = number(value, "melodyne_attack_seconds");
            row.melodyneDecayElongation = number(value, "melodyne_decay_elongation");
            row.hjmVersion = static_cast<int>(number(value, "hjm_version", 1.0));
            row.role = parseNativeSegmentRole(string(value, "native_role"));
            row.pronunciationKind = parsePronunciationKind(string(value, "pronunciation_kind"));
            if (!value.hasProperty("pronunciation_kind") && (row.role == NativeSegmentRole::breath || row.role == NativeSegmentRole::transition))
                row.pronunciationKind = row.role == NativeSegmentRole::breath ? PronunciationKind::breath : PronunciationKind::transition;
            row.simplifiedConsonant = static_cast<bool>(value.getProperty("simplified_consonant", false));
            row.vowelSplitsExplicit = static_cast<bool>(value.getProperty("vowel_splits_explicit", false));
            row.materialAudioId = string(value, "material_audio_id");
            row.materialRegionId = string(value, "material_region_id");
            row.connectedNextRegionId = string(value, "connected_next_region_id");
            for (const auto& alias : strings(value, "additional_aliases")) row.additionalAliases.add(alias);
            row.provenance = string(value, "native_provenance");
            row.confidence = static_cast<float>(number(value, "native_confidence"));
            if (const auto* segments = value.getProperty("native_segments", {}).getArray())
                for (const auto& segmentValue : *segments)
                    row.segments.push_back({
                        string(segmentValue, "id"), string(segmentValue, "alias"),
                        parseNativeSegmentRole(string(segmentValue, "role")),
                        number(segmentValue, "source_start_seconds"),
                        number(segmentValue, "source_end_seconds"),
                        string(segmentValue, "provenance"),
                        static_cast<float>(number(segmentValue, "confidence")),
                        number(segmentValue, "alignment_seconds"),
                        number(segmentValue, "overlap_seconds"),
                        static_cast<bool>(segmentValue.getProperty("stretchable", true)),
                        number(segmentValue, "stretch_weight", 1.0) });
            if (row.regionEndSeconds > row.regionStartSeconds)
                rows.push_back(std::move(row));
        }
    return rows;
}
}

McpServer::McpServer(const juce::StringArray& extraRoots)
    : ownedProject(std::make_unique<ProjectModel>()), project(*ownedProject)
{
    // Folders this server may read whatever else happens: given on the command
    // line as --roots=A;B, or in HACHISHIFTER_MCP_ROOTS.  Everything else it
    // may read it earns by being asked to work there.
    juce::StringArray named(extraRoots);
    named.addTokens(juce::SystemStats::getEnvironmentVariable(
        "HACHISHIFTER_MCP_ROOTS", {}), ";", "\"");
    for (const auto& entry : named)
    {
        const juce::File folder(entry.trim().unquoted());
        if (folder != juce::File() && folder.isDirectory())
            configuredRoots.push_back(folder);
    }

    formats.registerBasicFormats();
}

McpServer::McpServer(ProjectModel& model, AudioEngine* engine, const juce::StringArray& roots)
    : project(model), audio(engine)
{
    formats.registerBasicFormats();
    for (const auto& root : roots)
        if (juce::File(root).isDirectory()) configuredRoots.emplace_back(root);
}

int McpServer::run()
{
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) continue;
        const auto request = juce::JSON::parse(juce::String::fromUTF8(line.data(),
                                                                       static_cast<int>(line.size())));
        bool shouldRespond = true;
        auto response = handle(request, shouldRespond);
        if (shouldRespond)
        {
            std::cout << juce::JSON::toString(response, true).toStdString() << '\n';
            std::cout.flush();
        }
    }
    return 0;
}

// Every tool this server offers, with the parameters each takes.  Public so a
// check reads the same list a client is sent rather than a copy of it.
juce::var McpServer::diagnosticTools()
{
    return array({
            makeTool("project_new", "Create an empty HachiShifter project / 新建工程"),
            makeTool("project_open", "Open HJPX, legacy HSPX, MPD, MIDI, or audio from path / 打开工程或素材",
                withAnalysis({
                    { .name = "path", .type = "string",
                      .description = "The file to open: .hjpx, .hspx, .mpd, .mid or a recording.  "
                                     "A recording is analysed as it comes in.",
                      .required = true },
                })),
            makeTool("project_save", "Save current project as HJPX / 保存工程", {
                { .name = "path", .type = "string",
                  .description = "Where to write the .hjpx", .required = true },
            }),
            makeTool("project_snapshot", "Read every current track, clip, note, pitch and marker / 读取全部工程内容"),
            makeTool("import_audio", "Import an audio file at start_seconds / 导入音频",
                withAnalysis({
                    { .name = "path", .type = "string",
                      .description = "The recording to bring in", .required = true },
                    { .name = "start_seconds", .type = "number",
                      .description = "Where on the timeline it starts.  Default 0." },
                    { .name = "track_id", .type = "string",
                      .description = "Which track to put it on; empty makes a new one" },
                })),
            makeTool("analyse_audio", "Run configured GAME+FCPE analysis with native fallback and return actual backend / 执行 GAME+FCPE 分析并报告实际后端",
                withAnalysis({
                    { .name = "path", .type = "string",
                      .description = "The recording to analyse.  Nothing is added to the project.",
                      .required = true },
                })),
            makeTool("analysis_status", "Inspect GAME large/small, FCPE and inference availability / 查看 GAME、FCPE 与推理状态",
                withAnalysis({})),
            makeTool("import_midi", "Import MIDI notes and tempo / 导入 MIDI", {
                { .name = "path", .type = "string",
                  .description = "The MIDI file to add as tracks", .required = true },
            }),
            makeTool("export_midi", "Write all tracks or one track as MIDI, retaining conductor tempo/meter and lyrics / 导出 MIDI", {
                { .name = "path", .type = "string",
                  .description = "Where to write the .mid", .required = true },
                { .name = "track_id", .type = "string",
                  .description = "Optional track ID; empty exports the whole project. Unknown IDs fail without changing the destination." },
                { .name = "include_muted", .type = "boolean",
                  .description = "Include muted tracks and clips (default false). Reserved rests and native unpitched regions are always excluded." },
            }),
            makeTool("export_ust", "Export one monophonic track as UST, retaining imported opaque data / 导出单轨 UST", {
                {.name="path",.type="string",.description="Output .ust path",.required=true},
                {.name="track_id",.type="string",.description="Track to export",.required=true},
                {.name="encoding",.type="string",.description="auto (preserve), UTF-8, Shift-JIS, GBK, Big5 or GB18030"},
            }),
            makeTool("import_ust", "Import a UTAU project as a plain UTAU track / 导入 UST", {
                { .name = "path", .type = "string",
                  .description = "The .ust to add as a track", .required = true },
                {.name="encoding",.type="string",.description="auto, UTF-8, Shift-JIS, GBK, Big5 or GB18030"},
            }),
            makeTool("import_melodyne", "Import Melodyne MPD edits; recursive_media, preserve_edits and source_pitch control import / 导入 Melodyne 工程并控制素材搜索、工程编辑与原始 F0",
                withAnalysis({
                    { .name = "path", .type = "string",
                      .description = "The .mpd to open", .required = true },
                    { .name = "recursive_media", .type = "boolean",
                      .description = "Search subfolders for the recordings it names.  Default true." },
                    { .name = "preserve_edits", .type = "boolean",
                      .description = "Keep the edits saved in the document.  Default true." },
                    { .name = "source_pitch", .type = "string",
                      .description = "Where each note's source pitch comes from: keep what the "
                                     "document holds, or read it again here.  reanalyze, "
                                     "reanalyse and game_fcpe mean the same as game+fcpe.",
                      .choices = { "keep", "native", "game+fcpe" } },
                })),
            makeTool("set_tempo", "Set BPM and time signature / 设置速度与拍号", {
                { .name = "bpm", .type = "number",
                  .description = "Beats a minute, 20 to 400.  Default 120." },
                { .name = "numerator", .type = "integer",
                  .description = "Beats in a bar.  Default 4." },
                { .name = "denominator", .type = "integer",
                  .description = "What counts as a beat.  Default 4." },
            }),
            makeTool("add_track", "Create a melodic, audio or accompaniment track / 新建旋律、普通或伴奏轨道", {
                { .name = "name", .type = "string", .description = "What to call it" },
                { .name = "compose", .type = "boolean",
                  .description = "A melodic track that sings notes, rather than one that "
                                 "plays recordings.  Default true." },
                { .name = "accompaniment", .type = "boolean",
                  .description = "Backing audio with all tuning engines disabled. Overrides compose. Default false." },
            }),
            makeTool("set_track", "Set compose, mute, solo, gain, pan and render algorithms / 设置轨道及算法", {
                { .name = "track_id", .type = "string",
                  .description = "Which track to change", .required = true },
                { .name = "name", .type = "string", .description = "What to call it" },
                { .name = "compose", .type = "boolean",
                  .description = "Whether it sings notes" },
                { .name = "muted", .type = "boolean", .description = "Silence this track" },
                { .name = "solo", .type = "boolean", .description = "Silence every other track" },
                { .name = "volume", .type = "number",
                  .description = "Level as a multiplier; 1 leaves it alone" },
                { .name = "pan", .type = "number",
                  .description = "Where it sits, -1 left to 1 right" },
                { .name = "smooth_overlaps", .type = "boolean",
                  .description = "Crossfade where notes overlap" },
                { .name = "nsf_smooth_pitch_transitions", .type = "boolean",
                  .description = "Native NSF automatic pitch-jump smoothing; false retains authored discontinuities" },
                { .name = "nsf_noise_protection", .type = "boolean",
                  .description = "Native NSF harmonic/noise separation, source-noise time mapping and clear unvoiced PCM restoration; on by default" },
                { .name = "normalize_volume", .type = "boolean",
                  .description = "Even out the level across the track" },
                { .name = "pitch_algorithm", .type = "string",
                  .description = "What sings the notes.  utau is a plain UTAU track; utau4 (jie) "
                                 "and utaumou (mou) are the four-region modes.",
                  .choices = { "mld5", "mld3", "llsm2", "world", "vslib", "nsf-hifigan",
                               "utau", "utau4", "jie", "utaumou", "mou" } },
                { .name = "output_engine", .type = "string",
                  .description = "UTAU output backend, independent of classic/jie/mou editing mode. pc-nsf-hifigan selects HiFisampler (PC-NSF-HiFiGAN); inherit follows Settings; DiffSinger keeps its own renderer.",
                  .choices = { "inherit", "resampler", "pc-nsf-hifigan" } },
                { .name = "output_resampler", .type = "string", .description = "Per-track resampler path; empty follows Settings." },
                { .name = "output_wavtool", .type = "string", .description = "Per-track assembly tool path; empty follows Settings." },
                { .name = "stretch_algorithm", .type = "string",
                  .description = "What changes a recording's length",
                  .choices = { "melodyne-hybrid", "variable-mel-hop", "loop", "soundtouch",
                               "nsf-shift-then-splice", "hifishifter-mel", "wcsndm-nonloop" } },
                { .name = "render_order", .type = "string",
                  .description = "Whether notes are joined before or after they are processed",
                  .choices = { "process-then-splice", "stretch-splice-then-pitch" } },
                { .name = "merge_contiguous_source", .type = "boolean",
                  .description = "Decode explicitly connected same-source clips together, including process order; default false." },
                { .name = "utau_global_flags", .type = "string",
                  .description = "Flags handed to the engine for every note of this track" },
                { .name = "voicebank_directory", .type = "string",
                  .description = "The UTAU voicebank folder this track sings from" },
            }),
            makeTool("set_clip", "Set clip gain, fades and mute state / 设置采样增益、淡入淡出和静音", {
                { .name = "clip_id", .type = "string",
                  .description = "Which clip to change", .required = true },
                { .name = "gain", .type = "number",
                  .description = "Level as a multiplier; 1 leaves it alone" },
                { .name = "fade_in_seconds", .type = "number",
                  .description = "How long it fades in for" },
                { .name = "fade_out_seconds", .type = "number",
                  .description = "How long it fades out for" },
                { .name = "muted", .type = "boolean", .description = "Silence this clip" },
            }),
            makeTool("move_clip", "Move a clip on the timeline / 移动采样", {
                { .name = "clip_id", .type = "string",
                  .description = "Which clip to move", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where it starts afterwards", .required = true },
            }),
            makeTool("resize_clip", "Stretch a whole clip while preserving its source media / 整体拉伸采样并保留原始素材", {
                { .name = "clip_id", .type = "string",
                  .description = "Which clip to stretch", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where it starts afterwards", .required = true },
                { .name = "duration_seconds", .type = "number",
                  .description = "How long it lasts afterwards.  Default 0.25." },
            }),
            makeTool("duplicate_clip", "Deep-copy a clip and its notes to a timeline position / 深度复制采样及其音符到指定位置", {
                { .name = "clip_id", .type = "string",
                  .description = "The clip to copy", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where the copy starts; leave it out to place it after the original" },
                { .name = "track_id", .type = "string",
                  .description = "Which track the copy goes on; empty keeps it on its own" },
            }),
            makeTool("transpose_note", "Move a note and its whole contour / 整体移动音高线", {
                { .name = "note_id", .type = "string",
                  .description = "Which note to move", .required = true },
                { .name = "semitones", .type = "number",
                  .description = "How far to move it, in semitones", .required = true },
            }),
            makeTool("edit_notes_pitch", "Batch transpose, set, average or quantize note pitches / 批量移调、设置、平均或量化音符", {
                { .name = "note_ids", .type = "array",
                  .description = "The notes to change", .shape = []() { return typedSchema({}); } },
                { .name = "note_id", .type = "string",
                  .description = "One note, when there is only one" },
                { .name = "action", .type = "string",
                  .description = "What to do to them: move by cents, put them all on one pitch, "
                                 "put them on their average, or snap them to a step",
                  .choices = { "transpose", "set", "average", "quantize" } },
                { .name = "cents", .type = "number",
                  .description = "How far to move them, for transpose" },
                { .name = "midi", .type = "number",
                  .description = "The pitch they all take, for set.  Default 60." },
                { .name = "step_semitones", .type = "number",
                  .description = "The step they snap to, for quantize.  Default 1." },
            }),
            makeTool("duplicate_notes", "Deep-copy selected notes into a target clip / 深度复制所选音符到目标采样", {
                { .name = "note_ids", .type = "array",
                  .description = "The notes to copy", .shape = []() { return typedSchema({}); } },
                { .name = "note_id", .type = "string",
                  .description = "One note, when there is only one" },
                { .name = "clip_id", .type = "string",
                  .description = "The clip the copies go in", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where the first copy starts" },
            }),
            makeTool("resize_note", "Change note time bounds / 修改音符时间", {
                { .name = "note_id", .type = "string",
                  .description = "Which note to change", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where it starts, from the clip start", .required = true },
                { .name = "duration_seconds", .type = "number",
                  .description = "How long it lasts.  Default 0.25." },
            }),
            makeTool("set_note", "Set pitch, Robust Pitch Curve, tension, breath, formant, gain, Attack and consonant parameters / 设置稳健音高线及全部音符参数", {
                { .name = "note_id", .type = "string",
                  .description = "Which note to change", .required = true },
                { .name = "label", .type = "string",
                  .description = "The lyric sung on it" },
                { .name = "gain", .type = "number",
                  .description = "Level as a multiplier; 1 is as written" },
                { .name = "tension", .type = "number", .description = "Tension, -1 to 1" },
                { .name = "breath", .type = "number", .description = "Breathiness, -1 to 1" },
                { .name = "formant_semitones", .type = "number",
                  .description = "Formant shift, in semitones" },
                { .name = "drift", .type = "number",
                  .description = "How much of the recording's slow pitch movement is kept, 0 to 2" },
                { .name = "modulation", .type = "number",
                  .description = "How much of its vibrato is kept, 0 to 2" },
                { .name = "robust_pitch_curve", .type = "boolean",
                  .description = "Follow the drawn pitch line rather than the recording's own" },
                { .name = "consonant_seconds", .type = "number",
                  .description = "The head of the note that is not stretched" },
                { .name = "attack_speed", .type = "number",
                  .description = "How quickly the note is reached" },
                { .name = "amplitude_envelope", .type = "array",
                  .description = "[[seconds, dB], ...] from the note's start; negative seconds "
                                 "reach into the preutterance" },
                { .name = "tail_fade", .type = "string", .description = "OTO last-region amplitude effect for non-DS UTAU; off, linear, smooth", .choices = {"off","linear","smooth"} },
                { .name = "tail_fade_start_percent", .type = "number", .description = "Fade start within OTO tail, 0..99.9 percent" },
                { .name = "tail_fade_end_percent", .type = "number", .description = "Fade end within OTO tail, 0.1..100 percent; strictly after start" },
                { .name = "tail_fade_start_gain_percent", .type = "number", .description = "Amplitude multiplier before fade, 0..200 percent" },
                { .name = "tail_fade_end_gain_percent", .type = "number", .description = "Amplitude multiplier after fade, 0..200 percent; not greater than start gain" },
                { .name = "tail_fade_curve_power", .type = "number", .description = "Time curve exponent, 0.25..4; 1 is neutral" },
                { .name = "head_envelope", .type = "string", .description = "OTO first-region envelope mode: off, linear, curve", .choices = {"off","linear","curve"} },
                { .name = "head_envelope_custom_curve", .type = "boolean", .description = "Use Bezier control points for the head envelope" },
                { .name = "head_envelope_start_percent", .type = "number", .description = "First-region envelope start_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_end_percent", .type = "number", .description = "First-region envelope end_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_start_gain_percent", .type = "number", .description = "First-region envelope start_gain_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_end_gain_percent", .type = "number", .description = "First-region envelope end_gain_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_curve_power", .type = "number", .description = "First-region envelope curve_power; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_control1_time_percent", .type = "number", .description = "First-region envelope control1_time_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_control1_progress_percent", .type = "number", .description = "First-region envelope control1_progress_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_control2_time_percent", .type = "number", .description = "First-region envelope control2_time_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "head_envelope_control2_progress_percent", .type = "number", .description = "First-region envelope control2_progress_percent; gain 0..200%, ordered time/progress 0..100%, power 0.25..4" },
                { .name = "tail_fade_custom_curve", .type = "boolean", .description = "Use cubic Bezier controls in smooth mode; false preserves the legacy S curve" },
                { .name = "tail_fade_control1_time_percent", .type = "number", .description = "Bezier time coordinate, 0..100; control 1 must not exceed control 2" },
                { .name = "tail_fade_control1_progress_percent", .type = "number", .description = "Bezier fade progress coordinate, 0..100; control 1 must not exceed control 2" },
                { .name = "tail_fade_control2_time_percent", .type = "number", .description = "Bezier time coordinate, 0..100; control 1 must not exceed control 2" },
                { .name = "tail_fade_control2_progress_percent", .type = "number", .description = "Bezier fade progress coordinate, 0..100; control 1 must not exceed control 2" },
                { .name = "amplitude_envelope_base", .type = "number",
                  .description = "The envelope's height as a whole, in UTAU's linear percent: "
                                 "100 is the envelope as drawn, 200 twice as loud, 0 silence.  "
                                 "A note with no envelope has a flat 100% one, raised the same "
                                 "way.  0 to 200." },
                { .name = "utau_flags", .type = "string",
                  .description = "Flags for this note. HiFisampler supports g[-600,600], Hb[0,500], Hv[0,150], Ht[-100,100], HG[0,100], P[0,100], t[-1200,1200] cents, A[-100,100], G and He. WCSNDM flags depend on the selected resampler." },
                { .name = "utau_consonant_velocity", .type = "integer",
                  .description = "Consonant velocity, 0 to 200; 100 is as recorded" },
                { .name = "utau_splice", .type = "boolean",
                  .description = "Join this note to the one before it rather than re-attacking" },
                { .name = "jie_split", .type = "array",
                  .description = "Three rising fractions of the note naming where its four "
                                 "regions meet; anything else hands the split back to the engine" },
                { .name = "region_flags", .type = "array",
                  .description = "Flags per region, as [\"f1\",\"f2\",\"f3\",\"f4\"]; an empty "
                                 "entry leaves that region on the note's own flags",
                  .shape = []() { return typedSchema({}); } },
                { .name = "flag_split", .type = "boolean",
                  .description = "Whether region_flags are used at all.  Default true." },
                { .name = "flag_curve_enabled", .type = "boolean",
                  .description = "Whether this note's flag curves are drawn on" },
                { .name = "flag_curve", .type = "object",
                  .description = "One flag's curve across the note. HiFisampler keys are HIFI:g, HIFI:Hb, HIFI:Hv, HIFI:Ht, HIFI:HG, HIFI:P, HIFI:t, HIFI:A; these preserve the native flag units.",
                  .shape = flagCurveShape },
                { .name = "flag_curve_g", .type = "array",
                  .description = "The g curve on its own, as [[seconds, value], ...]; an empty "
                                 "list drops it" },
            }),
            makeTool("set_pitch_curve", "Draw an absolute target-pitch curve without replacing source F0 / 绘制目标音高线并保留原始 F0", {
                { .name = "note_id", .type = "string",
                  .description = "Which note the curve belongs to", .required = true },
                { .name = "points", .type = "array",
                  .description = "The curve, in order; at least one point",
                  .required = true, .shape = pitchPointShape },
            }),
            makeTool("add_note", "Create a note in a clip / 在采样中创建音符", {
                { .name = "clip_id", .type = "string",
                  .description = "The clip to put it in", .required = true },
                { .name = "start_seconds", .type = "number",
                  .description = "Where it starts, from the clip start", .required = true },
                { .name = "duration_seconds", .type = "number",
                  .description = "How long it lasts.  Default 0.25." },
                { .name = "midi", .type = "number",
                  .description = "Its pitch, as a MIDI note.  Default 60." },
            }),
            makeTool("remove_note", "Delete a note / 删除音符", {
                { .name = "note_id", .type = "string",
                  .description = "Which note to delete", .required = true },
            }),
            makeTool("toggle_note_connection", "Connect or separate adjacent notes / 连接或分离相邻音符", {
                { .name = "note_id", .type = "string",
                  .description = "The note whose join with the one before it is turned over",
                  .required = true },
            }),
            makeTool("set_notes_portamento", "Set real pitch-only glide bridges; never join source audio or change envelopes / 设置真实滑音连接，不拼接素材", {
                { .name="note_ids", .type="array", .description="Consecutive voiced notes on one track", .required=true,
                  .shape=[](){return typedSchema({});} },
                { .name="enabled", .type="boolean", .description="Enable or remove the selected pitch bridges; default true" },
                { .name="duration_seconds", .type="number", .description="Requested total bridge duration; default 0.1 seconds" },
            }),
            makeTool("remove_clip", "Delete a clip / 删除采样", {
                { .name = "clip_id", .type = "string",
                  .description = "Which clip to delete", .required = true },
            }),
            makeTool("remove_track", "Delete a track / 删除轨道", {
                { .name = "track_id", .type = "string",
                  .description = "Which track to delete", .required = true },
            }),
            makeTool("undo", "Undo the last project edit / 撤销工程编辑"),
            makeTool("redo", "Redo the last project edit / 重做工程编辑"),
            makeTool("utau_render_selection", "Choose exact UTAU/DS render scope; [] clears it, omitted IDs use GUI selection (headless: empty) / 仅渲染所选音符", {
                { .name = "note_ids", .type = "array",
                  .description = "Exact note IDs; an empty array clears the render selection",
                  .shape = []() { auto item=object(); set(item,"type","string"); return item; } },
                { .name = "note_id", .type = "string",
                  .description = "One note, when there is only one" },
                { .name = "whole_project", .type = "boolean",
                  .description = "Explicitly select every voicebank note; cannot be combined with note IDs" },
            }),
            makeTool("set_utau_resampler", "Point UTAU rendering at a resampler executable / 指定 UTAU 重采样器", {
                { .name = "path", .type = "string",
                  .description = "The resampler .exe to render with; it has to exist",
                  .required = true },
            }),
            makeTool("render_prepare", "Pre-render the current project with its selected algorithms / 按当前所选算法预渲染工程", {
                { .name = "wait", .type = "boolean",
                  .description = "Wait for the render to finish before answering.  Default false." },
                { .name = "timeout_seconds", .type = "number",
                  .description = "How long to wait, 0.1 to 3600.  Default 300." },
            }),
            makeTool("render_status", "Read pre-render progress and active backends / 读取预渲染进度与实际后端"),
            makeTool("export_wav", "Render every note and export to WAV; track_id exports one track alone, from_seconds/to_seconds one stretch / 渲染全曲并导出 WAV，track_id 可单独导出一个轨道", {
                { .name = "path", .type = "string",
                  .description = "Where to write the .wav", .required = true },
                { .name = "channels", .type = "integer", .description = "1 mono or 2 stereo; default 2" },
                { .name = "sample_rate", .type = "integer", .description = "8000..192000 Hz, 0 follows device" },
                { .name = "bit_depth", .type = "integer", .description = "16/24 PCM or 32 float; default 24" },
                { .name = "track_id", .type = "string",
                  .description = "Export this track alone; empty exports the mix" },
                { .name = "from_seconds", .type = "number",
                  .description = "Start of the stretch to write.  Default 0." },
                { .name = "to_seconds", .type = "number",
                  .description = "End of the stretch; 0 writes to the end of the song" },
                { .name = "timeout_seconds", .type = "number",
                  .description = "How long to wait for the render, 0.1 to 3600.  Default 300." },
            }),
            makeTool("transport_play", "Render if needed and start playback; play_until_seconds stops where a selection ends / 必要时预渲染并开始播放", {
                { .name = "position_seconds", .type = "number",
                  .description = "Where to play from; leave it out to carry on from here" },
                { .name = "play_until_seconds", .type = "number",
                  .description = "Where to stop, as a selection's end does" },
                { .name = "timeout_seconds", .type = "number",
                  .description = "How long to wait for the render, 0.1 to 3600.  Default 300." },
            }),
            makeTool("transport_stop", "Stop transport playback / 停止播放"),
            makeTool("transport_seek", "Seek transport to position_seconds / 跳转播放位置", {
                { .name = "position_seconds", .type = "number",
                  .description = "Where to move the playhead", .required = true },
            }),
            makeTool("transport_status", "Read playback position and render state / 读取播放位置与渲染状态"),
            makeTool("sample_settings_read", "Read or derive audio otto.ini regions / 读取或生成音频 otto.ini 分段", {
                { .name = "audio_path", .type = "string",
                  .description = "The recording whose regions are read; they are derived when "
                                 "it has no sidecar yet", .required = true },
            }),
            makeTool("sample_settings_save", "Save audio regions to otto.ini / 保存音频分段到 otto.ini", {
                { .name = "audio_path", .type = "string",
                  .description = "The recording the regions belong to", .required = true },
                { .name = "rows", .type = "array",
                  .description = "The regions to write; at least one", .required = true,
                  .shape = regionRowShape },
            }),
            makeTool("oto_import", "Import one audio file's regions from UTAU oto.ini / 从 UTAU oto.ini 导入单个音频分段", {
                { .name = "audio_path", .type = "string",
                  .description = "The recording the entries describe", .required = true },
                { .name = "oto_path", .type = "string",
                  .description = "The oto.ini to read", .required = true },
                { .name = "save_sidecar", .type = "boolean",
                  .description = "Write the regions to the recording's otto.ini as well.  "
                                 "Default true." },
            }),
            makeTool("oto_export", "Export one audio file's regions to UTAU oto.ini / 将单个音频分段导出为 UTAU oto.ini", {
                { .name = "audio_path", .type = "string",
                  .description = "The recording whose regions are written", .required = true },
                { .name = "oto_path", .type = "string",
                  .description = "The oto.ini to write", .required = true },
            }),
            makeTool("jie_oto_create", "Seed a voicebank's four-region oto4.ini from its oto.ini / 按 oto.ini 生成界•OTO", {
                { .name = "voicebank_path", .type = "string",
                  .description = "The voicebank folder holding oto.ini", .required = true },
            }),
            makeTool("voicebank_import", "Inspect a UTAU or native material folder without writing annotations / 只读检查 UTAU 或原生素材目录，不写入标注",  {
                { .name = "path", .type = "string",
                  .description = "The voicebank folder to read", .required = true },
            }),
            makeTool("read_file", "Read a byte range as base64 / 读取任意文件内容", {
                { .name = "path", .type = "string",
                  .description = "The file to read", .required = true },
                { .name = "offset", .type = "integer",
                  .description = "Where to start reading, in bytes.  Default 0." },
                { .name = "max_bytes", .type = "integer",
                  .description = "How much to read, 1 to 4194304.  Default 1048576." },
            }),
            makeTool("list_directory", "List a directory with type and size / 列出目录内容", {
                { .name = "path", .type = "string",
                  .description = "The folder to list", .required = true },
            })
    });
}

juce::var McpServer::handle(const juce::var& request, bool& shouldRespond)
{
    shouldRespond = request.hasProperty("id");
    const auto id = request.getProperty("id", {});
    const auto method = request.getProperty("method", {}).toString();
    auto response = object();
    set(response, "jsonrpc", "2.0");
    set(response, "id", id);
    if (method == "initialize")
    {
        auto result = object();
        set(result, "protocolVersion", "2025-06-18");
        auto capabilities = object();
        set(capabilities, "tools", object());
        set(capabilities, "resources", object());
        set(result, "capabilities", capabilities);
        auto serverInfo = object();
        set(serverInfo, "name", "hachishifter-next");
        set(serverInfo, "version", JUCE_APPLICATION_VERSION_STRING);
        set(result, "serverInfo", serverInfo);
        set(response, "result", result);
        return response;
    }
    if (method == "ping")
    {
        set(response, "result", object());
        return response;
    }
    if (method == "tools/list")
    {
        auto result = object();
        set(result, "tools", diagnosticTools());
        set(response, "result", result);
        return response;
    }
    if (method == "tools/call")
    {
        const auto params = request.getProperty("params", {});
        set(response, "result", callTool(string(params, "name"),
                                          params.getProperty("arguments", object())));
        return response;
    }
    if (method == "resources/list")
    {
        auto resource = object();
        set(resource, "uri", "hachishifter://project/current");
        set(resource, "name", "Current HachiShifter project");
        set(resource, "mimeType", "application/json");
        auto result = object();
        set(result, "resources", array({ resource }));
        set(response, "result", result);
        return response;
    }
    if (method == "resources/read")
    {
        const auto params = request.getProperty("params", {});
        if (string(params, "uri") != "hachishifter://project/current")
            return errorResponse(id, -32002, "Unknown resource");
        auto content = object();
        set(content, "uri", "hachishifter://project/current");
        set(content, "mimeType", "application/json");
        set(content, "text", juce::JSON::toString(projectJson(), false));
        auto result = object();
        set(result, "contents", array({ content }));
        set(response, "result", result);
        return response;
    }
    if (!shouldRespond) return {};
    return errorResponse(id, -32601, "Method not found");
}

bool McpServer::pathWithinRoots(const std::vector<juce::File>& roots,
                                const juce::File& target)
{
    if (target == juce::File()) return false;
    // A link is followed before it is judged, or a link inside a root would
    // hand back whatever it points at.  One level: a link somewhere further up
    // the path is not followed, which is said plainly in docs/mcp.md.
    const auto resolved = target.isSymbolicLink() ? target.getLinkedTarget() : target;
    for (const auto& root : roots)
    {
        if (root == juce::File()) continue;
        const auto within = root.isSymbolicLink() ? root.getLinkedTarget() : root;
        if (within == juce::File()) continue;
        // isAChildOf compares whole folders, so a root is not a prefix of a
        // name that merely begins with it.
        if (resolved == within || resolved.isAChildOf(within)) return true;
    }
    return false;
}

std::vector<juce::File> McpServer::projectRoots(const ProjectData& data)
{
    std::vector<juce::File> roots;
    const auto add = [&roots](const juce::File& folder)
    {
        if (folder == juce::File() || !folder.isDirectory()) return;
        for (const auto& known : roots) if (known == folder) return;
        roots.push_back(folder);
    };
    for (const auto& track : data.tracks)
    {
        add(track.voicebankDirectory);
        for (const auto& clip : track.clips)
            if (clip.sourceFile != juce::File())
                add(clip.sourceFile.getParentDirectory());
    }
    return roots;
}

std::vector<juce::File> McpServer::allowedRoots() const
{
    auto roots = configuredRoots;
    for (const auto& root : sessionRoots) roots.push_back(root);
    for (const auto& root : projectRoots(project.snapshot())) roots.push_back(root);
    return roots;
}

void McpServer::allow(const juce::File& file)
{
    const auto folder = file.isDirectory() ? file : file.getParentDirectory();
    if (folder == juce::File() || !folder.isDirectory()) return;
    for (const auto& known : sessionRoots) if (known == folder) return;
    sessionRoots.push_back(folder);
}

juce::var McpServer::callTool(const juce::String& name, const juce::var& args)
{
    const auto result = dispatch(name, args);
    // A tool that was handed a path and did its work makes that folder one the
    // session may read afterwards: it is where the work it was asked to do
    // lives.  Only on success, so a path that failed grants nothing.
    if (!static_cast<bool>(result.getProperty("isError", false)))
        for (const auto* key : { "path", "audio_path", "oto_path", "voicebank_path",
                                 "voicebank_directory" })
            if (const auto given = string(args, key); given.isNotEmpty())
                allow(juce::File(given));
    return result;
}

bool McpServer::resolveUtauRenderSelection(const ProjectData& data, const juce::var& args,
    const std::vector<juce::String>& current, std::vector<juce::String>& result, juce::String& error)
{
    result.clear(); error.clear();
    const bool whole = static_cast<bool>(args.getProperty("whole_project", false));
    if ((whole && (args.hasProperty("note_ids") || args.hasProperty("note_id")))
        || (args.hasProperty("note_ids") && args.hasProperty("note_id")))
    { error = "Choose only one scope: note_ids, note_id, or whole_project"; return false; }
    std::vector<juce::String> eligible;
    for (const auto& track : data.tracks)
        if (!track.accompaniment && trackUsesVoicebankSynthesis(track))
            for (const auto& clip : track.clips)
                for (const auto& note : clip.notes) eligible.push_back(note.id);
    if (whole) result = eligible;
    else if (args.hasProperty("note_ids"))
    {
        if (!args["note_ids"].isArray()) { error = "note_ids must be an array"; return false; }
        for (const auto& id : *args["note_ids"].getArray())
        {
            if (!id.isString() || id.toString().isEmpty())
            { error = "note_ids must contain nonempty strings"; return false; }
            result.push_back(id.toString());
        }
    }
    else if (args.hasProperty("note_id")) result.push_back(string(args, "note_id"));
    else result = current;
    for (const auto& id : result)
        if (std::find(eligible.begin(), eligible.end(), id) == eligible.end())
        { error = "Not a current UTAU/DS note: " + id; return false; }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return true;
}

juce::var McpServer::dispatch(const juce::String& name, const juce::var& args)
{
    juce::String error;
    if (audio == nullptr && (name.startsWith("transport_") || name.startsWith("render_")
        || name == "export_wav" || name == "utau_render_selection" || name == "set_utau_resampler"))
    {
        ownedAudio = std::make_unique<AudioEngine>();
        audio = ownedAudio.get();
    }
    if (name == "project_new")
    {
        project.clear();
        return toolResult("ok");
    }
    if (name == "project_snapshot")
        return toolResult(juce::JSON::toString(projectJson(), false));
    if (name == "project_save")
    {
        const juce::File file(string(args, "path"));
        if (project.save(file, error)) return toolResult("saved=" + file.getFullPathName());
    }
    else if (name == "project_open")
    {
        const juce::File file(string(args, "path"));
        if (file.hasFileExtension("mpd"))
        {
            if (auto imported = MelodyneImporter::importProject(file, error))
            {
                juce::StringArray annotationWarnings;
                SampleSettings::convertMelodyneProject(imported->project,
                                                       annotationWarnings);
                project.resetDocument(std::move(imported->project));
                return toolResult(summary(project.snapshot())
                    + (annotationWarnings.isEmpty() ? juce::String()
                        : "; annotation_warnings="
                            + annotationWarnings.joinIntoString(" | ")));
            }
        }
        else if (file.hasFileExtension("mid;midi"))
        {
            project.clear();
            if (project.addMidiFile(file, error)) return toolResult(summary(project.snapshot()));
        }
        else if (file.hasFileExtension("hjpx;hspx"))
        {
            if (project.load(file, error))
                return toolResult(summary(project.snapshot())
                    + (error.isNotEmpty() ? "; missing_media=" + error : juce::String()));
        }
        else if (auto reader = createAudioReader(formats, file))
        {
            project.clear();
            const auto clipId = project.addAudioFile(file,
                static_cast<double>(reader->lengthInSamples) / reader->sampleRate);
            auto analysis = AnalysisService::analyse(file, analysisConfig(args), error);
            (void) project.setClipAudioAnalysis(clipId, std::move(analysis.notes));
            return toolResult(summary(project.snapshot()));
        }
        else error = "Unsupported or unreadable file";
    }
    else if (name == "import_audio")
    {
        const juce::File file(string(args, "path"));
        if (auto reader = createAudioReader(formats, file))
        {
            const auto clipId = project.addAudioFile(file,
                static_cast<double>(reader->lengthInSamples) / reader->sampleRate,
                number(args, "start_seconds"), string(args, "track_id"));
            auto analysis = AnalysisService::analyse(file, analysisConfig(args), error);
            const auto backend = AnalysisService::backendText(analysis.status);
            const auto noteCount = analysis.notes.size();
            (void) project.setClipAudioAnalysis(clipId, std::move(analysis.notes));
            return toolResult("ok; backend=" + backend + "; notes="
                + juce::String(static_cast<juce::int64>(noteCount))
                + (analysis.warning.isNotEmpty() ? "; warning=" + analysis.warning
                                                  : juce::String()));
        }
        error = "Audio read failed";
    }
    else if (name == "analysis_status")
        return toolResult(analysisSummary(AnalysisService::status(analysisConfig(args))));
    else if (name == "analyse_audio")
    {
        const juce::File file(string(args, "path"));
        auto analysis = AnalysisService::analyse(file, analysisConfig(args), error);
        if (!analysis.notes.empty())
            return toolResult(analysisSummary(analysis.status) + "; notes="
                + juce::String(static_cast<juce::int64>(analysis.notes.size()))
                + (analysis.warning.isNotEmpty() ? "; warning=" + analysis.warning
                                                  : juce::String()));
    }
    else if (name == "import_midi")
    {
        if (project.addMidiFile(juce::File(string(args, "path")), error)) return toolResult("ok");
    }
    else if (name == "export_midi")
    {
        const juce::File file(string(args, "path"));
        if (ProjectModel::writeMidiFile(project.snapshot(), file, error,
                string(args, "track_id"), static_cast<bool>(args.getProperty("include_muted", false))))
            return toolResult("written=" + file.getFullPathName());
    }
    else if (name == "export_ust" || name == "import_ust")
    {
        const auto encoding=string(args,"encoding");const auto cp=LegacyTextCodec::codePage(encoding);
        if(cp==0&&encoding.isNotEmpty()&&!encoding.equalsIgnoreCase("auto"))return toolResult("Unsupported encoding: "+encoding,true);
        juce::StringArray warnings;
        const bool ok=name=="export_ust"?project.exportUst(juce::File(string(args,"path")),string(args,"track_id"),error,warnings,cp)
            :project.addUstFile(juce::File(string(args,"path")),error,warnings,ProjectModel::UstImportMode::addTrack,nullptr,cp);
        if (ok)
            return toolResult(warnings.isEmpty()
                ? juce::String("ok")
                : "ok; " + warnings.joinIntoString("; "));
    }
    else if (name == "import_melodyne")
    {
        MelodyneImportOptions options;
        options.recursiveMediaSearch = static_cast<bool>(
            args.getProperty("recursive_media", true));
        options.preserveProjectEdits = static_cast<bool>(
            args.getProperty("preserve_edits", true));
        if (auto imported = MelodyneImporter::importProject(
                juce::File(string(args, "path")), error, {}, options))
        {
            const auto sourcePitch = string(args, "source_pitch").toLowerCase();
            if (sourcePitch == "native" || sourcePitch == "reanalyze"
                || sourcePitch == "reanalyse" || sourcePitch == "game+fcpe"
                || sourcePitch == "game_fcpe")
            {
                juce::String pitchError;
                (void) AnalysisService::reanalyseProjectSourcePitch(
                    imported->project, analysisConfig(args), pitchError);
            }
            juce::StringArray annotationWarnings;
            SampleSettings::convertMelodyneProject(imported->project,
                                                   annotationWarnings);
            project.resetDocument(std::move(imported->project));
            return toolResult(summary(project.snapshot())
                + (annotationWarnings.isEmpty() ? juce::String()
                    : "; annotation_warnings="
                        + annotationWarnings.joinIntoString(" | ")));
        }
    }
    else if (name == "set_tempo")
    {
        project.setTempo(number(args, "bpm", 120.0), static_cast<int>(number(args, "numerator", 4.0)),
                         static_cast<int>(number(args, "denominator", 4.0)));
        return toolResult("ok");
    }
    else if (name == "add_track")
    {
        const auto compose = static_cast<bool>(args.getProperty("compose", true));
        const auto id = project.addTrack(string(args, "name"), compose, false,
            static_cast<bool>(args.getProperty("accompaniment", false)));
        return toolResult("track_id=" + id);
    }
    else if (name == "set_track")
    {
        const auto id = string(args, "track_id");
        for (const auto& track : project.snapshot().tracks)
            if (track.id == id && track.accompaniment
                && ((args.hasProperty("compose") && static_cast<bool>(args["compose"]))
                    || args.hasProperty("pitch_algorithm") || args.hasProperty("stretch_algorithm")
                    || args.hasProperty("render_order") || args.hasProperty("merge_contiguous_source") || args.hasProperty("voicebank_directory")
                    || args.hasProperty("utau_global_flags") || args.hasProperty("output_engine")
                    || args.hasProperty("output_resampler") || args.hasProperty("output_wavtool")))
                return toolResult("Accompaniment tracks play original audio and cannot enable tuning engines", true);
        if (args.hasProperty("pitch_algorithm"))
        {
            const auto chosen = string(args, "pitch_algorithm").toLowerCase();
            if (chosen == "mld5" || chosen == "mld3")
                return toolResult(chosen + " is disabled (incomplete algorithm)", true);
            if (chosen == "vslib")
                return toolResult("vslib native implementation is not integrated", true);
            if (chosen != "nsf-hifigan" && chosen != "world" && chosen != "llsm2"
                && chosen != "utau" && chosen != "utau4" && chosen != "utaumou")
                return toolResult("Unknown pitch algorithm: " + chosen, true);
        }
        if (args.hasProperty("name")) project.setTrackName(id, string(args, "name"));
        if (args.hasProperty("utau_global_flags"))
            project.setTrackUtauGlobalFlags(id, string(args, "utau_global_flags"));
        if (args.hasProperty("voicebank_directory"))
            project.setTrackVoicebankDirectory(id,
                juce::File(string(args, "voicebank_directory")));
        if (args.hasProperty("compose")) project.setTrackCompose(id, static_cast<bool>(args["compose"]));
        if (args.hasProperty("muted")) project.setTrackMuted(id, static_cast<bool>(args["muted"]));
        if (args.hasProperty("solo")) project.setTrackSolo(id, static_cast<bool>(args["solo"]));
        if (args.hasProperty("volume")) project.setTrackVolume(id, static_cast<float>(number(args, "volume", 1.0)));
        if (args.hasProperty("pan")) project.setTrackPan(id, static_cast<float>(number(args, "pan")));
        if (args.hasProperty("smooth_overlaps"))
            project.setTrackSmoothOverlaps(id, static_cast<bool>(args["smooth_overlaps"]));
        if (args.hasProperty("nsf_smooth_pitch_transitions"))
            project.setTrackNsfSmoothPitchTransitions(id, static_cast<bool>(args["nsf_smooth_pitch_transitions"]));
        if (args.hasProperty("nsf_noise_protection"))
            project.setTrackNsfNoiseProtection(id, static_cast<bool>(args["nsf_noise_protection"]));
        if (args.hasProperty("normalize_volume"))
            project.setTrackNormalizeVolume(id, static_cast<bool>(args["normalize_volume"]));
        if (args.hasProperty("pitch_algorithm"))
        {
            const auto value = string(args, "pitch_algorithm").toLowerCase();
            project.setTrackPitchAlgorithm(id,
                value == "nsf-hifigan" ? PitchAlgorithm::nsfHifigan
                : value == "world" ? PitchAlgorithm::world
                : value == "vslib" ? PitchAlgorithm::vocalShifter
                : value == "mld3" ? PitchAlgorithm::mld3
                : value == "llsm2" ? PitchAlgorithm::llsm2
                : value.startsWith("utau") ? PitchAlgorithm::utau
                : PitchAlgorithm::mld5);
            project.setTrackUtauMode(id, parseUtauMode(value));
        }
        if (args.hasProperty("output_engine") || args.hasProperty("output_resampler") || args.hasProperty("output_wavtool"))
        {
            for (const auto& track : project.snapshot().tracks) if (track.id == id) {
                if (!trackUsesVoicebankSynthesis(track) || trackIsDiffSinger(track))
                    return toolResult("Output engine selection requires a non-DiffSinger UTAU track", true);
                const auto key = args.hasProperty("output_engine") ? string(args, "output_engine") : utauOutputEngineKey(track.outputEngine);
                if (key != "inherit" && key != "resampler" && key != "pc-nsf-hifigan")
                    return toolResult("Invalid output_engine", true);
                project.setTrackOutputEngine(id, parseUtauOutputEngine(key),
                    key == "inherit" ? juce::File{} : args.hasProperty("output_resampler") ? juce::File(string(args, "output_resampler")) : track.outputResampler,
                    key == "inherit" ? juce::File{} : args.hasProperty("output_wavtool") ? juce::File(string(args, "output_wavtool")) : track.outputWavtool);
                break;
            }
        }
        if (args.hasProperty("stretch_algorithm"))
        {
            const auto value = string(args, "stretch_algorithm").toLowerCase();
            project.setTrackStretchAlgorithm(id,
                value == "variable-mel-hop" ? StretchAlgorithm::variableMelHop
                : value == "loop" ? StretchAlgorithm::loop
                : value == "soundtouch" ? StretchAlgorithm::soundTouch
                : value == "wcsndm-nonloop" ? StretchAlgorithm::wcsNonLoop
                : value == "hifishifter-mel" ? StretchAlgorithm::hifiShifterMel
                : value == "nsf-shift-then-splice" ? StretchAlgorithm::nsfShiftThenSplice
                : StretchAlgorithm::melodyneHybrid);
        }
        if (args.hasProperty("render_order") || args.hasProperty("merge_contiguous_source"))
        {
            auto order = RenderOrder::processThenSplice;
            for (const auto& track : project.snapshot().tracks) if (track.id == id) order = track.renderOrder;
            if (args.hasProperty("render_order"))
                order = string(args, "render_order").toLowerCase() == "stretch-splice-then-pitch"
                    ? RenderOrder::stretchSpliceThenPitch : RenderOrder::processThenSplice;
            project.setTrackRenderOrder(id, order, args.hasProperty("merge_contiguous_source")
                ? std::optional<bool>(static_cast<bool>(args["merge_contiguous_source"])) : std::nullopt);
        }
        return toolResult("ok");
    }
    else if (name == "set_clip")
    {
        const auto id = string(args, "clip_id");
        if (args.hasProperty("gain"))
            project.setClipGain(id, static_cast<float>(number(args, "gain", 1.0)));
        if (args.hasProperty("fade_in_seconds") || args.hasProperty("fade_out_seconds"))
        {
            auto fadeIn = number(args, "fade_in_seconds");
            auto fadeOut = number(args, "fade_out_seconds");
            const auto data = project.snapshot();
            for (const auto& track : data.tracks)
                for (const auto& clip : track.clips)
                    if (clip.id == id)
                    {
                        if (!args.hasProperty("fade_in_seconds")) fadeIn = clip.fadeInSeconds;
                        if (!args.hasProperty("fade_out_seconds")) fadeOut = clip.fadeOutSeconds;
                    }
            project.setClipFades(id, fadeIn, fadeOut);
        }
        if (args.hasProperty("muted"))
            project.setClipMuted(id, static_cast<bool>(args["muted"]));
        return toolResult("ok");
    }
    else if (name == "move_clip")
    {
        project.moveClip(string(args, "clip_id"), number(args, "start_seconds"));
        return toolResult("ok");
    }
    else if (name == "resize_clip")
    {
        project.resizeClip(string(args, "clip_id"), number(args, "start_seconds"),
                           number(args, "duration_seconds", 0.25));
        return toolResult("ok");
    }
    else if (name == "duplicate_clip")
    {
        const auto id = project.duplicateClip(string(args, "clip_id"),
            args.hasProperty("start_seconds") ? number(args, "start_seconds") : -1.0,
            string(args, "track_id"));
        return id.isNotEmpty() ? toolResult("clip_id=" + id)
                               : toolResult("Clip not found", true);
    }
    else if (name == "transpose_note")
    {
        project.transposeNote(string(args, "note_id"), static_cast<float>(number(args, "semitones")));
        return toolResult("ok");
    }
    else if (name == "edit_notes_pitch")
    {
        auto ids = strings(args, "note_ids");
        if (ids.empty())
            if (const auto id = string(args, "note_id"); id.isNotEmpty()) ids.push_back(id);
        const auto action = string(args, "action").toLowerCase();
        if (action == "set")
            project.setNotesMidi(ids, static_cast<float>(number(args, "midi", 60.0)));
        else if (action == "average")
            project.averageNotesMidi(ids);
        else if (action == "quantize")
            project.quantizeNotesMidi(ids,
                static_cast<float>(number(args, "step_semitones", 1.0)));
        else
            project.transposeNotes(ids,
                static_cast<float>(number(args, "cents") / 100.0));
        return toolResult("ok");
    }
    else if (name == "duplicate_notes")
    {
        auto ids = strings(args, "note_ids");
        if (ids.empty())
            if (const auto id = string(args, "note_id"); id.isNotEmpty()) ids.push_back(id);
        const auto inserted = project.duplicateNotes(ids, string(args, "clip_id"),
            number(args, "start_seconds"));
        juce::StringArray insertedText;
        for (const auto& id : inserted) insertedText.add(id);
        return !inserted.empty() ? toolResult("note_ids=" + insertedText.joinIntoString(","))
                                 : toolResult("Notes or target clip not found", true);
    }
    else if (name == "resize_note")
    {
        project.resizeNote(string(args, "note_id"), number(args, "start_seconds"),
                           number(args, "duration_seconds", 0.25));
        return toolResult("ok");
    }
    else if (name == "set_note")
    {
        const auto id = string(args, "note_id");
        const bool editsTail=args.hasProperty("tail_fade")||args.hasProperty("tail_fade_start_percent")||args.hasProperty("tail_fade_end_percent")||args.hasProperty("tail_fade_start_gain_percent")||args.hasProperty("tail_fade_end_gain_percent")||args.hasProperty("tail_fade_curve_power")||args.hasProperty("tail_fade_custom_curve")||args.hasProperty("tail_fade_control1_time_percent")||args.hasProperty("tail_fade_control1_progress_percent")||args.hasProperty("tail_fade_control2_time_percent")||args.hasProperty("tail_fade_control2_progress_percent")||args.hasProperty("head_envelope_start_percent")||args.hasProperty("head_envelope_end_percent")||args.hasProperty("head_envelope_start_gain_percent")||args.hasProperty("head_envelope_end_gain_percent")||args.hasProperty("head_envelope_curve_power")||args.hasProperty("head_envelope_control1_time_percent")||args.hasProperty("head_envelope_control1_progress_percent")||args.hasProperty("head_envelope_control2_time_percent")||args.hasProperty("head_envelope_control2_progress_percent")||args.hasProperty("head_envelope")||args.hasProperty("head_envelope_custom_curve");
        backend::TailFadeSettings tailSettings;int tailMode=0;
        if(editsTail)
        {
            bool eligible=false;const auto data=project.snapshot();
            for(const auto& t:data.tracks)for(const auto& c:t.clips)for(const auto& n:c.notes)
                if(n.id==id&&t.pitchAlgorithm==PitchAlgorithm::utau&&!trackIsDiffSinger(t))
                {eligible=true;tailSettings=n.utauTailFade;tailMode=n.utauTailFadeMode;}
            if(!eligible)return toolResult("tail_fade requires a non-DS UTAU note with OTO regions",true);
            if(args.hasProperty("tail_fade"))
            {
                const auto mode=string(args,"tail_fade");
                if(mode!="off"&&mode!="linear"&&mode!="smooth")return toolResult("tail_fade must be off, linear or smooth",true);
                tailMode=mode=="off"?0:mode=="linear"?1:2;
            }
            if(args.hasProperty("tail_fade_start_percent"))tailSettings.startFraction=number(args,"tail_fade_start_percent")/100.0;
            if(args.hasProperty("tail_fade_end_percent"))tailSettings.endFraction=number(args,"tail_fade_end_percent")/100.0;
            if(args.hasProperty("tail_fade_start_gain_percent"))tailSettings.startGain=number(args,"tail_fade_start_gain_percent")/100.0;
            if(args.hasProperty("tail_fade_end_gain_percent"))tailSettings.endGain=number(args,"tail_fade_end_gain_percent")/100.0;
            if(args.hasProperty("tail_fade_curve_power"))tailSettings.curvePower=number(args,"tail_fade_curve_power")/1.0;
            if(args.hasProperty("tail_fade_custom_curve"))tailSettings.customCurve=(bool)args.getProperty("tail_fade_custom_curve",false);
            if(args.hasProperty("tail_fade_control1_time_percent"))tailSettings.control1Time=number(args,"tail_fade_control1_time_percent")/100.0;
            if(args.hasProperty("tail_fade_control1_progress_percent"))tailSettings.control1Progress=number(args,"tail_fade_control1_progress_percent")/100.0;
            if(args.hasProperty("tail_fade_control2_time_percent"))tailSettings.control2Time=number(args,"tail_fade_control2_time_percent")/100.0;
            if(args.hasProperty("tail_fade_control2_progress_percent"))tailSettings.control2Progress=number(args,"tail_fade_control2_progress_percent")/100.0;
            if(args.hasProperty("head_envelope"))
            {
                const auto mode=string(args,"head_envelope");if(mode!="off"&&mode!="linear"&&mode!="curve")return toolResult("head_envelope must be off, linear or curve",true);
                tailSettings.head.mode=mode=="off"?0:mode=="linear"?1:2;
            }
            if(args.hasProperty("head_envelope_custom_curve"))tailSettings.head.customCurve=(bool)args.getProperty("head_envelope_custom_curve",false);
            if(args.hasProperty("head_envelope_start_percent"))tailSettings.head.startFraction=number(args,"head_envelope_start_percent")/100.0;
            if(args.hasProperty("head_envelope_end_percent"))tailSettings.head.endFraction=number(args,"head_envelope_end_percent")/100.0;
            if(args.hasProperty("head_envelope_start_gain_percent"))tailSettings.head.startGain=number(args,"head_envelope_start_gain_percent")/100.0;
            if(args.hasProperty("head_envelope_end_gain_percent"))tailSettings.head.endGain=number(args,"head_envelope_end_gain_percent")/100.0;
            if(args.hasProperty("head_envelope_curve_power"))tailSettings.head.curvePower=number(args,"head_envelope_curve_power")/1.0;
            if(args.hasProperty("head_envelope_control1_time_percent"))tailSettings.head.control1Time=number(args,"head_envelope_control1_time_percent")/100.0;
            if(args.hasProperty("head_envelope_control1_progress_percent"))tailSettings.head.control1Progress=number(args,"head_envelope_control1_progress_percent")/100.0;
            if(args.hasProperty("head_envelope_control2_time_percent"))tailSettings.head.control2Time=number(args,"head_envelope_control2_time_percent")/100.0;
            if(args.hasProperty("head_envelope_control2_progress_percent"))tailSettings.head.control2Progress=number(args,"head_envelope_control2_progress_percent")/100.0;
            if(!tailSettings.valid())return toolResult("Invalid tail fade settings: range must be ordered within 0..100%, gains 0..200% and decreasing, power 0.25..4, Bezier coordinates ordered within 0..100%",true);
        }
        if (args.hasProperty("label")) project.setNoteLabel(id, string(args, "label"));
        if (args.hasProperty("jie_split"))
        {
            // Three cumulative fractions of the note, or null to hand the note
            // back to the engine's own allocation.
            const auto value = args["jie_split"];
            if (const auto* fractions = value.getArray(); fractions != nullptr
                && fractions->size() == 3)
                project.setNotesUtauJieSplit({ id },
                    static_cast<double>((*fractions)[0]),
                    static_cast<double>((*fractions)[1]),
                    static_cast<double>((*fractions)[2]));
            else
                project.clearNotesUtauJieSplit({ id });
        }
        if (args.hasProperty("region_flags") || args.hasProperty("flag_split"))
        {
            // The four-region flags, as ["f1","f2","f3","f4"]; an empty entry
            // means that region keeps the note's own flags.
            juce::String parts[4];
            if (const auto* given = args["region_flags"].getArray(); given != nullptr)
                for (int index = 0; index < 4 && index < given->size(); ++index)
                    parts[index] = (*given)[index].toString();
            project.setNotesRegionFlags({ id },
                static_cast<bool>(args.getProperty("flag_split", true)),
                parts[0], parts[1], parts[2], parts[3]);
        }
        if (args.hasProperty("flag_curve_enabled"))
            project.setNotesUtauFlagCurveEnabled({ id },
                static_cast<bool>(args["flag_curve_enabled"]));
        if (args.hasProperty("flag_curve"))
        {
            // {"flag": "Mt", "points": [[seconds, value, shape?], ...]}; an
            // empty list drops that flag's curve and leaves the others.
            const auto flag = args["flag_curve"].getProperty("flag", "g").toString();
            std::vector<FlagCurvePoint> points;
            if (const auto* given = args["flag_curve"].getProperty("points", {})
                                        .getArray(); given != nullptr)
                for (const auto& entry : *given)
                    if (const auto* pair = entry.getArray(); pair != nullptr
                        && pair->size() >= 2)
                    {
                        FlagCurvePoint point;
                        point.timeSeconds = static_cast<double>((*pair)[0]);
                        point.value = static_cast<float>(
                            static_cast<double>((*pair)[1]));
                        if (pair->size() >= 3)
                            point.shape = parsePitchCurveShape((*pair)[2].toString());
                        points.push_back(point);
                    }
            project.setNoteUtauFlagCurve(id, flag, std::move(points));
        }
        if (args.hasProperty("flag_curve_g"))
        {
            // [[seconds, value], ...] against the note start, or an empty list
            // to drop the curve.  Seconds may be negative to reach into the
            // preutterance, exactly like an amplitude envelope point.
            std::vector<FlagCurvePoint> points;
            if (const auto* given = args["flag_curve_g"].getArray(); given != nullptr)
                for (const auto& entry : *given)
                    if (const auto* pair = entry.getArray(); pair != nullptr
                        && pair->size() >= 2)
                    {
                        FlagCurvePoint point;
                        point.timeSeconds = static_cast<double>((*pair)[0]);
                        point.value = static_cast<float>(
                            static_cast<double>((*pair)[1]));
                        // An optional third entry names the shape of the segment
                        // arriving here: linear (the default), smooth, ease-in,
                        // ease-out or custom-bezier.
                        if (pair->size() >= 3)
                            point.shape = parsePitchCurveShape((*pair)[2].toString());
                        points.push_back(point);
                    }
            project.setNoteUtauFlagCurve(id, "g", std::move(points));
        }
        if (args.hasProperty("utau_consonant_velocity"))
            project.setNotesUtauConsonantVelocity({ id },
                static_cast<int>(number(args, "utau_consonant_velocity", 100)));
        if (args.hasProperty("amplitude_envelope"))
        {
            // Pairs of [seconds, dB], relative to the note start.
            std::vector<AmplitudeEnvelopePoint> points;
            if (const auto* rows = args["amplitude_envelope"].getArray())
                for (const auto& row : *rows)
                    if (const auto* pair = row.getArray(); pair != nullptr
                        && pair->size() == 2)
                        points.push_back({ static_cast<double>((*pair)[0]),
                                           static_cast<float>((*pair)[1]) });
            project.setNoteAmplitudeEnvelope(id, std::move(points));
        }
        if (args.hasProperty("utau_splice"))
            project.setNotesUtauSplice({ id }, static_cast<bool>(args["utau_splice"]));
        if (args.hasProperty("utau_flags"))
            project.setNoteUtauFlags(id, string(args, "utau_flags"));
        if (args.hasProperty("modulation"))
            project.setNoteModulation(id, static_cast<float>(number(args, "modulation", 1.0)));
        if (args.hasProperty("drift"))
            project.setNoteDrift(id, static_cast<float>(number(args, "drift", 1.0)));
        if (args.hasProperty("tension"))
            project.setNoteTension(id, static_cast<float>(number(args, "tension")));
        if (args.hasProperty("breath"))
            project.setNoteBreath(id, static_cast<float>(number(args, "breath")));
        if (args.hasProperty("formant_semitones"))
            project.setNoteFormant(id, static_cast<float>(number(args, "formant_semitones")));
        if (args.hasProperty("gain"))
            project.setNoteGain(id, static_cast<float>(number(args, "gain", 1.0)));
        if(editsTail)project.setNotesTailFade({id},tailMode,tailSettings);
        if (args.hasProperty("amplitude_envelope_base"))
            project.setNotesAmplitudeEnvelopeBase({ id },
                static_cast<float>(number(args, "amplitude_envelope_base", 100.0)));
        if (args.hasProperty("robust_pitch_curve"))
            project.setNoteRobustPitchCurve(id,
                static_cast<bool>(args["robust_pitch_curve"]));
        if (args.hasProperty("consonant_seconds") || args.hasProperty("attack_speed"))
        {
            auto consonant = number(args, "consonant_seconds", 0.04);
            auto speed = static_cast<float>(number(args, "attack_speed", 1.0));
            const auto data = project.snapshot();
            for (const auto& track : data.tracks)
                for (const auto& clip : track.clips)
                    for (const auto& note : clip.notes)
                        if (note.id == id)
                        {
                            if (!args.hasProperty("consonant_seconds")) consonant = note.consonantSeconds;
                            if (!args.hasProperty("attack_speed")) speed = note.attackSpeed;
                        }
            project.setNoteAttack(id, consonant, speed);
        }
        return toolResult("ok");
    }
    else if (name == "set_pitch_curve")
    {
        std::vector<PitchCurveEditPoint> points;
        const auto pointValues = args.getProperty("points", {});
        if (const auto* values = pointValues.getArray())
            for (const auto& value : *values)
                points.push_back({ number(value, "time_seconds"),
                                   static_cast<float>(number(value, "midi", 60.0)) });
        if (project.setNotePitchCurve(string(args, "note_id"), std::move(points)))
            return toolResult("ok");
        error = "Pitch curve needs a valid note_id and at least one point";
    }
    else if (name == "add_note")
    {
        const auto id = project.addNote(string(args, "clip_id"), number(args, "start_seconds"),
            number(args, "duration_seconds", 0.25), static_cast<float>(number(args, "midi", 60.0)));
        return id.isNotEmpty() ? toolResult("note_id=" + id)
                               : toolResult("No compose clip accepts the note", true);
    }
    else if (name == "remove_note")
    {
        project.removeNote(string(args, "note_id"));
        return toolResult("ok");
    }
    else if (name == "toggle_note_connection")
    {
        project.toggleNoteConnection(string(args, "note_id"));
        return toolResult("ok");
    }
    else if (name == "set_notes_portamento")
    {
        const auto ids=strings(args,"note_ids");
        if(ids.size()<2)return toolResult("At least two consecutive note IDs are required",true);
        const auto enabled=static_cast<bool>(args.getProperty("enabled",true));
        const auto changed=project.setNotesPortamento(ids,enabled,error,number(args,"duration_seconds",.1));
        if(error.isNotEmpty())return toolResult(error,true);
        return toolResult(changed?"changed=1":"changed=0");
    }
    else if (name == "remove_clip")
    {
        project.removeClip(string(args, "clip_id"));
        return toolResult("ok");
    }
    else if (name == "remove_track")
    {
        project.removeTrack(string(args, "track_id"));
        return toolResult("ok");
    }
    else if (name == "undo")
    {
        const auto ok = project.undo();
        return toolResult(ok ? "ok" : "history_empty", !ok);
    }
    else if (name == "redo")
    {
        const auto ok = project.redo();
        return toolResult(ok ? "ok" : "history_empty", !ok);
    }
    else if (name == "set_utau_resampler")
    {
        const juce::File executable(string(args, "path"));
        if (!executable.existsAsFile())
            return toolResult("Resampler executable not found", true);
        audio->setUtauResamplerFile(executable);
        audioPrepared = false;
        return toolResult("ok");
    }
    else if (name == "utau_render_selection")
    {
        std::vector<juce::String> ids;
        if (!resolveUtauRenderSelection(project.snapshot(), args, {}, ids, error))
            return toolResult(error, true);
        audio->setUtauRenderNoteSelection(ids);
        audioPrepared = false;
        return toolResult("note_ids=" + juce::String(static_cast<int>(ids.size())));
    }
    else if (name == "render_prepare")
    {
        syncAudio();
        if (static_cast<bool>(args.getProperty("wait", false)))
        {
            const auto timeout = juce::jlimit(0.1, 3600.0, number(args, "timeout_seconds", 300.0));
            if (!waitForRender(timeout, error)) return toolResult(error, true);
        }
        return toolResult(juce::JSON::toString(transportStatusJson(), false));
    }
    else if (name == "render_status" || name == "transport_status")
    {
        return toolResult(juce::JSON::toString(transportStatusJson(), false));
    }
    else if (name == "export_wav")
    {
        // Export renders the whole song, as it does from the window: rendering
        // is selection-driven, so exporting what happened to be selected would
        // write one phrase and silence everywhere else.
        audio->selectEveryUtauNote(project.snapshot());
        audioPrepared = false;
        syncAudio(true);
        const auto timeout = juce::jlimit(0.1, 3600.0, number(args, "timeout_seconds", 300.0));
        if (!waitForRender(timeout, error)) return toolResult(error, true);
        const juce::File file(string(args, "path"));
        if (file.getFullPathName().isEmpty()) return toolResult("WAV output path is empty", true);
        // A track id exports that track alone, for one file per track; a
        // range writes only that stretch, as "export the last render" does.
        WavExportOptions options;
        options.channels = static_cast<int>(number(args, "channels", 2));
        options.sampleRate = static_cast<int>(number(args, "sample_rate", 0));
        options.bitDepth = static_cast<int>(number(args, "bit_depth", 24));
        if (!options.isValid()) return toolResult("Invalid WAV format", true);
        if (audio->exportWav(file, error, string(args, "track_id"),
                             number(args, "from_seconds", 0.0),
                             number(args, "to_seconds", 0.0), options))
        {
            auto value = object();
            set(value, "path", file.getFullPathName());
            set(value, "size", file.getSize());
            set(value, "backend", audio->activeRenderBackends());
            return toolResult(juce::JSON::toString(value, false));
        }
    }
    else if (name == "transport_play")
    {
        syncAudio();
        const auto timeout = juce::jlimit(0.1, 3600.0, number(args, "timeout_seconds", 300.0));
        if (!waitForRender(timeout, error)) return toolResult(error, true);
        if (args.hasProperty("position_seconds"))
            audio->setPosition(number(args, "position_seconds"));
        // Playing a selection stops where the selection ends; the window sets
        // this from the marquee, and it stays set after the transport stops.
        if (args.hasProperty("play_until_seconds"))
            audio->setPlayUntil(number(args, "play_until_seconds"));
        audio->play();
        return toolResult(juce::JSON::toString(transportStatusJson(), false));
    }
    else if (name == "transport_stop")
    {
        audio->stop();
        return toolResult(juce::JSON::toString(transportStatusJson(), false));
    }
    else if (name == "transport_seek")
    {
        audio->setPosition(number(args, "position_seconds"));
        return toolResult(juce::JSON::toString(transportStatusJson(), false));
    }
    else if (name == "sample_settings_read")
    {
        const juce::File file(string(args, "audio_path"));
        if (!file.existsAsFile()) return toolResult("Audio file not found", true);
        auto value = object();
        set(value, "audio_path", file.getFullPathName());
        set(value, "sidecar_path", SampleSettings::annotationFor(file).getFullPathName());
        set(value, "rows", sampleRowsJson(SampleSettings::loadOrDerive(file, project.snapshot())));
        return toolResult(juce::JSON::toString(value, false));
    }
    else if (name == "sample_settings_save")
    {
        const juce::File file(string(args, "audio_path"));
        if (!file.existsAsFile()) return toolResult("Audio file not found", true);
        const auto rows = sampleRowsFromJson(args.getProperty("rows", juce::var()));
        if (rows.empty()) return toolResult("rows must contain at least one valid region", true);
        if (SampleSettings::save(file, rows, error))
            return toolResult("saved=" + SampleSettings::annotationFor(file).getFullPathName()
                              + "; regions=" + juce::String(static_cast<juce::int64>(rows.size())));
    }
    else if (name == "oto_import")
    {
        const juce::File audioFile(string(args, "audio_path"));
        auto reader = createAudioReader(formats, audioFile);
        if (reader == nullptr || reader->sampleRate <= 0.0)
            return toolResult("Audio read failed", true);
        std::vector<SampleRegionSetting> rows;
        const auto duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        if (SampleSettings::importOto(juce::File(string(args, "oto_path")), audioFile,
                                      duration, rows, error))
        {
            if (static_cast<bool>(args.getProperty("save_sidecar", true))
                && !SampleSettings::save(audioFile, rows, error))
                return toolResult(error, true);
            auto value = object();
            set(value, "sidecar_path", SampleSettings::annotationFor(audioFile).getFullPathName());
            set(value, "rows", sampleRowsJson(rows));
            return toolResult(juce::JSON::toString(value, false));
        }
    }
    else if (name == "oto_export")
    {
        const juce::File audioFile(string(args, "audio_path"));
        auto reader = createAudioReader(formats, audioFile);
        if (reader == nullptr || reader->sampleRate <= 0.0)
            return toolResult("Audio read failed", true);
        const auto rows = SampleSettings::loadOrDerive(audioFile, project.snapshot());
        const auto duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
        const juce::File otoFile(string(args, "oto_path"));
        if (SampleSettings::exportOto(otoFile, audioFile, rows, duration, error))
            return toolResult("saved=" + otoFile.getFullPathName()
                              + "; regions=" + juce::String(static_cast<juce::int64>(rows.size())));
    }
    else if (name == "jie_oto_create")
    {
        const juce::File root(string(args, "voicebank_path"));
        auto written = 0;
        auto kept = 0;
        if (SampleSettings::createJieOto(root, written, kept, error))
            return toolResult("written=" + juce::String(written)
                              + "; kept=" + juce::String(kept));
    }
    else if (name == "voicebank_import")
    {
        juce::StringArray audioFiles;
        juce::StringArray warnings;
        auto sidecars = 0;
        auto regions = 0;
        if (SampleSettings::importVoicebank(juce::File(string(args, "path")), audioFiles,
                                            sidecars, regions, warnings))
        {
            auto value = object();
            std::vector<juce::var> files;
            for (const auto& file : audioFiles) files.emplace_back(file);
            std::vector<juce::var> warningValues;
            for (const auto& warning : warnings) warningValues.emplace_back(warning);
            set(value, "audio_files", array(std::move(files)));
            set(value, "sidecars_written", sidecars);
            set(value, "regions_written", regions);
            set(value, "warnings", array(std::move(warningValues)));
            return toolResult(juce::JSON::toString(value, false));
        }
        error = warnings.isEmpty() ? "Voicebank import failed" : warnings.joinIntoString("\n");
    }
    else if (name == "read_file")
    {
        const juce::File file(string(args, "path"));
        if (!pathWithinRoots(allowedRoots(), file))
            return toolResult(outsideRootsMessage(file, allowedRoots()), true);
        auto input = file.createInputStream();
        if (input != nullptr)
        {
            const auto offset = std::max<juce::int64>(0, static_cast<juce::int64>(number(args, "offset")));
            const auto requested = juce::jlimit(1, 4 * 1024 * 1024,
                                                 static_cast<int>(number(args, "max_bytes", 1024 * 1024)));
            input->setPosition(offset);
            juce::MemoryBlock bytes;
            input->readIntoMemoryBlock(bytes, requested);
            auto value = object();
            set(value, "path", file.getFullPathName());
            set(value, "offset", offset);
            set(value, "size", static_cast<juce::int64>(bytes.getSize()));
            set(value, "base64", juce::Base64::toBase64(bytes.getData(), bytes.getSize()));
            return toolResult(juce::JSON::toString(value, false));
        }
        error = "File read failed";
    }
    else if (name == "list_directory")
    {
        const juce::File directory(string(args, "path"));
        if (!pathWithinRoots(allowedRoots(), directory))
            return toolResult(outsideRootsMessage(directory, allowedRoots()), true);
        juce::Array<juce::File> entries;
        directory.findChildFiles(entries, juce::File::findFilesAndDirectories, false);
        std::vector<juce::var> values;
        for (const auto& entry : entries)
        {
            auto value = object();
            set(value, "name", entry.getFileName());
            set(value, "path", entry.getFullPathName());
            set(value, "directory", entry.isDirectory());
            set(value, "size", entry.isDirectory() ? juce::int64(0) : entry.getSize());
            values.push_back(std::move(value));
        }
        return toolResult(juce::JSON::toString(array(std::move(values)), false));
    }
    else error = "Unknown tool: " + name;
    return toolResult(error.isNotEmpty() ? error : "Operation failed", true);
}

juce::var McpServer::noteJson(const NoteData& note, bool includeCurves)
{
    auto noteValue = object();
    set(noteValue, "id", note.id);
    set(noteValue, "label", note.label);
    set(noteValue, "diffsinger_pronunciation", note.diffSingerPronunciation);
    set(noteValue, "diffsinger_timing", juce::JSON::parse(note.diffSingerTiming));
    if (includeCurves)
    {
        auto pitchPoints = [](const auto& points) {
            juce::Array<juce::var> values;
            for (const auto& p : points) values.add(juce::Array<juce::var>{p.timeSeconds, p.targetMidi});
            return juce::var(values);
        };
        set(noteValue, "pitch_control_points", pitchPoints(note.pitchControlPoints));
        set(noteValue, "diffsinger_pitch_reference", pitchPoints(note.diffSingerPitchReference));
        set(noteValue, "diffsinger_pitch_offset", pitchPoints(note.diffSingerPitchOffset));
    }
    set(noteValue, "native_unpitched", note.nativeUnpitched);
    set(noteValue, "native_role", nativeSegmentRoleName(note.nativeRole));
    set(noteValue, "pronunciation_kind", pronunciationKindName(note.pronunciationKind));
    set(noteValue, "simplified_consonant", note.simplifiedConsonant);
    set(noteValue, "vowel_splits_explicit", note.vowelSplitsExplicit);
    set(noteValue, "native_provenance", note.nativeProvenance);
    set(noteValue, "native_confidence", note.nativeConfidence);
    if (includeCurves)
    {
        std::vector<juce::var> nativeSegments;
        for (const auto& segment : note.nativeSegments)
        {
            auto segmentValue = object();
            set(segmentValue, "id", segment.id);
            set(segmentValue, "alias", segment.alias);
            set(segmentValue, "role", nativeSegmentRoleName(segment.role));
            set(segmentValue, "source_start_seconds", segment.sourceStartSeconds);
            set(segmentValue, "source_end_seconds", segment.sourceEndSeconds);
            set(segmentValue, "provenance", segment.provenance);
            set(segmentValue, "confidence", segment.confidence);
            set(segmentValue, "alignment_seconds", segment.alignmentSeconds);
            set(segmentValue, "overlap_seconds", segment.overlapSeconds);
            set(segmentValue, "stretchable", segment.stretchable);
            set(segmentValue, "stretch_weight", segment.stretchWeight);
            nativeSegments.push_back(std::move(segmentValue));
        }
        set(noteValue, "native_segments", array(std::move(nativeSegments)));
    }
    set(noteValue, "start_seconds", note.startSeconds);
    set(noteValue, "duration_seconds", note.durationSeconds);
    set(noteValue, "consonant_seconds", note.consonantSeconds);
    set(noteValue, "midi", note.midiNote);
    set(noteValue, "source_midi_center", note.sourceMidiCenter);
    set(noteValue, "modulation", note.modulation);
    set(noteValue, "flattened", note.modulation <= 1.0e-4f);
    set(noteValue, "drift", note.drift);
    set(noteValue, "tension", note.tension);
    set(noteValue, "breath", note.breath);
    set(noteValue, "formant_semitones", note.formantSemitones);
    set(noteValue, "gain", note.gain);
    set(noteValue,"tail_fade",note.utauTailFadeMode==1?"linear":note.utauTailFadeMode==2?"smooth":"off");
    set(noteValue,"tail_fade_start_percent",note.utauTailFade.startFraction*100.0);
    set(noteValue,"tail_fade_end_percent",note.utauTailFade.endFraction*100.0);
    set(noteValue,"tail_fade_start_gain_percent",note.utauTailFade.startGain*100.0);
    set(noteValue,"tail_fade_end_gain_percent",note.utauTailFade.endGain*100.0);
    set(noteValue,"tail_fade_curve_power",note.utauTailFade.curvePower*1.0);
    set(noteValue,"tail_fade_custom_curve",note.utauTailFade.customCurve);
    set(noteValue,"head_envelope",note.utauTailFade.head.mode==0?"off":note.utauTailFade.head.mode==1?"linear":"curve");
    set(noteValue,"head_envelope_custom_curve",note.utauTailFade.head.customCurve);
    set(noteValue,"head_envelope_start_percent",note.utauTailFade.head.startFraction*100.0);
    set(noteValue,"head_envelope_end_percent",note.utauTailFade.head.endFraction*100.0);
    set(noteValue,"head_envelope_start_gain_percent",note.utauTailFade.head.startGain*100.0);
    set(noteValue,"head_envelope_end_gain_percent",note.utauTailFade.head.endGain*100.0);
    set(noteValue,"head_envelope_curve_power",note.utauTailFade.head.curvePower*1.0);
    set(noteValue,"head_envelope_control1_time_percent",note.utauTailFade.head.control1Time*100.0);
    set(noteValue,"head_envelope_control1_progress_percent",note.utauTailFade.head.control1Progress*100.0);
    set(noteValue,"head_envelope_control2_time_percent",note.utauTailFade.head.control2Time*100.0);
    set(noteValue,"head_envelope_control2_progress_percent",note.utauTailFade.head.control2Progress*100.0);
    set(noteValue,"tail_fade_control1_time_percent",note.utauTailFade.control1Time*100.0);
    set(noteValue,"tail_fade_control1_progress_percent",note.utauTailFade.control1Progress*100.0);
    set(noteValue,"tail_fade_control2_time_percent",note.utauTailFade.control2Time*100.0);
    set(noteValue,"tail_fade_control2_progress_percent",note.utauTailFade.control2Progress*100.0);
    set(noteValue, "attack_speed", note.attackSpeed);
    set(noteValue, "robust_pitch_curve", note.robustPitchCurve);
    set(noteValue, "connected_previous", note.connectedToPrevious);
    set(noteValue, "connected_next", note.connectedToNext);
    // The flag state, all of it: the plain text, the four-region
    // split, and the per-frame curve.  These are kept side by side
    // rather than one replacing another, so a caller has to be able
    // to see that nothing was lost when the switch was thrown.
    set(noteValue, "utau_flags", note.utauFlags);
    set(noteValue, "flag_split", note.utauFlagSplit);
    std::vector<juce::var> regionFlags;
    for (const auto& text : { note.utauRegionFlags1, note.utauRegionFlags2,
                              note.utauRegionFlags3, note.utauRegionFlags4 })
        regionFlags.emplace_back(text);
    set(noteValue, "region_flags", array(std::move(regionFlags)));
    set(noteValue, "flag_curve_enabled", note.utauFlagCurveEnabled);
    if (includeCurves)
    {
        auto flagCurves = object();
        for (const auto& curve : note.utauFlagCurves)
        {
            std::vector<juce::var> written;
            for (const auto& point : curve.points)
            {
                std::vector<juce::var> pair;
                pair.emplace_back(point.timeSeconds);
                pair.emplace_back(static_cast<double>(point.value));
                pair.emplace_back(pitchCurveShapeName(point.shape));
                written.push_back(array(std::move(pair)));
            }
            set(flagCurves, curve.flag.toRawUTF8(), array(std::move(written)));
        }
        set(noteValue, "flag_curves", flagCurves);
        std::vector<juce::var> contour;
        for (const auto& point : note.contour)
        {
            auto pointValue = object();
            set(pointValue, "time_seconds", point.timeSeconds);
            set(pointValue, "relative_cents", point.relativeCents);
            set(pointValue, "without_vibrato_cents", point.withoutVibratoCents);
            set(pointValue, "rendered_target_cents", renderedPitchCents(note, point));
            set(pointValue, "voiced", point.voiced);
            set(pointValue, "manual_target_cents", point.manualTargetCents);
            set(pointValue, "has_manual_target", point.hasManualTarget);
            contour.push_back(std::move(pointValue));
        }
        set(noteValue, "contour", array(std::move(contour)));
    }
    std::vector<juce::var> markers;
    for (const auto marker : note.sibilantMarkers) markers.emplace_back(marker);
    set(noteValue, "sibilant_markers", array(std::move(markers)));
    return noteValue;
}

juce::var McpServer::projectJson() const
{
    const auto data = project.snapshot();
    auto root = object();
    set(root, "name", data.name);
    set(root, "bpm", data.bpm);
    set(root, "beat_origin_seconds", data.beatOriginSeconds);
    set(root, "numerator", data.numerator);
    set(root, "denominator", data.denominator);
    set(root, "grid", data.gridDivision);
    set(root, "base_scale", data.baseScale);
    set(root, "hamood", juce::JSON::parse(data.hamoodState));
    std::vector<juce::var> nativeConnections;
    for (const auto& connection : data.nativeConnections)
    {
        auto value = object();
        set(value, "id", connection.id);
        set(value, "left_note_id", connection.leftNoteId);
        set(value, "right_note_id", connection.rightNoteId);
        set(value, "type", connection.type);
        set(value, "boundary_seconds", connection.boundarySeconds);
        set(value, "pitch_endpoint_relative",connection.pitchCurveEndpointRelative);
        set(value, "pitch_start_offset_seconds",connection.pitchStartOffsetSeconds);
        set(value, "pitch_end_offset_seconds",connection.pitchEndOffsetSeconds);
        std::vector<juce::var> controls;
        for(const auto& point:connection.pitchCurve)
        {
            auto control=object();set(control,"time",point.timeSeconds);set(control,"midi_or_residual",point.targetMidi);
            set(control,"shape",pitchCurveShapeName(point.shape));controls.push_back(std::move(control));
        }
        set(value,"pitch_controls",array(std::move(controls)));
        nativeConnections.push_back(std::move(value));
    }
    set(root, "native_connections", array(std::move(nativeConnections)));
    std::vector<juce::var> tracks;
    for (const auto& track : data.tracks)
    {
        auto trackValue = object();
        set(trackValue, "id", track.id);
        set(trackValue, "name", track.name);
        set(trackValue, "voicebank_directory", track.voicebankDirectory.getFullPathName());
        set(trackValue, "output_engine", utauOutputEngineKey(track.outputEngine));
        if (track.outputEngine == UtauOutputEngine::pcNsfHifigan) set(trackValue, "output_engine_name", juce::String::fromUTF8("HiFisampler（PC-NSF-HiFiGAN）"));
        set(trackValue, "output_resampler", track.outputResampler.getFullPathName());
        set(trackValue, "output_wavtool", track.outputWavtool.getFullPathName());
        set(trackValue, "diffsinger", trackIsDiffSinger(track));
        set(trackValue, "compose", track.compose && !track.accompaniment);
        set(trackValue, "accompaniment", track.accompaniment);
        set(trackValue, "muted", track.muted);
        set(trackValue, "solo", track.solo);
        set(trackValue, "volume", track.volume);
        set(trackValue, "pan", track.pan);
        set(trackValue, "smooth_overlaps", track.smoothOverlaps);
        set(trackValue, "normalize_volume", track.normalizeVolume);
        set(trackValue, "nsf_smooth_pitch_transitions", track.nsfSmoothPitchTransitions);
        set(trackValue, "nsf_noise_protection", track.nsfNoiseProtection);
        set(trackValue, "pitch_algorithm",
            track.accompaniment ? juce::String("none") : track.pitchAlgorithm == PitchAlgorithm::utau
                ? utauModeKey(track.utauMode)
                : pitchAlgorithmText(track.pitchAlgorithm));
        set(trackValue, "stretch_algorithm", track.accompaniment ? juce::String("none") : stretchAlgorithmText(track.stretchAlgorithm));
        set(trackValue, "render_order", renderOrderText(track.renderOrder));
        set(trackValue, "merge_contiguous_source", track.mergeContiguousSource);
        std::vector<juce::var> clips;
        for (const auto& clip : track.clips)
        {
            auto clipValue = object();
            set(clipValue, "id", clip.id);
            set(clipValue, "source_file", clip.sourceFile.getFullPathName());
            set(clipValue, "start_seconds", clip.startSeconds);
            set(clipValue, "source_offset_seconds", clip.sourceOffsetSeconds);
            set(clipValue, "source_duration_seconds", clip.sourceDurationSeconds);
            set(clipValue, "duration_seconds", clip.durationSeconds);
            set(clipValue, "audio_start_seconds", clip.audioStartSeconds);
            set(clipValue, "audio_duration_seconds", clip.audioLength());
            set(clipValue, "fade_in_seconds", clip.fadeInSeconds);
            set(clipValue, "fade_out_seconds", clip.fadeOutSeconds);
            set(clipValue, "crossfade_in_seconds", clip.crossfadeInSeconds);
            set(clipValue, "crossfade_out_seconds", clip.crossfadeOutSeconds);
            set(clipValue, "gain", clip.gain);
            set(clipValue, "muted", clip.muted);
            std::vector<juce::var> sourceTimeMap;
            sourceTimeMap.reserve(clip.sourceTimeMap.size());
            for (const auto& point : clip.sourceTimeMap)
            {
                auto pointValue = object();
                set(pointValue, "target_seconds", point.targetSeconds);
                set(pointValue, "source_seconds", point.sourceSeconds);
                sourceTimeMap.push_back(std::move(pointValue));
            }
            set(clipValue, "source_time_map", array(std::move(sourceTimeMap)));
            std::vector<juce::var> notes;
            for (const auto& note : clip.notes)
            {
                notes.push_back(noteJson(note));
            }
            set(clipValue, "notes", array(std::move(notes)));
            clips.push_back(std::move(clipValue));
        }
        set(trackValue, "clips", array(std::move(clips)));
        tracks.push_back(std::move(trackValue));
    }
    set(root, "revision", static_cast<juce::int64>(project.revisionNumber()));
    set(root, "tracks", array(std::move(tracks)));
    return root;
}

juce::int64 McpServer::currentProjectFingerprint() const
{
    // Hash what the project serialiser writes, not the JSON view.  That view
    // is a hand-maintained projection and never carried Flags, vibrato, the
    // UTAU timing overrides or the region splits, so editing any of them left
    // the fingerprint unchanged and syncAudio() skipped the re-render -- the
    // engine then replayed the previous audio for an edit it never saw.
    return project.contentFingerprint();
}

void McpServer::syncAudio(bool diffSingerExport)
{
    const auto fingerprint = currentProjectFingerprint();
    const auto ds = juce::JSON::toString(DiffSingerRenderer::inferenceOptions(diffSingerExport));
    if (audioPrepared && fingerprint == preparedFingerprint && ds == preparedDiffSingerOptions) return;
    audio->stop();
    audio->syncProject(project.snapshot(), diffSingerExport);
    preparedDiffSingerOptions = ds;
    preparedFingerprint = fingerprint;
    audioPrepared = true;
}

bool McpServer::waitForRender(double timeoutSeconds, juce::String& error)
{
    const auto started = juce::Time::getMillisecondCounterHiRes();
    while (audio->renderProgress().has_value())
    {
        if ((juce::Time::getMillisecondCounterHiRes() - started) * 0.001 >= timeoutSeconds)
        {
            error = "Pre-render timed out after " + juce::String(timeoutSeconds, 1) + " seconds";
            return false;
        }
        juce::Thread::sleep(10);
    }
    error = audio->activeRenderWarnings();
    return error.isEmpty();
}

juce::var McpServer::transportStatusJson() const
{
    auto value = object();
    const auto fingerprint = currentProjectFingerprint();
    const auto prepared = audioPrepared && fingerprint == preparedFingerprint;
    const auto progress = prepared ? audio->renderProgress() : std::optional<double>();
    set(value, "prepared", prepared);
    set(value, "merged_phrase_count", audio->diagnosticMergedPhraseCount());
    set(value, "rendering", progress.has_value());
    set(value, "render_progress", progress.has_value() ? *progress : (prepared ? 1.0 : 0.0));
    set(value, "backend", prepared ? audio->activeRenderBackends() : juce::String());
    set(value, "playing", audio->isPlaying());
    set(value, "position_seconds", audio->position());
    return value;
}

juce::var McpServer::toolResult(const juce::String& text, bool isError)
{
    auto content = object();
    set(content, "type", "text");
    set(content, "text", text);
    auto result = object();
    set(result, "content", array({ content }));
    set(result, "isError", isError);
    return result;
}

juce::var McpServer::errorResponse(const juce::var& id, int code, const juce::String& message)
{
    auto error = object();
    set(error, "code", code);
    set(error, "message", message);
    auto response = object();
    set(response, "jsonrpc", "2.0");
    set(response, "id", id);
    set(response, "error", error);
    return response;
}
}
