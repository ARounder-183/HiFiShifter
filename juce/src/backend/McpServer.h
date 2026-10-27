#pragma once

#include "../AudioEngine.h"
#include "../ProjectModel.h"
#include "../SampleSettings.h"
#include "MelodyneImporter.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <memory>

namespace hachi::backend
{
class McpServer final
{
public:
    // extraRoots are folders read_file and list_directory may see whatever
     // else the session does: --roots=A;B on the command line, and the
     // HACHISHIFTER_MCP_ROOTS environment variable.
    explicit McpServer(const juce::StringArray& extraRoots = {});
    // Borrow the document owned by the window (or an isolated edit draft).
    McpServer(ProjectModel& model, AudioEngine* engine,
              const juce::StringArray& extraRoots = {});
    juce::var processRequest(const juce::var& request, bool& shouldRespond) { return handle(request, shouldRespond); }
    juce::var executeTool(const juce::String& name, const juce::var& args) { return callTool(name, args); }
    juce::var snapshotJson() const { return projectJson(); }
    static juce::var noteJson(const NoteData& note, bool includeCurves = true);
    void allowFile(const juce::File& file) { allow(file); }
    static juce::var result(const juce::String& text, bool failed = false) { return toolResult(text, failed); }
    int run();
    // Whether a path lies in one of these folders, or is one of them.  A root
    // contains a path only by whole folders: "D:/work" does not contain
    // "D:/workshop/secret", however much of the name they share.
    [[nodiscard]] static bool pathWithinRoots(const std::vector<juce::File>& roots,
                                              const juce::File& target);
    // The folders a project's own work lives in: the voicebanks its tracks
    // sing from and the folders holding the recordings its clips play.
    [[nodiscard]] static std::vector<juce::File> projectRoots(const ProjectData& data);
    // The tools this server offers, exactly as tools/list sends them.
    [[nodiscard]] static juce::var diagnosticTools();
    // Shared live/headless scope resolution; empty never means the whole score.
    static bool resolveUtauRenderSelection(const ProjectData&, const juce::var& arguments,
        const std::vector<juce::String>& currentSelection, std::vector<juce::String>& result,
        juce::String& error);

private:
    juce::var handle(const juce::var& request, bool& shouldRespond);
    juce::var callTool(const juce::String& name, const juce::var& arguments);
    juce::var dispatch(const juce::String& name, const juce::var& arguments);
    // Everywhere the session may read: what it was started with, what it has
    // opened or written since, and where the open project's own work lives.
    [[nodiscard]] std::vector<juce::File> allowedRoots() const;
    void allow(const juce::File& file);
    juce::var projectJson() const;
    juce::int64 currentProjectFingerprint() const;
    void syncAudio(bool diffSingerExport = false);
    bool waitForRender(double timeoutSeconds, juce::String& error);
    juce::var transportStatusJson() const;
    static juce::var toolResult(const juce::String& text, bool isError = false);
    static juce::var errorResponse(const juce::var& id, int code, const juce::String& message);

    std::unique_ptr<ProjectModel> ownedProject;
    ProjectModel& project;
    juce::AudioFormatManager formats;
    std::unique_ptr<AudioEngine> ownedAudio;
    AudioEngine* audio = nullptr;
    juce::int64 preparedFingerprint = 0;
    bool audioPrepared = false;
    juce::String preparedDiffSingerOptions;
    std::vector<juce::File> configuredRoots;
    std::vector<juce::File> sessionRoots;
};
}
