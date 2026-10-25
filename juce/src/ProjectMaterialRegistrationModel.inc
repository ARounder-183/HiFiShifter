// Include ONCE from ProjectModel.cpp, OUTSIDE its namespace, after the normal
// member definitions. PM.h owns declarations and includes the neutral Types.h.
// No alternate project serializer, source updater, audio analysis or UI action.
#include "ProjectMaterialRegistrationTypes.h"
#if JUCE_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hachi
{
namespace registration_model_detail
{
using juce::File;
using juce::String;
using Locator = ProjectMaterialRegistrationLocator;
using OwnerBinding = ProjectMaterialRegistrationOwnerBinding;
using NoteBinding = ProjectMaterialRegistrationNoteBinding;
bool reject(String& error, const String& message) { error = message; return false; }
String ownerKey(const Locator& l)
{ return l.trackId + "|" + l.parentClipId + "|" + l.sourcePartId; }
String pathKey(const File& f)
{
    auto key = f.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    key = key.toLowerCase();
#endif
    return key;
}
struct View
{
    ClipData* parent = nullptr;
    ClipData* owner = nullptr;
    NoteData* note = nullptr;
    std::size_t trackIndex = 0, clipIndex = 0;
};
bool findOwner(ProjectData& data, const Locator& where, View& result, String& error)
{
    int matches = 0;
    for (std::size_t ti = 0; ti < data.tracks.size(); ++ti)
    {
        auto& track = data.tracks[ti]; if (track.id != where.trackId) continue;
        if (!trackShowsAllNativeRegions(track) || track.accompaniment)
            return reject(error, "Registration adoption requires a native physical-file track");
        for (std::size_t ci = 0; ci < track.clips.size(); ++ci)
        {
            auto& parent = track.clips[ci]; if (parent.id != where.parentClipId) continue;
            const auto accept = [&](ClipData& owner)
            { result = {&parent, &owner, nullptr, ti, ci}; ++matches; };
            if (where.sourcePartId.isEmpty() && parent.parts.empty()) accept(parent);
            else for (auto& part : parent.parts) if (part.id == where.sourcePartId) accept(part);
        }
    }
    return matches == 1 || reject(error, "Missing/ambiguous registration source owner: " + ownerKey(where));
}
bool findNote(ProjectData& data, const Locator& where, View& result, String& error)
{
    if (where.noteId.isEmpty() || !findOwner(data, where, result, error)) return false;
    int globalMatches = 0, ownerMatches = 0;
    for (const auto& track : data.tracks) for (const auto& clip : track.clips)
        for (const auto& note : clip.notes) if (note.id == where.noteId) ++globalMatches;
    for (auto& note : result.parent->notes)
        if (note.id == where.noteId && note.clipPartId == where.sourcePartId)
        { result.note = &note; ++ownerMatches; }
    return (globalMatches == 1 && ownerMatches == 1)
        || reject(error, "Missing/ambiguous registration note ID or part ownership: " + where.noteId);
}
// Keep actual source bytes and manifests immutable from SHA validation through
// commit. SHARE_READ deliberately refuses currently-open writers and prevents
// new writers, deletes and path replacement. No mtime-only source proof.
struct ReadProof
{
    File file;
#if JUCE_WINDOWS
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~ReadProof() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#endif
    String sha;
};
using Proofs = std::map<String, std::unique_ptr<ReadProof>>;
bool proveFile(const File& file, const String& expectedSha, Proofs& proofs, String& error)
{
    const auto key = pathKey(file);
    if (const auto found = proofs.find(key); found != proofs.end())
        return expectedSha.isEmpty() || found->second->sha == expectedSha.toLowerCase()
            || reject(error, "Conflicting registration SHA proofs for one path");
    if (file == File{} || !file.existsAsFile()) return reject(error, "Registration source/manifest is missing: " + file.getFullPathName());
    auto proof = std::make_unique<ReadProof>(); proof->file = file;
#if JUCE_WINDOWS
    proof->handle = CreateFileW(file.getFullPathName().toWideCharPointer(), GENERIC_READ,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (proof->handle == INVALID_HANDLE_VALUE)
        return reject(error, "Cannot acquire immutable registration read proof (Win32 "
            + String(GetLastError()) + "): " + file.getFullPathName());
#else
    return reject(error, "Immutable registration adoption read proofs are not implemented on this platform");
#endif
    proof->sha = juce::SHA256(file).toHexString();
    if (proof->sha.isEmpty() || (expectedSha.isNotEmpty() && proof->sha != expectedSha.toLowerCase()))
        return reject(error, "Registration recording SHA mismatch; project was not changed: " + file.getFullPathName());
    proofs.emplace(key, std::move(proof)); return true;
}
void normalizeReferences(ClipData& clip)
{
    clip.sourceFile = File{};
    for (auto& note : clip.notes)
    {
        note.materialAudioId.clear(); note.materialRegionId.clear(); note.materialBoundAlias.clear();
        note.materialAliasFollowsPrimary = true; note.materialLabelFollowsAlias = true;
        // materialSourceStart/End are NOT normalized: they choose the existing
        // P/source origin and remain musical/source-clock authority.
        note.materialLocalOverride = false; note.materialReferenceUnresolved = false;
    }
    for (auto& part : clip.parts) normalizeReferences(part);
}
void writeSupplement(juce::MemoryOutputStream& stream, const ClipData& clip)
{
    // Authoritative serializer already writes raw source arrays/clocks. Retain
    // its otherwise omitted null/empty/complete distinctions and transient group.
    stream.writeString(clip.id); stream.writeBool(bool(clip.nativeSourcePitch));
    stream.writeBool(clip.nativeSourcePitchComplete); stream.writeString(clip.nativePitchGroupId);
    stream.writeInt64(static_cast<juce::int64>(clip.parts.size()));
    for (const auto& part : clip.parts) writeSupplement(stream, part);
}
}

juce::String ProjectModel::registrationMusicSourceFingerprint(const ProjectData& input)
{
    // Constructor only sets the untitled name; no reset/replace or normalization.
    ProjectModel serializer;
    serializer.project = input;
    for (auto& track : serializer.project.tracks)
        for (auto& clip : track.clips) registration_model_detail::normalizeReferences(clip);
    juce::MemoryOutputStream bytes;
    serializer.toValueTree(juce::File{}, true).writeToStream(bytes);
    for (const auto& track : input.tracks)
        for (const auto& clip : track.clips) registration_model_detail::writeSupplement(bytes, clip);
    return juce::SHA256(bytes.getData(), bytes.getDataSize()).toHexString();
}

bool ProjectModel::adoptRegisteredMaterialBindings(std::uint64_t expectedRevision,
    const juce::String& expectedFingerprint,
    const std::vector<ProjectMaterialRegistrationOwnerBinding>& owners,
    const std::vector<ProjectMaterialRegistrationNoteBinding>& notes, juce::String& error)
{
    using namespace registration_model_detail;
    error.clear();
    if (owners.empty() || expectedFingerprint.isEmpty()) return reject(error, "Registration adoption requires source owners and an exact fingerprint");
    ProjectData candidate;
    {
        const juce::ScopedLock guard(lock);
        if (revision != expectedRevision || suppressNestedUndo) return reject(error, "Stale or nested registration transaction");
        candidate = project; // One operation snapshot; commit never replaces the project.
    }
    if (registrationMusicSourceFingerprint(candidate) != expectedFingerprint)
        return reject(error, "Registration music/source-clock fingerprint is stale");
    Proofs proofs;
    std::map<String, const OwnerBinding*> ownerPatches;
    std::set<std::pair<std::size_t, std::size_t>> touchedParents;
    File publicationRoot;
    for (const auto& patch : owners)
    {
        if (patch.where.noteId.isNotEmpty() || patch.oldSource == patch.copiedSource
            || patch.sourceSha256.length() != 64 || !patch.sourceSha256.toLowerCase().containsOnly("0123456789abcdef")
            || !ownerPatches.emplace(ownerKey(patch.where), &patch).second)
            return reject(error, "Malformed/duplicate registration owner patch");
        if (publicationRoot == File{}) publicationRoot = patch.copiedSource.getParentDirectory();
        if (patch.copiedSource.getParentDirectory() != publicationRoot
            || patch.oldSource.getFileExtension() != patch.copiedSource.getFileExtension())
            return reject(error, "Registration copies require one publication folder and unchanged recording extensions");
        View view;
        if (!findOwner(candidate, patch.where, view, error)) return false;
        if (view.owner->sourceFile != patch.oldSource || !view.owner->parts.empty()
            || view.owner->materialSourceAnalysisPending || view.owner->nativeSourcePitchPending)
            return reject(error, "Original registration source binding changed or analysis is pending");
        if (!proveFile(patch.oldSource, patch.sourceSha256, proofs, error)
            || !proveFile(patch.copiedSource, patch.sourceSha256, proofs, error)) return false;
        touchedParents.emplace(view.trackIndex, view.clipIndex);
    }
    const auto manifest = publicationRoot.getChildFile("otto.ini");
    if (!proveFile(manifest, {}, proofs, error)) return false;
    OttoMaterialIO::Document document;
    if (!OttoMaterialIO::loadDocument(manifest, document, error)) return false;
    std::set<String> noteIds;
    for (const auto& patch : notes)
    {
        if (!patch.preserveLocalAuthority || patch.audioId.isEmpty() || patch.regionId.isEmpty()
            || !noteIds.insert(patch.where.noteId).second)
            return reject(error, "Registration must preserve local note authority and unique stable bindings");
        const auto owner = ownerPatches.find(ownerKey(patch.where));
        if (owner == ownerPatches.end() || owner->second->oldSource != patch.oldSource
            || owner->second->copiedSource != patch.copiedSource || owner->second->sourceSha256 != patch.sourceSha256)
            return reject(error, "Registration note patch does not match its physical owner patch");
        View view;
        if (!findNote(candidate, patch.where, view, error)) return false;
        if (view.note->materialAudioId != patch.oldAudioId || view.note->materialRegionId != patch.oldRegionId)
            return reject(error, "Original registration audio/region binding changed");
        int matches = 0;
        for (const auto& audio : document.audio)
            if (audio.id == patch.audioId && publicationRoot.getChildFile(audio.relativePath) == patch.copiedSource)
                for (const auto& row : audio.rows)
                    if (row.materialRegionId == patch.regionId && row.regionStartSeconds == patch.sourceStart
                        && row.regionEndSeconds == patch.sourceEnd) ++matches;
        if (matches != 1) return reject(error, "Published registration audio/region identity or source bounds mismatch");
        auto& note = *view.note;
        note.materialAudioId = patch.audioId; note.materialRegionId = patch.regionId;
        note.materialBoundAlias = "-"; note.materialAliasFollowsPrimary = true; note.materialLabelFollowsAlias = false;
        // Preserve old/missing material baseline bounds exactly. Filling these
        // from a different canonical row would move the native P helper.
        note.materialLocalOverride = true; note.materialReferenceUnresolved = false;
    }
    // Complete binding coverage: never relocate an owner's PCM but leave one of
    // its carried notes silently bound to the original library identity.
    for (const auto& entry : ownerPatches)
    {
        View view; if (!findOwner(candidate, entry.second->where, view, error)) return false;
        for (const auto& note : view.parent->notes)
            if (note.clipPartId == entry.second->where.sourcePartId && !noteIds.count(note.id))
                return reject(error, "Incomplete registration note-binding coverage for source owner");
        view.owner->sourceFile = entry.second->copiedSource;
    }
    if (registrationMusicSourceFingerprint(candidate) != expectedFingerprint)
        return reject(error, "Registration candidate changed musical/source-clock authority");
    {
        const juce::ScopedLock guard(lock);
        // ReadProof handles still prohibit source/manifest mutation or replacement.
        if (revision != expectedRevision || suppressNestedUndo
            || registrationMusicSourceFingerprint(project) != expectedFingerprint)
            return reject(error, "Project changed before registration adoption");
        // Recheck all excluded old refs too: these deliberately are NOT hashed.
        for (const auto& patch : owners)
        {
            View view; if (!findOwner(project, patch.where, view, error)) return false;
            if (view.owner->sourceFile != patch.oldSource) return reject(error, "Registration owner path changed before commit");
        }
        for (const auto& patch : notes)
        {
            View view; if (!findNote(project, patch.where, view, error)) return false;
            if (view.note->materialAudioId != patch.oldAudioId || view.note->materialRegionId != patch.oldRegionId)
                return reject(error, "Registration stable binding changed before commit");
        }
        // All allocations/validations/candidate edits completed before this point.
        // Move only touched prebuilt clip snapshots, retaining one normal undo.
        pushUndoLocked();
        for (const auto& location : touchedParents)
            project.tracks[location.first].clips[location.second] =
                std::move(candidate.tracks[location.first].clips[location.second]);
    }
    sendChangeMessage(); return true;
}
}
