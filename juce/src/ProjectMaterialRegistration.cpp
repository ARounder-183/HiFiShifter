#include "ProjectMaterialRegistration.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#if JUCE_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hachi
{
namespace
{
using Service = ProjectMaterialRegistration;
using juce::File;
using juce::String;
String mint(const char* prefix) { return String(prefix) + juce::Uuid().toString().removeCharacters("-"); }
bool fail(String& error, const String& message) { error = message; return false; }
bool cancelled(const Service::Cancel& cancel) { return cancel && cancel(); }
std::filesystem::path fsPath(const File& f)
{
#if JUCE_WINDOWS
    return std::filesystem::path(f.getFullPathName().toWideCharPointer());
#else
    return std::filesystem::path(f.getFullPathName().toStdString());
#endif
}
String pathKey(const File& f)
{
    auto path = f.getFullPathName().replaceCharacter('\\', '/');
#if JUCE_WINDOWS
    path = path.toLowerCase();
#endif
    return path;
}
// Physical file identity proves hardlink/symlink aliases; equal SHA alone does NOT.
String recordingKey(const File& file)
{
#if JUCE_WINDOWS
    const auto handle = CreateFileW(file.getFullPathName().toWideCharPointer(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle != INVALID_HANDLE_VALUE)
    {
        BY_HANDLE_FILE_INFORMATION info{};
        const bool ok = GetFileInformationByHandle(handle, &info) != 0;
        CloseHandle(handle);
        if (ok) return "physical:" + String(info.dwVolumeSerialNumber) + ":"
            + String(info.nFileIndexHigh) + ":" + String(info.nFileIndexLow);
    }
#endif
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(fsPath(file), ec);
    if (!ec)
    {
#if JUCE_WINDOWS
        return pathKey(File(String(canonical.wstring().c_str())));
#else
        return pathKey(File(String(canonical.string())));
#endif
    }
    return pathKey(file);
}
bool fingerprintFile(const File& file, String& hash, juce::int64& bytes, String& error)
{
    if (!file.existsAsFile()) return fail(error, "Recording is missing: " + file.getFullPathName());
    const auto size = file.getSize(); const auto time = file.getLastModificationTime();
    if (size <= 0) return fail(error, "Recording is empty: " + file.getFullPathName());
    const auto value = juce::SHA256(file).toHexString();
    if (value.isEmpty() || size != file.getSize() || time != file.getLastModificationTime())
        return fail(error, "Recording changed while hashing: " + file.getFullPathName());
    hash = value; bytes = size; return true;
}
String locatorKey(const Service::Locator& l)
{ return l.trackId + "|" + l.parentClipId + "|" + l.sourcePartId + "|" + l.noteId; }
struct View { const ClipData* owner; const NoteData* note; };
bool locate(const ProjectData& project, const Service::Locator& l, View& out)
{
    int matches = 0;
    for (const auto& track : project.tracks) if (track.id == l.trackId)
        for (const auto& clip : track.clips) if (clip.id == l.parentClipId)
        {
            const ClipData* owner = nullptr;
            if (l.sourcePartId.isEmpty() && clip.parts.empty()) owner = &clip;
            else for (const auto& part : clip.parts) if (part.id == l.sourcePartId) owner = &part;
            if (!owner) continue;
            for (const auto& note : clip.notes) if (note.id == l.noteId && note.clipPartId == l.sourcePartId)
            { out = {owner, &note}; ++matches; }
        }
    return matches == 1;
}
// Copy authoring facts ONLY. Musical tuning/envelopes in an imported project
// never become a global source default (including divergent repeated placements).
SampleRegionSetting authoringRow(const SampleRegionSetting& from)
{
    SampleRegionSetting row;
    row.name = "-";
    row.regionStartSeconds = from.regionStartSeconds; row.regionEndSeconds = from.regionEndSeconds;
    row.alignmentSeconds = from.alignmentSeconds; row.fixedDurationSeconds = from.fixedDurationSeconds;
    row.overlapSeconds = from.overlapSeconds; row.role = from.role;
    row.provenance = from.provenance; row.confidence = from.confidence;
    row.nativeUnpitched = from.nativeUnpitched;
    row.pronunciationKind = from.pronunciationKind; row.simplifiedConsonant = from.simplifiedConsonant;
    row.vowelSplitsExplicit = from.vowelSplitsExplicit; row.segments = from.segments;
    // Use shared C/P and explicit-V policy, not a registration-specific normalizer.
    // Raw invalid facts are validated BEFORE this canonical projection.
    auto local = pronunciationSegmentsFor(row);
    for (auto& segment : local)
    {
        segment.sourceStartSeconds += row.regionStartSeconds;
        segment.sourceEndSeconds += row.regionStartSeconds;
        segment.alignmentSeconds += row.regionStartSeconds;
    }
    row.segments = std::move(local);
    return row;
}
bool treeSafe(const File& root, String& error)
{
    std::error_code ec;
#if JUCE_WINDOWS
    const auto attributes = GetFileAttributesW(root.getFullPathName().toWideCharPointer());
    if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
        return fail(error, "Owned folder is missing or redirected; cleanup refused");
#endif
    for (std::filesystem::recursive_directory_iterator i(fsPath(root), ec), end; !ec && i != end; i.increment(ec))
    {
#if JUCE_WINDOWS
        const auto attributes = GetFileAttributesW(i->path().c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
            return fail(error, "Redirected child in owned folder; cleanup refused");
#else
        if (i->is_symlink(ec)) return fail(error, "Symlink in owned folder; cleanup refused");
#endif
    }
    if (ec) return fail(error, "Cannot verify owned folder tree");
    return true;
}
constexpr const char* markerName = ".hachi-registration-owner";
bool writeNewOwnershipMarker(const File& marker, const String& token, String& error)
{
    // This is a NEW file in an exclusively-created directory, not a replacement.
    // JUCE replaceWithText uses a dotfile temporary whose duplicated extension
    // can exceed MAX_PATH even when the marker itself is valid (26: 224 -> 263).
    // Never weaken ownership checks or overwrite a marker that already exists.
#if JUCE_WINDOWS
    const auto handle = CreateFileW(marker.getFullPathName().toWideCharPointer(), GENERIC_WRITE,
        0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        return fail(error, "Cannot exclusively create ownership marker (Win32 " + String(GetLastError())
            + ", path length " + String(marker.getFullPathName().length()) + "): " + marker.getFullPathName());
    DWORD written = 0;
    const auto count = static_cast<DWORD>(token.getNumBytesAsUTF8());
    const bool wrote = WriteFile(handle, token.toRawUTF8(), count, &written, nullptr) != 0 && written == count;
    const bool flushed = wrote && FlushFileBuffers(handle) != 0;
    const auto status = flushed ? DWORD(0) : GetLastError();
    CloseHandle(handle);
    if (!flushed) return fail(error, "Cannot write/flush ownership marker (Win32 " + String(status) + ")");
#else
    auto* file = std::fopen(marker.getFullPathName().toRawUTF8(), "wx");
    if (!file) return fail(error, "Cannot exclusively create ownership marker");
    const auto count = token.getNumBytesAsUTF8();
    const bool wrote = std::fwrite(token.toRawUTF8(), 1, count, file) == count;
    const bool flushed = std::fflush(file) == 0;
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !flushed || !closed) return fail(error, "Cannot write/flush ownership marker");
#endif
    return marker.loadFileAsString() == token || fail(error, "Ownership marker readback mismatch");
}
}

bool Service::resolveOwnerScope(const ProjectData& project, const OwnerScope& requested,
    std::vector<Locator>& selected, String& error)
{
    error.clear(); selected.clear();
    const auto same = [](const Locator& a, const Locator& b)
    { return a.trackId == b.trackId && a.parentClipId == b.parentClipId && a.sourcePartId == b.sourcePartId; };
    if (requested)
    {
        if (requested->empty()) return fail(error, "No physical source owners selected");
        for (std::size_t i = 0; i < requested->size(); ++i)
        {
            const auto& item = (*requested)[i];
            if (item.trackId.isEmpty() || item.parentClipId.isEmpty() || item.noteId.isNotEmpty())
                return fail(error, "Owner scope requires track/parent/part IDs and an EMPTY note ID");
            for (std::size_t j = 0; j < i; ++j) if (same(item, (*requested)[j]))
                return fail(error, "Duplicate selected physical owner");
        }
    }
    std::vector<Locator> resolved;
    for (const auto& track : project.tracks)
    {
        if (!trackShowsAllNativeRegions(track) || track.accompaniment) continue;
        for (const auto& parent : track.clips)
        {
            const auto append = [&](const ClipData& owner, const String& part)
            {
                if (owner.sourceFile == File{}) return;
                Locator id {track.id, parent.id, part, {}};
                if (!requested || std::any_of(requested->begin(), requested->end(), [&](const auto& wanted) { return same(id, wanted); }))
                    resolved.push_back(std::move(id));
            };
            if (parent.parts.empty()) append(parent, {});
            else for (const auto& part : parent.parts) append(part, part.id);
        }
    }
    for (std::size_t i = 0; i < resolved.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) if (same(resolved[i], resolved[j]))
            return fail(error, "Ambiguous physical owner identity in project");
    if (requested) for (const auto& wanted : *requested)
        if (std::none_of(resolved.begin(), resolved.end(), [&](const auto& found) { return same(wanted, found); }))
            return fail(error, "Selected source owner is missing, ineligible or has no physical recording");
    if (resolved.empty()) return fail(error, "No eligible native physical source owners");
    selected = std::move(resolved); return true;
}

Service::Plan Service::prepare(const ProjectData& project, std::uint64_t revision,
    const File& parent, const String& folderName, const std::vector<VariantFacts>& facts,
    const Fingerprint& fingerprint, Cancel cancel, const OwnerScope& scope)
{
    Plan plan; plan.expectedRevision = revision; plan.destinationParent = parent;
    const auto reject = [&](const String& why) { plan.diagnostics.add(why); };
    std::vector<Locator> selected; String selectionError;
    if (!resolveOwnerScope(project, scope, selected, selectionError)) { reject(selectionError); return plan; }
    const auto includesOwner = [&](const String& track, const String& parentId, const String& part)
    { return std::any_of(selected.begin(), selected.end(), [&](const auto& id)
        { return id.trackId == track && id.parentClipId == parentId && id.sourcePartId == part; }); };
    if (!fingerprint) { reject("Missing authoritative music/source-clock fingerprint callback"); return plan; }
    plan.musicSourceFingerprint = fingerprint(project);
    if (plan.musicSourceFingerprint.isEmpty()) { reject("Empty music/source-clock fingerprint"); return plan; }
    if (!parent.isDirectory() || folderName.isEmpty() || folderName == "." || folderName == ".."
        || folderName.containsAnyOf("/\\:") || File::createLegalFileName(folderName) != folderName
        || folderName.endsWithChar('.') || folderName.endsWithChar(' '))
    { reject("Choose an existing parent and a legal NEW child folder name"); return plan; }
    plan.publishedFolder = parent.getChildFile(folderName);
    if (plan.publishedFolder.exists()) { reject("Destination already exists; registration never replaces folders"); return plan; }
    std::map<String, std::size_t> recordingIndices;
    std::set<String> filenames, requiredNotes, mappedNotes, variants;
    String error;
    const auto addRecording = [&](const File& file) -> bool
    {
        if (cancelled(cancel)) { reject("Registration cancelled"); return false; }
        const auto key = recordingKey(file);
        auto found = recordingIndices.find(key);
        if (found != recordingIndices.end())
        {
            const auto& aliases = plan.recordings[found->second].aliases;
            if (std::find(aliases.begin(), aliases.end(), file) != aliases.end()) return true;
        }
        String hash; juce::int64 bytes = 0;
        if (!fingerprintFile(file, hash, bytes, error)) { reject(error); return false; }
        if (found != recordingIndices.end())
        {
            auto& recording = plan.recordings[found->second];
            if (recording.sha256 != hash || recording.bytes != bytes) { reject("Physical recording changed during enumeration"); return false; }
            if (std::find(recording.aliases.begin(), recording.aliases.end(), file) == recording.aliases.end()) recording.aliases.push_back(file);
            return true;
        }
        auto base = File::createLegalFileName(file.getFileNameWithoutExtension());
        if (base.isEmpty()) base = "recording";
        auto filename = base + file.getFileExtension();
        int suffix = 2;
        while (!filenames.insert(filename.toLowerCase()).second)
            filename = base + "_" + String(suffix++) + file.getFileExtension();
        recordingIndices[key] = plan.recordings.size();
        plan.recordings.push_back({key, hash, filename, bytes, file, {file}});
        return true;
    };
    for (const auto& track : project.tracks)
    {
        if (!trackShowsAllNativeRegions(track) || track.accompaniment) continue;
        for (const auto& clip : track.clips)
        {
            const auto visit = [&](const ClipData& owner, const String& part) -> bool
            {
                if (!includesOwner(track.id, clip.id, part)) return true;
                if (!owner.parts.empty()) { reject("Nested source owners require an explicit ownership adapter"); return false; }
                if (owner.sourceFile == File{}) return true;
                if (owner.materialSourceAnalysisPending || owner.nativeSourcePitchPending)
                { reject("Source analysis is pending; finish it before reference-only registration"); return false; }
                if (!addRecording(owner.sourceFile)) return false;
                const auto& recording = plan.recordings[recordingIndices.at(recordingKey(owner.sourceFile))];
                plan.owners.push_back({{track.id, clip.id, part, {}}, owner.sourceFile,
                    plan.publishedFolder.getChildFile(recording.copiedRelativePath), recording.sha256});
                for (const auto& note : clip.notes) if (note.clipPartId == part)
                    if (!requiredNotes.insert(locatorKey({track.id, clip.id, part, note.id})).second)
                    { reject("Duplicate project note/owner locator"); return false; }
                return true;
            };
            for (const auto& note : clip.notes)
            {
                if (scope && !includesOwner(track.id, clip.id, clip.parts.empty() ? String{} : note.clipPartId)) continue;
                int owners = 0;
                if (clip.parts.empty()) owners = note.clipPartId.isEmpty() ? 1 : 0;
                else for (const auto& part : clip.parts) if (part.id == note.clipPartId) ++owners;
                if (owners != 1) { reject("Ambiguous/orphan source ownership: " + note.id); return plan; }
            }
            if (clip.parts.empty()) { if (!visit(clip, {})) return plan; }
            else for (const auto& part : clip.parts) if (!visit(part, part.id)) return plan;
        }
    }
    if (plan.recordings.empty()) { reject("No eligible native physical recordings"); return plan; }
    std::set<String> coveredRecordings;
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    for (const auto& variant : facts)
    {
        if (cancelled(cancel)) { reject("Registration cancelled"); return plan; }
        const auto key = recordingKey(variant.recording);
        const auto found = recordingIndices.find(key);
        if (found == recordingIndices.end()) { reject("Annotation facts refer to a recording outside the project"); continue; }
        if (!variant.explicitPronunciationEvidence || variant.rows.empty())
        { reject("Unresolved pronunciation grouping: " + variant.recording.getFullPathName()); continue; }
        if (variant.variantKey.isEmpty() || !variants.insert(key + "|" + variant.variantKey).second)
        { reject("Missing/duplicate explicit annotation variant key"); continue; }
        coveredRecordings.insert(key);
        const auto& recording = plan.recordings[found->second];
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(recording.original));
        if (!reader || reader->sampleRate <= 0 || reader->lengthInSamples <= 0)
        { reject("Cannot validate source recording bounds"); continue; }
        const double duration = reader->lengthInSamples / reader->sampleRate;
        OttoMaterialIO::Audio audio; audio.id = mint("a"); audio.relativePath = recording.copiedRelativePath;
        std::map<String, String> regionIds;
        for (const auto& fact : variant.rows)
            if (fact.key.isEmpty() || !regionIds.emplace(fact.key, mint("p")).second)
                reject("Missing/duplicate carried pronunciation key");
        for (const auto& fact : variant.rows)
        {
            if (fact.key.isEmpty() || !regionIds.count(fact.key)) continue;
            // Validate raw evidence before shared helpers can clamp bad values.
            auto raw = fact.annotation; raw.name = "-"; raw.additionalAliases.clear();
            raw.materialAudioId = audio.id; raw.materialRegionId = regionIds[fact.key]; raw.connectedNextRegionId.clear();
            OttoMaterialIO::Document check; check.audio.push_back({audio.id, audio.relativePath, {raw}});
            if (!OttoMaterialIO::validate(check, error) || raw.regionEndSeconds > duration)
            { reject("Invalid carried source facts: " + fact.key + ": " + error); continue; }
            auto row = authoringRow(raw); row.materialAudioId = audio.id; row.materialRegionId = raw.materialRegionId;
            if (fact.nextRowKey.isNotEmpty())
            {
                const auto next = regionIds.find(fact.nextRowKey);
                if (next == regionIds.end()) reject("Unknown explicit next pronunciation key");
                else row.connectedNextRegionId = next->second;
            }
            audio.rows.push_back(row);
            for (const auto& member : fact.members)
            {
                View view{}; const auto memberKey = locatorKey(member);
                if (!requiredNotes.count(memberKey) || !locate(project, member, view)
                    || recordingKey(view.owner->sourceFile) != key || !mappedNotes.insert(memberKey).second)
                { reject("Ambiguous/duplicate/missing carried pronunciation member: " + memberKey); continue; }
                Binding binding; binding.where = member; binding.variantKey = variant.variantKey; binding.rowKey = fact.key;
                binding.oldSource = view.owner->sourceFile; binding.copiedSource = plan.publishedFolder.getChildFile(recording.copiedRelativePath);
                binding.sourceSha256 = recording.sha256; binding.oldAudioId = view.note->materialAudioId;
                binding.oldRegionId = view.note->materialRegionId; binding.audioId = audio.id; binding.regionId = row.materialRegionId;
                binding.sourceStart = row.regionStartSeconds; binding.sourceEnd = row.regionEndSeconds;
                plan.bindings.push_back(std::move(binding));
            }
        }
        Variant identity; identity.recordingKey = key; identity.variantKey = variant.variantKey; identity.audioId = audio.id;
        for (const auto& id : regionIds) identity.rowIds.push_back(id);
        plan.variants.push_back(std::move(identity));
        plan.document.audio.push_back(std::move(audio));
    }
    // Repeated actual usages may carry the exact same published authoring set.
    // Coalesce those sets only, never enumerate subsets/combinations. Normalize
    // identity for comparison through the shared lossless codec; physical paths,
    // all timing/authoring data and explicit next-row topology remain in the key.
    // Tuning/local snapshots never entered these source rows and remain per-note.
    if (plan.diagnostics.isEmpty())
    {
        std::map<String, std::size_t> annotationSets;
        std::vector<OttoMaterialIO::Audio> uniqueAudio;
        for (auto& audio : plan.document.audio)
        {
            auto canonical = audio;
            std::stable_sort(canonical.rows.begin(), canonical.rows.end(), [](const auto& a, const auto& b)
            {
                if (a.regionStartSeconds != b.regionStartSeconds) return a.regionStartSeconds < b.regionStartSeconds;
                return a.regionEndSeconds < b.regionEndSeconds;
            });
            std::map<String, String> normalizedIds;
            for (std::size_t i = 0; i < canonical.rows.size(); ++i)
                normalizedIds.emplace(canonical.rows[i].materialRegionId, "p" + String(static_cast<juce::int64>(i)));
            const auto orderedRows = canonical.rows; // exact old identity -> corresponding shared row
            canonical.id = "aCanonical";
            for (auto& row : canonical.rows)
            {
                row.materialAudioId = canonical.id;
                row.materialRegionId = normalizedIds.at(row.materialRegionId);
                if (row.connectedNextRegionId.isNotEmpty()) row.connectedNextRegionId = normalizedIds.at(row.connectedNextRegionId);
            }
            OttoMaterialIO::Document comparison; comparison.audio.push_back(std::move(canonical));
            String signature;
            if (!OttoMaterialIO::encode(comparison, signature, error))
            { reject("Cannot compare actual annotation sets: " + error); break; }
            const auto found = annotationSets.find(signature);
            if (found == annotationSets.end())
            {
                annotationSets.emplace(signature, uniqueAudio.size());
                // Store source-ordered rows so repeated placements map by the
                // same normalized row positions, not by note/vector order.
                audio.rows = orderedRows;
                uniqueAudio.push_back(std::move(audio));
                continue;
            }
            const auto& shared = uniqueAudio[found->second];
            std::map<String, String> sharedIds;
            for (std::size_t i = 0; i < orderedRows.size(); ++i)
                sharedIds.emplace(orderedRows[i].materialRegionId, shared.rows[i].materialRegionId);
            for (auto& binding : plan.bindings) if (binding.audioId == audio.id)
            {
                binding.audioId = shared.id; binding.regionId = sharedIds.at(binding.regionId);
            }
            for (auto& variant : plan.variants) if (variant.audioId == audio.id)
            {
                variant.audioId = shared.id;
                for (auto& row : variant.rowIds) row.second = sharedIds.at(row.second);
            }
        }
        if (plan.diagnostics.isEmpty()) plan.document.audio = std::move(uniqueAudio);
    }
    for (const auto& r : plan.recordings) if (!coveredRecordings.count(r.key))
        reject("Unresolved explicit pronunciation facts for recording: " + r.original.getFullPathName());
    for (const auto& note : requiredNotes) if (!mappedNotes.count(note)) reject("Unresolved pronunciation membership: " + note);
    if (!OttoMaterialIO::validate(plan.document, error)) reject("Shared OTTO codec cannot publish this candidate: " + error);
    if (plan.diagnostics.isEmpty())
    {
        plan.ready = true; ProjectData candidate;
        if (!makeCandidate(project, plan, fingerprint, candidate, error)) { plan.ready = false; reject(error); }
    }
    return plan;
}

bool Service::makeCandidate(const ProjectData& project, const Plan& plan, const Fingerprint& fingerprint,
    ProjectData& output, String& error)
{
    error.clear();
    if (!plan.ready || !fingerprint || fingerprint(project) != plan.musicSourceFingerprint)
        return fail(error, "Plan is unresolved or original music/source fingerprint changed");
    if (!OttoMaterialIO::validate(plan.document, error)) return false;
    const auto plannedSource = [&](const File& oldSource, const File& copied, const String& sha)
    {
        for (const auto& recording : plan.recordings)
            if (recording.sha256 == sha && copied == plan.publishedFolder.getChildFile(recording.copiedRelativePath)
                && std::find(recording.aliases.begin(), recording.aliases.end(), oldSource) != recording.aliases.end()) return true;
        return false;
    };
    for (const auto& binding : plan.bindings)
    {
        bool rowFound = false;
        for (const auto& audio : plan.document.audio) if (audio.id == binding.audioId
            && binding.copiedSource == plan.publishedFolder.getChildFile(audio.relativePath))
            for (const auto& row : audio.rows) if (row.materialRegionId == binding.regionId
                && row.regionStartSeconds == binding.sourceStart && row.regionEndSeconds == binding.sourceEnd) rowFound = true;
        if (!rowFound || !plannedSource(binding.oldSource, binding.copiedSource, binding.sourceSha256))
            return fail(error, "Candidate binding does not match planned recording/document identity");
    }
    auto candidate = project;
    for (const auto& owner : plan.owners)
    {
        if (!plannedSource(owner.oldSource, owner.copiedSource, owner.sourceSha256))
            return fail(error, "Candidate owner does not match planned copied recording");
        int matches = 0;
        for (auto& track : candidate.tracks) if (track.id == owner.where.trackId)
            for (auto& clip : track.clips) if (clip.id == owner.where.parentClipId)
            {
                const auto apply = [&](ClipData& target)
                {
                    if (target.sourceFile != owner.oldSource) return false;
                    target.sourceFile = owner.copiedSource; ++matches; return true;
                };
                if (owner.where.sourcePartId.isEmpty() && clip.parts.empty()) apply(clip);
                else for (auto& part : clip.parts) if (part.id == owner.where.sourcePartId) apply(part);
            }
        if (matches != 1) return fail(error, "Original physical source owner no longer matches registration plan");
    }
    for (const auto& binding : plan.bindings)
    {
        View view{};
        if (!locate(project, binding.where, view) || view.owner->sourceFile != binding.oldSource
            || view.note->materialAudioId != binding.oldAudioId || view.note->materialRegionId != binding.oldRegionId)
            return fail(error, "Original binding no longer matches registration plan");
        for (auto& track : candidate.tracks) if (track.id == binding.where.trackId)
            for (auto& clip : track.clips) if (clip.id == binding.where.parentClipId)
            {
                if (binding.where.sourcePartId.isEmpty()) clip.sourceFile = binding.copiedSource;
                else for (auto& part : clip.parts) if (part.id == binding.where.sourcePartId) part.sourceFile = binding.copiedSource;
                for (auto& note : clip.notes) if (note.id == binding.where.noteId)
                {
                    note.materialAudioId = binding.audioId; note.materialRegionId = binding.regionId;
                    note.materialBoundAlias = "-"; note.materialAliasFollowsPrimary = true;
                    note.materialLabelFollowsAlias = false;
                    // These bounds also choose the effective P/source origin.
                    // Keep the carried snapshot EXACT, including absent (-1)
                    // baselines; published row bounds are NOT a new local clock.
                    note.materialLocalOverride = binding.preserveLocalAuthority;
                    note.materialReferenceUnresolved = false;
                }
            }
    }
    if (fingerprint(candidate) != plan.musicSourceFingerprint)
        return fail(error, "Reference-only candidate changed music/source-clock fingerprint");
    output = std::move(candidate); return true;
}

std::unique_ptr<Service::Lease> Service::stage(const Plan& plan, String& error, Cancel cancel)
{
    error.clear();
    if (!plan.ready || !plan.destinationParent.isDirectory() || plan.publishedFolder.exists()
        || plan.publishedFolder.getParentDirectory() != plan.destinationParent)
    { fail(error, "Registration plan is not publishable into a new child folder"); return {}; }
    if (!OttoMaterialIO::validate(plan.document, error) || cancelled(cancel))
    { if (error.isEmpty()) error = "Registration cancelled"; return {}; }
    std::set<String> names;
    for (const auto& recording : plan.recordings)
    {
        const auto& name = recording.copiedRelativePath;
        if (name.isEmpty() || name.containsAnyOf("/\\\\:") || name == "." || name == ".."
            || File::createLegalFileName(name) != name || name.equalsIgnoreCase("otto.ini")
            || name.equalsIgnoreCase(markerName) || !names.insert(name.toLowerCase()).second)
        { fail(error, "Unsafe or duplicate copied recording filename"); return {}; }
    }
    if (names.empty()) { fail(error, "No physical recordings in plan"); return {}; }
    for (const auto& audio : plan.document.audio) if (!names.count(audio.relativePath.toLowerCase()))
    { fail(error, "Annotation section has no planned physical recording"); return {}; }
    auto lease = std::unique_ptr<Lease>(new Lease()); lease->prepared = plan; lease->ownershipToken = mint("registration_");
    lease->ownedFolder = plan.destinationParent.getChildFile(".hachi-stage-" + lease->ownershipToken);
    std::error_code ec;
    if (!std::filesystem::create_directory(fsPath(lease->ownedFolder), ec))
    { lease->released = true; fail(error, "Cannot exclusively create registration staging folder"); return {}; }
    if (!writeNewOwnershipMarker(lease->ownedFolder.getChildFile(markerName), lease->ownershipToken, error))
    {
        // Exclusive creation above proves ownership even before the marker exists.
        String safety;
        if (treeSafe(lease->ownedFolder, safety)) std::filesystem::remove_all(fsPath(lease->ownedFolder), ec);
        if (ec || safety.isNotEmpty()) error += "; cleanup could not be verified";
        lease->released = true; return {};
    }
    for (const auto& recording : plan.recordings)
    {
        if (cancelled(cancel)) { fail(error, "Registration cancelled"); return {}; }
        String hash; juce::int64 bytes;
        if (!fingerprintFile(recording.original, hash, bytes, error) || hash != recording.sha256 || bytes != recording.bytes)
        { if (error.isEmpty()) error = "Original recording changed before copying"; return {}; }
        if (!recording.original.copyFileTo(lease->ownedFolder.getChildFile(recording.copiedRelativePath)))
        { fail(error, "Cannot copy original recording bytes"); return {}; }
    }
    auto document = plan.document;
    if (!OttoMaterialIO::saveDocument(lease->ownedFolder.getChildFile("otto.ini"), document, error)
        || !lease->verify(error, cancel)) return {};
    return lease;
}

Service::Lease::~Lease()
{
    if (!released) { String error; rollback(error); }
}
bool Service::Lease::verify(String& error, Cancel cancel) const
{
    if (ownedFolder.getChildFile(markerName).loadFileAsString() != ownershipToken)
        return fail(error, "Registration ownership marker mismatch");
    if (!treeSafe(ownedFolder, error)) return false;
    for (const auto& recording : prepared.recordings)
    {
        if (cancelled(cancel)) return fail(error, "Registration cancelled");
        auto files = recording.aliases;
        files.push_back(recording.original);
        files.push_back(ownedFolder.getChildFile(recording.copiedRelativePath));
        for (const auto& file : files)
        {
            String hash; juce::int64 bytes;
            if (!fingerprintFile(file, hash, bytes, error)) return false;
            if (hash != recording.sha256 || bytes != recording.bytes) return fail(error, "Recording bytes changed; publication refused");
        }
    }
    OttoMaterialIO::Document current; String expectedText, currentText;
    return OttoMaterialIO::loadDocument(ownedFolder.getChildFile("otto.ini"), current, error)
        && OttoMaterialIO::encode(prepared.document, expectedText, error)
        && OttoMaterialIO::encode(current, currentText, error)
        && (expectedText == currentText || fail(error, "Staged annotation document changed"));
}
bool Service::Lease::publish(String& error, Cancel cancel)
{
    error.clear();
    if (released || isPublished) return fail(error, "Registration lease is no longer staged");
    if (!verify(error, cancel) || cancelled(cancel)) { if (error.isEmpty()) error = "Registration cancelled"; return false; }
    if (prepared.publishedFolder.exists()) return fail(error, "Destination appeared; existing folder retained");
#if JUCE_WINDOWS
    // No MOVEFILE_REPLACE_EXISTING: destination races fail, never overwrite.
    if (!MoveFileExW(ownedFolder.getFullPathName().toWideCharPointer(),
        prepared.publishedFolder.getFullPathName().toWideCharPointer(), MOVEFILE_WRITE_THROUGH))
        return fail(error, "Atomic no-replace folder publication failed");
#else
    return fail(error, "Atomic no-replace folder publication is not implemented on this platform");
#endif
    ownedFolder = prepared.publishedFolder; isPublished = true; return true;
}
bool Service::Lease::rollback(String& error)
{
    error.clear(); if (released) return true;
    if (ownedFolder == File{} || ownedFolder.getParentDirectory() != prepared.destinationParent
        || ownedFolder.getChildFile(markerName).loadFileAsString() != ownershipToken)
        return fail(error, "Owned registration folder cannot be proven; rollback refused");
    if (!treeSafe(ownedFolder, error)) return false;
    std::error_code ec; std::filesystem::remove_all(fsPath(ownedFolder), ec);
    if (ec) return fail(error, "Cannot remove owned registration folder");
    released = true; return true;
}
bool Service::Lease::release(String& error)
{
    error.clear();
    if (permanentlyRetained) return true;
    if (released || !isPublished) return fail(error, "Only a published owned lease can be released");
    // Successful adoption is the irreversible ownership boundary, not manager
    // preference IO. A damaged marker must never delete already-adopted media.
    retainAfterAdoption(); return true;
}
}
