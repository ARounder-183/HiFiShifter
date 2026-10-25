#pragma once
#include "../ProjectModel.h"
#include "../OttoMaterialIO.h"
#include <juce_cryptography/juce_cryptography.h>
#include <map>

namespace hachi
{
// Read-only canonical HJPX load and raw evidence report. This is NOT a grouping
// adapter: analysis-note epochs, voiced centres and adjacent clips never become
// authoring pronunciations here. Human/host can inspect real carried facts first.
inline bool runProjectMaterialRegistrationEvidenceAudit(const juce::File& originalProject,
    const juce::File& outputPrefix, juce::File& reportFile, juce::String& error)
{
    error.clear();
    if (!originalProject.existsAsFile() || !outputPrefix.isDirectory()) { error = "Existing project and output prefix required"; return false; }
    const auto projectSha = juce::SHA256(originalProject).toHexString();
    ProjectModel model; if (!model.load(originalProject, error) || error.isNotEmpty()) return false;
    const auto data = model.snapshot();
    auto root = std::make_unique<juce::DynamicObject>();
    root->setProperty("project", originalProject.getFullPathName()); root->setProperty("projectSHA256", projectSha);
    root->setProperty("inference", "NONE: raw carried evidence only; no note/event/analysis epoch is assumed to be a pronunciation");
    juce::Array<juce::var> owners;
    std::map<juce::String, juce::String> hashes;
    for (const auto& track : data.tracks) for (const auto& parent : track.clips)
    {
        const auto append = [&](const ClipData& source, const juce::String& partId)
        {
            if (source.sourceFile == juce::File{}) return;
            auto record = std::make_unique<juce::DynamicObject>();
            record->setProperty("trackId", track.id); record->setProperty("trackName", track.name);
            record->setProperty("parentClipId", parent.id); record->setProperty("partId", partId);
            record->setProperty("source", source.sourceFile.getFullPathName());
            const auto key = source.sourceFile.getFullPathName();
            if (!hashes.count(key)) hashes[key] = source.sourceFile.existsAsFile() ? juce::SHA256(source.sourceFile).toHexString() : juce::String("MISSING");
            record->setProperty("sourceSHA256", hashes[key]);
            record->setProperty("offset", source.sourceOffsetSeconds); record->setProperty("sourceDuration", source.sourceDurationSeconds);
            record->setProperty("targetDuration", source.durationSeconds); record->setProperty("audioStart", source.audioStartSeconds);
            record->setProperty("audioDuration", source.audioDurationSeconds);
            record->setProperty("fullSourcePitchFrames", source.nativeSourcePitch ? int(source.nativeSourcePitch->size()) : 0);
            record->setProperty("fullSourcePitchComplete", source.nativeSourcePitchComplete);
            record->setProperty("sourceAnalysisPending", source.materialSourceAnalysisPending || source.nativeSourcePitchPending);
            juce::Array<juce::var> clock;
            for (const auto& p : source.sourceTimeMap)
            { auto v = std::make_unique<juce::DynamicObject>(); v->setProperty("target", p.targetSeconds); v->setProperty("source", p.sourceSeconds); clock.add(juce::var(v.release())); }
            record->setProperty("rawSourceTimeMap", juce::var(clock));
            const auto manifest = OttoMaterialIO::findManifest(source.sourceFile);
            record->setProperty("nearestNativeManifest", manifest.getFullPathName());
            juce::Array<juce::var> variants;
            if (manifest.existsAsFile())
            {
                std::vector<OttoMaterialIO::Audio> entries; juce::String problem;
                if (OttoMaterialIO::listAudioVariants(manifest, source.sourceFile, entries, problem))
                    for (const auto& entry : entries)
                    { auto v = std::make_unique<juce::DynamicObject>(); v->setProperty("audioId", entry.id); v->setProperty("rows", int(entry.rows.size())); variants.add(juce::var(v.release())); }
                else record->setProperty("manifestDiagnostic", problem);
            }
            record->setProperty("existingVariants", juce::var(variants));
            juce::Array<juce::var> notes;
            for (const auto& note : parent.notes) if (note.clipPartId == partId)
            {
                auto n = std::make_unique<juce::DynamicObject>();
                n->setProperty("noteId", note.id); n->setProperty("label", note.label);
                n->setProperty("start", note.startSeconds); n->setProperty("duration", note.durationSeconds);
                n->setProperty("nativeSourceStart", note.nativeSourceStartSeconds); n->setProperty("nativeSourceEnd", note.nativeSourceEndSeconds);
                n->setProperty("materialSourceStart", note.materialSourceStartSeconds); n->setProperty("materialSourceEnd", note.materialSourceEndSeconds);
                n->setProperty("audioId", note.materialAudioId); n->setProperty("regionId", note.materialRegionId);
                n->setProperty("localOverride", note.materialLocalOverride); n->setProperty("unresolvedReference", note.materialReferenceUnresolved);
                n->setProperty("nativeProvenance", note.nativeProvenance); n->setProperty("role", nativeSegmentRoleName(note.nativeRole));
                n->setProperty("vowelSplitsExplicit", note.vowelSplitsExplicit); n->setProperty("consonantSeconds", note.consonantSeconds);
                n->setProperty("preutteranceOverride", note.utauPreutteranceOverrideEnabled); n->setProperty("preutteranceSeconds", note.utauPreutteranceSeconds);
                n->setProperty("attackSpeed", note.attackSpeed); n->setProperty("pitchContourFrames", int(note.contour.size()));
                juce::Array<juce::var> segments;
                for (const auto& segment : note.nativeSegments)
                {
                    auto s = std::make_unique<juce::DynamicObject>(); s->setProperty("id", segment.id); s->setProperty("alias", segment.alias);
                    s->setProperty("role", nativeSegmentRoleName(segment.role)); s->setProperty("start", segment.sourceStartSeconds);
                    s->setProperty("end", segment.sourceEndSeconds); s->setProperty("P", segment.alignmentSeconds);
                    s->setProperty("provenance", segment.provenance); s->setProperty("stretchable", segment.stretchable);
                    s->setProperty("weight", segment.stretchWeight); segments.add(juce::var(s.release()));
                }
                n->setProperty("rawNoteLocalSegments", juce::var(segments)); notes.add(juce::var(n.release()));
            }
            record->setProperty("notes", juce::var(notes)); owners.add(juce::var(record.release()));
        };
        if (parent.parts.empty()) append(parent, {}); else for (const auto& part : parent.parts) append(part, part.id);
    }
    root->setProperty("owners", juce::var(owners));
    if (juce::SHA256(originalProject).toHexString() != projectSha) { error = "Original project changed during read-only audit"; return false; }
    for (const auto& file : hashes)
        if (file.second != "MISSING" && juce::SHA256(juce::File(file.first)).toHexString() != file.second)
        { error = "Original source changed during read-only audit"; return false; }
    const auto directory = outputPrefix.getChildFile("registration-evidence-" + juce::Uuid().toString());
    if (directory.exists() || directory.createDirectory().failed()) { error = "Cannot create new audit output child"; return false; }
    const auto path = directory.getChildFile("evidence.json");
    const auto text = juce::JSON::toString(juce::var(root.release()));
    juce::FileOutputStream output(path);
    if (!output.openedOk() || !output.writeText(text, false, false, "\n")) { error = "Cannot write audit report"; return false; }
    output.flush(); if (output.getStatus().failed()) { error = output.getStatus().getErrorMessage(); return false; }
    reportFile = path; return true;
}
}
