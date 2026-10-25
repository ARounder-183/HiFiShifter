#pragma once
#include <juce_core/juce_core.h>

namespace hachi
{
// Dependency-neutral reference DTOs: ProjectModel need not include the service
// (which itself consumes ProjectData/SampleRegionSetting). No audio/edit payload.
struct ProjectMaterialRegistrationLocator
{
    juce::String trackId, parentClipId, sourcePartId, noteId;
};
struct ProjectMaterialRegistrationOwnerBinding
{
    ProjectMaterialRegistrationLocator where; // noteId must be empty
    juce::File oldSource, copiedSource;
    juce::String sourceSha256;
};
struct ProjectMaterialRegistrationNoteBinding
{
    ProjectMaterialRegistrationLocator where;
    juce::String variantKey, rowKey;
    juce::File oldSource, copiedSource;
    juce::String sourceSha256;
    juce::String oldAudioId, oldRegionId, audioId, regionId;
    // Published authoring-row bounds for ID validation ONLY. Never overwrite
    // the note's existing/missing materialSourceStart/End (effective P origin).
    double sourceStart = 0, sourceEnd = 0;
    // Registration carries the old note's complete source/musical authority;
    // source authoring normalization must not reapply over that exact snapshot.
    bool preserveLocalAuthority = true;
};
}
