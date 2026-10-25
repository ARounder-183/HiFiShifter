#pragma once
#include "TeioProjectMaterialRegistrationAudioGate.h"

namespace hachi
{
// Preserve the parent CLI spelling while keeping ONE authoritative real gate.
inline bool runRegisteredProjectAudioSmoke(const juce::File& originalProject,
    const juce::File& outputPrefix, const juce::File& nsfModel)
{
    return runTeioProjectMaterialRegistrationAudioGate(originalProject, outputPrefix, nsfModel);
}
}
