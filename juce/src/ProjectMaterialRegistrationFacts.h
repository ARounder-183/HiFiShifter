#pragma once
#include "ProjectMaterialRegistration.h"

namespace hachi
{
// Pure snapshot-to-facts adapter. The opt-in policy is a USER-CONFIRMED carried
// placement interpretation, NOT phonetic recovery or automatic annotation.
class ProjectMaterialRegistrationFacts final
{
public:
    enum class Policy { unresolved, ConfirmedCarriedPlacements };
    struct Result
    {
        std::vector<ProjectMaterialRegistration::VariantFacts> variants;
        juce::StringArray diagnostics;
        juce::StringArray warnings; // non-blocking missing legacy source-use provenance
        bool ready = false;
    };
    // Default deliberately refuses to turn project notes into pronunciation rows.
    // Confirmed policy: explicit source-usage group / linked source owners -> one
    // annotation variant, with one unnamed row per actual carried note placement.
    // Unlinked legacy owners remain independent (missing glide endpoints warn).
    // Never splits filenames, analysis segments/F0 frames, or guesses join partners.
    // Use the SAME scope when passing the returned facts to prepare. Selection
    // resolution is shared; present-empty never widens back to all owners.
    static Result derive(const ProjectData&, Policy = Policy::unresolved,
        const ProjectMaterialRegistration::OwnerScope& selectedOwners = std::nullopt);
};
}
