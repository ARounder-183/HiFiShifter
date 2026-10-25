#include "ProjectMaterialRegistrationFacts.h"
#include "NativeNoteTiming.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace hachi
{
ProjectMaterialRegistrationFacts::Result ProjectMaterialRegistrationFacts::derive(
    const ProjectData& project, Policy policy, const ProjectMaterialRegistration::OwnerScope& scope)
{
    using R = ProjectMaterialRegistration;
    Result result;
    if (policy != Policy::ConfirmedCarriedPlacements)
    {
        result.diagnostics.add("Pronunciation grouping is unresolved. Explicit human confirmation of carried-placement annotation variants is required; project notes and analysis frames are not inferred pronunciations.");
        return result;
    }
    std::vector<R::Locator> selected; juce::String selectionError;
    if (!R::resolveOwnerScope(project, scope, selected, selectionError))
    { result.diagnostics.add(selectionError); return result; }
    const auto includesOwner = [&](const juce::String& track, const juce::String& parent, const juce::String& part)
    { return std::any_of(selected.begin(), selected.end(), [&](const auto& id)
        { return id.trackId == track && id.parentClipId == parent && id.sourcePartId == part; }); };
    std::set<juce::String> variants, noteIds;
    struct OwnerFacts
    {
        const ClipData* parent;
        const ClipData* owner;
        std::vector<const NoteData*> members;
    };
    std::vector<OwnerFacts> owners;
    for (const auto& track : project.tracks)
    {
        if (!trackShowsAllNativeRegions(track) || track.accompaniment) continue;
        for (const auto& parent : track.clips)
        {
            const auto append = [&](const ClipData& owner, const juce::String& partId)
            {
                if (!includesOwner(track.id, parent.id, partId)) return;
                if (owner.sourceFile == juce::File{}) return;
                const auto key = "confirmed-carried:" + track.id + ":" + parent.id + ":" + partId;
                const auto reject = [&](const juce::String& why) { result.diagnostics.add(key + ": " + why); };
                if (!variants.insert(key).second || !owner.parts.empty())
                { reject("Duplicate or nested source owner requires an explicit ownership adapter"); return; }
                if (owner.materialSourceAnalysisPending || owner.nativeSourcePitchPending)
                { reject("Finish pending source analysis before taking a carried snapshot"); return; }
                // Check raw facts before calling the shared clock projection. No
                // registration-specific repair/clamp of a reversed/invalid clock.
                for (std::size_t i = 0; i < owner.sourceTimeMap.size(); ++i)
                {
                    const auto& p = owner.sourceTimeMap[i];
                    if (!std::isfinite(p.targetSeconds) || !std::isfinite(p.sourceSeconds)
                        || (i && (p.targetSeconds < owner.sourceTimeMap[i - 1].targetSeconds
                            || p.sourceSeconds < owner.sourceTimeMap[i - 1].sourceSeconds)))
                    { reject("Invalid/reversed carried source clock; no guessed reconstruction"); return; }
                }
                const auto clock = nativeClipClock(owner);
                if (clock.size() < 2) { reject("Missing carried source clock"); return; }
                R::VariantFacts variant; variant.recording = owner.sourceFile; variant.variantKey = key;
                variant.explicitPronunciationEvidence = true; // explicit confirmed policy, not inferred syllables
                std::vector<const NoteData*> members;
                for (const auto& note : parent.notes) if (note.clipPartId == partId)
                {
                    if (note.id.isEmpty() || !noteIds.insert(note.id).second)
                    { reject("Missing/duplicate carried note identity"); return; }
                    auto local = note; local.startSeconds -= partId.isEmpty() ? 0.0 : owner.startSeconds;
                    R::RowFacts fact; fact.key = note.id;
                    fact.annotation = nativeNoteSourceTiming(local, owner, clock);
                    fact.annotation.name = "-";
                    fact.annotation.role = note.nativeRole; fact.annotation.provenance = "confirmed-carried-placement";
                    fact.annotation.confidence = note.nativeConfidence; fact.annotation.nativeUnpitched = note.nativeUnpitched;
                    // Independent P and C come from the SAME existing timing
                    // projection as local editing; never use voiced-centre cuts.
                    fact.annotation.overlapSeconds = note.utauOverlapOverrideEnabled || std::abs(note.utauOverlapSeconds) > 1.e-12
                        ? note.utauOverlapSeconds : note.nativeSegments.empty() ? 0.0 : note.nativeSegments.front().overlapSeconds;
                    // Only the existing explicit authoring bit permits named,
                    // weighted manual V. Estimated legacy epochs stay local and
                    // are canonically coalesced by the shared prepare service.
                    fact.annotation.vowelSplitsExplicit = note.vowelSplitsExplicit;
                    fact.annotation.pronunciationKind = note.pronunciationKind;
                    fact.members = {{track.id, parent.id, partId, note.id}};
                    variant.rows.push_back(std::move(fact)); members.push_back(&note);
                }
                if (variant.rows.empty())
                {
                    // An existing physical playback window is itself a carried
                    // placement; no fabricated pitch/phoneme/named subdivision.
                    R::RowFacts fact; fact.key = key + ":owner-playback-window";
                    fact.annotation.name = "-"; fact.annotation.provenance = "confirmed-carried-placement";
                    fact.annotation.regionStartSeconds = owner.sourceOffsetSeconds
                        + nativeSourceTimeAt(clock, owner.audioStartSeconds);
                    fact.annotation.regionEndSeconds = owner.sourceOffsetSeconds
                        + nativeSourceTimeAt(clock, owner.audioStartSeconds + owner.audioLength());
                    fact.annotation.alignmentSeconds = fact.annotation.regionStartSeconds;
                    variant.rows.push_back(std::move(fact));
                }
                owners.push_back({&parent, &owner, std::move(members)});
                result.variants.push_back(std::move(variant));
            };
            // Refuse orphan note ownership rather than omitting a carried piece.
            for (const auto& note : parent.notes)
            {
                if (scope && !includesOwner(track.id, parent.id, parent.parts.empty() ? juce::String{} : note.clipPartId)) continue;
                int owners = parent.parts.empty() ? (note.clipPartId.isEmpty() ? 1 : 0) : 0;
                if (!parent.parts.empty()) for (const auto& part : parent.parts) if (part.id == note.clipPartId) ++owners;
                if (owners != 1) result.diagnostics.add("Missing/ambiguous carried part ownership: " + note.id);
            }
            if (parent.parts.empty()) append(parent, {});
            else for (const auto& part : parent.parts) append(part, part.id);
        }
    }
    // An owner is already an explicit carried placement. Join owners ONLY from
    // an explicit source-owner group or an unambiguous material source-row link.
    // NativeConnection/connectedToNext are pitch/amplitude edits, not this graph.
    std::vector<std::size_t> groups(owners.size());
    for (std::size_t i = 0; i < groups.size(); ++i) groups[i] = i;
    const auto root = [&](std::size_t i)
    { while (groups[i] != i) i = groups[i]; return i; };
    const auto join = [&](std::size_t a, std::size_t b)
    { a = root(a); b = root(b); if (a != b) groups[b] = a; };
    for (std::size_t i = 0; i < owners.size(); ++i)
        for (std::size_t j = i + 1; j < owners.size(); ++j)
            if (owners[i].owner->sourceFile == owners[j].owner->sourceFile
                && ((owners[i].parent == owners[j].parent && owners[i].parent->nativeAudioLinked)
                    || (owners[i].owner->materialSourceUsageGroupId.isNotEmpty()
                        && owners[i].owner->materialSourceUsageGroupId == owners[j].owner->materialSourceUsageGroupId)))
                join(i, j);
    std::set<const NoteData*> sourceLinkedFrom, sourceLinkedTo;
    for (std::size_t i = 0; i < owners.size(); ++i)
        for (std::size_t r = 0; r < owners[i].members.size(); ++r)
        {
            const auto& note = *owners[i].members[r];
            if (note.materialConnectedNextRegionId.isEmpty()) continue;
            std::vector<std::pair<std::size_t, std::size_t>> candidates, local, inUsage;
            for (std::size_t j = 0; j < owners.size(); ++j)
                for (std::size_t s = 0; s < owners[j].members.size(); ++s)
                {
                    const auto& next = *owners[j].members[s];
                    if (next.materialRegionId != note.materialConnectedNextRegionId
                        || next.materialAudioId != note.materialAudioId) continue;
                    candidates.emplace_back(j, s);
                    if (j == i) local.emplace_back(j, s);
                    if (root(j) == root(i)) inUsage.emplace_back(j, s);
                }
            // A row link inside its existing placement does not refer to every
            // repeated placement of that library row. Cross-owner resolution
            // requires exactly one endpoint; no nearest-time partner selection.
            if (!local.empty()) candidates = std::move(local);
            else if (!inUsage.empty()) candidates = std::move(inUsage);
            if (note.materialAudioId.isEmpty() || candidates.size() != 1)
            {
                result.diagnostics.add("Missing/ambiguous explicit source-link endpoint (or endpoint outside selected OwnerScope): " + note.id);
                continue;
            }
            const auto [j, s] = candidates.front();
            if (owners[i].owner->sourceFile != owners[j].owner->sourceFile)
            {
                result.diagnostics.add("Explicit source-use grouping requires the same resolved recording path: " + note.id);
                continue;
            }
            const auto* next = owners[j].members[s];
            result.variants[i].rows[r].nextRowKey = next->id;
            sourceLinkedFrom.insert(&note); sourceLinkedTo.insert(next); join(i, j);
        }
    for (std::size_t i = 0; i < owners.size(); ++i)
    {
        const auto& view = owners[i];
        // Legacy Melodyne imports retained these source-continuity flags but
        // dropped followingJoin.followingElement identity. They cannot identify a
        // partner by themselves. Keep the legacy owner fallback, but never
        // manufacture a continuous usage from temporal adjacency/pitch joins.
        const bool outgoing = view.owner->glideConnectedToNext;
        const bool incoming = view.owner->glideConnectedFromPrevious;
        const bool explicitGroup = view.parent->nativeAudioLinked || view.owner->materialSourceUsageGroupId.isNotEmpty();
        const bool hasOutgoing = std::any_of(view.members.begin(), view.members.end(),
            [&](const auto* n) { return sourceLinkedFrom.count(n) != 0; });
        const bool hasIncoming = std::any_of(view.members.begin(), view.members.end(),
            [&](const auto* n) { return sourceLinkedTo.count(n) != 0; });
        if (!explicitGroup && ((outgoing && !hasOutgoing) || (incoming && !hasIncoming)))
            result.warnings.addIfNotAlreadyThere("Legacy Melodyne source-use endpoints are absent; retaining owner variants without guessing groups from glide flags or temporal adjacency.");
    }
    std::vector<R::VariantFacts> grouped;
    std::map<std::size_t, std::size_t> outputGroups;
    for (std::size_t i = 0; i < owners.size(); ++i)
    {
        const auto group = root(i);
        const auto found = outputGroups.find(group);
        if (found == outputGroups.end())
        {
            outputGroups[group] = grouped.size();
            grouped.push_back(std::move(result.variants[i]));
        }
        else
        {
            auto& destination = grouped[found->second];
            destination.variantKey += "+" + result.variants[i].variantKey;
            for (auto& row : result.variants[i].rows) destination.rows.push_back(std::move(row));
        }
    }
    result.variants = std::move(grouped);
    if (result.variants.empty()) result.diagnostics.add("No eligible carried native physical-source placements");
    result.ready = !result.variants.empty() && result.diagnostics.isEmpty();
    return result;
}
}
