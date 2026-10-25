#include "MaterialAnnotationSession.h"
#include "OttoMaterialMigration.h"
#include "NativeSharedEnvelope.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <set>

namespace hachi
{
namespace
{
juce::String newId(char prefix)
{
    return juce::String::charToString(prefix) + juce::Uuid().toString().removeCharacters("-");
}
bool rejected(juce::String& error, const juce::String& text) { error = text; return false; }
bool elasticVowel(const NativeSegment& segment)
{ return segment.role == NativeSegmentRole::vowel && segment.stretchable; }
void canonicalDraftSegments(SampleRegionSetting& row)
{
    auto segments = pronunciationSegmentsFor(row); // Shared core returns local source coordinates.
    for (auto& segment : segments)
    {
        segment.sourceStartSeconds += row.regionStartSeconds;
        segment.sourceEndSeconds += row.regionStartSeconds;
        segment.alignmentSeconds += row.regionStartSeconds;
    }
    row.segments = std::move(segments);
}
std::vector<NativeSegment> clippedSegments(const SampleRegionSetting& source, double start, double end)
{
    std::vector<NativeSegment> result;
    for (auto s : source.segments)
    {
        s.sourceStartSeconds = std::max(start, s.sourceStartSeconds);
        s.sourceEndSeconds = std::min(end, s.sourceEndSeconds);
        if (s.sourceEndSeconds <= s.sourceStartSeconds + 1.e-9) continue;
        s.alignmentSeconds = std::clamp(s.alignmentSeconds, s.sourceStartSeconds, s.sourceEndSeconds);
        result.push_back(std::move(s));
    }
    return result;
}
}

std::unique_ptr<MaterialAnnotationSession> MaterialAnnotationSession::open(
    const juce::File& audio, const juce::File& file, juce::String& error,
    const juce::String& selectedAudioId)
{
    error.clear();
    juce::AudioFormatManager formats; formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(audio));
    if (!reader || !std::isfinite(reader->sampleRate) || reader->sampleRate <= 0
        || reader->lengthInSamples <= 0 || reader->numChannels == 0)
    { error = "Cannot read material audio: " + audio.getFullPathName(); return {}; }
    auto session = std::unique_ptr<MaterialAnnotationSession>(new MaterialAnnotationSession());
    session->audioFile = audio; session->manifest = file;
    session->duration = static_cast<double>(reader->lengthInSamples) / reader->sampleRate;
    OttoMaterialIO::Document document;
    if (file.exists() && !OttoMaterialIO::loadDocument(file, document, error)) return {};
    std::vector<OttoMaterialIO::Audio> variants;
    if (file.exists() && !OttoMaterialIO::listAudioVariants(file, audio, variants, error)) return {};
    if (selectedAudioId.isEmpty() && variants.size() > 1)
    { error = "Ambiguous audio annotation variants; select audioId"; return {}; }
    if (!variants.empty() || (file.exists() && selectedAudioId.isNotEmpty()))
    {
        session->audioId = selectedAudioId.isNotEmpty() ? selectedAudioId : variants.front().id;
        if (!OttoMaterialIO::loadAudio(file, audio, session->draft, error,
                                      &session->fingerprint, session->audioId)) return {};
    }
    else
    {
        // OTTO owns the folder even without this audio section. Otherwise read
        // all matching classic OTO aliases without publishing/converting files.
        // Bare audio starts empty: the user selects a range and creates an alias.
        if (file.exists()) session->draft.clear();
        else if (!OttoMaterialMigration::readAudioAnnotations(audio, file.getParentDirectory(),
                     UtauMode::classic, session->draft, error, selectedAudioId)) return {};
        if (!OttoMaterialIO::replaceAudio(document, file, audio, session->draft, error, selectedAudioId)) return {};
        // There was no matching section: replaceAudio appended exactly this one.
        session->draft = document.audio.back().rows;
        session->audioId = document.audio.back().id;
        session->unpublishedEmptySource = session->draft.empty();
        // Derive the exact observed token without publishing anything.
        session->fingerprint = document.existed
            ? "v1:" + juce::SHA256(document.fingerprint).toHexString() : juce::String("absent");
    }
    for (auto& row : session->draft)
    {
        row.materialAudioId = session->audioId;
        if (row.materialRegionId.isEmpty()) row.materialRegionId = newId('p');
        const auto previousCount = row.segments.size();
        canonicalDraftSegments(row);
        if (!row.vowelSplitsExplicit && previousCount != row.segments.size())
        {
            if (session->normalizationMessage.isNotEmpty()) session->normalizationMessage += "\n";
            session->normalizationMessage += row.name + ": automatic cuts normalized in draft ("
                + juce::String(static_cast<int>(previousCount)) + " -> " + juce::String(static_cast<int>(row.segments.size()))
                + "); source library unchanged until explicit Save.";
        }
    }
    if (!session->validate(session->draft, error)) return {};
    session->cleanDraft = session->draft;
    return session;
}

bool MaterialAnnotationSession::encodeDraft(const std::vector<SampleRegionSetting>& rows,
    juce::String& text, juce::String& error) const
{
    OttoMaterialIO::Document document;
    document.audio.push_back({audioId, audioFile.getRelativePathFrom(manifest.getParentDirectory()).replaceCharacter('\\', '/'), rows});
    // Local timing does not author the source graph. Keep its existing link in
    // the callback DTO, but do not require a private row to retime its neighbour.
    if (scope == EditScope::thisNote)
        for (auto& row : document.audio.front().rows) row.connectedNextRegionId.clear();
    return OttoMaterialIO::encode(document, text, error);
}

bool MaterialAnnotationSession::dirty() const
{
    juce::String current, baseline, error;
    return !encodeDraft(draft, current, error) || !encodeDraft(cleanDraft, baseline, error) || current != baseline;
}

bool MaterialAnnotationSession::enableNoteScope(NoteScopeBinding binding, juce::String& error)
{
    error.clear();
    if (scope != EditScope::source || dirty()) return rejected(error, "Finish or explicitly discard the current draft before binding a note");
    if (!binding.apply || binding.noteId.isEmpty() || binding.sourceFingerprint.isEmpty()
        || binding.sourceAudio != audioFile || binding.audioId != audioId
        || binding.regionId.isEmpty() || binding.localRow.materialAudioId != binding.audioId
        || binding.localRow.materialRegionId != binding.regionId)
        return rejected(error, "Local scope requires an explicit note, revision, source fingerprint and matching source identities");
    const auto found = std::find_if(draft.begin(), draft.end(), [&](const auto& row)
    { return row.materialAudioId == binding.audioId && row.materialRegionId == binding.regionId; });
    if (found == draft.end()) return rejected(error, "Local note has no pronunciation in the governing source; reload its binding");
    noteBinding = std::move(binding);
    noteScopeRefreshError.clear();
    return true;
}

bool MaterialAnnotationSession::refreshNoteScopeBinding(juce::String& error)
{
    error.clear();
    if (scope != EditScope::source || dirty())
        return rejected(error, "A note ticket can refresh only in clean source scope; local/dirty drafts retain their revision");
    if (!noteBinding.refresh) return rejected(error, "No read-only note binding refresh adapter is available");
    const auto rejectRefresh = [&](const juce::String& why)
    { noteScopeRefreshError = why; error = why; return false; };
    NoteScopeBinding refreshed;
    if (!noteBinding.refresh(noteBinding, refreshed, error))
        return rejectRefresh(error.isNotEmpty() ? error : juce::String("The published source no longer has a resolved note binding; reopen its context"));
    if (refreshed.noteId != noteBinding.noteId || refreshed.audioId != noteBinding.audioId
        || refreshed.regionId != noteBinding.regionId || refreshed.sourceAudio != noteBinding.sourceAudio
        || refreshed.sourceFingerprint != noteBinding.sourceFingerprint || refreshed.capability != noteBinding.capability)
        return rejectRefresh("Note/source identity or physical fingerprint changed; reopen the exact note context");
    if (!enableNoteScope(std::move(refreshed), error)) return rejectRefresh(error);
    return true; // Binding only: source draft, view, history and music unchanged.
}

bool MaterialAnnotationSession::setEditScope(EditScope target, juce::String& error)
{ return switchScope(target, false, error); }
bool MaterialAnnotationSession::discardAndSetEditScope(EditScope target, juce::String& error)
{ return switchScope(target, true, error); }

bool MaterialAnnotationSession::switchScope(EditScope target, bool discard, juce::String& error)
{
    error.clear();
    if (target == scope && !discard) return true;
    if (!discard && dirty()) return rejected(error, "Unpublished edits: apply/save or explicitly discard before switching scope");
    if (target == EditScope::source && scope == EditScope::source && discard && unpublishedEmptySource)
    {
        // No backing section exists yet. Discard this unpublished draft only;
        // retain its minted identity and original conflict token, never reopen
        // by path or substitute a newly appeared registration.
        draft = cleanDraft; undoStates.clear(); redoStates.clear();
        return true;
    }
    if (target == EditScope::thisNote)
    {
        if (!hasNoteScope()) return rejected(error, "No project-only note adapter is available");
        if (noteScopeRefreshError.isNotEmpty()) return rejected(error, noteScopeRefreshError);
        auto row = noteBinding.localRow;
        canonicalDraftSegments(row);
        const auto previous = scope;
        scope = EditScope::thisNote;
        if (!validate({row}, error)) { scope = previous; return false; }
        draft = {std::move(row)};
        normalizationMessage.clear();
    }
    else
    {
        // Reload the actual source, never promote a private note snapshot into
        // source authority. Existing OTTO (including absent rows) still governs.
        auto fresh = open(audioFile, manifest, error, audioId);
        if (!fresh) return false;
        scope = EditScope::source;
        draft = std::move(fresh->draft); audioId = fresh->audioId;
        fingerprint = fresh->fingerprint; duration = fresh->duration;
        normalizationMessage = fresh->normalizationMessage;
    }
    cleanDraft = draft; undoStates.clear(); redoStates.clear();
    return true;
}

bool MaterialAnnotationSession::restoreLocal(juce::String& error)
{
    error.clear();
    if (scope != EditScope::thisNote || !canRestoreLocal()) return rejected(error, "No local restore adapter is available");
    if (dirty()) return rejected(error, "Apply or explicitly discard the draft before restoring source settings");
    auto restored = noteBinding.localRow;
    auto revision = noteBinding.projectRevision;
    if (!noteBinding.restore(noteBinding, restored, revision, error)) return false;
    if (!validate({restored}, error)) return false;
    // The host validates its returned row before committing. Do not create a
    // second project transaction or write a manifest from this callback path.
    noteBinding.localRow = restored; noteBinding.projectRevision = revision;
    draft = {std::move(restored)}; cleanDraft = draft;
    undoStates.clear(); redoStates.clear();
    return true;
}

bool MaterialAnnotationSession::validate(const std::vector<SampleRegionSetting>& rows, juce::String& error) const
{
    if (scope == EditScope::thisNote)
    {
        if (rows.size() != 1 || rows.front().materialAudioId != noteBinding.audioId
            || rows.front().materialRegionId != noteBinding.regionId
            || rows.front().connectedNextRegionId != noteBinding.localRow.connectedNextRegionId)
            return rejected(error, "This-note scope cannot create/delete/split source pronunciations, edit other rows or author source links");
        if (isClassicNoteScope())
        {
            backend::UtauOtoOverride scalar;
            if (!ProjectModel::makeClassicNoteMaterialOto(rows.front(), scalar, error)) return false;
            if (rows.front().name != noteBinding.localRow.name
                || rows.front().additionalAliases != noteBinding.localRow.additionalAliases)
                return rejected(error, "Classic note OTO stores scalar timing only, not source aliases");
        }
    }
    for (const auto& row : rows)
    {
        if (row.regionEndSeconds > duration + 1.e-7)
            return rejected(error, "Pronunciation extends beyond the source audio");
        if (row.vowelSplitsExplicit)
            for (const auto& segment : row.segments) if (elasticVowel(segment))
                if (segment.alias.trim().isEmpty() || !std::isfinite(segment.stretchWeight) || segment.stretchWeight <= 0)
                    return rejected(error, "Manual vowels require explicit nonempty names and positive finite stretch weights");
    }
    juce::String encoded;
    return encodeDraft(rows, encoded, error);
}

bool MaterialAnnotationSession::commit(std::vector<SampleRegionSetting> next, juce::String& error)
{
    error.clear();
    for (auto& row : next)
    {
        const auto old = std::find_if(draft.begin(), draft.end(), [&](const auto& value) { return value.materialRegionId == row.materialRegionId; });
        const bool timingChanged = old == draft.end() || row.regionStartSeconds != old->regionStartSeconds
            || row.regionEndSeconds != old->regionEndSeconds || row.alignmentSeconds != old->alignmentSeconds
            || row.fixedDurationSeconds != old->fixedDurationSeconds || row.overlapSeconds != old->overlapSeconds
            || row.simplifiedConsonant != old->simplifiedConsonant;
        if (!timingChanged)
        {
            if (!row.vowelSplitsExplicit) canonicalDraftSegments(row);
            continue; // Kind/alias-only publication never retimes source markers.
        }
        if (!std::isfinite(row.regionStartSeconds) || !std::isfinite(row.regionEndSeconds)
            || !std::isfinite(row.alignmentSeconds) || !std::isfinite(row.fixedDurationSeconds)
            || !std::isfinite(row.overlapSeconds) || row.regionStartSeconds < 0
            || row.regionEndSeconds <= row.regionStartSeconds || row.regionEndSeconds > duration)
            return rejected(error, "Pronunciation timing must be finite and inside the unchanged recording");
        auto timing = pronunciationTimingFor(row);
        if (old != draft.end() && old->simplifiedConsonant && row.simplifiedConsonant
            && row.alignmentSeconds != old->alignmentSeconds && row.fixedDurationSeconds == old->fixedDurationSeconds
            && row.regionStartSeconds == old->regionStartSeconds && row.regionEndSeconds == old->regionEndSeconds)
        {
            auto linked = *old;
            if (!editPronunciationAlignment(linked, row.alignmentSeconds, error)) return false;
            timing.consonantEndSeconds = pronunciationTimingFor(linked).consonantEndSeconds;
        }
        if (!applyPronunciationTiming(row, timing, error)) return false;
    }
    if (!validate(next, error)) return false;
    juce::String oldText, newText;
    if (!encodeDraft(draft, oldText, error) || !encodeDraft(next, newText, error)) return false;
    if (oldText == newText) return true;
    if (undoStates.size() >= 100) undoStates.erase(undoStates.begin());
    undoStates.push_back(draft); redoStates.clear(); draft = std::move(next); return true;
}

bool MaterialAnnotationSession::createPronunciation(double start, double end, const juce::String& alias, juce::String& error)
{
    return insertPronunciation(start, end, alias, {}, false, error);
}

bool MaterialAnnotationSession::previewAdjacentRange(const juce::String& anchorId, bool before, double freeEdge,
    double& start, double& end, juce::String& error) const
{
    error.clear();
    const auto anchor = std::find_if(draft.begin(), draft.end(), [&](const auto& row) { return row.materialRegionId == anchorId; });
    if (anchor == draft.end() || !std::isfinite(freeEdge)) return rejected(error, "Select an existing pronunciation as the adjacent anchor");
    const auto join = before ? anchor->regionStartSeconds : anchor->regionEndSeconds;
    double bound = before ? 0.0 : duration;
    for (const auto& row : draft)
    {
        if (row.materialRegionId == anchorId) continue;
        if (before)
        {
            if (row.regionStartSeconds < join && row.regionEndSeconds > join)
                return rejected(error, "The anchor's left join is already occupied");
            if (row.regionEndSeconds <= join) bound = std::max(bound, row.regionEndSeconds);
        }
        else
        {
            if (row.regionStartSeconds < join && row.regionEndSeconds > join)
                return rejected(error, "The anchor's right join is already occupied");
            if (row.regionStartSeconds >= join) bound = std::min(bound, row.regionStartSeconds);
        }
    }
    start = before ? juce::jlimit(bound, join, freeEdge) : join;
    end = before ? join : juce::jlimit(join, bound, freeEdge);
    if (end - start < .001 - 1.e-9) return rejected(error, "No positive free adjacent source range is available on this side");
    for (const auto& row : draft)
        if (row.materialRegionId != anchorId && start < row.regionEndSeconds - 1.e-9 && end > row.regionStartSeconds + 1.e-9)
            return rejected(error, "Adjacent pronunciation cannot overlap an existing pronunciation");
    return true;
}

bool MaterialAnnotationSession::createAdjacentPronunciation(const juce::String& anchorId, bool before,
    double freeEdge, const juce::String& alias, juce::String& error)
{
    double start = 0, end = 0;
    if (!previewAdjacentRange(anchorId, before, freeEdge, start, end, error)) return false;
    return insertPronunciation(start, end, alias, anchorId, before, error);
}

bool MaterialAnnotationSession::insertPronunciation(double start, double end, const juce::String& alias,
    const juce::String& adjacentAnchor, bool before, juce::String& error)
{
    error.clear(); const auto name = alias.trim();
    if (!std::isfinite(start) || !std::isfinite(end) || start < 0 || end > duration || end <= start || name.isEmpty())
        return rejected(error, "Select a valid source region and enter a pronunciation alias");
    auto next = draft;
    for (auto& row : next)
        if (std::abs(row.regionStartSeconds-start) < 1.e-9 && std::abs(row.regionEndSeconds-end) < 1.e-9)
        {
            if (row.name != name) row.additionalAliases.addIfNotAlreadyThere(name);
            return commit(std::move(next), error);
        }
    SampleRegionSetting row; row.materialAudioId = audioId; row.materialRegionId = newId('p');
    row.name = name; row.regionStartSeconds = start; row.regionEndSeconds = end; row.alignmentSeconds = start;
    row.provenance = "user"; row.confidence = 1;
    row.role = NativeSegmentRole::vowel;
    row.segments = {{"segment_1", name, NativeSegmentRole::vowel, start, end, "user", 1.f, start, 0, true, 1}};
    if (adjacentAnchor.isNotEmpty())
    {
        auto anchor = std::find_if(next.begin(), next.end(), [&](const auto& value) { return value.materialRegionId == adjacentAnchor; });
        if (anchor == next.end()) return rejected(error, "Adjacent anchor no longer exists");
        if (before) row.connectedNextRegionId = anchor->materialRegionId;
        else
        {
            if (anchor->connectedNextRegionId.isNotEmpty()) return rejected(error, "Adjacent anchor already has a connected successor");
            anchor->connectedNextRegionId = row.materialRegionId;
        }
        int neighbors = 0;
        for (auto& neighbor : next)
        {
            if (neighbor.materialRegionId == adjacentAnchor) continue;
            if (before && std::abs(neighbor.regionEndSeconds - start) < 1.e-9)
            {
                if (++neighbors > 1 || neighbor.connectedNextRegionId.isNotEmpty())
                    return rejected(error, "Ambiguous or already-connected upstream pronunciation");
                neighbor.connectedNextRegionId = row.materialRegionId;
            }
            if (!before && std::abs(neighbor.regionStartSeconds - end) < 1.e-9)
            {
                if (++neighbors > 1) return rejected(error, "Ambiguous downstream pronunciation");
                row.connectedNextRegionId = neighbor.materialRegionId;
            }
        }
    }
    next.push_back(std::move(row));
    std::stable_sort(next.begin(), next.end(), [](const auto& a,const auto& b){return a.regionStartSeconds < b.regionStartSeconds;});
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::removeAlias(const juce::String& id, const juce::String& alias, juce::String& error)
{
    auto next = draft;
    const auto found = std::find_if(next.begin(),next.end(),[&](const auto& row){return row.materialRegionId==id;});
    if (found == next.end()) return rejected(error,"Pronunciation no longer exists");
    if (found->name == alias)
    {
        if (found->additionalAliases.isEmpty()) return erase({id},error);
        const auto old = found->name; found->name = found->additionalAliases[0]; found->additionalAliases.remove(0);
        for (auto& segment : found->segments)
            if (segment.alias == old && !(found->vowelSplitsExplicit && elasticVowel(segment))) segment.alias = found->name;
    }
    else found->additionalAliases.removeString(alias);
    return commit(std::move(next),error);
}

bool MaterialAnnotationSession::planSplitVowel(SampleRegionSetting& row, const juce::String& segmentId,
    double sourceCut, const juce::String& leftName, const juce::String& rightName, juce::String& error)
{
    error.clear();
    const auto leftAlias = leftName.trim(), rightAlias = rightName.trim();
    if (leftAlias.isEmpty() || rightAlias.isEmpty() || leftAlias == "-" || rightAlias == "-")
        return rejected(error, "Enter both manual vowel names before splitting; '-' is an unassigned placeholder");
    auto next = row; canonicalDraftSegments(next);
    const auto found = std::find_if(next.segments.begin(), next.segments.end(), [&](const auto& s) { return s.id == segmentId; });
    if (found == next.segments.end() || !elasticVowel(*found)) return rejected(error, "Select an existing elastic vowel segment");
    if (!std::isfinite(sourceCut) || sourceCut <= found->sourceStartSeconds + 1.e-6
        || sourceCut >= found->sourceEndSeconds - 1.e-6) return rejected(error, "Split must be strictly inside the selected physical vowel");
    auto right = *found; right.id = newId('s'); right.alias = rightAlias; right.sourceStartSeconds = sourceCut;
    found->sourceEndSeconds = sourceCut; found->alias = leftAlias;
    const auto index = static_cast<std::size_t>(found - next.segments.begin());
    next.segments.insert(next.segments.begin() + static_cast<std::ptrdiff_t>(index + 1), std::move(right));
    next.vowelSplitsExplicit = true; row = std::move(next); return true;
}

bool MaterialAnnotationSession::planEditVowel(SampleRegionSetting& row, const juce::String& segmentId,
    const juce::String& name, double weight, juce::String& error)
{
    error.clear(); const auto alias = name.trim();
    if (alias.isEmpty() || alias == "-" || !std::isfinite(weight) || weight <= 0)
        return rejected(error, "Enter an assigned vowel name (not '-') and positive finite stretch weight");
    auto next = row; canonicalDraftSegments(next);
    const auto found = std::find_if(next.segments.begin(), next.segments.end(), [&](const auto& s) { return s.id == segmentId; });
    if (found == next.segments.end() || !elasticVowel(*found)) return rejected(error, "Selected elastic vowel no longer exists");
    if (found->alias != alias || found->stretchWeight != weight)
    { found->alias = alias; found->stretchWeight = weight; next.vowelSplitsExplicit = true; }
    row = std::move(next); return true;
}

bool MaterialAnnotationSession::planMergeVowels(SampleRegionSetting& row, const juce::String& leftId, juce::String& error)
{
    error.clear(); auto next = row; canonicalDraftSegments(next);
    auto left = std::find_if(next.segments.begin(), next.segments.end(), [&](const auto& s) { return s.id == leftId; });
    if (left == next.segments.end() || !elasticVowel(*left) || left + 1 == next.segments.end()
        || !elasticVowel(*(left + 1)) || std::abs(left->sourceEndSeconds - (left + 1)->sourceStartSeconds) > 1.e-9)
        return rejected(error, "Only adjacent elastic vowel segments can merge");
    left->sourceEndSeconds = (left + 1)->sourceEndSeconds;
    next.segments.erase(left + 1);
    const auto vowels = std::count_if(next.segments.begin(), next.segments.end(), elasticVowel);
    const bool otherAuthoredZones = std::any_of(next.segments.begin(), next.segments.end(), [](const auto& s)
        { return s.role != NativeSegmentRole::consonant && s.role != NativeSegmentRole::vowel; });
    if (vowels == 1 && !otherAuthoredZones)
    {
        next.vowelSplitsExplicit = false;
        for (auto& segment : next.segments) if (elasticVowel(segment)) { segment.alias = next.name; segment.stretchWeight = 1; }
        canonicalDraftSegments(next);
    }
    row = std::move(next); return true;
}

bool MaterialAnnotationSession::splitVowel(const juce::String& id, const juce::String& segmentId, double sourceCut,
    const juce::String& leftName, const juce::String& rightName, juce::String& error)
{
    const auto found = std::find_if(draft.begin(), draft.end(), [&](const auto& row) { return row.materialRegionId == id; });
    if (found == draft.end()) return rejected(error, "Pronunciation no longer exists");
    auto row = *found;
    return planSplitVowel(row, segmentId, sourceCut, leftName, rightName, error) && setRow(id, row, error);
}
bool MaterialAnnotationSession::editVowel(const juce::String& id, const juce::String& segmentId,
    const juce::String& name, double weight, juce::String& error)
{
    const auto found = std::find_if(draft.begin(), draft.end(), [&](const auto& row) { return row.materialRegionId == id; });
    if (found == draft.end()) return rejected(error, "Pronunciation no longer exists");
    auto row = *found; return planEditVowel(row, segmentId, name, weight, error) && setRow(id, row, error);
}
bool MaterialAnnotationSession::mergeVowels(const juce::String& id, const juce::String& segmentId, juce::String& error)
{
    const auto found = std::find_if(draft.begin(), draft.end(), [&](const auto& row) { return row.materialRegionId == id; });
    if (found == draft.end()) return rejected(error, "Pronunciation no longer exists");
    auto row = *found; return planMergeVowels(row, segmentId, error) && setRow(id, row, error);
}

bool MaterialAnnotationSession::split(const std::vector<double>& sourceTimes, juce::String& error)
{
    error.clear(); auto cuts = sourceTimes;
    for (const auto t : cuts)
        if (!std::isfinite(t) || t <= 0 || t >= duration)
            return rejected(error, "Split points must be finite and inside the source audio");
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double a, double b) { return std::abs(a-b) < 1.e-9; }), cuts.end());
    auto next = draft;
    for (const auto cut : cuts)
    {
        std::size_t match = next.size(); int matches = 0;
        for (std::size_t i = 0; i < next.size(); ++i)
            if (cut > next[i].regionStartSeconds + 1.e-9 && cut < next[i].regionEndSeconds - 1.e-9)
            { match = i; ++matches; }
        if (matches > 1) return rejected(error, "Split is ambiguous between overlapping pronunciations; select one first");
        if (matches == 0)
        {
            const auto boundary = std::any_of(next.begin(), next.end(), [&](const auto& r)
                { return std::abs(cut-r.regionStartSeconds) < 1.e-9 || std::abs(cut-r.regionEndSeconds) < 1.e-9; });
            if (boundary) continue;
            return rejected(error, "Split falls outside the annotated pronunciation ranges");
        }
        const auto old = next[match];
        auto left = old, right = old;
        right.materialRegionId = newId('p');
        left.regionEndSeconds = cut; right.regionStartSeconds = cut;
        left.fixedDurationSeconds = std::min(old.fixedDurationSeconds, cut-old.regionStartSeconds);
        right.fixedDurationSeconds = std::max(0.0, old.regionStartSeconds + old.fixedDurationSeconds - cut);
        left.alignmentSeconds = std::clamp(old.alignmentSeconds, left.regionStartSeconds, left.regionEndSeconds);
        right.alignmentSeconds = std::clamp(old.alignmentSeconds, right.regionStartSeconds, right.regionEndSeconds);
        left.segments = clippedSegments(old, left.regionStartSeconds, left.regionEndSeconds);
        right.segments = clippedSegments(old, right.regionStartSeconds, right.regionEndSeconds);
        left.connectedNextRegionId = right.materialRegionId; // right keeps old successor.
        // Reuse the same interpolator and synthesized edge points as linked
        // project fragments, so cutting a ramp cannot change its heard gain.
        if (!old.amplitudeEnvelope.empty())
        {
            left.amplitudeEnvelope = sliceNativeSharedEnvelope(old.amplitudeEnvelope,
                0.0, cut-old.regionStartSeconds);
            right.amplitudeEnvelope = sliceNativeSharedEnvelope(old.amplitudeEnvelope,
                cut-old.regionStartSeconds, old.regionEndSeconds-old.regionStartSeconds);
        }
        next[match] = std::move(left); next.insert(next.begin()+static_cast<std::ptrdiff_t>(match+1), std::move(right));
    }
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::moveBoundary(const juce::String& leftId, double time, juce::String& error)
{
    auto next = draft;
    const auto left = std::find_if(next.begin(), next.end(), [&](const auto& row) { return row.materialRegionId == leftId; });
    if (left == next.end() || left->connectedNextRegionId.isEmpty())
        return rejected(error, "This boundary has no connected successor");
    const auto right = std::find_if(next.begin(), next.end(), [&](const auto& row) { return row.materialRegionId == left->connectedNextRegionId; });
    if (right == next.end() || !std::isfinite(time) || time <= left->regionStartSeconds + 1.e-6
        || time >= right->regionEndSeconds - 1.e-6)
        return rejected(error, "Boundary must stay inside its two adjacent pronunciations");
    const auto retime = [](SampleRegionSetting& row, double begin, double end)
    {
        const auto previous = row.regionStartSeconds;
        const auto scale = (end-begin)/(row.regionEndSeconds-previous);
        const auto at = [&](double value) { return begin + (value-previous)*scale; };
        row.fixedDurationSeconds *= scale;
        row.alignmentSeconds = at(row.alignmentSeconds);
        for (auto& segment : row.segments)
        {
            segment.sourceStartSeconds = at(segment.sourceStartSeconds);
            segment.sourceEndSeconds = at(segment.sourceEndSeconds);
            segment.alignmentSeconds = at(segment.alignmentSeconds);
        }
        for (auto& point : row.amplitudeEnvelope) point.timeSeconds *= scale;
        row.regionStartSeconds = begin; row.regionEndSeconds = end;
    };
    retime(*left, left->regionStartSeconds, time);
    retime(*right, time, right->regionEndSeconds);
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::rename(const juce::StringArray& names, juce::String& error)
{
    if (names.size() != static_cast<int>(draft.size())) return rejected(error, "Provide one primary alias per pronunciation");
    auto next = draft;
    for (std::size_t i = 0; i < next.size(); ++i)
    {
        const auto name = names[static_cast<int>(i)].trim();
        if (name.isEmpty()) return rejected(error, "A pronunciation alias cannot be empty");
        const auto old = next[i].name; next[i].name = name;
        for (auto& s : next[i].segments)
            if (s.alias == old && !(next[i].vowelSplitsExplicit && elasticVowel(s))) s.alias = name;
    }
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::setRow(const juce::String& id, const SampleRegionSetting& row, juce::String& error)
{
    auto next = draft;
    const auto found = std::find_if(next.begin(), next.end(), [&](const auto& r){ return r.materialRegionId == id; });
    if (found == next.end()) return rejected(error, "Pronunciation no longer exists");
    if (row.materialRegionId != id || row.materialAudioId != audioId)
        return rejected(error, "Editing cannot replace stable material identities");
    auto updated = row;
    const auto sameSegment = [](const auto& a, const auto& b)
    {
        return a.id == b.id && a.alias == b.alias && a.role == b.role
            && a.sourceStartSeconds == b.sourceStartSeconds && a.sourceEndSeconds == b.sourceEndSeconds
            && a.alignmentSeconds == b.alignmentSeconds && a.overlapSeconds == b.overlapSeconds
            && a.stretchable == b.stretchable && a.stretchWeight == b.stretchWeight
            && a.provenance == b.provenance && a.confidence == b.confidence;
    };
    if ((updated.regionStartSeconds != found->regionStartSeconds || updated.regionEndSeconds != found->regionEndSeconds)
        && updated.segments.size() == found->segments.size()
        && std::equal(updated.segments.begin(), updated.segments.end(), found->segments.begin(), sameSegment))
    {
        const auto oldLength = found->regionEndSeconds - found->regionStartSeconds;
        const auto newLength = updated.regionEndSeconds - updated.regionStartSeconds;
        for (auto& s : updated.segments)
        {
            const auto map = [&](double time) { return updated.regionStartSeconds
                + (time-found->regionStartSeconds) * newLength / oldLength; };
            s.sourceStartSeconds = map(s.sourceStartSeconds); s.sourceEndSeconds = map(s.sourceEndSeconds);
            s.alignmentSeconds = map(s.alignmentSeconds);
        }
    }
    const auto oldAlias = found->name;
    for (auto& s : updated.segments)
        if (s.alias == oldAlias && !(updated.vowelSplitsExplicit && elasticVowel(s))) s.alias = updated.name;
    *found = std::move(updated);
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::merge(const std::vector<juce::String>& ids, juce::String& error)
{
    if (ids.size() < 2) return rejected(error, "Select at least two adjacent pronunciations");
    const std::set<juce::String> selected(ids.begin(), ids.end());
    std::vector<std::size_t> positions;
    auto next = draft;
    std::stable_sort(next.begin(), next.end(), [](const auto& a, const auto& b) { return a.regionStartSeconds < b.regionStartSeconds; });
    for (std::size_t i = 0; i < next.size(); ++i) if (selected.count(next[i].materialRegionId)) positions.push_back(i);
    if (positions.size() != selected.size()) return rejected(error, "A selected pronunciation no longer exists");
    for (std::size_t i = 1; i < positions.size(); ++i)
        if (positions[i] != positions[i-1]+1
            || std::abs(next[positions[i-1]].regionEndSeconds-next[positions[i]].regionStartSeconds) > 1.e-7)
            return rejected(error, "Only adjacent, non-overlapping source pronunciations can be merged");
    const auto first = positions.front(), last = positions.back();
    auto merged = next[first];
    for (auto i = first+1; i <= last; ++i)
    {
        for (auto segment : next[i].segments)
        {
            if (std::any_of(merged.segments.begin(), merged.segments.end(), [&](const auto& s) { return s.id == segment.id; }))
                segment.id = newId('s');
            merged.segments.push_back(std::move(segment));
        }
        for (auto p : next[i].amplitudeEnvelope)
        { p.timeSeconds += next[i].regionStartSeconds-merged.regionStartSeconds; merged.amplitudeEnvelope.push_back(p); }
    }
    merged.regionEndSeconds = next[last].regionEndSeconds;
    merged.connectedNextRegionId = next[last].connectedNextRegionId;
    // Other pronunciation identities disappear only in the material draft.
    // Project publication preflight can block removal of referenced regions.
    for (auto& row : next)
        if (selected.count(row.connectedNextRegionId)) row.connectedNextRegionId = merged.materialRegionId;
    next.erase(next.begin()+static_cast<std::ptrdiff_t>(first), next.begin()+static_cast<std::ptrdiff_t>(last+1));
    next.insert(next.begin()+static_cast<std::ptrdiff_t>(first), std::move(merged));
    return commit(std::move(next), error);
}

bool MaterialAnnotationSession::erase(const std::vector<juce::String>& ids, juce::String& error)
{
    const std::set<juce::String> selected(ids.begin(), ids.end());
    auto next = draft;
    std::erase_if(next, [&](const auto& row) { return selected.count(row.materialRegionId) != 0; });
    for (auto& row : next) if (selected.count(row.connectedNextRegionId)) row.connectedNextRegionId.clear();
    return commit(std::move(next), error);
}
bool MaterialAnnotationSession::undo()
{
    if (undoStates.empty()) return false;
    redoStates.push_back(std::move(draft)); draft = std::move(undoStates.back()); undoStates.pop_back(); return true;
}
bool MaterialAnnotationSession::redo()
{
    if (redoStates.empty()) return false;
    undoStates.push_back(std::move(draft)); draft = std::move(redoStates.back()); redoStates.pop_back(); return true;
}
bool MaterialAnnotationSession::save(juce::String& error)
{
    error.clear();
    if (!validate(draft, error)) return false;
    // Destination policy is HERE so overview Save/Ctrl+S cannot bypass it.
    if (scope == EditScope::thisNote)
    {
        if (!hasNoteScope()) return rejected(error, "No project-only note adapter is available");
        auto revision = noteBinding.projectRevision;
        const auto callback = noteBinding.apply;
        if (!callback(noteBinding, draft.front(), revision, error)) return false;
        noteBinding.projectRevision = revision; noteBinding.localRow = draft.front();
        cleanDraft = draft;
        return true;
    }
    if (validatePublication && !validatePublication(draft, error)) return false;
    if (!OttoMaterialIO::saveAudio(manifest, audioFile, draft, error, fingerprint, audioId)) return false;
    if (!OttoMaterialIO::loadAudio(manifest, audioFile, draft, error, &fingerprint, audioId)) return false;
    unpublishedEmptySource = false;
    cleanDraft = draft;
    return true;
}
}
